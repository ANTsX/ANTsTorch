from __future__ import annotations

import numpy as np
import ants
import torch
import torch.nn as nn

def nbm_labeling(t1, device=None, verbose=False):
    """
    Perform CH13 and Nucleus basalis of Meynert (NBM) segmentation in 
    T1 images using Avants labels.
    """
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities import get_antstorch_data
    from ..architectures import create_unet_model_3d
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    which_template = "CIT168_T1w_700um_pad_adni"
    template_transform_type = "antsRegistrationSyNReproQuick[a]"
    template = ants.image_read(get_antstorch_data(which_template))
    template = ants.rank_intensity(template)

    template_seg = ants.image_read(get_antstorch_data("CIT168_basal_forebrain_adni"))
    center_of_mass_labels = (1, 2, 3, 4)
    cropped_template_size = (144, 96, 64)

    ################################
    # Preprocess images
    ################################
    t1 = ants.image_clone(t1, pixeltype="float")
    t1 = ants.iMath(t1, "Normalize")

    t1_preprocessing = preprocess_brain_image(t1,
        truncate_intensity=[1e-4, 0.999],
        brain_extraction_modality="t1threetissue",
        template=which_template,
        template_transform_type=template_transform_type,
        do_bias_correction=True,
        do_denoising=False,
        verbose=verbose)
        
    t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']
    t1_preprocessed = ants.rank_intensity(t1_preprocessed)

    template_seg_masked = ants.mask_image(template_seg, template_seg, center_of_mass_labels, binarize=True)
    center_of_mass = ants.get_center_of_mass(template_seg_masked)

    t1_cropped = ants.crop_image_from_center_point(t1_preprocessed, center_of_mass, cropped_template_size)

    ################################
    # Build model and load weights
    ################################
    
    class NBMNet(nn.Module):
        def __init__(self, unet0, unet1):
            super().__init__()
            self.unet0 = unet0
            self.unet1 = unet1
            
        def forward(self, x):
            out0 = self.unet0(x)
            if isinstance(out0, (list, tuple)): out0 = out0[0]
            # Concaténer le tenseur initial avec la sortie de U-Net 0
            next_in = torch.cat([x, out0], dim=1)
            out1 = self.unet1(next_in)
            if isinstance(out1, (list, tuple)): out1 = out1[0]
            return out1, out0

    nbm_lateral_labels = (0,)
    nbm_lateral_left_labels = (1, 3, 4, 5)
    nbm_lateral_right_labels = (2, 6, 7, 8)

    labels = sorted((*nbm_lateral_labels, *nbm_lateral_left_labels, *nbm_lateral_right_labels))
    number_of_outputs = len(labels)
    channel_size = 1

    if verbose:
        print("    NBM: Creating model and loading weights.")

    unet0 = create_unet_model_3d(input_channel_size=channel_size,
                                 number_of_outputs=1, 
                                 number_of_filters=(32, 64, 128, 256), 
                                 convolution_kernel_size=(3, 3, 3), 
                                 deconvolution_kernel_size=(2, 2, 2),
                                 pool_size=(2, 2, 2), strides=(2, 2, 2), 
                                 dropout_rate=0.0,
                                 mode="sigmoid")
    
    unet1 = create_unet_model_3d(input_channel_size=2,
                                 number_of_outputs=number_of_outputs,
                                 number_of_filters=(32, 64, 96, 128, 256),
                                 convolution_kernel_size=(3, 3, 3),
                                 deconvolution_kernel_size=(2, 2, 2),
                                 pool_size=(2, 2, 2), strides=(2, 2, 2),
                                 dropout_rate=0.0, 
                                 mode="classification")

    unet_model = NBMNet(unet0, unet1).to(device)

    nbm_weights = get_pretrained_network("deep_nbm_rank_pytorch")
    state = torch.load(nbm_weights, map_location="cpu")
    unet_model.load_state_dict(state, strict=False)
    unet_model.eval()
    unet_model = unet_model.to(device)

    ################################
    # Do prediction and normalize
    ################################

    if verbose:
        print("Model prediction using both the original and contralaterally flipped version.")

    # PyTorch Batch: [Batch, Channels, X, Y, Z]
    batchX = np.zeros((2, channel_size, *cropped_template_size), dtype=np.float32)
    batchX[0, 0, ...] = t1_cropped.numpy()
    batchX[1, 0, ...] = np.flip(batchX[0, 0, ...], axis=0)

    with torch.no_grad():
        x = torch.from_numpy(batchX).permute(0, 4, 1, 2, 3).float().to(device)  # [1,C,D,H,W]
        out1, out0 = unet_model(x)
        pred1 = out1.cpu().numpy()
        pred0 = out0.cpu().numpy()

    nbm_labels = sorted((*nbm_lateral_labels, *nbm_lateral_left_labels, *nbm_lateral_right_labels))
    probability_images = [None] * len(labels)
 
    for i in range(len(nbm_labels)):
        label = nbm_labels[i]
        if label == 0:
            probability_array = pred1[0, 0, :, :, :]
            probability_array_flipped = np.flip(pred1[1, 0, :, :, :], axis=0)
        else:
            label_index = nbm_labels.index(label)
            probability_array = pred1[0, label_index, :, :, :]
            if label in nbm_lateral_left_labels:
                left_index = nbm_lateral_left_labels.index(label)
                right_label = nbm_lateral_right_labels[left_index]
                label_flipped_index = nbm_labels.index(right_label)
            else:    
                right_index = nbm_lateral_right_labels.index(label)
                left_label = nbm_lateral_left_labels[right_index]
                label_flipped_index = nbm_labels.index(left_label)
            probability_array_flipped = np.flip(pred1[1, label_flipped_index, :, :, :], axis=0)

        probability_image = ants.from_numpy_like(0.5 * (probability_array + probability_array_flipped), t1_cropped)
        probability_image = ants.apply_transforms(fixed=t1,
             moving=probability_image,
             transformlist=t1_preprocessing['template_transforms']['invtransforms'],
             whichtoinvert=[True], interpolator="linear", singleprecision=True, verbose=verbose)
        probability_images[i] = probability_image
   
    image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return return_dict