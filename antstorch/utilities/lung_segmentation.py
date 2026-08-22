import numpy as np
import torch
import ants


def el_bicho(ventilation_image,
             mask,
             use_coarse_slices_only=True,
             device=None,
             verbose=False):

    """
    Perform functional lung segmentation using hyperpolarized gases.

    https://pubmed.ncbi.nlm.nih.gov/30195415/

    Ported from antspynet.utilities.el_bicho.

    Arguments
    ---------
    ventilation_image : ANTsImage
        input ventilation image.

    mask : ANTsImage
        input mask.

    use_coarse_slices_only : boolean
        If True, apply network only in the dimension of greatest slice thickness.
        If False, apply to all dimensions and average the results.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Ventilation segmentation and corresponding probability images

    Example
    -------
    >>> image = ants.image_read("ventilation.nii.gz")
    >>> mask = ants.image_read("mask.nii.gz")
    >>> lung_seg = el_bicho(image, mask, use_coarse_slices_only=True, verbose=False)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if ventilation_image.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if ventilation_image.shape != mask.shape:
        raise ValueError("Ventilation image and mask size are not the same size.")

    template_size = (256, 256)
    classes = (0, 1, 2, 3, 4)
    number_of_classification_labels = len(classes)

    image_modalities = ("Ventilation", "Mask")
    channel_size = len(image_modalities)

    preprocessed_image = (ventilation_image - ventilation_image.mean()) / ventilation_image.std()
    ants.set_direction(preprocessed_image, np.identity(3))

    mask_identity = ants.image_clone(mask)
    ants.set_direction(mask_identity, np.identity(3))

    model = create_unet_model_2d(
        input_channel_size=channel_size,
        number_of_outputs=number_of_classification_labels,
        number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        additional_options=("attentionGating",))

    if verbose:
        print("El Bicho: retrieving model weights.")

    weights_file_name = get_pretrained_network("elBicho_pytorch", target_file_name="elBicho_pytorch.pt")
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(device)

    spacing = ants.get_spacing(preprocessed_image)
    dimensions_to_predict = (spacing.index(max(spacing)),)
    if use_coarse_slices_only == False:
        dimensions_to_predict = list(range(3))

    total_number_of_slices = 0
    for d in range(len(dimensions_to_predict)):
        total_number_of_slices += preprocessed_image.shape[dimensions_to_predict[d]]

    batchX = np.zeros((total_number_of_slices, channel_size, *template_size), dtype=np.float32)

    slice_count = 0
    for d in range(len(dimensions_to_predict)):
        number_of_slices = preprocessed_image.shape[dimensions_to_predict[d]]

        if verbose == True:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            ventilation_slice = ants.pad_or_crop_image_to_size(ants.slice_image(preprocessed_image, dimensions_to_predict[d], i), template_size)
            batchX[slice_count, 0, :, :] = ventilation_slice.numpy().astype(np.float32)

            mask_slice = ants.pad_or_crop_image_to_size(ants.slice_image(mask_identity, dimensions_to_predict[d], i), template_size)
            batchX[slice_count, 1, :, :] = mask_slice.numpy().astype(np.float32)

            slice_count += 1

    if verbose == True:
        print("Prediction.")

    with torch.no_grad():
        x = torch.from_numpy(batchX).float().to(device)
        prediction = model(x).cpu().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    probability_images = list()
    for l in range(number_of_classification_labels):
        probability_images.append(ants.image_clone(mask) * 0)

    current_start_slice = 0
    for d in range(len(dimensions_to_predict)):
        current_end_slice = current_start_slice + preprocessed_image.shape[dimensions_to_predict[d]]
        which_batch_slices = range(current_start_slice, current_end_slice)

        for l in range(number_of_classification_labels):
            prediction_per_dimension = prediction[which_batch_slices, :, :, l]
            prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
            prediction_image = ants.copy_image_info(ventilation_image,
                ants.pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
                ventilation_image.shape))
            probability_images[l] = probability_images[l] + (prediction_image - probability_images[l]) / (d + 1)

        current_start_slice = current_end_slice + 1

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], mask * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], mask * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), mask * 0 + 1)[0]

    return {'segmentation_image': segmentation_image,
           'probability_images': probability_images}


def lung_pulmonary_artery_segmentation(ct,
                                       lung_mask=None,
                                       prediction_batch_size=16,
                                       patch_stride_length=32,
                                       device=None,
                                       verbose=False):

    """
    Perform pulmonary artery segmentation.  Training data taken from the
    PARSE2022 challenge (Luo, Gongning, et al. "Efficient automatic segmentation
    for multi-level pulmonary arteries: The PARSE challenge."
    https://arxiv.org/abs/2304.03708).

    Ported from antspynet.utilities.lung_pulmonary_artery_segmentation.

    Arguments
    ---------
    ct : ANTsImage
        input ct image

    lung_mask : ANTsImage
        input binary lung mask which defines the patch extraction.  If not supplied,
        one is estimated.

    prediction_batch_size : int
        Control memory usage for prediction.  More consequential for GPU-usage.

    patch_stride_length : 3-D tuple or int
        Dictates the stride length for accumulating predicting patches.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Segmentation probability image

    Example
    -------
    >>> ct = ants.image_read("ct.nii.gz")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import lung_extraction
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    patch_size = (160, 160, 160)

    if np.any(np.array(ct.shape) < np.array(patch_size)):
        raise ValueError("Images must be > 160 voxels per dimension.")

    if lung_mask is None:
        lung_ex = lung_extraction(ct, modality="ct", device=device, verbose=verbose)
        lung_mask = ants.threshold_image(lung_ex['segmentation_image'], 1, 3, 1, 0)
    ct_preprocessed = ants.image_clone(ct)
    ct_preprocessed = (ct_preprocessed + 800) / (500 + 800)
    ct_preprocessed[ct_preprocessed > 1.0] = 1.0
    ct_preprocessed[ct_preprocessed < 0.0] = 0.0

    if verbose:
        print("Load model and weights.")

    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3

    number_of_classification_labels = 1
    channel_size = 1

    model = create_unet_model_3d(
        input_channel_size=channel_size,
        number_of_outputs=number_of_classification_labels, mode="sigmoid",
        number_of_filters=(32, 64, 128, 256, 512),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0)

    weights_file_name = get_pretrained_network("pulmonaryArteryWeights_pytorch",
        target_file_name="pulmonaryArteryWeights_pytorch.pt")
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(device)

    if verbose:
        print("Extract patches.")

    ct_patches = ants.extract_image_patches(ct_preprocessed,
                                       patch_size=patch_size,
                                       max_number_of_patches="all",
                                       stride_length=patch_stride_length,
                                       mask_image=lung_mask,
                                       random_seed=None,
                                       return_as_array=True)
    total_number_of_patches = ct_patches.shape[0]

    number_of_batches = total_number_of_patches // prediction_batch_size
    residual_number_of_patches = total_number_of_patches - number_of_batches * prediction_batch_size
    if residual_number_of_patches > 0:
        number_of_batches = number_of_batches + 1

    if verbose:
        print("  Total number of patches: ", str(total_number_of_patches))
        print("  Prediction batch size: ", str(prediction_batch_size))
        print("  Number of batches: ", str(number_of_batches))

    prediction = np.zeros((total_number_of_patches, *patch_size, 1), dtype=np.float32)
    with torch.no_grad():
        for b in range(number_of_batches):
            if b < number_of_batches - 1 or residual_number_of_patches == 0:
                batch_n = prediction_batch_size
            else:
                batch_n = residual_number_of_patches
            indices = range(b * prediction_batch_size, b * prediction_batch_size + batch_n)

            batchX = np.zeros((batch_n, channel_size, *patch_size), dtype=np.float32)
            batchX[:, 0, :, :, :] = ct_patches[indices, :, :, :]

            if verbose:
                print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))
            x = torch.from_numpy(batchX).float().to(device)
            y = model(x).cpu().numpy()
            prediction[indices, :, :, :, 0] = y[:, 0, :, :, :]

    if verbose:
        print("Predict patches and reconstruct.")

    probability_image = ants.reconstruct_image_from_patches(np.squeeze(prediction[:, :, :, :, 0]),
                                                       stride_length=patch_stride_length,
                                                       domain_image=lung_mask,
                                                       domain_image_is_mask=True)
    return probability_image


def lung_airway_segmentation(ct,
                             lung_mask=None,
                             prediction_batch_size=16,
                             patch_stride_length=32,
                             device=None,
                             verbose=False):

    """
    Perform pulmonary airway segmentation from CT images.  Training data taken
    from the EXACT09 challenge.

    Ported from antspynet.utilities.lung_airway_segmentation.

    Arguments
    ---------
    ct : ANTsImage
        input ct image

    lung_mask : ANTsImage
        input binary lung mask which defines the patch extraction (label 1 = left lung,
        label 2 = right lung, label 3 = main airway).  If not supplied, one is estimated.

    prediction_batch_size : int
        Control memory usage for prediction.  More consequential for GPU-usage.

    patch_stride_length : 3-D tuple or int
        Dictates the stride length for accumulating predicting patches.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Segmentation probability image

    Example
    -------
    >>> ct = ants.image_read("ct.nii.gz")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import lung_extraction
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    patch_size = (160, 160, 160)

    if np.any(np.array(ct.shape) < np.array(patch_size)):
        raise ValueError("Images must be > 160 voxels per dimension.")

    if lung_mask is None:
        lung_ex = lung_extraction(ct, modality="ct", device=device, verbose=verbose)
        lung_mask = ants.iMath_MD(lung_ex['segmentation_image'], 2, 3)
        lung_mask = ants.threshold_image(lung_mask, 1, 3, 1, 0)

    ct_preprocessed = ants.image_clone(ct)
    ct_preprocessed = (ct_preprocessed + 800) / (500 + 800)
    ct_preprocessed[ct_preprocessed > 1.0] = 1.0
    ct_preprocessed[ct_preprocessed < 0.0] = 0.0

    if verbose:
        print("Load model and weights.")

    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3

    number_of_classification_labels = 2
    channel_size = 1

    model = create_unet_model_3d(
        input_channel_size=channel_size,
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(32, 64, 128, 256, 512),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0)

    weights_file_name = get_pretrained_network("pulmonaryAirwayWeights_pytorch",
        target_file_name="pulmonaryAirwayWeights_pytorch.pt")
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(device)

    if verbose:
        print("Extract patches.")

    ct_masked = ct_preprocessed * lung_mask
    ct_patches = ants.extract_image_patches(ct_masked,
                                       patch_size=patch_size,
                                       max_number_of_patches="all",
                                       stride_length=patch_stride_length,
                                       mask_image=lung_mask,
                                       random_seed=None,
                                       return_as_array=True)
    total_number_of_patches = ct_patches.shape[0]

    number_of_batches = total_number_of_patches // prediction_batch_size
    residual_number_of_patches = total_number_of_patches - number_of_batches * prediction_batch_size
    if residual_number_of_patches > 0:
        number_of_batches = number_of_batches + 1

    if verbose:
        print("  Total number of patches: ", str(total_number_of_patches))
        print("  Prediction batch size: ", str(prediction_batch_size))
        print("  Number of batches: ", str(number_of_batches))

    prediction = np.zeros((total_number_of_patches, *patch_size, 2), dtype=np.float32)
    with torch.no_grad():
        for b in range(number_of_batches):
            if b < number_of_batches - 1 or residual_number_of_patches == 0:
                batch_n = prediction_batch_size
            else:
                batch_n = residual_number_of_patches
            indices = range(b * prediction_batch_size, b * prediction_batch_size + batch_n)

            batchX = np.zeros((batch_n, channel_size, *patch_size), dtype=np.float32)
            batchX[:, 0, :, :, :] = ct_patches[indices, :, :, :]

            if verbose:
                print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))
            x = torch.from_numpy(batchX).float().to(device)
            y = model(x).cpu().numpy()
            prediction[indices, :, :, :, :] = np.transpose(y, (0, 2, 3, 4, 1))

    if verbose:
        print("Predict patches and reconstruct.")

    probability_image = ants.reconstruct_image_from_patches(np.squeeze(prediction[:, :, :, :, 1]),
                                                       stride_length=patch_stride_length,
                                                       domain_image=lung_mask,
                                                       domain_image_is_mask=True)
    return probability_image
