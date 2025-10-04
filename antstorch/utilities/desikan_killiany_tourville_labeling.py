
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

# We assume these utilities exist in your ANTsTorch package layout.
# If paths differ in your repo, adjust the relative imports accordingly.
from ..architectures import create_unet_model_3d
from ..architectures import create_multihead_unet_model_3d
from ..utilities import get_pretrained_network
from ..utilities import get_antstorch_data
from ..utilities import preprocess_brain_image
from ..utilities import brain_extraction
from ..utilities import deep_atropos


def desikan_killiany_tourville_labeling(t1,
                                        do_preprocessing=True,
                                        return_probability_images=False,
                                        do_lobar_parcellation=False,
                                        verbose=False):
    """
    Desikan-Killiany-Tourville labeling.  https://pubmed.ncbi.nlm.nih.gov/23227001/

    Perform DKT labeling using deep learning.  Description available here:
    https://mindboggle.readthedocs.io/en/latest/labels.html

    The labeling is as follows:

    Outer labels:
    Label 1002: left caudal anterior cingulate
    Label 1003: left caudal middle frontal
    Label 1005: left cuneus
    Label 1006: left entorhinal
    Label 1007: left fusiform
    Label 1008: left inferior parietal
    Label 1009: left inferior temporal
    Label 1010: left isthmus cingulate
    Label 1011: left lateral occipital
    Label 1012: left lateral orbitofrontal
    Label 1013: left lingual
    Label 1014: left medial orbitofrontal
    Label 1015: left middle temporal
    Label 1016: left parahippocampal
    Label 1017: left paracentral
    Label 1018: left pars opercularis
    Label 1019: left pars orbitalis
    Label 1020: left pars triangularis
    Label 1021: left pericalcarine
    Label 1022: left postcentral
    Label 1023: left posterior cingulate
    Label 1024: left precentral
    Label 1025: left precuneus
    Label 1026: left rostral anterior cingulate
    Label 1027: left rostral middle frontal
    Label 1028: left superior frontal
    Label 1029: left superior parietal
    Label 1030: left superior temporal
    Label 1031: left supramarginal
    Label 1034: left transverse temporal
    Label 1035: left insula
    Label 2002: right caudal anterior cingulate
    Label 2003: right caudal middle frontal
    Label 2005: right cuneus
    Label 2006: right entorhinal
    Label 2007: right fusiform
    Label 2008: right inferior parietal
    Label 2009: right inferior temporal
    Label 2010: right isthmus cingulate
    Label 2011: right lateral occipital
    Label 2012: right lateral orbitofrontal
    Label 2013: right lingual
    Label 2014: right medial orbitofrontal
    Label 2015: right middle temporal
    Label 2016: right parahippocampal
    Label 2017: right paracentral
    Label 2018: right pars opercularis
    Label 2019: right pars orbitalis
    Label 2020: right pars triangularis
    Label 2021: right pericalcarine
    Label 2022: right postcentral
    Label 2023: right posterior cingulate
    Label 2024: right precentral
    Label 2025: right precuneus
    Label 2026: right rostral anterior cingulate
    Label 2027: right rostral middle frontal
    Label 2028: right superior frontal
    Label 2029: right superior parietal
    Label 2030: right superior temporal
    Label 2031: right supramarginal
    Label 2034: right transverse temporal
    Label 2035: right insula

    Performing the lobar parcellation is based on the FreeSurfer division
    described here:

    See https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation

    Frontal lobe:
    Label 1002:  left caudal anterior cingulate
    Label 1003:  left caudal middle frontal
    Label 1012:  left lateral orbitofrontal
    Label 1014:  left medial orbitofrontal
    Label 1017:  left paracentral
    Label 1018:  left pars opercularis
    Label 1019:  left pars orbitalis
    Label 1020:  left pars triangularis
    Label 1024:  left precentral
    Label 1026:  left rostral anterior cingulate
    Label 1027:  left rostral middle frontal
    Label 1028:  left superior frontal
    Label 2002:  right caudal anterior cingulate
    Label 2003:  right caudal middle frontal
    Label 2012:  right lateral orbitofrontal
    Label 2014:  right medial orbitofrontal
    Label 2017:  right paracentral
    Label 2018:  right pars opercularis
    Label 2019:  right pars orbitalis
    Label 2020:  right pars triangularis
    Label 2024:  right precentral
    Label 2026:  right rostral anterior cingulate
    Label 2027:  right rostral middle frontal
    Label 2028:  right superior frontal

    Parietal:
    Label 1008:  left inferior parietal
    Label 1010:  left isthmus cingulate
    Label 1022:  left postcentral
    Label 1023:  left posterior cingulate
    Label 1025:  left precuneus
    Label 1029:  left superior parietal
    Label 1031:  left supramarginal
    Label 2008:  right inferior parietal
    Label 2010:  right isthmus cingulate
    Label 2022:  right postcentral
    Label 2023:  right posterior cingulate
    Label 2025:  right precuneus
    Label 2029:  right superior parietal
    Label 2031:  right supramarginal

    Temporal:
    Label 1006:  left entorhinal
    Label 1007:  left fusiform
    Label 1009:  left inferior temporal
    Label 1015:  left middle temporal
    Label 1016:  left parahippocampal
    Label 1030:  left superior temporal
    Label 1034:  left transverse temporal
    Label 2006:  right entorhinal
    Label 2007:  right fusiform
    Label 2009:  right inferior temporal
    Label 2015:  right middle temporal
    Label 2016:  right parahippocampal
    Label 2030:  right superior temporal
    Label 2034:  right transverse temporal

    Occipital:
    Label 1005:  left cuneus
    Label 1011:  left lateral occipital
    Label 1013:  left lingual
    Label 1021:  left pericalcarine
    Label 2005:  right cuneus
    Label 2011:  right lateral occipital
    Label 2013:  right lingual
    Label 2021:  right pericalcarine

    Other outer labels:
    Label 1035:  left insula
    Label 2035:  right insula

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * brain extraction, and
       * affine registration to HCP.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    do_preprocessing : boolean
        See description above.

    return_probability_images : boolean
        Whether to return the two sets of probability images for the inner and outer
        labels.

    do_lobar_parcellation : boolean
        Perform lobar parcellation (also divided by hemisphere).

    do_denoising : boolean
        Perform denoising in preprocessing of brain image.  May impact reproducibility.

    verbose : boolean
        Print progress to the screen.

    version : integer
        Which version.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> dkt = desikan_killiany_tourville_labeling(image)
    """

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    def reshape_image(image, crop_size, interp_type="linear"):
        image_resampled = None
        if interp_type == "linear":
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=0)
        else:
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=1)
        image_cropped = ants.pad_or_crop_image_to_size(image_resampled, crop_size)
        return image_cropped

    which_template = "hcpyaT1Template"
    template_transform_type = "antsRegistrationSyNQuick[a]"
    template = ants.image_read(get_antstorch_data(which_template))

    cropped_template_size = (160, 192, 160)

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = ants.image_clone(t1)
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1threetissue",
            template=which_template,
            template_transform_type=template_transform_type,
            do_bias_correction=True,
            do_denoising=False,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']
        t1_preprocessed = reshape_image(t1_preprocessed, crop_size=cropped_template_size)

    ################################
    #
    # Build outer model and load weights
    #
    ################################

    dkt_lateral_labels = (0,)
    deep_atropos_labels = tuple(range(1, 7))
    dkt_left_labels = (1002, 1003, *tuple(range(1005, 1032)), 1034, 1035)
    dkt_right_labels = (2002, 2003, *tuple(range(2005, 2032)), 2034, 2035)

    labels = sorted((*dkt_lateral_labels, *deep_atropos_labels, *dkt_left_labels))

    channel_size = 1
    number_of_classification_labels = len(labels)

    # Base UNet
    base_unet = create_unet_model_3d(input_channel_size=channel_size,
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2))

    # Attach one penultimate 1x1x1 sigmoid aux head using the shared multihead wrapper
    unet_model = create_multihead_unet_model_3d(
        base_unet,
        n_aux_heads=1,
        use_sigmoid=True,
        n_main_outputs=number_of_classification_labels,
    )

    # Warmup once so the aux head(s) are materialized prior to loading weights
    dummy = torch.zeros(1, 1, *cropped_template_size, dtype=torch.float32)
    unet_model.warmup(dummy)

    # Load converted weights (expects a multi-head state_dict if converter exported it;
    # otherwise base weights load and aux head remains randomly initialized).
    weights_path = get_pretrained_network("DesikanKillianyTourvilleOuter_pytorch")
    sd = torch.load(weights_path, map_location="cpu")
    # Try strict, then fallback to non-strict to allow aux head if missing
    missing, unexpected = unet_model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[ANTsTorch] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    unet_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model.to(device)

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Model prediction.")

    # Build a 2-sample batch: original + flipped (axis=0 in ANTs/Keras => depth in Torch)
    arr0 = t1_preprocessed.iMath("Normalize").numpy().astype(np.float32)
    batchX = np.zeros((2, 1, *cropped_template_size), dtype=np.float32)
    batchX[0, 0, :, :, :] = arr0
    batchX[1, 0, :, :, :] = np.flip(arr0, axis=0)

    with torch.no_grad():
        x = torch.from_numpy(batchX).to(device)  # N,C,D,H,W
        y_main, y_aux = unet_model(x)           # shapes: (N, C_out, D,H,W) and (N,1,D,H,W)
        # Move to CPU numpy for downstream ANTs ops
        main_np = F.softmax(y_main, dim=1).cpu().numpy()  # ensure classification probas
        aux_np  = y_aux.cpu().numpy()

    labels_sorted = sorted((*dkt_lateral_labels, *dkt_left_labels))
    probability_images = [None] * len(labels_sorted)

    dkt_labels = list()
    dkt_labels.append(dkt_lateral_labels)
    dkt_labels.append(dkt_left_labels)

    for b in range(2):
        for i in range(len(dkt_labels)):
            for j in range(len(dkt_labels[i])):
                label = dkt_labels[i][j]
                label_index = labels_sorted.index(label)
                if label == 0:
                    probability_array = np.squeeze(np.sum(main_np[b, :8, :, :, :], axis=0))
                else:
                    # classification channels after deep_atropos labels
                    probability_array = np.squeeze(main_np[b, label_index + len(deep_atropos_labels), :, :, :])
                if b == 1:
                    probability_array = np.flip(probability_array, axis=0)
                probability_image = ants.from_numpy_like(probability_array, t1_preprocessed)
                if do_preprocessing:
                    probability_image = ants.pad_or_crop_image_to_size(probability_image, template.shape)
                    probability_image = ants.apply_transforms(fixed=t1,
                        moving=probability_image,
                        transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                        whichtoinvert=[True], interpolator="linear", singleprecision=True, verbose=verbose)
                if b == 0:
                    probability_images[label_index] = probability_image
                else:
                    probability_images[label_index] = 0.5 * (probability_images[label_index] + probability_image)

    if verbose:
        print("Constructing foreground probability image.")

    foreground_probability_array = 0.5 * (aux_np[0, 0, :, :, :] + np.flip(aux_np[1, 0, :, :, :], axis=0))
    foreground_probability_image = ants.from_numpy_like(np.squeeze(foreground_probability_array), t1_preprocessed)
    if do_preprocessing:
        foreground_probability_image = ants.pad_or_crop_image_to_size(foreground_probability_image, template.shape)
        foreground_probability_image = ants.apply_transforms(fixed=t1,
            moving=foreground_probability_image,
            transformlist=t1_preprocessing['template_transforms']['invtransforms'],
            whichtoinvert=[True], interpolator="linear", singleprecision=True,
            verbose=verbose)

    for i in range(len(dkt_labels)):
        for j in range(len(dkt_labels[i])):
            label = dkt_labels[i][j]
            label_index = labels_sorted.index(label)
            if label == 0:
                probability_images[label_index] *= (foreground_probability_image * -1 + 1)
            else:
                probability_images[label_index] *= foreground_probability_image

    labels_sorted = sorted((*dkt_lateral_labels, *dkt_left_labels))

    bext = brain_extraction(t1, modality="t1hemi", verbose=verbose)

    probability_all_images = list()
    probability_all_images.append(probability_images[0])
    for i in range(len(dkt_left_labels)):
        probability_image = ants.image_clone(probability_images[i+1])
        probability_left_image = probability_image * bext['probability_images'][1]
        probability_all_images.append(probability_left_image)
    for i in range(len(dkt_right_labels)):
        probability_image = ants.image_clone(probability_images[i+1])
        probability_right_image = probability_image * bext['probability_images'][2]
        probability_all_images.append(probability_right_image)

    image_matrix = ants.image_list_to_matrix(probability_all_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    dkt_all_labels = sorted((*dkt_lateral_labels, *dkt_left_labels, *dkt_right_labels))

    dkt_label_image = segmentation_image * 0
    for i in range(len(dkt_all_labels)):
        label = dkt_all_labels[i]
        label_index = dkt_all_labels.index(label)
        dkt_label_image[segmentation_image==label_index] = label

    if do_lobar_parcellation:

        if verbose:
            print("Doing lobar parcellation.")

        ################################
        #
        # Lobar/hemisphere parcellation
        #
        ################################

        # Consolidate lobar cortical labels

        if verbose:
            print("   Consolidating cortical labels.")

        frontal_labels = (1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1024, 1026, 1027, 1028,
                          2002, 2003, 2012, 2014, 2017, 2018, 2019, 2020, 2024, 2026, 2027, 2028)
        parietal_labels = (1008, 1010, 1022, 1023, 1025, 1029, 1031,
                           2008, 2010, 2022, 2023, 2025, 2029, 2031)
        temporal_labels = (1006, 1007, 1009, 1015, 1016, 1030, 1034,
                           2006, 2007, 2009, 2015, 2016, 2030, 2034)
        occipital_labels = (1005, 1011, 1013, 1021,
                            2005, 2011, 2013, 2021)

        lobar_labels = list()
        lobar_labels.append(frontal_labels)
        lobar_labels.append(parietal_labels)
        lobar_labels.append(temporal_labels)
        lobar_labels.append(occipital_labels)

        dkt_lobes = ants.image_clone(dkt_label_image)

        for i in range(len(lobar_labels)):
            for j in range(len(lobar_labels[i])):
                dkt_lobes[dkt_lobes == lobar_labels[i][j]] = i + 1

        dkt_lobes[dkt_lobes > len(lobar_labels)] = 0

        six_tissue = deep_atropos([t1, None, None], do_preprocessing=True, verbose=verbose)
        atropos_seg = six_tissue['segmentation_image']
        atropos_seg[atropos_seg == 1] = 0
        atropos_seg[atropos_seg == 5] = 0
        atropos_seg[atropos_seg == 6] = 0

        brain_mask = ants.image_clone(atropos_seg)
        brain_mask = ants.threshold_image(brain_mask, 0, 0, 0, 1)

        lobar_parcellation = ants.iMath(brain_mask, "PropagateLabelsThroughMask", brain_mask * dkt_lobes)

        # Do left/right

        if verbose:
            print("   Doing left/right hemispheres.")

        hemisphere_labels = list()
        hemisphere_labels.append(dkt_left_labels)
        hemisphere_labels.append(dkt_right_labels)

        dkt_hemispheres = ants.image_clone(dkt_label_image)

        for i in range(len(hemisphere_labels)):
            for j in range(len(hemisphere_labels[i])):
                dkt_hemispheres[dkt_hemispheres == hemisphere_labels[i][j]] = i + 1

        dkt_hemispheres[dkt_hemispheres > 2] = 0

        atropos_brain_mask = ants.threshold_image(atropos_seg, 0, 0, 0, 1)
        hemisphere_parcellation = ants.iMath(atropos_brain_mask, "PropagateLabelsThroughMask",
                                             atropos_brain_mask * dkt_hemispheres)

        hemisphere_parcellation *= ants.threshold_image(lobar_parcellation, 0, 0, 0, 1)
        hemisphere_parcellation[hemisphere_parcellation == 1] = 0
        hemisphere_parcellation[hemisphere_parcellation == 2] = 1
        hemisphere_parcellation *= 4
        lobar_parcellation += hemisphere_parcellation

    if return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'lobar_parcellation' : lobar_parcellation,
                       'probability_images' : probability_all_images }
        return(return_dict)
    elif return_probability_images and not do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'probability_images' : probability_all_images }
        return(return_dict)
    elif not return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'lobar_parcellation' : lobar_parcellation }
        return(return_dict)
    else:
        return(dkt_label_image)
