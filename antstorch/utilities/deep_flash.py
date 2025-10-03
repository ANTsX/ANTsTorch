import numpy as np
import ants
import torch

def deep_flash(t1,
               t2=None,
               do_preprocessing=True,
               use_rank_intensity=True,
               verbose=False
               ):

    """
    Hippocampal/Enthorhinal segmentation using "Deep Flash"

    Perform hippocampal/entorhinal segmentation in T1 and T1/T2 images using
    labels from Mike Yassa's lab---https://faculty.sites.uci.edu/myassa/

    https://www.nature.com/articles/s41598-024-59440-6

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA2/CA3/CA4
    Label 14:  right DG/CA2/CA3/CA4
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * affine registration to the "deep flash" template.
    which is performed on the input images if do_preprocessing = True.

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    t2 : ANTsImage
        Optional 3-D T2-weighted brain image for yassa parcellation.  If
        specified, it is assumed to be pre-aligned to the t1.

    which_parcellation : string --- "yassa"
        See above label descriptions.

    do_preprocessing : boolean
        See description above.

    use_rank_intensity : boolean
        If false, use histogram matching with cropped template ROI.  Otherwise,
        use a rank intensity transform on the cropped ROI.  Only for "yassa"
        parcellation.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label and foreground.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_flash(image)
    """

    # --------------------------------
    # Early checks and options
    # --------------------------------
    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    ################################
    #
    # Options temporarily taken from the user
    #
    ################################

    # use_hierarchical_parcellation : boolean
    #     If True, the u-net model exposes additional outputs of the medial temporal lobe
    #     region, hippocampal, and entorhinal/perirhinal/parahippocampal regions.  Otherwise
    #     the only additional output is the medial temporal lobe.
    #
    # use_contralaterality : boolean
    #     Use both hemispherical models to also predict the corresponding contralateral
    #     segmentation and use both sets of priors to produce the results.
    #
    use_hierarchical_parcellation = True
    use_contralaterality = True

    ################################
    #
    # Preprocess input images
    #
    ################################

    t1_preprocessed = t1
    t1_mask = None
    t1_preprocessed_flipped = None

    # Load template(s)
    from ..utilities import brain_extraction
    from ..utilities import get_antstorch_data
    from ..utilities import get_pretrained_network
    from ..architectures import create_unet_model_3d
    from ..architectures import create_multihead_unet_model_3d

    def _batch_from_crops(
        t1_cropped, priors_list, image_size, use_contralaterality,
        t1_cropped_flipped=None, t2_cropped=None, t2_cropped_flipped=None
    ) -> np.ndarray:
        npri = len(priors_list)
        t2_flag = 1 if t2_cropped is not None else 0
        channel_size = 1 + t2_flag + npri
        N = 2 if use_contralaterality else 1
        batchX = np.zeros((N, *image_size, channel_size), dtype=np.float32)

        # channel 0: T1
        batchX[0, :, :, :, 0] = t1_cropped.numpy()
        if use_contralaterality and t1_cropped_flipped is not None:
            batchX[1, :, :, :, 0] = t1_cropped_flipped.numpy()

        # optional T2
        if t2_flag:
            batchX[0, :, :, :, 1] = t2_cropped.numpy()
            if use_contralaterality and t2_cropped_flipped is not None:
                batchX[1, :, :, :, 1] = t2_cropped_flipped.numpy()

        # priors stacked at end
        pri_start = channel_size - npri
        for i, p in enumerate(priors_list):
            arr = p.numpy()
            for j in range(N):
                batchX[j, :, :, :, pri_start + i] = arr

        return batchX

    def _predict_torch(model, batchX: np.ndarray, device: str = "cpu"):
        # NHWDC -> NCDHW
        x = torch.from_numpy(np.transpose(batchX, (0, 4, 1, 2, 3))).to(device)
        out = model(x)
        if isinstance(out, (list, tuple)):
            main = out[0].detach().cpu().numpy()
            auxs = [y.detach().cpu().numpy() for y in out[1:]]
            return (main, *auxs)
        else:
            return (out.detach().cpu().numpy(),)

    t1_template = ants.image_read(get_antstorch_data("deepFlashTemplateT1SkullStripped"))

    template_transforms = None
    if do_preprocessing:
        if verbose:
            print("Preprocessing T1.")

        # Brain extraction
        probability_mask = brain_extraction(t1_preprocessed, modality="t1", verbose=verbose)
        t1_mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        t1_preprocessed = t1_preprocessed * t1_mask

        # Bias correction
        t1_preprocessed = ants.n4_bias_field_correction(t1_preprocessed, t1_mask, shrink_factor=4, verbose=verbose)

        # Warp to template
        registration = ants.registration(fixed=t1_template, moving=t1_preprocessed,
                                         type_of_transform="antsRegistrationSyNQuickRepro[a]", verbose=verbose)
        template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                                   invtransforms=registration['invtransforms'])
        t1_preprocessed = registration['warpedmovout']

    if use_contralaterality:
        t1_preprocessed_array = t1_preprocessed.numpy()
        t1_preprocessed_array_flipped = np.flip(t1_preprocessed_array, axis=0)
        t1_preprocessed_flipped = ants.from_numpy(t1_preprocessed_array_flipped,
                                                  origin=t1_preprocessed.origin,
                                                  spacing=t1_preprocessed.spacing,
                                                  direction=t1_preprocessed.direction)

    t2_preprocessed = t2
    t2_preprocessed_flipped = None
    t2_template = None
    if t2 is not None:
        t2_template = ants.image_read(get_antstorch_data("deepFlashTemplateT2SkullStripped"))
        t2_template = ants.copy_image_info(t1_template, t2_template)

        if do_preprocessing:
            if verbose:
                print("Preprocessing T2.")

            # Use t1 mask
            t2_preprocessed = t2_preprocessed * t1_mask

            # Do bias correction
            t2_preprocessed = ants.n4_bias_field_correction(t2_preprocessed, t1_mask, shrink_factor=4, verbose=verbose)

            # Warp to template
            t2_preprocessed = ants.apply_transforms(fixed=t1_template,
                                                    moving=t2_preprocessed,
                                                    transformlist=template_transforms['fwdtransforms'],
                                                    verbose=verbose)

        if use_contralaterality:
            t2_preprocessed_array = t2_preprocessed.numpy()
            t2_preprocessed_array_flipped = np.flip(t2_preprocessed_array, axis=0)
            t2_preprocessed_flipped = ants.from_numpy(t2_preprocessed_array_flipped,
                                                      origin=t2_preprocessed.origin,
                                                      spacing=t2_preprocessed.spacing,
                                                      direction=t2_preprocessed.direction)

    ################################
    #
    # Initialize outputs and constants
    #
    ################################

    probability_images = list()
    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    image_size = (64, 64, 96)

    ################################
    #
    # Download spatial priors
    #
    ################################

    spatial_priors_file_name_path = get_antstorch_data("deepFlashPriors")
    spatial_priors = ants.image_read(spatial_priors_file_name_path)
    priors_image_list = ants.ndimage_to_list(spatial_priors)
    for i in range(len(priors_image_list)):
        priors_image_list[i] = ants.copy_image_info(t1_preprocessed, priors_image_list[i])

    labels_left = labels[1::2]
    priors_image_left_list = priors_image_list[1::2]
    
    probability_images_left = list()
    foreground_probability_images_left = list()
    lower_bound_left = (76, 74, 56)
    upper_bound_left = (140, 138, 152)
    priors_image_left_cropped_list  = [ants.crop_indices(p,  lower_bound_left,  upper_bound_left)  for p in priors_image_left_list]

    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    origin_left = tmp_cropped.origin

    spacing = tmp_cropped.spacing
    direction = tmp_cropped.direction

    t1_template_roi_left = ants.crop_indices(t1_template, lower_bound_left, upper_bound_left)
    t1_template_roi_left = (t1_template_roi_left - t1_template_roi_left.min()) / (t1_template_roi_left.max() - t1_template_roi_left.min()) * 2.0 - 1.0
    t2_template_roi_left = None
    if t2_template is not None:
        t2_template_roi_left = ants.crop_indices(t2_template, lower_bound_left, upper_bound_left)
        t2_template_roi_left = (t2_template_roi_left - t2_template_roi_left.min()) / (t2_template_roi_left.max() - t2_template_roi_left.min()) * 2.0 - 1.0

    labels_right = labels[2::2]
    priors_image_right_list = priors_image_list[2::2]

    probability_images_right = list()
    foreground_probability_images_right = list()
    lower_bound_right = (20, 74, 56)
    upper_bound_right = (84, 138, 152)
    priors_image_right_cropped_list = [ants.crop_indices(p,  lower_bound_right, upper_bound_right) for p in priors_image_right_list]

    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    origin_right = tmp_cropped.origin

    t1_template_roi_right = ants.crop_indices(t1_template, lower_bound_right, upper_bound_right)
    t1_template_roi_right = (t1_template_roi_right - t1_template_roi_right.min()) / (t1_template_roi_right.max() - t1_template_roi_right.min()) * 2.0 - 1.0
    t2_template_roi_right = None
    if t2_template is not None:
        t2_template_roi_right = ants.crop_indices(t2_template, lower_bound_right, upper_bound_right)
        t2_template_roi_right = (t2_template_roi_right - t2_template_roi_right.min()) / (t2_template_roi_right.max() - t2_template_roi_right.min()) * 2.0 - 1.0

    ################################
    #
    # Create model
    #
    ################################

    channel_size = 1 + len(labels_left)
    if t2 is not None:
        channel_size += 1

    number_of_classification_labels = 1 + len(labels_left)

    # 2) Create base u-net
    base_unet_model = create_unet_model_3d(
        input_channel_size=channel_size,
        number_of_outputs=number_of_classification_labels,
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        pad_crop="keras",
    )

    # 2) wrap for aux heads (penultimate features -> N times 1x1x1 conv)
    n_aux_heads = 3 if use_hierarchical_parcellation else 1

    unet_model = create_multihead_unet_model_3d(
        base_unet=base_unet_model,
        n_aux_heads=n_aux_heads,
        use_sigmoid=True,
        n_main_outputs=number_of_classification_labels,
    )

    # 3) warmup once so the wrapper materializes the heads
    with torch.no_grad():
        dummy = torch.zeros(1, channel_size, *image_size)
        _ = unet_model(dummy)

    ################################
    #
    # LEFT hemisphere
    #
    ################################

    # Determine network name
    network_name = 'deepFlashLeftT1'
    if t2 is not None:
        network_name = 'deepFlashLeftBoth'

    if use_hierarchical_parcellation:
        network_name += "Hierarchical"

    if use_rank_intensity:
        network_name += "_ri"

    if verbose:
        print("DeepFlash: retrieving model weights (left).")
    weights_file_name = get_pretrained_network(network_name + "_pytorch")
    state = torch.load(weights_file_name, map_location="cpu")
    missing, unexpected = unet_model.load_state_dict(state, strict=False)
    if verbose:
        print(f"[antstorch] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    unet_model.eval()

    if verbose:
        print("Prediction (left).")

    # Assemble batch for LEFT
    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    if use_rank_intensity:
        t1_cropped_in = ants.rank_intensity(t1_cropped)
    else:
        t1_cropped_in = ants.histogram_match_image(t1_cropped, t1_template_roi_left, 255, 64, False)

    t1_cropped_flipped = None
    if use_contralaterality:
        t1_cropped_flipped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_left, upper_bound_left)
        t1_cropped_flipped = ants.rank_intensity(t1_cropped_flipped) if use_rank_intensity else ants.histogram_match_image(t1_cropped_flipped, t1_template_roi_left, 255, 64, False)

    t2_cropped_in = None
    t2_cropped_flipped = None
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_left, upper_bound_left)
        t2_cropped_in = ants.rank_intensity(t2_cropped) if use_rank_intensity else ants.histogram_match_image(t2_cropped, t2_template_roi_left, 255, 64, False)
        if use_contralaterality:
            t2_cropped_flipped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_left, upper_bound_left)
            t2_cropped_flipped = ants.rank_intensity(t2_cropped_flipped) if use_rank_intensity else ants.histogram_match_image(t2_cropped_flipped, t2_template_roi_left, 255, 64, False)

    batchX = _batch_from_crops(
        t1_cropped_in,
        priors_image_left_cropped_list,
        image_size,
        use_contralaterality,
        t1_cropped_flipped=t1_cropped_flipped,
        t2_cropped=t2_cropped_in,
        t2_cropped_flipped=t2_cropped_flipped,
    )

    pred = _predict_torch(unet_model, batchX, device="cpu")  # (main, aux1, aux2, aux3)

    # Convert predictions to images and decrop/transform back
    # Main head (8 channels)
    main_probs = pred[0]  # [N,C,D,H,W]
    N = main_probs.shape[0]
    for i in range(1 + len(labels_left)):
        for j in range(N):
            vol = main_probs[j, i, ...]
            probability_image = ants.from_numpy(vol, origin=origin_left, spacing=spacing, direction=direction)
            if i > 0:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if j == 1:  # flipped
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                                                    origin=probability_image.origin,
                                                    spacing=probability_image.spacing,
                                                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                                                          moving=probability_image,
                                                          transformlist=template_transforms['invtransforms'],
                                                          whichtoinvert=[True], interpolator="linear", verbose=verbose)
            if j == 0:
                probability_images_left.append(probability_image)
            else:
                probability_images_right.append(probability_image)

    # Aux heads (3)
    for aidx in range(1, len(pred)):
        aux = pred[aidx]  # [N,1,D,H,W]
        for j in range(aux.shape[0]):
            vol = aux[j, 0, ...]
            probability_image = ants.from_numpy(vol, origin=origin_left, spacing=spacing, direction=direction)
            probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            if j == 1:
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                                                    origin=probability_image.origin,
                                                    spacing=probability_image.spacing,
                                                    direction=probability_image.direction)
            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                                                          moving=probability_image,
                                                          transformlist=template_transforms['invtransforms'],
                                                          whichtoinvert=[True], interpolator="linear", verbose=verbose)
            if j == 0:
                foreground_probability_images_left.append(probability_image)
            else:
                foreground_probability_images_right.append(probability_image)

    ################################
    #
    # RIGHT hemisphere
    #
    ################################

    # Determine network name
    network_name = 'deepFlashRightT1'
    if t2 is not None:
        network_name = 'deepFlashRightBoth'

    if use_hierarchical_parcellation:
        network_name += "Hierarchical"

    if use_rank_intensity:
        network_name += "_ri"

    if verbose:
        print("DeepFlash: retrieving model weights (right).")
    weights_file_name = get_pretrained_network(network_name + "_pytorch")
    state = torch.load(weights_file_name, map_location="cpu")
    missing, unexpected = unet_model.load_state_dict(state, strict=False)
    if verbose:
        print(f"[antstorch] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    unet_model.eval()

    if verbose:
        print("Prediction (right).")

    # Assemble batch for RIGHT
    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    if use_rank_intensity:
        t1_cropped_in = ants.rank_intensity(t1_cropped)
    else:
        t1_cropped_in = ants.histogram_match_image(t1_cropped, t1_template_roi_right, 255, 64, False)
    t1_cropped_flipped = None
    if use_contralaterality:
        t1_cropped_flipped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_right, upper_bound_right)
        t1_cropped_flipped = ants.rank_intensity(t1_cropped_flipped) if use_rank_intensity else ants.histogram_match_image(t1_cropped_flipped, t1_template_roi_right, 255, 64, False)
    t2_cropped_in = None
    t2_cropped_flipped = None
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_right, upper_bound_right)
        t2_cropped_in = ants.rank_intensity(t2_cropped) if use_rank_intensity else ants.histogram_match_image(t2_cropped, t2_template_roi_right, 255, 64, False)
        if use_contralaterality:
            t2_cropped_flipped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_right, upper_bound_right)
            t2_cropped_flipped = ants.rank_intensity(t2_cropped_flipped) if use_rank_intensity else ants.histogram_match_image(t2_cropped_flipped, t2_template_roi_right, 255, 64, False)

    batchX = _batch_from_crops(
        t1_cropped_in,
        priors_image_right_cropped_list,
        image_size,
        use_contralaterality,
        t1_cropped_flipped=t1_cropped_flipped,
        t2_cropped=t2_cropped_in,
        t2_cropped_flipped=t2_cropped_flipped,
    )

    pred = _predict_torch(unet_model, batchX, device="cpu")  # (main, aux1, aux2, aux3)

    # Main head (8 channels)
    main_probs = pred[0]  # [N,C,D,H,W]
    N = main_probs.shape[0]
    for i in range(1 + len(labels_right)):
        for j in range(N):
            vol = main_probs[j, i, ...]
            probability_image = ants.from_numpy(vol, origin=origin_right, spacing=spacing, direction=direction)
            if i > 0:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if j == 1:
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                                                    origin=probability_image.origin,
                                                    spacing=probability_image.spacing,
                                                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                                                          moving=probability_image,
                                                          transformlist=template_transforms['invtransforms'],
                                                          whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                if use_contralaterality:
                    # average with earlier flipped result from the LEFT pass
                    probability_images_right[i] = (probability_images_right[i] + probability_image) / 2
                else:
                    probability_images_right.append(probability_image)
            else:       # flipped
                probability_images_left[i] = (probability_images_left[i] + probability_image) / 2

    # Aux heads
    for aidx in range(1, len(pred)):
        aux = pred[aidx]  # [N,1,D,H,W]
        for j in range(aux.shape[0]):
            vol = aux[j, 0, ...]
            probability_image = ants.from_numpy(vol, origin=origin_right, spacing=spacing, direction=direction)
            probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)

            if j == 1:
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                                                    origin=probability_image.origin,
                                                    spacing=probability_image.spacing,
                                                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                                                          moving=probability_image,
                                                          transformlist=template_transforms['invtransforms'],
                                                          whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                if use_contralaterality:
                    foreground_probability_images_right[aidx-1] = (foreground_probability_images_right[aidx-1] + probability_image) / 2
                else:
                    foreground_probability_images_right.append(probability_image)
            else:
                foreground_probability_images_left[aidx-1] = (foreground_probability_images_left[aidx-1] + probability_image) / 2

    ################################
    #
    # Combine priors
    #
    ################################

    probability_background_image = ants.image_clone(t1) * 0
    for i in range(1, len(probability_images_left)):
        probability_background_image += probability_images_left[i]
    for i in range(1, len(probability_images_right)):
        probability_background_image += probability_images_right[i]

    probability_images.append(probability_background_image * -1 + 1)
    for i in range(1, len(probability_images_left)):
        probability_images.append(probability_images_left[i])
        probability_images.append(probability_images_right[i])

    ################################
    #
    # Convert to segmentation
    #
    ################################

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                             np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i, lab in enumerate(labels):
        relabeled_image[segmentation_image == i] = lab

    ################################
    #
    # Prepare return dict (hierarchical outputs)
    #
    ################################

    foreground_probability_images = list()
    for i in range(len(foreground_probability_images_left)):
        foreground_probability_images.append(foreground_probability_images_left[i] + foreground_probability_images_right[i])

    return_dict = {'segmentation_image' : relabeled_image,
                   'probability_images' : probability_images,
                   'medial_temporal_lobe_probability_image' : foreground_probability_images[0],
                   'other_region_probability_image' : foreground_probability_images[1],
                   'hippocampal_probability_image' : foreground_probability_images[2]
                    }
    return return_dict
