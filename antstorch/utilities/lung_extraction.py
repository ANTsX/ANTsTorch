import numpy as np
import torch
import ants


def lung_extraction(image,
                    modality="proton",
                    device=None,
                    verbose=False):

    """
    Perform lung extraction.

    Ported from antspynet.utilities.lung_extraction, using ANTsTorch's
    generic U-Net (2-D/3-D) architectures.  The pretrained weights are
    resolved via get_pretrained_network(<id>_pytorch) -- see
    tools/convert_antspynet_weights_to_antstorch.py for how to produce
    them from the original ANTsPyNet Keras weights.

    Arguments
    ---------
    image : ANTsImage
        input image

    modality : string
        Modality image type.  Options include "ct", "proton", "protonLobes",
        "maskLobes", "ventilation", and "xray".

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary of ANTs segmentation and probability images (or, for
    modality="ventilation", a single probability image -- matching the
    antspynet return convention).

    Example
    -------
    >>> output = lung_extraction(lung_image, modality="proton")
    """

    from ..architectures import create_unet_model_2d
    from ..architectures import create_unet_model_3d
    from ..architectures import create_multihead_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antstorch_data
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if image.dimension != 3 and modality != "xray":
        raise ValueError("Image dimension must be 3.")
    elif image.dimension != 2 and modality == "xray":
        raise ValueError("Image dimension must be 2.")

    def _load_state_dict(weights_file_name):
        sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        return sd

    if modality == "proton":

        weights_file_name = get_pretrained_network("protonLungMri_pytorch",
            target_file_name="protonLungMri_pytorch.pt")

        classes = ("background", "left_lung", "right_lung")
        number_of_classification_labels = len(classes)
        channel_size = 1

        reorient_template = ants.image_read(get_antstorch_data("protonLungTemplate"))
        resampled_image_size = reorient_template.shape

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_classification_labels,
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(7, 7, 5), deconvolution_kernel_size=(7, 7, 5),
            mode="classification")

        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)

        if verbose:
            print("Lung extraction:  normalizing image to the template.")

        center_of_mass_template = ants.get_center_of_mass(reorient_template * 0 + 1)
        center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)
        warped_image = ants.apply_ants_transform_to_image(xfrm, image, reorient_template)

        warped_array = warped_image.numpy().astype(np.float32)
        warped_array = (warped_array - warped_array.mean()) / warped_array.std()

        with torch.no_grad():
            x = torch.from_numpy(warped_array[None, None, ...]).float().to(device)
            y = model(x).squeeze(0).cpu().numpy()

        origin = warped_image.origin
        spacing = warped_image.spacing
        direction = warped_image.direction

        probability_images_array = list()
        for i in range(number_of_classification_labels):
            probability_images_array.append(
                ants.from_numpy(y[i], origin=origin, spacing=spacing, direction=direction))

        if verbose:
            print("Lung extraction:  renormalize probability mask to native space.")

        xfrm_inv = xfrm.invert()
        for i in range(number_of_classification_labels):
            probability_images_array[i] = xfrm_inv.apply_to_image(probability_images_array[i], image)

        image_matrix = ants.image_list_to_matrix(probability_images_array, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        return {'segmentation_image': segmentation_image,
               'probability_images': probability_images_array}

    elif modality == "protonLobes" or modality == "maskLobes":

        reorient_template = ants.image_read(get_antstorch_data("protonLungTemplate"))

        spatial_priors = ants.image_read(get_antstorch_data("protonLobePriors"))
        priors_image_list = ants.ndimage_to_list(spatial_priors)

        channel_size = 1 + len(priors_image_list)
        number_of_classification_labels = 1 + len(priors_image_list)

        base_model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_filters_at_base_layer=16, number_of_layers=4,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0, additional_options=("attentionGating",))

        if modality == "protonLobes":
            model = create_multihead_unet_model_3d(base_unet=base_model, n_aux_heads=1,
                use_sigmoid=True, n_main_outputs=number_of_classification_labels)
            weights_file_name = get_pretrained_network("protonLobes_pytorch",
                target_file_name="protonLobes_pytorch.pt")
        else:
            model = base_model
            weights_file_name = get_pretrained_network("maskLobes_pytorch",
                target_file_name="maskLobes_pytorch.pt")

        sd = _load_state_dict(weights_file_name)

        if verbose:
            print("Lung extraction:  normalizing image to the template.")

        center_of_mass_template = ants.get_center_of_mass(reorient_template * 0 + 1)
        center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)
        warped_image = ants.apply_ants_transform_to_image(xfrm, image, reorient_template)
        warped_array = warped_image.numpy().astype(np.float32)
        if modality == "protonLobes":
            warped_array = (warped_array - warped_array.mean()) / warped_array.std()
        else:
            warped_array = np.where(warped_array != 0, 1.0, 0.0).astype(np.float32)

        batchX = np.zeros((1, channel_size, *warped_array.shape), dtype=np.float32)
        batchX[0, 0, :, :, :] = warped_array
        for i in range(len(priors_image_list)):
            batchX[0, i + 1, :, :, :] = priors_image_list[i].numpy().astype(np.float32)

        model.eval()
        model = model.to(device)

        with torch.no_grad():
            x = torch.from_numpy(batchX).float().to(device)
            if modality == "protonLobes":
                _ = model(x)  # warmup: instantiates the auxiliary head's parameters
                model.load_state_dict(sd, strict=False)
                y_main, y_aux = model(x)
                y_main = y_main.squeeze(0).cpu().numpy()
                whole_lung_mask_array = y_aux.squeeze(0).squeeze(0).cpu().numpy()
            else:
                model.load_state_dict(sd, strict=True)
                y_main = model(x).squeeze(0).cpu().numpy()

        origin = warped_image.origin
        spacing = warped_image.spacing
        direction = warped_image.direction

        probability_images_array = list()
        for i in range(number_of_classification_labels):
            probability_images_array.append(
                ants.from_numpy(y_main[i], origin=origin, spacing=spacing, direction=direction))

        if verbose:
            print("Lung extraction:  renormalize probability images to native space.")

        xfrm_inv = xfrm.invert()
        for i in range(number_of_classification_labels):
            probability_images_array[i] = xfrm_inv.apply_to_image(probability_images_array[i], image)

        image_matrix = ants.image_list_to_matrix(probability_images_array, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        if modality == "protonLobes":
            whole_lung_mask = ants.from_numpy(whole_lung_mask_array, origin=origin, spacing=spacing, direction=direction)
            whole_lung_mask = xfrm_inv.apply_to_image(whole_lung_mask, image)
            return {'segmentation_image': segmentation_image,
                   'probability_images': probability_images_array,
                   'whole_lung_mask_image': whole_lung_mask}
        else:
            return {'segmentation_image': segmentation_image,
                   'probability_images': probability_images_array}

    elif modality == "ct":

        if verbose:
            print("Preprocess CT image.")

        def closest_simplified_direction_matrix(direction):
            closest = np.floor(np.abs(direction) + 0.5)
            closest[direction < 0] *= -1.0
            return closest

        simplified_direction = closest_simplified_direction_matrix(image.direction)

        reference_image_size = (128, 128, 128)

        ct_preprocessed = ants.resample_image(image, reference_image_size, use_voxels=True, interp_type=0)
        ct_preprocessed[ct_preprocessed < -1000] = -1000
        ct_preprocessed[ct_preprocessed > 400] = 400
        ct_preprocessed.set_direction(simplified_direction)
        ct_preprocessed.set_origin((0, 0, 0))
        ct_preprocessed.set_spacing((1, 1, 1))

        reference_image = ants.make_image(reference_image_size,
                                          voxval=0,
                                          spacing=(1, 1, 1),
                                          origin=(0, 0, 0),
                                          direction=np.identity(3))
        center_of_mass_reference = np.floor(ants.get_center_of_mass(reference_image * 0 + 1))
        center_of_mass_image = np.floor(ants.get_center_of_mass(ct_preprocessed * 0 + 1))
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_reference), translation=translation)
        ct_preprocessed = ((ct_preprocessed - ct_preprocessed.min()) /
            (ct_preprocessed.max() - ct_preprocessed.min()))
        ct_preprocessed_warped = ants.apply_ants_transform_to_image(
            xfrm, ct_preprocessed, reference_image, interpolation="nearestneighbor")
        ct_preprocessed_warped = ((ct_preprocessed_warped - ct_preprocessed_warped.min()) /
            (ct_preprocessed_warped.max() - ct_preprocessed_warped.min())) - 0.5

        if verbose:
            print("Build model and load weights.")

        weights_file_name = get_pretrained_network("lungCtWithPriorsSegmentationWeights_pytorch",
            target_file_name="lungCtWithPriorsSegmentationWeights_pytorch.pt")

        classes = ("background", "left lung", "right lung", "airways")
        number_of_classification_labels = len(classes)

        luna16_priors = ants.ndimage_to_list(ants.image_read(get_antstorch_data("luna16LungPriors")))
        for i in range(len(luna16_priors)):
            luna16_priors[i] = ants.resample_image(luna16_priors[i], reference_image_size, use_voxels=True)
        channel_size = len(luna16_priors) + 1

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            additional_options=("attentionGating",))
        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)

        if verbose:
            print("Prediction.")

        batchX = np.zeros((1, channel_size, *reference_image_size), dtype=np.float32)
        batchX[0, 0, :, :, :] = ct_preprocessed_warped.numpy().astype(np.float32)
        for i in range(len(luna16_priors)):
            batchX[0, i + 1, :, :, :] = luna16_priors[i].numpy().astype(np.float32) - 0.5

        with torch.no_grad():
            x = torch.from_numpy(batchX).float().to(device)
            y = model(x).squeeze(0).cpu().numpy()

        xfrm_inv = xfrm.invert()
        probability_images = list()
        for i in range(number_of_classification_labels):
            if verbose:
                print("Reconstructing image", classes[i])
            probability_image = ants.from_numpy(y[i],
                origin=ct_preprocessed_warped.origin, spacing=ct_preprocessed_warped.spacing,
                direction=ct_preprocessed_warped.direction)
            probability_image = xfrm_inv.apply_to_image(probability_image, ct_preprocessed)
            probability_image = ants.resample_image(probability_image,
               resample_params=image.shape, use_voxels=True, interp_type=0)
            probability_image = ants.copy_image_info(image, probability_image)
            probability_images.append(probability_image)

        image_matrix = ants.image_list_to_matrix(probability_images, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        return {'segmentation_image': segmentation_image,
               'probability_images': probability_images}

    elif modality == "ventilation":

        if verbose:
            print("Preprocess ventilation image.")

        template_size = (256, 256)
        channel_size = 1

        preprocessed_image = (image - image.mean()) / image.std()
        ants.set_direction(preprocessed_image, np.identity(3))

        model = create_unet_model_2d(
            input_channel_size=channel_size,
            number_of_outputs=1, mode='sigmoid',
            number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2))

        if verbose:
            print("Whole lung mask: retrieving model weights.")

        weights_file_name = get_pretrained_network("wholeLungMaskFromVentilation_pytorch",
            target_file_name="wholeLungMaskFromVentilation_pytorch.pt")
        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)

        spacing = ants.get_spacing(preprocessed_image)
        dimensions_to_predict = (spacing.index(max(spacing)),)

        total_number_of_slices = 0
        for d in range(len(dimensions_to_predict)):
            total_number_of_slices += preprocessed_image.shape[dimensions_to_predict[d]]

        batchX = np.zeros((total_number_of_slices, channel_size, *template_size), dtype=np.float32)

        slice_count = 0
        for d in range(len(dimensions_to_predict)):
            number_of_slices = preprocessed_image.shape[dimensions_to_predict[d]]

            if verbose:
                print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

            for i in range(number_of_slices):
                ventilation_slice = ants.pad_or_crop_image_to_size(ants.slice_image(preprocessed_image, dimensions_to_predict[d], i), template_size)
                batchX[slice_count, 0, :, :] = ventilation_slice.numpy().astype(np.float32)
                slice_count += 1

        if verbose:
            print("Prediction.")

        with torch.no_grad():
            x = torch.from_numpy(batchX).float().to(device)
            prediction = model(x).cpu().numpy()
        prediction = np.transpose(prediction, (0, 2, 3, 1))

        permutations = list()
        permutations.append((0, 1, 2))
        permutations.append((1, 0, 2))
        permutations.append((1, 2, 0))

        probability_image = ants.image_clone(image) * 0

        current_start_slice = 0
        for d in range(len(dimensions_to_predict)):
            current_end_slice = current_start_slice + preprocessed_image.shape[dimensions_to_predict[d]]
            which_batch_slices = range(current_start_slice, current_end_slice)

            prediction_per_dimension = prediction[which_batch_slices, :, :, 0]
            prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
            prediction_image = ants.copy_image_info(image,
                ants.pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
                image.shape))
            probability_image = probability_image + (prediction_image - probability_image) / (d + 1)

            current_start_slice = current_end_slice + 1

        return probability_image

    elif modality == "xray":

        weights_file_name = get_pretrained_network("xrayLungExtraction_pytorch",
            target_file_name="xrayLungExtraction_pytorch.pt")

        classes = ("background", "left_lung", "right_lung")
        number_of_classification_labels = len(classes)
        resampled_image_size = (256, 256)
        channel_size = 3

        resampled_image = ants.resample_image(image, resampled_image_size, use_voxels=True, interp_type=0)
        xray_lung_priors = ants.ndimage_to_list(ants.image_read(get_antstorch_data("xrayLungPriors")))

        model = create_unet_model_2d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_filters_at_base_layer=32, number_of_layers=4,
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0)
        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)

        batchX = np.zeros((1, channel_size, *resampled_image_size), dtype=np.float32)
        resampled_array = resampled_image.numpy().astype(np.float32)
        batchX[0, 0, :, :] = (resampled_array - resampled_array.min()) / (resampled_array.max() - resampled_array.min())
        batchX[0, 1, :, :] = xray_lung_priors[0].numpy().astype(np.float32)
        batchX[0, 2, :, :] = xray_lung_priors[1].numpy().astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(batchX).float().to(device)
            y = model(x).squeeze(0).cpu().numpy()

        origin = resampled_image.origin
        spacing = resampled_image.spacing
        direction = resampled_image.direction

        probability_images_array = list()
        for i in range(number_of_classification_labels):
            probability_images_array.append(
                ants.from_numpy(y[i], origin=origin, spacing=spacing, direction=direction))

        if verbose:
            print("Lung extraction:  renormalize probability mask to native space.")

        for i in range(number_of_classification_labels):
            probability_images_array[i] = ants.resample_image(probability_images_array[i],
                image.shape, use_voxels=True, interp_type=0)
            probability_images_array[i] = ants.copy_image_info(image, probability_images_array[i])

        image_matrix = ants.image_list_to_matrix(probability_images_array, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        return {'segmentation_image': segmentation_image,
               'probability_images': probability_images_array}

    else:
        raise ValueError("Unrecognized modality.")
