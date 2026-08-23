import numpy as np
import torch
import ants


def _load_state_dict(weights_file_name):
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def claustrum_segmentation(t1,
                           do_preprocessing=True,
                           use_ensemble=True,
                           device=None,
                           verbose=False):

    """
    Claustrum segmentation

    Described here:

        https://pubmed.ncbi.nlm.nih.gov/34520080/

    with the implementation available at:

        https://github.com/hongweilibran/claustrum_multi_view

    PyTorch port of antspynet.utilities.claustrum_segmentation.  Reuses the
    already-ported create_sysu_media_unet_model_2d(anatomy="claustrum")
    architecture -- no new architecture was needed for this port.

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image.

    do_preprocessing : boolean
        perform n4 bias correction (+ denoising, brain extraction).

    use_ensemble : boolean
        whether to use all 3 sets of weights (per view) or just the first.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Claustrum segmentation probability image.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> probability_mask = claustrum_segmentation(image)
    """

    from ..architectures import create_sysu_media_unet_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    image_size = (180, 180)

    ################################
    #
    # Preprocess images
    #
    ################################

    number_of_channels = 1
    t1_preprocessed = ants.image_clone(t1)
    brain_mask = ants.threshold_image(t1, 0, 0, 0, 1)
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=True,
            device=device,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"]
        brain_mask = t1_preprocessing["brain_mask"]

    reference_image = ants.make_image((170, 256, 256),
                                       voxval=1,
                                       spacing=(1, 1, 1),
                                       origin=(0, 0, 0),
                                       direction=np.identity(3))
    center_of_mass_reference = ants.get_center_of_mass(reference_image)
    center_of_mass_image = ants.get_center_of_mass(brain_mask)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    t1_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, t1_preprocessed, reference_image)
    brain_mask_warped = ants.threshold_image(
        ants.apply_ants_transform_to_image(xfrm, brain_mask, reference_image), 0.5, 1.1, 1, 0)

    ################################
    #
    # Gaussian normalize intensity based on brain mask
    #
    ################################

    mean_t1 = t1_preprocessed_warped[brain_mask_warped > 0].mean()
    std_t1 = t1_preprocessed_warped[brain_mask_warped > 0].std()
    t1_preprocessed_warped = (t1_preprocessed_warped - mean_t1) / std_t1

    t1_preprocessed_warped = t1_preprocessed_warped * brain_mask_warped

    ################################
    #
    # Build models and load weights
    #
    ################################

    number_of_models = 3 if use_ensemble else 1

    if verbose:
        print("Claustrum:  retrieving axial model weights.")

    unet_axial_models = list()
    for i in range(number_of_models):
        weights_file_name = get_pretrained_network(f"claustrum_axial_{i}_pytorch",
            target_file_name=f"claustrum_axial_{i}_pytorch.pt")
        model = create_sysu_media_unet_model_2d(number_of_channels, anatomy="claustrum")
        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)
        unet_axial_models.append(model)

    if verbose:
        print("Claustrum:  retrieving coronal model weights.")

    unet_coronal_models = list()
    for i in range(number_of_models):
        weights_file_name = get_pretrained_network(f"claustrum_coronal_{i}_pytorch",
            target_file_name=f"claustrum_coronal_{i}_pytorch.pt")
        model = create_sysu_media_unet_model_2d(number_of_channels, anatomy="claustrum")
        model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        model.eval()
        model = model.to(device)
        unet_coronal_models.append(model)

    ################################
    #
    # Extract slices
    #
    ################################

    dimensions_to_predict = [1, 2]

    batch_coronal_X = np.zeros((t1_preprocessed_warped.shape[1], number_of_channels, *image_size), dtype=np.float32)
    batch_axial_X = np.zeros((t1_preprocessed_warped.shape[2], number_of_channels, *image_size), dtype=np.float32)

    for d in range(len(dimensions_to_predict)):
        number_of_slices = t1_preprocessed_warped.shape[dimensions_to_predict[d]]

        if verbose:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            t1_slice = ants.pad_or_crop_image_to_size(ants.slice_image(t1_preprocessed_warped, dimensions_to_predict[d], i), image_size)
            if dimensions_to_predict[d] == 1:
                batch_coronal_X[i, 0, :, :] = np.rot90(t1_slice.numpy(), k=-1)
            else:
                batch_axial_X[i, 0, :, :] = np.rot90(t1_slice.numpy())

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    def predict_batch(models, batch_X):
        with torch.no_grad():
            x = torch.from_numpy(batch_X).float().to(device)
            prediction = models[0](x).cpu().numpy()
            for i in range(1, len(models)):
                prediction = prediction + models[i](x).cpu().numpy()
        prediction = prediction / len(models)
        return prediction

    if verbose:
        print("Coronal prediction.")
    prediction_coronal = predict_batch(unet_coronal_models, batch_coronal_X)
    for i in range(t1_preprocessed_warped.shape[1]):
        prediction_coronal[i, 0, :, :] = np.rot90(np.squeeze(prediction_coronal[i, 0, :, :]))

    if verbose:
        print("Axial prediction.")
    prediction_axial = predict_batch(unet_axial_models, batch_axial_X)
    for i in range(t1_preprocessed_warped.shape[2]):
        prediction_axial[i, 0, :, :] = np.rot90(np.squeeze(prediction_axial[i, 0, :, :]), k=-1)

    if verbose:
        print("Restack image and transform back to native space.")

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    prediction_image_average = ants.image_clone(t1_preprocessed_warped) * 0

    for d in range(len(dimensions_to_predict)):
        which_batch_slices = range(t1_preprocessed_warped.shape[dimensions_to_predict[d]])
        if dimensions_to_predict[d] == 1:
            prediction_per_dimension = prediction_coronal[which_batch_slices, :, :, :]
        else:
            prediction_per_dimension = prediction_axial[which_batch_slices, :, :, :]
        prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
        prediction_image = ants.copy_image_info(t1_preprocessed_warped,
            ants.pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
                t1_preprocessed_warped.shape))
        prediction_image_average = prediction_image_average + (prediction_image - prediction_image_average) / (d + 1)

    probability_image = ants.apply_ants_transform_to_image(ants.invert_ants_transform(xfrm),
        prediction_image_average, t1) * ants.threshold_image(brain_mask, 0.5, 1, 1, 0)

    return probability_image
