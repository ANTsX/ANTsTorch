import numpy as np
import torch
import ants


def _load_state_dict(weights_file_name):
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def hippmapp3r_segmentation(t1,
                            do_preprocessing=True,
                            number_of_monte_carlo_iterations=30,
                            device=None,
                            verbose=False):

    """
    Perform HippMapp3r (hippocampal) segmentation described in

     https://www.ncbi.nlm.nih.gov/pubmed/31609046

    with models and architecture ported from

    https://github.com/mgoubran/HippMapp3r

    Additional documentation and attribution resources found at

    https://hippmapp3r.readthedocs.io/en/latest/

    PyTorch port of antspynet.utilities.hippmapp3r_segmentation.

    Preprocessing consists of:
       * n4 bias correction and
       * brain extraction
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        input image

    do_preprocessing : boolean
        See description above.

    number_of_monte_carlo_iterations : integer
        Number of Monte Carlo (spatial dropout) iterations used by the
        refine-stage network.  ANTsPyNet uses 30.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    ANTs labeled hippocampal image.

    Example
    -------
    >>> mask = hippmapp3r_segmentation(t1)
    """

    from ..architectures import create_hippmapp3r_unet_model_3d
    from ..utilities import preprocess_brain_image
    from ..utilities import get_pretrained_network
    from ..utilities import get_antstorch_data
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if verbose:
        print("*************  Preprocessing  ***************")
        print("")

    t1_preprocessed = t1
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1",
            template=None,
            do_bias_correction=True,
            do_denoising=False,
            device=device,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing["brain_mask"]

    if verbose:
        print("*************  Initial stage segmentation  ***************")
        print("")

    # Normalize to mprage_hippmapp3r space
    if verbose:
        print("    HippMapp3r: template normalization.")

    template_file_name_path = get_antstorch_data("mprage_hippmapp3r")
    template_image = ants.image_read(template_file_name_path)

    registration = ants.registration(fixed=template_image, moving=t1_preprocessed,
        type_of_transform="antsRegistrationSyNQuickRepro[t]", verbose=verbose)
    image = registration["warpedmovout"]
    transforms = dict(fwdtransforms=registration["fwdtransforms"],
                       invtransforms=registration["invtransforms"])

    # Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
    if verbose:
        print("    HippMapp3r: threshold.")

    image_array = image.numpy()
    image_robust_range = np.quantile(image_array[np.where(image_array != 0)], (0.02, 0.98))
    threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
    thresholded_mask = ants.threshold_image(image, -10000, threshold_value, 0, 1)
    thresholded_image = image * thresholded_mask

    # Standardize image
    if verbose:
        print("    HippMapp3r: standardize.")

    mean_image = np.mean(thresholded_image[thresholded_mask == 1])
    sd_image = np.std(thresholded_image[thresholded_mask == 1])
    image_normalized = (image - mean_image) / sd_image
    image_normalized = image_normalized * thresholded_mask

    # Trim and resample image
    if verbose:
        print("    HippMapp3r: trim and resample to (160, 160, 128).")

    image_cropped = ants.crop_image(image_normalized, thresholded_mask, 1)
    shape_initial_stage = (160, 160, 128)
    image_resampled = ants.resample_image(image_cropped, shape_initial_stage, use_voxels=True, interp_type=1)

    if verbose:
        print("    HippMapp3r: generate first network and download weights.")

    model_initial_stage = create_hippmapp3r_unet_model_3d(1, do_first_network=True)
    initial_stage_weights_file_name = get_pretrained_network("hippMapp3rInitial_pytorch",
        target_file_name="hippMapp3rInitial_pytorch.pt")
    model_initial_stage.load_state_dict(_load_state_dict(initial_stage_weights_file_name), strict=True)
    model_initial_stage.eval()
    model_initial_stage = model_initial_stage.to(device)

    if verbose:
        print("    HippMapp3r: prediction.")

    data_initial_stage = np.expand_dims(image_resampled.numpy(), axis=(0, 1)).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(data_initial_stage).float().to(device)
        mask_array = model_initial_stage(x).cpu().numpy()
    mask_image_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(np.squeeze(mask_array)))
    mask_image = ants.resample_image(mask_image_resampled, image.shape, use_voxels=True, interp_type=0)
    mask_image[mask_image >= 0.5] = 1
    mask_image[mask_image < 0.5] = 0

    #########################################
    #
    # Perform refined (stage 2) segmentation
    #

    if verbose:
        print("")
        print("")
        print("*************  Refine stage segmentation  ***************")
        print("")

    mask_array = np.squeeze(mask_array)
    centroid_indices = np.where(mask_array == 1)
    centroid = np.zeros((3,))
    centroid[0] = centroid_indices[0].mean()
    centroid[1] = centroid_indices[1].mean()
    centroid[2] = centroid_indices[2].mean()

    shape_refine_stage = (112, 112, 64)
    lower = (np.floor(centroid - 0.5 * np.array(shape_refine_stage)) - 1).astype(int)
    upper = (lower + np.array(shape_refine_stage)).astype(int)

    image_trimmed = ants.crop_indices(image_resampled, lower.astype(int), upper.astype(int))

    if verbose:
        print("    HippMapp3r: generate second network and download weights.")

    model_refine_stage = create_hippmapp3r_unet_model_3d(1, do_first_network=False)
    refine_stage_weights_file_name = get_pretrained_network("hippMapp3rRefine_pytorch",
        target_file_name="hippMapp3rRefine_pytorch.pt")
    model_refine_stage.load_state_dict(_load_state_dict(refine_stage_weights_file_name), strict=True)
    model_refine_stage.eval()
    model_refine_stage = model_refine_stage.to(device)

    data_refine_stage = np.expand_dims(image_trimmed.numpy(), axis=(0, 1)).astype(np.float32)

    if verbose:
        print("    HippMapp3r: Monte Carlo iterations (SpatialDropout).")

    # Monte Carlo inference: only the Dropout3d submodules (inside the
    # residual blocks) need train()-mode behavior; everything else in this
    # architecture is mode-independent (InstanceNorm3d has
    # track_running_stats=False).
    model_refine_stage.eval()
    for module in model_refine_stage.modules():
        if isinstance(module, torch.nn.Dropout3d):
            module.train()

    prediction_refine_stage = np.zeros(shape_refine_stage, dtype=np.float32)
    with torch.no_grad():
        x_refine = torch.from_numpy(data_refine_stage).float().to(device)
        for i in range(number_of_monte_carlo_iterations):
            if verbose:
                print("        Monte Carlo iteration", i + 1, "out of", number_of_monte_carlo_iterations)
            prediction = np.squeeze(model_refine_stage(x_refine).cpu().numpy())
            prediction_refine_stage = (prediction + i * prediction_refine_stage) / (i + 1)

    prediction_refine_stage_array = np.zeros(image_resampled.shape, dtype=np.float32)
    prediction_refine_stage_array[lower[0]:upper[0],
                                  lower[1]:upper[1],
                                  lower[2]:upper[2]] = prediction_refine_stage
    probability_mask_refine_stage_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(prediction_refine_stage_array))

    segmentation_image_resampled = ants.label_clusters(
        ants.threshold_image(probability_mask_refine_stage_resampled, 0.0, 0.5, 0, 1), min_cluster_size=10)
    segmentation_image_resampled[segmentation_image_resampled > 2] = 0
    geom = ants.label_geometry_measures(segmentation_image_resampled)
    if len(geom["VolumeInMillimeters"]) < 2:
        raise ValueError("Error: left and right hippocampus not found.")

    if geom["Centroid_x"][0] < geom["Centroid_x"][1]:
        segmentation_image_resampled[segmentation_image_resampled == 1] = 3
        segmentation_image_resampled[segmentation_image_resampled == 2] = 1
        segmentation_image_resampled[segmentation_image_resampled == 3] = 2

    segmentation_image = ants.apply_transforms(fixed=t1,
        moving=segmentation_image_resampled, transformlist=transforms["invtransforms"],
        whichtoinvert=[True], interpolator="genericLabel", verbose=verbose)

    return segmentation_image
