"""
PyTorch port of antspynet.utilities.super_resolution_utilities
(apply_super_resolution_model_to_image only -- see module docstring below).
"""
import time

import numpy as np
import torch
import ants


def apply_super_resolution_model_to_image(image,
                                           model,
                                           target_range=(-127.5, 127.5),
                                           regression_order=None,
                                           device=None,
                                           verbose=False):

    """
    Apply a pretrained deep back-projection network (DBPN) model for
    super resolution.

    PyTorch port of antspynet.utilities.apply_super_resolution_model_to_image.
    Applies the network to the whole image in a single forward pass -- this
    matches ANTsPyNet's own behavior, which despite its patch-extraction
    machinery always calls it with patch_size=image.shape and
    max_number_of_patches=1 (i.e. no real tiling).  Warning: this can be
    memory-intensive for large volumes; run on CPU if the GPU/MPS device
    cannot accommodate the full image.

    Note the metric functions that accompany this in ANTsPyNet (mse, mae,
    psnr, ssim, gmsd) were not ported -- they are not required by
    mri_super_resolution and are straightforward to add later if needed.

    Arguments
    ---------
    image : ANTsImage
        input image (2-D or 3-D, single- or multi-channel).

    model : torch.nn.Module
        pretrained super-resolution model with weights already loaded via
        load_state_dict, in eval() mode (this function also calls
        model.eval() defensively).  Its declared input_channel_size must
        match image.components.

    target_range : 2-tuple
        (min, max) intensity range the network was trained on.  The input
        is rescaled into this range before inference; the order is
        normalized (smaller value first) if given reversed, matching
        ANTsPyNet.

    regression_order : integer, optional
        If specified, match intensities of the super-resolved output back
        to a target-resampled version of the input via
        ants.regression_match_image(..., poly_order=regression_order).

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Super-resolution ANTsImage, upscaled by whatever factor the network
    itself produces (inferred from the ratio of output shape to input
    shape, per spatial axis -- so anisotropic expansion factors, e.g.
    [1, 1, 2], are handled correctly).

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> image_sr = apply_super_resolution_model_to_image(image, model)
    """

    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if image.dimension not in (2, 3):
        raise ValueError("Image dimension must be 2 or 3.")

    if target_range[0] > target_range[1]:
        target_range = (target_range[1], target_range[0])

    model.eval()
    model = model.to(device)

    # Channel-first array: (C, *spatial), matching torch's expected layout
    # (as opposed to Keras's channels-last convention used in ANTsPyNet).
    if image.components == 1:
        image_array = np.expand_dims(image.numpy().astype(np.float32), axis=0)
    else:
        channels = ants.split_channels(image)
        image_array = np.stack([c.numpy().astype(np.float32) for c in channels], axis=0)

    image_array = image_array - image_array.min()
    image_array = (image_array / image_array.max() * (target_range[1] - target_range[0])
                   + target_range[0])

    batch_X = np.expand_dims(image_array, axis=0)  # (1, C, *spatial)

    if verbose:
        print("Prediction")
    start_time = time.time()
    with torch.no_grad():
        x = torch.from_numpy(batch_X).float().to(device)
        prediction = model(x).cpu().numpy()
    if verbose:
        print("  (elapsed time: ", time.time() - start_time, ")")

    prediction = prediction[0]  # drop batch dim -> (C, *spatial_sr)

    if verbose:
        print("Reconstruct intensities")

    intensity_range = (float(image.min()), float(image.max()))
    prediction = prediction - prediction.min()
    prediction = (prediction / prediction.max() * (intensity_range[1] - intensity_range[0])
                  + intensity_range[0])

    expansion_factor = np.asarray(prediction.shape[1:]) / np.asarray(image_array.shape[1:])

    if verbose:
        print("ExpansionFactor:", str(expansion_factor))

    if image.components == 1:
        prediction_image = ants.from_numpy(prediction[0])
    else:
        component_images = [ants.from_numpy(prediction[k]) for k in range(image.components)]
        prediction_image = ants.merge_channels(component_images)

    prediction_image = ants.copy_image_info(image, prediction_image)
    ants.set_spacing(prediction_image, tuple(np.asarray(image.spacing) / expansion_factor))

    if regression_order is not None:
        reference_image = ants.resample_image_to_target(image, prediction_image)
        prediction_image = ants.regression_match_image(
            prediction_image, reference_image, poly_order=regression_order)

    return prediction_image
