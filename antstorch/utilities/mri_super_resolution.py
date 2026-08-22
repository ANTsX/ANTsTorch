import torch
import ants


# Fallback DBPN hyperparameters, used only if a weights file does not carry
# its own embedded "architecture_kwargs" (see _load_state_dict_and_kwargs
# below). These match siq's (https://github.com/stnava/siq) default_dbpn()
# defaults -- number_of_base_filters=64, number_of_feature_filters=256,
# number_of_back_projection_stages=7, convolution_kernel_size=6 (always
# isotropic, confirmed directly from siq/get_data.py's dimensionality==3
# branch, which always passes (convn, convn, convn) even for anisotropic
# strides), last_convolution=3 -- but it is NOT confirmed that the real
# "sig_smallshort_train_*" weights were trained with these exact filter/
# stage counts rather than one of default_dbpn()'s smaller named presets
# ("small": nfilt=32/nff=64/nbp=4, "tiny": nfilt=32/nff=64/nbp=2). Every
# .pt produced by tools/convert_mri_super_resolution_bespoke.py embeds the
# real values read directly out of each file's own model_config JSON, so
# this fallback should never actually be used once weights are converted
# with that script -- it exists only so a bare (unwrapped) state_dict
# .pt still loads with a sane default.
_DBPN_3D_FALLBACK_KWARGS = dict(
    number_of_base_filters=64,
    number_of_feature_filters=256,
    number_of_back_projection_stages=7,
    convolution_kernel_size=6,
    last_convolution=3,
)

_VALID_EXPANSION_FACTORS = {
    (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 6), (2, 2, 2), (2, 2, 4),
}


def _load_state_dict_and_kwargs(weights_file_name, expansion_factor):
    """Returns (state_dict, architecture_kwargs). Prefers the
    "architecture_kwargs" saved alongside the weights by
    tools/convert_mri_super_resolution_bespoke.py (read directly from the
    real ANTsPyNet .h5's own model_config JSON at conversion time -- see
    that script's module docstring); falls back to
    _DBPN_3D_FALLBACK_KWARGS + strides=expansion_factor for a bare
    state_dict .pt that doesn't carry this metadata."""
    loaded = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        kwargs = loaded.get("architecture_kwargs")
        if kwargs is None:
            kwargs = dict(_DBPN_3D_FALLBACK_KWARGS, strides=expansion_factor)
        return loaded["state_dict"], kwargs
    return loaded, dict(_DBPN_3D_FALLBACK_KWARGS, strides=expansion_factor)


def mri_super_resolution(image,
                         expansion_factor=(1, 1, 2),
                         feature="vgg",
                         target_range=(1, 0),
                         poly_order="hist",
                         architecture_kwargs=None,
                         device=None,
                         verbose=False):

    """
    Perform super-resolution of MRI data using a deep back-projection
    network (DBPN).  Work described in

    https://www.medrxiv.org/content/10.1101/2023.02.02.23285376v1

    with the GitHub repo located at https://github.com/stnava/siq

    PyTorch port of antspynet.utilities.mri_super_resolution.  UNVERIFIED
    end-to-end: unlike the other antstorch application ports, no real
    converted weights file has been round-tripped against a real SIQ .h5 at
    port time. As of 2026-08-22 this builds
    antstorch.architectures.create_siq_dbpn_super_resolution_model_3d (NOT
    the older create_deep_back_projection_network_model_3d -- reading the
    real training code at https://github.com/stnava/siq showed the SIQ
    models scale up via UpSampling3D+Conv3D, not a learned Conv3DTranspose,
    a structurally different architecture, see that class's docstring). The
    real per-model hyperparameters (filter/stage counts) are read directly
    out of each real .h5's embedded model_config JSON by
    tools/convert_mri_super_resolution_bespoke.py and saved alongside the
    weights -- see _load_state_dict_and_kwargs above -- so no
    _DBPN_3D_CONFIGS-style lookup table is hardcoded here any more. Use
    architecture_kwargs to override anything once you've confirmed it.

    Note that some preprocessing possibilities for the input includes:
      * Truncate intensity (see ants.iMath(..., 'TruncateIntensity', ...)

    Arguments
    ---------
    image : ANTsImage
        magnetic resonance image (3-D).

    expansion_factor : 3-tuple
        Specifies the increase in resolution per dimension.  Possibilities
        include:
          * (1, 1, 2)
          * (1, 1, 3)
          * (1, 1, 4)
          * (1, 1, 6)
          * (2, 2, 2)
          * (2, 2, 4)

    feature : string
        Type of network.  Choices include "grader" or "vgg".  Note "grader"
        is not available for expansion_factor (1, 1, 6) (only "vgg" is).

    target_range : 2-tuple
        Range for apply_super_resolution_model_to_image.

    poly_order : int or 'hist' or None
        Parameter for regression matching or specification of histogram
        matching applied between the super-resolved output and the
        original input.  Set to None to skip intensity matching.

    architecture_kwargs : dict, optional
        Override any of the create_siq_dbpn_super_resolution_model_3d
        constructor kwargs (e.g. convolution_kernel_size,
        number_of_base_filters, number_of_back_projection_stages) beyond
        what was embedded in the weights file itself (or the fallback
        defaults, for an older bare state_dict .pt).

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    The super-resolved image.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> image_sr = mri_super_resolution(image)
    """

    from ..architectures import create_siq_dbpn_super_resolution_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import apply_super_resolution_model_to_image
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if image.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    expansion_factor = tuple(expansion_factor)
    if expansion_factor not in _VALID_EXPANSION_FACTORS:
        raise ValueError("expansion_factor must be one of: " +
                         str(sorted(_VALID_EXPANSION_FACTORS)))

    network_basename = ("sig_smallshort_train_" +
                        'x'.join(map(str, expansion_factor)) +
                        '_1chan_feat' + feature + 'L6_best_mdl')

    weights_file_name = get_pretrained_network(network_basename + "_pytorch",
        target_file_name=network_basename + "_pytorch.pt")

    state_dict, config = _load_state_dict_and_kwargs(weights_file_name, expansion_factor)
    config = dict(config)
    config.update(architecture_kwargs or {})

    model = create_siq_dbpn_super_resolution_model_3d(
        input_channel_size=1, number_of_outputs=1, **config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    image_sr = apply_super_resolution_model_to_image(
        image, model, target_range=target_range, regression_order=None,
        device=device, verbose=verbose)

    if poly_order is not None:
        if verbose:
            print("Match intensity with " + str(poly_order))
        if poly_order == "hist":
            if verbose:
                print("Histogram match input/output images.")
            image_sr = ants.histogram_match_image(image_sr, image)
        else:
            if verbose:
                print("Regression match input/output images.")
            image_resampled = ants.resample_image_to_target(image, image_sr)
            image_sr = ants.regression_match_image(image_sr, image_resampled, poly_order=poly_order)

    return image_sr
