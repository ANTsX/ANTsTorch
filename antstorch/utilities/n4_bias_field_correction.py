"""ANTsImage interface for differentiable ANTsTorch N4 correction."""

from typing import Optional

import ants
import torch

from ..bspline_flows import ImageDomain
from ..bspline_flows import n4_bias_field_correction_tensor
from ..syn.bridge import ants_image_to_tensor, tensor_to_ants_image
from .device_manager import get_default_device


def _validate_scalar_image(image, name: str) -> None:
    if not ants.is_image(image):
        raise TypeError(f"{name} must be an ANTsImage")
    if image.dimension not in (2, 3) or image.components != 1:
        raise ValueError(f"{name} must be a scalar 2-D or 3-D ANTsImage")


def _validate_optional_image(image, reference, name: str) -> None:
    if image is None:
        return
    _validate_scalar_image(image, name)
    if image.shape != reference.shape or not ants.image_physical_space_consistency(
        reference, image
    ):
        raise ValueError(f"{name} must occupy the same physical space as image")


def n4_bias_field_correction(
    image,
    mask=None,
    *,
    rescale_intensities: bool = False,
    shrink_factor: int = 4,
    convergence: Optional[dict] = None,
    spline_param=None,
    return_bias_field: bool = False,
    weight_mask=None,
    number_of_histogram_bins: int = 200,
    wiener_filter_noise: float = 0.01,
    bias_field_fwhm: float = 0.15,
    stable_accumulation: Optional[bool] = None,
    device=None,
    verbose: bool = False,
):
    """Correct an ANTsImage with ANTsTorch's differentiable N4 engine.

    This provisional high-level interface mirrors the principal options of
    :func:`ants.n4_bias_field_correction`. It converts ANTs images to tensors,
    runs the tensor implementation, and restores the input image geometry.

    Parameters
    ----------
    image : ANTsImage
        Scalar 2-D or 3-D image to correct.
    mask : ANTsImage, optional
        Nonzero voxels define the correction domain. The default includes the
        full image.
    rescale_intensities : bool
        Rescale corrected intensities to the masked input range.
    shrink_factor : int
        Subsampling factor used while estimating the bias field.
    convergence : dict, optional
        ``{"iters": [...], "tol": value}`` fitting schedule.
    spline_param : float or sequence, optional
        Physical knot spacing when scalar, or B-spline mesh size in ITK
        x-y-z order when a sequence.
    return_bias_field : bool
        Return the multiplicative bias field instead of the corrected image.
    weight_mask : ANTsImage, optional
        Nonnegative confidence weights in the same physical space as ``image``.
    number_of_histogram_bins : int
        Number of bins used for histogram sharpening.
    wiener_filter_noise : float
        Wiener filter noise parameter.
    bias_field_fwhm : float
        Bias-field full width at half maximum.
    stable_accumulation : bool, optional
        Use deterministic matrix reductions. The tensor engine defaults to
        this mode on MPS and to vectorized scatter elsewhere.
    device : str or torch.device, optional
        PyTorch device. The configured ANTsTorch default is used when omitted.
    verbose : bool
        Report per-level setup time and per-iteration convergence.

    Returns
    -------
    ANTsImage
        Corrected image, or the bias field when ``return_bias_field=True``.
    """
    _validate_scalar_image(image, "image")
    _validate_optional_image(mask, image, "mask")
    _validate_optional_image(weight_mask, image, "weight_mask")

    resolved_device = (
        torch.device(device) if device is not None else get_default_device()
    )
    domain = ImageDomain(
        size=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in image.spacing),
        origin=tuple(float(value) for value in image.origin),
        direction=tuple(
            tuple(float(value) for value in row) for row in image.direction
        ),
    )
    image_tensor = ants_image_to_tensor(
        image, resolved_device, normalize=False
    )
    mask_tensor = (
        ants_image_to_tensor(mask, resolved_device, normalize=False)
        if mask is not None
        else None
    )
    weight_tensor = (
        ants_image_to_tensor(weight_mask, resolved_device, normalize=False)
        if weight_mask is not None
        else None
    )
    result = n4_bias_field_correction_tensor(
        image_tensor,
        domain,
        mask_tensor,
        rescale_intensities=rescale_intensities,
        shrink_factor=shrink_factor,
        convergence=convergence,
        spline_param=spline_param,
        return_bias_field=return_bias_field,
        weight_mask=weight_tensor,
        number_of_histogram_bins=number_of_histogram_bins,
        wiener_filter_noise=wiener_filter_noise,
        bias_field_fwhm=bias_field_fwhm,
        stable_accumulation=stable_accumulation,
        verbose=verbose,
    )
    return tensor_to_ants_image(result, image)
