"""ANTsImage interface for the tensor DiReCT implementation."""

import torch

from ..bspline_flows import ImageDomain
from ..syn.bridge import ants_image_to_tensor, tensor_to_ants_image
from ..syn.core.pipeline import auto_detect_device
from .core import direct_cortical_thickness


def kelly_kapowski(
    segmentation,
    gray_matter,
    white_matter,
    *,
    iterations: int = 45,
    gradient_step: float = 0.025,
    smoothing_sigma: float = 1.0,
    velocity_smoothing_variance: float = 1.5,
    integration_points: int = 10,
    thickness_prior: float = 10.0,
    optimizer: str = "direct",
    regularizer: str = "gaussian",
    device=None,
    verbose: bool = False,
):
    """ANTsImage-compatible DiReCT entry point implemented with PyTorch."""
    if segmentation.dimension not in (2, 3):
        raise ValueError("kelly_kapowski supports 2-D and 3-D images")
    for name, image in (("gray_matter", gray_matter), ("white_matter", white_matter)):
        if image.dimension != segmentation.dimension or image.shape != segmentation.shape:
            raise ValueError(f"{name} must match the segmentation domain")
    resolved_device = torch.device(device) if device is not None else auto_detect_device()
    seg = ants_image_to_tensor(segmentation, resolved_device, normalize=False)
    gray = ants_image_to_tensor(gray_matter, resolved_device, normalize=False)
    white = ants_image_to_tensor(white_matter, resolved_device, normalize=False)
    identity = tuple(
        tuple(float(i == j) for j in range(segmentation.dimension))
        for i in range(segmentation.dimension)
    )
    domain = ImageDomain(
        tuple(int(v) for v in segmentation.shape),
        tuple(float(v) for v in segmentation.spacing),
        tuple(float(v) for v in segmentation.origin),
        identity,
    )
    result = direct_cortical_thickness(
        seg, gray, white, domain,
        iterations=iterations,
        gradient_step=gradient_step,
        smoothing_sigma=smoothing_sigma,
        velocity_smoothing_variance=velocity_smoothing_variance,
        integration_points=integration_points,
        thickness_prior=thickness_prior,
        optimizer=optimizer,
        regularizer=regularizer,
        verbose=verbose,
    )
    return tensor_to_ants_image(result.thickness, segmentation)
