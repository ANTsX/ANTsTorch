"""Contours and analytical forces used by DiReCT."""

import torch
from torch import Tensor

from ..syn.core.smoothing import separable_gaussian_filter


def binary_contour(mask: Tensor) -> Tensor:
    """Return the fully connected inner contour of a binary image."""
    if mask.ndim not in (4, 5) or mask.shape[1] != 1:
        raise ValueError("mask must have shape (N, 1, *spatial)")
    binary = mask > 0
    eroded = binary.clone()
    # ITK BinaryContourImageFilter uses FullyConnected=true in DiReCT.
    # Erode along each axis in sequence to include diagonal neighbours too.
    # Replicated edges match ITK's zero-flux boundary condition.
    for axis in range(2, mask.ndim):
        source = eroded
        updated = source.clone()
        for offset in (-1, 1):
            shifted = torch.roll(source, shifts=offset, dims=axis)
            boundary = [slice(None)] * binary.ndim
            boundary[axis] = 0 if offset == 1 else -1
            shifted[tuple(boundary)] = source[tuple(boundary)]
            updated &= shifted
        eroded = updated
    return (binary & ~eroded).to(mask.dtype)


def gaussian_scalar(image: Tensor, sigma, spacing=None, sigma_mode="voxel", maximum_error=None) -> Tensor:
    """Apply the shared separable Gaussian implementation to a scalar image."""
    field = image.movedim(1, -1)
    return separable_gaussian_filter(
        field, sigma=sigma, spacing=spacing, sigma_mode=sigma_mode, maximum_error=maximum_error
    ).movedim(-1, 1)


def normalized_probability_gradient(
    probability: Tensor,
    *,
    sigma: float,
    spacing,
    epsilon: float = 1e-3,
) -> Tensor:
    """Approximate smoothed gradient with ITK-order vector components.

    Unlike ITK's recursive Gaussian derivative, this smooths with a discrete
    Gaussian and then takes finite differences. See direct/README.md for
    the analytical direction test and limits of ANTs equivalence.
    """
    smoothed = gaussian_scalar(probability, sigma, spacing=spacing, sigma_mode="physical")
    spacing_torch = tuple(reversed(tuple(float(v) for v in spacing)))
    derivatives = torch.gradient(smoothed, spacing=spacing_torch, dim=tuple(range(2, smoothed.ndim)))
    gradient = torch.cat(tuple(reversed(derivatives)), dim=1)
    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    return torch.where(norm > epsilon, gradient / norm.clamp_min(epsilon), torch.zeros_like(gradient))


def direct_force(
    warped_white_probability: Tensor,
    gray_probability: Tensor,
    gray_mask: Tensor,
    gradient: Tensor,
    gradient_step: float,
) -> Tensor:
    """Compute the historical DiReCT demons-like update force."""
    delta = warped_white_probability - gray_probability
    speed = -delta * gray_probability * gray_mask * float(gradient_step)
    force = gradient * speed
    return torch.nan_to_num(force)
