"""Differentiable registration similarities and physical regularizers."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .bspline_domain import BSplineDomain


def mean_squared_error(fixed: Tensor, warped_moving: Tensor) -> Tensor:
    if fixed.shape != warped_moving.shape:
        raise ValueError("fixed and warped moving images must have identical shapes")
    return (fixed - warped_moving).square().mean()


def normalized_cross_correlation_loss(fixed: Tensor, warped_moving: Tensor, eps: float = 1e-8) -> Tensor:
    """Return one minus mean global normalized cross-correlation."""
    if fixed.shape != warped_moving.shape:
        raise ValueError("fixed and warped moving images must have identical shapes")
    axes = tuple(range(2, fixed.ndim))
    fixed_centered = fixed - fixed.mean(dim=axes, keepdim=True)
    moving_centered = warped_moving - warped_moving.mean(dim=axes, keepdim=True)
    numerator = (fixed_centered * moving_centered).sum(dim=axes)
    denominator = torch.sqrt(
        fixed_centered.square().sum(dim=axes) * moving_centered.square().sum(dim=axes) + eps
    )
    return 1.0 - (numerator / denominator).mean()


def ants_neighborhood_correlation_loss(fixed: Tensor, warped_moving: Tensor, radius=2) -> Tensor:
    """Negative mean squared local correlation used by ITK's ANTs metric.

    ``fixed`` and ``warped_moving`` have shape ``(N, C, Y, X)`` or
    ``(N, C, Z, Y, X)``. An integer radius is shared by every axis; a tuple
    uses ITK order ``(x, y[, z])``. Each window has size ``2 * radius + 1``
    and is truncated at image boundaries. The returned value is the negative
    mean of the squared, locally mean-centered correlation, matching the sign
    of ``itk::ANTSNeighborhoodCorrelationImageToImageMetricv4``.

    Unlike ITK's hand-derived dense-transform derivative approximation, this
    implementation uses PyTorch autograd to differentiate the exact tensor
    expression through all samples in each neighborhood.
    """
    if fixed.shape != warped_moving.shape:
        raise ValueError("fixed and warped moving images must have identical shapes")
    dimension = fixed.ndim - 2
    if dimension not in (2, 3):
        raise ValueError("ANTS neighborhood correlation supports 2-D or 3-D images")
    if not fixed.is_floating_point() or not warped_moving.is_floating_point():
        raise TypeError("fixed and warped moving images must be floating point")
    if isinstance(radius, bool):
        raise TypeError("radius must be an integer or a sequence of integers")
    if isinstance(radius, int):
        radius_itk = (radius,) * dimension
    else:
        radius_itk = tuple(radius)
        if len(radius_itk) != dimension:
            raise ValueError(f"radius must have {dimension} values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in radius_itk):
        raise TypeError("radius values must be integers")
    if any(value < 0 for value in radius_itk):
        raise ValueError("radius values must be non-negative")

    radius_torch = tuple(reversed(radius_itk))
    kernel_size = tuple(2 * value + 1 for value in radius_torch)
    channels = fixed.shape[1]
    kernel = fixed.new_ones((channels, 1) + kernel_size)
    convolution = F.conv2d if dimension == 2 else F.conv3d

    def window_sum(value):
        return convolution(value, kernel, padding=radius_torch, groups=channels)

    count_kernel = fixed.new_ones((1, 1) + kernel_size)
    count = convolution(
        fixed.new_ones((fixed.shape[0], 1) + fixed.shape[2:]),
        count_kernel,
        padding=radius_torch,
    )
    sum_fixed = window_sum(fixed)
    sum_moving = window_sum(warped_moving)
    sum_fixed2 = window_sum(fixed.square())
    sum_moving2 = window_sum(warped_moving.square())
    sum_fixed_moving = window_sum(fixed * warped_moving)
    fixed_mean = sum_fixed / count
    moving_mean = sum_moving / count
    fixed_variance = sum_fixed2 - 2.0 * fixed_mean * sum_fixed + count * fixed_mean.square()
    moving_variance = sum_moving2 - 2.0 * moving_mean * sum_moving + count * moving_mean.square()
    covariance = (
        sum_fixed_moving
        - moving_mean * sum_fixed
        - fixed_mean * sum_moving
        + count * fixed_mean * moving_mean
    )
    denominator = fixed_variance * moving_variance
    epsilon = torch.finfo(fixed.dtype).eps
    valid = denominator.abs() > epsilon
    # ``torch.where`` evaluates both branches. Dividing by the original zero
    # denominator would therefore create NaNs in autograd even where the ITK
    # constant-neighborhood fallback of one is selected.
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    local_correlation = torch.where(
        valid,
        covariance.square() / safe_denominator,
        torch.ones_like(denominator),
    )
    return -local_correlation.mean()


def squared_l2_energy(value: Tensor) -> Tensor:
    return value.square().mean()


def bending_energy(field: Tensor, domain: BSplineDomain) -> Tensor:
    """Mean squared physical second derivatives (mixed terms counted twice).

    ITK direction matrices are orthonormal, so this Hessian Frobenius norm is
    invariant to the direction rotation/reflection; spacing is handled exactly.
    """
    if field.ndim != domain.dimension + 2 or tuple(field.shape[2:]) != domain.torch_size:
        raise ValueError("field shape does not match domain")
    first = []
    for axis, spacing in enumerate(domain.spacing):
        torch_axis = field.ndim - 1 - axis
        first.append(torch.gradient(field, spacing=(spacing,), dim=(torch_axis,))[0])
    terms = []
    for i, derivative in enumerate(first):
        for j in range(i, domain.dimension):
            torch_axis = field.ndim - 1 - j
            second = torch.gradient(derivative, spacing=(domain.spacing[j],), dim=(torch_axis,))[0]
            terms.append(second.square().mean() * (2.0 if i != j else 1.0))
    return torch.stack(terms).sum()
