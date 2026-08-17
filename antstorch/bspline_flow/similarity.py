"""Differentiable registration similarities and physical regularizers."""

import torch
from torch import Tensor

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

