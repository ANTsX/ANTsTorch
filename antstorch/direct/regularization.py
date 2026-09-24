"""Velocity-field regularization for DiReCT."""

import math
import torch
from torch import Tensor

from ..syn.core.smoothing import (
    apply_dsti_green_operator,
    apply_sobolev_green_operator,
    separable_gaussian_filter,
)


def stationary_boundary(field: Tensor) -> Tensor:
    """Set every spatial boundary of a channel-first field to zero."""
    result = field.clone()
    for axis in range(2, field.ndim):
        first = [slice(None)] * field.ndim
        last = [slice(None)] * field.ndim
        first[axis] = 0
        last[axis] = -1
        result[tuple(first)] = 0
        result[tuple(last)] = 0
    return result


def regularize_velocity(
    field: Tensor,
    *,
    mode: str = "gaussian",
    variance: float = 1.5,
    spacing=None,
) -> Tensor:
    """Regularize a channel-first physical displacement field."""
    if variance <= 0 or mode == "none":
        return stationary_boundary(field)
    channel_last = field.movedim(1, -1)
    if mode == "gaussian":
        smoothed = separable_gaussian_filter(channel_last, sigma=math.sqrt(variance))
    elif mode == "sobolev":
        smoothed = apply_sobolev_green_operator(
            channel_last, fluid_sigma=variance, alpha=variance, spacing=spacing
        )
    elif mode == "dsti":
        smoothed = apply_dsti_green_operator(channel_last, fluid_sigma=variance, alpha=variance)
    else:
        raise ValueError("regularizer must be 'gaussian', 'sobolev', 'dsti', or 'none'")
    return stationary_boundary(smoothed.movedim(-1, 1))
