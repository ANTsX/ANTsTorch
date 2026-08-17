"""Scaling-and-squaring exponential of stationary physical velocity fields."""

import torch
from torch import Tensor, nn

from .bspline_domain import BSplineDomain
from .spatial_transform import compose_displacements


def scaling_and_squaring(velocity: Tensor, domain: BSplineDomain, steps: int = 7) -> Tensor:
    """Approximate ``Exp(velocity)`` as a fixed-to-moving displacement field."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    expected = (velocity.shape[0], domain.dimension) + domain.torch_size
    if tuple(velocity.shape) != expected:
        raise ValueError("velocity shape does not match domain")
    displacement = velocity / float(2**steps)
    for _ in range(steps):
        displacement = compose_displacements(displacement, displacement, domain)
    return displacement


class ScalingAndSquaring(nn.Module):
    def __init__(self, domain: BSplineDomain, steps: int = 7):
        super().__init__()
        self.domain = domain
        self.steps = steps

    def forward(self, velocity: Tensor) -> Tensor:
        return scaling_and_squaring(velocity, self.domain, self.steps)

