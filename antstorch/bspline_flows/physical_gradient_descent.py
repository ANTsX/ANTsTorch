"""Physically normalized gradient descent for B-spline coefficients."""

from math import isfinite
from typing import Callable, Dict, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .bspline_domain import ImageDomain


class PhysicalGradientDescent:
    """Update B-spline coefficients with a bounded dense physical step.

    Parameters
    ----------
    gradient_step : float
        Fraction of the current image voxel diagonal used as the maximum
        dense velocity update. Must be in ``[0.1, 0.25]``.
    momentum : float
        Classical coefficient-gradient momentum in ``[0, 1)``.
    smoothing_sigma : float
        Gaussian smoothing sigma in physical units, applied to the
        coefficient-gradient lattice before dense normalization. Zero disables
        smoothing.

    Notes
    -----
    :meth:`step` requires the B-spline synthesis callable and image domain
    because a coefficient-space norm does not determine a physical dense-field
    update. Optimizer state is reset automatically when the coefficient shape,
    dtype, or device changes, as it does during pyramid refinement.
    """

    def __init__(
        self,
        gradient_step: float = 0.2,
        momentum: float = 0.0,
        smoothing_sigma: float = 0.0,
    ):
        if (
            not isinstance(gradient_step, (int, float))
            or isinstance(gradient_step, bool)
            or not isfinite(gradient_step)
            or not 0.1 <= gradient_step <= 0.25
        ):
            raise ValueError("gradient_step must be finite and between 0.1 and 0.25")
        if (
            not isinstance(momentum, (int, float))
            or isinstance(momentum, bool)
            or not isfinite(momentum)
            or not 0.0 <= momentum < 1.0
        ):
            raise ValueError("momentum must be finite and in [0, 1)")
        if (
            not isinstance(smoothing_sigma, (int, float))
            or isinstance(smoothing_sigma, bool)
            or not isfinite(smoothing_sigma)
            or smoothing_sigma < 0.0
        ):
            raise ValueError("smoothing_sigma must be finite and non-negative")
        self.gradient_step = float(gradient_step)
        self.momentum = float(momentum)
        self.smoothing_sigma = float(smoothing_sigma)
        self._momentum_buffer = None

    def reset(self) -> None:
        """Discard accumulated momentum."""
        self._momentum_buffer = None

    def zero_grad(self, coefficients: Tensor) -> None:
        """Clear the coefficient gradient without allocating a zero tensor."""
        coefficients.grad = None

    def _smooth(self, gradient: Tensor, domain: ImageDomain, closed=False) -> Tensor:
        if self.smoothing_sigma == 0.0:
            return gradient
        dimension = domain.dimension
        closed_axes = (closed,) * dimension if isinstance(closed, bool) else tuple(closed)
        lattice_itk = tuple(reversed(gradient.shape[2:]))
        spans = tuple(size if periodic else size - 3 for size, periodic in zip(lattice_itk, closed_axes))
        coefficient_spacing = tuple(extent / span for extent, span in zip(domain.physical_extent, spans))
        sigma_torch = tuple(
            reversed(tuple(self.smoothing_sigma / spacing for spacing in coefficient_spacing))
        )
        axes, radii = [], []
        for sigma in sigma_torch:
            radius = max(1, int(3.0 * sigma + 0.5))
            radii.append(radius)
            coordinate = torch.arange(-radius, radius + 1, dtype=gradient.dtype, device=gradient.device)
            kernel = torch.exp(-0.5 * (coordinate / sigma).square())
            axes.append(kernel / kernel.sum())
        kernel = axes[0]
        for axis_kernel in axes[1:]:
            kernel = kernel.unsqueeze(-1) * axis_kernel.reshape((1,) * kernel.ndim + (-1,))
        kernel = kernel.reshape((1, 1) + kernel.shape).expand((gradient.shape[1], 1) + kernel.shape)
        padding = tuple(value for radius in reversed(radii) for value in (radius, radius))
        padded = F.pad(gradient, padding, mode="replicate")
        convolution = F.conv2d if dimension == 2 else F.conv3d
        return convolution(padded, kernel, groups=gradient.shape[1])

    @torch.no_grad()
    def step(
        self,
        coefficients: Tensor,
        synthesis: Callable[[Tensor], Tensor],
        domain: ImageDomain,
        *,
        closed: Union[bool, tuple] = False,
    ) -> Dict[str, Tensor]:
        """Apply one descent step and return its physical scaling statistics."""
        if coefficients.grad is None:
            raise RuntimeError("coefficients must have a gradient before step()")
        if not torch.isfinite(coefficients.grad).all():
            raise FloatingPointError("coefficient gradient is non-finite")
        direction = self._smooth(coefficients.grad, domain, closed)
        if self.momentum:
            if (
                self._momentum_buffer is None
                or self._momentum_buffer.shape != direction.shape
                or self._momentum_buffer.dtype != direction.dtype
                or self._momentum_buffer.device != direction.device
            ):
                self._momentum_buffer = torch.zeros_like(direction)
            self._momentum_buffer.mul_(self.momentum).add_(direction)
            direction = self._momentum_buffer

        dense_direction = synthesis(direction)
        maximum_norm = dense_direction.square().sum(dim=1).sqrt().flatten(start_dim=1).amax(dim=1)
        voxel_diagonal = sum(spacing**2 for spacing in domain.spacing) ** 0.5
        target = maximum_norm.new_full(maximum_norm.shape, self.gradient_step * voxel_diagonal)
        scale = torch.where(maximum_norm > 0, target / maximum_norm, torch.zeros_like(maximum_norm))
        coefficient_scale = scale.reshape((coefficients.shape[0], 1) + (1,) * domain.dimension)
        coefficients.add_(direction * coefficient_scale, alpha=-1.0)
        return {"maximum_norm": maximum_norm, "target_step": target, "scale": scale}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(gradient_step={self.gradient_step}, "
            f"momentum={self.momentum}, smoothing_sigma={self.smoothing_sigma})"
        )
