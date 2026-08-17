"""Differentiable, tensor-only cubic B-spline synthesis."""

from itertools import product
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .bspline_domain import BSplineDomain


def cubic_bspline_basis(value: Tensor) -> Tensor:
    """Evaluate ITK's centered, cardinal, cubic B-spline kernel."""
    absolute = value.abs()
    inner = (4.0 - 6.0 * absolute.square() + 3.0 * absolute.pow(3)) / 6.0
    outer = (2.0 - absolute).clamp_min(0.0).pow(3) / 6.0
    return torch.where(absolute < 1.0, inner, torch.where(absolute < 2.0, outer, value.new_zeros(())))


def _as_bools(value: Union[bool, Sequence[bool]], dimension: int) -> Tuple[bool, ...]:
    result = (value,) * dimension if isinstance(value, bool) else tuple(bool(v) for v in value)
    if len(result) != dimension:
        raise ValueError(f"closed must have {dimension} values")
    return result


def synthesize_bspline_velocity(
    coefficients: Tensor,
    domain: BSplineDomain,
    *,
    closed: Union[bool, Sequence[bool]] = False,
    stationary_boundary: bool = False,
    chunk_size: Optional[int] = 262144,
) -> Tensor:
    """Map cubic B-spline coefficients to a dense stationary velocity field.

    ``coefficients`` has shape ``(N, C, Ky, Kx)`` or ``(N, C, Kz, Ky, Kx)``.
    The result has shape ``(N, C, Y, X)`` or ``(N, C, Z, Y, X)``.  Coefficient
    and output spatial axes are PyTorch-reversed relative to ITK metadata.

    Open dimensions require at least four coefficients and use ``K - 3``
    spans. Closed dimensions use ``K`` spans and periodic coefficient lookup.
    ``stationary_boundary`` zeros every dense-domain face. ITK implements its
    option during coefficient fitting with high-weight zero observations; this
    exact synthesis mask is the meaningful analogue when coefficients are the
    input and no fitting is performed.
    """
    if not isinstance(domain, BSplineDomain):
        raise TypeError("domain must be a BSplineDomain")
    if coefficients.ndim != domain.dimension + 2:
        raise ValueError(f"expected {domain.dimension + 2}-D coefficients")
    if not (coefficients.is_floating_point() or coefficients.is_complex()):
        raise TypeError("coefficients must have a floating-point dtype")
    if coefficients.is_complex():
        raise TypeError("complex coefficients are not supported")

    lattice_torch = tuple(coefficients.shape[2:])
    lattice_itk = tuple(reversed(lattice_torch))
    closed_axes = _as_bools(closed, domain.dimension)
    if any(k < 4 for k in lattice_itk):
        raise ValueError("cubic synthesis requires at least four coefficients per dimension")

    point_count = prod(domain.size)
    if chunk_size is None:
        chunk_size = point_count
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive or None")

    flat_coefficients = coefficients.flatten(start_dim=2)
    output_chunks = []
    offsets = []
    stride = 1
    for k in lattice_itk:
        offsets.append(stride)
        stride *= k

    for begin in range(0, point_count, chunk_size):
        linear = torch.arange(begin, min(begin + chunk_size, point_count), device=coefficients.device)
        remaining = linear
        dense_indices = []
        for size in domain.size:  # x is fastest in both ITK and flattened Z,Y,X storage
            dense_indices.append(remaining.remainder(size))
            remaining = torch.div(remaining, size, rounding_mode="floor")

        neighbors = []
        weights = []
        for dense_index, dense_size, lattice_size, periodic in zip(
            dense_indices, domain.size, lattice_itk, closed_axes
        ):
            spans = lattice_size if periodic else lattice_size - 3
            # The last ITK image sample is the left-limit at the half-open endpoint.
            coordinate = dense_index.to(coefficients.dtype) * (float(spans) / float(dense_size - 1))
            coordinate = torch.where(
                dense_index == dense_size - 1,
                torch.nextafter(coordinate, coordinate.new_full((), float("-inf"))),
                coordinate,
            )
            base = torch.floor(coordinate).to(torch.long)
            local = torch.arange(4, device=coefficients.device)
            index = base[:, None] + local
            neighbors.append(index.remainder(lattice_size) if periodic else index)
            weights.append(cubic_bspline_basis(coordinate[:, None] - index.to(coordinate.dtype) + 1.0))

        chunk = coefficients.new_zeros((coefficients.shape[0], coefficients.shape[1], linear.numel()))
        for support in product(range(4), repeat=domain.dimension):
            coefficient_index = sum(
                neighbors[d][:, support[d]] * offsets[d] for d in range(domain.dimension)
            )
            weight = torch.ones(linear.numel(), dtype=coefficients.dtype, device=coefficients.device)
            for d in range(domain.dimension):
                weight = weight * weights[d][:, support[d]]
            chunk = chunk + flat_coefficients.index_select(2, coefficient_index) * weight[None, None, :]
        output_chunks.append(chunk)

    output = torch.cat(output_chunks, dim=2).reshape(coefficients.shape[:2] + domain.torch_size)
    if stationary_boundary:
        mask = torch.ones(domain.torch_size, dtype=coefficients.dtype, device=coefficients.device)
        for axis in range(domain.dimension):
            selection = [slice(None)] * domain.dimension
            selection[axis] = 0
            mask[tuple(selection)] = 0
            selection[axis] = -1
            mask[tuple(selection)] = 0
        output = output * mask
    return output


class CubicBSplineSynthesis(nn.Module):
    """``nn.Module`` wrapper around :func:`synthesize_bspline_velocity`."""

    def __init__(self, domain: BSplineDomain, *, closed=False, stationary_boundary=False, chunk_size=262144):
        super().__init__()
        self.domain = domain
        self.closed = _as_bools(closed, domain.dimension)
        self.stationary_boundary = bool(stationary_boundary)
        self.chunk_size = chunk_size

    def forward(self, coefficients: Tensor) -> Tensor:
        return synthesize_bspline_velocity(
            coefficients,
            self.domain,
            closed=self.closed,
            stationary_boundary=self.stationary_boundary,
            chunk_size=self.chunk_size,
        )

