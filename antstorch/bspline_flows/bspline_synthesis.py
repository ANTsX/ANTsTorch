"""Differentiable, tensor-only cubic B-spline synthesis."""

from itertools import product
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .bspline_domain import ImageDomain


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
    domain: ImageDomain,
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
    if not isinstance(domain, ImageDomain):
        raise TypeError("domain must be a ImageDomain")
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

    spans_per_axis = tuple(
        lattice_size if periodic else lattice_size - 3
        for lattice_size, periodic in zip(lattice_itk, closed_axes)
    )
    # A single open axis with only one span (the minimal 4-control-point
    # mesh) makes every dense sample along that axis share the same 4
    # neighbor indices, so the whole combined index table collapses onto a
    # handful of distinct values repeated over every sample. Gathering that
    # pattern in one vectorized ``index_select`` measures reliably slower
    # than the plain per-corner loop (~1.4x slower at spans==1, vs. ~2.5-3x
    # faster at spans>=2), so the loop is kept for that one degenerate case.
    use_vectorized_gather = min(spans_per_axis) > 1

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
            if not periodic:
                # Clamp against the *exact* mathematical upper bound (not
                # against a one-ULP nudge of whatever the multiplication
                # above actually computed). dense_index * spans / (dense_size
                # - 1) is only exactly `spans` in exact arithmetic; in
                # float32 it can land a few ULPs *above* `spans` -- not just
                # at the last sample, occasionally at interior samples too,
                # depending on spans/dense_size -- and a single nextafter()
                # step off of an already-overshot value isn't guaranteed to
                # bring it back under `spans`. That silently pushes
                # base = floor(coordinate) to `spans`, and the neighbor
                # index base+3 to `spans + 3 == lattice_size`, one past the
                # last valid coefficient index -- surfacing as an unrelated-
                # looking "index out of range in self" from index_select
                # deep in the vectorized gather path below. Clamping to the
                # exact value's nextafter (matching the already-correct
                # technique in _bspline_fit_geometry) is robust regardless
                # of how the raw coordinate was rounded.
                safe_max = torch.nextafter(
                    coordinate.new_full((), float(spans)), coordinate.new_full((), float("-inf"))
                )
                coordinate = coordinate.clamp_max(safe_max)
            base = torch.floor(coordinate).to(torch.long)
            local = torch.arange(4, device=coefficients.device)
            index = base[:, None] + local
            neighbors.append(index.remainder(lattice_size) if periodic else index)
            weights.append(cubic_bspline_basis(coordinate[:, None] - index.to(coordinate.dtype) + 1.0))

        if use_vectorized_gather:
            # Combine the per-axis 4-point stencils into one (chunk, 4**D)
            # index/weight table via broadcasting, then gather and reduce in
            # a single fused pass. This replaces a Python-level loop over
            # the 4**D corner combinations (each doing its own index_select
            # + multiply-add) with one vectorized gather; the result is
            # identical up to floating-point summation order. A flattened
            # index_select is used rather than the equivalent
            # multi-dimensional advanced-indexing gather
            # (``flat_coefficients[:, :, combined_index]``), since it lowers
            # to a single specialized gather kernel instead of the more
            # general (and costlier) advanced-indexing path.
            combined_index = neighbors[0] * offsets[0]
            combined_weight = weights[0]
            for d in range(1, domain.dimension):
                combined_index = combined_index.unsqueeze(-1) + (neighbors[d] * offsets[d]).unsqueeze(1)
                combined_weight = combined_weight.unsqueeze(-1) * weights[d].unsqueeze(1)
                combined_index = combined_index.reshape(combined_index.shape[0], -1)
                combined_weight = combined_weight.reshape(combined_weight.shape[0], -1)
            support_count = combined_index.shape[1]
            gathered = flat_coefficients.index_select(2, combined_index.reshape(-1)).reshape(
                coefficients.shape[0], coefficients.shape[1], linear.numel(), support_count
            )
            chunk = (gathered * combined_weight[None, None]).sum(dim=-1)
        else:
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


def _bspline_fit_geometry(
    sample_shape: Sequence[int],
    lattice_itk: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
):
    """Support-point indices/weights for one B-spline scattered-data update.

    ``sample_shape`` is a regular dense grid's spatial shape, in PyTorch
    order. This only depends on geometry (shapes and the lattice
    resolution), never on sample values, so callers with an iterative loop
    (e.g. N4) should compute it once per lattice resolution and reuse it
    across iterations rather than rebuilding it every time.
    """
    dimension = len(lattice_itk)
    if any(size < 2 for size in sample_shape):
        raise ValueError(
            "a fitting dimension has fewer than 2 samples "
            f"{tuple(sample_shape)}; scattered-data fitting needs at least 2."
        )
    torch_coordinates = torch.meshgrid(
        *[torch.arange(n, device=device) for n in sample_shape], indexing="ij"
    )
    itk_coordinates = tuple(reversed(torch_coordinates))

    neighbors, basis = [], []
    for coordinate, dense_size, lattice_size in zip(
        itk_coordinates, tuple(reversed(sample_shape)), lattice_itk
    ):
        spans = lattice_size - 3
        u = coordinate.to(dtype) * (float(spans) / float(dense_size - 1))
        u = u.clamp_max(torch.nextafter(u.new_tensor(float(spans)), u.new_tensor(float("-inf"))))
        base = torch.floor(u).to(torch.long)
        local = torch.arange(4, device=device)
        index = base[..., None] + local
        neighbors.append(index)
        basis.append(cubic_bspline_basis(u[..., None] - index.to(u.dtype) + 1.0))

    support_indices, support_basis = [], []
    strides, stride = [], 1
    for size in lattice_itk:
        strides.append(stride)
        stride *= size
    for support in product(range(4), repeat=dimension):
        index = sum(neighbors[d][..., support[d]] * strides[d] for d in range(dimension))
        value = torch.ones(sample_shape, dtype=dtype, device=device)
        for d in range(dimension):
            value = value * basis[d][..., support[d]]
        support_indices.append(index.flatten())
        support_basis.append(value.flatten())

    indices = torch.stack(support_indices)
    basis_values = torch.stack(support_basis)
    squared_sum = basis_values.square().sum(dim=0).clamp_min(eps)
    coefficient_count = stride
    return indices, basis_values, squared_sum, coefficient_count


def _bspline_fit_geometry_points(
    parametric_u: Sequence[Tensor],
    lattice_itk: Sequence[int],
    closed_axes: Sequence[bool],
    eps: float,
):
    """Build fit geometry for arbitrary parametric sample locations.

    This is the scattered-point analogue of :func:`_bspline_fit_geometry`.
    Each tensor in ``parametric_u`` contains one axis's internal B-spline
    coordinate for all samples. The returned representation is accepted by
    the same context and solve primitives as regular-grid geometry.
    """
    dimension = len(lattice_itk)
    device = parametric_u[0].device
    dtype = parametric_u[0].dtype
    point_count = parametric_u[0].shape[0]

    neighbors, basis = [], []
    for u, lattice_size, closed in zip(parametric_u, lattice_itk, closed_axes):
        base = torch.floor(u).to(torch.long)
        local = torch.arange(4, device=device)
        index = base[:, None] + local
        neighbors.append(index.remainder(lattice_size) if closed else index)
        basis.append(cubic_bspline_basis(u[:, None] - index.to(u.dtype) + 1.0))

    support_indices, support_basis = [], []
    strides, stride = [], 1
    for size in lattice_itk:
        strides.append(stride)
        stride *= size
    for support in product(range(4), repeat=dimension):
        index = sum(neighbors[d][:, support[d]] * strides[d] for d in range(dimension))
        value = torch.ones(point_count, dtype=dtype, device=device)
        for d in range(dimension):
            value = value * basis[d][:, support[d]]
        support_indices.append(index)
        support_basis.append(value)

    indices = torch.stack(support_indices)
    basis_values = torch.stack(support_basis)
    squared_sum = basis_values.square().sum(dim=0).clamp_min(eps)
    return indices, basis_values, squared_sum, stride


def _concat_bspline_fit_geometries(geometries):
    """Concatenate sample axes of geometries sharing one coefficient lattice."""
    if not geometries:
        raise ValueError("at least one B-spline fit geometry is required")
    coefficient_count = geometries[0][3]
    if any(geometry[3] != coefficient_count for geometry in geometries[1:]):
        raise ValueError("B-spline fit geometries must share a coefficient lattice")
    return (
        torch.cat([geometry[0] for geometry in geometries], dim=1),
        torch.cat([geometry[1] for geometry in geometries], dim=1),
        torch.cat([geometry[2] for geometry in geometries], dim=0),
        coefficient_count,
    )


def _select_bspline_fit_geometry(geometry, point_mask: Tensor):
    """Select a subset of samples from a fit geometry."""
    indices, basis_values, squared_sum, coefficient_count = geometry
    return (
        indices[:, point_mask],
        basis_values[:, point_mask],
        squared_sum[point_mask],
        coefficient_count,
    )


def _bspline_fit_context(weight_flat: Tensor, geometry, stable_accumulation: bool):
    """Precompute the value-independent half of the scattered-data update.

    ``omega`` (the normal-equation weight accumulator) depends only on the
    fit weights and the B-spline geometry -- never on the values being fit --
    so it is identical across repeated fits at the same lattice resolution
    with the same weights. Computing it once here instead of on every fit
    removes the single largest redundant cost for iterative callers (an
    unordered/one-hot reduction over every sample and every one of the
    ``4**D`` local support points). It also matters most on MPS, where the
    chunked one-hot reduction used for stability is the most expensive step.
    """
    indices, basis_values, squared_sum, coefficient_count = geometry
    batch_channels = weight_flat.shape[0]
    omega = weight_flat.new_zeros((batch_channels, coefficient_count))
    support_count, sample_count = indices.shape
    if stable_accumulation:
        # Avoid MPS atomic scatter reductions. Chunked one-hot support
        # matrices are reduced in a fixed order and accumulated with matrix
        # products. The chunk-wise ``delta_basis`` factors are geometry-only
        # and are cached for reuse by every fit's solve step.
        max_one_hot_elements = 8_000_000
        sample_chunk = max(1, max_one_hot_elements // (support_count * coefficient_count))
        coefficient_axis = torch.arange(coefficient_count, device=indices.device)
        chunks = []
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            chunk_indices = indices[:, begin:end].transpose(0, 1)
            chunk_basis = basis_values[:, begin:end].transpose(0, 1)
            assignment = (chunk_indices[..., None] == coefficient_axis).to(basis_values.dtype)
            omega_basis = (assignment * chunk_basis.square().unsqueeze(-1)).sum(dim=1)
            delta_basis = (
                assignment * (chunk_basis.pow(3) / squared_sum[begin:end, None]).unsqueeze(-1)
            ).sum(dim=1)
            omega = omega + weight_flat[:, begin:end] @ omega_basis
            chunks.append((begin, end, delta_basis))
        return {"mode": "stable", "chunks": chunks, "omega": omega}
    else:
        # A single vectorized scatter per statistic replaces 4**D kernel
        # launches while preserving the sparse ITK update formula. Sample
        # chunking bounds peak memory for large (e.g. shrink_factor=1 3-D)
        # volumes instead of materializing one (batch*support*sample) tensor.
        max_scatter_elements = 32_000_000
        sample_chunk = max(1, max_scatter_elements // (batch_channels * support_count))
        cubed_over_squared_sum = basis_values.pow(3) / squared_sum[None]
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            expanded_indices = indices[:, begin:end].reshape(-1)[None].expand(batch_channels, -1)
            omega_values = weight_flat[:, None, begin:end] * basis_values[:, begin:end].square()[None]
            omega.scatter_add_(1, expanded_indices, omega_values.reshape(batch_channels, -1))
        return {
            "mode": "vectorized",
            "omega": omega,
            "indices": indices,
            "cubed_over_squared_sum": cubed_over_squared_sum,
            "sample_chunk": sample_chunk,
        }


def _bspline_fit_solve(residual_flat: Tensor, weight_flat: Tensor, context: dict, eps: float) -> Tensor:
    """Per-call scattered-data update given a resolution-fixed context."""
    batch_channels = residual_flat.shape[0]
    omega = context["omega"]
    delta = residual_flat.new_zeros(omega.shape)
    if context["mode"] == "stable":
        for begin, end, delta_basis in context["chunks"]:
            delta = delta + (residual_flat[:, begin:end] * weight_flat[:, begin:end]) @ delta_basis
    else:
        indices = context["indices"]
        cubed_over_squared_sum = context["cubed_over_squared_sum"]
        sample_chunk = context["sample_chunk"]
        sample_count = residual_flat.shape[1]
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            expanded_indices = indices[:, begin:end].reshape(-1)[None].expand(batch_channels, -1)
            delta_values = (
                residual_flat[:, None, begin:end]
                * weight_flat[:, None, begin:end]
                * cubed_over_squared_sum[None, :, begin:end]
            )
            delta.scatter_add_(1, expanded_indices, delta_values.reshape(batch_channels, -1))
    return torch.where(omega > eps, delta / omega.clamp_min(eps), torch.zeros_like(delta))


def _refine_bspline_coefficients_1d(coefficients: Tensor, dim: int) -> Tensor:
    """Exact dyadic refinement of a uniform cubic B-spline lattice along one
    axis: ``K`` control points become ``2*K - 3``, representing the
    identical piecewise-cubic function at a finer control-point resolution
    (uniform cubic knot insertion / Lane-Riesenfeld subdivision). Derived
    from ITK's Cox-de-Boor shape functions
    (``itkCoxDeBoorBSplineKernelFunction``) and refinement matrix
    (``itkBSplineControlPointImageFilter::RefineControlPointLattice``), and
    independently verified against this package's own already-validated
    ``cubic_bspline_basis`` to floating-point-level agreement (~4e-10 max
    difference). No special boundary handling is required for an open
    (non-periodic) axis: every index referenced below stays in bounds for
    any ``K >= 4``, the minimum lattice size this package allows.
    """
    k_old = coefficients.shape[dim]
    if k_old < 4:
        raise ValueError("cubic B-spline refinement requires at least four control points")
    even = 0.5 * (coefficients.narrow(dim, 0, k_old - 1) + coefficients.narrow(dim, 1, k_old - 1))
    odd = (
        coefficients.narrow(dim, 0, k_old - 2)
        + 6.0 * coefficients.narrow(dim, 1, k_old - 2)
        + coefficients.narrow(dim, 2, k_old - 2)
    ) / 8.0
    interleaved = torch.stack([even.narrow(dim, 0, k_old - 2), odd], dim=dim + 1).flatten(dim, dim + 1)
    return torch.cat([interleaved, even.narrow(dim, k_old - 2, 1)], dim=dim)


def refine_bspline_coefficients(coefficients: Tensor) -> Tensor:
    """Refine every spatial axis of a coefficient lattice (``K -> 2*K - 3``
    per axis), reproducing the identical dense field at the finer
    resolution -- see ``_refine_bspline_coefficients_1d``. This is what
    lets a running coefficient lattice be carried, refined, and continued
    across a multi-resolution fit exactly as ITK's own
    ``m_LogBiasFieldControlPointLattice``/``m_PsiLattice`` are (see
    ``n4_bias_field_correction`` and ``fit_bspline_object_to_scattered_data``
    for two different callers of this same primitive).
    """
    refined = coefficients
    for dim in range(2, coefficients.ndim):
        refined = _refine_bspline_coefficients_1d(refined, dim)
    return refined


def fit_bspline_coefficients(
    values: Tensor,
    domain: ImageDomain,
    lattice_size: Sequence[int],
    weights: Optional[Tensor] = None,
    *,
    stable_accumulation: Optional[bool] = None,
    eps: float = 1e-6,
) -> Tensor:
    """One-shot scattered-data cubic B-spline coefficient fit on a regular
    dense grid -- the complementary "inverse" of
    :func:`synthesize_bspline_velocity` (dense field -> coefficients,
    instead of coefficients -> dense field). This is ITK's
    ``BSplineScatteredDataPointSetToImageFilter`` single-level (non-
    iterative) local least-squares approximation:

    .. code-block:: text

        omega += weight * B**2
        delta  += value * weight * B**3 / sum(B**2)
        coefficient = delta / omega        (0 where omega <= eps)

    ``values`` has shape ``(N, C, *domain.torch_size)``: the dense samples
    to approximate. ``lattice_size`` is the number of control points per
    axis in ITK ``(x, y[, z])`` order (at least 4 per axis -- the same
    convention as every coefficient tensor in this package). ``weights``
    defaults to a uniform weight of 1 and otherwise must be broadcastable to
    ``values``' shape (e.g. a mask or confidence map).

    Returns coefficients with shape ``(N, C, *reversed(lattice_size))``,
    directly usable with :func:`synthesize_bspline_velocity` (optionally at
    a *different* resolution or domain than ``values`` was fit from -- the
    coefficients describe a continuous B-spline surface, independent of any
    particular sampling grid).

    This performs one independent fit; it does not implement ITK's
    multi-level scattered-data refinement, and only open (non-periodic)
    lattices are supported. ``n4_bias_field_correction`` builds on the same
    underlying ``_bspline_fit_geometry``/``_bspline_fit_context``/
    ``_bspline_fit_solve`` primitives directly rather than through this
    wrapper, since its iterative loop needs to reuse the same geometry and
    ``omega`` across many fits at a fixed lattice resolution -- exactly the
    per-call recomputation this convenience wrapper is unsuitable for in a
    tight loop.
    """
    if not isinstance(domain, ImageDomain):
        raise TypeError("domain must be a ImageDomain")
    if values.ndim != domain.dimension + 2:
        raise ValueError(f"expected {domain.dimension + 2}-D values")
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")
    if tuple(values.shape[2:]) != domain.torch_size:
        raise ValueError("values shape does not match domain")
    lattice_itk = tuple(int(k) for k in lattice_size)
    if len(lattice_itk) != domain.dimension:
        raise ValueError(f"lattice_size must have {domain.dimension} values")
    if any(k < 4 for k in lattice_itk):
        raise ValueError("cubic fitting requires at least four control points per dimension")

    if weights is None:
        weight_tensor = values.new_ones(values.shape)
    else:
        weight_tensor = weights.to(dtype=values.dtype, device=values.device).expand_as(values)
    if stable_accumulation is None:
        stable_accumulation = values.device.type == "mps"

    batch_channels = values.shape[0] * values.shape[1]
    values_flat = values.reshape(batch_channels, -1)
    weight_flat = weight_tensor.reshape(batch_channels, -1)

    geometry = _bspline_fit_geometry(domain.torch_size, lattice_itk, values.dtype, values.device, eps)
    context = _bspline_fit_context(weight_flat, geometry, stable_accumulation)
    coefficients = _bspline_fit_solve(values_flat, weight_flat, context, eps)
    return coefficients.reshape((values.shape[0], values.shape[1]) + tuple(reversed(lattice_itk)))


class CubicBSplineSynthesis(nn.Module):
    """``nn.Module`` wrapper around :func:`synthesize_bspline_velocity`."""

    def __init__(self, domain: ImageDomain, *, closed=False, stationary_boundary=False, chunk_size=262144):
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
