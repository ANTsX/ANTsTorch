"""Differentiable, tensor-only scattered-data B-spline fitting.

PyTorch analogues of ANTsPy's ``fit_bspline_object_to_scattered_data`` and
``fit_bspline_displacement_field`` (``ants/registration/`` -- themselves
wrappers around ITK's ``itkBSplineScatteredDataPointSetToImageFilter`` and
``itkDisplacementFieldToBSplineImageFilter``). Argument names and semantics
follow ANTsPy closely; the differences are called out in each function's
docstring.

Unlike :func:`~.bspline_synthesis.fit_bspline_coefficients` (a single
dense-grid least-squares fit), this module implements ITK's actual
multi-level scattered-data algorithm: at each of ``number_of_fitting_levels``
levels, fit the current residual at every sample -> add it to a running
coefficient lattice -> exactly refine that lattice for the next, finer
level (see :func:`~.bspline_synthesis.refine_bspline_coefficients`). This is
the same accumulate/refine pattern
:func:`~.n4_bias_field_correction.n4_bias_field_correction` uses internally,
generalized here from N4's regular shrunk-image grid to arbitrary scattered
points with independent parametric locations.
"""

from itertools import product
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .bspline_domain import ImageDomain
from .bspline_synthesis import (
    _bspline_fit_context,
    _bspline_fit_geometry,
    _bspline_fit_geometry_points,
    _bspline_fit_solve,
    _as_bools,
    _concat_bspline_fit_geometries,
    _select_bspline_fit_geometry,
    cubic_bspline_basis,
    refine_bspline_coefficients,
    synthesize_bspline_velocity,
)


def _parametric_to_u(
    parametric_coordinate: Tensor,
    origin: float,
    spacing: float,
    domain_size: int,
    lattice_size: int,
    closed: bool,
) -> Tensor:
    """Map a physical/parametric coordinate to this package's internal
    B-spline u-parametrization, ``[0, spans]``, using the same "domain
    edge-to-edge spans the full parametric range" convention as every other
    u-mapping in this package (see ``_bspline_fit_geometry`` and
    ``synthesize_bspline_velocity``). For a coordinate that lands exactly on
    a dense, regularly-spaced grid of ``domain_size`` points running from
    ``origin`` to ``origin + spacing*(domain_size-1)``, this reduces exactly
    to ``_bspline_fit_geometry``'s grid-index formula -- so a scattered fit
    over a domain's own grid points agrees with the dense-grid fit path.
    """
    spans = float(lattice_size if closed else lattice_size - 3)
    normalized = (parametric_coordinate - origin) / spacing
    u = normalized * (spans / float(domain_size - 1))
    if closed:
        return u
    upper = torch.nextafter(u.new_tensor(spans), u.new_tensor(float("-inf")))
    return u.clamp(0.0, upper)


def _domain_boundary_mask(domain: ImageDomain, device) -> Tensor:
    """Flattened torch-order mask for the outermost voxel layer."""
    mask = torch.zeros(domain.torch_size, dtype=torch.bool, device=device)
    for axis in range(domain.dimension):
        lower = [slice(None)] * domain.dimension
        upper = [slice(None)] * domain.dimension
        lower[axis] = 0
        upper[axis] = -1
        mask[tuple(lower)] = True
        mask[tuple(upper)] = True
    return mask.reshape(-1)


def _evaluate_bspline_at_points(coefficients: Tensor, geometry) -> Tensor:
    """Evaluate a coefficient lattice at the points described by
    ``geometry`` (from ``_bspline_fit_geometry_points``) -- the
    scattered-point analogue of ``synthesize_bspline_velocity``'s
    per-chunk gather. Reuses the exact ``(indices, basis_values)`` support
    tables a fit step already built, so evaluating the current fit's
    reconstruction at the fitted points (to form the next level's residual)
    costs no extra geometry work.
    """
    indices, basis_values, _, _ = geometry
    batch_channels = coefficients.shape[0] * coefficients.shape[1]
    flat_coefficients = coefficients.reshape(batch_channels, -1)
    support_count, point_count = indices.shape
    gathered = flat_coefficients.index_select(1, indices.reshape(-1)).reshape(
        batch_channels, support_count, point_count
    )
    values = (gathered * basis_values[None]).sum(dim=1)
    return values.reshape(coefficients.shape[0], coefficients.shape[1], point_count)


def _mesh_size_to_lattice(mesh_size, dimension: int, spline_order: int, closed_axes) -> Tuple[int, ...]:
    if spline_order != 3:
        raise NotImplementedError("only cubic B-splines (spline_order=3) are currently supported")
    if isinstance(mesh_size, int):
        mesh_sizes = (mesh_size,) * dimension
    else:
        mesh_sizes = tuple(int(v) for v in mesh_size)
        if len(mesh_sizes) != dimension:
            raise ValueError(f"mesh_size must have {dimension} values")
    lattice_itk = tuple(m + spline_order for m in mesh_sizes)
    for size, closed in zip(lattice_itk, closed_axes):
        if closed and size < 2 * spline_order + 1:
            raise ValueError("closed parametric dimensions require mesh_size > spline_order")
        if size < 4:
            raise ValueError("mesh_size must be at least 1 per dimension")
    return lattice_itk


def fit_bspline_object_to_scattered_data(
    scattered_data,
    parametric_data,
    parametric_domain_origin: Sequence[float],
    parametric_domain_spacing: Sequence[float],
    parametric_domain_size: Sequence[int],
    is_parametric_dimension_closed: Optional[Union[bool, Sequence[bool]]] = None,
    data_weights=None,
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_order: int = 3,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
    stable_accumulation: Optional[bool] = None,
    return_coefficients: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Fit a smooth cubic B-spline object to scattered data -- a PyTorch
    analogue of ANTsPy's ``fit_bspline_object_to_scattered_data`` (a wrapper
    for ``itkBSplineScatteredDataPointSetToImageFilter``), entirely with
    differentiable tensor ops (no ITK, no NumPy in the fitting path).

    Argument names and semantics mirror ANTsPy so existing ANTsPy call
    sites translate directly: ``scattered_data`` is ``(P, data_dimension)``,
    ``parametric_data`` is ``(P, parametric_dimension)`` giving each row's
    location in the B-spline object's parametric domain (defined by
    ``parametric_domain_origin``/``_spacing``/``_size``, ITK order).

    Differences from ANTsPy, by design:

    * Only 2-D or 3-D parametric domains are supported (this package's
      ``ImageDomain``/``synthesize_bspline_velocity`` do not represent
      1-D curves or 4-D fields), and only ``spline_order=3`` (cubic,
      matching the rest of this package).
    * The return value is always a dense tensor sampling the fitted object
      over the full parametric domain, shape ``(1, data_dimension,
      *reversed(parametric_domain_size))`` -- this package's usual
      ``(N, C, *spatial)`` convention with ``N=1`` -- rather than a raw
      curve array or an ANTsImage. Pass ``return_coefficients=True`` to
      additionally get the raw coefficient lattice, e.g. for feeding into
      ``synthesize_bspline_velocity`` at a different resolution/domain.
      Concretely, this means the result is the ANTsPy/ITK output
      *transposed*: ``ants.fit_bspline_object_to_scattered_data(...)
      .numpy()`` keeps ITK's direct axis order (``(size_x, size_y[,
      size_z])``), while this function reverses it (``(N, C, size_y,
      size_x)`` in 2-D). ``result[0, c].numpy() == ants_result.numpy()[..., c].T``
      (2-D; a full axis reversal, not a literal ``.T``, in 3-D) -- the same
      relationship already used to compare ``n4_bias_field_correction``
      against ANTsPy (see ``test_n4_bias_field_correction.py``). Numerically
      the fit agrees with ANTsPy to float precision once this axis
      reversal is accounted for.

    Multi-level fitting matches ITK exactly: at each of
    ``number_of_fitting_levels`` levels (starting from ``mesh_size`` control
    points and doubling each level via ``refine_bspline_coefficients``), the
    residual between ``scattered_data`` and the current accumulated fit is
    computed at every point and added to the running coefficient lattice --
    the same accumulate/refine pattern used by ``n4_bias_field_correction``.
    """
    scattered_data = torch.as_tensor(scattered_data, dtype=dtype, device=device)
    parametric_data = torch.as_tensor(parametric_data, dtype=scattered_data.dtype, device=scattered_data.device)
    device = scattered_data.device
    dtype = scattered_data.dtype

    if scattered_data.ndim != 2:
        raise ValueError("scattered_data must be 2-D (points, data_dimension)")
    if parametric_data.ndim != 2:
        raise ValueError("parametric_data must be 2-D (points, parametric_dimension)")
    if scattered_data.shape[0] != parametric_data.shape[0]:
        raise ValueError("scattered_data and parametric_data must have the same number of points")

    parametric_dimension = parametric_data.shape[1]
    data_dimension = scattered_data.shape[1]
    point_count = scattered_data.shape[0]
    if parametric_dimension not in (2, 3):
        raise NotImplementedError(
            "only 2-D or 3-D parametric domains are currently supported (no 1-D curve fitting yet)"
        )
    if number_of_fitting_levels < 1:
        raise ValueError("number_of_fitting_levels must be at least 1")

    origin = tuple(float(v) for v in parametric_domain_origin)
    spacing = tuple(float(v) for v in parametric_domain_spacing)
    domain_size = tuple(int(v) for v in parametric_domain_size)
    if len(origin) != parametric_dimension or len(spacing) != parametric_dimension or len(domain_size) != parametric_dimension:
        raise ValueError("parametric_domain_origin/spacing/size must each have parametric_dimension values")

    closed_axes = _as_bools(
        False if is_parametric_dimension_closed is None else is_parametric_dimension_closed, parametric_dimension
    )
    lattice_itk = _mesh_size_to_lattice(mesh_size, parametric_dimension, spline_order, closed_axes)

    if data_weights is None:
        weight_points = scattered_data.new_ones(point_count)
    else:
        weight_points = torch.as_tensor(data_weights, dtype=dtype, device=device).reshape(-1)
        if weight_points.shape[0] != point_count:
            raise ValueError("data_weights must have one value per point")
    if stable_accumulation is None:
        stable_accumulation = device.type == "mps"

    values_flat = scattered_data.t().reshape(data_dimension, point_count)
    weight_flat = weight_points.unsqueeze(0).expand(data_dimension, -1)

    accumulated_coefficients = torch.zeros(
        (1, data_dimension) + tuple(reversed(lattice_itk)), dtype=dtype, device=device
    )
    current_lattice = lattice_itk
    for level in range(number_of_fitting_levels):
        # The scattered points' parametric *locations* never change across
        # levels, but their u-coordinates do: ``_parametric_to_u``'s mapping
        # depends on ``spans = lattice_size - 3``, which grows every
        # refined level, so u must be recomputed per level, not cached.
        parametric_u = tuple(
            _parametric_to_u(
                parametric_data[:, d], origin[d], spacing[d], domain_size[d], current_lattice[d], closed_axes[d]
            )
            for d in range(parametric_dimension)
        )
        geometry = _bspline_fit_geometry_points(parametric_u, current_lattice, closed_axes, eps)
        reconstruction_flat = _evaluate_bspline_at_points(accumulated_coefficients, geometry).reshape(
            data_dimension, point_count
        )
        residual_flat = values_flat - reconstruction_flat
        fit_context = _bspline_fit_context(weight_flat, geometry, stable_accumulation)
        update_flat = _bspline_fit_solve(residual_flat, weight_flat, fit_context, eps)
        accumulated_coefficients = accumulated_coefficients + update_flat.reshape(
            (1, data_dimension) + tuple(reversed(current_lattice))
        )
        if level + 1 < number_of_fitting_levels:
            accumulated_coefficients = refine_bspline_coefficients(accumulated_coefficients)
            current_lattice = tuple(2 * v - 3 for v in current_lattice)

    domain = ImageDomain(size=domain_size, spacing=spacing, origin=origin)
    dense = synthesize_bspline_velocity(accumulated_coefficients, domain, closed=closed_axes)
    if return_coefficients:
        return dense, accumulated_coefficients
    return dense


def _bspline_fit_dense_grid_chunked(
    values_flat: Tensor,
    weight_flat: Tensor,
    domain: ImageDomain,
    lattice_itk: Sequence[int],
    eps: float,
    chunk_size: int,
) -> Tensor:
    """Memory-bounded equivalent of ``_bspline_fit_geometry`` +
    ``_bspline_fit_context`` (vectorized mode) + ``_bspline_fit_solve``, for
    fitting a REGULAR DENSE GRID in one shot (no cross-call geometry reuse).

    ``_bspline_fit_geometry`` materializes one ``(4**D, N)`` index tensor and
    one same-shape basis-weight tensor for the *whole* dense grid (``N`` =
    every voxel) before any accumulation happens -- fine for the modest grids
    ``n4_bias_field_correction``'s iterative loop fits (which deliberately
    keeps that geometry resident across many fits at the same resolution, see
    ``_bspline_fit_context``'s docstring -- a real win there), but at a full
    native-resolution 3-D volume (e.g. 256x256x160 = ~10.5M voxels) with
    ``4**3 = 64`` local support points per sample, those two tensors alone
    are >8 GB, and ``_bspline_fit_context``'s vectorized path adds a further
    same-size ``cubed_over_squared_sum`` intermediate on top -- comfortably
    enough to exhaust a shared GPU's free memory (reported by a user running
    ``antstorch.syn.syn_registration(regularizer="bspline")`` -- the
    ``BSplineSyN`` regularizer, via :func:`fit_bspline_displacement_field`
    called through ``apply_bspline_smoothing_operator`` -- at full resolution
    with a fine update-field mesh: ``CUDA out of memory. Tried to allocate
    7.50 GiB``).

    This function is used ONLY for the one-shot, single-level, dense-grid
    case (see ``fit_bspline_displacement_field``'s call site below), which
    never benefits from geometry reuse in the first place (there is exactly
    one fit per call) -- so fusing geometry construction, the normal-equation
    accumulation, and the solve into one pass, processing ``chunk_size``
    samples at a time and immediately reducing each chunk into the
    ``coefficient_count``-sized ``omega``/``delta`` accumulators, never
    materializes anything larger than ``(4**D, chunk_size)``. Peak memory is
    then governed by ``chunk_size`` instead of the sample count -- the same
    principle :func:`~.bspline_synthesis.synthesize_bspline_velocity`
    already applies on the forward (coefficients -> dense field) path.
    Produces bit-for-bit (up to floating-point summation order) the same
    result as the unchunked path for any ``chunk_size`` -- verified directly
    against it before this was wired in.

    ``values_flat``/``weight_flat`` : ``(batch_channels, N)``, ``N`` =
    ``prod(domain.size)``, already in the dense grid's flattened (row-major,
    ITK x fastest) sample order -- exactly what ``fit_bspline_displacement_field``
    already builds for its unchunked path.
    """
    dtype, device = values_flat.dtype, values_flat.device
    dimension = len(lattice_itk)
    batch_channels = values_flat.shape[0]
    point_count = 1
    for size in domain.size:
        point_count *= size
    strides, stride = [], 1
    for size in lattice_itk:
        strides.append(stride)
        stride *= size
    coefficient_count = stride

    omega = values_flat.new_zeros((batch_channels, coefficient_count))
    delta = values_flat.new_zeros((batch_channels, coefficient_count))

    for begin in range(0, point_count, chunk_size):
        end = min(begin + chunk_size, point_count)
        remaining = torch.arange(begin, end, device=device)
        dense_indices = []
        for size in domain.size:  # ITK x fastest, matching _bspline_fit_geometry's flatten order
            dense_indices.append(remaining.remainder(size))
            remaining = torch.div(remaining, size, rounding_mode="floor")

        neighbors, basis = [], []
        for coordinate, dense_size, lattice_size in zip(dense_indices, domain.size, lattice_itk):
            spans = lattice_size - 3
            u = coordinate.to(dtype) * (float(spans) / float(dense_size - 1))
            # Same exact-bound clamp as _bspline_fit_geometry/synthesize_bspline_velocity.
            u = u.clamp_max(torch.nextafter(u.new_tensor(float(spans)), u.new_tensor(float("-inf"))))
            base = torch.floor(u).to(torch.long)
            local = torch.arange(4, device=device)
            index = base[..., None] + local
            neighbors.append(index)
            basis.append(cubic_bspline_basis(u[..., None] - index.to(u.dtype) + 1.0))

        chunk_len = end - begin
        squared_sum = values_flat.new_zeros((chunk_len,))
        support_indices, support_basis = [], []
        for support in product(range(4), repeat=dimension):
            index = sum(neighbors[d][..., support[d]] * strides[d] for d in range(dimension))
            value = torch.ones(chunk_len, dtype=dtype, device=device)
            for d in range(dimension):
                value = value * basis[d][..., support[d]]
            support_indices.append(index)
            support_basis.append(value)
            squared_sum = squared_sum + value.square()
        squared_sum = squared_sum.clamp_min(eps)

        values_chunk = values_flat[:, begin:end]
        weight_chunk = weight_flat[:, begin:end]
        for index, value in zip(support_indices, support_basis):
            expanded_index = index[None].expand(batch_channels, -1)
            omega.scatter_add_(1, expanded_index, weight_chunk * value[None].square())
            delta.scatter_add_(
                1, expanded_index, values_chunk * weight_chunk * (value.pow(3) / squared_sum)[None]
            )

    return torch.where(omega > eps, delta / omega.clamp_min(eps), torch.zeros_like(delta))


def fit_bspline_displacement_field(
    displacement_field: Optional[Tensor] = None,
    displacement_weight_image: Optional[Tensor] = None,
    displacement_origins=None,
    displacements=None,
    displacement_weights=None,
    domain: Optional[ImageDomain] = None,
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_order: int = 3,
    enforce_stationary_boundary: bool = True,
    *,
    eps: float = 1e-6,
    stable_accumulation: Optional[bool] = None,
    return_coefficients: bool = False,
    chunk_size: int = 262_144,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Fit and smooth a displacement field with cubic B-splines, from a
    dense field, scattered displacement points, or both together -- a
    PyTorch analogue of ANTsPy's ``fit_bspline_displacement_field`` (a
    wrapper for ``itkDisplacementFieldToBSplineImageFilter``, itself built
    on the same scattered-data filter as
    :func:`fit_bspline_object_to_scattered_data`).

    ``chunk_size`` only affects the common one-shot case -- a dense
    ``displacement_field``, no scattered points, ``number_of_fitting_levels
    == 1`` (exactly what ``apply_bspline_smoothing_operator``'s ``BSplineSyN``
    regularizer uses, once per iteration) -- where it bounds peak memory via
    :func:`_bspline_fit_dense_grid_chunked` instead of materializing a
    ``(4**D, N)`` geometry tensor for the whole dense grid at once; see that
    function's docstring. It is ignored for multi-level fits and fits
    involving scattered points, which still use the original unchunked
    geometry/context/solve path (unchanged).

    ``displacement_field`` (if given) is dense, shape ``(1, D, *spatial)``
    matching ``domain`` (``D`` = spatial dimension); every voxel becomes one
    weighted observation (weight 1, or ``displacement_weight_image``'s value
    at that voxel). ``displacement_origins``/``displacements`` (if given)
    are ``(P, D)`` scattered points with independent physical locations
    (not necessarily on the field's grid), each individually weighted by
    ``displacement_weights``. Both sources, when given together, are fit
    jointly in one pass -- their contributions to the underlying B-spline
    normal equations are simply combined (see
    ``_concat_bspline_fit_geometries``) before solving.

    Differences from ANTsPy, by design:

    * ``domain`` (this package's ``ImageDomain``) replaces ANTsPy's
      separate ``origin``/``spacing``/``size``/``direction`` arguments.
    * ``estimate_inverse`` is not implemented: ITK's inverse-field
      estimation is a materially different, iterative algorithm, out of
      scope here.
    * ``rasterize_points`` is not applicable -- this implementation is
      already fully vectorized rather than looping over points.
    * ``enforce_stationary_boundary`` follows ITK by fitting zero-valued
      observations on the domain's outermost voxel layer with weight
      ``1e10``. For a dense input field these replace its boundary samples;
      when fitting scattered points only they are added as synthetic
      observations. Consequently the boundary constraint affects the whole
      coefficient lattice when the B-spline basis has wide support; it is
      not a post-fit output mask.

    Returns a dense tensor, shape ``(1, D, *domain.torch_size)``. Pass
    ``return_coefficients=True`` to additionally get the raw coefficient
    lattice.
    """
    if spline_order != 3:
        raise NotImplementedError("only cubic B-splines (spline_order=3) are currently supported")
    if displacement_field is None and (displacement_origins is None or displacements is None):
        raise ValueError(
            "Either displacement_field or scattered points (displacement_origins + "
            "displacements) must be specified."
        )

    if displacement_field is not None:
        if displacement_field.ndim not in (4, 5):
            raise ValueError("displacement_field must be (1, D, *spatial)")
        if displacement_field.shape[0] != 1:
            raise ValueError("fit_bspline_displacement_field fits one field at a time (batch size 1)")
        dimension = displacement_field.ndim - 2
        if displacement_field.shape[1] != dimension:
            raise ValueError("displacement_field must have one channel per spatial dimension")
        domain = domain or ImageDomain(tuple(reversed(displacement_field.shape[2:])))
        if domain.torch_size != tuple(displacement_field.shape[2:]):
            raise ValueError("displacement_field shape does not match domain")
        device, dtype = displacement_field.device, displacement_field.dtype
    else:
        if domain is None:
            raise ValueError("domain must be specified when fitting from scattered points only")
        dimension = domain.dimension
        origins_tensor = torch.as_tensor(displacement_origins)
        device, dtype = origins_tensor.device, (
            origins_tensor.dtype if origins_tensor.is_floating_point() else torch.float32
        )

    closed_axes = _as_bools(False, dimension)
    lattice_itk = _mesh_size_to_lattice(mesh_size, dimension, spline_order, closed_axes)
    if stable_accumulation is None:
        stable_accumulation = device.type == "mps"

    if displacement_field is not None and displacement_origins is None and number_of_fitting_levels == 1:
        # The common one-shot case (see this function's docstring and
        # _bspline_fit_dense_grid_chunked's): a dense field, no scattered
        # points, a single fitting level -- exactly what
        # apply_bspline_smoothing_operator's BSplineSyN regularizer calls,
        # once per SyN iteration. Bypasses _bspline_fit_geometry entirely
        # (never materializes a (4**D, N) tensor for the whole grid) rather
        # than reusing it the way the multi-level/scattered-point path below
        # does, since there is exactly one fit here -- geometry reuse across
        # calls, the thing that path is built around, buys nothing.
        if displacement_weight_image is None:
            weight_field = displacement_field.new_ones((1, 1) + displacement_field.shape[2:])
        else:
            if tuple(displacement_weight_image.shape[2:]) != tuple(displacement_field.shape[2:]):
                raise ValueError("displacement_weight_image shape does not match displacement_field")
            weight_field = displacement_weight_image.to(dtype=dtype, device=device)
        field_values = displacement_field.reshape(dimension, -1)
        field_weights = weight_field.reshape(1, -1).expand(dimension, field_values.shape[1])
        if enforce_stationary_boundary:
            boundary_mask = _domain_boundary_mask(domain, device)
            field_values = field_values.masked_fill(boundary_mask.unsqueeze(0), 0.0)
            field_weights = field_weights.masked_fill(boundary_mask.unsqueeze(0), 1.0e10)
        coefficients_flat = _bspline_fit_dense_grid_chunked(
            field_values, field_weights, domain, lattice_itk, eps, chunk_size
        )
        accumulated_coefficients = coefficients_flat.reshape((1, dimension) + tuple(reversed(lattice_itk)))
        dense = synthesize_bspline_velocity(accumulated_coefficients, domain, chunk_size=chunk_size)
        if return_coefficients:
            return dense, accumulated_coefficients
        return dense

    geometries = []
    values_parts = []
    weight_parts = []
    boundary_mask = _domain_boundary_mask(domain, device) if enforce_stationary_boundary else None
    boundary_weight = 1.0e10

    if displacement_field is not None:
        if displacement_weight_image is None:
            weight_field = displacement_field.new_ones((1, 1) + displacement_field.shape[2:])
        else:
            if tuple(displacement_weight_image.shape[2:]) != tuple(displacement_field.shape[2:]):
                raise ValueError("displacement_weight_image shape does not match displacement_field")
            weight_field = displacement_weight_image.to(dtype=dtype, device=device)
        grid_geometry = _bspline_fit_geometry(domain.torch_size, lattice_itk, dtype, device, eps)
        grid_point_count = grid_geometry[0].shape[1]
        geometries.append(grid_geometry)
        field_values = displacement_field.reshape(dimension, -1)
        field_weights = weight_field.reshape(1, -1).expand(dimension, grid_point_count)
        if boundary_mask is not None:
            field_values = field_values.masked_fill(boundary_mask.unsqueeze(0), 0.0)
            field_weights = field_weights.masked_fill(boundary_mask.unsqueeze(0), boundary_weight)
        values_parts.append(field_values)
        weight_parts.append(field_weights)

    if displacement_field is None and boundary_mask is not None:
        boundary_geometry = _select_bspline_fit_geometry(
            _bspline_fit_geometry(domain.torch_size, lattice_itk, dtype, device, eps), boundary_mask
        )
        boundary_point_count = int(boundary_mask.sum().item())
        geometries.append(boundary_geometry)
        values_parts.append(torch.zeros(dimension, boundary_point_count, dtype=dtype, device=device))
        weight_parts.append(
            torch.full((dimension, boundary_point_count), boundary_weight, dtype=dtype, device=device)
        )

    if displacement_origins is not None:
        origins = torch.as_tensor(displacement_origins, dtype=dtype, device=device)
        values = torch.as_tensor(displacements, dtype=dtype, device=device)
        if origins.ndim != 2 or origins.shape[1] != dimension:
            raise ValueError(f"displacement_origins must be (P, {dimension})")
        if values.shape != origins.shape:
            raise ValueError("displacements must have the same shape as displacement_origins")
        point_count = origins.shape[0]
        if displacement_weights is None:
            point_weights = origins.new_ones(point_count)
        else:
            point_weights = torch.as_tensor(displacement_weights, dtype=dtype, device=device).reshape(-1)
            if point_weights.shape[0] != point_count:
                raise ValueError("displacement_weights must have one value per point")
        # origins are physical points in the *original* (unshrunk, ITK x,y[,z]
        # ordered) domain; domain.size/spacing/origin describe that same
        # physical domain, so this is exactly the same origin/spacing mapping
        # ``fit_bspline_object_to_scattered_data`` uses.
        parametric_u = tuple(
            _parametric_to_u(
                origins[:, d], domain.origin[d], domain.spacing[d], domain.size[d], lattice_itk[d], False
            )
            for d in range(dimension)
        )
        point_geometry = _bspline_fit_geometry_points(parametric_u, lattice_itk, closed_axes, eps)
        geometries.append(point_geometry)
        values_parts.append(values.t())
        weight_parts.append(point_weights.unsqueeze(0).expand(dimension, -1))

    geometry = geometries[0] if len(geometries) == 1 else _concat_bspline_fit_geometries(geometries)
    values_flat = torch.cat(values_parts, dim=1)
    weight_flat = torch.cat(weight_parts, dim=1)

    accumulated_coefficients = torch.zeros(
        (1, dimension) + tuple(reversed(lattice_itk)), dtype=dtype, device=device
    )
    current_lattice = lattice_itk
    current_geometry = geometry
    for level in range(number_of_fitting_levels):
        if level > 0:
            # Later levels need fresh geometry at the refined resolution;
            # only the first level's geometry was built above.
            parts = []
            if displacement_field is not None:
                parts.append(_bspline_fit_geometry(domain.torch_size, current_lattice, dtype, device, eps))
            elif boundary_mask is not None:
                parts.append(
                    _select_bspline_fit_geometry(
                        _bspline_fit_geometry(domain.torch_size, current_lattice, dtype, device, eps),
                        boundary_mask,
                    )
                )
            if displacement_origins is not None:
                u_current = tuple(
                    _parametric_to_u(
                        origins[:, d],
                        domain.origin[d],
                        domain.spacing[d],
                        domain.size[d],
                        current_lattice[d],
                        False,
                    )
                    for d in range(dimension)
                )
                parts.append(_bspline_fit_geometry_points(u_current, current_lattice, closed_axes, eps))
            current_geometry = parts[0] if len(parts) == 1 else _concat_bspline_fit_geometries(parts)

        reconstruction_flat = _evaluate_from_mixed_geometry(
            accumulated_coefficients, current_geometry, displacement_field is not None, domain, dimension
        )
        residual_flat = values_flat - reconstruction_flat
        fit_context = _bspline_fit_context(weight_flat, current_geometry, stable_accumulation)
        update_flat = _bspline_fit_solve(residual_flat, weight_flat, fit_context, eps)
        accumulated_coefficients = accumulated_coefficients + update_flat.reshape(
            (1, dimension) + tuple(reversed(current_lattice))
        )
        if level + 1 < number_of_fitting_levels:
            accumulated_coefficients = refine_bspline_coefficients(accumulated_coefficients)
            current_lattice = tuple(2 * v - 3 for v in current_lattice)

    dense = synthesize_bspline_velocity(accumulated_coefficients, domain)
    if return_coefficients:
        return dense, accumulated_coefficients
    return dense


def _evaluate_from_mixed_geometry(coefficients, geometry, has_grid, domain, dimension):
    """``_evaluate_bspline_at_points`` doesn't care whether ``geometry`` came
    from a dense grid, scattered points, or a concatenation of both -- it
    only uses ``indices``/``basis_values``, which are already merged by
    ``_concat_bspline_fit_geometries`` when needed. This thin wrapper exists
    only for a descriptive call site in ``fit_bspline_displacement_field``.
    """
    return _evaluate_bspline_at_points(coefficients, geometry).reshape(dimension, -1)
