"""Greedy symmetric SyN diffeomorphic registration, ported from ``syntx``'s ``syn.py``.

:func:`syn_registration` mirrors ``ants.registration()``'s calling
convention (``ants.ANTsImage`` in, a ``warpedmovout``/``fwdtransforms``/
``invtransforms`` result dictionary out) and, as of Etape 3, its *file*
convention as well: the affine and dense-SyN components are written
separately to disk as classic ANTs transform files (``0GenericAffine.mat``,
``1Warp.nii.gz``, ``1InverseWarp.nii.gz``, under ``outprefix``) and
``fwdtransforms``/``invtransforms`` are path lists built exactly like
``ants.registration()``'s own — so the result of this function is a
drop-in replacement anywhere ``ants.apply_transforms(transformlist=...)`` or
another ANTsX tool expects file-based transforms. See
:mod:`antstorch.ants_transform_io` for the exact file-naming and
list-ordering conventions being matched, and why they matter (the shared
affine file's automatic ``whichtoinvert`` inference in particular).

Per the project's Etape 2 scope ("Coherent avec l'existant"), the affine
initialization stage does *not* port ``syntx``'s separate ``robust_affine``
module. Instead it reuses/extends the lightweight native affine solver
already built for ``bspline_flows``
(:func:`antstorch.bspline_flows.affine_registration.affine_registration`,
itself built on :class:`antstorch.syn.core.affine.HierarchicalAffine`). The
dense SyN stage below is a genuine port of the greedy symmetric SyN
algorithm's structure: two half-warps deforming fixed and moving toward a
shared midpoint, an Eulerian (demons-style) compositional update, CFL-bounded
step sizes, antisymmetric (Frechet-mean) projection removing common-mode
drift, fluid (``flow_sigma``) and optional elastic (``total_sigma``)
regularization, in-loop half-warp inverse maintenance, and a final algebraic
forward/inverse construction built by swapping the two maintained half-warp
inverses rather than inverting the composed field numerically.

One deliberate simplification versus the original: rather than the original's
hand-derived image-gradient chain rule (sampling image gradients separately
and multiplying by an analytically-derived similarity-loss gradient, for
raw speed), this port lets PyTorch autograd differentiate the similarity
loss directly with respect to the two half-warps as leaf tensors each
iteration. This is mathematically equivalent for the metrics implemented
here (the same midpoint images, the same loss), simpler to verify, and
consistent with how :func:`antstorch.bspline_flows.registration.bspline_svf_registration`
itself already differentiates through its objective by full backpropagation
rather than a hand-derived gradient. A second, explicitly out-of-scope
simplification: the per-level divergence-retry-with-halved-CFL step described
in the source algorithm is not reproduced; CFL-bounded steps plus fluid
regularization are relied on for stability instead.
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import ants
import torch
import torch.nn.functional as F
from torch import Tensor

from ..ants_transform_io import build_transform_lists, default_outprefix, write_affine_transform
from .bridge import (
    ants_image_metadata,
    ants_image_to_tensor,
    apply_bspline_smoothing_operator,
    displacement_zyx_to_ants_image,
    flip_affine_xyz_to_zyx,
    image_domain_from_metadata,
    metadata_tensors_from_dict,
    tensor_to_ants_image,
)
from .core.grid import (
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    prepare_mid_images_and_gradients_torch,
)
from .core.inverse import update_inverse_field_nd
from .core.jacobian import compute_jacobian_determinant_nd
from .core.losses import local_ncc_loss_nd, mattes_mi_loss_nd
from .core.pipeline import auto_detect_device
from .core.smoothing import (
    apply_dsti_green_operator,
    apply_sobolev_green_operator,
    get_boundary_mask,
    separable_gaussian_filter,
)

_LINEAR_TRANSFORM_TYPES = ("Translation", "Rigid", "Similarity", "Affine")
_SYN_TRANSFORM_TYPES = ("SyN", "SyNOnly")
_SIMILARITY_METRICS = ("mse", "lncc", "cc", "lncc2", "cc2", "mattes", "mi")
_REGULARIZERS = ("gaussian", "sobolev", "dsti", "bspline")


def _level_values(value, levels: int, name: str) -> tuple:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value,) * levels
    values = tuple(value)
    if len(values) != levels:
        raise ValueError(f"{name} must have one value per resolution level")
    return values


def _gaussian_blur_image(image: Tensor, sigma: float) -> Tensor:
    """Isotropic separable Gaussian blur, ``sigma`` in voxel units."""
    if sigma <= 0:
        return image
    dim = image.ndim - 2
    device, dtype = image.device, image.dtype
    radius = max(1, int(3.0 * sigma + 0.5))
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d
    for _ in range(dim - 1):
        kernel = kernel.unsqueeze(-1) * kernel_1d.reshape((1,) * kernel.ndim + (-1,))
    kernel = kernel.reshape((1, 1) + kernel.shape).expand((image.shape[1], 1) + kernel.shape)
    padding = tuple(value for _ in range(dim) for value in (radius, radius))
    padded = F.pad(image, padding, mode="replicate")
    convolution = F.conv2d if dim == 2 else F.conv3d
    return convolution(padded, kernel, groups=image.shape[1])


def _downsample_metadata(meta: Dict[str, tuple], factor: int) -> Dict[str, tuple]:
    if factor == 1:
        return meta
    torch_shape = tuple(max(2, (size - 1) // factor + 1) for size in meta["torch_shape"])
    shape_itk = tuple(reversed(torch_shape))
    extent = tuple((size - 1) * spacing for size, spacing in zip(meta["shape"], meta["spacing"]))
    spacing = tuple(e / max(size - 1, 1) for e, size in zip(extent, shape_itk))
    new_meta = dict(meta)
    new_meta["shape"] = shape_itk
    new_meta["torch_shape"] = torch_shape
    new_meta["spacing"] = spacing
    return new_meta


def _downsample_image(image: Tensor, meta: Dict[str, tuple], factor: int) -> Tuple[Tensor, Dict[str, tuple]]:
    """Gaussian-presmooth (``sigma = log2(factor)`` voxels) then resample, preserving physical extent."""
    if factor == 1:
        return image, meta
    sigma = math.log2(factor)
    blurred = _gaussian_blur_image(image, sigma)
    new_meta = _downsample_metadata(meta, factor)
    mode = "bilinear" if len(new_meta["torch_shape"]) == 2 else "trilinear"
    resized = F.interpolate(blurred, size=new_meta["torch_shape"], mode=mode, align_corners=True)
    return resized, new_meta


def _upsample_field(field: Tensor, target_torch_shape: Tuple[int, ...]) -> Tensor:
    """Purely geometric resample of a channel-last physical-mm field onto a finer grid."""
    if tuple(field.shape[1:-1]) == tuple(target_torch_shape):
        return field
    dim = len(target_torch_shape)
    mode = "bilinear" if dim == 2 else "trilinear"
    field_cf = field.movedim(-1, 1)
    resized_cf = F.interpolate(field_cf, size=target_torch_shape, mode=mode, align_corners=True)
    return resized_cf.movedim(1, -1).contiguous()


def _physical_grid(meta: Dict[str, tuple], device, dtype) -> Tensor:
    return get_physical_grid_torch(
        meta["torch_shape"], meta["spacing"], meta["origin"], meta["direction"], device=device, dtype=dtype
    )


def _similarity_loss(name: str, I: Tensor, J: Tensor, mask: Optional[Tensor], window_size: int, num_bins: int) -> Tensor:
    if name == "mse":
        if mask is None:
            return (I - J).square().mean()
        denom = mask.sum().clamp_min(1e-8)
        return ((I - J).square() * mask).sum() / denom
    if name in ("lncc", "cc"):
        return local_ncc_loss_nd(I, J, mask=mask, window_size=window_size, squared=False)
    if name in ("lncc2", "cc2"):
        return local_ncc_loss_nd(I, J, mask=mask, window_size=window_size, squared=True)
    if name in ("mattes", "mi"):
        return mattes_mi_loss_nd(I, J, mask=mask, num_bins=num_bins)
    raise ValueError(f"Unknown similarity metric: {name!r}")


def _mesh_size_active(mesh_size: Union[int, Sequence[int]]) -> bool:
    """Whether a (possibly per-axis) base-level/level mesh size is "on".

    ``int`` values follow the existing convention throughout this module:
    ``<= 0`` means off (matches ``*_mesh_size_at_base_level=0`` disabling a
    B-spline regularizer term). A sequence is always the resolved, per-axis
    output of :func:`antstorch.bspline_flows.mesh_size_for_spline_distance`
    (or a caller-supplied per-axis mesh), which is never used to represent
    "off" -- an int (typically the untouched default ``0``) is used for that
    instead, so a sequence here is always active.
    """
    if isinstance(mesh_size, int):
        return mesh_size > 0
    return True


def _scale_mesh_size(mesh_size: Union[int, Sequence[int]], scale: int) -> Union[int, Tuple[int, ...]]:
    """Apply the per-pyramid-level base-mesh-size doubling to either an
    isotropic int or a per-axis sequence, preserving whichever shape was
    given (mirrors ``itk::TransformParametersAdaptor``'s own per-level
    control-point-grid doubling, applied per axis when the mesh is
    anisotropic)."""
    if isinstance(mesh_size, int):
        return mesh_size * scale
    return tuple(int(value) * scale for value in mesh_size)


def _mesh_size_control_points(mesh_size: Union[int, Sequence[int]], dim: int, spline_order: int = 3) -> Tuple[int, ...]:
    """``mesh_size -> number_of_control_points`` (``mesh_size + spline_order``
    per axis), broadcasting an isotropic int across ``dim`` axes."""
    if isinstance(mesh_size, int):
        return (mesh_size + spline_order,) * dim
    return tuple(int(value) + spline_order for value in mesh_size)


def _apply_regularizer(
    field: Tensor,
    regularizer: str,
    sigma: float,
    spacing_itk,
    *,
    mesh_size: Union[int, Sequence[int]] = 0,
    domain=None,
    spline_order: int = 3,
    enforce_stationary_boundary: bool = True,
) -> Tensor:
    if regularizer == "bspline":
        # Its own strength knob (mesh_size, ITK control-point count minus
        # spline_order) rather than sigma -- see apply_bspline_smoothing_operator.
        # mesh_size may be an isotropic int or a resolved per-axis sequence
        # (see _mesh_size_active); only the int form can mean "off".
        if not _mesh_size_active(mesh_size):
            return field
        if domain is None:
            raise ValueError("regularizer='bspline' requires domain")
        return apply_bspline_smoothing_operator(
            field, domain, mesh_size,
            spline_order=spline_order,
            enforce_stationary_boundary=enforce_stationary_boundary,
        )
    if sigma <= 0:
        return field
    if regularizer == "gaussian":
        return separable_gaussian_filter(field, sigma, spacing=spacing_itk, sigma_mode="physical")
    if regularizer == "sobolev":
        return apply_sobolev_green_operator(field, fluid_sigma=sigma, spacing=spacing_itk)
    if regularizer == "dsti":
        return apply_dsti_green_operator(field, fluid_sigma=sigma)
    raise ValueError(f"regularizer must be one of {_REGULARIZERS}, got {regularizer!r}")


def _cfl_normalize(grad: Tensor, spacing_t: Tensor, cfl_voxels: float) -> Tensor:
    voxel_grad = grad / spacing_t
    norm = torch.linalg.norm(voxel_grad, dim=-1)
    max_norm = norm.max().clamp_min(1e-12)
    return grad * (cfl_voxels / max_norm)


def _eulerian_update(warp: Tensor, delta: Tensor, X_phys: Tensor, meta_t: Dict[str, Tensor]) -> Tensor:
    """``phi_new = phi_old o (Id - delta) - delta`` (demons-style compositional update)."""
    coords_phys = X_phys - delta
    coords_norm = physical_to_normalized_torch_cached(
        coords_phys, meta_t["shape_t"], meta_t["spacing_t"], meta_t["origin_t"], meta_t["direction_t"]
    )
    sampled = F.grid_sample(warp.movedim(-1, 1), coords_norm, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.movedim(1, -1) - delta


def _compose_fixed_grid(first: Tensor, second: Tensor, X_phys: Tensor, meta_t: Dict[str, Tensor]) -> Tensor:
    """``T_second o T_first``: ``first(x) + second(x + first(x))``, both fields on the same (fixed) grid."""
    coords_phys = X_phys + first
    coords_norm = physical_to_normalized_torch_cached(
        coords_phys, meta_t["shape_t"], meta_t["spacing_t"], meta_t["origin_t"], meta_t["direction_t"]
    )
    sampled = F.grid_sample(second.movedim(-1, 1), coords_norm, mode="bilinear", padding_mode="border", align_corners=True)
    return first + sampled.movedim(1, -1)


def _fit_syn_level(
    I_curr: Tensor,
    J_curr: Tensor,
    fixed_meta: Dict[str, tuple],
    moving_meta: Dict[str, tuple],
    warp_l2r: Tensor,
    warp_r2l: Tensor,
    warp_l2r_inv: Tensor,
    warp_r2l_inv: Tensor,
    M_phys: Tensor,
    t_phys: Tensor,
    *,
    iterations: int,
    similarity: str,
    window_size: int,
    num_bins: int,
    flow_sigma: float,
    total_sigma: float,
    regularizer: str,
    update_field_mesh_size_at_base_level: Union[int, Sequence[int]],
    total_field_mesh_size_at_base_level: Union[int, Sequence[int]],
    bspline_enforce_stationary_boundary: bool,
    grad_step: float,
    shrink_factor: int,
    antisymmetric: bool,
    inverse_method: str,
    in_loop_inverse_steps: int,
    verbose: bool,
    level_index: int,
    num_levels: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list]:
    device, dtype = I_curr.device, I_curr.dtype
    fixed_meta_t = metadata_tensors_from_dict(fixed_meta, device, dtype)
    moving_meta_t = metadata_tensors_from_dict(moving_meta, device, dtype)
    X_phys = _physical_grid(fixed_meta, device, dtype)
    boundary_mask = get_boundary_mask(fixed_meta["torch_shape"], device, dtype)
    level_cfl_voxels = grad_step * math.sqrt(float(shrink_factor))

    # ITK's BSplineSyN doubles the update/total-field control-point mesh
    # (like any TransformParametersAdaptor) from the coarsest pyramid level
    # (level_index=0) to each successively finer one -- the B-spline analogue
    # of a fixed sigma being meaningful at every resolution. mesh_size=0
    # always means "off" (matches ANTs' totalFieldMeshSizeAtBaseLevel=0
    # default disabling total-field smoothing).
    bspline_domain = None
    update_mesh_size_level = 0
    total_mesh_size_level = 0
    if regularizer == "bspline":
        bspline_domain = image_domain_from_metadata(fixed_meta)
        level_scale = 2 ** level_index
        if _mesh_size_active(update_field_mesh_size_at_base_level):
            update_mesh_size_level = _scale_mesh_size(update_field_mesh_size_at_base_level, level_scale)
        if _mesh_size_active(total_field_mesh_size_at_base_level):
            total_mesh_size_level = _scale_mesh_size(total_field_mesh_size_at_base_level, level_scale)

    history = []
    for iteration in range(iterations):
        warp_l2r_leaf = warp_l2r.detach().requires_grad_(True)
        warp_r2l_leaf = warp_r2l.detach().requires_grad_(True)

        I_mid, J_mid, _, _, in_bounds_mask = prepare_mid_images_and_gradients_torch(
            warp_l2r_leaf,
            warp_r2l_leaf,
            warp_l2r_inv,
            warp_r2l_inv,
            I_curr,
            J_curr,
            X_phys,
            fixed_meta_t["shape_t"],
            fixed_meta_t["spacing_t"],
            fixed_meta_t["origin_t"],
            fixed_meta_t["direction_t"],
            moving_meta_t["shape_t"],
            moving_meta_t["spacing_t"],
            moving_meta_t["origin_t"],
            moving_meta_t["direction_t"],
            fixed_meta["spacing"],
            moving_meta["spacing"],
            M_phys,
            t_phys,
            None,
        )
        loss = _similarity_loss(similarity, I_mid, J_mid, in_bounds_mask, window_size, num_bins)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SyN loss at resolution level {level_index + 1}")
        loss.backward()
        grad_l = warp_l2r_leaf.grad
        grad_r = warp_r2l_leaf.grad
        if grad_l is None or grad_r is None or not torch.isfinite(grad_l).all() or not torch.isfinite(grad_r).all():
            raise FloatingPointError(f"non-finite SyN half-warp gradient at resolution level {level_index + 1}")

        grad_l = _apply_regularizer(
            grad_l * boundary_mask, regularizer, flow_sigma, fixed_meta["spacing"],
            mesh_size=update_mesh_size_level, domain=bspline_domain,
            enforce_stationary_boundary=bspline_enforce_stationary_boundary,
        )
        grad_r = _apply_regularizer(
            grad_r * boundary_mask, regularizer, flow_sigma, fixed_meta["spacing"],
            mesh_size=update_mesh_size_level, domain=bspline_domain,
            enforce_stationary_boundary=bspline_enforce_stationary_boundary,
        )

        delta_l = _cfl_normalize(grad_l, fixed_meta_t["spacing_t"], level_cfl_voxels)
        delta_r = _cfl_normalize(grad_r, fixed_meta_t["spacing_t"], level_cfl_voxels)
        if antisymmetric:
            common_mode = delta_l + delta_r
            delta_l = delta_l - 0.5 * common_mode
            delta_r = delta_r - 0.5 * common_mode

        warp_l2r = _eulerian_update(warp_l2r_leaf.detach(), delta_l, X_phys, fixed_meta_t)
        warp_r2l = _eulerian_update(warp_r2l_leaf.detach(), delta_r, X_phys, fixed_meta_t)

        if regularizer == "bspline":
            if _mesh_size_active(total_mesh_size_level):
                warp_l2r = apply_bspline_smoothing_operator(
                    warp_l2r, bspline_domain, total_mesh_size_level,
                    enforce_stationary_boundary=bspline_enforce_stationary_boundary,
                )
                warp_r2l = apply_bspline_smoothing_operator(
                    warp_r2l, bspline_domain, total_mesh_size_level,
                    enforce_stationary_boundary=bspline_enforce_stationary_boundary,
                )
        elif total_sigma > 0:
            warp_l2r = separable_gaussian_filter(warp_l2r, total_sigma, spacing=fixed_meta["spacing"], sigma_mode="physical")
            warp_r2l = separable_gaussian_filter(warp_r2l, total_sigma, spacing=fixed_meta["spacing"], sigma_mode="physical")

        warp_l2r_inv = update_inverse_field_nd(
            warp_l2r, warp_l2r_inv, steps=in_loop_inverse_steps, method=inverse_method,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )
        warp_r2l_inv = update_inverse_field_nd(
            warp_r2l, warp_r2l_inv, steps=in_loop_inverse_steps, method=inverse_method,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )

        current = float(loss.detach().item())
        history.append(current)
        if verbose:
            print(
                f"  [SyN level {level_index + 1}/{num_levels}] iteration {iteration + 1:04d}/{iterations}: "
                f"loss={current:.8g}"
            )

    return warp_l2r.detach(), warp_r2l.detach(), warp_l2r_inv.detach(), warp_r2l_inv.detach(), history


def _fit_affine_from_ants(
    fixed,
    moving,
    device,
    dtype,
    *,
    transform_type: str,
    similarity: str,
    neighborhood_radius,
    shrink_factors,
    smoothing_sigmas,
    iterations,
    learning_rate,
    multi_start: bool,
    center_of_mass_init: bool,
    verbose: bool,
):
    """Fit the affine initialization with :func:`antstorch.bspline_flows.affine_registration.affine_registration`."""
    # Imported lazily (not at module scope): antstorch.bspline_flows itself
    # imports antstorch.syn.core (for HierarchicalAffine), so an eager
    # module-level import here would create a circular import whenever
    # `import antstorch.bspline_flows` runs before `antstorch.syn.syn` has
    # finished loading (e.g. `from antstorch.bspline_flows import ...` as
    # the very first antstorch import in a test module).
    from antstorch.bspline_flows import ImageDomain
    from antstorch.bspline_flows import affine_registration as _bspline_affine_registration

    fixed_meta = ants_image_metadata(fixed)
    moving_meta = ants_image_metadata(moving)
    fixed_domain = ImageDomain(fixed_meta["shape"], fixed_meta["spacing"], fixed_meta["origin"], fixed_meta["direction"])
    moving_domain = ImageDomain(moving_meta["shape"], moving_meta["spacing"], moving_meta["origin"], moving_meta["direction"])
    fixed_tensor = ants_image_to_tensor(fixed, device, dtype)
    moving_tensor = ants_image_to_tensor(moving, device, dtype)
    result = _bspline_affine_registration(
        fixed_tensor,
        moving_tensor,
        fixed_domain,
        moving_domain,
        transform_type=transform_type,
        similarity=similarity,
        neighborhood_radius=neighborhood_radius,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        iterations=iterations,
        learning_rate=learning_rate,
        multi_start=multi_start,
        center_of_mass_init=center_of_mass_init,
        padding_mode="border",
        verbose=verbose,
    )
    # bspline_flows.affine_registration fits one affine per batch item; a
    # single ants.ANTsImage pair is always batch size 1.
    matrix_xyz = result["matrix"][0]
    translation_xyz = result["translation"][0]
    return matrix_xyz, translation_xyz, result


def syn_registration(
    fixed,
    moving,
    *,
    type_of_transform: str = "SyN",
    initial_affine: Optional[Tuple[Tensor, Tensor]] = None,
    affine_transform_type: str = "Affine",
    affine_similarity: str = "mse",
    affine_neighborhood_radius: Union[int, Sequence[int]] = 2,
    affine_shrink_factors: Sequence[int] = (4, 2, 1),
    affine_smoothing_sigmas: Optional[Union[float, Sequence[float]]] = (2.0, 1.0, 0.0),
    affine_iterations: Union[int, Sequence[int]] = (100, 100, 50),
    affine_learning_rate: Union[float, Sequence[float]] = (0.05, 0.03, 0.02),
    affine_multi_start: bool = True,
    affine_center_of_mass_init: bool = True,
    syn_metric: str = "lncc",
    neighborhood_radius: int = 2,
    num_bins: int = 32,
    levels: Sequence[int] = (4, 2, 1),
    reg_iterations: Sequence[int] = (100, 100, 50),
    grad_step: float = 0.5,
    flow_sigma: float = 3.0,
    total_sigma: float = 0.0,
    regularizer: str = "gaussian",
    update_field_mesh_size_at_base_level: int = 1,
    total_field_mesh_size_at_base_level: int = 0,
    update_field_spline_distance: Optional[Union[float, Sequence[float]]] = None,
    total_field_spline_distance: Optional[Union[float, Sequence[float]]] = None,
    bspline_enforce_stationary_boundary: bool = True,
    inverse_method: str = "anderson",
    in_loop_inverse_steps: int = 6,
    antisymmetric: bool = True,
    padding_mode: str = "zeros",
    outprefix: str = "",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Dict[str, object]:
    """Register ``moving`` onto ``fixed`` with (optionally affine-initialized) greedy symmetric SyN.

    Mirrors ``ants.registration()``'s calling convention: ``fixed``/``moving``
    are ``ants.ANTsImage`` objects, and the result dictionary uses the same
    ``warpedmovout``/``fwdtransforms``/``invtransforms`` naming. As in
    ``ants.registration()``, the affine and dense-SyN components are always
    kept and written separately: ``fwdtransforms``/``invtransforms`` are
    lists of file paths under ``outprefix`` (default: a fresh temp-file
    prefix, exactly like ``ants.registration()``'s own default), not a
    single composed in-memory field. ``outprefix + "0GenericAffine.mat"``
    holds the fitted/supplied affine (omitted when none applies, e.g.
    ``'SyNOnly'`` with no ``initial_affine``), and
    ``outprefix + "1Warp.nii.gz"`` / ``"1InverseWarp.nii.gz"`` hold the pure
    dense-SyN deformation (omitted for the linear-only transform types).
    This makes the result directly usable with
    ``ants.apply_transforms(transformlist=...)`` or any other ANTsX tool
    that expects file-based transforms — see
    :mod:`antstorch.ants_transform_io` for the exact conventions matched.

    ``type_of_transform`` selects the transform model:

    - ``'Translation'``, ``'Rigid'``, ``'Similarity'``, ``'Affine'``: fit
      only a linear transform (via
      :func:`antstorch.bspline_flows.affine_registration.affine_registration`)
      and return; no dense SyN stage runs.
    - ``'SyN'`` (default): fit an affine initialization first (unless
      ``initial_affine`` is supplied), then run the dense greedy symmetric
      SyN stage on top of it.
    - ``'SyNOnly'``: run the dense SyN stage directly at identity (or at
      ``initial_affine`` if supplied), skipping the internal affine fit.

    ``initial_affine``, if given, is a ``(matrix, translation)`` pair in ITK
    ``(x, y[, z])`` physical order — the exact convention returned by
    :func:`antstorch.bspline_flows.affine_registration.affine_registration`
    (``matrix``/``translation`` of shape ``(dim, dim)``/``(dim,)``, unbatched,
    since a single ``ants.ANTsImage`` pair is always batch size 1) — and is
    used verbatim as the fixed affine initialization, skipping the internal
    affine-fit stage entirely.

    The dense SyN stage follows a greedy symmetric formulation: two
    half-warps deform ``fixed`` and ``moving`` toward a shared midpoint each
    iteration (loss and gradients evaluated there), with an Eulerian
    (demons-style) compositional update, a CFL-bounded step (``grad_step``,
    the maximum per-voxel step in voxel units, scaled per level by
    ``sqrt(shrink_factor)``), antisymmetric (Frechet-mean) projection
    removing common-mode drift between the two half-warps (set
    ``antisymmetric=False`` to disable), fluid regularization of the raw
    per-iteration update (``flow_sigma``, via ``regularizer`` -
    ``'gaussian'``, ``'sobolev'``, ``'dsti'``, or ``'bspline'``) and optional
    elastic regularization of the composed field (``total_sigma`` for
    ``'gaussian'``/``'sobolev'``/``'dsti'``, plain Gaussian, disabled by
    default).

    ``regularizer='bspline'`` is the ANTs/ITK ``BSplineSyN`` regularizer: a
    single-level cubic B-spline fit (via
    :func:`antstorch.bspline_flows.bspline_scattered_data.fit_bspline_displacement_field`,
    the port of ``itkDisplacementFieldToBSplineImageFilter``) smooths the
    update field each iteration and, if enabled, the composed field
    periodically -- the direct analogue of ITK's
    ``itkBSplineSmoothingOnUpdateDisplacementFieldTransform``, as used by
    ``itk::BSplineSyNImageRegistrationMethod`` and ``antsRegistration``'s
    ``BSplineSyN[gradientStep, updateFieldMeshSizeAtBaseLevel,
    totalFieldMeshSizeAtBaseLevel, splineOrder]`` transform spec. In this
    mode, ``flow_sigma``/``total_sigma`` are ignored in favor of
    ``update_field_mesh_size_at_base_level`` (default ``1``, ITK's own
    class default of 4 control points minus ``spline_order=3``; cubic is
    the only spline order :mod:`antstorch.bspline_flows` supports) and
    ``total_field_mesh_size_at_base_level`` (default ``0`` = off, matching
    ANTs' own default). Both mesh sizes are given *at the base
    (coarsest) pyramid level* and doubled at each successively finer level
    -- the same doubling ``itk::TransformParametersAdaptor`` applies to the
    B-spline transform's control-point grid across ``ImageRegistrationMethodv4``
    resolution levels -- so a single base value stays a comparably strong
    regularizer at every ``levels`` entry, the way one ``flow_sigma``/
    ``total_sigma`` does for the other three regularizers.

    ``update_field_spline_distance``/``total_field_spline_distance``, when
    given, compute the corresponding ``*_mesh_size_at_base_level`` from a
    physical knot spacing against ``fixed``'s full native-resolution domain
    instead of an explicit integer -- ANTs' own "spline distance" convention
    (see :func:`antstorch.bspline_flows.mesh_size_for_spline_distance`), the
    exact un-padded formula ``itk::ants::RegistrationHelper::
    CalculateMeshSizeForSpecifiedKnotSpacing`` uses (``Examples/
    itkantsRegistrationHelper.cxx``, used by ``antsRegistration`` whenever a
    single scalar is given for ``BSplineSyN``'s ``updateFieldMeshSizeAtBaseLevel``/
    ``totalFieldMeshSizeAtBaseLevel``), and the same un-padded formula
    ``n4_bias_field_correction``'s scalar ``spline_param`` already uses.
    Real ANTs' own conversion is per-axis even for a single scalar distance
    (the physical field of view need not be isotropic), so the resolved
    mesh size -- and every subsequent per-level doubling -- can be
    anisotropic when a spline distance is used, unlike the plain-integer
    ``*_mesh_size_at_base_level`` parameters, which stay isotropic. Each is
    mutually exclusive with a non-default value of its corresponding
    ``*_mesh_size_at_base_level`` integer. As with ``bspline_svf_registration``,
    no image padding is performed -- the mesh size is an approximation of
    the requested spacing, exactly matching real ``antsRegistration``'s own
    (non-padding) registration-side behavior; this deliberately differs from
    real ANTs' ``N4BiasFieldCorrection``, which does pad.
    ``bspline_enforce_stationary_boundary`` (default ``True``, ITK's own
    default) keeps the field at zero on the domain's outermost voxel layer.
    Both half-warp inverses are numerically maintained
    in-loop (``inverse_method``, ``in_loop_inverse_steps``, warm-started
    across iterations and pyramid levels) and are what the final
    ``fwdtransforms``/``invtransforms`` are built from algebraically (a
    half-warp-inverse swap), not from any generic post-hoc field inversion.

    ``levels``/``reg_iterations`` define the multi-resolution pyramid
    (coarse to fine, e.g. ``levels=(4, 2, 1)``); unlike
    :func:`antstorch.bspline_flows.registration.bspline_svf_registration`'s
    ``shrink_factors``, this pyramid need not be a strict dyadic halving.

    Returns
    -------
    dict
        ``warpedmovout`` (``ants.ANTsImage``, ``moving`` warped onto the
        fixed grid), ``fwdtransforms``/``invtransforms`` (list of file
        paths, matching ``ants.registration()``'s own convention exactly —
        see above), ``jacobian`` (``ants.ANTsImage``, the physical Jacobian
        determinant of the total forward deformation, still computed
        in-memory since it has no ``ants.registration()`` equivalent),
        ``loss_history``/``level_loss_history`` (the SyN stage's
        per-iteration losses), ``affine_matrix``/``affine_translation`` (the
        same affine as ``0GenericAffine.mat``, also kept in-memory in ITK
        order for convenience; ``None`` for ``'SyNOnly'`` with no
        ``initial_affine``), ``affine_loss_history``/``affine_level_loss_history``
        (``None`` if no internal affine fit ran), and ``provenance`` (a dict
        recording the configuration actually used, including the resolved
        ``outprefix``).
    """
    if fixed.dimension != moving.dimension:
        raise ValueError("fixed and moving must have the same dimension")
    if fixed.dimension not in (2, 3):
        raise ValueError("syn_registration supports only 2-D or 3-D images")
    if type_of_transform not in _LINEAR_TRANSFORM_TYPES + _SYN_TRANSFORM_TYPES:
        raise ValueError(f"type_of_transform must be one of {_LINEAR_TRANSFORM_TYPES + _SYN_TRANSFORM_TYPES}")
    if syn_metric not in _SIMILARITY_METRICS:
        raise ValueError(f"syn_metric must be one of {_SIMILARITY_METRICS}")
    if regularizer not in _REGULARIZERS:
        raise ValueError(f"regularizer must be one of {_REGULARIZERS}")
    if update_field_mesh_size_at_base_level < 0:
        raise ValueError("update_field_mesh_size_at_base_level must be >= 0")
    if total_field_mesh_size_at_base_level < 0:
        raise ValueError("total_field_mesh_size_at_base_level must be >= 0")
    if update_field_spline_distance is not None and update_field_mesh_size_at_base_level != 1:
        raise ValueError(
            "update_field_spline_distance cannot be combined with a non-default "
            "update_field_mesh_size_at_base_level"
        )
    if total_field_spline_distance is not None and total_field_mesh_size_at_base_level != 0:
        raise ValueError(
            "total_field_spline_distance cannot be combined with a non-default "
            "total_field_mesh_size_at_base_level"
        )
    if (
        regularizer == "bspline"
        and update_field_mesh_size_at_base_level == 0
        and total_field_mesh_size_at_base_level == 0
        and update_field_spline_distance is None
        and total_field_spline_distance is None
    ):
        raise ValueError(
            "regularizer='bspline' requires update_field_mesh_size_at_base_level > 0, "
            "total_field_mesh_size_at_base_level > 0, update_field_spline_distance, "
            "and/or total_field_spline_distance"
        )
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    if len(levels) != len(reg_iterations):
        raise ValueError("levels and reg_iterations must have the same length")

    if device is None:
        auto_device = auto_detect_device(requested_device=None)
        if auto_device == "mps":
            # bspline_flows.affine_registration()'s differentiable warp
            # (used by every type_of_transform except 'SyNOnly' with no
            # internal affine fit) backpropagates through F.grid_sample,
            # whose backward pass is not implemented on MPS as of the
            # PyTorch versions this has been tested against
            # (NotImplementedError: aten::grid_sampler_2d_backward /
            # grid_sampler_3d_backward). Auto-detection therefore skips mps
            # and falls back to cpu; pass device='mps' explicitly to opt in
            # anyway (e.g. once PyTorch adds the missing op, or for
            # type_of_transform='SyNOnly' calls that never differentiate
            # through the affine warp).
            auto_device = "cpu"
        resolved_device = torch.device(auto_device)
    else:
        resolved_device = torch.device(device)
    resolved_outprefix = outprefix if outprefix else default_outprefix()
    dimension = fixed.dimension

    # --- Affine stage -----------------------------------------------------
    affine_result = None
    if initial_affine is not None:
        matrix_xyz, translation_xyz = initial_affine
        if tuple(matrix_xyz.shape) != (dimension, dimension) or tuple(translation_xyz.shape) != (dimension,):
            raise ValueError(f"initial_affine must be an unbatched ({dimension}, {dimension})/({dimension},) pair")
        matrix_xyz = matrix_xyz.to(device=resolved_device, dtype=dtype)
        translation_xyz = translation_xyz.to(device=resolved_device, dtype=dtype)
    elif type_of_transform in _LINEAR_TRANSFORM_TYPES or type_of_transform == "SyN":
        affine_type = type_of_transform if type_of_transform in _LINEAR_TRANSFORM_TYPES else affine_transform_type
        matrix_xyz, translation_xyz, affine_result = _fit_affine_from_ants(
            fixed, moving, resolved_device, dtype,
            transform_type=affine_type,
            similarity=affine_similarity,
            neighborhood_radius=affine_neighborhood_radius,
            shrink_factors=affine_shrink_factors,
            smoothing_sigmas=affine_smoothing_sigmas,
            iterations=affine_iterations,
            learning_rate=affine_learning_rate,
            multi_start=affine_multi_start,
            center_of_mass_init=affine_center_of_mass_init,
            verbose=verbose,
        )
    else:  # SyNOnly, no initial_affine supplied: identity.
        matrix_xyz = torch.eye(dimension, device=resolved_device, dtype=dtype)
        translation_xyz = torch.zeros(dimension, device=resolved_device, dtype=dtype)

    if type_of_transform in _LINEAR_TRANSFORM_TYPES:
        # Whether the affine came from an internal fit or was supplied
        # directly via initial_affine, (re)compute its dense field and
        # warped output directly from (matrix_xyz, translation_xyz) so both
        # paths return the same shape of result.
        from antstorch.bspline_flows import ImageDomain, affine_displacement_field, warp_image

        fixed_meta = ants_image_metadata(fixed)
        moving_meta = ants_image_metadata(moving)
        fixed_domain = ImageDomain(fixed_meta["shape"], fixed_meta["spacing"], fixed_meta["origin"], fixed_meta["direction"])
        moving_domain = ImageDomain(
            moving_meta["shape"], moving_meta["spacing"], moving_meta["origin"], moving_meta["direction"]
        )
        fixed_tensor = ants_image_to_tensor(fixed, resolved_device, dtype, normalize=False)
        moving_tensor = ants_image_to_tensor(moving, resolved_device, dtype, normalize=False)
        # Only the forward field is needed here: fwdtransforms/invtransforms
        # are now the written 0GenericAffine.mat path (both directions, via
        # ants.apply_transforms()'s whichtoinvert), not an in-memory inverse
        # field -- so no separate inverse displacement is computed.
        fwd_field_xyz = affine_displacement_field(matrix_xyz, translation_xyz, fixed_domain, fixed_tensor)
        warpedmovout_tensor = warp_image(
            moving_tensor, fwd_field_xyz, fixed_domain, moving_domain, padding_mode=padding_mode
        )

        matrix_out = matrix_xyz.detach().cpu()
        translation_out = translation_xyz.detach().cpu()
        warpedmovout = tensor_to_ants_image(warpedmovout_tensor, fixed)

        affine_path = f"{resolved_outprefix}0GenericAffine.mat"
        write_affine_transform(matrix_out, translation_out, dimension, affine_path)
        fwdtransforms, invtransforms = build_transform_lists(
            affine_path=affine_path, warp_path=None, inverse_warp_path=None
        )
        return {
            "warpedmovout": warpedmovout,
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
            "jacobian": None,
            "loss_history": None,
            "level_loss_history": None,
            "affine_matrix": matrix_out,
            "affine_translation": translation_out,
            "affine_loss_history": affine_result["loss_history"] if affine_result is not None else None,
            "affine_level_loss_history": affine_result["level_loss_history"] if affine_result is not None else None,
            "provenance": {
                "type_of_transform": type_of_transform,
                "affine_transform_type": type_of_transform,
                "device": str(resolved_device),
                "outprefix": resolved_outprefix,
            },
        }

    # --- Dense SyN stage ----------------------------------------------------
    M_phys, t_phys = flip_affine_xyz_to_zyx(matrix_xyz, translation_xyz)
    M_phys = M_phys.to(device=resolved_device, dtype=dtype)
    t_phys = t_phys.to(device=resolved_device, dtype=dtype)

    fixed_meta_full = ants_image_metadata(fixed)
    moving_meta_full = ants_image_metadata(moving)
    I_full = ants_image_to_tensor(fixed, resolved_device, dtype)
    J_full = ants_image_to_tensor(moving, resolved_device, dtype)

    # Resolve any spline-distance parameter into the base-level mesh size it
    # replaces, once, against fixed's FULL native-resolution domain -- matching
    # real ANTs' own CalculateMeshSizeForSpecifiedKnotSpacing, which is always
    # computed against the un-shrunk fixedImage before any pyramid downsampling
    # (see update_field_spline_distance's docstring above). The existing
    # per-level doubling (_fit_syn_level, and the verbose print below) then
    # applies to this resolved value exactly as it would to an explicit
    # update_field_mesh_size_at_base_level/total_field_mesh_size_at_base_level.
    resolved_update_field_mesh_size_at_base_level = update_field_mesh_size_at_base_level
    resolved_total_field_mesh_size_at_base_level = total_field_mesh_size_at_base_level
    if regularizer == "bspline" and (update_field_spline_distance is not None or total_field_spline_distance is not None):
        from antstorch.bspline_flows import mesh_size_for_spline_distance

        fixed_domain_full = image_domain_from_metadata(fixed_meta_full)
        if update_field_spline_distance is not None:
            resolved_update_field_mesh_size_at_base_level = mesh_size_for_spline_distance(
                fixed_domain_full, update_field_spline_distance
            )
        if total_field_spline_distance is not None:
            resolved_total_field_mesh_size_at_base_level = mesh_size_for_spline_distance(
                fixed_domain_full, total_field_spline_distance
            )

    window_size = 2 * int(neighborhood_radius) + 1
    num_levels = len(levels)

    warp_l2r = warp_r2l = warp_l2r_inv = warp_r2l_inv = None
    level_loss_history = []
    for level_index, (factor, iteration_count) in enumerate(zip(levels, reg_iterations)):
        I_level, fixed_meta_level = _downsample_image(I_full, fixed_meta_full, factor)
        J_level, moving_meta_level = _downsample_image(J_full, moving_meta_full, factor)
        if warp_l2r is None:
            zeros = torch.zeros(
                (1,) + fixed_meta_level["torch_shape"] + (dimension,), device=resolved_device, dtype=dtype
            )
            warp_l2r, warp_r2l = zeros.clone(), zeros.clone()
            warp_l2r_inv, warp_r2l_inv = zeros.clone(), zeros.clone()
        else:
            warp_l2r = _upsample_field(warp_l2r, fixed_meta_level["torch_shape"])
            warp_r2l = _upsample_field(warp_r2l, fixed_meta_level["torch_shape"])
            warp_l2r_inv = _upsample_field(warp_l2r_inv, fixed_meta_level["torch_shape"])
            warp_r2l_inv = _upsample_field(warp_r2l_inv, fixed_meta_level["torch_shape"])

        if verbose:
            message = (
                f"SyN resolution level {level_index + 1}/{num_levels}: shrink_factor={factor}, "
                f"fixed_size={fixed_meta_level['torch_shape']}, moving_size={moving_meta_level['torch_shape']}, "
            )
            if regularizer == "bspline":
                # Mirrors bspline_svf_registration()'s own verbose control-point
                # reporting (antstorch/bspline_flows/registration.py) -- same
                # mesh_size -> control_points = mesh_size + spline_order
                # relationship (cubic spline order, this package's only
                # supported order), doubled from the base level exactly as
                # _fit_syn_level itself doubles it before regularizing (see
                # the comment there). Uses the *resolved* base mesh size --
                # the raw *_mesh_size_at_base_level int, or the per-axis
                # mesh size when a *_spline_distance was given instead (see
                # its resolution above, before this loop).
                level_scale = 2**level_index
                dim = len(fixed_meta_level["torch_shape"])
                if _mesh_size_active(resolved_update_field_mesh_size_at_base_level):
                    update_mesh_size_level = _scale_mesh_size(resolved_update_field_mesh_size_at_base_level, level_scale)
                    update_control_points = _mesh_size_control_points(update_mesh_size_level, dim)
                    message += f"control_points={update_control_points}, "
                if _mesh_size_active(resolved_total_field_mesh_size_at_base_level):
                    total_mesh_size_level = _scale_mesh_size(resolved_total_field_mesh_size_at_base_level, level_scale)
                    total_field_control_points = _mesh_size_control_points(total_mesh_size_level, dim)
                    message += f"total_field_control_points={total_field_control_points}, "
            message += f"iterations={iteration_count}"
            print(message)

        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, history = _fit_syn_level(
            I_level, J_level, fixed_meta_level, moving_meta_level,
            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, M_phys, t_phys,
            iterations=iteration_count,
            similarity=syn_metric,
            window_size=window_size,
            num_bins=num_bins,
            flow_sigma=flow_sigma,
            total_sigma=total_sigma,
            regularizer=regularizer,
            update_field_mesh_size_at_base_level=resolved_update_field_mesh_size_at_base_level,
            total_field_mesh_size_at_base_level=resolved_total_field_mesh_size_at_base_level,
            bspline_enforce_stationary_boundary=bspline_enforce_stationary_boundary,
            grad_step=grad_step,
            shrink_factor=factor,
            antisymmetric=antisymmetric,
            inverse_method=inverse_method,
            in_loop_inverse_steps=in_loop_inverse_steps,
            verbose=verbose,
            level_index=level_index,
            num_levels=num_levels,
        )
        level_loss_history.append(history)

    loss_history = [value for level in level_loss_history for value in level]

    # --- Final algebraic forward/inverse construction (half-warp swap) ------
    fixed_meta_t_full = metadata_tensors_from_dict(fixed_meta_full, resolved_device, dtype)
    moving_meta_t_full = metadata_tensors_from_dict(moving_meta_full, resolved_device, dtype)
    X_phys_full = _physical_grid(fixed_meta_full, resolved_device, dtype)

    syn_forward = _compose_fixed_grid(warp_l2r_inv, warp_r2l, X_phys_full, fixed_meta_t_full)
    syn_inverse = _compose_fixed_grid(warp_r2l_inv, warp_l2r, X_phys_full, fixed_meta_t_full)

    # Total forward: apply the pure-SyN deformation, then the (fixed)
    # affine, matching prepare_mid_images_and_gradients_torch's own
    # `y_phys = phi_r2l_phys @ M_phys.t() + t_phys` convention.
    phi = X_phys_full + syn_forward
    moved_phys = torch.matmul(phi, M_phys.transpose(-1, -2)) + t_phys
    total_forward = moved_phys - X_phys_full

    inverse_matrix = torch.linalg.inv(M_phys)
    inverse_translation = -torch.matmul(inverse_matrix, t_phys)
    affine_inverse_displacement = (
        torch.matmul(X_phys_full, inverse_matrix.transpose(-1, -2)) + inverse_translation - X_phys_full
    )
    total_inverse = _compose_fixed_grid(affine_inverse_displacement, syn_inverse, X_phys_full, fixed_meta_t_full)

    warpedmovout = F.grid_sample(
        ants_image_to_tensor(moving, resolved_device, dtype, normalize=False),
        physical_to_normalized_torch_cached(
            X_phys_full + total_forward,
            moving_meta_t_full["shape_t"], moving_meta_t_full["spacing_t"],
            moving_meta_t_full["origin_t"], moving_meta_t_full["direction_t"],
        ),
        mode="bilinear", padding_mode=padding_mode, align_corners=True,
    )
    warpedfixout = F.grid_sample(
        ants_image_to_tensor(fixed, resolved_device, dtype, normalize=False),
        physical_to_normalized_torch_cached(
            X_phys_full + total_inverse,
            fixed_meta_t_full["shape_t"], fixed_meta_t_full["spacing_t"],
            fixed_meta_t_full["origin_t"], fixed_meta_t_full["direction_t"],
        ),
        mode="bilinear", padding_mode=padding_mode, align_corners=True,
    )

    # total_forward is already a genuine physical-mm displacement field
    # (not a normalized-grid one), so the lower-level, spacing-aware
    # Jacobian is the correct entry point here (compute_physical_jacobian_determinant
    # instead expects a normalized-space field unless its `is_physical`
    # tensor attribute is set).
    jacobian = compute_jacobian_determinant_nd(
        total_forward, physical_spacing=fixed_meta_t_full["spacing_t"]
    ).unsqueeze(0)

    # --- Write the affine and pure-SyN pieces separately (Etape 3) ----------
    # syn_forward/syn_inverse (built above via the half-warp swap) are the
    # *pure* dense deformation, computed before the fixed affine was folded
    # in to build total_forward/total_inverse -- exactly the piece
    # ants.registration() itself would write as "1Warp.nii.gz"/
    # "1InverseWarp.nii.gz", with no recomputation or decomposition needed.
    warp_path = f"{resolved_outprefix}1Warp.nii.gz"
    inverse_warp_path = f"{resolved_outprefix}1InverseWarp.nii.gz"
    ants.image_write(displacement_zyx_to_ants_image(syn_forward, fixed), warp_path)
    ants.image_write(displacement_zyx_to_ants_image(syn_inverse, fixed), inverse_warp_path)

    has_affine = initial_affine is not None or affine_result is not None
    affine_path = None
    if has_affine:
        affine_path = f"{resolved_outprefix}0GenericAffine.mat"
        write_affine_transform(matrix_xyz.detach().cpu(), translation_xyz.detach().cpu(), dimension, affine_path)

    fwdtransforms, invtransforms = build_transform_lists(
        affine_path=affine_path, warp_path=warp_path, inverse_warp_path=inverse_warp_path
    )

    result = {
        "warpedmovout": tensor_to_ants_image(warpedmovout, fixed),
        "warpedfixout": tensor_to_ants_image(warpedfixout, fixed),
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        "jacobian": tensor_to_ants_image(jacobian, fixed),
        "loss_history": loss_history,
        "level_loss_history": level_loss_history,
        "affine_matrix": matrix_xyz.detach().cpu(),
        "affine_translation": translation_xyz.detach().cpu(),
        "affine_loss_history": affine_result["loss_history"] if affine_result is not None else None,
        "affine_level_loss_history": affine_result["level_loss_history"] if affine_result is not None else None,
        "provenance": {
            "type_of_transform": type_of_transform,
            "syn_metric": syn_metric,
            "levels": tuple(levels),
            "reg_iterations": tuple(reg_iterations),
            "grad_step": grad_step,
            "flow_sigma": flow_sigma,
            "total_sigma": total_sigma,
            "regularizer": regularizer,
            "update_field_mesh_size_at_base_level": resolved_update_field_mesh_size_at_base_level,
            "total_field_mesh_size_at_base_level": resolved_total_field_mesh_size_at_base_level,
            "update_field_spline_distance": update_field_spline_distance,
            "total_field_spline_distance": total_field_spline_distance,
            "bspline_enforce_stationary_boundary": bspline_enforce_stationary_boundary,
            "antisymmetric": antisymmetric,
            "inverse_method": inverse_method,
            "affine_fit": affine_result is not None,
            "device": str(resolved_device),
            "outprefix": resolved_outprefix,
        },
    }
    return result
