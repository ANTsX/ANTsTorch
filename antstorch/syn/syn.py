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
consistent with how :func:`antstorch.bspline_flows.bspline_svf_registration.bspline_svf_registration`
itself already differentiates through its objective by full backpropagation
rather than a hand-derived gradient. A second, explicitly out-of-scope
simplification: the per-level divergence-retry-with-halved-CFL step described
in the source algorithm is not reproduced; CFL-bounded steps plus fluid
regularization are relied on for stability instead.
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..ants_transform_io import build_transform_lists, default_outprefix, write_affine_transform
from ..registration import reg_adam_direction
from .robust_affine import _mps_grid_sample_backward_available
from .core.regularizers import (
    COUPLED_CONTINUUM_REGULARIZERS,
    REGULARIZER_EXTRA_KEYS,
    build_level_regularizer,
)
from .bridge import (
    _resolve_intensity_normalization,
    ants_image_metadata,
    ants_image_to_tensor,
    apply_bspline_smoothing_operator,
    displacement_zyx_to_ants_image,
    flip_affine_xyz_to_zyx,
    image_domain_from_metadata,
    metadata_tensors_from_dict,
    tensor_to_ants_image,
)
from .._torch_compat import grid_sample, grid_sample_for_probe
from .core.grid import (
    grid_sample_nd,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    prepare_mid_images_and_gradients_torch,
)
from .core.inverse import update_inverse_field_nd, update_inverse_field_nd_hybrid_lm
from .core.jacobian import compute_jacobian_determinant_nd
from .core.losses import local_ncc_loss_chunked, local_ncc_loss_nd, mattes_mi_loss_nd, box_cc2_loss_nd, soft_dice_loss_nd
from .core.pipeline import auto_detect_device, mps_grid_sample_3d_available, mps_grid_sample_3d_forward_available
from .core.smoothing import (
    apply_dsti_green_operator,
    apply_sobolev_green_operator,
    get_boundary_mask,
    gaussian_blur_replicate,
    separable_gaussian_filter,
)

_LINEAR_TRANSFORM_TYPES = ("Translation", "Rigid", "Similarity", "Affine")
_CC_METRICS = ("cc", "lncc", "cc2", "lncc2")
_SYN_TRANSFORM_TYPES = ("SyN", "SyNOnly")
_SIMILARITY_METRICS = ("mse", "lncc", "cc", "lncc2", "cc2", "mattes", "mi", "box_cc2", "dice")
# syntx >= 6.43 / 6.44: the coupled continuum operators of core/regularizers.py ('navier' -- the
# canonical default since syntx 6.44 --, 'div_curl', 'solenoidal', ...) are applied to the raw
# half-warp gradient in physical mm, exactly as syntx.syn (build_level_regularizer,
# param_domain='physical_mm', with D_zyx direction conjugation).
_BASE_REGULARIZERS = ("gaussian", "sobolev", "dsti", "bspline")
_CONTINUUM_REGULARIZERS = tuple(sorted(COUPLED_CONTINUUM_REGULARIZERS))
_REGULARIZERS = _BASE_REGULARIZERS + _CONTINUUM_REGULARIZERS
# 'gradient_descent' (default): the existing CFL-bounded plain-gradient
# Eulerian update. 'reg_adam': ported from syntx's greedy.py
# GreedyRegistration.fit() -- per-voxel first/second Adam moments are
# accumulated across iterations (reset at the start of each pyramid level,
# matching syntx's "Reset Adam warm-up step at each scale level"), and the
# bias-corrected Adam step-direction quotient is regularized (via the same
# _apply_regularizer already used below) in place of the raw similarity
# gradient -- everything downstream (CFL bound, antisymmetric projection,
# Eulerian composition, in-loop inversion) is unchanged. See the project
# doc ("écart dsti", §34/§36/§39-40) for why this was added: to test
# whether syntx's own dsti-arm advantage comes from its Adam-momentum
# optimizer rather than from the full time-varying-velocity-field (TVF)
# architecture its benchmark harness actually routes 'dsti' through.
_OPTIMIZERS = ("gradient_descent", "cfl", "reg_adam", "syn_reg_adam")
# 'cfl' is syntx's name for the ITK-style CFL-bounded gradient step ('gradient_descent' here).
_OPTIMIZER_ALIASES = {"cfl": "gradient_descent"}


def _level_values(value, levels: int, name: str) -> tuple:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value,) * levels
    values = tuple(value)
    if len(values) != levels:
        raise ValueError(f"{name} must have one value per resolution level")
    return values


def _gaussian_blur_image(image: Tensor, sigma: float) -> Tensor:
    """Isotropic separable Gaussian blur, ``sigma`` in voxel units (replicate padding)."""
    if sigma <= 0:
        return image
    return gaussian_blur_replicate(image, (sigma,) * (image.ndim - 2))


def _shrunk_torch_shape(torch_shape: Tuple[int, ...], factor: int, syntx_parity: bool = False) -> Tuple[int, ...]:
    """Per-axis spatial shape after shrinking ``torch_shape`` by ``factor``.

    The exact formula ``_downsample_image``/``_fit_syn_level`` actually use to
    build each pyramid level -- factored out so ``_validate_pyramid_levels``
    (below) can predict every level's shape *before* any tensor work runs,
    from metadata alone, without drifting from the real downsampling code.
    """
    if factor == 1:
        return tuple(torch_shape)
    if syntx_parity:
        # syntx: F.interpolate(scale_factor=1/factor) -> floor(size / factor)
        return tuple(max(2, int(math.floor(size * (1.0 / factor)))) for size in torch_shape)
    return tuple(max(2, (size - 1) // factor + 1) for size in torch_shape)


def _validate_pyramid_levels(
    torch_shape: Tuple[int, ...],
    levels: Sequence[int],
    window_size: int,
    syntx_parity: bool = False,
    level_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
) -> None:
    """Reject a SyN resolution pyramid whose coarsest level(s) are too small
    to support the local similarity window, for ANY ``regularizer``/
    ``optimizer`` combination.

    ``_fit_syn_level`` computes the similarity loss (``cc2``/``lncc``/etc.)
    with a local window of ``window_size`` voxels on a side
    (``2 * neighborhood_radius + 1``). When a pyramid level's shrunk shape
    has an axis smaller than that window, the local-similarity computation
    is asked to look at a neighborhood larger than the image itself -- a
    degenerate level that can silently produce near-zero loss/gradient (and
    hence a near-identity or garbage warp) regardless of which regularizer
    or optimizer is in use; the fluid regularizers ('sobolev'/'dsti') make it
    worse still, since 'dsti''s Dirichlet boundary condition forces the
    field to zero at every face of a level that small, but the root problem
    -- a pyramid level too small for the similarity metric -- is shared by
    every ``regularizer``/``optimizer`` pair that reaches this loop (see the
    project doc, "pyramide dégénérée / dsti dice=0.0").

    This caught, concretely: an 8x-shrink level of a 24x28x24 test image
    collapsing to (3, 4, 3) voxels against the default ``window_size=5``
    (``neighborhood_radius=2``) -- ``gaussian``/``bspline`` degraded there
    without raising (dice 0.10-0.31 on that fixture), ``dsti``/``sobolev``
    collapsed to dice 0.0 because of the added Dirichlet-boundary/
    conservative-smoothing passes, but all four were already operating on a
    pyramid level that could not support the chosen similarity window.

    Raises
    ------
    ValueError
        Naming every offending level (1-indexed), its shrink factor, and the
        resulting shape, with a suggestion to drop/soften that level.
    """
    offending = []
    for level_index, factor in enumerate(levels):
        shrunk = (tuple(level_shapes[level_index]) if level_shapes is not None
                  else _shrunk_torch_shape(tuple(torch_shape), int(factor), syntx_parity))
        if min(shrunk) < window_size:
            offending.append((level_index, int(factor), shrunk))
    if offending:
        details = "; ".join(
            f"level {idx + 1}/{len(levels)} (shrink_factor={factor}) -> shape {shape}"
            for idx, factor, shape in offending
        )
        raise ValueError(
            f"Pyramid level too small for the similarity window (window_size={window_size}, "
            f"from neighborhood_radius): {details}. The fixed/moving image shape is "
            f"{tuple(torch_shape)}; every resolution level's shrunk shape must have every axis "
            f">= window_size. Use fewer/smaller shrink factors in 'levels' (and a matching "
            "'reg_iterations'), or a smaller 'neighborhood_radius', for this image size. This "
            "check applies regardless of 'regularizer'/'optimizer': a level this small produces "
            "unreliable or degenerate registration (near-zero gradient, near-identity or "
            "collapsed warp) for every combination, not only 'dsti'/'sobolev'."
        )


def _downsample_metadata(meta: Dict[str, tuple], factor: int, syntx_parity: bool = False) -> Dict[str, tuple]:
    if factor == 1:
        return meta
    torch_shape = _shrunk_torch_shape(meta["torch_shape"], factor, syntx_parity)
    shape_itk = tuple(reversed(torch_shape))
    extent = tuple((size - 1) * spacing for size, spacing in zip(meta["shape"], meta["spacing"]))
    spacing = tuple(e / max(size - 1, 1) for e, size in zip(extent, shape_itk))
    new_meta = dict(meta)
    new_meta["shape"] = shape_itk
    new_meta["torch_shape"] = torch_shape
    new_meta["spacing"] = spacing
    return new_meta


_PYRAMIDS = ("deformable", "legacy")


def _deformable_plan(meta: Dict[str, tuple], factor, reference_spacing=None, smoothing_sigma=None) -> dict:
    """One level of syntx's central 'deformable' pyramid plan (``syntx.pyramid``, syntx >= 6.2 / commit afcdf87, used by
    ``syntx.syn`` and ported for :mod:`antstorch.greedy`): millimetre-isotropic target ``factor x min(reference spacing)``,
    corner-aligned grid ``m = round((n - 1) s / t) + 1`` per axis with a 16-voxel floor, native grid at level 1, and an
    anti-aliasing Gaussian of ``0.5 x`` the axis coarsening factor (native voxels; ``smoothing_sigma`` replaces it) on the
    coarsened axes only."""
    from ..greedy import _plan_levels_deformable
    return _plan_levels_deformable(meta["torch_shape"], meta["spacing"], [factor], reference_spacing, sigma=smoothing_sigma)[0]


def _level_torch_shape(meta: Dict[str, tuple], factor: int, syntx_parity: bool = False, pyramid: str = "legacy",
                       reference_spacing=None) -> Tuple[int, ...]:
    """Spatial (torch-order) shape of a pyramid level, from metadata alone."""
    if pyramid == "deformable":
        return tuple(_deformable_plan(meta, factor, reference_spacing)["shape_zyx"])
    return _shrunk_torch_shape(meta["torch_shape"], factor, syntx_parity)


def _downsample_image(image: Tensor, meta: Dict[str, tuple], factor: int, syntx_parity: bool = False, smoothing_sigma=None,
                      pyramid: str = "legacy", reference_spacing=None) -> Tuple[Tensor, Dict[str, tuple]]:
    """One pyramid level of ``image`` and its metadata.

    ``pyramid='deformable'``: syntx's central planner (see :func:`_deformable_plan`); the moving image is planned on its
    own grid with the fixed image's spacing as ``reference_spacing``. ``pyramid='legacy'``: Gaussian presmoothing
    (``sigma = log2(factor)`` voxels, or ``smoothing_sigma``) then trilinear resampling to ``(n - 1) // factor + 1``
    voxels (``floor(n / factor)`` with ``syntx_parity``) -- syntx's pyramid before 2026-10-10.
    Both preserve the physical extent (corner-aligned grids)."""
    if pyramid == "deformable":
        from ..greedy import _apply_level
        plan = _deformable_plan(meta, factor, reference_spacing, smoothing_sigma)
        out = _apply_level(image, plan)
        if plan["is_identity"] and tuple(plan["shape_zyx"]) == tuple(meta["torch_shape"]):
            return out, meta
        new_meta = dict(meta)
        new_meta["torch_shape"] = tuple(int(v) for v in plan["shape_zyx"])
        new_meta["shape"] = tuple(reversed(new_meta["torch_shape"]))
        new_meta["spacing"] = tuple(float(v) for v in plan["spacing_xyz"])
        return out, new_meta
    if factor == 1 and (smoothing_sigma is None or smoothing_sigma == 0):
        return image, meta
    sigma = math.log2(factor) if smoothing_sigma is None else smoothing_sigma
    blurred = _gaussian_blur_image(image, sigma)
    new_meta = _downsample_metadata(meta, factor, syntx_parity)
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
    if name == "box_cc2":
        return box_cc2_loss_nd(I, J, window_size=window_size)
    if name == "dice":
        return soft_dice_loss_nd(I, J, mask=mask)
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
    gaussian_sigma_mode: str = "physical",
    conservative_smooth: bool = False,
    sobolev_alpha: Optional[float] = None,
    origin_itk=None,
    direction_itk=None,
    extra_reg_kwargs: Optional[Dict[str, object]] = None,
) -> Tensor:
    if regularizer in COUPLED_CONTINUUM_REGULARIZERS:
        # syntx.syn (>= 6.43): build_level_regularizer(param_domain='physical_mm') on the raw
        # gradient (channel-last, (z, y, x) components in mm), alpha = sobolev_alpha (None: the
        # operator's own default from fluid_sigma), no conservative post-filter.
        if sigma <= 0:
            return field
        reg_fn = build_level_regularizer(
            regularizer,
            alpha=sobolev_alpha,
            sobolev_alpha=sobolev_alpha,
            spacing=spacing_itk,
            origin=origin_itk,
            direction=direction_itk,
            fluid_sigma=sigma,
            param_domain="physical_mm",
            extra_reg_kwargs=dict(extra_reg_kwargs or {}),
        )
        return reg_fn(field)
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
        # gaussian_sigma_mode='physical' (default) scales sigma per axis by
        # spacing_itk, so the same nominal sigma is a comparably strong
        # filter at every resolution regardless of voxel anisotropy -- this
        # is this port's own, intentional convention. gaussian_sigma_mode=
        # 'voxel' applies sigma directly in voxels with no spacing scaling
        # instead, matching syntx.syn's own default 'gaussian' path (its
        # separable_gaussian_filter call site passes no spacing/sigma_mode
        # at all, so it defaults to voxel-space) -- pass this when comparing
        # numerically against syntx (see the project doc, "comparaison
        # syntx/antstorch, écart gaussian/sobolev").
        return separable_gaussian_filter(field, sigma, spacing=spacing_itk, sigma_mode=gaussian_sigma_mode)
    if regularizer in ("sobolev", "dsti"):
        # conservative_smooth=False (default) applies only the spectral
        # Green's-operator pass -- this port's own, intentional convention.
        # conservative_smooth=True additionally applies a second, spatial
        # Bessel-kernel Gaussian pass afterward, at half the Green's-operator
        # sigma and in voxel space (no physical-spacing scaling) -- matching
        # syntx.syn's own default ("conservative mode", fast_smooth=False)
        # sobolev/dsti path, which stacks its spectral Green's operator with
        # exactly this extra spatial pass. Pass this when comparing
        # numerically against syntx; see the gaussian branch above for why
        # this port does not do so by default.
        # sobolev_alpha (opt-in): explicit Green's-operator alpha, as syntx's
        # sobolev_alpha. Since syntx 6.40 both operators also receive the level
        # spacing (frequency grid / Laplacian eigenvalues in mm), so the spacing is
        # always passed here too. None keeps alpha = sigma/2.
        if regularizer == "sobolev":
            if sobolev_alpha is not None:
                result = apply_sobolev_green_operator(field, fluid_sigma=sigma, alpha=sobolev_alpha, spacing=spacing_itk)
            else:
                result = apply_sobolev_green_operator(field, fluid_sigma=sigma, spacing=spacing_itk)
        else:
            result = apply_dsti_green_operator(field, fluid_sigma=sigma, spacing=spacing_itk,
                                               **({"alpha": sobolev_alpha} if sobolev_alpha is not None else {}))
        if conservative_smooth:
            result = separable_gaussian_filter(result, sigma * 0.5, sigma_mode="voxel")
        return result
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
    sampled = grid_sample(warp.movedim(-1, 1), coords_norm, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.movedim(1, -1) - delta


def _compose_fixed_grid(first: Tensor, second: Tensor, X_phys: Tensor, meta_t: Dict[str, Tensor]) -> Tensor:
    """``T_second o T_first``: ``first(x) + second(x + first(x))``, both fields on the same (fixed) grid."""
    coords_phys = X_phys + first
    coords_norm = physical_to_normalized_torch_cached(
        coords_phys, meta_t["shape_t"], meta_t["spacing_t"], meta_t["origin_t"], meta_t["direction_t"]
    )
    sampled = grid_sample(second.movedim(-1, 1), coords_norm, mode="bilinear", padding_mode="border", align_corners=True)
    return first + sampled.movedim(1, -1)


def _fit_syn_level(
    I_curr: Tensor,
    J_curr: Tensor,
    fixed_meta: Dict[str, tuple],
    moving_meta: Dict[str, tuple],
    warps: List[Optional[Tensor]],
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
    gaussian_sigma_mode: str = "physical",
    conservative_smooth: bool = False,
    optimizer: str = "gradient_descent",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    regadam_grad_sigma: float = 1.8,
    regadam_quotient_sigma: float = 0.8,
    inverse_schedule: str = "per_iteration",
    end_of_level_inverse_steps: int = 30,
    inv_refresh_every: int = 1,
    inv_tolerance: float = 0.1,
    polish_inv_steps: int = 0,
    polish_inv_tol: float = 0.02,
    syntx_parity: bool = False,
    cfl_shrink_ratio: Optional[float] = None,
    use_analytical_gradients: bool = True,
    sobolev_alpha: Optional[float] = None,
    optimizer_lr: float = 1.0,
    metric_terms=None,
    metric_masks=None,
    metric_chunks: int = 1,
    stationary_boundary: bool = False,
    regularizer_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list]:
    # ``warps`` is the caller's [l2r, r2l, l2r_inv, r2l_inv] list, consumed
    # here: emptying it drops the caller's references, so each level-start
    # field is freed as soon as this function rebinds its own name (instead
    # of staying alive for the whole level). The inverses may be None when
    # inverse_schedule='end_of_level' (nothing reads them before the end).
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = warps
    warps.clear()
    device, dtype = I_curr.device, I_curr.dtype
    fixed_meta_t = metadata_tensors_from_dict(fixed_meta, device, dtype)
    moving_meta_t = metadata_tensors_from_dict(moving_meta, device, dtype)
    X_phys = _physical_grid(fixed_meta, device, dtype)
    boundary_mask = get_boundary_mask(fixed_meta["torch_shape"], device, dtype)
    # Geometry forwarded to the continuum regularizers (physical-space D_zyx conjugation).
    _reg_geom = dict(origin_itk=fixed_meta["origin"], direction_itk=fixed_meta["direction"],
                     extra_reg_kwargs=regularizer_kwargs)
    if syntx_parity:
        # syntx scales the CFL step by sqrt(shrink ratio) <= 1 (coarse levels take SMALLER steps); the default here
        # is the inverse. syntx >= 6.2 (deformable pyramid): the ratio is min over axes of finest / level spacing
        # (syntx.pyramid.cfl_shrink_ratio); before, curr_size / original_size of the z axis, ~ 1 / shrink_factor.
        ratio = float(cfl_shrink_ratio) if cfl_shrink_ratio is not None else 1.0 / float(shrink_factor)
        level_cfl_voxels = grad_step * math.sqrt(ratio)
    else:
        level_cfl_voxels = grad_step * math.sqrt(float(shrink_factor))
    best_loss = float("inf")
    best_state = None

    # reg_adam: per-voxel Adam first/second moments, reset at the start of
    # every pyramid level (this function is called once per level) --
    # mirrors syntx's greedy.py GreedyRegistration.fit() ("Reset Adam
    # warm-up step at each scale level"). Kept as plain tensors (not a
    # torch.optim.Optimizer over nn.Parameters) since warp_l2r/warp_r2l are
    # not themselves the optimized leaves here -- the *gradient* fed into
    # the existing regularizer/CFL/composition pipeline below is what
    # changes between the two optimizer modes, nothing downstream of it.
    use_reg_adam = optimizer == "reg_adam"
    if use_reg_adam:
        adam_state_l = None
        adam_state_r = None
    # 'syn_reg_adam': syntx.syn()/SyNTo's own reg_adam (NOT the greedy.py path ported above). Moments are
    # per-level, like syntx (self._adam_* are re-created at the start of each level).
    use_syn_reg_adam = optimizer == "syn_reg_adam"
    if use_syn_reg_adam:
        adam_beta1, adam_beta2 = adam_betas
        adam_t = 0
        adam_m_l = adam_m_r = adam_v_l = adam_v_r = None

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

    def _chunked_cc_term(left, right, a, b, valid, name, window, weight):
        # Slab-wise CC (core.losses.local_ncc_loss_chunked): backpropagates
        # weight * loss into a/b (and through them to the half-warps) slab by
        # slab, and returns the detached weighted loss -- plus a zero term
        # tied to the half-warps when they are given, so the caller's
        # loss.backward() still works and adds nothing.
        value, grad_a, grad_b = local_ncc_loss_chunked(
            a, b, valid, window_size=window, squared=name in ("cc2", "lncc2"), chunks=metric_chunks,
        )
        torch.autograd.backward((a, b), (grad_a * weight, grad_b * weight))
        value = weight * value
        if left is not None:
            return left.sum() * 0 + right.sum() * 0 + value
        return value

    def objective(left, right, left_inv, right_inv, backward=False):
        # With metric_terms and backward=True, each term is backpropagated as
        # soon as it is evaluated (into detached copies of the two sampling
        # grids), so only one term's graph -- and, for 'label_msq', one
        # label's on-the-fly masks -- is alive at a time; the accumulated grid
        # gradients are then pushed back to the half-warps. The returned loss
        # has the same value, but its graph only carries a zero term (its
        # backward() adds nothing). Ignored for the single-metric path.
        if metric_terms is None:
            a, b, _, _, valid = prepare_mid_images_and_gradients_torch(
                left, right, left_inv, right_inv, I_curr, J_curr, X_phys,
                fixed_meta_t["shape_t"], fixed_meta_t["spacing_t"], fixed_meta_t["origin_t"], fixed_meta_t["direction_t"],
                moving_meta_t["shape_t"], moving_meta_t["spacing_t"], moving_meta_t["origin_t"], moving_meta_t["direction_t"],
                fixed_meta["spacing"], moving_meta["spacing"], M_phys, t_phys, None,
                use_analytical_gradients=use_analytical_gradients,
                compute_gradients=False,
            )
            if metric_chunks > 1 and similarity in _CC_METRICS and backward and torch.is_grad_enabled():
                return _chunked_cc_term(left, right, a, b, valid, similarity, window_size, 1.0)
            return _similarity_loss(similarity, a, b, valid, window_size, num_bins)
        fixed_grid = physical_to_normalized_torch_cached(
            X_phys + left, fixed_meta_t["shape_t"], fixed_meta_t["spacing_t"],
            fixed_meta_t["origin_t"], fixed_meta_t["direction_t"],
        )
        moving_grid = physical_to_normalized_torch_cached(
            (X_phys + right) @ M_phys.t() + t_phys,
            moving_meta_t["shape_t"], moving_meta_t["spacing_t"],
            moving_meta_t["origin_t"], moving_meta_t["direction_t"],
        )
        valid = (((fixed_grid >= -1) & (fixed_grid <= 1)).all(-1)
                 & ((moving_grid >= -1) & (moving_grid <= 1)).all(-1)).unsqueeze(1).to(dtype)
        if metric_masks is not None:
            for mask, grid in zip(metric_masks, (fixed_grid, moving_grid)):
                if mask is not None:
                    sampled = grid_sample(mask, grid.detach(), mode="nearest", padding_mode="zeros", align_corners=True)
                    valid = valid * (sampled > 0).to(dtype)
        per_term = backward and torch.is_grad_enabled()
        if per_term:
            sample_fixed = fixed_grid.detach().requires_grad_(True)
            sample_moving = moving_grid.detach().requires_grad_(True)
        else:
            sample_fixed, sample_moving = fixed_grid, moving_grid
        loss = left.sum() * 0 + right.sum() * 0
        for name, a, b, weight, parameter in metric_terms:
            if name == "label_msq":
                a, b, name = (a == parameter).to(dtype), (b == parameter).to(dtype), "mse"
            a_mid = grid_sample_nd(a, sample_fixed, use_analytical_gradients=use_analytical_gradients)
            b_mid = grid_sample_nd(b, sample_moving, use_analytical_gradients=use_analytical_gradients)
            a = b = None
            radius = int(parameter) if name in ("cc", "lncc", "cc2", "lncc2", "box_cc2") else 0
            bins = int(parameter) if name in ("mi", "mattes") else num_bins
            if per_term and metric_chunks > 1 and name in _CC_METRICS:
                term = _chunked_cc_term(None, None, a_mid, b_mid, valid, name, 2 * radius + 1, weight)
            else:
                term = weight * _similarity_loss(name, a_mid, b_mid, valid, 2 * radius + 1, bins)
            a_mid = b_mid = None
            if per_term and term.requires_grad:
                term.backward()
                term = term.detach()
            loss = loss + term
            term = None
        if per_term:
            torch.autograd.backward((fixed_grid, moving_grid), (sample_fixed.grad, sample_moving.grad))
        return loss

    history = []
    for iteration in range(iterations):
        warp_l2r_leaf = warp_l2r.detach().requires_grad_(True)
        warp_r2l_leaf = warp_r2l.detach().requires_grad_(True)

        loss = objective(warp_l2r_leaf, warp_r2l_leaf, warp_l2r_inv, warp_r2l_inv, backward=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SyN loss at resolution level {level_index + 1}")
        if syntx_parity and float(loss.detach().item()) < best_loss:
            # syntx keeps the (pre-update) warps of the best loss seen.
            best_loss = float(loss.detach().item())
            best_state = tuple(None if field is None else field.detach().clone()
                               for field in (warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv))
        loss.backward()
        current = float(loss.detach().item())
        # Drop the loss (and with it the autograd graph, whose AccumulateGrad
        # nodes would otherwise keep this iteration's leaves alive into the
        # next one) and detach the gradients from their leaves.
        del loss
        grad_l, warp_l2r_leaf.grad = warp_l2r_leaf.grad, None
        grad_r, warp_r2l_leaf.grad = warp_r2l_leaf.grad, None
        if grad_l is None or grad_r is None or not torch.isfinite(grad_l).all() or not torch.isfinite(grad_r).all():
            raise FloatingPointError(f"non-finite SyN half-warp gradient at resolution level {level_index + 1}")

        if use_reg_adam:
            # Ported from syntx's greedy.py GreedyRegistration._fit_scale,
            # whose regadam path is THREE stages, not one (verified against
            # syntx/src/syntx/greedy.py directly after this port produced
            # near-random real-data Dice -- see project doc, RegAdam
            # real-data investigation):
            #   1. Pre-smooth the RAW similarity gradient with a Gaussian
            #      kernel (regadam_grad_sigma, voxel-space, syntx default
            #      1.8 == its 'flow_sigma') *before* accumulating Adam
            #      moments -- without this, the per-voxel Adam quotient is
            #      computed from spatially incoherent noise.
            #   2. Compute the bias-corrected Adam step-direction quotient
            #      from that smoothed gradient.
            #   3. Post-smooth the quotient itself with a SECOND, separate
            #      Gaussian kernel (regadam_quotient_sigma, voxel-space,
            #      syntx default 0.8 == its 'regadam_sigma') -- independent
            #      of, and in addition to, the level's own `regularizer`
            #      (dsti/sobolev/gaussian/bspline) applied further below.
            # Earlier versions of this port skipped stages 1 and 3 (see git
            # history), relying solely on the level regularizer to smooth
            # the Adam quotient -- that is NOT equivalent: without spatial
            # coherence in the gradient feeding Adam and in the resulting
            # quotient, per-voxel Adam-normalized directions are close to
            # sign noise (near-unit magnitude even in flat/background
            # regions with tiny raw gradients), which overwhelms the
            # registration on real data even though it passed synthetic
            # unit tests.
            if regadam_grad_sigma > 0:
                grad_l = separable_gaussian_filter(grad_l, regadam_grad_sigma, sigma_mode="voxel")
                grad_r = separable_gaussian_filter(grad_r, regadam_grad_sigma, sigma_mode="voxel")

            adam_state_l, grad_l = reg_adam_direction(
                grad_l, adam_state_l, betas=adam_betas, eps=adam_eps,
            )
            adam_state_r, grad_r = reg_adam_direction(
                grad_r, adam_state_r, betas=adam_betas, eps=adam_eps,
            )

            if regadam_quotient_sigma > 0:
                grad_l = separable_gaussian_filter(grad_l, regadam_quotient_sigma, sigma_mode="voxel")
                grad_r = separable_gaussian_filter(grad_r, regadam_quotient_sigma, sigma_mode="voxel")

        if use_syn_reg_adam:
            # syntx.syn() / SyNTo reg_adam update, step by step (syntx/syn.py, optimizer_type == 'reg_adam'):
            #  1. the raw half-warp gradient (boundary-masked) feeds the Adam moments UNSMOOTHED -- syntx
            #     >= 6.43 removed the earlier pre-Adam Green-operator pass ("H^4 double-filtering");
            #  2. bias-corrected Adam quotient of that gradient (moments reset every level);
            #  3. the quotient is smoothed by the level regularizer ONCE (no post-filter, no boundary mask);
            #  4. one CFL scale shared by both half-warps, clamped to <= 1 (steps are only ever shrunk), times
            #     optimizer_lr; the norm is taken in the field's own units, as syntx does.
            grad_l = grad_l * boundary_mask
            grad_r = grad_r * boundary_mask
            if adam_m_l is None:
                adam_m_l, adam_m_r = torch.zeros_like(grad_l), torch.zeros_like(grad_r)
                adam_v_l, adam_v_r = torch.zeros_like(grad_l), torch.zeros_like(grad_r)
            adam_t += 1
            bias1 = 1.0 - adam_beta1 ** adam_t
            bias2 = 1.0 - adam_beta2 ** adam_t
            bias2_sqrt = math.sqrt(bias2)
            step_size = bias2_sqrt / bias1
            eps_scaled = adam_eps * bias2_sqrt
            adam_m_l.mul_(adam_beta1).add_(grad_l, alpha=1.0 - adam_beta1)
            adam_v_l.mul_(adam_beta2).addcmul_(grad_l, grad_l, value=1.0 - adam_beta2)
            u_raw_l = (adam_m_l * step_size).div_(adam_v_l.sqrt().add_(eps_scaled))
            adam_m_r.mul_(adam_beta1).add_(grad_r, alpha=1.0 - adam_beta1)
            adam_v_r.mul_(adam_beta2).addcmul_(grad_r, grad_r, value=1.0 - adam_beta2)
            u_raw_r = (adam_m_r * step_size).div_(adam_v_r.sqrt().add_(eps_scaled))
            u_reg_l = _apply_regularizer(
                u_raw_l, regularizer, flow_sigma, fixed_meta["spacing"],
                gaussian_sigma_mode=gaussian_sigma_mode, conservative_smooth=False, sobolev_alpha=sobolev_alpha, **_reg_geom,
            )
            u_reg_r = _apply_regularizer(
                u_raw_r, regularizer, flow_sigma, fixed_meta["spacing"],
                gaussian_sigma_mode=gaussian_sigma_mode, conservative_smooth=False, sobolev_alpha=sobolev_alpha, **_reg_geom,
            )
            max_u = torch.maximum(torch.linalg.norm(u_reg_l, dim=-1).max(), torch.linalg.norm(u_reg_r, dim=-1).max()).clamp_min(1e-4)
            cfl_scale = (level_cfl_voxels / max_u).clamp_max(1.0)
            delta_l = (cfl_scale * float(optimizer_lr)) * u_reg_l
            delta_r = (cfl_scale * float(optimizer_lr)) * u_reg_r
        else:
            grad_l = _apply_regularizer(
                grad_l * boundary_mask, regularizer, flow_sigma, fixed_meta["spacing"],
                mesh_size=update_mesh_size_level, domain=bspline_domain,
                enforce_stationary_boundary=bspline_enforce_stationary_boundary,
                gaussian_sigma_mode=gaussian_sigma_mode, conservative_smooth=conservative_smooth,
                sobolev_alpha=sobolev_alpha, **_reg_geom,
            )
            grad_r = _apply_regularizer(
                grad_r * boundary_mask, regularizer, flow_sigma, fixed_meta["spacing"],
                mesh_size=update_mesh_size_level, domain=bspline_domain,
                enforce_stationary_boundary=bspline_enforce_stationary_boundary,
                gaussian_sigma_mode=gaussian_sigma_mode, conservative_smooth=conservative_smooth,
                sobolev_alpha=sobolev_alpha, **_reg_geom,
            )

            delta_l = _cfl_normalize(grad_l, fixed_meta_t["spacing_t"], level_cfl_voxels)
            delta_r = _cfl_normalize(grad_r, fixed_meta_t["spacing_t"], level_cfl_voxels)
        if antisymmetric:
            common_mode = delta_l + delta_r
            delta_l = delta_l - 0.5 * common_mode
            delta_r = delta_r - 0.5 * common_mode

        warp_l2r = _eulerian_update(warp_l2r_leaf.detach(), delta_l, X_phys, fixed_meta_t)
        warp_r2l = _eulerian_update(warp_r2l_leaf.detach(), delta_r, X_phys, fixed_meta_t)
        # Release this iteration's leaves, gradients and steps now rather than
        # when the next iteration rebinds them (they would otherwise sit on
        # top of the next forward pass's peak).
        warp_l2r_leaf = warp_r2l_leaf = grad_l = grad_r = delta_l = delta_r = None
        common_mode = u_raw_l = u_raw_r = u_reg_l = u_reg_r = None

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

        if stationary_boundary:
            # syntx (SyNTo stationary_boundary=True, ITK EnforceStationaryBoundary): zero displacement on every face
            # of both half-warps after each update and total-field smoothing, before the inverse refresh (the
            # inverse solvers keep their own result at zero there).
            warp_l2r = warp_l2r * boundary_mask
            warp_r2l = warp_r2l * boundary_mask

        # Nothing inside this loop reads warp_*_inv (prepare_mid_images_and_
        # gradients_torch accepts them but never uses them); they only feed the
        # final forward/inverse composition in syn_registration(). The default
        # 'per_iteration' schedule keeps the historical warm-started chain of
        # in_loop_inverse_steps solver steps after every iteration;
        # 'end_of_level' skips it here and inverts once after the last level's
        # iterations (see below).
        if inverse_schedule == "per_iteration":
            warp_l2r_inv = update_inverse_field_nd(
                warp_l2r, warp_l2r_inv, steps=in_loop_inverse_steps, method=inverse_method,
                spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
            )
            warp_r2l_inv = update_inverse_field_nd(
                warp_r2l, warp_r2l_inv, steps=in_loop_inverse_steps, method=inverse_method,
                spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
            )
        elif inverse_schedule == "refresh" and ((iteration + 1) % inv_refresh_every == 0 or iteration == iterations - 1):
            # syntx >= 6.35 ("Option A"): the warm-started in-loop inverses are refreshed every
            # inv_refresh_every iterations of this level (and at its last scheduled iteration),
            # with syntx's tolerance (max inv_tolerance, mean inv_tolerance / 100, in voxels).
            warp_l2r_inv = update_inverse_field_nd(
                warp_l2r, warp_l2r_inv, steps=in_loop_inverse_steps, method=inverse_method,
                max_error_threshold=inv_tolerance, mean_error_threshold=inv_tolerance * 0.01,
                spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
            )
            warp_r2l_inv = update_inverse_field_nd(
                warp_r2l, warp_r2l_inv, steps=in_loop_inverse_steps, method=inverse_method,
                max_error_threshold=inv_tolerance, mean_error_threshold=inv_tolerance * 0.01,
                spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
            )

        history.append(current)
        if verbose:
            print(
                f"  [SyN level {level_index + 1}/{num_levels}] iteration {iteration + 1:04d}/{iterations}: "
                f"loss={current:.8g}"
            )
        if syntx_parity and len(history) >= 10:
            # syntx check_convergence(window=10, slope_threshold=1e-6)
            y = np.asarray(history[-10:], dtype=np.float64)
            x = np.arange(10, dtype=np.float64)
            slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
            if abs(slope) <= 1e-6:
                break

    if syntx_parity and best_state is not None and len(history) > 0:
        with torch.no_grad():
            final_loss = float(objective(warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv).item())
        if final_loss >= best_loss:
            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = best_state

    if (inverse_schedule in ("per_iteration", "refresh") and iterations > 0
            and (polish_inv_steps > 0 or (inverse_schedule == "refresh" and inv_refresh_every > 1))):
        # syntx >= 6.35: polish both warm-started half-warp inverses at the end of every level
        # (polish_inv_steps solver steps, max error polish_inv_tol voxels, mean polish_inv_tol / 100),
        # after the best-state restore. With polish_inv_steps=0 and a refresh cadence > 1, syntx
        # still polishes, with max(in_loop_inverse_steps, 10) steps.
        polish_steps = polish_inv_steps if polish_inv_steps > 0 else max(int(in_loop_inverse_steps), 10)
        warp_l2r_inv = update_inverse_field_nd(
            warp_l2r, warp_l2r_inv, steps=polish_steps, method=inverse_method,
            max_error_threshold=polish_inv_tol, mean_error_threshold=polish_inv_tol * 0.01,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )
        warp_r2l_inv = update_inverse_field_nd(
            warp_r2l, warp_r2l_inv, steps=polish_steps, method=inverse_method,
            max_error_threshold=polish_inv_tol, mean_error_threshold=polish_inv_tol * 0.01,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )

    if inverse_schedule == "end_of_level" and level_index == num_levels - 1:
        # Only the final level's inverse is consumed (by the final composition
        # in syn_registration()); coarser levels' inverses would be discarded
        # here anyway, so they are skipped. Solved from the first-order guess
        # (-W) with end_of_level_inverse_steps steps.
        warp_l2r_inv = update_inverse_field_nd(
            warp_l2r, -warp_l2r, steps=end_of_level_inverse_steps, method=inverse_method,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )
        warp_r2l_inv = update_inverse_field_nd(
            warp_r2l, -warp_r2l, steps=end_of_level_inverse_steps, method=inverse_method,
            spacing=fixed_meta["spacing"], origin=fixed_meta["origin"], direction=fixed_meta["direction"], X_phys=X_phys,
        )

    return tuple(None if field is None else field.detach()
                 for field in (warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv)) + (history,)


def _fit_affine_from_ants(
    fixed,
    moving,
    device,
    dtype,
    *,
    transform_type: str,
    similarity: str,
    neighborhood_radius,
    num_bins: int,
    shrink_factors,
    smoothing_sigmas,
    iterations,
    learning_rate,
    multi_start: bool,
    center_of_mass_init: bool,
    verbose: bool,
    smoothing_sigmas_units: str = "mm",
    intensity_normalization: Union[bool, str] = "auto",
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
    fixed_tensor = ants_image_to_tensor(fixed, device, dtype, normalize=intensity_normalization)
    moving_tensor = ants_image_to_tensor(moving, device, dtype, normalize=intensity_normalization)
    result = _bspline_affine_registration(
        fixed_tensor,
        moving_tensor,
        fixed_domain,
        moving_domain,
        transform_type=transform_type,
        similarity=similarity,
        neighborhood_radius=neighborhood_radius,
        num_bins=num_bins,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        smoothing_sigmas_units=smoothing_sigmas_units,
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


# ---------------------------------------------------------------------------
# syntx-aligned defaults
# ---------------------------------------------------------------------------
# ``syn_registration`` resolves every hyperparameter left at ``None`` from this
# table, keyed by ``(regularizer, optimizer)``. The values reproduce the
# ``--matched`` configuration of tools/benchmarks/compare_syntx_antstorch_mindboggle.py
# (validated against syntx on Mindboggle-101): syntx treats ``flow_sigma`` as an
# ITK *variance* (sigma = sqrt(flow_sigma)), antstorch uses it as sigma, hence the
# sqrt() values below; cc2 similarity, (4, 2, 1) pyramid, voxel-unit gaussian
# sigma, conservative post-filter for sobolev/dsti, and the end-of-level inverse
# schedule (same Dice, ~20 % less time than the per-iteration chain).
# Anything the table does not state keeps antstorch's previous default
# (grad_step 0.5 for dsti and reg_adam, flow_sigma 3.0 where syntx has no
# equivalent). ``syntx_defaults=False`` restores the previous defaults wholesale.
# Inverse handling follows syntx's 'canonical_2026_11_11' defaults ("Option A", syntx >= 6.35.0):
# warm-started in-loop inverses (10 solver steps, as syntx.syn()) refreshed every (1, 2, 4)
# iterations from the coarsest to the finest level, a 30-step end-of-level polish (0.02 voxel),
# and a safeguarded Levenberg-Marquardt polish of the composed inverse (0.30 voxel).
_COMMON_SYNTX_DEFAULTS = dict(
    syn_metric="cc2", levels=(4, 2, 1), gaussian_sigma_mode="voxel", inverse_schedule="refresh",
    in_loop_inverse_steps=10, inv_refresh_every=(1, 2, 4), polish_inv_steps=30, polish_inv_tol=0.02,
    compose_polish_tol=0.30, pyramid="deformable", stationary_boundary=True,
    digital_diffeomorphism=True,
)
_SYNTX_DEFAULTS = {
    ("gaussian", "gradient_descent"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.25, reg_iterations=(100, 100, 20),
        flow_sigma=math.sqrt(3.0), conservative_smooth=False, sobolev_alpha=None),
    ("sobolev", "gradient_descent"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.4, reg_iterations=(100, 100, 20),
        flow_sigma=math.sqrt(2.4), conservative_smooth=True, sobolev_alpha=2.25),
    ("bspline", "gradient_descent"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.25, reg_iterations=(100, 100, 20),
        flow_sigma=3.0, conservative_smooth=False, sobolev_alpha=None),
    ("dsti", "gradient_descent"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 50, 10),
        flow_sigma=math.sqrt(3.0), conservative_smooth=True, sobolev_alpha=None),
    ("gaussian", "reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 100, 20),
        flow_sigma=3.0, conservative_smooth=False, sobolev_alpha=None),
    ("sobolev", "reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 100, 20),
        flow_sigma=3.0, conservative_smooth=True, sobolev_alpha=None),
    ("bspline", "reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 100, 20),
        flow_sigma=3.0, conservative_smooth=False, sobolev_alpha=None),
    ("sobolev", "syn_reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.4, reg_iterations=(100, 100, 20),
        flow_sigma=math.sqrt(2.4), conservative_smooth=True, sobolev_alpha=2.25),
    # syntx's 'syn_dsti1' benchmark arm: reg_adam + dsti1, grad_step 0.5, flow_sigma 3.0 (a variance, and only an
    # on/off gate there: the kernel is set by sobolev_alpha 1.0), (100, 50, 10), in_loop_inv_steps 10.
    ("dsti", "syn_reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 50, 10),
        flow_sigma=math.sqrt(3.0), conservative_smooth=True, sobolev_alpha=1.0),
    ("dsti", "reg_adam"): dict(
        _COMMON_SYNTX_DEFAULTS, grad_step=0.5, reg_iterations=(100, 50, 10),
        flow_sigma=math.sqrt(3.0), conservative_smooth=True, sobolev_alpha=None),
}
# syntx 6.44 'canonical_2026_11_20' (docs/provenance/mindboggle_90pair_common_navier_2026-11-20.md):
# regularizer='navier' (s=2.0, poisson_ratio=0.20), sobolev_alpha=1.5, grad_step=0.6, flow_sigma=2.4
# (an ITK variance in syntx -> sqrt here), total_sigma=0, optimizer='cfl', (100, 100, 20), and the same
# "Option A" inverse handling as before. syntx.syn()'s wrapper defaults do not depend on the regularizer,
# so every coupled continuum operator gets them (with its own operator defaults otherwise).
_CANONICAL_NAVIER_DEFAULTS = dict(
    _COMMON_SYNTX_DEFAULTS, grad_step=0.6, reg_iterations=(100, 100, 20),
    flow_sigma=math.sqrt(2.4), conservative_smooth=False, sobolev_alpha=1.5)
for _reg in _CONTINUUM_REGULARIZERS:
    for _opt in ("gradient_descent", "reg_adam", "syn_reg_adam"):
        _SYNTX_DEFAULTS[(_reg, _opt)] = _CANONICAL_NAVIER_DEFAULTS
del _reg, _opt
_DEFAULT_REGULARIZER = "navier"
_LEGACY_DEFAULTS = dict(
    syn_metric="lncc", levels=(4, 2, 1), reg_iterations=(100, 100, 50), grad_step=0.5, flow_sigma=3.0,
    gaussian_sigma_mode="physical", conservative_smooth=False, sobolev_alpha=None,
    inverse_schedule="per_iteration", in_loop_inverse_steps=6, inv_refresh_every=1, polish_inv_steps=0,
    polish_inv_tol=0.02, compose_polish_tol=0.0, pyramid="legacy", stationary_boundary=False,
    digital_diffeomorphism=False,
)


def syn_registration(
    fixed,
    moving,
    *,
    type_of_transform: str = "SyN",
    initial_affine: Optional[Tuple[Tensor, Tensor]] = None,
    affine_transform_type: str = "Affine",
    affine_similarity: str = "mse",
    affine_neighborhood_radius: Union[int, Sequence[int]] = 2,
    affine_num_bins: int = 32,
    affine_shrink_factors: Sequence[int] = (4, 2, 1),
    affine_smoothing_sigmas: Optional[Union[float, Sequence[float]]] = (2.0, 1.0, 0.0),
    affine_smoothing_sigmas_units: str = "vox",
    affine_iterations: Union[int, Sequence[int]] = (100, 100, 50),
    affine_learning_rate: Union[float, Sequence[float]] = (0.05, 0.03, 0.02),
    affine_multi_start: bool = True,
    affine_center_of_mass_init: bool = True,
    syn_metric: Optional[str] = None,
    neighborhood_radius: int = 2,
    num_bins: int = 32,
    levels: Optional[Sequence[int]] = None,
    reg_iterations: Optional[Sequence[int]] = None,
    grad_step: Optional[float] = None,
    flow_sigma: Optional[float] = None,
    total_sigma: float = 0.0,
    regularizer: Optional[str] = None,
    regularizer_kwargs: Optional[Dict[str, object]] = None,
    optimizer: str = "gradient_descent",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    regadam_grad_sigma: float = 1.8,
    regadam_quotient_sigma: float = 0.8,
    update_field_mesh_size_at_base_level: Optional[int] = None,
    total_field_mesh_size_at_base_level: int = 0,
    update_field_spline_distance: Optional[Union[float, Sequence[float]]] = None,
    total_field_spline_distance: Optional[Union[float, Sequence[float]]] = None,
    bspline_enforce_stationary_boundary: bool = True,
    inverse_method: str = "anderson",
    in_loop_inverse_steps: Optional[int] = None,
    inverse_schedule: Optional[str] = None,
    end_of_level_inverse_steps: int = 30,
    inv_refresh_every: Optional[Union[int, Sequence[int]]] = None,
    polish_inv_steps: Optional[int] = None,
    polish_inv_tol: Optional[float] = None,
    compose_polish_tol: Optional[float] = None,
    compose_polish_steps: int = 30,
    inv_tolerance: Optional[float] = None,
    pyramid: Optional[str] = None,
    stationary_boundary: Optional[bool] = None,
    digital_diffeomorphism: Optional[bool] = None,
    digital_diffeomorphism_floor: float = 0.05,
    digital_diffeomorphism_steps: int = 50,
    syntx_defaults: bool = True,
    syntx_parity: bool = False,
    use_analytical_gradients: bool = True,
    sobolev_alpha: Optional[float] = None,
    optimizer_lr: float = 1.0,
    metric_chunks: int = 1,
    antisymmetric: bool = True,
    gaussian_sigma_mode: Optional[str] = None,
    conservative_smooth: Optional[bool] = None,
    padding_mode: str = "zeros",
    intensity_normalization: Union[bool, str] = "auto",
    outprefix: str = "",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    smoothing_sigmas=None,
    multivariate_extras=None,
    primary_metric_weight: float = 1.0,
    mask=None,
    moving_mask=None,
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

    ``multivariate_extras`` adds (metric, fixed_image, moving_image, weight,
    parameter) terms to the same SyN objective. Images must share their
    respective primary domains. MSQ/MeanSquares aliases retain raw image
    amplitudes; other metrics use the standard intensity normalization.
    ``label_msq`` (alias ``LabelMSQ``) takes label maps and a label value as
    ``parameter``: it is MeanSquares between the binary masks
    ``fixed_image == parameter`` and ``moving_image == parameter``, built on
    the fly from the shared label maps instead of being stored per label.
    With multivariate terms, each term is backpropagated separately, so only
    one term's autograd graph is alive at a time.
    ``primary_metric_weight=0`` disables the primary metric. ``mask`` and
    ``moving_mask`` restrict midpoint sampling to their nonzero regions.
    ``smoothing_sigmas`` optionally sets the pyramid image smoothing in
    voxel units, one value per level. These options affect SyN only.

    ``metric_chunks`` (default 1) evaluates CC-family metrics (``cc``,
    ``lncc``, ``cc2``, ``lncc2``; primary and multivariate) in that many
    slabs along the longest axis, each with a ``2 * radius`` halo and
    backpropagated on its own, so the CC autograd graph is never held for
    the whole image (the per-voxel values, and hence the loss, are those of
    the unchunked metric). Lower peak memory for more kernel launches.

    ``affine_smoothing_sigmas`` are in full-resolution voxels by default
    (``affine_smoothing_sigmas_units='vox'``, as ANTs' ``2x1x0vox``), so the
    default (2, 1, 0) means the same amount of smoothing whatever the image
    spacing; ``'mm'`` gives physical units (the behaviour before this
    option existed, identical for 1 mm isotropic images).

    **Defaults.** ``syn_metric``, ``levels``, ``reg_iterations``, ``grad_step``,
    ``flow_sigma``, ``gaussian_sigma_mode``, ``conservative_smooth``,
    ``sobolev_alpha`` and ``inverse_schedule`` default to ``None``, meaning
    "use the syntx-aligned value for this ``(regularizer, optimizer)``" from
    ``_SYNTX_DEFAULTS``. ``regularizer=None`` (default) means ``'navier'``, syntx 6.44's
    canonical ``syntx.syn`` setting (``canonical_2026_11_20``): Navier-Cauchy Green's operator
    with ``s`` 2.0 and Poisson ratio 0.20, ``sobolev_alpha`` 1.5, ``grad_step`` 0.6,
    ``flow_sigma`` sqrt(2.4), (100, 100, 20); every coupled continuum regularizer
    (``'div_curl'``, ``'solenoidal'``, ``'beltrami'``, ...; see
    :mod:`antstorch.syn.core.regularizers`) uses these values, its extra operator parameters
    (``poisson_ratio``, ``s``, ``beta``, ``gamma``, ...) going in ``regularizer_kwargs``. The
    other presets are (e.g. ``'gaussian'``: cc2, ``grad_step`` 0.25,
    ``levels`` (4, 2, 1), ``reg_iterations`` (100, 100, 20), ``flow_sigma``
    sqrt(3) -- syntx's ``flow_sigma`` is an ITK variance, this function's is a
    sigma -- voxel-unit sigma, syntx's "Option A" inverse handling (see
    ``inverse_schedule`` below); ``'sobolev'``:
    ``grad_step`` 0.4, ``flow_sigma`` sqrt(2.4), ``sobolev_alpha`` 2.25,
    conservative post-filter; ``'dsti'``: ``reg_iterations`` (100, 50, 10),
    ``flow_sigma`` sqrt(3), conservative post-filter). Any value passed
    explicitly wins. ``syntx_defaults=False`` restores this port's previous
    defaults (lncc, ``grad_step`` 0.5, ``flow_sigma`` 3.0, ``levels`` (4, 2, 1),
    ``reg_iterations`` (100, 100, 50), ``'physical'`` sigma, spectral-only
    sobolev/dsti, ``'per_iteration'`` inverses, ``regularizer`` ``'gaussian'``). With the syntx defaults,
    ``sobolev_alpha=None`` cannot mean "no explicit alpha" for
    ``regularizer='sobolev'``; use ``syntx_defaults=False`` for that.
    ``levels`` and ``reg_iterations`` must be given together unless the
    explicit one has the same length as the default one.

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

    ``optimizer`` selects what the regularizer (``flow_sigma``) is applied
    to each iteration: ``'gradient_descent'`` (default) applies it directly
    to the raw similarity gradient, as above. ``'reg_adam'`` instead runs
    the three-stage pipeline ported from syntx's ``greedy.py``
    ``GreedyRegistration`` (verified line-for-line against syntx after an
    earlier, incomplete port produced near-random Dice on real Mindboggle-101
    data despite passing synthetic unit tests -- see project doc): (1) the
    raw similarity gradient is pre-smoothed with a voxel-space Gaussian
    kernel, ``regadam_grad_sigma`` (default 1.8, matches syntx's own
    ``flow_sigma`` default for this stage); (2) per-voxel Adam first/second
    moments (``adam_betas``, ``adam_eps``) are accumulated from that
    smoothed gradient across the level's iterations -- reset at the start
    of each pyramid level -- producing a bias-corrected step-direction
    quotient; (3) the quotient itself is smoothed with a SECOND, separate
    voxel-space Gaussian kernel, ``regadam_quotient_sigma`` (default 0.8,
    matches syntx's own ``regadam_sigma``). The level's own ``regularizer``
    (dsti/sobolev/gaussian/bspline) is then applied on top of stage 3's
    output, same as for ``'gradient_descent'``. Everything else (the CFL
    bound, antisymmetric projection, Eulerian composition) is unchanged;
    this option exists to isolate whether an accuracy gap against another
    library's regularizer comes from the regularizer itself or from the
    Adam-momentum optimizer it happens to be paired with there. Both new
    sigmas are no-ops (skipped) when ``optimizer='gradient_descent'``, and
    each can be set to ``0`` to disable that one stage independently while
    keeping ``'reg_adam'`` active.

    ``optimizer='syn_reg_adam'`` reproduces the ``reg_adam`` update of ``syntx.syn()`` (SyNTo) -- which is a
    different algorithm from the greedy.py path that ``'reg_adam'`` ports: the raw (boundary-masked)
    half-warp gradient is fed to bias-corrected Adam moments (reset every level), the quotient is smoothed
    once by the level regularizer (no post-filter; syntx >= 6.43 dropped the former pre-Adam smoothing,
    which filtered twice), and one CFL scale shared by both half-warps, clamped to <= 1 and multiplied by
    ``optimizer_lr`` (default 1.0, syntx's), sets the step. For the spectral (``'sobolev'``/``'dsti'``)
    and continuum regularizers. ``optimizer='cfl'`` is an alias of ``'gradient_descent'`` (syntx's name).

    ``use_analytical_gradients`` (default ``True``) only selects how the dense-SyN image sampling
    is differentiated with respect to the half-warps: ``True`` uses ``AnalyticalGridSample`` (the
    image's own central-difference gradient, resampled at the warped point -- needs only the
    ``grid_sample`` *forward* kernel, so 3-D SyN runs on MPS); ``False`` differentiates the
    bilinear interpolation itself through autograd (needs ``grid_sampler_*d_backward``; on an MPS
    build without it the call falls back to CPU with a ``RuntimeWarning``). The similarity metric
    is always differentiated by autograd. **This is not syntx's option of the same name**: in
    ``syntx.syn`` (default ``False``) ``True`` additionally replaces the LNCC gradient with the
    ITK/ANTs pseudo-gradient (``ANTsPseudoLNCC``), which this function never does -- so
    ``False`` here does not reproduce syntx's default any more than ``True`` does (both differ
    from it in the sampler derivative only; measured 2026-11-23 on a synthetic 3-D pair, neither
    setting is closer to ``syntx.syn``'s field).

    ``intensity_normalization`` (default ``'auto'``, as in syntx since its benchmark and
    ``syntx.syn`` both see ``normalize_image(method='auto')`` images) sets how ``fixed`` / ``moving``
    (and the intensity images of ``multivariate_extras``) are rescaled to ``[0, 1]`` before the
    affine fit and the SyN stage: ``'auto'`` clips each image to foreground percentiles chosen by
    entropy (:func:`antstorch.syn.core.utils.auto_select_intensity_percentiles`); ``'percentile'``
    to the fixed 2nd-98th foreground percentiles (the default before 2026-11-25); ``False``
    disables it. An image already in ``[0, 1]`` is left unchanged by either method. On
    Mindboggle-101 the fixed 2-98 clip cost up to 0.05 Dice against syntx on pairs where the
    entropy choice differs (e.g. (5, 99)), with a systematic change of mean Jacobian; with
    ``'auto'`` the two libraries agree (project doc ``claude/syntx-normalization-parity.md``).

    ``gaussian_sigma_mode`` and ``conservative_smooth`` tune two
    ``'gaussian'``/``'sobolev'``/``'dsti'`` regularizer-formula details that
    this port intentionally resolves differently from ``syntx.syn``'s own
    default path, found while comparing the two on real Mindboggle-101 pairs
    (project doc, "comparaison syntx/antstorch, écart gaussian/sobolev"):
    ``gaussian_sigma_mode='physical'`` (this port's previous default; now ``'voxel'`` unless ``syntx_defaults=False``)
    scales ``'gaussian'``'s ``flow_sigma`` per axis by voxel spacing, so a
    given nominal sigma is a comparably strong filter regardless of
    anisotropy; pass ``'voxel'`` to apply it directly in voxels with no
    spacing scaling instead, matching ``syntx.syn``'s own default
    ``'gaussian'`` behavior. ``conservative_smooth=False`` (this
    port's previous default; now ``True`` for ``'sobolev'``/``'dsti'`` unless ``syntx_defaults=False``) applies only the spectral Green's-operator pass for
    ``'sobolev'``/``'dsti'``; ``True`` additionally applies a second, spatial
    Bessel-kernel Gaussian pass afterward (at half the Green's-operator
    sigma, in voxel space), matching ``syntx.syn``'s own default
    ("conservative mode") ``'sobolev'``/``'dsti'`` path, which always stacks
    both. Neither default changes this port's own long-standing behavior;
    both exist to let a caller reproduce ``syntx.syn``'s numbers when that is
    the goal.

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
    ``update_field_mesh_size_at_base_level``/``update_field_spline_distance``
    and ``total_field_mesh_size_at_base_level``/``total_field_spline_distance``
    (below). Both resolved mesh sizes are applied *at the base
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
    mutually exclusive with its corresponding ``*_mesh_size_at_base_level``
    integer (explicitly giving both raises). As with ``bspline_svf_registration``,
    no image padding is performed -- the mesh size is an approximation of
    the requested spacing, exactly matching real ``antsRegistration``'s own
    (non-padding) registration-side behavior; this deliberately differs from
    real ANTs' ``N4BiasFieldCorrection``, which does pad.

    **Defaults:** ``update_field_mesh_size_at_base_level`` defaults to
    ``None``, which (when neither it nor ``update_field_spline_distance``
    is given) resolves to a physical spline distance of
    :data:`antstorch.bspline_flows.bspline_svf_registration.DEFAULT_BSPLINE_SPLINE_DISTANCE_MM`
    (26 mm) -- exactly as if ``update_field_spline_distance=26.0`` had been
    passed -- *not* ITK's own literal class default of ``1`` (4 control
    points), which this function used before 2026-08-26. This is the same
    default :func:`antstorch.bspline_flows.bspline_svf_registration.bspline_svf_registration`
    now uses for its own ``mesh_size``/``spline_distance``, so the two
    registration frameworks' B-spline density defaults agree without either
    caller specifying anything. ``total_field_mesh_size_at_base_level``
    keeps its own default of ``0`` (off, matching ``antsRegistration``'s own
    default) -- unaffected by this change; pass
    ``total_field_mesh_size_at_base_level`` or ``total_field_spline_distance``
    explicitly to enable it.
    ``bspline_enforce_stationary_boundary`` (default ``True``, ITK's own
    default) keeps the field at zero on the domain's outermost voxel layer.
    Both half-warp inverses are numerically maintained
    in-loop (``inverse_method``, ``in_loop_inverse_steps``, warm-started
    across iterations and pyramid levels) and are what the final
    ``fwdtransforms``/``invtransforms`` are built from algebraically (a
    half-warp-inverse swap), not from any generic post-hoc field inversion.

    ``inverse_schedule`` chooses *when* those inverses are maintained.
    ``'refresh'`` (the syntx-aligned default, syntx >= 6.35.0 "Option A",
    ``canonical_2026_11_11``) updates them, warm-started, with
    ``in_loop_inverse_steps`` (default 10) solver steps every
    ``inv_refresh_every`` iterations (default ``(1, 2, 4)``, one value per
    level from coarse to fine, the last value reused) and at each level's last
    scheduled iteration, with syntx's tolerance (max ``inv_tolerance``, default
    ``0.1 * min(spacing)``, mean 1 % of it); every level then ends with a
    ``polish_inv_steps`` (default 30) polish to ``polish_inv_tol`` (default
    0.02 voxel). After the final half-warp swap, the composed inverse is
    refined against the composed forward field by a safeguarded per-voxel
    Levenberg-Marquardt solver (``compose_polish_tol``, default 0.30 voxel,
    ``compose_polish_steps`` 30; 0 disables it). The historical behavior
    described above is ``'per_iteration'`` (every iteration, used with
    ``syntx_defaults=False``, where polishes are off).
    Nothing inside the SyN loop reads the inverses -- they only feed the final
    composition -- so ``'end_of_level'`` skips the per-iteration updates and
    inverts the final level's half-warps once, from the first-order guess
    ``-W``, with ``end_of_level_inverse_steps`` solver steps (``inverse_method``
    as usual). Intended as a large speed-up (the per-iteration chain was ~2/3
    of the loop time on MPS); the final inverse accuracy depends on
    ``end_of_level_inverse_steps`` rather than on the warm-started chain, so
    check Dice / identity error on your data before relying on it.

    ``pyramid`` selects how each level is built. ``'deformable'`` (the
    syntx-aligned default; syntx's central planner since its 2026-10-10
    refactor, commit afcdf87): millimetre-isotropic target spacing
    ``factor x min(fixed spacing)``, corner-aligned grid of
    ``round((n - 1) s / t) + 1`` samples per axis with a 16-voxel floor, the
    native grid at level 1, an anti-aliasing Gaussian of ``0.5 x`` the axis
    coarsening factor (or ``smoothing_sigmas``) on the coarsened axes only, the
    moving image on its own grid; with ``syntx_parity`` the CFL step uses
    syntx's ``min(finest / level spacing)`` ratio. ``'legacy'`` (with
    ``syntx_defaults=False``): ``log2(factor)``-voxel blur and
    ``(n - 1) // factor + 1`` samples (``floor(n / factor)`` with
    ``syntx_parity``), syntx's pyramid before that refactor.
    ``stationary_boundary`` (syntx-aligned default ``True``, as syntx's
    ``SyNTo``; ``False`` with ``syntx_defaults=False``) zeroes both half-warps
    on every face of the domain after each update, so the written warps are
    zero on the outermost voxel layer.

    ``levels``/``reg_iterations`` define the multi-resolution pyramid
    (coarse to fine, e.g. ``levels=(4, 2, 1)``); unlike
    :func:`antstorch.bspline_flows.bspline_svf_registration.bspline_svf_registration`'s
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
    if regularizer is None:
        # syntx >= 6.44: 'navier' (common Navier-Cauchy, s=2.0, nu=0.20); the previous default was
        # 'gaussian', which syntx_defaults=False keeps.
        regularizer = _DEFAULT_REGULARIZER if syntx_defaults else "gaussian"
    regularizer = str(regularizer).lower()
    optimizer = _OPTIMIZER_ALIASES.get(str(optimizer).lower(), str(optimizer).lower())
    if regularizer_kwargs is not None:
        _bad_reg_kw = sorted(set(regularizer_kwargs) - set(REGULARIZER_EXTRA_KEYS) - {"mask", "fixed_mask"})
        if _bad_reg_kw:
            raise ValueError(f"regularizer_kwargs: unknown key(s) {_bad_reg_kw}; expected a subset of "
                             f"{sorted(REGULARIZER_EXTRA_KEYS)}")
        if regularizer not in COUPLED_CONTINUUM_REGULARIZERS:
            raise ValueError("regularizer_kwargs only applies to the coupled continuum regularizers "
                             f"{_CONTINUUM_REGULARIZERS}, not {regularizer!r}")
        regularizer_kwargs = dict(regularizer_kwargs)
        for _mk in ("mask", "fixed_mask"):
            _mv = regularizer_kwargs.get(_mk)
            if _mv is not None and hasattr(_mv, "numpy") and hasattr(_mv, "spacing"):
                # ANTsImage (x, y, z) -> tensor (z, y, x) order; resampled to each level by the operator
                regularizer_kwargs[_mk] = torch.from_numpy(np.ascontiguousarray(_mv.numpy().T.astype(np.float32)))
    _defaults = (
        _SYNTX_DEFAULTS.get((regularizer, optimizer), _LEGACY_DEFAULTS) if syntx_defaults else _LEGACY_DEFAULTS
    )
    _levels_given, _iterations_given = levels is not None, reg_iterations is not None
    if syn_metric is None:
        syn_metric = _defaults["syn_metric"]
    if levels is None:
        levels = _defaults["levels"]
    if reg_iterations is None:
        reg_iterations = _defaults["reg_iterations"]
    if _levels_given != _iterations_given and len(levels) != len(reg_iterations):
        raise ValueError(
            "levels and reg_iterations must be given together: the default for the one omitted has "
            f"{len(_defaults['levels'])} entries (levels={tuple(_defaults['levels'])}, "
            f"reg_iterations={tuple(_defaults['reg_iterations'])}) but the one given has "
            f"{len(levels) if _levels_given else len(reg_iterations)}"
        )
    if grad_step is None:
        grad_step = _defaults["grad_step"]
    if flow_sigma is None:
        flow_sigma = _defaults["flow_sigma"]
    if inverse_schedule is None:
        inverse_schedule = _defaults["inverse_schedule"]
    if in_loop_inverse_steps is None:
        in_loop_inverse_steps = _defaults["in_loop_inverse_steps"]
    if inv_refresh_every is None:
        inv_refresh_every = _defaults["inv_refresh_every"]
    if polish_inv_steps is None:
        polish_inv_steps = _defaults["polish_inv_steps"]
    if polish_inv_tol is None:
        polish_inv_tol = _defaults["polish_inv_tol"]
    if compose_polish_tol is None:
        compose_polish_tol = _defaults["compose_polish_tol"]
    if pyramid is None:
        pyramid = _defaults["pyramid"]
    if stationary_boundary is None:
        stationary_boundary = _defaults["stationary_boundary"]
    if digital_diffeomorphism is None:
        digital_diffeomorphism = _defaults["digital_diffeomorphism"]
    if gaussian_sigma_mode is None:
        gaussian_sigma_mode = _defaults["gaussian_sigma_mode"]
    if conservative_smooth is None:
        conservative_smooth = _defaults["conservative_smooth"]
    if sobolev_alpha is None:
        sobolev_alpha = _defaults["sobolev_alpha"]

    from .core.utils import require_finite_images
    require_finite_images(fixed=fixed, moving=moving)

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
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {_OPTIMIZERS}")
    if optimizer == "syn_reg_adam" and regularizer in ("gaussian", "bspline"):
        raise ValueError(
            "optimizer='syn_reg_adam' reproduces syntx.syn()'s reg_adam for the spectral and continuum "
            "regularizers only ('sobolev', 'dsti', 'navier', ...); syntx smooths the Adam quotient "
            "differently for 'gaussian'/'bspline'"
        )
    if isinstance(metric_chunks, bool) or not isinstance(metric_chunks, int) or metric_chunks < 1:
        raise ValueError("metric_chunks must be a positive integer")
    if (metric_chunks > 1 and multivariate_extras is None and syn_metric not in _CC_METRICS
            and type_of_transform in _SYN_TRANSFORM_TYPES):
        raise ValueError("metric_chunks > 1 requires a CC-family syn_metric (cc, lncc, cc2, lncc2)")
    if affine_smoothing_sigmas_units not in ("mm", "vox"):
        raise ValueError("affine_smoothing_sigmas_units must be 'mm' or 'vox'")
    if pyramid not in _PYRAMIDS:
        raise ValueError(f"pyramid must be one of {_PYRAMIDS}")
    if inverse_schedule not in ("per_iteration", "refresh", "end_of_level"):
        raise ValueError("inverse_schedule must be 'per_iteration', 'refresh' or 'end_of_level'")
    _refresh = tuple(inv_refresh_every) if isinstance(inv_refresh_every, (list, tuple)) else (inv_refresh_every,)
    if len(_refresh) == 0 or any(isinstance(v, bool) or int(v) != v or int(v) < 1 for v in _refresh):
        raise ValueError("inv_refresh_every must be a positive integer or a sequence of positive integers")
    if int(in_loop_inverse_steps) < 0 or int(polish_inv_steps) < 0 or int(compose_polish_steps) < 0:
        raise ValueError("in_loop_inverse_steps, polish_inv_steps and compose_polish_steps must be >= 0")
    if float(polish_inv_tol) <= 0:
        raise ValueError("polish_inv_tol must be > 0")
    if compose_polish_tol is not None and float(compose_polish_tol) < 0:
        raise ValueError("compose_polish_tol must be >= 0 (0 disables the post-composition polish)")
    if end_of_level_inverse_steps < 1:
        raise ValueError("end_of_level_inverse_steps must be >= 1")
    if update_field_mesh_size_at_base_level is not None and update_field_mesh_size_at_base_level < 0:
        raise ValueError("update_field_mesh_size_at_base_level must be >= 0")
    if total_field_mesh_size_at_base_level < 0:
        raise ValueError("total_field_mesh_size_at_base_level must be >= 0")
    if update_field_spline_distance is not None and update_field_mesh_size_at_base_level is not None:
        raise ValueError(
            "update_field_spline_distance cannot be combined with "
            "update_field_mesh_size_at_base_level"
        )
    if total_field_spline_distance is not None and total_field_mesh_size_at_base_level != 0:
        raise ValueError(
            "total_field_spline_distance cannot be combined with a non-default "
            "total_field_mesh_size_at_base_level"
        )
    # update_field_mesh_size_at_base_level == 0 only when explicitly set to 0
    # (None, the default, resolves to an active 26mm spline distance below --
    # see the docstring's "Defaults" paragraph -- so it is never "off" unless
    # the caller explicitly asks for that).
    if (
        regularizer == "bspline"
        and update_field_mesh_size_at_base_level == 0
        and total_field_mesh_size_at_base_level == 0
        and update_field_spline_distance is None
        and total_field_spline_distance is None
    ):
        raise ValueError(
            "regularizer='bspline' requires an active update field (the default, or "
            "update_field_mesh_size_at_base_level > 0, or update_field_spline_distance) "
            "and/or total_field_mesh_size_at_base_level > 0 / total_field_spline_distance"
        )
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    intensity_normalization = _resolve_intensity_normalization(intensity_normalization) or False
    if len(levels) != len(reg_iterations):
        raise ValueError("levels and reg_iterations must have the same length")

    if smoothing_sigmas is not None:
        smoothing_sigmas = tuple(smoothing_sigmas)
        if len(smoothing_sigmas) != len(levels) or any(not math.isfinite(v) or v < 0 for v in smoothing_sigmas):
            raise ValueError("smoothing_sigmas must contain one finite nonnegative value per level")

    dimension = fixed.dimension
    if device is None:
        # Respect an explicit antstorch.set_default_device(); otherwise fall through to auto-detection below.
        from ..utilities.device_manager import get_user_default_device
        device = get_user_default_device()
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
    # Whether this call will run _fit_affine_from_ants() -> bspline_flows.
    # affine_registration() -- the one grid_sample-needing-backward code
    # path in this function that is NOT (as of this writing) routed through
    # AnalyticalGridSample, so it genuinely needs the native MPS backward
    # kernel. It runs exactly when no initial_affine is supplied and
    # type_of_transform requests a linear fit or the default 'SyN' (mirrors
    # the branching below, at "# --- Affine stage ---"); 'SyNOnly' (with or
    # without initial_affine) never reaches it.
    needs_internal_affine_fit = initial_affine is None and (
        type_of_transform in _LINEAR_TRANSFORM_TYPES or type_of_transform == "SyN"
    )
    # use_analytical_gradients=False differentiates the dense-SyN image sampling through the native
    # grid_sample backward (aten::grid_sampler_{2,3}d_backward), which MPS lacks on the PyTorch
    # builds tested (and, for 2-D, on some builds). The forward-only probes below assume the
    # AnalyticalGridSample path, so this case is checked first, for 2-D and 3-D alike.
    needs_sampler_backward = (
        not use_analytical_gradients and type_of_transform in _SYN_TRANSFORM_TYPES and sum(reg_iterations) > 0
    )
    if (resolved_device.type == "mps" and needs_sampler_backward
            and not _mps_grid_sample_backward_available(dimension)):
        warnings.warn(
            f"syn_registration: use_analytical_gradients=False needs the {dimension}-D "
            f"torch.nn.functional.grid_sample backward MPS kernel "
            f"(aten::grid_sampler_{dimension}d_backward), which this PyTorch build lacks, so "
            "falling back to CPU for this call. Keep use_analytical_gradients=True (the default: "
            "its hand-written backward only needs the forward kernel) to stay on MPS, or set "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 to let PyTorch fall back op by op.",
            RuntimeWarning,
        )
        resolved_device = torch.device("cpu")
    if resolved_device.type == "mps" and dimension == 3:
        if needs_internal_affine_fit and not mps_grid_sample_3d_available():
            # mps_grid_sample_3d_available() probes forward AND backward: PyTorch
            # PR #160541 (merged 2025-08-15, shipped starting with the 2.9.0
            # stable release) added a native MPS kernel for grid_sampler_3d
            # *forward*, closing the crash the backward-pass skip above doesn't
            # touch (e.g. 'SyNOnly' / no-grad calls) -- but no MPS kernel for
            # grid_sampler_3d_backward has been found as of this writing (see
            # https://github.com/pytorch/pytorch/issues/160237). The internal
            # affine fit (bspline_flows.affine_registration()) backpropagates
            # through a raw torch.nn.functional.grid_sample call every
            # iteration and genuinely needs that backward kernel -- there is
            # no way to make it work on MPS while it is missing, so silently
            # trying anyway would only replace this clear message with a
            # cryptic one.
            warnings.warn(
                "syn_registration: 3-D torch.nn.functional.grid_sample is missing its "
                "forward and/or backward MPS kernel in this PyTorch build "
                "(aten::grid_sampler_3d / grid_sampler_3d_backward; forward shipped in "
                "PyTorch 2.9.0 via https://github.com/pytorch/pytorch/pull/160541, but see "
                "https://github.com/pytorch/pytorch/issues/160237 for the backward gap). "
                "The internal affine fit needs the missing backward kernel, so falling "
                "back to CPU for this call. Pass initial_affine explicitly (skipping the "
                "internal fit) or use type_of_transform='SyNOnly' to avoid this, or set "
                "PYTORCH_ENABLE_MPS_FALLBACK=1 instead if you would rather PyTorch itself "
                "fall back transparently for every unimplemented MPS op.",
                RuntimeWarning,
            )
            resolved_device = torch.device("cpu")
        elif not needs_internal_affine_fit and not mps_grid_sample_3d_forward_available():
            # No internal affine fit runs (initial_affine was supplied, or
            # type_of_transform='SyNOnly' with none), so the only grid_sample
            # calls left are _fit_syn_level()'s, via
            # prepare_mid_images_and_gradients_torch() -> grid_sample_nd():
            # image/grid there always satisfy AnalyticalGridSample's dispatch
            # condition (grid requires grad -- it derives from the warp being
            # optimized; the fixed/moving image content never does), so its
            # hand-written backward runs instead of differentiating through
            # the native op -- and that backward itself only ever calls
            # grid_sample in plain forward mode (sampling the source image's
            # own spatial gradient), never grid_sampler_3d_backward. Only
            # forward needs to work, which is a strictly weaker (so more
            # often satisfied) requirement than the combined probe above; see
            # mps_grid_sample_3d_forward_available()'s docstring and the
            # project doc, "implémentation de grid_sampler_3d_backward pour
            # MPS". Only reached if even forward is missing (a PyTorch build
            # older than 2.9.0, or MPS unavailable).
            warnings.warn(
                "syn_registration: 3-D torch.nn.functional.grid_sample has no forward MPS "
                "kernel in this PyTorch build (aten::grid_sampler_3d; shipped in PyTorch "
                "2.9.0 via https://github.com/pytorch/pytorch/pull/160541). Falling back to "
                "CPU. Set PYTORCH_ENABLE_MPS_FALLBACK=1 instead if you would rather PyTorch "
                "itself fall back transparently for every unimplemented MPS op.",
                RuntimeWarning,
            )
            resolved_device = torch.device("cpu")
    resolved_outprefix = outprefix if outprefix else default_outprefix()

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
            num_bins=affine_num_bins,
            shrink_factors=affine_shrink_factors,
            smoothing_sigmas=affine_smoothing_sigmas,
            smoothing_sigmas_units=affine_smoothing_sigmas_units,
            iterations=affine_iterations,
            learning_rate=affine_learning_rate,
            multi_start=affine_multi_start,
            center_of_mass_init=affine_center_of_mass_init,
            verbose=verbose,
            intensity_normalization=intensity_normalization,
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
                "affine_smoothing_sigmas_units": affine_smoothing_sigmas_units,
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
    I_full = ants_image_to_tensor(fixed, resolved_device, dtype, normalize=intensity_normalization)
    J_full = ants_image_to_tensor(moving, resolved_device, dtype, normalize=intensity_normalization)

    # Extra images share the respective primary image domains. MSE masks
    # retain their binary amplitudes; intensity metrics use the usual normalization.
    metric_terms_full = None
    metric_masks_full = None
    if multivariate_extras is not None or mask is not None or moving_mask is not None or primary_metric_weight != 1:
        from .label_registration import _validate_image_domain, _metric_name
        if not math.isfinite(primary_metric_weight) or primary_metric_weight < 0:
            raise ValueError("primary_metric_weight must be finite and nonnegative")
        if syn_metric == "box_cc2" and primary_metric_weight and (mask is not None or moving_mask is not None):
            raise ValueError("box_cc2 does not support metric masks; use cc2 instead")
        metric_terms_full = []
        if primary_metric_weight:
            metric_terms_full.append((syn_metric, I_full, J_full, primary_metric_weight,
                                      neighborhood_radius if syn_metric not in ("mi", "mattes") else num_bins))
        # Each distinct metric image is validated and converted once, so the
        # 'label_msq' terms of one label map share a single tensor per side.
        converted = {}

        def metric_tensor(image, reference, role, normalize):
            key = (id(image), normalize)
            if key not in converted:
                _validate_image_domain(image, reference, role)
                converted[key] = (image, ants_image_to_tensor(image, resolved_device, dtype, normalize=normalize))
            return converted[key][1]

        for name, a, b, weight, parameter in (multivariate_extras or []):
            name = _metric_name(name)
            if not math.isfinite(weight) or weight < 0 or not math.isfinite(parameter) or parameter < 0:
                raise ValueError("metric weights and parameters must be finite and nonnegative")
            if name in ("cc", "lncc", "cc2", "lncc2", "mi", "mattes", "label_msq") and parameter != int(parameter):
                raise ValueError("Metric radius/bin count/label value must be an integer")
            if name == "label_msq" and parameter > 2**24:
                raise ValueError("label_msq label values must be representable exactly in float32 (<= 2^24)")
            if name in ("mi", "mattes") and parameter < 2:
                raise ValueError("MI metrics require at least two histogram bins")
            normalize = intensity_normalization if name not in ("mse", "label_msq") else False
            tensor_a = metric_tensor(a, fixed, "fixed metric image", normalize)
            tensor_b = metric_tensor(b, moving, "moving metric image", normalize)
            if weight:
                metric_terms_full.append((name, tensor_a, tensor_b, weight, parameter))
        if not metric_terms_full:
            raise ValueError("At least one metric must have a positive weight")
        masks = []
        for value, reference in ((mask, fixed), (moving_mask, moving)):
            if value is not None:
                _validate_image_domain(value, reference, "metric mask")
                value = ants_image_to_tensor(value != 0, resolved_device, dtype, normalize=False)
            masks.append(value)
        metric_masks_full = masks

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
    if regularizer == "bspline":
        need_domain = (
            update_field_spline_distance is not None
            or total_field_spline_distance is not None
            or update_field_mesh_size_at_base_level is None
        )
        if need_domain:
            from antstorch.bspline_flows import mesh_size_for_spline_distance
            from antstorch.bspline_flows.bspline_svf_registration import DEFAULT_BSPLINE_SPLINE_DISTANCE_MM

            fixed_domain_full = image_domain_from_metadata(fixed_meta_full)
            if update_field_spline_distance is not None:
                resolved_update_field_mesh_size_at_base_level = mesh_size_for_spline_distance(
                    fixed_domain_full, update_field_spline_distance
                )
            elif update_field_mesh_size_at_base_level is None:
                # Neither update_field_mesh_size_at_base_level nor
                # update_field_spline_distance given: default to a physical
                # spline distance (see the docstring's "Defaults" paragraph),
                # not ITK's literal mesh_size=1 this used before 2026-08-26.
                resolved_update_field_mesh_size_at_base_level = mesh_size_for_spline_distance(
                    fixed_domain_full, DEFAULT_BSPLINE_SPLINE_DISTANCE_MM
                )
            if total_field_spline_distance is not None:
                resolved_total_field_mesh_size_at_base_level = mesh_size_for_spline_distance(
                    fixed_domain_full, total_field_spline_distance
                )
    elif update_field_mesh_size_at_base_level is None:
        # Non-bspline regularizer: this parameter is never consumed, but
        # normalize None to the harmless literal 0 rather than leaving None,
        # so downstream code (provenance in particular) always sees a plain
        # int/tuple, never a bare None, for this field.
        resolved_update_field_mesh_size_at_base_level = 0

    window_size = 2 * int(neighborhood_radius) + 1
    num_levels = len(levels)

    # Guard against a degenerate pyramid level before any per-level fitting
    # runs, for every regularizer/optimizer combination alike -- see
    # _validate_pyramid_levels's docstring for why this is not specific to
    # 'dsti'/'sobolev' even though that is where it was first observed to
    # silently produce dice=0.0.
    if metric_terms_full is not None:
        window_size = max([1] + [2 * int(term[4]) + 1 for term in metric_terms_full
                                  if term[0] in ("cc", "lncc", "cc2", "lncc2", "box_cc2")])
    _fixed_level_metas = [
        _downsample_metadata(fixed_meta_full, int(f), syntx_parity) if pyramid == "legacy" else
        dict(fixed_meta_full, torch_shape=tuple(_deformable_plan(fixed_meta_full, f)["shape_zyx"]),
             spacing=tuple(_deformable_plan(fixed_meta_full, f)["spacing_xyz"]))
        for f in levels
    ]
    _validate_pyramid_levels(fixed_meta_full["torch_shape"], levels, window_size, syntx_parity,
                             level_shapes=[m["torch_shape"] for m in _fixed_level_metas])
    # syntx.pyramid.cfl_shrink_ratio: min over axes of (finest level spacing / level spacing).
    _finest_spacing = _fixed_level_metas[-1]["spacing"]
    _cfl_ratios = [min(float(fs) / float(cs) for fs, cs in zip(_finest_spacing, m["spacing"])) for m in _fixed_level_metas]
    # syntx: inv_tolerance defaults to 0.1 * min(full-resolution spacing); used as the in-loop
    # max-error threshold (voxels) of the 'refresh' schedule, the mean threshold being 1 % of it.
    resolved_inv_tolerance = (0.1 * float(min(fixed_meta_full["spacing"])) if inv_tolerance is None
                              else float(inv_tolerance))

    warp_l2r = warp_r2l = warp_l2r_inv = warp_r2l_inv = None
    level_loss_history = []
    for level_index, (factor, iteration_count) in enumerate(zip(levels, reg_iterations)):
        sigma = None if smoothing_sigmas is None else smoothing_sigmas[level_index]
        I_level, fixed_meta_level = _downsample_image(I_full, fixed_meta_full, factor, syntx_parity, sigma, pyramid=pyramid)
        J_level, moving_meta_level = _downsample_image(J_full, moving_meta_full, factor, syntx_parity, sigma, pyramid=pyramid,
                                                       reference_spacing=fixed_meta_full["spacing"])
        if warp_l2r is None:
            zeros = torch.zeros(
                (1,) + fixed_meta_level["torch_shape"] + (dimension,), device=resolved_device, dtype=dtype
            )
            warp_l2r, warp_r2l = zeros.clone(), zeros.clone()
            if inverse_schedule in ("per_iteration", "refresh"):
                warp_l2r_inv, warp_r2l_inv = zeros.clone(), zeros.clone()
            del zeros
        else:
            warp_l2r = _upsample_field(warp_l2r, fixed_meta_level["torch_shape"])
            warp_r2l = _upsample_field(warp_r2l, fixed_meta_level["torch_shape"])
            # 'end_of_level' never reads the inverses before the last level,
            # which restarts from -W: they are None until then, and are not
            # carried (or upsampled) across levels.
            if inverse_schedule in ("per_iteration", "refresh"):
                warp_l2r_inv = _upsample_field(warp_l2r_inv, fixed_meta_level["torch_shape"])
                warp_r2l_inv = _upsample_field(warp_r2l_inv, fixed_meta_level["torch_shape"])

        if verbose:
            message = (
                f"SyN resolution level {level_index + 1}/{num_levels}: shrink_factor={factor}, "
                f"fixed_size={fixed_meta_level['torch_shape']}, moving_size={moving_meta_level['torch_shape']}, "
            )
            if regularizer == "bspline":
                # Mirrors bspline_svf_registration()'s own verbose control-point
                # reporting (antstorch/bspline_flows/bspline_svf_registration.py) -- same
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

        metric_terms_level = None
        metric_masks_level = None
        if metric_terms_full is not None:
            metric_terms_level = []
            full_resolution = factor == 1 and (sigma is None or sigma == 0)
            for name, a, b, weight, parameter in metric_terms_full:
                if name == "label_msq":
                    if full_resolution:
                        # Masks are rebuilt from the label maps on the fly.
                        metric_terms_level.append((name, a, b, weight, parameter))
                        continue
                    # Coarser levels cache each label's downsampled mask (the
                    # full-resolution mask exists only while it is reduced).
                    a = (a == parameter).to(dtype)
                    b = (b == parameter).to(dtype)
                    name = "mse"
                metric_terms_level.append((name,
                    _downsample_image(a, fixed_meta_full, factor, syntx_parity, sigma, pyramid=pyramid)[0],
                    _downsample_image(b, moving_meta_full, factor, syntx_parity, sigma, pyramid=pyramid,
                                      reference_spacing=fixed_meta_full["spacing"])[0], weight, parameter))
                a = b = None
            metric_masks_level = [None if value is None else F.interpolate(
                value, size=meta["torch_shape"], mode="nearest")
                for value, meta in zip(metric_masks_full, (fixed_meta_level, moving_meta_level))]

        # Hand the level-start fields over in a list that _fit_syn_level()
        # empties, so this frame keeps no reference to them during the level.
        level_warps = [warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv]
        warp_l2r = warp_r2l = warp_l2r_inv = warp_r2l_inv = None
        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, history = _fit_syn_level(
            I_level, J_level, fixed_meta_level, moving_meta_level,
            level_warps, M_phys, t_phys,
            iterations=iteration_count,
            metric_terms=metric_terms_level,
            metric_masks=metric_masks_level,
            metric_chunks=metric_chunks,
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
            inverse_schedule=inverse_schedule,
            end_of_level_inverse_steps=end_of_level_inverse_steps,
            inv_refresh_every=int(_refresh[min(level_index, len(_refresh) - 1)]),
            inv_tolerance=resolved_inv_tolerance,
            polish_inv_steps=int(polish_inv_steps),
            polish_inv_tol=float(polish_inv_tol),
            syntx_parity=syntx_parity,
            cfl_shrink_ratio=_cfl_ratios[level_index],
            stationary_boundary=bool(stationary_boundary),
            use_analytical_gradients=use_analytical_gradients,
            sobolev_alpha=sobolev_alpha,
            verbose=verbose,
            level_index=level_index,
            num_levels=num_levels,
            gaussian_sigma_mode=gaussian_sigma_mode,
            conservative_smooth=conservative_smooth,
            optimizer=optimizer,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            optimizer_lr=optimizer_lr,
            regadam_grad_sigma=regadam_grad_sigma,
            regadam_quotient_sigma=regadam_quotient_sigma,
            regularizer_kwargs=regularizer_kwargs,
        )
        level_loss_history.append(history)

    loss_history = [value for level in level_loss_history for value in level]

    # --- Final algebraic forward/inverse construction (half-warp swap) ------
    fixed_meta_t_full = metadata_tensors_from_dict(fixed_meta_full, resolved_device, dtype)
    moving_meta_t_full = metadata_tensors_from_dict(moving_meta_full, resolved_device, dtype)
    X_phys_full = _physical_grid(fixed_meta_full, resolved_device, dtype)

    _run_finish = sum(reg_iterations) > 0
    _dd_stats = {}
    if digital_diffeomorphism and _run_finish:
        # syntx >= 6.36 (default-on digital diffeomorphism finish, syntx.syn's ``_dd_on`` path) and 6.37.2
        # (midpoint closure): each half-warp pair is made a digital diffeomorphism BEFORE the composition
        # (central-difference det(I + D_2h u) >= digital_diffeomorphism_floor and chord ICE <= ice_tol on the
        # sparse defect neighbourhood; bitwise untouched when nothing violates), the composed pair is LM
        # polished and then enforced again. syntx also pulls the final correction back onto its midpoint
        # half-fields (pullback_digital_diffeomorphism_to_halves); antstorch exports no midpoint field, so the
        # written transforms are the same without it.
        from .core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
        _dd_ice_tol = float(compose_polish_tol) if compose_polish_tol and float(compose_polish_tol) > 0 else 0.30
        _dd_kw = dict(spacing=fixed_meta_full["spacing"], origin=fixed_meta_full["origin"],
                      direction=fixed_meta_full["direction"], X_phys=X_phys_full, ice_tol=_dd_ice_tol,
                      jac_floor=float(digital_diffeomorphism_floor), steps=int(digital_diffeomorphism_steps),
                      stationary_boundary=bool(stationary_boundary))
        if stationary_boundary:
            _rim = get_boundary_mask(fixed_meta_full["torch_shape"], resolved_device, dtype)
            warp_l2r, warp_r2l = warp_l2r * _rim, warp_r2l * _rim
            warp_l2r_inv, warp_r2l_inv = warp_l2r_inv * _rim, warp_r2l_inv * _rim
        for _name, (_f, _i) in (("l2r", (warp_l2r, warp_l2r_inv)), ("r2l", (warp_r2l, warp_r2l_inv))):
            _st = {}
            _f, _i = enforce_digital_diffeomorphism_nd(_f, _i, stats=_st, **_dd_kw)
            _dd_stats[_name] = _st
            if _name == "l2r":
                warp_l2r, warp_l2r_inv = _f, _i
            else:
                warp_r2l, warp_r2l_inv = _f, _i
    syn_forward = _compose_fixed_grid(warp_l2r_inv, warp_r2l, X_phys_full, fixed_meta_t_full)
    syn_inverse = _compose_fixed_grid(warp_r2l_inv, warp_l2r, X_phys_full, fixed_meta_t_full)
    if digital_diffeomorphism and _run_finish and stationary_boundary:
        syn_forward, syn_inverse = syn_forward * _rim, syn_inverse * _rim
    if compose_polish_tol and float(compose_polish_tol) > 0 and compose_polish_steps > 0 and sum(reg_iterations) > 0:
        # syntx >= 6.35: safeguarded per-voxel Levenberg-Marquardt polish of the algebraic
        # (half-warp swap) inverse against the composed forward field, max error
        # compose_polish_tol voxels, mean min(compose_polish_tol / 2, 0.02).
        _cp_tol = float(compose_polish_tol)
        syn_inverse = update_inverse_field_nd_hybrid_lm(
            syn_forward, syn_inverse, steps=int(compose_polish_steps),
            max_error_threshold=_cp_tol, mean_error_threshold=min(_cp_tol * 0.5, 0.02),
            spacing=fixed_meta_full["spacing"], origin=fixed_meta_full["origin"],
            direction=fixed_meta_full["direction"], X_phys=X_phys_full, stationary_boundary=True,
        )
    if digital_diffeomorphism and _run_finish:
        _st = {}
        syn_forward, syn_inverse = enforce_digital_diffeomorphism_nd(syn_forward, syn_inverse, stats=_st, **_dd_kw)
        _dd_stats["composed"] = _st

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

    warpedmovout = grid_sample(
        ants_image_to_tensor(moving, resolved_device, dtype, normalize=False),
        physical_to_normalized_torch_cached(
            X_phys_full + total_forward,
            moving_meta_t_full["shape_t"], moving_meta_t_full["spacing_t"],
            moving_meta_t_full["origin_t"], moving_meta_t_full["direction_t"],
        ),
        mode="bilinear", padding_mode=padding_mode, align_corners=True,
    )
    warpedfixout = grid_sample(
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
            "smoothing_sigmas": smoothing_sigmas,
            "primary_metric_weight": primary_metric_weight,
            "multivariate_metrics": [] if metric_terms_full is None else [
                {"metric": term[0], "weight": term[3], "parameter": term[4]} for term in metric_terms_full],
            "metric_masks": (mask is not None, moving_mask is not None),
            "levels": tuple(levels),
            "reg_iterations": tuple(reg_iterations),
            "grad_step": grad_step,
            "flow_sigma": flow_sigma,
            "total_sigma": total_sigma,
            "regularizer": regularizer,
            "regularizer_kwargs": dict(regularizer_kwargs) if regularizer_kwargs else None,
            "optimizer": optimizer,
            "gaussian_sigma_mode": gaussian_sigma_mode,
            "conservative_smooth": conservative_smooth,
            "update_field_mesh_size_at_base_level": resolved_update_field_mesh_size_at_base_level,
            "total_field_mesh_size_at_base_level": resolved_total_field_mesh_size_at_base_level,
            "update_field_spline_distance": update_field_spline_distance,
            "total_field_spline_distance": total_field_spline_distance,
            "bspline_enforce_stationary_boundary": bspline_enforce_stationary_boundary,
            "antisymmetric": antisymmetric,
            "inverse_method": inverse_method,
            "metric_chunks": metric_chunks,
            "inverse_schedule": inverse_schedule,
            "pyramid": pyramid,
            "stationary_boundary": bool(stationary_boundary),
            "digital_diffeomorphism": bool(digital_diffeomorphism),
            "digital_diffeomorphism_floor": float(digital_diffeomorphism_floor),
            "digital_diffeomorphism_steps": int(digital_diffeomorphism_steps),
            "digital_diffeomorphism_stats": {
                k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in st.items()
                    if isinstance(vv, (int, float, bool))}
                for k, st in _dd_stats.items()
            },
            "level_shapes": [list(m["torch_shape"]) for m in _fixed_level_metas],
            "end_of_level_inverse_steps": end_of_level_inverse_steps,
            "in_loop_inverse_steps": int(in_loop_inverse_steps),
            "inv_refresh_every": list(_refresh),
            "inv_tolerance": float(resolved_inv_tolerance),
            "polish_inv_steps": int(polish_inv_steps),
            "polish_inv_tol": float(polish_inv_tol),
            "compose_polish_tol": float(compose_polish_tol or 0.0),
            "compose_polish_steps": int(compose_polish_steps),
            "syntx_parity": syntx_parity,
            "sobolev_alpha": sobolev_alpha,
            "syntx_defaults": syntx_defaults,
            "optimizer_lr": optimizer_lr,
            "use_analytical_gradients": use_analytical_gradients,
            "affine_fit": affine_result is not None,
            "affine_smoothing_sigmas_units": affine_smoothing_sigmas_units,
            "device": str(resolved_device),
            "outprefix": resolved_outprefix,
        },
    }
    return result
