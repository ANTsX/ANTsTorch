"""Greedy (one-directional) diffeomorphic registration -- ``antstorch.greedy``.

Port of ``syntx.greedy`` (``syntx/greedy.py``): a single forward displacement field ``u`` in the
fixed image's normalised ``[-1, 1]`` coordinates, refined at every iteration by composition with a
small smoothed update ``v``: ``(Id + u_new) = (Id + u) o (Id + v)`` -- the ANTs "Greedy SyN" scheme
without the symmetric midpoint. Each iteration costs two resamplings (warped image, field
composition), so it is much faster than :func:`antstorch.syn.syn_registration`. There is no inverse
unless ``return_inverse=True`` (then computed afterwards by an Anderson fixed-point solve).

Per iteration, per pyramid level (the same recipe as ``syntx.greedy``):

1. resample the moving image through ``affine o (Id + u)`` and evaluate the similarity loss against the
   fixed image (local cross-correlation ``cc2`` by default, gradient by autograd);
2. smooth the gradient (Gaussian ``flow_sigma`` in voxels), take a bias-corrected Adam step (moments
   reset at every level), optionally smooth the Adam quotient (``optimizer='regadam'``);
3. bound the step to ``learning_rate`` voxels-of-the-normalised-grid, then compose
   ``u_new(x) = v(x) + u(x + v(x))`` and smooth the whole field (Gaussian ``total_sigma``).

The initial affine (``antstorch.syn.robust_affine`` unless given) seeds the optimisation. The result is
exported as the standard ANTs transform list ``[warp, affine]`` (``[warp]`` if ``initial_transform`` is
``False``), so ``ants.apply_transforms(fixed, moving, reg['fwdtransforms'])`` applies the composed
transform in one interpolation. The affine is *not* baked into the displacement field.

Differences from ``syntx.greedy`` (deliberate, documented):

* only the ``'gaussian'`` regularizer is supported (``syntx.greedy`` documents the same restriction); the
  similarity metrics are the local cross-correlation family (``cc2`` / ``lncc`` / ``cc`` / ``ncc``) and
  the mean squared error (``mse`` / ``l2``);
* on CUDA / MPS the differentiable resampling uses ``F.grid_sample`` (``antstorch._torch_compat`` on MPS)
  instead of syntx's ``DeterministicGridSample``; on CPU the two are identical, on GPUs the backward pass
  may differ from syntx's at floating-point round-off level.

Multi-resolution pyramid: the 'deformable' pyramid of syntx (``syntx.pyramid``, syntx >= 6.2: levels isotropic in millimetres
from the images' voxel spacing, corner-aligned linear resampling after an anti-aliasing Gaussian, native grid at the finest
level, 16-voxel floor), ported here for that preset only. The fixed and the moving image each get their own grid; the field
lives on the fixed level grid. Earlier syntx versions (< 6.2) shrank every axis by the same factor and are not reproduced.

Intensity normalisation follows ``syntx.greedy``: ``normalize_image(method='auto')`` on both images.
"""

import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._torch_compat import grid_sample
from .ants_transform_io import write_affine_transform
from .syn.core.affine import parse_ants_affine
from .syn.core.convergence import DEFAULT_SLOPE_THRESHOLD, LossSlopeStop
from .syn.core.inverse import update_inverse_field_nd_anderson
from .syn.core.losses import _box_pool, box_mean_nd, local_ncc_loss_nd
from .syn.core.optimizers import FieldStepOptimizer, validate_regularizer_optimizer_pair
from .syn.core.regularizers import REGULARIZER_EXTRA_KEYS, build_level_regularizer, list_regularizers
from .syn.core.pipeline import auto_detect_device, cleanup_gpu
from .syn.core.smoothing import apply_sobolev_green_operator, separable_1d_filter
from .syn.core.utils import normalize_image
from .syn.robust_affine import robust_affine

__all__ = ["GreedyRegistrationModel", "greedy_registration", "greedy"]


# ----------------------------------------------------------------------------------------
# Small tensor helpers (the same numerics as syntx.core.grid / syntx.core.smoothing)
# ----------------------------------------------------------------------------------------
def _gaussian_1d_compact(sigma: float, truncated: float = 2.0, device="cpu",
                         dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    """1-D Gaussian kernel integrated over unit bins (erf differences), normalised to sum 1.

    Radius ``round(max(sigma * truncated, 0.5))`` (at least one tap each side). ``sigma`` in voxels.
    Returns ``None`` if ``sigma <= 0``.
    """
    if sigma <= 0.0:
        return None
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=dtype, device=device)
    t = 0.70710678 / float(sigma)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf()).clamp(min=0)
    return out / out.sum()


def _sample_field_cf(field_cf: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear",
                     padding_mode: str = "border") -> torch.Tensor:
    """Sample a channels-first field at normalised positions (``align_corners=True``); returns channels-last."""
    sampled_cf = grid_sample(field_cf, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return torch.movedim(sampled_cf, 1, -1)


def _compose_grids(grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
    """``grid1(grid2(x))`` for two normalised coordinate grids (B, *spatial, dim): bilinear, border padding."""
    return _sample_field_cf(torch.movedim(grid1, -1, 1), grid2, mode="bilinear", padding_mode="border")


def _resize_field(field: torch.Tensor, size, mode: Optional[str] = None) -> torch.Tensor:
    """Resize a channels-last field to ``size`` (tensor order) with ``F.interpolate`` (``align_corners=True``)."""
    dim = field.shape[-1]
    if mode is None:
        mode = "trilinear" if dim == 3 else "bilinear"
    resized_cf = F.interpolate(torch.movedim(field, -1, 1), size=size, mode=mode, align_corners=True)
    return torch.movedim(resized_cf, 1, -1)


def _masked_mse(moved: torch.Tensor, fixed: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diff = moved - fixed
    sq = diff * diff
    if mask is None:
        return torch.mean(sq)
    m = mask.to(sq.dtype)
    return torch.sum(sq * m) / torch.clamp_min(torch.sum(m), 1e-8)


def _build_loss(similarity_metric: str, window_size: int, squared: bool):
    """Loss callable ``loss(moved, fixed, mask=None)`` for ``similarity_metric`` (lower is better)."""
    metric = similarity_metric.lower()
    if metric in ("cc2", "lncc2", "cc_squared"):
        cc_squared = True
    elif metric in ("lncc", "cc", "ncc"):
        cc_squared = bool(squared)
    elif metric in ("mse", "l2"):
        if metric == "l2" and not squared:
            raise ValueError("antstorch.greedy: similarity_metric='l2' with squared=False is not supported")
        return _masked_mse
    else:
        raise ValueError(
            f"antstorch.greedy: unsupported similarity_metric {similarity_metric!r}; expected one of "
            "'cc2', 'lncc', 'cc', 'ncc', 'mse', 'l2'")

    def _cc(moved, fixed, mask=None):
        return local_ncc_loss_nd(moved, fixed, mask=mask, window_size=window_size, squared=cc_squared,
                                 use_ants_pseudo_gradient=False)

    # syntx >= 6.40: for 'cc2' / 'lncc' the fit loop caches the fixed image's local moments once per level
    # (the same formula as local_ncc_loss_nd, evaluated with the fixed-side terms precomputed).
    _cc.fast_lncc = metric in ("cc2", "lncc")
    _cc.squared = cc_squared
    return _cc


# ----------------------------------------------------------------------------------------
# The "deformable" image pyramid of syntx (syntx.pyramid, >= v6.2: plan_levels + build_pyramid, preset 'deformable')
# ----------------------------------------------------------------------------------------
_MIN_VOXELS_PER_AXIS = 16    # per-axis floor of the 'deformable' preset (an axis shorter than this keeps its native sampling)
_HARD_MIN_VOXELS = 2         # never fewer than 2 samples on an axis of >= 2 samples
_PYR_EPS = 1e-6


class _Level:
    """One pyramid level: ``tensor`` (B, C, *shape), ``shape`` (tensor order z, y, x) and the level grid
    ``spacing`` (ITK x, y, z order; used by the physical-space regularizers, syntx >= 6.42)."""
    __slots__ = ("tensor", "shape", "spacing")

    def __init__(self, tensor: torch.Tensor, shape: Tuple[int, ...], spacing: Optional[Tuple[float, ...]] = None):
        self.tensor, self.shape, self.spacing = tensor, shape, spacing


def _plan_axes_deformable(n_xyz, sp, t, floor_eff):
    """Per axis (native size n, spacing s, target mm t) -> (m, f, mode): corner-aligned ('deformable' preset), the first and
    last samples stay on the first and last native samples, ``m = round((n - 1) s / t) + 1``, ``f = (n - 1) / (m - 1)``;
    an axis that is not coarsened (``t <= s``) is left as it is. Never below ``min(floor, n)`` voxels."""
    out = []
    for n, s in zip(n_xyz, sp):
        if n < _HARD_MIN_VOXELS:
            out.append((n, 1.0, "identity"))
            continue
        coarser = t > s * (1.0 + _PYR_EPS)
        if not coarser:
            out.append((n, 1.0, "identity"))
            continue
        m = max(max(2, int(round((n - 1) * s / t)) + 1), min(floor_eff, n))
        f = (n - 1.0) / (m - 1.0)
        if m == n and abs(f - 1.0) < 1e-12:
            out.append((n, 1.0, "identity"))
            continue
        out.append((m, f, "linear"))
    return out


def _plan_levels_deformable(shape_zyx, spacing_xyz, levels, reference_spacing_xyz=None, sigma=None):
    """Plan every level of the 'deformable' pyramid (``syntx.pyramid.plan_levels(..., 'deformable')``): level L has the working
    spacing L x (finest spacing of the reference or of the image), isotropic in millimetres; default smoothing
    sigma = 0.5 x the axis's coarsening factor (native voxels) on every coarsened axis. ``sigma`` (a float or one value per
    level, native voxels, syntx ``sigma_mode='voxel'``) replaces that rule; an axis that is not coarsened is never smoothed.
    Returns a list of dicts (``spacing_xyz``: the level grid spacing, ``f * s`` per axis)."""
    n_zyx = tuple(int(x) for x in shape_zyx)
    sp = tuple(float(x) for x in spacing_xyz)
    ref = tuple(float(x) for x in reference_spacing_xyz) if reference_spacing_xyz is not None else sp
    n_xyz = n_zyx[::-1]
    levels = [float(x) for x in levels]
    if sigma is not None and not isinstance(sigma, (int, float)):
        sigma = [float(x) for x in sigma]
        if len(sigma) != len(levels):
            raise ValueError(f"sigma has {len(sigma)} entries for {len(levels)} levels")
    plans = []
    for li, L in enumerate(levels):
        t = max(L * min(ref), min(sp))
        axes = _plan_axes_deformable(n_xyz, sp, t, _MIN_VOXELS_PER_AXIS)
        m_xyz = tuple(a[0] for a in axes)
        f_xyz = tuple(a[1] for a in axes)
        modes = tuple(a[2] for a in axes)
        if sigma is None:
            sig_xyz = [0.5 * f if (md == "linear" and f > 1.0 + _PYR_EPS) else 0.0 for f, md in zip(f_xyz, modes)]
        else:
            sv = float(sigma if isinstance(sigma, (int, float)) else sigma[li])
            if sv < 0 or not math.isfinite(sv):
                raise ValueError(f"sigma must be finite and >= 0, got {sv}")
            sig_xyz = [sv] * len(f_xyz)
        sig_zyx = tuple(sg if (f > 1.0 + _PYR_EPS and sg > 0) else 0.0 for sg, f in zip(sig_xyz, f_xyz))[::-1]
        plans.append(dict(shape_zyx=tuple(m_xyz[::-1]), f_xyz=f_xyz, modes_xyz=modes, sigma_zyx=sig_zyx,
                          spacing_xyz=tuple(f * s for f, s in zip(f_xyz, sp)),
                          native_shape_zyx=n_zyx, is_identity=all(md == "identity" for md in modes)))
    return plans


def _linear_matrix(n_old: int, n_new: int, f: float) -> np.ndarray:
    """(n_new, n_old) linear interpolation weights of a signal sampled at centres to the positions ``j f`` (offset 0)."""
    W = np.zeros((n_new, n_old), dtype=np.float64)
    for j in range(n_new):
        c = min(max(j * f, 0.0), n_old - 1.0)
        i0 = int(np.floor(c))
        i1 = min(i0 + 1, n_old - 1)
        w1 = c - i0
        W[j, i0] += 1.0 - w1
        W[j, i1] += w1
    return W


def _smooth_replicate_cf(x: torch.Tensor, sig_zyx) -> torch.Tensor:
    """Per-axis ITK-style (Bessel) Gaussian of a channel-first tensor with replicate padding, axes with sigma <= 0 skipped:
    ``syntx.core.smoothing.separable_gaussian_filter(..., mode='replicate', kernel_type='bessel')``."""
    if not any(sg > 0 for sg in sig_zyx):
        return x
    from .syn.core.smoothing import _shifted_slice_blur_1d_inplace, get_cached_gaussian_kernel_1d
    v = x
    device, dtype = v.device, v.dtype
    dim = v.dim() - 2
    need_grad = bool(torch.is_grad_enabled() and v.requires_grad)
    w_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    # syntx >= 6.40 (separable_gaussian_filter): off MPS, a 3-D input without autograd is blurred by in-place
    # shifted-slice additions (same weights, no padded copy / conv3d workspace).
    if dim == 3 and device.type != "mps" and not need_grad:
        out = v
        for i, sig in enumerate(sig_zyx):
            if sig <= 0.0:
                continue
            weights = get_cached_gaussian_kernel_1d(sig, "cpu", w_dtype).view(-1).tolist()
            out = _shifted_slice_blur_1d_inplace(out, i + 2, weights, "replicate")
        return out if out.is_contiguous() else out.contiguous()
    if dim == 3 and device.type != "mps":
        C = v.shape[1]
        for i, sig in enumerate(sig_zyx):
            if sig <= 0.0:
                continue
            kernel_1d = get_cached_gaussian_kernel_1d(sig, device, dtype).squeeze(0)
            pad = kernel_1d.shape[-1] // 2
            if i == 0:
                k = kernel_1d.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), mode="replicate"), k, groups=C)
            elif i == 1:
                k = kernel_1d.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), mode="replicate"), k, groups=C)
            else:
                k = kernel_1d.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), mode="replicate"), k, groups=C)
        return v.contiguous()
    # 2-D and MPS: reshaped conv1d, one axis at a time (replicate padding)
    for i in range(dim):
        sig = sig_zyx[i]
        if sig <= 0.0:
            continue
        kernel = get_cached_gaussian_kernel_1d(sig, device, dtype)
        pad_size = kernel.shape[-1] // 2
        if device.type == "mps" and kernel.shape[-1] >= 33 and not need_grad:
            # syntx >= 6.40: avoid the multi-GB Metal conv1d im2col workspace of large kernels
            weights = get_cached_gaussian_kernel_1d(sig, "cpu", w_dtype).view(-1).tolist()
            v = _shifted_slice_blur_1d_inplace(v, i + 2, weights, "replicate")
            continue
        dims = list(range(v.ndim))
        dims[-1], dims[i + 2] = dims[i + 2], dims[-1]
        vp = v.permute(*dims).contiguous()
        vr = F.conv1d(F.pad(vp.view(-1, 1, vp.shape[-1]), (pad_size, pad_size), mode="replicate"), kernel)
        v = vr.view(*vp.shape).permute(*dims).contiguous()
    return v.contiguous()


def _apply_level(x: torch.Tensor, plan: dict) -> torch.Tensor:
    """Execute one planned level: smoothing, then one weight matrix per resampled axis (no ``F.interpolate``)."""
    x = _smooth_replicate_cf(x, plan["sigma_zyx"])
    if plan["is_identity"]:
        return x
    out = x
    for ax, (md, f, n_old, n_new) in enumerate(zip(plan["modes_xyz"][::-1], plan["f_xyz"][::-1],
                                                   plan["native_shape_zyx"], plan["shape_zyx"]), start=2):
        if md == "identity":
            continue
        W = torch.as_tensor(_linear_matrix(n_old, n_new, f), dtype=out.dtype, device=out.device)
        out = torch.movedim(torch.movedim(out, ax, -1) @ W.t(), -1, ax)
    return out.contiguous()


def _build_pyramid_deformable(image: torch.Tensor, spacing_xyz, levels, reference_spacing=None,
                              mask: Optional[torch.Tensor] = None):
    """Pyramid of ``image`` (B, C, *spatial) with ``spacing_xyz`` (x, y, z): a list of ``_Level`` (and of mask levels or None)."""
    plans = _plan_levels_deformable(tuple(image.shape[2:]), spacing_xyz, levels, reference_spacing)
    lv = [_Level(_apply_level(image, p), p["shape_zyx"], p["spacing_xyz"]) for p in plans]
    ml = ([_Level(_apply_level(mask.to(image.dtype), p), p["shape_zyx"], p["spacing_xyz"]) for p in plans]
          if mask is not None else None)
    return lv, ml


# ----------------------------------------------------------------------------------------
# Affine / displacement conversion (physical <-> normalised coordinates)
# ----------------------------------------------------------------------------------------
def _normalised_to_physical_matrix(image: "ants.ANTsImage") -> np.ndarray:
    """(dim+1, dim+1) matrix mapping the image's normalised ``[-1, 1]`` grid coordinates to physical (LPS)."""
    dim = image.dimension
    direction = np.array(image.direction).reshape(dim, dim)
    spacing = np.array(image.spacing)
    origin = np.array(image.origin)
    shape = np.array(image.shape)
    P = np.eye(dim + 1, dtype=np.float64)
    P[:dim, :dim] = direction @ np.diag(spacing) @ np.diag((shape - 1) / 2.0)
    P[:dim, dim] = direction @ (spacing * (shape - 1) / 2.0) + origin
    return P


def _build_torch_affine_matrix(fixed, moving, M_phys: np.ndarray, t_phys: np.ndarray,
                               device) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """``theta`` (1, dim, dim+1) mapping fixed normalised coordinates to moving normalised coordinates.

    ``y_norm = (P_M^-1 @ T_phys @ P_F) @ x_norm`` with ``P_*`` the normalised-to-physical matrices.
    """
    dim = fixed.dimension
    P_F = _normalised_to_physical_matrix(fixed)
    P_M = _normalised_to_physical_matrix(moving)
    T_phys = np.eye(dim + 1, dtype=np.float64)
    T_phys[:dim, :dim] = M_phys
    T_phys[:dim, dim] = t_phys
    T_norm = np.linalg.inv(P_M) @ T_phys @ P_F
    theta = torch.from_numpy(T_norm[:dim, :]).float().unsqueeze(0).to(device)
    return theta, P_F, P_M


def _apply_homogeneous(P: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``(P @ [points, 1]^T)^T`` for ``(N, dim)`` points and a ``(dim + 1, dim + 1)`` matrix.

    The result is the same as the plain matrix product. The floating-point error flags are silenced because
    NumPy 2.0.x on macOS (Accelerate BLAS) raises spurious "divide by zero / overflow / invalid value encountered in
    matmul" RuntimeWarnings on perfectly finite inputs; a genuine non-finite result is still caught downstream.
    """
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=-1)
    with np.errstate(all="ignore"):
        return (P @ homogeneous.T).T


def _ants_displacement_to_zyx(disp_img, device) -> torch.Tensor:
    """ANTs displacement image -> ``(1, *spatial, dim)`` tensor in tensor layout: spatial axes AND vector components
    reversed to (z, y, x) (``syntx.spatial.disp_itk_to_tensor``); the inverse of ``displacement_zyx_to_ants_image``."""
    arr = disp_img.numpy()
    dim = arr.shape[-1]
    arr = np.transpose(arr, (1, 0, 2)) if dim == 2 else np.transpose(arr, (2, 1, 0, 3))
    return torch.from_numpy(np.ascontiguousarray(arr[..., ::-1])).float().unsqueeze(0).to(device)


def _restrict_ants_displacement(disp_img, restrict_transformation):
    """Zero the displacement components (ITK x, y, z order) whose ``restrict_transformation`` weight is 0."""
    if restrict_transformation is None:
        return disp_img
    arr = disp_img.numpy()
    for i, w in enumerate(restrict_transformation):
        if w == 0.0:
            arr[..., i] = 0.0
    return ants.from_numpy(arr, origin=disp_img.origin, spacing=disp_img.spacing, direction=disp_img.direction,
                           has_components=True)


def _normalized_to_physical_disp(disp: torch.Tensor, shape, spacing, direction, dtype=torch.float32) -> torch.Tensor:
    """Normalised ``(x, y, z)``-component displacement -> tensor-layout physical one (``(z, y, x)`` components, mm)
    on the input's device: components reversed, scaled by ``(n - 1) / 2 * spacing`` per axis, rotated by the
    reversed direction matrix (``syntx.spatial.normalized_to_physical_disp``, syntx >= 6.40)."""
    dim = len(shape)
    device = disp.device
    disp_t = torch.flip(disp.to(dtype=dtype), dims=[-1])
    shape_t = torch.tensor(list(shape), device=device, dtype=dtype)
    spacing_t = torch.tensor(list(reversed([float(x) for x in spacing])), device=device, dtype=dtype)
    d = np.asarray(direction, dtype=np.float64).reshape(dim, dim)[::-1, ::-1].copy()
    direction_t = torch.tensor(d, device=device, dtype=dtype)
    scaled = disp_t * ((shape_t - 1.0) / 2.0 * spacing_t)
    return (scaled.reshape(-1, dim) @ direction_t.t()).reshape(disp_t.shape)


def _grid_to_ants_displacement(total_target_grid: torch.Tensor, fixed, P_F: np.ndarray, P_M: Optional[np.ndarray] = None,
                               restrict_transformation: Optional[Sequence[float]] = None):
    """Target coordinate grid (normalised) -> ANTs physical displacement image on the fixed grid (``y_phys - x_phys``)."""
    if P_M is None:
        P_M = P_F
    dim = fixed.dimension
    grid_np = total_target_grid[0].detach().cpu().numpy()
    if dim == 2:
        Y, X, _ = grid_np.shape
        grid_flat = grid_np.reshape(-1, 2)
        y_phys = _apply_homogeneous(P_M, grid_flat)[:, :2]
        mesh_y, mesh_x = np.meshgrid(np.linspace(-1, 1, Y), np.linspace(-1, 1, X), indexing="ij")
        fixed_norm_flat = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis=-1)
        x_phys = _apply_homogeneous(P_F, fixed_norm_flat)[:, :2]
        disp_itk = np.transpose((y_phys - x_phys).reshape(Y, X, 2), (1, 0, 2)).copy().astype(np.float32)
    elif dim == 3:
        Z, Y, X, _ = grid_np.shape
        grid_flat = grid_np.reshape(-1, 3)
        y_phys = _apply_homogeneous(P_M, grid_flat)[:, :3]
        mesh_z, mesh_y, mesh_x = np.meshgrid(np.linspace(-1, 1, Z), np.linspace(-1, 1, Y), np.linspace(-1, 1, X),
                                             indexing="ij")
        fixed_norm_flat = np.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], axis=-1)
        x_phys = _apply_homogeneous(P_F, fixed_norm_flat)[:, :3]
        disp_itk = np.transpose((y_phys - x_phys).reshape(Z, Y, X, 3), (2, 1, 0, 3)).copy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported spatial dimension: {dim}")
    if restrict_transformation is not None:
        for i, w in enumerate(restrict_transformation):
            if w == 0.0:
                disp_itk[..., i] = 0.0
    return ants.from_numpy(disp_itk, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction,
                           has_components=True)


# ----------------------------------------------------------------------------------------
# The optimisation model
# ----------------------------------------------------------------------------------------
class _Conv1x1AffineGrid(torch.autograd.Function):
    """``clamp(coords, -1, 1) @ M^T + t`` in closed form (syntx >= 6.40 ``greedy._Conv1x1AffineGrid``).

    Replaces the bilinear sampling of a stored ``affine_grid`` (same values: the affine is linear and the
    stored grid was sampled with border padding, i.e. at clamped coordinates) without keeping that
    ``(1, dim, *spatial)`` grid alive; 1x1 ``conv1d`` on MPS, ``addmm`` elsewhere. The backward pass
    zeroes the gradient of clamped (out-of-domain) coordinates, as the border-padded sampler did.
    """

    @staticmethod
    def forward(ctx, coords: torch.Tensor, M: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        in_bounds = (coords >= -1.0) & (coords <= 1.0)
        ctx.save_for_backward(M, in_bounds)
        dim = M.shape[0]
        c = coords.clamp(-1.0, 1.0)
        if coords.device.type == "mps":
            cf = c.view(1, -1, dim).transpose(1, 2).contiguous()
            out_cf = F.conv1d(cf, M.view(dim, dim, 1), t.view(dim))
            return out_cf.transpose(1, 2).contiguous().view_as(coords)
        return torch.addmm(t.view(1, dim), c.reshape(-1, dim), M.t()).view_as(coords)

    @staticmethod
    def backward(ctx, go: torch.Tensor):
        M, in_bounds = ctx.saved_tensors
        dim = M.shape[0]
        if go.device.type == "mps":
            go_cf = go.reshape(1, -1, dim).transpose(1, 2).contiguous()
            gin_cf = F.conv1d(go_cf, M.t().contiguous().view(dim, dim, 1))
            gin = gin_cf.transpose(1, 2).contiguous().view_as(go)
        else:
            gin = (go.reshape(-1, dim) @ M).view_as(go)
        return gin * in_bounds.to(gin.dtype), None, None


class GreedyRegistrationModel(nn.Module):
    """The tensor-level model behind :func:`greedy_registration` (``syntx.greedy.GreedyRegistrationModel``).

    ``fit`` returns the displacement field in normalised ``[-1, 1]`` coordinates, ``(1, *spatial, dim)``,
    tensor ``(z, y, x)`` spatial order, ``(x, y, z)`` components.

    Parameters
    ----------
    dim : {2, 3}
    learning_rate : float, default 0.5625
        Step size (``greedy_registration`` passes 1.5): with ``'cfl'`` the update's largest displacement is
        ``learning_rate`` voxels of the level grid, with ``'adam'`` / ``'reg_adam'``
        ``learning_rate / (grid size - 1)`` of the normalised domain.
    flow_sigma : float, default 1.8
        Gaussian sigma (voxels) smoothing the similarity gradient (``'gaussian'``); for the spectral / continuum
        regularizers, the natural scale when ``sobolev_alpha`` is None.
    total_sigma : float, default 0.28
        Gaussian sigma (voxels) smoothing the whole field after each update (0 = off).
    optimizer : {'cfl', 'adam', 'regadam' / 'reg_adam'} or None
        None: 'cfl' for the spectral / continuum regularizers, 'adam' otherwise (syntx >= 6.43,
        ``validate_regularizer_optimizer_pair``). 'regadam' additionally smooths the Adam quotient.
    regularizer : str, default 'gaussian'
        'gaussian' or any name of :func:`antstorch.syn.core.regularizers.list_regularizers`.
    sobolev_alpha : float, optional
        Spectral / continuum operator strength.
    cfl_momentum : float, default 0.85
        Velocity momentum of ``optimizer='cfl'``.
    extra_reg_kwargs : dict, optional
        Continuum operator parameters (``poisson_ratio``, ``s``, ...).
    regadam_sigma : float, default 0.8
    lncc_radius : int, default 2
        Local-correlation radius (window ``2 * radius + 1``).
    similarity_metric : str, default 'cc2'
        'cc2' / 'lncc' / 'cc' / 'ncc' (local correlation) or 'mse' / 'l2'.
    squared : bool, default True
        Squared local correlation for 'lncc' / 'cc' / 'ncc' ('cc2' is always squared).
    anderson, anderson_steps, anderson_m, anderson_freq
        Optional Anderson inverse-consistency projection of the field ('per_scale' or 'posthoc').
    padding_mode : str, default 'border'
    beta1, beta2, eps : Adam constants (0.9, 0.99, 1e-8).
    restrict_transformation : sequence of float, optional
        Per-axis weights in [0, 1] (XYZ order) multiplying the update.
    device : torch.device, optional
    """

    def __init__(self, dim: int = 3, learning_rate: float = 0.5625, flow_sigma: float = 1.8, total_sigma: float = 0.28,
                 optimizer: Optional[str] = None, regularizer: str = "gaussian", regadam_sigma: float = 0.8,
                 sobolev_alpha: Optional[float] = None, cfl_momentum: float = 0.85,
                 extra_reg_kwargs: Optional[Dict[str, Any]] = None,
                 lncc_radius: int = 2, similarity_metric: str = "cc2", squared: bool = True,
                 anderson: bool = False, anderson_steps: int = 5, anderson_m: int = 5,
                 anderson_freq: str = "per_scale", padding_mode: str = "border",
                 beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8,
                 restrict_transformation: Optional[Sequence[float]] = None,
                 device: Optional[torch.device] = None,
                 convergence_slope: Optional[float] = DEFAULT_SLOPE_THRESHOLD):
        super().__init__()
        # syntx >= 6.16: the shared deformable stopping rule (core.convergence.LossSlopeStop): the loss is read every
        # 5 iterations and a level ends once its least-squares slope over 10 samples is within +-convergence_slope.
        self.stop = LossSlopeStop(slope_threshold=convergence_slope)
        self.convergence: List[dict] = []
        self.dim = dim
        if restrict_transformation is not None:
            rt = [float(w) for w in restrict_transformation]
            if len(rt) != dim:
                raise ValueError(f"restrict_transformation must have length {dim} (one weight per physical axis, "
                                 f"XYZ order), got length {len(rt)}: {restrict_transformation!r}")
            if any(not (0.0 <= w <= 1.0) for w in rt):
                raise ValueError(f"restrict_transformation weights must be in [0, 1], got {rt!r}")
            self.restrict_transformation = tuple(rt)
        else:
            self.restrict_transformation = None
        reg = str(regularizer).lower()
        _valid_regs = set(list_regularizers()) | {"gaussian", "gauss", "sobolev"}
        if reg not in _valid_regs:
            raise ValueError(f"antstorch.greedy: unknown regularizer {regularizer!r}; expected one of {sorted(_valid_regs)}")
        self.learning_rate = learning_rate
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        # syntx >= 6.43: None -> 'cfl' for spectral/continuum regularizers, 'adam' otherwise; componentwise
        # Adam with a coupled continuum operator is refused (it breaks the div/curl coupling).
        self.optimizer = validate_regularizer_optimizer_pair(reg, optimizer, caller="GreedyRegistrationModel")
        if self.optimizer not in ("adam", "regadam", "reg_adam", "cfl"):
            raise ValueError(f"antstorch.greedy: optimizer must be 'adam', 'regadam' or 'cfl', got {optimizer!r}")
        self.regularizer = reg
        self.regadam_sigma = regadam_sigma
        self.sobolev_alpha = None if sobolev_alpha is None else float(sobolev_alpha)
        self.cfl_momentum = float(cfl_momentum)
        self.extra_reg_kwargs = dict(extra_reg_kwargs or {})
        self.lncc_radius = lncc_radius
        self.window_size = 2 * lncc_radius + 1
        self.similarity_metric = similarity_metric.lower()
        self.squared = squared
        self.anderson = anderson
        self.anderson_steps = anderson_steps
        self.anderson_m = anderson_m
        self.anderson_freq = anderson_freq.lower()
        self.padding_mode = padding_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device or torch.device("cpu")
        self.loss_fn = _build_loss(self.similarity_metric, self.window_size, squared)
        self.loss_history: List[float] = []
        self.warp = None
        self.theta = None

    def fit(self, fixed_tensor: torch.Tensor, moving_tensor: torch.Tensor, theta: torch.Tensor,
            scales: List[int], iterations: List[int], fixed_mask_tensor: Optional[torch.Tensor] = None,
            verbose: bool = False, spacing: Optional[Sequence[float]] = None,
            moving_spacing: Optional[Sequence[float]] = None, origin: Optional[Sequence[float]] = None,
            direction: Optional[Any] = None) -> torch.Tensor:
        """Multi-scale optimisation of the field.

        ``fixed_tensor`` / ``moving_tensor``: ``(1, 1, *spatial)``; ``theta``: ``(1, dim, dim + 1)`` affine in
        normalised coordinates (fixed -> moving); ``scales`` / ``iterations``: shrink factor and iterations per
        level. Returns (and stores as ``self.warp``) the field; ``self.loss_history`` keeps the loss.

        ``spacing`` / ``origin`` / ``direction`` (ITK x, y, z order) are required (``ValueError`` otherwise, syntx >= 6.42);
        ``moving_spacing`` defaults to the fixed spacing. The spacings
        feed the 'deformable' pyramid, so a level is isotropic in millimetres and a thick axis is not shrunk with the thin ones.
        ``origin`` / ``direction`` (ITK order) are only validated and forwarded to the regularizer (the field is
        regularized in the voxel frame, ``build_level_regularizer(param_domain='normalized_grid')``).
        """
        self.theta = theta
        full_shape = fixed_tensor.shape[2:]
        dim = self.dim
        mode = "trilinear" if dim == 3 else "bilinear"

        def smooth_field(f, gaussians):
            return separable_1d_filter(torch.movedim(f, -1, 1), gaussians).movedim(1, -1)

        # syntx >= 6.42: no silent unit-spacing / zero-origin / identity-direction fallback on raw tensors.
        from .syn.core.regularizers import _require_spacing, resolve_required_image_geometry
        sp_f, _orig_f, _dir_f = resolve_required_image_geometry(
            fixed_tensor, spacing, origin, direction, dim, "fixed", "GreedyRegistrationModel.fit",
            spacing_name="spacing", origin_name="origin", direction_name="direction")
        sp_m = _require_spacing(moving_spacing, dim, "GreedyRegistrationModel.fit") if moving_spacing is not None else sp_f
        # the pyramid: each image on its own grid, the warp lives on the FIXED level grid
        fixed_lv, mask_lv = _build_pyramid_deformable(fixed_tensor, sp_f, scales, mask=fixed_mask_tensor)
        moving_lv, _ = _build_pyramid_deformable(moving_tensor, sp_m, scales, reference_spacing=sp_f)

        init_size = list(fixed_lv[0].shape)
        warp = torch.zeros((1, *init_size, dim), dtype=torch.float32, device=self.device)
        # syntx >= 6.43: one stateful field-step engine for every optimizer ('cfl': CFL-normalised step with
        # velocity momentum; 'adam' / 'reg_adam': per-level Adam moments, resized with the field).
        step_opt = FieldStepOptimizer(mode=self.optimizer, param_domain="normalized_grid", beta1=self.beta1,
                                      beta2=self.beta2, eps=self.eps, cfl_momentum=self.cfl_momentum,
                                      sobolev_alpha=self.sobolev_alpha)
        step_opt.init_state(warp)
        self.loss_history = []
        regadam = self.optimizer in ("regadam", "reg_adam")
        M_theta = self.theta[0, :, :dim].contiguous()
        t_theta = self.theta[0, :, dim].contiguous()
        is_identity_theta = bool(torch.allclose(
            self.theta[0], torch.eye(dim, dim + 1, device=self.theta.device, dtype=self.theta.dtype), atol=1e-7))

        self.stop.records.clear()
        for level_idx, (scale, n_iters) in enumerate(zip(scales, iterations)):
            if n_iters <= 0:
                continue
            self.stop.start_level(level_idx, n_iters)
            size_down = list(fixed_lv[level_idx].shape)
            sp_level = fixed_lv[level_idx].spacing
            fi_down = fixed_lv[level_idx].tensor
            mi_down = moving_lv[level_idx].tensor
            mask_down = mask_lv[level_idx].tensor if mask_lv is not None else None

            # Interpolate the warp and the optimizer state to the current scale
            if list(warp.shape[1:-1]) != size_down:
                warp = _resize_field(warp, size_down, mode=mode)
                step_opt.on_level_change(size_down, interp_mode=mode)

            grid_shape = [1, 1, *size_down]
            id_grid = F.affine_grid(torch.eye(dim, dim + 1, device=self.device)[None], grid_shape, align_corners=True)

            grad_gaussians = ([_gaussian_1d_compact(self.flow_sigma, truncated=2.0, device=self.device)
                               for _ in range(dim)] if self.flow_sigma > 0 else None)
            reg_gaussians = ([_gaussian_1d_compact(self.regadam_sigma, truncated=2.0, device=self.device)
                              for _ in range(dim)] if (regadam and self.regadam_sigma > 0) else None)
            warp_gaussians = ([_gaussian_1d_compact(self.total_sigma, truncated=2.0, device=self.device)
                               for _ in range(dim)] if self.total_sigma > 0 else None)

            if self.regularizer not in ("gaussian", "gauss", "none", "identity"):
                _reg_kw = dict(alpha=self.sobolev_alpha, sobolev_alpha=self.sobolev_alpha, spacing=sp_level,
                               origin=origin, direction=direction, fluid_sigma=self.flow_sigma,
                               param_domain="normalized_grid", grid_shape=size_down, mask=mask_down,
                               fixed_mask=mask_down, extra_reg_kwargs=self.extra_reg_kwargs, output_units="voxel")
                grad_reg_fn = build_level_regularizer(self.regularizer, input_role="gradient", **_reg_kw)
                post_reg_fn = (build_level_regularizer(self.regularizer, input_role="velocity", **_reg_kw)
                               if regadam else None)
            else:
                grad_reg_fn = post_reg_fn = None
            grad_smooth_fn = ((lambda f, _g=grad_gaussians: smooth_field(f, _g))
                              if (self.flow_sigma > 0 and grad_gaussians is not None) else None)
            reg_smooth_fn = (lambda f, _g=reg_gaussians: smooth_field(f, _g)) if reg_gaussians is not None else None

            def _sobolev_fallback_fn(raw_u, ra_alpha, _sp=sp_level):
                return apply_sobolev_green_operator(raw_u.squeeze(0), fluid_sigma=ra_alpha, alpha=ra_alpha,
                                                    spacing=_sp).unsqueeze(0)

            restr_mask = (torch.tensor(self.restrict_transformation, device=self.device, dtype=warp.dtype)
                          .view(1, *([1] * dim), dim) if self.restrict_transformation is not None else None)

            # Constant fixed-image LNCC local mean / variance, once per level (syntx >= 6.40)
            use_fast_lncc = bool(getattr(self.loss_fn, "fast_lncc", False))
            if use_fast_lncc:
                ws_lvl = min(int(self.window_size), int(min(size_down)))
                if ws_lvl % 2 == 0:
                    ws_lvl = max(1, ws_lvl - 1)
                with torch.no_grad():
                    _j_mean = _box_pool(fi_down, ws_lvl, count_include_pad=False)
                    _lncc_M_c = fi_down - _j_mean
                    del _j_mean
                    _lncc_J_var = torch.clamp(_box_pool(_lncc_M_c.square(), ws_lvl, count_include_pad=False), min=0.0)
                    _lncc_safe_J_var = torch.clamp(_lncc_J_var, min=1e-6)
                    _lncc_j_valid = ((_lncc_J_var > 1e-6) & (mask_down > 0.5)) if mask_down is not None else None
                    del _lncc_J_var
            lncc_squared = bool(getattr(self.loss_fn, "squared", True))

            for it in range(n_iters):
                warp_param = warp.detach().requires_grad_(True)

                # A o (Id + u): deform in fixed-space coordinates (clamped to [-1, 1], as the border-padded sampling
                # of the affine grid did), then map through the affine in closed form.
                if is_identity_theta:
                    sample_grid = torch.clamp(id_grid + warp_param, -1.0, 1.0)
                else:
                    sample_grid = _Conv1x1AffineGrid.apply(id_grid + warp_param, M_theta, t_theta)
                moved = grid_sample(mi_down, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

                if use_fast_lncc:
                    I_mean = box_mean_nd(moved, ws_lvl)
                    F_c = moved - I_mean
                    del I_mean
                    c_ch = moved.shape[1]
                    sec = box_mean_nd(torch.cat([F_c.square(), F_c * _lncc_M_c], dim=1), ws_lvl)
                    I_var = torch.clamp(sec[:, :c_ch], min=0.0)
                    IJ_cov = sec[:, c_ch:]
                    del sec, F_c
                    safe_I_var = torch.clamp(I_var, min=1e-6)
                    if lncc_squared:
                        cc_metric = torch.clamp((IJ_cov ** 2) / (safe_I_var * _lncc_safe_J_var + 1e-8), min=0.0, max=1.0)
                    else:
                        cc_metric = torch.clamp(IJ_cov / (torch.sqrt(safe_I_var * _lncc_safe_J_var) + 1e-6),
                                                min=-1.0, max=1.0)
                    if mask_down is not None:
                        active = ((I_var > 1e-6) & _lncc_j_valid).to(moved.dtype)
                        loss = -torch.sum(cc_metric * active) / (torch.sum(active) + 1e-8)
                    else:
                        loss = -torch.mean(cc_metric)
                    del I_var, IJ_cov, safe_I_var, cc_metric
                else:
                    loss = self.loss_fn(moved, fi_down, mask=mask_down)
                self.loss_history.append(loss.detach())
                loss.backward()

                grad = warp_param.grad.detach()
                with torch.no_grad():
                    update = step_opt.step(grad, self.learning_rate, grid_shape=size_down, spacing=sp_level,
                                           reg_fn=grad_reg_fn, post_reg_fn=post_reg_fn,
                                           grad_smooth_fn=grad_smooth_fn, reg_smooth_fn=reg_smooth_fn,
                                           sobolev_fallback_fn=_sobolev_fallback_fn)
                del grad
                if restr_mask is not None:
                    update = update * restr_mask

                # Eulerian compositive pullback: u_{k+1}(x) = v_k(x) + u_k(x + v_k(x))
                pulled = _compose_grids(warp, id_grid + update.to(warp.dtype))
                u_new = update + pulled
                if self.total_sigma > 0 and warp_gaussians is not None:
                    u_new = smooth_field(u_new, warp_gaussians)
                if self.restrict_transformation is not None:
                    u_new = u_new * restr_mask
                warp = u_new.detach()

                if verbose and (it % 25 == 0 or it == n_iters - 1):
                    print(f"  [greedy] Level {level_idx} (scale {scale}) Iter {it + 1}/{n_iters} | Loss: {loss.item():.4f}")

                # the loss is read back every 5 iterations (one device sync); the level ends when it stopped decreasing
                if (it + 1) % 5 == 0 and self.stop.update(it, float(loss.detach())):
                    break
            self.stop.end_level(it + 1)

            if self.anderson and self.anderson_freq == "per_scale":
                if verbose:
                    print(f"  [greedy] Applying Anderson projection at scale {scale} ({self.anderson_steps} steps)...")
                warp_inv = update_inverse_field_nd_anderson(warp, None, steps=self.anderson_steps, m=self.anderson_m)
                warp = update_inverse_field_nd_anderson(warp_inv, None, steps=self.anderson_steps, m=self.anderson_m)

        if list(warp.shape[1:-1]) != list(full_shape):
            warp = _resize_field(warp, list(full_shape), mode=mode)

        if self.anderson and self.anderson_freq == "posthoc":
            if verbose:
                print(f"  [greedy] Applying post-hoc Anderson projection ({self.anderson_steps} steps)...")
            warp_inv = update_inverse_field_nd_anderson(warp, None, steps=self.anderson_steps, m=self.anderson_m)
            warp = update_inverse_field_nd_anderson(warp_inv, None, steps=self.anderson_steps, m=self.anderson_m)

        if self.restrict_transformation is not None:
            restr_mask = torch.tensor(self.restrict_transformation, device=self.device,
                                      dtype=warp.dtype).view(1, *([1] * dim), dim)
            warp = warp * restr_mask

        self.loss_history = [float(l.item()) if hasattr(l, "item") else float(l) for l in self.loss_history]
        self.convergence = list(self.stop.records)
        self.warp = warp
        return warp


# ----------------------------------------------------------------------------------------
# ANTs-level entry point
# ----------------------------------------------------------------------------------------
def greedy_registration(
    fixed: "ants.ANTsImage",
    moving: "ants.ANTsImage",
    reg_iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
    iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
    scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
    learning_rate: float = 1.5,
    flow_sigma: float = 1.62,
    total_sigma: float = 0.0,
    optimizer: Optional[str] = "cfl",
    regadam_sigma: float = 0.8,
    similarity_metric: str = "cc2",
    lncc_radius: int = 2,
    anderson: bool = False,
    anderson_steps: int = 5,
    anderson_m: int = 5,
    anderson_freq: str = "per_scale",
    return_inverse: bool = False,
    initial_transform: Optional[Union[str, List[str], Any, bool]] = None,
    affine_mode: str = "auto",
    device: Optional[str] = None,
    fixed_mask: Optional["ants.ANTsImage"] = None,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    seed: int = 42,
    restrict_transformation: Optional[Union[Sequence[float], Tuple[float, ...]]] = None,
    levels: Optional[Union[List[int], Tuple[int, ...]]] = None,
    cfl_momentum: float = 0.1875,
    regularizer: str = "navier",
    sobolev_alpha: Optional[float] = 2.625,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Greedy (one-directional) diffeomorphic registration of ``moving`` to ``fixed`` -- ``antstorch.greedy``.

    ::

        reg = antstorch.greedy.greedy(fixed, moving)
        reg['warpedmovout'], reg['fwdtransforms']        # [warp, affine] standard ANTs list

    Parameters mirror ``syntx.greedy`` (same defaults, tuned on Mindboggle pairs 77 / 44 / 0 there).

    Parameters
    ----------
    fixed, moving : ANTsImage
        2-D or 3-D. Both are intensity-normalised first (``normalize_image(method='auto')``).
    reg_iterations, iterations : list of int, default None
        Iterations per level (the two names are interchangeable). None: [100, 100, 20] (3-D),
        [100, 100, 100, 50] (2-D).
    scales : list of int, default None
        Shrink factor per level. None: [2**(L-1), ..., 1] for L = len(reg_iterations). ``levels`` is an alias.
    learning_rate : float, default 1.5
        Step size (``grad_step=`` is an alias): the largest displacement of an update, in voxels of the level grid
        (``'cfl'``) or ``learning_rate / (grid size - 1)`` of the normalised domain (``'adam'`` / ``'reg_adam'``).
    flow_sigma : float, default 1.62
        Gaussian sigma in voxels smoothing the gradient for ``regularizer='gaussian'`` (not the ITK variance
        convention of ``syn_registration``); the natural scale (alpha = flow_sigma / 2) of a continuum
        regularizer when ``sobolev_alpha`` is None.
    total_sigma : float, default 0.0
        Gaussian sigma in voxels smoothing the whole field after each update.
    optimizer : {'cfl', 'adam', 'regadam', 'reg_adam'} or None, default 'cfl'
        None: 'cfl' for the spectral / continuum regularizers, 'adam' for 'gaussian'. 'adam' with a coupled
        continuum regularizer raises ``ValueError`` (componentwise scaling breaks the div / curl coupling).
    regadam_sigma : float, default 0.8
    similarity_metric : str, default 'cc2'
        'cc2' / 'lncc' / 'cc' / 'ncc' (local correlation) or 'mse' / 'l2'.
    lncc_radius : int, default 2
    anderson, anderson_steps, anderson_m, anderson_freq
        Optional Anderson inverse-consistency projection (off by default).
    return_inverse : bool, default False
        Also compute the inverse field (Anderson fixed-point solve, >= 15 steps) and return it in
        ``'invtransforms'``. Without it there is no inverse.
    initial_transform : None, False, 'identity', str, list or ANTsTransform
        None: ``antstorch.syn.robust_affine(mode=affine_mode)``; False / 'identity': no affine; otherwise ANTs
        linear transform file(s) (a non-linear transform raises ``NotImplementedError``).
    affine_mode : str, default 'auto'
    device : str, optional
        Default: CUDA, else MPS, else CPU.
    fixed_mask : ANTsImage, optional
    verbose : bool, default False
    outprefix : str, optional
        Write the transforms with this file prefix (default: temporary files).
    seed : int, default 42
        Accepted for signature parity with ``syntx.greedy`` (the optimisation is deterministic).
    restrict_transformation : sequence of float, optional
    cfl_momentum : float, default 0.1875
        Velocity momentum of ``optimizer='cfl'`` (0 disables it).
    regularizer : str, default 'navier'
        'gaussian' or any operator of :func:`antstorch.syn.core.regularizers.list_regularizers` ('navier',
        'div_curl', 'sobolev', ...), applied to the gradient on the level grid in millimetres.
    sobolev_alpha : float or None, default 2.625
        Spectral / continuum operator strength (None: each operator's default from ``flow_sigma``).

    The defaults are syntx 6.44's canonical ``syntx.greedy`` setting (``canonical_2026_11_20``: Navier-Cauchy
    with ``s`` 2.0 and Poisson ratio 0.20, ``optimizer='cfl'``). The previous ones (syntx <= 6.41) were
    ``learning_rate=0.375, flow_sigma=1.8, total_sigma=0.28, optimizer='adam', regularizer='gaussian'``.

    **kwargs
        ``padding_mode`` ('border'), ``squared`` (True), ``interpolator`` ('linear', for the warped output),
        the continuum operator parameters (``poisson_ratio``, ``s``, ``beta``, ``gamma``, ...; see
        ``REGULARIZER_EXTRA_KEYS``; ``alpha`` is an alias of ``sobolev_alpha``), and the aliases ``grad_step`` /
        ``optimizer_type`` / ``project_inverse`` / ``anderson_projection``. Any other keyword raises ``TypeError``.

    Returns
    -------
    dict
        ``'warpedmovout'``, ``'warpedfixout'`` (None unless ``return_inverse``), ``'fwdtransforms'``
        (``[warp, affine]``), ``'invtransforms'`` / ``'whichtoinvert_inv'`` (empty unless ``return_inverse``),
        ``'model'`` (:class:`GreedyRegistrationModel`), ``'provenance'`` (runtimes, configuration) and
        ``'affine_transform'`` (the resolved initial affine, or None).
    """
    t0_total = time.time()
    dim = fixed.dimension
    torch_device = torch.device(auto_detect_device(backend="pytorch", requested_device=device))

    # 1. Multi-resolution schedule
    if reg_iterations is None:
        reg_iterations = iterations
    if reg_iterations is None and "iterations" in kwargs:
        reg_iterations = kwargs.pop("iterations")
    elif "iterations" in kwargs:
        kwargs.pop("iterations")
    if reg_iterations is None:
        reg_iterations = [100, 100, 20] if dim == 3 else [100, 100, 100, 50]
    elif isinstance(reg_iterations, int):
        reg_iterations = [reg_iterations]
    reg_iterations = list(reg_iterations)

    if scales is None and levels is not None:
        scales = levels
    if scales is None:
        num_levels = len(reg_iterations)
        scales = [2 ** i for i in range(num_levels)][::-1] if num_levels > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
    elif isinstance(scales, int):
        scales = [scales]
    scales = list(scales)

    # 2. Intensity normalisation (the same policy as syntx.greedy)
    fi_norm = normalize_image(fixed, method="auto")
    mi_norm = normalize_image(moving, method="auto")

    # 3. Initial affine
    t0_aff = time.time()
    aff_tx = None
    if initial_transform is None:
        if verbose:
            print("[greedy] Running antstorch.syn.robust_affine(mode='auto')...")
        reg_aff = robust_affine(fi_norm, mi_norm, mode=affine_mode, verbose=False, device=str(torch_device))
        aff_tx = reg_aff["fwdtransforms"][0]
    elif initial_transform is False or (isinstance(initial_transform, str) and initial_transform.lower() == "identity"):
        aff_tx = None
    else:
        aff_tx = initial_transform if isinstance(initial_transform, list) else [initial_transform]
    t1_aff = time.time() - t0_aff

    # 4. Physical and normalised affine matrices
    if aff_tx is not None:
        tx_list = aff_tx if isinstance(aff_tx, list) else [aff_tx]
        M_t, t_t = parse_ants_affine(tx_list, dim)
        if M_t is None:
            raise NotImplementedError(
                "greedy_registration: initial_transform contains a non-linear transform (e.g. a displacement field), "
                "which is not supported as a dense initial grid -- pass a linear transform (affine / rigid / "
                "similarity / ...), None (runs robust_affine), False, or 'identity' instead.")
        M_phys = M_t.detach().cpu().double().numpy()
        t_phys = t_t.detach().cpu().double().numpy()
    else:
        M_phys = np.eye(dim, dtype=np.float64)
        t_phys = np.zeros(dim, dtype=np.float64)

    theta, P_F, _P_M = _build_torch_affine_matrix(fi_norm, mi_norm, M_phys, t_phys, torch_device)

    # 5. Image tensors (ZYX layout)
    fi_t = torch.from_numpy(fi_norm.numpy().T).float().unsqueeze(0).unsqueeze(0).to(torch_device)
    mi_t = torch.from_numpy(mi_norm.numpy().T).float().unsqueeze(0).unsqueeze(0).to(torch_device)
    mask_t = None
    if fixed_mask is not None:
        mask_t = torch.from_numpy(fixed_mask.numpy().T).float().unsqueeze(0).unsqueeze(0).to(torch_device)

    # Aliases and keyword validation
    if "project_inverse" in kwargs:
        anderson = bool(kwargs.pop("project_inverse"))
    if "anderson_projection" in kwargs:
        anderson = bool(kwargs.pop("anderson_projection"))
    if "grad_step" in kwargs:
        learning_rate = float(kwargs.pop("grad_step"))
    if "optimizer_type" in kwargs:
        optimizer = str(kwargs.pop("optimizer_type"))
    padding_mode = str(kwargs.pop("padding_mode", "border"))
    cfl_momentum = float(kwargs.pop("cfl_momentum", cfl_momentum))
    extra_reg_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in REGULARIZER_EXTRA_KEYS}
    if "alpha" in extra_reg_kwargs:
        sobolev_alpha = float(extra_reg_kwargs.pop("alpha"))
    reg_mode = str(kwargs.pop("regularizer", regularizer)).lower()
    squared = bool(kwargs.pop("squared", True))
    interpolator = kwargs.pop("interpolator", "linear")
    # syntx >= 6.16 / 6.36 options (None / 0 disables the early stop; digital_diffeomorphism=False the finish polish)
    convergence_slope = kwargs.pop("convergence_slope", DEFAULT_SLOPE_THRESHOLD)
    digital_diffeomorphism = bool(kwargs.pop("digital_diffeomorphism", True))
    digital_diffeomorphism_floor = float(kwargs.pop("digital_diffeomorphism_floor", 0.05))
    digital_diffeomorphism_steps = int(kwargs.pop("digital_diffeomorphism_steps", 25))
    if kwargs:
        raise TypeError(f"antstorch.greedy() got unexpected keyword(s) {sorted(kwargs)}")

    # 6. Optimise
    model = GreedyRegistrationModel(
        dim=dim, learning_rate=learning_rate, flow_sigma=flow_sigma, total_sigma=total_sigma, optimizer=optimizer,
        regularizer=reg_mode, regadam_sigma=regadam_sigma, sobolev_alpha=sobolev_alpha, cfl_momentum=cfl_momentum,
        extra_reg_kwargs=extra_reg_kwargs, lncc_radius=lncc_radius,
        similarity_metric=similarity_metric, squared=squared, anderson=anderson, anderson_steps=anderson_steps,
        anderson_m=anderson_m, anderson_freq=anderson_freq, padding_mode=padding_mode,
        restrict_transformation=restrict_transformation, device=torch_device, convergence_slope=convergence_slope)
    t0_opt = time.time()
    warp = model.fit(fixed_tensor=fi_t, moving_tensor=mi_t, theta=theta, scales=scales, iterations=reg_iterations,
                     fixed_mask_tensor=mask_t, verbose=verbose,
                     spacing=tuple(float(x) for x in fixed.spacing), moving_spacing=tuple(float(x) for x in moving.spacing),
                     origin=tuple(float(x) for x in fixed.origin), direction=np.asarray(fixed.direction))
    t1_opt = time.time() - t0_opt

    # 7. Export: the non-linear displacement (affine kept separate) and the affine file. syntx >= 6.40 converts the
    # normalised field to the physical (tensor-layout, mm) displacement directly on the device, in float32
    # (``syntx.spatial.normalized_to_physical_disp``) instead of through a float64 CPU meshgrid, and applies
    # ``restrict_transformation`` to that tensor before the inverse / digital-diffeomorphism finish.
    full_shape = tuple(fi_t.shape[2:])
    with torch.no_grad():
        disp_t = _normalized_to_physical_disp(warp, full_shape, fixed.spacing, fixed.direction, dtype=torch.float32)
        if restrict_transformation is not None:
            restr_t_zyx = torch.tensor(list(reversed([0.0 if float(w) == 0.0 else 1.0 for w in restrict_transformation])),
                                       device=disp_t.device, dtype=disp_t.dtype).view(1, *([1] * dim), dim)
            disp_t = disp_t * restr_t_zyx
        else:
            restr_t_zyx = None

    # 7b. Physical inverse and/or digital-diffeomorphism finish (syntx >= 6.36, default on), both on the tensor
    # layout (axes AND components reversed).
    from .syn.bridge import displacement_zyx_to_ants_image
    _dd_on = digital_diffeomorphism and sum(reg_iterations) > 0
    _dd_kw = dict(spacing=fixed.spacing, origin=fixed.origin, direction=fixed.direction,
                  jac_floor=digital_diffeomorphism_floor, steps=digital_diffeomorphism_steps,
                  stationary_boundary=(padding_mode == "border"))
    inv_disp_img = None
    t1_inv = 0.0
    if return_inverse:
        t0_inv = time.time()
        if verbose:
            print("[greedy] Computing physical inverse displacement field with Anderson acceleration...")
        inv_disp_t = update_inverse_field_nd_anderson(
            W_disp=disp_t, W_inv_disp=None, steps=max(anderson_steps, 15), m=anderson_m,
            spacing=fixed.spacing, origin=fixed.origin, direction=fixed.direction)
        if _dd_on:
            from .syn.core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
            disp_t, inv_disp_t = enforce_digital_diffeomorphism_nd(disp_t, inv_disp_t, ice_tol=0.30, **_dd_kw)
        if restr_t_zyx is not None:
            with torch.no_grad():
                disp_t = disp_t * restr_t_zyx
                inv_disp_t = inv_disp_t * restr_t_zyx
        disp_img = displacement_zyx_to_ants_image(disp_t, fixed)
        inv_disp_img = displacement_zyx_to_ants_image(inv_disp_t, fixed)
        t1_inv = time.time() - t0_inv
    else:
        if _dd_on:
            from .syn.core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
            disp_t = enforce_digital_diffeomorphism_nd(disp_t, None, **_dd_kw)
            disp_t = disp_t[0] if isinstance(disp_t, tuple) else disp_t
            if restr_t_zyx is not None:
                with torch.no_grad():
                    disp_t = disp_t * restr_t_zyx
        disp_img = displacement_zyx_to_ants_image(disp_t, fixed)

    affine_file = None
    if aff_tx is not None:
        single = aff_tx if isinstance(aff_tx, str) else (aff_tx[0] if isinstance(aff_tx, list) and len(aff_tx) == 1 else None)
        if isinstance(single, str) and os.path.exists(single):
            affine_file = single  # a given affine file is referenced as is (no re-serialisation)
        elif outprefix is not None:
            affine_file = f"{outprefix}0GenericAffine.mat"
            write_affine_transform(M_phys, t_phys, dim, affine_file)
        else:
            affine_file = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name
            write_affine_transform(M_phys, t_phys, dim, affine_file)

    if outprefix is not None:
        os.makedirs(os.path.dirname(outprefix) or ".", exist_ok=True)
        fwd_file = f"{outprefix}1Warp.nii.gz"
    else:
        fwd_file = tempfile.NamedTemporaryFile(suffix="_fwd_warp.nii.gz", delete=False).name
    ants.image_write(disp_img, fwd_file)

    if affine_file is not None:
        if isinstance(aff_tx, list) and len(aff_tx) > 1 and outprefix is None:
            fwd_transforms = [fwd_file] + aff_tx
        else:
            fwd_transforms = [fwd_file, affine_file]
    else:
        fwd_transforms = [fwd_file]

    # 8. Warped moving image (single interpolation on the native moving image)
    warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms,
                                         interpolator=interpolator)

    # 9. Optional physical inverse
    inv_transforms: List[str] = []
    whichtoinvert_inv: List[bool] = []
    warpedfixout = None
    if return_inverse and inv_disp_img is not None:
        t0_inv = time.time()
        if outprefix is not None:
            inv_file = f"{outprefix}1InverseWarp.nii.gz"
        else:
            inv_file = tempfile.NamedTemporaryFile(suffix="_inv_warp.nii.gz", delete=False).name
        ants.image_write(inv_disp_img, inv_file)
        if affine_file is not None:
            if isinstance(aff_tx, list) and len(aff_tx) > 1 and outprefix is None:
                inv_transforms = aff_tx + [inv_file]
                whichtoinvert_inv = [True] * len(aff_tx) + [False]
            else:
                inv_transforms = [affine_file, inv_file]
                whichtoinvert_inv = [True, False]
        else:
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]
        warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms,
                                             whichtoinvert=whichtoinvert_inv, interpolator=interpolator)
        t1_inv += time.time() - t0_inv

    total_runtime = time.time() - t0_total
    provenance = {
        "algorithm": "antstorch.greedy",
        "formulation": "eulerian_compositive",
        "optimizer": model.optimizer,
        "regularizer": model.regularizer,
        "sobolev_alpha": model.sobolev_alpha,
        "cfl_momentum": model.cfl_momentum if model.optimizer == "cfl" else None,
        "regularizer_kwargs": dict(model.extra_reg_kwargs) or None,
        "regadam_sigma": regadam_sigma if model.optimizer in ("regadam", "reg_adam") else None,
        "reg_iterations": reg_iterations,
        "scales": scales,
        "learning_rate": learning_rate,
        "flow_sigma": flow_sigma,
        "total_sigma": total_sigma,
        "padding_mode": padding_mode,
        "similarity_metric": similarity_metric,
        "anderson": anderson,
        "anderson_steps": anderson_steps if (anderson or return_inverse) else 0,
        "anderson_freq": anderson_freq if anderson else "none",
        "return_inverse": return_inverse,
        "digital_diffeomorphism": digital_diffeomorphism,
        "convergence": model.stop.summary(),
        "runtime_total_sec": round(total_runtime, 2),
        "runtime_affine_sec": round(t1_aff, 2),
        "runtime_deformable_sec": round(t1_opt, 2),
        "runtime_inverse_sec": round(t1_inv, 2),
        "device": str(torch_device),
    }
    if verbose:
        print(f"[greedy] Complete in {total_runtime:.2f}s (Affine: {t1_aff:.2f}s, Deformable: {t1_opt:.2f}s, "
              f"Inverse: {t1_inv:.2f}s)")
    cleanup_gpu(torch_device)

    return {
        "warpedmovout": warpedmovout,
        "warpedfixout": warpedfixout,
        "fwdtransforms": fwd_transforms,
        "invtransforms": inv_transforms,
        "whichtoinvert_inv": whichtoinvert_inv,
        "model": model,
        "provenance": provenance,
        "affine_transform": aff_tx,
    }


# Alias
greedy = greedy_registration
