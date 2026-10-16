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
from .syn.core.inverse import update_inverse_field_nd_anderson
from .syn.core.losses import local_ncc_loss_nd
from .syn.core.pipeline import auto_detect_device, cleanup_gpu
from .syn.core.smoothing import separable_1d_filter
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

    return _cc


# ----------------------------------------------------------------------------------------
# The "deformable" image pyramid of syntx (syntx.pyramid, >= v6.2: plan_levels + build_pyramid, preset 'deformable')
# ----------------------------------------------------------------------------------------
_MIN_VOXELS_PER_AXIS = 16    # per-axis floor of the 'deformable' preset (an axis shorter than this keeps its native sampling)
_HARD_MIN_VOXELS = 2         # never fewer than 2 samples on an axis of >= 2 samples
_PYR_EPS = 1e-6


class _Level:
    """One pyramid level: ``tensor`` (B, C, *shape) and ``shape`` (tensor order z, y, x)."""
    __slots__ = ("tensor", "shape")

    def __init__(self, tensor: torch.Tensor, shape: Tuple[int, ...]):
        self.tensor, self.shape = tensor, shape


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


def _plan_levels_deformable(shape_zyx, spacing_xyz, levels, reference_spacing_xyz=None):
    """Plan every level of the 'deformable' pyramid (``syntx.pyramid.plan_levels(..., 'deformable')``): level L has the working
    spacing L x (finest spacing of the reference or of the image), isotropic in millimetres; default smoothing
    sigma = 0.5 x the axis's coarsening factor (native voxels) on every coarsened axis. Returns a list of dicts."""
    n_zyx = tuple(int(x) for x in shape_zyx)
    sp = tuple(float(x) for x in spacing_xyz)
    ref = tuple(float(x) for x in reference_spacing_xyz) if reference_spacing_xyz is not None else sp
    n_xyz = n_zyx[::-1]
    plans = []
    for L in [float(x) for x in levels]:
        t = max(L * min(ref), min(sp))
        axes = _plan_axes_deformable(n_xyz, sp, t, _MIN_VOXELS_PER_AXIS)
        m_xyz = tuple(a[0] for a in axes)
        f_xyz = tuple(a[1] for a in axes)
        modes = tuple(a[2] for a in axes)
        sig_xyz = [0.5 * f if (md == "linear" and f > 1.0 + _PYR_EPS) else 0.0 for f, md in zip(f_xyz, modes)]
        sig_zyx = tuple(sg if (f > 1.0 + _PYR_EPS and sg > 0) else 0.0 for sg, f in zip(sig_xyz, f_xyz))[::-1]
        plans.append(dict(shape_zyx=tuple(m_xyz[::-1]), f_xyz=f_xyz, modes_xyz=modes, sigma_zyx=sig_zyx,
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
    from .syn.core.smoothing import get_cached_gaussian_kernel_1d
    v = x
    device, dtype = v.device, v.dtype
    dim = v.dim() - 2
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
    lv = [_Level(_apply_level(image, p), p["shape_zyx"]) for p in plans]
    ml = [_Level(_apply_level(mask.to(image.dtype), p), p["shape_zyx"]) for p in plans] if mask is not None else None
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
class GreedyRegistrationModel(nn.Module):
    """The tensor-level model behind :func:`greedy_registration` (``syntx.greedy.GreedyRegistrationModel``).

    ``fit`` returns the displacement field in normalised ``[-1, 1]`` coordinates, ``(1, *spatial, dim)``,
    tensor ``(z, y, x)`` spatial order, ``(x, y, z)`` components.

    Parameters
    ----------
    dim : {2, 3}
    learning_rate : float, default 0.45
        Step size (``greedy_registration`` passes 0.375): the update's largest displacement is
        ``learning_rate / (grid size - 1)`` of the normalised domain.
    flow_sigma : float, default 1.8
        Gaussian sigma (voxels) smoothing the similarity gradient.
    total_sigma : float, default 0.28
        Gaussian sigma (voxels) smoothing the whole field after each update (0 = off).
    optimizer : {'adam', 'regadam' / 'reg_adam'}
        'regadam' additionally smooths the Adam quotient (``regadam_sigma``, Gaussian voxels).
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

    def __init__(self, dim: int = 3, learning_rate: float = 0.45, flow_sigma: float = 1.8, total_sigma: float = 0.28,
                 optimizer: str = "adam", regularizer: str = "gaussian", regadam_sigma: float = 0.8,
                 lncc_radius: int = 2, similarity_metric: str = "cc2", squared: bool = True,
                 anderson: bool = False, anderson_steps: int = 5, anderson_m: int = 5,
                 anderson_freq: str = "per_scale", padding_mode: str = "border",
                 beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8,
                 restrict_transformation: Optional[Sequence[float]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
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
        if str(regularizer).lower() not in ("gaussian", "gauss"):
            raise ValueError(f"antstorch.greedy: only the 'gaussian' regularizer is supported, got {regularizer!r}")
        self.learning_rate = learning_rate
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        self.optimizer = optimizer.lower()
        if self.optimizer not in ("adam", "regadam", "reg_adam"):
            raise ValueError(f"antstorch.greedy: optimizer must be 'adam' or 'regadam', got {optimizer!r}")
        self.regularizer = "gaussian"
        self.regadam_sigma = regadam_sigma
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
            moving_spacing: Optional[Sequence[float]] = None) -> torch.Tensor:
        """Multi-scale optimisation of the field.

        ``fixed_tensor`` / ``moving_tensor``: ``(1, 1, *spatial)``; ``theta``: ``(1, dim, dim + 1)`` affine in
        normalised coordinates (fixed -> moving); ``scales`` / ``iterations``: shrink factor and iterations per
        level. Returns (and stores as ``self.warp``) the field; ``self.loss_history`` keeps the loss.

        ``spacing`` / ``moving_spacing`` (ITK x, y, z order; default unit spacing, and the fixed spacing for the moving image)
        feed the 'deformable' pyramid, so a level is isotropic in millimetres and a thick axis is not shrunk with the thin ones.
        """
        self.theta = theta
        full_shape = fixed_tensor.shape[2:]
        dim = self.dim
        mode = "trilinear" if dim == 3 else "bilinear"

        def smooth_field(f, gaussians):
            return separable_1d_filter(torch.movedim(f, -1, 1), gaussians).movedim(1, -1)

        sp_f = tuple(float(x) for x in spacing) if spacing is not None else (1.0,) * dim
        sp_m = tuple(float(x) for x in moving_spacing) if moving_spacing is not None else sp_f
        # the pyramid: each image on its own grid, the warp lives on the FIXED level grid
        fixed_lv, mask_lv = _build_pyramid_deformable(fixed_tensor, sp_f, scales, mask=fixed_mask_tensor)
        moving_lv, _ = _build_pyramid_deformable(moving_tensor, sp_m, scales, reference_spacing=sp_f)

        init_size = list(fixed_lv[0].shape)
        warp = torch.zeros((1, *init_size, dim), dtype=torch.float32, device=self.device)
        exp_avg = torch.zeros_like(warp)
        exp_avg_sq = torch.zeros_like(warp)
        self.loss_history = []
        regadam = self.optimizer in ("regadam", "reg_adam")

        for level_idx, (scale, n_iters) in enumerate(zip(scales, iterations)):
            if n_iters <= 0:
                continue
            size_down = list(fixed_lv[level_idx].shape)
            fi_down = fixed_lv[level_idx].tensor
            mi_down = moving_lv[level_idx].tensor
            mask_down = mask_lv[level_idx].tensor if mask_lv is not None else None

            # Interpolate the warp and the Adam moments to the current scale
            if list(warp.shape[1:-1]) != size_down:
                warp = _resize_field(warp, size_down, mode=mode)
                exp_avg = _resize_field(exp_avg, size_down, mode=mode)
                exp_avg_sq = _resize_field(exp_avg_sq, size_down, mode=mode)

            grid_shape = [1, 1, *size_down]
            id_grid = F.affine_grid(torch.eye(dim, dim + 1, device=self.device)[None], grid_shape, align_corners=True)
            affine_grid = F.affine_grid(self.theta, grid_shape, align_corners=True)
            affine_grid_cf = torch.movedim(affine_grid, -1, 1)
            half_resolution = 1.0 / (max(size_down) - 1)
            step = 0  # Adam warm-up step reset at each scale level

            grad_gaussians = ([_gaussian_1d_compact(self.flow_sigma, truncated=2.0, device=self.device)
                               for _ in range(dim)] if self.flow_sigma > 0 else None)
            reg_gaussians = ([_gaussian_1d_compact(self.regadam_sigma, truncated=2.0, device=self.device)
                              for _ in range(dim)] if (regadam and self.regadam_sigma > 0) else None)
            warp_gaussians = ([_gaussian_1d_compact(self.total_sigma, truncated=2.0, device=self.device)
                               for _ in range(dim)] if self.total_sigma > 0 else None)

            for it in range(n_iters):
                warp_param = warp.detach().requires_grad_(True)

                # A o (Id + u): deform in fixed-space coordinates, then map through the affine.
                sample_grid = _sample_field_cf(affine_grid_cf, id_grid + warp_param)
                moved = grid_sample(mi_down, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

                loss = self.loss_fn(moved, fi_down, mask=mask_down)
                self.loss_history.append(loss.detach())
                loss.backward()

                grad = warp_param.grad.data
                if self.flow_sigma > 0 and grad_gaussians is not None:
                    grad = smooth_field(grad, grad_gaussians)

                step += 1
                exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                b1_corr = 1.0 - self.beta1 ** step
                b2_corr = 1.0 - self.beta2 ** step
                denom = (exp_avg_sq / b2_corr).sqrt().add_(self.eps)
                raw_update = (exp_avg / b1_corr) / denom

                if regadam and reg_gaussians is not None:
                    u_reg = smooth_field(raw_update, reg_gaussians)
                else:
                    u_reg = raw_update

                # Bound the maximum step
                gradmax = self.eps + u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                gradmax = gradmax.reshape(-1, *([1]) * (dim + 1)).clamp(min=1.0)
                update = -self.learning_rate * half_resolution * (u_reg / gradmax)
                if self.restrict_transformation is not None:
                    restr_mask = torch.tensor(self.restrict_transformation, device=self.device,
                                              dtype=update.dtype).view(1, *([1] * dim), dim)
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
    learning_rate: float = 0.375,
    flow_sigma: float = 1.8,
    total_sigma: float = 0.28,
    optimizer: str = "adam",
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
    learning_rate : float, default 0.375
        Step size (``grad_step=`` is an alias).
    flow_sigma : float, default 1.8
        Gaussian sigma in voxels smoothing the gradient (not the ITK variance convention of ``syn_registration``).
    total_sigma : float, default 0.28
        Gaussian sigma in voxels smoothing the whole field after each update.
    optimizer : {'adam', 'regadam', 'reg_adam'}, default 'adam'
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
    **kwargs
        ``padding_mode`` ('border'), ``squared`` (True), ``interpolator`` ('linear', for the warped output),
        ``regularizer`` (only 'gaussian'), and the aliases ``grad_step`` / ``optimizer_type`` /
        ``project_inverse`` / ``anderson_projection``. Any other keyword raises ``TypeError``.

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
    reg_mode = str(kwargs.pop("regularizer", "gaussian")).lower()
    squared = bool(kwargs.pop("squared", True))
    interpolator = kwargs.pop("interpolator", "linear")
    if kwargs:
        raise TypeError(f"antstorch.greedy() got unexpected keyword(s) {sorted(kwargs)}")

    # 6. Optimise
    model = GreedyRegistrationModel(
        dim=dim, learning_rate=learning_rate, flow_sigma=flow_sigma, total_sigma=total_sigma, optimizer=optimizer,
        regularizer=reg_mode, regadam_sigma=regadam_sigma, lncc_radius=lncc_radius,
        similarity_metric=similarity_metric, squared=squared, anderson=anderson, anderson_steps=anderson_steps,
        anderson_m=anderson_m, anderson_freq=anderson_freq, padding_mode=padding_mode,
        restrict_transformation=restrict_transformation, device=torch_device)
    t0_opt = time.time()
    warp = model.fit(fixed_tensor=fi_t, moving_tensor=mi_t, theta=theta, scales=scales, iterations=reg_iterations,
                     fixed_mask_tensor=mask_t, verbose=verbose,
                     spacing=tuple(float(x) for x in fixed.spacing), moving_spacing=tuple(float(x) for x in moving.spacing))
    t1_opt = time.time() - t0_opt

    # 7. Export: the non-linear displacement (affine kept separate) and the affine file
    full_shape = fi_t.shape[2:]
    full_id_grid = F.affine_grid(torch.eye(dim, dim + 1, device=warp.device)[None], [1, 1, *full_shape],
                                 align_corners=True)
    disp_img = _grid_to_ants_displacement(full_id_grid + warp, fixed, P_F, P_F,
                                          restrict_transformation=restrict_transformation)

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
    t1_inv = 0.0
    if return_inverse:
        t0_inv = time.time()
        if verbose:
            print("[greedy] Computing physical inverse displacement field with Anderson acceleration...")
        disp_t = torch.from_numpy(disp_img.numpy().T.copy()).float().unsqueeze(0).to(torch_device)
        inv_disp_t = update_inverse_field_nd_anderson(
            W_disp=disp_t, W_inv_disp=None, steps=max(anderson_steps, 15), m=anderson_m,
            spacing=fixed.spacing, origin=fixed.origin, direction=fixed.direction)
        inv_np = inv_disp_t.squeeze(0).cpu().numpy().T
        inv_disp_img = ants.from_numpy(inv_np.astype(np.float32), origin=fixed.origin, spacing=fixed.spacing,
                                       direction=fixed.direction, has_components=True)
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
        t1_inv = time.time() - t0_inv

    total_runtime = time.time() - t0_total
    provenance = {
        "algorithm": "antstorch.greedy",
        "formulation": "eulerian_compositive",
        "optimizer": optimizer,
        "regadam_sigma": regadam_sigma if optimizer in ("regadam", "reg_adam") else None,
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
