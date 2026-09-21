"""Native PyTorch multi-start robust affine registration.

Ported from ``syntx.robust_affine`` (PyTorch backend only), scoped to the
solver ``antstorch/syn/__init__.py``'s earlier docstring called out as
*not* ported ("Etape 2... reuses/extends ``affine_registration`` rather
than porting ``syntx``'s separate ``robust_affine`` module"). That decision
is revisited here: since 2026-09-15, syntx's default/modern engine
(``mode='auto'``/``'pytorch'``) is itself pure PyTorch -- no JAX, and no
mandatory landmark dependency -- so it is portable on the same terms as the
rest of :mod:`antstorch.syn`.

This is a deliberately narrower port than the full ``syntx.robust_affine``
module. Ported:

- The native PyTorch multi-start solver (:func:`_run_pytorch_affine_solver`
  and its helpers): cone-constrained Lie-algebra rotation search, SE(3)
  Riemannian-geodesic candidate clustering for hypothesis diversity, a
  multi-resolution Adam/L-BFGS schedule, and the deterministic (blocked-
  histogram) Mattes MI objective (:func:`antstorch.syn.core.losses.mattes_mi_loss_nd`).
- ``'com_only'`` (instant center-of-mass translation) and
  ``'pytorch'``/``'gpu'``/``'auto'``/``'fast'`` modes of :func:`robust_affine`.

Not ported (out of scope for this pass):

- ``enable_landmarks`` / SIFT3D-seeded candidates (``syntx.landmarks`` is a
  separate module, not ported here; disabled by default in syntx itself).
- ``mode='tournament'`` (orchestrates multiple engines including landmarks).
- The legacy multi-candidate ``ants.registration()`` C++ pipeline that
  syntx's own ``mode='ants_fast'`` (and ``'auto'``'s failure fallback) uses.
  Here, ``'auto'``/``'fast'`` fall back to a single, direct
  ``ants.registration(type_of_transform='Affine')`` call on any solver
  exception -- simpler than syntx's low-res multi-candidate ANTs fallback,
  and a deliberate deviation documented on :func:`robust_affine` itself.

Much of the physical-space/grid plumbing this solver needs was already
ported in :mod:`antstorch.syn.core.affine`/:mod:`antstorch.syn.core.grid`
(``get_rotation_matrix``, ``parse_ants_affine``) and
:mod:`antstorch.syn.bridge`/:mod:`antstorch.ants_transform_io`
(``ants_image_to_tensor``, ``write_affine_transform``); this module reuses
them rather than duplicating syntx's own ``spatial.py`` helpers.
"""

import functools
import logging
import os
import tempfile
import time
import warnings
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..ants_transform_io import write_affine_transform
from .bridge import ants_image_to_tensor
from .core.affine import get_rotation_matrix, parse_ants_affine
from .core.losses import mattes_mi_loss_nd, parzen_weights
from .core.utils import normalize_image

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _mps_grid_sample_backward_available(dim: int) -> bool:
    """Whether the installed PyTorch's MPS backend implements ``grid_sample``
    backward for the given spatial dimensionality.

    ``antstorch.syn.core.pipeline.mps_grid_sample_3d_available`` probes this
    for 3-D only, on the documented assumption that 2-D ``grid_sampler_2d``
    is fully implemented on MPS (forward *and* backward). That assumption
    does not hold on every PyTorch build: ``aten::grid_sampler_2d_backward``
    has been observed missing on MPS too (mirrors the tracked 3-D gap,
    https://github.com/pytorch/pytorch/issues/141287), which raises
    ``NotImplementedError`` the moment ``_run_pytorch_affine_solver`` calls
    ``loss.backward()``. This solver runs in both 2-D and 3-D, so it probes
    whichever dimensionality it is actually about to use rather than
    special-casing 3-D like the shared helper does. Probed empirically (a
    throwaway forward+backward call) and cached per dimension for the life
    of the process, so this automatically stops paying the CPU-fallback
    cost once a future PyTorch release ships the missing kernel.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        shape = (1, 1) + (2,) * dim
        grid_shape = (1,) + (2,) * dim + (dim,)
        probe_image = torch.zeros(shape, device="mps", requires_grad=True)
        probe_grid = torch.zeros(grid_shape, device="mps")
        warped = F.grid_sample(probe_image, probe_grid, align_corners=True)
        warped.sum().backward()
        return True
    except (NotImplementedError, RuntimeError):
        return False


def _resolve_solver_device(device_obj: "torch.device", dim: int) -> "torch.device":
    """Fall back to CPU when ``grid_sample`` backward is unavailable on ``device_obj``.

    ``_run_pytorch_affine_solver`` differentiates through
    ``F.grid_sample`` every optimization iteration (see :func:`objective`
    inside it); on an ``mps`` device without the backward kernel for this
    ``dim``, the very first ``loss.backward()`` would otherwise crash the
    run. CUDA and CPU are assumed complete and are never probed.
    """
    if device_obj.type != "mps" or _mps_grid_sample_backward_available(dim):
        return device_obj
    warnings.warn(
        f"robust_affine: {dim}-D torch.nn.functional.grid_sample is missing its backward "
        "MPS kernel in this PyTorch build (aten::grid_sampler_{}d_backward; see "
        "https://github.com/pytorch/pytorch/issues/141287). Falling back to CPU for this "
        "solver run. Set PYTORCH_ENABLE_MPS_FALLBACK=1 instead if you would rather PyTorch "
        "itself fall back transparently for every unimplemented MPS op.".format(dim),
        RuntimeWarning, stacklevel=3,
    )
    return torch.device("cpu")


__all__ = [
    "robust_center_of_mass",
    "compute_center_of_mass",
    "compute_fov_center",
    "robust_affine",
]


def robust_center_of_mass(
    img_ants,
    weighted: bool = True,
    threshold: float = 0.0,
    return_dict: bool = False,
):
    """Physical center of mass (CoM), computed independently on the positive and negative halves.

    For images with both positive and negative values (CT air/tissue,
    subtraction images), unpartitioned CoM suffers first-moment
    cancellation, corrupting the center by tens of mm. This computes:

    1. ``com_pos`` -- CoM of voxels ``> threshold`` (tissue mass).
    2. ``com_neg`` -- CoM of voxels ``< -threshold``, weighted by
       ``|I(x)|`` (air/cavity mass); ``None`` if no significant negative
       voxels exist (standard MRI/PET/histology).

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2-D or 3-D image.
    weighted : bool
        If ``True``, intensity-weighted physical CoM; if ``False``,
        geometric CoM of the respective mask.
    threshold : float
        Boundary separating the positive/negative domains.
    return_dict : bool
        If ``True``, returns ``{'positive': com_pos, 'negative': com_neg}``.

    Returns
    -------
    tuple of (np.ndarray or None, np.ndarray or None), or dict
        ``(com_pos, com_neg)`` physical-space coordinates.
    """
    arr = img_ants.numpy()
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    dim = img_ants.dimension

    thresh = max(float(threshold), 1e-6)

    pos_mask = arr > thresh
    com_pos = None
    if pos_mask.sum() >= 10:
        w_pos = np.where(pos_mask, arr, 0.0).astype(np.float64) if weighted else pos_mask.astype(np.float64)
        tot_pos = float(w_pos.sum())
        if tot_pos > 1e-6:
            vox_pos = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                axes = tuple(j for j in range(dim) if j != i)
                vox_pos[i] = (w_pos.sum(axis=axes) * np.arange(arr.shape[i])).sum() / tot_pos
            com_pos = origin + direction @ (vox_pos * spacing)

    neg_mask = arr < -thresh
    com_neg = None
    if neg_mask.sum() >= 10:
        w_neg = np.where(neg_mask, -arr, 0.0).astype(np.float64) if weighted else neg_mask.astype(np.float64)
        tot_neg = float(w_neg.sum())
        if tot_neg > 1e-6:
            vox_neg = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                axes = tuple(j for j in range(dim) if j != i)
                vox_neg[i] = (w_neg.sum(axis=axes) * np.arange(arr.shape[i])).sum() / tot_neg
            com_neg = origin + direction @ (vox_neg * spacing)

    if return_dict:
        return {"positive": com_pos, "negative": com_neg}
    return com_pos, com_neg


def compute_center_of_mass(img_ants, weighted: bool = True, return_both: bool = False):
    """Physical center of mass of a 2-D or 3-D ``ants.ANTsImage``.

    Gracefully handles negative-valued images (e.g. CT Hounsfield units)
    without first-moment cancellation, via :func:`robust_center_of_mass`.

    Parameters
    ----------
    img_ants : ants.ANTsImage
    weighted : bool
        Intensity-weighted CoM if ``True``, else geometric CoM of the
        non-zero foreground mask.
    return_both : bool
        If ``True``, returns ``(com_pos, com_neg)`` directly.

    Returns
    -------
    np.ndarray or tuple
        Physical-space coordinates of the predominant mass (or both halves).
    """
    com_pos, com_neg = robust_center_of_mass(img_ants, weighted=weighted)
    if return_both:
        return com_pos, com_neg
    if com_pos is not None:
        return com_pos
    if com_neg is not None:
        return com_neg
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    center_vox = (np.array(img_ants.shape) - 1.0) / 2.0
    return origin + direction @ (center_vox * spacing)


def compute_fov_center(img_ants) -> np.ndarray:
    """Geometric physical center (field-of-view midpoint) of an ``ants.ANTsImage``."""
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    shape = np.array(img_ants.shape)
    voxel_center = (shape - 1.0) * 0.5
    return origin + direction @ (voxel_center * spacing)


def _default_affine_schedule(dim: int, preset: str = "default") -> list:
    """Multi-resolution optimization schedule (one dict per stage).

    ``level``: pyramid downsampling factor (avg-pool). ``iters``: Adam/L-BFGS
    steps. ``dof``: ``'rigid'`` (translation + rotation) or ``'affine'``
    (+ log-scale + shear). ``lr``: per-parameter-group learning rates
    ``(t, omega, scale, shear)``. ``eta_min``: cosine-annealing floor
    (``None`` = constant). ``select``: if ``True``, keep only the
    best-scoring path (full-sample MI at this level) after the stage.
    ``full_grid``: if ``True``, use the regular voxel grid instead of a
    random point sample. ``sampling``: regular-grid sampling fraction for
    ``full_grid`` stages. ``optimizer``: ``'adam'`` (default) or ``'lbfgs'``.

    Presets (3-D), tuned by syntx on the 90-pair Mindboggle cohort against
    the ANTs C++ affine:

    - ``'default'``: L4 rigid (exact grid) -> L2 affine (10% sample,
      100 it, best-of-``n_starts``) -> L1 affine (2% sample, 40 it).
      ~5s on Apple Silicon MPS.
    - ``'accurate'``: L4 rigid (exact) -> L2 affine (regular 50% grid,
      100 it, select) -> L1; ~20s.
    - ``'fast'``: L4 rigid -> L2 affine -> L1 affine, all point-sampled;
      ~4-6s.
    """
    if dim == 3:
        if preset == "accurate":
            return [
                dict(level=4, iters=50, dof="rigid", lr=(0.04, 0.008, 0.0, 0.0), eta_min=0.002, select=False, full_grid=True, sampling=1.0),
                dict(level=2, iters=100, dof="affine", lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True, full_grid=True, sampling=0.5),
                dict(level=1, iters=30, dof="affine", lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4, select=False),
            ]
        if preset == "fast":
            return [
                dict(level=4, iters=50, dof="rigid", lr=(0.04, 0.008, 0.0, 0.0), eta_min=0.002, select=False, sampling=0.50),
                dict(level=2, iters=50, dof="affine", lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True, sampling=0.05),
                dict(level=1, iters=30, dof="affine", lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4, select=False, sampling=0.01),
            ]
        return [
            dict(level=4, iters=50, dof="rigid", lr=(0.04, 0.008, 0.0, 0.0), eta_min=0.002, select=False, sampling=1.0),
            dict(level=2, iters=100, dof="affine", lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True, sampling=0.10),
            dict(level=1, iters=40, dof="affine", lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4, select=False, sampling=0.02),
        ]
    return [
        dict(level=2, iters=50, dof="affine", lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True),
        dict(level=1, iters=30, dof="affine", lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4, select=False),
    ]


def _gaussian_kernel_1d_sep(sigma: float, device):
    """A single 1-D discrete Gaussian kernel (for separable N-D convolution), plus its radius."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def _separable_gaussian_smooth(x: torch.Tensor, sigma: float, dim: int) -> torch.Tensor:
    """Separable Gaussian smoothing of a channel-first ``(B, C, *spatial)`` tensor, 2-D or 3-D.

    A local, dependency-free replacement for ``syntx.landmarks.blob``'s
    ``_separable_gaussian3d`` (that module is not ported here) -- same
    separable-conv1d-per-axis approach as
    :func:`antstorch.syn.core.smoothing.separable_1d_filter`, applied along
    every spatial axis with a single isotropic ``sigma`` (in voxels).
    """
    k1d, radius = _gaussian_kernel_1d_sep(sigma, x.device)
    out = x
    if dim == 2:
        kv = k1d.view(1, 1, -1, 1)
        kh = k1d.view(1, 1, 1, -1)
        out = F.conv2d(out, kv, padding=(radius, 0))
        out = F.conv2d(out, kh, padding=(0, radius))
    elif dim == 3:
        kz = k1d.view(1, 1, -1, 1, 1)
        ky = k1d.view(1, 1, 1, -1, 1)
        kx = k1d.view(1, 1, 1, 1, -1)
        out = F.conv3d(out, kz, padding=(radius, 0, 0))
        out = F.conv3d(out, ky, padding=(0, radius, 0))
        out = F.conv3d(out, kx, padding=(0, 0, radius))
    else:
        raise ValueError(f"Only 2-D and 3-D are supported, got {dim}D.")
    return out


def _se3_distance(A1: np.ndarray, t1: np.ndarray, A2: np.ndarray, t2: np.ndarray, domain_diam: float = 200.0) -> float:
    """Riemannian geodesic distance on SE(3) between candidate transforms ``(A1, t1)`` and ``(A2, t2)``.

    ``d(T1, T2) = theta(R1, R2) + ||t1 - t2||_2 / domain_diam``, where
    ``theta`` is the exact geodesic distance on SO(3) (or SO(2) in 2-D).
    """
    dim = A1.shape[0]
    if dim == 3:
        U1, _, V1t = np.linalg.svd(A1[:3, :3])
        R1 = U1 @ V1t
        if np.linalg.det(R1) < 0:
            R1 = U1 @ np.diag([1.0, 1.0, -1.0]) @ V1t
        U2, _, V2t = np.linalg.svd(A2[:3, :3])
        R2 = U2 @ V2t
        if np.linalg.det(R2) < 0:
            R2 = U2 @ np.diag([1.0, 1.0, -1.0]) @ V2t
        R_rel = R1.T @ R2
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(tr))
    else:
        theta1 = np.arctan2(A1[1, 0], A1[0, 0])
        theta2 = np.arctan2(A2[1, 0], A2[0, 0])
        dth = abs(theta1 - theta2) % (2.0 * np.pi)
        theta = float(min(dth, 2.0 * np.pi - dth))
    dt = float(np.linalg.norm(t1 - t2) / max(domain_diam, 1.0))
    return theta + dt


def _cluster_candidates_se3(
    scored_candidates: list,
    n_starts: int,
    cluster_threshold: float = 0.35,
    domain_diam: float = 200.0,
) -> list:
    """Cluster candidate transforms by SE(3) Riemannian distance; keep the best of each cluster.

    Guarantees hypothesis diversity across distinct capture basins, rather
    than ``n_starts`` near-duplicate local perturbations of the single best
    coarse-level candidate.

    Parameters
    ----------
    scored_candidates : list of (loss, name, A, t)
        Sorted ascending by ``loss`` (lowest first).
    n_starts : int
        Number of paths to keep.
    cluster_threshold : float
        SE(3) geodesic distance (radians-equivalent) below which two
        candidates are considered the same cluster.
    domain_diam : float
        Physical domain diagonal, for scaling the translation term.
    """
    if len(scored_candidates) <= n_starts:
        return scored_candidates
    clusters = []
    for cand in scored_candidates:
        loss, name, A, t = cand
        assigned = False
        for cl in clusters:
            rep = cl[0]
            dist = _se3_distance(A, t, rep[2], rep[3], domain_diam=domain_diam)
            if dist < cluster_threshold:
                cl.append(cand)
                assigned = True
                break
        if not assigned:
            clusters.append([cand])
    selected = [cl[0] for cl in clusters]
    if len(selected) >= n_starts:
        return selected[:n_starts]
    already_selected_names = {c[1] for c in selected}
    remaining = [c for c in scored_candidates if c[1] not in already_selected_names]
    selected.extend(remaining[: n_starts - len(selected)])
    return selected


class _AffinePath:
    """One optimization path: ``A = R(omega) @ B @ diag(exp(scale)) @ Shear(shear)``.

    The physical-space mapping is ``y = A @ (x - C) + C + t`` for a fixed
    center ``C`` (the fixed image's center of mass) shared by every path in
    a solve.
    """

    def __init__(self, B: np.ndarray, t0: np.ndarray, dim: int, device, name: str, spacing: Optional[Sequence[float]] = None):
        self.dim, self.name = dim, name
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.t = torch.tensor(t0, dtype=torch.float32, device=device, requires_grad=True)
        self.omega = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)
        self.scale = torch.zeros(dim, dtype=torch.float32, device=device, requires_grad=True)
        self.shear = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)

        self.w_scale = None
        self.w_shear = None
        if spacing is not None:
            sp = np.asarray(spacing, dtype=np.float32)[:dim]
            s_min = max(float(sp.min()), 1e-4)
            w = sp / s_min
            self.w_scale = torch.tensor(w, dtype=torch.float32, device=device)
            if dim == 3:
                w_sh = [w[0] * w[1], w[0] * w[2], w[1] * w[2]]
                self.w_shear = torch.tensor(w_sh, dtype=torch.float32, device=device)
            else:
                self.w_shear = torch.tensor([w[0] * w[1]], dtype=torch.float32, device=device)

    def regularization_loss(self, dof: str, lambda_shear: float = 0.02, lambda_scale: float = 0.01) -> torch.Tensor:
        """Anisotropy-weighted Tikhonov penalty on scale/shear (zero for ``dof='rigid'``)."""
        if dof != "affine":
            return torch.tensor(0.0, device=self.t.device)
        reg = torch.tensor(0.0, device=self.t.device)
        reg = reg + lambda_scale * torch.sum((self.w_scale if self.w_scale is not None else 1.0) * (self.scale ** 2))
        reg = reg + lambda_shear * torch.sum((self.w_shear if self.w_shear is not None else 1.0) * (self.shear ** 2))
        return reg

    def matrix(self, dof: str) -> torch.Tensor:
        dim = self.dim
        R = get_rotation_matrix(self.omega, dim)
        A = R @ self.B
        if dof == "affine":
            S = torch.diag(torch.exp(torch.clamp(self.scale, -0.4, 0.4)))
            Sh = torch.eye(dim, device=A.device, dtype=A.dtype)
            if dim == 3:
                Sh = Sh.clone(); Sh[0, 1] = self.shear[0]; Sh[0, 2] = self.shear[1]; Sh[1, 2] = self.shear[2]
            else:
                Sh = Sh.clone(); Sh[0, 1] = self.shear[0]
            A = A @ S @ Sh
        return A

    def params(self, dof: str, lr):
        groups = [{"params": [self.t], "lr": lr[0]}, {"params": [self.omega], "lr": lr[1]}]
        if dof == "affine":
            groups += [{"params": [self.scale], "lr": lr[2]}, {"params": [self.shear], "lr": lr[3]}]
        return groups

    @torch.no_grad()
    def clamp_(self):
        self.scale.clamp_(-0.35, 0.35)
        self.shear.clamp_(-0.35, 0.35)
        self.omega.clamp_(-np.pi / 3, np.pi / 3)

    @torch.no_grad()
    def numpy_affine(self):
        """``(A, t)`` in the ``y = A(x - C) + C + t`` convention, ``float64`` numpy arrays."""
        A = self.matrix("affine").detach().cpu().numpy().astype(np.float64)
        return A, self.t.detach().cpu().numpy().astype(np.float64)


def _run_pytorch_affine_solver(
    fixed,
    moving,
    initial_tx_path: Optional[str] = None,
    device: str = "auto",
    verbose: bool = False,
    multi_start: bool = True,
    n_starts: int = 3,
    cone_angles_deg: Optional[list] = None,
    seed: int = 42,
    schedule: Optional[list] = None,
    preset: str = "default",
    sampling_percentage: Optional[float] = None,
    num_bins: int = 32,
    n_sample_points: Optional[int] = None,
    fixed_range=(0.0, 1.0),
    mask_mode: str = "none",
    smooth_sigma_per_level: float = 0.0,
    fg_dice_weight: float = 0.0,
    fg_level: float = 0.01,
    sample_weighting: str = "uniform",
    sample_seed: Optional[int] = None,
    enable_landmarks: bool = False,
    lambda_shear: float = 0.02,
    lambda_scale: float = 0.01,
    cluster_threshold: float = 0.35,
    **kwargs,
) -> dict:
    """Native PyTorch multi-resolution affine solver (``mode='pytorch'``), Mattes MI objective.

    Deterministic by construction: seeded candidate generation, the
    deterministic-blocked-histogram Mattes MI, and (optionally) a fixed
    seeded point sample per level.

    Parameters
    ----------
    initial_tx_path : str, optional
        ITK ``.mat`` file used as an additional start candidate.
    multi_start, n_starts, cone_angles_deg : bool, int, list, optional
        Coarse-level candidate search: identity-at-CoM plus single-axis
        cone rotations (default +-6 deg to +-24 deg at 6 deg steps in 3-D);
        the ``n_starts`` (default 3) SE(3)-diverse best-MI candidates are
        optimized in parallel until the schedule's ``select`` stage, then
        only the best path continues.
    schedule, preset : list, str
        See :func:`_default_affine_schedule`.
    sampling_percentage, n_sample_points, num_bins, fixed_range
        Mattes MI settings; images are foreground-normalized to ``[0, 1]``.
    mask_mode : {'none', 'union', 'fixed_fg'}
        Which voxels enter the MI histogram.
    smooth_sigma_per_level : float
        Anti-aliased pyramid: Gaussian-smooth (voxel-unit sigma) before
        avg-pooling at each level; ``0`` disables.
    sample_weighting : {'uniform', 'gradient'}
        Point-sample distribution; ``'gradient'`` concentrates samples near
        tissue boundaries (weighted by ``|grad(fixed)|``, via
        ``torch.gradient`` -- a dependency-free stand-in for
        ``syntx.landmarks.blob``'s shift-pad finite differences).
    fg_dice_weight : float
        Weight of an additional soft-Dice term between fixed/warped-moving
        foreground masks; ``0`` disables.
    enable_landmarks : bool
        Not supported by this port (``syntx.landmarks`` is not ported);
        ``True`` emits a warning and is otherwise ignored.
    lambda_shear, lambda_scale : float
        Anisotropy-weighted Tikhonov regularization for shear/scale.
    cluster_threshold : float
        SE(3) geodesic clustering threshold (radians) for hypothesis diversity.

    Returns
    -------
    dict
        ``'warpedmovout'``, ``'fwdtransforms'``, ``'invtransforms'``,
        ``'whichtoinvert_inv'``, ``'time'``/``'runtime_seconds'``, plus
        diagnostic fields (``'init_candidate'``, ``'candidates_scored'``, ...).
    """
    import ants

    if enable_landmarks:
        warnings.warn(
            "enable_landmarks=True has no effect: antstorch.syn.robust_affine does not port "
            "syntx.landmarks (SIFT3D-seeded candidates). Proceeding without it.",
            RuntimeWarning, stacklevel=2,
        )

    t0 = time.time()
    dim = fixed.dimension
    seed = 42 if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device in ("auto", None):
        device_obj = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device_obj = torch.device(device)
    device_obj = _resolve_solver_device(device_obj, dim)
    schedule = schedule or _default_affine_schedule(dim, preset)
    if sampling_percentage is None and n_sample_points is None:
        sampling_percentage = {"fast": 0.01, "accurate": 0.20}.get(preset, 0.02)
    n_starts = max(1, int(n_starts))
    sp_fix = fixed.spacing if hasattr(fixed, "spacing") else None

    # 1. centers, normalization, tensors --------------------------------------------------
    com_f_pos, com_f_neg = robust_center_of_mass(fixed, weighted=True)
    com_m_pos, com_m_neg = robust_center_of_mass(moving, weighted=True)
    com_f = np.asarray(com_f_pos if com_f_pos is not None else (com_f_neg if com_f_neg is not None else compute_center_of_mass(fixed, weighted=False)), dtype=np.float64)
    com_m = np.asarray(com_m_pos if com_m_pos is not None else (com_m_neg if com_m_neg is not None else compute_center_of_mass(moving, weighted=False)), dtype=np.float64)
    t_init = com_m - com_f

    fixed_norm = normalize_image(fixed, method="auto")
    moving_norm = normalize_image(moving, method="auto")
    fi_arr = ants_image_to_tensor(fixed_norm, device=device_obj, normalize=False)
    mi_arr = ants_image_to_tensor(moving_norm, device=device_obj, normalize=False)

    f32 = dict(dtype=torch.float32, device=device_obj)
    sp_xyz = torch.tensor(fixed.spacing, **f32); orig_xyz = torch.tensor(fixed.origin, **f32); dir_xyz = torch.tensor(fixed.direction, **f32)
    C_phys = torch.tensor(com_f, **f32)
    mi_orig = torch.tensor(moving.origin, **f32); mi_sp = torch.tensor(moving.spacing, **f32)
    mi_dir_inv_t = torch.inverse(torch.tensor(moving.direction, **f32)).t(); mi_shape = torch.tensor(moving.shape, **f32)

    # 2. pyramid: per level, the fixed image and physical coordinates of its (sampled) voxels
    levels = sorted({int(s_["level"]) for s_ in schedule}, reverse=True)
    pyr = {}
    gen = torch.Generator(device="cpu").manual_seed(seed if sample_seed is None else int(sample_seed))
    for level in levels:
        fi_src, mi_src = fi_arr, mi_arr
        sig = smooth_sigma_per_level * max(level - 1, 0)
        if sig > 0:
            fi_src = _separable_gaussian_smooth(fi_arr, sig, dim)
            mi_src = _separable_gaussian_smooth(mi_arr, sig, dim)
        if level > 1:
            pool = F.avg_pool3d if dim == 3 else F.avg_pool2d
            fi_lev, mi_lev = pool(fi_src, kernel_size=level, stride=level), pool(mi_src, kernel_size=level, stride=level)
        else:
            fi_lev, mi_lev = fi_src, mi_src
        shape_zyx = fi_lev.shape[2:]
        axes = [torch.arange(n, device=device_obj, dtype=torch.float32) for n in shape_zyx]
        mesh = torch.meshgrid(*axes, indexing="ij")
        # a pooled voxel i averages full-res voxels [level*i, level*i + level - 1]: its center is
        # level*i + (level-1)/2 in full-res continuous index space.
        vox_xyz = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, dim) * level + (level - 1) / 2.0
        phys_xyz = orig_xyz + (vox_xyz * sp_xyz) @ dir_xyz.t()
        fmask = fi_lev > fg_level
        mi_shape_pooled = torch.tensor(list(reversed(mi_lev.shape[2:])), **f32)
        entry = dict(fi=fi_lev, mi=mi_lev, mi_mask=(mi_lev > fg_level).to(mi_lev.dtype), shape=shape_zyx, phys=phys_xyz,
                     mask=fmask, points=None, level=level, mi_shape_pooled=mi_shape_pooled)

        stage_samplings = [s.get("sampling") for s in schedule if int(s["level"]) == level and s.get("sampling") is not None]
        level_sampling = stage_samplings[0] if stage_samplings else sampling_percentage

        k = None
        if n_sample_points is not None:
            domain = fmask.reshape(-1) if mask_mode == "fixed_fg" else torch.ones_like(fmask.reshape(-1))
            idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
            k = min(int(n_sample_points), idx_fg.numel())
        elif level_sampling is not None and 0.0 < float(level_sampling) < 1.0:
            domain = fmask.reshape(-1) if mask_mode == "fixed_fg" else torch.ones_like(fmask.reshape(-1))
            idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
            k = max(1000, min(int(round(idx_fg.numel() * float(level_sampling))), idx_fg.numel()))

        if k is not None:
            if sample_weighting == "gradient":
                grads = torch.gradient(fi_lev, dim=tuple(range(2, fi_lev.ndim)))
                gm = sum(g ** 2 for g in grads)
                w = (0.1 + torch.sqrt(gm).reshape(-1) / (torch.sqrt(gm).max() + 1e-8)).cpu()[idx_fg]
                gumbel = -torch.log(-torch.log(torch.rand(idx_fg.numel(), generator=gen).clamp_min(1e-12)))
                keys = torch.log(w) + gumbel
                sel = idx_fg[torch.topk(keys, k).indices].sort().values.to(device_obj)
            else:
                sel = idx_fg[torch.randperm(idx_fg.numel(), generator=gen)[:k]].sort().values.to(device_obj)
            entry["points"] = dict(phys=phys_xyz[sel], fvals=fi_lev.reshape(-1)[sel])
        pyr[level] = entry

    fixed_w_cache: dict = {}

    def warp_to_moving_norm(y_phys, e=None):
        y_vox = (y_phys - mi_orig) @ mi_dir_inv_t / mi_sp
        if e is None or e["level"] == 1:
            return 2.0 * (y_vox / (mi_shape - 1.0)) - 1.0
        lvl = e["level"]
        y_pooled = (y_vox - (lvl - 1) / 2.0) / lvl
        return 2.0 * (y_pooled / (e["mi_shape_pooled"] - 1.0)) - 1.0

    def objective(path: "_AffinePath", dof: str, level: int, full: bool = False, sampling: Optional[float] = None):
        e = pyr[level]
        samp = sampling_percentage if sampling is None else sampling
        A = path.matrix(dof)
        teff = path.t + C_phys - A @ C_phys
        if e["points"] is not None and not full:
            y = warp_to_moving_norm(e["points"]["phys"] @ A.t() + teff, e)
            grid = y.reshape(1, 1, *([1] * (dim - 2)), -1, dim) if dim == 3 else y.reshape(1, 1, -1, dim)
            w = F.grid_sample(e["mi"], grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
            fv = e["points"]["fvals"]
            m = None
            if mask_mode == "union":
                m = (fv > fg_level) | (w > fg_level)
            loss = mattes_mi_loss_nd(w, fv, mask=m, num_bins=num_bins, auto_mask=False, fixed_range=fixed_range)
            if fg_dice_weight > 0:
                wm = F.grid_sample(e["mi_mask"], grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
                fm = (fv > fg_level).to(wm.dtype)
                loss = loss + fg_dice_weight * (1.0 - 2.0 * (fm * wm).sum() / (fm.sum() + wm.sum() + 1e-6))
            if lambda_shear > 0 or lambda_scale > 0:
                loss = loss + path.regularization_loss(dof, lambda_shear=lambda_shear, lambda_scale=lambda_scale)
            return loss
        y = warp_to_moving_norm(e["phys"] @ A.t() + teff, e)
        grid = y.reshape(1, *e["shape"], dim)
        w = F.grid_sample(e["mi"], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        fw = None
        if mask_mode == "union":
            m = e["mask"] | (w > 0.01)
        elif mask_mode == "none":
            m = torch.ones_like(e["mask"])
            key = (level, samp, num_bins)
            if key not in fixed_w_cache:
                fv = e["fi"].flatten()
                if samp is not None and samp < 1.0:
                    fv = fv[:: max(1, int(1.0 / samp))].contiguous()
                lo, hi = fixed_range if not isinstance(fixed_range[0], (tuple, list)) else fixed_range[1]
                fixed_w_cache[key] = parzen_weights((fv - lo) / (hi - lo + 1e-8) * 2.0 - 1.0, num_bins).detach()
            fw = fixed_w_cache[key]
        else:
            m = e["mask"]
        loss = mattes_mi_loss_nd(w, e["fi"], mask=m, num_bins=num_bins, auto_mask=False,
                                  sampling_percentage=samp, fixed_range=fixed_range, fixed_weights=fw)
        if fg_dice_weight > 0:
            wm = F.grid_sample(e["mi_mask"], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            fm = e["mask"].to(wm.dtype)
            loss = loss + fg_dice_weight * (1.0 - 2.0 * (fm * wm).sum() / (fm.sum() + wm.sum() + 1e-6))
        if lambda_shear > 0 or lambda_scale > 0:
            loss = loss + path.regularization_loss(dof, lambda_shear=lambda_shear, lambda_scale=lambda_scale)
        return loss

    # 3. candidate starts, scored at the coarsest level -------------------------------------
    cands = []
    if initial_tx_path is not None and os.path.exists(str(initial_tx_path)):
        M0, t0_zero = parse_ants_affine(initial_tx_path, dim)
        if M0 is not None:
            A0 = M0.numpy().astype(np.float64)
            t_zero = t0_zero.numpy().astype(np.float64)
            # parse_ants_affine already resolves the file's own center of rotation into the
            # zero-centered convention y = A0 @ x + t_zero; re-center onto com_f (C):
            # y = A0 (x - C) + C + t_new  =>  t_new = t_zero + A0 @ C - C
            t_new = t_zero + A0 @ com_f - com_f
            cands.append(("Provided_Initial_Transform", A0, t_new))

    if not cands or multi_start:
        cands.append(("Identity_CoM", np.eye(dim), t_init))

        if com_f_neg is not None and com_m_neg is not None and multi_start:
            t_neg = com_m_neg - com_f_neg
            cands.append(("Identity_CoM_Negative", np.eye(dim), t_neg))
            if com_f_pos is not None and com_m_pos is not None and dim == 3:
                v_f = com_f_pos - com_f_neg
                v_m = com_m_pos - com_m_neg
                len_f, len_m = np.linalg.norm(v_f), np.linalg.norm(v_m)
                if len_f > 10.0 and len_m > 10.0:
                    u_f, u_m = v_f / len_f, v_m / len_m
                    dot_u = float(np.dot(u_f, u_m))
                    if -0.999 < dot_u < 0.98:
                        v_cross = np.cross(u_f, u_m)
                        s = np.linalg.norm(v_cross)
                        c = dot_u
                        if s > 1e-4:
                            vx = np.array([[0, -v_cross[2], v_cross[1]], [v_cross[2], 0, -v_cross[0]], [-v_cross[1], v_cross[0], 0]])
                            R_dipole = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
                            cands.append(("CoM_Dipole_Vector", R_dipole, t_init))

        if multi_start:
            if cone_angles_deg is None:
                cone_angles_deg = [-24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0]
            for deg in cone_angles_deg:
                if abs(deg) < 1e-3:
                    continue
                rad = np.radians(deg)
                if dim == 3:
                    for axis in ("pitch", "roll", "yaw"):
                        rx = rad if axis == "pitch" else 0.0; ry = rad if axis == "roll" else 0.0; rz = rad if axis == "yaw" else 0.0
                        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                        cands.append((f"CoM_{axis}_{deg:+.0f}deg", Rz @ Ry @ Rx, t_init))
                else:
                    cands.append((f"CoM_rot_{deg:+.0f}deg", np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]), t_init))

    coarse = levels[0]
    scored = []
    with torch.no_grad():
        for name, B, tt in cands:
            p = _AffinePath(B, tt, dim, device_obj, name, spacing=sp_fix)
            scored.append((float(objective(p, "rigid", coarse, full=True, sampling=1.0).item()), name, B, tt))
    scored.sort(key=lambda x: x[0])

    domain_diam = 200.0
    if hasattr(fixed, "spacing") and hasattr(fixed, "shape"):
        domain_diam = float(np.linalg.norm(np.array(fixed.spacing) * np.array(fixed.shape)))

    prov = [s for s in scored if s[1] == "Provided_Initial_Transform"]
    if prov:
        others = [s for s in scored if s[1] != "Provided_Initial_Transform"]
        clustered_others = _cluster_candidates_se3(others, n_starts - len(prov), cluster_threshold=cluster_threshold, domain_diam=domain_diam)
        keep = (prov + clustered_others)[:n_starts]
    else:
        keep = _cluster_candidates_se3(scored, n_starts, cluster_threshold=cluster_threshold, domain_diam=domain_diam)

    if verbose:
        print(f"[robust_affine mode='pytorch'] {len(cands)} start candidates scored at level {coarse}; "
              f"keeping {[f'{n} ({s:.4f})' for s, n, _, _ in keep]}", flush=True)
    paths = [_AffinePath(B, tt, dim, device_obj, name, spacing=sp_fix) for _, name, B, tt in keep]

    # 4. schedule -------------------------------------------------------------------------
    last_loss = {}
    for si, stage in enumerate(schedule):
        level, iters, dof, lr, eta_min = int(stage["level"]), int(stage["iters"]), stage["dof"], stage["lr"], stage.get("eta_min")
        use_full = bool(stage.get("full_grid", False)); samp = stage.get("sampling", None)
        if stage.get("optimizer", "adam") == "lbfgs":
            for p in paths:
                groups = p.params(dof, lr)
                plist = [g["params"][0] for g in groups]
                opt = torch.optim.LBFGS(plist, lr=1.0, max_iter=iters, history_size=int(stage.get("history", 10)),
                                         tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe")

                def closure(p=p):
                    opt.zero_grad(set_to_none=True)
                    l_ = objective(p, dof, level, full=use_full, sampling=samp)
                    l_.backward()
                    return l_

                loss = opt.step(closure)
                p.clamp_()
                last_loss[p.name] = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        else:
            opts = [torch.optim.Adam(p.params(dof, lr)) for p in paths]
            scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=iters, eta_min=eta_min) for o in opts] if eta_min is not None else []
            for it in range(iters):
                for pi, (p, opt) in enumerate(zip(paths, opts)):
                    opt.zero_grad(set_to_none=True)
                    loss = objective(p, dof, level, full=use_full, sampling=samp)
                    loss.backward()
                    opt.step()
                    if scheds:
                        scheds[pi].step()
                    p.clamp_()
                    last_loss[p.name] = float(loss.item())
        if stage.get("select", False) and len(paths) > 1:
            with torch.no_grad():
                full_scores = [float(objective(p, dof, level, full=True, sampling=1.0).item()) for p in paths]
            best = int(np.argmin(full_scores))
            if verbose:
                print(f"  stage {si} (level {level}): path scores {[(p.name, round(s_, 4)) for p, s_ in zip(paths, full_scores)]} -> keep '{paths[best].name}'", flush=True)
            paths = [paths[best]]
    if len(paths) > 1:
        with torch.no_grad():
            full_scores = [float(objective(p, schedule[-1]["dof"], int(schedule[-1]["level"]), full=True, sampling=1.0).item()) for p in paths]
        paths = [paths[int(np.argmin(full_scores))]]
    winner = paths[0]

    # 5. export --------------------------------------------------------------------------
    A_fin, t_fin = winner.numpy_affine()
    # convert from the com_f-centered convention y = A(x - C) + C + t back to the zero-centered
    # convention write_affine_transform()/parse_ants_affine() use, y = A @ x + t_zero:
    t_zero_fin = t_fin + com_f - A_fin @ com_f
    out_dir = tempfile.mkdtemp(prefix="robust_affine_pt_")
    final_tx_path = os.path.join(out_dir, "affine.mat")
    write_affine_transform(A_fin, t_zero_fin, dim, final_tx_path)
    warped_mov_out = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[final_tx_path], interpolator="linear")
    elapsed = time.time() - t0
    return {
        "warpedmovout": warped_mov_out,
        "fwdtransforms": [final_tx_path],
        "invtransforms": [final_tx_path],
        "whichtoinvert_inv": [True],
        "runtime_seconds": elapsed,
        "time": elapsed,
        "init_candidate": winner.name,
        "init_score": float(keep[0][0]),
        "final_loss": float(last_loss.get(winner.name, float("nan"))),
        "candidates_scored": [(n, s) for s, n, _, _ in scored],
        "status": "SUCCESS",
    }


def robust_affine(
    fixed,
    moving,
    initial_transform: Optional[str] = None,
    mode: str = "auto",
    multi_start: bool = True,
    n_starts: int = 3,
    cone_angles_deg: Optional[list] = None,
    backend: str = "pytorch",
    device: str = "auto",
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Fail-safe multi-start initial affine registration for 2-D and 3-D images.

    Ported from ``syntx.robust_affine.robust_affine``, restricted to the
    modes backed by the pure-PyTorch solver -- see this module's docstring
    for what was intentionally left out (landmarks, tournament, the legacy
    multi-candidate ANTs C++ fallback).

    Supported modes (``mode``)
    ---------------------------
    - ``'auto'`` / ``'fast'`` (default): the native PyTorch multi-resolution
      Mattes MI solver, with a **simple** fallback to a single
      ``ants.registration(type_of_transform='Affine')`` call (seeded at
      ``initial_transform`` if given) if the solver raises. This is a
      deliberate simplification of syntx's own ``'auto'`` fallback, which
      instead re-runs its full low-res multi-candidate ANTs C++ search --
      not ported here.
    - ``'pytorch'`` / ``'gpu'``: the same solver, without any fallback
      (re-raises on failure).
    - ``'com_only'`` / ``'translation_only'``: instant center-of-mass
      physical translation alignment (no optimization).

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
        Fixed/moving images, 2-D or 3-D.
    initial_transform : str, optional
        Path to an existing ITK ``.mat`` transform, used as an additional
        multi-start candidate.
    mode : str
        See above.
    multi_start, n_starts, cone_angles_deg
        Coarse-level candidate search settings; see
        :func:`_run_pytorch_affine_solver`.
    backend : str
        Present for interface parity with ``syntx.robust_affine``; only
        ``'pytorch'`` is supported here (no JAX backend).
    device : str
        ``'auto'`` (best available accelerator), ``'cpu'``, ``'cuda'``, or
        ``'mps'``.
    seed : int, optional
        Random seed (default: ``42``).
    verbose : bool
        Print diagnostic timing/score messages.
    **kwargs
        Forwarded to :func:`_run_pytorch_affine_solver` (e.g. ``preset``,
        ``num_bins``, ``sampling_percentage``, ``mask_mode``,
        ``fg_dice_weight``, ``lambda_shear``, ``lambda_scale``,
        ``cluster_threshold``).

    Returns
    -------
    dict
        ``'fwdtransforms'``, ``'invtransforms'``, ``'whichtoinvert_inv'``,
        ``'warpedmovout'``, ``'warpedfixout'``, ``'time'``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values, or ``backend`` is
        not ``'pytorch'``.
    """
    import ants

    if backend != "pytorch":
        raise ValueError(f"backend={backend!r} is not supported; antstorch.syn.robust_affine is PyTorch-only")

    t0 = time.time()
    if seed is None:
        seed = 42

    if mode in ("com_only", "translation_only"):
        com_f = compute_center_of_mass(fixed, weighted=True)
        com_m = compute_center_of_mass(moving, weighted=True)
        t_com = com_m - com_f
        dim = fixed.dimension
        out_dir = tempfile.mkdtemp(prefix="robust_aff_com_")
        tx_path = os.path.join(out_dir, "translation.mat")
        write_affine_transform(np.eye(dim), t_com, dim, tx_path)
        warped_mov = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        return {
            "fwdtransforms": [tx_path],
            "invtransforms": [tx_path],
            "whichtoinvert_inv": [True],
            "warpedmovout": warped_mov,
            "warpedfixout": fixed,
            "time": time.time() - t0,
        }

    if mode not in ("pytorch", "gpu", "pytorch_gpu", "auto", "fast"):
        raise ValueError(
            f"mode={mode!r} is not supported by this port. Supported: 'com_only', 'translation_only', "
            "'pytorch', 'gpu', 'pytorch_gpu', 'auto', 'fast' (landmarks/tournament/legacy ANTs "
            "multi-candidate fallback are not ported -- see this module's docstring)."
        )

    try:
        result = _run_pytorch_affine_solver(
            fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose,
            multi_start=multi_start, n_starts=n_starts, cone_angles_deg=cone_angles_deg,
            seed=seed, **kwargs,
        )
        result.setdefault("warpedfixout", fixed)
        return result
    except Exception as exc:
        if mode in ("pytorch", "gpu", "pytorch_gpu"):
            raise
        logger.warning("robust_affine: PyTorch solver failed (%s); falling back to a direct ants.registration() call", exc)
        if verbose:
            print(f"[robust_affine] PyTorch solver failed ({exc}); falling back to ants.registration().", flush=True)
        reg_a = ants.registration(
            fixed=fixed, moving=moving, type_of_transform="Affine",
            initial_transform=initial_transform, verbose=verbose,
        )
        return {
            "fwdtransforms": reg_a["fwdtransforms"],
            "invtransforms": reg_a["invtransforms"],
            "whichtoinvert_inv": [True],
            "warpedmovout": reg_a["warpedmovout"],
            "warpedfixout": reg_a.get("warpedfixout", fixed),
            "time": time.time() - t0,
        }
