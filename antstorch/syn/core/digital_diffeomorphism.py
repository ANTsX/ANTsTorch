"""
Digital diffeomorphism constraint & post-composition projection.

Port of ``syntx.core.digital_diffeomorphism`` (syntx v6.39.0, 2026-11-18), PyTorch functions only
(the JAX wrapper and the ANTs-image convenience wrapper of syntx are not ported). Used by
:func:`antstorch.syn.syn.syn_registration` and :func:`antstorch.greedy.greedy_registration`
(``digital_diffeomorphism=True``, the syntx-aligned default). The numerics are kept line for line;
see the upstream module for the derivations.

Continuous diffeomorphism guarantees (positive Euler step Jacobians along a velocity flow in
``tvf`` / ``syngs``, or positive half-field Jacobians at the midpoint in ``syn`` /
``syn_scattered``) do not automatically guarantee a **digital diffeomorphism** on the discrete
voxel lattice: multilinear re-tabulation of composed maps ``Phi_12 = phi_2 o phi_1^-1`` or
discrete trajectory integration can create isolated 2-voxel central-difference secant folds
(``det(I + D_{2h} u) <= 0``) or sub-voxel creases where asymmetric inverse polishing
(holding ``W_fwd`` fixed) cannot invert a locally 2-to-1 discrete grid map.

This module provides model-agnostic **digital diffeomorphism** primitives across all ``syntx``
registration families (``syn``, ``tvf``, ``syngs``, ``greedy``, ``syn_scattered``):

1. ``enforce_digital_diffeomorphism_nd``:
   - **Symmetric paired mode** (``W_inv is not None``): jointly updates ``(W_fwd, W_inv)`` on
     the sparse 2-ring dilated active defect neighborhood where bidirectional chord ICE exceeds
     ``ice_tol`` (voxels) or physical central-difference Jacobian determinant
     ``det(I + D_{2h} u)`` falls below ``jac_floor`` (respecting full ``spacing`` and
     ``direction`` cosines, or normalized ``[-1, 1]`` coordinates when ``spacing`` is omitted).
   - **Unidirectional mode** (``W_inv is None``, e.g. ``syntx.greedy`` with
     ``return_inverse=False``): unfolds any physical central-difference Jacobian violations
     ``det(I + D_{2h} W_fwd) < jac_floor`` under a proximity anchor ``anchor_weight * ||u - u_0||^2``
     without requiring an inverse field.
   - **Strict idempotency & locality**: if no interior voxel violates ``ice_tol`` or
     ``jac_floor``, returns inputs bitwise untouched with ``stats['steps'] == 0``. Otherwise,
     all voxels outside the dilated defect neighborhood remain bitwise identical.

2. ``pullback_digital_diffeomorphism_to_halves``:
   - Propagates a small endpoint correction ``delta_fwd = W_fwd_polished - W_fwd_init``
     symmetrically back onto midpoint half-fields ``(w_l2r, w_r2l, w_l2r_inv, w_r2l_inv)`` in
     both physical-mm layout (``syntx.syn``) and normalized ``[-1, 1]`` layout
     (``syntx.syn_scattered``), preserving half-field diffeomorphic integrity and keeping
     ``syntx.liouville_determinant`` consistent with exported endpoints.

(syntx's ``enforce_digital_diffeomorphism_jax`` and ANTs-image ``enforce_digital_diffeomorphism`` wrappers are not
ported.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .grid import (
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    sample_field_cf,
)
from .inverse import (
    _multilinear_val_and_jac_torch,
    _solve_spd_lm_torch,
)
from .smoothing import get_boundary_mask


def _cofactor_transpose_nd(F_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(det(F), cof(F))`` for ``(..., dim, dim)`` matrices ``F_mat`` in 2-D or 3-D,
    where ``cof(F)[..., i, j] = d det(F) / d F[..., i, j]``."""
    dim = F_mat.shape[-1]
    if dim == 2:
        a = F_mat[..., 0, 0]
        b = F_mat[..., 0, 1]
        c = F_mat[..., 1, 0]
        d = F_mat[..., 1, 1]
        det = a * d - b * c
        r0 = torch.stack([d, -c], dim=-1)
        r1 = torch.stack([-b, a], dim=-1)
        return det, torch.stack([r0, r1], dim=-2)

    a, b, c = F_mat[..., 0, 0], F_mat[..., 0, 1], F_mat[..., 0, 2]
    d, e, f = F_mat[..., 1, 0], F_mat[..., 1, 1], F_mat[..., 1, 2]
    g, h, k = F_mat[..., 2, 0], F_mat[..., 2, 1], F_mat[..., 2, 2]

    c00 = e * k - f * h
    c01 = -(d * k - f * g)
    c02 = d * h - e * g

    c10 = -(b * k - c * h)
    c11 = a * k - c * g
    c12 = -(a * h - b * g)

    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02
    cof = torch.stack(
        [
            torch.stack([c00, c01, c02], dim=-1),
            torch.stack([c10, c11, c12], dim=-1),
            torch.stack([c20, c21, c22], dim=-1),
        ],
        dim=-2,
    )
    return det, cof


def _physical_fd_det_only(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_inv: torch.Tensor,
) -> torch.Tensor:
    """Fast 1-pass physical central-difference ``det(I + (du/d_axis) @ D_rev_inv)`` without
    allocating cofactor tensors or running the 3-axis adjoint gradient pass."""
    dim = u.shape[-1]
    grads = torch.gradient(u, spacing=steps_rev, dim=tuple(range(1, dim + 1)))
    G = torch.stack(grads, dim=-1)  # (B, *spatial, comp_i, axis_j)
    del grads
    eye = torch.eye(dim, device=u.device, dtype=u.dtype)
    F_phys = G @ D_rev_inv + eye
    del G
    if dim == 2:
        return F_phys[..., 0, 0] * F_phys[..., 1, 1] - F_phys[..., 0, 1] * F_phys[..., 1, 0]
    a, b, c = F_phys[..., 0, 0], F_phys[..., 0, 1], F_phys[..., 0, 2]
    d, e, f = F_phys[..., 1, 0], F_phys[..., 1, 1], F_phys[..., 1, 2]
    g, h, k = F_phys[..., 2, 0], F_phys[..., 2, 1], F_phys[..., 2, 2]
    return a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)


def _det_only_nd(F_mat: torch.Tensor) -> torch.Tensor:
    """Compute ``det(F_mat)`` for ``(..., dim, dim)`` matrices in 2-D or 3-D without cofactors."""
    if F_mat.shape[-1] == 2:
        return F_mat[..., 0, 0] * F_mat[..., 1, 1] - F_mat[..., 0, 1] * F_mat[..., 1, 0]
    a, b, c = F_mat[..., 0, 0], F_mat[..., 0, 1], F_mat[..., 0, 2]
    d, e, f = F_mat[..., 1, 0], F_mat[..., 1, 1], F_mat[..., 1, 2]
    g, h, k = F_mat[..., 2, 0], F_mat[..., 2, 1], F_mat[..., 2, 2]
    return a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)


def _det_from_rows_nd(rows: list[torch.Tensor], s_D: float) -> torch.Tensor:
    """Compute ``s_D * det(stack(rows, dim=-2))`` directly from ``dim`` contiguous ``(..., dim)``
    row tensors with zero ``(..., dim, dim)`` allocation, zero ``.transpose``, and zero ``@ D_inv``."""
    if len(rows) == 2:
        r0, r1 = rows[0], rows[1]
        return float(s_D) * (r0[..., 0] * r1[..., 1] - r0[..., 1] * r1[..., 0])
    r0, r1, r2 = rows[0], rows[1], rows[2]
    return float(s_D) * (r0 * torch.linalg.cross(r1, r2, dim=-1)).sum(dim=-1)


def _d_rev_t_and_scale(D_rev_inv: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Invert tiny ``(dim, dim)`` ``D_rev_inv`` on CPU in float64 to obtain ``D_rev.T`` and
    ``s_D = det(D_rev_inv)`` so ``det(diff.T @ D_rev_inv + I) == s_D * det(D_rev.T + diff)``."""
    d_inv_np = D_rev_inv.detach().cpu().numpy().astype(np.float64)
    d_rev_np = np.linalg.inv(d_inv_np)
    s_D = float(np.linalg.det(d_inv_np))
    D_rev_T = torch.as_tensor(d_rev_np.T.copy(), device=D_rev_inv.device, dtype=D_rev_inv.dtype)
    return D_rev_T, s_D


def _eval_dual_fd_det_min_fast(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_T: torch.Tensor,
    s_D: float,
    include_4th: bool,
    use_physical: bool,
) -> torch.Tensor:
    """Single-pass fused 2nd-order + 4th-order physical Jacobian minimum ``min(det2, det4)``
    sharing ``i-1, i+1`` neighbor reads, allocating zero 6-D ``(..., dim, dim)`` tensors, and
    using the exact determinant identity ``det(I + G D_rev^-1) = s_D * det(D_rev^T + G_ax)``."""
    dim = u.shape[-1]
    u_fd = u if use_physical else torch.flip(u, dims=[-1])
    r2_list: list[torch.Tensor] = []
    r4_list: list[torch.Tensor] = []
    view_bcast_base = [1] * (u_fd.ndim - 1) + [dim]
    for ax in range(dim):
        n = u_fd.shape[ax + 1]
        h = float(steps_rev[ax])
        d_row = D_rev_T[ax].view(*view_bcast_base)
        if n <= 1:
            z_row = d_row.expand_as(u_fd)
            r2_list.append(z_row)
            if include_4th:
                r4_list.append(z_row)
            continue
        idx = torch.arange(n, device=u_fd.device)
        im1 = (idx - 1).clamp(0, n - 1)
        ip1 = (idx + 1).clamp(0, n - 1)
        u_m1 = torch.index_select(u_fd, ax + 1, im1)
        u_p1 = torch.index_select(u_fd, ax + 1, ip1)
        span2 = ((ip1 - im1).to(u_fd.dtype) * h).clamp(min=1e-12)
        sh_ax = [1] * u_fd.ndim
        sh_ax[ax + 1] = n
        inv_s2 = (1.0 / span2).view(*sh_ax)
        r2_list.append(d_row + (u_p1 - u_m1) * inv_s2)
        if include_4th:
            im2 = (idx - 2).clamp(0, n - 1)
            ip2 = (idx + 2).clamp(0, n - 1)
            u_m2 = torch.index_select(u_fd, ax + 1, im2)
            u_p2 = torch.index_select(u_fd, ax + 1, ip2)
            inv_12h = 1.0 / max(12.0 * h, 1e-12)
            r4_list.append(d_row + (u_m2 - 8.0 * u_m1 + 8.0 * u_p1 - u_p2) * inv_12h)
    det2 = _det_from_rows_nd(r2_list, s_D)
    if not include_4th:
        return det2
    det4 = _det_from_rows_nd(r4_list, s_D)
    return torch.minimum(det2, det4)


def _physical_fd4_det_only(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_inv: torch.Tensor,
) -> torch.Tensor:
    """Exact 4th-order 5-point central-difference physical Jacobian determinant matching ITK/ANTs
    ``CreateJacobianDeterminantImage`` (``ZeroFluxNeumann`` clamped indices ``i-2, i-1, i+1, i+2``)."""
    D_rev_T, s_D = _d_rev_t_and_scale(D_rev_inv)
    dim = u.shape[-1]
    r4_list: list[torch.Tensor] = []
    view_bcast_base = [1] * (u.ndim - 1) + [dim]
    for ax in range(dim):
        n = u.shape[ax + 1]
        h = float(steps_rev[ax])
        d_row = D_rev_T[ax].view(*view_bcast_base)
        if n <= 1:
            r4_list.append(d_row.expand_as(u))
            continue
        idx = torch.arange(n, device=u.device)
        im2 = (idx - 2).clamp(0, n - 1)
        im1 = (idx - 1).clamp(0, n - 1)
        ip1 = (idx + 1).clamp(0, n - 1)
        ip2 = (idx + 2).clamp(0, n - 1)
        inv_12h = 1.0 / max(12.0 * h, 1e-12)
        r4_list.append(
            d_row
            + (
                torch.index_select(u, ax + 1, im2)
                - 8.0 * torch.index_select(u, ax + 1, im1)
                + 8.0 * torch.index_select(u, ax + 1, ip1)
                - torch.index_select(u, ax + 1, ip2)
            )
            * inv_12h
        )
    return _det_from_rows_nd(r4_list, s_D)


def _physical_fd_det_and_grad(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_inv: torch.Tensor,
    jac_floor: float,
    voxel_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical central-difference ``det(I + (du/d_axis) @ D_rev_inv)``, quadratic hinge
    energy ``0.5 * sum(w * relu(jac_floor - det)^2)``, and exact discrete adjoint gradient w.r.t.
    ``u`` (shape ``(B, *spatial, dim)``, tensor component order)."""
    dim = u.shape[-1]
    grads = torch.gradient(u, spacing=steps_rev, dim=tuple(range(1, dim + 1)))
    G = torch.stack(grads, dim=-1)  # (B, *spatial, comp_i, axis_j)
    eye = torch.eye(dim, device=u.device, dtype=u.dtype)
    F_phys = G @ D_rev_inv + eye
    det, cof = _cofactor_transpose_nd(F_phys)
    viol = F.relu(float(jac_floor) - det)
    if voxel_weights is None:
        energy = 0.5 * (viol * viol).sum(dim=tuple(range(1, dim + 1)))  # (B,)
        w_viol = viol
    else:
        energy = 0.5 * (voxel_weights * viol * viol).sum(dim=tuple(range(1, dim + 1)))
        w_viol = voxel_weights * viol

    if not bool((w_viol > 0.0).any().item()):
        return det, energy, torch.zeros_like(u)

    dE_dF = -w_viol.unsqueeze(-1).unsqueeze(-1) * cof
    dE_dG = dE_dF @ D_rev_inv.transpose(-1, -2)

    grad_u = torch.zeros_like(u)
    for ax in range(dim):
        h = float(steps_rev[ax])
        n_ax = u.shape[ax + 1]
        p = dE_dG[..., ax]
        if n_ax < 2:
            continue
        adj = torch.zeros_like(p)
        sl_all = [slice(None)] * p.ndim

        def _sl(start, stop, _sl_all=sl_all, _ax=ax):
            s = list(_sl_all)
            s[_ax + 1] = slice(start, stop)
            return tuple(s)

        if n_ax == 2:
            adj[_sl(0, 1)] = (-p[_sl(0, 1)] - p[_sl(1, 2)]) / h
            adj[_sl(1, 2)] = (p[_sl(0, 1)] + p[_sl(1, 2)]) / h
        else:
            adj[_sl(0, 1)] = adj[_sl(0, 1)] - p[_sl(0, 1)] / h
            adj[_sl(1, 2)] = adj[_sl(1, 2)] + p[_sl(0, 1)] / h
            adj[_sl(2, n_ax)] = adj[_sl(2, n_ax)] + p[_sl(1, n_ax - 1)] / (2.0 * h)
            adj[_sl(0, n_ax - 2)] = adj[_sl(0, n_ax - 2)] - p[_sl(1, n_ax - 1)] / (2.0 * h)
            adj[_sl(n_ax - 1, n_ax)] = adj[_sl(n_ax - 1, n_ax)] + p[_sl(n_ax - 1, n_ax)] / h
            adj[_sl(n_ax - 2, n_ax - 1)] = adj[_sl(n_ax - 2, n_ax - 1)] - p[_sl(n_ax - 1, n_ax)] / h
        grad_u.add_(adj)

    return det, energy, grad_u


def _dilate_mask_nd(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of ``(B, *spatial)`` boolean mask with Chebyshev radius ``radius``
    without using 5-D ``F.pad`` (safe on MPS and CPU)."""
    if radius <= 0:
        return mask
    dim = mask.ndim - 1
    out = mask.clone()
    for _ in range(radius):
        cur = out
        nxt = cur.clone()
        for ax in range(1, dim + 1):
            n = cur.shape[ax]
            if n <= 1:
                continue
            sl_lo = [slice(None)] * cur.ndim
            sl_hi = [slice(None)] * cur.ndim
            sl_lo[ax] = slice(0, n - 1)
            sl_hi[ax] = slice(1, n)
            nxt[tuple(sl_lo)] = nxt[tuple(sl_lo)] | cur[tuple(sl_hi)]
            nxt[tuple(sl_hi)] = nxt[tuple(sl_hi)] | cur[tuple(sl_lo)]
        out = nxt
    return out


def _coords_to_flat_idx(coords: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    """Convert ``(N, 1 + dim)`` integer ``(b, i_0, ..., i_{dim-1})`` coordinates into 1-D flat
    indices into ``(B * prod(spatial),)``."""
    n_full = max(int(np.prod(spatial)), 1)
    fl = coords[:, 0] * n_full
    st = 1
    for d_ax in range(len(spatial) - 1, -1, -1):
        fl = fl + coords[:, d_ax + 1] * st
        st *= spatial[d_ax]
    return fl


def _build_sparse_fd_stencil(
    active_mask: torch.Tensor,
    steps_fd: list[float],
    dtype: torch.dtype,
):
    """Build exact sparse 2-voxel central-difference stencil tables around ``active_mask``
    (shape ``(B, *spatial)``).

    Any voxel outside ``sten_mask = _dilate_mask_nd(active_mask, 1)`` has zero neighbors inside
    ``active_mask``, so its central-difference Jacobian ``det(I + D_{2h} u)`` is strictly
    invariant to perturbations supported on ``active_mask``. Evaluating central differences on
    ``sten_mask`` requires field values only on ``pool_mask = _dilate_mask_nd(sten_mask, 1)``,
    reducing full-resolution 3-D ``(192x224x192 = 8.2M)`` per-iteration work to ``O(|active|)``
    even when defects are scattered across opposite hemispheres!
    """
    B = active_mask.shape[0]
    spatial = tuple(active_mask.shape[1:])
    dim = len(spatial)
    device = active_mask.device
    n_full = max(int(np.prod(spatial)), 1)

    sten_mask = _dilate_mask_nd(active_mask, 1)
    pool_mask = _dilate_mask_nd(sten_mask, 1)

    act_coords = torch.nonzero(active_mask, as_tuple=False)
    sten_coords = torch.nonzero(sten_mask, as_tuple=False)
    pool_coords = torch.nonzero(pool_mask, as_tuple=False)

    act_flat = _coords_to_flat_idx(act_coords, spatial)
    sten_flat = _coords_to_flat_idx(sten_coords, spatial)
    pool_flat = _coords_to_flat_idx(pool_coords, spatial)

    lut_pool = torch.full((B * n_full,), -1, dtype=torch.long, device=device)
    lut_pool[pool_flat] = torch.arange(pool_flat.shape[0], device=device)
    act_in_pool = lut_pool[act_flat]

    lo_in_pool_list = []
    hi_in_pool_list = []
    inv_span_list = []
    for ax in range(dim):
        n_ax = spatial[ax]
        h = float(steps_fd[ax])
        c_ax = sten_coords[:, ax + 1]
        if n_ax <= 1:
            lo_c = sten_coords
            hi_c = sten_coords
            inv_s = torch.zeros((sten_coords.shape[0],), device=device, dtype=dtype)
        else:
            lo_idx = (c_ax - 1).clamp(min=0)
            hi_idx = (c_ax + 1).clamp(max=n_ax - 1)
            lo_c = sten_coords.clone()
            hi_c = sten_coords.clone()
            lo_c[:, ax + 1] = lo_idx
            hi_c[:, ax + 1] = hi_idx
            span = (hi_idx - lo_idx).to(dtype) * h
            inv_s = torch.where(span > 0.0, 1.0 / span.clamp(min=1e-12), torch.zeros_like(span))
        lo_in_pool_list.append(lut_pool[ _coords_to_flat_idx(lo_c, spatial) ])
        hi_in_pool_list.append(lut_pool[ _coords_to_flat_idx(hi_c, spatial) ])
        inv_span_list.append(inv_s)

    del lut_pool
    lo_in_pool = torch.stack(lo_in_pool_list, dim=-1)    # (K_sten, dim)
    hi_in_pool = torch.stack(hi_in_pool_list, dim=-1)    # (K_sten, dim)
    inv_span = torch.stack(inv_span_list, dim=-1)        # (K_sten, dim)
    sten_batch = sten_coords[:, 0]
    return {
        "sten_mask": sten_mask,
        "pool_mask": pool_mask,
        "act_coords": act_coords,
        "act_flat": act_flat,
        "sten_flat": sten_flat,
        "pool_flat": pool_flat,
        "act_in_pool": act_in_pool,
        "lo_in_pool": lo_in_pool,
        "hi_in_pool": hi_in_pool,
        "inv_span": inv_span,
        "sten_batch": sten_batch,
    }


def _eval_sparse_fd_det(
    u_pool: torch.Tensor,
    stencil: dict,
    D_rev_inv: torch.Tensor,
    eye: torch.Tensor,
    use_physical: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact central-difference physical Jacobian determinant ``det(I + D_{2h} u)`` and
    cofactor matrix on the ``K_sten`` stencil coordinates from ``u_pool`` ``(K_pool, dim)``."""
    u_fd = u_pool if use_physical else torch.flip(u_pool, dims=[-1])
    u_hi = u_fd[stencil["hi_in_pool"]]  # (K_sten, dim_ax, dim_comp)
    u_lo = u_fd[stencil["lo_in_pool"]]  # (K_sten, dim_ax, dim_comp)
    # G[k, comp_i, axis_j] = (u_hi[k, axis_j, comp_i] - u_lo[k, axis_j, comp_i]) * inv_span[k, axis_j]
    diff = (u_hi - u_lo) * stencil["inv_span"].unsqueeze(-1)
    G = diff.transpose(-1, -2)
    F_phys = G @ D_rev_inv + eye
    return _cofactor_transpose_nd(F_phys)


def _build_sparse_dual_fd_stencil(
    active_mask: torch.Tensor,
    steps_fd: list[float],
    dtype: torch.dtype,
    include_4th: bool = True,
    lut_buf: Optional[torch.Tensor] = None,
) -> dict:
    """Build exact sparse dual-order (2nd-order ``D_{2h}`` + 4th-order ``D^{(4)}_h``) finite-difference
    stencil tables around ``active_mask`` ``(B, *spatial)``.

    With ``include_4th=True`` (physical-space mode), ``sten_r = 2`` captures every voxel whose
    3-point (``syntx.signed_jacobian_determinant``) or 5-point (``ants.create_jacobian_determinant_image``)
    central-difference Jacobian depends on ``active_mask``. Outside ``sten_mask``, both determinants
    are strictly invariant to updates supported on ``active_mask``.
    """
    B = active_mask.shape[0]
    spatial = tuple(active_mask.shape[1:])
    dim = len(spatial)
    device = active_mask.device
    n_full = max(int(np.prod(spatial)), 1)

    sten_r = 2 if include_4th else 1
    sten_mask = _dilate_mask_nd(active_mask, sten_r)
    pool_mask = _dilate_mask_nd(sten_mask, sten_r)

    act_coords = torch.nonzero(active_mask, as_tuple=False)
    sten_coords = torch.nonzero(sten_mask, as_tuple=False)
    pool_coords = torch.nonzero(pool_mask, as_tuple=False)

    act_flat = _coords_to_flat_idx(act_coords, spatial)
    sten_flat = _coords_to_flat_idx(sten_coords, spatial)
    pool_flat = _coords_to_flat_idx(pool_coords, spatial)

    own_lut = lut_buf is None or lut_buf.numel() != B * n_full
    lut_pool = (
        torch.full((B * n_full,), -1, dtype=torch.long, device=device)
        if own_lut
        else lut_buf
    )
    lut_pool[pool_flat] = torch.arange(pool_flat.shape[0], device=device)
    act_in_pool = lut_pool[act_flat]
    sten_in_pool = lut_pool[sten_flat]

    k_act = int(act_flat.shape[0])
    pool_to_act = torch.full(
        (pool_flat.shape[0],), k_act, dtype=torch.long, device=device
    )
    if k_act > 0:
        pool_to_act[act_in_pool] = torch.arange(k_act, device=device)
    sten_in_act = pool_to_act[sten_in_pool]

    im1_list, ip1_list, inv_span2_list = [], [], []
    im1_act_list, ip1_act_list = [], []
    inv_span_fwd_list, inv_span_bwd_list = [], []
    im2_list, ip2_list, inv_12h_list = [], [], []
    im2_act_list, ip2_act_list = [], []
    for ax in range(dim):
        n_ax = spatial[ax]
        h = float(steps_fd[ax])
        c_ax = sten_coords[:, ax + 1]
        if n_ax <= 1:
            z_idx = lut_pool[_coords_to_flat_idx(sten_coords, spatial)]
            z_act = pool_to_act[z_idx]
            im1_list.append(z_idx)
            ip1_list.append(z_idx)
            im1_act_list.append(z_act)
            ip1_act_list.append(z_act)
            z_span = torch.zeros((sten_coords.shape[0],), device=device, dtype=dtype)
            inv_span2_list.append(z_span)
            inv_span_fwd_list.append(z_span)
            inv_span_bwd_list.append(z_span)
            if include_4th:
                im2_list.append(z_idx)
                ip2_list.append(z_idx)
                im2_act_list.append(z_act)
                ip2_act_list.append(z_act)
                inv_12h_list.append(z_span)
        else:
            lo1 = (c_ax - 1).clamp(min=0)
            hi1 = (c_ax + 1).clamp(max=n_ax - 1)
            c_lo1 = sten_coords.clone()
            c_hi1 = sten_coords.clone()
            c_lo1[:, ax + 1] = lo1
            c_hi1[:, ax + 1] = hi1
            span2 = (hi1 - lo1).to(dtype) * h
            span_f = (hi1 - c_ax).to(dtype) * h
            span_b = (c_ax - lo1).to(dtype) * h
            inv_s2 = torch.where(span2 > 0.0, 1.0 / span2.clamp(min=1e-12), torch.zeros_like(span2))
            inv_sf = torch.where(span_f > 0.0, 1.0 / span_f.clamp(min=1e-12), torch.zeros_like(span_f))
            inv_sb = torch.where(span_b > 0.0, 1.0 / span_b.clamp(min=1e-12), torch.zeros_like(span_b))
            im1_p = lut_pool[_coords_to_flat_idx(c_lo1, spatial)]
            ip1_p = lut_pool[_coords_to_flat_idx(c_hi1, spatial)]
            im1_list.append(im1_p)
            ip1_list.append(ip1_p)
            im1_act_list.append(pool_to_act[im1_p])
            ip1_act_list.append(pool_to_act[ip1_p])
            inv_span2_list.append(inv_s2)
            inv_span_fwd_list.append(inv_sf)
            inv_span_bwd_list.append(inv_sb)
            if include_4th:
                lo2 = (c_ax - 2).clamp(min=0)
                hi2 = (c_ax + 2).clamp(max=n_ax - 1)
                c_lo2 = sten_coords.clone()
                c_hi2 = sten_coords.clone()
                c_lo2[:, ax + 1] = lo2
                c_hi2[:, ax + 1] = hi2
                im2_p = lut_pool[_coords_to_flat_idx(c_lo2, spatial)]
                ip2_p = lut_pool[_coords_to_flat_idx(c_hi2, spatial)]
                im2_list.append(im2_p)
                ip2_list.append(ip2_p)
                im2_act_list.append(pool_to_act[im2_p])
                ip2_act_list.append(pool_to_act[ip2_p])
                inv_12h_list.append(
                    torch.full(
                        (sten_coords.shape[0],),
                        1.0 / max(12.0 * h, 1e-12),
                        device=device,
                        dtype=dtype,
                    )
                )

    if own_lut:
        del lut_pool
    else:
        lut_pool[pool_flat] = -1
    out = {
        "include_4th": include_4th,
        "sten_mask": sten_mask,
        "pool_mask": pool_mask,
        "act_coords": act_coords,
        "act_flat": act_flat,
        "sten_flat": sten_flat,
        "pool_flat": pool_flat,
        "act_in_pool": act_in_pool,
        "sten_in_pool": sten_in_pool,
        "sten_in_act": sten_in_act,
        "im1": torch.stack(im1_list, dim=-1),
        "ip1": torch.stack(ip1_list, dim=-1),
        "inv_span2": torch.stack(inv_span2_list, dim=-1),
        "inv_span_fwd": torch.stack(inv_span_fwd_list, dim=-1),
        "inv_span_bwd": torch.stack(inv_span_bwd_list, dim=-1),
        "im1_ax": im1_list,
        "ip1_ax": ip1_list,
        "im1_act_ax": im1_act_list,
        "ip1_act_ax": ip1_act_list,
        "inv_s2_ax": [s.unsqueeze(-1) for s in inv_span2_list],
        "inv_sf_ax": [s.unsqueeze(-1) for s in inv_span_fwd_list],
        "inv_sb_ax": [s.unsqueeze(-1) for s in inv_span_bwd_list],
        "sten_batch": sten_coords[:, 0],
    }
    if include_4th:
        out["im2"] = torch.stack(im2_list, dim=-1)
        out["ip2"] = torch.stack(ip2_list, dim=-1)
        out["inv_12h"] = torch.stack(inv_12h_list, dim=-1)
        out["im2_ax"] = im2_list
        out["ip2_ax"] = ip2_list
        out["im2_act_ax"] = im2_act_list
        out["ip2_act_ax"] = ip2_act_list
        out["inv_12h_s"] = [
            0.0 if spatial[ax] <= 1 else (1.0 / max(12.0 * float(steps_fd[ax]), 1e-12))
            for ax in range(dim)
        ]
    return out


def _precompute_sparse_row_bases(
    u_pool0: torch.Tensor,
    stencil: dict,
    D_rev_T: torch.Tensor,
    use_physical: bool,
    include_cell: bool = False,
) -> dict:
    """Precompute constant baseline physical deformation rows ``M0 = D_rev^T + diff(u_0)`` as
    contiguous ``(K_sten, dim)`` tensors once per pass so inner Adam steps never recompute ``u_0``
    finite differences, ``.transpose``, or ``@ D_rev_inv``."""
    u_fd = u_pool0 if use_physical else torch.flip(u_pool0, dims=[-1])
    dim = u_fd.shape[-1]
    inc4 = stencil["include_4th"]
    m2, m4, mp, mm = [], [], [], []
    uc = u_fd[stencil["sten_in_pool"]] if include_cell else None
    for ax in range(dim):
        d_row = D_rev_T[ax].unsqueeze(0)
        hi1 = u_fd[stencil["ip1_ax"][ax]]
        lo1 = u_fd[stencil["im1_ax"][ax]]
        m2.append((d_row + (hi1 - lo1) * stencil["inv_s2_ax"][ax]).contiguous())
        if inc4:
            hi2 = u_fd[stencil["ip2_ax"][ax]]
            lo2 = u_fd[stencil["im2_ax"][ax]]
            m4.append(
                (
                    d_row
                    + (lo2 - 8.0 * lo1 + 8.0 * hi1 - hi2) * stencil["inv_12h_s"][ax]
                ).contiguous()
            )
        if include_cell:
            mp.append((d_row + (hi1 - uc) * stencil["inv_sf_ax"][ax]).contiguous())
            mm.append((d_row + (uc - lo1) * stencil["inv_sb_ax"][ax]).contiguous())
    return {"m2": m2, "m4": m4, "mp": mp, "mm": mm}


def _eval_sparse_row_step(
    delta: torch.Tensor,
    stencil: dict,
    row_bases: dict,
    s_D: float,
    use_physical: bool,
    include_cell: bool = False,
):
    """Single-gather fused evaluation of ``(det2, det4[, det_p, det_m])`` AND Dirichlet+Laplacian
    smoothness ``sm`` from perturbation ``delta`` ``(K_act, dim)`` using direct sentinel-row
    active indexing (zero ``z_pool.index_add`` atomic scatter on forward/backward)."""
    d_fd = delta if use_physical else torch.flip(delta, dims=[-1])
    dp = torch.cat([d_fd, d_fd.new_zeros((1, d_fd.shape[-1]))], dim=0)
    dc = dp[stencil["sten_in_act"]]
    dim = d_fd.shape[-1]
    inc4 = stencil["include_4th"]
    r2, r4, rp, rm = [], [], [], []
    sm = d_fd.new_zeros(())
    for ax in range(dim):
        hi1 = dp[stencil["ip1_act_ax"][ax]]
        lo1 = dp[stencil["im1_act_ax"][ax]]
        fd = hi1 - dc
        bd = dc - lo1
        sm = sm + (fd.square() + bd.square() + 0.5 * (fd - bd).square()).sum()
        r2.append(row_bases["m2"][ax] + (hi1 - lo1) * stencil["inv_s2_ax"][ax])
        if inc4:
            hi2 = dp[stencil["ip2_act_ax"][ax]]
            lo2 = dp[stencil["im2_act_ax"][ax]]
            r4.append(
                row_bases["m4"][ax]
                + (lo2 - 8.0 * lo1 + 8.0 * hi1 - hi2) * stencil["inv_12h_s"][ax]
            )
        if include_cell:
            rp.append(row_bases["mp"][ax] + fd * stencil["inv_sf_ax"][ax])
            rm.append(row_bases["mm"][ax] + bd * stencil["inv_sb_ax"][ax])
    det2 = _det_from_rows_nd(r2, s_D)
    det4 = _det_from_rows_nd(r4, s_D) if inc4 else det2
    if not include_cell:
        return det2, det4, sm
    det_p = _det_from_rows_nd(rp, s_D)
    det_m = _det_from_rows_nd(rm, s_D)
    return det2, det4, det_p, det_m, sm


def _eval_sparse_dual_fd_det(
    u_pool: torch.Tensor,
    stencil: dict,
    D_rev_inv: torch.Tensor,
    eye: torch.Tensor,
    use_physical: bool,
    include_cell: bool = False,
):
    """Evaluate 2nd-order ``det2`` and 4th-order ``det4`` physical Jacobian determinants (and
    optionally 1-voxel forward/backward cell secant determinants ``det_p``, ``det_m``) on
    ``K_sten`` stencil coordinates from ``u_pool`` ``(K_pool, dim)``."""
    D_rev_T, s_D = _d_rev_t_and_scale(D_rev_inv)
    bases = _precompute_sparse_row_bases(
        u_pool, stencil, D_rev_T, use_physical, include_cell=include_cell
    )
    det2 = _det_from_rows_nd(bases["m2"], s_D)
    det4 = _det_from_rows_nd(bases["m4"], s_D) if stencil["include_4th"] else det2
    if not include_cell:
        return det2, det4
    return (
        det2,
        det4,
        _det_from_rows_nd(bases["mp"], s_D),
        _det_from_rows_nd(bases["mm"], s_D),
    )


def _active_lse(v: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
    """Active-violator ``logsumexp`` smooth supremum norm without zero-padding dilution or
    device-to-host ``.item()`` / dynamic-shape boolean-mask syncs (static-shape on MPS/CPU)."""
    if v.numel() == 0:
        return v.new_zeros(())
    vmax = v.amax()
    pos = (v > 0.0).to(v.dtype)
    s = (pos * torch.exp(float(beta) * (v - vmax))).sum()
    lse_pos = vmax + torch.log(s.clamp_min(1.0)) / float(beta)
    return torch.where(pos.sum() > 0.0, lse_pos, v.new_zeros(()))


def _active_lse_batched(V: torch.Tensor, beta: float = 25.0, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Vectorized batched active-violator ``logsumexp`` over ``(M_terms, K)`` tensor ``V >= 0``
    in a single fused GPU pass with zero host syncs."""
    if V.numel() == 0:
        return V.new_zeros(())
    vmax = V.amax(dim=-1, keepdim=True)
    pos = (V > 0.0).to(V.dtype)
    s = (pos * torch.exp(float(beta) * (V - vmax))).sum(dim=-1)
    lse = vmax.squeeze(-1) + torch.log(s.clamp_min(1.0)) / float(beta)
    lse = torch.where(pos.sum(dim=-1) > 0.0, lse, torch.zeros_like(lse))
    return (lse * weights).sum() if weights is not None else lse.sum()


def _gather_multilinear_sparse(
    u0_flat: torch.Tensor,
    lut_act: torch.Tensor,
    delta_act: torch.Tensor,
    batch_idx: torch.Tensor,
    vox_coords: torch.Tensor,
    spatial: tuple[int, ...],
) -> torch.Tensor:
    """Differentiable sparse multilinear interpolation (``padding_mode='border'``,
    ``align_corners=True``) of ``u = u0 + scatter(delta_act)`` at continuous spatial voxel
    coordinates ``vox_coords`` ``(M, dim)`` for batch indices ``batch_idx`` ``(M,)``.

    Vectorized single-gather over all ``2**dim`` corners ``(M, 2**dim, dim)`` with sentinel zero-row
    lookup (zero Python corner loop, zero 5-D ``grid_sample``, fast and deterministic on MPS/CPU).
    """
    dtype = u0_flat.dtype
    if vox_coords.shape[0] == 0:
        return vox_coords.new_zeros((0, u0_flat.shape[-1]))
    c_idx, w_c = _static_multilinear_corners(batch_idx, vox_coords, spatial, dtype)
    val = u0_flat[c_idx]
    k_act = int(delta_act.shape[0])
    if k_act > 0 and lut_act.numel() > 0:
        d_pad = torch.cat([delta_act, delta_act.new_zeros((1, delta_act.shape[-1]))], dim=0)
        a_idx = lut_act[c_idx]
        a_safe = torch.where(a_idx >= 0, a_idx, k_act)
        val = val + d_pad[a_safe]
    wv = w_c.unsqueeze(-1) * val
    out = wv[:, 0]
    for ci in range(1, wv.shape[1]):
        out = out + wv[:, ci]
    return out


def _pointwise_lm_refine_subset(
    target_field: torch.Tensor,
    init_inv_field: torch.Tensor,
    subset_mask: torch.Tensor,
    X_base: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    right_mat: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    steps: int,
    tol_vox: float,
    max_step_unit: float,
) -> torch.Tensor:
    """Run damped Levenberg-Marquardt point-wise inversion ``v(x) + target_field(x + v(x)) = 0``
    strictly on ``subset_mask`` ``(B, *spatial)`` in ``O(|subset| * steps)`` time, leaving all
    voxels outside ``subset_mask`` bitwise untouched."""
    if steps <= 0 or not bool(subset_mask.any().item()):
        return init_inv_field
    B = target_field.shape[0]
    dim = target_field.shape[-1]
    device = target_field.device
    dtype = target_field.dtype
    I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
    tol_sq = float(tol_vox) * float(tol_vox)
    out = init_inv_field.clone()

    for b in range(B):
        idx = torch.nonzero(subset_mask[b], as_tuple=False)
        if idx.shape[0] == 0:
            continue
        idx_t = tuple(idx[:, d] for d in range(dim))
        u_grid = target_field[b]
        x_a = X_base[0 if X_base.shape[0] == 1 else b][idx_t]
        v_a = out[b][idx_t].clone()
        lam = torch.full((idx.shape[0],), 1e-3, device=device, dtype=dtype)

        for _it in range(int(steps)):
            y_vox = (x_a + v_a) @ A_to_vox + b_to_vox
            u_a, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
            r_a = v_a + u_a
            r_vox = r_a @ err_to_vox_mat
            r_sq = (r_vox * r_vox).sum(dim=-1)
            if (_it & 7) == 7 and float(r_sq.amax().item()) <= tol_sq:
                break
            J = I_d + grad_u_a
            JT = J.transpose(-1, -2)
            JTJ = JT @ J
            rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
            dv = _solve_spd_lm_torch(JTJ, rhs, lam)
            dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
            dv = torch.where(
                dv_norm > max_step_unit,
                dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                dv,
            )
            v_trial = v_a + dv
            y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
            u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
            r_t_vox = (v_trial + u_t) @ err_to_vox_mat
            r_sq_t = (r_t_vox * r_t_vox).sum(dim=-1)
            imp = r_sq_t < r_sq
            v_a = torch.where(imp.unsqueeze(-1), v_trial, v_a)
            lam = torch.where(
                imp,
                (lam * 0.25).clamp(min=1e-6),
                (lam * 8.0).clamp(max=1e8),
            )
        out[b][idx_t] = v_a
    return out


def _splat_multilinear_grad_inplace(
    grad_grid: torch.Tensor,
    vox_coords: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Adjoint of multilinear interpolation (``padding_mode='border', align_corners=True``):
    scatter-adds ``weights`` ``(N, dim)`` at continuous ``vox_coords`` ``(N, dim)`` into
    ``grad_grid`` ``(*spatial, dim)``."""
    dim = grad_grid.shape[-1]
    dtype = grad_grid.dtype
    if vox_coords.shape[0] == 0:
        return

    if dim == 2:
        H, W, _ = grad_grid.shape
        vy = vox_coords[:, 0].clamp(0.0, float(max(H - 1, 0)))
        vx = vox_coords[:, 1].clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = (vy - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx - ix0.to(dtype)).unsqueeze(-1)
        omy, omx = 1.0 - ty, 1.0 - tx

        flat = grad_grid.view(-1, 2)
        flat.index_add_(0, iy0 * W + ix0, weights * (omy * omx))
        flat.index_add_(0, iy0 * W + ix1, weights * (omy * tx))
        flat.index_add_(0, iy1 * W + ix0, weights * (ty * omx))
        flat.index_add_(0, iy1 * W + ix1, weights * (ty * tx))
        return

    Z, Y, X, _ = grad_grid.shape
    vz = vox_coords[:, 0].clamp(0.0, float(max(Z - 1, 0)))
    vy = vox_coords[:, 1].clamp(0.0, float(max(Y - 1, 0)))
    vx = vox_coords[:, 2].clamp(0.0, float(max(X - 1, 0)))
    iz0 = torch.floor(vz).long().clamp(0, max(Z - 2, 0))
    iy0 = torch.floor(vy).long().clamp(0, max(Y - 2, 0))
    ix0 = torch.floor(vx).long().clamp(0, max(X - 2, 0))
    iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
    iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
    ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
    tz = (vz - iz0.to(dtype)).unsqueeze(-1)
    ty = (vy - iy0.to(dtype)).unsqueeze(-1)
    tx = (vx - ix0.to(dtype)).unsqueeze(-1)
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx

    flat = grad_grid.view(-1, 3)
    sz, sy = Y * X, X
    flat.index_add_(0, iz0 * sz + iy0 * sy + ix0, weights * (omz * omy * omx))
    flat.index_add_(0, iz0 * sz + iy0 * sy + ix1, weights * (omz * omy * tx))
    flat.index_add_(0, iz0 * sz + iy1 * sy + ix0, weights * (omz * ty * omx))
    flat.index_add_(0, iz0 * sz + iy1 * sy + ix1, weights * (omz * ty * tx))
    flat.index_add_(0, iz1 * sz + iy0 * sy + ix0, weights * (tz * omy * omx))
    flat.index_add_(0, iz1 * sz + iy0 * sy + ix1, weights * (tz * omy * tx))
    flat.index_add_(0, iz1 * sz + iy1 * sy + ix0, weights * (tz * ty * omx))
    flat.index_add_(0, iz1 * sz + iy1 * sy + ix1, weights * (tz * ty * tx))


def _smooth_max_nd(
    values: torch.Tensor,
    spatial_dims: tuple[int, ...],
    beta: float = 20.0,
) -> torch.Tensor:
    """Differentiable log-sum-exp smooth ``L_infinity`` approximation
    ``(1 / beta) * logmeanexp(beta * values)`` over ``spatial_dims``.

    Unlike ``torch.mean(values**2)`` (which dilutes a single folded or inconsistent voxel by
    ``1 / N_voxels ~ 1e-6`` on a 3-D volume), ``logmeanexp`` concentrates ``softmax(beta * v)``
    gradient weight onto the worst-case ``L_infinity`` violations while converging smoothly to
    ``amax(values)`` as ``beta -> inf`` and remaining identically ``0`` when ``values == 0``
    everywhere.
    """
    if beta <= 0.0:
        return values.amax(dim=spatial_dims)
    flat = values.reshape(values.shape[0], -1)
    vmax = flat.amax(dim=1, keepdim=True)
    # Shifted logsumexp: lse_0(v) = (1/beta) * (log(sum_i exp(beta * v_i)) - log(N)).
    # Notice lse_0(0) == 0, lse_0(v) >= 0 for ALL v >= 0 (since exp(beta*v_i) >= 1 => sum >= N),
    # and grad_v lse_0(v) == softmax(beta * v) > 0 for ANY single violating voxel regardless of vmax.
    n_elem = float(max(flat.shape[1], 1))
    shifted = torch.exp(float(beta) * (flat - vmax))
    lse0 = vmax.squeeze(1) + (torch.log(shifted.sum(dim=1)) - float(np.log(n_elem))) / float(beta)
    return torch.where(
        vmax.squeeze(1) > 0.0,
        F.relu(lse0),
        torch.zeros_like(vmax.squeeze(1)),
    )


def digital_diffeomorphism_loss(
    warp_field: torch.Tensor,
    warp_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    jac_floor: float = 0.05,
    ice_tol: float = 0.0,
    ice_weight: float = 1.0,
    linf_beta: Optional[float] = 25.0,
    mean_weight: float = 0.1,
    stationary_boundary: bool = True,
    is_physical: Optional[bool] = None,
) -> torch.Tensor:
    """Differentiable ``L_infinity`` **digital diffeomorphism** loss following the Anderson
    worst-case (supremum-norm) inversion philosophy.

    Measures worst-case discrete 2-mm central-difference Jacobian folding violations
    ``v_jac(x) = relu(jac_floor - det(I + D_{2h} u(x)))`` (and, when ``warp_inv`` is supplied,
    both ``v_jac(u_inv)`` and bidirectional voxel-chord inverse-consistency violations
    ``v_ice(x) = relu(||r(x)||_vox - ice_tol)``) via a smooth ``L_infinity`` norm:

    - ``linf_beta > 0`` (default ``25.0``): smooth ``L_infinity`` ``logmeanexp`` norm
      ``(1/beta) log((1/|Omega|) sum_x exp(beta * v(x))) + mean_weight * mean(v(x)^2)``, which
      concentrates ``softmax(beta * v)`` gradient directly onto the worst offending voxels
      instead of diluting single-voxel folds by ``1 / N_voxels``.
    - ``linf_beta is None`` or ``<= 0``: exact ``L_infinity`` ``amax_x v(x)`` subgradient
      plus ``mean_weight * mean(v(x)^2)``.

    Strictly ``0.0`` when every interior voxel satisfies ``det >= jac_floor`` and
    ``ICE <= ice_tol``. Respects full ``spacing``, ``origin``, and oblique ``direction``
    cosines (as well as normalized ``[-1, 1]`` grids when ``is_physical=False``).
    """
    def _to_cl(w: torch.Tensor) -> torch.Tensor:
        cf = w.dim() >= 3 and w.shape[1] in (2, 3) and w.shape[-1] not in (2, 3)
        out = w.permute((0,) + tuple(range(2, w.dim())) + (1,)) if cf else w
        if out.dim() == out.shape[-1] + 1:
            out = out.unsqueeze(0)
        return out

    u_f = _to_cl(warp_field)
    u_i = _to_cl(warp_inv) if warp_inv is not None else None
    dim = u_f.shape[-1]
    spatial = tuple(u_f.shape[1:-1])
    device = u_f.device
    dtype = u_f.dtype
    spatial_dims = tuple(range(1, dim + 1))

    if is_physical is None:
        is_physical = getattr(
            warp_field, "is_physical", spacing is not None or X_phys is not None
        )

    bmask = get_boundary_mask(spatial, device, dtype)
    dom = (
        (bmask[0, ..., 0] > 0.5).to(dtype).unsqueeze(0)
        if stationary_boundary
        else torch.ones((1, *spatial), device=device, dtype=dtype)
    )

    if is_physical:
        sp = [1.0] * dim if spacing is None else [float(x) for x in spacing]
        org = [0.0] * dim if origin is None else [float(x) for x in origin]
        steps_rev = list(reversed(sp))
        origin_rev = list(reversed(org))
        dir_arr = (
            np.eye(dim)
            if direction is None
            else np.asarray(
                direction.detach().cpu().numpy()
                if isinstance(direction, torch.Tensor)
                else direction,
                dtype=np.float64,
            ).reshape(dim, dim)
        )
        dir_rev = dir_arr[::-1, ::-1].copy()
        D_inv_np = np.linalg.inv(dir_rev)
        D_rev_inv = torch.as_tensor(D_inv_np, device=device, dtype=dtype)
        comps_f = u_f
        comps_i = u_i
    else:
        steps_rev = [2.0 / max(n - 1, 1) for n in spatial]
        D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
        comps_f = torch.flip(u_f, dims=[-1])
        comps_i = torch.flip(u_i, dims=[-1]) if u_i is not None else None

    eye = torch.eye(dim, device=device, dtype=dtype)

    def _jac_viol(comps: torch.Tensor) -> torch.Tensor:
        grads = torch.gradient(comps, spacing=steps_rev, dim=spatial_dims)
        G = torch.stack(grads, dim=-1)
        det, _ = _cofactor_transpose_nd(G @ D_rev_inv + eye)
        return F.relu(float(jac_floor) - det) * dom

    def _reduce_linf(viol: torch.Tensor) -> torch.Tensor:
        if linf_beta is None or float(linf_beta) <= 0.0:
            linf_term = viol.amax(dim=spatial_dims)
        else:
            linf_term = _smooth_max_nd(viol, spatial_dims, beta=float(linf_beta))
        if float(mean_weight) > 0.0:
            linf_term = linf_term + float(mean_weight) * (viol * viol).mean(dim=spatial_dims)
        return linf_term.mean()

    loss = _reduce_linf(_jac_viol(comps_f))

    if u_i is not None:
        loss = loss + _reduce_linf(_jac_viol(comps_i))
        if float(ice_weight) > 0.0:
            if is_physical:
                X_b = (
                    X_phys
                    if X_phys is not None
                    else get_physical_grid_torch(
                        spatial, sp, org, dir_arr, device=device, dtype=dtype
                    )
                )
                spacing_t = torch.tensor(steps_rev, device=device, dtype=dtype)
                shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
                origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
                direction_t = torch.tensor(dir_rev, device=device, dtype=dtype)
                M_norm, b_norm = get_physical_to_normalized_affine(
                    shape_t, spacing_t, origin_t, direction_t
                )
                sp_rev_np = np.asarray(steps_rev, dtype=np.float64)
                err_to_vox_mat = torch.as_tensor(
                    (np.diag(1.0 / sp_rev_np) @ D_inv_np).T, device=device, dtype=dtype
                )
                y_norm = physical_to_normalized_fast(X_b + u_f, M_norm, b_norm)
                x_norm = physical_to_normalized_fast(X_b + u_i, M_norm, b_norm)
            else:
                grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
                mesh = torch.meshgrid(*grids, indexing="ij")
                X_b = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
                err_to_vox_mat = torch.diag(
                    torch.tensor(
                        [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
                        device=device,
                        dtype=dtype,
                    )
                )
                y_norm = X_b + u_f
                x_norm = X_b + u_i

            in_y = (((y_norm >= -1.0) & (y_norm <= 1.0)).all(dim=-1)).to(dtype) * dom
            in_x = (((x_norm >= -1.0) & (x_norm <= 1.0)).all(dim=-1)).to(dtype) * dom
            r_fi = (u_f + sample_field_cf(torch.movedim(u_i, -1, 1), y_norm)) @ err_to_vox_mat
            r_if = (u_i + sample_field_cf(torch.movedim(u_f, -1, 1), x_norm)) @ err_to_vox_mat
            # Smooth norm sqrt(||r||^2 + eps) for clean gradient at 0
            n_fi = torch.sqrt((r_fi * r_fi).sum(dim=-1) + 1e-12) - 1e-6
            n_if = torch.sqrt((r_if * r_if).sum(dim=-1) + 1e-12) - 1e-6
            v_fi = F.relu(n_fi - float(ice_tol)) * in_y
            v_if = F.relu(n_if - float(ice_tol)) * in_x
            loss = loss + float(ice_weight) * 0.5 * (_reduce_linf(v_fi) + _reduce_linf(v_if))

    return loss


def digital_diffeomorphism_penalty(
    warp_field: torch.Tensor,
    warp_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    jac_floor: float = 0.05,
    ice_tol: float = 0.0,
    linf_beta: Optional[float] = 25.0,
    mean_weight: float = 0.1,
    stationary_boundary: bool = True,
    is_physical: Optional[bool] = None,
) -> torch.Tensor:
    """Alias for ``digital_diffeomorphism_loss`` (smooth ``L_infinity`` digital diffeomorphism
    barrier following the Anderson worst-case inversion philosophy)."""
    return digital_diffeomorphism_loss(
        warp_field,
        warp_inv,
        spacing=spacing,
        origin=origin,
        direction=direction,
        jac_floor=jac_floor,
        ice_tol=ice_tol,
        linf_beta=linf_beta,
        mean_weight=mean_weight,
        stationary_boundary=stationary_boundary,
        is_physical=is_physical,
    )


def _unfold_field_adaptive_smooth(
    u_orig: torch.Tensor,
    det_min_init: torch.Tensor,
    dom_b: torch.Tensor,
    boundary_mask: torch.Tensor,
    stationary_boundary: bool,
    steps_fd: list[float],
    D_rev_inv: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    use_physical: bool,
    jac_floor: float,
    jac_weight: float,
    anchor_weight: float,
    steps: int,
    step_size: float,
    dilation_radius: int,
    mean_sp: float,
    max_det_cap: Optional[float] = 14.0,
    partner_field: Optional[torch.Tensor] = None,
    X_base: Optional[torch.Tensor] = None,
    A_to_vox: Optional[torch.Tensor] = None,
    b_to_vox: Optional[torch.Tensor] = None,
    ice_tol: float = 0.30,
    ice_strict_tol: Optional[float] = None,
    extra_seed_mask: Optional[torch.Tensor] = None,
    lut_buf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, float, int, torch.Tensor]:
    """Principled sparse shrinking-shell diffeomorphic projection solving:

    1. **Forward cell + dual-stencil unfolding** (``partner_field is None``): enforces
       ``det(I + D_{2h} u) >= jac_floor``, ``det(I + D^{(4)}_h u) >= jac_floor``, AND 1-voxel
       forward/backward cell secant bijectivity ``det(I + D^+_h u) >= eps_cell``,
       ``det(I + D^-_h u) >= eps_cell`` with constant (non-decaying) Dirichlet + Laplacian
       smoothness on ``delta u``. Cell secant bijectivity eliminates the antisymmetric central-FD
       ``(-1)^i`` checkerboard nullspace from first principles and guarantees every multilinear
       voxel cell is genuinely invertible.
    2. **Joint diffeomorphic inverse projection** (``partner_field is not None``): minimizes
       ``||u(x) + partner(x + u(x))||_vox^2 + lambda_reg ||nabla delta u||^2`` subject to
       ``min(det_{2h}, det_{4h}) >= jac_floor``, terminating across passes only when BOTH
       ``min_det >= jac_floor`` AND ``max_v_ice <= 0``.
    """
    device = u_orig.device
    dtype = u_orig.dtype
    B = u_orig.shape[0]
    dim = u_orig.shape[-1]
    spatial = tuple(u_orig.shape[1:-1])
    include_4th = bool(use_physical)
    det_min_cur = det_min_init.clone()
    cur_global_min = float(det_min_cur.amin().item())

    use_partner = (
        partner_field is not None
        and X_base is not None
        and A_to_vox is not None
        and b_to_vox is not None
    )
    has_extra_seeds = extra_seed_mask is not None and bool(extra_seed_mask.any().item())
    ever_active = torch.zeros_like(dom_b)
    if steps <= 0 or (cur_global_min >= float(jac_floor) and not has_extra_seeds):
        return u_orig, det_min_cur, cur_global_min, 0, ever_active

    u_cur = u_orig.clone()
    if stationary_boundary:
        u_cur.mul_(boundary_mask)
    orig_flat = u_orig.reshape(-1, dim)
    eye = torch.eye(dim, device=device, dtype=dtype)
    D_rev_T, s_D = _d_rev_t_and_scale(D_rev_inv)

    if use_partner:
        X_flat = X_base.expand(B, *spatial, dim).reshape(-1, dim)
        partner_flat = partner_field.reshape(-1, dim)
        vox_hi_t = torch.tensor(
            [float(max(s - 1, 0)) for s in spatial], device=device, dtype=dtype
        )
        empty_lut = torch.empty((0,), dtype=torch.long, device=device)
        empty_delta = u_orig.new_zeros((0, dim))

    max_passes = 10 if int(steps) >= 8 else 1
    lr_base = 0.075 * mean_sp * max(min(float(step_size) / 0.30, 2.0), 0.25)
    target_det = float(jac_floor) + 0.03
    cell_floor = min(float(jac_floor), 0.02) + 0.015
    steps_taken = 0
    accepted_any = False

    cur_ice_defects = (
        extra_seed_mask.clone() if (use_partner and has_extra_seeds) else torch.zeros_like(dom_b)
    )

    for p_idx in range(max_passes):
        rem_defects = (det_min_cur < float(jac_floor)) | cur_ice_defects
        if not bool(rem_defects.any().item()):
            break
        rad_p = int(dilation_radius) + min(p_idx // 2, 1)
        act_mask = _dilate_mask_nd(rem_defects, rad_p) & dom_b
        if not bool(act_mask.any().item()):
            break

        stencil = _build_sparse_dual_fd_stencil(
            act_mask, steps_fd, dtype, include_4th=include_4th, lut_buf=lut_buf
        )
        sten_flat = stencil["sten_flat"]
        sten_prev = det_min_cur.view(-1)[sten_flat].clone()
        init_min_sten = float(sten_prev.amin().item()) if sten_prev.numel() > 0 else float("inf")
        if sten_flat.shape[0] < det_min_cur.numel():
            det_min_cur.view(-1)[sten_flat] = float("inf")
            ext_min = float(det_min_cur.amin().item())
            det_min_cur.view(-1)[sten_flat] = sten_prev
        else:
            ext_min = float("inf")

        u_base_pool = u_cur.reshape(-1, dim)[stencil["pool_flat"]].clone()
        row_bases = _precompute_sparse_row_bases(
            u_base_pool, stencil, D_rev_T, use_physical, include_cell=not use_partner
        )
        act_flat = stencil["act_flat"]
        base_shift = (
            u_cur.reshape(-1, dim)[act_flat] - orig_flat[act_flat]
        ).contiguous()

        init_v_ice = 0.0
        strict_ice_u = float(ice_tol) if ice_strict_tol is None else float(ice_strict_tol)
        if use_partner:
            ice_flat = act_flat
            ice_batch = stencil["act_coords"][:, 0]
            X_ice = X_flat[ice_flat].contiguous()
            u_ice_base = u_cur.reshape(-1, dim)[ice_flat].clone()
            local_ice_cap = float(ice_tol) * 0.70
            yv0_p = (X_ice + u_ice_base) @ A_to_vox + b_to_vox
            iny0_p = (((yv0_p >= 0.0) & (yv0_p <= vox_hi_t)).all(dim=-1)).to(dtype)
            py0_p = _gather_multilinear_sparse(
                partner_flat, empty_lut, empty_delta, ice_batch, yv0_p, spatial
            )
            r0_p = (u_ice_base + py0_p) @ err_to_vox_mat
            n0_p = (torch.sqrt((r0_p * r0_p).sum(dim=-1) + 1e-12) - 1e-6) * iny0_p
            init_v_ice = float(F.relu(n0_p - strict_ice_u).amax().item()) if n0_p.numel() > 0 else 0.0

        n_act = int(act_flat.shape[0])
        if int(steps) >= 8 and n_act > 40_000 and p_idx == 0:
            iters_p = 44
            eta_frac = 0.30
        elif int(steps) >= 8 and n_act > 25_000 and p_idx >= 1:
            iters_p = 90
            eta_frac = 0.20
        else:
            iters_p = (
                max(int(steps) * 6, 160)
                if (p_idx == 0 and int(steps) >= 8)
                else (max(int(steps) * 4, 110) if int(steps) >= 8 else max(int(steps) * 8, 24))
            )
            eta_frac = 0.15
        lr_p = lr_base * (1.10 if p_idx >= 2 else 1.0)
        mem_scale = 0.15 if use_partner else 0.35
        anc_scale = 0.25 * float(anchor_weight)

        init_min_tot = min(ext_min, init_min_sten)
        ext_min_u_t = torch.tensor(ext_min, device=device, dtype=dtype)
        best_min_t = torch.tensor(init_min_tot, device=device, dtype=dtype)
        best_feas_t = torch.tensor(init_min_tot >= float(jac_floor), device=device, dtype=torch.bool)
        best_vice_t = torch.tensor(
            init_v_ice if (init_min_tot >= float(jac_floor)) else 1e9, device=device, dtype=dtype
        )
        imp_t = torch.tensor(False, device=device, dtype=torch.bool)
        ever_done_u = torch.tensor(False, device=device, dtype=torch.bool)
        jac_fl_t = float(jac_floor)

        with torch.enable_grad():
            df = torch.zeros(
                (act_flat.shape[0], dim), device=device, dtype=dtype, requires_grad=True
            )
            opt = torch.optim.Adam([df], lr=max(lr_p, 1e-4), foreach=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=iters_p, eta_min=max(lr_p * eta_frac, 1e-5)
            )
            best_df = df.detach().clone()

            for _it in range(iters_p):
                opt.zero_grad(set_to_none=True)
                if use_partner:
                    d2_s, d4_s, sm_df = _eval_sparse_row_step(
                        df, stencil, row_bases, s_D, use_physical, include_cell=False
                    )
                else:
                    d2_s, d4_s, dp_s, dm_s, sm_df = _eval_sparse_row_step(
                        df, stencil, row_bases, s_D, use_physical, include_cell=True
                    )
                dmin_s = torch.minimum(d2_s, d4_s) if include_4th else d2_s

                e_ice = df.new_zeros(())
                v_ice_strict = None
                if use_partner:
                    u_ice = u_ice_base + df
                    yv = (X_ice + u_ice) @ A_to_vox + b_to_vox
                    iny = (((yv >= 0.0) & (yv <= vox_hi_t)).all(dim=-1)).to(dtype)
                    py = _gather_multilinear_sparse(
                        partner_flat, empty_lut, empty_delta, ice_batch, yv, spatial
                    )
                    r_vox = (u_ice + py) @ err_to_vox_mat
                    n_ice = (torch.sqrt((r_vox * r_vox).sum(dim=-1) + 1e-12) - 1e-6) * iny
                    v_ice_strict = F.relu(n_ice - strict_ice_u)
                    v_ice_pen = F.relu(n_ice - local_ice_cap)
                    e_ice = (
                        28.0 * _active_lse(v_ice_pen, 25.0)
                        + 12.0 * (v_ice_pen * v_ice_pen).sum()
                        + 0.02 * (r_vox * r_vox * iny.unsqueeze(-1)).sum()
                    )

                with torch.no_grad():
                    s_min_t = (
                        torch.minimum(ext_min_u_t, dmin_s.amin())
                        if dmin_s.numel() > 0
                        else ext_min_u_t
                    )
                    max_vi_t = (
                        v_ice_strict.amax()
                        if (v_ice_strict is not None and v_ice_strict.numel() > 0)
                        else best_vice_t.new_zeros(())
                    )
                    feas_now = s_min_t >= jac_fl_t
                    done_u = feas_now & (max_vi_t <= 0.0)
                    if use_partner:
                        take_f = feas_now & ((~best_feas_t) | (max_vi_t < best_vice_t - 1e-4))
                    else:
                        take_f = feas_now & (~best_feas_t)
                    take_u = (~feas_now) & (~best_feas_t) & (s_min_t > best_min_t + 1e-5)
                    take = (~ever_done_u) & (take_f | take_u | done_u)
                    best_df = torch.where(take, df.detach(), best_df)
                    best_feas_t = best_feas_t | ((~ever_done_u) & (take_f | done_u))
                    best_vice_t = torch.where((~ever_done_u) & (take_f | done_u), max_vi_t, best_vice_t)
                    best_min_t = torch.where(take, s_min_t, best_min_t)
                    imp_t = imp_t | take
                    ever_done_u = ever_done_u | done_u
                    if (_it & 7) == 7 or _it == iters_p - 1:
                        if bool(ever_done_u.item()):
                            break

                v2 = F.relu(target_det - d2_s)
                if include_4th:
                    v4 = F.relu(target_det - d4_s)
                    if not use_partner:
                        vp_c = F.relu(cell_floor - dp_s)
                        vm_c = F.relu(cell_floor - dm_s)
                        e_lse = (
                            30.0 * _active_lse_batched(torch.stack([v2, v4], dim=0), 25.0)
                            + 15.0 * _active_lse_batched(torch.stack([vp_c, vm_c], dim=0), 20.0)
                        )
                        e_cell = 0.25 * float(jac_weight) * (vp_c.square().sum() + vm_c.square().sum())
                    else:
                        e_lse = 75.0 * _active_lse_batched(torch.stack([v2, v4], dim=0), 25.0)
                        e_cell = df.new_zeros(())
                    e_hng = float(jac_weight) * (
                        0.5 * (v2.square().sum() + v4.square().sum())
                        + 0.15 * (v2.sum() + v4.sum())
                    )
                else:
                    if not use_partner:
                        vp_c = F.relu(cell_floor - dp_s)
                        vm_c = F.relu(cell_floor - dm_s)
                        e_lse = (
                            30.0 * _active_lse(v2, 25.0)
                            + 15.0 * _active_lse_batched(torch.stack([vp_c, vm_c], dim=0), 20.0)
                        )
                        e_cell = 0.25 * float(jac_weight) * (vp_c.square().sum() + vm_c.square().sum())
                    else:
                        e_lse = 75.0 * _active_lse(v2, 25.0)
                        e_cell = df.new_zeros(())
                    e_hng = float(jac_weight) * (0.5 * v2.square().sum() + 0.15 * v2.sum())

                e_cap = df.new_zeros(())
                if max_det_cap is not None:
                    c2 = F.relu(d2_s - float(max_det_cap))
                    e_cap = 0.25 * c2.square().sum()

                tot_vox = (df if base_shift is None else (base_shift + df)) @ err_to_vox_mat
                e_reg = mem_scale * sm_df + anc_scale * tot_vox.square().sum()

                loss = e_lse + e_hng + e_cell + e_cap + e_ice + e_reg
                loss.backward()
                opt.step()
                sched.step()

        moved_act = best_df.abs().amax(dim=-1) > 1e-7
        if bool(imp_t.item()) and bool(moved_act.any().item()):
            u_cur.view(-1, dim).index_add_(0, act_flat[moved_act], best_df[moved_act])
            ever_active.view(-1)[act_flat[moved_act]] = True
            steps_taken += 1
            accepted_any = True
            d2_end, d4_end = _eval_sparse_dual_fd_det(
                u_cur.reshape(-1, dim)[stencil["pool_flat"]],
                stencil,
                D_rev_inv,
                eye,
                use_physical,
            )
            dmin_end = torch.minimum(d2_end, d4_end) if include_4th else d2_end
            det_min_cur.view(-1)[stencil["sten_flat"]] = dmin_end
            cur_global_min = min(ext_min, float(dmin_end.amin().item()))
            if use_partner:
                u_ice_end = u_cur.reshape(-1, dim)[ice_flat]
                yv_e = (X_ice + u_ice_end) @ A_to_vox + b_to_vox
                iny_e = (((yv_e >= 0.0) & (yv_e <= vox_hi_t)).all(dim=-1)).to(dtype)
                py_e = _gather_multilinear_sparse(
                    partner_flat, empty_lut, empty_delta, ice_batch, yv_e, spatial
                )
                r_e = (u_ice_end + py_e) @ err_to_vox_mat
                n_e = (torch.sqrt((r_e * r_e).sum(dim=-1) + 1e-12) - 1e-6) * iny_e
                still_ice = n_e > strict_ice_u
                cur_ice_defects.zero_()
                if bool(still_ice.any().item()):
                    cur_ice_defects.view(-1)[ice_flat[still_ice]] = True
                if cur_global_min >= float(jac_floor) and not bool(still_ice.any().item()):
                    break
            else:
                if cur_global_min >= float(jac_floor):
                    break
        else:
            break

    if not accepted_any:
        return u_orig, det_min_init, float(det_min_init.amin().item()), 0, ever_active

    u_cur = torch.where(ever_active.unsqueeze(-1), u_cur, u_orig)
    return u_cur, det_min_cur, cur_global_min, steps_taken, ever_active


def _safe_lm_subset_backtrack(
    target_field: torch.Tensor,
    inv_field: torch.Tensor,
    cand_mask: torch.Tensor,
    det_min_inv: torch.Tensor,
    X_base: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    right_mat: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    steps_fd: list[float],
    D_rev_inv: torch.Tensor,
    use_physical: bool,
    jac_floor: float,
    ice_tol: float,
    lm_steps: int,
    max_step_unit: float,
    warm_inv_field: Optional[torch.Tensor] = None,
    lut_buf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pointwise-monotone stencil-safe damped LM inverse sync: accepts updates only on voxels
    that strictly decrease their own squared chord residual (by ``>= 1e-5``) while keeping every
    dual-stencil ring above ``min(det_orig, jac_floor + 1e-5)``."""
    if lm_steps <= 0 or not bool(cand_mask.any().item()):
        return inv_field, det_min_inv, torch.zeros_like(cand_mask)

    device = inv_field.device
    dtype = inv_field.dtype
    include_4th = bool(use_physical)
    sten_r = 2 if include_4th else 1
    B = cand_mask.shape[0]
    dim = inv_field.shape[-1]
    spatial = tuple(cand_mask.shape[1:])
    eye = torch.eye(dim, device=device, dtype=dtype)

    stencil = _build_sparse_dual_fd_stencil(
        cand_mask, steps_fd, dtype, include_4th=include_4th, lut_buf=lut_buf
    )
    act_flat = stencil["act_flat"]
    if act_flat.shape[0] == 0:
        return inv_field, det_min_inv, torch.zeros_like(cand_mask)

    act_coords = torch.nonzero(cand_mask, as_tuple=False)
    act_batch = act_coords[:, 0]
    X_flat = X_base.expand(B, *spatial, dim).reshape(-1, dim)
    X_act = X_flat[act_flat]
    target_flat = target_field.reshape(-1, dim)
    base_flat = inv_field.reshape(-1, dim)
    base_act = base_flat[act_flat].clone()

    empty_lut = torch.empty((0,), dtype=torch.long, device=device)
    empty_delta = inv_field.new_zeros((0, dim))

    yv0 = (X_act + base_act) @ A_to_vox + b_to_vox
    ut0 = _gather_multilinear_sparse(
        target_flat, empty_lut, empty_delta, act_batch, yv0, spatial
    )
    r_sq_0 = (((base_act + ut0) @ err_to_vox_mat) ** 2).sum(dim=-1)

    seed_field = inv_field
    if warm_inv_field is not None:
        warm_act = warm_inv_field.reshape(-1, dim)[act_flat]
        yv_w = (X_act + warm_act) @ A_to_vox + b_to_vox
        ut_w = _gather_multilinear_sparse(
            target_flat, empty_lut, empty_delta, act_batch, yv_w, spatial
        )
        r_sq_w = (((warm_act + ut_w) @ err_to_vox_mat) ** 2).sum(dim=-1)
        use_w = r_sq_w < r_sq_0 - 1e-4
        if bool(use_w.any().item()):
            seed_field = inv_field.clone()
            seed_field.view(-1, dim)[act_flat[use_w]] = warm_act[use_w]

    u_lm = _pointwise_lm_refine_subset(
        target_field,
        seed_field,
        cand_mask,
        X_base,
        A_to_vox,
        b_to_vox,
        right_mat,
        err_to_vox_mat,
        steps=int(lm_steps),
        tol_vox=float(ice_tol) * 0.30,
        max_step_unit=max_step_unit,
    )

    delta_act = u_lm.reshape(-1, dim)[act_flat] - base_act
    scale_act = torch.ones((act_flat.shape[0], 1), device=device, dtype=dtype)
    # Zero immediately on voxels whose LM trial didn't move
    scale_act = torch.where(
        (delta_act.abs().amax(dim=-1, keepdim=True) > 1e-7),
        scale_act,
        torch.zeros_like(scale_act),
    )
    if not bool((scale_act > 0.0).any().item()):
        return inv_field, det_min_inv, torch.zeros_like(cand_mask)

    base_pool = base_flat[stencil["pool_flat"]].clone()
    sten_orig_min = det_min_inv.reshape(-1)[stencil["sten_flat"]]
    local_floor = torch.minimum(
        sten_orig_min, torch.full_like(sten_orig_min, float(jac_floor) + 1e-5)
    )

    n_full = max(int(np.prod(spatial)), 1)
    own_lut = lut_buf is None or lut_buf.numel() != B * n_full
    lut_act = (
        torch.full((B * n_full,), -1, dtype=torch.long, device=device)
        if own_lut
        else lut_buf
    )
    lut_act[act_flat] = torch.arange(act_flat.shape[0], device=device)

    while True:
        v_try = base_act + delta_act * scale_act
        yv_t = (X_act + v_try) @ A_to_vox + b_to_vox
        ut_t = _gather_multilinear_sparse(
            target_flat, empty_lut, empty_delta, act_batch, yv_t, spatial
        )
        r_sq_t = (((v_try + ut_t) @ err_to_vox_mat) ** 2).sum(dim=-1)
        no_imp = (scale_act.squeeze(-1) > 0.0) & (r_sq_t >= r_sq_0 - 1e-5)
        if bool(no_imp.any().item()):
            cur_ni = scale_act[no_imp]
            scale_act[no_imp] = torch.where(
                cur_ni > 0.25, cur_ni * 0.5, torch.zeros_like(cur_ni)
            )
            if not bool((scale_act > 0.0).any().item()):
                break
            continue

        u_pool = base_pool.index_add(0, stencil["act_in_pool"], delta_act * scale_act)
        d2_s, d4_s = _eval_sparse_dual_fd_det(u_pool, stencil, D_rev_inv, eye, use_physical)
        dmin_s = torch.minimum(d2_s, d4_s) if include_4th else d2_s
        bad = dmin_s < local_floor - 1e-6
        if not bool(bad.any().item()):
            break
        bad_mask = torch.zeros_like(cand_mask)
        bad_mask.view(-1)[stencil["sten_flat"][bad]] = True
        bad_ring = _dilate_mask_nd(bad_mask, sten_r) & cand_mask
        bad_act_idx = lut_act[
            _coords_to_flat_idx(torch.nonzero(bad_ring, as_tuple=False), spatial)
        ]
        bad_act_idx = bad_act_idx[bad_act_idx >= 0]
        cur_s = scale_act[bad_act_idx]
        if not bool((cur_s > 0.0).any().item()):
            break
        scale_act[bad_act_idx] = torch.where(
            cur_s > 0.125, cur_s * 0.5, torch.zeros_like(cur_s)
        )
        if not bool((scale_act > 0.0).any().item()):
            break

    if own_lut:
        del lut_act
    else:
        lut_act[act_flat] = -1
    used = (scale_act.squeeze(-1) > 0.0) & (delta_act.abs().amax(dim=-1) > 1e-7)
    if not bool(used.any().item()):
        return inv_field, det_min_inv, torch.zeros_like(cand_mask)

    out = inv_field.clone()
    out.view(-1, dim).index_add_(0, act_flat[used], (delta_act * scale_act)[used])
    d2_s, d4_s = _eval_sparse_dual_fd_det(
        out.reshape(-1, dim)[stencil["pool_flat"]], stencil, D_rev_inv, eye, use_physical
    )
    det_min_out = det_min_inv.clone()
    det_min_out.view(-1)[stencil["sten_flat"]] = (
        torch.minimum(d2_s, d4_s) if include_4th else d2_s
    )
    mod_mask = torch.zeros_like(cand_mask)
    mod_mask.view(-1)[act_flat[used]] = True
    return out, det_min_out, mod_mask


def _static_multilinear_corners(
    batch_idx: torch.Tensor,
    vox_coords: torch.Tensor,
    spatial: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(c_idx, w_c)`` each of shape ``(M, 2**dim)`` for static multilinear queries."""
    dim = len(spatial)
    n_full = max(int(np.prod(spatial)), 1)
    base_off = batch_idx * n_full
    if dim == 2:
        H, W = spatial
        vy = vox_coords[:, 0].clamp(0.0, float(max(H - 1, 0)))
        vx = vox_coords[:, 1].clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = vy - iy0.to(dtype)
        tx = vx - ix0.to(dtype)
        omy, omx = 1.0 - ty, 1.0 - tx
        c_idx = torch.stack(
            [
                base_off + iy0 * W + ix0,
                base_off + iy0 * W + ix1,
                base_off + iy1 * W + ix0,
                base_off + iy1 * W + ix1,
            ],
            dim=1,
        )
        w_c = torch.stack([omy * omx, omy * tx, ty * omx, ty * tx], dim=1)
        return c_idx, w_c
    Z, Y, X = spatial
    vz = vox_coords[:, 0].clamp(0.0, float(max(Z - 1, 0)))
    vy = vox_coords[:, 1].clamp(0.0, float(max(Y - 1, 0)))
    vx = vox_coords[:, 2].clamp(0.0, float(max(X - 1, 0)))
    iz0 = torch.floor(vz).long().clamp(0, max(Z - 2, 0))
    iy0 = torch.floor(vy).long().clamp(0, max(Y - 2, 0))
    ix0 = torch.floor(vx).long().clamp(0, max(X - 2, 0))
    iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
    iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
    ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
    tz = vz - iz0.to(dtype)
    ty = vy - iy0.to(dtype)
    tx = vx - ix0.to(dtype)
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx
    sz, sy = Y * X, X
    c_idx = torch.stack(
        [
            base_off + iz0 * sz + iy0 * sy + ix0,
            base_off + iz0 * sz + iy0 * sy + ix1,
            base_off + iz0 * sz + iy1 * sy + ix0,
            base_off + iz0 * sz + iy1 * sy + ix1,
            base_off + iz1 * sz + iy0 * sy + ix0,
            base_off + iz1 * sz + iy0 * sy + ix1,
            base_off + iz1 * sz + iy1 * sy + ix0,
            base_off + iz1 * sz + iy1 * sy + ix1,
        ],
        dim=1,
    )
    w_c = torch.stack(
        [
            omz * omy * omx,
            omz * omy * tx,
            omz * ty * omx,
            omz * ty * tx,
            tz * omy * omx,
            tz * omy * tx,
            tz * ty * omx,
            tz * ty * tx,
        ],
        dim=1,
    )
    return c_idx, w_c


def _prepare_chord_eval_split(
    eval_mask: torch.Tensor,
    act_mask: torch.Tensor,
    u_self_flat: torch.Tensor,
    u_other_flat: torch.Tensor,
    lut_act: torch.Tensor,
    X_flat: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    vox_hi_t: torch.Tensor,
    spatial: tuple[int, ...],
    extra_full_tensors: tuple[torch.Tensor, ...] = (),
) -> tuple[dict, tuple[torch.Tensor, ...]]:
    """Split ``eval_mask`` chord queries into moving (``q in act_mask``), static-active
    (``q not in act_mask`` with >=1 corner in ``act_mask``), and static-constant (0 corners in
    ``act_mask``), reordering ``extra_full_tensors`` identically so per-voxel caps align 1-to-1."""
    dtype = u_self_flat.dtype
    mov_mask = eval_mask & act_mask
    stat_mask = eval_mask & (~act_mask)

    mov_coords = torch.nonzero(mov_mask, as_tuple=False)
    mov_flat = _coords_to_flat_idx(mov_coords, spatial)
    mov_batch = mov_coords[:, 0]
    mov_act_idx = lut_act[mov_flat]
    X_mov = X_flat[mov_flat].contiguous()
    u_self_mov0 = u_self_flat[mov_flat].contiguous()

    stat_coords = torch.nonzero(stat_mask, as_tuple=False)
    stat_flat = _coords_to_flat_idx(stat_coords, spatial)
    if stat_flat.numel() > 0:
        stat_batch = stat_coords[:, 0]
        X_stat = X_flat[stat_flat]
        u_s0 = u_self_flat[stat_flat]
        vv_s = (X_stat + u_s0) @ A_to_vox + b_to_vox
        in_s = (((vv_s >= 0.0) & (vv_s <= vox_hi_t)).all(dim=-1)).to(dtype)
        c_idx, w_c = _static_multilinear_corners(stat_batch, vv_s, spatial, dtype)
        u_o0 = (u_other_flat[c_idx] * w_c.unsqueeze(-1)).sum(dim=1)
        base_sum = u_s0 + u_o0
        a_all = lut_act[c_idx]
        has_act = (a_all >= 0).any(dim=1)
        sa_flat = stat_flat[has_act]
        base_sum_sa = base_sum[has_act].contiguous()
        in_sa = in_s[has_act].contiguous()
        a_sa = a_all[has_act].clamp_min(0).contiguous()
        w_sa = (w_c[has_act] * (a_all[has_act] >= 0).to(dtype)).unsqueeze(-1).contiguous()
    else:
        sa_flat = mov_flat.new_zeros((0,))
        base_sum_sa = u_self_flat.new_zeros((0, u_self_flat.shape[-1]))
        in_sa = u_self_flat.new_zeros((0,))
        a_sa = mov_flat.new_zeros((0, 1))
        w_sa = u_self_flat.new_zeros((0, 1, 1))

    order_flat = torch.cat([mov_flat, sa_flat], dim=0)
    reordered_extras = tuple(t.reshape(-1)[order_flat].contiguous() for t in extra_full_tensors)
    plan = {
        "order_flat": order_flat,
        "mov_batch": mov_batch,
        "mov_act_idx": mov_act_idx,
        "X_mov": X_mov,
        "u_self_mov0": u_self_mov0,
        "base_sum_sa": base_sum_sa,
        "in_sa": in_sa,
        "a_sa": a_sa,
        "w_sa": w_sa,
    }
    return plan, reordered_extras


def _eval_chord_split(
    plan: dict,
    delta_self: torch.Tensor,
    delta_other: torch.Tensor,
    u_other_flat: torch.Tensor,
    lut_act: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    vox_hi_t: torch.Tensor,
    spatial: tuple[int, ...],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if plan["mov_act_idx"].numel() > 0:
        u_s = plan["u_self_mov0"] + delta_self[plan["mov_act_idx"]]
        vv = (plan["X_mov"] + u_s) @ A_to_vox + b_to_vox
        in_m = (((vv >= 0.0) & (vv <= vox_hi_t)).all(dim=-1)).to(u_s.dtype)
        u_o = _gather_multilinear_sparse(
            u_other_flat, lut_act, delta_other, plan["mov_batch"], vv, spatial
        )
        r_m = (u_s + u_o) @ err_to_vox_mat
        parts.append((torch.sqrt((r_m * r_m).sum(dim=-1) + 1e-12) - 1e-6) * in_m)
    if plan["in_sa"].numel() > 0:
        du_o = (delta_other[plan["a_sa"]] * plan["w_sa"]).sum(dim=1)
        r_sa = (plan["base_sum_sa"] + du_o) @ err_to_vox_mat
        parts.append((torch.sqrt((r_sa * r_sa).sum(dim=-1) + 1e-12) - 1e-6) * plan["in_sa"])
    if not parts:
        return delta_self.new_zeros((0,))
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _joint_symmetric_shell_polish(
    u_f_in: torch.Tensor,
    u_i_in: torch.Tensor,
    W_fwd_orig: torch.Tensor,
    det_min_f_in: torch.Tensor,
    det_min_i_in: torch.Tensor,
    n_fi0: torch.Tensor,
    n_fi_in: torch.Tensor,
    n_if_in: torch.Tensor,
    fold_shell: torch.Tensor,
    dom_b: torch.Tensor,
    boundary_mask: torch.Tensor,
    stationary_boundary: bool,
    steps_fd: list[float],
    D_rev_inv: torch.Tensor,
    X_base: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    use_physical: bool,
    jac_floor: float,
    ice_tol: float,
    jac_weight: float,
    anchor_weight: float,
    steps: int,
    step_size: float,
    dilation_radius: int,
    mean_sp: float,
    eval_chord_fn,
    scatter_fn,
    lut_buf: Optional[torch.Tensor] = None,
):
    """Principled joint symmetric shrinking-shell projection optimizing ``(df, di)`` simultaneously
    on the sparse defect shell.

    Why joint ``(df, di)`` optimization is mathematically required:
    If ``u_f`` is frozen at a sharp pinch ``det(I + nabla u_f) = jac_floor = 0.05``, its exact
    inverse has ``det(I + nabla u_f^{-1}) = 1 / 0.05 = 20`` across a sub-voxel layer, causing the
    4th-order 5-point stencil ``D^{(4)}_h(u_f^{-1})`` to Runge-undershoot negative (`< -2.0`).
    Optimizing ``(df, di)`` jointly lets gradients through ``r_if`` and ``r_fi`` smoothly widen
    ``u_f``'s pinch while keeping ``u_i`` diffeomorphic on BOTH ``D_{2h}`` and ``D^{(4)}_h`` AND
    driving chord inverse-consistency below ``0.80 * ice_tol`` simultaneously.
    """
    device = u_f_in.device
    dtype = u_f_in.dtype
    B = u_f_in.shape[0]
    dim = u_f_in.shape[-1]
    spatial = tuple(u_f_in.shape[1:-1])
    include_4th = bool(use_physical)
    n_full = max(int(np.prod(spatial)), 1)

    u_f = u_f_in.clone()
    u_i = u_i_in.clone()
    det_min_f = det_min_f_in.clone()
    det_min_i = det_min_i_in.clone()
    n_fi = n_fi_in.clone() if n_full > 65_536 else n_fi_in
    n_if = n_if_in.clone() if n_full > 65_536 else n_if_in
    ever_f = torch.zeros_like(dom_b)
    ever_i = torch.zeros_like(dom_b)
    steps_taken = 0

    if steps <= 0:
        return u_f, u_i, det_min_f, det_min_i, n_fi, n_if, 0, ever_f, ever_i

    eye = torch.eye(dim, device=device, dtype=dtype)
    D_rev_T, s_D = _d_rev_t_and_scale(D_rev_inv)
    X_flat = X_base.expand(B, *spatial, dim).reshape(-1, dim)
    orig_f_flat = W_fwd_orig.reshape(-1, dim)
    vox_hi_t = torch.tensor(
        [float(max(s - 1, 0)) for s in spatial], device=device, dtype=dtype
    )

    # On the fold shell, drive BOTH n_if and n_fi below 0.80 * ice_tol; outside the fold shell,
    # never treat pre-existing untouched trilinear curvature n_fi0 as a defect!
    target_ice = float(ice_tol) * 0.80
    cap_fi_full = torch.where(
        fold_shell,
        torch.full_like(n_fi0, target_ice),
        torch.maximum(n_fi0 + 0.02, torch.full_like(n_fi0, float(ice_tol))),
    )
    if n_full > 65_536:
        init_ice_base = max(float(ice_tol), float(n_if_in.amax().item()))
        strict_if_full = torch.where(
            fold_shell,
            torch.full_like(n_fi0, max(float(ice_tol), 0.65 * init_ice_base)),
            torch.maximum(n_if_in + 0.05, torch.full_like(n_fi0, float(ice_tol))),
        )
        strict_fi_full = torch.where(
            fold_shell,
            torch.full_like(n_fi0, max(float(ice_tol), 0.75 * init_ice_base)),
            torch.maximum(n_fi0 + 0.50, torch.full_like(n_fi0, 2.5 * float(ice_tol))),
        )
    else:
        strict_if_full = torch.full_like(n_fi0, float(ice_tol))
        strict_fi_full = torch.where(
            fold_shell,
            torch.full_like(n_fi0, float(ice_tol)),
            torch.maximum(n_fi0 + 0.05, torch.full_like(n_fi0, float(ice_tol))),
        )

    max_passes = (5 if n_full > 65_536 else 8) if int(steps) >= 8 else 1
    lr_base = 0.075 * mean_sp * max(min(float(step_size) / 0.30, 2.0), 0.25)
    target_det = float(jac_floor) + 0.045
    cell_floor = min(float(jac_floor), 0.02) + 0.01
    jac_fl_t = float(jac_floor)
    chord_exact = True

    for p_idx in range(max_passes):
        fold_def = (det_min_f < jac_fl_t) | (det_min_i < jac_fl_t)
        ice_def_if = dom_b & (n_if > strict_if_full)
        ice_def_fi = dom_b & (n_fi > strict_fi_full)
        rem_defects = fold_def | ice_def_if | ice_def_fi
        if not bool(rem_defects.any().item()):
            if not chord_exact:
                _, _, n_fi, n_if = eval_chord_fn(u_f, u_i)
                chord_exact = True
                ice_def_if = dom_b & (n_if > strict_if_full)
                ice_def_fi = dom_b & (n_fi > strict_fi_full)
                rem_defects = fold_def | ice_def_if | ice_def_fi
                if not bool(rem_defects.any().item()):
                    break
            else:
                break

        # Include mapped partner endpoints of active defects so both ends of every active chord
        # lie inside the joint active shell with full multi-voxel 5-point stencil clearance.
        src_f = (det_min_f < jac_fl_t) | ice_def_fi
        src_i = (det_min_i < jac_fl_t) | ice_def_if
        tgt_from_f, _ = (
            scatter_fn(src_f, u_f, dilate_r=0)
            if bool(src_f.any().item())
            else (torch.zeros_like(dom_b), None)
        )
        tgt_from_i, _ = (
            scatter_fn(src_i, u_i, dilate_r=0)
            if bool(src_i.any().item())
            else (torch.zeros_like(dom_b), None)
        )

        raw_seeds = rem_defects | tgt_from_f | tgt_from_i
        n_seeds = int(raw_seeds.sum().item())
        if n_full > 65_536:
            if n_seeds > 2_000 and p_idx < 3:
                rad_p = 1
            elif n_seeds <= 600 and p_idx >= 2:
                rad_p = int(dilation_radius) + 1
            else:
                rad_p = int(dilation_radius)
        else:
            rad_p = int(dilation_radius) + min(p_idx // 2, 1)
        act_mask = _dilate_mask_nd(raw_seeds, rad_p) & dom_b
        if not bool(act_mask.any().item()):
            break

        stencil = _build_sparse_dual_fd_stencil(
            act_mask, steps_fd, dtype, include_4th=include_4th, lut_buf=lut_buf
        )
        act_flat = stencil["act_flat"]
        sten_flat = stencil["sten_flat"]
        sten_prev_f = det_min_f.view(-1)[sten_flat].clone()
        sten_prev_i = det_min_i.view(-1)[sten_flat].clone()
        sten_min_f = float(sten_prev_f.amin().item()) if sten_prev_f.numel() > 0 else float("inf")
        sten_min_i = float(sten_prev_i.amin().item()) if sten_prev_i.numel() > 0 else float("inf")
        if sten_flat.shape[0] < det_min_f.numel():
            det_min_f.view(-1)[sten_flat] = float("inf")
            ext_min_f = float(det_min_f.amin().item())
            det_min_f.view(-1)[sten_flat] = sten_prev_f
            det_min_i.view(-1)[sten_flat] = float("inf")
            ext_min_i = float(det_min_i.amin().item())
            det_min_i.view(-1)[sten_flat] = sten_prev_i
        else:
            ext_min_f = float("inf")
            ext_min_i = float("inf")

        # Build sparse chord evaluation sets covering act_mask and its mapped preimages in O(|act_mask|)
        pre_if, _ = scatter_fn(act_mask, u_f, dilate_r=2)
        pre_fi, _ = scatter_fn(act_mask, u_i, dilate_r=2)
        eval_if_mask = (_dilate_mask_nd(act_mask, 1) | pre_if | ice_def_if) & dom_b
        eval_fi_mask = (_dilate_mask_nd(act_mask, 1) | pre_fi | ice_def_fi) & dom_b

        own_lut = lut_buf is None or lut_buf.numel() != B * n_full
        lut_act = (
            torch.full((B * n_full,), -1, dtype=torch.long, device=device)
            if own_lut
            else lut_buf
        )
        lut_act[act_flat] = torch.arange(act_flat.shape[0], device=device)

        uf_flat = u_f.reshape(-1, dim)
        ui_flat = u_i.reshape(-1, dim)
        uf_pool0 = uf_flat[stencil["pool_flat"]].clone()
        ui_pool0 = ui_flat[stencil["pool_flat"]].clone()
        row_bases_f = _precompute_sparse_row_bases(
            uf_pool0, stencil, D_rev_T, use_physical, include_cell=True
        )
        row_bases_i = _precompute_sparse_row_bases(
            ui_pool0, stencil, D_rev_T, use_physical, include_cell=False
        )
        base_shift_f = (uf_flat[act_flat] - orig_f_flat[act_flat]).contiguous()

        plan_if, (strict_if_sub,) = _prepare_chord_eval_split(
            eval_if_mask,
            act_mask,
            ui_flat,
            uf_flat,
            lut_act,
            X_flat,
            A_to_vox,
            b_to_vox,
            err_to_vox_mat,
            vox_hi_t,
            spatial,
            extra_full_tensors=(strict_if_full,),
        )
        plan_fi, (cap_fi_sub, strict_fi_sub) = _prepare_chord_eval_split(
            eval_fi_mask,
            act_mask,
            uf_flat,
            ui_flat,
            lut_act,
            X_flat,
            A_to_vox,
            b_to_vox,
            err_to_vox_mat,
            vox_hi_t,
            spatial,
            extra_full_tensors=(cap_fi_full, strict_fi_full),
        )

        init_mf = min(ext_min_f, sten_min_f)
        init_mi = min(ext_min_i, sten_min_i)
        init_feas = (init_mf >= jac_fl_t) and (init_mi >= jac_fl_t)

        if_ord = plan_if["order_flat"]
        fi_ord = plan_fi["order_flat"]
        init_vif = (
            float(F.relu(n_if.reshape(-1)[if_ord] - strict_if_sub).amax().item())
            if if_ord.numel() > 0
            else 0.0
        )
        init_vfi = (
            float(F.relu(n_fi.reshape(-1)[fi_ord] - strict_fi_sub).amax().item())
            if fi_ord.numel() > 0
            else 0.0
        )
        init_v_both = max(init_vif, init_vfi)

        ext_mf_t = torch.tensor(ext_min_f, device=device, dtype=dtype)
        ext_mi_t = torch.tensor(ext_min_i, device=device, dtype=dtype)
        best_feas_t = torch.tensor(init_feas, device=device, dtype=torch.bool)
        best_min_both_t = torch.tensor(min(init_mf, init_mi), device=device, dtype=dtype)
        best_v_both_t = torch.tensor(
            init_v_both if init_feas else 1e9, device=device, dtype=dtype
        )
        imp_j_t = torch.tensor(False, device=device, dtype=torch.bool)
        ever_done_j = torch.tensor(False, device=device, dtype=torch.bool)

        n_act = int(act_flat.shape[0])
        if n_full > 65_536 and int(steps) >= 8:
            if p_idx == 0 and n_act > 20_000:
                iters_p = 40
                eta_frac = 0.30
            elif n_act > 120_000:
                iters_p = 48
                eta_frac = 0.25
            elif n_act > 35_000:
                iters_p = 64
                eta_frac = 0.22
            elif n_act > 10_000:
                iters_p = 84
                eta_frac = 0.18
            else:
                iters_p = 112
                eta_frac = 0.15
        else:
            iters_p = max(int(steps) * 6, 160) if p_idx == 0 else max(int(steps) * 5, 130)
            eta_frac = 0.15
        lr_p = lr_base * (1.10 if p_idx >= 2 else 1.0)
        anc_f_w = 0.20 * float(anchor_weight)

        with torch.enable_grad():
            df = torch.zeros(
                (act_flat.shape[0], dim), device=device, dtype=dtype, requires_grad=True
            )
            di = torch.zeros(
                (act_flat.shape[0], dim), device=device, dtype=dtype, requires_grad=True
            )
            best_df = df.detach().clone()
            best_di = di.detach().clone()
            opt = torch.optim.Adam([df, di], lr=max(lr_p, 1e-4), foreach=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=iters_p, eta_min=max(lr_p * eta_frac, 1e-5)
            )

            for _it in range(iters_p):
                opt.zero_grad(set_to_none=True)
                d2f, d4f, dpf, dmf_cell, sm_f = _eval_sparse_row_step(
                    df, stencil, row_bases_f, s_D, use_physical, include_cell=True
                )
                d2i, d4i, sm_i = _eval_sparse_row_step(
                    di, stencil, row_bases_i, s_D, use_physical, include_cell=False
                )
                dmf = torch.minimum(d2f, d4f) if include_4th else d2f
                dmi = torch.minimum(d2i, d4i) if include_4th else d2i

                nif_k = _eval_chord_split(
                    plan_if,
                    di,
                    df,
                    uf_flat,
                    lut_act,
                    A_to_vox,
                    b_to_vox,
                    err_to_vox_mat,
                    vox_hi_t,
                    spatial,
                )
                nfi_k = _eval_chord_split(
                    plan_fi,
                    df,
                    di,
                    ui_flat,
                    lut_act,
                    A_to_vox,
                    b_to_vox,
                    err_to_vox_mat,
                    vox_hi_t,
                    spatial,
                )

                vif_pen = F.relu(nif_k - target_ice)
                vfi_pen = F.relu(nfi_k - cap_fi_sub)

                with torch.no_grad():
                    cur_mf_t = torch.minimum(ext_mf_t, dmf.amin()) if dmf.numel() > 0 else ext_mf_t
                    cur_mi_t = torch.minimum(ext_mi_t, dmi.amin()) if dmi.numel() > 0 else ext_mi_t
                    cur_min_both_t = torch.minimum(cur_mf_t, cur_mi_t)
                    vif_s_max = (
                        F.relu(nif_k - strict_if_sub).amax()
                        if nif_k.numel() > 0
                        else cur_mf_t.new_zeros(())
                    )
                    vfi_s_max = (
                        F.relu(nfi_k - strict_fi_sub).amax()
                        if nfi_k.numel() > 0
                        else cur_mf_t.new_zeros(())
                    )
                    cur_v_both_t = torch.maximum(vif_s_max, vfi_s_max)
                    feas_now = (cur_mf_t >= jac_fl_t) & (cur_mi_t >= jac_fl_t)
                    no_refold_f = (
                        (cur_mf_t >= jac_fl_t)
                        if (init_mf >= jac_fl_t)
                        else torch.ones_like(feas_now)
                    )
                    no_refold_i = (
                        (cur_mi_t >= jac_fl_t)
                        if (init_mi >= jac_fl_t)
                        else torch.ones_like(feas_now)
                    )
                    take_f = feas_now & ((~best_feas_t) | (cur_v_both_t < best_v_both_t - 1e-4))
                    take_u = (
                        (~feas_now)
                        & (~best_feas_t)
                        & no_refold_f
                        & no_refold_i
                        & (cur_min_both_t > best_min_both_t + 1e-5)
                    )
                    ice_gate = 1.0
                    v_pen_max = torch.maximum(
                        vif_pen.amax() if vif_pen.numel() > 0 else cur_mf_t.new_zeros(()),
                        vfi_pen.amax() if vfi_pen.numel() > 0 else cur_mf_t.new_zeros(()),
                    )
                    done_now = feas_now & (cur_v_both_t <= 0.0) & (v_pen_max <= 0.0)
                    take = (~ever_done_j) & (take_f | take_u | done_now)
                    best_df = torch.where(take, df.detach(), best_df)
                    best_di = torch.where(take, di.detach(), best_di)
                    best_feas_t = best_feas_t | ((~ever_done_j) & (take_f | done_now))
                    best_v_both_t = torch.where(
                        (~ever_done_j) & (take_f | done_now), cur_v_both_t, best_v_both_t
                    )
                    best_min_both_t = torch.where(take, cur_min_both_t, best_min_both_t)
                    imp_j_t = imp_j_t | take
                    ever_done_j = ever_done_j | done_now
                    if (_it & 7) == 7 or _it == iters_p - 1:
                        if bool(ever_done_j.item()):
                            break

                v2f = F.relu(target_det - d2f)
                v2i = F.relu(target_det - d2i)
                if include_4th:
                    v4f = F.relu(target_det - d4f)
                    v4i = F.relu(target_det - d4i)
                    v_jac = torch.stack([v2f, v4f, v2i, v4i], dim=0)
                else:
                    v_jac = torch.stack([v2f, v2i], dim=0)
                e_jac = 75.0 * _active_lse_batched(v_jac, 25.0) + float(jac_weight) * (
                    0.5 * v_jac.square().sum() + 0.15 * v_jac.sum()
                )

                v_cell = torch.stack(
                    [F.relu(cell_floor - dpf), F.relu(cell_floor - dmf_cell)], dim=0
                )
                e_cell = 15.0 * _active_lse_batched(v_cell, 20.0) + 0.25 * float(
                    jac_weight
                ) * v_cell.square().sum()

                e_ice = ice_gate * (
                    28.0 * (_active_lse(vif_pen, 22.0) + _active_lse(vfi_pen, 22.0))
                    + 12.0 * (vif_pen.square().sum() + vfi_pen.square().sum())
                    + 0.20 * (vif_pen.sum() + vfi_pen.sum())
                )

                tot_sf = (base_shift_f + df) @ err_to_vox_mat
                e_reg = 0.22 * sm_f + 0.10 * sm_i + anc_f_w * tot_sf.square().sum()

                loss = e_jac + e_cell + e_ice + e_reg
                loss.backward()
                opt.step()
                sched.step()

        if bool(imp_j_t.item()):
            moved_f = best_df.abs().amax(dim=-1) > 1e-7
            moved_i = best_di.abs().amax(dim=-1) > 1e-7
            has_mf = bool(moved_f.any().item())
            has_mi = bool(moved_i.any().item())
            if has_mf or has_mi:
                nif_best = (
                    _eval_chord_split(
                        plan_if,
                        best_di,
                        best_df,
                        uf_flat,
                        lut_act,
                        A_to_vox,
                        b_to_vox,
                        err_to_vox_mat,
                        vox_hi_t,
                        spatial,
                    )
                    if n_full > 65_536
                    else None
                )
                nfi_best = (
                    _eval_chord_split(
                        plan_fi,
                        best_df,
                        best_di,
                        ui_flat,
                        lut_act,
                        A_to_vox,
                        b_to_vox,
                        err_to_vox_mat,
                        vox_hi_t,
                        spatial,
                    )
                    if n_full > 65_536
                    else None
                )
                if has_mf:
                    u_f.view(-1, dim).index_add_(0, act_flat[moved_f], best_df[moved_f])
                    ever_f.view(-1)[act_flat[moved_f]] = True
                if has_mi:
                    u_i.view(-1, dim).index_add_(0, act_flat[moved_i], best_di[moved_i])
                    ever_i.view(-1)[act_flat[moved_i]] = True
                steps_taken += 1

                if own_lut:
                    del lut_act
                else:
                    lut_act[act_flat] = -1

                cur_min_df = ext_min_f
                if has_mf:
                    d2f_e, d4f_e = _eval_sparse_dual_fd_det(
                        u_f.reshape(-1, dim)[stencil["pool_flat"]],
                        stencil,
                        D_rev_inv,
                        eye,
                        use_physical,
                    )
                    dmf_e = torch.minimum(d2f_e, d4f_e) if include_4th else d2f_e
                    det_min_f.view(-1)[stencil["sten_flat"]] = dmf_e
                    cur_min_df = min(ext_min_f, float(dmf_e.amin().item()))
                else:
                    cur_min_df = init_mf

                cur_min_di = ext_min_i
                if has_mi:
                    d2i_e, d4i_e = _eval_sparse_dual_fd_det(
                        u_i.reshape(-1, dim)[stencil["pool_flat"]],
                        stencil,
                        D_rev_inv,
                        eye,
                        use_physical,
                    )
                    dmi_e = torch.minimum(d2i_e, d4i_e) if include_4th else d2i_e
                    det_min_i.view(-1)[stencil["sten_flat"]] = dmi_e
                    cur_min_di = min(ext_min_i, float(dmi_e.amin().item()))
                else:
                    cur_min_di = init_mi

                if n_full <= 65_536:
                    _, _, n_fi, n_if = eval_chord_fn(u_f, u_i)
                    chord_exact = True
                else:
                    if if_ord.numel() > 0:
                        n_if.view(-1)[if_ord] = nif_best
                    if fi_ord.numel() > 0:
                        n_fi.view(-1)[fi_ord] = nfi_best
                    chord_exact = False

                if cur_min_df >= jac_fl_t and cur_min_di >= jac_fl_t:
                    sparse_ice_ok = (
                        not bool((dom_b & (n_if > strict_if_full)).any().item())
                        and not bool((dom_b & (n_fi > strict_fi_full)).any().item())
                    )
                    if sparse_ice_ok:
                        _, _, n_fi, n_if = eval_chord_fn(u_f, u_i)
                        chord_exact = True
                        if (
                            not bool((dom_b & (n_if > strict_if_full)).any().item())
                            and not bool((dom_b & (n_fi > strict_fi_full)).any().item())
                        ):
                            break
            else:
                if own_lut:
                    del lut_act
                else:
                    lut_act[act_flat] = -1
                break
        else:
            if own_lut:
                del lut_act
            else:
                lut_act[act_flat] = -1
            if not chord_exact:
                _, _, n_fi, n_if = eval_chord_fn(u_f, u_i)
                chord_exact = True
                if n_full <= 65_536 and (
                    bool((dom_b & (n_if > strict_if_full)).any().item())
                    or bool((dom_b & (n_fi > strict_fi_full)).any().item())
                ):
                    continue
            break

    if not chord_exact and steps_taken > 0:
        _, _, n_fi, n_if = eval_chord_fn(u_f, u_i)

    return u_f, u_i, det_min_f, det_min_i, n_fi, n_if, steps_taken, ever_f, ever_i


def enforce_digital_diffeomorphism_nd(
    W_fwd: torch.Tensor,
    W_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    ice_tol: float = 0.30,
    jac_floor: float = 0.05,
    jac_weight: float = 200.0,
    anchor_weight: float = 0.05,
    steps: int = 25,
    step_size: float = 0.30,
    dilation_radius: int = 2,
    lm_sync_steps: int = 25,
    stationary_boundary: bool = True,
    stats: Optional[dict] = None,
):
    """Enforce the **digital diffeomorphism** constraint on discrete displacement field(s).

    - When ``W_inv`` is given (symmetric paired mode: ``syn``, ``tvf``, ``syngs``,
      ``syn_scattered``, or ``greedy(..., return_inverse=True)``): polishes ``(W_fwd, W_inv)``
      so both fields satisfy ``min(det2, det4) >= jac_floor`` (matching both
      ``syntx.signed_jacobian_determinant`` and ``ants.create_jacobian_determinant_image``) and
      tight inverse consistency, returning ``(W_fwd_polished, W_inv_polished)``.
    - When ``W_inv is None`` (unidirectional mode: ``greedy(..., return_inverse=False)``,
      ``compose_displacement_fields``, ``integrate_momentum``, ``build_template``): unfolds
      ``W_fwd`` on the sparse active fold neighborhood under smooth 1-voxel regularization and
      proximity anchor ``anchor_weight``, returning ``W_fwd_polished``.

    Idempotent: returns inputs bitwise untouched (``stats['steps'] == 0``) when all voxels
    already satisfy the digital diffeomorphism bounds or are at a constrained fixed point.
    """
    unidirectional = W_inv is None
    channels_first = False
    if W_fwd.dim() >= 3 and W_fwd.shape[1] in (2, 3) and W_fwd.shape[-1] not in (2, 3):
        channels_first = True
        perm = (0,) + tuple(range(2, W_fwd.dim())) + (1,)
        W_fwd_in = W_fwd.permute(perm)
        W_inv_in = W_inv.permute(perm) if W_inv is not None else None
    else:
        W_fwd_in = W_fwd
        W_inv_in = W_inv

    B = W_fwd_in.shape[0]
    dim = W_fwd_in.shape[-1]
    spatial = tuple(W_fwd_in.shape[1:-1])
    device = W_fwd_in.device
    dtype = W_fwd_in.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim)
    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys requires spacing, origin, and direction")

    use_physical = X_phys is not None or (
        spacing is not None and origin is not None and direction is not None
    )
    include_4th = bool(use_physical)

    if use_physical:
        spacing_rev = [float(s) for s in reversed(tuple(spacing))]
        origin_rev = [float(o) for o in reversed(tuple(origin))]
        dir_arr = np.asarray(
            direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else direction,
            dtype=np.float64,
        )
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()

        D_inv_np = np.linalg.inv(direction_rev)
        sp_rev_np = np.asarray(spacing_rev, dtype=np.float64)
        right_mat = torch.tensor(
            np.diag(1.0 / sp_rev_np) @ D_inv_np, device=device, dtype=dtype
        )
        D_rev_inv = torch.tensor(D_inv_np, device=device, dtype=dtype)
        err_to_vox_mat = right_mat.T
        steps_fd = spacing_rev
        mean_sp = float(np.mean(spacing_rev))
        max_step_unit = float(min(spacing_rev)) * 2.0
    else:
        D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
        err_to_vox_scale = torch.tensor(
            [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)], device=device, dtype=dtype
        )
        err_to_vox_mat = torch.diag(err_to_vox_scale)
        steps_fd = [2.0 / max(s - 1, 1) for s in spatial]
        mean_sp = float(np.mean(steps_fd))
        max_step_unit = (2.0 / float(max(max(spatial) - 1, 1))) * 2.0

    D_rev_T, s_D = _d_rev_t_and_scale(D_rev_inv)

    def _eval_det_min(u_t: torch.Tensor) -> torch.Tensor:
        return _eval_dual_fd_det_min_fast(
            u_t, steps_fd, D_rev_T, s_D, include_4th, use_physical
        )

    if unidirectional:
        with torch.no_grad():
            det_f0 = _eval_det_min(W_fwd_in)
            init_min_det_f = float(det_f0.amin().item())
            if steps <= 0 or init_min_det_f >= float(jac_floor):
                if stats is not None:
                    stats["steps"] = 0
                    stats["active_voxels"] = 0
                    stats["initial_min_det_fwd"] = init_min_det_f
                    stats["final_min_det_fwd"] = init_min_det_f
                    stats["initial_max_ice_vox"] = 0.0
                    stats["final_max_ice_vox"] = 0.0
                return W_fwd

    boundary_mask = get_boundary_mask(spatial, device, dtype)
    domain_mask = (
        (boundary_mask[0, ..., 0] > 0.5)
        if stationary_boundary
        else torch.ones(spatial, dtype=torch.bool, device=device)
    )

    if not unidirectional:
        if use_physical:
            if X_phys is None:
                X_phys = get_physical_grid_torch(
                    spatial, spacing, origin, direction, device=device, dtype=dtype
                )
            X_base = X_phys
            spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
            shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
            origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
            direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
            M_norm, b_norm = get_physical_to_normalized_affine(
                shape_t, spacing_t, origin_t, direction_t
            )
            A_to_vox = torch.tensor(D_inv_np.T / sp_rev_np[None, :], device=device, dtype=dtype)
            b_to_vox = torch.tensor(
                -np.asarray(origin_rev, dtype=np.float64) @ (D_inv_np.T / sp_rev_np[None, :]),
                device=device,
                dtype=dtype,
            )
        else:
            grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
            mesh = torch.meshgrid(*grids, indexing="ij")
            X_base = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
            k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
            p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
            A_to_vox = torch.tensor(p_flip @ np.diag(k_vec), device=device, dtype=dtype)
            b_to_vox = torch.tensor(k_vec.copy(), device=device, dtype=dtype)
            right_mat = torch.tensor((p_flip @ np.diag(k_vec)).T, device=device, dtype=dtype)

    def _has_nonzero_boundary(w_t: torch.Tensor) -> bool:
        for ax in range(dim):
            if bool((w_t.select(ax + 1, 0) != 0.0).any().item()) or bool(
                (w_t.select(ax + 1, spatial[ax] - 1) != 0.0).any().item()
            ):
                return True
        return False

    def _eval_chord_errors_vox(u_f: torch.Tensor, u_i: torch.Tensor):
        u_f_cf = torch.movedim(u_f, -1, 1)
        u_i_cf = torch.movedim(u_i, -1, 1)
        if use_physical:
            y_norm = physical_to_normalized_fast(X_base + u_f, M_norm, b_norm)
            x_norm = physical_to_normalized_fast(X_base + u_i, M_norm, b_norm)
        else:
            y_norm = X_base + u_f
            x_norm = X_base + u_i
        in_y = ((y_norm >= -1.0) & (y_norm <= 1.0)).all(dim=-1) & domain_mask.unsqueeze(0)
        in_x = ((x_norm >= -1.0) & (x_norm <= 1.0)).all(dim=-1) & domain_mask.unsqueeze(0)
        r_fi = (u_f + sample_field_cf(u_i_cf, y_norm)) @ err_to_vox_mat
        r_if = (u_i + sample_field_cf(u_f_cf, x_norm)) @ err_to_vox_mat
        n_fi = torch.linalg.vector_norm(r_fi, dim=-1) * in_y.to(dtype)
        n_if = torch.linalg.vector_norm(r_if, dim=-1) * in_x.to(dtype)
        return r_fi, r_if, n_fi, n_if

    def _scatter_mapped_targets(
        src_mask: torch.Tensor,
        disp_src: torch.Tensor,
        dilate_r: int = 1,
        need_warm: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """In ``O(|src_mask|)`` time, locate target grid voxels ``round(x + disp_src(x))``
        and build an optional ``-disp_src(x)`` warm-start field at those target voxels."""
        tgt = torch.zeros_like(dom_b)
        warm = disp_src.new_zeros(disp_src.shape) if need_warm else None
        any_hit = False
        for b_i in range(B):
            s_idx = torch.nonzero(src_mask[b_i], as_tuple=False)
            if s_idx.shape[0] == 0:
                continue
            any_hit = True
            itup = tuple(s_idx[:, d] for d in range(dim))
            xs = X_base[0 if X_base.shape[0] == 1 else b_i][itup]
            us = disp_src[b_i][itup]
            yv = torch.round((xs + us) @ A_to_vox + b_to_vox).long()
            for d_ax in range(dim):
                yv[:, d_ax].clamp_(0, spatial[d_ax] - 1)
            ttup = tuple(yv[:, d] for d in range(dim))
            tgt[b_i][ttup] = True
            if warm is not None:
                warm[b_i][ttup] = -us
        if not any_hit:
            return tgt, None
        if dilate_r > 0:
            tgt = _dilate_mask_nd(tgt, dilate_r) & dom_b
        else:
            tgt = tgt & dom_b
        return tgt, warm

    with torch.no_grad():
        dom_b = domain_mask.unsqueeze(0).expand(B, *spatial)
        if not unidirectional:
            det_f0 = _eval_det_min(W_fwd_in)
            init_min_det_f = float(det_f0.amin().item())
        n_full_lut = B * max(int(np.prod(spatial)), 1)

        if unidirectional:
            lut_shared = torch.full((n_full_lut,), -1, dtype=torch.long, device=device)
            u_f, _, final_min_df, steps_taken, ever_f = _unfold_field_adaptive_smooth(
                W_fwd_in,
                det_f0,
                dom_b,
                boundary_mask,
                stationary_boundary,
                steps_fd,
                D_rev_inv,
                err_to_vox_mat,
                use_physical,
                jac_floor,
                jac_weight,
                anchor_weight,
                steps,
                step_size,
                dilation_radius,
                mean_sp,
                max_det_cap=None,
                lut_buf=lut_shared,
            )
            del lut_shared
            n_active = int(ever_f.sum().item())
            if steps_taken == 0 or n_active == 0:
                if stats is not None:
                    stats["steps"] = 0
                    stats["active_voxels"] = 0
                    stats["initial_min_det_fwd"] = init_min_det_f
                    stats["final_min_det_fwd"] = init_min_det_f
                    stats["initial_max_ice_vox"] = 0.0
                    stats["final_max_ice_vox"] = 0.0
                return W_fwd

            if stats is not None:
                stats["steps"] = steps_taken
                stats["active_voxels"] = n_active
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = final_min_df
                stats["initial_max_ice_vox"] = 0.0
                stats["final_max_ice_vox"] = 0.0
            return torch.movedim(u_f, -1, 1) if channels_first else u_f

        # Symmetric paired mode (W_inv is not None)
        _, _, n_fi0, n_if0 = _eval_chord_errors_vox(W_fwd_in, W_inv_in)
        det_i0 = _eval_det_min(W_inv_in)
        init_min_det_i = float(det_i0.amin().item())
        init_max_if = float(n_if0.amax().item())

        # Theorem 2 (Idempotency): calculate_inverse_identity_error(W_fwd, W_inv) measures n_if;
        # n_fi has O(h^2 ||nabla^2 u_i||) trilinear curvature even at the exact lattice inverse.
        has_fold = init_min_det_f < float(jac_floor) or init_min_det_i < float(jac_floor)
        has_defect = has_fold or init_max_if > float(ice_tol)
        if steps <= 0 or not has_defect:
            if stats is not None:
                stats["steps"] = 0
                stats["active_voxels"] = 0
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = init_min_det_f
                stats["initial_min_det_inv"] = init_min_det_i
                stats["final_min_det_inv"] = init_min_det_i
                stats["initial_max_ice_vox"] = init_max_if
                stats["final_max_ice_vox"] = init_max_if
            return W_fwd, W_inv

        u_f = W_fwd_in.clone()
        u_i = W_inv_in.clone()
        b_dirty_f = False
        b_dirty_i = False
        if stationary_boundary:
            b_dirty_f = _has_nonzero_boundary(W_fwd_in)
            b_dirty_i = _has_nonzero_boundary(W_inv_in)
            if b_dirty_f:
                u_f.mul_(boundary_mask)
            if b_dirty_i:
                u_i.mul_(boundary_mask)
            det_min_f = _eval_det_min(u_f) if b_dirty_f else det_f0.clone()
            det_min_i = _eval_det_min(u_i) if b_dirty_i else det_i0.clone()
        else:
            det_min_f = det_f0.clone()
            det_min_i = det_i0.clone()

        init_fold_seed = (det_min_f < float(jac_floor)) | (det_min_i < float(jac_floor))
        ever_f = (
            ((u_f - W_fwd_in).abs().amax(dim=-1) > 0.0) if b_dirty_f else torch.zeros_like(dom_b)
        )
        ever_i = (
            ((u_i - W_inv_in).abs().amax(dim=-1) > 0.0) if b_dirty_i else torch.zeros_like(dom_b)
        )
        steps_taken = 1 if (b_dirty_f or b_dirty_i) else 0
        fs_rad = 1 if n_full_lut > 65_536 else (int(dilation_radius) + 2)
        fold_shell = _dilate_mask_nd(init_fold_seed, fs_rad) & dom_b
        _, _, n_fi, n_if = (
            _eval_chord_errors_vox(u_f, u_i)
            if (b_dirty_f or b_dirty_i)
            else (None, None, n_fi0, n_if0)
        )

        lut_shared = torch.full((n_full_lut,), -1, dtype=torch.long, device=device)

        # Step 1 (Coupled Joint Symmetric Shrinking-Shell Projection on (df, di) from iteration 0,
        # unfolding u_f and u_i together while keeping inverse chord coupling active throughout):
        need_joint = (
            bool((det_min_f < float(jac_floor)).any().item())
            or bool((det_min_i < float(jac_floor)).any().item())
            or bool((dom_b & (n_if > float(ice_tol))).any().item())
            or bool((fold_shell & (n_fi > float(ice_tol))).any().item())
        )
        if need_joint:
            (
                u_f,
                u_i,
                det_min_f,
                det_min_i,
                n_fi,
                n_if,
                st_j,
                act_jf,
                act_ji,
            ) = _joint_symmetric_shell_polish(
                u_f,
                u_i,
                W_fwd_in,
                det_min_f,
                det_min_i,
                n_fi0,
                n_fi,
                n_if,
                fold_shell,
                dom_b,
                boundary_mask,
                stationary_boundary,
                steps_fd,
                D_rev_inv,
                X_base,
                A_to_vox,
                b_to_vox,
                err_to_vox_mat,
                use_physical,
                jac_floor,
                ice_tol,
                jac_weight,
                anchor_weight,
                steps,
                step_size,
                dilation_radius,
                mean_sp,
                _eval_chord_errors_vox,
                _scatter_mapped_targets,
                lut_buf=lut_shared,
            )
            if st_j > 0:
                steps_taken += st_j
                ever_f = ever_f | act_jf
                ever_i = ever_i | act_ji

        # Step 2 (Partner-coupled diffeomorphic closure on any residual pocket where joint
        # chord tension held det < jac_floor, keeping ICE coupled to partner_field):
        if bool((det_min_f < float(jac_floor)).any().item()):
            u_f, det_min_f, _, st_f2, act_f2 = _unfold_field_adaptive_smooth(
                u_f,
                det_min_f,
                dom_b,
                boundary_mask,
                stationary_boundary,
                steps_fd,
                D_rev_inv,
                err_to_vox_mat,
                use_physical,
                jac_floor,
                jac_weight,
                anchor_weight,
                steps,
                step_size,
                dilation_radius,
                mean_sp,
                max_det_cap=None,
                partner_field=u_i,
                X_base=X_base,
                A_to_vox=A_to_vox,
                b_to_vox=b_to_vox,
                ice_tol=ice_tol,
                lut_buf=lut_shared,
            )
            if st_f2 > 0:
                steps_taken += st_f2
                ever_f = ever_f | act_f2
                _, _, n_fi, n_if = _eval_chord_errors_vox(u_f, u_i)

        if bool((det_min_i < float(jac_floor)).any().item()):
            u_i, det_min_i, _, st_i2, act_i2 = _unfold_field_adaptive_smooth(
                u_i,
                det_min_i,
                dom_b,
                boundary_mask,
                stationary_boundary,
                steps_fd,
                D_rev_inv,
                err_to_vox_mat,
                use_physical,
                jac_floor,
                jac_weight,
                anchor_weight,
                steps,
                step_size,
                dilation_radius,
                mean_sp,
                max_det_cap=None,
                partner_field=u_f,
                X_base=X_base,
                A_to_vox=A_to_vox,
                b_to_vox=b_to_vox,
                ice_tol=ice_tol,
                lut_buf=lut_shared,
            )
            if st_i2 > 0:
                steps_taken += st_i2
                ever_i = ever_i | act_i2
                _, _, n_fi, n_if = _eval_chord_errors_vox(u_f, u_i)

        if int(lm_sync_steps) > 0:
            cand_ie = dom_b & (n_if > float(ice_tol))
            if bool(cand_ie.any().item()):
                u_i, det_min_i, mod_ie = _safe_lm_subset_backtrack(
                    u_f,
                    u_i,
                    cand_ie,
                    det_min_i,
                    X_base,
                    A_to_vox,
                    b_to_vox,
                    right_mat,
                    err_to_vox_mat,
                    steps_fd,
                    D_rev_inv,
                    use_physical,
                    jac_floor,
                    ice_tol,
                    lm_sync_steps,
                    max_step_unit,
                    lut_buf=lut_shared,
                )
                if bool(mod_ie.any().item()):
                    ever_i = ever_i | mod_ie
                    steps_taken += 1
                    _, _, n_fi, n_if = _eval_chord_errors_vox(u_f, u_i)

        del lut_shared

        n_active = int((ever_f | ever_i).sum().item())
        if steps_taken == 0 or n_active == 0:
            if stats is not None:
                stats["steps"] = 0
                stats["active_voxels"] = 0
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = init_min_det_f
                stats["initial_min_det_inv"] = init_min_det_i
                stats["final_min_det_inv"] = init_min_det_i
                stats["initial_max_ice_vox"] = init_max_if
                stats["final_max_ice_vox"] = init_max_if
            return W_fwd, W_inv

        # Strict bitwise preservation of exterior voxels against -0.0 float addition
        u_f = torch.where(ever_f.unsqueeze(-1), u_f, W_fwd_in)
        u_i = torch.where(ever_i.unsqueeze(-1), u_i, W_inv_in)
        final_min_df = (
            float(_eval_det_min(u_f).amin().item())
            if b_dirty_f
            else float(det_min_f.amin().item())
        )
        final_min_di = (
            float(_eval_det_min(u_i).amin().item())
            if b_dirty_i
            else float(det_min_i.amin().item())
        )
        final_max_if = float(n_if.amax().item())

        if stats is not None:
            stats["steps"] = steps_taken
            stats["active_voxels"] = n_active
            stats["initial_min_det_fwd"] = init_min_det_f
            stats["final_min_det_fwd"] = final_min_df
            stats["initial_min_det_inv"] = init_min_det_i
            stats["final_min_det_inv"] = final_min_di
            stats["initial_max_ice_vox"] = init_max_if
            stats["final_max_ice_vox"] = final_max_if

    if channels_first:
        return torch.movedim(u_f, -1, 1), torch.movedim(u_i, -1, 1)
    return u_f, u_i


def pullback_digital_diffeomorphism_to_halves(
    delta_fwd: torch.Tensor,
    w_l2r: torch.Tensor,
    w_r2l: torch.Tensor,
    w_l2r_inv: torch.Tensor,
    w_r2l_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    jac_floor: float = 0.05,
    inv_tol: float = 0.05,
    inv_steps: int = 20,
    stationary_boundary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Propagate a small endpoint correction ``delta_fwd = W_fwd_polished - W_fwd_0``
    symmetrically back onto midpoint half-fields ``(w_l2r, w_r2l, w_l2r_inv, w_r2l_inv)``.

    Operates strictly in ``O(|support(delta_fwd)|)`` sparse point-wise time: only midpoint
    voxels whose forward/inverse trajectories touch ``support(delta_fwd)`` are resampled,
    inverted via damped Levenberg-Marquardt, and checked on their 1-ring central-difference
    Jacobian stencil, leaving all exterior half-warp voxels bitwise untouched.
    """
    if not bool((delta_fwd != 0.0).any().item()):
        return w_l2r, w_r2l, w_l2r_inv, w_r2l_inv

    with torch.no_grad():
        B = w_l2r.shape[0]
        dim = w_l2r.shape[-1]
        spatial = tuple(w_l2r.shape[1:-1])
        device = w_l2r.device
        dtype = w_l2r.dtype

        if spacing is not None and origin is None:
            origin = (0.0,) * dim
        if spacing is not None and direction is None:
            direction = np.eye(dim)
        use_physical = X_phys is not None or (
            spacing is not None and origin is not None and direction is not None
        )

        boundary_mask = get_boundary_mask(spatial, device, dtype)
        domain_mask = (
            (boundary_mask[0, ..., 0] > 0.5)
            if stationary_boundary
            else torch.ones(spatial, dtype=torch.bool, device=device)
        )
        dom_b = domain_mask.unsqueeze(0).expand(B, *spatial)

        nz_mask = (delta_fwd.abs().amax(dim=-1) > 0.0) & dom_b
        if not bool(nz_mask.any().item()):
            return w_l2r, w_r2l, w_l2r_inv, w_r2l_inv

        if use_physical:
            if X_phys is None:
                X_phys = get_physical_grid_torch(
                    spatial, spacing, origin, direction, device=device, dtype=dtype
                )
            X_base = X_phys
            spacing_rev = [float(s) for s in reversed(tuple(spacing))]
            origin_rev = [float(o) for o in reversed(tuple(origin))]
            dir_arr = np.asarray(
                direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else direction,
                dtype=np.float64,
            )
            if dir_arr.ndim == 1:
                dir_arr = dir_arr.reshape(dim, dim)
            direction_rev = dir_arr[::-1, ::-1].copy()
            D_inv_np = np.linalg.inv(direction_rev)
            sp_rev_np = np.asarray(spacing_rev, dtype=np.float64)

            A_to_vox = torch.tensor(D_inv_np.T / sp_rev_np[None, :], device=device, dtype=dtype)
            b_to_vox = torch.tensor(
                -np.asarray(origin_rev, dtype=np.float64) @ (D_inv_np.T / sp_rev_np[None, :]),
                device=device,
                dtype=dtype,
            )
            right_mat = torch.tensor(
                np.diag(1.0 / sp_rev_np) @ D_inv_np, device=device, dtype=dtype
            )
            err_to_vox_mat = right_mat.T
            D_rev_inv = torch.tensor(D_inv_np, device=device, dtype=dtype)
            steps_fd = spacing_rev
            max_step_unit = float(min(spacing_rev)) * 2.0
        else:
            grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
            mesh = torch.meshgrid(*grids, indexing="ij")
            X_base = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
            k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
            p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
            A_to_vox = torch.tensor(p_flip @ np.diag(k_vec), device=device, dtype=dtype)
            b_to_vox = torch.tensor(k_vec.copy(), device=device, dtype=dtype)
            right_mat = torch.tensor((p_flip @ np.diag(k_vec)).T, device=device, dtype=dtype)
            err_to_vox_mat = torch.diag(
                torch.tensor(
                    [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
                    device=device,
                    dtype=dtype,
                )
            )
            D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
            steps_fd = [2.0 / max(s - 1, 1) for s in spatial]
            max_step_unit = (2.0 / float(max(max(spatial) - 1, 1))) * 2.0

        I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
        eye = torch.eye(dim, device=device, dtype=dtype)

        # 1. Sparse half-pullback update on w_l2r_inv strictly on nz_mask
        dw_l2r_inv_act_list = []
        nz_coords = torch.nonzero(nz_mask, as_tuple=False)
        nz_flat = _coords_to_flat_idx(nz_coords, spatial)
        for b in range(B):
            act_idx = torch.nonzero(nz_mask[b], as_tuple=False)
            if act_idx.shape[0] == 0:
                continue
            idx_t = tuple(act_idx[:, d] for d in range(dim))
            x_a = X_base[0 if X_base.shape[0] == 1 else b][idx_t]
            m_vox = (x_a + w_l2r_inv[b][idx_t]) @ A_to_vox + b_to_vox
            _, grad_r2l = _multilinear_val_and_jac_torch(w_r2l[b], m_vox, right_mat)
            J = I_d + grad_r2l
            JT = J.transpose(-1, -2)
            JTJ = JT @ J
            rhs = 0.5 * (JT @ delta_fwd[b][idx_t].unsqueeze(-1)).squeeze(-1)
            lam = torch.full((act_idx.shape[0],), 0.1, device=device, dtype=dtype)
            dw_inv = _solve_spd_lm_torch(JTJ, rhs, lam)
            dw_l2r_inv_act_list.append(dw_inv)
        dw_l2r_inv_act = (
            torch.cat(dw_l2r_inv_act_list, dim=0)
            if dw_l2r_inv_act_list
            else w_l2r_inv.new_zeros((0, dim))
        )

        # 2. Identify midpoint voxels m where x(m) = X_base(m) + w_l2r(m) lands within 2 voxels
        # of nz_mask: outside this sparse set, BOTH delta_fwd(x(m)) == 0 AND cand_l2r_inv(x(m))
        # == w_l2r_inv(x(m)) are unchanged!
        nz_guard = _dilate_mask_nd(nz_mask, 2)
        x_from_m_vox = (X_base + w_l2r) @ A_to_vox + b_to_vox
        x_from_m_idx = torch.round(x_from_m_vox).long()
        for d_ax in range(dim):
            x_from_m_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)

        m_mask = torch.zeros_like(nz_mask)
        for b in range(B):
            land_m = nz_guard[b][tuple(x_from_m_idx[b, ..., d] for d in range(dim))]
            m_mask[b] = land_m & dom_b[b]
        del x_from_m_idx, nz_guard

        # Dilate m_mask by 1 voxel so cand_l2r LM refinement covers any trajectory shifts
        l2r_refine_mask = _dilate_mask_nd(m_mask, 1) & dom_b
        r2l_mod_mask = m_mask

        # Sparse gather of delta_fwd(X(m) + w_l2r(m)) only at m_mask!
        m_coords = torch.nonzero(m_mask, as_tuple=False)
        m_flat = _coords_to_flat_idx(m_coords, spatial)
        m_batch = m_coords[:, 0]
        x_m_vox_sub = x_from_m_vox.reshape(-1, dim)[m_flat]
        del x_from_m_vox

        empty_lut = torch.empty((0,), dtype=torch.long, device=device)
        empty_delta = delta_fwd.new_zeros((0, dim))
        delta_at_m_sub = _gather_multilinear_sparse(
            delta_fwd.reshape(-1, dim), empty_lut, empty_delta, m_batch, x_m_vox_sub, spatial
        )

        # Build sparse FD stencils ONLY around the modified midpoint neighborhoods!
        has_l2r_mod = bool(l2r_refine_mask.any().item())
        has_r2l_mod = bool(r2l_mod_mask.any().item())
        has_l2r_inv_mod = bool(nz_mask.any().item())
        w_l2r_flat = w_l2r.reshape(-1, dim)
        w_r2l_flat = w_r2l.reshape(-1, dim)
        w_l2r_inv_flat = w_l2r_inv.reshape(-1, dim)

        def _eval_dual_min(u_flat_src: torch.Tensor, sten_d: dict) -> torch.Tensor:
            d2_v, d4_v = _eval_sparse_dual_fd_det(
                u_flat_src[sten_d["pool_flat"]], sten_d, D_rev_inv, eye, use_physical
            )
            return torch.minimum(d2_v, d4_v) if use_physical else d2_v

        if has_l2r_inv_mod:
            sten_l2r_inv = _build_sparse_dual_fd_stencil(nz_mask, steps_fd, dtype, include_4th=use_physical)
            det_l2r_inv_0s = _eval_dual_min(w_l2r_inv_flat, sten_l2r_inv)
            sten_dom_l2r_inv = torch.ones_like(dom_b.reshape(-1)[sten_l2r_inv["sten_flat"]])
            floor_l2r_inv = min(
                float(jac_floor),
                float(det_l2r_inv_0s.amin().item())
                if det_l2r_inv_0s.numel() > 0
                else float(jac_floor),
            )
        else:
            sten_l2r_inv = None
            sten_dom_l2r_inv = None
            floor_l2r_inv = float(jac_floor)

        if has_l2r_mod:
            sten_l2r = _build_sparse_dual_fd_stencil(l2r_refine_mask, steps_fd, dtype, include_4th=use_physical)
            det_l2r_0s = _eval_dual_min(w_l2r_flat, sten_l2r)
            # Include boundary stencil voxels so one-sided boundary Jacobians cannot flip sign
            sten_dom_l2r = torch.ones_like(dom_b.reshape(-1)[sten_l2r["sten_flat"]])
            floor_l2r = min(
                float(jac_floor),
                float(det_l2r_0s.amin().item())
                if det_l2r_0s.numel() > 0
                else float(jac_floor),
            )
        else:
            sten_l2r = None
            sten_dom_l2r = None
            floor_l2r = float(jac_floor)

        if has_r2l_mod:
            sten_r2l = _build_sparse_dual_fd_stencil(r2l_mod_mask, steps_fd, dtype, include_4th=use_physical)
            det_r2l_0s = _eval_dual_min(w_r2l_flat, sten_r2l)
            sten_dom_r2l = torch.ones_like(dom_b.reshape(-1)[sten_r2l["sten_flat"]])
            floor_r2l = min(
                float(jac_floor),
                float(det_r2l_0s.amin().item())
                if det_r2l_0s.numel() > 0
                else float(jac_floor),
            )
        else:
            sten_r2l = None
            sten_dom_r2l = None
            floor_r2l = float(jac_floor)

        best_l2r = w_l2r
        best_r2l = w_r2l
        best_l2r_inv = w_l2r_inv
        best_r2l_inv = w_r2l_inv

        for scale in (1.0, 0.5, 0.25):
            cand_l2r_inv = w_l2r_inv.clone()
            if nz_flat.numel() > 0:
                cand_l2r_inv.view(-1, dim).index_add_(0, nz_flat, scale * dw_l2r_inv_act)

            cand_r2l = w_r2l.clone()
            if m_flat.numel() > 0:
                cand_r2l.view(-1, dim).index_add_(0, m_flat, (0.5 * scale) * delta_at_m_sub)

            cand_l2r = _pointwise_lm_refine_subset(
                cand_l2r_inv,
                w_l2r,
                l2r_refine_mask,
                X_base,
                A_to_vox,
                b_to_vox,
                right_mat,
                err_to_vox_mat,
                steps=int(inv_steps),
                tol_vox=float(inv_tol),
                max_step_unit=max_step_unit,
            )

            ok_l2r = True
            if has_l2r_mod and bool(sten_dom_l2r.any().item()):
                det_l2r_c = _eval_dual_min(cand_l2r.reshape(-1, dim), sten_l2r)
                ok_l2r = float(det_l2r_c[sten_dom_l2r].amin().item()) >= floor_l2r - 1e-4

            ok_r2l = True
            if has_r2l_mod and bool(sten_dom_r2l.any().item()):
                det_r2l_c = _eval_dual_min(cand_r2l.reshape(-1, dim), sten_r2l)
                ok_r2l = float(det_r2l_c[sten_dom_r2l].amin().item()) >= floor_r2l - 1e-4

            ok_l2r_inv = True
            if has_l2r_inv_mod and bool(sten_dom_l2r_inv.any().item()):
                det_l2r_inv_c = _eval_dual_min(cand_l2r_inv.reshape(-1, dim), sten_l2r_inv)
                ok_l2r_inv = float(det_l2r_inv_c[sten_dom_l2r_inv].amin().item()) >= floor_l2r_inv - 1e-4

            if ok_l2r and ok_r2l and ok_l2r_inv:
                best_l2r = cand_l2r
                best_r2l = cand_r2l
                best_l2r_inv = cand_l2r_inv
                if w_r2l_inv is not None and has_r2l_mod:
                    r2l_guard = _dilate_mask_nd(r2l_mod_mask, 2)
                    m_from_y_vox = (X_base + w_r2l_inv) @ A_to_vox + b_to_vox
                    m_from_y_idx = torch.round(m_from_y_vox).long()
                    for d_ax in range(dim):
                        m_from_y_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)
                    r2l_inv_mask = torch.zeros_like(nz_mask)
                    for b in range(B):
                        land_y = r2l_guard[b][tuple(m_from_y_idx[b, ..., d] for d in range(dim))]
                        r2l_inv_mask[b] = land_y & dom_b[b]
                    best_r2l_inv = _pointwise_lm_refine_subset(
                        cand_r2l,
                        w_r2l_inv,
                        r2l_inv_mask,
                        X_base,
                        A_to_vox,
                        b_to_vox,
                        right_mat,
                        err_to_vox_mat,
                        steps=int(inv_steps),
                        tol_vox=float(inv_tol),
                        max_step_unit=max_step_unit,
                    )
                break

        return best_l2r, best_r2l, best_l2r_inv, best_r2l_inv
