"""Displacement-field inversion and time-varying velocity-field integration.

Ported from ``syntx.core.inverse`` (PyTorch backend only). Provides three
independent solvers for the inverse of a displacement field —
:func:`update_inverse_field_nd` dispatches between the classic ITK
fixed-point iteration, an Anderson-accelerated variant, and a damped hybrid
Levenberg-Marquardt solver — plus an ODE integrator for time-varying
velocity fields (used by TVF/SyNGS-style transforms, not the stationary
B-spline SVF already implemented in ``antstorch.bspline_flows``).
"""

import numpy as np
import torch
import torch.nn.functional as F
from ..._torch_compat import grid_sample

from .smoothing import separable_gaussian_filter, get_boundary_mask
from .grid import (
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    grid_sample_nd
)


def _multilinear_val_and_jac_torch(
    u_grid: torch.Tensor,
    vox_coords: torch.Tensor,
    right_mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact multilinear value and Jacobian of ``u_grid`` at continuous voxel coordinates.

    Matches ``F.grid_sample(..., mode='bilinear', padding_mode='border', align_corners=True)``
    for both 2-D ``(H, W, 2)`` and 3-D ``(Z, Y, X, 3)`` fields. Cell derivatives are zeroed
    along any axis clamped at the domain border and mapped into target coordinate space via
    right-multiplication by ``right_mat`` (``du / dy = (du / dvox) @ right_mat``).
    """
    dim = u_grid.shape[-1]
    dtype = u_grid.dtype
    if dim == 2:
        H, W, _ = u_grid.shape
        vy, vx = vox_coords[:, 0], vox_coords[:, 1]
        in_y = ((vy >= 0.0) & (vy <= float(H - 1))).to(dtype)
        in_x = ((vx >= 0.0) & (vx <= float(W - 1))).to(dtype)
        vy_c = vy.clamp(0.0, float(max(H - 1, 0)))
        vx_c = vx.clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy_c).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx_c).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = (vy_c - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx_c - ix0.to(dtype)).unsqueeze(-1)
        omy, omx = 1.0 - ty, 1.0 - tx

        flat = u_grid.reshape(-1, 2)
        c00 = flat[iy0 * W + ix0]
        c01 = flat[iy0 * W + ix1]
        c10 = flat[iy1 * W + ix0]
        c11 = flat[iy1 * W + ix1]

        c0 = c00 * omx + c01 * tx
        c1 = c10 * omx + c11 * tx
        u_val = c0 * omy + c1 * ty

        du_dvy = (c1 - c0) * in_y.unsqueeze(-1)
        du_dvx = ((c01 - c00) * omy + (c11 - c10) * ty) * in_x.unsqueeze(-1)
        du_dvox = torch.stack([du_dvy, du_dvx], dim=-1)
        return u_val, du_dvox @ right_mat

    Z, Y, X, _ = u_grid.shape
    vz, vy, vx = vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]
    in_z = ((vz >= 0.0) & (vz <= float(Z - 1))).to(dtype)
    in_y = ((vy >= 0.0) & (vy <= float(Y - 1))).to(dtype)
    in_x = ((vx >= 0.0) & (vx <= float(X - 1))).to(dtype)
    vz_c = vz.clamp(0.0, float(max(Z - 1, 0)))
    vy_c = vy.clamp(0.0, float(max(Y - 1, 0)))
    vx_c = vx.clamp(0.0, float(max(X - 1, 0)))
    iz0 = torch.floor(vz_c).long().clamp(0, max(Z - 2, 0))
    iy0 = torch.floor(vy_c).long().clamp(0, max(Y - 2, 0))
    ix0 = torch.floor(vx_c).long().clamp(0, max(X - 2, 0))
    iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
    iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
    ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
    tz = (vz_c - iz0.to(dtype)).unsqueeze(-1)
    ty = (vy_c - iy0.to(dtype)).unsqueeze(-1)
    tx = (vx_c - ix0.to(dtype)).unsqueeze(-1)
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx

    flat = u_grid.reshape(-1, 3)
    sz, sy = Y * X, X
    c000 = flat[iz0 * sz + iy0 * sy + ix0]
    c001 = flat[iz0 * sz + iy0 * sy + ix1]
    c010 = flat[iz0 * sz + iy1 * sy + ix0]
    c011 = flat[iz0 * sz + iy1 * sy + ix1]
    c100 = flat[iz1 * sz + iy0 * sy + ix0]
    c101 = flat[iz1 * sz + iy0 * sy + ix1]
    c110 = flat[iz1 * sz + iy1 * sy + ix0]
    c111 = flat[iz1 * sz + iy1 * sy + ix1]

    c00 = c000 * omx + c001 * tx
    c01 = c010 * omx + c011 * tx
    c10 = c100 * omx + c101 * tx
    c11 = c110 * omx + c111 * tx
    c0 = c00 * omy + c01 * ty
    c1 = c10 * omy + c11 * ty
    u_val = c0 * omz + c1 * tz

    du_dvz = (c1 - c0) * in_z.unsqueeze(-1)
    du_dvy = ((c01 - c00) * omz + (c11 - c10) * tz) * in_y.unsqueeze(-1)
    du_dvx = (
        ((c001 - c000) * omy + (c011 - c010) * ty) * omz
        + ((c101 - c100) * omy + (c111 - c110) * ty) * tz
    ) * in_x.unsqueeze(-1)
    du_dvox = torch.stack([du_dvz, du_dvy, du_dvx], dim=-1)
    return u_val, du_dvox @ right_mat


def _solve_spd_lm_torch(
    JTJ: torch.Tensor,
    rhs: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Closed-form Jacobi-equilibrated Cholesky solve of ``(JTJ + mu * I) dv = rhs`` for 2-D/3-D."""
    dim = rhs.shape[-1]
    diag = JTJ.diagonal(dim1=-2, dim2=-1)
    diag_scale = diag.amax(dim=-1).clamp(min=1.0)
    mu = (lam * diag_scale).clamp(min=1e-6)
    h_diag = (diag + mu.unsqueeze(-1)).clamp(min=1e-12)
    s = h_diag.sqrt()

    if dim == 2:
        s0, s1 = s[:, 0], s[:, 1]
        h10 = JTJ[:, 1, 0] / (s1 * s0)
        b0 = rhs[:, 0] / s0
        b1 = rhs[:, 1] / s1
        l11 = (1.0 - h10 * h10).clamp(min=1e-7).sqrt()
        y0 = b0
        y1 = (b1 - h10 * y0) / l11
        z1 = y1 / l11
        z0 = y0 - h10 * z1
        return torch.stack([z0 / s0, z1 / s1], dim=-1)

    s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]
    h10 = JTJ[:, 1, 0] / (s1 * s0)
    h20 = JTJ[:, 2, 0] / (s2 * s0)
    h21 = JTJ[:, 2, 1] / (s2 * s1)
    b0 = rhs[:, 0] / s0
    b1 = rhs[:, 1] / s1
    b2 = rhs[:, 2] / s2
    l11 = (1.0 - h10 * h10).clamp(min=1e-7).sqrt()
    l21 = (h21 - h20 * h10) / l11
    l22 = (1.0 - h20 * h20 - l21 * l21).clamp(min=1e-7).sqrt()
    y0 = b0
    y1 = (b1 - h10 * y0) / l11
    y2 = (b2 - h20 * y0 - l21 * y1) / l22
    z2 = y2 / l22
    z1 = (y1 - l21 * z2) / l11
    z0 = y0 - h10 * z1 - h20 * z2
    return torch.stack([z0 / s0, z1 / s1, z2 / s2], dim=-1)


def update_inverse_field_nd_hybrid_lm(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    damping_factor: float = 1.0,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
    stationary_boundary: bool = True,
    stats: dict = None,
) -> torch.Tensor:
    """
    Inverse of a displacement field by safeguarded per-voxel Levenberg-Marquardt steps.

    Port of ``syntx.core.inverse.update_inverse_field_nd_hybrid_lm`` as of syntx v6.39.0
    (rewritten upstream in R32, 2026-11-11), replacing the earlier damped hybrid solver.

    Minimises ``0.5 * ||v(x) + u(x + v(x))||^2`` per voxel using the exact analytical
    multilinear Jacobian ``J(y) = I + grad_y u(y)`` evaluated at ``y = x + v(x)`` (respecting
    full ``spacing``, ``origin`` and ``direction`` cosines in physical mode, or normalised
    ``[-1, 1]`` coordinates when ``spacing`` is omitted). Each step solves the SPD normal
    equations ``(J^T J + mu I) dv = -J^T r``, caps ``||dv||`` to a trust-region radius, and
    accepts ``v + dv`` per voxel strictly when ``||r(v + dv)|| < ||r(v)||`` while adapting the
    damping parameter ``lambda``.

    When ``smoothing_sigma == 0``, only unconverged interior voxels are gathered into an
    active set so warm-started polishing runs in sparse ``O(N_active)`` time.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = tuple(W_disp.shape[1:-1])
    device = W_disp.device
    dtype = W_disp.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()
    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys needs the spacing / origin / direction it was built from")

    use_physical = (
        X_phys is not None
        or (spacing is not None and origin is not None and direction is not None)
    )
    boundary_mask = get_boundary_mask(spatial, device, dtype)
    W_disp_cf = torch.movedim(W_disp, -1, 1)

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(
                spatial, spacing, origin, direction, device=device, dtype=dtype
            )
        X_base = X_phys
        spacing_rev = tuple(float(s) for s in reversed(spacing))
        origin_rev = tuple(float(o) for o in reversed(origin))
        dir_arr = np.asarray(direction, dtype=np.float64)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()

        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)

        D_inv_np = np.linalg.inv(direction_rev)
        sp_rev_np = np.asarray(spacing_rev, dtype=np.float64)
        A_to_vox_np = D_inv_np.T / sp_rev_np[None, :]
        b_to_vox_np = -np.asarray(origin_rev, dtype=np.float64) @ A_to_vox_np
        right_mat_np = np.diag(1.0 / sp_rev_np) @ D_inv_np

        A_to_vox = torch.tensor(A_to_vox_np, device=device, dtype=dtype)
        b_to_vox = torch.tensor(b_to_vox_np, device=device, dtype=dtype)
        right_mat = torch.tensor(right_mat_np, device=device, dtype=dtype)
        err_to_vox_scale = 1.0 / spacing_t
        max_step_unit = float(min(spacing_rev)) * 2.0 * max(float(relaxation), 1e-3)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        X_base = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

        k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
        p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
        A_to_vox_np = p_flip @ np.diag(k_vec)
        b_to_vox_np = k_vec.copy()
        right_mat_np = A_to_vox_np.T

        A_to_vox = torch.tensor(A_to_vox_np, device=device, dtype=dtype)
        b_to_vox = torch.tensor(b_to_vox_np, device=device, dtype=dtype)
        right_mat = torch.tensor(right_mat_np, device=device, dtype=dtype)
        err_to_vox_scale = torch.tensor(
            [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
            device=device,
            dtype=dtype,
        )
        max_step_unit = (2.0 / float(max(max(spatial) - 1, 1))) * 2.0 * max(float(relaxation), 1e-3)

    def _eval_full_error(v_field: torch.Tensor) -> torch.Tensor:
        if use_physical:
            c_norm = physical_to_normalized_torch_cached(X_base + v_field, shape_t, spacing_t, origin_t, direction_t)
        else:
            c_norm = X_base + v_field
        sampled = grid_sample(W_disp_cf, c_norm, mode='bilinear', padding_mode='border', align_corners=True)
        return torch.movedim(sampled, 1, -1).add_(v_field)

    if W_inv_disp is None:
        v = torch.zeros_like(W_disp)
        best_r_sq = ((W_disp * err_to_vox_scale) ** 2).sum(dim=-1, keepdim=True)
        for alpha in (0.25, 0.5, 1.0):
            v_cand = W_disp * (-alpha)
            if stationary_boundary:
                v_cand.mul_(boundary_mask)
            err_cand = _eval_full_error(v_cand)
            r_cand_sq = ((err_cand * err_to_vox_scale) ** 2).sum(dim=-1, keepdim=True)
            better = r_cand_sq < best_r_sq
            v = torch.where(better, v_cand, v)
            best_r_sq = torch.where(better, r_cand_sq, best_r_sq)
        del v_cand, err_cand, r_cand_sq, best_r_sq, better
    else:
        v = W_inv_disp.clone()
        if stationary_boundary:
            v.mul_(boundary_mask)

    if steps <= 0:
        if stats is not None:
            stats['steps'] = 0
            stats['converged'] = False
        return torch.movedim(v, -1, 1) if channels_first else v

    domain_mask = (
        (boundary_mask[0, ..., 0] > 0.5)
        if stationary_boundary
        else torch.ones(spatial, dtype=torch.bool, device=device)
    )
    n_domain = max(int(domain_mask.sum().item()), 1)
    lam_init = 1e-3 * max(float(damping_factor), 1e-6)
    step_scale = float(relaxation) if float(relaxation) < 1.0 else 1.0
    I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)

    if smoothing_sigma > 0.0:
        active_idx_all = torch.nonzero(domain_mask, as_tuple=False)
        steps_taken = steps
        converged = False
        lam_dense = torch.full(
            (B, active_idx_all.shape[0]), lam_init, device=device, dtype=dtype
        )
        for iteration in range(steps):
            err_full = _eval_full_error(v)
            vox_norm = ((err_full * err_to_vox_scale) ** 2).sum(dim=-1).sqrt_()
            vox_norm_dom = vox_norm[:, domain_mask]
            max_err = float(vox_norm_dom.amax().item()) if vox_norm_dom.numel() > 0 else 0.0
            mean_err = float(vox_norm_dom.mean().item()) if vox_norm_dom.numel() > 0 else 0.0
            if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
                converged = True
                steps_taken = iteration
                break
            for b in range(B):
                if active_idx_all.shape[0] == 0:
                    continue
                u_grid = W_disp[b]
                x_grid_b = X_base[0 if X_base.shape[0] == 1 else b]
                idx_tuple = tuple(active_idx_all[:, d] for d in range(dim))
                x_a = x_grid_b[idx_tuple]
                v_a = v[b][idx_tuple]
                y_vox = (x_a + v_a) @ A_to_vox + b_to_vox
                u_a, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
                r_a = v_a + u_a
                r_vox_sq_a = ((r_a * err_to_vox_scale) ** 2).sum(dim=-1)
                J = I_d + grad_u_a
                JT = J.transpose(-1, -2)
                JTJ = JT @ J
                rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
                dv = _solve_spd_lm_torch(JTJ, rhs, lam_dense[b]) * step_scale
                dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
                dv = torch.where(
                    dv_norm > max_step_unit,
                    dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                    dv,
                )
                v_trial = v_a + dv
                y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
                u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
                r_t = v_trial + u_t
                r_vox_sq_t = ((r_t * err_to_vox_scale) ** 2).sum(dim=-1)
                imp = r_vox_sq_t < r_vox_sq_a
                v[b][idx_tuple] = torch.where(imp.unsqueeze(-1), v_trial, v_a)
                lam_dense[b] = torch.where(
                    imp,
                    (lam_dense[b] * 0.25).clamp(min=1e-6),
                    (lam_dense[b] * 8.0).clamp(max=1e8),
                )
            v = separable_gaussian_filter(v, smoothing_sigma)
            if stationary_boundary:
                v.mul_(boundary_mask)
        if stats is not None:
            stats['steps'] = steps_taken
            stats['converged'] = converged
        return torch.movedim(v, -1, 1) if channels_first else v

    # Sparse active-set safeguarded Levenberg-Marquardt (smoothing_sigma == 0)
    err_init = _eval_full_error(v)
    vox_err_sq = ((err_init * err_to_vox_scale) ** 2).sum(dim=-1)
    del err_init

    max_steps_used = 0
    all_converged = True
    max_chunk = 2_000_000

    for b in range(B):
        vox_sq_dom = vox_err_sq[b][domain_mask]
        if vox_sq_dom.numel() == 0:
            continue
        vox_err_dom = vox_sq_dom.sqrt()
        max_err_b = float(vox_err_dom.amax().item())
        if vox_err_dom.device.type == "mps":
            acc = 0.0
            for _s in range(0, vox_err_dom.numel(), 4_194_304):
                acc += float(vox_err_dom[_s : _s + 4_194_304].sum().item())
            mean_err_b = acc / n_domain
        else:
            mean_err_b = float(vox_err_dom.sum(dtype=torch.float64).item() / n_domain)
        del vox_err_dom
        if max_err_b <= max_error_threshold and mean_err_b <= mean_error_threshold:
            continue

        if mean_err_b <= mean_error_threshold:
            active_tol_vox = float(max_error_threshold)
        else:
            active_tol_vox = float(
                min(max_error_threshold, max(mean_error_threshold * 0.8, 1e-6))
            )
        target_vox_sq = max((0.25 * active_tol_vox) ** 2, 1e-12)

        active_idx_full = torch.nonzero(
            (vox_err_sq[b] > (active_tol_vox ** 2)) & domain_mask,
            as_tuple=False,
        )
        n_act_total = active_idx_full.shape[0]
        if n_act_total == 0:
            continue

        u_grid = W_disp[b]
        x_grid_b = X_base[0 if X_base.shape[0] == 1 else b]
        b_steps_used = 0
        b_any_unconverged = False

        for c_start in range(0, n_act_total, max_chunk):
            active_idx = active_idx_full[c_start : c_start + max_chunk]
            n_act = active_idx.shape[0]
            idx_tuple = tuple(active_idx[:, d] for d in range(dim))
            x_act = x_grid_b[idx_tuple]
            v_act = v[b][idx_tuple]

            y_vox = (x_act + v_act) @ A_to_vox + b_to_vox
            u_val, _ = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
            r = v_act + u_val
            r_vox_sq = ((r * err_to_vox_scale) ** 2).sum(dim=-1)
            lam = torch.full((n_act,), lam_init, device=device, dtype=dtype)
            alive = torch.arange(n_act, device=device)

            chunk_steps = 0
            for iteration in range(steps):
                alive = alive[(r_vox_sq[alive] > target_vox_sq) & (lam[alive] < 1e7)]
                if alive.numel() == 0:
                    break
                chunk_steps = iteration + 1
                x_a = x_act[alive]
                v_a = v_act[alive]
                r_a = r[alive]
                r_vox_sq_a = r_vox_sq[alive]
                lam_a = lam[alive]

                y_vox_a = (x_a + v_a) @ A_to_vox + b_to_vox
                _, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox_a, right_mat)
                J = I_d + grad_u_a
                JT = J.transpose(-1, -2)
                JTJ = JT @ J
                rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
                dv = _solve_spd_lm_torch(JTJ, rhs, lam_a) * step_scale
                dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
                dv = torch.where(
                    dv_norm > max_step_unit,
                    dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                    dv,
                )

                v_trial = v_a + dv
                y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
                u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
                r_t = v_trial + u_t
                r_vox_sq_t = ((r_t * err_to_vox_scale) ** 2).sum(dim=-1)

                imp = r_vox_sq_t < r_vox_sq_a
                v_act[alive] = torch.where(imp.unsqueeze(-1), v_trial, v_a)
                r[alive] = torch.where(imp.unsqueeze(-1), r_t, r_a)
                r_vox_sq[alive] = torch.where(imp, r_vox_sq_t, r_vox_sq_a)
                lam[alive] = torch.where(
                    imp,
                    (lam_a * 0.25).clamp(min=1e-6),
                    (lam_a * 8.0).clamp(max=1e8),
                )

            v[b][idx_tuple] = v_act
            b_steps_used = max(b_steps_used, chunk_steps)
            if bool((r_vox_sq > (max_error_threshold ** 2)).any().item()):
                b_any_unconverged = True

        max_steps_used = max(max_steps_used, b_steps_used)
        if b_any_unconverged:
            all_converged = False

    if stationary_boundary:
        v.mul_(boundary_mask)

    if stats is not None:
        stats['steps'] = max_steps_used
        stats['converged'] = all_converged

    return torch.movedim(v, -1, 1) if channels_first else v


def integrate_time_varying_velocity_field(
    velocity_fields,
    dt: float = 0.25,
    mode: str = 'forward',
    solver: str = 'rk4',
    spacing=None,
    origin=None,
    direction=None
):
    """Integrate a discretized time-varying velocity field sequence ``v(x, t)``.

    Not used by the stationary B-spline SVF transform in
    ``antstorch.bspline_flows`` — this is the building block for
    genuinely time-varying transforms (TVF, geodesic shooting), kept in
    ``core`` alongside the rest of the shared field-integration machinery.

    Parameters
    ----------
    velocity_fields : torch.Tensor or sequence of torch.Tensor
        Either a tensor of shape ``(T, B, *spatial, d)`` or a length-``T``
        sequence of ``(B, *spatial, d)`` tensors, one per discrete time step.
    dt : float
        Integration time step size.
    mode : {'forward', 'backward'}
        Integrate from ``t=0`` to ``t=1`` (``'forward'``) or ``t=1`` to
        ``t=0`` (``'backward'``).
    solver : {'rk4', 'midpoint', 'euler'}
        ODE integration scheme.
    spacing, origin, direction : sequence/array-like, optional
        Physical-space metadata. If any is omitted, the integration runs in
        normalized ``[-1, 1]`` coordinate space instead.

    Returns
    -------
    torch.Tensor
        The composite deformation field (as a displacement from identity),
        shape matching one entry of ``velocity_fields``.
    """
    if isinstance(velocity_fields, torch.Tensor) and velocity_fields.ndim == 5:
        T = velocity_fields.shape[0]
        vel_list = [velocity_fields[i] for i in range(T)]
    else:
        vel_list = list(velocity_fields)
        T = len(vel_list)

    B = vel_list[0].shape[0]
    dim = vel_list[0].shape[-1]
    spatial = vel_list[0].shape[1:-1]
    device = vel_list[0].device
    dtype = vel_list[0].dtype

    if spacing is not None and origin is not None and direction is not None:
        X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        direction_rev = np.asarray(direction)[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)

        phi = torch.zeros_like(vel_list[0])

        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0

        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)

            def eval_v(curr_phi):
                coords_phys = X_phys + curr_phi
                coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
                return torch.movedim(
                    grid_sample(v_k_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
                )

            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == 'midpoint':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                phi = phi + (sign * dt) * k2
            else:  # euler
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1

        return phi
    else:
        grids = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in spatial]
        identity = torch.stack(
            torch.meshgrid(*reversed(grids), indexing='ij')[::-1], dim=-1
        ).unsqueeze(0).expand(B, *spatial, dim)

        phi = torch.zeros_like(vel_list[0])
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0

        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)

            def eval_v(curr_phi):
                sample_coords = identity + curr_phi
                return torch.movedim(
                    grid_sample(v_k_cf, sample_coords, padding_mode='border', align_corners=True), 1, -1
                )

            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1

        return phi


def update_inverse_field_nd_anderson(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor,
    steps: int = 30,
    m: int = 5,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None
) -> torch.Tensor:
    """Anderson-accelerated fixed-point inversion of a displacement field.

    Wraps ITK's standard fixed-point iteration ``g(v)`` with Type-I Anderson
    acceleration (window size ``m``): at each step, a small
    ``(m_k + 1)``-dimensional least-squares problem is solved over the
    sliding window of recent residuals to find an extrapolated iterate,
    typically converging superlinearly rather than the plain fixed point's
    linear rate.

    Parameters
    ----------
    W_disp : torch.Tensor
        Forward displacement field of shape ``(B, *spatial, dim)``.
    W_inv_disp : torch.Tensor, optional
        Initial guess for the inverse field; defaults to ``-W_disp``.
    steps : int
        Maximum number of solver iterations.
    m : int
        Anderson acceleration window size.
    smoothing_sigma : float
        Smoothing sigma applied to each candidate update; ``0`` disables it.
    max_error_threshold, mean_error_threshold : float
        Convergence thresholds on the fixed-point residual's max/mean norm.
    spacing, origin, direction : sequence/array-like, optional
        Physical-space metadata; falls back to normalized ``[-1, 1]``
        coordinate space if any is missing.
    X_phys : torch.Tensor, optional
        Precomputed physical-coordinate grid.

    Returns
    -------
    torch.Tensor
        The estimated inverse displacement field, same shape as ``W_disp``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in (2, 3) and W_disp.shape[-1] not in (2, 3):
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in (2, 3):
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()

    use_physical = (X_phys is not None or
                    (spacing is not None and origin is not None and direction is not None))

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        W_disp_cf = torch.movedim(W_disp, -1, 1)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    def itk_fixed_point_step(v_curr, iteration):
        if use_physical:
            coords_phys = X_phys + v_curr
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t) ** 2, dim=-1, keepdim=True))
        else:
            coords = identity + v_curr
            forward_at_inv = torch.movedim(
                grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale) ** 2, dim=-1, keepdim=True))

        max_error_norm = float(scaled_norm.max())
        mean_error_norm = float(scaled_norm.mean())
        # Squared scaled residual norm at v_curr (syntx >= 6.35.4: the reference the Anderson
        # candidate is compared with).
        residual_fp_sq = float(torch.sum(scaled_norm ** 2))

        epsilon = 0.75 if iteration == 0 else 0.5
        update = -error
        clip_threshold = epsilon * max_error_norm
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            torch.ones_like(scaled_norm)
        )
        update = update * clip_scale
        v_new = v_curr + update * epsilon

        if smoothing_sigma > 0.0:
            if use_physical:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma, spacing=spacing)
            else:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma)

        v_new = v_new * boundary_mask
        return v_new, max_error_norm, mean_error_norm, residual_fp_sq

    v_k = W_inv_disp.clone()
    # Residual / fixed-point histories live in two contiguous (window, N)
    # buffers used as a ring (``order`` lists their rows oldest -> newest).
    # The difference columns r_newest - r_j and g_newest - g_j are formed
    # one at a time when needed instead of being materialized together, so
    # beyond the 2 * (m + 1) history rows at most two extra full-size
    # vectors are alive. Same arithmetic, in the same order, as building
    # the columns up front.
    window = max(1, min(int(steps), m + 1))
    R_buf = G_buf = None
    order = []

    for iteration in range(steps):
        g_k, max_err, mean_err, residual_fp_sq = itk_fixed_point_step(v_k, iteration)

        # Idempotent exit (syntx >= 6.35.4): when v_k already meets the thresholds it is
        # returned unchanged (no extra, unverified fixed-point step).
        if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
            break

        if R_buf is None:
            R_buf = torch.empty((window, g_k.numel()), device=g_k.device, dtype=g_k.dtype)
            G_buf = torch.empty_like(R_buf)
        slot = order.pop(0) if len(order) == window else len(order)
        order.append(slot)
        r_k = R_buf[slot]
        torch.sub(g_k.reshape(-1), v_k.reshape(-1), out=r_k)
        G_buf[slot].copy_(g_k.reshape(-1))

        m_k = len(order)

        if m_k < 2:
            v_k = g_k
        else:
            n_cols = m_k - 1
            r_newest = R_buf[order[-1]]
            g_newest = G_buf[order[-1]]

            gram = torch.zeros(n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(n_cols, device=device, dtype=dtype)
            for i_col in range(n_cols):
                dR_i = r_newest - R_buf[order[i_col]]
                rhs[i_col] = torch.dot(dR_i, r_newest)
                gram[i_col, i_col] = torch.dot(dR_i, dR_i)
                for j_col in range(i_col + 1, n_cols):
                    dR_j = r_newest - R_buf[order[j_col]]
                    val = torch.dot(dR_i, dR_j)
                    gram[i_col, j_col] = val
                    gram[j_col, i_col] = val
                    del dR_j
                del dR_i

            gram += 1e-10 * torch.eye(n_cols, device=device, dtype=dtype)

            try:
                gamma = torch.linalg.solve(gram, rhs)
            except torch.linalg.LinAlgError:
                v_k = g_k
                continue

            v_new_flat = g_newest.clone()
            for j in range(n_cols):
                column = g_newest - G_buf[order[j]]
                column += r_newest - R_buf[order[j]]
                v_new_flat.sub_(gamma[j] * column)
                del column

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate = v_candidate * boundary_mask
            del v_new_flat

            if use_physical:
                coords_norm_c = physical_to_normalized_torch_cached(X_phys + v_candidate, shape_t, spacing_t, origin_t, direction_t)
                error_c = torch.movedim(
                    grid_sample(W_disp_cf, coords_norm_c, padding_mode='border', align_corners=True), 1, -1
                )
                del coords_norm_c
                error_c += v_candidate
                residual_aa_sq = float(torch.sum((error_c / spacing_t) ** 2))
            else:
                error_c = torch.movedim(
                    grid_sample(W_disp_cf, identity + v_candidate, padding_mode='border', align_corners=True), 1, -1
                )
                error_c += v_candidate
                residual_aa_sq = float(torch.sum((error_c * voxel_scale) ** 2))
            del error_c

            # Accept the Anderson-extrapolated candidate unless its scaled residual is more than
            # 1.1x that of the current iterate v_k (syntx: squared norms, factor 1.1**2). Earlier
            # versions compared with the unscaled fixed-point step norm ||g_k - v_k||.
            if residual_aa_sq <= residual_fp_sq * 1.21:
                v_k = v_candidate
            else:
                v_k = g_k
            del v_candidate

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


def update_inverse_field_nd(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    method: str = 'anderson',
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
    max_iters=None,
    convergence_criterion=None,
    **kwargs
) -> torch.Tensor:
    """Dimension-agnostic inversion of a displacement field.

    Dispatches to one of three solvers, all converging to the same fixed
    point (matching ITK's ``itkInvertDisplacementFieldImageFilter``) but
    with different convergence behavior:

    - ``'fixed_point'`` (or any value not below): the classic ITK
      damped fixed-point iteration.
    - ``'anderson'`` (default): :func:`update_inverse_field_nd_anderson`,
      typically fastest to converge.
    - ``'hybrid_lm'``: :func:`update_inverse_field_nd_hybrid_lm`, most
      robust under large local deformations.

    Parameters
    ----------
    W_disp : torch.Tensor
        Forward displacement field of shape ``(B, *spatial, dim)``.
    W_inv_disp : torch.Tensor, optional
        Initial guess for the inverse field; defaults to ``-W_disp``.
    steps : int
        Maximum number of solver iterations.
    relaxation : float
        Relaxation factor (fixed-point solver only).
    smoothing_sigma : float
        Smoothing sigma applied to intermediate updates; ``0`` disables it.
    method : {'anderson', 'hybrid_lm', 'fixed_point'}
        Which solver to use.
    max_error_threshold, mean_error_threshold : float
        Convergence thresholds on the fixed-point residual's max/mean norm.
    spacing, origin, direction : sequence/array-like, optional
        Physical-space metadata; falls back to normalized ``[-1, 1]``
        coordinate space if any is missing.
    X_phys : torch.Tensor, optional
        Precomputed physical-coordinate grid.
    max_iters : int, optional
        Alias for ``steps`` (accepted for interface parity with ``syntx``);
        overrides ``steps`` when given.

    convergence_criterion : {None, 'both', 'either'}
        Fixed-point stopping rule. 'either' matches ITK: stop when either
        tolerance is satisfied. None preserves historical defaults ('both'
        for physical coordinates, 'either' for normalized coordinates).
        Explicit values require method='fixed_point'.

    Returns
    -------
    torch.Tensor
        The estimated inverse displacement field, same shape as ``W_disp``.
    """
    if convergence_criterion not in (None, "both", "either"):
        raise ValueError("convergence_criterion must be None, 'both', or 'either'")
    if convergence_criterion is not None and method != "fixed_point":
        raise ValueError("convergence_criterion requires method='fixed_point'")

    def converged(maximum, mean, default):
        checks = (maximum <= max_error_threshold, mean <= mean_error_threshold)
        return any(checks) if (convergence_criterion or default) == "either" else all(checks)

    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in (2, 3) and W_disp.shape[-1] not in (2, 3):
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in (2, 3):
            W_inv_disp = W_inv_disp.permute(perm)

    if max_iters is not None:
        steps = max_iters

    if method == 'hybrid_lm':
        res = update_inverse_field_nd_hybrid_lm(
            W_disp, W_inv_disp, steps=steps, relaxation=relaxation,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    if method == 'anderson':
        res = update_inverse_field_nd_anderson(
            W_disp, W_inv_disp, steps=steps,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    if W_inv_disp is None:  # syntx: first-order guess -u, as the other solvers
        W_inv_disp = -W_disp.clone()

    if X_phys is not None or (spacing is not None and origin is not None and direction is not None):
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        W_disp_cf = torch.movedim(W_disp, -1, 1)

        max_error_norm = float('inf')
        mean_error_norm = float('inf')

        for iteration in range(steps):

            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t) ** 2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())
            # Idempotent (syntx >= 6.35.4): test the current iterate before updating it.
            if converged(max_error_norm, mean_error_norm, "both"):
                break

            update = -error
            epsilon = 0.75 if iteration == 0 else 0.5
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = update * clip_scale
            W_inv_disp = W_inv_disp + update * epsilon

            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma, spacing=spacing)

            W_inv_disp = W_inv_disp * boundary_mask

        return torch.movedim(W_inv_disp, -1, 1) if channels_first else W_inv_disp
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)

        max_error_norm = float('inf')
        mean_error_norm = float('inf')

        for iteration in range(steps):

            coords = identity + W_inv_disp
            forward_at_inv = torch.movedim(
                grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale) ** 2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())
            # Idempotent (syntx >= 6.35.4): test the current iterate before updating it.
            if converged(max_error_norm, mean_error_norm, "either"):
                break

            epsilon = 0.75 if iteration == 0 else 0.5
            update = -error
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = update * clip_scale
            W_inv_disp = W_inv_disp + update * epsilon

            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma)

            W_inv_disp = W_inv_disp * boundary_mask

        return torch.movedim(W_inv_disp, -1, 1) if channels_first else W_inv_disp


def compute_inverse_identity_error_nd(
    warp_fwd: torch.Tensor,
    warp_inv: torch.Tensor,
    spacing=None,
    origin=None,
    direction=None,
    is_displacement: bool = True,
    fwd_is_disp: bool = None,
    inv_is_disp: bool = None
) -> torch.Tensor:
    """Compute the composed forward-then-inverse identity-error map.

    ``Error(x) = || disp_fwd(x) + disp_inv(x + disp_fwd(x)) ||``, evaluated
    in physical units (mm).

    Parameters
    ----------
    warp_fwd, warp_inv : torch.Tensor
        Forward/inverse fields of shape ``(B, *spatial, dim)`` or
        ``(*spatial, dim)`` (unsqueezed to add a batch dimension).
    spacing, origin, direction : sequence/array-like, optional
        Physical-space metadata; default to unit spacing, zero origin, and
        the identity direction.
    is_displacement : bool
        Default interpretation (displacement vs. absolute deformation) for
        both fields, overridden per-field by ``fwd_is_disp``/``inv_is_disp``.
    fwd_is_disp, inv_is_disp : bool, optional
        Per-field override of ``is_displacement``.

    Returns
    -------
    torch.Tensor
        Error-norm map of shape ``(B, *spatial)``, in physical units.
    """
    dim = warp_fwd.shape[-1]
    if warp_fwd.dim() == dim + 1:
        warp_fwd = warp_fwd.unsqueeze(0)
    if warp_inv.dim() == dim + 1:
        warp_inv = warp_inv.unsqueeze(0)

    spatial = warp_fwd.shape[1:-1]
    device = warp_fwd.device
    dtype = warp_fwd.dtype

    if spacing is None:
        spacing = [1.0] * dim
    if origin is None:
        origin = [0.0] * dim
    if direction is None:
        direction = np.eye(dim)
    else:
        direction = np.asarray(direction)[:dim, :dim]

    spatial_tuple = tuple(spatial)
    spacing_tuple = tuple(float(s) for s in spacing)
    origin_tuple = tuple(float(o) for o in origin)

    X_phys = get_physical_grid_torch(spatial_tuple, spacing_tuple, origin_tuple, direction, device=device, dtype=dtype)

    use_fwd_disp = is_displacement if fwd_is_disp is None else fwd_is_disp
    use_inv_disp = is_displacement if inv_is_disp is None else inv_is_disp

    disp_fwd = warp_fwd if use_fwd_disp else (warp_fwd - X_phys)
    disp_inv = warp_inv if use_inv_disp else (warp_inv - X_phys)

    y_norm = physical_to_normalized_torch(X_phys + disp_fwd, spatial_tuple, spacing_tuple, origin_tuple, direction)
    disp_inv_cf = torch.movedim(disp_inv, -1, 1)
    disp_inv_sampled_cf = grid_sample_nd(disp_inv_cf, y_norm, mode='bilinear', padding_mode='border')
    disp_inv_sampled = torch.movedim(disp_inv_sampled_cf, 1, -1)

    return torch.norm(disp_fwd + disp_inv_sampled, dim=-1)


def calculate_inverse_identity_error(W_disp: torch.Tensor, W_inv_disp: torch.Tensor, spacing, origin, direction) -> dict:
    """Summarize forward/inverse identity error, restricted to interior (non-boundary) voxels.

    ``Error(x) = || W_inv_disp(x) + W_disp(x + W_inv_disp(x)) ||``.

    Parameters
    ----------
    W_disp, W_inv_disp : torch.Tensor
        Forward/inverse displacement fields of shape ``(B, *spatial, dim)``
        or ``(*spatial, dim)``.
    spacing, origin, direction : sequence/array-like
        Physical-space metadata.

    Returns
    -------
    dict
        ``{'max_error': float, 'mean_error': float, 'error_map': torch.Tensor}``.
        ``mean_error`` and ``max_error`` are computed over interior voxels
        only (a 1-voxel boundary rim is excluded via
        :func:`antstorch.syn.core.smoothing.get_boundary_mask`).
    """
    dim = len(spacing)
    if W_disp.ndim == dim + 1:
        W_disp = W_disp.unsqueeze(0)
    if W_inv_disp.ndim == dim + 1:
        W_inv_disp = W_inv_disp.unsqueeze(0)
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
    coords_phys = X_phys + W_inv_disp

    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()

    shape_t = torch.tensor(spatial, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)

    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)

    forward_at_inv_cf = grid_sample(torch.movedim(W_disp, -1, 1), coords_norm, padding_mode='border', align_corners=True)
    forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)

    error = W_inv_disp + forward_at_inv

    inside_mask = (coords_norm >= -1) & (coords_norm <= 1)
    inside_mask = inside_mask.all(dim=-1, keepdim=True)

    boundary_mask = get_boundary_mask(spatial, device, dtype).squeeze(0).squeeze(-1)
    error = error * boundary_mask.unsqueeze(0).unsqueeze(-1) * inside_mask

    error_norm = torch.sqrt(torch.sum(error ** 2, dim=-1))
    return {
        'max_error': float(error_norm.max().item()),
        'mean_error': float(error_norm.sum().item() / (boundary_mask.sum().item() + 1e-8)),
        'error_map': error_norm.squeeze(0)
    }
