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

from .smoothing import separable_gaussian_filter, get_boundary_mask
from .grid import (
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    grid_sample_nd
)


def update_inverse_field_nd_hybrid_lm(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor,
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    damping_factor: float = 1.0,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None
) -> torch.Tensor:
    """Damped Levenberg-Marquardt hybrid solver for displacement-field inversion.

    Bridges 1st-order fixed-point iteration and a 2nd-order Newton solver by
    solving ``[I + grad(u) + lambda * I] * delta_v = -(v + u(y + v))`` at
    each step, where the local damping ``lambda(y)`` ramps up wherever
    ``det(I + grad(u)) < 0.2`` — giving quadratic Newton-like convergence in
    well-behaved regions while retaining fixed-point stability under large,
    locally non-invertible deformations.

    Parameters
    ----------
    W_disp : torch.Tensor
        Forward displacement field of shape ``(B, *spatial, dim)``, in
        physical mm units.
    W_inv_disp : torch.Tensor, optional
        Initial guess for the inverse field; defaults to ``-W_disp`` if
        ``None``.
    steps : int
        Maximum number of solver iterations.
    relaxation : float
        Global relaxation factor applied to each update.
    smoothing_sigma : float
        Physical-unit Gaussian smoothing sigma applied to the inverse field
        after each update; ``0`` disables smoothing.
    max_error_threshold, mean_error_threshold : float
        Convergence thresholds (physical units) on the composed
        forward-then-inverse identity residual's max/mean norm.
    damping_factor : float
        Scales the local Levenberg-Marquardt damping term.
    spacing, origin, direction : sequence/array-like, optional
        Physical-space metadata. If ``spacing`` is given without ``origin``
        or ``direction``, they default to zero origin / identity direction.
    X_phys : torch.Tensor, optional
        Precomputed physical-coordinate grid; built from
        ``spacing``/``origin``/``direction`` if not supplied.

    Returns
    -------
    torch.Tensor
        The estimated inverse displacement field, same shape as ``W_disp``.
        Falls back to :func:`update_inverse_field_nd` (ITK-style fixed
        point) if no physical-space metadata is available at all.
    """
    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    boundary_mask = get_boundary_mask(spatial, device, dtype)
    W_disp_cf = torch.movedim(W_disp, -1, 1)

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    if X_phys is not None or (spacing is not None and origin is not None and direction is not None):
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
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

        max_error_norm = float('inf')
        mean_error_norm = float('inf')

        for iteration in range(steps):
            if max_error_norm <= max_error_threshold and mean_error_norm <= mean_error_threshold:
                break

            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv

            scaled_norm = torch.sqrt(torch.sum((error / spacing_t) ** 2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())

            # Vectorized central finite differences for the spatial gradient grad_u, shape (B, *spatial, d, d).
            grad_dims = []
            for s_idx in range(dim):
                spatial_dim = s_idx + 1
                g_s = (torch.roll(forward_at_inv, -1, dims=spatial_dim) - torch.roll(forward_at_inv, 1, dims=spatial_dim)) / (2.0 * spacing_t[s_idx])
                grad_dims.append(g_s)
            grad_u = torch.stack(grad_dims, dim=-1)  # (B, *spatial, d, d)

            I_mat = torch.eye(dim, device=device, dtype=dtype).reshape(*([1] * (dim + 1)), dim, dim)
            J_mat = I_mat + grad_u

            if dim == 3:
                det_J = (
                    J_mat[..., 0, 0] * (J_mat[..., 1, 1] * J_mat[..., 2, 2] - J_mat[..., 1, 2] * J_mat[..., 2, 1]) -
                    J_mat[..., 0, 1] * (J_mat[..., 1, 0] * J_mat[..., 2, 2] - J_mat[..., 1, 2] * J_mat[..., 2, 0]) +
                    J_mat[..., 0, 2] * (J_mat[..., 1, 0] * J_mat[..., 2, 1] - J_mat[..., 1, 1] * J_mat[..., 2, 0])
                )
            else:
                det_J = J_mat[..., 0, 0] * J_mat[..., 1, 1] - J_mat[..., 0, 1] * J_mat[..., 1, 0]

            # Local Levenberg-Marquardt damping, ramping up wherever det(J) < 0.2.
            lambda_spatial = torch.clamp(0.2 - det_J.unsqueeze(-1), min=0.0, max=1.0) * damping_factor * 10.0

            M_lambda = J_mat + lambda_spatial.unsqueeze(-1) * I_mat

            # Closed-form Cramer's rule solve for M_lambda @ delta_v = -error.
            b_vec = -error
            a00 = M_lambda[..., 0, 0]
            a01 = M_lambda[..., 0, 1]
            a02 = M_lambda[..., 0, 2] if dim == 3 else torch.zeros_like(a00)
            a10 = M_lambda[..., 1, 0]
            a11 = M_lambda[..., 1, 1]
            a12 = M_lambda[..., 1, 2] if dim == 3 else torch.zeros_like(a00)
            a20 = M_lambda[..., 2, 0] if dim == 3 else torch.zeros_like(a00)
            a21 = M_lambda[..., 2, 1] if dim == 3 else torch.zeros_like(a00)
            a22 = M_lambda[..., 2, 2] if dim == 3 else torch.ones_like(a00)

            b0 = b_vec[..., 0]
            b1 = b_vec[..., 1]
            b2 = b_vec[..., 2] if dim == 3 else torch.zeros_like(b0)

            det_M = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
            safe_det = torch.where(det_M.abs() < 1e-6, torch.sign(det_M + 1e-6) * 1e-6, det_M)

            x0 = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / safe_det
            x1 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / safe_det
            if dim == 3:
                x2 = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / safe_det
                delta_v = torch.stack([x0, x1, x2], dim=-1)
            else:
                delta_v = torch.stack([x0, x1], dim=-1)

            epsilon = 0.75 if iteration == 0 else 0.5
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = delta_v * clip_scale
            W_inv_disp = W_inv_disp + update * relaxation * epsilon

            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma, spacing=spacing)
            W_inv_disp = W_inv_disp * boundary_mask

        return W_inv_disp
    else:
        return update_inverse_field_nd(W_disp, W_inv_disp, steps=steps, relaxation=relaxation, smoothing_sigma=smoothing_sigma)


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
                    F.grid_sample(v_k_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
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
                    F.grid_sample(v_k_cf, sample_coords, padding_mode='border', align_corners=True), 1, -1
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
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t) ** 2, dim=-1, keepdim=True))
        else:
            coords = identity + v_curr
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale) ** 2, dim=-1, keepdim=True))

        max_error_norm = float(scaled_norm.max())
        mean_error_norm = float(scaled_norm.mean())

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
        return v_new, max_error_norm, mean_error_norm

    v_k = W_inv_disp.clone()
    R_history = []
    G_history = []

    for iteration in range(steps):
        g_k, max_err, mean_err = itk_fixed_point_step(v_k, iteration)

        if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
            v_k = g_k
            break

        r_k = (g_k - v_k).reshape(-1)
        R_history.append(r_k)
        G_history.append(g_k.reshape(-1))

        if len(R_history) > m + 1:
            R_history.pop(0)
            G_history.pop(0)

        m_k = len(R_history)

        if m_k < 2:
            v_k = g_k
        else:
            n_cols = m_k - 1
            r_newest = R_history[-1]
            dR_cols = []
            for j in range(n_cols):
                dR_cols.append(r_newest - R_history[j])

            gram = torch.zeros(n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(n_cols, device=device, dtype=dtype)
            for i_col in range(n_cols):
                rhs[i_col] = torch.dot(dR_cols[i_col], r_newest)
                for j_col in range(i_col, n_cols):
                    val = torch.dot(dR_cols[i_col], dR_cols[j_col])
                    gram[i_col, j_col] = val
                    gram[j_col, i_col] = val

            gram += 1e-10 * torch.eye(n_cols, device=device, dtype=dtype)

            try:
                gamma = torch.linalg.solve(gram, rhs)
            except torch.linalg.LinAlgError:
                v_k = g_k
                continue

            g_newest = G_history[-1]
            v_new_flat = g_newest.clone()
            for j in range(n_cols):
                dG_j = g_newest - G_history[j]
                v_new_flat = v_new_flat - gamma[j] * (dG_j + dR_cols[j])

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate = v_candidate * boundary_mask

            if use_physical:
                coords_phys_c = X_phys + v_candidate
                coords_norm_c = physical_to_normalized_torch_cached(coords_phys_c, shape_t, spacing_t, origin_t, direction_t)
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_norm_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c / spacing_t) ** 2).sqrt())
            else:
                coords_c = identity + v_candidate
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c * voxel_scale) ** 2).sqrt())

            residual_fp = float(torch.dot(r_k, r_k).sqrt())

            # Only accept the Anderson-extrapolated candidate if it does not
            # meaningfully worsen the residual versus plain fixed point.
            if residual_aa <= residual_fp * 1.1:
                v_k = v_candidate
            else:
                v_k = g_k

    return v_k


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
    X_phys=None
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

    Returns
    -------
    torch.Tensor
        The estimated inverse displacement field, same shape as ``W_disp``.
    """
    if method == 'hybrid_lm':
        return update_inverse_field_nd_hybrid_lm(
            W_disp, W_inv_disp, steps=steps, relaxation=relaxation,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )

    if method == 'anderson':
        return update_inverse_field_nd_anderson(
            W_disp, W_inv_disp, steps=steps,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

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
            if max_error_norm <= max_error_threshold and mean_error_norm <= mean_error_threshold:
                break

            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t) ** 2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())

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

        return W_inv_disp
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
            if max_error_norm <= max_error_threshold or mean_error_norm <= mean_error_threshold:
                break

            coords = identity + W_inv_disp
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale) ** 2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())

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

        return W_inv_disp


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

    forward_at_inv_cf = F.grid_sample(torch.movedim(W_disp, -1, 1), coords_norm, padding_mode='border', align_corners=True)
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
