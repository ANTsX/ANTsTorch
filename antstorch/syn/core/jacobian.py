"""Spatial Jacobian and Jacobian-determinant computation for N-D vector fields.

Ported from ``syntx.core.jacobian`` (PyTorch backend only). All public
functions operate on batched vector fields of shape ``(B, *spatial, dim)``
(channel-last), matching the convention used throughout ``antstorch.syn``.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _spatial_jacobian_nd(field: torch.Tensor, physical_spacing=None, method='central') -> torch.Tensor:
    """Compute the spatial Jacobian of an N-D vector field.

    Parameters
    ----------
    field : torch.Tensor
        Vector field of shape ``(B, *spatial, d)``.
    physical_spacing : sequence of float, optional
        Physical voxel spacing per spatial axis, in the same (reversed
        PyTorch-storage) axis order as ``field``'s spatial dimensions. If
        ``None``, a normalized spacing of ``2 / (size - 1)`` is assumed per
        axis (i.e. ``field`` is treated as living on the ``[-1, 1]`` grid).
    method : {'central', 'bspline'}
        Finite-difference scheme: ``'central'`` uses ``torch.gradient``
        (2nd-order central differences); ``'bspline'`` uses a 4th-order
        accurate cubic B-spline derivative filter.

    Returns
    -------
    torch.Tensor
        Jacobian tensor of shape ``(B, *spatial, d, d)`` where
        ``J[..., i, j] = d field_i / d x_j``.
    """
    dim = field.shape[-1]
    spatial = field.shape[1:-1]
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (s - 1) for s in spatial]

    if method == 'bspline':
        # 4th-order accurate cubic B-spline derivative filter.
        grads = []
        for i, sp in enumerate(spacings):
            k_np = np.array([-1 / 12, -8 / 12, 0.0, 8 / 12, 1 / 12], dtype=np.float32) / sp
            k_t = torch.from_numpy(k_np).to(device=field.device, dtype=field.dtype)

            pad = [0, 0] + [0, 0] * (len(spatial) - 1 - i) + [2, 2] + [0, 0] * i
            padded = F.pad(field, pad, mode='replicate')

            perm = [0] + [j + 1 for j in range(len(spatial)) if j != i] + [i + 1, len(spatial) + 1]
            perm_inv = [0] + [0] * len(spatial) + [len(spatial) + 1]
            for orig_pos, p_val in enumerate(perm[1:-1], start=1):
                perm_inv[p_val] = orig_pos

            field_perm = padded.permute(perm)
            orig_shape = field_perm.shape
            flat_in = field_perm.reshape(-1, 1, orig_shape[-2])
            k_view = k_t.view(1, 1, 5)
            conv_out = F.conv1d(flat_in, k_view)
            conv_restored = conv_out.view(orig_shape[0], *orig_shape[1:-2], conv_out.shape[-1], orig_shape[-1])
            g_i = conv_restored.permute(perm_inv)
            grads.append(g_i)
        return torch.stack(grads, dim=-1)

    grads = torch.gradient(field, spacing=spacings, dim=list(range(1, len(spatial) + 1)))
    return torch.stack(grads, dim=-1)  # (B, *spatial, d, d)


def compute_jacobian_determinant_nd(warp_field: torch.Tensor, physical_spacing=None) -> torch.Tensor:
    """Compute the Jacobian determinant of a displacement or deformation field.

    Parameters
    ----------
    warp_field : torch.Tensor
        Field of shape ``(B, *spatial, dim)``. Whether it is treated as a
        *displacement* (``det(I + du/dx)``) or a *deformation* (identity
        grid plus ``warp_field``) depends on ``warp_field.is_physical``: a
        physical-space field (an attribute set by
        :class:`antstorch.syn.transform.SyNToTransform` and callers that
        pass ``physical_spacing``) is treated as a displacement; otherwise it
        is treated as a normalized-space deformation field.
    physical_spacing : sequence of float, optional
        Physical voxel spacing per spatial axis. Supplying this implies the
        displacement-field interpretation even if ``is_physical`` is unset.

    Returns
    -------
    torch.Tensor
        Jacobian determinant map of shape ``(B, *spatial)``.

    Raises
    ------
    ValueError
        If ``warp_field`` is not 2-D or 3-D.
    """
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype

    if warp_field.dim() == dim:
        warp_field = warp_field.unsqueeze(0)

    is_physical = getattr(warp_field, 'is_physical', physical_spacing is not None)

    if is_physical:
        if physical_spacing is not None:
            spacings = tuple(float(s) for s in physical_spacing)
        else:
            spacings = tuple(1.0 for _ in range(dim))

        grads = torch.gradient(warp_field, spacing=spacings, dim=tuple(range(1, dim + 1)))

        if dim == 2:
            # grads[0] is d/dy (spatial axis 1), grads[1] is d/dx (spatial axis 2).
            # warp[..., 0] is u_y, warp[..., 1] is u_x.
            du_y_dy = grads[0][..., 0]
            du_y_dx = grads[1][..., 0]
            du_x_dy = grads[0][..., 1]
            du_x_dx = grads[1][..., 1]

            j00 = 1.0 + du_x_dx
            j11 = 1.0 + du_y_dy
            j01 = du_x_dy
            j10 = du_y_dx
            return j00 * j11 - j01 * j10
        elif dim == 3:
            # grads[0]=d/dz, grads[1]=d/dy, grads[2]=d/dx.
            # warp[..., 0]=u_z, warp[..., 1]=u_y, warp[..., 2]=u_x.
            du_z_dz = grads[0][..., 0]
            du_z_dy = grads[1][..., 0]
            du_z_dx = grads[2][..., 0]

            du_y_dz = grads[0][..., 1]
            du_y_dy = grads[1][..., 1]
            du_y_dx = grads[2][..., 1]

            du_x_dz = grads[0][..., 2]
            du_x_dy = grads[1][..., 2]
            du_x_dx = grads[2][..., 2]

            j00 = 1.0 + du_x_dx
            j01 = du_x_dy
            j02 = du_x_dz

            j10 = du_y_dx
            j11 = 1.0 + du_y_dy
            j12 = du_y_dz

            j20 = du_z_dx
            j21 = du_z_dy
            j22 = 1.0 + du_z_dz

            return j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
        else:
            raise ValueError(f"Only 2-D and 3-D fields are supported, got a {dim}-D field.")
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(warp_field.shape[0], *([-1] * (dim + 1)))

        phi = identity + warp_field
        if physical_spacing is not None:
            spacings = list(physical_spacing)
        else:
            spacings = [2.0 / (size - 1) for size in spatial]
        grads = torch.gradient(phi, spacing=spacings, dim=list(range(1, dim + 1)))

        if dim == 2:
            j00 = grads[1][..., 0]
            j01 = grads[0][..., 0]
            j10 = grads[1][..., 1]
            j11 = grads[0][..., 1]
            return j00 * j11 - j01 * j10
        elif dim == 3:
            j00 = grads[2][..., 0]
            j01 = grads[1][..., 0]
            j02 = grads[0][..., 0]

            j10 = grads[2][..., 1]
            j11 = grads[1][..., 1]
            j12 = grads[0][..., 1]

            j20 = grads[2][..., 2]
            j21 = grads[1][..., 2]
            j22 = grads[0][..., 2]

            return j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
        else:
            raise ValueError(f"Only 2-D and 3-D fields are supported, got a {dim}-D field.")


def compute_jacobian_hinge_penalty(warp_field: torch.Tensor, physical_spacing=None, epsilon: float = 0.05) -> torch.Tensor:
    """Differentiable one-sided fold-prevention hinge penalty.

    ``L_hinge = mean( relu(epsilon - det(J))^2 )`` — zero wherever the local
    Jacobian determinant is at least ``epsilon``, and quadratically penalized
    as it approaches or crosses zero (a fold).

    Parameters
    ----------
    warp_field : torch.Tensor
        Field of shape ``(B, *spatial, dim)``; see
        :func:`compute_jacobian_determinant_nd` for the displacement vs.
        deformation interpretation.
    physical_spacing : sequence of float, optional
        Physical voxel spacing per spatial axis.
    epsilon : float
        Minimum acceptable Jacobian determinant before the penalty engages.

    Returns
    -------
    torch.Tensor
        Scalar hinge-penalty loss.
    """
    det_J = compute_jacobian_determinant_nd(warp_field, physical_spacing=physical_spacing)
    hinge = F.relu(epsilon - det_J)
    return torch.mean(hinge ** 2)


def compute_physical_jacobian_determinant(
    warp_field: torch.Tensor,
    direction: torch.Tensor,
    spacing: torch.Tensor
) -> torch.Tensor:
    """Compute the physical-space Jacobian determinant map from a displacement field.

    Evaluates spatial gradients using the physical spacing and direction
    matrix, constructs the deformation gradient ``F(x) = I + grad(u)(x)``,
    and returns its point-wise determinant. Non-positive values
    (``det(J) <= 0``) indicate topological grid folding and loss of
    diffeomorphic invertibility.

    Parameters
    ----------
    warp_field : torch.Tensor
        Displacement field of shape ``(B, *spatial, dim)``, in normalized or
        physical mm coordinates.
    direction : torch.Tensor or array-like
        Physical direction-cosine matrix of shape ``(dim, dim)``.
    spacing : torch.Tensor or array-like
        Physical voxel spacing vector, shape ``(dim,)``, in mm.

    Returns
    -------
    torch.Tensor
        Physical Jacobian determinant map of shape ``(B, *spatial)``.
    """
    is_physical = getattr(warp_field, 'is_physical', False)
    if is_physical:
        return compute_jacobian_determinant_nd(warp_field, physical_spacing=spacing)

    device = warp_field.device
    dtype = warp_field.dtype
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction, device=device, dtype=dtype)
    else:
        direction = direction.to(device=device, dtype=dtype)

    if not isinstance(spacing, torch.Tensor):
        spacing = torch.tensor(spacing, device=device, dtype=dtype)
    else:
        spacing = spacing.to(device=device, dtype=dtype)

    # 1. Voxel-space Jacobian via normalized-grid spacing.
    normalized_spacings = [2.0 / (s - 1) for s in spatial]
    grads = torch.gradient(warp_field, spacing=normalized_spacings, dim=list(range(1, dim + 1)))
    # Reverse gradient list to align with the (x, y, [z]) component convention.
    J_voxel = torch.stack(list(reversed(grads)), dim=-1)  # (B, *spatial, dim, dim)

    # 2. Voxel-to-physical matrices M and M_inv.
    M = direction * spacing.unsqueeze(0)  # (dim, dim)
    M_inv = direction.t() * (1.0 / spacing).unsqueeze(1)  # (dim, dim)

    # 3. Similarity transform J_phys = M @ J_voxel @ M_inv.
    J_phys = torch.einsum('ij,b...jk,kl->b...il', M, J_voxel, M_inv)

    # 4. Deformation gradient F = J_phys + I.
    F_mat = J_phys + torch.eye(dim, device=device, dtype=dtype)

    # 5. Determinant of F computed analytically to avoid MPS batch LU
    #    decomposition deadlocks.
    if dim == 2:
        a = F_mat[..., 0, 0]
        b = F_mat[..., 0, 1]
        c = F_mat[..., 1, 0]
        d = F_mat[..., 1, 1]
        jac_det_phys = a * d - b * c
    elif dim == 3:
        a = F_mat[..., 0, 0]
        b = F_mat[..., 0, 1]
        c = F_mat[..., 0, 2]
        d = F_mat[..., 1, 0]
        e = F_mat[..., 1, 1]
        f = F_mat[..., 1, 2]
        g = F_mat[..., 2, 0]
        h = F_mat[..., 2, 1]
        i = F_mat[..., 2, 2]
        jac_det_phys = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    else:
        jac_det_phys = torch.linalg.det(F_mat)

    return jac_det_phys
