"""Grid sampling and physical/normalized coordinate-grid utilities.

Ported from ``syntx.core.grid`` (PyTorch backend only). Follows the same
ITK-metadata-reversed-to-PyTorch-storage convention used throughout
``antstorch`` (see ``antstorch.bspline_flows.bspline_domain``): physical
metadata (spacing, origin, direction) is supplied in ITK ``(x, y[, z])``
order, and tensor spatial axes are the PyTorch-reversed ``(z, y, x)``.
"""

import numpy as np
import torch
import torch.nn.functional as F


def grid_sample_bspline_torch(
    image: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = 'border',
    align_corners: bool = True
) -> torch.Tensor:
    """C1-continuous 2-D/3-D cubic B-spline grid sampling.

    Unlike ``torch.nn.functional.grid_sample`` (which supports only
    nearest/bilinear/bicubic-2D interpolation), this evaluates a true cubic
    B-spline interpolant at arbitrary continuous sample locations, in both
    2-D and 3-D, with a continuous first derivative — useful when a smoother
    interpolant than trilinear is needed for image resampling or gradient
    computation.

    Parameters
    ----------
    image : torch.Tensor
        Image of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
    grid : torch.Tensor
        Sample locations in ``[-1, 1]``, of shape ``(B, H_out, W_out, 2)`` or
        ``(B, D_out, H_out, W_out, 3)``, in ``grid_sample`` axis order.
    padding_mode : {'border', 'zeros'}
        Out-of-bounds handling: ``'border'`` clamps to the nearest edge
        voxel; ``'zeros'`` treats out-of-bounds contributions as zero.
    align_corners : bool
        Must be ``True`` (the only convention implemented).

    Returns
    -------
    torch.Tensor
        Sampled image of shape ``(B, C, *grid.shape[1:-1])``.

    Raises
    ------
    ValueError
        If ``image`` is not 2-D or 3-D (excluding batch/channel dims).
    """
    ndim = image.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(f"Only 2-D and 3-D grid sampling are supported, got ndim={ndim}")

    B, C = image.shape[:2]
    device = image.device
    dtype = image.dtype
    spatial_target = grid.shape[1:-1]

    def get_w(u):
        u2 = u * u
        u3 = u2 * u
        w0 = (1.0 - u) ** 3 / 6.0
        w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
        w2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
        w3 = u3 / 6.0
        return [w0, w1, w2, w3]

    if ndim == 2:
        H, W = image.shape[2:]
        gx, gy = grid[..., 0], grid[..., 1]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        ix, iy = torch.floor(vx), torch.floor(vy)
        ux, uy = vx - ix, vy - iy
        wx, wy = get_w(ux), get_w(uy)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, H * W)

        for ky in range(4):
            jy_raw = (iy + ky - 1).long()
            jy = jy_raw.clamp(0, H - 1)
            valid_y = (jy_raw >= 0) & (jy_raw < H)
            w_y = wy[ky].unsqueeze(1)
            for kx in range(4):
                jx_raw = (ix + kx - 1).long()
                jx = jx_raw.clamp(0, W - 1)
                valid_x = (jx_raw >= 0) & (jx_raw < W)
                w_yx = w_y * wx[kx].unsqueeze(1)
                idx = (jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                sampled_flat = torch.gather(img_flat, 2, idx)
                sampled = sampled_flat.view(B, C, *spatial_target)
                if padding_mode == 'zeros':
                    valid_mask = (valid_y & valid_x).view(B, 1, *spatial_target)
                    sampled = sampled * valid_mask
                out = out + w_yx * sampled
        return out
    else:
        D, H, W = image.shape[2:]
        gx, gy, gz = grid[..., 0], grid[..., 1], grid[..., 2]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        vz = (gz + 1.0) * (D - 1) / 2.0 if align_corners else (gz + 1.0) * D / 2.0 - 0.5
        ix, iy, iz = torch.floor(vx), torch.floor(vy), torch.floor(vz)
        ux, uy, uz = vx - ix, vy - iy, vz - iz
        wx, wy, wz = get_w(ux), get_w(uy), get_w(uz)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, D * H * W)

        for kz in range(4):
            jz_raw = (iz + kz - 1).long()
            jz = jz_raw.clamp(0, D - 1)
            valid_z = (jz_raw >= 0) & (jz_raw < D)
            w_z = wz[kz].unsqueeze(1)
            for ky in range(4):
                jy_raw = (iy + ky - 1).long()
                jy = jy_raw.clamp(0, H - 1)
                valid_y = (jy_raw >= 0) & (jy_raw < H)
                w_zy = w_z * wy[ky].unsqueeze(1)
                for kx in range(4):
                    jx_raw = (ix + kx - 1).long()
                    jx = jx_raw.clamp(0, W - 1)
                    valid_x = (jx_raw >= 0) & (jx_raw < W)
                    w_zyx = w_zy * wx[kx].unsqueeze(1)
                    idx = (jz * (H * W) + jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                    sampled_flat = torch.gather(img_flat, 2, idx)
                    sampled = sampled_flat.view(B, C, *spatial_target)
                    if padding_mode == 'zeros':
                        valid_mask = (valid_z & valid_y & valid_x).view(B, 1, *spatial_target)
                        sampled = sampled * valid_mask
                    out = out + w_zyx * sampled
        return out


def _image_spatial_gradient(image):
    dim = image.dim() - 2
    if dim == 2:
        grad_x = (torch.roll(image, shifts=-1, dims=-1) - torch.roll(image, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(image, shifts=-1, dims=-2) - torch.roll(image, shifts=1, dims=-2)) / 2.0
        return torch.stack([grad_x, grad_y], dim=2)
    elif dim == 3:
        grad_x = (torch.roll(image, shifts=-1, dims=-1) - torch.roll(image, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(image, shifts=-1, dims=-2) - torch.roll(image, shifts=1, dims=-2)) / 2.0
        grad_z = (torch.roll(image, shifts=-1, dims=-3) - torch.roll(image, shifts=1, dims=-3)) / 2.0
        return torch.stack([grad_x, grad_y, grad_z], dim=2)
    return None


class AnalyticalGridSample(torch.autograd.Function):
    """``grid_sample`` with a hand-derived backward pass w.r.t. the sampling grid.

    Only the gradient with respect to ``grid`` is computed (matching the
    common case where the *source image* does not itself require a
    gradient, e.g. a fixed reference image). This avoids the memory and
    compute cost of PyTorch's default autograd path through
    ``grid_sample``, which differentiates with respect to both inputs.
    """

    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='border', align_corners=True):
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.save_for_backward(input, grid)
        if input.dtype != grid.dtype:
            input = input.to(grid.dtype)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners

        dim = input.dim() - 2
        spatial_shape = input.shape[2:]
        B, C = input.shape[:2]

        # 1. Spatial gradients of the source image: dI/dx, dI/dy, dI/dz.
        grad_I = _image_spatial_gradient(input)  # (B, C, dim, *spatial_shape)

        # 2. Sample source gradients at the grid lookup coordinates.
        grad_I_flat = grad_I.view(B, C * dim, *spatial_shape).to(dtype=grid.dtype)
        grad_I_sampled = F.grid_sample(grad_I_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        grad_I_sampled = grad_I_sampled.view(B, C, dim, *grid.shape[1:-1])  # (B, C, dim, *spatial_grid)

        # 3. Inner product with the incoming loss gradient.
        grad_out_cast = grad_output.to(dtype=grid.dtype)
        grad_grid = torch.sum(grad_out_cast.unsqueeze(2) * grad_I_sampled, dim=1).movedim(1, -1)  # (B, *spatial_grid, dim)

        # 4. Voxel-to-normalized grid coordinate scaling.
        scales = []
        for d in range(dim):
            size = spatial_shape[dim - 1 - d]  # X is dim - 1, Y is dim - 2, Z is dim - 3.
            s = (size - 1) / 2.0 if align_corners else size / 2.0
            scales.append(s)
        scale_t = torch.tensor(scales, dtype=grad_grid.dtype, device=grad_grid.device)
        grad_grid = grad_grid * scale_t

        return None, grad_grid, None, None, None


def grid_sample_nd(input, grid, mode='bilinear', padding_mode='border', align_corners=True, interpolator='linear', use_analytical_gradients=True):
    """Dispatching wrapper around ``grid_sample``/:func:`grid_sample_bspline_torch`/:class:`AnalyticalGridSample`.

    Parameters
    ----------
    input : torch.Tensor
        Image of shape ``(B, C, *spatial)``.
    grid : torch.Tensor
        Sample locations in ``[-1, 1]``, of shape ``(B, *spatial_out, dim)``.
    mode : str
        Interpolation mode; accepts ``torch.nn.functional.grid_sample``
        modes plus ``'bspline'``/``'nearestNeighbor'`` aliases.
    padding_mode : str
        Out-of-bounds handling, passed through to the selected backend.
    align_corners : bool
        Passed through to the selected backend.
    interpolator : str
        Alternate spelling for ``mode`` (accepted for ANTs-style call
        sites); ``mode`` and ``interpolator`` are checked independently, so
        either may request ``'bspline'``/``'nearestNeighbor'``.
    use_analytical_gradients : bool
        If ``True`` and ``grid.requires_grad`` while ``input`` does not,
        route through :class:`AnalyticalGridSample` for a cheaper backward
        pass. Ignored for the B-spline/nearest backends.

    Returns
    -------
    torch.Tensor
        Sampled tensor of shape ``(B, C, *grid.shape[1:-1])``.
    """
    if interpolator in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor') or mode in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor'):
        mode = 'nearest'
    if interpolator == 'bspline' or mode == 'bspline':
        return grid_sample_bspline_torch(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    if input.dtype != grid.dtype:
        input = input.to(grid.dtype)
    if use_analytical_gradients and grid.requires_grad and not input.requires_grad:
        return AnalyticalGridSample.apply(input, grid, mode, padding_mode, align_corners)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def compose_grids(grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
    """Compose two normalized coordinate grids: ``grid1 o grid2``.

    Parameters
    ----------
    grid1, grid2 : torch.Tensor
        Coordinate grids of shape ``(B, *spatial, dim)``, in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        The composed grid, same shape as the inputs: ``grid2`` resampled at
        ``grid1``'s locations.
    """
    grid1_cf = torch.movedim(grid1, -1, 1)
    composed_cf = F.grid_sample(grid1_cf, grid2, mode='bilinear', padding_mode='border', align_corners=True)
    return torch.movedim(composed_cf, 1, -1)


def _get_physical_grid_torch_yfirst(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    idxs = torch.stack(meshgrid, dim=-1)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)

    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)


def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    """Build the physical-coordinate grid for a PyTorch-ordered image shape.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape in PyTorch (reversed-ITK) axis order.
    spacing, origin : sequence of float
        Physical spacing/origin, in ITK ``(x, y[, z])`` order.
    direction : array-like
        Direction-cosine matrix, in ITK ``(x, y[, z])`` order (either
        ``(dim, dim)`` or already flattened).
    device, dtype : torch.device, torch.dtype
        Target tensor device/dtype.

    Returns
    -------
    torch.Tensor
        Physical-coordinate grid of shape ``(1, *shape, dim)``, with
        components in physical ``(x, y[, z])`` order.
    """
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _get_physical_grid_torch_yfirst(shape, spacing_rev, origin_rev, direction_rev, device, dtype)


def _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing, origin, direction):
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)

    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)

    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    inv_direction_t = torch.inverse(direction_t.t())
    rotated = diff @ inv_direction_t
    voxel_coords = rotated / spacing_t

    shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    # Flip from internal (z, y, x) order to grid_sample's expected (x, y, z) order.
    norm_coords = torch.flip(norm_coords, dims=[-1])
    return norm_coords.view(phys_coords.shape)


def physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction):
    """Convert physical-space coordinates to a normalized ``[-1, 1]`` sampling grid.

    Parameters
    ----------
    phys_coords : torch.Tensor
        Physical coordinates of shape ``(B, *spatial, dim)``, in physical
        ``(x, y[, z])`` component order.
    target_shape : sequence of int
        Spatial shape (of the image to be sampled) in PyTorch order.
    spacing, origin : sequence of float
        Physical spacing/origin of the image to be sampled, in ITK order.
    direction : array-like
        Direction-cosine matrix of the image to be sampled, in ITK order.

    Returns
    -------
    torch.Tensor
        Normalized grid, same shape as ``phys_coords``, usable directly with
        ``grid_sample``.
    """
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(target_shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing_rev, origin_rev, direction_rev)


def physical_to_normalized_torch_cached(phys_coords, shape_t, spacing_t, origin_t, direction_t):
    """Fast path for :func:`physical_to_normalized_torch` with precomputed metadata tensors.

    Parameters
    ----------
    phys_coords : torch.Tensor
        Physical coordinates of shape ``(B, *spatial, dim)``.
    shape_t, spacing_t, origin_t, direction_t : torch.Tensor
        The target image's shape/spacing/origin/direction, already
        converted to PyTorch tensors in reversed (z, y, x) order (as
        prepared once per call site rather than rebuilt every iteration of
        an optimization loop).

    Returns
    -------
    torch.Tensor
        Normalized grid, same shape as ``phys_coords``.
    """
    dim = phys_coords.shape[-1]
    flat_phys = phys_coords.view(-1, dim)
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = -(origin_t @ M) - 1.0
    flat_norm = flat_phys @ M + b
    norm_coords = torch.flip(flat_norm, dims=[-1])
    return norm_coords.view(phys_coords.shape)


def prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fixed_spacing, moving_spacing,
    M_phys, t_phys, initial_grid_level,
    interpolator='linear',
    grad_I_curr=None, grad_J_curr=None,
    use_analytical_gradients=True
):
    """Sample both images (and their gradients) at the symmetric SyN midpoint.

    Shared by the greedy-SyN update step: given the current forward/inverse
    half-warps for both the fixed-to-midpoint and moving-to-midpoint flows,
    this resamples the fixed and moving images (and, if not already
    supplied, their spatial gradients) at the midpoint locations, and
    returns an in-bounds mask marking where both samples are valid.

    Returns
    -------
    tuple of torch.Tensor
        ``(I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask)``.
    """
    from .jacobian import _spatial_jacobian_nd

    phi_l2r_phys = X_phys + warp_l2r
    coords_norm = physical_to_normalized_torch_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    I_mid = grid_sample_nd(I_curr, coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)

    phi_r2l_phys = X_phys + warp_r2l
    y_phys = phi_r2l_phys @ M_phys.t() + t_phys
    if initial_grid_level is not None:
        y_norm_fixed = physical_to_normalized_torch_cached(
            y_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
        )
        y_norm = compose_grids(initial_grid_level, y_norm_fixed)
    else:
        y_norm = physical_to_normalized_torch_cached(
            y_phys, moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t
        )

    J_mid = grid_sample_nd(J_curr, y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)

    if grad_I_curr is None:
        grad_I_curr = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=tuple(reversed(fixed_spacing))).squeeze(-2)
    if grad_J_curr is None:
        grad_J_curr = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=tuple(reversed(moving_spacing))).squeeze(-2)

    grad_I_mid_sampled = grid_sample_nd(grad_I_curr.movedim(-1, 1), coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1).contiguous()
    grad_I_mid_sampled = torch.matmul(grad_I_mid_sampled, fixed_direction_t.t())

    grad_J_mid_sampled = grid_sample_nd(grad_J_curr.movedim(-1, 1), y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1).contiguous()
    grad_J_mid_sampled = torch.matmul(grad_J_mid_sampled, moving_direction_t.t())
    grad_J_mid_sampled = torch.matmul(grad_J_mid_sampled, M_phys)

    dim = coords_norm.shape[-1]
    mask_I = (coords_norm[..., 0] >= -1.0) & (coords_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_I = mask_I & (coords_norm[..., d] >= -1.0) & (coords_norm[..., d] <= 1.0)

    mask_J = (y_norm[..., 0] >= -1.0) & (y_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_J = mask_J & (y_norm[..., d] >= -1.0) & (y_norm[..., d] <= 1.0)

    in_bounds_mask = (mask_I & mask_J).unsqueeze(1).to(dtype=I_mid.dtype)

    return I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask
