"""Linear (affine) transform parameterization and grid/physical-space conversions.

Ported from ``syntx.core.affine`` (PyTorch backend only). Provides a
Lie-algebra parameterized hierarchical affine module (``Translation`` ⊂
``Rigid`` ⊂ ``Similarity`` ⊂ ``Affine``) plus the linear-algebra bridging
functions between ``grid_sample``'s normalized ``[-1, 1]`` grid-space
affine matrices and ITK physical-space affine matrices, including parsing
existing ``ants`` linear transforms and building an initial sampling grid
from an arbitrary ``ants`` transform list (linear or non-linear).
"""

import os

import numpy as np
import torch
import torch.nn as nn


def get_rotation_matrix(omega: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Compute a 2D or 3D rotation matrix from a Lie algebra parameterization.

    Uses a first-order Taylor expansion near :math:`\omega = 0` to prevent
    zero-angle gradient locking and division-by-zero singularities during
    automatic differentiation.

    Parameters
    ----------
    omega : torch.Tensor
        Lie algebra rotation vector (1 element for a 2D angle; 3 elements
        ``[w0, w1, w2]`` for a 3D axis-angle vector).
    dim : int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    torch.Tensor
        Rotation matrix :math:`R \in SO(d)` of shape ``(2, 2)`` or ``(3, 3)``.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3.
    """
    device = omega.device
    dtype = omega.dtype
    if dim == 2:
        theta = omega[0]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t])
        ])
    elif dim == 3:
        theta2 = torch.sum(omega**2)
        is_zero = theta2 < 1e-16
        safe_theta2 = torch.where(is_zero, 1e-16, theta2)
        theta = torch.sqrt(safe_theta2)

        safe_theta = torch.where(is_zero, 1.0, theta)
        omega_norm = omega / safe_theta

        K_raw = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega[2], omega[1]]),
            torch.stack([omega[2], torch.tensor(0.0, device=device, dtype=dtype), -omega[0]]),
            torch.stack([-omega[1], omega[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])

        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[2], omega_norm[1]]),
            torch.stack([omega_norm[2], torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[0]]),
            torch.stack([-omega_norm[1], omega_norm[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)
        R_small = I + K_raw
        return torch.where(is_zero, R_small, R)
    else:
        raise ValueError("Only 2D and 3D are supported.")


class HierarchicalAffine(nn.Module):
    """Hierarchical differentiable linear transformation module.

    Parameterizes physical linear transformations using a Lie algebra
    :math:`SO(d)` rotation representation to eliminate gimbal lock and
    maintain continuous gradient flow at identity initialization.

    Supported transformation hierarchy (``transform_type``):

    - ``'Translation'``: :math:`d`-dimensional physical shift vector.
    - ``'Rigid'``: translation + :math:`SO(d)` Lie algebra rotation.
    - ``'Similarity'``: rigid + isotropic scaling factor :math:`s`.
    - ``'Affine'``: similarity + anisotropic scaling :math:`S` + upper-triangular shear matrix :math:`Sh`.

    Parameters
    ----------
    dim : int, default=3
        Spatial dimensionality (2 or 3).
    transform_type : str, default='Affine'
        Linear transformation model (``'Translation'``, ``'Rigid'``,
        ``'Similarity'``, ``'Affine'``).

    Attributes
    ----------
    translation : nn.Parameter
        Translation parameter vector of shape ``(dim,)``.
    omega : nn.Parameter
        Lie algebra rotation vector of shape ``(dim*(dim-1)//2,)``.
    scale : nn.Parameter or torch.Tensor
        Isotropic scaling factor.
    anisotropic_scale : nn.Parameter or torch.Tensor
        Per-axis scaling factor vector of shape ``(dim,)``.
    shear : nn.Parameter or torch.Tensor
        Upper-triangular shear parameter vector.
    """

    def __init__(self, dim: int = 3, transform_type: str = 'Affine'):
        super().__init__()
        self.dim = dim
        self.type = transform_type

        # Translation
        self.translation = nn.Parameter(torch.zeros(dim))

        # Rotation (Lie Algebra SO(d))
        num_rot = dim * (dim - 1) // 2
        self.omega = nn.Parameter(torch.zeros(num_rot))

        # Scale (Similarity)
        if transform_type in ['Similarity', 'Affine']:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

        # Shear/Anisotropic Scale
        if transform_type == 'Affine':
            self.anisotropic_scale = nn.Parameter(torch.ones(dim))
            self.shear = nn.Parameter(torch.zeros(num_rot))
        else:
            self.register_buffer('anisotropic_scale', torch.ones(dim))
            self.register_buffer('shear', torch.zeros(num_rot))

        self.register_buffer('T_init', None)

    def clamp_parameters(self):
        """Clamp scale/shear/rotation parameters to numerically safe ranges, in place."""
        with torch.no_grad():
            if isinstance(self.scale, nn.Parameter):
                self.scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.anisotropic_scale, nn.Parameter):
                self.anisotropic_scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.shear, nn.Parameter):
                self.shear.clamp_(min=-5.0, max=5.0)
            if isinstance(self.omega, nn.Parameter):
                self.omega.clamp_(min=-3.14159265, max=3.14159265)

    def get_matrix(self):
        """Assemble the full homogeneous transform matrix from the current parameters.

        Returns
        -------
        torch.Tensor
            Homogeneous matrix of shape ``(dim + 1, dim + 1)``, composed
            with ``self.T_init`` (if set) as ``T @ T_init``.
        """
        R = get_rotation_matrix(self.omega, self.dim)

        if self.type == 'Affine':
            S = torch.diag(self.anisotropic_scale * self.scale)
            Sh = torch.eye(self.dim, device=self.shear.device, dtype=self.shear.dtype)
            triu_indices = torch.triu_indices(self.dim, self.dim, offset=1)
            Sh[triu_indices[0], triu_indices[1]] = self.shear
            A = R @ S @ Sh
        else:
            A = R * self.scale

        T = torch.eye(self.dim + 1, device=self.translation.device, dtype=self.translation.dtype)
        T[:self.dim, :self.dim] = A
        T[:self.dim, self.dim] = self.translation

        if hasattr(self, 'T_init') and self.T_init is not None:
            return T @ self.T_init
        return T

    def get_affine_grid_matrix(self):
        """Return the ``(dim, dim + 1)`` matrix block consumed by ``grid_sample``-style APIs."""
        T = self.get_matrix()
        return T[:self.dim, :self.dim + 1]


def _grid_to_physical_affine_torch_yfirst(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction,
                                           moving_shape, moving_spacing, moving_origin, moving_direction):
    """Convert a grid-space affine matrix to physical space, given axes already in ``(y, x[, z])`` order.

    Internal helper for :func:`grid_to_physical_affine_torch`; see that
    function for the public, axis-order-agnostic entry point.
    """
    dim = len(fixed_shape)
    device = T_grid.device
    orig_dtype = T_grid.dtype
    calc_dtype = torch.float32

    Nx = torch.tensor(fixed_shape, device=device, dtype=calc_dtype)
    Ny = torch.tensor(moving_shape, device=device, dtype=calc_dtype)
    Sx = torch.tensor(fixed_spacing, device=device, dtype=calc_dtype)
    Sy = torch.tensor(moving_spacing, device=device, dtype=calc_dtype)
    Ox = torch.tensor(fixed_origin, device=device, dtype=calc_dtype)
    Oy = torch.tensor(moving_origin, device=device, dtype=calc_dtype)
    Dx = torch.tensor(fixed_direction, device=device, dtype=calc_dtype)
    Dy = torch.tensor(moving_direction, device=device, dtype=calc_dtype)

    Kx = torch.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = torch.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Kx_inv = torch.inverse(Kx)
    Sx_inv = torch.inverse(torch.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.t()
    bx = - Kx_inv @ Sx_inv @ Dx.t() @ Ox - Kx_inv @ Cx

    Vy = Dy @ torch.diag(Sy) @ Ky
    cy = Dy @ torch.diag(Sy) @ Cy + Oy

    A_grid = T_grid[:dim, :dim].to(calc_dtype)
    t_grid = T_grid[:dim, dim].to(calc_dtype)

    M_phys = (Vy @ A_grid @ Wx).to(orig_dtype)
    t_phys = (Vy @ (A_grid @ bx + t_grid) + cy).to(orig_dtype)
    return M_phys, t_phys


def grid_to_physical_affine_torch(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction,
                                   moving_shape, moving_spacing, moving_origin, moving_direction):
    """Convert a ``grid_sample``-space affine matrix to a physical-space affine matrix (tensor inputs).

    ``T_grid`` operates in ``grid_sample``'s ``(x, y[, z])`` axis order;
    this function permutes it to ``(z, y, x)`` before the underlying
    conversion, and returns the physical-space matrices in that same
    ``(z, y, x)`` order, matching ``antstorch.syn.core`` tensor
    conventions.

    Parameters
    ----------
    T_grid : torch.Tensor
        Homogeneous ``(dim + 1, dim + 1)`` grid-space affine matrix.
    fixed_shape, moving_shape : sequence of int
        Fixed/moving image shapes, in PyTorch (reversed-ITK, e.g.
        ``(z, y, x)``) axis order — unlike every other metadata argument
        here, shape is *not* reversed internally, so callers must pass
        ``tuple(reversed(image.shape))`` rather than ``image.shape``
        directly.
    fixed_spacing, moving_spacing : sequence of float
        Physical voxel spacing, in ITK ``(x, y[, z])`` order.
    fixed_origin, moving_origin : sequence of float
        Physical origin, in ITK ``(x, y[, z])`` order.
    fixed_direction, moving_direction : array_like
        Direction cosine matrix, in ITK ``(x, y[, z])`` order.

    Returns
    -------
    tuple of torch.Tensor
        ``(M_phys, t_phys)`` — the ``(dim, dim)`` linear part and
        ``(dim,)`` translation part of the physical-space affine
        transform, in ``(z, y, x)`` order.
    """
    dim = len(fixed_shape)
    # T_grid operates in grid_sample's XY order; permute to YX for _yfirst
    perm = list(range(dim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D
    T_yx = T_grid.clone()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]
    fs_rev = tuple(reversed(fixed_spacing))
    fo_rev = tuple(reversed(fixed_origin))
    fd_rev = np.asarray(fixed_direction)[::-1, ::-1].copy()
    ms_rev = tuple(reversed(moving_spacing))
    mo_rev = tuple(reversed(moving_origin))
    md_rev = np.asarray(moving_direction)[::-1, ::-1].copy()
    M_phys_zyx, t_phys_zyx = _grid_to_physical_affine_torch_yfirst(
        T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev
    )

    # Return ZYX physical affine matrices directly to match PyTorch tensor coordinate ordering (Z, Y, X)
    return M_phys_zyx, t_phys_zyx


def physical_to_grid_affine(M_phys, t_phys, fixed_img, moving_img):
    """Convert a physical-space affine transform to a ``grid_sample``-space affine matrix (numpy/ants inputs).

    Inverse operation of :func:`grid_to_physical_affine`.

    Parameters
    ----------
    M_phys : ndarray of shape (dim, dim)
        Physical-space linear part, in ITK ``(x, y[, z])`` order.
    t_phys : ndarray of shape (dim,)
        Physical-space translation part, in ITK ``(x, y[, z])`` order.
    fixed_img, moving_img : ants.ANTsImage
        Fixed and moving images supplying shape/spacing/origin/direction
        metadata.

    Returns
    -------
    ndarray of shape (dim + 1, dim + 1)
        Homogeneous affine matrix in ``grid_sample``'s ``(x, y[, z])``
        axis order.
    """
    dim = fixed_img.dimension
    Nx = np.array(fixed_img.shape)
    Ny = np.array(moving_img.shape)
    Sx = np.array(fixed_img.spacing)
    Sy = np.array(moving_img.spacing)
    Ox = np.array(fixed_img.origin)
    Oy = np.array(moving_img.origin)
    Dx = np.array(fixed_img.direction)
    Dy = np.array(moving_img.direction)

    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Wx_inv = Dx @ np.diag(Sx) @ Kx
    bx = - np.linalg.inv(Kx) @ np.linalg.inv(np.diag(Sx)) @ Dx.T @ Ox - np.linalg.inv(Kx) @ Cx

    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
    Vy_inv = np.linalg.inv(Vy)

    A_grid = Vy_inv @ M_phys @ Wx_inv
    t_grid = Vy_inv @ (t_phys - cy) - A_grid @ bx

    T_grid = np.eye(dim + 1, dtype=np.float32)
    T_grid[:dim, :dim] = A_grid
    T_grid[:dim, dim] = t_grid

    perm = list(range(dim - 1, -1, -1))
    T_xyz = T_grid.copy()
    T_xyz[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_xyz[:dim, dim] = T_grid[:dim, dim][perm]
    return T_xyz


def grid_to_physical_affine(T_grid, fixed, moving):
    """Convert a ``grid_sample``-space affine matrix to a physical-space affine transform (numpy/ants inputs).

    Numpy/``ants``-metadata counterpart to
    :func:`grid_to_physical_affine_torch`, returning ITK-order
    ``(x, y[, z])`` physical matrices rather than tensor-order
    ``(z, y, x)`` ones.

    Parameters
    ----------
    T_grid : ndarray of shape (dim + 1, dim + 1)
        Homogeneous grid-space affine matrix, in ``grid_sample``'s
        ``(x, y[, z])`` axis order.
    fixed, moving : ants.ANTsImage
        Fixed and moving images supplying shape/spacing/origin/direction
        metadata.

    Returns
    -------
    tuple of ndarray
        ``(M_phys_xyz, t_phys_xyz)`` in ITK ``(x, y[, z])`` physical
        order.
    """
    dim = len(fixed.shape)
    Nx = np.array(list(reversed(fixed.shape)), dtype=np.float32)
    Ny = np.array(list(reversed(moving.shape)), dtype=np.float32)

    # Reverse spacing, origin, and direction to match PyTorch/JAX (z, y, x) order
    Sx = np.array(fixed.spacing)[::-1]
    Sy = np.array(moving.spacing)[::-1]
    Ox = np.array(fixed.origin)[::-1]
    Oy = np.array(moving.origin)[::-1]
    Dx = np.array(fixed.direction)[::-1, ::-1]
    Dy = np.array(moving.direction)[::-1, ::-1]

    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0

    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Kx_inv = np.linalg.inv(Kx)
    Sx_inv = np.linalg.inv(np.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx

    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy

    perm = list(range(dim - 1, -1, -1))
    T_yx = T_grid.copy()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]

    A_grid = T_yx[:dim, :dim]
    t_grid = T_yx[:dim, dim]

    # Compute in (z, y, x) space
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy

    # Permute from (z, y, x) to (x, y, z) for ITK physical space
    P = np.eye(dim)[::-1]
    M_phys_xyz = P @ M_phys @ P
    t_phys_xyz = P @ t_phys

    return M_phys_xyz, t_phys_xyz


def parse_ants_affine(tx_list, dim):
    """Parse one or more ``ants`` linear transforms into composed ``(M_phys, t_phys)`` tensors.

    Accounts for each transform's center of rotation ``C`` per
    ``t_new = t + C - M @ C``, and composes multiple transforms in list
    order. Supports ``AffineTransform``, ``TranslationTransform``, and
    (as a point-based fallback) arbitrary transform files from which a
    linear mapping can be inferred by warping unit basis points.

    Parameters
    ----------
    tx_list : str, ants.ANTsTransform, or list thereof
        One or more transform file paths or in-memory ``ants``
        transforms, composed in the given order.
    dim : int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    tuple of torch.Tensor or (None, None)
        ``(M_phys, t_phys)`` — the composed ``(dim, dim)`` linear part
        and ``(dim,)`` translation part, as ``float32`` tensors; both
        ``None`` if no transform in ``tx_list`` could be parsed.
    """
    import ants

    if not isinstance(tx_list, (list, tuple)):
        tx_list = [tx_list]
    if len(tx_list) == 0:
        return None, None
    M_composed = np.eye(dim, dtype=np.float32)
    t_composed = np.zeros(dim, dtype=np.float32)
    parsed_any = False

    for tx_item in tx_list:
        tx = None
        try:
            if hasattr(tx_item, 'parameters') and hasattr(tx_item, 'fixed_parameters'):
                tx = tx_item
            elif isinstance(tx_item, str):
                try:
                    tx = ants.read_transform(tx_item)
                except Exception:
                    continue
        except Exception:
            continue

        if tx is None:
            continue

        params = None
        fixed_params = None
        try:
            params = tx.parameters
            fixed_params = tx.fixed_parameters
        except Exception:
            params = None

        if params is not None and len(params) == 12 and dim == 3:
            M = np.array(params[:9], dtype=np.float32).reshape(3, 3)
            t = np.array(params[9:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 3 else np.zeros(3, dtype=np.float32)
        elif params is not None and len(params) == 6 and dim == 2:
            M = np.array(params[:4], dtype=np.float32).reshape(2, 2)
            t = np.array(params[4:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 2 else np.zeros(2, dtype=np.float32)
        elif params is not None and len(params) == dim:  # TranslationTransform (2D: 2, 3D: 3)
            M = np.eye(dim, dtype=np.float32)
            t = np.array(params, dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == dim else np.zeros(dim, dtype=np.float32)
        elif isinstance(tx_item, str) and os.path.exists(tx_item):
            # Robust fallback: extract linear mapping via point transformations
            try:
                import pandas as pd
                if dim == 3:
                    pts = pd.DataFrame({'x': [0.0, 1.0, 0.0, 0.0], 'y': [0.0, 0.0, 1.0, 0.0], 'z': [0.0, 0.0, 0.0, 1.0]})
                    w_pts = ants.apply_transforms_to_points(dim=3, points=pts, transformlist=[tx_item])
                    p0 = np.array(w_pts.iloc[0])
                    p1 = np.array(w_pts.iloc[1]) - p0
                    p2 = np.array(w_pts.iloc[2]) - p0
                    p3 = np.array(w_pts.iloc[3]) - p0
                    M = np.column_stack([p1, p2, p3]).astype(np.float32)
                    t = p0.astype(np.float32)
                    C = np.zeros(3, dtype=np.float32)
                else:
                    pts = pd.DataFrame({'x': [0.0, 1.0, 0.0], 'y': [0.0, 0.0, 1.0]})
                    w_pts = ants.apply_transforms_to_points(dim=2, points=pts, transformlist=[tx_item])
                    p0 = np.array(w_pts.iloc[0])
                    p1 = np.array(w_pts.iloc[1]) - p0
                    p2 = np.array(w_pts.iloc[2]) - p0
                    M = np.column_stack([p1, p2]).astype(np.float32)
                    t = p0.astype(np.float32)
                    C = np.zeros(2, dtype=np.float32)
            except Exception:
                continue
        else:
            continue

        t_new = t + C - M @ C
        t_composed = M @ t_composed + t_new
        M_composed = M @ M_composed
        parsed_any = True

    if not parsed_any:
        return None, None

    M_phys = torch.from_numpy(M_composed).to(torch.float32)
    t_phys = torch.from_numpy(t_composed).to(torch.float32)
    return M_phys, t_phys


def compute_initial_grid(fixed, moving, tx_list):
    """Compute an initial ``grid_sample`` sampling grid from an arbitrary (possibly non-linear) ``ants`` transform list.

    Unlike :func:`grid_to_physical_affine`/:func:`physical_to_grid_affine`,
    which only handle linear transforms in closed form, this builds the
    grid numerically: it warps moving-space physical coordinate images
    through ``tx_list`` via ``ants.apply_transforms`` and maps the
    result back to normalized moving-space voxel indices, so ``tx_list``
    may include any combination of linear and non-linear transforms.

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
        Fixed and moving images.
    tx_list : list of str
        ``ants``-style forward transform list, applied via
        ``ants.apply_transforms``.

    Returns
    -------
    ndarray
        Sampling grid of shape ``(1, *fixed.shape, dim)``, in
        ``grid_sample``'s normalized ``[-1, 1]`` axis order matching
        ``antstorch.syn.core`` tensor conventions.
    """
    import ants
    dim = moving.dimension

    # 1. Get moving physical coordinates via numpy meshgrid
    shape = moving.shape
    grids = [np.arange(s) for s in shape]
    meshgrid_idxs = np.meshgrid(*grids, indexing='ij')
    idxs = np.stack(meshgrid_idxs, axis=-1)

    direction = np.array(moving.direction)
    spacing = np.array(moving.spacing)
    origin = np.array(moving.origin)

    idxs_flat = idxs.reshape(-1, dim)
    scaled_idxs = idxs_flat * spacing
    phys_flat = (direction @ scaled_idxs.T).T + origin
    coord_np = phys_flat.reshape(shape + (dim,)).astype(np.float32)

    # 2. Warp each coordinate component image to the fixed space
    warped_coords = []
    for d in range(dim):
        c_img = ants.from_numpy(coord_np[..., d], origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
        w_c_img = ants.apply_transforms(fixed=fixed, moving=c_img, transformlist=tx_list)
        warped_coords.append(w_c_img.numpy())

    moving_phys_at_fixed = np.stack(warped_coords, axis=-1)

    # 3. Map physical coordinates to voxel indices in moving space
    shape = moving_phys_at_fixed.shape
    phys_flat = moving_phys_at_fixed.reshape(-1, dim)

    direction_inv = np.linalg.inv(direction)
    diff = phys_flat - origin
    sp_idx = diff @ direction_inv.T
    voxel_idx = sp_idx / spacing

    # 4. Normalize voxel indices to [-1, 1] for grid_sample (x, y, [z]) convention
    normalized_coords = []
    for d in range(dim):
        N = moving.shape[d]
        norm_d = (voxel_idx[:, d] / (N - 1)) * 2.0 - 1.0
        normalized_coords.append(norm_d)

    normalized_grid_flat = np.stack(normalized_coords, axis=-1)

    grid = normalized_grid_flat.reshape(fixed.shape + (dim,))
    if dim == 2:
        grid = np.transpose(grid, (1, 0, 2))
    elif dim == 3:
        grid = np.transpose(grid, (2, 1, 0, 3))
    initial_grid = np.expand_dims(grid.astype(np.float32), axis=0)
    return initial_grid
