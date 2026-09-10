from __future__ import annotations

from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .device_manager import get_default_device


def _get_gaussian_kernel_1d(
    sigma: float,
    order: int,
    truncate: float = 4.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct a 1D Gaussian or Gaussian derivative kernel in PyTorch.
    """
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    phi = torch.exp(-0.5 * (x / sigma) ** 2)
    sum_phi = torch.sum(phi)
    if order == 0:
        return phi / sum_phi
    elif order == 1:
        dphi = -x / (sigma ** 2) * phi
        return dphi / sum_phi
    else:
        raise ValueError(f"Unsupported order {order}; only 0 and 1 are supported.")


def _pad_symmetric_1d(x: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    """
    Apply symmetric reflection padding (mirroring at edge voxels) along dimension `dim`.
    Matches numpy / ITK symmetric boundary conditions.
    """
    if radius <= 0:
        return x
    left = x.narrow(dim, 0, radius).flip(dim)
    right = x.narrow(dim, -radius, radius).flip(dim)
    return torch.cat([left, x, right], dim=dim)


def _gaussian_filter_3d(
    image_tensor: torch.Tensor,
    sigmas: tuple[float, float, float] | list[float],
    orders: tuple[int, int, int],
    truncate: float = 4.0,
) -> torch.Tensor:
    """
    Separable 3D Gaussian convolution along dimensions (D, H, W).
    `image_tensor` shape must be (1, 1, D, H, W).
    """
    out = image_tensor
    dev = out.device

    # Axis 0: Depth (dim 2 of 5D tensor)
    k_d = _get_gaussian_kernel_1d(sigmas[0], orders[0], truncate, device=dev).flip(0).view(1, 1, -1, 1, 1)
    r_d = k_d.shape[2] // 2
    out = _pad_symmetric_1d(out, r_d, 2)
    out = F.conv3d(out, k_d)

    # Axis 1: Height (dim 3 of 5D tensor)
    k_h = _get_gaussian_kernel_1d(sigmas[1], orders[1], truncate, device=dev).flip(0).view(1, 1, 1, -1, 1)
    r_h = k_h.shape[3] // 2
    out = _pad_symmetric_1d(out, r_h, 3)
    out = F.conv3d(out, k_h)

    # Axis 2: Width (dim 4 of 5D tensor)
    k_w = _get_gaussian_kernel_1d(sigmas[2], orders[2], truncate, device=dev).flip(0).view(1, 1, 1, 1, -1)
    r_w = k_w.shape[4] // 2
    out = _pad_symmetric_1d(out, r_w, 4)
    out = F.conv3d(out, k_w)

    return out


def weingarten_image_curvature(
    image: "ants.ANTsImage",
    sigma: float = 1.0,
    opt: Union[str, int] = "mean",
    mask: Optional[Union["ants.ANTsImage", np.ndarray, torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    chunk_size: int = 100000,
) -> "ants.ANTsImage":
    """
    Compute image mean, Gaussian, or surface characterization curvature using
    the Weingarten map in PyTorch with GPU/MPS/CPU acceleration.

    This implements the discrete level-set Weingarten curvature operator
    derived from differential geometry of implicit surfaces, providing exact
    mathematical parity with ANTsPy `ants.weingarten_image_curvature` / ITK.

    Arguments
    ---------
    image : ANTsImage
        Input 2D or 3D ANTsImage from which curvature is calculated.
    sigma : float
        Gaussian smoothing scale parameter in physical space (mm). Default is 1.0.
        Values <= 0.5 automatically fall back to 1.66 matching ITK conventions.
    opt : str or int
        Type of curvature to compute:
        - 'mean' (or 0): Mean curvature H = 0.5 * (k1 + k2).
        - 'gaussian' (or 6): Gaussian curvature K = k1 * k2.
        - 'characterize' (or 5): Topographic characterization into 8 discrete classes:
            1 = Peak, 2 = Pit, 3 = Saddle Ridge, 4 = Saddle Valley,
            5 = Ridge, 6 = Valley, 7 = Flat, 8 = Minimal Surface.
    mask : ANTsImage, np.ndarray, or torch.Tensor, optional
        Optional binary mask restricting computation to voxels where mask > 0.
        If None, all voxels with intensity > 0 (excluding a 3-voxel border) are processed.
    device : str or torch.device, optional
        Target device for PyTorch acceleration (e.g. 'cuda', 'mps', or 'cpu').
        If None, defaults to `antstorch.get_default_device()`.
    chunk_size : int
        Voxel batch chunk size for neighborhood sampling and Weingarten solve.
        Default is 100,000 to maintain low GPU memory footprint.

    Returns
    -------
    ANTsImage
        Curvature scalar map preserving original physical geometry and orientation.

    Example
    -------
    >>> import ants
    >>> import antstorch
    >>> img = ants.image_read(ants.get_ants_data('mni')).resample_image((3, 3, 3))
    >>> curv = antstorch.weingarten_image_curvature(img, sigma=1.5, opt='mean')
    """
    if opt == 6 or opt == "gaussian":
        opt_mode = "gaussian"
    elif opt == 5 or opt == "characterize":
        opt_mode = "characterize"
    else:
        opt_mode = "mean"

    if image.dimension not in {2, 3}:
        raise ValueError("image must be 2D or 3D")

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Replicate 2D slice into a 3D volume matching ANTsPy/ITK conventions
    if image.dimension == 2:
        d = image.shape
        temp = np.zeros(list(d) + [10], dtype=np.float32)
        voxvals = image.numpy()
        for k in range(1, 7):
            temp[: d[0], : d[1], k] = voxvals
        temp_img = ants.from_numpy(temp)
        myspc = list(image.spacing) + [min(image.spacing)]
        temp_img.set_spacing(myspc)
        temp_img.set_direction(np.eye(3))
        temp_img = temp_img.clone("float")
    else:
        temp_img = image.clone("float")
        temp_img.set_direction(np.eye(3))

    spacing = np.array(temp_img.spacing, dtype=np.float32)
    origin = np.array(temp_img.origin, dtype=np.float32)
    array = temp_img.numpy().astype(np.float32)

    if sigma <= 0.5:
        sigma = 1.66

    sigmas = [sigma / s for s in spacing]

    # Convert image to 5D PyTorch tensor: (1, 1, D, H, W)
    img_t = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    # Compute smoothed 3D spatial gradient derivatives with physical spacing scaling
    Ix_t = _gaussian_filter_3d(img_t, sigmas, (1, 0, 0)) / spacing[0]
    Iy_t = _gaussian_filter_3d(img_t, sigmas, (0, 1, 0)) / spacing[1]
    Iz_t = _gaussian_filter_3d(img_t, sigmas, (0, 0, 1)) / spacing[2]

    grad_vol = torch.cat([Ix_t, Iy_t, Iz_t], dim=1)  # (1, 3, D, H, W)

    # Process mask and 3-voxel border exclusion
    margin = 3
    if mask is not None:
        if hasattr(mask, "numpy"):
            mask_arr = mask.numpy() > 0
        else:
            mask_arr = np.asarray(mask) > 0

        if image.dimension == 2:
            padded_mask = np.zeros_like(array, dtype=bool)
            for k in range(1, 7):
                padded_mask[:, :, k] = mask_arr
            mask_bool = padded_mask & (array > 0.0)
        else:
            mask_bool = mask_arr & (array > 0.0)
    else:
        mask_bool = array > 0.0

    mask_np = np.array(mask_bool, copy=True)
    mask_np[:margin, :, :] = False
    mask_np[-margin:, :, :] = False
    mask_np[:, :margin, :] = False
    mask_np[:, -margin:, :] = False
    mask_np[:, :, :margin] = False
    mask_np[:, :, -margin:] = False

    valid_indices = np.argwhere(mask_np)
    num_voxels = len(valid_indices)

    out_array = np.zeros_like(array)

    if num_voxels > 0:
        # Precompute 27-neighborhood offsets and pseudo-inverse matrix D_pinv
        delta = 0.5 * min(spacing)
        steps = [-delta, 0.0, delta]
        grid_coeffs = np.array(np.meshgrid(steps, steps, steps)).T.reshape(-1, 3)

        D = np.stack(
            [np.ones_like(grid_coeffs[:, 1]), -grid_coeffs[:, 1], -grid_coeffs[:, 2]],
            axis=1,
        )  # (27, 3)
        D_pinv = np.linalg.pinv(D)  # (3, 27)

        grid_coeffs_t = torch.from_numpy(grid_coeffs.astype(np.float32)).to(device)
        D_pinv_t = torch.from_numpy(D_pinv.astype(np.float32)).to(device)
        spacing_t = torch.from_numpy(spacing).to(device)
        origin_t = torch.from_numpy(origin).to(device)

        results_mean = []
        results_gauss = []
        results_class = []

        shape_d, shape_h, shape_w = array.shape

        for start_idx in range(0, num_voxels, chunk_size):
            end_idx = min(start_idx + chunk_size, num_voxels)
            idx_chunk = torch.from_numpy(valid_indices[start_idx:end_idx]).to(device)

            # 1. Physical origin Q of target voxels
            Q = origin_t + spacing_t * idx_chunk  # (M, 3)

            # 2. Gradient at voxel centers
            Ix_val = Ix_t[0, 0, idx_chunk[:, 0], idx_chunk[:, 1], idx_chunk[:, 2]]
            Iy_val = Iy_t[0, 0, idx_chunk[:, 0], idx_chunk[:, 1], idx_chunk[:, 2]]
            Iz_val = Iz_t[0, 0, idx_chunk[:, 0], idx_chunk[:, 1], idx_chunk[:, 2]]

            normal_unnorm = torch.stack([Ix_val, Iy_val, Iz_val], dim=1)  # (M, 3)
            mag = torch.norm(normal_unnorm, dim=1, keepdim=True)
            normal = torch.where(mag > 1e-5, normal_unnorm / mag, torch.zeros_like(normal_unnorm))

            # 3. Construct orthonormal tangent frame (tangent1, tangent2)
            c0 = torch.abs(normal[:, 0]) > 0.1
            norm0 = 1.0 / torch.where(c0, normal[:, 0], torch.ones_like(normal[:, 0]))
            t1_c0 = torch.stack(
                [
                    -1.0 * norm0 * (normal[:, 1] + normal[:, 2]),
                    norm0 * normal[:, 0],
                    norm0 * normal[:, 0],
                ],
                dim=1,
            )

            c1 = (~c0) & (torch.abs(normal[:, 1]) > 0.1)
            norm1 = 1.0 / torch.where(c1, normal[:, 1], torch.ones_like(normal[:, 1]))
            t1_c1 = torch.stack(
                [
                    norm1 * normal[:, 1],
                    -1.0 * norm1 * (normal[:, 0] + normal[:, 2]),
                    norm1 * normal[:, 1],
                ],
                dim=1,
            )

            c2 = (~c0) & (~c1) & (torch.abs(normal[:, 2]) > 0.1)
            norm2 = 1.0 / torch.where(c2, normal[:, 2], torch.ones_like(normal[:, 2]))
            t1_c2 = torch.stack(
                [
                    norm2 * normal[:, 2],
                    norm2 * normal[:, 2],
                    -1.0 * norm2 * (normal[:, 0] + normal[:, 1]),
                ],
                dim=1,
            )

            tangent1 = torch.zeros_like(normal)
            tangent1 = torch.where(c0.unsqueeze(1), t1_c0, tangent1)
            tangent1 = torch.where(c1.unsqueeze(1), t1_c1, tangent1)
            tangent1 = torch.where(c2.unsqueeze(1), t1_c2, tangent1)

            t1_mag = torch.norm(tangent1, dim=1, keepdim=True)
            t1_mag = torch.where(t1_mag > 1e-9, t1_mag, torch.ones_like(t1_mag))
            tangent1 = tangent1 / t1_mag

            tangent2 = torch.cross(normal, tangent1, dim=1)
            t2_mag = torch.norm(tangent2, dim=1, keepdim=True)
            t2_mag = torch.where(t2_mag > 1e-9, t2_mag, torch.ones_like(t2_mag))
            tangent2 = tangent2 / t2_mag

            # 4. Generate 27 neighborhood points P in physical space
            ui = grid_coeffs_t[:, 1].view(1, 27, 1)
            vi = grid_coeffs_t[:, 2].view(1, 27, 1)
            zi = grid_coeffs_t[:, 0].view(1, 27, 1)

            P = (
                Q.unsqueeze(1)
                + ui * tangent1.unsqueeze(1)
                + vi * tangent2.unsqueeze(1)
                + zi * normal.unsqueeze(1)
            )  # (M, 27, 3)

            # Convert physical positions P to voxel grid coordinates
            voxel_coords = (P - origin_t) / spacing_t

            # Map coordinates to normalized range [-1, 1] for F.grid_sample:
            # PyTorch grid_sample expects (x, y, z) = (Width, Height, Depth)
            grid_w = 2.0 * voxel_coords[:, :, 2] / (shape_w - 1) - 1.0
            grid_h = 2.0 * voxel_coords[:, :, 1] / (shape_h - 1) - 1.0
            grid_d = 2.0 * voxel_coords[:, :, 0] / (shape_d - 1) - 1.0

            grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).view(1, -1, 27, 1, 3)

            # Sample gradients simultaneously for all 3 components: (1, 3, M, 27, 1)
            sampled = F.grid_sample(
                grad_vol,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            PN = sampled.squeeze(0).squeeze(-1).permute(1, 2, 0)  # (M, 27, 3)

            g_mag = torch.norm(PN, dim=-1, keepdim=True)
            g_mag = torch.where(g_mag > 1e-9, g_mag, torch.ones_like(g_mag))
            PN = PN / g_mag  # Normalized sampled normal vectors

            # 5. Weingarten shape matrix solve: C = D_pinv @ PN -> (M, 3, 3)
            C = torch.matmul(D_pinv_t.unsqueeze(0), PN)

            dNdu = C[:, 1, :]
            dNdv = C[:, 2, :]

            # Compute fundamental coefficients: a, b, c, d
            a = torch.sum(dNdu * tangent1, dim=1)
            b = torch.sum(dNdv * tangent1, dim=1)
            c = torch.sum(dNdu * tangent2, dim=1)
            d = torch.sum(dNdv * tangent2, dim=1)

            # Mean curvature H and Gaussian curvature K
            H = 0.5 * (a + d)
            K = a * d - b * c

            invalid = torch.isnan(H) | torch.isinf(H) | torch.isnan(K) | torch.isinf(K)
            H = torch.where(invalid, torch.zeros_like(H), H)
            K = torch.where(invalid, torch.zeros_like(K), K)

            # Surface characterization into 8 classes
            th = 1e-6
            th_k = th * th
            th_h = th
            conds = [
                (H > th_h) & (K > th_k),                          # 1: Peak
                (H < -th_h) & (K > th_k),                         # 2: Pit
                (H > th_h) & (K < -th_k),                         # 3: Saddle Ridge
                (H < -th_h) & (K < -th_k),                        # 4: Saddle Valley
                (H > th_h) & (torch.abs(K) <= th_k),              # 5: Ridge
                (H < -th_h) & (torch.abs(K) <= th_k),             # 6: Valley
                (torch.abs(H) <= th_h) & (torch.abs(K) <= th_k),  # 7: Flat
                (torch.abs(H) <= th_h) & (K < -th_k),             # 8: Minimal surface
            ]
            choices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            classes = torch.zeros_like(H)
            for cond, choice in zip(conds, choices):
                classes = torch.where(cond, torch.full_like(classes, choice), classes)

            results_mean.append(H.detach().cpu().numpy())
            results_gauss.append(K.detach().cpu().numpy())
            results_class.append(classes.detach().cpu().numpy())

        results_mean = np.concatenate(results_mean)
        results_gauss = np.concatenate(results_gauss)
        results_class = np.concatenate(results_class)

        if opt_mode == "mean":
            out_array[mask_np] = results_mean
        elif opt_mode == "gaussian":
            out_array[mask_np] = results_gauss
        elif opt_mode == "characterize":
            out_array[mask_np] = results_class

    out_img = ants.from_numpy(out_array)
    if image.dimension == 2:
        subarr = out_img.numpy()[:, :, 4]
        return ants.copy_image_info(image, ants.from_numpy(subarr))
    else:
        return ants.copy_image_info(image, out_img)
