"""Spatial regularization filters for displacement/velocity fields.

Ported from ``syntx.core.smoothing`` (PyTorch backend only). Provides a
separable Gaussian filter and two fluid-regularization Green's operators:

- :func:`apply_sobolev_green_operator` — an FFT-domain Sobolev filter
  (``1 / (1 + alpha * |k|^2) ** s``), the standard fluid-regularization
  kernel used by greedy SyN.
- :func:`apply_dsti_green_operator` (aka ``apply_dsti1_green_operator``) —
  the same Sobolev filter applied in Discrete Sine Transform Type-I (DST-I)
  space, which analytically enforces homogeneous Dirichlet boundary
  conditions (``v = 0`` at the domain boundary). This is the
  "Dirichlet-Shield" regularizer: it prevents boundary coordinate drift
  without needing a separate boundary mask, at the cost of one extra
  forward/inverse transform pair per axis.

All fields use the channel-last convention ``(B, *spatial, dim)``.
"""

import math
import warnings
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import numpy as np

_gaussian_kernel_cache = {}
_tensor_kernel_cache = {}


def get_cached_gaussian_kernel_1d(sig: float, device, dtype, maximum_error=None):
    """Return a cached 1-D discrete Gaussian kernel (modified Bessel weights).

    The kernel radius is chosen so that the truncated tail contributes less
    than 0.5% of the total mass (``scipy.special.ive`` evaluated at that
    radius falls below 0.005), then renormalized to sum to 1.

    Parameters
    ----------
    sig : float
        Kernel sigma, in voxels.
    device, dtype : torch.device, torch.dtype
        Target tensor device/dtype for the cached kernel.
    maximum_error : float, optional
        Truncate by retained mass (at least 1 - maximum_error), as in ITK,
        instead of the legacy coefficient cutoff.

    Returns
    -------
    torch.Tensor
        Kernel of shape ``(1, 1, 2 * radius + 1)``.
    """
    sig_key = round(float(sig), 5)
    kernel_key = (sig_key, maximum_error)
    cache_key = (kernel_key, str(device), str(dtype))
    if cache_key not in _tensor_kernel_cache:
        if kernel_key not in _gaussian_kernel_cache:
            from scipy.special import ive
            variance = float(sig_key) ** 2
            if maximum_error is None:
                radius = 0
                while ive(radius, variance) > 0.005:
                    radius += 1
            else:
                if not 0 < maximum_error < 1:
                    raise ValueError("maximum_error must be between zero and one")
                radius = 1
                mass = ive(0, variance) + 2 * ive(1, variance)
                while mass < 1 - maximum_error:
                    radius += 1
                    mass += 2 * ive(radius, variance)
            offsets = np.arange(-radius, radius + 1)
            k_np = np.array([ive(abs(k), variance) for k in offsets], dtype=np.float32)
            k_np /= k_np.sum()
            _gaussian_kernel_cache[kernel_key] = k_np
        k_np = _gaussian_kernel_cache[kernel_key]
        _tensor_kernel_cache[cache_key] = torch.from_numpy(k_np).to(device=device, dtype=dtype).view(1, 1, -1)
    return _tensor_kernel_cache[cache_key]


def separable_gaussian_filter(grid: torch.Tensor, sigma, spacing=None, sigma_mode='voxel', maximum_error=None) -> torch.Tensor:
    """Apply separable Gaussian filtering along each spatial dimension.

    Parameters
    ----------
    grid : torch.Tensor
        Field of shape ``(B, *spatial, dim)`` (channel-last).
    sigma : float or sequence of float
        Kernel sigma, shared across axes if a scalar, or one value per
        spatial axis (in the same reversed-PyTorch order as ``grid``'s
        spatial dimensions).
    spacing : sequence of float, optional
        Physical voxel spacing per axis, in ITK ``(x, y[, z])`` order. Only
        used when ``sigma_mode='physical'``.
    sigma_mode : {'voxel', 'physical'}
        ``'voxel'`` (default) uses ``sigma`` directly in voxels;
        ``'physical'`` scales the voxel sigma per axis by ``spacing``
        (clamped to ``[0.5, 10.0]`` voxels for numerical stability).
    maximum_error : float, optional
        ITK-style omitted-mass tolerance. None preserves the legacy cutoff.

    Returns
    -------
    torch.Tensor
        Filtered field, same shape as ``grid``. Returned unchanged if every
        resolved sigma is non-positive.
    """
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)

    if isinstance(sigma, (tuple, list)):
        sigma_list = [float(s) for s in sigma]
    elif sigma_mode == 'physical' and spacing is not None:
        spacing_rev = tuple(reversed(spacing))
        sigma_list = [float(np.clip(float(sigma) / sp, 0.5, 10.0)) for sp in spacing_rev]
    elif isinstance(sigma, (int, float)):
        sigma_list = [float(sigma)] * num_spatial
    else:
        sigma_list = [float(sigma)] * num_spatial

    if all(s <= 0.0 for s in sigma_list):
        return grid

    v = torch.movedim(grid, -1, 1)

    _is_mps = hasattr(device, 'type') and device.type == 'mps'
    if num_spatial == 3 and not _is_mps:
        # Fast path: F.conv3d with degenerate 1-D kernels. Skipped on MPS,
        # where large-volume conv3d with degenerate kernels is unreliable.
        C = v.shape[1]
        for i, sig in enumerate(sigma_list):
            if sig <= 0.0:
                continue
            kernel_1d = get_cached_gaussian_kernel_1d(sig, device, dtype, maximum_error=maximum_error).squeeze(0)
            pad = kernel_1d.shape[-1] // 2

            if i == 0:
                kz = kernel_1d.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), mode='replicate'), kz, groups=C)
            elif i == 1:
                ky = kernel_1d.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), mode='replicate'), ky, groups=C)
            elif i == 2:
                kx = kernel_1d.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), mode='replicate'), kx, groups=C)
        return torch.movedim(v, 1, -1).contiguous()

    for i in range(num_spatial):
        sig = sigma_list[i]
        if sig <= 0.0:
            continue

        kernel = get_cached_gaussian_kernel_1d(sig, device, dtype, maximum_error=maximum_error)
        kernel_size = kernel.shape[-1]
        pad_size = kernel_size // 2

        target_dim = i + 2
        dims = list(range(v.ndim))
        dims[-1], dims[target_dim] = dims[target_dim], dims[-1]
        v_permuted = v.permute(*dims).contiguous()

        last_dim_size = v_permuted.shape[-1]
        v_reshaped = v_permuted.view(-1, 1, last_dim_size)
        v_padded = F.pad(v_reshaped, (pad_size, pad_size), mode='replicate')

        v_conv = F.conv1d(v_padded, kernel)
        v_conv_reshaped = v_conv.view(*v_permuted.shape)
        v_out = v_conv_reshaped.permute(*dims).contiguous()
        v = v_out

    return torch.movedim(v, 1, -1).contiguous()


_SOBOLEV_FILTER_CACHE = {}


def _get_sobolev_filter_cached(spatial_shape, alpha_val, s, spacing, device, dtype):
    sp_tuple = tuple(float(x) for x in spacing) if spacing is not None else None
    cache_key = (tuple(spatial_shape), float(alpha_val), float(s), sp_tuple, str(device), str(dtype))
    if cache_key in _SOBOLEV_FILTER_CACHE:
        return _SOBOLEV_FILTER_CACHE[cache_key]

    dim = len(spatial_shape)
    # ``spacing`` is in ANTs/ITK (x, y, z) order while the spatial axes are in tensor (z, y, x)
    # order: reverse it (syntx >= 5.0). Before this fix the per-axis spacing was applied to the
    # wrong axis for anisotropic images.
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        if d == dim - 1:
            k_d = (torch.fft.rfftfreq(n_d, device=device) * (2.0 * math.pi)) / max(sp_d, 1e-4)
        else:
            k_d = (torch.fft.fftfreq(n_d, device=device) * (2.0 * math.pi)) / max(sp_d, 1e-4)
        k_axes.append(k_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    k_sq = sum(k_j ** 2 for k_j in k_mesh)
    K_fourier = (1.0 / ((1.0 + alpha_val * k_sq) ** s)).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Bound cache growth across many distinct (shape, alpha, spacing) calls.
    if len(_SOBOLEV_FILTER_CACHE) > 32:
        _SOBOLEV_FILTER_CACHE.clear()
    _SOBOLEV_FILTER_CACHE[cache_key] = K_fourier
    return K_fourier


def apply_sobolev_green_operator(m, fluid_sigma=3.0, alpha=None, border_width=0, spacing=None, pad_to_fast=False, **kwargs):
    """Fluid-regularize a field with the FFT-domain Sobolev Green's operator.

    Applies ``1 / (1 + alpha * |k|^2) ** 2`` in the frequency domain — a
    low-pass filter whose smoothness is controlled by ``alpha``, matching
    the fluid regularization used by greedy SyN.

    Parameters
    ----------
    m : torch.Tensor
        Field of shape ``(B, *spatial, dim)`` (channel-last).
    fluid_sigma : float
        Regularization strength; ignored (identity) if ``<= 0``. Used to
        derive ``alpha`` when ``alpha`` is not given directly
        (``alpha = fluid_sigma / 2``).
    alpha : float, optional
        Sobolev kernel parameter, overriding the ``fluid_sigma``-derived
        default.
    spacing : sequence of float, optional
        Physical voxel spacing per axis; scales the frequency grid so the
        same ``alpha`` yields comparable smoothing across resolutions.

    Returns
    -------
    torch.Tensor
        Regularized field, same shape as ``m``. Returned unchanged if
        ``fluid_sigma <= 0``.
    """
    if fluid_sigma <= 0:
        return m
    device = m.device
    dtype = m.dtype
    dim = m.ndim - 2  # input is (B, *spatial, channels)
    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0
    s = 2.0

    spatial_shape = m.shape[1:-1]
    spatial_dims = tuple(range(2, 2 + dim))
    # syntx >= 6.40: contiguous float32 channel-first copy, in-place spectral product (same values; the FFT
    # of a contiguous buffer is also what makes the result bit-identical to syntx's).
    m_cf = torch.movedim(m, -1, 1).to(torch.float32).contiguous()

    K_bc = _get_sobolev_filter_cached(spatial_shape, alpha_val, s, spacing, device, dtype)

    # torch.fft.rfftn/irfftn on MPS resize an internal work buffer from a
    # zero-dim placeholder every call, which PyTorch now flags as a
    # deprecated resize (see the matching note in
    # bspline_flows/n4_bias_field_correction.py) -- purely an MPS-backend
    # implementation detail (no out= argument is passed anywhere below;
    # CPU/CUDA never warn here), so it is suppressed locally rather than
    # left to spam every caller's warning filters on every SyN iteration.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*was resized since it had shape.*",
            category=UserWarning,
        )
        m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
        del m_cf
        m_fft.mul_(K_bc)
        v_cf = torch.fft.irfftn(m_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
        del m_fft

    return torch.movedim(v_cf, 1, -1)


def apply_dsti_green_operator(m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0):
    """Fluid-regularize a field with the DST-I "Dirichlet-Shield" Green's operator.

    Applies the same Sobolev smoothing kernel as
    :func:`apply_sobolev_green_operator`, but in Discrete Sine Transform
    Type-I space (via memory-efficient separable 1-D DST-I transforms built
    from real FFTs) rather than the standard periodic FFT. This analytically
    enforces homogeneous Dirichlet boundary conditions (``v = 0`` at every
    domain face), which prevents boundary coordinate drift without a
    separate boundary mask.

    Parameters
    ----------
    m : torch.Tensor
        Field of shape ``(B, *spatial, dim)`` (channel-last).
    fluid_sigma : float
        Regularization strength; ignored (identity) if ``<= 0``.
    alpha : float, optional
        Sobolev kernel parameter, overriding the ``fluid_sigma``-derived
        default (``alpha = fluid_sigma / 2``).
    spacing : sequence of float, optional
        Voxel spacing in ANTs/ITK ``(x, y, z)`` order (reversed internally).
        The Laplacian eigenvalues are divided by ``h_d ** 2``, as in syntx
        (>= 6.40). ``None`` keeps unit spacing (the previous behaviour).
    s : float, default 2.0
        Operator power.

    Returns
    -------
    torch.Tensor
        Regularized field, same shape as ``m``, with zero displacement at
        the domain boundary. Returned unchanged if ``fluid_sigma <= 0``.
    """
    if fluid_sigma <= 0:
        return m

    device = m.device
    dtype = m.dtype
    spatial_shape = m.shape[1:-1]
    dim = len(spatial_shape)

    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0
    s_val = float(s)

    spacing_zyx = tuple(reversed(tuple(spacing))) if spacing is not None else None
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        scale = (sp_d ** 2) if spacing is not None else 1.0
        lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2) / scale
        k_axes.append(lambda_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_j for k_j in k_mesh)
    K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s_val)

    # Channel-first representation: (B, C, *spatial).
    curr = m.movedim(-1, 1).to(torch.float32)

    # torch.fft.fft on MPS resizes an internal work buffer from a zero-dim
    # placeholder every call, which PyTorch now flags as a deprecated resize
    # (see the matching note on apply_sobolev_green_operator above and in
    # bspline_flows/n4_bias_field_correction.py) -- purely an MPS-backend
    # implementation detail (no out= argument is passed below; CPU/CUDA
    # never warn here), so it is suppressed locally rather than left to spam
    # every caller's warning filters on every SyN iteration of every
    # separable DST-I axis.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*was resized since it had shape.*",
            category=UserWarning,
        )
        # Forward separable DST-I across all spatial dimensions.
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            z_shape = list(curr.shape)
            z_shape[axis] = 1
            z = torch.zeros(z_shape, device=device, dtype=torch.float32)
            rev = -torch.flip(curr, dims=[axis])
            padded = torch.cat([z, curr, z, rev], dim=axis)
            transformed = torch.fft.fft(padded, dim=axis)
            curr = -torch.imag(transformed.narrow(axis, 1, n_d)).contiguous()

        # Multiply by the Dirichlet Sobolev Green's kernel.
        curr = curr * K_dst.unsqueeze(0).unsqueeze(0)

        # Inverse separable DST-I across all spatial dimensions.
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            z_shape = list(curr.shape)
            z_shape[axis] = 1
            z = torch.zeros(z_shape, device=device, dtype=torch.float32)
            rev = -torch.flip(curr, dims=[axis])
            padded = torch.cat([z, curr, z, rev], dim=axis)
            transformed = torch.fft.fft(padded, dim=axis)
            curr = (-torch.imag(transformed.narrow(axis, 1, n_d)) / (2.0 * (n_d + 1))).contiguous()

    return curr.to(dtype=dtype).movedim(1, -1)


def apply_dsti1_green_operator(m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0):
    """Alias for :func:`apply_dsti_green_operator`."""
    return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha, spacing=spacing, s=s)


def separable_1d_filter(x: torch.Tensor, kernels) -> torch.Tensor:
    """Apply separable 1-D convolutions (``padding='same'``, zero-padded) to a
    channel-first tensor.

    Unlike the rest of this module (channel-last ``(B, *spatial, dim)``),
    this operates on the plain image convention ``(B, C, *spatial)`` used by
    :class:`antstorch.syn.core.losses.BoxLNCCLoss` — each entry of
    ``kernels`` is applied along one spatial axis in turn via a reshaped
    ``F.conv1d``. Supports 2-D ``(B, C, H, W)`` and 3-D ``(B, C, D, H, W)``.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, C, *spatial)``.
    kernels : sequence of torch.Tensor or None
        One 1-D kernel per spatial axis (``len(kernels) == len(spatial)``);
        ``None`` (or a length-1 kernel equal to ``1.0``) skips that axis.

    Returns
    -------
    torch.Tensor
        Filtered tensor, same shape as ``x``.
    """
    spatial_dims = len(kernels)
    for d in range(spatial_dims):
        k = kernels[d]
        if k is None:
            continue
        k = k.view(1, 1, -1)
        if k.shape[-1] == 1 and k.squeeze() == 1.0:
            continue
        if spatial_dims == 3:
            if d == 0:   # Depth (dim 2)
                B, C, D, H, W = x.shape
                x = x.permute(0, 1, 3, 4, 2).reshape(B * C * H * W, 1, D)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, H, W, D).permute(0, 1, 4, 2, 3)
            elif d == 1:  # Height (dim 3)
                B, C, D, H, W = x.shape
                x = x.permute(0, 1, 2, 4, 3).reshape(B * C * D * W, 1, H)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, D, W, H).permute(0, 1, 2, 4, 3)
            elif d == 2:  # Width (dim 4)
                B, C, D, H, W = x.shape
                x = x.reshape(B * C * D * H, 1, W)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, D, H, W)
        elif spatial_dims == 2:
            if d == 0:   # Height (dim 2)
                B, C, H, W = x.shape
                x = x.permute(0, 1, 3, 2).reshape(B * C * W, 1, H)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, W, H).permute(0, 1, 3, 2)
            elif d == 1:  # Width (dim 3)
                B, C, H, W = x.shape
                x = x.reshape(B * C * H, 1, W)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, H, W)
    return x


def gaussian_blur_replicate(image: torch.Tensor, sigmas) -> torch.Tensor:
    """Separable Gaussian blur of a channels-first image with replicate padding.

    ``image`` is ``(B, C, *spatial)``; ``sigmas`` gives one voxel-unit sigma
    per spatial axis, in the same (PyTorch) axis order. Each 1-D kernel has
    radius ``max(1, int(3 * sigma + 0.5))`` and unit sum. The image is
    replicate-padded once on every axis, then filtered along each axis in
    turn as a weighted sum of shifted slices: this equals the dense N-D
    Gaussian on the same padded image (a separable kernel) up to
    floating-point rounding, at O(radius) instead of O(radius^dim) cost per
    voxel, and uses no convolution backend (conv2d/conv3d with 1-D kernels
    gave wrong results on macOS CPU).
    """
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if len(sigmas) != image.ndim - 2:
        raise ValueError("gaussian_blur_replicate needs one sigma per spatial axis")
    radii = [max(1, int(3.0 * sigma + 0.5)) for sigma in sigmas]
    padding = tuple(value for radius in reversed(radii) for value in (radius, radius))
    blurred = F.pad(image, padding, mode="replicate")
    for axis, (sigma, radius) in enumerate(zip(sigmas, radii)):
        coordinate = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
        kernel = torch.exp(-0.5 * (coordinate / sigma).square())
        kernel = kernel / kernel.sum()
        spatial_axis = 2 + axis
        length = blurred.shape[spatial_axis] - 2 * radius
        filtered = blurred.narrow(spatial_axis, 0, length) * kernel[0]
        for offset in range(1, 2 * radius + 1):
            filtered = filtered + blurred.narrow(spatial_axis, offset, length) * kernel[offset]
        blurred = filtered
    return blurred


def get_boundary_mask(spatial, device, dtype, rim_size=1):
    """Build a mask that is zero on the domain boundary and one in the interior.

    Parameters
    ----------
    spatial : sequence of int
        Spatial shape (excluding batch and channel dimensions).
    device, dtype : torch.device, torch.dtype
        Target tensor device/dtype.
    rim_size : int
        Thickness (in voxels) of the zeroed boundary rim on each face.

    Returns
    -------
    torch.Tensor
        Mask of shape ``(1, *spatial, 1)``.
    """
    boundary_mask = torch.ones((1, *spatial, 1), device=device, dtype=dtype)
    for i in range(len(spatial)):
        slices = [slice(None)] * boundary_mask.ndim
        slices[i + 1] = slice(0, rim_size)
        boundary_mask[tuple(slices)] = 0
        slices[i + 1] = slice(-rim_size, None)
        boundary_mask[tuple(slices)] = 0
    return boundary_mask


# ---------------------------------------------------------------------------
# Ported from syntx v6.44 (core/smoothing.py): compact erf Gaussian and the
# B-spline field smoother used by the regularizer registry (core/regularizers.py).
# ---------------------------------------------------------------------------

def gaussian_1d_compact(
    sigma: float,
    truncated: float = 2.0,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """1-D Gaussian kernel integrated over unit bins (erf differences), normalised to sum 1.

    Radius = round(max(sigma * truncated, 0.5)) (at least 1 tap each side), so the kernel has
    2 * radius + 1 taps. ``sigma`` in voxels. Returns a 1-D tensor, or None if sigma <= 0.
    """
    if sigma <= 0.0:
        return None
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=dtype, device=device)
    t = 0.70710678 / float(sigma)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf()).clamp(min=0)
    return out / out.sum()


def _resolve_sigmas(sigma, spacing, sigma_mode, num_spatial):
    """Per-axis sigmas (voxels, tensor order) for the Gaussian filters; ValueError for an
    unknown ``sigma_mode``, a wrong-length sigma sequence, a sequence with
    ``sigma_mode='physical'``, a non-positive spacing, or a ``spacing`` that would be
    ignored (voxel mode)."""
    if sigma_mode not in ('voxel', 'physical'):
        raise ValueError(f"sigma_mode must be 'voxel' or 'physical', got {sigma_mode!r}")
    if isinstance(sigma, (tuple, list)):
        if sigma_mode == 'physical':
            raise ValueError("a per-axis sigma sequence is in voxels: use sigma_mode='voxel'")
        if len(sigma) != num_spatial:
            raise ValueError(f"need one sigma per spatial axis ({num_spatial}), got {len(sigma)}")
        return [float(x) for x in sigma]
    if sigma_mode == 'physical':
        if spacing is None:
            raise ValueError("sigma_mode='physical' needs spacing")
        if any(not float(sp) > 0 for sp in spacing):
            raise ValueError(f"spacing must be positive, got {list(spacing)}")
        return [float(np.clip(float(sigma) / sp, 0.5, 10.0)) for sp in tuple(reversed(spacing))]
    if spacing is not None:
        raise ValueError("spacing is only used with sigma_mode='physical' (sigma in mm); in voxel "
                         "mode it would be ignored -- drop it or pass sigma_mode='physical'")
    return [float(sigma)] * num_spatial


def fast_separable_gaussian_filter(
    grid: torch.Tensor,
    sigma: Union[float, Sequence[float]],
    spacing: Optional[Sequence[float]] = None,
    sigma_mode: str = 'voxel',
    truncated: float = 2.0,
) -> torch.Tensor:
    """
    Separable Gaussian smoothing of a channel-last field with compact erf kernels
    (``gaussian_1d_compact``) and ZERO padding, so values are pulled toward 0 near the border.

    Parameters
    ----------
    grid : Tensor (B, *spatial, C), 2-D or 3-D spatial, any C.
    sigma : float or sequence
        Voxels. A sequence gives one sigma per spatial axis in tensor order (z, y, x) and is
        always taken in voxels.
    spacing : sequence, optional
        ANTs (x, y, z) order, required by and only allowed with ``sigma_mode='physical'`` and a
        scalar sigma: per-axis sigma = sigma / spacing, clipped to [0.5, 10] voxels. (Invalid
        combinations raise ValueError.)
    sigma_mode : {'voxel', 'physical'}, default 'voxel'
    truncated : float, default 2.0
        Kernel radius in sigmas.

    Returns
    -------
    Tensor of the input shape (contiguous); ``grid`` itself if every sigma <= 0. Axes with
    sigma <= 0 are not filtered.
    """
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)

    sigma_list = _resolve_sigmas(sigma, spacing, sigma_mode, num_spatial)

    if all(s <= 0.0 for s in sigma_list):
        return grid

    v = torch.movedim(grid, -1, 1)
    kernels = [gaussian_1d_compact(s, truncated=truncated, device=device, dtype=dtype) for s in sigma_list]
    v_filtered = separable_1d_filter(v, kernels)
    return torch.movedim(v_filtered, 1, -1).contiguous()


def smooth_displacement_field_bspline(
    field: torch.Tensor,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    mesh_size: Optional[Union[int, Sequence[int]]] = None,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    fluid_sigma: Optional[float] = None,
    enforce_stationary_boundary: bool = False,
    order: int = 3,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
) -> torch.Tensor:
    """Smooth a displacement / velocity field by a B-spline least-squares fit (``antstorch.bspline_flows``).

    Fits a B-spline control lattice to the dense field
    (``antstorch.bspline_flows.fit_bspline_displacement_field``, one fitting level) and
    evaluates it back on the grid -- the BSplineSyN-style regulariser. Each batch item is fitted
    separately.

    Parameters
    ----------
    field : torch.Tensor
        (B, *spatial, dim) or (*spatial, dim); spatial axes always in tensor order (z, y, x).
    spacing, origin : sequence of float, optional
        Grid geometry, in the order given by ``coord_convention``. Default 1 and 0 per axis.
    mesh_size : int or sequence of int, optional
        B-spline spans per axis (``coord_convention`` order); an int applies to all axes.
    spline_distance : float or sequence of float, optional
        Knot spacing in spacing units; mesh = ceil(physical extent / distance) per axis
        (``mesh_size_for_spline_distance``). A sequence follows ``coord_convention`` like
        ``spacing`` / ``mesh_size`` (reversed internally for 'zyx').
    fluid_sigma : float, optional
        Used only if neither of the above is given and > 0: spline distance =
        max(4 * fluid_sigma, 12).
    enforce_stationary_boundary : bool, default False
        Passed to the fit: pin the field to zero at the domain boundary.
    order : int, default 3
        Spline order.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Order of the vector components and of ``spacing`` / ``origin`` / sequence
        ``mesh_size``: 'xyz' = ANTs (x, y, z), 'zyx' = tensor order (reversed internally).

    Priority: ``spline_distance``, then ``mesh_size``, then ``fluid_sigma``, else 6 spans per
    axis; every axis gets at least 2 spans.

    Returns
    -------
    torch.Tensor
        Smoothed field of the input shape (dtype / device as returned by antstorch).
    """
    from ...bspline_flows import (
        fit_bspline_displacement_field,
        ImageDomain,
        mesh_size_for_spline_distance,
    )

    d = field.shape[-1]
    if field.dim() == d + 1:
        unbatched = True
        v = field.unsqueeze(0)
    elif field.dim() == d + 2:
        unbatched = False
        v = field
    else:
        raise ValueError(
            f"Expected field of dimension {d + 1} (*spatial, {d}) or "
            f"{d + 2} (B, *spatial, {d}), got {field.dim()}"
        )

    B = v.shape[0]
    spatial_shape = v.shape[1:-1]
    size_itk = tuple(reversed(spatial_shape))

    if spacing is not None:
        if coord_convention == 'zyx':
            spacing_itk = tuple(reversed(spacing))
        else:
            spacing_itk = tuple(float(s) for s in spacing)
    else:
        spacing_itk = (1.0,) * d

    if origin is not None:
        if coord_convention == 'zyx':
            origin_itk = tuple(reversed(origin))
        else:
            origin_itk = tuple(float(o) for o in origin)
    else:
        origin_itk = (0.0,) * d

    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        if coord_convention == 'zyx' and not isinstance(spline_distance, (int, float)):
            spline_distance = tuple(reversed(tuple(spline_distance)))
        mesh_size_itk = mesh_size_for_spline_distance(domain, spline_distance)
    elif mesh_size is not None:
        if isinstance(mesh_size, int):
            mesh_size_itk = (int(mesh_size),) * d
        else:
            mesh_size_itk = tuple(reversed(mesh_size)) if coord_convention == 'zyx' else tuple(mesh_size)
    elif fluid_sigma is not None and float(fluid_sigma) > 0:
        eff_dist = max(4.0 * float(fluid_sigma), 12.0)
        mesh_size_itk = mesh_size_for_spline_distance(domain, eff_dist)
    else:
        mesh_size_itk = (6,) * d

    # Ensure at least 2 intervals per axis for cubic splines
    mesh_size_itk = tuple(max(2, int(m)) for m in mesh_size_itk)

    outs = []
    for b in range(B):
        vb = v[b : b + 1]  # (1, *spatial, d)
        # Convert to channel-first (1, d, *spatial)
        if coord_convention == 'zyx':
            vb_itk = vb.movedim(-1, 1).flip(1)
        else:
            vb_itk = vb.movedim(-1, 1)

        sm = fit_bspline_displacement_field(
            displacement_field=vb_itk,
            domain=domain,
            number_of_fitting_levels=1,
            mesh_size=mesh_size_itk,
            spline_order=order,
            enforce_stationary_boundary=enforce_stationary_boundary,
        )

        if coord_convention == 'zyx':
            sm_syntx = sm.flip(1).movedim(1, -1)
        else:
            sm_syntx = sm.movedim(1, -1)

        outs.append(sm_syntx)

    res = torch.cat(outs, dim=0) if B > 1 else outs[0]
    return res.squeeze(0) if unbatched else res


# Ported from syntx v6.44: in-place shifted-slice 1-D blur (its CPU 3-D / MPS large-kernel smoothing path).
def _shifted_slice_blur_1d_inplace(
    x: torch.Tensor,
    sa: int,
    weights: Sequence[float],
    eff_mode: str,
) -> torch.Tensor:
    """1-D symmetric shifted-slice Gaussian filter along axis ``sa`` using in-place ``.add_``.

    Callers must ensure ``not (torch.is_grad_enabled() and x.requires_grad)`` and
    ``eff_mode in ('replicate', 'constant')``. Allocates zero padded buffers and avoids 5-D
    ``F.pad`` and ``im2col`` workspaces.
    """
    pad = len(weights) // 2
    length = x.shape[sa]
    replicate = (eff_mode == 'replicate')
    acc = x * float(weights[pad])
    first = x.narrow(sa, 0, 1) if replicate else None
    last = x.narrow(sa, length - 1, 1) if replicate else None
    for d in range(1, pad + 1):
        w = float(weights[pad + d])
        if d < length:
            acc.narrow(sa, d, length - d).add_(x.narrow(sa, 0, length - d), alpha=w)
            acc.narrow(sa, 0, length - d).add_(x.narrow(sa, d, length - d), alpha=w)
            if replicate:
                acc.narrow(sa, 0, d).add_(first, alpha=w)
                acc.narrow(sa, length - d, d).add_(last, alpha=w)
        elif replicate:
            acc.add_(first, alpha=w)
            acc.add_(last, alpha=w)
    return acc
