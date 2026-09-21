"""Similarity losses for intensity-based registration.

Ported from ``syntx.core.losses`` (PyTorch backend only): local normalized
cross-correlation (LNCC) with two autograd strategies, Mattes mutual
information via Parzen (cubic B-spline) windowing, a native sliding
box-filter LNCC/CC^2 (:func:`box_cc2_loss_nd`), distance-transform
similarity (:func:`distance_transform_loss`), and soft Dice
(:func:`soft_dice_loss_nd`).
"""

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


class AnalyticalLNCC(torch.autograd.Function):
    """Local NCC (correlation coefficient, not its square) with a hand-derived backward pass.

    Computes the forward value ``CC = cov(I, J) / sqrt(var(I) * var(J))``
    identical to the autograd path in :func:`local_ncc_loss_nd`
    (``squared=False``), but implements ``backward()`` manually so PyTorch
    never builds a memory-heavy autograd graph through ``avg_pool2d``/
    ``avg_pool3d``. This is as fast as :class:`ANTsPseudoLNCC` on backends
    where autograd through pooling is expensive (e.g. Apple MPS), while
    optimizing the true CC loss landscape rather than ITK's CC^2
    pseudo-derivative.

    The analytical gradient of ``-mean(CC)`` with respect to the center
    pixel ``J_c`` (symmetric for ``I_c``) is::

        dCC/dJ_c = (1/N) * 1/sqrt(var_I * var_J) * (I_c - CC * J_c)

    where ``I_c``, ``J_c`` are mean-subtracted center-pixel intensities and
    ``N`` is the window volume.
    """

    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim

        if dim == 2:
            pool_fn = F.avg_pool2d
        elif dim == 3:
            pool_fn = F.avg_pool3d
        else:
            raise ValueError(f"Only 2-D and 3-D images are supported, got {dim}D.")

        def box_filter(x):
            return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)

        I_mean = box_filter(I)
        J_mean = box_filter(J)

        F_centered = I - I_mean
        M_centered = J - J_mean

        I_var = torch.clamp(box_filter(F_centered ** 2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered ** 2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        denom = torch.sqrt(safe_I_var * safe_J_var) + 1e-6
        cc_raw = IJ_cov / denom
        cc = torch.clamp(cc_raw, min=-1.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, cc, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window

        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc)
            ctx.active = None

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, cc, safe_I_var, safe_J_var, mask = ctx.saved_tensors

        inv_denom = 1.0 / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)

        # dCC/dJ_c = (1/N) / sqrt(sFF * sMM) * (F_c - CC * M_c)
        # dCC/dI_c = (1/N) / sqrt(sFF * sMM) * (M_c - CC * F_c)
        # Loss is -CC, so negate.
        scale = -(1.0 / ctx.N_window) * inv_denom

        grad_J = scale * (F_centered - cc * M_centered)
        grad_I = scale * (M_centered - cc * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        return grad_I * grad_output, grad_J * grad_output, None, None


class ANTsPseudoLNCC(torch.autograd.Function):
    """Squared local NCC with ITK's hand-derived pseudo-gradient.

    Matches ``itk::ANTSNeighborhoodCorrelationImageToImageMetricv4``: the
    forward value is ``CC^2 = cov(I, J)^2 / (var(I) * var(J))``, and the
    backward pass uses ITK's analytical pseudo-derivative approximation
    rather than differentiating the pooling operations directly.
    """

    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim

        if dim == 2:
            pool_fn = F.avg_pool2d
        elif dim == 3:
            pool_fn = F.avg_pool3d
        else:
            raise ValueError(f"Only 2-D and 3-D images are supported, got {dim}D.")

        def box_filter(x):
            return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)

        I_mean = box_filter(I)
        J_mean = box_filter(J)

        F_centered = I - I_mean
        M_centered = J - J_mean

        I_var = torch.clamp(box_filter(F_centered ** 2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered ** 2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        # ITK uses CC^2: localCC = sFixedMoving * sFixedMoving / (sFixedFixed * sMovingMoving).
        cc2_raw = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc2 = torch.clamp(cc2_raw, min=0.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window

        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc2 * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc2)
            ctx.active = None

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask = ctx.saved_tensors

        s_FM = IJ_cov
        s_FF = safe_I_var
        s_MM = safe_J_var

        sFF_sMM = s_FF * s_MM + 1e-8

        # ITK's pseudo-derivative of +CC^2 w.r.t. the moving center pixel M_c:
        #   2/N * cov / (var_F * var_M) * (F_c - cov / var_M * M_c)
        # By symmetry, w.r.t. the fixed center pixel F_c:
        #   2/N * cov / (var_F * var_M) * (M_c - cov / var_F * F_c)
        # The loss is -CC^2, so its gradient is the negative of this.
        grad_factor = -2.0 * (1.0 / ctx.N_window) * (s_FM / sFF_sMM)

        grad_J = grad_factor * (F_centered - (s_FM / (s_MM + 1e-8)) * M_centered)
        grad_I = grad_factor * (M_centered - (s_FM / (s_FF + 1e-8)) * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        return grad_I * grad_output, grad_J * grad_output, None, None


def local_ncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 9,
    use_ants_pseudo_gradient: bool = False,
    squared: bool = False
) -> torch.Tensor:
    r"""Local Normalized Cross-Correlation (LNCC) loss between N-D images.

    Evaluates the local mean, variance, and covariance of ``I`` and ``J``
    over a sliding box window of size ``window_size``, with a variance floor
    (``max(var, 1e-6)``) to avoid ``1/var`` gradient spikes in flat-intensity
    or zero-padded background regions, and Cauchy-Schwarz clamping of the
    correlation coefficient to ``[-1, 1]`` to eliminate floating-point
    round-off overflow near sharp boundary edges.

    Parameters
    ----------
    I, J : torch.Tensor
        Image tensors of shape ``(B, 1, *spatial)``.
    mask : torch.Tensor, optional
        Binary mask of shape ``(B, 1, *spatial)`` identifying active
        evaluation voxels.
    window_size : int
        Sliding box-filter window size in voxels. Reduced (and forced odd)
        automatically if larger than the smallest spatial dimension.
    use_ants_pseudo_gradient : bool
        If ``True``, use ITK's analytical pseudo-gradient autograd function
        (:class:`ANTsPseudoLNCC`, implicitly optimizing CC^2) or
        :class:`AnalyticalLNCC` (true CC, same speed) instead of building an
        autograd graph through pooling.
    squared : bool
        If ``True``, optimize squared LNCC (CC^2) instead of CC — a
        multi-modal-friendlier metric.

    Returns
    -------
    torch.Tensor
        Scalar negative LNCC loss, in ``[-1.0, 0.0]`` (``-1.0`` is perfect
        alignment).

    Raises
    ------
    ValueError
        If ``I``/``J`` are not 2-D or 3-D.
    """
    dim = I.dim() - 2

    min_spatial = min(I.shape[2:])
    if window_size > min_spatial:
        window_size = min_spatial
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)

    if use_ants_pseudo_gradient and squared:
        return ANTsPseudoLNCC.apply(I, J, mask, window_size)
    elif use_ants_pseudo_gradient and not squared:
        return AnalyticalLNCC.apply(I, J, mask, window_size)

    pad = window_size // 2

    if dim == 2:
        pool_fn = F.avg_pool2d
    elif dim == 3:
        pool_fn = F.avg_pool3d
    else:
        raise ValueError(f"Only 2-D and 3-D images are supported, got {dim}D.")

    def box_filter(x):
        return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)

    I_mean = box_filter(I)
    J_mean = box_filter(J)

    I_var = torch.clamp(box_filter((I - I_mean) ** 2), min=0.0)
    J_var = torch.clamp(box_filter((J - J_mean) ** 2), min=0.0)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))

    var_floor = 1e-6
    safe_I_var = torch.clamp(I_var, min=var_floor)
    safe_J_var = torch.clamp(J_var, min=var_floor)

    if squared:
        cc_metric = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc_metric = torch.clamp(cc_metric, min=0.0, max=1.0)
    else:
        cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)
        cc_metric = torch.clamp(cc_raw, min=-1.0, max=1.0)

    if mask is not None:
        active_mask_float = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(dtype=I.dtype)
        return -torch.sum(cc_metric * active_mask_float) / (torch.sum(active_mask_float) + 1e-8)
    else:
        return -torch.mean(cc_metric)


def b_spline_3(x: torch.Tensor) -> torch.Tensor:
    """Cubic (3rd-order) B-spline kernel, used as the Parzen window in :func:`mattes_mi_loss_core`."""
    abs_x = torch.abs(x)
    y1 = (2.0 / 3.0) - abs_x ** 2 + 0.5 * abs_x ** 3
    y2 = (1.0 / 6.0) * (2.0 - abs_x) ** 3
    return torch.where(abs_x < 1.0, y1, torch.where(abs_x < 2.0, y2, torch.zeros_like(x)))


def _parzen_joint_histogram(w_x: torch.Tensor, w_y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Joint Parzen histogram ``H = w_x^T w_y`` for ``[N, B]`` B-spline weight matrices.

    Accumulated as a batch of ``chunk``-row blocks (``torch.bmm``) followed
    by a fixed-order sum, rather than one large ``matmul``.

    Ported from ``syntx.core.losses``: on Apple MPS, ``w_x.t() @ w_y`` with
    ``N`` on the order of ``1e5`` was found to be non-deterministic and, for
    concentrated histograms, wrong by up to ~15% (an affine solver produced
    different transforms on every first call). Blocked ``bmm`` with
    ``chunk=4096`` is bitwise deterministic across allocations and much
    closer to a float64 reference. Differentiable.
    """
    n, nb = w_x.shape
    pad = (-n) % chunk
    if pad:
        z = torch.zeros(pad, nb, dtype=w_x.dtype, device=w_x.device)
        w_x = torch.cat([w_x, z])
        w_y = torch.cat([w_y, z])
    a = w_x.view(-1, chunk, nb).transpose(1, 2)  # [M, B, chunk]
    b = w_y.view(-1, chunk, nb)                  # [M, chunk, B]
    return torch.bmm(a, b).sum(dim=0)             # [B, B]


def parzen_weights(
    v: torch.Tensor,
    num_bins: int = 32,
    min_val: float = -1.0,
    max_val: float = 1.0,
    pad: float = 2.0,
) -> torch.Tensor:
    """Cubic B-spline Parzen weights ``[N, num_bins]`` of a flat intensity vector.

    ``v`` is clamped/scaled to ``[min_val, max_val]`` and mapped onto a
    boundary-padded bin axis (``pad`` bins of margin on each side) so the
    weights form a strict partition of unity everywhere in range --
    avoiding the artificial boundary forces an unpadded bin axis produces
    near the histogram's edges. Cache the result for an image whose samples
    do not change across calls (e.g. the fixed image across optimization
    iterations).
    """
    v = torch.nan_to_num(torch.clamp(v.float(), min_val, max_val), nan=0.0)
    u_min, u_max = pad, float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)
    bins = torch.arange(num_bins, device=v.device, dtype=torch.float32).unsqueeze(0)
    return b_spline_3(u_min + (v.view(-1, 1) - min_val) * scale - bins)


def mattes_mi_from_weights(w_x: torch.Tensor, w_y: torch.Tensor) -> torch.Tensor:
    """Negative Mattes MI from Parzen weight matrices (deterministic blocked histogram)."""
    joint_hist = _parzen_joint_histogram(w_x, w_y)
    pxy = joint_hist / (joint_hist.sum() + 1e-8)
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)))


def mattes_mi_loss_core(
    I, J, mask=None, num_bins=32, min_val=-1.0, max_val=1.0, sampling_percentage=None,
    fixed_weights: Optional[torch.Tensor] = None,
):
    """Differentiable Mattes Mutual Information, via cubic B-spline Parzen windowing.

    Ported from ``syntx.core.losses`` (upgraded from this module's earlier,
    simpler version -- see git history): the joint histogram now uses a
    boundary-padded Parzen bin axis (:func:`parzen_weights`, strict
    partition of unity) accumulated via a deterministic blocked ``bmm``
    (:func:`_parzen_joint_histogram`) instead of an unpadded bin axis and a
    single large ``matmul`` -- more numerically stable on MPS/CUDA at large
    voxel counts, at the cost of a small shift in absolute MI values versus
    the previous implementation (the *ranking* of candidates under
    optimization is unaffected).

    Parameters
    ----------
    I, J : torch.Tensor
        Intensity tensors already scaled to ``[min_val, max_val]``.
    mask : torch.Tensor, optional
        Boolean or ``{0, 1}``-valued mask selecting which voxels contribute.
    num_bins : int
        Number of Parzen histogram bins.
    min_val, max_val : float
        Intensity range covered by the histogram bins.
    sampling_percentage : float, optional
        If given and ``< 1.0``, subsamples voxels with this fraction (via a
        fixed stride) before building the joint histogram.
    fixed_weights : torch.Tensor, optional
        Precomputed :func:`parzen_weights` of ``J`` (the fixed image in a
        registration loop, whose samples/weights do not change across
        iterations) -- valid only when its row count already matches ``I``'s
        selected/subsampled voxel count; halves the per-iteration cost when
        reused across many calls.

    Returns
    -------
    torch.Tensor
        Scalar negative mutual information (for minimization). Zero (with
        ``requires_grad=True``) if no voxels are selected.
    """
    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if sampling_percentage is not None and sampling_percentage < 1.0:
        stride = max(1, int(1.0 / sampling_percentage))
        x = x[::stride].contiguous()
        y = y[::stride].contiguous()

    if x.numel() == 0:
        return torch.tensor(0.0, device=I.device, requires_grad=True)

    w_x = parzen_weights(x, num_bins, min_val, max_val)
    w_y = fixed_weights if (fixed_weights is not None and fixed_weights.shape[0] == w_x.shape[0]) \
        else parzen_weights(y, num_bins, min_val, max_val)
    return mattes_mi_from_weights(w_x, w_y)


def mattes_mi_loss_nd(
    I, J, mask=None, num_bins=32, sampling_percentage=None, auto_mask=True,
    fixed_range=None, fixed_weights=None,
):
    """N-dimensional Mattes Mutual Information loss.

    Rescales ``I`` and ``J`` to ``[-1, 1]`` internally and, when
    ``auto_mask`` is enabled, restricts the joint histogram to voxels where
    either image is non-trivially non-zero — this excludes zero-padded
    background from contaminating the intensity distributions.

    Parameters
    ----------
    I, J : torch.Tensor
        Image tensors of any matching shape.
    mask : torch.Tensor, optional
        Additional mask combined (via logical AND) with the foreground mask.
    num_bins : int
        Number of Parzen histogram bins.
    sampling_percentage : float, optional
        Passed through to :func:`mattes_mi_loss_core`.
    auto_mask : bool
        If ``True``, exclude voxels where both ``|I|`` and ``|J|`` are
        below ``0.01``.
    fixed_range : None, ``(min, max)``, or ``((min_I, max_I), (min_J, max_J))``
        Intensity bounds used for the histogram axes. Default (``None``)
        recomputes the bounds from the *current* masked voxels of each
        image, so the axes move with the transform and MI values are not
        comparable across candidates evaluated at different parameters.
        Pass fixed bounds (ITK behavior) when comparing candidates during
        optimization, e.g. ``fixed_range=(0.0, 1.0)`` for
        foreground-normalized images.
    fixed_weights : torch.Tensor, optional
        Precomputed ``parzen_weights`` of the (masked, subsampled, scaled)
        ``J`` image -- valid only with ``fixed_range`` and an unchanging
        ``J`` sample; forwarded to :func:`mattes_mi_loss_core`.

    Returns
    -------
    torch.Tensor
        Scalar negative mutual information (for minimization).
    """
    if auto_mask:
        fg_mask = (I.abs() > 0.01) | (J.abs() > 0.01)
        if mask is not None:
            mask = (mask > 0.5) & fg_mask
        else:
            mask = fg_mask

    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if x.numel() == 0:
        return torch.tensor(0.0, device=I.device, requires_grad=True)

    if fixed_range is None:
        min_i, max_i = x.min().detach(), x.max().detach()
        min_j, max_j = y.min().detach(), y.max().detach()
    else:
        fr = fixed_range
        if isinstance(fr[0], (int, float)):
            fr = (fr, fr)
        min_i, max_i = float(fr[0][0]), float(fr[0][1])
        min_j, max_j = float(fr[1][0]), float(fr[1][1])

    x_scaled = (x - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
    y_scaled = (y - min_j) / (max_j - min_j + 1e-8) * 2.0 - 1.0

    return mattes_mi_loss_core(
        x_scaled, y_scaled, mask=None, num_bins=num_bins, min_val=-1.0, max_val=1.0,
        sampling_percentage=sampling_percentage,
        fixed_weights=fixed_weights if fixed_range is not None else None,
    )


class BoxLNCCLoss(torch.nn.Module):
    """Native sliding box-filter (squared) zero-normalized cross-correlation loss.

    Unlike :func:`local_ncc_loss_nd` (which uses ``avg_pool2d``/``avg_pool3d``
    for its box filter), this computes the box sums via
    :func:`antstorch.syn.core.smoothing.separable_1d_filter` -- a
    channel-first separable ``conv1d`` -- and features dual variance floors
    (``smooth_nr=smooth_dr=1e-5``): in a zero-padded or uniform background
    region this evaluates to ``(0 + 1e-5) / (0 + 1e-5) = 1.0``, treating flat
    background as perfectly correlated and suppressing peripheral boundary
    gradient artifacts at image edges.
    """

    def __init__(self, kernel_size: int = 5, smooth_nr: float = 1e-5, smooth_dr: float = 1e-5, squared: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared = squared
        self._target_cache = {}
        self._target_refs = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Sum of squares over a window can overflow/underflow float16 under
        # AMP autocast; disable it for this computation to stay in float32.
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            from .smoothing import separable_1d_filter
            dim = pred_f.dim() - 2
            kernel_vol = float(self.kernel_size ** dim)
            k = torch.ones(self.kernel_size, dtype=torch.float32, device=pred_f.device)
            kernels = [k] * dim

            target_id = id(target)
            version = getattr(target, '_version', 0)
            if not target.requires_grad and target_id in self._target_cache and self._target_cache[target_id][0] == version:
                _, t_sum, t_var = self._target_cache[target_id]
            else:
                t_sum = separable_1d_filter(target_f, kernels)
                t2_sum = separable_1d_filter(target_f * target_f, kernels)
                t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=self.smooth_dr)
                if not target.requires_grad:
                    if len(self._target_cache) >= 4:
                        self._target_cache.clear()
                        self._target_refs.clear()
                    self._target_cache[target_id] = (version, t_sum, t_var)
                    self._target_refs[target_id] = target

            p_sum = separable_1d_filter(pred_f, kernels)
            p2_sum = separable_1d_filter(pred_f * pred_f, kernels)
            tp_sum = separable_1d_filter(target_f * pred_f, kernels)

            cross = tp_sum - p_sum * t_sum / kernel_vol
            p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=self.smooth_dr)

            if self.squared:
                ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
            else:
                ncc = cross / (torch.sqrt(t_var * p_var) + self.smooth_dr)
            return -torch.mean(ncc)


def box_lncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    squared: bool = False,
) -> torch.Tensor:
    """Functional interface for :class:`BoxLNCCLoss` (linear CC if ``squared=False``, CC^2 if ``squared=True``)."""
    loss_fn = BoxLNCCLoss(kernel_size=window_size, smooth_nr=smooth_nr, smooth_dr=smooth_dr, squared=squared)
    return loss_fn(I, J)


def box_cc2_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> torch.Tensor:
    """Functional interface for squared Box-LNCC loss (Box-CC^2). See :class:`BoxLNCCLoss`."""
    return box_lncc_loss_nd(I, J, window_size=window_size, smooth_nr=smooth_nr, smooth_dr=smooth_dr, squared=True)


def compute_soft_distance_transform(
    image: torch.Tensor,
    sigma: float = 3.0,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Compute a soft, differentiable distance transform using the heat method.

    Approximates the Euclidean distance transform differentiably in PyTorch
    via ``D_sigma(x) = -sigma * log(G_sigma * mask)``. When physical spacing
    is provided, the diffusion filter scales by voxel spacing, yielding
    distances in physical millimeter space.

    Parameters
    ----------
    image : torch.Tensor of shape ``(B, C, *spatial)`` or ``(*spatial)``
        Input image or segmentation.
    sigma : float
        Diffusion scale, in physical units (mm) if ``spacing`` is given,
        voxel units otherwise.
    threshold : float, optional
        Foreground binarization threshold; if ``None``, uses a soft mask
        derived from the (peak-normalized) absolute intensity.
    spacing : sequence of float, optional
        Physical voxel spacing in ITK ``(sx, sy[, sz])`` convention.

    Returns
    -------
    torch.Tensor
        Smooth, autograd-differentiable distance field, same shape as
        ``image``.
    """
    orig_shape = image.shape
    if image.dim() == 2:
        img_nd = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        if spacing is not None and len(spacing) == 2:
            img_nd = image.unsqueeze(0)
        else:
            img_nd = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() in (4, 5):
        img_nd = image
    else:
        raise ValueError(f"Unsupported image dimension: {image.dim()}")

    if threshold is not None:
        mask = torch.sigmoid((img_nd - threshold) * 10.0)
    else:
        max_val = torch.amax(img_nd.abs(), dim=tuple(range(2, img_nd.dim())), keepdim=True).clamp_min(1e-6)
        mask = torch.clamp(img_nd.abs() / max_val, 0.0, 1.0)

    from .smoothing import separable_gaussian_filter
    mask_last = mask.movedim(1, -1)
    if spacing is not None:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma, spacing=spacing, sigma_mode='physical')
    else:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma)
    smoothed = smoothed_last.movedim(-1, 1).clamp_min(1e-8)

    dist = -float(sigma) * torch.log(smoothed)
    return dist.view(orig_shape)


def compute_image_distance_transform(
    image: Union[torch.Tensor, np.ndarray, Any],
    threshold: Optional[float] = None,
    tau: Optional[float] = None,
    signed: bool = False,
    sampling_spacing: Optional[Sequence[float]] = None,
    return_ants: bool = False,
) -> Union[torch.Tensor, Any]:
    """Compute an exact Euclidean distance transform (or smooth potential) from an image or mask.

    Distances are computed in physical space (mm); if ``image`` is an
    ``ants.ANTsImage``, its physical spacing is extracted automatically.

    Parameters
    ----------
    image : ants.ANTsImage, torch.Tensor, or np.ndarray
        Input image, segmentation, or edge map, of shape
        ``(B, C, *spatial)``, ``(B, 1, *spatial)``, or ``(*spatial)``.
    threshold : float, optional
        Foreground binarization threshold. If ``None``, uses non-zero
        voxels.
    tau : float, optional
        Bandwidth for the exponential potential ``P(x) = exp(-D(x) / tau)``.
        If ``None``, returns the raw distance.
    signed : bool
        If ``True``, returns a signed distance transform (negative inside,
        positive outside).
    sampling_spacing : sequence of float, optional
        Physical voxel spacing in ITK convention (``sx, sy[, sz]``). If
        ``image`` is an ``ANTsImage`` and this is ``None``, ``image.spacing``
        is used automatically.
    return_ants : bool
        If ``True`` and ``image`` is an ``ANTsImage``, returns an
        ``ANTsImage`` with matching geometry instead of a tensor.

    Returns
    -------
    torch.Tensor or ants.ANTsImage
        Distance transform in physical units (mm).
    """
    is_ants = hasattr(image, 'spacing') and hasattr(image, 'numpy')
    ref_image = image if is_ants else None

    if is_ants:
        if sampling_spacing is None:
            sampling_spacing = tuple(float(s) for s in image.spacing)
        img_np_raw = image.numpy().astype(np.float32)
        device = torch.device('cpu')
        dtype = torch.float32
        orig_shape = img_np_raw.shape
        img_nd = torch.from_numpy(img_np_raw).unsqueeze(0).unsqueeze(0)
        sampling = sampling_spacing
    else:
        if not isinstance(image, torch.Tensor):
            image_t = torch.as_tensor(image, dtype=torch.float32)
        else:
            image_t = image
        device = image_t.device
        dtype = image_t.dtype
        orig_shape = image_t.shape

        if image_t.dim() == 2:
            img_nd = image_t.unsqueeze(0).unsqueeze(0)
        elif image_t.dim() == 3:
            if sampling_spacing is not None and len(sampling_spacing) == 2:
                img_nd = image_t.unsqueeze(0)
            else:
                img_nd = image_t.unsqueeze(0).unsqueeze(0)
        elif image_t.dim() in (4, 5):
            img_nd = image_t
        else:
            raise ValueError(f"Unsupported image dimension: {image_t.dim()}")

        # PyTorch tensors order spatial dims (Z, Y, X) / (Y, X); ITK spacing
        # is ordered (sx, sy[, sz]) -- reverse to match.
        if sampling_spacing is not None:
            sampling = tuple(reversed(sampling_spacing))
        else:
            sampling = None

    B, C = img_nd.shape[:2]
    spatial_shape = img_nd.shape[2:]

    img_np = img_nd.detach().cpu().numpy()
    out_np = np.zeros_like(img_np, dtype=np.float32)

    for b in range(B):
        for c in range(C):
            sl = img_np[b, c]
            if threshold is not None:
                fg = sl > threshold
            else:
                max_v = float(np.max(np.abs(sl)))
                fg = np.abs(sl) > (1e-4 * max_v if max_v > 0 else 1e-4)

            if not np.any(fg):
                diag_span = float(np.sqrt(sum((dim_sz * (sp if sp else 1.0)) ** 2 for dim_sz, sp in zip(spatial_shape, sampling or [1.0] * len(spatial_shape)))))
                edt = np.ones_like(sl, dtype=np.float32) * diag_span
            elif np.all(fg):
                edt = np.zeros_like(sl, dtype=np.float32)
            else:
                if signed:
                    d_out = ndi.distance_transform_edt(~fg, sampling=sampling)
                    d_in = ndi.distance_transform_edt(fg, sampling=sampling)
                    edt = d_out - d_in
                else:
                    edt = ndi.distance_transform_edt(~fg, sampling=sampling)
            out_np[b, c] = edt

    if is_ants and return_ants:
        import ants
        out_res = out_np.reshape(orig_shape)
        if tau is not None:
            if tau <= 0.0:
                raise ValueError(f"tau must be strictly positive, got {tau}")
            out_res = np.exp(-np.abs(out_res) / float(tau))
        return ants.from_numpy(
            out_res.astype(np.float32),
            origin=ref_image.origin,
            spacing=ref_image.spacing,
            direction=ref_image.direction,
        )

    out_t = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    if tau is not None:
        if tau <= 0.0:
            raise ValueError(f"tau must be strictly positive, got {tau}")
        out_t = torch.exp(-torch.abs(out_t) / float(tau))
    return out_t.view(orig_shape)


def distance_transform_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    mode: Literal['potential_lncc', 'potential_mse', 'edt_mse', 'edt_l1', 'sdf_mse'] = 'potential_lncc',
    tau: float = 0.10,
    window_size: int = 9,
    mask: Optional[torch.Tensor] = None,
    is_distance_field: bool = False,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """General distance-transform similarity loss, in physical space.

    Computes distance-transform alignment between two images, segmentation
    masks, or precomputed distance fields in physical space (mm). Usable by
    any registration model, not only intensity-based ones.

    Parameters
    ----------
    I, J : torch.Tensor of shape ``(B, C, *spatial)``
        Input images, segmentations, or (if ``is_distance_field=True``)
        precomputed distance fields.
    mode : {'potential_lncc', 'potential_mse', 'edt_mse', 'edt_l1', 'sdf_mse'}
        Loss formulation:

        - ``'potential_lncc'``: negative LNCC on exponential distance
          potentials ``exp(-D / tau)``.
        - ``'potential_mse'``: MSE on exponential distance potentials.
        - ``'edt_mse'``: MSE on Euclidean distance fields.
        - ``'edt_l1'``: L1 error on Euclidean distance fields.
        - ``'sdf_mse'``: MSE on signed distance fields.
    tau : float
        Decay bandwidth for exponential potentials, in physical units (mm).
    window_size : int
        LNCC window size, used only when ``mode='potential_lncc'``.
    mask : torch.Tensor, optional
        Spatial domain mask.
    is_distance_field : bool
        If ``True``, ``I``/``J`` are already distance fields; if ``False``,
        soft distance transforms are computed from them first.
    threshold : float, optional
        Binarization threshold for ``I``/``J`` when
        ``is_distance_field=False``.
    spacing : sequence of float, optional
        Physical voxel spacing in ITK ``(sx, sy[, sz])`` convention.

    Returns
    -------
    torch.Tensor
        Scalar similarity loss.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported formulations.
    """
    if not is_distance_field:
        D_I = compute_soft_distance_transform(I, sigma=tau * 10.0 if tau else 3.0, threshold=threshold, spacing=spacing)
        D_J = compute_soft_distance_transform(J, sigma=tau * 10.0 if tau else 3.0, threshold=threshold, spacing=spacing)
    else:
        D_I = I
        D_J = J

    if mode == 'potential_lncc':
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        return local_ncc_loss_nd(P_I, P_J, mask=mask, window_size=window_size)
    elif mode == 'potential_mse':
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        if mask is not None:
            return torch.sum(((P_I - P_J) ** 2) * mask) / (mask.sum() + 1e-8)
        return F.mse_loss(P_I, P_J)
    elif mode in ('edt_mse', 'sdf_mse'):
        if mask is not None:
            return torch.sum(((D_I - D_J) ** 2) * mask) / (mask.sum() + 1e-8)
        return F.mse_loss(D_I, D_J)
    elif mode == 'edt_l1':
        if mask is not None:
            return torch.sum(torch.abs(D_I - D_J) * mask) / (mask.sum() + 1e-8)
        return F.l1_loss(D_I, D_J)
    else:
        raise ValueError(f"Unknown distance transform loss mode: '{mode}'")


def soft_dice_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft Dice loss between continuous probability/membership tensors.

    Supports single- or multi-channel tensors, in 2-D or 3-D.

    Parameters
    ----------
    I : torch.Tensor of shape ``(B, C, *spatial)``
        Fixed target probability/membership tensor.
    J : torch.Tensor of shape ``(B, C, *spatial)``
        Moving/deformed probability/membership tensor.
    mask : torch.Tensor, optional
        Spatial domain foreground mask of shape ``(B, 1, *spatial)`` or
        ``(B, C, *spatial)``.
    eps : float
        Safety epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Scalar soft Dice loss (``1.0 - mean_dice``).
    """
    if mask is not None:
        I = I * mask
        J = J * mask

    spatial_dims = tuple(range(2, I.ndim))
    intersection = 2.0 * torch.sum(I * J, dim=spatial_dims)
    cardinality = torch.sum(I ** 2 + J ** 2, dim=spatial_dims) + eps

    dice_per_channel = intersection / cardinality
    return 1.0 - torch.mean(dice_per_channel)
