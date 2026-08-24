"""Tensor and image intensity normalization helpers shared across ``antstorch.syn``.

Ported from ``syntx.core.utils`` (PyTorch backend only).
"""

import torch


def normalize_tensor(
    tensor: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8,
    p_min: float = 1.0,
    p_max: float = 99.0,
    dim=None,
    keepdim: bool = True
) -> torch.Tensor:
    """Normalize a PyTorch tensor using the specified strategy.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (any spatial dimension).
    method : str
        Normalization strategy:

        - ``'minmax'``: rescales values linearly to ``[0, 1]``.
        - ``'zscore'``: subtracts the mean and divides by the standard
          deviation (zero-mean, unit-variance).
        - ``'robust'`` / ``'percentile'``: rescales between the ``p_min`` and
          ``p_max`` percentiles and clamps to ``[0, 1]``.
        - ``'l2'`` / ``'unit_norm'``: scales the tensor by its L2 norm.
        - ``'l1'`` / ``'unit_sum'``: scales the tensor by its L1 norm.
        - ``'sigmoid'``: applies a logistic sigmoid transformation.
    eps : float
        Numerical stability floor to prevent division by zero.
    p_min, p_max : float
        Lower/upper percentile thresholds for ``'robust'`` scaling.
    dim : int or tuple of int, optional
        Dimension(s) over which to compute statistics. If ``None``,
        statistics are computed globally over all elements.
    keepdim : bool
        Retain reduced dimensions when ``dim`` is specified.

    Returns
    -------
    torch.Tensor
        Normalized tensor with the same shape and dtype as ``tensor``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    method = method.lower().strip()

    if method in ('minmax', '01'):
        if dim is None:
            t_min = tensor.min()
            t_max = tensor.max()
        else:
            t_min = tensor.amin(dim=dim, keepdim=keepdim)
            t_max = tensor.amax(dim=dim, keepdim=keepdim)
        return (tensor - t_min) / (t_max - t_min + eps)

    elif method in ('zscore', 'standard'):
        if dim is None:
            t_mean = tensor.mean()
            t_std = tensor.std(unbiased=False)
        else:
            t_mean = tensor.mean(dim=dim, keepdim=keepdim)
            t_std = tensor.std(dim=dim, keepdim=keepdim, unbiased=False)
        return (tensor - t_mean) / (t_std + eps)

    elif method in ('robust', 'percentile'):
        if dim is None:
            q_min = torch.quantile(tensor.float(), p_min / 100.0).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0).to(tensor.dtype)
        else:
            q_min = torch.quantile(tensor.float(), p_min / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
        res = (tensor - q_min) / (q_max - q_min + eps)
        return torch.clamp(res, 0.0, 1.0)

    elif method in ('l2', 'unit_norm'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=2)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('l1', 'unit_sum'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=1)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=1, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('sigmoid', 'logistic'):
        return torch.sigmoid(tensor)

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            "Options: 'minmax', 'zscore', 'robust', 'l2', 'l1', 'sigmoid'."
        )


def auto_select_intensity_percentiles(
    image,
    num_bins: int = 32,
    p_low_candidates: tuple = (0.5, 1.0, 2.0, 3.0, 5.0),
    p_high_candidates: tuple = (95.0, 97.0, 98.0, 99.0, 99.5),
    saturation_weight: float = 0.5
):
    """Select entropy-optimal intensity clipping percentiles ``(p_low, p_high)``.

    Maximizes marginal histogram Shannon entropy across Parzen bins with a
    boundary-saturation penalty, over a small grid of candidate percentiles.

    Parameters
    ----------
    image : ants.ANTsImage or np.ndarray
        Input image.
    num_bins : int
        Number of histogram bins (default: 32, matching Mattes MI).
    p_low_candidates, p_high_candidates : tuple of float
        Candidate lower/upper percentile thresholds.
    saturation_weight : float
        Penalty weight for voxels saturating in the boundary bins.

    Returns
    -------
    tuple of (float, float)
        The selected ``(p_low, p_high)`` percentiles.
    """
    import numpy as np
    arr = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    pos = arr[arr > 0]
    if len(pos) < 100:
        return (2.0, 98.0)

    # Subsample if large, for sub-millisecond evaluation.
    if len(pos) > 100000:
        stride = len(pos) // 100000
        sample_vox = pos[::stride]
    else:
        sample_vox = pos

    low_vals = np.percentile(sample_vox, p_low_candidates)
    high_vals = np.percentile(sample_vox, p_high_candidates)

    best_score = -float('inf')
    best_pair = (2.0, 98.0)

    for i, p_l in enumerate(p_low_candidates):
        v_l = float(low_vals[i])
        for j, p_h in enumerate(p_high_candidates):
            v_h = float(high_vals[j])
            if v_h <= v_l + 1e-4:
                continue

            scaled = np.clip((sample_vox - v_l) / (v_h - v_l), 0.0, 1.0)
            hist, _ = np.histogram(scaled, bins=num_bins, range=(0.0, 1.0))
            p_dist = hist.astype(np.float64) / hist.sum()

            p_active = p_dist[p_dist > 0]
            entropy = -np.sum(p_active * np.log2(p_active))
            saturation = p_dist[0] + p_dist[-1]

            score = entropy - saturation_weight * saturation
            if score > best_score:
                best_score = score
                best_pair = (float(p_l), float(p_h))

    return best_pair


def normalize_image(
    image,
    method: str = 'auto',
    p_min: float = 2.0,
    p_max: float = 98.0,
    foreground_only: bool = True,
    eps: float = 1e-6
):
    """Normalize an ``ants.ANTsImage`` or NumPy array for registration workflows.

    Parameters
    ----------
    image : ants.ANTsImage or np.ndarray
        Input image to normalize.
    method : str
        Normalization strategy: ``'auto'`` (entropy-optimal percentile
        selection via :func:`auto_select_intensity_percentiles`),
        ``'robust'`` / ``'percentile'`` (fixed ``p_min``/``p_max``),
        ``'minmax'``, or ``'zscore'``.
    p_min, p_max : float
        Lower/upper percentile thresholds (used unless ``method='auto'``).
    foreground_only : bool
        If ``True``, percentile statistics are computed strictly on non-zero
        foreground voxels.
    eps : float
        Numerical stability floor to prevent division by zero.

    Returns
    -------
    ants.ANTsImage or np.ndarray
        Normalized image with foreground intensities scaled to ``[0.0, 1.0]``
        (or z-scored, for ``method='zscore'``), matching the input type.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.
    """
    import numpy as np

    is_ants = hasattr(image, "numpy") and hasattr(image, "new_image_like")
    arr = image.numpy() if is_ants else np.asarray(image)

    method = method.lower().strip()

    if method in ('auto', 'entropy'):
        p_min, p_max = auto_select_intensity_percentiles(arr)
        pos = arr[arr > 0] if foreground_only else arr
        if len(pos) > 0:
            q_min = float(np.percentile(pos, p_min))
            q_max = float(np.percentile(pos, p_max))
            if q_max <= q_min + 1e-4:
                q_min = 0.0
                q_max = float(pos.max())
        else:
            q_min = float(arr.min())
            q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('robust', 'percentile'):
        pos = arr[arr > 0] if foreground_only else arr
        if len(pos) > 0:
            q_min = float(np.percentile(pos, p_min))
            q_max = float(np.percentile(pos, p_max))
            if q_max <= q_min + 1e-4:
                q_min = 0.0
                q_max = float(pos.max())
        else:
            q_min = float(arr.min())
            q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('minmax', '01'):
        q_min = float(arr.min())
        q_max = float(arr.max())
        norm_arr = np.clip((arr - q_min) / (q_max - q_min + eps), 0.0, 1.0).astype(np.float32)

    elif method in ('zscore', 'standard'):
        pos = arr[arr > 0] if foreground_only else arr
        mean = float(pos.mean()) if len(pos) > 0 else float(arr.mean())
        std = float(pos.std()) if len(pos) > 0 else float(arr.std())
        norm_arr = ((arr - mean) / (std + eps)).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            "Options: 'auto', 'robust', 'minmax', 'zscore'."
        )

    return image.new_image_like(norm_arr) if is_ants else norm_arr
