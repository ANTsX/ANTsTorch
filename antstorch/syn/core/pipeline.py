"""Device selection and image-to-tensor preprocessing utilities.

Ported from ``syntx.core.pipeline`` (PyTorch backend only). These helpers
bridge ``ants.ANTsImage`` inputs to the normalized, permuted tensors that
the rest of ``antstorch.syn.core`` operates on, and provide GPU/MPS memory
hygiene for iterative registration loops.
"""

import gc

import numpy as np


def auto_detect_device(backend='pytorch', requested_device=None):
    """Auto-detect the optimal compute device for a given backend.

    Parameters
    ----------
    backend : {'pytorch', 'jax'}
        Compute backend. Only ``'pytorch'`` is exercised by
        ``antstorch.syn``; ``'jax'`` is accepted for parity with
        ``syntx`` but is otherwise unused here.
    requested_device : str, optional
        If given, returned verbatim (lower-cased) instead of probing
        hardware — lets callers force a specific device.

    Returns
    -------
    str
        One of ``'cuda'``, ``'mps'``, ``'cpu'`` (or ``'jax'`` when
        ``backend='jax'``).
    """
    if requested_device is not None:
        return str(requested_device).lower()

    if backend == 'pytorch':
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    elif backend == 'jax':
        # JAX automatically uses the best available backend.
        return 'jax'
    return 'cpu'


def normalize_and_tensorize(fixed, moving, winsorize_quantiles=None, backend='pytorch', device='cpu'):
    """Winsorize, foreground-normalize, and tensorize a pair of ANTs images.

    Each image is rescaled to ``[0, 1]`` using its own foreground (voxels
    ``> 0``) 2nd-98th percentile range, then converted to a tensor and
    permuted from ITK physical axis order ``(x, y[, z])`` to the
    ``(B, C, z, y, x)`` (or ``(B, C, y, x)`` in 2D) layout used throughout
    ``antstorch.syn.core``.

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
        Fixed and moving images.
    winsorize_quantiles : tuple of float, optional
        Unused placeholder retained for interface parity with
        ``syntx``; the foreground percentiles are currently fixed at
        (2, 98).
    backend : {'pytorch', 'jax'}
        Tensor backend to construct.
    device : str
        Target device for the returned tensors (``backend='pytorch'``
        only).

    Returns
    -------
    tuple of tensor
        ``(I_tensor, J_tensor)`` — normalized, permuted, batched
        single-channel tensors for the fixed and moving images,
        respectively.

    Raises
    ------
    ValueError
        If ``backend`` is neither ``'pytorch'`` nor ``'jax'``.
    """
    fi_np = fixed.numpy()
    mi_np = moving.numpy()

    def _norm_fg(arr):
        pos = arr[arr > 0]
        if len(pos) > 0:
            p02 = float(np.percentile(pos, 2.0))
            p98 = float(np.percentile(pos, 98.0))
            if p98 <= p02 + 1e-4:
                p02 = 0.0
                p98 = float(pos.max())
        else:
            p02 = float(arr.min())
            p98 = float(arr.max())
        return np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)

    fi_norm = _norm_fg(fi_np)
    mi_norm = _norm_fg(mi_np)

    dim = fixed.dimension
    perm = [0, 1] + list(range(dim + 1, 1, -1))

    if backend == 'pytorch':
        import torch
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
    elif backend == 'jax':
        import jax.numpy as jnp
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fi_np.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *mi_np.shape).transpose(perm)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return I_tensor, J_tensor


def cleanup_gpu(device, backend='pytorch'):
    """Free GPU/MPS memory to prevent OOM errors in long registration loops.

    Parameters
    ----------
    device : str or torch.device
        Device whose cache should be cleared; only ``'cuda'`` and
        ``'mps'`` devices trigger an actual cache release.
    backend : {'pytorch', 'jax'}
        Only ``'pytorch'`` performs any action.
    """
    if backend == 'pytorch':
        import torch
        dev_str = str(device).lower() if device is not None else ''
        gc.collect()
        if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
