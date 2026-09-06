"""Device selection and image-to-tensor preprocessing utilities.

Ported from ``syntx.core.pipeline`` (PyTorch backend only). These helpers
bridge ``ants.ANTsImage`` inputs to the normalized, permuted tensors that
the rest of ``antstorch.syn.core`` operates on, and provide GPU/MPS memory
hygiene for iterative registration loops.
"""

import functools
import gc
import warnings

import numpy as np


@functools.lru_cache(maxsize=1)
def mps_grid_sample_3d_available() -> bool:
    """Whether the installed PyTorch's MPS backend implements 3-D ``grid_sample``.

    As of PyTorch 2.9 dev builds, the MPS backend has no kernel at all for
    ``aten::grid_sampler_3d`` -- not a numerical bug, an outright missing op
    -- so any 3-D ``torch.nn.functional.grid_sample`` call on an ``mps``
    tensor raises ``NotImplementedError`` immediately (forward pass, not
    just backward; see https://github.com/pytorch/pytorch/issues/160237,
    which tracks it under the broader MPS operator-coverage gap #141287).
    No merged fix as of this writing; PyTorch's own documented workaround is
    the process-wide ``PYTORCH_ENABLE_MPS_FALLBACK=1`` environment variable
    (silent, transparent CPU fallback for *every* unimplemented MPS op, with
    a performance cost).

    This is probed empirically -- a throwaway 2x2x2 call -- rather than
    hardcoded ``False``, so every caller of this function (and of
    :func:`relocate_tensors_avoiding_mps_grid_sample_3d`, which uses it)
    automatically stops paying the CPU-fallback cost the moment a future
    PyTorch release ships a real MPS kernel, with no code change needed
    here. Cached for the life of the process: PyTorch's op coverage cannot
    change mid-run.
    """
    import torch

    if not torch.backends.mps.is_available():
        return False
    try:
        probe_image = torch.zeros(1, 1, 2, 2, 2, device="mps")
        probe_grid = torch.zeros(1, 2, 2, 2, 3, device="mps")
        torch.nn.functional.grid_sample(probe_image, probe_grid, align_corners=True)
        return True
    except (NotImplementedError, RuntimeError):
        return False


def relocate_tensors_avoiding_mps_grid_sample_3d(dimension: int, context: str, **named_tensors):
    """Move MPS tensors to CPU together when 3-D ``grid_sample`` is unavailable there.

    Every dense warp in ``antstorch.bspline_flows`` (and ``antstorch.syn``)
    eventually calls ``torch.nn.functional.grid_sample`` -- directly, or via
    ``antstorch.bspline_flows.spatial_transform.warp_image()``. On a 3-D
    registration entirely on an ``mps`` device where
    :func:`mps_grid_sample_3d_available` is ``False``, that call would raise
    ``NotImplementedError: aten::grid_sampler_3d`` (see that function's
    docstring). This relocates every tensor among ``named_tensors`` (values
    may be a bare ``Tensor``, or a ``tuple``/``list`` of them, e.g. an
    ``initial_affine=(matrix, translation)`` pair) from ``mps`` to ``cpu``
    together, so the whole run proceeds consistently on one device instead
    of crashing partway through -- the same "repli automatique vers CPU"
    :func:`antstorch.syn.syn.syn_registration` already applies for its own
    device auto-detection, extended here to cover every entry point
    (``bspline_svf_registration``, ``gaussian_svf_registration``,
    ``affine_registration``) and an explicitly requested ``device='mps'``,
    not only auto-detection.

    Parameters
    ----------
    dimension : int
        The registration's spatial dimensionality (``fixed_domain.dimension``);
        a no-op for anything but ``3`` (2-D ``grid_sampler_2d`` forward is
        implemented on MPS).
    context : str
        Caller name, included in the warning message.
    **named_tensors
        Any number of ``name=value`` pairs; ``None``, non-tensor, or
        already-CPU/CUDA values pass through untouched.

    Returns
    -------
    dict
        The same names, each value moved to CPU if relocation was needed
        (identical dict otherwise) -- unpack with
        ``fixed, moving = relocated["fixed"], relocated["moving"]``.
    """
    import torch

    if dimension != 3:
        return named_tensors

    def _device_of(value):
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, (tuple, list)):
            for item in value:
                found = _device_of(item)
                if found is not None:
                    return found
        return None

    on_mps = any(
        (found := _device_of(value)) is not None and found.type == "mps"
        for value in named_tensors.values()
    )
    if not on_mps or mps_grid_sample_3d_available():
        return named_tensors

    warnings.warn(
        f"{context}: 3-D torch.nn.functional.grid_sample has no MPS kernel in this "
        "PyTorch build (aten::grid_sampler_3d; see "
        "https://github.com/pytorch/pytorch/issues/160237). Falling back to CPU for "
        "this registration. Set PYTORCH_ENABLE_MPS_FALLBACK=1 instead if you would "
        "rather PyTorch itself fall back transparently for every unimplemented MPS op.",
        RuntimeWarning,
        stacklevel=3,
    )

    def _to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.cpu() if value.device.type == "mps" else value
        if isinstance(value, tuple):
            return tuple(_to_cpu(item) for item in value)
        if isinstance(value, list):
            return [_to_cpu(item) for item in value]
        return value

    return {name: _to_cpu(value) for name, value in named_tensors.items()}


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
