import ants
import numpy as np
import torch

from antstorch.syn.core import (
    auto_detect_device,
    normalize_and_tensorize,
    cleanup_gpu,
)


def test_auto_detect_device_honors_explicit_request():
    assert auto_detect_device(requested_device='CUDA') == 'cuda'
    assert auto_detect_device(requested_device='cpu') == 'cpu'


def test_auto_detect_device_matches_torch_hardware_probe():
    # Mirror auto_detect_device's own probing order rather than hard-coding
    # an expected device: CI/sandbox machines typically have neither
    # CUDA nor MPS, but Apple Silicon development machines do have MPS.
    if torch.cuda.is_available():
        expected = 'cuda'
    elif torch.backends.mps.is_available():
        expected = 'mps'
    else:
        expected = 'cpu'
    assert auto_detect_device(backend='pytorch') == expected


def test_auto_detect_device_jax_backend():
    assert auto_detect_device(backend='jax') == 'jax'


def test_normalize_and_tensorize_shape_and_range():
    rng = np.random.default_rng(0)
    fixed_np = (rng.random((6, 7)) * 100).astype(np.float32)
    moving_np = (rng.random((6, 7)) * 100).astype(np.float32)
    fixed = ants.from_numpy(fixed_np)
    moving = ants.from_numpy(moving_np)

    I_tensor, J_tensor = normalize_and_tensorize(fixed, moving, backend='pytorch', device='cpu')

    # normalize_and_tensorize permutes the trailing two (spatial) axes, so a
    # (6, 7) numpy image becomes a (1, 1, 7, 6) tensor.
    assert I_tensor.shape == (1, 1, 7, 6)
    assert J_tensor.shape == (1, 1, 7, 6)
    assert I_tensor.dtype == torch.float32
    assert I_tensor.min().item() >= 0.0
    assert I_tensor.max().item() <= 1.0
    assert J_tensor.min().item() >= 0.0
    assert J_tensor.max().item() <= 1.0


def test_normalize_and_tensorize_matches_manual_foreground_percentile_normalization():
    rng = np.random.default_rng(1)
    fixed_np = (rng.random((5, 5)) * 50).astype(np.float32)
    fixed = ants.from_numpy(fixed_np)
    moving = ants.from_numpy(fixed_np.copy())

    I_tensor, _ = normalize_and_tensorize(fixed, moving, backend='pytorch', device='cpu')

    pos = fixed_np[fixed_np > 0]
    p02 = np.percentile(pos, 2.0)
    p98 = np.percentile(pos, 98.0)
    expected = np.clip((fixed_np - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)

    # I_tensor is permuted from (H, W) to (1, 1, W, H) per the perm = [0, 1, dim+1, ..., 2] convention.
    recovered = I_tensor[0, 0].numpy().T
    np.testing.assert_allclose(recovered, expected, atol=1e-5)


def test_cleanup_gpu_cpu_device_is_a_no_op():
    # Should not raise even though there is no GPU/MPS backend present.
    cleanup_gpu('cpu', backend='pytorch')
