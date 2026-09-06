from unittest import mock

import ants
import numpy as np
import pytest
import torch

from antstorch.syn.core import (
    auto_detect_device,
    mps_grid_sample_3d_available,
    normalize_and_tensorize,
    relocate_tensors_avoiding_mps_grid_sample_3d,
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


def test_mps_grid_sample_3d_available_is_false_without_mps_hardware():
    # A machine with no MPS backend at all (most CI/sandbox machines) must
    # get False without raising -- torch.backends.mps.is_available() itself
    # is the short-circuit in mps_grid_sample_3d_available(), so this holds
    # regardless of PyTorch version.
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available on this machine; see the probe test below instead")
    mps_grid_sample_3d_available.cache_clear()
    assert mps_grid_sample_3d_available() is False


def test_mps_grid_sample_3d_available_probes_rather_than_hardcodes_a_verdict():
    # On real Apple Silicon this can legitimately be True: PyTorch may have
    # since shipped a native MPS kernel for grid_sampler_3d (the gap tracked
    # by https://github.com/pytorch/pytorch/issues/160237), or the process
    # may have PYTORCH_ENABLE_MPS_FALLBACK=1 set, under which every
    # unimplemented MPS op (including this one) transparently falls back to
    # CPU and so "succeeds". Either way the whole point of probing instead
    # of hardcoding a verdict is that this function must not be pinned to
    # one answer -- only that it is a stable, non-raising bool.
    mps_grid_sample_3d_available.cache_clear()
    first = mps_grid_sample_3d_available()
    assert isinstance(first, bool)
    if not torch.backends.mps.is_available():
        assert first is False
    mps_grid_sample_3d_available.cache_clear()
    assert mps_grid_sample_3d_available() == first


def test_relocate_tensors_is_a_no_op_for_2d():
    fixed = torch.zeros(1, 1, 8, 7)
    moving = torch.zeros(1, 1, 8, 7)
    result = relocate_tensors_avoiding_mps_grid_sample_3d(2, "test", fixed=fixed, moving=moving)
    assert result["fixed"] is fixed
    assert result["moving"] is moving


def test_relocate_tensors_is_a_no_op_when_nothing_is_on_mps():
    fixed = torch.zeros(1, 1, 4, 4, 4)
    moving = torch.zeros(1, 1, 4, 4, 4)
    result = relocate_tensors_avoiding_mps_grid_sample_3d(
        3, "test", fixed=fixed, moving=moving, initial_affine=None
    )
    assert result["fixed"] is fixed
    assert result["moving"] is moving
    assert result["initial_affine"] is None


def test_relocate_tensors_passes_through_when_mps_grid_sample_3d_available():
    # Even a tensor reporting an "mps" device should pass through untouched
    # once the installed PyTorch build has a real 3-D grid_sample kernel.
    fixed = torch.zeros(1, 1, 4, 4, 4)
    with mock.patch.object(torch.Tensor, "device", new_callable=mock.PropertyMock) as device_mock, \
            mock.patch(
                "antstorch.syn.core.pipeline.mps_grid_sample_3d_available", return_value=True
            ):
        device_mock.return_value = torch.device("mps")
        result = relocate_tensors_avoiding_mps_grid_sample_3d(3, "test", fixed=fixed)
    assert result["fixed"] is fixed


def test_relocate_tensors_falls_back_to_cpu_for_mps_3d_without_kernel():
    # Simulates https://github.com/pytorch/pytorch/issues/160237 on hardware
    # this sandbox does not have: an "mps" tensor, 3-D, no 3-D grid_sample
    # kernel -- fixed/moving/initial_affine must relocate to cpu together,
    # with exactly one warning.
    fixed = torch.zeros(1, 1, 4, 4, 4)
    moving = torch.zeros(1, 1, 4, 4, 4)
    matrix = torch.eye(3)
    translation = torch.zeros(3)
    with mock.patch.object(torch.Tensor, "device", new_callable=mock.PropertyMock) as device_mock, \
            mock.patch(
                "antstorch.syn.core.pipeline.mps_grid_sample_3d_available", return_value=False
            ):
        device_mock.return_value = torch.device("mps")
        with pytest.warns(RuntimeWarning, match="grid_sampler_3d"):
            result = relocate_tensors_avoiding_mps_grid_sample_3d(
                3, "test_context", fixed=fixed, moving=moving, initial_affine=(matrix, translation)
            )
    # The mocked .device property reports "mps" for every tensor, including
    # results of .cpu() (whose real storage was cpu-resident all along), so
    # identity/device checks can't distinguish "relocated" from "untouched"
    # here. The warning firing (above) is the behavioral signal that the
    # mps + no-kernel branch actually ran; this just confirms values survive.
    torch.testing.assert_close(result["fixed"], fixed)
    torch.testing.assert_close(result["moving"], moving)
    relocated_matrix, relocated_translation = result["initial_affine"]
    torch.testing.assert_close(relocated_matrix, matrix)
    torch.testing.assert_close(relocated_translation, translation)
