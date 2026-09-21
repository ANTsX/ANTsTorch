"""Tests for antstorch.syn.robust_affine — the ported PyTorch-only core solver.

Scope reminder (see antstorch/syn/robust_affine.py's module docstring): this
port covers syntx.robust_affine's native PyTorch multi-start solver only.
Landmarks/SIFT3D, mode='tournament', and the legacy multi-candidate
ants.registration C++ fallback are intentionally excluded.
"""
import warnings
from unittest import mock

import ants
import numpy as np
import pytest
import scipy.ndimage as ndi
import torch

from antstorch.syn import (
    robust_affine,
    robust_center_of_mass,
    compute_center_of_mass,
    compute_fov_center,
)
from antstorch.syn.robust_affine import _mps_grid_sample_backward_available, _resolve_solver_device


def _synthetic_blob_2d(size=(64, 64)):
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    cy, cx = size[0] // 2, size[1] // 2
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 10.0 ** 2))).astype(np.float32)
    blob += 0.3 * np.exp(-(((yy - (cy - 12)) ** 2 + (xx - (cx + 13)) ** 2) / (2 * 5.0 ** 2))).astype(np.float32)
    return blob


def test_com_only_mode_produces_expected_translation_and_keys():
    blob = _synthetic_blob_2d()
    fixed = ants.from_numpy(blob)
    shifted = ndi.shift(blob, shift=(5, -3), order=1, mode="constant", cval=0.0)
    moving = ants.from_numpy(shifted.astype(np.float32))

    result = robust_affine(fixed, moving, mode="com_only", verbose=False)

    for key in ("warpedmovout", "warpedfixout", "fwdtransforms", "invtransforms", "whichtoinvert_inv"):
        assert key in result

    warped = result["warpedmovout"].numpy()
    # com_only should bring the shifted blob substantially closer to fixed
    # even though it is translation-only (no rotation/scale correction).
    diff_before = np.abs(blob - shifted).mean()
    diff_after = np.abs(blob - warped).mean()
    assert diff_after < diff_before


def test_translation_only_mode_is_an_alias_for_com_only():
    blob = _synthetic_blob_2d()
    fixed = ants.from_numpy(blob)
    moving = ants.from_numpy(blob.copy())
    result = robust_affine(fixed, moving, mode="translation_only", verbose=False)
    assert "warpedmovout" in result


# These four tests call robust_affine(mode='pytorch', ...) directly, which
# auto-selects the fastest available device. On a Mac whose PyTorch build is
# missing the MPS backward kernel for grid_sample (see
# _resolve_solver_device's docstring), that legitimately raises a
# RuntimeWarning documenting the automatic CPU fallback -- useful for a real
# caller, but not something these tests assert on, so it is filtered out of
# the run's warnings summary here rather than left to show up as noise.
_MPS_FALLBACK_WARNING = "grid_sample is missing its backward MPS kernel"


@pytest.mark.filterwarnings(f"ignore:.*{_MPS_FALLBACK_WARNING}.*:RuntimeWarning")
def test_pytorch_mode_recovers_known_translation():
    blob = _synthetic_blob_2d()
    fixed = ants.from_numpy(blob)
    shifted = ndi.shift(blob, shift=(5, -3), order=1, mode="constant", cval=0.0)
    moving = ants.from_numpy(shifted.astype(np.float32))

    result = robust_affine(
        fixed, moving, mode="pytorch", multi_start=True, n_starts=2, seed=42, verbose=False
    )

    assert result["status"] == "SUCCESS"
    warped = result["warpedmovout"].numpy()
    diff_before = np.abs(blob - shifted).mean()
    diff_after = np.abs(blob - warped).mean()
    assert diff_after < diff_before


@pytest.mark.filterwarnings(f"ignore:.*{_MPS_FALLBACK_WARNING}.*:RuntimeWarning")
def test_pytorch_mode_recovers_known_rotation():
    size = (96, 96)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    blob = np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 14.0 ** 2))).astype(np.float32)
    blob += 0.5 * np.exp(-(((yy - 25) ** 2 + (xx - 70) ** 2) / (2 * 7.0 ** 2))).astype(np.float32)
    blob += 0.5 * np.exp(-(((yy - 70) ** 2 + (xx - 25) ** 2) / (2 * 7.0 ** 2))).astype(np.float32)
    fixed = ants.from_numpy(blob)

    rotated = ndi.rotate(blob, angle=8.0, reshape=False, order=1, mode="constant", cval=0.0)
    moving = ants.from_numpy(rotated.astype(np.float32))

    result = robust_affine(
        fixed, moving, mode="pytorch", multi_start=True, n_starts=4, preset="accurate",
        seed=7, verbose=False,
    )

    assert result["status"] == "SUCCESS"
    warped = result["warpedmovout"].numpy()
    diff_before = np.abs(blob - rotated).mean()
    diff_after = np.abs(blob - warped).mean()
    assert diff_after < diff_before


@pytest.mark.filterwarnings(f"ignore:.*{_MPS_FALLBACK_WARNING}.*:RuntimeWarning")
def test_result_dict_has_expected_keys_for_pytorch_mode():
    blob = _synthetic_blob_2d()
    fixed = ants.from_numpy(blob)
    moving = ants.from_numpy(blob.copy())
    result = robust_affine(fixed, moving, mode="pytorch", multi_start=False, n_starts=1, verbose=False)
    expected = {
        "warpedmovout", "warpedfixout", "fwdtransforms", "invtransforms",
        "whichtoinvert_inv", "runtime_seconds", "time", "init_candidate",
        "init_score", "final_loss", "candidates_scored", "status",
    }
    assert expected.issubset(result.keys())


def test_unsupported_mode_raises_value_error():
    blob = _synthetic_blob_2d((32, 32))
    fixed = ants.from_numpy(blob)
    moving = ants.from_numpy(blob.copy())
    with pytest.raises(ValueError, match="not supported by this port"):
        robust_affine(fixed, moving, mode="bogus_mode")


def test_non_pytorch_backend_raises_value_error():
    blob = _synthetic_blob_2d((32, 32))
    fixed = ants.from_numpy(blob)
    moving = ants.from_numpy(blob.copy())
    with pytest.raises(ValueError, match="PyTorch-only"):
        robust_affine(fixed, moving, backend="jax")


@pytest.mark.filterwarnings(f"ignore:.*{_MPS_FALLBACK_WARNING}.*:RuntimeWarning")
def test_enable_landmarks_warns_without_erroring_in_pytorch_mode():
    blob = _synthetic_blob_2d((32, 32))
    fixed = ants.from_numpy(blob)
    moving = ants.from_numpy(blob.copy())
    with pytest.warns(RuntimeWarning, match="enable_landmarks=True has no effect"):
        result = robust_affine(
            fixed, moving, mode="pytorch", enable_landmarks=True,
            multi_start=False, n_starts=1, verbose=False,
        )
    assert result["status"] == "SUCCESS"


def test_robust_center_of_mass_runs_on_synthetic_blob():
    blob = _synthetic_blob_2d((32, 32))
    fixed = ants.from_numpy(blob)
    com = robust_center_of_mass(fixed)
    assert len(com) == 2


def test_compute_center_of_mass_matches_weighted_centroid():
    size = (40, 40)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    blob = np.exp(-(((yy - 10) ** 2 + (xx - 30) ** 2) / (2 * 4.0 ** 2))).astype(np.float32)
    fixed = ants.from_numpy(blob)
    com = compute_center_of_mass(fixed)
    # weighted centroid should land close to the Gaussian's own center
    assert abs(com[0] - 10) < 2.0
    assert abs(com[1] - 30) < 2.0


def test_resolve_solver_device_is_a_no_op_for_cpu_and_cuda():
    assert _resolve_solver_device(torch.device("cpu"), 2) == torch.device("cpu")
    if torch.cuda.is_available():
        assert _resolve_solver_device(torch.device("cuda"), 3) == torch.device("cuda")


def test_resolve_solver_device_falls_back_to_cpu_when_mps_grid_sample_backward_missing():
    # Regression test for the bug reported against this port: on a real Mac
    # with an MPS build missing aten::grid_sampler_2d_backward (or _3d_),
    # _run_pytorch_affine_solver's loss.backward() crashed with
    # NotImplementedError instead of falling back, the same class of gap
    # antstorch.syn.core.pipeline already works around for 3-D. This
    # sandbox has no MPS hardware, so the missing-kernel path is exercised
    # by mocking the probe rather than mocking torch.backends.mps itself.
    with mock.patch(
        "antstorch.syn.robust_affine._mps_grid_sample_backward_available", return_value=False
    ), mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        with pytest.warns(RuntimeWarning, match="grid_sample is missing its backward"):
            resolved = _resolve_solver_device(torch.device("mps"), 2)
    assert resolved == torch.device("cpu")


def test_resolve_solver_device_stays_on_mps_when_grid_sample_backward_available():
    with mock.patch(
        "antstorch.syn.robust_affine._mps_grid_sample_backward_available", return_value=True
    ), mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        resolved = _resolve_solver_device(torch.device("mps"), 2)
    assert resolved == torch.device("mps")


def test_mps_grid_sample_backward_probe_is_false_without_mps_hardware():
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available on this machine; see the mocked tests above instead")
    _mps_grid_sample_backward_available.cache_clear()
    assert _mps_grid_sample_backward_available(2) is False
    assert _mps_grid_sample_backward_available(3) is False


def test_compute_fov_center_is_geometric_center():
    blob = np.zeros((20, 30), dtype=np.float32)
    fixed = ants.from_numpy(blob)
    center = compute_fov_center(fixed)
    assert abs(center[0] - 9.5) < 1e-6
    assert abs(center[1] - 14.5) < 1e-6
