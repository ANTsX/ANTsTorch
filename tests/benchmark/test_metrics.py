"""Tests for antstorch.benchmark.metrics — Dice, Jacobian, and energy
primitives, ported verbatim from syntx.deformation_metrics."""
import numpy as np
import pytest

import ants

from antstorch.ants_transform_io import write_affine_transform
from antstorch.benchmark.metrics import (
    compute_bending_energy,
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_jacobian_metrics,
)

_SHAPE = (16, 18, 16)
_SPACING = (1.5, 1.5, 1.5)


def _labeled_image():
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij")
    cz, cy, cx = _SHAPE[0] / 2, _SHAPE[1] / 2, _SHAPE[2] / 2
    r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    labels = np.zeros(_SHAPE, dtype=np.float32)
    labels[r < 6] = 1
    labels[r < 3] = 2
    return ants.from_numpy(labels, spacing=_SPACING)


def _intensity_image():
    return ants.from_numpy(np.zeros(_SHAPE, dtype=np.float32), spacing=_SPACING)


def _identity_affine_path(tmp_path, dim=3):
    path = str(tmp_path / "identity_0GenericAffine.mat")
    write_affine_transform(np.eye(dim), np.zeros(dim), dim, path)
    return path


def test_compute_bidirectional_dice_identity_transform_gives_perfect_overlap(tmp_path):
    labels = _labeled_image()
    intensity = _intensity_image()
    identity_path = _identity_affine_path(tmp_path)

    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl=labels, ml=labels, fi=intensity, mi=intensity,
        fwdtransforms=[identity_path], invtransforms=[identity_path],
        whichtoinvert_inv=[True],
    )
    assert dice_fixed == pytest.approx(1.0, abs=1e-6)
    assert dice_moving == pytest.approx(1.0, abs=1e-6)
    assert dice_sym == pytest.approx(1.0, abs=1e-6)


def test_compute_bidirectional_dice_partial_overlap_is_between_zero_and_one(tmp_path):
    # A moving label map shifted far enough from the fixed one that their
    # regions only partially overlap.
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij")
    cz, cy, cx = _SHAPE[0] / 2, _SHAPE[1] / 2, _SHAPE[2] / 2
    r_fixed = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_shifted = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx - 4) ** 2)

    fixed_labels_arr = np.zeros(_SHAPE, dtype=np.float32)
    fixed_labels_arr[r_fixed < 6] = 1
    moving_labels_arr = np.zeros(_SHAPE, dtype=np.float32)
    moving_labels_arr[r_shifted < 6] = 1

    fixed_labels = ants.from_numpy(fixed_labels_arr, spacing=_SPACING)
    moving_labels = ants.from_numpy(moving_labels_arr, spacing=_SPACING)
    intensity = _intensity_image()
    identity_path = _identity_affine_path(tmp_path)

    _, _, dice_sym = compute_bidirectional_dice(
        fl=fixed_labels, ml=moving_labels, fi=intensity, mi=intensity,
        fwdtransforms=[identity_path], invtransforms=[identity_path],
        whichtoinvert_inv=[True],
    )
    assert np.isfinite(dice_sym)
    assert 0.0 < dice_sym < 1.0


def test_compute_bidirectional_dice_with_label_absent_from_one_side_stays_finite(tmp_path):
    # Regression test for a real bug seen on an inter-subject Mindboggle
    # pair: when a label present in one image has zero voxels in the
    # other (e.g. a small cortical region pushed entirely out of the
    # resampling domain by a large deformation), ants.label_overlap_measures
    # reports that label's ratio not as NaN/inf but as a large finite
    # sentinel value (approximately np.finfo(np.float64).max). A plain
    # np.isfinite() filter lets that sentinel through, and averaging
    # several such near-DBL_MAX values overflows float64 and silently
    # produces a literal inf Dice score -- exactly what was observed on
    # real data (dice_moving == inf for an inter-subject pair). Here we
    # construct two label maps that share label 1 (partial overlap) but
    # where label 2 exists only in the moving image and label 3 exists
    # only in the fixed image, forcing this exact degenerate 0/0 ratio
    # on both the fixed-space and moving-space passes.
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij")
    cz, cy, cx = _SHAPE[0] / 2, _SHAPE[1] / 2, _SHAPE[2] / 2
    r_fixed = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_shifted = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx - 4) ** 2)

    fixed_labels_arr = np.zeros(_SHAPE, dtype=np.float32)
    fixed_labels_arr[r_fixed < 6] = 1
    fixed_labels_arr[0:2, 0:2, 0:2] = 3  # label 3: present only in fixed

    moving_labels_arr = np.zeros(_SHAPE, dtype=np.float32)
    moving_labels_arr[r_shifted < 6] = 1
    moving_labels_arr[-2:, -2:, -2:] = 2  # label 2: present only in moving

    fixed_labels = ants.from_numpy(fixed_labels_arr, spacing=_SPACING)
    moving_labels = ants.from_numpy(moving_labels_arr, spacing=_SPACING)
    intensity = _intensity_image()
    identity_path = _identity_affine_path(tmp_path)

    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl=fixed_labels, ml=moving_labels, fi=intensity, mi=intensity,
        fwdtransforms=[identity_path], invtransforms=[identity_path],
        whichtoinvert_inv=[True],
    )
    assert np.isfinite(dice_fixed)
    assert np.isfinite(dice_moving)
    assert np.isfinite(dice_sym)
    assert 0.0 <= dice_fixed <= 1.0
    assert 0.0 <= dice_moving <= 1.0
    assert 0.0 <= dice_sym <= 1.0


def test_compute_jacobian_metrics_identity_warp_gives_unit_jacobian():
    fixed = ants.from_numpy(np.ones(_SHAPE, dtype=np.float32), spacing=_SPACING)
    zero_field = np.zeros(_SHAPE + (3,), dtype=np.float32)
    warp = ants.from_numpy(zero_field, spacing=_SPACING, has_components=True)

    jac = compute_jacobian_metrics(fixed, warp)
    assert jac["min"] == pytest.approx(1.0, abs=1e-3)
    assert jac["max"] == pytest.approx(1.0, abs=1e-3)
    assert jac["mean"] == pytest.approx(1.0, abs=1e-3)
    assert jac["folding_pct"] == pytest.approx(0.0, abs=1e-6)


def test_compute_harmonic_energy_zero_field_is_zero():
    zero_field = np.zeros(_SHAPE + (3,), dtype=np.float32)
    assert compute_harmonic_energy(zero_field, spacing=(1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-8)


def test_compute_bending_energy_zero_field_is_zero():
    zero_field = np.zeros(_SHAPE + (3,), dtype=np.float32)
    assert compute_bending_energy(zero_field, spacing=(1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-8)


def test_compute_harmonic_energy_linear_field_matches_constant_gradient():
    # A field whose x-component grows linearly along axis 0 with slope `a`
    # has d(field_x)/d(axis0) == a everywhere (away from edge effects), so
    # the mean-squared-gradient (this module's harmonic energy definition)
    # should be close to a**2.
    a = 2.0
    field = np.zeros(_SHAPE + (3,), dtype=np.float32)
    coords = np.arange(_SHAPE[0], dtype=np.float32).reshape(-1, 1, 1)
    field[..., 0] = a * coords
    energy = compute_harmonic_energy(field, spacing=(1.0, 1.0, 1.0))
    assert energy > 0.0
    # Loose bound: interior gradient contributes a**2 per matching term;
    # just check it's in a sane ballpark rather than pinning the exact
    # edge-effect-influenced mean.
    assert energy < (a ** 2) * 2


def test_compute_harmonic_energy_accepts_torch_tensor_and_ants_image():
    import torch
    field_np = np.random.default_rng(0).normal(size=_SHAPE + (3,)).astype(np.float32)
    e_np = compute_harmonic_energy(field_np, spacing=(1.0, 1.0, 1.0))

    field_torch = torch.from_numpy(field_np)
    e_torch = compute_harmonic_energy(field_torch, spacing=(1.0, 1.0, 1.0))
    assert e_np == pytest.approx(e_torch, rel=1e-5)

    field_ants = ants.from_numpy(field_np, spacing=(1.0, 1.0, 1.0), has_components=True)
    e_ants = compute_harmonic_energy(field_ants)
    assert e_np == pytest.approx(e_ants, rel=1e-5)
