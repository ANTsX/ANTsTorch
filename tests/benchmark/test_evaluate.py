"""End-to-end tests for antstorch.benchmark.evaluate.evaluate_mindboggle_pair()
against a tiny synthetic Mindboggle-style pair, covering all 6 ANTsTorch-native
model variants (gaussian_syn/sobolev_syn/dsti_syn/bspline_syn -- the four
antstorch.syn.syn_registration() regularizers, dense-SyN-stage only -- plus
bspline_svf and gaussian_svf, two stationary-velocity families). Runs the real
registration/metric pipeline (no mocking) on a small volume so wall-clock
time stays low."""
import os

import ants
import numpy as np
import pytest

from antstorch.benchmark.evaluate import DEFAULT_REG_ITERATIONS, evaluate_mindboggle_pair, evaluate_pair


def test_all_models_share_default_iteration_schedule():
    assert DEFAULT_REG_ITERATIONS == (100, 100, 50, 10)


def _assert_valid_success_record(rec, model):
    assert rec["status"] == "SUCCESS"
    assert rec["model_type"] == model
    assert rec["cohort_type"] == "intra"
    assert rec["fixed_id"] == "OASIS-TRT-20-1"
    assert rec["moving_id"] == "OASIS-TRT-20-2"
    assert np.isfinite(rec["dice_sym"])
    assert np.isfinite(rec["dice_fixed"])
    assert np.isfinite(rec["dice_moving"])
    assert np.isfinite(rec["affine_dice_sym"])
    assert np.isfinite(rec["folding_pct"])
    assert np.isfinite(rec["min_jacobian"])
    assert rec["runtime_seconds"] >= 0.0
    assert 0.0 <= rec["dice_sym"] <= 1.0
    assert len(rec["transforms"]["fwdtransforms"]) >= 1
    assert len(rec["transforms"]["invtransforms"]) >= 1


@pytest.mark.parametrize("model", ["gaussian_syn", "sobolev_syn", "dsti_syn", "bspline_syn"])
def test_evaluate_mindboggle_pair_syn_regularizers(mock_mindboggle_dataset, tmp_path, model):
    pairs_csv, data_dir = mock_mindboggle_dataset
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model=model,
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        use_n4=False,
        reg_iterations=[2, 2, 1, 1],
    )
    _assert_valid_success_record(rec, model)


def test_evaluate_mindboggle_pair_bspline_svf(mock_mindboggle_dataset, tmp_path):
    # Uses the function's own default mesh_size/shrink_factors/smoothing_sigmas
    # (only reg_iterations is shortened, for test speed) -- the mock dataset's
    # volume shape (see conftest.py) was chosen specifically so this default
    # bspline_svf configuration runs cleanly; smaller synthetic volumes can
    # trip a small-domain edge case in antstorch.bspline_flows.bspline_synthesis's
    # coefficient-lattice indexing that is unrelated to this benchmark port
    # and out of scope to fix here (real Mindboggle volumes are far larger
    # than any synthetic test fixture and never approach this edge).
    pairs_csv, data_dir = mock_mindboggle_dataset
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="bspline_svf",
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        use_n4=False,
        reg_iterations=[1, 1, 1, 1],
    )
    _assert_valid_success_record(rec, "bspline_svf")


def test_evaluate_mindboggle_pair_gaussian_svf(mock_mindboggle_dataset, tmp_path):
    pairs_csv, data_dir = mock_mindboggle_dataset
    registration_output_dir = tmp_path / "pair_000" / "gaussian_svf"
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="gaussian_svf",
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        registration_output_dir=str(registration_output_dir),
        use_n4=False,
        reg_iterations=[1, 1, 1, 1],
        update_field_sigma=1.0,
        total_field_sigma=0.25,
        squaring_steps=2,
    )
    _assert_valid_success_record(rec, "gaussian_svf")
    assert rec["warped_moving"] == str(registration_output_dir / "warped_moving.nii.gz")
    assert os.path.exists(rec["warped_moving"])
    assert rec["warped_moving_labels"] == str(registration_output_dir / "warped_moving_labels.nii.gz")
    assert os.path.exists(rec["warped_moving_labels"])
    assert os.path.exists(registration_output_dir / "registration_0GenericAffine.mat")
    assert os.path.exists(registration_output_dir / "registration_1Warp.nii.gz")
    assert os.path.exists(registration_output_dir / "registration_1InverseWarp.nii.gz")
    assert all(str(registration_output_dir) in path for path in rec["transforms"]["fwdtransforms"])


@pytest.mark.parametrize("model", ["gaussian_syn", "bspline_svf", "gaussian_svf"])
def test_warped_moving_labels_written_and_valid(mock_mindboggle_dataset, tmp_path, model):
    """registration_output_dir must also persist a nearest-neighbor-resampled
    warped moving label map, for every model family (dense SyN and both SVF
    variants) -- not just the warped intensity image."""
    pairs_csv, data_dir = mock_mindboggle_dataset
    registration_output_dir = tmp_path / "pair_000" / model
    kwargs = dict(
        pair_idx=0,
        model=model,
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        registration_output_dir=str(registration_output_dir),
        use_n4=False,
        reg_iterations=[1, 1, 1, 1],
    )
    if model == "gaussian_svf":
        kwargs.update(update_field_sigma=1.0, total_field_sigma=0.25, squaring_steps=2)
    rec = evaluate_mindboggle_pair(**kwargs)
    _assert_valid_success_record(rec, model)

    labels_path = rec["warped_moving_labels"]
    assert labels_path == str(registration_output_dir / "warped_moving_labels.nii.gz")
    assert os.path.exists(labels_path)

    warped_labels = ants.image_read(labels_path)
    moving_labels = ants.image_read(
        os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-2", "labels.DKT31.manual.nii.gz")
    )
    fixed_intensity = ants.image_read(
        os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz")
    )
    # Resampled onto the fixed grid, not the moving grid.
    assert warped_labels.shape == fixed_intensity.shape
    # Nearest-neighbor resampling: only label values already present in the
    # (integer-valued) moving label map may appear -- no interpolated
    # in-between values, unlike a bilinear/linear resample.
    original_labels = set(np.unique(moving_labels.numpy()).tolist())
    warped_values = set(np.unique(warped_labels.numpy()).tolist())
    assert warped_values <= original_labels


def test_warped_moving_labels_absent_without_registration_output_dir(mock_mindboggle_dataset, tmp_path):
    pairs_csv, data_dir = mock_mindboggle_dataset
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="bspline_svf",
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        use_n4=False,
        reg_iterations=[1, 1, 1, 1],
    )
    _assert_valid_success_record(rec, "bspline_svf")
    assert rec["warped_moving"] is None
    assert rec["warped_moving_labels"] is None


@pytest.mark.parametrize("model", ["bspline_svf", "gaussian_svf"])
def test_svf_models_warpedmovout_preserves_original_intensity_range(mock_mindboggle_dataset, tmp_path, model):
    """Regression test for the normalization asymmetry documented in the
    project doc (SVF-vs-SyN warpedmovout value range investigation):
    _run_bspline_svf()/_run_gaussian_svf() must reconstruct warpedmovout
    from the un-normalized moving image (like syn_registration() already
    does for the *_syn arms), not return the percentile-clip-normalized
    tensor used internally to drive the registration's similarity metric.
    The mock dataset's volumes (conftest.py) are ~N(100, 10) plus an
    additive blob, i.e. comfortably outside [0, 1] -- so a max value that
    stayed near 1.0 would mean warpedmovout was still in normalized space."""
    pairs_csv, data_dir = mock_mindboggle_dataset
    registration_output_dir = tmp_path / "pair_000" / model
    kwargs = dict(
        pair_idx=0,
        model=model,
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        registration_output_dir=str(registration_output_dir),
        use_n4=False,
        reg_iterations=[1, 1, 1, 1],
    )
    if model == "gaussian_svf":
        kwargs.update(update_field_sigma=1.0, total_field_sigma=0.25, squaring_steps=2)
    rec = evaluate_mindboggle_pair(**kwargs)
    _assert_valid_success_record(rec, model)

    warped = ants.image_read(rec["warped_moving"])
    moving = ants.image_read(os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-2", "t1weighted_brain.nii.gz"))
    assert warped.numpy().max() > 5.0, (
        f"{model}: warpedmovout max={warped.numpy().max():.4f} looks normalized to [0, 1] "
        "instead of the original image intensity range"
    )
    # Same order of magnitude as the original (un-normalized) moving image,
    # not clipped down to a [0, 1]-ish percentile-normalized range.
    assert warped.numpy().max() > 0.1 * moving.numpy().max()


def test_evaluate_pair_is_an_alias_for_evaluate_mindboggle_pair():
    assert evaluate_pair is evaluate_mindboggle_pair


def test_evaluate_mindboggle_pair_unknown_model_raises_value_error(mock_mindboggle_dataset, tmp_path):
    pairs_csv, data_dir = mock_mindboggle_dataset
    with pytest.raises(ValueError, match="Unknown registration model"):
        evaluate_mindboggle_pair(
            pair_idx=0,
            model="not_a_real_model",
            device="cpu",
            pairs_csv=pairs_csv,
            data_dir=data_dir,
            canonical_affine_dir=str(tmp_path / "canonical_affines"),
            use_n4=False,
        )


def test_evaluate_mindboggle_pair_shares_canonical_affine_across_models(mock_mindboggle_dataset, tmp_path):
    """Every model variant for a given pair must reuse the exact same
    canonical affine fit -- the fairness invariant this harness preserves
    from the syntx original. Verified here by checking that the second
    call's affine-fit stage hits the on-disk cache (the .mat file is not
    rewritten) rather than by comparing runtimes, which can be noisy."""
    pairs_csv, data_dir = mock_mindboggle_dataset
    canonical_affine_dir = str(tmp_path / "canonical_affines")

    rec1 = evaluate_mindboggle_pair(
        pair_idx=0, model="gaussian_syn", device="cpu",
        pairs_csv=pairs_csv, data_dir=data_dir,
        canonical_affine_dir=canonical_affine_dir, use_n4=False,
        reg_iterations=[2, 2, 1, 1],
    )
    affine_path = os.path.join(canonical_affine_dir, "pair_000_0GenericAffine.mat")
    assert os.path.exists(affine_path)
    mtime_after_first = os.path.getmtime(affine_path)

    rec2 = evaluate_mindboggle_pair(
        pair_idx=0, model="sobolev_syn", device="cpu",
        pairs_csv=pairs_csv, data_dir=data_dir,
        canonical_affine_dir=canonical_affine_dir, use_n4=False,
        reg_iterations=[2, 2, 1, 1],
    )
    assert os.path.getmtime(affine_path) == mtime_after_first
    assert rec1["affine_dice_sym"] == pytest.approx(rec2["affine_dice_sym"], abs=1e-9)
