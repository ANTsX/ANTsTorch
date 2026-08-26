"""End-to-end tests for antstorch.benchmark.evaluate.evaluate_mindboggle_pair()
against a tiny synthetic Mindboggle-style pair, covering all 5 ANTsTorch-native
model variants (the four antstorch.syn.syn_registration() regularizers plus
bspline_svf). Runs the real registration/metric pipeline (no mocking) on a
small volume so wall-clock time stays low."""
import os

import numpy as np
import pytest

from antstorch.benchmark.evaluate import evaluate_mindboggle_pair, evaluate_pair


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


@pytest.mark.parametrize("model", ["gaussian", "sobolev", "dsti", "bspline"])
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
        reg_iterations=[4, 4, 2],
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
        reg_iterations=[3, 3, 3],
    )
    _assert_valid_success_record(rec, "bspline_svf")


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
        pair_idx=0, model="gaussian", device="cpu",
        pairs_csv=pairs_csv, data_dir=data_dir,
        canonical_affine_dir=canonical_affine_dir, use_n4=False,
        reg_iterations=[4, 4, 2],
    )
    affine_path = os.path.join(canonical_affine_dir, "pair_000_0GenericAffine.mat")
    assert os.path.exists(affine_path)
    mtime_after_first = os.path.getmtime(affine_path)

    rec2 = evaluate_mindboggle_pair(
        pair_idx=0, model="sobolev", device="cpu",
        pairs_csv=pairs_csv, data_dir=data_dir,
        canonical_affine_dir=canonical_affine_dir, use_n4=False,
        reg_iterations=[4, 4, 2],
    )
    assert os.path.getmtime(affine_path) == mtime_after_first
    assert rec1["affine_dice_sym"] == pytest.approx(rec2["affine_dice_sym"], abs=1e-9)
