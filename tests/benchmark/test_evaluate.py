"""End-to-end tests for antstorch.benchmark.evaluate.evaluate_mindboggle_pair()
against a tiny synthetic Mindboggle-style pair, covering all 6 ANTsTorch-native
model variants (gaussian_syn/sobolev_syn/dsti_syn/bspline_syn -- the four
antstorch.syn.syn_registration() regularizers, dense-SyN-stage only -- plus
bspline_svf and gaussian_svf, two stationary-velocity families), plus
ants_syn_quick, a traditional (non-ANTsTorch) ANTs baseline run via a direct
ants.registration() call. Runs the real registration/metric pipeline (no
mocking) on a small volume so wall-clock time stays low."""
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


def test_evaluate_mindboggle_pair_ants_syn_quick(mock_mindboggle_dataset, tmp_path):
    # Traditional (non-ANTsTorch) ANTs baseline, added on top of the six
    # ANTsTorch-native model variants above: a direct
    # ants.registration(type_of_transform="antsRegistrationSyNQuick[so]")
    # call, run deformable-only on top of the harness's shared canonical
    # affine (see _ANTS_TRADITIONAL_MODELS/_run_ants_traditional() in
    # evaluate.py). No reg_iterations override here, deliberately: unlike
    # every other model family, this one has no harness-level default (the
    # antsRegistrationSyNQuick[x] preset supplies its own internal
    # multi-resolution schedule) -- the mock dataset's small volume keeps
    # this fast even at the preset's own default iteration counts.
    pairs_csv, data_dir = mock_mindboggle_dataset
    registration_output_dir = tmp_path / "pair_000" / "ants_syn_quick"
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="ants_syn_quick",
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        registration_output_dir=str(registration_output_dir),
        use_n4=False,
    )
    _assert_valid_success_record(rec, "ants_syn_quick")
    # ants.registration()'s own transform-file convention: fwdtransforms is
    # [warp, affine], invtransforms is [affine, inverse_warp] -- confirmed
    # directly against real Mindboggle data (project doc, § 32) to already
    # match what evaluate_mindboggle_pair() defaults whichtoinvert_inv to
    # ([True, False]) when a model doesn't supply its own, so no § 30-style
    # composition fixup is needed for this model family.
    assert len(rec["transforms"]["fwdtransforms"]) == 2
    assert len(rec["transforms"]["invtransforms"]) == 2
    assert rec["warped_moving"] == str(registration_output_dir / "warped_moving.nii.gz")
    assert os.path.exists(rec["warped_moving"])
    assert rec["warped_moving_labels"] == str(registration_output_dir / "warped_moving_labels.nii.gz")
    assert os.path.exists(rec["warped_moving_labels"])
    # No loss_history for this model family (ants.registration() has no
    # such concept) -- must stay None rather than erroring.
    assert rec.get("loss_history") is None


def test_ants_traditional_models_use_deformable_only_presets():
    # Documents/locks the fairness-invariant requirement itself (see the
    # docstring on _ANTS_TRADITIONAL_MODELS in evaluate.py): every entry
    # must be a deformable-only preset -- "SyNOnly" outright, or an
    # "antsRegistrationSyN*[x]" suffix ending in "o" (the "...[so]"/
    # "...[bo]" deformable-only presets), signifying no separate ANTs
    # rigid/affine stage duplicated on top of the shared canonical affine.
    # A future preset that violates this (e.g. a bare "...[s]"/"...[b]"
    # full-pipeline variant) would silently break the fairness invariant
    # every other model in this harness preserves -- this test exists so
    # that mistake fails loudly instead. Reuses the harness's own
    # _is_deformable_only_ants_transform() (single source of truth for this
    # check -- the same function evaluate_mindboggle_pair()'s dispatch uses
    # to decide whether a raw type_of_transform string is even eligible)
    # rather than re-implementing the pattern match here.
    from antstorch.benchmark.evaluate import _ANTS_TRADITIONAL_MODELS, _is_deformable_only_ants_transform

    for model_name, type_of_transform in _ANTS_TRADITIONAL_MODELS.items():
        assert _is_deformable_only_ants_transform(type_of_transform), (
            f"{model_name!r} -> {type_of_transform!r} is not a deformable-only "
            "preset; it would recompute its own rigid/affine stage and break "
            "the shared-canonical-affine fairness invariant"
        )


@pytest.mark.parametrize(
    "type_of_transform", ["SyNOnly", "antsRegistrationSyNQuick[so]", "antsRegistrationSyN[bo]"]
)
def test_is_deformable_only_ants_transform_accepts_deformable_only_presets(type_of_transform):
    from antstorch.benchmark.evaluate import _is_deformable_only_ants_transform

    assert _is_deformable_only_ants_transform(type_of_transform)


@pytest.mark.parametrize(
    "type_of_transform", ["SyN", "antsRegistrationSyNQuick[s]", "antsRegistrationSyN[b]", "Affine"]
)
def test_is_deformable_only_ants_transform_rejects_full_pipeline_presets(type_of_transform):
    from antstorch.benchmark.evaluate import _is_deformable_only_ants_transform

    assert not _is_deformable_only_ants_transform(type_of_transform)


def test_evaluate_mindboggle_pair_accepts_raw_ants_type_of_transform_directly(mock_mindboggle_dataset, tmp_path):
    # The user should be able to pass the exact ants.registration()
    # type_of_transform string as `model`, with no _ANTS_TRADITIONAL_MODELS
    # alias required -- e.g. model="antsRegistrationSyNQuick[so]" directly,
    # matching what they'd type when calling ants.registration() themselves.
    # Matched case-sensitively against the *original* `model` (not
    # model_lower): ants.registration()'s type_of_transform strings are
    # themselves case-sensitive, so this must not go through the harness's
    # usual lowercasing.
    pairs_csv, data_dir = mock_mindboggle_dataset
    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="antsRegistrationSyNQuick[so]",
        device="cpu",
        pairs_csv=pairs_csv,
        data_dir=data_dir,
        canonical_affine_dir=str(tmp_path / "canonical_affines"),
        use_n4=False,
    )
    assert rec["status"] == "SUCCESS"
    assert np.isfinite(rec["dice_sym"])
    assert 0.0 <= rec["dice_sym"] <= 1.0
    assert len(rec["transforms"]["fwdtransforms"]) == 2
    assert len(rec["transforms"]["invtransforms"]) == 2


def test_evaluate_mindboggle_pair_rejects_full_pipeline_ants_transform(mock_mindboggle_dataset, tmp_path):
    # A raw type_of_transform string that recomputes its own rigid/affine
    # stage (no deformable-only suffix) must still be rejected -- passing
    # the model string through directly must not bypass the
    # fairness-invariant guard that _ANTS_TRADITIONAL_MODELS entries get.
    pairs_csv, data_dir = mock_mindboggle_dataset
    with pytest.raises(ValueError, match="Unknown registration model"):
        evaluate_mindboggle_pair(
            pair_idx=0,
            model="antsRegistrationSyNQuick[s]",
            device="cpu",
            pairs_csv=pairs_csv,
            data_dir=data_dir,
            canonical_affine_dir=str(tmp_path / "canonical_affines"),
            use_n4=False,
        )


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


@pytest.mark.parametrize("model", ["bspline_svf", "gaussian_svf"])
def test_svf_models_warpedmovout_extrapolation_matches_fwdtransforms_files(mock_mindboggle_dataset, tmp_path, model):
    """Regression test for the bug reported right after warped_moving_labels.nii.gz
    was added (§ 28): warpedmovout was reconstructed with padding_mode="border"
    (clamp-to-edge), while the *same* fwdtransforms, applied through
    ants.apply_transforms() -- as both the warped label map and
    compute_bidirectional_dice() do -- always extrapolate out-of-domain
    points to 0. For the ~5% of voxels this mock pair's canonical affine
    pushes outside the moving image's domain, that mismatch made
    warpedmovout look like a smooth continuation of the brain while the
    label map went to background in the exact same voxels -- "the label
    image doesn't correspond to the transformed image". warpedmovout must
    extrapolate the same way (padding_mode="zeros") as
    ants.apply_transforms(transformlist=fwdtransforms) does, so all three
    consumers of the same transform (warpedmovout, the label map, Dice)
    agree everywhere, not just in the interior."""
    from antstorch.benchmark.evaluate import _fit_or_load_canonical_affine, _run_bspline_svf, _run_gaussian_svf

    pairs_csv, data_dir = mock_mindboggle_dataset
    fixed_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz")
    moving_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-2", "t1weighted_brain.nii.gz")
    fi = ants.image_read(fixed_path)
    mi = ants.image_read(moving_path)

    matrix, translation, _, _ = _fit_or_load_canonical_affine(
        fi, mi, 0, str(tmp_path / "canonical_affines"), "cpu", False
    )
    run_fn = _run_bspline_svf if model == "bspline_svf" else _run_gaussian_svf
    extra = {} if model == "bspline_svf" else dict(update_field_sigma=1.0, total_field_sigma=0.25, squaring_steps=2)
    res = run_fn(
        fi, mi, matrix, translation, device="cpu", reg_iterations=[1, 1, 1, 1],
        outprefix=str(tmp_path / "reg_"), verbose=False, **extra,
    )

    warpedmovout = res["warpedmovout"].numpy()
    warped_via_transformlist = ants.apply_transforms(
        fixed=fi, moving=mi, transformlist=res["fwdtransforms"], interpolator="linear"
    ).numpy()

    # Wherever ants.apply_transforms() extrapolates to 0 (out-of-domain),
    # warpedmovout must trend toward 0 there too -- not sit at a border-
    # clamped value comparable to the rest of the (real, nonzero-background)
    # brain image -- since the label map and Dice score are computed from
    # the exact same extrapolation. A tolerance-based ratio (rather than
    # requiring every voxel to be exactly 0) accounts for the narrow
    # boundary band where torch's grid_sample and ITK/ants' own resampler
    # can disagree by a fraction of a voxel on the in-/out-of-domain
    # boundary itself -- padding_mode="border" (the bug) makes this ratio
    # ~0.86 on this fixture; padding_mode="zeros" (the fix) brings it to ~0.17.
    extrapolated = warped_via_transformlist < 1e-3
    assert extrapolated.sum() > 0, "test fixture assumption failed: expected some out-of-domain voxels"
    extrapolated_mean = np.abs(warpedmovout[extrapolated]).mean()
    overall_mean = np.abs(warpedmovout).mean()
    ratio = extrapolated_mean / overall_mean
    assert ratio < 0.5, (
        f"{model}: warpedmovout's mean value in the region ants.apply_transforms() (and therefore the "
        f"warped label map) extrapolates to background is {ratio:.2f}x the image's overall mean -- too "
        "close to a border-clamped continuation of the brain rather than trending to 0, indicating a "
        "padding_mode mismatch between warpedmovout and the fwdtransforms files"
    )
    # And overall the two should closely agree (same geometric transform).
    corr = np.corrcoef(warpedmovout.ravel(), warped_via_transformlist.ravel())[0, 1]
    assert corr > 0.9, f"{model}: correlation {corr:.4f} between warpedmovout and its own fwdtransforms too low"


@pytest.mark.parametrize("model", ["bspline_svf", "gaussian_svf"])
def test_svf_models_write_single_composed_transform_files(mock_mindboggle_dataset, tmp_path, model):
    """Regression test for § 30: ants.apply_transforms()'s real composite-
    transform convention for transformlist=[warp_path, affine_path] turned
    out (verified empirically against real Mindboggle output, see the
    project doc) to compose warp-then-affine, the *opposite* order from
    what _run_bspline_svf()/_run_gaussian_svf() used to reconstruct
    warpedmovout ("affine first, then the SVF flow", the model's own
    internal training-time convention). Reusing the raw pure-SVF field next
    to a separate affine file therefore silently mismatched what
    ants.apply_transforms() reconstructs for the label map / Dice, even
    after § 29's padding fix. The fix: write a single, already-composed
    field for both directions and drop the separate affine piece from
    fwdtransforms/invtransforms entirely."""
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

    assert len(rec["transforms"]["fwdtransforms"]) == 1
    assert len(rec["transforms"]["invtransforms"]) == 1
    assert rec["transforms"]["whichtoinvert_inv"] == [False]
    # The affine .mat is still written to disk for transparency/debugging,
    # but is no longer part of the transform lists themselves.
    assert os.path.exists(registration_output_dir / "registration_0GenericAffine.mat")
    assert not any(p.endswith(".mat") for p in rec["transforms"]["fwdtransforms"])
    assert not any(p.endswith(".mat") for p in rec["transforms"]["invtransforms"])


@pytest.mark.parametrize("model", ["bspline_svf", "gaussian_svf"])
def test_svf_models_warpedmovout_matches_fwdtransforms_applied_to_intensity(mock_mindboggle_dataset, tmp_path, model):
    """Tighter regression test than the existing extrapolation-ratio check:
    with § 30's single-composed-field fix, warping the moving *intensity*
    image via ants.apply_transforms(transformlist=fwdtransforms) -- exactly
    what warped_moving_labels.nii.gz does for the label map -- must now
    agree almost exactly with warpedmovout (same underlying field, single
    piece, no separate affine re-composition for ants to get wrong), not
    just be strongly correlated."""
    from antstorch.benchmark.evaluate import _fit_or_load_canonical_affine, _run_bspline_svf, _run_gaussian_svf

    pairs_csv, data_dir = mock_mindboggle_dataset
    fixed_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz")
    moving_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-2", "t1weighted_brain.nii.gz")
    fi = ants.image_read(fixed_path)
    mi = ants.image_read(moving_path)

    matrix, translation, _, _ = _fit_or_load_canonical_affine(
        fi, mi, 0, str(tmp_path / "canonical_affines"), "cpu", False
    )
    run_fn = _run_bspline_svf if model == "bspline_svf" else _run_gaussian_svf
    extra = {} if model == "bspline_svf" else dict(update_field_sigma=1.0, total_field_sigma=0.25, squaring_steps=2)
    res = run_fn(
        fi, mi, matrix, translation, device="cpu", reg_iterations=[1, 1, 1, 1],
        outprefix=str(tmp_path / "reg_"), verbose=False, **extra,
    )

    warpedmovout = res["warpedmovout"].numpy()
    warped_via_transformlist = ants.apply_transforms(
        fixed=fi, moving=mi, transformlist=res["fwdtransforms"], interpolator="linear"
    ).numpy()

    # This mock dataset's volume is tiny (see conftest.py), so the boundary/
    # extrapolation band is a much larger fraction of the total volume here
    # than on a real ~256^3 Mindboggle image -- on real data (see the
    # project doc) this same comparison came out numerically identical
    # (100% voxel agreement on a categorical test pattern). 0.95 is still a
    # meaningfully tighter bound than the pre-existing extrapolation-ratio
    # test's 0.9 threshold, which predates this fix.
    corr = np.corrcoef(warpedmovout.ravel(), warped_via_transformlist.ravel())[0, 1]
    assert corr > 0.95, (
        f"{model}: correlation {corr:.6f} between warpedmovout and ants.apply_transforms(fwdtransforms) "
        "applied to the same intensity image -- expected near-exact agreement now that both use the "
        "same single, already-composed field"
    )
    diff = np.abs(warpedmovout - warped_via_transformlist)
    assert np.median(diff) < 1.0, f"{model}: median abs diff {np.median(diff):.4f} too large for the same field"


@pytest.mark.parametrize("model", ["bspline_svf", "gaussian_svf"])
def test_svf_models_loss_history_persisted(mock_mindboggle_dataset, tmp_path, model):
    """§ 30: registration_output_dir must also persist loss_history.json for
    the two SVF model families, so the actual optimization convergence
    curve can be inspected after a run."""
    import json

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
        reg_iterations=[2, 2, 1, 1],
    )
    if model == "gaussian_svf":
        kwargs.update(update_field_sigma=1.0, total_field_sigma=0.25, squaring_steps=2)
    rec = evaluate_mindboggle_pair(**kwargs)
    _assert_valid_success_record(rec, model)

    loss_history_path = rec["loss_history"]
    assert loss_history_path == str(registration_output_dir / "loss_history.json")
    assert os.path.exists(loss_history_path)
    with open(loss_history_path) as stream:
        payload = json.load(stream)
    assert "loss_history" in payload and "level_loss_history" in payload
    assert len(payload["loss_history"]) > 0
    assert all(isinstance(v, float) for v in payload["loss_history"])
    assert len(payload["level_loss_history"]) == 4  # one entry per shrink factor/level


def test_loss_history_absent_without_registration_output_dir(mock_mindboggle_dataset, tmp_path):
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
    assert rec["loss_history"] is None


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
