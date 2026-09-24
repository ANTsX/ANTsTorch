import ants
import numpy as np
import pytest
import torch

from antstorch.syn import syn_registration
from antstorch.syn.bridge import ants_image_metadata, metadata_tensors
from antstorch.syn.syn import _apply_regularizer, _compose_fixed_grid, _physical_grid


def _blob_2d(size, center, sigma=5.0, ramp=0.0):
    yy, xx = np.mgrid[0 : size[0], 0 : size[1]].astype(np.float32)
    array = np.exp(-(((yy - center[0]) / sigma) ** 2 + ((xx - center[1]) / sigma) ** 2))
    if ramp:
        array = array + ramp * np.clip(xx - center[1] - sigma, 0, None) / size[1]
    return array.astype(np.float32)


def _ants_pair_2d(size=(30, 28), fixed_center=(17, 16), moving_center=(11, 10), ramp=0.0):
    fixed_arr = _blob_2d(size, fixed_center, ramp=ramp)
    moving_arr = _blob_2d(size, moving_center, ramp=ramp)
    fixed = ants.from_numpy(np.ascontiguousarray(fixed_arr.T))
    moving = ants.from_numpy(np.ascontiguousarray(moving_arr.T))
    return fixed, moving


def _blob_3d(size, center, sigma=3.0):
    zz, yy, xx = np.mgrid[0 : size[0], 0 : size[1], 0 : size[2]].astype(np.float32)
    array = np.exp(
        -(((zz - center[0]) / sigma) ** 2 + ((yy - center[1]) / sigma) ** 2 + ((xx - center[2]) / sigma) ** 2)
    )
    return array.astype(np.float32)


def _ants_pair_3d(size=(10, 12, 11), fixed_center=(5, 6, 6), moving_center=(4, 5, 5)):
    fixed_arr = _blob_3d(size, fixed_center)
    moving_arr = _blob_3d(size, moving_center)
    fixed = ants.from_numpy(np.ascontiguousarray(fixed_arr.transpose(2, 1, 0)))
    moving = ants.from_numpy(np.ascontiguousarray(moving_arr.transpose(2, 1, 0)))
    return fixed, moving


# --- Parameter validation ---------------------------------------------------


def test_syn_registration_rejects_unknown_type_of_transform():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="type_of_transform"):
        syn_registration(fixed, moving, type_of_transform="Nonsense")


def test_syn_registration_rejects_unknown_syn_metric():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="syn_metric"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", syn_metric="nonsense")


def test_syn_registration_rejects_unknown_regularizer():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="regularizer"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", regularizer="nonsense")


def test_syn_registration_rejects_unknown_padding_mode():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="padding_mode"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", padding_mode="wrap")


def test_syn_registration_rejects_mismatched_pyramid_lengths():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="levels and reg_iterations"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", levels=(2, 1), reg_iterations=(10,))


def test_syn_registration_rejects_mismatched_dimensions():
    fixed, _ = _ants_pair_2d()
    moving3d, _ = _ants_pair_3d()
    with pytest.raises(ValueError, match="dimension"):
        syn_registration(fixed, moving3d, type_of_transform="SyNOnly")


def test_syn_registration_rejects_batched_initial_affine():
    fixed, moving = _ants_pair_2d()
    matrix = torch.eye(2).unsqueeze(0)  # (1, 2, 2): batched, not accepted here.
    translation = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="initial_affine"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", initial_affine=(matrix, translation))


def test_syn_registration_auto_detected_mps_falls_back_to_cpu(monkeypatch):
    # bspline_flows.affine_registration()'s differentiable warp backpropagates
    # through F.grid_sample, whose backward is not implemented on MPS in the
    # PyTorch versions this has been tested against (raises NotImplementedError
    # for aten::grid_sampler_2d_backward). On an Apple Silicon machine,
    # auto_detect_device() would otherwise silently pick 'mps' and crash any
    # syn_registration() call that performs an affine fit. Simulate that
    # hardware probe here (regardless of what this machine actually has) and
    # confirm syn_registration() steers auto-detection away from mps.
    monkeypatch.setattr("antstorch.syn.syn.auto_detect_device", lambda **kwargs: "mps")
    fixed, moving = _ants_pair_2d()
    result = syn_registration(
        fixed, moving, type_of_transform="Translation",
        affine_iterations=(5,), affine_shrink_factors=(1,), affine_smoothing_sigmas=(0.0,),
        affine_learning_rate=(0.05,),
    )
    assert result["provenance"]["device"] == "cpu"


def test_syn_registration_explicit_mps_request_is_honored(monkeypatch):
    # An explicit device='mps' request is trusted verbatim (no silent
    # fallback) -- the caller may know their PyTorch build/op combination
    # works, or be deliberately opting in ahead of a future PyTorch fix.
    # auto_detect_device() must never be consulted in this path; that is the
    # only thing this test asserts, via a sentinel that fails loudly if it
    # is. What happens *after* resolution is deliberately left unchecked:
    # it depends on the machine running the test. On real Apple Silicon
    # hardware, SyNOnly with no internal affine fit runs to completion on
    # 'mps' without error (its dense warp never hits the unsupported
    # grid_sampler backward op); on a machine with no MPS backend compiled
    # in at all (e.g. this sandbox), the very first tensor allocated there
    # raises NotImplementedError instead. Both outcomes are acceptable here.
    def _fail_if_called(**kwargs):
        raise AssertionError("auto_detect_device must not be called when device is given explicitly")

    monkeypatch.setattr("antstorch.syn.syn.auto_detect_device", _fail_if_called)
    fixed, moving = _ants_pair_2d()
    try:
        syn_registration(fixed, moving, type_of_transform="SyNOnly", levels=(1,), reg_iterations=(1,), device="mps")
    except AssertionError:
        raise
    except Exception:
        pass


# --- 3-D MPS grid_sample gating (forward-only vs. combined probe) ----------
#
# syn_registration()'s explicit-device='mps'/dimension==3 gate (see
# needs_internal_affine_fit in syn.py) uses one of two availability probes
# depending on whether an internal affine fit will run: the combined
# forward+backward probe (mps_grid_sample_3d_available) when it will
# (bspline_flows.affine_registration() backpropagates through a raw
# F.grid_sample call and needs real backward), or the weaker forward-only
# probe (mps_grid_sample_3d_forward_available) when it won't -- SyNOnly, or
# any call given an explicit initial_affine -- since _fit_syn_level()'s own
# grid_sample calls always route through AnalyticalGridSample there (see
# mps_grid_sample_3d_forward_available's docstring), which never needs
# backward. These tests exercise the gating decision itself (which probe is
# consulted, and whether its result triggers the CPU fallback) by mocking
# both probes and an explicit device='mps' request; they do not require
# real MPS hardware, and do not assert anything about whether the
# subsequent (unmocked) computation on a fake 'mps' device succeeds.

def _run_3d_mps_synonly(monkeypatch, forward_available, initial_affine=None):
    monkeypatch.setattr("antstorch.syn.syn.mps_grid_sample_3d_forward_available", lambda: forward_available)

    def _fail_if_called():
        raise AssertionError("mps_grid_sample_3d_available (combined probe) must not be called here")

    monkeypatch.setattr("antstorch.syn.syn.mps_grid_sample_3d_available", _fail_if_called)
    fixed, moving = _ants_pair_3d()
    kwargs = dict(type_of_transform="SyNOnly", levels=(1,), reg_iterations=(1,), device="mps")
    if initial_affine is not None:
        kwargs["initial_affine"] = initial_affine
    try:
        syn_registration(fixed, moving, **kwargs)
    except AssertionError:
        raise
    except Exception:
        pass


def test_syn_registration_3d_mps_synonly_uses_forward_only_probe_and_falls_back_when_missing(monkeypatch, recwarn):
    _run_3d_mps_synonly(monkeypatch, forward_available=False)
    matching = [w for w in recwarn.list if "no forward MPS kernel" in str(w.message)]
    assert len(matching) == 1


def test_syn_registration_3d_mps_synonly_uses_forward_only_probe_and_stays_on_mps_when_available(monkeypatch, recwarn):
    _run_3d_mps_synonly(monkeypatch, forward_available=True)
    matching = [w for w in recwarn.list if "grid_sample" in str(w.message) and "MPS kernel" in str(w.message)]
    assert matching == []


def test_syn_registration_3d_mps_synonly_with_explicit_initial_affine_also_uses_forward_only_probe(monkeypatch, recwarn):
    # initial_affine supplied -> no internal affine fit regardless of
    # type_of_transform, so this must take the same forward-only-probe path
    # as plain SyNOnly, not the combined-probe path.
    _run_3d_mps_synonly(
        monkeypatch, forward_available=True,
        initial_affine=(torch.eye(3), torch.zeros(3)),
    )
    matching = [w for w in recwarn.list if "grid_sample" in str(w.message) and "MPS kernel" in str(w.message)]
    assert matching == []


def test_syn_registration_3d_mps_with_internal_affine_fit_uses_combined_probe(monkeypatch, recwarn):
    # type_of_transform='SyN' with no initial_affine -> an internal affine
    # fit *will* run (bspline_flows.affine_registration(), unmigrated, needs
    # real backward), so the combined forward+backward probe must be
    # consulted here, not the forward-only one.
    monkeypatch.setattr("antstorch.syn.syn.mps_grid_sample_3d_available", lambda: False)

    def _fail_if_called():
        raise AssertionError("mps_grid_sample_3d_forward_available must not be called here")

    monkeypatch.setattr("antstorch.syn.syn.mps_grid_sample_3d_forward_available", _fail_if_called)
    fixed, moving = _ants_pair_3d()
    try:
        syn_registration(
            fixed, moving, type_of_transform="SyN",
            affine_iterations=(1,), affine_shrink_factors=(1,), affine_smoothing_sigmas=(0.0,),
            affine_learning_rate=(0.05,), levels=(1,), reg_iterations=(1,), device="mps",
        )
    except AssertionError:
        raise
    except Exception:
        pass
    matching = [w for w in recwarn.list if "backward gap" in str(w.message)]
    assert len(matching) == 1


# --- Linear-only transform types --------------------------------------------


@pytest.mark.parametrize("transform_type", ["Translation", "Rigid", "Affine"])
def test_syn_registration_linear_only_reduces_loss_and_has_no_syn_fields(transform_type):
    fixed, moving = _ants_pair_2d()
    result = syn_registration(
        fixed,
        moving,
        type_of_transform=transform_type,
        affine_iterations=(20, 15),
        affine_shrink_factors=(2, 1),
        affine_smoothing_sigmas=(0.5, 0.0),
        affine_learning_rate=(0.05, 0.02),
    )
    assert result["loss_history"] is None
    assert result["jacobian"] is None
    assert result["warpedmovout"].shape == fixed.shape
    # Linear-only: a single shared 0GenericAffine.mat file, reused (per
    # ants.registration()'s own convention) in both directions.
    assert result["fwdtransforms"] == [result["provenance"]["outprefix"] + "0GenericAffine.mat"]
    assert result["invtransforms"] == result["fwdtransforms"]
    affine_tx = ants.read_transform(result["fwdtransforms"][0])
    assert affine_tx.dimension == 2
    history = result["affine_loss_history"][0]
    before = float(np.mean((fixed.numpy() - moving.numpy()) ** 2))
    after = float(np.mean((fixed.numpy() - result["warpedmovout"].numpy()) ** 2))
    # Center-of-mass initialization already resolves most of this
    # translation-dominated example almost immediately, so a strict
    # last-iteration-vs-first-iteration comparison is not robust (a few
    # additional degrees of freedom can wobble around an already-tiny
    # loss); checking against the unregistered baseline is.
    assert min(history) < 0.1 * before
    assert after < before


def test_syn_registration_linear_only_with_explicit_initial_affine_skips_internal_fit():
    fixed, moving = _ants_pair_2d()
    matrix = torch.eye(2)
    translation = torch.tensor([0.5, -0.5])
    result = syn_registration(fixed, moving, type_of_transform="Affine", initial_affine=(matrix, translation))
    assert result["affine_loss_history"] is None
    torch.testing.assert_close(result["affine_matrix"], matrix)
    torch.testing.assert_close(result["affine_translation"], translation)


# --- Dense SyN stage ---------------------------------------------------------


def test_syn_only_reduces_loss():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
    )
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert len(result["level_loss_history"]) == 2
    assert result["provenance"]["affine_fit"] is False
    torch.testing.assert_close(result["affine_matrix"], torch.eye(2))


def test_syn_only_with_identity_initial_affine_matches_no_initial_affine():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    kwargs = dict(type_of_transform="SyNOnly", levels=(1,), reg_iterations=(10,), syn_metric="mse")
    baseline = syn_registration(fixed, moving, **kwargs)
    explicit = syn_registration(fixed, moving, initial_affine=(torch.eye(2), torch.zeros(2)), **kwargs)
    torch.testing.assert_close(
        torch.tensor(baseline["loss_history"]), torch.tensor(explicit["loss_history"])
    )


def test_syn_default_type_of_transform_fits_an_internal_affine_first():
    fixed, moving = _ants_pair_2d()
    result = syn_registration(
        fixed, moving, type_of_transform="SyN",
        affine_iterations=(15, 10), affine_shrink_factors=(2, 1), affine_smoothing_sigmas=(0.5, 0.0),
        affine_learning_rate=(0.05, 0.02),
        levels=(1,), reg_iterations=(10,), syn_metric="mse",
    )
    assert result["provenance"]["affine_fit"] is True
    assert result["affine_loss_history"] is not None
    before = float(np.mean((fixed.numpy() - moving.numpy()) ** 2))
    after = float(np.mean((fixed.numpy() - result["warpedmovout"].numpy()) ** 2))
    assert min(result["loss_history"]) < 0.1 * before
    assert after < before


def test_syn_only_reduces_loss_with_bspline_regularizer():
    # mesh_size=2 rather than the ITK class default of 1 (4 control points):
    # on a tiny synthetic image, a 4-control-point lattice is so coarse that
    # most of its few degrees of freedom go to satisfying the zero
    # (enforce_stationary_boundary) boundary constraint, which can transiently
    # *raise* the loss for the first iteration or two before it settles into a
    # monotonic decrease (verified interactively; not a bug -- ANTs' own
    # scripts, e.g. antsRegistrationSyN.sh, likewise never use the raw ITK
    # class-default mesh size in practice, always specifying a much finer one).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=2,
    )
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert result["provenance"]["regularizer"] == "bspline"
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == 2
    assert result["provenance"]["total_field_mesh_size_at_base_level"] == 0


# --- gaussian_sigma_mode / conservative_smooth (syntx-parity knobs) --------
#
# Added after comparing this port against syntx.syn on real Mindboggle-101
# pairs (project doc, "comparaison syntx/antstorch, écart gaussian/sobolev"):
# syntx.syn's own default 'gaussian' path applies flow_sigma directly in
# voxels (no physical-spacing scaling), and its default 'sobolev'/'dsti'
# path ("conservative mode") stacks a second spatial Gaussian pass on top of
# the spectral Green's operator -- both different from this port's own
# long-standing defaults. gaussian_sigma_mode/conservative_smooth let a
# caller opt into syntx's conventions without changing this port's defaults.

def _anisotropic_field_2d(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 12, 10, 2)


def test_apply_regularizer_gaussian_sigma_mode_defaults_to_physical():
    field = _anisotropic_field_2d()
    default = _apply_regularizer(field, "gaussian", 2.0, (2.0, 1.0))
    explicit_physical = _apply_regularizer(field, "gaussian", 2.0, (2.0, 1.0), gaussian_sigma_mode="physical")
    torch.testing.assert_close(default, explicit_physical)


def test_apply_regularizer_gaussian_sigma_mode_voxel_differs_from_physical_when_anisotropic():
    field = _anisotropic_field_2d()
    physical = _apply_regularizer(field, "gaussian", 2.0, (2.0, 1.0), gaussian_sigma_mode="physical")
    voxel = _apply_regularizer(field, "gaussian", 2.0, (2.0, 1.0), gaussian_sigma_mode="voxel")
    assert not torch.allclose(physical, voxel)


@pytest.mark.parametrize("regularizer", ["sobolev", "dsti"])
def test_apply_regularizer_conservative_smooth_defaults_to_off(regularizer):
    field = _anisotropic_field_2d()
    default = _apply_regularizer(field, regularizer, 2.0, (1.0, 1.0))
    explicit_off = _apply_regularizer(field, regularizer, 2.0, (1.0, 1.0), conservative_smooth=False)
    torch.testing.assert_close(default, explicit_off)


@pytest.mark.parametrize("regularizer", ["sobolev", "dsti"])
def test_apply_regularizer_conservative_smooth_true_differs_from_default(regularizer):
    field = _anisotropic_field_2d()
    default = _apply_regularizer(field, regularizer, 2.0, (1.0, 1.0))
    conservative = _apply_regularizer(field, regularizer, 2.0, (1.0, 1.0), conservative_smooth=True)
    assert not torch.allclose(default, conservative)


def test_apply_regularizer_conservative_smooth_is_a_no_op_for_gaussian_and_bspline():
    # conservative_smooth only means anything for the spectral regularizers
    # (sobolev/dsti); passing it for gaussian must not change behavior or
    # raise -- _apply_regularizer's gaussian branch never reads it.
    field = _anisotropic_field_2d()
    default = _apply_regularizer(field, "gaussian", 2.0, (1.0, 1.0))
    with_flag = _apply_regularizer(field, "gaussian", 2.0, (1.0, 1.0), conservative_smooth=True)
    torch.testing.assert_close(default, with_flag)


def test_syn_only_reduces_loss_with_conservative_smooth_sobolev():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
        regularizer="sobolev", conservative_smooth=True,
    )
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert result["provenance"]["conservative_smooth"] is True


def test_syn_only_reduces_loss_with_voxel_gaussian_sigma_mode():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
        regularizer="gaussian", gaussian_sigma_mode="voxel",
    )
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert result["provenance"]["gaussian_sigma_mode"] == "voxel"


def test_syn_registration_default_provenance_records_syntx_parity_knob_defaults():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse",
    )
    assert result["provenance"]["gaussian_sigma_mode"] == "physical"
    assert result["provenance"]["conservative_smooth"] is False


def test_syn_registration_bspline_regularizer_supports_total_field_smoothing_too():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(10,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=1,
        total_field_mesh_size_at_base_level=1,
    )
    assert np.isfinite(result["loss_history"]).all()
    assert result["provenance"]["total_field_mesh_size_at_base_level"] == 1


def test_syn_registration_bspline_regularizer_requires_a_positive_mesh_size():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="mesh_size"):
        syn_registration(
            fixed, moving, type_of_transform="SyNOnly", regularizer="bspline",
            update_field_mesh_size_at_base_level=0, total_field_mesh_size_at_base_level=0,
        )


def test_syn_registration_default_update_field_mesh_size_is_none_sentinel():
    import inspect

    from antstorch.syn.syn import syn_registration as _fn

    assert inspect.signature(_fn).parameters["update_field_mesh_size_at_base_level"].default is None


def test_syn_registration_bspline_default_resolves_to_26mm_spline_distance():
    # Per the user's explicit 2026-08-26 instruction ("le defaut pour tous
    # les recalages de bspline soit 26 mm"), leaving
    # update_field_mesh_size_at_base_level unset (regularizer='bspline') now
    # resolves exactly as if update_field_spline_distance=26.0 had been
    # passed, replacing the old literal update_field_mesh_size_at_base_level=1
    # default -- and matches bspline_svf_registration()'s own new default
    # (same DEFAULT_BSPLINE_SPLINE_DISTANCE_MM constant).
    from antstorch.bspline_flows import mesh_size_for_spline_distance
    from antstorch.bspline_flows.bspline_svf_registration import DEFAULT_BSPLINE_SPLINE_DISTANCE_MM
    from antstorch.syn.bridge import ants_image_metadata, image_domain_from_metadata

    assert DEFAULT_BSPLINE_SPLINE_DISTANCE_MM == 26.0

    fixed, moving = _ants_pair_2d(ramp=0.3)
    fixed_domain = image_domain_from_metadata(ants_image_metadata(fixed))
    expected_mesh = mesh_size_for_spline_distance(fixed_domain, DEFAULT_BSPLINE_SPLINE_DISTANCE_MM)

    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline",
    )
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == expected_mesh
    assert result["provenance"]["update_field_spline_distance"] is None
    assert result["provenance"]["total_field_mesh_size_at_base_level"] == 0
    assert np.isfinite(result["loss_history"]).all()


def test_syn_registration_explicit_update_field_mesh_size_still_overrides_default():
    # Backward compatibility: an explicit update_field_mesh_size_at_base_level
    # continues to bypass the new 26mm default entirely (unchanged behavior).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=4,
    )
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == 4


def test_syn_registration_non_bspline_regularizer_default_normalizes_to_zero():
    # regularizer != 'bspline': the None sentinel must never leak into
    # provenance -- it normalizes to 0 (matching the pre-existing
    # total_field_mesh_size_at_base_level=0 "off" convention), not the 26mm
    # default (which only applies when regularizer='bspline').
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="gaussian",
    )
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == 0


def test_syn_registration_rejects_negative_bspline_mesh_sizes():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="update_field_mesh_size_at_base_level"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", update_field_mesh_size_at_base_level=-1)
    with pytest.raises(ValueError, match="total_field_mesh_size_at_base_level"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", total_field_mesh_size_at_base_level=-1)


def test_syn_registration_verbose_bspline_regularizer_reports_control_points(capsys):
    # Mirrors bspline_svf_registration()'s own verbose control-point reporting
    # (antstorch/bspline_flows/bspline_svf_registration.py): same mesh_size -> control_points
    # = mesh_size + 3 relationship (cubic spline order), doubled from the base
    # level at each finer pyramid level exactly as _fit_syn_level itself doubles
    # it before regularizing. update_field_mesh_size_at_base_level=2 with
    # levels=(2, 1) should report control_points=(5, 5) at the coarsest level
    # (scale=1) and control_points=(7, 7) at the finest (scale=2). No aggregate
    # "total_control_points" count is reported -- it is just the product of the
    # per-axis tuple already shown, redundant information the user asked to drop.
    fixed, moving = _ants_pair_2d(ramp=0.3)
    syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(5, 5), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=2,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "control_points=(5, 5)" in out
    assert "control_points=(7, 7)" in out
    assert "total_control_points" not in out
    assert "total_field_control_points=" not in out


def test_syn_registration_verbose_bspline_regularizer_reports_total_field_control_points_too(capsys):
    fixed, moving = _ants_pair_2d(ramp=0.3)
    syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=1,
        total_field_mesh_size_at_base_level=1,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "control_points=(4, 4)" in out
    assert "total_field_control_points=(4, 4)" in out
    assert "total_control_points" not in out


def test_syn_registration_verbose_non_bspline_regularizer_omits_control_points(capsys):
    fixed, moving = _ants_pair_2d(ramp=0.3)
    syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(5, 5), syn_metric="mse", grad_step=0.4,
        regularizer="gaussian",
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "control_points=" not in out
    assert "iterations=5" in out


def test_syn_registration_update_field_spline_distance_resolves_per_axis_mesh():
    # fixed/moving from _ants_pair_2d() have ITK shape (28, 30), spacing
    # (1, 1) -> physical extent (27, 29) -> ANTs' own un-padded
    # ceil(extent / spline_distance) formula (mesh_size_for_spline_distance)
    # gives a genuinely *anisotropic* mesh from a single scalar distance --
    # ceil(27/14)=2, ceil(29/14)=3 -- exactly like real ANTs'
    # CalculateMeshSizeForSpecifiedKnotSpacing, whose SizeType output differs
    # per axis even for one scalar knot spacing whenever the domain isn't
    # square/cubic.
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_spline_distance=14.0,
    )
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == (2, 3)
    assert result["provenance"]["update_field_spline_distance"] == 14.0
    assert result["provenance"]["total_field_spline_distance"] is None
    assert np.isfinite(result["loss_history"]).all()


def test_syn_registration_total_field_spline_distance_resolves_per_axis_mesh():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=1,
        total_field_spline_distance=(9.0, 15.0),
    )
    assert result["provenance"]["total_field_mesh_size_at_base_level"] == (3, 2)
    assert result["provenance"]["total_field_spline_distance"] == (9.0, 15.0)
    assert np.isfinite(result["loss_history"]).all()


def test_syn_registration_spline_distance_mutually_exclusive_with_mesh_size():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="update_field_spline_distance"):
        syn_registration(
            fixed, moving, type_of_transform="SyNOnly", regularizer="bspline",
            update_field_mesh_size_at_base_level=2, update_field_spline_distance=14.0,
        )
    with pytest.raises(ValueError, match="total_field_spline_distance"):
        syn_registration(
            fixed, moving, type_of_transform="SyNOnly", regularizer="bspline",
            total_field_mesh_size_at_base_level=1, total_field_spline_distance=14.0,
        )


def test_syn_registration_spline_distance_alone_satisfies_positive_mesh_requirement():
    # update_field_mesh_size_at_base_level=0 (off) with only
    # total_field_spline_distance given must not trip the "regularizer=
    # 'bspline' requires an active mesh" validation -- it counts as active.
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(3,), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_mesh_size_at_base_level=0,
        total_field_spline_distance=14.0,
    )
    assert result["provenance"]["update_field_mesh_size_at_base_level"] == 0
    assert result["provenance"]["total_field_mesh_size_at_base_level"] == (2, 3)


def test_syn_registration_verbose_spline_distance_reports_doubled_per_axis_control_points(capsys):
    # Base mesh from update_field_spline_distance=14.0 is (2, 3) (see the
    # per-axis test above); doubled per finer pyramid level exactly like an
    # explicit integer mesh size -- level 0 (coarsest, scale=1):
    # control_points=(5, 6); level 1 (finest, scale=2): mesh (4, 6) ->
    # control_points=(7, 9).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(5, 5), syn_metric="mse", grad_step=0.4,
        regularizer="bspline", update_field_spline_distance=14.0,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "control_points=(5, 6)" in out
    assert "control_points=(7, 9)" in out


def test_syn_registration_reduces_intensity_mismatch_versus_unregistered():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyN",
        affine_iterations=(15, 10), affine_shrink_factors=(2, 1), affine_smoothing_sigmas=(0.5, 0.0),
        affine_learning_rate=(0.05, 0.02),
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
    )
    before = float(np.mean((fixed.numpy() - moving.numpy()) ** 2))
    after = float(np.mean((fixed.numpy() - result["warpedmovout"].numpy()) ** 2))
    assert after < before


def test_syn_registration_forward_inverse_are_approximate_half_warp_swaps():
    # A loose round-trip check: composing fwdtransforms with invtransforms on
    # the fixed grid should be close to identity (zero displacement),
    # exactly as the analytic affine+SVF composition in bspline_flows is
    # checked, but with a generous tolerance since the SyN half-warp
    # inverses are only maintained approximately (a handful of in-loop
    # Anderson steps, not driven to full convergence at these small
    # iteration counts).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(20, 20), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
    )
    meta = ants_image_metadata(fixed)
    meta_t = metadata_tensors(fixed, torch.device("cpu"), torch.float32)
    X_phys = _physical_grid(meta, torch.device("cpu"), torch.float32)

    def _field_to_tensor_zyx(image):
        array = image.numpy()[..., ::-1]
        dim = image.dimension
        axes = tuple(range(dim - 1, -1, -1)) + (dim,)
        return torch.from_numpy(np.ascontiguousarray(np.transpose(array, axes))).unsqueeze(0).float()

    # SyNOnly with no initial_affine: no affine ran, so each list holds only
    # the (inverse) warp path -- read it back from disk, proving the written
    # file round-trips exactly like the field would have in-memory before.
    assert result["fwdtransforms"] == [result["provenance"]["outprefix"] + "1Warp.nii.gz"]
    assert result["invtransforms"] == [result["provenance"]["outprefix"] + "1InverseWarp.nii.gz"]
    forward = _field_to_tensor_zyx(ants.image_read(result["fwdtransforms"][0]))
    inverse = _field_to_tensor_zyx(ants.image_read(result["invtransforms"][0]))
    composed = _compose_fixed_grid(forward, inverse, X_phys, meta_t)
    # Exclude a boundary margin, matching the analogous affine+SVF composition test.
    interior = composed[:, 2:-2, 2:-2]
    assert float(interior.abs().mean()) < 0.3


# --- Dimension generality ----------------------------------------------------


def test_syn_registration_supports_3d():
    fixed, moving = _ants_pair_3d()
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(4,), syn_metric="mse", grad_step=0.4, flow_sigma=1.5,
        in_loop_inverse_steps=2,
    )
    assert result["warpedmovout"].dimension == 3
    assert result["fwdtransforms"] == [result["provenance"]["outprefix"] + "1Warp.nii.gz"]
    warp_image = ants.image_read(result["fwdtransforms"][0])
    assert warp_image.components == 3
    assert torch.isfinite(torch.tensor(result["loss_history"])).all()


# --- Genuine ANTsX interop (Etape 3) -----------------------------------------


def test_syn_registration_fwdtransforms_are_usable_with_ants_apply_transforms():
    # The actual point of writing separate files: transformlist= should be
    # directly usable by ants.apply_transforms(), exactly as if it had come
    # from ants.registration() itself -- including the shared-affine-file
    # whichtoinvert default heuristic for invtransforms (matrix-then-warp).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyN",
        affine_iterations=(15, 10), affine_shrink_factors=(2, 1), affine_smoothing_sigmas=(0.5, 0.0),
        affine_learning_rate=(0.05, 0.02),
        levels=(1,), reg_iterations=(10,), syn_metric="mse",
    )
    assert len(result["fwdtransforms"]) == 2
    assert len(result["invtransforms"]) == 2

    # ants.apply_transforms() uses ITK's own resampler, independent of our
    # F.grid_sample-based warp_image() -- the two bilinear implementations
    # agree almost everywhere but differ by float32 interpolation noise at a
    # scattered handful of pixels (verified: mean abs diff ~2e-4, max
    # ~0.02, on a [0, 1]-ish intensity scale), so this checks agreement in
    # aggregate rather than requiring bit-identical output.
    applied_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=result["fwdtransforms"])
    diff = np.abs(applied_fwd.numpy() - result["warpedmovout"].numpy())
    assert diff.mean() < 1e-3
    assert diff.max() < 0.05

    # Round trip: fixed -> (fwd) -> moving space -> (inv) -> back to fixed
    # space should approximately recover the fixed image's own domain
    # (loose check -- this is a lossy resample round-trip, not an identity).
    applied_inv = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=result["invtransforms"])
    assert applied_inv.shape == moving.shape


def test_syn_registration_linear_only_fwdtransforms_usable_with_ants_apply_transforms():
    fixed, moving = _ants_pair_2d()
    result = syn_registration(
        fixed, moving, type_of_transform="Affine",
        affine_iterations=(15, 10), affine_shrink_factors=(2, 1), affine_smoothing_sigmas=(0.5, 0.0),
        affine_learning_rate=(0.05, 0.02),
    )
    applied = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=result["fwdtransforms"])
    diff = np.abs(applied.numpy() - result["warpedmovout"].numpy())
    assert diff.mean() < 1e-3
    assert diff.max() < 0.05


# --- optimizer='reg_adam' ----------------------------------------------------


def test_syn_registration_rejects_unknown_optimizer():
    fixed, moving = _ants_pair_2d()
    with pytest.raises(ValueError, match="optimizer"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", optimizer="nonsense")


def test_syn_only_reduces_loss_with_reg_adam_optimizer():
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(15, 10), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
        regularizer="dsti", optimizer="reg_adam",
    )
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert len(result["level_loss_history"]) == 2
    assert result["provenance"]["optimizer"] == "reg_adam"
    assert torch.isfinite(torch.tensor(result["loss_history"])).all()


def test_syn_registration_default_optimizer_is_gradient_descent():
    fixed, moving = _ants_pair_2d()
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly", levels=(1,), reg_iterations=(5,), syn_metric="mse",
    )
    assert result["provenance"]["optimizer"] == "gradient_descent"


def test_reg_adam_optimizer_changes_the_update_trajectory_vs_gradient_descent():
    # Same everything else (seeded identically via the same deterministic
    # synthetic pair and fixed hyperparameters) -- reg_adam's per-voxel Adam
    # moments should make its loss trajectory diverge from the plain
    # CFL-bounded gradient-descent trajectory after the first iteration
    # (where the two are identical, since Adam's bias-corrected quotient on
    # iteration 1 with zero-initialized moments is proportional to
    # sign(grad), not equal to grad -- so they can differ immediately).
    fixed, moving = _ants_pair_2d(ramp=0.3)
    kwargs = dict(
        type_of_transform="SyNOnly", levels=(1,), reg_iterations=(8,), syn_metric="mse",
        grad_step=0.4, flow_sigma=2.0, regularizer="dsti",
    )
    gd = syn_registration(fixed, moving, optimizer="gradient_descent", **kwargs)
    ra = syn_registration(fixed, moving, optimizer="reg_adam", **kwargs)
    gd_hist = torch.tensor(gd["loss_history"])
    ra_hist = torch.tensor(ra["loss_history"])
    assert torch.isfinite(gd_hist).all() and torch.isfinite(ra_hist).all()
    assert not torch.allclose(gd_hist, ra_hist)


def test_reg_adam_optimizer_resets_adam_moments_at_each_pyramid_level():
    # Purely a "does it run without shape errors across a level change"
    # check -- _fit_syn_level zero-initializes exp_avg/exp_avg_sq fresh on
    # every call (i.e. every level), so the per-level warp resolution change
    # must never hit a stale, wrongly-shaped Adam moment buffer.
    fixed, moving = _ants_pair_2d(ramp=0.3)
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(6, 6), syn_metric="mse", grad_step=0.4, flow_sigma=2.0,
        regularizer="sobolev", optimizer="reg_adam",
    )
    assert len(result["level_loss_history"]) == 2
    assert torch.isfinite(torch.tensor(result["loss_history"])).all()


def test_reg_adam_optimizer_works_in_3d():
    fixed, moving = _ants_pair_3d()
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        levels=(1,), reg_iterations=(5,), syn_metric="mse", grad_step=0.3, flow_sigma=1.5,
        regularizer="dsti", optimizer="reg_adam",
    )
    assert torch.isfinite(torch.tensor(result["loss_history"])).all()
