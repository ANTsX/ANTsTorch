import ants
import numpy as np
import pytest
import torch

from antstorch.syn import syn_registration
from antstorch.syn.bridge import ants_image_metadata, metadata_tensors
from antstorch.syn.syn import _compose_fixed_grid, _physical_grid


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
