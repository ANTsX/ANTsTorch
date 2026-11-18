"""Memory-motivated refactors of SyN (issue #91): behaviour must be unchanged.

Covers the reference releases in ``_fit_syn_level``/``syn_registration``,
``prepare_mid_images_and_gradients_torch(compute_gradients=False)``, the
ring-buffer Anderson inverse, the separable ``_smooth_image`` and
``syn_registration(affine_smoothing_sigmas_units=...)``.
"""

import gc
import weakref

import ants
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import antstorch.syn.syn as syn_module
from antstorch.bspline_flows import ImageDomain
from antstorch.bspline_flows.bspline_svf_registration import _smooth_image
from antstorch.syn import syn_registration
from antstorch.syn.bridge import ants_image_metadata, metadata_tensors_from_dict
from antstorch.syn.core import update_inverse_field_nd_anderson
from antstorch.syn.core.grid import prepare_mid_images_and_gradients_torch
from antstorch.syn.syn import _fit_syn_level, _physical_grid


def _blob_image(shape=(14, 16, 12), shift=0.0, spacing=(1.0, 1.0, 1.0)):
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    array = np.exp(-(((x - shape[2] / 2 - shift) / 3.0) ** 2 + ((y - shape[1] / 2) / 4.0) ** 2
                     + ((z - shape[0] / 2 + shift / 2) / 3.5) ** 2))
    return ants.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0)), spacing=spacing)


def _smooth_field(shape, seed, amplitude):
    gen = torch.Generator().manual_seed(seed)
    coarse = torch.randn((1, 3) + tuple(max(2, s // 3) for s in shape), generator=gen, dtype=torch.float64)
    fine = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    return (amplitude * fine).movedim(1, -1).contiguous()


def _level_inputs(dtype=torch.float64):
    fixed, moving = _blob_image(), _blob_image(shift=1.5)
    fixed_meta, moving_meta = ants_image_metadata(fixed), ants_image_metadata(moving)
    I = torch.from_numpy(fixed.numpy().transpose(2, 1, 0).copy()).to(dtype)[None, None]
    J = torch.from_numpy(moving.numpy().transpose(2, 1, 0).copy()).to(dtype)[None, None]
    return I, J, fixed_meta, moving_meta


_LEVEL_KWARGS = dict(
    similarity="cc2", window_size=5, num_bins=32, flow_sigma=1.0, total_sigma=0.0, regularizer="gaussian",
    update_field_mesh_size_at_base_level=0, total_field_mesh_size_at_base_level=0,
    bspline_enforce_stationary_boundary=True, grad_step=0.25, shrink_factor=1, antisymmetric=True,
    inverse_method="anderson", in_loop_inverse_steps=3, verbose=False, gaussian_sigma_mode="voxel",
)


# --- _fit_syn_level / syn_registration reference releases -------------------


def test_fit_syn_level_consumes_the_warp_list_and_frees_level_start_fields(monkeypatch):
    I, J, fixed_meta, moving_meta = _level_inputs()
    field_shape = (1,) + tuple(fixed_meta["torch_shape"]) + (3,)
    warps = [torch.zeros(field_shape, dtype=torch.float64) for _ in range(4)]
    start_refs = [weakref.ref(field) for field in warps[:2]]
    leaf_refs, alive_previous_leaves, alive_start, alive_previous_steps = [], [], [], []
    step_refs = []
    original_prepare = syn_module.prepare_mid_images_and_gradients_torch
    original_update = syn_module._eulerian_update

    def update_spy(warp, delta, *args):
        step_refs.append(weakref.ref(delta))
        return original_update(warp, delta, *args)

    def spy(left, right, *args, **kwargs):
        # Called once per iteration with that iteration's leaves; by then the
        # previous iteration's leaves, gradients and loss graph must be gone.
        gc.collect()
        alive_previous_leaves.append([ref() is not None for ref in leaf_refs])
        alive_start.append([ref() is not None for ref in start_refs])
        alive_previous_steps.append([ref() is not None for ref in step_refs])
        step_refs.clear()
        leaf_refs[:] = [weakref.ref(left), weakref.ref(right)]
        assert kwargs.get("compute_gradients") is False  # the objective never uses them
        return original_prepare(left, right, *args, **kwargs)

    monkeypatch.setattr(syn_module, "prepare_mid_images_and_gradients_torch", spy)
    monkeypatch.setattr(syn_module, "_eulerian_update", update_spy)
    M, t = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    outputs = _fit_syn_level(
        I, J, fixed_meta, moving_meta, warps, M, t, iterations=4, level_index=0, num_levels=1,
        inverse_schedule="per_iteration", **_LEVEL_KWARGS,
    )
    assert warps == []  # the caller's references were dropped
    assert len(alive_previous_leaves) == 4
    assert all(alive == [False, False] for alive in alive_previous_leaves[1:])
    # The previous iteration's steps (and hence gradients) are released too.
    assert all(alive == [False, False] for alive in alive_previous_steps[1:])
    # From iteration 1 on, nothing (caller, leaves, graph) keeps the level-start fields.
    assert all(alive == [False, False] for alive in alive_start[1:])
    assert len(outputs) == 5 and all(isinstance(field, torch.Tensor) for field in outputs[:4])


def test_fit_syn_level_end_of_level_skips_inverses_before_the_last_level():
    I, J, fixed_meta, moving_meta = _level_inputs()
    field_shape = (1,) + tuple(fixed_meta["torch_shape"]) + (3,)
    M, t = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    common = dict(iterations=2, inverse_schedule="end_of_level", **_LEVEL_KWARGS)
    coarse = _fit_syn_level(I, J, fixed_meta, moving_meta,
                            [torch.zeros(field_shape, dtype=torch.float64)] * 2 + [None, None], M, t,
                            level_index=0, num_levels=2, **common)
    assert coarse[2] is None and coarse[3] is None
    final = _fit_syn_level(I, J, fixed_meta, moving_meta, [coarse[0], coarse[1], None, None], M, t,
                           level_index=1, num_levels=2, **common)
    assert final[2] is not None and final[3] is not None
    assert final[2].shape == final[0].shape


@pytest.mark.parametrize("extra", [
    dict(),
    dict(inverse_schedule="per_iteration"),
    dict(inverse_schedule="per_iteration", syntx_parity=True),
    dict(syntx_parity=True),
])
def test_syn_registration_runs_with_both_inverse_schedules(extra):
    fixed, moving = _blob_image(), _blob_image(shift=1.5)
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    result = syn_registration(fixed, moving, type_of_transform="SyN", device="cpu", dtype=torch.float64,
                              initial_affine=(eye, zero), reg_iterations=(3, 2), levels=(2, 1), **extra)
    inverse = ants.image_read(result["invtransforms"][-1]).numpy()
    forward = ants.image_read(result["fwdtransforms"][0]).numpy()
    assert np.isfinite(forward).all() and np.isfinite(inverse).all()
    assert np.abs(forward).max() > 0


# --- prepare_mid_images_and_gradients_torch(compute_gradients=False) --------


def test_prepare_mid_images_without_gradients_matches_images_and_mask():
    I, J, fixed_meta, moving_meta = _level_inputs()
    fm = metadata_tensors_from_dict(fixed_meta, "cpu", torch.float64)
    mm = metadata_tensors_from_dict(moving_meta, "cpu", torch.float64)
    X = _physical_grid(fixed_meta, "cpu", torch.float64)
    w1 = _smooth_field(tuple(fixed_meta["torch_shape"]), 1, 0.6).requires_grad_(True)
    w2 = _smooth_field(tuple(fixed_meta["torch_shape"]), 2, 0.6).requires_grad_(True)
    M = torch.tensor([[1.02, 0.01, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.0]], dtype=torch.float64)
    t = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)

    def run(compute_gradients):
        return prepare_mid_images_and_gradients_torch(
            w1, w2, None, None, I, J, X,
            fm["shape_t"], fm["spacing_t"], fm["origin_t"], fm["direction_t"],
            mm["shape_t"], mm["spacing_t"], mm["origin_t"], mm["direction_t"],
            fixed_meta["spacing"], moving_meta["spacing"], M, t, None,
            compute_gradients=compute_gradients,
        )

    full, light = run(True), run(False)
    assert light[2] is None and light[3] is None
    assert full[2] is not None and full[2].shape == tuple(full[0].shape[:1]) + tuple(full[0].shape[2:]) + (3,)
    for index in (0, 1, 4):
        assert torch.equal(full[index], light[index])
    # Same gradient w.r.t. the half-warps (the image gradients never enter it).
    g_full = torch.autograd.grad((full[0] * full[1]).sum(), (w1, w2))
    g_light = torch.autograd.grad((light[0] * light[1]).sum(), (w1, w2))
    for a, b in zip(g_full, g_light):
        assert torch.equal(a, b)


# --- Anderson inverse with contiguous ring-buffer history --------------------

# Statistics of the list-based implementation at commit d4f15ad (float64,
# CPU) on the same field: the ring buffer reorders storage, not arithmetic.
_ANDERSON_GOLDEN = {
    (True, 1): (-89.81999707335363, 257.357588412169, -0.28652247780238876),
    (True, 3): (-91.66151355884601, 260.47161437386217, -0.2972190223616291),
    (True, 12): (-92.72016592220258, 262.1394262952167, -0.3029467908546647),
    (False, 1): (-4.182388069467968, 12.782067913265521, -0.013056976674428683),
    (False, 3): (-4.173233483308354, 12.824457639391746, -0.013004364671500856),
    (False, 12): (-4.170278132549515, 12.839712009678724, -0.01298194956220624),
}


@pytest.mark.parametrize("physical,steps", sorted(_ANDERSON_GOLDEN))
def test_anderson_ring_buffer_matches_list_based_reference(physical, steps):
    W = _smooth_field((9, 10, 8), 3, 0.8)
    kwargs = dict(spacing=(1.0, 1.2, 0.9), origin=(0.5, -1.0, 2.0)) if physical else {}
    if not physical:
        W = W * 0.05
    out = update_inverse_field_nd_anderson(W, None, steps=steps, m=5, max_error_threshold=0.0,
                                           mean_error_threshold=0.0, **kwargs)
    total, absolute, voxel = _ANDERSON_GOLDEN[(physical, steps)]
    assert float(out.sum()) == pytest.approx(total, rel=1e-12, abs=1e-12)
    assert float(out.abs().sum()) == pytest.approx(absolute, rel=1e-12)
    assert float(out[0, 4, 5, 3, 1]) == pytest.approx(voxel, rel=1e-10, abs=1e-14)


# --- separable _smooth_image and sigma units ---------------------------------


def _dense_reference(image, sigma_torch):
    """The pre-#91 dense N-D kernel (outer product of 1-D Gaussians)."""
    dim = image.ndim - 2
    axes, radii = [], []
    for axis_sigma in sigma_torch:
        radius = max(1, int(3.0 * axis_sigma + 0.5))
        radii.append(radius)
        coordinate = torch.arange(-radius, radius + 1, dtype=image.dtype)
        kernel_1d = torch.exp(-0.5 * (coordinate / axis_sigma).square())
        axes.append(kernel_1d / kernel_1d.sum())
    kernel = axes[0]
    for axis_kernel in axes[1:]:
        kernel = kernel.unsqueeze(-1) * axis_kernel.reshape((1,) * kernel.ndim + (-1,))
    kernel = kernel.reshape((1, 1) + kernel.shape).expand((image.shape[1], 1) + kernel.shape)
    padding = tuple(value for radius in reversed(radii) for value in (radius, radius))
    conv = F.conv2d if dim == 2 else F.conv3d
    return conv(F.pad(image, padding, mode="replicate"), kernel, groups=image.shape[1])


@pytest.mark.parametrize("dim", [2, 3])
def test_smooth_image_separable_equals_dense_kernel(dim):
    gen = torch.Generator().manual_seed(0)
    shape = (11, 13, 9)[:dim]
    image = torch.randn((1, 2) + shape, generator=gen, dtype=torch.float64)
    spacing = (0.8, 1.3, 1.1)[:dim]
    domain = ImageDomain(tuple(reversed(shape)), spacing)
    sigma = 1.4
    expected = _dense_reference(image, tuple(reversed([sigma / s for s in spacing])))
    torch.testing.assert_close(_smooth_image(image, domain, sigma), expected, rtol=0, atol=1e-12)
    expected_vox = _dense_reference(image, (sigma,) * dim)
    torch.testing.assert_close(_smooth_image(image, domain, sigma, "vox"), expected_vox, rtol=0, atol=1e-12)


def test_smooth_image_rejects_unknown_units():
    domain = ImageDomain((5, 5), (1.0, 1.0))
    with pytest.raises(ValueError, match="units"):
        _smooth_image(torch.zeros(1, 1, 5, 5), domain, 1.0, "inches")


def _affine_of(fixed, moving, **kwargs):
    result = syn_registration(fixed, moving, type_of_transform="Affine", device="cpu",
                              affine_iterations=(4, 3), affine_shrink_factors=(2, 1),
                              affine_learning_rate=(0.05, 0.05), affine_multi_start=False, **kwargs)
    return result


def test_affine_smoothing_sigmas_units_default_is_vox_and_is_recorded():
    fixed, moving = _blob_image(spacing=(2.0, 2.0, 2.0)), _blob_image(shift=1.0, spacing=(2.0, 2.0, 2.0))
    default = _affine_of(fixed, moving, affine_smoothing_sigmas=(1.0, 0.0))
    vox = _affine_of(fixed, moving, affine_smoothing_sigmas=(1.0, 0.0), affine_smoothing_sigmas_units="vox")
    mm = _affine_of(fixed, moving, affine_smoothing_sigmas=(2.0, 0.0), affine_smoothing_sigmas_units="mm")
    mm_same_number = _affine_of(fixed, moving, affine_smoothing_sigmas=(1.0, 0.0), affine_smoothing_sigmas_units="mm")
    assert default["provenance"]["affine_smoothing_sigmas_units"] == "vox"
    assert mm["provenance"]["affine_smoothing_sigmas_units"] == "mm"
    torch.testing.assert_close(default["affine_matrix"], vox["affine_matrix"], rtol=0, atol=0)
    # 1 voxel at 2 mm spacing is 2 mm (up to the 1-D kernels' floating point).
    torch.testing.assert_close(vox["affine_matrix"], mm["affine_matrix"], rtol=0, atol=1e-5)
    torch.testing.assert_close(vox["affine_translation"], mm["affine_translation"], rtol=0, atol=1e-5)
    assert not torch.allclose(vox["affine_matrix"], mm_same_number["affine_matrix"], rtol=0, atol=1e-7)


def test_affine_smoothing_sigmas_units_rejects_unknown_value():
    fixed, moving = _blob_image(), _blob_image(shift=1.0)
    with pytest.raises(ValueError, match="affine_smoothing_sigmas_units"):
        syn_registration(fixed, moving, type_of_transform="Affine", affine_smoothing_sigmas_units="pixels")
