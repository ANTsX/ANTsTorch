import ants
import numpy as np
import pytest
import torch

from antstorch.syn.bridge import (
    ants_image_metadata,
    ants_image_to_tensor,
    apply_bspline_smoothing_operator,
    displacement_xyz_to_ants_image,
    displacement_zyx_to_ants_image,
    flip_affine_xyz_to_zyx,
    image_domain_from_metadata,
    metadata_tensors,
    metadata_tensors_from_dict,
    tensor_to_ants_image,
)


def _ramp_image_2d(size=(9, 7)):
    yy, xx = np.mgrid[0 : size[0], 0 : size[1]].astype(np.float32)
    array = (xx + 2.0 * yy) + 1.0  # strictly positive, so percentile-clip has a well-defined foreground.
    return ants.from_numpy(np.ascontiguousarray(array.T))


def _ramp_image_3d(size=(6, 5, 4)):
    zz, yy, xx = np.mgrid[0 : size[0], 0 : size[1], 0 : size[2]].astype(np.float32)
    array = (xx + 2.0 * yy + 3.0 * zz) + 1.0
    return ants.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0)))


def test_ants_image_metadata_matches_shape_and_reverses_torch_shape():
    image = _ramp_image_2d((9, 7))
    meta = ants_image_metadata(image)
    assert meta["dimension"] == 2
    assert meta["shape"] == tuple(int(v) for v in image.shape)
    assert meta["torch_shape"] == tuple(reversed(meta["shape"]))
    assert meta["spacing"] == (1.0, 1.0)
    assert meta["origin"] == (0.0, 0.0)


def test_ants_image_to_tensor_shape_and_roundtrip_without_normalization_2d():
    image = _ramp_image_2d((9, 7))
    tensor = ants_image_to_tensor(image, normalize=False)
    assert tensor.shape == (1, 1) + tuple(reversed(image.shape))
    recovered = tensor_to_ants_image(tensor, image)
    np.testing.assert_allclose(recovered.numpy(), image.numpy(), atol=1e-5)


def test_ants_image_to_tensor_shape_and_roundtrip_without_normalization_3d():
    image = _ramp_image_3d((6, 5, 4))
    tensor = ants_image_to_tensor(image, normalize=False)
    assert tensor.shape == (1, 1) + tuple(reversed(image.shape))
    recovered = tensor_to_ants_image(tensor, image)
    np.testing.assert_allclose(recovered.numpy(), image.numpy(), atol=1e-5)


def test_ants_image_to_tensor_normalization_is_in_unit_range():
    image = _ramp_image_2d((9, 7))
    tensor = ants_image_to_tensor(image, normalize=True)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0 + 1e-6


def test_displacement_zyx_roundtrip_matches_known_shift_2d():
    reference = _ramp_image_2d((9, 7))
    dim = 2
    # A constant physical displacement of (dx, dy) = (1.5, -0.5), in the
    # (z, y, x)-style component order antstorch.syn.core uses internally.
    field = torch.zeros(1, 7, 9, dim)
    field[..., 0] = -0.5  # y-component
    field[..., 1] = 1.5  # x-component
    ants_field = displacement_zyx_to_ants_image(field, reference)
    assert ants_field.components == dim
    array = ants_field.numpy()  # ITK order, ITK (x, y) component order
    np.testing.assert_allclose(array[..., 0], 1.5, atol=1e-6)
    np.testing.assert_allclose(array[..., 1], -0.5, atol=1e-6)


def test_displacement_xyz_roundtrip_matches_known_shift_2d():
    reference = _ramp_image_2d((9, 7))
    dim = 2
    field = torch.zeros(1, dim, 7, 9)
    field[:, 0] = 1.5  # x-component (channel-first, ITK order)
    field[:, 1] = -0.5  # y-component
    ants_field = displacement_xyz_to_ants_image(field, reference)
    array = ants_field.numpy()
    np.testing.assert_allclose(array[..., 0], 1.5, atol=1e-6)
    np.testing.assert_allclose(array[..., 1], -0.5, atol=1e-6)


def test_metadata_tensors_matches_metadata_tensors_from_dict():
    image = _ramp_image_2d((9, 7))
    direct = metadata_tensors(image, torch.device("cpu"), torch.float32)
    indirect = metadata_tensors_from_dict(ants_image_metadata(image), torch.device("cpu"), torch.float32)
    for key in direct:
        torch.testing.assert_close(direct[key], indirect[key])


def test_flip_affine_xyz_to_zyx_is_self_inverse():
    torch.manual_seed(0)
    matrix = torch.randn(3, 3)
    translation = torch.randn(3)
    once_m, once_t = flip_affine_xyz_to_zyx(matrix, translation)
    twice_m, twice_t = flip_affine_xyz_to_zyx(once_m, once_t)
    torch.testing.assert_close(twice_m, matrix)
    torch.testing.assert_close(twice_t, translation)


def test_flip_affine_xyz_to_zyx_reverses_components_for_diagonal_case():
    # A pure per-axis scale (diagonal matrix) commutes with axis reversal on
    # the diagonal, but translation components must swap order.
    matrix = torch.diag(torch.tensor([2.0, 3.0, 4.0]))  # sx, sy, sz (xyz order)
    translation = torch.tensor([1.0, 2.0, 3.0])  # tx, ty, tz
    flipped_m, flipped_t = flip_affine_xyz_to_zyx(matrix, translation)
    torch.testing.assert_close(flipped_m, torch.diag(torch.tensor([4.0, 3.0, 2.0])))
    torch.testing.assert_close(flipped_t, torch.tensor([3.0, 2.0, 1.0]))


# --- B-spline smoothing operator (ANTs/ITK BSplineSyN regularizer) ---------


def test_image_domain_from_metadata_matches_ants_image_metadata():
    image = _ramp_image_2d((9, 7))
    meta = ants_image_metadata(image)
    domain = image_domain_from_metadata(meta)
    assert domain.size == meta["shape"]
    assert domain.spacing == meta["spacing"]
    assert domain.origin == meta["origin"]
    assert domain.torch_size == meta["torch_shape"]


def test_apply_bspline_smoothing_operator_preserves_shape_and_is_finite():
    image = _ramp_image_2d((17, 15))
    domain = image_domain_from_metadata(ants_image_metadata(image))
    torch.manual_seed(0)
    field = torch.randn(1, *domain.torch_size, 2)
    smoothed = apply_bspline_smoothing_operator(field, domain, mesh_size=2)
    assert smoothed.shape == field.shape
    assert torch.isfinite(smoothed).all()


def test_apply_bspline_smoothing_operator_reduces_high_frequency_roughness():
    # A coarse control-point mesh (mesh_size=1) fit to pure per-voxel noise
    # should behave as a strong low-pass filter: the smoothed field's
    # voxel-to-voxel differences should be much smaller than the raw noisy
    # field's, the same qualitative check a Gaussian/Sobolev/DST-I
    # regularizer would be expected to pass.
    image = _ramp_image_2d((25, 23))
    domain = image_domain_from_metadata(ants_image_metadata(image))
    torch.manual_seed(0)
    field = torch.randn(1, *domain.torch_size, 2)
    smoothed = apply_bspline_smoothing_operator(field, domain, mesh_size=1)

    def roughness(f):
        return (f[:, 1:, :, :] - f[:, :-1, :, :]).abs().mean() + (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()

    assert roughness(smoothed) < 0.25 * roughness(field)


def test_apply_bspline_smoothing_operator_enforces_stationary_boundary_by_default():
    image = _ramp_image_2d((17, 15))
    domain = image_domain_from_metadata(ants_image_metadata(image))
    field = torch.full((1, *domain.torch_size, 2), 5.0)
    smoothed = apply_bspline_smoothing_operator(field, domain, mesh_size=2, enforce_stationary_boundary=True)
    boundary = torch.cat(
        [smoothed[:, 0, :, :].reshape(-1), smoothed[:, -1, :, :].reshape(-1),
         smoothed[:, :, 0, :].reshape(-1), smoothed[:, :, -1, :].reshape(-1)]
    )
    assert boundary.abs().max() < 0.5  # far below the constant 5.0 field value


def test_apply_bspline_smoothing_operator_rejects_non_positive_mesh_size():
    image = _ramp_image_2d((9, 7))
    domain = image_domain_from_metadata(ants_image_metadata(image))
    field = torch.zeros(1, *domain.torch_size, 2)
    with pytest.raises(ValueError, match="mesh_size"):
        apply_bspline_smoothing_operator(field, domain, mesh_size=0)


def test_apply_bspline_smoothing_operator_is_invariant_to_chunk_size():
    # Regression test for a real CUDA OOM: at full native resolution with a
    # fine update-field mesh, antstorch.syn.syn_registration(regularizer=
    # "bspline") ("bspline_syn" in antstorch.benchmark) crashed with "CUDA
    # out of memory. Tried to allocate 7.50 GiB." at the finest pyramid
    # level -- traced to fit_bspline_displacement_field (called from here)
    # materializing one (4**D, N) index/weight tensor for the whole dense
    # grid at once. The fix chunks that fit over samples (see
    # _bspline_fit_dense_grid_chunked); this only bounds peak memory, so the
    # result must be identical (up to floating-point summation order)
    # regardless of chunk_size -- verified here across a wide range,
    # including a chunk_size far larger than the grid (effectively the old
    # unchunked behavior) down to one much smaller than a single dense-image
    # row.
    image = _ramp_image_2d((23, 19))
    domain = image_domain_from_metadata(ants_image_metadata(image))
    torch.manual_seed(0)
    field = torch.randn(1, *domain.torch_size, 2, dtype=torch.double)
    reference = apply_bspline_smoothing_operator(field, domain, mesh_size=3, chunk_size=1_000_000)
    for chunk_size in (17, 97, 4096):
        smoothed = apply_bspline_smoothing_operator(field, domain, mesh_size=3, chunk_size=chunk_size)
        torch.testing.assert_close(smoothed, reference, rtol=1e-10, atol=1e-10)
