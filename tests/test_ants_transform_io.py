"""Tests for antstorch.ants_transform_io, focused on the newly added
read_affine_transform() -- the exact inverse of write_affine_transform(),
needed so an on-disk canonical affine (e.g. syntx's shared per-pair
robust_affine() .mat file) can be fed to antstorch.syn.syn_registration()'s
and antstorch.bspline_flows.bspline_svf_registration.bspline_svf_registration()'s
initial_affine parameter.
"""
import numpy as np
import pytest

import ants

from antstorch.ants_transform_io import read_affine_transform, write_affine_transform


@pytest.mark.parametrize("dim", [2, 3])
def test_read_affine_transform_round_trips_write_affine_transform(dim, tmp_path):
    rng = np.random.RandomState(0)
    matrix = np.eye(dim) + 0.05 * rng.randn(dim, dim)
    translation = rng.randn(dim)
    path = str(tmp_path / "0GenericAffine.mat")

    write_affine_transform(matrix, translation, dim, path)
    read_matrix, read_translation = read_affine_transform(path, dim)

    assert read_matrix.shape == (dim, dim)
    assert read_translation.shape == (dim,)
    np.testing.assert_allclose(read_matrix, matrix, atol=1e-6)
    np.testing.assert_allclose(read_translation, translation, atol=1e-6)


def test_read_affine_transform_accepts_torch_tensor_inputs_to_write(tmp_path):
    import torch

    matrix = torch.eye(3) + 0.1 * torch.randn(3, 3)
    translation = torch.randn(3)
    path = str(tmp_path / "0GenericAffine.mat")

    write_affine_transform(matrix, translation, 3, path)
    read_matrix, read_translation = read_affine_transform(path, 3)

    np.testing.assert_allclose(read_matrix, matrix.numpy(), atol=1e-6)
    np.testing.assert_allclose(read_translation, translation.numpy(), atol=1e-6)


def test_read_affine_transform_rejects_dimension_mismatch(tmp_path):
    path = str(tmp_path / "0GenericAffine.mat")
    write_affine_transform(np.eye(3), np.zeros(3), 3, path)
    with pytest.raises(ValueError, match="dimension"):
        read_affine_transform(path, 2)


def test_read_affine_transform_rejects_non_affine_transform_type(tmp_path):
    path = str(tmp_path / "0Warp_as_transform.mat")
    # A transform type other than AffineTransform should be rejected with a
    # clear error rather than silently misreading its parameter layout.
    displacement_field_transform = ants.new_ants_transform(
        precision="float", dimension=3, transform_type="Euler3DTransform"
    )
    ants.write_transform(displacement_field_transform, path)
    with pytest.raises(ValueError, match="AffineTransform"):
        read_affine_transform(path, 3)
