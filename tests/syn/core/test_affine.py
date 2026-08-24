import math
import tempfile
import os

import ants
import numpy as np
import pytest
import torch

from antstorch.syn.core import (
    get_rotation_matrix,
    HierarchicalAffine,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    grid_to_physical_affine,
    parse_ants_affine,
    compute_initial_grid,
)


def test_get_rotation_matrix_2d_quarter_turn():
    omega = torch.tensor([math.pi / 2], dtype=torch.double)
    R = get_rotation_matrix(omega, dim=2)
    expected = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(R, expected, atol=1e-10, rtol=0)


def test_get_rotation_matrix_is_orthogonal_and_proper_3d():
    omega = torch.tensor([0.3, -0.6, 0.9], dtype=torch.double)
    R = get_rotation_matrix(omega, dim=3)
    torch.testing.assert_close(R @ R.t(), torch.eye(3, dtype=torch.double), atol=1e-10, rtol=0)
    assert torch.det(R).item() == pytest.approx(1.0, abs=1e-10)


def test_get_rotation_matrix_zero_omega_gradient_is_finite_3d():
    omega = torch.zeros(3, dtype=torch.double, requires_grad=True)
    R = get_rotation_matrix(omega, dim=3)
    R.sum().backward()
    assert torch.isfinite(omega.grad).all()


def test_get_rotation_matrix_rejects_invalid_dim():
    with pytest.raises(ValueError):
        get_rotation_matrix(torch.zeros(1), dim=4)


@pytest.mark.parametrize("transform_type", ["Translation", "Rigid", "Similarity", "Affine"])
def test_hierarchical_affine_zero_init_is_identity(transform_type):
    module = HierarchicalAffine(dim=3, transform_type=transform_type)
    T = module.get_matrix()
    torch.testing.assert_close(T, torch.eye(4), atol=1e-6, rtol=0)


def test_hierarchical_affine_translation_only_moves_translation_block():
    module = HierarchicalAffine(dim=3, transform_type='Translation')
    with torch.no_grad():
        module.translation.copy_(torch.tensor([1.0, 2.0, 3.0]))
    T = module.get_matrix()
    torch.testing.assert_close(T[:3, :3], torch.eye(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(T[:3, 3], torch.tensor([1.0, 2.0, 3.0]))


def test_hierarchical_affine_gradients_flow_for_affine_type():
    module = HierarchicalAffine(dim=3, transform_type='Affine')
    T = module.get_matrix()
    T.sum().backward()
    for p in module.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_hierarchical_affine_clamp_parameters_bounds():
    module = HierarchicalAffine(dim=3, transform_type='Affine')
    with torch.no_grad():
        module.scale.fill_(100.0)
        module.omega.fill_(10.0)
    module.clamp_parameters()
    assert module.scale.item() <= 20.0
    assert module.omega.abs().max().item() <= 3.14159265 + 1e-6


def _make_ants_image(shape, origin, spacing, direction):
    return ants.from_numpy(
        np.zeros(shape, dtype=np.float32), origin=origin, spacing=spacing, direction=np.array(direction)
    )


def test_physical_to_grid_affine_identity_for_matching_fixed_and_moving():
    # physical_to_grid_affine and grid_to_physical_affine each reference the
    # fixed/moving image shape with a different internal axis-order
    # convention (confirmed identical in the unmodified syntx source), so
    # they are not exact mutual inverses for a general (differently shaped)
    # fixed/moving pair. When fixed and moving share the same metadata,
    # however, the identity physical transform must round-trip exactly.
    image = _make_ants_image((6, 7), origin=(1.0, -2.0), spacing=(1.2, 0.8), direction=[[1.0, 0.0], [0.0, 1.0]])

    M_phys = np.eye(2)
    t_phys = np.zeros(2)

    T_grid = physical_to_grid_affine(M_phys, t_phys, image, image)
    np.testing.assert_allclose(T_grid, np.eye(3), atol=1e-6)

    M_back, t_back = grid_to_physical_affine(np.eye(3), image, image)
    np.testing.assert_allclose(M_back, M_phys, atol=1e-5)
    np.testing.assert_allclose(t_back, t_phys, atol=1e-5)


def test_grid_to_physical_affine_torch_matches_numpy_version_permuted():
    fixed = _make_ants_image((6, 7), origin=(1.0, -2.0), spacing=(1.2, 0.8), direction=[[1.0, 0.0], [0.0, 1.0]])
    moving = _make_ants_image((5, 6), origin=(0.3, 0.1), spacing=(0.9, 1.1), direction=[[0.0, -1.0], [1.0, 0.0]])

    rng = np.random.default_rng(0)
    T_grid_np = np.eye(3, dtype=np.float64)
    T_grid_np[:2, :2] = np.eye(2) + 0.05 * rng.standard_normal((2, 2))
    T_grid_np[:2, 2] = 0.1 * rng.standard_normal(2)

    M_phys_xyz, t_phys_xyz = grid_to_physical_affine(T_grid_np, fixed, moving)

    P = np.eye(2)[::-1]
    M_phys_zyx_expected = P @ M_phys_xyz @ P
    t_phys_zyx_expected = P @ t_phys_xyz

    # Unlike every other metadata argument, grid_to_physical_affine_torch
    # does not reverse fixed_shape/moving_shape internally, so (per its
    # docstring) callers must supply the PyTorch (reversed-ITK) shape here.
    T_grid_torch = torch.from_numpy(T_grid_np)
    M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
        T_grid_torch, tuple(reversed(fixed.shape)), fixed.spacing, fixed.origin, fixed.direction,
        tuple(reversed(moving.shape)), moving.spacing, moving.origin, moving.direction,
    )

    np.testing.assert_allclose(M_phys_zyx.numpy(), M_phys_zyx_expected, atol=1e-5)
    np.testing.assert_allclose(t_phys_zyx.numpy(), t_phys_zyx_expected, atol=1e-5)


def test_parse_ants_affine_empty_list_returns_none():
    M, t = parse_ants_affine([], dim=3)
    assert M is None
    assert t is None


def test_parse_ants_affine_recovers_translation_only_transform():
    tx = ants.new_ants_transform(transform_type='AffineTransform', dimension=3)
    tx.set_parameters(list(np.eye(3).flatten()) + [1.0, -2.0, 0.5])
    M, t = parse_ants_affine(tx, dim=3)
    torch.testing.assert_close(M, torch.eye(3, dtype=torch.float32), atol=1e-5, rtol=0)
    torch.testing.assert_close(t, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32), atol=1e-5, rtol=0)


def test_parse_ants_affine_composes_two_translations():
    tx1 = ants.new_ants_transform(transform_type='AffineTransform', dimension=2)
    tx1.set_parameters(list(np.eye(2).flatten()) + [1.0, 0.0])
    tx2 = ants.new_ants_transform(transform_type='AffineTransform', dimension=2)
    tx2.set_parameters(list(np.eye(2).flatten()) + [0.0, 2.0])
    M, t = parse_ants_affine([tx1, tx2], dim=2)
    torch.testing.assert_close(M, torch.eye(2, dtype=torch.float32), atol=1e-5, rtol=0)
    torch.testing.assert_close(t, torch.tensor([1.0, 2.0], dtype=torch.float32), atol=1e-5, rtol=0)


def test_compute_initial_grid_identity_transform_matches_direct_normalized_grid():
    fixed = ants.from_numpy(np.zeros((8, 8), dtype=np.float32))
    moving = ants.from_numpy(np.zeros((8, 8), dtype=np.float32))

    tx = ants.new_ants_transform(transform_type='AffineTransform', dimension=2)
    tx.set_parameters(list(np.eye(2).flatten()) + [0.0, 0.0])
    path = tempfile.mktemp(suffix='.mat')
    ants.write_transform(tx, path)
    try:
        grid = compute_initial_grid(fixed, moving, [path])
    finally:
        os.remove(path)

    assert grid.shape == (1, 8, 8, 2)
    # An identity transform should map every fixed-space voxel to itself in
    # moving space, i.e. the normalized grid should match the direct
    # normalized identity grid (within apply_transforms' interpolation
    # tolerance).
    grids_1d = [np.linspace(-1, 1, s) for s in (8, 8)]
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    identity = np.stack(list(reversed(mesh)), axis=-1)[None].astype(np.float32)
    np.testing.assert_allclose(grid, identity, atol=1e-4)
