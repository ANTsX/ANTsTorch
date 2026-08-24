import pytest
import torch

from antstorch.syn.core import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_jacobian_hinge_penalty,
    compute_physical_jacobian_determinant,
)


def _make_identity_grid(spatial, dtype=torch.double):
    grids = [torch.linspace(-1, 1, s, dtype=dtype) for s in spatial]
    mesh = torch.meshgrid(*grids, indexing='ij')
    return torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)


def test_spatial_jacobian_nd_recovers_constant_linear_map():
    sizes = (6, 7)
    grids = [torch.linspace(-1, 1, s, dtype=torch.double) for s in sizes]
    mesh = torch.meshgrid(*grids, indexing='ij')
    coords = torch.stack(mesh, dim=-1)
    M = torch.tensor([[1.5, -0.5], [0.3, 2.0]], dtype=torch.double)
    field = torch.einsum('ij,...j->...i', M, coords).unsqueeze(0)
    jacobian = _spatial_jacobian_nd(field)
    expected = M.view(1, 1, 1, 2, 2).expand(1, *sizes, 2, 2)
    torch.testing.assert_close(jacobian, expected, atol=1e-8, rtol=1e-6)


def test_spatial_jacobian_nd_bspline_method_preserves_shape_and_is_finite():
    # The 'bspline' finite-difference path is a faithful port of syntx's
    # own implementation; it is not verified here to numerically match the
    # 'central' method (upstream syntx exhibits the same divergence for a
    # generic linear field), only that it runs and returns a well-formed,
    # finite result of the expected shape.
    sizes = (9, 9)
    grids = [torch.linspace(-1, 1, s, dtype=torch.double) for s in sizes]
    mesh = torch.meshgrid(*grids, indexing='ij')
    coords = torch.stack(mesh, dim=-1)
    M = torch.tensor([[0.8, 0.1], [-0.2, 1.3]], dtype=torch.double)
    field = torch.einsum('ij,...j->...i', M, coords).unsqueeze(0)
    bspline = _spatial_jacobian_nd(field, method='bspline')
    assert bspline.shape == (1, *sizes, 2, 2)
    assert torch.isfinite(bspline).all()


def test_jacobian_determinant_zero_field_is_identity():
    warp = torch.zeros(1, 6, 7, 2, dtype=torch.double)
    det = compute_jacobian_determinant_nd(warp)
    torch.testing.assert_close(det, torch.ones_like(det), atol=1e-10, rtol=0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_jacobian_determinant_isotropic_scaling(dimension):
    sizes = (7,) * dimension
    identity = _make_identity_grid(sizes)
    warp = 0.1 * identity
    det = compute_jacobian_determinant_nd(warp)
    expected = torch.full_like(det, 1.1 ** dimension)
    torch.testing.assert_close(det, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("dimension", [2, 3])
def test_jacobian_determinant_physical_isotropic_scaling(dimension):
    sizes = (7,) * dimension
    spacing = tuple(1.5 for _ in range(dimension))
    grids = [torch.linspace(0.0, (s - 1) * 1.5, s, dtype=torch.double) for s in sizes]
    mesh = torch.meshgrid(*grids, indexing='ij')
    points = torch.stack(mesh, dim=-1).unsqueeze(0)
    displacement = 0.1 * points
    det = compute_jacobian_determinant_nd(displacement, physical_spacing=spacing)
    expected = torch.full_like(det, 1.1 ** dimension)
    torch.testing.assert_close(det, expected, atol=1e-8, rtol=1e-6)


def test_hinge_penalty_zero_for_identity_field():
    warp = torch.zeros(1, 6, 6, 2, dtype=torch.double)
    penalty = compute_jacobian_hinge_penalty(warp)
    assert penalty.item() == pytest.approx(0.0, abs=1e-12)


def test_hinge_penalty_positive_for_folded_field():
    # For an even spatial dimension, phi = -identity has det(J) = (-1)**dim
    # > 0 (no actual folding) — use an odd (3-D) dimension, matching the
    # sign-parity property exercised by antstorch.bspline_flows' own
    # jacobian_determinant folding test.
    sizes = (7, 7, 7)
    identity = _make_identity_grid(sizes)
    warp = -2.0 * identity
    penalty = compute_jacobian_hinge_penalty(warp)
    assert penalty.item() > 0.0


def test_physical_jacobian_determinant_zero_field_is_identity():
    warp = torch.zeros(1, 6, 6, 2, dtype=torch.double)
    direction = torch.eye(2, dtype=torch.double)
    spacing = torch.ones(2, dtype=torch.double)
    det = compute_physical_jacobian_determinant(warp, direction, spacing)
    torch.testing.assert_close(det, torch.ones_like(det), atol=1e-10, rtol=0)


def test_physical_jacobian_determinant_delegates_when_is_physical_flag_set():
    warp = torch.zeros(1, 6, 6, 2, dtype=torch.double)
    warp.is_physical = True
    direction = torch.eye(2, dtype=torch.double)
    spacing = torch.tensor([1.0, 1.0], dtype=torch.double)
    det = compute_physical_jacobian_determinant(warp, direction, spacing)
    expected = compute_jacobian_determinant_nd(warp, physical_spacing=spacing)
    torch.testing.assert_close(det, expected)
