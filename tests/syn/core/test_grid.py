import numpy as np
import pytest
import torch
import torch.nn.functional as F

from antstorch.syn.core import (
    grid_sample_bspline_torch,
    AnalyticalGridSample,
    grid_sample_nd,
    compose_grids,
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
)


def _identity_grid(shape, dtype=torch.double):
    grids = [torch.linspace(-1, 1, s, dtype=dtype) for s in shape]
    mesh = torch.meshgrid(*grids, indexing='ij')
    return torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)


def test_grid_sample_bspline_constant_image_partition_of_unity_2d():
    image = torch.full((1, 1, 8, 8), 3.0, dtype=torch.double)
    torch.manual_seed(0)
    grid = torch.rand(1, 5, 5, 2, dtype=torch.double) * 2 - 1
    sampled = grid_sample_bspline_torch(image, grid)
    torch.testing.assert_close(sampled, torch.full_like(sampled, 3.0), atol=1e-8, rtol=1e-6)


def test_grid_sample_bspline_constant_image_partition_of_unity_3d():
    image = torch.full((1, 1, 5, 5, 5), -2.0, dtype=torch.double)
    torch.manual_seed(1)
    grid = torch.rand(1, 3, 3, 3, 3, dtype=torch.double) * 2 - 1
    sampled = grid_sample_bspline_torch(image, grid)
    torch.testing.assert_close(sampled, torch.full_like(sampled, -2.0), atol=1e-8, rtol=1e-6)


def test_grid_sample_bspline_torch_output_shape():
    image = torch.randn(2, 3, 6, 5, dtype=torch.double)
    grid = torch.rand(2, 4, 4, 2, dtype=torch.double) * 2 - 1
    sampled = grid_sample_bspline_torch(image, grid)
    assert sampled.shape == (2, 3, 4, 4)


def test_grid_sample_bspline_torch_rejects_1d_image():
    image = torch.randn(1, 1, 8, dtype=torch.double)
    grid = torch.randn(1, 3, 1, dtype=torch.double)
    with pytest.raises(ValueError, match="2-D and 3-D"):
        grid_sample_bspline_torch(image, grid)


def test_grid_sample_nd_default_matches_functional_grid_sample():
    torch.manual_seed(2)
    image = torch.randn(1, 1, 6, 6, dtype=torch.double)
    grid = torch.rand(1, 4, 4, 2, dtype=torch.double) * 2 - 1
    out = grid_sample_nd(image, grid, use_analytical_gradients=False)
    expected = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    torch.testing.assert_close(out, expected)


def test_grid_sample_nd_nearest_alias():
    torch.manual_seed(3)
    image = torch.randn(1, 1, 6, 6, dtype=torch.double)
    grid = torch.rand(1, 4, 4, 2, dtype=torch.double) * 2 - 1
    out = grid_sample_nd(image, grid, interpolator='nearestNeighbor')
    expected = F.grid_sample(image, grid, mode='nearest', padding_mode='border', align_corners=True)
    torch.testing.assert_close(out, expected)


def test_grid_sample_nd_bspline_alias_matches_direct_call():
    image = torch.full((1, 1, 6, 6), 4.0, dtype=torch.double)
    torch.manual_seed(4)
    grid = torch.rand(1, 3, 3, 2, dtype=torch.double) * 2 - 1
    out = grid_sample_nd(image, grid, mode='bspline')
    torch.testing.assert_close(out, torch.full_like(out, 4.0), atol=1e-8, rtol=1e-6)


def test_grid_sample_nd_analytical_gradients_are_finite():
    torch.manual_seed(5)
    image = torch.randn(1, 1, 6, 6, dtype=torch.double)
    grid = (torch.rand(1, 4, 4, 2, dtype=torch.double) * 2 - 1).requires_grad_()
    out = grid_sample_nd(image, grid, use_analytical_gradients=True)
    out.sum().backward()
    assert grid.grad is not None
    assert torch.isfinite(grid.grad).all()


def test_analytical_grid_sample_matches_autograd_reference_for_smooth_image():
    # AnalyticalGridSample's backward pass approximates the source image's
    # spatial gradient with a simple central difference (see
    # _image_spatial_gradient) rather than differentiating the bilinear
    # interpolation exactly, so it is a fast approximation, not an exact
    # match to PyTorch's default autograd path through grid_sample — hence
    # no gradcheck here. On a smooth (low-frequency) image, where that
    # central-difference approximation is accurate, its gradient should
    # still closely track the exact autograd gradient.
    torch.manual_seed(0)
    n = 32
    xs = torch.linspace(0, 2 * torch.pi, n, dtype=torch.double)
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    image = (torch.sin(xx) * torch.cos(yy)).unsqueeze(0).unsqueeze(0)

    base = torch.linspace(-0.6, 0.6, 5, dtype=torch.double)
    gy, gx = torch.meshgrid(base + 0.017, base - 0.031, indexing='ij')
    grid_ref = torch.stack([gx, gy], dim=-1).unsqueeze(0).requires_grad_()
    grid_ana = grid_ref.detach().clone().requires_grad_()

    F.grid_sample(image, grid_ref, mode='bilinear', padding_mode='border', align_corners=True).sum().backward()
    AnalyticalGridSample.apply(image, grid_ana).sum().backward()

    relative_error = (grid_ref.grad - grid_ana.grad).norm() / grid_ref.grad.norm()
    assert relative_error.item() < 0.1


def test_compose_grids_with_identity_first_grid_returns_second():
    shape = (6, 7)
    identity = _identity_grid(shape)
    torch.manual_seed(7)
    grid2 = torch.rand(1, *shape, 2, dtype=torch.double) * 1.6 - 0.8
    composed = compose_grids(identity, grid2)
    torch.testing.assert_close(composed, grid2, atol=1e-6, rtol=1e-5)


def test_physical_grid_roundtrips_through_normalized_conversion():
    shape = (5, 6)
    spacing = (1.3, 0.8)
    origin = (2.0, -1.0)
    direction = ((0.0, -1.0), (1.0, 0.0))
    phys = get_physical_grid_torch(shape, spacing, origin, direction, dtype=torch.double)
    norm = physical_to_normalized_torch(phys, shape, spacing, origin, direction)
    identity = _identity_grid(shape)
    torch.testing.assert_close(norm, identity, atol=1e-8, rtol=1e-6)


def test_physical_to_normalized_cached_matches_uncached():
    shape = (4, 5)
    spacing = (1.1, 0.9)
    origin = (0.5, -0.2)
    direction = ((1.0, 0.0), (0.0, 1.0))
    phys = get_physical_grid_torch(shape, spacing, origin, direction, dtype=torch.double)
    norm_uncached = physical_to_normalized_torch(phys, shape, spacing, origin, direction)

    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    shape_t = torch.tensor(shape, dtype=torch.double)
    spacing_t = torch.tensor(spacing_rev, dtype=torch.double)
    origin_t = torch.tensor(origin_rev, dtype=torch.double)
    direction_t = torch.tensor(direction_rev, dtype=torch.double)
    norm_cached = physical_to_normalized_torch_cached(phys, shape_t, spacing_t, origin_t, direction_t)

    torch.testing.assert_close(norm_cached, norm_uncached, atol=1e-8, rtol=1e-6)
