import pytest
import torch

from antstorch.bspline_flow import (
    BSplineDomain,
    compose_displacements,
    jacobian_determinant,
    physical_grid,
    warp_image,
)


@pytest.mark.parametrize("size", [(9, 7), (7, 6, 5)])
def test_zero_displacement_is_identity(size):
    domain = BSplineDomain(size)
    image = torch.randn((2, 2) + domain.torch_size, dtype=torch.double)
    zero = torch.zeros((2, len(size)) + domain.torch_size, dtype=torch.double)
    torch.testing.assert_close(warp_image(image, zero, domain), image, rtol=0, atol=2e-15)


def test_image_impulse_translation_sign():
    domain = BSplineDomain((9, 7))
    moving = torch.zeros(1, 1, 7, 9, dtype=torch.double)
    moving[0, 0, 3, 5] = 1
    displacement = torch.zeros(1, 2, 7, 9, dtype=torch.double)
    displacement[:, 0] = 2.0  # fixed x samples moving x+2
    warped = warp_image(moving, displacement, domain)
    assert warped[0, 0, 3, 3].item() == pytest.approx(1.0)
    assert torch.count_nonzero(warped > 0.5) == 1


def test_spacing_and_nonidentity_direction_are_physical():
    domain = BSplineDomain(
        (8, 7), spacing=(2.0, 3.0), origin=(11.0, -4.0), direction=((0.0, -1.0), (1.0, 0.0))
    )
    moving = torch.arange(8, dtype=torch.double)[None, None, None, :].expand(1, 1, 7, 8)
    displacement = torch.zeros(1, 2, 7, 8, dtype=torch.double)
    displacement[:, 1] = 2.0  # direction[:, x] * spacing_x = (0, 2)
    warped = warp_image(moving, displacement, domain, padding_mode="border")
    expected = torch.arange(1, 9, dtype=torch.double).clamp_max(7)[None, None, None, :].expand_as(moving)
    torch.testing.assert_close(warped, expected, rtol=0, atol=2e-15)


def test_displacement_composition_constant_translations():
    domain = BSplineDomain((8, 7), spacing=(0.5, 2.0))
    first = torch.zeros(2, 2, 7, 8, dtype=torch.double)
    second = torch.zeros_like(first)
    first[:, 0] = 1.25
    second[:, 1] = -3.0
    composed = compose_displacements(first, second, domain)
    torch.testing.assert_close(composed, first + second, rtol=0, atol=2e-15)


def test_warp_gradcheck():
    domain = BSplineDomain((4, 4))
    image = torch.randn(1, 1, 4, 4, dtype=torch.double)
    # Bilinear interpolation is not differentiable exactly at integer sample
    # locations; check at a generic sub-voxel location instead.
    displacement = torch.empty(1, 2, 4, 4, dtype=torch.double)
    displacement[:, 0] = 0.23
    displacement[:, 1] = -0.17
    displacement.requires_grad_()
    assert torch.autograd.gradcheck(lambda value: warp_image(image, value, domain), (displacement,), atol=1e-5)


@pytest.mark.parametrize("dimension", [2, 3])
def test_jacobian_determinant_and_folding(dimension):
    size = (7,) * dimension
    domain = BSplineDomain(size, spacing=(1.5,) * dimension)
    points = physical_grid(domain, torch.empty((), dtype=torch.double))
    displacement = 0.1 * points
    determinant = jacobian_determinant(displacement, domain)
    torch.testing.assert_close(determinant, torch.full_like(determinant, 1.1**dimension), rtol=1e-13, atol=1e-13)
    folding = -2.0 * points
    folding_det = jacobian_determinant(folding, domain)
    assert torch.count_nonzero(folding_det <= 0) == (0 if dimension == 2 else folding_det.numel())
