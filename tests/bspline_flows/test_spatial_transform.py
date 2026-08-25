import pytest
import torch

from antstorch.bspline_flows import (
    ImageDomain,
    affine_displacement_field,
    compose_displacements,
    jacobian_determinant,
    physical_grid,
    warp_image,
)


@pytest.mark.parametrize("size", [(9, 7), (7, 6, 5)])
def test_zero_displacement_is_identity(size):
    domain = ImageDomain(size)
    image = torch.randn((2, 2) + domain.torch_size, dtype=torch.double)
    zero = torch.zeros((2, len(size)) + domain.torch_size, dtype=torch.double)
    torch.testing.assert_close(warp_image(image, zero, domain), image, rtol=0, atol=2e-15)


def test_image_impulse_translation_sign():
    domain = ImageDomain((9, 7))
    moving = torch.zeros(1, 1, 7, 9, dtype=torch.double)
    moving[0, 0, 3, 5] = 1
    displacement = torch.zeros(1, 2, 7, 9, dtype=torch.double)
    displacement[:, 0] = 2.0  # fixed x samples moving x+2
    warped = warp_image(moving, displacement, domain)
    assert warped[0, 0, 3, 3].item() == pytest.approx(1.0)
    assert torch.count_nonzero(warped > 0.5) == 1


def test_spacing_and_nonidentity_direction_are_physical():
    domain = ImageDomain(
        (8, 7), spacing=(2.0, 3.0), origin=(11.0, -4.0), direction=((0.0, -1.0), (1.0, 0.0))
    )
    moving = torch.arange(8, dtype=torch.double)[None, None, None, :].expand(1, 1, 7, 8)
    displacement = torch.zeros(1, 2, 7, 8, dtype=torch.double)
    displacement[:, 1] = 2.0  # direction[:, x] * spacing_x = (0, 2)
    warped = warp_image(moving, displacement, domain, padding_mode="border")
    expected = torch.arange(1, 9, dtype=torch.double).clamp_max(7)[None, None, None, :].expand_as(moving)
    torch.testing.assert_close(warped, expected, rtol=0, atol=2e-15)


def test_displacement_composition_constant_translations():
    domain = ImageDomain((8, 7), spacing=(0.5, 2.0))
    first = torch.zeros(2, 2, 7, 8, dtype=torch.double)
    second = torch.zeros_like(first)
    first[:, 0] = 1.25
    second[:, 1] = -3.0
    composed = compose_displacements(first, second, domain)
    torch.testing.assert_close(composed, first + second, rtol=0, atol=2e-15)


def test_warp_gradcheck():
    domain = ImageDomain((4, 4))
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
    domain = ImageDomain(size, spacing=(1.5,) * dimension)
    points = physical_grid(domain, torch.empty((), dtype=torch.double))
    displacement = 0.1 * points
    determinant = jacobian_determinant(displacement, domain)
    torch.testing.assert_close(determinant, torch.full_like(determinant, 1.1**dimension), rtol=1e-13, atol=1e-13)
    folding = -2.0 * points
    folding_det = jacobian_determinant(folding, domain)
    assert torch.count_nonzero(folding_det <= 0) == (0 if dimension == 2 else folding_det.numel())


def test_affine_displacement_field_identity_is_zero():
    domain = ImageDomain((6, 5))
    reference = torch.zeros(1, 1, 5, 6, dtype=torch.double)
    matrix = torch.eye(2, dtype=torch.double)
    translation = torch.zeros(2, dtype=torch.double)
    field = affine_displacement_field(matrix, translation, domain, reference)
    assert field.shape == (1, 2, 5, 6)
    torch.testing.assert_close(field, torch.zeros_like(field), rtol=0, atol=1e-14)


def test_affine_displacement_field_constant_translation_matches_manual():
    domain = ImageDomain((6, 5), spacing=(1.5, 2.0))
    reference = torch.zeros(1, 1, 5, 6, dtype=torch.double)
    matrix = torch.eye(2, dtype=torch.double)
    translation = torch.tensor([3.0, -1.0], dtype=torch.double)
    field = affine_displacement_field(matrix, translation, domain, reference)
    expected = translation.reshape(1, 2, 1, 1).expand(1, 2, 5, 6)
    torch.testing.assert_close(field, expected, rtol=0, atol=1e-13)


def test_affine_displacement_field_matches_physical_grid_formula():
    domain = ImageDomain((5, 4), spacing=(1.0, 0.5), origin=(2.0, -3.0))
    reference = torch.zeros(1, 1, 4, 5, dtype=torch.double)
    matrix = torch.tensor([[1.2, 0.1], [-0.2, 0.9]], dtype=torch.double)
    translation = torch.tensor([0.5, -0.25], dtype=torch.double)
    field = affine_displacement_field(matrix, translation, domain, reference)
    points = physical_grid(domain, reference).squeeze(0)
    expected = torch.einsum("ij,j...->i...", matrix, points) + translation.reshape(2, 1, 1) - points
    torch.testing.assert_close(field, expected.unsqueeze(0), rtol=0, atol=1e-12)


def test_affine_displacement_field_broadcasts_unbatched_to_batch():
    domain = ImageDomain((5, 4))
    reference = torch.zeros(3, 1, 4, 5, dtype=torch.double)
    matrix = torch.eye(2, dtype=torch.double)
    translation = torch.tensor([1.0, 2.0], dtype=torch.double)
    field = affine_displacement_field(matrix, translation, domain, reference)
    assert field.shape == (3, 2, 4, 5)
    for i in range(3):
        torch.testing.assert_close(field[i], field[0])


def test_affine_displacement_field_matches_batched_matrix_and_translation():
    domain = ImageDomain((5, 4))
    reference = torch.zeros(2, 1, 4, 5, dtype=torch.double)
    matrix = torch.eye(2, dtype=torch.double).unsqueeze(0).repeat(2, 1, 1)
    matrix[1] *= 2.0
    translation = torch.zeros(2, 2, dtype=torch.double)
    translation[1] = torch.tensor([1.0, -1.0], dtype=torch.double)
    field = affine_displacement_field(matrix, translation, domain, reference)
    torch.testing.assert_close(field[0], torch.zeros(2, 4, 5, dtype=torch.double), rtol=0, atol=1e-14)
    assert not torch.allclose(field[1], torch.zeros_like(field[1]))


def test_affine_displacement_field_composes_correctly_with_warp_image():
    # An affine displacement field, used through warp_image, must reproduce a
    # direct affine resample: p_moving = matrix @ p_fixed + translation.
    domain = ImageDomain((10, 9), spacing=(1.0, 1.0))
    moving = torch.arange(90, dtype=torch.double).reshape(1, 1, 9, 10)
    translation = torch.tensor([2.0, 0.0], dtype=torch.double)
    field = affine_displacement_field(torch.eye(2, dtype=torch.double), translation, domain, moving)
    warped = warp_image(moving, field, domain, padding_mode="border")
    index = (torch.arange(10) + 2).clamp(max=9)
    expected = moving[..., index]
    torch.testing.assert_close(warped, expected, rtol=0, atol=2e-13)


@pytest.mark.parametrize(
    "matrix_shape,translation_shape",
    [((2, 3), (2,)), ((3, 3), (2,)), ((2, 2), (3,)), ((2, 2, 2), (3, 2))],
)
def test_affine_displacement_field_rejects_bad_shapes(matrix_shape, translation_shape):
    domain = ImageDomain((5, 4))
    reference = torch.zeros(1, 1, 4, 5, dtype=torch.double)
    with pytest.raises(ValueError):
        affine_displacement_field(
            torch.zeros(matrix_shape, dtype=torch.double),
            torch.zeros(translation_shape, dtype=torch.double),
            domain,
            reference,
        )
