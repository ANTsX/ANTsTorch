import pytest
import torch

from antstorch.bspline_flows import (
    ImageDomain,
    DeterministicBSplineRegistration,
    affine_displacement_field,
    compose_displacements,
    jacobian_determinant,
    warp_image,
)


@pytest.mark.parametrize(
    "size,lattice",
    [((12, 10), (6, 7)), ((9, 8, 7), (5, 6, 7))],
)
def test_synthetic_constant_translation_registration(size, lattice):
    dimension = len(size)
    domain = ImageDomain(size)
    moving = torch.randn((1, 1) + domain.torch_size, dtype=torch.double)
    coefficients = torch.zeros((1, dimension) + lattice, dtype=torch.double)
    coefficients[:, 0] = 0.4
    known = torch.zeros((1, dimension) + domain.torch_size, dtype=torch.double)
    known[:, 0] = 0.4
    fixed = warp_image(moving, known, domain, padding_mode="border")
    # A non-zero constant translation is incompatible with a stationary
    # boundary, so disable that independently useful registration default.
    model = DeterministicBSplineRegistration(
        domain,
        squaring_steps=5,
        padding_mode="border",
        stationary_boundary=False,
    )
    result = model(coefficients, moving, fixed)
    torch.testing.assert_close(result["velocity"], known, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(result["displacement"], known, rtol=2e-14, atol=2e-14)
    assert result["similarity"].item() < 1e-27
    assert torch.all(result["jacobian_determinant"] > 0)


def test_image_loss_gradient_reaches_bspline_coefficients():
    torch.manual_seed(10)
    domain = ImageDomain((9, 8))
    moving = torch.randn(1, 1, 8, 9, dtype=torch.double)
    fixed = torch.roll(moving, shifts=1, dims=-1)
    coefficients = (torch.randn(1, 2, 6, 7, dtype=torch.double) * 0.01).requires_grad_()
    model = DeterministicBSplineRegistration(
        domain, squaring_steps=3, coefficient_weight=1e-3, velocity_weight=1e-3, bending_weight=1e-3
    )
    result = model(coefficients, moving, fixed)
    result["loss"].backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert coefficients.grad.abs().max() > 0


def test_regularization_terms_are_optional_and_reported():
    domain = ImageDomain((7, 6))
    coefficients = torch.randn(1, 2, 5, 6) * 0.01
    image = torch.randn(1, 1, 6, 7)
    model = DeterministicBSplineRegistration(
        domain, coefficient_weight=0.1, velocity_weight=0.2, bending_weight=0.3
    )
    result = model(coefficients, image, image)
    expected = (
        result["similarity"]
        + 0.1 * result["coefficient_regularization"]
        + 0.2 * result["velocity_regularization"]
        + 0.3 * result["bending_regularization"]
    )
    torch.testing.assert_close(result["loss"], expected)


def test_zero_coefficients_with_initial_affine_displacement_reduces_to_the_affine():
    # With zero coefficients the SVF is exactly zero everywhere, so the total
    # transform must reduce to the supplied affine displacement alone.
    domain = ImageDomain((10, 9))
    moving = torch.randn(1, 1, 9, 10, dtype=torch.double)
    coefficients = torch.zeros(1, 2, 6, 5, dtype=torch.double)
    affine_field = affine_displacement_field(
        torch.eye(2, dtype=torch.double), torch.tensor([1.5, -0.5], dtype=torch.double), domain, moving
    )
    model = DeterministicBSplineRegistration(domain, squaring_steps=4, padding_mode="border")
    result = model.transform(coefficients, moving, initial_affine_displacement=affine_field)
    torch.testing.assert_close(result["velocity"], torch.zeros_like(result["velocity"]), rtol=0, atol=2e-14)
    torch.testing.assert_close(result["displacement"], affine_field, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        result["warped_moving"],
        warp_image(moving, affine_field, domain, padding_mode="border"),
        rtol=0,
        atol=2e-12,
    )


def test_initial_affine_displacement_matches_explicit_composition():
    domain = ImageDomain((9, 8))
    moving = torch.randn(1, 1, 8, 9, dtype=torch.double)
    coefficients = (torch.randn(1, 2, 5, 6, dtype=torch.double) * 0.05).requires_grad_(False)
    affine_field = affine_displacement_field(
        torch.eye(2, dtype=torch.double), torch.tensor([0.6, -1.1], dtype=torch.double), domain, moving
    )
    model = DeterministicBSplineRegistration(domain, squaring_steps=4)
    result = model.transform(coefficients, moving, initial_affine_displacement=affine_field)
    without_affine = model.transform(coefficients, moving)
    expected = compose_displacements(affine_field, without_affine["displacement"], domain)
    torch.testing.assert_close(result["displacement"], expected, rtol=0, atol=2e-13)
    torch.testing.assert_close(result["svf_displacement"], without_affine["displacement"], rtol=0, atol=2e-14)


def test_forward_reports_jacobian_of_the_composed_transform():
    domain = ImageDomain((8, 7))
    moving = torch.randn(1, 1, 7, 8, dtype=torch.double)
    fixed = torch.randn(1, 1, 7, 8, dtype=torch.double)
    coefficients = (torch.randn(1, 2, 5, 6, dtype=torch.double) * 0.05).requires_grad_(False)
    affine_field = affine_displacement_field(
        torch.eye(2, dtype=torch.double), torch.tensor([0.3, 0.2], dtype=torch.double), domain, moving
    )
    model = DeterministicBSplineRegistration(domain, squaring_steps=4)
    with_affine = model(coefficients, moving, fixed, initial_affine_displacement=affine_field)
    torch.testing.assert_close(
        with_affine["jacobian_determinant"],
        jacobian_determinant(with_affine["displacement"], domain),
        rtol=0,
        atol=2e-12,
    )
