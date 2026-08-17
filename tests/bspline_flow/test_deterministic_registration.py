import pytest
import torch

from antstorch.bspline_flow import BSplineDomain, DeterministicBSplineRegistration, warp_image


@pytest.mark.parametrize(
    "size,lattice",
    [((12, 10), (6, 7)), ((9, 8, 7), (5, 6, 7))],
)
def test_synthetic_constant_translation_registration(size, lattice):
    dimension = len(size)
    domain = BSplineDomain(size)
    moving = torch.randn((1, 1) + domain.torch_size, dtype=torch.double)
    coefficients = torch.zeros((1, dimension) + lattice, dtype=torch.double)
    coefficients[:, 0] = 0.4
    known = torch.zeros((1, dimension) + domain.torch_size, dtype=torch.double)
    known[:, 0] = 0.4
    fixed = warp_image(moving, known, domain, padding_mode="border")
    model = DeterministicBSplineRegistration(domain, squaring_steps=5, padding_mode="border")
    result = model(coefficients, moving, fixed)
    torch.testing.assert_close(result["velocity"], known, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(result["displacement"], known, rtol=2e-14, atol=2e-14)
    assert result["similarity"].item() < 1e-27
    assert torch.all(result["jacobian_determinant"] > 0)


def test_image_loss_gradient_reaches_bspline_coefficients():
    torch.manual_seed(10)
    domain = BSplineDomain((9, 8))
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
    domain = BSplineDomain((7, 6))
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

