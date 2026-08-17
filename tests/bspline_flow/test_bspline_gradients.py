import pytest
import torch

from antstorch.bspline_flow import BSplineDomain, CubicBSplineSynthesis


@pytest.mark.parametrize(
    "domain,shape",
    [(BSplineDomain((5, 4)), (1, 2, 4, 5)), (BSplineDomain((4, 3, 3)), (1, 3, 4, 4, 4))],
)
def test_gradcheck(domain, shape):
    coefficients = torch.randn(shape, dtype=torch.double, requires_grad=True)
    layer = CubicBSplineSynthesis(domain, chunk_size=7)
    assert torch.autograd.gradcheck(layer, (coefficients,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_gradient_reaches_coefficients():
    coefficients = torch.randn(2, 2, 5, 6, dtype=torch.double, requires_grad=True)
    CubicBSplineSynthesis(BSplineDomain((7, 6)))(coefficients).square().mean().backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert torch.count_nonzero(coefficients.grad) > 0
