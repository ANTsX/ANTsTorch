import pytest
import torch

from antstorch.bspline_flow import BSplineDomain, DeterministicBSplineRegistration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dimension", [2, 3])
def test_cpu_gpu_agreement(dimension):
    domain = BSplineDomain((7,) * dimension, spacing=(1.2,) * dimension)
    lattice = (5,) * dimension
    coefficients = torch.randn((1, dimension) + lattice, dtype=torch.float64) * 0.02
    moving = torch.randn((1, 1) + domain.torch_size, dtype=torch.float64)
    fixed = torch.randn_like(moving)
    model = DeterministicBSplineRegistration(domain, squaring_steps=3)
    cpu = model(coefficients, moving, fixed)
    gpu = model.cuda()(coefficients.cuda(), moving.cuda(), fixed.cuda())
    torch.testing.assert_close(gpu["warped_moving"].cpu(), cpu["warped_moving"], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(gpu["loss"].cpu(), cpu["loss"], rtol=2e-6, atol=2e-6)
