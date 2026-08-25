import pytest
import torch

from antstorch.bspline_flows import ImageDomain, CubicBSplineSynthesis


def test_cpu_dtype_and_device_are_preserved():
    coefficients = torch.randn(1, 2, 5, 6, dtype=torch.float32)
    output = CubicBSplineSynthesis(ImageDomain((8, 7)))(coefficients)
    assert output.device.type == "cpu"
    assert output.dtype == coefficients.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_and_backward():
    coefficients = torch.randn(1, 3, 5, 6, 7, device="cuda", requires_grad=True)
    output = CubicBSplineSynthesis(ImageDomain((8, 7, 6)), chunk_size=31)(coefficients)
    output.sum().backward()
    assert output.device.type == "cuda"
    assert coefficients.grad is not None
