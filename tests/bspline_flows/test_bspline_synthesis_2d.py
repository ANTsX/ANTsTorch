import pytest
import torch

from antstorch.bspline_flows import BSplineDomain, CubicBSplineSynthesis


def test_zero_and_constant_vector_fields_2d():
    domain = BSplineDomain((11, 7), spacing=(1.3, 2.1), origin=(-4.0, 8.0), direction=((0.0, -1.0), (1.0, 0.0)))
    layer = CubicBSplineSynthesis(domain, chunk_size=13)
    zero = torch.zeros(2, 2, 6, 7, dtype=torch.double)
    torch.testing.assert_close(layer(zero), torch.zeros(2, 2, 7, 11, dtype=torch.double))
    constant = torch.tensor([2.5, -3.0], dtype=torch.double)[None, :, None, None].expand_as(zero).clone()
    expected = torch.tensor([2.5, -3.0], dtype=torch.double)[None, :, None, None].expand(2, 2, 7, 11)
    torch.testing.assert_close(layer(constant), expected, rtol=1e-14, atol=1e-14)


def test_impulse_has_compact_tensor_product_support_2d():
    domain = BSplineDomain((17, 13))
    coefficients = torch.zeros(1, 2, 7, 8, dtype=torch.double)
    coefficients[0, 0, 3, 4] = 1.0
    output = CubicBSplineSynthesis(domain)(coefficients)[0, 0]
    assert output.max() > 0
    assert torch.count_nonzero(output) < output.numel()
    assert torch.all(output >= 0)


def test_stationary_boundary_zeros_all_faces_2d():
    output = CubicBSplineSynthesis(BSplineDomain((9, 8)), stationary_boundary=True)(torch.randn(1, 2, 6, 7))
    assert torch.count_nonzero(output[..., 0, :]) == 0
    assert torch.count_nonzero(output[..., -1, :]) == 0
    assert torch.count_nonzero(output[..., :, 0]) == 0
    assert torch.count_nonzero(output[..., :, -1]) == 0
