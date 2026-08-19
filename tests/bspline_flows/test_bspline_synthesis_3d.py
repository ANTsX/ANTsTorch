import torch

from antstorch.bspline_flows import BSplineDomain, CubicBSplineSynthesis


def test_zero_constant_batch_and_chunks_3d():
    domain = BSplineDomain((8, 7, 6), spacing=(0.7, 1.2, 2.4), origin=(3, -2, 9))
    coefficients = torch.ones(2, 3, 5, 6, 7, dtype=torch.double)
    coefficients[:, 0] *= 1.25
    coefficients[:, 1] *= -2.0
    coefficients[:, 2] *= 4.5
    output = CubicBSplineSynthesis(domain, chunk_size=17)(coefficients)
    assert output.shape == (2, 3, 6, 7, 8)
    expected = torch.tensor([1.25, -2.0, 4.5], dtype=torch.double)[None, :, None, None, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=2e-14, atol=2e-14)


def test_periodic_closed_dimensions_preserve_constant_3d():
    coefficients = torch.full((1, 3, 4, 5, 6), 0.75, dtype=torch.double)
    output = CubicBSplineSynthesis(BSplineDomain((9, 8, 7)), closed=True)(coefficients)
    torch.testing.assert_close(output, torch.full_like(output, 0.75), rtol=2e-14, atol=2e-14)
