import pytest
import torch

from antstorch.bspline_flows import cubic_bspline_basis


def test_cubic_basis_known_values_and_support():
    x = torch.tensor([-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0], dtype=torch.double)
    expected = torch.tensor([0.0, 1 / 48, 1 / 6, 2 / 3, 1 / 6, 1 / 48, 0.0], dtype=torch.double)
    torch.testing.assert_close(cubic_bspline_basis(x), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("fraction", [0.0, 0.125, 0.5, 0.999])
def test_four_translates_form_partition_of_unity(fraction):
    offsets = torch.arange(4, dtype=torch.double)
    weights = cubic_bspline_basis(torch.tensor(fraction) - offsets + 1.0)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-14)
