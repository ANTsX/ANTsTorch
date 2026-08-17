import numpy as np
import pytest
import torch

from antstorch.bspline_flow import BSplineDomain, CubicBSplineSynthesis
from antstorch.bspline_flow.itk_reference import coefficient_lattice_metadata, numpy_itk_reconstruction


@pytest.mark.parametrize("dimension", [2, 3])
def test_agrees_with_literal_itk_reconstruction_semantics(dimension):
    rng = np.random.default_rng(8128 + dimension)
    lattice_itk = (6, 5) if dimension == 2 else (6, 5, 4)
    output_size = (9, 7) if dimension == 2 else (7, 6, 5)
    components = dimension
    coefficients_itk = rng.normal(size=lattice_itk + (components,))
    reference_itk = numpy_itk_reconstruction(coefficients_itk, output_size)

    spatial_axes = tuple(range(dimension - 1, -1, -1))
    coefficients_torch = np.transpose(coefficients_itk, (dimension,) + spatial_axes)[None]
    actual = CubicBSplineSynthesis(BSplineDomain(output_size), chunk_size=11)(torch.from_numpy(coefficients_torch))[0]
    reference_torch = np.transpose(reference_itk, (dimension,) + spatial_axes)
    np.testing.assert_allclose(actual.numpy(), reference_torch, rtol=2e-14, atol=2e-14)


def test_coefficient_lattice_geometry_matches_itk_formula():
    domain = BSplineDomain((11, 9), spacing=(2.0, 3.0), origin=(10.0, 20.0), direction=((0.0, -1.0), (1.0, 0.0)))
    metadata = coefficient_lattice_metadata(domain, (8, 7))
    assert metadata["spacing"] == pytest.approx((4.0, 6.0))
    assert metadata["origin"] == pytest.approx((16.0, 16.0))
