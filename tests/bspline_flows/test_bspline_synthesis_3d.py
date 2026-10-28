import torch

from antstorch.bspline_flows import ImageDomain, CubicBSplineSynthesis


def test_zero_constant_batch_and_chunks_3d():
    domain = ImageDomain((8, 7, 6), spacing=(0.7, 1.2, 2.4), origin=(3, -2, 9))
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
    output = CubicBSplineSynthesis(ImageDomain((9, 8, 7)), closed=True)(coefficients)
    torch.testing.assert_close(output, torch.full_like(output, 0.75), rtol=2e-14, atol=2e-14)


def test_separable_kronecker_fit_matches_scattered_one_hot_3d():
    from antstorch.bspline_flows.bspline_synthesis import (
        _bspline_fit_context,
        _bspline_fit_geometry,
        _bspline_fit_solve,
    )

    torch.manual_seed(42)
    sample_shape = (14, 11, 9)
    lattice_itk = (6, 5, 7)
    eps = 1e-6
    n_samples = sample_shape[0] * sample_shape[1] * sample_shape[2]
    weights = torch.rand(2, n_samples, dtype=torch.double)
    residuals = torch.randn(2, n_samples, dtype=torch.double)

    geom_sep = _bspline_fit_geometry(sample_shape, lattice_itk, torch.double, torch.device("cpu"), eps)
    assert getattr(geom_sep, "separable", None) is not None
    ctx_sep = _bspline_fit_context(weights, geom_sep, stable_accumulation=True)
    assert ctx_sep["mode"] == "separable"
    out_sep = _bspline_fit_solve(residuals, weights, ctx_sep, eps)

    # Strip .separable by converting to a plain 4-tuple so _bspline_fit_context exercises
    # both the stable one-hot and vectorized scatter paths on the exact same inputs.
    geom_tuple = tuple(geom_sep)
    ctx_stable = _bspline_fit_context(weights, geom_tuple, stable_accumulation=True)
    assert ctx_stable["mode"] == "stable"
    out_stable = _bspline_fit_solve(residuals, weights, ctx_stable, eps)

    ctx_vec = _bspline_fit_context(weights, geom_tuple, stable_accumulation=False)
    assert ctx_vec["mode"] == "vectorized"
    out_vec = _bspline_fit_solve(residuals, weights, ctx_vec, eps)

    torch.testing.assert_close(out_sep, out_stable, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(out_sep, out_vec, rtol=1e-13, atol=1e-13)
