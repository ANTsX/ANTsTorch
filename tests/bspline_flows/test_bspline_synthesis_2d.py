import pytest
import torch

from antstorch.bspline_flows import ImageDomain, CubicBSplineSynthesis


def test_zero_and_constant_vector_fields_2d():
    domain = ImageDomain((11, 7), spacing=(1.3, 2.1), origin=(-4.0, 8.0), direction=((0.0, -1.0), (1.0, 0.0)))
    layer = CubicBSplineSynthesis(domain, chunk_size=13)
    zero = torch.zeros(2, 2, 6, 7, dtype=torch.double)
    torch.testing.assert_close(layer(zero), torch.zeros(2, 2, 7, 11, dtype=torch.double))
    constant = torch.tensor([2.5, -3.0], dtype=torch.double)[None, :, None, None].expand_as(zero).clone()
    expected = torch.tensor([2.5, -3.0], dtype=torch.double)[None, :, None, None].expand(2, 2, 7, 11)
    torch.testing.assert_close(layer(constant), expected, rtol=1e-14, atol=1e-14)


def test_impulse_has_compact_tensor_product_support_2d():
    domain = ImageDomain((17, 13))
    coefficients = torch.zeros(1, 2, 7, 8, dtype=torch.double)
    coefficients[0, 0, 3, 4] = 1.0
    output = CubicBSplineSynthesis(domain)(coefficients)[0, 0]
    assert output.max() > 0
    assert torch.count_nonzero(output) < output.numel()
    assert torch.all(output >= 0)


def test_stationary_boundary_zeros_all_faces_2d():
    output = CubicBSplineSynthesis(ImageDomain((9, 8)), stationary_boundary=True)(torch.randn(1, 2, 6, 7))
    assert torch.count_nonzero(output[..., 0, :]) == 0
    assert torch.count_nonzero(output[..., -1, :]) == 0
    assert torch.count_nonzero(output[..., :, 0]) == 0
    assert torch.count_nonzero(output[..., :, -1]) == 0


# Regression coverage for a float32 boundary-clamp bug: for an open axis,
# `dense_index * spans / (dense_size - 1)` is only exactly `spans` at the
# last sample in exact arithmetic. In float32 it can round to a value a few
# ULPs *above* `spans` -- not only at the last sample, occasionally at an
# interior one too, depending on the spans/dense_size ratio -- and nudging
# that already-overshot value down by a single nextafter() step does not
# reliably bring it back under `spans`. That silently set
# base = floor(coordinate) == spans, so the neighbor stencil base+3 reached
# `spans + 3 == lattice_size`, one index past the last valid coefficient --
# surfacing deep inside the vectorized gather as an unrelated-looking
# "index out of range in self" from index_select, not as anything pointing
# at boundary arithmetic. A scan of (dense_size, control_points) pairs
# across dense_size in [20, 300) found this in ~4.7% of valid combinations
# (482 / 10260) before the fix (clamping against the *exact* mathematical
# bound via nextafter(spans, -inf), the same technique already used by
# _bspline_fit_geometry, rather than nudging whatever value the raw
# multiplication happened to produce); the parameters below are a handful of
# those confirmed-failing combinations, kept as a fixed regression set
# alongside the dynamic scan so a future regression is caught even if the
# scan's range is later narrowed for speed. Reported by a user running the
# real ANTsTorch benchmark harness (antstorch.benchmark, 'bspline_syn' model)
# on real Mindboggle-101 data at the finest pyramid level, shape
# (160, 256, 256), 107 control points per axis -- reproduced and fixed here
# at a far smaller, fast-to-test scale since the bug is about the
# (dense_size, control_points) *ratio*, not overall size.
_KNOWN_FAILING_DENSE_CONTROL_POINT_PAIRS = [
    (27, 10), (40, 13), (40, 25), (46, 10), (46, 28), (52, 16), (86, 25), (86, 55),
]


@pytest.mark.parametrize("dense_size,control_points", _KNOWN_FAILING_DENSE_CONTROL_POINT_PAIRS)
def test_open_axis_boundary_clamp_does_not_overrun_lattice_2d(dense_size, control_points):
    # The second axis is kept minimal (4 control points, 4 samples) so this
    # stays cheap; the bug is per-axis independent, so exercising it on one
    # axis at a time is sufficient and keeps the parametrized sweep fast.
    domain = ImageDomain((dense_size, 4), spacing=(1.0, 1.0), origin=(0.0, 0.0), direction=((1.0, 0.0), (0.0, 1.0)))
    coefficients = torch.randn(1, 1, 4, control_points, dtype=torch.float32)
    output = CubicBSplineSynthesis(domain)(coefficients)
    assert output.shape == (1, 1, 4, dense_size)
    assert torch.isfinite(output).all()


def test_open_axis_boundary_clamp_matches_float64_reference():
    # float64 has enough precision that the pre-fix formula never overshoots
    # `spans` for these parameters, so it is a trustworthy reference for
    # what the float32 path (now clamped) should agree with.
    domain = ImageDomain((27, 4), spacing=(1.0, 1.0), origin=(0.0, 0.0), direction=((1.0, 0.0), (0.0, 1.0)))
    coefficients32 = torch.randn(1, 1, 4, 10, dtype=torch.float32)
    coefficients64 = coefficients32.to(torch.float64)
    output32 = CubicBSplineSynthesis(domain)(coefficients32)
    output64 = CubicBSplineSynthesis(domain)(coefficients64)
    torch.testing.assert_close(output32.to(torch.float64), output64, atol=1e-5, rtol=1e-5)
