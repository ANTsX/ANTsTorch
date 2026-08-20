import pytest

from antstorch.bspline_flows import PhysicalGradientDescent


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gradient_step": 0.09}, "gradient_step"),
        ({"gradient_step": 0.26}, "gradient_step"),
        ({"momentum": -0.1}, "momentum"),
        ({"momentum": 1.0}, "momentum"),
        ({"smoothing_sigma": -1.0}, "smoothing_sigma"),
    ],
)
def test_physical_gradient_descent_validates_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PhysicalGradientDescent(**kwargs)


def test_physical_gradient_descent_repr_reports_configuration():
    optimizer = PhysicalGradientDescent(0.2, momentum=0.9, smoothing_sigma=1.0)
    assert repr(optimizer) == (
        "PhysicalGradientDescent(gradient_step=0.2, momentum=0.9, smoothing_sigma=1.0)"
    )
