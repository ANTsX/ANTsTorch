import pytest
import torch

from antstorch.bspline_flows import (
    BSplineDomain,
    PhysicalGradientDescent,
    compose_displacements,
    registration,
    warp_image,
)


def _blob(domain, dtype=torch.float32):
    axes = [torch.linspace(-1, 1, size, dtype=dtype) for size in domain.torch_size]
    coordinates = torch.meshgrid(*axes, indexing="ij")
    return torch.exp(-8 * sum(axis.square() for axis in coordinates))[None, None]


@pytest.mark.parametrize("size", [(9, 8), (7, 6, 5)])
def test_registration_identity_shapes_and_finite_results(size):
    domain = BSplineDomain(size)
    image = _blob(domain)
    result = registration(image, image, domain, iterations=0, mesh_size=1, squaring_steps=2)

    expected_field = (1, len(size)) + domain.torch_size
    assert result["warpedmovout"].shape == image.shape
    assert result["velocity"].shape == expected_field
    assert result["fwdtransforms"].shape == expected_field
    assert result["invtransforms"].shape == expected_field
    assert result["jacobian_determinant"].shape == (1,) + domain.torch_size
    assert result["loss"].item() == pytest.approx(0.0)
    assert torch.isfinite(result["jacobian_determinant"]).all()
    assert not result["loss"].requires_grad


def test_registration_reduces_loss_for_synthetic_translation():
    torch.manual_seed(4)
    domain = BSplineDomain((16, 14))
    moving = _blob(domain)
    displacement = torch.zeros((1, 2) + domain.torch_size)
    displacement[:, 0] = 0.7
    fixed = warp_image(moving, displacement, domain, padding_mode="border")
    initial = registration(fixed, moving, domain, iterations=0, mesh_size=1, padding_mode="border")["loss"]
    result = registration(
        fixed,
        moving,
        domain,
        iterations=30,
        learning_rate=0.08,
        mesh_size=1,
        squaring_steps=3,
        padding_mode="border",
    )
    assert result["loss"] < initial
    assert result["loss_history"][-1] < result["loss_history"][0]


@pytest.mark.parametrize("similarity", ["mse", "ncc", "ants_ncc"])
def test_registration_similarity_modes_and_final_graph(similarity):
    domain = BSplineDomain((8, 7))
    moving = _blob(domain).requires_grad_()
    fixed = torch.roll(moving.detach(), 1, -1)
    result = registration(
        fixed,
        moving,
        domain,
        iterations=1,
        similarity=similarity,
        detach_outputs=False,
        return_loss_history=False,
        squaring_steps=1,
    )
    result["loss"].backward()
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()
    assert result["coefficients"].grad is not None
    assert torch.isfinite(result["coefficients"].grad).all()
    assert result["loss_history"] is None


def test_forward_inverse_composition_is_near_identity():
    domain = BSplineDomain((12, 11))
    coefficients = torch.randn(1, 2, 5, 5) * 0.02
    result = registration(
        _blob(domain),
        _blob(domain),
        domain,
        initial_coefficients=coefficients,
        iterations=0,
        squaring_steps=5,
    )
    composition = compose_displacements(
        result["fwdtransforms"], result["invtransforms"], domain
    )
    assert composition.abs().max() < 2e-3


def test_distinct_moving_domain_and_batch_are_supported():
    fixed_domain = BSplineDomain((8, 7), spacing=(1.0, 1.0))
    moving_domain = BSplineDomain((10, 9), spacing=(0.8, 0.75))
    fixed = torch.zeros(2, 1, *fixed_domain.torch_size)
    moving = torch.zeros(2, 1, *moving_domain.torch_size)
    result = registration(fixed, moving, fixed_domain, moving_domain, iterations=0)
    assert result["warpedmovout"].shape == fixed.shape
    assert result["coefficients"].shape[:2] == (2, 2)


def test_multiresolution_refines_lattice_and_reports_each_level():
    domain = BSplineDomain((17, 15), spacing=(0.8, 1.2), origin=(2.0, -3.0))
    image = _blob(domain)
    result = registration(
        image,
        image,
        domain,
        mesh_size=1,
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(1.5, 0.75, 0.0),
        iterations=(0, 0, 0),
    )
    # Open cubic lattice refinement: 4 -> 5 -> 7 control points.
    assert result["coefficients"].shape == (1, 2, 7, 7)
    assert result["level_loss_history"] == [[], [], []]
    assert result["warpedmovout"].shape == image.shape
    assert result["loss"].item() == pytest.approx(0.0)


def test_multiresolution_optimization_runs_coarse_to_fine():
    domain = BSplineDomain((17, 15))
    moving = _blob(domain)
    fixed = torch.roll(moving, 1, -1)
    result = registration(
        fixed,
        moving,
        domain,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(2, 2),
        learning_rate=(0.05, 0.02),
        squaring_steps=1,
    )
    assert len(result["level_loss_history"]) == 2
    assert [len(level) for level in result["level_loss_history"]] == [2, 2]
    assert result["loss_history"] == sum(result["level_loss_history"], [])
    assert torch.isfinite(result["loss"])


def test_multiresolution_3d_smoothing_and_refinement():
    domain = BSplineDomain((9, 8, 7), spacing=(0.7, 1.1, 1.4))
    image = _blob(domain)
    result = registration(
        image,
        image,
        domain,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.8, 0.0),
        iterations=(0, 0),
        squaring_steps=1,
    )
    assert result["coefficients"].shape == (1, 3, 5, 5, 5)
    assert result["velocity"].shape == (1, 3) + domain.torch_size
    assert torch.isfinite(result["jacobian_determinant"]).all()


def test_verbose_reports_levels_and_iterations(capsys):
    domain = BSplineDomain((8, 7))
    image = _blob(domain)
    registration(
        image,
        image,
        domain,
        shrink_factors=(2, 1),
        iterations=(1, 1),
        squaring_steps=1,
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "ANTsTorch B-spline SVF registration configuration:" in output
    assert "similarity: mse" in output
    assert "stationary_boundary: True" in output
    assert "shrink_factors: (2, 1)" in output
    assert "iterations: (1, 1)" in output
    assert "Resolution level 1/2" in output
    assert "Resolution level 2/2" in output
    assert "control_points=(4, 4), total_control_points=16" in output
    assert "control_points=(5, 5), total_control_points=25" in output
    assert output.count("iteration 0001") == 2
    assert "loss=" in output


def test_physical_gradient_descent_controls_dense_update_magnitude():
    torch.manual_seed(12)
    domain = BSplineDomain((9, 8), spacing=(2.0, 3.0))
    moving = torch.randn(1, 1, *domain.torch_size)
    fixed = torch.roll(moving, 1, -1)
    gradient_step = 0.1
    result = registration(
        fixed,
        moving,
        domain,
        optimizer="physical_gradient_descent",
        gradient_step=gradient_step,
        iterations=1,
        stationary_boundary=False,
        squaring_steps=1,
    )
    maximum_update = result["velocity"].square().sum(dim=1).sqrt().amax()
    expected = gradient_step * (2.0**2 + 3.0**2) ** 0.5
    torch.testing.assert_close(maximum_update, maximum_update.new_tensor(expected))


def test_physical_gradient_descent_instance_supports_momentum_and_smoothing():
    torch.manual_seed(13)
    domain = BSplineDomain((9, 8), spacing=(1.5, 2.0))
    moving = torch.randn(1, 1, *domain.torch_size)
    fixed = torch.roll(moving, 1, -2)
    optimizer = PhysicalGradientDescent(
        gradient_step=0.1,
        momentum=0.9,
        smoothing_sigma=1.0,
    )
    result = registration(
        fixed,
        moving,
        domain,
        optimizer=optimizer,
        iterations=2,
        stationary_boundary=False,
        squaring_steps=1,
    )
    assert torch.isfinite(result["loss"])
    assert optimizer._momentum_buffer is not None
    assert torch.isfinite(optimizer._momentum_buffer).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"optimizer": "sgd"}, "optimizer"),
        ({"optimizer": "physical_gradient_descent", "gradient_step": 0.09}, "gradient_step"),
        ({"optimizer": "physical_gradient_descent", "gradient_step": 0.26}, "gradient_step"),
        ({"similarity": "mi"}, "similarity"),
        ({"iterations": -1}, "iterations"),
        ({"mesh_size": (1, 2, 3)}, "mesh_size"),
        ({"coefficient_grid_size": 3}, "coefficient_grid_size"),
        ({"convergence_tolerance": -1.0}, "convergence_tolerance"),
        ({"shrink_factors": (3, 1)}, "shrink_factors"),
        ({"shrink_factors": (2, 1), "iterations": (1,)}, "iterations"),
        ({"shrink_factors": (2, 1), "smoothing_sigmas": (1.0,)}, "smoothing_sigmas"),
        ({"shrink_factors": (2, 1), "closed": True}, "closed axes"),
    ],
)
def test_registration_parameter_validation(kwargs, match):
    domain = BSplineDomain((8, 7))
    image = _blob(domain)
    with pytest.raises(ValueError, match=match):
        registration(image, image, domain, **kwargs)


def test_registration_rejects_incompatible_image_domain():
    domain = BSplineDomain((8, 7))
    with pytest.raises(ValueError, match="fixed tensor shape"):
        registration(torch.zeros(1, 1, 6, 8), torch.zeros(1, 1, 7, 8), domain)
