import pytest
import torch

from antstorch.bspline_flows import (
    ImageDomain,
    PhysicalGradientDescent,
    affine_displacement_field,
    compose_displacements,
    bspline_svf_registration,
    warp_image,
)


def _blob(domain, dtype=torch.float32):
    axes = [torch.linspace(-1, 1, size, dtype=dtype) for size in domain.torch_size]
    coordinates = torch.meshgrid(*axes, indexing="ij")
    return torch.exp(-8 * sum(axis.square() for axis in coordinates))[None, None]


@pytest.mark.parametrize("size", [(9, 8), (7, 6, 5)])
def test_registration_identity_shapes_and_finite_results(size):
    domain = ImageDomain(size)
    image = _blob(domain)
    result = bspline_svf_registration(image, image, domain, iterations=0, mesh_size=1, squaring_steps=2)

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
    domain = ImageDomain((16, 14))
    moving = _blob(domain)
    displacement = torch.zeros((1, 2) + domain.torch_size)
    displacement[:, 0] = 0.7
    fixed = warp_image(moving, displacement, domain, padding_mode="border")
    initial = bspline_svf_registration(fixed, moving, domain, iterations=0, mesh_size=1, padding_mode="border")["loss"]
    result = bspline_svf_registration(
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
    domain = ImageDomain((8, 7))
    moving = _blob(domain).requires_grad_()
    fixed = torch.roll(moving.detach(), 1, -1)
    result = bspline_svf_registration(
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
    domain = ImageDomain((12, 11))
    coefficients = torch.randn(1, 2, 5, 5) * 0.02
    result = bspline_svf_registration(
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


def test_registration_initial_affine_with_zero_svf_matches_affine_alone():
    # With mesh_size=1, iterations=0 the coefficients (and hence the SVF)
    # are exactly zero, so fwdtransforms (the *pure* SVF piece, now always
    # separate from the affine) must itself be exactly zero, the affine
    # must come back verbatim as affine_matrix/affine_translation, and the
    # composed warpedmovout (still computed internally) must still equal
    # the affine-alone warp.
    domain = ImageDomain((10, 9), spacing=(1.0, 1.0))
    moving = _blob(domain, dtype=torch.double)
    matrix = torch.eye(2, dtype=torch.double)
    translation = torch.tensor([1.5, -0.5], dtype=torch.double)
    expected_field = affine_displacement_field(matrix, translation, domain, moving)
    fixed = warp_image(moving, expected_field, domain, padding_mode="border")
    result = bspline_svf_registration(
        fixed,
        moving,
        domain,
        initial_affine=(matrix, translation),
        iterations=0,
        mesh_size=1,
        padding_mode="border",
    )
    torch.testing.assert_close(
        result["fwdtransforms"], torch.zeros_like(expected_field), rtol=0, atol=1e-10
    )
    torch.testing.assert_close(result["affine_matrix"], matrix.unsqueeze(0))
    torch.testing.assert_close(result["affine_translation"], translation.unsqueeze(0))
    torch.testing.assert_close(
        result["warpedmovout"],
        warp_image(moving, expected_field, domain, padding_mode="border"),
        rtol=0,
        atol=1e-10,
    )
    assert result["loss"].item() == pytest.approx(0.0, abs=1e-10)


def test_forward_inverse_composition_with_initial_affine_is_near_identity():
    torch.manual_seed(7)
    domain = ImageDomain((24, 22))
    coefficients = torch.randn(1, 2, 5, 5) * 0.02
    matrix = torch.tensor([[1.05, 0.03], [-0.02, 0.97]])
    translation = torch.tensor([0.4, -0.3])
    result = bspline_svf_registration(
        _blob(domain),
        _blob(domain),
        domain,
        initial_coefficients=coefficients,
        initial_affine=(matrix, translation),
        iterations=0,
        squaring_steps=5,
    )
    composition = compose_displacements(result["fwdtransforms"], result["invtransforms"], domain)
    # A non-trivial affine maps points near the domain boundary outside the
    # domain; compose_displacements then samples with padding_mode="border",
    # which is an approximation there by construction (not specific to this
    # feature — the same clamping applies to any dense field composition).
    # Check the interior, away from that boundary effect.
    interior = composition[..., 2:-2, 2:-2]
    assert interior.abs().max() < 2e-3


def test_registration_initial_affine_accepts_unbatched_and_batched_forms():
    domain = ImageDomain((8, 7))
    fixed = torch.zeros(2, 1, *domain.torch_size)
    moving = torch.zeros(2, 1, *domain.torch_size)
    unbatched = bspline_svf_registration(
        fixed, moving, domain, initial_affine=(torch.eye(2), torch.zeros(2)), iterations=0
    )
    batched = bspline_svf_registration(
        fixed,
        moving,
        domain,
        initial_affine=(torch.eye(2).unsqueeze(0).repeat(2, 1, 1), torch.zeros(2, 2)),
        iterations=0,
    )
    torch.testing.assert_close(unbatched["fwdtransforms"], batched["fwdtransforms"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"initial_affine": (torch.zeros(3, 3), torch.zeros(2))}, "initial_affine matrix"),
        ({"initial_affine": (torch.zeros(2, 2), torch.zeros(3))}, "initial_affine translation"),
        ({"initial_affine": (torch.zeros(2, 2, dtype=torch.double), torch.zeros(2))}, "initial_affine"),
    ],
)
def test_registration_initial_affine_validation(kwargs, match):
    domain = ImageDomain((8, 7))
    image = _blob(domain)
    with pytest.raises(ValueError, match=match):
        bspline_svf_registration(image, image, domain, **kwargs)


def test_registration_rejects_non_tuple_initial_affine():
    domain = ImageDomain((8, 7))
    image = _blob(domain)
    with pytest.raises(TypeError, match="initial_affine"):
        bspline_svf_registration(image, image, domain, initial_affine=[torch.eye(2), torch.zeros(2)])


def test_distinct_moving_domain_and_batch_are_supported():
    fixed_domain = ImageDomain((8, 7), spacing=(1.0, 1.0))
    moving_domain = ImageDomain((10, 9), spacing=(0.8, 0.75))
    fixed = torch.zeros(2, 1, *fixed_domain.torch_size)
    moving = torch.zeros(2, 1, *moving_domain.torch_size)
    result = bspline_svf_registration(fixed, moving, fixed_domain, moving_domain, iterations=0)
    assert result["warpedmovout"].shape == fixed.shape
    assert result["coefficients"].shape[:2] == (2, 2)


def test_multiresolution_refines_lattice_and_reports_each_level():
    domain = ImageDomain((17, 15), spacing=(0.8, 1.2), origin=(2.0, -3.0))
    image = _blob(domain)
    result = bspline_svf_registration(
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
    domain = ImageDomain((17, 15))
    moving = _blob(domain)
    fixed = torch.roll(moving, 1, -1)
    result = bspline_svf_registration(
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
    domain = ImageDomain((9, 8, 7), spacing=(0.7, 1.1, 1.4))
    image = _blob(domain)
    result = bspline_svf_registration(
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
    domain = ImageDomain((8, 7))
    image = _blob(domain)
    bspline_svf_registration(
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
    domain = ImageDomain((9, 8), spacing=(2.0, 3.0))
    moving = torch.randn(1, 1, *domain.torch_size)
    fixed = torch.roll(moving, 1, -1)
    gradient_step = 0.1
    result = bspline_svf_registration(
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
    domain = ImageDomain((9, 8), spacing=(1.5, 2.0))
    moving = torch.randn(1, 1, *domain.torch_size)
    fixed = torch.roll(moving, 1, -2)
    optimizer = PhysicalGradientDescent(
        gradient_step=0.1,
        momentum=0.9,
        smoothing_sigma=1.0,
    )
    result = bspline_svf_registration(
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
    domain = ImageDomain((8, 7))
    image = _blob(domain)
    with pytest.raises(ValueError, match=match):
        bspline_svf_registration(image, image, domain, **kwargs)


def test_registration_rejects_incompatible_image_domain():
    domain = ImageDomain((8, 7))
    with pytest.raises(ValueError, match="fixed tensor shape"):
        bspline_svf_registration(torch.zeros(1, 1, 6, 8), torch.zeros(1, 1, 7, 8), domain)
