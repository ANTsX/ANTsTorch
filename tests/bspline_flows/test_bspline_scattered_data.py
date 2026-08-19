import numpy as np
import pytest
import torch

from antstorch.bspline_flows import BSplineDomain, fit_bspline_coefficients
from antstorch.bspline_flows.bspline_scattered_data import (
    fit_bspline_displacement_field,
    fit_bspline_object_to_scattered_data,
)
from antstorch.bspline_flows.bspline_synthesis import synthesize_bspline_velocity


def test_single_level_scattered_fit_matches_dense_grid_fit():
    # Fitting every pixel of a regular grid as an independent scattered
    # point, at a single level, must reduce exactly to the already-validated
    # dense-grid least-squares fit (`fit_bspline_coefficients`).
    torch.manual_seed(0)
    H, W = 11, 14
    domain = BSplineDomain((W, H))
    values = torch.rand(1, 1, H, W, dtype=torch.double)
    lattice = (5, 6)

    # `lattice` is ITK (x, y) order, matching fit_bspline_coefficients'
    # convention -- mesh_size below must use the same axis order.
    dense_coefficients = fit_bspline_coefficients(values, domain, lattice)
    dense = synthesize_bspline_velocity(dense_coefficients, domain)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    scattered_data = values.reshape(-1, 1).to(torch.double)
    parametric_data = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.double)
    scattered_dense, scattered_coefficients = fit_bspline_object_to_scattered_data(
        scattered_data,
        parametric_data,
        parametric_domain_origin=[0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0],
        parametric_domain_size=[W, H],
        number_of_fitting_levels=1,
        mesh_size=(lattice[0] - 3, lattice[1] - 3),
        dtype=torch.double,
        return_coefficients=True,
    )
    torch.testing.assert_close(scattered_coefficients, dense_coefficients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(scattered_dense, dense, rtol=1e-12, atol=1e-12)


def test_multi_level_fit_reduces_residual_each_level():
    torch.manual_seed(3)
    point_count = 400
    parametric = torch.rand(point_count, 2, dtype=torch.double) * torch.tensor([29.0, 19.0])
    scattered = torch.sin(parametric[:, :1] * 0.3) * torch.cos(parametric[:, 1:] * 0.2)

    residuals = []
    for levels in (1, 2, 3, 4, 5):
        dense = fit_bspline_object_to_scattered_data(
            scattered,
            parametric,
            parametric_domain_origin=[0.0, 0.0],
            parametric_domain_spacing=[1.0, 1.0],
            parametric_domain_size=[30, 20],
            number_of_fitting_levels=levels,
            mesh_size=1,
            dtype=torch.double,
        )
        u = parametric[:, 0].round().long().clamp(0, 29)
        v = parametric[:, 1].round().long().clamp(0, 19)
        fitted_at_points = dense[0, 0, v, u]
        residuals.append((scattered[:, 0] - fitted_at_points).square().mean().item())

    for earlier, later in zip(residuals, residuals[1:]):
        assert later <= earlier + 1e-9


def test_fit_bspline_object_to_scattered_data_agrees_with_antspy_scalar():
    ants = pytest.importorskip("ants")
    rng = np.random.default_rng(42)
    size_x, size_y = 40, 25
    point_count = 800
    scattered = rng.normal(size=(point_count, 1))
    parametric = np.column_stack(
        [rng.uniform(0, size_x - 1, point_count), rng.uniform(0, size_y - 1, point_count)]
    )
    weights = rng.uniform(0.5, 1.5, point_count)

    kwargs = dict(
        parametric_domain_origin=[0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0],
        parametric_domain_size=[size_x, size_y],
        number_of_fitting_levels=4,
        mesh_size=2,
    )
    ants_arr = ants.fit_bspline_object_to_scattered_data(scattered, parametric, data_weights=weights, **kwargs).numpy()
    torch_arr = fit_bspline_object_to_scattered_data(
        scattered, parametric, data_weights=weights, dtype=torch.float64, **kwargs
    )[0, 0].numpy()

    # See fit_bspline_object_to_scattered_data's docstring: this package's
    # (N, C, *spatial) output is the ANTsPy/ITK array with a full axis
    # reversal (`.T` in 2-D).
    np.testing.assert_allclose(torch_arr.T, ants_arr, rtol=1e-3, atol=1e-3)


def test_fit_bspline_object_to_scattered_data_agrees_with_antspy_vector():
    ants = pytest.importorskip("ants")
    rng = np.random.default_rng(7)
    size = [60, 45]
    point_count = 50
    points = rng.uniform([5, 5], [54, 39], size=(point_count, 2))
    deltas = rng.normal(size=(point_count, 2)) * 3.0

    kwargs = dict(
        parametric_domain_origin=[0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0],
        parametric_domain_size=size,
        number_of_fitting_levels=3,
        mesh_size=(1, 1),
    )
    ants_arr = ants.fit_bspline_object_to_scattered_data(deltas, points, **kwargs).numpy()
    torch_arr = fit_bspline_object_to_scattered_data(deltas, points, dtype=torch.float64, **kwargs)[0].numpy()

    np.testing.assert_allclose(torch_arr[0].T, ants_arr[..., 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(torch_arr[1].T, ants_arr[..., 1], rtol=1e-3, atol=1e-3)


def test_fit_bspline_displacement_field_from_points_agrees_with_antspy():
    # enforce_stationary_boundary=False isolates the underlying fit itself
    # from the (documented, intentionally different) boundary treatment --
    # see the enforce_stationary_boundary note in the docstring.
    ants = pytest.importorskip("ants")
    rng = np.random.default_rng(7)
    size = [60, 45]
    point_count = 50
    points = rng.uniform([5, 5], [54, 39], size=(point_count, 2))
    deltas = rng.normal(size=(point_count, 2)) * 3.0

    ants_arr = ants.fit_bspline_displacement_field(
        displacement_origins=points,
        displacements=deltas,
        origin=[0.0, 0.0],
        spacing=[1.0, 1.0],
        size=size,
        direction=np.eye(2),
        number_of_fitting_levels=3,
        mesh_size=(1, 1),
        enforce_stationary_boundary=False,
    ).numpy()

    domain = BSplineDomain(size=size, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    torch_arr = fit_bspline_displacement_field(
        displacement_origins=points,
        displacements=deltas,
        domain=domain,
        number_of_fitting_levels=3,
        mesh_size=(1, 1),
        enforce_stationary_boundary=False,
    )[0].numpy()

    np.testing.assert_allclose(torch_arr[0].T, ants_arr[..., 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(torch_arr[1].T, ants_arr[..., 1], rtol=1e-3, atol=1e-3)


def test_fit_bspline_displacement_field_enforces_stationary_boundary():
    torch.manual_seed(5)
    domain = BSplineDomain((20, 16))
    field = torch.randn(1, 2, *domain.torch_size, dtype=torch.double) * 0.1
    fitted = fit_bspline_displacement_field(
        displacement_field=field,
        domain=domain,
        number_of_fitting_levels=2,
        mesh_size=1,
        enforce_stationary_boundary=True,
    )
    assert torch.all(fitted[..., 0, :] == 0)
    assert torch.all(fitted[..., -1, :] == 0)
    assert torch.all(fitted[..., :, 0] == 0)
    assert torch.all(fitted[..., :, -1] == 0)


def test_fit_bspline_displacement_field_combines_grid_and_points():
    torch.manual_seed(9)
    domain = BSplineDomain((18, 15))
    field = torch.randn(1, 2, *domain.torch_size, dtype=torch.double) * 0.05
    points = torch.rand(10, 2, dtype=torch.double) * torch.tensor([17.0, 14.0])
    deltas = torch.randn(10, 2, dtype=torch.double) * 0.2

    combined = fit_bspline_displacement_field(
        displacement_field=field,
        displacement_origins=points,
        displacements=deltas,
        domain=domain,
        number_of_fitting_levels=2,
        mesh_size=1,
    )
    field_only = fit_bspline_displacement_field(
        displacement_field=field, domain=domain, number_of_fitting_levels=2, mesh_size=1
    )
    assert torch.isfinite(combined).all()
    assert not torch.allclose(combined, field_only)


def test_fit_bspline_displacement_field_requires_field_or_points():
    domain = BSplineDomain((10, 10))
    with pytest.raises(ValueError):
        fit_bspline_displacement_field(domain=domain)


def test_fit_functions_reject_non_cubic_spline_order():
    domain = BSplineDomain((10, 10))
    field = torch.zeros(1, 2, *domain.torch_size)
    with pytest.raises(NotImplementedError):
        fit_bspline_displacement_field(displacement_field=field, domain=domain, spline_order=2)
    with pytest.raises(NotImplementedError):
        fit_bspline_object_to_scattered_data(
            torch.zeros(5, 1),
            torch.zeros(5, 2),
            parametric_domain_origin=[0.0, 0.0],
            parametric_domain_spacing=[1.0, 1.0],
            parametric_domain_size=[10, 10],
            spline_order=2,
        )


def test_gradients_flow_through_scattered_fit():
    torch.manual_seed(2)
    parametric = torch.rand(30, 2, dtype=torch.double) * torch.tensor([19.0, 14.0])
    scattered = torch.randn(30, 1, dtype=torch.double, requires_grad=True)
    dense = fit_bspline_object_to_scattered_data(
        scattered,
        parametric,
        parametric_domain_origin=[0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0],
        parametric_domain_size=[20, 15],
        number_of_fitting_levels=2,
        mesh_size=1,
        dtype=torch.double,
    )
    dense.sum().backward()
    assert scattered.grad is not None
    assert torch.isfinite(scattered.grad).all()
    assert scattered.grad.abs().sum() > 0
