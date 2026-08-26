import numpy as np
import pytest
import torch

from antstorch.bspline_flows import ImageDomain, fit_bspline_coefficients
from antstorch.bspline_flows.bspline_scattered_data import (
    _as_bools,
    _bspline_fit_context,
    _bspline_fit_geometry,
    _bspline_fit_solve,
    _domain_boundary_mask,
    _mesh_size_to_lattice,
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
    domain = ImageDomain((W, H))
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

    domain = ImageDomain(size=size, spacing=(1.0, 1.0), origin=(0.0, 0.0))
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
    domain = ImageDomain((20, 16))
    field = torch.randn(1, 2, *domain.torch_size, dtype=torch.double) * 0.1
    fitted = fit_bspline_displacement_field(
        displacement_field=field,
        domain=domain,
        number_of_fitting_levels=2,
        mesh_size=1,
        enforce_stationary_boundary=True,
    )
    # ITK's finite 1e10 boundary weight constrains these values very close
    # to zero; unlike the former post-fit mask, it does not make them
    # bitwise zero.
    assert torch.all(fitted[..., 0, :].abs() < 1e-9)
    assert torch.all(fitted[..., -1, :].abs() < 1e-9)
    assert torch.all(fitted[..., :, 0].abs() < 1e-9)
    assert torch.all(fitted[..., :, -1].abs() < 1e-9)


def test_fit_bspline_displacement_field_combines_grid_and_points():
    torch.manual_seed(9)
    domain = ImageDomain((18, 15))
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
        enforce_stationary_boundary=False,
    )
    field_only = fit_bspline_displacement_field(
        displacement_field=field,
        domain=domain,
        number_of_fitting_levels=2,
        mesh_size=1,
        enforce_stationary_boundary=False,
    )
    assert torch.isfinite(combined).all()
    assert not torch.allclose(combined, field_only)


def test_fit_bspline_displacement_field_requires_field_or_points():
    domain = ImageDomain((10, 10))
    with pytest.raises(ValueError):
        fit_bspline_displacement_field(domain=domain)


def test_fit_functions_reject_non_cubic_spline_order():
    domain = ImageDomain((10, 10))
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


# --- Chunked single-level dense-grid fit path (memory fix) -----------------
#
# fit_bspline_displacement_field(displacement_field=..., number_of_fitting_
# levels=1, no scattered points) -- exactly what
# antstorch.syn.bridge.apply_bspline_smoothing_operator calls, once per SyN
# iteration, for the BSplineSyN regularizer ("bspline_syn" in
# antstorch.benchmark) -- now takes a fast path that fuses geometry
# construction, normal-equation accumulation, and the solve into one chunked
# pass (_bspline_fit_dense_grid_chunked), instead of materializing a
# (4**D, N) index/weight tensor for the whole dense grid up front. Reported
# by a user as a real CUDA OOM ("Tried to allocate 7.50 GiB") running
# syn_registration(regularizer="bspline") at full native resolution
# (256, 256, 160) with a fine update-field mesh (107 control points/axis) on
# a GPU with limited free memory. These tests exercise the new fast path
# directly, at a much smaller scale for speed, since the bug is about the
# unchunked construction being O(4**D * N) regardless of absolute size.


def _old_unchunked_single_level_fit(field, domain, mesh_size, enforce_stationary_boundary, eps=1e-6):
    """Reference: the exact pre-fix computation (still available via the
    lower-level primitives, which are untouched -- only
    fit_bspline_displacement_field's own single-level/no-scattered-points
    call path was changed), used to confirm the new fast path is numerically
    equivalent rather than just plausible-looking.
    """
    dim = field.shape[1]
    dtype, device = field.dtype, field.device
    closed_axes = _as_bools(False, dim)
    lattice_itk = _mesh_size_to_lattice(mesh_size, dim, 3, closed_axes)
    geometry = _bspline_fit_geometry(domain.torch_size, lattice_itk, dtype, device, eps)
    grid_point_count = geometry[0].shape[1]
    weight_field = field.new_ones((1, 1) + field.shape[2:])
    field_values = field.reshape(dim, -1)
    field_weights = weight_field.reshape(1, -1).expand(dim, grid_point_count)
    if enforce_stationary_boundary:
        mask = _domain_boundary_mask(domain, device)
        field_values = field_values.masked_fill(mask.unsqueeze(0), 0.0)
        field_weights = field_weights.masked_fill(mask.unsqueeze(0), 1.0e10)
    context = _bspline_fit_context(field_weights, geometry, device.type == "mps")
    coefficients_flat = _bspline_fit_solve(field_values, field_weights, context, eps)
    accumulated = coefficients_flat.reshape((1, dim) + tuple(reversed(lattice_itk)))
    return synthesize_bspline_velocity(accumulated, domain)


@pytest.mark.parametrize("dim,size,mesh_size", [(2, (23, 19), 3), (2, (17, 13), 1), (3, (14, 11, 9), 2)])
@pytest.mark.parametrize("enforce_stationary_boundary", [True, False])
def test_single_level_dense_fit_matches_old_unchunked_path(dim, size, mesh_size, enforce_stationary_boundary):
    torch.manual_seed(1)
    domain = ImageDomain(size)
    field = torch.randn(1, dim, *domain.torch_size, dtype=torch.double)
    reference = _old_unchunked_single_level_fit(field, domain, mesh_size, enforce_stationary_boundary)
    fast = fit_bspline_displacement_field(
        displacement_field=field,
        domain=domain,
        number_of_fitting_levels=1,
        mesh_size=mesh_size,
        enforce_stationary_boundary=enforce_stationary_boundary,
        chunk_size=97,
    )
    torch.testing.assert_close(fast, reference, rtol=1e-9, atol=1e-9)


def test_single_level_dense_fit_is_invariant_to_chunk_size():
    torch.manual_seed(2)
    domain = ImageDomain((26, 18, 12))
    field = torch.randn(1, 3, *domain.torch_size, dtype=torch.double)
    reference = fit_bspline_displacement_field(
        displacement_field=field, domain=domain, number_of_fitting_levels=1, mesh_size=4, chunk_size=10_000_000,
    )
    for chunk_size in (37, 4096):
        result = fit_bspline_displacement_field(
            displacement_field=field, domain=domain, number_of_fitting_levels=1, mesh_size=4, chunk_size=chunk_size,
        )
        torch.testing.assert_close(result, reference, rtol=1e-9, atol=1e-9)


def test_single_level_dense_fit_gradients_match_old_unchunked_path():
    torch.manual_seed(3)
    domain = ImageDomain((15, 13))
    base_field = torch.randn(1, 2, *domain.torch_size, dtype=torch.double)

    field_ref = base_field.clone().requires_grad_(True)
    _old_unchunked_single_level_fit(field_ref, domain, mesh_size=2, enforce_stationary_boundary=True).pow(2).sum().backward()

    field_fast = base_field.clone().requires_grad_(True)
    fit_bspline_displacement_field(
        displacement_field=field_fast, domain=domain, number_of_fitting_levels=1,
        mesh_size=2, enforce_stationary_boundary=True, chunk_size=53,
    ).pow(2).sum().backward()

    torch.testing.assert_close(field_fast.grad, field_ref.grad, rtol=1e-9, atol=1e-9)


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
