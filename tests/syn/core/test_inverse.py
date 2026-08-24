import numpy as np
import torch

from antstorch.syn.core import (
    update_inverse_field_nd_hybrid_lm,
    integrate_time_varying_velocity_field,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
)

_SPATIAL = (9, 9)
_SPACING = (1.0, 1.0)
_ORIGIN = (0.0, 0.0)
_DIRECTION = ((1.0, 0.0), (0.0, 1.0))


def _constant_translation_field(shift):
    shift_t = torch.tensor(shift, dtype=torch.double)
    return shift_t.view(1, 1, 1, 2).expand(1, *_SPATIAL, 2).clone()


def test_update_inverse_field_nd_anderson_recovers_constant_translation():
    W_disp = _constant_translation_field([0.7, -0.3])
    W_inv = update_inverse_field_nd_anderson(
        W_disp, None, steps=50, spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    expected = -W_disp
    torch.testing.assert_close(W_inv[:, 2:-2, 2:-2, :], expected[:, 2:-2, 2:-2, :], atol=0.05, rtol=0)


def test_update_inverse_field_nd_hybrid_lm_recovers_constant_translation():
    W_disp = _constant_translation_field([0.5, 0.4])
    W_inv = update_inverse_field_nd_hybrid_lm(
        W_disp, None, steps=50, spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    expected = -W_disp
    torch.testing.assert_close(W_inv[:, 2:-2, 2:-2, :], expected[:, 2:-2, 2:-2, :], atol=0.05, rtol=0)


def test_update_inverse_field_nd_dispatches_to_matching_solver():
    W_disp = _constant_translation_field([0.6, -0.2])
    dispatched = update_inverse_field_nd(
        W_disp, None, steps=20, method='anderson', spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    direct = update_inverse_field_nd_anderson(
        W_disp, None, steps=20, spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    torch.testing.assert_close(dispatched, direct)

    dispatched_lm = update_inverse_field_nd(
        W_disp, None, steps=20, method='hybrid_lm', spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    direct_lm = update_inverse_field_nd_hybrid_lm(
        W_disp, None, steps=20, spacing=_SPACING, origin=_ORIGIN, direction=_DIRECTION
    )
    torch.testing.assert_close(dispatched_lm, direct_lm)


def test_update_inverse_field_nd_fixed_point_without_physical_metadata():
    # Unlike the 'anderson'/'hybrid_lm' solvers, the base 'fixed_point' path
    # does not default a None W_inv_disp to -W_disp, so an explicit initial
    # guess must be supplied.
    dim = 2
    W_disp = (0.15 * torch.ones(1, *_SPATIAL, dim, dtype=torch.double))
    W_inv = update_inverse_field_nd(W_disp, -W_disp.clone(), steps=50, method='fixed_point')
    expected = -W_disp
    torch.testing.assert_close(W_inv[:, 2:-2, 2:-2, :], expected[:, 2:-2, 2:-2, :], atol=0.05, rtol=0)


def test_compute_inverse_identity_error_nd_zero_for_exact_inverse():
    fwd = _constant_translation_field([0.4, -0.6])
    inv = -fwd
    err = compute_inverse_identity_error_nd(fwd, inv, spacing=_SPACING, origin=_ORIGIN, direction=np.eye(2))
    assert err.max().item() < 1e-6


def test_compute_inverse_identity_error_nd_positive_for_bad_inverse():
    fwd = _constant_translation_field([0.4, -0.6])
    bad_inv = torch.zeros_like(fwd)
    err = compute_inverse_identity_error_nd(fwd, bad_inv, spacing=_SPACING, origin=_ORIGIN, direction=np.eye(2))
    assert err.mean().item() > 0.3


def test_calculate_inverse_identity_error_reports_small_error_for_exact_inverse():
    fwd = _constant_translation_field([0.3, 0.2])
    inv = -fwd
    result = calculate_inverse_identity_error(fwd, inv, _SPACING, _ORIGIN, _DIRECTION)
    assert result['max_error'] < 1e-6
    assert result['mean_error'] < 1e-6
    assert result['error_map'].shape == _SPATIAL


def test_integrate_time_varying_velocity_field_zero_velocity_is_identity():
    spatial = (6, 6)
    dim = 2
    vel = [torch.zeros(1, *spatial, dim, dtype=torch.double) for _ in range(4)]
    phi = integrate_time_varying_velocity_field(vel, dt=0.25)
    torch.testing.assert_close(phi, torch.zeros_like(phi), atol=1e-10, rtol=0)


def test_integrate_time_varying_velocity_field_constant_velocity_matches_analytic_sum():
    spatial = (9, 9)
    dim = 2
    T = 4
    dt = 0.1
    v_const = 0.05 * torch.ones(1, *spatial, dim, dtype=torch.double)
    vel = [v_const.clone() for _ in range(T)]

    phi_euler = integrate_time_varying_velocity_field(vel, dt=dt, solver='euler')
    phi_rk4 = integrate_time_varying_velocity_field(vel, dt=dt, solver='rk4')
    expected = v_const * dt * T

    torch.testing.assert_close(phi_euler, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(phi_rk4, expected, atol=1e-6, rtol=1e-5)
