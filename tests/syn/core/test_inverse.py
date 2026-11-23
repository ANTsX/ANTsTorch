import pytest
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


# --- Ported from syntx: channels-first auto-detection ---


def test_update_inverse_field_nd_anderson_channels_first_matches_channels_last():
    torch.manual_seed(0)
    spatial = (8, 9, 10)
    dim = 3
    W_cl = 0.05 * torch.randn(1, *spatial, dim, dtype=torch.double)
    inv_cl = update_inverse_field_nd_anderson(W_cl, None, steps=10)

    W_cf = torch.movedim(W_cl, -1, 1)
    inv_cf = update_inverse_field_nd_anderson(W_cf, None, steps=10)

    assert inv_cf.shape == W_cf.shape
    torch.testing.assert_close(torch.movedim(inv_cf, 1, -1), inv_cl, atol=1e-10, rtol=0)


def test_update_inverse_field_nd_hybrid_lm_channels_first_matches_channels_last():
    torch.manual_seed(1)
    spatial = (7, 8, 9)
    dim = 3
    W_cl = 0.05 * torch.randn(1, *spatial, dim, dtype=torch.double)
    inv_cl = update_inverse_field_nd_hybrid_lm(
        W_cl, None, steps=8, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    W_cf = torch.movedim(W_cl, -1, 1)
    inv_cf = update_inverse_field_nd_hybrid_lm(
        W_cf, None, steps=8, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert inv_cf.shape == W_cf.shape
    torch.testing.assert_close(torch.movedim(inv_cf, 1, -1), inv_cl, atol=1e-10, rtol=0)


def test_update_inverse_field_nd_dispatcher_channels_first_matches_channels_last():
    torch.manual_seed(2)
    spatial = (8, 8, 8)
    dim = 3
    W_cl = 0.05 * torch.randn(1, *spatial, dim, dtype=torch.double)
    for method in ('anderson', 'fixed_point'):
        inv_cl = update_inverse_field_nd(W_cl, -W_cl.clone(), steps=5, method=method)
        W_cf = torch.movedim(W_cl, -1, 1)
        inv_cf = update_inverse_field_nd(W_cf, -W_cf.clone(), steps=5, method=method)
        assert inv_cf.shape == W_cf.shape, method
        torch.testing.assert_close(torch.movedim(inv_cf, 1, -1), inv_cl, atol=1e-10, rtol=0)


def test_update_inverse_field_nd_2d_is_unaffected_by_channels_first_detection():
    # 2-D fields never trigger the (B, dim, *spatial) heuristic (dim=2 spatial
    # axes can genuinely be the last axis with size 2), so channels-last stays
    # the only supported convention here -- this just checks nothing crashes
    # or silently permutes a 2-D field.
    W_cl = 0.05 * torch.randn(1, 6, 7, 2, dtype=torch.double)
    inv = update_inverse_field_nd_anderson(W_cl, None, steps=5)
    assert inv.shape == W_cl.shape


def test_update_inverse_field_nd_max_iters_alias_overrides_steps():
    torch.manual_seed(3)
    W_cl = 0.05 * torch.randn(1, 6, 6, 2, dtype=torch.double)
    inv_via_steps = update_inverse_field_nd(W_cl, -W_cl.clone(), steps=3, method='anderson')
    inv_via_alias = update_inverse_field_nd(W_cl, -W_cl.clone(), steps=999, max_iters=3, method='anderson')
    torch.testing.assert_close(inv_via_alias, inv_via_steps, atol=1e-10, rtol=0)


def test_fixed_point_stopping_option_preserves_physical_default():
    field = _constant_translation_field([0.05, 0.0])
    options = dict(method='fixed_point', steps=20, spacing=_SPACING,
                   origin=_ORIGIN, direction=_DIRECTION)
    initial = torch.zeros_like(field)
    default = update_inverse_field_nd(field, initial, **options)
    both = update_inverse_field_nd(field, initial, convergence_criterion='both', **options)
    either = update_inverse_field_nd(field, initial, convergence_criterion='either', **options)
    torch.testing.assert_close(default, both, atol=0, rtol=0)
    assert not torch.allclose(default, either)


# --- syntx v6.35.4 / R32 behaviour (ported 2026-11-18) -----------------------------------------------

def _smooth_field_3d(shape=(12, 14, 10), amplitude=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    coarse = torch.randn((1, 3, 3, 3, 3), generator=g, dtype=torch.float64)
    field = torch.nn.functional.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    field = amplitude * field.movedim(1, -1)
    from antstorch.syn.core.smoothing import get_boundary_mask
    return field * get_boundary_mask(shape, field.device, field.dtype)


@pytest.mark.parametrize("method", ["anderson", "fixed_point"])
def test_inverse_solvers_are_idempotent_on_a_converged_guess(method):
    from antstorch.syn.core.inverse import update_inverse_field_nd
    W = _smooth_field_3d()
    sp, org, dr = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3).flatten()
    inv = update_inverse_field_nd(W, None, steps=200, method=method, spacing=sp, origin=org, direction=dr,
                                  max_error_threshold=1e-3, mean_error_threshold=1e-5)
    again = update_inverse_field_nd(W, inv.clone(), steps=10, method=method, spacing=sp, origin=org, direction=dr,
                                    max_error_threshold=1e-3, mean_error_threshold=1e-5)
    assert torch.equal(again, inv)


def test_hybrid_lm_polish_reduces_inverse_error_and_never_increases_it_per_voxel():
    from antstorch.syn.core.inverse import (
        compute_inverse_identity_error_nd, update_inverse_field_nd, update_inverse_field_nd_hybrid_lm,
    )
    W = _smooth_field_3d(amplitude=1.5, seed=3)
    sp, org, dr = (1.0, 1.2, 0.9), (0.0, 0.0, 0.0), np.eye(3)
    rough = update_inverse_field_nd(W, None, steps=3, method="anderson", spacing=sp, origin=org, direction=dr)
    polished = update_inverse_field_nd_hybrid_lm(W, rough, steps=30, max_error_threshold=0.01,
                                                 mean_error_threshold=0.001, spacing=sp, origin=org, direction=dr)
    # The LM minimises the residual v(x) + u(x + v(x)) voxel by voxel, accepting strictly improving steps only.
    e0 = compute_inverse_identity_error_nd(rough, W, spacing=sp, origin=org, direction=dr)
    e1 = compute_inverse_identity_error_nd(polished, W, spacing=sp, origin=org, direction=dr)
    assert float(e1.max()) < float(e0.max())
    assert float(e1.mean()) < float(e0.mean())
    assert bool((e1 <= e0 + 1e-9).all())
