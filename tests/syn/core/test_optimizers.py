import pytest
import torch

from antstorch.syn.core import (
    LARS,
    RegAdam,
    get_cfl_max_norm,
    compute_cfl_step,
    check_convergence,
)


def test_lars_step_moves_against_gradient():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = LARS([p], lr=0.5, trust_coefficient=0.1)
    p.grad = torch.tensor([1.0, 0.0])
    opt.step()
    assert p[0].item() < 1.0
    assert p[1].item() == pytest.approx(1.0)


def test_lars_step_skips_parameters_without_gradient():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = LARS([p])
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([1.0, 1.0]))


def test_regadam_descends_simple_quadratic():
    target = torch.tensor([3.0, -2.0], dtype=torch.double)
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.double))
    opt = RegAdam([p], lr=0.1, regularizer='none', max_step_norm=None)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = torch.sum((p - target) ** 2)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]
    torch.testing.assert_close(p.detach(), target, atol=0.05, rtol=0)


def test_regadam_respects_cfl_step_bound():
    p = torch.nn.Parameter(torch.zeros(1, 3, 3, 2, dtype=torch.double))
    opt = RegAdam([p], lr=10.0, regularizer='none', max_step_norm=0.1, spacing=[1.0, 1.0])
    p.grad = torch.ones_like(p)
    opt.step()
    step_norm = torch.sqrt((p.detach() ** 2).sum(dim=-1))
    assert step_norm.max().item() <= 0.1 + 1e-6


def test_get_cfl_max_norm():
    velocity = torch.zeros(1, 4, 4, 2)
    velocity[0, 0, 0, 0] = 3.0
    velocity[0, 0, 0, 1] = 4.0
    assert get_cfl_max_norm(velocity, [1.0, 1.0]) == pytest.approx(5.0)


def test_get_cfl_max_norm_normalizes_by_spacing():
    velocity = torch.zeros(1, 2, 2, 2)
    velocity[0, 0, 0, 0] = 2.0
    velocity[0, 0, 0, 1] = 0.0
    assert get_cfl_max_norm(velocity, [2.0, 1.0]) == pytest.approx(1.0)


def test_compute_cfl_step_uses_cfl_step_key():
    step = compute_cfl_step({'cfl_step': 0.5}, shrink_ratio=4.0)
    assert step == pytest.approx(0.5 * 2.0)


def test_compute_cfl_step_falls_back_to_grad_step():
    step = compute_cfl_step({'grad_step': 0.3}, shrink_ratio=1.0)
    assert step == pytest.approx(0.3)


def test_compute_cfl_step_uses_default():
    step = compute_cfl_step({}, shrink_ratio=1.0, default_grad_step=0.1)
    assert step == pytest.approx(0.1)


def test_check_convergence_false_for_short_history():
    assert check_convergence([1.0, 0.9], window_size=10) is False


def test_check_convergence_true_for_flat_losses():
    assert bool(check_convergence([1.0] * 15, window_size=10)) is True


def test_check_convergence_false_for_decreasing_losses():
    losses = [10.0 - 0.5 * i for i in range(15)]
    assert bool(check_convergence(losses, window_size=10, slope_threshold=1e-8)) is False
