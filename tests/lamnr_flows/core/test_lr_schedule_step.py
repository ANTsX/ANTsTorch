"""The warm-up/decay LambdaLR must keep stepping after the warm-up when the decay is enabled."""
import pytest
import torch

from antstorch.lamnr_flows.core.train_lamnr_glow_base import lr_schedule_needs_step, make_warmup


def _lr_after(n_steps, warmup, gamma, steps, base=1e-3):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base)
    sched = make_warmup(opt, warmup, gamma, steps)
    for it in range(1, n_steps + 1):
        opt.step()
        if sched is not None and lr_schedule_needs_step(it, warmup, gamma, steps):
            sched.step()
    return opt.param_groups[0]["lr"]


def test_warmup_only_stops_after_the_warmup():
    assert lr_schedule_needs_step(10, 10, 1.0, 0)
    assert not lr_schedule_needs_step(11, 10, 1.0, 0)
    assert not lr_schedule_needs_step(11, 10, 0.5, 0)       # decay without steps: disabled


def test_decay_keeps_stepping_after_the_warmup():
    assert lr_schedule_needs_step(10_000, 10, 0.5, 100)


def test_learning_rate_decays_exponentially_after_the_warmup():
    lr = _lr_after(400, warmup=10, gamma=0.5, steps=100)
    assert lr == pytest.approx(1e-3 * 0.5 ** (400 / 100), rel=0.02)


def test_learning_rate_is_constant_without_decay():
    assert _lr_after(400, warmup=10, gamma=1.0, steps=100) == pytest.approx(1e-3)
