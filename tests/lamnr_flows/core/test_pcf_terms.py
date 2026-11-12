"""Principal component flow terms (--pcf-alpha, --axis-beta) of the LAMNr Glow trainers."""
import copy

import pytest
import torch

from antsnormflows import MultiscaleFlow
from antsnormflows.distributions.base import DiagGaussian
from antsnormflows.flows import GlowBlock2d, Merge, Squeeze2d

from antstorch.lamnr_flows.misc.pcf_terms import axis_alignment_loss, pcf_penalty_bpd, ramp
from antstorch.lamnr_flows.scripts import train_lamnr_glow_2d as g2
from antstorch.lamnr_flows.scripts import train_lamnr_glow_3d as g3

from .test_glow_image_channels import cohort


def small_glow(seed=0, channels=2, size=8):
    torch.manual_seed(seed)
    levels, flows, merges, base = 2, [], [], []
    for i in range(levels):
        flows.append([GlowBlock2d(channels * 2 ** (levels + 1 - i), 8, split_mode="channel", scale=True),
                      Squeeze2d()])
        if i > 0:
            merges.append(Merge())
            shape = (channels * 2 ** (levels - i), size // 2 ** (levels - i), size // 2 ** (levels - i))
        else:
            shape = (channels * 2 ** (levels + 1), size // 2 ** levels, size // 2 ** levels)
        base.append(DiagGaussian(shape))
    model = MultiscaleFlow(base, flows, merges, class_cond=False)
    model.inverse_and_log_det(torch.rand(4, channels, size, size))   # ActNorm initialization
    return model


def test_ramp():
    assert ramp(5, 0) == 1.0 and ramp(0, 10) == 0.0 and ramp(5, 10) == 0.5 and ramp(50, 10) == 1.0


def test_pcf_penalty_is_nonnegative_and_differentiable():
    model = small_glow().train()
    x = torch.rand(3, 2, 8, 8)
    penalty = pcf_penalty_bpd(model, x, num_dims=10 ** 6, n_dims=2 * 8 * 8)   # exact
    assert penalty.dim() == 0 and torch.isfinite(penalty) and penalty >= 0
    penalty.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_axis_alignment_is_zero_for_identical_views_and_bounded_otherwise():
    m1 = small_glow(seed=0).eval()
    m2 = copy.deepcopy(m1)
    x = torch.rand(3, 2, 8, 8)
    assert axis_alignment_loss([m1, m2], [x, x.clone()], num_dims=3).item() == pytest.approx(0.0, abs=1e-6)
    m3 = small_glow(seed=1).train()
    loss = axis_alignment_loss([m1, m3], [x, torch.rand(3, 2, 8, 8)], num_dims=2)
    assert 0.0 <= loss.item() <= 2.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m3.parameters())


def test_axis_alignment_rejects_mismatched_grids_and_single_view():
    m = small_glow()
    with pytest.raises(ValueError):
        axis_alignment_loss([m], [torch.rand(2, 2, 8, 8)], num_dims=1)
    with pytest.raises(ValueError):
        axis_alignment_loss([m, small_glow(size=16)], [torch.rand(2, 2, 8, 8), torch.rand(2, 2, 16, 16)], num_dims=1)


def _cli(tmp_path, views, channels_per_view, dim, extra):
    cli = ['--channels-per-view', str(channels_per_view)]
    for v in views:
        cli += ['--view', *v]
    cli += ['--split-manifest', str(tmp_path / 'split.csv'), '--H', '16', '--W', '16',
            '--L', '1', '--K', '1', '--hidden', '8', '--batch', '1', '--num-workers', '0',
            '--train-samples', '2', '--max-iter', '2', '--eval-interval', '2',
            '--devices', 'cpu', '--precision', 'float', '--sample-mode', 'off',
            '--disable-aug-anneal', '--out-dir', str(tmp_path / 'run')] + extra
    cli += ['--D', '16', '--spatial-dims', '3'] if dim == 3 else ['--slice-idx', '5']
    return cli


@pytest.mark.parametrize('dim,module,trainer_class', [(2, g2, g2.LAMNrGlow2DTrainer), (3, g3, g3.LAMNrGlow3DTrainer)])
def test_pcf_cpu_training_two_channels(tmp_path, dim, module, trainer_class):
    views, _, _, _ = cohort(tmp_path)
    args = module._build_args(_cli(tmp_path, [views[0], views[1]], 2, dim,
                                   ['--pcf-alpha', '0.1', '--pcf-dims', '2', '--pcf-warmup', '1']))
    assert args.pcf_alpha == 0.1 and args.pcf_dims == 2
    trainer = trainer_class()
    trainer.setup(args)
    trainer.train_loader.dataset.do_data_augmentation = False
    trainer.train()
    assert all(torch.isfinite(p).all() for m in trainer.models for p in m.parameters())


def test_axis_alignment_cpu_training_two_views(tmp_path):
    views, _, _, _ = cohort(tmp_path)
    args = g2._build_args(_cli(tmp_path, [views[0], views[1]], 1, 2,
                               ['--pcf-alpha', '0.1', '--axis-beta', '0.5', '--axis-dims', '2']))
    assert args.num_views == 2
    trainer = g2.LAMNrGlow2DTrainer()
    trainer.setup(args)
    trainer.train_loader.dataset.do_data_augmentation = False
    trainer.train()
    assert all(torch.isfinite(p).all() for m in trainer.models for p in m.parameters())


def test_axis_beta_requires_two_views(tmp_path):
    views, _, _, _ = cohort(tmp_path)
    args = g2._build_args(_cli(tmp_path, [views[0], views[1]], 2, 2, ['--axis-beta', '0.5']))
    trainer = g2.LAMNrGlow2DTrainer()
    trainer.setup(args)
    with pytest.raises(ValueError):
        trainer.train()
