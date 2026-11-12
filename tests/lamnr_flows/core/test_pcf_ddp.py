"""PCF and axis-alignment terms under DistributedDataParallel (2 CPU processes, gloo).

The terms are computed inside GlowStepWrapper.forward, i.e. inside DDP's forward.
The synchronized gradients must equal those of one process that averages the two
ranks' losses, for the likelihood, the PCF penalty and the axis alignment together.
"""
import copy
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from antsnormflows import MultiscaleFlow
from antsnormflows.distributions.base import DiagGaussian
from antsnormflows.flows import GlowBlock2d, Merge, Squeeze2d

from antstorch.lamnr_flows.core.train_lamnr_glow_base import GlowDDP, GlowStepWrapper
from antstorch.lamnr_flows.misc.pcf_terms import axis_loss_from_magnitudes

ALPHA, BETA, SEED = 0.3, 0.5, 1234


def _glow(seed, channels=2, size=8):
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
    model.inverse_and_log_det(torch.rand(8, channels, size, size))   # ActNorm initialization
    with torch.no_grad():                                           # leave the near-identity init
        for p in model.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return model.train()


def _data():
    g = torch.Generator().manual_seed(7)
    return [torch.rand(8, 2, 8, 8, generator=g) for _ in range(2)]   # two views, 8 samples


def _loss(models, xs):
    """Trainer objective on one batch: bpd + alpha * PCF (bits/dim) + beta * axis loss."""
    n_dims = xs[0][0].numel()
    kw = dict(pcf_dims=10 ** 6, axis_dims=2, axis_seed=SEED)        # exact PCF: no sampling noise
    outs = [m(x, **kw) for m, x in zip(models, xs)]
    bpd = sum(-o[0].mean() / (n_dims * math.log(2.0)) for o in outs)
    pcf = sum(o[2].float().mean() for o in outs) / (n_dims * math.log(2.0))
    axis = axis_loss_from_magnitudes([o[3] for o in outs])
    return bpd + ALPHA * pcf + BETA * axis


def _worker(rank, world, init_file, out_dir):
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    models = [GlowDDP(GlowStepWrapper(_glow(s))) for s in (0, 1)]
    xs = [x[4 * rank:4 * rank + 4] for x in _data()]
    _loss(models, xs).backward()
    grads = [[p.grad.detach().clone() for p in m.module.model.parameters()] for m in models]
    torch.save(grads, os.path.join(out_dir, f"grads_{rank}.pt"))
    dist.destroy_process_group()


def test_ddp_gradients_match_single_process(tmp_path):
    mp.spawn(_worker, args=(2, str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"grads_{r}.pt") for r in range(2)]

    # Reference: one process, same models and halves, losses averaged as DDP does.
    models = [GlowStepWrapper(_glow(s)) for s in (0, 1)]
    data = _data()
    loss = 0.5 * sum(_loss(models, [x[4 * r:4 * r + 4] for x in data]) for r in range(2))
    loss.backward()
    ref = [[p.grad for p in m.model.parameters()] for m in models]

    for v in range(2):
        for g0, g1, gr in zip(ranks[0][v], ranks[1][v], ref[v]):
            torch.testing.assert_close(g0, g1, rtol=1e-5, atol=1e-6)    # ranks synchronized
            torch.testing.assert_close(g0, gr, rtol=1e-4, atol=1e-5)    # equal to the reference
    assert any(g.abs().sum() > 0 for g in ranks[0][0])


def test_wrapper_returns_two_outputs_without_pcf():
    out = GlowStepWrapper(_glow(0))(torch.rand(2, 2, 8, 8))
    assert len(out) == 2


def test_wrapper_penalty_matches_standalone():
    from antsnormflows.utils import pcf_penalty
    model = _glow(0).double().eval()
    x = torch.rand(3, 2, 8, 8, dtype=torch.float64)
    _, _, pen, mags = GlowStepWrapper(model)(x, pcf_dims=10 ** 6)
    assert mags is None
    torch.testing.assert_close(pen, pcf_penalty(model, x, num_dims=10 ** 6), rtol=1e-6, atol=1e-6)


def test_wrapper_subset_and_levels():
    """pcf_samples: likelihood on the full batch, penalty on the first M samples."""
    from antsnormflows.utils import pcf_penalty
    model = _glow(0).double().eval()
    x = torch.rand(4, 2, 8, 8, dtype=torch.float64)
    logp, _, pen, mags = GlowStepWrapper(model)(x, pcf_dims=10 ** 6, pcf_samples=2)
    assert logp.shape == (4,) and pen.shape == (2,) and mags is None
    torch.testing.assert_close(pen, pcf_penalty(model, x[:2], num_dims=10 ** 6), rtol=1e-6, atol=1e-6)
    _, _, pen1, _ = GlowStepWrapper(model)(x, pcf_dims=10 ** 6, pcf_samples=2, pcf_levels=1)
    assert pen1.shape == (2,) and torch.isfinite(pen1).all()


def _worker_lazy(rank, world, init_file, out_dir):
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    model = GlowDDP(GlowStepWrapper(_glow(0)))
    x = _data()[0][4 * rank:4 * rank + 4]
    logp, _, pen, _ = model(x, pcf_dims=10 ** 6, pcf_samples=2)
    n = x[0].numel()
    (-logp.mean() / (n * math.log(2.0)) + ALPHA * pen.mean() / (n * math.log(2.0))).backward()
    torch.save([p.grad.detach().clone() for p in model.module.model.parameters()],
               os.path.join(out_dir, f"lazy_{rank}.pt"))
    dist.destroy_process_group()


def test_ddp_gradients_with_pcf_subset(tmp_path):
    mp.spawn(_worker_lazy, args=(2, str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"lazy_{r}.pt") for r in range(2)]
    model = GlowStepWrapper(_glow(0))
    total = 0
    for r in range(2):
        x = _data()[0][4 * r:4 * r + 4]
        logp, _, pen, _ = model(x, pcf_dims=10 ** 6, pcf_samples=2)
        n = x[0].numel()
        total = total + 0.5 * (-logp.mean() / (n * math.log(2.0)) + ALPHA * pen.mean() / (n * math.log(2.0)))
    total.backward()
    for g0, g1, gr in zip(ranks[0], ranks[1], [p.grad for p in model.model.parameters()]):
        torch.testing.assert_close(g0, g1, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g0, gr, rtol=1e-4, atol=1e-5)
