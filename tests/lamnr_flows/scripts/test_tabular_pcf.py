"""Principal component flow penalty (--pcf-alpha) in the tabular LAMNr trainer."""
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import antsnormflows as nf
from antsnormflows.utils import pcf_penalty
from antstorch.lamnr_flows.architectures.create_normalizing_flow_model import (
    create_real_nvp_normalizing_flow_model as create_rnvp,
)
from antstorch.lamnr_flows.misc.pcf_terms import pcf_penalty_bpd
from antstorch.lamnr_flows.scripts import train_lamnr_flows_tabular as tab


def _views(tmp_path, n=120):
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 3, n)
    a = pd.DataFrame({"x": t * np.cos(2 * t), "y": t * np.sin(2 * t)})
    b = pd.DataFrame({"u": t + rng.normal(0, 0.1, n), "v": t ** 2, "w": rng.normal(size=n)})
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    a.to_csv(paths[0], index=False)
    b.to_csv(paths[1], index=False)
    return [str(p) for p in paths]


def test_pcf_penalty_bpd_on_real_nvp_is_exact_for_small_views():
    torch.manual_seed(0)
    model = create_rnvp(3, K=4, q0=nf.distributions.DiagGaussian(3)).eval()
    x = torch.randn(10, 3)
    exact = pcf_penalty(model, x, num_dims=3).mean()
    assert pcf_penalty_bpd(model, x, num_dims=3, n_dims=3).item() == pytest.approx(
        exact.item() / (3 * np.log(2.0)), rel=1e-5)


def test_pcf_cli_and_training(tmp_path, monkeypatch):
    views = _views(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_lamnr_flows_tabular.py", "--views", *views, "--out-dir", str(tmp_path / "run"),
        "--K", "2", "--batch-size", "32", "--max-iter", "3", "--eval-interval", "3",
        "--pca-latent-dimension", "2", "--warmup-iters", "1",
        "--pcf-alpha", "0.5", "--pcf-dims", "8", "--pcf-warmup", "2", "--grad-norm-every", "1"])
    args = tab._build_args()
    assert args.pcf_alpha == 0.5 and args.pcf_dims == 8 and not hasattr(args, "axis_beta")
    trainer = tab.TabularLAMNrTrainer()
    trainer.setup(args)
    trainer.train()
    assert all(torch.isfinite(p).all() for m in trainer.models for p in m.parameters())
    df = pd.read_csv(tmp_path / "run" / "metrics.csv")
    assert {"pcf", "pcf_weight", "objective"} <= set(df.columns) and df["pcf"].notna().all()
    assert df["grad_norm"].notna().all() and df["grad_norm_reg"].notna().all()


def test_defaults_leave_pcf_off(tmp_path, monkeypatch):
    views = _views(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_lamnr_flows_tabular.py", "--views", *views])
    args = tab._build_args()
    assert args.pcf_alpha == 0.0


def test_pcf_training_makes_jacobian_rows_orthogonal():
    """On a 2-D spiral, training with the penalty drives -I_hat towards 0."""
    def data(n):
        t = torch.rand(n) * 3
        return torch.stack([t * torch.cos(2 * t), t * torch.sin(2 * t)], 1) + 0.05 * torch.randn(n, 2)

    finals = {}
    for alpha in (0.0, 1.0):
        torch.manual_seed(0)
        model = create_rnvp(2, K=8, q0=nf.distributions.DiagGaussian(2))
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        for _ in range(600):
            x = data(256)
            loss = (-model.log_prob(x)).mean()
            if alpha:
                loss = loss + alpha * pcf_penalty(model, x, num_dims=2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        finals[alpha] = float(pcf_penalty(model, data(2000), num_dims=2).mean())
    assert finals[1.0] < 0.5 * finals[0.0]
