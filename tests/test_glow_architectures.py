# tests/test_glow_architectures.py
import os
import sys
from typing import Tuple, List, Union

import pytest
import torch

from antstorch import (
    create_glow_normalizing_flow_model_2d,
    create_glow_normalizing_flow_model_3d,
)

@pytest.fixture(scope="module")
def device():
    # Respect an override like: PYTEST_DEVICE=cuda:0
    override = os.environ.get("PYTEST_DEVICE", "").strip()
    if override:
        return torch.device(override)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _roundtrip_assertions(
    model,
    x: torch.Tensor,
    max_err_tol: float = 1e-4,
    mean_err_tol: float = 1e-5,
    logdet_tol: float = 1e-4,
):
    """
    Run inverse → forward and assert:
      * reconstruction error is tiny
      * per-sample (fwd_logdet + inv_logdet) is ~ 0
    Works for both nf.NormalizingFlow and nf.MultiscaleFlow (z may be a list).
    """
    model.eval()
    with torch.no_grad():
        z, inv_logdet = model.inverse_and_log_det(x)
        x_rec, fwd_logdet = model.forward_and_log_det(z)

        # Compute reconstruction error in float32 for a stable comparison
        rec = (x_rec.to(dtype=torch.float32) - x.to(dtype=torch.float32)).abs()
        max_err = float(rec.max().detach().cpu())
        mean_err = float(rec.mean().detach().cpu())
        assert max_err <= max_err_tol, f"max|x-x_rec|={max_err:g} > {max_err_tol:g}"
        assert mean_err <= mean_err_tol, f"mean|x-x_rec|={mean_err:g} > {mean_err_tol:g}"

        # Log-det sanity
        total = (fwd_logdet + inv_logdet).detach().cpu().float()
        max_abs = float(total.abs().max())
        assert max_abs <= logdet_tol, f"max|fwd+inv logdet|={max_abs:g} > {logdet_tol:g}"


# -----------------------
# 2-D: parameterized tests
# -----------------------

@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        # Small & quick; covers C=1 (top-level unsqueeze first makes channel-split valid)
        ((1, 32, 32), 2, 2, 32, 2),
        # Also test C=2
        ((2, 32, 32), 2, 2, 32, 2),
    ],
    ids=["2d_C1_L2K2", "2d_C2_L2K2"],
)
def test_glow2d_roundtrip(device, shape, L, K, hidden, batch):
    torch.manual_seed(1234)
    C, H, W = shape
    model = create_glow_normalizing_flow_model_2d(
        input_shape=(C, H, W),
        L=L,
        K=K,
        hidden_channels=hidden,
        base="glow",
        glowbase_logscale_factor=3.0,
        glowbase_min_log=-5.0,
        glowbase_max_log=5.0,
        split_mode="channel",
        scale=True,
        scale_map="tanh",
        leaky=0.0,
        net_actnorm=True,
        scale_cap=3.0,
    ).to(device=device)

    x = torch.randn(batch, C, H, W, device=device, dtype=torch.float32)
    _roundtrip_assertions(model, x)


# -----------------------
# 3-D: parameterized tests
# -----------------------

@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        # Keep volumes modest so CI is snappy; D/H/W must be divisible by 2**L
        ((1, 16, 16, 16), 2, 1, 32, 1),
        ((1, 16, 32, 32), 2, 2, 32, 1),
    ],
    ids=["3d_small_L2K1", "3d_mid_L2K2"],
)
def test_glow3d_roundtrip(device, shape, L, K, hidden, batch):
    torch.manual_seed(4321)
    C, D, H, W = shape
    model = create_glow_normalizing_flow_model_3d(
        input_shape=(C, D, H, W),
        L=L,
        K=K,
        hidden_channels=hidden,
        base="glow",
        glowbase_logscale_factor=3.0,
        glowbase_min_log=-5.0,
        glowbase_max_log=5.0,
        split_mode="channel",  # works with the validated per-level ordering
        scale=True,
        scale_map="tanh",
        leaky=0.0,
        net_actnorm=True,
        scale_cap=3.0,
    ).to(device=device)

    x = torch.randn(batch, C, D, H, W, device=device, dtype=torch.float32)
    _roundtrip_assertions(model, x, max_err_tol=2e-4, mean_err_tol=2e-5, logdet_tol=2e-4)
