# tests/test_glow_architectures.py
import os
from math import prod
from typing import Tuple

import pytest
import torch
import torch.nn as nn

import antstorch


# -----------------------
# helpers
# -----------------------

@torch.no_grad()
def _roundtrip_assertions(
    model,
    x: torch.Tensor,
    max_err_tol: float = 1e-4,
    mean_err_tol: float = 1e-5,
    logdet_tol: float = 1e-4,
):
    """
    Run inverse → forward and assert:
      • reconstruction error is tiny
      • per-sample (fwd_logdet + inv_logdet) ~ 0
    Works for nf.MultiscaleFlow (z may be a list) or single-scale models.
    """
    model.eval()
    z, inv_logdet = model.inverse_and_log_det(x)
    x_rec, fwd_logdet = model.forward_and_log_det(z)

    # reconstruction
    rec = (x_rec.to(torch.float32) - x.to(torch.float32)).abs()
    max_err = float(rec.max().detach().cpu())
    mean_err = float(rec.mean().detach().cpu())
    assert max_err <= max_err_tol, f"max|x-x_rec|={max_err:g} > {max_err_tol:g}"
    assert mean_err <= mean_err_tol, f"mean|x-x_rec|={mean_err:g} > {mean_err_tol:g}"

    # logdet consistency (per-sample)
    s = (fwd_logdet + inv_logdet).detach().cpu().to(torch.float32)
    max_abs = float(s.abs().max())
    assert max_abs <= logdet_tol, f"max|fwd+inv logdet|={max_abs:g} > {logdet_tol:g}"


def _bases_of(model):
    """Return list of base distributions (handles q0s ModuleList, q0 list, or single q0)."""
    if hasattr(model, "q0s"):
        q0s = getattr(model, "q0s")
        if isinstance(q0s, (list, tuple, nn.ModuleList)):
            return list(q0s)
    if hasattr(model, "q0"):
        q0 = getattr(model, "q0")
        if isinstance(q0, (list, tuple, nn.ModuleList)):
            return list(q0)
        if q0 is not None:
            return [q0]
    raise RuntimeError("No base distribution(s) found on model (q0/q0s).")


@torch.no_grad()
def _safe_log_prob_sum_pairwise(bases, z_list):
    """
    Pair each latent zi with a base that accepts its shape. If lengths match, we zip.
    Otherwise, try to find a base with matching .shape or that can .log_prob(zi).
    """
    zs = z_list if isinstance(z_list, (list, tuple)) else [z_list]
    used = set()
    def _pick_base_for(z):
        # 1) try direct index if available
        idx = len(used)
        if idx < len(bases):
            return idx
        # 2) try by declared .shape
        for j, b in enumerate(bases):
            if j in used:
                continue
            shp = getattr(b, "shape", None)
            if shp is not None and tuple(shp) == tuple(z.shape[1:]):
                return j
        # 3) try by probing .log_prob
        for j, b in enumerate(bases):
            if j in used:
                continue
            try:
                _ = b.log_prob(z)
                return j
            except Exception:
                pass
        raise RuntimeError("Could not find a matching base for latent of shape "
                           f"{tuple(z.shape[1:])}")

    total = 0.0
    for z in zs:
        j = _pick_base_for(z)
        used.add(j)
        total = total + bases[j].log_prob(z)
    return total


@torch.no_grad()
def _log_prob_exact(model, x: torch.Tensor) -> torch.Tensor:
    """
    Compute exact log_prob via inverse: sum_i log q0_i(z_i) + log|det ∂f^{-1}/∂x|.
    """
    z_list, inv_logdet = model.inverse_and_log_det(x)
    bases = _bases_of(model)
    base_lp = _safe_log_prob_sum_pairwise(bases, z_list)
    lp = base_lp + inv_logdet
    assert torch.isfinite(lp).all(), "Non-finite log_prob encountered"
    return lp


@torch.no_grad()
def _sample_and_likelihood_assertions(
    model,
    input_shape: Tuple[int, ...],
    n: int = 3,
    atol: float = 5e-4,
    rtol: float = 5e-4,
):
    """
    Calls model.sample(n), normalizes whether it returns x or latents, and checks:
      • sampled x has shape (n, *input_shape)
      • model.log_prob(x_s) is finite and per-sample
      • if sample() returned log_q, it matches model.log_prob(x_s) within tol.
    Works for 2D or 3D (any event rank).
    """
    model.eval()
    out = model.sample(n)

    # Normalize return form
    if isinstance(out, tuple) and len(out) == 2:
        a, log_q = out
    else:
        a, log_q = out, None

    exp_x_shape = (n, *tuple(input_shape))

    # Determine whether 'a' is x or latents
    if isinstance(a, (list, tuple)):
        x_s, _ = model.forward_and_log_det(a)
    elif isinstance(a, torch.Tensor) and tuple(a.shape) == exp_x_shape:
        x_s = a
    else:
        # assume it's a single latent tensor; forward it
        x_s, _ = model.forward_and_log_det(a)

    # Shape check
    assert tuple(x_s.shape) == exp_x_shape, \
        f"sampled x has wrong shape {tuple(x_s.shape)}; expected {exp_x_shape}"

    # Likelihood on samples
    lp_model = model.log_prob(x_s)
    assert lp_model.dim() == 1 and lp_model.shape[0] == n, \
        f"log_prob(sampled x) has wrong shape {tuple(lp_model.shape)}"
    assert torch.isfinite(lp_model).all(), "Non-finite log_prob on sampled x"

    # If sample() provided log_q, compare with model.log_prob(x_s)
    if log_q is not None:
        lp_m = lp_model.detach().cpu().float()
        lq = torch.as_tensor(log_q).detach().cpu().float()
        assert torch.allclose(lp_m, lq, atol=atol, rtol=rtol), \
            f"log_prob(sampled x) != sample()'s log_q (max diff {(lp_m - lq).abs().max().item():g})"


# -----------------------
# pytest params
# -----------------------

def _device_params():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda:0")
    return devs


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        ((1, 64, 64), 3, 7, 128, 4),  # 2D: C1, L=3
        ((2, 64, 64), 4, 8, 128, 2),  # 2D: C2, L=4
    ],
)
def test_glow2d_roundtrip_and_likelihood(device, shape, L, K, hidden, batch):
    C, H, W = shape
    model = antstorch.create_glow_normalizing_flow_model_2d(
        input_shape=(C, H, W),
        L=L, K=K, hidden_channels=hidden,
        base="glow",
        split_mode="channel",
        scale=True,
        scale_map="tanh",
        leaky=0.0,
        net_actnorm=True,
        scale_cap=3.0,
        verbose=True,  # prints latent/base shapes (should be side-effect free)
    ).to(device=device)

    x = torch.randn(batch, C, H, W, device=device, dtype=torch.float32)

    # roundtrip & logdet consistency
    _roundtrip_assertions(model, x, max_err_tol=1e-1, mean_err_tol=1e-1, logdet_tol=1e-1)

    # exact likelihood via inverse should match model.log_prob
    lp_exact = _log_prob_exact(model, x)
    lp_model = model.log_prob(x)
    assert torch.allclose(lp_model, lp_exact, atol=1e-5, rtol=1e-5), \
        "model.log_prob != exact(log p) (2D)"

    # sampling: return shape + likelihood sanity
    _sample_and_likelihood_assertions(model, (C, H, W), n=3)


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        ((1, 32, 64, 128), 3, 7, 128, 2),  # 3D: C1, L=3
        ((2, 32, 64, 128), 4, 8, 128, 2),  # 3D: C2, L=4
        # ((1, 192, 256, 256), 3, 7, 128, 2),  # 3D: C1, L=3
    ],
)
def test_glow3d_roundtrip_and_likelihood(device, shape, L, K, hidden, batch):
    C, D, H, W = shape
    model = antstorch.create_glow_normalizing_flow_model_3d(
        input_shape=(C, D, H, W),
        L=L, K=K, hidden_channels=hidden,
        base="glow",
        split_mode="channel",
        scale=True,
        scale_map="tanh",
        leaky=0.0,
        net_actnorm=True,
        scale_cap=3.0,
        verbose=True,  # prints latent/base shapes (should be side-effect free)
    ).to(device=device)

    x = torch.randn(batch, C, D, H, W, device=device, dtype=torch.float32)

    # roundtrip & logdet consistency
    _roundtrip_assertions(model, x, max_err_tol=1e-1, mean_err_tol=1e-1, logdet_tol=1e-1)

    # exact likelihood via inverse should match model.log_prob
    lp_exact = _log_prob_exact(model, x)
    lp_model = model.log_prob(x)
    assert torch.allclose(lp_model, lp_exact, atol=1e-5, rtol=1e-5), \
        "model.log_prob != exact(log p) (3D)"

    # sampling: return shape + likelihood sanity
    _sample_and_likelihood_assertions(model, (C, D, H, W), n=2)
