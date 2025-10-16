# tests/test_glow_architectures.py
import os
import sys
from typing import Tuple

import pytest
import torch
import torch.nn as nn

# --- Prefer antstorch; fall back to a local builder module if running standalone ---
try:
    from antstorch import (
        create_glow_normalizing_flow_model_2d,
        create_glow_normalizing_flow_model_3d,
    )
except Exception:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    sys.path.insert(0, repo_root)
    from create_normalizing_flow_model import (  # type: ignore
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

        rec = (x_rec.to(dtype=torch.float32) - x.to(dtype=torch.float32)).abs()
        max_err = float(rec.max().detach().cpu())
        mean_err = float(rec.mean().detach().cpu())
        assert max_err <= max_err_tol, f"max|x-x_rec|={max_err:g} > {max_err_tol:g}"
        assert mean_err <= mean_err_tol, f"mean|x-x_rec|={mean_err:g} > {mean_err_tol:g}"

        total = (fwd_logdet + inv_logdet).detach().cpu().float()
        max_abs = float(total.abs().max())
        assert max_abs <= logdet_tol, f"max|fwd+inv logdet|={max_abs:g} > {logdet_tol:g}"


# ---------- NEW: likelihood helper to catch base–latent shape issues ----------

def _bases_of(model):
    """
    Return list of base distributions for the model (handles q0s ModuleList, q0 list, or single q0).
    """
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


def _safe_log_prob_sum_pairwise(bases, z_list):
    """
    Pair each latent zi with a base b that accepts its shape. We try each unused base
    and keep the first that works (i.e., doesn't raise a size mismatch in log_prob).
    If any zi cannot be paired, raise with a helpful diagnostic.
    """
    used = [False] * len(bases)
    total = None
    diag = []

    for idx, zi in enumerate(z_list):
        paired = False
        zi_shape = tuple(map(int, zi.shape))
        for j, b in enumerate(bases):
            if used[j]:
                continue
            try:
                lp = b.log_prob(zi)
                # success: shapes are compatible with this base
                used[j] = True
                total = lp if total is None else (total + lp)
                paired = True
                break
            except Exception as e:
                # keep a short diagnostic note, but keep trying others
                msg = str(e)
                if len(msg) > 120:
                    msg = msg[:117] + "..."
                diag.append(f"  zi[{idx}] {zi_shape} vs base[{j}] -> {msg}")
                continue

        if not paired:
            # no base worked for this zi; include diags to help fix the builder
            detail = "\n".join(diag[-min(6, len(diag)):])  # last few attempts
            raise AssertionError(
                "No compatible base found for latent "
                f"zi[{idx}] shape={zi_shape}.\nRecent attempts:\n{detail}"
            )

    # If some bases weren't used, that's fine for this test (it can happen if the model had extras),
    # but most Glow multiscale builders use exactly one base per latent.
    return total


def _log_prob_exact(model, x: torch.Tensor) -> torch.Tensor:
    """
    Exact log p(x) = sum_i log p_i(z_i) + log|det J|, pairing latents to bases by shape.
    Catches base–latent shape mistakes even if ordering differs.
    """
    z, logdet = model.inverse_and_log_det(x)

    if isinstance(z, (list, tuple)):
        bases = _bases_of(model)
        base_lp = _safe_log_prob_sum_pairwise(bases, list(z))
    else:
        # single-scale: require exactly one base
        bases = _bases_of(model)
        assert len(bases) >= 1, "Expected at least one base for single-scale flow"
        base_lp = bases[0].log_prob(z)

    lp = base_lp + logdet
    # Per-sample vector and finite
    assert lp.dim() == 1 and lp.shape[0] == x.shape[0], f"log_prob shape {tuple(lp.shape)} mismatch"
    assert torch.isfinite(lp).all(), "Non-finite log_prob encountered"
    return lp


# -----------------------
# 2-D: parameterized tests
# -----------------------

@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        ((1, 32, 32), 3, 32, 256, 2),  # C=1 (top-level unsqueeze makes channel-split valid)
        ((2, 32, 64), 4, 32, 256, 1),  # C=2
    ],
    ids=["2d_C1_L3_K32_256", "2d_C2_L4_K32_256"],
)
def test_glow2d_roundtrip_and_likelihood(device, shape, L, K, hidden, batch):
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
        verbose=True,
    ).to(device=device)

    x = torch.randn(batch, C, H, W, device=device, dtype=torch.float32)
    _roundtrip_assertions(model, x, max_err_tol=1e-1, mean_err_tol=1e-1, logdet_tol=1e-1)
    _ = _log_prob_exact(model, x)  # <-- new: catch base–latent mismatches


# -----------------------
# 3-D: parameterized tests
# -----------------------

@pytest.mark.parametrize(
    "shape,L,K,hidden,batch",
    [
        ((1, 32, 32, 32), 3, 32, 256, 2),  # C=1 (top-level unsqueeze makes channel-split valid)
        ((2, 32, 64, 128), 4, 32, 256, 1),  # C=2
    ],
    ids=["3d_C1_L3_K32_256", "3d_C2_L4_K32_256"],
)

def test_glow3d_roundtrip_and_likelihood(device, shape, L, K, hidden, batch):
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
        split_mode="channel",
        scale=True,
        scale_map="tanh",
        leaky=0.0,
        net_actnorm=True,
        scale_cap=3.0,
        verbose=True,
    ).to(device=device)

    x = torch.randn(batch, C, D, H, W, device=device, dtype=torch.float32)
    _roundtrip_assertions(model, x, max_err_tol=1e-1, mean_err_tol=1e-1, logdet_tol=1e-1)
    _ = _log_prob_exact(model, x)  # <-- new: catch base–latent mismatches
