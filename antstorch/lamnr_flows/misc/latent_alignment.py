"""
latent_alignment.py
===================
Factory class that owns **all** latent-space alignment logic for the LAMNr
Glow trainers.  The training loop needs only to call:

    loss_total, L_align, w_nll, w_align = manager.compute(
        lat_flat, L_nll, it, s_nll, s_align
    )

and then ``loss_total.backward()``.

No normalizing-flow mathematics live here — only the alignment objectives
(VICReg, Barlow Twins, InfoNCE, HSIC, Pearson, MSE) and the optional
shared-subspace screening (CCA / HSIC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Re-export screening helpers so importers can stay in one place
# ---------------------------------------------------------------------------

Method = Literal["none", "cca", "hsic"]


@dataclass
class ScreenState:
    method: Method = "none"
    proj_dim: int = 0
    keep_dim: int = 0
    n_views: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    projectors: Optional[List[torch.Tensor]] = None  # for CCA  (D, r)
    masks: Optional[List[torch.Tensor]] = None        # for HSIC (D,)
    meta: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Screening internals
# ---------------------------------------------------------------------------

def _whiten(
    F: torch.Tensor, ridge: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = F.mean(dim=0, keepdim=True)
    X = F - mu
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    cov = cov + ridge * torch.eye(cov.shape[0], device=F.device, dtype=F.dtype)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=1e-12)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    return X @ inv_sqrt, mu, inv_sqrt


@torch.no_grad()
def _cca_pair(
    A: torch.Tensor, B: torch.Tensor, ridge: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xa, _, Wa = _whiten(A, ridge=ridge)
    Xb, _, Wb = _whiten(B, ridge=ridge)
    M = Xa.T @ Xb / max(1, A.shape[0] - 1)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Ua = Wa @ U
    Vb = Wb @ Vh.T
    return Ua, S, Vb


@torch.no_grad()
def _screen_cca(
    feats: List[torch.Tensor], keep_dim: int, ridge: float = 1e-3
) -> Tuple[List[torch.Tensor], Dict]:
    n = len(feats)
    B, D = feats[0].shape
    accum = [
        torch.zeros(D, D, device=feats[0].device, dtype=feats[0].dtype)
        for _ in range(n)
    ]
    spectra = []
    for i in range(n):
        for j in range(i + 1, n):
            Ui, S, Vj = _cca_pair(feats[i], feats[j], ridge=ridge)
            ui = Ui[:, :keep_dim]
            vj = Vj[:, :keep_dim]
            accum[i] = accum[i] + ui @ ui.T
            accum[j] = accum[j] + vj @ vj.T
            spectra.append(S.detach().cpu())
    projectors = []
    for i in range(n):
        A = accum[i] / max(1, n - 1) + 1e-6 * torch.eye(
            accum[i].shape[0], device=accum[i].device, dtype=accum[i].dtype
        )
        ev, evc = torch.linalg.eigh(A)
        idx = torch.argsort(ev, descending=True)[:keep_dim]
        projectors.append(evc[:, idx])
    info = {
        "cca_keep_dim": int(keep_dim),
        "mean_spectrum": (
            torch.stack(spectra).mean(dim=0).tolist() if len(spectra) else None
        ),
    }
    return projectors, info


def _rbf_kernel(
    x: torch.Tensor, gamma: Optional[float] = None
) -> torch.Tensor:
    B = x.shape[0]
    x_norm = (x * x).sum(1).view(-1, 1)
    dist = x_norm + x_norm.T - 2.0 * (x @ x.T)
    if gamma is None:
        vals = dist.detach()
        median = torch.median(
            vals[~torch.eye(B, dtype=torch.bool, device=x.device)]
        )
        if median <= 0:
            median = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        gamma = 1.0 / (2.0 * median)
    K = torch.exp(-gamma * dist)
    H = torch.eye(B, device=x.device, dtype=x.dtype) - (1.0 / B) * torch.ones(
        B, B, device=x.device, dtype=x.dtype
    )
    return H @ K @ H


@torch.no_grad()
def _hsic_unbiased(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    K_off = K[mask].view(B, B - 1)
    L_off = L[mask].view(B, B - 1)
    term1 = (K_off * L_off).sum() / (B * (B - 3))
    K1 = K.sum(dim=1) - torch.diagonal(K)
    L1 = L.sum(dim=1) - torch.diagonal(L)
    term2 = (K1 * L1).sum() / (B * (B - 3) * (B - 1))
    return term1 - term2


@torch.no_grad()
def _screen_hsic(
    feats: List[torch.Tensor],
    keep_frac: float,
    prefilter_frac: float = 0.5,
) -> Tuple[List[torch.Tensor], Dict]:
    n = len(feats)
    B, D = feats[0].shape
    r = max(1, int(round(D * keep_frac)))
    k_pref = max(1, int(round(D * prefilter_frac)))
    Z = [
        (Fv - Fv.mean(dim=0, keepdim=True)) / (Fv.std(dim=0, keepdim=True) + 1e-6)
        for Fv in feats
    ]
    pearson_scores = [
        torch.zeros(D, device=feats[0].device, dtype=feats[0].dtype) for _ in range(n)
    ]
    for v in range(n):
        others = [Z[u] for u in range(n) if u != v]
        Zcat = torch.cat(others, dim=1) if others else None
        if Zcat is None or Zcat.shape[1] == 0:
            continue
        zmean = Zcat.mean(dim=1, keepdim=True)
        a = Z[v]
        num = (a * zmean).sum(dim=0)
        den = a.pow(2).sum(dim=0).sqrt() * (
            zmean.pow(2).sum(dim=0).sqrt().squeeze(0) + 1e-8
        )
        pearson_scores[v] = (num / (den + 1e-8)).abs()

    hsic_scores = [
        torch.zeros(D, device=feats[0].device, dtype=feats[0].dtype) for _ in range(n)
    ]
    for v in range(n):
        top_idx = torch.topk(pearson_scores[v], k=k_pref, largest=True).indices
        others = [Z[u] for u in range(n) if u != v]
        Zcat = torch.cat(others, dim=1) if others else None
        if Zcat is None or Zcat.shape[1] == 0:
            continue
        y = Zcat.mean(dim=1, keepdim=True)
        for d in top_idx.tolist():
            x = Z[v][:, d : d + 1]
            hsic_scores[v][d] = _hsic_unbiased(x, y)

    masks, kept_counts = [], []
    for v in range(n):
        idx = torch.topk(hsic_scores[v], k=r, largest=True).indices
        mask = torch.zeros(D, dtype=torch.bool, device=feats[0].device)
        mask[idx] = True
        masks.append(mask)
        kept_counts.append(int(mask.sum().item()))
    info = {"keep_dim": r, "prefilter_dim": k_pref, "kept_per_view": kept_counts}
    return masks, info


def _update_screen(
    feats: List[torch.Tensor],
    state: Optional[ScreenState],
    method: Method = "none",
    keep_frac: float = 0.5,
    ridge: float = 1e-3,
    refresh: bool = False,
    prefilter_frac: float = 0.5,
) -> ScreenState:
    if method == "none":
        return ScreenState(method="none")
    assert 0.0 < keep_frac <= 1.0
    B, D = feats[0].shape
    device, dtype = feats[0].device, feats[0].dtype
    n_views = len(feats)
    r = max(1, int(round(D * keep_frac)))
    if state is None or (
        state.method != method
        or state.proj_dim != D
        or state.n_views != n_views
        or state.keep_dim != r
    ):
        state = ScreenState(
            method=method,
            proj_dim=D,
            keep_dim=r,
            n_views=n_views,
            device=device,
            dtype=dtype,
            projectors=None,
            masks=None,
            meta={},
        )
    if not refresh:
        return state
    if method == "cca":
        projectors, info = _screen_cca(feats, keep_dim=r, ridge=ridge)
        state.projectors = [P.to(device=device, dtype=dtype) for P in projectors]
        state.masks = None
        state.meta = {"cca_info": info, "keep_dim": r}
    elif method == "hsic":
        masks, info = _screen_hsic(feats, keep_frac=keep_frac, prefilter_frac=prefilter_frac)
        state.masks = [m.to(device=device, dtype=dtype) for m in masks]
        state.projectors = None
        state.meta = {"hsic_info": info, "keep_dim": r}
    return state


@torch.no_grad()
def _apply_screen(
    feats: List[torch.Tensor], state: Optional[ScreenState]
) -> List[torch.Tensor]:
    """Apply learned screening transform, or identity if not yet computed."""
    if state is None or state.method == "none":
        return feats
    if state.method == "cca":
        if state.projectors is None:
            return feats
        return [f @ P for f, P in zip(feats, state.projectors)]
    if state.method == "hsic":
        if state.masks is None:
            return feats
        return [f[:, m] for f, m in zip(feats, state.masks)]
    return feats


# ---------------------------------------------------------------------------
# Projector MLP
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Lightweight 2-layer MLP projection head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D]
        return self.net(x)


# ---------------------------------------------------------------------------
# Latent flattening helper (shared 2D / 3D)
# ---------------------------------------------------------------------------

def flatten_latents(z, target_pool_size: int = 2) -> torch.Tensor:
    """
    LAMNr strategy: extract only the deepest latent level and adaptive-pool
    it to a fixed size before feeding the Projector MLP.

    Supports 2D (N, C, H, W) and 3D (N, C, D, H, W) tensors.
    """
    zs = z if isinstance(z, (list, tuple)) else [z]
    deepest_z = zs[-1]

    if deepest_z.ndim == 5:
        z_pooled = F.adaptive_avg_pool3d(
            deepest_z,
            (target_pool_size, target_pool_size, target_pool_size),
        )
        return z_pooled.flatten(1)
    elif deepest_z.ndim == 4:
        z_pooled = F.adaptive_avg_pool2d(
            deepest_z, (target_pool_size, target_pool_size)
        )
        return z_pooled.flatten(1)
    else:
        return deepest_z.flatten(1)


# ---------------------------------------------------------------------------
# LatentAlignmentLossManager — the Factory
# ---------------------------------------------------------------------------

class LatentAlignmentLossManager:
    """
    Owns **all** latent-alignment loss computation for the LAMNr Glow trainers.

    The training loop only needs to call::

        loss_total, L_align, w_nll, w_align = manager.compute(
            lat_flat   = [...],   # per-view flattened latents (no grad needed here)
            L_nll      = ...,     # NLL loss tensor (with grad)
            it         = it,      # current global iteration
            s_nll      = ...,     # Kendall log-var parameter (or None)
            s_align    = ...,     # Kendall log-var parameter (or None)
        )
        loss_total.backward()

    Supported alignment losses
    --------------------------
    vicreg, barlow, infonce, hsic, pearson, mse, none

    Supported screening methods
    ---------------------------
    cca, hsic, none
    """

    def __init__(
        self,
        args,
        projectors: Optional[nn.ModuleList],
        device: torch.device,
    ) -> None:
        self.args = args
        self.projectors = projectors
        self.device = device
        self._screen_state: Optional[ScreenState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        lat_flat: List[torch.Tensor],
        L_nll: torch.Tensor,
        it: int,
        s_nll: Optional[nn.Parameter],
        s_align: Optional[nn.Parameter],
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Returns
        -------
        loss_total : torch.Tensor  — backpropagatable combined loss
        L_align    : torch.Tensor  — alignment term only (for logging)
        w_nll      : float         — effective NLL weight (for logging)
        w_align    : float         — effective alignment weight (for logging)
        """
        args = self.args
        L_align = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if args.align != "none" and it >= args.align_warmup and self.projectors is not None:
            # 1. Project latents
            feats = [self.projectors[i](lat_flat[i]) for i in range(len(lat_flat))]
            feats = [f.float() for f in feats]

            # 2. Optional shared-subspace screening
            feats_screened = self._maybe_screen(feats, it)

            # 3. Dispatch to alignment loss
            L_align = self._compute_raw_align(feats_screened)

        # 4. Combine with NLL
        loss_total, w_nll, w_align = self._combine(L_nll, L_align, s_nll, s_align, it)
        return loss_total, L_align, w_nll, w_align

    @property
    def screen_state(self) -> Optional[ScreenState]:
        """Expose internal screening state (for checkpointing if needed)."""
        return self._screen_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_raw_align(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Dispatch to the correct alignment loss function."""
        import antstorch
        args = self.args
        dev = self.device

        if args.align == "vicreg":
            L_inv_cov = antstorch.vicreg_multi(
                feats,
                w_inv=float(args.vicreg_inv),
                w_var=0.0,
                w_cov=float(args.vicreg_cov),
                gamma=1.0,
            )
            L_var = torch.tensor(0.0, device=dev)
            for vi, feat in enumerate(feats):
                w_var_v = (
                    args.vicreg_var[vi]
                    if vi < len(args.vicreg_var)
                    else args.vicreg_var[0]
                )
                gamma_v = (
                    args.vicreg_gamma[vi]
                    if vi < len(args.vicreg_gamma)
                    else args.vicreg_gamma[0]
                )
                std = torch.sqrt(feat.var(dim=0) + 1e-4)
                L_var = L_var + w_var_v * torch.mean(F.relu(gamma_v - std))
            return L_inv_cov + L_var

        elif args.align == "barlow":
            return antstorch.barlow_twins_multi(feats, lam=float(args.barlow_lambda))

        elif args.align == "infonce":
            return antstorch.info_nce_multi(feats, T=float(args.temperature))

        elif args.align == "hsic":
            return antstorch.hsic_multi(feats, sigma=float(args.hsic_sigma))

        elif args.align == "pearson":
            return antstorch.pearson_multi(feats)

        elif args.align == "mse":
            return antstorch.lpnorm_multi(feats, p=2.0)

        return torch.tensor(0.0, device=self.device)

    def _maybe_screen(
        self, feats: List[torch.Tensor], it: int
    ) -> List[torch.Tensor]:
        """Run CCA/HSIC screening if configured, or return feats unchanged."""
        args = self.args
        if args.screen == "none" or it < args.screen_warmup:
            return feats

        do_refresh: bool
        if self._screen_state is None:
            do_refresh = True
        else:
            do_refresh = (
                args.screen_refresh > 0
                and (it - args.screen_warmup) % args.screen_refresh == 0
            )

        self._screen_state = _update_screen(
            feats,
            state=self._screen_state,
            method=args.screen,
            keep_frac=args.screen_frac,
            ridge=args.cca_ridge,
            prefilter_frac=args.prefilter_frac,
            refresh=do_refresh,
        )
        return _apply_screen(feats, self._screen_state)

    def _combine(
        self,
        L_nll: torch.Tensor,
        L_align: torch.Tensor,
        s_nll: Optional[nn.Parameter],
        s_align: Optional[nn.Parameter],
        it: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """Combine NLL and alignment losses using fixed or Kendall weighting."""
        args = self.args

        if args.weighting == "fixed" or args.align == "none":
            w_align = float(args.align_weight) if args.align != "none" else 0.0
            loss_total = L_nll + (w_align * L_align if args.align != "none" else 0.0)
            return loss_total, 1.0, w_align

        # Kendall & Gal uncertainty weighting
        s_nll_eff   = torch.clamp(s_nll,   -5.0, 5.0)
        s_align_eff = torch.clamp(s_align, -5.0, 5.0)
        loss_total  = torch.exp(-s_nll_eff)   * L_nll   + s_nll_eff
        loss_total  = loss_total + torch.exp(-s_align_eff) * L_align + s_align_eff
        w_nll   = float(torch.exp(-s_nll_eff).detach().cpu().item())
        w_align = float(torch.exp(-s_align_eff).detach().cpu().item())
        return loss_total, w_nll, w_align
