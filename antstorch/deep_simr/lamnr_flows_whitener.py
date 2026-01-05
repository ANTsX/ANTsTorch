from __future__ import annotations

import math, os, json
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import normflows as nf

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    tqdm = None

from . import latent_alignment as la


# Keep package-relative imports to match ANTsTorch layout
from ..architectures.create_normalizing_flow_model import (
    create_real_nvp_normalizing_flow_model as create_rnvp
)
from ..utilities.dataframe_dataset import MultiViewDataFrameDataset
from torch.utils.data import DataLoader


# -------------------------------
# Helpers
# -------------------------------

class _AlphaSchedule:
    def __init__(self, start: float, end: float, total_steps: int, mode: str = "cosine"):
        self.start = float(start); self.end = float(end)
        self.total = max(1, int(total_steps)); self.mode = mode

    def value(self, step: int) -> float:
        t = min(max(step, 0), self.total)
        if self.mode == "linear":
            return self.start + (self.end - self.start) * (t / self.total)
        if self.mode == "exp":
            a, b = max(self.start, 1e-12), max(self.end, 1e-12)
            logv = math.log(a) + (math.log(b) - math.log(a)) * (t / self.total)
            return float(math.exp(logv))
        # cosine (default)
        cosw = 0.5 * (1 - math.cos(math.pi * t / self.total))
        return self.start + (self.end - self.start) * cosw


def _ensure_device(cuda_device: Optional[str]) -> torch.device:
    if cuda_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested '{cuda_device}' but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cuda_device)


def _bits_per_dim(nll: torch.Tensor, D_total: int) -> float:
    return float(nll.detach().cpu().item()) / (D_total * math.log(2))


def _inverse_with_guard(model, xb):
    """
    Handle inverse() across wrappers:
      - returns Tensor -> (z, zeros)
      - returns (z, log_det) -> as-is
      - returns (z, log_det, *rest) -> take first two
      - if model has .flow/.flows with .inverse, try those as fallback
    """
    last = None
    for attr_chain in (("inverse",), ("flow","inverse"), ("flows","inverse")):
        try:
            tgt = model
            for attr in attr_chain:
                tgt = getattr(tgt, attr)
            out = tgt(xb)
            if isinstance(out, (tuple, list)):
                if len(out) >= 2:
                    z, log_det = out[0], out[1]
                else:
                    z = out[0]
                    log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
            else:
                z = out
                log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
            return z, log_det
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to run inverse on {type(model)} with input {tuple(xb.shape)}") from last


def _extract_whitened_from_z(model, z: torch.Tensor) -> torch.Tensor:
    """Return L-dim whitened (GaussianPCA) or standardized (DiagGaussian) coords."""
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        W, loc = q0.W, q0.loc
        if isinstance(loc, torch.Tensor) and loc.dim() == 2 and loc.shape[0] == 1:
            loc = loc.squeeze(0)
        return torch.matmul(z - loc, W.T).to(z.dtype)
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if isinstance(loc, torch.Tensor) and isinstance(scale, torch.Tensor):
        if loc.dim() == 2 and loc.shape[0] == 1: loc = loc.squeeze(0)
        if scale.dim() == 2 and scale.shape[0] == 1: scale = scale.squeeze(0)
        return ((z - loc) / (scale + 1e-12)).to(z.dtype)
    return z.to(z.dtype)


def _barlow_twins_align_square(h1: torch.Tensor, h2: torch.Tensor, lam_diag: float, lam_off: float, eps: float = 1e-5) -> torch.Tensor:
    """Barlow Twins alignment (square loss) on standardized features."""
    h1 = (h1 - h1.mean(0)) / (h1.std(0, unbiased=False) + eps)
    h2 = (h2 - h2.mean(0)) / (h2.std(0, unbiased=False) + eps)
    N = h1.size(0)
    C = (h1.T @ h2) / max(1, N)
    I = torch.eye(C.size(0), device=C.device, dtype=C.dtype)
    diag = ((C - I).diag() ** 2).sum()
    off = (C - torch.diag(torch.diag(C))).pow(2).sum()
    return lam_diag * diag + lam_off * off


def _corr_square(h1: torch.Tensor, h2: torch.Tensor, want: str = "maximize", eps: float = 1e-6) -> torch.Tensor:
    """Scalarized correlation objective."""
    h1 = (h1 - h1.mean(0)) / (h1.std(0, unbiased=False) + eps)
    h2 = (h2 - h2.mean(0)) / (h2.std(0, unbiased=False) + eps)
    C = torch.mean(h1 * h2, dim=0)
    m = C.abs().mean()
    if want == "maximize":
        return 1.0 - m
    elif want == "minimize":
        return m
    else:
        raise ValueError("want must be 'maximize' or 'minimize'")


def _save_checkpoint(path: str, step: int, models):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"step": int(step), "models": [m.state_dict() for m in models]}
    torch.save(payload, path)


def _try_resume(path: Optional[str], device: torch.device):
    if path is None:
        return None
    if not os.path.isfile(path):
        print(f"[warn] resume_checkpoint not found: {path}")
        return None
    try:
        ckpt = torch.load(path, map_location=device)
        return ckpt
    except Exception as e:
        print(f"[warn] failed to load resume checkpoint: {e}")
        return None


def _build_base_distribution(
    D: int,
    base_distribution: str = "GaussianPCA",
    pca_latent_dimension: Optional[int] = None,
    min_log: float = -5.0,
    max_log: float = 5.0,
    sigma: float = 0.1,
):
    base = base_distribution.lower()
    if base in ("gaussianpca", "pca"):
        if pca_latent_dimension is None:
            raise ValueError("pca_latent_dimension must be provided when using GaussianPCA.")
        return nf.distributions.GaussianPCA(D, latent_dim=int(pca_latent_dimension), sigma=sigma)
    elif base in ("diag", "diaggaussian", "gauss_diag", "diaggaussian"):
        try:
            return nf.distributions.DiagGaussian(D, trainable=True, min_log=min_log, max_log=max_log)
        except TypeError:
            return nf.distributions.DiagGaussian(D, trainable=True)
    else:
        raise ValueError(f"Unknown base distribution: {base_distribution}")


def _collect_dataset_normalizers(ds: MultiViewDataFrameDataset) -> List[Dict[str, Any]]:
    """Export per-view normalization stats from the dataset in a portable schema."""
    out: List[Dict[str, Any]] = []
    for v in ds.view_names:
        mode = ds._norm_mode[v]  # internal but consistent in our implementation
        st = ds._state[v]
        if mode is None:
            out.append({"mode": None})
        elif mode == "0mean":
            out.append({
                "mode": "0mean",
                "mean": st.mean.astype(float).tolist(),
                "std":  st.std.astype(float).tolist(),
            })
        elif mode == "01":
            out.append({
                "mode": "01",
                "vmin": st.vmin.astype(float).tolist(),
                "vrng": st.eps_rng.astype(float).tolist(),
            })
        else:
            raise ValueError(f"Unsupported normalization mode '{mode}' for export.")
    return out


def _dump_dataset_normalizers_json(path: str, norm_list: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(norm_list, f, indent=2)


# -------------------------------
# Main trainer
# -------------------------------

def lamnr_flows_whitener(
    views: List[pd.DataFrame],
    *,

    # Splits / batching
    val_fraction: float = 0.2,
    batch_size: int = 256,
    val_batch_size: int = 2048,
    seed: int = 0,
    cuda_device: Optional[str] = None,

    # Dataset-owned normalization & jitter
    normalization: Optional[Union[str, Dict[str, Optional[str]]]] = "0mean",
    add_noise_in: str = "normalized",
    impute: str = "mean",
    nonfinite_clean: Optional[str] = None,
    nonfinite_scope: str = "numeric_only",
    max_row_na_frac: float = 0.0,

    # Flow config
    K: int = 64,
    leaky_relu_negative_slope: float = 0.0,

    # Base distribution
    base_distribution: str = "GaussianPCA",
    pca_latent_dimension: Optional[int] = 4,
    base_min_log: float = -5.0,
    base_max_log: float = 5.0,
    base_sigma: float = 0.1,

    # Flow stability
    scale_cap: float = 3.0,
    spectral_norm_scales: bool = False,
    additive_first_n: int = 0,
    actnorm_every: int = 1,
    mask_mode: str = "alternating",

    # Optimization
    lr: float = 2e-4,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps_opt: float = 1e-8,
    weight_decay: float = 0.0,
    max_iter: int = 5000,
    grad_clip: float = 5.0,

    # Jitter schedule
    jitter_alpha: float = 0.0,
    jitter_alpha_end: float = 0.0,
    jitter_alpha_mode: str = "cosine",
    jitter_alpha_total_steps: Optional[int] = None,  # None -> default to max_iter

    # Tradeoff / penalties
    tradeoff_mode: str = "uncertainty",
    target_ratio: float = 9.0,
    lambda_penalty: float = 1.0,
    ema_beta: float = 0.98,

    penalty_type: str = "barlow_twins_align",
    bt_lambda_diag: float = 1.0,
    bt_lambda_offdiag: float = 5e-3,
    bt_eps: float = 1e-6,
    info_nce_T: float = 0.2,
    vicreg_w_inv: float = 25.0,
    vicreg_w_var: float = 25.0,
    vicreg_w_cov: float = 1.0,
    vicreg_gamma: float = 1.0,
    hsic_sigma: float = 0.0,
    penalty_warmup_iters: int = 400,

    # Validation / early stopping
    val_interval: int = 200,
    early_stop_enabled: bool = False,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_iters: int = 600,
    early_stop_beta: float = 0.98,

    # Checkpointing / resume
    save_checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    restore_best_for_final_eval: bool = True,

    # Bookkeeping
    best_selection_metric: str = "val_bpd",
    verbose: bool = False,

    # Optional regularizer
    scale_penalty_weight: float = 0.0,

    # Export stats to disk (optional)
    dataset_normalizers_dump_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a multi-view RealNVP-style normalizing-flow *whitener* with an optional PCA base
    and a cross-view alignment penalty, using a robust data pipeline built on
    `MultiViewDataFrameDataset`.

    Pipeline (high level)
    ---------------------
    1) Wrap per-view pandas DataFrames with `MultiViewDataFrameDataset`:
       - Align indices across views.
       - Handle non-finite values (drop/impute) consistently across views.
       - Optionally apply dataset-owned normalization and input jitter.
    2) Build one flow per view with the chosen base distribution:
       - "GaussianPCA": k-dim PCA Gaussian base (latent dimension = `pca_latent_dimension`).
       - "DiagGaussian": factorized full-dim Gaussian base.
    3) Train by maximizing likelihood (minimizing NLL). If multiple views are present,
       apply a cross-view alignment penalty on “whitened” latents.
    4) A controller (`tradeoff_mode`) adjusts the penalty weight λ_eff relative to NLL:
       - "uncertainty": λ_eff ∝ |NLL| / |penalty| (smoothed, sign-robust).
       - "ema":         λ_eff ∝ EMA(NLL) / EMA(penalty) (sign-robust).
       λ_eff is clamped and warmed up for stability.
    5) Validate periodically, track the best state by `best_selection_metric`,
       and optionally restore it at the end.

    Parameters
    ----------
    views : list[pd.DataFrame]
        One DataFrame per view (rows = subjects, columns = features) with a shared index.
        Categorical encoding should be handled upstream or by the dataset wrapper.

    # Splits / batching
    val_fraction : float, default 0.2
        Fraction of samples reserved for validation (random split).
    batch_size : int, default 256
        Training batch size.
    val_batch_size : int, default 2048
        Validation batch size (can be larger for throughput).
    seed : int, default 0
        RNG seed for splitting and initialization.
    cuda_device : str or None, default None
        Device spec, e.g. "cuda:0" or None for CPU/auto.

    # Dataset-owned normalization & jitter
    normalization : {"0mean","zscore", None} or dict, default "0mean"
        Per-view, dataset-owned normalization strategy. If dict, may specify per-view
        modes or None to disable for a view. Typical:
          - "0mean": subtract column means.
          - "zscore": (x - mean) / std with std floors.
          - None: no dataset normalization (use model-side stats only).
    add_noise_in : {"raw","normalized","none"}, default "normalized"
        Where to add dataset-level Gaussian noise (jitter). Only applied if the dataset
        is configured to add noise. "normalized" means after normalization; "raw" before.
    impute : {"none","zero","mean"}, default "mean"
        Per-view imputation rule applied inside the dataset for non-finite values.
    nonfinite_clean : {None,"drop","impute_zero","impute_mean"}, default None
        Cross-view cleaning policy applied **before** building tensors:
          - "drop": drop any row where any view violates finiteness (per `nonfinite_scope`).
          - "impute_zero"/"impute_mean": fill non-finites consistently across views.
          - None: disable extra cleaning (use `impute` only).
    nonfinite_scope : {"numeric_only","all_columns"}, default "numeric_only"
        Whether finiteness checks/cleaning are restricted to numeric columns.
    max_row_na_frac : float, default 0.0
        For "drop" cleaning, maximum allowed fraction of non-finite entries per row.

    # Flow config
    K : int, default 64
        Number of coupling layers per view.
    leaky_relu_negative_slope : float, default 0.0
        Negative slope for internal activations (if used).

    # Base distribution
    base_distribution : {"GaussianPCA","DiagGaussian"}, default "GaussianPCA"
        Choice of base density for each view’s flow.
    pca_latent_dimension : int or None, default 4
        PCA latent dimension k (for "GaussianPCA"). Must be ≤ feature dimension D and ≤
        effective rank of the view.
    base_min_log, base_max_log : float, defaults -5.0, 5.0
        Log-scale clamps for base parameters (numerical safety).
    base_sigma : float, default 0.1
        Base Gaussian scale (σ). Acts as a prior scale; higher values are more forgiving.

    # Flow stability
    scale_cap : float, default 3.0
        Clamp for coupling log-scale/scale outputs to prevent explosions.
    spectral_norm_scales : bool, default False
        Apply spectral normalization to scale networks.
    additive_first_n : int, default 0
        Use additive couplings (no scaling) for the first N layers as a warmup.
    actnorm_every : int, default 1
        Insert ActNorm layers at this cadence (1 = every block).
    mask_mode : str, default "alternating"
        Coupling mask strategy ("alternating", "rolling", etc.).

    # Optimization
    lr : float, default 2e-4
        AdamW learning rate.
    betas : (float, float), default (0.9, 0.98)
        AdamW β parameters.
    eps_opt : float, default 1e-8
        Optimizer epsilon.
    weight_decay : float, default 0.0
        AdamW weight decay.
    max_iter : int, default 5000
        Number of training steps.
    grad_clip : float, default 5.0
        Global gradient-norm clip.

    # Jitter schedule
    jitter_alpha : float, default 0.0
        Initial input noise std (model-side jitter; separate from dataset noise).
    jitter_alpha_end : float, default 0.0
        Final noise std.
    jitter_alpha_mode : {"cosine","linear"}, default "cosine"
        Schedule shape.
    jitter_alpha_total_steps : int or None, default None
        Number of steps across which to schedule from start→end. If None, uses `max_iter`.

    # Tradeoff / penalties
    tradeoff_mode : {"uncertainty","ema"}, default "uncertainty"
        Controller that scales the alignment penalty relative to NLL.
          • "uncertainty": λ_eff ∝ |NLL| / |penalty| (we use a small EMA of magnitudes
            for stability; λ_eff is clamped and warmed up).
          • "ema":         λ_eff ∝ EMA(NLL) / EMA(penalty) (sign-robust EMAs).
    target_ratio : float, default 9.0
        Desired NLL:penalty ratio used by the controller (larger → smaller λ_eff).
    lambda_penalty : float, default 1.0
        Global multiplier on λ_eff.
    ema_beta : float, default 0.98
        EMA smoothing factor for "ema" mode; a lighter EMA is also used inside
        "uncertainty" to reduce jitter.

    penalty_type : {"barlow_twins_align","correlate","decorrelate","pearson","barlow_twins_multi","vicreg","info_nce","hsic","none"}, default "barlow_twins_align"
        Alignment objective on whitened latents across views:
          • "barlow_twins_align":   Barlow Twins-style cross-correlation to identity (diag≈1, off-diag≈0).
          • "correlate":            maximize correlation (scalarized Pearson) between views.
          • "decorrelate":          minimize correlation (ablation).
          • "pearson":              use latent_alignment.pearson_multi to maximize diagonal Pearson corr.
          • "barlow_twins_multi":   use latent_alignment.barlow_twins_multi on whitened latents.
          • "vicreg":               use latent_alignment.vicreg_multi (variance–invariance–covariance).
          • "info_nce":             use latent_alignment.info_nce_multi (SimCLR-style NT-Xent).
          • "hsic":                 use latent_alignment.hsic_multi (RBF HSIC; maximizes dependence).
          • "none":                 disable alignment penalty entirely.
    bt_lambda_diag : float, default 1.0
        Diagonal weight for Barlow-Twins-style alignment.
    bt_lambda_offdiag : float, default 5e-3
        Off-diagonal weight.
    bt_eps : float, default 1e-6
        Numerical epsilon in correlation/whitening ops.
    info_nce_T : float, default 0.2
        Temperature for InfoNCE / NT-Xent when ``penalty_type="info_nce"``.
    vicreg_w_inv : float, default 25.0
        Invariance (MSE) weight for VICReg when ``penalty_type="vicreg"``.
    vicreg_w_var : float, default 25.0
        Variance-floor weight for VICReg when ``penalty_type="vicreg"``.
    vicreg_w_cov : float, default 1.0
        Covariance off-diagonal weight for VICReg when ``penalty_type="vicreg"``.
    vicreg_gamma : float, default 1.0
        Target standard deviation (gamma) for VICReg variance term.
    hsic_sigma : float, default 0.0
        RBF kernel bandwidth for HSIC when ``penalty_type="hsic"``; 0.0 = median heuristic per batch.
    penalty_warmup_iters : int, default 400
        Warmup period: linearly introduce the alignment term and cap λ_eff early.

    # Validation / early stopping
    val_interval : int, default 200
        Validate (and optionally checkpoint) every N steps.
    early_stop_enabled : bool, default False
        Enable early stopping on `best_selection_metric`.
    early_stop_patience : int, default 300
        Steps without improvement after `early_stop_min_iters` before stopping.
    early_stop_min_delta : float, default 1e-4
        Required improvement to reset patience.
    early_stop_min_iters : int, default 600
        Do not early-stop before this many steps.
    early_stop_beta : float, default 0.98
        Smoothing factor for the `smooth_total` metric.

    # Checkpointing / resume
    save_checkpoint_dir : str or None, default None
        Directory to write checkpoints (per `checkpoint_interval` and "latest.pth").
    checkpoint_interval : int or None, default None
        If None, checkpoint every `val_interval`; otherwise, every `checkpoint_interval`.
    resume_checkpoint : str or None, default None
        Path to a checkpoint to resume (models, standardizers, step).
    restore_best_for_final_eval : bool, default True
        Restore the best weights (by `best_selection_metric`) before returning.

    # Bookkeeping
    best_selection_metric : {"val_bpd","smooth_total"}, default "val_bpd"
        Which metric defines “best” for model selection.
    verbose : bool, default False
        Print detailed training logs.

    # Optional regularizer
    scale_penalty_weight : float, default 0.0
        L2 penalty on scale/log-scale parameters (stability regularization).

    # Export stats to disk (optional)
    dataset_normalizers_dump_path : str or None, default None
        If set, write dataset-owned normalizer statistics (e.g., means/stds per view/column)
        to this path for reproducibility/auditing.

    Returns
    -------
    dict
        {
          "models": list[torch.nn.Module],
              # One trained flow per view.
          "standardizers": list[{"mean": (1, D_v), "std": (1, D_v)}],
              # Per-view train-split statistics used for model-side standardization
              # (if applicable to your implementation).
          "metrics": {
              "best_step": int,
              "best_metric": float,
              "best_metric_name": str,
          }
        }

    Notes
    -----
    • Negative bpd is expected on continuous data; use it *relatively* for model selection.
    • For "GaussianPCA", ensure `pca_latent_dimension ≤ D` and ≤ effective rank; drop or
      repair degenerate columns upstream (dataset can help via `nonfinite_*`).
    • λ_eff is kept non-negative and clamped to a safe range; during warmup the alignment
      term is introduced gently to avoid destabilizing early optimization.
    • If you enable dataset normalization **and** perform model-side standardization, make
      sure you are not double-normalizing unless that’s intentional—prefer one place.
    • Repro tip: persist both dataset normalizers (`dataset_normalizers_dump_path`) and the
      model’s per-view standardizers (returned) alongside checkpoints.
    """

    # Device / seeds
    torch.manual_seed(seed); np.random.seed(seed)
    device = _ensure_device(cuda_device)

    base_lower = base_distribution.lower()
    if base_lower in ("gaussianpca", "pca") and pca_latent_dimension is None:
        raise ValueError("pca_latent_dimension must be provided when base_distribution='GaussianPCA'.")

    # Align rows, split
    view_names = [f"v{i}" for i in range(len(views))]
    view_map = {vn: df for vn, df in zip(view_names, views)}
    ds_probe = MultiViewDataFrameDataset(
        views=view_map, normalization=None, alpha=0.0, add_noise_in='none',
        impute="mean", nonfinite_clean=None, concat_views=False, dtype=torch.float32,
    )
    N = len(ds_probe)
    perm = torch.randperm(N)
    n_val = int(N * float(val_fraction))
    val_idx_t   = perm[:n_val]
    train_idx_t = perm[n_val:]
    train_ids = ds_probe.index[train_idx_t.numpy()]
    val_ids   = ds_probe.index[val_idx_t.numpy()]

    views_train = {vn: df.loc[train_ids] for vn, df in view_map.items()}
    views_val   = {vn: df.loc[val_ids]   for vn, df in view_map.items()}

    ds_train = MultiViewDataFrameDataset(
        views=views_train, normalization=normalization, alpha=0.0, add_noise_in=add_noise_in,
        impute=impute, nonfinite_clean=nonfinite_clean, nonfinite_scope=nonfinite_scope,
        max_row_na_frac=max_row_na_frac, concat_views=False, dtype=torch.float32,
    )
    ds_val = MultiViewDataFrameDataset(
        views=views_val, normalization=normalization, alpha=0.0, add_noise_in="none",
        impute=impute, nonfinite_clean=nonfinite_clean, nonfinite_scope=nonfinite_scope,
        max_row_na_frac=max_row_na_frac, concat_views=False, dtype=torch.float32,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False,
                              worker_init_fn=MultiViewDataFrameDataset.worker_init_fn)
    val_loader   = DataLoader(ds_val,   batch_size=val_batch_size, shuffle=False, drop_last=False)

    first_batch = next(iter(train_loader))
    Ds = [int(first_batch['views'][vn].shape[1]) for vn in view_names]

    # Build models
    models = []; opt_params = []
    for D in Ds:
        q0 = _build_base_distribution(D, base_distribution, pca_latent_dimension, base_min_log, base_max_log, base_sigma)
        model = create_rnvp(
            latent_size=D, K=K, q0=q0,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            scale_cap=scale_cap, spectral_norm_scales=spectral_norm_scales,
            additive_first_n=additive_first_n, actnorm_every=actnorm_every, mask_mode=mask_mode,
        ).to(device)
        if base_lower in ("gaussianpca", "pca"):
            if not (hasattr(model.q0, "W") and hasattr(model.q0, "loc")):
                raise RuntimeError("GaussianPCA base must expose 'W' and 'loc'.")
        models.append(model); opt_params += list(model.parameters())

    # For DiagGaussian penalties we need equal dims
    if base_lower.startswith("diag"):
        if not all(D == Ds[0] for D in Ds):
            raise RuntimeError(f"DiagGaussian cross-view penalty requires equal dims across views, got {Ds}.")

    # Resume
    start_step = 0
    ckpt = _try_resume(resume_checkpoint, device)
    if ckpt is not None:
        try:
            for m, sd in zip(models, ckpt["models"]):
                m.load_state_dict(sd)
            start_step = int(ckpt.get("step", 0))
            if verbose:
                print(f"[info] resumed from '{resume_checkpoint}' @ step {start_step}")
        except Exception as e:
            print(f"[warn] resume failed to load state dicts: {e}")

    opt = optim.AdamW(opt_params, lr=lr, betas=betas, eps=eps_opt, weight_decay=weight_decay)

    # Alpha schedule horizon
    total_steps_for_sched = jitter_alpha_total_steps if jitter_alpha_total_steps is not None else max_iter
    jitter_sched = _AlphaSchedule(jitter_alpha, jitter_alpha_end, total_steps_for_sched, jitter_alpha_mode)

    ema_nll = None
    ema_pen = None

    best_metric = float('inf')
    best_state = None
    best_step = -1
    patience_counter = 0
    smooth_total = None

    smooth_u_n = None   # EMA of |nll|  (for 'uncertainty' mode)
    smooth_u_p = None   # EMA of |pen|  (for 'uncertainty' mode)

    if save_checkpoint_dir is not None:
        os.makedirs(save_checkpoint_dir, exist_ok=True)
    ckpt_every = checkpoint_interval if (checkpoint_interval is not None) else val_interval

    # -------------------------------
    # Training loop
    # -------------------------------
    if verbose and tqdm is not None:
        iter_range = tqdm(
            range(start_step, max_iter),
            initial=start_step,
            total=max_iter,
            desc="training",
            dynamic_ncols=True,
        )
    else:
        iter_range = range(start_step, max_iter)

    for step in iter_range:
        for m in models: m.train()

        current_alpha = float(jitter_sched.value(step))
        ds_train.set_alpha(current_alpha, add_noise_in=add_noise_in)

        # one pass over loader
        for batch in train_loader:
            batch_X = [batch['views'][vn].to(device) for vn in view_names]

            nll_sum = 0.0
            H = []
            for m, xb in zip(models, batch_X):
                z, log_det = _inverse_with_guard(m, xb)
                nll = -(m.q0.log_prob(z) + log_det).mean()
                nll_sum = nll_sum + nll
                h = _extract_whitened_from_z(m, z)
                H.append(h)

            # --- Cross-view penalty ---
            penalty_active = (len(models) >= 2 and penalty_type != "none")
            pen = torch.tensor(0.0, device=H[0].device)
            if penalty_active:
                if penalty_type == "barlow_twins_align":
                    for i in range(len(models)):
                        for j in range(i + 1, len(models)):
                            pen = pen + _barlow_twins_align_square(
                                H[i], H[j], bt_lambda_diag, bt_lambda_offdiag, bt_eps
                            )
                elif penalty_type in ("decorrelate", "correlate"):
                    want = "minimize" if penalty_type == "decorrelate" else "maximize"
                    for i in range(len(models)):
                        for j in range(i + 1, len(models)):
                            pen = pen + _corr_square(H[i], H[j], want=want)
                elif penalty_type == "pearson":
                    pen = pen + la.pearson_multi(H)
                elif penalty_type == "barlow_twins_multi":
                    pen = pen + la.barlow_twins_multi(H, lam=float(bt_lambda_offdiag))
                elif penalty_type == "info_nce":
                    pen = pen + la.info_nce_multi(H, T=float(info_nce_T))
                elif penalty_type == "vicreg":
                    pen = pen + la.vicreg_multi(
                        H,
                        w_inv=float(vicreg_w_inv),
                        w_var=float(vicreg_w_var),
                        w_cov=float(vicreg_w_cov),
                        gamma=float(vicreg_gamma),
                    )
                elif penalty_type == "hsic":
                    pen = pen + la.hsic_multi(H, sigma=float(hsic_sigma))
                else:
                    raise ValueError(f"Unknown penalty_type: {penalty_type}")

            # Tradeoff scaling
            warm = min(1.0, (step + 1) / max(1, penalty_warmup_iters))
            if penalty_active:
                if tradeoff_mode == "ema":
                    ema_nll = nll_sum.detach() if ema_nll is None else (ema_beta * ema_nll + (1 - ema_beta) * nll_sum.detach())
                    ema_pen = pen.detach() if ema_pen is None else (ema_beta * ema_pen + (1 - ema_beta) * pen.detach())
                    # Use magnitudes so weight stays positive even if nll < 0
                    ema_n_mag = ema_nll.abs().clamp(min=1e-4)
                    ema_p_mag = ema_pen.clamp(min=1e-4)

                    # same ratio, but stable in sign and scale; log version avoids overflow/underflow
                    pen_scale = torch.exp(torch.log(ema_n_mag) - torch.log(ema_p_mag) - math.log(max(1e-8, float(target_ratio))))
                    lam_eff = lambda_penalty * pen_scale

                    # final safety clamps (prevents collapse to zero or blowups)
                    lam_eff = torch.nan_to_num(lam_eff, nan=0.0, posinf=1e3, neginf=0.0).clamp(min=1e-3, max=1e3)
                elif tradeoff_mode == "uncertainty":
                    eps = 1e-4
                    lam_min, lam_max = 1e-3, 1e3     # expose as CLI if you like
                    beta = 0.9                       # light smoothing to avoid jitter
                    tr = max(1e-8, float(target_ratio))

                    # Magnitudes so weight stays positive even if nll < 0
                    n_mag = nll_sum.detach().abs().clamp(min=eps)
                    p_mag = pen.detach().abs().clamp(min=eps)

                    # Smooth a bit (EMA)
                    smooth_u_n = n_mag if smooth_u_n is None else beta * smooth_u_n + (1 - beta) * n_mag
                    smooth_u_p = p_mag if smooth_u_p is None else beta * smooth_u_p + (1 - beta) * p_mag

                    # Ratio in log-space for stability, then exp back
                    ratio_log = torch.log(smooth_u_n) - torch.log(smooth_u_p) - math.log(tr)
                    pen_scale = torch.exp(ratio_log)

                    lam_eff = lambda_penalty * pen_scale
                    lam_eff = torch.nan_to_num(lam_eff, nan=0.0, posinf=lam_max, neginf=0.0).clamp(lam_min, lam_max)

                    # Gentle warmup: keep it small early on so NLL can stabilize
                    if step < penalty_warmup_iters:
                        # linear ramp or cap — choose one; cap is simplest:
                        lam_eff = lam_eff.clamp(max=0.1)

                elif tradeoff_mode == "fixed":
                    lam_eff = torch.tensor(float(lambda_penalty), device=nll_sum.device)
                else:
                    raise ValueError(f"Unknown tradeoff_mode: {tradeoff_mode}")
            else:
                lam_eff = torch.tensor(0.0, device=nll_sum.device)

            lam_eff = torch.nan_to_num(lam_eff, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            total = nll_sum + warm * lam_eff * pen

            # Optional scale regularizer
            if scale_penalty_weight and scale_penalty_weight > 0:
                sp = None
                for m in models:
                    for name, p in m.named_parameters():
                        if any(k in name.lower() for k in ("scale", "log_scale", "s_log", "logs")):
                            sp = p.pow(2).sum() if sp is None else sp + p.pow(2).sum()
                if sp is not None:
                    total = total + float(scale_penalty_weight) * sp

            opt.zero_grad(set_to_none=True)
            if torch.isfinite(total):
                total.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], grad_clip)
                opt.step()

        # ---------------------------
        # Validation
        # ---------------------------
        do_val = ((step + 1) % max(1, val_interval) == 0) or (step == max_iter - 1)
        if do_val:
            for m in models: m.eval()
            with torch.no_grad():
                ds_val.set_alpha(0.0, add_noise_in="none")
                nll_val_sum = 0.0
                count = 0
                for batch in val_loader:
                    for i, vn in enumerate(view_names):
                        xb = batch['views'][vn].to(device)
                        z, log_det = _inverse_with_guard(models[i], xb)
                        nll_val_sum += (-(models[i].q0.log_prob(z) + log_det)).sum()
                    count += xb.shape[0]
                nll_val = nll_val_sum / max(1, count)
                D_total = sum(Ds)
                val_bpd = _bits_per_dim(nll_val, D_total)

                # smooth training total for stability
                smooth_total = float(total.detach().cpu()) if smooth_total is None else \
                               (early_stop_beta * float(smooth_total) + (1 - early_stop_beta) * float(total.detach().cpu()))
                metric = val_bpd if best_selection_metric == "val_bpd" else smooth_total

                # early-stop style best tracking
                improved = (metric + early_stop_min_delta < best_metric) and (step + 1) >= early_stop_min_iters
                if improved:
                    best_metric = metric
                    best_state = [m.state_dict() for m in models]
                    best_step = step + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    lam_eff_value = float(lam_eff.detach().cpu()) if penalty_active else 0.0
                    pen_val = float(pen.detach().cpu()) if penalty_active else 0.0
                    print(f"[{step+1:5d}/{max_iter}] (alpha={current_alpha:.4f}): "
                          f"train_total={float(total):.4f}, smooth_total={float(smooth_total):.4f}, "
                          f"val_bpd={val_bpd:.4f}, lam_eff={lam_eff_value:.2e}, penalty={(pen_val if penalty_active else 0.0):.4f}")

                if early_stop_enabled and (step + 1) >= early_stop_min_iters and patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"[early stop] step {step+1} best {best_selection_metric}={best_metric:.4f} @ step {best_step}")
                    break

        # ---------------------------
        # Checkpointing
        # ---------------------------
        if save_checkpoint_dir is not None:
            if ((step + 1) % max(1, (checkpoint_interval or ckpt_every)) == 0) or (step == max_iter - 1):
                ckpt_path = os.path.join(save_checkpoint_dir, f"ckpt_step{step+1}.pth")
                _save_checkpoint(ckpt_path, step + 1, models)
                latest_path = os.path.join(save_checkpoint_dir, "latest.pth")
                _save_checkpoint(latest_path, step + 1, models)

    # Restore best for final eval if requested
    if restore_best_for_final_eval and (best_state is not None):
        for m, sd in zip(models, best_state):
            m.load_state_dict(sd)

    # Export dataset normalizers (from TRAIN dataset)
    dataset_normalizers = _collect_dataset_normalizers(ds_train)
    if dataset_normalizers_dump_path is not None:
        try:
            _dump_dataset_normalizers_json(dataset_normalizers_dump_path, dataset_normalizers)
        except Exception as e:
            print(f"[warn] failed to dump dataset_normalizers to '{dataset_normalizers_dump_path}': {e}")

    metrics = {"best_step": best_step if best_state is not None else max_iter,
               "best_metric": best_metric, "best_metric_name": best_selection_metric}

    return {"models": models, "metrics": metrics, "dataset_normalizers": dataset_normalizers}
