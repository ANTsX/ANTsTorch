# normalizing_simr_flows_whitener.py
# Full-featured trainer: dataset-owned normalization/jitter + alignment penalties,
# early stopping, checkpointing/resume, scale regularizer, EMA tradeoff, and
# robust inverse() handling. Exports dataset_normalizers used by the apply step.
#
# Drop-in replacement for antstorch.deep_simr.normalizing_simr_flows_whitener

from __future__ import annotations

import math, os, json
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import normflows as nf

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

def normalizing_simr_flows_whitener(
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
    tradeoff_mode: str = "ema",
    target_ratio: float = 9.0,
    lambda_penalty: float = 1.0,
    ema_beta: float = 0.98,

    penalty_type: str = "barlow_twins_align",
    bt_lambda_diag: float = 1.0,
    bt_lambda_offdiag: float = 5e-3,
    bt_eps: float = 1e-6,
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
    Train one normalizing-flow model per view with dataset-owned preprocessing.

    Summary
    -------
    - The dataset handles normalization (None | "0mean" | "01"), optional Gaussian
      alpha-jitter (in "raw" or "normalized" space), simple imputation, and basic
      non-finite cleanup.
    - Each view gets its own flow; base distributions: GaussianPCA (latent dim k)
      or DiagGaussian.
    - Optional cross-view alignment penalty (Barlow Twins-like or correlation)
      with EMA-scaled tradeoff.

    Returns
    -------
    dict
      - "models": List[nn.Module]
      - "metrics": {"best_step","best_metric","best_metric_name"}
      - "dataset_normalizers": per-view stats for apply()
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

    if save_checkpoint_dir is not None:
        os.makedirs(save_checkpoint_dir, exist_ok=True)
    ckpt_every = checkpoint_interval if (checkpoint_interval is not None) else val_interval

    # -------------------------------
    # Training loop
    # -------------------------------
    for step in range(start_step, max_iter):
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
            penalty_active = (len(models) >= 2)
            pen = torch.tensor(0.0, device=H[0].device)
            if penalty_active:
                if penalty_type == "barlow_twins_align":
                    for i in range(len(models)):
                        for j in range(i + 1, len(models)):
                            pen = pen + _barlow_twins_align_square(H[i], H[j], bt_lambda_diag, bt_lambda_offdiag, bt_eps)
                elif penalty_type in ("decorrelate", "correlate"):
                    want = "minimize" if penalty_type == "decorrelate" else "maximize"
                    for i in range(len(models)):
                        for j in range(i + 1, len(models)):
                            pen = pen + _corr_square(H[i], H[j], want=want)
                else:
                    raise ValueError(f"Unknown penalty_type: {penalty_type}")

            # Tradeoff scaling
            warm = min(1.0, (step + 1) / max(1, penalty_warmup_iters))
            if penalty_active:
                if tradeoff_mode == "ema":
                    ema_nll = nll_sum.detach() if ema_nll is None else (ema_beta * ema_nll + (1 - ema_beta) * nll_sum.detach())
                    ema_pen = pen.detach() if ema_pen is None else (ema_beta * ema_pen + (1 - ema_beta) * pen.detach())
                    pen_scale = (ema_nll / (ema_pen + 1e-8)) / max(1e-8, float(target_ratio))
                    lam_eff = lambda_penalty * pen_scale
                elif tradeoff_mode == "uncertainty":
                    lam_eff = lambda_penalty * (nll_sum.detach() / (pen.detach() + 1e-8))
                elif tradeoff_mode == "fixed":
                    lam_eff = lambda_penalty
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
