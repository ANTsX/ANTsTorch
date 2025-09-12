
import math
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import normflows as nf

from ..architectures.create_normalizing_flow_model import (
    create_real_nvp_normalizing_flow_model as create_rnvp
)

from torch.utils.data import DataLoader, Subset
from ..utilities.dataframe_dataset import MultiViewDataFrameDataset


# -------------------------------
# Helpers
# -------------------------------

class _AlphaSchedule:
    def __init__(self, start: float, end: float, total_steps: int, mode: str = "cosine"):
        self.start = float(start)
        self.end = float(end)
        self.total = max(1, int(total_steps))
        self.mode = mode

    def value(self, step: int) -> float:
        t = min(max(step, 0), self.total)
        if self.mode == "linear":
            return self.start + (self.end - self.start) * (t / self.total)
        if self.mode == "exp":
            a, b = max(self.start, 1e-12), max(self.end, 1e-12)
            logv = math.log(a) + (math.log(b) - math.log(a)) * (t / self.total)
            return float(math.exp(logv))
        cosw = 0.5 * (1 - math.cos(math.pi * t / self.total))
        return self.start + (self.end - self.start) * cosw


def _ensure_device(cuda_device: Optional[str]) -> torch.device:
    if cuda_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA device '{cuda_device}' requested but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cuda_device)


def _standardize_fit(X: torch.Tensor):
    mean = X.mean(0, keepdim=True)
    std = X.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std


def _standardize_apply(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean) / std


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


def _batchify(N: int, batch_size: int):
    idx = torch.randperm(N)
    for i in range(0, N, batch_size):
        yield idx[i:i+batch_size]


def _barlow_twins_align_square(h1: torch.Tensor, h2: torch.Tensor, lam_diag: float, lam_off: float, eps: float = 1e-5) -> torch.Tensor:
    h1 = (h1 - h1.mean(0)) / (h1.std(0, unbiased=False) + eps)
    h2 = (h2 - h2.mean(0)) / (h2.std(0, unbiased=False) + eps)
    N = h1.size(0)
    C = (h1.T @ h2) / max(1, N)  # (k x k)
    I = torch.eye(C.size(0), device=C.device, dtype=C.dtype)
    diag = ((C - I).diag() ** 2).sum()
    off = (C - torch.diag(torch.diag(C))).pow(2).sum()
    return lam_diag * diag + lam_off * off


def _corr_square(h1: torch.Tensor, h2: torch.Tensor, want: str = "maximize", eps: float = 1e-6) -> torch.Tensor:
    h1 = (h1 - h1.mean(0)) / (h1.std(0, unbiased=False) + eps)
    h2 = (h2 - h2.mean(0)) / (h2.std(0, unbiased=False) + eps)
    C = torch.mean(h1 * h2, dim=0)  # (k,)
    m = C.abs().mean()
    if want == "maximize":
        return 1.0 - m
    elif want == "minimize":
        return m
    else:
        raise ValueError("want must be 'maximize' or 'minimize'")


def _bits_per_dim(nll: torch.Tensor, D_total: int) -> float:
    return float(nll.detach().cpu().item()) / (D_total * math.log(2))


def _save_checkpoint(path: str, step: int, models, standardizers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": int(step),
        "models": [m.state_dict() for m in models],
        "standardizers": [{"mean": s["mean"], "std": s["std"]} for s in standardizers],
    }
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


def _inverse_with_guard(model, xb):
    try:
        out = model.inverse(xb)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        elif isinstance(out, torch.Tensor):
            z = out
            log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            return z, log_det
    except Exception as e1:
        last = e1

    for attr in ("flow", "flows"):
        if hasattr(model, attr):
            inv = getattr(getattr(model, attr), "inverse", None)
            if callable(inv):
                try:
                    out = inv(xb)
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        return out[0], out[1]
                    elif isinstance(out, torch.Tensor):
                        z = out
                        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
                        return z, log_det
                except Exception as e2:
                    last = e2

    raise RuntimeError(
        f"Failed to run inverse on model {type(model)} with input shape {tuple(xb.shape)} "
        f"(dtype={xb.dtype})."
    ) from last


def _extract_whitened_from_z(model, z: torch.Tensor) -> torch.Tensor:
    """
    If base is GaussianPCA: return PCA-whitened coordinates (N, k) using (z - loc) @ W^T.
    If base is DiagGaussian: return standardized z, i.e., (z - loc)/scale when available, else z.
    Accepts q0.loc of shape (D,) or (1, D).
    """
    q0 = model.q0
    # GaussianPCA path
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        W = q0.W         # expected (k, D)
        loc = q0.loc     # expected (D,) or (1, D)
        if isinstance(loc, torch.Tensor) and loc.dim() == 2 and loc.shape[0] == 1:
            loc = loc.squeeze(0)
        if W.dim() == 2 and isinstance(loc, torch.Tensor) and loc.dim() == 1 and W.shape[1] == z.shape[1] and loc.shape[0] == z.shape[1]:
            return torch.matmul(z - loc, W.T).to(z.dtype)  # (N, k)
        # fall through to DiagGaussian-style standardization if shapes mismatch

    # DiagGaussian (or other) path
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if isinstance(loc, torch.Tensor) and loc.dim() == 2 and loc.shape[0] == 1:
        loc = loc.squeeze(0)
    if isinstance(scale, torch.Tensor) and scale.dim() == 2 and scale.shape[0] == 1:
        scale = scale.squeeze(0)
    if isinstance(loc, torch.Tensor) and isinstance(scale, torch.Tensor):
        return ((z - loc) / (scale + 1e-12)).to(z.dtype)
    if isinstance(loc, torch.Tensor) and loc.dim() == 1 and loc.shape[0] == z.shape[1]:
        return (z - loc).to(z.dtype)
    return z.to(z.dtype)


def _scale_param_penalty(models) -> torch.Tensor:
    total = None
    for m in models:
        has_named = hasattr(m, "named_parameters")
        if not has_named:
            continue
        for name, p in m.named_parameters():
            lname = name.lower()
            if any(key in lname for key in ("scale", "log_scale", "s_log", "logs")):
                val = (p ** 2).sum()
                total = val if total is None else (total + val)
    if total is None:
        return torch.tensor(0.0)
    return total

def normalizing_simr_flows_whitener(
    views: List[pd.DataFrame],
    *,
    # Data / split / batching
    val_fraction: float = 0.2,
    batch_size: int = 256,
    val_batch_size: int = 2048,
    seed: int = 0,
    cuda_device: Optional[str] = None,

    # Flow config
    K: int = 64,
    leaky_relu_negative_slope: float = 0.0,

    # Base distribution
    base_distribution: str = "GaussianPCA",
    pca_latent_dimension: Optional[int] = 4,
    base_min_log: float = -5.0,
    base_max_log: float = 5.0,
    base_sigma: float = 0.1,

    # Flow stability knobs
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

    # Jitter alpha schedule
    jitter_alpha: float = 0.0,
    jitter_alpha_end: float = 0.0,
    jitter_alpha_mode: str = "cosine",
    jitter_alpha_total_steps: None,

    # Tradeoff controller
    tradeoff_mode: str = "ema",
    target_ratio: float = 9.0,
    lambda_penalty: float = 1.0,
    ema_beta: float = 0.98,

    # Penalty type/params
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

    # Optional: penalize scale/log-scale parameters
    scale_penalty_weight: float = 0.0,
) -> Dict[str, Any]:

    # ----------------------------
    # Device / seeds / base checks
    # ----------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _ensure_device(cuda_device)

    base_lower = base_distribution.lower()
    if base_lower in ("gaussianpca", "pca") and pca_latent_dimension is None:
        raise ValueError("pca_latent_dimension must be provided when base_distribution='GaussianPCA'.")

    # ----------------------------
    # Wrap inputs in the Dataset
    # ----------------------------
    # Give view names v0, v1, ... (order preserved)
    view_names = [f"v{i}" for i in range(len(views))]
    view_map = {vn: df for vn, df in zip(view_names, views)}

    # Important: we set normalization=None so the dataset only aligns/encodes.
    # We still fit our own per-view standardizers below to match original behavior.
    ds_full = MultiViewDataFrameDataset(
        views=view_map,
        normalization=None,      # keep raw numeric scale; we standardize below
        alpha=0.0,               # no dataset noise
        add_noise_in='none',
        impute='mean',           # ensure finite values after encoding
        nonfinite_clean='mean',
        concat_views=False,
        dtype=torch.float32,
    )
    N = len(ds_full)

    # Train/val split (by index so both views and masks stay aligned)
    perm = torch.randperm(N)
    n_val = int(N * float(val_fraction))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    ds_train = Subset(ds_full, train_idx)
    ds_val   = Subset(ds_full, val_idx)

    # DataLoaders (use dataset worker_init if you later add augmentation)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=val_batch_size, shuffle=False, drop_last=False)

    # ----------------------------
    # Discover per-view dims (D_v)
    # ----------------------------
    first_batch = next(iter(train_loader))
    Ds = [int(first_batch['views'][vn].shape[1]) for vn in view_names]

    # ----------------------------
    # Fit per-view standardizers on TRAIN subset
    # ----------------------------
    # We stream through train_loader once to compute mean/std per view.
    sums = [torch.zeros(D, device=device) for D in Ds]
    cnts = [0 for _ in Ds]
    for batch in train_loader:
        for i, vn in enumerate(view_names):
            xb = batch['views'][vn].to(device)
            sums[i] += xb.sum(dim=0)
            cnts[i] += xb.shape[0]
    means = [s / max(1, c) for s, c in zip(sums, cnts)]

    ssq = [torch.zeros(D, device=device) for D in Ds]
    for batch in train_loader:
        for i, vn in enumerate(view_names):
            xb = batch['views'][vn].to(device)
            diff = xb - means[i][None, :]
            ssq[i] += (diff * diff).sum(dim=0)
    stds = [torch.sqrt(ss / max(1, cnts[i])) for i, ss in enumerate(ssq)]
    stds = [s.clamp_min(1e-6) for s in stds]

    standardizers = [{'mean': m.detach().cpu().unsqueeze(0), 'std': s.detach().cpu().unsqueeze(0)}
                     for m, s in zip(means, stds)]

    # ----------------------------
    # Build per-view models
    # ----------------------------
    models = []
    opt_params = []
    for D in Ds:
        q0 = _build_base_distribution(
            D=D,
            base_distribution=base_distribution,
            pca_latent_dimension=pca_latent_dimension,
            min_log=base_min_log,
            max_log=base_max_log,
            sigma=base_sigma,
        )
        model = create_rnvp(
            latent_size=D,
            K=K,
            q0=q0,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            scale_cap=scale_cap,
            spectral_norm_scales=spectral_norm_scales,
            additive_first_n=additive_first_n,
            actnorm_every=actnorm_every,
            mask_mode=mask_mode,
        ).to(device)

        # Validate PCA base shape where applicable
        if base_lower in ("gaussianpca", "pca"):
            if not (hasattr(model.q0, "W") and hasattr(model.q0, "loc")):
                raise RuntimeError("GaussianPCA base must expose 'W' and 'loc'.")
            W = model.q0.W
            if pca_latent_dimension is not None and W.shape[0] != int(pca_latent_dimension):
                raise RuntimeError(f"GaussianPCA latent_dim mismatch: expected k={int(pca_latent_dimension)} "
                                   f"but got W.shape[0]={W.shape[0]} for D={D}.")

        models.append(model)
        opt_params += list(model.parameters())

    # For DiagGaussian: require equal dims across views for cross-view penalty
    dims_equal = all(D == Ds[0] for D in Ds)
    if base_lower in ("diag", "diaggaussian", "gauss_diag", "diaggaussian") and not dims_equal:
        raise RuntimeError(f"DiagGaussian cross-view penalty requires equal dims across views, got {Ds}.")

    # ----------------------------
    # Resume (if any)
    # ----------------------------
    start_step = 0
    ckpt = _try_resume(resume_checkpoint, device)
    if ckpt is not None:
        try:
            for m, sd in zip(models, ckpt["models"]):
                m.load_state_dict(sd)
            if "standardizers" in ckpt:
                standardizers = ckpt["standardizers"]
            start_step = int(ckpt.get("step", 0))
            if verbose:
                print(f"[info] resumed from '{resume_checkpoint}' @ step {start_step}")
        except Exception as e:
            print(f"[warn] resume failed to load state dicts: {e}")

    # ----------------------------
    # Optimizer, schedulers
    # ----------------------------
    opt = optim.AdamW(opt_params, lr=lr, betas=betas, eps=eps_opt, weight_decay=weight_decay)
    if jitter_alpha_total_steps is None:
        jitter_sched = _AlphaSchedule(jitter_alpha, jitter_alpha_end, max_iter, jitter_alpha_mode)
    else:
        jitter_sched = _AlphaSchedule(jitter_alpha, jitter_alpha_end, jitter_alpha_total_steps, jitter_alpha_mode)
    ema_nll = None
    ema_pen = None

    # Early stop bookkeeping
    best_metric = float('inf')
    best_state = None
    best_std = None
    best_step = -1
    patience_counter = 0
    smooth_total = None

    if save_checkpoint_dir is not None:
        os.makedirs(save_checkpoint_dir, exist_ok=True)
    ckpt_every = checkpoint_interval if (checkpoint_interval is not None) else val_interval

    # ----------------------------
    # Training loop
    # ----------------------------
    # We iterate over the train_loader each step; this matches your original per-step batching,
    # but now comes from the dataset (aligned, finite, per-view tensors).
    for step in range(start_step, max_iter):
        for m in models:
            m.train()

        for batch in train_loader:
            J = float(jitter_sched.value(step))

            # Build standardized per-view batches
            batch_X = []
            for i, vn in enumerate(view_names):
                xb = batch['views'][vn].to(device)
                # whitener-standardize here (dataset normalization=None)
                mu = standardizers[i]['mean'].to(device)
                sd = standardizers[i]['std'].to(device)
                xb = (xb - mu) / sd
                if J > 0:
                    xb = xb + J * torch.randn_like(xb)
                batch_X.append(xb)

            # Forward: inverse + nll + features
            nll_sum = 0.0
            H = []
            for m, xb in zip(models, batch_X):
                z, log_det = _inverse_with_guard(m, xb)
                nll = -(m.q0.log_prob(z) + log_det).mean()
                nll_sum = nll_sum + nll
                h = _extract_whitened_from_z(m, z)
                H.append(h)

            # Cross-view penalty (same as before)
            can_penalize = True
            if base_lower in ("gaussianpca", "pca"):
                ks = [h.shape[1] for h in H]
                if not all(k == ks[0] for k in ks):
                    raise RuntimeError(f"GaussianPCA views disagree on k: {ks}")
            else:
                ks = [h.shape[1] for h in H]
                if not all(k == ks[0] for k in ks):
                    raise RuntimeError(f"DiagGaussian standardized dims disagree across views: {ks}")

            penalty_active = can_penalize and (len(models) >= 2)
            pen = torch.tensor(0.0, device=H[0].device)
            if penalty_active:
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        if penalty_type == "barlow_twins_align":
                            pen = pen + _barlow_twins_align_square(H[i], H[j], bt_lambda_diag, bt_lambda_offdiag, bt_eps)
                        elif penalty_type == "correlate":
                            pen = pen + _corr_square(H[i], H[j], want="maximize")
                        elif penalty_type == "decorrelate":
                            pen = pen + _corr_square(H[i], H[j], want="minimize")
                        else:
                            raise ValueError(f"Unknown penalty_type: {penalty_type}")

            warm = min(1.0, (step + 1) / max(1, penalty_warmup_iters))
            if penalty_active:
                ema_nll = nll_sum.detach() if ema_nll is None else (ema_beta * ema_nll + (1 - ema_beta) * nll_sum.detach())
                ema_pen = pen.detach() if ema_pen is None else (ema_beta * ema_pen + (1 - ema_beta) * pen.detach())
                if tradeoff_mode == "ema":
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

            lam_eff = torch.nan_to_num(lam_eff, nan=0.0, posinf=0.0, neginf=0.0)
            lam_eff = lam_eff.clamp(min=0.0)

            total = nll_sum + warm * lam_eff * pen

            if scale_penalty_weight and scale_penalty_weight > 0:
                sp = _scale_param_penalty(models).to(total.device)
                total = total + float(scale_penalty_weight) * sp

            opt.zero_grad(set_to_none=True)
            if torch.isfinite(total):
                total.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], grad_clip)
                opt.step()

        # ----------------------------
        # Validation (unchanged)
        # ----------------------------
        do_val = ((step + 1) % max(1, val_interval) == 0) or (step == max_iter - 1)
        if do_val:
            for m in models:
                m.eval()
            with torch.no_grad():
                nll_val_sum = 0.0
                count = 0
                for batch in val_loader:
                    for i, vn in enumerate(view_names):
                        xb = batch['views'][vn].to(device)
                        mu = standardizers[i]['mean'].to(device)
                        sd = standardizers[i]['std'].to(device)
                        xb = (xb - mu) / sd
                        z, log_det = _inverse_with_guard(models[i], xb)
                        nll_val_sum += (-(models[i].q0.log_prob(z) + log_det)).sum()
                    count += xb.shape[0]
                nll_val = nll_val_sum / max(1, count)
                D_total = sum(Ds)
                val_bpd = _bits_per_dim(nll_val, D_total)

                # smooth_total tracks train_total via EMA-like smoothing
                smooth_total = float(total.detach().cpu()) if smooth_total is None else \
                               (early_stop_beta * float(smooth_total) + (1 - early_stop_beta) * float(total.detach().cpu()))
                metric = val_bpd if best_selection_metric == "val_bpd" else smooth_total

                improved = (metric + early_stop_min_delta < best_metric) and (step + 1) >= early_stop_min_iters
                if improved:
                    best_metric = metric
                    best_state = [m.state_dict() for m in models]
                    best_std = [{'mean': s['mean'].clone(), 'std': s['std'].clone()} for s in standardizers]
                    best_step = step + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    alpha_now = float(jitter_sched.value(step))
                    penalty_last = float(pen.detach().cpu()) if penalty_active else 0.0
                    lam_eff_value = float(lam_eff.detach().cpu()) if penalty_active else 0.0
                    st = float(smooth_total) if smooth_total is not None else float('nan')
                    print(f"[{step+1:5d}/{max_iter}] (alpha={alpha_now:.4f}): "
                          f"train_total={float(total):.4f}, smooth_total={st:.4f}, "
                          f"val_bpd={val_bpd:.4f}, lam_eff={lam_eff_value:.2e}, penalty={penalty_last:.4f}")

                if early_stop_enabled and (step + 1) >= early_stop_min_iters and patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"[early stop] step {step+1} best {best_selection_metric}={best_metric:.4f} @ step {best_step}")
                    break

        # ----------------------------
        # Checkpointing (unchanged)
        # ----------------------------
        if save_checkpoint_dir is not None:
            if ((step + 1) % max(1, ckpt_every) == 0) or (step == max_iter - 1):
                ckpt_path = os.path.join(save_checkpoint_dir, f"ckpt_step{step+1}.pth")
                _save_checkpoint(ckpt_path, step + 1, models, standardizers)
                latest_path = os.path.join(save_checkpoint_dir, "latest.pth")
                _save_checkpoint(latest_path, step + 1, models, standardizers)

    # Restore best for final eval if requested
    if restore_best_for_final_eval and (best_state is not None):
        for m, sd in zip(models, best_state):
            m.load_state_dict(sd)
        standardizers = best_std

    metrics = {
        "best_step": best_step if best_state is not None else max_iter,
        "best_metric": best_metric,
        "best_metric_name": best_selection_metric,
    }

    return {
        "models": models,
        "standardizers": standardizers,
        "metrics": metrics,
    }
