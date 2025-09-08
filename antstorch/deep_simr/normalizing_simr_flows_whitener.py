import os, json, time
from pathlib import Path
from typing import List, Union, Dict, Any
from functools import partial
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utilities import MultiViewDataFrameDataset

import normflows as nf

from ..architectures import create_real_nvp_normalizing_flow_model
from ..utilities import absolute_pearson_correlation

# -------------------------
# Small utilities
# -------------------------

def _split_indices(N: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(val_fraction * N))
    return idx[n_val:], idx[:n_val]  # train_idx, val_idx


def _mv_collate(batch, view_names):
    # batch: list of dicts: {'views': {v: Tensor[D_v]}, 'masks': {...}, ...}
    per_view = {v: [] for v in view_names}
    for sample in batch:
        for v in view_names:
            per_view[v].append(sample['views'][v])
    # stack each view to (B, D_v) and return in view_names order
    return [torch.stack(per_view[v], dim=0) for v in view_names]


def _eval_val_bpd(models, val_loader, device, total_dims, verbose: bool):
    was_train = [m.training for m in models]
    for m in models: m.eval()
    try:
        with torch.inference_mode():
            total = 0.0; count = 0; skipped = 0
            for xs in val_loader:
                xs = [x.to(device, non_blocking=True).double() for x in xs]
                for m, model in enumerate(models):
                    kld = model.forward_kld(xs[m]).double()
                    if not torch.isfinite(kld):
                        skipped += 1; continue
                    total += kld.item(); count += 1
            if skipped > 0 and verbose:
                print(f"[VAL][WARN] skipped {skipped} non-finite chunks")
            avg_kld = total / max(count, 1)
            return avg_kld / float(total_dims)
    finally:
        for flag, m in zip(was_train, models):
            m.train(flag)


def _save_models(models, base_prefix: str, view_names: List[str], tag: str):
    paths = []
    for m, name in zip(models, view_names):
        p = f"{base_prefix}_model_{name}_{tag}.pt"
        torch.save(m.state_dict(), p)
        paths.append(p)
    return paths


def _bt_batch_norm(x: torch.Tensor, eps: float = 1e-6):
    xm = x - x.mean(dim=0, keepdim=True)
    xs = torch.sqrt(xm.var(dim=0, unbiased=False, keepdim=True) + eps)
    return xm / xs


def _bt_pair_loss(a: torch.Tensor, b: torch.Tensor, lambda_diag=1.0, lambda_offdiag=5e-3):
    B = a.shape[0]
    C = (a.T @ b) / max(B, 1)
    on_diag = (torch.diagonal(C) - 1.0).pow(2).sum()
    off_diag = (C - torch.diag(torch.diagonal(C))).pow(2).sum()
    return lambda_diag * on_diag + lambda_offdiag * off_diag


def _extract_whitened_from_z(model, z: torch.Tensor) -> torch.Tensor:
    """
    Return PCA-whitened coordinates (if GaussianPCA) or standardized z (if DiagGaussian).
    Falls back to z if parameters are missing.
    """
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        # GaussianPCA: whitened = (z - loc) @ W^T
        return torch.matmul(z - q0.loc, q0.W.T).double()
    # DiagGaussian: (z - loc)/scale if available
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if loc is not None and scale is not None:
        return ((z - loc) / (scale + 1e-12)).double()
    return z.double()


def _attach_scale_capture(model):
    """
    Register forward hooks on every MaskedAffineFlow.s head (the bounded MLP).
    Captures ŝ tensors during forward/inverse passes into model._scale_tensors.
    """
    handles = []
    model._scale_tensors = []

    def _hook(_module, _inp, out):
        # out is ŝ for the current block & batch
        if torch.is_tensor(out):
            model._scale_tensors.append(out.double())

    for f in model.flows:
        # Only attach to affine coupling blocks, not ActNorm, etc.
        if isinstance(f, nf.flows.MaskedAffineFlow) and isinstance(f.s, nn.Module):
            handles.append(f.s.register_forward_hook(_hook))
    return handles


def _scale_penalty(models, weight: float = 1e-4):
    """
    Compute mean |ŝ| across all blocks and all views for the current batch,
    then scale by 'weight'. Safe if no tensors were captured.
    """
    total = torch.zeros((), dtype=torch.double, device=next(models[0].parameters()).device)
    count = 0
    for m in models:
        if hasattr(m, "_scale_tensors"):
            for s_hat in m._scale_tensors:
                total = total + s_hat.abs().mean()
                count += 1
    if count == 0:
        return torch.zeros_like(total)
    return weight * (total / count)

# -------------------------
# Alpha schedule and checkpoints
# -------------------------
class _AlphaSchedule:
    def __init__(self, alpha_start: float, alpha_end: float, total_steps: int, mode: str = "cosine"):
        self.a0 = float(alpha_start)
        self.a1 = float(alpha_end)
        self.T = max(1, int(total_steps))
        self.mode = mode

    def __call__(self, step: int) -> float:
        import math
        t = min(max(int(step), 0), self.T)
        if self.mode == "cosine":
            return self.a1 + 0.5 * (self.a0 - self.a1) * (1.0 + math.cos(math.pi * t / self.T))
        elif self.mode == "linear":
            return self.a0 + (self.a1 - self.a0) * (t / self.T) if self.T > 0 else self.a1
        elif self.mode == "exp":
            k = 5.0
            return max(self.a1, self.a0 * math.exp(-k * t / self.T))
        else:
            return self.a0

def _save_checkpoint(path: str,
                     models: list,
                     view_names: list,
                     optimizer,
                     scheduler,
                     epoch: int,
                     global_step: int,
                     alpha_now: float,
                     best_val_bpd: float,
                     extra_meta: dict | None = None):
    device = next(models[0].parameters()).device
    state = {
        "models": {vn: m.state_dict() for vn, m in zip(view_names, models)},
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "alpha_now": float(alpha_now),
        "best_val_bpd": None if best_val_bpd is None else float(best_val_bpd),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "meta": {} if extra_meta is None else extra_meta,
        "device": str(device),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)

def _load_checkpoint(path: str, models: list, view_names: list,
                     optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    for vn, m in zip(view_names, models):
        m.load_state_dict(ckpt["models"][vn], strict=True)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    # Restore RNG (optional, but helps reproducibility across resumes)
    if "rng" in ckpt and ckpt["rng"]:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            for i, s in enumerate(ckpt["rng"]["cuda"]):
                torch.cuda.set_rng_state(s, device=i)
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["random"])
    return ckpt





# -------------------------
# Main trainer
# -------------------------
def normalizing_simr_flows_whitener(
    # Core data/model setup
    views: Union[List[pd.DataFrame], pd.DataFrame],
    pca_latent_dimension: int = 4,
    K: int = 64,
    leaky_relu_negative_slope: float = 0.2,
    base_distribution: str = "GaussianPCA",     # {"DiagGaussian","GaussianPCA"}
    jitter_alpha: float = 0.0,                

    # Optimization
    lr: float = 1e-4,
    batch_size: int = 512,
    weight_decay: float = 1e-5,
    max_iter: int = 1200,
    cuda_device: str = "cuda:0",
    seed: int = 0,

    # Tradeoff / weighting
    tradeoff_mode: str = "uncertainty",         # {"ema","uncertainty","fixed"}
    target_ratio: float = 9.0,                  # EMA mode
    lambda_penalty: float = 1.0,                # Fixed mode
    ema_beta: float = 0.98,

    # Penalty/alignment knobs
    penalty_type: str = "barlow_twins_align",   # {"decorrelate","correlate","barlow_twins_align"}
    bt_lambda_diag: float = 1.0,
    bt_lambda_offdiag: float = 5e-3,
    bt_eps: float = 1e-6,
    penalty_warmup_iters: int = 400,

    # Validation
    val_fraction: float = 0.2,
    val_interval: int = 200,
    val_batch_size: int = 2048,

    # Early stopping
    early_stop_enabled: bool = True,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_iters: int = 600,
    early_stop_beta: float = 0.98,

    # Checkpointing / Output
    best_selection_metric: str = "val_bpd",     # {"val_bpd","smooth_total"}
    restore_best_for_final_eval: bool = True,
    output_prefix: Union[str, None] = None,     # if not None: save models/metrics using this prefix



    # Checkpointing / Resume
    resume_checkpoint: Union[str, None] = None,
    save_checkpoint_dir: Union[str, None] = None,
    checkpoint_interval: Union[int, None] = None,  # defaults to val_interval if None

    # Jitter annealing
    jitter_alpha_end: float = 0.0,
    jitter_alpha_mode: str = "cosine",
    jitter_alpha_total_steps: int = 20000,
    # Verbosity
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train one or more normalizing-flow "whitener" models for (single or multi-view) data.

    Parameters
    ----------
    views : list[pd.DataFrame] | pd.DataFrame
        One or more data frames (rows = samples; columns = features).
        Single DataFrame → single-view mode (no cross-view penalty).
    pca_latent_dimension : int, default=4
        Latent dimensionality for GaussianPCA base. Ignored if base_distribution="DiagGaussian".
    K : int, default=64
        Number of coupling blocks in the RealNVP flow built via
        `antstorch.create_real_nvp_normalizing_flow_model`.
    leaky_relu_negative_slope : float, default=0.2
        Negative slope for LeakyReLU activations.
    base_distribution : {"DiagGaussian","GaussianPCA"}, default="GaussianPCA"
        Base distribution choice. GaussianPCA enables PCA-based whitening of the latent.
    jitter_alpha : float, default=0.0
        Gaussian noise level for quasi-categorical features.  Only applied to training data.

    lr, batch_size, weight_decay, max_iter, cuda_device, seed : see code defaults.

    tradeoff_mode : {"ema","uncertainty","fixed"}, default="uncertainty"
        Combination strategy for likelihood (KLD per-dim) and cross-view penalty.
        Single-view mode optimizes only the likelihood (penalty disabled).
    target_ratio, lambda_penalty, ema_beta : see code defaults.

    penalty_type : {"decorrelate","correlate","barlow_twins_align"}, default="barlow_twins_align"
        Multi-view penalty:
          - "decorrelate": minimize |corr| (→ 0)
          - "correlate"  : maximize |corr| (→ 1)
          - "barlow_twins_align": diag→1, off-diag→0 on standardized latents
    bt_lambda_diag, bt_lambda_offdiag, bt_eps, penalty_warmup_iters : see code defaults.

    val_fraction, val_interval, val_batch_size : validation controls.

    early_stop_* : early stopping controls on smoothed total training loss.

    best_selection_metric : {"val_bpd","smooth_total"}, default="val_bpd"
        Selects which metric controls "best" checkpointing.
    restore_best_for_final_eval : bool, default=True
        Reload best weights before final validation.
    output_prefix : str | None
        If set, saves weights & JSON metadata under this prefix.
resume_checkpoint : str | None
        If set, load optimizer/scheduler/model state and resume training.
save_checkpoint_dir : str | None
        If set, save full checkpoints (models, optimizer, scheduler, RNG) here.
checkpoint_interval : int | None
        Save a checkpoint every N steps (defaults to val_interval if None).
jitter_alpha_end, jitter_alpha_mode, jitter_alpha_total_steps : control annealing of jitter.
verbose : bool
        Console logging.

    Returns
    -------
    dict
        {
          "models": list[torch.nn.Module],
          "metrics": {...},
          "history": {...},
          "config": {...},
          "standardizers": {view_idx: {"mean": list, "std": list}, ...},
          "best_paths": list[str] | None,
          "last_paths": list[str] | None,
          "view_names": list[str],
        }
    """
    t0 = time.time()

    # Determine checkpoint cadence
    if checkpoint_interval is None:
        checkpoint_interval = val_interval

    # Alpha schedule (anneal jitter)
    alpha_sched = _AlphaSchedule(jitter_alpha, jitter_alpha_end, jitter_alpha_total_steps, jitter_alpha_mode)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    # Normalize views input
    if isinstance(views, pd.DataFrame):
        views = [views]
    assert len(views) >= 1, "Provide at least one DataFrame in 'views'."
    view_names = [f"view{i}" for i in range(len(views))]

    # Row/shape checks
    nrows_set = {len(df) for df in views}
    if len(nrows_set) != 1:
        raise ValueError(f"All views must have equal row counts; got {nrows_set}")
    N = nrows_set.pop()

    dims = [df.shape[1] for df in views]
    total_dims = float(sum(dims))
    min_dim = min(dims)

    if base_distribution == "GaussianPCA" and pca_latent_dimension > min_dim:
        if verbose:
            print(f"[WARN] pca_latent_dimension={pca_latent_dimension} > min(view_dims)={min_dim}, clipping.")
        pca_latent_dimension = min_dim

    # Split (train/val) using positional indices → convert to index labels for slicing
    train_idx, val_idx = _split_indices(N, val_fraction, seed)
    index_labels = views[0].index
    train_labels = index_labels[train_idx]
    val_labels   = index_labels[val_idx]

    # Build per-split views (slice each DF by labels)
    views_train = {nm: df.loc[train_labels] for nm, df in zip(view_names, views)}
    views_val   = {nm: df.loc[val_labels]   for nm, df in zip(view_names, views)}

    # --- NEW: dataset instances (z-score by default; impute after norm; noise in normalized space) ---
    ds_train = MultiViewDataFrameDataset(
        views=views_train,
        normalization="0mean",          # or {"view0":"01", ...} per view
        alpha=jitter_alpha,             # typical to disable noise for likelihood-only training; set >0 if desired
        add_noise_in="normalized",      # 'raw'|'normalized'|'none'
        impute="mean",                  # 'none'|'mean'|'zero'
        concat_views=False,             # keep per-view tensors
        number_of_samples=None,         # use true length
        dtype=torch.float64,            # match your model .double()
    )
        
    ds_val = MultiViewDataFrameDataset(
        views=views_val,
        normalization="0mean",
        alpha=0.0,
        add_noise_in="none",
        impute="mean",
        concat_views=False,
        number_of_samples=None,
        dtype=torch.float64,
    )

    # DataLoaders
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0,  # set >0 if you like; seed workers via worker_init_fn below
        collate_fn=partial(_mv_collate, view_names=view_names),
        worker_init_fn=MultiViewDataFrameDataset.worker_init_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
        collate_fn=partial(_mv_collate, view_names=view_names),
        worker_init_fn=MultiViewDataFrameDataset.worker_init_fn,
    )

    # Initialize dataset jitter alpha
    if hasattr(ds_train, 'set_alpha'):
        ds_train.set_alpha(float(jitter_alpha))
    else:
        ds_train.alpha = float(jitter_alpha)

    dims = [ds_train.view_dim(v) for v in view_names]
    total_dims = float(sum(dims))
    min_dim = min(dims)

    # now do the PCA latent clip using the *encoded* dims
    if base_distribution == "GaussianPCA" and pca_latent_dimension > min_dim:
        if verbose:
            print(f"[WARN] pca_latent_dimension={pca_latent_dimension} > min(view_dims)={min_dim}, clipping.")
        pca_latent_dimension = min_dim

    standardizers = {}
    for i, v in enumerate(view_names):
        st = ds_train._state[v]  # means/eps_std are finite + epsilon-protected
        # Match prior contract: mean/std in *raw* units for "0mean"
        standardizers[i] = {"mean": st.mean.tolist(), "std": st.eps_std.tolist()}

    # Build models
    models: List[torch.nn.Module] = []
    params: List[torch.nn.Parameter] = []
    for d in dims:
        if base_distribution == "GaussianPCA":
            q0 = nf.distributions.GaussianPCA(d, latent_dim=pca_latent_dimension)
        elif base_distribution == "DiagGaussian":
            q0 = nf.distributions.DiagGaussian(d)
        else:
            raise ValueError(f"Unknown base_distribution: {base_distribution}")

        model = create_real_nvp_normalizing_flow_model(
            d, K=K, q0=q0, leaky_relu_negative_slope=leaky_relu_negative_slope,
            scale_cap=3.0, spectral_norm_scales=True
        ).to(device).double()
        models.append(model)
        params += list(model.parameters())

    _scale_hook_handles = []
    for m in models:
        _scale_hook_handles.extend(_attach_scale_capture(m))

    # Tradeoff params
    single_view = (len(models) == 1)
    if single_view and verbose:
        print("[INFO] Single-view mode detected: alignment/correlation penalty is disabled.")

    s_kld = s_pen = None
    if tradeoff_mode == "uncertainty":
        s_kld = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.double))
        params.append(s_kld)
        if not single_view:
            s_pen = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.double))
            params.append(s_pen)

    optimizer = torch.optim.Adamax(params, lr=lr, weight_decay=weight_decay)
    warmup_iters = 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter - warmup_iters, eta_min=lr*0.1)

    
    # Resume if requested
    start_step = 0
    if resume_checkpoint:
        if verbose:
            print(f"[RESUME] Loading checkpoint from {resume_checkpoint}")
        ckpt = _load_checkpoint(resume_checkpoint, models, view_names, optimizer=optimizer, scheduler=scheduler)
        start_step = int(ckpt.get("global_step", 0))
        # Restore dataset alpha
        alpha_now = float(ckpt.get("alpha_now", jitter_alpha))
        if hasattr(ds_train, 'set_alpha'):
            ds_train.set_alpha(alpha_now)
        else:
            ds_train.alpha = alpha_now
        # Restore best val bpd if present
        if ckpt.get("best_val_bpd") is not None:
            best_val_bpd = float(ckpt["best_val_bpd"])
        

    # EMA state (for "ema" tradeoff)
    ema_kld = 0.0
    ema_pen = 0.0
    eps = 1e-8

    # Logging/early stop
    best_smooth_total = float("inf")
    smooth_total = None
    no_improve = 0
    best_val_bpd = float("inf")

    it_hist, total_hist, kld_hist, pen_hist = [], [], [], []

    # Checkpoints
    if output_prefix:
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Train loop
    train_iter = iter(train_loader)
    for step in range(start_step, max_iter):
        optimizer.zero_grad()
        try:
            xs = next(train_iter)            # xs is list[Tensor] ordered by view_names
        except StopIteration:
            train_iter = iter(train_loader)
            xs = next(train_iter)

        xs = [x.to(device).double() for x in xs]

        # Update jitter temperature via schedule
        alpha_now = alpha_sched(step)
        if hasattr(ds_train, 'set_alpha'):
            ds_train.set_alpha(alpha_now)
        else:
            ds_train.alpha = float(alpha_now)


        for m in models:
            if hasattr(m, "_scale_tensors"):
                m._scale_tensors.clear()

        zs = [models[m].inverse(xs[m]) for m in range(len(models))]

        # Likelihood
        loss_kld_raw = torch.zeros((), device=device, dtype=torch.double)
        for m in range(len(models)):
            loss_kld_raw = loss_kld_raw + models[m].forward_kld(xs[m]).double()
        loss_kld_bpd = loss_kld_raw / (total_dims + eps)

        # Penalty (multi-view only)
        loss_pen_raw = torch.zeros((), device=device, dtype=torch.double)
        if not single_view:
            for m in range(len(models)):
                for n in range(m + 1, len(models)):
                    # Use whitened coordinates for consistency with penalties
                    eps_m = _extract_whitened_from_z(models[m], zs[m])
                    eps_n = _extract_whitened_from_z(models[n], zs[n])
                    eps_m = torch.where(torch.isfinite(eps_m), eps_m, torch.zeros_like(eps_m))
                    eps_n = torch.where(torch.isfinite(eps_n), eps_n, torch.zeros_like(eps_n))

                    if penalty_type == "barlow_twins_align":
                        a = _bt_batch_norm(eps_m, eps=bt_eps)
                        b = _bt_batch_norm(eps_n, eps=bt_eps)
                        loss_pen_raw = loss_pen_raw + _bt_pair_loss(a, b, bt_lambda_diag, bt_lambda_offdiag)

                    elif penalty_type == "decorrelate":
                        corr = absolute_pearson_correlation(eps_m, eps_n, 1e-8).double()
                        loss_pen_raw = loss_pen_raw + corr

                    elif penalty_type == "correlate":
                        corr = absolute_pearson_correlation(eps_m, eps_n, 1e-8).double()
                        loss_pen_raw = loss_pen_raw + (1.0 - corr)

                    else:
                        raise ValueError(f"Unknown penalty_type: {penalty_type}")

            # Warm-up
            if penalty_warmup_iters and penalty_warmup_iters > 0:
                warm = min(1.0, float(step + 1) / float(penalty_warmup_iters))
                loss_pen_raw = warm * loss_pen_raw

        # Combine
        if tradeoff_mode == "uncertainty":
            if single_view:
                loss = torch.exp(-s_kld) * loss_kld_bpd + 0.5 * s_kld
            else:
                loss = torch.exp(-s_kld) * loss_kld_bpd + torch.exp(-s_pen) * loss_pen_raw + 0.5 * (s_kld + s_pen)
        elif tradeoff_mode == "ema":
            ema_kld = ema_beta * ema_kld + (1 - ema_beta) * float(loss_kld_bpd.detach().cpu().numpy())
            if not single_view:
                ema_pen = ema_beta * ema_pen + (1 - ema_beta) * float(loss_pen_raw.detach().cpu().numpy())
            L1n = loss_kld_bpd / (ema_kld + eps)
            if single_view:
                loss = L1n
            else:
                L2n = loss_pen_raw / (ema_pen + eps)
                w1 = target_ratio / (target_ratio + 1.0)
                w2 = 1.0 - w1
                loss = w1 * L1n + w2 * L2n
        elif tradeoff_mode == "fixed":
            if single_view:
                loss = loss_kld_bpd
            else:
                loss = loss_kld_bpd + float(lambda_penalty) * loss_pen_raw
        else:
            raise ValueError(f"Unknown tradeoff_mode: {tradeoff_mode}")
 
        loss = loss + _scale_penalty(models, weight=1e-4)
 
        if not torch.isfinite(loss):
            if verbose:
                print(f"[WARN] Non-finite total loss at step {step}; skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        if step < warmup_iters:
            # linear warmup from 0 → lr
            for g in optimizer.param_groups:
                g['lr'] = lr * float(step + 1) / float(warmup_iters)
        else:
            scheduler.step()

        # Logs
        curr_total = float(loss.detach().cpu().numpy())
        curr_kld   = float(loss_kld_bpd.detach().cpu().numpy())
        curr_pen   = float(loss_pen_raw.detach().cpu().numpy()) if not single_view else None

        it_hist.append(step)
        total_hist.append(curr_total)
        kld_hist.append(curr_kld)
        pen_hist.append(curr_pen)

        # Smooth total + early stop
        smooth_total = curr_total if smooth_total is None else (early_stop_beta * smooth_total + (1 - early_stop_beta) * curr_total)
        if best_selection_metric == "smooth_total":
            if (smooth_total + early_stop_min_delta) < best_smooth_total:
                best_smooth_total = smooth_total
                if output_prefix:
                    _save_models(models, output_prefix, view_names, "best")
        if (smooth_total + early_stop_min_delta) < best_smooth_total:
            no_improve = 0
        else:
            no_improve += 1
        if early_stop_enabled and (step + 1) >= early_stop_min_iters and no_improve >= early_stop_patience:
            if verbose:
                print(f"[INFO] Early stopping at step {step+1}; best smooth_total={best_smooth_total:.6f}")
            break

        # Periodic validation
        if (step + 1) % val_interval == 0:
            bpd = _eval_val_bpd(models, val_loader, device, total_dims, verbose)
            if verbose:
                print(f"[VAL] Iteration {step+1} (alpha={alpha_now:.4f}): smooth_total={smooth_total:.6f}, bpd={bpd:.6f}")
            if best_selection_metric == "val_bpd":
                if bpd < best_val_bpd:
                    best_val_bpd = bpd
                    if output_prefix:
                        _save_models(models, output_prefix, view_names, "best")
            if output_prefix:
                _save_models(models, output_prefix, view_names, "last")

        # Periodic checkpoint save
        if save_checkpoint_dir and ((step + 1) % checkpoint_interval == 0):
            ckpt_path = os.path.join(save_checkpoint_dir, f"ckpt_step{step+1}.pt")
            _save_checkpoint(ckpt_path, models, view_names, optimizer, scheduler,
            epoch=0, global_step=step+1, alpha_now=alpha_now,
            best_val_bpd=best_val_bpd,
            extra_meta={"views": view_names, "dims": dims})


    # Finalize checkpoints

    # Save a final checkpoint
    if save_checkpoint_dir:
        final_ckpt_path = os.path.join(save_checkpoint_dir, f"ckpt_final_step{step+1 if 'step' in locals() else 0}.pt")
        _save_checkpoint(final_ckpt_path, models, view_names, optimizer, scheduler,
        epoch=0, global_step=(step+1 if 'step' in locals() else 0),
        alpha_now=(alpha_now if 'alpha_now' in locals() else jitter_alpha_end),
        best_val_bpd=best_val_bpd,
        extra_meta={"views": view_names, "dims": dims})

    if output_prefix:
        _save_models(models, output_prefix, view_names, "last")

    # Ensure there is a best
    if output_prefix and best_selection_metric in {"val_bpd", "smooth_total"}:
        for nm in view_names:
            p_best = f"{output_prefix}_model_{nm}_best.pt"
            p_last = f"{output_prefix}_model_{nm}_last.pt"
            if not Path(p_best).exists() and Path(p_last).exists():
                torch.save(torch.load(p_last, map_location=device), p_best)

    # Restore best for final eval
    if restore_best_for_final_eval and output_prefix:
        for nm, m in zip(view_names, models):
            p_best = f"{output_prefix}_model_{nm}_best.pt"
            if Path(p_best).exists():
                m.load_state_dict(torch.load(p_best, map_location=device))

    # Final validation
    final_bpd = _eval_val_bpd(models, val_loader, device, total_dims, verbose)

    history = {
        "iter": it_hist,
        "loss_total": total_hist,
        "loss_kld_per_dim": kld_hist,
        "loss_penalty": pen_hist if not single_view else None,
    }
    metrics = {
        "val_bpd_mean": float(final_bpd) if np.isfinite(final_bpd) else None,
        "steps": int(it_hist[-1]) + 1 if len(it_hist) else 0,
        "final_loss": float(total_hist[-1]) if len(total_hist) else None,
        "final_kld_per_dim": float(kld_hist[-1]) if len(kld_hist) else None,
        "final_penalty": float(pen_hist[-1]) if (not single_view and len(pen_hist)) else None,
        "elapsed_sec": float(time.time() - t0),
    }
    config = {
        "views": [list(df.columns) if hasattr(df, "columns") else f"dim{d}" for df, d in zip(views, dims)],
        "pca_latent_dimension": pca_latent_dimension,
        "K": K,
        "leaky_relu_negative_slope": leaky_relu_negative_slope,
        "base_distribution": base_distribution,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "max_iter": max_iter,
        "cuda_device": cuda_device,
        "seed": seed,
        "tradeoff_mode": tradeoff_mode,
        "target_ratio": target_ratio,
        "lambda_penalty": lambda_penalty,
        "ema_beta": ema_beta,
        "penalty_type": penalty_type,
        "bt_lambda_diag": bt_lambda_diag,
        "bt_lambda_offdiag": bt_lambda_offdiag,
        "bt_eps": bt_eps,
        "penalty_warmup_iters": penalty_warmup_iters,
        "val_fraction": val_fraction,
        "val_interval": val_interval,
        "val_batch_size": val_batch_size,
        "early_stop_enabled": early_stop_enabled,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_min_iters": early_stop_min_iters,
        "early_stop_beta": early_stop_beta,
        "best_selection_metric": best_selection_metric,
        "restore_best_for_final_eval": restore_best_for_final_eval,
        "output_prefix": output_prefix,
        "single_view": single_view,
    }

    # Save metadata if requested
    if output_prefix:
        outdir = Path(os.path.dirname(output_prefix)) if os.path.dirname(output_prefix) else Path(".")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        (outdir / "config.json").write_text(json.dumps(config, indent=2))
        # Save standardizers too
        (outdir / "standardizers.json").write_text(json.dumps(standardizers, indent=2))

    return {
        "models": models,
        "metrics": metrics,
        "history": history,
        "config": config,
        "standardizers": standardizers,
        "best_paths": [f"{output_prefix}_model_{nm}_best.pt" for nm in view_names] if output_prefix else None,
        "last_paths": [f"{output_prefix}_model_{nm}_last.pt" for nm in view_names] if output_prefix else None,
        "view_names": view_names,
    }