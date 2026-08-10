"""
base_glow_trainer.py
===============
Abstract base class for the LAMNr Glow 2D and 3D trainers.

Subclasses must implement:
  - build_models(args)  -> List[nn.Module]
  - build_loaders(args) -> (train_loader, val_loader, global_step)
  - extract_view(batch, vi) -> torch.Tensor   # view extraction + to01

The base class owns:
  - The unified training loop (gradient accumulation, AMP, EMA)
  - Strict per-iteration memory management (gc.collect + explicit del)
  - Checkpoint save / load with automatic DataParallel prefix stripping
  - All shared utility functions (moved here from the two trainer scripts)
"""

from __future__ import annotations

import abc
import copy
import csv
import gc
import json
import os
import platform
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision as tv
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import ants
import antstorch

from antstorch.lamnr_flows.misc.latent_alignment import (
    LatentAlignmentLossManager,
    Projector,
    ScreenState,
    flatten_latents,
)


# ---------------------------------------------------------------------------
# DataParallel wrappers (shared by 2D and 3D)
# ---------------------------------------------------------------------------

class GlowStepWrapper(nn.Module):
    """Wraps a Glow model so DataParallel can scatter the forward pass."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, logdet = self.model.inverse_and_log_det(x)
        m = self.model
        if hasattr(m, "q0s"):
            bases = m.q0s
        elif hasattr(m, "q0"):
            bases = m.q0
        else:
            raise RuntimeError("Model has no base distribution (q0 / q0s).")
        if not isinstance(bases, (list, tuple, nn.ModuleList)):
            bases = [bases]
        if isinstance(z, (list, tuple)):
            if len(bases) == 1 and len(z) > 1:
                bases = list(bases) * len(z)
            base_lp = sum(b.log_prob(zi) for b, zi in zip(bases, z))
        else:
            base_lp = bases[0].log_prob(z)
        log_prob = base_lp + logdet
        z_flat = flatten_latents(z)
        return log_prob, z_flat

    def inverse_and_log_det(self, x):
        return self.model.inverse_and_log_det(x)

    def log_prob(self, x):
        return self.model.log_prob(x)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


class GlowDataParallel(nn.DataParallel):
    """nn.DataParallel with explicit redirections for Glow-specific methods.

    Kept around for single-process multi-GPU debugging (e.g. --devices
    cuda:0,cuda:1 without torchrun) -- but nn.DataParallel scatters the
    forward pass across GPUs using a Python ThreadPoolExecutor
    (parallel_apply), and PyTorch's autograd engine backward pass runs
    per-thread. In practice this made torch.autograd.set_detect_anomaly's
    reported crash site non-deterministic across runs when a NaN gradient
    appeared, and single-GPU (no DataParallel at all) reliably avoided the
    issue -- consistent with a threading-related race in DataParallel's
    backward, a known class of issue with this (largely superseded) API.
    GlowDDP below is the safe multi-GPU path.
    """

    def log_prob(self, x):
        return self.module.log_prob(x)

    def inverse_and_log_det(self, x):
        return self.module.inverse_and_log_det(x)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)


class GlowDDP(DistributedDataParallel):
    """
    DistributedDataParallel with explicit redirections for Glow-specific
    methods, mirroring GlowDataParallel.

    Unlike nn.DataParallel (one process, multiple threads, one thread per
    GPU), DDP runs one process per GPU: no shared Python-level threading
    during forward/backward, gradient synchronization happens via NCCL
    all-reduce after backward() completes locally on each process. This is
    both the standard recommended multi-GPU approach in PyTorch and, in our
    case, resolved a NaN-gradient issue that only reproduced under
    DataParallel's multi-threaded backward.

    IMPORTANT: log_prob()/inverse_and_log_det()/sample() below call
    self.module directly, bypassing DDP's forward() -- which means they do
    NOT trigger gradient synchronization. This is intentional and correct
    for eval/sampling (no backward involved there), but must never be used
    for the training forward pass -- that always goes through __call__
    (DDP's own forward), same as the base nn.Module convention.
    """

    def log_prob(self, x):
        return self.module.log_prob(x)

    def inverse_and_log_det(self, x):
        return self.module.inverse_and_log_det(x)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_hw_divisible(
    H: int,
    W: int,
    L: int,
    D: Optional[int] = None,
    spatial_dims: int = 2,
) -> None:
    r = 2 ** L
    if H % r or W % r:
        raise ValueError(
            f"H and W must be divisible by 2**L={r}. Got H={H}, W={W}, L={L}"
        )
    if spatial_dims == 3:
        if D is None:
            raise ValueError("D must be provided when spatial_dims=3.")
        if D % r:
            raise ValueError(
                f"D must be divisible by 2**L={r}. Got D={D}, L={L}"
            )


def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    if x.ndim < 4:
        return x
    spatial_dims = tuple(range(2, x.ndim))
    x_min = x.amin(dim=spatial_dims, keepdim=True)
    x_max = x.amax(dim=spatial_dims, keepdim=True)
    norm = (x - x_min) / (x_max - x_min + eps)
    return torch.clamp(norm, 1e-5, 1.0 - 1e-5)


def bits_per_dim(logp: torch.Tensor, num_dims: int) -> torch.Tensor:
    return -logp / (np.log(2.0) * float(num_dims))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def make_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    decay_gamma: float,
    decay_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if warmup_iters <= 0 and (decay_gamma == 1.0 or decay_steps <= 0):
        return None

    def lr_lambda(step: int) -> float:
        s = max(1, step)
        scale = 1.0
        if warmup_iters > 0 and s < warmup_iters:
            scale *= s / float(warmup_iters)
        if decay_gamma != 1.0 and decay_steps > 0:
            scale *= decay_gamma ** (s / float(decay_steps))
        return scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def _copy_actnorm_state(src: nn.Module, dst: nn.Module) -> None:
    for ms, md in zip(src.modules(), dst.modules()):
        if "actnorm" in ms.__class__.__name__.lower():
            for fld in ("logs", "log_scale", "scale", "weight"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("bias", "b"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("initialized", "is_initialized", "inited"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    try:
                        getattr(md, fld).data.copy_(getattr(ms, fld).data)
                    except Exception:
                        setattr(md, fld, bool(getattr(ms, fld)))

@torch.no_grad()
def _prime_if_needed(model: nn.Module, x: torch.Tensor) -> None:
    x1 = x[:1]
    
    # --- PLINDAGE ARCHITECTURAL COMPATIBLE 2D / 3D ---
    # Si le tenseur est en 3D (sans batch/canal) -> (1, 1, H, W, D) ou (1, 1, H, W)
    if x1.ndim == 3:
        x1 = x1.unsqueeze(0).unsqueeze(1)

    # Si le tenseur a 4 dimensions, il faut distinguer la 2D de la 3D :
    elif x1.ndim == 4:
        # Cas A : C'est un lot 2D déjà formaté avec son canal unitaire (B, 1, H, W) ou (B, 3, H, W)
        if x1.shape[1] == 1 or x1.shape[1] == 3:
            pass
        # Cas B : C'est un lot 3D brut qui n'a pas encore son canal (B, H, W, D) -> (B, 1, H, W, D)
        else:
            x1 = x1.unsqueeze(1)
    # -------------------------------------------------
        
    p = next(model.parameters(), None)
    dev = p.device if p is not None else x1.device
    
    # --- CORRECTION DE LA FAUTE DE FRAPPE ---
    x1 = x1.to(dev, dtype=torch.float32)
    # ----------------------------------------
    
    try:
        _ = model.inverse_and_log_det(x1)
    except Exception:
        _ = model.log_prob(x1)

def log_prob_exact(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z, logdet = model.inverse_and_log_det(x)

    def bases_of(m):
        if hasattr(m, "q0s"):
            q0s = getattr(m, "q0s")
            if isinstance(q0s, (list, tuple, nn.ModuleList)):
                return list(q0s)
        if hasattr(m, "q0"):
            q0 = getattr(m, "q0")
            if isinstance(q0, (list, tuple, nn.ModuleList)):
                return list(q0)
            if q0 is not None:
                return [q0]
        raise RuntimeError("No base distribution(s) on model (q0/q0s)")

    if isinstance(z, (list, tuple)):
        bases = bases_of(model)
        if len(bases) == 1 and len(z) > 1:
            bases = bases * len(z)
        if len(bases) != len(z):
            raise RuntimeError(f"bases ({len(bases)}) != latents ({len(z)})")
        base_lp = sum(b.log_prob(zi) for b, zi in zip(bases, z))
    else:
        base_lp = bases_of(model)[0].log_prob(z)
    return base_lp + logdet


@torch.no_grad()
def warmup_actnorm_with_real_batch(model: nn.Module, x_real: torch.Tensor) -> None:
    dev = next(model.parameters()).device
    x1 = x_real[:1].to(dev, torch.float32)
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue


def _extract_views_from_batch(batch, num_views: Optional[int] = None) -> List[torch.Tensor]:
    """Normalize any multi-view batch format into a list of per-view tensors."""
    if isinstance(batch, tuple) and len(batch) > 0 and (
        torch.is_tensor(batch[0]) or isinstance(batch[0], (list, tuple, dict))
    ):
        return _extract_views_from_batch(batch[0], num_views=num_views)

    if isinstance(batch, dict):
        if "x" in batch:
            return _extract_views_from_batch(batch["x"], num_views=num_views)
        if "views" in batch:
            vs = batch["views"]
            if isinstance(vs, (list, tuple)) and len(vs) > 0 and torch.is_tensor(vs[0]):
                return list(vs)
            raise ValueError("Batch['views'] not in expected list/tuple[tensor] format.")
        for v in batch.values():
            if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
                return list(v)
        raise ValueError("Batch dict format not recognized for multi-view data.")

    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        return list(batch)

    if torch.is_tensor(batch):
        if batch.ndim == 5:
            B, V, C, H, W = batch.shape
            return [batch[:, vi, :, :, :] for vi in range(V)]
        elif batch.ndim == 4:
            if num_views is None or num_views <= 1:
                return [batch]
            B, Ctot, H, W = batch.shape
            if Ctot % num_views != 0:
                raise ValueError(
                    f"Cannot split (B,C,H,W)=({B},{Ctot},{H},{W}) into {num_views} views."
                )
            Cpv = Ctot // num_views
            return [batch[:, vi * Cpv : (vi + 1) * Cpv, :, :] for vi in range(num_views)]
        else:
            raise ValueError(f"Unsupported tensor ndim={batch.ndim}; expected 4 or 5.")

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def _coerce_nchw_4d(
    x, target_hw: Optional[Tuple[int, int]] = None, axis=-1) -> torch.Tensor:
    """Coerce sample output to (N, C, H, W), handling 2D and 3D tensors."""
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3, 4, 5)]
        if not cands:
            raise ValueError("No tensor candidates in sample output.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 5:
                mid = t.shape[-1] // 2
                t = t[..., mid]
            elif t.dim() == 3:
                if t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1).contiguous()
                t = t.unsqueeze(0)
            elif t.dim() == 4:
                if t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
                    t = t.permute(0, 3, 1, 2).contiguous()
            fixed.append(t)
            areas.append(int(t.shape[-2]) * int(t.shape[-1]))
        x = fixed[int(torch.tensor(areas, dtype=torch.float32).argmax().item())]

    if not torch.is_tensor(x):
        raise ValueError(f"Sample output is not a tensor: {type(x)}")
    if x.dim() == 5:
        mid = x.shape[-1] // 2
        x = x[..., mid]
    if x.dim() == 3:
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.dim() == 4 and x.size(1) not in (1, 3):
        mid = x.shape[axis] // 2
        x = torch.select(x, dim=axis, index=mid)
        x = x.unsqueeze(1)
    x = torch.clamp(x, 0, 1).float()
    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if (H, W) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return x


def _make_grid_canvas(x: torch.Tensor, nrow: int = 10) -> torch.Tensor:
    assert torch.is_tensor(x) and x.dim() == 4, "x must be (N,C,H,W)"
    N, C, H, W = x.shape
    cols = int(nrow)
    rows = (N + cols - 1) // cols
    canvas = x.new_zeros(C, rows * H, cols * W)
    for idx in range(N):
        r, c = idx // cols, idx % cols
        canvas[:, r * H : (r + 1) * H, c * W : (c + 1) * W] = x[idx]
    return canvas


@torch.no_grad()
def _sample_chunk(
    model: nn.Module,
    n: int,
    temp: float,
    warm_x=None,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Draw a single chunk of ``n`` samples from ``model``, with the same
    GPU/CPU fallback chain used previously in ``_save_samples_grid``.

    Returns (x, None) on success — where x is the raw sample tensor, still
    on its original device — or (None, error_message) on failure.
    """
    temp_tensor = torch.tensor(temp, dtype=torch.float32)
    device_original = next(model.parameters()).device
    try:
        try:
            s = model.sample(n, temperature=temp_tensor)
        except TypeError:
            s = model.sample(n)
    except Exception as e:
        msg = str(e).lower()
        if "latent shapes unknown" in msg and warm_x is not None:
            _prime_if_needed(model, warm_x)
            try:
                try:
                    s = model.sample(n, temperature=temp_tensor)
                except TypeError:
                    s = model.sample(n)
            except Exception as e2:
                return None, str(e2)
        else:
            try:
                model.to("cpu")
                temp_cpu = temp_tensor.to("cpu")
                try:
                    s = model.sample(n, temperature=temp_cpu)
                except TypeError:
                    s = model.sample(n)
                if isinstance(s, (list, tuple)):
                    s = [t.to(device_original) if isinstance(t, torch.Tensor) else t for t in s]
                elif isinstance(s, torch.Tensor):
                    s = s.to(device_original)
                model.to(device_original)
            except Exception as e_cpu:
                model.to(device_original)
                return None, f"Primary failed: {e}. CPU fallback failed: {e_cpu}"

    x = s[0] if isinstance(s, (list, tuple)) else s
    return x, None


@torch.no_grad()
def _save_samples_grid(
    model: nn.Module,
    n: int,
    temp: float,
    out_prefix,
    nrow: int = 10,
    target_hw=None,
    warm_x=None,
    which_type: str = "to01",
    chunk_size: int = 20,
) -> Tuple[bool, Optional[str]]:
    """
    Generates ``n`` samples from ``model`` in chunks of at most
    ``chunk_size``.

    Rationale: model.sample() is called directly on the underlying module
    (see GlowStepWrapper.sample / GlowDataParallel.sample), which bypasses
    DataParallel's multi-GPU scatter entirely — the whole generative pass
    runs on a single GPU. Drawing all ``n`` samples (previously 100) in one
    shot creates a large single-device memory spike every --eval-interval
    iterations; on deep 3D flows (L*K coupling steps) at full resolution
    this was tipping GPU0 into OOM a few hundred iterations later. Chunking
    + moving each chunk to CPU immediately + clearing the CUDA cache between
    chunks bounds that peak.
    """
    chunks: List[torch.Tensor] = []
    remaining = int(n)
    csize = max(1, int(chunk_size))
    while remaining > 0:
        k = min(csize, remaining)
        x_chunk, err = _sample_chunk(model, k, temp, warm_x=warm_x)
        if x_chunk is None:
            return False, err
        chunks.append(x_chunk.detach().to("cpu"))
        del x_chunk
        remaining -= k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        x = torch.cat(chunks, dim=0)
        del chunks
        x = _coerce_nchw_4d(x, target_hw=target_hw)

        # Instead of discarding the whole grid on non-finite values, sanitize
        # them in place (NaN -> 0, +/-Inf -> the finite min/max observed in
        # this batch) so a sample grid still gets saved every plot interval.
        # This is purely for the diagnostic PNG; it has no effect on the
        # training loss/gradients.
        sanitize_note = None
        nonfinite_mask = ~torch.isfinite(x)
        n_nonfinite = int(nonfinite_mask.sum().item())
        if n_nonfinite > 0:
            finite_vals = x[~nonfinite_mask]
            if finite_vals.numel() > 0:
                fmin = float(finite_vals.min())
                fmax = float(finite_vals.max())
            else:
                fmin, fmax = 0.0, 1.0
            x = torch.nan_to_num(x, nan=0.0, posinf=fmax, neginf=fmin)
            pct = 100.0 * n_nonfinite / x.numel()
            sanitize_note = f"sanitized {n_nonfinite} non-finite voxel(s) ({pct:.3f}%)"

        _std = x.std().item()

        valid = {"to01", "clamp", "both"}
        if which_type not in valid:
            which_type = "to01"

        x_to01  = to01(x)  if which_type in ("to01", "both") else None
        x_clamp = x.clamp(0, 1) if which_type in ("clamp", "both") else None

        def _save(img_batch, suffix):
            if img_batch is None:
                return
            if img_batch.shape[0] < n:
                reps = (n + img_batch.shape[0] - 1) // img_batch.shape[0]
                img_batch = img_batch.repeat(reps, 1, 1, 1)
            img_batch = img_batch[:n]
            grid = _make_grid_canvas(img_batch, nrow=nrow)
            tv.utils.save_image(grid, str(out_prefix) + suffix)

        _save(x_to01,  "_to01.png")
        _save(x_clamp, "_clamp.png")
        return True, sanitize_note
    except Exception as e:
        return False, str(e)


def _save_metric_plots(
    csv_path: Path, out_dir: Path, remove_spikes: bool = False
) -> None:
    if not csv_path.exists():
        return
    iters, losses, bpds = [], [], []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    it, loss, bpd = int(float(row[0])), float(row[1]), float(row[2])
                    iters.append(it)
                    losses.append(loss)
                    bpds.append(bpd)
                except ValueError:
                    continue
        if len(iters) < 2:
            return
        if remove_spikes and len(losses) > 10:
            s_losses = pd.Series(losses)
            w = min(50, max(5, len(losses) // 10))
            rolling_med = s_losses.rolling(window=w, center=True, min_periods=1).median()
            diff = np.abs(s_losses - rolling_med)
            rolling_mad = diff.rolling(window=w, center=True, min_periods=1).median()
            is_spike = diff > (5 * rolling_mad + 1e-6)
            losses = np.where(is_spike, np.nan, losses)
            bpds   = np.where(is_spike, np.nan, bpds)
        for values, ylabel, title, fname in [
            (losses, "loss",    "Training loss",        "loss_curve.png"),
            (bpds,   "sum_bpd", "Sum BPD (train)",      "bpd_curve.png"),
        ]:
            plt.figure()
            plt.plot(iters, values)
            plt.xlabel("iter"); plt.ylabel(ylabel); plt.title(title)
            plt.tight_layout()
            plt.savefig(out_dir / fname); plt.close()
    except Exception:
        pass


def screen_dump_run_config(
    args, out_dir: Path, note: str = "", dataset_info: Optional[dict] = None
) -> None:
    def _fmt_bool(x):
        return "true" if bool(x) else "false"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args))
    
    # Calculs dérivés
    cfg["grad_accum"]     = int(cfg.get("grad_accum", 1))
    cfg["effective_batch"] = int(cfg.get("batch", 0)) * cfg["grad_accum"]

    env = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python":            platform.python_version(),
        "torch":             torch.__version__,
        "cuda_available":    torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    
    # Fusion des infos dataset si fournies
    if dataset_info:
        cfg["dataset_info"] = dataset_info

    # Sauvegarde JSON (machine-readable)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({"env": env, "config": cfg, "note": note}, f, indent=2)

    # Construction de l'affichage humain (pretty TXT)
    rows = [
        f"[run] {env['timestamp']} | Py {env['python']} | torch {env['torch']} "
        f"| cuda={_fmt_bool(env['cuda_available'])} (n={env['cuda_device_count']})"
    ]
    if note:
        rows.append(f"[note] {note}")

    def add(k, v):
        rows.append(f"{k:>24}: {'None' if v is None else v}")

    # Core architecture
    add("out_dir", cfg.get("out_dir"))
    add("spatial_dims", cfg.get("spatial_dims"))
    add("views", getattr(args, "num_views", None))
    add("H×WxD", f"{cfg.get('H')}×{cfg.get('W')}×{cfg.get('D')}")
    add("L / K / hidden", f"{cfg.get('L')} / {cfg.get('K')} / {cfg.get('hidden')}")
    add("base", cfg.get("base"))
    add("net_actnorm", _fmt_bool(cfg.get("net_actnorm")))
    add("precision / amp_dtype", f"{cfg.get('precision')} / {cfg.get('amp_dtype')}")
    add("devices", cfg.get("devices"))
    add("num_workers", cfg.get("num_workers"))
    add("seed", cfg.get("seed"))

    # Training & Optimization
    add("batch", cfg.get("batch"))
    add("grad_accum", cfg.get("grad_accum"))
    add("effective_batch", cfg.get("effective_batch"))
    add("max_iter / extra", f"{cfg.get('max_iter')} / {cfg.get('extra_iters')}")
    add("eval / plot interval", f"{cfg.get('eval_interval')} / {cfg.get('plot_interval')}")
    add("lr / warmup", f"{cfg.get('lr')} / {cfg.get('warmup_iters')}")
    add("grad_clip", cfg.get("grad_clip"))
    add("weight_decay", cfg.get("weight_decay"))
    add("ema / decay", f"{_fmt_bool(cfg.get('ema'))} / {cfg.get('ema_decay')}")

    add("lr_decay_gamma/steps", f"{cfg.get('lr_decay_gamma')} / {cfg.get('lr_decay_steps')}")
    add(
        "plateau (fac/pat/thr/cd)",
        f"{cfg.get('plateau_factor')} / {cfg.get('plateau_patience')} / "
        f"{cfg.get('plateau_threshold')} / {cfg.get('plateau_cooldown')}",
    )
    add("min_lr", cfg.get("min_lr"))

    # Checkpointing / resume
    add("resume", cfg.get("resume") or None)
    add("auto_resume / use_ckpt_config", f"{_fmt_bool(cfg.get('auto_resume'))} / {_fmt_bool(cfg.get('use_ckpt_config'))}")

    # Data & Augmentation
    add("slice_idx", cfg.get("slice_idx"))
    add("val_frac", cfg.get("val_frac"))
    add("subject_limit", cfg.get("subject_limit") or None)
    add("train / val samples", f"{cfg.get('train_samples')} / {cfg.get('val_samples')}")
    add("disable_aug_anneal", _fmt_bool(cfg.get("disable_aug_anneal")))
    add("aug_schedules", cfg.get("aug_schedules"))

    # Alignment & VICReg
    add("align", cfg.get("align"))
    add("weighting", cfg.get("weighting"))
    add("align_weight/warmup", f"{cfg.get('align_weight')} / {cfg.get('align_warmup')}")
    add("proj_dim / proj_hidden", f"{cfg.get('proj_dim')} / {cfg.get('proj_hidden')}")
    add("vicreg (i/v/c/g)", f"{cfg.get('vicreg_inv')}/{cfg.get('vicreg_var')}/{cfg.get('vicreg_cov')}/{cfg.get('vicreg_gamma')}")
    add("temperature (infonce)", cfg.get("temperature"))
    add("barlow_lambda", cfg.get("barlow_lambda"))
    add("hsic_sigma", cfg.get("hsic_sigma"))
    add("init_logvar (nll/align)", f"{cfg.get('init_logvar_nll')} / {cfg.get('init_logvar_align')}")

    # CCA Screening
    add("screen", cfg.get("screen"))
    add("screen_frac", cfg.get("screen_frac"))
    add("screen_warmup / refresh", f"{cfg.get('screen_warmup')} / {cfg.get('screen_refresh')}")
    add("cca_ridge", cfg.get("cca_ridge"))
    add("prefilter_frac", cfg.get("prefilter_frac"))

    # Glow Specifics
    add("sample_mode / temp", f"{cfg.get('sample_mode')} / {cfg.get('sample_temp')}")
    add("sample_chunk_size", cfg.get("sample_chunk_size"))
    add("grad_checkpoint", cfg.get("grad_checkpoint"))
    add("smooth_alpha", cfg.get("smooth_alpha"))
    add("scale_map / scale_cap", f"{cfg.get('scale_map')} / {cfg.get('scale_cap')}")
    add("actnorm_scale_cap", cfg.get("actnorm_scale_cap"))
    add(
        "glowbase (min/max log, logscale_factor)",
        f"{cfg.get('glowbase_min_log')} / {cfg.get('glowbase_max_log')} / "
        f"{cfg.get('glowbase_logscale_factor')}",
    )

    if dataset_info:
        rows.append("-" * 60)
        for k, v in dataset_info.items():
            add(k, v)

    txt = "\n".join(rows) + "\n"
    print("\n" + txt)
    with open(out_dir / "run_config.txt", "a") as f:
        f.write(txt)


# ---------------------------------------------------------------------------
# BaseLAMNrTrainer
# ---------------------------------------------------------------------------

class BaseLAMNrTrainer(abc.ABC):
    """
    Abstract base class for the LAMNr Glow 2D and 3D trainers.

    Subclasses must implement
    -------------------------
    build_models(args)  -> List[nn.Module]
    build_loaders(args) -> (train_loader, val_loader, global_step)
    extract_view(batch, vi, dev) -> torch.Tensor  # view extraction + to01
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_models(self, args) -> List[nn.Module]:
        ...

    @abc.abstractmethod
    def build_loaders(self, args):
        ...

    @abc.abstractmethod
    def extract_view(
        self, batch: object, vi: int, dev: torch.device
    ) -> torch.Tensor:
        """Return a single view tensor shaped (B, C, *spatial), normalized to [0,1]."""
        ...

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, args) -> None:
        """Call once from main() after arg parsing."""
        self.args = args
        set_deterministic(args.seed)

        # Optional: torch.autograd.set_detect_anomaly(True) makes backward()
        # raise immediately at the *forward* op that produced a NaN/Inf
        # gradient, with a full stack trace, instead of just silently
        # yielding a non-finite grad norm downstream (which our grad-finite
        # guard in train() now catches, but only tells you *that* it
        # happened, not *where*). Off by default: it's substantially
        # slower, so only enable for a short diagnostic run.
        if bool(getattr(args, "detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)
            tqdm.write("[debug] torch.autograd.set_detect_anomaly(True) enabled — expect a large slowdown")

        # Distributed (DDP) process group + device.
        #
        # Launched via `torchrun --nproc_per_node=N ...`: torchrun sets
        # RANK/LOCAL_RANK/WORLD_SIZE env vars and spawns one process per
        # GPU. We detect that here and init the NCCL process group; this
        # replaces nn.DataParallel (one process, N threads) with N
        # independent processes synchronizing gradients via all-reduce --
        # no cross-GPU Python threading, which is what made DataParallel's
        # backward non-deterministic under a NaN gradient (see GlowDDP
        # docstring). Falls back to the previous single-process behavior
        # (including optional DataParallel via --devices cuda:0,cuda:1)
        # when not launched under torchrun, so single-GPU debugging and
        # --detect-anomaly runs still work unchanged.
        self.is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
        if self.is_ddp:
            self.rank       = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            dev = torch.device(f"cuda:{self.local_rank}")
            tqdm.write(
                f"[ddp] rank {self.rank}/{self.world_size} bound to "
                f"{dev} (local_rank={self.local_rank})"
            )
        else:
            self.rank       = 0
            self.local_rank = 0
            self.world_size = 1
            if args.devices.lower() == "cpu":
                dev = torch.device("cpu")
            elif args.devices == "mps" and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device(args.devices.split(",")[0])
        self.dev = dev
        self.is_main_process = (self.rank == 0)

        # AMP
        if args.precision == "double":
            self.model_dtype = torch.float64
            self.amp_enabled = False
            self.amp_dtype   = None
        elif args.precision == "float":
            self.model_dtype = torch.float32
            self.amp_enabled = False
            self.amp_dtype   = None
        else:
            self.model_dtype = torch.float32
            self.amp_enabled = True
            if (
                args.amp_dtype == "bf16"
                and dev.type == "cuda"
                and torch.cuda.is_bf16_supported()
            ):
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16

        self.scaler = torch.amp.GradScaler(
            enabled=(self.amp_enabled and self.amp_dtype == torch.float16),
            init_scale=2.0 ** 12,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=200,
        )

        # Data
        self.train_loader, self.val_loader, self.global_step = self.build_loaders(args)

        # Models
        self.models: List[nn.Module] = self.build_models(args)
        self.ema_models: Optional[List[nn.Module]] = None

        # ActNorm warmup with real data
        with torch.no_grad():
            try:
                warm_batch = next(iter(self.train_loader))
                xs = _extract_views_from_batch(warm_batch, num_views=len(self.models))
                for vi, m in enumerate(self.models):
                    _prime_if_needed(m, xs[vi])
            except StopIteration:
                pass

        # Projectors + alignment manager
        self.projectors: Optional[nn.ModuleList] = None
        if args.align != "none":
            with torch.no_grad():
                x_tmpl = to01(xs[0][:1].to(dtype=torch.float32, device=dev))
                z_probe, _ = self.models[0].inverse_and_log_det(x_tmpl)
                flat_dim = flatten_latents(z_probe).size(1)
            self.projectors = nn.ModuleList([
                Projector(flat_dim, args.proj_hidden, args.proj_dim)
                .to(dtype=torch.float32, device=dev)
                .train()
                for _ in range(len(self.models))
            ])

        self.align_mgr = LatentAlignmentLossManager(
            args=args,
            projectors=self.projectors,
            device=dev,
        )

        # Kendall scalars
        self.s_nll = self.s_align = None
        if args.weighting == "kendall" and args.align != "none":
            self.s_nll   = nn.Parameter(
                torch.tensor([args.init_logvar_nll],   device=dev, dtype=torch.float32)
            )
            self.s_align = nn.Parameter(
                torch.tensor([args.init_logvar_align], device=dev, dtype=torch.float32)
            )

        # Optimizer & schedulers
        param_groups = [{"params": [p for m in self.models for p in m.parameters()]}]
        if self.projectors is not None:
            param_groups.append({"params": list(self.projectors.parameters())})
        if self.s_nll is not None:
            param_groups.append(
                {"params": [self.s_nll, self.s_align], "weight_decay": 0.0}
            )
        # AdamW, not Adamax: Adamax tracks a per-parameter *max* gradient
        # magnitude (exp_inf) that can decay near zero for a parameter
        # that goes a while without a meaningful gradient, then produce a
        # huge update the instant it next sees one -- root-caused (across
        # several rollback iterations on the 96x64x96/L=5/hidden=128 run)
        # as the recurring source of "finite but exploded" and "briefly
        # plausible then permanently NaN" training blowups that kept
        # slipping past per-step safety nets. AdamW's second-moment
        # estimate is an EMA of squared gradients, which is smoother and
        # doesn't have that same collapse-then-spike failure mode.
        self.opt = torch.optim.AdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
        self.warm = make_warmup(
            self.opt, args.warmup_iters, args.lr_decay_gamma, args.lr_decay_steps
        )
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=args.plateau_factor,
            patience=getattr(args, "plateau_patience", 4),
            threshold=args.plateau_threshold,
            cooldown=getattr(args, "plateau_cooldown", 0),
            min_lr=getattr(args, "min_lr", 1e-6),
        )

        # Paths
        self.run_dir    = Path(args.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "training_state.pt"
        self.csv_path   = self.run_dir / "metrics.csv"

        # Checkpoint resume
        self.start_iter = self._maybe_resume(args)
        if args.extra_iters > 0:
            args.max_iter = (self.start_iter - 1) + args.extra_iters

        # Sync global_step
        with self.global_step.get_lock():
            self.global_step.value = int(self.start_iter)
        for loader in (self.train_loader, self.val_loader):
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "global_step_ref"):
                try:
                    loader.dataset.global_step_ref.value = self.start_iter
                except Exception:
                    pass

        # Prime latent-shape caches
        self._prime_all_latent_shapes()

        # CSV header + run-config dump: rank-0-only file writes.
        if self.rank == 0:
            if not self.csv_path.exists():
                with open(self.csv_path, "w") as f:
                    f.write("iter,loss,sum_bpd,lr\n")
            else:
                try:
                    df = pd.read_csv(self.csv_path)
                    df = df[df["iter"] < self.start_iter]
                    df.to_csv(self.csv_path, index=False)
                except Exception as e:
                    print(f"[warn] Could not clean CSV: {e}")

            try:
                dataset_info = {
                    "train_len": len(getattr(self.train_loader.dataset, "images", [])),
                    "val_len":   len(getattr(self.val_loader.dataset,   "images", [])),
                    "batch_size": args.batch,
                    "grad_accum": int(getattr(args, "grad_accum", 1)),
                    "effective_batch": int(args.batch) * int(getattr(args, "grad_accum", 1)),
                }
            except Exception:
                dataset_info = {"note": "dataset stats unavailable"}
            screen_dump_run_config(args, self.run_dir, note="post-dataset build",
                                   dataset_info=dataset_info)
        if self.is_ddp:
            dist.barrier()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_dp_prefix(state_dict: dict) -> dict:
        """
        Remove DataParallel / GlowStepWrapper / torch.compile wrapper prefixes
        ('module.', 'model.', '_orig_mod.') from all keys **before writing to
        disk**.

        Guarantees that saved checkpoints are always compatible with inference
        scripts that load a bare Glow model — even if training used
        DataParallel and/or ANTSNF_TORCH_COMPILE=1 (torch.compile wraps the
        module and prefixes every state_dict key with '_orig_mod.'; without
        stripping it here, a checkpoint saved from a compiled model has 100%
        of its keys mismatched against a bare model at inference time, which
        `load_state_dict` fails on and then silently masks via its
        strict=False fallback -- leaving the whole model at its random
        from-scratch initialization).
        """
        return {
            k.replace("module.", "").replace("model.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

    def save_checkpoint(self, it: int) -> None:
        """Save latest + milestone checkpoints with clean (no-prefix) state dicts."""
        blob = {
            "iter":    it + 1,
            "opt":     self.opt.state_dict(),
            "warm":    (self.warm.state_dict() if self.warm else None),
            "models":  [self._strip_dp_prefix(m.state_dict()) for m in self.models],
            "ema":     (
                [self._strip_dp_prefix(em.state_dict()) for em in self.ema_models]
                if self.ema_models else None
            ),
            "proj":    (self.projectors.state_dict() if self.projectors else None),
            "kendall": {
                "s_nll":   float(self.s_nll.detach().cpu())   if self.s_nll   else None,
                "s_align": float(self.s_align.detach().cpu()) if self.s_align else None,
            },
            # "s_cap_wired_to_conv": True marks checkpoints trained with the
            # antsnormflows fix (commit 2249ecd) that correctly wires the
            # configured scale_cap into Invertible1x1(x1)Conv, instead of the
            # conv silently using its own hardcoded default (2.5). Inference
            # code (lamnr_glow_tool_{2,3}d.py's build_model) checks this flag
            # to decide whether to pin the conv to the legacy cap for
            # backward compatibility. Do not remove this key without also
            # checking those call sites.
            "config":  {**vars(self.args), "s_cap_wired_to_conv": True},
            "scaler":  (
                self.scaler.state_dict()
                if self.scaler is not None and self.scaler.is_enabled()
                else None
            ),
        }
        # 1. Latest — for --auto-resume
        torch.save(blob, self.state_path)
        # 2. Milestone
        iter_path = self.run_dir / f"training_state_it{it:06d}.pt"
        torch.save(blob, iter_path)
        # 3. Purge old milestones
        self.cleanup_checkpoints()
        tqdm.write(f"[ckpt] saved {iter_path.name} (and updated latest)")

    def cleanup_checkpoints(self, keep_every: Optional[int] = None) -> None:
        """
        Keep only milestone checkpoints that are multiples of keep_every.

        keep_every defaults to args.eval_interval (i.e. every saved
        milestone is kept) rather than a large fixed constant. Runs that
        diverge before reaching a large fixed threshold (e.g. the previous
        hardcoded 10_000/20_000) would otherwise have every intermediate
        milestone deleted right after it's written -- including the one
        from the very save that just ran -- leaving only the single
        "latest" checkpoint as a rollback point, which is often already
        past the point of numerical drift. Keeping every eval-interval
        milestone trades disk space for the ability to roll back to a
        recent, still-healthy checkpoint (see the [DISK ALERT] warning in
        the 3D override for space monitoring).
        """
        if keep_every is None:
            keep_every = int(getattr(self.args, "eval_interval", 1000)) or 1000
        for f in self.run_dir.glob("training_state_it*.pt"):
            try:
                it_num = int(f.stem.split("it")[-1])
                if it_num % keep_every != 0:
                    f.unlink()
            except (ValueError, IndexError):
                continue

    @staticmethod
    def _ema_source(m: nn.Module) -> nn.Module:
        """
        Unwrap DataParallel/DDP/GlowStepWrapper before deep-copying for an
        EMA model.

        Deep-copying a DDP-wrapped module directly is risky -- DDP holds
        process-group/reducer state that isn't generally deepcopy-safe --
        and pointless even when it works: EMA models are only ever used for
        eval/sampling (no backward), so they never need DataParallel's
        thread-scatter or DDP's gradient-sync machinery, just a plain
        forward pass on a single device.

        We unwrap two levels for a wrapped model -- m.module (the
        GlowStepWrapper) then .model (the bare MultiscaleFlow) -- not just
        one. GlowStepWrapper exists solely to give DataParallel/DDP a
        forward() with the (log_prob, z_flat) tuple signature needed for
        scatter/gather; it contributes no parameters of its own. Stopping
        at GlowStepWrapper left EMA state_dicts keyed with a "model."
        prefix that clean_sd (fully stripped by _strip_dp_prefix) doesn't
        have, breaking resume. MultiscaleFlow has its own native
        log_prob()/sample(), so unwrapping all the way down is safe for
        every EMA use site.
        """
        return m.module.model if isinstance(m, (GlowDataParallel, GlowDDP)) else m

    def _load_model_state(self, m: nn.Module, sd: dict) -> None:
        """Load state dict into m, stripping DataParallel/DDP prefixes from sd."""
        clean_sd = self._strip_dp_prefix(sd)
        if isinstance(m, (GlowDataParallel, GlowDDP)):
            m.module.model.load_state_dict(clean_sd)
        else:
            m.load_state_dict(clean_sd)

    def _maybe_resume(self, args) -> int:
        """Return start_iter (1 if fresh, resumed iter otherwise)."""
        # Set unconditionally so the post-reset re-warmup check in train()
        # (getattr(self, "_opt_reset_at_iter", None)) has a defined value
        # on every path, not just the one that actually resets optimizer
        # state below.
        self._opt_reset_at_iter = None
        resume_path = None
        if args.resume:
            rp = Path(args.resume)
            if not rp.exists():
                raise FileNotFoundError(f"--resume file not found: {rp}")
            resume_path = rp
        elif args.auto_resume and self.state_path.exists():
            resume_path = self.state_path

        if resume_path is None:
            return 1

        # First pass (CPU): read config
        blob_cpu = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = blob_cpu.get("config", {})
        if ckpt_cfg and "num_views" not in ckpt_cfg and "modalities" in ckpt_cfg:
            try:
                args.num_views = len(ckpt_cfg.get("modalities") or [])
            except Exception:
                pass

        arch_keys = [
            "num_views", "H", "W", "L", "K", "hidden", "base",
            "glowbase_logscale_factor", "glowbase_min_log", "glowbase_max_log",
            "scale_map", "scale_cap", "net_actnorm",
        ]
        if args.use_ckpt_config and ckpt_cfg:
            for k in arch_keys:
                if k in ckpt_cfg:
                    setattr(args, k, ckpt_cfg[k])
        mismatches = [
            k for k in arch_keys
            if k in ckpt_cfg and getattr(args, k, None) != ckpt_cfg[k]
        ]
        if args.use_ckpt_config and mismatches:
            print("[resume] arch overrides:", {k: (getattr(args, k), ckpt_cfg[k]) for k in mismatches})

        # Second pass (target device)
        blob = torch.load(resume_path, map_location=self.dev, weights_only=False)
        start_iter = int(blob.get("iter", 1))

        # Optimizer
        #
        # A checkpoint saved under a different optimizer *class* (e.g. an
        # older run's Adamax, before the AdamW switch) stores per-param
        # state keyed 'exp_inf' instead of AdamW's 'exp_avg_sq'.
        # load_state_dict() itself doesn't validate key names against the
        # target optimizer's step() -- it happily copies the dict over --
        # so a mismatch wouldn't surface here, it would KeyError much
        # later inside .step(), deep into training. Detect it up front
        # and fall back to fresh optimizer state (keeping the checkpoint's
        # lr/betas/eps/weight_decay) instead of risking that.
        def _copy_opt_hparams(saved_blob):
            try:
                g0 = saved_blob["opt"]["param_groups"][0]
                for k in ("lr", "betas", "eps", "weight_decay"):
                    if k in g0:
                        for g in self.opt.param_groups:
                            g[k] = g0[k]
            except Exception:
                pass

        saved_state = blob.get("opt", {}).get("state", {})
        current_needs_avg_sq = isinstance(self.opt, (torch.optim.Adam, torch.optim.AdamW))
        opt_state_incompatible = (
            current_needs_avg_sq
            and bool(saved_state)
            and not any("exp_avg_sq" in v for v in saved_state.values())
        )

        if opt_state_incompatible:
            print(
                "[resume] checkpoint optimizer state looks like it's from a "
                "different optimizer class (e.g. Adamax); starting optimizer "
                "moment buffers fresh instead of loading incompatible state."
            )
            _copy_opt_hparams(blob)
            # Flag the reset so train()'s post-reset re-warmup can damp the
            # applied lr for the first few hundred iterations -- a fresh
            # AdamW's t=1 bias correction otherwise lands a full-nominal-lr
            # update on every parameter at once, which a model already
            # trained this far can't absorb (see train() for the full
            # explanation).
            self._opt_reset_at_iter = start_iter
        else:
            try:
                self.opt.load_state_dict(blob["opt"])
            except Exception as e:
                print(f"[resume] optimizer not loaded ({e}); using fresh.")
                _copy_opt_hparams(blob)
                self._opt_reset_at_iter = start_iter

        # Scaler
        if self.scaler is not None and "scaler" in blob:
            try:
                self.scaler.load_state_dict(blob["scaler"])
                print("[resume] restored GradScaler state")
            except Exception as e:
                print(f"[resume] GradScaler not loaded ({e}); starting fresh")

        # Warmup scheduler
        if self.warm and blob.get("warm") is not None:
            self.warm.load_state_dict(blob["warm"])

        # Model weights
        if blob.get("models") is not None:
            for m, sd in zip(self.models, blob["models"]):
                self._load_model_state(m, sd)

        # EMA weights
        if args.ema and blob.get("ema") is not None:
            self.ema_models = [
                copy.deepcopy(self._ema_source(m)).eval().to(dtype=torch.float32, device=self.dev)
                for m in self.models
            ]
            for em in self.ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            for em, sd in zip(self.ema_models, blob["ema"]):
                self._load_model_state(em, sd)

        # Projectors
        if blob.get("proj") is not None and self.projectors is not None:
            try:
                self.projectors.load_state_dict(blob["proj"])
                tqdm.write("[resume] restored projectors")
            except Exception as e:
                tqdm.write(f"[resume] projectors not loaded: {e}")

        # Kendall scalars
        if blob.get("kendall") is not None and self.s_nll is not None:
            try:
                kd = blob["kendall"]
                if kd.get("s_nll")   is not None: self.s_nll.data.fill_(float(kd["s_nll"]))
                if kd.get("s_align") is not None: self.s_align.data.fill_(float(kd["s_align"]))
                tqdm.write(
                    f"[resume] Kendall s_nll={float(self.s_nll):.3f}, "
                    f"s_align={float(self.s_align):.3f}"
                )
            except Exception as e:
                tqdm.write(f"[resume] Kendall scalars not loaded: {e}")

        tqdm.write(f"[resume] from {resume_path} @ iter {start_iter}")
        return start_iter

    def _prime_all_latent_shapes(self) -> None:
        try:
            batch = next(iter(self.train_loader))
        except StopIteration:
            return
        xs = _extract_views_from_batch(batch, num_views=len(self.models))
        with torch.no_grad():
            for vi, m in enumerate(self.models):
                xb = xs[vi][:1]
                if xb.ndim == 3:
                    # (B, H, W) -> (B, 1, H, W)
                    xb = xb.unsqueeze(1)
                elif xb.ndim == 4 and xb.shape[1] not in (1, 3):
                    # Raw 3-D volume batch without a channel dim: (B, H, W, D) -> (B, 1, H, W, D)
                    xb = xb.unsqueeze(1)
                xb = xb.to(dtype=torch.float32, device=self.dev)
                p = next(m.parameters(), None)
                if p is not None and xb.dtype != p.dtype:
                    xb = xb.to(p.dtype)
                with torch.amp.autocast(device_type=self.dev.type, enabled=False):
                    try:
                        _ = m.log_prob(xb)
                    except Exception as ex:
                        print(f"[prime] base view{vi} failed: {ex}")
                    if self.ema_models is not None:
                        try:
                            _ = self.ema_models[vi].log_prob(xb)
                        except Exception as ex:
                            print(f"[prime] ema view{vi} failed: {ex}")

    # ------------------------------------------------------------------
    # DDP: cross-rank agreement on "skip this step" decisions
    # ------------------------------------------------------------------

    def _sync_skip_flag(self, is_bad: bool) -> bool:
        """
        Under DDP, each rank sees a different micro-batch (DistributedSampler
        shards the data), so a NaN/anomaly decision computed from local data
        -- bad_batch, non-finite L_nll/loss_total -- can differ across ranks.
        If rank 0 decides to `continue` (skip backward()) while rank 1
        proceeds to call it, backward()'s NCCL all-reduce is a collective op:
        rank 1 blocks forever waiting for a contribution rank 0 never sends.
        That's a silent hang, not a crash -- much worse than the bug it would
        be masking.

        This all-reduces a boolean "I want to skip" flag with MAX, so if
        *any* rank wants to skip, *all* ranks skip together -- every rank
        takes the same branch every iteration, guaranteeing backward() is
        either called by everyone or no one.

        (The post-backward non-finite-grad-norm check further down doesn't
        need this: DDP's backward() has already all-reduced gradients across
        ranks by the time we compute the norm, so that value is already
        identical on every rank -- no separate sync required there.)
        """
        if not self.is_ddp:
            return is_bad
        flag = torch.tensor(1.0 if is_bad else 0.0, device=self.dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _reload_last_checkpoint_inplace(self) -> bool:
        """
        Reload model/EMA/optimizer/scaler/warmup state IN PLACE from the
        last "latest" checkpoint on disk (self.state_path), WITHOUT
        touching the training-loop iteration counter `it`.

        Used by the rollback watchdog in train(): the per-step rollback
        (params + optimizer state restored from an in-memory snapshot
        taken one iteration ago) can only undo the *most recent* step --
        if the forward pass is already permanently non-finite by the time
        that snapshot was taken (see the long comment above the snapshot
        in train() for why this happens despite the delta/replay checks),
        every retry re-enters the same broken state and the per-step
        rollback can never escape it on its own (observed in practice as
        hundreds to thousands of consecutive "[anomaly] skipping..."
        messages with zero progress). The last checkpoint written to disk
        is always a point where a full eval pass already succeeded (see
        save_checkpoint() / train()'s eval branch, which only runs after a
        *successful* step), so it's the nearest known-good state to fall
        back to automatically instead of requiring an operator to notice
        the stall, kill the job, and restart with --resume by hand.

        Deliberately does NOT rewind `it`: the outer loop is a plain
        `for it in range(start_iter, max_iter+1)`, so rewinding it would
        require restructuring the loop into a manually-driven `while`.
        Reloading weights/optimizer state to a healthy point while letting
        `it` (and therefore the lr schedule) continue climbing is a
        smaller, lower-risk change -- the model becomes healthy again and
        training makes real progress again, just measured against a
        slightly more-advanced lr/warmup position than the reloaded
        weights originally saw. The existing lr_backoff mechanism already
        damps the applied lr for a while after any rollback streak, which
        covers this gap.

        Returns True if a checkpoint existed and was reloaded, False if
        self.state_path doesn't exist yet (e.g. the run stalled before its
        first eval_interval) -- in that case there is no fallback and the
        caller should let the existing skip-forever behavior continue
        rather than raising.
        """
        if not self.state_path.exists():
            return False

        blob = torch.load(self.state_path, map_location=self.dev, weights_only=False)

        if blob.get("models") is not None:
            for m, sd in zip(self.models, blob["models"]):
                self._load_model_state(m, sd)

        if self.args.ema and blob.get("ema") is not None and self.ema_models is not None:
            for em, sd in zip(self.ema_models, blob["ema"]):
                self._load_model_state(em, sd)

        try:
            self.opt.load_state_dict(blob["opt"])
        except Exception as e:
            tqdm.write(
                f"[watchdog] optimizer state not reloaded ({e}); keeping "
                f"current optimizer state alongside the reloaded weights."
            )

        if self.scaler is not None and blob.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(blob["scaler"])
            except Exception:
                pass

        if self.warm and blob.get("warm") is not None:
            try:
                self.warm.load_state_dict(blob["warm"])
            except Exception:
                pass

        ckpt_iter = int(blob.get("iter", -1))
        tqdm.write(
            f"[watchdog] stuck rollback streak detected -- reloaded last "
            f"known-good checkpoint (saved at iter {ckpt_iter}) from "
            f"{self.state_path}; continuing forward without rewinding the "
            f"iteration counter."
        )
        return True

    # Number of consecutive rollbacks (per-step skip or post-step undo,
    # tracked by the same `rollback_streak` counter) after which the
    # per-step rollback is treated as having failed to recover and the
    # watchdog falls back to reloading the last checkpoint from disk
    # instead of continuing to skip indefinitely. Deliberately well above
    # the existing lr-backoff threshold (15) -- that backoff is tried
    # first and given a real chance to work before this heavier fallback
    # fires.
    ROLLBACK_WATCHDOG_LIMIT = 50

    # Sanity ceiling on the raw per-iteration loss magnitude for an
    # otherwise-accepted step. This problem's healthy loss/bpd range is
    # roughly [-5, 5] (see the [eval] avg_bpd lines in the training log);
    # a step whose loss lands orders of magnitude beyond that (e.g. the
    # 76433.73 observed in practice) is not a healthy update just because
    # it happened to pass the finiteness/delta/replay checks. Counted
    # toward the same rollback_streak/watchdog machinery as an explicitly
    # rejected step -- see the comment at its use site in train().
    BAD_LOSS_MAGNITUDE = 100.0

    def _watchdog_check(self, rollback_streak: int, lr_backoff: float):
        """
        Call after every rollback_streak increment. Once the streak
        crosses ROLLBACK_WATCHDOG_LIMIT, the per-step rollback has
        provably failed to recover on its own -- fall back to the last
        on-disk checkpoint (see _reload_last_checkpoint_inplace) instead
        of skipping forever. Returns the (possibly reset) streak/backoff.
        """
        if rollback_streak > 0 and rollback_streak % self.ROLLBACK_WATCHDOG_LIMIT == 0:
            reloaded = self._reload_last_checkpoint_inplace()
            if reloaded:
                rollback_streak = 0
                # Cool down harder than a normal post-rollback backoff:
                # the model was just healed from a confirmed permanent
                # corruption, so give it a wider margin before trusting
                # the full nominal lr again. Heals back up via the
                # existing lr_backoff *= 1.2 on each later successful step.
                lr_backoff = min(lr_backoff, 0.1)
            else:
                tqdm.write(
                    f"[watchdog] {rollback_streak} consecutive rollbacks "
                    f"and no checkpoint on disk yet to fall back to -- "
                    f"continuing to skip and hoping for recovery."
                )
        return rollback_streak, lr_backoff

    def train(self) -> None:  # noqa: C901
        args      = self.args
        dev       = self.dev
        models    = self.models
        n_views   = len(models)
        n_dims    = int(np.prod(getattr(args, "input_shape", (1, args.H, args.W))))

        alpha             = float(args.smooth_alpha)
        ema_loss_disp     = None
        ema_sum_bpd_disp  = None
        ema_bpd_views_disp = [None] * n_views

        # DistributedSampler needs set_epoch() before each new epoch's
        # iterator is created, or every rank re-draws the *same* shuffled
        # shard every epoch (the sampler's shuffle is seeded by the epoch
        # number). getattr(...) is a no-op for the non-DDP / non-sampler
        # case (plain shuffle=True DataLoader has no .sampler.set_epoch).
        train_epoch = 0
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(train_epoch)
        train_iter      = iter(self.train_loader)
        input_data_saved = False

        # Rollback-streak LR backoff.
        #
        # The rollback safety net (params + optimizer state + delta +
        # replay checks) reliably prevents corruption, but on its own it
        # has no way to *escape* a fragile region: it keeps retrying at
        # the same lr, and if that lr is what's causing the blowup, nothing
        # about a plain retry changes the outcome -- observed in practice
        # as dozens to hundreds of consecutive rollbacks with zero
        # progress (e.g. iter 3163 onward). Track a streak counter and, once
        # rollbacks start piling up, automatically back off the *applied*
        # lr (same restore-after-step mechanism as the post-reset
        # re-warmup) so later retries are progressively gentler instead of
        # identical. Recovers gradually back to full lr on the next
        # successful step, and resets fully once training is clearly
        # healthy again.
        rollback_streak = 0
        lr_backoff = 1.0

        if self.rank == 0:
            tqdm.write(
                f"[info] training {n_views} view(s); "
                f"params/view: {[n_params(m) for m in models]}"
            )
            pbar = tqdm(
                total=args.max_iter,
                initial=self.start_iter - 1,
                dynamic_ncols=True,
                desc="train",
            )

        for it in range(self.start_iter, args.max_iter + 1):
            grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
            self.opt.zero_grad(set_to_none=True)

            # Logging accumulators
            loss_acc  = torch.tensor(0.0, device=dev, dtype=torch.float32)
            align_acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
            bpd_acc   = 0.0
            bpd_views_acc: Optional[List[float]] = None

            bad_update = False
            x_last     = None
            w_nll, w_align = 1.0, 0.0

            # ── Gradient accumulation loop ────────────────────────────
            for micro in range(grad_accum):
                # Fetch next batch (restart iterator if exhausted)
                try:
                    x = next(train_iter)
                except StopIteration:
                    train_epoch += 1
                    if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                        self.train_loader.sampler.set_epoch(train_epoch)
                    train_iter = iter(self.train_loader)
                    x = next(train_iter)
                    # Epoch boundary: flush GPU caches
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                x_last = x

                L_nll         = torch.tensor(0.0, device=dev, dtype=torch.float32)
                curr_bpd_views: List[float] = []
                sum_bpd       = 0.0
                lat_flat: List[torch.Tensor] = []

                amp_ctx = (
                    torch.amp.autocast(dev.type, dtype=self.amp_dtype)
                    if self.amp_enabled else nullcontext()
                )

                xs_train = _extract_views_from_batch(x, num_views=n_views)

                with amp_ctx:
                    bad_batch = False
                    for vi, m in enumerate(models):
                        x_v = self.extract_view(x, vi, dev)

                        # Diagnostic only (does not alter control flow).
                        #
                        # Every optimizer-side fix so far (Adamax->AdamW,
                        # tighter grad_clip, post-reset re-warmup) failed
                        # to move the failure point, and a *fresh* run
                        # (no resumed weights, no Adamax history to carry
                        # fragility) still broke in nearly the same
                        # iteration range as every resumed attempt. With
                        # seed=0 fixing the DistributedSampler's shuffle
                        # order and ImageDataset's augmentation draws, the
                        # sequence of samples hitting the model at a given
                        # iteration is nearly identical run to run --
                        # consistent with a specific pathological sample
                        # (corrupt voxels, or an extreme augmentation draw
                        # on one of only 50 CuratedCohort subjects) rather
                        # than an optimizer dynamics problem. Log it if the
                        # *input* itself is already non-finite or extreme,
                        # before it ever reaches the model, so we can
                        # confirm or rule this out directly instead of
                        # continuing to guess on the optimizer side.
                        if not torch.isfinite(x_v).all():
                            n_bad = int((~torch.isfinite(x_v)).sum().item())
                            tqdm.write(
                                f"[data] non-finite INPUT at view {vi}, iter {it}: "
                                f"{n_bad}/{x_v.numel()} elements non-finite"
                            )
                        else:
                            x_min = float(x_v.min().item())
                            x_max = float(x_v.max().item())
                            if abs(x_min) > 1e4 or abs(x_max) > 1e4:
                                tqdm.write(
                                    f"[data] extreme INPUT range at view {vi}, "
                                    f"iter {it}: min={x_min:.4g} max={x_max:.4g}"
                                )

                        # Forward pass. Must go through __call__ (m(...))
                        # for both GlowDataParallel and GlowDDP -- calling
                        # .log_prob()/.inverse_and_log_det() directly (the
                        # else branch) bypasses DDP's forward(), which is
                        # what registers the autograd hooks DDP needs to
                        # all-reduce gradients in backward(). Skipping that
                        # wouldn't just be a perf issue, it would silently
                        # stop gradient synchronization across ranks.
                        if isinstance(m, (GlowDataParallel, GlowDDP)):
                            logp_v, zflat = m(x_v.float())
                        else:
                            logp_v = m.log_prob(x_v.float())
                            z_v, _ = m.inverse_and_log_det(x_v.float())
                            zflat  = flatten_latents(z_v)

                        if not torch.isfinite(logp_v).all():
                            tqdm.write(f"[nan] non-finite logp at view {vi}, iter {it}")
                            bad_batch = True
                            # --- Strict cleanup to prevent nanobind leaks ---
                            del x_v, logp_v
                            gc.collect()
                            break

                        bpd_v = bits_per_dim(logp_v, n_dims).mean()
                        L_nll = L_nll + bpd_v
                        curr_bpd_views.append(bpd_v.item())
                        sum_bpd += bpd_v.item()
                        lat_flat.append(torch.nan_to_num(zflat))

                        # Explicit cleanup of view tensor and intermediates
                        del x_v, logp_v, zflat
                        # Note: z_v and bpd_v are referenced by the computation graph;
                        # del here only drops Python refs — backward() is still intact.

                local_bad = bad_batch or not torch.isfinite(L_nll) or abs(L_nll.item()) > 1e7
                if self._sync_skip_flag(local_bad):
                    tqdm.write(
                        f"[anomaly] skipping iter {it} "
                        f"(bad_batch={bad_batch}, L_nll={L_nll.item():.2f}"
                        + (", flagged by another rank" if (self.is_ddp and not local_bad) else "")
                        + ")"
                    )
                    bad_update = True
                    # Cleanup this micro-batch before breaking
                    del xs_train, lat_flat
                    gc.collect()
                    break

                # Alignment loss + combined loss
                loss_total, L_align, w_nll, w_align = self.align_mgr.compute(
                    lat_flat=lat_flat,
                    L_nll=L_nll,
                    it=it,
                    s_nll=self.s_nll,
                    s_align=self.s_align,
                )

                local_bad = not torch.isfinite(loss_total)
                if self._sync_skip_flag(local_bad):
                    tqdm.write(
                        f"[nan] loss_total non-finite at iter {it}; skipping"
                        + (" (flagged by another rank)" if (self.is_ddp and not local_bad) else "")
                    )
                    bad_update = True
                    del xs_train, lat_flat
                    gc.collect()
                    break

                # Backward
                loss_scaled = loss_total / float(grad_accum)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                # Accumulate for logging
                loss_acc  = loss_acc  + loss_total.detach().float()
                align_acc = align_acc + L_align.detach().float()
                bpd_acc  += float(sum_bpd)
                if bpd_views_acc is None:
                    bpd_views_acc = [0.0] * len(curr_bpd_views)
                for _i in range(len(curr_bpd_views)):
                    bpd_views_acc[_i] += float(curr_bpd_views[_i])

                # ── Strict per-micro-step memory cleanup ─────────────
                del xs_train, lat_flat
                gc.collect()
                # ─────────────────────────────────────────────────────

            # ── End of gradient accumulation ─────────────────────────

            if bad_update:
                # Count toward the same streak as post-step rollbacks below
                # -- in practice the two interleave during a stuck stretch
                # (forward already non-finite on some batches, others make
                # it to a step that then gets rolled back), and only
                # tracking one type would undercount how long training's
                # actually been stuck and delay the lr backoff.
                rollback_streak += 1
                if rollback_streak % 15 == 0:
                    lr_backoff = max(0.02, lr_backoff * 0.5)
                rollback_streak, lr_backoff = self._watchdog_check(rollback_streak, lr_backoff)
                self.opt.zero_grad(set_to_none=True)
                continue

            # Gradient clip + optimizer step
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.opt)
            all_params = [p for g in self.opt.param_groups for p in g["params"]]
            total_norm = torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=float(getattr(args, "grad_clip", 2.0))
            )

            # Explicit finite-gradient guard.
            #
            # The forward-pass anomaly checks above only inspect the loss
            # value, not the gradients produced by backward(). GradScaler
            # normally provides this protection (it skips optimizer.step()
            # on non-finite grads) -- but self.scaler is only enabled for
            # fp16 (see setup()); for bf16 (our default under
            # --precision mixed) or fp32 it stays disabled, since bf16
            # doesn't need loss-scaling. That leaves no safety net here: a
            # single NaN/Inf gradient would otherwise permanently poison
            # Adamax's moving averages (exp_avg/exp_inf) for the affected
            # parameters, producing NaN logp on every subsequent iteration
            # with no recovery (observed in practice around iter ~2335).
            if not torch.isfinite(total_norm):
                tqdm.write(
                    f"[anomaly] skipping optimizer step at iter {it} "
                    f"(non-finite grad norm={float(total_norm):.2f})"
                )
                # Count toward the same watchdog streak as the other anomaly
                # branches (bad_update above, post-step rollback below). Previously
                # this branch skipped straight to `continue` without touching
                # rollback_streak at all -- every repeat of this exact message
                # (the one seen looping hundreds of times in practice) counted for
                # nothing, so the consecutive-streak watchdog could never reach its
                # threshold no matter how long training stayed stuck this way. No
                # extra DDP sync needed here (unlike bad_update/local_bad_params):
                # DDP's backward() already all-reduces gradients across ranks before
                # this point, so total_norm's finiteness already agrees across ranks.
                rollback_streak += 1
                if rollback_streak % 15 == 0:
                    lr_backoff = max(0.02, lr_backoff * 0.5)
                rollback_streak, lr_backoff = self._watchdog_check(rollback_streak, lr_backoff)
                self.opt.zero_grad(set_to_none=True)
                continue






            # Post-step parameter-finiteness guard.
            #
            # isfinite(total_norm) above only proves the *gradient* going
            # into Adam was finite -- it says nothing about whether the
            # *parameters* Adam produces from that gradient stay
            # well-behaved. A small, cleanly-clipped gradient can still
            # nudge a numerically sensitive parameter into a regime where
            # the *next* forward pass overflows internally. Unlike the
            # bad_batch/non-finite-grad-norm checks above (which catch
            # trouble before stepping), this failure mode has no recovery
            # once it happens: forward becomes permanently non-finite, so
            # every later iteration just re-enters those earlier skip
            # branches forever without ever attempting a fresh update --
            # a training run that stalls for good after a single bad step,
            # often long before the next checkpoint (eval_interval-gated)
            # can save a clean state to fall back to. Snapshot params
            # before stepping and, if the step leaves any parameter
            # non-finite, roll back and treat it like the other skip
            # cases instead of permanently poisoning the model.
            param_snapshot = [p.detach().clone() for p in all_params]

            # Snapshot Adamax's per-parameter moment state too (exp_avg,
            # exp_inf, step). .step() mutates self.opt.state[p] in place
            # unconditionally -- rejecting the resulting *parameters*
            # below does nothing to undo that. Left unrepaired, this is
            # self-reinforcing: a parameter whose Adamax exp_inf
            # (infinity-norm accumulator) has decayed near zero produces
            # a huge update the instant it next sees a gradient; we reject
            # the parameters, but exp_inf/exp_avg already absorbed that
            # gradient, so the very next retry is primed to reproduce the
            # same blow-up -- observed in practice as the identical
            # rollback message firing every single iteration with zero
            # progress. Restore optimizer state in lockstep with
            # parameters so a rejected step is a true no-op, not just a
            # cosmetic one.
            opt_state_snapshot = {
                p: {k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in self.opt.state[p].items()}
                for p in all_params if p in self.opt.state
            }

            # Post-optimizer-reset re-warmup.
            #
            # When _maybe_resume() falls back to a fresh optimizer state
            # (e.g. the Adamax -> AdamW switch: incompatible moment
            # buffers, see _maybe_resume), step 1 of a brand-new AdamW is
            # not "gentle" just because args.lr/warmup have already
            # damped the *nominal* lr down to ~2e-6 by this point in the
            # outer warmup schedule. Adam-family bias correction at t=1
            # normalizes exp_avg by (1-beta1^1) and exp_avg_sq by
            # (1-beta2^1), which makes the *first* update for every
            # parameter with a nonzero gradient land at roughly the full
            # nominal lr magnitude, uniformly across all ~742M
            # parameters simultaneously, regardless of how small that
            # parameter's gradient history "should" make its effective
            # step. On a model already trained 1000 iterations (not a
            # fresh random init, where this same shock is normal and
            # harmless), that uniform simultaneous nudge compounds through
            # L=5 levels of multiplicative/exponential flow math into an
            # exploded forward pass -- reproducing identically on every
            # retry since it's structural (driven by t=1 bias correction),
            # not data-dependent, so the earlier per-step rollback just
            # spins forever without progress. Temporarily scale down the
            # *applied* lr (restored immediately after step()) for a
            # short window following any optimizer-state reset, so the
            # moment estimates get a chance to build up real history
            # before bias correction lets the full nominal lr through.
            reset_at = getattr(self, "_opt_reset_at_iter", None)
            reset_rewarmup_iters = 300
            reset_damp = 1.0
            if reset_at is not None and (it - reset_at) < reset_rewarmup_iters:
                reset_damp = 0.02 + 0.98 * ((it - reset_at) / reset_rewarmup_iters)

            damp = min(reset_damp, lr_backoff)
            damped_lrs = None
            if damp < 1.0:
                damped_lrs = [g["lr"] for g in self.opt.param_groups]
                for g in self.opt.param_groups:
                    g["lr"] = g["lr"] * damp

            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            if damped_lrs is not None:
                for g, lr0 in zip(self.opt.param_groups, damped_lrs):
                    g["lr"] = lr0

            local_bad_params = any(not torch.isfinite(p).all() for p in all_params)
            reason = "non-finite parameter(s) after step"

            # A forward replay on x_last (below) only proves the new
            # parameters are safe *for the exact batch that produced their
            # gradient* -- nearly tautological, since the step was
            # computed to fit that batch. It does not prove they're safe
            # for the *next* (different, randomly-augmented) batch, and in
            # practice a step has passed that replay check and still gone
            # on to permanently break forward one or two iterations later
            # (iter 1214 here: replay-clean step committed, then iter 1215
            # onward stuck non-finite forever). Adamax's known failure
            # mode -- a parameter whose exp_inf (infinity-norm
            # accumulator) has decayed near zero produces a huge update
            # the instant it next sees a gradient -- shows up as an
            # implausibly large *single-step* jump for that parameter
            # specifically, regardless of whether that jump happens to
            # still "read fine" on x_last. Catch that directly and
            # data-independently: no healthy Adamax step at this LR
            # should move any single parameter by more than ~1.0 in one
            # iteration, so treat one that does as bad without needing a
            # forward pass to prove it.
            if not local_bad_params:
                max_delta = max(
                    (p.detach() - snap).abs().max().item()
                    for p, snap in zip(all_params, param_snapshot)
                )
                if max_delta > 1.0:
                    local_bad_params = True
                    reason = f"implausibly large single-step parameter update (max |delta|={max_delta:.3g})"

            # Belt-and-suspenders: a *finite* parameter is not the same
            # guarantee as a *safe* one. Adam can land a parameter on a
            # value that is technically finite yet numerically extreme
            # enough that some internal exp()/log()/reciprocal deeper in
            # the flow overflows on the very next forward pass -- exactly
            # the failure observed at iter 1318 (params passed a
            # finiteness check, forward still went permanently non-finite
            # one iteration later). The only way to actually catch that
            # is to try a forward pass with the freshly-stepped params
            # before committing to them, so do a cheap forward-only replay
            # (no backward, no data augmentation redraw -- reuses x_last)
            # right here. Kept as a second line of defense alongside the
            # delta check above, since the two catch different failure
            # shapes.
            if not local_bad_params and x_last is not None:
                val_amp_ctx = (
                    torch.amp.autocast(dev.type, dtype=self.amp_dtype)
                    if self.amp_enabled else nullcontext()
                )
                with torch.no_grad(), val_amp_ctx:
                    for vi, m in enumerate(models):
                        x_v_chk = self.extract_view(x_last, vi, dev)
                        if isinstance(m, (GlowDataParallel, GlowDDP)):
                            logp_chk, _ = m(x_v_chk.float())
                        else:
                            logp_chk = m.log_prob(x_v_chk.float())
                        # isfinite() alone isn't enough: a wildly exploded
                        # (but still finite) bpd -- ~1e31, seen in
                        # practice -- sails right through an isfinite()
                        # check yet is exactly the kind of blown-up state
                        # that poisons every later iteration. Apply the
                        # same >1e7 magnitude guard the main loop already
                        # uses for L_nll (see local_bad below) so this
                        # replay check can't be fooled by "finite but
                        # absurd" the way the plain finiteness checks were.
                        bpd_chk = bits_per_dim(logp_chk, n_dims).mean()
                        chk_bad = (
                            not torch.isfinite(bpd_chk).all()
                            or abs(bpd_chk.item()) > 1e7
                        )
                        del x_v_chk, logp_chk, bpd_chk
                        if chk_bad:
                            local_bad_params = True
                            reason = "post-step forward replay non-finite or exploded"
                            break

            if self._sync_skip_flag(local_bad_params):
                rollback_streak += 1
                backoff_note = ""
                if rollback_streak % 15 == 0:
                    prev_backoff = lr_backoff
                    lr_backoff = max(0.02, lr_backoff * 0.5)
                    if lr_backoff != prev_backoff:
                        backoff_note = (
                            f"; {rollback_streak} in a row, backing lr off to "
                            f"{lr_backoff:.3g}x"
                        )
                tqdm.write(
                    f"[anomaly] rolling back optimizer step at iter {it} "
                    f"({reason})"
                    + (", flagged by another rank" if (self.is_ddp and not local_bad_params) else "")
                    + backoff_note
                )
                with torch.no_grad():
                    for p, snap in zip(all_params, param_snapshot):
                        p.data.copy_(snap)
                    for p, st_snap in opt_state_snapshot.items():
                        for k, v in st_snap.items():
                            if torch.is_tensor(v):
                                self.opt.state[p][k].copy_(v)
                            else:
                                self.opt.state[p][k] = v
                del param_snapshot, opt_state_snapshot
                rollback_streak, lr_backoff = self._watchdog_check(rollback_streak, lr_backoff)
                self.opt.zero_grad(set_to_none=True)
                continue
            del param_snapshot, opt_state_snapshot

            # A successful step: let the rollback streak/backoff heal
            # instead of resetting instantly to full lr, in case the same
            # fragile region is still nearby.
            rollback_streak = 0
            if lr_backoff < 1.0:
                lr_backoff = min(1.0, lr_backoff * 1.2)

            # Use last micro-batch for EMA ActNorm warmup
            x = x_last

            # Averaged metrics
            curr_loss      = float(loss_acc.item())  / float(grad_accum)
            L_align_log    = float(align_acc.item()) / float(grad_accum)
            sum_bpd        = bpd_acc / float(grad_accum)
            curr_bpd_views = [v / float(grad_accum) for v in (bpd_views_acc or [])]

            # Magnitude-based watchdog trigger.
            #
            # The consecutive-rollback counter (above, reset to 0 a few
            # lines up) only fires on steps that get explicitly REJECTED
            # (non-finite grad, non-finite params, oversized delta, or a
            # bad forward replay). A step can pass every one of those
            # checks yet still land on a catastrophically bad-but-finite
            # loss -- observed in practice (iter 2027, loss=76433.73,
            # against a healthy range of roughly [-5, 5] for this
            # problem). Such a step is accepted as "successful" above,
            # which resets rollback_streak to 0 and heals lr_backoff
            # upward, even though nothing about it was healthy. If this
            # then alternates with stretches of rejected rollbacks that
            # individually never reach ROLLBACK_WATCHDOG_LIMIT before the
            # next bad-but-accepted step resets the count, the
            # consecutive-streak watchdog can never fire at all --
            # training stays stuck in a terrible-loss regime indefinitely
            # with no automatic recovery (observed in practice: repeated
            # "skipping optimizer step (non-finite grad norm=inf)" bursts
            # that never individually reach 50-in-a-row). Catch this
            # directly: any accepted step whose loss magnitude is far
            # outside the sane range counts toward the same watchdog
            # streak regardless of the per-step checks it happened to
            # pass.
            #
            # MUST go through _sync_skip_flag, same as every other
            # anomaly branch above: under DDP each rank sees a different
            # micro-batch, so curr_loss differs across ranks. An earlier
            # version of this check incremented rollback_streak (and
            # could trigger _reload_last_checkpoint_inplace) locally,
            # per-rank, with no synchronization -- if only the rank that
            # happened to see the bad batch reloaded the checkpoint while
            # the other rank did not, the two ranks' weights permanently
            # diverge. All-reducing gradients from mismatched replicas
            # after that point is mathematically incoherent and produced
            # exactly the symptom observed in practice: an unrecoverable
            # "non-finite grad norm=inf" stall for the rest of the run,
            # immediately following a magnitude-triggered watchdog event.
            local_bad_loss = abs(curr_loss) > self.BAD_LOSS_MAGNITUDE
            if self._sync_skip_flag(local_bad_loss):
                rollback_streak += 1
                tqdm.write(
                    f"[watchdog] accepted step at iter {it} has an absurd "
                    f"loss magnitude (loss={curr_loss:.3g}"
                    + (", flagged by another rank" if (self.is_ddp and not local_bad_loss) else "")
                    + f"); counting it toward the rollback streak despite "
                    f"passing the per-step checks."
                )
                rollback_streak, lr_backoff = self._watchdog_check(rollback_streak, lr_backoff)

            # Lazy EMA init (after first successful update)
            if args.ema and self.ema_models is None:
                self.ema_models = [
                    copy.deepcopy(self._ema_source(m)).eval().to(dtype=torch.float32, device=dev)
                    for m in models
                ]
                for em in self.ema_models:
                    for p in em.parameters():
                        p.requires_grad_(False)
                with torch.no_grad():
                    for vi, (m, em) in enumerate(zip(models, self.ema_models)):
                        # Pre-existing off-by-one bug when m is wrapped
                        # (DataParallel/DDP): src.modules() then yields an
                        # extra leading entry (the wrapper itself) that
                        # dst.modules() (unwrapped em) doesn't have, so the
                        # zip() inside _copy_actnorm_state silently paired
                        # up the wrong ActNorm modules. Unwrap m the same
                        # way em was unwrapped (_ema_source) so both sides
                        # of the zip start from the same module structure.
                        _copy_actnorm_state(self._ema_source(m), em)
                        xv_real = self.extract_view(x, vi, dev)
                        warmup_actnorm_with_real_batch(em, xv_real)
                        del xv_real
                tqdm.write("[ema] initialized from base after first update")

            # EMA weight update
            if self.ema_models is not None:
                with torch.no_grad():
                    for em, m in zip(self.ema_models, models):
                        for p_em, p in zip(em.parameters(), m.parameters()):
                            p_em.data.mul_(args.ema_decay).add_(
                                p.data, alpha=1.0 - args.ema_decay
                            )

            # LR warmup step
            if self.warm is not None and it <= args.warmup_iters:
                self.warm.step()

            # Global step counter
            with self.global_step.get_lock():
                self.global_step.value += 1

            lr_now = self.opt.param_groups[0]["lr"]

            # EMA display metrics
            if ema_loss_disp is None:
                ema_loss_disp      = curr_loss
                ema_sum_bpd_disp   = sum_bpd
                ema_bpd_views_disp = list(curr_bpd_views)
            else:
                a = alpha
                ema_loss_disp    = (1.0 - a) * ema_loss_disp    + a * curr_loss
                ema_sum_bpd_disp = (1.0 - a) * ema_sum_bpd_disp + a * sum_bpd
                for i in range(n_views):
                    ema_bpd_views_disp[i] = (
                        (1.0 - a) * ema_bpd_views_disp[i] + a * curr_bpd_views[i]
                    )

            postfix = {
                "iter":  it,
                "loss":  f"{curr_loss:.4f}",
                "loss~": f"{ema_loss_disp:.4f}",
                "bpd":   f"{sum_bpd:.3f}",
                "bpd~":  f"{ema_sum_bpd_disp:.3f}",
                "lr":    f"{lr_now:.2e}",
                "align": f"{L_align_log:.4f}",
                "mode":  args.align,
                "w_nll": f"{w_nll:.2f}",
                "w_aln": f"{w_align:.2f}",
            }
            for i in range(n_views):
                postfix[f"v{i}"] = f"{curr_bpd_views[i]:.3f}/{ema_bpd_views_disp[i]:.3f}"
            # Progress bar, file writes (input grids, CSV, samples, plots,
            # checkpoints) are rank-0-only under DDP: every rank computes an
            # identical (synced-gradient) update, so having every rank also
            # write the same files would just race/duplicate for no benefit.
            if self.rank == 0:
                pbar.set_postfix(postfix)
                pbar.update(1)

                # One-time input data grid
                if not input_data_saved:
                    # _coerce_nchw_4d is defined in this module — call directly
                    eval_m = self.ema_models if self.ema_models else models
                    ok, err = self._save_input_grids(eval_m, it)
                    if ok:
                        tqdm.write(f"[samples] saved input data grids @ iter {it}")
                        input_data_saved = True
                    else:
                        tqdm.write(f"[warn] input data grid failed: {err}")

                # CSV row
                with open(self.csv_path, "a") as f:
                    f.write(f"{it},{curr_loss:.6f},{sum_bpd:.6f},{lr_now:.6g}\n")

            # Eval + checkpoint. _run_eval must be called by *every* rank
            # (it does a dist.broadcast internally to keep each rank's LR
            # scheduler in sync -- see its docstring); the rest is rank-0-only.
            if it % args.eval_interval == 0:
                self._run_eval(it, n_dims)
                if self.rank == 0:
                    self._run_sample_plots(it)
                    _save_metric_plots(self.csv_path, self.run_dir, remove_spikes=True)
                    self.save_checkpoint(it)
                if self.is_ddp:
                    dist.barrier()

            # ── End-of-iteration cleanup ──────────────────────────────
            del x, x_last
            gc.collect()
            # ─────────────────────────────────────────────────────────

        if self.rank == 0:
            pbar.close()
            print("Done. Run dir:", str(self.run_dir))
        if self.is_ddp:
            dist.barrier()
            dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def _run_eval(self, it: int, n_dims: int) -> None:
        # Only rank 0 iterates val_loader (it's unsharded -- every rank
        # would otherwise redundantly evaluate the identical full val set).
        # But self.plateau (ReduceLROnPlateau) drives self.opt's LR, and
        # each rank owns its own independent optimizer instance (DDP only
        # syncs gradients, not optimizer/scheduler state) -- so if only
        # rank 0 ever called plateau.step(), only rank 0's LR would get
        # reduced, silently desynchronizing per-rank optimizers over a long
        # run. Broadcast rank 0's avg_bpd so every rank's plateau scheduler
        # advances identically.
        args        = self.args
        dev         = self.dev
        eval_models = self.ema_models if self.ema_models else self.models

        if self.rank == 0:
            with torch.no_grad():
                bpd_acc         = []
                self._tmpl_by_view = [None] * len(eval_models)
                vbar = tqdm(total=10, leave=False, dynamic_ncols=True, desc=f"val@{it}")

                for j, batch_val in enumerate(self.val_loader):
                    for vi, m in enumerate(eval_models):
                        xv = self.extract_view(batch_val, vi, dev)
                        self._tmpl_by_view[vi] = xv
                        lp = m.log_prob(xv.float())
                        lp = torch.nan_to_num(lp, nan=-1e9, posinf=-1e9, neginf=-1e9)
                        bpd_acc.append(bits_per_dim(lp, n_dims).mean().item())
                        del xv
                    vbar.update(1)
                    if len(bpd_acc) >= 10:
                        break
                vbar.close()

                avg_bpd = float(np.mean(bpd_acc)) if bpd_acc else float("nan")
        else:
            avg_bpd = float("nan")

        if self.is_ddp:
            bpd_tensor = torch.tensor(avg_bpd, device=dev, dtype=torch.float32)
            dist.broadcast(bpd_tensor, src=0)
            avg_bpd = float(bpd_tensor.item())

        self.plateau.step(avg_bpd)
        if self.rank == 0:
            lr_now = self.opt.param_groups[0]["lr"]
            tqdm.write(f"[eval] iter={it} avg_bpd={avg_bpd:.4f} lr={lr_now:.2e}")

    def _run_sample_plots(self, it: int) -> None:
        args        = self.args
        eval_models = self.ema_models if self.ema_models else self.models
        tmpl        = getattr(self, "_tmpl_by_view", [None] * len(eval_models))

        if args.sample_mode == "off":
            tqdm.write("[samples] skipping previews (--sample-mode off)")
            return

        with torch.no_grad():
            if args.sample_mode == "model":
                n_samples = 100
                nrow      = 10
                shared_seed = int(getattr(args, "seed", 42)) + it
                any_ok = False
                for vi, m in enumerate(eval_models):
                    if tmpl[vi] is None:
                        continue
                    _prime_if_needed(m, tmpl[vi])
                    warmup_actnorm_with_real_batch(m, tmpl[vi])
                    cpu_state  = torch.random.get_rng_state()
                    cuda_states = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None
                    )
                    try:
                        torch.manual_seed(shared_seed)
                        ok, err = _save_samples_grid(
                            m, n_samples, args.sample_temp,
                            self.run_dir / f"samples_view{vi}_it{it:06d}",
                            nrow=nrow,
                            target_hw=(args.H, args.W),
                            warm_x=tmpl[vi],
                            which_type=getattr(args, "sample_grid_norm", "to01"),
                            chunk_size=int(getattr(args, "sample_chunk_size", 20)),
                        )
                    finally:
                        torch.random.set_rng_state(cpu_state)
                        if cuda_states is not None:
                            torch.cuda.set_rng_state_all(cuda_states)
                    if not ok:
                        tqdm.write(f"[warn] model sampling failed view {vi} @ {it}: {err}")
                    elif err:
                        tqdm.write(f"[info] model sample grid view {vi} @ {it}: {err}")
                    any_ok = any_ok or ok
                if any_ok:
                    tqdm.write(f"[samples] saved model sample grids @ iter {it}")

    def _save_input_grids(
        self, eval_models: List[nn.Module], it: int
    ) -> Tuple[bool, Optional[str]]:
        args = self.args
        dev = self.dev
        run_dir = self.run_dir
        n_views = len(eval_models)
        
        # 1. Collecte des batchs jusqu'à remplir 100 images
        all_batches = []
        total_count = 0
        try:
            for batch in self.train_loader:
                all_batches.append(batch)
                # Calcule la taille du batch (générique)
                b_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
                total_count += b_size
                if total_count >= 100:
                    break
        except StopIteration:
            return False, "loader exhausted"

        try:
            for vi in range(n_views):
                # 2. Concaténation des vues de tous les batchs récupérés
                list_of_views = []
                current_vi_count = 0
                
                for batch in all_batches:
                    xs = _extract_views_from_batch(batch, num_views=n_views)
                    x_tensor = xs[vi].to(device=dev, dtype=torch.float32)
                    
                    # Normalisation géométrique
                    if x_tensor.ndim == 3:   # 2D (B, H, W) -> (B, 1, H, W)
                        x_tensor = x_tensor.unsqueeze(1)
                    elif x_tensor.ndim == 4: # 3D (B, H, W, D) -> (B, 1, H, W) par coupe
                        x_tensor = x_tensor
                    
                    list_of_views.append(x_tensor)
                    current_vi_count += x_tensor.shape[0]
                    if current_vi_count >= 100:
                        break
                
                # 3. Assemblage final et découpage à 100
                x_all = torch.cat(list_of_views, dim=0)[:100]
                
                # 4. Normalisation et sauvegarde
                imgs = _coerce_nchw_4d(x_all, target_hw=(args.H, args.W), axis=-1)
                
                grid = tv.utils.make_grid(imgs, nrow=10, padding=2, normalize=False)
                tv.utils.save_image(grid, str(run_dir / f"input_data_view{vi}.png"))

                del x_all, list_of_views, imgs
            
            del all_batches
            gc.collect()
            return True, None
        except Exception as e:
            return False, str(e)