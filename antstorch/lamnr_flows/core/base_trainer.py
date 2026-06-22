"""
base_trainer.py
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
import torchvision as tv
from tqdm.auto import tqdm

import ants
import antstorch

from antstorch.lamnr_flows.latent_alignment import (
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
    """nn.DataParallel with explicit redirections for Glow-specific methods."""

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
    if x1.ndim == 3:
        x1 = x1.unsqueeze(1)
    p = next(model.parameters(), None)
    dev = p.device if p is not None else x1.device
    x1 = x1.to(dev, dtype=torch.float32)
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
    x, target_hw: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
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
        x = x.mean(dim=1, keepdim=True)
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
def _save_samples_grid(
    model: nn.Module,
    n: int,
    temp: float,
    out_prefix,
    nrow: int = 10,
    target_hw=None,
    warm_x=None,
    which_type: str = "to01",
) -> Tuple[bool, Optional[str]]:
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
                return False, str(e2)
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
                return False, f"Primary failed: {e}. CPU fallback failed: {e_cpu}"

    try:
        x = s[0] if isinstance(s, (list, tuple)) else s
        x = _coerce_nchw_4d(x, target_hw=target_hw)
        _std = x.std().item() if torch.isfinite(x).all() else 0.0
        assert torch.isfinite(x).all(), "non-finite in sample grid"

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
        return True, None
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
    cfg["grad_accum"]    = int(cfg.get("grad_accum", 1))
    cfg["effective_batch"] = int(cfg.get("batch", 0)) * cfg["grad_accum"]

    env = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python":            platform.python_version(),
        "torch":             torch.__version__,
        "cuda_available":    torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if dataset_info:
        cfg["dataset_info"] = dataset_info

    with open(out_dir / "run_config.json", "w") as f:
        json.dump({"env": env, "config": cfg, "note": note}, f, indent=2)

    rows = [
        f"[run] {env['timestamp']} | Py {env['python']} | torch {env['torch']} "
        f"| cuda={_fmt_bool(env['cuda_available'])} (n={env['cuda_device_count']})"
    ]
    if note:
        rows.append(f"[note] {note}")

    def add(k, v):
        rows.append(f"{k:>24}: {'None' if v is None else v}")

    add("out_dir", cfg.get("out_dir"))
    add("views",   getattr(args, "num_views", None))
    add("L / K / hidden", f"{cfg.get('L')} / {cfg.get('K')} / {cfg.get('hidden')}")
    add("precision / amp_dtype", f"{cfg.get('precision')} / {cfg.get('amp_dtype')}")
    add("batch / grad_accum / eff_batch",
        f"{cfg.get('batch')} / {cfg.get('grad_accum')} / {cfg.get('effective_batch')}")
    add("max_iter / extra",  f"{cfg.get('max_iter')} / {cfg.get('extra_iters')}")
    add("lr / warmup",       f"{cfg.get('lr')} / {cfg.get('warmup_iters')}")
    add("grad_clip",         cfg.get("grad_clip"))
    add("ema / decay",       f"{_fmt_bool(cfg.get('ema'))} / {cfg.get('ema_decay')}")
    add("align",             cfg.get("align"))
    add("align_weight/warmup",
        f"{cfg.get('align_weight')} / {cfg.get('align_warmup')}")
    add("vicreg (i/v/c/g)",
        f"{cfg.get('vicreg_inv')}/{cfg.get('vicreg_var')}/{cfg.get('vicreg_cov')}/{cfg.get('vicreg_gamma')}")
    add("screen",  cfg.get("screen"))
    add("devices", cfg.get("devices"))
    add("seed",    cfg.get("seed"))

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

        # Device
        if args.devices.lower() == "cpu":
            dev = torch.device("cpu")
        elif args.devices == "mps" and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device(args.devices.split(",")[0])
        self.dev = dev

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
        self.opt = torch.optim.Adamax(
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

        # CSV header
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

        # Run-config dump
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

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_dp_prefix(state_dict: dict) -> dict:
        """
        Remove DataParallel / GlowStepWrapper prefixes ('module.' and 'model.')
        from all keys **before writing to disk**.

        Guarantees that saved checkpoints are always compatible with inference
        scripts that load a bare Glow model — even if training used DataParallel.
        """
        return {
            k.replace("module.", "").replace("model.", ""): v
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
            "config":  vars(self.args),
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

    def cleanup_checkpoints(self, keep_every: int = 10_000) -> None:
        """Keep only milestone checkpoints that are multiples of keep_every."""
        for f in self.run_dir.glob("training_state_it*.pt"):
            try:
                it_num = int(f.stem.split("it")[-1])
                if it_num % keep_every != 0:
                    f.unlink()
            except (ValueError, IndexError):
                continue

    def _load_model_state(self, m: nn.Module, sd: dict) -> None:
        """Load state dict into m, stripping DataParallel prefixes from sd."""
        clean_sd = self._strip_dp_prefix(sd)
        if isinstance(m, GlowDataParallel):
            m.module.model.load_state_dict(clean_sd)
        else:
            m.load_state_dict(clean_sd)

    def _maybe_resume(self, args) -> int:
        """Return start_iter (1 if fresh, resumed iter otherwise)."""
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
        try:
            self.opt.load_state_dict(blob["opt"])
        except Exception as e:
            print(f"[resume] optimizer not loaded ({e}); using fresh.")
            try:
                g0 = blob["opt"]["param_groups"][0]
                for k in ("lr", "betas", "eps", "weight_decay"):
                    if k in g0:
                        for g in self.opt.param_groups:
                            g[k] = g0[k]
            except Exception:
                pass

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
                copy.deepcopy(m).eval().to(dtype=torch.float32, device=self.dev)
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
    # Training loop
    # ------------------------------------------------------------------

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

        train_iter      = iter(self.train_loader)
        input_data_saved = False

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

                        # Forward pass
                        if isinstance(m, GlowDataParallel):
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

                if bad_batch or not torch.isfinite(L_nll) or abs(L_nll.item()) > 1e7:
                    tqdm.write(
                        f"[anomaly] skipping iter {it} "
                        f"(bad_batch={bad_batch}, L_nll={L_nll.item():.2f})"
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

                if not torch.isfinite(loss_total):
                    tqdm.write(f"[nan] loss_total non-finite at iter {it}; skipping")
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
                self.opt.zero_grad(set_to_none=True)
                continue

            # Gradient clip + optimizer step
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.opt)
            all_params = [p for g in self.opt.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=float(getattr(args, "grad_clip", 2.0))
            )
            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            # Use last micro-batch for EMA ActNorm warmup
            x = x_last

            # Averaged metrics
            curr_loss      = float(loss_acc.item())  / float(grad_accum)
            L_align_log    = float(align_acc.item()) / float(grad_accum)
            sum_bpd        = bpd_acc / float(grad_accum)
            curr_bpd_views = [v / float(grad_accum) for v in (bpd_views_acc or [])]

            # Lazy EMA init (after first successful update)
            if args.ema and self.ema_models is None:
                self.ema_models = [
                    copy.deepcopy(m).eval().to(dtype=torch.float32, device=dev)
                    for m in models
                ]
                for em in self.ema_models:
                    for p in em.parameters():
                        p.requires_grad_(False)
                with torch.no_grad():
                    for vi, (m, em) in enumerate(zip(models, self.ema_models)):
                        _copy_actnorm_state(m, em)
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

            # Eval + checkpoint
            if it % args.eval_interval == 0:
                self._run_eval(it, n_dims)
                self._run_sample_plots(it)
                _save_metric_plots(self.csv_path, self.run_dir, remove_spikes=True)
                self.save_checkpoint(it)

            # ── End-of-iteration cleanup ──────────────────────────────
            del x, x_last
            gc.collect()
            # ─────────────────────────────────────────────────────────

        pbar.close()
        print("Done. Run dir:", str(self.run_dir))

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def _run_eval(self, it: int, n_dims: int) -> None:
        args        = self.args
        dev         = self.dev
        eval_models = self.ema_models if self.ema_models else self.models

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
        self.plateau.step(avg_bpd)
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
                        )
                    finally:
                        torch.random.set_rng_state(cpu_state)
                        if cuda_states is not None:
                            torch.cuda.set_rng_state_all(cuda_states)
                    if not ok:
                        tqdm.write(f"[warn] model sampling failed view {vi} @ {it}: {err}")
                    any_ok = any_ok or ok
                if any_ok:
                    tqdm.write(f"[samples] saved model sample grids @ iter {it}")

    def _save_input_grids(
        self, eval_models: List[nn.Module], it: int
    ) -> Tuple[bool, Optional[str]]:
        """Save one input-data grid per view (first call only)."""
        args    = self.args
        dev     = self.dev
        run_dir = self.run_dir
        n_views = len(eval_models)

        try:
            batch = next(iter(self.train_loader))
        except StopIteration:
            return False, "empty train loader"

        try:
            xs = _extract_views_from_batch(batch, num_views=n_views)
            for vi in range(n_views):
                x_v = to01(xs[vi].to(dtype=torch.float32, device=dev))
                x_v = x_v[:100]
                imgs = _coerce_nchw_4d(x_v, target_hw=(args.H, args.W))
                grid = tv.utils.make_grid(imgs, nrow=10, padding=2, normalize=False)
                tv.utils.save_image(
                    grid, str(run_dir / f"input_data_view{vi}.png")
                )
                del x_v, imgs
            del xs, batch
            gc.collect()
            return True, None
        except Exception as e:
            return False, str(e)
