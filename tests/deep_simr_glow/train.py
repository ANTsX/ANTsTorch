#!/usr/bin/env python3
"""
Glow 2D trainer (ANTsTorch builder) with:
  • Multi-view (T2/T1/FA), exact log-likelihood
  • Warmup + ReduceLROnPlateau, grad clip, AMP (bf16/fp16), EMA
  • Resume, CSV logging, preview grids (model or val data)
  • Alignment across views: --align {none,infonce,barlow,vicreg,hsic,pearson}
  • Optional Kendall & Gal uncertainty weighting: --weighting {fixed,kendall}
  • Synthetic data fallback: --data synthetic
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision as tv


from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

import ants
import antstorch
import normflows as nf

from contextlib import nullcontext

def _no_autocast_for(device_type: str):
    # keep NF math in FP32 for stability under AMP
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:
        return nullcontext()

# ------------------------- small utils -------------------------

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _check_hw_divisible(H: int, W: int, L: int):
    r = 2 ** L
    if (H % r) or (W % r):
        raise ValueError(f"H and W must be divisible by 2**L={r}. Got H={H}, W={W}, L={L}")

def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def bits_per_dim(logp: torch.Tensor, num_dims: int) -> torch.Tensor:
    return -logp / (np.log(2.0) * float(num_dims))  # [B]

def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

# Ensure ActNorm init & inverse happen in FP32 (safe w/ AMP)
def flow_inverse_fp32(model, x, dev_type):
    with _no_autocast_for(dev_type):     # you already have this context helper
        z_list, ldj = model.inverse_and_log_det(x.float())
    return z_list, ldj

# Put this helper near your other utils (top-level)
def build_lat_flat_fp32(models, batch_views, dev_type, device):
    """
    batch_views: (B, V, H, W) tensor from your train loader (same one used for log_prob)
    Returns: list of flattened z per view in FP32
    """
    lat_flat = []
    for vi, m in enumerate(models):
        xv = to01(batch_views[:, vi:vi+1, :, :].to(device)).to(torch.float32)
        with _no_autocast_for(dev_type):                # <-- force FP32 for ActNorm/init
            z_list, _ = m.inverse_and_log_det(xv)
        z_last = torch.flatten(z_list[-1], 1)           # or however you pick/aggregate z
        lat_flat.append(z_last)
    return lat_flat


# --- helpers (put near imports) ---
@torch.no_grad()
def _copy_actnorm_state(src, dst):
    for ms, md in zip(src.modules(), dst.modules()):
        if "actnorm" in ms.__class__.__name__.lower():
            for fld in ("logs","log_scale","scale","weight"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("bias","b"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("initialized","is_initialized","inited"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    try: getattr(md, fld).data.copy_(getattr(ms, fld).data)
                    except Exception: setattr(md, fld, bool(getattr(ms, fld)))

# Robust, version-agnostic exact log p(x)
def log_prob_exact(model, x: torch.Tensor) -> torch.Tensor:
    """
    Exact log p(x) = Σ_i log p_i(z_i) + log|det J|.
    Works for nf.MultiscaleFlow (z is list) and single-scale (z is tensor).
    """
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

    return base_lp + logdet  # shape [B]

def make_warmup(optimizer, warmup_iters: int, decay_gamma: float, decay_steps: int):
    if warmup_iters <= 0 and (decay_gamma == 1.0 or decay_steps <= 0):
        return None
    def lr_lambda(step):
        s = max(1, step)
        scale = 1.0
        if warmup_iters > 0 and s < warmup_iters:
            scale *= s / float(warmup_iters)
        if decay_gamma != 1.0 and decay_steps > 0:
            scale *= (decay_gamma ** (s / float(decay_steps)))
        return scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ------------------------- alignment helpers -------------------------
class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):  # [B, D]
        return self.net(x)

def _flatten_latents(z):
    zs = z if isinstance(z, (list, tuple)) else [z]
    return torch.cat([zi.flatten(1) for zi in zs], dim=1)  # [B, sum_i CiHiWi]


# ------------------------- viz helpers -------------------------

def _extract_views_from_batch(batch, num_views: int | None = None):
    """
    Normalize a multi-view batch into a list [x_view0, x_view1, ...], each (B,C,H,W).

    Supported input forms:
      - dict with 'x' or 'views' (list/tuple of tensors)
      - list/tuple of per-view tensors
      - tuple like (x, y, ...) where x is a tensor or multi-view container
      - torch.Tensor of shape:
          (B, V, C, H, W)  -> unstack along V
          (B, C_total, H, W) with C_total % num_views == 0 -> split channels
    """
    import torch

    # If it's a (inputs, labels, ...) style tuple, peel off the first element.
    if isinstance(batch, tuple) and len(batch) > 0 and (
        torch.is_tensor(batch[0]) or isinstance(batch[0], (list, tuple, dict))
    ):
        return _extract_views_from_batch(batch[0], num_views=num_views)

    # Dict-based batches
    if isinstance(batch, dict):
        if 'x' in batch:
            return _extract_views_from_batch(batch['x'], num_views=num_views)
        if 'views' in batch:
            vs = batch['views']
            if isinstance(vs, (list, tuple)) and len(vs) > 0 and torch.is_tensor(vs[0]):
                return list(vs)
            raise ValueError("Batch['views'] not in expected list/tuple[tensor] format.")
        # Try any list/tuple-of-tensors value
        for v in batch.values():
            if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
                return list(v)
        raise ValueError("Batch dict format not recognized for multi-view data.")

    # Already a list/tuple of per-view tensors
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        return list(batch)

    # Tensor formats
    if torch.is_tensor(batch):
        if batch.ndim == 5:
            # (B, V, C, H, W)
            B, V, C, H, W = batch.shape
            return [batch[:, vi, :, :, :] for vi in range(V)]
        elif batch.ndim == 4:
            # (B, C_total, H, W) possibly packed views
            if num_views is None or num_views <= 1:
                return [batch]
            B, Ctot, H, W = batch.shape
            if Ctot % num_views != 0:
                raise ValueError(
                    f"Cannot split (B,C,H,W)=({B},{Ctot},{H},{W}) into {num_views} views: "
                    f"C ({Ctot}) not divisible by num_views."
                )
            Cpv = Ctot // num_views
            return [batch[:, vi*Cpv:(vi+1)*Cpv, :, :] for vi in range(num_views)]
        else:
            raise ValueError(f"Unsupported tensor ndim={batch.ndim}; expected 4 or 5.")

    raise ValueError(f"Unsupported batch type for multi-view extraction: {type(batch)}")

def _save_grid_from_tensor(x, out_path: Path, nrow: int, target_hw=None, value_range=None):
    """
    Save a grid image from a tensor x of shape (N,C,H,W).
    """
    x = x.detach().cpu()
    if target_hw is not None and (x.shape[-2] != target_hw[0] or x.shape[-1] != target_hw[1]):
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
    grid = tv.utils.make_grid(x, nrow=nrow, normalize=(value_range is None), value_range=value_range)
    tv.utils.save_image(grid, str(out_path))

def save_coordinated_input_grids(val_loader, num_views: int, out_dir: Path,
                                 n: int = 100, nrow: int = 10, target_hw=None, device="cpu"):
    """
    Collect exactly the SAME n samples (subjects) once and write one grid per view.
    Returns (ok: bool, err: Optional[str]).
    """
    samples_per_view = [ [] for _ in range(num_views) ]
    collected = 0

    # Single pass over the loader to lock subject alignment
    for batch in val_loader:
        xs = _extract_views_from_batch(batch, num_views=num_views)  # [x_v0, x_v1, ...]
        if len(xs) != num_views:
            return False, f"Expected {num_views} views, got {len(xs)}."
        B = xs[0].shape[0]
        take = min(n - collected, B)
        if take <= 0:
            break
        for vi in range(num_views):
            xvi = xs[vi][:take].to(device, non_blocking=True)
            samples_per_view[vi].append(xvi)
        collected += take
        if collected >= n:
            break

    if collected == 0:
        return False, "val_loader yielded no samples."

    for vi in range(num_views):
        x = torch.cat(samples_per_view[vi], dim=0)[:n]  # (n,C,H,W)
        out_path = out_dir / f"input_data_view{vi}.png"
        _save_grid_from_tensor(x, out_path, nrow=nrow, target_hw=target_hw)

    return True, None

def _make_grid_canvas(x, nrow=10):
    """
    x: Tensor (N,C,H,W). Returns a single (C, H_total, W_total) canvas.
    """
    assert torch.is_tensor(x) and x.dim() == 4, "x must be (N,C,H,W) tensor"
    N, C, H, W = x.shape
    cols = int(nrow)
    rows = (N + cols - 1) // cols
    canvas = x.new_zeros(C, rows * H, cols * W)
    for idx in range(N):
        r = idx // cols
        c = idx % cols
        canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = x[idx]
    return canvas

def _coerce_nchw_4d(x, target_hw=None):
    """
    Ensure x is a 4D float tensor (N,C,H,W), optionally resized to target_hw=(H,W).
    Handles: tensor, (tensor, logw), list/tuple of tensors (picks largest H*W).
    Converts channel-last to channel-first, clamps to [0,1], reduces odd C to 1.
    """
    # If list/tuple, pick largest candidate (by spatial area) after normalizing shapes
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3,4)]
        if not cands:
            raise ValueError("No tensor candidates in sample output.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 3:  # CHW or HWC
                if t.shape[-1] in (1,3) and (t.shape[0] not in (1,3)):
                    t = t.permute(2,0,1).contiguous()
                t = t.unsqueeze(0)  # NCHW
            elif t.dim() == 4:
                if t.shape[-1] in (1,3) and t.shape[1] not in (1,3):
                    t = t.permute(0,3,1,2).contiguous()
            fixed.append(t)
            areas.append(int(t.shape[-1]) * int(t.shape[-2]))
        x = fixed[int(torch.tensor(areas).argmax().item())]

    # Tensor path
    if not torch.is_tensor(x):
        raise ValueError(f"Sample output is not a tensor: {type(x)}")

    if x.dim() == 3:
        if x.shape[-1] in (1,3) and x.shape[0] not in (1,3):
            x = x.permute(2,0,1).contiguous()
        x = x.unsqueeze(0)

    if x.dim() == 4 and x.shape[-1] in (1,3) and x.shape[1] not in (1,3):
        x = x.permute(0,3,1,2).contiguous()

    if x.size(1) not in (1,3):
        x = x.mean(dim=1, keepdim=True)

    x = torch.clamp(x, 0, 1).float()

    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if (H, W) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)

    return x

@torch.no_grad()
def _save_samples_grid(model, n, temp, out_path, nrow=10, target_hw=None):
    """
    Try model.sample; coerce to (N,C,H,W); tile; save.
    """
    try:
        try:
            s = model.sample(n, temperature=temp)   
        except TypeError:
            s = model.sample(n)                     
        x = s[0] if isinstance(s, (list, tuple)) else s
        x = _coerce_nchw_4d(x, target_hw=target_hw)
        # If std is suspiciously tiny, retry with manual latent sampling
        try:
            _std = x.std().item()
        except Exception:
            _std = 0.0
        if _std < 1e-5:
            try:
                x = _manual_prior_sample(model, n, temp, x_template=None)
                x = _coerce_nchw_4d(x, target_hw=target_hw)
            except Exception:
                pass
        # If std is suspiciously tiny, retry with manual latent sampling
        if torch.isfinite(x).all():
            _std = x.std().item()
            if _std < 1e-5:
                try:
                    x = _manual_prior_sample(model, n, temp)
                    x = _coerce_nchw_4d(x, target_hw=target_hw)
                except Exception:
                    pass
        x = to01(x)
        assert torch.isfinite(x).all(), "non-finite in sample grid"
        if x.shape[0] < n:
            reps = (n + x.shape[0] - 1) // x.shape[0]
            x = x.repeat(reps, 1, 1, 1)
        x = x[:n]
        grid = _make_grid_canvas(x, nrow=nrow)
        tv.utils.save_image(grid, str(out_path))
        return True, None
    except Exception as e:
        return False, str(e)

@torch.no_grad()
def _save_batch_grid_from_loader(val_loader, view_idx, out_path, n=100, nrow=10, target_hw=None, device=None):
    """
    Draw images from val_loader for a given view and save a grid.
    """
    try:
        xs = []
        for batch in val_loader:
            x = batch[:, view_idx:view_idx+1, :, :]
            if device is not None:
                x = x.to(device)
            x = to01(x.float())
            if target_hw is not None:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            xs.append(x.cpu())
            if sum(b.size(0) for b in xs) >= n:
                break
        if not xs:
            return False, "val_loader empty"
        x = torch.cat(xs, dim=0)[:n]
        grid = _make_grid_canvas(x, nrow=nrow)
        tv.utils.save_image(grid, str(out_path))
        return True, None
    except Exception as e:
        return False, str(e)

def _save_metric_plots(csv_path: Path, out_dir: Path):
    """
    Reads metrics.csv and writes loss and bpd line plots as PNGs.
    """
    if not csv_path.exists():
        return
    iters, losses, bpds = [], [], []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                it, loss, bpd = int(float(row[0])), float(row[1]), float(row[2])
                iters.append(it); losses.append(loss); bpds.append(bpd)
        if len(iters) < 2:
            return
        # Loss
        plt.figure()
        plt.plot(iters, losses)
        plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()
        # BPD
        plt.figure()
        plt.plot(iters, bpds)
        plt.xlabel("iter"); plt.ylabel("sum_bpd"); plt.title("Sum BPD (training batches)")
        plt.tight_layout()
        plt.savefig(out_dir / "bpd_curve.png")
        plt.close()
    except Exception:
        pass

# ------------------------- data -------------------------

def load_hcpya_slices(mods: List[str], H: int, W: int, slice_idx=120):
    keys = dict(T2="hcpyaT2Template", T1="hcpyaT1Template", FA="hcpyaFATemplate")
    imgs = [ants.image_read(antstorch.get_antstorch_data(keys[m])) for m in mods]
    slcs = [ants.slice_image(im, axis=2, idx=slice_idx, collapse_strategy=1) for im in imgs]
    tmpl = ants.resample_image(slcs[0], (H, W), use_voxels=True)
    return slcs, tmpl

def build_loaders(mods, H, W, train_samples, val_samples, batch, num_workers, do_aug=True):
    slcs, tmpl = load_hcpya_slices(mods, H, W)
    train = antstorch.ImageDataset(
        images=[slcs],
        template=tmpl,
        do_data_augmentation=do_aug,
        data_augmentation_transform_type="affineAndDeformation",
        data_augmentation_sd_affine=0.02,
        data_augmentation_sd_deformation=10.0,
        number_of_samples=int(train_samples),
    )
    val = antstorch.ImageDataset(
        images=[slcs],
        template=tmpl,
        do_data_augmentation=True,
        data_augmentation_transform_type="affineAndDeformation",
        data_augmentation_sd_affine=0.05,
        data_augmentation_sd_deformation=0.2,
        data_augmentation_noise_model="additivegaussian",
        data_augmentation_sd_simulated_bias_field=1.0,
        data_augmentation_sd_histogram_warping=0.05,
        number_of_samples=int(val_samples),
    )
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val,   batch_size=min(16, batch), shuffle=False, num_workers=max(1, num_workers // 2))
    return train_loader, val_loader

class _SyntheticDataset(Dataset):
    def __init__(self, n_samples:int, views:int, H:int, W:int, seed:int=0):
        self.n = int(n_samples); self.v = int(views); self.H = int(H); self.W = int(W)
        g = torch.Generator().manual_seed(seed)
        self.data = torch.rand((self.n, self.v, self.H, self.W), generator=g)
    def __len__(self): return self.n
    def __getitem__(self, i): return self.data[i]

def build_loaders2(mods, H, W, train_samples, val_samples, batch, num_workers, data_source="hcpya", seed=0):
    if data_source == "synthetic":
        views = len(mods)
        train = _SyntheticDataset(n_samples=int(train_samples), views=views, H=int(H), W=int(W), seed=seed)
        val   = _SyntheticDataset(n_samples=int(val_samples),   views=views, H=int(H), W=int(W), seed=seed+1)
        train_loader = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val,   batch_size=min(16, batch), shuffle=False, num_workers=0)
        return train_loader, val_loader
    else:
        return build_loaders(mods, H, W, train_samples, val_samples, batch, num_workers, do_aug=True)

def ensure_shapes_cached(model, x_template: torch.Tensor):
    """
    Make sure MultiscaleFlow has cached latent shapes for sampling.
    If sample() complains, do a 1-sample forward to populate cache.
    """
    try:
        _ = model.sample(1)  # if cache exists this is a no-op
        return
    except Exception:
        pass
    with torch.no_grad():
        x1 = x_template[:1].to(next(model.parameters()).device).to(torch.float32)
        try:
            _ = model.log_prob(x1)
        except Exception:
            _ = model.inverse_and_log_det(x1)

@torch.no_grad()
def warmup_actnorm_with_real_batch(model, x_real: torch.Tensor):
    """
    Ensures ActNorm in `model` is data-initialized using REAL data.
    Does a 1-sample likelihood pass; safe to call once per model/view.
    """
    dev = next(model.parameters()).device
    x1 = x_real[:1].to(dev, torch.float32)
    # Prefer a path that triggers ActNorm data-init
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue

@torch.no_grad()
def _manual_prior_sample(model, n: int, temp: float = 1.0, x_template: torch.Tensor = None):
    """
    Fallback sampler that:
      1) infers latent shapes via inverse on a REAL template,
      2) draws Gaussian latents scaled by temp,
      3) maps z -> x via model's 'forward_from_latents' (or similar).
    Returns an (N,C,H,W) tensor.
    """
    # Device/dtype
    p = next(model.parameters())
    dev, dt = p.device, torch.float32

    # Build a template if none provided
    if x_template is None:
        H = W = 64
        if hasattr(model, "input_shape") and isinstance(model.input_shape, (tuple, list)) and len(model.input_shape) >= 3:
            H, W = int(model.input_shape[-2]), int(model.input_shape[-1])
        x_template = torch.randn(1, 1, H, W, device=dev, dtype=dt) * 0.1

    # 1) infer latent list via inverse pass
    z_tmpl, _ = model.inverse_and_log_det(x_template[:1].to(dev, dt))
    if isinstance(z_tmpl, torch.Tensor):
        z_tmpl = [z_tmpl]

    # 2) sample latents with temperature
    z_list = [torch.randn(n, *z.shape[1:], device=dev, dtype=z.dtype) * float(temp) for z in z_tmpl]

    # 3) map latents to x using best-available API
    for fn in ("forward_from_latents", "forward", "sample_from_latents", "_forward"):
        if hasattr(model, fn):
            out = getattr(model, fn)(z_list)
            return out[0] if isinstance(out, (list, tuple)) else out

    # Last resort: now that shapes exist, try model.sample
    s = model.sample(n, T=temp) if hasattr(model, "sample") and ("T" in model.sample.__code__.co_varnames) else model.sample(n)
    return s[0] if isinstance(s, (list, tuple)) else s

# ------------------------- main -------------------------


def main():
    ap = argparse.ArgumentParser("Glow 2D (builder) trainer")
    ap.add_argument("--modalities", nargs="+", default=["T2"], choices=["T2","T1","FA"])
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=96)

    ap.add_argument("--base", type=str, default="glow", choices=["glow","diag"])
    ap.add_argument("--glowbase-logscale-factor", type=float, default=3.0)
    ap.add_argument("--glowbase-min-log", type=float, default=-5.0)
    ap.add_argument("--glowbase-max-log", type=float, default=5.0)
    ap.add_argument("--scale-map", type=str, default="tanh", choices=["tanh","exp","sigmoid","sigmoid_inv"])
    ap.add_argument("--scale-cap", type=float, default=3.0)
    ap.add_argument("--net-actnorm", action="store_true", help="ActNorm in coupling subnets")

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-samples", type=int, default=6000)
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=30000, help="Target total iterations for this run")
    ap.add_argument("--extra-iters", type=int, default=0, help="If >0, ignore --max-iter and run this many more iterations from the resume point")
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--plot-interval", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--precision", type=str, default="mixed", choices=["double","float","mixed"])
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-iters", type=int, default=800)
    ap.add_argument("--lr-decay-gamma", type=float, default=1.0)
    ap.add_argument("--lr-decay-steps", type=int, default=0)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=4)
    ap.add_argument("--plateau-threshold", type=float, default=1e-4)
    ap.add_argument("--plateau-cooldown", type=int, default=0)
    ap.add_argument("--min-lr", type=float, default=1e-6)

    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9995)

    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    ap.add_argument("--auto-resume", action="store_true", help="If set, try <out-dir>/training_state.pt when --resume is not provided")
    ap.add_argument("--out-dir", type=str, default="runs_glow2d_builder")
    ap.add_argument("--data", type=str, choices=["hcpya","synthetic"], default="hcpya", help="Use HCP-YA slices or synthetic noise data")
    ap.add_argument("--synthetic-samples", type=int, default=8192, help="Dataset size per split when using synthetic data")
    ap.add_argument("--smooth-alpha", type=float, default=0.1, help="EMA smoothing factor in (0,1]; higher = faster")

    # Alignment & weighting
    ap.add_argument("--align", choices=["none","infonce","barlow","vicreg","hsic","pearson"], default="none", help="Latent alignment loss across views")
    ap.add_argument("--align-weight", type=float, default=0.05, help="Fixed weight for alignment loss (if --weighting=fixed)")
    ap.add_argument("--align-warmup", type=int, default=500, help="Number of warm-up alignment its")
    ap.add_argument("--proj-dim", type=int, default=256, help="Projection head output dim")
    ap.add_argument("--proj-hidden", type=int, default=512, help="Projection MLP hidden dim")
    ap.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")
    ap.add_argument("--barlow-lambda", type=float, default=5e-3, help="Off-diagonal weight (lambda) for Barlow Twins")
    ap.add_argument("--weighting", choices=["fixed","kendall"], default="fixed", help="Loss weighting strategy")
    ap.add_argument("--init-logvar-nll", type=float, default=0.0, help="Init log variance (s) for NLL in Kendall weighting")
    ap.add_argument("--init-logvar-align", type=float, default=0.0, help="Init log variance (s) for ALIGN in Kendall weighting")
    # VICReg hyperparameters
    ap.add_argument("--vicreg-inv", type=float, default=25.0, help="VICReg invariance weight (MSE between views)")
    ap.add_argument("--vicreg-var", type=float, default=25.0, help="VICReg variance weight (keep per-dim std above gamma)")
    ap.add_argument("--vicreg-cov", type=float, default=1.0,  help="VICReg covariance weight (penalize off-diagonals)")
    ap.add_argument("--vicreg-gamma", type=float, default=1.0, help="VICReg variance floor (target std per feature)")

    # HSIC hyperparameters (RBF kernel)
    ap.add_argument("--hsic-sigma", type=float, default=0.0, help="RBF bandwidth; 0 -> median heuristic per batch")

    # Preview grids
    ap.add_argument("--sample-mode", type=str, choices=["model","data","off"], default="model",
                    help="How to produce preview grids during eval: model sampling, random val batch, or skip")
    ap.add_argument("--sample-temp", type=float, default=1.0,
                help="Sampling temperature: scales prior noise (z = T·ε) when --sample-mode model")

    args = ap.parse_args()

    # Device + precision
    set_deterministic(args.seed)
    dev = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])

    if args.precision == "double":
        model_dtype = torch.float64
        amp_enabled = False
        amp_dtype = None
    elif args.precision == "float":
        model_dtype = torch.float32
        amp_enabled = False
        amp_dtype = None
    else:
        model_dtype = torch.float32  # keep params in FP32
        amp_enabled = True
        if args.amp_dtype == "bf16" and dev.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    # GradScaler: enable only for fp16 (bf16 doesn't need scaling)
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    # Sizes
    _check_hw_divisible(args.H, args.W, args.L)
    C = 1
    input_shape = (C, args.H, args.W)
    n_dims = int(np.prod(input_shape))

    # Data
    try:
        train_loader, val_loader = build_loaders2(
            args.modalities, args.H, args.W,
            args.train_samples if args.data!="synthetic" else args.synthetic_samples,
            args.val_samples if args.data!="synthetic" else max(256, args.batch*4),
            args.batch, args.num_workers,
            data_source=args.data, seed=args.seed
        )
    except Exception as e:
        import traceback
        print("[data] failed to build loaders:", repr(e))
        traceback.print_exc()
        print("Hint: try --data synthetic to verify training loop independent of ANTs/ANTsTorch data availability.")
        raise
    train_iter = iter(train_loader)
    input_data_sampled = False

    # Build models using the builder
    from antstorch import create_glow_normalizing_flow_model_2d

    models: List[nf.Flow] = []
    for _ in args.modalities:
        m = create_glow_normalizing_flow_model_2d(
            input_shape=input_shape,
            L=args.L, K=args.K, hidden_channels=args.hidden,
            base=args.base,
            glowbase_logscale_factor=args.glowbase_logscale_factor,
            glowbase_min_log=args.glowbase_min_log,
            glowbase_max_log=args.glowbase_max_log,
            split_mode="channel",
            scale=True,
            scale_map=args.scale_map,
            leaky=0.0,
            net_actnorm=bool(args.net_actnorm),
            scale_cap=args.scale_cap,
        ).to(dev).float().train()  # force FP32 params
        for name, p in m.named_parameters():
            if p.dtype != torch.float32:
                print(f"[warn] casting param {name} from {p.dtype} -> float32")
                p.data = p.data.float()
        if not hasattr(m, 'input_shape'):
            m.input_shape = input_shape
        models.append(m)

    # ---------------- EMA (lazy init) ----------------
    ema_models = None  # will be created after the first real optimizer step

    # ---- One-time ActNorm warmup on REAL data (before projectors) ----
    with torch.no_grad():
        warm_batch = next(iter(train_loader))               # (B, V, H, W)
        for vi, m in enumerate(models):
            xv0 = to01(warm_batch[:, vi:vi+1].to(dev)).to(torch.float32)
            warmup_actnorm_with_real_batch(m, xv0)          # uses log_prob/inverse internally

    # --- projection heads for alignment ---
    projectors = None
    if args.align != "none":
        with torch.no_grad():
            # use the same warmed real batch as template (any view is fine for dim)
            x_tmpl = to01(warm_batch[:, 0:1].to(dev)).to(torch.float32)
            with _no_autocast_for(dev.type):                # force FP32
                z_probe, _ = models[0].inverse_and_log_det(x_tmpl[:1])
            flat_dim = _flatten_latents(z_probe).size(1)
        projectors = nn.ModuleList([
            Projector(flat_dim, args.proj_hidden, args.proj_dim).to(dev).train()
            for _ in range(len(models))
        ])

    # --- Kendall & Gal weighting scalars ---
    s_nll = s_align = None
    if args.weighting == "kendall" and args.align != "none":
        s_nll   = nn.Parameter(torch.tensor([args.init_logvar_nll], device=dev))
        s_align = nn.Parameter(torch.tensor([args.init_logvar_align], device=dev))

    # Optimizer + schedulers
    param_groups = [{"params": [p for m in models for p in m.parameters()]}]
    if projectors is not None:
        param_groups.append({"params": [p for p in projectors.parameters()]})
    if s_nll is not None:
        param_groups.append({"params": [s_nll, s_align], "weight_decay": 0.0})
    opt = torch.optim.Adamax(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    warm = make_warmup(opt, args.warmup_iters, args.lr_decay_gamma, args.lr_decay_steps)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.plateau_factor,
        patience=args.plateau_patience, threshold=args.plateau_threshold,
        cooldown=args.plateau_cooldown, min_lr=args.min_lr
    )

    # Out dir, resume
    run_dir = Path(args.out_dir); run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "training_state.pt"
    csv_path   = run_dir / "metrics.csv"

    start_iter = 1
    resume_path = None
    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            resume_path = rp
        else:
            raise FileNotFoundError(f"--resume specified but file not found: {rp}")
    elif args.auto_resume and state_path.exists():
        resume_path = state_path

    if resume_path is not None:
        blob = torch.load(resume_path, map_location=dev, weights_only=False)
        start_iter = int(blob.get("iter", 1))
        opt.load_state_dict(blob.get("opt", opt.state_dict()))
        if warm and blob.get("warm") is not None:
            warm.load_state_dict(blob["warm"])
        if blob.get("models") is not None:
            for m, sd in zip(models, blob["models"]):
                m.load_state_dict(sd)
        # EMA: create lazily and then load if present
        if args.ema and blob.get("ema") is not None:
            import copy
            ema_models = [copy.deepcopy(m).eval().to(dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            for em, sd in zip(ema_models, blob["ema"]):
                em.load_state_dict(sd)
        # restore projectors if present
        if blob.get("proj") is not None and projectors is not None:
            try:
                projectors.load_state_dict(blob["proj"])
                tqdm.write("[resume] restored projectors")
            except Exception as e:
                tqdm.write(f"[resume] warning: could not load projectors: {e}")
        # restore Kendall scalars
        if blob.get("kendall") is not None and s_nll is not None:
            try:
                kd = blob["kendall"]
                if kd.get("s_nll") is not None:  s_nll.data.fill_(float(kd["s_nll"]))
                if kd.get("s_align") is not None: s_align.data.fill_(float(kd["s_align"]))
                tqdm.write(f"[resume] restored Kendall s_nll={float(s_nll.item()):.3f}, s_align={float(s_align.item()):.3f}")
            except Exception as e:
                tqdm.write(f"[resume] warning: could not load Kendall scalars: {e}")
        tqdm.write(f"[resume] from {str(resume_path)} @ iter {start_iter}")

    # If user asked for extra iters, override max-iter to be (already_done + extra)
    if args.extra_iters > 0:
        args.max_iter = (start_iter - 1) + args.extra_iters

    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("iter,loss,sum_bpd,lr\n")

    # ------------------------- train loop -------------------------
    n_views = len(models)
    tqdm.write(f"[info] training {n_views} view(s); params per view: {[n_params(m) for m in models]}")

    # Running averages (EMA) display (simple, not saved)
    alpha = float(args.smooth_alpha)
    ema_loss_disp = None
    ema_sum_bpd_disp = None
    ema_bpd_views_disp = [None] * n_views

    pbar = tqdm(total=args.max_iter, initial=start_iter - 1, dynamic_ncols=True, desc="train")

    for it in range(start_iter, args.max_iter + 1):
        opt.zero_grad(set_to_none=True)
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        # Forward
        L_nll = torch.tensor(0.0, device=dev, dtype=torch.float32)
        curr_bpd_views = []
        sum_bpd = 0.0

        lat_flat = []

        if amp_enabled:
            from contextlib import nullcontext
            ctx = torch.amp.autocast(dev.type, dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            bad_batch = False
            for vi, m in enumerate(models):
                x_v = to01(x[:, vi:vi+1, :, :].to(dev)).float()

                # Keep NF math in FP32 for stability, and let the model do log_prob
                with _no_autocast_for(dev.type):
                    logp_v = m.log_prob(x_v.float())            # <-- model computes base+logdet
                    z_v, _ = m.inverse_and_log_det(x_v.float()) # <-- still get latents for alignment

                if not torch.isfinite(logp_v).all():
                    tqdm.write(f"[nan] non-finite logp in view {vi} at iter {it}; skipping step")
                    bad_batch = True
                    break

                bpd_v   = -logp_v / (np.log(2.0) * float(n_dims))
                bpd_mean = float(bpd_v.mean().detach().cpu().item())
                curr_bpd_views.append(bpd_mean)
                sum_bpd += bpd_mean

                L_nll = L_nll - logp_v.mean()
                if isinstance(z_v, (list, tuple)):
                    zflat = torch.cat([zi.flatten(1) for zi in z_v], dim=1)
                else:
                    zflat = z_v.flatten(1)
                lat_flat.append(torch.nan_to_num(zflat))

        if bad_batch or (not torch.isfinite(L_nll)):
            tqdm.write(f"[nan] skipping iter {it} (bad_batch={bad_batch}, L_nll finite={torch.isfinite(L_nll).item()})")
            continue

        # Alignment loss (if requested)
        L_align = torch.tensor(0.0, device=dev)
        if args.align != "none" and it >= args.align_warmup:
            feats = [projectors[i](lat_flat[i]) for i in range(len(lat_flat))]
            feats = [f.float() for f in feats]  # keep loss math in fp32

            if args.align == "barlow":
                L_align = antstorch.barlow_twins_multi(feats, lam=float(args.barlow_lambda))
            elif args.align == "vicreg":
                L_align = antstorch.vicreg_multi(
                    feats,
                    w_inv=float(args.vicreg_inv),
                    w_var=float(args.vicreg_var),
                    w_cov=float(args.vicreg_cov),
                    gamma=float(args.vicreg_gamma),
                )
            elif args.align == "infonce":
                L_align = antstorch.info_nce_multi(feats, T=float(args.temperature))
            elif args.align == "hsic":
                L_align = antstorch.hsic_multi(feats, sigma=float(args.hsic_sigma))
            elif args.align == "pearson":
                L_align = antstorch.pearson_multi(feats)

        # Combine with fixed or Kendall weighting
        if args.weighting == "fixed" or args.align == "none":
            loss_total = L_nll + (args.align_weight * L_align if args.align != "none" else 0.0)
            w_nll = 1.0
            w_align = float(args.align_weight if args.align != "none" else 0.0)
        else:
            s_nll_eff   = torch.clamp(torch.nan_to_num(s_nll,   nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            s_align_eff = torch.clamp(torch.nan_to_num(s_align, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            L_align = torch.nan_to_num(L_align, nan=0.0, posinf=0.0, neginf=0.0)
            L_nll   = torch.nan_to_num(L_nll,   nan=0.0, posinf=0.0, neginf=0.0)
            loss_total = torch.exp(-s_nll_eff) * L_nll + s_nll_eff
            loss_total = loss_total + torch.exp(-s_align_eff) * L_align + s_align_eff
            if not torch.isfinite(loss_total):
                tqdm.write(f"[nan] loss_total non-finite at iter {it}; skipping step")
                continue
            w_nll   = float(torch.exp(-s_nll_eff).detach().cpu().item())
            w_align = float(torch.exp(-s_align_eff).detach().cpu().item())

        # Backprop + clip + step
        all_params = []
        for g in param_groups:
            for p in g["params"]:
                if isinstance(p, torch.Tensor) and p.grad is not None:
                    all_params.append(p)
        if scaler.is_enabled():
            scaler.scale(loss_total).backward()
            scaler.unscale_(opt)  # grads must be FP32
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
            opt.step()

        # -------- Lazy EMA init right after the first real update --------
        if args.ema and ema_models is None:
            import copy
            ema_models = [copy.deepcopy(m).eval().to(dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            # copy ActNorm state from base → EMA, then warm on REAL batch
            with torch.no_grad():
                for vi, (m, em) in enumerate(zip(models, ema_models)):
                    _copy_actnorm_state(m, em)  # <-- helper you added
                    xv_real = to01(x[:, vi:vi+1, :, :].to(dev)).float()
                    warmup_actnorm_with_real_batch(em, xv_real)  # <-- helper you added
            tqdm.write("[ema] initialized from base after first update")

        # EMA update each step
        if ema_models is not None:
            with torch.no_grad():
                for em, m in zip(ema_models, models):
                    for p_em, p in zip(em.parameters(), m.parameters()):
                        p_em.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        # Warmup/exp decay
        if warm is not None:
            warm.step()

        # Update tqdm with live metrics
        lr_now = opt.param_groups[0]["lr"]

        # EMA smoothing for display
        curr_loss = float(loss_total.detach().cpu().item())
        if ema_loss_disp is None:
            ema_loss_disp = curr_loss
            ema_sum_bpd_disp = sum_bpd
            for i in range(n_views):
                ema_bpd_views_disp[i] = curr_bpd_views[i]
        else:
            a = alpha
            ema_loss_disp = (1.0 - a) * ema_loss_disp + a * curr_loss
            ema_sum_bpd_disp = (1.0 - a) * ema_sum_bpd_disp + a * sum_bpd
            for i in range(n_views):
                ema_bpd_views_disp[i] = (1.0 - a) * ema_bpd_views_disp[i] + a * curr_bpd_views[i]

        postfix = {
            "iter": it,
            "loss": f"{curr_loss:.4f}",
            "loss~": f"{ema_loss_disp:.4f}",
            "bpd": f"{sum_bpd:.3f}",
            "bpd~": f"{ema_sum_bpd_disp:.3f}",
            "lr": f"{lr_now:.2e}",
            "align": f"{float(L_align.detach().cpu().item()):.4f}",
            "mode": args.align,
            "w_nll": f"{w_nll:.2f}",
            "w_aln": f"{w_align:.2f}",
        }
        for i in range(n_views):
            postfix[f"v{i}"] = f"{curr_bpd_views[i]:.3f}/{ema_bpd_views_disp[i]:.3f}"

        pbar.set_postfix(postfix)
        pbar.update(1)


        # Sample the input data for reference
        if not input_data_sampled:
            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                num_views = len(eval_models)

                ok, err = save_coordinated_input_grids(
                    val_loader,
                    num_views=num_views,
                    out_dir=run_dir,
                    n=100,
                    nrow=10,
                    target_hw=(args.H, args.W),
                    device=dev,
                )

                if ok:
                    tqdm.write(f"[samples] saved coordinated input data grids @ iter {it}")
                    input_data_sampled = True
                else:
                    tqdm.write(f"[warn] input data grid failed @ iter {it}: {err}")

        # Eval + plateau + sampling
        if it % args.eval_interval == 0:
            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                bpd_acc = []
                tmpl_by_view = [None] * len(eval_models)
                vbar = tqdm(total=10, leave=False, dynamic_ncols=True, desc=f"val@{it}")
                for j, batch_val in enumerate(val_loader):
                    bpd_views = []
                    for vi, m in enumerate(eval_models):
                        xv = to01(batch_val[:, vi:vi+1, :, :].to(dev)).to(torch.float32)
                        tmpl_by_view[vi] = xv
                        with _no_autocast_for(dev.type):
                            lp = m.log_prob(xv.float())
                        lp = torch.nan_to_num(lp, nan=-1e9, posinf=-1e9, neginf=-1e9)
                        bpd_views.append(bits_per_dim(lp, n_dims).mean().item())
                    bpd_acc.append(np.mean(bpd_views))
                    vbar.update(1)
                    if len(bpd_acc) >= 10:
                        break
                vbar.close()
                avg_bpd = float(np.mean(bpd_acc)) if bpd_acc else float("nan")
            plateau.step(avg_bpd)
            tqdm.write(f"[eval] iter={it} avg_bpd={avg_bpd:.4f} lr={lr_now:.2e}")

            # preview grids + metric plots
            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models

                if args.sample_mode == "model":
                    any_ok = False
                    n_samples, nrow = 100, 10
                    shared_seed = int(getattr(args, "seed", 12345)) + int(it)  # vary across evals but match across views

                    for vi, m in enumerate(eval_models):
                        if tmpl_by_view[vi] is None:
                            tqdm.write(f"[warn] no real template available for view {vi}; skipping model samples this eval")
                            continue

                        # Warm ActNorm on REAL data for the exact model we will sample (EMA or base)
                        warmup_actnorm_with_real_batch(m, tmpl_by_view[vi])

                        # Snapshot RNG state (CPU + CUDA) so we can reseed for coordination and then restore.
                        cpu_state = torch.random.get_rng_state()
                        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                        try:
                            # Reseed so every view draws the *same* z for sample index i.
                            torch.manual_seed(shared_seed)

                            ok, err = _save_samples_grid(
                                m,
                                n_samples,
                                args.sample_temp,
                                run_dir / f"samples_view{vi}_it{it:06d}.png",
                                nrow=nrow,
                                target_hw=(args.H, args.W),
                            )
                        finally:
                            # Restore RNG states to avoid side effects on the rest of training/eval.
                            torch.random.set_rng_state(cpu_state)
                            if cuda_states is not None:
                                torch.cuda.set_rng_state_all(cuda_states)

                        if not ok:
                            tqdm.write(f"[warn] model sampling failed for view {vi} at iter {it}: {err}")
                        any_ok = any_ok or ok

                    if any_ok:
                        tqdm.write(f"[samples] saved *coordinated* model sample grids @ iter {it}")

                elif args.sample_mode == "data":
                    # Coordinate the subjects across views by collecting once and writing per-view grids
                    ok, err = save_coordinated_input_grids(
                        val_loader,
                        num_views=len(eval_models),
                        out_dir=run_dir,
                        n=100,
                        nrow=10,
                        target_hw=(args.H, args.W),
                        device=dev,
                    )
                    if ok:
                        tqdm.write(f"[samples] saved *coordinated* validation-batch grids @ iter {it}")
                    else:
                        tqdm.write(f"[warn] val-batch grid failed at iter {it}: {err}")

                else:
                    tqdm.write("[samples] skipping previews (--sample-mode off)")
            _save_metric_plots(csv_path, run_dir)

        # CSV log
        with open(csv_path, "a") as f:
            f.write(f"{it},{curr_loss:.6f},{sum_bpd:.6f},{lr_now:.6g}\n")

        # Lightweight checkpoint
        if it % args.eval_interval == 0:
            blob = {
                "iter": it + 1,
                "opt": opt.state_dict(),
                "warm": (warm.state_dict() if warm else None),
                "models": [m.state_dict() for m in models],
                "ema": ([em.state_dict() for em in ema_models] if ema_models is not None else None),
                "proj": (projectors.state_dict() if projectors is not None else None),
                "kendall": ({
                    "s_nll": float(s_nll.detach().cpu()) if s_nll is not None else None,
                    "s_align": float(s_align.detach().cpu()) if s_align is not None else None,
                }),
                "config": vars(args),
            }
            torch.save(blob, state_path)
            tqdm.write(f"[ckpt] saved: {str(state_path)}")

    pbar.close()
    print("Done. Run dir:", str(run_dir))



if __name__ == "__main__":
    main()
