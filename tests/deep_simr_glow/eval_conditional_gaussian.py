#!/usr/bin/env python3
"""
eval_conditional_gaussian.py

Evaluate conditional-Gaussian cross-view imputation for multiview Glow models
trained with train.py. Loads a run's checkpoint, rebuilds the same model(s),
collects latents on an augmented dataset, fits a joint Gaussian over the
concatenated per-view latents, and evaluates imputations for all observed→missing
permutations (skipping 1-view runs).

Outputs:
  - <out-dir>/gaussian_stats.pt          (mu, Sigma, slices, shapes)
  - <out-dir>/impute_metrics.csv         (permutation metrics)
  - <out-dir>/impute_examples_*.png      (grids of GT vs. imputations per case)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import math
import csv
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision as tv

import antstorch
from antstorch import create_glow_normalizing_flow_model_2d

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")




# ------------------------- helpers (copied/adapted) -------------------------

def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _no_autocast_for(device_type: str):
    from contextlib import nullcontext
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:
        return nullcontext()

def build_loaders_for_eval(mods, H, W, gauss_samples, eval_samples, batch, num_workers, data_source="hcpya", seed=0):
    """
    Build two datasets:
      - 'gauss': used to fit the Gaussian (with augmentation)
      - 'eval' : used to compute imputation metrics (with augmentation for diversity)
    """
    # Reuse the same ImageDataset interface as train.py
    # We'll configure two datasets with their own number_of_samples.
    slcs, tmpl = _load_hcpya_slices(mods, H, W) if data_source=="hcpya" else _load_synthetic_slices(mods, H, W, seed)
    gauss_ds = antstorch.ImageDataset(
        images=[slcs], template=tmpl,
        do_data_augmentation=True,
        data_augmentation_transform_type="affineAndDeformation",
        data_augmentation_sd_affine=0.02,
        data_augmentation_sd_deformation=10.0,
        number_of_samples=int(gauss_samples),
    )
    eval_ds = antstorch.ImageDataset(
        images=[slcs], template=tmpl,
        do_data_augmentation=True,
        data_augmentation_transform_type="affineAndDeformation",
        data_augmentation_sd_affine=0.05,
        data_augmentation_sd_deformation=0.2,
        data_augmentation_noise_model="additivegaussian",
        data_augmentation_sd_simulated_bias_field=1.0,
        data_augmentation_sd_histogram_warping=0.05,
        number_of_samples=int(eval_samples),
    )
    gauss_loader = torch.utils.data.DataLoader(gauss_ds, batch_size=batch, shuffle=True, num_workers=num_workers)
    eval_loader  = torch.utils.data.DataLoader(eval_ds,  batch_size=min(16,batch), shuffle=False, num_workers=max(1, num_workers//2))
    return gauss_loader, eval_loader

def _load_hcpya_slices(mods: List[str], H:int, W:int, slice_idx:int=120):
    keys = dict(T2="hcpyaT2Template", T1="hcpyaT1Template", FA="hcpyaFATemplate")
    import ants
    imgs = [ants.image_read(antstorch.get_antstorch_data(keys[m])) for m in mods]
    slcs = [ants.slice_image(im, axis=2, idx=slice_idx, collapse_strategy=1) for im in imgs]
    tmpl = ants.resample_image(slcs[0], (H, W), use_voxels=True)
    return slcs, tmpl

class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n:int, v:int, H:int, W:int, seed:int=0):
        self.n=n; self.v=v; self.H=H; self.W=W
        g = torch.Generator().manual_seed(seed)
        self.data = torch.rand((n, v, H, W), generator=g)
    def __len__(self): return self.n
    def __getitem__(self, i): return self.data[i]

def _load_synthetic_slices(mods, H, W, seed):
    # For synthetic, mimic antstorch API: return list of 2D images and a "template" image
    v = len(mods)
    ds = _SyntheticDataset(1, v, H, W, seed=seed)
    x = ds[0]  # (v, H, W)
    # Convert to ANTs images if available; otherwise return tensors and let ImageDataset handle
    import ants
    ims = []
    for vi in range(v):
        arr = x[vi].numpy().astype(np.float32)
        im = ants.from_numpy(arr)
        ims.append(im)
    tmpl = ants.from_numpy(np.zeros((H, W), dtype=np.float32))
    return ims, tmpl

def _extract_views_from_batch(batch, num_views: int | None = None):
    # Minimal extractor for (B,V,H,W) batches produced by antstorch ImageDataset
    if torch.is_tensor(batch):
        if batch.dim() == 4:  # (B,V,H,W)
            B, V, H, W = batch.shape
            return [batch[:, vi:vi+1, :, :].contiguous() for vi in range(V)]
        elif batch.dim() == 5:  # (B,V,C,H,W)
            B, V, C, H, W = batch.shape
            return [batch[:, vi, :, :, :].contiguous() for vi in range(V)]
    if isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
        return list(batch)
    if isinstance(batch, dict) and "x" in batch:
        return _extract_views_from_batch(batch["x"], num_views=num_views)
    raise ValueError(f"Unrecognized batch format: {type(batch)}")

@torch.no_grad()
def warmup_actnorm_with_real_batch(model, x_real: torch.Tensor):
    dev = next(model.parameters()).device
    x1 = x_real[:1].to(dev, torch.float32)
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue

def flatten_latents_list(z_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([zi.flatten(1) for zi in z_list], dim=1)

def get_latent_shapes(model, x_template: torch.Tensor) -> List[Tuple[int,...]]:
    """Run a single inverse pass to record per-level shapes (B,C,H,W)."""
    with torch.no_grad():
        z_list, _ = model.inverse_and_log_det(x_template[:1].to(next(model.parameters()).device).float())
        if isinstance(z_list, torch.Tensor):
            z_list = [z_list]
        shapes = [tuple(z.shape) for z in z_list]  # each starts with B
    return shapes

def flatten_from_models(models: List[nn.Module], batch_views: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    latents_flat = []
    for vi, m in enumerate(models):
        xv = to01(batch_views[vi].to(device)).to(torch.float32)
        with _no_autocast_for(device.type):
            z_list, _ = m.inverse_and_log_det(xv)
        if isinstance(z_list, torch.Tensor):
            z_list = [z_list]
        zflat = flatten_latents_list(z_list)  # (B, Dvi)
        latents_flat.append(zflat)
    return latents_flat

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    # clamp to [0,1] for stability if your pipeline is in [0,1]
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)

    # batch-safe MSE (averaged over all dims and batch)
    mse_t = torch.mean((x - y) ** 2)

    # guard: replace non-finite with +inf so PSNR becomes 0 dB, not an exception
    if not torch.isfinite(mse_t):
        return float('nan')  # we'll nan-mean later

    mse = float(mse_t.item())
    if mse <= 0.0:
        # perfect match or numeric underflow -> treat as very high PSNR
        return 99.0
    return 10.0 * math.log10((data_range * data_range) / max(mse, eps))

def save_grid(x: torch.Tensor, path: Path, nrow: int = 8):
    x = torch.clamp(x.detach().cpu().float(), 0, 1)
    grid = tv.utils.make_grid(x, nrow=nrow, normalize=False)
    tv.utils.save_image(grid, str(path))


# ------------------------- quality metrics -------------------------
def _gaussian_window_2d(win_size: int, sigma: float, device, dtype):
    ax = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = torch.outer(g, g)
    w2d = w2d / w2d.sum()
    return w2d.view(1, 1, win_size, win_size)

@torch.no_grad()
def ssim2d(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, win_size: int = 11, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03):
    B, C, H, W = x.shape
    assert C == 1, "SSIM expects single-channel images"
    device = x.device; dtype = x.dtype
    w = _gaussian_window_2d(win_size, sigma, device, dtype)
    pad = win_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    y_pad = F.pad(y, (pad, pad, pad, pad), mode='reflect')
    mu_x = F.conv2d(x_pad, w)
    mu_y = F.conv2d(y_pad, w)
    mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x_pad * x_pad, w) - mu_x2
    sigma_y2 = F.conv2d(y_pad * y_pad, w) - mu_y2
    sigma_xy = F.conv2d(x_pad * y_pad, w) - mu_xy
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.flatten(2).mean(dim=2).squeeze(1)

@torch.no_grad()
def conditional_gaussian_nll(z_obs, z_true_m, idx_obs, idx_mis, mu, Sigma, jitter: float = 1e-4):
    device = z_obs.device
    mu_O = mu[idx_obs].to(device)
    mu_M = mu[idx_mis].to(device)
    S_OO = Sigma[idx_obs][:, idx_obs].to(device).clone()
    S_MO = Sigma[idx_mis][:, idx_obs].to(device)
    S_MM = Sigma[idx_mis][:, idx_mis].to(device)
    S_OO = S_OO + jitter * torch.eye(S_OO.shape[0], device=device)
    L = torch.linalg.cholesky(S_OO)
    diff = (z_obs - mu_O).T
    y = torch.linalg.solve_triangular(L, diff, upper=False)
    x = torch.linalg.solve_triangular(L.T, y, upper=True)
    mean_cond = (mu_M.unsqueeze(1) + S_MO @ x).T
    S_OM = S_MO.T
    yK = torch.linalg.solve_triangular(L, S_OM, upper=False)
    K  = torch.linalg.solve_triangular(L.T, yK, upper=True)
    S_cond = S_MM - S_MO @ K
    S_cond = (S_cond + S_cond.T) * 0.5
    Lc = torch.linalg.cholesky(S_cond + jitter * torch.eye(S_cond.shape[0], device=device))
    resid = (z_true_m - mean_cond)
    t = torch.linalg.solve_triangular(Lc, resid.T, upper=False)
    quad = (t * t).sum(dim=0)
    logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
    d = S_cond.shape[0]
    const = d * torch.log(torch.tensor(2.0 * 3.141592653589793, device=device))
    nll = 0.5 * (logdet + quad + const)
    return nll
# ------------------------- conditional Gaussian -------------------------


@torch.no_grad()
def conditional_gaussian_impute(z_obs, idx_obs, idx_mis, mu, Sigma, jitter=1e-4, sample=False, tau: float = 1.0):
    device = z_obs.device
    mu_O = mu[idx_obs].to(device)
    mu_M = mu[idx_mis].to(device)
    S_OO = Sigma[idx_obs][:, idx_obs].to(device).clone()
    S_MO = Sigma[idx_mis][:, idx_obs].to(device)
    S_MM = Sigma[idx_mis][:, idx_mis].to(device)

    # Stabilize and factorize
    S_OO = S_OO + jitter * torch.eye(S_OO.shape[0], device=device)
    L = torch.linalg.cholesky(S_OO)

    # Solve S_OO^{-1} (z_O - mu_O) for each batch via triangular solves
    # diff: (B, d_obs) -> transpose to (d_obs, B)
    diff = (z_obs - mu_O).T  # (d_obs, B)
    y = torch.linalg.solve_triangular(L, diff, upper=False)         # L y = diff
    x = torch.linalg.solve_triangular(L.T, y, upper=True)           # L^T x = y
    alpha = x.T                                                     # (B, d_obs)

    mean_cond = mu_M + alpha @ S_MO.T               # (B, d_mis)

    if not sample:
        return mean_cond

    # Conditional covariance via solves: S_M|O = S_MM - S_MO S_OO^{-1} S_OM
    S_OM = S_MO.T                                   # (d_obs, d_mis)
    # Solve S_OO K = S_OM  -> K = S_OO^{-1} S_OM using the Cholesky factors
    yK = torch.linalg.solve_triangular(L, S_OM, upper=False)        # (d_obs, d_mis)
    K  = torch.linalg.solve_triangular(L.T, yK, upper=True)         # (d_obs, d_mis)
    S_cond = S_MM - S_MO @ K                        # (d_mis, d_mis)

    # Symmetrize and jitter, then sample
    S_cond = (S_cond + S_cond.T) * 0.5
    Lc = torch.linalg.cholesky(S_cond + jitter * torch.eye(S_cond.shape[0], device=device))
    eps = torch.randn(z_obs.shape[0], idx_mis.numel(), device=device)
    return mean_cond + (tau * eps) @ Lc.T

# ------------------------- model (re)build & checkpoint -------------------------

def build_models_from_config(cfg: dict, device: torch.device) -> List[nn.Module]:
    C, H, W = 1, int(cfg["H"]), int(cfg["W"])
    input_shape = (C, H, W)
    models: List[nn.Module] = []
    for _ in cfg["modalities"]:
        m = create_glow_normalizing_flow_model_2d(
            input_shape=input_shape,
            L=int(cfg["L"]), K=int(cfg["K"]), hidden_channels=int(cfg["hidden"]),
            base=str(cfg.get("base","glow")),
            glowbase_logscale_factor=float(cfg.get("glowbase_logscale_factor",3.0)),
            glowbase_min_log=float(cfg.get("glowbase_min_log",-5.0)),
            glowbase_max_log=float(cfg.get("glowbase_max_log",5.0)),
            split_mode="channel",
            scale=True,
            scale_map=str(cfg.get("scale_map","tanh")),
            leaky=0.0,
            net_actnorm=bool(cfg.get("net_actnorm", False)),
            scale_cap=float(cfg.get("scale_cap", 3.0)),
        ).to(device).float().eval()
        if not hasattr(m, "input_shape"):
            m.input_shape = input_shape
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models

def load_checkpoint(checkpoint_path: Path, device: torch.device):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return blob

def choose_weights(blob: dict, use_ema: bool) -> List[dict]:
    if use_ema and (blob.get("ema") is not None):
        return blob["ema"]
    return blob["models"]

# ------------------------- per-level helpers -------------------------
def build_level_indexers(modalities, shapes_by_view, view_slices, device):
    """
    Returns:
      L: number of latent levels
      level_all_glb: [L] 1D LongTensor of global indices for each level (concat across views)
      level_rel_per_view: [L] dict view->1D LongTensor of relative indices within that level vector
      level_glb_per_view: [L] dict view->1D LongTensor of global indices for that view's slice at the level
      level_dims: [L] total dimensionality per level (sum over views)
    """
    import torch
    assert len(shapes_by_view) == len(modalities)
    L = len(shapes_by_view[0])  # assume equal across views
    # per-view per-level sizes
    sizes = {v: [] for v in modalities}
    for v, shapes in zip(modalities, shapes_by_view):
        for shp in shapes:  # (B,C,H,W)
            C, H, W = int(shp[1]), int(shp[2]), int(shp[3])
            sizes[v].append(C * H * W)

    level_all_glb, level_rel_per_view, level_glb_per_view, level_dims = [], [], [], []
    for l in range(L):
        # global indices for each view at this level
        glb_list = []
        for v in modalities:
            off_in_view = sum(sizes[v][:l])
            glb_start = view_slices[v].start + off_in_view
            glb_end   = glb_start + sizes[v][l]
            glb_list.append(torch.arange(glb_start, glb_end, device=device, dtype=torch.long))
        glb_concat = torch.cat(glb_list, dim=0)

        # relative indices per view within that level vector
        rel_map, glb_map = {}, {}
        rel_off, k = 0, 0
        for v in modalities:
            dvl = sizes[v][l]
            rel = torch.arange(rel_off, rel_off + dvl, device=device, dtype=torch.long)
            rel_map[v] = rel
            glb_map[v] = glb_concat[k:k+dvl]
            rel_off += dvl
            k += dvl

        level_all_glb.append(glb_concat)
        level_rel_per_view.append(rel_map)
        level_glb_per_view.append(glb_map)
        level_dims.append(int(rel_off))

    return L, level_all_glb, level_rel_per_view, level_glb_per_view, level_dims

def fit_cov_oas(Z):  # Z: (N, D) CPU tensor
    """
    Oracle Approximating Shrinkage estimator toward scaled identity.
    Returns (mu, Sigma).
    """
    mu = Z.mean(dim=0)
    Xc = Z - mu
    N, D = Z.shape
    S = (Xc.T @ Xc) / max(1, N - 1)
    trS = torch.trace(S)
    trS2 = torch.sum(S * S)
    I = torch.eye(D)
    # OAS weight (clipped to [0,1])
    num = (1 - 2.0 / D) * trS2 + (trS ** 2)
    den = (N + 1.0 - 2.0 / D) * (trS2 - (trS ** 2) / D) + 1e-12
    alpha = float(torch.clamp(num / den, 0.0, 1.0))
    Sigma = (1 - alpha) * S + alpha * (trS / D) * I
    return mu, Sigma

# ------------------------- level weighting & CCA helpers -------------------------
def compute_level_weights(obs, mis, L, level_stats, level_rel_per_view, mode="none"):
    """
    Returns list w_l (length L). If mode='corr': w_l ∝ ||Σ_MO^l||_F and normalized to sum=1.
    """
    if mode == "none":
        return [1.0 for _ in range(L)]
    ws = []
    for l in range(L):
        S = level_stats[l]["Sigma"]  # CPU
        idx_obs_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in obs], dim=0)
        idx_mis_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in mis], dim=0)
        if idx_obs_rel.numel() == 0 or idx_mis_rel.numel() == 0:
            ws.append(0.0); continue
        block = S[idx_mis_rel][:, idx_obs_rel]
        ws.append(float(torch.linalg.matrix_norm(block, ord='fro').item()))
    s = sum(ws)
    if s <= 0:
        return [1.0 for _ in range(L)]
    return [w/s for w in ws]

def fit_cca_ridge(X, Y, k=128, alpha=1e-2, eps=1e-6):
    """
    X: (N, dx), Y: (N, dy) CPU tensors, centered.
    Returns: {"A","W","k_eff","cond_XtX"}; A in R^{dx×k_eff}, W in R^{k_eff×dy}.
    """
    N, dx = X.shape; dy = Y.shape[1]
    Cxx = (X.T @ X) / max(1, N - 1) + eps * torch.eye(dx)
    Cyy = (Y.T @ Y) / max(1, N - 1) + eps * torch.eye(dy)
    Cxy = (X.T @ Y) / max(1, N - 1)

    Lx = torch.linalg.cholesky(Cxx)
    Ly = torch.linalg.cholesky(Cyy)

    # T = Lx^{-1} Cxy Ly^{-T}
    Z = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    T = torch.linalg.solve_triangular(Ly.T, Z.T, upper=True).T

    U, S, Vh = torch.linalg.svd(T, full_matrices=False)

    # SVD truncation (keep reliable canonical directions)
    svd_eps = 1e-3
    mask = (S > svd_eps)
    k_eff = int(torch.count_nonzero(mask).item())
    k_eff = max(0, min(k_eff, k))

    if k_eff == 0:
        # No reliable canonical directions; return a dummy mapping
        return {"A": None, "W": None, "k_eff": 0, "cond_XtX": float("inf")}

    U_k = U[:, :k_eff]
    A = torch.linalg.solve_triangular(Lx, U_k, upper=False)    # Cxx^{-1/2} U_k
    Xk = X @ A                                                 # (N, k_eff)

    # Ridge solve
    XtX = Xk.T @ Xk
    cond_XtX = float(torch.linalg.cond(XtX + eps * torch.eye(k_eff)).item())
    W = torch.linalg.solve(XtX + alpha * torch.eye(k_eff), Xk.T @ Y)

    return {"A": A, "W": W, "k_eff": k_eff, "cond_XtX": cond_XtX}

def apply_cca_ridge(mapping, z_obs, mu_O, mu_M):
    """z_obs on device; mu_* on device. Returns predicted z_M."""
    A = mapping["A"].to(z_obs.device); W = mapping["W"].to(z_obs.device)
    Xk = (z_obs - mu_O) @ A
    Yc = Xk @ W
    return Yc + mu_M

def safe_latent(z_pred, mu, Sigma, mode="clamp", clamp_k=3.0, gamma=0.7, mahal_pct=0.99):
    """
    z_pred, mu: (B,d) on device. Sigma: (d,d) on device.
    mode:
      - "clamp": elementwise clamp to mu ± k*std
      - "shrink": z = mu + gamma*(z_pred - mu)
      - "mahal": shrink into ellipsoid with radius at desired percentile
    """
    if mode == "none":
        return z_pred
    d = z_pred.shape[1]
    if mode == "clamp":
        std = torch.sqrt(torch.clamp(torch.diag(Sigma), min=1e-12)).unsqueeze(0)
        lo = mu.unsqueeze(0) - clamp_k * std
        hi = mu.unsqueeze(0) + clamp_k * std
        return torch.max(torch.min(z_pred, hi), lo)
    if mode == "shrink":
        return mu.unsqueeze(0) + gamma * (z_pred - mu.unsqueeze(0))
    if mode == "mahal":
        # shrink to lie within a chi-square radius (approx) for percentile
        r2_target = torch.tensor(scipy.stats.chi2.ppf(mahal_pct, df=d), device=z_pred.device, dtype=z_pred.dtype)
        L = torch.linalg.cholesky(Sigma + 1e-6*torch.eye(d, device=Sigma.device, dtype=Sigma.dtype))
        resid = z_pred - mu.unsqueeze(0)          # (B,d)
        t = torch.linalg.solve_triangular(L, resid.T, upper=False)  # (d,B)
        r2 = (t*t).sum(dim=0)                     # (B,)
        scale = torch.sqrt(r2_target / torch.clamp(r2, min=1.0))    # <= 1 for r2>target
        return mu.unsqueeze(0) + (resid * scale.unsqueeze(1))
    return z_pred

def perm_energy(obs, mis, L, level_stats, level_rel_per_view):
    E = 0.0
    for l in range(L):
        S = level_stats[l]["Sigma"]  # CPU
        idx_obs_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in obs], 0)
        idx_mis_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in mis], 0)
        if idx_obs_rel.numel()==0 or idx_mis_rel.numel()==0: 
            continue
        E += torch.linalg.matrix_norm(S[idx_mis_rel][:, idx_obs_rel], ord='fro').item()
    return E

# ------------------------- main pipeline -------------------------

def fit_cov_shrink_diag(Z, lam=0.15):
    """
    Diagonal-target shrinkage: Sigma = (1-lam)*S + lam*diag(S).
    Keeps per-dimension variances; damps cross-covariances.
    Returns (mu, Sigma).
    """
    mu = Z.mean(dim=0)
    Xc = Z - mu
    N, D = Z.shape
    S = (Xc.T @ Xc) / max(1, N - 1)
    diagS = torch.diag(torch.diag(S))
    lam = float(max(0.0, min(1.0, lam)))
    Sigma = (1 - lam) * S + lam * diagS
    return mu, Sigma

def fit_cov_lowrank_diag(Z, rank=64):
    """
    PPCA-style covariance: Sigma ≈ V_r diag(λ_r - σ²)_+ V_r^T + σ² I.
    Computed via economy SVD on centered data. Returns (mu, Sigma).
    """
    mu = Z.mean(dim=0)
    Xc = Z - mu
    N, D = Xc.shape
    if N == 0 or D == 0:
        return mu, torch.eye(D)
    # economy SVD on Xc (N x D): Xc = U diag(S) Vt
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    eig = (S**2) / max(1, N - 1)         # eigenvalues of sample covariance
    r = int(max(1, min(rank, eig.numel())))
    # noise floor σ² from discarded spectrum
    if eig.numel() > r:
        sigma2 = float(eig[r:].mean().item())
    else:
        sigma2 = float(eig[-1].item()) if eig.numel() else 1e-4
    Vr = Vt[:r, :].T                     # (D, r)
    lam_r = eig[:r] - sigma2
    lam_r = torch.clamp(lam_r, min=0.0)
    Sigma = Vr @ torch.diag(lam_r) @ Vr.T + sigma2 * torch.eye(D)
    return mu, Sigma

def main():
    ap = argparse.ArgumentParser("Conditional-Gaussian evaluation for Glow runs")
    ap.add_argument("--run-dir", type=str, required=True, 
                    help="Directory containing training_state.pt")
    ap.add_argument("--checkpoint", type=str, default="", 
                    help="Optional explicit checkpoint path; defaults to <run-dir>/training_state.pt")
    ap.add_argument("--use-ema", action="store_true", 
                    help="Use EMA weights if available")
    ap.add_argument("--gauss-samples", type=int, default=2000, 
                    help="Number of augmented samples to fit the Gaussian")
    ap.add_argument("--eval-samples", type=int, default=256, 
                    help="Number of samples for metric evaluation")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--shrinkage", type=float, default=1e-3, 
                    help="Ridge shrinkage lambda for Sigma")
    ap.add_argument("--cov-debug", action="store_true", help="Print per-level cross-covariance diagnostics")
    ap.add_argument("--cov-estimator", choices=["oas","diag","lowrank"], default="diag",
                    help="Per-level covariance estimator for latent Gaussian")
    ap.add_argument("--cov-lam", type=float, default=0.15, help="Diagonal shrink strength for cov-estimator=diag")
    ap.add_argument("--lowrank-r", type=int, default=64,
                    help="Rank for lowrank+diag covariance estimator")
    ap.add_argument("--jitter", type=float, default=1e-4, 
                    help="PD jitter added to Sigma blocks for solves")
    ap.add_argument("--sample", action="store_true", 
                    help="Draw stochastic samples instead of mean imputation")
    ap.add_argument("--tau", type=float, default=1.0, 
                    help="Sampling temperature for stochastic imputation")
    ap.add_argument("--cov-mode", type=str, default="perlevel", choices=["global","perlevel"], 
                    help="Use one global Gaussian or per-latent-level Gaussians (recommended).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--level-weights", type=str, default="none", choices=["none","corr"],
                    help="Scale per-level cross-covariance by a weight; 'corr' uses Frobenius norm of Σ_MO per level.")
    ap.add_argument("--cca", type=str, default="off", choices=["off","global","perlevel"],
                    help="Use CCA-ridge mapping z_O→z_M instead of Gaussian conditioning.")
    ap.add_argument("--cca-k", type=int, default=128,
                    help="Top-k canonical components for CCA-ridge.")
    ap.add_argument("--cca-alpha", type=float, default=1e-2,
                    help="Ridge regularization for the regression in the CCA head.")
    ap.add_argument("--cca-safe", type=str, default="clamp", choices=["none","clamp","shrink","mahal"],
                    help="Safety for CCA predictions in latent space.")
    ap.add_argument("--cca-clamp-k", type=float, default=3.0, 
                    help="Std devs for clamp mode.")
    ap.add_argument("--cca-gamma", type=float, default=0.7, 
                    help="Shrink factor for shrink mode.")
    ap.add_argument("--cca-mahal-pct", type=float, default=0.99, 
                    help="Target ellipsoid percentile for mahal mode.")
    ap.add_argument("--cca-perm-min-energy", type=float, default=40.0,
                    help="If sum_l ||Σ_MO^l||_F < this, skip CCA and use Gaussian for the whole permutation.")
    ap.add_argument("--cca-min-k", type=int, default=8,
                    help="Skip CCA at a level if effective k after truncation is below this.")
    ap.add_argument("--cca-max-cond", type=float, default=1e6,
                    help="Skip CCA at a level if cond(Xk^T Xk) exceeds this.")

    ap.add_argument("--eval-tag", type=str, default="", 
                    help="Append a tag to eval_gaussian output dir.")

    args = ap.parse_args()

    device = torch.device("cpu") if args.device.lower() == "cpu" else torch.device(args.device)

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "training_state.pt")
    out_dir = run_dir / ("eval_gaussian" + (f"_{args.eval_tag}" if args.eval_tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    blob = load_checkpoint(ckpt_path, device=device)
    cfg = blob.get("config", None)
    assert cfg is not None, "Checkpoint missing 'config'—cannot rebuild models."

    modalities: List[str] = cfg["modalities"]
    if len(modalities) <= 1:
        print("[skip] single-view run; conditional imputation not applicable.")
        return

    # Rebuild models & load weights
    models = build_models_from_config(cfg, device=device)
    state_dicts = choose_weights(blob, use_ema=args.use_ema)
    assert len(state_dicts) == len(models), "Mismatch #models vs #state_dicts"
    for m, sd in zip(models, state_dicts):
        m.load_state_dict(sd, strict=True)
        # warm ActNorm with a dummy real batch later (after loaders)

    # Load data (use same source/settings as training)
    data_source = cfg.get("data", "hcpya")
    gauss_loader, eval_loader = build_loaders_for_eval(
        modalities, int(cfg["H"]), int(cfg["W"]),
        gauss_samples=int(args.gauss_samples),
        eval_samples=int(args.eval_samples),
        batch=int(args.batch),
        num_workers=int(args.num_workers),
        data_source=data_source,
        seed=int(cfg.get("seed", 0)),
    )

    # Warm actnorm with a real batch for each view
    warm_batch = next(iter(eval_loader))
    views = _extract_views_from_batch(warm_batch, num_views=len(modalities))
    for vi, m in enumerate(models):
        warmup_actnorm_with_real_batch(m, to01(views[vi].to(device)).float())

    # ----------------- Collect latents for Gaussian fit -----------------
    z_all_views = []  # concatenated per-sample across views
    with torch.no_grad():
        for batch in gauss_loader:
            xs = _extract_views_from_batch(batch, num_views=len(modalities))
            lat_flat = flatten_from_models(models, xs, device=device)  # list of (B, Dv)
            z_concat = torch.cat(lat_flat, dim=1)  # (B, sum Dv)
            z_all_views.append(z_concat.cpu())
    if not z_all_views:
        raise RuntimeError("No data collected for Gaussian fit; check loaders.")
    Z = torch.cat(z_all_views, dim=0)  # (N, D)

    mu = Z.mean(dim=0)
    Xc = Z - mu
    N, D = Z.shape
    S = (Xc.T @ Xc) / max(1, N - 1)              # sample covariance
    trS = torch.trace(S)
    trS2 = torch.sum(S * S)
    I = torch.eye(D, device=S.device)

    # Oracle Approximating Shrinkage (Chen et al., 2010)
    num = (1 - 2.0 / D) * trS2 + (trS ** 2)
    den = (N + 1.0 - 2.0 / D) * (trS2 - (trS ** 2) / D) + 1e-12
    alpha = torch.clamp(num / den, 0.0, 1.0)

    Sigma = (1 - alpha) * S + alpha * (trS / D) * I

    # --- Build view slices and latent shapes BEFORE per-level indexers ---
    # Grab one warm batch from any existing loader
    try:
        warm_loader = eval_loader if eval_loader is not None else gauss_loader
        warm_batch = next(iter(warm_loader))
    except Exception:
        # If we're in the stats pass context, fall back to gauss_loader
        warm_batch = next(iter(gauss_loader))

    # Extract per-view tensors and map to [0,1]
    xs_warm = _extract_views_from_batch(warm_batch, num_views=len(modalities))
    xs_warm01 = [to01(xv.to(device)).float() for xv in xs_warm]

    # Infer latent shapes for each view by running a single inverse pass
    shapes_by_view = []
    with torch.no_grad():
        for vi, m in enumerate(models):
            z_lvl, _ = m.inverse_and_log_det(xs_warm01[vi])
            if isinstance(z_lvl, torch.Tensor):
                z_lvl = [z_lvl]
            shapes_by_view.append([z_i.shape for z_i in z_lvl])

    # Compute global flat index slices per view (concatenate all levels)
    start = 0
    view_slices = {}
    for vname, shapes in zip(modalities, shapes_by_view):
        d = 0
        for shp in shapes:  # (B,C,H,W)
            C, H, W = int(shp[1]), int(shp[2]), int(shp[3])
            d += C * H * W
        view_slices[vname] = slice(start, start + d)
        start += d
    # --- END shapes/view_slices prep ---

    # Build per-level indexers
    L, level_all_glb, level_rel_per_view, level_glb_per_view, level_dims = build_level_indexers(
        modalities, shapes_by_view, view_slices, device
    )

    # Fit per-level Gaussians (OAS + tiny jitter). Use the same collected latents (z_all_views).
    Z_levels = [ [] for _ in range(L) ]  # list of lists of (B, D_l)
    with torch.no_grad():
        for Zb in z_all_views:  # each is a (B, D_total) CPU tensor chunk
            for l in range(L):
                Z_levels[l].append(Zb[:, level_all_glb[l].cpu()])

    
    level_stats = []
    for l in range(L):
        Zl = torch.cat(Z_levels[l], dim=0)  # (N, D_l)
        estimator = getattr(args, "cov_estimator", "diag")
        if estimator == "lowrank":
            mu_l, Sigma_l = fit_cov_lowrank_diag(Zl, rank=int(getattr(args, "lowrank_r", 64)))
        elif estimator == "diag":
            mu_l, Sigma_l = fit_cov_shrink_diag(Zl, lam=float(getattr(args, "cov_lam", 0.15)))
        else:  # oas
            mu_l, Sigma_l = fit_cov_oas(Zl)
        # common ridge for numerical stability
        D_l = Sigma_l.shape[0]
        Sigma_l = Sigma_l + float(args.shrinkage) * torch.eye(D_l)
        level_stats.append({"mu": mu_l, "Sigma": Sigma_l})
        if getattr(args, "cov_debug", False):
            try:
                S = Sigma_l
                blocks = []
                for i, vi in enumerate(modalities):
                    for j, vj in enumerate(modalities):
                        if i == j: continue
                        ri = level_rel_per_view[l][vi].cpu()
                        rj = level_rel_per_view[l][vj].cpu()
                        if ri.numel()==0 or rj.numel()==0: continue
                        block = S[rj][:, ri]
                        blocks.append(torch.linalg.matrix_norm(block, ord='fro').item())
                tot = torch.linalg.matrix_norm(S, ord='fro').item()
                if blocks:
                    print(f"[cov-debug] level {l}: cross Fro={max(blocks):.3e}  total Fro={tot:.3e}  ratio={max(blocks)/tot:.4f}")
                else:
                    print(f"[cov-debug] level {l}: no cross blocks")
            except Exception as e:
                print(f"[cov-debug] level {l}: error: {e}")



    # Save stats (global + per-level)
    torch.save({
        "mu": mu, "Sigma": Sigma,
        "modalities": modalities,
        "view_slices": {k:(v.start,v.stop) for k,v in view_slices.items()},
        "shapes_by_view": shapes_by_view,
        "L": L, "level_stats": level_stats,
    }, out_dir / "gaussian_stats.pt")




    # (moved save below after per-level stats)

    # ----------------- Evaluate permutations -----------------
    # Define all observed→missing sets for 3 views
    V = modalities
    perms = [
        (["FA"], ["T1","T2"]),
        (["T1"], ["T2","FA"]),
        (["T2"], ["T1","FA"]),
        (["T1","T2"], ["FA"]),
        (["T1","FA"], ["T2"]),
        (["T2","FA"], ["T1"]),
    ]
    # Keep only those consistent with available views
    perms = [p for p in perms if all(v in V for v in p[0]+p[1])]

    metrics_path = out_dir / "impute_metrics.csv"
    with open(metrics_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["observed","missing","N",
                    "PSNR_mean","PSNR_std",
                    "SSIM_mean","SSIM_std",
                    "MSE_mean","MSE_std",
                    "zNLL_mean","zNLL_std",
                    "zNLL_per_dim_mean","zNLL_per_dim_std","d_mis"])

        for obs, mis in perms:
            # index vectors
            idx_obs = torch.cat([torch.arange(view_slices[v].start, view_slices[v].stop) for v in obs]).to(device)
            idx_mis = torch.cat([torch.arange(view_slices[v].start, view_slices[v].stop) for v in mis]).to(device)

            E_perm = perm_energy(obs, mis, L, level_stats, level_rel_per_view)
            use_cca_perm = (args.cca != "off") and (E_perm >= float(args.cca_perm_min_energy))

            # Per-permutation level weights (if perlevel and weighted)
            level_w = None
            if args.cov_mode == "perlevel" and args.level_weights != "none":
                level_w = compute_level_weights(obs, mis, L, level_stats, level_rel_per_view, mode=args.level_weights)

            # Optional CCA-ridge mapping(s) from training latents Z (CPU)
            cca_maps = None
            if use_cca_perm:
                cca_maps = {}
                if args.cca == "global":
                    X = Z[:, idx_obs.cpu()] - mu[idx_obs.cpu()]
                    Y = Z[:, idx_mis.cpu()] - mu[idx_mis.cpu()]
                    cca_maps["global"] = fit_cca_ridge(X, Y, k=int(args.cca_k), alpha=float(args.cca_alpha))
                else:  # perlevel
                    for l in range(L):
                        idx_obs_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in obs], dim=0)
                        idx_mis_rel = torch.cat([level_rel_per_view[l][v].cpu() for v in mis], dim=0)
                        if idx_obs_rel.numel() == 0 or idx_mis_rel.numel() == 0:
                            continue

                        idx_obs_glb = level_all_glb[l].cpu()[idx_obs_rel]   # 1D LongTensor (CPU)
                        idx_mis_glb = level_all_glb[l].cpu()[idx_mis_rel]

                        # Now index CPU latent chunks
                        X = torch.cat([Zb[:, idx_obs_glb] for Zb in z_all_views], dim=0)  # (N, d_obs_l), CPU
                        Y = torch.cat([Zb[:, idx_mis_glb] for Zb in z_all_views], dim=0)  # (N, d_mis_l), CPU

                        # Center per level (explicit Xc, Yc)
                        mu_O_l = X.mean(dim=0)
                        mu_M_l = Y.mean(dim=0)
                        Xc = X - mu_O_l
                        Yc = Y - mu_M_l

                        # Fit CCA-ridge and keep diagnostics
                        m = fit_cca_ridge(Xc, Yc, k=int(args.cca_k), alpha=float(args.cca_alpha))
                        k_eff = int(m.get("k_eff", m.get("k", 0)))
                        cond = float(m.get("cond_XtX", float("inf")))

                        # Gate by thresholds (add these CLI args if you haven't yet)
                        #   --cca-min-k 8   --cca-max-cond 1e6
                        ok = (k_eff >= int(args.cca_min_k)) and (cond <= float(args.cca_max_cond)) and (m.get("A") is not None) and (m.get("W") is not None)

                        cca_maps[l] = {
                            "map": m,          # dict with A, W, k_eff, cond_XtX
                            "mu_O": mu_O_l,    # CPU means
                            "mu_M": mu_M_l,
                            "ok": ok
                        }

            psnrs, mses = [], []
            ssims, nlls, nlls_per_dim = [], [], []
            # also save example grids
            examples_gt = []
            examples_hat = []

            with torch.no_grad():
                for batch in eval_loader:
                    xs = _extract_views_from_batch(batch, num_views=len(modalities))
                    xs01 = [to01(xv.to(device)).float() for xv in xs]

                    # Encode *all* views (for GT); form observed flat vector
                    lat_flat = flatten_from_models(models, xs01, device=device)  # list (B, Dv)
                    z_full = torch.cat(lat_flat, dim=1)  # (B, D)
                    z_obs = z_full[:, idx_obs]
                    # Latent conditional NLL for true z_M
                    if args.cov_mode == "global":
                        z_true_m = z_full[:, idx_mis]
                        nll_batch = conditional_gaussian_nll(
                            z_obs=z_obs, z_true_m=z_true_m,
                            idx_obs=idx_obs, idx_mis=idx_mis,
                            mu=mu.to(device), Sigma=Sigma.to(device), jitter=float(args.jitter))
                        nlls.extend(nll_batch.detach().cpu().tolist())
                        dmis = idx_mis.numel()
                        nlls_per_dim.extend((nll_batch / max(1, dmis)).detach().cpu().tolist())

                    else:
                        nll_sum = torch.zeros(z_full.shape[0], device=device)
                        for l in range(L):
                            z_lvl_full = z_full[:, level_all_glb[l]]
                            idx_obs_rel = torch.cat([level_rel_per_view[l][v] for v in obs], dim=0)
                            idx_mis_rel = torch.cat([level_rel_per_view[l][v] for v in mis], dim=0)
                            if idx_mis_rel.numel() == 0 or idx_obs_rel.numel() == 0:
                                continue
                            z_obs_l = z_lvl_full[:, idx_obs_rel]
                            z_true_m_l = z_lvl_full[:, idx_mis_rel]
                            mu_l = level_stats[l]["mu"].to(device)
                            Sig_l = level_stats[l]["Sigma"].to(device)
                            nll_l = conditional_gaussian_nll(
                                z_obs=z_obs_l, z_true_m=z_true_m_l,
                                idx_obs=idx_obs_rel, idx_mis=idx_mis_rel,
                                mu=mu_l, Sigma=Sig_l, jitter=float(args.jitter))
                            nll_sum = nll_sum + nll_l
                        nlls.extend(nll_sum.detach().cpu().tolist())
                        dmis_total = 0
                        for l in range(L):
                            idx_mis_rel = torch.cat([level_rel_per_view[l][v] for v in mis], dim=0)
                            dmis_total += int(idx_mis_rel.numel())
                        dmis_total = max(1, dmis_total)
                        nlls_per_dim.extend((nll_sum / dmis_total).detach().cpu().tolist())

                    # --- before the imputation branch ---
                    z_mis = None
                    z_full_hat = None

                    # Impute missing block(s): choose imputer
                    if args.cca == "global":
                        mu_O = mu[idx_obs.cpu()].to(device); mu_M = mu[idx_mis.cpu()].to(device)
                        z_mis = apply_cca_ridge(cca_maps["global"], z_obs, mu_O, mu_M)
                        z_mis = safe_latent(z_mis, mu_M, Sigma[idx_mis][:, idx_mis].to(device),
                                            mode=args.cca_safe, clamp_k=args.cca_clamp_k,
                                            gamma=args.cca_gamma, mahal_pct=args.cca_mahal_pct)
                        z_full_hat = z_full.clone()
                        z_full_hat[:, idx_mis] = z_mis

                    elif args.cca == "perlevel":
                        z_full_hat = z_full.clone()
                        for l in range(L):
                            idx_obs_rel = torch.cat([level_rel_per_view[l][v] for v in obs], dim=0)
                            idx_mis_rel = torch.cat([level_rel_per_view[l][v] for v in mis], dim=0)
                            if (l not in cca_maps) or idx_mis_rel.numel() == 0:
                                continue
                            mu_O_l = cca_maps[l]["mu_O"].to(device); mu_M_l = cca_maps[l]["mu_M"].to(device)
                            z_lvl_full = z_full[:, level_all_glb[l]]
                            z_obs_l = z_lvl_full[:, idx_obs_rel]
                            m = cca_maps.get(l, None)
                            use_cca = (m is not None) and m.get("ok", False) and (idx_mis_rel.numel() > 0)

                            if use_cca:
                                z_mis_l_cca = apply_cca_ridge(m["map"], z_obs_l, m["mu_O"].to(device), m["mu_M"].to(device))
                                # optional: safety + blend here
                                z_mis_l = z_mis_l_cca
                            else:
                                # fall back to Gaussian for this level
                                z_mis_l = conditional_gaussian_impute(
                                    z_obs=z_lvl_full[:, idx_obs_rel],
                                    idx_obs=idx_obs_rel, idx_mis=idx_mis_rel,
                                    mu=mu_l, Sigma=level_stats[l]["Sigma"].to(device),
                                    jitter=float(args.jitter), sample=bool(args.sample), tau=float(args.tau)
                                )
                            S_MM_l = level_stats[l]["Sigma"].to(device)[idx_mis_rel][:, idx_mis_rel]
                            z_mis_l = safe_latent(z_mis_l, mu_M_l, S_MM_l,
                                                mode=args.cca_safe, clamp_k=args.cca_clamp_k,
                                                gamma=args.cca_gamma, mahal_pct=args.cca_mahal_pct)
                            idx_mis_glb = torch.cat([level_glb_per_view[l][v] for v in mis], dim=0)
                            z_full_hat[:, idx_mis_glb] = z_mis_l

                    else:
                        if args.cov_mode == "global":
                            z_mis = conditional_gaussian_impute(
                                z_obs=z_obs, idx_obs=idx_obs, idx_mis=idx_mis,
                                mu=mu.to(device), Sigma=Sigma.to(device),
                                jitter=float(args.jitter), sample=bool(args.sample), tau=float(args.tau)
                            )  # (B, d_mis)
                            z_full_hat = z_full.clone()
                            z_full_hat[:, idx_mis] = z_mis
                        else:
                            # Per-level Gaussian (optionally with level weights)
                            z_full_hat = z_full.clone()
                            for l in range(L):
                                z_lvl_full = z_full[:, level_all_glb[l]]
                                idx_obs_rel = torch.cat([level_rel_per_view[l][v] for v in obs], dim=0)
                                idx_mis_rel = torch.cat([level_rel_per_view[l][v] for v in mis], dim=0)
                                if idx_mis_rel.numel() == 0: 
                                    continue
                                mu_l = level_stats[l]["mu"].to(device)
                                S_l = level_stats[l]["Sigma"].clone()
                                if level_w is not None:
                                    wl = level_w[l]
                                    io = idx_mis_rel.cpu(); jo = idx_obs_rel.cpu()
                                    S_l[io[:,None], jo] = S_l[io[:,None], jo] * wl
                                    S_l[jo[:,None], io] = S_l[jo[:,None], io] * wl
                                z_mis_l = conditional_gaussian_impute(
                                    z_obs=z_lvl_full[:, idx_obs_rel],
                                    idx_obs=idx_obs_rel, idx_mis=idx_mis_rel,
                                    mu=mu_l, Sigma=S_l.to(device),
                                    jitter=float(args.jitter), sample=bool(args.sample), tau=float(args.tau)
                                )
                                idx_mis_glb = torch.cat([level_glb_per_view[l][v] for v in mis], dim=0)
                                z_full_hat[:, idx_mis_glb] = z_mis_l

                    # --- unify outputs for downstream metrics ---
                    if z_full_hat is None:
                        raise RuntimeError("Imputation branch did not produce z_full_hat.")
                    if z_mis is None:
                        # derive flat predicted missing slice from the assembled vector
                        z_mis = z_full_hat[:, idx_mis]

                    # Split back into per-view flats
                    z_view_flat_hat = {}
                    for v in V:
                        sl = view_slices[v]
                        z_view_flat_hat[v] = z_full_hat[:, sl.start:sl.stop]  # (B, Dv)

                    # Decode only missing views and compare
                    for v in mis:
                        # Unflatten to z_list by shapes
                        shapes = shapes_by_view[V.index(v)]
                        # compute per-level sizes
                        per_sizes = []
                        for shp in shapes:
                            C = int(shp[1]); per = C
                            for s in shp[2:]:
                                per *= int(s)
                            per_sizes.append(per)
                        # split flat into list
                        chunks = list(torch.split(z_view_flat_hat[v], per_sizes, dim=1))
                        # reshape each chunk to (B,C,H,W)
                        B = chunks[0].shape[0]
                        z_list = []
                        k = 0
                        for shp, ch in zip(shapes, chunks):
                            C = int(shp[1]); H_l = int(shp[2]); W_l = int(shp[3])
                            z_list.append(ch.reshape(B, C, H_l, W_l))

                        # decode via best-available method
                        m = models[V.index(v)]
                        x_hat = None

                        # z_list is the per-level latent list we rebuilt. Some builds expect a list,
                        # others a single tensor if there is only one level.
                        z_pack = z_list if len(z_list) > 1 else z_list[0]

                        # 1) Your fork: forward(z) does z->x
                        out, _ = m.forward_and_log_det(z_pack)        
                        x_hat = out[0] if isinstance(out, (list, tuple)) else out

                        # GT tensor for this view
                        x_gt = xs01[V.index(v)]  # (B,1,H,W), in [0,1]

                        # Ensure shapes match GT
                        if x_hat.shape[-2:] != xs01[V.index(v)].shape[-2:]:
                            x_hat = F.interpolate(x_hat, size=xs01[V.index(v)].shape[-2:], mode="bilinear", align_corners=False)

                        # Metrics
                        mse = torch.mean((x_hat - x_gt) ** 2).item()
                        mses.append(mse)
                        psnrs.append(psnr(x_hat, x_gt))
                        try:
                            ssim_vals = ssim2d(x_hat, x_gt)
                            ssims.extend([float(v) for v in ssim_vals.detach().cpu()])
                        except Exception:
                            pass
                        # Collect a few examples
                        examples_gt.append(x_gt[:4].detach().cpu())
                        examples_hat.append(x_hat[:4].detach().cpu())

            # Write metrics
            if len(psnrs) == 0:
                print(f"[warn] no eval samples for obs={obs}->mis={mis}")
                continue

            PS  = float(np.nanmean(psnrs));   PSs = float(np.nanstd(psnrs))
            ME  = float(np.nanmean(mses));    MEs = float(np.nanstd(mses))

            SS  = float(np.nanmean(ssims)) if ssims else float("nan")
            SSs = float(np.nanstd(ssims))  if ssims else float("nan")
            ZN  = float(np.nanmean(nlls))  if nlls  else float("nan")
            ZNs = float(np.nanstd(nlls))   if nlls  else float("nan")
            ZPd = float(np.nanmean(nlls_per_dim)) if nlls_per_dim else float("nan")
            ZPds= float(np.nanstd(nlls_per_dim))  if nlls_per_dim else float("nan")
            print(f"[{'+'.join(obs)} -> {'+'.join(mis)}] "
                f"PSNR={PS:.2f}±{PSs:.2f}  "
                f"SSIM={(np.mean(ssims) if ssims else float('nan')):.3f}  "
                f"zNLL/d={(np.mean(nlls_per_dim) if nlls_per_dim else float('nan')):.3f}  "
                f"MSE={ME:.5f}±{MEs:.5f} (N={len(psnrs)})")
            
            # Save example grids
            if examples_gt and examples_hat:
                gt = torch.cat(examples_gt, dim=0)[:32]
                hat = torch.cat(examples_hat, dim=0)[:32]
                save_grid(gt, out_dir / f"gt_{'+'.join(mis)}_given_{'+'.join(obs)}.png", nrow=8)
                save_grid(hat, out_dir / f"hat_{'+'.join(mis)}_given_{'+'.join(obs)}.png", nrow=8)

    print("Done. See:", str(out_dir))

if __name__ == "__main__":
    main()
