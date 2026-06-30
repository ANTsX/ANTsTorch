#!/usr/bin/env python3
"""
lamnr_glow_tool_base.py — Shared base for LAM-Flow (Glow 2D & 3D) inference tools.

Contains:
  - All shared utility functions (covariance estimation, Gaussian fitting,
    checkpoint loading, manifest parsing, latent manipulation, SLERP, etc.)
  - GlowToolBase: abstract class parameterised by spatial dimensionality.
    Subclasses (GlowTool2D, GlowTool3D) implement dimension-specific I/O
    while inheriting the full subcommand logic.

Design goals
------------
* Zero duplicated logic between 2D and 3D tools.
* CLI interfaces in 2D and 3D shims remain identical to the original scripts.
* DataParallel ``module.`` prefix is stripped transparently on checkpoint load.
* gc.collect() + explicit del enforced after every heavy encode loop.
* SLERP in recon-interpolate correctly centres on mu before rotation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# MPS safety patch
# ---------------------------------------------------------------------------

_orig_double = torch.Tensor.double

_orig_double = torch.Tensor.double

def _mps_safe_double(self, *args, **kwargs):
    """Downgrade float64 → float32 on Apple Silicon (MPS does not support float64)."""
    if self.device.type == "mps":
        return self.float(*args, **kwargs)
    return _orig_double(self, *args, **kwargs)

torch.Tensor.double = _mps_safe_double

# ---------------------------------------------------------------------------
# Dimension-agnostic parsers (shared)
# ---------------------------------------------------------------------------

def parse_mn(spec: str) -> Tuple[int, int]:
    try:
        m, n = spec.lower().split("x")
        M, N = int(m), int(n)
        assert M > 0 and N > 0
        return M, N
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid-size '{spec}'. Expected like '6x8' (rows×cols)."
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Normalisation (dimension-agnostic: works for 4D and 5D tensors)
# ---------------------------------------------------------------------------

def to01(x: torch.Tensor, eps: float = 1e-8, winsorize: bool = True) -> torch.Tensor:
    """
    Normalise all spatial dimensions to [0, 1].
    Works for (N,C,H,W) and (N,C,H,W,D) tensors.
    If winsorize=True, clips 1%/99% quantiles to preserve contrast.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    if x.ndim < 4:
        return x
    spatial_dims = tuple(range(2, x.ndim))
    if winsorize:
        x_calc = x.float()
        x_flat = x_calc.flatten(start_dim=2)
        q_low  = torch.quantile(x_flat, 0.01, dim=2, keepdim=True).to(x.dtype)
        q_high = torch.quantile(x_flat, 0.99, dim=2, keepdim=True).to(x.dtype)
        view_shape = x.shape[:2] + (1,) * len(spatial_dims)
        x_min = q_low.view(view_shape)
        x_max = q_high.view(view_shape)
        x = torch.clamp(x, min=x_min, max=x_max)
    else:
        x_min = x.amin(dim=spatial_dims, keepdim=True)
        x_max = x.amax(dim=spatial_dims, keepdim=True)
    norm = (x - x_min) / (x_max - x_min + eps)
    return torch.clamp(norm, eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def resolve_ckpt_path(p: Path) -> Path:
    if p.is_dir():
        for name in ("training_state.pt", "checkpoint.pt", "ckpt.pt", "model.pt"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No checkpoint found under directory: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _strip_dp_prefix(sd: dict) -> dict:
    """Remove DataParallel 'module.' prefix from state_dict keys."""
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()}
    return sd


def load_weights_into_model(
    model,
    blob: dict,
    view_idx: int,
    prefer_ema: bool = True,
    view_name: Optional[str] = None,
    cfg_views: Optional[List[str]] = None,
):
    """
    Robustly load per-view weights into *model* from a checkpoint blob.

    Priority:
      (a) EMA list  – blob["ema"][slot]
      (b) models list – blob["models"][slot]
      (c) single state_dict – blob["state_dict"]
      (d) raw param dict

    DataParallel ``module.`` prefixes are stripped transparently.
    """
    def try_load(sd: dict):
        sd = _strip_dp_prefix(sd)
        try:
            model.load_state_dict(sd, strict=True)
            return True, None
        except Exception as e:
            try:
                model.load_state_dict(sd, strict=False)
                return True, f"non-strict: {e}"
            except Exception as e2:
                return False, f"failed: {e2}"

    def extract_sd(candidate):
        if isinstance(candidate, dict):
            if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                return candidate["state_dict"]
            return candidate
        return None

    vidx_eff = int(view_idx)
    if cfg_views and view_name in cfg_views:
        vidx_eff = cfg_views.index(view_name)

    # (a) EMA
    if prefer_ema and isinstance(blob.get("ema"), (list, tuple)) and len(blob["ema"]) > 0:
        k = max(0, min(vidx_eff, len(blob["ema"]) - 1))
        sd = extract_sd(blob["ema"][k])
        if sd is not None:
            ok, _ = try_load(sd)
            if ok:
                return True, ("ema", f"slot={k}")

    # (b) models list
    if isinstance(blob.get("models"), (list, tuple)) and len(blob["models"]) > 0:
        k = max(0, min(vidx_eff, len(blob["models"]) - 1))
        sd = extract_sd(blob["models"][k])
        if sd is not None:
            ok, _ = try_load(sd)
            if ok:
                return True, ("models", f"slot={k}")

    # (c) single state_dict
    if isinstance(blob.get("state_dict"), dict):
        ok, _ = try_load(blob["state_dict"])
        if ok:
            return True, ("state_dict", None)

    # (d) raw dict
    if (isinstance(blob, dict)
            and all(isinstance(k, str) for k in blob.keys())
            and any("." in k for k in blob.keys())):
        ok, _ = try_load(blob)
        if ok:
            return True, ("raw", None)

    return False, ("none", "no recognizable weights in blob")


def _ckpt_fingerprint(ckpt_path: Path) -> str:
    try:
        h = hashlib.sha1()
        h.update(ckpt_path.read_bytes()[:1024 * 1024])
        h.update(str(ckpt_path.stat().st_size).encode())
        h.update(str(int(ckpt_path.stat().st_mtime)).encode())
        return h.hexdigest()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def _read_manifest_csv(manifest_path: Path) -> Dict[str, List[str]]:
    """Read a CSV manifest → {column_header: [path strings]}."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        raise RuntimeError("Manifest is empty.")
    header = [h.strip() for h in rows[0]]
    if any(h == "" for h in header):
        raise RuntimeError("Manifest header has empty column name(s).")
    cols: Dict[str, List[str]] = {h: [] for h in header}
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            raise RuntimeError(
                f"Row {i} has {len(r)} cells but header has {len(header)} columns."
            )
        for h, v in zip(header, r):
            cols[h].append(v.strip())
    return cols


def _resolve_views(
    cols: Dict[str, List[str]],
    manifest_dir: Path,
    views_cli: Optional[str],
) -> Tuple[List[str], List[List[Path]]]:
    """Resolve view names to absolute Path lists from manifest columns."""
    if views_cli is None or views_cli.strip() == "":
        view_names = list(cols.keys())
    else:
        view_names = [v.strip() for v in views_cli.split(",") if v.strip()]
        for v in view_names:
            if v not in cols:
                raise RuntimeError(f"--views specified '{v}', not found in manifest header.")
    per_view_paths: List[List[Path]] = []
    for v in view_names:
        paths_v: List[Path] = []
        for s in cols[v]:
            if s == "":
                raise RuntimeError(f"Manifest has blank cell under view '{v}'.")
            pth = Path(s)
            if not pth.is_absolute():
                pth = (manifest_dir / pth).resolve()
            if not pth.exists() or not pth.is_file():
                raise FileNotFoundError(f"Missing/unreadable file for view '{v}': {pth}")
            paths_v.append(pth)
        per_view_paths.append(paths_v)
    n_set = {len(x) for x in per_view_paths}
    if len(n_set) != 1:
        raise RuntimeError(
            f"Views have inconsistent row counts: {[len(x) for x in per_view_paths]}"
        )
    return view_names, per_view_paths


def _gather_val_paths(val_list: Optional[List[str]], limit: int) -> List[Path]:
    """
    Unified input: tokens may be globs, .txt file lists, or direct image paths.
    Returns up to *limit* unique, existing Paths.
    """
    from glob import glob

    paths: List[Path] = []
    for tok in (val_list or []):
        tok = os.path.expandvars(os.path.expanduser(tok))
        p = Path(tok)
        if p.exists() and p.is_file():
            if p.suffix.lower() in (".txt", ".lst", ".csv"):
                try:
                    with open(p) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                paths.append(
                                    Path(os.path.expandvars(os.path.expanduser(line)))
                                )
                except Exception:
                    pass
            else:
                paths.append(p)
        else:
            for g in sorted(glob(tok, recursive=True)):
                gp = Path(g)
                if gp.exists() and gp.is_file():
                    paths.append(gp)
    seen: set = set()
    uniq: List[Path] = []
    for p in paths:
        if p not in seen and p.exists() and p.is_file():
            uniq.append(p)
            seen.add(p)
        if len(uniq) >= int(limit):
            break
    return uniq


# ---------------------------------------------------------------------------
# Latent space helpers
# ---------------------------------------------------------------------------

def _flatten_latents_by_level(z_list) -> List[torch.Tensor]:
    """
    Flatten per-level latent tensors to (B, D_l) 2-D.
    Supports 4D (2D flows) and 5D (3D flows) tensors.
    """
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
    outs = []
    for z in z_list:
        B = z.shape[0]
        outs.append(z.reshape(B, -1))  # dimension-agnostic flatten
    return outs


def _concat_views_per_level(
    z_per_view_per_level: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """V × L × (N, D_lv)  →  L × (N, sum_v D_lv)."""
    V = len(z_per_view_per_level)
    L = len(z_per_view_per_level[0])
    return [
        torch.cat([z_per_view_per_level[v][l] for v in range(V)], dim=1)
        for l in range(L)
    ]


@torch.no_grad()
def _encode_latents(model, xb: torch.Tensor) -> List[torch.Tensor]:
    """Push batch x → multiscale latents z_list (inverse flow)."""
    device_type = xb.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        if hasattr(model, "inverse_and_log_det"):
            z, _ = model.inverse_and_log_det(xb)
        elif hasattr(model, "inverse"):
            z, _ = model.inverse(xb)
        else:
            raise RuntimeError("Model lacks inverse mapping (inverse_and_log_det).")
    return z if isinstance(z, (list, tuple)) else [z]


@torch.no_grad()
def _decode_latents_generic(
    model,
    z_list: List[torch.Tensor],
    coerce_fn,
    target_size,
) -> torch.Tensor:
    """Decode multiscale latents → image tensor via forward flow. Dimension-agnostic."""
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
    device = z_list[0].device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        if hasattr(model, "forward_and_log_det"):
            xh, _ = model.forward_and_log_det(z_list)
        else:
            raise RuntimeError("Model does not expose forward_and_log_det(z_list).")
    return coerce_fn(xh, target_size)


# ---------------------------------------------------------------------------
# Gaussian / covariance estimation
# ---------------------------------------------------------------------------

def _np_stats(mat) -> Dict[str, float]:
    if isinstance(mat, dict) and mat.get("type") == "lowrank":
        return _lowrank_stats(mat)
    if mat.ndim == 2:
        vals = np.linalg.eigvalsh(mat)
    else:
        vals = np.asarray(mat).ravel()
    vmin = float(vals.min(initial=np.inf))
    vmax = float(vals.max(initial=0.0))
    cond = float(vmax / (vmin + 1e-12)) if vmax > 0 else float("inf")
    return {"lambda_min": vmin, "lambda_max": vmax, "cond": cond}


def _cov_full(X: np.ndarray, ridge: float) -> np.ndarray:
    N = X.shape[0]
    S = (X.T @ X) / max(1, (N - 1))
    if ridge and ridge > 0.0:
        S = S + float(ridge) * np.eye(S.shape[0], dtype=S.dtype)
    return S


def _cov_diag(X: np.ndarray, ridge: float) -> np.ndarray:
    var = X.var(axis=0, ddof=1)
    if ridge and ridge > 0.0:
        var = var + float(ridge)
    return var


def _cov_oas(X: np.ndarray, extra_ridge: float) -> np.ndarray:
    """
    Oracle Approximating Shrinkage toward scaled identity: (1-a)S + a*(tr(S)/p)I
    Uses Chen et al. 2010 closed form.
    """
    N, p = X.shape
    S = (X.T @ X) / max(1, (N - 1))
    mu = np.trace(S) / p
    # Frobenius norm of S
    trS2 = float(np.sum(S * S))
    trS = float(np.trace(S))
    # OAS shrinkage factor
    # guard for tiny denominators
    denom = (N + 1 - 2.0 / p) * (trS2 - (trS * trS) / p)
    if denom <= 0:
        a = 1.0
    else:
        a = ((1.0 - 2.0 / p) * trS2 + (trS * trS)) / denom
        a = max(0.0, min(1.0, a))
    S_shrunk = (1.0 - a) * S + a * mu * np.eye(p, dtype=S.dtype)
    if extra_ridge and extra_ridge > 0.0:
        S_shrunk = S_shrunk + float(extra_ridge) * np.eye(p, dtype=S.dtype)
    return S_shrunk


def _lowrank_from_Xc(Xc: np.ndarray, rank: int, sigma2: float | str, extra_ridge: float) -> dict:
    N, D = Xc.shape
    rmax = min(D, max(1, N - 1))
    r = int(max(1, min(rank, rmax)))
    
    # 1. SVD Randomisée via PyTorch (Extrêmement rapide)
    # Convertir en tenseur PyTorch
    Xc_tensor = torch.tensor(Xc, dtype=torch.float32)
    
    # q=r force l'algorithme à ne chercher que les 'r' premières composantes (ici 256)
    _, S_tensor, V_tensor = torch.svd_lowrank(Xc_tensor, q=r)
    
    Svals = S_tensor.numpy()
    # torch.svd_lowrank retourne V avec la dimension (D, r)
    # Cela correspond directement à la forme transposée dont nous avons besoin pour U_cov
    U_cov = V_tensor.numpy().copy() 
    
    # 2. Calcul des valeurs propres pour les composantes principales
    eig_r = (Svals ** 2) / max(1, (N - 1))
    
    # 3. Estimation du bruit résiduel (sigma2)
    if isinstance(sigma2, str) and sigma2.lower() == "auto":
        # Variance totale = somme des carrés des éléments de Xc divisée par (N-1)
        total_variance = np.sum(Xc ** 2) / max(1, (N - 1))
        explained_variance = np.sum(eig_r)
        
        # Le reste de la variance est attribué au bruit
        residual_variance = max(0.0, total_variance - explained_variance)
        num_remaining_eigs = min(N, D) - r
        
        sigma2_val = float(residual_variance / num_remaining_eigs) if num_remaining_eigs > 0 else 0.0
    else:
        sigma2_val = float(sigma2)
        
    sigma2_val += extra_ridge
    return {"type": "lowrank", "U": U_cov, "eig": eig_r, "sigma2": sigma2_val}

def _lowrank_stats(sig: dict) -> dict:
    eig    = np.asarray(sig.get("eig", []), dtype=float)
    sigma2 = float(sig.get("sigma2", 0.0))
    lam_min = float(sigma2)
    lam_max = float((eig.max() if eig.size > 0 else 0.0) + sigma2)
    cond    = float(lam_max / (lam_min + 1e-12)) if lam_max > 0 else float("inf")
    return {"lambda_min": lam_min, "lambda_max": lam_max, "cond": cond}

def _fit_gaussian_blocks(X_blocks: List[np.ndarray], estimator: str, shrinkage: float, cov_lam: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit Gaussian to concatenated blocks (per level). Returns (mu, Sigma, meta).
    For 'diag', Sigma is 1D vector of variances.
    """
    X = np.concatenate(X_blocks, axis=1) if len(X_blocks) > 1 else X_blocks[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    est = estimator.lower()
    if est == "full":
        Sigma = _cov_full(Xc, ridge=float(shrinkage) + float(cov_lam))
    elif est == "diag":
        Sigma = _cov_diag(Xc, ridge=float(shrinkage) + float(cov_lam))
    elif est in ("oas", "lw", "ledoitwolf"):
        # Treat lw as oas for now; both shrink toward scaled identity; OAS has closed form
        Sigma = _cov_oas(Xc, extra_ridge=float(cov_lam))
    else:
        raise RuntimeError(f"Unknown --cov-estimator: {estimator}")
    stats = _np_stats(Sigma if Sigma.ndim == 2 else Sigma)
    return mu, Sigma, stats

def _sanitize_latents_array(X, cap_quantile=99.9, hard_cap=None):
    X = np.asarray(X, dtype=np.float64)
    stats = {}
    nf = ~np.isfinite(X)
    nf_count = int(nf.sum())
    if nf_count:
        X[nf] = 0.0
    stats["nonfinite"] = nf_count

    if hard_cap is None:
        q = np.percentile(np.abs(X), [50, 90, 99, cap_quantile])
        cap = float(q[-1] + 1e-12)
        stats["abs_quantiles"] = {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), f"p{cap_quantile}": float(q[3])}
    else:
        cap = float(hard_cap)
        stats["abs_quantiles"] = None

    pre = X.copy()
    np.clip(X, -cap, cap, out=X)
    clipped = int(np.sum(pre != X))
    stats["cap"] = cap
    stats["clipped"] = clipped
    return X, stats


# ---------------------------------------------------------------------------
# Gaussian model serialisation / deserialisation
# ---------------------------------------------------------------------------

def _save_gauss_npz(blob: Dict[str, Any], out_path: Path):
    pack = {
        "mode":      blob["mode"],
        "estimator": blob["estimator"],
        "N":  np.int64(blob["N"]),
        "H":  np.int64(blob["H"]),
        "W":  np.int64(blob["W"]),
        "L":  np.int64(blob["L"]),
        "views":       np.array(blob["views"], dtype=object),
        "dims_json":   json.dumps(blob["dims_per_level_per_view"]),
        "shapes_json": json.dumps(blob["shapes_by_view"]),
        "slices_json": json.dumps(blob["level_view_slices"]),
        "stats_json":  json.dumps(blob["stats"]),
    }
    # Optionally persist D for 3D blobs
    if blob.get("D") is not None:
        pack["D"] = np.int64(blob["D"])

    if blob["mode"] == "perlevel":
        for i, mu in enumerate(blob["mu"]):
            pack[f"mu_{i}"] = np.asarray(mu)
            S = blob["Sigma"][i]
            if isinstance(S, dict) and S.get("type") == "lowrank":
                pack[f"Sigma_{i}_type"]   = "lowrank"
                pack[f"Sigma_{i}_U"]      = np.asarray(S["U"])
                pack[f"Sigma_{i}_eig"]    = np.asarray(S["eig"])
                pack[f"Sigma_{i}_sigma2"] = np.asarray([S["sigma2"]], dtype=np.float64)
            else:
                pack[f"Sigma_{i}"] = np.asarray(S)
    else:
        pack["mu"] = np.asarray(blob["mu"])
        S = blob["Sigma"]
        if isinstance(S, dict) and S.get("type") == "lowrank":
            pack["Sigma_type"]   = "lowrank"
            pack["Sigma_U"]      = np.asarray(S["U"])
            pack["Sigma_eig"]    = np.asarray(S["eig"])
            pack["Sigma_sigma2"] = np.asarray([S["sigma2"]], dtype=np.float64)
        else:
            pack["Sigma"] = np.asarray(S)
    np.savez_compressed(out_path, **pack)


def _load_gaussian_model(gauss_path: Path) -> Dict[str, Any]:
    """
    Load Gaussian model saved by gauss-fit (.pt or .npz).
    Compatible with both 2D and 3D blobs (D key optional for 2D).
    """
    gauss_path = Path(gauss_path)
    if not gauss_path.exists():
        raise FileNotFoundError(f"Gaussian file not found: {gauss_path}")

    if str(gauss_path).endswith(".pt"):
        try:
            blob = torch.load(gauss_path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(
                f"[warn] weights_only load failed ({e.__class__.__name__}: {e}); "
                "retrying without weights_only"
            )
            blob = torch.load(gauss_path, map_location="cpu")
        return blob

    npz  = np.load(str(gauss_path), allow_pickle=True)
    keys = set(npz.files)
    blob: Dict[str, Any] = {}

    def _scalar(k, cast=int, default=None):
        if k in keys:
            try:
                return cast(np.array(npz[k]).ravel()[0])
            except Exception:
                try:
                    return cast(npz[k].tolist())
                except Exception:
                    return cast(npz[k])
        return default

    blob["mode"]      = np.array(npz["mode"]).tolist() if "mode" in keys else "perlevel"
    blob["estimator"] = np.array(npz["estimator"]).tolist() if "estimator" in keys else "full"
    blob["N"] = _scalar("N", int, None)
    blob["H"] = _scalar("H", int, None)
    blob["W"] = _scalar("W", int, None)
    blob["D"] = _scalar("D", int, None)   # None for 2D blobs
    blob["L"] = _scalar("L", int, None)

    if "views" in keys:
        vv = np.array(npz["views"]).tolist()
        blob["views"] = [str(x) for x in (vv if isinstance(vv, list) else [vv])]
    if "dims_json"   in keys:
        blob["dims_per_level_per_view"] = json.loads(str(np.array(npz["dims_json"]).tolist()))
    if "stats_json"  in keys:
        blob["stats"]           = json.loads(str(np.array(npz["stats_json"]).tolist()))
    if "shapes_json" in keys:
        blob["shapes_by_view"]  = json.loads(str(np.array(npz["shapes_json"]).tolist()))
    if "slices_json" in keys:
        blob["level_view_slices"] = json.loads(str(np.array(npz["slices_json"]).tolist()))

    L = int(blob.get("L", 0) or 0)
    if any(f.startswith("mu_") for f in keys):
        mu_list, Sig_list = [], []
        for i in range(L):
            mu_list.append(np.array(npz[f"mu_{i}"]))
            if (f"Sigma_{i}_type" in keys
                    and str(np.array(npz[f"Sigma_{i}_type"]).tolist()) == "lowrank"):
                Sig_list.append({
                    "type":   "lowrank",
                    "U":      np.array(npz[f"Sigma_{i}_U"]),
                    "eig":    np.array(npz[f"Sigma_{i}_eig"]),
                    "sigma2": float(np.array(npz[f"Sigma_{i}_sigma2"]).ravel()[0]),
                })
            else:
                Sig_list.append(np.array(npz.get(f"Sigma_{i}")))
        blob["mu"]    = mu_list
        blob["Sigma"] = Sig_list
        blob["mode"]  = "perlevel"
        return blob

    if "mu" in keys:
        blob["mu"] = np.array(npz["mu"])
        if "Sigma_type" in keys and str(np.array(npz["Sigma_type"]).tolist()) == "lowrank":
            blob["Sigma"] = {
                "type":   "lowrank",
                "U":      np.array(npz["Sigma_U"]),
                "eig":    np.array(npz["Sigma_eig"]),
                "sigma2": float(np.array(npz["Sigma_sigma2"]).ravel()[0]),
            }
        elif "Sigma" in keys:
            blob["Sigma"] = np.array(npz["Sigma"])
        return blob

    if "mu" in keys and np.array(npz["mu"]).dtype == object:
        blob["mu"] = np.array(npz["mu"]).tolist()
        if "Sigma" in keys:
            blob["Sigma"] = np.array(npz["Sigma"]).tolist()
        return blob

    raise RuntimeError(f"Unrecognized NPZ contents in {gauss_path}; keys={sorted(keys)}")


# ---------------------------------------------------------------------------
# Gaussian blob validation
# ---------------------------------------------------------------------------

def _validate_gauss_blob(g: dict):
    """
    Validate Gaussian blob from gauss-fit.
    Returns (views, dims_tbl, shapes_by_view, L) or raises RuntimeError.
    Works for both 2D (shapes: C,H,W) and 3D (shapes: C,H,W,D).
    """
    import math as _math

    def _prod_all(t):
        try:
            return _math.prod(int(v) for v in t)
        except Exception:
            return None

    errors = []
    views         = g.get("views", None)
    dims_tbl      = g.get("dims_per_level_per_view", None)
    shapes_by_view= g.get("shapes_by_view", None)
    L_raw         = g.get("L", None)

    if not isinstance(views, (list, tuple)) or len(views) == 0 or not all(
        isinstance(v, str) for v in views
    ):
        errors.append(
            f"- 'views' missing or invalid; expected non-empty list[str], "
            f"got: {type(views).__name__}"
        )
    if dims_tbl is None or not isinstance(dims_tbl, (list, tuple)):
        errors.append(
            f"- 'dims_per_level_per_view' missing; expected list[list[int]], "
            f"got: {type(dims_tbl).__name__}"
        )
    if shapes_by_view is None or not isinstance(shapes_by_view, (list, tuple)):
        errors.append(
            f"- 'shapes_by_view' missing; expected list[list[tuple]], "
            f"got: {type(shapes_by_view).__name__}"
        )
    try:
        L = int(L_raw)
        if L <= 0:
            errors.append(f"- 'L' non-positive: {L_raw!r}")
    except Exception:
        errors.append(f"- 'L' missing or not an int: {L_raw!r}")

    if errors:
        raise RuntimeError("[gauss] Invalid Gaussian file structure:\n" + "\n".join(errors))

    V = len(views)
    if len(dims_tbl) != V:
        errors.append(f"- dims_per_level_per_view V={len(dims_tbl)} but views V={V}")
    if len(shapes_by_view) != V:
        errors.append(f"- shapes_by_view V={len(shapes_by_view)} but views V={V}")

    bad_d = [vi for vi in range(V)
             if not isinstance(dims_tbl[vi], (list, tuple)) or len(dims_tbl[vi]) != L]
    bad_s = [vi for vi in range(V)
             if not isinstance(shapes_by_view[vi], (list, tuple)) or len(shapes_by_view[vi]) != L]

    mismatches = []
    for vi in range(V):
        if vi in bad_d or vi in bad_s:
            continue
        for l in range(L):
            try:
                d_tbl = int(np.asarray(dims_tbl[vi][l]).item()
                            if hasattr(dims_tbl[vi][l], "item") else dims_tbl[vi][l])
            except Exception:
                d_tbl = None
            d_shp = _prod_all(shapes_by_view[vi][l])
            if d_tbl is None or d_shp is None or d_tbl != d_shp:
                mismatches.append((vi, l, d_tbl, d_shp))
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break

    if mismatches:
        msg = "\n".join(
            [f"  - view[{vi}]='{views[vi]}', level {l}: dims_tbl={dt} vs Prod(shape)={ds}"
             for (vi, l, dt, ds) in mismatches]
        )
        errors.append(
            f"- dims_per_level_per_view does not match shapes_by_view (up to 20):\n{msg}"
        )

    if errors:
        footer = (
            "\nHints:\n"
            "  • Re-run gauss-fit if model config (H/W/D, K, levels) changed.\n"
            "  • --views in gauss-fit must match the manifest column order.\n"
            "  • Ensure gauss file was written by the current version."
        )
        raise RuntimeError(
            "[gauss] Inconsistent Gaussian metadata:\n" + "\n".join(errors) + footer
        )

    dims_tbl_py = [
        [int(np.asarray(d).item() if hasattr(d, "item") else d) for d in row]
        for row in dims_tbl
    ]
    return views, dims_tbl_py, shapes_by_view, L


# ---------------------------------------------------------------------------
# Lowrank conditional mean (push-through / Woodbury identity)
# ---------------------------------------------------------------------------

def _cond_mean_block_lowrank(U: np.ndarray, eig: np.ndarray, sigma2: float,
                             idx_U: list, idx_O: list,
                             mu: np.ndarray, ZO: np.ndarray,
                             base_ridge: float = 1e-4):
    U = np.asarray(U, dtype=np.float64)
    eig = np.asarray(eig, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    
    if ZO.ndim == 1:
        ZO = ZO[None, :]
        
    U_O = U[idx_O, :] 
    U_U = U[idx_U, :] 
    
    s2 = max(float(sigma2) + float(base_ridge), 1e-6)
        
    dO = (ZO - mu[idx_O][None, :]).T 
    
    sqrt_eig = np.sqrt(np.clip(eig, 0.0, None))
    A_T = U_O.T * sqrt_eig[:, None]  
    
    K = A_T @ A_T.T + s2 * np.eye(len(eig), dtype=np.float64) 
    rhs = A_T @ dO 
    
    try:
        w = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K, rhs, rcond=None)
        
    v_target = sqrt_eig[:, None] * w 
    projection = U_U @ v_target 
    zU = mu[idx_U][:, None] + projection
    
    return zU.T 

# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_temperature(model, n: int, temp: float):
    """
    Robust sampler that tries several APIs to honour *temp*:
      1) model.sample(n, temperature=temp)
      2) model.sample(n, T=temp)
      3) Set q0.T / q0.temperature temporarily, then model.sample(n)
      4) Fallback: model.sample(n) with warning
    """
    def _try_set_q0_temp(model, temp):
        q0 = getattr(model, "q0", None)
        if q0 is None:
            return None
        bases = []
        if isinstance(q0, (list, tuple)):
            bases = list(q0)
        else:
            for attr in ("q0s", "bases", "base"):
                cand = getattr(q0, attr, None)
                if cand is not None:
                    bases = list(cand) if isinstance(cand, (list, tuple)) else [cand]
                    break
            if not bases:
                bases = [q0]
        prev = []
        did_any = False
        for b in bases:
            for attr in ("T", "temperature"):
                if hasattr(b, attr):
                    try:
                        prev.append((attr, b, float(getattr(b, attr))))
                        setattr(b, attr, float(temp))
                        did_any = True
                        break
                    except Exception:
                        pass
        return (bases, prev) if did_any else None

    def _restore_q0_temp(handle):
        if handle is None:
            return
        _, prev = handle
        for attr, b, val in prev:
            try:
                setattr(b, attr, val)
            except Exception:
                pass

    try:
        return model.sample(n, temperature=float(temp))
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return model.sample(n, T=float(temp))
    except TypeError:
        pass
    except Exception:
        pass
    handle = None
    try:
        handle = _try_set_q0_temp(model, float(temp))
        if handle is not None:
            return model.sample(n)
    finally:
        _restore_q0_temp(handle)
    print(
        f"[warn] temperature={temp} may be ignored — no compatible API found."
    )
    return model.sample(n)


# ---------------------------------------------------------------------------
# Reconstruction utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_batch(model, xb: torch.Tensor):
    """Round-trip x → z → x_hat using the flow APIs."""
    if hasattr(model, "inverse_and_log_det"):
        z, _ = model.inverse_and_log_det(xb)
    elif hasattr(model, "inverse"):
        z, _ = model.inverse(xb)
    else:
        raise RuntimeError("Model lacks inverse mapping.")
    z_list = z if isinstance(z, (list, tuple)) else [z]
    if hasattr(model, "forward_and_log_det"):
        xh, _ = model.forward_and_log_det(z_list)
        return xh
    raise RuntimeError("Model does not expose forward_and_log_det.")


# ---------------------------------------------------------------------------
# Warmup utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def warmup_actnorm_with_real_batch(model, x_real: torch.Tensor):
    """Run one real-batch pass to stabilise ActNorm data-dependent init."""
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        return
    x1 = x_real[:1].to(dev, torch.float32)
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue


# ---------------------------------------------------------------------------
# GlowToolBase — abstract class
# ---------------------------------------------------------------------------

class GlowToolBase(ABC):
    """
    Abstract base for lamnr_glow_tool_2d and lamnr_glow_tool_3d.

    Subclasses must implement the dimension-specific abstract methods below.
    All shared subcommand logic (gauss-fit, gauss-impute, recon-interpolate,
    calc-distance, recon-template, recon-cohort-template, recon-temperature,
    recon) is implemented here and calls self.* hooks.
    """

    # ------------------------------------------------------------------ #
    # Abstract spatial properties                                          #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def ndim(self) -> int:
        """4 for 2D, 5 for 3D."""
        ...

    @property
    @abstractmethod
    def interp_mode(self) -> str:
        """'bilinear' for 2D, 'trilinear' for 3D."""
        ...

    @property
    @abstractmethod
    def default_cov_estimator(self) -> str:
        """Default covariance estimator ('full' for 2D, 'lowrank' for 3D)."""
        ...

    @property
    @abstractmethod
    def default_cov_rank(self) -> int:
        """Default rank for lowrank covariance (64 for 2D, 128 for 3D)."""
        ...

    # ------------------------------------------------------------------ #
    # Abstract I/O methods                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def parse_size(self, spec: str):
        """Parse size string → tuple of ints (H,W) or (H,W,D)."""
        ...

    @abstractmethod
    def parse_spacing(self, spec: str):
        """Parse spacing string → tuple of floats."""
        ...

    @abstractmethod
    def read_image(self, path: Path, target_size) -> torch.Tensor:
        """Read one image/volume from disk → float32 tensor in [0,1]."""
        ...

    @abstractmethod
    def coerce_nd(self, x, target_size) -> torch.Tensor:
        """Coerce model output to the canonical ndim tensor shape."""
        ...

    @abstractmethod
    def save_volume(self, x: torch.Tensor, path, **kw):
        """Save a volume/slice tensor to disk."""
        ...

    @abstractmethod
    def build_model(self, cfg: dict, device: torch.device, target_size) -> nn.Module:
        """Build and return a Glow model moved to device and in eval/float mode."""
        ...

    @abstractmethod
    def prime_if_needed(self, model, size, device):
        """Run a dummy forward pass to initialise ActNorm statistics."""
        ...

    @abstractmethod
    def edit_latents_to_mean(
        self,
        z_list: List[torch.Tensor],
        gauss_blob: dict,
        view_name: str,
        levels_to_edit: List[int],
        **kw,
    ) -> List[torch.Tensor]:
        """Edit latents at specified levels (mean, zero, pc, pc_denoise)."""
        ...

    # ------------------------------------------------------------------ #
    # Optional hooks (subclasses may override)                             #
    # ------------------------------------------------------------------ #

    def _add_size_arg(self, ap: argparse.ArgumentParser, required: bool = True):
        """
        Add the dimension-specific size argument.
        2D: --slice-axis / --slice-index  (image size comes from checkpoint config)
        3D: --volume-size HxWxD
        Default: no-op (each subclass overrides in the shim).
        """

    def _get_target_size(self, args, cfg: dict):
        """Return target_size from args or config dict. Subclass must implement."""
        raise NotImplementedError

    def _add_sampling_size_arg(self, ap: argparse.ArgumentParser):
        """Add the size argument used by the sample subcommand."""

    # ------------------------------------------------------------------ #
    # Shared decode helper                                                 #
    # ------------------------------------------------------------------ #

    def decode_latents(
        self,
        model,
        z_list: List[torch.Tensor],
        target_size,
    ) -> torch.Tensor:
        """Decode latent list → image tensor using self.coerce_nd."""
        return _decode_latents_generic(
            model, z_list,
            lambda x, sz: self.coerce_nd(x, sz),
            target_size,
        )

    # ------------------------------------------------------------------ #
    # CLI dispatch                                                         #
    # ------------------------------------------------------------------ #

    def run(self, argv=None):
        import inspect
        if argv is None:
            argv = sys.argv[1:]
        cmds = {}
        for name, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("cmd_"):
                key = name[4:].replace("_", "-")
                cmds[key] = fn

        if not argv or argv[0] in ("-h", "--help", "--version"):
            if argv and argv[0] == "--version":
                from importlib.metadata import version
                try:
                    print(version("lamnrflows"))
                except Exception:
                    print("version unknown")
                return
            print(f"Available subcommands: {', '.join(sorted(cmds))}")
            return

        sub = argv[0]
        if sub not in cmds:
            print(f"[error] Unknown subcommand '{sub}'. Available: {sorted(cmds)}")
            sys.exit(1)
        cmds[sub](argv[1:])

    # ================================================================== #
    # Shared subcommands (concrete implementations)                       #
    # These methods call self.* hooks for dimension-specific operations.  #
    # ================================================================== #

    # ------------------------------------------------------------------
    # gauss-fit
    # ------------------------------------------------------------------

    def cmd_gauss_fit(self, argv=None):
        """Encode cohort → fit per-level Gaussian model → save .npz/.pt."""
        ap = argparse.ArgumentParser("gauss-fit")
        ap.add_argument("--ckpt",            type=str, required=True)
        ap.add_argument("--manifest",        type=str, required=True)
        ap.add_argument("--views",           type=str, default=None)
        ap.add_argument("--batch",           type=int, default=4)
        ap.add_argument("--devices",         type=str, default="cuda:0")
        ap.add_argument("--cov-mode",        default="perlevel",
                        choices=["perlevel", "merged"])
        ap.add_argument("--cov-estimator",   default=self.default_cov_estimator,
                        choices=["full", "diag", "oas", "lw", "lowrank"])
        ap.add_argument("--rank",            type=int, default=self.default_cov_rank)
        ap.add_argument("--sigma2",          type=str, default="auto")
        ap.add_argument("--shrinkage",       type=str, default="1e-6")
        ap.add_argument("--cov-lam",         type=float, default=1e-6)
        ap.add_argument("--jitter",          type=float, default=1e-4)
        ap.add_argument("--gauss-out",       type=str, required=True)
        ap.add_argument("--gauss-summary",   type=str, default="")
        ap.add_argument("--save-fp",         type=int, default=64,
                        choices=[16, 32, 64])
        ap.add_argument("--ema",             action=argparse.BooleanOptionalAction,
                        default=True)
        ap.add_argument("--no-scrub",        action="store_true", default=False,
                        help="Désactive le rejet automatique des sujets aberrants.")
        ap.add_argument("--scrub-threshold", type=float, default=None,
                        help="Seuil de variance/déviation pour exclure un sujet.")
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        shrinkage = float(args.shrinkage)

        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        manifest_path = Path(args.manifest)
        cols = _read_manifest_csv(manifest_path)
        view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)

        V = len(view_names)
        N = len(per_view_paths[0])
        print(f"[gauss-fit] {V} view(s), {N} subjects, target_size={target_size}")
        print(f"[gauss-fit] cov-mode={args.cov_mode}, estimator={args.cov_estimator}, "
              f"rank={args.rank}, sigma2={args.sigma2}")

        fp_dtype = {16: torch.float16, 32: torch.float32, 64: torch.float64}[args.save_fp]

        # ── Per-view encoding ────────────────────────────────────────────
        latents_per_view: List[List[List[torch.Tensor]]] = []  # V × L × subjects
        bad_paths: List[str] = []
        N_original = N

        for vi, v_name in enumerate(view_names):
            print(f"[gauss-fit] encoding view '{v_name}' ({vi+1}/{V})...")
            model = self.build_model(cfg, device, target_size)
            ok, src = load_weights_into_model(
                model, blob, vi, prefer_ema=args.ema,
                view_name=v_name, cfg_views=cfg_views
            )

            if not ok:
                raise RuntimeError(f"Could not load weights for view '{v_name}'")
            print(f"  [ckpt] loaded from {src}")
            self.prime_if_needed(model, target_size, device)

            # Probe latent shapes with a dummy forward
            z_shapes = self._probe_latent_shapes(model, target_size, device)
            L = len(z_shapes)
            print(f"  [gauss-fit] L={L} levels, shapes={z_shapes}")

            # Encode all subjects
            latents_this_view: List[List[torch.Tensor]] = [[] for _ in range(L)]
            xbatch: List[torch.Tensor] = []
            path_batch: List[str] = []

            def _flush_batch(xlist, pbatch):
                if not xlist:
                    return
                try:
                    xb = torch.stack(xlist, dim=0).to(device)

                    z_list = _encode_latents(model, xb)
                    z_flat = _flatten_latents_by_level(z_list)
                    for l_idx in range(L):
                        latents_this_view[l_idx].append(
                            z_flat[l_idx].cpu().to(fp_dtype)
                        )
                    del xb, z_list, z_flat

                except Exception as e:
                    print(f"  [ERROR] Échec encodage pour le batch {pbatch}")
                    print(f"  [ERROR] Message d'erreur : {e}")
                    import traceback
                    traceback.print_exc()
                    for p in pbatch:
                        bad_paths.append(str(p))

                finally:
                    gc.collect()
                    
            for i, img_path in enumerate(tqdm(per_view_paths[vi], desc=f"  {v_name}")):
                try:
                    x = self.read_image(img_path, target_size)
                    xbatch.append(x)
                    path_batch.append(str(img_path))
                except Exception as e:
                    # Rendre l'erreur plus visible et afficher le traceback
                    print(f"\n[CRITICAL ERROR] Failed to process {img_path}")
                    print(f"Error details: {e}")
                    import traceback
                    traceback.print_exc()
                    bad_paths.append(str(img_path))
                    continue

                if len(xbatch) >= int(args.batch):
                    _flush_batch(xbatch, path_batch)
                    xbatch, path_batch = [], []
                    # Free ANTs C++ objects
                    gc.collect()

            _flush_batch(xbatch, path_batch)
            del xbatch, path_batch, model
            gc.collect()

            latents_per_view.append(latents_this_view)

        # Diagnostic de population
        for vi_ in range(V):
            for l_idx in range(L):
                count = len(latents_per_view[vi_][l_idx])
                print(f"[debug] View {vi_}, Level {l_idx} has {count} latents.")

        def _scrub_outliers(threshold=1e6):
            """
            Exclut les sujets dont la valeur absolue maximale dépasse le seuil,
            reproduisant la logique de main_gauss_fit.
            """
            N_enc = min(len(latents_per_view[vi_][0]) for vi_ in range(V))
            bad_indices = set()
            
            for vi_ in range(V):
                for l_idx in range(L):
                    chunks = latents_per_view[vi_][l_idx]
                    if not chunks:
                        continue
                    
                    # Les tenseurs sont déjà aplatis en (N, D_l)
                    Z = torch.cat(chunks, dim=0)
                    row_max = Z.abs().amax(dim=1)
                    
                    for i in range(N_enc):
                        if row_max[i].item() > threshold:
                            bad_indices.add(i)
            
            keep_idx = [i for i in range(N_enc) if i not in bad_indices]
            
            if bad_indices:
                print(f"[scrub] dropped {len(bad_indices)} subjects > {threshold:g}; new N={len(keep_idx)}")
                
            return keep_idx

        if args.no_scrub:
            N_enc = min(len(latents_per_view[vi_][0]) for vi_ in range(V))
            keep_idx = list(range(N_enc))
            print(f"[gauss-fit] no-scrub active: forcing all {len(keep_idx)} subjects.")
        else:
            keep_idx = _scrub_outliers(threshold=args.scrub_threshold)
        N_kept = len(keep_idx)    

        def _select(chunks, keep_idx):
            if not chunks:
                raise RuntimeError(
                    f" Erreur Critique : Aucun vecteur latent n'a pu être accumulé pour cette vue/ce niveau. "
                    f"Veuillez vérifier les messages '[warn] batch encode failed' ci-dessus pour identifier l'erreur d'encodage."
                )
            z = torch.cat(chunks, dim=0)
            return z[keep_idx]

        # ── Compute dims_per_level_per_view and shapes_by_view ──────────
        dims_per_level_per_view = [
            [int(np.prod(z_shapes[l])) for l in range(L)]
            for _ in range(V)
        ]
        shapes_by_view = [
            [list(z_shapes[l]) for l in range(L)]
            for _ in range(V)
        ]

        # ── Build level_view_slices ──────────────────────────────────────
        level_view_slices = []
        for l in range(L):
            off = 0
            row = {}
            for vi_ in range(V):
                d = dims_per_level_per_view[vi_][l]
                row[vi_] = (off, off + d)
                off += d
            level_view_slices.append(row)

        # ── Fit Gaussian ─────────────────────────────────────────────────
        mu_list, Sigma_list, stats_list = [], [], []

        if args.cov_mode == "perlevel":
            for l in range(L):
                blocks = [_select(latents_per_view[vi_][l], keep_idx) for vi_ in range(V)]
                X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
                
                mu = X.mean(axis=0)
                Xc = X - mu
                # cap_quant=0.99 est une valeur classique pour éviter les outliers
                Xc_clean, _ = _sanitize_latents_array(Xc, cap_quantile=99)

                if args.cov_estimator == "lowrank":
                    Sig = _lowrank_from_Xc(Xc_clean, rank=args.rank,
                                           sigma2=args.sigma2, extra_ridge=shrinkage)
                    sts = _lowrank_stats(Sig)
                else:
                    mu, Sig, sts = _fit_gaussian_blocks(
                        blocks, args.cov_estimator, shrinkage, args.cov_lam
                    )
                mu_list.append(mu)
                Sigma_list.append(Sig)
                stats_list.append(sts)
                print(f"  level {l}: cond={sts['cond']:.2e}, "
                      f"λ_min={sts['lambda_min']:.2e}, λ_max={sts['lambda_max']:.2e}")
                # Clean up
                del blocks
                gc.collect()
        else:
            # merged mode
            all_blocks = []
            for vi_ in range(V):
                for l in range(L):
                    all_blocks.append(_select(latents_per_view[vi_][l], keep_idx))
            if args.cov_estimator == "lowrank":
                X = np.concatenate(all_blocks, axis=1)
                mu  = X.mean(axis=0)
                Xc  = X - mu
                Sig = _lowrank_from_Xc(Xc, rank=args.rank,
                                       sigma2=args.sigma2, extra_ridge=shrinkage)
                sts = _lowrank_stats(Sig)
            else:
                mu, Sig, sts = _fit_gaussian_blocks(
                    all_blocks, args.cov_estimator, shrinkage, args.cov_lam
                )
            mu_list   = mu
            Sigma_list= Sig
            stats_list= [sts]
            del all_blocks
            gc.collect()

        # ── Serialise ────────────────────────────────────────────────────
        out_blob: Dict[str, Any] = {
            "mode":       args.cov_mode,
            "estimator":  args.cov_estimator,
            "shrinkage":  shrinkage,
            "cov_lam":    args.cov_lam,
            "jitter":     args.jitter,
            "views":      view_names,
            "N":          N_kept,
            "H":          int(target_size[0]),
            "W":          int(target_size[1]),
            "L":          L,
            "mu":         mu_list,
            "Sigma":      Sigma_list,
            "dims_per_level_per_view": dims_per_level_per_view,
            "shapes_by_view":          shapes_by_view,
            "level_view_slices":       level_view_slices,
            "stats":      stats_list,
            "ckpt_fingerprint": _ckpt_fingerprint(ckpt_path),
        }
        if len(target_size) == 3:
            out_blob["D"] = int(target_size[2])

        gauss_out = Path(args.gauss_out)
        gauss_out.parent.mkdir(parents=True, exist_ok=True)
        if str(gauss_out).endswith(".pt"):
            torch.save(out_blob, str(gauss_out))
        else:
            _save_gauss_npz(out_blob, gauss_out)
        print(f"[gauss-fit] saved → {gauss_out}")

        if args.gauss_summary:
            js = {k: out_blob[k] for k in
                  ("mode", "estimator", "views", "N", "H", "W", "L",
                   "dims_per_level_per_view", "stats", "ckpt_fingerprint")}
            if "D" in out_blob:
                js["D"] = out_blob["D"]
            js["dropped_subjects"] = {
                "count": int(N_original - N_kept),
                "original_N": int(N_original),
                "kept_N": int(N_kept),
            }
            js_path = Path(args.gauss_summary)
            js_path.parent.mkdir(parents=True, exist_ok=True)
            with open(js_path, "w") as f:
                json.dump(js, f, indent=2)
            print(f"[gauss-fit] summary → {js_path}")

    # ------------------------------------------------------------------
    # gauss-impute
    # ------------------------------------------------------------------

    def cmd_gauss_impute(self, argv=None):
        """Condition on observed views → impute target views via Gaussian regression."""
        ap = argparse.ArgumentParser("gauss-impute")
        ap.add_argument("--ckpt",      type=str, required=True)
        ap.add_argument("--gauss",     type=str, required=True)
        ap.add_argument("--manifest",  type=str, required=True)
        ap.add_argument("--views",     type=str, required=True)
        ap.add_argument("--observed",  type=str, required=True)
        ap.add_argument("--target",    type=str, required=True)
        ap.add_argument("--tau",       type=float, default=1.0)
        ap.add_argument("--devices",   type=str, default="cuda:0")
        ap.add_argument("--batch",     type=int, default=1)
        ap.add_argument("--outdir",    type=str, required=True)
        ap.add_argument("--output-format", default="nii.gz",
                        choices=["nii", "nii.gz", "png"])
        ap.add_argument("--pairs-csv", type=str, default=None)
        ap.add_argument("--ema",       action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        gauss_blob = _load_gaussian_model(Path(args.gauss))
        _validate_gauss_blob(gauss_blob)
        gauss_views = gauss_blob["views"]
        L = gauss_blob["L"]

        manifest_path = Path(args.manifest)
        cols = _read_manifest_csv(manifest_path)
        view_names_all = [v.strip() for v in args.views.split(",") if v.strip()]
        obs_names      = [v.strip() for v in args.observed.split(",") if v.strip()]
        tgt_names      = [v.strip() for v in args.target.split(",") if v.strip()]

        view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
        N = len(per_view_paths[0])
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[gauss-impute] {N} subjects, obs={obs_names}, tgt={tgt_names}")

        # ── Encode observed views ────────────────────────────────────────
        obs_latents_per_level: List[Optional[torch.Tensor]] = [None] * L
        for v_name in obs_names:
            vi = view_names.index(v_name)
            model = self.build_model(cfg, device, target_size)
            ok, src = load_weights_into_model(
                model, blob, vi, prefer_ema=args.ema,
                view_name=v_name, cfg_views=cfg_views
            )
            if not ok:
                raise RuntimeError(f"Could not load weights for view '{v_name}'")
            self.prime_if_needed(model, target_size, device)

            for i, img_path in enumerate(tqdm(per_view_paths[vi], desc=f"obs {v_name}")):
                x = self.read_image(img_path, target_size).unsqueeze(0).to(device)
                z_list = _encode_latents(model, x)
                z_flat = _flatten_latents_by_level(z_list)
                for l in range(L):
                    if obs_latents_per_level[l] is None:
                        obs_latents_per_level[l] = z_flat[l].cpu()
                    else:
                        obs_latents_per_level[l] = torch.cat(
                            [obs_latents_per_level[l], z_flat[l].cpu()], dim=0
                        )
                del x, z_list, z_flat
                gc.collect()

            del model
            gc.collect()

        # ── Impute + decode target views ─────────────────────────────────
        for tgt_name in tgt_names:
            tgt_vi_gauss = gauss_views.index(tgt_name)
            vi_model = view_names.index(tgt_name)
            model = self.build_model(cfg, device, target_size)
            ok, _ = load_weights_into_model(
                model, blob, vi_model, prefer_ema=args.ema,
                view_name=tgt_name, cfg_views=cfg_views
            )
            if not ok:
                raise RuntimeError(f"Could not load weights for target view '{tgt_name}'")
            self.prime_if_needed(model, target_size, device)

            for i in tqdm(range(N), desc=f"impute {tgt_name}"):
                z_imputed = []
                for l in range(L):
                    mu_l      = np.asarray(gauss_blob["mu"][l], dtype=np.float64)
                    Sigma_l   = gauss_blob["Sigma"][l]
                    slices_l  = gauss_blob["level_view_slices"][l]

                    # Observed indices
                    obs_idx = np.concatenate([
                        np.arange(*slices_l[gauss_views.index(ov)])
                        for ov in obs_names
                        if gauss_views.index(ov) in {int(k) for k in slices_l}
                    ])
                    tgt_idx = np.arange(*slices_l[tgt_vi_gauss])

                    z_obs_i = obs_latents_per_level[l][i].float().numpy()
                    z_obs_centered = z_obs_i[obs_idx] - mu_l[obs_idx]

                    if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                        U      = np.asarray(Sigma_l["U"], dtype=np.float64)
                        eig    = np.asarray(Sigma_l["eig"], dtype=np.float64)
                        sigma2 = float(Sigma_l.get("sigma2", 0.0))
                        cond_mean = _cond_mean_block_lowrank(
                            U, eig, sigma2,
                            idx_U=obs_idx, idx_T=tgt_idx,
                            mu=mu_l, ZO=z_obs_centered[np.newaxis, :],
                        ).ravel()
                    else:
                        S = np.asarray(Sigma_l, dtype=np.float64)
                        S_OO = S[np.ix_(obs_idx, obs_idx)]
                        S_TO = S[np.ix_(tgt_idx, obs_idx)]
                        try:
                            S_OO_inv = np.linalg.solve(
                                S_OO + 1e-6 * np.eye(len(obs_idx)),
                                np.eye(len(obs_idx))
                            )
                        except np.linalg.LinAlgError:
                            S_OO_inv = np.linalg.pinv(S_OO)
                        cond_mean = S_TO @ S_OO_inv @ z_obs_centered

                    z_tgt = mu_l[tgt_idx] + cond_mean * float(args.tau)
                    z_tgt = np.clip(z_tgt, -20.0, 20.0)

                    shapes_l = gauss_blob["shapes_by_view"][tgt_vi_gauss][l]
                    z_tgt_t  = torch.from_numpy(
                        z_tgt.astype(np.float32)
                    ).view(1, *shapes_l).to(device)
                    z_imputed.append(z_tgt_t)

                x_hat = self.decode_latents(model, z_imputed, target_size)

                subj_stem = Path(per_view_paths[view_names.index(tgt_name)][i]).stem
                ext = f".{args.output_format}"
                out_path = out_dir / f"{subj_stem}_{tgt_name}{ext}"
                self.save_volume(x_hat, out_path)
                del x_hat, z_imputed
                gc.collect()

            del model
            gc.collect()

        print(f"[gauss-impute] done → {out_path}")

    # ------------------------------------------------------------------
    # recon
    # ------------------------------------------------------------------

    def cmd_recon(self, argv=None):
        """Round-trip reconstruction sanity check (x → z → x_hat)."""
        ap = argparse.ArgumentParser("recon")
        ap.add_argument("--ckpt",      type=str, required=True)
        ap.add_argument("--manifest",  type=str, required=True)
        ap.add_argument("--views",     type=str, required=True)
        ap.add_argument("--view-index",type=int, default=0)
        ap.add_argument("--batch",     type=int, default=4)
        ap.add_argument("--devices",   type=str, default="cuda:0")
        ap.add_argument("--out",       type=str, required=True)
        ap.add_argument("--gauss",     type=str, default=None)
        ap.add_argument("--edit-levels",  type=str, default="none")
        ap.add_argument("--edit-what",    default="mean",
                        choices=["mean", "zero", "pc", "pc_denoise"])
        ap.add_argument("--edit-pc-index",  type=int,   default=0)
        ap.add_argument("--edit-pc-scale",  type=float, default=2.0)
        ap.add_argument("--edit-pc-center", default="sample",
                        choices=["sample", "mean"])
        ap.add_argument("--edit-pc-k",   type=int,   default=64)
        ap.add_argument("--edit-pc-beta",type=float, default=0.0)
        ap.add_argument("--ema",         action=argparse.BooleanOptionalAction, default=True)
        ap.add_argument("--reference-image", type=str, default=None)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        manifest_path = Path(args.manifest)
        cols = _read_manifest_csv(manifest_path)
        view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
        vi = int(args.view_index)
        v_name = view_names[vi]

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[recon] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        gauss_blob = None
        levels_to_edit: List[int] = []
        if args.gauss and args.edit_levels != "none":
            gauss_blob = _load_gaussian_model(Path(args.gauss))
            levels_to_edit = [int(l) for l in args.edit_levels.split(",") if l.strip().isdigit()]

        panels = []
        for img_path in tqdm(per_view_paths[vi], desc="recon"):
            x = self.read_image(img_path, target_size).unsqueeze(0).to(device)
            with torch.no_grad():
                z_list = _encode_latents(model, x)
                x_hat  = self.decode_latents(model, z_list, target_size)
                if gauss_blob and levels_to_edit:
                    z_edited = self.edit_latents_to_mean(
                        z_list, gauss_blob, v_name, levels_to_edit,
                        mode=args.edit_what,
                        pc_index=args.edit_pc_index,
                        pc_scale=args.edit_pc_scale,
                        pc_center=args.edit_pc_center,
                        pc_k=args.edit_pc_k,
                        pc_beta=args.edit_pc_beta,
                    )
                    x_hat_e = self.decode_latents(model, z_edited, target_size)
                    diff = (x - x_hat_e).abs()
                    panels.append(
                        torch.cat([
                            self.coerce_nd(x,      target_size),
                            self.coerce_nd(x_hat,  target_size),
                            self.coerce_nd(x_hat_e,target_size),
                            self.coerce_nd(diff,   target_size),
                        ], dim=0)
                    )
                else:
                    diff = (x - x_hat).abs()
                    panels.append(
                        torch.cat([
                            self.coerce_nd(x,     target_size),
                            self.coerce_nd(x_hat, target_size),
                            self.coerce_nd(diff,  target_size),
                        ], dim=0)
                    )
            del x, z_list, x_hat
            gc.collect()

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_panels = torch.cat(panels, dim=0)
        n_cols = 5 if (gauss_blob and levels_to_edit) else 3
        self.save_volume(all_panels, out_path, nrow=n_cols)
        print(f"[recon] saved → {out_path}")

    # ------------------------------------------------------------------
    # recon-template
    # ------------------------------------------------------------------

    def cmd_recon_template(self, argv=None):
        """Decode the Gaussian mean (µ) as a population template."""
        ap = argparse.ArgumentParser("recon-template")
        ap.add_argument("--ckpt",       type=str, required=True)
        ap.add_argument("--gauss",      type=str, required=True)
        ap.add_argument("--views",      type=str, required=True)
        ap.add_argument("--view-index", type=int, default=0)
        ap.add_argument("--devices",    type=str, default="cuda:0")
        ap.add_argument("--out",        type=str, required=True,
                        help="Output file path (png, nii, nii.gz)")
        ap.add_argument("--mc-samples", type=int, default=0)
        ap.add_argument("--mc-temp",    type=float, default=1.0)
        ap.add_argument("--seed",       type=int, default=12345)
        ap.add_argument("--sharpen-image", action="store_true")
        ap.add_argument("--ema",        action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        mc_n   = max(0, int(args.mc_samples))
        if mc_n > 0:
            set_deterministic(int(args.seed))

        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        gauss_blob = _load_gaussian_model(Path(args.gauss))
        views, _, shapes_by_view, L = _validate_gauss_blob(gauss_blob)

        view_names = [v.strip() for v in args.views.split(",") if v.strip()]
        vi = int(args.view_index)
        v_name = view_names[vi]
        v_idx_gauss = views.index(v_name)

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[recon-template] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        mu_list = gauss_blob["mu"]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if mc_n == 0:
            # Decode µ directly
            z_mu = [
                torch.from_numpy(
                    np.asarray(mu_list[l], dtype=np.float32)
                ).view(1, *shapes_by_view[v_idx_gauss][l]).to(device)
                for l in range(L)
            ]
            with torch.no_grad():
                x_mu = self.decode_latents(model, z_mu, target_size)
            x_mu = to01(x_mu, winsorize=True)
            if args.sharpen_image:
                try:
                    import ants as _ants
                    arr = x_mu.squeeze().cpu().numpy()
                    img = _ants.from_numpy(arr)
                    arr2 = _ants.iMath_sharpen(img).numpy()
                    x_mu = torch.from_numpy(arr2).view_as(x_mu)
                except Exception as e:
                    print(f"[warn] sharpen failed: {e}")
            self.save_volume(x_mu, out_path)
        else:
            # Monte Carlo average
            Sigma_list = gauss_blob.get("Sigma", None)
            acc = None
            for s_i in tqdm(range(mc_n), desc="MC samples"):
                z_samp = []
                for l in range(L):
                    mu_l  = np.asarray(mu_list[l], dtype=np.float64)
                    Sig_l = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list
                    shp   = shapes_by_view[v_idx_gauss][l]
                    D_l   = int(np.prod(shp))
                    if isinstance(Sig_l, dict) and Sig_l.get("type") == "lowrank":
                        U      = np.asarray(Sig_l["U"], dtype=np.float64)
                        eig    = np.asarray(Sig_l["eig"], dtype=np.float64)
                        sigma2 = float(Sig_l.get("sigma2", 0.0))
                        xi = np.random.randn(U.shape[1]) * float(args.mc_temp)
                        eps = np.random.randn(D_l) * (sigma2 ** 0.5) * float(args.mc_temp)
                        z_np = mu_l + U @ (np.sqrt(eig) * xi) + eps
                    elif Sig_l is not None:
                        S = np.asarray(Sig_l, dtype=np.float64)
                        noise = np.random.randn(D_l)
                        if S.ndim == 1:
                            z_np = mu_l + noise * (S ** 0.5) * float(args.mc_temp)
                        else:
                            try:
                                L_ch = np.linalg.cholesky(S + 1e-8 * np.eye(D_l))
                                z_np = mu_l + L_ch @ noise * float(args.mc_temp)
                            except np.linalg.LinAlgError:
                                z_np = mu_l
                    else:
                        z_np = mu_l

                    z_t = torch.from_numpy(z_np.astype(np.float32)).view(1, *shp).to(device)
                    z_samp.append(z_t)

                with torch.no_grad():
                    x_samp = self.decode_latents(model, z_samp, target_size)
                if acc is None:
                    acc = x_samp.cpu().float()
                else:
                    acc += x_samp.cpu().float()
                del x_samp, z_samp
                gc.collect()

            x_avg = to01(acc / mc_n, winsorize=True)
            self.save_volume(x_avg, out_path)

        print(f"[recon-template] saved → {out_path}")

    # ------------------------------------------------------------------
    # recon-cohort-template
    # ------------------------------------------------------------------

    def cmd_recon_cohort_template(self, argv=None):
        """Encode cohort → Fréchet mean on the sphere → decode."""
        ap = argparse.ArgumentParser("recon-cohort-template")
        ap.add_argument("--ckpt",        type=str, required=True)
        ap.add_argument("--manifest",    type=str, required=True)
        ap.add_argument("--views",       type=str, required=True)
        ap.add_argument("--view-index",  type=int, default=0)
        ap.add_argument("--devices",     type=str, default="cuda:0")
        ap.add_argument("--out",         type=str, required=True)
        ap.add_argument("--sharpen-image", action="store_true")
        ap.add_argument("--ema",         action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        manifest_path = Path(args.manifest)
        cols = _read_manifest_csv(manifest_path)
        view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
        vi = int(args.view_index)
        v_name = view_names[vi]

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[recon-cohort-template] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        # Encode all subjects
        z_stacks: List[List[torch.Tensor]] = []
        for img_path in tqdm(per_view_paths[vi], desc="encoding"):
            x = self.read_image(img_path, target_size).unsqueeze(0).to(device)
            z_list = _encode_latents(model, x)
            z_stacks.append([z.cpu() for z in z_list])
            del x, z_list
            gc.collect()

        L = len(z_stacks[0])

        # Fréchet mean per level
        def frechet_mean_spherical(z_pts, n_iters=20, tol=1e-5):
            mu = z_pts[0].clone()
            for _ in range(n_iters):
                mu_norm = mu / (mu.norm() + 1e-12)
                tangents = []
                for z in z_pts:
                    z_n = z / (z.norm() + 1e-12)
                    dot = float(torch.clamp((mu_norm * z_n).sum(), -1.0, 1.0))
                    angle = math.acos(dot)
                    if angle < 1e-9:
                        tangents.append(torch.zeros_like(z_n))
                    else:
                        tangents.append(
                            angle / math.sin(angle) * (z_n - dot * mu_norm)
                        )
                mean_tan = torch.stack(tangents).mean(0)
                if mean_tan.norm() < tol:
                    break
                mu = torch.cos(mean_tan.norm()) * mu_norm + \
                     torch.sin(mean_tan.norm()) * mean_tan / (mean_tan.norm() + 1e-12)
                mu = mu * (sum(z.norm() for z in z_pts) / len(z_pts))
            return mu

        z_mean_list = []
        for l in range(L):
            pts = [z_stacks[i][l].flatten() for i in range(len(z_stacks))]
            mu_flat = frechet_mean_spherical(pts)
            shp = z_stacks[0][l].shape
            z_mean_list.append(mu_flat.view(shp).unsqueeze(0).to(device))

        with torch.no_grad():
            x_mean = self.decode_latents(model, z_mean_list, target_size)
        x_mean = to01(x_mean, winsorize=True)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_volume(x_mean, out_path)
        print(f"[recon-cohort-template] saved → {out_path}")

    # ------------------------------------------------------------------
    # recon-temperature
    # ------------------------------------------------------------------

    def cmd_recon_temperature(self, argv=None):
        """Scale latents by temperature τ → decode."""
        ap = argparse.ArgumentParser("recon-temperature")
        ap.add_argument("--ckpt",        type=str, required=True)
        ap.add_argument("--manifest",    type=str, required=True)
        ap.add_argument("--views",       type=str, required=True)
        ap.add_argument("--view-index",  type=int, default=0)
        ap.add_argument("--batch",       type=int, default=1)
        ap.add_argument("--devices",     type=str, default="cuda:0")
        ap.add_argument("--out",         type=str, required=True)
        ap.add_argument("--tau",         type=float, default=0.99)
        ap.add_argument("--tau-level",   action="append", default=None,
                        help="Per-level τ override, format 'level,value'")
        ap.add_argument("--ema",         action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        manifest_path = Path(args.manifest)
        cols = _read_manifest_csv(manifest_path)
        view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
        vi = int(args.view_index)
        v_name = view_names[vi]

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[recon-temperature] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        # Per-level τ overrides
        tau_level: Dict[int, float] = {}
        for spec in (args.tau_level or []):
            try:
                l_str, t_str = spec.split(",")
                tau_level[int(l_str)] = float(t_str)
            except Exception:
                print(f"[warn] ignoring invalid --tau-level spec: {spec!r}")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(per_view_paths[vi], desc="recon-temperature"):
            x = self.read_image(img_path, target_size).unsqueeze(0).to(device)
            stem = Path(img_path).stem
            with torch.no_grad():
                z_list = _encode_latents(model, x)
                z_scaled = []
                for l, z_l in enumerate(z_list):
                    t = tau_level.get(l, args.tau)
                    z_scaled.append(z_l * float(t))
                x_hat = self.decode_latents(model, z_scaled, target_size)

            ext = "".join(out_path.suffixes) or ".nii.gz"
            name = out_path.stem.rstrip("".join(out_path.suffixes)) if out_path.is_dir() \
                   else out_path.parent / f"{stem}_tau{args.tau}{ext}"
            self.save_volume(x_hat, out_path if not out_path.is_dir() else name)
            del x, z_list, z_scaled, x_hat
            gc.collect()

        print(f"[recon-temperature] done")

    # ------------------------------------------------------------------
    # recon-interpolate
    # ------------------------------------------------------------------

    def cmd_recon_interpolate(self, argv=None):
        """
        SLERP/NLERP in latent space between a source and a target.

        Implementation:
          1. Encode source and target images to latents (z_src, z_tgt).
          2. If target is the Gaussian mean (--gauss), z_tgt = µ.
          3. Centre both around µ before interpolation to preserve the manifold.
          4. SLERP between (z_src - µ) and (z_tgt - µ).
          5. Decode µ + interpolated.
        """
        ap = argparse.ArgumentParser("recon-interpolate")
        ap.add_argument("--ckpt",          type=str, required=True)
        ap.add_argument("--gauss",         type=str, required=True)
        ap.add_argument("--views",         type=str, required=True)
        ap.add_argument("--view-index",    type=int, default=0)
        ap.add_argument("--batch",         type=int, default=1)
        ap.add_argument("--devices",       type=str, default="cuda:0")
        ap.add_argument("--out",           type=str, required=True)
        ap.add_argument("--manifest",      type=str, default=None)
        ap.add_argument("--source-image",  type=str, default=None)
        ap.add_argument("--target-image",  type=str, default=None)
        ap.add_argument("--t",             type=float, default=0.5,
                        help="Interpolation t ∈ [0,1]: 0=source, 1=target/mean")
        ap.add_argument("--interp-level",  action="append", default=None,
                        help="Per-level t override, format 'level,t'")
        ap.add_argument("--interp-type",   default="slerp",
                        choices=["slerp", "nlerp"])
        ap.add_argument("--ema",           action=argparse.BooleanOptionalAction,
                        default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        gauss_blob = _load_gaussian_model(Path(args.gauss))
        gauss_views, _, shapes_by_view, L = _validate_gauss_blob(gauss_blob)
        mu_list = gauss_blob["mu"]

        view_names = [v.strip() for v in args.views.split(",") if v.strip()]
        vi = int(args.view_index)
        v_name = view_names[vi]
        v_idx_gauss = gauss_views.index(v_name)

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[recon-interpolate] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        # Per-level t overrides
        t_level: Dict[int, float] = {}
        for spec in (args.interp_level or []):
            try:
                l_str, t_str = spec.split(",")
                t_level[int(l_str)] = float(t_str)
            except Exception:
                print(f"[warn] ignoring invalid --interp-level spec: {spec!r}")

        # SLERP kernel (with µ-centering)
        def slerp_flat(t: float, v0: np.ndarray, v1: np.ndarray,
                       DOT_THRESHOLD: float = 0.9995) -> np.ndarray:
            """SLERP between v0 and v1 (mean-centred)."""
            v0n = v0 / (np.linalg.norm(v0) + 1e-12)
            v1n = v1 / (np.linalg.norm(v1) + 1e-12)
            dot = float(np.clip((v0n * v1n).sum(), -1.0, 1.0))
            if abs(dot) > DOT_THRESHOLD:
                return v0 + t * (v1 - v0)  # near-parallel: lerp
            theta_0 = math.acos(dot)
            sin_0   = math.sin(theta_0)
            return (math.sin((1.0 - t) * theta_0) / sin_0) * v0 + \
                   (math.sin(t * theta_0) / sin_0) * v1

        def nlerp_flat(t: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
            lerp = v0 + t * (v1 - v0)
            return lerp / (np.linalg.norm(lerp) + 1e-12) * \
                   ((1.0 - t) * np.linalg.norm(v0) + t * np.linalg.norm(v1))

        interp_fn = slerp_flat if args.interp_type == "slerp" else nlerp_flat

        # Resolve source images
        if args.manifest:
            manifest_path = Path(args.manifest)
            cols = _read_manifest_csv(manifest_path)
            _, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
            source_paths = per_view_paths[vi]
        elif args.source_image:
            source_paths = [Path(args.source_image)]
        else:
            raise RuntimeError("Either --manifest or --source-image is required.")

        # Resolve target
        target_is_gauss = (args.target_image is None)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for src_path in tqdm(source_paths, desc="interpolate"):
            x_src = self.read_image(src_path, target_size).unsqueeze(0).to(device)
            with torch.no_grad():
                z_src_list = _encode_latents(model, x_src)

            if target_is_gauss:
                # Target = Gaussian mean per level
                z_tgt_list = [
                    torch.from_numpy(
                        np.asarray(mu_list[l], dtype=np.float32)
                    ).view(1, *shapes_by_view[v_idx_gauss][l]).to(device)
                    for l in range(L)
                ]
            else:
                x_tgt = self.read_image(
                    Path(args.target_image), target_size
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    z_tgt_list = _encode_latents(model, x_tgt)
                del x_tgt

            # Interpolate with µ-centring at each level
            z_interp_list = []
            for l in range(len(z_src_list)):
                t_l = t_level.get(l, args.t)

                # µ for this view at this level
                mu_l_np = np.asarray(mu_list[l], dtype=np.float64)
                shp = shapes_by_view[v_idx_gauss][l]
                mu_flat = mu_l_np.ravel()  # (D_l,)

                z_s_flat = z_src_list[l].cpu().float().numpy().ravel()
                z_t_flat = z_tgt_list[l].cpu().float().numpy().ravel()

                # Centre around µ before rotation
                z_s_c = z_s_flat - mu_flat
                z_t_c = z_t_flat - mu_flat

                z_interp_c = interp_fn(t_l, z_s_c, z_t_c)

                # Restore µ offset
                z_interp_flat = z_interp_c + mu_flat
                z_interp_t = torch.from_numpy(
                    z_interp_flat.astype(np.float32)
                ).view(1, *shp).to(device)
                z_interp_list.append(z_interp_t)

            with torch.no_grad():
                x_interp = self.decode_latents(model, z_interp_list, target_size)

            self.save_volume(x_interp, out_path)
            del x_src, z_src_list, z_tgt_list, z_interp_list, x_interp
            gc.collect()

        print(f"[recon-interpolate] done → {out_path}")

    # ------------------------------------------------------------------
    # calc-distance
    # ------------------------------------------------------------------

    def cmd_calc_distance(self, argv=None):
        """Compute per-subject distances (Euclidean/Mahalanobis/geodesic) to Gaussian mean."""
        ap = argparse.ArgumentParser("calc-distance")
        ap.add_argument("--ckpt",            type=str, required=True)
        ap.add_argument("--gauss",           type=str, required=True)
        ap.add_argument("--views",           type=str, required=True)
        ap.add_argument("--view-index",      type=int, default=0)
        ap.add_argument("--batch",           type=int, default=4)
        ap.add_argument("--workers",         type=int, default=4)
        ap.add_argument("--devices",         type=str, default="cuda:0")
        ap.add_argument("--out",             type=str, required=True)
        ap.add_argument("--save-levels",     action=argparse.BooleanOptionalAction, default=True)
        ap.add_argument("--distance-metric", default="geodesic",
                        choices=["euclidean", "mahalanobis", "geodesic"])
        ap.add_argument("--variance-epsilon",type=float, default=1e-6)
        ap.add_argument("--manifest",        type=str, default=None)
        ap.add_argument("--source-image",    type=str, default=None)
        ap.add_argument("--target-image",    type=str, default=None)
        ap.add_argument("--pairwise",        action=argparse.BooleanOptionalAction, default=False)
        ap.add_argument("--ema",             action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        gauss_blob = _load_gaussian_model(Path(args.gauss))
        gauss_views, _, shapes_by_view, L = _validate_gauss_blob(gauss_blob)
        mu_list    = gauss_blob["mu"]
        Sigma_list = gauss_blob.get("Sigma", None)

        view_names = [v.strip() for v in args.views.split(",") if v.strip()]
        vi = int(args.view_index)
        v_name = view_names[vi]

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, vi, prefer_ema=args.ema,
            view_name=v_name, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view '{v_name}'")
        print(f"[calc-distance] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        if args.manifest:
            manifest_path = Path(args.manifest)
            cols = _read_manifest_csv(manifest_path)
            _, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
            source_paths = per_view_paths[vi]
        elif args.source_image:
            source_paths = [Path(args.source_image)]
        else:
            raise RuntimeError("--manifest or --source-image required.")

        def _dist_at_level(z_flat: np.ndarray, mu_l: np.ndarray,
                           Sigma_l, l_idx: int) -> float:
            d = z_flat - mu_l
            metric = args.distance_metric
            if metric == "euclidean":
                return float(np.linalg.norm(d))
            elif metric == "geodesic":
                z_n   = z_flat / (np.linalg.norm(z_flat) + 1e-12)
                mu_n  = mu_l   / (np.linalg.norm(mu_l)   + 1e-12)
                cos   = float(np.clip((z_n * mu_n).sum(), -1.0, 1.0))
                return float(math.acos(cos))
            else:  # mahalanobis
                if Sigma_l is None:
                    return float(np.linalg.norm(d))
                if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                    U      = np.asarray(Sigma_l["U"], dtype=np.float64)
                    eig    = np.asarray(Sigma_l["eig"], dtype=np.float64)
                    sigma2 = float(Sigma_l.get("sigma2", args.variance_epsilon))
                    # Mahalanobis via push-through
                    Ud = U.T @ d
                    mahal2 = float((Ud * Ud / (eig + sigma2)).sum()
                                   + np.dot(d, d) / sigma2
                                   - float((Ud * Ud).sum()) / sigma2)
                    return float(abs(mahal2) ** 0.5)
                S = np.asarray(Sigma_l, dtype=np.float64)
                D = len(mu_l)
                if S.ndim == 1:
                    var = S + args.variance_epsilon
                    return float(np.sqrt((d * d / var).sum()))
                try:
                    S_inv = np.linalg.solve(S + args.variance_epsilon * np.eye(D), np.eye(D))
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)
                return float(math.sqrt(max(0.0, float(d @ S_inv @ d))))

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        v_idx_gauss = gauss_views.index(v_name) if v_name in gauss_views else vi

        for img_path in tqdm(source_paths, desc="calc-distance"):
            x = self.read_image(img_path, target_size).unsqueeze(0).to(device)
            with torch.no_grad():
                z_list = _encode_latents(model, x)
            z_flat_list = _flatten_latents_by_level(z_list)

            row = {"path": str(img_path)}
            total = 0.0
            for l in range(L):
                mu_l = np.asarray(mu_list[l], dtype=np.float64)
                
                level_slices = gauss_blob.get("level_view_slices", None)[l]
                
                if "0" in level_slices:
                    view_key = "0"
                elif str(v_idx_gauss) in level_slices:
                    view_key = str(v_idx_gauss)
                else:
                    view_key = list(level_slices.keys())[0]

                a, b = level_slices[view_key]
                mu_l_v = mu_l[a:b]
                
                Sig_l  = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list
                z_np   = z_flat_list[l][0].cpu().float().numpy().ravel()
                dist_l = _dist_at_level(z_np, mu_l_v, Sig_l, l)
                if args.save_levels:
                    row[f"dist_level_{l}"] = dist_l
                total += dist_l

            row["dist_total"] = total
            rows.append(row)
            del x, z_list, z_flat_list
            gc.collect()

        with open(out_path, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        print(f"[calc-distance] {len(rows)} subjects → {out_path}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _probe_latent_shapes(self, model, target_size, device) -> List[Tuple]:
        """
        Run one dummy forward pass to discover latent shapes per flow level.
        Returns a list of shapes (without batch dim), e.g. [(C0, H0, W0), ...].
        """
        x_dummy = torch.zeros(1, 1, *target_size, device=device, dtype=torch.float32)
        try:
            if hasattr(model, "inverse_and_log_det"):
                z, _ = model.inverse_and_log_det(x_dummy)
            elif hasattr(model, "inverse"):
                z, _ = model.inverse(x_dummy)
            else:
                raise RuntimeError("Model lacks inverse mapping.")
        except Exception as e:
            print(f"[warn] _probe_latent_shapes failed: {e}. Returning single level.")
            return [(1, *target_size)]
        z_list = z if isinstance(z, (list, tuple)) else [z]
        return [tuple(z_l.shape[1:]) for z_l in z_list]

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def cmd_sample(self, argv=None):
        """Generate random samples from the flow and save to files."""
        ap = argparse.ArgumentParser("sample")
        ap.add_argument("--ckpt",        type=str, required=True)
        ap.add_argument("--view-index",  type=int, default=0)
        ap.add_argument("--n-samples",   type=int, default=4)
        ap.add_argument("--temperature", type=float, default=1.0)
        ap.add_argument("--seed",        type=int, default=12345)
        ap.add_argument("--devices",     type=str, default="cuda:0")
        ap.add_argument("--out",         type=str, required=True,
                        help="Output path: directory for NIfTI, or file for PNG grid.")
        ap.add_argument("--grid-size",   type=parse_mn, default=None,
                        help="MxN grid layout for PNG output.")
        ap.add_argument("--ema",         action=argparse.BooleanOptionalAction, default=True)
        self._add_size_arg(ap, required=True)
        args = ap.parse_args(argv)

        set_deterministic(int(args.seed))
        device = torch.device(args.devices)
        ckpt_path = resolve_ckpt_path(Path(args.ckpt))
        blob      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg       = blob.get("config", blob.get("cfg", {}))
        cfg_views = cfg.get("views", None)
        target_size = self._get_target_size(args, cfg)

        model = self.build_model(cfg, device, target_size)
        ok, src = load_weights_into_model(
            model, blob, int(args.view_index), prefer_ema=args.ema,
            cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Could not load weights for view {args.view_index}")
        print(f"[sample] loaded from {src}")
        self.prime_if_needed(model, target_size, device)

        print(f"[sample] generating {args.n_samples} samples @ temp={args.temperature}")
        with torch.no_grad():
            x = sample_with_temperature(model, int(args.n_samples), float(args.temperature))
        if isinstance(x, (list, tuple)):
            x = x[0]

        out_path = Path(args.out)

        if out_path.suffix in (".png", ".jpg", ".jpeg") or args.grid_size is not None:
            # Save PNG grid
            M, N = args.grid_size if args.grid_size is not None else (1, int(args.n_samples))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_volume(x, out_path, nrow=int(N))
            print(f"[sample] grid saved → {out_path}")
        else:
            # Save individual volumes
            out_path.mkdir(parents=True, exist_ok=True)
            for i in range(x.shape[0]):
                xi = x[i:i+1]
                self.save_volume(xi, out_path / f"sample_{i:04d}.nii.gz")
            print(f"[sample] {x.shape[0]} volumes saved → {out_path}")
