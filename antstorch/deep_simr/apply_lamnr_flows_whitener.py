# apply_lamnr_flows_whitener.py
# Apply trained normalizing-flow whitener models to new data.
# Updated to support dataset-driven normalization (z-score or min–max), to
# remove the hard dependency on legacy "standardizers", and to coerce any
# list-based stats (e.g., JSON) into NumPy arrays before math.

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


# -------------------------------
# Utilities
# -------------------------------

def _ensure_list(x):
    return x if isinstance(x, list) else [x]


def _to_df(arr: np.ndarray, like_df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    cols = [f"{col_prefix}{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=like_df.index, columns=cols)


def _coerce_stats_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    """Convert list-based stats (from JSON or trainer .tolist()) to numpy arrays."""
    if d is None:
        return None
    mode = d.get("mode", None)
    if mode is None:
        return {"mode": None}
    if mode == "0mean":
        mean = np.asarray(d.get("mean", None), dtype=np.float64)
        std  = np.asarray(d.get("std",  None), dtype=np.float64)
        if mean is None or std is None:
            raise ValueError("Normalization stats for mode '0mean' require 'mean' and 'std'.")
        return {"mode": "0mean", "mean": mean, "std": std}
    if mode == "01":
        vmin = np.asarray(d.get("vmin", None), dtype=np.float64)
        vrng = np.asarray(d.get("vrng", None), dtype=np.float64)
        if vmin is None or vrng is None:
            raise ValueError("Normalization stats for mode '01' require 'vmin' and 'vrng'.")
        # avoid zeros
        vrng = np.maximum(vrng, 1e-8)
        return {"mode": "01", "vmin": vmin, "vrng": vrng}
    # passthrough for any custom mode
    return d


# ---------- Normalization helpers (dataset-style) ----------

def _fit_norm_stats(df: pd.DataFrame, mode: Optional[str]) -> Dict[str, np.ndarray]:
    """Fit normalization stats on a numeric-only view, mimicking the dataset logic."""
    if mode is None:
        return {"mode": None}
    x = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
    if x.size == 0:
        return {"mode": None}
    if mode == "0mean":
        mean = np.nanmean(x, axis=0)
        std  = np.nanstd(x, axis=0)
        std  = np.maximum(std, 1e-8)
        return {"mode": "0mean", "mean": mean, "std": std}
    elif mode == "01":
        vmin = np.nanmin(x, axis=0)
        vmax = np.nanmax(x, axis=0)
        rng  = np.maximum(vmax - vmin, 1e-8)
        return {"mode": "01", "vmin": vmin, "vrng": rng}
    else:
        raise ValueError(f"Unsupported normalization mode '{mode}'.")


def _apply_norm(x_np: np.ndarray, stats: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Apply normalization to numeric features stored contiguously at the front."""
    if stats is None:
        return x_np
    stats = _coerce_stats_dict(stats)
    mode = stats.get("mode", None)
    if mode is None:
        return x_np
    x = x_np.copy()
    if mode == "0mean":
        mean = stats["mean"]
        std  = stats["std"]
        std_safe = np.where((np.isfinite(std)) & (std > 0), std, 1.0)
        mean_safe = np.where(np.isfinite(mean), mean, 0.0)
        z = (x - mean_safe) / std_safe
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    elif mode == "01":
        vmin = stats["vmin"]
        vrng = stats["vrng"]
        z = (x - vmin) / vrng
        z = np.clip(z, 0.0, 1.0)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        return x


def _invert_norm(z_np: np.ndarray, stats: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if stats is None:
        return z_np
    stats = _coerce_stats_dict(stats)
    mode = stats.get("mode", None)
    if mode is None:
        return z_np
    if mode == "0mean":
        mean = stats["mean"]
        std  = stats["std"]
        std_safe = np.where((np.isfinite(std)) & (std > 0), std, 1.0)
        mean_safe = np.where(np.isfinite(mean), mean, 0.0)
        return z_np * std_safe + mean_safe
    elif mode == "01":
        vmin = stats["vmin"]
        vrng = stats["vrng"]
        return z_np * vrng + vmin
    else:
        return z_np


# ---------- Whitening helpers (latent-space transforms) ----------

def _extract_whitened_from_z(model, z: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc") and hasattr(q0, "log_sigma"):
        zc = (z - q0.loc).to(torch.float64)       # (N, D)
        W  = q0.W.to(torch.float64)               # (L, D)
        sigma2 = torch.exp(2.0 * q0.log_sigma.to(torch.float64))  # scalar
        A = W @ W.T + sigma2 * torch.eye(W.shape[0], dtype=W.dtype, device=W.device)  # (L, L)
        A_inv = torch.linalg.inv(A)
        return zc @ W.T @ A_inv                    # (N, L)
    # DiagGaussian fallback
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if (loc is not None) and (scale is not None):
        return ((z - loc) / (scale + 1e-12)).to(torch.float64)
    return z.to(torch.float64)


def _z_from_whitened(model, eps: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        return eps.to(torch.float64) @ q0.W.to(torch.float64) + q0.loc.to(torch.float64)
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if (loc is not None) and (scale is not None):
        return eps * scale + loc
    raise ValueError("Expected GaussianPCA with W/loc or DiagGaussian with loc/scale.")


def _extract_whitened_from_z_full(model, z: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    zc = (z - q0.loc).to(torch.float64)                 # (N, D)
    W  = q0.W.to(torch.float64)                         # (L, D)
    sigma2 = torch.exp(2.0 * q0.log_sigma.to(torch.float64))
    Sigma = W.T @ W + sigma2 * torch.eye(W.shape[1], dtype=W.dtype, device=W.device)  # (D, D)
    L = torch.linalg.cholesky(Sigma)                    # lower
    yT = torch.linalg.solve(L.T, zc.T)                  # (D, N)
    return yT.T                                         # (N, D)


def _z_from_whitened_full(model, eps_full: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    W  = q0.W.to(torch.float64)
    sigma2 = torch.exp(2.0 * q0.log_sigma.to(torch.float64))
    Sigma = W.T @ W + sigma2 * torch.eye(W.shape[1], dtype=W.dtype, device=W.device)
    L = torch.linalg.cholesky(Sigma)
    return eps_full.to(torch.float64) @ L.T + q0.loc.to(torch.float64)


# -------------------------------
# Main apply function
# -------------------------------

def apply_lamnr_flows_whitener(
    trainer_output: Dict[str, Any] | List[torch.nn.Module],
    data: pd.DataFrame | List[pd.DataFrame],
    direction: str = "forward",              # {"forward", "inverse"}
    output_space: str = "z",                 # {"z", "whitened", "whitened_full"} for direction="forward"
    input_space: str = "z",                  # {"z", "whitened", "whitened_full"} for direction="inverse"
    batch_size: int = 4096,
    device: str = "cpu",
    # --- New normalization API ---
    normalization_mode: Union[None, str, List[Optional[str]]] = "0mean",
    normalization_stats: Optional[Union[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]] = None,
    fit_stats_on_data_if_missing: bool = False,
) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Apply trained normalizing-flow whitener model(s) to new data with support
    for dataset-style normalization.
    """
    # Unpack models and try to pull any embedded normalizers from newer trainer outputs
    if isinstance(trainer_output, dict):
        models = trainer_output.get("models", None)
        if models is None:
            raise ValueError("trainer_output dict does not contain 'models'.")
        if normalization_stats is None:
            # Prefer new key "dataset_normalizers"; fall back to old "standardizers" (z-score only)
            normalization_stats = trainer_output.get("dataset_normalizers", None)
            legacy = trainer_output.get("standardizers", None)
            if normalization_stats is None and legacy is not None:
                # Convert legacy mean/std to new schema
                norm_list = []
                for sd in legacy:
                    norm_list.append({"mode": "0mean",
                                      "mean": np.asarray(sd["mean"], dtype=np.float64),
                                      "std":  np.asarray(sd["std"],  dtype=np.float64)})
                normalization_stats = norm_list
    else:
        models = trainer_output

    models = _ensure_list(models)
    data_list = _ensure_list(data)
    if len(models) != len(data_list):
        raise ValueError(f"Number of models ({len(models)}) must match number of views ({len(data_list)}).")

    # Normalize normalization_mode to per-view list
    if isinstance(normalization_mode, list):
        modes = normalization_mode
    else:
        modes = [normalization_mode] * len(models)

    # Normalize normalization_stats to list aligned with views; coerce arrays
    stats_list: List[Optional[Dict[str, np.ndarray]]] = [None] * len(models)
    if normalization_stats is not None:
        if isinstance(normalization_stats, dict) and all(isinstance(k, int) for k in normalization_stats.keys()):
            for i in range(len(models)):
                stats_list[i] = _coerce_stats_dict(normalization_stats.get(i, None))
        else:
            stats_list = [ _coerce_stats_dict(s) for s in list(normalization_stats) ]

    # Prepare models
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    for m in models:
        m.to(device_t).double()
        m.eval()

    outputs: List[pd.DataFrame] = []
    for v_idx, (df, model, mode) in enumerate(zip(data_list, models, modes)):
        X_np = df.to_numpy().astype(np.float64, copy=False)

        # Ensure stats present if a mode is requested
        stats = stats_list[v_idx]
        if mode is not None and stats is None:
            if fit_stats_on_data_if_missing:
                stats = _fit_norm_stats(df, mode)
            else:
                raise ValueError(
                    f"No normalization_stats provided for view {v_idx} "
                    f"while normalization_mode='{mode}'. Either supply stats "
                    f"(trainer_output['dataset_normalizers'] or argument "
                    f"'normalization_stats'), or set fit_stats_on_data_if_missing=True, "
                    f"or disable normalization by passing normalization_mode=None."
                )

        # Forward: normalize -> flow.forward -> optional whiten
        if direction == "forward":
            X_proc = _apply_norm(X_np, stats) if stats is not None else X_np
            Z_out = []
            with torch.inference_mode():
                for start in range(0, X_proc.shape[0], batch_size):
                    stop = min(start + batch_size, X_proc.shape[0])
                    xb = torch.from_numpy(X_proc[start:stop]).to(device_t).double()

                    z = model.forward(xb)
                    if isinstance(z, (tuple, list)):
                        z = z[0]

                    if output_space == "whitened":
                        z = _extract_whitened_from_z(model, z)
                    elif output_space == "whitened_full":
                        z = _extract_whitened_from_z_full(model, z)
                    elif output_space == "z":
                        z = z.to(torch.float64)
                    else:
                        raise ValueError("output_space must be one of {'z','whitened','whitened_full'}.")

                    Z_out.append(z.detach().cpu().numpy())

            Y = np.vstack(Z_out)

            if output_space == "whitened":
                col_prefix = "whitened_"
            elif output_space == "whitened_full":
                col_prefix = "whitened_full_"
            else:
                col_prefix = "z_"
            outputs.append(_to_df(Y, df, col_prefix))

        # Inverse: map provided latent -> x (then de-normalize if needed)
        elif direction == "inverse":
            Z_out = []
            with torch.inference_mode():
                for start in range(0, X_np.shape[0], batch_size):
                    stop = min(start + batch_size, X_np.shape[0])
                    xb = torch.from_numpy(X_np[start:stop]).to(device_t).double()

                    q0 = model.q0
                    if input_space == "whitened":
                        if hasattr(q0, "W"):
                            L, D = q0.W.shape
                            if xb.shape[1] != L:
                                raise ValueError(f"input_space='whitened' expects dim={L}, got {xb.shape[1]}.")
                        z = _z_from_whitened(model, xb)
                    elif input_space == "whitened_full":
                        if hasattr(q0, "W"):
                            L, D = q0.W.shape
                            if xb.shape[1] != D:
                                raise ValueError(f"input_space='whitened_full' expects dim={D}, got {xb.shape[1]}.")
                        z = _z_from_whitened_full(model, xb)
                    elif input_space == "z":
                        expected_D = None
                        if hasattr(q0, "loc") and q0.loc is not None:
                            expected_D = int(q0.loc.shape[-1])
                        elif hasattr(q0, "W"):
                            expected_D = int(q0.W.shape[1])  # (L, D)
                        if expected_D is not None and xb.shape[1] != expected_D:
                            raise ValueError(f"input_space='z' expects dim={expected_D}, got {xb.shape[1]}.")
                        z = xb
                    else:
                        raise ValueError("input_space must be one of {'z','whitened','whitened_full'}.")

                    xrec = model.inverse(z)
                    if isinstance(xrec, (tuple, list)):
                        xrec = xrec[0]
                    Z_out.append(xrec.detach().cpu().numpy())

            Y = np.vstack(Z_out)
            # De-normalize back to raw space if stats were applied on inputs during training
            Y = _invert_norm(Y, stats) if stats is not None else Y
            outputs.append(_to_df(Y, df, "x_reconstructed_"))

        else:
            raise ValueError("direction must be 'forward' or 'inverse'")

    return outputs[0] if isinstance(data, pd.DataFrame) else outputs
