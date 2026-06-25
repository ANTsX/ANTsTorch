#!/usr/bin/env python3
"""
lamnr_glow_tool_3d_new.py — LAM-Flow (Glow 3D) Inference & Analysis Toolkit

Thin shim over lamnr_glow_tool_base.GlowToolBase.
All shared logic (gauss-fit, gauss-impute, recon-template, recon-interpolate,
calc-distance, sample, etc.) lives in the base class. This file implements 
only the 3D-specific I/O hooks:
  - NIfTI volumetric extraction via ANTs
  - 5D (B, C, H, W, D) tensor coercion
  - build_model (3D variant)
  - prime_if_needed (3D variant)
  - save_single / save_volume (NIfTI export)

v0.5.5-refactored
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from antstorch.lamnr_flows.architectures import create_glow_normalizing_flow_model_3d
except ImportError:
    print("[warn] 'antstorch' not found. Ensure it is installed for 3D Glow models.")
    create_glow_normalizing_flow_model_3d = None

# Import the shared base class
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import GlowToolBase, _validate_gauss_blob

# ─────────────────────────────────────────────────────────────────────────────
# 3D Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _read_image_3d(pth: Path, target_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[torch.Tensor, tuple]:
    img = ants.image_read(str(pth))
    native_spacing = img.spacing

    if target_shape is not None:
        img = ants.resample_image(
            img, 
            resample_params=tuple(target_shape), 
            use_voxels=True, 
            interp_type=0
        )

    arr = img.numpy()
    t = torch.from_numpy(arr).to(torch.float32)

    # ─── FORCE EXACTLY 4D: (1, H, W, D) ───
    t = t.squeeze() # Remove all arbitrary dimensions, making it (H, W, D)
    t = t.unsqueeze(0) # Add single channel dimension -> (1, H, W, D)

    return t, native_spacing

def _save_nifti(tensor: torch.Tensor, out_path: Path, spacing: Optional[tuple] = None):
    """
    Save a (1, 1, H, W, D) or (B, C, H, W, D) tensor to NIfTI.
    """
    arr = tensor.detach().cpu().numpy()
    
    # Squeeze out Batch and Channel dimensions if they are 1
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
        
    img = ants.from_numpy(arr)
    if spacing is not None:
        try:
            img.set_spacing(spacing)
        except Exception as e:
            print(f"[warn] Could not set spacing: {e}")
            
    ants.image_write(img, str(out_path))

def _edit_latents_to_mean_for_view_3d(
    z_list: List[torch.Tensor],
    gauss_blob: Dict,
    view_name: str,
    levels_to_edit: List[int],
    mode: str = "mean",
    pc_index: int = 0,
    pc_scale: float = 2.0,
    pc_center: str = "sample",
    pc_k: int = 64,
    pc_beta: float = 0.0,
) -> List[torch.Tensor]:
    """
    Édition des vecteurs latents pour le modèle 3D (Tenseurs 5D : B, C, H, W, D).
    Adaptation directe de la logique 2D pour les volumes NIfTI.
    """
    if not levels_to_edit:
        return z_list

    views, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)
    try:
        v_idx = views.index(view_name)
    except ValueError:
        raise RuntimeError(f"[recon] View '{view_name}' not in Gaussian header {views}.")

    mu_list    = gauss_blob["mu"]
    Sigma_list = gauss_blob.get("Sigma", None)

    raw_slices = gauss_blob.get("level_view_slices", None)
    V = len(views)
    level_view_slices = []
    if raw_slices is not None:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict):
                level_view_slices.append({int(k): tuple(v) for k, v in row.items()})
            else:
                level_view_slices.append({vi: tuple(row[vi]) for vi in range(V)})
    else:
        for l in range(L):
            off, row_int = 0, {}
            for vi in range(V):
                d = int(np.asarray(dims_tbl[vi][l]).item()
                        if hasattr(dims_tbl[vi][l], "item") else dims_tbl[vi][l])
                row_int[vi] = (off, off + d)
                off += d
            level_view_slices.append(row_int)

    levels_set = {int(l) for l in levels_to_edit}
    z_out: List[torch.Tensor] = []

    for l, z_l in enumerate(z_list):
        if l not in levels_set:
            z_out.append(z_l)
            continue

        if z_l.ndim != 5:
            raise RuntimeError(f"[recon] Expected 5D latent at level {l}, got {z_l.shape}.")

        B, C, H, W, D = z_l.shape
        Cg, Hg, Wg, Dg = shapes_by_view[v_idx][l]
        if (C, H, W, D) != (Cg, Hg, Wg, Dg):
            raise RuntimeError(
                f"[recon] Shape mismatch level {l}, view '{view_name}': "
                f"model ({C},{H},{W},{D}) vs Gauss ({Cg},{Hg},{Wg},{Dg})."
            )

        a, b = level_view_slices[l][v_idx]
        mu_level   = np.asarray(mu_list[l], dtype=np.float64).ravel()
        mu_view_flat = mu_level[a:b]
        
        # Formatage du vecteur moyen en 5D
        mu_view = torch.as_tensor(
            mu_view_flat, dtype=z_l.dtype, device=z_l.device
        ).view(1, C, H, W, D)

        if mode == "mean":
            z_l_edit = mu_view.expand(B, C, H, W, D)

        elif mode == "zero":
            z_l_edit = torch.zeros_like(z_l)

        elif mode in ("pc", "pc_denoise"):
            Sigma_l = (Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list)
            Dv = C * H * W * D
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U      = np.asarray(Sigma_l["U"], dtype=np.float64)
                eig    = np.asarray(Sigma_l["eig"], dtype=np.float64)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))
                U_v    = U[a:b, :]
                Sv     = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv += sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                Sv = np.diag(S[a:b]) if S.ndim == 1 else S[a:b, a:b]

            Sv = 0.5 * (Sv + Sv.T)
            w_eig, V_mat = np.linalg.eigh(Sv)

            if mode == "pc":
                k   = int(pc_index)
                col = -1 - k
                direction_np = V_mat[:, col]
                lam  = float(max(w_eig[col], 0.0))
                step = float(pc_scale) * (lam ** 0.5 if lam > 0.0 else 0.0)
                direction_t = torch.from_numpy(
                    direction_np.astype(np.float32)
                ).view(1, C, H, W, D).to(z_l.device, z_l.dtype)
                base = (mu_view.expand(B, C, H, W, D) if pc_center.lower() == "mean" else z_l)
                z_l_edit = base + step * direction_t
                print(f"[recon] level {l}, '{view_name}': PC{pc_index} "
                      f"λ={lam:.3e}, step={step:.3e}, center={pc_center}")
            else:  # pc_denoise
                V_desc  = V_mat[:, ::-1]
                k_keep  = min(max(int(pc_k), 0), V_desc.shape[1])
                V_t     = torch.from_numpy(V_desc.astype(np.float32)).to(z_l.device, z_l.dtype)
                z_flat  = z_l.view(B, -1)
                mu_flat = mu_view.view(1, -1)
                y = torch.matmul(z_flat - mu_flat, V_t)
                if k_keep < V_t.shape[1]:
                    tail = y[:, k_keep:]
                    y[:, k_keep:] = 0.0 if float(pc_beta) == 0.0 else float(pc_beta) * tail
                z_flat_edit = mu_flat + torch.matmul(y, V_t.T)
                z_l_edit    = z_flat_edit.view(B, C, H, W, D)
                print(f"[recon] level {l}, '{view_name}': "
                      f"pc_denoise k_keep={k_keep}, β={pc_beta:.3f}")
        else:
            raise ValueError(f"[recon] Unknown edit mode '{mode}'.")

        z_out.append(z_l_edit)
    return z_out

# ─────────────────────────────────────────────────────────────────────────────
# 3D Tool Class
# ─────────────────────────────────────────────────────────────────────────────

class GlowTool3D(GlowToolBase):
    """3D implementation of the LAM-Flow toolkit."""
    
    def _add_spatial_args(self, parser: argparse.ArgumentParser):
        """Add 3D-specific command line arguments."""
        parser.add_argument("--spatial-dims", type=int, nargs=3, help="H W D for 3D volume")
        parser.add_argument("--H", type=int, help="Height")
        parser.add_argument("--W", type=int, help="Width")
        parser.add_argument("--D", type=int, help="Depth")
        
    def _get_target_size(self, args: argparse.Namespace, cfg: dict) -> Tuple[int, int, int]:
        """Extract the (H, W, D) target size from arguments or config."""
        if getattr(args, "spatial_dims", None) is not None and len(args.spatial_dims) == 3:
            return tuple(args.spatial_dims)
        if getattr(args, "H", None) is not None and getattr(args, "W", None) is not None and getattr(args, "D", None) is not None:
            return (args.H, args.W, args.D)
        
        # Fallback to model config
        if "target_shape" in cfg and len(cfg["target_shape"]) == 3:
            return tuple(cfg["target_shape"])
        if "H" in cfg and "W" in cfg and "D" in cfg:
            return (cfg["H"], cfg["W"], cfg["D"])
            
        raise ValueError("Could not determine 3D spatial dimensions (H, W, D). Please specify in args or config.")
        
    def build_model(self, cfg: dict, device: torch.device, target_size: Tuple[int, int, int]) -> nn.Module:
        """Instantiate the 3D Glow model from antstorch."""
        if create_glow_normalizing_flow_model_3d is None:
            raise RuntimeError("antstorch.create_glow_normalizing_flow_model_3d is required.")
            
        H, W, D = target_size
        C = cfg.get("C", 1)
        
        # Normalize K and hidden logic (same as training scripts)
        K = cfg.get("K", 16)
        L = cfg.get("L", 3)
        hidden = cfg.get("hidden", 64)
        
        if isinstance(K, list) and len(K) == 1: K = K[0]
        if isinstance(hidden, list) and len(hidden) == 1: hidden = hidden[0]
        
        if isinstance(K, int): K = [K] * L
        if isinstance(hidden, int): hidden = [hidden] * L
        
        model = create_glow_normalizing_flow_model_3d(
            input_shape=(C, H, W, D),
            L=L,
            K=K,
            hidden_channels=hidden,
            base=cfg.get("base", "glow"),
            glowbase_logscale_factor=cfg.get("glowbase_logscale_factor", 3.0),
            glowbase_min_log=cfg.get("glowbase_min_log", -1.0),
            glowbase_max_log=cfg.get("glowbase_max_log", 1.0),
            split_mode="channel", 
            scale=True, 
            scale_map=cfg.get("scale_map", "tanh"),
            leaky=0.0, 
            net_actnorm=bool(cfg.get("net_actnorm", False)), 
            scale_cap=cfg.get("scale_cap", 1.5)
        )
        
        return model.to(device)
        
    def prime_if_needed(self, model, target_size, device):
        """Prime the multiscale 3D Glow model using a multi-view dummy list."""
        # 1. Récupération dynamique du nombre de vues depuis l'arborescence des arguments
        num_views = getattr(model, "views", 1)
        
        # 2. Construction d'une liste de tenseurs 5D (un par vue)
        # Chaque tenseur respecte la forme (B=1, C=1, H, W, D)
        dummy_input = [
            torch.zeros([1, 1] + list(target_size), device=device)
            for _ in range(num_views)
        ]
        
        # 3. Si le modèle n'a qu'une seule vue, on extrait le tenseur unique 
        # pour éviter de passer une liste inutile
        if num_views == 1:
            dummy_input = dummy_input[0]

        # 4. Priming sécurisé de l'ActNorm sans calcul de gradient
        with torch.no_grad():
            try:
                model.forward_and_log_det(dummy_input)
            except Exception:
                # Fallback de secours sur le calcul de log-probabilité si forward échoue
                if isinstance(dummy_input, list):
                    _ = [model.log_prob(d) for d in dummy_input]
                else:
                    model.log_prob(dummy_input)

    def read_image(self, path: Path, target_size: Tuple[int, int, int], args: argparse.Namespace = None) -> torch.Tensor:
        """Read a 3D image, returning the PyTorch tensor."""
        t, _ = _read_image_3d(path, target_size)
        return t
        
    def save_single(self, x_tensor: torch.Tensor, out_path: Path, **kwargs):
        """Save a single 3D volume to disk (NIfTI)."""
        spacing = kwargs.get("spacing", None)
        # Force default extension if not provided
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".nii.gz")
        _save_nifti(x_tensor, out_path, spacing=spacing)
        
    def save_volume(self, x_tensor: torch.Tensor, out_path: Path, nrow: int = 1, **kwargs):
        """Save a batch of 3D volumes. For 3D, this creates a 4D NIfTI."""
        spacing = kwargs.get("spacing", None)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".nii.gz")
        _save_nifti(x_tensor, out_path, spacing=spacing)

    def ndim(self) -> int:
        """Retourne le nombre de dimensions spatiales."""
        return 3

    def interp_mode(self) -> str:
        """Mode d'interpolation PyTorch pour le redimensionnement 3D."""
        return "trilinear"

    def default_cov_rank(self) -> int:
        """Rang par défaut pour l'estimation de covariance Woodbury en 3D."""
        return 64  # Évite l'explosion de la RAM par rapport à 256

    def default_cov_estimator(self) -> str:
        """Estimateur de covariance par défaut."""
        return "empirical"

    def coerce_nd(self, tensor: torch.Tensor) -> torch.Tensor:
        """Force le tenseur au format 5D (B, C, H, W, D) requis pour la 3D."""
        while tensor.dim() < 5:
            tensor = tensor.unsqueeze(0)
        return tensor

    def parse_size(self, size_arg) -> tuple:
        """Convertit l'argument de taille en tuple de 3 entiers (H, W, D)."""
        if isinstance(size_arg, str):
            return tuple(map(int, size_arg.strip().split()))
        return tuple(size_arg)

    def parse_spacing(self, spacing_arg) -> tuple:
        """Convertit l'argument de spacing en tuple de 3 floats."""
        if isinstance(spacing_arg, str):
            return tuple(map(float, spacing_arg.strip().split()))
        return tuple(spacing_arg)

    def edit_latents_to_mean(
        self,
        z_list: List[torch.Tensor],
        gauss_blob: dict,
        view_name: str,
        levels_to_edit: List[int],
        **kw,
    ) -> List[torch.Tensor]:
        """Délégation de l'édition des vecteurs latents 3D vers la fonction utilitaire."""
        return _edit_latents_to_mean_for_view_3d(
            z_list, gauss_blob, view_name, levels_to_edit, **kw
        )



# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility: expose module-level main_* aliases pointing to the
# class methods so that any code/bash scripts importing old main_* functions still work.
# ─────────────────────────────────────────────────────────────────────────────

_tool = GlowTool3D()

main_gauss_fit             = _tool.cmd_gauss_fit
main_gauss_impute          = _tool.cmd_gauss_impute
main_recon                 = _tool.cmd_recon
main_recon_template        = _tool.cmd_recon_template
main_recon_cohort_template = _tool.cmd_recon_cohort_template
main_recon_temperature     = _tool.cmd_recon_temperature
main_recon_interpolate     = _tool.cmd_recon_interpolate
main_calc_distance         = _tool.cmd_calc_distance
main_sample                = _tool.cmd_sample

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tool.run()