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
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import GlowToolBase

# ─────────────────────────────────────────────────────────────────────────────
# 3D Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _read_image_3d(pth: Path, target_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[torch.Tensor, tuple]:
    """
    Read a 3D NIfTI image via ANTs and return a (1, 1, H, W, D) float32 tensor
    along with its native spacing.
    """
    img = ants.image_read(str(pth))
    arr = img.numpy()
    
    # Coerce to float32 and add Batch/Channel dimensions (1, 1, H, W, D)
    t = torch.from_numpy(arr).to(torch.float32)
    while t.dim() < 5:
        t = t.unsqueeze(0)
        
    return t, img.spacing

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
            num_levels=L,
            num_steps_per_level=K,
            hidden_channels=hidden,
            use_actnorm=cfg.get("use_actnorm", True),
            use_split=cfg.get("use_split", True),
            apply_background_mask=cfg.get("apply_background_mask", False),
        )
        return model.to(device)
        
    def prime_if_needed(self, model: nn.Module, target_size: Tuple[int, int, int], device: torch.device):
        """Run a dummy forward pass to initialize ActNorm layers on the specific device."""
        H, W, D = target_size
        dummy = torch.randn(1, 1, H, W, D, device=device)
        with torch.no_grad():
            model.forward_and_log_det(dummy)
            
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

    def edit_latents_to_mean(self, latents: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Recentrage des vecteurs latents par rapport à la moyenne de la cohorte."""
        return latents - mu




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