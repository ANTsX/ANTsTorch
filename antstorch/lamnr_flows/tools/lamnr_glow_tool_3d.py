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
from typing import Optional, Tuple

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

def _save_nifti(tensor: torch.Tensor, out_path: Path, spacing: Optional[tuple] = None):
    """
    Save a (1, 1, H, W, D) or (B, C, H, W, D) tensor to NIfTI.

    B == 1 -> plain 3D NIfTI (H, W, D).
    B  > 1 -> 4D NIfTI (H, W, D, B) -- e.g. `recon`'s multi-subject panel
    (original / reconstruction / diff stacked along the batch dim). ANTs has
    no 5D image type (`fromNumpyF5` does not exist), so a (B, C, H, W, D)
    array with B > 1 must be reduced to 4D, not just have leading
    size-1 dims squeezed away.
    """
    arr = tensor.detach().cpu().numpy()

    # Squeeze out a size-1 Channel dimension first (arr is (B, C, H, W, D)
    # from _coerce_nchwd_5d, which always forces C == 1).
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0]  # -> (B, H, W, D)

    # Squeeze out a size-1 leading dim (covers plain 3D/4D inputs too).
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4:
        # Remaining leading axis (size > 1) is a genuine batch of volumes
        # (e.g. panels from `recon`) -- move it last so ANTs sees a normal
        # 4D (H, W, D, N) image instead of failing on a 5D array.
        arr = np.moveaxis(arr, 0, -1)

    img = ants.from_numpy(arr)
    if spacing is not None:
        try:
            img.set_spacing(spacing)
        except Exception as e:
            print(f"[warn] Could not set spacing: {e}")
            
    ants.image_write(img, str(out_path))

def _coerce_nchwd_5d(x, target_hwd=None):
    """Coerce n'importe quel tenseur/liste en sortie vers (B, C, H, W, D) float32."""
    if isinstance(x, (list, tuple)):
        # Filtrer pour ne garder que les tenseurs 4D ou 5D
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (4, 5)]
        if not cands:
            raise RuntimeError("La sortie n'est pas un tenseur 4D/5D.")
        
        # Sélectionner le candidat avec le plus grand volume spatial (H * W * D)
        volumes = [int(t.shape[-1]) * int(t.shape[-2]) * int(t.shape[-3]) for t in cands]
        x = cands[int(torch.tensor(volumes).argmax().item())]

    if not torch.is_tensor(x):
        raise RuntimeError(f"Type de sortie inattendu : {type(x)}")

    # Si 4D (C, H, W, D), ajouter la dimension batch -> (1, C, H, W, D)
    if x.dim() == 4:
        x = x.unsqueeze(0)
    
    # Si la dimension canal est > 1, moyenner pour obtenir C=1
    if x.size(1) > 1:
        x = x.mean(dim=1, keepdim=True)
    
    x = x.float()
    
    # Normalisation automatique vers [0, 1] si nécessaire
    try:
        # NOTE: if x contains NaN, `x.amin() < 0.0` and `x.amax() > 1.0` are
        # BOTH silently False (any comparison against NaN is False), so a
        # fully-NaN tensor used to skip to01() entirely -- no normalization,
        # no warning, just raw NaN passed downstream. Check finiteness
        # explicitly so a NaN/Inf decode routes through to01() (which now
        # warns loudly) instead of slipping past this guard unnoticed.
        needs_norm = (not torch.isfinite(x).all()) or bool(x.amin() < 0.0) or bool(x.amax() > 1.0)
        if needs_norm:
            x = to01(x, winsorize=True)
    except Exception:
        pass

    # Interpolation vers la taille cible (H, W, D)
    if target_hwd is not None:
        target_shape = (int(target_hwd[0]), int(target_hwd[1]), int(target_hwd[2]))
        if (x.shape[-3], x.shape[-2], x.shape[-1]) != target_shape:
            x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
            
    return x

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

        # Diagnostic: surface which architecture hyperparameters came from
        # the checkpoint's own saved config vs. fell back to this tool's
        # (possibly different from the training script's) defaults. A
        # silent fallback here means the built model can numerically
        # diverge from what was actually trained, even though weight
        # loading itself succeeds cleanly.
        _resolved = {
            "base":                     ("base", "glow"),
            "glowbase_logscale_factor": ("glowbase_logscale_factor", 3.0),
            "glowbase_min_log":         ("glowbase_min_log", -1.0),
            "glowbase_max_log":         ("glowbase_max_log", 1.0),
            "scale_map":                ("scale_map", "tanh"),
            "net_actnorm":              ("net_actnorm", False),
            "scale_cap":                ("scale_cap", 1.5),
        }
        for label, (key, default) in _resolved.items():
            present = key in cfg
            print(f"  [build_model] {label}={cfg.get(key, default)!r} "
                  f"({'from checkpoint config' if present else f'FALLBACK DEFAULT, key {key!r} missing from cfg'})")

        # Backward compatibility: checkpoints trained before antsnormflows
        # commit 2249ecd ("BUG: Minor numerical fixes") were trained with
        # Invertible1x1x1Conv silently ignoring `scale_cap` and always using
        # its own hardcoded default (2.5), regardless of what the config
        # says. Loading those weights under the now-correct (config-driven)
        # behavior reinterprets the trained log_S parameters under a
        # different clamp than they were fit under, which can catastrophically
        # break decode (confirmed empirically: 45/45 subjects non-finite with
        # the fix vs. 2/45 -- the expected outlier rate -- with the legacy
        # cap). Checkpoints trained AFTER the fix record
        # "s_cap_wired_to_conv": True in their config; its absence means the
        # checkpoint predates the fix and needs the legacy cap preserved.
        s_cap_wired = bool(cfg.get("s_cap_wired_to_conv", False))
        legacy_conv_cap = None if s_cap_wired else 2.5
        print(f"  [build_model] s_cap_wired_to_conv={s_cap_wired!r} "
              f"({'from checkpoint config' if 's_cap_wired_to_conv' in cfg else 'missing from cfg -- assuming a pre-fix checkpoint'}) "
              f"-> legacy_conv_cap={legacy_conv_cap!r} "
              f"({'invertible conv uses scale_cap normally' if legacy_conv_cap is None else 'invertible conv pinned to the pre-fix hardcoded cap'})")

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
            scale_cap=cfg.get("scale_cap", 1.5),
            legacy_conv_cap=legacy_conv_cap,
        )

        return model.to(device)

    def prime_if_needed(self, model, target_size, device):
        """Prime the multiscale 3D Glow model using a data-shaped dummy input."""
        num_views = getattr(model, "views", 1)

        dummy_input = [
            torch.zeros([1, 1] + list(target_size), device=device)
            for _ in range(num_views)
        ]
        if num_views == 1:
            dummy_input = dummy_input[0]

        with torch.no_grad():
            try:
                if isinstance(dummy_input, list):
                    _ = [model.inverse_and_log_det(d) for d in dummy_input]
                else:
                    model.inverse_and_log_det(dummy_input)   # ✅ corrigé : même direction que _encode_latents
            except Exception:
                if isinstance(dummy_input, list):
                    _ = [model.log_prob(d) for d in dummy_input]
                else:
                    model.log_prob(dummy_input)

    def read_image(self, path: "Path", target_size, **kw) -> torch.Tensor:
        import ants
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"{path}")
        
        img = ants.image_read(str(path))
        H, W, D = target_size
        
        resize_factor = min(float(H)/float(img.shape[0]), 
                            float(W)/float(img.shape[1]),
                            float(D)/float(img.shape[2]))
        
        spacing = (img.spacing[0] / resize_factor, 
                   img.spacing[1] / resize_factor,
                   img.spacing[2] / resize_factor)   
        
        img = ants.resample_image(img, spacing, use_voxels=False, interp_type=0)
        img = ants.pad_or_crop_image_to_size(img, (H, W, D))
        
        arr = img.numpy()
        if arr.ndim == 3: 
            arr = arr[np.newaxis, ...] # (1, H, W, D)
            
        t = torch.from_numpy(arr).float()
        
        x_min = t.amin(dim=(1, 2, 3), keepdim=True)
        x_max = t.amax(dim=(1, 2, 3), keepdim=True)
        t = (t - x_min) / (x_max - x_min + 1e-8)
        
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

    def coerce_nd(self, x, target_size) -> torch.Tensor:
        return _coerce_nchwd_5d(x, target_hwd=target_size)

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
