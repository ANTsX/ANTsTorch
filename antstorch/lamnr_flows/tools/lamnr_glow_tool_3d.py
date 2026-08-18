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
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import GlowToolBase, to01

# ─────────────────────────────────────────────────────────────────────────────
# 3D Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _save_nifti(tensor: torch.Tensor, out_path: Path, spacing: Optional[tuple] = None):
    """
    Enregistre un SEUL volume 3D (ou une image 2D) au format NIfTI.

    Accepte un tenseur avec des dimensions de batch/canal superflues en
    tête (ex : (1, 1, H, W, D)), qui sont retirées avant l'écriture.

    Ne doit JAMAIS recevoir une pile de plusieurs volumes/panneaux
    empilés le long de l'axe 0 (ex : [x, x_hat, diff] pour plusieurs
    sujets) — un NIfTI 4D encoderait alors ambiguïment cet empilement
    comme un axe spatial ou temporel selon la convention ants/NIfTI, ce
    qui a été la source d'un bug réel (axe "panneaux" pris pour un axe
    spatial, axe de profondeur D pris pour le temps). Pour sauvegarder
    plusieurs volumes, utiliser save_volume, qui écrit un fichier NIfTI
    3D distinct par volume.
    """
    arr = tensor.detach().cpu().numpy()

    # Retirer les dimensions de taille 1 en tête (batch, canal) jusqu'à
    # atteindre une image 2D ou un volume 3D.
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"_save_nifti attend une image 2D ou un volume 3D après "
            f"réduction des dimensions de taille 1 ; forme obtenue : "
            f"{arr.shape}. Utilisez save_volume pour sauvegarder "
            f"plusieurs volumes (un fichier NIfTI 3D par volume)."
        )

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

    # Normalisation automatique vers [0, 1] si nécessaire.
    #
    # Tolérance epsilon plutôt qu'un seuil strict 0.0/1.0 : un modèle qui
    # reconstruit quasiment parfaitement peut déborder de quelques 1e-6/1e-5
    # par pur bruit de calcul flottant (ex : x_hat.min() = -4.45e-06 observé
    # sur une reconstruction dont l'écart réel avec x était de 0.000115).
    # Avec un seuil strict, ce dépassement microscopique déclenchait un
    # winsorize/renormalisation (percentiles 1%/99%) sur TOUT le volume --
    # ce qui redistribue la dynamique de x_hat très différemment de x (qui,
    # lui, reste à [0,1] exact et n'est jamais rescalé), gonflant
    # artificiellement (x - x_hat) bien au-delà de l'erreur réelle. Ce n'est
    # déclenché maintenant que par un dépassement réel (> tol), signe d'un
    # vrai problème de décodage plutôt que d'un artefact numérique bénin.
    tol = 1e-3
    try:
        if x.amin() < -tol or x.amax() > 1.0 + tol:
            x = to01(x, winsorize=True)
        else:
            x = x.clamp(0.0, 1.0)
    except Exception as e:
        print(f"[warn] Échec de la normalisation to01 : {e}")

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
            # Checkpoints trained before antsnormflows commit 2249ecd
            # (2026-07-26, conv) / f047e4e (2026-07-27, ActNorm) were built
            # with GlowBlock3d calling Invertible1x1x1Conv(channels, use_lu)
            # and ActNorm(...) with NO cap argument at all -- so those layers
            # silently used their own hardcoded defaults (2.5 and 5.0
            # respectively), never the model's configured scale_cap. Post-fix,
            # GlowBlock3d always wires the cap explicitly (falling back to
            # scale_cap when conv_s_cap/actnorm_s_cap are None); loading an old
            # checkpoint under that code recalibrates those two layers to
            # whatever scale_cap is configured (e.g. 1.5) instead of what they
            # were actually trained under -- this is exactly what broke
            # decode/reconstruction for ventilation_64x48x64_K32_L4_HC96
            # (trained 2026-07-08/09, before the fix).
            #
            # Default to the legacy caps (2.5 / 5.0) whenever the checkpoint's
            # own saved config doesn't say otherwise: every checkpoint trained
            # before this option existed simply lacks these keys entirely
            # (confirmed for all runs3d/runs2d checkpoints on disk as of
            # 2026-07-29), so an *absent* key means "legacy, needs the old
            # caps". train_lamnr_glow_3d.py now always saves both keys
            # (dict(vars(args)) in run_config.json), even when left at their
            # None default -- so a checkpoint trained after this change has
            # the key *present* with value None, cfg.get(...) returns that
            # None (not the 2.5/5.0 default below), and the conv/ActNorm
            # correctly use scale_cap like everything else. A checkpoint can
            # still explicitly force legacy behavior (or opt out) by setting
            # these keys itself.
            legacy_conv_cap=cfg.get("legacy_conv_cap", 2.5),
            actnorm_scale_cap=cfg.get("actnorm_scale_cap", 5.0),
            # Inference-time-only safety net for the affine coupling's
            # unbounded shift term (see AffineCoupling's docstring): traced
            # empirically to be the dominant cause of the temperature > 1.0
            # sample blow-up (values reaching +-hundreds vs the healthy
            # [~0, ~1] range at temp<=1.0), not Invertible1x1x1Conv/mixing.py,
            # which already has its own (separately fixed) off-diagonal cap
            # and was not the bottleneck here. 15.0 is generous relative to
            # every in-distribution activation magnitude observed while
            # tracing temp=1.0 sampling on this checkpoint, so it is inert
            # for temp<=1.0 use and only engages in the temperature>1 tail.
            # Not a learned parameter -- purely a runtime bound, so applying
            # it here does not require retraining and does not affect the
            # separate, currently-running training process (which builds
            # its own model instance via train_lamnr_glow_3d.py, untouched
            # by this tool-only default).
            shift_cap=cfg.get("shift_cap", 15.0),
            # This is the actual lever that fixes temperature > 1.0 blow-up
            # (see gen_clamp's docstring on create_glow_normalizing_flow_model_3d):
            # traced empirically -- the affine coupling's multiplicative
            # scale term pins at its own cap for several consecutive blocks
            # once sampling drifts out of distribution, compounding
            # exponentially; shift_cap above does not touch this. 25.0 is
            # generous relative to the largest inter-block activation
            # magnitude observed at temperature<=1.0 on this checkpoint
            # (stays under ~10), so it is inert for temp<=1.0 use and only
            # interrupts the temperature>1 runaway feedback loop.
            gen_clamp=cfg.get("gen_clamp", 25.0),
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

        # 4. Priming sécurisé de l'ActNorm sans calcul de gradient.
        # Doit passer par inverse_and_log_det (x -> z) : c'est la seule
        # méthode qui met en cache model._latent_shapes (cf. core.py), ce
        # dont sample()/sample_with_temperature() ont besoin ensuite.
        # forward_and_log_det va dans l'autre sens (z -> x) et ne met
        # jamais en cache les shapes, quel que soit l'input qu'on lui donne.
        with torch.no_grad():
            if isinstance(dummy_input, list):
                for d in dummy_input:
                    model.inverse_and_log_det(d)
            else:
                model.inverse_and_log_det(dummy_input)

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
        """
        Enregistre un lot de volumes 3D empilés le long de l'axe 0 (ex :
        les panneaux [x, x_hat, diff] produits par `recon`, répétés pour
        plusieurs sujets) sous forme de fichiers NIfTI 3D SÉPARÉS — un par
        volume — plutôt qu'un unique NIfTI 4D.

        _save_nifti reste volontairement limitée aux volumes 3D/images 2D
        uniques (voir sa docstring) : empiler plusieurs panneaux dans un
        NIfTI 4D est ambigu, la convention ants/NIfTI pouvant traiter cet
        axe comme spatial ou temporel selon sa position — source d'un bug
        réel rencontré précédemment (axe "panneaux" pris pour un axe
        spatial, axe de profondeur D pris pour le temps).

        `nrow` est interprété comme le nombre de volumes par groupe (ex :
        3 pour [x, x_hat, diff], 4 pour [x, x_hat, x_hat_e, diff] avec
        édition de latents). Fichiers de sortie nommés
        '{stem}_item{k:03d}_{label}.nii.gz' quand plusieurs groupes sont
        présents, ou '{stem}_{label}.nii.gz' pour un seul groupe de
        plusieurs volumes.
        """
        spacing = kwargs.get("spacing", None)
        out_path = Path(out_path)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".nii.gz")

        arr = x_tensor.detach().cpu()
        # Retirer l'axe canal (index 1) s'il vaut 1, sans toucher à
        # l'axe 0 (empilement des volumes/panneaux).
        if arr.dim() == 5 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]  # (N_total, H, W, D)

        n_total = int(arr.shape[0]) if arr.dim() >= 4 else 1

        # Cas majoritaire et déjà correct : un seul volume -> comportement
        # inchangé, un seul fichier écrit directement sous out_path.
        if n_total <= 1:
            _save_nifti(x_tensor, out_path, spacing=spacing)
            return

        n_panels = max(1, int(nrow))
        if n_total % n_panels != 0:
            print(
                f"[warn] save_volume : {n_total} volumes non divisibles "
                f"par nrow={n_panels} ; sauvegarde individuelle sans "
                f"regroupement par panneau."
            )
            n_panels = 1

        default_labels = {
            1: ["vol"],
            3: ["orig", "recon", "diff"],
            4: ["orig", "recon", "recon_edit", "diff"],
        }
        labels = default_labels.get(n_panels, [f"panel{p}" for p in range(n_panels)])

        n_groups = n_total // n_panels
        stem = out_path.name[:-len(".nii.gz")] if out_path.name.endswith(".nii.gz") else out_path.stem
        parent = out_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        written = []
        for g in range(n_groups):
            for p in range(n_panels):
                idx = g * n_panels + p
                vol = arr[idx]  # (H, W, D) après retrait du canal
                label = labels[p]
                if n_groups == 1:
                    fname = f"{stem}_{label}.nii.gz"
                else:
                    fname = f"{stem}_item{g:03d}_{label}.nii.gz"
                fpath = parent / fname
                _save_nifti(vol, fpath, spacing=spacing)
                written.append(fpath)

        print(f"[save_volume] {len(written)} fichier(s) NIfTI 3D écrit(s) sous {parent}/")

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
