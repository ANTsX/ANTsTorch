"""
train_lamnr_glow_2d.py
======================
2D LAMNr Glow trainer — thin subclass of BaseLAMNrTrainer.

This script retains only 2D-specific concerns:
  - CLI argument definitions (adds --H, --W, --slice-idx, --sample-grid-norm)
  - 2D data loading via build_loaders_from_globs()
  - PNGMultiViewDataset for PNG inputs
  - LAMNrGlow2DTrainer.build_models()  → create_glow_normalizing_flow_model_2d
  - LAMNrGlow2DTrainer.extract_view()  → 2D slice view extraction + to01

All shared logic (training loop, gradient accumulation, alignment losses,
memory management, checkpoint save/load) lives in base_trainer.py.
"""

from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import ants
import antstorch
import antsnormflows as nf

from antstorch.lamnr_flows.core.base_trainer import (
    BaseLAMNrTrainer,
    GlowDataParallel,
    GlowStepWrapper,
    _check_hw_divisible,
    _extract_views_from_batch,
    set_deterministic,
    to01,
    screen_dump_run_config,
)


# ---------------------------------------------------------------------------
# AugSchedulerWrapper (pickling-safe, needed by antstorch.ImageDataset)
# ---------------------------------------------------------------------------

class AugSchedulerWrapper:
    """Wrapper to allow pickling of the aug scheduler."""
    def __init__(self, sched):
        self.sched = sched

    def __call__(self, step: int):
        return self.sched.step(step)


# ---------------------------------------------------------------------------
# PNG dataset (2D only)
# ---------------------------------------------------------------------------

import torch
import numpy as np
from pathlib import Path
from torchvision.io import image as io
from torchvision.transforms import v2
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# PNG dataset (2D only) - Adapté pour l'entraînement sur Visages
# ---------------------------------------------------------------------------

class PNGMultiViewDataset(Dataset):
    """Multi-view dataset backed by PNG files."""

    def __init__(self, images_list, target_size=(128, 128), do_aug=False):
        self.images_list = images_list
        self.do_aug      = do_aug
        
        from torchvision.transforms import InterpolationMode
        
        # Transformations de base (toujours appliquées)
        self.base_transform = v2.Compose([
            v2.Resize(target_size, antialias=True, interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),            
        ])
        
        # Transformations spatiales (si augmentation activée)
        if self.do_aug:
            self.spatial_transforms = v2.Compose([
                # LE PLUS IMPORTANT POUR LES VISAGES : Double la taille du dataset virtuellement
                v2.RandomHorizontalFlip(p=0.5), 
                
                # Optionnel : de très légères rotations/translations si les visages ne sont pas parfaitement centrés
                # v2.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)), 
                
                # Désactivé : l'ElasticTransform détruit la structure photoréaliste des visages
                # v2.ElasticTransform(alpha=50.0, sigma=5.0),
            ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        views   = self.images_list[idx]
        tensors = []
        for v in views:
            if isinstance(v, (str, Path)):
                img = io.read_image(str(v), mode=io.ImageReadMode.RGB)
            elif isinstance(v, np.ndarray):
                img = torch.from_numpy(v).permute(2, 0, 1)
            else:
                img = v
            img = self.base_transform(img)
            tensors.append(img)

        # On empile pour obtenir la forme (Vues, Canaux, Hauteur, Largeur) -> (V, C, H, W)
        stacked = torch.stack(tensors)
        
        if self.do_aug:
            # 1. APPLICATION DES TRANSFORMATIONS SPATIALES (manquait dans le code d'origine)
            # Note: torchvision v2 gère intelligemment la dimension 'V' supplémentaire
            # S'il y a plusieurs vues (T1/T2 ou autre), elles subiront exactement le même retournement !
            stacked = self.spatial_transforms(stacked)
            
            # 2. Ajout du bruit de déquantification (essentiel pour les Normalizing Flows)
            noise   = torch.randn_like(stacked) * 0.05 # Pensez à baisser ce 0.05 si les visages sont trop flous
            stacked = torch.clamp(stacked + noise, 0.0, 1.0)
            
        return stacked
    

# ---------------------------------------------------------------------------
# 2D data loader
# ---------------------------------------------------------------------------

def build_loaders_from_globs(
    view_specs,
    H, W,
    train_samples, val_samples,
    batch, num_workers,
    slice_idx: int,
    val_frac: float,
    subject_limit: Optional[int],
    do_aug: bool = True,
    aug_schedules=None,
    disable_aug_anneal: bool = False,
    seed: int = 0,
):
    def _expand_globs_per_view(view_specs):
        import glob, os
        per_view_files = []
        for specs in view_specs:
            paths = []
            for pat in specs:
                pat = os.path.expanduser(pat)
                paths.extend(glob.glob(pat))
            paths = sorted({str(p) for p in paths})
            per_view_files.append([Path(p) for p in paths])
        return per_view_files

    def _group_by_subject(per_view_files):
        from collections import defaultdict
        per_view_by_subj = []
        subj_sets        = []
        for files in per_view_files:
            d = defaultdict(list)
            for f in files:
                subj = f.name.split("_")[0]
                d[subj].append(f)
            for k in d:
                d[k] = sorted(d[k])
            per_view_by_subj.append(d)
            subj_sets.append(set(d.keys()))
        common_subj = set.intersection(*subj_sets) if subj_sets else set()
        if not common_subj:
            raise RuntimeError("No common subjects found across views.")
        per_subj = {}
        for s in sorted(common_subj):
            counts = [len(d[s]) for d in per_view_by_subj]
            if len(set(counts)) != 1:
                raise RuntimeError(f"Subject {s} has different counts across views: {counts}")
            M = counts[0]
            per_subj[s] = [
                [per_view_by_subj[v][s][k] for v in range(len(per_view_by_subj))]
                for k in range(M)
            ]
        return per_subj

    def _read_slice(path: Path, idx: int, H: int, W: int):
        import ants
        import numpy as np
        from PIL import Image

        if path.suffix.lower() == ".png":
            pil_img = Image.open(path).convert("RGB") # Force le mode monocanal (Grayscale)
            
            pil_img = pil_img.resize((W, H), resample=Image.Resampling.BILINEAR)

            import torchvision.transforms.functional as TF
            arr = TF.to_tensor(pil_img) 
            return arr
            
        # Branche NIfTI standard (inchangée)
        im = ants.image_read(str(path))
        if im.dimension == 3:
            slc = ants.slice_image(im, axis=2, idx=idx, collapse_strategy=1)
        else:
            slc = im
            
        resize_factor = min(float(H) / slc.shape[0], float(W) / slc.shape[1])
        spacing = (slc.spacing[0] / resize_factor, slc.spacing[1] / resize_factor)
        slc = ants.resample_image(slc, spacing, use_voxels=False, interp_type=0)
        slc = ants.pad_or_crop_image_to_size(slc, (H, W))
        return slc

    per_view_files  = _expand_globs_per_view(view_specs)
    per_subj        = _group_by_subject(per_view_files)
    subjects        = list(sorted(per_subj.keys()))
    if subject_limit and subject_limit > 0:
        subjects = subjects[: int(subject_limit)]

    def _load_subject(s):
        return [
            [_read_slice(Path(f), slice_idx, H, W) for f in sample]
            for sample in per_subj[s]
        ]

    if num_workers > 0:
        print(f"[info] Loading {len(subjects)} subjects using {num_workers} threads…")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            images_by_subject = list(
                tqdm(ex.map(_load_subject, subjects), total=len(subjects), desc="Loading")
            )
    else:
        images_by_subject = [
            _load_subject(s) for s in tqdm(subjects, desc="Loading")
        ]

    if not images_by_subject:
        raise RuntimeError("No images assembled from --view globs.")

    n_subj = len(images_by_subject)
    rng    = np.random.default_rng(seed)
    idx    = np.arange(n_subj)
    rng.shuffle(idx)
    n_val  = min(max(0, int(round(float(val_frac) * n_subj))), n_subj - 1)
    val_set = set(idx[:n_val].tolist())

    images_train = [s for si, subj in enumerate(images_by_subject)
                    if si not in val_set for s in subj]
    images_val   = [s for si, subj in enumerate(images_by_subject)
                    if si in val_set for s in subj]
    if not images_train:
        raise RuntimeError("Split produced an empty training set.")

    tmpl = images_train[0][0]

    if aug_schedules and not disable_aug_anneal:
        sched       = antstorch.MultiParamScheduler(antstorch.parse_schedules(aug_schedules))
        aug_sched_fn = AugSchedulerWrapper(sched)
    else:
        aug_sched_fn = None

    global_step = Value("i", 0)

    if isinstance(tmpl, ants.core.ants_image.ANTsImage):
        train_ds = antstorch.ImageDataset(
            images=images_train, template=tmpl,
            do_data_augmentation=do_aug,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.05,
            data_augmentation_sd_deformation=10.0,
            data_augmentation_noise_model="additivegaussian",
            data_augmentation_noise_parameters=(0.0, 0.05),
            data_augmentation_sd_simulated_bias_field=0.00000001,
            data_augmentation_sd_histogram_warping=0.025,
            number_of_samples=int(train_samples),
            aug_scheduler=aug_sched_fn,
        )
        val_ds = antstorch.ImageDataset(
            images=(images_val if images_val else images_train[:1]),
            template=tmpl,
            do_data_augmentation=True,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.0,
            data_augmentation_sd_deformation=0.0,
            data_augmentation_noise_model="additivegaussian",
            data_augmentation_noise_parameters=(0.0, 0.0),
            data_augmentation_sd_simulated_bias_field=0.0,
            data_augmentation_sd_histogram_warping=0.0,
            number_of_samples=int(val_samples),
        )
    else:
        train_ds = PNGMultiViewDataset(
            images_list=images_train, 
            target_size=(H, W), 
            do_aug=do_aug  
        )
        
        # La validation NE DOIT PAS être augmentée
        val_ds = PNGMultiViewDataset(
            images_list=(images_val if images_val else images_train[:1]),
            target_size=(H, W), 
            do_aug=False  # CRITIQUE : Toujours False pour l'évaluation !
        )

    train_ds.global_step_ref = global_step

    if torch.cuda.is_available():
        dev_type = "cuda"
    elif torch.backends.mps.is_available():
        dev_type = "mps"
    else:
        dev_type = "cpu"
    use_pin_memory = (dev_type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(16, batch), shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, global_step


# ---------------------------------------------------------------------------
# 2D Trainer subclass
# ---------------------------------------------------------------------------

class LAMNrGlow2DTrainer(BaseLAMNrTrainer):
    """
    2D-specific subclass of BaseLAMNrTrainer.

    Overrides
    ---------
    build_models  → create_glow_normalizing_flow_model_2d
    build_loaders → build_loaders_from_globs (2D slice extraction)
    extract_view  → handles (B, V, C, H, W) or (B, C, H, W) batches
    """

    def build_models(self, args) -> List[nn.Module]:
        from antstorch import create_glow_normalizing_flow_model_2d
        dev         = self.dev
        input_shape = self.input_shape
        models      = []

        for vi in range(args.num_views):
            m = create_glow_normalizing_flow_model_2d(
                input_shape=input_shape,
                L=args.L, K=args.K,
                hidden_channels=args.hidden,
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
            )
            m = m.to(dtype=torch.float32, device=dev).float().train()

            # Force ActNorm initialisation on the target device
            with torch.no_grad():
                dummy = torch.randn((1, *input_shape), device=dev, dtype=torch.float32)
                print(f"[init] Initializing ActNorm for view {vi} on {dev}…")
                try:
                    _ = m.log_prob(dummy)
                except Exception as e:
                    print(f"[warn] ActNorm init warning view {vi}: {e}")
                del dummy
                gc.collect()

            if not hasattr(m, "input_shape"):
                m.input_shape = input_shape

            if torch.cuda.device_count() > 1 and len(args.devices.split(",")) > 1:
                print(f"[info] Wrapping view {vi} in DataParallel on {args.devices}")
                device_ids = [int(d.split(":")[-1]) for d in args.devices.split(",")]
                models.append(GlowDataParallel(GlowStepWrapper(m), device_ids=device_ids))
            else:
                models.append(m)

        return models

    def build_loaders(self, args):
        train_loader, val_loader, global_step = build_loaders_from_globs(
            view_specs=args.view,
            H=args.H, W=args.W,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            batch=args.batch,
            num_workers=args.num_workers,
            slice_idx=args.slice_idx,
            val_frac=float(args.val_frac),
            subject_limit=(args.subject_limit if args.subject_limit > 0 else None),
            do_aug=True,
            aug_schedules=(args.aug_schedules if not args.disable_aug_anneal else None),
            disable_aug_anneal=args.disable_aug_anneal,
            seed=args.seed,
        )
        # Determine channel count from first batch
        sample_batch = next(iter(train_loader))
        C = sample_batch.shape[2] if sample_batch.ndim == 5 else sample_batch.shape[1]
        self.input_shape = (C, args.H, args.W)
        args.C            = C
        args.channels     = C
        args.input_shape  = self.input_shape
        return train_loader, val_loader, global_step

    def extract_view(
        self, batch: object, vi: int, dev: torch.device
    ) -> torch.Tensor:
        """Extract view vi from a 2D batch and normalize to [0, 1]."""
        xs = _extract_views_from_batch(batch, num_views=self.args.num_views)
        x_v = xs[vi].to(dev, dtype=torch.float32)
        return x_v       


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("LAMNr Glow 2D trainer")

    # Input data
    ap.add_argument("--view", action="append", nargs="+", required=True,
        help="Path patterns for each modality. Use one --view per modality.")
    ap.add_argument("--H", type=int, default=128, help="Target image height (pixels).")
    ap.add_argument("--W", type=int, default=128, help="Target image width (pixels).")

    # Architecture
    ap.add_argument("--L", type=int, default=4, help="Number of resolution levels.")
    ap.add_argument("--K", type=int, nargs="+", default=[32],
        help="Flow steps per level (single int or list matching L).")
    ap.add_argument("--hidden", type=int, nargs="+", default=[96],
        help="Hidden channels in coupling networks.")
    ap.add_argument("--base", type=str, default="glow", choices=["glow", "diag"],
        help="Base distribution type.")
    ap.add_argument("--glowbase-logscale-factor", type=float, default=3.0)
    ap.add_argument("--glowbase-min-log",         type=float, default=-5.0)
    ap.add_argument("--glowbase-max-log",         type=float, default=5.0)
    ap.add_argument("--scale-map", type=str, default="tanh",
        choices=["tanh", "exp", "sigmoid", "sigmoid_inv"])
    ap.add_argument("--scale-cap", type=float, default=2.0)
    ap.add_argument("--net-actnorm", action="store_true")

    # Training loop
    ap.add_argument("--batch",          type=int,   default=32)
    ap.add_argument("--train-samples",  type=int,   default=6000)
    ap.add_argument("--val-samples",    type=int,   default=256)
    ap.add_argument("--max-iter",       type=int,   default=30000)
    ap.add_argument("--extra-iters",    type=int,   default=0)
    ap.add_argument("--eval-interval",  type=int,   default=1000)
    ap.add_argument("--plot-interval",  type=int,   default=1000)
    ap.add_argument("--num-workers",    type=int,   default=4)

    # Hardware & precision
    ap.add_argument("--devices",   type=str, default="cuda:0")
    ap.add_argument("--precision", type=str, default="mixed",
        choices=["double", "float", "mixed"])
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--seed",      type=int, default=0)

    # Optimizer & scheduler
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--weight-decay",     type=float, default=1e-5)
    ap.add_argument("--warmup-iters",     type=int,   default=800)
    ap.add_argument("--lr-decay-gamma",   type=float, default=1.0)
    ap.add_argument("--lr-decay-steps",   type=int,   default=0)
    ap.add_argument("--plateau-factor",   type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int,   default=4)
    ap.add_argument("--plateau-threshold",type=float, default=1e-4)
    ap.add_argument("--plateau-cooldown", type=int,   default=0)
    ap.add_argument("--min-lr",           type=float, default=1e-6)

    # Gradients & EMA
    ap.add_argument("--grad-clip",  type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int,   default=1)
    ap.add_argument("--ema",        action="store_true")
    ap.add_argument("--ema-decay",  type=float, default=0.9995)

    # Checkpointing
    ap.add_argument("--resume",      type=str,  default="")
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--out-dir",     type=str,  default="runs_glow2d")
    ap.add_argument("--use-ckpt-config", action="store_true")

    # Dataset
    ap.add_argument("--slice-idx",      type=int,   default=120)
    ap.add_argument("--val-frac",       type=float, default=0.10)
    ap.add_argument("--subject-limit",  type=int,   default=0)
    ap.add_argument("--smooth-alpha",   type=float, default=0.1)

    # Alignment
    ap.add_argument("--align", default="none",
        choices=["none","infonce","barlow","vicreg","hsic","pearson","mse"])
    ap.add_argument("--align-weight",   type=float, default=0.05)
    ap.add_argument("--align-warmup",   type=int,   default=500)
    ap.add_argument("--proj-dim",       type=int,   default=256)
    ap.add_argument("--proj-hidden",    type=int,   default=512)
    ap.add_argument("--temperature",    type=float, default=0.1)
    ap.add_argument("--barlow-lambda",  type=float, default=5e-3)
    ap.add_argument("--weighting",      default="fixed", choices=["fixed","kendall"])
    ap.add_argument("--init-logvar-nll",   type=float, default=0.0)
    ap.add_argument("--init-logvar-align", type=float, default=0.0)
    ap.add_argument("--vicreg-inv",    type=float, default=25.0)
    ap.add_argument("--vicreg-cov",    type=float, default=1.0)
    ap.add_argument("--vicreg-var",    type=float, nargs="+", default=[25.0])
    ap.add_argument("--vicreg-gamma",  type=float, nargs="+", default=[1.0])
    ap.add_argument("--hsic-sigma",    type=float, default=0.0)

    # Augmentation
    ap.add_argument("--aug-schedules", type=str,
        default=(
            "noise_std:cos:0.05->0.00@150k,"
            "sd_affine:linear:0.05->0.00@80k,"
            "sd_deformation:cos:0.20->0.00@100k,"
            "sd_simulated_bias_field:cos:1.00->0.00@120k,"
            "sd_histogram_warping:exp:0.05->0.00@120k"
        ))
    ap.add_argument("--disable-aug-anneal", action="store_true")

    # Previews
    ap.add_argument("--sample-mode", default="model", choices=["model","data","off"])
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--sample-grid-norm", default="to01", choices=["to01","clamp","both"])

    # Screening
    ap.add_argument("--screen",        default="none", choices=["none","cca","hsic"])
    ap.add_argument("--screen-warmup", type=int,   default=1000)
    ap.add_argument("--screen-refresh",type=int,   default=0)
    ap.add_argument("--screen-frac",   type=float, default=0.5)
    ap.add_argument("--cca-ridge",     type=float, default=1e-3)
    ap.add_argument("--prefilter-frac",type=float, default=0.5)

    args = ap.parse_args()
    args.num_views = len(args.view)

    # Normalise K and hidden
    if isinstance(args.K, list):
        if len(args.K) == 1:
            args.K = args.K[0]
        elif len(args.K) != args.L:
            raise ValueError(f"len(K)={len(args.K)} must equal L={args.L}")
        else:
            args.K = tuple(args.K)
    if isinstance(args.hidden, list):
        if len(args.hidden) == 1:
            args.hidden = args.hidden[0]
        elif len(args.hidden) != args.L:
            raise ValueError(f"len(hidden)={len(args.hidden)} must equal L={args.L}")
        else:
            args.hidden = tuple(args.hidden)

    _check_hw_divisible(args.H, args.W, args.L)
    return args


def main():
    args    = _build_args()
    trainer = LAMNrGlow2DTrainer()
    trainer.setup(args)
    screen_dump_run_config(args, Path(args.out_dir))
    trainer.train()


if __name__ == "__main__":
    main()
