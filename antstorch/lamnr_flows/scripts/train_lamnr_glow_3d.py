"""
train_lamnr_glow_3d.py
======================
3D LAMNr Glow trainer — thin subclass of BaseLAMNrTrainer.

This script retains only 3D-specific concerns:
  - CLI argument definitions (adds --D, --spatial-dims)
  - 3D volumetric data loading via build_loaders_from_globs_3d()
  - LAMNrGlow3DTrainer.build_models()  → create_glow_normalizing_flow_model_3d
  - LAMNrGlow3DTrainer.extract_view()  → 3D volume view extraction + to01

All shared logic (training loop, gradient accumulation, alignment losses,
memory management, checkpoint save/load) lives in base_trainer.py.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import ants
import antstorch
import normflows as nf

from antstorch.lamnr_flows.core.base_trainer import (
    BaseLAMNrTrainer,
    GlowDataParallel,
    GlowStepWrapper,
    _check_hw_divisible,
    _extract_views_from_batch,
    set_deterministic,
    to01,
)


# ---------------------------------------------------------------------------
# 3D data loader
# ---------------------------------------------------------------------------

def build_loaders_from_globs_3d(
    view_specs,
    H, W, D,
    train_samples, val_samples,
    batch, num_workers,
    val_frac: float,
    subject_limit: Optional[int],
    do_aug: bool = True,
    aug_schedules=None,
    disable_aug_anneal: bool = False,
    seed: int = 0,
):
    """Load 3D ANTs volumes for multi-view training."""

    def _expand_globs_per_view(view_specs):
        import glob, os
        per_view_files = []
        for spec in view_specs:
            files = []
            for g in spec:
                g = os.path.expanduser(g)
                files.extend(sorted(glob.glob(g)))
            per_view_files.append([Path(f) for f in files])
        return per_view_files

    def _group_by_subject(per_view_files):
        import re
        from collections import defaultdict
        n_views     = len(per_view_files)
        subj_to_files = [defaultdict(list) for _ in range(n_views)]

        def _key(p: Path):
            match = re.search(r"(sub-[a-zA-Z0-9]+)", str(p))
            return match.group(1) if match else p.parent.name

        for vi, files in enumerate(per_view_files):
            for f in files:
                subj_to_files[vi][_key(f)].append(f)

        common = set(subj_to_files[0].keys())
        for vi in range(1, n_views):
            common &= set(subj_to_files[vi].keys())

        subj_map = {}
        for s in sorted(common):
            per_view_lists = []
            for vi in range(n_views):
                flist = sorted(subj_to_files[vi][s])
                if not flist:
                    break
                per_view_lists.append(flist)
            if len(per_view_lists) == n_views:
                subj_map[s] = list(zip(*per_view_lists))
        return subj_map

    def _read_volume(path: Path, H: int, W: int, D: int):
        img = ants.image_read(str(path))
        resize_factor = min(
            float(H) / img.shape[0],
            float(W) / img.shape[1],
            float(D) / img.shape[2],
        )
        spacing = (
            img.spacing[0] / resize_factor,
            img.spacing[1] / resize_factor,
            img.spacing[2] / resize_factor,
        )
        img = ants.resample_image(img, spacing, use_voxels=False, interp_type=0)
        img = ants.pad_or_crop_image_to_size(img, (H, W, D))
        return img

    per_view_files  = _expand_globs_per_view(view_specs)
    per_subj        = _group_by_subject(per_view_files)
    subjects        = list(sorted(per_subj.keys()))
    if subject_limit and subject_limit > 0:
        subjects = subjects[: int(subject_limit)]

    def _load_subject(s):
        return [
            [_read_volume(Path(f), H, W, D) for f in sample]
            for sample in per_subj[s]
        ]

    if num_workers > 0:
        print(f"[info] Loading {len(subjects)} subjects using {num_workers} threads…")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            images_by_subject = list(
                tqdm(ex.map(_load_subject, subjects), total=len(subjects), desc="Loading 3D")
            )
    else:
        images_by_subject = [
            _load_subject(s) for s in tqdm(subjects, desc="Loading 3D")
        ]

    if not images_by_subject:
        raise RuntimeError("No 3D images assembled from --view globs.")

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
        sched = antstorch.MultiParamScheduler(antstorch.parse_schedules(aug_schedules))
        def aug_sched_fn(step: int):
            return sched.step(step)
    else:
        aug_sched_fn = None

    global_step = Value("i", 0)

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
    train_ds.global_step_ref = global_step

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
# 3D Trainer subclass
# ---------------------------------------------------------------------------

class LAMNrGlow3DTrainer(BaseLAMNrTrainer):
    """
    3D-specific subclass of BaseLAMNrTrainer.

    Overrides
    ---------
    build_models  → create_glow_normalizing_flow_model_2d or _3d depending on --spatial-dims
    build_loaders → build_loaders_from_globs_3d (full volumetric loading)
    extract_view  → handles (B, V, C, H, W, D) or standard channel-interleaved batches
    cleanup_checkpoints → 3D milestone = every 20 000 iterations + disk-space alert
    """

    def build_models(self, args) -> List[nn.Module]:
        from antstorch import create_glow_normalizing_flow_model_2d
        try:
            from antstorch import create_glow_normalizing_flow_model_3d
        except Exception:
            create_glow_normalizing_flow_model_3d = None

        dev         = self.dev
        input_shape = self.input_shape
        models      = []

        for vi in range(args.num_views):
            if args.spatial_dims == 2:
                m = create_glow_normalizing_flow_model_2d(
                    input_shape=input_shape, L=args.L, K=args.K,
                    hidden_channels=args.hidden, base=args.base,
                    glowbase_logscale_factor=args.glowbase_logscale_factor,
                    glowbase_min_log=args.glowbase_min_log,
                    glowbase_max_log=args.glowbase_max_log,
                    split_mode="channel", scale=True, scale_map=args.scale_map,
                    leaky=0.0, net_actnorm=bool(args.net_actnorm), scale_cap=args.scale_cap,
                )
            else:
                if create_glow_normalizing_flow_model_3d is None:
                    raise RuntimeError(
                        "antstorch.create_glow_normalizing_flow_model_3d not available; "
                        "update antstorch or use --spatial-dims=2."
                    )
                m = create_glow_normalizing_flow_model_3d(
                    input_shape=input_shape, L=args.L, K=args.K,
                    hidden_channels=args.hidden, base=args.base,
                    glowbase_logscale_factor=args.glowbase_logscale_factor,
                    glowbase_min_log=args.glowbase_min_log,
                    glowbase_max_log=args.glowbase_max_log,
                    split_mode="channel", scale=True, scale_map=args.scale_map,
                    leaky=0.0, net_actnorm=bool(args.net_actnorm), scale_cap=args.scale_cap,
                )

            m = m.to(dev).float().train()
            # Cast any non-float32 parameters
            for name, p in m.named_parameters():
                if p.dtype != torch.float32:
                    print(f"[warn] casting param {name} {p.dtype} → float32")
                    p.data = p.data.float()

            # Force ActNorm init on device
            with torch.no_grad():
                dummy_shape = (1, *input_shape)
                dummy       = torch.randn(dummy_shape, device=dev, dtype=torch.float32)
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
        train_loader, val_loader, global_step = build_loaders_from_globs_3d(
            view_specs=args.view,
            H=args.H, W=args.W, D=args.D,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            batch=args.batch,
            num_workers=args.num_workers,
            val_frac=float(args.val_frac),
            subject_limit=(args.subject_limit if args.subject_limit > 0 else None),
            do_aug=True,
            aug_schedules=(args.aug_schedules if not args.disable_aug_anneal else None),
            disable_aug_anneal=args.disable_aug_anneal,
            seed=args.seed,
        )
        # 3D volumes: shape is always (C=1, H, W, D)
        C = 1
        if args.spatial_dims == 2:
            self.input_shape = (C, args.H, args.W)
        else:
            self.input_shape = (C, args.H, args.W, args.D)
        args.C           = C
        args.channels    = C
        args.input_shape = self.input_shape
        return train_loader, val_loader, global_step

    def extract_view(
        self, batch: object, vi: int, dev: torch.device
    ) -> torch.Tensor:
        """
        Extract view vi from a 3D batch.

        The 3D ImageDataset stacks views along dim=1 as (B, n_views, H, W, D),
        so we slice [:, vi:vi+1, ...] to get (B, 1, H, W, D).
        """
        if torch.is_tensor(batch):
            x_v = batch[:, vi : vi + 1, ...].to(dev)
        else:
            xs  = _extract_views_from_batch(batch, num_views=self.args.num_views)
            x_v = xs[vi].to(dev)
        return to01(x_v.to(dtype=torch.float32))

    def cleanup_checkpoints(self, keep_every: int = 20_000) -> None:
        """3D variant — longer milestone interval + disk-space warning."""
        for f in self.run_dir.glob("training_state_it*.pt"):
            try:
                it_num = int(f.stem.split("it")[-1])
                if it_num % keep_every != 0:
                    f.unlink()
            except (ValueError, IndexError):
                continue
        # Alert if less than 50 GB remaining (3D checkpoints are large)
        total, used, free = shutil.disk_usage(self.run_dir)
        free_gb = free // (2 ** 30)
        if free_gb < 50:
            print(f"[DISK ALERT] Only {free_gb} GB remaining on partition!")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("LAMNr Glow 3D trainer")

    # Input data
    ap.add_argument("--view", action="append", nargs="+", required=True,
        help="Path patterns for each modality. Use one --view per modality.")
    ap.add_argument("--H", type=int, default=128, help="Target height (voxels).")
    ap.add_argument("--W", type=int, default=128, help="Target width  (voxels).")
    ap.add_argument("--D", type=int, default=128, help="Target depth  (voxels, 3D only).")
    ap.add_argument("--spatial-dims", type=int, choices=[2, 3], default=2,
        help="2 = independent-slice training, 3 = full volumetric training.")

    # Architecture
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, nargs="+", default=[32])
    ap.add_argument("--hidden", type=int, nargs="+", default=[96])
    ap.add_argument("--base", type=str, default="glow", choices=["glow", "diag"])
    ap.add_argument("--glowbase-logscale-factor", type=float, default=3.0)
    ap.add_argument("--glowbase-min-log",         type=float, default=-5.0)
    ap.add_argument("--glowbase-max-log",         type=float, default=5.0)
    ap.add_argument("--scale-map", default="tanh",
        choices=["tanh", "exp", "sigmoid", "sigmoid_inv"])
    ap.add_argument("--scale-cap",   type=float, default=2.0)
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
    ap.add_argument("--precision", default="mixed", choices=["double","float","mixed"])
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16","fp16"])
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
    ap.add_argument("--resume",          type=str,  default="")
    ap.add_argument("--auto-resume",     action="store_true")
    ap.add_argument("--out-dir",         type=str,  default="runs_glow3d")
    ap.add_argument("--use-ckpt-config", action="store_true")

    # Dataset
    ap.add_argument("--slice-idx",     type=int,   default=120)
    ap.add_argument("--val-frac",      type=float, default=0.10)
    ap.add_argument("--subject-limit", type=int,   default=0)
    ap.add_argument("--smooth-alpha",  type=float, default=0.1)

    # Alignment
    ap.add_argument("--align", default="none",
        choices=["none","infonce","barlow","vicreg","hsic","pearson","mse"])
    ap.add_argument("--align-weight",      type=float, default=0.05)
    ap.add_argument("--align-warmup",      type=int,   default=500)
    ap.add_argument("--proj-dim",          type=int,   default=256)
    ap.add_argument("--proj-hidden",       type=int,   default=512)
    ap.add_argument("--temperature",       type=float, default=0.1)
    ap.add_argument("--barlow-lambda",     type=float, default=5e-3)
    ap.add_argument("--weighting",         default="fixed", choices=["fixed","kendall"])
    ap.add_argument("--init-logvar-nll",   type=float, default=0.0)
    ap.add_argument("--init-logvar-align", type=float, default=0.0)
    ap.add_argument("--vicreg-inv",   type=float, default=25.0)
    ap.add_argument("--vicreg-cov",   type=float, default=1.0)
    ap.add_argument("--vicreg-var",   type=float, nargs="+", default=[25.0])
    ap.add_argument("--vicreg-gamma", type=float, nargs="+", default=[1.0])
    ap.add_argument("--hsic-sigma",   type=float, default=0.0)

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

    # Screening
    ap.add_argument("--screen",         default="none", choices=["none","cca","hsic"])
    ap.add_argument("--screen-warmup",  type=int,   default=1000)
    ap.add_argument("--screen-refresh", type=int,   default=0)
    ap.add_argument("--screen-frac",    type=float, default=0.5)
    ap.add_argument("--cca-ridge",      type=float, default=1e-3)
    ap.add_argument("--prefilter-frac", type=float, default=0.5)

    args = ap.parse_args()
    args.num_views = len(args.view)

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

    if args.spatial_dims == 2:
        _check_hw_divisible(args.H, args.W, args.L)
    else:
        _check_hw_divisible(args.H, args.W, args.L, D=args.D, spatial_dims=3)

    return args


def main():
    args    = _build_args()
    trainer = LAMNrGlow3DTrainer()
    trainer.setup(args)
    trainer.train()


if __name__ == "__main__":
    main()
