"""
train_lamnr_flows_tabular.py
================================
TabularLAMNrTrainer — thin subclass of BaseLAMNrTrainer for multi-view
tabular normalizing-flow whiteners (RealNVP, nn.Linear-based).

Key differences vs. the Glow 2D/3D trainers
---------------------------------------------
* Input shape is (D,) — a flat feature vector, no spatial dimensions.
  ``args.H / args.W / args.D`` are NOT required.
* Models use ``create_real_nvp_normalizing_flow_model`` (MLP coupling layers)
  instead of Glow (conv coupling layers).
* Data comes from CSV files / DataFrames via ``CSVMultiViewDataset``.
  No ANTsImage objects → no nanobind memory leaks, but gc.collect() is still
  called at iteration boundaries to release large numpy temporaries.
* ``TabularNormalizer`` (z-score or min-max) is saved inside the checkpoint
  blob under ``"dataset_normalizers"`` to guarantee inference reproducibility.
* No DataParallel: tabular flows are lightweight; ``GlowDataParallel`` and
  ``GlowStepWrapper`` are NOT used.
* All tensors passed to models are cast to ``torch.float32`` to avoid the
  classic ``RuntimeError: expected scalar type Double but found Float`` on
  GPU/MPS (pandas DataFrames often produce float64 arrays).

Usage
-----
Standalone:
    python train_lamnr_flows_tabular.py \\
        --views view0.csv view1.csv \\
        --out-dir runs_tabular --max-iter 5000 \\
        --align vicreg --norm 0mean

As a library:
    from train_lamnr_flows_tabular import TabularLAMNrTrainer
    trainer = TabularLAMNrTrainer()
    trainer.setup(args)
    trainer.train()
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import antsnormflows as nf

from antstorch.architectures.create_normalizing_flow_model import (
    create_real_nvp_normalizing_flow_model as create_rnvp,
)
from antstorch.utilities.dataframe_dataset import MultiViewDataFrameDataset

from antstorch.lamnr_flows.core.base_trainer import (
    BaseLAMNrTrainer,
    bits_per_dim,
    make_warmup,
    n_params,
    set_deterministic,
    _save_metric_plots,
)
from antstorch.lamnr_flows.latent_alignment import (
    LatentAlignmentLossManager,
    Projector,
)


# ---------------------------------------------------------------------------
# TabularNormalizer — persistent, checkpoint-safe scaler
# ---------------------------------------------------------------------------

class TabularNormalizer:
    """
    Column-wise normalizer for tabular data.

    Modes
    -----
    '0mean' : z-score  (mean=0, std=1 per column, std floor at 1e-8)
    '01'    : min-max  ([0, 1] per column, range floor at 1e-8)
    'none'  : identity (no transformation)

    All ``transform()`` outputs are ``torch.float32`` regardless of input dtype.
    This prevents ``RuntimeError: expected scalar type Double but found Float``
    on GPU/MPS when pandas produces float64 arrays.
    """

    def __init__(self, mode: str = "0mean"):
        mode = str(mode).lower()
        if mode not in ("0mean", "01", "none"):
            raise ValueError(f"TabularNormalizer mode must be '0mean', '01', or 'none'; got '{mode}'")
        self.mode    = mode
        self._mean:  Optional[np.ndarray] = None
        self._std:   Optional[np.ndarray] = None
        self._vmin:  Optional[np.ndarray] = None
        self._vrng:  Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "TabularNormalizer":
        """Fit statistics on the training split (called once, on CPU numpy)."""
        X = np.asarray(X, dtype=np.float64)
        # Impute NaN with column mean before computing stats
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_mask  = ~np.isfinite(X)
        if nan_mask.any():
            col_idx = np.where(nan_mask)[1]
            X[nan_mask] = col_means[col_idx]

        if self.mode == "0mean":
            self._mean = col_means.astype(np.float32)
            std        = X.std(axis=0)
            self._std  = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        elif self.mode == "01":
            vmin       = X.min(axis=0)
            vmax       = X.max(axis=0)
            rng        = vmax - vmin
            self._vmin = vmin.astype(np.float32)
            self._vrng = np.where(rng > 1e-8, rng, 1.0).astype(np.float32)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> torch.Tensor:
        """
        Apply the fitted normalization.

        Returns a ``torch.float32`` tensor — always, regardless of input dtype.
        """
        X = np.asarray(X, dtype=np.float64)
        # Impute NaN
        if not np.all(np.isfinite(X)):
            col_means = (
                self._mean.astype(np.float64)
                if self._mean is not None
                else np.zeros(X.shape[1])
            )
            nan_mask = ~np.isfinite(X)
            X[nan_mask] = col_means[np.where(nan_mask)[1]]

        if self.mode == "0mean" and self._fitted:
            X = (X - self._mean) / self._std
        elif self.mode == "01" and self._fitted:
            X = (X - self._vmin) / self._vrng
            X = np.clip(X, 0.0, 1.0)

        # Explicit float32 cast — critical for GPU/MPS correctness
        return torch.from_numpy(X.astype(np.float32, copy=False))

    def inverse_transform(self, t: torch.Tensor) -> torch.Tensor:
        """Invert the normalization (for reconstruction export)."""
        X = t.detach().cpu().float().numpy()
        if self.mode == "0mean" and self._fitted:
            X = X * self._std + self._mean
        elif self.mode == "01" and self._fitted:
            X = X * self._vrng + self._vmin
        return torch.from_numpy(X.astype(np.float32))

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mode":    self.mode,
            "mean":    self._mean.tolist()  if self._mean  is not None else None,
            "std":     self._std.tolist()   if self._std   is not None else None,
            "vmin":    self._vmin.tolist()  if self._vmin  is not None else None,
            "vrng":    self._vrng.tolist()  if self._vrng  is not None else None,
            "fitted":  self._fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mode    = d["mode"]
        self._mean   = np.array(d["mean"],  dtype=np.float32) if d["mean"]  else None
        self._std    = np.array(d["std"],   dtype=np.float32) if d["std"]   else None
        self._vmin   = np.array(d["vmin"],  dtype=np.float32) if d["vmin"]  else None
        self._vrng   = np.array(d["vrng"],  dtype=np.float32) if d["vrng"]  else None
        self._fitted = bool(d.get("fitted", False))

    def __repr__(self) -> str:
        return (
            f"TabularNormalizer(mode='{self.mode}', "
            f"fitted={self._fitted}, "
            f"D={len(self._mean) if self._mean is not None else 'N/A'})"
        )


# ---------------------------------------------------------------------------
# CSVMultiViewDataset — PyTorch Dataset for tabular multi-view data
# ---------------------------------------------------------------------------

class CSVMultiViewDataset(Dataset):
    """
    Multi-view tabular dataset backed by CSV files or pre-loaded DataFrames.

    Returns a list of per-view ``torch.float32`` tensors (one per view).
    The ``TabularNormalizer`` is fit on the data passed to the constructor
    and applied in ``__getitem__``.

    Parameters
    ----------
    dfs       : list of pd.DataFrame — one per view, same row count.
    normalizers : list of TabularNormalizer — already fitted on train split.
    noise_std : float — additive Gaussian jitter in normalized space (0 = off).
    """

    def __init__(
        self,
        dfs:         List[pd.DataFrame],
        normalizers: List[TabularNormalizer],
        noise_std:   float = 0.0,
    ):
        if len(dfs) != len(normalizers):
            raise ValueError("dfs and normalizers must have the same length.")
        n_rows = {len(df) for df in dfs}
        if len(n_rows) != 1:
            raise ValueError(f"All views must have equal row counts; got {n_rows}.")

        # Store raw numpy arrays to avoid DataFrame overhead in __getitem__
        self._arrays = [df.to_numpy(dtype=np.float64) for df in dfs]
        self.normalizers = normalizers
        self.noise_std   = float(noise_std)
        self._n          = len(dfs[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        out = []
        for arr, norm in zip(self._arrays, self.normalizers):
            row = arr[idx : idx + 1]      # shape (1, D) — 2-D for transform()
            t   = norm.transform(row)     # always float32
            t   = t.squeeze(0)            # → (D,)
            if self.noise_std > 0.0:
                t = t + torch.randn_like(t) * self.noise_std
            out.append(t)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_views(view_paths: List[str]) -> List[pd.DataFrame]:
    """Load CSV views, coercing all columns to numeric (non-numeric → NaN)."""
    dfs = []
    for p in view_paths:
        df = pd.read_csv(p)
        df = df.apply(pd.to_numeric, errors="coerce")
        dfs.append(df)
    n_rows = {len(df) for df in dfs}
    if len(n_rows) != 1:
        raise ValueError(f"All CSV views must have the same number of rows; got {n_rows}")
    return dfs


def _build_base_distribution(
    D: int,
    base: str = "GaussianPCA",
    pca_latent_dim: int = 4,
    base_min_log: float = -5.0,
    base_max_log: float = 5.0,
    base_sigma: float = 0.1,
) -> nn.Module:
    b = base.lower()
    if b in ("gaussianpca", "pca"):
        return nf.distributions.GaussianPCA(
            D, latent_dim=int(pca_latent_dim), sigma=base_sigma
        )
    elif b in ("diag", "diaggaussian"):
        try:
            return nf.distributions.DiagGaussian(
                D, trainable=True, min_log=base_min_log, max_log=base_max_log
            )
        except TypeError:
            return nf.distributions.DiagGaussian(D, trainable=True)
    else:
        raise ValueError(f"Unknown base distribution: '{base}'")


def _inverse_with_guard(
    model: nn.Module, xb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model.inverse() handling various output shapes."""
    out = model.inverse(xb)
    if isinstance(out, (list, tuple)):
        z, log_det = out[0], out[1]
    else:
        z       = out
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
    return z, log_det


def _extract_whitened(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Extract whitened / PCA-projected latents from base distribution."""
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        W, loc = q0.W, q0.loc
        if isinstance(loc, torch.Tensor) and loc.dim() == 2:
            loc = loc.squeeze(0)
        return torch.matmul(z - loc, W.T).to(z.dtype)
    loc   = getattr(q0, "loc",   None)
    scale = getattr(q0, "scale", None)
    if isinstance(loc, torch.Tensor) and isinstance(scale, torch.Tensor):
        if loc.dim()   == 2: loc   = loc.squeeze(0)
        if scale.dim() == 2: scale = scale.squeeze(0)
        return ((z - loc) / (scale + 1e-12)).to(z.dtype)
    return z.to(z.dtype)


# ---------------------------------------------------------------------------
# TabularLAMNrTrainer
# ---------------------------------------------------------------------------

class TabularLAMNrTrainer(BaseLAMNrTrainer):
    """
    Tabular flow trainer as a subclass of BaseLAMNrTrainer.

    Overrides
    ---------
    build_models   → create_real_nvp_normalizing_flow_model (MLP coupling layers)
    build_loaders  → CSVMultiViewDataset backed DataLoaders (no ANTsImage)
    extract_view   → returns tensor[vi] from list-batch, cast to float32
    save_checkpoint → adds "dataset_normalizers" key to checkpoint blob
    train()        → tabular-specific loop (NLL + alignment, no spatial dims)

    float32 safety
    --------------
    Both CSVMultiViewDataset.normalizers.transform() and extract_view() guarantee
    float32 output.  The explicit .to(dtype=torch.float32) call in extract_view()
    is a belt-and-suspenders guard against upstream dtype surprises.

    Spatial dimensions
    ------------------
    args.H / args.W / args.D are NOT used and are never required.
    input_shape = (D_view,) — a flat vector of feature dimension D_view.
    BaseLAMNrTrainer._check_hw_divisible() is NOT called.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def build_models(self, args) -> List[nn.Module]:
        dev     = self.dev
        models  = []
        for vi, D in enumerate(self._view_dims):
            q0 = _build_base_distribution(
                D               = D,
                base            = args.base_distribution,
                pca_latent_dim  = args.pca_latent_dimension,
                base_min_log    = args.base_min_log,
                base_max_log    = args.base_max_log,
                base_sigma      = args.base_sigma,
            )
            m = create_rnvp(
                latent_size              = D,
                K                        = args.K,
                q0                       = q0,
                leaky_relu_negative_slope= args.leaky_relu_negative_slope,
                mlp_width                = args.hidden_channels,
                scale_cap                = args.scale_cap,
                spectral_norm_scales     = args.spectral_norm_scales,
                additive_first_n         = args.additive_first_n,
                actnorm_every            = args.actnorm_every,
                mask_mode                = args.mask_mode,
            ).to(device=dev, dtype=torch.float32)
            m.train()
            print(
                f"[init] view {vi}: D={D}, params={n_params(m):,}, "
                f"base={args.base_distribution}"
            )
            models.append(m)
        return models

    def build_loaders(self, args):
        """
        Load CSV views → fit TabularNormalizer on train split →
        build CSVMultiViewDataset → DataLoader.

        Sets self._view_dims and self.normalizers for use by build_models()
        and save_checkpoint().
        """
        rng   = np.random.default_rng(args.seed)
        dfs   = _load_views(args.views)
        N     = len(dfs[0])
        idx   = np.arange(N)
        rng.shuffle(idx)
        n_val = min(max(0, int(round(float(args.val_fraction) * N))), N - 1)
        val_idx   = set(idx[:n_val].tolist())
        train_idx = [i for i in idx if i not in val_idx]
        val_idx   = list(val_idx)

        dfs_train = [df.iloc[train_idx].reset_index(drop=True) for df in dfs]
        dfs_val   = [df.iloc[val_idx].reset_index(drop=True)   for df in dfs]

        # Fit one normalizer per view (on train split only)
        self.normalizers: List[TabularNormalizer] = []
        for df_tr in dfs_train:
            n = TabularNormalizer(mode=args.normalization)
            n.fit(df_tr.to_numpy(dtype=np.float64))
            self.normalizers.append(n)

        self._view_dims: List[int] = [df.shape[1] for df in dfs_train]

        # Store as args.input_shape — flat tuple, no spatial dims
        args.input_shape = tuple(self._view_dims)
        args.num_views   = len(dfs)

        train_ds = CSVMultiViewDataset(
            dfs_train, self.normalizers,
            noise_std=float(getattr(args, "jitter_alpha", 0.0)),
        )
        val_ds = CSVMultiViewDataset(
            dfs_val if dfs_val[0].shape[0] > 0 else dfs_train,
            self.normalizers,
            noise_std=0.0,
        )

        global_step = Value("i", 0)

        if torch.cuda.is_available():
            pin = True
        elif torch.backends.mps.is_available():
            pin = False
        else:
            pin = False

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=pin, collate_fn=_tabular_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(2048, max(args.batch_size, 256)),
            shuffle=False,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=pin, collate_fn=_tabular_collate,
        )

        print(
            f"[data] {len(dfs)} views | "
            f"train={len(train_ds)} val={len(val_ds)} | "
            f"dims={self._view_dims}"
        )
        return train_loader, val_loader, global_step

    def extract_view(
        self, batch: object, vi: int, dev: torch.device
    ) -> torch.Tensor:
        """
        Extract view vi from a list-batch and guarantee float32.

        CSVMultiViewDataset already returns float32, but the explicit cast is
        a safety net against any upstream dtype change (e.g. collate_fn).
        """
        if isinstance(batch, (list, tuple)):
            return batch[vi].to(device=dev, dtype=torch.float32)
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    # ------------------------------------------------------------------
    # Checkpoint — add dataset_normalizers key
    # ------------------------------------------------------------------

    def save_checkpoint(self, it: int) -> None:
        """
        Override: inject 'dataset_normalizers' into the blob so that
        inference code can reproduce the exact same normalization as training.
        """
        blob = {
            "iter":    it + 1,
            "opt":     self.opt.state_dict(),
            "warm":    (self.warm.state_dict() if self.warm else None),
            "models":  [self._strip_dp_prefix(m.state_dict()) for m in self.models],
            "ema":     (
                [self._strip_dp_prefix(em.state_dict()) for em in self.ema_models]
                if self.ema_models else None
            ),
            "proj":    (self.projectors.state_dict() if self.projectors else None),
            "kendall": {
                "s_nll":   float(self.s_nll.detach().cpu())   if self.s_nll   else None,
                "s_align": float(self.s_align.detach().cpu()) if self.s_align else None,
            },
            "config":  vars(self.args),
            "scaler":  (
                self.scaler.state_dict()
                if self.scaler is not None and self.scaler.is_enabled()
                else None
            ),
            # ── Tabular-specific ──────────────────────────────────────
            "dataset_normalizers": [n.state_dict() for n in self.normalizers],
            "view_dims":           list(self._view_dims),
        }
        torch.save(blob, self.state_path)
        iter_path = self.run_dir / f"training_state_it{it:06d}.pt"
        torch.save(blob, iter_path)
        self.cleanup_checkpoints()
        tqdm.write(f"[ckpt] saved {iter_path.name} (normalizers included)")

    def _maybe_resume(self, args) -> int:
        """Extend parent resume to also reload TabularNormalizer stats."""
        start_iter = super()._maybe_resume(args)

        # Reload normalizer stats if present in the checkpoint
        resume_path = None
        if args.resume:
            resume_path = Path(args.resume)
        elif args.auto_resume and self.state_path.exists():
            resume_path = self.state_path

        if resume_path is not None and resume_path.exists():
            try:
                blob = torch.load(resume_path, map_location="cpu", weights_only=False)
                if "dataset_normalizers" in blob and hasattr(self, "normalizers"):
                    for norm_obj, d in zip(self.normalizers, blob["dataset_normalizers"]):
                        norm_obj.load_state_dict(d)
                    tqdm.write("[resume] TabularNormalizer stats restored from checkpoint.")
                if "view_dims" in blob:
                    self._view_dims = list(blob["view_dims"])
            except Exception as e:
                tqdm.write(f"[resume] Could not restore normalizers: {e}")

        return start_iter

    @torch.no_grad()
    def export_tabular_results(self):
        """
        Calcule et exporte z, les données blanchies et les reconstructions 
        au format CSV pour chaque vue à la fin de l'entraînement.
        """
        args = self.args
        if not (args.save_z or args.save_whitened or args.save_recon):
            return

        print("\n📊 Initialisation de l'exportation des résultats tabulaires...")
        for model in self.models:
            model.eval()

        # Dictionnaires pour accumuler les données à travers les batchs
        accum_z = {vi: [] for vi in range(args.num_views)}
        accum_whitened = {vi: [] for vi in range(args.num_views)}
        accum_recon = {vi: [] for vi in range(args.num_views)}

        # Passer sur l'ensemble du train_loader pour collecter les transformations
        for batch in self.train_loader:
            # v2 utilise _extract_views_from_batch pour séparer les vues tabulaires
            xs = [batch[vi].to(device=self.dev, dtype=torch.float32) for vi in range(args.num_views)]
            
            for vi in range(args.num_views):
                model = self.models[vi]
                x_v = xs[vi]

                # 1. Calcul de z (Latent brut) via la passe inverse du flux
                z, _ = model.inverse_and_log_det(x_v)
                if args.save_z:
                    accum_z[vi].append(z.detach().cpu().numpy())

                # 2. Calcul des latents blanchis (standardisés)
                if args.save_whitened:
                    # Récupération du normaliseur sauvegardé dans le v2 pour cette vue
                    if hasattr(self, "dataset_normalizers") and self.dataset_normalizers[vi] is not None:
                        # Si votre normaliseur a une fonction spécifique ou VICReg
                        z_norm = self.dataset_normalizers[vi].normalize(z)
                    else:
                        z_norm = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
                    accum_whitened[vi].append(z_norm.detach().cpu().numpy())

                # 3. Calcul de la reconstruction (Retour vers l'espace observé)
                if args.save_recon:
                    x_rec, _ = model.forward_and_log_det(z)
                    accum_recon[vi].append(x_rec.detach().cpu().numpy())

        # Sauvegarde effective sur le disque dans le dossier de sortie
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for vi in range(args.num_views):
            view_name = args.view[vi] if hasattr(args, "view") else f"view_{vi}"
            
            if args.save_z and len(accum_z[vi]) > 0:
                full_z = np.concatenate(accum_z[vi], axis=0)
                df_z = pd.DataFrame(full_z)
                path_z = out_dir / f"{view_name}_latent_z.csv"
                df_z.to_csv(path_z, index=False)
                print(f"  [Export] Coordonnées z sauvées : {path_z}")

            if args.save_whitened and len(accum_whitened[vi]) > 0:
                full_w = np.concatenate(accum_whitened[vi], axis=0)
                df_w = pd.DataFrame(full_w)
                path_w = out_dir / f"{view_name}_whitened_epsilon.csv"
                df_w.to_csv(path_w, index=False)
                print(f"  [Export] Données blanchies sauvées : {path_w}")

            if args.save_recon and len(accum_recon[vi]) > 0:
                full_r = np.concatenate(accum_recon[vi], axis=0)
                df_r = pd.DataFrame(full_r)
                path_r = out_dir / f"{view_name}_reconstructed_x.csv"
                df_r.to_csv(path_r, index=False)
                print(f"  [Export] Reconstructions sauvées : {path_r}")

    # ------------------------------------------------------------------
    # Tabular training loop
    # ------------------------------------------------------------------

    def train(self) -> None:  # noqa: C901
        """
        Tabular training loop.

        Differences from BaseLAMNrTrainer.train():
        - Input tensors are flat (D,) vectors — no spatial pooling.
        - NLL computed via model.log_prob(x) where x is (B, D) float32.
        - Latent z extracted via model.inverse() for alignment manager.
        - No EMA ActNorm warmup (RealNVP actnorm is data-independent).
        - No input grid saving (no image output).
        - gc.collect() at every iteration to free large numpy temporaries.
        """
        args     = self.args
        dev      = self.dev
        models   = self.models
        n_views  = len(models)
        alpha    = float(args.smooth_alpha)

        ema_loss_disp    = None
        ema_bpd_disp: List[Optional[float]] = [None] * n_views

        train_iter = iter(self.train_loader)

        pbar = tqdm(
            total=args.max_iter,
            initial=self.start_iter - 1,
            dynamic_ncols=True,
            desc="train-tab",
        )

        for it in range(self.start_iter, args.max_iter + 1):
            grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
            self.opt.zero_grad(set_to_none=True)

            loss_acc  = torch.tensor(0.0, device=dev, dtype=torch.float32)
            align_acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
            bpd_acc:   List[float] = [0.0] * n_views
            bad_update = False
            x_last     = None
            w_nll, w_align = 1.0, 0.0

            # ── Gradient accumulation micro-steps ──────────────────────
            for _micro in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                    gc.collect()

                x_last = batch

                L_nll      = torch.tensor(0.0, device=dev, dtype=torch.float32)
                lat_flat:  List[torch.Tensor] = []
                bad_batch  = False

                for vi, m in enumerate(models):
                    # ── Strict float32 cast — GPU/MPS safety ─────────
                    x_v = self.extract_view(batch, vi, dev)   # (B, D) float32

                    logp_v = m.log_prob(x_v)                  # (B,)
                    if not torch.isfinite(logp_v).all():
                        tqdm.write(f"[nan] non-finite logp view {vi} @ iter {it}")
                        bad_batch = True
                        del x_v, logp_v
                        gc.collect()
                        break

                    # NLL as bits-per-dim (D is the feature dimension)
                    D     = float(x_v.shape[1])
                    bpd_v = (-logp_v / (math.log(2.0) * D)).mean()
                    L_nll = L_nll + bpd_v
                    bpd_acc[vi] += bpd_v.item()

                    # Latent for alignment
                    z_v, _ = _inverse_with_guard(m, x_v)
                    # Tabular: z is already flat (B, D) — no pooling needed
                    lat_flat.append(torch.nan_to_num(z_v.float()))

                    del x_v, z_v, logp_v

                if bad_batch or not torch.isfinite(L_nll) or abs(L_nll.item()) > 1e7:
                    tqdm.write(f"[anomaly] skipping iter {it}")
                    bad_update = True
                    del lat_flat
                    gc.collect()
                    break

                # Alignment loss
                loss_total, L_align, w_nll, w_align = self.align_mgr.compute(
                    lat_flat=lat_flat,
                    L_nll=L_nll,
                    it=it,
                    s_nll=self.s_nll,
                    s_align=self.s_align,
                )

                if not torch.isfinite(loss_total):
                    tqdm.write(f"[nan] loss_total non-finite @ iter {it}")
                    bad_update = True
                    del lat_flat
                    gc.collect()
                    break

                (loss_total / float(grad_accum)).backward()

                loss_acc  = loss_acc  + loss_total.detach().float()
                align_acc = align_acc + L_align.detach().float()

                # ── Strict per-micro-step cleanup ─────────────────────
                del lat_flat
                gc.collect()
                # ─────────────────────────────────────────────────────

            # ── End grad-accum ─────────────────────────────────────────

            if bad_update:
                self.opt.zero_grad(set_to_none=True)
                continue

            # Gradient clip + step
            all_params = [p for g in self.opt.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=float(getattr(args, "grad_clip", 5.0))
            )
            self.opt.step()

            # LR warmup
            if self.warm is not None and it <= args.warmup_iters:
                self.warm.step()

            with self.global_step.get_lock():
                self.global_step.value += 1

            lr_now = self.opt.param_groups[0]["lr"]

            # Averaged metrics
            curr_loss  = float(loss_acc.item()) / float(grad_accum)
            L_align_log= float(align_acc.item()) / float(grad_accum)
            curr_bpd   = [b / float(grad_accum) for b in bpd_acc]

            # EMA display
            if ema_loss_disp is None:
                ema_loss_disp = curr_loss
                ema_bpd_disp  = list(curr_bpd)
            else:
                a = alpha
                ema_loss_disp = (1.0 - a) * ema_loss_disp + a * curr_loss
                for i in range(n_views):
                    ema_bpd_disp[i] = (1.0 - a) * ema_bpd_disp[i] + a * curr_bpd[i]

            postfix = {
                "loss":  f"{curr_loss:.4f}",
                "loss~": f"{ema_loss_disp:.4f}",
                "align": f"{L_align_log:.4f}",
                "mode":  args.align,
                "lr":    f"{lr_now:.2e}",
            }
            for i in range(n_views):
                postfix[f"bpd{i}"] = f"{curr_bpd[i]:.3f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

            # CSV row
            with open(self.csv_path, "a") as f:
                f.write(f"{it},{curr_loss:.6f},{sum(curr_bpd):.6f},{lr_now:.6g}\n")

            # Eval + checkpoint
            if it % args.eval_interval == 0:
                self._run_val_tabular(it)
                _save_metric_plots(self.csv_path, self.run_dir, remove_spikes=True)
                self.save_checkpoint(it)

            # ── End-of-iteration cleanup ──────────────────────────────
            del x_last, batch
            gc.collect()
            # ─────────────────────────────────────────────────────────

        pbar.close()
        print("Done. Run dir:", str(self.run_dir))

    def _run_val_tabular(self, it: int) -> None:
        """Run one validation pass and update the LR plateau scheduler."""
        dev    = self.dev
        models = self.models
        bpd_vals: List[float] = []
        with torch.no_grad():
            for j, batch_val in enumerate(self.val_loader):
                for vi, m in enumerate(models):
                    x_v   = self.extract_view(batch_val, vi, dev)  # float32
                    logp  = m.log_prob(x_v)
                    logp  = torch.nan_to_num(logp, nan=-1e9, posinf=-1e9, neginf=-1e9)
                    D     = float(x_v.shape[1])
                    bpd   = (-logp / (math.log(2.0) * D)).mean().item()
                    bpd_vals.append(bpd)
                    del x_v
                if j >= 9:
                    break
        avg_bpd = float(np.mean(bpd_vals)) if bpd_vals else float("nan")
        self.plateau.step(avg_bpd)
        lr_now = self.opt.param_groups[0]["lr"]
        tqdm.write(f"[eval] iter={it} avg_bpd={avg_bpd:.4f} lr={lr_now:.2e}")

    # ------------------------------------------------------------------
    # Override setup() to skip image-specific initialization
    # ------------------------------------------------------------------

    def setup(self, args) -> None:
        """
        Tabular-safe setup.

        - Calls build_loaders() first (to populate self._view_dims).
        - Skips _check_hw_divisible() (no spatial dims).
        - Skips ActNorm warmup with dummy images.
        - Projector dim = view feature dim D (no flatten_latents pooling needed).
        """
        import platform, datetime
        from pathlib import Path

        self.args = args
        set_deterministic(args.seed)

        # Device
        if args.devices.lower() == "cpu":
            dev = torch.device("cpu")
        elif args.devices == "mps" and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device(args.devices.split(",")[0])
        self.dev = dev

        # AMP — tabular flows are small, mixed precision rarely helps but is
        # supported for consistency with the base class.
        self.model_dtype = torch.float32
        self.amp_enabled = False
        self.amp_dtype   = None
        self.scaler = torch.amp.GradScaler(enabled=False)

        # Data loaders (also sets self._view_dims + self.normalizers)
        self.train_loader, self.val_loader, self.global_step = self.build_loaders(args)

        # Models
        self.models: List[nn.Module] = self.build_models(args)
        self.ema_models = None

        # Projectors (for alignment; one per view; dim = feature dim D_v)
        self.projectors: Optional[nn.ModuleList] = None
        if args.align != "none":
            self.projectors = nn.ModuleList([
                Projector(D, args.proj_hidden, args.proj_dim)
                .to(dtype=torch.float32, device=dev)
                .train()
                for D in self._view_dims
            ])

        self.align_mgr = LatentAlignmentLossManager(
            args=args,
            projectors=self.projectors,
            device=dev,
        )

        # Kendall scalars
        self.s_nll = self.s_align = None
        if args.weighting == "kendall" and args.align != "none":
            self.s_nll   = nn.Parameter(
                torch.tensor([0.0], device=dev, dtype=torch.float32)
            )
            self.s_align = nn.Parameter(
                torch.tensor([0.0], device=dev, dtype=torch.float32)
            )

        # Optimizer
        param_groups = [{"params": [p for m in self.models for p in m.parameters()]}]
        if self.projectors:
            param_groups.append({"params": list(self.projectors.parameters())})
        if self.s_nll is not None:
            param_groups.append(
                {"params": [self.s_nll, self.s_align], "weight_decay": 0.0}
            )
        self.opt = torch.optim.AdamW(
            param_groups, lr=args.lr,
            betas=getattr(args, "betas", (0.9, 0.98)),
            weight_decay=args.weight_decay,
        )
        self.warm = make_warmup(
            self.opt, args.warmup_iters,
            getattr(args, "lr_decay_gamma", 1.0),
            getattr(args, "lr_decay_steps", 0),
        )
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min",
            factor=getattr(args, "plateau_factor", 0.5),
            patience=getattr(args, "plateau_patience", 5),
            threshold=getattr(args, "plateau_threshold", 1e-4),
            min_lr=getattr(args, "min_lr", 1e-6),
        )

        # Paths
        self.run_dir    = Path(args.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "training_state.pt"
        self.csv_path   = self.run_dir / "metrics.csv"

        # Resume
        self.start_iter = self._maybe_resume(args)
        if args.extra_iters > 0:
            args.max_iter = (self.start_iter - 1) + args.extra_iters

        with self.global_step.get_lock():
            self.global_step.value = int(self.start_iter)

        # CSV header
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                f.write("iter,loss,sum_bpd,lr\n")

        # Screen dump
        print(f"\n[tabular trainer] {len(self.models)} view(s), dims={self._view_dims}")
        print(f"  norm={args.normalization}  align={args.align}  "
              f"batch={args.batch_size}  max_iter={args.max_iter}")
        print(f"  out_dir={self.run_dir}")
        for vi, m in enumerate(self.models):
            print(f"  view {vi}: D={self._view_dims[vi]}, params={n_params(m):,}")

    # ------------------------------------------------------------------
    # Export: latents z, whitened ε, reconstructions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def export(self) -> None:
        """
        Post-training export of per-view latents and/or reconstructions.

        Controlled by args flags:
          --save-z         : raw flow latents z  → {out_dir}/z_view{i}.csv
          --save-whitened  : PCA/whitened coords → {out_dir}/whitened_view{i}.csv
                             (only meaningful with base_distribution='GaussianPCA')
          --save-recon     : inverse-flow recon  → {out_dir}/recon_view{i}.csv
                             de-normalized via TabularNormalizer (observed scale)

        All outputs are produced from the full dataset (train + val concatenated)
        in inference mode (no dropout, no jitter).
        Uses EMA models when available, otherwise base models.
        """
        args = self.args
        do_z      = getattr(args, "save_z",       False)
        do_wh     = getattr(args, "save_whitened", False)
        do_recon  = getattr(args, "save_recon",   False)

        if not (do_z or do_wh or do_recon):
            return

        print("[export] Running post-training export...")
        dev     = self.dev
        models  = self.ema_models if self.ema_models else self.models
        n_views = len(models)

        for m in models:
            m.eval()

        # Build a full (train+val) inference dataset — no jitter, no shuffle
        # We reload the original CSVs so no rows are excluded.
        raw_dfs = _load_views(args.views)
        infer_ds = CSVMultiViewDataset(
            raw_dfs, self.normalizers, noise_std=0.0
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=min(2048, max(args.batch_size, 256)),
            shuffle=False,
            num_workers=int(getattr(args, "num_workers", 0)),
            collate_fn=_tabular_collate,
        )

        # Accumulators
        z_acc:    List[List[np.ndarray]] = [[] for _ in range(n_views)]
        wh_acc:   List[List[np.ndarray]] = [[] for _ in range(n_views)]
        rec_acc:  List[List[np.ndarray]] = [[] for _ in range(n_views)]

        for batch in tqdm(infer_loader, desc="export", leave=False):
            for vi, m in enumerate(models):
                x_v = self.extract_view(batch, vi, dev)   # (B, D) float32

                # Forward: x → z
                z_v, _ = _inverse_with_guard(m, x_v)      # (B, D)

                if do_z:
                    z_acc[vi].append(z_v.cpu().float().numpy())

                if do_wh:
                    wh = _extract_whitened(m, z_v)        # (B, L) or (B, D)
                    wh_acc[vi].append(wh.cpu().float().numpy())

                if do_recon:
                    # Inverse flow: z → x_hat (normalized space)
                    try:
                        out = m.forward(z_v)
                        x_hat = out[0] if isinstance(out, (list, tuple)) else out
                    except Exception:
                        # fallback: use q0.sample() if forward() not available
                        x_hat = z_v

                    # De-normalize: bring back to observed scale
                    x_hat_np = x_hat.cpu().float().numpy()
                    x_hat_denorm = self.normalizers[vi].inverse_transform(
                        torch.from_numpy(x_hat_np)
                    ).numpy()
                    rec_acc[vi].append(x_hat_denorm)

                del x_v, z_v

            gc.collect()

        # Write CSVs
        out_dir = self.run_dir
        for vi in range(n_views):
            if do_z and z_acc[vi]:
                arr = np.concatenate(z_acc[vi], axis=0)
                cols = [f"z{j}" for j in range(arr.shape[1])]
                pd.DataFrame(arr, columns=cols).to_csv(
                    out_dir / f"z_view{vi}.csv", index=False
                )
                print(f"[export] z_view{vi}.csv  shape={arr.shape}")

            if do_wh and wh_acc[vi]:
                arr = np.concatenate(wh_acc[vi], axis=0)
                cols = [f"eps{j}" for j in range(arr.shape[1])]
                pd.DataFrame(arr, columns=cols).to_csv(
                    out_dir / f"whitened_view{vi}.csv", index=False
                )
                print(f"[export] whitened_view{vi}.csv  shape={arr.shape}")

            if do_recon and rec_acc[vi]:
                arr = np.concatenate(rec_acc[vi], axis=0)
                # Reuse original column names from the input CSV
                orig_cols = list(raw_dfs[vi].columns)
                if len(orig_cols) == arr.shape[1]:
                    cols = orig_cols
                else:
                    cols = [f"feat{j}" for j in range(arr.shape[1])]
                pd.DataFrame(arr, columns=cols).to_csv(
                    out_dir / f"recon_view{vi}.csv", index=False
                )
                print(f"[export] recon_view{vi}.csv  shape={arr.shape}")

        print(f"[export] Done. Files written to: {out_dir}")

        # Restore training mode
        for m in self.models:
            m.train()



# ---------------------------------------------------------------------------
# Collate function — ensures batch is a list of stacked view tensors
# ---------------------------------------------------------------------------

def _tabular_collate(samples: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Each sample is a list of per-view float32 tensors of shape (D_v,).
    Returns a list of per-view batched tensors of shape (B, D_v).
    """
    n_views = len(samples[0])
    return [
        torch.stack([s[vi] for s in samples], dim=0)
        for vi in range(n_views)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("LAMNr tabular flow trainer (BaseLAMNrTrainer subclass)")

    # Input
    ap.add_argument("--views", nargs="+", required=True,
        help="CSV file paths, one per view.")
    ap.add_argument("--out-dir", type=str, default="runs_tabular",
        help="Output directory.")

    # Normalization
    ap.add_argument("--normalization", default="0mean", choices=["0mean", "01", "none"],
        help="Per-view feature normalization (z-score | min-max | none).")
    ap.add_argument("--jitter-alpha", type=float, default=0.0,
        help="Additive Gaussian noise std in normalized space (0 = off).")

    # Architecture
    ap.add_argument("--base-distribution", default="GaussianPCA",
        choices=["GaussianPCA", "DiagGaussian"])
    ap.add_argument("--pca-latent-dimension", type=int, default=4)
    ap.add_argument("--base-min-log", type=float, default=-5.0)
    ap.add_argument("--base-max-log", type=float, default=5.0)
    ap.add_argument("--base-sigma",   type=float, default=0.1)
    ap.add_argument("--K",             type=int,   default=64)
    ap.add_argument("--hidden-channels", type=int, default=None)
    ap.add_argument("--leaky-relu-negative-slope", type=float, default=0.0)
    ap.add_argument("--scale-cap",         type=float, default=3.0)
    ap.add_argument("--spectral-norm-scales", action="store_true")
    ap.add_argument("--additive-first-n",  type=int, default=0)
    ap.add_argument("--actnorm-every",     type=int, default=1)
    ap.add_argument("--mask-mode", default="alternating",
        choices=["alternating", "rolling"])

    # Training loop
    ap.add_argument("--batch-size",    type=int,   default=256,  dest="batch_size")
    ap.add_argument("--val-fraction",  type=float, default=0.20, dest="val_fraction")
    ap.add_argument("--max-iter",      type=int,   default=5000)
    ap.add_argument("--extra-iters",   type=int,   default=0)
    ap.add_argument("--eval-interval", type=int,   default=500)
    ap.add_argument("--num-workers",   type=int,   default=0)

    # Hardware
    ap.add_argument("--devices", type=str, default="cpu",
        help="Device string, e.g. 'cpu', 'cuda:0', 'mps'.")
    ap.add_argument("--seed",    type=int, default=0)

    # Optimizer & scheduler
    ap.add_argument("--lr",               type=float, default=2e-4)
    ap.add_argument("--weight-decay",     type=float, default=0.0)
    ap.add_argument("--warmup-iters",     type=int,   default=200)
    ap.add_argument("--lr-decay-gamma",   type=float, default=1.0)
    ap.add_argument("--lr-decay-steps",   type=int,   default=0)
    ap.add_argument("--plateau-factor",   type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int,   default=5)
    ap.add_argument("--plateau-threshold",type=float, default=1e-4)
    ap.add_argument("--min-lr",           type=float, default=1e-6)

    # Gradients & accum
    ap.add_argument("--grad-clip",  type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int,   default=1)

    # EMA (optional — off by default for tabular)
    ap.add_argument("--ema",       action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9995)

    # Checkpointing
    ap.add_argument("--resume",          type=str,  default="")
    ap.add_argument("--auto-resume",     action="store_true")
    ap.add_argument("--use-ckpt-config", action="store_true")

    # EMA display smoothing
    ap.add_argument("--smooth-alpha", type=float, default=0.1)

    # Alignment
    ap.add_argument("--align", default="none",
        choices=["none", "infonce", "barlow", "vicreg", "hsic", "pearson", "mse"])
    ap.add_argument("--align-weight",      type=float, default=0.05)
    ap.add_argument("--align-warmup",      type=int,   default=200)
    ap.add_argument("--proj-dim",          type=int,   default=64)
    ap.add_argument("--proj-hidden",       type=int,   default=128)
    ap.add_argument("--temperature",       type=float, default=0.1)
    ap.add_argument("--barlow-lambda",     type=float, default=5e-3)
    ap.add_argument("--weighting",         default="fixed", choices=["fixed", "kendall"])
    ap.add_argument("--init-logvar-nll",   type=float, default=0.0)
    ap.add_argument("--init-logvar-align", type=float, default=0.0)
    ap.add_argument("--vicreg-inv",   type=float, default=25.0)
    ap.add_argument("--vicreg-cov",   type=float, default=1.0)
    ap.add_argument("--vicreg-var",   type=float, nargs="+", default=[25.0])
    ap.add_argument("--vicreg-gamma", type=float, nargs="+", default=[1.0])
    ap.add_argument("--hsic-sigma",   type=float, default=0.0)

    # Screening (CCA / HSIC subspace selection)
    ap.add_argument("--screen",         default="none", choices=["none", "cca", "hsic"])
    ap.add_argument("--screen-warmup",  type=int,   default=500)
    ap.add_argument("--screen-refresh", type=int,   default=0)
    ap.add_argument("--screen-frac",    type=float, default=0.5)
    ap.add_argument("--cca-ridge",      type=float, default=1e-3)
    ap.add_argument("--prefilter-frac", type=float, default=0.5)

    ap.add_argument("--save-z", action="store_true", 
                    help="Exporter les coordonnées latentes brutes z (un CSV par vue)")
    ap.add_argument("--save-whitened", action="store_true", 
                    help="Exporter les coordonnées latentes standardisées/blanchies (un CSV par vue)")
    ap.add_argument("--save-recon", action="store_true", 
                    help="Exporter les reconstructions inversées dans l'échelle observée")

    # Image-specific args set to None (unused but needed by BaseLAMNrTrainer)
    ap.set_defaults(H=None, W=None, D=None, sample_mode="off",
                    sample_temp=1.0, sample_grid_norm="to01")

    args = ap.parse_args()
    args.num_views = len(args.views)  # updated in build_loaders anyway

    return args


def main():
    args    = _build_args()
    trainer = TabularLAMNrTrainer()
    trainer.setup(args)
    trainer.train()
    trainer.export()


if __name__ == "__main__":
    main()
