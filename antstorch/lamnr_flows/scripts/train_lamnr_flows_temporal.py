#!/usr/bin/env python3
"""LAMNr with temporal CONDITIONAL flows (multiview, signal1d windows) — companion of ``train_lamnr_flows_hybrid``.

Same inputs (manifest + views JSON with ``signal1d`` views: ``name``, ``path_column``, ``shape`` [C, L], ``layout`` CL), same command
line for everything the two trainers share, and the same outputs in the run directory under the same names:

    run_config.json / run_config.txt        startup dump
    metrics.csv                             iter,loss,align,sum_bpd,val_loss,lr,bpd_<view>...,val_bpd_<view>...  (one row per iteration)
    training_state.pt                       latest checkpoint (+ training_state_it<N>.pt milestones: --keep-last, --keep-every)
    previews/<view>_recon_it<N>.png         original | reconstruction pairs   (every --preview-interval)
    previews/<view>_samples_it<N>.png       autoregressive samples            (every --preview-interval)
    objectives.png, bpd_by_view.png, val_bpd_by_view.png
    export/<view>_reconstructions.csv and export/<view>/reconstructions/row_<row>.npy   (--save-recon, like the hybrid trainer)

Extras that the hybrid trainer does not have (clearly separate files): ``best.pt`` (EMA weights of the best validation bpd, used for
export) and ``export/features.npz`` (+ ``export/untrained.npz`` with --untrained): per-sequence NLL, pooled context states, shared
embeddings and raw statistics for the evaluation script. ``bpd`` = bits per dimension of the scored positions (positions >= --burn).

Model. Each view is an autoregressive conditional flow over time:  log p(x_{1:T}) = sum_t log p(x_t | h_{t-1}),  h = causal TCN state
of the past frames of the same view. Building blocks from the ANTsX packages:
  ANTsNormalizingFlows : CoupledRationalQuadraticSpline (num_context_channels), Permute, distributions.base.ConditionalDiagGaussian,
                         core.ConditionalNormalizingFlow
  ANTsTorch            : lamnr_flows.misc.latent_alignment.LatentAlignmentLossManager / Projector (vicreg, barlow, infonce, hsic, ...),
                         and, when importable, the plotting / image-grid helpers of train_lamnr_flows_hybrid.
Written here (neither package has a causal encoder or a sequence wrapper): CausalTCN, window stitching, the loop.
The shared embedding of a view is Projector_v(mean_t h_t, t >= burn - 1); the alignment loss acts on these embeddings (positives = the
views of the same sequence).

Windows are stitched back into the per-pass sequences they were cut from (hop inferred from the overlap, which is verified); each view
is standardised with training statistics and gets a small fixed Gaussian noise (seeded per sequence, identical at train and export).

Not implemented / accepted only so that hybrid command lines run unchanged: --precision/--amp-dtype (float32), --num-workers,
--train-samples/--val-samples, --plateau-* (no plateau scheduler), --disable-augmentation, --alignment-latents, --alignment-pool-size,
--grad-checkpoint, --scale-cap, --weighting kendall, DDP. --save-z / --save-whitened do nothing for signal views (as in the hybrid trainer).
Modes: train (default) | --export-only (loads best.pt, or training_state.pt with --export-from last; --untrained = random weights).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import shutil
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from antsnormflows.core import ConditionalNormalizingFlow
from antsnormflows.distributions.base import ConditionalDiagGaussian, _clamp_log_scale
from antsnormflows.flows import CoupledRationalQuadraticSpline, Permute
from antstorch.lamnr_flows.misc.latent_alignment import LatentAlignmentLossManager, Projector

try:                                                   # identical figures / grids to the hybrid trainer whenever it can be imported
    from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import HybridLAMNrTrainer as _Hybrid
    from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import _save_hybrid_metric_plots as save_metric_plots
    display_slice, write_grid = _Hybrid._display_slice, _Hybrid._write_grid
    HELPERS = "train_lamnr_flows_hybrid"
except Exception:                                      # noqa: BLE001  (local copies of the same helpers)
    HELPERS = "local copies"

    def save_metric_plots(csv_path: Path, out_dir: Path) -> None:
        if not csv_path.exists():
            return
        try:
            import matplotlib.pyplot as plt
            frame = pd.read_csv(csv_path)
            if len(frame) < 2:
                return
            groups = [([c for c in ("loss", "align", "val_loss") if c in frame], "objective", "objectives.png"),
                      ([c for c in frame if c.startswith("bpd_")], "bits per dimension", "bpd_by_view.png"),
                      ([c for c in frame if c.startswith("val_bpd_")], "validation bits per dimension", "val_bpd_by_view.png")]
            for columns, ylabel, filename in groups:
                if not columns:
                    continue
                figure, axis = plt.subplots()
                for column in columns:
                    axis.plot(frame["iter"], frame[column], label=column)
                axis.set_xlabel("iteration")
                axis.set_ylabel(ylabel)
                axis.legend()
                figure.tight_layout()
                figure.savefig(out_dir / filename)
                plt.close(figure)
        except Exception as error:                     # noqa: BLE001
            tqdm.write(f"[metrics] plot generation skipped: {error}")

    def display_slice(tensor, mode: str = "clamp") -> np.ndarray:
        array = np.nan_to_num(tensor.detach().float().cpu().numpy())

        def _minmax(a):
            lo, hi = float(a.min()), float(a.max())
            return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        def _pclamp(a):
            lo, hi = np.percentile(a, [1, 99])
            return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if mode == "to01":
            return _minmax(array)
        if mode == "both":
            return _minmax(_pclamp(array))
        return _pclamp(array)

    def write_grid(images: Sequence[np.ndarray], path: Path, columns: int = 4) -> None:
        from PIL import Image
        if not images:
            return
        tiles = [Image.fromarray((image * 255).astype(np.uint8), mode="L") for image in images]
        width, height = max(x.width for x in tiles), max(x.height for x in tiles)
        rows = math.ceil(len(tiles) / columns)
        canvas = Image.new("L", (columns * width, rows * height), color=0)
        for index, tile in enumerate(tiles):
            canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)

ARCHS = {"long": (3, (1, 2, 4)), "mid": (3, (1, 2)), "short": (2, (1,))}      # (kernel, dilations): receptive field 15 / 7 / 2 frames
LN2 = math.log(2.0)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------ data
def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for i, part in enumerate(p.parts):
        if part.startswith("phase2_"):
            q = root.joinpath(*p.parts[i:])
            if q.exists():
                return q
    raise FileNotFoundError(path)


def build_sequences(man: pd.DataFrame, root: Path, col: str, limit: int = 0):
    """Stitch the windows of every contiguous run of a pass back into one (C, T) sequence."""
    man = man.sort_values(["pass_id", "start_frame"]).reset_index(drop=True)
    steps = man.groupby("pass_id").start_frame.diff().dropna()
    step_mode = int(steps[steps > 0].mode().iloc[0])
    seqs, meta, hop, n_bad = [], [], None, 0
    for k, (pid, g) in enumerate(man.groupby("pass_id", sort=False)):
        if limit and k >= limit:
            break
        wins = [np.load(_resolve(p, root)).astype(np.float32) for p in g[col]]
        starts = g.start_frame.to_numpy()
        runs, cur = [], [0]
        for i in range(1, len(wins)):
            if starts[i] - starts[i - 1] == step_mode:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        for r in runs:
            if hop is None and len(r) >= 2:
                W = wins[r[0]].shape[1]
                hop = min(range(1, W), key=lambda h: float(np.abs(wins[r[1]][:, :W - h] - wins[r[0]][:, h:]).mean()))
            seq, ok = wins[r[0]], True
            for i in r[1:]:
                ov = wins[i].shape[1] - hop
                if float(np.abs(wins[i][:, :ov] - seq[:, -ov:]).mean()) > 1e-3 * (np.abs(seq).mean() + 1e-9):
                    ok = False
                seq = np.concatenate([seq, wins[i][:, -hop:]], 1)
            n_bad += (not ok)
            row = g.iloc[r[0]]
            seqs.append(seq)
            meta.append(dict(pass_id=pid, start_frame=int(starts[r[0]]), subject_id=row.subject_id, split=str(row.get("split", "train")),
                             speed=str(row.speed).upper(), jacket=str(row.jacket) if "jacket" in g.columns else "all"))
    print(f"[data] {col}: {len(seqs)} sequences (length median {int(np.median([s.shape[1] for s in seqs]))}, "
          f"min {min(s.shape[1] for s in seqs)}, max {max(s.shape[1] for s in seqs)}); hop {hop}; runs with overlap mismatch: {n_bad}")
    return seqs, pd.DataFrame(meta)


def standardise(seqs, norm, noise, noise_seed, meta):
    out = []
    for s, pid, st in zip(seqs, meta.pass_id, meta.start_frame):
        x = (s - norm["mu"]) / norm["sd"]
        if noise > 0:
            g = np.random.default_rng(zlib.crc32(f"{pid}|{st}".encode()) ^ noise_seed)
            x = x + noise * g.standard_normal(x.shape)
        out.append(x.astype(np.float32))
    return out


def pad_batch(seqs, idx, device):
    L = max(seqs[i].shape[1] for i in idx)
    x = np.zeros((len(idx), seqs[idx[0]].shape[0], L), np.float32)
    lens = np.zeros(len(idx), int)
    for b, i in enumerate(idx):
        x[b, :, :seqs[i].shape[1]] = seqs[i]
        lens[b] = seqs[i].shape[1]
    return torch.tensor(x, device=device), torch.tensor(lens, device=device)


# ------------------------------------------------------------------ model
class CausalTCN(nn.Module):
    def __init__(self, c, h, dilations, k):
        super().__init__()
        self.k, self.dil = k, dilations
        self.convs = nn.ModuleList([nn.Conv1d(c if i == 0 else h, h, k, dilation=d) for i, d in enumerate(dilations)])

    def forward(self, x):                      # (B, C, T) -> (B, H, T); output j depends on x[..., :j+1] only
        for i, (conv, d) in enumerate(zip(self.convs, self.dil)):
            y = F.gelu(conv(F.pad(x, ((self.k - 1) * d, 0))))
            x = y if i == 0 else x + y
        return x


class ViewModel(nn.Module):
    def __init__(self, C, a):
        super().__init__()
        k, dil = ARCHS[a.arch]
        self.C = C
        self.enc = CausalTCN(C, a.ctx_dim, dil, k)
        flows = []
        for j in range(a.K):
            flows.append(CoupledRationalQuadraticSpline(C, a.num_blocks, a.hidden, num_context_channels=a.ctx_dim, num_bins=a.num_bins,
                                                        tail_bound=a.tail_bound, reverse_mask=bool(j % 2)))
            flows.append(Permute(C, mode="shuffle"))
        base = nn.Linear(a.ctx_dim, 2 * C)
        nn.init.zeros_(base.weight)
        nn.init.zeros_(base.bias)
        self.flow = ConditionalNormalizingFlow(ConditionalDiagGaussian(C, base), flows)

    def _context(self, x):                     # h_0 .. h_{T-2}: (B, H, T-1)
        return self.enc(x[:, :, :-1])

    def forward(self, x):                      # log p of positions 1..T-1 (B, T-1) and context states (B, H, T-1)
        h = self._context(x)
        B, _, T1 = h.shape
        tgt = x[:, :, 1:].permute(0, 2, 1).reshape(B * T1, -1)
        lp = self.flow.log_prob(tgt, context=h.permute(0, 2, 1).reshape(B * T1, -1)).view(B, T1)
        return lp, h

    @torch.no_grad()
    def reconstruct(self, x):                  # flow latents and round trip of positions 1..T-1 (frame 0 is context only)
        h = self._context(x)
        B, _, T1 = h.shape
        hc = h.permute(0, 2, 1).reshape(B * T1, -1)
        z = x[:, :, 1:].permute(0, 2, 1).reshape(B * T1, -1)
        for f in reversed(self.flow.flows):
            z, _ = f.inverse(z, context=hc)
        latent = z
        for f in self.flow.flows:
            z, _ = f(z, context=hc)
        rec = z.view(B, T1, -1).permute(0, 2, 1)
        return latent.view(B, T1, -1).permute(0, 2, 1), torch.cat([x[:, :, :1], rec], 2)

    @torch.no_grad()
    def generate(self, x0, T, temperature):    # autoregressive sampling from the first frame: (B, C, 1) -> (B, C, T)
        q0, x = self.flow.q0, x0
        for _ in range(T - 1):
            h = self.enc(x)[:, :, -1]
            enc = q0.context_encoder(h)
            mean, log_scale = enc[..., : enc.shape[-1] // 2], enc[..., enc.shape[-1] // 2:]
            z = mean + temperature * torch.exp(_clamp_log_scale(log_scale, q0.min_log, q0.max_log)) * torch.randn_like(mean)
            for f in self.flow.flows:
                z, _ = f(z, context=h)
            x = torch.cat([x, z[:, :, None]], 2)
        return x


def pool_state(h, lens, burn):
    j = torch.arange(h.shape[2], device=h.device)[None]
    m = ((j >= burn - 1) & (j <= lens[:, None] - 2)).float()
    return (h * m[:, None]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)


def score_mask(lens, T1, burn, device):
    t = torch.arange(1, T1 + 1, device=device)[None]
    return (t < lens[:, None]) & (t >= burn)


# ------------------------------------------------------------------ trainer
class TemporalLAMNrTrainer:
    def setup(self, args: argparse.Namespace) -> None:
        self.args = args
        self.world_size = 1
        self.run_dir = Path(args.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "training_state.pt"
        self.best_path = self.run_dir / "best.pt"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.dev = torch.device("cuda" if args.devices == "cuda" and torch.cuda.is_available() else
                                "mps" if args.devices == "mps" else "cpu")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        cfg = json.load(open(args.config))
        self.views = cfg["views"]
        assert all(v["type"] == "signal1d" for v in self.views), "only signal1d views are supported"
        assert len(self.views) >= 2, "need at least two views for the alignment"
        self.names = [v["name"] for v in self.views]
        subject_column = args.subject_column or cfg.get("subject_column", "subject_id")
        root = Path(args.data_root) if args.data_root else Path(args.manifest).resolve().parent.parent
        self.full_frame = pd.read_csv(args.manifest)
        raw, self.meta = [], None
        for v in self.views:
            s, m = build_sequences(self.full_frame, root, v["path_column"], args.limit_passes)
            raw.append(s)
            self.meta = m
        assert all(len(r) == len(raw[0]) and all(x.shape[1] == y.shape[1] for x, y in zip(r, raw[0])) for r in raw), "views differ"

        ck = None
        self._resume_path = None
        if args.export_only:
            path = Path(args.checkpoint) if args.checkpoint else (self.state_path if args.export_from == "last" else self.best_path)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self._ck_export = ck
            self._resume_path = path
            cfg_a = {**ck["config"], "devices": args.devices}
            for k in ("arch", "ctx_dim", "K", "hidden", "num_blocks", "num_bins", "tail_bound", "proj_dim", "proj_hidden", "burn",
                      "noise", "noise_seed", "seed"):
                setattr(args, k, cfg_a[k])
            self.normalizers = ck["normalizers"]
            split = np.array(["test"] * len(self.meta))
        else:
            resume = args.resume or (str(self.state_path) if args.auto_resume and self.state_path.exists() else "")
            if resume:
                ck = torch.load(resume, map_location="cpu", weights_only=False)
                self._resume_path = Path(resume)
                if args.use_ckpt_config:
                    for k in ("arch", "ctx_dim", "K", "hidden", "num_blocks", "num_bins", "tail_bound", "proj_dim", "proj_hidden"):
                        setattr(args, k, ck["config"][k])
            split = self.meta.split.to_numpy().copy()
            if "val" not in set(split):
                subs = np.array(sorted(set(self.meta.subject_id[split == "train"])))
                nv = max(1, int(round(args.val_fraction * len(subs))))
                vs = set(np.random.default_rng(args.seed).choice(subs, nv, replace=False))
                split = np.where(self.meta.subject_id.isin(vs), "val", split)
            self.train_ix, self.val_ix = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
            self.normalizers = {}
            for name, r in zip(self.names, raw):
                fr = np.concatenate([r[i] for i in self.train_ix], 1)
                self.normalizers[name] = {"mu": fr.mean(1, keepdims=True), "sd": fr.std(1, keepdims=True) + 1e-6}
        self.seq_split = split
        self.S = [standardise(raw[i], self.normalizers[n], args.noise, args.noise_seed, self.meta) for i, n in enumerate(self.names)]
        self.raw = raw

        Cs = [s[0].shape[0] for s in self.S]
        self.models = nn.ModuleList([ViewModel(C, args) for C in Cs]).to(self.dev)
        self.projectors = nn.ModuleList([Projector(args.ctx_dim, args.proj_hidden, args.proj_dim) for _ in Cs]).to(self.dev)
        self.ema_models = self.ema_projectors = None
        self.start_iter, self.best, self.bad = 1, 1e9, 0
        self.last_val_bpds = [float("nan")] * len(self.views)
        if args.export_only:
            if args.untrained:
                torch.manual_seed(args.seed + 100)
                self.models = nn.ModuleList([ViewModel(C, args) for C in Cs]).to(self.dev)
                self.projectors = nn.ModuleList([Projector(args.ctx_dim, args.proj_hidden, args.proj_dim) for _ in Cs]).to(self.dev)
                self.export_models, self.export_projectors = self.models, self.projectors
            else:
                key_m, key_p = ("ema_models", "ema_projectors") if ck.get("ema_models") is not None else ("models", "projectors")
                self.models.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck[key_m]) for k, v in sd.items()})
                self.projectors.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck[key_p]) for k, v in sd.items()})
                self.export_models, self.export_projectors = self.models, self.projectors
            return

        params = list(self.models.parameters()) + list(self.projectors.parameters())
        self.opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        if ck is not None:
            self.models.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck["models"]) for k, v in sd.items()})
            self.projectors.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck["projectors"]) for k, v in sd.items()})
            self.opt.load_state_dict(ck["optimizer"])
            self.start_iter = int(ck["iter"])
            self.best = float(ck.get("best", 1e9))
            self.normalizers = ck["normalizers"]
        if args.ema:
            self.ema_models, self.ema_projectors = copy.deepcopy(self.models), copy.deepcopy(self.projectors)
            if ck is not None and ck.get("ema_models") is not None:
                self.ema_models.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck["ema_models"]) for k, v in sd.items()})
                self.ema_projectors.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck["ema_projectors"]) for k, v in sd.items()})
            for p in list(self.ema_models.parameters()) + list(self.ema_projectors.parameters()):
                p.requires_grad_(False)
        self.align_mgr = LatentAlignmentLossManager(args, self.projectors, self.dev)
        self.max_iter = args.max_iter + args.extra_iters

        availability = {n: int(len(self.train_ix)) for n in self.names}
        feature_dims = [args.ctx_dim] * len(self.views)
        (self.run_dir / "run_config.json").write_text(json.dumps({
            "trainer": "temporal", "world_size": 1, "device": str(self.dev), "arguments": vars(args), "views": self.views,
            "availability": availability, "projector_input_dimensions": feature_dims}, indent=2, default=str))
        summary = self._format_run_summary(subject_column, feature_dims, availability)
        print("\n" + summary)
        (self.run_dir / "run_config.txt").write_text(summary + "\n")
        for view, model in zip(self.views, self.models):
            print(f"[init] {view['name']} (signal1d): {n_params(model):,} parameters")
        if ck is not None:
            tqdm.write(f"[resume] from {self._resume_path} @ iter {self.start_iter}")
            self._clean_csv_after(self.start_iter)
        header = ("iter,loss,align,sum_bpd,val_loss,lr," + ",".join(f"bpd_{n}" for n in self.names) + "," +
                  ",".join(f"val_bpd_{n}" for n in self.names))
        if not self.metrics_path.exists() or self.start_iter == 1:
            self.metrics_path.write_text(header + "\n")

    def _clean_csv_after(self, start_iter: int) -> None:
        if not self.metrics_path.exists():
            return
        try:
            df = pd.read_csv(self.metrics_path)
            df = df[df["iter"] < start_iter]
            df.to_csv(self.metrics_path, index=False, float_format="%.8g")
        except Exception as e:                         # noqa: BLE001
            print(f"[warn] Could not clean CSV: {e}")

    def _format_run_summary(self, subject_column, feature_dims, availability) -> str:
        args = self.args
        rows = [f"[run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Py {platform.python_version()} | torch {torch.__version__} | "
                f"cuda={str(torch.cuda.is_available()).lower()} (n={torch.cuda.device_count()})",
                "[note] temporal conditional-flow trainer (signal1d views; helpers: " + HELPERS + ")"]

        def add(label: str, value: Any) -> None:
            rows.append(f"{label:>28}: {'None' if value is None else value}")
        add("out_dir", args.out_dir)
        add("manifest / config", f"{args.manifest} / {args.config}")
        add("world_size / device", f"1 / {self.dev}")
        add("views", len(self.views))
        add("precision / amp_dtype", "float / (not used)")
        add("seed", args.seed)
        add("batch local per GPU", args.batch_size)
        add("grad_accum", args.accum_steps)
        add("effective global batch", int(args.batch_size) * int(args.accum_steps))
        add("max_iter / extra", f"{args.max_iter} / {args.extra_iters}")
        add("start / target iteration", f"{self.start_iter} / {self.max_iter}")
        add("planned steps this run", max(0, self.max_iter - self.start_iter + 1))
        add("eval / preview interval", f"{args.eval_interval} / {args.preview_interval}")
        add("lr / warmup", f"{args.lr} / {args.warmup_iters}")
        add("grad_clip / weight_decay", f"{args.grad_clip} / {args.weight_decay}")
        add("ema / decay", f"{args.ema} / {args.ema_decay}")
        add("lr_decay gamma / steps", f"{args.lr_decay_gamma} / {args.lr_decay_steps}")
        add("min_lr", args.min_lr)
        add("resume argument", args.resume or None)
        add("resolved checkpoint", str(self._resume_path) if self._resume_path is not None else None)
        add("auto_resume / ckpt config", f"{args.auto_resume} / {args.use_ckpt_config}")
        add("manifest rows", len(self.full_frame))
        add("subject column", subject_column)
        add("train / val sequences", f"{len(self.train_ix)} / {len(self.val_ix)}")
        add("validation mode", "held-out subjects")
        add("sequence burn-in / noise", f"{args.burn} / {args.noise}")
        add("align / weighting", f"{args.align} / {args.weighting}")
        add("align weight / warmup", f"{args.align_weight} / {args.align_warmup}")
        add("alignment latents", "pooled context state (all views)")
        add("proj dim / hidden", f"{args.proj_dim} / {args.proj_hidden}")
        add("vicreg inv/var/cov/gamma", f"{args.vicreg_inv} / {args.vicreg_var} / {args.vicreg_cov} / {args.vicreg_gamma}")
        add("screen / fraction", f"{args.screen} / {args.screen_frac}")
        add("screen warmup / refresh", f"{args.screen_warmup} / {args.screen_refresh}")
        add("cca ridge / prefilter", f"{args.cca_ridge} / {args.prefilter_frac}")
        add("sample mode / temp", f"{args.sample_mode} / {args.sample_temp}")
        add("preview samples / columns", f"{args.preview_samples} / {args.preview_columns}")
        add("flow K / hidden / blocks / bins", f"{args.K} / {args.hidden} / {args.num_blocks} / {args.num_bins}")
        add("encoder arch / ctx dim", f"{args.arch} / {args.ctx_dim}  (receptive field {1 + (ARCHS[args.arch][0] - 1) * sum(ARCHS[args.arch][1])} frames)")
        rows.append("-" * 72)
        for i, (view, model) in enumerate(zip(self.views, self.models)):
            add(f"view[{i}] name / type", f"{view['name']} / signal1d")
            add(f"view[{i}] observed train", availability[view["name"]])
            add(f"view[{i}] flow parameters", f"{n_params(model):,}")
            add(f"view[{i}] projector input", feature_dims[i])
            add(f"view[{i}] shape / channels", f"{tuple(view['shape'][1:])} / {view['shape'][0]}")
            add(f"view[{i}] model", "temporal conditional flow (config 'model' entry ignored)")
        rows.append("-" * 72)
        return "\n".join(rows)

    # ---------------- losses
    def _batch_loss(self, idx, iteration, models, projectors):
        args = self.args
        L_nll, lat, bpds = torch.tensor(0.0, device=self.dev), [], []
        for v, model in enumerate(models):
            x, lens = pad_batch(self.S[v], idx, self.dev)
            lp, h = model(x)
            valid = score_mask(lens, lp.shape[1], args.burn, self.dev)
            bpd = -(lp * valid).sum() / valid.sum().clamp(min=1) / (LN2 * model.C)
            L_nll = L_nll + bpd
            bpds.append(float(bpd.detach()))
            lat.append(torch.nan_to_num(pool_state(h, lens, args.burn).float()))
        mgr = self.align_mgr
        if projectors is not self.projectors:
            mgr = LatentAlignmentLossManager(args, projectors, self.dev)
        loss, align, _, _ = mgr.compute(lat, L_nll, iteration, None, None)
        return loss, align, bpds

    def lr_at(self, iteration: int) -> float:
        a = self.args
        if iteration <= a.warmup_iters:
            return a.lr * iteration / max(a.warmup_iters, 1)
        return max(a.min_lr, a.lr * (a.lr_decay_gamma ** ((iteration - 1 - a.warmup_iters) // max(a.lr_decay_steps, 1))))

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_models is None:
            return
        for src, dst in ((self.models, self.ema_models), (self.projectors, self.ema_projectors)):
            sd = src.state_dict()
            for k, v in dst.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.args.ema_decay).add_(sd[k].detach(), alpha=1 - self.args.ema_decay)
                else:
                    v.copy_(sd[k])

    def _active(self):
        return (self.ema_models if self.ema_models is not None else self.models,
                self.ema_projectors if self.ema_projectors is not None else self.projectors)

    # ---------------- training
    def train(self) -> None:
        a = self.args
        rng = np.random.default_rng(a.seed + self.start_iter)
        order, pos = rng.permutation(self.train_ix), 0
        ema_loss_disp = ema_align_disp = None
        t0 = time.time()
        pbar = tqdm(range(self.start_iter, self.max_iter + 1), desc="train-temporal", disable=a.no_progress,
                    initial=max(0, self.start_iter - 1), total=self.max_iter)
        for iteration in pbar:
            self.models.train()
            self.projectors.train()
            lr = self.lr_at(iteration)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            loss_sum = align_sum = 0.0
            bpd_sum = np.zeros(len(self.views))
            for _ in range(a.accum_steps):
                if pos + a.batch_size > len(order):
                    order, pos = rng.permutation(self.train_ix), 0
                idx = list(order[pos:pos + a.batch_size])
                pos += a.batch_size
                loss, align, bpds = self._batch_loss(idx, iteration, self.models, self.projectors)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at iteration {iteration}")
                (loss / a.accum_steps).backward()
                loss_sum += float(loss.detach())
                align_sum += float(align.detach())
                bpd_sum += np.array(bpds)
            nn.utils.clip_grad_norm_(list(self.models.parameters()) + list(self.projectors.parameters()), a.grad_clip)
            self.opt.step()
            self._update_ema()
            denom = float(a.accum_steps)
            mean_loss, mean_align = loss_sum / denom, align_sum / denom
            bpds = list(bpd_sum / denom)
            sum_bpd = sum(x for x in bpds if math.isfinite(x))
            val_loss = float("nan")
            do_eval = iteration % a.eval_interval == 0 or iteration == self.max_iter
            if do_eval:
                val_loss = self.validate(iteration)
            if a.smooth_alpha > 0:
                ema_loss_disp = mean_loss if ema_loss_disp is None else (1 - a.smooth_alpha) * ema_loss_disp + a.smooth_alpha * mean_loss
                ema_align_disp = mean_align if ema_align_disp is None else (1 - a.smooth_alpha) * ema_align_disp + a.smooth_alpha * mean_align
                disp_loss, disp_align = ema_loss_disp, ema_align_disp
            else:
                disp_loss, disp_align = mean_loss, mean_align
            pbar.set_postfix(loss=f"{disp_loss:.4f}", align=f"{disp_align:.4f}")
            row = [iteration, mean_loss, mean_align, sum_bpd, val_loss, lr, *bpds, *self.last_val_bpds]
            with open(self.metrics_path, "a") as stream:
                stream.write(",".join(f"{x:.8g}" for x in row) + "\n")
            if do_eval:
                vb = sum(self.last_val_bpds)
                improved = vb < self.best - 1e-4
                if improved:
                    self.best, self.bad = vb, 0
                else:
                    self.bad += 1
                self.save_checkpoint(iteration, best=improved)
                self._save_previews(iteration)
                save_metric_plots(self.metrics_path, self.run_dir)
                tqdm.write(f"[val] best sum_bpd={self.best:.6g} ({self.bad} evaluation(s) since the best; {(time.time() - t0) / 60:.1f} min)")
                if a.patience_evals and self.bad >= a.patience_evals:
                    tqdm.write(f"[early-stop] no improvement for {self.bad} evaluations")
                    break
            gc.collect()
        pbar.close()
        self.export()

    @torch.no_grad()
    def validate(self, iteration: int) -> float:
        a = self.args
        models, projectors = self._active()
        models.eval()
        projectors.eval()
        tot_loss, n_loss = 0.0, 0
        tot_bpd, n_bpd = np.zeros(len(self.views)), 0
        for b, i in enumerate(range(0, len(self.val_ix), a.batch_size)):
            idx = list(self.val_ix[i:i + a.batch_size])
            if len(idx) < 4:
                continue
            loss, _, bpds = self._batch_loss(idx, iteration, models, projectors)
            if torch.isfinite(loss):
                tot_loss += float(loss)
                n_loss += 1
            tot_bpd += np.array(bpds)
            n_bpd += 1
            if a.val_batches and b + 1 >= a.val_batches:
                break
        self.last_val_bpds = list(tot_bpd / max(n_bpd, 1))
        value = tot_loss / n_loss if n_loss else float("nan")
        tqdm.write(f"[val] iter={iteration} loss={value:.6g} ({'EMA' if self.ema_models is not None else 'base'}; "
                   + ", ".join(f"{n}={x:.4g}" for n, x in zip(self.names, self.last_val_bpds)) + ")")
        return value

    # ---------------- checkpoints
    def _blob(self, iteration: int, best: bool = False) -> Dict[str, Any]:
        return {
            "iter": iteration + 1,
            "models": [m.state_dict() for m in self.models],
            "ema_models": [m.state_dict() for m in self.ema_models] if self.ema_models is not None else None,
            "projectors": [p.state_dict() for p in self.projectors],
            "ema_projectors": [p.state_dict() for p in self.ema_projectors] if self.ema_projectors is not None else None,
            "optimizer": self.opt.state_dict(), "warmup": None, "plateau": None, "scaler": None,
            "normalizers": self.normalizers, "views": self.views, "augmentation_groups": {},
            "config": vars(self.args), "kendall": {"s_nll": None, "s_align": None},
            "best": self.best, "val_bpds": self.last_val_bpds,
        }

    def save_checkpoint(self, iteration: int, best: bool = False) -> None:
        blob = self._blob(iteration)
        torch.save(blob, self.state_path)
        torch.save(blob, self.run_dir / f"training_state_it{iteration:06d}.pt")
        if best:
            torch.save({**blob, "best_iter": iteration}, self.best_path)
            tqdm.write(f"[ckpt] saved best.pt (iter {iteration}, sum val bpd {self.best:.6g})")
        self.cleanup_checkpoints()
        free_gb = shutil.disk_usage(self.run_dir).free / 2**30
        if free_gb < self.args.disk_warning_gb:
            tqdm.write(f"[disk warning] only {free_gb:.1f} GiB free in {self.run_dir}")

    def cleanup_checkpoints(self) -> None:
        files = sorted(self.run_dir.glob("training_state_it*.pt"))
        keep = set(files[-self.args.keep_last:]) if self.args.keep_last else set()
        for path in files:
            try:
                iteration = int(path.stem.rsplit("it", 1)[1])
            except (ValueError, IndexError):
                keep.add(path)
                continue
            if self.args.keep_every > 0 and iteration % self.args.keep_every == 0:
                keep.add(path)
        for path in files:
            if path not in keep:
                path.unlink()

    # ---------------- previews
    @torch.no_grad()
    def _save_previews(self, iteration: int) -> None:
        a = self.args
        if a.preview_interval <= 0 or iteration % a.preview_interval:
            return
        models, _ = self._active()
        models.eval()
        idx = list(self.val_ix[: a.preview_samples])
        preview_dir = self.run_dir / "previews"
        for v, (name, model) in enumerate(zip(self.names, models)):
            x, lens = pad_batch(self.S[v], idx, self.dev)
            latent, rec = model.reconstruct(x)
            keep = torch.arange(x.shape[2], device=self.dev)[None] < lens[:, None]
            finite = torch.isfinite(rec)
            safe = torch.nan_to_num(rec.float())
            err = ((safe - x.float()).abs())[keep[:, None, :].expand_as(x)]
            tqdm.write(f"[recon] iter={iteration} view={name} mae={float(err.mean()):.6g} rmse={float(err.square().mean().sqrt()):.6g} "
                       f"max={float(err.max()):.6g} finite={float(finite.float().mean()):.6f} "
                       f"x_range=[{float(x.min()):.6g},{float(x.max()):.6g}] recon_range=[{float(safe.min()):.6g},{float(safe.max()):.6g}] "
                       f"latent_abs_max={float(torch.nan_to_num(latent.float()).abs().max()):.6g}")
            panels = []
            for b in range(x.shape[0]):
                T = int(lens[b])
                panels.extend([display_slice(x[b, :, :T]), display_slice(rec[b, :, :T])])
            write_grid(panels, preview_dir / f"{name}_recon_it{iteration:06d}.png", columns=2)
            if a.sample_mode == "model":
                T = int(np.median([self.S[v][i].shape[1] for i in self.val_ix]))
                x0 = x[:, :, :1]
                if x0.shape[0] < a.preview_samples:
                    x0 = x0.repeat(math.ceil(a.preview_samples / x0.shape[0]), 1, 1)[: a.preview_samples]
                samples = model.generate(x0, T, a.sample_temp).cpu()
                write_grid([display_slice(s, mode=a.sample_grid_norm) for s in samples],
                           preview_dir / f"{name}_samples_it{iteration:06d}.png", columns=a.preview_columns)

    # ---------------- export
    @torch.no_grad()
    def export(self) -> None:
        a = self.args
        if a.export_only:
            models, projectors, tag = self.export_models, self.export_projectors, (a.export_tag or ("untrained" if a.untrained else "features"))
        else:
            if self.best_path.exists():                       # the exported weights are those of the best validation bpd
                ck = torch.load(self.best_path, map_location=self.dev, weights_only=False)
                key_m, key_p = ("ema_models", "ema_projectors") if ck.get("ema_models") is not None else ("models", "projectors")
                models = nn.ModuleList([ViewModel(s[0].shape[0], a) for s in self.S]).to(self.dev)
                projectors = nn.ModuleList([Projector(a.ctx_dim, a.proj_hidden, a.proj_dim) for _ in self.S]).to(self.dev)
                models.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck[key_m]) for k, v in sd.items()})
                projectors.load_state_dict({f"{i}.{k}": v for i, sd in enumerate(ck[key_p]) for k, v in sd.items()})
            else:
                models, projectors = self._active()
            tag = a.export_tag or "features"
        models.eval()
        projectors.eval()
        export_dir = self.run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        N, V, bs = len(self.S[0]), len(self.S), 64
        out: Dict[str, Any] = {}
        for v in range(V):
            out[f"nll_{v}"] = np.full(N, np.nan)
            out[f"h_{v}"], out[f"z_{v}"] = [], []
        recon_records = {n: [] for n in self.names}
        for i in range(0, N, bs):
            idx = list(range(i, min(i + bs, N)))
            for v, (name, model) in enumerate(zip(self.names, models)):
                x, lens = pad_batch(self.S[v], idx, self.dev)
                lp, h = model(x)
                ev = score_mask(lens, lp.shape[1], a.burn, self.dev)
                nll = -(lp * ev).sum(1) / ev.sum(1).clamp(min=1) / model.C
                nll[ev.sum(1) == 0] = float("nan")
                out[f"nll_{v}"][idx] = nll.cpu().numpy()
                hp = pool_state(h, lens, a.burn)
                out[f"h_{v}"].append(hp.cpu().numpy())
                out[f"z_{v}"].append(projectors[v](hp).cpu().numpy())
                if a.save_recon and not a.untrained and len(recon_records[name]) < a.export_max_samples:
                    _, rec = model.reconstruct(x)
                    vdir = export_dir / name / "reconstructions"
                    vdir.mkdir(parents=True, exist_ok=True)
                    for b, row in enumerate(idx):
                        if len(recon_records[name]) >= a.export_max_samples:
                            break
                        T = int(lens[b])
                        arr = rec[b, :, :T].float().cpu().numpy()
                        nm = self.normalizers[name]
                        arr = (arr * nm["sd"] + nm["mu"]).astype(np.float32)           # original units
                        path = vdir / f"row_{int(row):06d}.npy"
                        np.save(path, arr)
                        recon_records[name].append({"row": int(row), "path": str(path)})
        for name, records in recon_records.items():
            if records:
                pd.DataFrame(records).to_csv(export_dir / f"{name}_reconstructions.csv", index=False)
        for v in range(V):
            out[f"h_{v}"] = np.concatenate(out[f"h_{v}"])
            out[f"z_{v}"] = np.concatenate(out[f"z_{v}"])
            out[f"raw_{v}"] = np.array([np.r_[s[:, a.burn - 1:].mean(1), s[:, a.burn - 1:].std(1)] if s.shape[1] > a.burn else
                                        np.full(2 * s.shape[0], np.nan) for s in self.S[v]])
        out["length"] = np.array([s.shape[1] for s in self.S[0]])
        for k in ("pass_id", "subject_id", "split", "speed", "jacket"):
            out[k] = self.meta[k].to_numpy().astype(str)
        out["view_names"] = np.array(self.names)
        np.savez(export_dir / f"{tag}.npz", **out)
        print(f"[export] wrote EMA-aware outputs to {export_dir}  ({tag}.npz: {N} sequences)")


def _build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with one row per window (pass_id, start_frame, subject_id, split, speed, <path columns>)")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="runs_temporal")
    p.add_argument("--subject-column", default="")
    p.add_argument("--devices", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-root", default="", help="folder holding phase2_*/ when the manifest paths are absolute elsewhere")
    p.add_argument("--limit-passes", type=int, default=0, help="smoke test: only the first N passes")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.125, help="subject-wise; used only if the manifest has no 'val' split")
    p.add_argument("--num-workers", type=int, default=0)                    # accepted, unused
    p.add_argument("--train-samples", type=int, default=0)                  # accepted, unused
    p.add_argument("--val-samples", type=int, default=0)                    # accepted, unused
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--precision", default="float", choices=["float", "mixed"])        # accepted, unused
    p.add_argument("--amp-dtype", default="bf16", choices=["fp16", "bf16"])           # accepted, unused
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--plateau-factor", type=float, default=0.999999)                  # accepted, unused
    p.add_argument("--plateau-patience", type=int, default=100000)
    p.add_argument("--plateau-threshold", type=float, default=1e-3)
    p.add_argument("--plateau-cooldown", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=5e-5)
    p.add_argument("--resume", default="")
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--use-ckpt-config", action="store_true")
    p.add_argument("--extra-iters", type=int, default=0)
    p.add_argument("--smooth-alpha", type=float, default=0.1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--disable-augmentation", action="store_true")                     # accepted, unused
    p.add_argument("--grad-checkpoint", default="auto", choices=["auto", "on", "off"])  # accepted, unused
    p.add_argument("--scale-cap", type=float, default=3.0)                            # accepted, unused
    # temporal model
    p.add_argument("--arch", default="long", choices=list(ARCHS))
    p.add_argument("--ctx-dim", type=int, default=64)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=4.0)
    p.add_argument("--burn", type=int, default=8, help="positions < burn are context only (not scored)")
    p.add_argument("--noise", type=float, default=0.02, help="fixed seeded Gaussian noise in SD units")
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--patience-evals", type=int, default=0, help="stop after this many evaluations without improvement (0 = off)")
    # alignment (same names and defaults as the hybrid trainer)
    p.add_argument("--align", default="vicreg", choices=["vicreg", "barlow", "infonce", "hsic", "pearson", "mse", "none"])
    p.add_argument("--align-weight", type=float, default=0.05)
    p.add_argument("--align-warmup", type=int, default=200)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=128)
    p.add_argument("--alignment-latents", default="all-pooled")                       # accepted, unused (always pooled context state)
    p.add_argument("--alignment-pool-size", type=int, default=2)                      # accepted, unused
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--barlow-lambda", type=float, default=5e-3)
    p.add_argument("--weighting", default="fixed", choices=["fixed"])
    p.add_argument("--vicreg-inv", type=float, default=25.0)
    p.add_argument("--vicreg-cov", type=float, default=1.0)
    p.add_argument("--vicreg-var", type=float, nargs="+", default=[25.0])
    p.add_argument("--vicreg-gamma", type=float, nargs="+", default=[1.0])
    p.add_argument("--hsic-sigma", type=float, default=0.0)
    p.add_argument("--screen", default="none", choices=["none", "cca", "hsic"])
    p.add_argument("--screen-warmup", type=int, default=500)
    p.add_argument("--screen-refresh", type=int, default=0)
    p.add_argument("--screen-frac", type=float, default=0.5)
    p.add_argument("--cca-ridge", type=float, default=1e-3)
    p.add_argument("--prefilter-frac", type=float, default=0.5)
    # previews / checkpoints / export (same names as the hybrid trainer)
    p.add_argument("--preview-interval", type=int, default=500)
    p.add_argument("--preview-samples", type=int, default=8)
    p.add_argument("--preview-columns", type=int, default=4)
    p.add_argument("--sample-mode", default="model", choices=["off", "model"])
    p.add_argument("--sample-temp", type=float, default=1.0)
    p.add_argument("--sample-grid-norm", default="to01")
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--keep-every", type=int, default=5000)
    p.add_argument("--disk-warning-gb", type=float, default=10.0)
    p.add_argument("--save-z", action="store_true")                                   # no effect for signal views (as in the hybrid trainer)
    p.add_argument("--save-whitened", action="store_true")
    p.add_argument("--save-recon", action="store_true")
    p.add_argument("--export-max-samples", type=int, default=100)
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--export-from", default="best", choices=["best", "last"], help="--export-only: best.pt or training_state.pt")
    p.add_argument("--checkpoint", default="", help="--export-only: explicit checkpoint path")
    p.add_argument("--untrained", action="store_true", help="--export-only: random weights of the same architecture (seed + 100)")
    p.add_argument("--export-tag", default="")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_args(argv)
    trainer = TemporalLAMNrTrainer()
    trainer.setup(args)
    if args.export_only:
        trainer.export()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
