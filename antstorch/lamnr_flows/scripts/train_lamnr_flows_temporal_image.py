#!/usr/bin/env python3
"""Temporal CONDITIONAL Glow on segmentation images — variant A of the PiqueFlows conditional-flow study.

    log p(x_t | x_{t-m}, ..., x_{t-1})      x_t = 6 x 64 x 64 DensePose part fractions (one frame of a pass, 15 Hz)

A multiscale Glow (squeeze -> K x [ActNorm, invertible 1x1 conv, affine coupling] -> split, L levels) whose coupling networks also see a
spatial context: a small convolutional pyramid over the m previous frames of the same pass (stacked as channels) gives, at every level, a
feature map at that level's resolution that is concatenated to the coupling input. With --ctx-frames 0 the same model is an unconditional
Glow (no context): its bits per dimension minus the conditional one is the information the past carries about the present frame.

Building blocks from ANTsNormalizingFlows: ActNorm, Invertible1x1Conv, AffineCouplingBlock, distributions.GlowBase. Written here, because
GlowBlock2d / MultiscaleFlow have no context argument: the conditional coupling network, the context pyramid, and the multiscale
wrapper (squeeze = pixel_unshuffle, a parameter-free reshape). Same architecture family as MoGlow (Henter et al. 2020) but for images.

Data: the Phase 1 cycle manifest (one row per frame: subject_id, split, pass_id, speed, jacket, frame, phase, seg) and its views JSON
(``seg`` view: path_column, channels, dequantize = number of levels K). Frames of one pass are cut into runs of constant frame step; every
position t >= --burn of a run is a training example (the first --burn frames are context only, so models with different --ctx-frames score
the same frames). Values lie on the grid n/K, n = 0..K; dequantisation x = (n + u) / (K + 1), u ~ U[0,1) when training, u seeded per frame
for validation/export, and u = 0.5 for the context frames (deterministic). bits/dim are continuous; discrete bpd = bpd + log2(K + 1).
No pose, phase, speed or any other label enters the model.

Outputs (same names as train_lamnr_flows_hybrid / train_lamnr_flows_temporal):
    run_config.json/txt, metrics.csv (iter,loss,align,sum_bpd,val_loss,lr,bpd_<view>,val_bpd_<view>), training_state*.pt, best.pt,
    previews/<view>_recon_it######.png (original | reconstruction) and <view>_samples_it######.png (rows: real frames / sampled rollout),
    objectives.png, bpd_by_view.png, val_bpd_by_view.png, export/features.npz, export/<view>_reconstructions.csv (+ npy, --save-recon).
export/features.npz: per-sequence nll_0 (conditional bpd), nll_shuf_0 (context taken from another sequence: the information carried by
the *right* context is nll_shuf_0 - nll_0), h_0 (mean global context feature), length, pass_id, subject_id, split, speed, jacket;
per-frame arrays frame_* (sequence index, t, frame number, phase, bpd, bpd_shuf).

Not implemented / accepted only so that hybrid command lines run unchanged: precision/amp, num-workers, plateau scheduler, DDP, alignment
(single view: loss = bits per dimension, align = 0).
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from antsnormflows.distributions.base import GlowBase
from antsnormflows.flows import ActNorm, AffineCouplingBlock, Invertible1x1Conv

try:
    from antstorch.lamnr_flows.scripts.train_lamnr_flows_temporal import save_metric_plots, write_grid
except Exception:                                                  # noqa: BLE001
    try:
        from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import _save_hybrid_metric_plots as save_metric_plots
        from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import HybridLAMNrTrainer as _H
        write_grid = _H._write_grid
    except Exception:                                              # noqa: BLE001
        def save_metric_plots(csv_path, out_dir):                  # noqa: D401
            return None

        def write_grid(images, path, columns=4):
            from PIL import Image
            tiles = [Image.fromarray((np.clip(i, 0, 1) * 255).astype(np.uint8), mode="L") for i in images]
            w, h = max(t.width for t in tiles), max(t.height for t in tiles)
            canvas = Image.new("L", (columns * w, math.ceil(len(tiles) / columns) * h))
            for k, t in enumerate(tiles):
                canvas.paste(t, ((k % columns) * w, (k // columns) * h))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)

LN2 = math.log(2.0)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------ data
def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "seg":
            q = root.joinpath(*parts[i:])
            if q.exists():
                return q
    raise FileNotFoundError(path)


def load_data(a, view: dict):
    man = pd.read_csv(a.manifest)
    if a.jacket:
        man = man[man.jacket == a.jacket]
    path_col = view["path_column"]
    levels = int(view.get("dequantize", 0) or 0)
    if levels < 1:
        raise SystemExit("the view needs 'dequantize': K (number of grid levels)")
    man = man.sort_values(["pass_id", "frame"]).reset_index(drop=True)
    steps = man.groupby("pass_id").frame.diff().dropna()
    step = int(steps[steps > 0].mode().iloc[0])
    runs, meta = [], []
    for k, (pid, g) in enumerate(man.groupby("pass_id", sort=False)):
        if a.limit_passes and k >= a.limit_passes:
            break
        idx, fr = g.index.to_numpy(), g.frame.to_numpy()
        cut = [0] + [i for i in range(1, len(idx)) if fr[i] - fr[i - 1] != step] + [len(idx)]
        for s, e in zip(cut[:-1], cut[1:]):
            if e - s <= a.burn:
                continue
            row = g.iloc[s]
            runs.append(idx[s:e])
            meta.append(dict(pass_id=pid, subject_id=row.subject_id, split=str(row.get("split", "train")), speed=str(row.speed).upper(),
                             jacket=str(row.jacket), first_frame=int(fr[s]), length=int(e - s)))
    used = np.unique(np.concatenate(runs))
    remap = -np.ones(len(man), np.int64)
    remap[used] = np.arange(len(used))
    runs = [remap[r] for r in runs]
    sub = man.loc[used].reset_index(drop=True)
    root = Path(a.data_root) if a.data_root else Path(a.manifest).resolve().parent
    paths = [_resolve(p, root) for p in sub[path_col]]

    def _load(p):
        arr = np.load(p).astype(np.float32)
        return np.clip(np.rint(arr * levels), 0, levels).astype(np.uint8)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        X = np.stack(list(tqdm(ex.map(_load, paths), total=len(paths), desc="load frames", disable=a.no_progress)))
    print(f"[data] {len(runs)} runs from {sub.pass_id.nunique()} passes, {len(sub)} frames, X {X.shape} uint8 "
          f"({X.nbytes / 2**30:.2f} GiB, {time.time() - t0:.0f} s); frame step {step}; run length median {int(np.median([len(r) for r in runs]))}, "
          f"min {min(len(r) for r in runs)}, max {max(len(r) for r in runs)}")
    phase = sub["phase"].to_numpy(np.float32) if "phase" in sub else np.full(len(sub), np.nan, np.float32)
    return X, runs, pd.DataFrame(meta), sub.frame.to_numpy(), phase, levels


# ------------------------------------------------------------------ model
class _Ctx:
    """Plain holder (not a module): the context feature map of the level currently being evaluated."""
    value = None


class CtxEncoder(nn.Module):
    """m previous frames (B, m*C, H, W) -> feature maps at H/2, H/4, ..., H/2^L (one per Glow level)."""

    def __init__(self, cin, ch, L):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, ch, 3, 1, 1), nn.GELU())
        self.downs = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, 2, 1), nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU()) for _ in range(L)])

    def forward(self, c):
        h, feats = self.stem(c), []
        for d in self.downs:
            h = d(h)
            feats.append(h)
        return feats


class CondParamMap(nn.Module):
    def __init__(self, c1, ctx_ch, hidden, cout, holder):
        super().__init__()
        self.holder = [holder]                                   # list: keeps the holder out of the module tree
        self.ctx_ch = ctx_ch
        self.net = nn.Sequential(nn.Conv2d(c1 + ctx_ch, hidden, 3, 1, 1), nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, cout, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z1):
        if self.ctx_ch:
            z1 = torch.cat([z1, self.holder[0].value], 1)
        return self.net(z1)


class CondGlowBlock(nn.Module):
    """ActNorm, invertible 1x1 convolution, affine coupling whose network sees the context. encode: data -> latent; decode: latent -> data."""

    def __init__(self, C, hidden, ctx_ch, holder, s_cap):
        super().__init__()
        pm = CondParamMap(C // 2, ctx_ch, hidden, 2 * (C // 2), holder)
        self.flows = nn.ModuleList([ActNorm((C, 1, 1), log_s_cap=s_cap), Invertible1x1Conv(C, True, s_cap=s_cap),
                                    AffineCouplingBlock(pm, True, "tanh", "channel", s_cap)])
        self.clamp = 1e4

    def _bound(self, z):
        return torch.clamp(torch.nan_to_num(z, nan=0.0, posinf=self.clamp, neginf=-self.clamp), -self.clamp, self.clamp)

    def encode(self, x):
        ld = x.new_zeros(x.shape[0])
        for f in reversed(self.flows):
            x, l = f.inverse(x)
            x, ld = self._bound(x), ld + l
        return x, ld

    def decode(self, z):
        for f in self.flows:
            z, _ = f(z)
            z = self._bound(z)
        return z


class CondGlow2d(nn.Module):
    def __init__(self, C, H, W, L, K, hidden, ctx_in, ctx_ch, s_cap=2.0):
        super().__init__()
        self.C, self.H, self.W, self.L, self.D = C, H, W, L, C * H * W
        self.ctx_ch = ctx_ch if ctx_in > 0 else 0
        self.encoder = CtxEncoder(ctx_in, ctx_ch, L) if ctx_in > 0 else None
        self.holders = [_Ctx() for _ in range(L)]
        self.levels, self.q0, self.shapes = nn.ModuleList(), nn.ModuleList(), []
        c = C
        for l in range(L):
            c4, res = 4 * c, (H // 2 ** (l + 1), W // 2 ** (l + 1))
            self.levels.append(nn.ModuleList([CondGlowBlock(c4, hidden, self.ctx_ch, self.holders[l], s_cap) for _ in range(K)]))
            emit = c4 if l == L - 1 else c4 // 2
            self.shapes.append((emit,) + res)
            self.q0.append(GlowBase((emit,) + res))
            c = c4 // 2

    def _feats(self, ctx):
        if self.encoder is None:
            return [None] * self.L, None
        feats = self.encoder(ctx)
        return feats, feats[-1].mean((2, 3))

    def encode(self, x, ctx):
        feats, g = self._feats(ctx)
        zs, total = [], x.new_zeros(x.shape[0])
        for l in range(self.L):
            x = F.pixel_unshuffle(x, 2)
            self.holders[l].value = feats[l]
            for blk in self.levels[l]:
                x, ld = blk.encode(x)
                total = total + ld
            if l < self.L - 1:
                z, x = x.chunk(2, 1)
            else:
                z = x
            zs.append(z)
        return zs, total, g

    def decode(self, zs, ctx):
        feats, _ = self._feats(ctx)
        h = zs[-1]
        for l in reversed(range(self.L)):
            if l < self.L - 1:
                h = torch.cat([zs[l], h], 1)
            self.holders[l].value = feats[l]
            for blk in reversed(self.levels[l]):
                h = blk.decode(h)
            h = F.pixel_shuffle(h, 2)
        return h

    def log_prob(self, x, ctx):
        zs, ld, g = self.encode(x, ctx)
        lp = ld
        for z, q in zip(zs, self.q0):
            lp = lp + q.log_prob(z)
        return lp, g

    @torch.no_grad()
    def sample(self, n, ctx, temperature=1.0):
        zs = []
        for q in self.q0:
            q.temperature = temperature
            zs.append(q(n)[0])
            q.temperature = None
        return self.decode(zs, ctx)


def label_map(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) part fractions -> grey-level label image (0 = background)."""
    x = torch.nan_to_num(x.detach().float().cpu())
    val, arg = x.max(0)
    out = (arg.float() + 1) / x.shape[0]
    return (out * (val > 0.25)).numpy()


# ------------------------------------------------------------------ trainer
class TemporalImageTrainer:
    def setup(self, a: argparse.Namespace) -> None:
        self.a = a
        self.run_dir = Path(a.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path, self.best_path, self.metrics_path = (self.run_dir / "training_state.pt", self.run_dir / "best.pt",
                                                              self.run_dir / "metrics.csv")
        self.dev = torch.device("cuda" if a.devices == "cuda" and torch.cuda.is_available() else "mps" if a.devices == "mps" else "cpu")
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        cfg = json.load(open(a.config))
        views = [v for v in cfg["views"] if v["name"] == a.view]
        if not views:
            raise SystemExit(f"view {a.view!r} not in {a.config}")
        self.view = views[0]
        self.name = self.view["name"]
        self.X, self.seqs, self.meta, self.frames, self.phase, self.levels = load_data(a, self.view)
        self.C, (self.H, self.W) = self.X.shape[1], self.X.shape[2:]
        self.m = a.ctx_frames
        assert self.m <= a.burn, "--ctx-frames must be <= --burn"

        ck, self._resume_path = None, None
        if a.export_only:
            path = Path(a.checkpoint) if a.checkpoint else (self.state_path if a.export_from == "last" else self.best_path)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self._resume_path = path
            for k in ("L", "K", "hidden", "ctx_frames", "ctx_ch", "scale_cap", "burn", "noise_seed", "seed"):
                setattr(a, k, ck["config"][k])
            self.m = a.ctx_frames
        else:
            resume = a.resume or (str(self.state_path) if a.auto_resume and self.state_path.exists() else "")
            if resume:
                ck = torch.load(resume, map_location="cpu", weights_only=False)
                self._resume_path = Path(resume)
        split = self.meta.split.to_numpy().copy()
        if "val" not in set(split):
            subs = np.array(sorted(set(self.meta.subject_id[split == "train"])))
            vs = set(np.random.default_rng(a.seed).choice(subs, max(1, int(round(a.val_fraction * len(subs)))), replace=False))
            split = np.where(self.meta.subject_id.isin(vs), "val", split)
        self.split = split
        self.train_ix, self.val_ix = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
        self.model = CondGlow2d(self.C, self.H, self.W, a.L, a.K, a.hidden, self.m * self.C, a.ctx_ch, a.scale_cap).to(self.dev)
        self.start_iter, self.best, self.bad, self.last_val = 1, 1e9, 0, float("nan")
        self.ema = None
        # fixed validation positions
        rng = np.random.default_rng(a.seed + 7)
        pos = np.array([(s, t) for s in self.val_ix for t in range(a.burn, len(self.seqs[s]))])
        self.val_pos = pos[rng.permutation(len(pos))[: a.val_positions]] if len(pos) else pos
        self.tr_pos = np.array([(s, t) for s in self.train_ix for t in range(a.burn, len(self.seqs[s]))])

        if a.export_only:
            self.model.load_state_dict(ck["ema_models"][0] if ck.get("ema_models") else ck["models"][0])
            return
        # data-dependent ActNorm init on a first batch (before the EMA copy)
        self.model.train()
        with torch.no_grad():
            x, c = self.batch(self.tr_pos[np.random.default_rng(a.seed).integers(0, len(self.tr_pos), a.batch_size)], train=True)
            self.model.log_prob(x, c)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if ck is not None:
            self.model.load_state_dict(ck["models"][0])
            self.opt.load_state_dict(ck["optimizer"])
            self.start_iter, self.best = int(ck["iter"]), float(ck.get("best", 1e9))
        if a.ema:
            self.ema = copy.deepcopy(self.model)
            if ck is not None and ck.get("ema_models"):
                self.ema.load_state_dict(ck["ema_models"][0])
            for p in self.ema.parameters():
                p.requires_grad_(False)
        self.max_iter = a.max_iter + a.extra_iters
        (self.run_dir / "run_config.json").write_text(json.dumps({"trainer": "temporal-image", "device": str(self.dev), "arguments": vars(a),
                                                                  "views": [self.view], "levels": self.levels}, indent=2, default=str))
        summary = self._summary()
        print("\n" + summary)
        (self.run_dir / "run_config.txt").write_text(summary + "\n")
        if ck is not None:
            tqdm.write(f"[resume] from {self._resume_path} @ iter {self.start_iter}")
            if self.metrics_path.exists():
                df = pd.read_csv(self.metrics_path)
                df[df["iter"] < self.start_iter].to_csv(self.metrics_path, index=False, float_format="%.8g")
        if not self.metrics_path.exists() or self.start_iter == 1:
            self.metrics_path.write_text(f"iter,loss,align,sum_bpd,val_loss,lr,bpd_{self.name},val_bpd_{self.name}\n")

    def _summary(self) -> str:
        a = self.a
        rows = [f"[run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Py {platform.python_version()} | torch {torch.__version__} | "
                f"cuda={str(torch.cuda.is_available()).lower()} (n={torch.cuda.device_count()})",
                "[note] temporal conditional Glow on images (variant A); context = previous frames of the same pass"]

        def add(k, v):
            rows.append(f"{k:>28}: {'None' if v is None else v}")
        add("out_dir", a.out_dir)
        add("manifest / config", f"{a.manifest} / {a.config}")
        add("view / channels / size", f"{self.name} / {self.C} / {self.H}x{self.W}")
        add("dequantize levels", self.levels)
        add("jacket filter", a.jacket or "all")
        add("seed / device", f"{a.seed} / {self.dev}")
        add("batch / accum / effective", f"{a.batch_size} / {a.accum_steps} / {a.batch_size * a.accum_steps}")
        add("max_iter / extra", f"{a.max_iter} / {a.extra_iters}")
        add("start / target iteration", f"{self.start_iter} / {self.max_iter}")
        add("eval / preview interval", f"{a.eval_interval} / {a.preview_interval}")
        add("lr / warmup / min_lr", f"{a.lr} / {a.warmup_iters} / {a.min_lr}")
        add("grad_clip / weight_decay", f"{a.grad_clip} / {a.weight_decay}")
        add("ema / decay", f"{a.ema} / {a.ema_decay}")
        add("resolved checkpoint", str(self._resume_path) if self._resume_path else None)
        add("runs train / val", f"{len(self.train_ix)} / {len(self.val_ix)}  (positions {len(self.tr_pos)} / {len(self.val_pos)} used)")
        add("context frames / burn", f"{a.ctx_frames} / {a.burn}")
        add("glow L / K / hidden", f"{a.L} / {a.K} / {a.hidden}")
        add("context channels", a.ctx_ch)
        add("scale cap", a.scale_cap)
        add("parameters", f"{n_params(self.model):,}")
        add("patience (evals)", a.patience_evals)
        return "\n".join(rows)

    # ---------------- batches
    def _u(self, rows):
        out = np.empty((len(rows), self.C, self.H, self.W), np.float32)
        for i, r in enumerate(rows):
            out[i] = np.random.default_rng((zlib.crc32(str(int(r)).encode()) ^ self.a.noise_seed) & 0xFFFFFFFF).random(out.shape[1:], np.float32)
        return torch.from_numpy(out)

    def _ctx(self, pos):
        if self.m == 0:
            return None
        rows = np.stack([self.seqs[s][t - self.m:t] for s, t in pos])                      # (B, m)
        c = (torch.from_numpy(self.X[rows]).to(self.dev).float() + 0.5) / (self.levels + 1) - 0.5   # (B, m, C, H, W)
        return c.reshape(len(pos), self.m * self.C, self.H, self.W)

    def batch(self, pos, train, ctx_pos=None):
        rows = np.array([self.seqs[s][t] for s, t in pos])
        n = torch.from_numpy(self.X[rows]).to(self.dev).float()
        u = torch.rand_like(n) if train else self._u(rows).to(self.dev)
        return (n + u) / (self.levels + 1), self._ctx(pos if ctx_pos is None else ctx_pos)

    def bpd(self, model, x, c):
        lp, g = model.log_prob(x, c)
        return -lp / (self.model.D * LN2), g

    def lr_at(self, it):
        a = self.a
        if it <= a.warmup_iters:
            return a.lr * it / max(a.warmup_iters, 1)
        return max(a.min_lr, a.lr * (a.lr_decay_gamma ** ((it - 1 - a.warmup_iters) // max(a.lr_decay_steps, 1))))

    # ---------------- training
    def train(self) -> None:
        a = self.a
        rng = np.random.default_rng(a.seed + self.start_iter)
        ema_l = None
        t0 = time.time()
        pbar = tqdm(range(self.start_iter, self.max_iter + 1), desc="train-temporal-image", disable=a.no_progress,
                    initial=max(0, self.start_iter - 1), total=self.max_iter)
        for it in pbar:
            self.model.train()
            lr = self.lr_at(it)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(a.accum_steps):
                pos = self.tr_pos[rng.integers(0, len(self.tr_pos), a.batch_size)]
                x, c = self.batch(pos, train=True)
                b, _ = self.bpd(self.model, x, c)
                loss = b.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at iteration {it}")
                (loss / a.accum_steps).backward()
                tot += float(loss.detach())
            nn.utils.clip_grad_norm_(self.model.parameters(), a.grad_clip)
            self.opt.step()
            if self.ema is not None:
                with torch.no_grad():
                    sd = self.model.state_dict()
                    for k, v in self.ema.state_dict().items():
                        if v.dtype.is_floating_point:
                            v.mul_(a.ema_decay).add_(sd[k].detach(), alpha=1 - a.ema_decay)
                        else:
                            v.copy_(sd[k])
            mean = tot / a.accum_steps
            do_eval = it % a.eval_interval == 0 or it == self.max_iter
            val = float("nan")
            if do_eval:
                val = self.validate(it)
            ema_l = mean if ema_l is None else (1 - a.smooth_alpha) * ema_l + a.smooth_alpha * mean
            pbar.set_postfix(bpd=f"{ema_l:.4f}")
            with open(self.metrics_path, "a") as fh:
                fh.write(",".join(f"{v:.8g}" for v in [it, mean, 0.0, mean, val, lr, mean, self.last_val]) + "\n")
            if do_eval:
                improved = val < self.best - 1e-4
                self.best, self.bad = (val, 0) if improved else (self.best, self.bad + 1)
                self.save_checkpoint(it, improved)
                self._previews(it)
                save_metric_plots(self.metrics_path, self.run_dir)
                tqdm.write(f"[val] best bpd={self.best:.6g} ({self.bad} evaluation(s) since the best; {(time.time() - t0) / 60:.1f} min)")
                if a.patience_evals and self.bad >= a.patience_evals:
                    tqdm.write(f"[early-stop] no improvement for {self.bad} evaluations")
                    break
            gc.collect()
        pbar.close()
        self.export()

    @torch.no_grad()
    def validate(self, it: int) -> float:
        model = self.ema if self.ema is not None else self.model
        model.eval()
        vals = []
        for i in range(0, len(self.val_pos), self.a.val_bs):
            x, c = self.batch(self.val_pos[i:i + self.a.val_bs], train=False)
            vals.append(self.bpd(model, x, c)[0].cpu())
        v = float(torch.cat(vals).mean()) if vals else float("nan")
        self.last_val = v
        tqdm.write(f"[val] iter={it} loss={v:.6g} ({'EMA' if self.ema is not None else 'base'}; {self.name}={v:.4g}; "
                   f"discrete bpd {v + math.log2(self.levels + 1):.4g})")
        return v

    # ---------------- checkpoints
    def save_checkpoint(self, it: int, best: bool) -> None:
        blob = {"iter": it + 1, "models": [self.model.state_dict()], "ema_models": [self.ema.state_dict()] if self.ema is not None else None,
                "optimizer": self.opt.state_dict(), "config": vars(self.a), "best": self.best, "val_bpds": [self.last_val],
                "view": self.view, "levels": self.levels}
        torch.save(blob, self.state_path)
        torch.save(blob, self.run_dir / f"training_state_it{it:06d}.pt")
        if best:
            torch.save({**blob, "best_iter": it}, self.best_path)
            tqdm.write(f"[ckpt] saved best.pt (iter {it}, val bpd {self.best:.6g})")
        files = sorted(self.run_dir.glob("training_state_it*.pt"))
        keep = set(files[-self.a.keep_last:]) if self.a.keep_last else set()
        for p in files:
            n = int(p.stem.rsplit("it", 1)[1])
            if self.a.keep_every > 0 and n % self.a.keep_every == 0:
                keep.add(p)
        for p in files:
            if p not in keep:
                p.unlink()
        if shutil.disk_usage(self.run_dir).free / 2**30 < self.a.disk_warning_gb:
            tqdm.write(f"[disk warning] low disk space in {self.run_dir}")

    # ---------------- previews
    @torch.no_grad()
    def _previews(self, it: int) -> None:
        a = self.a
        if a.preview_interval <= 0 or it % a.preview_interval or not len(self.val_pos):
            return
        model = self.ema if self.ema is not None else self.model
        model.eval()
        pos = self.val_pos[:a.preview_samples]
        x, c = self.batch(pos, train=False)
        zs, _, _ = model.encode(x, c)
        rec = model.decode(zs, c)
        err = (rec - x).abs()
        tqdm.write(f"[recon] iter={it} view={self.name} mae={float(err.mean()):.6g} rmse={float(err.square().mean().sqrt()):.6g} "
                   f"max={float(err.max()):.6g} finite={float(torch.isfinite(rec).float().mean()):.6f} "
                   f"x_range=[{float(x.min()):.6g},{float(x.max()):.6g}] recon_range=[{float(rec.min()):.6g},{float(rec.max()):.6g}]")
        tiles = []
        for b in range(len(pos)):
            tiles += [label_map(x[b]), label_map(rec[b])]
        write_grid(tiles, self.run_dir / "previews" / f"{self.name}_recon_it{it:06d}.png", columns=2)
        if a.sample_mode != "model":
            return
        R, tiles = a.rollout_len, []
        longs = [s for s in self.val_ix if len(self.seqs[s]) >= a.burn + R][: min(4, a.preview_samples)]
        if not longs:
            return
        t0 = a.burn
        real = [[self.X[self.seqs[s][t0 + r]] for r in range(R)] for s in longs]
        frames = [((torch.from_numpy(np.stack([self.X[self.seqs[s][t0 - self.m + j]] for s in longs])).to(self.dev).float() + 0.5)
                   / (self.levels + 1) - 0.5) for j in range(self.m)]
        samples = []
        for r in range(R):
            ctx = torch.cat(frames[-self.m:], 1) if self.m else None
            xs = model.sample(len(longs), ctx, a.sample_temp)
            samples.append(xs)
            frames.append(xs - 0.5)
        for i in range(len(longs)):
            tiles += [label_map(torch.from_numpy(real[i][r].astype(np.float32) / self.levels)) for r in range(R)]
            tiles += [label_map(samples[r][i]) for r in range(R)]
        write_grid(tiles, self.run_dir / "previews" / f"{self.name}_samples_it{it:06d}.png", columns=R)

    # ---------------- export
    @torch.no_grad()
    def export(self) -> None:
        a = self.a
        if not a.export_only and self.best_path.exists():
            ck = torch.load(self.best_path, map_location=self.dev, weights_only=False)
            self.model.load_state_dict(ck["ema_models"][0] if ck.get("ema_models") else ck["models"][0])
            model = self.model
        elif a.export_only:
            model = self.model
        else:
            model = self.ema if self.ema is not None else self.model
        model.eval()
        splits = set(a.export_splits.split(","))
        ex = self.run_dir / "export"
        ex.mkdir(parents=True, exist_ok=True)
        sel = [s for s in range(len(self.seqs)) if self.split[s] in splits]
        pos = np.array([(s, t) for s in sel for t in range(a.burn, len(self.seqs[s]))])
        rng = np.random.default_rng(a.seed + 11)
        pool = pos[rng.permutation(len(pos))]
        shuf = pool[:, :]                                   # context donors: the same positions, permuted, avoiding the same pass
        bp, bs, hs = np.zeros(len(pos)), np.full(len(pos), np.nan), np.zeros((len(pos), a.ctx_ch if self.m else 1), np.float32)
        pid = self.meta.pass_id.to_numpy()
        recon_rows = []
        for i in tqdm(range(0, len(pos), a.val_bs), desc="export", disable=a.no_progress):
            p = pos[i:i + a.val_bs]
            x, c = self.batch(p, train=False)
            b, g = self.bpd(model, x, c)
            bp[i:i + len(p)] = b.cpu().numpy()
            if self.m:
                hs[i:i + len(p)] = g.cpu().numpy()
                donors = shuf[(np.arange(i, i + len(p)) * 7919 + 13) % len(shuf)].copy()
                for k in range(len(p)):                      # never a donor from the same pass
                    j = 0
                    while pid[donors[k][0]] == pid[p[k][0]] and j < 20:
                        donors[k] = shuf[rng.integers(0, len(shuf))]
                        j += 1
                _, cs = self.batch(p, train=False, ctx_pos=donors)
                bs[i:i + len(p)] = self.bpd(model, x, cs)[0].cpu().numpy()
            if a.save_recon and len(recon_rows) < a.export_max_samples:
                zs, _, _ = model.encode(x, c)
                rec = model.decode(zs, c).float().cpu().numpy()
                vdir = ex / self.name / "reconstructions"
                vdir.mkdir(parents=True, exist_ok=True)
                for k in range(len(p)):
                    if len(recon_rows) >= a.export_max_samples:
                        break
                    r = int(self.seqs[p[k][0]][p[k][1]])
                    path = vdir / f"row_{r:06d}.npy"
                    np.save(path, (rec[k] * (self.levels + 1) - 0.5).astype(np.float16))     # in grid units n
                    recon_rows.append({"row": r, "path": str(path)})
        if recon_rows:
            pd.DataFrame(recon_rows).to_csv(ex / f"{self.name}_reconstructions.csv", index=False)
        seq_ids = pos[:, 0]
        n_seq = len(sel)
        out: Dict[str, Any] = {"nll_0": np.full(n_seq, np.nan), "nll_shuf_0": np.full(n_seq, np.nan), "h_0": np.zeros((n_seq, hs.shape[1]), np.float32),
                               "length": np.array([len(self.seqs[s]) for s in sel])}
        for k, s in enumerate(sel):
            m = seq_ids == s
            out["nll_0"][k], out["nll_shuf_0"][k], out["h_0"][k] = bp[m].mean(), np.nanmean(bs[m]) if self.m else np.nan, hs[m].mean(0)
        for key in ("pass_id", "subject_id", "split", "speed", "jacket"):
            out[key] = self.meta[key].to_numpy().astype(str)[sel]
        out["view_names"] = np.array([self.name])
        out["frame_seq"] = np.array([sel.index(s) for s in seq_ids])
        out["frame_t"] = pos[:, 1]
        out["frame_number"] = np.array([self.frames[self.seqs[s][t]] for s, t in pos])
        out["frame_phase"] = np.array([self.phase[self.seqs[s][t]] for s, t in pos])
        out["frame_bpd"], out["frame_bpd_shuf"] = bp, bs
        np.savez(ex / "features.npz", **out)
        te = np.array([self.split[s] == "test" for s in sel])
        msg = f"[export] {n_seq} runs, {len(pos)} frames ({a.export_splits}) -> {ex}/features.npz"
        if te.any():
            msg += f" | test bpd {np.mean(out['nll_0'][te]):.4f}" + (f", shuffled-context {np.nanmean(out['nll_shuf_0'][te]):.4f}" if self.m else "")
        print(msg)


def _build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--view", default="seg")
    p.add_argument("--out-dir", default="runs_temporal_image")
    p.add_argument("--devices", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="")
    p.add_argument("--jacket", default="WoJ", help="keep only this jacket condition ('' = all)")
    p.add_argument("--limit-passes", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.125)
    p.add_argument("--noise-seed", type=int, default=12345)
    # model
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--K", type=int, default=6, help="Glow blocks per level")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ctx-frames", type=int, default=4, help="m previous frames as context (0 = unconditional Glow)")
    p.add_argument("--ctx-ch", type=int, default=32)
    p.add_argument("--burn", type=int, default=4, help="positions < burn are context only (never scored)")
    p.add_argument("--scale-cap", type=float, default=2.0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--extra-iters", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--val-bs", type=int, default=64)
    p.add_argument("--val-batches", type=int, default=20)                   # kept for command-line compatibility
    p.add_argument("--val-positions", type=int, default=1280)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=5e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--smooth-alpha", type=float, default=0.1)
    p.add_argument("--patience-evals", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    for name in ("--precision", "--amp-dtype", "--num-workers", "--train-samples", "--val-samples", "--plateau-factor"):
        p.add_argument(name, default=None)                                  # accepted, unused
    # previews / checkpoints / export
    p.add_argument("--preview-interval", type=int, default=1000)
    p.add_argument("--preview-samples", type=int, default=8)
    p.add_argument("--sample-mode", default="model", choices=["off", "model"])
    p.add_argument("--sample-temp", type=float, default=0.8)
    p.add_argument("--rollout-len", type=int, default=8)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--keep-every", type=int, default=5000)
    p.add_argument("--disk-warning-gb", type=float, default=10.0)
    p.add_argument("--save-recon", action="store_true")
    p.add_argument("--export-max-samples", type=int, default=100)
    p.add_argument("--export-splits", default="val,test")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--export-from", default="best", choices=["best", "last"])
    p.add_argument("--checkpoint", default="")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = _build_args(argv)
    t = TemporalImageTrainer()
    t.setup(a)
    t.export() if a.export_only else t.train()


if __name__ == "__main__":
    main()
