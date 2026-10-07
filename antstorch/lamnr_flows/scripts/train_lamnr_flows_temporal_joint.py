#!/usr/bin/env python3
"""LAMNr with temporal CONDITIONAL flows on pose AND segmentation images (option 1 of the PiqueFlows conditional-flow study).

    pose :  log p(p_t | p_{t-m..t-1})   p_t = 32 pelvis-centred AlphaPose coordinates      (conditional flow on vectors)
    seg  :  log p(x_t | x_{t-m..t-1})   x_t = 6 x 64 x 64 DensePose part fractions         (conditional Glow, variant A)

Two autoregressive conditional flows, one per view, trained on the SAME frames (position t of a pass, 15 Hz). Each flow has its own context
state of the m previous frames of its own view:
    pose : h_t = MLP(flatten of the m previous poses)                       (--ctx-dim)
    seg  : g_t = spatial mean of the last feature map of the conv pyramid over the m previous frames   (--ctx-ch)
LAMNr alignment: Projector_v(state_v) are aligned across the two views at the same position t with VICReg (ANTsTorch
LatentAlignmentLossManager; positives = the two views of the same frame, negatives = the other positions of the batch):
    loss = bpd_pose + bpd_seg + align_weight * VICReg(Projector_pose(h_t), Projector_seg(g_t)).
--align none is the control without alignment (two independent flows trained together). --ctx-frames 0 gives two unconditional flows (no
states, hence no alignment). No label (phase, speed, subject, clothes, gait parameter) enters the model.

Building blocks. ANTsNormalizingFlows: ActNorm, Invertible1x1Conv, AffineCouplingBlock, GlowBase, CoupledRationalQuadraticSpline, Permute,
LULinearPermute, ConditionalDiagGaussian, ConditionalNormalizingFlow. ANTsTorch: LatentAlignmentLossManager, Projector. The image flow is
CondGlow2d of train_lamnr_flows_temporal_image (imported); the pose flow follows ViewModel of train_lamnr_flows_temporal with an MLP context
over the m previous frames instead of a causal TCN. The same family as MoGlow (Henter et al. 2020).

Data: the Phase 1 cycle manifest (one row per frame: subject_id, split, pass_id, speed, jacket, frame, phase, seg path, pose columns k<j>x,
k<j>y) and its views JSON (seg view: path_column, dequantize = K levels; pose columns from the pose view or --pose-columns, else k0x..k16y
without the right hip). Pose is standardised with training statistics; Gaussian noise of --pose-noise SD units is added (fresh at every
training step, fixed and seeded per frame for validation/export). Seg dequantisation: x = (n + u) / (K + 1), u ~ U[0,1) when training,
seeded for validation/export, 0.5 for the context frames. bits/dim are continuous: discrete seg bpd = bpd + log2(K + 1).

Outputs (same names as the other trainers): run_config.json/txt, metrics.csv (iter,loss,align,sum_bpd,val_loss,lr,bpd_pose,bpd_seg,
val_bpd_pose,val_bpd_seg), training_state*.pt, best.pt (lowest validation sum of the two bpd, EMA weights), previews/seg_recon_it######.png,
previews/seg_samples_it######.png, objectives.png, bpd_by_view.png, val_bpd_by_view.png, export/features.npz (+ untrained.npz).
export/features.npz: per sequence nll_0 (pose), nll_1 (seg), nll_shuf_<v> (context from another pass), h_<v> (mean context state), z_<v> (mean
projector output), length, pass_id, subject_id, split, speed, jacket, view_names; per frame (positions >= burn of the exported splits)
frame_seq, frame_t, frame_number, frame_phase, frame_bpd (n,2), frame_bpd_shuf (n,2), frame_h0, frame_h1, frame_z0, frame_z1.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import re
import shutil
import sys
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

from antsnormflows.core import ConditionalNormalizingFlow
from antsnormflows.distributions.base import ConditionalDiagGaussian
from antsnormflows.flows import ActNorm, CoupledRationalQuadraticSpline, LULinearPermute, Permute
from antstorch.lamnr_flows.misc.latent_alignment import LatentAlignmentLossManager, Projector

try:
    from antstorch.lamnr_flows.scripts.train_lamnr_flows_temporal_image import (CondGlow2d, _resolve, label_map, save_metric_plots,
                                                                                  write_grid)
except ImportError:                                                  # run from the folder holding both scripts
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_lamnr_flows_temporal_image import CondGlow2d, _resolve, label_map, save_metric_plots, write_grid

LN2 = math.log(2.0)
NAMES = ["pose", "seg"]


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------ pose flow
class NoCtx(nn.Module):
    """Lets a context-free flow layer (ActNorm) sit in a ConditionalNormalizingFlow, which passes ``context=`` to every layer."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, z, context=None):
        return self.flow(z)

    def inverse(self, z, context=None):
        return self.flow.inverse(z)


class CondAffineCoupling(nn.Module):
    """Glow/MoGlow affine coupling on vectors (B, C) with a context. antsnormflows convention: forward = latent -> data, inverse = data -> latent."""

    def __init__(self, C, ctx, hidden, num_blocks, s_cap=2.0):
        super().__init__()
        self.c1, self.s_cap = (C + 1) // 2, s_cap
        layers, d = [], self.c1 + ctx
        for _ in range(max(num_blocks, 1)):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        last = nn.Linear(d, 2 * (C // 2))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def _st(self, z1, context):
        t, s = self.net(torch.cat([z1, context], -1)).chunk(2, -1)
        return t, self.s_cap * torch.tanh(s / self.s_cap)

    def forward(self, z, context=None):
        z1, z2 = z[:, :self.c1], z[:, self.c1:]
        t, s = self._st(z1, context)
        return torch.cat([z1, z2 * torch.exp(s) + t], -1), s.sum(-1)

    def inverse(self, x, context=None):
        x1, x2 = x[:, :self.c1], x[:, self.c1:]
        t, s = self._st(x1, context)
        return torch.cat([x1, (x2 - t) * torch.exp(-s)], -1), -s.sum(-1)


class PoseFlow(nn.Module):
    """p(p_t | m previous poses): MLP context state + conditional flow (rational-quadratic spline or MoGlow-style affine couplings)."""

    def __init__(self, C, m, a):
        super().__init__()
        self.C, self.m, self.ctx_dim = C, m, a.ctx_dim
        self.enc = nn.Sequential(nn.Linear(m * C, a.ctx_dim), nn.GELU(), nn.Linear(a.ctx_dim, a.ctx_dim)) if m > 0 else None
        flows = []
        for j in range(a.pose_K):
            if a.coupling == "affine":
                flows += [NoCtx(ActNorm((C,))), LULinearPermute(C), CondAffineCoupling(C, a.ctx_dim, a.pose_hidden, a.num_blocks)]
            else:
                flows.append(CoupledRationalQuadraticSpline(C, a.num_blocks, a.pose_hidden, num_context_channels=a.ctx_dim, num_bins=a.num_bins,
                                                            tail_bound=a.tail_bound, reverse_mask=bool(j % 2)))
                flows.append(Permute(C, mode="shuffle"))
        base = nn.Linear(a.ctx_dim, 2 * C)
        nn.init.zeros_(base.weight)
        nn.init.zeros_(base.bias)
        self.flow = ConditionalNormalizingFlow(ConditionalDiagGaussian(C, base), flows)

    def state(self, ctx, B, device):
        return self.enc(ctx) if self.enc is not None else torch.zeros(B, self.ctx_dim, device=device)

    def log_prob(self, x, ctx):
        h = self.state(ctx, x.shape[0], x.device)
        return self.flow.log_prob(x, context=h), h

    @torch.no_grad()
    def roundtrip(self, x, ctx):
        h = self.state(ctx, x.shape[0], x.device)
        z = x
        for f in reversed(self.flow.flows):
            z, _ = f.inverse(z, context=h)
        for f in self.flow.flows:
            z, _ = f(z, context=h)
        return z


class Joint(nn.Module):
    def __init__(self, a, Cp, C, H, W, m):
        super().__init__()
        self.pose = PoseFlow(Cp, m, a)
        self.seg = CondGlow2d(C, H, W, a.L, a.K, a.hidden, m * C, a.ctx_ch, a.scale_cap)
        self.proj = nn.ModuleList([Projector(a.ctx_dim, a.proj_hidden, a.proj_dim),
                                   Projector(a.ctx_ch if m > 0 else 1, a.proj_hidden, a.proj_dim)])


# ------------------------------------------------------------------ data
def pose_columns(man: pd.DataFrame, cfg_view: Optional[dict], explicit: str) -> List[str]:
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
    else:
        cols = None
        for key in ("columns", "feature_columns", "features", "keypoint_columns"):
            if cfg_view and isinstance(cfg_view.get(key), list) and cfg_view[key]:
                cols = list(cfg_view[key])
                break
        if cols is None:
            found = sorted([c for c in man.columns if re.fullmatch(r"k\d+[xy]", c)], key=lambda c: (int(c[1:-1]), c[-1]))
            if len(found) == 34:                                        # COCO 17 keypoints: drop the right hip (x_rhip = -x_lhip once centred)
                found = [c for c in found if c not in ("k12x", "k12y")]
            cols = found
    missing = [c for c in cols if c not in man.columns]
    if missing or not cols:
        raise SystemExit(f"pose columns not found in the manifest: {missing or 'none resolved'} (use --pose-columns a,b,c)")
    return cols


def load_data(a, view: dict, pose_view: Optional[dict]):
    man = pd.read_csv(a.manifest)
    if a.jacket:
        man = man[man.jacket == a.jacket]
    path_col = view["path_column"]
    levels = int(view.get("dequantize", 0) or 0)
    if levels < 1:
        raise SystemExit("the seg view needs 'dequantize': K (number of grid levels)")
    pcols = pose_columns(man, pose_view, a.pose_columns)
    man = man.dropna(subset=pcols).sort_values(["pass_id", "frame"]).reset_index(drop=True)
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
    P = sub[pcols].to_numpy(np.float32)
    print(f"[data] {len(runs)} runs from {sub.pass_id.nunique()} passes, {len(sub)} frames, X {X.shape} uint8 ({X.nbytes / 2**30:.2f} GiB, "
          f"{time.time() - t0:.0f} s); pose {P.shape} from {len(pcols)} columns ({pcols[0]} .. {pcols[-1]}); frame step {step}; "
          f"run length median {int(np.median([len(r) for r in runs]))}, min {min(len(r) for r in runs)}, max {max(len(r) for r in runs)}")
    phase = sub["phase"].to_numpy(np.float32) if "phase" in sub else np.full(len(sub), np.nan, np.float32)
    return X, P, pcols, runs, pd.DataFrame(meta), sub.frame.to_numpy(), phase, levels


# ------------------------------------------------------------------ trainer
class JointTrainer:
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
        views = {v["name"]: v for v in cfg["views"]}
        if a.view not in views:
            raise SystemExit(f"view {a.view!r} not in {a.config}")
        self.view, pose_view = views[a.view], views.get(a.pose_view)
        self.X, P, self.pcols, self.seqs, self.meta, self.frames, self.phase, self.levels = load_data(a, self.view, pose_view)
        self.C, (self.H, self.W), self.Cp = self.X.shape[1], self.X.shape[2:], P.shape[1]
        self.m = a.ctx_frames
        assert self.m <= a.burn, "--ctx-frames must be <= --burn"

        ck, self._resume_path = None, None
        if a.export_only:
            path = Path(a.checkpoint) if a.checkpoint else (self.state_path if a.export_from == "last" else self.best_path)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self._resume_path = path
            for k in ("L", "K", "hidden", "ctx_frames", "ctx_ch", "scale_cap", "burn", "noise_seed", "seed", "ctx_dim", "pose_K", "pose_hidden",
                      "num_blocks", "num_bins", "tail_bound", "coupling", "proj_hidden", "proj_dim", "pose_noise"):
                if k in ck["config"]:
                    setattr(a, k, ck["config"][k])
            self.m = a.ctx_frames
            self.norm = ck["norm"]
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
        if not a.export_only:
            tr_rows = np.unique(np.concatenate([self.seqs[s] for s in self.train_ix]))
            self.norm = {"mu": P[tr_rows].mean(0, keepdims=True), "sd": P[tr_rows].std(0, keepdims=True) + 1e-6}
            if (self.norm["sd"] < 1e-4).any():
                print("[warn] near-constant pose columns:", [self.pcols[i] for i in np.flatnonzero(self.norm["sd"][0] < 1e-4)])
        self.P = ((P - self.norm["mu"]) / self.norm["sd"]).astype(np.float32)
        fixed = np.stack([np.random.default_rng((zlib.crc32(str(i).encode()) ^ a.noise_seed) & 0xFFFFFFFF).standard_normal(self.Cp)
                          for i in range(len(self.P))]).astype(np.float32)
        self.Pn = self.P + a.pose_noise * fixed                                  # fixed noisy copy for validation / export

        self.model = Joint(a, self.Cp, self.C, self.H, self.W, self.m).to(self.dev)
        self.D_img = self.model.seg.D
        self.start_iter, self.best, self.bad = 1, 1e9, 0
        self.last_val = [float("nan")] * 2
        self.ema = None
        rng = np.random.default_rng(a.seed + 7)
        pos = np.array([(s, t) for s in self.val_ix for t in range(a.burn, len(self.seqs[s]))])
        self.val_pos = pos[rng.permutation(len(pos))[: a.val_positions]] if len(pos) else pos
        self.tr_pos = np.array([(s, t) for s in self.train_ix for t in range(a.burn, len(self.seqs[s]))])

        if a.export_only:
            if a.untrained:
                torch.manual_seed(a.seed + 100)
                self.model = Joint(a, self.Cp, self.C, self.H, self.W, self.m).to(self.dev)
            else:
                self.model.load_state_dict(ck["ema_models"][0] if ck.get("ema_models") else ck["models"][0])
            return
        if a.align != "none" and self.m == 0:
            print("[warn] --ctx-frames 0 has no context states: alignment switched off")
            a.align = "none"
        self.model.train()                                                       # data-dependent ActNorm init, before the EMA copy
        with torch.no_grad():
            b = self.batch(self.tr_pos[np.random.default_rng(a.seed).integers(0, len(self.tr_pos), a.batch_size)], train=True)
            self._nll(self.model, b)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if ck is not None:
            self.model.load_state_dict(ck["models"][0])
            self.opt.load_state_dict(ck["optimizer"])
            self.start_iter, self.best = int(ck["iter"]), float(ck.get("best", 1e9))
            self.norm = ck["norm"]
        if a.ema:
            self.ema = copy.deepcopy(self.model)
            if ck is not None and ck.get("ema_models"):
                self.ema.load_state_dict(ck["ema_models"][0])
            for p in self.ema.parameters():
                p.requires_grad_(False)
        self.align_mgr = LatentAlignmentLossManager(a, self.model.proj, self.dev)
        self.max_iter = a.max_iter + a.extra_iters
        (self.run_dir / "run_config.json").write_text(json.dumps({"trainer": "temporal-joint", "device": str(self.dev), "arguments": vars(a),
                                                                  "views": [self.view], "pose_columns": self.pcols, "levels": self.levels},
                                                                 indent=2, default=str))
        summary = self._summary()
        print("\n" + summary)
        (self.run_dir / "run_config.txt").write_text(summary + "\n")
        if ck is not None:
            tqdm.write(f"[resume] from {self._resume_path} @ iter {self.start_iter}")
            if self.metrics_path.exists():
                df = pd.read_csv(self.metrics_path)
                df[df["iter"] < self.start_iter].to_csv(self.metrics_path, index=False, float_format="%.8g")
        if not self.metrics_path.exists() or self.start_iter == 1:
            self.metrics_path.write_text("iter,loss,align,sum_bpd,val_loss,lr,bpd_pose,bpd_seg,val_bpd_pose,val_bpd_seg\n")

    def _summary(self) -> str:
        a = self.a
        rows = [f"[run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Py {platform.python_version()} | torch {torch.__version__} | "
                f"cuda={str(torch.cuda.is_available()).lower()} (n={torch.cuda.device_count()})",
                "[note] temporal conditional flows on pose + seg images aligned by VICReg on the context states (option 1)"]

        def add(k, v):
            rows.append(f"{k:>28}: {'None' if v is None else v}")
        add("out_dir", a.out_dir)
        add("manifest / config", f"{a.manifest} / {a.config}")
        add("seg view / channels / size", f"{a.view} / {self.C} / {self.H}x{self.W}  (K = {self.levels})")
        add("pose view / dimensions", f"{a.pose_view} / {self.Cp}")
        add("jacket filter", a.jacket or "all")
        add("seed / device", f"{a.seed} / {self.dev}")
        add("batch / accum / effective", f"{a.batch_size} / {a.accum_steps} / {a.batch_size * a.accum_steps}")
        add("max_iter / extra", f"{a.max_iter} / {a.extra_iters}")
        add("start / target iteration", f"{self.start_iter} / {self.max_iter}")
        add("eval / preview interval", f"{a.eval_interval} / {a.preview_interval}")
        add("lr / warmup / min_lr", f"{a.lr} / {a.warmup_iters} / {a.min_lr}")
        add("lr decay gamma / steps", f"{a.lr_decay_gamma} / {a.lr_decay_steps}")
        add("grad_clip / weight_decay", f"{a.grad_clip} / {a.weight_decay}")
        add("ema / decay", f"{a.ema} / {a.ema_decay}")
        add("resolved checkpoint", str(self._resume_path) if self._resume_path else None)
        add("runs train / val", f"{len(self.train_ix)} / {len(self.val_ix)}  (positions {len(self.tr_pos)} / {len(self.val_pos)} used)")
        add("context frames / burn", f"{a.ctx_frames} / {a.burn}")
        add("seg glow L / K / hidden", f"{a.L} / {a.K} / {a.hidden}  (ctx channels {a.ctx_ch}, scale cap {a.scale_cap})")
        add("pose coupling / K / hidden", f"{a.coupling} / {a.pose_K} / {a.pose_hidden}  (ctx dim {a.ctx_dim}, blocks {a.num_blocks}, bins {a.num_bins})")
        add("pose noise (SD units)", a.pose_noise)
        add("alignment / weight / warmup", f"{a.align} / {a.align_weight} / {a.align_warmup}")
        add("projector dim / hidden", f"{a.proj_dim} / {a.proj_hidden}")
        add("parameters pose / seg", f"{n_params(self.model.pose):,} / {n_params(self.model.seg):,}")
        add("patience (evals)", a.patience_evals)
        return "\n".join(rows)

    # ---------------- batches
    def _u(self, rows):
        out = np.empty((len(rows), self.C, self.H, self.W), np.float32)
        for i, r in enumerate(rows):
            out[i] = np.random.default_rng((zlib.crc32(str(int(r)).encode()) ^ self.a.noise_seed) & 0xFFFFFFFF).random(out.shape[1:], np.float32)
        return torch.from_numpy(out)

    def batch(self, pos, train, ctx_pos=None):
        cpos = pos if ctx_pos is None else ctx_pos
        rows = np.array([self.seqs[s][t] for s, t in pos])
        n = torch.from_numpy(self.X[rows]).to(self.dev).float()
        u = torch.rand_like(n) if train else self._u(rows).to(self.dev)
        xi = (n + u) / (self.levels + 1)
        P = self.P if train else self.Pn
        xp = torch.from_numpy(P[rows]).to(self.dev)
        if train:
            xp = xp + self.a.pose_noise * torch.randn_like(xp)
        ci = cp = None
        if self.m:
            crow = np.stack([self.seqs[s][t - self.m:t] for s, t in cpos])                         # (B, m)
            ci = ((torch.from_numpy(self.X[crow]).to(self.dev).float() + 0.5) / (self.levels + 1) - 0.5).reshape(len(pos), self.m * self.C, self.H, self.W)
            cpn = torch.from_numpy(P[crow]).to(self.dev)
            if train:
                cpn = cpn + self.a.pose_noise * torch.randn_like(cpn)
            cp = cpn.reshape(len(pos), self.m * self.Cp)
        return dict(xi=xi, ci=ci, xp=xp, cp=cp)

    def _nll(self, model, b):
        lp_i, g = model.seg.log_prob(b["xi"], b["ci"])
        lp_p, h = model.pose.log_prob(b["xp"], b["cp"])
        return -lp_p / (self.Cp * LN2), -lp_i / (self.D_img * LN2), h, g

    def _loss(self, model, b, it, mgr):
        bp, bi, h, g = self._nll(model, b)
        L_nll = bp.mean() + bi.mean()
        if self.m and self.a.align != "none":
            loss, align, _, _ = mgr.compute([h, g], L_nll, it, None, None)
        else:
            loss, align = L_nll, torch.zeros((), device=self.dev)
        return loss, align, float(bp.mean().detach()), float(bi.mean().detach())

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
        pbar = tqdm(range(self.start_iter, self.max_iter + 1), desc="train-joint", disable=a.no_progress,
                    initial=max(0, self.start_iter - 1), total=self.max_iter)
        for it in pbar:
            self.model.train()
            lr = self.lr_at(it)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            tot = tal = tbp = tbi = 0.0
            for _ in range(a.accum_steps):
                pos = self.tr_pos[rng.integers(0, len(self.tr_pos), a.batch_size)]
                loss, align, bp, bi = self._loss(self.model, self.batch(pos, train=True), it, self.align_mgr)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at iteration {it}")
                (loss / a.accum_steps).backward()
                tot += float(loss.detach())
                tal += float(align.detach())
                tbp += bp
                tbi += bi
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
            n = float(a.accum_steps)
            loss_m, align_m, bp_m, bi_m = tot / n, tal / n, tbp / n, tbi / n
            do_eval = it % a.eval_interval == 0 or it == self.max_iter
            val = float("nan")
            if do_eval:
                val = self.validate(it)
            ema_l = loss_m if ema_l is None else (1 - a.smooth_alpha) * ema_l + a.smooth_alpha * loss_m
            pbar.set_postfix(loss=f"{ema_l:.4f}", align=f"{align_m:.3f}", pose=f"{bp_m:.3f}", seg=f"{bi_m:.3f}")
            with open(self.metrics_path, "a") as fh:
                fh.write(",".join(f"{v:.8g}" for v in [it, loss_m, align_m, bp_m + bi_m, val, lr, bp_m, bi_m, *self.last_val]) + "\n")
            if do_eval:
                vsum = sum(self.last_val)
                improved = vsum < self.best - 1e-4
                self.best, self.bad = (vsum, 0) if improved else (self.best, self.bad + 1)
                self.save_checkpoint(it, improved)
                self._previews(it)
                save_metric_plots(self.metrics_path, self.run_dir)
                tqdm.write(f"[val] best sum_bpd={self.best:.6g} ({self.bad} evaluation(s) since the best; {(time.time() - t0) / 60:.1f} min)")
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
        bps, bis, als = [], [], []
        for i in range(0, len(self.val_pos), self.a.val_bs):
            b = self.batch(self.val_pos[i:i + self.a.val_bs], train=False)
            bp, bi, h, g = self._nll(model, b)
            bps.append(bp.cpu())
            bis.append(bi.cpu())
            if self.m and self.a.align != "none" and len(bp) > 3:
                mgr = LatentAlignmentLossManager(self.a, model.proj, self.dev)                       # projectors of the evaluated (EMA) model
                als.append(float(mgr.compute([h, g], bp.mean() + bi.mean(), 10 ** 9, None, None)[1]))
        vp, vi = (float(torch.cat(bps).mean()), float(torch.cat(bis).mean())) if bps else (float("nan"), float("nan"))
        self.last_val = [vp, vi]
        v = vp + vi + (self.a.align_weight * float(np.mean(als)) if als else 0.0)
        tqdm.write(f"[val] iter={it} loss={v:.6g} ({'EMA' if self.ema is not None else 'base'}; pose={vp:.4g}, seg={vi:.4g} "
                   f"(discrete {vi + math.log2(self.levels + 1):.4g}); align={float(np.mean(als)) if als else 0.0:.4g})")
        return v

    # ---------------- checkpoints
    def save_checkpoint(self, it: int, best: bool) -> None:
        blob = {"iter": it + 1, "models": [self.model.state_dict()], "ema_models": [self.ema.state_dict()] if self.ema is not None else None,
                "optimizer": self.opt.state_dict(), "config": vars(self.a), "best": self.best, "val_bpds": list(self.last_val),
                "view": self.view, "levels": self.levels, "norm": self.norm, "pose_columns": self.pcols}
        torch.save(blob, self.state_path)
        torch.save(blob, self.run_dir / f"training_state_it{it:06d}.pt")
        if best:
            torch.save({**blob, "best_iter": it}, self.best_path)
            tqdm.write(f"[ckpt] saved best.pt (iter {it}, val sum_bpd {self.best:.6g})")
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
        b = self.batch(pos, train=False)
        zs, _, _ = model.seg.encode(b["xi"], b["ci"])
        rec = model.seg.decode(zs, b["ci"])
        err = (rec - b["xi"]).abs()
        perr = (model.pose.roundtrip(b["xp"], b["cp"]) - b["xp"]).abs()
        tqdm.write(f"[recon] iter={it} seg mae={float(err.mean()):.6g} rmse={float(err.square().mean().sqrt()):.6g} max={float(err.max()):.6g} | "
                   f"pose round-trip max abs error {float(perr.max()):.3g} (SD units)")
        tiles = []
        for k in range(len(pos)):
            tiles += [label_map(b["xi"][k]), label_map(rec[k])]
        write_grid(tiles, self.run_dir / "previews" / f"seg_recon_it{it:06d}.png", columns=2)
        if a.sample_mode != "model" or self.m == 0:
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
            xs = model.seg.sample(len(longs), torch.cat(frames[-self.m:], 1), a.sample_temp)
            samples.append(xs)
            frames.append(xs - 0.5)
        for i in range(len(longs)):
            tiles += [label_map(torch.from_numpy(real[i][r].astype(np.float32) / self.levels)) for r in range(R)]
            tiles += [label_map(samples[r][i]) for r in range(R)]
        write_grid(tiles, self.run_dir / "previews" / f"seg_samples_it{it:06d}.png", columns=R)

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
        tag = a.export_tag or ("untrained" if a.untrained else "features")
        splits = set(a.export_splits.split(","))
        ex = self.run_dir / "export"
        ex.mkdir(parents=True, exist_ok=True)
        sel = [s for s in range(len(self.seqs)) if self.split[s] in splits]
        pos = np.array([(s, t) for s in sel for t in range(a.burn, len(self.seqs[s]))])
        rng = np.random.default_rng(a.seed + 11)
        pool = pos[rng.permutation(len(pos))]
        N = len(pos)
        dims = (a.ctx_dim, a.ctx_ch if self.m else 1)
        bp, bs = np.zeros((N, 2)), np.full((N, 2), np.nan)
        Hs = [np.zeros((N, d), np.float32) for d in dims]
        Zs = [np.zeros((N, a.proj_dim), np.float32) for _ in dims]
        pid = self.meta.pass_id.to_numpy()
        recon_rows = []
        for i in tqdm(range(0, N, a.val_bs), desc="export", disable=a.no_progress):
            p = pos[i:i + a.val_bs]
            b = self.batch(p, train=False)
            bpp, bpi, h, g = self._nll(model, b)
            sl = slice(i, i + len(p))
            bp[sl, 0], bp[sl, 1] = bpp.cpu().numpy(), bpi.cpu().numpy()
            for v, st in enumerate((h, g)):
                st = st if st is not None else torch.zeros(len(p), 1, device=self.dev)
                Hs[v][sl] = st.cpu().numpy()
                Zs[v][sl] = model.proj[v](st).cpu().numpy()
            if self.m:
                donors = pool[(np.arange(i, i + len(p)) * 7919 + 13) % len(pool)].copy()
                for k in range(len(p)):
                    j = 0
                    while pid[donors[k][0]] == pid[p[k][0]] and j < 20:
                        donors[k] = pool[rng.integers(0, len(pool))]
                        j += 1
                bs_ = self.batch(p, train=False, ctx_pos=donors)
                sp, si, _, _ = self._nll(model, bs_)
                bs[sl, 0], bs[sl, 1] = sp.cpu().numpy(), si.cpu().numpy()
            if a.save_recon and not a.untrained and len(recon_rows) < a.export_max_samples:
                zs, _, _ = model.seg.encode(b["xi"], b["ci"])
                rec = model.seg.decode(zs, b["ci"]).float().cpu().numpy()
                vdir = ex / "seg" / "reconstructions"
                vdir.mkdir(parents=True, exist_ok=True)
                for k in range(len(p)):
                    if len(recon_rows) >= a.export_max_samples:
                        break
                    r = int(self.seqs[p[k][0]][p[k][1]])
                    path = vdir / f"row_{r:06d}.npy"
                    np.save(path, (rec[k] * (self.levels + 1) - 0.5).astype(np.float16))
                    recon_rows.append({"row": r, "path": str(path)})
        if recon_rows:
            pd.DataFrame(recon_rows).to_csv(ex / "seg_reconstructions.csv", index=False)
        seq_ids, n_seq = pos[:, 0], len(sel)
        out: Dict[str, Any] = {"length": np.array([len(self.seqs[s]) for s in sel])}
        for v in range(2):
            out[f"nll_{v}"], out[f"nll_shuf_{v}"] = np.full(n_seq, np.nan), np.full(n_seq, np.nan)
            out[f"h_{v}"], out[f"z_{v}"] = np.zeros((n_seq, dims[v]), np.float32), np.zeros((n_seq, a.proj_dim), np.float32)
        for k, s in enumerate(sel):
            msk = seq_ids == s
            for v in range(2):
                out[f"nll_{v}"][k] = bp[msk, v].mean()
                out[f"nll_shuf_{v}"][k] = np.nanmean(bs[msk, v]) if self.m else np.nan
                out[f"h_{v}"][k], out[f"z_{v}"][k] = Hs[v][msk].mean(0), Zs[v][msk].mean(0)
        for key in ("pass_id", "subject_id", "split", "speed", "jacket"):
            out[key] = self.meta[key].to_numpy().astype(str)[sel]
        out["view_names"] = np.array(NAMES)
        smap = {s: k for k, s in enumerate(sel)}
        out["frame_seq"] = np.array([smap[s] for s in seq_ids])
        out["frame_t"] = pos[:, 1]
        out["frame_number"] = np.array([self.frames[self.seqs[s][t]] for s, t in pos])
        out["frame_phase"] = np.array([self.phase[self.seqs[s][t]] for s, t in pos])
        out["frame_bpd"], out["frame_bpd_shuf"] = bp, bs
        for v in range(2):
            out[f"frame_h{v}"], out[f"frame_z{v}"] = Hs[v], Zs[v]
        np.savez(ex / f"{tag}.npz", **out)
        te = np.array([self.split[s] == "test" for s in sel])
        msg = f"[export] {n_seq} runs, {N} frames ({a.export_splits}) -> {ex}/{tag}.npz"
        if te.any():
            msg += f" | test bpd pose {np.mean(out['nll_0'][te]):.4f}, seg {np.mean(out['nll_1'][te]):.4f}"
            if self.m:
                msg += f" | shuffled-context pose {np.nanmean(out['nll_shuf_0'][te]):.4f}, seg {np.nanmean(out['nll_shuf_1'][te]):.4f}"
        print(msg)


def _build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--view", default="seg", help="name of the image view in the config")
    p.add_argument("--pose-view", default="pose", help="name of the pose view in the config (columns read from it when it lists them)")
    p.add_argument("--pose-columns", default="", help="comma-separated pose columns of the manifest (default: view columns, else k0x..k16y without the right hip)")
    p.add_argument("--out-dir", default="runs_temporal_joint")
    p.add_argument("--devices", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="")
    p.add_argument("--jacket", default="WoJ", help="keep only this jacket condition ('' = all)")
    p.add_argument("--limit-passes", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.125)
    p.add_argument("--noise-seed", type=int, default=12345)
    # image flow
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--K", type=int, default=6, help="Glow blocks per level (image flow)")
    p.add_argument("--hidden", type=int, default=128, help="hidden channels of the image coupling networks")
    p.add_argument("--ctx-frames", type=int, default=4, help="m previous frames as context for both views (0 = unconditional, no alignment)")
    p.add_argument("--ctx-ch", type=int, default=32)
    p.add_argument("--burn", type=int, default=4, help="positions < burn are context only (never scored)")
    p.add_argument("--scale-cap", type=float, default=2.0)
    # pose flow
    p.add_argument("--coupling", default="spline", choices=["spline", "affine"])
    p.add_argument("--pose-K", type=int, default=4)
    p.add_argument("--pose-hidden", type=int, default=64)
    p.add_argument("--ctx-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=4.0)
    p.add_argument("--pose-noise", type=float, default=0.05, help="Gaussian noise in SD units (fresh when training, fixed for validation/export)")
    # optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument("--extra-iters", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--val-bs", type=int, default=64)
    p.add_argument("--val-batches", type=int, default=20)                   # kept for command-line compatibility
    p.add_argument("--val-positions", type=int, default=1280)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=5e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=5000)
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
    # alignment (same names and defaults as the temporal trainer)
    p.add_argument("--align", default="vicreg", choices=["vicreg", "barlow", "infonce", "hsic", "pearson", "mse", "none"])
    p.add_argument("--align-weight", type=float, default=0.05)
    p.add_argument("--align-warmup", type=int, default=200)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=128)
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
    p.add_argument("--untrained", action="store_true", help="--export-only: random weights of the same architecture (seed + 100)")
    p.add_argument("--export-tag", default="")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = _build_args(argv)
    t = JointTrainer()
    t.setup(a)
    t.export() if a.export_only else t.train()


if __name__ == "__main__":
    main()
