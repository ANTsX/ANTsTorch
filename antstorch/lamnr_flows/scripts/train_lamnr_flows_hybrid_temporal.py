"""Hybrid TEMPORAL LAMNr trainer: autoregressive conditional flows for tabular, signal1d and image2d views.

Temporal counterpart of ``train_lamnr_flows_hybrid``: same manifest + JSON view configuration, same command-line names, same outputs. Each
view is an autoregressive conditional flow over time,

    log p(x_t | x_{t-m}, ..., x_{t-1})        (one flow per view, trained on the same positions t of the same sequences),

and the views are aligned through their *context states* (the summary of the past each flow conditions on) with any of the LAMNr
alignment losses (``--align vicreg|barlow|infonce|hsic|pearson|mse|none``, same options as the hybrid trainer). Nothing here is specific
to one application: the views, their columns/paths and their names come from the configuration; the time structure from the manifest.

View types (``HybridViewSpec`` of the hybrid trainer; same JSON keys)::

    {"subject_column": "subject_id",
     "views": [
       {"name": "keypoints", "type": "tabular", "columns": ["k0x", "k0y", ...], "normalization": "0mean"},
       {"name": "masks", "type": "image2d", "path_column": "seg", "channels": 6, "shape": [64, 64],
        "intensity": "none", "dequantize": 49, "display": "labels"},
       {"name": "pcs", "type": "signal1d", "path_column": "window_file", "shape": [64, 20], "layout": "CL",
        "time_column": "start_frame", "normalization": "0mean"}]}

* ``tabular``  one row = one time step: a vector of ``columns``.  Context: window of the m previous rows (``--arch mlp|long|mid|short|lstm``).
* ``image2d``  one row = one time step: an image file (``path_column``; ``intensity`` none/0mean/to01, ``dequantize`` as in the hybrid
  trainer). Flow: conditional multiscale Glow (squeeze, ActNorm, invertible 1x1 convolution, affine coupling, split), whose coupling networks
  see a convolutional pyramid over the m previous images (MoGlow-like, for images). ``"display": "labels"`` draws multichannel masks as
  a label map in the previews.
* ``signal1d`` rows are overlapping windows (``(channels, length)`` files, ``time_column`` = start of the window); they are stitched back
  into per-sequence streams of channel vectors (hop inferred from the overlap, which is verified) and modelled like tabular steps.
``image3d`` is not supported. Per-view overrides go in the view's ``"model"`` block: ``ctx_frames``, ``arch``, ``ctx_dim``, ``K``, ``hidden``,
``coupling``, ``num_blocks``, ``num_bins``, ``tail_bound`` (vector views); ``L``, ``K``, ``hidden``, ``ctx_ch``, ``scale_cap`` (images).

Sequences: ``--sequence-column`` (default pass_id) and ``--time-column`` (default frame); rows of a sequence are cut into runs of constant
time step; positions t >= ``--burn`` of every run are training examples (the first ``burn`` steps are context only, so models with different
context lengths score the same positions; ``burn`` must be >= every view's ``ctx_frames`` and defaults to their maximum). Rows lacking any
view (NaN columns, missing files) are dropped before the runs are built. Splits: the manifest ``--split-column`` (train/val/test; test is
never trained on and is exported with val), else a subject-wise split with ``--val-fraction``; if the column has no ``val`` the validation
set is carved subject-wise out of train. ``--filter col=value[,value]`` (repeatable) restricts the manifest.

Likelihood: bits per dimension of the scored positions (continuous). Image views on a grid {0, 1/K, ..., 1} use ``"dequantize": K``:
x = (n + u) / (K + 1) with u ~ U[0,1) in training and, by default, seeded per frame in validation/export (``--dequantize-eval
midpoint`` gives the hybrid trainer's u = 0.5); context frames always use u = 0.5. Discrete bpd = continuous + log2(K + 1). Tabular and
signal targets get Gaussian noise (``--tabular-noise-std``, or the view's ``augmentation`` key ``tabular_noise_std`` / ``noise_std``) in training only;
validation and export are clean. No label enters the model. The hybrid image augmentation and the signal roll are not applied.

Outputs (same names as train_lamnr_flows_hybrid; extras marked +):
    run_config.json / run_config.txt, metrics.csv (iter,loss,align,sum_bpd,val_loss,lr,bpd_<view>...,val_bpd_<view>...), training_state.pt and
    training_state_it######.pt (--keep-last / --keep-every), previews/<view>_recon_it######.png and <view>_samples_it######.png (image views;
    samples are rollouts from the validation context), objectives.png, bpd_by_view.png, val_bpd_by_view.png,
    export/<view>_latents.csv / _whitened.csv / _reconstructions.csv (--save-z / --save-whitened / --save-recon; vector views), export/<view>/reconstructions/row_*.npy
    + export/<view>_reconstructions.csv (image views),
  + best.pt (EMA weights of the lowest validation sum of bpd; used for the export unless --export-from last),
  + export/features.npz and, with --export-only --untrained, export/untrained.npz: per sequence nll_<v>, nll_shuf_<v> (context taken from another
    sequence: information of the right context = nll_shuf - nll), h_<v> (mean context state), z_<v> (mean projector output), length, pass_id,
    subject_id, split, <annotation columns>, view_names; per frame (positions >= burn of the exported splits) frame_seq, frame_t, frame_number,
    frame_<annotation>, frame_bpd (n, V), frame_bpd_shuf, frame_h<v>, frame_z<v>. ``--annotations col,...`` picks the manifest columns exported.

Alignment latents are the context states (``--alignment-latents context``, the only choice: the latents of a conditional flow are
noise given the context). The projectors always exist; with ``--align none`` they stay at initialisation, which gives the untrained-projector
control for the retrieval/read-out analyses.

Not implemented / accepted only so that hybrid command lines run unchanged: DDP and DataParallel (single device), image/signal augmentation,
--train-samples, --num-workers, --alignment-pool-size, --grad-checkpoint, --augmentation-*, --aug-schedules, --horizontal-flip-probability,
--image-noise-std, --image-base (only glow). ``--precision`` defaults to float here (mixed is implemented, not validated for these flows).

Run with::

    python -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid_temporal \
        --manifest manifest.csv --config views.json --filter jacket=WoJ --ctx-frames 4 --align vicreg --out-dir runs_temporal
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import shutil
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from antsnormflows.core import ConditionalNormalizingFlow
from antsnormflows.distributions.base import ConditionalDiagGaussian, GlowBase, _clamp_log_scale
from antsnormflows.flows import (ActNorm, AffineCouplingBlock, CoupledRationalQuadraticSpline, Invertible1x1Conv, LULinearPermute,
                                 Permute)

from antstorch.lamnr_flows.core.train_lamnr_glow_base import make_warmup, n_params, set_deterministic
from antstorch.lamnr_flows.misc.channel_normalizer import ChannelNormalizer
from antstorch.lamnr_flows.misc.latent_alignment import LatentAlignmentLossManager, Projector
from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import (HybridLAMNrTrainer, HybridViewSpec, _is_present, _load_image,
                                                                    _load_signal, _save_hybrid_metric_plots)
from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import TabularNormalizer

LN2 = math.log(2.0)
ARCHS = {"long": (3, (1, 2, 4)), "mid": (3, (1, 2)), "short": (2, (1,))}       # causal TCN (kernel, dilations): receptive field 15 / 7 / 2 frames
CONTEXT_ARCHS = ["mlp", "lstm", *ARCHS]
write_grid, display_slice = HybridLAMNrTrainer._write_grid, HybridLAMNrTrainer._display_slice


# ------------------------------------------------------------------ context encoders and flows (vector views)
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


class CausalLSTM(nn.Module):
    """MoGlow-style recurrent context: LSTM over the past frames (B, C, T) -> (B, H, T)."""

    def __init__(self, c, h, layers=1):
        super().__init__()
        self.inp = nn.Linear(c, h)
        self.rnn = nn.LSTM(h, h, layers, batch_first=True)

    def forward(self, x):
        y, _ = self.rnn(F.gelu(self.inp(x.transpose(1, 2))))
        return y.transpose(1, 2)


class WindowEncoder(nn.Module):
    """m previous steps (B, m, C) -> context state (B, ctx_dim)."""

    def __init__(self, C, m, ctx_dim, arch):
        super().__init__()
        self.arch = arch
        if arch == "mlp":
            self.net = nn.Sequential(nn.Linear(m * C, ctx_dim), nn.GELU(), nn.Linear(ctx_dim, ctx_dim))
        elif arch == "lstm":
            self.net = CausalLSTM(C, ctx_dim)
        else:
            k, dil = ARCHS[arch]
            self.net = CausalTCN(C, ctx_dim, dil, k)

    def forward(self, ctx):
        return self.net(ctx.flatten(1)) if self.arch == "mlp" else self.net(ctx.transpose(1, 2))[:, :, -1]


class NoCtx(nn.Module):
    """Lets a context-free layer (ActNorm) sit in a ConditionalNormalizingFlow, which passes ``context=`` to every layer."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, z, context=None):
        return self.flow(z)

    def inverse(self, z, context=None):
        return self.flow.inverse(z)


class CondAffineCoupling(nn.Module):
    """Glow/MoGlow affine coupling on vectors with a context. antsnormflows convention: forward = latent -> data, inverse = data -> latent."""

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


class VectorFlow(nn.Module):
    """p(x_t | m previous vectors): window encoder + conditional flow (rational-quadratic spline or MoGlow-style affine couplings)."""

    def __init__(self, C, m, cfg):
        super().__init__()
        self.C, self.m, self.ctx_dim = C, m, int(cfg["ctx_dim"])
        self.enc = WindowEncoder(C, m, self.ctx_dim, cfg["arch"]) if m > 0 else None
        flows = []
        for j in range(int(cfg["K"])):
            if cfg["coupling"] == "affine":
                flows += [NoCtx(ActNorm((C,))), LULinearPermute(C), CondAffineCoupling(C, self.ctx_dim, int(cfg["hidden"]), int(cfg["num_blocks"]))]
            else:
                flows.append(CoupledRationalQuadraticSpline(C, int(cfg["num_blocks"]), int(cfg["hidden"]), num_context_channels=self.ctx_dim,
                                                            num_bins=int(cfg["num_bins"]), tail_bound=float(cfg["tail_bound"]),
                                                            reverse_mask=bool(j % 2)))
                flows.append(Permute(C, mode="shuffle"))
        base = nn.Linear(self.ctx_dim, 2 * C)
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
        latent = z
        for f in self.flow.flows:
            z, _ = f(z, context=h)
        return latent, z

    @torch.no_grad()
    def whitened(self, latent, ctx):
        h = self.state(ctx, latent.shape[0], latent.device)
        enc = self.flow.q0.context_encoder(h)
        mean, log_scale = enc[..., : enc.shape[-1] // 2], enc[..., enc.shape[-1] // 2:]
        q0 = self.flow.q0
        return (latent - mean) * torch.exp(-_clamp_log_scale(log_scale, q0.min_log, q0.max_log))


# ------------------------------------------------------------------ conditional Glow (image views)
class _Ctx:
    """Plain holder (not a module): the context feature map of the level currently being evaluated."""
    value = None

    def __deepcopy__(self, memo):                                # EMA copies must not carry the autograd graph of the last batch
        return _Ctx()


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


class ImageFlow(nn.Module):
    def __init__(self, C, H, W, m, cfg):
        super().__init__()
        self.m = m
        self.net = CondGlow2d(C, H, W, int(cfg["L"]), int(cfg["K"]), int(cfg["hidden"]), m * C, int(cfg["ctx_ch"]), float(cfg["scale_cap"]))
        self.state_dim = int(cfg["ctx_ch"]) if m > 0 else 1

    def log_prob(self, x, ctx):
        lp, g = self.net.log_prob(x, ctx if self.m else None)
        return lp, (g if g is not None else torch.zeros(x.shape[0], 1, device=x.device))

    @torch.no_grad()
    def roundtrip(self, x, ctx):
        zs, _, _ = self.net.encode(x, ctx if self.m else None)
        return zs, self.net.decode(zs, ctx if self.m else None)


def label_map(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) channel fractions -> grey-level label image (0 = background)."""
    x = torch.nan_to_num(x.detach().float().cpu())
    val, arg = x.max(0)
    return ((arg.float() + 1) / x.shape[0] * (val > 0.25)).numpy()


# ------------------------------------------------------------------ data
def parse_views(path: str) -> Tuple[List[HybridViewSpec], List[dict], Dict[str, Any]]:
    cfg = json.load(open(path))
    raw = list(cfg.get("views", []))
    views = [HybridViewSpec.from_dict(v) for v in raw]
    if not views:
        raise ValueError("The configuration needs at least one view.")
    if len({v.name for v in views}) != len(views):
        raise ValueError(f"View names must be unique; got {[v.name for v in views]}.")
    for v in views:
        if v.kind == "image3d":
            raise ValueError(f"View {v.name!r}: image3d is not supported by the temporal trainer.")
    return views, raw, cfg


def view_config(view: HybridViewSpec, args) -> Dict[str, Any]:
    g = view.model.get
    if view.kind == "image2d":
        return dict(L=g("L", args.image_L), K=g("K", args.image_K), hidden=g("hidden", args.image_hidden), ctx_ch=g("ctx_ch", args.ctx_ch),
                    scale_cap=g("scale_cap", args.scale_cap), ctx_frames=int(g("ctx_frames", args.ctx_frames)))
    hidden = g("hidden", args.tabular_hidden)
    return dict(arch=g("arch", args.arch), ctx_dim=g("ctx_dim", args.ctx_dim), K=g("K", args.tabular_K), hidden=128 if hidden is None else hidden,
                coupling=g("coupling", args.coupling), num_blocks=g("num_blocks", args.num_blocks), num_bins=g("num_bins", args.num_bins),
                tail_bound=g("tail_bound", args.tail_bound), ctx_frames=int(g("ctx_frames", args.ctx_frames)))


def filter_manifest(man: pd.DataFrame, filters: Sequence[str]) -> pd.DataFrame:
    for f in filters:
        col, vals = f.split("=", 1)
        if col not in man.columns:
            raise ValueError(f"--filter column {col!r} is not in the manifest.")
        man = man[man[col].astype(str).isin([v.strip() for v in vals.split(",")])]
    return man


def frame_runs(man: pd.DataFrame, seq_col: str, time_col: str, burn: int):
    """man sorted by (sequence, time): runs of constant time step with more than ``burn`` rows."""
    steps = man.groupby(seq_col)[time_col].diff().dropna()
    step = int(steps[steps > 0].mode().iloc[0])
    runs, keys = [], []
    for pid, g in man.groupby(seq_col, sort=False):
        idx, tm = g.index.to_numpy(), g[time_col].to_numpy()
        cut = [0] + [i for i in range(1, len(idx)) if tm[i] - tm[i - 1] != step] + [len(idx)]
        for s, e in zip(cut[:-1], cut[1:]):
            if e - s > burn:
                runs.append(idx[s:e])
                keys.append((str(pid), int(tm[s])))
    return runs, keys, step


def stitch_windows(man: pd.DataFrame, view: HybridViewSpec, seq_col: str, time_col: str, burn: int, no_progress: bool):
    """Overlapping windows (C, W) -> per-run streams (T, C); runs = contiguous windows. Returns streams, keys, first-row index, times."""
    man = man.sort_values([seq_col, time_col], kind="stable")
    steps = man.groupby(seq_col)[time_col].diff().dropna()
    step = int(steps[steps > 0].mode().iloc[0])
    streams, keys, first_rows, times, hop, n_bad = [], [], [], [], None, 0
    for pid, g in tqdm(list(man.groupby(seq_col, sort=False)), desc=f"stitch {view.name}", disable=no_progress):
        wins = [_load_signal(str(p), view).numpy() for p in g[view.path_column]]
        starts, idx = g[time_col].to_numpy(), g.index.to_numpy()
        cuts = [0] + [i for i in range(1, len(wins)) if starts[i] - starts[i - 1] != step] + [len(wins)]
        for s, e in zip(cuts[:-1], cuts[1:]):
            w = wins[s:e]
            if hop is None and len(w) >= 2:
                W = w[0].shape[1]
                hop = min(range(1, W), key=lambda h: float(np.abs(w[1][:, :W - h] - w[0][:, h:]).mean()))
            seq = w[0]
            for x in w[1:]:
                ov = x.shape[1] - hop
                if float(np.abs(x[:, :ov] - seq[:, -ov:]).mean()) > 1e-3 * (np.abs(seq).mean() + 1e-9):
                    n_bad += 1
                seq = np.concatenate([seq, x[:, -hop:]], 1)
            if seq.shape[1] > burn:
                streams.append(seq.T.astype(np.float32))
                keys.append((str(pid), int(starts[s])))
                first_rows.append(int(idx[s]))
                times.append(int(starts[s]) + np.arange(seq.shape[1]))
    print(f"[data] {view.name}: {len(streams)} stitched sequences (length median {int(np.median([len(s) for s in streams]))}); hop {hop}; "
          f"runs with overlap mismatch: {n_bad}")
    return streams, keys, first_rows, times


def interp_u(rows, shape, seed):
    out = np.empty((len(rows),) + tuple(shape), np.float32)
    for i, r in enumerate(rows):
        out[i] = np.random.default_rng((zlib.crc32(str(r).encode()) ^ seed) & 0xFFFFFFFF).random(shape, np.float32)
    return torch.from_numpy(out)


# ------------------------------------------------------------------ trainer
class HybridTemporalLAMNrTrainer:
    def setup(self, args: argparse.Namespace) -> None:
        self.args = a = args
        set_deterministic(a.seed, a.deterministic)
        if a.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 or "," in a.devices:
            raise SystemExit("DDP / DataParallel are not implemented in the temporal trainer: use a single device.")
        if a.devices == "mps" and torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        elif a.devices.startswith("cuda") and torch.cuda.is_available():
            self.dev = torch.device(a.devices)
        else:
            self.dev = torch.device("cpu")
        self.rank = 0
        self.run_dir = Path(a.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path, self.best_path, self.metrics_path = (self.run_dir / "training_state.pt", self.run_dir / "best.pt",
                                                              self.run_dir / "metrics.csv")
        self.views, self.raw_views, cfg = parse_views(a.config)
        self._resume_path: Optional[Path] = None
        self.ck = None
        if a.export_only:
            path = Path(a.checkpoint) if a.checkpoint else (self.state_path if a.export_from == "last" else self.best_path)
            self._resume_path = path
        elif a.resume:
            self._resume_path = Path(a.resume)
        elif a.auto_resume and self.state_path.exists():
            self._resume_path = self.state_path
        if self._resume_path is not None and self._resume_path.exists():
            self.ck = torch.load(self._resume_path, map_location="cpu", weights_only=False)
            self._reconcile_config_with_checkpoint(self._resume_path)
        elif a.export_only:
            raise SystemExit(f"--export-only: checkpoint {self._resume_path} not found.")
        self.cfgs = [view_config(v, a) for v in self.views]
        self.m = [c["ctx_frames"] for c in self.cfgs]
        a.burn = max(self.m) if a.burn is None else a.burn
        if a.burn < max(self.m):
            raise SystemExit(f"--burn ({a.burn}) must be >= every view's ctx_frames ({self.m}).")
        if min(self.m) == 0 and a.align != "none":
            print("[warn] a view with ctx_frames 0 has no context state: alignment switched off")
            a.align = "none"
        self.names = [v.name for v in self.views]
        self.V = len(self.views)
        self.labels_views = set(x for x in a.display_labels.split(",") if x) | {r["name"] for r in self.raw_views if r.get("display") == "labels"}

        self._load_data(cfg)
        self.models = nn.ModuleList([self._build_model(v, c, m) for v, c, m in zip(self.views, self.cfgs, self.m)]).to(self.dev)
        dims = [mod.state_dim if isinstance(mod, ImageFlow) else mod.ctx_dim for mod in self.models]
        self.projectors = nn.ModuleList([Projector(d, a.proj_hidden, a.proj_dim) for d in dims]).to(self.dev)
        self.feature_dims = dims
        self.ema_models: Optional[nn.ModuleList] = None
        self.ema_projectors: Optional[nn.ModuleList] = None
        self.last_val_bpds = [float("nan")] * self.V
        self.start_iter, self.best, self.bad, self.anomaly_streak = 1, 1e9, 0, 0

        if a.export_only:
            self._load_export_weights()
            return
        self.align_mgr = LatentAlignmentLossManager(a, self.projectors, self.dev)
        self.s_nll = self.s_align = None
        parameters = list(self.models.parameters()) + list(self.projectors.parameters())
        if a.weighting == "kendall" and a.align != "none":
            self.s_nll = nn.Parameter(torch.tensor([a.init_logvar_nll], device=self.dev))
            self.s_align = nn.Parameter(torch.tensor([a.init_logvar_align], device=self.dev))
            parameters += [self.s_nll, self.s_align]
        # data-dependent ActNorm initialisation on a first batch, before the optimizer / EMA exist
        self.models.train()
        with torch.no_grad():
            first = self._gather(self.tr_pos[np.random.default_rng(a.seed).integers(0, len(self.tr_pos), a.batch_size)], train=True)
            self._batch_loss(first, 0)
        self.opt = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=a.weight_decay)
        self.warm = make_warmup(self.opt, a.warmup_iters, a.lr_decay_gamma, a.lr_decay_steps)
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=a.plateau_factor, patience=a.plateau_patience,
                                                                  threshold=a.plateau_threshold, cooldown=a.plateau_cooldown, min_lr=a.min_lr)
        self.amp_enabled = a.precision == "mixed" and self.dev.type == "cuda"
        self.amp_dtype = torch.bfloat16 if a.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.amp_enabled and self.amp_dtype == torch.float16))
        if self.ck is not None:
            self.start_iter = self._load_checkpoint(self.ck)
        if a.extra_iters > 0:
            a.max_iter = (self.start_iter - 1) + a.extra_iters
        if not self.metrics_path.exists() or self.start_iter == 1:
            cols = ["iter", "loss", "align", "sum_bpd", "val_loss", "lr"] + [f"bpd_{n}" for n in self.names] + [f"val_bpd_{n}" for n in self.names]
            self.metrics_path.write_text(",".join(cols) + "\n")
        elif self.metrics_path.exists():
            df = pd.read_csv(self.metrics_path)
            df[df["iter"] < self.start_iter].to_csv(self.metrics_path, index=False, float_format="%.8g")
        availability = {v.name: int(len(self.train_ix)) for v in self.views}
        (self.run_dir / "run_config.json").write_text(json.dumps({
            "trainer": "hybrid-temporal", "world_size": 1, "device": str(self.dev), "arguments": vars(a),
            "views": [v.__dict__ for v in self.views], "availability": availability, "projector_input_dimensions": dims,
            "view_configs": self.cfgs}, indent=2, default=str))
        summary = self._format_run_summary(availability)
        print("\n" + summary)
        (self.run_dir / "run_config.txt").write_text(summary + "\n")
        if self.ck is not None:
            tqdm.write(f"[resume] from {self._resume_path} @ iter {self.start_iter}")

    # ---------------- configuration / checkpoint reconciliation
    def _reconcile_config_with_checkpoint(self, path: Path) -> None:
        a, ck = self.args, self.ck
        ck_views, ck_cfg = ck.get("views"), ck.get("config", {})
        arch_keys = ["image_L", "image_K", "image_hidden", "scale_cap", "tabular_K", "tabular_hidden", "arch", "coupling", "ctx_dim", "ctx_ch",
                     "ctx_frames", "num_blocks", "num_bins", "tail_bound", "proj_dim", "proj_hidden"]
        mism_views = []
        if ck_views:
            cur = {v.name: v for v in self.views}
            for raw in ck_views:
                v = cur.get(raw.get("name"))
                if v is None:
                    mism_views.append((raw.get("name"), "<view>", "absent from --config", "present in checkpoint"))
                    continue
                for f in ("kind", "shape", "channels", "columns", "model"):
                    if getattr(v, f) != raw.get(f):
                        mism_views.append((v.name, f, getattr(v, f), raw.get(f)))
        mism_args = [k for k in arch_keys if k in ck_cfg and getattr(a, k, None) != ck_cfg[k]]
        if not mism_views and not mism_args:
            return
        if a.use_ckpt_config or a.export_only:
            if mism_views and ck_views:
                self.views = [HybridViewSpec.from_dict(r) for r in ck_views]
                self.raw_views = [{**r} for r in ck_views]
            for k in mism_args:
                setattr(a, k, ck_cfg[k])
            if "burn" in ck_cfg:
                a.burn = ck_cfg["burn"]
            tqdm.write(f"[resume] adopted the checkpoint's architecture ({len(mism_views)} view field(s), {len(mism_args)} global arg(s)) from {path}")
        else:
            details = "; ".join(f"view {n!r} field {f!r}: --config={c!r} vs checkpoint={k!r}" for n, f, c, k in mism_views)
            details += "; ".join(f"{k}: now {getattr(a, k)!r} vs checkpoint {ck_cfg[k]!r}" for k in mism_args)
            raise ValueError(f"Resume architecture mismatch at {path}: {details}. Fix the arguments or pass --use-ckpt-config.")

    # ---------------- data
    def _load_data(self, cfg: Dict[str, Any]) -> None:
        a = self.args
        man = pd.read_csv(a.manifest)
        man["__row__"] = np.arange(len(man))
        man = filter_manifest(man, a.filter)
        seq_col, time_col = a.sequence_column, a.time_column
        subject_col = a.subject_column or cfg.get("subject_column", "subject_id")
        self.subject_col = subject_col
        need = {seq_col}
        for v, r in zip(self.views, self.raw_views):
            need |= set(v.columns) if v.kind == "tabular" else {v.path_column}
            need.add(r.get("time_column", time_col))
        miss = sorted(c for c in need if c not in man.columns)
        if miss:
            raise ValueError(f"The manifest is missing columns: {miss}")
        if a.subject_limit > 0 and subject_col in man.columns:
            man = man[man[subject_col].isin(man[subject_col].drop_duplicates().iloc[:a.subject_limit])]
        man = man.reset_index(drop=True)
        avail = np.ones(len(man), bool)
        for v in self.views:
            if v.kind == "tabular":
                avail &= np.isfinite(man[v.columns].apply(pd.to_numeric, errors="coerce").to_numpy()).all(1)
            else:
                avail &= man[v.path_column].map(lambda p: _is_present(p) and Path(str(p)).expanduser().exists()).to_numpy()
        man = man[avail].reset_index(drop=True)
        self.man = man

        frame_views = [i for i, v in enumerate(self.views) if v.kind != "signal1d"]
        sig_views = [i for i, v in enumerate(self.views) if v.kind == "signal1d"]
        self.arrays: List[Any] = [None] * self.V
        self.seqs: List[List[np.ndarray]] = [None] * self.V
        self.normalizers: Dict[str, Any] = {}
        ref_keys = ref_rows = ref_times = None
        if frame_views:
            tcols = {self.raw_views[i].get("time_column", time_col) for i in frame_views}
            if len(tcols) != 1:
                raise ValueError("Frame-wise views must share one time column.")
            tc = tcols.pop()
            fm = man.sort_values([seq_col, tc], kind="stable").reset_index(drop=True)
            runs, keys, step = frame_runs(fm, seq_col, tc, a.burn)
            if not runs:
                raise ValueError("No run is longer than --burn: nothing to train on.")
            used = np.unique(np.concatenate(runs))
            remap = -np.ones(len(fm), np.int64)
            remap[used] = np.arange(len(used))
            ids = [remap[r] for r in runs]
            sub = fm.loc[used].reset_index(drop=True)
            ref_keys = keys
            ref_rows = [fm.loc[r, "__row__"].to_numpy() for r in runs]
            ref_times = [fm.loc[r, tc].to_numpy() for r in runs]
            self.step = step
            for i in frame_views:
                self.seqs[i] = ids
            self._sub = sub
        if sig_views:
            for i in sig_views:
                tc = self.raw_views[i].get("time_column", time_col)
                streams, keys, first_rows, times = stitch_windows(man, self.views[i], seq_col, tc, a.burn, a.no_progress)
                lens = np.cumsum([0] + [len(s) for s in streams])
                self.arrays[i] = np.concatenate(streams).astype(np.float32)
                self.seqs[i] = [np.arange(lens[k], lens[k + 1]) for k in range(len(streams))]
                if ref_keys is None:
                    ref_keys = keys
                    ref_rows = [np.full(len(s), r) for s, r in zip(streams, [man.loc[f, "__row__"] for f in first_rows])]
                    ref_times = times
                    self.step = 1
                elif keys != ref_keys or [len(s) for s in self.seqs[i]] != [len(s) for s in self.seqs[frame_views[0] if frame_views else i]]:
                    raise ValueError(f"View {self.views[i].name!r}: its sequences (stitched windows) differ from the other views' "
                                     f"({len(keys)} vs {len(ref_keys)} sequences, keys or lengths); views must share one time base.")
        self.keys, self.seq_rows, self.seq_times = ref_keys, ref_rows, ref_times
        full = self.man.set_index("__row__")
        first_row = [r[0] for r in self.seq_rows]
        self.meta = pd.DataFrame({"pass_id": [k[0] for k in self.keys], "t0": [k[1] for k in self.keys],
                                  "length": [len(r) for r in self.seq_rows]})
        self.meta["subject_id"] = full.loc[first_row, subject_col].astype(str).to_numpy() if subject_col in full.columns else np.arange(len(first_row)).astype(str)
        self.meta["split"] = full.loc[first_row, a.split_column].astype(str).to_numpy() if a.split_column in full.columns else "train"
        self.annots = [c for c in a.annotations.split(",") if c and c in full.columns]
        for c in self.annots:
            self.meta[c] = full.loc[first_row, c].to_numpy()
        split = self.meta.split.to_numpy().copy()
        if "val" not in set(split):
            subs = np.array(sorted(set(self.meta.subject_id[split == "train"])))
            nv = min(max(0, int(round(a.val_fraction * len(subs)))), max(len(subs) - 1, 0))
            if nv == 0 and len(subs) > 1:
                nv = 1
            vs = set(np.random.default_rng(a.seed).choice(subs, nv, replace=False)) if nv else set()
            split = np.where(self.meta.subject_id.isin(vs) & (split == "train"), "val", split)
        self.split = split
        self.train_ix, self.val_ix = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
        if len(self.train_ix) == 0:
            raise ValueError("No training sequence.")
        if len(self.val_ix) == 0:
            self.val_ix = self.train_ix
        self._load_view_arrays()
        pos = lambda ix: np.array([(s, t) for s in ix for t in range(a.burn, len(self.seqs[0][s]))])
        self.tr_pos = pos(self.train_ix)
        allv = pos(self.val_ix)
        rng = np.random.default_rng(a.seed + 7)
        self.val_pos = allv[rng.permutation(len(allv))[: a.val_samples or len(allv)]] if len(allv) else allv
        sizes = ", ".join(f"{n}: {tuple(self.arrays[i].shape[1:])}" for i, n in enumerate(self.names))
        print(f"[data] {len(self.keys)} sequences (train {len(self.train_ix)} / val {len(self.val_ix)} / test {int((split == 'test').sum())}), "
              f"positions train {len(self.tr_pos)} / val {len(self.val_pos)}; burn {a.burn}; views {sizes}")

    def _load_view_arrays(self) -> None:
        a = self.args
        norm_states = (self.ck or {}).get("normalizers", {}) if self.ck is not None else {}
        train_ids = lambda i: np.unique(np.concatenate([self.seqs[i][s] for s in self.train_ix]))
        for i, v in enumerate(self.views):
            if v.kind == "tabular":
                raw = self._sub[v.columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
                norm = TabularNormalizer(v.normalization)
                if v.name in norm_states:
                    norm.load_state_dict(norm_states[v.name])
                else:
                    norm.fit(raw[train_ids(i)])
                self.normalizers[v.name] = norm
                self.arrays[i] = norm.transform(raw).numpy().astype(np.float32)
            elif v.kind == "signal1d":
                norm = ChannelNormalizer(v.normalization, kind="signal1d")
                if v.name in norm_states:
                    norm.load_state_dict(norm_states[v.name])
                else:
                    norm.fit(torch.from_numpy(self.arrays[i][train_ids(i)].T.copy())[None].unbind(0))
                self.normalizers[v.name] = norm
                self.arrays[i] = norm.transform_batch(torch.from_numpy(self.arrays[i]).T[None])[0].T.contiguous().numpy().astype(np.float32)
            else:
                paths = [Path(str(p)).expanduser() for p in self._sub[v.path_column]]
                root = Path(a.data_root) if a.data_root else Path(a.manifest).resolve().parent

                def _p(p):
                    if p.exists():
                        return p
                    for k in range(len(p.parts) - 1, -1, -1):
                        q = root.joinpath(*p.parts[k:])
                        if q.exists():
                            return q
                    raise FileNotFoundError(p)

                def _load(p):
                    t = _load_image(str(_p(p)), v)
                    if v.dequantize:
                        return torch.clamp(torch.round(t * v.dequantize), 0, v.dequantize).to(torch.uint8).numpy()
                    return t.numpy()
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=8) as ex:
                    X = np.stack(list(tqdm(ex.map(_load, paths), total=len(paths), desc=f"load {v.name}", disable=a.no_progress)))
                if v.intensity == "0mean":
                    norm = ChannelNormalizer("0mean", kind="image2d")
                    if v.name in norm_states:
                        norm.load_state_dict(norm_states[v.name])
                    else:
                        norm.fit(torch.from_numpy(X[train_ids(i)]).unbind(0))
                    self.normalizers[v.name] = norm
                    X = norm.transform_batch(torch.from_numpy(X)).numpy()
                self.arrays[i] = X
                print(f"[data] {v.name}: images {X.shape} {X.dtype} ({X.nbytes / 2**30:.2f} GiB, {time.time() - t0:.0f} s)")

    # ---------------- model
    def _build_model(self, view: HybridViewSpec, cfg: Dict[str, Any], m: int) -> nn.Module:
        if view.kind == "image2d":
            model = ImageFlow(view.channels, view.shape[0], view.shape[1], m, cfg)
        else:
            C = len(view.columns) if view.kind == "tabular" else view.channels
            model = VectorFlow(C, m, cfg)
        print(f"[init] {view.name} ({view.kind}): {n_params(model):,} parameters")
        return model

    def _load_export_weights(self) -> None:
        a = self.args
        if a.untrained:
            torch.manual_seed(a.seed + 100)
            self.models = nn.ModuleList([self._build_model(v, c, m) for v, c, m in zip(self.views, self.cfgs, self.m)]).to(self.dev)
            self.projectors = nn.ModuleList([Projector(d, a.proj_hidden, a.proj_dim) for d in self.feature_dims]).to(self.dev)
            return
        ck = self.ck
        key = "ema_models" if ck.get("ema_models") is not None else "models"
        for mod, sd in zip(self.models, ck[key]):
            mod.load_state_dict(sd)
        pkey = "ema_projectors" if ck.get("ema_projectors") is not None else "projectors"
        for mod, sd in zip(self.projectors, ck[pkey]):
            mod.load_state_dict(sd)
        self.normalizers_ck = ck.get("normalizers", {})

    def _format_run_summary(self, availability) -> str:
        a = self.args
        rows = [f"[run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Py {platform.python_version()} | torch {torch.__version__} | "
                f"cuda={str(torch.cuda.is_available()).lower()} (n={torch.cuda.device_count()})",
                "[note] hybrid temporal trainer: autoregressive conditional flows aligned through their context states"]

        def add(k, v):
            rows.append(f"{k:>28}: {'None' if v is None else v}")
        add("out_dir", a.out_dir)
        add("manifest / config", f"{a.manifest} / {a.config}")
        add("filters", a.filter or None)
        add("world_size / device", f"1 / {self.dev}")
        add("views", self.V)
        add("precision / amp_dtype", f"{a.precision} / {a.amp_dtype}")
        add("seed", a.seed)
        add("batch / accum / effective", f"{a.batch_size} / {a.accum_steps} / {a.batch_size * a.accum_steps}")
        add("max_iter / extra", f"{a.max_iter} / {a.extra_iters}")
        add("start / target iteration", f"{self.start_iter} / {a.max_iter}")
        add("eval / preview interval", f"{a.eval_interval} / {a.preview_interval}")
        add("lr / warmup", f"{a.lr} / {a.warmup_iters}")
        add("grad_clip / weight_decay", f"{a.grad_clip} / {a.weight_decay}")
        add("ema / decay", f"{a.ema} / {a.ema_decay}")
        add("lr_decay gamma / steps", f"{a.lr_decay_gamma} / {a.lr_decay_steps}")
        add("plateau fac/pat/thr/cd", f"{a.plateau_factor} / {a.plateau_patience} / {a.plateau_threshold} / {a.plateau_cooldown}")
        add("min_lr", a.min_lr)
        add("resume argument", a.resume or None)
        add("resolved checkpoint", str(self._resume_path) if self._resume_path is not None and self.ck is not None else None)
        add("auto_resume / ckpt config", f"{a.auto_resume} / {a.use_ckpt_config}")
        add("manifest rows used", len(self.man))
        add("sequence / time / subject", f"{a.sequence_column} / {a.time_column} / {self.subject_col}")
        add("sequences train / val / test", f"{len(self.train_ix)} / {len(self.val_ix)} / {int((self.split == 'test').sum())}")
        add("positions train / val", f"{len(self.tr_pos)} / {len(self.val_pos)}")
        add("burn / ctx frames per view", f"{a.burn} / {self.m}")
        add("dequantize eval", a.dequantize_eval)
        add("align / weighting", f"{a.align} / {a.weighting}")
        add("align weight / warmup", f"{a.align_weight} / {a.align_warmup}")
        add("alignment latents", "context states")
        add("proj dim / hidden", f"{a.proj_dim} / {a.proj_hidden}")
        add("vicreg inv/var/cov/gamma", f"{a.vicreg_inv} / {a.vicreg_var} / {a.vicreg_cov} / {a.vicreg_gamma}")
        add("screen / fraction", f"{a.screen} / {a.screen_frac}")
        add("sample mode / temp", f"{a.sample_mode} / {a.sample_temp}")
        add("preview samples / columns", f"{a.preview_samples} / {a.preview_columns}")
        rows.append("-" * 72)
        for i, v in enumerate(self.views):
            add(f"view[{i}] name / type", f"{v.name} / {v.kind}")
            add(f"view[{i}] flow parameters", f"{n_params(self.models[i]):,}")
            add(f"view[{i}] projector input", self.feature_dims[i])
            if v.kind == "tabular":
                add(f"view[{i}] columns", v.columns)
                add(f"view[{i}] normalization", v.normalization)
            else:
                add(f"view[{i}] shape / channels", f"{v.shape} / {v.channels}")
                if v.kind == "image2d":
                    add(f"view[{i}] intensity", v.intensity)
                    if v.dequantize:
                        add(f"view[{i}] dequantize (levels)", v.dequantize)
            add(f"view[{i}] temporal model", self.cfgs[i])
        rows.append("-" * 72)
        return "\n".join(rows)

    # ---------------- batches
    def _noise_std(self, v: HybridViewSpec) -> float:
        if self.args.disable_augmentation:
            return 0.0
        key = "tabular_noise_std" if v.kind == "tabular" else "noise_std"
        return float(v.augmentation.get(key, self.args.tabular_noise_std))

    def _gather(self, pos, train: bool, ctx_pos=None):
        """Per view: (target (B, ...), context (B, m, C) or (B, m*C, H, W) or None)."""
        out = []
        cpos = pos if ctx_pos is None else ctx_pos
        for i, v in enumerate(self.views):
            m, arr = self.m[i], self.arrays[i]
            rows = np.array([self.seqs[i][s][t] for s, t in pos])
            crow = np.stack([self.seqs[i][s][t - m:t] for s, t in cpos]) if m else None
            if v.kind == "image2d":
                n = torch.from_numpy(arr[rows]).to(self.dev).float()
                if v.dequantize:
                    if train or self.args.dequantize_eval == "midpoint":
                        u = torch.rand_like(n) if train else torch.full_like(n, 0.5)
                    else:
                        u = interp_u([f"{v.name}:{r}" for r in rows], n.shape[1:], self.args.noise_seed).to(self.dev)
                    x = (n + u) / (v.dequantize + 1)
                else:
                    x = n
                ctx = None
                if m:
                    c = torch.from_numpy(arr[crow]).to(self.dev).float()
                    if v.dequantize:
                        c = (c + 0.5) / (v.dequantize + 1) - 0.5
                    elif v.intensity == "to01":
                        c = self._to01(c) - 0.5
                    ctx = c.reshape(len(pos), m * v.channels, *c.shape[3:])
                if v.intensity == "to01" and not v.dequantize:
                    x = self._to01(x)
                out.append((x, ctx))
            else:
                x = torch.from_numpy(arr[rows]).to(self.dev)
                sd = self._noise_std(v)
                if train and sd > 0:
                    x = x + sd * torch.randn_like(x)
                ctx = torch.from_numpy(arr[crow]).to(self.dev) if m else None
                out.append((x, ctx))
        return out

    @staticmethod
    def _to01(t):
        flat = t.flatten(-3) if t.ndim >= 4 else t.flatten(1)
        lo, hi = flat.min(-1).values, flat.max(-1).values
        shape = [-1] + [1] * (t.ndim - 1) if t.ndim == 4 else [-1] * 1 + [1] * (t.ndim - 1)
        if t.ndim == 5:                                              # (B, m, C, H, W): per frame
            lo, hi = lo.reshape(*t.shape[:2], 1, 1, 1), hi.reshape(*t.shape[:2], 1, 1, 1)
            return (t - lo) / (hi - lo).clamp_min(1e-6)
        return (t - lo.reshape(shape)) / (hi - lo).reshape(shape).clamp_min(1e-6)

    def _batch_loss(self, batch, iteration, models=None, projectors=None):
        models = self.models if models is None else models
        L_nll = torch.zeros((), device=self.dev)
        states, bpds = [], []
        for (x, ctx), model in zip(batch, models):
            lp, st = model.log_prob(x, ctx)
            bpd = (-lp / (LN2 * x[0].numel())).mean()
            L_nll = L_nll + bpd
            bpds.append(float(bpd.detach()))
            states.append(torch.nan_to_num(st.float()))
        mgr = self.align_mgr
        if projectors is not None and projectors is not self.projectors:
            mgr = LatentAlignmentLossManager(self.args, projectors, self.dev)
        loss, align, _, _ = mgr.compute(states, L_nll, iteration, getattr(self, "s_nll", None), getattr(self, "s_align", None))
        return loss, align, bpds

    # ---------------- training
    def _init_or_update_ema(self) -> None:
        if not self.args.ema:
            return
        if self.ema_models is None:
            self.ema_models = nn.ModuleList([copy.deepcopy(m).eval().requires_grad_(False) for m in self.models])
            self.ema_projectors = nn.ModuleList([copy.deepcopy(p).eval().requires_grad_(False) for p in self.projectors])
            return
        d = float(self.args.ema_decay)
        with torch.no_grad():
            for tgt, src in ((self.ema_models, self.models), (self.ema_projectors, self.projectors)):
                for t, s in zip(tgt, src):
                    for pt, ps in zip(t.parameters(), s.parameters()):
                        pt.lerp_(ps.detach(), 1.0 - d)
                    for bt, bs in zip(t.buffers(), s.buffers()):
                        bt.copy_(bs)

    def _recover_after_anomaly(self, iteration: int, reason: str) -> None:
        a = self.args
        self.anomaly_streak += 1

        def _backed_off(lr):
            return min(lr, max(a.min_lr, lr * a.nan_lr_backoff))
        backed = _backed_off(self.opt.param_groups[0]["lr"])
        for g in self.opt.param_groups:
            g["lr"] = _backed_off(g["lr"])
        tqdm.write(f"[anomaly] iter={iteration}: {reason}; update rejected, streak={self.anomaly_streak}, lr={self.opt.param_groups[0]['lr']:.3g}")
        if self.anomaly_streak >= a.anomaly_reload_after and self.state_path.exists():
            self._load_checkpoint(torch.load(self.state_path, map_location=self.dev, weights_only=False), load_iteration=False)
            for g in self.opt.param_groups:
                g["lr"] = min(g["lr"], backed)
            self.anomaly_streak = 0
            tqdm.write("[anomaly] restored the last validated checkpoint")

    def train(self) -> None:
        a = self.args
        rng = np.random.default_rng(a.seed + self.start_iter)
        ema_l = ema_a = None
        t0 = time.time()
        pbar = tqdm(range(self.start_iter, a.max_iter + 1), desc="train-hybrid-temporal", disable=a.no_progress,
                    initial=max(0, self.start_iter - 1), total=a.max_iter)
        for it in pbar:
            self.models.train()
            self.projectors.train()
            self.opt.zero_grad(set_to_none=True)
            loss_sum = align_sum = 0.0
            bpd_sum = np.zeros(self.V)
            bad = False
            for _ in range(a.accum_steps):
                try:
                    pos = self.tr_pos[rng.integers(0, len(self.tr_pos), a.batch_size)]
                    with torch.amp.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                        loss, align, bpds = self._batch_loss(self._gather(pos, train=True), it)
                        scaled = loss / a.accum_steps
                    if not bool(torch.isfinite(loss).item()):
                        bad = True
                        break
                    self.scaler.scale(scaled).backward()
                    loss_sum += float(loss.detach())
                    align_sum += float(align.detach())
                    bpd_sum += np.array(bpds)
                except (FloatingPointError, RuntimeError) as error:
                    if "out of memory" in str(error).lower() and self.dev.type == "cuda":
                        torch.cuda.empty_cache()
                    tqdm.write(f"[anomaly] forward/backward error: {error}")
                    bad = True
                    break
            if bad:
                self.opt.zero_grad(set_to_none=True)
                self._recover_after_anomaly(it, "non-finite loss or failed batch")
                continue
            self.scaler.unscale_(self.opt)
            if self.s_nll is not None and False:
                pass
            params = [p for g in self.opt.param_groups for p in g["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
            bad_grad = not bool(torch.isfinite(grad_norm).item())
            if a.exploding_grad_norm > 0:
                bad_grad = bad_grad or float(grad_norm) > a.exploding_grad_norm
            if bad_grad:
                self.opt.zero_grad(set_to_none=True)
                self._recover_after_anomaly(it, f"gradient norm={float(grad_norm):.4g}")
                continue
            snap = [p.detach().clone() for p in params]
            opt_snap = {p: {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in self.opt.state[p].items()}
                        for p in params if p in self.opt.state}
            self.scaler.step(self.opt)
            self.scaler.update()
            bad_params = any(not bool(torch.isfinite(p).all().item()) for p in params)
            upd = math.sqrt(sum(float(torch.sum((p.detach() - o) ** 2).item()) for p, o in zip(params, snap)))
            if bad_params or (a.max_update_norm > 0 and (not math.isfinite(upd) or upd > a.max_update_norm)):
                with torch.no_grad():
                    for p, o in zip(params, snap):
                        p.copy_(o)
                    for p, st in opt_snap.items():
                        for k, o in st.items():
                            if torch.is_tensor(o):
                                self.opt.state[p][k].copy_(o)
                            else:
                                self.opt.state[p][k] = o
                self._recover_after_anomaly(it, "non-finite parameter after step" if bad_params else f"parameter update norm={upd:.4g}")
                continue
            self.anomaly_streak = 0
            self._init_or_update_ema()
            if self.warm is not None and it <= a.warmup_iters:
                self.warm.step()
            n = float(a.accum_steps)
            mean_loss, mean_align, bpds = loss_sum / n, align_sum / n, list(bpd_sum / n)
            sum_bpd = sum(x for x in bpds if math.isfinite(x))
            lr = self.opt.param_groups[0]["lr"]
            val_loss = float("nan")
            do_eval = it % a.eval_interval == 0 or it == a.max_iter
            if do_eval:
                val_loss = self.validate(it)
                self.plateau.step(val_loss)
            if a.smooth_alpha > 0:
                ema_l = mean_loss if ema_l is None else (1 - a.smooth_alpha) * ema_l + a.smooth_alpha * mean_loss
                ema_a = mean_align if ema_a is None else (1 - a.smooth_alpha) * ema_a + a.smooth_alpha * mean_align
                dl, da = ema_l, ema_a
            else:
                dl, da = mean_loss, mean_align
            pbar.set_postfix(loss=f"{dl:.4f}", align=f"{da:.4f}")
            with open(self.metrics_path, "a") as fh:
                fh.write(",".join(f"{x:.8g}" for x in [it, mean_loss, mean_align, sum_bpd, val_loss, lr, *bpds, *self.last_val_bpds]) + "\n")
            if do_eval:
                vsum = sum(x for x in self.last_val_bpds if math.isfinite(x))
                improved = vsum < self.best - 1e-4
                self.best, self.bad = (vsum, 0) if improved else (self.best, self.bad + 1)
                self.save_checkpoint(it, improved)
                self._save_previews(it)
                _save_hybrid_metric_plots(self.metrics_path, self.run_dir)
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
        models = self.ema_models if self.ema_models is not None else self.models
        projectors = self.ema_projectors if self.ema_projectors is not None else self.projectors
        for m in models:
            m.eval()
        tot, n_b, bp = 0.0, 0, np.zeros(self.V)
        for i in range(0, len(self.val_pos), a.val_bs):
            batch = self._gather(self.val_pos[i:i + a.val_bs], train=False)
            with torch.amp.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss, _, bpds = self._batch_loss(batch, 10 ** 9, models=models, projectors=projectors)
            if torch.isfinite(loss):
                tot += float(loss)
            bp += np.array(bpds)
            n_b += 1
            if a.val_batches and n_b >= a.val_batches:
                break
        for m in self.models:
            m.train()
        value = tot / max(n_b, 1)
        self.last_val_bpds = list(bp / max(n_b, 1))
        tqdm.write(f"[val] iter={iteration} loss={value:.6g} ({'EMA' if self.ema_models is not None else 'base'}; "
                   + ", ".join(f"{n}={x:.4g}" for n, x in zip(self.names, self.last_val_bpds)) + ")")
        return value

    # ---------------- checkpoints
    def _checkpoint_blob(self, iteration: int) -> Dict[str, Any]:
        return {
            "iter": iteration + 1,
            "models": [m.state_dict() for m in self.models],
            "ema_models": [m.state_dict() for m in self.ema_models] if self.ema_models is not None else None,
            "projectors": [p.state_dict() for p in self.projectors],
            "ema_projectors": [p.state_dict() for p in self.ema_projectors] if self.ema_projectors is not None else None,
            "optimizer": self.opt.state_dict(),
            "warmup": self.warm.state_dict() if self.warm is not None else None,
            "plateau": self.plateau.state_dict(),
            "scaler": self.scaler.state_dict(),
            "normalizers": {k: v.state_dict() for k, v in self.normalizers.items()},
            "views": [v.__dict__ for v in self.views],
            "config": {**vars(self.args)},
            "best": self.best, "val_bpds": list(self.last_val_bpds),
            "kendall": {"s_nll": None if self.s_nll is None else float(self.s_nll.detach().cpu()),
                        "s_align": None if self.s_align is None else float(self.s_align.detach().cpu())},
        }

    def save_checkpoint(self, iteration: int, best: bool) -> None:
        blob = self._checkpoint_blob(iteration)
        torch.save(blob, self.state_path)
        torch.save(blob, self.run_dir / f"training_state_it{iteration:06d}.pt")
        if best:
            torch.save({**blob, "best_iter": iteration}, self.best_path)
            tqdm.write(f"[ckpt] saved best.pt (iter {iteration}, val sum_bpd {self.best:.6g})")
        self.cleanup_checkpoints()
        free = shutil.disk_usage(self.run_dir).free / 2**30
        if free < self.args.disk_warning_gb:
            tqdm.write(f"[disk warning] only {free:.1f} GiB free in {self.run_dir}")

    def cleanup_checkpoints(self) -> None:
        files = sorted(self.run_dir.glob("training_state_it*.pt"))
        keep = set(files[-self.args.keep_last:]) if self.args.keep_last else set()
        for p in files:
            try:
                it = int(p.stem.rsplit("it", 1)[1])
            except (ValueError, IndexError):
                keep.add(p)
                continue
            if self.args.keep_every > 0 and it % self.args.keep_every == 0:
                keep.add(p)
        for p in files:
            if p not in keep:
                p.unlink()

    def _load_checkpoint(self, blob: Dict[str, Any], load_iteration: bool = True) -> int:
        for m, sd in zip(self.models, blob["models"]):
            m.load_state_dict(sd)
        if blob.get("projectors") is not None:
            for p, sd in zip(self.projectors, blob["projectors"]):
                p.load_state_dict(sd)
        self.opt.load_state_dict(blob["optimizer"])
        if self.warm is not None and blob.get("warmup") is not None:
            self.warm.load_state_dict(blob["warmup"])
        if blob.get("plateau") is not None:
            self.plateau.load_state_dict(blob["plateau"])
        if blob.get("scaler"):
            self.scaler.load_state_dict(blob["scaler"])
        if self.args.ema and blob.get("ema_models") is not None:
            self.ema_models = nn.ModuleList([copy.deepcopy(m).eval().requires_grad_(False) for m in self.models])
            for m, sd in zip(self.ema_models, blob["ema_models"]):
                m.load_state_dict(sd)
            if blob.get("ema_projectors") is not None:
                self.ema_projectors = nn.ModuleList([copy.deepcopy(p).eval().requires_grad_(False) for p in self.projectors])
                for p, sd in zip(self.ema_projectors, blob["ema_projectors"]):
                    p.load_state_dict(sd)
        for name, st in blob.get("normalizers", {}).items():
            if name in self.normalizers:
                self.normalizers[name].load_state_dict(st)
        k = blob.get("kendall", {})
        if self.s_nll is not None and k.get("s_nll") is not None:
            self.s_nll.data.fill_(float(k["s_nll"]))
        if self.s_align is not None and k.get("s_align") is not None:
            self.s_align.data.fill_(float(k["s_align"]))
        self.best = float(blob.get("best", self.best))
        return int(blob.get("iter", 1)) if load_iteration else self.start_iter

    # ---------------- previews
    def _show(self, view: HybridViewSpec, t: torch.Tensor, mode: str = "clamp") -> np.ndarray:
        return label_map(t) if view.name in self.labels_views else display_slice(t, mode)

    @torch.no_grad()
    def _save_previews(self, iteration: int) -> None:
        a = self.args
        if a.preview_interval <= 0 or iteration % a.preview_interval or not len(self.val_pos):
            return
        models = self.ema_models if self.ema_models is not None else self.models
        batch = self._gather(self.val_pos[:a.preview_samples], train=False)
        pdir = self.run_dir / "previews"
        for i, (v, model) in enumerate(zip(self.views, models)):
            model.eval()
            x, ctx = batch[i]
            z, rec = model.roundtrip(x, ctx)
            err = (torch.nan_to_num(rec.float()) - x.float()).abs()
            levels = z if isinstance(z, (list, tuple)) else [z]
            zmax = max(float(torch.nan_to_num(l.detach().float()).abs().max().cpu()) for l in levels)
            tqdm.write(f"[recon] iter={iteration} view={v.name} mae={float(err.mean()):.6g} rmse={float(err.square().mean().sqrt()):.6g} "
                       f"max={float(err.max()):.6g} finite={float(torch.isfinite(rec).float().mean()):.6f} "
                       f"x_range=[{float(x.min()):.6g},{float(x.max()):.6g}] recon_range=[{float(torch.nan_to_num(rec).min()):.6g},"
                       f"{float(torch.nan_to_num(rec).max()):.6g}] latent_abs_max={zmax:.6g}")
            if v.kind != "image2d":
                continue
            tiles = []
            for b in range(len(x)):
                tiles += [self._show(v, x[b]), self._show(v, rec[b])]
            write_grid(tiles, pdir / f"{v.name}_recon_it{iteration:06d}.png", columns=2)
            if a.sample_mode != "model":
                continue
            m, net = self.m[i], model.net
            if m == 0:
                s = net.sample(a.preview_samples, None, a.sample_temp)
                write_grid([self._show(v, s[b], a.sample_grid_norm if a.sample_grid_norm != "both" else "both") for b in range(len(s))],
                           pdir / f"{v.name}_samples_it{iteration:06d}.png", columns=a.preview_columns)
                continue
            R = a.rollout_len
            longs = [s for s in self.val_ix if len(self.seqs[i][s]) >= a.burn + R][: min(4, a.preview_samples)]
            if not longs:
                continue
            t0, arr = a.burn, self.arrays[i]
            real = [[arr[self.seqs[i][s][t0 + r]] for r in range(R)] for s in longs]

            def _cx(f):
                c = torch.from_numpy(np.stack(f)).to(self.dev).float()
                return (c + 0.5) / (v.dequantize + 1) - 0.5 if v.dequantize else (self._to01(c) - 0.5 if v.intensity == "to01" else c)
            frames = [_cx([arr[self.seqs[i][s][t0 - m + j]] for s in longs]) for j in range(m)]
            samples = []
            for r in range(R):
                xs = net.sample(len(longs), torch.cat(frames[-m:], 1), a.sample_temp)
                samples.append(xs)
                frames.append(xs - 0.5 if (v.dequantize or v.intensity == "to01") else xs)
            tiles = []
            for k in range(len(longs)):
                rr = [torch.from_numpy(real[k][r].astype(np.float32) / (v.dequantize or 1)) for r in range(R)]
                tiles += [self._show(v, t) for t in rr] + [self._show(v, samples[r][k]) for r in range(R)]
            write_grid(tiles, pdir / f"{v.name}_samples_it{iteration:06d}.png", columns=R)

    # ---------------- export
    @torch.no_grad()
    def export(self) -> None:
        a = self.args
        if not a.export_only and self.best_path.exists() and a.export_from == "best":
            ck = torch.load(self.best_path, map_location=self.dev, weights_only=False)
            key = "ema_models" if ck.get("ema_models") is not None else "models"
            for m, sd in zip(self.models, ck[key]):
                m.load_state_dict(sd)
            pkey = "ema_projectors" if ck.get("ema_projectors") is not None else "projectors"
            for p, sd in zip(self.projectors, ck[pkey]):
                p.load_state_dict(sd)
            models, projectors = self.models, self.projectors
        elif a.export_only:
            models, projectors = self.models, self.projectors
        else:
            models = self.ema_models if self.ema_models is not None else self.models
            projectors = self.ema_projectors if self.ema_projectors is not None else self.projectors
        for m in list(models) + list(projectors):
            m.eval()
        tag = a.export_tag or ("untrained" if a.untrained else "features")
        splits = set(a.export_splits.split(","))
        ex = self.run_dir / "export"
        ex.mkdir(parents=True, exist_ok=True)
        sel = [s for s in range(len(self.seqs[0])) if self.split[s] in splits]
        pos = np.array([(s, t) for s in sel for t in range(a.burn, len(self.seqs[0][s]))])
        N, V = len(pos), self.V
        rng = np.random.default_rng(a.seed + 11)
        pool = pos[rng.permutation(N)]
        has_ctx = min(self.m) > 0
        bp, bs = np.zeros((N, V)), np.full((N, V), np.nan)
        Hs = [np.zeros((N, d), np.float32) for d in self.feature_dims]
        Zs = [np.zeros((N, a.proj_dim), np.float32) for _ in range(V)]
        pid = self.meta.pass_id.to_numpy()
        rec_img: Dict[str, List[dict]] = {v.name: [] for v in self.views if v.kind == "image2d"}
        rec_vec = {v.name: {"row": [], "z": [], "w": [], "recon": []} for v in self.views if v.kind != "image2d"}
        for i in tqdm(range(0, N, a.val_bs), desc="export", disable=a.no_progress):
            p = pos[i:i + a.val_bs]
            sl = slice(i, i + len(p))
            batch = self._gather(p, train=False)
            donors = None
            if has_ctx:
                donors = pool[(np.arange(i, i + len(p)) * 7919 + 13) % len(pool)].copy()
                for k in range(len(p)):
                    j = 0
                    while pid[donors[k][0]] == pid[p[k][0]] and j < 20:
                        donors[k] = pool[rng.integers(0, len(pool))]
                        j += 1
                shuf = self._gather(p, train=False, ctx_pos=donors)
            for v, (view, model, proj) in enumerate(zip(self.views, models, projectors)):
                x, ctx = batch[v]
                lp, st = model.log_prob(x, ctx)
                bp[sl, v] = (-lp / (LN2 * x[0].numel())).cpu().numpy()
                Hs[v][sl] = st.float().cpu().numpy()
                Zs[v][sl] = proj(st.float()).cpu().numpy()
                if has_ctx:
                    lps, _ = model.log_prob(x, shuf[v][1])
                    bs[sl, v] = (-lps / (LN2 * x[0].numel())).cpu().numpy()
                want = a.save_recon or (a.save_z or a.save_whitened) and view.kind != "image2d"
                if want and not a.untrained:
                    rows = np.array([self.seq_rows[s][t] for s, t in p])
                    z, rec = model.roundtrip(x, ctx)
                    if view.kind == "image2d":
                        if a.save_recon and len(rec_img[view.name]) < a.export_max_samples:
                            vdir = ex / view.name / "reconstructions"
                            vdir.mkdir(parents=True, exist_ok=True)
                            r_np = rec.float().cpu()
                            if view.dequantize:
                                r_np = r_np * (view.dequantize + 1) - 0.5
                            elif view.name in self.normalizers:
                                r_np = torch.stack([self.normalizers[view.name].inverse_transform(t) for t in r_np])
                            for k in range(len(p)):
                                if len(rec_img[view.name]) >= a.export_max_samples:
                                    break
                                out = vdir / f"row_{int(rows[k]):06d}.npy"
                                np.save(out, r_np[k].numpy().astype(np.float16 if view.dequantize else np.float32))
                                rec_img[view.name].append({"row": int(rows[k]), "path": str(out)})
                    else:
                        r = rec_vec[view.name]
                        r["row"].extend(rows.tolist())
                        if a.save_z:
                            r["z"].append(z.float().cpu().numpy())
                        if a.save_whitened:
                            r["w"].append(model.whitened(z, ctx).float().cpu().numpy())
                        if a.save_recon:
                            rc = rec.float().cpu()
                            if view.name in self.normalizers:
                                nm = self.normalizers[view.name]
                                rc = nm.inverse_transform(rc.T).T if view.kind == "signal1d" else nm.inverse_transform(rc)
                            r["recon"].append(rc.numpy())
        for view in self.views:
            if view.kind == "image2d":
                if rec_img[view.name]:
                    pd.DataFrame(rec_img[view.name]).to_csv(ex / f"{view.name}_reconstructions.csv", index=False)
                continue
            r = rec_vec[view.name]
            ids = pd.DataFrame({"row": r["row"]})
            cols = view.columns if view.kind == "tabular" else [f"c{j}" for j in range(view.channels)]
            if a.save_z and r["z"]:
                arr = np.concatenate(r["z"])
                ids.join(pd.DataFrame(arr, columns=[f"z{j}" for j in range(arr.shape[1])])).to_csv(ex / f"{view.name}_latents.csv", index=False)
            if a.save_whitened and r["w"]:
                arr = np.concatenate(r["w"])
                ids.join(pd.DataFrame(arr, columns=[f"epsilon{j}" for j in range(arr.shape[1])])).to_csv(ex / f"{view.name}_whitened.csv", index=False)
            if a.save_recon and r["recon"]:
                ids.join(pd.DataFrame(np.concatenate(r["recon"]), columns=cols)).to_csv(ex / f"{view.name}_reconstructions.csv", index=False)

        seq_ids, n_seq = pos[:, 0], len(sel)
        out: Dict[str, Any] = {"length": np.array([len(self.seqs[0][s]) for s in sel])}
        for v in range(V):
            out[f"nll_{v}"], out[f"nll_shuf_{v}"] = np.full(n_seq, np.nan), np.full(n_seq, np.nan)
            out[f"h_{v}"], out[f"z_{v}"] = np.zeros((n_seq, Hs[v].shape[1]), np.float32), np.zeros((n_seq, a.proj_dim), np.float32)
        for k, s in enumerate(sel):
            msk = seq_ids == s
            for v in range(V):
                out[f"nll_{v}"][k] = bp[msk, v].mean()
                out[f"nll_shuf_{v}"][k] = np.nanmean(bs[msk, v]) if has_ctx else np.nan
                out[f"h_{v}"][k], out[f"z_{v}"][k] = Hs[v][msk].mean(0), Zs[v][msk].mean(0)
        for key in ("pass_id", "subject_id", "split", *self.annots):
            out[key] = self.meta[key].to_numpy().astype(str)[sel] if key in ("pass_id", "subject_id", "split") or self.meta[key].dtype == object else self.meta[key].to_numpy()[sel]
        out["view_names"] = np.array(self.names)
        smap = {s: k for k, s in enumerate(sel)}
        out["frame_seq"] = np.array([smap[s] for s in seq_ids])
        out["frame_t"] = pos[:, 1]
        out["frame_number"] = np.array([self.seq_times[s][t] for s, t in pos])
        full = self.man.set_index("__row__")
        for c in self.annots:
            out[f"frame_{c}"] = full.loc[[self.seq_rows[s][t] for s, t in pos], c].to_numpy()
        out["frame_bpd"], out["frame_bpd_shuf"] = bp, bs
        for v in range(V):
            out[f"frame_h{v}"], out[f"frame_z{v}"] = Hs[v], Zs[v]
        np.savez(ex / f"{tag}.npz", **out)
        te = np.array([self.split[s] == "test" for s in sel])
        msg = f"[export] {n_seq} sequences, {N} positions ({a.export_splits}) -> {ex}/{tag}.npz"
        if te.any():
            msg += " | test bpd " + ", ".join(f"{n} {np.mean(out[f'nll_{v}'][te]):.4f}" for v, n in enumerate(self.names))
            if has_ctx:
                msg += " | shuffled-context " + ", ".join(f"{n} {np.nanmean(out[f'nll_shuf_{v}'][te]):.4f}" for v, n in enumerate(self.names))
        print(msg)


def _build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Hybrid temporal LAMNr trainer (conditional flows)", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with one row per time step (window for signal1d views).")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="runs_hybrid_temporal")
    p.add_argument("--subject-column", default="")
    p.add_argument("--data-root", default="", help="Folder to look in when image paths of the manifest are absolute elsewhere (matched on the path tail).")
    p.add_argument("--filter", action="append", default=[], metavar="COL=V[,V]", help="Keep manifest rows with COL in the given values (repeatable).")
    p.add_argument("--sequence-column", default="pass_id")
    p.add_argument("--time-column", default="frame")
    p.add_argument("--split-column", default="split")
    p.add_argument("--annotations", default="", help="Manifest columns exported per sequence and per frame (e.g. phase,speed,jacket).")
    p.add_argument("--devices", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=32, help="Positions per batch.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)                        # accepted, unused
    p.add_argument("--train-samples", type=int, default=0)                      # accepted, unused
    p.add_argument("--val-samples", type=int, default=1280, help="Fixed validation positions (0 = all).")
    p.add_argument("--val-bs", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=0, help="Optional cap on validation batches (0 = all fixed positions).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-gamma", type=float, default=1.0)
    p.add_argument("--lr-decay-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--exploding-grad-norm", type=float, default=0.0)
    p.add_argument("--max-update-norm", type=float, default=1e3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--ddp-find-unused", action=argparse.BooleanOptionalAction, default=False)    # accepted, unused
    p.add_argument("--precision", default="float", choices=["float", "mixed"])
    p.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=4)
    p.add_argument("--plateau-threshold", type=float, default=1e-4)
    p.add_argument("--plateau-cooldown", type=int, default=0)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--nan-lr-backoff", type=float, default=0.5)
    p.add_argument("--anomaly-reload-after", type=int, default=5)
    p.add_argument("--resume", default="")
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--use-ckpt-config", action="store_true")
    p.add_argument("--extra-iters", type=int, default=0)
    p.add_argument("--subject-limit", type=int, default=0, help="Debug: only the first N subjects of the manifest.")
    p.add_argument("--detect-anomaly", action="store_true")
    p.add_argument("--smooth-alpha", type=float, default=0.1)
    p.add_argument("--patience-evals", type=int, default=0, help="Stop after this many evaluations without a better validation sum of bpd (0 = off).")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--noise-seed", type=int, default=12345, help="Seed of the per-frame dequantisation noise at validation/export.")
    p.add_argument("--dequantize-eval", default="seeded", choices=["seeded", "midpoint"],
                   help="u at validation/export: seeded uniform per frame, or 0.5 (hybrid trainer).")
    p.add_argument("--tabular-noise-std", type=float, default=0.0, help="Gaussian noise (normalised units) on tabular/signal targets in training.")
    for name, typ, default in (("--image-noise-std", float, 0.05), ("--augmentation-transform-type", str, "affineAndDeformation"),
                               ("--augmentation-sd-affine", float, 0.05), ("--augmentation-sd-deformation", float, 10.0),
                               ("--augmentation-noise-model", str, "additivegaussian"), ("--augmentation-sd-bias-field", float, 1e-8),
                               ("--augmentation-sd-histogram-warping", float, 0.025), ("--horizontal-flip-probability", float, 0.0),
                               ("--aug-schedules", str, ""), ("--alignment-pool-size", int, 2), ("--grad-checkpoint", str, "auto"),
                               ("--image-base", str, "glow"), ("--tabular-base", str, "DiagGaussian")):
        p.add_argument(name, type=typ, default=default)                         # accepted, unused
    p.add_argument("--disable-augmentation", action="store_true", help="No training noise on tabular/signal targets.")
    p.add_argument("--disable-aug-anneal", action="store_true")                 # accepted, unused
    # models
    p.add_argument("--ctx-frames", type=int, default=4, help="m previous steps as context (0 = unconditional flows); per view: model.ctx_frames.")
    p.add_argument("--burn", type=int, default=None, help="Positions < burn are context only (default: the largest ctx_frames).")
    p.add_argument("--image-L", type=int, default=3)
    p.add_argument("--image-K", type=int, default=16, help="Glow blocks per level.")
    p.add_argument("--image-hidden", type=int, default=128)
    p.add_argument("--ctx-ch", type=int, default=32, help="Channels of the image context pyramid (= image state dimension).")
    p.add_argument("--scale-cap", type=float, default=3.0)
    p.add_argument("--tabular-K", type=int, default=32, help="Coupling blocks of vector (tabular / signal1d) flows.")
    p.add_argument("--tabular-hidden", type=int, default=None)
    p.add_argument("--arch", default="mlp", choices=CONTEXT_ARCHS, help="Context encoder of vector views: mlp over the window, causal TCN (long/mid/short), lstm.")
    p.add_argument("--coupling", default="spline", choices=["spline", "affine"])
    p.add_argument("--ctx-dim", type=int, default=64, help="Context state dimension of vector views.")
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=4.0)
    # alignment (same names and defaults as the hybrid trainer)
    p.add_argument("--align", default="vicreg", choices=["none", "infonce", "barlow", "vicreg", "hsic", "pearson", "mse"])
    p.add_argument("--align-weight", type=float, default=0.05)
    p.add_argument("--align-warmup", type=int, default=200)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=128)
    p.add_argument("--alignment-latents", default="context", choices=["context"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--barlow-lambda", type=float, default=5e-3)
    p.add_argument("--weighting", default="fixed", choices=["fixed", "kendall"])
    p.add_argument("--init-logvar-nll", type=float, default=0.0)
    p.add_argument("--init-logvar-align", type=float, default=0.0)
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
    p.add_argument("--preview-interval", type=int, default=500)
    p.add_argument("--preview-samples", type=int, default=8)
    p.add_argument("--preview-columns", type=int, default=4)
    p.add_argument("--sample-mode", default="model", choices=["off", "model"])
    p.add_argument("--sample-temp", type=float, default=1.0)
    p.add_argument("--sample-grid-norm", default="to01", choices=["to01", "clamp", "both"])
    p.add_argument("--sample-chunk-size", type=int, default=20)                 # accepted, unused
    p.add_argument("--rollout-len", type=int, default=8, help="Length of the rollouts in the image sample previews.")
    p.add_argument("--display-labels", default="", help="Comma-separated image views drawn as label maps (or \"display\": \"labels\" in the config).")
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--keep-every", type=int, default=5000)
    p.add_argument("--disk-warning-gb", type=float, default=10.0)
    p.add_argument("--save-z", action="store_true")
    p.add_argument("--save-whitened", action="store_true")
    p.add_argument("--save-recon", action="store_true")
    p.add_argument("--export-max-samples", type=int, default=100)
    p.add_argument("--export-splits", default="val,test")
    p.add_argument("--export-only", action="store_true", help="Load best.pt (or --export-from last / --checkpoint) and export only.")
    p.add_argument("--export-from", default="best", choices=["best", "last"])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--untrained", action="store_true", help="--export-only: random weights of the same architecture (seed + 100).")
    p.add_argument("--export-tag", default="")
    p.add_argument("--diagnose-invertibility", action="store_true", help="Print round-trip metrics, write previews at iteration 0 and exit.")
    args = p.parse_args(argv)
    if args.accum_steps < 1:
        p.error("--accum-steps must be >= 1")
    if args.preview_samples < 1 or args.preview_columns < 1:
        p.error("--preview-samples and --preview-columns must be >= 1")
    if args.keep_last < 0 or args.keep_every < 0:
        p.error("checkpoint retention values must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    trainer = HybridTemporalLAMNrTrainer()
    args = _build_args(argv)
    trainer.setup(args)
    if args.diagnose_invertibility:
        trainer._save_previews(0)
    elif args.export_only:
        trainer.export()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
