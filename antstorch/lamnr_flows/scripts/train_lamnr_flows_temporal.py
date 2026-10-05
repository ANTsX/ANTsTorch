#!/usr/bin/env python3
"""LAMNr with temporal CONDITIONAL flows (multiview, signal1d windows).

Companion of ``train_lamnr_flows_hybrid`` for dynamic data. Same inputs (manifest + views JSON config with ``signal1d`` views,
``path_column`` / ``shape`` / ``layout: CL``), same kind of command line, but each view is modelled by an autoregressive
conditional flow over time instead of a static flow over the window:

    log p(x_{1:T}) = sum_t log p(x_t | h_{t-1}),    h = causal TCN state of the past frames of the same view.

Building blocks taken from the ANTsX packages:
  * ANTsNormalizingFlows : CoupledRationalQuadraticSpline (num_context_channels), Permute, distributions.base.ConditionalDiagGaussian,
                           core.ConditionalNormalizingFlow  (log_prob(x, context=h))
  * ANTsTorch            : lamnr_flows.misc.latent_alignment.Projector, lamnr_flows.misc.alignment_losses.vicreg_multi
Written here (neither package has a causal / recurrent encoder or a sequence wrapper): CausalTCN, window stitching, the loop.

Windows are stitched back into the per-pass sequences they were cut from (hop inferred from the overlap, which is verified);
every view is standardised with training statistics and receives a small fixed Gaussian noise (seeded per sequence, so train and
export see the same data). The shared embedding is z_v = Projector_v(mean_t h_t, t >= burn-1); VICReg aligns the views of the same
sequence; alignment weight 0 gives independent conditional flows. Loss = sum_v NLL_v (nats / dim) + align_weight * VICReg.

Modes
  train   (default)  iteration-based loop with warm-up, step decay, grad clipping, EMA, periodic validation NLL, last.pt / best.pt,
                     --auto-resume. best.pt holds the EMA weights of the best validation NLL.
  export  (--export-only) loads best.pt (or --checkpoint) and writes <out-dir>/export/<tag>.npz for ANY manifest: per-sequence test NLL
                     of each view (t >= burn), pooled context states h_v, shared embeddings z_v, raw statistics, metadata.
                     --untrained re-initialises the weights (same architecture, seed + 100): the reference for "does training matter".
Single GPU / CPU only (no DDP).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from antsnormflows.core import ConditionalNormalizingFlow
from antsnormflows.distributions.base import ConditionalDiagGaussian
from antsnormflows.flows import CoupledRationalQuadraticSpline, Permute
from antstorch.lamnr_flows.misc.alignment_losses import vicreg_multi
from antstorch.lamnr_flows.misc.latent_alignment import Projector

ARCHS = {"long": (3, (1, 2, 4)), "mid": (3, (1, 2)), "short": (2, (1,))}      # (kernel, dilations): receptive field 15 / 7 / 2 frames


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


def build_sequences(man: pd.DataFrame, root: Path, col: str, limit: int = 0, quiet: bool = False):
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
                h = hop
                ov = wins[i].shape[1] - h
                if float(np.abs(wins[i][:, :ov] - seq[:, -ov:]).mean()) > 1e-3 * (np.abs(seq).mean() + 1e-9):
                    ok = False
                seq = np.concatenate([seq, wins[i][:, -h:]], 1)
            n_bad += (not ok)
            row = g.iloc[r[0]]
            seqs.append(seq)
            meta.append(dict(pass_id=pid, start_frame=int(starts[r[0]]), subject_id=row.subject_id, split=str(row.get("split", "train")),
                             speed=str(row.speed).upper(), jacket=str(row.jacket) if "jacket" in g.columns else "all"))
    if not quiet:
        print(f"[{col}] stitched {len(seqs)} sequences (length median {int(np.median([s.shape[1] for s in seqs]))}, "
              f"min {min(s.shape[1] for s in seqs)}, max {max(s.shape[1] for s in seqs)}); hop {hop}; runs with overlap mismatch: {n_bad}")
    return seqs, pd.DataFrame(meta)


def standardise(seqs, stats, noise, noise_seed, meta):
    out = []
    for s, pid, st in zip(seqs, meta.pass_id, meta.start_frame):
        x = (s - stats["mu"]) / stats["sd"]
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

    def forward(self, x):                      # x (B, C, T) -> log p of positions 1..T-1 (B, T-1), states h_0..h_{T-2} (B, H, T-1)
        h = self.enc(x[:, :, :-1])
        B, _, T1 = h.shape
        tgt = x[:, :, 1:].permute(0, 2, 1).reshape(B * T1, -1)
        lp = self.flow.log_prob(tgt, context=h.permute(0, 2, 1).reshape(B * T1, -1)).view(B, T1)
        return lp, h


class Temporal(nn.Module):
    def __init__(self, Cs, a):
        super().__init__()
        self.views = nn.ModuleList([ViewModel(C, a) for C in Cs])
        self.proj = nn.ModuleList([Projector(a.ctx_dim, a.proj_hidden, a.proj_dim) for _ in Cs])


def pool_state(h, lens, burn):
    j = torch.arange(h.shape[2], device=h.device)[None]
    m = ((j >= burn - 1) & (j <= lens[:, None] - 2)).float()
    return (h * m[:, None]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)


def eval_mask(lens, T1, burn, device):
    t = torch.arange(1, T1 + 1, device=device)[None]
    valid = t < lens[:, None]
    return valid, valid & (t >= burn)


@torch.no_grad()
def validate(model, S, idx_all, dev, burn, bs=64):
    """Mean NLL per view (nats/dim, positions >= burn) over the sequences in idx_all."""
    model.eval()
    tot = [0.0] * len(S)
    cnt = 0
    for i in range(0, len(idx_all), bs):
        idx = list(idx_all[i:i + bs])
        for v, vm in enumerate(model.views):
            x, lens = pad_batch(S[v], idx, dev)
            lp, _ = vm(x)
            _, ev = eval_mask(lens, lp.shape[1], burn, dev)
            tot[v] += float(-(lp * ev).sum() / vm.C)
        cnt += int(eval_mask(lens, lp.shape[1], burn, dev)[1].sum())
    return [t / max(cnt, 1) for t in tot]


@torch.no_grad()
def export_arrays(model, S, meta, dev, burn, bs=64):
    model.eval()
    N, V = len(S[0]), len(S)
    out = {}
    for v in range(V):
        out[f"nll_{v}"] = np.full(N, np.nan)
        out[f"h_{v}"], out[f"z_{v}"] = [], []
    for i in range(0, N, bs):
        idx = list(range(i, min(i + bs, N)))
        for v, vm in enumerate(model.views):
            x, lens = pad_batch(S[v], idx, dev)
            lp, h = vm(x)
            _, ev = eval_mask(lens, lp.shape[1], burn, dev)
            nll = -(lp * ev).sum(1) / ev.sum(1).clamp(min=1) / vm.C
            nll[ev.sum(1) == 0] = float("nan")
            out[f"nll_{v}"][idx] = nll.cpu().numpy()
            hp = pool_state(h, lens, burn)
            out[f"h_{v}"].append(hp.cpu().numpy())
            out[f"z_{v}"].append(model.proj[v](hp).cpu().numpy())
    for v in range(V):
        out[f"h_{v}"] = np.concatenate(out[f"h_{v}"])
        out[f"z_{v}"] = np.concatenate(out[f"z_{v}"])
        out[f"raw_{v}"] = np.array([np.r_[s[:, burn - 1:].mean(1), s[:, burn - 1:].std(1)] if s.shape[1] > burn else
                                    np.full(2 * s.shape[0], np.nan) for s in S[v]])
    out["length"] = np.array([s.shape[1] for s in S[0]])
    for k in ("pass_id", "subject_id", "split", "speed", "jacket"):
        out[k] = meta[k].to_numpy().astype(str)
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True, help="views JSON (signal1d views: name, path_column, shape [C, L], layout CL)")
    p.add_argument("--out-dir", default="runs_temporal")
    p.add_argument("--devices", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="", help="folder holding phase2_*/ when the manifest paths are absolute elsewhere")
    p.add_argument("--limit-passes", type=int, default=0, help="smoke test: only the first N passes")
    # data
    p.add_argument("--val-fraction", type=float, default=0.125, help="subject-wise, used only if the manifest has no 'val' split")
    p.add_argument("--noise", type=float, default=0.02, help="fixed seeded Gaussian noise in SD units")
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--burn", type=int, default=8, help="positions < burn are context only (not scored)")
    # model
    p.add_argument("--arch", default="long", choices=list(ARCHS))
    p.add_argument("--ctx-dim", type=int, default=64)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=4.0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=5e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--patience-evals", type=int, default=0, help="stop after this many evaluations without improvement (0 = off)")
    p.add_argument("--smooth-alpha", type=float, default=0.1, help="smoothing of the loss shown in the progress bar")
    p.add_argument("--no-progress", action="store_true", help="disable the tqdm bar (periodic [val] lines are still printed)")
    p.add_argument("--resume", default="")
    p.add_argument("--auto-resume", action="store_true")
    # alignment (names as in the hybrid trainer)
    p.add_argument("--align", default="vicreg", choices=["vicreg", "none"])
    p.add_argument("--align-weight", type=float, default=0.1)
    p.add_argument("--align-warmup", type=int, default=500)
    p.add_argument("--proj-dim", type=int, default=8)
    p.add_argument("--proj-hidden", type=int, default=32)
    p.add_argument("--vicreg-inv", type=float, default=1.0)
    p.add_argument("--vicreg-var", type=float, default=1.0)
    p.add_argument("--vicreg-cov", type=float, default=0.04)
    p.add_argument("--vicreg-gamma", type=float, default=1.0)
    # export
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--untrained", action="store_true", help="with --export-only: random weights of the same architecture")
    p.add_argument("--checkpoint", default="", help="with --export-only: default <out-dir>/best.pt")
    p.add_argument("--export-tag", default="")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if a.devices == "cuda" and torch.cuda.is_available() else
                       "mps" if a.devices == "mps" else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = json.load(open(a.config))
    views = cfg["views"]
    assert all(v["type"] == "signal1d" for v in views), "only signal1d views are supported"
    assert len(views) >= 2, "need at least two views for the alignment"
    root = Path(a.data_root) if a.data_root else Path(a.manifest).resolve().parent.parent
    man = pd.read_csv(a.manifest)

    if a.export_only:
        ck = torch.load(a.checkpoint or out / "best.pt", map_location="cpu", weights_only=False)
        ta = argparse.Namespace(**{**ck["args"], "devices": a.devices})
        S0, meta = None, None
        raw = []
        for v in views:
            s, m = build_sequences(man, root, v["path_column"], a.limit_passes)
            raw.append(s)
            meta = m
        S = [standardise(raw[i], {"mu": ck["stats"][i]["mu"], "sd": ck["stats"][i]["sd"]}, ta.noise, ta.noise_seed, meta) for i in range(len(views))]
        model = Temporal([s[0].shape[0] for s in S], ta).to(dev)
        if a.untrained:
            torch.manual_seed(ta.seed + 100)
            model = Temporal([s[0].shape[0] for s in S], ta).to(dev)
            tag = a.export_tag or "untrained"
        else:
            model.load_state_dict(ck["model"])
            tag = a.export_tag or "features"
        arrays = export_arrays(model, S, meta, dev, ta.burn)
        arrays["view_names"] = np.array([v["name"] for v in views])
        (out / "export").mkdir(exist_ok=True)
        np.savez(out / "export" / f"{tag}.npz", **arrays)
        print(f"export: {out / 'export' / (tag + '.npz')}  ({len(meta)} sequences)")
        return

    # ---------------- data
    raw, meta = [], None
    for v in views:
        s, m = build_sequences(man, root, v["path_column"], a.limit_passes)
        raw.append(s)
        meta = m
    assert all(len(r) == len(raw[0]) and all(x.shape[1] == y.shape[1] for x, y in zip(r, raw[0])) for r in raw), "views differ in length"
    split = meta.split.to_numpy().copy()
    if "val" not in set(split):
        subs = np.array(sorted(set(meta.subject_id[split == "train"])))
        nv = max(1, int(round(a.val_fraction * len(subs))))
        vs = set(np.random.default_rng(a.seed).choice(subs, nv, replace=False))
        split = np.where(meta.subject_id.isin(vs), "val", split)
    tr_ix, va_ix = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    stats = []
    for r in raw:
        fr = np.concatenate([r[i] for i in tr_ix], 1)
        stats.append({"mu": fr.mean(1, keepdims=True), "sd": fr.std(1, keepdims=True) + 1e-6})
    S = [standardise(raw[i], stats[i], a.noise, a.noise_seed, meta) for i in range(len(views))]
    print(f"views {[v['name'] for v in views]} channels {[s[0].shape[0] for s in S]}; sequences train {len(tr_ix)}, val {len(va_ix)}; device {dev}")

    model = Temporal([s[0].shape[0] for s in S], a).to(dev)
    for vw, vm, pj in zip(views, model.views, model.proj):
        n_par = sum(q.numel() for q in vm.parameters()) + sum(q.numel() for q in pj.parameters())
        print(f"[init] {vw['name']} (temporal conditional signal1d): {n_par:,} parameters")
    print(f"[init] receptive field {1 + (ARCHS[a.arch][0] - 1) * sum(ARCHS[a.arch][1])} frames; context dim {a.ctx_dim}; K {a.K}; arch {a.arch}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    ema = copy.deepcopy(model.state_dict()) if a.ema else None
    it, best, bad = 0, 1e9, 0
    resume = a.resume or (str(out / "last.pt") if a.auto_resume and (out / "last.pt").exists() else "")
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["raw"])
        opt.load_state_dict(ck["opt"])
        ema = ck.get("ema", ema)
        it, best = ck["it"], ck["best"]
        tqdm.write(f"[resume] from {resume} @ iter {it}")

    def lr_at(i):
        if i < a.warmup_iters:
            return a.lr * (i + 1) / a.warmup_iters
        return max(a.min_lr, a.lr * (a.lr_decay_gamma ** ((i - a.warmup_iters) // max(a.lr_decay_steps, 1))))

    def save(path, weights, extra=None):
        torch.save({"model": weights, "raw": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "it": it, "best": best,
                    "args": vars(a), "stats": stats, "views": [v["name"] for v in views], **(extra or {})}, path)

    names = [v["name"] for v in views]
    log_path = out / "metrics.csv"
    if not log_path.exists():
        log_path.write_text("it,lr," + ",".join(f"train_nll_{n}" for n in names) + ",align," +
                            ",".join(f"val_nll_{n}" for n in names) + ",val_total\n")
    rng = np.random.default_rng(a.seed + it)
    order, pos = rng.permutation(tr_ix), 0
    run_nll, run_align, run_n = np.zeros(len(S)), 0.0, 0
    t0 = time.time()
    disp_loss = disp_align = None
    pbar = tqdm(total=a.max_iter, initial=it, desc="train-temporal", disable=a.no_progress)
    while it < a.max_iter:
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        opt.zero_grad()
        for _ in range(a.accum_steps):
            if pos + a.batch_size > len(order):
                order, pos = rng.permutation(tr_ix), 0
            idx = list(order[pos:pos + a.batch_size])
            pos += a.batch_size
            loss, pooled = 0.0, []
            for v, vm in enumerate(model.views):
                x, lens = pad_batch(S[v], idx, dev)
                lp, h = vm(x)
                valid, _ = eval_mask(lens, lp.shape[1], 1, dev)
                valid = valid & (torch.arange(1, lp.shape[1] + 1, device=dev)[None] >= a.burn)
                nll = -(lp * valid).sum() / valid.sum() / vm.C
                loss = loss + nll
                run_nll[v] += float(nll.detach())
                pooled.append(pool_state(h, lens, a.burn))
            if a.align == "vicreg" and a.align_weight > 0 and it >= a.align_warmup:
                z = [model.proj[v](pooled[v]) for v in range(len(S))]
                al = vicreg_multi(z, w_inv=a.vicreg_inv, w_var=a.vicreg_var, w_cov=a.vicreg_cov, gamma=a.vicreg_gamma)
                loss = loss + a.align_weight * al
                run_align += float(al.detach())
            (loss / a.accum_steps).backward()
            run_n += 1
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {it}")
        nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        it += 1
        pbar.update(1)
        cur = float(loss.detach())
        cur_align = run_align / max(run_n, 1)
        disp_loss = cur if disp_loss is None else (1 - a.smooth_alpha) * disp_loss + a.smooth_alpha * cur
        disp_align = cur_align if disp_align is None else (1 - a.smooth_alpha) * disp_align + a.smooth_alpha * cur_align
        pbar.set_postfix(loss=f"{disp_loss:.4f}", align=f"{disp_align:.4f}")
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[k].copy_(v)
        if it % a.eval_interval == 0 or it == a.max_iter:
            raw_state = copy.deepcopy(model.state_dict())
            if ema is not None:
                model.load_state_dict(ema)
            vn = validate(model, S, va_ix, dev, a.burn)
            vt = float(sum(vn))
            weights = copy.deepcopy(model.state_dict())
            model.load_state_dict(raw_state)
            if vt < best - 1e-4:
                best, bad = vt, 0
                save(out / "best.pt", weights, {"best_it": it})
                tqdm.write(f"[ckpt] saved best.pt (iter {it}, val {vt:.6g})")
            else:
                bad += 1
            save(out / "last.pt", weights)
            row = [it, f"{lr_at(it):.2e}"] + [f"{x / max(run_n, 1):.4f}" for x in run_nll] + [f"{run_align / max(run_n, 1):.4f}"] + \
                  [f"{x:.4f}" for x in vn] + [f"{vt:.4f}"]
            with open(log_path, "a") as f:
                f.write(",".join(map(str, row)) + "\n")
            tqdm.write(f"[val] iter={it} loss={vt:.6g} ({'EMA' if ema is not None else 'base'}; "
                       + ", ".join(f"{n}={x:.4g}" for n, x in zip(names, vn)) + f"; best={best:.6g}; "
                       + f"train " + ", ".join(f"{n}={x / max(run_n, 1):.4g}" for n, x in zip(names, run_nll))
                       + f"; align={run_align / max(run_n, 1):.4g}; lr={lr_at(it):.2e}; {(time.time() - t0) / 60:.1f} min)")
            run_nll, run_align, run_n = np.zeros(len(S)), 0.0, 0
            if a.patience_evals and bad >= a.patience_evals:
                tqdm.write(f"[early-stop] no improvement for {bad} evaluations")
                break
    pbar.close()
    print(f"done. best validation NLL (sum over views) {best:.4f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
