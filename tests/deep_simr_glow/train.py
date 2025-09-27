# train_glow_model_joint_2d_uncertainty_es_plus_dir_safeguarded.py
# ----------------------------------------------------------------------------
# Based on: train_glow_model_joint_2d_uncertainty_es_plus_dir.py
# Adds the requested "minimal first-step" stability trio:
#   1) NaN/Inf sentry checks (loss, grads, probe forward pass) with diagnostics
#   2) Gradient clipping (correctly ordered with AMP) + grad-norm logging
#   3) LR warmup (existing) + ReduceLROnPlateau on avg RAW val_bpd
# Plus:
#   - Numeric explosion sentries on |z| and |sample|
#   - CSV logging for z_abs_max
# ----------------------------------------------------------------------------

import os, json, math, csv, shutil, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader

import normflows as nf
from normflows import distributions as nfd

import ants, antstorch

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Helpers
# ---------------------------

def set_deterministic(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def nonfinite_and_outlier_clean(x: torch.Tensor, clamp_abs: float = 1e6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None:
        x = torch.clamp(x, -clamp_abs, clamp_abs)
    return x

def per_sample_minmax01(x: torch.Tensor, eps: float = 1e-8):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def bits_per_dim_from_kld(model, x: torch.Tensor, num_dims: int):
    b = x.shape[0]
    total = 0.0
    with torch.no_grad():
        for i in range(b):
            loss_i = model.forward_kld(x[i:i+1, ...])
            total += float(loss_i.detach().cpu().item())
    avg = total / b
    return avg / (np.log(2.0) * num_dims)

def simple_moving_average(x, w=200):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:  # no smoothing if too short
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="valid")  # length N-w+1

def evaluate_val_bpd(models, loader, device, num_dims, max_batches=10):
    for m in range(len(models)): models[m].eval()
    view_sums = [0.0 for _ in models]; view_counts = [0 for _ in models]
    batches_done = 0
    for x in loader:
        if batches_done >= max_batches: break
        for m in range(len(models)):
            x_m = x[:, m:m+1, :, :].to(device)
            x_m = per_sample_minmax01(nonfinite_and_outlier_clean(x_m)).to(next(models[m].parameters()).dtype)
            bpd = bits_per_dim_from_kld(models[m], x_m, num_dims)
            view_sums[m] += bpd; view_counts[m] += 1
        batches_done += 1
    vals = [ (view_sums[m] / max(1, view_counts[m])) for m in range(len(models)) ]
    for m in range(len(models)): models[m].train()
    return vals

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True); return p

def make_warmup_scheduler(optimizer, warmup_iters:int, decay_gamma:float, decay_steps:int):
    """
    Keeps previous LambdaLR behavior for warmup (and optional exp decay).
    This plays fine with an additional ReduceLROnPlateau that we step at eval.
    """
    if warmup_iters<=0 and (decay_gamma==1.0 or decay_steps<=0):
        return None
    def lr_lambda(step):
        s = max(1, step)  # 1-based for stability
        scale = 1.0
        if warmup_iters > 0 and s < warmup_iters:
            scale *= s / float(warmup_iters)
        if decay_gamma != 1.0 and decay_steps > 0:
            scale *= (decay_gamma ** (s / float(decay_steps)))
        return scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def clone_ema_model(model):
    import copy as _copy
    ema = _copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema

@torch.no_grad()
def ema_update(ema_model, model, decay: float):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

# ---- Sentry tools -----------------------------------------------------------

def grad_is_finite_and_below(param_list, max_abs: float = 1e6):
    for p in param_list:
        if p.grad is None: 
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            return False
        if torch.isnan(g).any() or torch.isinf(g).any():
            return False
        if g.abs().max().item() > max_abs:
            return False
    return True

@torch.no_grad()
def probe_forward(models, x_probe, device, model_dtype,
                  z_absmax_thresh=None,
                  sample_absmax_thresh=None,
                  sample_quantile: float = 0.999):
    """
    Quantile-based sample sentry: robust to lone extreme pixels.
    Also reports z-abs-max (strict) for visibility.
    """
    stats = {"ok": True, "per_view": []}
    try:
        for m in range(len(models)):
            x_m = per_sample_minmax01(nonfinite_and_outlier_clean(
                x_probe[:, m:m+1, :, :].to(device)
            )).to(model_dtype)

            # Smoke-test core ops
            z_m, _ = models[m].inverse_and_log_det(x_m)
            kld = models[m].forward_kld(x_m)
            xs, _ = models[m].sample(x_m.shape[0])

            # z stats (strict max for observability)
            zs = torch.cat([z_m[p].reshape(z_m[p].shape[0], -1) for p in range(len(z_m))], dim=1)
            z_abs_max = float(zs.abs().max().item())
            z_mean = float(zs.mean().item())

            # sample stats: quantile across pixels per sample, then max across batch
            flat = xs.abs().reshape(xs.shape[0], -1)
            q = float(sample_quantile)
            if q >= 1.0:
                x_q_stat = float(flat.max().item())
            else:
                xq = torch.quantile(flat, q, dim=1)   # shape [batch]
                x_q_stat = float(xq.max().item())

            stats["per_view"].append({
                "z_abs_max": z_abs_max,
                "z_mean": z_mean,
                "x_q_stat": x_q_stat,
                "kld": float(kld.detach().cpu().item())
            })

            if (not np.isfinite(z_abs_max)) or (not np.isfinite(z_mean)) or (not np.isfinite(x_q_stat)):
                stats["ok"] = False
            if (z_absmax_thresh is not None) and (z_abs_max > float(z_absmax_thresh)):
                stats["ok"] = False
            if (sample_absmax_thresh is not None) and (x_q_stat > float(sample_absmax_thresh)):
                stats["ok"] = False
    except Exception as e:
        stats["ok"] = False
        stats["error"] = str(e)
    return stats

def dump_sentry_package(trial_dir: Path, it: int, models, optimizer, loss_val: float, extra: dict):
    dump_dir = ensure_dir(trial_dir / f"SENTRY_DUMP_iter{it:07d}")
    # Save model weights
    for mi, mod in enumerate(extra.get("modalities", [f"view{mi}" for mi in range(len(models))])):
        try:
            models[mi].save(str(dump_dir / f"model_{mod}.pt"))
        except Exception:
            torch.save(models[mi].state_dict(), dump_dir / f"model_{mod}_state_dict.pt")
    # Save optimizer state
    try:
        torch.save(optimizer.state_dict(), dump_dir / "optimizer_state.pt")
    except Exception:
        pass
    # Save meta JSON
    meta = {
        "iter": it,
        "loss": float(loss_val) if np.isfinite(loss_val) else str(loss_val),
        "extra": extra,
    }
    with open(dump_dir/"meta.json","w") as f:
        json.dump(meta, f, indent=2)
    return dump_dir

# ---------------------------
# CLI
# ---------------------------

import argparse
parser = argparse.ArgumentParser(description="Glow 2D with learned uncertainty, early stopping + AMP/EMA/scheduler/init/resume + penalty direction + sentry checks.")
parser.add_argument("--devices", type=str, default="cuda:0")
parser.add_argument("--modalities", type=str, nargs="+", default=["T2","T1","FA"])
parser.add_argument("--L", type=int, default=4)
parser.add_argument("--K", type=int, default=2)
parser.add_argument("--hidden-channels", type=int, default=64)
parser.add_argument("--resampled-size", type=int, nargs=2, default=[128,128])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--max-iter", type=int, default=100000)
parser.add_argument("--plot-interval", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=1000)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--train-samples", type=int, default=100)
parser.add_argument("--val-samples", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min-lr", type=float, default=1e-6, help="Floor LR for Plateau scheduler.")
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--penalty", type=str, default="pearson", choices=["pearson","mine"], help="Cross-view dependence measure.")
# NEW: Direction + targets
parser.add_argument("--penalty-direction", type=str, default="separate", choices=["separate","align"],
                    help="'separate' = decorrelate/independence; 'align' = encourage correlation/MI")
parser.add_argument("--mi-target", type=float, default=0.2, help="Target MI for align mode with MINE (hinge at tau).")
parser.add_argument("--mine-lr", type=float, default=1e-6)
parser.add_argument("--mine-update-freq", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="runs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--val-max-batches", type=int, default=10)
parser.add_argument("--init-log-sigma-kld", type=float, default=0.0)
parser.add_argument("--init-log-sigma-pen", type=float, default=0.0)
# Early stopping
parser.add_argument("--early-stopping-patience", type=int, default=0, help="Patience in eval steps; 0 disables early stopping")
parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement in avg val_bpd to reset patience")
# Precision / AMP
parser.add_argument("--precision", type=str, default="double", choices=["double","float","mixed"], help="Compute precision: double=fp64, float=fp32, mixed=AMP on fp32 weights")
# EMA
parser.add_argument("--ema", action="store_true", help="Enable EMA evaluation/sampling")
parser.add_argument("--ema-decay", type=float, default=0.999)
# LR schedule (warmup retained; exp decay optional)
parser.add_argument("--warmup-iters", type=int, default=400)
parser.add_argument("--lr-decay-gamma", type=float, default=1.0, help="Exp decay gamma per (steps/decay_steps). Set 1.0 to disable.")
parser.add_argument("--lr-decay-steps", type=int, default=0)
# NEW: ReduceLROnPlateau on avg RAW val_bpd
parser.add_argument("--plateau-factor", type=float, default=0.5, help="Multiply LR by this factor on plateau.")
parser.add_argument("--plateau-patience", type=int, default=3, help="Eval windows to wait before reducing LR.")
parser.add_argument("--plateau-threshold", type=float, default=1e-4, help="Min improvement to reset plateau patience.")
parser.add_argument("--plateau-cooldown", type=int, default=0)
# Init & resume
parser.add_argument("--actnorm-init", action="store_true", help="Run a data-dependent init pass for ActNorm on first batch")
parser.add_argument("--resume", type=str, default="", help="Path to an existing run dir to resume from (expects training_state.pt)")
parser.add_argument("--base", type=str, default="diag", choices=["diag","glow"])
parser.add_argument("--glowbase-logscale-factor", type=float, default=3.0)
parser.add_argument("--glowbase-min-log", type=float, default=-5.0)
parser.add_argument("--glowbase-max-log", type=float, default=5.0)
parser.add_argument("--scale-map", type=str, default="tanh", choices=["tanh","exp","sigmoid","sigmoid_inv"], help="Coupling scale map for Glow blocks")
parser.add_argument("--scale-cap", type=float, default=3.0, help="Clamp on log-scale (|s|<=cap) inside coupling when scale-map uses tanh-bound")
parser.add_argument("--net-actnorm", action="store_true", help="Enable ActNorm in coupling subnets (ConvNet2d/3d)")
parser.add_argument("--net-leaky", type=float, default=0.1, help="Leaky slope for coupling subnets")
# NEW: Sentry controls
parser.add_argument("--sentry-interval", type=int, default=200, help="How often (iters) to run sentry checks. 0 disables.")
parser.add_argument("--sentry-grad-absmax", type=float, default=1e6, help="Declares non-finite if any grad has abs() > this.")
parser.add_argument("--sentry-sample-quantile", type=float, default=0.999, help="Quantile of |sample| used for sentry (1.0 = strict max).")
parser.add_argument("--sentry-probe-batchsize", type=int, default=4, help="Mini-batch size for probe forward.")
parser.add_argument("--sentry-dump", action="store_true", help="Dump models/optimizer/meta if a sentry fails.")
parser.add_argument("--sentry-z-absmax", type=float, default=1e3, help="Fail sentry if any |z| exceeds this.")
parser.add_argument("--sentry-sample-absmax", type=float, default=10.0, help="Fail sentry if any sample pixel exceeds this (pre-clamp).")

args = parser.parse_args()

set_deterministic(args.seed)

# Devices
dev_list = [d.strip() for d in args.devices.split(",")]
device = torch.device("cpu") if dev_list[0].lower()=="cpu" else torch.device(dev_list[0])

# Precision setup
if args.precision == "double":
    model_dtype = torch.float64
    amp_enabled = False
elif args.precision == "float":
    model_dtype = torch.float32
    amp_enabled = False
else:  # mixed
    model_dtype = torch.float32
    amp_enabled = True
    amp_dtype = torch.float16  # safe default
scaler_flow = torch.amp.GradScaler('cuda', enabled=amp_enabled)
scaler_mine = torch.amp.GradScaler('cuda', enabled=amp_enabled)

# ---------------------------
# Define flow arch per view
# ---------------------------

channels = 1; resampled_image_size = tuple(args.resampled_size)
input_shape = (channels, *resampled_image_size); n_dims = int(np.prod(input_shape))
L,K,hidden_channels = args.L, args.K, args.hidden_channels
split_mode, scale = 'channel', True
use_mutual_information_penalty = (args.penalty == "mine")

models = []; combined_model_parameters=[]; mine_latent_dim=0
print(f"[GlowConfig] scale_map={args.scale_map} scale_cap={args.scale_cap} net_actnorm={args.net_actnorm} net_leaky={args.net_leaky}")
for m in range(len(args.modalities)):
    q0=[]; merges=[]; flows=[]
    for i in range(L):
        flows_ = [nf.flows.GlowBlock2d(channels * 2 ** (L+1-i), hidden_channels, split_mode=split_mode, scale=scale,
                                scale_map=args.scale_map, leaky=args.net_leaky, net_actnorm=args.net_actnorm,
                                s_cap=args.scale_cap) for j in range(K)]
        flows_ += [nf.flows.Squeeze2d()]; flows += [flows_]
        latent_shape = (input_shape[0] * 2 ** (L-i if i>0 else L+1),
                        input_shape[1] // 2 ** (L-i if i>0 else L),
                        input_shape[2] // 2 ** (L-i if i>0 else L))
        if m==0: mine_latent_dim += math.prod(latent_shape)
        if args.base == "glow":
            q0 += [nfd.GlowBase(
                latent_shape,
                logscale_factor=args.glowbase_logscale_factor,
                min_log=args.glowbase_min_log,
                max_log=args.glowbase_max_log
            )]
        else:
            q0 += [nfd.DiagGaussian(latent_shape)]
        if i>0: merges += [nf.flows.Merge()]
    model = nf.MultiscaleFlow(q0, flows, merges)
    models.append(model); combined_model_parameters+=list(model.parameters())
for m in range(len(models)):
    models[m]=models[m].to(device).to(model_dtype).train()

# Learnable log-sigmas (match model dtype)
log_sigma_kld = torch.nn.Parameter(torch.tensor(float(args.init_log_sigma_kld), dtype=model_dtype, device=device))
log_sigma_pen = torch.nn.Parameter(torch.tensor(float(args.init_log_sigma_pen), dtype=model_dtype, device=device))
combined_model_parameters += [log_sigma_kld, log_sigma_pen]

# EMA setup (flow models only)
ema_models = None
if args.ema:
    ema_models = [clone_ema_model(models[m]) for m in range(len(models))]

# Penalty setup
if use_mutual_information_penalty:
    penalty_string="Mutual information"; mine_nets=[]; ma_ets=[]; combined_penalty_parameters=[]
    for n in range(sum(1 for m in range(len(models)) for n in range(m+1,len(models)))):
        net=antstorch.MINE(mine_latent_dim,mine_latent_dim).to(device).to(model_dtype)
        mine_nets.append(net); combined_penalty_parameters+=list(net.parameters()); ma_ets.append(None)
else: penalty_string="Pearson Correlation"; mine_nets=None; ma_ets=None; combined_penalty_parameters=[]

# Data
hcpya_images=[ants.image_read(antstorch.get_antstorch_data("hcpyaT2Template")),
              ants.image_read(antstorch.get_antstorch_data("hcpyaT1Template")),
              ants.image_read(antstorch.get_antstorch_data("hcpyaFATemplate"))]
hcpya_slices=[ants.slice_image(im,axis=2,idx=120,collapse_strategy=1) for im in hcpya_images]
template=ants.resample_image(hcpya_slices[0],resampled_image_size,use_voxels=True)
train_loader=DataLoader(antstorch.ImageDataset(images=[hcpya_slices],template=template,do_data_augmentation=True,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.02,data_augmentation_sd_deformation=10.0,
            number_of_samples=int(args.train_samples)),
            batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
val_loader=DataLoader(antstorch.ImageDataset(images=[hcpya_slices],template=template,do_data_augmentation=True,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.05,data_augmentation_sd_deformation=0.2,
            data_augmentation_noise_model="additivegaussian",data_augmentation_sd_simulated_bias_field=1.0,
            data_augmentation_sd_histogram_warping=0.05,number_of_samples=int(args.val_samples)),
            batch_size=min(16,args.batch_size),shuffle=False,num_workers=max(1,args.num_workers//2))
train_iter=iter(train_loader)

# Trial dir & resume
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.resume:
    trial_dir = Path(args.resume)
    if not trial_dir.exists(): raise FileNotFoundError(f"--resume path not found: {trial_dir}")
else:
    trial_dir=ensure_dir(Path(args.out_dir)/f"glow_joint2d_uncertES_plus_dir_safeguarded_{timestamp}")

state_path = trial_dir/"training_state.pt"
log_path=trial_dir/"metrics_log.csv"
config_path = trial_dir/"config.json"

# Save/merge config
config = {
    "timestamp": timestamp,
    "devices": args.devices,
    "modalities": args.modalities,
    "L": L, "K": K, "hidden_channels": hidden_channels,
    "resampled_size": resampled_image_size,
    "batch_size": args.batch_size,
    "max_iter": args.max_iter,
    "plot_interval": args.plot_interval,
    "eval_interval": args.eval_interval,
    "train_samples": args.train_samples,
    "val_samples": args.val_samples,
    "lr": args.lr, "weight_decay": args.weight_decay,
    "penalty": args.penalty,
    "penalty_direction": args.penalty_direction,
    "mi_target": args.mi_target,
    "mine_lr": args.mine_lr, "mine_update_freq": args.mine_update_freq,
    "seed": args.seed,
    "precision": args.precision,
    "ema": args.ema, "ema_decay": args.ema_decay,
    "warmup_iters": args.warmup_iters,
    "lr_decay_gamma": args.lr_decay_gamma, "lr_decay_steps": args.lr_decay_steps,
    "plateau": {
        "factor": args.plateau_factor,
        "patience": args.plateau_patience,
        "threshold": args.plateau_threshold,
        "cooldown": args.plateau_cooldown,
        "min_lr": args.min_lr,
    },
    "actnorm_init": args.actnorm_init,
    "init_log_sigma_kld": args.init_log_sigma_kld,
    "init_log_sigma_pen": args.init_log_sigma_pen,
    "sentry": {
        "interval": args.sentry_interval,
        "grad_absmax": args.sentry_grad_absmax,
        "probe_batchsize": args.sentry_probe_batchsize,
        "dump": bool(args.sentry_dump),
        "z_absmax": args.sentry_z_absmax,
        "sample_absmax": args.sentry_sample_absmax,
    },
    "normflows_version": getattr(nf, "__version__", "unknown"),
    "torch_version": torch.__version__
}
# On fresh run, write config; on resume, do not overwrite
if not args.resume:
    with open(config_path, "w") as f: json.dump(config, f, indent=2)

# CSV header
if not args.resume or not log_path.exists():
    with open(log_path,"w",newline="") as f:
        writer=csv.writer(f)
        header=["iter","loss_total","loss_kld","penalty_term","log_sigma_kld","log_sigma_pen",
                "w_kld","w_pen",
                "avg_val_bpd_raw","avg_val_bpd_ema","lr","grad_norm","z_abs_max","sentry_ok"]
        for m in range(len(models)): header+=[f"val_bpd_{args.modalities[m]}"]
        writer.writerow(header)

# Checkpoints
last_ckpts=[trial_dir/f"model_{mod}_last.pt" for mod in args.modalities]
best_ckpts=[trial_dir/f"model_{mod}_best.pt" for mod in args.modalities]
best_val_bpd=[float("inf")]*len(models)  # per view
best_avg=float("inf"); no_improve=0
start_iter = 1

# Optimizers
flow_optimizer=torch.optim.Adamax(combined_model_parameters,lr=float(args.lr),weight_decay=float(args.weight_decay))
mine_optimizer=torch.optim.Adamax(combined_penalty_parameters,lr=float(args.mine_lr),weight_decay=float(args.weight_decay)) if use_mutual_information_penalty else None

# Schedulers
warmup_sched = make_warmup_scheduler(flow_optimizer, args.warmup_iters, args.lr_decay_gamma, args.lr_decay_steps)
plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    flow_optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience,
    threshold=args.plateau_threshold, cooldown=args.plateau_cooldown, min_lr=args.min_lr, verbose=True
)

# ---------------------------
# Resume logic (if any)
# ---------------------------
if args.resume and state_path.exists():
    ckpt = torch.load(state_path, map_location=device, weights_only=False)
    # Load flow weights if last exists
    for m in range(len(models)):
        model_path = trial_dir / f"model_{args.modalities[m]}_last.pt"
        if model_path.exists():
            models[m].load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    # Load optimizers/scheduler/scalers/state
    flow_optimizer.load_state_dict(ckpt.get("flow_optimizer", flow_optimizer.state_dict()))
    if mine_optimizer and "mine_optimizer" in ckpt and ckpt["mine_optimizer"] is not None: mine_optimizer.load_state_dict(ckpt["mine_optimizer"])
    if warmup_sched and "scheduler" in ckpt and ckpt["scheduler"] is not None: warmup_sched.load_state_dict(ckpt["scheduler"])
    if "best_val_bpd" in ckpt: best_val_bpd = ckpt["best_val_bpd"]
    if "best_avg" in ckpt: best_avg = ckpt["best_avg"]
    if "no_improve" in ckpt: no_improve = ckpt["no_improve"]
    if "iter" in ckpt: start_iter = ckpt["iter"]
    # EMA
    if args.ema and "ema_state_dicts" in ckpt and ckpt["ema_state_dicts"] is not None:
        ema_models = [clone_ema_model(models[m]) for m in range(len(models))]  # ensure same arch
        for m, sd in enumerate(ckpt["ema_state_dicts"]):
            ema_models[m].load_state_dict(sd)
    print(f"Resumed from {state_path}, starting at iter {start_iter}")

# ---------------------------
# ActNorm data-dependent init
# ---------------------------
if args.actnorm_init and not args.resume:
    try:
        with torch.no_grad():
            x_init = next(train_iter)
            for m in range(len(models)):
                x_m = per_sample_minmax01(nonfinite_and_outlier_clean(x_init[:,m:m+1,:,:].to(device))).to(model_dtype)
                _ = models[m].forward_kld(x_m)  # one pass to set ActNorm stats
        print("ActNorm data-dependent init done.")
    except Exception as e:
        print(f"ActNorm init skipped due to: {e}")

# ---------------------------
# Training loop
# ---------------------------

loss_iter = []; loss_kld_hist = []; penalty_hist = []; loss_hist = []; loss_conv = []
grad_norm_hist = []; sentry_ok_hist = []; z_abs_max_hist = []
w_kld_hist, w_pen_hist = [], []

for it in tqdm(range(start_iter, int(args.max_iter)+1)):
    flow_optimizer.zero_grad()
    if mine_optimizer: mine_optimizer.zero_grad()

    # Fetch batch
    try: x = next(train_iter)
    except StopIteration: train_iter = iter(train_loader); x = next(train_iter)

    # Encode each view
    z=[]; loss_kld = torch.tensor(0.0, device=device, dtype=model_dtype)
    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=(amp_dtype if amp_enabled else None)):
        for m in range(len(models)):
            x_m = per_sample_minmax01(nonfinite_and_outlier_clean(x[:,m:m+1,:,:].to(device))).to(model_dtype)
            z_m, _ = models[m].inverse_and_log_det(x_m)
            z.append(z_m)
            loss_kld = loss_kld + models[m].forward_kld(x_m)

    # Quick z explosion guard (per batch) — use 99.9% quantile of |z| instead of strict max
    try:
        q = 0.999  # quantile for the z-stat guard
        z_view_stats = []
        for m in range(len(z)):
            # Flatten multiscale latents for this view and compute per-sample |z|
            flat_m = torch.cat(
                [z[m][p].reshape(z[m][p].shape[0], -1).to(torch.float32) for p in range(len(z[m]))],
                dim=1,
            ).abs()
            # Per-sample quantile, then take batch max for this view
            zq_m = torch.quantile(flat_m, q, dim=1)  # shape: [batch]
            z_view_stats.append(zq_m.max())
        # Max across views → single batch statistic
        z_abs_max = float(torch.stack(z_view_stats).max().item())
    except Exception:
        z_abs_max = float("nan")

    if np.isfinite(z_abs_max) and z_abs_max > float(args.sentry_z_absmax):
        extra = {
            "why": f"z |.| q{q*100:.1f} over threshold",
            "z_q": float(z_abs_max),
            "modalities": args.modalities,
        }
        if args.sentry_dump:
            dump_dir = dump_sentry_package(trial_dir, it, models, flow_optimizer, float("nan"), extra)
            print(f"[SENTRY] z quantile guard at iter {it}. Dumped to: {dump_dir}")
        raise FloatingPointError(
            f"z |.| q{q*100:.1f} = {z_abs_max} exceeds threshold {args.sentry_z_absmax} at iter {it}"
        )

    # Cross-view penalty (depends on direction)
    penalty_term = torch.tensor(0.0, device=device, dtype=model_dtype)
    pair_idx = 0
    for m in range(len(models)):
        for n in range(m+1,len(models)):
            zm = torch.cat([z[m][p].reshape(z[m][p].shape[0],-1).to(torch.float32) for p in range(len(z[m]))],dim=1)
            zn = torch.cat([z[n][p].reshape(z[n][p].shape[0],-1).to(torch.float32) for p in range(len(z[n]))],dim=1)
            if use_mutual_information_penalty:
                with torch.amp.autocast('cuda',enabled=amp_enabled, dtype=(amp_dtype if amp_enabled else None)):
                    mi_est, ma_ets[pair_idx] = antstorch.mutual_information_mine(zm, zn, mine_nets[pair_idx], ma_ets[pair_idx], alpha=0.0001, loss_type='fdiv')
                # Train MINE periodically (maximize MI)
                if it % args.mine_update_freq == 0:
                    scaler_mine.scale(-mi_est).backward(retain_graph=True)
                    # Clip MINE grads
                    torch.nn.utils.clip_grad_norm_(mine_nets[pair_idx].parameters(), max_norm=float(args.grad_clip))
                    scaler_mine.step(mine_optimizer); scaler_mine.update(); mine_optimizer.zero_grad()
                # Direction shaping
                if args.penalty_direction == "separate":
                    term = mi_est
                else:  # align
                    tau = float(args.mi_target)
                    tau_t = torch.tensor(tau, device=mi_est.device, dtype=mi_est.dtype)
                    term = torch.relu(tau_t - mi_est)
                penalty_term = penalty_term + term.to(model_dtype)
            else:
                corr_value = antstorch.absolute_pearson_correlation(zm, zn, 1e-6)  # in [0,1]
                if args.penalty_direction == "separate":
                    term = corr_value
                else:  # align
                    term = torch.clamp(1.0 - corr_value, min=0.0)
                penalty_term = penalty_term + term.to(model_dtype)
            pair_idx += 1

    # Learned-uncertainty combination
    clamped_lsig_kld = torch.clamp(log_sigma_kld, -5.0, 5.0)
    clamped_lsig_pen = torch.clamp(log_sigma_pen, -5.0, 5.0)

    w_kld = 0.5 * torch.exp(-2*clamped_lsig_kld)  # scalar
    w_pen = 0.5 * torch.exp(-2*clamped_lsig_pen)  # scalar
    w_kld_hist.append(float(w_kld.detach().cpu().item()))
    w_pen_hist.append(float(w_pen.detach().cpu().item()))

    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=(amp_dtype if amp_enabled else None)):
        loss = (0.5 * torch.exp(-2*clamped_lsig_kld) * loss_kld + clamped_lsig_kld) \
             + (0.5 * torch.exp(-2*clamped_lsig_pen) * penalty_term + clamped_lsig_pen)

    # Sentry pre-check on loss
    if not torch.isfinite(loss):
        extra = {"why": "non-finite loss", "modalities": args.modalities}
        if args.sentry_dump:
            dump_dir = dump_sentry_package(trial_dir, it, models, flow_optimizer, float("nan"), extra)
            print(f"[SENTRY] Non-finite loss at iter {it}. Dumped to: {dump_dir}")
        raise FloatingPointError(f"Non-finite loss at iter {it}")

    # Backprop + clip + step
    if amp_enabled:
        scaler_flow.scale(loss).backward()
        # Unscale gradients before clipping so max_norm is meaningful
        scaler_flow.unscale_(flow_optimizer)
        total_grad_norm = float(torch.nn.utils.clip_grad_norm_(combined_model_parameters, max_norm=float(args.grad_clip)))
        scaler_flow.step(flow_optimizer); scaler_flow.update()
    else:
        loss.backward()
        total_grad_norm = float(torch.nn.utils.clip_grad_norm_(combined_model_parameters, max_norm=float(args.grad_clip)))
        flow_optimizer.step()

    # Sentry gradient check (every N steps)
    sentry_ok = True
    if args.sentry_interval > 0 and (it % args.sentry_interval == 0):
        sentry_ok = grad_is_finite_and_below(combined_model_parameters, max_abs=float(args.sentry_grad_absmax))
        # Probe forward on a tiny batch from val set
        try:
            x_probe = next(iter(val_loader))
            x_probe = x_probe[:max(1, int(args.sentry_probe_batchsize))]
            pf_stats = probe_forward(models, x_probe, device, model_dtype,
                                     z_absmax_thresh=args.sentry_z_absmax,
                                     sample_absmax_thresh=args.sentry_sample_absmax,
                                     sample_quantile=float(args.sentry_sample_quantile))
            sentry_ok = sentry_ok and pf_stats.get("ok", False)
        except Exception as e:
            pf_stats = {"ok": False, "error": str(e)}
            sentry_ok = False
        if not sentry_ok:
            extra = {
                "why": "sentry failure (grad or probe forward)",
                "grad_ok": bool(grad_is_finite_and_below(combined_model_parameters, max_abs=float(args.sentry_grad_absmax))),
                "probe": pf_stats,
                "lr": float(flow_optimizer.param_groups[0]["lr"]),
                "modalities": args.modalities,
            }
            if args.sentry_dump:
                dump_dir = dump_sentry_package(trial_dir, it, models, flow_optimizer, float(loss.detach().cpu().item()), extra)
                print(f"[SENTRY] Failure at iter {it}. Dumped to: {dump_dir}")
            raise FloatingPointError(f"Sentry failure at iter {it}: {extra}")

    # EMA update after optimizer step
    if ema_models is not None:
        for m in range(len(models)):
            ema_update(ema_models[m], models[m], decay=float(args.ema_decay))

    # Scheduler step (warmup/exp-decay)
    if warmup_sched is not None:
        warmup_sched.step()

    # Track
    curr_loss = float(loss.detach().cpu().item())
    loss_hist.append(curr_loss)
    loss_kld_hist.append(float(loss_kld.detach().cpu().item()))
    penalty_hist.append(float(penalty_term.detach().cpu().item()))
    loss_iter.append(it)
    grad_norm_hist.append(total_grad_norm)
    z_abs_max_hist.append(float(z_abs_max))
    sentry_ok_hist.append(bool(sentry_ok))
    try: loss_conv.append(float(ants.convergence_monitoring(np.array(loss_hist), 100)))
    except Exception: loss_conv.append(np.nan)

    # Periodic evaluation
    write_state = False
    if it % args.eval_interval == 0:
        val_bpds_raw = evaluate_val_bpd(models, val_loader, device, n_dims, max_batches=args.val_max_batches)
        avg_val_raw = float(np.mean(val_bpds_raw))

        # EMA eval
        avg_val_ema = float("nan")
        if ema_models is not None:
            val_bpds_ema = evaluate_val_bpd(ema_models, val_loader, device, n_dims, max_batches=args.val_max_batches)
            avg_val_ema = float(np.mean(val_bpds_ema))

        # Save last + best-per-view (raw)
        for m in range(len(models)):
            models[m].save(str(last_ckpts[m]))
            if val_bpds_raw[m] < best_val_bpd[m]:
                best_val_bpd[m] = val_bpds_raw[m]
                models[m].save(str(best_ckpts[m]))

        # ReduceLROnPlateau step on RAW avg val_bpd
        plateau_sched.step(avg_val_raw)

        # Log row
        with open(log_path,"a",newline="") as f:
            writer=csv.writer(f)
            lr_now = flow_optimizer.param_groups[0]["lr"]
            row=[it,
                 curr_loss,
                 float(loss_kld_hist[-1]) if len(loss_kld_hist) else float("nan"),
                 float(penalty_hist[-1]) if len(penalty_hist) else float("nan"),
                 float(clamped_lsig_kld.detach().cpu().item()),
                 float(clamped_lsig_pen.detach().cpu().item()),
                 float(w_kld.detach().cpu().item()),
                 float(w_pen.detach().cpu().item()),
                 avg_val_raw, avg_val_ema, lr_now, float(grad_norm_hist[-1]), float(z_abs_max_hist[-1]), int(sentry_ok_hist[-1])] + [float(v) for v in val_bpds_raw]
            writer.writerow(row)

        # Early stopping on RAW avg val_bpd
        if args.early_stopping_patience>0:
            if avg_val_raw < best_avg - args.min_delta:
                best_avg = avg_val_raw; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.early_stopping_patience:
                    with open(trial_dir/"EARLY_STOPPING.txt","w") as f:
                        f.write(f"Stopped at iter {it}, best RAW avg val_bpd {best_avg}\n")
                    print("Early stopping triggered.")
                    write_state = True
                    break

        write_state = True

    # Periodic plotting & sampling
    if it % args.plot_interval == 0:

        # Loss plots
        plt.figure(figsize=(40,10))

        kld_bpd = (-np.array(loss_kld_hist, dtype=float)) / (np.log(2.0) * n_dims)
        tot_bpd = (-np.array(loss_hist, dtype=float)) / (np.log(2.0) * n_dims)

        # Panel 1 — KLD (bits/dim), smoothed
        ax1 = plt.subplot(1,4,1)
        ax1.plot(loss_iter, simple_moving_average(kld_bpd, w=200), label='−KLD (bits/dim, SMA)')
        ax1.set_xlabel('Iteration'); ax1.set_ylabel('bits/dim'); ax1.set_title('KLD (per-dim)')
        ax1.grid(True); ax1.legend(); ax1.ticklabel_format(style='plain', axis='y')
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

        # Panel 2 — Penalty term (smoothed)
        ax2 = plt.subplot(1,4,2)
        ax2.plot(loss_iter, simple_moving_average(penalty_hist, w=200), label='Penalty (SMA)')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('value'); ax2.set_title('Penalty term')
        ax2.grid(True); ax2.legend(); ax2.ticklabel_format(style='plain', axis='y')

        # Panel 3 — Total (≈NLL) as bits/dim, smoothed
        ax3 = plt.subplot(1,4,3)
        ax3.plot(loss_iter, simple_moving_average(tot_bpd, w=200), label='Total (≈NLL) BPD (SMA)')
        ax3.set_xlabel('Iteration'); ax3.set_ylabel('bits/dim'); ax3.set_title('Total (per-dim)')
        ax3.grid(True); ax3.legend(); ax3.ticklabel_format(style='plain', axis='y')
        ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

        # Panel 4 — Grad norm (log) with clip line
        ax4 = plt.subplot(1,4,4)
        g = np.asarray(grad_norm_hist, dtype=float)
        ax4.semilogy(loss_iter[-len(g):], g, label='||grad|| (pre-clip)')
        ax4.axhline(y=max(float(args.grad_clip), 1e-12), linestyle='--', linewidth=1.5, label=f'clip={args.grad_clip}')
        ax4.set_xlabel('Iteration'); ax4.set_ylabel('||grad||'); ax4.set_title('Grad Norm (log)')
        ax4.grid(True, which='both'); ax4.legend()

        plt.tight_layout()
        plt.savefig(str(trial_dir / "loss_hist_glow_2d_uncertES_plus_dir_safeguarded.pdf"))
        plt.close()

        # Samples per model (RAW)
        for m in range(len(models)):
            nf.utils.clear_grad(models[m])
            with torch.no_grad():
                x_s, _ = models[m].sample(100)
                x_s = torch.clamp(x_s, 0, 1)
                grid = tv.utils.make_grid(x_s.to(torch.float32), nrow=10)
                plt.figure(figsize=(10,10)); plt.imshow(np.transpose(grid.cpu().numpy(), (1,2,0))); plt.axis("off")
                plt.savefig(str(trial_dir / f"samples_glow_2d_model_{args.modalities[m]}_safeguarded.pdf")); plt.close()

        # Samples per model (EMA) if enabled
        if ema_models is not None:
            for m in range(len(ema_models)):
                with torch.no_grad():
                    x_s, _ = ema_models[m].sample(100)
                    x_s = torch.clamp(x_s, 0, 1)
                    grid = tv.utils.make_grid(x_s.to(torch.float32), nrow=10)
                    plt.figure(figsize=(10,10)); plt.imshow(np.transpose(grid.cpu().numpy(), (1,2,0))); plt.axis("off")
                    plt.savefig(str(trial_dir / f"samples_glow_2d_model_{args.modalities[m]}_EMA_safeguarded.pdf")); plt.close()

    # Save state periodically (after evals)
    if (it % args.eval_interval == 0) or write_state:
        save_blob = {
            "iter": it+1,
            "flow_optimizer": flow_optimizer.state_dict(),
            "mine_optimizer": (mine_optimizer.state_dict() if mine_optimizer else None),
            "scheduler": (warmup_sched.state_dict() if warmup_sched is not None else None),
            "best_val_bpd": best_val_bpd,
            "best_avg": best_avg,
            "no_improve": no_improve,
            "ema_state_dicts": ([ema_models[m].state_dict() for m in range(len(ema_models))] if ema_models is not None else None),
        }
        torch.save(save_blob, state_path)

# Finalize: write BEST checkpoints as *_final.pt
for m,mod in enumerate(args.modalities):
    final_path = trial_dir/f"model_{mod}_final.pt"
    if Path(best_ckpts[m]).exists():
        shutil.copy2(best_ckpts[m], final_path)
        print(f"Wrote final (best) checkpoint for {mod} -> {final_path}")
    else:
        models[m].save(str(final_path))
        print(f"No best found for {mod}; saved current model as final -> {final_path}")

print("Training complete. Run dir:",trial_dir)