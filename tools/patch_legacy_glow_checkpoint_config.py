#!/usr/bin/env python3
"""
Tag a pre-fix 3D Glow checkpoint with the invertible-conv / ActNorm log-scale
caps it was actually trained under, so GlowTool3D.build_model reconstructs it
correctly.

Background
----------
Before antsnormflows commit f047e4e ("BUG: s_cap in 1x1x1", 2026-07-27),
GlowBlock3d instantiated its per-block layers like this:

    self.flows += [ActNorm((channels,) + (1, 1, 1))]                # no log_s_cap
    self.flows += [Invertible1x1x1Conv(channels, use_lu)]            # no s_cap

Neither call passed a cap, so both layers silently used their own hardcoded
defaults -- log_s_cap=5.0 for ActNorm, s_cap=2.5 for Invertible1x1x1Conv --
regardless of whatever `scale_cap` was configured for the run (e.g. 1.5).

After the fix, GlowBlock3d always wires the cap explicitly:

    ActNorm(..., log_s_cap=(actnorm_s_cap if actnorm_s_cap is not None else s_cap))
    Invertible1x1x1Conv(..., s_cap=(conv_s_cap if conv_s_cap is not None else s_cap))

So loading a pre-fix checkpoint with today's code silently recalibrates those
two layers to `scale_cap` (e.g. 1.5) instead of the caps they were actually
trained under (5.0 / 2.5). Because the raw learned parameters (log_S in the
conv, s in ActNorm) get passed through a *different* tanh bound than the one
they were optimized against, the effective per-block scale changes -- and
compounded across dozens of blocks, this is enough to blow up the decode
(generative) direction while leaving encode comparatively stable.

`GlowTool3D.build_model` (antstorch/lamnr_flows/tools/lamnr_glow_tool_3d.py)
now reads two optional keys from the checkpoint's own saved config,
`legacy_conv_cap` and `actnorm_scale_cap`, and passes them straight through to
`create_glow_normalizing_flow_model_3d`. They default to None (i.e. "trust the
model's current scale_cap", correct for checkpoints trained after the fix).
This script patches an existing checkpoint's saved config in place so those
two keys travel with it from now on.

Usage
-----
    python tools/patch_legacy_glow_checkpoint_config.py \\
        /path/to/training_state.pt [more_checkpoints.pt ...]

    # dry run first (recommended):
    python tools/patch_legacy_glow_checkpoint_config.py --dry-run /path/to/*.pt

By default this only touches checkpoints trained *before* the fix landed
(2026-07-27) -- gated on the "env.timestamp" field antstorch's training
scripts write into run_config.json / the checkpoint's own metadata, when
present. If the timestamp can't be determined, the checkpoint is patched
anyway (pass --skip-unknown-date to instead skip it), since essentially every
checkpoint that exists today predates the fix.

A ``.bak`` copy of each checkpoint is written before overwriting, unless
``--no-backup`` is given.
"""

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import torch

FIX_DATE = dt.datetime(2026, 7, 27, 15, 13, 47)  # antsnormflows commit f047e4e


def _training_timestamp(blob: dict):
    """Best-effort extraction of a training start timestamp from the blob."""
    for key in ("env", "meta", "metadata"):
        sub = blob.get(key)
        if isinstance(sub, dict) and "timestamp" in sub:
            try:
                return dt.datetime.strptime(sub["timestamp"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass
    cfg = blob.get("config", blob.get("cfg", {}))
    if isinstance(cfg, dict):
        ts = cfg.get("timestamp")
        if ts:
            try:
                return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass
    return None


def patch_one(path: Path, legacy_conv_cap: float, actnorm_scale_cap: float,
               dry_run: bool, backup: bool, force: bool, skip_unknown_date: bool) -> bool:
    print(f"[patch] {path}")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)

    cfg_key = "config" if "config" in blob else ("cfg" if "cfg" in blob else None)
    if cfg_key is None:
        print("  -> no 'config'/'cfg' key in checkpoint blob, skipping.")
        return False
    cfg = blob[cfg_key]

    ts = _training_timestamp(blob)
    if ts is not None:
        print(f"  training timestamp: {ts}  (fix landed {FIX_DATE})")
        if ts >= FIX_DATE:
            print("  -> trained after the antsnormflows fix; leaving untouched.")
            return False
    else:
        print("  training timestamp: unknown")
        if skip_unknown_date:
            print("  -> --skip-unknown-date set; skipping.")
            return False

    already = cfg.get("legacy_conv_cap") is not None or cfg.get("actnorm_scale_cap") is not None
    if already and not force:
        print(f"  -> already has legacy_conv_cap={cfg.get('legacy_conv_cap')!r}, "
              f"actnorm_scale_cap={cfg.get('actnorm_scale_cap')!r}; use --force to overwrite.")
        return False

    print(f"  setting legacy_conv_cap={legacy_conv_cap}, actnorm_scale_cap={actnorm_scale_cap}")
    if dry_run:
        print("  -> dry run, not writing.")
        return True

    cfg["legacy_conv_cap"] = legacy_conv_cap
    cfg["actnorm_scale_cap"] = actnorm_scale_cap
    blob[cfg_key] = cfg

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  backup written to {bak}")
        else:
            print(f"  backup already exists at {bak}, not overwriting.")

    torch.save(blob, str(path))
    print("  -> patched and saved.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=str, help="training_state*.pt file(s) to patch")
    ap.add_argument("--legacy-conv-cap", type=float, default=2.5,
                     help="Value to store for Invertible1x1x1Conv's cap (default: 2.5, the "
                          "pre-fix hardcoded default).")
    ap.add_argument("--actnorm-scale-cap", type=float, default=5.0,
                     help="Value to store for ActNorm's log-scale cap (default: 5.0, the "
                          "pre-fix hardcoded default).")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Skip writing a .bak copy.")
    ap.add_argument("--force", action="store_true",
                     help="Overwrite legacy_conv_cap/actnorm_scale_cap even if already set.")
    ap.add_argument("--skip-unknown-date", action="store_true",
                     help="Skip checkpoints whose training timestamp can't be determined "
                          "(default: patch them anyway).")
    args = ap.parse_args()

    n_patched = 0
    for c in args.checkpoints:
        path = Path(c)
        if not path.exists():
            print(f"[patch] {path} does not exist, skipping.", file=sys.stderr)
            continue
        if patch_one(path, args.legacy_conv_cap, args.actnorm_scale_cap,
                      args.dry_run, args.backup, args.force, args.skip_unknown_date):
            n_patched += 1

    print(f"\n{n_patched}/{len(args.checkpoints)} checkpoint(s) "
          f"{'would be ' if args.dry_run else ''}patched.")


if __name__ == "__main__":
    main()
