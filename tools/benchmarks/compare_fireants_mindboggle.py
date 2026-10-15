#!/usr/bin/env python3
"""FireANTs (github.com/rohitrango/fireants, ~/Pkg/fireants) as a third arm of the syntx-vs-antstorch
Mindboggle-101 comparison (tools/benchmarks/compare_syntx_antstorch_mindboggle.py), in the same FAIR mode.

Revision 2026-10-14. What "fair" means here
-------------------------------------------
  - The sibling script's fair mode drives syntx and antstorch (syntx first; its cached affine is copied into
    antstorch's canonical-affine cache; antstorch runs syntx's loop: per-iteration inverses with 10 steps,
    syntx parity, autograd; dsti with syntx.syn()'s reg_adam). FireANTs is run AFTER syntx on each pair and
    reads the same cached affine (`_fit_or_load_canonical_affine` on the shared cache), so the three libraries
    start from the identical affine. The record says which affine FireANTs used (`_affine_source`).
  - Times are deformable-only for all three: for FireANTs this is `GreedyRegistration(...)` construction +
    `optimize()` bracketed by `torch.cuda.synchronize()`. The affine, NIfTI writing/loading, warp saving and
    the Dice/Jacobian scoring are outside the timer.
  - One throwaway warm-up run per library on the first pair (not recorded) unless `--no-warmup`.
  - FireANTs has one configuration (GreedyRegistration: Adam, compositive warp, CC kernel 5, Gaussian
    warp/gradient smoothing sigma 1.0), so it runs ONCE per pair and is compared to every syntx/antstorch model.
  - FireANTs is a different algorithm from syntx/antstorch: this is a library comparison, not a parity check.
  - The default iteration budget is [100,100,20] (same as syntx/antstorch gaussian/sobolev), pyramid [4,2,1];
    `--tutorial-schedule` uses FireANTs' own [200,100,25].
  - GPU only: FireANTs needs CUDA here. `--device` defaults to `cuda`; anything else exits with a message.
  - Known FireANTs limitation: its field inversion is imperfect, which lowers `dice_sym` (uses the inverse
    warp); `dice_fixed` (forward warp only) is the more reliable FireANTs number and is reported too.

Same Dice / Jacobian scoring as the sibling script (`compute_bidirectional_dice`, `compute_jacobian_metrics`).

Usage (next to compare_syntx_antstorch_mindboggle.py, e.g. in tools/benchmarks/):

    python tools/benchmarks/compare_fireants_mindboggle.py --pairs 0 1 2 --out-dir <dir>

Requires `~/Pkg/fireants` cloned on the machine that runs this (same convention as `~/Pkg/syntx`).
"""
import torch  # noqa: F401  (keep BEFORE any `import ants`: OpenMP runtime load order, see tests/conftest.py)
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

FIREANTS_ROOT = os.path.expanduser("~/Pkg/fireants")
ANTSTORCH_ROOT = os.path.expanduser("~/Pkg/ANTsTorch")
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING_SCRIPT = os.path.join(_HERE, "compare_syntx_antstorch_mindboggle.py")

# GreedyRegistration settings. optimizer_lr stays at FireANTs' own tutorial value (0.5): a 2026-09-22 run showed that
# halving it undertrains FireANTs (project doc). Only the iteration budget differs between the two schedules.
_FIREANTS_KWARGS = {
    "matched": dict(scales=[4, 2, 1], iterations=[100, 100, 20], cc_kernel_size=5, optimizer_lr=0.5),
    "tutorial": dict(scales=[4, 2, 1], iterations=[200, 100, 25], cc_kernel_size=5, optimizer_lr=0.5),
}
FIREANTS_SCHEDULE = "matched"
FIREANTS_OVERRIDES = {}  # optional diagnostic overrides of _FIREANTS_KWARGS entries (set from the CLI)
FIREANTS_TOLERANCE = None  # None: FireANTs' own early-stop setting (tolerance 1e-6 on the loss slope, 10 iterations)
FIREANTS_LABEL = "fireants"


def _load_sibling_module():
    if not os.path.exists(_SIBLING_SCRIPT):
        raise FileNotFoundError(
            f"Expected compare_syntx_antstorch_mindboggle.py next to this script at '{_SIBLING_SCRIPT}' "
            f"(its fair-mode runners are reused) -- keep both files in tools/benchmarks/."
        )
    spec = importlib.util.spec_from_file_location("compare_syntx_antstorch_mindboggle", _SIBLING_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sync(device):
    import torch as _t
    if str(device).startswith("cuda") and _t.cuda.is_available():
        _t.cuda.synchronize(device)


def _affine_source(out_dir, pair_idx):
    """'syntx' if the cached affine for this pair was copied from syntx's result, else 'antstorch-fit'/'missing'."""
    d = os.path.join(out_dir, "antstorch_canonical_affines")
    if not os.path.exists(os.path.join(d, f"pair_{pair_idx:03d}_0GenericAffine.mat")):
        return "missing"
    try:
        with open(os.path.join(d, f"pair_{pair_idx:03d}_affine_info.json")) as f:
            return json.load(f).get("source", "antstorch-fit")
    except (OSError, ValueError):
        return "antstorch-fit"


def _run_fireants(pair_idx, device, out_dir, data_dir):
    """One FireANTs registration for one pair. `_deformable_seconds` times only the registration itself."""
    t0 = time.time()
    sys.path.insert(0, FIREANTS_ROOT)
    sys.path.insert(0, ANTSTORCH_ROOT)
    try:
        import torch
        import ants
        from fireants.io import Image, BatchedImages
        from fireants.registration.greedy import GreedyRegistration
        from antstorch.benchmark.data import load_mindboggle_pair
        from antstorch.benchmark.evaluate import _fit_or_load_canonical_affine, clean_device_cache
        from antstorch.benchmark.metrics import compute_bidirectional_dice, compute_jacobian_metrics

        clean_device_cache()
        pair_data = load_mindboggle_pair(pair_idx=pair_idx, data_dir=data_dir, use_n4=True)
        fi_ants, mi_ants = pair_data["fixed"], pair_data["moving"]
        fl_ants, ml_ants = pair_data["fixed_label"], pair_data["moving_label"]
        dim = fi_ants.dimension

        # The shared affine: syntx's, copied into this cache by the sibling's fair mode (syntx runs first).
        canonical_affine_dir = os.path.join(out_dir, "antstorch_canonical_affines")
        matrix, translation, _t_aff, _affine_path = _fit_or_load_canonical_affine(
            fi_ants, mi_ants, pair_idx, canonical_affine_dir, device, False
        )
        affine_source = _affine_source(out_dir, pair_idx)

        pair_out_dir = os.path.join(out_dir, "fireants_registration", f"pair_{pair_idx:03d}")
        os.makedirs(pair_out_dir, exist_ok=True)
        fixed_path = os.path.join(pair_out_dir, "fixed.nii.gz")
        moving_path = os.path.join(pair_out_dir, "moving.nii.gz")
        ants.image_write(fi_ants, fixed_path)
        ants.image_write(mi_ants, moving_path)
        fixed_batch = BatchedImages([Image.load_file(fixed_path, device=device)])
        moving_batch = BatchedImages([Image.load_file(moving_path, device=device)])

        affine_4x4 = torch.eye(dim + 1, dtype=torch.float32)
        affine_4x4[:dim, :dim] = matrix
        affine_4x4[:dim, -1] = translation
        init_affine = affine_4x4.unsqueeze(0).to(device)

        kw = dict(_FIREANTS_KWARGS[FIREANTS_SCHEDULE])
        kw.update(FIREANTS_OVERRIDES)
        smooth_warp_sigma = float(kw.pop("smooth_warp_sigma", 1.0))
        extra_kw = {} if FIREANTS_TOLERANCE is None else dict(tolerance=float(FIREANTS_TOLERANCE))
        _sync(device)
        t_start = time.time()
        reg_obj = GreedyRegistration(
            scales=kw["scales"], iterations=kw["iterations"],
            fixed_images=fixed_batch, moving_images=moving_batch,
            loss_type="cc", cc_kernel_size=kw["cc_kernel_size"],
            deformation_type="compositive",
            optimizer="adam", optimizer_lr=kw["optimizer_lr"],
            smooth_warp_sigma=smooth_warp_sigma, smooth_grad_sigma=1.0,
            init_affine=init_affine, **extra_kw,
        )
        # Count the iterations FireANTs really runs per pyramid level: its ConvergenceMonitor can stop a level early
        # (it stops when the slope of the last 10 losses exceeds `tolerance`), so the nominal budget may not be spent.
        iters_run, losses = [], []
        mon = reg_obj.convergence_monitor
        _orig_reset, _orig_converged = mon.reset, mon.converged

        def _reset():
            iters_run.append(0)
            losses.append([])
            _orig_reset()

        def _converged(loss=None):
            iters_run[-1] += 1
            if loss is not None:
                losses[-1].append(float(loss))
            return _orig_converged(loss)
        mon.reset, mon.converged = _reset, _converged
        reg_obj.optimize()
        _sync(device)
        deformable_seconds = time.time() - t_start

        fwd_warp_path = os.path.join(pair_out_dir, "1Warp.nii.gz")
        inv_warp_path = os.path.join(pair_out_dir, "1InverseWarp.nii.gz")
        reg_obj.save_as_ants_transforms(fwd_warp_path, save_inverse=False)
        reg_obj.save_as_ants_transforms(inv_warp_path, save_inverse=True)

        clean_device_cache()
        df_fixed, df_moving, dice_sym = compute_bidirectional_dice(
            fl_ants, ml_ants, fi_ants, mi_ants, [fwd_warp_path], [inv_warp_path], [False]
        )
        jac = compute_jacobian_metrics(fi_ants, fwd_warp_path)
        rec = {
            "pair_idx": int(pair_idx), "model_type": "greedy (fireants)",
            "cohort_type": pair_data["pair_type"], "fixed_id": pair_data["fixed_id"], "moving_id": pair_data["moving_id"],
            "dice_sym": float(dice_sym), "dice_fixed": float(df_fixed), "dice_moving": float(df_moving),
            "folding_pct": float(jac.get("folding_pct", float("nan"))),
            "min_jacobian": float(jac.get("min", float("nan"))),
            "runtime_seconds": deformable_seconds, "_deformable_seconds": deformable_seconds,
            "_wall_seconds": time.time() - t0, "_affine_source": affine_source,
            "_fireants_schedule": FIREANTS_SCHEDULE,
            "_fireants_iterations_run": list(iters_run), "_fireants_iterations_nominal": list(kw["iterations"]),
            "_fireants_tolerance": FIREANTS_TOLERANCE,
            # loss (negative local CC, lower is better) at the first / last iteration of each pyramid level
            "_fireants_loss_first_last": [[l[0], l[-1]] if l else None for l in losses],
            "_fireants_overrides": dict(FIREANTS_OVERRIDES),
            "_library": "fireants", "_regularizer": FIREANTS_LABEL, "_status": "SUCCESS",
        }
        if affine_source != "syntx":
            print(f"    !! FireANTs pair {pair_idx}: affine source is '{affine_source}', not syntx's -- "
                  f"the start is not shared with syntx.", flush=True)
    except Exception as e:  # noqa: BLE001
        rec = {
            "_library": "fireants", "_regularizer": FIREANTS_LABEL, "pair_idx": pair_idx,
            "_status": "FAILED", "_error": str(e), "_traceback": traceback.format_exc(),
            "_wall_seconds": time.time() - t0,
        }
    return rec


SWEEP_GRID = (
    [dict(smooth_warp_sigma=sg, cc_kernel_size=k) for sg in (0.25, 0.5, 1.0, 2.0) for k in (5, 7, 9)]
    + [dict(smooth_warp_sigma=0.5, cc_kernel_size=7, optimizer_lr=lr) for lr in (0.1, 0.25, 1.0)]
)


def _run_sweep(pairs, device, out_dir, data_dir):
    """FireANTs-only hyper-parameter sweep (no syntx / ANTsTorch): full nominal budget (no early stop), the shared
    syntx affines already cached in `out_dir/antstorch_canonical_affines` by a previous run of this script."""
    global FIREANTS_OVERRIDES, FIREANTS_TOLERANCE
    import numpy as np
    missing = [p for p in pairs if _affine_source(out_dir, p) != "syntx"]
    if missing:
        sys.exit(f"--fireants-sweep needs syntx's cached affines in '{out_dir}/antstorch_canonical_affines' "
                 f"(pairs without one: {missing}). Point --out-dir at a directory from an earlier run of this script "
                 f"that covered these pairs.")
    FIREANTS_TOLERANCE = 1e9
    rows, raw = [], []
    for cfg in SWEEP_GRID:
        FIREANTS_OVERRIDES = dict(cfg)
        recs = [_run_fireants(p, device, out_dir, data_dir) for p in pairs]
        raw.append({"config": cfg, "records": recs})
        ok = [r for r in recs if r.get("_status") == "SUCCESS"]
        print(f"  {cfg}: " + (f"dice_sym={np.mean([r['dice_sym'] for r in ok]):.4f} "
                              f"dice_fixed={np.mean([r['dice_fixed'] for r in ok]):.4f} (n={len(ok)}/{len(recs)})"
                              if ok else "all failed"), flush=True)
        if ok:
            last = [r["_fireants_loss_first_last"][-1][1] for r in ok if r.get("_fireants_loss_first_last")
                    and r["_fireants_loss_first_last"][-1]]
            rows.append((cfg, len(ok), len(recs), np.mean([r["dice_sym"] for r in ok]), np.mean([r["dice_fixed"] for r in ok]),
                         np.mean([r["_deformable_seconds"] for r in ok]), float(np.mean(last)) if last else float("nan"),
                         float(np.median([r["folding_pct"] for r in ok]))))
        with open(os.path.join(out_dir, "fireants_sweep.json"), "w") as f:
            json.dump(raw, f, indent=2)
    md = ["# FireANTs hyper-parameter sweep (full budget, no early stop; shared syntx affine)", "",
          f"Pairs: {pairs}; schedule: {_FIREANTS_KWARGS[FIREANTS_SCHEDULE]['iterations']}; other settings: scripts defaults.", "",
          "| smooth_warp_sigma | cc_kernel_size | optimizer_lr | n | Dice sym | Dice fixed | t déf. (s) | final-level loss | folding % (median) |",
          "|---|---|---|---|---|---|---|---|---|"]
    base = _FIREANTS_KWARGS[FIREANTS_SCHEDULE]
    for cfg, n, tot, ds, df, t, ll, fo in sorted(rows, key=lambda r: -r[4]):
        md.append(f"| {cfg.get('smooth_warp_sigma', 1.0)} | {cfg.get('cc_kernel_size', base['cc_kernel_size'])} | "
                  f"{cfg.get('optimizer_lr', base['optimizer_lr'])} | {n}/{tot} | {ds:.4f} | {df:.4f} | {t:.1f} | {ll:.4f} | {fo:.4f} |")
    md.append("")
    md.append("Sorted by Dice fixed (forward warp only; FireANTs' inverse is imperfect). For reference, the syntx / ANTsTorch "
              "Dice and the affine-only Dice of these pairs are in the comparison_summary.md of the run that produced the affines.")
    path = os.path.join(out_dir, "fireants_sweep.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + f"\n\nMarkdown: {path}")


def _aggregate3(results, models, pairs):
    """Paired statistics on pairs where syntx, antstorch AND fireants all succeeded (deformable-only times)."""
    import numpy as np
    lines = ["| Modèle | n | Dice affine (syntx) | Dice syntx | Dice antstorch | Dice fireants (sym / fixe) | Δ antstorch−syntx (moy ± ES) | "
             "Δ fireants−syntx (moy ± ES) | t déf. syntx | t déf. antstorch | t déf. fireants (s) | "
             "repli. % médiane s / a / f |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in results:
        if r.get("_status") == "SUCCESS":
            by.setdefault((r["_regularizer"], r["_library"]), {})[r["pair_idx"]] = r
    F = by.get((FIREANTS_LABEL, "fireants"), {})
    for reg in models:
        A, S = by.get((reg, "antstorch"), {}), by.get((reg, "syntx"), {})
        ps = sorted(set(A) & set(S) & set(F))
        if not ps:
            continue
        g = lambda D, k: np.array([D[p][k] for p in ps], dtype=float)  # noqa: E731
        da, ds, df, dff = g(A, "dice_sym"), g(S, "dice_sym"), g(F, "dice_sym"), g(F, "dice_fixed")
        ta, ts, tf = g(A, "_deformable_seconds"), g(S, "_deformable_seconds"), g(F, "_deformable_seconds")

        def se(x):
            return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        xa, xf = da - ds, df - ds
        fm = [np.median(g(D, "folding_pct")) for D in (S, A, F)]
        aff = np.array([S[p].get("syntx_affine_dice_sym", float("nan")) for p in ps], dtype=float)
        lines.append(
            f"| {reg} | {len(ps)} | {np.nanmean(aff):.4f} | {ds.mean():.4f} | {da.mean():.4f} | {df.mean():.4f} / {dff.mean():.4f} | "
            f"{xa.mean():+.4f} ± {se(xa):.4f} | {xf.mean():+.4f} ± {se(xf):.4f} | {ts.mean():.1f} | {ta.mean():.1f} | "
            f"{tf.mean():.1f} | {fm[0]:.4f} / {fm[1]:.4f} / {fm[2]:.4f} |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, nargs="+", default=[0, 24, 88])
    ap.add_argument("--models", nargs="+", default=["gaussian", "sobolev", "dsti"],
                    choices=["gaussian", "sobolev", "dsti"],
                    help="syntx/antstorch models (FireANTs runs once per pair and is compared to each).")
    ap.add_argument("--device", default="cuda", help="CUDA device for all three libraries (default: cuda).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tutorial-schedule", action="store_true",
                    help="FireANTs iterations [200,100,25] (its tutorial) instead of [100,100,20] (same budget as syntx).")
    ap.add_argument("--fireants-no-early-stop", action="store_true",
                    help="Disable FireANTs' convergence-based early stop (tolerance=1e9) so every nominal iteration runs. "
                         "Use to test whether the early stop explains FireANTs' short runtime / low Dice.")
    ap.add_argument("--fireants-iterations", type=int, nargs="+", default=None,
                    help="Diagnostic: override FireANTs' per-level iterations (one value per pyramid level, 3). "
                         "'1 1 1' (almost no optimisation) tests the affine export/scoring chain: the resulting Dice should "
                         "be close to the affine-only Dice reported in the summary.")
    ap.add_argument("--fireants-lr", type=float, default=None, help="Diagnostic: FireANTs Adam learning rate (default 0.5).")
    ap.add_argument("--fireants-warp-sigma", type=float, default=None,
                    help="Diagnostic: FireANTs smooth_warp_sigma (this script's default 1.0; FireANTs' own default 0.5).")
    ap.add_argument("--fireants-cc-kernel", type=int, default=None, help="Diagnostic: FireANTs cc_kernel_size (default 5).")
    ap.add_argument("--fireants-sweep", action="store_true",
                    help="FireANTs-only sweep of smooth_warp_sigma x cc_kernel_size (and the learning rate), full budget, "
                         "on --pairs. Reuses the syntx affines cached in --out-dir by an earlier run (syntx / ANTsTorch are "
                         "NOT run). Writes fireants_sweep.md / fireants_sweep.json.")
    ap.add_argument("--own-defaults", action="store_true",
                    help="Disable the fair mode for syntx/antstorch (see the sibling script). The affine is then NOT shared.")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--matched", action="store_true", help=argparse.SUPPRESS)  # deprecated no-op
    args = ap.parse_args()

    import torch
    if not (str(args.device).startswith("cuda") and torch.cuda.is_available()):
        sys.exit(f"FireANTs benchmark needs a CUDA GPU (got --device {args.device!r}, "
                 f"cuda available: {torch.cuda.is_available()}). Run it on the GPU machine (e.g. hyperwolf).")

    global FIREANTS_SCHEDULE, FIREANTS_TOLERANCE, FIREANTS_OVERRIDES
    FIREANTS_SCHEDULE = "tutorial" if args.tutorial_schedule else "matched"
    FIREANTS_TOLERANCE = 1e9 if args.fireants_no_early_stop else None
    if args.fireants_iterations is not None:
        FIREANTS_OVERRIDES["iterations"] = list(args.fireants_iterations)
    if args.fireants_lr is not None:
        FIREANTS_OVERRIDES["optimizer_lr"] = args.fireants_lr
    if args.fireants_warp_sigma is not None:
        FIREANTS_OVERRIDES["smooth_warp_sigma"] = args.fireants_warp_sigma
    if args.fireants_cc_kernel is not None:
        FIREANTS_OVERRIDES["cc_kernel_size"] = args.fireants_cc_kernel
    if args.out_dir is None:
        args.out_dir = os.path.expanduser("~/Desktop/fireants_syntx_antstorch_mindboggle_comparison")
    os.makedirs(args.out_dir, exist_ok=True)

    sib = _load_sibling_module()
    if args.fireants_sweep:
        _run_sweep(args.pairs, args.device, args.out_dir, sib.DATA_DIR)
        return
    sib.FAIR = not args.own_defaults
    libs = (("syntx", sib._run_syntx), ("antstorch", sib._run_antstorch)) if sib.FAIR \
        else (("antstorch", sib._run_antstorch), ("syntx", sib._run_syntx))
    if args.own_defaults:
        print("!! --own-defaults: syntx and antstorch fit separate affines; FireANTs will use antstorch's.", flush=True)

    if not args.no_warmup:
        warm_dir = os.path.join(args.out_dir, "_warmup")
        os.makedirs(warm_dir, exist_ok=True)
        print(f"[warm-up] pair {args.pairs[0]} (not recorded) ...", flush=True)
        sib._run_pair_model(args.models[0], args.pairs[0], args.device, warm_dir, libs)
        _run_fireants(args.pairs[0], args.device, warm_dir, sib.DATA_DIR)
        sib._SHARED_AFFINE_DONE.clear()

    results = []
    json_path = os.path.join(args.out_dir, "comparison_results.json")

    def keep(rec, label):
        results.append(rec)
        ok = rec.get("_status") == "SUCCESS"
        extra = f" iterations_run={rec['_fireants_iterations_run']}/{rec['_fireants_iterations_nominal']}" if ok and "_fireants_iterations_run" in rec else ""
        if ok and "_fireants_loss_first_last" in rec:
            extra += " loss_first->last=" + str([[round(a, 4), round(b, 4)] if a is not None else None for a, b in
                                                 [(x[0], x[1]) if x else (None, None) for x in rec["_fireants_loss_first_last"]]])
        print(f"    {label}: " + (f"dice_sym={rec['dice_sym']:.4f} deformable={rec['_deformable_seconds']:.1f}s{extra}" if ok
                                  else f"{rec.get('_status')}: {rec.get('_error')}"), flush=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    for pair_idx in args.pairs:
        for reg in args.models:
            print(f"pair {pair_idx} / {reg} ...", flush=True)
            for (lib, _fn), rec in zip(libs, sib._run_pair_model(reg, pair_idx, args.device, args.out_dir, libs)):
                keep(rec, lib)
        print(f"pair {pair_idx} / fireants ...", flush=True)  # after syntx: the shared affine is in the cache
        keep(_run_fireants(pair_idx, args.device, args.out_dir, sib.DATA_DIR), "fireants")

    by_key = {}
    for r in results:
        by_key.setdefault((r.get("pair_idx"), r["_regularizer"]), {})[r["_library"]] = r
    table = ["| Paire | Modèle | syntx | antstorch | fireants (sym / fixe) | Δ a−s | Δ f−s | "
             "t déf. s | t déf. a | t déf. f (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pair_idx in args.pairs:
        f = by_key.get((pair_idx, FIREANTS_LABEL), {}).get("fireants", {})
        f_ok = f.get("_status") == "SUCCESS"
        for reg in args.models:
            e = by_key.get((pair_idx, reg), {})
            s, a = e.get("syntx", {}), e.get("antstorch", {})
            s_ok, a_ok = s.get("_status") == "SUCCESS", a.get("_status") == "SUCCESS"
            table.append(
                f"| {pair_idx} | {reg} | {s['dice_sym']:.4f} | " if s_ok else f"| {pair_idx} | {reg} | ÉCHEC | ")
            table[-1] += (f"{a['dice_sym']:.4f} | " if a_ok else "ÉCHEC | ")
            table[-1] += (f"{f['dice_sym']:.4f} / {f['dice_fixed']:.4f} | " if f_ok else "ÉCHEC | ")
            table[-1] += (f"{a['dice_sym'] - s['dice_sym']:+.4f} | " if (s_ok and a_ok) else "— | ")
            table[-1] += (f"{f['dice_sym'] - s['dice_sym']:+.4f} | " if (s_ok and f_ok) else "— | ")
            table[-1] += (f"{s['_deformable_seconds']:.1f} | " if s_ok else "— | ")
            table[-1] += (f"{a['_deformable_seconds']:.1f} | " if a_ok else "— | ")
            table[-1] += (f"{f['_deformable_seconds']:.1f} |" if f_ok else "— |")

    md = ["# Comparaison syntx vs antstorch vs FireANTs — Mindboggle-101", "",
          f"Paires : {args.pairs}  ", f"Modèles syntx/antstorch : {args.models}  ", f"Device : {args.device}  ",
          f"Mode syntx/antstorch : {'JUSTE (affine syntx partagée, antstorch iso-algorithme)' if sib.FAIR else 'défauts propres'}  ",
          f"FireANTs : GreedyRegistration, early stop {'désactivé' if FIREANTS_TOLERANCE else 'par défaut de FireANTs'}, itérations {FIREANTS_OVERRIDES.get('iterations', _FIREANTS_KWARGS[FIREANTS_SCHEDULE]['iterations'])}, surcharges {FIREANTS_OVERRIDES or 'aucune'}, "
          f"affine de la source '{_affine_source(args.out_dir, args.pairs[0])}' (paire {args.pairs[0]}) ; temps = construction + optimize, "
          f"synchronisé, hors affine/E-S/métriques  ", "",
          "## Résumé apparié (paires où les trois bibliothèques ont réussi)", ""]
    md += _aggregate3(results, args.models, args.pairs)
    md += ["", "## Détail par paire", ""] + table + ["", f"Résultats bruts : `{json_path}`"]
    md_path = os.path.join(args.out_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + f"\n\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
