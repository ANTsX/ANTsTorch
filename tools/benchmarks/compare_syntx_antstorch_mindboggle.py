#!/usr/bin/env python3
"""Compare syntx vs antstorch on real Mindboggle-101 pairs, dense-SyN family.

Runs the same 4 SyN regularizer variants (gaussian, sobolev, dsti, bspline)
through both libraries' own Mindboggle benchmark harnesses on the same
pairs, using each library's own affine initialization. antstorch's
syn_registration() defaults are aligned with syntx's (cc2, grad_step /
flow_sigma / sobolev_alpha / iteration schedule per regularizer, flow_sigma
read as an ITK variance, voxel-unit gaussian sigma, conservative post-filter,
end-of-level inverse schedule -- see _SYNTX_DEFAULTS in antstorch/syn/syn.py),
so a bare ``python compare_syntx_antstorch_mindboggle.py`` is already the
matched comparison; the options below are only overrides/ablations.

Model-name mapping (see the project doc, this session's investigation):
  - gaussian, sobolev: 1:1 named models in both
    syntx.benchmark.evaluate.evaluate_mindboggle_pair(model=...) and
    antstorch.benchmark.evaluate.evaluate_mindboggle_pair(model="<x>_syn").
  - dsti: syntx names it "syn_dsti1" (dsti1 regularizer via the reg_adam
    optimizer branch); antstorch names it "dsti_syn".
  - <x>_regadam (x in gaussian/sobolev/dsti/bspline): antstorch-only
    variants. dsti_regadam was added first, 2026-09-24 (project doc Section
    40); the other three followed once it turned out optimizer='reg_adam'
    is a fully generic switch in antstorch.syn.syn_registration(),
    independent of the regularizer (project doc Section 41). Each is the
    SAME regularizer as its <x>_syn counterpart but run through a
    lightweight port of syntx's RegAdam optimizer pattern (Adam-momentum
    quotient applied to the gradient before the existing regularizer call,
    reset each pyramid level -- see antstorch/syn/syn.py, optimizer=
    "reg_adam") instead of antstorch's default plain gradient descent. Each
    is compared against the SAME syntx arm as its <x>_syn counterpart --
    there is no separate syntx-side "_regadam" model of its own for any of
    the four -- the point is to see whether matching syntx's optimizer, on
    antstorch's own greedy-SyN loop, closes any of the gap between antstorch
    and syntx for that regularizer (first explored for dsti in Sections
    34/36/39; the same question can now be asked of the other three).
  - bspline: antstorch exposes it directly as "bspline_syn" (one of the
    four _SYN_REGULARIZERS). syntx.benchmark.evaluate does NOT expose a
    "bspline" model name at all (its error message lists supported models
    and 'bspline' is not among them), even though the lower-level
    syntx.syn(..., regularizer='bspline', ...) call it dispatches to for
    the other three variants works fine with regularizer='bspline' too
    (syntx.syn._apply_regularizer has a 'bspline'/'bsplinesyn' branch that
    needs no extra caller-supplied kwargs). So _syntx_bspline_pair_eval()
    below is NOT part of syntx's own benchmark harness -- it is our own
    thin wrapper, built by mirroring the "sobolev" branch of
    syntx.benchmark.evaluate.evaluate_mindboggle_pair line for line (same
    canonical-affine caching, same iteration schedule, same metric
    computation) with regularizer="bspline" substituted in. Flagged in the
    output as "syntx_bspline (non-officiel)" so it isn't mistaken for an
    upstream syntx benchmark arm.

Output: writes a JSON file with one record per (pair, model, library), plus
a Markdown summary table with syntx vs antstorch deltas, printed to stdout
and saved alongside the JSON.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

PAIR_INDICES_DEFAULT = [0, 24, 88]
_BASE_MODELS = ["gaussian", "sobolev", "dsti", "bspline"]
MODELS = _BASE_MODELS + [f"{m}_regadam" for m in _BASE_MODELS]

SYNTX_ROOT = os.path.expanduser("~/Pkg/syntx")
ANTSTORCH_ROOT = os.path.expanduser("~/Pkg/ANTsTorch")
DATA_DIR = os.path.expanduser("~/Data/Public/Mindboggle/Volumes")


def _base_reg(reg):
    """Strips a "_regadam" suffix, if present: gaussian_regadam -> gaussian,
    dsti -> dsti. Every "_regadam" model is compared against the same
    syntx-side arm as its non-"_regadam" counterpart (see module docstring)
    -- there is no separate syntx "_regadam" model for any of the four.
    """
    return reg[: -len("_regadam")] if reg.endswith("_regadam") else reg


# syntx model-name aliases for its evaluate_mindboggle_pair(model=...).
# "bspline" has no entry here -- see the "bspline" branch of _run_syntx()
# below, which routes it (and "bspline_regadam") to _syntx_bspline_pair_eval()
# instead, since syntx's own harness exposes no "bspline" model name at all.
_SYNTX_MODEL_NAME = {
    "gaussian": "gaussian",
    "sobolev": "sobolev",
    "dsti": "syn_dsti1",
}
# antstorch.benchmark.evaluate_mindboggle_pair(model=...) name for each
# --models entry: the four base regularizers use the "_syn" suffix; the
# four "_regadam" variants already match their antstorch.benchmark
# _SYN_REGULARIZERS/_SYN_OPTIMIZER_OVERRIDE registry key directly.
_ANTSTORCH_MODEL_NAME = {m: f"{m}_syn" for m in _BASE_MODELS}
_ANTSTORCH_MODEL_NAME.update({f"{m}_regadam": f"{m}_regadam" for m in _BASE_MODELS})


# antstorch side: nothing to set here any more. The matched hyperparameters
# (formerly MATCHED_KWARGS, validated against syntx on Mindboggle-101) are now
# the defaults of antstorch.syn.syn_registration(), keyed by (regularizer,
# optimizer) in _SYNTX_DEFAULTS; use --legacy-defaults for the previous ones.
# syntx side: its gaussian/sobolev harness arms are pinned to cc2 and
# (100,100,20) so both sides always run the same schedule.
SYNTX_KWARGS = {
    "gaussian": dict(similarity_metric="cc2", reg_iterations=[100, 100, 20]),
    "sobolev": dict(similarity_metric="cc2", reg_iterations=[100, 100, 20]),
}


ANTSTORCH_EXTRA = {}  # set from CLI in main(); antstorch arms only (syntx arms are unaffected)


def _run_antstorch(reg, pair_idx, device, out_dir, matched=True):  # `matched` kept for old callers; ignored
    sys.path.insert(0, ANTSTORCH_ROOT)
    from antstorch.benchmark.evaluate import evaluate_mindboggle_pair as antstorch_eval

    model = _ANTSTORCH_MODEL_NAME[reg]
    extra = dict(ANTSTORCH_EXTRA)
    t0 = time.time()
    try:
        rec = antstorch_eval(
            pair_idx=pair_idx,
            model=model,
            device=device,
            data_dir=DATA_DIR,
            canonical_affine_dir=os.path.join(out_dir, "antstorch_canonical_affines"),
            verbose=False,
            **extra,
        )
        rec["_wall_seconds"] = time.time() - t0
        rec["_library"] = "antstorch"
        rec["_regularizer"] = reg
        rec["_status"] = "SUCCESS"
    except Exception as e:
        rec = {
            "_library": "antstorch", "_regularizer": reg, "pair_idx": pair_idx,
            "_status": "FAILED", "_error": str(e), "_traceback": traceback.format_exc(),
            "_wall_seconds": time.time() - t0,
        }
    return rec


def _run_syntx(reg, pair_idx, device, out_dir, matched=True):  # `matched` kept for old callers; ignored
    sys.path.insert(0, os.path.join(SYNTX_ROOT, "src"))
    os.chdir(SYNTX_ROOT)  # syntx.benchmark.evaluate caches to "results/..." relative to cwd
    import syntx
    from syntx.benchmark.evaluate import evaluate_mindboggle_pair as syntx_eval

    # A "_regadam" reg has no syntx-side model of its own -- run the SAME
    # syntx-side evaluation as its base regularizer (see _base_reg() and the
    # module docstring). rec["_regularizer"] below is still set to the
    # original (unstripped) `reg`, so main()'s by_key grouping still pairs
    # this record against the matching antstorch "_regadam" record.
    base = _base_reg(reg)
    extra = SYNTX_KWARGS.get(base, {})
    t0 = time.time()
    try:
        if base == "bspline":
            # Already grad_step=0.25 / similarity_metric='cc2' /
            # reg_iterations=[100,100,20] by construction (see the function
            # docstring) -- nothing further to force here.
            rec = _syntx_bspline_pair_eval(pair_idx=pair_idx, device=device, verbose=False)
        else:
            rec = syntx_eval(
                pair_idx=pair_idx,
                model=_SYNTX_MODEL_NAME[base],
                device=device,
                data_dir=DATA_DIR,
                pairs_csv="examples/pairs.csv",
                verbose=False,
                **extra,
            )
        rec["_wall_seconds"] = time.time() - t0
        rec["_library"] = "syntx"
        rec["_regularizer"] = reg
        rec["_status"] = "SUCCESS"
    except Exception as e:
        rec = {
            "_library": "syntx", "_regularizer": reg, "pair_idx": pair_idx,
            "_status": "FAILED", "_error": str(e), "_traceback": traceback.format_exc(),
            "_wall_seconds": time.time() - t0,
        }
    return rec


def _syntx_bspline_pair_eval(pair_idx, device=None, verbose=False, seed=42):
    """Non-official 4th arm for the syntx side: mirrors the 'sobolev' branch
    of syntx.benchmark.evaluate.evaluate_mindboggle_pair (same affine cache,
    same iteration schedule/metric/inverse method) with regularizer='bspline'
    substituted in, since syntx's own benchmark harness does not expose a
    'bspline' model name. See the module docstring above.
    """
    import time as _time
    import json as _json
    import numpy as np
    import torch
    import ants
    import syntx
    from syntx.benchmark.data import load_mindboggle_pair
    from syntx.benchmark.evaluate import (
        clean_device_cache, normalize_intensity, AFFINE_BACKEND_KEY,
    )
    from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

    clean_device_cache()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed + pair_idx)
    np.random.seed(seed + pair_idx)

    pair_data = load_mindboggle_pair(pair_idx=pair_idx, pairs_csv="examples/pairs.csv", data_dir=DATA_DIR, use_n4=True)
    fi_raw, mi_raw = pair_data["fixed"], pair_data["moving"]
    fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
    fixed_id, moving_id, cohort_type = pair_data["fixed_id"], pair_data["moving_id"], pair_data["pair_type"]

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    canonical_affine_dir = "results/canonical_affines"
    os.makedirs(canonical_affine_dir, exist_ok=True)
    aff_suffix = f"_{AFFINE_BACKEND_KEY}"
    aff_mat_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}{aff_suffix}_affine.mat")
    aff_info_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}{aff_suffix}_affine_info.json")

    aff_0 = None
    if os.path.exists(aff_mat_path) and os.path.exists(aff_info_path):
        try:
            with open(aff_info_path, "r") as f:
                aff_info = _json.load(f)
            if aff_info.get("affine_backend") == AFFINE_BACKEND_KEY:
                aff_0 = aff_mat_path
                t_aff = float(aff_info.get("runtime_seconds", 0.0))
                aff_dice_sym = float(aff_info.get("dice_sym", 0.0))
        except Exception:
            aff_0 = None

    if aff_0 is None:
        t0_aff = _time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=verbose)
        t_aff = _time.time() - t0_aff
        import shutil
        shutil.copyfile(reg_aff["fwdtransforms"][0], aff_mat_path)
        aff_0 = aff_mat_path
        clean_device_cache()
        _, _, aff_dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, [aff_mat_path], [aff_mat_path], [True])
        with open(aff_info_path, "w") as f:
            _json.dump({"dice_sym": float(aff_dice_sym), "runtime_seconds": float(t_aff),
                        "pair_idx": pair_idx, "affine_backend": AFFINE_BACKEND_KEY}, f, indent=2)

    clean_device_cache()
    t0_reg = _time.time()
    res_reg = syntx.syn(
        fixed=fi, moving=mi, initial_transform=aff_0,
        backend="pytorch", device=device,
        grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
        reg_iterations=[100, 100, 20], similarity_metric="cc2",
        use_ants_pseudo_gradient=False, use_analytical_gradients=False,
        syn_sampling=2, fast_smooth=False, inverse_method="anderson",
        formulation="eulerian", regularizer="bspline",
        antisymmetric=True, verbose=verbose,
    )
    t_reg = _time.time() - t0_reg

    fwd_tx = res_reg["fwdtransforms"]
    inv_tx = res_reg["invtransforms"]
    which_inv = res_reg.get("whichtoinvert_inv", [True, False])
    df_fixed, df_moving, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)
    jac = compute_jacobian_metrics(fi, fwd_tx[0]) if fwd_tx else {"folding_pct": float("nan"), "min": float("nan")}

    return {
        "pair_idx": int(pair_idx), "model_type": "bspline (non-officiel)",
        "cohort_type": cohort_type, "fixed_id": fixed_id, "moving_id": moving_id,
        "syntx_affine_dice_sym": float(aff_dice_sym), "syntx_affine_time": float(t_aff),
        "dice_sym": float(dice_sym), "dice_fixed": float(df_fixed), "dice_moving": float(df_moving),
        "folding_pct": float(jac.get("folding_pct", float("nan"))),
        "min_jacobian": float(jac.get("min", float("nan"))),
        "runtime_seconds": float(t_reg),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, nargs="+", default=PAIR_INDICES_DEFAULT)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu. Default: auto-detect (cuda, then mps, then cpu), passed identically to both libraries.")
    ap.add_argument("--inverse-schedule", choices=["per_iteration", "end_of_level"], default=None,
                    help="When antstorch maintains the half-warp inverses (antstorch arms only). 'end_of_level' inverts once "
                         "after the last level instead of after every iteration (see syn_registration docstring); "
                         "validate Dice against a per_iteration run first.")
    ap.add_argument("--end-of-level-inverse-steps", type=int, default=None,
                    help="Solver steps for the single end-of-level inversion (default 30).")
    ap.add_argument("--syntx-parity", action="store_true",
                    help="antstorch only: mimic syntx's loop (CFL scaled by sqrt(curr/orig size), early stop on loss slope "
                         "<=1e-6 over 10 its, restore best-loss warps).")
    ap.add_argument("--no-analytical-gradients", action="store_true",
                    help="antstorch only: use autograd through the image interpolation (as syntx's benchmark does, "
                         "use_analytical_gradients=False) instead of antstorch's analytical image-gradient backward.")
    ap.add_argument("--skip-syntx", action="store_true",
                    help="Run only the antstorch arms (syntx records are left out; the summary shows '—' for syntx). "
                         "Use for antstorch-only ablations such as --inverse-method, where the syntx arms would be identical "
                         "to an earlier run.")
    ap.add_argument("--inverse-method", choices=["anderson", "fixed_point", "hybrid_lm"], default=None,
                    help="In-loop field-inversion solver for the ANTSTORCH arms only (default: antstorch's own, anderson). "
                         "fixed_point measured ~2.2x faster than anderson on MPS at 6 steps (bench_inverse_field.py); "
                         "syntx arms keep their own solver, so use this as an ablation, not the main comparison.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--legacy-defaults", action="store_true",
                    help="antstorch only: restore antstorch's previous defaults (lncc, grad_step 0.5, flow_sigma 3.0, "
                         "(100,100,50) iterations, physical sigma, per-iteration inverses) instead of the syntx-aligned ones "
                         "(syn_registration(syntx_defaults=False)). Writes to an '_legacy' out-dir suffix.")
    ap.add_argument("--matched", action="store_true", help=argparse.SUPPRESS)  # deprecated no-op: matched is the default
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")
    if args.out_dir is None:
        base = os.path.expanduser("~/Desktop/syntx_antstorch_mindboggle_comparison")
        args.out_dir = base + ("_legacy" if args.legacy_defaults else "")

    if args.legacy_defaults:
        ANTSTORCH_EXTRA["syntx_defaults"] = False
    if args.inverse_method:
        ANTSTORCH_EXTRA["inverse_method"] = args.inverse_method
    if args.inverse_schedule:
        ANTSTORCH_EXTRA["inverse_schedule"] = args.inverse_schedule
    if args.no_analytical_gradients:
        ANTSTORCH_EXTRA["use_analytical_gradients"] = False
    if args.syntx_parity:
        ANTSTORCH_EXTRA["syntx_parity"] = True
    if args.end_of_level_inverse_steps is not None:
        ANTSTORCH_EXTRA["end_of_level_inverse_steps"] = args.end_of_level_inverse_steps
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    json_path = os.path.join(args.out_dir, "comparison_results.json")

    libs = (("antstorch", _run_antstorch),) if args.skip_syntx else (("antstorch", _run_antstorch), ("syntx", _run_syntx))
    total = len(args.pairs) * len(args.models) * len(libs)
    done = 0
    for pair_idx in args.pairs:
        for reg in args.models:
            for lib, fn in libs:
                print(f"[{done + 1}/{total}] {lib} / {reg} / pair {pair_idx} ...", flush=True)
                rec = fn(reg, pair_idx, args.device, args.out_dir)
                results.append(rec)
                done += 1
                status = rec.get("_status")
                dice = rec.get("dice_sym", float("nan"))
                secs = rec.get("_wall_seconds", float("nan"))
                print(f"    -> {status} dice_sym={dice:.4f} wall={secs:.1f}s" if status == "SUCCESS"
                      else f"    -> {status}: {rec.get('_error')}", flush=True)
                # Write incrementally so a partial run is never lost.
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)

    # Build summary table
    by_key = {}
    for r in results:
        by_key.setdefault((r.get("pair_idx"), r["_regularizer"]), {})[r["_library"]] = r

    table_lines = ["| Paire | Modèle | syntx dice_sym | antstorch dice_sym | Δ (antstorch − syntx) | syntx t(s) | antstorch t(s) |",
                   "|---|---|---|---|---|---|---|"]
    for pair_idx in args.pairs:
        for reg in args.models:
            pair = by_key.get((pair_idx, reg), {})
            s, a = pair.get("syntx", {}), pair.get("antstorch", {})
            s_ok, a_ok = s.get("_status") == "SUCCESS", a.get("_status") == "SUCCESS"
            s_dice = f"{s['dice_sym']:.4f}" if s_ok else ("—" if args.skip_syntx else "ÉCHEC")
            a_dice = f"{a['dice_sym']:.4f}" if a_ok else "ÉCHEC"
            delta = f"{(a['dice_sym'] - s['dice_sym']):+.4f}" if (s_ok and a_ok) else "—"
            s_t = f"{s.get('runtime_seconds', s.get('_wall_seconds', float('nan'))):.1f}" if s_ok else "—"
            a_t = f"{a.get('runtime_seconds', a.get('_wall_seconds', float('nan'))):.1f}" if a_ok else "—"
            base = _base_reg(reg)
            if reg == "bspline":
                label = reg + " (non-officiel)"
            elif reg == "bspline_regadam":
                label = reg + " (non-officiel, vs syntx bspline)"
            elif reg.endswith("_regadam"):
                label = reg + f" (vs syntx {base})"
            else:
                label = reg
            table_lines.append(f"| {pair_idx} | {label} | {s_dice} | {a_dice} | {delta} | {s_t} | {a_t} |")

    md_lines = ["# Comparaison syntx vs antstorch — Mindboggle-101 (dense SyN)", "",
                f"Paires : {args.pairs}  ", f"Modèles : {args.models}  ", f"Device : {args.device or 'auto'}  ", f"antstorch inverse_method : {args.inverse_method or 'default (anderson)'}  ", f"antstorch defaults : {'legacy' if args.legacy_defaults else 'syntx-aligned'}  ",
                f"antstorch inverse_schedule : {args.inverse_schedule or ('per_iteration' if args.legacy_defaults else 'end_of_level')}"
                + (f" ({args.end_of_level_inverse_steps or 30} steps)" if (args.inverse_schedule or ('per_iteration' if args.legacy_defaults else 'end_of_level')) == "end_of_level" else "") + "  ",
                f"syntx arms : {'skipped' if args.skip_syntx else 'run'}", ""]
    md_lines.extend(table_lines)
    md_lines.append("")
    md_lines.append(f"Résultats bruts : `{json_path}`")

    md_path = os.path.join(args.out_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))
    print(f"\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
