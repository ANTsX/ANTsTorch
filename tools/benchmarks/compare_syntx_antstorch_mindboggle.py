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

FireANTs (optional, --fireants; merged 2026-10-15 from the former compare_fireants_mindboggle.py, which remains as a thin
entry point forwarding here): a third arm, GreedyRegistration, run ONCE per pair after syntx, from the same shared syntx affine, on
the same auto-normalised images (antstorch.syn.core.utils.normalize_image, numerically identical to syntx's), CUDA only. Times are
deformable-only (construction + optimize, torch.cuda.synchronize()). FireANTs is a different algorithm (Adam, compositive warp): this
is a library comparison, not a parity check; its inverse is imperfect, so Dice fixed is reported next to Dice sym, and folding is
reported because FireANTs' best Dice settings fold the field (its documentation in syntx reports 3.7 % at its tuned setting).
--fireants-sweep runs a FireANTs-only hyper-parameter sweep on affines cached by an earlier run. Diagnostics: --fireants-iterations,
--fireants-lr, --fireants-warp-sigma, --fireants-grad-sigma, --fireants-cc-kernel, --fireants-preset, --fireants-no-early-stop.

Output: writes a JSON file with one record per (pair, model, library), plus
a Markdown summary table with syntx vs antstorch deltas, printed to stdout
and saved alongside the JSON.
"""
import torch  # noqa: F401  (keep BEFORE any `import ants`: OpenMP runtime load order, see tests/conftest.py)
import argparse
import json
import math
import os
import shutil
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

# --- "fair" mode (default of the CLI; module default False so tools importing _run_* keep their behavior) ---
# Isolates the dense-SyN algorithm: (1) antstorch starts from the SAME affine file syntx used (copied from the
# syntx result, see _share_affine); (2) antstorch runs syntx's loop: in-loop inverses of both half-warps at every
# iteration with in_loop_inv_steps=10 (syntx.syn()/registration() default, passed to SyNTo in all three arms),
# syntx's CFL scale / slope early stop / best-state restore / floor pyramid (syntx_parity), and autograd through
# the image interpolation (use_analytical_gradients=False, as every syntx benchmark arm); (3) "dsti" is compared
# with antstorch's 'syn_reg_adam' optimizer, a port of syntx.syn()'s reg_adam (syntx's dsti arm = reg_adam + dsti1,
# sobolev_alpha 1.0, in_loop_inv_steps 10);
# (4) times compared are DEFORMABLE-ONLY on both sides (affine time reported separately).
FAIR = False
FAIR_ANTSTORCH_KWARGS = dict(
    inverse_schedule="per_iteration", in_loop_inverse_steps=10, syntx_parity=True, use_analytical_gradients=False,
)
FAIR_MODEL_OVERRIDES = {  # reg -> (antstorch evaluate model name, extra kwargs)
    "dsti": ("dsti_syn", dict(optimizer="syn_reg_adam")),  # syntx.syn()'s reg_adam; alpha 1.0 from the defaults table
}
DEFAULT_MODELS = ["gaussian", "sobolev", "dsti"]
_SHARED_AFFINE_DONE = set()


def _run_antstorch(reg, pair_idx, device, out_dir, matched=True):  # `matched` kept for old callers; ignored
    sys.path.insert(0, ANTSTORCH_ROOT)
    from antstorch.benchmark.evaluate import evaluate_mindboggle_pair as antstorch_eval

    model = _ANTSTORCH_MODEL_NAME[reg]
    extra = {}
    if FAIR:
        extra.update(FAIR_ANTSTORCH_KWARGS)
        if reg in FAIR_MODEL_OVERRIDES:
            model, model_extra = FAIR_MODEL_OVERRIDES[reg]
            extra.update(model_extra)
    extra.update(ANTSTORCH_EXTRA)
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
        rec["_antstorch_model"] = model
        rec["_deformable_seconds"] = rec.get("deformable_seconds", rec["runtime_seconds"] - rec.get("affine_seconds", 0.0))
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
        # syntx's runtime_seconds = deformable + the (possibly cached) affine time; our own bspline wrapper
        # (_syntx_bspline_pair_eval) reports the deformable time alone.
        rec["_deformable_seconds"] = (rec["runtime_seconds"] if base == "bspline"
                                      else rec["runtime_seconds"] - rec.get("syntx_affine_time", 0.0))
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


def _share_affine(syntx_rec, pair_idx, out_dir):
    """Copy the affine syntx used for this pair into antstorch's canonical-affine cache (once per pair), so the
    antstorch arms start from exactly the same transform; the affine's fit time is stored beside it."""
    key = (os.path.abspath(out_dir), pair_idx)
    if key in _SHARED_AFFINE_DONE:
        return True
    try:
        src = [t for t in syntx_rec["transforms"]["fwdtransforms"] if str(t).endswith(".mat")][-1]
        dst_dir = os.path.join(out_dir, "antstorch_canonical_affines")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, os.path.join(dst_dir, f"pair_{pair_idx:03d}_0GenericAffine.mat"))
        import json as _json
        with open(os.path.join(dst_dir, f"pair_{pair_idx:03d}_affine_info.json"), "w") as f:
            _json.dump({"runtime_seconds": float(syntx_rec.get("syntx_affine_time", 0.0)), "pair_idx": int(pair_idx),
                        "source": "syntx"}, f)
        _SHARED_AFFINE_DONE.add(key)
        return True
    except Exception as e:  # noqa: BLE001 - never let affine sharing kill a long run
        print(f"    !! could not share syntx's affine for pair {pair_idx} ({e}); antstorch will fit its own", flush=True)
        return False


def _run_pair_model(reg, pair_idx, device, out_dir, libs):
    """One (pair, model): libs run in the given order; in fair mode syntx goes first and shares its affine."""
    recs = []
    for lib, fn in libs:
        rec = fn(reg, pair_idx, device, out_dir)
        if FAIR and lib == "syntx" and rec.get("_status") == "SUCCESS":
            rec["_affine_shared"] = _share_affine(rec, pair_idx, out_dir)
        if FAIR and lib == "antstorch" and rec.get("_status") == "SUCCESS":
            gap = None
            for r in recs:
                if r.get("_library") == "syntx" and r.get("_status") == "SUCCESS":
                    gap = abs(rec["affine_dice_sym"] - r["syntx_affine_dice_sym"])
            rec["_affine_dice_gap_vs_syntx"] = gap
            if gap is not None and gap > 1e-3:
                print(f"    !! affine Dice differs between libraries by {gap:.4f}: the shared affine was not used", flush=True)
        recs.append(rec)
    return recs


def _aggregate(results, models, skip_syntx):
    """Paired statistics over the pairs where both libraries succeeded (deformable-only times)."""
    import numpy as np
    lines = ["| Modèle | n | Dice syntx | Dice antstorch | Δ moyen ± ES | Δ médian | antstorch gagne | t déf. syntx (s) | t déf. antstorch (s) | rapport | repli. % syntx / antstorch (médiane) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in results:
        if r.get("_status") == "SUCCESS":
            by.setdefault((r["_regularizer"], r["_library"]), {})[r["pair_idx"]] = r
    for reg in models:
        A, S = by.get((reg, "antstorch"), {}), by.get((reg, "syntx"), {})
        pairs = sorted(set(A) & set(S)) if not skip_syntx else sorted(A)
        if not pairs:
            continue
        da = np.array([A[p]["dice_sym"] for p in pairs])
        ta = np.array([A[p]["_deformable_seconds"] for p in pairs])
        fa = np.median([A[p]["folding_pct"] for p in pairs])
        if skip_syntx:
            lines.append(f"| {reg} | {len(pairs)} | — | {da.mean():.4f} | — | — | — | — | {ta.mean():.1f} | — | — / {fa:.4f} |")
            continue
        ds = np.array([S[p]["dice_sym"] for p in pairs])
        ts = np.array([S[p]["_deformable_seconds"] for p in pairs])
        fs = np.median([S[p]["folding_pct"] for p in pairs])
        d = da - ds
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        lines.append(f"| {reg} | {len(pairs)} | {ds.mean():.4f} | {da.mean():.4f} | {d.mean():+.4f} ± {se:.4f} | {np.median(d):+.4f} | "
                     f"{int((d > 0).sum())}/{len(d)} | {ts.mean():.1f} | {ta.mean():.1f} | {ta.mean() / ts.mean():.2f}× | {fs:.4f} / {fa:.4f} |")
    return lines


# ======================================================================================================================
# FireANTs (optional third arm, --fireants). Merged from the former compare_fireants_mindboggle.py (2026-10-15).
# ======================================================================================================================
FIREANTS_ROOT = os.path.expanduser("~/Pkg/fireants")


# GreedyRegistration settings. optimizer_lr stays at FireANTs' own tutorial value (0.5): a 2026-09-22 run showed that
# halving it undertrains FireANTs (project doc). Only the iteration budget differs between the two schedules.
_FIREANTS_KWARGS = {
    "matched": dict(scales=[4, 2, 1], iterations=[100, 100, 20], cc_kernel_size=5, optimizer_lr=0.5),
    "tutorial": dict(scales=[4, 2, 1], iterations=[200, 100, 25], cc_kernel_size=5, optimizer_lr=0.5),
}
FIREANTS_SCHEDULE = "matched"
# Preset = the "Tuned Greedy Compositive" configuration of syntx's examples/compare_fireants_vs_syntx.py / docs/v_fire_ants/README.md
# (grad=2.0, warp=0.35, lr=0.4, iterations [100,100,50], CC window 5). That document reports 3.7 % folded voxels for it.
FIREANTS_PRESETS = {
    "syntx-doc": dict(smooth_warp_sigma=0.35, smooth_grad_sigma=2.0, optimizer_lr=0.4, cc_kernel_size=5, iterations=[100, 100, 50]),
}
FIREANTS_NORMALIZE = True  # feed FireANTs the same auto-normalised images syntx's benchmark registers (see _run_fireants)
FIREANTS_OVERRIDES = {}  # optional diagnostic overrides of _FIREANTS_KWARGS entries (set from the CLI)
FIREANTS_TOLERANCE = None  # None: FireANTs' own early-stop setting (tolerance 1e-6 on the loss slope, 10 iterations)
FIREANTS_LABEL = "fireants"


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
        if FIREANTS_NORMALIZE:
            # syntx's benchmark registers images normalised with normalize_image(method='auto') (entropy-optimal foreground
            # percentile clip to [0, 1]); antstorch.syn.core.utils.normalize_image is the same function (checked numerically).
            # Scoring below still uses the original images: only the intensities fed to FireANTs change.
            from antstorch.syn.core.utils import normalize_image
            ants.image_write(normalize_image(fi_ants, method="auto"), fixed_path)
            ants.image_write(normalize_image(mi_ants, method="auto"), moving_path)
        else:
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
        smooth_grad_sigma = float(kw.pop("smooth_grad_sigma", 1.0))
        extra_kw = {} if FIREANTS_TOLERANCE is None else dict(tolerance=float(FIREANTS_TOLERANCE))
        _sync(device)
        t_start = time.time()
        reg_obj = GreedyRegistration(
            scales=kw["scales"], iterations=kw["iterations"],
            fixed_images=fixed_batch, moving_images=moving_batch,
            loss_type="cc", cc_kernel_size=kw["cc_kernel_size"],
            deformation_type="compositive",
            optimizer="adam", optimizer_lr=kw["optimizer_lr"],
            smooth_warp_sigma=smooth_warp_sigma, smooth_grad_sigma=smooth_grad_sigma,
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
            "_fireants_schedule": FIREANTS_SCHEDULE, "_fireants_normalized": bool(FIREANTS_NORMALIZE),
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


SWEEP_GRIDS = {
    # set 1: smoothing x CC window, then the learning rate at (0.5, 7)
    1: ([dict(smooth_warp_sigma=sg, cc_kernel_size=k) for sg in (0.25, 0.5, 1.0, 2.0) for k in (5, 7, 9)]
        + [dict(smooth_warp_sigma=0.5, cc_kernel_size=7, optimizer_lr=lr) for lr in (0.1, 0.25, 1.0)]),
    # set 2 (after set 1: Dice kept rising with the learning rate up to 1.0): larger learning rates x smoothing, window 7
    2: [dict(smooth_warp_sigma=sg, cc_kernel_size=7, optimizer_lr=lr) for lr in (1.0, 2.0, 4.0) for sg in (0.35, 0.5, 0.75)],
    # set 3: gradient smoothing (never varied in sets 1-2, fixed at 1.0) x warp smoothing, at syntx's documented lr 0.4 / window 5
    3: [dict(smooth_grad_sigma=g, smooth_warp_sigma=w, optimizer_lr=0.4, cc_kernel_size=5)
        for g in (1.0, 2.0, 3.0) for w in (0.25, 0.35, 0.5)],
}
FOLDING_LIMIT_PCT = 0.01  # a FireANTs setting is only "comparable" to syntx / ANTsTorch if its median folding stays below this
SWEEP_SET = 1


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
    for cfg in SWEEP_GRIDS[SWEEP_SET]:
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
          "| smooth_warp_sigma | smooth_grad_sigma | cc_kernel_size | optimizer_lr | n | Dice sym | Dice fixed | t déf. (s) | final-level loss | folding % (median) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    base = _FIREANTS_KWARGS[FIREANTS_SCHEDULE]
    for cfg, n, tot, ds, df, t, ll, fo in sorted(rows, key=lambda r: -r[4]):
        md.append(f"| {cfg.get('smooth_warp_sigma', 1.0)} | {cfg.get('smooth_grad_sigma', 1.0)} | {cfg.get('cc_kernel_size', base['cc_kernel_size'])} | "
                  f"{cfg.get('optimizer_lr', base['optimizer_lr'])} | {n}/{tot} | {ds:.4f} | {df:.4f} | {t:.1f} | {ll:.4f} | {fo:.4f} |")
    md.append("")
    ok_rows = [r for r in rows if r[7] < FOLDING_LIMIT_PCT]
    if ok_rows:
        b = max(ok_rows, key=lambda r: r[4])
        md.append(f"**Best setting with median folding < {FOLDING_LIMIT_PCT} %** (comparable regularity to syntx / ANTsTorch): "
                  f"{b[0]} -> Dice fixed {b[4]:.4f}, Dice sym {b[3]:.4f}, folding {b[7]:.4f} %.")
    else:
        md.append(f"No setting kept the median folding below {FOLDING_LIMIT_PCT} %.")
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
    lines = ["| Modèle | n | Dice affine (syntx) | Dice syntx | Dice antstorch | Dice fireants (sym / fixe) | Dice fixe syntx / antstorch | Δ antstorch−syntx (moy ± ES) | "
             "Δ fireants−syntx (moy ± ES) | t déf. syntx | t déf. antstorch | t déf. fireants (s) | "
             "repli. % médiane s / a / f |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
            f"{np.nanmean([S[p].get('dice_fixed', float('nan')) for p in ps]):.4f} / "
            f"{np.nanmean([A[p].get('dice_fixed', float('nan')) for p in ps]):.4f} | "
            f"{xa.mean():+.4f} ± {se(xa):.4f} | {xf.mean():+.4f} ± {se(xf):.4f} | {ts.mean():.1f} | {ta.mean():.1f} | "
            f"{tf.mean():.1f} | {fm[0]:.4f} / {fm[1]:.4f} / {fm[2]:.4f} |")
    return lines


def _fireants_pair_table(results, pairs, models):
    """Per-pair table: Dice (sym) of syntx / ANTsTorch / FireANTs (sym and fixed) and deformable times."""
    by_key = {}
    for r in results:
        by_key.setdefault((r.get("pair_idx"), r["_regularizer"]), {})[r["_library"]] = r
    out = ["| Paire | Modèle | syntx | antstorch | fireants (sym / fixe) | Δ a−s | Δ f−s | t déf. s | t déf. a | t déf. f (s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for pair_idx in pairs:
        f = by_key.get((pair_idx, FIREANTS_LABEL), {}).get("fireants", {})
        f_ok = f.get("_status") == "SUCCESS"
        for reg in models:
            e = by_key.get((pair_idx, reg), {})
            s, a = e.get("syntx", {}), e.get("antstorch", {})
            s_ok, a_ok = s.get("_status") == "SUCCESS", a.get("_status") == "SUCCESS"
            cells = [str(pair_idx), reg,
                     f"{s['dice_sym']:.4f}" if s_ok else "ÉCHEC", f"{a['dice_sym']:.4f}" if a_ok else "ÉCHEC",
                     f"{f['dice_sym']:.4f} / {f['dice_fixed']:.4f}" if f_ok else "ÉCHEC",
                     f"{a['dice_sym'] - s['dice_sym']:+.4f}" if (s_ok and a_ok) else "—",
                     f"{f['dice_sym'] - s['dice_sym']:+.4f}" if (s_ok and f_ok) else "—",
                     f"{s['_deformable_seconds']:.1f}" if s_ok else "—", f"{a['_deformable_seconds']:.1f}" if a_ok else "—",
                     f"{f['_deformable_seconds']:.1f}" if f_ok else "—"]
            out.append("| " + " | ".join(cells) + " |")
    return out


def main():
    global FAIR
    global FIREANTS_SCHEDULE, FIREANTS_TOLERANCE, FIREANTS_OVERRIDES, SWEEP_SET, FIREANTS_NORMALIZE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, nargs="+", default=PAIR_INDICES_DEFAULT)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODELS,
                    help="Default: gaussian sobolev dsti (the arms with a syntx counterpart of the same algorithm).")
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
    ap.add_argument("--own-defaults", action="store_true",
                    help="Disable the fair mode: antstorch runs with its own defaults (end-of-level inverses, analytical "
                         "gradients, no syntx parity) and fits its own affine, as before 2026-10-14. Default (fair): syntx runs "
                         "first, antstorch starts from the same affine file and runs syntx's loop (per-iteration inverses with "
                         "10 steps, syntx parity, autograd; dsti with syntx.syn()'s reg_adam), and times compared are deformable-only.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the throwaway first run per library (device/kernel initialisation) that is excluded from the timings.")
    # --- FireANTs (optional third arm; CUDA only) ---
    ap.add_argument("--fireants", action="store_true",
                    help="Add FireANTs (GreedyRegistration) as a third arm. It runs ONCE per pair, after syntx, from the same shared "
                         "syntx affine, on the same auto-normalised images; times are deformable-only and synchronized. Needs a CUDA "
                         "GPU and ~/Pkg/fireants. Reported with Dice sym and Dice fixed (FireANTs' inverse is imperfect).")
    ap.add_argument("--fireants-sweep", action="store_true",
                    help="FireANTs-only hyper-parameter sweep (implies --fireants; syntx / ANTsTorch are NOT run). Reuses the syntx "
                         "affines cached in --out-dir by an earlier run. Writes fireants_sweep.md / fireants_sweep.json.")
    ap.add_argument("--sweep-set", type=int, choices=sorted(SWEEP_GRIDS), default=1,
                    help="Grid for --fireants-sweep: 1 = warp smoothing x CC window (+ small lr set); 2 = larger learning rates "
                         "(1, 2, 4) x warp smoothing; 3 = gradient smoothing (1, 2, 3) x warp smoothing (0.25, 0.35, 0.5) at lr 0.4.")
    ap.add_argument("--fireants-preset", choices=sorted(FIREANTS_PRESETS), default=None,
                    help="Start from a named FireANTs configuration; explicit --fireants-* options still override it. 'syntx-doc' = the "
                         "tuned greedy-compositive setting of syntx's docs/v_fire_ants (which reports 3.7 %% folding).")
    ap.add_argument("--fireants-iterations", type=int, nargs="+", default=None,
                    help="FireANTs iterations per pyramid level (3 values). '1 1 1' with --fireants-lr 0 tests the affine export / "
                         "scoring chain: the Dice should equal the affine-only Dice.")
    ap.add_argument("--fireants-lr", type=float, default=None, help="FireANTs Adam learning rate (default 0.5).")
    ap.add_argument("--fireants-warp-sigma", type=float, default=None,
                    help="FireANTs smooth_warp_sigma (this script's default 1.0; FireANTs' own default 0.5).")
    ap.add_argument("--fireants-grad-sigma", type=float, default=None, help="FireANTs smooth_grad_sigma (default 1.0).")
    ap.add_argument("--fireants-cc-kernel", type=int, default=None, help="FireANTs cc_kernel_size (default 5).")
    ap.add_argument("--fireants-no-early-stop", action="store_true",
                    help="Disable FireANTs' convergence-based early stop (tolerance=1e9) so every nominal iteration runs.")
    ap.add_argument("--tutorial-schedule", action="store_true",
                    help="FireANTs iterations [200,100,25] (its tutorial) instead of [100,100,20] (same budget as syntx).")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Feed FireANTs the raw images instead of the auto-normalised ones syntx registers.")
    args = ap.parse_args()
    if args.fireants_sweep:
        args.fireants = True
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")
    if args.out_dir is None:
        base = os.path.expanduser("~/Desktop/syntx_antstorch_mindboggle_comparison")
        args.out_dir = base + ("_legacy" if args.legacy_defaults else "") + ("_own" if args.own_defaults or args.legacy_defaults else "")
    FAIR = not (args.own_defaults or args.legacy_defaults)
    if args.fireants:
        import torch  # (local: main() imports torch locally elsewhere, which makes the name local to this function)
        if not (str(args.device).startswith("cuda") and torch.cuda.is_available()):
            sys.exit(f"--fireants needs a CUDA GPU (got --device {args.device!r}, cuda available: {torch.cuda.is_available()}). "
                     f"Run it on the GPU machine (e.g. hyperwolf), or drop --fireants.")
        FIREANTS_SCHEDULE = "tutorial" if args.tutorial_schedule else "matched"
        FIREANTS_TOLERANCE = 1e9 if args.fireants_no_early_stop else None
        SWEEP_SET = args.sweep_set
        FIREANTS_NORMALIZE = not args.no_normalize
        if args.fireants_preset:
            FIREANTS_OVERRIDES.update(FIREANTS_PRESETS[args.fireants_preset])
        if args.fireants_grad_sigma is not None:
            FIREANTS_OVERRIDES["smooth_grad_sigma"] = args.fireants_grad_sigma
        if args.fireants_iterations is not None:
            FIREANTS_OVERRIDES["iterations"] = list(args.fireants_iterations)
        if args.fireants_lr is not None:
            FIREANTS_OVERRIDES["optimizer_lr"] = args.fireants_lr
        if args.fireants_warp_sigma is not None:
            FIREANTS_OVERRIDES["smooth_warp_sigma"] = args.fireants_warp_sigma
        if args.fireants_cc_kernel is not None:
            FIREANTS_OVERRIDES["cc_kernel_size"] = args.fireants_cc_kernel
    if FAIR and args.skip_syntx:
        print("!! --skip-syntx with the fair mode: syntx's affine is unavailable, antstorch will use/fit its own affine "
              "(the Dice comparison is then no longer on a shared start).", flush=True)

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
    if args.fireants_sweep:
        _run_sweep(args.pairs, args.device, args.out_dir, DATA_DIR)
        return
    results = []
    json_path = os.path.join(args.out_dir, "comparison_results.json")

    if args.skip_syntx:
        libs = (("antstorch", _run_antstorch),)
    elif FAIR:
        libs = (("syntx", _run_syntx), ("antstorch", _run_antstorch))  # syntx first: it produces the shared affine
    else:
        libs = (("antstorch", _run_antstorch), ("syntx", _run_syntx))

    if not args.no_warmup and args.pairs and args.models:
        # Throwaway run of every library on the first pair/model so CUDA/MPS initialisation, kernel compilation and
        # allocator warm-up are not charged to the first timed pair. Results are discarded.
        warm_dir = os.path.join(args.out_dir, "_warmup")
        os.makedirs(warm_dir, exist_ok=True)
        print(f"[warm-up] pair {args.pairs[0]} / {args.models[0]} (not recorded) ...", flush=True)
        _run_pair_model(args.models[0], args.pairs[0], args.device, warm_dir, libs)
        if args.fireants:
            _run_fireants(args.pairs[0], args.device, warm_dir, DATA_DIR)
        _SHARED_AFFINE_DONE.clear()

    total = len(args.pairs) * len(args.models) * len(libs)
    done = 0
    for pair_idx in args.pairs:
        for reg in args.models:
            print(f"pair {pair_idx} / {reg} ...", flush=True)
            for lib_pos, rec in enumerate(_run_pair_model(reg, pair_idx, args.device, args.out_dir, libs)):
                lib = libs[lib_pos][0]
                print(f"[{done + 1}/{total}] {lib} / {reg} / pair {pair_idx}", flush=True)
                results.append(rec)
                done += 1
                status = rec.get("_status")
                dice = rec.get("dice_sym", float("nan"))
                secs = rec.get("_deformable_seconds", float("nan"))
                print(f"    -> {status} dice_sym={dice:.4f} deformable={secs:.1f}s" if status == "SUCCESS"
                      else f"    -> {status}: {rec.get('_error')}", flush=True)
                # Write incrementally so a partial run is never lost.
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
        if args.fireants:  # once per pair, after syntx has produced (and shared) the affine
            print(f"pair {pair_idx} / fireants ...", flush=True)
            rec = _run_fireants(pair_idx, args.device, args.out_dir, DATA_DIR)
            results.append(rec)
            if rec.get("_status") == "SUCCESS":
                print(f"    -> fireants dice_sym={rec['dice_sym']:.4f} dice_fixed={rec['dice_fixed']:.4f} "
                      f"deformable={rec['_deformable_seconds']:.1f}s iterations_run={rec['_fireants_iterations_run']}/"
                      f"{rec['_fireants_iterations_nominal']}", flush=True)
            else:
                print(f"    -> fireants {rec.get('_status')}: {rec.get('_error')}", flush=True)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

    # Build summary table
    by_key = {}
    for r in results:
        by_key.setdefault((r.get("pair_idx"), r["_regularizer"]), {})[r["_library"]] = r

    table_lines = ["| Paire | Modèle | syntx dice_sym | antstorch dice_sym | Δ (antstorch − syntx) | syntx t déf. (s) | antstorch t déf. (s) |",
                   "|---|---|---|---|---|---|---|"]
    for pair_idx in args.pairs:
        for reg in args.models:
            pair = by_key.get((pair_idx, reg), {})
            s, a = pair.get("syntx", {}), pair.get("antstorch", {})
            s_ok, a_ok = s.get("_status") == "SUCCESS", a.get("_status") == "SUCCESS"
            s_dice = f"{s['dice_sym']:.4f}" if s_ok else ("—" if args.skip_syntx else "ÉCHEC")
            a_dice = f"{a['dice_sym']:.4f}" if a_ok else "ÉCHEC"
            delta = f"{(a['dice_sym'] - s['dice_sym']):+.4f}" if (s_ok and a_ok) else "—"
            s_t = f"{s.get('_deformable_seconds', float('nan')):.1f}" if s_ok else "—"
            a_t = f"{a.get('_deformable_seconds', float('nan')):.1f}" if a_ok else "—"
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

    _eff_schedule = args.inverse_schedule or ("per_iteration" if (FAIR or args.legacy_defaults) else "end_of_level")
    md_lines = ["# Comparaison syntx vs antstorch — Mindboggle-101 (dense SyN)", "",
                f"Paires : {args.pairs}  ", f"Modèles : {args.models}  ", f"Device : {args.device or 'auto'}  ", f"antstorch inverse_method : {args.inverse_method or 'default (anderson)'}  ", f"Mode : {'JUSTE — affine syntx partagée ; antstorch iso-algorithme (inverses par itération 10 pas, parité syntx, autograd ; dsti = syn_reg_adam) ; temps déformable seul' if FAIR else 'défauts propres de chaque bibliothèque (affines propres)'}  ",
                f"antstorch defaults : {'legacy' if args.legacy_defaults else 'syntx-aligned'}  ",
f"antstorch inverse_schedule : {_eff_schedule}"
                + (f" ({args.end_of_level_inverse_steps or 30} steps)" if _eff_schedule == "end_of_level" else "") + "  ",
                f"syntx arms : {'skipped' if args.skip_syntx else 'run'}", ""]
    md_lines.append("## Résumé apparié (paires où les deux bibliothèques ont réussi ; temps = partie déformable seule)")
    md_lines.append("")
    md_lines.extend(_aggregate(results, args.models, args.skip_syntx))
    md_lines.append("")
    md_lines.append("## Détail par paire")
    md_lines.append("")
    md_lines.extend(table_lines)
    md_lines.append("")
    if args.fireants:
        md_lines.append("## FireANTs (3ᵉ bras ; une exécution par paire, comparée à chaque modèle)")
        md_lines.append("")
        md_lines.append(f"GreedyRegistration, itérations {FIREANTS_OVERRIDES.get('iterations', _FIREANTS_KWARGS[FIREANTS_SCHEDULE]['iterations'])}, "
                        f"surcharges {FIREANTS_OVERRIDES or 'aucune'}, images {'normalisées (auto)' if FIREANTS_NORMALIZE else 'brutes'}, "
                        f"early stop {'désactivé' if FIREANTS_TOLERANCE else 'par défaut de FireANTs'} ; temps = construction + optimize, "
                        f"synchronisé, hors affine / E-S / métriques. `dice_fixed` (déformation directe seule) est la mesure la plus "
                        f"fiable pour FireANTs : son inverse est imparfait.")
        md_lines.append("")
        md_lines.extend(_aggregate3(results, args.models, args.pairs))
        md_lines.append("")
        md_lines.extend(_fireants_pair_table(results, args.pairs, args.models))
        md_lines.append("")
    md_lines.append(f"Résultats bruts : `{json_path}`")

    md_path = os.path.join(args.out_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))
    print(f"\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
