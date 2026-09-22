#!/usr/bin/env python3
"""Add FireANTs (github.com/rohitrango/fireants, ~/Pkg/fireants) as a third
arm to the syntx-vs-antstorch Mindboggle-101 comparison
(tools/compare_syntx_antstorch_mindboggle.py).

Design decisions (confirmed with the user 2026-09-22, see the project doc,
"Cadre du recalage d'ANTsTorch", section on the FireANTs integration):

  - FireANTs engine used: `fireants.registration.greedy.GreedyRegistration`
    (compositive warp, Gaussian warp/grad-field smoothing) -- the direct
    conceptual analog of syntx/antstorch's own greedy dense-SyN loop, and
    therefore compared against the "gaussian" regularizer arm only. FireANTs
    has no Sobolev/DST-I/B-spline regularizer equivalent in this codebase,
    so `--models sobolev dsti bspline` are accepted (to keep the syntx/
    antstorch columns of the table populated) but the FireANTs column is
    reported as "n/a" for anything other than "gaussian".
  - Affine initialization: FireANTs' own `GreedyRegistration(init_affine=...)`
    is fed the *same* shared canonical affine antstorch's own `gaussian_syn`/
    `sobolev_syn`/`dsti_syn`/`bspline_syn` arms already use (via
    `antstorch.benchmark.evaluate._fit_or_load_canonical_affine`, reusing the
    exact same on-disk `.mat` cache under `--out-dir/antstorch_canonical_affines`)
    -- not FireANTs' own affine stage. This preserves the "every model variant
    starts from the identical affine" fairness invariant already established
    for every antstorch arm (project doc, §17/§18).

    Coordinate-convention note (verified by reading fireants/io/image.py and
    antstorch/ants_transform_io.py side by side, not assumed): both express
    their affine matrices in plain ITK physical `(x, y[, z])` order --
    `Image`'s `px2phy` is built directly from `itk_image.GetSpacing()`/
    `GetOrigin()`/`GetDirection()`, exactly like `ants`/ITK's own
    `AffineTransform` parameters that `ants_transform_io.read_affine_transform()`
    reads back. `GreedyRegistration.get_inverse_warp_parameters()` shows
    `self.affine` is sandwiched as `moving_p2t @ self.affine @ fixed_t2p`,
    i.e. `self.affine` itself must already be a *physical*-space affine
    mapping fixed-physical to moving-physical points -- the same direction
    `ants.registration()`'s own `0GenericAffine.mat` encodes (project doc,
    §30's checkerboard-verified convention). No axis reordering or LPS/RAS
    flip is needed: the canonical-affine `(matrix, translation)` pair is
    embedded directly into a `[1, dim+1, dim+1]` homogeneous tensor and
    passed straight through as `init_affine`.
  - Output format: FireANTs' own `DeformableMixin.save_as_ants_transforms()`
    already writes a single ANTs-compatible displacement-field `.nii.gz`
    with the affine baked in (physical space, `CopyInformation`-matched to
    the fixed image for the forward field and to the moving image for the
    inverse field) -- exactly the same "single pre-composed warp" convention
    antstorch's own `*_svf` model families use (project doc, §30):
    `fwdtransforms=[warp]`, `invtransforms=[inverse_warp]`,
    `whichtoinvert_inv=[False]`. No separate affine file is written or
    needed for FireANTs specifically (its affine is folded into the warp).

Reuses `antstorch.benchmark.data.load_mindboggle_pair` (same N4-corrected
pair loading as every other arm), `antstorch.benchmark.evaluate.
_fit_or_load_canonical_affine` (same canonical-affine cache), and
`antstorch.benchmark.metrics.compute_bidirectional_dice`/
`compute_jacobian_metrics` (same Dice/Jacobian scoring) -- so FireANTs'
numbers are directly comparable to every number already produced by
tools/compare_syntx_antstorch_mindboggle.py, not a separately-defined
metric.

This is a *separate* script from compare_syntx_antstorch_mindboggle.py
(kept intact) that re-runs syntx and antstorch itself for the requested
pairs/models (by importing its `_run_syntx`/`_run_antstorch` functions
directly -- both scripts must live in the same directory) and adds a third
`fireants` column, so a single run produces one 3-way table.

Usage (place next to compare_syntx_antstorch_mindboggle.py, e.g. in
`tools/`):

    python tools/compare_fireants_mindboggle.py --pairs 0 24 88 \\
        --models gaussian --device cuda:1

Requires `~/Pkg/fireants` to be cloned on whatever machine this runs on
(same convention as `~/Pkg/syntx`/`~/Pkg/ANTsTorch` already used by
compare_syntx_antstorch_mindboggle.py) -- confirmed present on cerulean
under `/Users/ntustison/Pkg/fireants` as of 2026-09-22; clone it in the
same place on any other machine (e.g. hyperwolf) before running there.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

PAIR_INDICES_DEFAULT = [0, 24, 88]
MODELS = ["gaussian", "sobolev", "dsti", "bspline"]
FIREANTS_SUPPORTED_MODELS = {"gaussian"}

SYNTX_ROOT = os.path.expanduser("~/Pkg/syntx")
ANTSTORCH_ROOT = os.path.expanduser("~/Pkg/ANTsTorch")
FIREANTS_ROOT = os.path.expanduser("~/Pkg/fireants")
DATA_DIR = os.path.expanduser("~/Data/Public/Mindboggle/Volumes")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING_SCRIPT = os.path.join(_HERE, "compare_syntx_antstorch_mindboggle.py")


def _load_sibling_module():
    """Imports `_run_syntx`/`_run_antstorch` from the existing comparison
    script sitting next to this one, rather than duplicating their (fairly
    involved) canonical-affine/matched-kwargs logic.
    """
    if not os.path.exists(_SIBLING_SCRIPT):
        raise FileNotFoundError(
            f"Expected compare_syntx_antstorch_mindboggle.py next to this script "
            f"at '{_SIBLING_SCRIPT}' (to reuse its _run_syntx/_run_antstorch) -- "
            f"place both files in the same directory (e.g. tools/)."
        )
    spec = importlib.util.spec_from_file_location("compare_syntx_antstorch_mindboggle", _SIBLING_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Same --matched schedule used for "gaussian" in the sibling script (grad_step
# 0.25 / cc2 / [4,2,1] / [100,100,20]), translated into FireANTs'
# GreedyRegistration kwargs. Left as a fixed default (not currently
# conditional on a --matched flag here) since FireANTs has no separate
# "out-of-the-box default" schedule of its own worth comparing against --
# its tutorial defaults (iterations=[200,100,25], optimizer_lr=0.5) are used
# for an --matched=False run instead.
#
# Revision 2026-09-22 (project doc §39): the original True-branch used
# optimizer_lr=0.25 -- an untested guess, halved from FireANTs' own tutorial
# default with no empirical basis. A real 10-pair run showed this actually
# *undertrains* GreedyRegistration relative to the unmatched (tutorial-lr)
# run: dice_fixed (the forward-registration-only metric, unaffected by
# FireANTs' separate field-inversion issue) dropped on the same pairs
# (0/24/88: 0.465/0.440/0.490 unmatched -> 0.4225/0.3949/0.4492 matched),
# and the overall fireants-vs-syntx gap widened slightly (-0.228 -> -0.242
# mean delta) instead of narrowing. optimizer_lr is therefore kept at
# FireANTs' own tutorial value (0.5) in both branches now -- only the
# iteration budget is still reduced for the matched branch, for rough
# wall-clock parity with the syntx/antstorch matched schedule, without
# artificially throttling the learning rate FireANTs' own authors tuned.
_FIREANTS_KWARGS = {
    True: dict(scales=[4, 2, 1], iterations=[100, 100, 20], cc_kernel_size=5, optimizer_lr=0.5),
    False: dict(scales=[4, 2, 1], iterations=[200, 100, 25], cc_kernel_size=5, optimizer_lr=0.5),
}


def _run_fireants(reg, pair_idx, device, out_dir, matched=False):
    t0 = time.time()
    if reg not in FIREANTS_SUPPORTED_MODELS:
        return {
            "_library": "fireants", "_regularizer": reg, "pair_idx": pair_idx,
            "_status": "SKIPPED",
            "_error": (
                f"FireANTs arm only implemented for {sorted(FIREANTS_SUPPORTED_MODELS)} "
                f"(GreedyRegistration has no sobolev/dsti/bspline regularizer equivalent "
                f"in this codebase) -- got '{reg}'."
            ),
            "_wall_seconds": time.time() - t0,
        }

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
        resolved_device = device or ("cuda" if torch.cuda.is_available() else
                                      ("mps" if torch.backends.mps.is_available() else "cpu"))

        pair_data = load_mindboggle_pair(pair_idx=pair_idx, data_dir=DATA_DIR, use_n4=True)
        fi_ants, mi_ants = pair_data["fixed"], pair_data["moving"]
        fl_ants, ml_ants = pair_data["fixed_label"], pair_data["moving_label"]
        fixed_id, moving_id, cohort_type = pair_data["fixed_id"], pair_data["moving_id"], pair_data["pair_type"]
        dim = fi_ants.dimension

        # Same shared canonical-affine cache the antstorch arms use, so the
        # fireants arm starts from the identical affine as gaussian_syn/
        # sobolev_syn/dsti_syn/bspline_syn for this pair.
        canonical_affine_dir = os.path.join(out_dir, "antstorch_canonical_affines")
        matrix, translation, _t_aff, _affine_path = _fit_or_load_canonical_affine(
            fi_ants, mi_ants, pair_idx, canonical_affine_dir, resolved_device, False
        )

        pair_out_dir = os.path.join(out_dir, "fireants_registration", f"pair_{pair_idx:03d}_{reg}")
        os.makedirs(pair_out_dir, exist_ok=True)
        fixed_path = os.path.join(pair_out_dir, "fixed.nii.gz")
        moving_path = os.path.join(pair_out_dir, "moving.nii.gz")
        ants.image_write(fi_ants, fixed_path)
        ants.image_write(mi_ants, moving_path)

        fixed_img = Image.load_file(fixed_path, device=resolved_device)
        moving_img = Image.load_file(moving_path, device=resolved_device)
        fixed_batch = BatchedImages([fixed_img])
        moving_batch = BatchedImages([moving_img])

        affine_4x4 = torch.eye(dim + 1, dtype=torch.float32)
        affine_4x4[:dim, :dim] = matrix
        affine_4x4[:dim, -1] = translation
        init_affine = affine_4x4.unsqueeze(0).to(resolved_device)

        kw = _FIREANTS_KWARGS[bool(matched)]
        reg_obj = GreedyRegistration(
            scales=kw["scales"], iterations=kw["iterations"],
            fixed_images=fixed_batch, moving_images=moving_batch,
            loss_type="cc", cc_kernel_size=kw["cc_kernel_size"],
            deformation_type="compositive",
            optimizer="adam", optimizer_lr=kw["optimizer_lr"],
            smooth_warp_sigma=1.0, smooth_grad_sigma=1.0,
            init_affine=init_affine,
        )
        reg_obj.optimize()

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
            "cohort_type": cohort_type, "fixed_id": fixed_id, "moving_id": moving_id,
            "dice_sym": float(dice_sym), "dice_fixed": float(df_fixed), "dice_moving": float(df_moving),
            "folding_pct": float(jac.get("folding_pct", float("nan"))),
            "min_jacobian": float(jac.get("min", float("nan"))),
            "runtime_seconds": time.time() - t0,
            "_wall_seconds": time.time() - t0,
            "_library": "fireants", "_regularizer": reg, "pair_idx": pair_idx,
            "_status": "SUCCESS",
        }
    except Exception as e:
        rec = {
            "_library": "fireants", "_regularizer": reg, "pair_idx": pair_idx,
            "_status": "FAILED", "_error": str(e), "_traceback": traceback.format_exc(),
            "_wall_seconds": time.time() - t0,
        }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, nargs="+", default=PAIR_INDICES_DEFAULT)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--matched", action="store_true",
                     help="Same meaning as in compare_syntx_antstorch_mindboggle.py for the "
                          "syntx/antstorch columns; also selects the matched FireANTs "
                          "GreedyRegistration schedule (see _FIREANTS_KWARGS).")
    args = ap.parse_args()
    if args.out_dir is None:
        base = os.path.expanduser("~/Desktop/fireants_syntx_antstorch_mindboggle_comparison")
        args.out_dir = base + ("_matched" if args.matched else "")
    os.makedirs(args.out_dir, exist_ok=True)

    sibling = _load_sibling_module()

    results = []
    json_path = os.path.join(args.out_dir, "comparison_results.json")
    runners = (("antstorch", sibling._run_antstorch), ("syntx", sibling._run_syntx), ("fireants", _run_fireants))
    total = len(args.pairs) * len(args.models) * len(runners)
    done = 0
    for pair_idx in args.pairs:
        for reg in args.models:
            for lib, fn in runners:
                print(f"[{done + 1}/{total}] {lib} / {reg} / pair {pair_idx} ...", flush=True)
                rec = fn(reg, pair_idx, args.device, args.out_dir, matched=args.matched)
                results.append(rec)
                done += 1
                status = rec.get("_status")
                dice = rec.get("dice_sym", float("nan"))
                secs = rec.get("_wall_seconds", float("nan"))
                if status == "SUCCESS":
                    print(f"    -> {status} dice_sym={dice:.4f} wall={secs:.1f}s", flush=True)
                else:
                    print(f"    -> {status}: {rec.get('_error')}", flush=True)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)

    by_key = {}
    for r in results:
        by_key.setdefault((r.get("pair_idx"), r["_regularizer"]), {})[r["_library"]] = r

    table_lines = [
        "| Paire | Modèle | syntx dice_sym | antstorch dice_sym | fireants dice_sym | "
        "Δ (antstorch − syntx) | Δ (fireants − syntx) | syntx t(s) | antstorch t(s) | fireants t(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pair_idx in args.pairs:
        for reg in args.models:
            entry = by_key.get((pair_idx, reg), {})
            s, a, f = entry.get("syntx", {}), entry.get("antstorch", {}), entry.get("fireants", {})
            s_ok, a_ok = s.get("_status") == "SUCCESS", a.get("_status") == "SUCCESS"
            f_ok = f.get("_status") == "SUCCESS"
            s_dice = f"{s['dice_sym']:.4f}" if s_ok else "ÉCHEC"
            a_dice = f"{a['dice_sym']:.4f}" if a_ok else "ÉCHEC"
            f_dice = f"{f['dice_sym']:.4f}" if f_ok else ("n/a" if f.get("_status") == "SKIPPED" else "ÉCHEC")
            delta_a = f"{(a['dice_sym'] - s['dice_sym']):+.4f}" if (s_ok and a_ok) else "—"
            delta_f = f"{(f['dice_sym'] - s['dice_sym']):+.4f}" if (s_ok and f_ok) else "—"
            s_t = f"{s.get('runtime_seconds', s.get('_wall_seconds', float('nan'))):.1f}" if s_ok else "—"
            a_t = f"{a.get('runtime_seconds', a.get('_wall_seconds', float('nan'))):.1f}" if a_ok else "—"
            f_t = f"{f.get('runtime_seconds', f.get('_wall_seconds', float('nan'))):.1f}" if f_ok else "—"
            label = reg + (" (non-officiel)" if reg == "bspline" else "")
            table_lines.append(
                f"| {pair_idx} | {label} | {s_dice} | {a_dice} | {f_dice} | {delta_a} | {delta_f} | {s_t} | {a_t} | {f_t} |"
            )

    table = "\n".join(table_lines)
    print("\n" + table)
    with open(os.path.join(args.out_dir, "comparison_table.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
