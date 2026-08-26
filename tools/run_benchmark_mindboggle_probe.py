#!/usr/bin/env python3
"""Run antstorch.benchmark.evaluate_mindboggle_pair() over a small probe set
of Mindboggle-101 pairs, across every ANTsTorch-native model variant.

antstorch.benchmark was ported as a "core only" harness (see the project
doc, section 18): it has evaluate_mindboggle_pair() but deliberately no
CLI and no cohort orchestrator (resume/cache/JSON aggregation across all 90
pairs). This script is the standalone demo/driver that fills that gap for a
first local run -- in the same spirit as tools/run_syn_registration.py and
tools/run_bspline_svf_registration.py, not a re-introduction of the ported-out
orchestrator (no resume, no cache, no parallel dispatch).

By default it evaluates the same 6-pair probe subset used elsewhere in this
project's history ({0, 1, 2, 45, 67, 82}) across all 5 ANTsTorch model
variants -- gaussian_syn, sobolev_syn, dsti_syn, bspline_syn (all four the
dense antstorch.syn.syn_registration() SyN stage, differing only in fluid/
B-spline regularizer), plus bspline_svf (a different transformation family
entirely, a stationary velocity field -- the '_syn' vs. '_svf' suffixes are
deliberate disambiguation, not decoration) -- 30 registrations total.
Results are written as JSON plus a plain-text summary table; the canonical
affine per pair is fit once and cached to disk, then reused across every
model variant for that pair (the fairness invariant the harness preserves
from syntx.benchmark).

Example
-------
First, just check the dataset is where the harness expects it::

    PYTHONPATH=. python tools/run_benchmark_mindboggle_probe.py --check-only \\
        --data-dir /Users/ntustison/Data/Public/Mindboggle/Volumes

Run the default 6-pair x 5-model probe (uses ANTSTORCH_MINDBOGGLE_DATA_DIR
if set, otherwise pass --data-dir explicitly)::

    PYTHONPATH=. python tools/run_benchmark_mindboggle_probe.py \\
        --data-dir /Users/ntustison/Data/Public/Mindboggle/Volumes \\
        --device mps --verbose

Run just two models on a couple of pairs, with a faster iteration schedule
for a quick smoke test before committing to the full probe::

    PYTHONPATH=. python tools/run_benchmark_mindboggle_probe.py \\
        --pair-idx 0 1 --models gaussian_syn bspline_svf \\
        --reg-iterations 20 20 10 --device mps

Run the full 90-pair cohort (no resume/cache -- expect a long run; see the
project doc's runtime note, roughly 40 minutes per model variant on MPS for
the syntx harness, as a rough order of magnitude)::

    PYTHONPATH=. python tools/run_benchmark_mindboggle_probe.py \\
        --pair-idx $(seq 0 89) --device mps
"""

import argparse
import json
import time
from pathlib import Path

from antstorch.benchmark import (
    DEFAULT_PAIRS_CSV,
    check_mindboggle_data,
    evaluate_mindboggle_pair,
)

DEFAULT_PROBE_PAIRS = (0, 1, 2, 45, 67, 82)
DEFAULT_MODELS = ("gaussian_syn", "sobolev_syn", "dsti_syn", "bspline_syn", "bspline_svf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Mindboggle data root (contains e.g. OASIS-TRT-20_volumes/). "
        "If omitted, resolved from $ANTSTORCH_MINDBOGGLE_DATA_DIR (or $SYNTX_DATA_DIR as a "
        "fallback), then ~/data/mindboggle/volumes.",
    )
    parser.add_argument("--pairs-csv", default=DEFAULT_PAIRS_CSV, help="Pairs CSV (default: the 90-pair definition bundled with antstorch.benchmark)")
    parser.add_argument("--pair-idx", type=int, nargs="+", default=list(DEFAULT_PROBE_PAIRS), help="Pair indices to evaluate")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS) + ["svf"], help="Model variants to run per pair")
    parser.add_argument("--device", default=None, help="PyTorch device: cpu, cuda, or mps (default: auto-detected)")
    parser.add_argument("--canonical-affine-dir", default="results/canonical_affines", help="Per-pair canonical affine cache, shared across models")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_probe_output"))
    parser.add_argument("--reg-iterations", type=int, nargs="+", default=None, help="Override the default iteration schedule for every model")
    parser.add_argument("--no-n4", action="store_false", dest="use_n4", help="Skip ANTsTorch's own N4 bias-field correction/caching")
    parser.add_argument("--check-only", action="store_true", help="Only run check_mindboggle_data() and print the report, then exit")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(use_n4=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    is_valid, report = check_mindboggle_data(pairs_csv=args.pairs_csv, data_dir=args.data_dir, verbose=True)
    print(f"[check_mindboggle_data] data_dir={report['data_dir']}")
    print(f"[check_mindboggle_data] pairs_csv={report['pairs_csv_path']} ({report['total_pairs_in_csv']} pairs)")
    print(f"[check_mindboggle_data] available_pairs={report['available_pairs']}/{report['total_pairs_in_csv']}")
    if not is_valid:
        print(f"[check_mindboggle_data] {len(report['missing_files'])} missing file(s) -- see setup instructions above.")
        if not args.check_only:
            raise SystemExit(1)
    if args.check_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t_start = time.time()

    for pair_idx in args.pair_idx:
        for model in args.models:
            label = f"pair={pair_idx} model={model}"
            kwargs = {}
            if args.reg_iterations is not None:
                kwargs["reg_iterations"] = args.reg_iterations
            try:
                t0 = time.time()
                rec = evaluate_mindboggle_pair(
                    pair_idx=pair_idx,
                    model=model,
                    device=args.device,
                    pairs_csv=args.pairs_csv,
                    data_dir=args.data_dir,
                    canonical_affine_dir=args.canonical_affine_dir,
                    verbose=args.verbose,
                    use_n4=args.use_n4,
                    **kwargs,
                )
                elapsed = time.time() - t0
                print(
                    f"[{label}] SUCCESS dice_sym={rec['dice_sym']:.4f} "
                    f"affine_dice_sym={rec['affine_dice_sym']:.4f} "
                    f"folding_pct={rec['folding_pct']:.4f} runtime={elapsed:.1f}s"
                )
                records.append(rec)
            except Exception as e:
                print(f"[{label}] FAILED: {e}")
                records.append({"pair_idx": pair_idx, "model_type": model, "status": "ERROR", "error": str(e)})

    total_elapsed = time.time() - t_start
    results_path = args.output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)

    print(f"\n{len(records)} evaluations in {total_elapsed:.1f}s. Results written to: {results_path.resolve()}")

    print("\n=== Summary (mean over successful pairs, per model) ===")
    for model in args.models:
        model_records = [r for r in records if r.get("model_type") == model and r.get("status") == "SUCCESS"]
        if not model_records:
            print(f"  {model}: no successful evaluations")
            continue
        mean_dice = sum(r["dice_sym"] for r in model_records) / len(model_records)
        mean_folding = sum(r["folding_pct"] for r in model_records) / len(model_records)
        mean_runtime = sum(r["runtime_seconds"] for r in model_records) / len(model_records)
        print(
            f"  {model}: n={len(model_records)} mean_dice_sym={mean_dice:.4f} "
            f"mean_folding_pct={mean_folding:.4f} mean_runtime={mean_runtime:.1f}s"
        )


if __name__ == "__main__":
    main()
