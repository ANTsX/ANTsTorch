
#!/usr/bin/env python3
# (see printed usage when running with --help)
import argparse, csv, sys, math
from pathlib import Path
from collections import defaultdict

def _f(x, nd=4, default=""):
    try: return f"{float(x):.{nd}f}"
    except Exception: return default

def load_metrics_csv(path: Path):
    """Load impute_metrics.csv robustly.
    Handles cases where:
      * Some rows have more columns than the first header (csv.DictReader then inserts an entry under key None).
      * Extra header rows are accidentally written as data rows.
      * Whitespace / missing fields / 'nan' strings are present.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw is None:
                continue
            # Drop any spillover columns that land under the None key
            # (happens when later rows have more columns than the first header line).
            spill = raw.pop(None, None)
            # Normalize keys/values and guard against None keys.
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                ks = k.strip()
                if not ks:
                    continue
                if isinstance(v, str):
                    v = v.strip()
                row[ks] = v
            # Skip duplicate header rows that sometimes get written into the file.
            if row.get("observed", "").lower() == "observed" and row.get("missing", "").lower() == "missing":
                continue

            # Coerce types (be forgiving).
            try:
                row["N"] = int(row.get("N", 0) or 0)
            except Exception:
                row["N"] = 0
            for k in ("PSNR_mean","PSNR_std","MSE_mean","MSE_std",
                      "SSIM_mean","SSIM_std","zNLL_mean","zNLL_std"):
                try:
                    row[k] = float(row.get(k, "nan"))
                except Exception:
                    row[k] = float("nan")
            rows.append(row)
    return rows

def categorize_perm(obs: str, mis: str):
    n_obs = 1 + obs.count("+") if obs else 0
    n_mis = 1 + mis.count("+") if mis else 0
    if n_obs == 2 and n_mis == 1: return "one_missing"
    if n_obs == 1 and n_mis == 2: return "two_missing"
    return "other"

def avg(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals)/len(vals) if vals else float("nan")

def summarize_run(out_dir: Path):
    path = out_dir / "impute_metrics.csv"
    rows = load_metrics_csv(path)
    agg = defaultdict(list)
    for row in rows:
        cat = categorize_perm(row.get("observed", ""), row.get("missing", ""))
        agg[cat].append(row); agg["all"].append(row)
    def collect(key, bucket):
        return [r[key] for r in agg[bucket] if key in r]
    return {
        "run": out_dir.parent.name if out_dir.name.startswith("eval_") else out_dir.name,
        "path": str(out_dir),
        "PSNR_all": avg(collect("PSNR_mean", "all")),
        "PSNR_one_missing": avg(collect("PSNR_mean", "one_missing")),
        "PSNR_two_missing": avg(collect("PSNR_mean", "two_missing")),
        "SSIM_all": avg(collect("SSIM_mean", "all")),
        "SSIM_one_missing": avg(collect("SSIM_mean", "one_missing")),
        "SSIM_two_missing": avg(collect("SSIM_mean", "two_missing")),
        "MSE_all": avg(collect("MSE_mean", "all")),
        "MSE_one_missing": avg(collect("MSE_mean", "one_missing")),
        "MSE_two_missing": avg(collect("MSE_mean", "two_missing")),
        "zNLL_all": avg(collect("zNLL_mean", "all")),
        "zNLL_one_missing": avg(collect("zNLL_mean", "one_missing")),
        "zNLL_two_missing": avg(collect("zNLL_mean", "two_missing")),
    }

def main():
    ap = argparse.ArgumentParser("Summarize eval_gaussian metrics across runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-subdir", type=str, default="eval_gauss")
    ap.add_argument("--rank-by", type=str, default="PSNR", choices=["PSNR","SSIM","zNLL","composite"])
    ap.add_argument("--w-psnr", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--w-znll", type=float, default=1.0)
    args = ap.parse_args()

    # Discover run directories that contain the given out-subdir/impute_metrics.csv.
    run_dirs = []
    for p in args.runs:
        P = Path(p)
        if P.is_dir():
            # If the given path itself has the subdir file, use it.
            if (P / args.out_subdir / "impute_metrics.csv").exists():
                run_dirs.append(P / args.out_subdir)
            else:
                # Otherwise, search immediate children.
                for d in sorted(P.iterdir()):
                    cand = d / args.out_subdir / "impute_metrics.csv"
                    if cand.exists():
                        run_dirs.append(d / args.out_subdir)

    if not run_dirs:
        print("No run directories with ", args.out_subdir, "/impute_metrics.csv found.", file=sys.stderr); sys.exit(2)

    table = [summarize_run(d) for d in run_dirs]

    def rank_key(rec):
        # Higher is better for PSNR / SSIM, so return the value directly and sort with reverse=True
        if args.rank_by == "PSNR":
            v = rec["PSNR_all"]
            return (v if not math.isnan(v) else -1e9, )
        if args.rank_by == "SSIM":
            v = rec["SSIM_all"]
            return (v if not math.isnan(v) else -1e9, )
        # Lower is better for zNLL, so sort ascending via the key (reverse=False handled below)
        if args.rank_by == "zNLL":
            v = rec["zNLL_all"]
            return (v if not math.isnan(v) else 1e9, )
        # Composite: min–max normalize across runs; larger score is better
        ps = [r["PSNR_all"] for r in table if not math.isnan(r["PSNR_all"])]
        ss = [r["SSIM_all"] for r in table if not math.isnan(r["SSIM_all"])]
        nl = [r["zNLL_all"] for r in table if not math.isnan(r["zNLL_all"])]
        def mm(v, lo, hi):
            if math.isnan(v) or hi <= lo: return 0.0
            return (v - lo) / (hi - lo)
        pmin,pmax = (min(ps), max(ps)) if ps else (0.0,1.0)
        smin,smax = (min(ss), max(ss)) if ss else (0.0,1.0)
        nmin,nmax = (min(nl), max(nl)) if nl else (0.0,1.0)
        p = mm(rec["PSNR_all"], pmin, pmax)
        s = mm(rec["SSIM_all"], smin, smax)
        n = mm(rec["zNLL_all"], nmin, nmax)
        score = args.w_psnr * p + args.w_ssim * s + args.w_znll * (1.0 - n)  # lower zNLL => higher score
        return (score, )

    reverse = args.rank_by in ("PSNR","SSIM","composite")
    table.sort(key=rank_key, reverse=reverse)

    out = Path("alignment_summary.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank","run",
            "PSNR_all","PSNR_one_missing","PSNR_two_missing",
            "SSIM_all","SSIM_one_missing","SSIM_two_missing",
            "MSE_all","MSE_one_missing","MSE_two_missing",
            "zNLL_all","zNLL_one_missing","zNLL_two_missing",
            "path"
        ])
        for i, r in enumerate(table, 1):
            w.writerow([
                i, r["run"],
                _f(r["PSNR_all"]), _f(r["PSNR_one_missing"]), _f(r["PSNR_two_missing"]),
                _f(r["SSIM_all"]), _f(r["SSIM_one_missing"]), _f(r["SSIM_two_missing"]),
                _f(r["MSE_all"], nd=6), _f(r["MSE_one_missing"], nd=6), _f(r["MSE_two_missing"], nd=6),
                _f(r["zNLL_all"], nd=6), _f(r["zNLL_one_missing"], nd=6), _f(r["zNLL_two_missing"], nd=6),
                r["path"]
            ])

    print(f"=== Alignment strategy ranking (by {args.rank_by}) ===")
    for i, r in enumerate(table, 1):
        print(f"{i:>2}. {r['run']}: "
              f"PSNR={_f(r['PSNR_all'])}  SSIM={_f(r['SSIM_all'])}  zNLL={_f(r['zNLL_all'], nd=3)}  "
              f"(1->2 PSNR={_f(r['PSNR_two_missing'])}, 2->1 PSNR={_f(r['PSNR_one_missing'])})")
    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
