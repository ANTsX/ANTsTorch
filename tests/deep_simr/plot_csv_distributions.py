#!/usr/bin/env python3
"""
Plot distributions for each numeric column in one or more CSV files, using separate subplots.

Robust to extreme values with optional clipping/winsorizing and log/symlog scaling.

Features:
  - Automatically selects numeric columns.
  - Paginates across multiple figures if there are many columns.
  - Options for bins, KDE overlay (if SciPy available), NaN handling, standardization, and axis sharing.
  - Saves one or more figures per CSV into an output directory.
  - Optional Gaussianity tests:
    * Univariate (per column): D'Agostino & Pearson (default) or Shapiro–Wilk) -> p-value in title.
    * Multivariate (Mardia skewness/kurtosis) -> printed to stdout.
  - Robustness for heavy tails / extreme values:
    * --clip-quantiles qlow qhigh: drop values outside [qlow, qhigh] percentiles.
    * --winsorize-quantiles qlow qhigh: cap values to [qlow, qhigh] percentiles.
    * --xlim-quantiles qlow qhigh: set plot/view range without altering data; tails still counted in outer bins.
    * --logx / --symlogx: log-scale x (for positive) or symmetric log (for mixed-sign) axes.
    * KDE grid limited to quantile window to prevent huge ranges; optional subsampling for KDE.

Usage:
  python plot_csv_distributions.py data.csv --kde --test-gaussianity --clip-quantiles 0.5 99.5 --ncols 4

Dependencies:
  - pandas, matplotlib, numpy
  - Optional: scipy (for KDE & Gaussianity tests). If not present, those parts are skipped gracefully.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import List, Iterable, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde, normaltest, shapiro, chi2, norm
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    gaussian_kde = None
    normaltest = None
    shapiro = None
    chi2 = None
    norm = None

def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def sanitize_file_stem(p: Path) -> str:
    return p.stem.replace(' ', '_')

def pct_window(x: np.ndarray, q: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if q is None:
        lo, hi = np.nanmin(x), np.nanmax(x)
    else:
        ql, qh = q
        lo, hi = np.nanpercentile(x, ql), np.nanpercentile(x, qh)
        if not np.isfinite(lo): lo = np.nanmin(x)
        if not np.isfinite(hi): hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)

def apply_clip(x: np.ndarray, q: Optional[Tuple[float,float]], winsorize: bool=False) -> np.ndarray:
    if q is None:
        return x
    lo, hi = np.nanpercentile(x, q[0]), np.nanpercentile(x, q[1])
    if winsorize:
        x = np.clip(x, lo, hi)
        return x
    else:
        mask = (x >= lo) & (x <= hi)
        return x[mask]

def mardia_tests(X: np.ndarray) -> dict:
    n, p = X.shape
    if n <= p:
        raise ValueError(f"Need n > p for Mardia tests; got n={n}, p={p}")
    Xm = X - X.mean(axis=0, keepdims=True)
    S = np.cov(Xm, rowvar=False, bias=False)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(p)
        S_inv = np.linalg.inv(S + ridge)
    A = Xm @ S_inv @ Xm.T
    b1p = (1.0 / (n**2)) * np.sum(A**3)
    skew_df = int(p * (p + 1) * (p + 2) / 6.0)
    skew_stat = n * b1p / 6.0
    skew_p = chi2.sf(skew_stat, df=skew_df) if (_HAVE_SCIPY and chi2 is not None) else np.nan
    di = np.diag(A)
    b2p = np.mean(di**2)
    expected = p * (p + 2.0)
    var_b2 = (8.0 * p * (p + 2.0)) / n
    kurt_z = (b2p - expected) / np.sqrt(var_b2)
    kurt_p = 2.0 * norm.sf(abs(kurt_z)) if (_HAVE_SCIPY and norm is not None) else np.nan
    return {
        'b1p': b1p,
        'skew_chi2': skew_stat,
        'skew_df': skew_df,
        'skew_p': skew_p,
        'b2p': b2p,
        'kurt_z': kurt_z,
        'kurt_p': kurt_p,
    }

def plot_csv(
    csv_path: Path,
    out_dir: Path,
    bins: str | int = "auto",
    ncols: int = 3,
    max_subplots_per_fig: int = 16,
    sharex: bool = False,
    sharey: bool = False,
    dropna: bool = True,
    kde: bool = False,
    standardize: bool = False,
    dpi: int = 150,
    figsize_w: float = 4.0,
    figsize_h: float = 3.0,
    tight_layout: bool = True,
    file_tag: Optional[str] = None,
    test_gaussianity: bool = False,
    uni_test: str = "dagostino",
    multi_test: str = "mardia",
    clip_quantiles: Optional[Tuple[float,float]] = None,
    winsorize_quantiles: Optional[Tuple[float,float]] = None,
    xlim_quantiles: Optional[Tuple[float,float]] = None,
    logx: bool = False,
    symlogx: bool = False,
    kde_quantiles: Optional[Tuple[float,float]] = (1.0, 99.0),
    max_kde_n: int = 20000,
    density: bool = True,
) -> list[Path]:
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print(f"[WARN] No numeric columns in {csv_path}. Skipping.")
        return []

    data = {}
    for c in num_cols:
        x = df[c].to_numpy(dtype=float)
        if dropna:
            x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        # Robust clipping/winsorizing
        if clip_quantiles is not None:
            x = apply_clip(x, clip_quantiles, winsorize=False)
        if winsorize_quantiles is not None:
            x = apply_clip(x, winsorize_quantiles, winsorize=True)
        if x.size == 0:
            continue
        if standardize:
            mu = np.nanmean(x)
            sigma = np.nanstd(x)
            if sigma > 0 and np.isfinite(sigma):
                x = (x - mu) / sigma
        data[c] = x

    kept_cols = list(data.keys())
    if not kept_cols:
        print(f"[WARN] All numeric columns in {csv_path} were empty after filtering. Skipping.")
        return []

    # Multivariate test (complete cases)
    if test_gaussianity and multi_test == "mardia":
        X = df[kept_cols].to_numpy(dtype=float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        if Xc.shape[0] <= Xc.shape[1]:
            print(f"[WARN] Skipping multivariate test for {csv_path.name}: need n > p (n={Xc.shape[0]}, p={Xc.shape[1]})")
        else:
            try:
                stats = mardia_tests(Xc)
                print(f"[MARDIA] {csv_path.name}: skew χ²={stats['skew_chi2']:.3f} (df={int(stats['skew_df'])}), p={stats['skew_p'] if not np.isnan(stats['skew_p']) else 'NA'}; "
                      f"kurt z={stats['kurt_z']:.3f}, p={stats['kurt_p'] if not np.isnan(stats['kurt_p']) else 'NA'}")
            except Exception as e:
                print(f"[WARN] Multivariate Gaussianity test failed for {csv_path.name}: {e}")

    n_per_fig = min(max_subplots_per_fig, ncols * math.ceil(max_subplots_per_fig / ncols))
    saved_paths: list[Path] = []
    page_idx = 1
    for col_chunk in chunked(kept_cols, n_per_fig):
        nplots = len(col_chunk)
        nrows = math.ceil(nplots / ncols)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex=sharex, sharey=sharey,
            figsize=(figsize_w * ncols, figsize_h * nrows),
            squeeze=False
        )

        for i, col in enumerate(col_chunk):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            x = data[col]

            # Determine plotting range to avoid massive spans
            hist_range = pct_window(x, xlim_quantiles)

            # Histogram
            ax.hist(x, bins=bins, range=hist_range, density=density, alpha=0.7, edgecolor='none')

            # KDE overlay (optional & robust range)
            if kde and _HAVE_SCIPY and gaussian_kde is not None and x.size > 1:
                try:
                    qk = kde_quantiles if kde_quantiles is not None else (1.0, 99.0)
                    lo_kde, hi_kde = pct_window(x, qk)
                    xs = np.linspace(lo_kde, hi_kde, 400)
                    x_kde = x
                    if x_kde.size > max_kde_n:
                        rs = np.random.default_rng(123)
                        x_kde = rs.choice(x_kde, size=max_kde_n, replace=False)
                    kde_est = gaussian_kde(x_kde)
                    ax.plot(xs, kde_est(xs), linewidth=1.5)
                except Exception as e:
                    ax.text(0.5, 0.9, f"KDE failed: {e}", transform=ax.transAxes, ha='center', va='top', fontsize=8)

            # Title with optional univariate p-value
            title = col
            if test_gaussianity and _HAVE_SCIPY:
                pval = None
                try:
                    if uni_test.lower().startswith('d'):
                        stat, pval = normaltest(x, nan_policy='omit')
                    elif uni_test.lower().startswith('s'):
                        stat, pval = shapiro(x)
                except Exception:
                    pval = None
                if pval is not None:
                    title = f"{col} (p={pval:.3g})"
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

            # Axis scaling options
            if symlogx:
                ax.set_xscale('symlog', linthresh=1e-6)
            elif logx:
                xmin = max(hist_range[0], np.nextafter(0, 1))
                ax.set_xlim(left=xmin, right=hist_range[1])
                try:
                    ax.set_xscale('log')
                except ValueError:
                    pass

        # Hide any unused axes
        total_axes = nrows * ncols
        for j in range(nplots, total_axes):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')

        stem = sanitize_file_stem(csv_path)
        tag = f"_{file_tag}" if file_tag else ""
        suffix = f"_page{page_idx}" if len(kept_cols) > n_per_fig else ""
        fig.suptitle(f"Distributions: {stem}{tag}", fontsize=12)
        if tight_layout:
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}{tag}{suffix}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(out_path)
        page_idx += 1

    print(f"[INFO] Saved {len(saved_paths)} figure(s) for {csv_path}")
    return saved_paths

def parse_args():
    p = argparse.ArgumentParser(description="Plot distributions for numeric columns in CSV files.")
    p.add_argument("csvs", nargs='+', type=Path, help="One or more CSV files")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("figs"), help="Output directory for figures")
    p.add_argument("--bins", default="auto", help="Histogram bins ('auto', 'fd', integer, etc.)")
    p.add_argument("--ncols", type=int, default=3, help="Number of subplot columns per figure")
    p.add_argument("--max-subplots-per-fig", type=int, default=16, help="Maximum subplots per figure (pagination)")
    p.add_argument("--sharex", action="store_true", help="Share x-axis across subplots")
    p.add_argument("--sharey", action="store_true", help="Share y-axis across subplots")
    p.add_argument("--keep-na", action="store_true", help="Keep NaNs/infs (by default they are dropped)")
    p.add_argument("--kde", action="store_true", help="Overlay KDE curve (requires SciPy)")
    p.add_argument("--standardize", action="store_true", help="Z-score standardize each column before plotting")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(4.0, 3.0),
                   help="Size (inches) of each subplot (width height)")
    p.add_argument("--tag", type=str, default=None, help="Optional tag to include in output filename")
    p.add_argument("--test-gaussianity", action="store_true",
                   help="Run univariate (subplot titles) and multivariate (stdout) Gaussianity tests. Requires SciPy for p-values.")
    p.add_argument("--uni-test", choices=["dagostino", "shapiro"], default="dagostino",
                   help="Univariate normality test to use when --test-gaussianity is set.")
    p.add_argument("--multi-test", choices=["mardia"], default="mardia",
                   help="Multivariate normality test to run when --test-gaussianity is set.")
    # Robustness flags
    p.add_argument("--clip-quantiles", type=float, nargs=2, metavar=("QLOW","QHIGH"),
                   help="Drop values outside these percentiles (e.g., 0.5 99.5)")
    p.add_argument("--winsorize-quantiles", type=float, nargs=2, metavar=("QLOW","QHIGH"),
                   help="Cap values to these percentiles (e.g., 1 99)")
    p.add_argument("--xlim-quantiles", type=float, nargs=2, metavar=("QLOW","QHIGH"),
                   help="Set histogram/view range by percentiles without altering data (e.g., 0.5 99.5)")
    p.add_argument("--logx", action="store_true", help="Use log scale on x-axis (positive data only)")
    p.add_argument("--symlogx", action="store_true", help="Use symmetric log scale on x-axis (for mixed-sign data)")
    p.add_argument("--kde-quantiles", type=float, nargs=2, metavar=("QLOW","QHIGH"),
                   default=(1.0, 99.0), help="Percentile window to compute KDE grid (default 1 99)")
    p.add_argument("--max-kde-n", type=int, default=20000, help="Subsample size cap for KDE to improve speed on large columns")
    p.add_argument("--counts", action="store_true", help="Plot raw counts instead of probability density (i.e., set density=False)")
    return p.parse_args()

def main():
    args = parse_args()
    bins = args.bins
    if isinstance(bins, str) and bins.isdigit():
        bins = int(bins)
    for csv in args.csvs:
        if not csv.exists():
            print(f"[ERROR] File not found: {csv}", file=sys.stderr)
            continue
        plot_csv(
            csv_path=csv,
            out_dir=args.out_dir,
            bins=bins,
            ncols=max(1, args.ncols),
            max_subplots_per_fig=max(1, args.max_subplots_per_fig),
            sharex=args.sharex,
            sharey=args.sharey,
            dropna=not args.keep_na,
            kde=args.kde,
            standardize=args.standardize,
            dpi=args.dpi,
            figsize_w=args.figsize[0],
            figsize_h=args.figsize[1],
            file_tag=args.tag,
            test_gaussianity=args.test_gaussianity,
            uni_test=args.uni_test,
            multi_test=args.multi_test,
            clip_quantiles=tuple(args.clip_quantiles) if args.clip_quantiles else None,
            winsorize_quantiles=tuple(args.winsorize_quantiles) if args.winsorize_quantiles else None,
            xlim_quantiles=tuple(args.xlim_quantiles) if args.xlim_quantiles else None,
            logx=args.logx,
            symlogx=args.symlogx,
            kde_quantiles=tuple(args.kde_quantiles) if args.kde_quantiles else None,
            max_kde_n=max(1000, args.max_kde_n),
            density=not args.counts,
        )

if __name__ == "__main__":
    main()
