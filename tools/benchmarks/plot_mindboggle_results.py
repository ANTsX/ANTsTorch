#!/usr/bin/env python
"""Violin plots of the Mindboggle benchmark, one violin per arm and model, over the pairs (models on the x axis).

Reads the per-pair records of a run directory (the --out-dir of compare_syntx_antstorch_mindboggle.py):

    comparison_results.json              syntx / ANTsTorch / FireANTs 'matched' arms of the main run
    fireants_<preset>_results.json       separate --fireants-only runs (optional, any number)
    ants_<model>_results.json            separate --ants-only runs (optional, any number)

and writes three PNG files:

    figures/mindboggle_dice.png       distribution over the pairs of Dice fixed
    figures/mindboggle_time.png       distribution over the pairs of the deformable-only time (log scale)
    figures/mindboggle_jacobian.png   two panels: folding % (voxels with det <= 0, log scale, zeros on the floor) and sigma(log J)

Dice fixed is the Dice in the fixed space (the forward warp only), the one measure available for every arm. In each violin the white
dot is the median and the thin bar the interquartile range. The `*_regadam` dense models have no syntx violin: syntx has no reg_adam
counterpart for them (its record is the one of the base model, see the README).

Usage:  python tools/benchmarks/plot_mindboggle_results.py --results-dir <run dir> [--out-dir tools/benchmarks/figures]
Needs numpy and matplotlib only. Re-run it after a new run (for instance the 90-pair run).
"""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Library colours: categorical slots 1-4 of the validated default palette, in fixed order (light surface). Identity is also carried
# by the legend, the x labels, the median labels and, for the CPU baseline, a hatch.
COLORS = {"syntx": "#2a78d6", "ANTsTorch": "#eb6834", "FireANTs": "#1baf7a", "ANTs C++": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
ORDER = ["gaussian", "gaussian_regadam", "sobolev", "sobolev_regadam", "dsti", "dsti_regadam", "bspline", "bspline_regadam",
         "greedy", "greedy_regadam", "bspline_svf", "gaussian_svf"]


def _load(path):
    with open(path) as f:
        return [r for r in json.load(f) if r.get("_status") == "SUCCESS"]


def collect(results_dir):
    """Ordered {x label: {library: [(pair, dice_fixed, seconds), ...]}}."""
    g = {}

    def add(label, lib, rec):
        row = {"pair": rec["pair_idx"], "dice": float(rec["dice_fixed"]), "time": float(rec["_deformable_seconds"]),
               "fold": float(rec.get("jac_folding_pct", float("nan"))), "logstd": float(rec.get("jac_log_std", float("nan")))}
        g.setdefault(label, {}).setdefault(lib, []).append(row)

    main = os.path.join(results_dir, "comparison_results.json")
    if not os.path.exists(main):
        raise SystemExit(f"{main} not found: give the --out-dir of a run of compare_syntx_antstorch_mindboggle.py")
    for r in _load(main):
        reg, lib = r["_regularizer"], r["_library"]
        if lib == "syntx" and reg.endswith("_regadam") and reg != "greedy_regadam":
            continue  # same syntx run as the base model
        if lib in ("syntx", "antstorch"):
            add(reg, "syntx" if lib == "syntx" else "ANTsTorch", r)
        elif lib == "fireants":
            add("FireANTs\nmatched", "FireANTs", r)
    for path in sorted(glob.glob(os.path.join(results_dir, "fireants_*_results.json"))):
        preset = os.path.basename(path)[len("fireants_"):-len("_results.json")].replace("_", "-")
        for r in _load(path):
            add(f"FireANTs\n{preset}", "FireANTs", r)
    for path in sorted(glob.glob(os.path.join(results_dir, "ants_*_results.json"))):
        for r in _load(path):
            add(r["_regularizer"], "ANTs C++", r)
    fa_order = ["FireANTs\nmatched", "FireANTs\ndefault", "FireANTs\nfine-tuned-syntx"]
    labels = [m for m in ORDER if m in g] + [m for m in fa_order if m in g]
    labels += [m for m in g if m not in labels and not m.startswith("FireANTs")] + [m for m in g if m.startswith("FireANTs") and m not in labels]
    # ANTs baselines go last
    labels = [m for m in labels if "ants_" not in m] + [m for m in labels if "ants_" in m]
    return {m: {lib: sorted(v, key=lambda r: r["pair"]) for lib, v in g[m].items()} for m in labels}


def panel(ax, g, field, ylabel, fmt, kind="linear", ylim=None, ref=None, from_zero=False):
    """One violin panel. kind: 'linear', 'log' (positive values) or 'logfloor' (values >= 0, zeros drawn on a floor of 1e-4)."""
    labels = list(g)
    n = len(labels)
    width = 0.38
    floor = 1e-4
    tf = {"linear": lambda v: v, "log": np.log10, "logfloor": lambda v: np.log10(np.maximum(v, floor))}[kind]
    tops, bottoms = [], []
    for i, lab in enumerate(labels):
        libs = [lib for lib in g[lab] if np.isfinite([r[field] for r in g[lab][lib]]).any()]
        offs = [0.0] if len(libs) == 1 else [(-0.5 + k) * (width + 0.04) for k in range(len(libs))]
        for lib, off in zip(libs, offs):
            v = np.array([r[field] for r in g[lab][lib]], dtype=float)
            v = v[np.isfinite(v)]
            data = tf(v)
            parts = ax.violinplot([data], positions=[i + off], widths=width * 1.15, showextrema=False, showmedians=False, bw_method=0.35)
            for b in parts["bodies"]:
                b.set_facecolor(COLORS[lib])
                b.set_edgecolor(SURFACE)
                b.set_linewidth(1.0)
                b.set_alpha(0.95)
                b.set_zorder(3)
                if lib == "ANTs C++":
                    b.set_hatch("////")
            q1, med, q3 = np.percentile(data, [25, 50, 75])
            ax.plot([i + off, i + off], [q1, q3], color=INK, linewidth=1.6, solid_capstyle="butt", zorder=4)
            ax.plot([i + off], [med], marker="o", markersize=4.2, color="white", markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
            tops.append(data.max())
            bottoms.append(data.min())
            pad = {"linear": 0.012 * (ylim[1] - ylim[0]) if ylim else 0.006, "log": 0.03, "logfloor": 0.12}[kind]
            ax.text(i + off, data.max() + pad, fmt.format(float(np.median(v))), ha="center", va="bottom", fontsize=6.3, color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    ax.set_xticks(range(n))
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if kind == "log":
        ticks = [1, 2, 5, 10, 20, 50, 100]
        lo, hi = min(min(bottoms), 0.0), max(tops) + 0.18
        ax.set_ylim(lo - 0.05, hi)
        keep = [t for t in ticks if lo - 0.05 <= np.log10(t) <= hi]
        ax.set_yticks([np.log10(t) for t in keep])
        ax.set_yticklabels([str(t) for t in keep])
    elif kind == "logfloor":
        ticks = [floor, 1e-3, 1e-2, 1e-1, 1, 10]
        hi = max(tops) + 0.45
        ax.set_ylim(np.log10(floor) - 0.35, hi)
        keep = [t for t in ticks if np.log10(t) <= hi]
        ax.set_yticks([np.log10(t) for t in keep])
        ax.set_yticklabels(["0" if t == floor else f"{t:g}" for t in keep])
    elif ylim:
        ax.set_ylim(*ylim)
    elif from_zero:
        ax.set_ylim(0, max(tops) * 1.12)
    if ref is not None:
        y = np.log10(ref[0]) if kind != "linear" else ref[0]
        ax.axhline(y, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.text(n - 0.35, y, ref[1], ha="right", va="bottom", fontsize=7, color=INK2)
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def legend(ax, g):
    present = [lib for lib in COLORS if any(lib in v for v in g.values())]
    handles = [Patch(facecolor=COLORS[lib], edgecolor=SURFACE, label=lib + (" (CPU)" if lib == "ANTs C++" else ""),
                     hatch="////" if lib == "ANTs C++" else None) for lib in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(handles), frameon=False, fontsize=8,
              labelcolor=INK2, handlelength=1.2, columnspacing=1.6)


def figure(g, panels, path, note=None, height=5.2):
    """panels: list of dicts for panel(); the x labels are drawn under the last one."""
    labels = list(g)
    n = len(labels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(9.0, 0.78 * n + 2), height), dpi=160, sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)
    for ax, kw in zip(axes, panels):
        panel(ax, g, **kw)
    legend(axes[0], g)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8, color=INK2)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="--out-dir of the benchmark run (comparison_results.json and the separate run files)")
    ap.add_argument("--out-dir", default=os.path.join(here, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    g = collect(a.results_dir)
    npairs = sorted({len(v) for libs in g.values() for v in libs.values()})
    n_txt = f"{npairs[0]}" if len(npairs) == 1 else f"{npairs[0]}-{npairs[-1]}"
    note = f"One violin per arm over the {n_txt} pairs; dot = median (value printed above), bar = interquartile range."
    figure(g, [dict(field="dice", ylabel="Dice fixed (distribution over the pairs)", fmt="{:.3f}")],
           os.path.join(a.out_dir, "mindboggle_dice.png"),
           note=note + " Dice in the fixed space: the forward warp only, available for every arm.")
    figure(g, [dict(field="time", ylabel="Deformable-only time, s (log scale)", fmt="{:.1f}", kind="log")],
           os.path.join(a.out_dir, "mindboggle_time.png"),
           note=note + " ANTs C++ (hatched) ran on the CPU: not comparable with the GPU arms.")
    figure(g, [dict(field="fold", ylabel="Folding, % of voxels with det J <= 0 (log scale)", fmt="{:.4f}", kind="logfloor",
                    ref=(0.01, "0.01 %")),
               dict(field="logstd", ylabel="σ(log J) (distribution over the pairs)", fmt="{:.2f}", from_zero=True)],
           os.path.join(a.out_dir, "mindboggle_jacobian.png"), height=8.2,
           note=note + " Jacobian of the forward warp, same finite-difference measure for every arm; zero folding is drawn on the floor of the "
                       "top panel (axis label 0). The unofficial syntx bspline arm has no Jacobian.")


if __name__ == "__main__":
    main()
