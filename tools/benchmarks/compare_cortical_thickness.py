#!/usr/bin/env python3
"""Compare ANTs KellyKapowski with ANTsTorch's tensor DiReCT engine.

Deep Atropos is run once.  Its segmentation and tissue-probability images
are then shared by both implementations, so the reported difference isolates
the cortical-thickness engines rather than the upstream segmentation.

By default the benchmark uses ANTsTorch's ``S_template3`` example::

    python tools/benchmarks/compare_cortical_thickness.py

Run a shorter diagnostic on MPS and write results elsewhere::

    python tools/benchmarks/compare_cortical_thickness.py \
        --device mps --iterations 5 --output-dir /tmp/direct_comparison
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import ants
import numpy as np
import torch

import antstorch


def _finite_float(value: float) -> Optional[float]:
    """Return a JSON-safe float, or ``None`` for a non-finite value."""
    value = float(value)
    return value if np.isfinite(value) else None


def compare_arrays(
    ants_array: np.ndarray,
    antstorch_array: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Summarize two continuous maps over a common boolean mask."""
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(ants_array)
        & np.isfinite(antstorch_array)
    )
    if not np.any(valid):
        raise ValueError("The comparison mask contains no finite voxels")

    reference = np.asarray(ants_array, dtype=np.float64)[valid]
    candidate = np.asarray(antstorch_array, dtype=np.float64)[valid]
    difference = candidate - reference
    absolute_difference = np.abs(difference)

    correlation = None
    if reference.size > 1 and reference.std() > 0 and candidate.std() > 0:
        correlation = _finite_float(np.corrcoef(reference, candidate)[0, 1])

    return {
        "voxel_count": int(reference.size),
        "ants": {
            "mean_mm": _finite_float(reference.mean()),
            "std_mm": _finite_float(reference.std()),
            "median_mm": _finite_float(np.median(reference)),
            "maximum_mm": _finite_float(reference.max()),
            "nonzero_fraction": _finite_float(np.mean(reference > 0)),
        },
        "antstorch": {
            "mean_mm": _finite_float(candidate.mean()),
            "std_mm": _finite_float(candidate.std()),
            "median_mm": _finite_float(np.median(candidate)),
            "maximum_mm": _finite_float(candidate.max()),
            "nonzero_fraction": _finite_float(np.mean(candidate > 0)),
        },
        "difference_antstorch_minus_ants": {
            "mean_mm": _finite_float(difference.mean()),
            "mae_mm": _finite_float(absolute_difference.mean()),
            "median_absolute_error_mm": _finite_float(np.median(absolute_difference)),
            "p95_absolute_error_mm": _finite_float(np.percentile(absolute_difference, 95)),
            "maximum_absolute_error_mm": _finite_float(absolute_difference.max()),
            "rmse_mm": _finite_float(np.sqrt(np.mean(difference**2))),
            "pearson_correlation": correlation,
        },
    }


def _image_from_array(array: np.ndarray, reference: ants.ANTsImage) -> ants.ANTsImage:
    return ants.from_numpy(
        np.asarray(array, dtype=np.float32),
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def _synchronize(device: torch.device) -> None:
    """Wait for accelerator work before recording a timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _format_metric(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _write_summary(path: Path, results: dict) -> None:
    gm = results["metrics"]["gray_matter"]
    error = gm["difference_antstorch_minus_ants"]
    timing = results["timing_seconds"]
    lines = [
        "# Cortical-thickness comparison",
        "",
        f"Input: `{results['input']}`",
        "",
        "Both engines used the same Deep Atropos segmentation and tissue probabilities.",
        "",
        "| Measurement | Value |",
        "|---|---:|",
        f"| Deep Atropos | {timing['deep_atropos']:.3f} s |",
        f"| ANTs KellyKapowski | {timing['ants_kelly_kapowski']:.3f} s |",
        f"| ANTsTorch DiReCT | {timing['antstorch_direct']:.3f} s |",
        f"| ANTsTorch speed ratio | {timing['antstorch_speedup_vs_ants']:.3f}x |",
        f"| GM voxels compared | {gm['voxel_count']} |",
        f"| GM MAE | {_format_metric(error['mae_mm'])} mm |",
        f"| GM RMSE | {_format_metric(error['rmse_mm'])} mm |",
        f"| GM 95th percentile absolute error | {_format_metric(error['p95_absolute_error_mm'])} mm |",
        f"| GM Pearson correlation | {_format_metric(error['pearson_correlation'])} |",
        "",
        "The speed ratio is `ANTs time / ANTsTorch time`; values above 1 favor ANTsTorch.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        help="Optional T1 image; the default is antstorch data 'S_template3'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cortical_thickness_comparison"),
        help="Directory for images and reports (default: cortical_thickness_comparison).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="ANTsTorch device, e.g. cpu, mps, or cuda (default: configured device).",
    )
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--gradient-step", type=float, default=0.025)
    parser.add_argument("--velocity-smoothing-variance", type=float, default=1.5)
    parser.add_argument(
        "--optimizer", choices=("direct", "reg_adam"), default="direct"
    )
    parser.add_argument(
        "--regularizer",
        choices=("gaussian", "sobolev", "dsti", "none"),
        default="gaussian",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.gradient_step <= 0:
        parser.error("--gradient-step must be positive")
    if args.velocity_smoothing_variance <= 0:
        parser.error("--velocity-smoothing-variance must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else antstorch.get_default_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available")

    input_path = args.image or antstorch.get_antstorch_data("S_template3")
    t1 = ants.image_read(input_path).clone("float")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print("Running Deep Atropos once for the shared inputs...")
    _synchronize(device)
    start = time.perf_counter()
    atropos = antstorch.deep_atropos(
        [t1, None, None],
        do_preprocessing=True,
        device=device,
        verbose=args.verbose,
    )
    _synchronize(device)
    atropos_seconds = time.perf_counter() - start

    segmentation = ants.image_clone(atropos["segmentation_image"])
    segmentation[segmentation == 4] = 3
    gray_matter = atropos["probability_images"][2]
    white_matter = atropos["probability_images"][3] + atropos["probability_images"][4]

    print("Running ANTs KellyKapowski...")
    start = time.perf_counter()
    ants_thickness = ants.kelly_kapowski(
        s=segmentation,
        g=gray_matter,
        w=white_matter,
        its=args.iterations,
        r=args.gradient_step,
        m=args.velocity_smoothing_variance,
        x=0,
        verbose=int(args.verbose),
    )
    ants_seconds = time.perf_counter() - start

    print(f"Running ANTsTorch DiReCT on {device}...")
    _synchronize(device)
    start = time.perf_counter()
    antstorch_thickness = antstorch.direct.kelly_kapowski(
        segmentation,
        gray_matter,
        white_matter,
        iterations=args.iterations,
        gradient_step=args.gradient_step,
        velocity_smoothing_variance=args.velocity_smoothing_variance,
        optimizer=args.optimizer,
        regularizer=args.regularizer,
        device=device,
        verbose=args.verbose,
    )
    _synchronize(device)
    antstorch_seconds = time.perf_counter() - start

    ants_array = ants_thickness.numpy().astype(np.float64, copy=False)
    antstorch_array = antstorch_thickness.numpy().astype(np.float64, copy=False)
    difference_array = antstorch_array - ants_array
    segmentation_array = segmentation.numpy()
    gray_matter_mask = segmentation_array == 2
    foreground_mask = segmentation_array > 0

    outputs = {
        "t1": args.output_dir / "t1.nii.gz",
        "segmentation": args.output_dir / "direct_segmentation.nii.gz",
        "gray_matter_probability": args.output_dir / "gray_matter_probability.nii.gz",
        "white_matter_probability": args.output_dir / "white_matter_probability.nii.gz",
        "ants_thickness": args.output_dir / "ants_thickness.nii.gz",
        "antstorch_thickness": args.output_dir / "antstorch_thickness.nii.gz",
        "difference": args.output_dir / "antstorch_minus_ants.nii.gz",
        "absolute_difference": args.output_dir / "absolute_difference.nii.gz",
    }
    images = {
        "t1": t1,
        "segmentation": segmentation,
        "gray_matter_probability": gray_matter,
        "white_matter_probability": white_matter,
        "ants_thickness": ants_thickness,
        "antstorch_thickness": antstorch_thickness,
        "difference": _image_from_array(difference_array, segmentation),
        "absolute_difference": _image_from_array(np.abs(difference_array), segmentation),
    }
    for name, image in images.items():
        ants.image_write(image, str(outputs[name]))

    speedup = ants_seconds / antstorch_seconds if antstorch_seconds > 0 else None
    results = {
        "input": str(input_path),
        "device": str(device),
        "parameters": {
            "iterations": args.iterations,
            "gradient_step": args.gradient_step,
            "velocity_smoothing_variance": args.velocity_smoothing_variance,
            "antstorch_optimizer": args.optimizer,
            "antstorch_regularizer": args.regularizer,
        },
        "timing_seconds": {
            "deep_atropos": atropos_seconds,
            "ants_kelly_kapowski": ants_seconds,
            "antstorch_direct": antstorch_seconds,
            "antstorch_speedup_vs_ants": speedup,
        },
        "metrics": {
            "gray_matter": compare_arrays(
                ants_array, antstorch_array, gray_matter_mask
            ),
            "segmented_foreground": compare_arrays(
                ants_array, antstorch_array, foreground_mask
            ),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    json_path = args.output_dir / "comparison.json"
    summary_path = args.output_dir / "summary.md"
    json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(summary_path, results)

    gm_error = results["metrics"]["gray_matter"][
        "difference_antstorch_minus_ants"
    ]
    print(f"Deep Atropos:        {atropos_seconds:.3f} s (shared; excluded from engine timing)")
    print(f"ANTs KellyKapowski:  {ants_seconds:.3f} s")
    print(f"ANTsTorch DiReCT:    {antstorch_seconds:.3f} s")
    print(f"ANTsTorch speed ratio: {_format_metric(speedup)}x")
    print(f"GM MAE:              {_format_metric(gm_error['mae_mm'])} mm")
    print(f"GM RMSE:             {_format_metric(gm_error['rmse_mm'])} mm")
    print(f"GM Pearson r:        {_format_metric(gm_error['pearson_correlation'])}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
