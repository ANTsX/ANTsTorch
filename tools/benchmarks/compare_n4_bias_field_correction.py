#!/usr/bin/env python3
"""Compare ANTs and differentiable ANTsTorch N4 bias correction.

Examples
--------
Use the bundled 2-D ``r16`` image::

    python tools/benchmarks/compare_n4_bias_field_correction.py

Use another 2-D or 3-D image and CUDA, if available::

    python tools/benchmarks/compare_n4_bias_field_correction.py image.nii.gz --device cuda

Write the output images to a dedicated directory::

    python tools/benchmarks/compare_n4_bias_field_correction.py --output-dir results/n4
"""

import argparse
import time
from pathlib import Path

import ants
import numpy as np
import torch

import antstorch


def normalized_bias_array(image: ants.ANTsImage) -> np.ndarray:
    """Remove N4's arbitrary global multiplicative bias-field scale."""
    array = image.numpy().astype(np.float64)
    return array / np.exp(np.log(np.clip(array, 1e-12, None)).mean())


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        help="Input image. When omitted, ants.get_ants_data('r16') is used.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--shrink-factor", type=int, default=4)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[50, 50, 50, 50],
        help="Iterations at each fitting level (default: 20 20)",
    )
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print iteration progress from both ANTs and ANTsTorch N4.",
    )
    parser.add_argument(
        "--stable-accumulation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control deterministic ANTsTorch reductions. The default uses "
            "stable accumulation on MPS and fast accumulation elsewhere; "
            "use --no-stable-accumulation to avoid slow MPS level setup."
        ),
    )
    parser.add_argument(
        "--mesh-size",
        type=int,
        nargs="+",
        default=None,
        help="B-spline mesh size in ITK x-y-z order (default: one span per axis)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output images; it is created if needed (default: current directory).",
    )
    parser.add_argument("--output-prefix", default="n4_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available")

    input_path = args.image or ants.get_ants_data("r16")
    t1 = ants.image_read(input_path).clone("float")
    mask = t1 * 0 + 1
    mesh_size = args.mesh_size or [4] * t1.dimension
    if len(mesh_size) != t1.dimension:
        raise ValueError(f"--mesh-size needs {t1.dimension} values for this image")
    convergence = {"iters": args.iterations, "tol": args.tolerance}
    stable_accumulation = args.stable_accumulation
    if stable_accumulation is None:
        stable_accumulation = device.type == "mps"

    start = time.perf_counter()
    if args.verbose:
        print("Running ANTs N4 corrected-image pass...")
    n4_ants = ants.n4_bias_field_correction(
        t1,
        mask=mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=mesh_size,
        verbose=args.verbose,
    )
    if args.verbose:
        print("Running ANTs N4 bias-field pass...")
    bias_ants = ants.n4_bias_field_correction(
        t1,
        mask=mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=mesh_size,
        return_bias_field=True,
        verbose=args.verbose,
    )
    ants_seconds = time.perf_counter() - start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    if args.verbose:
        print("Running ANTsTorch N4 corrected-image pass...")
    n4_torch = antstorch.n4_bias_field_correction(
        t1,
        mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=tuple(mesh_size),
        stable_accumulation=stable_accumulation,
        device=device,
        verbose=args.verbose,
    )
    if args.verbose:
        print("Running ANTsTorch N4 bias-field pass...")
    bias_torch = antstorch.n4_bias_field_correction(
        t1,
        mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=tuple(mesh_size),
        return_bias_field=True,
        stable_accumulation=stable_accumulation,
        device=device,
        verbose=args.verbose,
    )
    synchronize(device)
    torch_seconds = time.perf_counter() - start

    corrected_ants_array = n4_ants.numpy().astype(np.float64)
    corrected_torch_array = n4_torch.numpy().astype(np.float64)
    corrected_difference = corrected_torch_array - corrected_ants_array
    intensity_scale = np.sum(corrected_ants_array * corrected_torch_array) / np.sum(corrected_torch_array**2)
    aligned_corrected_difference = intensity_scale * corrected_torch_array - corrected_ants_array
    normalized_ants_bias = normalized_bias_array(bias_ants)
    normalized_torch_bias = normalized_bias_array(bias_torch)
    bias_difference = np.log(normalized_torch_bias) - np.log(normalized_ants_bias)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.output_prefix
    ants.image_write(n4_ants, f"{prefix}_ants_corrected.nii.gz")
    ants.image_write(n4_torch, f"{prefix}_antstorch_corrected.nii.gz")
    ants.image_write(bias_ants, f"{prefix}_ants_bias.nii.gz")
    ants.image_write(bias_torch, f"{prefix}_antstorch_bias.nii.gz")

    print(f"Input: {input_path}")
    print(f"Geometry: size={t1.shape}, spacing={t1.spacing}, origin={t1.origin}")
    print(f"ANTs runtime:      {ants_seconds:.3f} s (correction + bias-field run)")
    print(f"ANTsTorch runtime: {torch_seconds:.3f} s on {device} (correction + bias-field run)")
    print(f"ANTs intensity range: {corrected_ants_array.min():.6g} to {corrected_ants_array.max():.6g}")
    print(f"ANTsTorch intensity range: {corrected_torch_array.min():.6g} to {corrected_torch_array.max():.6g}") 
    print(f"ANTs bias-field range: {normalized_ants_bias.min():.6g} to {normalized_ants_bias.max():.6g}")
    print(f"ANTsTorch bias-field range: {normalized_torch_bias.min():.6g} to {normalized_torch_bias.max():.6g}")
    print(
        "B-spline accumulation: "
        f"{'stable matrix reduction' if stable_accumulation else 'vectorized scatter'}"
    )
    print(f"Corrected-image RMSE: {np.sqrt(np.mean(corrected_difference**2)):.6g}")
    print(
        "Scale-aligned corrected-image RMSE: "
        f"{np.sqrt(np.mean(aligned_corrected_difference**2)):.6g} (scale={intensity_scale:.8g})"
    )
    print(f"Normalized log-bias MAE: {np.mean(np.abs(bias_difference)):.6g}")
    print(
        "Normalized log-bias correlation: "
        f"{np.corrcoef(np.log(normalized_ants_bias).ravel(), np.log(normalized_torch_bias).ravel())[0, 1]:.8f}"
    )
    if device.type == "cuda":
        print(f"ANTsTorch CUDA peak memory: {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB")
    print(f"Outputs written with prefix: {prefix}")


if __name__ == "__main__":
    main()
