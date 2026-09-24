#!/usr/bin/env python3
"""Compare ANTs and ANTsTorch adaptive non-local means denoising.

Both ``ants.denoise_image`` and ``antstorch.denoise_image`` receive the same
ANTsImage, mask and parameters. The script reports runtimes, agreement
metrics (RMSE, MAE, maximum difference, correlation) and writes the denoised
images, the removed-noise images and the ANTsTorch - ANTs difference.

ITK's multi-threaded denoiser accumulates into overlapping patches without
a fixed order, so repeated ANTs runs are not bit-identical. With
``--repeats 2`` or more, the ANTs run-to-run difference is also reported; it
is the floor below which ANTs/ANTsTorch differences are not meaningful.
``--ants-threads 1`` makes the ANTs reference deterministic.

Examples
--------
Use the bundled 2-D ``r16`` image::

    python tools/benchmarks/compare_denoise_image.py

Gaussian noise model, larger search radius, MPS::

    python tools/benchmarks/compare_denoise_image.py --noise-model Gaussian -r 3 --device mps

A 3-D image with a foreground mask, shrink factor and added Rician noise::

    python tools/benchmarks/compare_denoise_image.py image.nii.gz --auto-mask \\
        --shrink-factor 2 --add-noise 0.05 --output-dir results/denoise

Deterministic single-threaded ANTs reference with timing over 5 runs::

    python tools/benchmarks/compare_denoise_image.py --ants-threads 1 --repeats 5
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np


def synchronize(device) -> None:
    """Wait for asynchronous accelerator work before timing boundaries."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def parse_radius(value: str):
    """Accept ``2``, ``2x2`` or ``2x2x1`` like ANTsPy."""
    return value if "x" in value else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Input 2-D or 3-D image. When omitted, ants.get_ants_data('r16') is used.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda or mps")
    parser.add_argument("--noise-model", choices=("Rician", "Gaussian"), default="Rician")
    parser.add_argument("-p", "--patch-radius", type=parse_radius, default=1,
                        help="Patch radius, e.g. 1 or 1x1x1 (default: 1)")
    parser.add_argument("-r", "--search-radius", type=parse_radius, default=2,
                        help="Search radius, e.g. 2 or 2x2x2 (default: 2)")
    parser.add_argument("--shrink-factor", type=int, default=1)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--mask", type=Path, help="Mask image in the input's physical space")
    mask_group.add_argument("--auto-mask", action="store_true",
                            help="Use ants.get_mask(image) as the mask")
    parser.add_argument(
        "--add-noise",
        type=float,
        default=0.0,
        help=(
            "Add synthetic noise before denoising, with standard deviation given "
            "as a fraction of the maximum intensity (Rician or Gaussian, following "
            "--noise-model). Default: 0, no added noise."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --add-noise")
    parser.add_argument(
        "--ants-threads",
        type=int,
        default=None,
        help="ITK thread count for ANTs (1 gives a deterministic reference). Default: ITK default.",
    )
    parser.add_argument("--repeats", type=int, default=1,
                        help="Timed runs per implementation; the median is reported (default: 1)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the untimed ANTsTorch warm-up run")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for output images; created if needed (default: current directory)")
    parser.add_argument("--output-prefix", default="denoise_comparison")
    return parser.parse_args()


def add_noise(image, fraction: float, noise_model: str, seed: int):
    import ants

    rng = np.random.default_rng(seed)
    array = image.numpy().astype(np.float64)
    sigma = fraction * float(array.max())
    if noise_model == "Rician":
        real = array + rng.normal(0.0, sigma, array.shape)
        imaginary = rng.normal(0.0, sigma, array.shape)
        noisy = np.sqrt(real**2 + imaginary**2)
    else:
        noisy = array + rng.normal(0.0, sigma, array.shape)
    return ants.from_numpy(
        noisy.astype(np.float32),
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    ), sigma


def metrics(reference: np.ndarray, test: np.ndarray, region: np.ndarray) -> dict:
    ref = reference[region]
    tst = test[region]
    diff = tst - ref
    intensity_range = float(ref.max() - ref.min()) or 1.0
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {
        "rmse": rmse,
        "relative_rmse": rmse / intensity_range,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "correlation": float(np.corrcoef(ref.ravel(), tst.ravel())[0, 1]),
    }


def print_metrics(label: str, values: dict) -> None:
    print(f"{label}:")
    print(f"  RMSE:            {values['rmse']:.6g} ({100 * values['relative_rmse']:.4f} % of range)")
    print(f"  MAE:             {values['mae']:.6g}")
    print(f"  Max |diff|:      {values['max_abs']:.6g}")
    print(f"  Correlation:     {values['correlation']:.10f}")


def main() -> None:
    args = parse_args()
    if args.ants_threads is not None:
        # ITK reads this when the first filter is created; set before importing ants.
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.ants_threads)

    import ants
    import torch

    import antstorch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    input_path = args.image or ants.get_ants_data("r16")
    image = ants.image_read(input_path).clone("float")
    if image.dimension not in (2, 3) or image.components != 1:
        raise ValueError("This comparison expects a scalar 2-D or 3-D image")

    sigma = None
    if args.add_noise > 0:
        image, sigma = add_noise(image, args.add_noise, args.noise_model, args.seed)

    mask = None
    if args.mask is not None:
        mask = ants.image_read(str(args.mask))
        if mask.shape != image.shape or not ants.image_physical_space_consistency(image, mask):
            raise ValueError("--mask must occupy the same physical space as the image")
    elif args.auto_mask:
        mask = ants.get_mask(image)
    region = mask.numpy() > 0 if mask is not None else np.ones(image.shape, dtype=bool)

    options = {
        "shrink_factor": args.shrink_factor,
        "p": args.patch_radius,
        "r": args.search_radius,
        "noise_model": args.noise_model,
    }

    # ANTs reference runs.
    ants_results, ants_times = [], []
    for run in range(args.repeats):
        if args.verbose:
            print(f"Running ANTs denoise_image ({run + 1}/{args.repeats})...")
        start = time.perf_counter()
        ants_results.append(ants.denoise_image(image, mask=mask, v=int(args.verbose), **options))
        ants_times.append(time.perf_counter() - start)

    # ANTsTorch runs; the warm-up absorbs one-time kernel and allocator setup.
    if not args.no_warmup:
        if args.verbose:
            print("Running ANTsTorch warm-up...")
        antstorch.denoise_image(image, mask=mask, device=device, **options)
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch_results, torch_times = [], []
    for run in range(args.repeats):
        if args.verbose:
            print(f"Running ANTsTorch denoise_image ({run + 1}/{args.repeats})...")
        synchronize(device)
        start = time.perf_counter()
        torch_results.append(
            antstorch.denoise_image(image, mask=mask, device=device, verbose=args.verbose, **options)
        )
        synchronize(device)
        torch_times.append(time.perf_counter() - start)

    input_array = image.numpy().astype(np.float64)
    ants_array = ants_results[0].numpy().astype(np.float64)
    torch_array = torch_results[0].numpy().astype(np.float64)
    if not ants.image_physical_space_consistency(image, torch_results[0]):
        print("WARNING: ANTsTorch output geometry differs from the input geometry")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.output_prefix
    if sigma is not None:
        ants.image_write(image, f"{prefix}_input_noisy.nii.gz")
    ants.image_write(ants_results[0], f"{prefix}_ants_denoised.nii.gz")
    ants.image_write(torch_results[0], f"{prefix}_antstorch_denoised.nii.gz")
    ants.image_write(image - ants_results[0], f"{prefix}_ants_noise.nii.gz")
    ants.image_write(image - torch_results[0], f"{prefix}_antstorch_noise.nii.gz")
    ants.image_write(torch_results[0] - ants_results[0], f"{prefix}_difference.nii.gz")

    print(f"Input: {input_path}")
    print(f"Geometry: size={image.shape}, spacing={image.spacing}, origin={image.origin}")
    if sigma is not None:
        print(f"Added {args.noise_model} noise: sigma={sigma:.6g} ({args.add_noise:g} x max), seed={args.seed}")
    print(
        f"Parameters: noise_model={args.noise_model}, p={args.patch_radius}, "
        f"r={args.search_radius}, shrink_factor={args.shrink_factor}, "
        f"mask={'file' if args.mask else 'auto' if args.auto_mask else 'none'} "
        f"({int(region.sum())} voxels compared)"
    )
    threads = args.ants_threads if args.ants_threads is not None else "ITK default"
    print(f"ANTs runtime:      {np.median(ants_times):.4f} s median of {args.repeats} (threads: {threads})")
    print(f"ANTsTorch runtime: {np.median(torch_times):.4f} s median of {args.repeats} on {device}"
          f"{'' if args.no_warmup else ' (after warm-up)'}")
    print(f"Speed ratio (ANTs / ANTsTorch): {np.median(ants_times) / np.median(torch_times):.2f}x")
    print(f"Input intensity range:     {input_array[region].min():.6g} to {input_array[region].max():.6g}")
    print(f"ANTs output range:         {ants_array[region].min():.6g} to {ants_array[region].max():.6g}")
    print(f"ANTsTorch output range:    {torch_array[region].min():.6g} to {torch_array[region].max():.6g}")
    print(f"Removed-noise std, ANTs:      {np.std((input_array - ants_array)[region]):.6g}")
    print(f"Removed-noise std, ANTsTorch: {np.std((input_array - torch_array)[region]):.6g}")
    print_metrics("ANTsTorch vs ANTs (denoised image)", metrics(ants_array, torch_array, region))
    print_metrics(
        "ANTsTorch vs ANTs (removed noise)",
        metrics(input_array - ants_array, input_array - torch_array, region),
    )

    if args.repeats >= 2:
        ants_spread = max(
            float(np.max(np.abs(r.numpy().astype(np.float64) - ants_array))) for r in ants_results[1:]
        )
        torch_spread = max(
            float(np.max(np.abs(r.numpy().astype(np.float64) - torch_array))) for r in torch_results[1:]
        )
        print(f"ANTs run-to-run max |diff|:      {ants_spread:.6g}")
        print(f"ANTsTorch run-to-run max |diff|: {torch_spread:.6g}")

    if device.type == "cuda":
        print(f"ANTsTorch CUDA peak memory: {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB")
    print(f"Outputs written with prefix: {prefix}")


if __name__ == "__main__":
    main()
