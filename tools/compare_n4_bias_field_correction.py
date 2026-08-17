#!/usr/bin/env python3
"""Compare ANTs and differentiable ANTsTorch N4 bias correction.

Examples
--------
Use the bundled 2-D ``r16`` image::

    python tools/compare_n4_bias_field_correction.py

Use another 2-D or 3-D image and CUDA, if available::

    python tools/compare_n4_bias_field_correction.py image.nii.gz --device cuda
"""

import argparse
import time
from pathlib import Path

import ants
import numpy as np
import torch

import antstorch


def ants_to_torch(image: ants.ANTsImage, device: torch.device) -> torch.Tensor:
    """Convert ANTs x-y-z array storage to N-C-(D)-H-W PyTorch storage."""
    array_itk = image.numpy().astype(np.float32, copy=False)
    spatial_axes = tuple(range(image.dimension - 1, -1, -1))
    array_torch = np.ascontiguousarray(np.transpose(array_itk, spatial_axes))
    return torch.from_numpy(array_torch).unsqueeze(0).unsqueeze(0).to(device)


def torch_to_ants(tensor: torch.Tensor, reference: ants.ANTsImage) -> ants.ANTsImage:
    """Convert a singleton N-C PyTorch tensor back to reference ANTs geometry."""
    if tensor.shape[:2] != (1, 1):
        raise ValueError("This example expects a singleton batch and scalar image channel")
    array_torch = tensor.detach().cpu().numpy()[0, 0]
    spatial_axes = tuple(range(reference.dimension - 1, -1, -1))
    array_itk = np.ascontiguousarray(np.transpose(array_torch, spatial_axes))
    return ants.from_numpy(
        array_itk,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


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
        "--mesh-size",
        type=int,
        nargs="+",
        default=None,
        help="B-spline mesh size in ITK x-y-z order (default: one span per axis)",
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

    start = time.perf_counter()
    n4_ants = ants.n4_bias_field_correction(
        t1,
        mask=mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=mesh_size,
    )
    bias_ants = ants.n4_bias_field_correction(
        t1,
        mask=mask,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=mesh_size,
        return_bias_field=True,
    )
    ants_seconds = time.perf_counter() - start

    domain = antstorch.BSplineDomain(
        size=tuple(int(value) for value in t1.shape),
        spacing=tuple(float(value) for value in t1.spacing),
        origin=tuple(float(value) for value in t1.origin),
        direction=tuple(tuple(float(value) for value in row) for row in t1.direction),
    )
    t1_tensor = ants_to_torch(t1, device)
    mask_tensor = ants_to_torch(mask, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    n4_torch_tensor = antstorch.n4_bias_field_correction(
        t1_tensor,
        domain,
        mask_tensor,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=tuple(mesh_size),
    )
    bias_torch_tensor = antstorch.n4_bias_field_correction(
        t1_tensor,
        domain,
        mask_tensor,
        shrink_factor=args.shrink_factor,
        convergence=convergence,
        spline_param=tuple(mesh_size),
        return_bias_field=True,
    )
    synchronize(device)
    torch_seconds = time.perf_counter() - start

    n4_torch = torch_to_ants(n4_torch_tensor, t1)
    bias_torch = torch_to_ants(bias_torch_tensor, t1)
    corrected_ants_array = n4_ants.numpy().astype(np.float64)
    corrected_torch_array = n4_torch.numpy().astype(np.float64)
    corrected_difference = corrected_torch_array - corrected_ants_array
    intensity_scale = np.sum(corrected_ants_array * corrected_torch_array) / np.sum(corrected_torch_array**2)
    aligned_corrected_difference = intensity_scale * corrected_torch_array - corrected_ants_array
    normalized_ants_bias = normalized_bias_array(bias_ants)
    normalized_torch_bias = normalized_bias_array(bias_torch)
    bias_difference = np.log(normalized_torch_bias) - np.log(normalized_ants_bias)

    prefix = Path(args.output_prefix)
    ants.image_write(n4_ants, f"{prefix}_ants_corrected.nii.gz")
    ants.image_write(n4_torch, f"{prefix}_antstorch_corrected.nii.gz")
    ants.image_write(bias_ants, f"{prefix}_ants_bias.nii.gz")
    ants.image_write(bias_torch, f"{prefix}_antstorch_bias.nii.gz")

    print(f"Input: {input_path}")
    print(f"Geometry: size={t1.shape}, spacing={t1.spacing}, origin={t1.origin}")
    print(f"ANTs runtime:      {ants_seconds:.3f} s (correction + bias-field run)")
    print(f"ANTsTorch runtime: {torch_seconds:.3f} s on {device} (correction + bias-field run)")
    print(f"B-spline accumulation: {'stable matrix reduction' if device.type == 'mps' else 'vectorized scatter'}")
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
