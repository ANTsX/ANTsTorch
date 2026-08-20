#!/usr/bin/env python3
"""Run ANTsTorch B-spline SVF registration on the ANTs r30/r64 images.

Example
-------
Run the default three-level registration on CPU::

    PYTHONPATH=. python tools/run_registration.py

Use an accelerator and fewer iterations::

    PYTHONPATH=. python tools/run_registration.py \
        --device mps --iterations 40 30 20 --output-dir registration_output

Use the local ANTs neighborhood-correlation metric::

    PYTHONPATH=. python tools/run_registration.py \
        --similarity ants_ncc --neighborhood-radius 2 --verbose
"""

import argparse
import json
import time
from pathlib import Path

import ants
import numpy as np
import torch

from antstorch.bspline_flows import BSplineDomain, PhysicalGradientDescent, registration


def ants_to_torch(image: ants.ANTsImage, device: torch.device) -> torch.Tensor:
    """Convert ANTs x-y storage to singleton N-C-Y-X PyTorch storage."""
    if image.dimension != 2 or image.components != 1:
        raise ValueError("This example expects a scalar 2-D ANTs image")
    array = np.ascontiguousarray(image.numpy().astype(np.float32, copy=False).T)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def image_domain(image: ants.ANTsImage) -> BSplineDomain:
    """Copy all physical-space metadata from an ANTs image."""
    return BSplineDomain(
        size=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in image.spacing),
        origin=tuple(float(value) for value in image.origin),
        direction=tuple(tuple(float(value) for value in row) for row in image.direction),
    )


def torch_image_to_ants(tensor: torch.Tensor, reference: ants.ANTsImage) -> ants.ANTsImage:
    """Convert a singleton N-C-Y-X tensor to a scalar ANTs image."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy()[0, 0].T)
    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def torch_field_to_ants(tensor: torch.Tensor, reference: ants.ANTsImage) -> ants.ANTsImage:
    """Convert N-(x,y)-Y-X physical vectors to an ANTs vector image."""
    if tensor.shape[0] != 1 or tensor.shape[1] != 2:
        raise ValueError("This example expects a singleton batch of 2-D vectors")
    array = np.ascontiguousarray(tensor.detach().cpu().permute(0, 3, 2, 1).numpy()[0])
    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
        has_components=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=Path("registration_output"))
    parser.add_argument("--mesh-size", type=int, nargs=2, default=(5, 5), metavar=("X", "Y"))
    parser.add_argument("--shrink-factors", type=int, nargs="+", default=(8, 4, 2, 1))
    parser.add_argument("--smoothing-sigmas", type=float, nargs="+", default=(3.0, 2.0, 1.0, 0.0))
    parser.add_argument("--iterations", type=int, nargs="+", default=(100, 70, 40, 20))
    parser.add_argument("--learning-rate", type=float, nargs="+", default=(0.03, 0.02, 0.01, 0.005))
    parser.add_argument(
        "--optimizer",
        choices=("physical_gradient_descent", "adam", "lbfgs"),
        default="physical_gradient_descent",
    )
    parser.add_argument(
        "--gradient-step",
        type=float,
        default=0.2,
        help="Physical gradient step lambda in [0.1, 0.25] (default: 0.2)",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--gradient-smoothing-sigma",
        type=float,
        default=1.0,
        help="Coefficient-gradient smoothing sigma in physical units (default: 1.0)",
    )
    parser.add_argument("--similarity", choices=("mse", "ncc", "ants_ncc"), default="ants_ncc")
    parser.add_argument("--neighborhood-radius", type=int, default=4)
    parser.add_argument("--coefficient-weight", type=float, default=0.0)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--bending-weight", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true", help="Print per-level optimization progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    level_count = len(args.shrink_factors)
    for name, values in (
        ("--smoothing-sigmas", args.smoothing_sigmas),
        ("--iterations", args.iterations),
        ("--learning-rate", args.learning_rate),
    ):
        if len(values) != level_count:
            raise ValueError(f"{name} must have one value per shrink factor")

    fixed_ants = ants.image_read(ants.get_ants_data("r30")).clone("float")
    moving_ants = ants.image_read(ants.get_ants_data("r27")).clone("float")
    fixed = ants_to_torch(fixed_ants, device)
    moving = ants_to_torch(moving_ants, device)
    fixed_domain = image_domain(fixed_ants)
    moving_domain = image_domain(moving_ants)

    start = time.perf_counter()
    optimizer = (
        PhysicalGradientDescent(
            gradient_step=args.gradient_step,
            momentum=args.momentum,
            smoothing_sigma=args.gradient_smoothing_sigma,
        )
        if args.optimizer == "physical_gradient_descent"
        else args.optimizer
    )
    result = registration(
        fixed=fixed,
        moving=moving,
        fixed_domain=fixed_domain,
        moving_domain=moving_domain,
        mesh_size=tuple(args.mesh_size),
        shrink_factors=tuple(args.shrink_factors),
        smoothing_sigmas=tuple(args.smoothing_sigmas),
        iterations=tuple(args.iterations),
        learning_rate=tuple(args.learning_rate),
        optimizer=optimizer,
        gradient_step=args.gradient_step,
        similarity=args.similarity,
        neighborhood_radius=args.neighborhood_radius,
        coefficient_weight=args.coefficient_weight,
        velocity_weight=args.velocity_weight,
        bending_weight=args.bending_weight,
        padding_mode="border",
        stationary_boundary=True,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(fixed_ants, str(args.output_dir / "fixed_r30.nii.gz"))
    ants.image_write(moving_ants, str(args.output_dir / "moving_r27.nii.gz"))
    ants.image_write(
        torch_image_to_ants(result["warped_moving"], fixed_ants),
        str(args.output_dir / "warped_r27.nii.gz"),
    )
    for name in ("velocity", "forward_displacement", "inverse_displacement"):
        ants.image_write(
            torch_field_to_ants(result[name], fixed_ants),
            str(args.output_dir / f"{name}.nii.gz"),
        )
    torch.save(result["coefficients"].cpu(), args.output_dir / "coefficients.pt")
    with (args.output_dir / "loss_history.json").open("w", encoding="utf-8") as stream:
        json.dump(result["level_loss_history"], stream, indent=2)

    print(f"Registration completed in {elapsed:.2f} seconds on {device}.")
    print(f"Final loss: {result['loss'].item():.6g}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
