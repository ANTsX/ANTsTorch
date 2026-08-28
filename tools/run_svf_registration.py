#!/usr/bin/env python3
"""Run ANTsTorch B-spline or Gaussian SVF registration on ANTs 2-D images.

Example
-------
Run the default four-level registration on CPU::

    PYTHONPATH=. python tools/run_svf_registration.py

Use an accelerator and fewer iterations::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --device mps --iterations 40 30 20 --output-dir registration_output

Run the dense Gaussian-regularized SVF::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --transform-type gaussian_svf --update-field-sigma 3 \
        --total-field-sigma 0.5 --verbose

Use the local ANTs neighborhood-correlation metric::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --similarity ants_ncc --neighborhood-radius 2 --verbose

Run an affine pre-registration before the selected SVF (bspline_flows has no
affine/rigid initialization of its own; see
``antstorch.bspline_flows.affine_registration``)::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --affine --affine-transform-type Rigid --verbose
"""

import argparse
import json
import time
from pathlib import Path

import ants
import numpy as np
import torch

from antstorch.ants_transform_io import write_affine_transform
from antstorch.bspline_flows import (
    ImageDomain,
    PhysicalGradientDescent,
    affine_registration,
    bspline_svf_registration,
    gaussian_svf_registration,
)


def ants_to_torch(image: ants.ANTsImage, device: torch.device) -> torch.Tensor:
    """Convert ANTs x-y storage to singleton N-C-Y-X PyTorch storage."""
    if image.dimension != 2 or image.components != 1:
        raise ValueError("This example expects a scalar 2-D ANTs image")
    array = np.ascontiguousarray(image.numpy().astype(np.float32, copy=False).T)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def image_domain(image: ants.ANTsImage) -> ImageDomain:
    """Copy all physical-space metadata from an ANTs image."""
    return ImageDomain(
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
    parser.add_argument(
        "--transform-type",
        choices=("bspline_svf", "gaussian_svf"),
        default="bspline_svf",
        help="SVF parameterization/regularizer (default: bspline_svf)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=Path("registration_output"))
    parser.add_argument("--spline-distance", type=float, default=26.0)
    parser.add_argument("--shrink-factors", type=int, nargs="+", default=(8, 4, 2, 1))
    parser.add_argument("--smoothing-sigmas", type=float, nargs="+", default=(3.0, 2.0, 1.0, 0.0))
    parser.add_argument("--iterations", type=int, nargs="+", default=(100, 100, 50, 10))
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
    parser.add_argument(
        "--update-field-sigma",
        type=float,
        default=3.0,
        help="Gaussian SVF update-field sigma in physical units (default: 3.0)",
    )
    parser.add_argument(
        "--total-field-sigma",
        type=float,
        default=0.5,
        help="Gaussian SVF accumulated-velocity sigma in physical units (default: 0.5)",
    )
    parser.add_argument("--similarity", choices=("mse", "ncc", "ants_ncc"), default="ants_ncc")
    parser.add_argument("--neighborhood-radius", type=int, default=4)
    parser.add_argument("--coefficient-weight", type=float, default=0.0)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--bending-weight", type=float, default=0.0)
    parser.add_argument(
        "--affine",
        action="store_true",
        help="Run an affine pre-registration (antstorch.bspline_flows.affine_registration) "
        "before the selected SVF, and pass its result as the registration's initial_affine",
    )
    parser.add_argument(
        "--affine-transform-type",
        choices=("Translation", "Rigid", "Similarity", "Affine"),
        default="Affine",
        help="Linear-transform hierarchy for the affine pre-registration (default: Affine)",
    )
    parser.add_argument("--affine-similarity", choices=("mse", "ncc", "ants_ncc"), default="mse")
    parser.add_argument("--affine-neighborhood-radius", type=int, default=4)
    parser.add_argument("--affine-shrink-factors", type=int, nargs="+", default=(4, 2, 1))
    parser.add_argument("--affine-smoothing-sigmas", type=float, nargs="+", default=(2.0, 1.0, 0.0))
    parser.add_argument("--affine-iterations", type=int, nargs="+", default=(100, 75, 50))
    parser.add_argument("--affine-learning-rate", type=float, nargs="+", default=(0.05, 0.03, 0.02))
    parser.add_argument(
        "--affine-single-start",
        action="store_true",
        help="Disable the multi-start seed-rotation search (start from identity only)",
    )
    parser.add_argument(
        "--affine-no-center-of-mass-init",
        action="store_true",
        help="Disable center-of-mass translation initialization for the affine fit",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-level optimization progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if args.transform_type == "gaussian_svf" and args.optimizer != "physical_gradient_descent":
        raise ValueError("gaussian_svf requires --optimizer physical_gradient_descent")
    level_count = len(args.shrink_factors)
    for name, values in (
        ("--smoothing-sigmas", args.smoothing_sigmas),
        ("--iterations", args.iterations),
        ("--learning-rate", args.learning_rate),
    ):
        if len(values) != level_count:
            raise ValueError(f"{name} must have one value per shrink factor")
    if args.affine:
        affine_level_count = len(args.affine_shrink_factors)
        for name, values in (
            ("--affine-smoothing-sigmas", args.affine_smoothing_sigmas),
            ("--affine-iterations", args.affine_iterations),
            ("--affine-learning-rate", args.affine_learning_rate),
        ):
            if len(values) != affine_level_count:
                raise ValueError(f"{name} must have one value per affine shrink factor")

    fixed_ants = ants.image_read(ants.get_ants_data("r30")).clone("float")
    moving_ants = ants.image_read(ants.get_ants_data("r27")).clone("float")
    fixed = ants_to_torch(fixed_ants, device)
    moving = ants_to_torch(moving_ants, device)
    fixed_domain = image_domain(fixed_ants)
    moving_domain = image_domain(moving_ants)

    affine_result = None
    initial_affine = None
    affine_elapsed = 0.0
    if args.affine:
        affine_start = time.perf_counter()
        affine_result = affine_registration(
            fixed=fixed,
            moving=moving,
            fixed_domain=fixed_domain,
            moving_domain=moving_domain,
            transform_type=args.affine_transform_type,
            similarity=args.affine_similarity,
            neighborhood_radius=args.affine_neighborhood_radius,
            shrink_factors=tuple(args.affine_shrink_factors),
            smoothing_sigmas=tuple(args.affine_smoothing_sigmas),
            iterations=tuple(args.affine_iterations),
            learning_rate=tuple(args.affine_learning_rate),
            multi_start=not args.affine_single_start,
            center_of_mass_init=not args.affine_no_center_of_mass_init,
            padding_mode="border",
            verbose=args.verbose,
        )
        affine_elapsed = time.perf_counter() - affine_start
        initial_affine = (affine_result["matrix"], affine_result["translation"])

    start = time.perf_counter()
    optimizer = (
        PhysicalGradientDescent(
            gradient_step=args.gradient_step,
            momentum=args.momentum,
            # B-spline SVF smooths the coefficient gradient here. Gaussian
            # SVF instead uses its distinct --update-field-sigma below.
            smoothing_sigma=(
                args.gradient_smoothing_sigma if args.transform_type == "bspline_svf" else 0.0
            ),
        )
        if args.optimizer == "physical_gradient_descent"
        else args.optimizer
    )
    common_registration_kwargs = dict(
        fixed=fixed,
        moving=moving,
        fixed_domain=fixed_domain,
        moving_domain=moving_domain,
        shrink_factors=tuple(args.shrink_factors),
        smoothing_sigmas=tuple(args.smoothing_sigmas),
        iterations=tuple(args.iterations),
        optimizer=optimizer,
        gradient_step=args.gradient_step,
        similarity=args.similarity,
        neighborhood_radius=args.neighborhood_radius,
        velocity_weight=args.velocity_weight,
        bending_weight=args.bending_weight,
        initial_affine=initial_affine,
        padding_mode="border",
        stationary_boundary=True,
        verbose=args.verbose,
    )
    if args.transform_type == "bspline_svf":
        result = bspline_svf_registration(
            **common_registration_kwargs,
            spline_distance=args.spline_distance,
            learning_rate=tuple(args.learning_rate),
            coefficient_weight=args.coefficient_weight,
        )
    else:
        result = gaussian_svf_registration(
            **common_registration_kwargs,
            update_field_sigma=args.update_field_sigma,
            total_field_sigma=args.total_field_sigma,
        )
    elapsed = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(fixed_ants, str(args.output_dir / "fixed_r30.nii.gz"))
    ants.image_write(moving_ants, str(args.output_dir / "moving_r27.nii.gz"))
    ants.image_write(
        torch_image_to_ants(result["warpedmovout"], fixed_ants),
        str(args.output_dir / "warped_r27.nii.gz"),
    )
    # fwdtransforms/invtransforms are always the pure selected SVF
    # piece alone -- never composed with the affine below -- matching the
    # separated-transform convention also used by antstorch.syn.syn_registration().
    # The total forward map is affine-then-SVF; compose svf_*_displacement.nii.gz
    # with affine_*_displacement.nii.gz (antstorch.bspline_flows.spatial_transform
    # .compose_displacements) if the composed field is needed.
    for name, filename in (
        ("velocity", "velocity"),
        ("fwdtransforms", "svf_forward_displacement"),
        ("invtransforms", "svf_inverse_displacement"),
    ):
        ants.image_write(
            torch_field_to_ants(result[name], fixed_ants),
            str(args.output_dir / f"{filename}.nii.gz"),
        )
    if "coefficients" in result:
        torch.save(result["coefficients"].cpu(), args.output_dir / "coefficients.pt")
    with (args.output_dir / "loss_history.json").open("w", encoding="utf-8") as stream:
        json.dump(result["level_loss_history"], stream, indent=2)
    if result["affine_matrix"] is not None:
        # A real, ANTsX-compatible affine transform file (usable with
        # antsApplyTransforms / ants.apply_transforms), in addition to the
        # affine_transform.pt raw tensor save below -- see
        # antstorch.ants_transform_io for the exact file convention matched.
        write_affine_transform(
            result["affine_matrix"][0],
            result["affine_translation"][0],
            dim=2,
            filename=str(args.output_dir / "total_affine_0GenericAffine.mat"),
        )

    if affine_result is not None:
        ants.image_write(
            torch_image_to_ants(affine_result["warpedmovout"], fixed_ants),
            str(args.output_dir / "affine_warped_r27.nii.gz"),
        )
        for name, filename in (
            ("fwdtransforms", "affine_forward_displacement"),
            ("invtransforms", "affine_inverse_displacement"),
        ):
            ants.image_write(
                torch_field_to_ants(affine_result[name], fixed_ants),
                str(args.output_dir / f"{filename}.nii.gz"),
            )
        torch.save(
            {"matrix": affine_result["matrix"].cpu(), "translation": affine_result["translation"].cpu()},
            args.output_dir / "affine_transform.pt",
        )
        with (args.output_dir / "affine_loss_history.json").open("w", encoding="utf-8") as stream:
            json.dump(affine_result["level_loss_history"], stream, indent=2)

    print(f"{args.transform_type} registration completed in {elapsed:.2f} seconds on {device}.")
    if affine_result is not None:
        print(
            f"  (preceded by an affine pre-registration: {affine_elapsed:.2f} seconds, "
            f"transform_type={args.affine_transform_type})"
        )
    print(f"Final loss: {result['loss'].item():.6g}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
