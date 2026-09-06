#!/usr/bin/env python3
"""Run ANTsTorch B-spline or Gaussian SVF registration on 2-D or 3-D ANTs images.

By default, registers the bundled ANTs ``r30``/``r27`` 2-D demo images; pass
``--fixed``/``--moving`` to register your own images instead, 2-D or 3-D --
``bspline_svf_registration()``/``gaussian_svf_registration()`` are both
dimension-agnostic already (the same functions this project's Mindboggle
benchmark runs on real 3-D brain volumes elsewhere). What used to make this
script 2-D-only was its own hand-rolled ANTsPy<->PyTorch tensor-conversion
helpers, not those registration functions; they have been replaced with the
dimension-agnostic equivalents from ``antstorch.syn.bridge``
(``ants_image_to_tensor``/``tensor_to_ants_image``/
``displacement_xyz_to_ants_image``), which that module's own docstring
already documents as bridging exactly these fields.

Example
-------
Run the default four-level registration on CPU::

    PYTHONPATH=. python tools/run_svf_registration.py

Register your own images instead of the bundled r30/r27 demo pair -- 2-D or
3-D, e.g. a real Mindboggle-style pair::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --fixed /path/to/fixed.nii.gz --moving /path/to/moving.nii.gz

Use an accelerator and fewer iterations::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --device mps --iterations 40 30 20 --output-dir registration_output

Run the dense Gaussian-regularized SVF::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --transform-type gaussian_svf --update-field-sigma 3 \
        --total-field-sigma 0.5 --verbose

Use the squared local normalized cross-correlation metric (the same
similarity vocabulary as run_syn_registration.py's --similarity)::

    PYTHONPATH=. python tools/run_svf_registration.py \
        --similarity cc2 --neighborhood-radius 2 --verbose

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
import torch

from antstorch.ants_transform_io import write_affine_transform
from antstorch.bspline_flows import (
    ImageDomain,
    PhysicalGradientDescent,
    affine_registration,
    bspline_svf_registration,
    gaussian_svf_registration,
)
from antstorch.syn.bridge import (
    ants_image_to_tensor,
    displacement_xyz_to_ants_image,
    tensor_to_ants_image,
)


def ants_to_torch(image: ants.ANTsImage, device: torch.device) -> torch.Tensor:
    """Convert a scalar 2-D or 3-D ANTs image to a singleton ``(1, 1, *torch_shape)``
    tensor, raw (unnormalized) intensities -- a thin wrapper over
    :func:`antstorch.syn.bridge.ants_image_to_tensor` (``normalize=False``,
    matching this script's own historical behavior; that function defaults
    to ``normalize=True``, a percentile-clip normalization this script has
    never applied)."""
    if image.components != 1:
        raise ValueError("This example expects a scalar ANTs image")
    return ants_image_to_tensor(image, device=device, normalize=False)


def image_domain(image: ants.ANTsImage) -> ImageDomain:
    """Copy all physical-space metadata from an ANTs image."""
    return ImageDomain(
        size=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in image.spacing),
        origin=tuple(float(value) for value in image.origin),
        direction=tuple(tuple(float(value) for value in row) for row in image.direction),
    )


def torch_image_to_ants(tensor: torch.Tensor, reference: ants.ANTsImage) -> ants.ANTsImage:
    """Convert a singleton ``(1, 1, *torch_shape)`` tensor to a scalar ANTs
    image -- :func:`antstorch.syn.bridge.tensor_to_ants_image`, dimension-agnostic."""
    return tensor_to_ants_image(tensor, reference)


def torch_field_to_ants(tensor: torch.Tensor, reference: ants.ANTsImage) -> ants.ANTsImage:
    """Convert a channel-first ``(1, dim, *torch_shape)`` ITK-order physical
    displacement field to an ANTs vector image --
    :func:`antstorch.syn.bridge.displacement_xyz_to_ants_image`, the exact
    convention ``bspline_svf_registration()``/``gaussian_svf_registration()``
    return their fields in (per that function's own docstring), dimension-agnostic."""
    return displacement_xyz_to_ants_image(tensor, reference)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fixed",
        type=Path,
        default=None,
        help="Path to the fixed image (scalar, 2-D or 3-D). Defaults to the bundled "
        "ANTs 'r30' demo image (ants.get_ants_data('r30')) when omitted.",
    )
    parser.add_argument(
        "--moving",
        type=Path,
        default=None,
        help="Path to the moving image (scalar, 2-D or 3-D). Defaults to the bundled "
        "ANTs 'r27' demo image (ants.get_ants_data('r27')) when omitted.",
    )
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
    parser.add_argument(
        "--similarity",
        choices=("mse", "lncc", "cc", "lncc2", "cc2", "mattes", "mi"),
        default="lncc",
        help="Similarity metric for the SVF stage -- identical vocabulary/implementation "
        "as run_syn_registration.py's --similarity (default: lncc)",
    )
    parser.add_argument("--neighborhood-radius", type=int, default=4, help="Window radius for lncc/cc")
    parser.add_argument("--num-bins", type=int, default=32, help="Histogram bins for mattes/mi")
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
    parser.add_argument(
        "--affine-similarity",
        choices=("mse", "lncc", "cc", "lncc2", "cc2", "mattes", "mi"),
        default="mse",
    )
    parser.add_argument("--affine-neighborhood-radius", type=int, default=4)
    parser.add_argument("--affine-num-bins", type=int, default=32, help="Histogram bins for affine mattes/mi")
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

    # --fixed/--moving default to the bundled ANTs r30/r27 (2-D) demo pair
    # when omitted -- ants.get_ants_data() resolves those from ANTsPy's own
    # packaged test data, no network access needed. Any scalar 2-D or 3-D
    # image works when given explicitly (ants_to_torch() above still rejects
    # a non-scalar/multi-component image with a clear message).
    fixed_path = str(args.fixed) if args.fixed is not None else ants.get_ants_data("r30")
    moving_path = str(args.moving) if args.moving is not None else ants.get_ants_data("r27")
    fixed_ants = ants.image_read(fixed_path).clone("float")
    moving_ants = ants.image_read(moving_path).clone("float")
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
            num_bins=args.affine_num_bins,
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
        num_bins=args.num_bins,
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
    # Generic output filenames (fixed.nii.gz/moving.nii.gz/warped_moving.nii.gz)
    # rather than the old r30/r27-specific ones -- those names are wrong/
    # misleading once --fixed/--moving point at arbitrary images.
    ants.image_write(fixed_ants, str(args.output_dir / "fixed.nii.gz"))
    ants.image_write(moving_ants, str(args.output_dir / "moving.nii.gz"))
    ants.image_write(
        torch_image_to_ants(result["warpedmovout"], fixed_ants),
        str(args.output_dir / "warped_moving.nii.gz"),
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
            dim=fixed_ants.dimension,
            filename=str(args.output_dir / "total_affine_0GenericAffine.mat"),
        )

    if affine_result is not None:
        ants.image_write(
            torch_image_to_ants(affine_result["warpedmovout"], fixed_ants),
            str(args.output_dir / "affine_warped_moving.nii.gz"),
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
