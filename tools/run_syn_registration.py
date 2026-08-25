#!/usr/bin/env python3
"""Run ANTsTorch greedy symmetric SyN registration on the ANTs r30/r27 images.

Sibling script to ``tools/run_bspline_svf_registration.py``, built as closely
as possible to the same CLI shape and output-artifact conventions, but using
:func:`antstorch.syn.syn_registration` -- the affine+SyN framework -- instead
of :func:`antstorch.bspline_flows.bspline_svf_registration`. Two differences
follow directly from how that framework is designed, not from a choice made
here:

- ``syn_registration`` accepts and returns ``ants.ANTsImage`` objects (and
  ANTs transform-file paths) directly, so none of
  ``run_bspline_svf_registration.py``'s manual ANTsPy<->PyTorch tensor
  conversion helpers are needed here.
- The default ``--type-of-transform SyN`` already fits an affine
  initialization internally before the dense deformable stage -- unlike
  ``run_bspline_svf_registration.py``, whose default run is SVF-only unless
  ``--affine`` is passed. This matches ``syn_registration()``'s own default
  (and ``ants.registration()``'s), so it is kept rather than special-cased
  away; pass ``--type-of-transform SyNOnly`` for the closest behavioral
  analog of the B-spline script's default.

Example
-------
Run the default affine+SyN registration on CPU::

    PYTHONPATH=. python tools/run_syn_registration.py

Use an accelerator and fewer iterations::

    PYTHONPATH=. python tools/run_syn_registration.py \
        --device mps --reg-iterations 40 30 20 --output-dir syn_registration_output

Use the local ANTs neighborhood-correlation metric for the dense SyN stage::

    PYTHONPATH=. python tools/run_syn_registration.py \
        --syn-metric cc --neighborhood-radius 2 --verbose

Run the dense SyN stage alone, with no affine initialization (closest analog
of ``run_bspline_svf_registration.py``'s own default)::

    PYTHONPATH=. python tools/run_syn_registration.py \
        --type-of-transform SyNOnly --verbose

Fit only an affine transform, no dense stage::

    PYTHONPATH=. python tools/run_syn_registration.py \
        --type-of-transform Rigid --verbose
"""

import argparse
import json
import time
from pathlib import Path

import ants
import torch

from antstorch.syn import syn_registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=Path("syn_registration_output"))
    parser.add_argument(
        "--type-of-transform",
        choices=("Translation", "Rigid", "Similarity", "Affine", "SyN", "SyNOnly"),
        default="SyN",
        help="Transform model: a linear type fits only that transform; 'SyN' (default) fits an "
        "affine initialization then the dense stage; 'SyNOnly' skips the affine fit entirely",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=(4, 2, 1), help="Dense-SyN pyramid shrink factors, coarse to fine")
    parser.add_argument("--reg-iterations", type=int, nargs="+", default=(100, 100, 50))
    parser.add_argument("--grad-step", type=float, default=0.5, help="CFL-bounded per-voxel step, in voxel units")
    parser.add_argument("--flow-sigma", type=float, default=3.0, help="Fluid regularization sigma applied to each raw update")
    parser.add_argument("--total-sigma", type=float, default=0.0, help="Optional elastic (Gaussian) regularization of the composed field")
    parser.add_argument("--regularizer", choices=("gaussian", "sobolev", "dsti"), default="gaussian")
    parser.add_argument("--inverse-method", choices=("fixed_point", "anderson", "hybrid_lm"), default="anderson")
    parser.add_argument("--in-loop-inverse-steps", type=int, default=6)
    parser.add_argument(
        "--no-antisymmetric",
        action="store_false",
        dest="antisymmetric",
        help="Disable the antisymmetric (Frechet-mean) common-mode projection between the two half-warps",
    )
    parser.add_argument("--syn-metric", choices=("mse", "lncc", "cc", "lncc2", "cc2", "mattes", "mi"), default="lncc")
    parser.add_argument("--neighborhood-radius", type=int, default=2, help="Window radius for lncc/cc, or the SyN-stage local metric")
    parser.add_argument("--num-bins", type=int, default=32, help="Histogram bins for mattes/mi")
    parser.add_argument("--affine-transform-type", choices=("Translation", "Rigid", "Similarity", "Affine"), default="Affine")
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
    parser.set_defaults(antisymmetric=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if len(args.levels) != len(args.reg_iterations):
        raise ValueError("--levels and --reg-iterations must have the same number of values")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(fixed_ants, str(args.output_dir / "fixed_r30.nii.gz"))
    ants.image_write(moving_ants, str(args.output_dir / "moving_r27.nii.gz"))

    start = time.perf_counter()
    result = syn_registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=args.type_of_transform,
        affine_transform_type=args.affine_transform_type,
        affine_similarity=args.affine_similarity,
        affine_neighborhood_radius=args.affine_neighborhood_radius,
        affine_shrink_factors=tuple(args.affine_shrink_factors),
        affine_smoothing_sigmas=tuple(args.affine_smoothing_sigmas),
        affine_iterations=tuple(args.affine_iterations),
        affine_learning_rate=tuple(args.affine_learning_rate),
        affine_multi_start=not args.affine_single_start,
        affine_center_of_mass_init=not args.affine_no_center_of_mass_init,
        syn_metric=args.syn_metric,
        neighborhood_radius=args.neighborhood_radius,
        num_bins=args.num_bins,
        levels=tuple(args.levels),
        reg_iterations=tuple(args.reg_iterations),
        grad_step=args.grad_step,
        flow_sigma=args.flow_sigma,
        total_sigma=args.total_sigma,
        regularizer=args.regularizer,
        inverse_method=args.inverse_method,
        in_loop_inverse_steps=args.in_loop_inverse_steps,
        antisymmetric=args.antisymmetric,
        # A trailing "/" makes syn_registration write its transform files
        # directly into --output-dir, under the same 0GenericAffine.mat /
        # 1Warp.nii.gz / 1InverseWarp.nii.gz naming ants.registration() uses
        # -- no separate write step needed, unlike the B-spline script's
        # manual write_affine_transform() call.
        outprefix=str(args.output_dir) + "/",
        device=args.device,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start

    ants.image_write(result["warpedmovout"], str(args.output_dir / "warped_r27.nii.gz"))
    if result.get("warpedfixout") is not None:
        # Specific to SyN's symmetric formulation -- fixed pulled back onto
        # the moving grid by the inverse transform. No B-spline-script analog.
        ants.image_write(result["warpedfixout"], str(args.output_dir / "warped_fixed_r30.nii.gz"))
    if result["jacobian"] is not None:
        ants.image_write(result["jacobian"], str(args.output_dir / "jacobian.nii.gz"))

    if result["loss_history"] is not None:
        with (args.output_dir / "loss_history.json").open("w", encoding="utf-8") as stream:
            json.dump(result["level_loss_history"], stream, indent=2)

    if result["affine_matrix"] is not None:
        # 0GenericAffine.mat was already written by syn_registration itself
        # (via outprefix, above); this raw tensor save is only for a direct
        # PyTorch reload, matching the B-spline script's affine_transform.pt.
        torch.save(
            {"matrix": result["affine_matrix"], "translation": result["affine_translation"]},
            args.output_dir / "affine_transform.pt",
        )
    if result["affine_loss_history"] is not None:
        with (args.output_dir / "affine_loss_history.json").open("w", encoding="utf-8") as stream:
            json.dump(result["affine_level_loss_history"], stream, indent=2)

    print(f"Registration completed in {elapsed:.2f} seconds on {result['provenance']['device']}.")
    print(f"Type of transform: {args.type_of_transform}")
    if result["loss_history"]:
        print(f"Final SyN loss: {result['loss_history'][-1]:.6g}")
    if result["affine_loss_history"] is not None:
        # affine_loss_history is batched (one history per batch item); a
        # single ants.ANTsImage pair is always batch size 1.
        print(f"Final affine loss: {result['affine_loss_history'][0][-1]:.6g}")
    print(f"Transform files: fwdtransforms={result['fwdtransforms']}, invtransforms={result['invtransforms']}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
