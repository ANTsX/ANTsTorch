#!/usr/bin/env python3
"""Run ANTsTorch greedy (one-directional, diffeomorphic) registration on 2-D or 3-D ANTs images.

Sibling of ``run_syn_registration.py`` and ``run_svf_registration.py``, with the
same CLI shape and output conventions, using :func:`antstorch.greedy.greedy_registration`
(the benchmark's ``greedy`` and ``greedy_regadam`` arms). By default it registers
the bundled ANTs ``r30``/``r27`` demo images; pass ``--fixed``/``--moving`` for your own.

Differences from the SyN script, which follow from the method:

- The registration is one-directional: there is no inverse unless ``--return-inverse``
  is passed (Anderson fixed-point solve, at least 15 steps).
- The affine initialization is ``antstorch.syn.robust_affine`` by default
  (``--affine-mode``); ``--no-affine`` skips it and ``--initial-transform`` supplies
  ANTs linear transform file(s) instead (for example a shared affine from another arm).
- Options left unset use the library defaults (those of ``syntx.greedy``: 3-D iterations
  100 100 20, learning rate 0.375, flow sigma 1.8, total sigma 0.28, metric cc2).

Example
-------
Default greedy registration (Adam optimizer) on CPU::

    PYTHONPATH=. python tools/registration/run_greedy_registration.py

The ``greedy_regadam`` variant (Adam quotient smoothed by ``--regadam-sigma``)::

    PYTHONPATH=. python tools/registration/run_greedy_registration.py \\
        --optimizer regadam --device cuda --verbose

Your own images, with the inverse, reusing an affine fitted elsewhere::

    PYTHONPATH=. python tools/registration/run_greedy_registration.py \\
        --fixed fixed.nii.gz --moving moving.nii.gz --return-inverse \\
        --initial-transform 0GenericAffine.mat --output-dir greedy_output
"""

import argparse
import time
from pathlib import Path

import torch  # noqa: F401  (keep BEFORE `import ants`: OpenMP runtime load order, see tests/conftest.py)
import ants
import torch

from antstorch.greedy import greedy_registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", type=Path, default=None,
                        help="Path to the fixed image (default: the bundled ANTs 'r30' demo image)")
    parser.add_argument("--moving", type=Path, default=None,
                        help="Path to the moving image (default: the bundled ANTs 'r27' demo image)")
    parser.add_argument("--device", default=None, help="PyTorch device: cpu, cuda, or mps (default: CUDA, else MPS, else CPU)")
    parser.add_argument("--output-dir", type=Path, default=Path("greedy_registration_output"))
    parser.add_argument("--optimizer", choices=("adam", "regadam"), default="adam",
                        help="'adam' (benchmark arm 'greedy') or 'regadam' (arm 'greedy_regadam')")
    parser.add_argument("--regadam-sigma", type=float, default=None, help="regadam: Adam-quotient smoothing sigma in voxels (default 0.8)")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Shrink factors, coarse to fine (default: 2**(L-1), ..., 1)")
    parser.add_argument("--reg-iterations", type=int, nargs="+", default=None,
                        help="Iterations per level (default: 100 100 20 in 3-D, 100 100 100 50 in 2-D)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Step size (default 0.375)")
    parser.add_argument("--flow-sigma", type=float, default=None, help="Gaussian sigma (voxels) smoothing the gradient (default 1.8)")
    parser.add_argument("--total-sigma", type=float, default=None, help="Gaussian sigma (voxels) smoothing the whole field (default 0.28)")
    parser.add_argument("--similarity", choices=("cc2", "lncc", "cc", "ncc", "mse", "l2"), default=None,
                        help="Similarity metric (default cc2)")
    parser.add_argument("--lncc-radius", type=int, default=None, help="Window radius for the local correlation metrics (default 2)")
    parser.add_argument("--return-inverse", action="store_true", help="Also compute the inverse field (no inverse otherwise)")
    parser.add_argument("--anderson", action="store_true", help="Optional Anderson inverse-consistency projection")
    parser.add_argument("--affine-mode", default=None, help="antstorch.syn.robust_affine mode for the initial affine (default 'auto')")
    parser.add_argument("--no-affine", action="store_true", help="Skip the affine initialization (identity)")
    parser.add_argument("--initial-transform", type=str, nargs="+", default=None,
                        help="ANTs linear transform file(s) to use as the initial affine instead of robust_affine")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print per-level optimization progress")
    args = parser.parse_args()
    if args.no_affine and args.initial_transform:
        parser.error("--no-affine and --initial-transform are mutually exclusive")
    if args.levels and args.reg_iterations and len(args.levels) != len(args.reg_iterations):
        parser.error("--levels and --reg-iterations must have the same number of values")
    return args


def main() -> None:
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")

    fixed_path = str(args.fixed) if args.fixed is not None else ants.get_ants_data("r30")
    moving_path = str(args.moving) if args.moving is not None else ants.get_ants_data("r27")
    fixed_ants = ants.image_read(fixed_path).clone("float")
    moving_ants = ants.image_read(moving_path).clone("float")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(fixed_ants, str(args.output_dir / "fixed.nii.gz"))
    ants.image_write(moving_ants, str(args.output_dir / "moving.nii.gz"))

    # Only the options that were given are forwarded, so unset ones keep the library defaults.
    optional = {
        "levels": tuple(args.levels) if args.levels else None,
        "reg_iterations": tuple(args.reg_iterations) if args.reg_iterations else None,
        "learning_rate": args.learning_rate,
        "flow_sigma": args.flow_sigma,
        "total_sigma": args.total_sigma,
        "regadam_sigma": args.regadam_sigma,
        "similarity_metric": args.similarity,
        "lncc_radius": args.lncc_radius,
        "affine_mode": args.affine_mode,
        "seed": args.seed,
        "device": args.device,
    }
    kwargs = {k: v for k, v in optional.items() if v is not None}
    if args.no_affine:
        kwargs["initial_transform"] = False
    elif args.initial_transform:
        kwargs["initial_transform"] = args.initial_transform if len(args.initial_transform) > 1 else args.initial_transform[0]

    start = time.perf_counter()
    result = greedy_registration(
        fixed_ants,
        moving_ants,
        optimizer=args.optimizer,
        anderson=args.anderson,
        return_inverse=args.return_inverse,
        # A trailing "/" writes 1Warp.nii.gz / 0GenericAffine.mat (and 1InverseWarp.nii.gz) into --output-dir,
        # the naming ants.registration() and run_syn_registration.py use.
        outprefix=str(args.output_dir) + "/",
        verbose=args.verbose,
        **kwargs,
    )
    elapsed = time.perf_counter() - start

    ants.image_write(result["warpedmovout"], str(args.output_dir / "warped_moving.nii.gz"))
    if result.get("warpedfixout") is not None:
        ants.image_write(result["warpedfixout"], str(args.output_dir / "warped_fixed.nii.gz"))

    print(f"Registration completed in {elapsed:.2f} seconds.")
    print(f"Optimizer: {args.optimizer}")
    print(f"Transform files: fwdtransforms={result['fwdtransforms']}, invtransforms={result['invtransforms']}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
