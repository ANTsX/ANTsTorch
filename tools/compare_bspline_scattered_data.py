#!/usr/bin/env python3
"""Compare ANTs and differentiable ANTsTorch scattered-data B-spline fitting.

Compares ``antstorch.fit_bspline_object_to_scattered_data`` and
``antstorch.fit_bspline_displacement_field`` against their ANTsPy
counterparts (``ants.fit_bspline_object_to_scattered_data`` /
``ants.fit_bspline_displacement_field``), the same comparison methodology
as ``tools/compare_n4_bias_field_correction.py``.

Three independent comparisons run by default:

1. ``fit_bspline_object_to_scattered_data`` on scalar data -- either an
   image's own pixel grid (every pixel a point) or a random subsample of
   it (genuinely scattered, off-grid points).
2. ``fit_bspline_displacement_field`` from scattered displacement points
   (random locations and random displacement vectors).
3. ``fit_bspline_displacement_field`` from a dense displacement field
   (best-effort: some ANTsPy builds do not export the underlying
   ``fitBsplineDisplacementFieldD*`` library symbol for this input mode --
   if so, this comparison is skipped with a clear message instead of
   crashing the whole run).

Both implementations return this package's ``(N, C, *reversed(size))``
tensor convention, which is the ANTsPy/ITK array with every spatial axis
reversed (see ``fit_bspline_object_to_scattered_data``'s docstring) --
this script accounts for that before comparing.

Examples
--------
Synthetic scattered data, no image or network access required::

    python tools/compare_bspline_scattered_data.py

Use the bundled 2-D ``r16`` image's pixel grid::

    python tools/compare_bspline_scattered_data.py --image r16

Use 2000 random (off-grid) pixels from an image as scattered points::

    python tools/compare_bspline_scattered_data.py --image r16 --num-points 2000

3-D synthetic data with a finer mesh and CUDA::

    python tools/compare_bspline_scattered_data.py --dimension 3 --mesh-size 3 3 3 --device cuda
"""

import argparse
import time
from pathlib import Path

import ants
import numpy as np
import torch

import antstorch


def reverse_axes(array: np.ndarray) -> np.ndarray:
    """This package's ``(N, C, *reversed(size))`` convention <-> ANTsPy/ITK's
    direct ``(size_x, size_y[, size_z])`` array order: a full reversal of
    every spatial axis (``.T`` in 2-D; not a literal ``.T`` in 3-D). See
    ``fit_bspline_object_to_scattered_data``'s docstring. Only valid for a
    purely spatial (scalar) array -- use ``vector_to_ants_order`` for a
    ``(D, *spatial)`` vector field, where the leading component axis must
    NOT be reversed along with the spatial ones.
    """
    return np.ascontiguousarray(np.transpose(array, tuple(range(array.ndim - 1, -1, -1))))


def vector_to_ants_order(array: np.ndarray) -> np.ndarray:
    """``(D, *torch_reversed_spatial)`` -> ``(*ants_direct_spatial, D)``:
    reverse only the spatial axes (leave axis 0, the vector component axis,
    in place) then move it to the end, matching
    ``ants.from_numpy(..., has_components=True)``'s array order.
    """
    spatial_axes_reversed = (0,) + tuple(range(array.ndim - 1, 0, -1))
    reordered = np.transpose(array, spatial_axes_reversed)
    return np.ascontiguousarray(np.moveaxis(reordered, 0, -1))


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def report(label: str, ants_array: np.ndarray, torch_array: np.ndarray) -> None:
    """Print RMSE/MAE/correlation between an ANTsPy array (direct order)
    and this package's dense output (already reordered to match).
    """
    difference = torch_array - ants_array
    rmse = np.sqrt(np.mean(difference**2))
    mae = np.mean(np.abs(difference))
    correlation = np.corrcoef(ants_array.ravel(), torch_array.ravel())[0, 1]
    print(f"  {label}: RMSE={rmse:.6g}  MAE={mae:.6g}  correlation={correlation:.8f}")


def load_image_or_none(name: str):
    if name is None:
        return None
    path = Path(name)
    if not path.exists() and name in ("r16", "r27", "r62", "r64", "r85"):
        name = ants.get_ants_data(name)
    return ants.image_read(str(name)).clone("double")


def grid_parametric_points(size):
    """Every index of a ``size``-shaped (ITK order) grid, as ``(P, dim)``
    points -- ``indexing="ij"`` keeps flatten order consistent with a
    ``size``-shaped NumPy array's own ``.ravel()`` order, so ``arr.reshape(-1,
    1)`` and this function's output describe the same points in the same
    order.
    """
    axes = [np.arange(s) for s in size]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh]).astype(np.float64)


def compare_object_to_scattered_data(args, rng: np.random.Generator, device: torch.device) -> None:
    print("\n=== fit_bspline_object_to_scattered_data (scalar) ===")
    image = load_image_or_none(args.image)
    if image is not None:
        size = tuple(int(v) for v in image.shape)
        values = image.numpy().astype(np.float64)
        origin = tuple(float(v) for v in image.origin)
        spacing = tuple(float(v) for v in image.spacing)
    else:
        size = tuple(args.size[: args.dimension]) if args.size else (48, 40, 32)[: args.dimension]
        axes = [np.linspace(-1, 1, s) for s in size]
        mesh = np.meshgrid(*axes, indexing="ij")
        values = np.sin(2.0 * mesh[0]) * np.cos(2.0 * mesh[1])
        if args.dimension == 3:
            values = values * np.cos(1.5 * mesh[2])
        origin = (0.0,) * args.dimension
        spacing = (1.0,) * args.dimension
    dimension = len(size)

    parametric_full = grid_parametric_points(size)
    scattered_full = values.reshape(-1, 1)
    if args.num_points and args.num_points < scattered_full.shape[0]:
        # Genuinely scattered (off-grid) points: sample continuous
        # parametric locations, not just a subset of the grid indices.
        points = rng.uniform(0.0, np.array(size, dtype=np.float64) - 1.0, size=(args.num_points, dimension))
        # Nearest-neighbor lookup into the dense grid gives a well-defined
        # scalar value at each off-grid parametric location without
        # requiring interpolation machinery in this comparison script.
        indices = np.rint(points).astype(np.int64)
        for d in range(dimension):
            indices[:, d] = np.clip(indices[:, d], 0, size[d] - 1)
        flat_indices = np.ravel_multi_index(indices.T, size)
        scattered_data = scattered_full[flat_indices]
        parametric_data = points
    else:
        scattered_data = scattered_full
        parametric_data = parametric_full

    weights = rng.uniform(0.5, 1.5, size=scattered_data.shape[0]) if args.random_weights else None

    kwargs = dict(
        parametric_domain_origin=list(origin),
        parametric_domain_spacing=list(spacing),
        parametric_domain_size=list(size),
        number_of_fitting_levels=args.number_of_fitting_levels,
        mesh_size=args.mesh_size[:dimension] if args.mesh_size else 4,
    )

    start = time.perf_counter()
    ants_result = ants.fit_bspline_object_to_scattered_data(scattered_data, parametric_data, data_weights=weights, **kwargs)
    ants_seconds = time.perf_counter() - start
    ants_array = ants_result.numpy().astype(np.float64)

    torch_device = torch.device(args.device)
    synchronize(torch_device)
    start = time.perf_counter()
    torch_dense = antstorch.fit_bspline_object_to_scattered_data(
        scattered_data, parametric_data, data_weights=weights, device=torch_device, dtype=torch.float64, **kwargs
    )
    synchronize(torch_device)
    torch_seconds = time.perf_counter() - start
    torch_array = reverse_axes(torch_dense[0, 0].detach().cpu().numpy())

    print(f"  points={scattered_data.shape[0]}, size={size}, mesh_size={kwargs['mesh_size']}, levels={kwargs['number_of_fitting_levels']}")
    print(f"  ANTs runtime:      {ants_seconds:.3f} s")
    print(f"  ANTsTorch runtime: {torch_seconds:.3f} s on {torch_device}")
    report("scattered fit", ants_array, torch_array)

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        ants.image_write(ants_result, f"{prefix}_ants_scattered_fit.nii.gz")
        ants.image_write(ants.from_numpy(torch_array, origin=origin, spacing=spacing), f"{prefix}_antstorch_scattered_fit.nii.gz")


def compare_displacement_field_from_points(args, rng: np.random.Generator, device: torch.device) -> None:
    print("\n=== fit_bspline_displacement_field (scattered points) ===")
    dimension = args.dimension
    size = tuple(args.size[:dimension]) if args.size else (60, 45, 32)[:dimension]
    point_count = args.num_points or 200
    margin = 5.0
    lower = np.full(dimension, margin)
    upper = np.array(size, dtype=np.float64) - 1.0 - margin
    points = rng.uniform(lower, upper, size=(point_count, dimension))
    deltas = rng.normal(scale=3.0, size=(point_count, dimension))
    weights = rng.uniform(0.5, 1.5, size=point_count) if args.random_weights else None

    if args.enforce_stationary_boundary:
        print(
            "  NOTE: --enforce-stationary-boundary is on. ANTsTorch applies it as a "
            "post-hoc edge mask, while ITK adds high-weight zero observations during "
            "the fit -- a documented simplification (see fit_bspline_displacement_field's "
            "docstring). Expect a larger disagreement than with it off; pass "
            "--no-enforce-stationary-boundary to validate the fit itself."
        )

    kwargs = dict(
        number_of_fitting_levels=args.number_of_fitting_levels,
        mesh_size=args.mesh_size[:dimension] if args.mesh_size else 1,
        enforce_stationary_boundary=args.enforce_stationary_boundary,
    )

    start = time.perf_counter()
    ants_result = ants.fit_bspline_displacement_field(
        displacement_origins=points,
        displacements=deltas,
        displacement_weights=weights,
        origin=[0.0] * dimension,
        spacing=[1.0] * dimension,
        size=list(size),
        direction=np.eye(dimension),
        **kwargs,
    )
    ants_seconds = time.perf_counter() - start
    ants_array = ants_result.numpy().astype(np.float64)

    torch_device = torch.device(args.device)
    domain = antstorch.ImageDomain(size=size, spacing=(1.0,) * dimension, origin=(0.0,) * dimension)
    synchronize(torch_device)
    start = time.perf_counter()
    torch_dense = antstorch.fit_bspline_displacement_field(
        displacement_origins=torch.as_tensor(points, device=torch_device),
        displacements=torch.as_tensor(deltas, device=torch_device),
        displacement_weights=None if weights is None else torch.as_tensor(weights, device=torch_device),
        domain=domain,
        **kwargs,
    )
    synchronize(torch_device)
    torch_seconds = time.perf_counter() - start
    torch_array = vector_to_ants_order(torch_dense[0].detach().cpu().numpy())

    print(f"  points={point_count}, size={size}, mesh_size={kwargs['mesh_size']}, levels={kwargs['number_of_fitting_levels']}")
    print(f"  ANTs runtime:      {ants_seconds:.3f} s")
    print(f"  ANTsTorch runtime: {torch_seconds:.3f} s on {torch_device}")
    for component in range(dimension):
        report(f"component {component}", ants_array[..., component], torch_array[..., component])

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        ants.image_write(ants_result, f"{prefix}_ants_displacement_points.nii.gz")
        ants.image_write(
            ants.from_numpy(torch_array, origin=[0.0] * dimension, spacing=[1.0] * dimension, has_components=True),
            f"{prefix}_antstorch_displacement_points.nii.gz",
        )


def compare_displacement_field_from_dense_field(args, rng: np.random.Generator, device: torch.device) -> None:
    print("\n=== fit_bspline_displacement_field (dense field) ===")
    dimension = args.dimension
    size = tuple(args.size[:dimension]) if args.size else (40, 32, 24)[:dimension]
    # ants.from_numpy(..., has_components=True) expects the array shaped
    # directly in ITK order (size_x, size_y[, size_z], components).
    field = rng.normal(scale=0.1, size=size + (dimension,))

    kwargs = dict(
        number_of_fitting_levels=args.number_of_fitting_levels,
        mesh_size=args.mesh_size[:dimension] if args.mesh_size else 1,
        enforce_stationary_boundary=args.enforce_stationary_boundary,
    )

    try:
        ants_field_image = ants.from_numpy(field, spacing=(1.0,) * dimension, has_components=True)
        start = time.perf_counter()
        ants_result = ants.fit_bspline_displacement_field(
            displacement_field=ants_field_image,
            origin=[0.0] * dimension,
            spacing=[1.0] * dimension,
            size=list(size),
            direction=np.eye(dimension),
            **kwargs,
        )
        ants_seconds = time.perf_counter() - start
    except (RuntimeError, AttributeError) as error:
        print(f"  SKIPPED: this ANTsPy build could not run the dense-field path ({error}).")
        return
    ants_array = ants_result.numpy().astype(np.float64)

    torch_device = torch.device(args.device)
    # ANTsPy stores vector fields as ``(*size, D)`` in direct ITK order,
    # whereas ANTsTorch uses ``(N, D, *reversed(size))``.
    component_first_reversed = np.transpose(
        field, (field.ndim - 1,) + tuple(range(field.ndim - 2, -1, -1))
    )
    field_torch = torch.as_tensor(
        np.ascontiguousarray(component_first_reversed), device=torch_device
    ).unsqueeze(0)
    domain = antstorch.ImageDomain(size=size, spacing=(1.0,) * dimension, origin=(0.0,) * dimension)
    synchronize(torch_device)
    start = time.perf_counter()
    torch_dense = antstorch.fit_bspline_displacement_field(displacement_field=field_torch, domain=domain, **kwargs)
    synchronize(torch_device)
    torch_seconds = time.perf_counter() - start
    torch_array = vector_to_ants_order(torch_dense[0].detach().cpu().numpy())

    print(f"  size={size}, mesh_size={kwargs['mesh_size']}, levels={kwargs['number_of_fitting_levels']}")
    print(f"  ANTs runtime:      {ants_seconds:.3f} s")
    print(f"  ANTsTorch runtime: {torch_seconds:.3f} s on {torch_device}")
    for component in range(dimension):
        report(f"component {component}", ants_array[..., component], torch_array[..., component])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="Image path, or a bundled ANTsPy test image name (e.g. r16). Omit for synthetic data.")
    parser.add_argument("--dimension", type=int, default=2, choices=(2, 3), help="Dimension for synthetic data (ignored when --image is given).")
    parser.add_argument("--size", type=int, nargs="+", default=None, help="Domain size for synthetic data, ITK order.")
    parser.add_argument("--num-points", type=int, default=None, help="Random/off-grid point count (comparison 1) or scattered point count (comparison 2). Default: full grid for comparison 1, 200 for comparison 2.")
    parser.add_argument("--mesh-size", type=int, nargs="+", default=None, help="B-spline mesh size, ITK order (default: 4 for comparison 1, 1 for comparisons 2-3).")
    parser.add_argument("--number-of-fitting-levels", type=int, default=4)
    parser.add_argument("--random-weights", action="store_true", help="Use random per-point weights instead of uniform.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--enforce-stationary-boundary", dest="enforce_stationary_boundary", action="store_true", default=False)
    parser.add_argument("--no-enforce-stationary-boundary", dest="enforce_stationary_boundary", action="store_false")
    parser.add_argument("--skip-object-to-scattered-data", action="store_true")
    parser.add_argument("--skip-displacement-points", action="store_true")
    parser.add_argument("--skip-displacement-dense-field", action="store_true")
    parser.add_argument("--output-prefix", default="bspline_scattered_comparison", help="Prefix for written .nii.gz outputs, or '' to skip writing files.")
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available")
    rng = np.random.default_rng(args.seed)

    if not args.skip_object_to_scattered_data:
        compare_object_to_scattered_data(args, rng, device)
    if not args.skip_displacement_points:
        compare_displacement_field_from_points(args, rng, device)
    if not args.skip_displacement_dense_field:
        compare_displacement_field_from_dense_field(args, rng, device)


if __name__ == "__main__":
    main()
