"""High-level optimization interface for B-spline SVF registration."""

from math import isfinite, prod
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .bspline_domain import BSplineDomain
from .bspline_synthesis import refine_bspline_coefficients
from .deterministic_registration import DeterministicBSplineRegistration
from .physical_gradient_descent import PhysicalGradientDescent


def _axis_values(value, dimension: int, name: str, *, minimum: int) -> tuple:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or a sequence of integers")
    if isinstance(value, int):
        values = (value,) * dimension
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise TypeError(f"{name} must be an integer or a sequence of integers") from error
        if len(values) != dimension:
            raise ValueError(f"{name} must have {dimension} values")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise TypeError(f"{name} values must be integers")
    if any(v < minimum for v in values):
        raise ValueError(f"{name} values must be at least {minimum}")
    return values


def _closed_axes(closed, dimension: int) -> tuple:
    if isinstance(closed, bool):
        return (closed,) * dimension
    values = tuple(bool(value) for value in closed)
    if len(values) != dimension:
        raise ValueError(f"closed must have {dimension} values")
    return values


def _validate_images(
    fixed: Tensor,
    moving: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: BSplineDomain,
) -> None:
    if not isinstance(fixed, Tensor) or not isinstance(moving, Tensor):
        raise TypeError("fixed and moving must be torch tensors")
    if fixed.ndim != fixed_domain.dimension + 2 or tuple(fixed.shape[2:]) != fixed_domain.torch_size:
        raise ValueError("fixed tensor shape does not match fixed_domain")
    if moving.ndim != moving_domain.dimension + 2 or tuple(moving.shape[2:]) != moving_domain.torch_size:
        raise ValueError("moving tensor shape does not match moving_domain")
    if fixed.shape[:2] != moving.shape[:2]:
        raise ValueError("fixed and moving must have identical batch and channel sizes")
    if not fixed.is_floating_point() or not moving.is_floating_point():
        raise TypeError("fixed and moving must have floating-point dtypes")
    if fixed.dtype != moving.dtype or fixed.device != moving.device:
        raise ValueError("fixed and moving must have the same dtype and device")


def _level_values(value, levels: int, name: str, cast, minimum) -> tuple:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = (cast(value),) * levels
    else:
        try:
            values = tuple(cast(item) for item in value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar or a sequence") from error
        if len(values) != levels:
            raise ValueError(f"{name} must have one value per resolution level")
    if any(not isfinite(item) or item < minimum for item in values):
        raise ValueError(f"{name} values must be finite and at least {minimum}")
    return values


def _pyramid_configuration(shrink_factors, smoothing_sigmas, iterations, learning_rate):
    try:
        factors = tuple(shrink_factors)
    except TypeError as error:
        raise TypeError("shrink_factors must be a sequence of positive integers") from error
    if not factors or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in factors):
        raise ValueError("shrink_factors must contain positive integers")
    if factors[-1] != 1 or any(coarse != 2 * fine for coarse, fine in zip(factors, factors[1:])):
        raise ValueError("shrink_factors must be a dyadic coarse-to-fine sequence ending in 1")
    levels = len(factors)
    sigmas = _level_values(
        tuple(0.0 for _ in factors) if smoothing_sigmas is None else smoothing_sigmas,
        levels,
        "smoothing_sigmas",
        float,
        0.0,
    )
    iteration_inputs = (iterations,) * levels if isinstance(iterations, int) else tuple(iterations)
    if len(iteration_inputs) != levels:
        raise ValueError("iterations must have one value per resolution level")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in iteration_inputs):
        raise ValueError("iterations values must be non-negative integers")
    level_iterations = iteration_inputs
    level_rates = _level_values(learning_rate, levels, "learning_rate", float, 0.0)
    if any(rate == 0 for rate in level_rates):
        raise ValueError("learning_rate values must be positive")
    return factors, sigmas, level_iterations, level_rates


def _smooth_image(image: Tensor, domain: BSplineDomain, sigma: float) -> Tensor:
    """Gaussian smoothing with ``sigma`` in physical domain units."""
    if sigma == 0:
        return image
    sigma_torch = tuple(reversed(tuple(sigma / spacing for spacing in domain.spacing)))
    axes = []
    radii = []
    for axis_sigma in sigma_torch:
        radius = max(1, int(3.0 * axis_sigma + 0.5))
        radii.append(radius)
        coordinate = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
        axis_kernel = torch.exp(-0.5 * (coordinate / axis_sigma).square())
        axes.append(axis_kernel / axis_kernel.sum())
    kernel = axes[0]
    for axis_kernel in axes[1:]:
        kernel = kernel.unsqueeze(-1) * axis_kernel.reshape((1,) * kernel.ndim + (-1,))
    kernel = kernel.reshape((1, 1) + kernel.shape).expand((image.shape[1], 1) + kernel.shape)
    padding = tuple(value for radius in reversed(radii) for value in (radius, radius))
    padded = F.pad(image, padding, mode="replicate")
    convolution = F.conv2d if domain.dimension == 2 else F.conv3d
    return convolution(padded, kernel, groups=image.shape[1])


def _downsample(image: Tensor, domain: BSplineDomain, factor: int):
    if factor == 1:
        return image, domain
    size_itk = tuple(max(2, (size - 1) // factor + 1) for size in domain.size)
    spacing = tuple(extent / (size - 1) for extent, size in zip(domain.physical_extent, size_itk))
    reduced_domain = BSplineDomain(size_itk, spacing, domain.origin, domain.direction)
    mode = "bilinear" if domain.dimension == 2 else "trilinear"
    reduced = F.interpolate(image, size=reduced_domain.torch_size, mode=mode, align_corners=True)
    return reduced, reduced_domain


def registration(
    fixed: Tensor,
    moving: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: Optional[BSplineDomain] = None,
    *,
    mesh_size: Union[int, Sequence[int]] = 1,
    coefficient_grid_size: Optional[Union[int, Sequence[int]]] = None,
    iterations: Union[int, Sequence[int]] = 100,
    learning_rate: Union[float, Sequence[float]] = 1e-2,
    optimizer: Union[str, PhysicalGradientDescent] = "physical_gradient_descent",
    gradient_step: float = 0.2,
    momentum: float = 0.0,
    gradient_smoothing_sigma: float = 0.0,
    similarity: str = "mse",
    neighborhood_radius: Union[int, Sequence[int]] = 2,
    coefficient_weight: float = 0.0,
    velocity_weight: float = 0.0,
    bending_weight: float = 0.0,
    squaring_steps: int = 7,
    padding_mode: str = "zeros",
    closed=False,
    stationary_boundary: bool = True,
    convergence_tolerance: Optional[float] = None,
    return_loss_history: bool = True,
    initial_coefficients: Optional[Tensor] = None,
    shrink_factors: Sequence[int] = (1,),
    smoothing_sigmas: Optional[Union[float, Sequence[float]]] = None,
    verbose: bool = False,
    detach_outputs: bool = True,
    synthesis_chunk_size: Optional[int] = 262144,
) -> Dict[str, object]:
    """Optimize a cubic B-spline stationary-velocity transformation.

    Images have shape ``(N, C, Y, X)`` or ``(N, C, Z, Y, X)``. Domain,
    mesh, and coefficient-grid tuples use physical ITK axis order
    ``(x, y[, z])``; tensor spatial axes are reversed. Velocity and
    displacement channels are physical x, y, (z) components and their values
    are in the physical units of the domains. Batches of any positive size are
    supported, with one coefficient lattice optimized per batch item.

    ``similarity="ants_ncc"`` selects the squared local neighborhood
    correlation used by ITK/ANTs. ``neighborhood_radius`` is an integer or an
    ITK-order ``(x, y[, z])`` tuple; its default of 2 matches ITK.

    ``optimizer="physical_gradient_descent"`` normalizes each batch item's
    coefficient-gradient direction after B-spline synthesis so that the
    maximum dense velocity update has physical magnitude ``gradient_step``
    times the current level's voxel diagonal. ``gradient_step`` must be in
    ``[0.1, 0.25]``. ``learning_rate`` is ignored by this optimizer.
    A configured :class:`PhysicalGradientDescent` instance can be supplied
    instead to enable coefficient-gradient momentum and physical smoothing.

    ``mesh_size`` is the number of B-spline spans. For open axes the
    coefficient-grid size is ``mesh_size + 3``; for closed axes it equals the
    mesh size. ``coefficient_grid_size`` can instead specify the control-point
    count directly. Supplying ``initial_coefficients`` makes its spatial shape
    authoritative and cannot be combined with ``coefficient_grid_size``.

    ``shrink_factors`` defines a dyadic coarse-to-fine pyramid ending in 1,
    for example ``(4, 2, 1)``. ``smoothing_sigmas`` are Gaussian sigmas in
    physical units. ``iterations`` and ``learning_rate`` may be scalars or
    contain one value per level. Reduced images retain their physical extent,
    and the open B-spline lattice is refined exactly between levels.
    Multi-resolution registration of closed axes is not yet supported.

    Set ``verbose=True`` to report each pyramid level and optimization loss.

    By default result tensors are detached, and iteration losses are stored as
    Python floats so optimization graphs are not retained. Set
    ``detach_outputs=False`` to retain the graph for the final evaluation only.
    The inverse transform is independently computed as ``exp(-velocity)``.

    Returns
    -------
    dict
        ``warpedmovout``, ``fwdtransforms``, and ``invtransforms`` follow the
        naming convention of ``ants.registration``: the moving image warped
        onto the fixed grid, the fixed-to-moving displacement field, and the
        moving-to-fixed displacement field, respectively. Unlike
        ``ants.registration``, these are in-memory tensors rather than paths
        to files on disk. The dictionary also carries fields with no
        ``ants.registration`` equivalent: ``velocity`` (the stationary
        velocity field), ``coefficients`` (the optimized B-spline control
        points), ``loss``, ``similarity``, ``coefficient_regularization``,
        ``velocity_regularization``, ``bending_regularization``,
        ``jacobian_determinant``, ``loss_history``, and
        ``level_loss_history``.
    """
    if not isinstance(fixed_domain, BSplineDomain):
        raise TypeError("fixed_domain must be a BSplineDomain")
    moving_domain = fixed_domain if moving_domain is None else moving_domain
    if not isinstance(moving_domain, BSplineDomain):
        raise TypeError("moving_domain must be a BSplineDomain")
    if fixed_domain.dimension != moving_domain.dimension:
        raise ValueError("fixed_domain and moving_domain must have the same dimension")
    _validate_images(fixed, moving, fixed_domain, moving_domain)
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    factors, sigmas, level_iterations, level_rates = _pyramid_configuration(
        shrink_factors, smoothing_sigmas, iterations, learning_rate
    )
    optimizer_name = "physical_gradient_descent" if isinstance(optimizer, PhysicalGradientDescent) else optimizer
    if optimizer_name not in ("adam", "lbfgs", "physical_gradient_descent"):
        raise ValueError("optimizer must be 'adam', 'lbfgs', or 'physical_gradient_descent'")
    physical_optimizer = (
        optimizer
        if isinstance(optimizer, PhysicalGradientDescent)
        else PhysicalGradientDescent(gradient_step, momentum, gradient_smoothing_sigma)
        if optimizer_name == "physical_gradient_descent"
        else None
    )
    if similarity not in ("mse", "ncc", "ants_ncc"):
        raise ValueError("similarity must be 'mse', 'ncc', or 'ants_ncc'")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    for name, weight in (("coefficient_weight", coefficient_weight), ("velocity_weight", velocity_weight), ("bending_weight", bending_weight)):
        if not isinstance(weight, (int, float)) or not isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if convergence_tolerance is not None and (
        not isinstance(convergence_tolerance, (int, float))
        or not isfinite(convergence_tolerance)
        or convergence_tolerance < 0
    ):
        raise ValueError("convergence_tolerance must be finite and non-negative or None")

    dimension = fixed_domain.dimension
    closed_axes = _closed_axes(closed, dimension)
    if len(factors) > 1 and any(closed_axes):
        raise ValueError("multi-resolution registration does not yet support closed axes")
    if initial_coefficients is not None:
        if coefficient_grid_size is not None:
            raise ValueError("coefficient_grid_size cannot be used with initial_coefficients")
        expected_prefix = (fixed.shape[0], dimension)
        if initial_coefficients.ndim != dimension + 2 or initial_coefficients.shape[:2] != expected_prefix:
            raise ValueError("initial_coefficients has an incompatible batch, vector, or spatial rank")
        if any(size < 4 for size in initial_coefficients.shape[2:]):
            raise ValueError("initial_coefficients requires at least four control points per axis")
        if not initial_coefficients.is_floating_point():
            raise TypeError("initial_coefficients must have a floating-point dtype")
        if initial_coefficients.dtype != fixed.dtype or initial_coefficients.device != fixed.device:
            raise ValueError("initial_coefficients must match the fixed image dtype and device")
        coefficients = initial_coefficients.detach().clone().requires_grad_(True)
    else:
        if coefficient_grid_size is None:
            mesh = _axis_values(mesh_size, dimension, "mesh_size", minimum=1)
            lattice_itk = tuple(m if periodic else m + 3 for m, periodic in zip(mesh, closed_axes))
            if any(periodic and size < 4 for periodic, size in zip(closed_axes, lattice_itk)):
                raise ValueError("closed axes require mesh_size of at least 4")
        else:
            lattice_itk = _axis_values(coefficient_grid_size, dimension, "coefficient_grid_size", minimum=4)
        coefficients = torch.zeros(
            (fixed.shape[0], dimension) + tuple(reversed(lattice_itk)),
            dtype=fixed.dtype,
            device=fixed.device,
            requires_grad=True,
        )

    if verbose:
        print("ANTsTorch B-spline SVF registration configuration:")
        configuration = (
            ("fixed_domain", fixed_domain),
            ("moving_domain", moving_domain),
            ("fixed_shape", tuple(fixed.shape)),
            ("moving_shape", tuple(moving.shape)),
            ("dtype", fixed.dtype),
            ("device", fixed.device),
            ("initial_coefficient_shape", tuple(coefficients.shape)),
            ("mesh_size", mesh_size),
            ("coefficient_grid_size", coefficient_grid_size),
            ("initial_coefficients_provided", initial_coefficients is not None),
            ("shrink_factors", factors),
            ("smoothing_sigmas", sigmas),
            ("iterations", level_iterations),
            ("learning_rate", level_rates),
            ("optimizer", optimizer_name),
            ("gradient_step", physical_optimizer.gradient_step if physical_optimizer else gradient_step),
            ("momentum", physical_optimizer.momentum if physical_optimizer else momentum),
            (
                "gradient_smoothing_sigma",
                physical_optimizer.smoothing_sigma if physical_optimizer else gradient_smoothing_sigma,
            ),
            ("similarity", similarity),
            ("neighborhood_radius", neighborhood_radius),
            ("coefficient_weight", coefficient_weight),
            ("velocity_weight", velocity_weight),
            ("bending_weight", bending_weight),
            ("squaring_steps", squaring_steps),
            ("padding_mode", padding_mode),
            ("closed", closed),
            ("stationary_boundary", stationary_boundary),
            ("convergence_tolerance", convergence_tolerance),
            ("return_loss_history", return_loss_history),
            ("detach_outputs", detach_outputs),
            ("synthesis_chunk_size", synthesis_chunk_size),
        )
        for name, value in configuration:
            print(f"  {name}: {value}")

    history = []
    level_history = []
    for level, (factor, sigma, iteration_count, rate) in enumerate(
        zip(factors, sigmas, level_iterations, level_rates)
    ):
        if level:
            coefficients = refine_bspline_coefficients(coefficients.detach()).requires_grad_(True)
        fixed_level, fixed_level_domain = _downsample(
            _smooth_image(fixed, fixed_domain, sigma), fixed_domain, factor
        )
        moving_level, moving_level_domain = _downsample(
            _smooth_image(moving, moving_domain, sigma), moving_domain, factor
        )
        if verbose:
            voxel_diagonal = sum(spacing**2 for spacing in fixed_level_domain.spacing) ** 0.5
            control_points = tuple(reversed(coefficients.shape[2:]))
            print(
                f"Resolution level {level + 1}/{len(factors)}: "
                f"shrink_factor={factor}, smoothing_sigma={sigma:g}, "
                f"fixed_size={fixed_level_domain.size}, "
                f"moving_size={moving_level_domain.size}, "
                f"control_points={control_points}, "
                f"total_control_points={prod(control_points)}, "
                f"iterations={iteration_count}"
            )
            if optimizer_name == "physical_gradient_descent":
                print(
                    f"  physical_gradient_step={physical_optimizer.gradient_step * voxel_diagonal:.8g} "
                    f"({physical_optimizer.gradient_step:g} * voxel_diagonal {voxel_diagonal:.8g})"
                )
        model = DeterministicBSplineRegistration(
            fixed_level_domain,
            moving_level_domain,
            squaring_steps=squaring_steps,
            similarity=similarity,
            neighborhood_radius=neighborhood_radius,
            padding_mode=padding_mode,
            coefficient_weight=coefficient_weight,
            velocity_weight=velocity_weight,
            bending_weight=bending_weight,
            closed=closed,
            stationary_boundary=stationary_boundary,
            synthesis_chunk_size=synthesis_chunk_size,
        )
        if optimizer_name == "adam":
            optimizer_impl = torch.optim.Adam([coefficients], lr=rate)
        elif optimizer_name == "lbfgs":
            optimizer_impl = torch.optim.LBFGS([coefficients], lr=rate, max_iter=20)
        else:
            optimizer_impl = None
            physical_optimizer.reset()
        current_level_history = []
        previous = None
        for _ in range(iteration_count):
            def closure():
                if optimizer_impl is None:
                    coefficients.grad = None
                else:
                    optimizer_impl.zero_grad()
                loss = model(coefficients, moving_level, fixed_level)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite registration loss at resolution level {level + 1}"
                    )
                loss.backward()
                if coefficients.grad is None or not torch.isfinite(coefficients.grad).all():
                    raise FloatingPointError(
                        f"non-finite coefficient gradient at resolution level {level + 1}"
                    )
                return loss

            if optimizer_name == "physical_gradient_descent":
                closure()
                physical_optimizer.step(
                    coefficients, model.synthesis, fixed_level_domain, closed=closed
                )
            elif optimizer_name == "lbfgs":
                optimizer_impl.step(closure)
            else:
                closure()
                optimizer_impl.step()
            with torch.no_grad():
                current = float(model(coefficients, moving_level, fixed_level)["loss"].item())
            current_level_history.append(current)
            if verbose:
                print(f"  iteration {len(current_level_history):04d}: loss={current:.8g}")
            if previous is not None and convergence_tolerance is not None:
                if abs(previous - current) <= convergence_tolerance * max(1.0, abs(previous)):
                    if verbose:
                        print(
                            "  convergence reached: "
                            f"change={abs(previous - current):.8g}, "
                            f"tolerance={convergence_tolerance:.8g}"
                        )
                    break
            previous = current
        level_history.append(current_level_history)
        history.extend(current_level_history)

    # The final pyramid level is full resolution by construction.
    result = model(coefficients, moving, fixed)
    result["warpedmovout"] = result.pop("warped_moving")
    result["fwdtransforms"] = result.pop("displacement")
    result["invtransforms"] = model.exponential(-result["velocity"])
    result["coefficients"] = coefficients
    result["loss_history"] = history if return_loss_history else None
    result["level_loss_history"] = level_history if return_loss_history else None
    if detach_outputs:
        result = {key: value.detach() if isinstance(value, Tensor) else value for key, value in result.items()}
    return result
