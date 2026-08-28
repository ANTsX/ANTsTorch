"""Dense Gaussian-regularized stationary-velocity registration."""

from math import isfinite
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .bspline_domain import ImageDomain
from .physical_gradient_descent import PhysicalGradientDescent
from .registration import (
    _downsample,
    _pyramid_configuration,
    _smooth_image,
    _validate_images,
    _validate_initial_affine,
)
from .scaling_and_squaring import scaling_and_squaring
from .similarity import (
    ants_neighborhood_correlation_loss,
    bending_energy,
    mean_squared_error,
    normalized_cross_correlation_loss,
    squared_l2_energy,
)
from .spatial_transform import (
    affine_displacement_field,
    compose_displacements,
    jacobian_determinant,
    warp_image,
)


def _validate_sigma(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _smooth_vector_field(field: Tensor, domain: ImageDomain, sigma: float) -> Tensor:
    """Smooth vector components independently; ``sigma`` is in physical units."""
    if sigma == 0:
        return field
    sigma_torch = tuple(reversed(tuple(sigma / spacing for spacing in domain.spacing)))
    result = field
    convolution = F.conv2d if domain.dimension == 2 else F.conv3d
    # Separable convolution avoids a prohibitively large 3-D dense kernel.
    for axis, axis_sigma in enumerate(sigma_torch):
        radius = max(1, int(3.0 * axis_sigma + 0.5))
        coordinate = torch.arange(-radius, radius + 1, dtype=field.dtype, device=field.device)
        kernel_1d = torch.exp(-0.5 * (coordinate / axis_sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        shape = [1] * domain.dimension
        shape[axis] = kernel_1d.numel()
        kernel = kernel_1d.reshape((1, 1) + tuple(shape)).expand((field.shape[1], 1) + tuple(shape))
        padding = [0] * (2 * domain.dimension)
        reverse_axis = domain.dimension - 1 - axis
        padding[2 * reverse_axis : 2 * reverse_axis + 2] = [radius, radius]
        result = convolution(F.pad(result, tuple(padding), mode="replicate"), kernel, groups=field.shape[1])
    return result


def _zero_boundary(field: Tensor) -> Tensor:
    result = field.clone()
    for axis in range(2, field.ndim):
        lower = [slice(None)] * field.ndim
        upper = [slice(None)] * field.ndim
        lower[axis] = 0
        upper[axis] = -1
        result[tuple(lower)] = 0
        result[tuple(upper)] = 0
    return result


def _evaluate(
    velocity: Tensor,
    fixed: Tensor,
    moving: Tensor,
    fixed_domain: ImageDomain,
    moving_domain: ImageDomain,
    *,
    similarity: str,
    neighborhood_radius,
    velocity_weight: float,
    bending_weight: float,
    squaring_steps: int,
    padding_mode: str,
    initial_affine_displacement: Optional[Tensor],
) -> Dict[str, Tensor]:
    svf_displacement = scaling_and_squaring(velocity, fixed_domain, squaring_steps)
    displacement = (
        svf_displacement
        if initial_affine_displacement is None
        else compose_displacements(initial_affine_displacement, svf_displacement, fixed_domain)
    )
    warped = warp_image(moving, displacement, fixed_domain, moving_domain, padding_mode=padding_mode)
    if similarity == "mse":
        similarity_value = mean_squared_error(fixed, warped)
    elif similarity == "ncc":
        similarity_value = normalized_cross_correlation_loss(fixed, warped)
    else:
        similarity_value = ants_neighborhood_correlation_loss(fixed, warped, neighborhood_radius)
    velocity_regularization = squared_l2_energy(velocity)
    bending_regularization = bending_energy(velocity, fixed_domain)
    return {
        "warped_moving": warped,
        "velocity": velocity,
        "svf_displacement": svf_displacement,
        "displacement": displacement,
        "similarity": similarity_value,
        "velocity_regularization": velocity_regularization,
        "bending_regularization": bending_regularization,
        "loss": similarity_value + velocity_weight * velocity_regularization + bending_weight * bending_regularization,
        "jacobian_determinant": jacobian_determinant(displacement, fixed_domain),
    }


def gaussian_svf_registration(
    fixed: Tensor,
    moving: Tensor,
    fixed_domain: ImageDomain,
    moving_domain: Optional[ImageDomain] = None,
    *,
    iterations: Union[int, Sequence[int]] = 100,
    optimizer: Optional[PhysicalGradientDescent] = None,
    gradient_step: float = 0.2,
    momentum: float = 0.0,
    update_field_sigma: float = 3.0,
    total_field_sigma: float = 0.5,
    similarity: str = "mse",
    neighborhood_radius: Union[int, Sequence[int]] = 2,
    velocity_weight: float = 0.0,
    bending_weight: float = 0.0,
    squaring_steps: int = 7,
    padding_mode: str = "zeros",
    stationary_boundary: bool = True,
    convergence_tolerance: Optional[float] = None,
    return_loss_history: bool = True,
    initial_velocity: Optional[Tensor] = None,
    initial_affine: Optional[Tuple[Tensor, Tensor]] = None,
    shrink_factors: Sequence[int] = (1,),
    smoothing_sigmas: Optional[Union[float, Sequence[float]]] = None,
    verbose: bool = False,
    detach_outputs: bool = True,
) -> Dict[str, object]:
    """Register images with a dense Gaussian-regularized stationary velocity.

    The update gradient is Gaussian-smoothed by ``update_field_sigma`` before
    its maximum vector norm is scaled to ``gradient_step * voxel_diagonal``.
    After every update the accumulated velocity is smoothed by
    ``total_field_sigma``. Both sigmas use physical domain units. The forward
    and inverse transforms are ``exp(v)`` and ``exp(-v)`` respectively.

    A :class:`PhysicalGradientDescent` instance may specify ``gradient_step``
    and ``momentum``; its ``smoothing_sigma`` replaces ``update_field_sigma``
    when nonzero. Unlike B-spline SVF registration, optimizer state and the
    velocity parameters are dense voxel fields.
    """
    if not isinstance(fixed_domain, ImageDomain):
        raise TypeError("fixed_domain must be a ImageDomain")
    moving_domain = fixed_domain if moving_domain is None else moving_domain
    if not isinstance(moving_domain, ImageDomain):
        raise TypeError("moving_domain must be a ImageDomain")
    if fixed_domain.dimension != moving_domain.dimension:
        raise ValueError("fixed_domain and moving_domain must have the same dimension")
    _validate_images(fixed, moving, fixed_domain, moving_domain)
    if not isinstance(verbose, bool) or not isinstance(stationary_boundary, bool):
        raise TypeError("verbose and stationary_boundary must be bools")
    if similarity not in ("mse", "ncc", "ants_ncc"):
        raise ValueError("similarity must be 'mse', 'ncc', or 'ants_ncc'")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    for name, weight in (("velocity_weight", velocity_weight), ("bending_weight", bending_weight)):
        _validate_sigma(weight, name)
    update_field_sigma = _validate_sigma(update_field_sigma, "update_field_sigma")
    total_field_sigma = _validate_sigma(total_field_sigma, "total_field_sigma")
    if optimizer is None:
        optimizer = PhysicalGradientDescent(gradient_step, momentum)
    elif not isinstance(optimizer, PhysicalGradientDescent):
        raise TypeError("optimizer must be a PhysicalGradientDescent instance or None")
    if optimizer.smoothing_sigma:
        update_field_sigma = optimizer.smoothing_sigma
    factors, sigmas, level_iterations, _ = _pyramid_configuration(
        shrink_factors, smoothing_sigmas, iterations, 1.0
    )
    initial_affine = _validate_initial_affine(initial_affine, fixed, fixed_domain.dimension)
    if convergence_tolerance is not None:
        _validate_sigma(convergence_tolerance, "convergence_tolerance")
    expected_velocity_shape = (fixed.shape[0], fixed_domain.dimension) + fixed_domain.torch_size
    if initial_velocity is not None:
        if tuple(initial_velocity.shape) != expected_velocity_shape:
            raise ValueError("initial_velocity shape does not match fixed_domain")
        if initial_velocity.dtype != fixed.dtype or initial_velocity.device != fixed.device:
            raise ValueError("initial_velocity must match the fixed image dtype and device")

    if verbose:
        print("ANTsTorch Gaussian SVF registration configuration:")
        configuration = (
            ("fixed_domain", fixed_domain), ("moving_domain", moving_domain),
            ("fixed_shape", tuple(fixed.shape)), ("moving_shape", tuple(moving.shape)),
            ("dtype", fixed.dtype), ("device", fixed.device),
            ("shrink_factors", factors), ("smoothing_sigmas", sigmas),
            ("iterations", level_iterations), ("gradient_step", optimizer.gradient_step),
            ("momentum", optimizer.momentum), ("update_field_sigma", update_field_sigma),
            ("total_field_sigma", total_field_sigma), ("similarity", similarity),
            ("neighborhood_radius", neighborhood_radius), ("velocity_weight", velocity_weight),
            ("bending_weight", bending_weight), ("squaring_steps", squaring_steps),
            ("padding_mode", padding_mode), ("stationary_boundary", stationary_boundary),
            ("convergence_tolerance", convergence_tolerance),
            ("initial_velocity_provided", initial_velocity is not None),
            ("initial_affine_provided", initial_affine is not None),
            ("return_loss_history", return_loss_history), ("detach_outputs", detach_outputs),
        )
        for name, value in configuration:
            print(f"  {name}: {value}")

    velocity = None
    momentum_buffer = None
    history, level_history = [], []
    for level, (factor, sigma, iteration_count) in enumerate(zip(factors, sigmas, level_iterations)):
        fixed_level, fixed_level_domain = _downsample(_smooth_image(fixed, fixed_domain, sigma), fixed_domain, factor)
        moving_level, moving_level_domain = _downsample(_smooth_image(moving, moving_domain, sigma), moving_domain, factor)
        if velocity is None:
            source = initial_velocity
            if source is None:
                velocity = fixed_level.new_zeros(
                    (fixed.shape[0], fixed_domain.dimension) + fixed_level_domain.torch_size
                )
            else:
                mode = "bilinear" if fixed_domain.dimension == 2 else "trilinear"
                velocity = F.interpolate(source, size=fixed_level_domain.torch_size, mode=mode, align_corners=True)
        else:
            mode = "bilinear" if fixed_domain.dimension == 2 else "trilinear"
            velocity = F.interpolate(velocity.detach(), size=fixed_level_domain.torch_size, mode=mode, align_corners=True)
        velocity = velocity.requires_grad_(True)
        affine_level = (
            affine_displacement_field(initial_affine[0], initial_affine[1], fixed_level_domain, fixed_level)
            if initial_affine is not None else None
        )
        optimizer.reset()
        momentum_buffer = None
        if verbose:
            voxel_diagonal = sum(spacing ** 2 for spacing in fixed_level_domain.spacing) ** 0.5
            print(
                f"Resolution level {level + 1}/{len(factors)}: shrink_factor={factor}, "
                f"smoothing_sigma={sigma:g}, fixed_size={fixed_level_domain.size}, "
                f"moving_size={moving_level_domain.size}, velocity_parameters={velocity.numel()}, "
                f"iterations={iteration_count}"
            )
            print(f"  physical_gradient_step={optimizer.gradient_step * voxel_diagonal:.8g} "
                  f"({optimizer.gradient_step:g} * voxel_diagonal {voxel_diagonal:.8g})")
        current_history, previous = [], None
        for _ in range(iteration_count):
            velocity.grad = None
            result = _evaluate(
                velocity, fixed_level, moving_level, fixed_level_domain, moving_level_domain,
                similarity=similarity, neighborhood_radius=neighborhood_radius,
                velocity_weight=velocity_weight, bending_weight=bending_weight,
                squaring_steps=squaring_steps, padding_mode=padding_mode,
                initial_affine_displacement=affine_level,
            )
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"non-finite registration loss at resolution level {level + 1}")
            result["loss"].backward()
            direction = _smooth_vector_field(velocity.grad, fixed_level_domain, update_field_sigma)
            if optimizer.momentum:
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(direction)
                momentum_buffer.mul_(optimizer.momentum).add_(direction)
                direction = momentum_buffer
            maximum_norm = direction.square().sum(1).sqrt().flatten(1).amax(1)
            voxel_diagonal = sum(spacing ** 2 for spacing in fixed_level_domain.spacing) ** 0.5
            target = maximum_norm.new_full(maximum_norm.shape, optimizer.gradient_step * voxel_diagonal)
            scale = torch.where(maximum_norm > 0, target / maximum_norm, torch.zeros_like(maximum_norm))
            scale = scale.reshape((velocity.shape[0], 1) + (1,) * fixed_domain.dimension)
            with torch.no_grad():
                velocity.add_(direction * scale, alpha=-1.0)
                velocity.copy_(_smooth_vector_field(velocity, fixed_level_domain, total_field_sigma))
                if stationary_boundary:
                    velocity.copy_(_zero_boundary(velocity))
                current = float(_evaluate(
                    velocity, fixed_level, moving_level, fixed_level_domain, moving_level_domain,
                    similarity=similarity, neighborhood_radius=neighborhood_radius,
                    velocity_weight=velocity_weight, bending_weight=bending_weight,
                    squaring_steps=squaring_steps, padding_mode=padding_mode,
                    initial_affine_displacement=affine_level,
                )["loss"].item())
            current_history.append(current)
            if verbose:
                print(f"  iteration {len(current_history):04d}: loss={current:.8g}")
            if previous is not None and convergence_tolerance is not None and (
                abs(previous - current) <= convergence_tolerance * max(1.0, abs(previous))
            ):
                break
            previous = current
        level_history.append(current_history)
        history.extend(current_history)

    affine_full = (
        affine_displacement_field(initial_affine[0], initial_affine[1], fixed_domain, fixed)
        if initial_affine is not None else None
    )
    result = _evaluate(
        velocity, fixed, moving, fixed_domain, moving_domain,
        similarity=similarity, neighborhood_radius=neighborhood_radius,
        velocity_weight=velocity_weight, bending_weight=bending_weight,
        squaring_steps=squaring_steps, padding_mode=padding_mode,
        initial_affine_displacement=affine_full,
    )
    result["warpedmovout"] = result.pop("warped_moving")
    result.pop("displacement", None)
    result["fwdtransforms"] = result.pop("svf_displacement")
    result["invtransforms"] = scaling_and_squaring(-velocity, fixed_domain, squaring_steps)
    if initial_affine is None:
        result.update({key: None for key in (
            "affine_matrix", "affine_translation", "affine_matrix_inverse", "affine_translation_inverse"
        )})
    else:
        matrix, translation = initial_affine
        inverse_matrix = torch.linalg.inv(matrix)
        result.update({
            "affine_matrix": matrix, "affine_translation": translation,
            "affine_matrix_inverse": inverse_matrix,
            "affine_translation_inverse": -torch.einsum("nij,nj->ni", inverse_matrix, translation),
        })
    result["loss_history"] = history if return_loss_history else None
    result["level_loss_history"] = level_history if return_loss_history else None
    if detach_outputs:
        result = {key: value.detach() if isinstance(value, Tensor) else value for key, value in result.items()}
    return result
