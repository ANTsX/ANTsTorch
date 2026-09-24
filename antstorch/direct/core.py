"""Tensor implementation of the historical DiReCT thickness iteration."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ..bspline_flows import ImageDomain, compose_displacements, warp_image
from ..registration import reg_adam_direction
from ..syn.core.inverse import update_inverse_field_nd
from .forces import binary_contour, direct_force, gaussian_scalar, normalized_probability_gradient
from .regularization import regularize_velocity


@dataclass
class DiReCTResult:
    thickness: Tensor
    velocity: Tensor
    energy_history: List[float]


def _invert(field: Tensor, initial: Tensor, domain: ImageDomain, iterations: int) -> Tensor:
    # syn.core uses reversed vector components; DiReCT uses ITK x-y-z.
    reversed_field = field.movedim(1, -1).flip(-1)
    reversed_initial = initial.movedim(1, -1).flip(-1)
    result = update_inverse_field_nd(
        reversed_field,
        reversed_initial,
        steps=iterations,
        method="fixed_point",
        spacing=domain.spacing,
        origin=domain.origin,
        direction=domain.direction,
        max_error_threshold=0.1,
        mean_error_threshold=0.001,
    )
    return result.flip(-1).movedim(-1, 1)


@torch.no_grad()
def direct_cortical_thickness(
    segmentation: Tensor,
    gray_probability: Tensor,
    white_probability: Tensor,
    domain: ImageDomain,
    *,
    gray_label: int = 2,
    white_label: int = 3,
    iterations: int = 45,
    gradient_step: float = 0.025,
    integration_points: int = 10,
    thickness_prior: float = 10.0,
    smoothing_sigma: float = 1.0,
    velocity_smoothing_variance: float = 1.5,
    inverse_iterations: int = 20,
    optimizer: str = "direct",
    regularizer: str = "gaussian",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    verbose: bool = False,
) -> DiReCTResult:
    """Estimate cortical thickness from a hard segmentation and GM/WM probabilities."""
    expected_image_shape = (1, 1) + domain.torch_size
    for name, value in (("segmentation", segmentation), ("gray_probability", gray_probability),
                        ("white_probability", white_probability)):
        if tuple(value.shape) != expected_image_shape:
            raise ValueError(f"{name} must have shape {expected_image_shape}")
    if optimizer not in ("direct", "reg_adam"):
        raise ValueError("optimizer must be 'direct' or 'reg_adam'")
    if iterations < 1 or integration_points < 1:
        raise ValueError("iterations and integration_points must be positive")

    dtype, device = gray_probability.dtype, gray_probability.device
    gray_mask = (segmentation == gray_label).to(dtype)
    white_mask = (segmentation == white_label).to(dtype)
    matter_mask = ((gray_mask + white_mask) > 0).to(dtype)
    matter_contour = binary_contour(matter_mask)
    white_contour = binary_contour(white_mask)
    active = ((segmentation != 0) & ((white_contour > 0) | (matter_contour > 0) | (gray_mask > 0))).to(dtype)

    field_shape = (1, domain.dimension) + domain.torch_size
    velocity = torch.zeros(field_shape, dtype=dtype, device=device)
    integrated = torch.zeros_like(velocity)
    thickness = torch.zeros_like(gray_probability)
    energy_history: List[float] = []
    adam_state = None

    for outer in range(iterations):
        forward_increment = torch.zeros_like(velocity)
        inverse_field = torch.zeros_like(velocity)
        inverse_increment = torch.zeros_like(velocity)
        hit = torch.zeros_like(gray_probability)
        total = torch.zeros_like(gray_probability)
        energy = torch.zeros((), dtype=dtype, device=device)

        for point in range(integration_points):
            inverse_field = compose_displacements(inverse_field, inverse_increment, domain)
            warped_white = warp_image(white_probability, inverse_field, domain, padding_mode="border")
            warped_contour = warp_image(white_contour, inverse_field, domain, padding_mode="zeros")
            warped_thickness = warp_image(thickness, inverse_field, domain, padding_mode="zeros")
            gradient = normalized_probability_gradient(
                warped_white, sigma=smoothing_sigma, spacing=domain.spacing
            )
            force = direct_force(
                warped_white, gray_probability, gray_mask, gradient, gradient_step
            )
            forward_increment.add_(force)
            energy.add_(((warped_white - gray_probability).abs() * gray_mask).sum())

            if point == 0:
                hit.copy_(white_contour)
                norm = torch.linalg.vector_norm(integrated, dim=1, keepdim=True)
                thickness.copy_(norm * white_contour)
                total.copy_(thickness)
                integrated.zero_()
            else:
                hit.add_(warped_contour * gray_mask)
                total.add_(warped_thickness * gray_mask)

            inverse_field.mul_(active)
            velocity.mul_(active)
            integrated.mul_(active)
            inverse_increment.copy_(velocity * active)
            integrated = _invert(inverse_field, integrated, domain, inverse_iterations)
            inverse_field = _invert(integrated, inverse_field, domain, inverse_iterations)

        smooth_hit = gaussian_scalar(hit, smoothing_sigma)
        smooth_total = gaussian_scalar(total, smoothing_sigma)
        estimate = torch.where(
            smooth_hit > 0.001,
            (smooth_total / smooth_hit.clamp_min(0.001)).clamp_min(0),
            torch.zeros_like(smooth_total),
        ) * gray_mask
        thickness.copy_(estimate)

        update = forward_increment
        if optimizer == "reg_adam":
            adam_state, update_last = reg_adam_direction(
                update.movedim(1, -1), adam_state, betas=adam_betas, eps=adam_eps
            )
            update = update_last.movedim(-1, 1)
        velocity.add_(update)
        if thickness_prior > 0:
            fraction = (float(thickness_prior) / thickness.clamp_min(1e-6)).clamp(max=1.0)
            velocity.mul_(torch.where(gray_mask > 0, fraction.square(), torch.ones_like(fraction)))
        velocity = regularize_velocity(
            velocity, mode=regularizer, variance=velocity_smoothing_variance, spacing=domain.spacing
        )

        gm_count = gray_mask.sum().clamp_min(1)
        current_energy = float((energy / (gm_count * integration_points)).item())
        energy_history.append(current_energy)
        if verbose:
            print(f"DiReCT iteration {outer + 1}/{iterations}: energy={current_energy:.8g}")

    return DiReCTResult(thickness=thickness, velocity=velocity, energy_history=energy_history)
