"""Tensor implementation of the historical DiReCT thickness iteration."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .._torch_compat import grid_sample
from ..bspline_flows import ImageDomain
from ..bspline_flows.spatial_transform import ALIGN_CORNERS
from ..registration import reg_adam_direction
from ..syn.core.smoothing import get_boundary_mask
from .forces import binary_contour, direct_force, gaussian_scalar, normalized_probability_gradient
from .regularization import regularize_velocity


@dataclass
class DiReCTResult:
    thickness: Tensor
    velocity: Tensor
    energy_history: List[float]


class _DirectGridContext:
    """Precomputed normalized sampling grid and physical-to-normalized displacement transform on ``domain``."""

    def __init__(
        self,
        domain: ImageDomain,
        dtype: torch.dtype,
        device: torch.device,
        full_num_voxels: int | None = None,
    ) -> None:
        self.domain = domain
        self.dim = domain.dimension
        self.dtype = dtype
        self.device = device
        if full_num_voxels is not None:
            if int(full_num_voxels) < 1:
                raise ValueError("full_num_voxels must be positive")
            self.mean_denom = float(int(full_num_voxels))
        else:
            self.mean_denom = float(math.prod(domain.torch_size))

        axes = [torch.arange(n, dtype=dtype, device=device) for n in domain.torch_size]
        torch_coords = torch.meshgrid(*axes, indexing="ij")
        itk_index = torch.stack(tuple(reversed(torch_coords)), dim=-1)  # (*torch_size, dim) in x,y[,z]
        size_t = torch.tensor(domain.size, dtype=dtype, device=device)
        spacing_t = torch.tensor(domain.spacing, dtype=dtype, device=device)
        dir_t = torch.tensor(domain.direction, dtype=dtype, device=device)
        inv_dir = torch.linalg.inv(dir_t)

        # Normalized base grid [-1, 1] of shape (1, *torch_size, dim)
        denom = (size_t - 1.0).clamp_min(1.0)
        self.base_grid = (2.0 * itk_index / denom - 1.0).unsqueeze(0).contiguous()

        # Linear map from physical displacement u (x,y[,z] mm) to normalized grid offset:
        # delta_norm = diag(2 / ((size - 1) * spacing)) @ D^{-1} @ u
        scale_vec = 2.0 / (denom * spacing_t)
        u_mat = scale_vec.unsqueeze(-1) * inv_dir  # (dim, dim)
        off_diag = u_mat - torch.diag(torch.diagonal(u_mat))
        self.is_diagonal = bool(float(off_diag.abs().max().item()) <= 1e-7)
        self.u_scale = torch.diagonal(u_mat).contiguous()
        self.u_mat_t = u_mat.T.contiguous()

        # Precomputed boundary mask and inverse-spacing vector for fixed-point inversion
        bmask_zyx = get_boundary_mask(domain.torch_size, device, dtype)  # (1, *torch_size, 1)
        self.boundary_mask_cf = bmask_zyx.movedim(-1, 1).contiguous()    # (1, 1, *torch_size)
        self.inv_spacing_cf = (1.0 / spacing_t).view((1, self.dim) + (1,) * self.dim).contiguous()
        self.one_scalar = torch.ones((), dtype=dtype, device=device)

    def sampling_grid(self, displacement: Tensor) -> Tensor:
        u_last = displacement.movedim(1, -1)
        if self.is_diagonal:
            return self.base_grid + u_last * self.u_scale
        return self.base_grid + torch.matmul(u_last, self.u_mat_t)


def _invert_with_ctx(
    field: Tensor,
    initial: Tensor,
    ctx: _DirectGridContext,
    iterations: int,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    full_num_voxels: int | None = None,
) -> Tensor:
    """Exact channel-first ITK fixed-point displacement inversion with precomputed domain tensors."""
    if full_num_voxels is not None and int(full_num_voxels) < 1:
        raise ValueError("full_num_voxels must be positive")
    mean_denom = float(int(full_num_voxels)) if full_num_voxels is not None else ctx.mean_denom
    v = initial
    max_err = float("inf")
    mean_err = float("inf")
    for iteration in range(iterations):
        if iteration > 0 and (max_err <= max_error_threshold or mean_err <= mean_error_threshold):
            break
        coords_norm = ctx.sampling_grid(v)
        fwd_at_inv = grid_sample(field, coords_norm, mode="bilinear", padding_mode="border", align_corners=ALIGN_CORNERS)
        del coords_norm
        error = v + fwd_at_inv
        del fwd_at_inv
        scaled_sq = (error * ctx.inv_spacing_cf).square().sum(dim=1, keepdim=True)
        scaled_norm = torch.sqrt(scaled_sq)
        del scaled_sq
        max_err, mean_err = torch.stack([scaled_norm.amax(), scaled_norm.sum() / mean_denom]).tolist()

        epsilon = 0.75 if iteration == 0 else 0.5
        clip_threshold = epsilon * max_err
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp_min(1e-10),
            ctx.one_scalar,
        )
        del scaled_norm
        v = (v - (epsilon * clip_scale) * error) * ctx.boundary_mask_cf
        del error, clip_scale
    return v


def _invert(
    field: Tensor,
    initial: Tensor,
    domain: ImageDomain,
    iterations: int,
    ctx: Optional[_DirectGridContext] = None,
) -> Tensor:
    if ctx is None:
        ctx = _DirectGridContext(domain, field.dtype, field.device)
    return _invert_with_ctx(field, initial, ctx, iterations)


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
    full_num_voxels: int | None = None,
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
    ctx = _DirectGridContext(domain, dtype, device, full_num_voxels=full_num_voxels)

    gray_mask = (segmentation == gray_label).to(dtype)
    white_mask = (segmentation == white_label).to(dtype)
    matter_mask = ((gray_mask + white_mask) > 0).to(dtype)
    matter_contour = binary_contour(matter_mask)
    white_contour = binary_contour(white_mask)
    active = ((segmentation != 0) & ((white_contour > 0) | (matter_contour > 0) | (gray_mask > 0))).to(dtype)
    gm_denom = gray_mask.sum().clamp_min(1) * float(integration_points)
    ones_img = torch.ones_like(gray_probability)
    zeros_img = torch.zeros_like(gray_probability)

    field_shape = (1, domain.dimension) + domain.torch_size
    velocity = torch.zeros(field_shape, dtype=dtype, device=device)
    integrated = torch.zeros_like(velocity)
    thickness = torch.zeros_like(gray_probability)
    energy_steps = torch.empty((iterations,), dtype=dtype, device=device)
    adam_state = None

    for outer in range(iterations):
        forward_increment = torch.zeros_like(velocity)
        inverse_field = torch.zeros_like(velocity)
        inverse_increment = torch.zeros_like(velocity)
        hit = torch.zeros_like(gray_probability)
        total = torch.zeros_like(gray_probability)
        energy = torch.zeros((), dtype=dtype, device=device)

        for point in range(integration_points):
            comp_grid = ctx.sampling_grid(inverse_field)
            inverse_field = inverse_field + grid_sample(
                inverse_increment, comp_grid, mode="bilinear", padding_mode="border", align_corners=ALIGN_CORNERS
            )
            del comp_grid

            inv_grid = ctx.sampling_grid(inverse_field)
            warped_white = grid_sample(
                white_probability, inv_grid, mode="bilinear", padding_mode="border", align_corners=ALIGN_CORNERS
            )
            if point > 0:
                warped_ct = grid_sample(
                    torch.cat([white_contour, thickness], dim=1),
                    inv_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=ALIGN_CORNERS,
                )
            del inv_grid

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
                hit.add_(warped_ct[:, 0:1] * gray_mask)
                total.add_(warped_ct[:, 1:2] * gray_mask)
                del warped_ct

            inverse_field.mul_(active)
            velocity.mul_(active)
            integrated.mul_(active)
            inverse_increment.copy_(velocity * active)
            integrated = _invert_with_ctx(inverse_field, integrated, ctx, inverse_iterations)
            inverse_field = _invert_with_ctx(integrated, inverse_field, ctx, inverse_iterations)

        # ANTs uses this parameter as variance here, but sigma for the gradient.
        # See direct/README.md for the distinct smoothing conventions.
        smooth_hit = gaussian_scalar(hit, smoothing_sigma ** 0.5, maximum_error=0.01)
        smooth_total = gaussian_scalar(total, smoothing_sigma ** 0.5, maximum_error=0.01)
        estimate = torch.where(
            smooth_hit > 0.001,
            (smooth_total / smooth_hit.clamp_min(0.001)).clamp_min(0),
            zeros_img,
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
            velocity.mul_(torch.where(gray_mask > 0, fraction.square(), ones_img))
        velocity = regularize_velocity(
            velocity, mode=regularizer, variance=velocity_smoothing_variance, spacing=domain.spacing
        )

        # Tensor reduction order differs from ANTs sequential float accumulation.
        # Energy agreement alone is not a thickness-parity check (direct/README.md).
        step_energy = energy / gm_denom
        energy_steps[outer] = step_energy
        if verbose:
            print(f"DiReCT iteration {outer + 1}/{iterations}: energy={float(step_energy.item()):.8g}")

    energy_history = [float(v) for v in energy_steps.tolist()]
    return DiReCTResult(thickness=thickness, velocity=velocity, energy_history=energy_history)
