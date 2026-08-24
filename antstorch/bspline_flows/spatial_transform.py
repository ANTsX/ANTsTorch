"""Physical-coordinate dense transforms for 2-D and 3-D tensors.

ITK metadata and vector components use x-y-(z) order. PyTorch spatial storage
is H-W or D-H-W. A displacement ``u`` maps a fixed physical point ``p`` to
the moving-image sample location ``p + u(p)``.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .bspline_domain import BSplineDomain


ALIGN_CORNERS = True


def _geometry_tensor(values, reference: Tensor) -> Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _validate_field(field: Tensor, domain: BSplineDomain, name: str = "field") -> None:
    expected = (field.shape[0], domain.dimension) + domain.torch_size
    if tuple(field.shape) != expected:
        raise ValueError(f"{name} must have shape (N, {domain.dimension}, {', '.join(map(str, domain.torch_size))})")
    if not field.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def physical_grid(domain: BSplineDomain, reference: Tensor) -> Tensor:
    """Return a singleton-batch grid of physical points, channels x-y-(z)."""
    axes = [torch.arange(n, dtype=reference.dtype, device=reference.device) for n in domain.torch_size]
    torch_coordinates = torch.meshgrid(*axes, indexing="ij")
    itk_index = torch.stack(tuple(reversed(torch_coordinates)), dim=0)
    spacing = _geometry_tensor(domain.spacing, reference).reshape((-1,) + (1,) * domain.dimension)
    origin = _geometry_tensor(domain.origin, reference).reshape((-1,) + (1,) * domain.dimension)
    direction = _geometry_tensor(domain.direction, reference)
    scaled_index = itk_index * spacing
    return (origin + torch.einsum("ij,j...->i...", direction, scaled_index)).unsqueeze(0)


def affine_displacement_field(matrix: Tensor, translation: Tensor, domain: BSplineDomain, reference: Tensor) -> Tensor:
    """Physical displacement field for the affine map ``p -> matrix @ p + translation``.

    Returns ``u`` with ``u(p) = matrix @ p + translation - p`` evaluated at
    every physical point of ``domain``, in the same ``(N, dim, *torch_size)``
    displacement convention used throughout this module — directly usable
    with :func:`warp_image` and :func:`compose_displacements`, and as the
    ``initial_affine`` argument of
    :func:`antstorch.bspline_flows.registration.registration`.

    Parameters
    ----------
    matrix : Tensor
        Physical-space affine linear part, shape ``(dim, dim)`` (applied to
        every batch item) or ``(N, dim, dim)`` (one matrix per batch item).
    translation : Tensor
        Physical-space affine translation, shape ``(dim,)`` or ``(N, dim)``,
        matching ``matrix``'s batching.
    domain : BSplineDomain
        Domain the field is evaluated on.
    reference : Tensor
        Supplies the output dtype/device (and, when ``matrix``/``translation``
        are unbatched, the batch size ``N`` to broadcast to via its leading
        dimension).

    Returns
    -------
    Tensor
        Displacement field of shape ``(N, dim, *domain.torch_size)``.
    """
    dimension = domain.dimension
    matrix_t = _geometry_tensor(matrix, reference)
    translation_t = _geometry_tensor(translation, reference)
    if matrix_t.ndim == 2:
        matrix_t = matrix_t.unsqueeze(0).expand(reference.shape[0], -1, -1)
    if translation_t.ndim == 1:
        translation_t = translation_t.unsqueeze(0).expand(reference.shape[0], -1)
    if matrix_t.ndim != 3 or tuple(matrix_t.shape[-2:]) != (dimension, dimension):
        raise ValueError(f"matrix must have shape ({dimension}, {dimension}) or (N, {dimension}, {dimension})")
    if translation_t.ndim != 2 or translation_t.shape[-1] != dimension:
        raise ValueError(f"translation must have shape ({dimension},) or (N, {dimension})")
    if matrix_t.shape[0] != translation_t.shape[0]:
        raise ValueError("matrix and translation must have matching batch sizes")

    points = physical_grid(domain, reference).squeeze(0)  # (dim, *torch_size)
    mapped = torch.einsum("nij,j...->ni...", matrix_t, points)
    mapped = mapped + translation_t.reshape(translation_t.shape + (1,) * dimension)
    return mapped - points.unsqueeze(0)


def displacement_to_sampling_grid(
    displacement: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: BSplineDomain,
) -> Tensor:
    """Convert a physical fixed-to-moving displacement to a grid_sample grid."""
    if fixed_domain.dimension != moving_domain.dimension:
        raise ValueError("fixed and moving domains must have the same dimension")
    _validate_field(displacement, fixed_domain, "displacement")
    points = physical_grid(fixed_domain, displacement) + displacement
    origin = _geometry_tensor(moving_domain.origin, displacement).reshape(
        (1, -1) + (1,) * moving_domain.dimension
    )
    inverse_direction = torch.linalg.inv(_geometry_tensor(moving_domain.direction, displacement))
    continuous = torch.einsum("ij,nj...->ni...", inverse_direction, points - origin)
    spacing = _geometry_tensor(moving_domain.spacing, displacement).reshape(
        (1, -1) + (1,) * moving_domain.dimension
    )
    continuous = continuous / spacing
    size = _geometry_tensor(moving_domain.size, displacement).reshape(
        (1, -1) + (1,) * moving_domain.dimension
    )
    normalized = 2.0 * continuous / (size - 1.0) - 1.0
    permutation = (0,) + tuple(range(2, 2 + moving_domain.dimension)) + (1,)
    return normalized.permute(permutation)


def warp_image(
    moving: Tensor,
    displacement: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: BSplineDomain = None,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Pull ``moving`` onto the fixed grid using ``p_moving = p_fixed + u``."""
    moving_domain = moving_domain or fixed_domain
    if moving.ndim != moving_domain.dimension + 2 or tuple(moving.shape[2:]) != moving_domain.torch_size:
        raise ValueError("moving tensor shape does not match moving_domain")
    if moving.shape[0] != displacement.shape[0]:
        raise ValueError("moving image and displacement batch sizes must match")
    grid = displacement_to_sampling_grid(displacement, fixed_domain, moving_domain)
    return F.grid_sample(
        moving, grid, mode=mode, padding_mode=padding_mode, align_corners=ALIGN_CORNERS
    )


def compose_displacements(first: Tensor, second: Tensor, domain: BSplineDomain) -> Tensor:
    """Return ``T_second o T_first``: ``first(x) + second(x + first(x))``."""
    _validate_field(first, domain, "first")
    _validate_field(second, domain, "second")
    if first.shape != second.shape:
        raise ValueError("displacements must have identical shapes")
    sampled_second = warp_image(
        second, first, domain, domain, mode="bilinear", padding_mode="border"
    )
    return first + sampled_second


def jacobian_determinant(displacement: Tensor, domain: BSplineDomain) -> Tensor:
    """Compute ``det(I + du/dp)`` in physical coordinates."""
    _validate_field(displacement, domain, "displacement")
    derivatives = []
    for itk_axis, spacing in enumerate(domain.spacing):
        torch_axis = displacement.ndim - 1 - itk_axis
        derivatives.append(torch.gradient(displacement, spacing=(spacing,), dim=(torch_axis,))[0])
    du_dq = torch.stack(derivatives, dim=2)  # N, output component, input x/y/z, ...
    inverse_direction = torch.linalg.inv(_geometry_tensor(domain.direction, displacement))
    du_dp = torch.einsum("nij...,jk->nik...", du_dq, inverse_direction)
    identity = torch.eye(domain.dimension, dtype=displacement.dtype, device=displacement.device)
    identity = identity.reshape((1, domain.dimension, domain.dimension) + (1,) * domain.dimension)
    jacobian = du_dp + identity
    permutation = (0,) + tuple(range(3, 3 + domain.dimension)) + (1, 2)
    return torch.linalg.det(jacobian.permute(permutation))


def folding_count(displacement: Tensor, domain: BSplineDomain) -> Tensor:
    """Number of non-positive Jacobian determinants in each batch item."""
    determinant = jacobian_determinant(displacement, domain)
    return (determinant <= 0).flatten(start_dim=1).sum(dim=1)

