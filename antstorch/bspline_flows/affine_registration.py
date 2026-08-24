"""Affine pre-registration for antstorch.bspline_flows.

``antstorch.bspline_flows.registration`` estimates only a B-spline
stationary velocity field — by design it has no rigid or affine
initialization (see ``docs/antsx_tutorial_bspline_flows.md``). This module
fills that gap with a lightweight, native-PyTorch affine solver built from
already-ported ``syntx`` primitives (:class:`antstorch.syn.core.affine.HierarchicalAffine`,
a Lie-algebra ``SO(d)`` rotation parameterization), reusing
``antstorch.bspline_flows``'s own physical-coordinate conventions,
similarity metrics, and multi-resolution pyramid machinery end to end — no
``ants.ANTsImage`` dependency, and no normalized ``[-1, 1]`` grid_sample
convention anywhere (unlike syntx's own affine machinery, which is built
around that convention). The estimated affine is returned as a plain
``(matrix, translation)`` physical-space pair, directly usable both
standalone and as the ``initial_affine`` argument of
:func:`antstorch.bspline_flows.registration.registration`.

Robustness against 180-degree "flip" local minima — the main practical
failure mode of a naive single-start affine fit, and the core value
proposition of syntx's own ``robust_affine`` — is approximated here with a
*simplified* multi-start search: center-of-mass translation initialization
plus a handful of candidate seed rotations (identity and a 180-degree
rotation about each principal axis), scored at the coarsest pyramid level
before the full multi-resolution optimization begins. This is intentionally
lighter than syntx's ``robust_affine`` (which also searches finer rotation
cones and offers an ``ants.registration`` C++ fallback) — see the
integration proposal in the project notes for the tradeoff.
"""

import math
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor

from antstorch.syn.core.affine import HierarchicalAffine, get_rotation_matrix

from .bspline_domain import BSplineDomain
from .registration import _downsample, _pyramid_configuration, _smooth_image
from .similarity import (
    ants_neighborhood_correlation_loss,
    mean_squared_error,
    normalized_cross_correlation_loss,
)
from .spatial_transform import affine_displacement_field, physical_grid, warp_image


def _similarity_loss(name: str, fixed: Tensor, warped: Tensor, neighborhood_radius) -> Tensor:
    if name == "mse":
        return mean_squared_error(fixed, warped)
    if name == "ncc":
        return normalized_cross_correlation_loss(fixed, warped)
    return ants_neighborhood_correlation_loss(fixed, warped, neighborhood_radius)


def _center_of_mass(image: Tensor, domain: BSplineDomain) -> Tensor:
    """Intensity-weighted physical center of mass of a single-item image, shape ``(dim,)``."""
    points = physical_grid(domain, image).squeeze(0)  # (dim, *spatial)
    weights = image[0].clamp_min(0).mean(dim=0)  # (*spatial), averaged over channels
    total = weights.sum().clamp_min(1e-8)
    return (points * weights.unsqueeze(0)).flatten(1).sum(dim=1) / total


def _seed_rotations(dimension: int, device, dtype) -> Sequence[Tensor]:
    """Identity plus a 180-degree rotation about each principal axis, as Lie-algebra vectors."""
    if dimension == 2:
        return [
            torch.zeros(1, device=device, dtype=dtype),
            torch.tensor([math.pi], device=device, dtype=dtype),
        ]
    seeds = [torch.zeros(3, device=device, dtype=dtype)]
    for axis in range(3):
        omega = torch.zeros(3, device=device, dtype=dtype)
        omega[axis] = math.pi
        seeds.append(omega)
    return seeds


def _fit_single_affine(
    fixed_item: Tensor,
    moving_item: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: BSplineDomain,
    *,
    transform_type: str,
    similarity: str,
    neighborhood_radius,
    factors,
    sigmas,
    level_iterations,
    level_rates,
    multi_start: bool,
    center_of_mass_init: bool,
    padding_mode: str,
    convergence_tolerance: Optional[float],
    verbose: bool,
    batch_index: int,
):
    dimension = fixed_domain.dimension
    device, dtype = fixed_item.device, fixed_item.dtype
    module = HierarchicalAffine(dim=dimension, transform_type=transform_type).to(device=device, dtype=dtype)
    if transform_type == "Translation":
        # HierarchicalAffine always registers `omega` as an nn.Parameter
        # (unlike scale/anisotropic_scale/shear, which are buffers outside
        # their owning transform types), so a pure translation fit must
        # explicitly freeze it to keep rotation out of the optimization.
        module.omega.requires_grad_(False)

    coarse_fixed, coarse_fixed_domain = _downsample(_smooth_image(fixed_item, fixed_domain, sigmas[0]), fixed_domain, factors[0])
    coarse_moving, coarse_moving_domain = _downsample(_smooth_image(moving_item, moving_domain, sigmas[0]), moving_domain, factors[0])

    with torch.no_grad():
        fixed_com = _center_of_mass(fixed_item, fixed_domain) if center_of_mass_init else torch.zeros(dimension, device=device, dtype=dtype)
        moving_com = _center_of_mass(moving_item, moving_domain) if center_of_mass_init else torch.zeros(dimension, device=device, dtype=dtype)

        seed_rotations = _seed_rotations(dimension, device, dtype) if multi_start else [torch.zeros_like(module.omega)]
        best_score, best_omega, best_translation = None, seed_rotations[0], torch.zeros(dimension, device=device, dtype=dtype)
        for omega in seed_rotations:
            rotation = get_rotation_matrix(omega, dimension)
            translation = (moving_com - rotation @ fixed_com) if center_of_mass_init else torch.zeros(dimension, device=device, dtype=dtype)
            candidate_field = affine_displacement_field(rotation, translation, coarse_fixed_domain, coarse_fixed)
            warped = warp_image(coarse_moving, candidate_field, coarse_fixed_domain, coarse_moving_domain, padding_mode=padding_mode)
            score = float(_similarity_loss(similarity, coarse_fixed, warped, neighborhood_radius).item())
            if best_score is None or score < best_score:
                best_score, best_omega, best_translation = score, omega.clone(), translation.clone()

        module.omega.copy_(best_omega)
        module.translation.copy_(best_translation)

    history, level_history = [], []
    for level, (factor, sigma, iteration_count, rate) in enumerate(zip(factors, sigmas, level_iterations, level_rates)):
        fixed_level, fixed_level_domain = _downsample(_smooth_image(fixed_item, fixed_domain, sigma), fixed_domain, factor)
        moving_level, moving_level_domain = _downsample(_smooth_image(moving_item, moving_domain, sigma), moving_domain, factor)
        optimizer = torch.optim.Adam(module.parameters(), lr=rate)
        current_level_history = []
        previous = None
        for _ in range(iteration_count):
            optimizer.zero_grad()
            homogeneous = module.get_matrix()
            matrix, translation = homogeneous[:dimension, :dimension], homogeneous[:dimension, dimension]
            field = affine_displacement_field(matrix, translation, fixed_level_domain, fixed_level)
            warped = warp_image(moving_level, field, fixed_level_domain, moving_level_domain, padding_mode=padding_mode)
            loss = _similarity_loss(similarity, fixed_level, warped, neighborhood_radius)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite affine registration loss at resolution level {level + 1} (batch item {batch_index})"
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                module.clamp_parameters()
            current = float(loss.item())
            current_level_history.append(current)
            if verbose:
                print(f"  [affine batch {batch_index}] level {level + 1} iteration {len(current_level_history):04d}: loss={current:.8g}")
            if previous is not None and convergence_tolerance is not None:
                if abs(previous - current) <= convergence_tolerance * max(1.0, abs(previous)):
                    if verbose:
                        print(f"  [affine batch {batch_index}] convergence reached at level {level + 1}")
                    break
            previous = current
        level_history.append(current_level_history)
        history.extend(current_level_history)

    with torch.no_grad():
        homogeneous = module.get_matrix()
        matrix = homogeneous[:dimension, :dimension].clone()
        translation = homogeneous[:dimension, dimension].clone()
    return matrix, translation, history, level_history


def affine_registration(
    fixed: Tensor,
    moving: Tensor,
    fixed_domain: BSplineDomain,
    moving_domain: Optional[BSplineDomain] = None,
    *,
    transform_type: str = "Affine",
    similarity: str = "mse",
    neighborhood_radius: Union[int, Sequence[int]] = 2,
    iterations: Union[int, Sequence[int]] = (100, 100, 100),
    learning_rate: Union[float, Sequence[float]] = 1e-2,
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Optional[Union[float, Sequence[float]]] = None,
    multi_start: bool = True,
    center_of_mass_init: bool = True,
    padding_mode: str = "border",
    convergence_tolerance: Optional[float] = None,
    return_loss_history: bool = True,
    verbose: bool = False,
    detach_outputs: bool = True,
) -> Dict[str, object]:
    """Estimate a physical-space affine transform aligning ``moving`` onto ``fixed``.

    Images have shape ``(N, C, Y, X)`` or ``(N, C, Z, Y, X)``, exactly as in
    :func:`antstorch.bspline_flows.registration.registration`; each batch
    item is fit an independent affine transform (mirroring that function's
    per-batch-item coefficient lattices).

    ``transform_type`` selects the linear-transform hierarchy from
    :class:`antstorch.syn.core.affine.HierarchicalAffine`: ``'Translation'``,
    ``'Rigid'``, ``'Similarity'``, or ``'Affine'`` (default).

    Parameters
    ----------
    fixed, moving : Tensor
        Images to register, matching ``fixed_domain``/``moving_domain``.
    fixed_domain, moving_domain : BSplineDomain
        Physical metadata for ``fixed``/``moving``; ``moving_domain``
        defaults to ``fixed_domain``.
    transform_type : str
        Linear-transform hierarchy, see above.
    similarity : {'mse', 'ncc', 'ants_ncc'}
        Similarity metric, matching ``registration()``'s options.
    neighborhood_radius : int or sequence of int
        Passed through to ``ants_neighborhood_correlation_loss`` when
        ``similarity='ants_ncc'``.
    iterations, learning_rate, shrink_factors, smoothing_sigmas :
        Multi-resolution pyramid configuration, matching
        ``registration()``'s parameters of the same name.
    multi_start : bool
        If ``True``, seed the optimization from whichever of a small set of
        candidate rotations (identity plus a 180-degree rotation about each
        principal axis) scores best at the coarsest pyramid level, guarding
        against flip-related local minima. If ``False``, always start from
        the identity rotation.
    center_of_mass_init : bool
        If ``True``, initialize (and re-initialize, per rotation candidate)
        the translation so that ``fixed``'s center of mass maps onto
        ``moving``'s.
    padding_mode : {'border', 'zeros', 'reflection'}
        Out-of-bounds handling for the affine warp.
    convergence_tolerance : float, optional
        Per-level early-stopping tolerance, matching ``registration()``.
    return_loss_history : bool
        If ``True``, include ``loss_history``/``level_loss_history`` in the
        result.
    verbose : bool
        Print per-iteration progress.
    detach_outputs : bool
        If ``True`` (default), detach all returned tensors.

    Returns
    -------
    dict
        ``matrix`` (``(N, dim, dim)``) and ``translation`` (``(N, dim)``)
        give the fitted physical-space affine map
        ``p_moving = matrix @ p_fixed + translation`` per batch item — the
        pair to pass as ``initial_affine=(matrix, translation)`` to
        :func:`antstorch.bspline_flows.registration.registration`.
        ``warpedmovout`` is ``moving`` warped onto the fixed grid by this
        affine alone. ``fwdtransforms``/``invtransforms`` are the
        corresponding dense physical displacement fields (same convention as
        ``registration()``'s output), included for direct inspection.
        ``loss_history``/``level_loss_history`` are lists of per-batch-item
        histories.
    """
    if not isinstance(fixed_domain, BSplineDomain):
        raise TypeError("fixed_domain must be a BSplineDomain")
    moving_domain = fixed_domain if moving_domain is None else moving_domain
    if not isinstance(moving_domain, BSplineDomain):
        raise TypeError("moving_domain must be a BSplineDomain")
    if fixed_domain.dimension != moving_domain.dimension:
        raise ValueError("fixed_domain and moving_domain must have the same dimension")
    if not isinstance(fixed, Tensor) or not isinstance(moving, Tensor):
        raise TypeError("fixed and moving must be torch tensors")
    if fixed.ndim != fixed_domain.dimension + 2 or tuple(fixed.shape[2:]) != fixed_domain.torch_size:
        raise ValueError("fixed tensor shape does not match fixed_domain")
    if moving.ndim != moving_domain.dimension + 2 or tuple(moving.shape[2:]) != moving_domain.torch_size:
        raise ValueError("moving tensor shape does not match moving_domain")
    if fixed.shape[0] != moving.shape[0]:
        raise ValueError("fixed and moving must have the same batch size")
    if fixed.dtype != moving.dtype or fixed.device != moving.device:
        raise ValueError("fixed and moving must have the same dtype and device")
    if transform_type not in ("Translation", "Rigid", "Similarity", "Affine"):
        raise ValueError("transform_type must be 'Translation', 'Rigid', 'Similarity', or 'Affine'")
    if similarity not in ("mse", "ncc", "ants_ncc"):
        raise ValueError("similarity must be 'mse', 'ncc', or 'ants_ncc'")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")

    factors, sigmas, level_iterations, level_rates = _pyramid_configuration(
        shrink_factors, smoothing_sigmas, iterations, learning_rate
    )

    dimension = fixed_domain.dimension
    batch_size = fixed.shape[0]
    matrices, translations, histories, level_histories = [], [], [], []
    for batch_index in range(batch_size):
        matrix, translation, history, level_history = _fit_single_affine(
            fixed[batch_index : batch_index + 1],
            moving[batch_index : batch_index + 1],
            fixed_domain,
            moving_domain,
            transform_type=transform_type,
            similarity=similarity,
            neighborhood_radius=neighborhood_radius,
            factors=factors,
            sigmas=sigmas,
            level_iterations=level_iterations,
            level_rates=level_rates,
            multi_start=multi_start,
            center_of_mass_init=center_of_mass_init,
            padding_mode=padding_mode,
            convergence_tolerance=convergence_tolerance,
            verbose=verbose,
            batch_index=batch_index,
        )
        matrices.append(matrix)
        translations.append(translation)
        histories.append(history)
        level_histories.append(level_history)

    matrix = torch.stack(matrices, dim=0)
    translation = torch.stack(translations, dim=0)

    fwdtransforms = affine_displacement_field(matrix, translation, fixed_domain, fixed)
    inverse_matrix = torch.linalg.inv(matrix)
    inverse_translation = -torch.einsum("nij,nj->ni", inverse_matrix, translation)
    invtransforms = affine_displacement_field(inverse_matrix, inverse_translation, fixed_domain, fixed)
    warpedmovout = warp_image(moving, fwdtransforms, fixed_domain, moving_domain, padding_mode=padding_mode)

    result = {
        "matrix": matrix,
        "translation": translation,
        "warpedmovout": warpedmovout,
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        "loss_history": histories if return_loss_history else None,
        "level_loss_history": level_histories if return_loss_history else None,
    }
    if detach_outputs:
        result = {key: value.detach() if isinstance(value, Tensor) else value for key, value in result.items()}
    return result
