"""antstorch.benchmark.metrics — Topological and Overlap Evaluation Primitives
==============================================================================

Ported from ``syntx.deformation_metrics`` (which has no dependency on the
rest of ``syntx`` beyond ``ants``/``numpy``/``pandas``/``torch``), so this
module is a faithful copy rather than a reimplementation — the metric
definitions themselves are exactly what the ``syntx.benchmark`` Mindboggle-101
harness already used and reported, which is worth preserving verbatim for
anyone comparing results across the two harnesses.

Provides a unified API for evaluating the accuracy (Dice overlap) and
topological/energetic properties (Jacobian determinant, harmonic/bending
energy) of deformation fields produced by :mod:`antstorch.syn`/
:mod:`antstorch.bspline_flows`.
"""

import numpy as np
import torch
import ants
import pandas as pd


def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv=None):
    """Computes bidirectional fixed, moving, and symmetric mean Dice scores.

    Parameters
    ----------
    fl, ml : ants.ANTsImage
        Fixed/moving integer label maps (e.g. DKT31 cortical parcellations).
    fi, mi : ants.ANTsImage
        Fixed/moving intensity images (used only as the resampling grid).
    fwdtransforms, invtransforms : list of str
        Transform file paths, in ``ants.apply_transforms(transformlist=...)``
        order (e.g. ``[warp, affine]`` / ``[affine, inverse_warp]``).
    whichtoinvert_inv : list of bool, optional
        Passed to ``ants.apply_transforms`` for the inverse (moving-space)
        direction; defaults to inverting only the leading transform, which
        is correct for the ``[affine, inverse_warp]`` convention used
        throughout this package (see ``antstorch.ants_transform_io``).

    Returns
    -------
    dice_fixed, dice_moving, dice_sym : float
    """
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [True] + [False] * (len(invtransforms) - 1) if len(invtransforms) > 0 else []

    # 1. Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    vals_fixed = pd.to_numeric(df_fixed[col_fixed], errors='coerce').to_numpy(dtype=np.float64)
    vals_fixed = vals_fixed[np.isfinite(vals_fixed)]
    dice_fixed = float(np.mean(vals_fixed)) if len(vals_fixed) > 0 else 0.0

    # 2. Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ov_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    vals_moving = pd.to_numeric(df_moving[col_moving], errors='coerce').to_numpy(dtype=np.float64)
    vals_moving = vals_moving[np.isfinite(vals_moving)]
    dice_moving = float(np.mean(vals_moving)) if len(vals_moving) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def _to_numpy_warp(warp) -> np.ndarray:
    """Safely extracts a numpy array from an ANTsImage, PyTorch tensor, numpy array, or file path."""
    if isinstance(warp, str):
        try:
            warp = ants.image_read(warp)
        except Exception:
            raise ValueError(f"Could not load deformation field from path: {warp}")

    if hasattr(warp, 'numpy'):
        return warp.numpy()
    elif isinstance(warp, torch.Tensor):
        return warp.detach().cpu().numpy()
    return np.asarray(warp)


def compute_harmonic_energy(warp, spacing=None) -> float:
    """Domain-wide Harmonic (Membrane) Energy: the Frobenius norm of the
    Jacobian (1st-order spatial derivative) of a displacement field.

    Parameters
    ----------
    warp : ants.ANTsImage, torch.Tensor, np.ndarray, or str
        Displacement field array of shape ``[..., dim]``.
    spacing : tuple of float, optional
        Physical voxel spacing; auto-extracted if ``warp`` is an
        ``ants.ANTsImage``, else defaults to ``1.0`` per axis.

    Returns
    -------
    float
    """
    if spacing is None and hasattr(warp, 'spacing'):
        spacing = warp.spacing

    warp_np = _to_numpy_warp(warp)
    dim = warp_np.shape[-1]

    if spacing is None:
        spacing = (1.0,) * dim

    gradient_list = [np.gradient(warp_np[..., k], *spacing, axis=range(dim)) for k in range(dim)]

    total_hrm = 0.0
    for k in range(dim):
        for j in range(dim):
            grad_kj = gradient_list[k][j]
            total_hrm += float(np.mean(grad_kj**2))

    return total_hrm


def compute_bending_energy(warp, spacing=None) -> float:
    """Domain-wide Thin-Plate Bending Energy: the Frobenius norm of the
    Hessian (2nd-order spatial derivative) of a displacement field.

    Parameters
    ----------
    warp : ants.ANTsImage, torch.Tensor, np.ndarray, or str
        Displacement field array of shape ``[..., dim]``.
    spacing : tuple of float, optional
        Physical voxel spacing; auto-extracted if ``warp`` is an
        ``ants.ANTsImage``, else defaults to ``1.0`` per axis.

    Returns
    -------
    float
    """
    if spacing is None and hasattr(warp, 'spacing'):
        spacing = warp.spacing

    warp_np = _to_numpy_warp(warp)
    dim = warp_np.shape[-1]

    if spacing is None:
        spacing = (1.0,) * dim

    gradient_list = [np.gradient(warp_np[..., k], *spacing, axis=range(dim)) for k in range(dim)]

    total_bnd = 0.0
    for k in range(dim):
        for j in range(dim):
            grad_kj = gradient_list[k][j]
            grad2_kj = np.gradient(grad_kj, *spacing, axis=range(dim))
            for i in range(dim):
                total_bnd += float(np.mean(grad2_kj[i]**2))

    return total_bnd


def compute_jacobian_metrics(fixed_image, warp) -> dict:
    """Jacobian determinant statistics (min, max, mean, folding percentage)
    of a deformation field within the foreground mask of the fixed image.

    Parameters
    ----------
    fixed_image : ants.ANTsImage
        The fixed target image, used for geometry and foreground masking.
    warp : ants.ANTsImage or str
        The displacement field.

    Returns
    -------
    dict
        ``'min'``, ``'max'``, ``'mean'``, ``'folding_pct'``.
    """
    if isinstance(warp, str):
        warp = ants.image_read(warp)

    jac_ants = ants.create_jacobian_determinant_image(fixed_image, warp, do_log=False)
    jac_arr = jac_ants.numpy()

    # Restrict to foreground mask
    valid_mask = ants.get_mask(fixed_image).numpy() > 0
    if not np.any(valid_mask):
        valid_mask = np.ones_like(jac_arr, dtype=bool)

    valid_jac = jac_arr[valid_mask]

    return {
        "min": float(np.min(valid_jac)),
        "max": float(np.max(valid_jac)),
        "mean": float(np.mean(valid_jac)),
        "folding_pct": float(np.mean(valid_jac <= 0.0) * 100.0)
    }
