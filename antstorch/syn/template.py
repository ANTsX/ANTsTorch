"""Unbiased group-template construction with ANTsTorch SyN.

Port of ``syntx.template.build_template`` (itself a port of
``ants.build_template``) onto :func:`antstorch.syn.syn_registration`.  Scope of
this port (decided 2026-10-08): the SyN / SyNOnly path of the PyTorch backend
only.  ``syntx``'s one-directional ``type_of_transform='Greedy'`` mode
(``syntx.greedy`` is not ported) and its ``backend='ants'`` fallback
(``ants.build_template`` already covers that) are intentionally left out.

Each iteration registers every image to the current template, averages the
warped images, then moves that average back by ``gradient_step`` times the mean
(negated) warp and the inverse of the average affine, so the template drifts
toward the centre of shape of the group.  Per iteration the size and roughness
of the mean warp (RMS displacement, membrane and bending energy) and the mean
absolute template change are recorded.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ants
import numpy as np

__all__ = ["build_template"]

_MANAGED_KWARGS = ("initial_affine", "outprefix")


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    """Normalise per-image weights so they sum to 1 (uniform when ``None``)."""
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"len(weights)={w.size} must equal len(image_list)={n}")
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must have a positive sum.")
    return w / total


def _initial_template(image_list: List[ants.ANTsImage], weights: Sequence[float]) -> ants.ANTsImage:
    """Weighted average of ``image_list``, resampled onto the first image."""
    template = image_list[0] * 0
    for image, weight in zip(image_list, weights):
        template = template + ants.resample_image_to_target(image * float(weight), template)
    return template


def _template_change(old: ants.ANTsImage, new: ants.ANTsImage) -> float:
    """Mean absolute voxel-intensity change between two templates."""
    return float(np.mean(np.abs(old.numpy().astype(np.float64) - new.numpy().astype(np.float64))))


def _compute_shape_residual(mean_warp: ants.ANTsImage) -> Dict[str, float]:
    """Size and roughness of the mean warp field.

    ``l2_norm`` is the RMS displacement magnitude (physical units),
    ``membrane_energy`` the mean squared Frobenius norm of the Jacobian of the
    displacement and ``bending_energy`` that of its Hessian.
    """
    arr = mean_warp.numpy().astype(np.float64)  # (*spatial, dim)
    spacing = [float(s) for s in mean_warp.spacing]
    dim = arr.shape[-1]

    l2_norm = float(np.sqrt(np.mean(np.sum(arr**2, axis=-1))))

    membrane = 0.0
    bending = 0.0
    for k in range(dim):
        component = arr[..., k]
        for j in range(dim):
            grad_j = np.gradient(component, spacing[j], axis=j)
            membrane += float(np.mean(grad_j**2))
            for i in range(dim):
                bending += float(np.mean(np.gradient(grad_j, spacing[i], axis=i) ** 2))
    return {"l2_norm": l2_norm, "membrane_energy": membrane, "bending_energy": bending}


def build_template(
    initial_template: Optional[ants.ANTsImage] = None,
    image_list: Optional[List[ants.ANTsImage]] = None,
    iterations: int = 3,
    gradient_step: float = 0.2,
    blending_weight: float = 0.75,
    weights: Optional[Sequence[float]] = None,
    use_no_rigid: bool = True,
    output_dir: Optional[str] = None,
    type_of_transform: str = "SyN",
    convergence_threshold: float = 0.0,
    affine_every_iteration: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build an unbiased group template with :func:`antstorch.syn.syn_registration`.

    Parameters
    ----------
    initial_template : ants.ANTsImage, optional
        Starting template.  Default: the weighted average of ``image_list``
        (resampled onto the first image).
    image_list : list of ants.ANTsImage
        The images (required, non-empty).
    iterations : int, default 3
    gradient_step : float in [0, 1], default 0.2
        Fraction of the mean warp applied to the template each iteration.
    blending_weight : float in (0, 1], default 0.75
        ``template = w * template + (1 - w) * sharpened(template)``;
        ``1`` disables sharpening.
    weights : sequence of float, optional
        Per-image weights (normalised to sum to 1; default uniform).
    use_no_rigid : bool, default True
        Average the affines without their rigid part
        (``ants.average_affine_transform_no_rigid``, falling back to
        ``ants.average_affine_transform`` if that fails).  Equivalent to
        ``useNoRigid`` in ``ants.build_template`` / ``syntx.build_template``.
    output_dir : str, optional
        Where per-iteration transforms are written (default: a temp directory).
    type_of_transform : str, default 'SyN'
        Per-image registration type, passed to ``syn_registration``.  With
        ``'SyN'`` the first iteration fits each image's affine; later
        iterations run ``'SyNOnly'`` starting from that cached affine (unless
        ``affine_every_iteration``), which avoids affine drift as the template
        moves.  Linear-only types (``'Rigid'``, ``'Affine'``, ...) are accepted:
        no mean warp exists then and only the affine is re-centred.
    convergence_threshold : float, default 0.0
        Stop early when the mean warp's RMS size falls below this (``0`` = never).
    affine_every_iteration : bool, default False
        ``True`` runs the full ``type_of_transform`` (affine included) every iteration.
    verbose : bool, default False
    **kwargs
        Forwarded to ``syn_registration`` (``regularizer``, ``syn_metric``,
        ``levels``, ``reg_iterations``, ``device``, ...).  ``initial_affine`` and
        ``outprefix`` are managed here and rejected.

    Returns
    -------
    dict
        ``template``; ``warped_images`` / ``fwdtransforms`` / ``invtransforms``
        (per image, last iteration); ``convergence`` (mean absolute template
        change per iteration); ``shape_residuals`` (per iteration: ``l2_norm``,
        ``membrane_energy``, ``bending_energy`` of the mean warp);
        ``n_iterations``; ``weights``; ``work_dir``; ``elapsed_sec``.
    """
    from .syn import syn_registration  # local import: keeps ``antstorch.syn`` import order simple

    t0 = time.time()
    if image_list is None or len(image_list) == 0:
        raise ValueError("image_list must be a non-empty list of ANTsImages.")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not (0.0 < blending_weight <= 1.0):
        raise ValueError(f"blending_weight must be in (0, 1], got {blending_weight}")
    if not (0.0 <= gradient_step <= 1.0):
        raise ValueError(f"gradient_step must be in [0, 1], got {gradient_step}")
    for name in _MANAGED_KWARGS:
        if name in kwargs:
            raise TypeError(f"build_template manages {name!r} itself; remove it from the keyword arguments")

    n = len(image_list)
    weights = _normalize_weights(weights, n)

    work_dir = tempfile.mkdtemp(prefix="antstorch_tmpl_") if output_dir is None else output_dir
    os.makedirs(work_dir, exist_ok=True)

    def make_outprefix(it: int, k: int) -> str:
        d = os.path.join(work_dir, f"iter{it:02d}_img{k:04d}")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "out")

    xavg = _initial_template(image_list, weights) if initial_template is None else initial_template.clone()

    shape_residuals: List[Dict[str, float]] = []
    convergence: List[float] = []
    last_warped: List[Optional[ants.ANTsImage]] = [None] * n
    last_fwd: List[List[str]] = [[] for _ in range(n)]
    last_inv: List[List[str]] = [[] for _ in range(n)]
    # Each image's affine from its last full registration, as an ITK-order
    # (matrix, translation) pair, fed back as ``initial_affine`` to 'SyNOnly'.
    image_affine: List[Optional[Tuple[Any, Any]]] = [None] * n

    is_syn = type_of_transform.lower() == "syn"
    for it in range(iterations):
        if verbose:
            print(f"[build_template] Iteration {it + 1}/{iterations}")
        iter_tot = "SyNOnly" if (is_syn and it > 0 and not affine_every_iteration) else type_of_transform
        affine_files: List[str] = []
        wavg: Optional[ants.ANTsImage] = None
        xavg_new: Optional[ants.ANTsImage] = None
        has_field = False

        for k in range(n):
            if verbose:
                print(f"  Registering subject {k + 1}/{n} ...", end=" ", flush=True)
            reg_kw = dict(kwargs)
            if iter_tot == "SyNOnly" and image_affine[k] is not None:
                reg_kw["initial_affine"] = image_affine[k]
            w1 = syn_registration(
                xavg,
                image_list[k],
                type_of_transform=iter_tot,
                outprefix=make_outprefix(it, k),
                verbose=verbose,
                **reg_kw,
            )
            has_field = w1["jacobian"] is not None  # dense stage ran
            # fwdtransforms = [warp, affine] (SyN), [warp] (SyNOnly, no affine) or [affine] (linear only)
            if w1["fwdtransforms"][-1].endswith(".mat"):
                affine_files.append(w1["fwdtransforms"][-1])
            if iter_tot != "SyNOnly" and w1["affine_matrix"] is not None:
                image_affine[k] = (w1["affine_matrix"], w1["affine_translation"])

            last_warped[k], last_fwd[k], last_inv[k] = w1["warpedmovout"], w1["fwdtransforms"], w1["invtransforms"]

            weight = float(weights[k])
            warped = w1["warpedmovout"] * weight
            xavg_new = warped if xavg_new is None else xavg_new + warped
            if has_field:
                field = ants.image_read(w1["fwdtransforms"][0]) * weight
                wavg = field if wavg is None else wavg + field
            if verbose:
                print("done")

        if has_field and wavg is not None:
            sr = _compute_shape_residual(wavg)
        else:
            sr = {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}
        shape_residuals.append(sr)
        if verbose:
            print(
                f"  Iteration {it} shape residual - L2: {sr['l2_norm']:.4f}, "
                f"membrane: {sr['membrane_energy']:.6f}, bending: {sr['bending_energy']:.8f}"
            )

        xavg_pre = xavg.clone()

        affine_fn: Optional[str] = None
        if affine_files:
            try:
                if use_no_rigid:
                    avg_affine = ants.average_affine_transform_no_rigid(affine_files)
                else:
                    avg_affine = ants.average_affine_transform(affine_files)
            except Exception:
                if not use_no_rigid:
                    raise
                avg_affine = ants.average_affine_transform(affine_files)
            affine_fn = os.path.join(work_dir, f"avgAffine_{it}.mat")
            ants.write_transform(avg_affine, affine_fn)

        if has_field and wavg is not None:
            scaled = wavg * ((-1.0) * gradient_step)
            if affine_fn is not None:
                scaled = ants.apply_transforms(
                    fixed=xavg_new, moving=scaled, imagetype=1, transformlist=affine_fn, whichtoinvert=[1]
                )
            warp_fn = os.path.join(work_dir, f"avgWarp_{it}.nii.gz")
            ants.image_write(scaled, warp_fn)
            transforms = [warp_fn] + ([affine_fn] if affine_fn is not None else [])
            invert = [0] + ([1] if affine_fn is not None else [])
            xavg = ants.apply_transforms(
                fixed=xavg_new, moving=xavg_new, transformlist=transforms, whichtoinvert=invert
            )
        elif affine_fn is not None:
            xavg = ants.apply_transforms(
                fixed=xavg_new, moving=xavg_new, transformlist=[affine_fn], whichtoinvert=[1]
            )
        else:
            xavg = xavg_new

        if blending_weight < 1.0:
            xavg = xavg * blending_weight + ants.iMath(xavg, "Sharpen") * (1.0 - blending_weight)

        change = _template_change(xavg_pre, xavg)
        convergence.append(change)
        if verbose:
            print(f"  Template mean absolute change: {change:.6f}")

        if convergence_threshold > 0.0 and sr["l2_norm"] < convergence_threshold:
            if verbose:
                print(f"  Converged: shape residual {sr['l2_norm']:.6f} < {convergence_threshold:.6f}")
            break

    return {
        "template": xavg,
        "warped_images": last_warped,
        "fwdtransforms": last_fwd,
        "invtransforms": last_inv,
        "convergence": convergence,
        "shape_residuals": shape_residuals,
        "n_iterations": len(shape_residuals),
        "weights": weights,
        "work_dir": work_dir,
        "elapsed_sec": time.time() - t0,
    }
