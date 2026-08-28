"""Shared ANTsX-format transform export helpers.

This module is the one place ``antstorch`` writes real, file-based ANTs
transforms — the classic ``antsRegistration`` convention of a numbered
affine ``.mat`` plus forward/inverse displacement-field ``.nii.gz`` files,
rather than a single in-memory composed field. It exists so that
``antstorch.syn.syn_registration()`` (and, by explicit request, any other
ANTsTorch registration entry point that chooses to use it) can hand back
transforms that interoperate with the rest of the ANTsX ecosystem —
``ants.apply_transforms(transformlist=...)``, ``antsApplyTransforms`` on the
command line, or any other tool that expects real transform files — exactly
as ``ants.registration()`` itself does.

Ground truth for the conventions replicated here was taken directly from
``ants.registration()``'s own implementation (``antspyx``'s
``ants.core.ants_registration``), not guessed or inferred:

- File naming: ``{outprefix}0GenericAffine.mat``, ``{outprefix}1Warp.nii.gz``,
  ``{outprefix}1InverseWarp.nii.gz`` (the leading digit is antsRegistration's
  own stage counter; ``0`` = the affine stage, ``1`` = the deformable stage).
- List construction: ``fwdtransforms = [warp, affine]`` (deformation first,
  affine last) and ``invtransforms = [affine, inverse_warp]`` (affine first,
  inverse deformation last) — **the same affine file is reused in both
  lists**. This is deliberate on ANTs' part: ``ants.apply_transforms()``'s
  ``whichtoinvert`` parameter defaults to ``(True, False)`` precisely when a
  transform list is "a matrix followed by a warp field", so passing
  ``invtransforms`` to ``ants.apply_transforms()`` inverts the shared affine
  automatically, with no explicit ``whichtoinvert`` needed from the caller.
- Affine parameter layout: an ITK ``AffineTransform``'s parameters are the
  row-major-raveled linear matrix followed by the translation vector, with
  fixed parameters (the rotation center) left at the origin, since
  ``antstorch``'s affines are already expressed about the physical origin.

This mirrors (and was validated against) ``syntx``'s own
``export_ants_affine_transform`` helper in ``transform.py``, adjusted to
write only the forward affine file — the inverse direction is represented by
reusing that same file with ``whichtoinvert``, exactly like
``ants.registration()``, rather than writing a second redundant file.
"""

import os
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import ants


def write_affine_transform(
    matrix: Union[torch.Tensor, np.ndarray],
    translation: Union[torch.Tensor, np.ndarray],
    dim: int,
    filename: str,
) -> "ants.core.ants_transform.ANTsTransform":
    """Write a physical-space affine ``(matrix, translation)`` as an ITK ``...GenericAffine.mat`` file.

    ``matrix``/``translation`` follow the same convention used throughout
    ``antstorch.bspline_flows``/``antstorch.syn``: ``matrix`` maps a physical
    point ``p`` via ``p_mapped = matrix @ p + translation``, in ITK
    ``(x, y[, z])`` physical order. Only the forward direction is ever
    written to disk; callers needing the inverse should reuse this same file
    with ``ants.apply_transforms(..., whichtoinvert=[True, ...])`` (or rely
    on its default heuristic — see the module docstring) rather than writing
    a second file, matching ``ants.registration()``'s own convention.

    Parameters
    ----------
    matrix : Tensor or ndarray, shape ``(dim, dim)``
    translation : Tensor or ndarray, shape ``(dim,)``
    dim : int
        Spatial dimensionality (2 or 3).
    filename : str
        Destination path, e.g. ``f"{outprefix}0GenericAffine.mat"``. Parent
        directories are created if needed.

    Returns
    -------
    ants.ANTsTransform
        The forward transform object that was written.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    if isinstance(translation, torch.Tensor):
        translation = translation.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if matrix.shape != (dim, dim):
        raise ValueError(f"matrix must have shape ({dim}, {dim}), got {tuple(matrix.shape)}")
    if translation.shape != (dim,):
        raise ValueError(f"translation must have shape ({dim},), got {tuple(translation.shape)}")

    transform = ants.new_ants_transform(precision="float", dimension=dim, transform_type="AffineTransform")
    transform.set_parameters(np.concatenate([matrix.ravel(), translation]))
    transform.set_fixed_parameters(np.zeros(dim))

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    ants.write_transform(transform, filename)
    return transform


def read_affine_transform(filename: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read an ITK ``...GenericAffine.mat`` file back into a physical-space ``(matrix, translation)`` pair.

    The exact inverse of :func:`write_affine_transform`: an ``AffineTransform``'s
    parameters are the row-major-raveled linear matrix followed by the
    translation vector, ITK ``(x, y[, z])`` physical order — the fixed
    parameters (rotation center) are ignored, matching
    :func:`write_affine_transform` always leaving them at the origin.

    This is the format any consumer that shares the canonical affine as a
    file (rather than in-memory tensors) needs to bridge back into
    ``antstorch``'s own ``(matrix, translation)`` convention — e.g. handing
    a pre-computed ``.mat`` affine (such as ``syntx.robust_affine``'s
    output) to :func:`antstorch.syn.syn_registration`'s or
    :func:`antstorch.bspline_flows.bspline_svf_registration.bspline_svf_registration`'s
    ``initial_affine`` parameter, both of which expect this exact
    ``(matrix, translation)`` ITK-order pair.

    Parameters
    ----------
    filename : str
        Path to an ITK ``AffineTransform`` ``.mat`` file (2-D or 3-D).
    dim : int
        Spatial dimensionality (2 or 3); validated against the file's own
        transform dimension.

    Returns
    -------
    matrix : ndarray, shape ``(dim, dim)``, float64
    translation : ndarray, shape ``(dim,)``, float64
    """
    transform = ants.read_transform(filename)
    if transform.dimension != dim:
        raise ValueError(
            f"transform at '{filename}' has dimension {transform.dimension}, expected {dim}"
        )
    if transform.transform_type != "AffineTransform":
        raise ValueError(
            f"transform at '{filename}' has type '{transform.transform_type}', expected 'AffineTransform'"
        )
    params = np.asarray(transform.parameters, dtype=np.float64)
    matrix = np.ascontiguousarray(params[: dim * dim].reshape(dim, dim))
    translation = np.ascontiguousarray(params[dim * dim: dim * dim + dim])
    return matrix, translation


def default_outprefix() -> str:
    """A fresh temporary-file prefix, matching ``ants.registration()``'s own default.

    ``ants.registration()`` itself falls back to ``tempfile.mktemp()`` when
    no ``outprefix`` is supplied; this is that exact same fallback, exposed
    so callers building an ``outprefix`` default do not have to duplicate
    the choice of temp-file mechanism.
    """
    return tempfile.mktemp()


def build_transform_lists(
    *,
    affine_path: Optional[str],
    warp_path: Optional[str],
    inverse_warp_path: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Build ``(fwdtransforms, invtransforms)`` path lists matching ``ants.registration()``'s convention.

    Exactly reproduces the list construction found in ``ants.registration()``
    (moving-to-fixed forward list has the deformation first and the affine
    last; fixed-to-moving inverse list has the affine first and the inverse
    deformation last, with the *same* affine file reused in both lists —
    see the module docstring for why that specific ordering matters).

    Parameters
    ----------
    affine_path : str, optional
        Path a ``0GenericAffine.mat`` was written to, or ``None`` if no
        affine stage ran (e.g. ``'SyNOnly'`` with no ``initial_affine``).
    warp_path, inverse_warp_path : str, optional
        Paths the forward/inverse deformation fields were written to, or
        ``None`` if no deformable stage ran (a linear-only transform type).

    Returns
    -------
    fwdtransforms, invtransforms : list of str
    """
    if affine_path is None and warp_path is None:
        raise ValueError("at least one of affine_path/warp_path must be given")
    if warp_path is None:
        # Linear-only: the single affine file serves both directions.
        return [affine_path], [affine_path]
    if affine_path is None:
        # A pure deformable stage with no affine component at all.
        return [warp_path], [inverse_warp_path]
    return [warp_path, affine_path], [affine_path, inverse_warp_path]
