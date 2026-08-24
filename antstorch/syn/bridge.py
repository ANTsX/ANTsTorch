"""``ants.ANTsImage`` <-> in-memory tensor bridge for :mod:`antstorch.syn`.

``antstorch.syn.core`` operates purely on PyTorch tensors and plain
metadata tuples; this module is the (thin, explicit) boundary that lets
:func:`antstorch.syn.syn.syn_registration` accept and return
``ants.ANTsImage`` objects, matching ``ants.registration()``'s calling
convention, while every actual computation stays in memory (no temporary
NIfTI files are ever written) per the project's in-memory-tensor scope.

Two axis-order conventions are bridged here:

- ``ants.ANTsImage`` metadata (spacing/origin/direction) and its ``.numpy()``
  array are in ITK ``(x, y[, z])`` order; PyTorch tensor storage reverses the
  *spatial* axes to ``(z, y, x)`` (see ``antstorch.bspline_flows`` for the
  same convention).
- ``antstorch.syn.core``'s physical-space affine matrices/translations
  (``M_phys``/``t_phys``, e.g. as consumed by
  ``prepare_mid_images_and_gradients_torch``) and its displacement-field
  *component* order both follow that same reversed ``(z, y, x)`` convention
  — unlike ``antstorch.bspline_flows``, whose vector components stay in ITK
  ``(x, y[, z])`` order throughout. :func:`flip_affine_xyz_to_zyx` converts
  between the two.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _percentile_clip_normalize(array: np.ndarray) -> np.ndarray:
    """Foreground (``> 0``) 2nd-98th percentile clip to ``[0, 1]``.

    Matches :func:`antstorch.syn.core.pipeline.normalize_and_tensorize`'s
    actual normalization exactly, factored out so it can be applied to a
    single image (not just a fixed/moving pair).
    """
    positive = array[array > 0]
    if positive.size > 0:
        low = float(np.percentile(positive, 2.0))
        high = float(np.percentile(positive, 98.0))
        if high <= low + 1e-4:
            low = 0.0
            high = float(positive.max())
    else:
        low = float(array.min())
        high = float(array.max())
    return np.clip((array - low) / (high - low + 1e-6), 0.0, 1.0).astype(np.float32)


def ants_image_metadata(image) -> Dict[str, tuple]:
    """Return an ``ants.ANTsImage``'s spatial metadata as plain ITK-order tuples.

    Returns
    -------
    dict
        ``{'dimension', 'shape', 'torch_shape', 'spacing', 'origin', 'direction'}``:
        ``shape``/``spacing``/``origin`` are ITK ``(x, y[, z])`` order;
        ``torch_shape`` is the PyTorch-reversed spatial shape; ``direction``
        is a nested ITK-order tuple.
    """
    dimension = image.dimension
    shape = tuple(int(value) for value in image.shape)
    return {
        "dimension": dimension,
        "shape": shape,
        "torch_shape": tuple(reversed(shape)),
        "spacing": tuple(float(value) for value in image.spacing),
        "origin": tuple(float(value) for value in image.origin),
        "direction": tuple(tuple(float(value) for value in row) for row in np.asarray(image.direction).tolist()),
    }


def ants_image_to_tensor(
    image,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    *,
    normalize: bool = True,
) -> Tensor:
    """Convert a scalar ``ants.ANTsImage`` to a singleton ``(1, 1, *torch_shape)`` tensor.

    Parameters
    ----------
    image : ants.ANTsImage
        Scalar 2-D or 3-D image.
    device, dtype : torch.device, torch.dtype
        Target tensor device/dtype.
    normalize : bool
        If ``True`` (default), apply the same foreground percentile-clip
        normalization as :func:`antstorch.syn.core.pipeline.normalize_and_tensorize`.

    Returns
    -------
    Tensor
        Shape ``(1, 1, *torch_shape)``, spatial axes in reversed
        (``(z, y, x)``-style) PyTorch order.
    """
    if image.dimension not in (2, 3) or image.components != 1:
        raise ValueError("ants_image_to_tensor expects a scalar 2-D or 3-D ANTsImage")
    array = image.numpy().astype(np.float32, copy=False)
    if normalize:
        array = _percentile_clip_normalize(array)
    dimension = image.dimension
    reversed_axes = tuple(range(dimension - 1, -1, -1))
    array = np.ascontiguousarray(np.transpose(array, reversed_axes))
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def tensor_to_ants_image(tensor: Tensor, reference):
    """Convert a singleton ``(1, 1, *torch_shape)`` tensor to a scalar ``ants.ANTsImage``.

    Parameters
    ----------
    tensor : Tensor
        Shape ``(1, 1, *torch_shape)``.
    reference : ants.ANTsImage
        Supplies spacing/origin/direction for the output image.

    Returns
    -------
    ants.ANTsImage
    """
    import ants

    if tensor.ndim != reference.dimension + 2 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError("tensor must have shape (1, 1, *torch_shape)")
    dimension = reference.dimension
    reversed_axes = tuple(range(dimension - 1, -1, -1))
    array = np.ascontiguousarray(np.transpose(tensor.detach().cpu().numpy()[0, 0], reversed_axes))
    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def displacement_zyx_to_ants_image(field: Tensor, reference):
    """Convert a channel-last ``(1, *torch_shape, dim)`` ``(z, y, x)``-order physical field to an ``ants`` vector image.

    Parameters
    ----------
    field : Tensor
        Displacement field of shape ``(1, *torch_shape, dim)``, physical mm
        units, vector components in reversed ``(z, y, x)``-style order —
        ``antstorch.syn.core``'s convention.
    reference : ants.ANTsImage
        Supplies spacing/origin/direction for the output image.

    Returns
    -------
    ants.ANTsImage
        Vector image with components reordered to ITK ``(x, y[, z])``, as
        ``ants`` expects.
    """
    import ants

    dimension = reference.dimension
    if field.ndim != dimension + 2 or field.shape[0] != 1 or field.shape[-1] != dimension:
        raise ValueError(f"field must have shape (1, *torch_shape, {dimension})")
    reversed_axes = tuple(range(dimension - 1, -1, -1))
    array = field.detach().cpu().numpy()[0][..., ::-1]  # reverse component order z,y,x -> x,y,z
    array = np.ascontiguousarray(np.transpose(array, reversed_axes + (dimension,)))
    return ants.from_numpy(
        array.astype(np.float32, copy=False),
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
        has_components=True,
    )


def displacement_xyz_to_ants_image(field: Tensor, reference):
    """Convert a channel-first ``(1, dim, *torch_shape)`` ITK ``(x, y[, z])``-order physical field to an ``ants`` vector image.

    Bridges :mod:`antstorch.bspline_flows`'s own displacement-field
    convention (channel-first, ITK component order, e.g. the fields
    returned by ``affine_registration()``/``registration()``) directly to
    ``ants``, without going through the ``(z, y, x)``-order convention used
    by :func:`displacement_zyx_to_ants_image`.

    Parameters
    ----------
    field : Tensor
        Shape ``(1, dim, *torch_shape)``, physical mm units, ITK
        ``(x, y[, z])`` vector component order.
    reference : ants.ANTsImage
        Supplies spacing/origin/direction for the output image.

    Returns
    -------
    ants.ANTsImage
    """
    import ants

    dimension = reference.dimension
    if field.ndim != dimension + 2 or field.shape[0] != 1 or field.shape[1] != dimension:
        raise ValueError(f"field must have shape (1, {dimension}, *torch_shape)")
    array = field.detach().cpu().numpy()[0]  # (dim, *torch_shape)
    array = np.moveaxis(array, 0, -1)  # (*torch_shape, dim), ITK component order already
    reversed_axes = tuple(range(dimension - 1, -1, -1)) + (dimension,)
    array = np.ascontiguousarray(np.transpose(array, reversed_axes))
    return ants.from_numpy(
        array.astype(np.float32, copy=False),
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
        has_components=True,
    )


def metadata_tensors_from_dict(meta: Dict[str, tuple], device, dtype):
    """Same as :func:`metadata_tensors`, from an already-extracted metadata dict (e.g. a downsampled pyramid level)."""
    spacing_rev = tuple(reversed(meta["spacing"]))
    origin_rev = tuple(reversed(meta["origin"]))
    direction_rev = np.asarray(meta["direction"])[::-1, ::-1].copy()
    return {
        "shape_t": torch.tensor(list(meta["torch_shape"]), device=device, dtype=dtype),
        "spacing_t": torch.tensor(spacing_rev, device=device, dtype=dtype),
        "origin_t": torch.tensor(origin_rev, device=device, dtype=dtype),
        "direction_t": torch.tensor(direction_rev, device=device, dtype=dtype),
    }


def metadata_tensors(image, device, dtype):
    """Precompute the reversed-order metadata tensors consumed by cached ``antstorch.syn.core`` helpers.

    Returns
    -------
    dict
        ``{'shape_t', 'spacing_t', 'origin_t', 'direction_t'}`` — all in
        reversed ``(z, y, x)``-style order, as expected by
        ``physical_to_normalized_torch_cached``/
        ``prepare_mid_images_and_gradients_torch``.
    """
    return metadata_tensors_from_dict(ants_image_metadata(image), device, dtype)


def flip_affine_xyz_to_zyx(matrix: Tensor, translation: Tensor) -> Tuple[Tensor, Tensor]:
    """Reorder a physical-space affine transform from ITK ``(x, y[, z])`` to ``antstorch.syn.core``'s ``(z, y, x)`` convention.

    Conjugates ``matrix`` by the axis-reversal permutation and reverses
    ``translation``'s components. Used to bridge
    :func:`antstorch.bspline_flows.affine_registration.affine_registration`'s
    output (ITK ``(x, y[, z])`` order, per ``bspline_flows``' own vector
    convention) into the ``M_phys``/``t_phys`` order expected throughout
    ``antstorch.syn.core``. Self-inverse: applying it twice returns the
    original ordering.

    Parameters
    ----------
    matrix : Tensor
        Shape ``(..., dim, dim)``.
    translation : Tensor
        Shape ``(..., dim)``, matching ``matrix``'s leading (batch) shape.

    Returns
    -------
    tuple of Tensor
        ``(matrix_zyx, translation_zyx)``, same shapes as the inputs.
    """
    dim = matrix.shape[-1]
    permutation = torch.eye(dim, dtype=matrix.dtype, device=matrix.device).flip(0)
    matrix_zyx = torch.einsum("ij,...jk,kl->...il", permutation, matrix, permutation)
    translation_zyx = torch.einsum("ij,...j->...i", permutation, translation)
    return matrix_zyx, translation_zyx
