"""Independent CPU reference utilities matching ITK's reconstruction loop.

These helpers are deliberately outside the synthesis/autograd path.  The
NumPy evaluator is useful for reference fixtures.  ``coefficient_lattice_metadata``
reproduces ITK's reported coefficient-image geometry for initialization and
interchange with ITK.
"""

from itertools import product
from typing import Sequence

import numpy as np

from .bspline_domain import ImageDomain


def _kernel(value):
    value = abs(float(value))
    if value < 1.0:
        return (4.0 - 6.0 * value**2 + 3.0 * value**3) / 6.0
    if value < 2.0:
        return (2.0 - value) ** 3 / 6.0
    return 0.0


def numpy_itk_reconstruction(coefficients: np.ndarray, output_size: Sequence[int], closed=False):
    """Literal scalar-loop reconstruction in ITK x-y-z array order.

    Input shape is ``(Kx, Ky[, Kz], C)`` and output shape is
    ``(X, Y[, Z], C)``. This intentionally differs from PyTorch storage order.
    """
    dimension = len(output_size)
    closed = (closed,) * dimension if isinstance(closed, bool) else tuple(closed)
    lattice_size = coefficients.shape[:dimension]
    result = np.zeros(tuple(output_size) + (coefficients.shape[-1],), dtype=coefficients.dtype)
    for dense_index in np.ndindex(*output_size):
        axis_indices, axis_weights = [], []
        for d in range(dimension):
            spans = lattice_size[d] if closed[d] else lattice_size[d] - 3
            u = spans * dense_index[d] / (output_size[d] - 1)
            if dense_index[d] == output_size[d] - 1:
                u = np.nextafter(float(spans), -np.inf)
            base = int(u)
            axis_indices.append([(base + j) % lattice_size[d] if closed[d] else base + j for j in range(4)])
            axis_weights.append([_kernel(u - base - j + 1.0) for j in range(4)])
        for support in product(range(4), repeat=dimension):
            index = tuple(axis_indices[d][support[d]] for d in range(dimension))
            weight = np.prod([axis_weights[d][support[d]] for d in range(dimension)])
            result[dense_index] += coefficients[index] * weight
    return result


def coefficient_lattice_metadata(domain: ImageDomain, lattice_size: Sequence[int], closed=False):
    """Return ITK-order coefficient-lattice size, spacing, origin, direction."""
    closed = (closed,) * domain.dimension if isinstance(closed, bool) else tuple(closed)
    spans = tuple(k if periodic else k - 3 for k, periodic in zip(lattice_size, closed))
    spacing = tuple(extent / span for extent, span in zip(domain.physical_extent, spans))
    shift = np.asarray([-value for value in spacing])  # -0.5 * spacing * (3 - 1)
    origin = np.asarray(domain.origin) + np.asarray(domain.direction).dot(shift)
    return {
        "size": tuple(lattice_size),
        "spacing": spacing,
        "origin": tuple(origin.tolist()),
        "direction": domain.direction,
    }

