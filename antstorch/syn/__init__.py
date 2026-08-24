"""antstorch.syn — SyN-family diffeomorphic registration, ported from syntx.

Phase 1 of the syntx → ANTsTorch integration exposes only the shared
low-level primitives in :mod:`antstorch.syn.core` (grid sampling,
smoothing/regularization, similarity losses, Jacobian computation, field
inversion, and CFL-bounded optimizers). The top-level registration
entry point (``syn_registration``, mirroring ``ants.registration``'s
``warpedmovout`` / ``fwdtransforms`` / ``invtransforms`` output
convention) and the ``robust_affine`` initializer are planned for a later
phase and are not yet available here.

Exports are explicit and non-wildcard: ``antstorch/__init__.py`` does
``from .bspline_flows import *``, and this package must never silently
shadow names already exposed that way (``registration``, ``warp_image``,
``jacobian_determinant``, ``compose_displacements``, ``folding_count``,
``physical_grid``, ``displacement_to_sampling_grid``).
"""

from . import core

__all__ = [
    'core',
]
