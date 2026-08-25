"""antstorch.syn — SyN-family diffeomorphic registration, ported from syntx.

Phase 1 of the syntx -> ANTsTorch integration exposed only the shared
low-level primitives in :mod:`antstorch.syn.core` (grid sampling,
smoothing/regularization, similarity losses, Jacobian computation, field
inversion, and CFL-bounded optimizers). Etape 2 adds the top-level
registration entry point, :func:`syn_registration` (mirroring
``ants.registration``'s ``warpedmovout`` / ``fwdtransforms`` /
``invtransforms`` output convention, but with in-memory tensors throughout
rather than files on disk) plus the ``ants.ANTsImage`` <-> tensor bridge in
:mod:`antstorch.syn.bridge`. Per the integration proposal's Etape 2 scope,
the affine initialization stage reuses/extends
:func:`antstorch.bspline_flows.affine_registration.affine_registration`
rather than porting ``syntx``'s separate ``robust_affine`` module.

Exports are explicit and non-wildcard: ``antstorch/__init__.py`` does
``from .bspline_flows import *``, and this package must never silently
shadow names already exposed that way (``bspline_svf_registration``, ``warp_image``,
``jacobian_determinant``, ``compose_displacements``, ``folding_count``,
``physical_grid``, ``displacement_to_sampling_grid``).
"""

from . import core
from .bridge import (
    ants_image_metadata,
    ants_image_to_tensor,
    apply_bspline_smoothing_operator,
    displacement_xyz_to_ants_image,
    displacement_zyx_to_ants_image,
    flip_affine_xyz_to_zyx,
    image_domain_from_metadata,
    metadata_tensors,
    metadata_tensors_from_dict,
    tensor_to_ants_image,
)
from .syn import syn_registration

__all__ = [
    'core',
    'syn_registration',
    'ants_image_metadata',
    'ants_image_to_tensor',
    'tensor_to_ants_image',
    'displacement_zyx_to_ants_image',
    'displacement_xyz_to_ants_image',
    'metadata_tensors',
    'metadata_tensors_from_dict',
    'flip_affine_xyz_to_zyx',
    'apply_bspline_smoothing_operator',
    'image_domain_from_metadata',
]
