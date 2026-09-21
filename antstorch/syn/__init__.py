"""antstorch.syn — SyN-family diffeomorphic registration, ported from syntx.

Phase 1 of the syntx -> ANTsTorch integration exposed only the shared
low-level primitives in :mod:`antstorch.syn.core` (grid sampling,
smoothing/regularization, similarity losses, Jacobian computation, field
inversion, and CFL-bounded optimizers). Etape 2 adds the top-level
registration entry point, :func:`syn_registration` (mirroring
``ants.registration``'s ``warpedmovout`` / ``fwdtransforms`` /
``invtransforms`` output convention, but with in-memory tensors throughout
rather than files on disk) plus the ``ants.ANTsImage`` <-> tensor bridge in
:mod:`antstorch.syn.bridge`. :func:`robust_affine` additionally ports the
core PyTorch multi-start solver from ``syntx``'s ``robust_affine`` module
(cone-constrained rotation search, SE(3) candidate clustering, and a
multi-resolution Adam/L-BFGS Mattes-MI schedule). The port is scoped to
that solver only: landmark/SIFT3D-seeded candidates
(``syntx.landmarks``), ``mode='tournament'``, and the legacy multi-candidate
``ants.registration`` C++ fallback pipeline are intentionally excluded --
see :mod:`antstorch.syn.robust_affine` for the full scope note.

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
from .robust_affine import (
    robust_affine,
    robust_center_of_mass,
    compute_center_of_mass,
    compute_fov_center,
)

__all__ = [
    'core',
    'syn_registration',
    'robust_affine',
    'robust_center_of_mass',
    'compute_center_of_mass',
    'compute_fov_center',
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
