"""antstorch.syn.core — Shared low-level algorithmic primitives for SyN-family registration.

Ported from ``syntx.core`` (PyTorch backend only; the JAX backend and
deep-learning feature-space losses are intentionally excluded from this
phase-1 integration). This module is kept separate from
``antstorch.bspline_flows`` rather than fused into it: the two frameworks
share overlapping *concerns* (grid sampling, smoothing, Jacobians,
similarity losses) but different *conventions* (this module follows
``syntx``'s field/optimizer/inversion machinery for classical SyN-style
greedy diffeomorphic registration, while ``bspline_flows`` follows its own
B-spline SVF parameterization). Exports here are explicit and non-wildcard
so importing this module can never silently shadow names already exposed
by ``antstorch.bspline_flows`` (e.g. ``bspline_svf_registration``, ``warp_image``,
``jacobian_determinant``, ``physical_grid``).
"""

from .affine import (
    get_rotation_matrix,
    HierarchicalAffine,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    grid_to_physical_affine,
    parse_ants_affine,
    compute_initial_grid,
)
from .grid import (
    grid_sample_bspline_torch,
    AnalyticalGridSample,
    grid_sample_nd,
    compose_grids,
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    prepare_mid_images_and_gradients_torch,
)
from .smoothing import (
    separable_gaussian_filter,
    get_cached_gaussian_kernel_1d,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)
from .losses import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    local_ncc_loss_nd,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
)
from .jacobian import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_jacobian_hinge_penalty,
    compute_physical_jacobian_determinant,
)
from .inverse import (
    update_inverse_field_nd_hybrid_lm,
    integrate_time_varying_velocity_field,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
)
from .optimizers import (
    LARS,
    RegAdam,
    SobolevAdam,
    GaussianAdam,
    get_cfl_max_norm,
    compute_cfl_step,
    check_convergence,
)
from .pipeline import (
    auto_detect_device,
    normalize_and_tensorize,
    cleanup_gpu,
)
from .utils import (
    normalize_tensor,
    auto_select_intensity_percentiles,
    normalize_image,
)

__all__ = [
    'get_rotation_matrix',
    'HierarchicalAffine',
    'grid_to_physical_affine_torch',
    'physical_to_grid_affine',
    'grid_to_physical_affine',
    'parse_ants_affine',
    'compute_initial_grid',
    'grid_sample_bspline_torch',
    'AnalyticalGridSample',
    'grid_sample_nd',
    'compose_grids',
    'get_physical_grid_torch',
    'physical_to_normalized_torch',
    'physical_to_normalized_torch_cached',
    'prepare_mid_images_and_gradients_torch',
    'separable_gaussian_filter',
    'get_cached_gaussian_kernel_1d',
    'apply_sobolev_green_operator',
    'apply_dsti_green_operator',
    'apply_dsti1_green_operator',
    'get_boundary_mask',
    'AnalyticalLNCC',
    'ANTsPseudoLNCC',
    'local_ncc_loss_nd',
    'b_spline_3',
    'mattes_mi_loss_core',
    'mattes_mi_loss_nd',
    '_spatial_jacobian_nd',
    'compute_jacobian_determinant_nd',
    'compute_jacobian_hinge_penalty',
    'compute_physical_jacobian_determinant',
    'update_inverse_field_nd_hybrid_lm',
    'integrate_time_varying_velocity_field',
    'update_inverse_field_nd_anderson',
    'update_inverse_field_nd',
    'compute_inverse_identity_error_nd',
    'calculate_inverse_identity_error',
    'LARS',
    'RegAdam',
    'SobolevAdam',
    'GaussianAdam',
    'get_cfl_max_norm',
    'compute_cfl_step',
    'check_convergence',
    'auto_detect_device',
    'normalize_and_tensorize',
    'cleanup_gpu',
    'normalize_tensor',
    'auto_select_intensity_percentiles',
    'normalize_image',
]
