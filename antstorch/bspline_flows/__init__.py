"""Differentiable ITK-compatible cubic B-spline field synthesis."""

from .affine_registration import affine_registration
from .bspline_domain import ImageDomain, mesh_size_for_spline_distance
from .bspline_scattered_data import fit_bspline_displacement_field, fit_bspline_object_to_scattered_data
from .bspline_synthesis import (
    CubicBSplineSynthesis,
    cubic_bspline_basis,
    fit_bspline_coefficients,
    refine_bspline_coefficients,
    synthesize_bspline_velocity,
)
from .deterministic_registration import DeterministicBSplineRegistration
from .n4_bias_field_correction import DEFAULT_N4_SPLINE_DISTANCE_MM, N4BiasFieldCorrection, n4_bias_field_correction
from .physical_gradient_descent import PhysicalGradientDescent
from .gaussian_svf_registration import gaussian_svf_registration
from .registration import DEFAULT_BSPLINE_SPLINE_DISTANCE_MM, bspline_svf_registration
from .scaling_and_squaring import ScalingAndSquaring, scaling_and_squaring
from .similarity import (
    ants_neighborhood_correlation_loss,
    bending_energy,
    mean_squared_error,
    normalized_cross_correlation_loss,
    squared_l2_energy,
)
from .spatial_transform import (
    ALIGN_CORNERS,
    affine_displacement_field,
    compose_displacements,
    displacement_to_sampling_grid,
    folding_count,
    jacobian_determinant,
    physical_grid,
    warp_image,
)

__all__ = [
    "affine_registration",
    "ImageDomain",
    "mesh_size_for_spline_distance",
    "CubicBSplineSynthesis",
    "cubic_bspline_basis",
    "fit_bspline_coefficients",
    "fit_bspline_displacement_field",
    "fit_bspline_object_to_scattered_data",
    "refine_bspline_coefficients",
    "synthesize_bspline_velocity",
    "DeterministicBSplineRegistration",
    "N4BiasFieldCorrection",
    "n4_bias_field_correction",
    "DEFAULT_N4_SPLINE_DISTANCE_MM",
    "PhysicalGradientDescent",
    "gaussian_svf_registration",
    "bspline_svf_registration",
    "DEFAULT_BSPLINE_SPLINE_DISTANCE_MM",
    "ScalingAndSquaring",
    "scaling_and_squaring",
    "bending_energy",
    "ants_neighborhood_correlation_loss",
    "mean_squared_error",
    "normalized_cross_correlation_loss",
    "squared_l2_energy",
    "ALIGN_CORNERS",
    "affine_displacement_field",
    "compose_displacements",
    "displacement_to_sampling_grid",
    "folding_count",
    "jacobian_determinant",
    "physical_grid",
    "warp_image",
]
