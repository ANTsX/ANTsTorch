"""Differentiable ITK-compatible cubic B-spline field synthesis."""

from .bspline_domain import BSplineDomain
from .bspline_scattered_data import fit_bspline_displacement_field, fit_bspline_object_to_scattered_data
from .bspline_synthesis import (
    CubicBSplineSynthesis,
    cubic_bspline_basis,
    fit_bspline_coefficients,
    refine_bspline_coefficients,
    synthesize_bspline_velocity,
)
from .deterministic_registration import DeterministicBSplineRegistration
from .n4_bias_field_correction import N4BiasFieldCorrection, n4_bias_field_correction
from .scaling_and_squaring import ScalingAndSquaring, scaling_and_squaring
from .similarity import bending_energy, mean_squared_error, normalized_cross_correlation_loss, squared_l2_energy
from .spatial_transform import (
    ALIGN_CORNERS,
    compose_displacements,
    displacement_to_sampling_grid,
    folding_count,
    jacobian_determinant,
    physical_grid,
    warp_image,
)

__all__ = [
    "BSplineDomain",
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
    "ScalingAndSquaring",
    "scaling_and_squaring",
    "bending_energy",
    "mean_squared_error",
    "normalized_cross_correlation_loss",
    "squared_l2_energy",
    "ALIGN_CORNERS",
    "compose_displacements",
    "displacement_to_sampling_grid",
    "folding_count",
    "jacobian_determinant",
    "physical_grid",
    "warp_image",
]
