"""End-to-end deterministic B-spline stationary-velocity registration."""

from typing import Dict, Optional

from torch import Tensor, nn

from .bspline_domain import BSplineDomain
from .bspline_synthesis import CubicBSplineSynthesis
from .scaling_and_squaring import ScalingAndSquaring
from .similarity import (
    ants_neighborhood_correlation_loss,
    bending_energy,
    mean_squared_error,
    normalized_cross_correlation_loss,
    squared_l2_energy,
)
from .spatial_transform import jacobian_determinant, warp_image


class DeterministicBSplineRegistration(nn.Module):
    """Coefficients -> velocity -> Exp(v) -> warped moving image -> loss."""

    def __init__(
        self,
        fixed_domain: BSplineDomain,
        moving_domain: Optional[BSplineDomain] = None,
        *,
        squaring_steps: int = 7,
        similarity: str = "ants_ncc",
        neighborhood_radius=2,
        padding_mode: str = "zeros",
        coefficient_weight: float = 0.0,
        velocity_weight: float = 0.0,
        bending_weight: float = 0.0,
        closed=False,
        stationary_boundary: bool = True,
        synthesis_chunk_size: Optional[int] = 262144,
    ):
        super().__init__()
        self.fixed_domain = fixed_domain
        self.moving_domain = moving_domain or fixed_domain
        self.synthesis = CubicBSplineSynthesis(
            fixed_domain,
            closed=closed,
            stationary_boundary=stationary_boundary,
            chunk_size=synthesis_chunk_size,
        )
        self.exponential = ScalingAndSquaring(fixed_domain, squaring_steps)
        if similarity not in ("mse", "ncc", "ants_ncc"):
            raise ValueError("similarity must be 'mse', 'ncc', or 'ants_ncc'")
        self.similarity = similarity
        self.neighborhood_radius = neighborhood_radius
        self.padding_mode = padding_mode
        self.coefficient_weight = float(coefficient_weight)
        self.velocity_weight = float(velocity_weight)
        self.bending_weight = float(bending_weight)

    def transform(self, coefficients: Tensor, moving: Tensor) -> Dict[str, Tensor]:
        velocity = self.synthesis(coefficients)
        displacement = self.exponential(velocity)
        warped = warp_image(
            moving,
            displacement,
            self.fixed_domain,
            self.moving_domain,
            padding_mode=self.padding_mode,
        )
        return {"velocity": velocity, "displacement": displacement, "warped_moving": warped}

    def forward(self, coefficients: Tensor, moving: Tensor, fixed: Tensor) -> Dict[str, Tensor]:
        result = self.transform(coefficients, moving)
        if self.similarity == "mse":
            similarity = mean_squared_error(fixed, result["warped_moving"])
        elif self.similarity == "ncc":
            similarity = normalized_cross_correlation_loss(fixed, result["warped_moving"])
        else:
            similarity = ants_neighborhood_correlation_loss(
                fixed, result["warped_moving"], self.neighborhood_radius
            )
        coefficient_regularization = squared_l2_energy(coefficients)
        velocity_regularization = squared_l2_energy(result["velocity"])
        bending_regularization = bending_energy(result["velocity"], self.fixed_domain)
        loss = (
            similarity
            + self.coefficient_weight * coefficient_regularization
            + self.velocity_weight * velocity_regularization
            + self.bending_weight * bending_regularization
        )
        result.update(
            {
                "loss": loss,
                "similarity": similarity,
                "coefficient_regularization": coefficient_regularization,
                "velocity_regularization": velocity_regularization,
                "bending_regularization": bending_regularization,
                "jacobian_determinant": jacobian_determinant(result["displacement"], self.fixed_domain),
            }
        )
        return result
