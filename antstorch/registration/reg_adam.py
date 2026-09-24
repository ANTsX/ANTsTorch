"""Functional RegAdam update shared by registration algorithms."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class RegAdamState:
    """Per-element first and second moments for a RegAdam update."""

    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor

    @classmethod
    def zeros_like(cls, value: Tensor) -> "RegAdamState":
        """Create a zero-initialized state matching ``value``."""
        return cls(
            step=0,
            exp_avg=torch.zeros_like(value),
            exp_avg_sq=torch.zeros_like(value),
        )


@torch.no_grad()
def reg_adam_direction(
    gradient: Tensor,
    state: Optional[RegAdamState] = None,
    *,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Tuple[RegAdamState, Tensor]:
    """Return Adam's bias-corrected direction for a registration force.

    The function deliberately stops before spatial regularization and CFL
    normalization.  Those operations depend on the field convention and the
    registration model, whereas the moment update is identical for SyN,
    DiReCT, and future registration algorithms.

    Parameters
    ----------
    gradient : torch.Tensor
        Raw similarity gradient or analytical registration force.
    state : RegAdamState, optional
        Existing moments. A new zero state is allocated when omitted.
    betas : pair of float
        First- and second-moment decay coefficients.
    eps : float
        Numerical stability term in the Adam denominator.

    Returns
    -------
    state : RegAdamState
        Updated moment state. The supplied state is updated in place.
    direction : torch.Tensor
        Bias-corrected Adam direction, before regularization and CFL limiting.
    """
    beta1, beta2 = (float(betas[0]), float(betas[1]))
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("betas must lie in [0, 1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    if state is None:
        state = RegAdamState.zeros_like(gradient)
    elif state.exp_avg.shape != gradient.shape or state.exp_avg_sq.shape != gradient.shape:
        raise ValueError("RegAdam state and gradient must have the same shape")
    elif state.exp_avg.device != gradient.device or state.exp_avg_sq.device != gradient.device:
        raise ValueError("RegAdam state and gradient must be on the same device")
    elif state.exp_avg.dtype != gradient.dtype or state.exp_avg_sq.dtype != gradient.dtype:
        raise ValueError("RegAdam state and gradient must have the same dtype")

    state.step += 1
    state.exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    state.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1 ** state.step
    bias_correction2 = 1.0 - beta2 ** state.step
    denominator = (state.exp_avg_sq / bias_correction2).sqrt().add_(eps)
    direction = (state.exp_avg / bias_correction1) / denominator
    return state, direction
