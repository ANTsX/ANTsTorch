"""Optimizers for velocity/deformation-field parameters.

Ported from ``syntx.core.optimizers`` (PyTorch backend only). ``RegAdam``
folds spatial regularization directly into the Adam step direction and
bounds the resulting physical displacement per the Courant-Friedrichs-Lewy
(CFL) condition, so that no single optimization step can move any point by
more than a fraction of a voxel.
"""

import math
import numpy as np
import torch


class LARS(torch.optim.Optimizer):
    r"""Layer-wise Adaptive Rate Scaling (LARS) optimizer for velocity parameters.

    Rescales each parameter's update magnitude by a trust-ratio:

    .. math::

        \text{trust\_ratio} = \eta \cdot
        \frac{\max(\lVert p \rVert_2, 1.0)}{\lVert g \rVert_2 + \epsilon}

    which helps prevent momentum collapse on the smooth loss plateaus that
    similarity metrics like LNCC can produce during deformable optimization.

    Parameters
    ----------
    params : iterable
        Parameters to optimize, or parameter-group dicts.
    lr : float
        Base learning rate.
    trust_coefficient : float
        Trust-ratio scaling factor :math:`\eta`.
    eps : float
        Numerical stability epsilon in the trust-ratio denominator.
    """

    def __init__(self, params, lr=0.80, trust_coefficient=0.05, eps=1e-8):
        defaults = dict(lr=lr, trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            trust_coeff = group['trust_coefficient']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                p_norm = torch.norm(p)
                g_norm = torch.norm(g)
                p_norm_effective = torch.clamp(p_norm, min=1.0)

                if g_norm > 0:
                    trust_ratio = trust_coeff * p_norm_effective / (g_norm + eps)
                else:
                    trust_ratio = 1.0

                local_lr = lr * trust_ratio
                p.sub_(g * local_lr)
        return loss


class RegAdam(torch.optim.Optimizer):
    """Adam with spatial regularization applied to the step direction, and a CFL step bound.

    Computes standard Adam first/second moments, applies the selected
    spatial regularizer (Gaussian, Sobolev, DST-I, or a custom callable)
    directly to the raw Adam step-direction quotient
    (``m_hat / (sqrt(v_hat) + eps)``), then rescales the resulting update so
    its maximum physical displacement never exceeds ``max_step_norm`` voxels
    (the Courant-Friedrichs-Lewy limit) — bounding how far any single step
    can move a point, independent of the regularizer's own magnitude.

    Parameters
    ----------
    params : iterable
        Parameters to optimize, or parameter-group dicts.
    lr : float
        Base learning rate.
    betas : tuple of float
        Adam's ``(beta1, beta2)`` moment-decay coefficients.
    eps : float
        Adam's numerical stability epsilon.
    regularizer : {'sobolev', 'gaussian', 'dsti', 'dsti1', 'none'}
        Which spatial regularizer to apply to the step direction; ignored
        if ``regularizer_fn`` is given.
    regularizer_fn : callable, optional
        Custom ``step -> regularized_step`` callable, taking precedence over
        ``regularizer``.
    sobolev_alpha : float
        Sobolev/DST-I kernel parameter (see
        :func:`antstorch.syn.core.smoothing.apply_sobolev_green_operator`).
    gaussian_sigma : float
        Gaussian kernel sigma, used when ``regularizer='gaussian'``.
    max_step_norm : float
        Maximum per-step physical displacement, in voxel-diagonal units;
        set to ``None`` or ``0`` to disable CFL bounding.
    spacing : sequence of float, optional
        Physical voxel spacing, used both by the spatial regularizer and by
        the CFL bound.
    """

    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8,
                 regularizer='sobolev', regularizer_fn=None,
                 sobolev_alpha=0.035, gaussian_sigma=1.5,
                 max_step_norm=0.50, spacing=None):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            regularizer=regularizer, regularizer_fn=regularizer_fn,
            sobolev_alpha=sobolev_alpha, gaussian_sigma=gaussian_sigma,
            max_step_norm=max_step_norm, spacing=spacing
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            reg_mode = group.get('regularizer', 'sobolev')
            reg_fn = group.get('regularizer_fn')
            alpha = group.get('sobolev_alpha', 0.035)
            gauss_sig = group.get('gaussian_sigma', 1.5)
            spacing = group.get('spacing')
            max_step_norm = group.get('max_step_norm', 0.50)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                k = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** k
                bias_corr2 = 1.0 - beta2 ** k

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                raw_step = (exp_avg / bias_corr1) / denom

                if reg_fn is not None:
                    smooth_step = reg_fn(raw_step)
                elif reg_mode == 'gaussian' or (gauss_sig is not None and gauss_sig > 0 and reg_mode != 'sobolev'):
                    from .smoothing import separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = separable_gaussian_filter(s, sigma=gauss_sig, spacing=spacing)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = separable_gaussian_filter(raw_step, sigma=gauss_sig, spacing=spacing)
                    else:
                        smooth_step = raw_step
                elif reg_mode == 'sobolev' and alpha is not None and alpha > 0:
                    from .smoothing import apply_sobolev_green_operator
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = apply_sobolev_green_operator(s, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = apply_sobolev_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                    else:
                        smooth_step = raw_step
                elif reg_mode == 'dsti' and alpha is not None and alpha > 0:
                    from .smoothing import apply_dsti_green_operator
                    smooth_step = apply_dsti_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha)
                elif reg_mode == 'dsti1' and alpha is not None and alpha > 0:
                    from .smoothing import apply_dsti1_green_operator
                    smooth_step = apply_dsti1_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha)
                else:
                    smooth_step = raw_step

                # Enforce the CFL step bound so a single step cannot move any
                # point by more than max_step_norm voxels (discrete trajectory
                # crossover prevention).
                if max_step_norm is not None and max_step_norm > 0:
                    min_sp = min(spacing) if spacing is not None else 1.0
                    step_mag = torch.sqrt(torch.sum(smooth_step ** 2, dim=-1))
                    max_disp = float(step_mag.max().item()) / max(min_sp, 1e-4)
                    effective_step = max_disp * lr
                    if effective_step > max_step_norm:
                        scale = max_step_norm / max(effective_step, 1e-6)
                        smooth_step = smooth_step * scale

                p.sub_(smooth_step, alpha=lr)

        return loss


# Aliases for backwards compatibility and specialized naming.
SobolevAdam = RegAdam
GaussianAdam = RegAdam


def get_cfl_max_norm(velocity: torch.Tensor, spacing: list) -> float:
    """Return the maximum per-voxel displacement (normalized by spacing) in a velocity field.

    Parameters
    ----------
    velocity : torch.Tensor
        Field of shape ``(*spatial, dim)`` or ``(B, *spatial, dim)``.
    spacing : sequence of float
        Physical voxel spacing per axis, matching ``velocity``'s spatial
        axis order.

    Returns
    -------
    float
        The maximum spacing-normalized displacement magnitude, useful for
        applying Courant-Friedrichs-Lewy (CFL) limits to spatial-grid
        deformation updates.
    """
    device = velocity.device
    dim = velocity.shape[-1]
    spacing_t = torch.tensor(spacing, device=device, dtype=torch.float32).view(*([1] * (velocity.ndim - 1)), dim)
    v_norm_voxel = velocity / spacing_t
    max_norm = torch.max(torch.linalg.norm(v_norm_voxel, dim=-1)).item()
    return max_norm


def compute_cfl_step(kwargs: dict, shrink_ratio: float, default_grad_step: float = 0.25) -> float:
    """Compute the effective CFL-constrained gradient step size at a pyramid level.

    Parameters
    ----------
    kwargs : dict
        Options dict; reads ``kwargs['cfl_step']`` (falling back to
        ``kwargs['grad_step']``, then ``default_grad_step``).
    shrink_ratio : float
        The current pyramid level's physical shrink ratio.
    default_grad_step : float
        Fallback step size if neither key is present in ``kwargs``.

    Returns
    -------
    float
        ``cfl_step_val * sqrt(shrink_ratio)``.
    """
    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', default_grad_step)))
    return float(cfl_step_val) * math.sqrt(shrink_ratio)


def check_convergence(losses, window_size: int = 10, slope_threshold: float = 1e-8) -> bool:
    """Check whether an optimization loss has converged over a sliding window.

    Fits a least-squares line to the last ``window_size`` loss values and
    reports convergence once its slope is no longer meaningfully negative.

    Parameters
    ----------
    losses : sequence of float
        Loss history, oldest first.
    window_size : int
        Number of most-recent losses to fit.
    slope_threshold : float
        Convergence is reported once ``slope >= -slope_threshold``.

    Returns
    -------
    bool
        ``True`` if converged. Always ``False`` if fewer than
        ``window_size`` losses are available.
    """
    if len(losses) < window_size:
        return False
    y = np.array(losses[-window_size:])
    x = np.arange(window_size)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-8:
        return False
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return slope >= -slope_threshold
