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

from typing import Callable, Dict, Optional, Sequence, Tuple

from .regularizers import (
    COUPLED_CONTINUUM_REGULARIZERS,
    SPECTRAL_REGULARIZERS,
    _require_spacing,
    get_regularizer,
)


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


_REGADAM_REGULARIZERS = SPECTRAL_REGULARIZERS | frozenset({
    'gaussian', 'gauss', 'none', 'identity',
    'compact_gaussian', 'compact', 'fast_gaussian', 'erf', 'bspline',
})


class RegAdam(torch.optim.Optimizer):
    """
    Adam whose step is spatially smoothed and then bounded (CFL-style), for dense velocity
    fields.

    Per parameter tensor: standard Adam moments with bias correction give the point-wise
    direction d = m_hat / (sqrt(v_hat) + floor); d is smoothed by the selected regulariser,
    scaled down if lr * max_x |d(x) / spacing| > ``max_step_norm``, and applied as
    p <- p - lr * d.

    Parameters are expected channel-last, (B, *spatial, dim) (4-D / 5-D tensors) or
    (B, 1, *spatial, dim) (squeezed for smoothing); other shapes are not smoothed.
    """
    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8,
                 regularizer='sobolev', regularizer_fn=None,
                 sobolev_alpha=None, dsti_alpha=None, gaussian_sigma=1.5,
                 max_step_norm=0.50, spacing=None, eps_rel=0.0,
                 origin=None, direction=None, extra_reg_kwargs=None):
        """
        Parameters
        ----------
        params : iterable
            Parameters or parameter-group dicts.
        lr : float, default 0.80
        betas : (float, float), default (0.9, 0.999)
        eps : float, default 1e-8
            Absolute floor added to the Adam denominator.
        regularizer : str, default 'sobolev'
            Smoothing of the Adam direction (ignored if ``regularizer_fn`` is given):
            - 'sobolev': ``apply_sobolev_green_operator`` (FFT, periodic) with alpha =
              ``sobolev_alpha`` (legacy fallback 0.035 when None) and ``spacing``.
            - 'dsti' / 'dsti1' (the same operator): ``apply_dsti_green_operator`` (zero
              Dirichlet) with alpha = ``dsti_alpha`` and ``spacing``.
            - 'gaussian' / 'gauss' / 'bspline': ``separable_gaussian_filter`` (Bessel, replicate
              padding), sigma = ``gaussian_sigma`` voxels (none if <= 0; 'bspline' models smooth
              the field themselves, the step is Gaussian-smoothed).
            - 'compact_gaussian' / 'compact' / 'fast_gaussian' / 'erf':
              ``fast_separable_gaussian_filter`` (zero padding), sigma = ``gaussian_sigma``.
            - continuum operators: forwarded to ``get_regularizer`` with ``spacing``,
              ``origin``, ``direction``, and ``extra_reg_kwargs``.
            - 'none': no smoothing.
            Other strings raise ValueError.
        regularizer_fn : callable, optional
            d -> smoothed d, used instead of ``regularizer``.
        sobolev_alpha : float, optional
            alpha of the spectral regularisers (None lets continuum operators derive their
            own default; legacy 'sobolev' falls back to 0.035).
        dsti_alpha : float, optional
            alpha of 'dsti' / 'dsti1' (default ``sobolev_alpha``).
        gaussian_sigma : float, default 1.5
            Gaussian sigma in voxels (``spacing`` has no effect on it).
        max_step_norm : float, default 0.50
            Bound on lr * max |d / spacing| (voxels when spacing is in mm); None or <= 0
            disables it. The whole tensor is rescaled, not clipped voxel-wise.
        spacing : sequence, optional
            ANTs (x, y, z) order; used by spectral/continuum operators and the per-axis
            anisotropic step bound (None = unit spacing).
        eps_rel : float
            Relative floor of the Adam denominator: ``max(eps, eps_rel * max(sqrt(v_hat)))``.
        origin, direction : optional
            ITK physical origin and direction cosine matrix forwarded to continuum
            operators.
        extra_reg_kwargs : dict, optional
            Extra keyword arguments forwarded to ``get_regularizer``.
        """
        if regularizer_fn is None and str(regularizer).lower() not in _REGADAM_REGULARIZERS:
            raise ValueError(f"RegAdam: unknown regularizer {regularizer!r}; expected one of "
                             f"{sorted(_REGADAM_REGULARIZERS)} or a regularizer_fn")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, eps_rel=eps_rel,
            regularizer=regularizer, regularizer_fn=regularizer_fn,
            sobolev_alpha=sobolev_alpha,
            dsti_alpha=dsti_alpha if dsti_alpha is not None else sobolev_alpha,
            gaussian_sigma=gaussian_sigma,
            max_step_norm=max_step_norm, spacing=spacing,
            origin=origin, direction=direction,
            extra_reg_kwargs=dict(extra_reg_kwargs) if extra_reg_kwargs is not None else {},
        )
        super(RegAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """One RegAdam update of every parameter with a gradient (in place). Returns
        ``closure()`` (evaluated with grad enabled) or None. The step bound is computed on the
        device; only ``eps_rel`` > 0 synchronises (``float(denom.max())``)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            eps_rel = group.get('eps_rel', 0.0)
            reg_mode = str(group.get('regularizer', 'sobolev')).lower()
            reg_fn = group.get('regularizer_fn')
            alpha = group.get('sobolev_alpha', None)
            dsti_alpha = group.get('dsti_alpha', alpha)
            gauss_sig = group.get('gaussian_sigma', 1.5)
            spacing = group.get('spacing')
            origin = group.get('origin')
            direction = group.get('direction')
            extra_reg_kwargs = group.get('extra_reg_kwargs') or {}
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

                # Standard Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** k
                bias_corr2 = 1.0 - beta2 ** k

                # Raw point-wise step direction quotient
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_corr2)
                floor = max(eps, eps_rel * float(denom.max())) if eps_rel > 0 else eps
                denom = denom.add_(floor)
                raw_step = (exp_avg / bias_corr1) / denom

                # Apply elected regularization directly to the Adam step direction
                if reg_fn is not None:
                    smooth_step = reg_fn(raw_step)
                elif reg_mode in ('compact_gaussian', 'compact', 'fast_gaussian', 'erf'):
                    from .smoothing import fast_separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = fast_separable_gaussian_filter(s, sigma=gauss_sig)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = fast_separable_gaussian_filter(raw_step, sigma=gauss_sig)
                    else:
                        smooth_step = raw_step
                elif reg_mode in ('gaussian', 'gauss', 'bspline') and gauss_sig is not None and gauss_sig > 0:
                    from .smoothing import separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = separable_gaussian_filter(s, sigma=gauss_sig)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = separable_gaussian_filter(raw_step, sigma=gauss_sig)
                    else:
                        smooth_step = raw_step
                elif reg_mode == 'sobolev':
                    alpha_sob = 0.035 if alpha is None else float(alpha)
                    if alpha_sob > 0:
                        from .smoothing import apply_sobolev_green_operator
                        if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                            s = raw_step.squeeze(1)
                            smooth_s = apply_sobolev_green_operator(
                                s, fluid_sigma=alpha_sob, alpha=alpha_sob, spacing=spacing
                            )
                            smooth_step = smooth_s.unsqueeze(1)
                        elif raw_step.ndim in (4, 5):
                            smooth_step = apply_sobolev_green_operator(
                                raw_step, fluid_sigma=alpha_sob, alpha=alpha_sob, spacing=spacing
                            )
                        else:
                            smooth_step = raw_step
                    else:
                        smooth_step = raw_step
                elif reg_mode in ('dsti', 'dsti1', 'dst_i', 'dst1', 'dirichlet'):
                    d_alpha = 0.035 if dsti_alpha is None else float(dsti_alpha)
                    if d_alpha > 0:
                        # 'dsti' and 'dsti1' are the same operator (apply_dsti1_green_operator is an alias)
                        from .smoothing import apply_dsti_green_operator
                        if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                            s = raw_step.squeeze(1)
                            smooth_step = apply_dsti_green_operator(
                                s, fluid_sigma=d_alpha, alpha=d_alpha, spacing=spacing
                            ).unsqueeze(1)
                        elif raw_step.ndim in (4, 5):
                            smooth_step = apply_dsti_green_operator(
                                raw_step, fluid_sigma=d_alpha, alpha=d_alpha, spacing=spacing
                            )
                        else:
                            smooth_step = raw_step
                    else:
                        smooth_step = raw_step
                elif reg_mode not in ('none', 'identity'):
                    if alpha is not None and float(alpha) <= 0.0:
                        smooth_step = raw_step
                    else:
                        reg_builder = get_regularizer(
                            reg_mode,
                            alpha=alpha,
                            sobolev_alpha=alpha,
                            fluid_sigma=gauss_sig if gauss_sig is not None else 3.0,
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                            **extra_reg_kwargs,
                        )
                        smooth_step = reg_builder(raw_step)
                else:
                    smooth_step = raw_step

                # Enforce anisotropic per-axis CFL step bound on device (no host sync)
                if max_step_norm is not None and max_step_norm > 0:
                    if spacing is not None:
                        from .inverse import _scaled_sq_sum
                        sp = _require_spacing(spacing, smooth_step.shape[-1], "RegAdam")
                        sp_zyx_t = torch.tensor(
                            tuple(reversed(sp)),
                            device=smooth_step.device,
                            dtype=smooth_step.dtype,
                        )
                        effective_step = _scaled_sq_sum(smooth_step, sp_zyx_t, divide=True).amax().sqrt() * lr
                    else:
                        effective_step = torch.sqrt(torch.sum(smooth_step ** 2, dim=-1)).amax() * lr
                    scale = torch.clamp(max_step_norm / effective_step.clamp_min(1e-6), max=1.0)
                    smooth_step = smooth_step * scale

                p.sub_(smooth_step, alpha=lr)

        return loss


# Aliases for backwards compatibility and specialized naming
SobolevAdam = RegAdam
GaussianAdam = RegAdam


_PLAIN_COMPONENTWISE_OPTIMIZERS = frozenset({'adam', 'adamw', 'rmsprop', 'rprop'})


def validate_regularizer_optimizer_pair(
    regularizer: str,
    optimizer: Optional[str],
    caller: str = "syntx",
) -> str:
    """Resolve default optimizer and validate compatibility with ``regularizer``.

    When ``optimizer is None``, automatically resolves to ``'cfl'`` for
    ``SPECTRAL_REGULARIZERS`` and ``'adam'`` for spatial/identity regularizers.
    When an explicit plain componentwise optimizer (``'adam'``, ``'adamw'``,
    ``'rmsprop'``, ``'rprop'``) is combined with a coupled continuum PDE regularizer
    (``COUPLED_CONTINUUM_REGULARIZERS``), raises ``ValueError`` because componentwise
    variance/sign scaling breaks ``div(v)`` / ``curl(v)`` operator coupling.
    """
    reg = str(regularizer).lower()
    if optimizer is None:
        return 'cfl' if reg in SPECTRAL_REGULARIZERS else 'adam'
    opt = str(optimizer).lower()
    if reg in COUPLED_CONTINUUM_REGULARIZERS and opt in _PLAIN_COMPONENTWISE_OPTIMIZERS:
        raise ValueError(
            f"{caller}: optimizer={optimizer!r} is incompatible with coupled continuum "
            f"regularizer={regularizer!r} because componentwise variance/sign division "
            f"destroys div(v)/curl(v) PDE coupling. Use optimizer='cfl' or optimizer='reg_adam'."
        )
    return opt


class FieldStepOptimizer:
    """Stateful, allocation-free, zero-host-sync field update engine for dense registration.

    Supports ``mode in ('cfl', 'adam', 'reg_adam')`` across both
    ``param_domain='normalized_grid'`` (``greedy``, ``scattered``) and
    ``param_domain='physical_mm'`` (``syn``, ``tvf``, ``syngs``).
    """

    def __init__(
        self,
        mode: str = 'cfl',
        param_domain: str = 'normalized_grid',
        *,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        cfl_momentum: float = 0.85,
        sobolev_alpha: Optional[float] = None,
    ):
        m = str(mode).lower()
        if m == 'regadam':
            m = 'reg_adam'
        if m not in ('cfl', 'adam', 'reg_adam'):
            raise ValueError(
                f"FieldStepOptimizer: mode must be 'cfl', 'adam', or 'reg_adam', got {mode!r}"
            )
        dom = str(param_domain).lower()
        if dom not in ('normalized_grid', 'physical_mm'):
            raise ValueError(
                f"FieldStepOptimizer: param_domain must be 'normalized_grid' or 'physical_mm', "
                f"got {param_domain!r}"
            )
        self.mode = m
        self.param_domain = dom
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.cfl_momentum = float(cfl_momentum)
        self.sobolev_alpha = None if sobolev_alpha is None else float(sobolev_alpha)

        self.vel_mom: Optional[torch.Tensor] = None
        self.exp_avg: Optional[torch.Tensor] = None
        self.exp_avg_sq: Optional[torch.Tensor] = None
        self._step: int = 0
        self._geom_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[float, ...], str, str],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
        ] = {}

    def init_state(self, template: torch.Tensor) -> None:
        """Allocate optimizer state matching ``template`` (one tensor for CFL, two for Adam)."""
        if self.mode == 'cfl':
            self.vel_mom = torch.zeros_like(template)
            self.exp_avg = None
            self.exp_avg_sq = None
        else:
            self.vel_mom = None
            self.exp_avg = torch.zeros_like(template)
            self.exp_avg_sq = torch.zeros_like(template)
        self._step = 0

    def on_level_change(self, new_shape: Sequence[int], *, interp_mode: str = 'trilinear') -> None:
        """Update optimizer buffers for a new pyramid level shape."""
        target = list(new_shape)
        if self.mode == 'cfl':
            if self.vel_mom is None or list(self.vel_mom.shape[1:-1]) != target:
                if self.vel_mom is not None:
                    dim = self.vel_mom.shape[-1]
                    self.vel_mom = torch.zeros(
                        (1, *target, dim),
                        dtype=self.vel_mom.dtype,
                        device=self.vel_mom.device,
                    )
        else:
            from .grid import resize_field
            if self.exp_avg is not None and list(self.exp_avg.shape[1:-1]) != target:
                self.exp_avg = resize_field(self.exp_avg, target, mode=interp_mode)
            if self.exp_avg_sq is not None and list(self.exp_avg_sq.shape[1:-1]) != target:
                self.exp_avg_sq = resize_field(self.exp_avg_sq, target, mode=interp_mode)
            self._step = 0

    def _get_level_geom(
        self,
        grid_shape: Sequence[int],
        spacing: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        shape_t = tuple(int(n) for n in grid_shape)
        dim = len(shape_t)
        sp_xyz = _require_spacing(spacing, dim, "FieldStepOptimizer")
        key = (shape_t, sp_xyz, str(device), str(dtype))
        cached = self._geom_cache.get(key)
        if cached is not None:
            return cached
        size_xyz = tuple(reversed(shape_t))
        sp_t_xyz = torch.tensor(sp_xyz, device=device, dtype=dtype).view(1, *([1] * dim), dim)
        sp_t_zyx = torch.tensor(tuple(reversed(sp_xyz)), device=device, dtype=dtype).view(
            1, *([1] * dim), dim
        )
        half_res_xyz = torch.tensor(
            [0.0 if n <= 1 else 1.0 / (n - 1) for n in size_xyz],
            device=device,
            dtype=dtype,
        ).view(1, *([1] * dim), dim)
        phys_ext_xyz = [max(1, n - 1) * float(s) for n, s in zip(size_xyz, sp_xyz)]
        min_ext = min(phys_ext_xyz)
        chain_scale_xyz = torch.tensor(
            [min_ext / e for e in phys_ext_xyz],
            device=device,
            dtype=dtype,
        ).view(1, *([1] * dim), dim)
        half_resolution = 1.0 / max(1, max(shape_t) - 1)
        out = (sp_t_xyz, sp_t_zyx, half_res_xyz, chain_scale_xyz, half_resolution)
        self._geom_cache[key] = out
        return out

    @torch.no_grad()
    def step(
        self,
        grad: torch.Tensor,
        learning_rate: float,
        *,
        grid_shape: Sequence[int],
        spacing: Sequence[float],
        reg_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        post_reg_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        grad_smooth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reg_smooth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sobolev_fallback_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute one displacement update ``delta`` (to be added/composed to the field)."""
        if (self.mode == 'cfl' and self.vel_mom is None) or (
            self.mode != 'cfl' and self.exp_avg is None
        ):
            self.init_state(grad)

        dim = grad.shape[-1]
        sp_t_xyz, sp_t_zyx, half_res_xyz, chain_scale_xyz, half_resolution = self._get_level_geom(
            grid_shape, spacing, grad.device, grad.dtype
        )

        if self.mode == 'cfl':
            if self.param_domain == 'normalized_grid':
                if reg_fn is not None:
                    v_vox = reg_fn(grad)
                else:
                    f_mm = grad * chain_scale_xyz
                    v_mm = grad_smooth_fn(f_mm) if grad_smooth_fn is not None else f_mm
                    v_vox = v_mm / sp_t_xyz
            else:
                v_mm = (
                    reg_fn(grad)
                    if reg_fn is not None
                    else (grad_smooth_fn(grad) if grad_smooth_fn is not None else grad)
                )
                v_vox = v_mm / sp_t_zyx

            if self.cfl_momentum > 0.0:
                self.vel_mom.mul_(self.cfl_momentum).add_(v_vox, alpha=1.0 - self.cfl_momentum)
                m_vox = self.vel_mom
            else:
                m_vox = v_vox

            v_norm_max = m_vox.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
            v_norm_max = v_norm_max.reshape(-1, *([1] * (dim + 1)))
            m_unit = torch.where(
                v_norm_max > 1e-20,
                m_vox / v_norm_max.clamp(min=1e-20),
                torch.zeros_like(m_vox),
            )
            step_scale = half_res_xyz if self.param_domain == 'normalized_grid' else sp_t_zyx
            return -float(learning_rate) * step_scale * m_unit

        # 'adam' / 'reg_adam'
        g = grad
        if self.mode == 'reg_adam' and (reg_fn is not None or post_reg_fn is not None):
            # Continuum/spectral RegAdam: feed raw gradient into Adam moments, filter quotient ONCE
            pass
        elif reg_fn is not None:
            g = reg_fn(g)
        elif grad_smooth_fn is not None:
            g = grad_smooth_fn(g)

        self._step += 1
        self.exp_avg.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

        b1_corr = 1.0 - self.beta1 ** self._step
        b2_corr = 1.0 - self.beta2 ** self._step
        b2_sqrt = math.sqrt(b2_corr)
        step_size = b2_sqrt / b1_corr
        eps_scaled = self.eps * b2_sqrt
        raw_update = (self.exp_avg * step_size).div_(self.exp_avg_sq.sqrt().add_(eps_scaled))

        if self.mode == 'reg_adam':
            active_post_fn = post_reg_fn if post_reg_fn is not None else reg_fn
            if active_post_fn is not None:
                u_reg = active_post_fn(raw_update)
                max_norm = u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                max_norm = max_norm.reshape(-1, *([1] * (dim + 1)))
                step_scale = half_res_xyz if self.param_domain == 'normalized_grid' else sp_t_zyx
                return torch.where(
                    max_norm > 1e-20,
                    -float(learning_rate) * step_scale * (u_reg / max_norm.clamp(min=1e-20)),
                    torch.zeros_like(u_reg),
                )
            if reg_smooth_fn is not None:
                u_reg = reg_smooth_fn(raw_update)
            else:
                ra_alpha = 0.035 if self.sobolev_alpha is None else float(self.sobolev_alpha)
                if ra_alpha > 0 and sobolev_fallback_fn is not None:
                    u_reg = sobolev_fallback_fn(raw_update, ra_alpha)
                else:
                    u_reg = raw_update
        else:
            u_reg = raw_update

        gradmax = (
            self.eps + u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
        ).reshape(-1, *([1] * (dim + 1))).clamp(min=1.0)
        step_scale = half_resolution if self.param_domain == 'normalized_grid' else 1.0
        return -float(learning_rate) * step_scale * (u_reg / gradmax)


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
