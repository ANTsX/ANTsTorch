"""Principal component flow (PCF) terms for the LAMNr Glow trainers.

Objective (per training step):

    L = L_like + lambda * L_align(phi(z_S^(v)))
          + alpha * sum_v (-I_hat^(v))                  # --pcf-alpha
          + beta  * L_axis(|g_k^(1)|, |g_k^(2)|)        # --axis-beta

* -I_hat^(v) is the PCF penalty of view v (Cunningham, Cobb and Jha, ICML 2022),
  sum_k log||dz_k/dx|| - log|det dz/dx| >= 0, zero when the rows of the encoder
  Jacobian are orthogonal. It is estimated by
  ``antsnormflows.MultiscaleFlow.pcf_regularizer`` and converted here to bits per
  dimension (divided by n_dims * ln 2) so that it is on the scale of the
  likelihood term.
* L_axis aligns latent axes across views. When the encoder rows are orthogonal,
  the decoder column dx/dz_k is parallel to the encoder row g_k = dz_k/dx, so the
  rows tell where in the image each latent coordinate acts. For latent
  coordinates k drawn at random (the same k in every view), L_axis is
  1 - cos(|g_k^(v)|, |g_k^(w)|), averaged over pairs of views, where |g| is the
  per-voxel magnitude summed over channels. Views must share the latent layout
  and the spatial grid.
"""
import math
from typing import List, Sequence

import torch
import torch.nn as nn


def unwrap_flow(m: nn.Module) -> nn.Module:
    """Return the bare MultiscaleFlow behind DataParallel/DDP/GlowStepWrapper."""
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "model") and hasattr(m.model, "inverse_and_log_det"):
        m = m.model
    return m


def ramp(it: int, warmup: int) -> float:
    """Linear warm-up factor in [0, 1] (1 when warmup <= 0)."""
    if warmup <= 0:
        return 1.0
    return min(1.0, max(0.0, it / float(warmup)))


def pcf_penalty_bpd(model: nn.Module, x: torch.Tensor, num_dims: int, n_dims: int) -> torch.Tensor:
    """Batch-mean PCF penalty of one view, in bits per dimension (differentiable)."""
    flow = unwrap_flow(model)
    if not hasattr(flow, "pcf_regularizer"):
        raise RuntimeError(
            "--pcf-alpha requires antsnormflows with MultiscaleFlow.pcf_regularizer "
            "(ANTsNormalizingFlows core.py); update the installed antsnormflows."
        )
    penalty = flow.pcf_regularizer(x.float(), num_dims=num_dims, create_graph=True)
    return penalty.mean() / (n_dims * math.log(2.0))


def _flat_levels(z) -> List[torch.Tensor]:
    z = z if isinstance(z, (list, tuple)) else [z]
    return [zl.reshape(zl.shape[0], -1) for zl in z]


def axis_alignment_loss(models: Sequence[nn.Module], xs: Sequence[torch.Tensor], num_dims: int) -> torch.Tensor:
    """Cross-view alignment of encoder-row magnitudes for shared random latent coordinates."""
    if len(models) < 2:
        raise ValueError("--axis-beta needs at least two views.")
    spatial = {tuple(x.shape[2:]) for x in xs}
    if len(spatial) != 1:
        raise ValueError(f"--axis-beta needs views on the same spatial grid; got {sorted(spatial)}.")
    xs_req, levels = [], []
    for m, x in zip(models, xs):
        xr = x.detach().float().requires_grad_(True)
        z, _ = unwrap_flow(m).inverse_and_log_det(xr)
        xs_req.append(xr); levels.append(_flat_levels(z))
    sizes = [tuple(zl.shape[1] for zl in lv) for lv in levels]
    if len(set(sizes)) != 1:
        raise ValueError(f"--axis-beta needs the same latent layout in every view; got {sizes}.")
    batch = xs[0].shape[0]; dev = xs[0].device
    rows_idx = torch.arange(batch, device=dev)
    terms = []
    for li, size in enumerate(sizes[0]):
        for _ in range(num_dims):
            idx = torch.randint(size, (batch,), device=dev)
            mags = []
            for v in range(len(models)):
                zf = levels[v][li]
                sel = torch.zeros_like(zf); sel[rows_idx, idx] = 1.0
                g = torch.autograd.grad(zf, xs_req[v], grad_outputs=sel, create_graph=True, retain_graph=True)[0]
                mags.append(g.abs().sum(dim=1).reshape(batch, -1))
            for a in range(len(mags)):
                for b in range(a + 1, len(mags)):
                    cos = torch.nn.functional.cosine_similarity(mags[a], mags[b], dim=1, eps=1e-12)
                    terms.append(1.0 - cos)
    return torch.stack(terms).mean()


def add_pcf_args(ap) -> None:
    """Register the PCF / axis-alignment options shared by the Glow trainers."""
    g = ap.add_argument_group("principal component flow")
    g.add_argument("--pcf-alpha", type=float, default=0.0, metavar="ALPHA",
        help="Weight of the PCF penalty -I_hat (bits/dim), summed over views. 0 (default): off.")
    g.add_argument("--pcf-dims", type=int, default=1, metavar="N",
        help="Latent coordinates sampled per level and per sample for the PCF penalty "
             "(one encoder vector-Jacobian product each). Default 1.")
    g.add_argument("--pcf-warmup", type=int, default=0, metavar="ITERS",
        help="Linear warm-up of --pcf-alpha over ITERS iterations. Default 0.")
    g.add_argument("--axis-beta", type=float, default=0.0, metavar="BETA",
        help="Weight of the cross-view axis alignment 1 - cos(|g_k^(v)|, |g_k^(w)|) "
             "(needs >= 2 views on the same grid). 0 (default): off.")
    g.add_argument("--axis-dims", type=int, default=1, metavar="N",
        help="Shared latent coordinates sampled per level for --axis-beta. Default 1.")
    g.add_argument("--axis-warmup", type=int, default=0, metavar="ITERS",
        help="Linear warm-up of --axis-beta over ITERS iterations. Default 0.")
