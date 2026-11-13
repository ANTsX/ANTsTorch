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


def pcf_penalty_bpd(model: nn.Module, x: torch.Tensor, num_dims: int, n_dims: int,
                    num_levels: int = 0) -> torch.Tensor:
    """Batch-mean PCF penalty of one view, in bits per dimension (differentiable).

    Works for any antsnormflows flow with ``inverse_and_log_det`` (Glow, RealNVP,
    ConvFlow1d) through ``antsnormflows.utils.pcf_penalty``; falls back to
    ``MultiscaleFlow.pcf_regularizer`` with older antsnormflows versions.
    """
    flow = unwrap_flow(model)
    try:
        from antsnormflows.utils import pcf_penalty
    except ImportError:
        pcf_penalty = None
    if pcf_penalty is not None:
        penalty = pcf_penalty(flow, x.float(), num_dims=num_dims, create_graph=True,
                              num_levels=num_levels or None)
    elif hasattr(flow, "pcf_regularizer"):
        penalty = flow.pcf_regularizer(x.float(), num_dims=num_dims, create_graph=True)
    else:
        raise RuntimeError(
            "--pcf-alpha requires antsnormflows with utils.pcf_penalty (or "
            "MultiscaleFlow.pcf_regularizer); update the installed antsnormflows."
        )
    return penalty.mean() / (n_dims * math.log(2.0))


def axis_row_magnitudes(x: torch.Tensor, z, num_dims: int, seed: int) -> torch.Tensor:
    """Per-voxel magnitudes |g_k| of encoder rows for shared random coordinates k.

    x must require grad and be the input z was computed from. The coordinates are
    drawn with a CPU generator seeded by ``seed``, so every view (and every
    DataParallel replica) given the same seed, batch size and latent layout draws
    the same k. Returns a tensor (batch, levels * num_dims, voxels), kept in the
    graph (create_graph=True).
    """
    gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))
    batch = x.shape[0]; rows = torch.arange(batch, device=x.device)
    mags = []
    for zl in _flat_levels(z):
        for _ in range(num_dims):
            idx = torch.randint(zl.shape[1], (batch,), generator=gen).to(x.device)
            sel = torch.zeros_like(zl); sel[rows, idx] = 1.0
            g = torch.autograd.grad(zl, x, grad_outputs=sel, create_graph=True, retain_graph=True)[0]
            mags.append(g.abs().sum(dim=1).reshape(batch, -1))
    return torch.stack(mags, dim=1)


def axis_loss_from_magnitudes(mags: Sequence[torch.Tensor]) -> torch.Tensor:
    """1 - cos between views of the row magnitudes from `axis_row_magnitudes`, averaged."""
    if len(mags) < 2:
        raise ValueError("--axis-beta needs at least two views.")
    shapes = {tuple(m.shape) for m in mags}
    if len(shapes) != 1:
        raise ValueError(f"--axis-beta needs the same grid and latent layout in every view; got {sorted(shapes)}.")
    terms = []
    for a in range(len(mags)):
        for b in range(a + 1, len(mags)):
            terms.append(1.0 - torch.nn.functional.cosine_similarity(mags[a], mags[b], dim=2, eps=1e-12))
    return torch.stack(terms).mean()


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


def grad_norm_decomposition(main_loss: torch.Tensor, reg_loss, params):
    """Gradient norms of the main objective and of the regularizers, and their cosine.

    Uses torch.autograd.grad with retain_graph=True, so the graph stays usable for the
    real backward(); it does not accumulate .grad and (checked with DDP/gloo) does not
    disturb DistributedDataParallel's gradient synchronization. Values are local to the
    process (not averaged over ranks). Returns (norm_main, norm_reg, cos); norm_reg and
    cos are None when reg_loss is None.
    """
    params = [p for p in params if p.requires_grad]
    g_main = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    flat_main = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float()
                           for g, p in zip(g_main, params)])
    n_main = float(flat_main.norm())
    if reg_loss is None or not torch.is_tensor(reg_loss) or not reg_loss.requires_grad:
        return n_main, None, None
    g_reg = torch.autograd.grad(reg_loss, params, retain_graph=True, allow_unused=True)
    flat_reg = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float()
                          for g, p in zip(g_reg, params)])
    n_reg = float(flat_reg.norm())
    cos = float(torch.dot(flat_main, flat_reg) / (flat_main.norm() * flat_reg.norm() + 1e-30))
    return n_main, n_reg, cos


def add_grad_norm_args(ap) -> None:
    """--grad-norm-every: gradient-norm diagnostics (all LAMNr trainers)."""
    g = ap.add_argument_group("gradient diagnostics")
    g.add_argument("--grad-norm-every", type=int, default=0, metavar="N",
        help="Every N iterations, log in metrics.csv the gradient norms of the main objective "
             "(likelihood + alignment) and of the regularizers (PCF, axis) separately, and their "
             "cosine, before clipping (first micro-batch, rank 0; costs two extra backward passes "
             "on those iterations). With lazy PCF choose N a multiple of --pcf-every. 0 (default): "
             "off. The total gradient norm before clipping is always logged.")


def add_pcf_args(ap, axis: bool = True) -> None:
    """Register the PCF options, and the axis-alignment options unless axis=False
    (views without a common spatial grid, e.g. tabular). Also registers --grad-norm-every."""
    add_grad_norm_args(ap)
    g = ap.add_argument_group("principal component flow")
    g.add_argument("--pcf-alpha", type=float, default=0.0, metavar="ALPHA",
        help="Weight of the PCF penalty -I_hat (bits/dim), summed over views. 0 (default): off.")
    g.add_argument("--pcf-dims", type=int, default=1, metavar="N",
        help="Latent coordinates sampled per level and per sample for the PCF penalty "
             "(one encoder vector-Jacobian product each); >= the latent size gives the "
             "exact penalty (cheap for tabular views). Default 1.")
    g.add_argument("--pcf-warmup", type=int, default=0, metavar="ITERS",
        help="Linear warm-up of --pcf-alpha over ITERS iterations. Default 0.")
    g.add_argument("--pcf-every", type=int, default=1, metavar="N",
        help="Lazy regularization: compute the PCF penalty every N iterations only, "
             "with its weight multiplied by N (same expected gradient). Default 1.")
    g.add_argument("--pcf-samples", type=int, default=0, metavar="M",
        help="Compute the PCF penalty on the first M samples of each micro-batch (per "
             "process/replica), in a separate float32 pass, while the likelihood keeps the "
             "full batch and its precision. 0 (default): the whole micro-batch, sharing the "
             "likelihood's encoder pass (which then runs in float32).")
    g.add_argument("--pcf-levels-per-step", type=int, default=0, metavar="K",
        help="Multiscale flows: draw K latent levels at random for the PCF penalty at each "
             "step (unbiased reweighting), i.e. K vector-Jacobian products instead of one per "
             "level. 0 (default): every level.")
    if not axis:
        return
    g.add_argument("--axis-beta", type=float, default=0.0, metavar="BETA",
        help="Weight of the cross-view axis alignment 1 - cos(|g_k^(v)|, |g_k^(w)|) "
             "(needs >= 2 views on the same grid). 0 (default): off.")
    g.add_argument("--axis-dims", type=int, default=1, metavar="N",
        help="Shared latent coordinates sampled per level for --axis-beta. Default 1.")
    g.add_argument("--axis-warmup", type=int, default=0, metavar="ITERS",
        help="Linear warm-up of --axis-beta over ITERS iterations. Default 0.")
