
import torch
import torch.nn as nn
import normflows as nf
from normflows import distributions as nfd

class _BoundedMLP(nn.Module):
    """
    MLP with tanh-bounded output for the log-scale head.
    - Keeps last Linear zero-initialized (identity start).
    - Optional spectral norm on hidden (pre-final) Linear layers.
    """
    def __init__(self, layers, leaky: float, scale_cap: float = 3.0,
                 init_zeros: bool = True, spectral_norm: bool = False):
        super().__init__()
        mlp = nf.nets.MLP(layers, leaky=leaky, init_zeros=init_zeros)
        if spectral_norm:
            linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
            for i, lin in enumerate(linears):
                if i < len(linears) - 1:  # apply to hidden layers only
                    nn.utils.parametrizations.spectral_norm(lin)
        self.mlp = mlp
        self.scale_cap = float(scale_cap)

    def forward(self, x):
        raw = self.mlp(x)
        return self.scale_cap * torch.tanh(raw)


def _make_masks(latent_size: int, mode: str = "alternating", rolls: int = 0):
    """
    Build binary masks (b, 1-b). Optionally roll the mask by 'rolls' to change which dims are active.
    """
    b = torch.tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)], dtype=torch.float32)
    if mode == "rolling" and rolls:
        b = torch.roll(b, shifts=rolls, dims=0)
    return b, (1.0 - b)


def create_real_nvp_normalizing_flow_model(
    latent_size: int,
    K: int = 64,
    q0=None,
    leaky_relu_negative_slope: float = 0.0,
    *,
    scale_cap: float = 3.0,
    spectral_norm_scales: bool = False,
    mask_mode: str = "alternating",          # "alternating" (default) or "rolling"
    additive_first_n: int = 0,               # use additive couplings (no scale) for first N blocks
    actnorm_every: int = 1,                  # insert ActNorm after every 'actnorm_every' couplings
):
    """
    Create a RealNVP model with bounded log-scales for numerical stability.

    Parameters
    ----------
    latent_size : int
        Input dimensionality for this view.
    K : int, default=64
        Number of coupling layers (each layer = MaskedAffineFlow; ActNorm inserted per 'actnorm_every').
    q0 : normflows distribution or None
        Base distribution (e.g., nf.distributions.DiagGaussian(latent_size) or GaussianPCA).
    leaky_relu_negative_slope : float, default=0.0
        Negative slope for LeakyReLU activations in the coupling MLPs.
    scale_cap : float, default=3.0
        Bound on the log-scale output ŝ = scale_cap * tanh(raw).
    spectral_norm_scales : bool, default=False
        Apply spectral normalization to hidden Linear layers in the scale head MLP.
    mask_mode : str, default="alternating"
        If "rolling", progressively roll the binary mask to spread scaling across coordinates.
    additive_first_n : int, default=0
        Use additive couplings (scale head disabled) for the first N layers to avoid early exponentials.
    actnorm_every : int, default=1
        Insert ActNorm after every 'actnorm_every' coupling layers (1 = after each).

    Returns
    -------
    normflows.NormalizingFlow
    """

    flows = []
    b0, b1 = _make_masks(latent_size, mode=mask_mode, rolls=0)

    for i in range(K):
        # Potentially roll masks to avoid repeatedly scaling the same coordinates
        if mask_mode == "rolling":
            b, b_alt = _make_masks(latent_size, mode="rolling", rolls=i % latent_size)
        else:
            b, b_alt = (b0, b1) if (i % 2 == 0) else (b1, b0)

        # Shift head (t): zero-initialized last layer → identity at start
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size],
                        leaky=leaky_relu_negative_slope,
                        init_zeros=True)

        # Scale head (s): bounded via tanh; or disabled for additive layers
        if i < additive_first_n:
            s = None  # MaskedAffineFlow interprets None as "no scale"
        else:
            s = _BoundedMLP([latent_size, 2 * latent_size, latent_size],
                            leaky=leaky_relu_negative_slope,
                            scale_cap=scale_cap,
                            init_zeros=True,
                            spectral_norm=spectral_norm_scales)

        # Affine coupling with mask
        flows.append(nf.flows.MaskedAffineFlow(b, t, s))

        # Optionally add ActNorm
        if (actnorm_every > 0) and ((i + 1) % actnorm_every == 0):
            flows.append(nf.flows.ActNorm(latent_size))

    model = nf.NormalizingFlow(q0=q0, flows=flows)
    return model


import math
from typing import Sequence, Tuple, Literal, Optional


def _check_power_of_two_divisibility(spatial: Sequence[int], L: int, dims: int) -> None:
    """Ensure each spatial dimension is divisible by 2**L (for L squeezes)."""
    if len(spatial) != dims:
        raise ValueError(f"Expected {dims} spatial dims, got {len(spatial)}: {spatial}")
    req = 2 ** L
    bad = [s for s in spatial if (s % req) != 0]
    if bad:
        raise ValueError(
            f"Each spatial dim must be divisible by 2**L={req}. Got {spatial} with L={L}."
        )

def create_glow_normalizing_flow_model_2d(
    input_shape: Tuple[int, int, int],               # (C, H, W)
    *,
    L: int,
    K: int,
    hidden_channels: int,
    base: Literal["glow", "diag"] = "glow",
    glowbase_logscale_factor: float = 3.0,
    glowbase_min_log: float = -5.0,
    glowbase_max_log: float = 5.0,
    split_mode: Literal["channel"] = "channel",
    scale: bool = True,
    scale_map: Optional[str] = "tanh",                 # forwarded to GlowBlock2d
    leaky: float = 0.0,
    net_actnorm: bool = True,
    scale_cap: float = 3.0                           # s_cap in GlowBlock
):
    """
    Create a canonical 2-D Glow normalizing flow model with multiscale factorization.

    This implementation follows the standard Glow architecture (Kingma & Dhariwal, 2018),
    where each level performs a spatial squeeze followed by a sequence of invertible
    transformations (GlowBlocks), and optionally splits off part of the activations into
    a latent variable modeled by a prior distribution.

    Structure per level i = 0..L-1:
        Squeeze2d → [GlowBlock2d] × K → (Split latent_i)

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape (C, H, W).
    L : int
        Number of multiscale levels (squeeze + flow block groups).
    K : int
        Number of Glow blocks (Invertible 1×1 conv + coupling) per level.
    hidden_channels : int
        Number of hidden channels in the internal convolutional subnetworks.
    base : {"glow", "diag"}, default="glow"
        Choice of prior for each latent chunk:
          * "glow" → bounded-channel GlowBase prior (per-channel mean/log-scale).
          * "diag" → diagonal Gaussian prior (DiagGaussian).
    glowbase_logscale_factor : float, default=3.0
        Multiplicative factor applied to the raw log-scale output in GlowBase.
    glowbase_min_log, glowbase_max_log : float, default=(-5.0, 5.0)
        Lower and upper bounds on the channel-wise log-scale in GlowBase.
    split_mode : {"channel"}, default="channel"
        How to split activations between flow and latent branches. Currently "channel" only.
    scale : bool, default=True
        Enable multiplicative scaling in coupling layers (if False, additive-only).
    scale_map : str or None, optional
        Name of the nonlinearity applied to the scale head output (passed to GlowBlock2d).
    leaky : float, default=0.0
        Negative slope for LeakyReLU activations in the coupling subnetworks.
    net_actnorm : bool, default=True
        Apply ActNorm after each coupling block for per-channel normalization.
    scale_cap : float, default=3.0
        Bound on the log-scale parameter ŝ = scale_cap * tanh(raw) within each coupling.

    Returns
    -------
    nf.MultiscaleFlow
        A canonical 2-D Glow model composed of L multiscale levels, each with K GlowBlocks
        and its own latent prior. The resulting flow supports sampling, density estimation,
        and latent-space inference through nf.MultiscaleFlow interfaces.

    Notes
    -----
    • Each spatial dimension must be divisible by 2**L to support repeated squeezes.
    • Each Squeeze2d operation increases the channel dimension by a factor of 4 and
      halves both spatial dimensions.
    """
    
    C, H, W = input_shape
    _check_power_of_two_divisibility((H, W), L=L, dims=2)

    q0 = []
    flows = []
    merges = []

    for i in range(L):
        ch_in = C * (4 ** (i + 1))   # blocks see post-squeeze channels (inverse hits squeeze first)

        level_flows = [
            nf.flows.GlowBlock2d(
                ch_in,
                hidden_channels,
                split_mode=split_mode,
                scale=scale,
                scale_map="tanh",
                leaky=leaky,
                net_actnorm=net_actnorm,
                s_cap=scale_cap,
            )
            for _ in range(K)
        ]
        level_flows.append(nf.flows.Squeeze2d())   # squeeze at END of level (forward)
        flows.append(level_flows)

        lat_shape = (
            ch_in // 2,                              # split sends half to the prior
            H // (2 ** (i + 1)),
            W // (2 ** (i + 1)),
        )
        q0.append(
            nfd.GlowBase(lat_shape, logscale_factor=glowbase_logscale_factor,
                        min_log=glowbase_min_log, max_log=glowbase_max_log)
            if base == "glow" else
            nfd.DiagGaussian(lat_shape)
        )

        if i > 0:
            merges.append(nf.flows.Merge())
            
    return nf.MultiscaleFlow(q0, flows, merges)


def create_glow_normalizing_flow_model_3d(
    input_shape: Tuple[int, int, int, int],          # (C, D, H, W)
    *,
    L: int,
    K: int,
    hidden_channels: int,
    base: Literal["glow", "diag"] = "glow",
    glowbase_logscale_factor: float = 3.0,
    glowbase_min_log: float = -5.0,
    glowbase_max_log: float = 5.0,
    split_mode: Literal["channel"] = "channel",
    scale: bool = True,
    scale_map: Optional[str] = "tanh",                 # forwarded to GlowBlock3d
    leaky: float = 0.0,
    net_actnorm: bool = True,
    scale_cap: float = 3.0                           # s_cap in GlowBlock
):
    """
    Create a canonical 3-D Glow normalizing flow model with multiscale factorization.

    This implementation extends the original Glow architecture to 3-D volumetric data,
    following the same multiscale design used in the 2-D case. Each level begins with
    a spatial squeeze (2×2×2 grouping of voxels into channels) followed by a stack of
    invertible GlowBlocks, and optionally splits off a latent tensor modeled by a prior.

    Structure per level i = 0..L-1:
        Squeeze3d → [GlowBlock3d] × K → (Split latent_i)

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape (C, D, H, W).
    L : int
        Number of multiscale levels (squeeze + flow block groups).
    K : int
        Number of Glow blocks per level.
    hidden_channels : int
        Number of hidden channels in the internal convolutional subnetworks.
    base : {"glow", "diag"}, default="glow"
        Choice of prior for each latent chunk:
          * "glow" → bounded-channel GlowBase prior (per-channel mean/log-scale).
          * "diag" → diagonal Gaussian prior (DiagGaussian).
    glowbase_logscale_factor : float, default=3.0
        Multiplicative factor applied to the raw log-scale output in GlowBase.
    glowbase_min_log, glowbase_max_log : float, default=(-5.0, 5.0)
        Lower and upper bounds on the channel-wise log-scale in GlowBase.
    split_mode : {"channel"}, default="channel"
        How to split activations between flow and latent branches. Currently "channel" only.
    scale : bool, default=True
        Enable multiplicative scaling in coupling layers (if False, additive-only).
    scale_map : str or None, optional
        Name of the nonlinearity applied to the scale head output (passed to GlowBlock3d).
    leaky : float, default=0.0
        Negative slope for LeakyReLU activations in the coupling subnetworks.
    net_actnorm : bool, default=True
        Apply ActNorm after each coupling block for per-channel normalization.
    scale_cap : float, default=3.0
        Bound on the log-scale parameter ŝ = scale_cap * tanh(raw) within each coupling.

    Returns
    -------
    nf.MultiscaleFlow
        A canonical 3-D Glow model composed of L multiscale levels, each with K GlowBlocks
        and its own latent prior. The resulting flow supports sampling, density estimation,
        and latent-space inference through nf.MultiscaleFlow interfaces.

    Notes
    -----
    • Each spatial dimension (D, H, W) must be divisible by 2**L.
    • Each Squeeze3d operation increases the channel dimension by a factor of 8 and
      halves each spatial dimension.
    """
    
    C, D, H, W = input_shape
    _check_power_of_two_divisibility((D, H, W), L=L, dims=3)

    q0 = []
    flows = []
    merges = []

    for i in range(L):
        ch_in = C * (8 ** (i + 1))

        level_flows = [
            nf.flows.GlowBlock3d(
                ch_in,
                hidden_channels,
                split_mode=split_mode,
                scale=scale,
                scale_map="tanh",
                leaky=leaky,
                net_actnorm=net_actnorm,
                s_cap=scale_cap,
            )
            for _ in range(K)
        ]
        level_flows.append(nf.flows.Squeeze3d())
        flows.append(level_flows)

        lat_shape = (
            ch_in // 2,
            D // (2 ** (i + 1)),
            H // (2 ** (i + 1)),
            W // (2 ** (i + 1)),
        )
        q0.append(
            nfd.GlowBase(lat_shape, logscale_factor=glowbase_logscale_factor,
                        min_log=glowbase_min_log, max_log=glowbase_max_log)
            if base == "glow" else
            nfd.DiagGaussian(lat_shape)
        )

        if i > 0:
            merges.append(nf.flows.Merge())

    return nf.MultiscaleFlow(q0, flows, merges)
