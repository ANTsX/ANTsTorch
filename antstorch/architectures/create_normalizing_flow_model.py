
import torch
import torch.nn as nn
import normflows as nf
from normflows import distributions as nfd

import copy

from typing import Optional, Sequence, Tuple, Literal
from itertools import zip_longest

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

def _check_power_of_two_divisibility(spatial: Sequence[int], L: int, dims: int) -> None:
    if len(spatial) != dims:
        raise ValueError(f"Expected {dims} spatial dims, got {len(spatial)}: {spatial}")
    req = 2 ** L
    bad = [s for s in spatial if (s % req) != 0]
    if bad:
        raise ValueError(
            f"Each spatial dim must be divisible by 2**L={req}. Got {spatial} with L={L}."
        )

@torch.no_grad()
def _print_model_summary(model, input_shape, batch=2):

    def _collect_bases(model):
        if hasattr(model, "q0"):
            b = model.q0
        elif hasattr(model, "q0s"):
            b = model.q0s
        else:
            return []
        return list(b) if isinstance(b, (list, tuple, nn.ModuleList)) else [b]

    # (Optional: deepcopy + randn probe to avoid ActNorm side-effects)
    x = torch.randn(batch, *tuple(input_shape), dtype=torch.float32)
    z_list, _ = model.inverse_and_log_det(x)
    zs = z_list if isinstance(z_list, (list, tuple)) else [z_list]
    latent_shapes = [tuple(z.shape[1:]) for z in zs]

    bases = _collect_bases(model)
    base_shapes = []
    base_names  = []
    for b in bases:
        # prefer declared shape; else sample one
        shp = getattr(b, "shape", None)
        if shp is None:
            s, _ = (b(1) if callable(b) else b.sample(1))
            shp = tuple(s.shape[1:])
        base_shapes.append(tuple(shp))
        # distribution “name”
        base_names.append(getattr(b, "name", None) or b.__class__.__name__)

    print(f"[model] input_shape={tuple(input_shape)} | levels={len(latent_shapes)}")
    for i, (ls, bs, bn) in enumerate(zip(latent_shapes, base_shapes, base_names)):
        status = "OK" if bs == ls else "MISMATCH"
        print(f"  level {i:02d}: latent={ls}  base={bs}  dist={bn}  -> {status}")


def create_glow_normalizing_flow_model_2d(
    input_shape: Tuple[int, int, int],  # (C, H, W)
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
    scale_map: Optional[str] = "tanh",
    leaky: float = 0.0,
    net_actnorm: bool = True,
    scale_cap: float = 3.0,
    verbose: bool = False,
) -> nf.MultiscaleFlow:
    """
    Create a multiscale 2-D Glow model with bounded log-scales and numerically stable base distributions.

    This builder constructs a classic multiscale Glow (RealNVP-style) pyramid for images, using per-level
    ordering **[GlowBlock2d] × K → Squeeze2d → Split** in forward. Consequently, on inverse the first
    operation at each level is an **Unsqueeze** (which increases channels by ×4) before any coupling blocks
    run. This ordering keeps channel-split couplings well-posed even when the input has a single channel.

    Channel plan (with channel-split):
      Let C be the input channels. For `L` levels indexed `i = 0..L-1` (top is i = L-1):
        • Channels entering level i (forward):      c_in(i) = C * 2**(L-1-i)
        • Blocks actually run on (inverse, post-unsqueeze):  4 * c_in(i)
        • Latent peeled at level i (after squeeze & split):  (2 * c_in(i), H/2**(i+1), W/2**(i+1))

    Parameters
    ----------
    input_shape : (int, int, int)
        Input tensor shape as (C, H, W). H and W must be divisible by 2**L.
    L : int
        Number of multiscale levels. Each level performs K Glow blocks, then a squeeze and a split.
    K : int
        Number of Glow blocks (ActNorm + 1x1 invertible conv + affine coupling) per level.
    hidden_channels : int
        Width of the conditioner CNN inside each affine coupling at all levels.
    base : {"glow", "diag"}, default="glow"
        Base distribution for each latent split. "glow" uses a GlowBase (bounded log-scale),
        "diag" uses a diagonal Gaussian.
    glowbase_logscale_factor : float, default=3.0
        Scale factor applied to the base distribution’s tanh-bounded log-scales.
    glowbase_min_log : float, default=-5.0
        Minimum (clipped) log-scale for the base distribution.
    glowbase_max_log : float, default=5.0
        Maximum (clipped) log-scale for the base distribution.
    split_mode : {"channel", "checkerboard"}, default="channel"
        Splitting strategy inside couplings. "channel" halves channels along C; "checkerboard" splits by
        spatial mask. With the chosen per-level ordering, "channel" works for C ≥ 1 because Unsqueeze runs first.
    scale : bool, default=True
        Whether to learn scale (s) in the affine coupling. If False, coupling is additive.
    scale_map : {"tanh", None}, default="tanh"
        Nonlinearity to bound the coupling log-scale ŝ (e.g., ŝ = scale_cap * tanh(raw)).
    leaky : float, default=0.0
        Negative slope for any LeakyReLU used inside the conditioner CNN.
    net_actnorm : bool, default=True
        Use ActNorm inside the conditioner CNNs.
    scale_cap : float, default=3.0
        Magnitude cap for the coupling scale head (used with `scale_map="tanh"`).

    Returns
    -------
    nf.MultiscaleFlow
        A normflows MultiscaleFlow where `forward_and_log_det(z_list)` reconstructs x from
        multiscale latents, and `inverse_and_log_det(x)` returns a list of latents (one per level).

    Notes
    -----
    • Per-level forward order is `[GlowBlock2d] × K → Squeeze2d → Split`. Inverse therefore begins with
      Unsqueeze at each level, then runs the K blocks. Each block is constructed with the number of channels
      it actually receives in inverse, i.e., `4 * c_in(i)`.
    • For channel-split couplings, ensure H and W are divisible by 2**L. The top-level `C=1` case is supported
      specifically because Unsqueeze happens before the first split in inverse.
    """

    C, H, W = input_shape
    _check_power_of_two_divisibility((H, W), L=L, dims=2)

    if split_mode not in ("channel", "checkerboard"):
        raise ValueError(f"Unknown split_mode={split_mode!r}")

    q0, flows, merges = [], [], []

    for i in range(L):
        # Channels entering level i (forward) given your MultiscaleFlow indexing:
        # top level (i=L-1) sees C; next sees 2C; etc.
        c_in = C * (2 ** (L - 1 - i))

        # In inverse, blocks run right after an unsqueeze → they see 4 * c_in channels
        block_channels = 4 * c_in

        level_flows = [
            nf.flows.GlowBlock2d(
                block_channels,          # <-- build for what it *actually* sees
                hidden_channels,
                split_mode=split_mode,   # 'channel' is fine now
                scale=scale,
                scale_map=scale_map,
                leaky=leaky,
                net_actnorm=net_actnorm,
                s_cap=scale_cap,
            )
            for _ in range(K)
        ]
        level_flows.append(nf.flows.Squeeze2d())
        flows.append(level_flows)

        lat_ch = (4 * c_in) if i == 0 else (2 * c_in)
        lat_shape = (lat_ch, H // (2 ** (L - i)), W // (2 ** (L - i)))

        q0.append(
            nfd.GlowBase(
                lat_shape,
                logscale_factor=glowbase_logscale_factor,
                min_log=glowbase_min_log,
                max_log=glowbase_max_log,
            )
            if base == "glow"
            else nfd.DiagGaussian(lat_shape)
        )

        if i > 0:
            merges.append(nf.flows.Merge())

    model = nf.MultiscaleFlow(q0, flows, merges)

    if verbose:
        print("Created the following 2D GLOW model:")
        _print_model_summary(model, input_shape=input_shape)

    return model


def create_glow_normalizing_flow_model_3d(
    input_shape: Tuple[int, int, int, int],  # (C, D, H, W)
    *,
    L: int,
    K: int,
    hidden_channels: int,
    base: Literal["glow", "diag"] = "glow",
    glowbase_logscale_factor: float = 3.0,
    glowbase_min_log: float = -5.0,
    glowbase_max_log: float = 5.0,
    split_mode: Literal["channel", "checkerboard"] = "channel",
    scale: bool = True,
    scale_map: Optional[str] = "tanh",
    leaky: float = 0.0,
    net_actnorm: bool = True,
    scale_cap: float = 3.0,
    verbose: bool = False,
) -> nf.MultiscaleFlow:
    """
    Create a multiscale 3-D Glow model with bounded log-scales for numerical stability.

    The per-level forward ordering is **[GlowBlock3d] × K → Squeeze3d → Split** (with Squeeze3d packing
    2×2×2 neighborhoods, i.e., channels×8 and spatial dims halved). Consequently, on inverse each level
    begins with **Unsqueeze3d** before any coupling runs, which keeps channel-split couplings well-posed
    even when the input has a single channel.

    3-D channel plan (channel split):
      For L levels indexed i = 0..L-1 (top is i = L-1), let C be the input channels. Then
        • channels entering level i (forward):         c_in(i) = C * 4**(L-1-i)
        • channels seen by blocks (inverse, post-unsqueeze):  8 * c_in(i)
        • latent peeled at level i (after squeeze & split):   4 * c_in(i)
        • latent spatial size:                         (D/2**(i+1), H/2**(i+1), W/2**(i+1))

    Parameters
    ----------
    input_shape : (int, int, int, int)
        Input tensor shape as (C, D, H, W). D, H, and W must each be divisible by 2**L.
    L : int
        Number of multiscale levels. Each level performs K Glow blocks, then a squeeze and a split.
    K : int
        Number of Glow blocks (ActNorm + invertible 1×1 conv + affine coupling) per level.
    hidden_channels : int
        Width of the conditioner CNN inside each affine coupling at all levels.
    base : {"glow", "diag"}, default="glow"
        Base distribution for each latent split. "glow" uses GlowBase (tanh-bounded log-scale);
        "diag" uses a diagonal Gaussian.
    glowbase_logscale_factor : float, default=3.0
        Scale factor applied to the base distribution’s tanh-bounded log-scales.
    glowbase_min_log : float, default=-5.0
        Minimum (clipped) log-scale for the base distribution.
    glowbase_max_log : float, default=5.0
        Maximum (clipped) log-scale for the base distribution.
    split_mode : {"channel", "checkerboard"}, default="channel"
        Coupling split strategy. "channel" halves channels; "checkerboard" uses a spatial mask. With the
        chosen per-level ordering, inverse starts with Unsqueeze3d, so channel split works for C ≥ 1.
    scale : bool, default=True
        Whether to learn the scale (s) in the affine coupling. If False, coupling is additive.
    scale_map : {"tanh", None}, default="tanh"
        Nonlinearity to bound the coupling log-scale, e.g., ŝ = scale_cap * tanh(raw).
    leaky : float, default=0.0
        Negative slope for any LeakyReLU used in the conditioner CNNs.
    net_actnorm : bool, default=True
        Use ActNorm inside the conditioner CNNs.
    scale_cap : float, default=3.0
        Magnitude cap applied to the coupling scale head (used with `scale_map="tanh"`).

    Returns
    -------
    nf.MultiscaleFlow
        A normflows MultiscaleFlow. `inverse_and_log_det(x)` returns a list of latents (one per level)
        and log-det; `forward_and_log_det(z_list)` reconstructs x and returns the forward log-det.

    Notes
    -----
    • Per-level forward order is `[GlowBlock3d] × K → Squeeze3d → Split`. Inverse therefore begins with
      Unsqueeze3d at each level, then runs the K blocks. Each block is constructed with the number of
      channels it actually receives in inverse, i.e., `8 * c_in(i)`.
    • Ensure D, H, W are multiples of `2**L`.
    """

    C, D, H, W = input_shape
    _check_power_of_two_divisibility((D, H, W), L=L, dims=3)

    if split_mode not in ("channel", "checkerboard"):
        raise ValueError(f"Unknown split_mode={split_mode!r}")

    q0, flows, merges = [], [], []

    for i in range(L):
        # Channels entering level i in forward (top is i=L-1)
        c_in = C * (4 ** (L - 1 - i))

        # Blocks actually run (in inverse) on post-unsqueeze activations
        block_channels = 8 * c_in           # <-- key for 3-D

        if split_mode == "channel" and (block_channels % 2 != 0):
            raise ValueError(f"Channel split needs even channels at level {i}, got {block_channels}.")

        # Build K Glow blocks for this level
        level_flows = [
            nf.flows.GlowBlock3d(
                block_channels,             # <-- must match what it will *see* (not c_in)
                hidden_channels,
                split_mode=split_mode,      # 'channel' works with this ordering
                scale=scale,
                scale_map=scale_map,
                leaky=leaky,
                net_actnorm=net_actnorm,
                s_cap=scale_cap,
            )
            for _ in range(K)
        ]
        level_flows.append(nf.flows.Squeeze3d())
        flows.append(level_flows)

        lat_ch = (8 * c_in) if i == 0 else (4 * c_in)
        lat_shape = (lat_ch, D // (2 ** (L - i)), H // (2 ** (L - i)), W // (2 ** (L - i)))

        q0.append(
            nfd.GlowBase(
                lat_shape,
                logscale_factor=glowbase_logscale_factor,
                min_log=glowbase_min_log,
                max_log=glowbase_max_log,
            )
            if base == "glow" else nfd.DiagGaussian(lat_shape)
        )

        if i > 0:
            merges.append(nf.flows.Merge())

    model = nf.MultiscaleFlow(q0, flows, merges)

    if verbose:
        print("Created the following 3D GLOW model:")
        _print_model_summary(model, input_shape=input_shape)

    return model
