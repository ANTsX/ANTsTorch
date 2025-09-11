
import torch
import torch.nn as nn
import normflows as nf


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
