
import torch
import torch.nn as nn
import normflows as nf


class _BoundedMLP(nn.Module):
    def __init__(self, layers, leaky: float, scale_cap: float = 3.0,
                 init_zeros: bool = True, spectral_norm: bool = False):
        super().__init__()
        mlp = nf.nets.MLP(layers, leaky=leaky, init_zeros=init_zeros)
        if spectral_norm:
            # Apply SN to all Linear layers EXCEPT the last one (zero-init for identity)
            linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
            for i, lin in enumerate(linears):
                if i < len(linears) - 1:  # hidden layers only
                    nn.utils.parametrizations.spectral_norm(lin)
        self.mlp = mlp
        self.scale_cap = float(scale_cap)

    def forward(self, x):
        raw = self.mlp(x)
        return self.scale_cap * torch.tanh(raw)


def create_real_nvp_normalizing_flow_model(
    latent_size: int,
    K: int = 64,
    q0=None,
    leaky_relu_negative_slope: float = 0.0,
    *,
    scale_cap: float = 3.0,
    spectral_norm_scales: bool = False,
):
    """
    Create a RealNVP model with bounded log-scales for numerical stability.

    Parameters
    ----------
    latent_size : int
        Input dimensionality for this view.
    K : int, default=64
        Number of coupling blocks (each block = MaskedAffineFlow + ActNorm).
    q0 : normflows distribution or None
        Base distribution (e.g., `nf.distributions.DiagGaussian(latent_size)` or
        `nf.distributions.GaussianPCA(latent_size, latent_dim=...)`).
    leaky_relu_negative_slope : float, default=0.0
        Negative slope for LeakyReLU activations in the coupling MLPs.
    scale_cap : float, default=3.0
        Bound on the log-scale output `ŝ = scale_cap * tanh(raw)`. Keeps exp(ŝ) in
        [exp(-scale_cap), exp(scale_cap)] to avoid overflow/underflow during inverse.
    spectral_norm_scales : bool, default=False
        If True, apply spectral normalization to all Linear layers in the scale head MLP.

    Returns
    -------
    normflows.NormalizingFlow
        A RealNVP model: alternating masked affine couplings + ActNorm, with bounded scales.

    Example
    -------
    >>> import normflows as nf
    >>> d = 512
    >>> q0 = nf.distributions.GaussianPCA(d, latent_dim=4)
    >>> model = create_real_nvp_normalizing_flow_model(
    ...     latent_size=d, K=8, q0=q0, leaky_relu_negative_slope=0.1,
    ...     scale_cap=3.0, spectral_norm_scales=False
    ... )
    >>> # `model.inverse(x)` and `model.forward_kld(x)` are now more numerically robust.

    Notes
    -----
    * This preserves your original architecture choices (alternating masks, ActNorm),
      but **replaces the unbounded scale head** to prevent Inf/NaN latents stemming from
      `exp(s)` extremes. :contentReference[oaicite:2]{index=2}
    * Keep `init_zeros=True` in both heads to start near identity, which plays well with ActNorm.
    """
    flows = []

    # Alternating binary mask [1,0,1,0,...]
    b = torch.tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)], dtype=torch.float32)

    for i in range(K):
        # Shift head (t): zero-initialized last layer → identity at start
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size],
                        leaky=leaky_relu_negative_slope,
                        init_zeros=True)

        # Scale head (s): bounded via tanh; zero-initialized last layer → identity at start
        s = _BoundedMLP([latent_size, 2 * latent_size, latent_size],
                        leaky=leaky_relu_negative_slope,
                        scale_cap=scale_cap,
                        init_zeros=True,
                        spectral_norm=spectral_norm_scales)

        # Affine coupling with alternating masks
        if i % 2 == 0:
            flows.append(nf.flows.MaskedAffineFlow(b, t, s))
        else:
            flows.append(nf.flows.MaskedAffineFlow(1 - b, t, s))

        # ActNorm after each coupling (data-dependent init on first batch)
        flows.append(nf.flows.ActNorm(latent_size))

    model = nf.NormalizingFlow(q0=q0, flows=flows)
    return model
