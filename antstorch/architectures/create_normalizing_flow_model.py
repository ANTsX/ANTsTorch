import torch

import normflows as nf

def create_real_nvp_normalizing_flow_model(latent_size,
                                           K=64,
                                           q0=None, 
                                           leaky_relu_negative_slope=0.0):
    """
    Create Real NVP model.

    Arguments
    ---------
    latent_size : integer
        Input size.

    K : integer
        Number of layers    

    q0 : base distribution
        Base distribution defined in the normflows package
        e.g., diagonal gaussian q0 = nf. DiagGaussian(latent_size).
        None is also a possibility. 

    leaky_relu_negative_slope : float
        Negative slope value for leaky relu layers.    

    Returns
    -------
    PyTorch model
        A PyTorch model defining the network.

    Example
    -------
    >>> import normflows as nf
    >>> input_size = 512
    >>> q0 = nf.distributions.GaussianPCA(input_size, latent_dim=4)
    >>> model = antstorch.create_real_nvp_normalizing_flow_model(input_size,
                                                                 K=4, 
                                                                 q0=q0, 
                                                                 leaky_relu_negative_slope=0.0)
                                                                 
    """

    flows = []

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size],
                        leaky=leaky_relu_negative_slope, 
                        init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size],
                        leaky=leaky_relu_negative_slope, 
                        init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    model = nf.NormalizingFlow(q0=q0, flows=flows)

    return model
