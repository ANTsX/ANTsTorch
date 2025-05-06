import torch

import normflows as nf

def create_real_nvp_normalizing_flow_model(latent_size,
                                           pca_latent_dimension, 
                                           K=64):

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    q0 = nf.distributions.GaussianPCA(latent_size, latent_dim=pca_latent_dimension)
    model = nf.NormalizingFlow(q0=q0, flows=flows)

    return model
