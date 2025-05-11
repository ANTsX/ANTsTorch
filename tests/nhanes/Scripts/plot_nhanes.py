import torch
import normflows as nf
import pandas as pd
import random
import numpy as np
import umap

import antstorch

from matplotlib import pyplot as plt

cuda_device = 'cuda:0'

base_directory = "/home/ntustison/Data/NVP_nhanes/"
which = "nh_list_5"

csv_file = base_directory + "Data/" + which + ".csv"
model_file = base_directory + "Scripts/model_joint_" + which + ".pt"
umap_plot_file = base_directory + "Scripts/Plots/umap_joint_" + which + ".png"

pca_latent_dim = 4

# Set up datasets/dataloaders

dataset = antstorch.CsvDataset(csv_file=csv_file, number_of_samples=1000000)
number_of_columns = dataset.csv_data.shape[1]

# Define model
K = 64
torch.manual_seed(0)

latent_size = number_of_columns
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

# Define model
K = 64
torch.manual_seed(0)

q0 = nf.distributions.GaussianPCA(number_of_columns, latent_dim=pca_latent_dim)
model = antstorch.create_real_nvp_normalizing_flow_model(number_of_columns,
                                                         q0=q0)

# Move model on GPU if available
enable_cuda = True
device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)
model = model.double()

# Plot

number_of_samples = dataset.csv_data.shape[0]
model.load(model_file)
z = torch.tensor(np.random.normal(0, 1, (number_of_samples, pca_latent_dim)), dtype=torch.float64)
z = z.to(device, non_blocking=True)
x = model.forward(torch.matmul(z, q0.W))
# Sanity check:
# zz = model.inverse(x)

fit = umap.UMAP()
u = fit.fit_transform(dataset.normalize_data(dataset.csv_data))

x_np = x.cpu().detach().numpy()
nan_rows = np.isnan(x_np).any(axis=1)
inf_rows = np.isinf(x_np).any(axis=1)
is_finite = np.isfinite(x_np)
finite_row_indices = np.where(np.all(is_finite, axis=1))[0]
x_np = x_np[finite_row_indices,:]
u2 = fit.transform(x_np)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('UMAP embedding')
ax1.scatter(u[:,0], u[:,1], c='blue', s=10, alpha=0.5)
ax1.set_title("original data")
ax2.scatter(u2[:,0], u2[:,1], c='orange', s=10, alpha=0.5)
ax2.set_title("generated data")
plt.savefig(umap_plot_file)
plt.show()