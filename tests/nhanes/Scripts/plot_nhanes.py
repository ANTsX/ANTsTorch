import torch
import normflows as nf
import pandas as pd
import random
import numpy as np
import umap

from torch.utils.data import Dataset

from matplotlib import pyplot as plt

cuda_device = 'cuda:0'

base_directory = "/home/ntustison/Data/NVP_nhanes/"
which = "nh_list_2"

csv_file = base_directory + "Data/" + which + ".csv"
model_file = base_directory + "Scripts/model_pca_" + which + ".pt"
umap_plot_file = base_directory + "Scripts/Plots/umap_pca_" + which + ".png"

pca_latent_dim = 4

# Set up datasets/dataloaders

class CsvDataset(Dataset):
    def __init__(self,
                 csv_file,
                 alpha=0.01,
                 do_normalize=True,
                 do_data_augmentation=True,
                 number_of_samples=1):
        self.csv_file = csv_file
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.do_normalize = do_normalize
        self.do_data_augmentation = do_data_augmentation
        csv = pd.read_csv(csv_file)
        self.csv_data_colnames = csv.columns  
        self.csv_data = csv.to_numpy()
        self.number_of_measurements = self.csv_data.shape[0]
        self.csv_std = np.std(self.csv_data, axis=0)
        self.csv_mean = np.mean(self.csv_data, axis=0)
        self.csv_min = np.min(self.csv_data, axis=0)
        self.csv_max = np.max(self.csv_data, axis=0)
    def __len__(self):
        return self.number_of_samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        random_index = random.sample(range(self.number_of_measurements), k=1)[0]
        random_measurement = self.csv_data[random_index,:]
        if self.do_data_augmentation:
            random_measurement += np.random.normal(np.zeros(random_measurement.shape),
                                                   self.alpha * self.csv_std)
        if self.do_normalize:
            random_measurement = self.normalize_data(random_measurement)
        return random_measurement 
    def normalize_data(self, data):
        if len(data.shape) == 2:
            min = np.tile(self.csv_min, (self.number_of_measurements, 1))
            max = np.tile(self.csv_max, (self.number_of_measurements, 1))
            normalized_data = (data - min) / (max - min)
        else:
            normalized_data = (data - self.csv_min) / (self.csv_max - self.csv_min) 
        return normalized_data       
    def denormalize_data(self, data):
        if len(data.shape) == 2:
            min = np.tile(self.csv_min, (self.number_of_measurements, 1))
            max = np.tile(self.csv_max, (self.number_of_measurements, 1))
            denormalized_data = data * (max - min) + min
        else:
            denormalized_data = data * (self.csv_max - self.csv_min) + self.csv_min
        return denormalized_data       

dataset = CsvDataset(csv_file=csv_file, number_of_samples=1000000)
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

# Construct flow model
# q0 = nf.distributions.DiagGaussian(latent_size)
q0 = nf.distributions.GaussianPCA(latent_size, latent_dim=pca_latent_dim)
model = nf.NormalizingFlow(q0=q0, flows=flows)

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
x_np = x_np[~nan_rows & ~inf_rows]
u2 = fit.transform(x_np)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('UMAP embedding')
ax1.scatter(u[:,0], u[:,1], c='blue', s=10, alpha=0.5)
ax1.set_title("original data")
ax2.scatter(u2[:,0], u2[:,1], c='orange', s=10, alpha=0.5)
ax2.set_title("generated data")
plt.savefig(umap_plot_file)
plt.show()