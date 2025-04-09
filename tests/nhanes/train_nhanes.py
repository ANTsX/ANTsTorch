import torch
import numpy as np
import normflows as nf
import os
import pandas as pd
import random

from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm

base_directory = "/home/ntustison/Data/NVP_nhanes/Data/"

cuda_device = 'cuda:0'
csv_file = base_directory + "nh_list_2.csv"

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
        self.csv_data = pd.read_csv(csv_file).to_numpy()
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
            # random_measurement = (random_measurement - self.csv_mean) / self.csv_std 
            random_measurement = (random_measurement - self.csv_min) / (self.csv_max  - self.csv_min)
        return random_measurement 

training_dataset = CsvDataset(csv_file=csv_file, number_of_samples=1000000)
testing_dataset = CsvDataset(csv_file=csv_file, number_of_samples=1000000)

training_dataloader = DataLoader(training_dataset, batch_size=64,
                                 shuffle=True, num_workers=4)
testing_dataloader = DataLoader(testing_dataset, batch_size=32,
                                shuffle=True, num_workers=4)

training_iterator = iter(training_dataloader)

number_of_columns = training_dataset.csv_data.shape[1]

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
q0 = nf.distributions.DiagGaussian(latent_size)
model = nf.NormalizingFlow(q0=q0, flows=flows)

# Move model on GPU if available
enable_cuda = True
device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)
model = model.double()


# Train model
max_iter = 20000
lr = 1e-4
weight_decay = 0.0
show_iter = 100

loss_hist = np.zeros((max_iter, 2))

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

count_iter = 0
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    try:
        x = next(training_iterator)
    except StopIteration:
        training_iterator = iter(training_dataloader)
        x = next(training_iterator)
    x = x.to(device, non_blocking=True)
    loss = model.forward_kld(x)
    
    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist[count_iter, 0] = it
    loss_hist[count_iter, 1] = loss.cpu().detach().numpy().item()
    if (count_iter + 1) % show_iter == 0:
        plt.plot(loss_hist[:count_iter+1,0], loss_hist[:count_iter+1,1], label="Loss")
        plt.grid(True)
        plt.axis('tight')
        plt.savefig("Plots/loss_" + str(count_iter) + ".png")

    # Clear gradients
    nf.utils.clear_grad(model)

    # Delete variables to prevent out of memory errors
    del loss
    del x

    count_iter += 1

