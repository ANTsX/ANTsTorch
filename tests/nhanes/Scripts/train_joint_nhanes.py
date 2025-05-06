import torch
import numpy as np
import normflows as nf
import pandas as pd
import random

import antstorch

from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm

cuda_device = 'cuda:0'
torch.manual_seed(0)

base_directory = "../"
which = ["nh_list_2", "nh_list_3", "nh_list_4", "nh_list_5"]

# model_file = base_directory + "Scripts/model_pca_" + which + ".pt"
loss_plot_file_prefix = base_directory + "Scripts/Plots/loss_joint"

pca_latent_dimension = 4
mi_beta = 0.1
show_iter = 100

max_iter = 5000
lr = 1e-4
weight_decay = 0.0


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

def create_normalizing_flow_model(latent_size,
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

training_datasets = list()
training_dataloaders = list()
training_iterators = list()
models = list()
combined_model_parameters = list()

print("Loading training data and generating models.")
for i in range(len(which)):
    print("  Dataset", which[i])
    csv_file = base_directory + "Data/" + which[i] + ".csv"
    training_datasets.append(CsvDataset(csv_file=csv_file, number_of_samples=1000000))
    training_dataloaders.append(DataLoader(training_datasets[i], batch_size=64,
                                          shuffle=True, num_workers=4))
    training_iterators.append(iter(training_dataloaders[i]))
    number_of_columns = training_datasets[i].csv_data.shape[1]
    print("    model hyperparameters: ")
    print("      latent size: ", str(number_of_columns))
    print("      column names: ", training_datasets[i].csv_data_colnames)
    nf_model = create_normalizing_flow_model(number_of_columns,
                                             pca_latent_dimension=pca_latent_dimension)
    models.append(nf_model)
    combined_model_parameters += list(models[i].parameters())
    # Move model on GPU if available
    enable_cuda = True
    device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
    models[i] = models[i].to(device)
    models[i] = models[i].double()


loss_hist = np.zeros((max_iter, 2))

model_files = list()
for m in range(len(models)):
    models[m].train()
    model_files.append(base_directory + "Scripts/model_joint_" + which[m] + ".pt")

optimizer = torch.optim.Adam(combined_model_parameters, lr=lr, weight_decay=weight_decay)

count_iter = 0
min_loss = 100000000
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    loss = torch.tensor(0.0)
    loss = loss.to(device, non_blocking=True)

    x = list()
    z = list()    
    for m in range(len(models)):
        try:
            x.append(next(training_iterators[m]))
        except StopIteration:
            training_iterator = iter(training_dataloaders[m])
            x.append(next(training_iterator))
        x[m] = x[m].to(device, non_blocking=True)
        z.append(models[m].inverse(x[m]))
        z[m] = z[m].to(device, non_blocking=True)

    # Likelihood
    for m in range(len(models)):
        loss += models[m].forward_kld(x[m])

    # Mutual information
    for m in range(len(models)):
        for n in range(m + 1, len(models)):
            loss += (mi_beta * antstorch.mutual_information_kde(z[m], z[n])) 

    if (count_iter + 1) % show_iter == 0:
        plt.plot(loss_hist[:count_iter+1,0], loss_hist[:count_iter+1,1], label="Loss")
        plt.grid(True)
        plt.axis('tight')
        plt.savefig(loss_plot_file_prefix + ".png")

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist[count_iter, 0] = it
    loss_hist[count_iter, 1] = loss.cpu().detach().numpy().item()
    if loss_hist[count_iter, 1] < min_loss:
        min_loss = loss_hist[count_iter, 1]
        for m in range(len(models)):
            models[m].save(model_files[m])

    for m in range(len(models)):
        nf.utils.clear_grad(models[m])

    # Delete variables to prevent out of memory errors
    del loss
    del x

    count_iter += 1

