import torch
import numpy as np
import normflows as nf
import pandas as pd

import antstorch

import normflows as nf

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm

cuda_device = 'cuda:0'

base_directory = "../"
which = "nh_list_2"

csv_file = base_directory + "Data/" + which + ".csv"
model_file = base_directory + "Scripts/model_pca_" + which + ".pt"
loss_plot_file_prefix = base_directory + "Scripts/Plots/loss_pca_" + which

pca_latent_dim = 4
show_iter = 100

training_dataset = antstorch.DataFrame(dataframe=pd.read(csv_file), number_of_samples=1000000)
training_dataloader = DataLoader(training_dataset, batch_size=64,
                                 shuffle=True, num_workers=4)
training_iterator = iter(training_dataloader)

# testing_dataset = CsvDataset(csv_file=csv_file, number_of_samples=1000000)
# testing_dataloader = DataLoader(testing_dataset, batch_size=32,
#                                 shuffle=True, num_workers=4)

number_of_columns = training_dataset.csv_data.shape[1]

print("Latent size: ", str(number_of_columns))
print("Column names: ")
print(training_dataset.colnames)

# Define model
K = 64
torch.manual_seed(0)

model = antstorch.create_real_nvp_normalizing_flow_model(number_of_columns,
                                                         pca_latent_dimension=pca_latent_dim)

enable_cuda = True
device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)
model = model.double()

# Train model
max_iter = 5000
lr = 1e-4
weight_decay = 0.0

loss_hist = np.zeros((max_iter, 2))

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

count_iter = 0
min_loss = 100000000
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
    if loss_hist[count_iter, 1] < min_loss:
        min_loss = loss_hist[count_iter, 1]
        model.save(model_file)

    if (count_iter + 1) % show_iter == 0:
        plt.plot(loss_hist[:count_iter+1,0], loss_hist[:count_iter+1,1], label="Loss")
        plt.grid(True)
        plt.axis('tight')
        plt.savefig(loss_plot_file_prefix + ".png")

    # Clear gradients
    nf.utils.clear_grad(model)

    # Delete variables to prevent out of memory errors
    del loss
    del x

    count_iter += 1

