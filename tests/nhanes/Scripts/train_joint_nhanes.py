import torch
import numpy as np
import pandas as pd

import antstorch

import normflows as nf

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm

cuda_device = 'cuda:1'
torch.manual_seed(0)

base_directory = "../"
which = ["nh_list_2", "nh_list_3", "nh_list_4", "nh_list_5"]

# model_file = base_directory + "Scripts/model_pca_" + which + ".pt"
loss_plot_file_prefix = base_directory + "Scripts/loss_joint"

pca_latent_dimension = 4
mi_beta = 1000.0
show_iter = 100

max_iter = 5000
weight_decay = 0.0
plot_interval = 100

training_datasets = list()
training_dataloaders = list()
training_iterators = list()
models = list()
combined_model_parameters = list()

print("Loading training data and generating models.")
for i in range(len(which)):
    print("  Dataset", which[i])
    csv_file = base_directory + "Data/" + which[i] + ".csv"
    training_datasets.append(antstorch.DataFrameDataset(dataframe=pd.read_csv(csv_file), 
                                                        number_of_samples=1000000))
    training_dataloaders.append(DataLoader(training_datasets[i], batch_size=512,
                                          shuffle=True, num_workers=4))
    training_iterators.append(iter(training_dataloaders[i]))
    number_of_columns = len(training_datasets[i].dataframe.columns)
    print("    model hyperparameters: ")
    print("      latent size: ", str(number_of_columns))
    print("      column names: ", training_datasets[i].dataframe.columns)
    q0 = nf.distributions.GaussianPCA(number_of_columns, latent_dim=pca_latent_dimension)
    nf_model = antstorch.create_real_nvp_normalizing_flow_model(number_of_columns,
                                                                q0=q0,
                                                                leaky_relu_negative_slope=0.2)
    models.append(nf_model)
    combined_model_parameters += list(models[i].parameters())
    # Move model on GPU if available
    enable_cuda = True
    device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
    models[i] = models[i].to(device)
    models[i] = models[i].double()

loss_hist = np.array([])
loss_kld_hist = np.array([])
loss_mi_hist = np.array([])
loss_iter = np.array([])

model_files = list()
for m in range(len(models)):
    models[m].train()
    model_files.append(base_directory + "Scripts/model_joint_" + which[m] + ".pt")

# Set up mutual information part

mine_nets = []
ma_ets = []
combined_mine_parameters = list()
mine_reg_lambda = 1e-3 
mine_update_frequency = 5 

num_model_pairs = sum(1 for m in range(len(models)) for n in range(m + 1, len(models)))

for n in range(num_model_pairs):
    net = antstorch.MINE(pca_latent_dimension, pca_latent_dimension).to(device)
    mine_nets.append(net)
    combined_mine_parameters += list(mine_nets[n].parameters())
    ma_ets.append(None)

flow_optimizer = torch.optim.Adamax(combined_model_parameters, lr=1e-4, weight_decay=1e-5)
mine_optimizer = torch.optim.Adamax(combined_mine_parameters, lr=1e-6, weight_decay=1e-5)

count_iter = 0
min_loss = 100000000
for it in tqdm(range(max_iter)):
    flow_optimizer.zero_grad()
    mine_optimizer.zero_grad()

    loss_kld = torch.tensor(0.0, device=device)
    loss_mi = torch.tensor(0.0, device=device)

    x = []
    z = []
    for m in range(len(models)):
        try:
            x.append(next(training_iterators[m]))
        except StopIteration:
            training_iterator = iter(training_dataloaders[m])
            x.append(next(training_iterator))
        x[m] = x[m].to(device).double()
        z_m = models[m].inverse(x[m])
        z.append(z_m)

    # Likelihood
    for m in range(len(models)):
        loss_kld += models[m].forward_kld(x[m])

    # Mutual information
    pair_idx = 0
    for m in range(len(models)):
        for n in range(m + 1, len(models)):
            q0_m = models[m].q0
            Wm = q0_m.W.T
            locm = q0_m.loc
            eps_m = torch.matmul(z[m] - locm, torch.linalg.pinv(Wm.T)).float()

            q0_n = models[n].q0
            Wn = q0_n.W.T
            locn = q0_n.loc
            eps_n = torch.matmul(z[n] - locn, torch.linalg.pinv(Wn.T)).float()

            mi_est, ma_ets[pair_idx] = antstorch.mutual_information_mine(eps_m, 
                                                                         eps_n, 
                                                                         mine_nets[pair_idx], 
                                                                         ma_ets[pair_idx],
                                                                         alpha=0.0001)
            # Compute regularization term
            reg = 0.0
            for param in mine_nets[pair_idx].parameters():
                reg += torch.norm(param, 2)

            # Update MINE only every 'update_frequency'
            if i % mine_update_frequency == 0:    
                (-mi_est + mine_reg_lambda * reg).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(mine_nets[pair_idx].parameters(), max_norm=1.0)
                mine_optimizer.step()
                mine_optimizer.zero_grad()

            # Add detached MI to loss
            loss_mi += (mi_beta * mi_est.detach())

            pair_idx += 1

    loss = loss_kld + loss_mi

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        flow_optimizer.step()
        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
        loss_kld_hist = np.append(loss_kld_hist, loss_kld.detach().to('cpu').numpy())
        loss_mi_hist = np.append(loss_mi_hist, loss_mi.detach().to('cpu').numpy())
        loss_iter = np.append(loss_iter, it)

    if (it + 1) % show_iter == 0:
        plt.figure(figsize=(10, 10))
        plt.figure(figsize=(30, 10))

        # Subplot 1: KLD loss
        plt.subplot(1, 3, 1)
        plt.plot(loss_iter, loss_kld_hist, label='KLD loss', color='tab:blue')
        plt.xlabel('Iteration')
        plt.ylabel('KLD loss')
        plt.title('KLD Loss')
        plt.grid(True)
        plt.legend()

        # Subplot 2: MI loss
        plt.subplot(1, 3, 2)
        plt.plot(loss_iter, -loss_mi_hist, label='-MI penalty', color='tab:orange')
        plt.xlabel('Iteration')
        plt.ylabel('Neg. MI')
        plt.title('Negative MI (Penalty)')
        plt.grid(True)
        plt.legend()

        # Subplot 3: Total loss
        plt.subplot(1, 3, 3)
        plt.plot(loss_iter, loss_hist, label='Total loss', color='tab:green')
        plt.xlabel('Iteration')
        plt.ylabel('Total loss')
        plt.title('Total Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig("loss_hist_glow_2d.pdf")
        plt.close()
        for m in range(len(models)):
            models[m].save(model_files[m])

    for m in range(len(models)):
        nf.utils.clear_grad(models[m])

    # Delete variables to prevent out of memory errors
    del loss
    del x

    count_iter += 1

