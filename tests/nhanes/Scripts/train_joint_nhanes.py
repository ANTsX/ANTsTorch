import torch
import numpy as np
import pandas as pd
import antstorch
import normflows as nf
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

############################################################################

cuda_device = 'cuda:1'
torch.manual_seed(0)

base_directory = "../"
which = ["nh_list_2", "nh_list_3", "nh_list_4", "nh_list_5"]

loss_plot_file_prefix = base_directory + "Scripts/loss_joint_nhanes"

pca_latent_dimension = 4
use_mutual_information_penalty = False
show_iter = 100
max_iter = 2000
weight_decay = 0.0
plot_interval = 100

beta = 1.0


training_datasets = list()
training_dataloaders = list()
training_iterators = list()
models = list()
combined_model_parameters = list()

print("Loading training data and generating models.")
for i in range(len(which)):
    print("  Reading dataset", which[i])
    csv_file = base_directory + "Data/" + which[i] + ".csv"
    training_datasets.append(antstorch.DataFrameDataset(dataframe=pd.read_csv(csv_file), 
                                                        number_of_samples=1000000,
                                                        normalization_type="0mean"))
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
loss_penalty_hist = np.array([])
loss_iter = np.array([])

model_files = list()
for m in range(len(models)):
    models[m].train()
    model_files.append(base_directory + "Scripts/model_joint_" + which[m] + ".pt")
    if os.path.exists(model_files[m]):
        print("Loading " + model_files[m])
        models[m].load_state_dict(torch.load(model_files[m]))

# Set up penalty part

if use_mutual_information_penalty:
    penalty_string = "Mutual information" 
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
else:
    penalty_string = "Pearson Correlation"


flow_optimizer = torch.optim.Adamax(combined_model_parameters, lr=1e-4, weight_decay=1e-5)
if use_mutual_information_penalty:
    mine_optimizer = torch.optim.Adamax(combined_mine_parameters, lr=1e-6, weight_decay=1e-5)

count_iter = 0
min_loss = 100000000
for i in tqdm(range(max_iter)):
    flow_optimizer.zero_grad()

    if use_mutual_information_penalty:
        mine_optimizer.zero_grad()

    loss_kld = torch.tensor(0.0, device=device)
    loss_penalty = torch.tensor(0.0, device=device)

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

    # Penalty term
    pair_idx = 0
    for m in range(len(models)):
        for n in range(m + 1, len(models)):
            q0_m = models[m].q0
            Wm = q0_m.W.T
            locm = q0_m.loc
#            eps_m = torch.matmul(z[m] - locm, torch.linalg.pinv(Wm.T)).float()
            eps_m = torch.matmul(z[m] - locm, Wm).float()

            q0_n = models[n].q0
            Wn = q0_n.W.T
            locn = q0_n.loc
#            eps_n = torch.matmul(z[n] - locn, torch.linalg.pinv(Wn.T)).float()
            eps_n = torch.matmul(z[n] - locn, Wn).float()

            if use_mutual_information_penalty:   

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

                # Add detached MI estimate to loss
                if i > 0:
                    last_n_elements = min(100, i)
                    local_beta = loss_hist[-last_n_elements:].mean() * 0.10
                else:
                    local_beta = beta
                loss_penalty += (local_beta * mi_est.detach())

            else:

                corr_value = antstorch.absolute_pearson_correlation(eps_m, eps_n, 1e-6)
                if i > 0:
                    last_n_elements = min(100, i)
                    local_beta = loss_hist[-last_n_elements:].mean() * 0.10
                else:
                    local_beta = beta
                loss_penalty += (local_beta * -corr_value)

            pair_idx += 1

    loss = loss_kld + loss_penalty

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        flow_optimizer.step()
        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
        loss_kld_hist = np.append(loss_kld_hist, loss_kld.detach().to('cpu').numpy())
        loss_penalty_hist = np.append(loss_penalty_hist, loss_penalty.detach().to('cpu').numpy())
        loss_iter = np.append(loss_iter, i)

    if (i + 1) % show_iter == 0:
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

        # Subplot 2: Penalty loss
        plt.subplot(1, 3, 2)
        plt.plot(loss_iter, -loss_penalty_hist, label='Penalty', color='tab:orange')
        plt.xlabel('Iteration')
        plt.ylabel(penalty_string)
        plt.title('Penalty')
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
        plt.savefig(loss_plot_file_prefix + ".pdf")
        plt.close()

    for m in range(len(models)):
        models[m].save(model_files[m])

    for m in range(len(models)):
        nf.utils.clear_grad(models[m])

    # Delete variables to prevent out of memory errors
    del loss
    del x

    count_iter += 1

print("Transform training data to Gaussian.")
for m in range(len(which)):
    models[m].eval()
    with torch.inference_mode():
        print("  Writing transformed dataset z_", which[m], sep="")
        csv_file = base_directory + "Data/" + which[m] + ".csv"
        pd_x = pd.read_csv(csv_file)
        df_x = pd_x.to_numpy()
        df_x = (df_x - np.mean(df_x, axis=0)) / np.std(df_x, axis=0)
        x = torch.from_numpy(df_x).to(device).double()
        z = models[m].inverse(x)
        df_z = pd.DataFrame(z.cpu().detach().numpy(), columns=pd_x.columns)
        csv_file_z = base_directory + "Data/z_joint_" + which[m] + ".csv"
        df_z.to_csv(csv_file_z, index=False)
