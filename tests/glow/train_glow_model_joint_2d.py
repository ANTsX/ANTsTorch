# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import glob
import os
import math

import ants 
import antspynet
import antstorch

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define flows
L = 4
K = 2
hidden_channels = 64
resampled_image_size = (64, 64)
max_iter = 1000000
plot_interval = 100

############################################################################

cuda_device = 'cuda:0'
torch.manual_seed(0)
channels = 1
input_shape = (channels, *resampled_image_size)
n_dims = np.prod(input_shape)
split_mode = 'channel'
scale = True
use_mutual_information_penalty = False
beta = 100.0

# Set up flows, distributions and merge operations for each of T1, T2, FA

modalities = ['T1', 'T2', 'FA']

models = list()
combined_model_parameters = list()

mine_latent_dim = 0

for m in range(len(modalities)):
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock2d(channels * 2 ** (L + 1 - i), hidden_channels,
                                            split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze2d()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        if m == 0:
            mine_latent_dim += math.prod(latent_shape)
        q0 += [nf.distributions.DiagGaussian(latent_shape)]
    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)
    models.append(model)
    combined_model_parameters += list(models[m].parameters())
    # Move model on GPU if available
    enable_cuda = True
    device = torch.device(cuda_device if torch.cuda.is_available() and enable_cuda else 'cpu')
    models[m] = models[m].to(device)
    models[m] = models[m].double()

model_files = list()
for m in range(len(models)):
    models[m].train()
    model_files.append("model_joint_2d" + modalities[m] + ".pt")
    if os.path.exists(model_files[m]):
        print("Loading " + model_files[m])
        models[m].load_state_dict(torch.load(model_files[m]))

# Set up penalty part

if use_mutual_information_penalty:

    penalty_string = "Mutual information"
 
    mine_nets = []
    ma_ets = []
    combined_penalty_parameters = list()
    mine_reg_lambda = 1e-2
    mine_update_frequency = 10

    num_model_pairs = sum(1 for m in range(len(models)) for n in range(m + 1, len(models)))

    for n in range(num_model_pairs):
        net = antstorch.MINE(mine_latent_dim, mine_latent_dim).to(device)
        mine_nets.append(net)
        combined_penalty_parameters += list(mine_nets[n].parameters())
        ma_ets.append(None)
else:

    penalty_string = "Pearson Correlation"
        

loss_hist = np.zeros((max_iter, 2))

# Prepare training data
batch_size = 128

hcpya_images = list()
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaT1Template")))
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaT2Template")))
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaFATemplate")))

hcpya_slices = list()
for i in range(len(hcpya_images)):
    hcpya_slices.append(ants.slice_image(hcpya_images[i], 
                                         axis=2, idx=120, 
                                         collapse_strategy=1))
template = ants.resample_image(hcpya_slices[0],
                               resampled_image_size,
                               use_voxels=True)

transformed_dataset = antstorch.ImageDataset(images=[hcpya_slices],
                                             template=template,
                                             do_data_augmentation=True,
                                             data_augmentation_transform_type="affineAndDeformation",
                                             data_augmentation_sd_affine=0.02,
                                             data_augmentation_sd_deformation=10.0,
                                             data_augmentation_noise_model=None,
                                             data_augmentation_sd_simulated_bias_field=0.0,
                                             data_augmentation_sd_histogram_warping=0.05,
                                             number_of_samples=100)
transformed_dataset_testing = antstorch.ImageDataset(images=[hcpya_slices],
                                                     template=template,
                                                     do_data_augmentation=True,
                                                     data_augmentation_transform_type="affineAndDeformation",
                                                     data_augmentation_sd_affine=0.05,
                                                     data_augmentation_sd_deformation=0.2,
                                                     data_augmentation_noise_model="additivegaussian",
                                                     data_augmentation_sd_simulated_bias_field=1.0,
                                                     data_augmentation_sd_histogram_warping=0.05,
                                                     number_of_samples=16)
train_loader = DataLoader(transformed_dataset, batch_size=16,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(transformed_dataset_testing, batch_size=16,
                         shuffle=True, num_workers=4)

train_iter = iter(train_loader)

# Train model

loss_hist = np.array([])
loss_kld_hist = np.array([])
loss_penalty_hist = np.array([])
loss_iter = np.array([])

flow_optimizer = torch.optim.Adamax(combined_model_parameters, lr=1e-4, weight_decay=1e-5)
if use_mutual_information_penalty:
    mine_optimizer = torch.optim.Adamax(combined_penalty_parameters, lr=1e-6, weight_decay=1e-5)


for i in tqdm(range(max_iter)):
    flow_optimizer.zero_grad()
    
    if use_mutual_information_penalty:
        mine_optimizer.zero_grad()

    try:
        x = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x = next(train_iter)

    # for b in range(4):   
    #     for m in range(len(models)):
    #         x_m = x[b,m:m+1,:,:].numpy()
    #         ants.image_write(ants.from_numpy(x_m), "x_" + str(b) + str(m) + ".nii.gz")
    # raise ValueError("HERE")

    loss_kld = torch.tensor(0.0, device=device)
    loss_penalty = torch.tensor(0.0, device=device)
 
    z = []
    for m in range(len(models)):
        x_m = x[:,m:m+1,:,:].to(device)
        z_m, _ = models[m].inverse_and_log_det(x_m)
        z.append(z_m)
        loss_kld += models[m].forward_kld(x_m)

    # Penalty term
    pair_idx = 0
    for m in range(len(models)):
        for n in range(m + 1, len(models)):
            zm_flat = [z[m][p].reshape(z[m][p].shape[0], -1).float() for p in range(len(z[m]))]
            zn_flat = [z[n][p].reshape(z[n][p].shape[0], -1).float() for p in range(len(z[n]))]
            zm = torch.cat(zm_flat, dim=1)
            zn = torch.cat(zn_flat, dim=1)

            if use_mutual_information_penalty: 

                mi_est, ma_ets[pair_idx] = antstorch.mutual_information_mine(zm, 
                                                                             zn, 
                                                                             mine_nets[pair_idx], 
                                                                             ma_ets[pair_idx],
                                                                             alpha=0.0001,
                                                                             loss_type='fdiv')
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
                loss_penalty += (beta * mi_est.detach())

            else:
                corr_value = antstorch.absolute_pearson_correlation(zm, zn, 1e-6)
                loss_penalty += (beta * -corr_value)

            pair_idx += 1

    loss = loss_kld + loss_penalty

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        flow_optimizer.step()
        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
        loss_kld_hist = np.append(loss_kld_hist, loss_kld.detach().to('cpu').numpy())
        loss_penalty_hist = np.append(loss_penalty_hist, loss_penalty.detach().to('cpu').numpy())
        loss_iter = np.append(loss_iter, i)

    if (i + 1) % plot_interval == 0:
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
        plt.savefig("loss_hist_glow_2d.pdf")
        plt.close()
        for m in range(len(models)):
            models[m].save(model_files[m])

        for m in range(len(models)):
            nf.utils.clear_grad(models[m])

            with torch.no_grad():
                for m in range(len(models)):
                    x, _ = models[m].sample(100)
                    x_ = torch.clamp(x, 0, 1)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=10).cpu().numpy(), (1, 2, 0)))
                    # plt.show()
                    plt.savefig("samples_glow_2d_model" + str(m) + ".pdf")
                    plt.close()
    

