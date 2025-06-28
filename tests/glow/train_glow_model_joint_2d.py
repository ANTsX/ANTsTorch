# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import glob
import os

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
resampled_image_size = (128,128)
max_iter = 1000000
plot_interval = 10000

############################################################################

cuda_device = 'cuda:0'
torch.manual_seed(0)
channels = 1
input_shape = (channels, *resampled_image_size)
n_dims = np.prod(input_shape)
split_mode = 'channel'
scale = True
mi_beta = 0.1

# Set up flows, distributions and merge operations for each of T1, T2, FA

modalities = ['T1', 'T2', 'FA']

models = list()
combined_model_parameters = list()

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

loss_hist = np.zeros((max_iter, 2))

model_files = list()
for m in range(len(models)):
    models[m].train()
    model_files.append("model_joint_2d" + modalities[m] + ".pt")


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
loss_iter = np.array([])

optimizer = torch.optim.Adamax(combined_model_parameters, lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    optimizer.zero_grad()

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

    loss = torch.tensor(0.0)
    loss = loss.to(device, non_blocking=True)
 
    z = list()
    for m in range(len(models)):
        x_m = x[:,m:m+1,:,:].to(device, non_blocking=True)
        z_m, _ = models[m].inverse_and_log_det(x_m)
        for n in range(len(z_m)):
            z_m[n].to(device, non_blocking=True)
        z.append(z_m)
        loss += models[m].forward_kld(x_m)

    # Mutual information
    for m in range(len(models)):
        for n in range(m + 1, len(models)):
            for p in range(len(z[m])):
                loss += (mi_beta * antstorch.mutual_information_kde(z[m][p].reshape(1, -1), 
                                                                    z[n][p].reshape(1, -1))) 

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        if i == 0:
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
            loss_iter = np.append(loss_iter, i)
        else:
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
            loss_iter = np.append(loss_iter, i)

    if (i + 1) % plot_interval == 0:
        plt.figure(figsize=(10, 10))
        plt.plot(loss_iter, loss_hist, label='loss')
        plt.legend()
        # plt.show()
        plt.savefig("loss_hist_glow_2d_" + str(i+1) + ".pdf")
        for m in range(len(models)):
            models[m].save(model_files[m])

        # Model samples
        num_sample = 10

        with torch.no_grad():
            # y = torch.arange(1).repeat(num_sample).to(device)
            for m in range(len(models)):
                x, _ = models[m].sample(100)
                x_ = torch.clamp(x, 0, 1)
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=10).cpu().numpy(), (1, 2, 0)))
                # plt.show()
                plt.savefig("samples_glow_2d_" + str(i+1) + "_model" + str(m) + ".pdf")
    

