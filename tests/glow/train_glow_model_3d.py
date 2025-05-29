# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import glob
import os

import ants 
import antstorch

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

# Set up model

# Define flows
L = 3
K = 2
hidden_channels = 64
resampled_image_size = (128, 128, 128)
max_iter = 1000000
plot_interval = 10000
model_file = "model_glow_3d.pt"

############################################################################

torch.manual_seed(0)
channels = 1
input_shape = (channels, *resampled_image_size)
n_dims = np.prod(input_shape)
split_mode = 'channel'
scale = True

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    number_of_channels = channels * 2 ** (2 * (L - i) + 1)   
    for j in range(K):
        flows_ += [nf.flows.GlowBlock3d(number_of_channels, hidden_channels,
                                        split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze3d()]
    flows += [flows_]

    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (number_of_channels // 2,
                        input_shape[1] // 2 ** (L - i),
                        input_shape[2] // 2 ** (L - i),
                        input_shape[3] // 2 ** (L - i))
    else:
        latent_shape = (number_of_channels,
                        input_shape[1] // 2 ** L,
                        input_shape[2] // 2 ** L,
                        input_shape[3] // 2 ** L)
    q0 += [nf.distributions.DiagGaussian(latent_shape)]

# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges)

if os.path.exists(model_file):
    print("Loading existing file.")
    model.load(model_file)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

# Prepare training data
batch_size = 128

image = ants.image_read(antstorch.get_antstorch_data("kirby"))
template = ants.resample_image(image,
                               resampled_image_size,
                               use_voxels=True)

transformed_dataset = antstorch.ImageDataset(images=[image],
                                             template=template,
                                             duplicate_channels=channels,
                                             number_of_samples=100)
transformed_dataset_testing = antstorch.ImageDataset(images=[image],
                                                     template=template,
                                                     duplicate_channels=channels,
                                                     number_of_samples=16)
train_loader = DataLoader(transformed_dataset, batch_size=16,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(transformed_dataset_testing, batch_size=16,
                         shuffle=True, num_workers=4)

train_iter = iter(train_loader)

# Train model

loss_hist = np.array([])
loss_iter = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    try:
        x = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device))

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        if i == 0:
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
            loss_iter = np.append(loss_iter, i)
        else:
            if loss.detach().to('cpu').numpy() < loss_hist[0]:
                loss.backward()
                optimizer.step()
                loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
                loss_iter = np.append(loss_iter, i)

    if (i + 1) % plot_interval == 0:
        plt.figure(figsize=(10, 10))
        plt.plot(loss_iter, loss_hist, label='loss')
        plt.legend()
        # plt.show()
        plt.savefig("loss_hist_glow_3d_" + str(i+1) + ".pdf")
        model.save(model_file)

        # Model samples
        num_sample = 10

# Get bits per dim
n = 0
bpd_cum = 0
with torch.no_grad():
    for x in iter(test_loader):
        nll = model(x.to(device))
        nll_np = nll.cpu().numpy() 
        bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
        n += len(x) - np.sum(np.isnan(nll_np))
        
    print('Bits per dim: ', bpd_cum / n)

