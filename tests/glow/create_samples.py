# Import required packages
import torch
import numpy as np
import normflows as nf
import os

import ants 
import antspynet
import antstorch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define flows
L = 4
K = 2
hidden_channels = 64
resampled_image_size = (128,128)
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

modalities = ['T2', 'T1', 'FA']

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
    models[m].eval()
    model_files.append("model_joint_2d" + modalities[m] + ".pt")
    if os.path.exists(model_files[m]):
        print("Loading " + model_files[m])
        models[m].load_state_dict(torch.load(model_files[m]))

# Prepare training data

hcpya_images = list()
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaT2Template")))
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaT1Template")))
hcpya_images.append(ants.image_read(antspynet.get_antsxnet_data("hcpyaFATemplate")))

hcpya_slices = list()
for i in range(len(hcpya_images)):
    hcpya_slices.append(ants.slice_image(hcpya_images[i], 
                                         axis=2, idx=120, 
                                         collapse_strategy=1))

template = ants.resample_image(hcpya_slices[0],
                               resampled_image_size,
                               use_voxels=True)

for m in range(len(models)):
    print("HERE: ", str(m))
    x_m = models[m].sample(1)
    x_m_sample = np.squeeze(x_m[0].detach().cpu().numpy())
    x_m_sample[x_m_sample > 1] = 1.0
    x_m_sample[x_m_sample < 0] = 0.0
    x_m_image = ants.from_numpy(x_m_sample)
    ants.image_write(x_m_image, "eps0_" + modalities[m] + ".nii.gz")

