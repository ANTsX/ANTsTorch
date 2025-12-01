import ants
import torch
from torchvision import models

import numpy as np
import pandas as pd

def chexnet(image,
            verbose=False):

    """
    Predict lung disease category from chest x-ray.

    Disease categories:
        'Atelectasis'
        'Cardiomegaly'
        'Effusion'
        'Infiltration'
        'Mass'
        'Nodule'
        'Pneumonia'
        'Pneumothorax'
        'Consolidation'
        'Edema'
        'Emphysema'
        'Fibrosis'
        'Pleural_Thickening'
        'Hernia'

    Reproducing this work:   https://github.com/jrzech/reproduce-chexnet
      fork:  https://github.com/ntustison/reproduce-chexnet

    NB:  There are slight differences due to the internal PyTorch transforms
    that are not reproduced here, i.e.,

    >>> image = Image.open(image_file)
    >>> image = image.convert('RGB')
    >>>
    >>> # use imagenet mean,std for normalization
    >>> mean = [0.485, 0.456, 0.406]
    >>> std = [0.229, 0.224, 0.225]
    >>> data_transforms = transforms.Compose([transforms.Resize(224),
    >>>                                       transforms.CenterCrop(224),
    >>>                                       transforms.ToTensor(),
    >>>                                       transforms.Normalize(mean, std)
    >>>                                     ])

    Arguments
    ---------
    image : ANTsImage
        2-D coronal x-ray image.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------

    Data frame with probability values for each disease category.

    Example
    -------
    >>> image = ants.image_read()
    >>> classification = antstorch.chexnet(image)
    """

    from ..utilities import get_pretrained_network

    if image.dimension != 2:
        raise ValueError( "Image dimension must be 2." )

    disease_categories = ['Atelectasis',
                          'Cardiomegaly',
                          'Effusion',
                          'Infiltration',
                          'Mass',
                          'Nodule',
                          'Pneumonia',
                          'Pneumothorax',
                          'Consolidation',
                          'Edema',
                          'Emphysema',
                          'Fibrosis',
                          'Pleural_Thickening',
                          'Hernia']

    ################################
    #
    # Load model and weights
    #
    ################################

    if verbose:
        print("Loading model and weights.")

    weights_file_name = get_pretrained_network("chexnet_repro_pytorch")

    model = models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(torch.nn.Linear(model.classifier.in_features,
                                                           len(disease_categories)),
                                           torch.nn.Sigmoid())
    model.eval()
    model.load_state_dict(torch.load(weights_file_name))

    ################################
    #
    # Prepare image
    #
    ################################

    if verbose:
        print("Image preprocessing.")

    image_size = (224, 224)
    image = ants.resample_image(image, image_size, use_voxels=True, interp_type=0)
    image_array = (image.numpy() - image.min()) / (image.max() - image.min())

    # use imagenet mean,std for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    number_of_channels = 3

    batchX = np.zeros((*image_size, number_of_channels))
    for c in range(number_of_channels):
        batchX[:,:,c] = (image_array - imagenet_mean[c]) / (imagenet_std[c])

    # swap color axis because
    # numpy image: H x W x D x C
    # torch image: C x H x W x D

    batchX = batchX.transpose((2, 1, 0))
    batchX = np.expand_dims(batchX, 0)

    if verbose:
        print("Prediction.")
    batchY = model((torch.from_numpy(batchX)).float())

    disease_df = pd.DataFrame(batchY.detach().numpy(), columns = disease_categories)

    return disease_df

