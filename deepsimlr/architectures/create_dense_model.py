import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class create_dense_model(nn.Module):
    """
    Simple multilayer dense network.

    Arguments
    ---------
    input_vector size : integer
        Specifies the length of the input vector.

    number_of_filters_at_base_layer : integer
        number of filters at the initial dense layer.  This number is halved for
        each subsequent layer.

    number_of_layers : integer
        Number of dense layers defining the model.

    mode : string
        "regression" or "classification".

    number_of_classification_labels : integer
        Specifies output for "classification" networks.

    Returns
    -------
    PyTorch model
        A PyTorch model defining the network.

    Example
    -------
    >>> model = deepsimlr.create_dense_model(128)
    >>> torchinfo.summary(model)
    """

    def __init__(self, input_vector_size,
                       number_of_filters_at_base_layer=512,
                       number_of_layers=2,
                       mode='classification',
                       number_of_classification_labels=1000
                      ):

        super(create_dense_model, self).__init__()


        in_channels = input_vector_size
        out_channels = number_of_filters_at_base_layer

        self.dense_layers  = nn.ModuleList()
        for _ in range(number_of_layers):

            self.dense_layers.append(nn.Sequential(
                                         nn.Linear(in_features=in_channels,
                                                   out_features=out_channels),
                                         nn.LeakyReLU(0.2)))
            in_channels = out_channels
            out_channels = int(in_channels / 2)

        out_channels *= 2

        if mode == 'classification':
            self.dense_layers.append(nn.Sequential(
                                         nn.Linear(in_features=out_channels,
                                                   out_features=number_of_classification_labels),
                                         nn.Softmax(dim=1)))
        elif mode == 'regression':
            self.dense_layers.append(nn.Linear(in_features=out_channels,
                                               out_features=number_of_classification_labels))
        else:
            raise ValueError('mode must be either `classification` or `regression`.')

    def forward(self, x):

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)

        return(x)


