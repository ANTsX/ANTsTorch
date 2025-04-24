import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

class create_resnet_model_2d(nn.Module):
    """
    2-D implementation of the ResNet deep learning architecture.

    Creates a Pytorch model of the ResNet deep learning architecture for image
    classification based on our ANTsPyNet Keras implementation:

        https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/architectures/create_resnet_model.py

    The architectural variants are (or should be) mostly identical to the
    ANTsXNet Keras implementation.  However, some differences exist:
      * The initialization in the dense layers of the squeeze-and-excite-block is different
        as explained here:

        https://stats.stackexchange.com/questions/484062/he-normal-keras-is-truncated-when-kaiming-normal-pytorch-is-not

      * Batch normalization is different between Keras and PyTorch as explained here:

        https://stackoverflow.com/questions/60079783/difference-between-keras-batchnormalization-and-pytorchs-batchnorm2d


    Arguments
    ---------
    input_channel_size : integer
        Used for specifying the input tensor shape.

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    residual_block_schedule : tuple
        A tuple defining the how many residual blocks repeats for each layer.

    lowest_resolution : integer
        Number of filters at the initial layer.

    cardinality : integer
        perform ResNet (cardinality = 1) or ResNeX (cardinality is not 1 but,
        instead, powers of 2---try '32').

    squeeze_and_excite : boolean
        add the squeeze-and-excite block variant.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    PyTorch model
        A 2-D PyTorch model defining the ResNet network.

    Example
    -------
    >>> model = antstorch.create_resnet_model_2d(input_channel_size=3)
    >>> torchinfo.summary(model, input_size=(1, 3, 128, 128))
    """

    def __init__(self, input_channel_size,
                       number_of_classification_labels=1000,
                       layers=(1, 2, 3, 4),
                       residual_block_schedule=(3, 4, 6, 3),
                       lowest_resolution=64,
                       cardinality=1,
                       squeeze_and_excite=False,
                       mode='classification'
                      ):

        super(create_resnet_model_2d, self).__init__()

        class Conv2dSame(torch.nn.Conv2d):

            def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
                return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

            def forward(self, x):
                ih, iw = x.size()[-2:]

                pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
                pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

                if pad_h > 0 or pad_w > 0:
                    x = F.pad(
                        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                    )
                return F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class MaxPool2dSame(torch.nn.MaxPool2d):

            def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
                return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

            def forward(self, x):
                ih, iw = x.size()[-2:]

                pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
                pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

                if pad_h > 0 or pad_w > 0:
                    x = F.pad(
                        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                    )
                return F.max_pool2d(
                    x,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.ceil_mode,
                    self.return_indices
                )


        def add_common_layers(number_of_features):
            x = nn.Sequential(nn.BatchNorm2d(number_of_features), nn.LeakyReLU(0.3))
            return x

        def grouped_convolution_layer_2d(in_channels,
                                         out_channels,
                                         strides):
            if out_channels % cardinality != 0:
                raise ValueError('number_of_filters `%` cardinality != 0')

            conv = Conv2dSame(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=strides,
                              groups=cardinality)
            return conv

        class squeeze_and_excite_block_2d(nn.Module):
            def __init__(self, in_channels,
                               out_channels,
                               ratio=16):
                super(squeeze_and_excite_block_2d, self).__init__()

                self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.block_shape = (1, out_channels, 1, 1)
                self.dense = nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=out_channels // ratio, bias=False),
                    nn.ReLU(),
                    nn.Linear(in_features=out_channels // ratio, out_features=out_channels, bias=False),
                    nn.Sigmoid())

            def forward(self, x):
                x_se = self.global_pool(x)
                x_se = x_se.view(x_se.size(0), -1)
                x_se = self.dense(x_se)
                x_se = torch.reshape(x_se, self.block_shape)
                return torch.mul(x, x_se)

        class residual_block_2d(nn.Module):
            def __init__(self, in_channels,
                               number_of_filters_in,
                               number_of_filters_out,
                               strides = (1, 1),
                               project_shortcut=False,
                               squeeze_and_excite_local=False):
                super(residual_block_2d, self).__init__()

                self.conv1 = nn.Sequential(
                                Conv2dSame(in_channels=in_channels,
                                           out_channels=number_of_filters_in,
                                           kernel_size=(1, 1),
                                           stride=(1, 1)),
                                add_common_layers(number_of_filters_in))

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                self.conv2 = nn.Sequential(
                    grouped_convolution_layer_2d(in_channels=number_of_filters_in,
                                                 out_channels=number_of_filters_in,
                                                 strides=strides),
                    add_common_layers(number_of_filters_in))

                self.conv3 = nn.Sequential(
                                Conv2dSame(in_channels=number_of_filters_in,
                                           out_channels=number_of_filters_out,
                                           kernel_size=(1, 1),
                                           stride=(1, 1)),
                                nn.BatchNorm2d(number_of_filters_out))

                self.shortcut = None
                if project_shortcut or strides != (1, 1):
                    self.shortcut = nn.Sequential(
                                       Conv2dSame(in_channels=in_channels,
                                                  out_channels=number_of_filters_out,
                                                  kernel_size=(1, 1),
                                                  stride=strides),
                                       nn.BatchNorm2d(number_of_filters_out))

                self.squeeze_and_excite_layer = None
                if squeeze_and_excite_local:
                    self.squeeze_and_excite_layer = squeeze_and_excite_block_2d(in_channels=number_of_filters_out,
                                                                                out_channels=number_of_filters_out)
                self.leaky_relu = nn.LeakyReLU(0.3)

            def forward(self, x):
                shortcut = x
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                if self.shortcut is not None:
                    shortcut = self.shortcut(shortcut)
                if self.squeeze_and_excite_layer is not None:
                    x = self.squeeze_and_excite_layer(x)
                x = torch.add(shortcut, x)
                x = self.leaky_relu(x)
                return x

        n_filters = lowest_resolution

        self.init_conv = nn.Sequential(Conv2dSame(in_channels=input_channel_size,
                                                  out_channels=n_filters,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2)),
                                       add_common_layers(n_filters))
        self.max_pool = MaxPool2dSame(kernel_size=(3, 3),
                                      stride=(2, 2),
                                      dilation=(1, 1))

        self.model_residual_layers = nn.ModuleList()
        for i in range(len(layers)):
            n_filters_in = lowest_resolution * 2**layers[i]
            n_filters_out = 2 * n_filters_in

            for j in range(residual_block_schedule[i]):
                project_shortcut = False
                if i == 0 and j == 0:
                    project_shortcut = True

                if i > 0 and j == 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

                self.model_residual_layers.append(
                    residual_block_2d(in_channels=n_filters,
                                      number_of_filters_in=n_filters_in,
                                      number_of_filters_out=n_filters_out,
                                      strides=strides,
                                      project_shortcut=project_shortcut,
                                      squeeze_and_excite_local=squeeze_and_excite))
                n_filters = n_filters_out

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.dense = None
        if mode == 'classification':
            self.dense = nn.Sequential(
                nn.Linear(in_features=n_filters_out,
                          out_features=number_of_classification_labels),
                nn.Softmax(dim=1))
        elif mode == 'regression':
            self.dense = nn.Linear(in_features=n_filters_out,
                          out_features=number_of_classification_labels)
        else:
            raise ValueError('mode must be either `classification` or `regression`.')

    def forward(self, x):
        x = self.init_conv(x)
        x = self.max_pool(x)
        for i in range(len(self.model_residual_layers)):
            x = self.model_residual_layers[i](x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return(x)

class create_resnet_model_3d(nn.Module):
    """
    3-D implementation of the ResNet deep learning architecture.

    Creates a Pytorch model of the ResNet deep learning architecture for image
    classification based on our ANTsPyNet Keras implementation:

        https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/architectures/create_resnet_model.py

    The architectural variants are (or should be) mostly identical to the
    ANTsXNet Keras implementation.  However, some differences exist:
      * The initialization in the dense layers of the squeeze-and-excite-block is different
        as explained here:

        https://stats.stackexchange.com/questions/484062/he-normal-keras-is-truncated-when-kaiming-normal-pytorch-is-not

      * Batch normalization is different between Keras and PyTorch as explained here:

        https://stackoverflow.com/questions/60079783/difference-between-keras-batchnormalization-and-pytorchs-batchnorm2d


    Arguments
    ---------
    input_channel_size : integer
        Used for specifying the input tensor shape.

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    residual_block_schedule : tuple
        A tuple defining the how many residual blocks repeats for each layer.

    lowest_resolution : integer
        Number of filters at the initial layer.

    cardinality : integer
        perform ResNet (cardinality = 1) or ResNeX (cardinality is not 1 but,
        instead, powers of 2---try '32').

    squeeze_and_excite : boolean
        add the squeeze-and-excite block variant.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    PyTorch model
        A 3-D PyTorch model defining the ResNet network.

    Example
    -------
    >>> model = antstorch.create_resnet_model_3d(input_channel_size=3)
    >>> torchinfo.summary(model, input_size=(1, 3, 128, 128))
    """

    def __init__(self, input_channel_size,
                       number_of_classification_labels=1000,
                       layers=(1, 2, 3, 4),
                       residual_block_schedule=(3, 4, 6, 3),
                       lowest_resolution=64,
                       cardinality=1,
                       squeeze_and_excite=False,
                       mode='classification'
                      ):

        super(create_resnet_model_3d, self).__init__()

        class Conv3dSame(torch.nn.Conv3d):

            def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
                return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

            def forward(self, x):
                ih, iw, id = x.size()[-3:]

                pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
                pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
                pad_d = self.calc_same_pad(i=id, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    x = F.pad(
                        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2]
                    )
                return F.conv3d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class MaxPool3dSame(torch.nn.MaxPool3d):

            def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
                return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

            def forward(self, x):
                ih, iw, id = x.size()[-3:]

                pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
                pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
                pad_d = self.calc_same_pad(i=id, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    x = F.pad(
                        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2]
                    )
                return F.max_pool3d(
                    x,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.ceil_mode,
                    self.return_indices
                )


        def add_common_layers(number_of_features):
            x = nn.Sequential(nn.BatchNorm3d(number_of_features), nn.LeakyReLU(0.3))
            return x

        def grouped_convolution_layer_3d(in_channels,
                                         out_channels,
                                         strides):
            if out_channels % cardinality != 0:
                raise ValueError('number_of_filters `%` cardinality != 0')

            conv = Conv3dSame(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3, 3),
                              stride=strides,
                              groups=cardinality)
            return conv

        class squeeze_and_excite_block_3d(nn.Module):
            def __init__(self, in_channels,
                               out_channels,
                               ratio=16):
                super(squeeze_and_excite_block_3d, self).__init__()

                self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
                self.block_shape = (1, out_channels, 1, 1, 1)
                self.dense = nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=out_channels // ratio, bias=False),
                    nn.ReLU(),
                    nn.Linear(in_features=out_channels // ratio, out_features=out_channels, bias=False),
                    nn.Sigmoid())

            def forward(self, x):
                x_se = self.global_pool(x)
                x_se = x_se.view(x_se.size(0), -1)
                x_se = self.dense(x_se)
                x_se = torch.reshape(x_se, self.block_shape)
                return torch.mul(x, x_se)

        class residual_block_3d(nn.Module):
            def __init__(self, in_channels,
                               number_of_filters_in,
                               number_of_filters_out,
                               strides = (1, 1, 1),
                               project_shortcut=False,
                               squeeze_and_excite_local=False):
                super(residual_block_3d, self).__init__()

                self.conv1 = nn.Sequential(
                                Conv3dSame(in_channels=in_channels,
                                           out_channels=number_of_filters_in,
                                           kernel_size=(1, 1, 1),
                                           stride=(1, 1, 1)),
                                add_common_layers(number_of_filters_in))

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                self.conv2 = nn.Sequential(
                    grouped_convolution_layer_3d(in_channels=number_of_filters_in,
                                                 out_channels=number_of_filters_in,
                                                 strides=strides),
                    add_common_layers(number_of_filters_in))

                self.conv3 = nn.Sequential(
                                Conv3dSame(in_channels=number_of_filters_in,
                                           out_channels=number_of_filters_out,
                                           kernel_size=(1, 1, 1),
                                           stride=(1, 1, 1)),
                                nn.BatchNorm3d(number_of_filters_out))

                self.shortcut = None
                if project_shortcut or strides != (1, 1, 1):
                    self.shortcut = nn.Sequential(
                                       Conv3dSame(in_channels=in_channels,
                                                  out_channels=number_of_filters_out,
                                                  kernel_size=(1, 1, 1),
                                                  stride=strides),
                                       nn.BatchNorm3d(number_of_filters_out))

                self.squeeze_and_excite_layer = None
                if squeeze_and_excite_local:
                    self.squeeze_and_excite_layer = squeeze_and_excite_block_3d(in_channels=number_of_filters_out,
                                                                                out_channels=number_of_filters_out)
                self.leaky_relu = nn.LeakyReLU(0.3)

            def forward(self, x):
                shortcut = x
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                if self.shortcut is not None:
                    shortcut = self.shortcut(shortcut)
                if self.squeeze_and_excite_layer is not None:
                    x = self.squeeze_and_excite_layer(x)
                x = torch.add(shortcut, x)
                x = self.leaky_relu(x)
                return x

        n_filters = lowest_resolution

        self.init_conv = nn.Sequential(Conv3dSame(in_channels=input_channel_size,
                                                  out_channels=n_filters,
                                                  kernel_size=(7, 7, 7),
                                                  stride=(2, 2, 2)),
                                       add_common_layers(n_filters))
        self.max_pool = MaxPool3dSame(kernel_size=(3, 3, 3),
                                      stride=(2, 2, 2),
                                      dilation=(1, 1, 1))

        self.model_residual_layers = nn.ModuleList()
        for i in range(len(layers)):
            n_filters_in = lowest_resolution * 2**layers[i]
            n_filters_out = 2 * n_filters_in

            for j in range(residual_block_schedule[i]):
                project_shortcut = False
                if i == 0 and j == 0:
                    project_shortcut = True

                if i > 0 and j == 0:
                    strides = (2, 2, 2)
                else:
                    strides = (1, 1, 1)

                self.model_residual_layers.append(
                    residual_block_3d(in_channels=n_filters,
                                      number_of_filters_in=n_filters_in,
                                      number_of_filters_out=n_filters_out,
                                      strides=strides,
                                      project_shortcut=project_shortcut,
                                      squeeze_and_excite_local=squeeze_and_excite))
                n_filters = n_filters_out

        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.dense = None
        if mode == 'classification':
            self.dense = nn.Sequential(
                nn.Linear(in_features=n_filters_out,
                          out_features=number_of_classification_labels),
                nn.Softmax(dim=1))
        elif mode == 'regression':
            self.dense = nn.Linear(in_features=n_filters_out,
                          out_features=number_of_classification_labels)
        else:
            raise ValueError('mode must be either `classification` or `regression`.')

    def forward(self, x):
        x = self.init_conv(x)
        x = self.max_pool(x)
        for i in range(len(self.model_residual_layers)):
            x = self.model_residual_layers[i](x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return(x)


