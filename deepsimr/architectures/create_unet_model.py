import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class create_unet_model_2d(nn.Module):
    """
    2-D implementation of the U-net deep learning architecture.

    Creates a Pytorch model of the U-net deep learning architecture for image
    segmentation and regression based on our ANTsPyNet Keras implementation:

        https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/architectures/create_unet_model.py

    The architectural variants are (or should be) mostly identical to the
    ANTsXNet u-net Keras implementation.  However, some differences exist:
      * using nnUnetActivationStyle as the instance normalization layer
        implementations are different between Keras and PyTorch.
      * the weight_decay parameter used in the Keras implementation for
        l2 regularization is not implemented  in PyTorch.  Rather it is
        typically added onto the loss during optimization as explained
        here:

        https://discuss.pytorch.org/t/pytorch-equivalent-for-kernel-regulariser-in-tensorflow/142909

    Arguments
    ---------
    input_channel_size : integer
        Used for specifying the input tensor shape.

    number_of_outputs : integer
        Meaning depends on the mode.  For `classification` this is the number of
        segmentation labels.  For `regression` this is the number of outputs.

    number_of_layers : integer
        number of encoding/decoding layers.

    number_of_filters_at_base_layer : integer
        number of filters at the beginning and end of the `U`.  Doubles at each
        descending/ascending layer.

    number_of_filters : tuple
        tuple explicitly setting the number of filters at each layer.  One can
        either set this or number_of_layers and  number_of_filters_at_base_layer.
        Default = None.

    convolution_kernel_size : tuple of length 2
        Defines the kernel size during the encoding.

    deconvolution_kernel_size : tuple of length 2
        Defines the kernel size during the decoding.

    pool_size : tuple of length 2
        Defines the region for each pooling layer.

    strides : tuple of length 2
        Strides for the convolutional layers.

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    mode :  string
        `classification`, `regression`, or `sigmoid`.  Default = `classification`.

    additional_options : string or tuple of strings
        specific configuration add-ons/tweaks:
            * "attentionGating" -- attention-unet variant in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "nnUnetActivationStyle" -- U-net activation explained in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "initialConvolutionalKernelSize[X]" -- Set the first two convolutional layer kernel sizes to X.

    Returns
    -------
    PyTorch model
        A 2-D PyTorch model defining the U-net network.

    Example
    -------
    >>> model = deepsimr.create_unet_model_2d(input_channel_size=3)
    >>> torchinfo.summary(model, input_size=(1, 3, 128, 128))
    """

    def __init__(self, input_channel_size,
                       number_of_outputs=2,
                       number_of_layers=4,
                       number_of_filters_at_base_layer=32,
                       number_of_filters=None,
                       convolution_kernel_size=(3, 3),
                       deconvolution_kernel_size=(2, 2),
                       pool_size=(2, 2),
                       strides=(2, 2),
                       dropout_rate=0.0,
                       mode='classification',
                       additional_options=None
                      ):

        super(create_unet_model_2d, self).__init__()

        def nn_unet_activation_2d(number_of_features):
            x = nn.Sequential(nn.InstanceNorm2d(number_of_features, affine=True), nn.LeakyReLU(0.01))
            return x

        class attention_gate_2d(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.x_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(1, 1),
                                         stride=(1, 1),
                                         padding="valid")
                self.g_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding="valid")
                self.f_conv = nn.Sequential(nn.ReLU(),
                                            nn.Conv2d(in_channels=out_channels,
                                                      out_channels=1,
                                                      kernel_size=(1, 1),
                                                      stride=(1, 1),
                                                      padding="valid"),
                                            nn.Sigmoid())
            def forward(self, x, g):
                x_theta = self.x_conv(x)
                g_phi = self.g_conv(g)
                f = torch.add(x_theta, g_phi)
                f_psi = self.f_conv(f)
                attention = torch.multiply(x, f_psi)
                return attention

        initial_convolution_kernel_size = convolution_kernel_size
        add_attention_gating = False
        nn_unet_activation_style = False

        if additional_options is not None:

            if "attentionGating" in additional_options:
                add_attention_gating = True

            if "nnUnetActivationStyle" in additional_options:
                nn_unet_activation_style = True

            option = [o for o in additional_options if o.startswith('initialConvolutionKernelSize')]
            if not not option:
                initial_convolution_kernel_size = option[0].replace("initialConvolutionKernelSize", "")
                initial_convolution_kernel_size = initial_convolution_kernel_size.replace("[", "")
                initial_convolution_kernel_size = int(initial_convolution_kernel_size.replace("]", ""))

        # Specify the number of filters

        number_of_layers = number_of_layers

        if number_of_filters is not None:
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = list()
            for i in range(number_of_layers):
                number_of_filters.append(number_of_filters_at_base_layer * 2**i)

        # Encoding path

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)

        self.encoding_convolution_layers = nn.ModuleList()
        for i in range(number_of_layers):

            conv1 = None
            if i == 0:
                conv1 = nn.Conv2d(in_channels=input_channel_size,
                                  out_channels=number_of_filters[i],
                                  kernel_size=initial_convolution_kernel_size,
                                  padding='same')
            else:
                conv1 = nn.Conv2d(in_channels=number_of_filters[i-1],
                                  out_channels=number_of_filters[i],
                                  kernel_size=convolution_kernel_size,
                                  padding='same')

            conv2 = None
            if i == 0:
                conv2 = nn.Conv2d(in_channels=number_of_filters[i],
                                  out_channels=number_of_filters[i],
                                  kernel_size=initial_convolution_kernel_size,
                                  padding='same')
            else:
                conv2 = nn.Conv2d(in_channels=number_of_filters[i],
                                  out_channels=number_of_filters[i],
                                  kernel_size=convolution_kernel_size,
                                  padding='same')

            if nn_unet_activation_style:
                if dropout_rate > 0.0:
                   self.encoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_2d(number_of_filters[i]),
                                     nn.Dropout(dropout_rate),
                                     conv2, nn_unet_activation_2d(number_of_filters[i])))
                else:
                   self.encoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_2d(number_of_filters[i]),
                                     conv2, nn_unet_activation_2d(number_of_filters[i])))
            else:
                if dropout_rate > 0.0:
                    self.encoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2,
                                      nn.Dropout(dropout_rate),
                                      nn.ReLU()))
                else:
                    self.encoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()))

        # Decoding path

        self.upsample = nn.Upsample(scale_factor=pool_size)

        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()
        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose2d(in_channels=number_of_filters[number_of_layers-i],
                                        out_channels=number_of_filters[number_of_layers-i-1],
                                        kernel_size=deconvolution_kernel_size,
                                        padding=1)
            if nn_unet_activation_style:
                self.decoding_convolution_transpose_layers.append(
                    nn.Sequential(deconv, nn_unet_activation_2d(number_of_filters[number_of_layers-i-1])))
            else:
                self.decoding_convolution_transpose_layers.append(deconv)

            if add_attention_gating:
                self.decoding_attention_gating_layers.append(
                     attention_gate_2d(number_of_filters[number_of_layers-i-1],
                                       number_of_filters[number_of_layers-i-1] // 4))

            conv1 = nn.Conv2d(in_channels=number_of_filters[number_of_layers-i],
                              out_channels=number_of_filters[number_of_layers-i-1],
                              kernel_size=convolution_kernel_size,
                              padding="same")
            conv2 = nn.Conv2d(in_channels=number_of_filters[number_of_layers-i-1],
                              out_channels=number_of_filters[number_of_layers-i-1],
                              kernel_size=convolution_kernel_size,
                              padding="same")

            if nn_unet_activation_style:
                if dropout_rate > 0.0:
                   self.decoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_2d(number_of_filters[number_of_layers-i-1]),
                                     nn.Dropout(dropout_rate),
                                     conv2, nn_unet_activation_2d(number_of_filters[number_of_layers-i-1])))
                else:
                   self.decoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_2d(number_of_filters[number_of_layers-i-1]),
                                     conv2, nn_unet_activation_2d(number_of_filters[number_of_layers-i-1])))
            else:
                if dropout_rate > 0.0:
                    self.decoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2,
                                      nn.Dropout(dropout_rate),
                                      nn.ReLU()))
                else:
                    self.decoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()))

        conv = nn.Conv2d(in_channels=number_of_filters[0],
                         out_channels=number_of_outputs,
                         kernel_size=1,
                         padding='same')

        if mode == 'sigmoid':
            self.output = nn.Sequential(conv, nn.Sigmoid())
        elif mode == 'classification':
            self.output = nn.Sequential(conv, nn.Softmax(dim=1))
        elif mode == 'regression':
            self.output = nn.Sequential(conv, nn.Linear())
        else:
            raise ValueError('mode must be either `classification`, `regression` or `sigmoid`.')

    def forward(self, x):

        # Encoding path

        number_of_layers = len(self.encoding_convolution_layers)

        encoding_path = x
        encoding_tensor_layers = list()
        for i in range(number_of_layers):
            encoding_path = self.encoding_convolution_layers[i](encoding_path)
            encoding_tensor_layers.append(encoding_path)
            if i < number_of_layers - 1:
                encoding_path = self.pool(encoding_path)

        # Decoding path
        decoding_path = encoding_tensor_layers[number_of_layers-1]
        for i in range(1, number_of_layers):
            input_size = np.array(decoding_path.size()[2:])
            decoding_path = self.decoding_convolution_transpose_layers[i-1](decoding_path)

            # This code is necessary because "padding=same" doesn't
            # exist for conv transpose layers in PyTorch.
            size_difference = input_size - np.array(decoding_path.size()[2:])
            padding = list()
            padding.append(size_difference[0] // 2)
            padding.append(size_difference[0] - padding[-1])
            padding.append(size_difference[1] // 2)
            padding.append(size_difference[1] - padding[-1])
            decoding_path = F.pad(decoding_path, padding, "constant", 0)

            decoding_path = self.upsample(decoding_path)
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i-1](decoding_path, encoding_tensor_layers[number_of_layers-i-1])
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, encoding_tensor_layers[number_of_layers-i-1]], 1)
            decoding_path = self.decoding_convolution_layers[i-1](decoding_path)

        output = self.output(decoding_path)

        return output


class create_unet_model_3d(nn.Module):
    """
    3-D implementation of the U-net deep learning architecture.

    Creates a Pytorch model of the U-net deep learning architecture for image
    segmentation and regression based on our ANTsPyNet Keras implementation:

        https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/architectures/create_unet_model.py

    The architectural variants are (or should be) mostly identical to the
    ANTsXNet u-net Keras implementation.  However, some differences can be induced
    by:
      * using nnUnetActivationStyle as the instance normalization layer
        implementations are different between Keras and PyTorch.
      * the weight_decay parameter used in the Keras implementation for
        l2 regularization is not implemented  in PyTorch.  Rather it is
        typically added onto the loss during optimization as explained
        here:

        https://discuss.pytorch.org/t/pytorch-equivalent-for-kernel-regulariser-in-tensorflow/142909

    Arguments
    ---------
    input_channel_size : integer
        Used for specifying the input tensor shape.

    number_of_outputs : integer
        Meaning depends on the mode.  For `classification` this is the number of
        segmentation labels.  For `regression` this is the number of outputs.

    number_of_layers : integer
        number of encoding/decoding layers.

    number_of_filters_at_base_layer : integer
        number of filters at the beginning and end of the `U`.  Doubles at each
        descending/ascending layer.

    number_of_filters : tuple
        tuple explicitly setting the number of filters at each layer.  One can
        either set this or number_of_layers and  number_of_filters_at_base_layer.
        Default = None.

    convolution_kernel_size : tuple of length 3
        Defines the kernel size during the encoding.

    deconvolution_kernel_size : tuple of length 3
        Defines the kernel size during the decoding.

    pool_size : tuple of length 3
        Defines the region for each pooling layer.

    strides : tuple of length 3
        Strides for the convolutional layers.

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    mode :  string
        `classification`, `regression`, or `sigmoid`.  Default = `classification`.

    additional_options : string or tuple of strings
        specific configuration add-ons/tweaks:
            * "attentionGating" -- attention-unet variant in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "nnUnetActivationStyle" -- U-net activation explained in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "initialConvolutionalKernelSize[X]" -- Set the first two convolutional layer kernel sizes to X.

    Returns
    -------
    PyTorch model
        A 3-D PyTorch model defining the U-net network.

    Example
    -------
    >>> model = deepsimr.create_unet_model_3d(input_channel_size=3)
    >>> torchinfo.summary(model, input_size=(1, 3, 128, 128, 128))
    """

    def __init__(self, input_channel_size,
                       number_of_outputs=2,
                       number_of_layers=4,
                       number_of_filters_at_base_layer=32,
                       number_of_filters=None,
                       convolution_kernel_size=(3, 3, 3),
                       deconvolution_kernel_size=(2, 2, 2),
                       pool_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       dropout_rate=0.5,
                       mode='classification',
                       additional_options=None
                      ):

        super(create_unet_model_3d, self).__init__()

        def nn_unet_activation_3d(number_of_features):
            x = nn.Sequential(nn.InstanceNorm3d(number_of_features, affine=True), nn.LeakyReLU(0.01))
            return x

        class attention_gate_3d(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.x_conv = nn.Conv3d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(1, 1, 1),
                                         stride=(1, 1, 1),
                                         padding="valid")
                self.g_conv = nn.Conv3d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1, 1),
                                        stride=(1, 1, 1),
                                        padding="valid")
                self.f_conv = nn.Sequential(nn.ReLU(),
                                            nn.Conv3d(in_channels=out_channels,
                                                      out_channels=1,
                                                      kernel_size=(1, 1, 1),
                                                      stride=(1, 1, 1),
                                                      padding="valid"),
                                            nn.Sigmoid())
            def forward(self, x, g):
                x_theta = self.x_conv(x)
                g_phi = self.g_conv(g)
                f = torch.add(x_theta, g_phi)
                f_psi = self.f_conv(f)
                attention = torch.multiply(x, f_psi)
                return attention

        initial_convolution_kernel_size = convolution_kernel_size
        add_attention_gating = False
        nn_unet_activation_style = False

        if additional_options is not None:

            if "attentionGating" in additional_options:
                add_attention_gating = True

            if "nnUnetActivationStyle" in additional_options:
                nn_unet_activation_style = True

            option = [o for o in additional_options if o.startswith('initialConvolutionKernelSize')]
            if not not option:
                initial_convolution_kernel_size = option[0].replace("initialConvolutionKernelSize", "")
                initial_convolution_kernel_size = initial_convolution_kernel_size.replace("[", "")
                initial_convolution_kernel_size = int(initial_convolution_kernel_size.replace("]", ""))

        # Specify the number of filters

        number_of_layers = number_of_layers

        if number_of_filters is not None:
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = list()
            for i in range(number_of_layers):
                number_of_filters.append(number_of_filters_at_base_layer * 2**i)

        # Encoding path

        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=strides)

        self.encoding_convolution_layers = nn.ModuleList()
        for i in range(number_of_layers):

            conv1 = None
            if i == 0:
                conv1 = nn.Conv3d(in_channels=input_channel_size,
                                  out_channels=number_of_filters[i],
                                  kernel_size=initial_convolution_kernel_size,
                                  padding='same')
            else:
                conv1 = nn.Conv3d(in_channels=number_of_filters[i-1],
                                  out_channels=number_of_filters[i],
                                  kernel_size=convolution_kernel_size,
                                  padding='same')

            conv2 = None
            if i == 0:
                conv2 = nn.Conv3d(in_channels=number_of_filters[i],
                                  out_channels=number_of_filters[i],
                                  kernel_size=initial_convolution_kernel_size,
                                  padding='same')
            else:
                conv2 = nn.Conv3d(in_channels=number_of_filters[i],
                                  out_channels=number_of_filters[i],
                                  kernel_size=convolution_kernel_size,
                                  padding='same')

            if nn_unet_activation_style:
                if dropout_rate > 0.0:
                   self.encoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_3d(number_of_filters[i]),
                                     nn.Dropout(dropout_rate),
                                     conv2, nn_unet_activation_3d(number_of_filters[i])))
                else:
                   self.encoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_3d(number_of_filters[i]),
                                     conv2, nn_unet_activation_3d(number_of_filters[i])))
            else:
                if dropout_rate > 0.0:
                    self.encoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2,
                                      nn.Dropout(dropout_rate),
                                      nn.ReLU()))
                else:
                    self.encoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()))

        # Decoding path

        self.upsample = nn.Upsample(scale_factor=pool_size)

        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()
        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose3d(in_channels=number_of_filters[number_of_layers-i],
                                        out_channels=number_of_filters[number_of_layers-i-1],
                                        kernel_size=deconvolution_kernel_size,
                                        padding=1)
            if nn_unet_activation_style:
                self.decoding_convolution_transpose_layers.append(
                    nn.Sequential(deconv, nn_unet_activation_3d(number_of_filters[number_of_layers-i-1])))
            else:
                self.decoding_convolution_transpose_layers.append(deconv)

            if add_attention_gating:
                self.decoding_attention_gating_layers.append(
                     attention_gate_3d(number_of_filters[number_of_layers-i-1],
                                       number_of_filters[number_of_layers-i-1] // 4))

            conv1 = nn.Conv3d(in_channels=number_of_filters[number_of_layers-i],
                              out_channels=number_of_filters[number_of_layers-i-1],
                              kernel_size=convolution_kernel_size,
                              padding="same")
            conv2 = nn.Conv3d(in_channels=number_of_filters[number_of_layers-i-1],
                              out_channels=number_of_filters[number_of_layers-i-1],
                              kernel_size=convolution_kernel_size,
                              padding="same")

            if nn_unet_activation_style:
                if dropout_rate > 0.0:
                   self.decoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_3d(number_of_filters[number_of_layers-i-1]),
                                     nn.Dropout(dropout_rate),
                                     conv2, nn_unet_activation_3d(number_of_filters[number_of_layers-i-1])))
                else:
                   self.decoding_convolution_layers.append(
                       nn.Sequential(conv1, nn_unet_activation_3d(number_of_filters[number_of_layers-i-1]),
                                     conv2, nn_unet_activation_3d(number_of_filters[number_of_layers-i-1])))
            else:
                if dropout_rate > 0.0:
                    self.decoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2,
                                      nn.Dropout(dropout_rate),
                                      nn.ReLU()))
                else:
                    self.decoding_convolution_layers.append(
                        nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()))

        conv = nn.Conv3d(in_channels=number_of_filters[0],
                         out_channels=number_of_outputs,
                         kernel_size=1,
                         padding='same')

        if mode == 'sigmoid':
            self.output = nn.Sequential(conv, nn.Sigmoid())
        elif mode == 'classification':
            self.output = nn.Sequential(conv, nn.Softmax(dim=1))
        elif mode == 'regression':
            self.output = nn.Sequential(conv, nn.Linear())
        else:
            raise ValueError('mode must be either `classification`, `regression` or `sigmoid`.')

    def forward(self, x):

        # Encoding path

        number_of_layers = len(self.encoding_convolution_layers)

        encoding_path = x
        encoding_tensor_layers = list()
        for i in range(number_of_layers):
            encoding_path = self.encoding_convolution_layers[i](encoding_path)
            encoding_tensor_layers.append(encoding_path)
            if i < number_of_layers - 1:
                encoding_path = self.pool(encoding_path)

        # Decoding path
        decoding_path = encoding_tensor_layers[number_of_layers-1]
        for i in range(1, number_of_layers):
            input_size = np.array(decoding_path.size()[2:])
            decoding_path = self.decoding_convolution_transpose_layers[i-1](decoding_path)

            # This code is necessary because "padding=same" doesn't
            # exist for conv transpose layers in PyTorch.
            size_difference = input_size - np.array(decoding_path.size()[2:])
            padding = list()
            padding.append(size_difference[0] // 2)
            padding.append(size_difference[0] - padding[-1])
            padding.append(size_difference[1] // 2)
            padding.append(size_difference[1] - padding[-1])
            padding.append(size_difference[2] // 2)
            padding.append(size_difference[2] - padding[-1])
            decoding_path = F.pad(decoding_path, padding, "constant", 0)

            decoding_path = self.upsample(decoding_path)
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i-1](decoding_path, encoding_tensor_layers[number_of_layers-i-1])
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, encoding_tensor_layers[number_of_layers-i-1]], 1)
            decoding_path = self.decoding_convolution_layers[i-1](decoding_path)

        output = self.output(decoding_path)

        return output

