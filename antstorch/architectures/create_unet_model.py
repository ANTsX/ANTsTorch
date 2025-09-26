#!/usr/bin/env python3
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

    Architectural notes vs Keras:
      * "nnUnetActivationStyle" uses InstanceNorm + LeakyReLU blocks to better mirror nnU-Net style.
      * Keras L2 weight decay is typically applied via optimizer weight_decay in PyTorch (not per-layer).

    Arguments
    ---------
    input_channel_size : int
        Number of input channels.

    number_of_outputs : int
        For `classification`, number of labels. For `regression`, number of outputs.

    number_of_layers : int
        Number of encoding/decoding layers.

    number_of_filters_at_base_layer : int
        Filters at the first (and last) U level; doubles each down/up level.

    number_of_filters : tuple[int, ...] or None
        Explicit per-level filters. If None, computed from base & layers.

    convolution_kernel_size : tuple[int, int]
    deconvolution_kernel_size : tuple[int, int]
    pool_size : tuple[int, int]
    strides : tuple[int, int]

    dropout_rate : float
    mode : {'classification','regression','sigmoid'}
    additional_options : str | tuple[str, ...]
        * 'attentionGating'
        * 'nnUnetActivationStyle'
        * 'initialConvolutionalKernelSize[X]'  (e.g., initialConvolutionalKernelSize[5])

    Returns
    -------
    nn.Module
    """

    def __init__(
        self,
        input_channel_size,
        number_of_outputs=2,
        number_of_layers=4,
        number_of_filters_at_base_layer=32,
        number_of_filters=None,
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=None,
    ):
        super(create_unet_model_2d, self).__init__()

        def nn_unet_activation_2d(n_feat: int) -> nn.Sequential:
            return nn.Sequential(nn.InstanceNorm2d(n_feat, affine=True), nn.LeakyReLU(0.01))

        class attention_gate_2d(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.x_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="valid")
                self.g_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="valid")
                self.f_conv = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding="valid"),
                    nn.Sigmoid(),
                )

            def forward(self, x, g):
                x_theta = self.x_conv(x)
                g_phi = self.g_conv(g)
                f = torch.add(x_theta, g_phi)
                f_psi = self.f_conv(f)
                return torch.multiply(x, f_psi)

        initial_convolution_kernel_size = convolution_kernel_size
        add_attention_gating = False
        nn_unet_activation_style = False

        if additional_options is not None:
            if "attentionGating" in additional_options:
                add_attention_gating = True
            if "nnUnetActivationStyle" in additional_options:
                nn_unet_activation_style = True

            option = [o for o in additional_options if str(o).startswith("initialConvolutionKernelSize")]
            if option:
                val = option[0].replace("initialConvolutionKernelSize", "").replace("[", "").replace("]", "")
                initial_convolution_kernel_size = (int(val), int(val))

        # Number of filters per level
        if number_of_filters is not None:
            number_of_filters = list(number_of_filters)
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = [number_of_filters_at_base_layer * 2**i for i in range(number_of_layers)]

        # Encoding
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)
        self.encoding_convolution_layers = nn.ModuleList()
        for i in range(number_of_layers):
            if i == 0:
                conv1 = nn.Conv2d(input_channel_size, number_of_filters[i], kernel_size=initial_convolution_kernel_size, padding="same")
            else:
                conv1 = nn.Conv2d(number_of_filters[i - 1], number_of_filters[i], kernel_size=convolution_kernel_size, padding="same")
            conv2 = nn.Conv2d(number_of_filters[i], number_of_filters[i], kernel_size=(initial_convolution_kernel_size if i == 0 else convolution_kernel_size), padding="same")

            if nn_unet_activation_style:
                block = [
                    conv1,
                    nn_unet_activation_2d(number_of_filters[i]),
                    *( [nn.Dropout(dropout_rate)] if dropout_rate > 0.0 else [] ),
                    conv2,
                    nn_unet_activation_2d(number_of_filters[i]),
                ]
                self.encoding_convolution_layers.append(nn.Sequential(*block))
            else:
                block = [conv1, nn.ReLU(), conv2]
                if dropout_rate > 0.0:
                    block.insert(2, nn.Dropout(dropout_rate))
                    block.append(nn.ReLU())
                else:
                    block.append(nn.ReLU())
                self.encoding_convolution_layers.append(nn.Sequential(*block))

        # Decoding
        self.upsample = nn.Upsample(scale_factor=pool_size)
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose2d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                padding=1,
            )
            if nn_unet_activation_style:
                self.decoding_convolution_transpose_layers.append(
                    nn.Sequential(deconv, nn_unet_activation_2d(number_of_filters[number_of_layers - i - 1]))
                )
            else:
                self.decoding_convolution_transpose_layers.append(deconv)

            if add_attention_gating:
                self.decoding_attention_gating_layers.append(
                    attention_gate_2d(
                        number_of_filters[number_of_layers - i - 1],
                        number_of_filters[number_of_layers - i - 1] // 4,
                    )
                )

            conv1 = nn.Conv2d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=convolution_kernel_size,
                padding="same",
            )
            conv2 = nn.Conv2d(
                in_channels=number_of_filters[number_of_layers - i - 1],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=convolution_kernel_size,
                padding="same",
            )

            if nn_unet_activation_style:
                block = [
                    conv1,
                    nn_unet_activation_2d(number_of_filters[number_of_layers - i - 1]),
                    *( [nn.Dropout(dropout_rate)] if dropout_rate > 0.0 else [] ),
                    conv2,
                    nn_unet_activation_2d(number_of_filters[number_of_layers - i - 1]),
                ]
                self.decoding_convolution_layers.append(nn.Sequential(*block))
            else:
                block = [conv1, nn.ReLU(), conv2]
                if dropout_rate > 0.0:
                    block.insert(2, nn.Dropout(dropout_rate))
                    block.append(nn.ReLU())
                else:
                    block.append(nn.ReLU())
                self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = nn.Conv2d(number_of_filters[0], number_of_outputs, kernel_size=1, padding="same")
        if mode == "sigmoid":
            self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification":
            self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression":
            # For regression we typically use a linear head without activation.
            self.output = nn.Sequential(head_conv)
        else:
            raise ValueError("mode must be either `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        number_of_layers = len(self.encoding_convolution_layers)

        # Encoding
        encoding_path = x
        encoding_tensor_layers = []
        for i in range(number_of_layers):
            encoding_path = self.encoding_convolution_layers[i](encoding_path)
            encoding_tensor_layers.append(encoding_path)
            if i < number_of_layers - 1:
                encoding_path = self.pool(encoding_path)

        # Decoding
        decoding_path = encoding_tensor_layers[number_of_layers - 1]
        for i in range(1, number_of_layers):
            input_size = np.array(decoding_path.size()[2:])
            decoding_path = self.decoding_convolution_transpose_layers[i - 1](decoding_path)

            # Adjust because ConvTranspose "same" isn't available.
            size_difference = input_size - np.array(decoding_path.size()[2:])
            padding = [
                size_difference[0] // 2,
                size_difference[0] - (size_difference[0] // 2),
                size_difference[1] // 2,
                size_difference[1] - (size_difference[1] // 2),
            ]
            decoding_path = F.pad(decoding_path, padding, "constant", 0)

            decoding_path = self.upsample(decoding_path)
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i - 1](
                    decoding_path, encoding_tensor_layers[number_of_layers - i - 1]
                )
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, encoding_tensor_layers[number_of_layers - i - 1]], 1)
            decoding_path = self.decoding_convolution_layers[i - 1](decoding_path)

        return self.output(decoding_path)


class create_unet_model_3d(nn.Module):
    """
    3-D implementation of the U-net deep learning architecture.
    See the 2D version docstring for argument descriptions.
    """

    def __init__(
        self,
        input_channel_size,
        number_of_outputs=2,
        number_of_layers=4,
        number_of_filters_at_base_layer=32,
        number_of_filters=None,
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.5,
        mode="classification",
        additional_options=None,
    ):
        super(create_unet_model_3d, self).__init__()

        def nn_unet_activation_3d(n_feat: int) -> nn.Sequential:
            return nn.Sequential(nn.InstanceNorm3d(n_feat, affine=True), nn.LeakyReLU(0.01))

        class attention_gate_3d(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.x_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding="valid")
                self.g_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding="valid")
                self.f_conv = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding="valid"),
                    nn.Sigmoid(),
                )

            def forward(self, x, g):
                x_theta = self.x_conv(x)
                g_phi = self.g_conv(g)
                f = torch.add(x_theta, g_phi)
                f_psi = self.f_conv(f)
                return torch.multiply(x, f_psi)

        initial_convolution_kernel_size = convolution_kernel_size
        add_attention_gating = False
        nn_unet_activation_style = False

        if additional_options is not None:
            if "attentionGating" in additional_options:
                add_attention_gating = True
            if "nnUnetActivationStyle" in additional_options:
                nn_unet_activation_style = True

            option = [o for o in additional_options if str(o).startswith("initialConvolutionKernelSize")]
            if option:
                val = option[0].replace("initialConvolutionKernelSize", "").replace("[", "").replace("]", "")
                initial_convolution_kernel_size = (int(val), int(val), int(val))

        # Number of filters per level
        if number_of_filters is not None:
            number_of_filters = list(number_of_filters)
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = [number_of_filters_at_base_layer * 2**i for i in range(number_of_layers)]

        # Encoding
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=strides)
        self.encoding_convolution_layers = nn.ModuleList()
        for i in range(number_of_layers):
            if i == 0:
                conv1 = nn.Conv3d(input_channel_size, number_of_filters[i], kernel_size=initial_convolution_kernel_size, padding="same")
            else:
                conv1 = nn.Conv3d(number_of_filters[i - 1], number_of_filters[i], kernel_size=convolution_kernel_size, padding="same")

            conv2 = nn.Conv3d(
                in_channels=number_of_filters[i],
                out_channels=number_of_filters[i],
                kernel_size=(initial_convolution_kernel_size if i == 0 else convolution_kernel_size),
                padding="same",
            )

            if nn_unet_activation_style:
                block = [
                    conv1,
                    nn_unet_activation_3d(number_of_filters[i]),
                    *( [nn.Dropout(dropout_rate)] if dropout_rate > 0.0 else [] ),
                    conv2,
                    nn_unet_activation_3d(number_of_filters[i]),
                ]
                self.encoding_convolution_layers.append(nn.Sequential(*block))
            else:
                block = [conv1, nn.ReLU(), conv2]
                if dropout_rate > 0.0:
                    block.insert(2, nn.Dropout(dropout_rate))
                    block.append(nn.ReLU())
                else:
                    block.append(nn.ReLU())
                self.encoding_convolution_layers.append(nn.Sequential(*block))

        # Decoding
        self.upsample = nn.Upsample(scale_factor=pool_size)
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose3d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                padding=1,
            )
            if nn_unet_activation_style:
                self.decoding_convolution_transpose_layers.append(
                    nn.Sequential(deconv, nn_unet_activation_3d(number_of_filters[number_of_layers - i - 1]))
                )
            else:
                self.decoding_convolution_transpose_layers.append(deconv)

            if add_attention_gating:
                self.decoding_attention_gating_layers.append(
                    attention_gate_3d(
                        number_of_filters[number_of_layers - i - 1],
                        number_of_filters[number_of_layers - i - 1] // 4,
                    )
                )

            conv1 = nn.Conv3d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=convolution_kernel_size,
                padding="same",
            )
            conv2 = nn.Conv3d(
                in_channels=number_of_filters[number_of_layers - i - 1],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=convolution_kernel_size,
                padding="same",
            )

            if nn_unet_activation_style:
                block = [
                    conv1,
                    nn_unet_activation_3d(number_of_filters[number_of_layers - i - 1]),
                    *( [nn.Dropout(dropout_rate)] if dropout_rate > 0.0 else [] ),
                    conv2,
                    nn_unet_activation_3d(number_of_filters[number_of_layers - i - 1]),
                ]
                self.decoding_convolution_layers.append(nn.Sequential(*block))
            else:
                block = [conv1, nn.ReLU(), conv2]
                if dropout_rate > 0.0:
                    block.insert(2, nn.Dropout(dropout_rate))
                    block.append(nn.ReLU())
                else:
                    block.append(nn.ReLU())
                self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = nn.Conv3d(number_of_filters[0], number_of_outputs, kernel_size=1, padding="same")
        if mode == "sigmoid":
            self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification":
            self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression":
            self.output = nn.Sequential(head_conv)
        else:
            raise ValueError("mode must be either `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        number_of_layers = len(self.encoding_convolution_layers)

        # Encoding
        encoding_path = x
        encoding_tensor_layers = []
        for i in range(number_of_layers):
            encoding_path = self.encoding_convolution_layers[i](encoding_path)
            encoding_tensor_layers.append(encoding_path)
            if i < number_of_layers - 1:
                encoding_path = self.pool(encoding_path)

        # Decoding
        decoding_path = encoding_tensor_layers[number_of_layers - 1]
        for i in range(1, number_of_layers):
            input_size = np.array(decoding_path.size()[2:])
            decoding_path = self.decoding_convolution_transpose_layers[i - 1](decoding_path)

            # Adjust because ConvTranspose "same" isn't available.
            size_difference = input_size - np.array(decoding_path.size()[2:])
            padding = [
                size_difference[0] // 2,
                size_difference[0] - (size_difference[0] // 2),
                size_difference[1] // 2,
                size_difference[1] - (size_difference[1] // 2),
                size_difference[2] // 2,
                size_difference[2] - (size_difference[2] // 2),
            ]
            decoding_path = F.pad(decoding_path, padding, "constant", 0)

            decoding_path = self.upsample(decoding_path)
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i - 1](
                    decoding_path, encoding_tensor_layers[number_of_layers - i - 1]
                )
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, encoding_tensor_layers[number_of_layers - i - 1]], 1)
            decoding_path = self.decoding_convolution_layers[i - 1](decoding_path)

        return self.output(decoding_path)
