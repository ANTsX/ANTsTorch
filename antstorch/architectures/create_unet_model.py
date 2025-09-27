#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

def _tf_style_match(dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """
    Make `dec` match `skip` spatial size using TF-like alignment:
      • If smaller: pad on the RIGHT (max index) per spatial dim.
      • If larger: crop from the LEFT (min index) per spatial dim.
    No interpolation; preserves crisp boundaries and avoids centroid drift.

    Supports:
      - 2D: dec, skip shapes [N, C, H, W]
      - 3D: dec, skip shapes [N, C, D, H, W]
    """
    if dec.dim() not in (4, 5):
        raise ValueError(f"Expected 4D or 5D tensors, got dec.dim()={dec.dim()}")

    # Compute diffs per spatial dim (positive => need right pad; negative => need left crop)
    if dec.dim() == 4:
        # 2D: H, W
        dH = skip.size(2) - dec.size(2)
        dW = skip.size(3) - dec.size(3)

        # Right-pad (pad on max index side only)
        if dH > 0 or dW > 0:
            # F.pad order for 2D: (W_left, W_right, H_left, H_right)
            dec = F.pad(dec, (0, max(0, dW), 0, max(0, dH)))

        # Left-crop (crop from min index side only)
        h0 = max(dec.size(2) - skip.size(2), 0)
        w0 = max(dec.size(3) - skip.size(3), 0)
        if h0 or w0:
            dec = dec[..., h0:, w0:]

    else:
        # 3D: D, H, W
        dD = skip.size(2) - dec.size(2)
        dH = skip.size(3) - dec.size(3)
        dW = skip.size(4) - dec.size(4)

        # Right-pad
        if dD > 0 or dH > 0 or dW > 0:
            # F.pad order for 3D: (W_left, W_right, H_left, H_right, D_left, D_right)
            dec = F.pad(dec, (0, max(0, dW), 0, max(0, dH), 0, max(0, dD)))

        # Left-crop
        d0 = max(dec.size(2) - skip.size(2), 0)
        h0 = max(dec.size(3) - skip.size(3), 0)
        w0 = max(dec.size(4) - skip.size(4), 0)
        if d0 or h0 or w0:
            dec = dec[..., d0:, h0:, w0:]

    # Optional debug check:
    # assert dec.shape[2:] == skip.shape[2:], f"Mismatch after tf_style_match: {dec.shape} vs {skip.shape}"
    return dec

def _same_pad_1d_tf(L_in, stride, kernel, dilation=1):
    # TF SAME: out = ceil(L/stride)
    out = math.ceil(L_in / stride)
    needed = max(0, (out - 1) * stride + (kernel - 1) * dilation + 1 - L_in)
    # TF puts the extra 1 (if odd) on the RIGHT (max index) side
    pad_left = needed // 2
    pad_right = needed - pad_left
    return pad_left, pad_right

def _same_pad_nd_tf(size_in, stride, kernel, dilation):
    # size_in: (D,H,W) or (H,W)
    # stride/kernel/dilation: tuples of same length
    pads = []
    # Compute in W,H,(D) order for F.pad
    for L, s, k, d in reversed(list(zip(size_in, stride, kernel, dilation))):
        l, r = _same_pad_1d_tf(L, s, k, d)
        pads.extend([l, r])  # will reverse once more below
    # F.pad order for 3D is (W_left,W_right,H_left,H_right,D_left,D_right)
    return tuple(pads)

class _Conv2dSameTF(nn.Module):
    """
    TF/Keras-like SAME padding with parameter names compatible with nn.Conv3d:
    state_dict keys are '<name>.weight' and '<name>.bias'.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True, groups=1):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)*2
        if isinstance(stride, int):      stride = (stride,)*2
        if isinstance(dilation, int):    dilation = (dilation,)*2

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Register parameters with the SAME NAMES as nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None

        # Initialize like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_ch * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        pads = _same_pad_nd_tf(x.shape[-3:], self.stride, self.kernel_size, self.dilation)
        x = F.pad(x, pads)
        return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                        padding=0, dilation=self.dilation, groups=self.groups)

class _Conv3dSame(nn.Module):
    """
    TF/Keras-like SAME padding with parameter names compatible with nn.Conv3d:
    state_dict keys are '<name>.weight' and '<name>.bias'.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True, groups=1):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)*3
        if isinstance(stride, int):      stride = (stride,)*3
        if isinstance(dilation, int):    dilation = (dilation,)*3

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Register parameters with the SAME NAMES as nn.Conv3d
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None

        # Initialize like nn.Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_ch * kernel_size[0] * kernel_size[1] * kernel_size[2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        pads = _same_pad_nd_tf(x.shape[-3:], self.stride, self.kernel_size, self.dilation)
        x = F.pad(x, pads)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride,
                        padding=0, dilation=self.dilation, groups=self.groups)

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
                conv1 = _Conv2dSame(input_channel_size, number_of_filters[i],
                   kernel_size=initial_convolution_kernel_size, stride=1, bias=True)
            else:
                conv1 = _Conv2dSame(number_of_filters[i - 1], number_of_filters[i],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

            conv2 = _Conv2dSame(number_of_filters[i], number_of_filters[i],
                   kernel_size=(initial_convolution_kernel_size if i == 0 else convolution_kernel_size),
                   stride=1, bias=True)       

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
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="bilinear", align_corners=False)
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose2d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                stride=strides,
                padding=0,
                output_padding=0,
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

            conv1 = _Conv2dSame(number_of_filters[number_of_layers - i],
                   number_of_filters[number_of_layers - i - 1],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

            conv2 = _Conv2dSame(number_of_filters[number_of_layers - i - 1],
                   number_of_filters[number_of_layers - i - 1],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

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

        head_conv = _Conv2dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
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
            skip = encoding_tensor_layers[number_of_layers - i - 1]
            decoding_path = self.decoding_convolution_transpose_layers[i - 1](decoding_path)

            # Make sure spatial dims match the *skip* before concatenation
            if decoding_path.shape[2:] != skip.shape[2:]:
                decoding_path = _tf_style_match(decoding_path, skip)

            # Optional attention gate
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i - 1](skip, decoding_path)
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, skip], 1)

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
        dropout_rate=0.0,
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
                conv1 = _Conv3dSame(input_channel_size, number_of_filters[i],
                   kernel_size=initial_convolution_kernel_size, stride=1, bias=True)
            else:
                conv1 = _Conv3dSame(number_of_filters[i - 1], number_of_filters[i],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

            conv2 = _Conv3dSame(number_of_filters[i], number_of_filters[i],
                   kernel_size=(initial_convolution_kernel_size if i == 0 else convolution_kernel_size),
                   stride=1, bias=True)       

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
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="trilinear", align_corners=False)
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.decoding_attention_gating_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = nn.ConvTranspose3d(
                in_channels=number_of_filters[number_of_layers - i],
                out_channels=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                stride=strides,
                padding=0,
                output_padding=0,
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

            conv1 = _Conv3dSame(number_of_filters[number_of_layers - i],
                   number_of_filters[number_of_layers - i - 1],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

            conv2 = _Conv3dSame(number_of_filters[number_of_layers - i - 1],
                   number_of_filters[number_of_layers - i - 1],
                   kernel_size=convolution_kernel_size, stride=1, bias=True)

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

        head_conv = _Conv3dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
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
            skip = encoding_tensor_layers[number_of_layers - i - 1]
            decoding_path = self.decoding_convolution_transpose_layers[i - 1](decoding_path)

            # Make sure spatial dims match the *skip* before concatenation
            if decoding_path.shape[2:] != skip.shape[2:]:
                decoding_path = _tf_style_match(decoding_path, skip)

            # Optional attention gate
            if len(self.decoding_attention_gating_layers) > 0:
                attention = self.decoding_attention_gating_layers[i - 1](skip, decoding_path)
                decoding_path = torch.cat([decoding_path, attention], 1)
            else:
                decoding_path = torch.cat([decoding_path, skip], 1)

            decoding_path = self.decoding_convolution_layers[i - 1](decoding_path)

        return self.output(decoding_path)
