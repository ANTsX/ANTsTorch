#!/usr/bin/env python3
"""
PyTorch port of antspynet.architectures.create_deep_back_projection_network_model.

2-D implementation of the deep back-projection network (DBPN) for image
super resolution.  See:

    https://arxiv.org/abs/1803.02735

Ported to match, layer for layer, the Keras implementation in ANTsPyNet so
that weights converted via tools/convert_antspynet_weights_to_antstorch.py
map cleanly onto this state_dict.
"""
import torch
import torch.nn as nn

from .create_unet_model import _Conv2dSame


def _deconv_same_stride_params(kernel_size, stride):
    """
    Padding/output_padding for a ConvTranspose (any dimensionality) that
    reproduces Keras Conv*DTranspose(padding='same') behavior, i.e.
    output_size == input_size * stride, for arbitrary (fixed) kernel_size
    and stride (kernel_size >= stride).
    """
    pads, outpads = [], []
    for k, s in zip(kernel_size, stride):
        p = max(0, (k - s + 1) // 2)
        op = s - k + 2 * p
        op = max(0, min(op, s - 1)) if s > 1 else 0
        pads.append(p)
        outpads.append(op)
    return tuple(pads), tuple(outpads)


class _ConvTranspose2dSameStride(nn.Module):
    """Keras-'same'-padding ConvTranspose2d for an arbitrary (fixed) stride."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias=True):
        super().__init__()
        ks = (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
        st = (stride,) * 2 if isinstance(stride, int) else tuple(stride)
        pad, outpad = _deconv_same_stride_params(ks, st)
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, ks, stride=st, padding=pad,
                                          output_padding=outpad, bias=bias)

    def forward(self, x):
        return self.deconv(x)


class _DBPNUpBlock2D(nn.Module):
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv2dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _ConvTranspose2dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv2dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up1 = _ConvTranspose2dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        L = self.dense_act(self.dense(x))
        H0 = self.up0_act(self.up0(L))
        L0 = self.down0_act(self.down0(H0))
        E = L0 - L
        H1 = self.up1_act(self.up1(E))
        return H0 + H1


class _DBPNDownBlock2D(nn.Module):
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv2dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv2dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _ConvTranspose2dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down1 = _Conv2dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        H = self.dense_act(self.dense(x))
        L0 = self.down0_act(self.down0(H))
        H0 = self.up0_act(self.up0(L0))
        E = H0 - H
        L1 = self.down1_act(self.down1(E))
        return L0 + L1


class create_deep_back_projection_network_model_2d(nn.Module):
    """
    2-D deep back-projection network.

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (e.g. 3 for RGB).

    number_of_outputs : integer
        Number of outputs (e.g., 3 for RGB images).

    number_of_base_filters : integer
        Number of base filters.

    number_of_feature_filters : integer
        Number of feature filters.

    number_of_back_projection_stages : integer
        Number of up-down-projection stages (in addition to the initial up block).

    convolution_kernel_size : tuple of length 2
        Kernel size for the up/down-projection convolutions.  Dependent on the
        scale factor: 2x -> (6, 6), 4x -> (8, 8), 8x -> (12, 12).

    strides : tuple of length 2
        Strides for the up/down-projection convolutions (the super-resolution
        scale factor): 2x -> (2, 2), 4x -> (4, 4), 8x -> (8, 8).

    last_convolution : tuple of length 2
        Kernel size for the final convolutional layer.

    Example
    -------
    >>> model = create_deep_back_projection_network_model_2d(1)
    """
    def __init__(self, input_channel_size,
                 number_of_outputs=1,
                 number_of_base_filters=64,
                 number_of_feature_filters=256,
                 number_of_back_projection_stages=7,
                 convolution_kernel_size=(12, 12),
                 strides=(8, 8),
                 last_convolution=(3, 3)):
        super().__init__()
        self.number_of_back_projection_stages = number_of_back_projection_stages

        self.feature_extraction = _Conv2dSame(input_channel_size, number_of_feature_filters,
                                               kernel_size=3, stride=1, bias=True)
        self.feature_extraction_act = nn.PReLU(num_parameters=number_of_feature_filters, init=0.0)

        self.smash = _Conv2dSame(number_of_feature_filters, number_of_base_filters,
                                  kernel_size=1, stride=1, bias=True)
        self.smash_act = nn.PReLU(num_parameters=number_of_base_filters, init=0.0)

        base = number_of_base_filters

        # Pre-loop up block: input is the `base`-channel output of the smash layer.
        self.up_block_0 = _DBPNUpBlock2D(base, base, convolution_kernel_size, strides)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(number_of_back_projection_stages):
            # At the start of stage i, up_projection_blocks holds (i + 1) entries
            # (1 from the pre-loop block, plus one appended per prior stage).
            down_in_ch = (i + 1) * base
            self.down_blocks.append(_DBPNDownBlock2D(down_in_ch, base, convolution_kernel_size, strides))
            # After this stage's down-block is appended, down_projection_blocks
            # also holds (i + 1) entries.
            up_in_ch = (i + 1) * base
            self.up_blocks.append(_DBPNUpBlock2D(up_in_ch, base, convolution_kernel_size, strides))

        final_in_ch = (number_of_back_projection_stages + 1) * base
        self.output_conv = _Conv2dSame(final_in_ch, number_of_outputs,
                                        kernel_size=last_convolution, stride=1, bias=True)

    def forward(self, x):
        model = self.feature_extraction_act(self.feature_extraction(x))
        model = self.smash_act(self.smash(model))

        up_projection_blocks = [self.up_block_0(model)]
        down_projection_blocks = []

        model = up_projection_blocks[-1]
        for i in range(self.number_of_back_projection_stages):
            model = self.down_blocks[i](model)
            down_projection_blocks.append(model)
            model = torch.cat(down_projection_blocks, dim=1)

            model = self.up_blocks[i](model)
            up_projection_blocks.append(model)
            model = torch.cat(up_projection_blocks, dim=1)

        return self.output_conv(model)
