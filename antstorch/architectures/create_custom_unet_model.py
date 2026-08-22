#!/usr/bin/env python3
"""
PyTorch ports of a subset of antspynet.architectures.create_custom_unet_model:

    * create_sysu_media_unet_model_2d          (white_matter_hyperintensity_segmentation, claustrum_segmentation)
    * create_sysu_media_unet_model_3d          (white_matter_hyperintensity_segmentation)
    * create_hypermapp3r_unet_model_3d         (white_matter_hyperintensity_segmentation)
    * create_shiva_unet_model_3d               (white_matter_hyperintensity_segmentation)
    * create_hippmapp3r_unet_model_3d          (hippmapp3r_segmentation, added 2026-08-22)
    * create_hypothalamus_unet_model_3d        (hypothalamus_segmentation, added 2026-08-22)

Ported layer for layer, matching the Keras originals, so that weights
converted via tools/convert_antspynet_weights_to_antstorch.py map cleanly
onto these state_dicts.
"""
import torch
import torch.nn as nn

from .create_unet_model import _Conv2dSame, _Conv3dSame, _center_pad_crop, _align_leading_3d


# ---------------------------------------------------------------------------
# sysu_media (2017 MICCAI WMH challenge winner) -- 2D / 3D
# ---------------------------------------------------------------------------

class create_sysu_media_unet_model_2d(nn.Module):
    """
    2-D sysu_media U-net (MICCAI 2017 WMH challenge).

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (1 for flair-only, 2 for t1/flair).

    anatomy : string
        "wmh" or "claustrum".  Determines default filter sizes and whether
        the first encoding block uses a 5x5 kernel.

    Example
    -------
    >>> model = create_sysu_media_unet_model_2d(1)
    """
    def __init__(self, input_channel_size, anatomy="wmh"):
        super().__init__()

        if anatomy == "wmh":
            number_of_filters = (64, 96, 128, 256, 512)
        elif anatomy == "claustrum":
            number_of_filters = (32, 64, 96, 128, 256)
        else:
            raise ValueError("anatomy must be 'wmh' or 'claustrum'.")
        self.number_of_filters = number_of_filters

        self.encoding_layers = nn.ModuleList()
        in_ch = input_channel_size
        for i in range(len(number_of_filters)):
            k1, k2 = 3, 3
            if i == 0 and anatomy == "wmh":
                k1, k2 = 5, 5
            elif i == 3:
                k1, k2 = 3, 4
            block = nn.Sequential(
                _Conv2dSame(in_ch, number_of_filters[i], kernel_size=k1, stride=1, bias=True),
                nn.ReLU(),
                _Conv2dSame(number_of_filters[i], number_of_filters[i], kernel_size=k2, stride=1, bias=True),
                nn.ReLU(),
            )
            self.encoding_layers.append(block)
            in_ch = number_of_filters[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.decoding_layers = nn.ModuleList()
        for i in range(len(number_of_filters) - 2, -1, -1):
            concat_ch = number_of_filters[i + 1] if i == len(number_of_filters) - 2 else number_of_filters[i + 1]
            in_ch_decode = number_of_filters[i + 1] + number_of_filters[i]
            block = nn.Sequential(
                _Conv2dSame(in_ch_decode, number_of_filters[i], kernel_size=3, stride=1, bias=True),
                nn.ReLU(),
                _Conv2dSame(number_of_filters[i], number_of_filters[i], kernel_size=3, stride=1, bias=True),
                nn.ReLU(),
            )
            self.decoding_layers.append(block)

        self.output_conv = _Conv2dSame(number_of_filters[0], 1, kernel_size=1, stride=1, bias=True)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        L = len(self.encoding_layers)
        skips = []
        enc = x
        for i in range(L):
            enc = self.encoding_layers[i](enc)
            skips.append(enc)
            if i < L - 1:
                enc = self.pool(enc)

        outputs = skips[-1]
        for idx, i in enumerate(range(L - 2, -1, -1)):
            upsampled = self.upsample(outputs)
            skip = _center_pad_crop(skips[i], upsampled.shape[-2:])
            outputs = torch.cat([upsampled, skip], dim=1)
            outputs = self.decoding_layers[idx](outputs)

        outputs = _center_pad_crop(outputs, x.shape[-2:])
        return self.output_act(self.output_conv(outputs))


class create_sysu_media_unet_model_3d(nn.Module):
    """
    3-D variation of the sysu_media U-net architecture.

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels.

    number_of_filters : tuple, optional
        Overrides the default per-anatomy filter counts.

    anatomy : string
        "wmh" or "claustrum".

    Example
    -------
    >>> model = create_sysu_media_unet_model_3d(1)
    """
    def __init__(self, input_channel_size, number_of_filters=None, anatomy="wmh"):
        super().__init__()

        if number_of_filters is None:
            if anatomy == "wmh":
                number_of_filters = (64, 96, 128, 256, 512)
            elif anatomy == "claustrum":
                number_of_filters = (32, 64, 96, 128, 256)
            else:
                raise ValueError("anatomy must be 'wmh' or 'claustrum'.")
        self.number_of_filters = tuple(number_of_filters)

        self.encoding_layers = nn.ModuleList()
        in_ch = input_channel_size
        for i in range(len(self.number_of_filters)):
            k1, k2 = 3, 3
            if i == 0 and anatomy == "wmh":
                k1, k2 = 5, 5
            elif i == 3:
                k1, k2 = 3, 4
            block = nn.Sequential(
                _Conv3dSame(in_ch, self.number_of_filters[i], kernel_size=k1, stride=1, bias=True),
                nn.ReLU(),
                _Conv3dSame(self.number_of_filters[i], self.number_of_filters[i], kernel_size=k2, stride=1, bias=True),
                nn.ReLU(),
            )
            self.encoding_layers.append(block)
            in_ch = self.number_of_filters[i]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.decoding_layers = nn.ModuleList()
        for i in range(len(self.number_of_filters) - 2, -1, -1):
            in_ch_decode = self.number_of_filters[i + 1] + self.number_of_filters[i]
            block = nn.Sequential(
                _Conv3dSame(in_ch_decode, self.number_of_filters[i], kernel_size=3, stride=1, bias=True),
                nn.ReLU(),
                _Conv3dSame(self.number_of_filters[i], self.number_of_filters[i], kernel_size=3, stride=1, bias=True),
                nn.ReLU(),
            )
            self.decoding_layers.append(block)

        self.output_conv = _Conv3dSame(self.number_of_filters[0], 1, kernel_size=1, stride=1, bias=True)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        L = len(self.encoding_layers)
        skips = []
        enc = x
        for i in range(L):
            enc = self.encoding_layers[i](enc)
            skips.append(enc)
            if i < L - 1:
                enc = self.pool(enc)

        outputs = skips[-1]
        for idx, i in enumerate(range(L - 2, -1, -1)):
            upsampled = self.upsample(outputs)
            skip = _center_pad_crop(skips[i], upsampled.shape[-3:])
            outputs = torch.cat([upsampled, skip], dim=1)
            outputs = self.decoding_layers[idx](outputs)

        outputs = _center_pad_crop(outputs, x.shape[-3:])
        return self.output_act(self.output_conv(outputs))


# ---------------------------------------------------------------------------
# HyperMapp3r
# ---------------------------------------------------------------------------

class _ConvBBlock3D(nn.Module):
    """Conv3d(same) -> InstanceNorm3d(per-channel affine) -> LeakyReLU(0.3)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = _Conv3dSame(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm = nn.InstanceNorm3d(out_ch, eps=1e-3, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _HyperMapp3rResidualBlock(nn.Module):
    def __init__(self, number_of_filters):
        super().__init__()
        self.conv1 = _ConvBBlock3D(number_of_filters, number_of_filters)
        self.dropout = nn.Dropout3d(p=0.3)
        self.conv2 = _ConvBBlock3D(number_of_filters, number_of_filters)

    def forward(self, x):
        return self.conv2(self.dropout(self.conv1(x)))


class _HyperMapp3rFeatureBlock(nn.Module):
    def __init__(self, in_ch, number_of_filters):
        super().__init__()
        self.conv1 = _ConvBBlock3D(in_ch, number_of_filters, kernel_size=3)
        self.conv2 = _ConvBBlock3D(number_of_filters, number_of_filters, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class _HyperMapp3rUpsampleBlock(nn.Module):
    def __init__(self, in_ch, number_of_filters):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _ConvBBlock3D(in_ch, number_of_filters)

    def forward(self, x):
        return self.conv(self.upsample(x))


class create_hypermapp3r_unet_model_3d(nn.Module):
    """
    3-D HyperMapp3r U-net (white matter hyperintensity segmentation).

    https://pubmed.ncbi.nlm.nih.gov/35088930/

    Note: the model uses per-channel SpatialDropout3D (nn.Dropout3d here) as
    part of a Monte Carlo dropout inference scheme.  Dropout only zeroes
    activations while its submodules are in `.train()` mode -- callers doing
    Monte Carlo inference should call `.train()` on just the Dropout3d
    submodules (leaving everything else in eval-equivalent behavior, since
    InstanceNorm3d here has track_running_stats=False and is mode-independent).

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (2 for t1 + flair).

    Example
    -------
    >>> model = create_hypermapp3r_unet_model_3d(2)
    """
    def __init__(self, input_channel_size):
        super().__init__()

        number_of_filters_at_base_layer = 8
        number_of_layers = 4
        self.number_of_layers = number_of_layers

        filters = [number_of_filters_at_base_layer * 2 ** i for i in range(number_of_layers)]
        self.filters = filters

        self.encoding_conv = nn.ModuleList()
        self.encoding_residual = nn.ModuleList()
        in_ch = input_channel_size
        for i in range(number_of_layers):
            stride = 1 if i == 0 else 2
            self.encoding_conv.append(_ConvBBlock3D(in_ch, filters[i], kernel_size=3, stride=stride))
            self.encoding_residual.append(_HyperMapp3rResidualBlock(filters[i]))
            in_ch = filters[i]

        # Decoding path.
        self.up0 = _HyperMapp3rUpsampleBlock(filters[3], filters[2])
        self.feature64 = _HyperMapp3rFeatureBlock(filters[2] + filters[2], filters[2])
        self.up1 = _HyperMapp3rUpsampleBlock(filters[2], filters[1])
        self.back64 = _Conv3dSame(filters[2], 1, kernel_size=1, stride=1, bias=True)

        self.feature32 = _HyperMapp3rFeatureBlock(filters[1] + filters[1], filters[1])
        self.up2 = _HyperMapp3rUpsampleBlock(filters[1], filters[0])
        self.back32 = _Conv3dSame(filters[1], 1, kernel_size=1, stride=1, bias=True)

        self.final_conv1 = _ConvBBlock3D(filters[0] + filters[0], filters[0], kernel_size=3)
        self.final_conv2 = _ConvBBlock3D(filters[0], filters[0], kernel_size=1)
        self.final_conv3 = _Conv3dSame(filters[0], 1, kernel_size=1, stride=1, bias=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        add = None
        encoding = []
        for i in range(self.number_of_layers):
            conv = self.encoding_conv[i](x if i == 0 else add)
            residual = self.encoding_residual[i](conv)
            add = conv + residual
            encoding.append(add)

        outputs = encoding[3]
        outputs = self.up0(outputs)

        skip2 = _align_leading_3d(encoding[2], outputs.shape[-3:])
        feature64 = self.feature64(torch.cat([skip2, outputs], dim=1))
        outputs = self.up1(feature64)
        back64 = self.back64(feature64)
        back64 = self.upsample(back64)

        skip1 = _align_leading_3d(encoding[1], outputs.shape[-3:])
        feature32 = self.feature32(torch.cat([skip1, outputs], dim=1))
        outputs = self.up2(feature32)
        back32 = self.back32(feature32)
        back32 = _align_leading_3d(back32, back64.shape[-3:]) if back32.shape[-3:] != back64.shape[-3:] else back32
        back32 = back64 + back32
        back32 = self.upsample(back32)

        skip0 = _align_leading_3d(encoding[0], outputs.shape[-3:])
        outputs = self.final_conv1(torch.cat([skip0, outputs], dim=1))
        outputs = self.final_conv2(outputs)
        outputs = self.final_conv3(outputs)
        back32 = _align_leading_3d(back32, outputs.shape[-3:]) if back32.shape[-3:] != outputs.shape[-3:] else back32
        outputs = back32 + outputs

        return self.output_act(outputs)


# ---------------------------------------------------------------------------
# SHIVA (PVS / WMH)
# ---------------------------------------------------------------------------

class create_shiva_unet_model_3d(nn.Module):
    """
    3-D SHIVA U-net architecture used for PVS and WMH segmentation.

    * PVS:  https://pubmed.ncbi.nlm.nih.gov/34262443/
    * WMH:  https://pubmed.ncbi.nlm.nih.gov/38050769/

    Arguments
    ---------
    number_of_modalities : integer
        Number of input channels.

    number_of_outputs : integer
        Number of outputs per voxel.  Determines the final activation
        (1 = sigmoid, >1 = softmax).

    Example
    -------
    >>> model = create_shiva_unet_model_3d(number_of_modalities=1)
    """
    def __init__(self, number_of_modalities=1, number_of_outputs=1):
        super().__init__()

        number_of_filters = (10, 18, 32, 58, 104, 187, 337)
        self.number_of_filters = number_of_filters
        self.number_of_outputs = number_of_outputs

        def conv_bn_swish(in_ch, out_ch):
            return nn.Sequential(
                _Conv3dSame(in_ch, out_ch, kernel_size=3, stride=1, bias=False),
                nn.BatchNorm3d(out_ch, eps=1e-3),
                nn.SiLU(),
            )

        self.encoding_layers = nn.ModuleList()
        self.encoding_dropouts = nn.ModuleList()
        in_ch = number_of_modalities
        for i in range(len(number_of_filters)):
            block = nn.Sequential(
                conv_bn_swish(in_ch, number_of_filters[i]),
                conv_bn_swish(number_of_filters[i], number_of_filters[i]),
            )
            self.encoding_layers.append(block)
            dropout_rate = 0.05 if i == 0 else 0.5
            self.encoding_dropouts.append(nn.Dropout3d(p=dropout_rate))
            in_ch = number_of_filters[i]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.decoding_conv1 = nn.ModuleList()
        self.decoding_conv2 = nn.ModuleList()
        self.decoding_dropouts = nn.ModuleList()
        up_ch = number_of_filters[-1]
        for i in range(len(number_of_filters) - 1, -1, -1):
            concat_ch = up_ch + number_of_filters[i]
            self.decoding_conv1.append(conv_bn_swish(concat_ch, concat_ch))
            self.decoding_conv2.append(conv_bn_swish(concat_ch, number_of_filters[i]))
            self.decoding_dropouts.append(nn.Dropout3d(p=0.5))
            up_ch = number_of_filters[i]

        self.final_conv1 = conv_bn_swish(number_of_filters[0], 10)
        self.final_conv2 = conv_bn_swish(10, 10)
        self.output_conv = _Conv3dSame(10, number_of_outputs, kernel_size=1, stride=1, bias=True)
        self.output_act = nn.Sigmoid() if number_of_outputs == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        L = len(self.encoding_layers)
        skips = []
        outputs = x
        for i in range(L):
            outputs = self.encoding_layers[i](outputs)
            skips.append(outputs)
            outputs = self.pool(outputs)
            outputs = self.encoding_dropouts[i](outputs)

        for idx, i in enumerate(range(L - 1, -1, -1)):
            upsampled = self.upsample(outputs)
            skip = skips[i]
            if i > 0 and skip.shape[-3:] != upsampled.shape[-3:]:
                upsampled = _center_pad_crop(upsampled, skip.shape[-3:])
                outputs = torch.cat([upsampled, skip], dim=1)
            else:
                outputs = torch.cat([upsampled, skip], dim=1)
            outputs = self.decoding_conv1[idx](outputs)
            outputs = self.decoding_conv2[idx](outputs)
            outputs = self.decoding_dropouts[idx](outputs)

        outputs = self.final_conv1(outputs)
        outputs = self.final_conv2(outputs)
        outputs = self.output_conv(outputs)
        return self.output_act(outputs)


# ---------------------------------------------------------------------------
# HippMapp3r (hippocampus segmentation, 2-stage)
# ---------------------------------------------------------------------------

class create_hippmapp3r_unet_model_3d(nn.Module):
    """
    3-D HippMapp3r U-net (hippocampal segmentation, 2-stage network).

    https://www.ncbi.nlm.nih.gov/pubmed/31609046
    https://github.com/mgoubran/HippMapp3r

    Shares its encoding/decoding building blocks (residual-augmented conv
    blocks, nearest-neighbor upsample blocks, feature blocks, dual side-output
    "back64"/"back32" heads combined additively before the final sigmoid)
    with create_hypermapp3r_unet_model_3d above -- ported layer for layer from
    the same Keras source family. The two differ in: number of encoding
    layers (variable here: 6 for the initial-stage network, 5 for the
    refine-stage network, vs. always 4 for hypermapp3r), the initial-stage
    network has one extra decoding stage the refine-stage network skips, and
    the initial-stage network's back64/back32/final-output 1x1 convs include
    InstanceNorm+LeakyReLU (matching Keras `convB_3d_layer`) while the
    refine-stage network's use plain Conv3d (matching Keras `Conv3D` with no
    normalization) -- this asymmetry is exactly what the original Keras
    source does (see `do_first_network` branches in
    antspynet.architectures.create_hippmapp3r_unet_model_3d).

    Note: like hypermapp3r, the refine-stage network relies on Monte Carlo
    SpatialDropout3D at inference time (30 iterations in
    antspynet.utilities.hippmapp3r_segmentation) -- callers doing Monte Carlo
    inference should call `.train()` on just the Dropout3d submodules (inside
    the residual blocks), leaving everything else in eval-equivalent behavior
    (InstanceNorm3d here has track_running_stats=False and is mode-independent).

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (1 -- single T1 channel).

    do_first_network : boolean
        True builds the initial-stage (coarse mask) network (6 encoding
        layers); False builds the refine-stage network (5 encoding layers).

    Example
    -------
    >>> model_initial = create_hippmapp3r_unet_model_3d(1, do_first_network=True)
    >>> model_refine = create_hippmapp3r_unet_model_3d(1, do_first_network=False)
    """
    def __init__(self, input_channel_size, do_first_network=True):
        super().__init__()
        self.do_first_network = do_first_network

        number_of_filters_at_base_layer = 16
        number_of_layers = 6 if do_first_network else 5
        self.number_of_layers = number_of_layers

        filters = [number_of_filters_at_base_layer * 2 ** i for i in range(number_of_layers)]
        self.filters = filters

        # Encoding path: same conv(stride)+residual+Add pattern as hypermapp3r.
        self.encoding_conv = nn.ModuleList()
        self.encoding_residual = nn.ModuleList()
        in_ch = input_channel_size
        for i in range(number_of_layers):
            stride = 1 if i == 0 else 2
            self.encoding_conv.append(_ConvBBlock3D(in_ch, filters[i], kernel_size=3, stride=stride))
            self.encoding_residual.append(_HyperMapp3rResidualBlock(filters[i]))
            in_ch = filters[i]

        # Decoding path.
        top_filters = filters[number_of_layers - 2]
        self.up_top = _HyperMapp3rUpsampleBlock(filters[number_of_layers - 1], top_filters)

        if do_first_network:
            self.feature_extra = _HyperMapp3rFeatureBlock(top_filters + filters[4], top_filters)
            next_filters = top_filters // 2
            self.up_extra = _HyperMapp3rUpsampleBlock(top_filters, next_filters)
            stage3_in = next_filters
        else:
            stage3_in = top_filters

        self.feature3 = _HyperMapp3rFeatureBlock(stage3_in + filters[3], stage3_in)
        f2 = stage3_in // 2
        self.up3 = _HyperMapp3rUpsampleBlock(stage3_in, f2)

        self.feature64 = _HyperMapp3rFeatureBlock(f2 + filters[2], f2)
        f1 = f2 // 2
        self.up64 = _HyperMapp3rUpsampleBlock(f2, f1)

        self.feature32 = _HyperMapp3rFeatureBlock(f1 + filters[1], f1)
        f0 = f1 // 2
        self.up32 = _HyperMapp3rUpsampleBlock(f1, f0)

        # back64 / back32 / final-output 1x1 convs: convB (norm+act) for the
        # initial-stage network, plain Conv3d for the refine-stage network.
        if do_first_network:
            self.back64_conv = _ConvBBlock3D(f2, 1, kernel_size=1)
            self.back32_conv = _ConvBBlock3D(f1, 1, kernel_size=1)
            self.final_out_conv = _ConvBBlock3D(f0, 1, kernel_size=1)
        else:
            self.back64_conv = _Conv3dSame(f2, 1, kernel_size=1, stride=1, bias=True)
            self.back32_conv = _Conv3dSame(f1, 1, kernel_size=1, stride=1, bias=True)
            self.final_out_conv = _Conv3dSame(f0, 1, kernel_size=1, stride=1, bias=True)

        self.final_conv1 = _ConvBBlock3D(f0 + filters[0], f0, kernel_size=3)
        self.final_conv2 = _ConvBBlock3D(f0, f0, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        add = None
        encoding = []
        for i in range(self.number_of_layers):
            conv = self.encoding_conv[i](x if i == 0 else add)
            residual = self.encoding_residual[i](conv)
            add = conv + residual
            encoding.append(add)

        outputs = encoding[self.number_of_layers - 1]
        outputs = self.up_top(outputs)

        if self.do_first_network:
            skip4 = _align_leading_3d(encoding[4], outputs.shape[-3:])
            outputs = self.feature_extra(torch.cat([skip4, outputs], dim=1))
            outputs = self.up_extra(outputs)

        skip3 = _align_leading_3d(encoding[3], outputs.shape[-3:])
        outputs = self.feature3(torch.cat([skip3, outputs], dim=1))
        outputs = self.up3(outputs)

        skip2 = _align_leading_3d(encoding[2], outputs.shape[-3:])
        feature64 = self.feature64(torch.cat([skip2, outputs], dim=1))
        outputs = self.up64(feature64)
        back64 = self.back64_conv(feature64)
        back64 = self.upsample(back64)

        skip1 = _align_leading_3d(encoding[1], outputs.shape[-3:])
        feature32 = self.feature32(torch.cat([skip1, outputs], dim=1))
        outputs = self.up32(feature32)
        back32 = self.back32_conv(feature32)
        back32 = _align_leading_3d(back32, back64.shape[-3:]) if back32.shape[-3:] != back64.shape[-3:] else back32
        back32 = back64 + back32
        back32 = self.upsample(back32)

        skip0 = _align_leading_3d(encoding[0], outputs.shape[-3:])
        outputs = self.final_conv1(torch.cat([skip0, outputs], dim=1))
        outputs = self.final_conv2(outputs)
        outputs = self.final_out_conv(outputs)
        back32 = _align_leading_3d(back32, outputs.shape[-3:]) if back32.shape[-3:] != outputs.shape[-3:] else back32
        outputs = back32 + outputs

        return self.output_act(outputs)


# ---------------------------------------------------------------------------
# Hypothalamus segmentation
# ---------------------------------------------------------------------------

class create_hypothalamus_unet_model_3d(nn.Module):
    """
    3-D U-net for hypothalamus (and subunit) segmentation.

    https://pubmed.ncbi.nlm.nih.gov/32853816/
    https://github.com/BBillot/hypothalamus_seg

    A simple 3-level U-net: 24 -> 48 -> 96 filters, ELU activations,
    BatchNorm3d after each encoding/decoding block, softmax (11-class)
    output. Ported layer for layer from the Keras source, including one
    subtlety: each encoder level's skip connection is taken from the
    convolution output *before* BatchNorm is applied (BatchNorm updates the
    variable used for pooling/bottleneck only) -- matching
    antspynet.architectures.create_hypothalamus_unet_model_3d exactly.

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (1 -- single T1 channel; the original
        Keras constructor hardcodes this itself via `Input(shape=(*size, 1))`).

    number_of_outputs : integer
        Number of segmentation classes (11: background + 10 subunit labels).

    Example
    -------
    >>> model = create_hypothalamus_unet_model_3d()
    """
    def __init__(self, input_channel_size=1, number_of_outputs=11):
        super().__init__()

        number_of_layers = 3
        self.number_of_layers = number_of_layers
        base = 24
        filters = [base * 2 ** i for i in range(number_of_layers)]
        self.filters = filters

        def conv_elu(in_ch, out_ch):
            return nn.Sequential(
                _Conv3dSame(in_ch, out_ch, kernel_size=3, stride=1, bias=True),
                nn.ELU(),
            )

        self.encoding_conv1 = nn.ModuleList()
        self.encoding_conv2 = nn.ModuleList()
        self.encoding_bn = nn.ModuleList()
        in_ch = input_channel_size
        for i in range(number_of_layers):
            self.encoding_conv1.append(conv_elu(in_ch, filters[i]))
            self.encoding_conv2.append(conv_elu(filters[i], filters[i]))
            self.encoding_bn.append(nn.BatchNorm3d(filters[i]))
            in_ch = filters[i]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.decoding_conv1 = nn.ModuleList()
        self.decoding_conv2 = nn.ModuleList()
        self.decoding_bn = nn.ModuleList()
        for i in range(1, number_of_layers):
            idx = number_of_layers - i - 1
            concat_ch = filters[idx] + filters[number_of_layers - i]
            self.decoding_conv1.append(conv_elu(concat_ch, filters[idx]))
            self.decoding_conv2.append(conv_elu(filters[idx], filters[idx]))
            self.decoding_bn.append(nn.BatchNorm3d(filters[idx]))

        self.output_conv = _Conv3dSame(filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
        self.output_act = nn.Softmax(dim=1)

    def forward(self, x):
        skips = []
        pool = None
        outputs = None
        for i in range(self.number_of_layers):
            conv = self.encoding_conv1[i](x if i == 0 else pool)
            conv = self.encoding_conv2[i](conv)
            skips.append(conv)  # pre-BatchNorm skip -- matches Keras exactly
            conv = self.encoding_bn[i](conv)
            if i < self.number_of_layers - 1:
                pool = self.pool(conv)
            else:
                outputs = conv

        for idx, i in enumerate(range(1, self.number_of_layers)):
            deconv = self.upsample(outputs)
            skip = _align_leading_3d(skips[self.number_of_layers - i - 1], deconv.shape[-3:])
            outputs = torch.cat([skip, deconv], dim=1)
            outputs = self.decoding_conv1[idx](outputs)
            outputs = self.decoding_conv2[idx](outputs)
            outputs = self.decoding_bn[idx](outputs)

        return self.output_act(self.output_conv(outputs))
