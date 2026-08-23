#!/usr/bin/env python3
"""
PyTorch port of antspynet.architectures.create_deep_back_projection_network_model.

2-D and 3-D implementations of the deep back-projection network (DBPN) for
image super resolution.  See:

    https://arxiv.org/abs/1803.02735

Ported to match, layer for layer, the Keras implementation in ANTsPyNet so
that weights converted via tools/convert_antspynet_weights_to_antstorch.py
map cleanly onto this state_dict.

The 3-D variant (added 2026-08-22 for antstorch.utilities.mri_super_resolution)
mirrors the 2-D class exactly, dimension for dimension -- see that class's
docstring for the general DBPN block structure, which is identical here.
"""
import torch
import torch.nn as nn

from .create_unet_model import _Conv2dSame, _Conv3dSame


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


# =============================================================================
# SIQ-style DBPN 3D (added 2026-08-22 for the actual `sig_smallshort_train_*`
# weights, https://github.com/stnava/siq).
#
# IMPORTANT: this is a DIFFERENT architecture from
# create_deep_back_projection_network_model_3d above, not just a different
# hyperparameter choice. Confirmed 2026-08-22 by reading siq's own model
# builder (siq/get_data.py: dbpn() / default_dbpn()) directly on GitHub:
# antspynet's own create_deep_back_projection_network_model_3d (above) scales
# up via Conv3DTranspose (a learned deconvolution whose kernel_size/stride
# both equal the SR factor's kernel/stride), but siq's dbpn() scales up via
# UpSampling3D (a fixed, non-learnable nearest-neighbor resize) immediately
# followed by a plain stride-1 Conv3D with a FIXED kernel_size=(3,3,3) --
# structurally different ops with different weight shapes. Loading real SIQ
# weights into create_deep_back_projection_network_model_3d would therefore
# never work, independent of any layer-ordering question -- the up-path
# literally does not have the same set of learnable tensors.
#
# Confirmed from siq's source (get_data.py, dbpn()/default_dbpn()):
#   * up-scaling: UpSampling3D(size=strides) -> Conv3D(kernel_size=(3,3,3),
#     strides=(1,1,1)) -- called twice per block (up0, up1 in an up-block;
#     once as up0 in a down-block).
#   * down-scaling ("Scale down"/residual down): Conv3D(kernel_size=
#     convolution_kernel_size, strides=strides) -- a genuinely strided conv,
#     NOT preceded by any resize.
#   * convolution_kernel_size is ALWAYS isotropic ((convn,)*3, e.g. (6,6,6)
#     for the "L6" in these model names) even when `strides` (the actual SR
#     factor) is anisotropic, e.g. (1,1,2) -- confirmed directly in
#     default_dbpn()'s dimensionality==3 branch, which always passes
#     convolution_kernel_size=(convn, convn, convn).
#   * every block (up_block_0 and every down/up block in the loop) always
#     includes the "dense" 1x1x1-conv+PReLU pre-step (matches the existing
#     ConvTranspose-based classes above, which already assume this).
#
# UNVERIFIED end-to-end: no real sig_smallshort_train_*.h5 was available to
# round-trip against at the time this class was written -- only the
# topology/op-types are confirmed (by reading siq's real source directly,
# not guessed). The exact number_of_base_filters/number_of_feature_filters/
# number_of_back_projection_stages actually used to train these specific
# weights (vs. default_dbpn()'s "large"/"small"/"tiny" presets) is NOT
# confirmed -- tools/convert_mri_super_resolution_bespoke.py reads these
# directly from each real .h5's embedded model_config JSON at conversion
# time rather than hardcoding a guess, which sidesteps that uncertainty.
# =============================================================================

class _UpsampleThenConv3dSame(nn.Module):
    """UpSampling3D(size=strides) -> Conv3D(kernel_size=3, stride=1,
    padding='same'), matching siq.get_data.dbpn()'s up-scaling op exactly
    (see class-group docstring above). Only `.conv` carries weights."""
    def __init__(self, in_ch, out_ch, scale_factor, bias=True):
        super().__init__()
        sf = (scale_factor,) * 3 if isinstance(scale_factor, int) else tuple(scale_factor)
        self.upsample = nn.Upsample(scale_factor=sf, mode="nearest")
        self.conv = _Conv3dSame(in_ch, out_ch, kernel_size=3, stride=1, bias=bias)

    def forward(self, x):
        return self.conv(self.upsample(x))


class _SIQDBPNUpBlock3D(nn.Module):
    """Op order matches siq.get_data.dbpn()'s up_block_3d exactly:
    dense -> up0 (resize+conv) -> down0 (strided conv) -> up1 (resize+conv),
    with L/H0/L0/E/H1 computed identically to the ConvTranspose-based
    _DBPNUpBlock3D above (only up0/up1's internals differ)."""
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv3dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _UpsampleThenConv3dSame(number_of_filters, number_of_filters, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up1 = _UpsampleThenConv3dSame(number_of_filters, number_of_filters, strides, bias=True)
        self.up1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        L = self.dense_act(self.dense(x))
        H0 = self.up0_act(self.up0(L))
        L0 = self.down0_act(self.down0(H0))
        E = L0 - L
        H1 = self.up1_act(self.up1(E))
        return H0 + H1


class _SIQDBPNDownBlock3D(nn.Module):
    """Op order matches siq.get_data.dbpn()'s down_block_3d exactly:
    dense -> down0 (strided conv) -> up0 (resize+conv) -> down1 (strided
    conv)."""
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv3dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _UpsampleThenConv3dSame(number_of_filters, number_of_filters, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down1 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        H = self.dense_act(self.dense(x))
        L0 = self.down0_act(self.down0(H))
        H0 = self.up0_act(self.up0(L0))
        E = H0 - H
        L1 = self.down1_act(self.down1(E))
        return L0 + L1


class create_siq_dbpn_super_resolution_model_3d(nn.Module):
    """
    3-D deep back-projection network matching the actual architecture used
    to train ANTsPyNet's `sig_smallshort_train_*` MRI super-resolution
    weights (https://github.com/stnava/siq, get_data.py: dbpn()/
    default_dbpn()) -- NOT the same op-types as
    create_deep_back_projection_network_model_3d above. See the class-group
    docstring above this class for the full derivation and the confidence
    caveat.

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (1 for the SIQ MRI models).

    number_of_outputs : integer
        Number of outputs (1 for the SIQ MRI models).

    number_of_base_filters : integer
        Number of base filters (siq's `nfilt`; default_dbpn()'s unnamed/
        "large" default is 64, "small" is 32, "tiny" is 32).

    number_of_feature_filters : integer
        Number of feature-extraction filters (siq's `nff`; default 256,
        "small"/"tiny" is 64).

    number_of_back_projection_stages : integer
        Number of up-down-projection stages in the main loop, in addition
        to the initial up block (siq's `nbp`; default 7, "small" is 4,
        "tiny" is 2).

    convolution_kernel_size : integer or tuple of length 3
        Kernel size for the strided down-projection convolutions (siq's
        `convn`, always isotropic -- e.g. 6 for the "L6" in these model
        names).

    strides : tuple of length 3
        Per-axis super-resolution scale factor (siq's `strider`), e.g.
        (1, 1, 2).

    last_convolution : integer or tuple of length 3
        Kernel size for the final reconstruction convolution (siq's
        `lastconv`, default 3).

    Example
    -------
    >>> model = create_siq_dbpn_super_resolution_model_3d(1, strides=(1, 1, 2))
    """
    def __init__(self, input_channel_size,
                 number_of_outputs=1,
                 number_of_base_filters=64,
                 number_of_feature_filters=256,
                 number_of_back_projection_stages=7,
                 convolution_kernel_size=6,
                 strides=(1, 1, 2),
                 last_convolution=3):
        super().__init__()
        self.number_of_back_projection_stages = number_of_back_projection_stages

        self.feature_extraction = _Conv3dSame(input_channel_size, number_of_feature_filters,
                                               kernel_size=3, stride=1, bias=True)
        self.feature_extraction_act = nn.PReLU(num_parameters=number_of_feature_filters, init=0.0)

        self.smash = _Conv3dSame(number_of_feature_filters, number_of_base_filters,
                                  kernel_size=1, stride=1, bias=True)
        self.smash_act = nn.PReLU(num_parameters=number_of_base_filters, init=0.0)

        base = number_of_base_filters

        self.up_block_0 = _SIQDBPNUpBlock3D(base, base, convolution_kernel_size, strides)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(number_of_back_projection_stages):
            down_in_ch = (i + 1) * base
            self.down_blocks.append(_SIQDBPNDownBlock3D(down_in_ch, base, convolution_kernel_size, strides))
            up_in_ch = (i + 1) * base
            self.up_blocks.append(_SIQDBPNUpBlock3D(up_in_ch, base, convolution_kernel_size, strides))

        final_in_ch = (number_of_back_projection_stages + 1) * base
        self.output_conv = _Conv3dSame(final_in_ch, number_of_outputs,
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


class _ConvTranspose3dSameStride(nn.Module):
    """Keras-'same'-padding ConvTranspose3d for an arbitrary (fixed) stride."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias=True):
        super().__init__()
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        st = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        pad, outpad = _deconv_same_stride_params(ks, st)
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, ks, stride=st, padding=pad,
                                          output_padding=outpad, bias=bias)

    def forward(self, x):
        return self.deconv(x)


class _DBPNUpBlock3D(nn.Module):
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv3dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _ConvTranspose3dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up1 = _ConvTranspose3dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        L = self.dense_act(self.dense(x))
        H0 = self.up0_act(self.up0(L))
        L0 = self.down0_act(self.down0(H0))
        E = L0 - L
        H1 = self.up1_act(self.up1(E))
        return H0 + H1


class _DBPNDownBlock3D(nn.Module):
    def __init__(self, in_channels, number_of_filters, kernel_size, strides):
        super().__init__()
        self.dense = _Conv3dSame(in_channels, number_of_filters, kernel_size=1, stride=1, bias=True)
        self.dense_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down0 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.up0 = _ConvTranspose3dSameStride(number_of_filters, number_of_filters, kernel_size, strides, bias=True)
        self.up0_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)
        self.down1 = _Conv3dSame(number_of_filters, number_of_filters, kernel_size=kernel_size, stride=strides, bias=True)
        self.down1_act = nn.PReLU(num_parameters=number_of_filters, init=0.0)

    def forward(self, x):
        H = self.dense_act(self.dense(x))
        L0 = self.down0_act(self.down0(H))
        H0 = self.up0_act(self.up0(L0))
        E = H0 - H
        L1 = self.down1_act(self.down1(E))
        return L0 + L1


class create_deep_back_projection_network_model_3d(nn.Module):
    """
    3-D deep back-projection network.

    Dimension-for-dimension port of create_deep_back_projection_network_model_2d
    (see that class's docstring for the general block structure) -- the only
    difference is Conv3d/ConvTranspose3d in place of Conv2d/ConvTranspose2d,
    and 3-tuple kernel_size/strides/last_convolution.

    Added 2026-08-22 for antstorch.utilities.mri_super_resolution (the SIQ
    DBPN models at https://github.com/stnava/siq).  Unlike the 2-D case
    (mouse_histology_super_resolution, a single fixed 2x/RGB configuration
    matched against a real converted weights file), no real SIQ weights file
    was available to verify this 3-D class against at port time -- only the
    architecture code itself has been checked (matches the 2-D class exactly,
    dimension for dimension; forward pass exercised on random input of
    various shapes with no errors). Treat as unverified until a real SIQ
    .h5/SavedModel is converted and round-tripped through this class.

    Arguments
    ---------
    input_channel_size : integer
        Number of input channels (1 for the SIQ MRI models).

    number_of_outputs : integer
        Number of outputs (1 for the SIQ MRI models).

    number_of_base_filters : integer
        Number of base filters.

    number_of_feature_filters : integer
        Number of feature filters.

    number_of_back_projection_stages : integer
        Number of up-down-projection stages (in addition to the initial up block).

    convolution_kernel_size : tuple of length 3
        Kernel size for the up/down-projection convolutions, one value per
        spatial axis -- independent per axis to support the SIQ models'
        anisotropic expansion factors (e.g. [1,1,2]).

    strides : tuple of length 3
        Strides for the up/down-projection convolutions, i.e. the
        per-axis super-resolution scale factor (e.g. [1,1,2] for 2x
        upsampling along the last axis only, with no change along the
        other two).

    last_convolution : tuple of length 3
        Kernel size for the final convolutional layer.

    Example
    -------
    >>> model = create_deep_back_projection_network_model_3d(1, strides=(1,1,2), convolution_kernel_size=(5,5,6))
    """
    def __init__(self, input_channel_size,
                 number_of_outputs=1,
                 number_of_base_filters=64,
                 number_of_feature_filters=256,
                 number_of_back_projection_stages=7,
                 convolution_kernel_size=(12, 12, 12),
                 strides=(8, 8, 8),
                 last_convolution=(3, 3, 3)):
        super().__init__()
        self.number_of_back_projection_stages = number_of_back_projection_stages

        self.feature_extraction = _Conv3dSame(input_channel_size, number_of_feature_filters,
                                               kernel_size=3, stride=1, bias=True)
        self.feature_extraction_act = nn.PReLU(num_parameters=number_of_feature_filters, init=0.0)

        self.smash = _Conv3dSame(number_of_feature_filters, number_of_base_filters,
                                  kernel_size=1, stride=1, bias=True)
        self.smash_act = nn.PReLU(num_parameters=number_of_base_filters, init=0.0)

        base = number_of_base_filters

        # Pre-loop up block: input is the `base`-channel output of the smash layer.
        self.up_block_0 = _DBPNUpBlock3D(base, base, convolution_kernel_size, strides)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(number_of_back_projection_stages):
            down_in_ch = (i + 1) * base
            self.down_blocks.append(_DBPNDownBlock3D(down_in_ch, base, convolution_kernel_size, strides))
            up_in_ch = (i + 1) * base
            self.up_blocks.append(_DBPNUpBlock3D(up_in_ch, base, convolution_kernel_size, strides))

        final_in_ch = (number_of_back_projection_stages + 1) * base
        self.output_conv = _Conv3dSame(final_in_ch, number_of_outputs,
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
