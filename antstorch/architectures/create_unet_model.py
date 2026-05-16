#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANTsPyNetInstanceNorm(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dims = tuple(range(1, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        std = torch.sqrt(var)
        return self.weight * ((x - mean) / (std + self.eps)) + self.bias

def _center_pad_crop(x: torch.Tensor, target_spatial):
    if x.dim() not in (4, 5): raise ValueError(f"Expected NCHW or NCDHW, got {tuple(x.shape)}")
    if x.dim() == 4:
        H, W = x.shape[-2:]
        tH, tW = map(int, target_spatial)
        if H != tH:
            dh = tH - H
            if dh > 0:
                top = dh // 2
                x = F.pad(x, (0, 0, top, dh - top))
            else:
                cut = -dh
                top = cut // 2
                x = x[:, :, top:H - (cut - top), :]
        if W != tW:
            dw = tW - W
            if dw > 0:
                left = dw // 2
                x = F.pad(x, (left, dw - left, 0, 0))
            else:
                cut = -dw
                left = cut // 2
                x = x[:, :, :, left:W - (cut - left)]
        return x
    D, H, W = x.shape[-3:]
    tD, tH, tW = map(int, target_spatial)
    if D != tD:
        dd = tD - D
        if dd > 0:
            front = dd // 2
            x = F.pad(x, (0, 0, 0, 0, front, dd - front))
        else:
            cut = -dd
            front = cut // 2
            x = x[:, :, front:D - (cut - front), :, :]
    if H != tH:
        dh = tH - H
        if dh > 0:
            top = dh // 2
            x = F.pad(x, (0, 0, top, dh - top, 0, 0))
        else:
            cut = -dh
            top = cut // 2
            x = x[:, :, :, top:H - (cut - top), :]
    if W != tW:
        dw = tW - W
        if dw > 0:
            left = dw // 2
            x = F.pad(x, (left, dw - left, 0, 0, 0, 0))
        else:
            cut = -dw
            left = cut // 2
            x = x[:, :, :, :, left:W - (cut - left)]
    return x

def _deconv_same_params(kernel_size):
    if isinstance(kernel_size, int): ks = (kernel_size,)
    else: ks = tuple(kernel_size)
    pad = tuple(int(k // 2) for k in ks)
    outpad = tuple(0 for _ in ks)
    return pad, outpad

def _align_leading_3d(x, target_spatial):
    tD, tH, tW = map(int, target_spatial)
    D, H, W = x.shape[-3:]
    if D < tD: x = F.pad(x, (0, 0, 0, 0, tD - D, 0))
    elif D > tD: x = x[:, :, D - tD:, :, :]
    if H < tH: x = F.pad(x, (0, 0, tH - H, 0, 0, 0))
    elif H > tH: x = x[:, :, :, H - tH:, :]
    if W < tW: x = F.pad(x, (tW - W, 0, 0, 0, 0, 0))
    elif W > tW: x = x[:, :, :, :, W - tW:]
    return x

def _align_leading_2d(x, target_spatial):
    tH, tW = map(int, target_spatial)
    H, W = x.shape[-2:]
    if H < tH: x = F.pad(x, (0, 0, tH - H, 0))
    elif H > tH: x = x[:, :, H - tH:, :]
    if W < tW: x = F.pad(x, (tW - W, 0, 0, 0))
    elif W > tW: x = x[:, :, :, W - tW:]
    return x

class _Conv2dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True, groups=1):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)*2
        if isinstance(stride, int):      stride = (stride,)*2
        if isinstance(dilation, int):    dilation = (dilation,)*2
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_ch * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        def _same(L, s, k, d):
            out = math.ceil(L / s)
            need = max(0, (out - 1)*s + (k - 1)*d + 1 - L)
            return need//2, need - need//2
        pH = _same(x.size(-2), self.stride[0], self.kernel_size[0], self.dilation[0])
        pW = _same(x.size(-1), self.stride[1], self.kernel_size[1], self.dilation[1])
        x = F.pad(x, (pW[0], pW[1], pH[0], pH[1]))
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

class _Conv3dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True, groups=1):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)*3
        if isinstance(stride, int):      stride = (stride,)*3
        if isinstance(dilation, int):    dilation = (dilation,)*3
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_ch * kernel_size[0] * kernel_size[1] * kernel_size[2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        def _same(L, s, k, d):
            out = math.ceil(L / s)
            need = max(0, (out - 1)*s + (k - 1)*d + 1 - L)
            return need//2, need - need//2
        pD = _same(x.size(-3), self.stride[0], self.kernel_size[0], self.dilation[0])
        pH = _same(x.size(-2), self.stride[1], self.kernel_size[1], self.dilation[1])
        pW = _same(x.size(-1), self.stride[2], self.kernel_size[2], self.dilation[2])
        x = F.pad(x, (pW[0], pW[1], pH[0], pH[1], pD[0], pD[1]))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

def _same_align_deconv_2d_keras(y, target_spatial, kernel_size):
    ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(int(k) for k in kernel_size)
    tH, tW = map(int, target_spatial)
    H, W = y.shape[-2:]
    dH, dW = tH - H, tW - W
    if ks[0] % 2 == 0:
        if dH == 1: y = F.pad(y, (0, 0, 1, 0))
        elif dH == -1: y = y[:, :, 1:, :]
    if ks[1] % 2 == 0:
        if dW == 1: y = F.pad(y, (1, 0, 0, 0))
        elif dW == -1: y = y[:, :, :, 1:]
    if y.shape[-2:] != (tH, tW): y = _center_pad_crop(y, (tH, tW))
    return y

class _ConvTranspose2dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        ks = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        pad, outpad = _deconv_same_params(ks)
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, ks, stride=1, padding=pad, output_padding=outpad, bias=bias)
        self.kernel_size = ks
    def forward(self, x):
        return _same_align_deconv_2d_keras(self.deconv(x), x.shape[-2:], self.kernel_size)

def _same_align_deconv_3d_keras(y, target_spatial, kernel_size):
    ks = (kernel_size,)*3 if isinstance(kernel_size, int) else tuple(int(k) for k in kernel_size)
    tD, tH, tW = map(int, target_spatial)
    D, H, W = y.shape[-3:]
    dD, dH, dW = tD - D, tH - H, tW - W
    if ks[0] % 2 == 0:
        if dD == 1: y = F.pad(y, (0,0, 0,0, 1,0))
        elif dD == -1: y = y[:, :, 1:, :, :]
    if ks[1] % 2 == 0:
        if dH == 1: y = F.pad(y, (0,0, 1,0, 0,0))
        elif dH == -1: y = y[:, :, :, 1:, :]
    if ks[2] % 2 == 0:
        if dW == 1: y = F.pad(y, (1,0, 0,0, 0,0))
        elif dW == -1: y = y[:, :, :, :, 1:]
    if y.shape[-3:] != (tD, tH, tW): y = _center_pad_crop(y, (tD, tH, tW))
    return y

class _ConvTranspose3dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        ks = (kernel_size,)*3 if isinstance(kernel_size, int) else kernel_size
        pad, outpad = _deconv_same_params(ks)
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, ks, stride=1, padding=pad, output_padding=outpad, bias=bias)
        self.kernel_size = ks
    def forward(self, x):
        return _same_align_deconv_3d_keras(self.deconv(x), x.shape[-3:], self.kernel_size)

class _ConvTranspose2dKerasExact(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__()
        self.ks = (kernel_size,)*2 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, self.ks, stride=stride, padding=0, bias=bias)
    def forward(self, x):
        y = self.deconv(x)
        slices = [slice(None), slice(None)]
        for k in self.ks:
            pad_front = (k - 1) // 2
            pad_back = (k - 1) - pad_front
            slices.append(slice(pad_front, -pad_back if pad_back > 0 else None))
        return y[tuple(slices)]

class _ConvTranspose3dKerasExact(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__()
        self.ks = (kernel_size,)*3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, self.ks, stride=stride, padding=0, bias=bias)
    def forward(self, x):
        y = self.deconv(x)
        slices = [slice(None), slice(None)]
        for k in self.ks:
            pad_front = (k - 1) // 2
            pad_back = (k - 1) - pad_front
            slices.append(slice(pad_front, -pad_back if pad_back > 0 else None))
        return y[tuple(slices)]

class AttentionGate2d(nn.Module):
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))
        self.theta = _Conv2dSame(in_ch_x, inter_ch, kernel_size=1, stride=1, bias=True)
        self.phi   = _Conv2dSame(in_ch_g, inter_ch, kernel_size=1, stride=1, bias=True)
        self.psi   = _Conv2dSame(inter_ch, 1, kernel_size=1, stride=1, bias=True)
        self.relu  = nn.ReLU()
        self.sig   = nn.Sigmoid()
    def forward(self, x, g):
        if x.shape[-2:] != g.shape[-2:]: g = _align_leading_2d(g, x.shape[-2:])
        f = self.relu(self.theta(x) + self.phi(g))
        return x * self.sig(self.psi(f))

class AttentionGate3d(nn.Module):
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))
        self.theta = _Conv3dSame(in_ch_x, inter_ch, kernel_size=1, stride=1, bias=True)
        self.phi   = _Conv3dSame(in_ch_g, inter_ch, kernel_size=1, stride=1, bias=True)
        self.psi   = _Conv3dSame(inter_ch, 1, kernel_size=1, stride=1, bias=True)
        self.relu  = nn.ReLU()
        self.sig   = nn.Sigmoid()
    def forward(self, x, g):
        if x.shape[-3:] != g.shape[-3:]: g = _align_leading_3d(g, x.shape[-3:])
        f = self.relu(self.theta(x) + self.phi(g))
        return x * self.sig(self.psi(f))

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, up, skip):
        if up.dim() == 5: skip = _align_leading_3d(skip, up.shape[-3:])
        else: skip = _align_leading_2d(skip, up.shape[-2:])
        return torch.cat([up, skip], dim=self.dim)

def _is_conv3d_like(m: nn.Module) -> bool:
    W = getattr(m, "weight", None)
    if W is not None:
        try: return W.dim() == 5
        except Exception: pass
    return isinstance(m, nn.Conv3d)

def _get_kernel_size(m: nn.Module):
    ks = getattr(m, "kernel_size", None)
    if ks is None:
        W = getattr(m, "weight", None)
        if W is not None and W.dim() == 5: return tuple(W.shape[2:5])
        return None
    if isinstance(ks, int): return (ks, ks, ks)
    return tuple(ks)

def _get_out_channels(m: nn.Module) -> int | None:
    W = getattr(m, "weight", None)
    return int(W.shape[0]) if (W is not None and W.dim() == 5) else None

def _find_final_output_conv3d(module: nn.Module, n_main_outputs: int | None = None) -> nn.Module | None:
    last_any, last_head = None, None
    for m in module.modules():
        if not _is_conv3d_like(m): continue
        last_any = m
        if _get_kernel_size(m) == (1, 1, 1):
            if n_main_outputs is None or (_get_out_channels(m) == n_main_outputs): last_head = m
    return last_head or last_any

class create_multihead_unet_model_3d(nn.Module):
    def __init__(self, base_unet: nn.Module, n_aux_heads: int, use_sigmoid: bool = True, n_main_outputs: int | None = None):
        super().__init__()
        self.base = base_unet
        self.n_aux_heads = n_aux_heads
        self._feat = None
        self.heads = nn.ModuleList()
        self._heads_built = False
        self.activ = nn.Sigmoid() if use_sigmoid else nn.Identity()
        last_conv = _find_final_output_conv3d(self.base, n_main_outputs)
        if last_conv is None: raise RuntimeError("Could not find a 3D conv in base_unet to hook as penultimate layer.")
        def _grab_input(module, inp): self._feat = inp[0]
        self._hook = last_conv.register_forward_pre_hook(_grab_input)
    def _maybe_build_heads(self):
        if self._heads_built: return
        if self._feat is None: raise RuntimeError("Penultimate feature is not captured yet; run one forward pass first.")
        in_ch = self._feat.shape[1]
        for _ in range(self.n_aux_heads): self.heads.append(nn.Conv3d(in_ch, 1, kernel_size=1, bias=True))
        self._heads_built = True
    @torch.no_grad()
    def warmup(self, sample):
        _ = self.base(sample)
        self._maybe_build_heads()
    def forward(self, x):
        y_main = self.base(x)
        if not self._heads_built: self._maybe_build_heads()
        aux = [self.activ(h(self._feat)) for h in self.heads]
        return (y_main, *aux)

class create_unet_model_2d(nn.Module):
    def __init__(
        self, input_channel_size, number_of_outputs=2, number_of_layers=4,
        number_of_filters_at_base_layer=32, number_of_filters=None,
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2), strides=(2, 2), dropout_rate=0.0,
        mode="classification", pad_crop="keras", additional_options=None,
    ):
        super().__init__()
        self.add_attention_gating_2d = False
        self.add_nn_unet_activation = False
        self.use_keras_deconv = False
        if additional_options is not None:
            opts = set(additional_options) if isinstance(additional_options, (list, tuple, set)) else {additional_options}
            if ("attentionGating" in opts) or ("attention_gating" in opts): self.add_attention_gating_2d = True
            if ("nnUnetActivationStyle" in opts) or ("nn_unet_activation_style" in opts): self.add_nn_unet_activation = True
            if ("kerasDeconvolutionStyle" in opts) or ("keras_deconvolution_style" in opts): self.use_keras_deconv = True

        def _get_activation():
            if self.add_nn_unet_activation: return nn.Sequential(ANTsPyNetInstanceNorm(eps=1e-3), nn.LeakyReLU(negative_slope=0.01, inplace=True))
            return nn.ReLU()

        if number_of_filters is not None:
            number_of_filters = list(number_of_filters)
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = [number_of_filters_at_base_layer * 2**i for i in range(number_of_layers)]

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)
        self.encoding_convolution_layers = nn.ModuleList()
        self.pad_crop = pad_crop
        for i in range(number_of_layers):
            in_ch = input_channel_size if i == 0 else number_of_filters[i - 1]
            conv1 = _Conv2dSame(in_ch, number_of_filters[i], kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv2dSame(number_of_filters[i], number_of_filters[i], kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, _get_activation()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, _get_activation()]
            self.encoding_convolution_layers.append(nn.Sequential(*block))

        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.deconv_activations = nn.ModuleList()
        if self.add_attention_gating_2d: self.attn_gates_2d = nn.ModuleList()

        for i in range(1, number_of_layers):
            out_ch = number_of_filters[number_of_layers - i - 1]
            in_ch_deconv = number_of_filters[number_of_layers - i]
            
            if self.use_keras_deconv:
                deconv = _ConvTranspose2dKerasExact(in_ch=in_ch_deconv, out_ch=out_ch, kernel_size=deconvolution_kernel_size, stride=1, bias=True)
            else:
                deconv = _ConvTranspose2dSame(in_ch=in_ch_deconv, out_ch=out_ch, kernel_size=deconvolution_kernel_size, bias=True)
            self.decoding_convolution_transpose_layers.append(deconv)
            
            self.deconv_activations.append(_get_activation() if self.add_nn_unet_activation else nn.Identity())
            self.concat_layers.append(Concat(dim=1))
            if self.add_attention_gating_2d: self.attn_gates_2d.append(AttentionGate2d(in_ch_x=out_ch, in_ch_g=out_ch, inter_ch=max(1, out_ch // 4)))

            conv1 = _Conv2dSame(2 * out_ch, out_ch, kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv2dSame(out_ch, out_ch, kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, _get_activation()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, _get_activation()]
            self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = _Conv2dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
        if mode == "sigmoid": self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification": self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression": self.output = nn.Sequential(head_conv)
        else: raise ValueError("mode must be `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        L = len(self.encoding_convolution_layers)
        skips, enc = [], x
        for i in range(L):
            enc = self.encoding_convolution_layers[i](enc)
            skips.append(enc)
            if i < L - 1: enc = self.pool(enc)

        dec = skips[-1]
        for i in range(1, L):
            skip = skips[L - i - 1]
            dec = self.decoding_convolution_transpose_layers[i - 1](dec)
            dec = self.deconv_activations[i - 1](dec)
            
            # ---> LA LIGNE CLÉ QUI ÉTAIT DÉSACTIVÉE <---
            dec = self.upsample(dec)
            
            if self.pad_crop == "keras": dec = _align_leading_2d(dec, skip.shape[-2:])
            elif self.pad_crop == "center": dec = _center_pad_crop(dec, skip.shape[-2:])
            else: raise ValueError("Unrecognized pad_crop option.")    

            if self.add_attention_gating_2d: dec = torch.cat([dec, self.attn_gates_2d[i - 1](dec, skip)], dim=1)
            else: dec = self.concat_layers[i - 1](dec, skip)
            dec = self.decoding_convolution_layers[i - 1](dec)
        return self.output(dec)

class create_unet_model_3d(nn.Module):
    def __init__(
        self, input_channel_size, number_of_outputs=2, number_of_layers=4,
        number_of_filters_at_base_layer=32, number_of_filters=None,
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2), strides=(2, 2, 2), dropout_rate=0.0,
        mode="classification", pad_crop="keras", additional_options=None,
    ):
        super().__init__()
        self.add_attention_gating_3d = False
        self.add_nn_unet_activation = False
        self.use_keras_deconv = False
        
        if additional_options is not None:
            opts = set(additional_options) if isinstance(additional_options, (list, tuple, set)) else {additional_options}
            if ("attentionGating" in opts) or ("attention_gating" in opts): self.add_attention_gating_3d = True
            if ("nnUnetActivationStyle" in opts) or ("nn_unet_activation_style" in opts): self.add_nn_unet_activation = True
            if ("kerasDeconvolutionStyle" in opts) or ("keras_deconvolution_style" in opts): self.use_keras_deconv = True

        def _get_activation():
            if self.add_nn_unet_activation: return nn.Sequential(ANTsPyNetInstanceNorm(eps=1e-3), nn.LeakyReLU(negative_slope=0.01, inplace=True))
            return nn.ReLU()

        if number_of_filters is not None:
            number_of_filters = list(number_of_filters)
            number_of_layers = len(number_of_filters)
        else:
            number_of_filters = [number_of_filters_at_base_layer * 2**i for i in range(number_of_layers)]

        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=strides)
        self.encoding_convolution_layers = nn.ModuleList()
        self.pad_crop = pad_crop
        for i in range(number_of_layers):
            in_ch = input_channel_size if i == 0 else number_of_filters[i - 1]
            conv1 = _Conv3dSame(in_ch, number_of_filters[i], kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv3dSame(number_of_filters[i], number_of_filters[i], kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, _get_activation()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, _get_activation()]
            self.encoding_convolution_layers.append(nn.Sequential(*block))

        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()
        self.deconv_activations = nn.ModuleList()
        if self.add_attention_gating_3d: self.attn_gates_3d = nn.ModuleList()

        for i in range(1, number_of_layers):
            out_ch = number_of_filters[number_of_layers - i - 1]
            in_ch_deconv = number_of_filters[number_of_layers - i]
            
            if self.use_keras_deconv:
                deconv = _ConvTranspose3dKerasExact(in_ch=in_ch_deconv, out_ch=out_ch, kernel_size=deconvolution_kernel_size, stride=1, bias=True)
            else:
                deconv = _ConvTranspose3dSame(in_ch=in_ch_deconv, out_ch=out_ch, kernel_size=deconvolution_kernel_size, bias=True)
            self.decoding_convolution_transpose_layers.append(deconv)
            
            self.deconv_activations.append(_get_activation() if self.add_nn_unet_activation else nn.Identity())
            self.concat_layers.append(Concat(dim=1))
            if self.add_attention_gating_3d: self.attn_gates_3d.append(AttentionGate3d(in_ch_x=out_ch, in_ch_g=out_ch, inter_ch=max(1, out_ch // 4)))

            conv1 = _Conv3dSame(2 * out_ch, out_ch, kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv3dSame(out_ch, out_ch, kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, _get_activation()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, _get_activation()]
            self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = _Conv3dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
        if mode == "sigmoid": self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification": self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression": self.output = nn.Sequential(head_conv)
        else: raise ValueError("mode must be `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        L = len(self.encoding_convolution_layers)
        skips, enc = [], x
        for i in range(L):
            enc = self.encoding_convolution_layers[i](enc)
            skips.append(enc)
            if i < L - 1: enc = self.pool(enc)

        dec = skips[-1]
        for i in range(1, L):
            skip = skips[L - i - 1]
            dec = self.decoding_convolution_transpose_layers[i - 1](dec)
            dec = self.deconv_activations[i - 1](dec)
            
            # ---> LA LIGNE CLÉ QUI ÉTAIT DÉSACTIVÉE <---
            dec = self.upsample(dec)

            if self.pad_crop == "keras": dec = _align_leading_3d(dec, skip.shape[-3:])
            elif self.pad_crop == "center": dec = _center_pad_crop(dec, skip.shape[-3:])
            else: raise ValueError("Unrecognized pad_crop option.")

            if self.add_attention_gating_3d: dec = torch.cat([dec, self.attn_gates_3d[i - 1](dec, skip)], dim=1)
            else: dec = self.concat_layers[i - 1](dec, skip)
            dec = self.decoding_convolution_layers[i - 1](dec)
        return self.output(dec)