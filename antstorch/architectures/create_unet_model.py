
#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helpers: centered pad/crop to avoid spatial shift ----------


def _keras_pad_crop(dec: torch.Tensor, target_spatial):
    """
    Keras-align `dec` to `target_spatial` with asymmetric (right/back-heavy) pad/crop.
    For each spatial dim: if dec is larger, take a *leading* slice [0:target];
    if smaller, pad zeros on the *end* only. This emulates TF/Keras 'same' behavior
    for even kernels and avoids cumulative half-voxel drift across decoder stages.
    Works for 2D (NCHW) and 3D (NCDHW).
    """
    if dec.dim() not in (4, 5):
        raise ValueError(f"Expected NCHW/NCDHW, got shape {dec.shape}")
    spatial = dec.shape[-2:] if dec.dim() == 4 else dec.shape[-3:]
    target = tuple(int(t) for t in target_spatial)
    if spatial == target:
        return dec

    if dec.dim() == 4:
        H, W = spatial
        tH, tW = target
        # Crop or pad height (end-only)
        if H > tH:
            dec = dec[:, :, 0:tH, :]
        elif H < tH:
            pad_h = tH - H
            dec = F.pad(dec, (0, 0, 0, pad_h))  # (W_left, W_right, H_top, H_bottom)
        # Crop or pad width (end-only)
        if W > tW:
            dec = dec[:, :, :, 0:tW]
        elif W < tW:
            pad_w = tW - W
            dec = F.pad(dec, (0, pad_w, 0, 0))
        return dec

    # 3D
    D, H, W = spatial
    tD, tH, tW = target
    # Depth
    if D > tD:
        dec = dec[:, :, 0:tD, :, :]
    elif D < tD:
        pad_d = tD - D
        dec = F.pad(dec, (0, 0, 0, 0, 0, pad_d))  # (W_l,W_r,H_t,H_b,D_front,D_back)
    # Height
    if H > tH:
        dec = dec[:, :, :, 0:tH, :]
    elif H < tH:
        pad_h = tH - H
        dec = F.pad(dec, (0, 0, 0, pad_h, 0, 0))
    # Width
    if W > tW:
        dec = dec[:, :, :, :, 0:tW]
    elif W < tW:
        pad_w = tW - W
        dec = F.pad(dec, (0, pad_w, 0, 0, 0, 0))
    return dec

import torch
import torch.nn.functional as F

def _center_pad_crop(x: torch.Tensor, target_spatial):
    """
    Center-align `x` to `target_spatial` by symmetric pad/crop.

    Supports:
      - 2D: x.shape = (N, C, H, W),  target_spatial = (tH, tW)
      - 3D: x.shape = (N, C, D, H, W), target_spatial = (tD, tH, tW)

    Notes on F.pad arg order:
      2D -> (W_left, W_right, H_top, H_bottom)
      3D -> (W_left, W_right, H_top, H_bottom, D_front, D_back)
    """
    if x.dim() not in (4, 5):
        raise ValueError(f"Expected NCHW or NCDHW, got {tuple(x.shape)}")

    if x.dim() == 4:
        H, W = x.shape[-2:]
        tH, tW = map(int, target_spatial)

        # H (top/bottom)
        if H != tH:
            dh = tH - H
            if dh > 0:
                top = dh // 2
                bot  = dh - top
                x = F.pad(x, (0, 0, top, bot))
            else:
                cut = -dh
                top = cut // 2
                bot  = cut - top
                x = x[:, :, top:H - bot, :]

        # W (left/right)
        if W != tW:
            dw = tW - W
            if dw > 0:
                left = dw // 2
                right = dw - left
                x = F.pad(x, (left, right, 0, 0))
            else:
                cut = -dw
                left = cut // 2
                right = cut - left
                x = x[:, :, :, left:W - right]

        return x

    # x.dim() == 5
    D, H, W = x.shape[-3:]
    tD, tH, tW = map(int, target_spatial)

    # D (front/back)
    if D != tD:
        dd = tD - D
        if dd > 0:
            front = dd // 2
            back  = dd - front
            x = F.pad(x, (0, 0, 0, 0, front, back))
        else:
            cut = -dd
            front = cut // 2
            back  = cut - front
            x = x[:, :, front:D - back, :, :]

    # H (top/bottom)
    if H != tH:
        dh = tH - H
        if dh > 0:
            top = dh // 2
            bot  = dh - top
            x = F.pad(x, (0, 0, top, bot, 0, 0))
        else:
            cut = -dh
            top = cut // 2
            bot  = cut - top
            x = x[:, :, :, top:H - bot, :]

    # W (left/right)
    if W != tW:
        dw = tW - W
        if dw > 0:
            left = dw // 2
            right = dw - left
            x = F.pad(x, (left, right, 0, 0, 0, 0))
        else:
            cut = -dw
            left = cut // 2
            right = cut - left
            x = x[:, :, :, :, left:W - right]

    return x

def _deconv_same_params(kernel_size):
    if isinstance(kernel_size, int):
        ks = (kernel_size,)
    else:
        ks = tuple(kernel_size)
    # for stride=1 transpose conv; output_padding must be 0
    pad = tuple(int(k // 2) for k in ks)
    outpad = tuple(0 for _ in ks)
    return pad, outpad

import torch.nn.functional as F

# 3D: F.pad takes (W_left, W_right, H_top, H_bottom, D_front, D_back)
def _align_leading_3d(x, target_spatial):
    tD, tH, tW = map(int, target_spatial)
    D, H, W = x.shape[-3:]

    # Depth
    if D < tD:  # pad at the *front*
        x = F.pad(x, (0, 0, 0, 0, tD - D, 0))
    elif D > tD:  # crop at the *front*
        x = x[:, :, D - tD:, :, :]

    # Height
    if H < tH:  # pad at the *top*
        x = F.pad(x, (0, 0, tH - H, 0, 0, 0))
    elif H > tH:  # crop at the *top*
        x = x[:, :, :, H - tH:, :]

    # Width
    if W < tW:  # pad at the *left*
        x = F.pad(x, (tW - W, 0, 0, 0, 0, 0))
    elif W > tW:  # crop at the *left*
        x = x[:, :, :, :, W - tW:]

    return x

# 2D: F.pad takes (W_left, W_right, H_top, H_bottom)
def _align_leading_2d(x, target_spatial):
    tH, tW = map(int, target_spatial)
    H, W = x.shape[-2:]

    # Height
    if H < tH:
        x = F.pad(x, (0, 0, tH - H, 0))
    elif H > tH:
        x = x[:, :, H - tH:, :]

    # Width
    if W < tW:
        x = F.pad(x, (tW - W, 0, 0, 0))
    elif W > tW:
        x = x[:, :, :, W - tW:]

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
        kH,kW = self.kernel_size
        sH,sW = self.stride
        dH,dW = self.dilation
        def _same(L, s, k, d):
            out = math.ceil(L / s)
            need = max(0, (out - 1)*s + (k - 1)*d + 1 - L)
            return need//2, need - need//2
        pH = _same(x.size(-2), sH, kH, dH)
        pW = _same(x.size(-1), sW, kW, dW)
        x = F.pad(x, (pW[0], pW[1], pH[0], pH[1]))
        return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                        padding=0, dilation=self.dilation, groups=self.groups)

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
        kD,kH,kW = self.kernel_size
        sD,sH,sW = self.stride
        dD,dH,dW = self.dilation
        def _same(L, s, k, d):
            out = math.ceil(L / s)
            need = max(0, (out - 1)*s + (k - 1)*d + 1 - L)
            return need//2, need - need//2
        pD = _same(x.size(-3), sD, kD, dD)
        pH = _same(x.size(-2), sH, kH, dH)
        pW = _same(x.size(-1), sW, kW, dW)
        x = F.pad(x, (pW[0], pW[1], pH[0], pH[1], pD[0], pD[1]))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride,
                        padding=0, dilation=self.dilation, groups=self.groups)

def _same_align_deconv_2d_keras(y, target_spatial, kernel_size):
    """
    Keras-like SAME alignment for ConvTranspose2d with stride=1.
    For *even* kernel sizes, align origin by applying crop/pad on the *leading* side
    (top/left). For odd kernels, symmetric center is already exact.

    y:              (N, C, H, W) - output of ConvTranspose2d
    target_spatial: (tH, tW)     - input tensor's spatial size (what we want to match)
    kernel_size:    int or (kH, kW)
    """
    if isinstance(kernel_size, int):
        ks = (kernel_size, kernel_size)
    else:
        ks = tuple(int(k) for k in kernel_size)

    tH, tW = map(int, target_spatial)
    H, W = y.shape[-2:]

    dH = tH - H  # +1 means we need to add one voxel; -1 means we must remove one
    dW = tW - W

    # F.pad order for 2D is (W_left, W_right, H_top, H_bottom)

    # Height (top/bottom)
    if ks[0] % 2 == 0:
        if dH == 1:      # need +1 → pad at the *top*
            y = F.pad(y, (0, 0, 1, 0))
        elif dH == -1:   # need -1 → crop at the *top*
            y = y[:, :, 1:, :]

    # Width (left/right)
    if ks[1] % 2 == 0:
        if dW == 1:      # need +1 → pad at the *left*
            y = F.pad(y, (1, 0, 0, 0))
        elif dW == -1:   # need -1 → crop at the *left*
            y = y[:, :, :, 1:]

    # If anything still off (rare), fall back to symmetric center
    if y.shape[-2:] != (tH, tW):
        y = _center_pad_crop(y, (tH, tW))

    return y

class _ConvTranspose2dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)*2
        pad, outpad = _deconv_same_params(kernel_size)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=kernel_size, stride=1,
            padding=pad, output_padding=outpad, bias=bias
        )
        self.kernel_size = kernel_size  # <-- keep this for alignment

    def forward(self, x):
        y = self.deconv(x)
        # Keras-like SAME alignment for even kernels, stride=1
        return _same_align_deconv_2d_keras(y, x.shape[-2:], self.kernel_size)

def _same_align_deconv_3d_keras(y, target_spatial, kernel_size):
    """
    Keras-like SAME alignment for ConvTranspose3d with stride=1.
    For *even* kernel sizes, align origin by applying crop/pad on the *leading* side
    (front/top/left). For odd kernels, symmetric center works exact.
    y: N,C,D,H,W ; target_spatial: (tD,tH,tW) from the *input* tensor.
    """
    if isinstance(kernel_size, int):
        ks = (kernel_size,)*3
    else:
        ks = tuple(int(k) for k in kernel_size)
    tD, tH, tW = map(int, target_spatial)
    D, H, W = y.shape[-3:]

    # Helper: crop/pad 1 voxel on the *leading* side for each axis where needed
    # delta = target - current
    dD = tD - D
    dH = tH - H
    dW = tW - W

    # For even kernels, prefer leading-side adjust; for odd, nothing needed typically.
    # Depth (front/back)
    if ks[0] % 2 == 0:
        if dD == 1:      # need +1 voxel → pad at the *front*
            y = F.pad(y, (0,0, 0,0, 1,0))
        elif dD == -1:   # need -1 voxel → crop at the *front*
            y = y[:, :, 1:, :, :]

    # Height (top/bottom)
    if ks[1] % 2 == 0:
        if dH == 1:      # pad top
            y = F.pad(y, (0,0, 1,0, 0,0))
        elif dH == -1:   # crop top
            y = y[:, :, :, 1:, :]

    # Width (left/right)
    if ks[2] % 2 == 0:
        if dW == 1:      # pad left
            y = F.pad(y, (1,0, 0,0, 0,0))
        elif dW == -1:   # crop left
            y = y[:, :, :, :, 1:]

    # If anything else remains (shouldn't for stride=1), fall back to symmetric center
    if y.shape[-3:] != (tD, tH, tW):
        y = _center_pad_crop(y, (tD, tH, tW))
    return y

class _ConvTranspose3dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)*3
        # Use padding=1 for odd kernels (exact), padding=1 for even (needs +1 adjust), or padding=0 then crop.
        # Either works with the aligner below; keep your existing pad/outpad if you like.
        pad, outpad = _deconv_same_params(kernel_size)
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size,
                                         stride=1, padding=pad,
                                         output_padding=outpad, bias=bias)
        self.kernel_size = kernel_size

    def forward(self, x):
        y = self.deconv(x)
        # Align to input spatial dims using Keras-like leading-side rule for even kernels
        return _same_align_deconv_3d_keras(y, x.shape[-3:], self.kernel_size)


# ---------- Concat module to expose in CSV ----------

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, up, skip):
        # Align skip to the spatial size of up using KERAS-like leading-side rule
        if up.dim() == 5:
            skip = _align_leading_3d(skip, up.shape[-3:])
        else:
            skip = _align_leading_2d(skip, up.shape[-2:])
        return torch.cat([up, skip], dim=self.dim)

# ---------- Wraps an existing UNet3D model ----------

def _is_conv3d_like(m: nn.Module) -> bool:
    W = getattr(m, "weight", None)
    if W is not None:
        try:
            return W.dim() == 5  # (outC, inC, kD, kH, kW)
        except Exception:
            pass
    return isinstance(m, nn.Conv3d)

def _get_kernel_size(m: nn.Module):
    ks = getattr(m, "kernel_size", None)
    if ks is None:
        W = getattr(m, "weight", None)
        if W is not None and W.dim() == 5:
            return tuple(W.shape[2:5])  # (kD, kH, kW)
        return None
    if isinstance(ks, int):
        return (ks, ks, ks)
    return tuple(ks)

def _get_out_channels(m: nn.Module) -> int | None:
    W = getattr(m, "weight", None)
    return int(W.shape[0]) if (W is not None and W.dim() == 5) else None

def _find_final_output_conv3d(module: nn.Module, n_main_outputs: int | None = None) -> nn.Module | None:
    """
    Find the final conv-like module, preferably the 1x1x1 main output head.
    """
    last_any = None
    last_head = None
    for m in module.modules():
        if not _is_conv3d_like(m):
            continue
        last_any = m
        ks = _get_kernel_size(m)
        oc = _get_out_channels(m)
        if ks == (1, 1, 1):
            if n_main_outputs is None or (oc == n_main_outputs):
                last_head = m
    return last_head or last_any


class create_multihead_unet_model_3d(nn.Module):
    def __init__(self, base_unet: nn.Module, n_aux_heads: int,
                 use_sigmoid: bool = True, n_main_outputs: int | None = None):
        super().__init__()
        self.base = base_unet
        self.n_aux_heads = n_aux_heads
        self._feat = None
        self.heads = nn.ModuleList()
        self._heads_built = False
        self.activ = nn.Sigmoid() if use_sigmoid else nn.Identity()

        # NEW: find the final 1x1x1 output conv and hook its *input*
        last_conv = _find_final_output_conv3d(self.base, n_main_outputs)
        if last_conv is None:
            raise RuntimeError("Could not find a 3D conv in base_unet to hook as penultimate layer.")

        def _grab_input(module, inp):
            self._feat = inp[0]
        self._hook = last_conv.register_forward_pre_hook(_grab_input)

        # We'll lazily create the heads after we see the first forward pass
        # (so we know feature channels). Alternatively, you can pass in_channels explicitly.
        self._heads_built = False
        self.heads = nn.ModuleList()
        self.activ = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def _maybe_build_heads(self):
        if self._heads_built:
            return
        if self._feat is None:
            raise RuntimeError("Penultimate feature is not captured yet; run one forward pass first.")
        in_ch = self._feat.shape[1]
        for _ in range(self.n_aux_heads):
            self.heads.append(nn.Conv3d(in_ch, 1, kernel_size=1, bias=True))
        self._heads_built = True

    @torch.no_grad()
    def warmup(self, sample):
        # One dry run to infer penultimate channels and build heads.
        _ = self.base(sample)
        self._maybe_build_heads()

    def forward(self, x):
        y_main = self.base(x)               # triggers the hook -> self._feat
        if not self._heads_built:
            self._maybe_build_heads()
        aux = [self.activ(h(self._feat)) for h in self.heads]
        # match Keras outputs order: [main, output1, (output2), (output3)]
        return (y_main, *aux)


# ---------- UNet 2D & 3D with centered alignment ----------

class create_unet_model_2d(nn.Module):
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
        pad_crop="keras",
        additional_options=None,
    ):
        super().__init__()

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
            block = [conv1, nn.ReLU()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, nn.ReLU()]
            self.encoding_convolution_layers.append(nn.Sequential(*block))

        # ---- 2D decoder (Keras-style: deconv stride=1 + upsample×2) ----
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = _ConvTranspose2dSame(
                in_ch=number_of_filters[number_of_layers - i],
                out_ch=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                bias=True,
            )
            self.decoding_convolution_transpose_layers.append(deconv)
            self.concat_layers.append(Concat(dim=1))

            # After concat: channels double -> 2 * out_ch
            out_ch = number_of_filters[number_of_layers - i - 1]
            in_ch_post_concat = 2 * out_ch
            conv1 = _Conv2dSame(in_ch_post_concat, out_ch,
                                kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv2dSame(out_ch, out_ch,
                                kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, nn.ReLU()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, nn.ReLU()]
            self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = _Conv2dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
        if mode == "sigmoid":
            self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification":
            self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression":
            self.output = nn.Sequential(head_conv)
        else:
            raise ValueError("mode must be `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        L = len(self.encoding_convolution_layers)
        skips = []
        enc = x
        for i in range(L):
            enc = self.encoding_convolution_layers[i](enc)
            skips.append(enc)
            if i < L - 1:
                enc = self.pool(enc)

        dec = skips[-1]
        for i in range(1, L):
            skip = skips[L - i - 1]
            # 1) deconv (stride=1)
            dec = self.decoding_convolution_transpose_layers[i - 1](dec)

            # 2) upsample ×2 (nearest) to approach skip size
            dec = self.upsample(dec)

            # 3) final safety align to exact skip size (handles odd dims)
            if self.pad_crop == "keras":
                dec = _align_leading_3d(dec, skip.shape[-2:])
            elif self.pad_crop == "center":    
                dec = _center_pad_crop(dec, skip.shape[-2:])
            else:
                raise ValueError("Unrecognized pad_crop option.")    

            # 4) concat in Keras order [up, skip] along channels
            dec = self.concat_layers[i - 1](dec, skip)

            # 5) two convs
            dec = self.decoding_convolution_layers[i - 1](dec)


        return self.output(dec)

class create_unet_model_3d(nn.Module):
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
        pad_crop="keras",
        additional_options=None,
    ):
        super().__init__()

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
            block = [conv1, nn.ReLU()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, nn.ReLU()]
            self.encoding_convolution_layers.append(nn.Sequential(*block))

        # ---- 3D decoder (Keras-style: deconv stride=1 + upsample×2) ----
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")
        self.decoding_convolution_transpose_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.decoding_convolution_layers = nn.ModuleList()

        for i in range(1, number_of_layers):
            deconv = _ConvTranspose3dSame(
                in_ch=number_of_filters[number_of_layers - i],
                out_ch=number_of_filters[number_of_layers - i - 1],
                kernel_size=deconvolution_kernel_size,
                bias=True,
            )
            self.decoding_convolution_transpose_layers.append(deconv)
            self.concat_layers.append(Concat(dim=1))

            # After concat: channels double -> 2 * out_ch
            out_ch = number_of_filters[number_of_layers - i - 1]
            in_ch_post_concat = 2 * out_ch
            conv1 = _Conv3dSame(in_ch_post_concat, out_ch,
                                kernel_size=convolution_kernel_size, stride=1, bias=True)
            conv2 = _Conv3dSame(out_ch, out_ch,
                                kernel_size=convolution_kernel_size, stride=1, bias=True)
            block = [conv1, nn.ReLU()]
            if dropout_rate > 0.0: block += [nn.Dropout(dropout_rate)]
            block += [conv2, nn.ReLU()]
            self.decoding_convolution_layers.append(nn.Sequential(*block))

        head_conv = _Conv3dSame(number_of_filters[0], number_of_outputs, kernel_size=1, stride=1, bias=True)
        if mode == "sigmoid":
            self.output = nn.Sequential(head_conv, nn.Sigmoid())
        elif mode == "classification":
            self.output = nn.Sequential(head_conv, nn.Softmax(dim=1))
        elif mode == "regression":
            self.output = nn.Sequential(head_conv)
        else:
            raise ValueError("mode must be `classification`, `regression` or `sigmoid`.")

    def forward(self, x):
        L = len(self.encoding_convolution_layers)
        skips = []
        enc = x
        for i in range(L):
            enc = self.encoding_convolution_layers[i](enc)
            skips.append(enc)
            if i < L - 1:
                enc = self.pool(enc)

        dec = skips[-1]
        for i in range(1, L):
            skip = skips[L - i - 1]
            # 1) deconv (stride=1)
            dec = self.decoding_convolution_transpose_layers[i - 1](dec)

            # 2) upsample ×2 (nearest) to approach skip size
            dec = self.upsample(dec)

            # 3) final safety align to exact skip size (handles odd dims)
            if self.pad_crop == "keras":
                # Keras SAME alignment for even kernels: adjust on *leading* side
                dec = _align_leading_3d(dec, skip.shape[-3:])
            elif self.pad_crop == "center":
                dec = _center_pad_crop(dec, skip.shape[-3:])
            else:
                raise ValueError("Unrecognized pad_crop option.")

            # 4) concat in Keras order [up, skip] along channels
            dec = self.concat_layers[i - 1](dec, skip)

            # 5) two convs
            dec = self.decoding_convolution_layers[i - 1](dec)

        return self.output(dec)
