#!/usr/bin/env python3
"""
convert_lung_mouse_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for the 20 lung_extraction / lung_segmentation / mouse.py
`_pytorch` ids that still have no URL mapping in get_pretrained_network.py
(see the "" placeholders added there on 2026-08-22). Modeled directly on
tools/convert_wmh_bespoke.py's method and CLI: read each .h5 file's root
`layer_names` attribute (Keras's authoritative, fully-ordered list of every
layer in true construction order), extract convolution weight tensors in
that positional order, and assign them to the corresponding ANTsTorch
module via an explicit, architecture-derived role mapping. Each conversion
is verified 3 ways: load_state_dict(strict=True) with zero missing/
unexpected keys, a real forward pass with a sane output range (sigmoid in
[0,1] / softmax summing to 1), and a reload-from-disk identity check
(bit-exact, atol=1e-6).

Unlike the WMH bespoke architectures, all 20 ids here are built from
ANTsTorch's *generic* U-Net (create_unet_model_2d/3d -- see
antstorch/architectures/create_unet_model.py), so a single positional
extraction routine (`convert_generic_unet`) covers all of them, dispatched
by three ingredients read directly off the .h5 file plus the fixed
hyperparameters already recorded in antstorch/utilities/lung_extraction.py,
lung_segmentation.py and mouse.py:

  1. Deconvolution (ConvTranspose) layers -- identified by Keras's own
     auto-naming ("conv{2,3}d_transpose" vs plain "conv{2,3}d"), matched
     1:1 in h5 order against antstorch's `decoding_convolution_transpose_
     layers` keys.
  2. "Regular" convolution layers (kernel spatial size != 1) -- the
     encoding/decoding block convs, matched 1:1 in h5 order against
     antstorch's `encoding_convolution_layers` + `decoding_convolution_
     layers` keys (encoding block first, exactly like convert_wmh_bespoke
     .py's sysu_media handling).
  3. "Pointwise" (1x1 / 1x1x1 kernel) convolution layers -- these are,
     in h5 construction order: the attention-gate theta/phi/psi triplets
     (one triplet per decoder level, only for the 10 ids built with
     additional_options=("attentionGating", ...)), then the main output
     conv, then any auxiliary head convs (protonLobes only, via
     create_multihead_unet_model_3d).

CONFIDENCE NOTES (please read before trusting a converted .pt blindly):

  * The 10 ids with NO attention gating (protonLungMri,
    wholeLungMaskFromVentilation, xrayLungExtraction, pulmonaryArteryWeights,
    pulmonaryAirwayWeights, mouseT2wBrainExtraction3D,
    mouseT2wBrainParcellation3DNick/Tct/Jay) are structurally almost
    identical to convert_wmh_bespoke.py's sysu_media case -- HIGH
    confidence.
  * The 9 single-head ids WITH attention gating (maskLobes,
    lungCtWithPriorsSegmentationWeights, elBicho, ex5_coronal/sagittal,
    allen_brain_mask/leftright_coronal/cerebellum_sagittal/coronal) rely on
    the standard Oktay et al. attention-gate design (3 convs per gate:
    theta, phi, psi) -- the same assumption already encoded in
    tools/convert_antspynet_weights_to_antstorch.py's TF-based converter.
    GOOD confidence, but not independently verified end-to-end against a
    real antspynet Keras model in this environment (no TensorFlow here).
  * protonLobes (attention gating + create_multihead_unet_model_3d) adds a
    further assumption: the auxiliary head's 1x1x1 conv is the LAST
    pointwise conv in construction order (i.e. created after the main
    output conv). This mirrors how ANTsTorch's own multihead wrapper works
    but has NOT been cross-checked against antspynet's actual Keras
    construction order for this specific model. MODERATE confidence --
    strongly recommend a numerical sanity check (compare against a real
    antspynet inference) before relying on this one in production.
  * allen_sr_weights (DBPN, a completely different architecture -- see
    create_deep_back_projection_network_model.py) uses an explicit,
    hand-derived role list mirroring the ported torch module's __init__
    order, plus a PReLU alpha-shape assumption (Keras PReLU alpha reduced
    to a per-channel vector if it isn't already 1-D) that is NOT verified
    since (per scripts/verify_applications/README.md) the source .h5 for
    this id has not even been located in ~/.keras/ANTsXNet/ yet. BEST
    EFFORT only -- treat this one as a starting point to debug against,
    not a ready-to-trust converter.

Usage (run on a machine with h5py + torch installed; antstorch's source
tree needs to be importable -- point --antstorch-src at the repo root):

    python convert_lung_mouse_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch

    # convert just one file:
    python convert_lung_mouse_bespoke.py --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch --antstorch-src ~/Pkg/ANTsTorch \\
        --only protonLungMri
"""
import argparse
import importlib.util
import os
import re
import sys
import types

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import antstorch's create_unet_model.py / create_deep_back_projection_
# network_model.py directly by file path, bypassing the package __init__.py
# (so this script has no dependency on the rest of the antstorch package
# being importable). Load create_unet_model first: create_deep_back_
# projection_network_model.py does `from .create_unet_model import
# _Conv2dSame`, a relative import that resolves against the already-loaded
# module in sys.modules.
# ---------------------------------------------------------------------------

def _load_arch_modules(antstorch_src):
    arch_dir = os.path.join(antstorch_src, "antstorch", "architectures")
    pkg_name = "_antstorch_arch_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [arch_dir]
    sys.modules[pkg_name] = pkg

    def load(modname, filename):
        path = os.path.join(arch_dir, filename)
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{modname}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    cu = load("create_unet_model", "create_unet_model.py")
    cd = load("create_deep_back_projection_network_model", "create_deep_back_projection_network_model.py")
    return cu, cd


def natkey(s):
    """Full natural sort key -- see convert_wmh_bespoke.py for rationale."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _weights_root(f):
    if "layer_names" in f.attrs:
        return f
    if "model_weights" in f and "layer_names" in f["model_weights"].attrs:
        return f["model_weights"]
    raise KeyError("Could not find a `layer_names` attribute at root or under 'model_weights'.")


def h5_layer_names(path):
    with h5py.File(path, "r") as f:
        root = _weights_root(f)
        raw = root.attrs["layer_names"]
        return [n.decode() if isinstance(n, bytes) else n for n in raw]


def h5_get_wb(f, name, has_bias=True):
    root = _weights_root(f)
    g = root[f"{name}/{name}"]
    W = np.array(g["kernel:0"])
    b = np.array(g["bias:0"]) if (has_bias and "bias:0" in g) else None
    return W, b


def h5_get_prelu_alpha(f, name):
    root = _weights_root(f)
    g = root[f"{name}/{name}"]
    return np.array(g["alpha:0"])


def to_torch_conv_weight(W):
    """Keras (kD,kH,kW,inC,outC) or (kH,kW,inC,outC) -> torch (outC,inC,*kernel)."""
    if W.ndim == 5:
        return np.transpose(W, (4, 3, 0, 1, 2)).astype(np.float32)
    elif W.ndim == 4:
        return np.transpose(W, (3, 2, 0, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unexpected kernel ndim {W.ndim}")


def set_conv(sd, prefix, W, b, expect_bias=True):
    wkey = f"{prefix}.weight"
    torchW = to_torch_conv_weight(W)
    assert sd[wkey].shape == torch.Size(torchW.shape), (prefix, sd[wkey].shape, torchW.shape)
    sd[wkey] = torch.from_numpy(torchW)
    bkey = f"{prefix}.bias"
    if expect_bias:
        assert b is not None and bkey in sd, (prefix, "expected bias")
        assert sd[bkey].shape == torch.Size(b.shape), (prefix, sd[bkey].shape, b.shape)
        sd[bkey] = torch.from_numpy(b.astype(np.float32))
    else:
        assert b is None, (prefix, "unexpected bias in h5")


def set_prelu(sd, prefix, alpha, num_parameters):
    key = f"{prefix}.weight"
    a = alpha
    if a.ndim > 1:
        # Keras default PReLU alpha is per-element unless shared_axes was set
        # at training time (unknown here) -- reduce over all non-channel
        # (leading) axes to a single per-channel vector. Exact if
        # shared_axes=[1,2] (the usual convnet convention) was used; an
        # approximation (mean over spatial positions) otherwise. See the
        # DBPN confidence note in the module docstring.
        axes = tuple(range(a.ndim - 1))
        a = a.mean(axis=axes)
    a = a.reshape(-1).astype(np.float32)
    assert a.shape[0] == num_parameters, (prefix, a.shape, num_parameters)
    sd[key] = torch.from_numpy(a)


# ---------------------------------------------------------------------------
# Generic U-Net (create_unet_model_2d/3d) -- covers 19 of the 20 ids.
# ---------------------------------------------------------------------------

def _kernel_info(f, root, name, ndim):
    if name not in root:
        return None
    g = root[name]
    if name not in g:
        return None
    g2 = g[name]
    if "kernel:0" not in g2:
        return None
    shape = g2["kernel:0"].shape
    if len(shape) != ndim:
        return None
    return shape


def classify_conv_layers(h5_path, dimension):
    """Split h5 layer_names (construction order) into (deconv_names,
    regular_conv_names, pointwise_conv_names) using Keras's own auto-naming
    (conv{2,3}d vs conv{2,3}d_transpose) plus kernel-shape inspection
    (spatial dims all == 1 => 'pointwise': attention theta/phi/psi, the
    main output conv, and any auxiliary head convs)."""
    ndim = dimension + 2
    base = f"conv{dimension}d"
    transpose_re = re.compile(rf"^{base}_transpose(_\d+)?$")
    plain_re = re.compile(rf"^{base}(_\d+)?$")
    names = h5_layer_names(h5_path)
    deconv, regular, pointwise = [], [], []
    with h5py.File(h5_path, "r") as f:
        root = _weights_root(f)
        for n in names:
            if transpose_re.match(n):
                shape = _kernel_info(f, root, n, ndim)
                if shape is not None:
                    deconv.append(n)
            elif plain_re.match(n):
                shape = _kernel_info(f, root, n, ndim)
                if shape is not None:
                    if all(k == 1 for k in shape[:-2]):
                        pointwise.append(n)
                    else:
                        regular.append(n)
    return deconv, regular, pointwise


def infer_input_channel_size(h5_path, dimension, regular_names=None):
    """Infer input_channel_size from the first 'regular' (non-1x1) conv
    layer's Keras kernel shape -- used for the priors-dependent branches
    (protonLobes, maskLobes, the CT priors branch, mouse Jay parcellation)
    whose channel count depends on how many prior label images ship with
    the corresponding get_antstorch_data(...) template, which this script
    has no way to count independently."""
    if regular_names is None:
        _, regular_names, _ = classify_conv_layers(h5_path, dimension)
    with h5py.File(h5_path, "r") as f:
        root = _weights_root(f)
        name = regular_names[0]
        shape = root[name][name]["kernel:0"].shape
    return int(shape[-2])


def convert_generic_unet(h5_path, model, dimension, number_of_layers,
                          has_attention=False, base_prefix="", n_aux_heads=0):
    deconv_names, regular_names, pointwise_names = classify_conv_layers(h5_path, dimension)

    expected_deconv = number_of_layers - 1
    assert len(deconv_names) == expected_deconv, (h5_path, "deconv count", len(deconv_names), expected_deconv)

    expected_regular = 4 * number_of_layers - 2
    assert len(regular_names) == expected_regular, (h5_path, "regular conv count", len(regular_names), expected_regular)

    expected_attn = 3 * (number_of_layers - 1) if has_attention else 0
    expected_pointwise = expected_attn + 1 + n_aux_heads
    assert len(pointwise_names) == expected_pointwise, (h5_path, "pointwise conv count", len(pointwise_names), expected_pointwise)

    sd = {k: v.clone() for k, v in model.state_dict().items()}

    t_deconv = sorted(
        [k[:-len(".weight")] for k in sd if k.endswith(".weight") and "decoding_convolution_transpose_layers" in k],
        key=natkey,
    )
    assert len(t_deconv) == len(deconv_names), (h5_path, "torch deconv key count", len(t_deconv), len(deconv_names))

    t_enc = sorted(
        [k[:-len(".weight")] for k in sd if k.endswith(".weight") and "encoding_convolution_layers" in k],
        key=natkey,
    )
    t_dec = sorted(
        [k[:-len(".weight")] for k in sd if k.endswith(".weight") and "decoding_convolution_layers" in k],
        key=natkey,
    )
    t_regular = t_enc + t_dec
    assert len(t_regular) == len(regular_names), (h5_path, "torch regular-conv key count", len(t_regular), len(regular_names))

    attn_dim_key = f"attn_gates_{dimension}d"

    with h5py.File(h5_path, "r") as f:
        for tkey, hname in zip(t_deconv, deconv_names):
            W, b = h5_get_wb(f, hname)
            set_conv(sd, tkey, W, b, expect_bias=True)

        for tkey, hname in zip(t_regular, regular_names):
            W, b = h5_get_wb(f, hname)
            set_conv(sd, tkey, W, b, expect_bias=True)

        idx = 0
        if has_attention:
            for lvl in range(number_of_layers - 1):
                for role in ("theta", "phi", "psi"):
                    hname = pointwise_names[idx]; idx += 1
                    W, b = h5_get_wb(f, hname)
                    set_conv(sd, f"{base_prefix}{attn_dim_key}.{lvl}.{role}", W, b, expect_bias=True)

        hname = pointwise_names[idx]; idx += 1
        W, b = h5_get_wb(f, hname)
        set_conv(sd, f"{base_prefix}output.0", W, b, expect_bias=True)

        for i in range(n_aux_heads):
            hname = pointwise_names[idx]; idx += 1
            W, b = h5_get_wb(f, hname)
            set_conv(sd, f"heads.{i}", W, b, expect_bias=True)

        assert idx == len(pointwise_names)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (h5_path, missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# DBPN (allen_sr_weights only) -- explicit role list mirroring the ported
# torch module's __init__ order (see the confidence note in the module
# docstring: BEST EFFORT, not verified against a real .h5 in this session).
# ---------------------------------------------------------------------------

def _dbpn_roles(number_of_back_projection_stages=7):
    roles = [
        ("feature_extraction", "conv"), ("feature_extraction_act", "prelu"),
        ("smash", "conv"), ("smash_act", "prelu"),
        ("up_block_0.dense", "conv"), ("up_block_0.dense_act", "prelu"),
        ("up_block_0.up0", "deconv"), ("up_block_0.up0_act", "prelu"),
        ("up_block_0.down0", "conv"), ("up_block_0.down0_act", "prelu"),
        ("up_block_0.up1", "deconv"), ("up_block_0.up1_act", "prelu"),
    ]
    for i in range(number_of_back_projection_stages):
        roles += [
            (f"down_blocks.{i}.dense", "conv"), (f"down_blocks.{i}.dense_act", "prelu"),
            (f"down_blocks.{i}.down0", "conv"), (f"down_blocks.{i}.down0_act", "prelu"),
            (f"down_blocks.{i}.up0", "deconv"), (f"down_blocks.{i}.up0_act", "prelu"),
            (f"down_blocks.{i}.down1", "conv"), (f"down_blocks.{i}.down1_act", "prelu"),
        ]
        roles += [
            (f"up_blocks.{i}.dense", "conv"), (f"up_blocks.{i}.dense_act", "prelu"),
            (f"up_blocks.{i}.up0", "deconv"), (f"up_blocks.{i}.up0_act", "prelu"),
            (f"up_blocks.{i}.down0", "conv"), (f"up_blocks.{i}.down0_act", "prelu"),
            (f"up_blocks.{i}.up1", "deconv"), (f"up_blocks.{i}.up1_act", "prelu"),
        ]
    roles.append(("output_conv", "conv"))
    return roles


def convert_dbpn(h5_path, model, number_of_back_projection_stages=7):
    names = h5_layer_names(h5_path)
    conv_names = [n for n in names if re.match(r"^conv2d(_\d+)?$", n)]
    deconv_names = [n for n in names if re.match(r"^conv2d_transpose(_\d+)?$", n)]
    prelu_names = [n for n in names if re.match(r"^p_re_lu(_\d+)?$", n)]

    roles = _dbpn_roles(number_of_back_projection_stages)
    n_conv_roles = sum(1 for _, k in roles if k == "conv")
    n_deconv_roles = sum(1 for _, k in roles if k == "deconv")
    n_prelu_roles = sum(1 for _, k in roles if k == "prelu")
    assert len(conv_names) == n_conv_roles, ("conv count", len(conv_names), n_conv_roles)
    assert len(deconv_names) == n_deconv_roles, ("deconv count", len(deconv_names), n_deconv_roles)
    assert len(prelu_names) == n_prelu_roles, (
        "PReLU count -- if this fails, the h5 may not name PReLU layers "
        "'p_re_lu' as assumed; inspect h5_layer_names(h5_path) directly.",
        len(prelu_names), n_prelu_roles,
    )

    sd = {k: v.clone() for k, v in model.state_dict().items()}

    ci = di = pi = 0
    with h5py.File(h5_path, "r") as f:
        for role, kind in roles:
            if kind == "conv":
                W, b = h5_get_wb(f, conv_names[ci]); ci += 1
                set_conv(sd, role, W, b, expect_bias=True)
            elif kind == "deconv":
                W, b = h5_get_wb(f, deconv_names[di]); di += 1
                set_conv(sd, f"{role}.deconv", W, b, expect_bias=True)
            else:
                num_params = sd[f"{role}.weight"].shape[0]
                alpha = h5_get_prelu_alpha(f, prelu_names[pi]); pi += 1
                set_prelu(sd, role, alpha, num_params)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_and_save(model, out_path, build_fn, x, mode, multihead=False):
    model.eval()
    with torch.no_grad():
        y = model(x)

    if multihead:
        y_main, y_aux = y[0], y[1]
        s = y_main.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3), "main softmax not normalized"
        assert torch.all((y_aux >= 0) & (y_aux <= 1)), "aux sigmoid output out of [0,1]"
    elif mode == "sigmoid":
        assert torch.all((y >= 0) & (y <= 1)), "sigmoid output out of [0,1]"
    elif mode == "softmax":
        s = y.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3), "softmax not normalized"
    elif mode == "regression":
        pass
    else:
        raise ValueError(f"Unrecognized verify mode {mode!r}")

    torch.save(model.state_dict(), out_path)

    model2 = build_fn()
    if multihead:
        with torch.no_grad():
            _ = model2(x)  # warmup: instantiate the auxiliary head(s) before loading
    model2.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True), strict=True)
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)

    if multihead:
        assert torch.allclose(y[0], y2[0], atol=1e-6) and torch.allclose(y[1], y2[1], atol=1e-6), f"reload mismatch for {out_path}"
        shape = tuple(y[0].shape)
    else:
        assert torch.allclose(y, y2, atol=1e-6), f"reload mismatch for {out_path}"
        shape = tuple(y.shape)
    print(f"OK -> {out_path}  output shape {shape}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest():
    m = []

    # ---- no attention gating (high confidence) ----
    m.append(dict(stem="protonLungMri", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128), conv_ks=(7, 7, 5), deconv_ks=(7, 7, 5),
                   mode="classification", has_attention=False,
                   in_ch=1, out_ch=3, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="wholeLungMaskFromVentilation", kind="generic", dimension=2,
                   filters=(32, 64, 128, 256), conv_ks=(3, 3), deconv_ks=(2, 2),
                   mode="sigmoid", has_attention=False,
                   in_ch=1, out_ch=1, verify_spatial=(128, 128)))

    m.append(dict(stem="xrayLungExtraction", kind="generic", dimension=2,
                   filters=(32, 64, 128, 256), conv_ks=(3, 3), deconv_ks=(2, 2),
                   mode="classification", has_attention=False,
                   in_ch=3, out_ch=3, verify_spatial=(128, 128)))

    m.append(dict(stem="pulmonaryArteryWeights", kind="generic", dimension=3,
                   filters=(32, 64, 128, 256, 512), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="sigmoid", has_attention=False,
                   in_ch=1, out_ch=1, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="pulmonaryAirwayWeights", kind="generic", dimension=3,
                   filters=(32, 64, 128, 256, 512), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=False,
                   in_ch=1, out_ch=2, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="mouseT2wBrainExtraction3D", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="sigmoid", has_attention=False,
                   in_ch=1, out_ch=1, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="mouseT2wBrainParcellation3DNick", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128, 256), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=False,
                   in_ch=7, out_ch=7, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="mouseT2wBrainParcellation3DTct", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128, 256), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=False,
                   in_ch=8, out_ch=8, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="mouseSTPTBrainParcellation3DJay", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128, 256), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=False,
                   in_ch="infer", out_ch="same_as_in", verify_spatial=(64, 64, 64)))

    # ---- attention gating, single head (good confidence) ----
    m.append(dict(stem="maskLobes", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("attentionGating",),
                   in_ch="infer", out_ch="same_as_in", verify_spatial=(64, 64, 64)))

    m.append(dict(stem="lungCtWithPriorsSegmentationWeights", kind="generic", dimension=3,
                   filters=(16, 32, 64, 128), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("attentionGating",),
                   in_ch="infer", out_ch=4, verify_spatial=(64, 64, 64)))

    m.append(dict(stem="elBicho", kind="generic", dimension=2,
                   filters=(32, 64, 128, 256), conv_ks=(3, 3), deconv_ks=(2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("attentionGating",),
                   in_ch=2, out_ch=5, verify_spatial=(128, 128)))

    for stem in ("ex5_coronal_weights", "ex5_sagittal_weights"):
        m.append(dict(stem=stem, kind="generic", dimension=2,
                       filters=(64, 96, 128, 256, 512), conv_ks=(3, 3), deconv_ks=(2, 2),
                       mode="classification", has_attention=True,
                       additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
                       in_ch=1, out_ch=2, verify_spatial=(128, 128)))

    m.append(dict(stem="allen_brain_mask_weights", kind="generic", dimension=2,
                   filters=(64, 96, 128, 256, 512), conv_ks=(3, 3), deconv_ks=(2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
                   in_ch=1, out_ch=2, verify_spatial=(128, 128)))

    m.append(dict(stem="allen_brain_leftright_coronal_mask_weights", kind="generic", dimension=2,
                   filters=(64, 96, 128, 256, 512), conv_ks=(3, 3), deconv_ks=(2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
                   in_ch=1, out_ch=3, verify_spatial=(128, 128)))

    for stem in ("allen_cerebellum_sagittal_mask_weights", "allen_cerebellum_coronal_mask_weights"):
        m.append(dict(stem=stem, kind="generic", dimension=2,
                       filters=(64, 96, 128, 256, 512), conv_ks=(3, 3), deconv_ks=(2, 2),
                       mode="sigmoid", has_attention=True,
                       additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
                       in_ch=1, out_ch=1, verify_spatial=(128, 128)))

    # ---- attention gating + multihead (moderate confidence -- see docstring) ----
    m.append(dict(stem="protonLobes", kind="multihead", dimension=3,
                   filters=(16, 32, 64, 128), conv_ks=(3, 3, 3), deconv_ks=(2, 2, 2),
                   mode="classification", has_attention=True,
                   additional_options=("attentionGating",), n_aux_heads=1,
                   in_ch="infer", out_ch="same_as_in", verify_spatial=(64, 64, 64)))

    # ---- DBPN, distinct architecture (best effort -- see docstring) ----
    m.append(dict(stem="allen_sr_weights", kind="dbpn",
                   in_ch=3, out_ch=3, conv_ks=(6, 6), strides=(2, 2), stages=7,
                   verify_spatial=(32, 32)))

    return m


def build_generic(row, in_ch, out_ch, cu):
    ctor = cu.create_unet_model_3d if row["dimension"] == 3 else cu.create_unet_model_2d
    return ctor(
        input_channel_size=in_ch,
        number_of_outputs=out_ch,
        number_of_filters=row["filters"],
        convolution_kernel_size=row["conv_ks"],
        deconvolution_kernel_size=row["deconv_ks"],
        dropout_rate=0.0,
        mode=row["mode"],
        additional_options=row.get("additional_options"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", required=True, help="Directory with the ANTsPyNet .h5 files (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", required=True, help="Directory to write <stem>_pytorch.pt files (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", required=True, help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--only", default=None, help="Only convert this one h5 stem (e.g. protonLungMri)")
    args = p.parse_args()

    weights_dir = os.path.expanduser(args.weights_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cu, cd = _load_arch_modules(os.path.expanduser(args.antstorch_src))

    manifest = build_manifest()
    if args.only:
        manifest = [row for row in manifest if row["stem"] == args.only]
        if not manifest:
            print(f"No manifest entry named {args.only!r}")
            return

    results = {}
    for row in manifest:
        stem = row["stem"]
        h5_path = os.path.join(weights_dir, f"{stem}.h5")
        out_path = os.path.join(out_dir, f"{stem}_pytorch.pt")
        if not os.path.exists(h5_path):
            print(f"[skip] {stem}: {h5_path} not found")
            continue
        try:
            if row["kind"] == "dbpn":
                def build(row=row):
                    return cd.create_deep_back_projection_network_model_2d(
                        input_channel_size=row["in_ch"], number_of_outputs=row["out_ch"],
                        convolution_kernel_size=row["conv_ks"], strides=row["strides"],
                        number_of_back_projection_stages=row["stages"])
                model = build()
                convert_dbpn(h5_path, model, number_of_back_projection_stages=row["stages"])
                x = torch.randn(1, row["in_ch"], *row["verify_spatial"])
                verify_and_save(model, out_path, build, x, mode="regression")

            else:
                dimension = row["dimension"]
                in_ch = row["in_ch"]
                if in_ch == "infer":
                    in_ch = infer_input_channel_size(h5_path, dimension)
                    print(f"[{stem}] inferred input_channel_size={in_ch} from h5")
                out_ch = row["out_ch"]
                if out_ch == "same_as_in":
                    out_ch = in_ch

                vmode = {"classification": "softmax", "sigmoid": "sigmoid", "regression": "regression"}[row["mode"]]

                if row["kind"] == "generic":
                    def build(row=row, in_ch=in_ch, out_ch=out_ch):
                        return build_generic(row, in_ch, out_ch, cu)
                    model = build()
                    number_of_layers = len(row["filters"])
                    convert_generic_unet(h5_path, model, dimension, number_of_layers,
                                          has_attention=row["has_attention"])
                    x = torch.randn(1, in_ch, *row["verify_spatial"])
                    verify_and_save(model, out_path, build, x, mode=vmode)

                elif row["kind"] == "multihead":
                    def build(row=row, in_ch=in_ch, out_ch=out_ch):
                        base = build_generic(row, in_ch, out_ch, cu)
                        return cu.create_multihead_unet_model_3d(
                            base_unet=base, n_aux_heads=row["n_aux_heads"],
                            use_sigmoid=True, n_main_outputs=out_ch)
                    model = build()
                    x = torch.randn(1, in_ch, *row["verify_spatial"])
                    with torch.no_grad():
                        _ = model(x)  # warmup: instantiate the auxiliary head(s)
                    number_of_layers = len(row["filters"])
                    convert_generic_unet(h5_path, model, dimension, number_of_layers,
                                          has_attention=row["has_attention"],
                                          base_prefix="base.", n_aux_heads=row["n_aux_heads"])
                    verify_and_save(model, out_path, build, x, mode=vmode, multihead=True)

                else:
                    raise ValueError(row["kind"])

            results[stem] = "OK"
        except Exception as e:
            results[stem] = f"FAILED: {e!r}"
            print(f"[{stem}] FAILED: {e!r}")

    print("\n==== SUMMARY ====")
    for stem, status in results.items():
        print(f"{stem}: {status}")


if __name__ == "__main__":
    main()
