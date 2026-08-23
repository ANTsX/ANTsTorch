#!/usr/bin/env python3
"""
convert_hippmapp3r_hypothalamus_claustrum_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for the 3 high-confidence applications ported 2026-08-22:

    * create_hippmapp3r_unet_model_3d   (hippMapp3rInitial, hippMapp3rRefine)
    * create_hypothalamus_unet_model_3d (hypothalamus)
    * create_sysu_media_unet_model_2d(anatomy="claustrum")
                                         (claustrum_axial_{0,1,2},
                                          claustrum_coronal_{0,1,2})

This reuses the low-level h5/torch plumbing from convert_wmh_bespoke.py
(h5_layer_names, h5_get_wb, h5_get_norm, set_conv, set_instance_norm,
set_batch_norm, verify_and_save, natkey, _load_arch_modules) by importing it
directly -- it must live in the same directory as this script. claustrum
needs NO new conversion logic at all: it reuses convert_wmh_bespoke.py's
own convert_sysu_media() unchanged (that function derives its expected conv
order straight from the built model's own state_dict keys, so it already
works for any anatomy/filter configuration of create_sysu_media_unet_model_2d,
not just anatomy="wmh").

hippmapp3r and hypothalamus need new role-position logic since they are new
architectures:

  * hypothalamus -- UPDATE 2026-08-22, confirmed against Nick's real
    hypothalamus.h5: this file is NOT a plain save_weights() of the
    inference U-Net with Keras auto-naming ("conv3d"/"conv3d_N") as
    originally assumed (and as held for every other branch-free
    architecture converted this session, e.g. `lung_proton`). It is the
    FULL TRAINING GRAPH from the original
    https://github.com/BBillot/hypothalamus_seg implementation --
    augmentation layers (bool_flip/lambda_*/spatial_transformer_1/
    resize_*) and dice-loss layers bracket the actual U-Net, and every
    layer (including the U-Net's own convs/BNs) was given an EXPLICIT
    name by that training script: `unet_conv_downarm_{i}_{0,1}`,
    `unet_bn_down_{i}`, `unet_conv_uparm_{3,4}_{0,1}`, `unet_bn_up_{0,1}`,
    `unet_likelihood`. convert_hypothalamus() below now matches these
    exact names directly (no positional/regex guessing at all for this
    one) -- see its own docstring-comment block for the full mapping.
    Confirmed high-confidence: matching by exact string name has no
    topological-sort-divergence risk, unlike hippmapp3r below.

  * hippmapp3r shares its "back64"/"back32" side-output pattern with
    hypermapp3r (verified, in convert_wmh_bespoke.py): a side-branch conv
    whose output feeds an Add() used later in the graph gets positioned by
    Keras's functional-API topological sort *right before that Add()*, not
    at its literal point of creation in the Python source -- confirmed
    against a real hyperMapp3r.h5 file. hippmapp3r's decoder tail
    (final_conv1 -> final_conv2 -> final-output-conv, with back64/back32
    Add()'d in) is structurally IDENTICAL to hypermapp3r's tail (same 3
    final blocks, same 2 side branches feeding the same 2 Add()s), so
    _hippmapp3r_roles() below places back64_conv/back32_conv by direct
    analogy to hypermapp3r's *verified* real order (between final_conv1/
    final_conv2, and between final_conv2/final_out_conv, respectively) --
    see antspynet's create_hippmapp3r_unet_model_3d source for the literal
    (different) call order this deliberately does NOT follow.

    ⚠️  UNVERIFIED AGAINST A REAL FILE: no hippMapp3rInitial.h5 /
    hippMapp3rRefine.h5 was available in the sandbox this converter was
    written in. The role order is high-confidence by structural analogy,
    not confirmed the way every other converter in this session's toolkit
    was. Run this against your real hippMapp3rInitial.h5/hippMapp3rRefine.h5
    with --only hippMapp3rInitial first and inspect the result carefully
    (or, if load_state_dict/verify_and_save raises an AssertionError, that
    itself is useful signal that the position guess needs correcting) before
    trusting it for production use.

Usage (same convention as convert_wmh_bespoke.py / convert_lung_mouse_bespoke.py):

    python convert_hippmapp3r_hypothalamus_claustrum_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch

    # convert just one file:
    python convert_hippmapp3r_hypothalamus_claustrum_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch --only hippMapp3rInitial
"""
import argparse
import os
import re
import sys

import h5py
import torch

# Reuse the low-level h5/torch plumbing from convert_wmh_bespoke.py -- must
# live alongside this script (both under tools/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_wmh_bespoke import (
    _load_arch_modules,
    h5_layer_names,
    h5_get_wb,
    h5_get_norm,
    set_conv,
    set_instance_norm,
    set_batch_norm,
    verify_and_save,
    convert_sysu_media,
)


# ---------------------------------------------------------------------------
# hippmapp3r: explicit role list, in (believed) h5 positional order -- see
# module docstring above for the derivation and the unverified-confidence
# caveat.
# ---------------------------------------------------------------------------

def _hippmapp3r_roles(do_first_network):
    """(torch_prefix, has_norm) pairs in believed h5 layer_names order.
    has_norm selects between a _ConvBBlock3D wrapper (conv+InstanceNorm,
    prefix gets '.conv'/'.norm' appended) and a bare conv (prefix used
    directly) -- matching antstorch's create_hippmapp3r_unet_model_3d,
    where do_first_network=True wraps every tail conv in _ConvBBlock3D and
    do_first_network=False leaves back64_conv/back32_conv/final_out_conv
    as plain (unwrapped) convs."""
    roles = []
    number_of_layers = 6 if do_first_network else 5
    for i in range(number_of_layers):
        roles.append((f"encoding_conv.{i}", True))
        roles.append((f"encoding_residual.{i}.conv1", True))
        roles.append((f"encoding_residual.{i}.conv2", True))
    roles.append(("up_top.conv", True))
    if do_first_network:
        roles.append(("feature_extra.conv1", True))
        roles.append(("feature_extra.conv2", True))
        roles.append(("up_extra.conv", True))
    roles.append(("feature3.conv1", True))
    roles.append(("feature3.conv2", True))
    roles.append(("up3.conv", True))
    roles.append(("feature64.conv1", True))
    roles.append(("feature64.conv2", True))
    roles.append(("up64.conv", True))
    roles.append(("feature32.conv1", True))
    roles.append(("feature32.conv2", True))
    roles.append(("up32.conv", True))
    # Tail, in *believed* real order -- see module docstring.
    roles.append(("final_conv1", True))
    roles.append(("back64_conv", do_first_network))
    roles.append(("final_conv2", True))
    roles.append(("back32_conv", do_first_network))
    roles.append(("final_out_conv", do_first_network))
    return roles


def convert_hippmapp3r(h5_path, model, do_first_network):
    names = h5_layer_names(h5_path)
    conv_names = [n for n in names if re.match(r"^conv3d(_\d+)?$", n)]
    norm_names = [n for n in names if re.match(r"^instance_normalization(_\d+)?$", n)]

    roles = _hippmapp3r_roles(do_first_network)
    assert len(conv_names) == len(roles), (len(conv_names), len(roles))
    n_norm_roles = sum(1 for _, has_norm in roles if has_norm)
    assert len(norm_names) == n_norm_roles, (len(norm_names), n_norm_roles)

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with h5py.File(h5_path, "r") as f:
        norm_idx = 0
        for (prefix, has_norm), hname in zip(roles, conv_names):
            if has_norm:
                W, b = h5_get_wb(f, hname)
                set_conv(sd, f"{prefix}.conv", W, b, expect_bias=True)
                gamma, beta = h5_get_norm(f, norm_names[norm_idx], "instance")
                norm_idx += 1
                set_instance_norm(sd, f"{prefix}.norm", gamma, beta)
            else:
                W, b = h5_get_wb(f, hname)
                set_conv(sd, prefix, W, b, expect_bias=True)
        assert norm_idx == len(norm_names)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# hypothalamus: NOT the generic-auto-naming case assumed originally.
#
# Confirmed 2026-08-22 against Nick's real hypothalamus.h5 (root-level
# layer_names, 81 entries): this .h5 is not a bare save_weights() of the
# inference U-Net antspynet's create_hypothalamus_unet_model_3d docstring
# describes -- it is the FULL training graph from the original
# https://github.com/BBillot/hypothalamus_seg implementation, including
# augmentation layers (bool_flip/switch_idx_flipping/lambda_*/
# spatial_transformer_1/resize_*), and dice-loss layers appended at the
# end (dice/dice_loss/mean_dice_loss/average_mean_dice_loss). None of the
# convolutional layers are named "conv3d"/"conv3d_N" (Keras auto-naming) --
# every layer in this graph was given an EXPLICIT name by the training
# script, so the regex-based extraction used for every other converter in
# this toolkit (based on the "conv3d_N" auto-naming, positionally ordered
# via `layer_names`) does not apply here and matched zero layers.
#
# The inference-relevant subset is unambiguous by name alone, no ordering
# guesswork needed at all:
#   unet_conv_downarm_{i}_{0,1}  -- 2 convs per encoder level i=0,1,2
#   unet_bn_down_{i}             -- 1 BN per encoder level i=0,1,2 (after
#                                    the 2nd conv, matching antspynet's
#                                    "conv = BatchNormalization()(conv)"
#                                    applied at every level incl. bottleneck)
#   unet_conv_uparm_{3,4}_{0,1}  -- 2 convs per decoder level (2 levels)
#   unet_bn_up_{0,1}             -- 1 BN per decoder level
#   unet_likelihood              -- final 1x1x1 conv (11 channels, softmax
#                                    baked into the Keras activation kwarg)
#   unet_prediction               -- no learnable weights (not extracted)
# Verified: 11 conv names + 5 bn names above == the 11/5 counts already
# expected from antstorch's create_hypothalamus_unet_model_3d topology
# (see _HYPOTHALAMUS_CONV_ORDER/_HYPOTHALAMUS_BN_AFTER below), and their
# relative order in Nick's real layer_names list matches this topology
# exactly. Because the h5 names are matched by EXACT STRING, not position,
# this mapping has no topological-sort-divergence risk at all (unlike the
# hippmapp3r side-branch situation below).
# ---------------------------------------------------------------------------

_HYPOTHALAMUS_H5_CONV_NAMES = (
    "unet_conv_downarm_0_0", "unet_conv_downarm_0_1",
    "unet_conv_downarm_1_0", "unet_conv_downarm_1_1",
    "unet_conv_downarm_2_0", "unet_conv_downarm_2_1",
    "unet_conv_uparm_3_0", "unet_conv_uparm_3_1",
    "unet_conv_uparm_4_0", "unet_conv_uparm_4_1",
    "unet_likelihood",
)
_HYPOTHALAMUS_H5_BN_NAMES = (
    "unet_bn_down_0", "unet_bn_down_1", "unet_bn_down_2",
    "unet_bn_up_0", "unet_bn_up_1",
)

# Each conv_elu() block in antstorch's create_hypothalamus_unet_model_3d is
# an nn.Sequential(_Conv3dSame, nn.ELU()), so the actual conv weight lives
# one level deeper at "<block>.0" -- except output_conv, which is a bare
# _Conv3dSame (no ELU/Sequential wrapper, since the final activation is a
# separate nn.Softmax).
_HYPOTHALAMUS_CONV_ORDER = (
    "encoding_conv1.0.0", "encoding_conv2.0.0",
    "encoding_conv1.1.0", "encoding_conv2.1.0",
    "encoding_conv1.2.0", "encoding_conv2.2.0",
    "decoding_conv1.0.0", "decoding_conv2.0.0",
    "decoding_conv1.1.0", "decoding_conv2.1.0",
    "output_conv",
)
_HYPOTHALAMUS_BN_AFTER = (
    "encoding_bn.0", "encoding_bn.1", "encoding_bn.2",
    "decoding_bn.0", "decoding_bn.1",
)


def convert_hypothalamus(h5_path, model):
    assert len(_HYPOTHALAMUS_H5_CONV_NAMES) == len(_HYPOTHALAMUS_CONV_ORDER)
    assert len(_HYPOTHALAMUS_H5_BN_NAMES) == len(_HYPOTHALAMUS_BN_AFTER)

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with h5py.File(h5_path, "r") as f:
        for h5_name, prefix in zip(_HYPOTHALAMUS_H5_CONV_NAMES, _HYPOTHALAMUS_CONV_ORDER):
            W, b = h5_get_wb(f, h5_name)
            set_conv(sd, prefix, W, b, expect_bias=True)
        for h5_name, prefix in zip(_HYPOTHALAMUS_H5_BN_NAMES, _HYPOTHALAMUS_BN_AFTER):
            gamma, beta, mean, var = h5_get_norm(f, h5_name, "batch")
            set_batch_norm(sd, prefix, gamma, beta, mean, var)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# Manifest: (h5 filename stem, architecture, kwargs)
# .h5 stems confirmed against the real antspynet get_pretrained_network.py
# switcher dict (id -> default target_file_name = id + ".h5"), fetched
# 2026-08-22 from https://github.com/ANTsX/ANTsPyNet.
# ---------------------------------------------------------------------------

def build_manifest():
    m = []
    m.append(("hippMapp3rInitial", "hippmapp3r", dict(input_channel_size=1, do_first_network=True)))
    m.append(("hippMapp3rRefine", "hippmapp3r", dict(input_channel_size=1, do_first_network=False)))
    m.append(("hypothalamus", "hypothalamus", dict(input_channel_size=1, number_of_outputs=11)))
    for i in range(3):
        m.append((f"claustrum_axial_{i}", "sysu2d", dict(input_channel_size=1, anatomy="claustrum")))
    for i in range(3):
        m.append((f"claustrum_coronal_{i}", "sysu2d", dict(input_channel_size=1, anatomy="claustrum")))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", required=True, help="Directory with the ANTsPyNet .h5 files (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", required=True, help="Directory to write <prefix>_pytorch.pt files (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", required=True, help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--only", default=None, help="Only convert this one h5 stem (e.g. hippMapp3rInitial)")
    args = p.parse_args()

    weights_dir = os.path.expanduser(args.weights_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ccu = _load_arch_modules(os.path.expanduser(args.antstorch_src))

    manifest = build_manifest()
    if args.only:
        manifest = [row for row in manifest if row[0] == args.only]
        if not manifest:
            print(f"No manifest entry named {args.only!r}")
            return

    results = {}
    for stem, arch, kwargs in manifest:
        h5_path = os.path.join(weights_dir, f"{stem}.h5")
        out_path = os.path.join(out_dir, f"{stem}_pytorch.pt")
        if not os.path.exists(h5_path):
            print(f"[skip] {stem}: {h5_path} not found")
            continue
        try:
            if arch == "hippmapp3r":
                do_first_network = kwargs["do_first_network"]
                def build(kwargs=kwargs):
                    return ccu.create_hippmapp3r_unet_model_3d(**kwargs)
                model = build()
                convert_hippmapp3r(h5_path, model, do_first_network)
                shape = (160, 160, 128) if do_first_network else (112, 112, 64)
                x = torch.randn(1, kwargs["input_channel_size"], *shape)
                verify_and_save(model, out_path, build, x, mode="sigmoid")
            elif arch == "hypothalamus":
                def build(kwargs=kwargs):
                    return ccu.create_hypothalamus_unet_model_3d(**kwargs)
                model = build()
                convert_hypothalamus(h5_path, model)
                x = torch.randn(1, kwargs["input_channel_size"], 64, 64, 64)
                verify_and_save(model, out_path, build, x, mode="softmax")
            elif arch == "sysu2d":
                def build(kwargs=kwargs):
                    return ccu.create_sysu_media_unet_model_2d(**kwargs)
                model = build()
                convert_sysu_media(h5_path, model, dimension=2)
                x = torch.randn(1, kwargs["input_channel_size"], 180, 180)
                verify_and_save(model, out_path, build, x, mode="sigmoid")
            else:
                raise ValueError(arch)
            results[stem] = "OK"
        except Exception as e:
            results[stem] = f"FAILED: {e!r}"
            print(f"[{stem}] FAILED: {e!r}")

    print("\n==== SUMMARY ====")
    for stem, status in results.items():
        print(f"{stem}: {status}")


if __name__ == "__main__":
    main()
