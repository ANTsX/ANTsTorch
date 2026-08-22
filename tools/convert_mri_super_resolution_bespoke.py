#!/usr/bin/env python3
"""
convert_mri_super_resolution_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for the 11 SIQ deep back-projection network (DBPN) MRI
super-resolution weights used by antstorch.utilities.mri_super_resolution:

    sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl
    sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl
    sig_smallshort_train_1x1x3_1chan_featgraderL6_best_mdl
    sig_smallshort_train_1x1x3_1chan_featvggL6_best_mdl
    sig_smallshort_train_1x1x4_1chan_featgraderL6_best_mdl
    sig_smallshort_train_1x1x4_1chan_featvggL6_best_mdl
    sig_smallshort_train_1x1x6_1chan_featvggL6_best_mdl      (no grader variant)
    sig_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl
    sig_smallshort_train_2x2x2_1chan_featvggL6_best_mdl
    sig_smallshort_train_2x2x4_1chan_featgraderL6_best_mdl
    sig_smallshort_train_2x2x4_1chan_featvggL6_best_mdl

IMPORTANT ARCHITECTURE CORRECTION (2026-08-22): these weights are NOT
compatible with antstorch's create_deep_back_projection_network_model_3d
(the ConvTranspose-based DBPN already in this repo, used for
mouse_histology_super_resolution). Reading the real training code directly
(https://github.com/stnava/siq, siq/get_data.py: dbpn()/default_dbpn())
shows the SIQ models scale up via UpSampling3D (fixed nearest-neighbor
resize) + a plain stride-1 Conv3D(kernel_size=3), NOT a learned
Conv3DTranspose -- a structurally different set of learnable tensors, not
just a different hyperparameter choice. This converter targets the new
`create_siq_dbpn_super_resolution_model_3d` class added to
antstorch/architectures/create_deep_back_projection_network_model.py
2026-08-22 specifically to match this real op-for-op (see that class's
docstring for the full derivation).

METHOD -- config-driven, no hardcoded hyperparameter table: each of these
.h5 files is a FULL Keras model save (`model.save()`, `compile=False`
loaded via `tf.keras.models.load_model` in ANTsPyNet), which embeds the
entire architecture as a `model_config` JSON attribute alongside the
weights. Rather than guessing number_of_base_filters/number_of_feature_
filters/number_of_back_projection_stages/convolution_kernel_size (siq's
default_dbpn() offers multiple named presets -- "large"/"small"/"tiny" --
with different filter counts and stage counts, and it is not confirmed
which preset "smallshort" corresponds to), this script reads them directly
out of each file's own model_config at conversion time:

  * Conv3D layer order in model_config['config']['layers'] is the same
    order Keras used to build the weight file's own `layer_names` (both
    are derived from `model.layers`), so no separate layer_names parsing
    or positional-order guessing is needed for topology -- the config
    *is* the ground truth for both weight-file order AND every
    hyperparameter (filters, kernel_size, strides) simultaneously.
  * The first 2 Conv3D layers are always feature_extraction (kernel=3)
    then smash (kernel=1) (see the class docstring above); the last is
    always output_conv; everything between comes in stride-4 quads
    corresponding to the DBPN blocks (up_block_0 first, then
    (down_block, up_block) pairs for the loop stages) -- see
    _extract_roles() below.
  * Each quad's TYPE (up-block vs down-block) is detected from its 2nd
    conv's stride: (1,1,1) means it's the "up0" resize-then-conv (an
    up-block), anything else means it's the strided "down0" (a
    down-block) -- this works regardless of the actual filter/kernel/
    stage counts, so no table is needed at all.

UNVERIFIED END-TO-END: no real sig_smallshort_train_*.h5 was available to
test this against at write time (this script was authored purely from
reading real ANTsPyNet/siq source, the same method used successfully for
convert_hippmapp3r_hypothalamus_claustrum_bespoke.py's hypothalamus fix).
The role-position logic has ONLY been checked with a synthetic round-trip
test (fabricated fake .h5 + model_config matching the hypothesized
layout -- proves the extraction code is self-consistent, not that a real
SIQ file matches this layout). Run this against your real .h5 files with
--only <one_small_id> first (recommend the smallest/fastest:
sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl) and inspect the
result -- if load_state_dict/verify_and_save raises, that itself is a
useful signal about what needs correcting (most likely: the quad-grouping
assumption, or a different default_dbpn() preset than assumed).

Usage (same convention as the other bespoke converters in tools/):

    python convert_mri_super_resolution_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch

    # convert just one file (recommended first):
    python convert_mri_super_resolution_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch \\
        --only sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import h5py
import numpy as np
import torch

# Reuse the low-level h5/torch plumbing already validated in
# convert_wmh_bespoke.py -- must live alongside this script (both under
# tools/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_wmh_bespoke import _weights_root, h5_get_wb, set_conv


# ---------------------------------------------------------------------------
# Import antstorch's create_deep_back_projection_network_model.py directly
# by file path (it in turn imports _Conv3dSame from create_unet_model.py in
# the same directory, so that has to be loaded first too), bypassing the
# package __init__.py.
# ---------------------------------------------------------------------------

def _load_dbpn_module(antstorch_src):
    arch_dir = os.path.join(antstorch_src, "antstorch", "architectures")
    pkg_name = "_antstorch_dbpn_pkg"
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

    load("create_unet_model", "create_unet_model.py")
    dbpn = load("create_deep_back_projection_network_model", "create_deep_back_projection_network_model.py")
    return dbpn


# ---------------------------------------------------------------------------
# model_config parsing
# ---------------------------------------------------------------------------

def _model_config(f):
    """A full Keras `model.save()` .h5 stores the whole-model JSON config as
    a root-level `model_config` attribute (distinct from `model_weights`,
    which holds only the weight arrays + their own `layer_names`)."""
    if "model_config" in f.attrs:
        raw = f.attrs["model_config"]
    elif "model_weights" in f and "model_config" in f["model_weights"].attrs:
        raw = f["model_weights"].attrs["model_config"]
    else:
        raise KeyError(
            "No `model_config` attribute found at root or under 'model_weights' -- "
            "this .h5 does not look like a full tf.keras.models.load_model()-compatible "
            "save (expected for a SIQ model)."
        )
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _config_layers(config):
    try:
        return config["config"]["layers"]
    except KeyError:
        raise KeyError("model_config JSON did not have the expected ['config']['layers'] structure.")


def h5_get_prelu_alpha(f, name):
    root = _weights_root(f)
    g = root[f"{name}/{name}"]
    alpha = np.array(g["alpha:0"])
    return alpha.reshape(-1)  # Keras shared_axes=[1,2,3] -> effectively per-channel


def set_prelu(sd, prefix, alpha):
    key = f"{prefix}.weight"
    assert sd[key].shape == torch.Size(alpha.shape), (prefix, sd[key].shape, alpha.shape)
    sd[key] = torch.from_numpy(alpha.astype(np.float32))


# ---------------------------------------------------------------------------
# Role extraction -- entirely config-driven, no hardcoded filter/stage
# counts. See module docstring for the derivation.
# ---------------------------------------------------------------------------

def _extract_roles(config):
    layers = _config_layers(config)
    conv_layers = []
    prelu_layers = []
    for layer in layers:
        cls = layer.get("class_name")
        cfg = layer.get("config", {})
        if cls == "Conv3D":
            conv_layers.append(dict(
                name=cfg["name"],
                filters=cfg["filters"],
                kernel_size=tuple(cfg["kernel_size"]),
                strides=tuple(cfg["strides"]),
            ))
        elif cls == "PReLU":
            prelu_layers.append(cfg["name"])

    assert len(conv_layers) >= 4, f"too few Conv3D layers found ({len(conv_layers)})"
    feature_extraction, smash = conv_layers[0], conv_layers[1]
    output_conv = conv_layers[-1]
    middle = conv_layers[2:-1]
    assert len(middle) % 4 == 0, (
        f"expected the {len(middle)} conv layers between smash and output_conv to be a "
        "multiple of 4 (one quad per DBPN block) -- role-extraction assumption failed, "
        "see module docstring."
    )
    n_quads = len(middle) // 4
    quads = [middle[i * 4:(i + 1) * 4] for i in range(n_quads)]

    def quad_type(q):
        return "up" if tuple(q[1]["strides"]) == (1, 1, 1) else "down"

    types_seq = [quad_type(q) for q in quads]
    assert types_seq[0] == "up", f"expected the first block to be the initial up-block, got {types_seq[0]!r}"
    rest = types_seq[1:]
    assert len(rest) % 2 == 0, f"expected (down, up) pairs after the initial up-block, got odd count {len(rest)}"
    number_of_back_projection_stages = len(rest) // 2
    for i in range(number_of_back_projection_stages):
        pair = rest[2 * i:2 * i + 2]
        assert pair == ["down", "up"], f"unexpected block-type pattern at stage {i}: {pair}"

    # kernel_size/strides for the strided convs -- read from up_block_0's
    # own down0 (quads[0][2], always present since quads[0] is an up-type
    # quad: dense, up0, down0, up1).
    convolution_kernel_size = quads[0][2]["kernel_size"]
    strides = quads[0][2]["strides"]

    roles = []  # (torch_prefix, h5_conv_name, h5_prelu_name_or_None)
    roles.append(("feature_extraction", feature_extraction["name"], prelu_layers[0]))
    roles.append(("smash", smash["name"], prelu_layers[1]))

    idx = 2  # parallel index into (conv order, prelu order) -- 1:1 for every conv except output_conv
    for qi, (q, qtype) in enumerate(zip(quads, types_seq)):
        if qi == 0:
            prefix = "up_block_0"
        else:
            stage = (qi - 1) // 2
            prefix = f"down_blocks.{stage}" if qtype == "down" else f"up_blocks.{stage}"
        if qtype == "up":
            sub = [("dense", q[0]), ("up0.conv", q[1]), ("down0", q[2]), ("up1.conv", q[3])]
        else:
            sub = [("dense", q[0]), ("down0", q[1]), ("up0.conv", q[2]), ("down1", q[3])]
        for suffix, conv_entry in sub:
            roles.append((f"{prefix}.{suffix}", conv_entry["name"], prelu_layers[idx]))
            idx += 1

    roles.append(("output_conv", output_conv["name"], None))

    kwargs = dict(
        number_of_feature_filters=feature_extraction["filters"],
        number_of_base_filters=smash["filters"],
        number_of_back_projection_stages=number_of_back_projection_stages,
        convolution_kernel_size=convolution_kernel_size,
        strides=strides,
        last_convolution=output_conv["kernel_size"],
    )
    return roles, kwargs


def convert_mri_super_resolution(h5_path, dbpn_module):
    with h5py.File(h5_path, "r") as f:
        config = _model_config(f)
        roles, kwargs = _extract_roles(config)

        model = dbpn_module.create_siq_dbpn_super_resolution_model_3d(
            input_channel_size=1, number_of_outputs=1, **kwargs)
        sd = {k: v.clone() for k, v in model.state_dict().items()}

        for prefix, conv_name, prelu_name in roles:
            W, b = h5_get_wb(f, conv_name)
            set_conv(sd, prefix, W, b, expect_bias=True)
            if prelu_name is not None:
                alpha = h5_get_prelu_alpha(f, prelu_name)
                set_prelu(sd, _act_prefix(prefix), alpha)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model, kwargs


def _act_prefix(conv_prefix):
    """dense -> dense_act, up0.conv -> up0_act, down0 -> down0_act, etc."""
    if conv_prefix.endswith(".conv"):
        return conv_prefix[: -len(".conv")] + "_act"
    return conv_prefix + "_act"


def save_and_verify(model, kwargs, build_fn, out_path, x):
    """Unlike the other bespoke converters' verify_and_save() (which saves a
    bare state_dict), this saves {"state_dict": ..., "architecture_kwargs":
    ...} -- because number_of_base_filters/number_of_feature_filters/
    number_of_back_projection_stages are only known after reading each
    file's own model_config (see module docstring), mri_super_resolution()
    needs them persisted alongside the weights rather than hardcoded in a
    lookup table. antstorch.utilities.mri_super_resolution's
    _load_state_dict() helper already unwraps a "state_dict" key when
    present, so this format is backward compatible with that existing
    loading code -- only the addition of reading "architecture_kwargs" is
    new."""
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "output contains non-finite values"

    payload = {"state_dict": model.state_dict(), "architecture_kwargs": kwargs}
    torch.save(payload, out_path)

    model2 = build_fn()
    loaded = torch.load(out_path, map_location="cpu", weights_only=True)
    model2.load_state_dict(loaded["state_dict"], strict=True)
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y, y2, atol=1e-6), f"reload mismatch for {out_path}"
    print(f"OK -> {out_path}  output shape {tuple(y.shape)}  kwargs={kwargs}")


# ---------------------------------------------------------------------------
# Manifest: the 11 real sig_smallshort ids, confirmed 2026-08-22 directly
# against antspynet's own get_pretrained_network.py switcher dict on GitHub
# (these already have real figshare URLs there for the ORIGINAL .h5 -- see
# tools/download_antspynet_h5_weights.py).
# ---------------------------------------------------------------------------

def build_manifest():
    ids = [
        "sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl",
        "sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl",
        "sig_smallshort_train_1x1x3_1chan_featgraderL6_best_mdl",
        "sig_smallshort_train_1x1x3_1chan_featvggL6_best_mdl",
        "sig_smallshort_train_1x1x4_1chan_featgraderL6_best_mdl",
        "sig_smallshort_train_1x1x4_1chan_featvggL6_best_mdl",
        "sig_smallshort_train_1x1x6_1chan_featvggL6_best_mdl",
        "sig_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl",
        "sig_smallshort_train_2x2x2_1chan_featvggL6_best_mdl",
        "sig_smallshort_train_2x2x4_1chan_featgraderL6_best_mdl",
        "sig_smallshort_train_2x2x4_1chan_featvggL6_best_mdl",
    ]
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", required=True, help="Directory with the ANTsPyNet .h5 files (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", required=True, help="Directory to write <id>_pytorch.pt files (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", required=True, help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--only", default=None, help="Only convert this one h5 stem")
    args = p.parse_args()

    weights_dir = os.path.expanduser(args.weights_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dbpn_module = _load_dbpn_module(os.path.expanduser(args.antstorch_src))

    manifest = build_manifest()
    if args.only:
        manifest = [s for s in manifest if s == args.only]
        if not manifest:
            print(f"No manifest entry named {args.only!r}")
            return

    results = {}
    for stem in manifest:
        h5_path = os.path.join(weights_dir, f"{stem}.h5")
        out_path = os.path.join(out_dir, f"{stem}_pytorch.pt")
        if not os.path.exists(h5_path):
            print(f"[skip] {stem}: {h5_path} not found")
            continue
        try:
            model, kwargs = convert_mri_super_resolution(h5_path, dbpn_module)
            print(f"[{stem}] discovered config from model_config JSON: {kwargs}")

            def build(kwargs=kwargs):
                return dbpn_module.create_siq_dbpn_super_resolution_model_3d(
                    input_channel_size=1, number_of_outputs=1, **kwargs)

            x = torch.randn(1, 1, 24, 24, 24)
            save_and_verify(model, kwargs, build, out_path, x)
            results[stem] = "OK"
        except Exception as e:
            results[stem] = f"FAILED: {e!r}"
            print(f"[{stem}] FAILED: {e!r}")

    print("\n==== SUMMARY ====")
    for stem, status in results.items():
        print(f"{stem}: {status}")


if __name__ == "__main__":
    main()
