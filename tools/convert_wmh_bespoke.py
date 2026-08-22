#!/usr/bin/env python3
"""
convert_wmh_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for the 4 "bespoke" white_matter_hyperintensity_segmentation
architectures that convert_antspynet_weights_to_antstorch.py does NOT
support (they are not create_unet_model_2d/3d-based):

    * create_sysu_media_unet_model_2d   (sysuMediaWmhFlairOnlyModel{0,1,2},
                                          sysuMediaWmhFlairT1Model{0,1,2})
    * create_sysu_media_unet_model_3d   (antsxnetWmh, antsxnetWmhOr)
    * create_hypermapp3r_unet_model_3d  (hyperMapp3r)
    * create_shiva_unet_model_3d        (pvs_shiva_t1_{0..5},
                                          pvs_shiva_t1_flair_{0..4},
                                          wmh_shiva_flair_{0..4},
                                          wmh_shiva_t1_flair_{0..4})

Method: read each .h5 file's root attribute `layer_names` -- Keras's
authoritative, fully-ordered list of every layer in true construction/graph
order (NOT the same as the per-class auto-naming counter, e.g. "conv3d_17",
which can be assigned out of graph order whenever a tensor is consumed by
more than one downstream layer -- verified empirically for hypermapp3r,
where the "back64"/"back32" side-branches get naming-counter numbers that
do not match their position in `layer_names`). Convolution (and, for shiva,
BatchNormalization) weight tensors are extracted in this positional order
and assigned to the corresponding ANTsTorch module by an explicit,
architecture-specific role mapping that was derived by cross-referencing
kernel shapes against antspynet's Keras source
(antspynet/architectures/create_custom_unet_model.py) and antstorch's
state_dict key layout (antstorch/architectures/create_custom_unet_model.py).

Each conversion is verified 3 ways: load_state_dict(strict=True) with zero
missing/unexpected keys, a real forward pass with a sane output range
(sigmoid in [0,1] / softmax summing to 1), and a reload-from-disk identity
check (bit-exact, atol=1e-6) confirming the saved .pt reproduces the
in-memory model's output.

Usage (run on a machine with h5py + torch installed; antstorch's source
tree needs to be importable -- point --antstorch-src at the repo root):

    python convert_wmh_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch

    # convert just one file:
    python convert_wmh_bespoke.py --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch --antstorch-src ~/Pkg/ANTsTorch \\
        --only hyperMapp3r
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
# Import antstorch's create_unet_model.py / create_custom_unet_model.py
# directly by file path, bypassing the package __init__.py (so this script
# has no dependency on the rest of the antstorch package being importable).
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

    load("create_unet_model", "create_unet_model.py")
    ccu = load("create_custom_unet_model", "create_custom_unet_model.py")
    return ccu


def natkey(s):
    """Full natural sort key: split into alternating text/number tokens so that
    e.g. 'encoding_layers.1.0' sorts before 'encoding_layers.1.2' and after
    'encoding_layers.0.2' (a single trailing-number key would conflate the
    block index and the within-block conv index)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _weights_root(f):
    """Some .h5 files store weights via `save_weights()` (root-level
    `layer_names` attr, weight groups directly under root) while others are
    a full `model.save()` (root has model_config/training_config, and the
    weights + their own `layer_names` attr live one level down, under
    'model_weights'). Detect which and return the group to read from."""
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


def h5_get_norm(f, name, kind):
    """kind: 'instance' (beta,gamma) or 'batch' (beta,gamma,moving_mean,moving_variance)."""
    root = _weights_root(f)
    g = root[f"{name}/{name}"]
    gamma = np.array(g["gamma:0"])
    beta = np.array(g["beta:0"])
    if kind == "batch":
        mean = np.array(g["moving_mean:0"])
        var = np.array(g["moving_variance:0"])
        return gamma, beta, mean, var
    return gamma, beta


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


def set_instance_norm(sd, prefix, gamma, beta):
    sd[f"{prefix}.weight"] = torch.from_numpy(gamma.astype(np.float32))
    sd[f"{prefix}.bias"] = torch.from_numpy(beta.astype(np.float32))


def set_batch_norm(sd, prefix, gamma, beta, mean, var):
    sd[f"{prefix}.weight"] = torch.from_numpy(gamma.astype(np.float32))
    sd[f"{prefix}.bias"] = torch.from_numpy(beta.astype(np.float32))
    sd[f"{prefix}.running_mean"] = torch.from_numpy(mean.astype(np.float32))
    sd[f"{prefix}.running_var"] = torch.from_numpy(var.astype(np.float32))


# ---------------------------------------------------------------------------
# sysu_media (2D / 3D) -- simple: all conv layers in positional order,
# encoder blocks then decoder blocks then output conv. No norm layers.
# ---------------------------------------------------------------------------

def _is_conv_layer(f, name, ndim):
    """Content-based conv-layer test (name conventions vary across .h5
    save formats -- e.g. some sysu_media weight files use explicit names
    like 'conv1_1' instead of the auto-generated 'conv2d_N')."""
    root = _weights_root(f)
    if name not in root:
        return False
    g = root[name]
    if name not in g:
        return False
    g2 = g[name]
    if "kernel:0" not in g2:
        return False
    return g2["kernel:0"].ndim == ndim


def convert_sysu_media(h5_path, model, dimension):
    names = h5_layer_names(h5_path)
    ndim = dimension + 2
    with h5py.File(h5_path, "r") as f:
        conv_names = [n for n in names if _is_conv_layer(f, n, ndim)]

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    conv_keys = sorted(
        [k[:-len(".weight")] for k in sd if k.endswith(".weight") and "encoding_layers" in k],
        key=natkey,
    ) + sorted(
        [k[:-len(".weight")] for k in sd if k.endswith(".weight") and "decoding_layers" in k],
        key=natkey,
    ) + ["output_conv"]
    assert len(conv_keys) == len(conv_names), (len(conv_keys), len(conv_names))

    with h5py.File(h5_path, "r") as f:
        for tkey, hname in zip(conv_keys, conv_names):
            W, b = h5_get_wb(f, hname)
            set_conv(sd, tkey, W, b, expect_bias=True)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# hypermapp3r: explicit role list, in h5 positional (layer_names) order --
# see module docstring for why this can't be a simple natural-sort zip.
# ---------------------------------------------------------------------------

_HYPERMAPP3R_ROLES = [
    # (torch_prefix, has_norm)
    ("encoding_conv.0", True), ("encoding_residual.0.conv1", True), ("encoding_residual.0.conv2", True),
    ("encoding_conv.1", True), ("encoding_residual.1.conv1", True), ("encoding_residual.1.conv2", True),
    ("encoding_conv.2", True), ("encoding_residual.2.conv1", True), ("encoding_residual.2.conv2", True),
    ("encoding_conv.3", True), ("encoding_residual.3.conv1", True), ("encoding_residual.3.conv2", True),
    ("up0.conv", True),
    ("feature64.conv1", True), ("feature64.conv2", True),
    ("up1.conv", True),
    ("feature32.conv1", True), ("feature32.conv2", True),
    ("up2.conv", True),
    ("final_conv1", True),
    ("back64", False),
    ("final_conv2", True),
    ("back32", False),
    ("final_conv3", False),
]


def convert_hypermapp3r(h5_path, model):
    names = h5_layer_names(h5_path)
    conv_names = [n for n in names if re.match(r"^conv3d(_\d+)?$", n)]
    norm_names = [n for n in names if re.match(r"^instance_normalization(_\d+)?$", n)]

    assert len(conv_names) == len(_HYPERMAPP3R_ROLES), (len(conv_names), len(_HYPERMAPP3R_ROLES))
    n_norm_roles = sum(1 for _, has_norm in _HYPERMAPP3R_ROLES if has_norm)
    assert len(norm_names) == n_norm_roles, (len(norm_names), n_norm_roles)

    sd = {k: v.clone() for k, v in model.state_dict().items()}

    with h5py.File(h5_path, "r") as f:
        norm_idx = 0
        for (prefix, has_norm), hname in zip(_HYPERMAPP3R_ROLES, conv_names):
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
# shiva: explicit role list (linear architecture, h5 positional order
# matches the per-level interleaved conv1/conv2 construction directly).
# ---------------------------------------------------------------------------

def _shiva_roles(number_of_filters=(10, 18, 32, 58, 104, 187, 337)):
    roles = []
    for i in range(len(number_of_filters)):
        roles.append((f"encoding_layers.{i}.0", "batch"))
        roles.append((f"encoding_layers.{i}.1", "batch"))
    for idx in range(len(number_of_filters)):
        roles.append((f"decoding_conv1.{idx}", "batch"))
        roles.append((f"decoding_conv2.{idx}", "batch"))
    roles.append(("final_conv1", "batch"))
    roles.append(("final_conv2", "batch"))
    roles.append(("output_conv", None))
    return roles


def convert_shiva(h5_path, model):
    names = h5_layer_names(h5_path)
    conv_names = [n for n in names if re.match(r"^conv3d(_\d+)?$", n)]
    bn_names = [n for n in names if re.match(r"^batch_normalization(_\d+)?$", n)]

    roles = _shiva_roles()
    assert len(conv_names) == len(roles), (len(conv_names), len(roles))
    n_bn_roles = sum(1 for _, k in roles if k == "batch")
    assert len(bn_names) == n_bn_roles, (len(bn_names), n_bn_roles)

    sd = {k: v.clone() for k, v in model.state_dict().items()}

    with h5py.File(h5_path, "r") as f:
        bn_idx = 0
        for (prefix, kind), hname in zip(roles, conv_names):
            if kind == "batch":
                W, _b = h5_get_wb(f, hname, has_bias=False)
                set_conv(sd, f"{prefix}.0", W, None, expect_bias=False)
                gamma, beta, mean, var = h5_get_norm(f, bn_names[bn_idx], "batch")
                bn_idx += 1
                set_batch_norm(sd, f"{prefix}.1", gamma, beta, mean, var)
            else:
                W, b = h5_get_wb(f, hname)
                set_conv(sd, prefix, W, b, expect_bias=True)
        assert bn_idx == len(bn_names)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_and_save(model, out_path, build_fn, x, mode):
    model.eval()
    with torch.no_grad():
        y = model(x)
    if mode == "sigmoid":
        assert torch.all((y >= 0) & (y <= 1)), "sigmoid output out of [0,1]"
    elif mode == "softmax":
        s = y.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3), "softmax not normalized"

    torch.save(model.state_dict(), out_path)

    model2 = build_fn()
    model2.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True), strict=True)
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y, y2, atol=1e-6), f"reload mismatch for {out_path}"
    print(f"OK -> {out_path}  output shape {tuple(y.shape)}")


# ---------------------------------------------------------------------------
# Manifest: (h5 filename stem, architecture, kwargs)
# ---------------------------------------------------------------------------

def build_manifest():
    m = []
    for i in range(3):
        m.append((f"sysuMediaWmhFlairOnlyModel{i}", "sysu2d", dict(input_channel_size=1)))
    for i in range(3):
        m.append((f"sysuMediaWmhFlairT1Model{i}", "sysu2d", dict(input_channel_size=2)))
    m.append(("antsxnetWmh", "sysu3d", dict(input_channel_size=2, number_of_filters=(64, 96, 128, 256, 512))))
    m.append(("antsxnetWmhOr", "sysu3d", dict(input_channel_size=2, number_of_filters=(64, 96, 128, 256, 512))))
    m.append(("hyperMapp3r", "hypermapp3r", dict(input_channel_size=2)))
    for i in range(6):
        m.append((f"pvs_shiva_t1_{i}", "shiva", dict(number_of_modalities=1, number_of_outputs=1)))
    for i in range(5):
        m.append((f"pvs_shiva_t1_flair_{i}", "shiva", dict(number_of_modalities=2, number_of_outputs=1)))
    for i in range(5):
        m.append((f"wmh_shiva_flair_{i}", "shiva", dict(number_of_modalities=1, number_of_outputs=1)))
    for i in range(5):
        m.append((f"wmh_shiva_t1_flair_{i}", "shiva", dict(number_of_modalities=2, number_of_outputs=1)))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", required=True, help="Directory with the ANTsPyNet .h5 files (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", required=True, help="Directory to write <prefix>_pytorch.pt files (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", required=True, help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--only", default=None, help="Only convert this one h5 stem (e.g. hyperMapp3r)")
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
            if arch == "sysu2d":
                def build(kwargs=kwargs):
                    return ccu.create_sysu_media_unet_model_2d(**kwargs)
                model = build()
                convert_sysu_media(h5_path, model, dimension=2)
                x = torch.randn(1, kwargs["input_channel_size"], 200, 200)
                verify_and_save(model, out_path, build, x, mode="sigmoid")
            elif arch == "sysu3d":
                def build(kwargs=kwargs):
                    return ccu.create_sysu_media_unet_model_3d(**kwargs)
                model = build()
                convert_sysu_media(h5_path, model, dimension=3)
                x = torch.randn(1, kwargs["input_channel_size"], 64, 64, 64)
                verify_and_save(model, out_path, build, x, mode="sigmoid")
            elif arch == "hypermapp3r":
                def build(kwargs=kwargs):
                    return ccu.create_hypermapp3r_unet_model_3d(**kwargs)
                model = build()
                convert_hypermapp3r(h5_path, model)
                x = torch.randn(1, kwargs["input_channel_size"], 32, 32, 32)
                verify_and_save(model, out_path, build, x, mode="sigmoid")
            elif arch == "shiva":
                def build(kwargs=kwargs):
                    return ccu.create_shiva_unet_model_3d(**kwargs)
                model = build()
                convert_shiva(h5_path, model)
                # 7 encoder pooling levels (2**7=128) -- needs a spatial size
                # that stays >=1 at the bottleneck; use a modest multiple of 128.
                x = torch.randn(1, kwargs["number_of_modalities"], 128, 128, 128)
                mode = "sigmoid" if kwargs["number_of_outputs"] == 1 else "softmax"
                verify_and_save(model, out_path, build, x, mode=mode)
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
