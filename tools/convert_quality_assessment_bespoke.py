#!/usr/bin/env python3
"""
convert_quality_assessment_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for the 4 quality_assessment.py models used by
tid_neural_image_assessment(): tidsQualityAssessment, koniqMS, koniqMS2,
koniqMS3.

--------------------------------------------------------------------------
Major discovery (2026-08-23), resolving the biggest open confidence gap in
the whole ANTsPyNet -> ANTsTorch port: unlike every other application in
this session, quality_assessment.py never had an explicit Keras
architecture *constructor* anywhere in ANTsPyNet's source -- it only ever
calls tf.keras.models.load_model() on a full saved model. The port
therefore shipped with an explicitly-flagged, UNVERIFIED placeholder:
create_resnet_model_2d(input_channel_size=3, number_of_outputs=2,
mode="regression") with every other argument left at its default.

Inspecting koniqMS3.h5's own embedded model_config JSON (the same
technique that resolved the SIQ DBPN uncertainty for mri_super_resolution)
confirmed the base topology is a standard bottleneck ResNet (the same
shape antstorch's create_resnet_model_2d already builds) -- but converting
against a real file (2026-08-23) surfaced two real bugs in the first
version of this script, both now fixed:

  1. The first version tried to recover each Conv2D layer's *creation*
     order from its Keras auto-name (conv2d, conv2d_1, conv2d_2, ...) via
     a regex that silently failed to extract the numeric suffix whenever
     the name contained a digit before it -- which EVERY "conv2dN" name
     does (the literal substring "conv2d" itself contains "2"). The regex
     match failed unconditionally, the sort key was always 0, and the
     "sort" became a no-op that left Conv2D layers in raw *topological*
     order instead. Exactly like hypermapp3r/shiva (3rd wave earlier this
     session, see convert_wmh_bespoke.py's docstring), topological order
     can diverge from creation order whenever a tensor feeds more than one
     downstream layer -- which is exactly the shape of a residual block's
     shortcut branch. The result: conv3 and the projection shortcut conv
     (same kernel_size, same output filters -- genuinely indistinguishable
     by config alone) got silently swapped for every block that has a
     shortcut, confirmed against Nick's real conversion error
     (model_residual_layers.0.conv3 expected in_channels=128, got 64 --
     exactly the shortcut's shape).

  2. lowest_resolution (and the rest of the block schedule) was inferred
     during role *validation* but never threaded into the kwargs used to
     actually build the antstorch model, which stayed hardcoded at the
     ResNet-50 default (lowest_resolution=64) regardless. tidsQualityAssessment
     turned out to have lowest_resolution=16 (a smaller network than the
     other 3) -- confirming that these 4 models do NOT all share identical
     hyperparameters after all, contrary to the assumption in the first
     version of this script.

Both bugs are structural: guessing an order and validating only
kernel_size/filters (which cannot distinguish conv3 from its shortcut) is
not sufficient. This version abandons name-based ordering entirely and
instead walks the real Keras functional-model GRAPH via each layer's
`inbound_nodes` (available in model_config, the same source used for
everything else here): starting from the InputLayer, it re-discovers the
stem, then walks block-by-block purely from tensor connectivity --
conv1/shortcut are disambiguated by which one has fewer output filters
(conv1) vs. matches conv3's filters (shortcut is unambiguous whenever
present: only one of the two candidates sourced from the block's own
input has filters == the block's output filters), conv2 and conv3 are
found by simply following the single main-path chain forward. Both
lowest_resolution and the full (layers, residual_block_schedule) tuple
fall out of this walk directly from the real filter progression and block
grouping, per file, instead of being assumed. This also means the
converter no longer assumes a fixed ResNet-50 schedule at all -- whatever
schedule a given file actually has, in whatever order Keras happened to
serialize it, this converter will discover it.

This has only been validated end-to-end (round-trip synthetic .h5,
graph walk reproducing the exact real block structure) against koniqMS3's
real model_config. It is presumed -- not yet independently verified -- that
tidsQualityAssessment/koniqMS/koniqMS2 are all standard bottleneck ResNets
of this same general shape (just, per the lowest_resolution finding above,
not necessarily identical hyperparameters); the converter makes no
architecture assumption beyond "a Conv2D/BatchNormalization/LeakyReLU
bottleneck-residual functional graph with a single Dense regression head"
-- if a file doesn't fit that shape (e.g. squeeze-and-excite layers, which
this version explicitly detects and rejects rather than silently
mishandles, or grouped/ResNeXt convolutions), it fails loudly with a clear
error rather than converting incorrectly.

Usage (run on a machine with h5py + torch installed; antstorch's source
tree needs to be importable -- point --antstorch-src at the repo root):

    python convert_quality_assessment_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch

    # convert just one file (recommended first):
    python convert_quality_assessment_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch --only koniqMS3
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import types

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reuse the low-level h5 / state_dict helpers from convert_wmh_bespoke.py
# (must stay in the same tools/ directory) instead of duplicating them.
# ---------------------------------------------------------------------------

def _load_wmh_helpers():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "convert_wmh_bespoke.py")
    spec = importlib.util.spec_from_file_location("_convert_wmh_bespoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_resnet_module(antstorch_src):
    arch_dir = os.path.join(antstorch_src, "antstorch", "architectures")
    pkg_name = "_antstorch_arch_pkg_resnet"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [arch_dir]
    sys.modules[pkg_name] = pkg

    path = os.path.join(arch_dir, "create_resnet_model.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.create_resnet_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# model_config introspection
# ---------------------------------------------------------------------------

def _model_config(h5_path):
    with h5py.File(h5_path, "r") as f:
        config = f.attrs.get("model_config")
        if config is None and "model_weights" in f:
            config = f["model_weights"].attrs.get("model_config")
        if config is None:
            raise KeyError(f"No model_config attribute found in {h5_path}")
        if isinstance(config, bytes):
            config = config.decode("utf-8")
        return json.loads(config)


class _Graph:
    """Thin wrapper around a Keras functional model_config's layer list,
    giving name -> layer lookup and, crucially, name -> [source layer
    names] via inbound_nodes -- the only reliable way to distinguish
    conv3 from a same-shaped projection shortcut (see module docstring)."""

    def __init__(self, cfg):
        self.layers = cfg["config"]["layers"]
        self.by_name = {l["config"]["name"]: l for l in self.layers}
        self.input_name = cfg["config"]["input_layers"][0][0]

    def inputs_of(self, name):
        node = self.by_name[name]["inbound_nodes"]
        if not node:
            return []
        return [edge[0] for edge in node[0]]

    def find(self, class_name, source=None, kernel_size=None, filters=None):
        """All layer names matching the given filters. `source`, if given,
        requires inputs_of(name) == [source] (single-input layers only)."""
        out = []
        for name, l in self.by_name.items():
            if l["class_name"] != class_name:
                continue
            cfg = l["config"]
            if kernel_size is not None and tuple(cfg.get("kernel_size", ())) != kernel_size:
                continue
            if filters is not None and cfg.get("filters") != filters:
                continue
            if source is not None and self.inputs_of(name) != [source]:
                continue
            out.append(name)
        return out

    def find_one(self, *args, **kwargs):
        matches = self.find(*args, **kwargs)
        if len(matches) != 1:
            raise AssertionError(f"expected exactly 1 match for find({args!r}, {kwargs!r}), got {matches!r}")
        return matches[0]


def _extract_roles_and_kwargs(h5_path):
    """Walks the real Keras functional graph (from model_config) to
    reconstruct, per residual block, the exact (conv1, conv2, conv3,
    [shortcut]) role assignment -- see module docstring for why this
    replaces name-order-based matching entirely. Returns (roles, kwargs)
    where roles is a list of (torch_prefix, style, conv_name, bn_name) in
    the order needed to build antstorch's state_dict, and kwargs are the
    create_resnet_model_2d() constructor args this specific file actually
    needs (input_channel_size, number_of_outputs, mode, layers,
    residual_block_schedule, lowest_resolution) -- all derived from the
    file, nothing assumed."""
    cfg = _model_config(h5_path)
    g = _Graph(cfg)

    # Reject architectures this converter doesn't understand rather than
    # silently mis-converting them.
    class_counts = {}
    for l in g.layers:
        class_counts[l["class_name"]] = class_counts.get(l["class_name"], 0) + 1
    unsupported = set(class_counts) - {
        "InputLayer", "Conv2D", "BatchNormalization", "LeakyReLU",
        "MaxPooling2D", "Add", "GlobalAveragePooling2D", "Dense",
    }
    if unsupported:
        raise AssertionError(
            f"{h5_path}: model contains layer types this converter doesn't "
            f"support ({sorted(unsupported)}) -- likely squeeze-and-excite "
            f"or grouped/ResNeXt convolutions. Needs manual inspection, not "
            f"a blind conversion attempt."
        )

    input_layer = g.by_name[g.input_name]
    input_channel_size = input_layer["config"]["batch_input_shape"][-1]

    dense_name = g.find_one("Dense")
    dense_cfg = g.by_name[dense_name]["config"]
    number_of_outputs = dense_cfg["units"]
    activation = dense_cfg["activation"]
    mode = {"linear": "regression", "softmax": "classification", "sigmoid": "sigmoid"}.get(activation)
    if mode is None:
        raise ValueError(f"Unrecognized final Dense activation: {activation!r}")

    gap_name = g.find_one("GlobalAveragePooling2D")
    final_block_output = g.inputs_of(gap_name)[0]

    # --- stem: Conv2D(7x7) from the input, then BN, LeakyReLU, MaxPool ---
    stem_conv = g.find_one("Conv2D", source=g.input_name, kernel_size=(7, 7))
    lowest_resolution = g.by_name[stem_conv]["config"]["filters"]
    stem_bn = g.find_one("BatchNormalization", source=stem_conv)
    stem_relu = g.find_one("LeakyReLU", source=stem_bn)
    block_input = g.find_one("MaxPooling2D", source=stem_relu)

    roles = [("init_conv", "wrapped", stem_conv, stem_bn)]
    stage_filters_in = []
    stage_block_counts = []
    block_idx = 0

    while True:
        # conv1 and (if present) the projection shortcut both source
        # directly from block_input via a 1x1 Conv2D -- disambiguate by
        # filters: conv1's filters are always smaller (n_filters_in),
        # the shortcut's always match conv3's (n_filters_out).
        candidates_1x1 = g.find("Conv2D", source=block_input, kernel_size=(1, 1))
        if not (1 <= len(candidates_1x1) <= 2):
            raise AssertionError(
                f"{h5_path}: expected 1 or 2 1x1 Conv2D layers sourced from "
                f"{block_input!r} (conv1, optionally + shortcut), got {candidates_1x1!r}"
            )
        candidates_1x1.sort(key=lambda n: g.by_name[n]["config"]["filters"])
        conv1 = candidates_1x1[0]
        shortcut = candidates_1x1[1] if len(candidates_1x1) == 2 else None
        filters_in = g.by_name[conv1]["config"]["filters"]

        bn1 = g.find_one("BatchNormalization", source=conv1)
        relu1 = g.find_one("LeakyReLU", source=bn1)
        conv2 = g.find_one("Conv2D", source=relu1, kernel_size=(3, 3))
        bn2 = g.find_one("BatchNormalization", source=conv2)
        relu2 = g.find_one("LeakyReLU", source=bn2)
        conv3 = g.find_one("Conv2D", source=relu2, kernel_size=(1, 1))
        bn3 = g.find_one("BatchNormalization", source=conv3)
        filters_out = g.by_name[conv3]["config"]["filters"]

        if shortcut is not None:
            if g.by_name[shortcut]["config"]["filters"] != filters_out:
                raise AssertionError(
                    f"{h5_path}: shortcut candidate {shortcut!r} filters "
                    f"{g.by_name[shortcut]['config']['filters']} != conv3 filters {filters_out}"
                )
            shortcut_bn = g.find_one("BatchNormalization", source=shortcut)
            add_sources = {bn3, shortcut_bn}
        else:
            add_sources = {bn3, block_input}

        add_candidates = [n for n in g.find("Add") if set(g.inputs_of(n)) == add_sources]
        if len(add_candidates) != 1:
            raise AssertionError(
                f"{h5_path}: expected exactly 1 Add layer with inputs {add_sources!r}, "
                f"got {add_candidates!r}"
            )
        add_name = add_candidates[0]
        block_relu = g.find_one("LeakyReLU", source=add_name)

        prefix = f"model_residual_layers.{block_idx}"
        roles.append((f"{prefix}.conv1", "wrapped", conv1, bn1))
        roles.append((f"{prefix}.conv2", "wrapped", conv2, bn2))
        roles.append((f"{prefix}.conv3", "direct", conv3, bn3))
        if shortcut is not None:
            roles.append((f"{prefix}.shortcut", "direct", shortcut, shortcut_bn))

        if stage_filters_in and stage_filters_in[-1] == filters_in:
            stage_block_counts[-1] += 1
        else:
            stage_filters_in.append(filters_in)
            stage_block_counts.append(1)

        block_idx += 1
        if block_relu == final_block_output:
            break
        block_input = block_relu

    layers_tuple = []
    for f_in in stage_filters_in:
        ratio = f_in / lowest_resolution
        log2_ratio = math.log2(ratio)
        if abs(log2_ratio - round(log2_ratio)) > 1e-6:
            raise AssertionError(
                f"{h5_path}: stage filters_in={f_in} is not lowest_resolution "
                f"({lowest_resolution}) times a power of 2 -- doesn't fit "
                f"antstorch's create_resnet_model_2d layers= formula."
            )
        layers_tuple.append(round(log2_ratio))

    kwargs = dict(input_channel_size=input_channel_size,
                   number_of_outputs=number_of_outputs,
                   mode=mode,
                   layers=tuple(layers_tuple),
                   residual_block_schedule=tuple(stage_block_counts),
                   lowest_resolution=lowest_resolution)
    return roles, dense_name, kwargs


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_quality_assessment(h5_path, w):
    """w: the low-level helper module loaded from convert_wmh_bespoke.py."""
    roles, dense_name, kwargs = _extract_roles_and_kwargs(h5_path)

    resnet_mod = _RESNET_MODULE
    model = resnet_mod.create_resnet_model_2d(**kwargs)
    sd = {k: v.clone() for k, v in model.state_dict().items()}

    expected_prefixes = set()
    for k in sd:
        if k.endswith(".weight") and (".conv" in k or k.startswith("init_conv") or "shortcut" in k):
            expected_prefixes.add(k.rsplit(".", 2)[0] if ".1.0." in k or k.endswith(".1.0.weight") else k[:-len(".weight")])

    with h5py.File(h5_path, "r") as f:
        for prefix, style, conv_hname, bn_hname in roles:
            W, b = w.h5_get_wb(f, conv_hname)
            conv_prefix = f"{prefix}.0"
            w.set_conv(sd, conv_prefix, W, b, expect_bias=True)

            gamma, beta, mean, var = w.h5_get_norm(f, bn_hname, "batch")
            bn_prefix = f"{prefix}.1.0" if style == "wrapped" else f"{prefix}.1"
            w.set_batch_norm(sd, bn_prefix, gamma, beta, mean, var)

        Wd, bd = w.h5_get_wb(f, dense_name)
        torchWd = np.transpose(Wd, (1, 0)).astype(np.float32)  # Keras (in,out) -> torch (out,in)
        assert sd["dense.weight"].shape == torch.Size(torchWd.shape), (
            "dense.weight", sd["dense.weight"].shape, torchWd.shape)
        sd["dense.weight"] = torch.from_numpy(torchWd)
        assert sd["dense.bias"].shape == torch.Size(bd.shape), ("dense.bias", sd["dense.bias"].shape, bd.shape)
        sd["dense.bias"] = torch.from_numpy(bd.astype(np.float32))

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model, kwargs


def save_and_verify(model, kwargs, out_path, resnet_mod, x):
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), f"non-finite output for {out_path}"

    torch.save({"state_dict": model.state_dict(), "architecture_kwargs": kwargs}, out_path)

    model2 = resnet_mod.create_resnet_model_2d(**kwargs)
    loaded = torch.load(out_path, map_location="cpu", weights_only=True)
    model2.load_state_dict(loaded["state_dict"], strict=True)
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y, y2, atol=1e-6), f"reload mismatch for {out_path}"
    print(f"OK -> {out_path}  output shape {tuple(y.shape)}  kwargs={kwargs}")


def build_manifest():
    return ["tidsQualityAssessment", "koniqMS", "koniqMS2", "koniqMS3"]


_RESNET_MODULE = None


def main():
    global _RESNET_MODULE

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", required=True, help="Directory with the ANTsPyNet .h5 files (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", required=True, help="Directory to write <prefix>_pytorch.pt files (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", required=True, help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--only", default=None, help="Only convert this one h5 stem (e.g. koniqMS3)")
    args = p.parse_args()

    weights_dir = os.path.expanduser(args.weights_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    w = _load_wmh_helpers()
    _RESNET_MODULE = _load_resnet_module(os.path.expanduser(args.antstorch_src))

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
            model, kwargs = convert_quality_assessment(h5_path, w)
            # Patch-sized random input (101x101, as used by tid_neural_image_assessment's
            # patch-based mode) -- input_channel_size comes from the file itself.
            x = torch.randn(1, kwargs["input_channel_size"], 101, 101)
            save_and_verify(model, kwargs, out_path, _RESNET_MODULE, x)
            results[stem] = "OK"
        except Exception as e:
            results[stem] = f"FAILED: {e!r}"
            print(f"[{stem}] FAILED: {e!r}")

    print("\n==== SUMMARY ====")
    for stem, status in results.items():
        print(f"{stem}: {status}")


if __name__ == "__main__":
    main()
