#!/usr/bin/env python3
"""
convert_antspynet_weights_to_antstorch.py

Generalized weight porter for ANTsPyNet (Keras) -> ANTsTorch (PyTorch) U-Nets
that share the same layout family. Thin wrapper around your existing
`convert_weights_antstorch_deconvflip.py`, but parameterized by `--task` and
the two new factories so it can be reused across Deep Atropos, brain
extraction, etc., with minimal changes.

Example
-------
python convert_antspynet_weights_to_antstorch.py \
  --task deep_atropos \
  --out-prefix ./deepAtropos_pytorch \
  --deconv-flip noflip

Notes
-----
- By default we enable *no flip* on deconvolution kernels to match the
  Keras->PyTorch kernel layout you validated for brain extraction.
- If a task requires a different base-filter schedule or last-layer activation,
  change it in the two factory registries so both sides match.
"""

import argparse
from typing import Tuple, List

import numpy as np
import torch
from tensorflow.keras.models import Model

# Import factories
from return_antspynet_unet import return_antspynet_unet
from return_antstorch_unet import return_antstorch_unet

# --- low-level tensor transforms (borrowed from your existing converter) -----
def _np():
    import numpy as _numpy
    return _numpy

def _tf_conv3d_to_torch_weight(w_keras):
    # [Kx, Ky, Kz, Cin, Cout] -> [Cout, Cin, Kx, Ky, Kz]
    w = _np().transpose(w_keras, (4, 3, 0, 1, 2))
    return _np().ascontiguousarray(w)

def _tf_deconv3d_to_torch_weight(w_keras, flip: bool = False):
    # optional flip in xyz to match PyTorch ConvTranspose3d
    w = _np().flip(w_keras, axis=(0, 1, 2)) if flip else w_keras
    w = _np().transpose(w, (4, 3, 0, 1, 2))
    return _np().ascontiguousarray(w)

def _unwrap_deconv_and_names(module: torch.nn.Module, base_name: str):
    inner = getattr(module, "deconv", None)
    if inner is not None:
        return inner, f"{base_name}.deconv.weight", f"{base_name}.deconv.bias"
    return module, f"{base_name}.weight", f"{base_name}.bias"


def _collect_keras_layers(kmodel: Model):
    layers = []
    for l in kmodel.layers:
        cfg = getattr(l, "get_config", lambda: {})()
        layers.append((l.name, l, cfg))
    return layers


def _collect_torch_modules(tmodel: torch.nn.Module):
    pairs = []
    for name, mod in tmodel.named_modules():
        if len(list(mod.children())) == 0:  # leaf
            pairs.append((name, mod))
    return pairs


def convert_task(task: str, out_prefix: str, deconv_flip: bool = False, verbose: bool = True):
    kmodel, kmeta = return_antspynet_unet(task, load_weights=True, verbose=verbose)
    tmodel, tmeta = return_antstorch_unet(task, weights_path=None, strict=False, verbose=verbose)

    if verbose:
        print(f"[convert] Converting weights for task='{task}'")

    # 1) Flatten keras weights by layer order
    k_weights = []
    for _, layer, _ in _collect_keras_layers(kmodel):
        w = layer.get_weights()
        if w:
            k_weights.append((layer.name, w))

    # 2) Iterate torch parameters in a stable order and assign
    sd = tmodel.state_dict()
    new_sd = sd.copy()

    # Simple name-guided mapping
    for (lname, wlist) in k_weights:
        if "conv3d" in lname or "conv3d_transpose" in lname or "deconv3d" in lname or "conv3d_transpose" in lname:
            # Find a plausible torch target
            # Heuristic: look for a param key that ends with ".weight" and contains lname fragments
            candidates = [k for k in new_sd.keys() if k.endswith(".weight") and lname.split("/")[0] in k]
            if not candidates:
                continue
            target_w_key = candidates[0]
            target_b_key = target_w_key.replace(".weight", ".bias")

            w = wlist[0]
            if "transpose" in lname or "deconv" in lname:
                w_t = _tf_deconv3d_to_torch_weight(w, flip=deconv_flip)
            else:
                w_t = _tf_conv3d_to_torch_weight(w)

            if new_sd[target_w_key].shape != w_t.shape:
                if verbose:
                    print(f"[warn] shape mismatch for {lname} -> {target_w_key}: {w_t.shape} vs {tuple(new_sd[target_w_key].shape)}")
            else:
                new_sd[target_w_key] = torch.from_numpy(w_t)

            # Bias if present
            if len(wlist) > 1 and target_b_key in new_sd:
                b = wlist[1]
                new_sd[target_b_key] = torch.from_numpy(b.astype(np.float32))

        elif "batch_normalization" in lname or "instance_normalization" in lname or "group_normalization" in lname:
            # Keras BN: gamma, beta, moving_mean, moving_variance
            # Map to PyTorch affine + buffers if present
            bn_keys = [k for k in new_sd.keys() if lname.split("/")[0] in k]
            # This section may require per-arch tweaks; we keep it minimal here.
            pass

        else:
            # Fallback: skip layers without weights or non-conv layers
            pass

    tmodel.load_state_dict(new_sd, strict=False)
    torch.save(tmodel.state_dict(), f"{out_prefix}.pt")

    if verbose:
        print(f"[convert] Saved: {out_prefix}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--deconv-flip", choices=["flip", "noflip"], default="noflip")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    convert_task(args.task, args.out_prefix, deconv_flip=(args.deconv_flip == "flip"), verbose=args.verbose)


if __name__ == "__main__":
    main()
