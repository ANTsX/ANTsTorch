#!/usr/bin/env python3
"""
convert_antspynet_weights_to_antstorch.py (clean, full, wrapper-aware)

- Builds the Keras model (ANTsPyNet) and loads its H5 weights.
- Builds the Torch model (ANTsTorch) via your factory (which may return a wrapper with base.* params).
- Copies base UNet weights (encoding, deconv, decoding, main output) from Keras -> Torch.
- Copies auxiliary 1x1x1 heads (off penultimate) when the Keras model exposes multiple outputs.
- Robust to ".weight" vs ".conv.weight" naming and to "base." prefix (wrapper).

CLI:
  python convert_antspynet_weights_to_antstorch.py \
    --task deep_flash_left_t1 \
    --out-prefix ~/.antstorch/deepFlashLeftT1Hierarchical_pytorch \
    --deconv-flip noflip \
    --verbose
"""

import argparse
import re
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

# Factories you already have in your project (ensure PYTHONPATH has this dir)
from return_antspynet_unet import return_antspynet_unet
from return_antstorch_unet import return_antstorch_unet

# ---- KERAS/TENSORFLOW UTILITIES ----
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv3D, Conv3DTranspose
except Exception as e:
    tf = None
    Model = None
    Conv3D = None
    Conv3DTranspose = None


def natural_key(s: str):
    """Sort helper: split digits vs non-digits for natural ordering."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---------- Key resolution & prefix helpers ----------

def _resolve_tkey(t_key: str, sd: Dict[str, torch.Tensor]) -> str:
    """
    Resolve a target torch key to an actual key present in the state_dict `sd`.
    Tries common variants (.weight -> .conv.weight / .bias -> .conv.bias)
    and falls back to a short fuzzy search. Returns a key (may be original).
    """
    if t_key in sd:
        return t_key

    # Common wrapped-conv variants
    if t_key.endswith(".weight"):
        alt = t_key[:-len(".weight")] + ".conv.weight"
        if alt in sd:
            return alt
    if t_key.endswith(".bias"):
        alt = t_key[:-len(".bias")] + ".conv.bias"
        if alt in sd:
            return alt

    # Fuzzy: same stem + common suffix
    stem = t_key.rsplit(".", 1)[0]
    suffixes = [".weight", ".conv.weight", ".bias", ".conv.bias"]
    candidates = [k for k in sd.keys() if stem in k and any(k.endswith(suf) for suf in suffixes)]
    if len(candidates) == 1:
        return candidates[0]

    # Broader fuzzy: last path tail
    parts = t_key.split(".")
    tail = ".".join(parts[-4:])
    for k in sd.keys():
        if k.endswith(tail) or k.endswith(tail.replace(".weight", ".conv.weight")) or k.endswith(tail.replace(".bias", ".conv.bias")):
            return k

    return t_key  # let caller raise if missing


def _strip_base_prefix(sd: Dict[str, torch.Tensor], base_prefix: str) -> Dict[str, torch.Tensor]:
    """Return a view of sd with base_prefix stripped from keys (for collection/profiling)."""
    if not base_prefix:
        return sd
    return { (k[len(base_prefix):] if k.startswith(base_prefix) else k): v for k, v in sd.items() }


def _to_torch_conv_weight_from_keras(W: np.ndarray) -> np.ndarray:
    """
    Keras Conv3D kernel:          (kD, kH, kW, inC, outC)
    Keras Conv3DTranspose kernel: (kD, kH, kW, outC, inC)
    Torch Conv3d weight:          (outC, inC, kD, kH, kW)
    Torch ConvTranspose3d weight: (inC, outC, kD, kH, kW)

    For both, the correct transpose is (4, 3, 0, 1, 2) because the last two dims swap.
    """
    if W.ndim != 5:
        raise ValueError(f"Expected 5D kernel, got {W.ndim}D with shape {W.shape}")
    return np.transpose(W, (4, 3, 0, 1, 2)).astype(np.float32)


def _maybe_flip_spatial(torch_weight: np.ndarray, do_flip: bool) -> np.ndarray:
    """Optionally flip along spatial dims (kD, kH, kW) for ConvTranspose3d kernels after mapping."""
    if not do_flip:
        return torch_weight
    return torch_weight[..., ::-1, ::-1, ::-1].copy()


# ---------- Keras model inspection ----------

def _keras_output_layer_names(kmodel: "Model") -> List[str]:
    outs = kmodel.outputs if isinstance(kmodel.outputs, (list, tuple)) else [kmodel.outputs]
    names = []
    for t in outs:
        kh = getattr(t, "_keras_history", None)
        if kh is None:
            # fallback: try tensor's originating layer name if present
            try:
                names.append(t.node.outbound_layer.name)
            except Exception:
                names.append(getattr(getattr(t, "op", None), "name", "unknown_output"))
        else:
            layer = getattr(kh, "layer", None) or (kh[0] if isinstance(kh, (list, tuple)) else None)
            names.append(layer.name if layer is not None else str(kh))
    return names


def _collect_keras_convs_and_dect(kmodel: "Model",
                                  exclude_layer_names: List[str]) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]],
                                                                            List[Tuple[str, np.ndarray, np.ndarray]],
                                                                            Tuple[str, np.ndarray, np.ndarray]]:
    """
    Return (conv_list, deconvtranspose_list, main_out) where each list contains tuples of (name, W, b).

    - conv_list: all Conv3D kernels EXCEPT the main output conv and any aux heads.
    - deconvtranspose_list: all Conv3DTranspose kernels (in model order).
    - main_out: (name, W, b) for the main output conv (the first output layer).
    """
    if tf is None or Conv3D is None:
        raise RuntimeError("TensorFlow/Keras not available in environment.")

    out_names = _keras_output_layer_names(kmodel)
    main_out_name = out_names[0]  # first output is main head

    conv_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    deconv_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    main_out: Tuple[str, np.ndarray, np.ndarray] | None = None

    for layer in kmodel.layers:
        if isinstance(layer, Conv3DTranspose):
            weights = layer.get_weights()
            if len(weights) == 2:
                W, b = weights
            elif len(weights) == 1:
                W, b = weights[0], None
            else:
                continue
            deconv_list.append((layer.name, W, b))
        elif isinstance(layer, Conv3D):
            weights = layer.get_weights()
            if len(weights) == 2:
                W, b = weights
            elif len(weights) == 1:
                W, b = weights[0], None
            else:
                continue
            if layer.name == main_out_name:
                main_out = (layer.name, W, b)
            elif layer.name in exclude_layer_names:
                # skip aux heads
                continue
            else:
                conv_list.append((layer.name, W, b))

    if main_out is None:
        # Fallback: try last Conv3D layer if outputs failed
        for layer in reversed(kmodel.layers):
            if isinstance(layer, Conv3D):
                weights = layer.get_weights()
                if len(weights) == 2:
                    W, b = weights
                elif len(weights) == 1:
                    W, b = weights[0], None
                else:
                    continue
                main_out = (layer.name, W, b)
                break

    if main_out is None:
        raise RuntimeError("Could not find main output Conv3D layer in Keras model.")

    return conv_list, deconv_list, main_out


# ---------- Torch state_dict scanning ----------

def _list_weight_keys(sd_view: Dict[str, torch.Tensor], contains: str) -> List[str]:
    return sorted([k for k, v in sd_view.items()
                   if contains in k and k.endswith(("weight", "conv.weight")) and v.ndim == 5],
                  key=natural_key)


def _list_deconv_keys(sd_view: Dict[str, torch.Tensor]) -> List[str]:
    # typical pattern names
    patterns = ["decoding_convolution_transpose_layers", "deconvolution", "up_convolution", "upconvolution", "transpose"]
    keys = []
    for p in patterns:
        keys.extend(_list_weight_keys(sd_view, p))
    # de-dup while keeping order
    seen = set()
    ordered = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _list_conv_keys(sd_view: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    enc = _list_weight_keys(sd_view, "encoding_convolution_layers")
    dec = _list_weight_keys(sd_view, "decoding_convolution_layers")
    return enc, dec


def _find_main_out_keys(sd_view: Dict[str, torch.Tensor], number_of_outputs: int) -> Tuple[str, str | None]:
    """
    Heuristic: choose the last 1x1x1 3D conv whose out_channels == number_of_outputs.
    Return (weight_key, bias_key_or_None).
    """
    candidates = []
    for k, v in sd_view.items():
        if not (k.endswith(("weight", "conv.weight")) and v.ndim == 5):
            continue
        shape = tuple(v.shape)
        kD, kH, kW = shape[2:5]
        outC = shape[0]
        if (kD, kH, kW) == (1, 1, 1) and outC == number_of_outputs:
            candidates.append(k)
    if not candidates:
        # fallback: pick the very last 1x1x1 conv by order
        for k, v in sd_view.items():
            if v.ndim == 5 and tuple(v.shape[2:5]) == (1, 1, 1):
                candidates.append(k)
    if not candidates:
        raise RuntimeError("Could not locate main output 1x1x1 conv in Torch model.")
    wkey = sorted(candidates, key=natural_key)[-1]
    bkey = wkey.replace(".weight", ".bias")
    if bkey not in sd_view:
        # try conv.bias
        alt = wkey.replace(".conv.weight", ".conv.bias")
        bkey = alt if alt in sd_view else None
    return wkey, bkey


# ---------- Main conversion ----------

def convert_task(task: str, out_prefix: str, deconv_flip: bool = False, verbose: bool = True) -> str:
    # Build models
    kmodel, kspec = return_antspynet_unet(task, load_weights=True, verbose=verbose)
    tmodel, tspec = return_antstorch_unet(task, weights_path=None, strict=False, verbose=verbose)

    # Copy state_dict
    tmodel = tmodel.to("cpu")
    new_sd: Dict[str, torch.Tensor] = {k: v.clone().detach() for k, v in tmodel.state_dict().items()}

    # Wrapper awareness
    base_prefix = "base." if any(k.startswith("base.") for k in new_sd.keys()) else ""
    sd_view = _strip_base_prefix(new_sd, base_prefix)

    # Warmup (for wrappers that lazily build heads)
    with torch.no_grad():
        in_ch = int(tspec["input_image_size"][-1])
        depth = len(tspec.get("number_of_filters", [])) or 5
        min_side = 2 ** depth
        try:
            _ = tmodel(torch.zeros(1, in_ch, min_side, min_side, min_side))
        except Exception:
            # try a fixed small size if exact power-of-two fails
            _ = tmodel(torch.zeros(1, in_ch, 32, 32, 32))

    # Keras outputs and aux names
    out_layer_names = _keras_output_layer_names(kmodel)
    n_aux = max(0, len(out_layer_names) - 1)
    if verbose:
        print(f"[convert] Keras outputs: {out_layer_names} -> aux heads: {n_aux}")

    aux_names = out_layer_names[1:]

    # Collect Keras convs/deconvs and main output
    k_convs, k_dect, k_main = _collect_keras_convs_and_dect(kmodel, exclude_layer_names=aux_names)

    # Collect Torch keys
    t_dect = _list_deconv_keys(sd_view)
    t_enc, t_dec = _list_conv_keys(sd_view)
    if verbose:
        def prof_dect(lst):  # show (in->out) channel profile
            out = []
            for k in lst:
                v = sd_view[k]
                # ConvTranspose3d weight shape: (inC, outC, kD, kH, kW)
                inC, outC = int(v.shape[0]), int(v.shape[1])
                out.append((inC, outC))
            return out
        print(f"[profile] Keras deconv (in->out) per layer: {[(w.shape[3], w.shape[4]) for (_, w, _) in k_dect]}")
        print(f"[profile] Torch  deconv (in->out) per layer: {prof_dect(t_dect)}")

    # Basic sanity checks
    if len(k_dect) != len(t_dect):
        raise RuntimeError(
            f"Deconv count mismatch: Keras={len(k_dect)} vs Torch={len(t_dect)}. "
            f"Patterns may differ; adjust _list_deconv_keys()."
        )

    total_t_convs = len(t_enc) + len(t_dec)
    if len(k_convs) != total_t_convs:
        raise RuntimeError(
            f"Conv count mismatch: Keras(non-out, non-aux)={len(k_convs)} vs "
            f"Torch enc+dec={total_t_convs}. Check ordering or patterns."
        )

    # ---------- Assign deconv (ConvTranspose) weights ----------
    for (lname, kW, kB), tkey in zip(k_dect, t_dect):
        torchW = _to_torch_conv_weight_from_keras(kW)
        torchW = _maybe_flip_spatial(torchW, deconv_flip)
        assign_key = base_prefix + tkey
        # shape guard
        rk = _resolve_tkey(assign_key, new_sd)
        tgt = new_sd[rk].cpu().numpy()
        if torchW.shape != tgt.shape:
            # Sometimes in/out reversed; try transpose last two dims
            alt = torchW.transpose(1, 0, 2, 3, 4)
            if alt.shape == tgt.shape:
                torchW = alt
        # assign
        new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
        # bias if exists
        if kB is not None:
            bkey = assign_key.replace(".weight", ".bias")
            bkey = _resolve_tkey(bkey, new_sd)
            if bkey in new_sd:
                if new_sd[bkey].numel() == kB.size:
                    new_sd[bkey] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[bkey].dtype)

    # ---------- Assign encoding convs then decoding convs ----------
    # Map Keras convs in order to Torch enc then dec
    idx = 0
    for tkey in t_enc:
        lname, kW, kB = k_convs[idx]
        idx += 1
        torchW = _to_torch_conv_weight_from_keras(kW)
        assign_key = base_prefix + tkey
        rk = _resolve_tkey(assign_key, new_sd)
        tgt = new_sd[rk].cpu().numpy()
        if torchW.shape != tgt.shape and torchW.transpose(1, 0, 2, 3, 4).shape == tgt.shape:
            torchW = torchW.transpose(1, 0, 2, 3, 4)
        new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
        if kB is not None:
            bkey = _resolve_tkey(assign_key.replace(".weight", ".bias"), new_sd)
            if bkey in new_sd and new_sd[bkey].numel() == kB.size:
                new_sd[bkey] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[bkey].dtype)

    for tkey in t_dec:
        lname, kW, kB = k_convs[idx]
        idx += 1
        torchW = _to_torch_conv_weight_from_keras(kW)
        assign_key = base_prefix + tkey
        rk = _resolve_tkey(assign_key, new_sd)
        tgt = new_sd[rk].cpu().numpy()
        if torchW.shape != tgt.shape and torchW.transpose(1, 0, 2, 3, 4).shape == tgt.shape:
            torchW = torchW.transpose(1, 0, 2, 3, 4)
        new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
        if kB is not None:
            bkey = _resolve_tkey(assign_key.replace(".weight", ".bias"), new_sd)
            if bkey in new_sd and new_sd[bkey].numel() == kB.size:
                new_sd[bkey] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[bkey].dtype)

    # ---------- Assign main output head ----------
    main_name, kW_main, kB_main = k_main
    out_wkey_view, out_bkey_view = _find_main_out_keys(sd_view, int(tspec["number_of_outputs"]))
    out_wkey = base_prefix + out_wkey_view
    out_bkey = base_prefix + out_bkey_view if out_bkey_view else None

    torchW = _to_torch_conv_weight_from_keras(kW_main)
    rk_out = _resolve_tkey(out_wkey, new_sd)
    tgt = new_sd[rk_out].cpu().numpy()
    if torchW.shape != tgt.shape and torchW.transpose(1, 0, 2, 3, 4).shape == tgt.shape:
        torchW = torchW.transpose(1, 0, 2, 3, 4)
    new_sd[rk_out] = torch.from_numpy(torchW).to(new_sd[rk_out].dtype)
    if kB_main is not None and out_bkey is not None:
        rb = _resolve_tkey(out_bkey, new_sd)
        if rb in new_sd and new_sd[rb].numel() == kB_main.size:
            new_sd[rb] = torch.from_numpy(kB_main.astype(np.float32)).to(new_sd[rb].dtype)

    # ---------- Assign auxiliary heads (heads.0/1/2 ...) ----------
    if n_aux > 0:
        for i, lname in enumerate(aux_names):
            layer = kmodel.get_layer(name=lname)
            weights = layer.get_weights()
            if len(weights) == 2:
                kW, kB = weights
            elif len(weights) == 1:
                kW, kB = weights[0], None
            else:
                continue
            torchW = _to_torch_conv_weight_from_keras(kW)
            wkey = f"heads.{i}.weight"
            rk = _resolve_tkey(wkey, new_sd)
            if rk not in new_sd:
                raise KeyError(f"Aux head #{i} weight key not found: {wkey}")
            tgt = new_sd[rk].cpu().numpy()
            if torchW.shape != tgt.shape and torchW.transpose(1, 0, 2, 3, 4).shape == tgt.shape:
                torchW = torchW.transpose(1, 0, 2, 3, 4)
            new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
            if kB is not None:
                bkey = f"heads.{i}.bias"
                rb = _resolve_tkey(bkey, new_sd)
                if rb in new_sd and new_sd[rb].numel() == kB.size:
                    new_sd[rb] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[rb].dtype)

    # Load into model (non-strict to tolerate any unused buffers) then save just the state_dict
    missing, unexpected = tmodel.load_state_dict(new_sd, strict=False)
    if verbose:
        if isinstance(missing, (list, tuple)) and isinstance(unexpected, (list, tuple)):
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:5] or unexpected[:5]:
                print("  e.g. missing[:5]   =", missing[:5])
                print("  e.g. unexpected[:5]=", unexpected[:5])

    out_path = out_prefix
    if not out_path.endswith(".pt"):
        out_path += ".pt"
    torch.save(tmodel.state_dict(), out_path)
    if verbose:
        print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--out-prefix", type=str, required=True)
    p.add_argument("--deconv-flip", type=str, default="noflip", choices=["flip", "noflip"])
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    deconv_flip = (args.deconv_flip == "flip")
    convert_task(args.task, args.out_prefix, deconv_flip=deconv_flip, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
