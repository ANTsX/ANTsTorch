#!/usr/bin/env python3
"""
convert_antspynet_weights_to_antstorch.py (clean, full, wrapper-aware, 2D/3D-aware)

- Builds the Keras model (ANTsPyNet) and loads its H5 weights.
- Builds the Torch model (ANTsTorch) via your factory (which may return a wrapper with base.* params).
- Copies base UNet weights (encoding, deconv, decoding, main output) from Keras -> Torch.
- Copies auxiliary 1x1(x1) heads (off penultimate) when the Keras model exposes multiple outputs.
- Robust to ".weight" vs ".conv.weight" naming and to "base." prefix (wrapper).
- Supports both 2-D (Conv2D/Conv2DTranspose, 4-D kernels) and 3-D (Conv3D/Conv3DTranspose,
  5-D kernels) tasks, dispatched from tasks_registry via spec["dimension"].

  NOTE: the 2-D code path added here mirrors the pre-existing, in-production 3-D path
  structurally (same key-resolution / shape-guard / transpose-fallback logic, generalized
  from 5-D to 4-D kernels). It has been validated architecturally (the ANTsTorch side --
  model construction, key naming, warmup/head instantiation) but NOT run end-to-end against
  real Keras/antspynet weights, since TensorFlow/antspynet are not available in the
  environment this port was written in. Do one small 2-D conversion (e.g. --task lung_xray)
  and sanity-check the resulting missing/unexpected key counts and a forward-pass numerical
  comparison against the original Keras model before relying on it in production.

  Also note: this tool only supports tasks built from the *generic* U-Net
  (create_unet_model_2d/3d) via tasks_registry.py. It does NOT support the bespoke
  sysu_media / hypermapp3r / shiva / DBPN architectures -- see tasks_registry.py's
  module docstring.

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
from tasks_registry import get_task_spec

# ---- KERAS/TENSORFLOW UTILITIES ----
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose
except Exception as e:
    tf = None
    Model = None
    Conv3D = None
    Conv3DTranspose = None
    Conv2D = None
    Conv2DTranspose = None


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
    3-D:
      Keras Conv3D kernel:          (kD, kH, kW, inC, outC)
      Keras Conv3DTranspose kernel: (kD, kH, kW, outC, inC)
      Torch Conv3d weight:          (outC, inC, kD, kH, kW)
      Torch ConvTranspose3d weight: (inC, outC, kD, kH, kW)
      -> transpose (4, 3, 0, 1, 2) (the last two dims swap to the front, spatial dims follow).

    2-D:
      Keras Conv2D kernel:          (kH, kW, inC, outC)
      Keras Conv2DTranspose kernel: (kH, kW, outC, inC)
      Torch Conv2d weight:          (outC, inC, kH, kW)
      Torch ConvTranspose2d weight: (inC, outC, kH, kW)
      -> transpose (3, 2, 0, 1) (same pattern, one fewer spatial dim).
    """
    if W.ndim == 5:
        return np.transpose(W, (4, 3, 0, 1, 2)).astype(np.float32)
    elif W.ndim == 4:
        return np.transpose(W, (3, 2, 0, 1)).astype(np.float32)
    else:
        raise ValueError(f"Expected 4D (2-D conv) or 5D (3-D conv) kernel, got {W.ndim}D with shape {W.shape}")


def _maybe_flip_spatial(torch_weight: np.ndarray, do_flip: bool) -> np.ndarray:
    """Optionally flip along spatial dims for ConvTranspose kernels after mapping."""
    if not do_flip:
        return torch_weight
    if torch_weight.ndim == 5:
        return torch_weight[..., ::-1, ::-1, ::-1].copy()
    elif torch_weight.ndim == 4:
        return torch_weight[..., ::-1, ::-1].copy()
    else:
        raise ValueError(f"Expected 4D or 5D torch weight, got {torch_weight.ndim}D.")


def _transpose_in_out(W: np.ndarray) -> np.ndarray:
    """Swap the first two (outC, inC) axes of a torch-layout conv weight, keeping spatial dims."""
    if W.ndim == 5:
        return W.transpose(1, 0, 2, 3, 4)
    elif W.ndim == 4:
        return W.transpose(1, 0, 2, 3)
    else:
        raise ValueError(f"Expected 4D or 5D array, got {W.ndim}D.")


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
                                  exclude_layer_names: List[str],
                                  attention_gating: bool = False,
                                  dimension: int = 3) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]],
                                                                            List[Tuple[str, np.ndarray, np.ndarray]],
                                                                            Tuple[str, np.ndarray, np.ndarray],
                                                                            List[Tuple[str, np.ndarray, np.ndarray]]]:
    """
    Return (conv_list, deconvtranspose_list, main_out, attn_list) where each list
    contains tuples of (name, W, b).

    - conv_list: all Conv kernels EXCEPT the main output conv, any aux heads, and
      (if attention_gating) the 1x1(x1) attention convs.
    - deconvtranspose_list: all ConvTranspose kernels (in model order).
    - main_out: (name, W, b) for the main output conv (the first output layer).
    """
    if tf is None or Conv3D is None:
        raise RuntimeError("TensorFlow/Keras not available in environment.")

    if dimension == 3:
        ConvND, ConvNDTranspose, one_kernel = Conv3D, Conv3DTranspose, (1, 1, 1)
    elif dimension == 2:
        ConvND, ConvNDTranspose, one_kernel = Conv2D, Conv2DTranspose, (1, 1)
    else:
        raise ValueError(f"Unsupported dimension {dimension} (must be 2 or 3).")

    out_names = _keras_output_layer_names(kmodel)
    main_out_name = out_names[0]  # first output is main head

    conv_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    deconv_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    attn_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    main_out: Tuple[str, np.ndarray, np.ndarray] | None = None

    for layer in kmodel.layers:
        if isinstance(layer, ConvNDTranspose):
            weights = layer.get_weights()
            if len(weights) == 2:
                W, b = weights
            elif len(weights) == 1:
                W, b = weights[0], None
            else:
                continue
            deconv_list.append((layer.name, W, b))
        elif isinstance(layer, ConvND):
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
                # if attention enabled, siphon off 1x1(x1) convs (excluding main/aux) as attention layers
                if attention_gating and tuple(getattr(layer, 'kernel_size', ())) == one_kernel:
                    attn_list.append((layer.name, W, b))
                else:
                    conv_list.append((layer.name, W, b))

    if main_out is None:
        # Fallback: try last ConvND layer if outputs failed
        for layer in reversed(kmodel.layers):
            if isinstance(layer, ConvND):
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
        raise RuntimeError("Could not find main output conv layer in Keras model.")

    return conv_list, deconv_list, main_out, attn_list


# ---------- Torch state_dict scanning ----------

def _list_weight_keys(sd_view: Dict[str, torch.Tensor], contains: str, ndim: int) -> List[str]:
    return sorted([k for k, v in sd_view.items()
                   if contains in k and k.endswith(("weight", "conv.weight")) and v.ndim == ndim],
                  key=natural_key)


def _list_deconv_keys(sd_view: Dict[str, torch.Tensor], ndim: int) -> List[str]:
    # typical pattern names
    patterns = ["decoding_convolution_transpose_layers", "deconvolution", "up_convolution", "upconvolution", "transpose"]
    keys = []
    for p in patterns:
        keys.extend(_list_weight_keys(sd_view, p, ndim))
    # de-dup while keeping order
    seen = set()
    ordered = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _list_conv_keys(sd_view: Dict[str, torch.Tensor], ndim: int) -> Tuple[List[str], List[str]]:
    enc = _list_weight_keys(sd_view, "encoding_convolution_layers", ndim)
    dec = _list_weight_keys(sd_view, "decoding_convolution_layers", ndim)
    return enc, dec


def _list_attention_keys(sd_view: Dict[str, torch.Tensor]) -> List[Dict[str, str]]:
    """
    Return a list of dictionaries per decoder level with expected keys for attention gates:
      {"theta_w": "...", "theta_b": "...", "phi_w": "...", "phi_b": "...", "psi_w": "...", "psi_b": "..."}
    Searches for modules under 'attn_gates_3d.<i>.(theta|phi|psi)' (and 2d variant).
    """
    out = []
    # find whether we have 3d or 2d
    prefixes = set()
    for k in sd_view.keys():
        if k.startswith("attn_gates_3d."):
            prefixes.add("attn_gates_3d")
        if k.startswith("attn_gates_2d."):
            prefixes.add("attn_gates_2d")
    for pref in sorted(prefixes):
        # find indices
        idxs = sorted({ int(m.group(1)) for k in sd_view.keys() for m in [re.search(rf"^{pref}\.(\d+)\.", k)] if m })
        for i in idxs:
            # weight keys might be '.weight' or '.conv.weight'
            def _find_key(stem):
                # prefer '.weight' direct, otherwise '.conv.weight'
                k1 = f"{pref}.{i}.{stem}.weight"
                if k1 in sd_view:
                    return k1
                k2 = f"{pref}.{i}.{stem}.conv.weight"
                if k2 in sd_view:
                    return k2
                # bias
                b1 = f"{pref}.{i}.{stem}.bias"
                if b1 in sd_view:
                    return b1  # caller will use replace()
                b2 = f"{pref}.{i}.{stem}.conv.bias"
                if b2 in sd_view:
                    return b2
                return None

            tw = _find_key("theta")
            pw = _find_key("phi")
            qw = _find_key("psi")
            # derive bias keys
            tb = tw.replace(".weight", ".bias") if tw else None
            if tb and tb not in sd_view:
                tb = tw.replace(".conv.weight", ".conv.bias")
            pb = pw.replace(".weight", ".bias") if pw else None
            if pb and pb not in sd_view:
                pb = pw.replace(".conv.weight", ".conv.bias")
            qb = qw.replace(".weight", ".bias") if qw else None
            if qb and qb not in sd_view:
                qb = qw.replace(".conv.weight", ".conv.bias")
            if tw and pw and qw:
                out.append(dict(theta_w=tw, theta_b=tb, phi_w=pw, phi_b=pb, psi_w=qw, psi_b=qb))
    return out


def _find_main_out_keys(sd_view: Dict[str, torch.Tensor], number_of_outputs: int, ndim: int) -> Tuple[str, str | None]:
    """
    Heuristic: choose the last 1x1(x1) conv whose out_channels == number_of_outputs.
    Return (weight_key, bias_key_or_None).
    """
    one_kernel = (1,) * (ndim - 2)
    candidates = []
    for k, v in sd_view.items():
        if not (k.endswith(("weight", "conv.weight")) and v.ndim == ndim):
            continue
        shape = tuple(v.shape)
        kernel = shape[2:ndim]
        outC = shape[0]
        if kernel == one_kernel and outC == number_of_outputs:
            candidates.append(k)
    if not candidates:
        # fallback: pick the very last 1x1(x1) conv by order
        for k, v in sd_view.items():
            if v.ndim == ndim and tuple(v.shape[2:ndim]) == one_kernel:
                candidates.append(k)
    if not candidates:
        raise RuntimeError("Could not locate main output 1x1(x1) conv in Torch model.")
    wkey = sorted(candidates, key=natural_key)[-1]
    bkey = wkey.replace(".weight", ".bias")
    if bkey not in sd_view:
        # try conv.bias
        alt = wkey.replace(".conv.weight", ".conv.bias")
        bkey = alt if alt in sd_view else None
    return wkey, bkey


# ---------- Main conversion ----------

def convert_task(task: str, out_prefix: str, deconv_flip: bool = False, verbose: bool = True, attention_gating: bool | None = None) -> str:
    # Build models
    kmodel, kspec = return_antspynet_unet(task, load_weights=True, verbose=verbose)
    tmodel, tspec = return_antstorch_unet(task, weights_path=None, strict=False, verbose=verbose)

    dimension = int(tspec.get("dimension", 3))
    ndim = dimension + 2  # torch conv weight rank: (outC, inC, *spatial)

    # Determine attention from spec unless explicitly provided
    if attention_gating is None:
        opts = tspec.get("additional_options") or []
        if isinstance(opts, str):
            opts = [opts]
        attention_gating = any(o in ("attentionGating", "attention_gating") for o in opts)

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
        spatial_shape = (min_side,) * dimension
        try:
            _ = tmodel(torch.zeros(1, in_ch, *spatial_shape))
        except Exception:
            # try a fixed small size if exact power-of-two fails
            fallback_shape = (32,) * dimension
            _ = tmodel(torch.zeros(1, in_ch, *fallback_shape))

    # Keras outputs and aux names
    out_layer_names = _keras_output_layer_names(kmodel)
    n_aux = max(0, len(out_layer_names) - 1)
    if verbose:
        print(f"[convert] dimension={dimension} Keras outputs: {out_layer_names} -> aux heads: {n_aux}")

    aux_names = out_layer_names[1:]

    # Collect Keras convs/deconvs and main output
    k_convs, k_dect, k_main, k_attn = _collect_keras_convs_and_dect(
        kmodel, exclude_layer_names=aux_names, attention_gating=attention_gating, dimension=dimension)

    # Collect Torch keys
    t_dect = _list_deconv_keys(sd_view, ndim)
    t_enc, t_dec = _list_conv_keys(sd_view, ndim)
    t_attn = _list_attention_keys(sd_view) if attention_gating else []
    if verbose:
        def prof_dect(lst):  # show (in->out) channel profile
            out = []
            for k in lst:
                v = sd_view[k]
                # ConvTranspose weight shape: (inC, outC, *spatial)
                inC, outC = int(v.shape[0]), int(v.shape[1])
                out.append((inC, outC))
            return out
        if dimension == 3:
            print(f"[profile] Keras deconv (in->out) per layer: {[(w.shape[3], w.shape[4]) for (_, w, _) in k_dect]}")
        else:
            print(f"[profile] Keras deconv (in->out) per layer: {[(w.shape[2], w.shape[3]) for (_, w, _) in k_dect]}")
        print(f"[profile] Torch  deconv (in->out) per layer: {prof_dect(t_dect)}")

    # Basic sanity checks
    if len(k_dect) != len(t_dect):
        raise RuntimeError(
            f"Deconv count mismatch: Keras={len(k_dect)} vs Torch={len(t_dect)}. "
            f"Patterns may differ; adjust _list_deconv_keys()."
        )

    total_t_convs = len(t_enc) + len(t_dec)
    if len(k_convs) != total_t_convs:
        if attention_gating:
            # often extra 1x1(x1) convs in Keras were siphoned into k_attn; allow mismatch here
            pass
        else:
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
            alt = _transpose_in_out(torchW)
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
        if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
            torchW = _transpose_in_out(torchW)
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
        if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
            torchW = _transpose_in_out(torchW)
        new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
        if kB is not None:
            bkey = _resolve_tkey(assign_key.replace(".weight", ".bias"), new_sd)
            if bkey in new_sd and new_sd[bkey].numel() == kB.size:
                new_sd[bkey] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[bkey].dtype)

    # ---------- Assign main output head ----------
    main_name, kW_main, kB_main = k_main
    out_wkey_view, out_bkey_view = _find_main_out_keys(sd_view, int(tspec["number_of_outputs"]), ndim)
    out_wkey = base_prefix + out_wkey_view
    out_bkey = base_prefix + out_bkey_view if out_bkey_view else None

    torchW = _to_torch_conv_weight_from_keras(kW_main)
    rk_out = _resolve_tkey(out_wkey, new_sd)
    tgt = new_sd[rk_out].cpu().numpy()
    if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
        torchW = _transpose_in_out(torchW)
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
            if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
                torchW = _transpose_in_out(torchW)
            new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
            if kB is not None:
                bkey = f"heads.{i}.bias"
                rb = _resolve_tkey(bkey, new_sd)
                if rb in new_sd and new_sd[rb].numel() == kB.size:
                    new_sd[rb] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[rb].dtype)

    # Load into model (non-strict to tolerate any unused buffers) then save just the state_dict

    # ---------- Assign attention gate weights ----------
    if attention_gating:
        if verbose:
            print(f"[convert] Attention gating detected. Keras attn convs: {len(k_attn)}, Torch gates: {len(t_attn)}")
        # Expect triplets per level: theta (inter), phi (inter), psi (1)
        if len(t_attn) * 3 != len(k_attn):
            # Best-effort: continue if numbers differ, but warn.
            print(f"[warn] Attention conv count mismatch: keras={len(k_attn)} vs torch triplets={len(t_attn)*3}")
        # Assign in order; try to pair by expected out_channels (psi has outC==1)
        ki = 0
        for lvl, keys in enumerate(t_attn):
            # theta
            if ki < len(k_attn):
                _, kW, kB = k_attn[ki]; ki += 1
                torchW = _to_torch_conv_weight_from_keras(kW)
                rk = _resolve_tkey(base_prefix + keys["theta_w"], new_sd)
                tgt = new_sd[rk].cpu().numpy()
                if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
                    torchW = _transpose_in_out(torchW)
                new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
                if kB is not None and keys["theta_b"] and keys["theta_b"] in new_sd and new_sd[keys["theta_b"]].numel() == kB.size:
                    new_sd[keys["theta_b"]] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[keys["theta_b"]].dtype)
            # phi
            if ki < len(k_attn):
                _, kW, kB = k_attn[ki]; ki += 1
                torchW = _to_torch_conv_weight_from_keras(kW)
                rk = _resolve_tkey(base_prefix + keys["phi_w"], new_sd)
                tgt = new_sd[rk].cpu().numpy()
                if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
                    torchW = _transpose_in_out(torchW)
                new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
                if kB is not None and keys["phi_b"] and keys["phi_b"] in new_sd and new_sd[keys["phi_b"]].numel() == kB.size:
                    new_sd[keys["phi_b"]] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[keys["phi_b"]].dtype)
            # psi (filters=1)
            if ki < len(k_attn):
                # attempt to pick the next 1-filter conv among the next two if ordering differs
                cand_idx = ki
                chosen = cand_idx
                for j in range(ki, min(ki+2, len(k_attn))):
                    _, kWcand, _ = k_attn[j]
                    if int(kWcand.shape[-1]) == 1:  # Keras Conv kernel: (..., inC, outC)
                        chosen = j
                        break
                name, kW, kB = k_attn[chosen]
                # swap if chose the second in the window
                if chosen != ki:
                    k_attn[ki], k_attn[chosen] = k_attn[chosen], k_attn[ki]
                ki += 1
                torchW = _to_torch_conv_weight_from_keras(kW)
                rk = _resolve_tkey(base_prefix + keys["psi_w"], new_sd)
                tgt = new_sd[rk].cpu().numpy()
                if torchW.shape != tgt.shape and _transpose_in_out(torchW).shape == tgt.shape:
                    torchW = _transpose_in_out(torchW)
                new_sd[rk] = torch.from_numpy(torchW).to(new_sd[rk].dtype)
                if kB is not None and keys["psi_b"] and keys["psi_b"] in new_sd and new_sd[keys["psi_b"]].numel() == kB.size:
                    new_sd[keys["psi_b"]] = torch.from_numpy(kB.astype(np.float32)).to(new_sd[keys["psi_b"]].dtype
                                                                                       )
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
    p.add_argument("--attention-gating", type=str, default="auto", choices=["auto","on","off"], help="Convert attention-gating weights if present.")
    args = p.parse_args()

    deconv_flip = (args.deconv_flip == "flip")
    attn_flag = None if args.attention_gating == "auto" else (args.attention_gating == "on")
    convert_task(args.task, args.out_prefix, deconv_flip=deconv_flip, verbose=args.verbose, attention_gating=attn_flag)


if __name__ == "__main__":
    sys.exit(main())
