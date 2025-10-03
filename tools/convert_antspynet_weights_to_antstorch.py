#!/usr/bin/env python3
# See header in previous attempt; full, self-contained script below.

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,
    Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D
)
from tensorflow.keras.models import Model

def _dect_channel_profile_keras(k_dect):
    # Returns list of (in_c, out_c) per deconv layer in stage order
    prof = []
    for s in sorted(k_dect.keys()):
        stage = k_dect[s]
        for name, w in stage:
            # Keras Conv3DTranspose kernel: (kD,kH,kW,out_c,in_c)
            ker = w[0]
            if ker.ndim == 5:
                out_c, in_c = ker.shape[3], ker.shape[4]
            elif ker.ndim == 4:  # 2D
                out_c, in_c = ker.shape[2], ker.shape[3]
            else:
                continue
            prof.append((int(in_c), int(out_c)))
    return prof

def _dect_channel_profile_torch(t_dect, state_dict):
    prof = []
    for s in range(len(t_dect)):
        for tw, tb in t_dect[s]:
            ker = state_dict[tw].cpu().numpy()
            # Torch ConvTranspose3d weight: (in_c, out_c, kD,kH,kW)
            if ker.ndim == 5:
                in_c, out_c = ker.shape[0], ker.shape[1]
            elif ker.ndim == 4:
                in_c, out_c = ker.shape[0], ker.shape[1]
            else:
                continue
            prof.append((int(in_c), int(out_c)))
    return prof

def _reindex_dect_by_inchannels_desc(dect_dict):
    """Return a new dict with stages reindexed 0..n-1 ordered by decreasing in_channels."""
    if not isinstance(dect_dict, dict) or not dect_dict:
        return dect_dict
    stage_info = []
    for s in sorted(dect_dict.keys()):
        stage = dect_dict[s]
        in_c = -1
        for _name, w in stage:
            ker = w[0]
            if ker.ndim == 5:
                # Keras deconv: (kD,kH,kW,out_c,in_c)
                in_c = int(ker.shape[4])
                break
            elif ker.ndim == 4:
                in_c = int(ker.shape[3])
                break
        stage_info.append((s, in_c))
    # Order deepest -> shallowest by in_channels
    order = [s for (s, _ic) in sorted(stage_info, key=lambda x: -x[1])]
    return {i: dect_dict[s] for i, s in enumerate(order)}

def _reindex_consecutive_drop_empty(stage_dict):
    """Return dict with stages reindexed 0..n-1 in ascending original key order, dropping empty blocks."""
    if not isinstance(stage_dict, dict) or not stage_dict:
        return stage_dict
    ordered = []
    for k in sorted(stage_dict.keys()):
        block = stage_dict[k]
        if block:  # drop empties
            ordered.append(block)
    return {i: ordered[i] for i in range(len(ordered))}




def _import_factories():
    try:
        from return_antspynet_unet import return_antspynet_unet as k_fn
    except Exception:
        from return_antspynet_unet_for_task import return_antspynet_unet_for_task as k_fn
    try:
        from return_antstorch_unet import return_antstorch_unet as t_fn
    except Exception:
        from return_antstorch_unet_for_task import return_antstorch_unet_for_task as t_fn
    return k_fn, t_fn

def _np():
    import numpy as _n
    return _n

def _to_torch_conv_weight_from_keras(w_keras: np.ndarray) -> np.ndarray:
    w = _np().transpose(w_keras, (4, 3, 0, 1, 2))
    return _np().ascontiguousarray(w)

def _to_torch_deconv_weight_from_keras(w_keras: np.ndarray, flip_xyz: bool = False) -> np.ndarray:
    w = _np().flip(w_keras, axis=(0, 1, 2)) if flip_xyz else w_keras
    w = _np().transpose(w, (3, 4, 0, 1, 2))  # (Cin, Cout, Kx, Ky, Kz)
    return _np().ascontiguousarray(w)

def _is_conv(layer) -> bool:
    return isinstance(layer, (Conv2D, Conv3D))

def _is_deconv(layer) -> bool:
    return isinstance(layer, (Conv2DTranspose, Conv3DTranspose))

def _is_pool(layer) -> bool:
    return isinstance(layer, (MaxPooling2D, MaxPooling3D))

def _is_upsample(layer) -> bool:
    return isinstance(layer, (UpSampling2D, UpSampling3D))

def collect_keras_unet_ordered(model: Model):
    enc = {}
    dec = {}
    dect = {}
    out = []

    in_decoder = False
    enc_stage = 0
    dec_stage = -1
    last_spatial = None

    for l in model.layers:
        # Try to infer spatial dims from output shape
        cur_shape = getattr(l, "output_shape", None)
        if isinstance(cur_shape, (tuple, list)) and len(cur_shape) >= 4:
            if len(cur_shape) == 5:
                cur_spatial = tuple(cur_shape[1:4])  # 3D
            else:
                cur_spatial = tuple(cur_shape[1:3])  # 2D
        else:
            cur_spatial = last_spatial

        if _is_pool(l):
            if not in_decoder:
                enc_stage += 1
            last_spatial = cur_spatial
            continue

        if _is_upsample(l):
            if not in_decoder:
                in_decoder = True
            dec_stage += 1  # each upsample starts a new decoder stage
            last_spatial = cur_spatial
            continue

        if _is_deconv(l):
            if not in_decoder:
                in_decoder = True
                dec_stage = 0
            else:
                if last_spatial is not None and cur_spatial is not None and cur_spatial != last_spatial:
                    dec_stage += 1
            dect.setdefault(dec_stage, []).append((l.name, l.get_weights()))
            last_spatial = cur_spatial
            continue

        if _is_conv(l):
            w = l.get_weights()
            if w:
                if not in_decoder:
                    enc.setdefault(enc_stage, []).append((l.name, w))
                else:
                    dec.setdefault(max(dec_stage, 0), []).append((l.name, w))
            last_spatial = cur_spatial
            continue

    # Try to peel off a 1x1 (2D) or 1x1x1 (3D) conv at the tail as output head
    if dec:
        last_stage = max(dec.keys())
        last_block = dec.get(last_stage, [])
        if last_block:
            lname, lw = last_block[-1]
            if lw and len(lw) > 0:
                kshape = lw[0].shape
                is_1x1_2d = (len(kshape) == 4 and kshape[0:2] == (1, 1))
                is_1x1_3d = (len(kshape) == 5 and kshape[0:3] == (1, 1, 1))
                if is_1x1_2d or is_1x1_3d:
                    out.append((lname, lw))
                    dec[last_stage] = last_block[:-1]
    dect = _reindex_dect_by_inchannels_desc(dect)
    enc = _reindex_consecutive_drop_empty(enc)
    dec = _reindex_consecutive_drop_empty(dec)
    return enc, dect, dec, out

def collect_torch_unet_ordered(state_dict: Dict[str, torch.Tensor]):
    sd = state_dict
    def has_key(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in sd.keys())

    groups = {"enc": [], "dect": [], "dec": [], "out": []}

    i = 0
    while has_key(f"encoding_convolution_layers.{i}"):
        w0 = f"encoding_convolution_layers.{i}.0.weight"
        b0 = f"encoding_convolution_layers.{i}.0.bias"
        w1 = f"encoding_convolution_layers.{i}.2.weight"
        b1 = f"encoding_convolution_layers.{i}.2.bias"
        stage = []
        if w0 in sd: stage.append((w0, b0 if b0 in sd else None))
        if w1 in sd: stage.append((w1, b1 if b1 in sd else None))
        if stage:
            groups["enc"].append(stage)
        i += 1

    i = 0
    while has_key(f"decoding_convolution_transpose_layers.{i}"):
        wkey1 = f"decoding_convolution_transpose_layers.{i}.deconv.weight"
        bkey1 = f"decoding_convolution_transpose_layers.{i}.deconv.bias"
        wkey2 = f"decoding_convolution_transpose_layers.{i}.weight"
        bkey2 = f"decoding_convolution_transpose_layers.{i}.bias"
        if wkey1 in sd or wkey2 in sd:
            w = wkey1 if wkey1 in sd else wkey2
            b = bkey1 if wkey1 in sd else (bkey2 if bkey2 in sd else None)
            groups["dect"].append([(w, b)])
        i += 1

    i = 0
    while has_key(f"decoding_convolution_layers.{i}"):
        w0 = f"decoding_convolution_layers.{i}.0.weight"
        b0 = f"decoding_convolution_layers.{i}.0.bias"
        w1 = f"decoding_convolution_layers.{i}.2.weight"
        b1 = f"decoding_convolution_layers.{i}.2.bias"
        stage = []
        if w0 in sd: stage.append((w0, b0 if b0 in sd else None))
        if w1 in sd: stage.append((w1, b1 if b1 in sd else None))
        if stage:
            groups["dec"].append(stage)
        i += 1

    if "output.0.weight" in sd:
        groups["out"] = [("output.0.weight", "output.0.bias" if "output.0.bias" in sd else None)]
    elif "output.weight" in sd:
        groups["out"] = [("output.weight", "output.bias" if "output.bias" in sd else None)]
    return groups

def convert_task(task: str, out_prefix: str, deconv_flip: bool = False, verbose: bool = True) -> str:
    k_factory, t_factory = _import_factories()

    kmodel, kmeta = k_factory(task, load_weights=True, verbose=verbose)
    tmodel, tmeta = t_factory(task, weights_path=None, strict=False, verbose=verbose)

    device = torch.device("cpu")
    tmodel = tmodel.to(device)

    # Work on a detached editable copy of the state_dict
    new_sd = {k: v.clone().detach() for k, v in tmodel.state_dict().items()}


    if verbose:
        print(f"[convert] Converting weights for task='{task}'")

    k_enc, k_dect, k_dec, k_out = collect_keras_unet_ordered(kmodel)
    k_blocks = {'enc': k_enc, 'dect': k_dect, 'dec': k_dec, 'out': k_out}
    t_sd = tmodel.state_dict()
    t_blocks = collect_torch_unet_ordered(t_sd)

    # --- Channel-profile diagnostic (helps spot factory width mismatches) ---
    keras_dect_prof = _dect_channel_profile_keras(k_blocks['dect'])
    torch_dect_prof = _dect_channel_profile_torch(t_blocks['dect'], new_sd)
    if verbose:
        print(f"[profile] Keras deconv (in->out) per layer: {keras_dect_prof}")
        print(f"[profile] Torch  deconv (in->out) per layer: {torch_dect_prof}")
    # Quick guard: compare multisets of (in,out) ignoring order within stages
    if len(keras_dect_prof) == len(torch_dect_prof):
        mismatches = [(k, t) for k, t in zip(keras_dect_prof, torch_dect_prof) if k != t]
    else:
        mismatches = [("len", (len(keras_dect_prof), len(torch_dect_prof)))]
    if mismatches:
        # If the first mismatch shows clear scale differences (e.g., 32->64 vs 128->64),
        # it's almost always a width schedule mismatch between factories.
        k_hint = sorted(set([k for k,_ in keras_dect_prof]))
        # Suggest encoder base filters from Keras (first encoder conv out channels)
        try:
            first_enc = k_blocks['enc'][0][0][1][0]  # kernel for first conv in first stage
            if first_enc.ndim == 5:
                k_base = int(first_enc.shape[-1])
            else:
                k_base = int(first_enc.shape[-1])
        except Exception:
            k_base = None
        raise RuntimeError(
            "Channel profile mismatch between Keras and Torch UNet factories.\n"
            f"Keras deconv (in->out): {keras_dect_prof}\n"
            f"Torch  deconv (in->out): {torch_dect_prof}\n"
            "Likely cause: Torch factory width schedule differs from Keras for this task.\n"
            "Fix: Build the Torch UNet with the SAME per-stage filter widths as the Keras model.\n"
            + (f"Hint: Keras base encoder out-ch seems to be ~{k_base}. Configure your antstorch factory accordingly." if k_base else "" ) + "\n"
            "After aligning widths, rerun the converter."
        )

    new_sd = dict(t_sd)
    mapping = {"enc": [], "dect": [], "dec": [], "out": []}

    def assign(t_key: str, src_np: np.ndarray, what: str):
        tgt = new_sd[t_key].cpu().numpy()
        if tgt.shape != src_np.shape:
            raise ValueError(f"Shape mismatch assigning {what}: target {t_key} {tgt.shape} vs src {src_np.shape}")
        new_sd[t_key] = torch.from_numpy(src_np).to(new_sd[t_key].dtype)

    # enc
    n_enc = min(len(k_blocks["enc"]), len(t_blocks["enc"]))
    for s in range(n_enc):
        k_stage = k_blocks["enc"][s]
        t_stage = t_blocks["enc"][s]
        n_slots = min(len(k_stage), len(t_stage))
        for j in range(n_slots):
            kname, kw = k_stage[j]
            tw, tb = t_stage[j]
            kmapped = _to_torch_conv_weight_from_keras(kw[0])
            assign(tw, kmapped, f"enc[{s}].{j}.weight")
            if len(kw) > 1 and tb is not None:
                assign(tb, kw[1].astype(np.float32), f"enc[{s}].{j}.bias")
            mapping["enc"].append({"keras": kname, "torch_w": tw, "torch_b": tb})

    # dect
        n_dect = min(len(k_blocks["dect"]), len(t_blocks["dect"]))
    for s in range(n_dect):
        k_stage = k_blocks["dect"][s]
        t_stage = t_blocks["dect"][s]
        used = set()
        for j, (kname, kw) in enumerate(k_stage):
            km0 = _to_torch_deconv_weight_from_keras(kw[0], flip_xyz=deconv_flip)
            km1 = km0.transpose(1, 0, 2, 3, 4)  # Cin<->Cout swap candidate
            matched = False
            for tidx, (tw, tb) in enumerate(t_stage):
                if tidx in used:
                    continue
                t_weight = new_sd[tw].cpu().numpy()
                if km0.shape == t_weight.shape:
                    assign(tw, km0, f"dect[{s}].{j}.weight")
                    if len(kw) > 1 and tb is not None:
                        assign(tb, kw[1].astype(np.float32), f"dect[{s}].{j}.bias")
                    mapping["dect"].append({"keras": kname, "torch_w": tw, "torch_b": tb})
                    used.add(tidx)
                    matched = True
                    break
                if km1.shape == t_weight.shape:
                    assign(tw, km1, f"dect[{s}].{j}.weight")
                    if len(kw) > 1 and tb is not None:
                        assign(tb, kw[1].astype(np.float32), f"dect[{s}].{j}.bias")
                    mapping["dect"].append({"keras": kname, "torch_w": tw, "torch_b": tb})
                    used.add(tidx)
                    matched = True
                    break
            if not matched:
                # Helpful diagnostics
                t_shapes = [new_sd[tw].cpu().numpy().shape for (tw, _tb) in t_stage if t_stage.index((tw, _tb)) not in used]
                raise ValueError(
                    f"Could not match Keras deconv at stage {s}, idx {j} (km0 {km0.shape} / km1 {km1.shape}) "
                    f"to any remaining Torch shapes {t_shapes}"
                )

    # dec
    n_dec = min(len(k_blocks["dec"]), len(t_blocks["dec"]))
    for s in range(n_dec):
        k_stage = k_blocks["dec"][s]
        t_stage = t_blocks["dec"][s]
        n_slots = min(len(k_stage), len(t_stage))
        for j in range(n_slots):
            kname, kw = k_stage[j]
            tw, tb = t_stage[j]
            kmapped = _to_torch_conv_weight_from_keras(kw[0])
            assign(tw, kmapped, f"dec[{s}].{j}.weight")
            if len(kw) > 1 and tb is not None:
                assign(tb, kw[1].astype(np.float32), f"dec[{s}].{j}.bias")
            mapping["dec"].append({"keras": kname, "torch_w": tw, "torch_b": tb})

    # out
    if k_blocks["out"] and t_blocks["out"]:
        kname, kw = k_blocks["out"][-1]
        tw, tb = t_blocks["out"][-1]
        kmapped = _to_torch_conv_weight_from_keras(kw[0])
        t_weight = new_sd[tw].cpu().numpy()
        if kmapped.shape != t_weight.shape and kmapped.transpose(1,0,2,3,4).shape == t_weight.shape:
            kmapped = kmapped.transpose(1,0,2,3,4)
        assign(tw, kmapped, "out.weight")
        if len(kw) > 1 and tb is not None:
            assign(tb, kw[1].astype(np.float32), "out.bias")
        mapping["out"].append({"keras": kname, "torch_w": tw, "torch_b": tb})

    # strict load
    tmodel.load_state_dict(new_sd, strict=True)

    out_path = f"{out_prefix}.pt"
    torch.save(tmodel.state_dict(), out_path)
    with open(f"{out_prefix}_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--deconv-flip", choices=["flip", "noflip"], default="noflip")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = convert_task(args.task, args.out_prefix, deconv_flip=(args.deconv_flip == "flip"), verbose=args.verbose)
    print(f"[convert] Saved: {out}")
    print(f"[convert] Mapping JSON: {args.out_prefix}_mapping.json")

if __name__ == "__main__":
    main()
