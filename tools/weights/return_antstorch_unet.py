#!/usr/bin/env python3
"""
return_antstorch_unet.py  (refactored, 2D/3D-aware)

Build a PyTorch (ANTsTorch) U-Net from a shared task registry and optionally load weights.

Dispatches on spec["dimension"] (2 or 3, default 3) between
create_unet_model_2d / create_unet_model_3d.  Auxiliary-head wrapping
(create_multihead_unet_model_3d) is currently only available for 3-D tasks
in ANTsTorch; a 2-D task with n_aux_heads > 0 will raise clearly rather than
silently building the wrong thing.
"""

from __future__ import annotations

from tasks_registry import get_task_spec
from antstorch import create_unet_model_2d, create_unet_model_3d, create_multihead_unet_model_3d
import torch.nn as nn
import torch

def return_antstorch_unet(task: str, weights_path: str | None = None, strict: bool = True, verbose: bool = False):
    spec = get_task_spec(task)
    dimension = int(spec.get("dimension", 3))
    in_ch = int(spec["input_image_size"][-1])
    spatial = tuple(spec["input_image_size"][:-1])
    n_aux = int(spec.get("n_aux_heads", 0) or 0)

    # 1) build base UNet (single-head) exactly as you already do
    common_kwargs = dict(
        input_channel_size=in_ch,
        number_of_outputs=spec["number_of_outputs"],
        number_of_filters=tuple(spec["number_of_filters"]),
        convolution_kernel_size=tuple(spec["convolution_kernel_size"]),
        deconvolution_kernel_size=tuple(spec["deconvolution_kernel_size"]),
        pool_size=tuple(spec["pool_size"]),
        strides=tuple(spec["strides"]),
        dropout_rate=float(spec["dropout_rate"]),
        mode=spec["mode"],
        pad_crop="keras",
        additional_options=spec.get("additional_options"),
    )
    if dimension == 3:
        base = create_unet_model_3d(**common_kwargs)
    elif dimension == 2:
        base = create_unet_model_2d(**common_kwargs)
    else:
        raise ValueError(f"Unsupported dimension {dimension} (must be 2 or 3).")

    # 2) wrap with the multi-head class if needed
    model = base
    if n_aux > 0:
        if dimension != 3:
            raise NotImplementedError(
                f"Task '{task}' requests {n_aux} auxiliary head(s) with dimension={dimension}, "
                "but create_multihead_unet_model_3d is only implemented for 3-D base UNets."
            )
        if verbose:
            print(f"[antstorch] Wrapping base UNet with {n_aux} aux head(s)")
        n_main = int(spec["number_of_outputs"])
        model = create_multihead_unet_model_3d(
            base_unet=base,
            n_aux_heads=n_aux,
            use_sigmoid=True,
            n_main_outputs=n_main
        )
        # 3) run a tiny warmup forward ONCE so the heads are instantiated
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *spatial)
            _ = model(dummy)  # this triggers the hook and builds heads

    # 4) load weights if provided
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if verbose:
            print(f"[antstorch] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:5] or unexpected[:5]:
                print("  e.g. missing[:5]   =", missing[:5])
                print("  e.g. unexpected[:5]=", unexpected[:5])

    return model, spec

if __name__ == "__main__":
    m, meta = return_antstorch_unet("deep_atropos", weights_path=None, verbose=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(n_params, "parameters")
