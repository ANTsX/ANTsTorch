#!/usr/bin/env python3
"""
return_antstorch_unet.py  (refactored)

Build a PyTorch (ANTsTorch) U-Net from a shared task registry and optionally load weights.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from tasks_registry import get_task_spec, weights_prefix_for
from antstorch.architectures import create_unet_model_3d


def _create_unet_from_spec(spec: Dict[str, Any]) -> nn.Module:
    # spec["input_image_size"] mirrors Keras, i.e., (D, H, W, C_in)
    spatial = tuple(spec["input_image_size"][:3])
    in_channels = int(spec["input_image_size"][-1])

    # Delegate to generic creator with dimensions=3
    return create_unet_model_3d(
        input_channel_size=in_channels,        # C_in
        number_of_outputs=spec["number_of_outputs"],
        number_of_filters=spec["number_of_filters"],
        convolution_kernel_size=spec["convolution_kernel_size"],
        deconvolution_kernel_size=spec["deconvolution_kernel_size"],
        pool_size=spec["pool_size"],
        strides=spec["strides"],
        dropout_rate=spec["dropout_rate"],
        mode=spec["mode"],
        pad_crop="keras"
    )


def return_antstorch_unet(
    task: str,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    spec = get_task_spec(task)
    model = _create_unet_from_spec(spec)

    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)

    resolved = weights_path
    if resolved is None:
        resolved = weights_prefix_for(task, "antstorch")  # e.g., deepAtropos_pytorch.pt

    if isinstance(resolved, str):
        try:
            sd = torch.load(resolved, map_location=dev)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if verbose:
                print(f"[ANTsTorch] Loading weights for task='{task}' from: {resolved}")
            model.load_state_dict(sd, strict=strict)
        except FileNotFoundError:
            if verbose:
                print(f"[ANTsTorch] Weights file not found: {resolved}. Returning uninitialized model.")

    meta = dict(task=task, spec=spec, weights_path=resolved)
    return model, meta


if __name__ == "__main__":
    m, meta = return_antstorch_unet("deep_atropos", weights_path=None, verbose=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(n_params, "parameters")
