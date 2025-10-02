#!/usr/bin/env python3
"""
return_antspynet_unet.py  (refactored)

Build a Keras (ANTsPyNet) U-Net from a shared task registry and optionally load weights.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import antspynet
from tensorflow.keras import Model

from tasks_registry import get_task_spec, weights_prefix_for


def _create_unet_from_spec(spec: Dict[str, Any]) -> Model:
    return antspynet.architectures.create_unet_model_3d(
        input_image_size=spec["input_image_size"],
        number_of_outputs=spec["number_of_outputs"],
        number_of_filters=spec["number_of_filters"],
        convolution_kernel_size=spec["convolution_kernel_size"],
        deconvolution_kernel_size=spec["deconvolution_kernel_size"],
        pool_size=spec["pool_size"],
        strides=spec["strides"],
        dropout_rate=spec["dropout_rate"],
        mode=spec["mode"]
    )


def return_antspynet_unet(
    task: str,
    load_weights: bool = True,
    weights_file: str | None = None,
    verbose: bool = True,
) -> Tuple[Model, Dict[str, Any]]:
    spec = get_task_spec(task)
    kmodel = _create_unet_from_spec(spec)

    resolved_weights = None
    if load_weights:
        if weights_file is None:
            prefix = weights_prefix_for(task, "antspynet")
            if verbose:
                print(f"Getting antspynet network weights:  {prefix}")
            import antspynet as _apn
            resolved_weights = _apn.get_pretrained_network(prefix)
        else:
            resolved_weights = weights_file

        if verbose:
            print(f"[ANTsPyNet] Loading weights for task='{task}' from: {resolved_weights}")
        kmodel.load_weights(resolved_weights)

    meta = dict(task=task, spec=spec, weights_path=resolved_weights)
    return kmodel, meta


if __name__ == "__main__":
    m, meta = return_antspynet_unet("deep_atropos", load_weights=False)
    print(m.count_params(), "parameters")
