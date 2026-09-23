#!/usr/bin/env python3
"""
return_antspynet_unet.py  (refactored, 2D/3D-aware)

Build a Keras (ANTsPyNet) U-Net from a shared task registry and optionally load weights.

Dispatches on spec["dimension"] (2 or 3, default 3) between
antspynet.architectures.create_unet_model_2d / create_unet_model_3d.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import antspynet
from tensorflow.keras import Model

from tasks_registry import get_task_spec, weights_prefix_for

# --- helper: add N aux 1x1(x1) heads off the penultimate tensor ---
def _attach_aux_heads_keras(unet_model, n_aux_heads: int, dimension: int = 3):
    """
    Returns a new Keras Model whose outputs are:
      [unet_model.output, output1, output2, ...]
    where each output_i is a 1x1(x1) Conv (filters=1, activation='sigmoid')
    applied to the penultimate tensor (input to the final conv).
    """
    from tensorflow.keras.models import Model

    if dimension == 3:
        from tensorflow.keras.layers import Conv3D as ConvND
        kernel_size = (1, 1, 1)
    elif dimension == 2:
        from tensorflow.keras.layers import Conv2D as ConvND
        kernel_size = (1, 1)
    else:
        raise ValueError("dimension must be 2 or 3.")

    # penultimate = input to last ConvND in the UNet (i.e., second to last layer's output)
    penultimate = unet_model.layers[-2].output

    aux_outs = []
    for i in range(n_aux_heads):
        aux = ConvND(
            filters=1, kernel_size=kernel_size,
            activation='sigmoid', name=f'aux_head_{i+1}'
        )(penultimate)
        aux_outs.append(aux)

    return Model(inputs=unet_model.input, outputs=[unet_model.output, *aux_outs])

def _create_unet_from_spec(spec: Dict[str, Any]) -> Model:
    dimension = int(spec.get("dimension", 3))
    if dimension == 3:
        return antspynet.architectures.create_unet_model_3d(
            input_image_size=spec["input_image_size"],
            number_of_outputs=spec["number_of_outputs"],
            number_of_filters=spec["number_of_filters"],
            convolution_kernel_size=spec["convolution_kernel_size"],
            deconvolution_kernel_size=spec["deconvolution_kernel_size"],
            pool_size=spec["pool_size"],
            strides=spec["strides"],
            dropout_rate=spec["dropout_rate"],
            mode=spec["mode"],
            additional_options=spec.get("additional_options")
        )
    elif dimension == 2:
        return antspynet.architectures.create_unet_model_2d(
            input_image_size=spec["input_image_size"],
            number_of_outputs=spec["number_of_outputs"],
            number_of_filters=spec["number_of_filters"],
            convolution_kernel_size=spec["convolution_kernel_size"],
            deconvolution_kernel_size=spec["deconvolution_kernel_size"],
            pool_size=spec["pool_size"],
            strides=spec["strides"],
            dropout_rate=spec["dropout_rate"],
            mode=spec["mode"],
            additional_options=spec.get("additional_options")
        )
    else:
        raise ValueError(f"Unsupported dimension {dimension} (must be 2 or 3).")

def return_antspynet_unet(
    task: str,
    load_weights: bool = True,
    weights_file: str | None = None,
    verbose: bool = True,
) -> Tuple[Model, Dict[str, Any]]:

    spec = get_task_spec(task)
    dimension = int(spec.get("dimension", 3))
    kmodel = _create_unet_from_spec(spec)

    n_aux = int(spec.get("n_aux_heads", 0) or 0)
    if n_aux > 0:
        if verbose:
            print(f"[antspynet] Attaching {n_aux} auxiliary head(s) for '{task}' (dimension={dimension})")
        kmodel = _attach_aux_heads_keras(kmodel, n_aux, dimension=dimension)

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
