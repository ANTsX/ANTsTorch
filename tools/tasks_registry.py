#!/usr/bin/env python3
"""
tasks_registry.py

Single source of truth for U-Net task specs shared by ANTsPyNet (Keras) and
ANTsTorch (PyTorch) factories.

- Keep only COMMON architectural hyperparameters here.
- Weight filename conventions are provided via helper functions.
"""

from __future__ import annotations
from typing import Dict, Any, List

# Common architectural specs per task
_TASKS: Dict[str, Dict[str, Any]] = {
    "deep_atropos_t1": dict(
        dimension=3,
        input_image_size=(192, 224, 192, 1+6),
        number_of_outputs=7,                  # 6-tissue segmentation
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification"
    ),
    "deep_atropos_t1_t2": dict(
        dimension=3,
        input_image_size=(192, 224, 192, 2+6),
        number_of_outputs=7,                  # 6-tissue segmentation
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification"
    ),
    "deep_atropos_t1_fa": dict(
        dimension=3,
        input_image_size=(192, 224, 192, 2+6),
        number_of_outputs=7,                  # 6-tissue segmentation
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification"
    ),
    "deep_atropos_t1_t2_fa": dict(
        dimension=3,
        input_image_size=(192, 224, 192, 3+6),
        number_of_outputs=7,                  # 6-tissue segmentation
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification"
    ),
    "brain_extraction_t1": dict(
        dimension=3,
        input_image_size=(None, None, None, 1),
        number_of_outputs=1,                  # sigmoid
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="sigmoid"
    ),
}

# Weight file naming conventions for each framework
# For ANTsPyNet (Keras), use the existing published prefixes
_ANTSPYNET_PREFIX: Dict[str, str] = {
    "deep_atropos_t1": "DeepAtroposHcpT1Weights",
    "deep_atropos_t1_t2": "DeepAtroposHcpT1T2Weights",
    "deep_atropos_t1_fa": "DeepAtroposHcpT1FAWeights",
    "deep_atropos_t1_t2_fa": "DeepAtroposHcpT1T2FAWeights",
    "brain_extraction_t1": "brainExtractionRobustT1",
}

# For ANTsTorch (PyTorch), we standardize on "<prefix>_pytorch.pt"
_ANTSTORCH_PREFIX: Dict[str, str] = {
    task: f"{prefix}_pytorch.pt" for task, prefix in _ANTSPYNET_PREFIX.items()
}


def get_task_spec(task: str) -> Dict[str, Any]:
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {list(_TASKS.keys())}")
    return _TASKS[task].copy()


def list_tasks() -> List[str]:
    return sorted(_TASKS.keys())


def weights_prefix_for(task: str, framework: str) -> str:
    if framework == "antspynet":
        if task not in _ANTSPYNET_PREFIX:
            raise ValueError(f"No ANTsPyNet prefix for '{task}'.")
        return _ANTSPYNET_PREFIX[task]
    elif framework == "antstorch":
        if task not in _ANTSTORCH_PREFIX:
            raise ValueError(f"No ANTsTorch prefix for '{task}'.")
        return _ANTSTORCH_PREFIX[task]
    else:
        raise ValueError("framework must be 'antspynet' or 'antstorch'")
