#!/usr/bin/env python3
"""
tasks_registry.py

Single source of truth for U-Net task specs shared by ANTsPyNet (Keras) and
ANTsTorch (PyTorch) factories.

- Keep only COMMON architectural hyperparameters here.
- Weight filename conventions are provided via helper functions.
"""

from __future__ import annotations
# --- add near the top of the file, after imports ---
from typing import Dict, Any, List, Optional

# Default fillers so older tasks remain valid
_DEFAULTS: Dict[str, Any] = {
    "n_aux_heads": 0,
    "aux_head_names": None,  # or [] if you prefer strict lists
}

def _with_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(_DEFAULTS)
    out.update(d or {})
    return out

# Common architectural specs per task
_TASKS: Dict[str, Dict[str, Any]] = {
    "brain_extraction_t1": _with_defaults(dict(
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
    )),
    "deep_atropos_t1": _with_defaults(dict(
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
    )),
    "deep_atropos_t1_t2": _with_defaults(dict(
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
    )),
    "deep_atropos_t1_fa": _with_defaults(dict(
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
    )),
    "deep_atropos_t1_t2_fa": _with_defaults(dict(
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
    )),

    "deep_flash_left_t1": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 1+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_right_t1": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 1+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_left_t1_ri": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 1+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_right_t1_ri": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 1+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    

    "deep_flash_left_both": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 2+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_right_both": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 2+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_left_both_ri": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 2+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),    
    "deep_flash_right_both_ri": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 2+7),
        number_of_outputs=8,                      # main head classes (e.g., bg vs ROI)
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=3,          
        aux_head_names=["mtl", "ec_peri_phc", "hipp"],  # optional, purely metadata
    )),  
    "hoa_labeling": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 176, 160, 1),
        number_of_outputs=23,  
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        n_aux_heads=1,          
        aux_head_names=[],  # optional, purely metadata
    )),  

}

# Weight file naming conventions for each framework
# For ANTsPyNet (Keras), use the existing published prefixes
_ANTSPYNET_PREFIX: Dict[str, str] = {
    "brain_extraction_t1": "brainExtractionRobustT1",
    "deep_atropos_t1": "DeepAtroposHcpT1Weights",
    "deep_atropos_t1_t2": "DeepAtroposHcpT1T2Weights",
    "deep_atropos_t1_fa": "DeepAtroposHcpT1FAWeights",
    "deep_atropos_t1_t2_fa": "DeepAtroposHcpT1T2FAWeights",
    "deep_flash_left_t1": "deepFlashLeftT1Hierarchical",
    "deep_flash_right_t1": "deepFlashRightT1Hierarchical",
    "deep_flash_left_both": "deepFlashLeftBothHierarchical",
    "deep_flash_right_both": "deepFlashRightBothHierarchical",
    "deep_flash_left_t1_ri": "deepFlashLeftT1Hierarchical_ri",
    "deep_flash_right_t1_ri": "deepFlashRightT1Hierarchical_ri",
    "deep_flash_left_both_ri": "deepFlashLeftBothHierarchical_ri",
    "deep_flash_right_both_ri": "deepFlashRightBothHierarchical_ri",
    "hoa_labeling": "HarvardOxfordAtlasSubcortical",
}

# For ANTsTorch (PyTorch), we standardize on "<prefix>_pytorch.pt"
_ANTSTORCH_PREFIX: Dict[str, str] = {
    task: f"{prefix}_pytorch.pt" for task, prefix in _ANTSPYNET_PREFIX.items()
}

def get_task_spec(task: str) -> Dict[str, Any]:
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'.")
    spec = _TASKS[task]
    # If older tasks weren’t wrapped with _with_defaults, still fill now:
    if "n_aux_heads" not in spec:
        spec = _with_defaults(spec)
        _TASKS[task] = spec
    return spec

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
