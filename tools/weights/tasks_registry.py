#!/usr/bin/env python3
"""
tasks_registry.py

Single source of truth for U-Net task specs shared by ANTsPyNet (Keras) and
ANTsTorch (PyTorch) factories.

- Keep only COMMON architectural hyperparameters here.
- Weight filename conventions are provided via helper functions.

NOTE on new (lung / mouse) task entries added below:

Only tasks whose architecture hyperparameters (channel count, number of
outputs, filters, kernel sizes) are *certain* -- i.e. directly stated or
unambiguously derivable from the antspynet/antstorch source, not merely
guessed -- were added here. In particular the following are intentionally
NOT registered because their input channel count depends on data read at
runtime (number of prior label images in an atlas, etc.) and hardcoding a
guessed value here would risk silently producing an incorrectly-shaped
conversion:

    * lung "protonLobes" / "maskLobes"  (channel_size = 1 + len(protonLobePriors))
    * lung "ct"                          (channel_size = 1 + len(luna16LungPriors))
    * mouse_brain_parcellation "jay"     (channel_size = 1 + number_of_nonzero_labels,
                                           and unlike "nick"/"tct" the docstring does
                                           not enumerate the label count)

Also NOT registered/supported by this generic-UNet-based registry+tool at
all: create_sysu_media_unet_model_2d/3d, create_hypermapp3r_unet_model_3d,
create_shiva_unet_model_3d, and create_deep_back_projection_network_model_2d.
These are bespoke architectures (not create_unet_model_2d/3d-based) with
their own layer naming conventions, so convert_antspynet_weights_to_antstorch.py
(which assumes the generic UNet's "encoding_convolution_layers" /
"decoding_convolution_layers" / "decoding_convolution_transpose_layers" /
"heads.N" / "attn_gates_{2d,3d}" module-name conventions) cannot be reused
for them as-is. Converting their weights requires dedicated per-architecture
scripts -- out of scope here.
"""

from __future__ import annotations
# --- add near the top of the file, after imports ---
from typing import Dict, Any, List, Optional

# Default fillers so older tasks remain valid
_DEFAULTS: Dict[str, Any] = {
    "dimension": 3,
    "n_aux_heads": 0,
    "aux_head_names": None,  # or [] if you prefer strict lists
    "additional_options": None,
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
    "dkt_labeling": _with_defaults(dict(
        dimension=3,
        input_image_size=(160, 192, 160, 1),
        number_of_outputs=38,
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=[],
        n_aux_heads=1,
        aux_head_names=[],  # optional, purely metadata
    )),
    "cerebellum_whole": _with_defaults(dict(
        dimension=3,
        input_image_size=(240, 144, 144, 2),
        number_of_outputs=2,
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=['attentionGating'],
        n_aux_heads=0,
        aux_head_names=[],  # optional, purely metadata
    )),
    "cerebellum_tissue": _with_defaults(dict(
        dimension=3,
        input_image_size=(240, 144, 144, 4),
        number_of_outputs=4,
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=[],
        n_aux_heads=0,
        aux_head_names=[],  # optional, purely metadata
    )),
    "cerebellum_labels": _with_defaults(dict(
        dimension=3,
        input_image_size=(240, 144, 144, 25),
        number_of_outputs=25,
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=['attentionGating'],
        n_aux_heads=0,
        aux_head_names=[],  # optional, purely metadata
    )),

    # ------------------------------------------------------------------
    # lung_extraction / lung_segmentation (generic-UNet-compatible only;
    # see module docstring for tasks intentionally NOT registered)
    # ------------------------------------------------------------------
    "lung_proton": _with_defaults(dict(
        dimension=3,
        input_image_size=(128, 128, 128, 1),   # placeholder spatial size; actual
                                                # runtime shape is data-dependent
                                                # (protonLungTemplate.shape) but conv
                                                # kernel shapes (and hence conversion)
                                                # do not depend on it.
        number_of_outputs=3,                   # background, left_lung, right_lung
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(7, 7, 5),
        deconvolution_kernel_size=(7, 7, 5),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
    )),
    "lung_ventilation": _with_defaults(dict(
        dimension=2,
        input_image_size=(256, 256, 1),
        number_of_outputs=1,                   # sigmoid whole-lung mask
        number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="sigmoid",
    )),
    "lung_xray": _with_defaults(dict(
        dimension=2,
        input_image_size=(256, 256, 3),        # image + 2 xray lung priors
        number_of_outputs=3,                   # background, left_lung, right_lung
        number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
    )),

    # ------------------------------------------------------------------
    # mouse.py (generic-UNet-compatible only; see module docstring for
    # tasks intentionally NOT registered, e.g. "jay" parcellation)
    # ------------------------------------------------------------------
    "mouse_t2_brain_extraction": _with_defaults(dict(
        dimension=3,
        input_image_size=(176, 176, 176, 1),
        number_of_outputs=1,                   # sigmoid
        number_of_filters=(16, 32, 64, 128),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="sigmoid",
    )),
    "mouse_ex5_coronal": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=2,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_ex5_sagittal": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=2,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_histology_brain_mask": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=2,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_histology_hemispherical_coronal_mask": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=3,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="classification",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_cerebellum_mask_sagittal": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=1,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="sigmoid",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_cerebellum_mask_coronal": _with_defaults(dict(
        dimension=2,
        input_image_size=(512, 512, 1),
        number_of_outputs=1,
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3),
        deconvolution_kernel_size=(2, 2),
        pool_size=(2, 2),
        strides=(2, 2),
        dropout_rate=0.0,
        mode="sigmoid",
        additional_options=["initialConvolutionKernelSize[5]", "attentionGating"],
    )),
    "mouse_brain_parcellation_nick": _with_defaults(dict(
        dimension=3,
        input_image_size=(176, 128, 240, 7),   # 1 image + 6 label priors
        number_of_outputs=7,
        number_of_filters=(16, 32, 64, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
    )),
    "mouse_brain_parcellation_tct": _with_defaults(dict(
        dimension=3,
        input_image_size=(176, 128, 240, 8),   # 1 image + 7 label priors (verified
                                                # against the real pretrained weights:
                                                # first encoder conv has 8 input channels,
                                                # not 7 -- the docstring's "tct" label list
                                                # in mouse.py enumerates 7 nonzero labels)
        number_of_outputs=8,
        number_of_filters=(16, 32, 64, 128, 256),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        dropout_rate=0.0,
        mode="classification",
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
    "dkt_labeling": "DesikanKillianyTourvilleOuter",
    "cerebellum_whole": "cerebellumWhole",
    "cerebellum_tissue": "cerebellumTissue",
    "cerebellum_labels": "cerebellumLabels",

    "lung_proton": "protonLungMri",
    "lung_ventilation": "wholeLungMaskFromVentilation",
    "lung_xray": "xrayLungExtraction",

    "mouse_t2_brain_extraction": "mouseT2wBrainExtraction3D",
    "mouse_ex5_coronal": "ex5_coronal_weights",
    "mouse_ex5_sagittal": "ex5_sagittal_weights",
    "mouse_histology_brain_mask": "allen_brain_mask_weights",
    "mouse_histology_hemispherical_coronal_mask": "allen_brain_leftright_coronal_mask_weights",
    "mouse_cerebellum_mask_sagittal": "allen_cerebellum_sagittal_mask_weights",
    "mouse_cerebellum_mask_coronal": "allen_cerebellum_coronal_mask_weights",
    "mouse_brain_parcellation_nick": "mouseT2wBrainParcellation3DNick",
    "mouse_brain_parcellation_tct": "mouseT2wBrainParcellation3DTct",
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
    if "n_aux_heads" not in spec or "dimension" not in spec:
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
