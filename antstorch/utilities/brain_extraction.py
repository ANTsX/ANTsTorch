
# antstorch/utilities/brain_extraction.py

from __future__ import annotations

import re
import numpy as np
import torch
import torch.nn.functional as F
import ants

from ..architectures.create_unet_model import create_unet_model_3d
from ..utilities.get_pretrained_network import get_pretrained_network
from ..utilities.get_antstorch_data import get_antstorch_data


# -----------------------------
# Helper: ensure float image
# -----------------------------
def _ensure_float(img: ants.ANTsImage) -> ants.ANTsImage:
    return img if img.pixeltype == 'float' else img.clone('float')


def _to_tensor_list(imgs):
    arrs = [i.numpy().astype(np.float32) for i in imgs]  # each [X,Y,Z]
    stacked = np.stack(arrs, axis=0)  # [C, X, Y, Z]
    return torch.from_numpy(stacked)[None, ...]  # [1, C, D, H, W]


def _to_ants_like(npvol, like: ants.ANTsImage) -> ants.ANTsImage:
    out = ants.from_numpy(npvol)
    return ants.copy_image_info(like, out)


# -----------------------------
# Infer architecture from a .pt state_dict (no extra files needed)
# -----------------------------
def _infer_unet3d_from_state_dict(state_dict):
    # unwrap {"state_dict": ...}
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        sd = state_dict["state_dict"]
    else:
        sd = state_dict

    enc_filters = []
    pat = re.compile(r"encoding_convolution_layers\.(\d+)\..*\.0\.weight")  # first conv weight in block
    for k, w in sd.items():
        if not isinstance(w, torch.Tensor):
            continue
        m = pat.search(k)
        if m:
            i = int(m.group(1))
            outc = int(w.shape[0])  # [out,in,kD,kH,kW]
            while len(enc_filters) <= i:
                enc_filters.append(None)
            enc_filters[i] = outc

    enc_filters = [f for f in enc_filters if f is not None]
    if not enc_filters:
        # fallback guess
        enc_filters = [16, 32, 64, 128]

    # output channels
    head_w = None
    for k, w in sd.items():
        if isinstance(w, torch.Tensor) and (k.endswith("output.0.weight") or k.endswith("head.weight")):
            head_w = w; break
    out_channels = int(head_w.shape[0]) if head_w is not None else 1
    mode = "sigmoid" if out_channels == 1 else "classification"

    num_layers = len(enc_filters)
    return out_channels, enc_filters, num_layers, mode, sd


# -----------------------------
# Public API
# -----------------------------
def brain_extraction(image, modality, verbose: bool = False):
    """
    Mirror antspynet brain_extraction() behavior with Torch backbone:
      - same modality routing and templates
      - same center-of-mass Euler3D alignment
      - same intensity normalization policy
      - identical post-processing & inverse mapping
    """
    # ---------------------
    # Channel handling
    # ---------------------
    channel_size = 1
    if isinstance(image, list):
        channel_size = len(image)

    input_images = []
    if channel_size == 1:
        if modality in ("t1hemi", "t1lobes"):
            bext = brain_extraction(image, modality="t1threetissue", verbose=verbose)
            mask = ants.threshold_image(bext['segmentation_image'], 1, 1, 1, 0)
            input_images.append(image * mask)
        else:
            input_images.append(image)
    else:
        input_images = image

    if input_images[0].dimension != 3:
        raise ValueError("Image dimension must be 3.")

    input_images = [_ensure_float(i) for i in input_images]

    # ---------------------
    # Combined path
    # ---------------------
    if "t1combined" in modality:
        morphological_radius = 12
        if '[' in modality and ']' in modality:
            try:
                morphological_radius = int(modality.split("[")[1].split("]")[0])
            except Exception:
                pass

        brain_extraction_t1 = brain_extraction(image, modality="t1", verbose=verbose)
        brain_mask = ants.iMath_get_largest_component(ants.threshold_image(brain_extraction_t1, 0.5, 10000))
        brain_mask = ants.morphology(brain_mask, "close", morphological_radius).iMath_fill_holes()

        brain_extraction_t1nobrainer = brain_extraction(image * ants.iMath_MD(brain_mask, radius=morphological_radius),
                                                        modality="t1nobrainer", verbose=verbose)
        brain_extraction_combined = ants.iMath_fill_holes(
            ants.iMath_get_largest_component(brain_extraction_t1nobrainer * brain_mask))
        brain_extraction_combined = brain_extraction_combined + ants.iMath_ME(brain_mask, morphological_radius) + brain_mask
        return brain_extraction_combined

    # ---------------------
    # Modality routing
    # ---------------------
    if modality != "t1nobrainer":
        weights_prefix = None
        is_standard_network = False

        if modality == "t1.v0":
            weights_prefix = "brainExtraction"
        elif modality == "t1.v1":
            weights_prefix = "brainExtractionT1v1"; is_standard_network = True
        elif modality == "t1":
            weights_prefix = "brainExtractionRobustT1"; is_standard_network = True
        elif modality == "t2.v0":
            weights_prefix = "brainExtractionT2"
        elif modality == "t2":
            weights_prefix = "brainExtractionRobustT2"; is_standard_network = True
        elif modality == "t2star":
            weights_prefix = "brainExtractionRobustT2Star"; is_standard_network = True
        elif modality == "flair.v0":
            weights_prefix = "brainExtractionFLAIR"
        elif modality == "flair":
            weights_prefix = "brainExtractionRobustFLAIR"; is_standard_network = True
        elif modality == "bold.v0":
            weights_prefix = "brainExtractionBOLD"
        elif modality == "bold":
            weights_prefix = "brainExtractionRobustBOLD"; is_standard_network = True
        elif modality == "fa.v0":
            weights_prefix = "brainExtractionFA"
        elif modality == "fa":
            weights_prefix = "brainExtractionRobustFA"; is_standard_network = True
        elif modality == "mra":
            weights_prefix = "brainExtractionMra"; is_standard_network = True
        elif modality == "t1t2infant":
            weights_prefix = "brainExtractionInfantT1T2"
        elif modality == "t1infant":
            weights_prefix = "brainExtractionInfantT1"
        elif modality == "t2infant":
            weights_prefix = "brainExtractionInfantT2"
        elif modality == "t1threetissue":
            weights_prefix = "brainExtractionBrainWeb20"; is_standard_network = True
        elif modality == "t1hemi":
            weights_prefix = "brainExtractionT1Hemi"; is_standard_network = True
        elif modality == "t1lobes":
            weights_prefix = "brainExtractionT1Lobes"; is_standard_network = True
        else:
            raise ValueError("Unknown modality type.")

        if verbose:
            print("Brain extraction:  retrieving model weights.")
        weights_file_name = get_pretrained_network(f"{weights_prefix}_pytorch", target_file_name=f"{weights_prefix}_pytorch.pt")

        if verbose:
            print("Brain extraction:  retrieving template.")

        if modality == "t1threetissue":
            reorient_template = ants.image_read(get_antstorch_data("nki"))
        elif modality in ("t1hemi", "t1lobes"):
            reorient_template = ants.image_read(get_antstorch_data("hcpyaT1Template"))
            reorient_template_mask = ants.image_read(get_antstorch_data("hcpyaTemplateBrainMask"))
            reorient_template = reorient_template * reorient_template_mask
            reorient_template = ants.resample_image(reorient_template, (1, 1, 1), use_voxels=False, interp_type=0)
            reorient_template = ants.pad_or_crop_image_to_size(reorient_template, (160, 192, 160))
            xfrm_tmp = ants.create_ants_transform(transform_type="Euler3DTransform",
                                center=np.asarray(ants.get_center_of_mass(reorient_template)), translation=(0, 0, -10))
            reorient_template = xfrm_tmp.apply_to_image(reorient_template)
        else:
            reorient_template = ants.image_read(get_antstorch_data("S_template3"))
            if is_standard_network and (modality != "t1.v1" and modality != "mra"):
                ants.set_spacing(reorient_template, (1.5, 1.5, 1.5))
        resampled_image_size = reorient_template.shape

        # ---------------------
        # Build model EXACTLY matching weights
        # ---------------------
        sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
        out_channels, filters, num_layers, mode, sd = _infer_unet3d_from_state_dict(sd)

        # For multi-class modalities, override to fixed label counts like antspynet
        if modality in ("t1threetissue", "t1hemi", "t1lobes"):
            mode = "classification"
            if modality == "t1threetissue":
                out_channels = 4  # background + 3 classes
            elif modality == "t1hemi":
                out_channels = 3  # background + L/R
            elif modality == "t1lobes":
                out_channels = 6  # background + 5
        elif is_standard_network:
            mode = "sigmoid"; out_channels = 1
        else:
            mode = "classification"; out_channels = 2

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=out_channels,
            number_of_layers=num_layers,
            number_of_filters=filters,
            number_of_filters_at_base_layer=filters[0],
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            dropout_rate=0.0,
            pad_crop="center",
            mode=mode)

        ret = model.load_state_dict(sd, strict=False)

        # PyTorch version–safe access
        missing = getattr(ret, "missing_keys", None)
        unexpected = getattr(ret, "unexpected_keys", None)
        if missing is None or unexpected is None:
            # older tuple-style return
            missing, unexpected = ret

        if missing or unexpected:
            print("[weights check] missing:", len(missing), "unexpected:", len(unexpected))
            print("  e.g. missing[:8]   =", missing[:8])
            print("  e.g. unexpected[:8]=", unexpected[:8])

        model.eval()

        if verbose:
            print("Brain extraction:  normalizing image to the template.")

        # ---------------------
        # CoM alignment (Euler3D)
        # ---------------------
        center_of_mass_template = ants.get_center_of_mass(reorient_template)
        center_of_mass_image = ants.get_center_of_mass(input_images[0])
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                                          center=np.asarray(center_of_mass_template),
                                          translation=translation)

        # ---------------------
        # Assemble batch in template space with antspynet normalization rules
        # ---------------------
        batchX = np.zeros((1, *resampled_image_size, channel_size), dtype=np.float32)

        for i in range(len(input_images)):
            warped_image = ants.apply_ants_transform_to_image(xfrm, input_images[i], reorient_template)
            if is_standard_network and modality != "t1.v1":
                batchX[0, :, :, :, i] = ants.iMath(warped_image, "Normalize").numpy().astype(np.float32)
            else:
                warped_array = warped_image.numpy().astype(np.float32)
                mu = warped_array.mean()
                sigma = warped_array.std()
                if sigma == 0.0:
                    batchX[0, :, :, :, i] = 0.0
                else:
                    batchX[0, :, :, :, i] = (warped_array - mu) / sigma

        if verbose:
            print("Brain extraction:  prediction and decoding.")

        with torch.no_grad():
            x = torch.from_numpy(batchX).permute(0, 4, 1, 2, 3)  # [1,C,D,H,W]
            y = model(x).squeeze(0).cpu().numpy()  # [C_out,D,H,W]

        # Convert to probability images in template space
        prob_images_template = []
        if mode == "sigmoid":
            prob_images_template.append(_to_ants_like(y[0], reorient_template))
        else:
            for c in range(out_channels):
                prob_images_template.append(_to_ants_like(y[c], reorient_template))

        if verbose:
            print("Brain extraction:  renormalize probability mask to native space.")

        xfrm_inv = xfrm.invert()

        if modality in ("t1threetissue", "t1hemi", "t1lobes"):
            probability_images_warped = [xfrm_inv.apply_to_image(pi, input_images[0]) for pi in prob_images_template]
            image_matrix = ants.image_list_to_matrix(probability_images_warped, input_images[0] * 0 + 1)
            segmentation_matrix = np.argmax(image_matrix, axis=0)
            segmentation_image = ants.matrix_to_images(
                np.expand_dims(segmentation_matrix, axis=0), input_images[0] * 0 + 1
            )[0]
            return {"segmentation_image": segmentation_image,
                    "probability_images": probability_images_warped}
        else:
            # For binary/sigmoid case, last channel corresponds to brain
            prob_img_template = prob_images_template[-1]
            probability_image = xfrm_inv.apply_to_image(prob_img_template, input_images[0])
            return probability_image

    # ---------------------
    # NoBrainer branch mirrors antspynet
    # ---------------------
    else:
        if verbose:
            print("NoBrainer:  generating network.")
        # For NoBrainer, keep original behavior via Keras model; if you want Torch version,
        # wire a separate no-brainer Torch model here.
        from antspynet.architectures import create_nobrainer_unet_model_3d  # type: ignore
        model = create_nobrainer_unet_model_3d((None, None, None, 1))
        weights_file_name = get_pretrained_network("brainExtractionNoBrainer")
        model.load_weights(weights_file_name)

        if verbose:
            print("NoBrainer:  preprocessing (intensity truncation and resampling).")
        image_array = image.numpy()
        nz = image_array[np.where(image_array != 0)]
        image_robust_range = np.quantile(nz, (0.02, 0.98)) if nz.size > 0 else (0.0, 1.0)
        threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]

        thresholded_mask = ants.threshold_image(image, -10000, threshold_value, 0, 1)
        thresholded_image = image * thresholded_mask

        image_resampled = ants.resample_image(thresholded_image, (256, 256, 256), use_voxels=True)
        image_array = np.expand_dims(image_resampled.numpy(), axis=0)
        image_array = np.expand_dims(image_array, axis=-1)

        if verbose:
            print("NoBrainer:  predicting mask.")
        brain_mask_array = np.squeeze(model.predict(image_array, verbose=verbose))
        brain_mask_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(brain_mask_array))
        brain_mask_image = ants.resample_image(brain_mask_resampled, image.shape, use_voxels=True, interp_type=1)

        spacing = ants.get_spacing(image)
        spacing_product = spacing[0] * spacing[1] * spacing[2]
        minimum_brain_volume = round(649933.7 / spacing_product)
        brain_mask_labeled = ants.label_clusters(brain_mask_image, minimum_brain_volume)
        return brain_mask_labeled
