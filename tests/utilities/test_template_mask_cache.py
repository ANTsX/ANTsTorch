"""The registration template's brain mask is extracted once, not once per call."""
import ants
import pytest

import antstorch
from antstorch.utilities import preprocess_image as pi


def test_template_mask_is_cached(monkeypatch):
    pi._TEMPLATE_MASK_CACHE.clear()
    template = ants.image_read(ants.get_ants_data("r16")).clone("float")
    moving = ants.image_read(ants.get_ants_data("r64")).clone("float")
    seen = []

    def fake_brain_extraction(image, modality, device=None, verbose=False):
        seen.append(image)
        return image * 0 + 1  # probability image: everything is brain

    monkeypatch.setattr(antstorch.utilities, "brain_extraction", fake_brain_extraction)

    kwargs = dict(
        truncate_intensity=None, brain_extraction_modality="t1",
        template=template, template_transform_type="Rigid",
        do_bias_correction=False, do_denoising=False, verbose=False,
    )
    pi.preprocess_brain_image(moving, **kwargs)
    pi.preprocess_brain_image(moving, **kwargs)

    # 2 calls x 1 extraction of the moving image + 1 extraction of the template
    assert len(seen) == 3
    assert sum(1 for im in seen if im is template) == 1


def test_dkt_and_hoa_do_not_forward_brain_mask_to_t1hemi(monkeypatch):
    import numpy as np
    import torch

    from antstorch.utilities._cache import (
        _DATA_CACHE,
        _MODEL_CACHE,
        clear_utility_caches,
    )
    from antstorch.utilities.desikan_killiany_tourville_labeling import (
        desikan_killiany_tourville_labeling,
    )
    from antstorch.utilities.harvard_oxford_atlas_labeling import (
        harvard_oxford_atlas_labeling,
    )

    t1 = ants.from_numpy(np.ones((12, 12, 12), dtype=np.float32))
    native_mask = ants.from_numpy(np.ones((12, 12, 12), dtype=np.float32))
    template_img = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
    template_space_mask = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))

    class _FakeMultiheadModel:
        def __init__(self, c_out):
            self.c_out = c_out

        def __call__(self, x):
            spatial = tuple(x.shape[2:])
            y_main = torch.zeros((x.shape[0], self.c_out, *spatial), dtype=torch.float32)
            y_aux = torch.zeros((x.shape[0], 1, *spatial), dtype=torch.float32)
            return y_main, y_aux

    def fake_preprocess(image, **kwargs):
        return {
            "preprocessed_image": template_img.clone(),
            "brain_mask": template_space_mask.clone(),
            "template_transforms": {"fwdtransforms": [], "invtransforms": []},
        }

    hemi_calls = []

    def fake_brain_extraction(image, modality, device=None, verbose=False, **kwargs):
        hemi_calls.append({"image": image, "modality": modality, "kwargs": dict(kwargs)})
        ones = image * 0 + 1
        return {
            "segmentation_image": ones.clone(),
            "probability_images": [ones * 0, ones.clone(), ones.clone()],
        }

    monkeypatch.setattr(antstorch.utilities, "preprocess_brain_image", fake_preprocess)
    monkeypatch.setattr(antstorch.utilities, "brain_extraction", fake_brain_extraction)
    monkeypatch.setattr(antstorch.utilities, "get_pretrained_network", lambda name: f"{name}.pt")
    monkeypatch.setattr(
        ants,
        "apply_transforms",
        lambda fixed, moving, transformlist, **kwargs: fixed * 0,
    )

    clear_utility_caches()
    try:
        _DATA_CACHE[("template_image", "hcpyaT1Template")] = template_img
        _MODEL_CACHE[("dkt_outer", "cpu")] = _FakeMultiheadModel(38)
        _MODEL_CACHE[("hoa_subcortical", "cpu")] = _FakeMultiheadModel(23)

        for passed_mask in (None, native_mask):
            desikan_killiany_tourville_labeling(
                t1,
                do_preprocessing=True,
                do_lobar_parcellation=False,
                brain_mask=passed_mask,
                device="cpu",
                verbose=False,
            )
            harvard_oxford_atlas_labeling(
                t1,
                do_preprocessing=True,
                brain_mask=passed_mask,
                device="cpu",
                verbose=False,
            )
    finally:
        clear_utility_caches()

    assert len(hemi_calls) == 4
    for call in hemi_calls:
        assert call["modality"] == "t1hemi"
        assert call["kwargs"] == {}
        assert call["image"] is not template_space_mask
