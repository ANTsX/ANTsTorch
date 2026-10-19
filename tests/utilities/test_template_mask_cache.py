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
