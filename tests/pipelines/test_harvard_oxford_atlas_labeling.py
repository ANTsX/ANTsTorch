"""Integration test verifying all 32 Harvard-Oxford Atlas labels are predicted."""
import ants
import antstorch
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_data, pytest.mark.slow]


def test_harvard_oxford_atlas_labeling_predicts_all_32_labels():
    tpl = ants.image_read(antstorch.get_antstorch_data("hcpyaT1Template"))
    bseg = ants.image_read(
        antstorch.get_antstorch_data("hcpyaTemplateBrainSegmentation")
    )
    bmask = ants.threshold_image(bseg, 1, 100, 1, 0)
    t1_res = ants.resample_image(
        tpl * bmask, (1, 1, 1), use_voxels=False, interp_type=0
    )
    t1_cropped = ants.pad_or_crop_image_to_size(t1_res, (160, 176, 160))

    res = antstorch.harvard_oxford_atlas_labeling(
        t1_cropped, do_preprocessing=False, verbose=False
    )

    seg = res["segmentation_image"]
    assert len(res["probability_images"]) == 33

    present = set(np.unique(seg.numpy()).astype(int)) - {0}
    expected = set(range(1, 33))
    missing = sorted(expected - present)
    assert missing == [], f"Missing HOA labels: {missing}"
    # Explicitly assert bilateral Nucleus Accumbens (labels 7 and 8)
    assert int((seg.numpy() == 7).sum()) > 0
    assert int((seg.numpy() == 8).sum()) > 0
