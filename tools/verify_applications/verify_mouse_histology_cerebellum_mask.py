"""
Verify antstorch.mouse_histology_cerebellum_mask on real data.

Same structural stand-in as verify_mouse_histology_brain_mask.py (a
middle slice of DevCCF_P04_STPT_50um). view="sagittal" (the default)
selects allen_cerebellum_sagittal_mask_weights_pytorch; pass
view="coronal" to exercise the other weights file instead.

Requires converted weights: allen_cerebellum_sagittal_mask_weights_pytorch
(or allen_cerebellum_coronal_mask_weights_pytorch for view="coronal").
Neither .h5 was located in ~/.keras/ANTsXNet/ (see the project's
gap-analysis doc) -- expect a "no cached weights" error from
get_pretrained_network until one is found/converted.
"""
import ants
import antstorch

from _common import middle_slice, summarize


def main():
    # Structural stand-in only -- see module docstring.
    volume = ants.image_read(antstorch.get_antstorch_data("DevCCF_P04_STPT_50um"))
    image = middle_slice(volume, axis=2)

    mask = antstorch.mouse_histology_cerebellum_mask(image, view="sagittal", verbose=True)
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
