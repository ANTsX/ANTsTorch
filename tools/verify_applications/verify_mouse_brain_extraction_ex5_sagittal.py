"""
Verify antstorch.mouse_brain_extraction(modality="ex5sagittal") on real
data.

Same structural stand-in as verify_mouse_brain_extraction_ex5_coronal.py
(a middle slice of DevCCF_P04_STPT_50um) -- see that script's docstring
for the caveat. Only the weights file differs between the two modalities;
the architecture is identical.

Requires converted weights: ex5_sagittal_weights_pytorch -- already
delivered to ~/.antstorch/ this session, so this one should work out of
the box.
"""
import ants
import antstorch

from _common import middle_slice, summarize


def main():
    # Structural stand-in only -- see module docstring.
    volume = ants.image_read(antstorch.get_antstorch_data("DevCCF_P04_STPT_50um"))
    image = middle_slice(volume, axis=2)

    mask = antstorch.mouse_brain_extraction(image, modality="ex5sagittal", verbose=True)
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
