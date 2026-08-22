"""
Verify antstorch.mouse_histology_brain_mask on real data.

No bundled mouse histology sample exists in get_antstorch_data, so this
uses a middle 2-D slice of the bundled DevCCF_P04_STPT_50um volume (real
serial-two-photon-tomography data) as a STRUCTURAL stand-in -- correct
2-D single-channel shape, but not the intended histology acquisition.

Requires converted weights: allen_brain_mask_weights_pytorch -- already
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

    mask = antstorch.mouse_histology_brain_mask(image, verbose=True)
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
