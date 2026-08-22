"""
Verify antstorch.mouse_brain_extraction(modality="ex5coronal") on real data.

This modality targets E13.5/E15.5 mouse embryonic histology, for which no
bundled sample exists in get_antstorch_data. This uses a middle 2-D slice
of the bundled DevCCF_P04_STPT_50um volume (real postnatal-day-4
serial-two-photon-tomography data) as a STRUCTURAL stand-in -- correct
2-D single-channel shape, but not the intended embryonic histology
acquisition.

Requires converted weights: ex5_coronal_weights_pytorch -- already
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

    mask = antstorch.mouse_brain_extraction(image, modality="ex5coronal", verbose=True)
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
