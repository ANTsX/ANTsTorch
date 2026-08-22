"""
Verify antstorch.lung_extraction(modality="xray") on real data.

No bundled chest x-ray sample exists in get_antstorch_data, so this uses
a middle 2-D slice of the bundled ctLungTemplate as a STRUCTURAL stand-in
(real grayscale image, correct 2-D shape -- not an actual radiograph).
The function reads xrayLungPriors itself internally. modality="xray" is
the only branch that requires image.dimension == 2 (all others require 3).

Requires converted weights: xrayLungExtraction_pytorch -- already
delivered to ~/.antstorch/ this session, so this one should work out of
the box.
"""
import ants
import antstorch
import tensorflow as tf

from _common import middle_slice, summarize


def main():
    # Structural stand-in only -- see module docstring.
    cxr_file = tf.keras.utils.get_file(fname="cxr.nii.gz", origin="https://ndownloader.figshare.com/files/42934237", force_download=True)
    cxr = ants.image_read(cxr_file)

    output = antstorch.lung_extraction(cxr, modality="xray", verbose=True)
    ants.image_write(cxr, "lung_extraction_xray.nii.gz")
    ants.image_write(output['segmentation_image'], "lung_extraction_probability_xray.nii.gz")
    summarize(output)
    return output


if __name__ == "__main__":
    main()
