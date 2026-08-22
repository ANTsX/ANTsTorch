"""
Verify antstorch.mouse_brain_extraction(modality="t2") on real data.

Uses ANTsTorch's own bundled DevCCF_P56_MRI-T2_50um as the input image --
a real T2-weighted mouse brain MRI, distinct from the bsplineT2MouseTemplate
the function warps it to internally, so this exercises a genuine (if
easy) registration case in addition to the model itself.

Requires converted weights: mouseT2wBrainExtraction3D_pytorch -- already
delivered to ~/.antstorch/ this session, so this one should work out of
the box.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    image_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309", force_download=True)
    image = ants.image_read(image_file)

    mask = antstorch.mouse_brain_extraction(image, modality="t2", verbose=True)
    ants.image_write(image, "mouse_brain_extraction_t2.nii.gz") 
    ants.image_write(mask, "mouse_brain_extraction_probability_t2.nii.gz")
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
