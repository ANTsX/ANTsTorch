"""
Verify antstorch.lung_extraction(modality="proton") on real data.

Uses ANTsTorch's own bundled protonLungTemplate as the input image (a
real proton-density lung MRI). This is the same image the function
internally aligns *to*, so the registration step is close to identity --
a smoke test of the full pipeline (real preprocessing, real model, real
weights, real reconstruction), not a held-out validation case.

Requires converted weights: protonLungMri_pytorch -- already delivered to
~/.antstorch/ this session, so this one should work out of the box.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    proton_file = tf.keras.utils.get_file(fname="protonLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934228", force_download=True)
    proton = ants.image_read(proton_file)

    output = antstorch.lung_extraction(proton, modality="proton", verbose=True)
    ants.image_write(proton, "lung_extraction_proton.nii.gz")
    ants.image_write(output['segmentation_image'], "lung_extraction_probability_proton.nii.gz")  
    summarize(output)
    return output


if __name__ == "__main__":
    main()
