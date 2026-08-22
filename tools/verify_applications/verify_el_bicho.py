"""
Verify antstorch.el_bicho (functional lung segmentation) on real data.

No bundled hyperpolarized-gas ventilation MRI sample exists in
get_antstorch_data, so this uses the bundled ctLungTemplate as a
STRUCTURAL stand-in for the ventilation image, with a mask derived from
it via ants.get_mask() (el_bicho requires ventilation_image.shape ==
mask.shape). Real 3-D volume, correct dtype -- not the intended
acquisition modality.

Requires converted weights: elBicho_pytorch. Never converted in this
session -- expect a "no cached weights" error from get_pretrained_network
until that's done.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    
    # Process the proton image to get the lung mask
    proton_file = tf.keras.utils.get_file(fname="protonLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934228", force_download=True)
    proton = ants.image_read(proton_file)
    lung_ex = antstorch.lung_extraction(proton, modality="proton", verbose=True)
    mask = ants.threshold_image(lung_ex['segmentation_image'], 0, 0, 0, 1 )

    # Structural stand-in only -- see module docstring.
    ventilation_file = tf.keras.utils.get_file(fname="ventilationLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934231", force_download=True)
    ventilation_image = ants.image_read(ventilation_file)

    output = antstorch.el_bicho(ventilation_image, mask, verbose=True)
    ants.image_write(ventilation_image, "el_bicho_ventilation.nii.gz")
    ants.image_write(output['segmentation_image'], "el_bicho_segmentation.nii.gz")
    summarize(output)
    return output


if __name__ == "__main__":
    main()
