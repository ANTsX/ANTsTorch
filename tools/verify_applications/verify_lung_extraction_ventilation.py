"""
Verify antstorch.lung_extraction(modality="ventilation") on real data.

No bundled hyperpolarized-gas ventilation MRI sample exists in
get_antstorch_data, so this uses the bundled ctLungTemplate as a
STRUCTURAL stand-in (real 3-D lung volume, correct dtype -- but not the
intended acquisition modality). Good enough to confirm the 2-D slice
extraction / sigmoid U-Net / reconstruction pipeline actually runs; not a
clinically meaningful result.

Requires converted weights: wholeLungMaskFromVentilation_pytorch --
already delivered to ~/.antstorch/ this session, so this one should work
out of the box.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    ventilation_file = tf.keras.utils.get_file(fname="ventilationLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934231", force_download=True)
    ventilation = ants.image_read(ventilation_file)

    output = antstorch.lung_extraction(ventilation, modality="ventilation", verbose=True)
    ants.image_write(ventilation, "lung_extraction_ventilation.nii.gz")
    ants.image_write(output['segmentation_image'], "lung_extraction_probability_ventilation.nii.gz") 
    summarize(output)
    return output


if __name__ == "__main__":
    main()
