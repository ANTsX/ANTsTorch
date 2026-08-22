"""
Verify antstorch.lung_airway_segmentation on real data.

Uses the bundled ctLungTemplate directly. lung_mask=None (the default),
so this also exercises lung_extraction(modality="ct") internally
(followed by ants.iMath_MD dilation) to derive the airway-search mask
before patch extraction.

Requires converted weights: pulmonaryAirwayWeights_pytorch, AND
(transitively) lungCtWithPriorsSegmentationWeights_pytorch. Neither was
converted in this session -- expect a "no cached weights" error from
get_pretrained_network until both are done.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    ct_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934234", force_download=True)
    ct = ants.image_read(ct_file)

    probability_image = antstorch.lung_airway_segmentation(ct, verbose=True)
    ants.image_write(ct, "lung_airway_segmentation_ct.nii.gz")
    ants.image_write(probability_image, "lung_airway_segmentation_probability.nii.gz")
    summarize(probability_image)
    return probability_image


if __name__ == "__main__":
    main()
