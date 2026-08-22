"""
Verify antstorch.lung_extraction(modality="ct") on real data.

Uses ANTsTorch's own bundled ctLungTemplate as input (a real CT lung
image); the function reads luna16LungPriors itself internally.

Requires converted weights: lungCtWithPriorsSegmentationWeights_pytorch.
Never converted in this session -- expect a "no cached weights" error
from get_pretrained_network until that's done.
"""
import ants
import antstorch
import tensorflow as tf


from _common import summarize


def main():
    ct_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934234", force_download=True)
    image = ants.image_read(ct_file)

    output = antstorch.lung_extraction(image, modality="ct", verbose=True)
    ants.image_write(image, "lung_extraction_ct.nii.gz")
    ants.image_write(output['segmentation_image'], "lung_extraction_probability_ct.nii.gz")
    summarize(output)
    return output


if __name__ == "__main__":
    main()
