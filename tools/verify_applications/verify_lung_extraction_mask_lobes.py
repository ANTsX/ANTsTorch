"""
Verify antstorch.lung_extraction(modality="maskLobes") on real data.

Same base architecture/input as protonLobes but without the auxiliary
multihead (strict=True load of the base U-Net only), and the input is
binarized rather than intensity-normalized before warping.

Requires converted weights: maskLobes_pytorch. Never converted in this
session -- expect a "no cached weights" error from get_pretrained_network
until that's done.
"""
import ants
import antstorch

from _common import summarize


def main():
    image = ants.image_read(antstorch.get_antstorch_data("protonLungTemplate"))
    image = ants.threshold_image(image, 0.5, 10.0, 1, 0)

    output = antstorch.lung_extraction(image, modality="maskLobes", verbose=True)
    ants.image_write(image, "lung_extraction_mask_lobes.nii.gz")
    ants.image_write(output['segmentation_image'], "lung_extraction_probability_mask_lobes.nii.gz")
    summarize(output)
    return output


if __name__ == "__main__":
    main()
