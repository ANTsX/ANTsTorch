"""
Verify antstorch.lung_pulmonary_artery_segmentation on real data.

Uses the bundled ctLungTemplate directly. lung_mask=None (the default),
so this also exercises lung_extraction(modality="ct") internally to
derive the lung mask before patch extraction -- a real dependency chain,
not just the artery model in isolation. Requires the image to be >160
voxels per dimension (ctLungTemplate is).

Requires converted weights: pulmonaryArteryWeights_pytorch, AND
(transitively, via the internal lung_extraction(modality="ct") call)
lungCtWithPriorsSegmentationWeights_pytorch. Neither was converted in
this session -- expect a "no cached weights" error from
get_pretrained_network until both are done.
"""
import ants
import antstorch

from _common import summarize


def main():
    ct = ants.image_read(antstorch.get_antstorch_data("ctLungTemplate"))
    ct = ants.iMath_pad(ct, 24)

    probability_image = antstorch.lung_pulmonary_artery_segmentation(ct, verbose=True)
    summarize(probability_image)
    return probability_image


if __name__ == "__main__":
    main()
