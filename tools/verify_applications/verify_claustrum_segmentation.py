"""
Verify antstorch.claustrum_segmentation on real data.

Uses the same real T1 image as the other verify scripts (figshare id
40251796). use_ensemble=False picks a single model per view (axial/coronal)
instead of the full 3-model ensemble, for a faster smoke test.

Requires converted weights: claustrum_axial_0_pytorch, claustrum_coronal_0_pytorch
(plus the _1/_2 variants if you set use_ensemble=True). None have been
converted yet (no matching converter has been run for claustrum -- see the
project's gap-analysis doc), so this will fail with a "no cached weights"
error from get_pretrained_network until that's done.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)

    probability_image = antstorch.claustrum_segmentation(t1, use_ensemble=False, verbose=True)
    ants.image_write(t1, "claustrum_t1.nii.gz")
    ants.image_write(probability_image, "claustrum_segmentation_probability.nii.gz")
    summarize(probability_image)
    return probability_image


if __name__ == "__main__":
    main()
