"""
Verify antstorch.hypothalamus_segmentation on real data.

Uses the same real T1 image as the other verify scripts (figshare id
40251796).

Requires converted weights: hypothalamus_pytorch. Not yet delivered to your
machine (no matching converter has been run -- see the project's
gap-analysis doc), so this will fail with a "no cached weights" error from
get_pretrained_network until that's done.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)

    result = antstorch.hypothalamus_segmentation(t1, verbose=True)
    ants.image_write(t1, "hypothalamus_t1.nii.gz")
    ants.image_write(result["segmentation_image"], "hypothalamus_segmentation.nii.gz")
    summarize(result)
    return result


if __name__ == "__main__":
    main()
