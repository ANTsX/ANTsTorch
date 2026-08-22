"""
Verify antstorch.hypermapp3r_segmentation on real data.

Note the argument order is (t1, flair) -- reversed relative to
sysu_media_wmh_segmentation's (flair, t1). Uses the same real T1/FLAIR
pair as the other WMH verify scripts.

Requires converted weights: hyperMapp3r_pytorch -- already delivered to
~/.antstorch/ this session, so this one should work out of the box.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, flair_path = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)

    mask = antstorch.hypermapp3r_segmentation(
        t1, flair, number_of_monte_carlo_iterations=10, verbose=True
    )
    ants.image_write(t1, "hypermapp3r_t1.nii.gz")
    ants.image_write(mask, "hypermapp3r_segmentation.nii.gz")
    summarize(mask)
    return mask


if __name__ == "__main__":
    main()
