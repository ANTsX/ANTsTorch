"""
Verify antstorch.wmh_segmentation (patch-based, sysu_media 3-D
architecture) on real data.

Note t1 is a required positional argument here (unlike the other WMH
functions where one modality is optional). do_preprocessing=True runs
deep_atropos internally to derive the white-matter mask, so this exercises
a large chunk of the ported pipeline beyond just the WMH model itself.

Requires converted weights: antsxnetWmhOr_pytorch (use_combined_model=True,
the default) or antsxnetWmh_pytorch. Neither is yet delivered to your
machine -- run tools/convert_wmh_bespoke.py locally first.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, flair_path = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    t1 = ants.iMath_pad(t1, 20)
    flair = ants.image_read(flair_path)
    flair = ants.iMath_pad(flair, 20)

    probability_image = antstorch.wmh_segmentation(flair, t1, verbose=True)
    summarize(probability_image)
    return probability_image


if __name__ == "__main__":
    main()
