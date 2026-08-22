"""
Verify antstorch.sysu_media_wmh_segmentation on real data.

Adapted from the snippet Nick pulled from his tutorial notes (originally
downloaded via tf.keras.utils.get_file -- replaced here with a plain
urllib download so this doesn't need tensorflow, which antstorch has no
dependency on). Same source images (figshare ids 40251796 / 40251793).

Requires converted weights: sysuMediaWmhFlairOnlyModel{0,1,2}_pytorch
(use_ensemble=True averages all 3). These are NOT yet delivered to your
machine -- run tools/convert_wmh_bespoke.py locally first (see the
project's gap-analysis doc), or this will fail with a "no cached weights"
error from get_pretrained_network.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, flair_path = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)

    wmh = antstorch.sysu_media_wmh_segmentation(flair, t1, verbose=True)
    summarize(wmh)
    return wmh


if __name__ == "__main__":
    main()
