"""
Verify antstorch.shiva_wmh_segmentation on real data.

Argument order is (flair, t1=None); passing both uses the T1+FLAIR
ensemble (wmh_shiva_t1_flair_{0..4}). which_model=0 picks a single model
instead of the full "all" ensemble, for a faster smoke test.

Requires converted weights: wmh_shiva_t1_flair_{0..4}_pytorch. None of
these are yet delivered to your machine -- run tools/convert_wmh_bespoke.py
locally first.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, flair_path = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)

    wmh = antstorch.shiva_wmh_segmentation(flair, t1, which_model=0, verbose=True)
    ants.image_write(t1, "shiva_wmh_segmentation_t1.nii.gz")
    ants.image_write(flair, "shiva_wmh_segmentation_flair.nii.gz")
    ants.image_write(wmh, "shiva_wmh_segmentation_probability.nii.gz")
    summarize(wmh)
    return wmh


if __name__ == "__main__":
    main()
