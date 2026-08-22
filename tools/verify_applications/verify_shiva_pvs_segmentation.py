"""
Verify antstorch.shiva_pvs_segmentation (perivascular spaces) on real data.

Argument order is (t1, flair=None); passing both uses the T1+FLAIR
ensemble (pvs_shiva_t1_flair_{0..4}) rather than the T1-only one
(pvs_shiva_t1_{0..5}). which_model="all" (the default) averages the full
5-model ensemble -- pass a single integer (e.g. which_model=0) for a much
faster one-model smoke test.

Requires converted weights: pvs_shiva_t1_flair_{0..4}_pytorch. None of
these are yet delivered to your machine (21 shiva files total, >3 GB --
see the project's gap-analysis doc) -- run tools/convert_wmh_bespoke.py
locally first.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, flair_path = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    flair = ants.image_read(flair_path)

    pvs = antstorch.shiva_pvs_segmentation(t1, flair, which_model=0, verbose=True)
    ants.image_write(t1, "shiva_pvs_segmentation_t1.nii.gz")
    ants.image_write(flair, "shiva_pvs_segmentation_flair.nii.gz")
    ants.image_write(pvs, "shiva_pvs_segmentation_probability.nii.gz")
    summarize(pvs)
    return pvs


if __name__ == "__main__":
    main()
