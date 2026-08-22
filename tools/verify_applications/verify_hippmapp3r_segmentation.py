"""
Verify antstorch.hippmapp3r_segmentation on real data.

Uses the same real T1 image as the other verify scripts (figshare id
40251796) -- only the T1 channel is needed for hippmapp3r_segmentation
(unlike the WMH scripts, which need T1+FLAIR).
number_of_monte_carlo_iterations is reduced to 10 (from the default 30) for
a faster smoke test, matching the same reduction used in
verify_hypermapp3r_segmentation.py.

Requires converted weights: hippMapp3rInitial_pytorch, hippMapp3rRefine_pytorch.
Neither has been converted yet (no matching converter has been run for
hippmapp3r -- see the project's gap-analysis doc), so this will fail with a
"no cached weights" error from get_pretrained_network until that's done.
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)

    segmentation = antstorch.hippmapp3r_segmentation(
        t1, number_of_monte_carlo_iterations=10, verbose=True
    )
    ants.image_write(t1, "hippmapp3r_t1.nii.gz")
    ants.image_write(segmentation, "hippmapp3r_segmentation.nii.gz")
    summarize(segmentation)
    return segmentation


if __name__ == "__main__":
    main()
