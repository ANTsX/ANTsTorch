"""
Verify antstorch.mri_super_resolution on real data.

Uses the same real T1 image as the other verify scripts (figshare id
40251796). Runs with the function's defaults: expansion_factor=(1, 1, 2),
feature="vgg" -- i.e. the "sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl"
SIQ DBPN model.

Requires converted weights: sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl_pytorch.
Not yet converted on your machine as of this writing -- run
tools/convert_mri_super_resolution_bespoke.py first (see the project's
gap-analysis doc for the exact command), so this will fail with a "no
cached weights" error from get_pretrained_network until that's done.

⚠️ This is the newest and least-verified application in this folder: the
create_siq_dbpn_super_resolution_model_3d architecture (UpSampling3D+Conv3D,
not ConvTranspose) and tools/convert_mri_super_resolution_bespoke.py (which
reads convolution_kernel_size/number_of_base_filters/etc. directly out of
each real .h5's embedded model_config JSON) have only been checked against
a synthetic round-trip .h5 -- never against one of the real
"sig_smallshort_train_*" files. A first real run here is exactly the test
that's still missing. If it fails, the first things to inspect are: (1)
the converter's printed architecture_kwargs vs. what you'd expect for a
"smallshort" model, and (2) whether load_state_dict(strict=True) raised
during conversion (it would have aborted the .pt write already, so this
script wouldn't even find the weights).
"""
import ants
import antstorch

from _common import get_t1_flair_pair, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)

    image_sr = antstorch.mri_super_resolution(t1, verbose=True)
    ants.image_write(t1, "mri_super_resolution_input.nii.gz")
    ants.image_write(image_sr, "mri_super_resolution_output.nii.gz")
    summarize(image_sr)
    return image_sr


if __name__ == "__main__":
    main()
