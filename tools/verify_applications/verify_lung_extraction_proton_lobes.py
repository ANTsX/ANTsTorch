"""
Verify antstorch.lung_extraction(modality="protonLobes") on real data.

Uses the bundled protonLungTemplate as input; the function reads
protonLobePriors itself internally. This is the modality that wraps the
base U-Net in create_multihead_unet_model_3d (1 auxiliary sigmoid head
for the whole-lung mask) -- a good check that the multihead
warmup/strict=False load path works with real weights, not just random
init (see tests/test_lung_extraction_architectures.py for the
architecture-only version of this check).

Requires converted weights: protonLobes_pytorch. This one was never
converted in this session (channel count depends on protonLobePriors,
which wasn't attempted) -- expect a "no cached weights" error from
get_pretrained_network until that's done.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    proton_file = tf.keras.utils.get_file(fname="protonLung.nii.gz", origin="https://ndownloader.figshare.com/files/42934228", force_download=True)
    proton = ants.image_read(proton_file)

    # Proton (lobes)
    lung_ex = antstorch.lung_extraction(proton, modality="protonLobes", verbose=True)
    ants.image_write(proton, "lung_extraction_proton_lobes.nii.gz")
    ants.image_write(lung_ex['segmentation_image'], "lung_extraction_probability_proton_lobes.nii.gz")
    summarize(lung_ex)
    return lung_ex


if __name__ == "__main__":
    main()
