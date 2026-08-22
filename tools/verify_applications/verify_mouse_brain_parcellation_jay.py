"""
Verify antstorch.mouse_brain_parcellation(which_parcellation="jay") on
real data.

Uses the bundled DevCCF_P04_STPT_50um image (real serial-two-photon-
tomography data -- the correct modality for this parcellation, unlike the
STPT stand-ins used elsewhere in this folder). Note: mask=None makes the
function call mouse_brain_extraction(image, modality="t2", ...)
internally regardless of which_parcellation -- that's the ported
behavior as written (mirrors antspynet), not something this script works
around.

Requires converted weights: mouseT2wBrainExtraction3D_pytorch (for the
internal mask step, already delivered) and
mouseSTPTBrainParcellation3DJay_pytorch (NOT converted in this session --
this .h5 was never located in ~/.keras/ANTsXNet/, see the project's
gap-analysis doc). Expect a "no cached weights" error at the parcellation
step until that weights file is found/converted.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    mouse_t2_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309", force_download=True)
    mouse_t2 = ants.image_read(mouse_t2_file)
    mouse_t2_n4 = ants.n4_bias_field_correction(mouse_t2, 
                                                rescale_intensities=True,
                                                shrink_factor=2, 
                                                convergence={'iters': [50, 50, 50, 50], 'tol': 0.0}, 
                                                spline_param=20, verbose=True)
    output = antstorch.mouse_brain_parcellation(mouse_t2_n4, which_parcellation="jay", verbose=True)
    summarize(output)
    return output


if __name__ == "__main__":
    main()
