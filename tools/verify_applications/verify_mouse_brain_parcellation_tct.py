"""
Verify antstorch.mouse_brain_parcellation(which_parcellation="tct") on
real data.

Same setup as verify_mouse_brain_parcellation_nick.py. This is the
variant whose channel-count bug (registered as 7 instead of 8) was found
and fixed in tasks_registry.py during weight conversion earlier this
session -- a real end-to-end run here is the strongest confirmation that
the fix was correct.

Requires converted weights: mouseT2wBrainExtraction3D_pytorch (for the
internal mask step) and mouseT2wBrainParcellation3DTct_pytorch. Both
already delivered to ~/.antstorch/ this session, so this one should work
out of the box.
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

    output = antstorch.mouse_brain_parcellation(mouse_t2_n4, which_parcellation="tct", verbose=True)
    summarize(output)
    return output


if __name__ == "__main__":
    main()
