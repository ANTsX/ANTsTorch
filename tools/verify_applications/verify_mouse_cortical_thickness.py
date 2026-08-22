"""
Verify antstorch.mouse_cortical_thickness on real data.

This is a thin wrapper: mouse_brain_parcellation(which_parcellation="nick")
followed by ants.kelly_kapowski(). Uses the bundled
DevCCF_P56_MRI-T2_50um image, same as the nick-parcellation verify
script, plus real KellyKapowski cortical-thickness estimation (no
learned model of its own, but a real, occasionally-slow ANTs numerical
optimization -- expect this script to take noticeably longer than the
others).

Requires the same weights as verify_mouse_brain_parcellation_nick.py
(mouseT2wBrainExtraction3D_pytorch, mouseT2wBrainParcellation3DNick_pytorch)
-- both already delivered to ~/.antstorch/ this session.
"""
import ants
import antstorch
import tensorflow as tf

from _common import summarize


def main():
    mouse_t2_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309", force_download=True)
    mouse_t2 = ants.image_read(mouse_t2_file)

    output = antstorch.mouse_cortical_thickness(mouse_t2, verbose=True)
    summarize(output)
    return output


if __name__ == "__main__":
    main()
