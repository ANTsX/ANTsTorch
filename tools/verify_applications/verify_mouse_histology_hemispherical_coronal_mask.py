"""
Verify antstorch.mouse_histology_hemispherical_coronal_mask on real data.

Same structural stand-in as verify_mouse_histology_brain_mask.py (a
middle slice of DevCCF_P04_STPT_50um) -- see that script's docstring for
the caveat. This is the 3-class variant (background/left/right
hemisphere) using create_multihead-free classification over 3 outputs.

Requires converted weights: allen_brain_leftright_coronal_mask_weights_pytorch.
This .h5 was never located in ~/.keras/ANTsXNet/ (see the project's
gap-analysis doc) -- expect a "no cached weights" error from
get_pretrained_network until it's found/converted.
"""
import ants
import antstorch

from _common import middle_slice, summarize


def main():
    # Structural stand-in only -- see module docstring.
    volume = ants.image_read(antstorch.get_antstorch_data("DevCCF_P04_STPT_50um"))
    image = middle_slice(volume, axis=2)

    output = antstorch.mouse_histology_hemispherical_coronal_mask(image, verbose=True)
    summarize(output)
    return output


if __name__ == "__main__":
    main()
