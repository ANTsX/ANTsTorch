"""
Verify antstorch.mouse_histology_super_resolution (DBPN, 2x) on real data.

This function requires a 2-D, 3-component (RGB) input. No bundled color
histology sample exists, so this builds a SYNTHETIC RGB image by
replicating a grayscale middle slice of DevCCF_P04_STPT_50um into 3
identical channels. That exercises the real channel-splitting /
per-channel normalization / DBPN forward pass / regression-match
reconstruction code path with correct tensor shapes, but the result will
look like a flat-gray upsampled image, not genuine color super-resolution.

Requires converted weights: allen_sr_weights_pytorch. No matching .h5 was
ever located in ~/.keras/ANTsXNet/, and no dedicated converter was
written for the DBPN architecture this session (see "Prochaines etapes
possibles" in the project's gap-analysis doc) -- expect a "no cached
weights" error from get_pretrained_network until that's done.
"""
import ants
import antstorch

from _common import middle_slice, to_fake_rgb, summarize


def main():
    # Synthetic RGB stand-in only -- see module docstring.
    volume = ants.image_read(antstorch.get_antstorch_data("DevCCF_P04_STPT_50um"))
    gray = middle_slice(volume, axis=2)
    image = to_fake_rgb(gray)

    output = antstorch.mouse_histology_super_resolution(image, verbose=True)
    summarize(output)
    return output


if __name__ == "__main__":
    main()
