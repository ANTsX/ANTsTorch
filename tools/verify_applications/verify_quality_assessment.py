"""
Verify antstorch.tid_neural_image_assessment on real data.

Updated 2026-08-23: unlike when this script was first written, real weights
CAN now exist for these models -- koniqMS3.h5's embedded model_config JSON
confirmed the ResNet architecture antstorch already assumed (see the
confidence note at the top of antstorch/utilities/quality_assessment.py and
tools/convert_quality_assessment_bespoke.py). This script now tries the
real built-in path first (which_model="koniqMS3", the one file whose
architecture has actually been confirmed against a real model_config), and
falls back to the untrained-placeholder smoke test if the weights haven't
been converted yet on this machine.

If the real path succeeds, the printed MOS/sharpness numbers are genuine
model output (for whatever that's worth pointed at a T1 slice rather than
a natural photograph -- koniqMS3 was trained on photographic MOS data, not
medical images, so don't read anything into the actual score). If it falls
back, the numbers are MEANINGLESS (random weights) and only confirm the
real code path (patch/global extraction, batching, prediction,
reconstruction) runs end-to-end without crashing. patch_size="global" is
used in both cases to keep this fast.
"""
import ants
import antstorch
from antstorch.utilities.quality_assessment import _default_qa_resnet_model

from _common import get_t1_flair_pair, middle_slice, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    image_2d = middle_slice(t1, axis=2)

    try:
        result = antstorch.tid_neural_image_assessment(
            image_2d, which_model="koniqMS3", patch_size="global", verbose=True
        )
        print("Used real converted koniqMS3 weights.")
    except ValueError as e:
        print(f"koniqMS3_pytorch not converted yet ({e}); "
              "falling back to an untrained placeholder model -- see module docstring.")
        model = _default_qa_resnet_model()
        result = antstorch.tid_neural_image_assessment(
            image_2d, which_model=model, patch_size="global", verbose=True
        )

    summarize(result)
    return result


if __name__ == "__main__":
    main()
