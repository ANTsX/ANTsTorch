"""
Verify antstorch.tid_neural_image_assessment on real data.

IMPORTANT -- unlike every other script in this folder, this one does NOT
exercise real trained weights: no tidsQualityAssessment/koniqMS* weights
exist to convert, because no explicit ResNet architecture-constructor for
those models exists anywhere in ANTsPyNet to convert them against (see the
module-level confidence note at the top of
antstorch/utilities/quality_assessment.py, and the project's gap-analysis
doc).

Instead this smoke-tests the documented work-around: passing an
already-built torch.nn.Module directly via `which_model`, which is the
recommended way to sidestep the architecture uncertainty. The model built
here is randomly initialized (untrained), so the resulting MOS numbers are
MEANINGLESS as quality scores -- this only confirms the real code path
(patch/global extraction, batching, prediction, reconstruction) runs
end-to-end on a real image without crashing. patch_size="global" is used to
keep this fast (no patch extraction/reconstruction).
"""
import ants
import antstorch
from antstorch.utilities.quality_assessment import _default_qa_resnet_model

from _common import get_t1_flair_pair, middle_slice, summarize


def main():
    t1_path, _ = get_t1_flair_pair()
    t1 = ants.image_read(t1_path)
    image_2d = middle_slice(t1, axis=2)

    # Untrained stand-in model -- see module docstring above.
    model = _default_qa_resnet_model(number_of_outputs=2)

    result = antstorch.tid_neural_image_assessment(
        image_2d, which_model=model, patch_size="global", verbose=True
    )
    summarize(result)
    return result


if __name__ == "__main__":
    main()
