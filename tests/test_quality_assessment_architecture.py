"""
Tests for antstorch.utilities.quality_assessment -- kept separate from
test_hippocampus_hypothalamus_claustrum_architectures.py on purpose, because
this module has a materially different (much lower) confidence status: no
architecture-constructor function for the real tidsQualityAssessment/koniqMS*
ResNet(s) exists anywhere in ANTsPyNet (they are only ever loaded whole from a
saved Keras model). See the module-level docstring in
antstorch/utilities/quality_assessment.py for the full explanation.

Two things ARE independently testable without a real trained model, and are
covered here:

    1. random_mask() -- pure numpy subsampling logic, no neural network and
       no ants image I/O involved (it operates the same way on a plain numpy
       array as it does on an ANTsImage, since both support elementwise `==`,
       `*`, and boolean-indexed assignment the same way). Includes a
       regression test for the off-by-one bug found and fixed during porting
       (random.randint(0, xsz) -> random.randint(0, xsz - 1)).

    2. _default_qa_resnet_model() -- confirms the placeholder wiring around
       create_resnet_model_2d(mode="regression") actually builds and runs a
       forward pass without crashing, and that mode="regression" really does
       produce a plain nn.Linear head (no softmax) as tid_neural_image_assessment
       expects. This does NOT validate the placeholder against any known-correct
       output -- there is nothing to validate against yet (see caveat above).
       architecture_kwargs is used to shrink the ResNet (fewer layers/blocks)
       purely so the test stays fast; it does not change what is being checked.
"""
import unittest

import numpy as np
import torch

from antstorch.utilities.quality_assessment import random_mask, _default_qa_resnet_model

DEVICE = torch.device("cpu")


class Test_random_mask(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_no_out_of_bounds_index_regression(self):
        """Regression test for the ANTsPyNet off-by-one bug: with the
        original `random.randint(0, xsz)` (inclusive on both ends), the
        out-of-bounds index `xsz` could be drawn against a length-xsz
        `binvec`, raising an IndexError. Run enough trials against a small
        mask (xsz = 9) that the bug, if it recurred, would very likely
        trigger."""
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:3, :3] = 1  # xsz = 9
        for _ in range(500):
            out = random_mask(mask.copy(), 5)
            self.assertEqual(out.shape, mask.shape)

    def test_n_greater_than_available_returns_input_unchanged(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[0, :3] = 1  # xsz = 3
        out = random_mask(mask.copy(), 10)
        np.testing.assert_array_equal(out, mask)

    def test_subsamples_only_from_original_support(self):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:5, :5] = 1  # xsz = 25
        out = random_mask(mask.copy(), 5)
        # Every nonzero voxel in the output must have been nonzero in the input.
        self.assertTrue(np.all(mask[out == 1] == 1))
        # Sampling is with replacement, so at most n nonzero voxels come out.
        self.assertLessEqual(int((out == 1).sum()), 5)


class Test_default_qa_resnet_model(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_finite(self):
        model = _default_qa_resnet_model(
            number_of_outputs=2,
            architecture_kwargs=dict(layers=(1, 2), residual_block_schedule=(1, 1),
                                      lowest_resolution=8),
        ).to(DEVICE)

        x = torch.randn(4, 3, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (4, 2))
        self.assertTrue(torch.isfinite(y).all())

    def test_regression_mode_has_no_softmax_head(self):
        model = _default_qa_resnet_model(
            architecture_kwargs=dict(layers=(1,), residual_block_schedule=(1,), lowest_resolution=8),
        )
        self.assertIsInstance(model.dense, torch.nn.Linear)


if __name__ == "__main__":
    unittest.main()
