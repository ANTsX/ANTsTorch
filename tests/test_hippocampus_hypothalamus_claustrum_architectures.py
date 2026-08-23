"""
Architecture-only tests for the three high-confidence applications ported
2026-08-22 (see the project's gap-analysis doc):

    * create_hippmapp3r_unet_model_3d  (hippmapp3r_segmentation, both stages)
    * create_hypothalamus_unet_model_3d (hypothalamus_segmentation)
    * create_sysu_media_unet_model_2d(anatomy="claustrum") (claustrum_segmentation
                                        -- the class itself was already covered by
                                        test_wmh_architectures.py under anatomy="wmh";
                                        this file adds the anatomy="claustrum" branch,
                                        which uses a different filter schedule and skips
                                        the wmh-only 5x5 first-layer kernel)

quality_assessment is deliberately NOT covered here -- see
test_quality_assessment_architecture.py, which carries its own confidence
caveat (no verified architecture exists for that module).

See tests/_arch_common.py for scope / rationale (architecture-only, no
pretrained weights, no ants image I/O).

hippmapp3r's initial-stage network pools 5 times (do_first_network=True ->
6 encoding layers, stride 1 then 5x stride 2) and its refine-stage network
pools 4 times (do_first_network=False -> 5 encoding layers). Both are
tested at 64**3 -- confirmed during porting to be small enough to avoid the
OOM that hit hypothalamus at its full production resolution (204, 256, 256),
while still exercising every skip/upsample/back64/back32 branch.
"""
import unittest

import torch

from antstorch.architectures import (
    create_hippmapp3r_unet_model_3d,
    create_hypothalamus_unet_model_3d,
    create_sysu_media_unet_model_2d,
)

from _arch_common import DEVICE, assert_sigmoid_range, assert_softmax_output


class Test_hippmapp3r_initial_stage(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_hippmapp3r_unet_model_3d(1, do_first_network=True).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64, 64))
        assert_sigmoid_range(y)


class Test_hippmapp3r_refine_stage(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_hippmapp3r_unet_model_3d(1, do_first_network=False).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64, 64))
        assert_sigmoid_range(y)

    def test_monte_carlo_dropout_is_stochastic(self):
        """hippmapp3r_segmentation() runs N Monte Carlo dropout iterations on the
        refine-stage network only, by putting just the Dropout3d submodules
        (inside the residual blocks) into train() mode -- see the corresponding
        block in hippmapp3r_segmentation.py. Verify that this actually makes
        consecutive forward passes differ, exactly mirroring the equivalent
        hypermapp3r test in test_wmh_architectures.py."""
        model = create_hippmapp3r_unet_model_3d(1, do_first_network=False).to(DEVICE).eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout3d):
                m.train()

        x = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)

        self.assertFalse(torch.allclose(y1, y2))


class Test_hypothalamus_unet_model_3d(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_hypothalamus_unet_model_3d(input_channel_size=1, number_of_outputs=11).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 11, 64, 64, 64))
        assert_softmax_output(y, channel_dim=1)

    def test_forward_non_cubic_pool_friendly_shape(self):
        """Confirms shape-robustness at a second, non-power-of-two size (the
        pre-BatchNorm skip / _align_leading_3d handling was verified by hand
        against odd input shapes during porting)."""
        model = create_hypothalamus_unet_model_3d(input_channel_size=1, number_of_outputs=11).to(DEVICE).eval()

        x = torch.randn(1, 1, 48, 48, 48, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 11, 48, 48, 48))
        assert_softmax_output(y, channel_dim=1)


class Test_sysu_media_unet_model_2d_claustrum_anatomy(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_sysu_media_unet_model_2d(1, anatomy="claustrum").to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64))
        assert_sigmoid_range(y)


if __name__ == "__main__":
    unittest.main()
