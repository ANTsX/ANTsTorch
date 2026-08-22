"""
Architecture-only tests for
antstorch.utilities.white_matter_hyperintensity_segmentation.

Covers the four bespoke architectures used by the module's five public
functions:

    * create_sysu_media_unet_model_2d  (sysu_media_wmh_segmentation, FLAIR-only and FLAIR+T1)
    * create_sysu_media_unet_model_3d  (wmh_segmentation)
    * create_hypermapp3r_unet_model_3d (hypermapp3r_segmentation)
    * create_shiva_unet_model_3d       (shiva_pvs_segmentation, shiva_wmh_segmentation
                                         -- both build the identical class, so it is
                                         only exercised once per modality count here)

See tests/_arch_common.py for scope / rationale (architecture-only, no
pretrained weights).

The shiva architecture unconditionally pools 7 times (see
create_shiva_unet_model_3d.forward), so it requires an input >= 128
voxels per spatial dimension -- anything smaller makes MaxPool3d
collapse a dimension to 0 partway through the encoder. That makes the
shiva tests noticeably slower than the others; they are marked `slow`
(deselect with `pytest -m "not slow"`).
"""
import unittest

import torch
import torch.nn as nn
import pytest

from antstorch.architectures import (
    create_sysu_media_unet_model_2d,
    create_sysu_media_unet_model_3d,
    create_hypermapp3r_unet_model_3d,
    create_shiva_unet_model_3d,
)

from _arch_common import DEVICE, assert_sigmoid_range


class Test_sysu_media_wmh_segmentation_flair_only(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_sysu_media_unet_model_2d(1).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64))
        assert_sigmoid_range(y)


class Test_sysu_media_wmh_segmentation_flair_t1(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_sysu_media_unet_model_2d(2).to(DEVICE).eval()

        x = torch.randn(1, 2, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64))
        assert_sigmoid_range(y)


class Test_wmh_segmentation(unittest.TestCase):
    """3-D sysu_media architecture used by wmh_segmentation() for both the
    'combined' (antsxnetWmhOr) and 'original' (antsxnetWmh) weight sets --
    architecturally identical, only the weights differ."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_sysu_media_unet_model_3d(
            2, number_of_filters=(64, 96, 128, 256, 512)
        ).to(DEVICE).eval()

        x = torch.randn(1, 2, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y)


class Test_hypermapp3r_segmentation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_hypermapp3r_unet_model_3d(2).to(DEVICE).eval()

        x = torch.randn(1, 2, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y)

    def test_monte_carlo_dropout_is_stochastic(self):
        """hypermapp3r_segmentation() runs N Monte Carlo dropout iterations
        by putting only the Dropout3d submodules into train() mode (see
        _enable_mc_dropout in white_matter_hyperintensity_segmentation.py).
        Verify that this actually makes consecutive forward passes differ."""
        model = create_hypermapp3r_unet_model_3d(2).to(DEVICE).eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

        x = torch.randn(1, 2, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)

        self.assertFalse(torch.allclose(y1, y2))


class Test_shiva_unet_model_3d(unittest.TestCase):
    """Shared architecture for shiva_pvs_segmentation() and
    shiva_wmh_segmentation() (T1-only and T1+FLAIR variants).  Requires an
    input >= 128 voxels per dimension -- see module docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @pytest.mark.slow
    def test_forward_shape_and_range_single_modality(self):
        model = create_shiva_unet_model_3d(number_of_modalities=1).to(DEVICE).eval()

        x = torch.randn(1, 1, 128, 128, 128, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 128, 128, 128))
        assert_sigmoid_range(y)

    @pytest.mark.slow
    def test_forward_shape_and_range_dual_modality(self):
        model = create_shiva_unet_model_3d(number_of_modalities=2).to(DEVICE).eval()

        x = torch.randn(1, 2, 128, 128, 128, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 128, 128, 128))
        assert_sigmoid_range(y)


if __name__ == "__main__":
    unittest.main()
