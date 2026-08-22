"""
Architecture-only tests for antstorch.utilities.lung_segmentation.

Covers el_bicho, lung_pulmonary_artery_segmentation and
lung_airway_segmentation -- see tests/_arch_common.py for scope /
rationale (architecture-only, no pretrained weights).
"""
import unittest

import torch

from antstorch.architectures import create_unet_model_2d, create_unet_model_3d

from _arch_common import DEVICE, assert_softmax_output, assert_sigmoid_range


class Test_el_bicho(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=2,
            number_of_outputs=5,
            number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            additional_options=("attentionGating",),
        ).to(DEVICE).eval()

        x = torch.randn(1, 2, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 5, 64, 64))
        assert_softmax_output(y)


class Test_lung_pulmonary_artery_segmentation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_3d(
            input_channel_size=1,
            number_of_outputs=1, mode="sigmoid",
            number_of_filters=(32, 64, 128, 256, 512),
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0,
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y)


class Test_lung_airway_segmentation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_3d(
            input_channel_size=1,
            number_of_outputs=2, mode="classification",
            number_of_filters=(32, 64, 128, 256, 512),
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0,
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 2, 32, 32, 32))
        assert_softmax_output(y)


if __name__ == "__main__":
    unittest.main()
