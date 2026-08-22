"""
Architecture-only tests for antstorch.utilities.lung_extraction.

No pretrained weights are required -- each test builds the same
architecture (identical class + kwargs) that lung_extraction() builds
internally for a given `modality`, runs one forward pass on random
input, and checks output shape / value range.  See tests/_arch_common.py
for the shared assertion helpers and the rationale for this scope.

Two of the branches (protonLobes/maskLobes, ct) size their input
channels from ANTsXNet prior-image counts that are only known once the
real prior data is loaded (protonLobePriors, luna16LungPriors).  These
tests substitute a small placeholder prior count -- enough to exercise
the architecture wiring (attentionGating, and for protonLobes,
create_multihead_unet_model_3d) without needing real data.
"""
import unittest

import torch

from antstorch.architectures import (
    create_unet_model_2d,
    create_unet_model_3d,
    create_multihead_unet_model_3d,
)

from _arch_common import DEVICE, assert_softmax_output, assert_sigmoid_range


class Test_lung_extraction_proton(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_3d(
            input_channel_size=1,
            number_of_outputs=3,
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(7, 7, 5), deconvolution_kernel_size=(7, 7, 5),
            mode="classification",
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 3, 32, 32, 32))
        assert_softmax_output(y)


class Test_lung_extraction_protonLobes(unittest.TestCase):
    """protonLobes wraps the base U-Net in create_multihead_unet_model_3d
    (1 auxiliary sigmoid head predicting the whole-lung mask)."""

    NUM_LOBE_PRIORS = 5  # placeholder -- real count comes from protonLobePriors data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        channel_size = 1 + self.NUM_LOBE_PRIORS
        number_of_outputs = 1 + self.NUM_LOBE_PRIORS

        base_model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_outputs, mode="classification",
            number_of_filters_at_base_layer=16, number_of_layers=4,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0, additional_options=("attentionGating",),
        )
        model = create_multihead_unet_model_3d(
            base_unet=base_model, n_aux_heads=1, use_sigmoid=True,
            n_main_outputs=number_of_outputs,
        ).to(DEVICE).eval()

        x = torch.randn(1, channel_size, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y_main, y_aux = model(x)

        self.assertEqual(tuple(y_main.shape), (1, number_of_outputs, 32, 32, 32))
        assert_softmax_output(y_main)

        self.assertEqual(tuple(y_aux.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y_aux)


class Test_lung_extraction_maskLobes(unittest.TestCase):
    NUM_LOBE_PRIORS = 5  # placeholder -- must match Test_lung_extraction_protonLobes

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        channel_size = 1 + self.NUM_LOBE_PRIORS
        number_of_outputs = 1 + self.NUM_LOBE_PRIORS

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_outputs, mode="classification",
            number_of_filters_at_base_layer=16, number_of_layers=4,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0, additional_options=("attentionGating",),
        ).to(DEVICE).eval()

        x = torch.randn(1, channel_size, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, number_of_outputs, 32, 32, 32))
        assert_softmax_output(y)


class Test_lung_extraction_ct(unittest.TestCase):
    NUM_LUNA16_PRIORS = 6  # placeholder -- real count comes from luna16LungPriors data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        channel_size = self.NUM_LUNA16_PRIORS + 1
        number_of_outputs = 4  # background, left lung, right lung, airways

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_outputs, mode="classification",
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            additional_options=("attentionGating",),
        ).to(DEVICE).eval()

        x = torch.randn(1, channel_size, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, number_of_outputs, 32, 32, 32))
        assert_softmax_output(y)


class Test_lung_extraction_ventilation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=1,
            number_of_outputs=1, mode="sigmoid",
            number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64))
        assert_sigmoid_range(y)


class Test_lung_extraction_xray(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=3,
            number_of_outputs=3, mode="classification",
            number_of_filters_at_base_layer=32, number_of_layers=4,
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0,
        ).to(DEVICE).eval()

        x = torch.randn(1, 3, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))
        assert_softmax_output(y)


if __name__ == "__main__":
    unittest.main()
