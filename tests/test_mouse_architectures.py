"""
Architecture-only tests for antstorch.utilities.mouse.

Covers mouse_brain_extraction ("t2" and "ex5" branches),
mouse_brain_parcellation ("nick" and "tct"), mouse_histology_brain_mask,
mouse_histology_hemispherical_coronal_mask,
mouse_histology_cerebellum_mask, and mouse_histology_super_resolution.
See tests/_arch_common.py for scope / rationale (architecture-only, no
pretrained weights).

mouse_cortical_thickness is not covered directly: it is a thin wrapper
around mouse_brain_parcellation("nick") (already covered below) plus
ants.kelly_kapowski(), which has no learned architecture of its own.

mouse_brain_parcellation("jay") is architecturally identical to "nick" /
"tct" (same create_unet_model_3d call, just a different channel count
driven by the STPT template's label count), so it is not duplicated
here.
"""
import unittest

import torch

from antstorch.architectures import (
    create_unet_model_2d,
    create_unet_model_3d,
    create_deep_back_projection_network_model_2d,
)

from _arch_common import DEVICE, assert_softmax_output, assert_sigmoid_range, assert_finite


class Test_mouse_brain_extraction_t2(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_3d(
            input_channel_size=1,
            number_of_outputs=1, mode="sigmoid",
            number_of_filters=(16, 32, 64, 128),
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y)


class Test_mouse_brain_extraction_ex5(unittest.TestCase):
    """Shared architecture for modality="ex5coronal" and "ex5sagittal"
    (only the pretrained weights file differs between the two)."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=1,
            number_of_outputs=2, mode="classification",
            number_of_filters=(64, 96, 128, 256, 512),
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0,
            additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 2, 64, 64))
        assert_softmax_output(y)


class Test_mouse_brain_parcellation_nick(unittest.TestCase):
    # DevCCF "nick" parcellation has 6 non-background labels -> channel_size=7, outputs=7
    NUMBER_OF_NONZERO_LABELS = 6

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        channel_size = 1 + self.NUMBER_OF_NONZERO_LABELS
        number_of_outputs = 1 + self.NUMBER_OF_NONZERO_LABELS

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_outputs,
            mode="classification",
            number_of_filters=(16, 32, 64, 128, 256),
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
        ).to(DEVICE).eval()

        x = torch.randn(1, channel_size, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, number_of_outputs, 32, 32, 32))
        assert_softmax_output(y)


class Test_mouse_brain_parcellation_tct(unittest.TestCase):
    # DevCCF "tct" parcellation has 7 non-background labels -> channel_size=8, outputs=8.
    # (See tasks_registry.py bugfix, 2026-08-22: this variant was previously
    # mis-registered with channel_size=7 / outputs=7, copied from "nick".)
    NUMBER_OF_NONZERO_LABELS = 7

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        channel_size = 1 + self.NUMBER_OF_NONZERO_LABELS
        number_of_outputs = 1 + self.NUMBER_OF_NONZERO_LABELS

        model = create_unet_model_3d(
            input_channel_size=channel_size,
            number_of_outputs=number_of_outputs,
            mode="classification",
            number_of_filters=(16, 32, 64, 128, 256),
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
        ).to(DEVICE).eval()

        x = torch.randn(1, channel_size, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, number_of_outputs, 32, 32, 32))
        assert_softmax_output(y)


class Test_mouse_histology_brain_mask(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=1,
            number_of_outputs=2, mode="classification",
            number_of_filters=(64, 96, 128, 256, 512),
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0,
            additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 2, 64, 64))
        assert_softmax_output(y)


class Test_mouse_histology_hemispherical_coronal_mask(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=1,
            number_of_outputs=3, mode="classification",
            number_of_filters=(64, 96, 128, 256, 512),
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0,
            additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))
        assert_softmax_output(y)


class Test_mouse_histology_cerebellum_mask(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_and_range(self):
        model = create_unet_model_2d(
            input_channel_size=1,
            number_of_outputs=1, mode="sigmoid",
            number_of_filters=(64, 96, 128, 256, 512),
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0,
            additional_options=("initialConvolutionKernelSize[5]", "attentionGating"),
        ).to(DEVICE).eval()

        x = torch.randn(1, 1, 64, 64, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 64, 64))
        assert_sigmoid_range(y)


class Test_mouse_histology_super_resolution(unittest.TestCase):
    """Deep back-projection network, 2x super-resolution (256x256 -> 512x512
    in the real application; exercised here at a much smaller scale)."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_shape_2x_upsampling(self):
        model = create_deep_back_projection_network_model_2d(
            input_channel_size=3,
            number_of_outputs=3, convolution_kernel_size=(6, 6), strides=(2, 2),
        ).to(DEVICE).eval()

        x = torch.randn(1, 3, 32, 32, device=DEVICE)
        with torch.no_grad():
            y = model(x)

        # Regression output (no final activation) -- only shape / finiteness checked.
        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))
        assert_finite(y)


if __name__ == "__main__":
    unittest.main()
