"""Architecture and wiring tests for antstorch.utilities.harvard_oxford_atlas_labeling."""
import inspect
import unittest

import torch
import torch.nn.functional as F
from antstorch.architectures import (
    create_multihead_unet_model_3d,
    create_unet_model_3d,
)
from antstorch.architectures.create_unet_model import _ConvTranspose3dKerasExact
from antstorch.utilities import harvard_oxford_atlas_labeling as hoa_mod

from tests.support.architecture_checks import (
    DEVICE,
    assert_sigmoid_range,
    assert_softmax_output,
)


class TestHarvardOxfordAtlasArchitecture(unittest.TestCase):
    NUMBER_OF_OUTPUTS = 23
    NUMBER_OF_FILTERS = (16, 32, 64, 128)

    def _build_model(self):
        base = create_unet_model_3d(
            input_channel_size=1,
            number_of_outputs=self.NUMBER_OF_OUTPUTS,
            number_of_filters=self.NUMBER_OF_FILTERS,
            dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3),
            deconvolution_kernel_size=(2, 2, 2),
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            mode="classification",
            pad_crop="keras",
            additional_options=["kerasDeconvolutionStyle"],
        )
        return create_multihead_unet_model_3d(
            base_unet=base,
            n_aux_heads=1,
            use_sigmoid=True,
            n_main_outputs=self.NUMBER_OF_OUTPUTS,
        )

    def test_forward_shape_and_ranges(self):
        model = self._build_model().to(DEVICE).eval()
        x = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
        with torch.no_grad():
            y_main, y_aux = model(x)

        self.assertEqual(tuple(y_main.shape), (1, self.NUMBER_OF_OUTPUTS, 32, 32, 32))
        assert_softmax_output(y_main)

        self.assertEqual(tuple(y_aux.shape), (1, 1, 32, 32, 32))
        assert_sigmoid_range(y_aux)

    def test_uses_keras_exact_deconvolution_layers(self):
        model = self._build_model()
        deconvs = list(model.base.decoding_convolution_transpose_layers)
        self.assertEqual(len(deconvs), 3)
        for layer in deconvs:
            self.assertIsInstance(layer, _ConvTranspose3dKerasExact)

    def test_keras_exact_deconv_has_no_leading_zero_pad_shift(self):
        layer = _ConvTranspose3dKerasExact(
            in_ch=8, out_ch=4, kernel_size=(2, 2, 2), stride=1, bias=True
        ).eval()
        x = torch.randn(1, 8, 6, 6, 6)
        with torch.no_grad():
            actual = layer(x)
            full = F.conv_transpose3d(
                x, layer.deconv.weight, layer.deconv.bias, stride=1, padding=0
            )
            expected = full[:, :, :-1, :-1, :-1]
        self.assertEqual(tuple(actual.shape), (1, 4, 6, 6, 6))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_strict_state_dict_roundtrip(self):
        m1 = self._build_model().eval()
        sample = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            y1_main, y1_aux = m1(sample)
        sd = {k: v.clone() for k, v in m1.state_dict().items()}
        self.assertEqual(len(sd), 38)
        self.assertIn("heads.0.weight", sd)
        self.assertIn("heads.0.bias", sd)

        m2 = self._build_model().eval()
        with torch.no_grad():
            _ = m2(torch.zeros_like(sample))
        missing, unexpected = m2.load_state_dict(sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        with torch.no_grad():
            y2_main, y2_aux = m2(sample)
        self.assertTrue(torch.allclose(y1_main, y2_main, atol=1e-6))
        self.assertTrue(torch.allclose(y1_aux, y2_aux, atol=1e-6))

    def test_source_enforces_keras_deconv_and_strict_load(self):
        src = inspect.getsource(hoa_mod)
        self.assertIn("kerasDeconvolutionStyle", src)
        self.assertIn("strict=True", src)
        self.assertNotIn("strict=False", src)


if __name__ == "__main__":
    unittest.main()
