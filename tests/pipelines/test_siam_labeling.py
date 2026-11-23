"""Exercise the real optional nnU-Net preprocessing and prediction API."""
import importlib
import json

import ants
import numpy as np
import pytest
import torch


pytestmark = pytest.mark.integration


def test_siam_with_real_nnunet_predictor(tmp_path, monkeypatch):
    pytest.importorskip("nnunetv2")
    pytest.importorskip("nibabel")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    siam = importlib.import_module("antstorch.utilities.siam_labeling")
    configuration = {
        "data_identifier": "test", "preprocessor_name": "DefaultPreprocessor",
        "batch_size": 1, "patch_size": [16, 16, 16], "spacing": [1., 1., 1.],
        "normalization_schemes": ["ZScoreNormalization"], "use_mask_for_norm": [False],
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0},
        "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0},
        "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0},
        "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": {
                "n_stages": 3, "features_per_stage": [4, 8, 16],
                "conv_op": "torch.nn.Conv3d", "kernel_sizes": [[3, 3, 3]] * 3,
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                "n_conv_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1],
                "conv_bias": True, "norm_op": "torch.nn.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None, "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
        },
    }
    plans = {
        "dataset_name": "test", "plans_name": "test", "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
        "configurations": {"3d_fullres": configuration},
        "foreground_intensity_properties_per_channel": {"0": {}},
    }
    labels = {"background": 0, **{f"label_{i}": i for i in range(1, 18)}}
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    (tmp_path / "dataset.json").write_text(json.dumps(
        {"channel_names": {"0": "T1"}, "labels": labels, "file_ending": ".nii.gz"}))
    kwargs = dict(configuration["architecture"]["arch_kwargs"])
    kwargs.update(conv_op=torch.nn.Conv3d, norm_op=torch.nn.InstanceNorm3d,
                  nonlin=torch.nn.LeakyReLU)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        network = PlainConvUNet(input_channels=1, num_classes=18,
                               deep_supervision=False, **kwargs)
    (tmp_path / "fold_0").mkdir()
    torch.save({"trainer_name": "nnUNetTrainer", "init_args": {"configuration": "3d_fullres"},
                "inference_allowed_mirroring_axes": None, "network_weights": network.state_dict()},
               tmp_path / "fold_0/checkpoint_final.pth")
    monkeypatch.setattr(siam, "_get_siam_model", lambda verbose: (tmp_path, (0,)))

    # RAS in the ANTs LPS coordinate system, with anisotropic sampling.
    array = np.random.default_rng(42).normal(size=(9, 10, 11)).astype(np.float32)
    image = ants.from_numpy(array, spacing=(0.9, 1.1, 1.2), origin=(10, 20, -30),
                           direction=np.diag([-1., -1., 1.]))
    result = siam.siam_labeling(image, return_probability_images=True, device="cpu")
    segmentation_only = siam.siam_labeling(image, device="cpu")
    assert "probability_images" not in segmentation_only
    np.testing.assert_array_equal(segmentation_only["segmentation_image"].numpy(),
                                  result["segmentation_image"].numpy())
    probabilities = np.stack([p.numpy() for p in result["probability_images"]])
    assert probabilities.shape == (18, *image.shape)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1., atol=1e-6)
    assert ants.image_physical_space_consistency(result["segmentation_image"], image)
    np.testing.assert_array_equal(result["segmentation_image"].numpy(), probabilities.argmax(axis=0))

    # Compare against nnU-Net's file reader, as used by the original SIAM.
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    filename = tmp_path / "input.nii.gz"
    ants.image_write(image, str(filename))
    data, properties = SimpleITKIO().read_images([str(filename)])
    reference = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                               use_mirroring=False, perform_everything_on_device=False,
                               device=torch.device("cpu"), allow_tqdm=False)
    reference.initialize_from_trained_model_folder(str(tmp_path), use_folds=(0,))
    seg, proba = reference.predict_single_npy_array(
        data, properties, save_or_return_probabilities=True)
    np.testing.assert_array_equal(result["segmentation_image"].numpy(), seg.transpose(2, 1, 0))
    np.testing.assert_allclose(probabilities, proba.transpose(0, 3, 2, 1), atol=1e-6)
