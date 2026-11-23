"""SIAM geometry and model-cache contracts without downloading weights."""
import builtins
import hashlib
import importlib
import io
import json
import tarfile
from unittest.mock import Mock

import ants
import numpy as np
import pytest
import torch


siam = importlib.import_module("antstorch.utilities.siam_labeling")


@pytest.mark.parametrize("direction", [
    np.eye(3),
    np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]]),
    np.array([[0.8, -0.6, 0.], [0.6, 0.8, 0.], [0., 0., 1.]]),
])
@pytest.mark.parametrize("return_probability_images", [True, False])
def test_native_geometry_and_probability_order(monkeypatch, tmp_path, direction,
                                               return_probability_images):
    nib = pytest.importorskip("nibabel")
    array = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 18).astype(np.float32)
    image = ants.from_numpy(array, origin=(10., -20., 30.),
                           spacing=(0.8, 1.3, 2.1), direction=direction)
    init = Mock()
    predict = Mock()

    class Predictor:
        dataset_json = {"labels": {"background": 0, "Anomalies": 17}}

        def __init__(self, **kwargs):
            init(**kwargs)

        def initialize_from_trained_model_folder(self, *args, **kwargs):
            assert args == (str(tmp_path),)
            assert kwargs == {"use_folds": (0, 1, 2, 3, 4),
                              "checkpoint_name": "checkpoint_final.pth"}

        def predict_single_npy_array(self, data, properties, **kwargs):
            predict(data, properties, **kwargs)
            seg = data[0].astype(np.uint8)
            if not kwargs["save_or_return_probabilities"]:
                return seg
            probabilities = np.stack([seg == label for label in range(18)]).astype(np.float32)
            return seg, probabilities

    monkeypatch.setattr(siam, "_load_siam_dependencies", lambda: (nib, Predictor))
    monkeypatch.setattr(siam, "_get_siam_model", lambda verbose: (tmp_path, (0, 1, 2, 3, 4)))
    monkeypatch.setattr(siam, "get_default_device", lambda: torch.device("cpu"))
    result = siam.siam_labeling(image, return_probability_images=return_probability_images)
    assert init.call_args.kwargs["device"] == torch.device("cpu")
    assert init.call_args.kwargs["use_mirroring"] is False
    assert init.call_args.kwargs["use_gaussian"] is True
    assert init.call_args.kwargs["tile_step_size"] == 0.5
    assert init.call_args.kwargs["perform_everything_on_device"] is False
    assert predict.call_args.kwargs == {"save_or_return_probabilities": return_probability_images}
    assert predict.call_args.args[0].dtype == np.float32
    # Verify SimpleITK axis order and spacing on a known orientation.
    if np.array_equal(direction, np.eye(3)):
        np.testing.assert_array_equal(predict.call_args.args[0][0], array[::-1, ::-1, :].transpose(2, 1, 0))
        np.testing.assert_allclose(predict.call_args.args[1]["spacing"], (2.1, 1.3, 0.8))
    np.testing.assert_array_equal(result["segmentation_image"].numpy(), array)
    assert ("probability_images" in result) is return_probability_images
    for label, probability in enumerate(result.get("probability_images", [])):
        np.testing.assert_array_equal(probability.numpy(), array == label)
        assert ants.image_physical_space_consistency(probability, image)
    assert ants.image_physical_space_consistency(result["segmentation_image"], image)
    np.testing.assert_array_equal(image.numpy(), array)


@pytest.mark.parametrize("array", [np.ones((4, 4)), np.ones((4, 4, 4)),
                                   np.full((4, 4, 4), np.nan)])
def test_reject_invalid_images_before_loading_dependencies(monkeypatch, array):
    load = Mock(side_effect=AssertionError("No dependencies or downloads expected"))
    monkeypatch.setattr(siam, "_load_siam_dependencies", load)
    with pytest.raises(ValueError):
        siam.siam_labeling(ants.from_numpy(array.astype(np.float32)))
    load.assert_not_called()


def test_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def missing_nnunet(name, *args, **kwargs):
        if name.startswith("nnunetv2"):
            raise ModuleNotFoundError("nnunetv2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_nnunet)
    with pytest.raises(ImportError, match=r"antstorch\[siam\]"):
        siam._load_siam_dependencies()


def model_archive(extra_members=()):
    output = io.BytesIO()
    files = {"dataset.json": b'{"labels": {"background": 0}}',
             "plans.json": b'{}', "fold_0/checkpoint_final.pth": b'checkpoint'}
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(siam._MODEL_NAME + "/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        for member in extra_members:
            archive.addfile(member, io.BytesIO(b'bad'))
    return output.getvalue()


def mock_download(monkeypatch, tmp_path, data):
    archive = tmp_path / "siamV03.tar.gz"
    archive.write_bytes(data)
    download = Mock(return_value=str(archive))
    monkeypatch.setattr(siam, "get_antstorch_data", download)
    monkeypatch.setattr(siam, "get_antstorch_cache_directory", lambda: str(tmp_path))
    return download


def test_download_checksum_extraction_and_cache_reuse(monkeypatch, tmp_path):
    data = model_archive()
    monkeypatch.setattr(siam, "_ARCHIVE_MD5", hashlib.md5(data).hexdigest())
    download = mock_download(monkeypatch, tmp_path, data)
    folder, folds = siam._get_siam_model()
    assert folds == (0,)
    assert (folder / "fold_0/checkpoint_final.pth").read_bytes() == b'checkpoint'
    assert siam._get_siam_model() == (folder, folds)
    download.assert_called_once_with("siamV03", show_progress=False)
    assert (tmp_path / "siamV03.tar.gz").is_file()
    assert not list(folder.parent.glob(".install-*"))
    assert not list(folder.parent.rglob("*.tar.gz"))


def test_bad_checksum_never_installs_model(monkeypatch, tmp_path):
    mock_download(monkeypatch, tmp_path, model_archive())
    with pytest.raises(ValueError, match="checksum"):
        siam._get_siam_model()
    cache = tmp_path / "siam/v0.3"
    assert not (cache / siam._MODEL_NAME).exists()
    assert not list(cache.glob(".install-*"))


def test_extract_ignores_paths_outside_model(monkeypatch, tmp_path):
    traversal = tarfile.TarInfo(siam._MODEL_NAME + "/../../escaped")
    traversal.size = 3
    archive_path = tmp_path / "model.tar.gz"
    archive_path.write_bytes(model_archive([traversal]))
    extracted = tmp_path / "extracted"
    siam._extract_model(archive_path, extracted)
    assert not (tmp_path / "escaped").exists()
    assert siam._model_folds(extracted) == (0,)


def test_extract_rejects_linked_checkpoints(tmp_path):
    link = tarfile.TarInfo(siam._MODEL_NAME + "/fold_1/checkpoint_final.pth")
    link.type = tarfile.SYMTYPE
    link.linkname = "/elsewhere"
    archive_path = tmp_path / "model.tar.gz"
    archive_path.write_bytes(model_archive([link]))
    with pytest.raises(ValueError, match="non-regular"):
        siam._extract_model(archive_path, tmp_path / "extracted")


def test_incomplete_cache_is_not_accepted(tmp_path):
    (tmp_path / "antstorch_model.json").write_text(json.dumps(
        {"archive_md5": siam._MODEL_MD5, "folds": [0]}))
    assert siam._model_folds(tmp_path) is None


def mock_predictor(monkeypatch, tmp_path, folds=(0, 1, 2, 3, 4)):
    nib = pytest.importorskip("nibabel")
    calls = {}

    class Predictor:
        dataset_json = {"labels": {"background": 0, "Anomalies": 17}}

        def __init__(self, **kwargs):
            pass

        def initialize_from_trained_model_folder(self, folder, use_folds, checkpoint_name):
            calls["use_folds"] = use_folds

        def predict_single_npy_array(self, data, properties, **kwargs):
            calls["spacing"] = properties["spacing"]
            return data[0].astype(np.uint8)

    monkeypatch.setattr(siam, "_load_siam_dependencies", lambda: (nib, Predictor))
    monkeypatch.setattr(siam, "_get_siam_model", lambda verbose: (tmp_path, folds))
    monkeypatch.setattr(siam, "get_default_device", lambda: torch.device("cpu"))
    return calls


@pytest.mark.parametrize("voxel_size, expected", [(None, (2.1, 1.3, 0.8)), (0.75, (0.75,) * 3)])
def test_voxel_size_changes_only_model_spacing(monkeypatch, tmp_path, voxel_size, expected):
    calls = mock_predictor(monkeypatch, tmp_path)
    array = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 18).astype(np.float32)
    direction = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    image = ants.from_numpy(array, origin=(10., -20., 30.), spacing=(0.8, 1.3, 2.1),
                           direction=direction)
    result = siam.siam_labeling(image, voxel_size=voxel_size)
    np.testing.assert_allclose(sorted(calls["spacing"]), sorted(expected))
    if voxel_size is not None:
        assert calls["spacing"] == expected
    np.testing.assert_array_equal(result["segmentation_image"].numpy(), array)
    assert ants.image_physical_space_consistency(result["segmentation_image"], image)


@pytest.mark.parametrize("number_of_folds, expected", [(None, (0, 1, 2, 3, 4)), (1, (0,)), (3, (0, 1, 2))])
def test_number_of_folds_selects_first_folds(monkeypatch, tmp_path, number_of_folds, expected):
    calls = mock_predictor(monkeypatch, tmp_path)
    image = ants.from_numpy((np.arange(64).reshape(4, 4, 4) % 18).astype(np.float32))
    siam.siam_labeling(image, number_of_folds=number_of_folds)
    assert tuple(calls["use_folds"]) == expected


def test_too_many_folds(monkeypatch, tmp_path):
    mock_predictor(monkeypatch, tmp_path, folds=(0, 1))
    image = ants.from_numpy((np.arange(64).reshape(4, 4, 4) % 18).astype(np.float32))
    with pytest.raises(ValueError, match="at most 2"):
        siam.siam_labeling(image, number_of_folds=3)


@pytest.mark.parametrize("kwargs", [{"voxel_size": 0}, {"voxel_size": -0.75},
                                    {"voxel_size": float("nan")}, {"number_of_folds": 0},
                                    {"number_of_folds": 1.5}, {"number_of_folds": True}])
def test_reject_invalid_options_before_loading_dependencies(monkeypatch, kwargs):
    load = Mock(side_effect=AssertionError("No dependencies or downloads expected"))
    monkeypatch.setattr(siam, "_load_siam_dependencies", load)
    image = ants.from_numpy((np.arange(64).reshape(4, 4, 4) % 18).astype(np.float32))
    with pytest.raises(ValueError):
        siam.siam_labeling(image, **kwargs)
    load.assert_not_called()
