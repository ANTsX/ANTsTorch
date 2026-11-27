"""Data identifiers, archive filenames, checksums and cache reuse."""
import importlib
from unittest.mock import Mock

import pytest


data = importlib.import_module("antstorch.utilities.get_antstorch_data")


@pytest.mark.parametrize("file_id,filename,file_number,sha256", [
    ("siamV03", "siamV03.tar.gz", "71795392",
     "0a8e6964f1dbb9aab85b0041d70b26262bc1e261f9e9cbe1187f3a0b3ced641b"),
    ("kirby", "kirby.nii.gz", "25620107", None),
    ("tutorialLungCT", "tutorialLungCT.nii.gz", "42934234", None),
    ("tutorialLungProton", "tutorialLungProton.nii.gz", "42934228", None),
    ("tutorialLungVentilation", "tutorialLungVentilation.nii.gz", "42934231", None),
    ("tutorialWmhT1", "tutorialWmhT1.nii.gz", "40251796", None),
    ("tutorialWmhFlair", "tutorialWmhFlair.nii.gz", "40251793", None),
    ("tutorialMouseT2", "tutorialMouseT2.nii.gz", "45289309", None),
    ("tutorialMra", "tutorialMra.nii.gz", "46406755", None),
    ("magetCerebellumxTemplate0GenericAffine",
     "magetCerebellumxTemplate0GenericAffine.mat", "41052575", None),
])
def test_data_download_and_cache(monkeypatch, tmp_path, file_id, filename, file_number, sha256):
    monkeypatch.setattr(data, "get_antstorch_cache_directory", lambda: str(tmp_path))
    download = Mock()
    monkeypatch.setattr(data, "download_url_to_file", download)
    target = tmp_path / filename
    assert data.get_antstorch_data(file_id, show_progress=False) == str(target)
    download.assert_called_once_with(
        "https://ndownloader.figshare.com/files/" + file_number, str(target),
        hash_prefix=sha256, progress=False)
    target.touch()
    download.reset_mock()
    assert data.get_antstorch_data(file_id) == str(target)
    download.assert_not_called()


def test_siam_identifier_and_custom_filename(monkeypatch, tmp_path):
    assert "siamV03" in data.get_antstorch_data("show")
    monkeypatch.setattr(data, "get_antstorch_cache_directory", lambda: str(tmp_path))
    download = Mock()
    monkeypatch.setattr(data, "download_url_to_file", download)
    target = tmp_path / "custom.tar.gz"
    assert data.get_antstorch_data("siamV03", target_file_name=target.name) == str(target)
    assert download.call_args.args[1] == str(target)
    assert download.call_args.kwargs["hash_prefix"] == data._EXPECTED_SHA256["siamV03"]
