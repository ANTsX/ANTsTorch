"""Tests for SHA-256 cache verification and atomic downloads in get_pretrained_network."""
import hashlib
import importlib
import os
import warnings

import antstorch
import pytest
from antstorch.utilities.get_antstorch_data import (
    get_antstorch_cache_directory,
    set_antstorch_cache_directory,
)

gpn_mod = importlib.import_module("antstorch.utilities.get_pretrained_network")


@pytest.fixture()
def isolated_cache_dir(tmp_path):
    prev = get_antstorch_cache_directory()
    set_antstorch_cache_directory(str(tmp_path))
    try:
        yield tmp_path
    finally:
        set_antstorch_cache_directory(prev)


def test_show_returns_valid_ids_including_hoa():
    ids = antstorch.get_pretrained_network("show")
    assert "HarvardOxfordAtlasSubcortical_pytorch" in ids
    assert len(ids) == len(set(ids))


def test_missing_or_unknown_file_id_raises():
    with pytest.raises(ValueError, match="Missing file id"):
        antstorch.get_pretrained_network(None)
    with pytest.raises(ValueError):
        antstorch.get_pretrained_network("nonexistent_model_id_xyz")


def test_verified_cache_hit_skips_download(isolated_cache_dir, monkeypatch):
    payload = b"verified-hoa-weights-payload"
    digest = hashlib.sha256(payload).hexdigest()
    target = isolated_cache_dir / "HarvardOxfordAtlasSubcortical_pytorch.pt"
    target.write_bytes(payload)

    monkeypatch.setitem(
        gpn_mod._EXPECTED_SHA256, "HarvardOxfordAtlasSubcortical_pytorch", digest
    )

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("download_url_to_file should not be called on verified cache hit")

    monkeypatch.setattr(gpn_mod, "download_url_to_file", _unexpected_download)

    resolved = antstorch.get_pretrained_network(
        "HarvardOxfordAtlasSubcortical_pytorch", show_progress=False
    )
    assert resolved == str(target)


def test_stale_cached_file_warns_and_atomically_redownloads(
    isolated_cache_dir, monkeypatch
):
    stale_payload = b"stale-pre-update-hoa-weights"
    fresh_payload = b"fresh-verified-67802100-hoa-weights"
    fresh_digest = hashlib.sha256(fresh_payload).hexdigest()

    target = isolated_cache_dir / "HarvardOxfordAtlasSubcortical_pytorch.pt"
    target.write_bytes(stale_payload)

    monkeypatch.setitem(
        gpn_mod._EXPECTED_SHA256, "HarvardOxfordAtlasSubcortical_pytorch", fresh_digest
    )
    calls = []

    def _fake_download(url, dst, hash_prefix=None, progress=True):
        calls.append((url, dst, hash_prefix))
        with open(dst, "wb") as fh:
            fh.write(fresh_payload)

    monkeypatch.setattr(gpn_mod, "download_url_to_file", _fake_download)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = antstorch.get_pretrained_network(
            "HarvardOxfordAtlasSubcortical_pytorch", show_progress=False
        )

    assert resolved == str(target)
    assert target.read_bytes() == fresh_payload
    assert len(calls) == 1
    assert calls[0][2] == fresh_digest
    assert len(caught) == 1
    assert "re-downloading" in str(caught[0].message)
    partials = [p for p in os.listdir(isolated_cache_dir) if p.endswith(".partial")]
    assert partials == []


def test_corrupt_download_raises_and_cleans_partial(isolated_cache_dir, monkeypatch):
    expected_digest = hashlib.sha256(b"expected-weights").hexdigest()
    monkeypatch.setitem(
        gpn_mod._EXPECTED_SHA256,
        "HarvardOxfordAtlasSubcortical_pytorch",
        expected_digest,
    )

    def _corrupt_download(url, dst, hash_prefix=None, progress=True):
        with open(dst, "wb") as fh:
            fh.write(b"corrupted-bytes")

    monkeypatch.setattr(gpn_mod, "download_url_to_file", _corrupt_download)

    with pytest.raises(RuntimeError, match="expected"):
        antstorch.get_pretrained_network(
            "HarvardOxfordAtlasSubcortical_pytorch", show_progress=False
        )

    partials = [p for p in os.listdir(isolated_cache_dir) if p.endswith(".partial")]
    assert partials == []
    assert not (isolated_cache_dir / "HarvardOxfordAtlasSubcortical_pytorch.pt").exists()


def test_custom_target_file_name_bypasses_default_digest(
    isolated_cache_dir, monkeypatch
):
    custom = isolated_cache_dir / "custom_override.pt"
    custom.write_bytes(b"user-supplied-custom-checkpoint")

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("download_url_to_file should not be called")

    monkeypatch.setattr(gpn_mod, "download_url_to_file", _unexpected_download)

    resolved = antstorch.get_pretrained_network(
        "HarvardOxfordAtlasSubcortical_pytorch",
        target_file_name="custom_override.pt",
        show_progress=False,
    )
    assert resolved == str(custom)
