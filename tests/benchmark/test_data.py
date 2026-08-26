"""Tests for antstorch.benchmark.data — dataset resolution, integrity
checking, and single-pair loading."""
import os

import numpy as np
import pytest

from antstorch.benchmark.data import (
    DEFAULT_DATA_DIR_ENV,
    check_mindboggle_data,
    get_n4_cached_subject_volume,
    load_mindboggle_pair,
    resolve_data_dir,
)


def test_resolve_data_dir_raises_with_setup_instructions_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(DEFAULT_DATA_DIR_ENV, raising=False)
    monkeypatch.delenv("SYNTX_DATA_DIR", raising=False)
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(missing)


def test_resolve_data_dir_accepts_explicit_existing_directory(tmp_path):
    d = tmp_path / "exists"
    d.mkdir()
    assert resolve_data_dir(str(d)) == os.path.abspath(str(d))


def test_resolve_data_dir_uses_env_var_when_arg_omitted(tmp_path, monkeypatch):
    d = tmp_path / "from_env"
    d.mkdir()
    monkeypatch.setenv(DEFAULT_DATA_DIR_ENV, str(d))
    assert resolve_data_dir(None) == os.path.abspath(str(d))


def test_check_mindboggle_data_valid_dataset(mock_mindboggle_dataset):
    pairs_csv, data_dir = mock_mindboggle_dataset
    is_valid, report = check_mindboggle_data(pairs_csv=pairs_csv, data_dir=data_dir)
    assert is_valid is True
    assert report["total_pairs_in_csv"] == 1
    assert report["available_pairs"] == 1
    assert report["missing_files"] == []


def test_check_mindboggle_data_reports_missing_files(mock_mindboggle_dataset, tmp_path):
    pairs_csv, data_dir = mock_mindboggle_dataset
    # Delete one required file to force a reported gap.
    missing_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-2", "t1weighted_brain.nii.gz")
    os.remove(missing_path)
    is_valid, report = check_mindboggle_data(pairs_csv=pairs_csv, data_dir=data_dir)
    assert is_valid is False
    assert report["available_pairs"] == 0
    assert missing_path in report["missing_files"]


def test_check_mindboggle_data_missing_csv_returns_false(tmp_path):
    is_valid, report = check_mindboggle_data(pairs_csv=str(tmp_path / "nope.csv"), data_dir=str(tmp_path))
    assert is_valid is False
    assert report["pairs_csv_exists"] is False


def test_load_mindboggle_pair_returns_expected_structure(mock_mindboggle_dataset):
    pairs_csv, data_dir = mock_mindboggle_dataset
    pair = load_mindboggle_pair(pair_idx=0, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=False)

    assert pair["pair_idx"] == 0
    assert pair["fixed_id"] == "OASIS-TRT-20-1"
    assert pair["moving_id"] == "OASIS-TRT-20-2"
    assert pair["pair_type"] == "intra"
    assert pair["fixed"].dimension == 3
    assert pair["moving"].dimension == 3
    assert pair["fixed_label"].numpy().max() > 0
    assert pair["moving_label"].numpy().max() > 0


def test_load_mindboggle_pair_out_of_range_raises_index_error(mock_mindboggle_dataset):
    pairs_csv, data_dir = mock_mindboggle_dataset
    with pytest.raises(IndexError):
        load_mindboggle_pair(pair_idx=5, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=False)


def test_load_mindboggle_pair_missing_volume_raises_file_not_found(mock_mindboggle_dataset):
    pairs_csv, data_dir = mock_mindboggle_dataset
    os.remove(os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz"))
    with pytest.raises(FileNotFoundError):
        load_mindboggle_pair(pair_idx=0, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=False)


def test_get_n4_cached_subject_volume_use_n4_false_is_passthrough(mock_mindboggle_dataset):
    _, data_dir = mock_mindboggle_dataset
    raw_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz")
    img = get_n4_cached_subject_volume("OASIS-TRT-20", "OASIS-TRT-20-1", raw_path, data_dir, use_n4=False)
    assert img.dimension == 3


def test_get_n4_cached_subject_volume_computes_and_caches(mock_mindboggle_dataset):
    _, data_dir = mock_mindboggle_dataset
    raw_path = os.path.join(data_dir, "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain.nii.gz")
    cache_file = os.path.join(data_dir, ".n4_cache", "OASIS-TRT-20_volumes", "OASIS-TRT-20-1", "t1weighted_brain_n4.nii.gz")
    assert not os.path.exists(cache_file)

    img = get_n4_cached_subject_volume("OASIS-TRT-20", "OASIS-TRT-20-1", raw_path, data_dir, use_n4=True)
    assert img.dimension == 3
    assert os.path.exists(cache_file)

    # Second call must hit the cache rather than recompute.
    cached_mtime = os.path.getmtime(cache_file)
    img2 = get_n4_cached_subject_volume("OASIS-TRT-20", "OASIS-TRT-20-1", raw_path, data_dir, use_n4=True)
    assert os.path.getmtime(cache_file) == cached_mtime
    np.testing.assert_allclose(img.numpy(), img2.numpy())
