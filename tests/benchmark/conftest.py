"""Shared fixtures for antstorch.benchmark tests: a tiny synthetic
Mindboggle-style dataset (small random 3-D volumes + integer label maps +
a 1-row pairs.csv), so the full evaluate_mindboggle_pair() pipeline can run
end-to-end without the real ~101-subject Mindboggle-101 dataset.
"""
import os

import numpy as np
import pytest

import ants

_SHAPE = (24, 28, 24)
_SPACING = (2.0, 2.0, 2.0)


def _make_subject_volume(rng, shape=_SHAPE, spacing=_SPACING, blob_scale=150.0):
    vol = rng.normal(loc=100.0, scale=10.0, size=shape).astype(np.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
    blob = np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)))
    vol = np.clip(vol + blob_scale * blob, 0, None)
    return ants.from_numpy(vol.astype(np.float32), spacing=spacing)


def _make_subject_labels(shape=_SHAPE, spacing=_SPACING):
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
    r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    labels = np.zeros(shape, dtype=np.float32)
    labels[r < 8] = 1
    labels[r < 5] = 2
    labels[r < 2] = 3
    return ants.from_numpy(labels.astype(np.float32), spacing=spacing)


def write_mock_subject(data_dir, cohort, subject, seed):
    rng = np.random.default_rng(seed)
    d = os.path.join(data_dir, f"{cohort}_volumes", subject)
    os.makedirs(d, exist_ok=True)
    ants.image_write(_make_subject_volume(rng), os.path.join(d, "t1weighted_brain.nii.gz"))
    ants.image_write(_make_subject_labels(), os.path.join(d, "labels.DKT31.manual.nii.gz"))


@pytest.fixture
def mock_mindboggle_dataset(tmp_path):
    """Writes a 1-intra-pair mock dataset under tmp_path; returns (pairs_csv, data_dir)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_mock_subject(str(data_dir), "OASIS-TRT-20", "OASIS-TRT-20-1", seed=1)
    write_mock_subject(str(data_dir), "OASIS-TRT-20", "OASIS-TRT-20-2", seed=2)

    pairs_csv = tmp_path / "pairs.csv"
    pairs_csv.write_text(
        "type,cohort1,subject1,cohort2,subject2\n"
        "intra,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-2\n"
    )
    return str(pairs_csv), str(data_dir)
