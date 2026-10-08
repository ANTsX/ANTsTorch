"""Tests for ``antstorch.syn.build_template`` (port of ``syntx.build_template``).

CPU-only, tiny 2D phantoms (32x32) with short SyN schedules for speed.
"""

import os

import ants
import numpy as np
import pytest

from antstorch.syn import build_template
from antstorch.syn.template import (
    _compute_shape_residual,
    _initial_template,
    _normalize_weights,
    _template_change,
)

FAST = dict(levels=(2, 1), reg_iterations=(6, 3))


def _disc(cx, cy, size=32, radius=6):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2).astype(np.float32)
    return ants.from_numpy(arr)


def _cohort(offsets=((-3, 0), (3, 0), (0, -3), (0, 3)), size=32):
    c = size // 2
    return [_disc(c + dx, c + dy, size) for dx, dy in offsets]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestNormalizeWeights:
    def test_uniform_default(self):
        np.testing.assert_allclose(_normalize_weights(None, 4), [0.25] * 4)

    def test_normalised_to_one(self):
        w = _normalize_weights([1, 1, 2], 3)
        assert w.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(w, [0.25, 0.25, 0.5])

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            _normalize_weights([1, 2], 3)

    def test_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_weights([1, -1, 1], 3)

    def test_zero_sum(self):
        with pytest.raises(ValueError, match="positive sum"):
            _normalize_weights([0, 0], 2)


class TestInitialTemplate:
    def test_uniform_mean(self):
        imgs = _cohort()
        got = _initial_template(imgs, _normalize_weights(None, 4))
        expected = np.mean([i.numpy() for i in imgs], axis=0)
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-6)

    def test_weights_select_image(self):
        imgs = _cohort()
        got = _initial_template(imgs, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(got.numpy(), imgs[0].numpy(), atol=1e-6)


class TestTemplateChange:
    def test_identical_is_zero(self):
        img = _disc(16, 16)
        assert _template_change(img, img) == 0.0

    def test_known_value(self):
        a = ants.from_numpy(np.zeros((4, 4), dtype=np.float32))
        b = ants.from_numpy(np.full((4, 4), 0.5, dtype=np.float32))
        assert _template_change(a, b) == pytest.approx(0.5)


class TestShapeResidual:
    def _field(self, arr, spacing=(1.0, 1.0)):
        return ants.from_numpy(arr.astype(np.float32), spacing=spacing, has_components=True)

    def test_zero_field(self):
        sr = _compute_shape_residual(self._field(np.zeros((8, 8, 2))))
        assert sr == {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}

    def test_constant_translation(self):
        arr = np.zeros((8, 8, 2))
        arr[..., 0] = 3.0
        arr[..., 1] = 4.0
        sr = _compute_shape_residual(self._field(arr))
        assert sr["l2_norm"] == pytest.approx(5.0)
        assert sr["membrane_energy"] == pytest.approx(0.0, abs=1e-9)
        assert sr["bending_energy"] == pytest.approx(0.0, abs=1e-9)

    def test_linear_ramp_has_membrane_but_no_bending(self):
        xx = np.tile(np.arange(16, dtype=np.float64), (16, 1))
        arr = np.stack([xx, np.zeros_like(xx)], axis=-1)
        sr = _compute_shape_residual(self._field(arr))
        assert sr["membrane_energy"] == pytest.approx(1.0)
        assert sr["bending_energy"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_image_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_template(image_list=[])

    def test_missing_image_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_template()

    @pytest.mark.parametrize("bw", [0.0, -0.1, 1.5])
    def test_bad_blending_weight(self, bw):
        with pytest.raises(ValueError, match="blending_weight"):
            build_template(image_list=_cohort(), blending_weight=bw)

    @pytest.mark.parametrize("gs", [-0.1, 1.1])
    def test_bad_gradient_step(self, gs):
        with pytest.raises(ValueError, match="gradient_step"):
            build_template(image_list=_cohort(), gradient_step=gs)

    def test_bad_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            build_template(image_list=_cohort(), iterations=0)

    def test_bad_weights_length(self):
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            build_template(image_list=_cohort(), weights=[1, 2])

    @pytest.mark.parametrize("name", ["initial_affine", "outprefix"])
    def test_managed_kwargs_rejected(self, name):
        with pytest.raises(TypeError, match=name):
            build_template(image_list=_cohort(), **{name: None})


# ---------------------------------------------------------------------------
# integration (runs syn_registration)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def syn_result(tmp_path_factory):
    work = tmp_path_factory.mktemp("tmpl")
    out = build_template(
        image_list=_cohort(), iterations=2, output_dir=str(work), blending_weight=0.75, **FAST
    )
    return out, work


class TestBuildTemplateSyN:
    def test_schema(self, syn_result):
        out, _ = syn_result
        assert set(out) == {
            "template",
            "warped_images",
            "fwdtransforms",
            "invtransforms",
            "convergence",
            "shape_residuals",
            "n_iterations",
            "weights",
            "work_dir",
            "elapsed_sec",
        }

    def test_template_geometry(self, syn_result):
        out, _ = syn_result
        assert isinstance(out["template"], ants.ANTsImage)
        assert out["template"].shape == (32, 32)
        assert np.isfinite(out["template"].numpy()).all()

    def test_iteration_bookkeeping(self, syn_result):
        out, _ = syn_result
        assert out["n_iterations"] == 2
        assert len(out["convergence"]) == 2
        assert len(out["shape_residuals"]) == 2
        assert all(np.isfinite(c) and c >= 0 for c in out["convergence"])
        for sr in out["shape_residuals"]:
            assert set(sr) == {"l2_norm", "membrane_energy", "bending_energy"}
            assert all(np.isfinite(v) and v >= 0 for v in sr.values())

    def test_per_image_outputs(self, syn_result):
        out, _ = syn_result
        assert len(out["warped_images"]) == 4
        for fwd, inv in zip(out["fwdtransforms"], out["invtransforms"]):
            # ants.registration() convention: [warp, affine] / [affine, inverse warp]
            assert len(fwd) == 2 and len(inv) == 2
            assert fwd[0].endswith("1Warp.nii.gz") and fwd[1].endswith("0GenericAffine.mat")
            assert inv[0].endswith("0GenericAffine.mat") and inv[1].endswith("1InverseWarp.nii.gz")
            assert all(os.path.exists(p) for p in fwd + inv)

    def test_work_dir_artifacts(self, syn_result):
        out, work = syn_result
        assert out["work_dir"] == str(work)
        assert (work / "avgAffine_0.mat").exists()
        assert (work / "avgWarp_0.nii.gz").exists()
        assert (work / "avgWarp_1.nii.gz").exists()

    def test_second_iteration_reuses_affine(self, syn_result):
        # SyNOnly from the cached affine: the affine file written at iteration 1
        # must equal the one fitted at iteration 0 (no affine drift).
        _, work = syn_result
        a0 = ants.read_transform(str(work / "iter00_img0000" / "out0GenericAffine.mat"))
        a1 = ants.read_transform(str(work / "iter01_img0000" / "out0GenericAffine.mat"))
        np.testing.assert_allclose(a0.parameters, a1.parameters, atol=1e-4)

    def test_weights_normalised(self, syn_result):
        out, _ = syn_result
        np.testing.assert_allclose(out["weights"], [0.25] * 4)

    def test_default_work_dir_is_temporary(self):
        out = build_template(image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, **FAST)
        assert os.path.isdir(out["work_dir"])
        assert "antstorch_tmpl_" in os.path.basename(out["work_dir"])


class TestOptions:
    def test_initial_template_is_used_and_not_mutated(self):
        imgs = _cohort(offsets=((-2, 0), (2, 0)))
        start = _disc(16, 16)
        before = start.numpy().copy()
        out = build_template(initial_template=start, image_list=imgs, iterations=1, **FAST)
        np.testing.assert_array_equal(start.numpy(), before)
        assert out["template"].shape == start.shape

    def test_no_blending_matches_weight_one(self):
        imgs = _cohort(offsets=((-2, 0), (2, 0)))
        out = build_template(image_list=imgs, iterations=1, blending_weight=1.0, **FAST)
        assert np.isfinite(out["template"].numpy()).all()

    def test_convergence_threshold_stops_early(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=4,
            convergence_threshold=1e6,
            **FAST,
        )
        assert out["n_iterations"] == 1

    def test_affine_every_iteration(self, tmp_path):
        build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=2,
            affine_every_iteration=True,
            output_dir=str(tmp_path),
            **FAST,
        )
        # full registration at iteration 1 as well: affine fitted again, warp still present
        assert (tmp_path / "iter01_img0000" / "out0GenericAffine.mat").exists()
        assert (tmp_path / "iter01_img0000" / "out1Warp.nii.gz").exists()

    def test_use_no_rigid_false(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, use_no_rigid=False, **FAST
        )
        assert np.isfinite(out["template"].numpy()).all()

    def test_linear_only_type_has_no_warp(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=1,
            type_of_transform="Rigid",
        )
        assert out["shape_residuals"][0] == {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}
        assert all(len(f) == 1 for f in out["fwdtransforms"])

    def test_syn_kwargs_forwarded(self):
        # a regularizer keyword must reach syn_registration without error
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, regularizer="sobolev", **FAST
        )
        assert np.isfinite(out["template"].numpy()).all()


def test_identical_images_are_a_fixed_point():
    """Registering identical images leaves the template (almost) unchanged and the mean warp tiny."""
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float64)
    blob = ants.from_numpy(np.exp(-(((xx - 24) / 7.0) ** 2 + ((yy - 24) / 7.0) ** 2)).astype(np.float32))
    out = build_template(
        image_list=[blob, blob.clone(), blob.clone()], iterations=2, blending_weight=1.0, **FAST
    )
    corr = np.corrcoef(out["template"].numpy().ravel(), blob.numpy().ravel())[0, 1]
    assert corr > 0.99
    assert max(sr["l2_norm"] for sr in out["shape_residuals"]) < 0.5
