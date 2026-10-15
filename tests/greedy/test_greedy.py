"""Tests for antstorch.greedy (port of syntx.greedy)."""
import ants
import numpy as np
import pytest

import antstorch
from antstorch.greedy import GreedyRegistrationModel, greedy_registration


def _blob(shift, shape=(32, 36, 28)):
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    im = np.exp(-(((x - 14 - shift[0]) / 7) ** 2 + ((y - 18 - shift[1]) / 9) ** 2 + ((z - 16) / 6) ** 2))
    im += 0.6 * np.exp(-(((x - 9) / 3) ** 2 + ((y - 21) / 3) ** 2 + ((z - 14) / 3) ** 2))
    return ants.from_numpy(im.astype("float32"))


def _require_central_pyramid():
    """antstorch.greedy ports the syntx.greedy that uses syntx.pyramid.build_pyramid (syntx >= 6.2)."""
    pyr = pytest.importorskip("syntx.pyramid", reason="syntx has no central pyramid")
    if not hasattr(pyr, "build_pyramid"):
        pytest.skip("syntx < 6.2: greedy has no central pyramid (antstorch.greedy ports the newer one)")


def _corr(a, b):
    return float(np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1])


def test_exported():
    from antstorch import greedy as mod
    assert mod.greedy is greedy_registration


def test_improves_similarity_and_exports_transforms(tmp_path):
    fx, mv = _blob((0, 0)), _blob((3, -2))
    reg = greedy_registration(fx, mv, reg_iterations=[15, 10, 5], initial_transform=False, device="cpu",
                              outprefix=str(tmp_path / "g"))
    assert len(reg["fwdtransforms"]) == 1
    assert reg["invtransforms"] == [] and reg["warpedfixout"] is None
    assert _corr(reg["warpedmovout"], fx) > _corr(mv, fx)
    disp = ants.image_read(reg["fwdtransforms"][0])
    assert disp.components == 3 and disp.shape == fx.shape


def test_return_inverse(tmp_path):
    fx, mv = _blob((0, 0)), _blob((2, 1))
    reg = greedy_registration(fx, mv, reg_iterations=[8, 4], initial_transform=False, device="cpu",
                              return_inverse=True, outprefix=str(tmp_path / "g"))
    assert len(reg["invtransforms"]) == 1 and reg["warpedfixout"] is not None


def test_bad_keyword_and_regularizer():
    fx = _blob((0, 0))
    with pytest.raises(TypeError):
        greedy_registration(fx, fx, bogus=1, initial_transform=False, device="cpu")
    with pytest.raises(ValueError):
        GreedyRegistrationModel(regularizer="sobolev")
    with pytest.raises(ValueError):
        GreedyRegistrationModel(similarity_metric="nope")


def test_parity_with_syntx_greedy_cpu():
    """Same inputs, no affine: the displacement fields are identical (bit for bit on CPU)."""
    sg = pytest.importorskip("syntx.greedy", reason="syntx not installed")
    _require_central_pyramid()
    fx, mv = _blob((0, 0)), _blob((3, -2))
    kw = dict(reg_iterations=[20, 10, 5], initial_transform=False, device="cpu")
    a = greedy_registration(fx, mv, **kw)
    s = sg.greedy_registration(fx, mv, **kw)
    da = ants.image_read(a["fwdtransforms"][0]).numpy()
    ds = ants.image_read(s["fwdtransforms"][0]).numpy()
    np.testing.assert_allclose(
        da, ds, atol=1e-5,
        err_msg=f"antstorch.greedy differs from the installed syntx.greedy ({sg.__file__}); the comparison is only "
                "meaningful against the same syntx version the port was written from (syntx >= 6.1.1 sources).")


def test_parity_with_syntx_greedy_cpu_anisotropic():
    """Anisotropic spacing and a moving image on its own grid exercise the spacing-aware 'deformable' pyramid."""
    sg = pytest.importorskip("syntx.greedy", reason="syntx not installed")
    _require_central_pyramid()
    fx = _blob((0, 0), shape=(40, 44, 24)); fx.set_spacing((1.0, 1.0, 3.0))
    mv = _blob((2, -2), shape=(36, 40, 22)); mv.set_spacing((1.2, 1.2, 3.3))
    kw = dict(reg_iterations=[6, 4, 3], initial_transform=False, device="cpu")
    wa = greedy_registration(fx, mv, **kw)["model"].warp.detach().numpy()
    ws = sg.greedy_registration(fx, mv, **kw)["model"].warp.detach().numpy()
    np.testing.assert_allclose(wa, ws, atol=1e-5)
