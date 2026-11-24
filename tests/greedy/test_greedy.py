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
        GreedyRegistrationModel(regularizer="nope")
    with pytest.raises(ValueError):  # componentwise Adam breaks the div/curl coupling (syntx >= 6.43)
        GreedyRegistrationModel(regularizer="navier", optimizer="adam")
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


def test_return_inverse_is_a_true_inverse(tmp_path):
    """Since 2026-11-18 (syntx >= 6.36) the physical inverse is solved in the tensor layout (axes and vector
    components reversed); before, the components stayed in (x, y, z) order and the 'inverse' was wrong by ~mm."""
    import torch
    from antstorch.greedy import _ants_displacement_to_zyx
    from antstorch.syn.core.inverse import compute_inverse_identity_error_nd
    fx, mv = _blob((0, 0)), _blob((3, -2))
    reg = greedy_registration(fx, mv, reg_iterations=[20, 10, 5], initial_transform=False, device="cpu",
                              return_inverse=True, outprefix=str(tmp_path / "g"))
    fwd = _ants_displacement_to_zyx(ants.image_read(reg["fwdtransforms"][0]), "cpu").double()
    inv = _ants_displacement_to_zyx(ants.image_read(reg["invtransforms"][0]), "cpu").double()
    assert float(fwd.abs().max()) > 1.0
    # Median over the volume: the flat background of these blobs is poorly constrained (syntx's own inverse has a
    # ~15 mm maximum error there too). Measured 2026-11-18: median 0.024 mm (the former layout gave 0.198 mm).
    err = compute_inverse_identity_error_nd(inv, fwd, spacing=fx.spacing, origin=fx.origin, direction=np.eye(3))
    assert float(err.median()) < 0.06


def test_convergence_record_and_options():
    fx, mv = _blob((0, 0)), _blob((3, -2))
    reg = greedy_registration(fx, mv, reg_iterations=[12, 8], initial_transform=False, device="cpu")
    conv = reg["provenance"]["convergence"]
    assert conv["slope_threshold"] == 1e-6 and [r["level"] for r in conv["levels"]] == [0, 1]
    assert reg["provenance"]["digital_diffeomorphism"] is True
    off = greedy_registration(fx, mv, reg_iterations=[12, 8], initial_transform=False, device="cpu",
                              convergence_slope=None, digital_diffeomorphism=False)
    assert off["provenance"]["convergence"]["slope_threshold"] is None
    assert all(r["iterations"] == r["cap"] for r in off["provenance"]["convergence"]["levels"])


# --- syntx 6.40-6.44 alignment (2026-11-23) -----------------------------------


def test_defaults_are_canonical_navier_cfl():
    fx, mv = _blob((0, 0)), _blob((3, -2))
    reg = greedy_registration(fx, mv, reg_iterations=[6, 4], initial_transform=False, device="cpu")
    prov = reg["provenance"]
    assert prov["regularizer"] == "navier" and prov["optimizer"] == "cfl"
    assert prov["learning_rate"] == 1.5 and prov["flow_sigma"] == 1.62 and prov["total_sigma"] == 0.0
    assert prov["sobolev_alpha"] == 2.625 and prov["cfl_momentum"] == 0.1875
    assert _corr(reg["warpedmovout"], fx) > _corr(mv, fx)


def test_continuum_kwargs_are_forwarded():
    fx, mv = _blob((0, 0)), _blob((3, -2))
    kw = dict(reg_iterations=[6, 4], initial_transform=False, device="cpu", digital_diffeomorphism=False)
    a = greedy_registration(fx, mv, **kw)["model"].warp
    b = greedy_registration(fx, mv, poisson_ratio=0.45, **kw)
    assert b["provenance"]["regularizer_kwargs"] == {"poisson_ratio": 0.45}
    assert not np.array_equal(a.numpy(), b["model"].warp.numpy())


def test_fit_requires_physical_geometry():
    import torch
    model = GreedyRegistrationModel(dim=3)
    x = torch.rand(1, 1, 16, 16, 16)
    theta = torch.eye(3, 4)[None]
    with pytest.raises(ValueError, match="required"):
        model.fit(x, x, theta, scales=[1], iterations=[1])


def test_conv1x1_affine_grid_matches_border_sampled_affine_grid():
    import torch
    import torch.nn.functional as F
    from antstorch.greedy import _Conv1x1AffineGrid, _sample_field_cf
    torch.manual_seed(0)
    theta = torch.eye(3, 4)[None] + 0.05 * torch.randn(1, 3, 4)
    shape = [1, 1, 9, 10, 11]
    id_grid = F.affine_grid(torch.eye(3, 4)[None], shape, align_corners=True)
    coords = (id_grid + 0.3 * torch.randn_like(id_grid)).requires_grad_(True)
    a = _Conv1x1AffineGrid.apply(coords, theta[0, :, :3].contiguous(), theta[0, :, 3].contiguous())
    ref = _sample_field_cf(torch.movedim(F.affine_grid(theta, shape, align_corners=True), -1, 1), coords)
    torch.testing.assert_close(a, ref, rtol=1e-5, atol=1e-5)
    a.sum().backward()
    outside = (coords.detach().abs() > 1.0)
    assert float(coords.grad[outside].abs().max()) == 0.0
