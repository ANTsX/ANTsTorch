"""--mean-method spherical-mahalanobis of recon-cohort-template and its building blocks:
the low-rank whitener, the Frechet mean on the unit sphere and the centred spherical mean."""
import ants
import numpy as np
import pytest
import torch

import antstorch.lamnr_flows.core.lamnr_glow_tool_base as base
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (
    _LowRankWhitener, _frechet_mean_unit_sphere, _spherical_centred_mean)
from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D

# numpy's matmul on macOS (Accelerate) raises spurious FP warnings on finite data; the
# assertions below check the values, so these three messages are not informative.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

F64 = torch.float64


def _dense_sigma(U, eig, psi):
    return np.diag(psi) + U @ np.diag(eig) @ U.T


def _lowrank(d=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(d, k))                 # deliberately not orthonormal
    eig = np.array([5.0, 2.0, 0.5])[:k]
    psi = rng.uniform(0.2, 1.0, d)
    return U, eig, psi


# --------------------------------------------------------------------------- #
# _LowRankWhitener
# --------------------------------------------------------------------------- #

class TestWhitener:
    def test_whitening_gives_identity_covariance(self):
        U, eig, psi = _lowrank()
        wh = _LowRankWhitener(U, eig, psi)
        W = wh.whiten(torch.eye(len(psi), dtype=F64)).T.numpy()
        assert np.allclose(W @ _dense_sigma(U, eig, psi) @ W.T, np.eye(len(psi)), atol=1e-9)

    def test_colouring_inverts_whitening_and_has_covariance_sigma(self):
        U, eig, psi = _lowrank()
        wh = _LowRankWhitener(U, eig, psi)
        eye = torch.eye(len(psi), dtype=F64)
        W, C = wh.whiten(eye).T.numpy(), wh.colour(eye).T.numpy()
        assert np.allclose(C @ W, np.eye(len(psi)), atol=1e-9)
        assert np.allclose(C @ C.T, _dense_sigma(U, eig, psi), atol=1e-9)

    def test_squared_norm_is_the_mahalanobis_distance(self):
        U, eig, psi = _lowrank(seed=1)
        wh = _LowRankWhitener(U, eig, psi)
        c = torch.randn(5, len(psi), dtype=F64, generator=torch.Generator().manual_seed(0))
        expected = np.einsum("ni,ij,nj->n", c.numpy(), np.linalg.inv(_dense_sigma(U, eig, psi)), c.numpy())
        assert np.allclose(wh.whiten(c).pow(2).sum(1).numpy(), expected, atol=1e-8)

    def test_single_vector_input(self):
        U, eig, psi = _lowrank()
        wh = _LowRankWhitener(U, eig, psi)
        c = torch.randn(len(psi), dtype=F64, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(wh.whiten(c), wh.whiten(c[None])[0])
        assert torch.allclose(wh.colour(wh.whiten(c)), c)

    def test_isotropic_noise_from_sigma2(self):
        U, eig, _ = _lowrank()
        Sigma_view = {"type": "lowrank", "U": U, "eig": eig, "sigma2": 0.7}
        wh = _LowRankWhitener.from_covariance(Sigma_view)
        W = wh.whiten(torch.eye(U.shape[0], dtype=F64)).T.numpy()
        assert np.allclose(W @ _dense_sigma(U, eig, np.full(U.shape[0], 0.7)) @ W.T,
                           np.eye(U.shape[0]), atol=1e-9)

    def test_dense_covariance_is_refused(self):
        with pytest.raises(ValueError, match="low-rank"):
            _LowRankWhitener.from_covariance(np.eye(4))


# --------------------------------------------------------------------------- #
# _frechet_mean_unit_sphere
# --------------------------------------------------------------------------- #

class TestFrechet:
    def test_symmetric_points_give_the_axis(self):
        t = 0.3
        pts = torch.tensor([[np.cos(t), np.sin(t), 0.0], [np.cos(t), -np.sin(t), 0.0]], dtype=F64)
        direction, converged, _, tangent, objective, resultant = _frechet_mean_unit_sphere(pts)
        assert converged and tangent < 1e-7
        assert torch.allclose(direction, torch.tensor([1.0, 0.0, 0.0], dtype=F64), atol=1e-6)
        assert objective == pytest.approx(t * t, rel=1e-6)
        assert resultant == pytest.approx(np.cos(t), rel=1e-9)

    def test_identical_points(self):
        pts = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=F64)
        direction, converged, iterations, _, objective, _ = _frechet_mean_unit_sphere(pts)
        assert converged and iterations == 1 and objective == pytest.approx(0.0, abs=1e-12)
        assert torch.allclose(direction, pts[0])

    def test_result_is_a_unit_vector_even_for_antipodal_points(self):
        pts = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=F64)
        direction = _frechet_mean_unit_sphere(pts)[0]
        assert direction.norm().item() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# _spherical_centred_mean
# --------------------------------------------------------------------------- #

def _cohort(d=12, n=6, seed=2):
    U, eig, psi = _lowrank(d=d, seed=seed)
    Sigma_view = {"type": "lowrank", "U": U, "eig": eig, "sigma2": 0.5, "psi": psi}
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=d)
    flat = centre + rng.normal(size=(n, d)) @ np.linalg.cholesky(_dense_sigma(U, eig, psi)).T
    return torch.from_numpy(flat), torch.from_numpy(centre), Sigma_view, (U, eig, psi)


class TestSphericalCentredMean:
    def test_euclidean_radius_is_the_given_typical_radius(self):
        flat, centre, Sigma_view, _ = _cohort()
        mean, info = _spherical_centred_mean(flat, centre, Sigma_view, 7.5, metric="euclidean")
        assert (mean - centre).norm().item() == pytest.approx(7.5, rel=1e-9)
        assert info["radius"] == 7.5 and info["metric"] == "euclidean"

    def test_mahalanobis_radius_is_the_median_mahalanobis_norm(self):
        flat, centre, Sigma_view, (U, eig, psi) = _cohort()
        Sinv = np.linalg.inv(_dense_sigma(U, eig, psi))
        c = (flat - centre).numpy()
        norms = np.sqrt(np.einsum("ni,ij,nj->n", c, Sinv, c))
        mean, info = _spherical_centred_mean(flat, centre, Sigma_view, 123.0, metric="mahalanobis")
        m = (mean - centre).numpy()
        assert np.sqrt(m @ Sinv @ m) == pytest.approx(np.median(norms), rel=1e-8)
        assert info["radius"] == pytest.approx(np.median(norms), rel=1e-9)

    def test_identity_covariance_makes_both_metrics_agree(self):
        d = 10
        Sigma_view = {"type": "lowrank", "U": np.zeros((d, 2)), "eig": np.zeros(2),
                      "sigma2": 1.0, "psi": np.ones(d)}
        rng = np.random.default_rng(3)
        centre = torch.from_numpy(rng.normal(size=d))
        flat = centre + torch.from_numpy(rng.normal(size=(5, d)))
        radius = float(torch.quantile((flat - centre).norm(dim=1), 0.5))
        a, _ = _spherical_centred_mean(flat, centre, Sigma_view, radius, metric="euclidean")
        b, _ = _spherical_centred_mean(flat, centre, Sigma_view, radius, metric="mahalanobis")
        assert torch.allclose(a, b, atol=1e-9)

    def test_latent_equal_to_the_centre_is_refused(self):
        flat, centre, Sigma_view, _ = _cohort()
        flat[0] = centre
        with pytest.raises(ValueError, match="cannot be projected"):
            _spherical_centred_mean(flat, centre, Sigma_view, 5.0)

    def test_unknown_metric(self):
        flat, centre, Sigma_view, _ = _cohort()
        with pytest.raises(ValueError, match="metric"):
            _spherical_centred_mean(flat, centre, Sigma_view, 5.0, metric="cosine")


# --------------------------------------------------------------------------- #
# recon-cohort-template (model, weights and Gaussian file stubbed)
# --------------------------------------------------------------------------- #

SHAPE = (4, 4, 4)
D_LEVEL = 32   # each stub level holds half of the volume: (1, 1, 2, 4, 4)


class _StubModel:
    """Encoder with two 'levels': the two halves of the volume along the first axis."""
    def inverse_and_log_det(self, x):
        return [x[:, :, :2].clone(), x[:, :, 2:].clone()], torch.zeros(1)


def _stub_gaussian(seed=4):
    rng = np.random.default_rng(seed)
    mus, sigmas, params = [], [], []
    for _ in range(2):
        U = rng.normal(size=(D_LEVEL, 3))
        eig = np.array([4.0, 2.0, 1.0])
        psi = rng.uniform(0.05, 0.3, D_LEVEL)
        mus.append(rng.uniform(0.2, 0.8, D_LEVEL))
        sigmas.append({"type": "lowrank", "U": U, "eig": eig, "sigma2": 0.1, "psi": psi})
        params.append((U, eig, psi))
    blob = {"mu": mus, "Sigma": sigmas,
            "dims_per_level_per_view": [[D_LEVEL, D_LEVEL]],
            "typical_radii_by_view": [[3.0, 2.0]]}
    return blob, params


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    images, firsts = [], []
    for k in range(5):
        a = rng.random(SHAPE).astype(np.float32) * (1 + k)
        d = tmp_path / f"sub{k}"
        d.mkdir()
        ants.image_write(ants.from_numpy(a, spacing=(2.0, 2.0, 2.0)), str(d / "vent.nii.gz"))
        images.append(a)
        firsts.append(d / "vent.nii.gz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("vent\n" + "\n".join(str(p) for p in firsts) + "\n")
    ckpt = tmp_path / "training_state.pt"
    torch.save({"config": {"intensity": "to01"}}, ckpt)

    gauss, params = _stub_gaussian()
    captured = {}
    monkeypatch.setattr(GlowTool3D, "build_model", lambda self, cfg, device, size: _StubModel())
    monkeypatch.setattr(GlowTool3D, "prime_if_needed", lambda self, *a, **k: None)
    monkeypatch.setattr(base, "load_weights_into_model", lambda *a, **k: (True, "stub"))
    monkeypatch.setattr(base, "_load_gaussian_model", lambda path: gauss)
    monkeypatch.setattr(base, "_validate_gauss_blob", lambda blob: (["vent"], None, None, None))

    def decode(self, model, z_list, size):
        captured["z"] = [z.detach().cpu().double().reshape(-1) for z in z_list]
        return torch.cat(list(z_list), dim=2).clamp(0, 1)
    monkeypatch.setattr(GlowTool3D, "decode_latents", decode)
    return dict(images=images, manifest=manifest, ckpt=ckpt, gauss=gauss, params=params,
                captured=captured, root=tmp_path)


def _encoded(image):
    """Latents of one subject as the stub encoder gives them after read_image's min-max."""
    a = (image - image.min()) / (image.max() - image.min() + 1e-8)
    a = np.clip(a, 1e-5, 1 - 1e-5)
    return [a[:2].reshape(-1).astype(np.float64), a[2:].reshape(-1).astype(np.float64)]


def _run(env, method, extra=()):
    argv = ["--ckpt", str(env["ckpt"]), "--manifest", str(env["manifest"]), "--views", "vent",
            "--view-index", "0", "--devices", "cpu", "--spatial-dims", *map(str, SHAPE),
            "--mean-method", method, "--out", str(env["root"] / "out" / "t.nii.gz"), *extra]
    GlowTool3D().cmd_recon_cohort_template(argv)


@pytest.mark.parametrize("method", ["spherical-centered", "spherical-mahalanobis"])
def test_gauss_is_required(stubbed, method):
    with pytest.raises(ValueError, match="requires --gauss"):
        _run(stubbed, method)


def test_euclidean_sphere_uses_the_stored_typical_radius(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz"])
    for l, (z, radius) in enumerate(zip(stubbed["captured"]["z"], (3.0, 2.0))):
        mu = np.asarray(stubbed["gauss"]["mu"][l])
        assert np.linalg.norm(z.numpy() - mu) == pytest.approx(radius, rel=1e-5)


def test_mahalanobis_sphere_uses_the_cohort_median_mahalanobis_norm(stubbed):
    _run(stubbed, "spherical-mahalanobis", ["--gauss", "stub.npz"])
    encoded = [_encoded(a) for a in stubbed["images"]]
    for l in range(2):
        U, eig, psi = stubbed["params"][l]
        Sinv = np.linalg.inv(_dense_sigma(U, eig, psi))
        mu = np.asarray(stubbed["gauss"]["mu"][l])
        norms = [np.sqrt((e[l] - mu) @ Sinv @ (e[l] - mu)) for e in encoded]
        got = stubbed["captured"]["z"][l].numpy() - mu
        assert np.sqrt(got @ Sinv @ got) == pytest.approx(np.median(norms), rel=1e-4)


def test_the_two_spherical_methods_differ(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz"])
    euclid = [z.clone() for z in stubbed["captured"]["z"]]
    _run(stubbed, "spherical-mahalanobis", ["--gauss", "stub.npz"])
    assert not torch.allclose(euclid[0], stubbed["captured"]["z"][0], atol=1e-3)


def test_dense_gaussian_is_refused_by_the_mahalanobis_method(stubbed):
    stubbed["gauss"]["Sigma"] = [np.eye(D_LEVEL), np.eye(D_LEVEL)]
    with pytest.raises(ValueError, match="low-rank"):
        _run(stubbed, "spherical-mahalanobis", ["--gauss", "stub.npz"])


def test_output_is_written(stubbed):
    _run(stubbed, "spherical-mahalanobis", ["--gauss", "stub.npz"])
    out = ants.image_read(str(stubbed["root"] / "out" / "t.nii.gz"))
    assert out.shape == SHAPE and np.isfinite(out.numpy()).all()


# --------------------------------------------------------------------------- #
# --spherical-radius-scale
# --------------------------------------------------------------------------- #

class TestRadiusScale:
    def test_euclidean_radius_is_scaled_and_direction_kept(self):
        flat, centre, Sigma_view, _ = _cohort()
        full, _ = _spherical_centred_mean(flat, centre, Sigma_view, 7.5)
        part, info = _spherical_centred_mean(flat, centre, Sigma_view, 7.5, radius_scale=0.8)
        assert (part - centre).norm().item() == pytest.approx(0.8 * 7.5, rel=1e-9)
        assert torch.allclose(part - centre, 0.8 * (full - centre), atol=1e-9)
        assert info["radius"] == pytest.approx(6.0) and info["radius_scale"] == 0.8

    def test_mahalanobis_radius_is_scaled(self):
        flat, centre, Sigma_view, (U, eig, psi) = _cohort()
        Sinv = np.linalg.inv(_dense_sigma(U, eig, psi))
        full, info1 = _spherical_centred_mean(flat, centre, Sigma_view, 1.0, metric="mahalanobis")
        part, info2 = _spherical_centred_mean(flat, centre, Sigma_view, 1.0, metric="mahalanobis",
                                              radius_scale=0.5)
        m = (part - centre).numpy()
        assert np.sqrt(m @ Sinv @ m) == pytest.approx(0.5 * info1["radius"], rel=1e-8)
        assert torch.allclose(part - centre, 0.5 * (full - centre), atol=1e-8)

    def test_default_is_one(self):
        flat, centre, Sigma_view, _ = _cohort()
        a, _ = _spherical_centred_mean(flat, centre, Sigma_view, 7.5)
        b, _ = _spherical_centred_mean(flat, centre, Sigma_view, 7.5, radius_scale=1.0)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("scale", [0.0, -0.5])
    def test_non_positive_scale_is_refused(self, scale):
        flat, centre, Sigma_view, _ = _cohort()
        with pytest.raises(ValueError, match="radius_scale"):
            _spherical_centred_mean(flat, centre, Sigma_view, 7.5, radius_scale=scale)


def test_cli_scale_shrinks_the_euclidean_sphere(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-radius-scale", "0.5"])
    for l, (z, radius) in enumerate(zip(stubbed["captured"]["z"], (3.0, 2.0))):
        mu = np.asarray(stubbed["gauss"]["mu"][l])
        assert np.linalg.norm(z.numpy() - mu) == pytest.approx(0.5 * radius, rel=1e-5)


def test_cli_scale_shrinks_the_mahalanobis_sphere(stubbed):
    _run(stubbed, "spherical-mahalanobis", ["--gauss", "stub.npz", "--spherical-radius-scale", "0.5"])
    encoded = [_encoded(a) for a in stubbed["images"]]
    for l in range(2):
        U, eig, psi = stubbed["params"][l]
        Sinv = np.linalg.inv(_dense_sigma(U, eig, psi))
        mu = np.asarray(stubbed["gauss"]["mu"][l])
        norms = [np.sqrt((e[l] - mu) @ Sinv @ (e[l] - mu)) for e in encoded]
        got = stubbed["captured"]["z"][l].numpy() - mu
        assert np.sqrt(got @ Sinv @ got) == pytest.approx(0.5 * np.median(norms), rel=1e-4)


@pytest.mark.parametrize("method", ["euclidean", "image"])
def test_cli_scale_is_refused_for_non_spherical_methods(stubbed, method):
    with pytest.raises(ValueError, match="only applies"):
        _run(stubbed, method, ["--spherical-radius-scale", "0.8"])


@pytest.mark.parametrize("scale", ["0", "-1"])
def test_cli_non_positive_scale_is_refused(stubbed, scale):
    with pytest.raises(ValueError, match="must be positive"):
        _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-radius-scale", scale])
