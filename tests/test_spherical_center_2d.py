"""2D tool: recon-cohort-template --mean-method spherical-centered with --spherical-center
{mean,origin}. The command is shared with the 3D tool; this checks it through GlowTool2D.

Subjects are 3D NIfTI volumes from which one slice is read (the normal 2D input). A stub
encoder splits the slice into two 'levels' (upper and lower half) and the decoder joins them."""
import ants
import numpy as np
import pytest
import torch

import antstorch.lamnr_flows.core.lamnr_glow_tool_base as base
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import _spherical_centred_mean
from antstorch.lamnr_flows.tools.lamnr_glow_tool_2d import GlowTool2D

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

H, W, NZ = 8, 8, 6
SLICE = 3
D_LEVEL = (H // 2) * W      # 32 values per level
F64 = torch.float64


class _StubModel:
    def inverse_and_log_det(self, x):
        return [x[..., : H // 2, :].clone(), x[..., H // 2:, :].clone()], torch.zeros(1)


def _stub_gaussian(seed=4):
    rng = np.random.default_rng(seed)
    mus, sigmas = [], []
    for _ in range(2):
        U = rng.normal(size=(D_LEVEL, 3))
        psi = rng.uniform(0.05, 0.3, D_LEVEL)
        mus.append(rng.uniform(0.2, 0.8, D_LEVEL))
        sigmas.append({"type": "lowrank", "U": U, "eig": np.array([4.0, 2.0, 1.0]),
                       "sigma2": 0.1, "psi": psi})
    return {"mu": mus, "Sigma": sigmas, "dims_per_level_per_view": [[D_LEVEL, D_LEVEL]],
            "typical_radii_by_view": [[3.0, 2.0]]}


def _expected_input(volume):
    a = volume[:, :, SLICE]
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    volumes, firsts = [], []
    for k in range(5):
        d = tmp_path / f"sub{k}"
        d.mkdir()
        v = (rng.random((H, W, NZ)) * (1.0 + k)).astype(np.float32)
        ants.image_write(ants.from_numpy(v, spacing=(2.0, 2.0, 2.0)), str(d / "vent.nii.gz"))
        volumes.append(v)
        firsts.append(d / "vent.nii.gz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("vent\n" + "\n".join(str(p) for p in firsts) + "\n")
    ckpt = tmp_path / "training_state.pt"
    torch.save({"config": {"intensity": "to01", "H": H, "W": W}}, ckpt)

    gauss = _stub_gaussian()
    captured = {}
    monkeypatch.setattr(GlowTool2D, "build_model", lambda self, cfg, device, size: _StubModel())
    monkeypatch.setattr(GlowTool2D, "prime_if_needed", lambda self, *a, **k: None)
    monkeypatch.setattr(base, "load_weights_into_model", lambda *a, **k: (True, "stub"))
    monkeypatch.setattr(base, "_load_gaussian_model", lambda path: gauss)
    monkeypatch.setattr(base, "_validate_gauss_blob", lambda blob: (["vent"], None, None, None))

    def decode(self, model, z_list, size):
        captured["z"] = [z.detach().cpu().double().reshape(-1) for z in z_list]
        return torch.cat(list(z_list), dim=-2)
    monkeypatch.setattr(GlowTool2D, "decode_latents", decode)
    return dict(volumes=volumes, manifest=manifest, ckpt=ckpt, gauss=gauss,
                captured=captured, root=tmp_path)


def _run(env, method, extra=()):
    argv = ["--ckpt", str(env["ckpt"]), "--manifest", str(env["manifest"]), "--views", "vent",
            "--view-index", "0", "--devices", "cpu", "--slice-axis", "2",
            "--slice-index", str(SLICE), "--mean-method", method,
            "--out-range-mode", "clamp", "--out", str(env["root"] / "out" / "t.nii.gz"), *extra]
    GlowTool2D().cmd_recon_cohort_template(argv)


def _rows(env, level):
    out = []
    for v in env["volumes"]:
        a = _expected_input(v)
        half = a[: H // 2] if level == 0 else a[H // 2:]
        out.append(half.reshape(-1).astype(np.float64))
    return torch.tensor(np.stack(out), dtype=F64)


def test_centred_sphere_uses_the_stored_radius(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz"])
    for l, (z, radius) in enumerate(zip(stubbed["captured"]["z"], (3.0, 2.0))):
        mu = np.asarray(stubbed["gauss"]["mu"][l])
        assert np.linalg.norm(z.numpy() - mu) == pytest.approx(radius, rel=1e-4)


def test_origin_uses_the_median_latent_norm_and_no_centre(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-center", "origin"])
    for l in range(2):
        rows = _rows(stubbed, l)
        expected, _ = _spherical_centred_mean(rows, torch.zeros(rows.shape[1], dtype=F64), None,
                                              1.0, radius_source="rows")
        assert np.allclose(stubbed["captured"]["z"][l].numpy(), expected.numpy(), atol=1e-4)


def test_origin_scale_shrinks_the_sphere(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-center", "origin"])
    full = [z.clone() for z in stubbed["captured"]["z"]]
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-center", "origin",
                                         "--spherical-radius-scale", "0.5"])
    for a, b in zip(full, stubbed["captured"]["z"]):
        assert np.allclose(b.numpy(), 0.5 * a.numpy(), atol=1e-4)


def test_origin_differs_from_the_centred_mean(stubbed):
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz"])
    centred = [z.clone() for z in stubbed["captured"]["z"]]
    _run(stubbed, "spherical-centered", ["--gauss", "stub.npz", "--spherical-center", "origin"])
    assert any(not np.allclose(a.numpy(), b.numpy()) for a, b in zip(centred, stubbed["captured"]["z"]))


@pytest.mark.parametrize("method", ["euclidean", "image", "spherical-mahalanobis"])
def test_origin_is_refused_for_other_methods(stubbed, method):
    with pytest.raises(ValueError, match="spherical-centered"):
        _run(stubbed, method, ["--gauss", "stub.npz", "--spherical-center", "origin"])
