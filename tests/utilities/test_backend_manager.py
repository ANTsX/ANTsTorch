import sys

import pytest

import antstorch
from antstorch.utilities import backend_manager as bm
from antstorch.utilities import backend_ops


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ANTSTORCH_BACKEND", raising=False)
    bm.set_default_backend(None)
    yield
    bm.set_default_backend(None)


def test_default_is_antstorch():
    assert bm.get_default_backend() == {
        "n4": "antstorch", "denoise": "antstorch", "registration": "antstorch", "kelly_kapowski": "antstorch"
    }


def test_resolution_order(monkeypatch):
    monkeypatch.setenv("ANTSTORCH_BACKEND", "n4=ants,denoise=ants")
    assert bm.resolve_backend(None, "n4") == "ants"
    assert bm.resolve_backend(None, "registration") == "antstorch"
    antstorch.set_default_backend({"registration": "ants"})
    assert bm.resolve_backend(None, "registration") == "ants"
    assert bm.resolve_backend("antstorch", "n4") == "antstorch"
    assert bm.resolve_backend({"n4": "antstorch"}, "denoise") == "ants"


@pytest.mark.parametrize("bad", ["gpu", {"foo": "ants"}, 3])
def test_invalid_values_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        bm.resolve_backend(bad, "n4")


def test_unmapped_registration_type_falls_back_to_ants(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_ops.ants, "registration", lambda **kw: calls.append(kw) or {"ok": 1})
    with pytest.warns(RuntimeWarning):
        out = backend_ops.run_registration(
            "fixed", "moving", "antsRegistrationSyN[s]", backend="antstorch"
        )
    assert out == {"ok": 1} and calls[0]["type_of_transform"] == "antsRegistrationSyN[s]"


@pytest.mark.parametrize("name, expected", [
    ("antsRegistrationSyNQuickRepro[a]", "Affine"),
    ("antsRegistrationSyNQuickRepro[s]", "SyN"),
    ("antsRegistrationSyNQuickRepro[t]", "Translation"),
    ("antsRegistrationSyNQuick[r]", "Rigid"),
])
def test_quick_presets_map_to_syn_registration_types(name, expected):
    assert backend_ops._SYN_TRANSFORM_MAP[name] == expected


def test_kelly_kapowski_ants_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_ops.ants, "kelly_kapowski", lambda **kw: calls.append(kw) or "thickness")
    out = backend_ops.run_kelly_kapowski("seg", "gm", "wm", backend="ants", verbose=True)
    assert out == "thickness"
    assert calls[0] == dict(s="seg", g="gm", w="wm", its=45, r=0.025, m=1.5, x=0, verbose=1)


def test_kelly_kapowski_ants_thickness_prior(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_ops.ants, "kelly_kapowski", lambda **kw: calls.append(kw) or "thickness")
    backend_ops.run_kelly_kapowski("seg", "gm", "wm", gradient_step=0.0025, thickness_prior=10, backend="ants")
    assert calls[0]["t"] == 10 and calls[0]["r"] == 0.0025


class _Img:
    def clone(self, pixeltype):
        return self


@pytest.mark.parametrize("name, engine", [
    ("antsRegistrationSyNQuickRepro[a]", "robust_affine"),
    ("Affine", "robust_affine"),
    ("antsRegistrationSyNQuickRepro[s]", "greedy_registration"),
    ("SyN", "greedy_registration"),
    ("Similarity", "syn_registration"),
    ("antsRegistrationSyNQuick[r]", "syn_registration"),
    ("antsRegistrationSyNQuickRepro[t]", "syn_registration"),
])
def test_antstorch_registration_engine(monkeypatch, name, engine):
    called = []

    def fake(label):
        return lambda *a, **k: called.append((label, k)) or {"engine": label}

    # `antstorch.syn.robust_affine` is shadowed by the function of the same name,
    # so patch the modules through sys.modules.
    monkeypatch.setattr(sys.modules["antstorch.syn.robust_affine"], "robust_affine", fake("robust_affine"))
    monkeypatch.setattr(sys.modules["antstorch.greedy"], "greedy_registration", fake("greedy_registration"))
    monkeypatch.setattr(sys.modules["antstorch.syn.syn"], "syn_registration", fake("syn_registration"))
    monkeypatch.setattr(antstorch.syn, "syn_registration", fake("syn_registration"))

    out = backend_ops.run_registration(_Img(), _Img(), name, backend="antstorch", device="cpu")
    assert out == {"engine": engine}
    assert called[0][1]["device"] in ("cpu",) or str(called[0][1]["device"]) == "cpu"
