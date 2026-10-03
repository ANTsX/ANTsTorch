# tests/test_factorized_glow3d.py
"""(2+1)D variant of create_glow_normalizing_flow_model_3d (squeeze_time / kernel_size / temporal_init).

The default arguments must reproduce the historical model exactly.
"""
import pytest
import torch

from antstorch.lamnr_flows.architectures import create_glow_normalizing_flow_model_3d as make3d

FACTORIZED = ((1, 3, 3), (3, 1, 1), (1, 3, 3))
SHAPE = (2, 8, 16, 16)            # (C, T, H, W): time is the first spatial axis


def _build(seed=0, perturb=0.05, **kw):
    torch.manual_seed(seed)
    args = dict(L=2, K=[2, 1], hidden_channels=[8, 6], net_actnorm=False, scale_cap=3.0, actnorm_scale_cap=5.0)
    args.update(kw)
    shape = args.pop("input_shape", SHAPE)
    model = make3d(shape, **args)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(perturb * torch.randn(p.shape, generator=g))
    return model


def _x(batch=2, shape=SHAPE, seed=7):
    return torch.rand(batch, *shape, generator=torch.Generator().manual_seed(seed))


def _roundtrip(model, x, tol=1e-9):
    model = model.double().eval()
    x = x.double()
    with torch.no_grad():
        model.log_prob(x)                                     # data-dependent initializations, if any
        z, ld_inv = model.inverse_and_log_det(x)
        x_rec, ld_fwd = model.forward_and_log_det(z)
    assert float((x_rec - x).abs().max()) < tol
    assert float((ld_inv + ld_fwd).abs().max()) < tol
    return z


# ----------------------------------------------------------------- defaults unchanged
def test_defaults_are_the_historical_model():
    a, b = _build(), _build(squeeze_time=True, kernel_size=None, temporal_init="default")
    assert [(n, tuple(p.shape)) for n, p in a.named_parameters()] == [(n, tuple(p.shape)) for n, p in b.named_parameters()]
    assert sum(p.numel() for p in a.parameters()) == 63328          # recorded from the code before this change
    x = _x()
    with torch.no_grad():
        assert torch.equal(a.eval().log_prob(x), b.eval().log_prob(x))
        z, _ = a.inverse_and_log_det(x)
    assert [tuple(t.shape) for t in z] == [(2, 64, 2, 4, 4), (2, 8, 4, 8, 8)]


def test_per_level_all_true_equals_scalar_true():
    a, b = _build(squeeze_time=True), _build(squeeze_time=[True, True])
    x = _x()
    with torch.no_grad():
        assert torch.equal(a.eval().log_prob(x), b.eval().log_prob(x))


# ----------------------------------------------------------------- factorized plans
@pytest.mark.parametrize("squeeze_time, expected", [
    (False,          [(16, 8, 4, 4), (4, 8, 8, 8)]),
    ([False, False], [(16, 8, 4, 4), (4, 8, 8, 8)]),
    ([True, False],  [(32, 4, 4, 4), (4, 8, 8, 8)]),   # index 0 = deepest level
    ([False, True],  [(32, 4, 4, 4), (8, 4, 8, 8)]),
])
def test_latent_shapes_and_total_dimension(squeeze_time, expected):
    model = _build(squeeze_time=squeeze_time)
    z = _roundtrip(model, _x())
    got = [tuple(t.shape[1:]) for t in z]
    assert got == expected, got
    assert sum(int(torch.tensor(s).prod()) for s in got) == int(torch.tensor(SHAPE).prod())


@pytest.mark.parametrize("kw", [
    dict(squeeze_time=False, kernel_size=FACTORIZED),
    dict(squeeze_time=False, kernel_size=FACTORIZED, temporal_init="identity"),
    dict(squeeze_time=[True, False], kernel_size=FACTORIZED),
    dict(squeeze_time=True, kernel_size=FACTORIZED),
    dict(squeeze_time=False),
    dict(squeeze_time=False, kernel_size=[[1, 3, 3], [3, 1, 1], [1, 3, 3]]),      # JSON-style lists
])
def test_roundtrip_and_logdet(kw):
    _roundtrip(_build(**kw), _x())


def test_log_prob_is_finite_and_trains():
    model = _build(squeeze_time=False, kernel_size=FACTORIZED, temporal_init="identity", perturb=0.0)
    x = _x()
    lp = model.log_prob(x)
    assert torch.isfinite(lp).all()
    (-lp.mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_time_is_not_squeezed_when_flag_is_false():
    """With squeeze_time=False and purely spatial kernels, latent coefficients at time t depend on frame t only."""
    model = _build(squeeze_time=False, kernel_size=((1, 3, 3), (1, 1, 1), (1, 3, 3)), perturb=0.2).double().eval()
    x = _x().double()
    x2 = x.clone()
    x2[:, :, 3] += 0.3
    with torch.no_grad():
        model.log_prob(x)
        z1, _ = model.inverse_and_log_det(x)
        z2, _ = model.inverse_and_log_det(x2)
    for a, b in zip(z1, z2):
        diff = (a - b).abs().amax(dim=(0, 1, 3, 4))            # (T,) -- time axis stays axis 2
        assert diff[3] > 0
        assert torch.all(diff[[0, 1, 2, 4, 5, 6, 7]] == 0)


def test_temporal_kernel_couples_frames():
    model = _build(squeeze_time=False, kernel_size=FACTORIZED, perturb=0.2).double().eval()
    x = _x().double()
    x2 = x.clone()
    x2[:, :, 3] += 0.3
    with torch.no_grad():
        model.log_prob(x)
        z1, _ = model.inverse_and_log_det(x)
        z2, _ = model.inverse_and_log_det(x2)
    spill = sum(float((a - b).abs().amax(dim=(0, 1, 3, 4))[[2, 4]].sum()) for a, b in zip(z1, z2))
    assert spill > 0


# ----------------------------------------------------------------- argument validation
def test_divisibility_and_argument_errors():
    with pytest.raises(ValueError):                              # historical rule: every axis divisible by 2**L
        make3d((2, 6, 16, 16), L=2, K=1, hidden_channels=4)
    model = make3d((2, 6, 16, 16), L=2, K=1, hidden_channels=4, squeeze_time=False)   # T no longer squeezed
    model.log_prob(_x(shape=(2, 6, 16, 16)))
    with pytest.raises(ValueError):                              # one squeezing level -> T divisible by 2
        make3d((2, 3, 16, 16), L=2, K=1, hidden_channels=4, squeeze_time=[True, False])
    with pytest.raises(ValueError):
        make3d(SHAPE, L=2, K=1, hidden_channels=4, squeeze_time=[True, False, True])
    with pytest.raises(ValueError):
        make3d(SHAPE, L=2, K=1, hidden_channels=4, kernel_size=((1, 3, 3), (3, 1, 1)))
    with pytest.raises(ValueError):
        make3d(SHAPE, L=2, K=1, hidden_channels=4, temporal_init="dirac")


def test_temporal_identity_is_applied_to_every_block():
    model = _build(squeeze_time=False, kernel_size=FACTORIZED, temporal_init="identity", perturb=0.0)
    mids = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d) and tuple(m.kernel_size) == (3, 1, 1)]
    assert len(mids) == 3                                          # K=[2, 1] -> 3 coupling networks
    for m in mids:
        w = m.weight.detach()
        assert float(w.sum()) == m.out_channels                    # one unit tap per output channel
        assert torch.equal(w[:, :, 1, 0, 0], torch.eye(m.out_channels))
        if m.bias is not None:
            assert float(m.bias.detach().abs().sum()) == 0.0
