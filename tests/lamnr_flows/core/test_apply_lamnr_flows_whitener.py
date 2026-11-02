import antsnormflows as nf
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from antstorch.lamnr_flows.apply_lamnr_flows_whitener import (
    _extract_whitened_from_z,
    _extract_whitened_from_z_full,
    _z_from_whitened,
    _z_from_whitened_full,
    apply_lamnr_flows_whitener,
)
from antstorch.lamnr_flows.lamnr_flows_whitener import (
    _extract_whitened_from_z as trainer_extract_whitened_from_z,
)
from antstorch.lamnr_flows.lamnr_flows_whitener import (
    lamnr_flows_whitener,
)
from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import (
    _extract_whitened as tabular_extract_whitened,
)


class _AffineBijectionModel(nn.Module):
    """Exact invertible linear flow wrapper around a base distribution q0."""

    def __init__(self, q0: nn.Module, weight: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.q0 = q0
        self.register_buffer("weight", weight.clone())
        self.register_buffer("weight_inv", torch.linalg.inv(weight))
        self.register_buffer("shift", shift.clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.weight.T + self.shift

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) @ self.weight_inv.T


def _make_pca_model(dim: int = 5, latent_dim: int = 3, sigma: float = 0.25, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    q0 = nf.distributions.GaussianPCA(dim=dim, latent_dim=latent_dim, sigma=sigma)
    with torch.no_grad():
        q0.W.copy_(torch.randn(latent_dim, dim, generator=gen))
        q0.loc.copy_(torch.randn(1, dim, generator=gen))
    weight = torch.eye(dim, dtype=torch.float64) + 0.2 * torch.randn(
        dim, dim, generator=gen, dtype=torch.float64
    )
    shift = torch.randn(1, dim, generator=gen, dtype=torch.float64)
    return _AffineBijectionModel(q0, weight, shift)


def _make_diag_model(dim: int = 4, seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    q0 = nf.distributions.DiagGaussian(dim, trainable=True, min_log=-2.0, max_log=2.0)
    with torch.no_grad():
        q0.loc.copy_(torch.randn(1, dim, generator=gen))
        # Include values outside [-2, 2] to verify min_log / max_log clamping matches DiagGaussian
        q0.log_scale.copy_(torch.tensor([[-3.5, -0.4, 0.7, 3.2]])[:, :dim])
    weight = torch.eye(dim, dtype=torch.float64) + 0.15 * torch.randn(
        dim, dim, generator=gen, dtype=torch.float64
    )
    shift = torch.randn(1, dim, generator=gen, dtype=torch.float64)
    return _AffineBijectionModel(q0, weight, shift)


def test_whitened_full_operator_gives_identity_covariance():
    model = _make_pca_model(dim=5, latent_dim=3, sigma=0.2)
    q0 = model.q0
    W = q0.W.detach().double()
    sigma2 = torch.exp(2.0 * q0.log_sigma.detach().double())
    Sigma = W.T @ W + sigma2 * torch.eye(5, dtype=torch.float64)

    # Extract linear operator M (column convention: y_col = M zc_col) via basis rows + loc
    basis = torch.eye(5, dtype=torch.float64) + q0.loc.detach().double()
    M = _extract_whitened_from_z_full(model, basis).T  # (D, D)

    whitened_cov = M @ Sigma @ M.T
    assert torch.allclose(whitened_cov, torch.eye(5, dtype=torch.float64), atol=1e-12)


def test_whitened_full_roundtrip_and_monte_carlo_covariance():
    torch.manual_seed(42)
    model = _make_pca_model(dim=5, latent_dim=3, sigma=0.3)
    q0 = model.q0
    W = q0.W.detach().double()
    sigma2 = torch.exp(2.0 * q0.log_sigma.detach().double())
    Sigma = W.T @ W + sigma2 * torch.eye(5, dtype=torch.float64)
    L_chol = torch.linalg.cholesky(Sigma)

    eps_true = torch.randn(25000, 5, dtype=torch.float64)
    z = eps_true @ L_chol.T + q0.loc.detach().double()

    w = _extract_whitened_from_z_full(model, z)
    assert torch.allclose(w, eps_true, atol=1e-12)

    emp_cov = (w.T @ w) / w.shape[0]
    assert torch.allclose(emp_cov, torch.eye(5, dtype=torch.float64), atol=3e-2)

    z_rec = _z_from_whitened_full(model, w)
    assert torch.allclose(z_rec, z, atol=1e-12)


def test_apply_whitener_end_to_end_roundtrip_pca():
    torch.manual_seed(123)
    model = _make_pca_model(dim=4, latent_dim=2, sigma=0.15)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(64, 4)), columns=["a", "b", "c", "d"])
    stats = [
        {
            "mode": "0mean",
            "mean": df.mean(axis=0).to_numpy(),
            "std": df.std(axis=0, ddof=0).to_numpy(),
        }
    ]
    bundle = {"models": [model], "dataset_normalizers": stats}

    w_full = apply_lamnr_flows_whitener(
        bundle, df, direction="forward", output_space="whitened_full"
    )
    assert list(w_full.columns) == [f"whitened_full_{i}" for i in range(4)]

    x_rec = apply_lamnr_flows_whitener(
        bundle, w_full, direction="inverse", input_space="whitened_full"
    )
    np.testing.assert_allclose(x_rec.to_numpy(), df.to_numpy(), atol=1e-11)


def test_extract_whitened_pca_recovers_latent_coords_and_matches_trainers():
    torch.manual_seed(9)
    model = _make_pca_model(dim=6, latent_dim=3, sigma=1e-7)
    eps = torch.randn(32, 3, dtype=torch.float64)
    z = eps @ model.q0.W.detach().double() + model.q0.loc.detach().double()

    w_apply = _extract_whitened_from_z(model, z)
    assert torch.allclose(w_apply, eps, atol=1e-6)

    z_rec = _z_from_whitened(model, w_apply)
    assert torch.allclose(z_rec, z, atol=1e-6)

    z_f32 = z.float().requires_grad_(True)
    w_trainer = trainer_extract_whitened_from_z(model, z_f32)
    w_tabular = tabular_extract_whitened(model, z_f32.detach())
    assert w_trainer.dtype == torch.float32
    assert torch.allclose(w_trainer.double(), w_apply, atol=1e-5)
    assert torch.allclose(w_tabular.double(), w_apply, atol=1e-5)

    w_trainer.sum().backward()
    assert z_f32.grad is not None and torch.isfinite(z_f32.grad).all()


@pytest.mark.parametrize("space", ["whitened", "whitened_full"])
def test_diag_gaussian_log_scale_whitening_and_roundtrip(space: str):
    torch.manual_seed(21)
    model = _make_diag_model(dim=4)
    q0 = model.q0
    assert not hasattr(q0, "scale") and hasattr(q0, "log_scale")

    eps = torch.randn(128, 4, dtype=torch.float64)
    clamped_log_scale = torch.clamp(q0.log_scale.detach().double(), q0.min_log, q0.max_log)
    z = q0.loc.detach().double() + torch.exp(clamped_log_scale) * eps

    w_apply = _extract_whitened_from_z(model, z)
    w_full = _extract_whitened_from_z_full(model, z)
    w_trainer = trainer_extract_whitened_from_z(model, z.float())
    w_tabular = tabular_extract_whitened(model, z.float())

    assert torch.allclose(w_apply, eps, atol=1e-12)
    assert torch.allclose(w_full, eps, atol=1e-12)
    assert torch.allclose(w_trainer.double(), eps, atol=1e-6)
    assert torch.allclose(w_tabular.double(), eps, atol=1e-6)

    df = pd.DataFrame(np.random.default_rng(3).normal(size=(40, 4)), columns=["f0", "f1", "f2", "f3"])
    w_df = apply_lamnr_flows_whitener(
        [model], df, direction="forward", output_space=space, normalization_mode=None
    )
    x_rec = apply_lamnr_flows_whitener(
        [model], w_df, direction="inverse", input_space=space, normalization_mode=None
    )
    np.testing.assert_allclose(x_rec.to_numpy(), df.to_numpy(), atol=1e-11)


def test_custom_scale_attribute_fallback():
    class _CustomDiag(nn.Module):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float64))
            self.scale = nn.Parameter(torch.tensor([[0.5, 4.0]], dtype=torch.float64))

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.q0 = _CustomDiag()

    holder = _Holder()
    eps = torch.tensor([[2.0, -0.5], [-1.0, 1.5]], dtype=torch.float64)
    z = eps * holder.q0.scale + holder.q0.loc
    assert torch.allclose(_extract_whitened_from_z(holder, z), eps, atol=1e-12)
    assert torch.allclose(_z_from_whitened(holder, eps), z, atol=1e-12)
    assert torch.allclose(_extract_whitened_from_z_full(holder, z), eps, atol=1e-12)
    assert torch.allclose(_z_from_whitened_full(holder, eps), z, atol=1e-12)


@pytest.mark.parametrize("base_distribution", ["GaussianPCA", "DiagGaussian"])
def test_lamnr_flows_whitener_to_apply_pipeline_roundtrip(base_distribution: str):
    rng = np.random.default_rng(17)
    df0 = pd.DataFrame(rng.normal(size=(48, 4)), columns=["v0_0", "v0_1", "v0_2", "v0_3"])
    df1 = pd.DataFrame(rng.normal(size=(48, 4)), columns=["v1_0", "v1_1", "v1_2", "v1_3"])

    out = lamnr_flows_whitener(
        [df0, df1],
        batch_size=16,
        val_batch_size=32,
        cuda_device="cpu",
        K=2,
        hidden=8,
        base_distribution=base_distribution,
        pca_latent_dimension=2 if base_distribution == "GaussianPCA" else None,
        max_iter=2,
        val_interval=2,
        penalty_type="barlow_twins_align",
        verbose=False,
    )

    whitened_full = apply_lamnr_flows_whitener(
        out, [df0, df1], direction="forward", output_space="whitened_full", device="cpu"
    )
    reconstructed = apply_lamnr_flows_whitener(
        out, whitened_full, direction="inverse", input_space="whitened_full", device="cpu"
    )
    np.testing.assert_allclose(reconstructed[0].to_numpy(), df0.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(reconstructed[1].to_numpy(), df1.to_numpy(), atol=1e-5)

