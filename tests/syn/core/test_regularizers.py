# Ported from syntx v6.44.0 tests/test_regularizers.py (imports rewritten to antstorch.syn.core).
"""
Unit tests and counterfactual benchmarks for incompressibility and continuum-mechanics
regularization operators in syntx.core.regularizers and syntx.core.jacobian.

All tests execute in milliseconds on CPU with deterministic seeds.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from antstorch.syn.core.regularizers import (
    compute_divergence_nd,
    compute_curl_nd,
    project_solenoidal,
    apply_solenoidal_sobolev_operator,
    apply_div_curl_green_operator,
    apply_navier_green_operator,
    apply_masked_incompressible_filter,
    apply_beltrami_regularizer,
    apply_poroelastic_filter,
    apply_hyperelastic_regularizer,
    get_regularizer,
    register_regularizer,
    list_regularizers,
)
from antstorch.syn.core.optimizers import RegAdam


# -----------------------------------------------------------------------------
# 1. Registry and Factory Tests
# -----------------------------------------------------------------------------
def test_registry_and_list_regularizers():
    regs = list_regularizers()
    expected = [
        'sobolev', 'gaussian', 'dsti', 'dsti1', 'bspline',
        'solenoidal', 'leray', 'solenoidal_sobolev',
        'div_curl', 'helmholtz', 'navier', 'stokes', 'elastic',
        'masked_incompressible', 'none', 'identity',
    ]
    for exp in expected:
        assert exp in regs, f"Expected {exp} in registered regularizers"

    # Test custom registration
    @register_regularizer('my_custom_identity')
    def _custom_builder(**kwargs):
        return lambda v: v * 2.0

    assert 'my_custom_identity' in list_regularizers()
    custom_fn = get_regularizer('my_custom_identity')
    x = torch.ones(1, 8, 8, 2)
    assert torch.allclose(custom_fn(x), 2.0 * x)

    # Test unknown name raises ValueError
    with pytest.raises(ValueError, match="unknown regularizer"):
        get_regularizer("nonexistent_reg_xyz")


# -----------------------------------------------------------------------------
# 2. Solenoidal (Leray) Divergence-Free Projection Tests (2D & 3D)
# -----------------------------------------------------------------------------
def test_solenoidal_projection_annihilates_divergence_2d_and_3d():
    torch.manual_seed(42)

    # 2D test
    v_2d = torch.randn(1, 16, 16, 2, dtype=torch.float32)
    v_sol_2d = project_solenoidal(v_2d, spacing=(1.0, 1.0))
    assert v_sol_2d.shape == v_2d.shape

    # Spectral divergence must be machine-epsilon zero
    div_spec_2d = compute_divergence_nd(v_sol_2d, spacing=(1.0, 1.0), method='spectral')
    assert float(div_spec_2d.abs().max().item()) < 1e-5

    # 3D test
    v_3d = torch.randn(1, 12, 12, 12, 3, dtype=torch.float32)
    v_sol_3d = project_solenoidal(v_3d, spacing=(1.0, 1.0, 1.0))
    assert v_sol_3d.shape == v_3d.shape

    div_spec_3d = compute_divergence_nd(v_sol_3d, spacing=(1.0, 1.0, 1.0), method='spectral')
    assert float(div_spec_3d.abs().max().item()) < 1e-5

    # Test unbatched input shape
    v_unbatched = torch.randn(14, 14, 2, dtype=torch.float32)
    v_sol_un = project_solenoidal(v_unbatched, spacing=(1.0, 1.0))
    assert v_sol_un.shape == (14, 14, 2)
    div_un = compute_divergence_nd(v_sol_un, spacing=(1.0, 1.0), method='spectral')
    assert float(div_un.abs().max().item()) < 1e-5


# -----------------------------------------------------------------------------
# 3. Counterfactual Tests: Pure Gradient vs Pure Rotational Fields
# -----------------------------------------------------------------------------
def test_counterfactual_pure_divergence_vs_pure_rotational_2d():
    """
    Counterfactual physical verification:
    - Pure divergence (gradient of potential) field MUST be annihilated by solenoidal projection.
    - Pure rotational (curl of streamfunction) field MUST be 100% preserved by solenoidal projection.
    """
    H, W = 32, 32
    y = torch.arange(H, dtype=torch.float32) / H * (2.0 * math.pi)
    x = torch.arange(W, dtype=torch.float32) / W * (2.0 * math.pi)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    # Potential phi = sin(gy) * cos(gx)
    # Pure gradient: v_y = dphi/dy = cos(gy)*cos(gx), v_x = dphi/dx = -sin(gy)*sin(gx)
    v_grad = torch.stack([
        torch.cos(gy) * torch.cos(gx),
        -torch.sin(gy) * torch.sin(gx),
    ], dim=-1).unsqueeze(0)

    # Streamfunction psi = sin(gx) * cos(gy)
    # Pure curl: v_y = dpsi/dx = cos(gx)*cos(gy), v_x = -dpsi/dy = sin(gx)*sin(gy)
    # Analytical div(v_rot) = d(cos(gx)cos(gy))/dy + d(sin(gx)sin(gy))/dx = -sin(gy)cos(gx) + sin(gy)cos(gx) = 0
    v_rot = torch.stack([
        torch.cos(gx) * torch.cos(gy),
        torch.sin(gx) * torch.sin(gy),
    ], dim=-1).unsqueeze(0)

    # Counterfactual 1: Pure gradient field is annihilated
    v_sol_from_grad = project_solenoidal(v_grad, spacing=(1.0, 1.0))
    grad_norm_init = float(v_grad.norm().item())
    grad_norm_proj = float(v_sol_from_grad.norm().item())
    assert (grad_norm_proj / grad_norm_init) < 1e-4, (
        f"Pure divergence field was not annihilated: {grad_norm_proj:.4e} vs init {grad_norm_init:.4e}"
    )

    # Counterfactual 2: Pure rotational field is preserved exactly
    v_sol_from_rot = project_solenoidal(v_rot, spacing=(1.0, 1.0))
    rot_diff = float((v_sol_from_rot - v_rot).norm().item())
    rot_norm_init = float(v_rot.norm().item())
    assert (rot_diff / rot_norm_init) < 1e-4, (
        f"Pure rotational field was not preserved: relative diff = {rot_diff / rot_norm_init:.4e}"
    )


# -----------------------------------------------------------------------------
# 4. Solenoidal Sobolev Operator Tests
# -----------------------------------------------------------------------------
def test_solenoidal_sobolev_operator():
    torch.manual_seed(42)
    m = torch.randn(1, 20, 20, 2, dtype=torch.float32)

    # Apply solenoidal Sobolev smoothing
    v_smoothed = apply_solenoidal_sobolev_operator(m, alpha=2.0, fluid_sigma=3.0, spacing=(1.0, 1.0), s=2.0)
    assert v_smoothed.shape == m.shape

    # 1. Spectral divergence is zero
    div_spec = compute_divergence_nd(v_smoothed, spacing=(1.0, 1.0), method='spectral')
    assert float(div_spec.abs().max().item()) < 1e-5

    # 2. High-frequency energy is heavily attenuated (smoothing property)
    # The gradient of the smoothed field has much smaller norm than the original field
    diff_init = float(torch.gradient(m[..., 0], dim=1)[0].norm().item())
    diff_smooth = float(torch.gradient(v_smoothed[..., 0], dim=1)[0].norm().item())
    assert diff_smooth < 0.25 * diff_init, "Field was not effectively smoothed by Sobolev kernel"


# -----------------------------------------------------------------------------
# 5. Decoupled Div-Curl (Helmholtz) Operator Tests
# -----------------------------------------------------------------------------
def test_div_curl_green_operator_decoupling():
    """
    Verify that Div-Curl Green's operator attenuates divergence vs curl independently:
    Setting beta >> gamma heavily attenuates gradient modes while preserving curl modes.
    """
    H, W = 32, 32
    y = torch.arange(H, dtype=torch.float32) / H * (2.0 * math.pi)
    x = torch.arange(W, dtype=torch.float32) / W * (2.0 * math.pi)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    v_grad = torch.stack([torch.cos(gy) * torch.cos(gx), -torch.sin(gy) * torch.sin(gx)], dim=-1).unsqueeze(0)
    v_rot = torch.stack([torch.cos(gx) * torch.cos(gy), torch.sin(gx) * torch.sin(gy)], dim=-1).unsqueeze(0)

    # Case A: Heavy dilatation penalty (beta = 100.0, gamma = 1.0)
    out_grad = apply_div_curl_green_operator(v_grad, alpha=1.0, beta=100.0, gamma=1.0, spacing=(1.0, 1.0))
    out_rot = apply_div_curl_green_operator(v_rot, alpha=1.0, beta=100.0, gamma=1.0, spacing=(1.0, 1.0))

    gain_grad = float(out_grad.norm().item()) / float(v_grad.norm().item())
    gain_rot = float(out_rot.norm().item()) / float(v_rot.norm().item())

    # Divergence modes should be attenuated significantly more than rotational modes (at least 5x)
    assert gain_grad < 0.20 * gain_rot, (
        f"Dilatation was not selectively suppressed: gain_grad={gain_grad:.4f}, gain_rot={gain_rot:.4f}"
    )

    # Case B: Balanced isotropic (beta = gamma = 1.0)
    out_grad_iso = apply_div_curl_green_operator(v_grad, alpha=1.0, beta=1.0, gamma=1.0, spacing=(1.0, 1.0))
    out_rot_iso = apply_div_curl_green_operator(v_rot, alpha=1.0, beta=1.0, gamma=1.0, spacing=(1.0, 1.0))

    gain_grad_iso = float(out_grad_iso.norm().item()) / float(v_grad.norm().item())
    gain_rot_iso = float(out_rot_iso.norm().item()) / float(v_rot.norm().item())
    assert abs(gain_grad_iso - gain_rot_iso) < 1e-4, "Isotropic div-curl should attenuate equally"


# -----------------------------------------------------------------------------
# 6. Navier-Cauchy / Stokes Operator Tests
# -----------------------------------------------------------------------------
def test_navier_green_operator_poisson_ratio():
    """
    Verify Navier operator approaches Stokes solenoidal projection as Poisson ratio nu -> 0.5.
    """
    torch.manual_seed(42)
    m = torch.randn(1, 16, 16, 2, dtype=torch.float32)

    # nu = 0.50 (incompressible limit)
    v_stokes = apply_navier_green_operator(m, alpha=1.0, poisson_ratio=0.50, spacing=(1.0, 1.0))
    div_stokes = compute_divergence_nd(v_stokes, spacing=(1.0, 1.0), method='spectral')
    assert float(div_stokes.abs().max().item()) < 1e-5, "Stokes limit did not produce zero divergence"

    # nu = 0.30 (compressible) vs nu = 0.49 (nearly incompressible soft tissue)
    v_comp = apply_navier_green_operator(m, alpha=1.0, poisson_ratio=0.30, spacing=(1.0, 1.0))
    v_soft = apply_navier_green_operator(m, alpha=1.0, poisson_ratio=0.49, spacing=(1.0, 1.0))

    div_comp_rms = float(compute_divergence_nd(v_comp, spacing=(1.0, 1.0), method='spectral').pow(2).mean().sqrt().item())
    div_soft_rms = float(compute_divergence_nd(v_soft, spacing=(1.0, 1.0), method='spectral').pow(2).mean().sqrt().item())

    # Divergence in soft tissue (nu=0.49) should be significantly lower than compressible (nu=0.30)
    assert div_soft_rms < 0.10 * div_comp_rms, (
        f"nu=0.49 divergence ({div_soft_rms:.4e}) was not << nu=0.30 ({div_comp_rms:.4e})"
    )


# -----------------------------------------------------------------------------
# 7. Mask-Gated / Spatially-Varying Incompressibility Tests
# -----------------------------------------------------------------------------
def test_masked_incompressible_filter():
    """
    Verify that masked incompressibility selectively eliminates divergence inside the tissue mask
    while preserving dilatation in the unmasked region.
    """
    H, W = 32, 32
    torch.manual_seed(42)
    v = torch.randn(1, H, W, 2, dtype=torch.float32)

    # Create a circular mask (1 inside radius R, 0 outside)
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    mask = (gx**2 + gy**2 <= 0.25).float().unsqueeze(0)  # inner disk

    v_filtered = apply_masked_incompressible_filter(v, mask=mask, spacing=(1.0, 1.0), num_iters=3)

    div_orig = compute_divergence_nd(v, spacing=(1.0, 1.0), method='spectral')
    div_filt = compute_divergence_nd(v_filtered, spacing=(1.0, 1.0), method='spectral')

    # Measure divergence inside mask vs outside mask
    inside_orig = float((div_orig * mask).pow(2).sum().sqrt().item())
    inside_filt = float((div_filt * mask).pow(2).sum().sqrt().item())
    outside_orig = float((div_orig * (1.0 - mask)).pow(2).sum().sqrt().item())
    outside_filt = float((div_filt * (1.0 - mask)).pow(2).sum().sqrt().item())

    # Divergence inside the mask should be reduced by >= 80%
    reduction_inside = 1.0 - (inside_filt / inside_orig)
    assert reduction_inside >= 0.80, f"Divergence reduction inside mask was only {reduction_inside * 100:.1f}%"

    # Outside the mask, divergence should be substantially preserved
    assert outside_filt > 0.50 * outside_orig, "Unmasked region divergence was suppressed excessively"


# -----------------------------------------------------------------------------
# 8. Differential Operator Utilities (compute_divergence_nd, compute_curl_nd)
# -----------------------------------------------------------------------------
def test_differential_operators_accuracy():
    """
    Verify central difference divergence and curl against analytical quadratic test field:
    v_y = y^2 + y * x, v_x = x^2 + 2 * y * x
    => div(v) = (2y + x) + (2x + 2y) = 4y + 3x
    => curl(v) = dvx/dy - dvy/dx = 2x - y.
    """
    H, W = 21, 21
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    sp_y = float(y[1] - y[0])
    sp_x = float(x[1] - x[0])
    spacing = (sp_x, sp_y)  # ANTs order (x, y)

    v = torch.stack([gy**2 + gy * gx, gx**2 + 2.0 * gy * gx], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    div_num = compute_divergence_nd(v, spacing=spacing, method='central')
    div_exact = 4.0 * gy + 3.0 * gx

    # On interior voxels (excluding 1-voxel border of one-sided difference)
    err_div = float((div_num[:, 1:-1, 1:-1] - div_exact[1:-1, 1:-1]).abs().max().item())
    assert err_div < 1e-4, f"Divergence numerical error too large: {err_div:.4e}"

    curl_num = compute_curl_nd(v, spacing=spacing)
    curl_exact = 2.0 * gx - gy
    err_curl = float((curl_num[:, 1:-1, 1:-1] - curl_exact[1:-1, 1:-1]).abs().max().item())
    assert err_curl < 1e-4, f"Curl numerical error too large: {err_curl:.4e}"


# -----------------------------------------------------------------------------
# 9. Counterfactual Tests: Log-Jacobian Volume vs Deviatoric Shear Penalties
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 10. RegAdam Optimization Integration Tests
# -----------------------------------------------------------------------------
def test_regadam_with_continuum_regularizers():
    """
    Verify that RegAdam seamlessly updates parameters with the new regularizer strings.
    """
    torch.manual_seed(42)

    for reg_name in ['solenoidal', 'div_curl', 'navier', 'beltrami', 'poroelastic', 'hyperelastic']:
        p = torch.nn.Parameter(torch.randn(1, 12, 12, 2, dtype=torch.float32))
        opt = RegAdam([p], lr=0.10, regularizer=reg_name, max_step_norm=0.50, spacing=(1.0, 1.0))

        # Forward loss and gradient step
        loss = (p ** 2).sum()
        loss.backward()
        p_init = p.clone().detach()

        opt.step()

        # Parameter must have updated without NaNs or Infs
        assert not torch.isnan(p).any(), f"NaN in RegAdam update for regularizer={reg_name}"
        assert not torch.isinf(p).any(), f"Inf in RegAdam update for regularizer={reg_name}"
        assert not torch.allclose(p, p_init), f"Parameter did not update for regularizer={reg_name}"


# -----------------------------------------------------------------------------
# 11. Beltrami / Quasiconformal Distortion Regularizer Tests
# -----------------------------------------------------------------------------
def test_beltrami_quasiconformal_shear_damping():
    """
    Verify that Beltrami / Quasiconformal regularizer smooths fields while damping
    deviatoric shear strain modes.
    """
    torch.manual_seed(42)
    H, W, D = 16, 16, 16
    v = torch.randn(1, H, W, D, 3) * 0.1

    v_beltrami = apply_beltrami_regularizer(v, alpha=2.0, spacing=(1.0, 1.0, 1.0))
    assert v_beltrami.shape == v.shape
    assert not torch.isnan(v_beltrami).any()

    # Damping: high frequency random field energy must be significantly attenuated
    assert v_beltrami.norm() < v.norm()

    # In 2D: verify conformal behavior
    v2d = torch.randn(1, 20, 20, 2) * 0.1
    v2d_beltrami = apply_beltrami_regularizer(v2d, alpha=2.0, spacing=(1.0, 1.0))
    assert v2d_beltrami.shape == v2d.shape
    assert not torch.isnan(v2d_beltrami).any()


# -----------------------------------------------------------------------------
# 12. Two-Phase Poroelastic / Darcy-Stokes Consolidation Filter Tests
# -----------------------------------------------------------------------------
def test_poroelastic_consolidation_filter():
    """
    Verify that Two-Phase Poroelastic filter separates deformation into solenoidal
    solid skeleton and Darcy fluid flux, correctly dampening masked parenchyma divergence.
    """
    torch.manual_seed(42)
    H, W = 24, 24
    v = torch.randn(1, H, W, 2) * 0.1

    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    mask = (gx**2 + gy**2 <= 0.36).float().unsqueeze(0)  # inner parenchyma

    v_poro = apply_poroelastic_filter(v, alpha=2.0, darcy_permeability=0.1, mask=mask, spacing=(1.0, 1.0))
    assert v_poro.shape == v.shape
    assert not torch.isnan(v_poro).any()

    div_orig = compute_divergence_nd(v, spacing=(1.0, 1.0), method='spectral')
    div_poro = compute_divergence_nd(v_poro, spacing=(1.0, 1.0), method='spectral')

    inside_orig = float((div_orig * mask).pow(2).sum().sqrt().item())
    inside_poro = float((div_poro * mask).pow(2).sum().sqrt().item())

    # Divergence inside parenchyma should be attenuated by >= 60% due to solenoidal solid matrix
    reduction = 1.0 - (inside_poro / inside_orig)
    assert reduction >= 0.60, f"Poroelastic divergence reduction was only {reduction * 100:.1f}%"


# -----------------------------------------------------------------------------
# 13. Hyperelastic Field Regularizer Tests
# -----------------------------------------------------------------------------
def test_hyperelastic_field_regularizer():
    """
    Verify that apply_hyperelastic_regularizer heavily penalizes divergent modes.
    """
    torch.manual_seed(42)
    H, W = 24, 24
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    # Construct divergent field
    v = torch.stack([gy, gx], dim=-1).unsqueeze(0)  # div(v) = 2.0 everywhere
    v_hyp = apply_hyperelastic_regularizer(v, alpha=1.0, bulk_modulus=20.0, spacing=(1.0, 1.0))

    assert v_hyp.shape == v.shape
    assert not torch.isnan(v_hyp).any()

    div_orig = compute_divergence_nd(v, spacing=(1.0, 1.0), method='spectral')
    div_hyp = compute_divergence_nd(v_hyp, spacing=(1.0, 1.0), method='spectral')
    assert div_hyp.abs().mean() < div_orig.abs().mean(), "Hyperelastic filter failed to damp volumetric dilation"


# -----------------------------------------------------------------------------
# 14. Full Registry Coverage for All 7 Continuum Paradigms + Aliases
# -----------------------------------------------------------------------------
def test_all_continuum_paradigms_in_registry():
    """
    Verify that every brainstormed paradigm and its aliases resolve cleanly through get_regularizer.
    """
    paradigms = [
        'solenoidal', 'leray', 'incompressible', 'solenoidal_sobolev',
        'div_curl', 'helmholtz',
        'navier', 'stokes', 'elastic',
        'masked_incompressible',
        'hyperelastic', 'simo_pister', 'log_jacobian',
        'beltrami', 'quasiconformal', 'conformal',
        'poroelastic', 'darcy_stokes', 'biot',
        'sobolev', 'gaussian', 'dsti', 'dsti1', 'bspline', 'identity', 'none',
    ]

    field = torch.randn(1, 16, 16, 2)
    for p in paradigms:
        reg_fn = get_regularizer(p, alpha=1.0, fluid_sigma=2.0, spacing=(1.0, 1.0))
        out = reg_fn(field)
        assert out.shape == field.shape, f"Shape mismatch for regularizer {p}"
        assert not torch.isnan(out).any(), f"NaN for regularizer {p}"


# -----------------------------------------------------------------------------
# 15. Benchmark Call Parameter Collision Invariance Test
# -----------------------------------------------------------------------------
def test_syn_registration_runs_every_continuum_regularizer():
    """ANTsTorch counterpart of syntx's benchmark-collision test: every coupled continuum
    regularizer (with its extra operator parameters) runs through syn_registration()."""
    import numpy as np
    import ants
    from antstorch.syn import syn_registration

    arr_f = np.zeros((24, 24), dtype=np.float32)
    arr_f[6:18, 6:18] = 1.0
    arr_m = np.zeros((24, 24), dtype=np.float32)
    arr_m[7:19, 5:17] = 1.0
    fi = ants.from_numpy(arr_f)
    mi = ants.from_numpy(arr_m)
    mask = ants.from_numpy((arr_f > 0).astype(np.float32))  # an ANTsImage mask is put in tensor order

    cases = [
        ('solenoidal', {}),
        ('div_curl', {'beta': 25.0, 'gamma': 1.0}),
        ('navier', {'poisson_ratio': 0.49}),
        ('masked_incompressible', {'mask': mask}),
        ('hyperelastic', {'bulk_modulus': 10.0}),
        ('beltrami', {}),
        ('poroelastic', {'mask': mask, 'darcy_permeability': 0.20}),
    ]
    for reg_name, extra_kw in cases:
        res = syn_registration(fi, mi, type_of_transform="SyNOnly", levels=(2, 1), reg_iterations=(2, 1),
                               regularizer=reg_name, regularizer_kwargs=extra_kw or None, device='cpu')
        assert res["provenance"]["regularizer"] == reg_name
        assert np.isfinite(res["warpedmovout"].numpy()).all()


# -----------------------------------------------------------------------------
# 9. Relative Aspect Ratio Scaling & Pyramid Scale Invariance Tests (GEMINI.md Rule 2)
# -----------------------------------------------------------------------------
def test_relative_aspect_ratio_pyramid_scale_invariance():
    """
    Verify GEMINI.md Rule 2:
    Continuum regularizers must normalize frequency grids by relative aspect ratio
    (spacing / min(spacing)) so that:
    1. Multi-resolution pyramid downsampling (e.g. 1mm -> 4mm) preserves identical
       relative frequency meshes and does NOT compound L^(2s) over-damping at fine levels.
    2. Anisotropic physical spacing (e.g. 1x1x3 mm) correctly weights through-plane vs
       in-plane wave numbers by the true geometric aspect ratio.
    """
    from antstorch.syn.core.regularizers import _get_k_mesh_rfft, apply_div_curl_green_operator

    device = torch.device('cpu')
    shape = (16, 16, 16)

    # 1. Isotropic pyramid levels must be bitwise identical
    _, k_sq_fine_iso, _ = _get_k_mesh_rfft(shape, spacing=(1.0, 1.0, 1.0), device=device, relative_spacing=True)
    _, k_sq_coarse_iso, _ = _get_k_mesh_rfft(shape, spacing=(4.0, 4.0, 4.0), device=device, relative_spacing=True)
    with pytest.raises(ValueError, match="spacing"):
        _get_k_mesh_rfft(shape, spacing=None, device=device, relative_spacing=True)

    assert torch.allclose(k_sq_fine_iso, k_sq_coarse_iso)

    # 2. Anisotropic pyramid levels must be bitwise identical
    k_mesh_fine_aniso, k_sq_fine_aniso, _ = _get_k_mesh_rfft(
        shape, spacing=(1.0, 1.0, 3.0), device=device, relative_spacing=True
    )
    k_mesh_coarse_aniso, k_sq_coarse_aniso, _ = _get_k_mesh_rfft(
        shape, spacing=(4.0, 4.0, 12.0), device=device, relative_spacing=True
    )

    assert torch.allclose(k_sq_fine_aniso, k_sq_coarse_aniso)

    # In tensor order (z, y, x), axis 0 is z (spacing 3.0), axes 1 & 2 are y & x (spacing 1.0)
    # Max frequency along z must be exactly 1/3 of max frequency along x or y
    max_kz = k_mesh_fine_aniso[0].abs().max().item()
    max_ky = k_mesh_fine_aniso[1].abs().max().item()
    max_kx = k_mesh_fine_aniso[2].abs().max().item()

    assert abs(max_kx - max_ky) < 1e-5
    assert abs(max_kz - max_kx / 3.0) < 1e-4

    # 3. Continuum operator produces identical normalized filtering across scaled spacing inputs
    v = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32)
    filtered_1mm = apply_div_curl_green_operator(v, spacing=(1.0, 1.0, 3.0))
    filtered_4mm = apply_div_curl_green_operator(v, spacing=(4.0, 4.0, 12.0))

    assert torch.allclose(filtered_1mm, filtered_4mm, atol=1e-6)



