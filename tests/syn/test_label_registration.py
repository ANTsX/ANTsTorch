import ants
import numpy as np
import pytest
import torch

from antstorch import label_image_registration
from antstorch.syn.label_registration import _landmark_affine


def labels(d=2, shift=0):
    array = np.zeros((24,) * d, dtype=np.float32)
    centers = [(6, 6), (16, 6), (6, 16)] if d == 2 else [(6, 6, 6), (16, 6, 6), (6, 16, 6), (6, 6, 16)]
    for i, center in enumerate(centers, 1):
        array[tuple(slice(c - 2 + (shift if axis == 0 else 0), c + 2 + (shift if axis == 0 else 0))
                    for axis, c in enumerate(center))] = i
    direction = np.eye(d)
    direction[:2, :2] = [[0, -1], [1, 0]]
    return ants.from_numpy(array, spacing=tuple(1 + .3 * i for i in range(d)),
                           origin=tuple(10 + i for i in range(d)), direction=direction)


@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('kind', ['rigid', 'similarity', 'affine'])
def test_landmarks_and_export(tmp_path, d, kind):
    fixed, moving = labels(d), labels(d, 2)
    result = label_image_registration(fixed, moving, initial_transforms=kind,
        type_of_deformable_transform=None, device='cpu', output_prefix=str(tmp_path / 'reg'))
    warped = ants.apply_transforms(fixed, moving, result['fwdtransforms'], interpolator='genericLabel')
    assert np.array_equal(warped.numpy(), fixed.numpy())
    inverse = ants.apply_transforms(moving, fixed, result['invtransforms'],
                                    whichtoinvert=[True], interpolator='genericLabel')
    assert np.array_equal(inverse.numpy(), moving.numpy())


@pytest.mark.parametrize('kind', ['rigid', 'similarity', 'affine'])
def test_landmark_rotation_scale(kind):
    x = np.array([[0., 0.], [4, 0], [0, 3], [3, 2]])
    a = np.array([[0., -1], [1, 0]])
    if kind != 'rigid':
        a *= 1.7
    if kind == 'affine':
        a[0, 0] = .2
    y = x @ a.T + [7, -3]
    m, t = _landmark_affine(x, y, kind)
    np.testing.assert_allclose(x @ m.T + t, y, atol=1e-10)


@pytest.mark.parametrize('d', [2, 3])
def test_deformable_labels_improve_and_export(tmp_path, d):
    fixed, moving = labels(d), labels(d, 1)
    result = label_image_registration(fixed, moving, initial_transforms='identity',
        levels=(1,), reg_iterations=(12,), flow_sigma=.6, grad_step=.2,
        device='cpu', output_prefix=str(tmp_path / 'reg'))
    assert np.isfinite(result['loss_history']).all()
    assert result['loss_history'][-1] < result['loss_history'][0]
    warped = ants.apply_transforms(fixed, moving, result['fwdtransforms'], interpolator='genericLabel')
    assert np.mean(warped.numpy() != fixed.numpy()) < np.mean(moving.numpy() != fixed.numpy())


def test_multiple_label_and_intensity_pairs_masks_custom_schedule(tmp_path):
    fixed, moving = labels(), labels(shift=1)
    mask = fixed.new_image_like(np.ones(fixed.shape, dtype=np.float32))
    result = label_image_registration([fixed, fixed], [moving, moving],
        fixed_intensity_images=fixed, moving_intensity_images=moving,
        fixed_mask=mask, moving_mask=mask, label_image_weighting=[.5, 2],
        initial_transforms='identity', type_of_deformable_transform=['SyN', .2, 'CC', 1, (2,), (0,), (1,)],
        device='cpu', output_prefix=str(tmp_path / 'reg'))
    assert len(result['common_label_ids']) == 2
    assert len(result['loss_history']) == 2
    assert np.isfinite(result['loss_history']).all()


@pytest.mark.parametrize('case', ['no_common', 'fractional', 'weight', 'domain', 'degenerate', 'nonlinear'])
def test_validation(tmp_path, case):
    fixed, moving = labels(), labels()
    options = dict(levels=(1,), reg_iterations=(0,), device='cpu', output_prefix=str(tmp_path / 'reg'))
    if case == 'no_common': moving = moving * 10
    if case == 'fractional': moving = moving * .5
    if case == 'weight': options['label_image_weighting'] = -1
    if case == 'domain': options['fixed_intensity_images'], options['moving_intensity_images'] = labels(3), moving
    if case == 'degenerate': fixed, moving = fixed == 1, moving == 1
    if case == 'nonlinear': options['initial_transforms'] = ['warp.nii.gz']
    with pytest.raises(ValueError):
        label_image_registration(fixed, moving, **options)


@pytest.mark.parametrize('device', ['mps', 'cuda'])
@pytest.mark.parametrize('d', [2, 3])
def test_accelerator(tmp_path, device, d):
    available = torch.backends.mps.is_available() if device == 'mps' else torch.cuda.is_available()
    if not available:
        pytest.skip(f'{device} unavailable')
    if device == 'mps' and d == 3:
        from antstorch.syn.core.pipeline import mps_grid_sample_3d_forward_available
        if not mps_grid_sample_3d_forward_available():
            pytest.skip('3-D MPS sampling unavailable in this PyTorch build')
    fixed, moving = labels(d), labels(d, shift=1)
    result = label_image_registration(fixed, moving, initial_transforms='identity',
        levels=(1,), reg_iterations=(3,), device=device, output_prefix=str(tmp_path / 'reg'))
    assert result['provenance']['device'] == device
    assert np.isfinite(result['loss_history']).all()


def test_affine_file_initialization_matches_landmarks(tmp_path):
    fixed, moving = labels(), labels(shift=2)
    first = label_image_registration(fixed, moving, type_of_deformable_transform=None,
        device='cpu', output_prefix=str(tmp_path / 'first'))
    second = label_image_registration(fixed, moving, initial_transforms=first['fwdtransforms'],
        levels=(1,), reg_iterations=(0,), device='cpu', output_prefix=str(tmp_path / 'second'))
    warped = ants.apply_transforms(fixed, moving, second['fwdtransforms'], interpolator='genericLabel')
    assert np.array_equal(warped.numpy(), fixed.numpy())


def test_metric_weighting_and_mask_exclusion(tmp_path):
    from antstorch.syn import syn_registration
    fixed, moving = labels(), labels(shift=1)
    # At zero updates the same binary mismatch has exactly proportional loss;
    # an unchanged label outside the mask must contribute nothing.
    options = dict(type_of_transform='SyNOnly', primary_metric_weight=0,
                   syn_metric='mse', levels=(1,), reg_iterations=(1,), device='cpu')
    a, b = fixed == 1, moving == 1
    r1 = syn_registration(fixed, moving, multivariate_extras=[('MSQ', a, b, 1, 0)],
                          outprefix=str(tmp_path / 'a'), **options)
    r2 = syn_registration(fixed, moving, multivariate_extras=[('MSQ', a, b, 3, 0)],
                          outprefix=str(tmp_path / 'b'), **options)
    assert r2['loss_history'][0] == pytest.approx(3 * r1['loss_history'][0])
    mask = fixed == 2
    masked = syn_registration(fixed, moving, multivariate_extras=[('MSQ', a, b, 1, 0)],
        mask=mask, outprefix=str(tmp_path / 'c'), **options)
    assert masked['loss_history'] == [0.0]


@pytest.mark.parametrize('regularizer', ['gaussian', 'sobolev', 'dsti', 'bspline'])
def test_regularizers_and_best_state_use_multivariate_objective(tmp_path, regularizer):
    fixed, moving = labels(), labels(shift=1)
    result = label_image_registration(fixed, moving, initial_transforms='identity',
        type_of_deformable_transform='antsRegistrationSyN[bo]' if regularizer == 'bspline' else 'antsRegistrationSyN[so]',
        regularizer=regularizer, syntx_parity=True, levels=(2, 1), reg_iterations=(2, 2),
        device='cpu', output_prefix=str(tmp_path / 'reg'))
    assert len(result['loss_history']) == 4
    assert np.isfinite(result['loss_history']).all()


@pytest.mark.parametrize('d', [2, 3])
def test_identity_on_different_domains(tmp_path, d):
    fixed = labels(d)
    moving = ants.resample_image(fixed, (28,) * d, use_voxels=True, interp_type=1)
    result = label_image_registration(fixed, moving, initial_transforms='identity',
        levels=(1,), reg_iterations=(1,), device='cpu', output_prefix=str(tmp_path / 'reg'))
    assert np.isfinite(result['loss_history']).all()
