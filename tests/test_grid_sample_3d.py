import pytest
import torch
from antstorch._grid_sample_3d import grid_sample_3d


@pytest.mark.parametrize('padding', ['zeros', 'border', 'reflection'])
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
def test_values_and_gradients(padding, align, mode):
    torch.manual_seed(17)
    image = torch.randn(2, 2, 4, 5, 6, dtype=torch.double, requires_grad=True)
    grid = (torch.rand(2, 3, 4, 5, 3, dtype=torch.double) * 5 - 2.5).requires_grad_()
    expected = torch.nn.functional.grid_sample(image, grid, mode=mode, padding_mode=padding, align_corners=align)
    actual = grid_sample_3d(image, grid, mode=mode, padding_mode=padding, align_corners=align, chunk_size=7)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weights = torch.randn_like(expected)
    ref_grad = torch.autograd.grad(expected, (image, grid), weights)
    got_grad = torch.autograd.grad(actual, (image, grid), weights)
    for got, ref in zip(got_grad, ref_grad):
        torch.testing.assert_close(got, ref, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('padding', ['zeros', 'border', 'reflection'])
@pytest.mark.parametrize('align', [False, True])
def test_singleton_and_edges(padding, align):
    image = torch.arange(6., dtype=torch.double).reshape(1, 1, 1, 2, 3)
    grid = torch.tensor([-3., -1., -.5, 0., .5, 1., 3.], dtype=torch.double)
    grid = grid[:, None].expand(-1, 3).reshape(1, 1, 1, 7, 3)
    expected = torch.nn.functional.grid_sample(image, grid, padding_mode=padding, align_corners=align)
    torch.testing.assert_close(grid_sample_3d(image, grid, padding_mode=padding, align_corners=align), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_forward_and_backward():
    torch.manual_seed(4)
    image = torch.randn(1, 2, 4, 5, 6, requires_grad=True)
    grid = (torch.rand(1, 3, 4, 5, 3) * 2.4 - 1.2).requires_grad_()
    ref = torch.nn.functional.grid_sample(image, grid, padding_mode='border', align_corners=True)
    expected = torch.autograd.grad(ref.square().sum(), (image, grid))
    m_image = image.detach().to('mps').requires_grad_()
    m_grid = grid.detach().to('mps').requires_grad_()
    actual = grid_sample_3d(m_image, m_grid, padding_mode='border', align_corners=True, chunk_size=13)
    gradients = torch.autograd.grad(actual.square().sum(), (m_image, m_grid))
    torch.testing.assert_close(actual.cpu(), ref, atol=2e-5, rtol=2e-5)
    for got, target in zip(gradients, expected):
        torch.testing.assert_close(got.cpu(), target, atol=1e-4, rtol=1e-4)


def test_experimental_dispatch_is_opt_in_and_mps_only(monkeypatch):
    from types import SimpleNamespace
    from antstorch import _torch_compat as compat
    monkeypatch.setattr(compat.F, 'grid_sample', lambda *a, **kw: 'native')
    monkeypatch.setattr(compat, 'grid_sample_3d', lambda *a, **kw: 'portable')
    mps = SimpleNamespace(device=SimpleNamespace(type='mps'), ndim=5)
    cpu = SimpleNamespace(device=SimpleNamespace(type='cpu'), ndim=5)
    monkeypatch.setenv('ANTSTORCH_MPS_GRID_SAMPLE', 'native')
    assert compat.grid_sample_for_probe(mps, None) == 'native'
    monkeypatch.setenv('ANTSTORCH_MPS_GRID_SAMPLE', 'torch')
    assert compat.grid_sample_for_probe(mps, None) == 'portable'
    assert compat.grid_sample_for_probe(cpu, None) == 'native'


def test_gradcheck():
    torch.manual_seed(12)
    image = torch.randn(1, 1, 3, 3, 3, dtype=torch.double, requires_grad=True)
    grid = (torch.rand(1, 1, 2, 2, 3, dtype=torch.double) - .5).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: grid_sample_3d(a, b, align_corners=True),
                                    (image, grid))


@pytest.mark.parametrize('available,expected', [('native','native'),('metal','metal'),('none','error')])
def test_auto_backend_priority(monkeypatch,available,expected):
    from types import SimpleNamespace
    from antstorch import _torch_compat as compat
    from antstorch import _grid_sample_3d_metal as metal
    monkeypatch.delenv('ANTSTORCH_MPS_GRID_SAMPLE',raising=False)
    monkeypatch.setattr(compat,'_available',lambda backend,*args: backend==available)
    monkeypatch.setattr(compat.F,'grid_sample',lambda *a,**kw:'native')
    monkeypatch.setattr(metal,'grid_sample_3d_metal',lambda *a,**kw:'metal')
    image=SimpleNamespace(device=SimpleNamespace(type='mps'),ndim=5,dtype=torch.float32,requires_grad=False)
    grid=SimpleNamespace(requires_grad=True)
    if expected=='error':
        with pytest.raises(NotImplementedError,match='neither native nor Metal'):
            compat.grid_sample_for_probe(image,grid)
    else:
        assert compat.grid_sample_for_probe(image,grid)==expected
