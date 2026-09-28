import pytest
import torch
from antstorch._grid_sample_3d_metal import grid_sample_3d_metal

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')


@pytest.mark.parametrize('padding',['zeros','border','reflection'])
@pytest.mark.parametrize('align',[False,True])
@pytest.mark.parametrize('mode',['bilinear','nearest'])
def test_metal_values_and_gradients(padding,align,mode):
    torch.manual_seed(17)
    image=torch.randn(2,2,4,5,6,requires_grad=True)
    grid=(torch.rand(2,3,4,5,3)*5-2.5).requires_grad_()
    expected=torch.nn.functional.grid_sample(image,grid,mode=mode,padding_mode=padding,align_corners=align)
    weight=torch.randn_like(expected)
    grads=torch.autograd.grad(expected,(image,grid),weight)
    a=image.detach().to('mps').requires_grad_(); b=grid.detach().to('mps').requires_grad_()
    actual=grid_sample_3d_metal(a,b,mode=mode,padding_mode=padding,align_corners=align)
    got=torch.autograd.grad(actual,(a,b),weight.to('mps'))
    torch.testing.assert_close(actual.cpu(),expected,atol=5e-6,rtol=2e-5)
    for x,y in zip(got,grads): torch.testing.assert_close(x.cpu(),y,atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('padding',['zeros','border','reflection'])
@pytest.mark.parametrize('align',[False,True])
def test_edges_singleton_noncontiguous(padding,align):
    image=torch.arange(12.).reshape(1,1,1,3,4).transpose(3,4)
    values=torch.tensor([-3.,-1.,-.5,0.,.5,1.,3.])
    grid=values[:,None].expand(-1,3).reshape(1,1,1,7,3)
    expected=torch.nn.functional.grid_sample(image,grid,padding_mode=padding,align_corners=align)
    actual=grid_sample_3d_metal(image.to('mps'),grid.to('mps'),padding_mode=padding,align_corners=align)
    torch.testing.assert_close(actual.cpu(),expected)


def test_grid_only_gradient_and_sampling_probe(monkeypatch):
    from antstorch._torch_compat import grid_sample
    from antstorch.syn.core.pipeline import mps_grid_sample_3d_available, mps_grid_sample_3d_forward_available
    monkeypatch.setenv('ANTSTORCH_MPS_GRID_SAMPLE','metal')
    mps_grid_sample_3d_available.cache_clear()
    mps_grid_sample_3d_forward_available.cache_clear()
    try:
        assert mps_grid_sample_3d_available()
        assert mps_grid_sample_3d_forward_available()
        image=torch.randn(1,1,4,4,4,device='mps')
        grid=torch.zeros(1,2,2,2,3,device='mps',requires_grad=True)
        grid_sample(image,grid,align_corners=True).sum().backward()
        assert torch.isfinite(grid.grad).all()
    finally:
        mps_grid_sample_3d_available.cache_clear()
        mps_grid_sample_3d_forward_available.cache_clear()


def test_direct_with_metal_matches_cpu(monkeypatch):
    from antstorch.direct import direct_cortical_thickness
    from antstorch.bspline_flows import ImageDomain
    monkeypatch.setenv('ANTSTORCH_MPS_GRID_SAMPLE','metal')
    z,y,x=torch.meshgrid(*[torch.arange(12.)]*3,indexing='ij')
    radius=((x-5.5)**2+(y-5.5)**2+(z-5.5)**2).sqrt()
    white=radius<2.5; gray=(radius>=2.5)&(radius<4)
    seg=(3*white+2*gray).float()[None,None]
    gm=gray.float()[None,None]*.9;wm=white.float()[None,None]*.9
    domain=ImageDomain((12,12,12))
    options=dict(iterations=3,integration_points=3,inverse_iterations=3)
    expected=direct_cortical_thickness(seg,gm,wm,domain,**options)
    actual=direct_cortical_thickness(seg.to('mps'),gm.to('mps'),wm.to('mps'),domain,**options)
    torch.testing.assert_close(actual.thickness.cpu(),expected.thickness,atol=1e-4,rtol=1e-3)


def test_deterministic_image_gradient_rejected():
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        image=torch.ones(1,1,3,3,3,device='mps',requires_grad=True)
        grid=torch.zeros(1,1,1,1,3,device='mps')
        with pytest.raises(RuntimeError,match='nondeterministic'):
            grid_sample_3d_metal(image,grid).sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_storage_offsets_and_empty_grid():
    image=torch.randn(2,1,4,5,6,device='mps')[1:]
    grid=torch.zeros(2,2,3,4,3,device='mps')[1:]
    expected=torch.nn.functional.grid_sample(image.cpu(),grid.cpu(),align_corners=False)
    actual=grid_sample_3d_metal(image,grid)
    torch.testing.assert_close(actual.cpu(),expected)
    empty=torch.empty(1,0,2,3,3,device='mps',requires_grad=True)
    out=grid_sample_3d_metal(image,empty)
    assert out.shape==(1,1,0,2,3)
    out.sum().backward()
    assert empty.grad.shape==empty.shape


def test_repeated_sampling_with_temporary_inputs():
    # Reuse allocator storage between asynchronous shader and Torch operations.
    # Without the completion barrier, full DiReCT runs showed intermittent drift.
    torch.manual_seed(9)
    image=torch.randn(1,2,16,17,18)
    grid=torch.rand(1,12,13,14,3)*2-1
    ref=torch.nn.functional.grid_sample(image+1,grid,padding_mode='border',align_corners=True)
    a=image.to('mps'); b=grid.to('mps')
    for _ in range(12):
        out=grid_sample_3d_metal(a+1,b+0,padding_mode='border',align_corners=True)
        churn=torch.empty_like(a).fill_(1000)
        torch.testing.assert_close(out.cpu(),ref,atol=5e-6,rtol=2e-5)
        assert churn.shape==a.shape


@pytest.mark.parametrize('algorithm',['bspline','syn'])
def test_auto_registration_matches_cpu(monkeypatch,algorithm):
    import ants
    import numpy as np
    from antstorch.bspline_flows import ImageDomain,bspline_svf_registration
    from antstorch.syn import syn_registration
    from antstorch.syn.core.pipeline import mps_grid_sample_3d_available,mps_grid_sample_3d_forward_available
    monkeypatch.delenv('ANTSTORCH_MPS_GRID_SAMPLE',raising=False)
    mps_grid_sample_3d_available.cache_clear()
    mps_grid_sample_3d_forward_available.cache_clear()
    z,y,x=torch.meshgrid(*[torch.linspace(-1,1,16)]*3,indexing='ij')
    moving=torch.exp(-7*(x*x+y*y+z*z))[None,None]
    fixed=torch.roll(moving,1,-1)
    outputs=[]
    for device in ['cpu','mps']:
        if algorithm=='bspline':
            result=bspline_svf_registration(fixed.to(device),moving.to(device),ImageDomain((16,16,16)),
                iterations=5,mesh_size=1,squaring_steps=2,similarity='mse',learning_rate=.05)
            assert result['fwdtransforms'].device.type==device
            outputs.append(result['warpedmovout'].detach().cpu().numpy())
        else:
            f=ants.from_numpy(fixed[0,0].numpy().transpose(2,1,0).copy())
            m=ants.from_numpy(moving[0,0].numpy().transpose(2,1,0).copy())
            result=syn_registration(f,m,type_of_transform='SyNOnly',levels=(1,),reg_iterations=(5,),
                                    device=device,syn_metric='mse')
            outputs.append(result['warpedmovout'].numpy())
    np.testing.assert_allclose(outputs[0],outputs[1],atol=2e-5,rtol=2e-4)
    mps_grid_sample_3d_available.cache_clear()
    mps_grid_sample_3d_forward_available.cache_clear()
