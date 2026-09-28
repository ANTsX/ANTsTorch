"""Fused float32 MPS sampler with first-order autograd support."""
from functools import lru_cache
import torch
from torch.autograd.function import once_differentiable

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
// p: N,C,D,H,W,P,align,padding,nearest,need_input_grad,need_grid_grad
inline float coordinate(float u, int size, int align, int padding, thread float& grad) {
    if (isnan(u)) u = -1.0f;
    grad = align ? (size-1)*0.5f : size*0.5f;
    float x = align ? (u+1)*(size-1)*0.5f : ((u+1)*size-1)*0.5f;
    if (padding == 2) {
        float lo = align ? 0.0f : -0.5f;
        float span = align ? float(size-1) : float(size);
        if (span == 0) { x=0; grad=0; }
        else {
            float shifted=x-lo;
            float dist=abs(shifted);
            float turns=floor(dist/span);
            float rem=fmod(dist,span);
            bool reverse=fmod(turns,2.0f)!=0;
            x=lo+(reverse ? span-rem : rem);
            grad *= (shifted<0 ? -1.0f : 1.0f)*(reverse ? -1.0f : 1.0f);
        }
    }
    if (padding != 0) {
        if (x<=0) {x=0; grad=0;}
        else if (x>=size-1) {x=size-1; grad=0;}
    }
    return x;
}
inline bool valid(int x,int y,int z,constant int* p) {
    return x>=0 && x<p[4] && y>=0 && y<p[3] && z>=0 && z<p[2];
}
inline uint voxel_offset(int n,int c,int x,int y,int z,constant int* p) {
    return (((uint(n)*p[1]+c)*p[2]+z)*p[3]+y)*p[4]+x;
}
inline void add_float(device atomic_uint* ptr,float v) {
    uint old=atomic_load_explicit(ptr,memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(ptr,&old,as_type<uint>(as_type<float>(old)+v),
            memory_order_relaxed,memory_order_relaxed)) {}
}
kernel void sample_forward(const device float* image, const device float* grid,
                           device float* output, constant int* p,
                           uint tid [[thread_position_in_grid]]) {
    if (tid>=uint(p[0]*p[1]*p[5])) return;
    int point=tid%p[5], c=(tid/p[5])%p[1], n=tid/(p[5]*p[1]);
    uint gi=(n*p[5]+point)*3;
    float unused;
    float x=coordinate(grid[gi],p[4],p[6],p[7],unused);
    float y=coordinate(grid[gi+1],p[3],p[6],p[7],unused);
    float z=coordinate(grid[gi+2],p[2],p[6],p[7],unused);
    // Nonfinite/far-out zero padding must not overflow float-to-int conversion.
    if (!isfinite(x)||!isfinite(y)||!isfinite(z)||x < -1||x > p[4]||y < -1||y > p[3]||z < -1||z > p[2]) {output[tid]=0;return;}
    if (p[8]) {
        int ix=int(rint(x)),iy=int(rint(y)),iz=int(rint(z));
        output[tid]=valid(ix,iy,iz,p) ? image[voxel_offset(n,c,ix,iy,iz,p)] : 0;
        return;
    }
    int ix=int(floor(x)),iy=int(floor(y)),iz=int(floor(z));
    float fx=x-ix,fy=y-iy,fz=z-iz,sum=0;
    for(int dz=0;dz<2;dz++) for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
        if(valid(ix+dx,iy+dy,iz+dz,p))
            sum+=image[voxel_offset(n,c,ix+dx,iy+dy,iz+dz,p)]*(dx?fx:1-fx)*(dy?fy:1-fy)*(dz?fz:1-fz);
    }
    output[tid]=sum;
}
kernel void sample_backward(const device float* image,const device float* grid,
                            const device float* gout,device atomic_uint* gimage,
                            device float* ggrid,constant int* p,
                            uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p[0]*p[5])) return;
    int point=tid%p[5],n=tid/p[5]; uint gi=tid*3;
    float sx,sy,sz;
    float x=coordinate(grid[gi],p[4],p[6],p[7],sx);
    float y=coordinate(grid[gi+1],p[3],p[6],p[7],sy);
    float z=coordinate(grid[gi+2],p[2],p[6],p[7],sz);
    if (!isfinite(x)||!isfinite(y)||!isfinite(z)||x < -1||x > p[4]||y < -1||y > p[3]||z < -1||z > p[2]) return;
    if(p[8]) {
        int ix=int(rint(x)),iy=int(rint(y)),iz=int(rint(z));
        if(p[9] && valid(ix,iy,iz,p)) for(int c=0;c<p[1];c++)
            add_float(gimage+voxel_offset(n,c,ix,iy,iz,p),gout[(n*p[1]+c)*p[5]+point]);
        return;
    }
    int ix=int(floor(x)),iy=int(floor(y)),iz=int(floor(z));
    float fx=x-ix,fy=y-iy,fz=z-iz,gx=0,gy=0,gz=0;
    for(int c=0;c<p[1];c++) {
        float g=gout[(n*p[1]+c)*p[5]+point];
        for(int dz=0;dz<2;dz++) for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
            if(!valid(ix+dx,iy+dy,iz+dz,p)) continue;
            uint idx=voxel_offset(n,c,ix+dx,iy+dy,iz+dz,p);
            float wx=dx?fx:1-fx,wy=dy?fy:1-fy,wz=dz?fz:1-fz;
            if(p[9]) add_float(gimage+idx,g*wx*wy*wz);
            if(p[10]) {
                float v=image[idx]*g;
                gx+=v*(dx?1.0f:-1.0f)*wy*wz;
                gy+=v*wx*(dy?1.0f:-1.0f)*wz;
                gz+=v*wx*wy*(dz?1.0f:-1.0f);
            }
        }
    }
    if(p[10]) {ggrid[gi]=gx*sx;ggrid[gi+1]=gy*sy;ggrid[gi+2]=gz*sz;}
}
'''


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


class _Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, grid, mode, padding, align):
        image, grid = image.contiguous(), grid.contiguous()
        n,c,d,h,w=image.shape
        points=grid.shape[1]*grid.shape[2]*grid.shape[3]
        params=torch.tensor([n,c,d,h,w,points,int(align),padding,int(mode=='nearest'),
                             int(ctx.needs_input_grad[0]),int(ctx.needs_input_grad[1])],
                            dtype=torch.int32,device=image.device)
        output=torch.empty((n,c,*grid.shape[1:4]),dtype=image.dtype,device=image.device)
        if output.numel():
            _library().sample_forward(image,grid,output,params,threads=output.numel())
            # PyTorch 2.7 raw shader launches must complete before temporary
            # tensor storage can be recycled by later Torch graph operations.
            # Removing this barrier caused intermittent full-volume corruption.
            torch.mps.synchronize()
        ctx.save_for_backward(image,grid,params)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        image,grid,params=ctx.saved_tensors
        need_image,need_grid=ctx.needs_input_grad[:2]
        if need_image and torch.are_deterministic_algorithms_enabled():
            raise RuntimeError('Metal grid_sample image backward uses nondeterministic atomic additions')
        gi=torch.zeros_like(image) if need_image else image.new_zeros(1)
        gg=torch.zeros_like(grid) if need_grid else image.new_zeros(1)
        points=grid.numel()//3
        if points:
            grad = grad.contiguous()
            _library().sample_backward(image,grid,grad,gi,gg,params,threads=points)
            torch.mps.synchronize()
        return gi if need_image else None,gg if need_grid else None,None,None,None


def grid_sample_3d_metal(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """Fused MPS float32 interpolation; first-order derivatives only.

    Input-gradient atomic sums can be nondeterministic. Finite coordinates are
    the supported contract; NaNs are mapped to -1 as in grid_sample's docs.
    """
    if input.ndim!=5 or grid.ndim!=5 or grid.shape[-1]!=3:
        raise ValueError('expected input NCDHW and grid NDHW3')
    if input.device.type!='mps' or input.device!=grid.device or input.dtype!=torch.float32 or grid.dtype!=torch.float32:
        raise ValueError('Metal sampler requires float32 input and grid on the same MPS device')
    if input.shape[0]!=grid.shape[0] or min(input.shape[2:])<1:
        raise ValueError('batch sizes must agree and input spatial dimensions must be nonempty')
    if max(input.numel(),grid.numel(),input.shape[1]*(grid.numel()//3))>=2**31:
        raise ValueError('Metal sampler currently requires 32-bit indexing')
    if mode not in ('bilinear','nearest') or padding_mode not in ('zeros','border','reflection'):
        raise ValueError('unsupported sampling or padding mode')
    if not hasattr(torch.mps,'compile_shader'):
        raise RuntimeError('This PyTorch build lacks torch.mps.compile_shader')
    return _Sample.apply(input,grid,mode,{'zeros':0,'border':1,'reflection':2}[padding_mode],bool(align_corners))
