#!/usr/bin/env python3
"""Compare MPS 3-D samplers with synchronized timings and CPU-reference errors."""
import argparse
import json
import time
import statistics
import torch
from antstorch._grid_sample_3d import grid_sample_3d
from antstorch._grid_sample_3d_metal import grid_sample_3d_metal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',type=int,nargs=3,default=[189,233,197],metavar=('D','H','W'))
    parser.add_argument('--channels',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    if min(*args.shape,args.channels,args.repeats)<1: parser.error('dimensions and repeats must be positive')
    torch.manual_seed(123)
    shape=tuple(args.shape)
    image=torch.rand(1,args.channels,*shape,device='mps')
    z,y,x=torch.meshgrid(*[torch.linspace(-1,1,s,device='mps') for s in shape],indexing='ij')
    grid=torch.stack((x,y,z),-1)[None]
    grid=grid+torch.randn_like(grid)*0.002
    options=dict(padding_mode='border',align_corners=True)
    def fallback(a,b):
        return torch.nn.functional.grid_sample(a.cpu(),b.cpu(),**options).to('mps')
    methods={'cpu_fallback':fallback,
             'torch_portable':lambda a,b:grid_sample_3d(a,b,**options),
             'metal':lambda a,b:grid_sample_3d_metal(a,b,**options)}
    reference=fallback(image,grid).cpu()
    report={'torch':torch.__version__,'shape':shape,'channels':args.channels,'repeats':args.repeats,'methods':{}}
    with torch.no_grad():
        for name,fn in methods.items():
            out=fn(image,grid);torch.mps.synchronize()
            difference=(out.cpu()-reference).double()
            times=[]
            for _ in range(args.repeats):
                torch.mps.synchronize();start=time.perf_counter()
                out=fn(image,grid);torch.mps.synchronize()
                times.append(time.perf_counter()-start)
            result={'median_seconds':statistics.median(times),'times_seconds':times,
                    'max_abs_error':difference.abs().max().item(),
                    'rmse':difference.square().mean().sqrt().item()}
            report['methods'][name]=result
            print(name,json.dumps(result),flush=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
