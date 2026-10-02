#!/usr/bin/env python3
"""Micro-benchmark + equivalence check for antstorch's in-loop field inversion.

Compares, on synthetic smooth displacement fields shaped like the SyN loop's
(zero on the boundary rim, small perturbation of a previously inverted field,
steps=in_loop_inverse_steps=6):

  anderson   -- the library's Anderson-accelerated inversion (baseline)
  fixed      -- plain ITK fixed-point, same number of steps (is Anderson worth
                its extra cost at steps=6?)
(A vectorized Anderson rewrite was tried on 2026-09-30 and measured SLOWER on
MPS, 0.75-0.89x, so it was reverted.)

Reports time (device-synchronized, median of --repeats), speedup vs anderson,
max |x - anderson| (same device), and the identity error
|| v(x) + W(x + v(x)) || (mean / max, interior voxels) of each result.

Run when the Mac/GPU is otherwise idle:
    python3 bench_inverse_field.py --device mps
    python3 bench_inverse_field.py --device cuda:1 --shapes 48,56,48 96,112,96 182,218,182
"""
import argparse, os, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/Pkg/ANTsTorch"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from antstorch.syn.core import inverse as I  # noqa: E402
from antstorch.syn.core.smoothing import get_boundary_mask  # noqa: E402


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def smooth_field(shape, amp_mm, device, gen):
    coarse = torch.randn((1, 3, 6, 7, 6), generator=gen)
    f = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    f = f / f.abs().max() * amp_mm
    f = f.permute(0, 2, 3, 4, 1).contiguous().to(device)
    return f * get_boundary_mask(shape, device, f.dtype)


def timeit(fn, dev, repeats):
    fn(); sync(dev)  # warm-up (MPS kernel compilation, allocator)
    ts = []
    for _ in range(repeats):
        sync(dev); t0 = time.perf_counter(); out = fn(); sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--shapes", nargs="+", default=["46,55,46", "91,109,91", "182,218,182"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--amp", type=float, default=3.0, help="forward field amplitude (mm)")
    ap.add_argument("--pert", type=float, default=0.3, help="per-iteration perturbation (mm)")
    ap.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    a = ap.parse_args()

    dev = torch.device(a.device)
    gen = torch.Generator().manual_seed(0)
    kw = dict(spacing=tuple(a.spacing), origin=(0.0, 0.0, 0.0), direction=np.eye(3).flatten())

    for s in a.shapes:
        shape = tuple(int(x) for x in s.split(","))
        W0 = smooth_field(shape, a.amp, dev, gen)
        # A previously converged inverse, as the SyN loop warm-starts from.
        inv0 = I.update_inverse_field_nd(W0, -W0, steps=30, method="fixed_point", **kw)
        W1 = W0 + smooth_field(shape, a.pert, dev, gen)
        print(f"\n== shape {shape} ({np.prod(shape)/1e6:.1f} Mvox), steps={a.steps}, device={dev} ==")

        runs = {
            "anderson": lambda: I.update_inverse_field_nd_anderson(W1, inv0, steps=a.steps, **kw),
            "fixed": lambda: I.update_inverse_field_nd(W1, inv0, steps=a.steps, method="fixed_point", **kw),
        }
        res, t_ref = {}, None
        print(f"{'method':10s} {'time s':>8s} {'speedup':>8s} {'max|x-and.|':>11s} {'id.err mean':>12s} {'id.err max':>11s}")
        for name, fn in runs.items():
            t, out = timeit(fn, dev, a.repeats)
            res[name] = out
            if name == "anderson":
                t_ref = t
            err = I.calculate_inverse_identity_error(W1, out, tuple(a.spacing), kw["origin"], np.eye(3))
            d = (out - res["anderson"]).abs().max().item()
            print(f"{name:10s} {t:8.3f} {t_ref / t:8.2f} {d:11.2e} {err['mean_error']:12.3e} {err['max_error']:11.3e}")
        del W0, W1, inv0, res
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
