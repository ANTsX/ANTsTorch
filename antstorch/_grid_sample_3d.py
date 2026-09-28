"""Differentiable 3-D sampling using portable Torch tensor operations.

Experimental: chunking bounds forward temporaries, not autograd saved tensors.
Coordinate order, padding and align_corners follow torch.grid_sample.
"""
import itertools
import torch


def grid_sample_3d(input, grid, mode="bilinear", padding_mode="zeros",
                   align_corners=False, *, chunk_size=65536):
    """Sample NCDHW data; supports gradients of input and sampling coordinates."""
    if input.ndim != 5 or grid.ndim != 5 or grid.shape[-1] != 3:
        raise ValueError("expected input NCDHW and grid NDHW3")
    if input.shape[0] != grid.shape[0] or input.device != grid.device or input.dtype != grid.dtype:
        raise ValueError("input/grid batch, device and dtype must agree")
    if not input.is_floating_point() or min(input.shape[2:]) < 1:
        raise ValueError("expected floating input with nonempty spatial dimensions")
    if mode not in ("bilinear", "nearest") or padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("unsupported interpolation or padding mode")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    n, c, d, h, w = input.shape
    flat = input.reshape(n, c, -1)
    points = grid.reshape(n, -1, 3)
    outputs = []
    for start in range(0, points.shape[1], chunk_size):
        coordinates = []
        for axis, size in enumerate((w, h, d)):
            raw = points[:, start:start + chunk_size, axis]
            raw = torch.where(torch.isnan(raw), -torch.ones_like(raw), raw)
            x = (raw + 1) * (size - 1) / 2 if align_corners else ((raw + 1) * size - 1) / 2
            if padding_mode == "reflection":
                low, high = (0., float(size - 1)) if align_corners else (-0.5, size - 0.5)
                span = high - low
                if span == 0:
                    x = x * 0
                else:
                    distance = (x - low).abs()
                    remainder = distance.remainder(span)
                    x = torch.where((torch.floor(distance / span).remainder(2)) == 0,
                                    remainder, span - remainder) + low
            if padding_mode != "zeros":
                # Explicit inequalities give zero coordinate derivative at edges,
                # as in native grid_sample's clipping operation.
                x = torch.where(x <= 0, torch.zeros_like(x),
                                torch.where(x >= size - 1, torch.full_like(x, size - 1), x))
            coordinates.append(x)
        if mode == "nearest":
            bases = [x.round() for x in coordinates]
            corners = [(0, 0, 0)]
        else:
            bases = [x.floor() for x in coordinates]
            corners = itertools.product((0, 1), repeat=3)
        value = None
        for offsets in corners:
            indices = [base + offset for base, offset in zip(bases, offsets)]
            valid = torch.ones_like(indices[0], dtype=torch.bool)
            for index, size in zip(indices, (w, h, d)):
                valid = valid & (index >= 0) & (index < size)
            ix, iy, iz = [index.clamp(0, size - 1).long()
                          for index, size in zip(indices, (w, h, d))]
            linear = ix + w * (iy + h * iz)
            sampled = torch.gather(flat, 2, linear[:, None].expand(-1, c, -1))
            weight = valid.to(input.dtype)
            if mode == "bilinear":
                for x, base, offset in zip(coordinates, bases, offsets):
                    fraction = x - base
                    weight = weight * (fraction if offset else 1 - fraction)
            term = sampled * weight[:, None]
            value = term if value is None else value + term
        # Native nearest sampling returns zero (not unused) coordinate gradients.
        if mode == "nearest":
            value = value + sum(coordinates)[:, None] * 0
        outputs.append(value)
    if not outputs:
        return input.new_empty((n, c, *grid.shape[1:4])) + (input.sum() + grid.sum()) * 0
    return torch.cat(outputs, dim=2).reshape(n, c, *grid.shape[1:4])
