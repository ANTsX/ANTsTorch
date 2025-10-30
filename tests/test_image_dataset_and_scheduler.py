# tests/test_image_dataset_and_scheduler.py
import math
import random
from typing import List

import numpy as np
import pytest
import torch

import antstorch
import ants
from torch.utils.data import DataLoader


def _opt(request, name: str, default):
    try:
        return request.config.getoption(name, default=default)
    except Exception:
        return default


def load_hcpya_slices(mods: List[str], H: int, W: int, slice_idx: int = 120):
    keys = dict(T2="hcpyaT2Template", T1="hcpyaT1Template", FA="hcpyaFATemplate")
    imgs = [ants.image_read(antstorch.get_antstorch_data(keys[m])) for m in mods]
    slcs = [ants.slice_image(im, axis=2, idx=slice_idx, collapse_strategy=1) for im in imgs]
    tmpl = ants.resample_image(slcs[0], (H, W), use_voxels=True)
    return slcs, tmpl


def _normalize01(x: np.ndarray) -> np.ndarray:
    """Per-image robust min-max to [0,1] using 1-99% percentiles."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return x


def _to_images_from_batch(bchw: torch.Tensor, channel: int = 0) -> List[np.ndarray]:
    """Split a (B,C,H,W) tensor into a list of 2D float32 images in [0,1]."""
    x = bchw.detach().cpu().numpy()
    imgs = [x[i, channel] for i in range(x.shape[0])]
    return [_normalize01(im) for im in imgs]


def _resample_to_size(im01: np.ndarray, tile_size: int) -> np.ndarray:
    """Resample a [0,1] image to (tile_size, tile_size) using ANTs (linear)."""
    if im01.shape == (tile_size, tile_size):
        return im01
    ref = ants.from_numpy(np.zeros((tile_size, tile_size), dtype=np.float32))
    a = ants.from_numpy(im01.astype(np.float32))
    r = ants.resample_image_to_target(a, ref, interp_type="linear")
    return r.numpy().astype(np.float32)


def _save_mosaic_png(images01: List[np.ndarray], rows: int, cols: int,
                     tile_size: int, out_path: str):
    """
    Tile per-image [0,1] arrays into a rows×cols mosaic and save as PNG (uint8).
    Falls back to ANTs writer if Pillow is unavailable.
    """
    # Resample tiles if needed
    tiles = [(_resample_to_size(im, tile_size) if im.shape != (tile_size, tile_size) else im)
             for im in images01]

    # Assemble float mosaic in [0,1]
    mosaic = np.zeros((rows * tile_size, cols * tile_size), dtype=np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(tiles):
                break
            rr = r * tile_size
            cc = c * tile_size
            mosaic[rr:rr + tile_size, cc:cc + tile_size] = tiles[k]
            k += 1

    # Convert to uint8 [0,255] for PNG
    mosaic_u8 = np.clip(mosaic * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Prefer Pillow (PNG-native, robust)
    try:
        from PIL import Image
        Image.fromarray(mosaic_u8, mode='L').save(out_path)
        return
    except Exception:
        pass  # fall back to ANTs

    # Fallback: ANTs writer (must be uint8/uint16 for PNG)
    ants.image_write(ants.from_numpy(mosaic_u8), out_path)


def test_image_dataset_with_scheduler_integration(tmp_path, request):
    dump = bool(_opt(request, "--dump-aug-samples", False))
    steps = int(_opt(request, "--aug-steps", 4))
    mods = [m.strip() for m in str(_opt(request, "--mods", "T1")).split(",") if m.strip()]
    grid = int(_opt(request, "--grid", 10))           # rows=cols=grid
    tile = int(_opt(request, "--tile-size", 128))     # tile (and dataset) size
    channel_to_show = int(_opt(request, "--preview-channel", 0))

    np.random.seed(1234); random.seed(1234); torch.manual_seed(1234)

    H = W = tile
    slcs, tmpl = load_hcpya_slices(mods, H, W)

    # Ensure enough samples if dumping full mosaics (grid^2 per step)
    tiles_needed = grid * grid
    n_samples = steps * tiles_needed if dump else max(steps * 2, 8)

    ds = antstorch.ImageDataset(
        images=[slcs],
        template=tmpl,
        do_data_augmentation=True,
        data_augmentation_transform_type="affineAndDeformation",
        data_augmentation_sd_affine=0.02,
        data_augmentation_sd_deformation=1.0,
        data_augmentation_noise_model="additivegaussian",
        data_augmentation_noise_parameters=(0.0, 0.02),
        data_augmentation_sd_simulated_bias_field=1e-8,
        data_augmentation_sd_histogram_warping=0.02,
        number_of_samples=int(n_samples),
    )

    batch_size = min(32, tiles_needed) if dump else 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Linear anneal to 0 for all five ANTs knobs over 'steps'
    spec = (
        f"noise_std:linear:0.02->0.00@{steps},"
        f"sd_affine:linear:0.02->0.00@{steps},"
        f"sd_deformation:linear:1.00->0.00@{steps},"
        f"sd_simulated_bias_field:linear:1e-8->0.00@{steps},"
        f"sd_histogram_warping:linear:0.02->0.00@{steps}"
    )
    mp = antstorch.MultiParamScheduler(antstorch.parse_schedules(spec))

    it = iter(loader)
    prev_vals = None

    for step in range(steps):
        vals = mp.step(step)

        # Drive the dataset's ANTs knobs each step
        ds.data_augmentation_sd_affine = float(vals["sd_affine"])
        ds.data_augmentation_sd_deformation = float(vals["sd_deformation"])
        ds.data_augmentation_sd_simulated_bias_field = float(vals["sd_simulated_bias_field"])
        ds.data_augmentation_sd_histogram_warping = float(vals["sd_histogram_warping"])
        ds.data_augmentation_noise_parameters = (0.0, float(vals["noise_std"]))

        batch = next(it)
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 4 and batch.shape[1] == len(mods)
        assert batch.shape[2:] == (H, W)

        if dump:
            # Gather exactly grid^2 images of the chosen channel
            imgs = _to_images_from_batch(batch, channel=channel_to_show)
            while len(imgs) < tiles_needed:
                try:
                    more = next(it)
                except StopIteration:
                    it = iter(loader)
                    more = next(it)
                imgs.extend(_to_images_from_batch(more, channel=channel_to_show))
            imgs = imgs[:tiles_needed]

            out_path = tmp_path / f"aug_step{step:03d}.png"
            _save_mosaic_png(imgs, rows=grid, cols=grid, tile_size=tile, out_path=str(out_path))
            print(f"[dump] saved mosaic: {out_path}")

        # monotone towards 0.0
        if prev_vals is not None:
            assert vals["sd_affine"] <= prev_vals["sd_affine"] + 1e-12
            assert vals["sd_deformation"] <= prev_vals["sd_deformation"] + 1e-12
            assert vals["sd_histogram_warping"] <= prev_vals["sd_histogram_warping"] + 1e-12
            assert vals["noise_std"] <= prev_vals["noise_std"] + 1e-12
        prev_vals = vals

    end_vals = mp.step(steps)
    for k in ("sd_affine", "sd_deformation", "sd_simulated_bias_field", "sd_histogram_warping", "noise_std"):
        assert end_vals[k] == pytest.approx(0.0, abs=1e-12)
