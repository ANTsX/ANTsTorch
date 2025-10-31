# tests/test_image_dataset_and_scheduler.py
import random
from typing import List

import numpy as np
import pytest
import torch

import antstorch
import ants
from torch.utils.data import DataLoader

# # Quick run:
# pytest -q tests/test_image_dataset_and_scheduler.py -vv
#
# # Save results:
# pytest -q -s tests/test_image_dataset_and_scheduler.py -vv \
#   --aug-schedules "noise_std:cos:0.1->0.00@10,sd_affine:cos:0.05->0.00@10,sd_deformation:cos:15.0->0.0@10,sd_simulated_bias_field:cos:0.5->0.0@10,sd_histogram_warping:cos:0.05->0.0@10" \
#   --aug-steps 6 --dump-aug-samples --grid 10 --tile-size 128 --preview-channel 0



# -----------------------------
# Robust CLI option getter
# -----------------------------
def _opt(request, name: str, default):
    try:
        return request.config.getoption(name, default=default)
    except Exception:
        return default


# -----------------------------
# Data helpers
# -----------------------------
def load_hcpya_slices(mods: List[str], H: int, W: int, slice_idx: int = 120):
    keys = dict(T2="hcpyaT2Template", T1="hcpyaT1Template", FA="hcpyaFATemplate")
    imgs = [ants.image_read(antstorch.get_antstorch_data(keys[m])) for m in mods]
    slcs = [ants.slice_image(im, axis=2, idx=slice_idx, collapse_strategy=1) for im in imgs]
    tmpl = ants.resample_image(slcs[0], (H, W), use_voxels=True)
    return slcs, tmpl


def _normalize01(x: np.ndarray) -> np.ndarray:
    """Per-image robust min-max to [0,1] using 1–99%."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return x


def _to_images_from_batch(bchw: torch.Tensor, channel: int = 0):
    """Split a (B,C,H,W) tensor into a list of 2D float32 images in [0,1]."""
    x = bchw.detach().cpu().numpy()
    imgs = [x[i, channel] for i in range(x.shape[0])]
    return [_normalize01(im) for im in imgs]


def _resample_to_size(im01: np.ndarray, tile_size: int) -> np.ndarray:
    """Resample a [0,1] image to (tile_size, tile_size) via ANTs linear."""
    if im01.shape == (tile_size, tile_size):
        return im01
    ref = ants.from_numpy(np.zeros((tile_size, tile_size), dtype=np.float32))
    a = ants.from_numpy(im01.astype(np.float32))
    r = ants.resample_image_to_target(a, ref, interp_type="linear")
    return r.numpy().astype(np.float32)


def _save_mosaic_png(images01, rows: int, cols: int, tile_size: int, out_path: str):
    """Tile per-image [0,1] arrays into rows×cols, save PNG (uint8)."""
    tiles = [(_resample_to_size(im, tile_size) if im.shape != (tile_size, tile_size) else im)
             for im in images01]

    mosaic = np.zeros((rows * tile_size, cols * tile_size), dtype=np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(tiles):
                break
            rr, cc = r * tile_size, c * tile_size
            mosaic[rr:rr + tile_size, cc:cc + tile_size] = tiles[k]
            k += 1

    # [0,1] -> uint8
    mosaic_u8 = np.clip(mosaic * 255.0 + 0.5, 0, 255).astype(np.uint8)

    try:
        from PIL import Image
        Image.fromarray(mosaic_u8, mode="L").save(out_path)
        return
    except Exception:
        pass
    ants.image_write(ants.from_numpy(mosaic_u8), out_path)


# -----------------------------
# Test
# -----------------------------
def test_image_dataset_with_scheduler_integration(tmp_path, request):
    # General knobs
    dump = bool(_opt(request, "--dump-aug-samples", False))
    grid = int(_opt(request, "--grid", 10))              # rows=cols
    tile = int(_opt(request, "--tile-size", 128))        # tile (and dataset) size
    channel_to_show = int(_opt(request, "--preview-channel", 0))
    mods = [m.strip() for m in str(_opt(request, "--mods", "T2")).split(",") if m.strip()]

    # Anneal spec (DSL) like the trainer; if empty, we synthesize a tiny default
    aug_spec = str(_opt(request, "--aug-schedules", "")).strip()

    # If user provided a spec, we’ll parse it and infer a nice loop length.
    # If not, fall back to a tiny cosine default driven by --aug-steps (default 4).
    aug_steps_opt = int(_opt(request, "--aug-steps", 4))

    if aug_spec:
        schedules = antstorch.parse_schedules(aug_spec)
        # Derive a reasonable loop length; keep tests snappy unless user overrides --aug-steps
        implied = max(max(1, s.total_steps) + max(0, s.delay) for s in schedules)
        steps = aug_steps_opt if aug_steps_opt > 0 else min(implied, 12)  # cap to keep CI quick
    else:
        steps = max(1, aug_steps_opt)
        schedules = antstorch.parse_schedules(
            ",".join([
                f"noise_std:cos:0.02->0.00@{steps}",
                f"sd_affine:cos:0.02->0.00@{steps}",
                f"sd_deformation:cos:1.00->0.00@{steps}",
                f"sd_simulated_bias_field:cos:1e-8->0.00@{steps}",
                f"sd_histogram_warping:cos:0.02->0.00@{steps}",
            ])
        )

    mp = antstorch.MultiParamScheduler(schedules)

    np.random.seed(1234); random.seed(1234); torch.manual_seed(1234)

    # Build data (H=W=tile to match mosaic tiles)
    H = W = tile
    slcs, tmpl = load_hcpya_slices(mods, H, W)

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
    it = iter(loader)

    prev_vals = None
    for step in range(steps):
        vals = mp.step(step)

        # Drive ANTs knobs
        ds.data_augmentation_sd_affine = float(vals.get("sd_affine", 0.0))
        ds.data_augmentation_sd_deformation = float(vals.get("sd_deformation", 0.0))
        ds.data_augmentation_sd_simulated_bias_field = float(vals.get("sd_simulated_bias_field", 0.0))
        ds.data_augmentation_sd_histogram_warping = float(vals.get("sd_histogram_warping", 0.0))
        ds.data_augmentation_noise_parameters = (0.0, float(vals.get("noise_std", 0.0)))

        batch = next(it)  # (B,C,H,W)
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

            aug_vals = {
                "noise_std": float(
                    ds.data_augmentation_noise_parameters[1]
                    if isinstance(ds.data_augmentation_noise_parameters, (tuple, list))
                    else ds.data_augmentation_noise_parameters
                ),
                "sd_affine": float(ds.data_augmentation_sd_affine),
                "sd_deformation": float(ds.data_augmentation_sd_deformation),
                "sd_simulated_bias_field": float(ds.data_augmentation_sd_simulated_bias_field),
                "sd_histogram_warping": float(ds.data_augmentation_sd_histogram_warping),
            }

            # Nice, parseable one-liner
            kv = ", ".join(f"{k}={v:.6g}" for k, v in aug_vals.items())
            print(f"[dump] step={step:03d} -> {out_path} | {kv}")

        # Non-increasing toward targets for common decay schedules
        if prev_vals is not None:
            for k in ("sd_affine", "sd_deformation", "sd_histogram_warping", "noise_std"):
                if k in vals and k in prev_vals:
                    assert vals[k] <= prev_vals[k] + 1e-12
        prev_vals = vals

    # Endpoint consistency: compare against schedule-defined value at the same step
    # (works even if we capped steps for speed or the user provided delays/floors)
    end_vals = mp.step(steps)
    for s in schedules:
        expect = s.value_at(steps)  # authoritative
        got = end_vals.get(s.name)
        if got is not None:
            assert got == pytest.approx(expect, rel=1e-7, abs=1e-9)
