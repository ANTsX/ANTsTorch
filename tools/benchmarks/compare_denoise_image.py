#!/usr/bin/env python3
"""Compare ANTs and ANTsTorch adaptive non-local means denoising.

Both ``ants.denoise_image`` and ``antstorch.denoise_image`` receive the same
ANTsImage, mask and parameters. The script reports runtimes, agreement
metrics (RMSE, MAE, maximum difference, correlation) and writes the denoised
images, the removed-noise images and the ANTsTorch - ANTs difference.

ITK's multi-threaded denoiser accumulates into overlapping patches without
a fixed order, so repeated ANTs runs are not bit-identical. With
``--repeats 2`` or more, the ANTs run-to-run difference is also reported; it
is the floor below which ANTs/ANTsTorch differences are not meaningful.
``--ants-threads 1`` makes the ANTs reference deterministic.

Examples
--------
Use the bundled 2-D ``r16`` image::

    python tools/benchmarks/compare_denoise_image.py

Gaussian noise model, larger search radius, MPS::

    python tools/benchmarks/compare_denoise_image.py --noise-model Gaussian -r 3 --device mps

A 3-D image with a foreground mask, shrink factor and added Rician noise::

    python tools/benchmarks/compare_denoise_image.py image.nii.gz --auto-mask \\
        --shrink-factor 2 --add-noise 0.05 --output-dir results/denoise

Deterministic single-threaded ANTs reference with timing over 5 runs::

    python tools/benchmarks/compare_denoise_image.py --ants-threads 1 --repeats 5
"""

import argparse
import base64
import datetime
import io
import os
import time
from pathlib import Path

import numpy as np



def synchronize(device) -> None:
    """Wait for asynchronous accelerator work before timing boundaries."""
    import torch

    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_cuda_peak_memory(device) -> bool:
    """Reset optional CUDA memory statistics without blocking the benchmark."""
    import torch

    try:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    except (RuntimeError, TypeError) as error:
        print(
            "WARNING: CUDA peak-memory statistics are unavailable; "
            f"continuing without them ({error})"
        )
        return False
    return True


def cuda_peak_memory(device):
    """Read optional peak memory from the selected CUDA device."""
    import torch

    try:
        with torch.cuda.device(device):
            return torch.cuda.max_memory_allocated()
    except (RuntimeError, TypeError) as error:
        print(f"WARNING: Could not read CUDA peak-memory statistics ({error})")
        return None


def parse_radius(value: str):
    """Accept ``2``, ``2x2`` or ``2x2x1`` like ANTsPy."""
    return value if "x" in value else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Input 2-D or 3-D image. When omitted, ants.get_ants_data('r16') is used.",
    )
    default_device = os.environ.get("DEVICE", "cpu")
    parser.add_argument(
        "--device",
        default=default_device,
        help=f"PyTorch device: cpu, cuda or mps (default: DEVICE env var or cpu, currently '{default_device}')",
    )
    parser.add_argument("--noise-model", choices=("Rician", "Gaussian"), default="Rician")
    parser.add_argument("-p", "--patch-radius", type=parse_radius, default=1,
                        help="Patch radius, e.g. 1 or 1x1x1 (default: 1)")
    parser.add_argument("-r", "--search-radius", type=parse_radius, default=2,
                        help="Search radius, e.g. 2 or 2x2x2 (default: 2)")
    parser.add_argument("--shrink-factor", type=int, default=1)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--mask", type=Path, help="Mask image in the input's physical space")
    mask_group.add_argument("--auto-mask", action="store_true",
                            help="Use ants.get_mask(image) as the mask")
    parser.add_argument(
        "--add-noise",
        type=float,
        default=0.0,
        help=(
            "Add synthetic noise before denoising, with standard deviation given "
            "as a fraction of the maximum intensity (Rician or Gaussian, following "
            "--noise-model). Default: 0, no added noise."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --add-noise")
    parser.add_argument(
        "--ants-threads",
        type=int,
        default=None,
        help="ITK thread count for ANTs (1 gives a deterministic reference). Default: ITK default.",
    )
    parser.add_argument("--repeats", type=int, default=1,
                        help="Timed runs per implementation; the median is reported (default: 1)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the lightweight pipeline warm-up run")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for output images; created if needed (default: current directory)")
    parser.add_argument("--output-prefix", default="denoise_comparison")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip generating an HTML comparison report")
    return parser.parse_args()


def add_noise(image, fraction: float, noise_model: str, seed: int):
    import ants

    rng = np.random.default_rng(seed)
    array = image.numpy().astype(np.float64)
    sigma = fraction * float(array.max())
    if noise_model == "Rician":
        real = array + rng.normal(0.0, sigma, array.shape)
        imaginary = rng.normal(0.0, sigma, array.shape)
        noisy = np.sqrt(real**2 + imaginary**2)
    else:
        noisy = array + rng.normal(0.0, sigma, array.shape)
    return ants.from_numpy(
        noisy.astype(np.float32),
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    ), sigma


def metrics(reference: np.ndarray, test: np.ndarray, region: np.ndarray) -> dict:
    ref = reference[region]
    tst = test[region]
    diff = tst - ref
    intensity_range = float(ref.max() - ref.min()) or 1.0
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {
        "rmse": rmse,
        "relative_rmse": rmse / intensity_range,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "correlation": float(np.corrcoef(ref.ravel(), tst.ravel())[0, 1]),
    }


def print_metrics(label: str, values: dict) -> None:
    print(f"{label}:")
    print(f"  RMSE:            {values['rmse']:.6g} ({100 * values['relative_rmse']:.4f} % of range)")
    print(f"  MAE:             {values['mae']:.6g}")
    print(f"  Max |diff|:      {values['max_abs']:.6g}")
    print(f"  Correlation:     {values['correlation']:.10f}")


def warmup_pipeline(device, dimension: int, noise_model: str, verbose: bool = False):
    """Prime ITK thread pool and accelerator kernel/allocator caches using a tiny dummy grid.

    Returns (ants_warmup_seconds, torch_warmup_seconds).
    """
    import ants
    import antstorch

    dummy_shape = (16, 16, 16) if dimension == 3 else (32, 32)
    dummy_arr = np.ones(dummy_shape, dtype=np.float32) * 100.0
    dummy_img = ants.from_numpy(dummy_arr)

    if verbose:
        print(f"Priming ANTs and {device.type.upper()} pipelines with lightweight dummy grid {dummy_shape}...")

    # ANTs warm-up: primes ITK global thread pool & filter structures
    start_ants = time.perf_counter()
    ants.denoise_image(dummy_img, p=1, r=1, noise_model=noise_model)
    ants_warmup_time = time.perf_counter() - start_ants

    # ANTsTorch warm-up: primes PyTorch JIT, kernel compilation & allocator
    synchronize(device)
    start_torch = time.perf_counter()
    antstorch.denoise_image(dummy_img, p=1, r=1, noise_model=noise_model, device=device)
    synchronize(device)
    torch_warmup_time = time.perf_counter() - start_torch

    if verbose:
        print(
            f"Pipeline warm-up complete: ANTs={ants_warmup_time:.4f} s, "
            f"ANTsTorch ({device})={torch_warmup_time:.4f} s"
        )
    return ants_warmup_time, torch_warmup_time


def format_run_times(times: list) -> str:
    """Format single run or multi-run timing breakdown."""
    if len(times) == 1:
        return f"{times[0]:.4f} s"
    steady = times[1:]
    return (
        f"{np.median(times):.4f} s median of {len(times)} "
        f"(run 1 [cold]: {times[0]:.4f} s, runs 2..{len(times)}: min {min(steady):.4f} s, max {max(steady):.4f} s)"
    )


def render_slice_b64(image, cmap: str = "gray", title: str = "", is_diff: bool = False) -> str:
    """Render 2D or 3D orthogonal montage into a base64 PNG string adhering to medical viewing rules."""
    import ants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image.dimension == 2:
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
        arr = image.numpy()
        aspect = float(image.spacing[1] / image.spacing[0])
        im = ax.imshow(arr.T, cmap=cmap, aspect=aspect, origin="lower")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        if is_diff:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # 3D: Reorient to standard RAI (Radiological convention)
    # Axial: A top, P bottom, Patient R on viewer L
    # Coronal: S top, I bottom, Patient R on viewer L
    # Sagittal: S top, I bottom, Anterior on viewer L
    image_rai = ants.reorient_image2(image, "RAI")
    arr = image_rai.numpy()
    sp = image_rai.spacing
    nx, ny, nz = arr.shape

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=120, layout="constrained")

    # 1. Axial (cut along z)
    axial_slice = arr[:, :, nz // 2].T
    axes[0].imshow(axial_slice, cmap=cmap, aspect=float(sp[1] / sp[0]), origin="upper")
    axes[0].set_title(f"{title} - Axial\n(R-L / A-P)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # 2. Coronal (cut along y)
    coronal_slice = arr[:, ny // 2, ::-1].T
    axes[1].imshow(coronal_slice, cmap=cmap, aspect=float(sp[2] / sp[0]), origin="upper")
    axes[1].set_title(f"{title} - Coronal\n(R-L / S-I)", fontsize=9, fontweight="bold")
    axes[1].axis("off")

    # 3. Sagittal (cut along x)
    sagittal_slice = arr[nx // 2, :, ::-1].T
    im2 = axes[2].imshow(sagittal_slice, cmap=cmap, aspect=float(sp[2] / sp[1]), origin="upper")
    axes[2].set_title(f"{title} - Sagittal\n(A-P / S-I)", fontsize=9, fontweight="bold")
    axes[2].axis("off")

    if is_diff:
        fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_html_report(
    report_path: Path,
    image,
    ants_denoised,
    torch_denoised,
    ants_noise,
    torch_noise,
    diff_img,
    ants_times: list,
    torch_times: list,
    ants_warmup,
    torch_warmup,
    device,
    threads,
    options: dict,
    input_path: str,
    denoised_metrics: dict,
    noise_metrics: dict,
    ants_spread=None,
    torch_spread=None,
    cuda_peak_mb=None,
) -> None:
    """Generate a comprehensive visual HTML report with embedded slice montages."""
    img_b64 = render_slice_b64(image, cmap="gray", title="Input Image")
    ants_den_b64 = render_slice_b64(ants_denoised, cmap="gray", title="ANTs Denoised")
    torch_den_b64 = render_slice_b64(torch_denoised, cmap="gray", title="ANTsTorch Denoised")
    ants_noise_b64 = render_slice_b64(ants_noise, cmap="gray", title="ANTs Removed Noise")
    torch_noise_b64 = render_slice_b64(torch_noise, cmap="gray", title="ANTsTorch Removed Noise")
    import ants

    diff_abs_img = ants.from_numpy(
        np.abs(diff_img.numpy()).astype(np.float32),
        origin=diff_img.origin,
        spacing=diff_img.spacing,
        direction=diff_img.direction,
    )
    diff_b64 = render_slice_b64(diff_abs_img, cmap="inferno", title="|ANTsTorch - ANTs|", is_diff=True)

    median_ants = float(np.median(ants_times))
    median_torch = float(np.median(torch_times))
    speedup = median_ants / median_torch if median_torch > 0 else 0.0

    ants_steady = ants_times[1:] if len(ants_times) > 1 else ants_times
    torch_steady = torch_times[1:] if len(torch_times) > 1 else torch_times
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ANTs vs ANTsTorch Denoise Benchmark Report</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-green: #4ade80;
    --accent-amber: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 20px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
  }}
  h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
  .subtitle {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .kpi-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 16px 20px;
  }}
  .kpi-label {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; color: #fff; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
  .section {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
  }}
  .section-title {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; color: var(--accent); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    text-align: left;
  }}
  th, td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--card-border);
  }}
  th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge-torch {{ background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }}
  .badge-ants {{ background: rgba(251, 191, 36, 0.15); color: var(--accent-amber); border: 1px solid rgba(251, 191, 36, 0.3); }}
  .visual-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 20px;
  }}
  .visual-card {{
    background: #151f32;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }}
  .visual-card img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    margin-top: 8px;
  }}
  .footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--card-border);
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Adaptive Non-Local Means Denoising Benchmark</h1>
      <div class="subtitle">ANTs (ITK) vs ANTsTorch Comparison Report &bull; {date_str}</div>
    </div>
    <div>
      <span class="badge badge-torch">Device: {device}</span>
      <span class="badge badge-ants">Threads: {threads}</span>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Speed Ratio</div>
      <div class="kpi-value" style="color: {'#4ade80' if speedup >= 1.0 else '#fbbf24'};">{speedup:.2f}x</div>
      <div class="kpi-sub">{'ANTsTorch faster' if speedup >= 1.0 else 'ANTs faster'}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">ANTsTorch Runtime</div>
      <div class="kpi-value">{median_torch:.3f} s</div>
      <div class="kpi-sub">median of {len(torch_times)} run(s)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">ANTs Runtime</div>
      <div class="kpi-value">{median_ants:.3f} s</div>
      <div class="kpi-sub">median of {len(ants_times)} run(s)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Pearson Correlation</div>
      <div class="kpi-value">{denoised_metrics['correlation']:.6f}</div>
      <div class="kpi-sub">Image agreement metric</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Relative RMSE</div>
      <div class="kpi-value">{100 * denoised_metrics['relative_rmse']:.3f} %</div>
      <div class="kpi-sub">RMSE: {denoised_metrics['rmse']:.4g}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Performance &amp; Timing Breakdown</div>
    <table>
      <thead>
        <tr>
          <th>Implementation</th>
          <th>Backend</th>
          <th>Warm-up (dummy grid)</th>
          <th>Run 1 (Cold)</th>
          <th>Runs 2..{max(len(ants_times), 2)} (Steady)</th>
          <th>Median Runtime</th>
          <th>Speedup</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="badge badge-ants">ANTs (ITK)</span></td>
          <td>CPU ({threads} threads)</td>
          <td>{f"{ants_warmup:.4f} s" if ants_warmup is not None else "skipped"}</td>
          <td>{ants_times[0]:.4f} s</td>
          <td>{f"{min(ants_steady):.4f} - {max(ants_steady):.4f} s" if len(ants_times) > 1 else "N/A"}</td>
          <td><strong>{median_ants:.4f} s</strong></td>
          <td>1.00x</td>
        </tr>
        <tr>
          <td><span class="badge badge-torch">ANTsTorch</span></td>
          <td>PyTorch ({device})</td>
          <td>{f"{torch_warmup:.4f} s" if torch_warmup is not None else "skipped"}</td>
          <td>{torch_times[0]:.4f} s</td>
          <td>{f"{min(torch_steady):.4f} - {max(torch_steady):.4f} s" if len(torch_times) > 1 else "N/A"}</td>
          <td><strong>{median_torch:.4f} s</strong></td>
          <td><strong style="color: {'#4ade80' if speedup >= 1.0 else '#fbbf24'};">{speedup:.2f}x</strong></td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Accuracy &amp; Parity Metrics</div>
    <table>
      <thead>
        <tr>
          <th>Comparison Target</th>
          <th>RMSE</th>
          <th>Relative RMSE (% range)</th>
          <th>MAE</th>
          <th>Max |Diff|</th>
          <th>Pearson Correlation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Denoised Image</strong></td>
          <td>{denoised_metrics['rmse']:.6g}</td>
          <td>{100 * denoised_metrics['relative_rmse']:.4f} %</td>
          <td>{denoised_metrics['mae']:.6g}</td>
          <td>{denoised_metrics['max_abs']:.6g}</td>
          <td>{denoised_metrics['correlation']:.8f}</td>
        </tr>
        <tr>
          <td><strong>Removed Noise Residual</strong></td>
          <td>{noise_metrics['rmse']:.6g}</td>
          <td>{100 * noise_metrics['relative_rmse']:.4f} %</td>
          <td>{noise_metrics['mae']:.6g}</td>
          <td>{noise_metrics['max_abs']:.6g}</td>
          <td>{noise_metrics['correlation']:.8f}</td>
        </tr>
        {f'''<tr>
          <td><strong>Run-to-Run Spread (Repeats)</strong></td>
          <td colspan="5">ANTs max run-to-run diff: <code>{ants_spread:.6g}</code> &bull; ANTsTorch max run-to-run diff: <code>{torch_spread:.6g}</code></td>
        </tr>''' if ants_spread is not None and torch_spread is not None else ''}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Visual Inspection (Physical Orientation Preserved)</div>
    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
      Adheres to radiological viewing conventions: Axial (A at top, P at bottom, Patient R on viewer L), Coronal (S at top, I at bottom, Patient R on viewer L), Sagittal (S at top, I at bottom, Anterior on viewer L). True anatomical aspect ratios strictly enforced.
    </div>
    <div class="visual-grid">
      <div class="visual-card">
        <div><strong>Input Image</strong></div>
        <img src="data:image/png;base64,{img_b64}" alt="Input Image">
      </div>
      <div class="visual-card">
        <div><strong>ANTs Denoised</strong></div>
        <img src="data:image/png;base64,{ants_den_b64}" alt="ANTs Denoised">
      </div>
      <div class="visual-card">
        <div><strong>ANTsTorch Denoised</strong></div>
        <img src="data:image/png;base64,{torch_den_b64}" alt="ANTsTorch Denoised">
      </div>
      <div class="visual-card">
        <div><strong>|ANTsTorch - ANTs| Absolute Difference</strong></div>
        <img src="data:image/png;base64,{diff_b64}" alt="Absolute Difference">
      </div>
      <div class="visual-card">
        <div><strong>ANTs Removed Noise</strong></div>
        <img src="data:image/png;base64,{ants_noise_b64}" alt="ANTs Removed Noise">
      </div>
      <div class="visual-card">
        <div><strong>ANTsTorch Removed Noise</strong></div>
        <img src="data:image/png;base64,{torch_noise_b64}" alt="ANTsTorch Removed Noise">
      </div>
    </div>
  </div>

  <div class="footer">
    Input: <code>{input_path}</code> &bull; Size: {image.shape} &bull; Spacing: {image.spacing} &bull; Noise model: {options['noise_model']} &bull; Radius: p={options['p']}, r={options['r']}
  </div>
</div>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    args = parse_args()
    if args.ants_threads is not None:
        # ITK reads this when the first filter is created; set before importing ants.
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.ants_threads)

    import ants
    import torch

    import antstorch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    input_path = args.image or ants.get_ants_data("r16")
    image = ants.image_read(input_path).clone("float")
    if image.dimension not in (2, 3) or image.components != 1:
        raise ValueError("This comparison expects a scalar 2-D or 3-D image")

    sigma = None
    if args.add_noise > 0:
        image, sigma = add_noise(image, args.add_noise, args.noise_model, args.seed)

    mask = None
    if args.mask is not None:
        mask = ants.image_read(str(args.mask))
        if mask.shape != image.shape or not ants.image_physical_space_consistency(image, mask):
            raise ValueError("--mask must occupy the same physical space as the image")
    elif args.auto_mask:
        mask = ants.get_mask(image)
    region = mask.numpy() > 0 if mask is not None else np.ones(image.shape, dtype=bool)

    options = {
        "shrink_factor": args.shrink_factor,
        "p": args.patch_radius,
        "r": args.search_radius,
        "noise_model": args.noise_model,
    }

    # Lightweight symmetric warm-up (primes ITK thread pools & GPU kernel/allocator caches)
    ants_warmup_time, torch_warmup_time = None, None
    if not args.no_warmup:
        ants_warmup_time, torch_warmup_time = warmup_pipeline(
            device, image.dimension, args.noise_model, verbose=args.verbose
        )

    # ANTs reference runs
    ants_results, ants_times = [], []
    for run in range(args.repeats):
        if args.verbose:
            print(f"Running ANTs denoise_image ({run + 1}/{args.repeats})...")
        start = time.perf_counter()
        ants_results.append(ants.denoise_image(image, mask=mask, v=int(args.verbose), **options))
        ants_times.append(time.perf_counter() - start)

    # ANTsTorch runs
    track_cuda_memory = device.type == "cuda" and reset_cuda_peak_memory(device)
    torch_results, torch_times = [], []
    for run in range(args.repeats):
        if args.verbose:
            print(f"Running ANTsTorch denoise_image ({run + 1}/{args.repeats})...")
        synchronize(device)
        start = time.perf_counter()
        torch_results.append(
            antstorch.denoise_image(image, mask=mask, device=device, verbose=args.verbose, **options)
        )
        synchronize(device)
        torch_times.append(time.perf_counter() - start)

    input_array = image.numpy().astype(np.float64)
    ants_array = ants_results[0].numpy().astype(np.float64)
    torch_array = torch_results[0].numpy().astype(np.float64)
    if not ants.image_physical_space_consistency(image, torch_results[0]):
        print("WARNING: ANTsTorch output geometry differs from the input geometry")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.output_prefix
    if sigma is not None:
        ants.image_write(image, f"{prefix}_input_noisy.nii.gz")
    ants.image_write(ants_results[0], f"{prefix}_ants_denoised.nii.gz")
    ants.image_write(torch_results[0], f"{prefix}_antstorch_denoised.nii.gz")
    ants.image_write(image - ants_results[0], f"{prefix}_ants_noise.nii.gz")
    ants.image_write(image - torch_results[0], f"{prefix}_antstorch_noise.nii.gz")
    diff_image = torch_results[0] - ants_results[0]
    ants.image_write(diff_image, f"{prefix}_difference.nii.gz")

    print(f"Input: {input_path}")
    print(f"Geometry: size={image.shape}, spacing={image.spacing}, origin={image.origin}")
    if sigma is not None:
        print(f"Added {args.noise_model} noise: sigma={sigma:.6g} ({args.add_noise:g} x max), seed={args.seed}")
    print(
        f"Parameters: noise_model={args.noise_model}, p={args.patch_radius}, "
        f"r={args.search_radius}, shrink_factor={args.shrink_factor}, "
        f"mask={'file' if args.mask else 'auto' if args.auto_mask else 'none'} "
        f"({int(region.sum())} voxels compared)"
    )
    if ants_warmup_time is not None and torch_warmup_time is not None:
        print(
            f"Warm-up (dummy grid):  ANTs: {ants_warmup_time:.4f} s | "
            f"ANTsTorch: {torch_warmup_time:.4f} s on {device}"
        )
    else:
        print("Warm-up:               skipped (--no-warmup)")

    threads = args.ants_threads if args.ants_threads is not None else "ITK default"
    print(f"ANTs runtime:          {format_run_times(ants_times)} (threads: {threads})")
    print(f"ANTsTorch runtime:     {format_run_times(torch_times)} on {device}")
    print(f"Speed ratio (ANTs / ANTsTorch): {np.median(ants_times) / np.median(torch_times):.2f}x")
    print(f"Input intensity range:     {input_array[region].min():.6g} to {input_array[region].max():.6g}")
    print(f"ANTs output range:         {ants_array[region].min():.6g} to {ants_array[region].max():.6g}")
    print(f"ANTsTorch output range:    {torch_array[region].min():.6g} to {torch_array[region].max():.6g}")
    print(f"Removed-noise std, ANTs:      {np.std((input_array - ants_array)[region]):.6g}")
    print(f"Removed-noise std, ANTsTorch: {np.std((input_array - torch_array)[region]):.6g}")

    denoised_metrics = metrics(ants_array, torch_array, region)
    noise_metrics = metrics(input_array - ants_array, input_array - torch_array, region)
    print_metrics("ANTsTorch vs ANTs (denoised image)", denoised_metrics)
    print_metrics("ANTsTorch vs ANTs (removed noise)", noise_metrics)

    ants_spread, torch_spread = None, None
    if args.repeats >= 2:
        ants_spread = max(
            float(np.max(np.abs(r.numpy().astype(np.float64) - ants_array))) for r in ants_results[1:]
        )
        torch_spread = max(
            float(np.max(np.abs(r.numpy().astype(np.float64) - torch_array))) for r in torch_results[1:]
        )
        print(f"ANTs run-to-run max |diff|:      {ants_spread:.6g}")
        print(f"ANTsTorch run-to-run max |diff|: {torch_spread:.6g}")

    peak_memory_mb = None
    if track_cuda_memory:
        peak_memory = cuda_peak_memory(device)
        if peak_memory is not None:
            peak_memory_mb = peak_memory / 2**20
            print(f"ANTsTorch CUDA peak memory: {peak_memory_mb:.1f} MiB")
    print(f"Outputs written with prefix: {prefix}")

    if not args.no_report:
        report_path = Path(f"{prefix}_report.html")
        generate_html_report(
            report_path=report_path,
            image=image,
            ants_denoised=ants_results[0],
            torch_denoised=torch_results[0],
            ants_noise=image - ants_results[0],
            torch_noise=image - torch_results[0],
            diff_img=diff_image,
            ants_times=ants_times,
            torch_times=torch_times,
            ants_warmup=ants_warmup_time,
            torch_warmup=torch_warmup_time,
            device=device,
            threads=threads,
            options=options,
            input_path=str(input_path),
            denoised_metrics=denoised_metrics,
            noise_metrics=noise_metrics,
            ants_spread=ants_spread,
            torch_spread=torch_spread,
            cuda_peak_mb=peak_memory_mb,
        )
        print(f"HTML comparison report generated: {report_path.resolve()}")
        print(f"To view the HTML report, run: open \"{report_path.resolve()}\"")


if __name__ == "__main__":
    main()
