"""Repeated, synchronized primitive timings (first sample is a cold run)."""
import json
import time
from pathlib import Path

import numpy as np
import torch


def synchronize(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device is not None and device.type == "mps":
        torch.mps.synchronize()


def measure(label, function, repeats, records, device=None):
    if repeats < 1:
        raise ValueError("--repeats must be at least 1")
    samples = []
    result = None
    for index in range(repeats):
        # Release the previous output before measuring the next allocation peak.
        result = None
        synchronize(device)
        start = time.perf_counter()
        result = function()
        synchronize(device)
        samples.append(time.perf_counter() - start)
        print(f"{label} run {index + 1}/{repeats}: {samples[-1]:.6f} s", flush=True)
    record = summarize(samples)
    q1, q3 = record["q1_seconds"], record["q3_seconds"]
    records[label] = record
    print(f"{label}: median {record['median_seconds']:.6f} s, "
          f"IQR [{q1:.6f}, {q3:.6f}] s (N={repeats}; first sample cold)", flush=True)
    return result, record["median_seconds"]


def summarize(samples):
    q1, q3 = np.percentile(samples, [25, 75])
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "q1_seconds": float(q1), "q3_seconds": float(q3),
        "min_seconds": min(samples), "max_seconds": max(samples),
        "first_sample_cold": True,
    }


def write_timings(path, records):
    Path(path).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
