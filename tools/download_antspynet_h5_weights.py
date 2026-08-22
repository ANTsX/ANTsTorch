#!/usr/bin/env python3
"""
download_antspynet_h5_weights.py

Batch/parallel downloader for the *original* ANTsPyNet Keras `.h5` weight
files (lung_extraction, lung_segmentation, mouse.py, and
white_matter_hyperintensity_segmentation) -- the inputs that
tools/convert_wmh_bespoke.py and tools/convert_lung_mouse_bespoke.py read
from `~/.keras/ANTsXNet/` to produce the ANTsTorch `_pytorch.pt` files.

Figshare does not offer FTP access -- every one of these files is served
over plain HTTPS from a fixed `ndownloader.figshare.com/files/<id>` URL
(the same URL your browser hits when you click "Download" on a figshare
page). This script just fetches those URLs directly, in parallel, with a
thread pool -- which is the practical "faster than clicking each figshare
page by hand" answer. The file numbers below were read directly from
ANTsPyNet's own `antspynet/utilities/get_pretrained_network.py` on GitHub
and cross-checked twice.

Already-downloaded files are skipped by default (pass --force to
redownload). A file that isn't public on figshare yet has no entry in the
manifest below and is reported clearly at the end rather than silently
skipped -- see the NOT_YET_PUBLIC list.

Usage:
    python download_antspynet_h5_weights.py
    python download_antspynet_h5_weights.py --out-dir ~/.keras/ANTsXNet --workers 8
    python download_antspynet_h5_weights.py --only protonLungMri hyperMapp3r
    python download_antspynet_h5_weights.py --force   # redownload everything
"""
import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

FIGSHARE_URL = "https://ndownloader.figshare.com/files/{id}"

# stem (matches <stem>.h5 in ANTsPyNet's own cache dir) -> figshare file id
MANIFEST = {
    # -- lung_extraction / lung_segmentation --
    "protonLungMri": 13606799,
    "protonLobes": 30678455,
    "maskLobes": 30678458,
    "lungCtWithPriorsSegmentationWeights": 28357818,
    "wholeLungMaskFromVentilation": 28914441,
    "xrayLungExtraction": 41965818,
    "elBicho": 26736779,
    "pulmonaryArteryWeights": 46400752,
    "pulmonaryAirwayWeights": 45187168,

    # -- mouse.py --
    "mouseT2wBrainExtraction3D": 49188910,
    "ex5_coronal_weights": 42434193,
    "ex5_sagittal_weights": 42434202,
    "mouseT2wBrainParcellation3DNick": 44714944,
    "mouseT2wBrainParcellation3DTct": 47214538,
    "mouseSTPTBrainParcellation3DJay": 46710592,
    "allen_brain_mask_weights": 36999880,

    # -- white_matter_hyperintensity_segmentation.py --
    "sysuMediaWmhFlairOnlyModel0": 22898441,
    "sysuMediaWmhFlairOnlyModel1": 22898570,
    "sysuMediaWmhFlairOnlyModel2": 22898438,
    "sysuMediaWmhFlairT1Model0": 22898450,
    "sysuMediaWmhFlairT1Model1": 22898453,
    "sysuMediaWmhFlairT1Model2": 22898459,
    "antsxnetWmh": 42301059,
    "antsxnetWmhOr": 42301056,
    "hyperMapp3r": 38790702,
    "pvs_shiva_t1_0": 48660169,
    "pvs_shiva_t1_1": 48660193,
    "pvs_shiva_t1_2": 48660199,
    "pvs_shiva_t1_3": 48660178,
    "pvs_shiva_t1_4": 48660172,
    "pvs_shiva_t1_5": 48660187,
    "pvs_shiva_t1_flair_0": 48660181,
    "pvs_shiva_t1_flair_1": 48660175,
    "pvs_shiva_t1_flair_2": 48660184,
    "pvs_shiva_t1_flair_3": 48660190,
    "pvs_shiva_t1_flair_4": 48660196,
    "wmh_shiva_flair_0": 48660487,
    "wmh_shiva_flair_1": 48660496,
    "wmh_shiva_flair_2": 48660493,
    "wmh_shiva_flair_3": 48660490,
    "wmh_shiva_flair_4": 48660511,
    "wmh_shiva_t1_flair_0": 48660529,
    "wmh_shiva_t1_flair_1": 48660547,
    "wmh_shiva_t1_flair_2": 48660499,
    "wmh_shiva_t1_flair_3": 48660550,
    "wmh_shiva_t1_flair_4": 48660544,
}

# Ids used by convert_lung_mouse_bespoke.py that have NO entry in ANTsPyNet's
# own get_pretrained_network.py as of this writing -- confirmed absent by
# two independent scans of the real source, not just a naming mismatch on
# this end. Nothing can download these until ANTsPyNet's maintainers
# publish them (or you locate/regenerate the .h5 some other way).
NOT_YET_PUBLIC = [
    "allen_brain_leftright_coronal_mask_weights",
    "allen_cerebellum_sagittal_mask_weights",
    "allen_cerebellum_coronal_mask_weights",
    "allen_sr_weights",
]


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_one(stem, file_id, out_dir, force):
    out_path = os.path.join(out_dir, f"{stem}.h5")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not force:
        return stem, "skip", os.path.getsize(out_path), None

    url = FIGSHARE_URL.format(id=file_id)
    tmp_path = out_path + ".part"
    start = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out_f.write(chunk)
        os.replace(tmp_path, out_path)  # atomic -- no truncated .h5 left behind on interruption
        elapsed = time.time() - start
        size = os.path.getsize(out_path)
        return stem, "ok", size, elapsed
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return stem, "fail", 0, str(e)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="~/.keras/ANTsXNet", help="Directory to save .h5 files into (default: ~/.keras/ANTsXNet)")
    p.add_argument("--workers", type=int, default=8, help="Parallel download threads (default: 8)")
    p.add_argument("--only", nargs="+", default=None, help="Only download these stems (space-separated, no .h5 extension)")
    p.add_argument("--force", action="store_true", help="Redownload even if the file already exists")
    args = p.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = MANIFEST
    if args.only:
        unknown = [s for s in args.only if s not in manifest]
        if unknown:
            print(f"Unknown stem(s), not in the manifest: {unknown}")
            print(f"Known stems: {sorted(manifest)}")
            sys.exit(1)
        manifest = {s: manifest[s] for s in args.only}

    print(f"Downloading {len(manifest)} file(s) to {out_dir} with {args.workers} parallel workers...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, stem, fid, out_dir, args.force): stem for stem, fid in manifest.items()}
        for fut in as_completed(futures):
            stem, status, size, extra = fut.result()
            if status == "skip":
                print(f"  [skip]  {stem}.h5 already present ({human_size(size)})")
            elif status == "ok":
                print(f"  [ok]    {stem}.h5  {human_size(size)}  ({extra:.1f}s)")
            else:
                print(f"  [FAIL]  {stem}.h5  -- {extra}")
            results.append((stem, status))

    n_ok = sum(1 for _, s in results if s == "ok")
    n_skip = sum(1 for _, s in results if s == "skip")
    n_fail = sum(1 for _, s in results if s == "fail")
    print(f"\n{n_ok} downloaded, {n_skip} already present, {n_fail} failed (of {len(manifest)} requested).")

    if not args.only:
        print(
            f"\n{len(NOT_YET_PUBLIC)} id(s) used by convert_lung_mouse_bespoke.py have NO known figshare URL "
            f"(not found in ANTsPyNet's own get_pretrained_network.py -- not a bug in this script, "
            f"they simply haven't been published there yet):"
        )
        for stem in NOT_YET_PUBLIC:
            print(f"  - {stem}.h5")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
