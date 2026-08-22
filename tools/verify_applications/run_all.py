"""
Run every verify_*.py script in this folder as a subprocess and print a
PASS/FAIL summary at the end.

Each script is fully independent (own imports, own data), so a crash in
one (e.g. missing pretrained weights) can't take down the others -- this
is exactly the point: some of these WILL fail today because not all
weights have been converted/delivered yet (see README.md for the current
status table). A failing script here is useful information, not a bug in
the runner.

Usage
-----
    python run_all.py                     # run everything
    python run_all.py verify_lung_*.py     # run a subset (shell glob)
    python run_all.py --list               # just list discovered scripts
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def discover_scripts(patterns=None):
    if not patterns:
        patterns = ["verify_*.py"]
    found = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(HERE, pattern)):
            found.add(path)
    return sorted(found)


def main():
    args = sys.argv[1:]
    if "--list" in args:
        args.remove("--list")
        for path in discover_scripts(args or None):
            print(os.path.basename(path))
        return

    scripts = discover_scripts(args or None)
    if not scripts:
        print("No verify_*.py scripts matched.")
        return

    results = []
    for path in scripts:
        name = os.path.basename(path)
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        start = time.time()
        proc = subprocess.run([sys.executable, path], cwd=HERE)
        elapsed = time.time() - start
        ok = proc.returncode == 0
        results.append((name, ok, elapsed))
        print(f"--- {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) ---")

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(name) for name, _, _ in results)
    for name, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.ljust(width)}  ({elapsed:.1f}s)")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} passed")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
