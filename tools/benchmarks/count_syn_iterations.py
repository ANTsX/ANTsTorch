#!/usr/bin/env python
"""Count the SyN iterations actually executed per level (antstorch default,
antstorch --syntx-parity, syntx) on one real Mindboggle pair, with the
--matched schedule. Answers: does the loss-slope early stop ever fire, and
does syntx really run fewer iterations than antstorch?

  python tools/benchmarks/count_syn_iterations.py --reg gaussian --pair 0 --device mps
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_syntx_antstorch_mindboggle as cmp  # noqa: E402

sys.path.insert(0, cmp.ANTSTORCH_ROOT)


def run_antstorch(reg, pair, device, out_dir, parity):
    import antstorch.syn.syn as S
    orig = S._fit_syn_level
    per_level = []

    def wrap(*a, **k):
        out = orig(*a, **k)
        per_level.append(len(out[-1]))
        return out
    S._fit_syn_level = wrap
    cmp.ANTSTORCH_EXTRA.clear()
    cmp.ANTSTORCH_EXTRA["inverse_schedule"] = "end_of_level"
    if parity:
        cmp.ANTSTORCH_EXTRA["syntx_parity"] = True
    t0 = time.time()
    rec = cmp._run_antstorch(reg, pair, device, out_dir, matched=True)
    S._fit_syn_level = orig
    return dict(per_level=per_level, dice=rec.get("dice_sym"), wall=time.time() - t0, status=rec["_status"])


def run_syntx(reg, pair, device, out_dir):
    sys.path.insert(0, os.path.join(cmp.SYNTX_ROOT, "src"))
    import importlib
    import syntx  # noqa: F401  (package attr 'syn' is a function; get the module itself)
    SS = importlib.import_module("syntx.syn")
    SS = sys.modules["syntx.syn"]
    orig = SS.check_convergence
    calls = {"n": 0, "true": 0}

    def spy(losses, *a, **k):
        r = orig(losses, *a, **k)
        calls["n"] += 1
        calls["true"] += int(bool(r))
        return r
    SS.check_convergence = spy
    t0 = time.time()
    rec = cmp._run_syntx(reg, pair, device, out_dir, matched=True)
    SS.check_convergence = orig
    n_losses = None
    for key in ("syn_losses", "loss_history"):
        v = rec.get(key)
        if isinstance(v, list):
            n_losses = len(v); break
    return dict(total_loss_evals=n_losses, convergence_calls=calls["n"], convergence_true=calls["true"],
                dice=rec.get("dice_sym"), wall=time.time() - t0, status=rec["_status"],
                err=rec.get("_error"), rec_keys=sorted(rec.keys()) if n_losses is None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default="gaussian")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/Desktop/count_iters"))
    ap.add_argument("--skip-syntx", action="store_true")
    ap.add_argument("--only-syntx", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    if not a.only_syntx:
        res["antstorch_default"] = run_antstorch(a.reg, a.pair, a.device, a.out_dir, False)
        print("antstorch default:", res["antstorch_default"], flush=True)
        res["antstorch_parity"] = run_antstorch(a.reg, a.pair, a.device, a.out_dir, True)
        print("antstorch parity :", res["antstorch_parity"], flush=True)
    if not a.skip_syntx:
        res["syntx"] = run_syntx(a.reg, a.pair, a.device, a.out_dir)
        print("syntx            :", res["syntx"], flush=True)
    json.dump(res, open(os.path.join(a.out_dir, "iteration_counts.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
