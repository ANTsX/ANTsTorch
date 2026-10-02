#!/usr/bin/env python3
"""Per-stage timing of antstorch's dense SyN loop on one Mindboggle pair.

Usage (Mac idle, or on hyperwolf -- do NOT run during the full benchmark):
    python3 profile_antstorch_syn_stages.py --model gaussian_syn --device mps
    python3 profile_antstorch_syn_stages.py --model gaussian_syn --device cuda:1

Does two things:
  1. Prints the effective kwargs syn_registration() receives (test 1), using
     syn_registration()'s syntx-aligned defaults (as compare_syntx_antstorch_mindboggle.py).
  2. Times each stage with device synchronization (test 2): calls, total
     seconds, % of wall. Iterations are shortened (default 10,10,5 / 10,10,5,1) -- ratios
     stay valid, runtime drops ~10x.
"""
import argparse, collections, functools, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_syntx_antstorch_mindboggle as cmp  # noqa: E402

sys.path.insert(0, cmp.ANTSTORCH_ROOT)
import torch  # noqa: E402

STATS = collections.defaultdict(lambda: [0, 0.0])
_DEV = {"type": "cpu"}


def _sync():
    if _DEV["type"] == "mps":
        torch.mps.synchronize()
    elif _DEV["type"] == "cuda":
        torch.cuda.synchronize()


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        _sync(); t0 = time.perf_counter()
        out = fn(*a, **k)
        _sync(); dt = time.perf_counter() - t0
        STATS[name][0] += 1; STATS[name][1] += dt
        return out
    return wrapper


def install_patches():
    import antstorch.syn.syn as S
    import antstorch.syn.core.grid as G
    import antstorch.syn.core.jacobian as J
    import antstorch.syn.core.smoothing as SM

    # Names looked up in antstorch.syn.syn's namespace
    for n in ("prepare_mid_images_and_gradients_torch", "_similarity_loss", "_apply_regularizer",
              "_cfl_normalize", "_eulerian_update", "update_inverse_field_nd",
              "separable_gaussian_filter", "reg_adam_direction", "_fit_syn_level"):
        if hasattr(S, n):
            setattr(S, n, timed(f"syn.{n}", getattr(S, n)))
    # Looked up at call time inside prepare_mid_images_and_gradients_torch
    J._spatial_jacobian_nd = timed("jacobian._spatial_jacobian_nd (image grads, fwd)", J._spatial_jacobian_nd)
    G._image_spatial_gradient = timed("grid._image_spatial_gradient (bwd + fwd)", G._image_spatial_gradient)
    G.grid_sample_nd = timed("grid.grid_sample_nd", G.grid_sample_nd)
    G.AnalyticalGridSample.backward = staticmethod(timed("AnalyticalGridSample.backward", G.AnalyticalGridSample.backward))
    # ``prepare_mid_images...`` resolves grid_sample_nd via grid.py globals -> patched above.


def log_kwargs():
    import antstorch.benchmark.evaluate as E
    orig = E.syn_registration

    def spy(**kw):
        keys = ("type_of_transform", "regularizer", "optimizer", "levels", "reg_iterations", "grad_step",
                "syn_metric", "neighborhood_radius", "flow_sigma", "total_sigma", "antisymmetric",
                "inverse_method", "in_loop_inverse_steps", "inverse_schedule", "end_of_level_inverse_steps", "gaussian_sigma_mode", "conservative_smooth",
                "regadam_grad_sigma", "regadam_quotient_sigma")
        print("\n== syn_registration kwargs passed by the harness ==")
        for k in keys:
            print(f"  {k:24s} {kw.get(k, '<not passed -> syn_registration default>')}")
        return timed(f"syn_registration TOTAL ({kw.get('type_of_transform')})", orig)(**kw)
    E.syn_registration = spy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gaussian_syn",
                    help="antstorch model name, e.g. gaussian_syn, sobolev_syn, dsti_syn, bspline_syn, gaussian_regadam ...")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--iters", type=int, nargs="+", default=None, help="one value per level; default 10 10 5 (3 levels) or 10 10 5 1 (4 levels)")
    ap.add_argument("--no-matched", action="store_true")
    ap.add_argument("--inverse-schedule", choices=["per_iteration", "end_of_level"], default=None)
    ap.add_argument("--end-of-level-inverse-steps", type=int, default=None)
    ap.add_argument("--inverse-method", choices=["anderson", "fixed_point", "hybrid_lm"], default=None)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/Desktop/profile_stages"))
    ap.add_argument("--affine-dir", default=os.path.expanduser("~/Desktop/smoke_test_matched/antstorch_canonical_affines"),
                    help="reuse the benchmark run's cached affines (avoids refitting the affine)")
    a = ap.parse_args()

    _DEV["type"] = torch.device(a.device).type
    reg = a.model[:-4] if a.model.endswith("_syn") else a.model  # 'gaussian_syn' -> 'gaussian'
    # syntx-aligned defaults live in syn_registration() (_SYNTX_DEFAULTS); the profile only needs the level
    # count to shorten reg_iterations consistently, so fetch that one entry explicitly.
    from antstorch.syn.syn import _SYNTX_DEFAULTS
    _base = reg[: -len("_regadam")] if reg.endswith("_regadam") else reg
    _opt = "reg_adam" if reg.endswith("_regadam") else "gradient_descent"
    if a.no_matched:
        extra = {"syntx_defaults": False, "levels": (8, 4, 2, 1)}  # the historical harness pyramid
    else:
        extra = {"levels": tuple(_SYNTX_DEFAULTS[(_base, _opt)]["levels"])}
    # Shorten the schedule but keep the harness's own level structure:
    # levels come from the syntx-aligned defaults above (or the historical (8,4,2,1) with --no-matched).
    n_levels = len(extra.get("levels", (8, 4, 2, 1)))
    if a.iters is not None:
        if len(a.iters) != n_levels:
            sys.exit(f"--iters needs {n_levels} values for this model (levels={extra.get('levels', (8, 4, 2, 1))})")
        extra["reg_iterations"] = tuple(a.iters)
    else:
        extra["reg_iterations"] = (10, 10, 5) if n_levels == 3 else (10, 10, 5, 1)

    if a.inverse_schedule:
        extra["inverse_schedule"] = a.inverse_schedule
    if a.end_of_level_inverse_steps is not None:
        extra["end_of_level_inverse_steps"] = a.end_of_level_inverse_steps
    if a.inverse_method:
        extra["inverse_method"] = a.inverse_method

    log_kwargs()
    install_patches()
    from antstorch.benchmark.evaluate import evaluate_mindboggle_pair
    os.makedirs(a.out_dir, exist_ok=True)

    # warm-up excluded from nothing: first call includes kernel compilation on MPS; report both.
    t0 = time.perf_counter()
    rec = evaluate_mindboggle_pair(
        pair_idx=a.pair, model=a.model, device=a.device, data_dir=cmp.DATA_DIR,
        canonical_affine_dir=a.affine_dir,
        verbose=False, **extra)
    _sync(); wall = time.perf_counter() - t0

    reg_t = rec.get("runtime_seconds", float("nan"))
    print(f"\n== {a.model} pair {a.pair} device {a.device} | wall {wall:.1f}s | registration runtime {reg_t:.1f}s "
          f"| dice_sym {rec.get('dice_sym', float('nan')):.4f} ==")
    fit = STATS.get("syn._fit_syn_level", [0, 0.0])[1]
    print(f"{'stage':62s} {'calls':>7s} {'total s':>9s} {'% of fit':>9s}")
    for name, (n, t) in sorted(STATS.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:62s} {n:7d} {t:9.2f} {100 * t / fit if fit else float('nan'):9.1f}")
    print("\nNote: entries are nested (e.g. grid_sample_nd is inside prepare_mid_images...; "
          "_spatial_jacobian_nd inside prepare_mid_images...), so percentages do not sum to 100. "
          "Backward of the similarity loss = fit - forward stages - AnalyticalGridSample.backward.")


if __name__ == "__main__":
    main()
