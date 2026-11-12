"""Standalone supervised-arm routing and reporting in the comparison CLI."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / 'benchmarks' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('model', ['label_syn', 'label_intensity_syn'])
def test_label_models_route_only_to_antstorch_and_appear_in_summary(tmp_path, monkeypatch, model):
    comparison = load_script('compare_syntx_antstorch_mindboggle')
    monkeypatch.setattr(comparison, '_add_jacobian', lambda rec, pair: rec)
    monkeypatch.setattr(comparison, 'SCRATCH_BASE', str(tmp_path))
    monkeypatch.setattr(comparison, 'FAIR', True)
    monkeypatch.setattr(comparison, '_affine_source', lambda *args: 'syntx')
    record = dict(pair_idx=0, _library='antstorch', _regularizer=model, _status='SUCCESS',
                  dice_sym=.8, dice_fixed=.8, _deformable_seconds=1., jac_folding_pct=0.)

    def no_syntx(*args):
        pytest.fail('Label-guided models have no syntx counterpart')

    records = comparison._run_pair_model(model, 0, 'cpu', str(tmp_path),
        [('syntx', no_syntx), ('antstorch', lambda *args: dict(record))])
    assert len(records) == 1
    assert records[0]['_affine_source'] == 'syntx'
    summary = '\n'.join(comparison._aggregate(records, [model], False))
    assert f'{model} (supervised)' in summary
    assert '0.8000' in summary

    (tmp_path / 'comparison_results.json').write_text(json.dumps(records))
    plots = load_script('plot_mindboggle_results')
    assert f'{model} (supervised)' in plots.collect(str(tmp_path))


def test_label_cli_keeps_native_gradients_and_forwards_label_weight(tmp_path, monkeypatch):
    from antstorch.benchmark import evaluate
    comparison = load_script('compare_syntx_antstorch_mindboggle')
    monkeypatch.setattr(comparison, 'FAIR', True)
    monkeypatch.setattr(comparison, 'LABEL_WEIGHT', .25)
    captured = {}

    def evaluate_stub(**kwargs):
        captured.update(kwargs)
        return dict(runtime_seconds=2., affine_seconds=1.)

    monkeypatch.setattr(evaluate, 'evaluate_mindboggle_pair', evaluate_stub)
    result = comparison._run_antstorch('label_intensity_syn', 0, 'mps', str(tmp_path))
    assert result['_status'] == 'SUCCESS'
    assert captured['label_image_weighting'] == .25
    assert 'use_analytical_gradients' not in captured
    assert 'syntx_parity' not in captured
    assert captured['model'] == 'label_intensity_syn'


def test_label_only_cli_writes_both_supervised_results(mock_mindboggle_dataset, tmp_path, monkeypatch):
    import torch
    from antstorch.ants_transform_io import write_affine_transform
    from antstorch.benchmark import evaluate

    pairs_csv, data_dir = mock_mindboggle_dataset
    comparison = load_script('compare_syntx_antstorch_mindboggle')
    out = tmp_path / 'comparison'
    affines = out / 'antstorch_canonical_affines'
    affines.mkdir(parents=True)
    write_affine_transform(torch.eye(3), torch.zeros(3), 3, str(affines / 'pair_000_0GenericAffine.mat'))
    (affines / 'pair_000_affine_info.json').write_text(json.dumps({'source': 'syntx', 'runtime_seconds': 0.}))
    from antstorch.benchmark.data import load_mindboggle_pair
    fixed = load_mindboggle_pair(0, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=False)['fixed']
    monkeypatch.setattr(comparison, '_JAC_LAST', (0, fixed))
    real_evaluate = evaluate.evaluate_mindboggle_pair

    def tiny_evaluate(**kwargs):
        kwargs.update(pairs_csv=pairs_csv, data_dir=data_dir, use_n4=False,
                      levels=(1,), reg_iterations=(1,))
        return real_evaluate(**kwargs)

    monkeypatch.setattr(evaluate, 'evaluate_mindboggle_pair', tiny_evaluate)
    monkeypatch.setattr(comparison.sys, 'argv', ['benchmark', '--pairs', '0', '--models',
        'label_syn', 'label_intensity_syn', '--skip-syntx', '--no-warmup', '--device', 'cpu',
        '--out-dir', str(out)])
    comparison.main()
    records = json.loads((out / 'comparison_results.json').read_text())
    assert len(records) == 2
    assert all(r['_status'] == 'SUCCESS' and r['supervision'] == 'evaluation_labels' for r in records)
    assert all(r['_affine_source'] == 'syntx' for r in records)
    assert all('_jac_error' not in r for r in records)
    summaries = list(out.glob('*.md'))
    assert summaries
    assert 'Supervised label-guided arms' in summaries[0].read_text()


def test_script_selects_checkout_before_dependency_can_import_installed_package(tmp_path):
    import os
    import subprocess
    import sys

    stale = tmp_path / 'installed' / 'antstorch'
    stale.mkdir(parents=True)
    (stale / '__init__.py').write_text("raise RuntimeError('stale installed ANTsTorch imported')\n")
    script = ROOT / 'tools' / 'benchmarks' / 'compare_syntx_antstorch_mindboggle.py'
    # Launch outside the checkout with a stale installation on PYTHONPATH.
    # A dependency may import antstorch before _run_antstorch is reached.
    code = (
        'import runpy; '
        f'runpy.run_path({str(script)!r}, run_name="benchmark_import_check"); '
        'import antstorch.benchmark.evaluate as e; '
        f'assert e.__file__ == {str(ROOT / "antstorch/benchmark/evaluate.py")!r}; '
        'assert "label_syn" in e._LABEL_MODELS'
    )
    environment = dict(os.environ, PYTHONPATH=str(stale.parent))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
