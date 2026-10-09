"""Synthetic maximin geometry and existing-pool runner wiring; no fitted campaigns."""
import csv
import json
import sys

import numpy as np
import pytest

from ..eig.joint import TIE_ATOL
from . import boxinggym_core as box
from . import core_count as core
from . import design_policies as policy
from .test_named_inputs import bg_run_args, bg_scalar_stub_fits, bg_stub_fits
from .test_prewave_fit_timeout import _count_result


def rows(path):
    """Read a JSON-lines file and return its parsed rows."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_maximin_uses_declared_axis_ranges_not_raw_distance():
    """Test that maximin uses declared axis ranges not raw distance."""
    candidates = np.array([[80., .2], [20., .9]])
    chosen, table, record = policy.select_space_filling(candidates, [[0., 0.]], [[0., 100.], [0., 1.]])
    assert chosen == [20., .9]
    np.testing.assert_allclose([row['min_normalized_distance'] for row in table],
                               [np.sqrt(.8**2+.2**2), np.sqrt(.2**2+.9**2)])
    assert record['normalization_bounds'] == [[0., 100.], [0., 1.]]
    assert record['observed_design_count'] == 1
    assert record['chosen_index'] == 1
    assert record['uses_posterior'] is False and record['uses_rng'] is False


def test_maximin_uses_minimum_over_every_observed_design():
    """Test that maximin uses minimum over every observed design."""
    chosen, table, _ = policy.select_space_filling([0., .25, .5, .75, 1.], [0., .6], [0., 1.])
    assert chosen == 1.
    np.testing.assert_allclose([row['min_normalized_distance'] for row in table], [0., .25, .1, .15, .4])


@pytest.mark.parametrize('candidates,expected', [([.8, .5, .2], .2),
    ([[1., 1.], [.5, .5], [0., 0.]], [0., 0.])])
def test_no_history_uses_normalized_center_and_registered_tie(candidates, expected):
    """Test that no history uses normalized center and registered tie."""
    dimensions = 1 if np.asarray(candidates).ndim == 1 else 2
    chosen, _, record = policy.select_space_filling(candidates, [], [[0., 1.]]*dimensions)
    assert chosen == expected
    assert record['n_tied_at_max'] == 2
    assert record['tie_atol'] == TIE_ATOL
    assert record['empty_history_rule'] == 'farthest_from_normalized_center'


def test_maximin_respects_unrounded_registered_tie_tolerance():
    """Test that maximin respects unrounded registered tie tolerance."""
    smaller = 1.-TIE_ATOL/2
    chosen, _, record = policy.select_space_filling([1., smaller], [0.], [[0., 1.]])
    assert chosen == smaller and record['n_tied_at_max'] == 2


def test_maximin_is_invariant_to_pool_order_and_affine_unit_change():
    """Test that maximin is invariant to pool order and affine unit change."""
    candidates = np.array([[.1, .8], [.7, .4], [.9, .2]])
    history = np.array([[.1, .1], [.6, .6]])
    chosen, _, _ = policy.select_space_filling(candidates, history, [[0., 1.], [0., 1.]])
    scale, offset = np.array([100., 3.]), np.array([-10., 12.])
    transformed = candidates[::-1]*scale+offset
    actual, _, _ = policy.select_space_filling(transformed, history*scale+offset,
                                               np.column_stack([offset, offset+scale]))
    np.testing.assert_allclose(actual, np.asarray(chosen)*scale+offset)


def test_maximin_select_never_reads_posterior_or_random_generator(monkeypatch):
    """Test that maximin select never reads posterior or random generator."""
    def forbidden(*args, **kwargs):
        pytest.fail('space_filling reached posterior or RNG work')
    monkeypatch.setattr(policy, 'criteria_table', forbidden)
    monkeypatch.setattr(policy, 'model_logweights', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    chosen, table, record = policy.select(None, [.2, .8], 123, 'space_filling',
                                           observed_designs=[.3], bounds=[[0., 1.]])
    assert chosen == .8
    assert record['criteria'] == table and record['criteria_columns'] == ['min_normalized_distance']
    assert all(set(row) == {'design', 'min_normalized_distance'} for row in table)


@pytest.mark.parametrize('candidates,history,bounds', [([], [], [[0, 1]]),
    ([np.nan], [], [[0, 1]]), ([.2], [np.inf], [[0, 1]]),
    ([.2], [], [[1, 1]]), ([.2], [], [[1, 0]]),
    ([.2], [], [[0, np.inf]]), ([[.2, .3]], [], [[0, 1]]),
    ([[.2, .3]], [[.4]], [[0, 1], [0, 1]]),
    ([.2], [], [[-1.e308, 1.e308]]), ([1.01], [], [[0, 1]]),
    ([.2], [-.01], [[0, 1]]), ([.2], np.empty((0, 2)), [[0, 1]]),
    ([.2], np.empty((1, 0)), [[0, 1]]), ([.2], np.empty((0, 1, 1)), [[0, 1]])])
def test_invalid_geometry_is_not_silently_repaired(candidates, history, bounds):
    """Test that invalid geometry is not silently repaired."""
    with pytest.raises(ValueError):
        policy.select_space_filling(candidates, history, bounds)


@pytest.mark.parametrize('cell', ['001', '011', '101', '111'])
def test_space_filling_is_only_the_design_level_one_variant(cell):
    """Test that space filling is only the design level one variant."""
    assert policy.check_policy(cell, 'space_filling') is True
    assert policy.variant_token('space_filling', True) == 'design_space_filling'
    for invalid in [cell[:2]+'0', cell[:2]+'2']:
        with pytest.raises(ValueError):
            policy.check_policy(invalid, 'space_filling')


def test_core_maximin_uses_all_public_history_and_preserves_generated_pool(tmp_path, monkeypatch):
    """Test that core maximin uses all public history and preserves generated pool."""
    monkeypatch.setattr(core, 'fit', lambda hypothesis, *args: _count_result(hypothesis))
    def forbidden(*args, **kwargs):
        pytest.fail('maximin acquisition computed information')
    monkeypatch.setattr(policy, 'criteria_table', forbidden)
    monkeypatch.setattr(core, 'choose_epig', forbidden)
    output = tmp_path/'core'
    monkeypatch.setattr(sys, 'argv', ['core', '--output', str(output), '--profile', 'dev',
        '--cell', '001', '--rounds', '3', '--proposer', 'mock', '--design-policy', 'space_filling'])
    core.main()
    observations = rows(output/'observations.jsonl')
    config = json.loads((output/'config.json').read_text())
    assert config['design_normalization_bounds'] == [[0., 1.]]
    assert config['design_policy_definition'] == policy.SPACE_FILLING_DEFINITION
    hashes = set()
    for t in range(1, 4):
        record = json.loads((output/'acquisition'/f't{t:02d}.json').read_text())
        previous = [row['u'] for row in observations if row['round'] < t]
        assert record['policy']['observed_design_count'] == len(previous)
        values = [row['design'] for row in record['rows']]
        np.testing.assert_array_equal(values, np.linspace(0, 1, config['design_count']))
        expected = [min(abs(value-observed) for observed in previous) for value in values]
        np.testing.assert_allclose([row['min_normalized_distance'] for row in record['rows']], expected)
        hashes.add(record['candidate_set_sha256'])
    assert len(hashes) == 1
    with (output/'points.csv').open() as stream:
        assert {row['variant'] for row in csv.DictReader(stream)} == {'design_space_filling'}


@pytest.mark.usefixtures('require_boxinggym_source')
@pytest.mark.parametrize('task', ['death', 'dugongs', 'hyperbolic'])
@pytest.mark.parametrize('design_policy', ['space_filling', 'point_disagreement'])
def test_boxing_new_policies_use_existing_candidates_without_eig(tmp_path, monkeypatch, task, design_policy):
    """Test that boxing new policies use existing candidates without eig."""
    bg_stub_fits(monkeypatch)
    bg_scalar_stub_fits(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail('new Box policy computed EIG or EPIG')
    monkeypatch.setattr(box, 'count_choose', forbidden)
    monkeypatch.setattr(box, 'joint_eig', forbidden)
    monkeypatch.setattr(box, 'choose_epig', forbidden)
    if design_policy == 'space_filling':
        monkeypatch.setattr(box, 'point_disagreement', forbidden)
        monkeypatch.setattr(box, 'scalar_point_disagreement', forbidden)
    output = tmp_path/task
    box.run(bg_run_args(output, task=task, cell='001', rounds=2, design_policy=design_policy))
    config = json.loads((output/'config.json').read_text())
    observations = rows(output/'observations.jsonl')
    key = policy.OBJECTIVE_COLUMN[design_policy]
    assert 'multi_component_eig_rule' not in config['task_spec']
    hashes = set()
    for t in [1, 2]:
        record = json.loads((output/'acquisition'/f't{t:02d}.json').read_text())
        assert record['target'] == record['policy']['design_policy'] == design_policy
        assert record['policy']['objective_column'] == key
        assert record['policy']['criteria'] == record['rows']
        assert record['chosen'] in [row['design'] for row in record['rows']]
        assert all(np.isfinite(row[key]) for row in record['rows'])
        assert record['policy']['uses_rng'] is False
        if design_policy == 'space_filling':
            assert record['policy']['observed_design_count'] == sum(row['round'] < t for row in observations)
            assert record['policy']['normalization_bounds'] == [item['range'] for item in config['interface']['inputs']]
        hashes.add(record['candidate_set_sha256'])
    assert len(hashes) == 1
    with (output/'points.csv').open() as stream:
        assert {row['variant'] for row in csv.DictReader(stream)} == {policy.VARIANT_TOKEN[design_policy]}
    assert json.loads((output/'summary.json').read_text())['paid_api_calls'] == 0


STUDY_BASE = '2e6496319efa91bf771dfafedfd973487f1f1611'
LEGACY_HARNESS = r'''
from datetime import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
spec = importlib.util.spec_from_file_location(
    'bayes_model_discovery.experimental._study_fixture', sys.argv[1])
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2001, 1, 1, tzinfo=tz)
output, runner, cell, policy_name, fit_kind = sys.argv[2:]
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(fixture.core.time, 'perf_counter', lambda: 0.0)
    patch.setattr(fixture.core.resource, 'getrusage',
                  lambda *_: SimpleNamespace(ru_utime=0.0, ru_stime=0.0))
    patch.setattr(fixture.core, 'datetime', FrozenDatetime)
    patch.setattr(fixture.box, 'datetime', FrozenDatetime)
    if runner == 'core':
        patch.setattr(fixture.core, 'fit', lambda h, *args: fixture._count_result(h))
        argv = ['core', '--output', output, '--profile', 'dev', '--cell', cell,
                '--rounds', '1', '--proposer', 'mock']
        if policy_name != 'default':
            argv += ['--design-policy', policy_name]
        patch.setattr(sys, 'argv', argv)
        fixture.core.main()
    else:
        if fit_kind == 'real':
            settings = fixture.box.SMCSettings
            patch.setattr(fixture.box, 'SMCSettings', lambda *_: settings(16, 2, 2))
        else:
            fixture.bg_stub_fits(patch)
        kwargs = {} if policy_name == 'default' else {'design_policy': policy_name}
        fixture.box.run(fixture.bg_run_args(Path(output), task='death', cell=cell,
                                           seed=73, rounds=1, **kwargs))
'''


def _parity_normalize_timing(value):
    """Only measured execution-time fields vary in bounded real SMC."""
    from .test_boxinggym_safeguards_parity import VOLATILE_KEYS
    if isinstance(value, dict):
        return {key: _parity_normalize_timing(item) for key, item in value.items()
                if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_parity_normalize_timing(item) for item in value]
    return value


@pytest.mark.parametrize('runner,cell,design_policy,fit_kind', [
    ('core', '000', 'default', 'fake'), ('core', '001', 'default', 'fake'),
    ('core', '100', 'default', 'fake'), ('core', '101', 'default', 'fake'),
    ('core', '002', 'default', 'fake'), ('core', '001', 'joint_eig', 'fake'),
    ('core', '001', 'structure_eig', 'fake'), ('core', '001', 'point_disagreement', 'fake'),
    ('core', '001', 'uniform_random', 'fake'), ('core', '002', 'epig', 'fake'),
    ('boxinggym', '000', 'default', 'fake'), ('boxinggym', '001', 'default', 'fake'),
    ('boxinggym', '100', 'default', 'fake'), ('boxinggym', '101', 'default', 'fake'),
    ('boxinggym', '002', 'default', 'fake'), ('boxinggym', '001', 'joint_eig', 'fake'),
    ('boxinggym', '002', 'epig', 'fake'), ('boxinggym', '001', 'default', 'real'),
])
@pytest.mark.skip(reason="parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
def test_opt_in_existing_core_boxing_artifact_bytes(tmp_path, runner, cell, design_policy,
                                                   fit_kind, request):
    """Test that opt in existing core boxing artifact bytes."""
    import hashlib
    import os
    import subprocess
    from pathlib import Path

    if os.environ.get('BMD_BASELINE_COMPARE') != '1':
        pytest.skip('set BMD_BASELINE_COMPARE=1 for the design-study baseline artifact parity')
    baseline_text = os.environ.get('BMD_DESIGN_STUDY_BASELINE')
    if not baseline_text:
        pytest.skip('set BMD_DESIGN_STUDY_BASELINE to a read-only checkout of the design-study baseline')
    if runner == 'boxinggym':
        request.getfixturevalue('require_boxinggym_source')
    root = Path(__file__).resolve().parents[3]
    baseline = Path(baseline_text).resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=baseline,
                                   text=True).strip() == STUDY_BASE
    output, saved, configs = tmp_path/'same_output', [], []
    for label, source in [('baseline', baseline), ('current', root)]:
        env = os.environ.copy()
        env.update(PYTHONPATH=f'{source}/src:{source}:' + env.get('PYTHONPATH', ''),
                   PYTHONDONTWRITEBYTECODE='1', OPENAI_API_KEY='', ANTHROPIC_API_KEY='',
                   OPENAI_COMPAT_API_KEY='', OPENROUTER_API_KEY='',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.run([sys.executable, '-B', '-c', LEGACY_HARNESS,
            str(Path(__file__).resolve()), str(output), runner, cell, design_policy, fit_kind],
            cwd=source, env=env, capture_output=True, text=True, timeout=300, check=False)
        (tmp_path/f'{label}.log').write_text(process.stdout+process.stderr)
        assert process.returncode == 0, process.stdout+process.stderr
        config = json.loads((output/'config.json').read_text())
        if fit_kind == 'real':
            assert config['smc']['particles'] == 16 and config['smc']['chains'] == 2
        for name, digest in config['source_files_sha256'].items():
            assert digest == hashlib.sha256(
                (source/'src/bayes_model_discovery/experimental'/name).read_bytes()).hexdigest()
        configs.append(config)
        destination = tmp_path/label
        output.rename(destination)
        saved.append(destination)
    before, after = saved
    paths = lambda directory: sorted(str(p.relative_to(directory)) for p in directory.rglob('*')
                                     if p.is_file())
    assert paths(before) == paths(after)
    assert configs[0]['source_files_sha256'].keys() == configs[1]['source_files_sha256'].keys()
    replacements = [(configs[1]['source_files_sha256'][key].encode(), digest.encode())
                    for key, digest in configs[0]['source_files_sha256'].items()]
    report = []
    for name in paths(before):
        left, right = (before/name).read_bytes(), (after/name).read_bytes()
        original_right = right
        for current, previous in replacements:
            right = right.replace(current, previous)
        right = right.replace((str(root)+'/').encode(), (str(baseline)+'/').encode())
        if fit_kind == 'real' and name.endswith('.json'):
            assert _parity_normalize_timing(json.loads(left)) == _parity_normalize_timing(json.loads(right)), name
        elif fit_kind == 'real' and name.endswith('.jsonl'):
            assert [_parity_normalize_timing(json.loads(row)) for row in left.splitlines()] == [
                _parity_normalize_timing(json.loads(row)) for row in right.splitlines()], name
        else:
            assert left == right, name
        report.append({'artifact': name, 'raw_bytes_identical': left == original_right,
            'comparison': 'verified_source_provenance_and_timing' if fit_kind == 'real'
                          and name.endswith(('.json', '.jsonl')) else 'verified_source_provenance_only'})
    (tmp_path/'parity_report.json').write_text(json.dumps({
        'baseline': STUDY_BASE, 'runner': runner, 'cell': cell, 'policy': design_policy,
        'fits': 'real SMC 16 particles x 2 chains' if fit_kind == 'real' else 'synthetic posterior banks',
        'rounds': 1, 'all_artifacts_including_observational_sidecars': report,
    }, indent=2)+'\n')


@pytest.mark.usefixtures('require_boxinggym_source')
@pytest.mark.parametrize('design_policy', ['space_filling', 'point_disagreement'])
def test_boxing_new_multi_output_objective_metadata(tmp_path, monkeypatch, design_policy):
    """Test that boxing new multi output objective metadata."""
    bg_scalar_stub_fits(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail('new Box policy computed information')
    monkeypatch.setattr(box, 'joint_eig', forbidden)
    monkeypatch.setattr(box, 'choose_epig', forbidden)
    output = tmp_path/'multi'
    box.run(bg_run_args(output, task='emotion', cell='001', rounds=1,
                        design_policy=design_policy))
    config = json.loads((output/'config.json').read_text())
    acquisition = json.loads((output/'acquisition/t01.json').read_text())
    assert 'multi_component_eig_rule' not in config['task_spec']
    key = 'multi_component_point_disagreement_rule'
    if design_policy == 'point_disagreement':
        assert config['task_spec'][key] == 'sum_of_component_point_disagreements'
        record = acquisition['policy']
        assert record['component_rule'] == config['task_spec'][key]
        values = np.asarray(record['component_objectives'])
        assert values.shape == (8, len(acquisition['rows']))
        np.testing.assert_array_equal(values.sum(axis=0),
            [row['point_disagreement'] for row in acquisition['rows']])
    else:
        assert key not in config['task_spec']
        assert 'component_rule' not in acquisition['policy']
    assert json.loads((output/'summary.json').read_text())['paid_api_calls'] == 0
