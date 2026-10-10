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
