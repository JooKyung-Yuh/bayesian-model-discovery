"""Run label policy registration and runner gates; synthetic files, zero model calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .run_label_policy import (
    PAPER_NEURON_BINDINGS,
    PROMPT_BUNDLES,
    RULE_VERSION,
    bmd_runtime_settings,
    canonical_bytes,
    parse_llm_max_tokens,
    parse_llm_reasoning,
    policy_source_sha256,
    recheck_manifest,
    resolve_label,
)

#: Synthetic only: the policy leaves the number to the runner (the neuron runner sets its own).
PAPER_NEURON_ACTIVE_MAXIMUM = 6
#: The prompt identity a core/boxinggym/chembench runner puts on the namespace it
#: resolves its receipt with. Synthetic - no packaged prompt declares this version.
PROMPT_DECLARATION = {'prompt_version': 'v9.9', 'prompt_sha256': 'a' * 64}


def runner_args(**overrides):
    """A runner-shaped namespace, carrying the prompt identity policy v4 requires."""
    return SimpleNamespace(**{'profile': 'paper', 'rounds': 1, **PROMPT_DECLARATION, **overrides})


def registration(tmp_path, *, profile='paper', proposer='openai', cell='100',
                 bundle='core', task='structure_out__truth_F2', seed=11,
                 label='production', model=None, settings=None, host=None, order=None):
    """Write a synthetic registration manifest under `tmp_path`; return (manifest reference, verification context, manifest)."""
    if model is None and proposer != 'mock':
        model = 'gpt-4o' if proposer == 'openai' else 'test-model'
    if settings is None:
        namespace = runner_args(profile=profile)
        if profile == 'paper_neuron':
            namespace.active_maximum = PAPER_NEURON_ACTIVE_MAXIMUM
        settings = bmd_runtime_settings(bundle, namespace)
    rows = tmp_path / 'rows.txt'
    rows.write_text(f'{bundle} {task} {cell} {seed}\n')
    manifest = {
        'schema': 'research_wave_manifest_v1', 'run_label': label,
        'cohort': 'synthetic-test-only', 'provider': proposer, 'model': model,
        'base_url_host': host, 'provider_order': order,
        'runtime_settings': {bundle: {'profile': profile, **settings}},
        'rounds': {bundle: settings['rounds']},
        'policies': {bundle: {k: settings[k] for k in ('d0_policy', 'q_policy') if k in settings}},
        'calls': [{'bundle': bundle, 'file': rows.name, 'sha256': digest(rows.read_bytes())}],
    }
    path = tmp_path / 'manifest.json'
    ref = write_manifest(path, manifest)
    context = {'profile': profile, 'proposer': proposer, 'cell': cell, 'model': model, 'bundle': bundle, 'task': task, 'seed': seed, 'base_url_host': host, 'provider_order': order, 'settings': settings}
    return ref, context, manifest


def digest(data):
    """Return the lowercase hex SHA-256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def write_manifest(path, manifest):
    """Write `manifest` with its self-hash to `path` in canonical bytes; return its path/sha256 reference."""
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    manifest = {**body, 'manifest_sha256': digest(canonical_bytes(body))}
    path.write_bytes(canonical_bytes(manifest))
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


@pytest.mark.parametrize('profile', ['dev', 'paper'])
@pytest.mark.parametrize('proposer', ['mock', 'anthropic', 'openai', 'openai_compat'])
@pytest.mark.parametrize('cell', ['000', '100'])
@pytest.mark.parametrize('reference', ['missing', 'invalid', 'match', 'provider_mismatch', 'model_mismatch'])
@pytest.mark.parametrize('requested', [None, 'development', 'production'])
def test_full_truth_table(tmp_path, profile, proposer, cell, reference, requested):
    """Production labels need a paper profile, a paid proposer, a live cell and a matching registration; mismatches always fail."""
    host = 'openrouter.ai' if proposer == 'openai_compat' else None
    ref, context, _ = registration(tmp_path, profile=profile, proposer=proposer, cell=cell, host=host)
    if reference == 'missing':
        ref = None
    elif reference == 'invalid':
        ref['sha256'] = '0' * 64
    elif reference == 'provider_mismatch':
        context['proposer'] = 'anthropic' if proposer != 'anthropic' else 'openai'
    elif reference == 'model_mismatch':
        context['model'] = 'different-model'
    mismatch = reference not in ('missing', 'match')
    allowed = profile == 'paper' and proposer != 'mock' and cell[0] == '1' and reference == 'match'
    if mismatch or (requested == 'production' and not allowed):
        with pytest.raises(ValueError):
            resolve_label(requested, manifest_ref=ref, **context)
    else:
        result = resolve_label(requested, manifest_ref=ref, **context)
        assert result['run_label'] == (requested or 'development')
        assert result['rule_version'] == RULE_VERSION
        assert recheck_manifest(result) == result


@pytest.mark.parametrize('mutation', ['self_hash', 'schema', 'missing_cohort', 'model', 'host', 'order',
    'profile', 'particles', 'chains', 'rounds', 'rounds_conflict', 'policy_conflict',
    'call_digest', 'call_escape', 'task', 'cell', 'seed', 'row_duplicate', 'runtime_extra', 'bool_particle'])
def test_binding_failures(tmp_path, mutation):
    """Every mutation of the manifest or the runner context is refused for every requested label."""
    ref, context, manifest = registration(tmp_path)
    path = Path(ref['path'])
    if mutation == 'self_hash':
        raw = json.loads(path.read_bytes()); raw['manifest_sha256'] = 'a' * 64
        path.write_bytes(canonical_bytes(raw)); ref['sha256'] = digest(path.read_bytes())
    else:
        if mutation == 'schema': manifest['schema'] = 'unknown'
        elif mutation == 'missing_cohort': del manifest['cohort']
        elif mutation == 'model': manifest['model'] = 'wrong'
        elif mutation == 'host': manifest['base_url_host'] = 'elsewhere.invalid'
        elif mutation == 'order': manifest['provider_order'] = ['another']
        elif mutation == 'profile': manifest['runtime_settings']['core']['profile'] = 'dev'
        elif mutation in ('particles', 'chains', 'rounds'): context['settings'][mutation] += 1
        elif mutation == 'rounds_conflict': manifest['rounds']['core'] = 2
        elif mutation == 'policy_conflict': manifest['policies']['core']['d0_policy'] = 'other'
        elif mutation == 'call_digest': manifest['calls'][0]['sha256'] = '0' * 64
        elif mutation == 'call_escape': manifest['calls'][0]['file'] = '../rows.txt'
        elif mutation in ('task', 'cell', 'seed'): context[mutation] = {'task': 'other', 'cell': '101', 'seed': 17}[mutation]
        elif mutation == 'row_duplicate': manifest['calls'].append(copy.deepcopy(manifest['calls'][0]))
        elif mutation == 'runtime_extra': context['settings']['unregistered'] = 1
        elif mutation == 'bool_particle': manifest['runtime_settings']['core']['particles'] = True
        ref = write_manifest(path, manifest)
    for requested in (None, 'development', 'production'):
        with pytest.raises(ValueError):
            resolve_label(requested, manifest_ref=ref, **context)


@pytest.mark.parametrize('change', ['manifest', 'rows', 'receipt_label', 'receipt_binding', 'receipt_row'])
def test_publication_recheck_catches_replacement(tmp_path, change):
    """Re-checking a receipt at publication catches a replaced manifest, rows file, label, binding or row."""
    ref, context, _ = registration(tmp_path)
    receipt = resolve_label('production', manifest_ref=ref, **context)
    if change in ('manifest', 'rows'):
        path = Path(ref['path']) if change == 'manifest' else tmp_path / 'rows.txt'
        path.write_bytes(path.read_bytes() + b'\n')
    elif change == 'receipt_label': receipt['run_label'] = 'development'
    elif change == 'receipt_binding': receipt['binding']['seed'] = 17
    else: receipt['matched_row']['line'] = 2
    with pytest.raises(ValueError): recheck_manifest(receipt)


def test_development_manifest_cannot_authorize_production(tmp_path):
    """A development manifest never authorizes a production label."""
    ref, context, _ = registration(tmp_path, label='development')
    assert resolve_label(None, manifest_ref=ref, **context)['run_label'] == 'development'
    with pytest.raises(ValueError): resolve_label('production', manifest_ref=ref, **context)


def test_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    """A manifest with duplicate keys or non-finite numbers is refused."""
    ref, context, _ = registration(tmp_path)
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        Path(ref['path']).write_bytes(raw); ref['sha256'] = digest(raw)
        with pytest.raises(ValueError): resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench', 'neuron'])
def test_actual_runner_settings(bundle):
    """The runtime settings declared for each bundle carry the expected particle, chain and policy values."""
    args = runner_args(rounds=16)
    settings = bmd_runtime_settings(bundle, args)
    assert settings['particles'] == 500 and settings['chains'] == 2
    if bundle in ('core', 'boxinggym'): assert settings['proposal_policy'] == 'P1'
    if bundle == 'chembench':
        assert settings['q'] == 256 and settings['d0_fixed_inputs'] is None
        args.q = 7; args.d0_fixed_inputs = {'T': 300.0}
        assert bmd_runtime_settings(bundle, args)['q'] == 7
        assert bmd_runtime_settings(bundle, args)['d0_fixed_inputs'] == {'T': 300.0}


class ReachedValidatedOutput(Exception):
    """Stop before any live client or scientific run in runner gate tests."""


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench', 'neuron'])
@pytest.mark.parametrize('case', ['valid', 'missing', 'bad_digest', 'different_settings'])
def test_runner_gate_before_output(tmp_path, monkeypatch, bundle, case):
    """Each runner verifies the registration before it creates any output folder."""
    out = tmp_path / 'must-not-be-created'
    if bundle == 'core':
        from . import core_count as runner
        from .test_count_proposer import run_args
        args = run_args(out, profile='paper', proposer='openai', llm_model='gpt-4o',
                        allow_paid_api=True, run_label='production', rounds=1, cell='100')
        task = 'structure_out__truth_F2'
    elif bundle == 'boxinggym':
        from .test_paper_profile_seal import adapter, bg_run_args
        runner = adapter()
        args = bg_run_args(out, profile='paper', proposer='openai', llm_model='gpt-4o',
                           allow_paid_api=True, run_label='production', cell='100')
        task = args.task
    elif bundle == 'chembench':
        from . import chembench_core as runner
        from .test_chembench_core import _args
        oracle_manifest = tmp_path / 'oracle.json'
        oracle_manifest.write_text(json.dumps([{'id': 'synthetic', 'domain': 'd0', 'difficulty': 'easy', 'law_version': 'v0'}]))
        from bayes_model_discovery.adapters import chembench_observation
        monkeypatch.setattr(chembench_observation, 'sample_designs',
                            lambda count, seed: [{'C_A': float(seed) + i} for i in range(count)])
        args = _args(out, oracle_manifest, profile='paper', proposer='openai',
                     llm_model='gpt-4o', allow_paid_api=True, run_label='production',
                     particles=None, q=None, task='synthetic', cell='100', structure_proposals='free')
        task = args.task
    else:
        pytest.importorskip('neuronbench')
        from . import neuron_core as runner
        args = runner.build_parser().parse_args(['--output', str(out), '--profile', 'paper',
                    '--proposer', 'openai', '--llm-model', 'gpt-4o', '--allow-paid-api',
                    '--run-label', 'production', '--world', 'h_sag', '--cell', '110', '--rounds', '1'])
        task = args.world
    declared = args
    if bundle in PROMPT_BUNDLES:
        from .count_proposer import prompt_identity
        declared = SimpleNamespace(**{**vars(args),
                                      **prompt_identity(getattr(args, 'priors', 'off') == 'llm')})
    ref, _, _ = registration(tmp_path, bundle=bundle, task=task, seed=args.seed, cell=args.cell,
                              settings=bmd_runtime_settings(bundle, declared))
    if case != 'missing': args.manifest_ref = json.dumps(ref)
    if case == 'bad_digest':
        ref['sha256'] = '0' * 64; args.manifest_ref = json.dumps(ref)
    if case == 'different_settings': args.rounds += 1
    original_mkdir = Path.mkdir
    def stop_at_output(path, *a, **kw):
        """Raise once the runner reaches the validated output folder; otherwise behave like Path.mkdir."""
        if path == out: raise ReachedValidatedOutput
        return original_mkdir(path, *a, **kw)
    monkeypatch.setattr(Path, 'mkdir', stop_at_output)
    with pytest.raises(ReachedValidatedOutput if case == 'valid' else ValueError):
        runner.run(args)
    assert not out.exists()


@pytest.mark.parametrize('bundle,key,new_value', [
    ('core', 'proposal_rounds', '20'), ('core', 'proposal_limit', 2),
    ('core', 'repair_attempts', 0), ('core', 'replicate', 1),
    ('core', 'design_policy', 'model_eig'), ('boxinggym', 'truth_in_pool', True),
    ('boxinggym', 'prior_template', 'other'), ('chembench', 'noise', .2),
    ('chembench', 'q', 8), ('chembench', 'd0_fixed_inputs', {'T': 300.0}),
])
def test_scientific_override_needs_its_own_registration(tmp_path, bundle, key, new_value):
    """Overriding a scientific setting requires its own registration."""
    ref, context, _ = registration(tmp_path, bundle=bundle)
    args = runner_args(**{key: new_value})
    context['settings'] = bmd_runtime_settings(bundle, args)
    with pytest.raises(ValueError, match='runtime settings'):
        resolve_label('production', manifest_ref=ref, **context)


def test_policy_source_is_part_of_receipt(tmp_path):
    """The receipt carries the policy source digest, and a wrong digest fails the re-check."""
    from . import run_label_policy
    ref, context, _ = registration(tmp_path)
    receipt = resolve_label('production', manifest_ref=ref, **context)
    assert receipt['policy_source_sha256'] == digest(Path(run_label_policy.__file__).read_bytes())
    receipt['policy_source_sha256'] = '0' * 64
    with pytest.raises(ValueError): recheck_manifest(receipt)


# ------------------------------------------------------------------ neuron profile and decoding options
# Introduced with rule v3: the `paper_neuron` profile, the neuron forward backend and the two
# decoding options become registered settings. Every fixture below is synthetic.
REPOSITORY = Path(__file__).resolve().parents[3]
SNAPSHOT = REPOSITORY / 'tools/fb1_snapshot'


def neuron_args(**overrides):
    """A runner-shaped namespace for the neuron bundle's declaration."""
    values = {'profile': 'paper_neuron', 'rounds': 1, 'particles': 200,
              'active_maximum': PAPER_NEURON_ACTIVE_MAXIMUM}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_p1_rule_version_snapshot_copy_and_hash_row():
    """The rule version, the snapshot copy of the policy and its hash row agree."""
    assert RULE_VERSION == 'run_label_policy_v4_20260918'
    source = (REPOSITORY / 'src/bayes_model_discovery/experimental/run_label_policy.py').read_bytes()
    assert (SNAPSHOT / 'run_label_policy.py').read_bytes() == source
    rows = (SNAPSHOT / 'SHA256SUMS').read_text().splitlines()
    entry = next(row for row in rows if row.endswith('  ./run_label_policy.py'))
    assert entry.split()[0] == digest(source)
    runner = next(row for row in rows if row.endswith('  ./run_gravity.py'))
    assert runner.split()[0] == digest((SNAPSHOT / 'run_gravity.py').read_bytes())


@pytest.mark.parametrize('proposer', ['mock', 'openai'])
@pytest.mark.parametrize('cell', ['000', '100'])
@pytest.mark.parametrize('reference', ['missing', 'match', 'mismatch'])
@pytest.mark.parametrize('requested', [None, 'development', 'production'])
def test_p1_paper_neuron_truth_table(tmp_path, proposer, cell, reference, requested):
    """Truth table of the neuron paper profile: label, reference and cell against what is allowed."""
    ref, context, _ = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                   proposer=proposer, cell=cell, task='h_sag')
    if reference == 'missing':
        ref = None
    elif reference == 'mismatch':
        context['seed'] = 17
    allowed = proposer != 'mock' and cell[0] == '1' and reference == 'match'
    if reference == 'mismatch' or (requested == 'production' and not allowed):
        with pytest.raises(ValueError):
            resolve_label(requested, manifest_ref=ref, **context)
    else:
        result = resolve_label(requested, manifest_ref=ref, **context)
        assert result['run_label'] == (requested or 'development')
        assert result['rule_version'] == RULE_VERSION
        assert recheck_manifest(result) == result


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench'])
def test_p1_paper_neuron_is_refused_outside_the_neuron_bundle(tmp_path, bundle):
    """The neuron paper profile is refused for every other bundle."""
    ref, context, manifest = registration(tmp_path, bundle=bundle, task='other')
    manifest['runtime_settings'][bundle]['profile'] = 'paper_neuron'
    ref = write_manifest(Path(ref['path']), manifest)
    context['profile'] = 'paper_neuron'
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('key,value', [
    ('particles', 500), ('chains', 3), ('kernel_policy', 'mh_adaptive'),
    ('steps_per_stage', 4), ('ess_threshold', 0.9), ('feature_space', 'features_mV_3d'),
    ('active_maximum', 0), ('active_maximum', 'six'), ('active_maximum', True),
])
def test_p1_paper_neuron_value_bindings_are_refused_when_wrong(tmp_path, key, value):
    """Wrong registered values for the neuron profile's bound settings are refused."""
    ref, context, manifest = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                          task='h_sag')
    manifest['runtime_settings']['neuron'][key] = value
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('key', sorted({*PAPER_NEURON_BINDINGS, 'active_maximum'}))
def test_p1_paper_neuron_required_keys_are_not_optional(tmp_path, key):
    """Each required neuron setting must be present in the registration."""
    ref, context, manifest = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                          task='h_sag')
    del manifest['runtime_settings']['neuron'][key]
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError, match='Incomplete registered runtime settings'):
        resolve_label(None, manifest_ref=ref, **context)


def test_p1_paper_neuron_declaration_must_equal_the_registration(tmp_path):
    """The neuron runner's declaration must equal the registered settings."""
    ref, context, _ = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                   task='h_sag')
    declared = bmd_runtime_settings('neuron', neuron_args())
    assert declared['particles'] == 200 and declared['chains'] == 2
    assert {key: declared[key] for key in PAPER_NEURON_BINDINGS} == PAPER_NEURON_BINDINGS
    assert declared['active_maximum'] == PAPER_NEURON_ACTIVE_MAXIMUM
    assert declared['forward_backend'] == 'native'
    context['settings'] = bmd_runtime_settings(
        'neuron', neuron_args(active_maximum=PAPER_NEURON_ACTIVE_MAXIMUM + 6))
    with pytest.raises(ValueError, match='runtime settings'):
        resolve_label(None, manifest_ref=ref, **context)
    with pytest.raises(ValueError, match='active maximum'):
        bmd_runtime_settings('neuron', neuron_args(active_maximum=None))


@pytest.mark.parametrize('case', [
    'registered_batch_runner_native', 'registered_native_runner_batch', 'missing_sha',
    'short_sha', 'uppercase_sha', 'mismatched_sha', 'sha_on_native', 'unknown_backend',
])
def test_p1_neuron_forward_backend_is_registered(tmp_path, case):
    """The neuron forward backend and its equivalence digest must match between registration and runner."""
    report = 'a' * 64
    batch = neuron_args(forward_backend='batch', forward_equivalence_sha=report)
    registered = bmd_runtime_settings('neuron', batch if 'registered_batch' in case
                                      or case in ('missing_sha', 'short_sha', 'uppercase_sha',
                                                  'mismatched_sha', 'unknown_backend')
                                      else neuron_args())
    ref, context, manifest = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                          task='h_sag', settings=registered)
    settings = manifest['runtime_settings']['neuron']
    if case == 'registered_batch_runner_native':
        context['settings'] = bmd_runtime_settings('neuron', neuron_args())
    elif case == 'registered_native_runner_batch':
        context['settings'] = bmd_runtime_settings('neuron', batch)
    elif case == 'missing_sha':
        del settings['forward_equivalence_sha']
        context['settings'] = dict(context['settings'])
        context['settings'].pop('forward_equivalence_sha')
    elif case == 'short_sha':
        settings['forward_equivalence_sha'] = 'a' * 63
        context['settings']['forward_equivalence_sha'] = 'a' * 63
    elif case == 'uppercase_sha':
        settings['forward_equivalence_sha'] = 'A' * 64
        context['settings']['forward_equivalence_sha'] = 'A' * 64
    elif case == 'mismatched_sha':
        settings['forward_equivalence_sha'] = 'b' * 64
    elif case == 'sha_on_native':
        settings['forward_equivalence_sha'] = report
        context['settings']['forward_equivalence_sha'] = report
    else:
        settings['forward_backend'] = 'vectorised'
        context['settings']['forward_backend'] = 'vectorised'
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


def test_p1_matching_batch_forward_is_accepted(tmp_path):
    """A batch forward backend with a matching equivalence digest is accepted."""
    batch = neuron_args(forward_backend='batch', forward_equivalence_sha='a' * 64)
    settings = bmd_runtime_settings('neuron', batch)
    assert settings['forward_backend'] == 'batch'
    assert settings['forward_equivalence_sha'] == 'a' * 64
    ref, context, _ = registration(tmp_path, profile='paper_neuron', bundle='neuron',
                                   task='h_sag', settings=settings)
    assert resolve_label(None, manifest_ref=ref, **context)['run_label'] == 'development'
    with pytest.raises(ValueError, match='SHA256'):
        bmd_runtime_settings('neuron', neuron_args(forward_backend='batch'))
    with pytest.raises(ValueError, match='native or batch'):
        bmd_runtime_settings('neuron', neuron_args(forward_backend='cuda'))


@pytest.mark.parametrize('spec,expected', [
    ('effort=high', {'effort': 'high'}),
    ('effort=none,exclude', {'effort': 'none', 'exclude': True}),
    ('exclude, max_tokens=512', {'max_tokens': 512, 'exclude': True}),
    ({'effort': 'minimal'}, {'effort': 'minimal'}),
    ({'exclude': True, 'max_tokens': 7}, {'max_tokens': 7, 'exclude': True}),
    # `thinking=adaptive` rides with an effort level, canonically last.
    ('effort=high,thinking=adaptive', {'effort': 'high', 'thinking': 'adaptive'}),
    ({'thinking': 'adaptive', 'effort': 'low'}, {'effort': 'low', 'thinking': 'adaptive'}),
    ('thinking=adaptive,exclude,effort=max', {'effort': 'max', 'exclude': True, 'thinking': 'adaptive'}),
    ('display=summarized,thinking=adaptive,effort=medium',
     {'effort': 'medium', 'thinking': 'adaptive', 'display': 'summarized'}),
    ({'display': 'summarized', 'thinking': 'adaptive', 'effort': 'medium'},
     {'effort': 'medium', 'thinking': 'adaptive', 'display': 'summarized'}),
    ('display=summarized,exclude,effort=medium,thinking=adaptive',
     {'effort': 'medium', 'exclude': True, 'thinking': 'adaptive', 'display': 'summarized'}),
])
def test_p1_reasoning_grammar_matches_the_runner_parser(spec, expected):
    """The policy's reasoning grammar parses exactly like the runner's parser."""
    from .count_proposer import normalize_reasoning
    assert parse_llm_reasoning(spec) == expected == normalize_reasoning(spec)
    assert list(parse_llm_reasoning(spec)) == list(expected) == list(normalize_reasoning(spec))


@pytest.mark.parametrize('spec', [
    '', '   ', 'effort=high,max_tokens=512', 'effort=louder', 'effort', 'exclude=true',
    'max_tokens=0', 'max_tokens=-1', 'max_tokens=x', 'effort=high,effort=low',
    {'effort': 'high', 'max_tokens': 4}, {'exclude': True}, {'effort': 'high', 'extra': 1},
    {'max_tokens': True}, {'max_tokens': 1.5}, {'effort': 'high', 'exclude': 'yes'}, 7, None, [],
    # A thinking field without an effort level, or with any other value, is refused.
    'thinking=adaptive', 'max_tokens=512,thinking=adaptive', 'effort=high,thinking=enabled',
    {'effort': 'high', 'thinking': 'high'}, {'max_tokens': 5, 'thinking': 'adaptive'},
    'display=summarized', 'effort=medium,display=summarized',
    'effort=medium,thinking=adaptive,display=omitted',
    'effort=medium,thinking=adaptive,display=updates',
    'effort=medium,thinking=adaptive,display=summarized,display=summarized',
    'max_tokens=512,thinking=adaptive,display=summarized',
    {'effort': 'medium', 'display': 'summarized'},
    {'effort': 'medium', 'thinking': 'adaptive', 'display': 'omitted'},
    {'max_tokens': 512, 'thinking': 'adaptive', 'display': 'summarized'},
])
def test_p1_reasoning_grammar_refuses_what_the_runner_refuses(spec):
    """The policy refuses every reasoning spec the runner's parser refuses."""
    from .count_proposer import normalize_reasoning
    with pytest.raises(ValueError):
        parse_llm_reasoning(spec)
    if spec is not None:
        with pytest.raises(ValueError):
            normalize_reasoning(spec)


@pytest.mark.parametrize('value', [0, -1, 32769, True, 1.0, '0', 'x', None, {}])
def test_p1_max_tokens_range_is_refused_outside_one_to_the_limit(value):
    """A max-tokens value outside 1 to the limit is refused."""
    with pytest.raises(ValueError):
        parse_llm_max_tokens(value)


@pytest.mark.parametrize('value,expected', [(1, 1), (32768, 32768), ('4000', 4000)])
def test_p1_max_tokens_accepts_the_stated_budget(value, expected):
    """A max-tokens value inside the range parses to the stated integer."""
    assert parse_llm_max_tokens(value) == expected


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench', 'neuron'])
def test_p1_stated_decoding_is_a_registered_setting(tmp_path, bundle):
    """Decoding options stated by a run become registered settings that the manifest must carry."""
    stated = {'llm_reasoning': {'effort': 'none'}, 'llm_max_tokens': 4000}
    args = neuron_args(profile='paper', particles=None) if bundle == 'neuron' else \
        runner_args()
    for key, value in stated.items():
        setattr(args, key, value)
    settings = bmd_runtime_settings(bundle, args)
    assert {key: settings[key] for key in stated} == stated
    ref, context, manifest = registration(tmp_path, bundle=bundle, task='h_sag',
                                          settings=settings)
    assert resolve_label('production', manifest_ref=ref, **context)['run_label'] == 'production'
    # The same run without the registration of one option is refused.
    del manifest['runtime_settings'][bundle]['llm_reasoning']
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError, match='runtime settings'):
        resolve_label('production', manifest_ref=ref, **context)


@pytest.mark.parametrize('value', ['effort=high', {'effort': 'louder'}, {'max_tokens': 0},
                                   {'effort': 'high', 'max_tokens': 4}])
def test_p1_registered_reasoning_must_obey_the_grammar(tmp_path, value):
    """A registered reasoning value that breaks the grammar is refused."""
    stated = {'llm_reasoning': {'effort': 'none'}}
    args = runner_args(**stated)
    ref, context, manifest = registration(tmp_path, settings=bmd_runtime_settings('core', args))
    manifest['runtime_settings']['core']['llm_reasoning'] = value
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('value', ['4000', 0, 32769, True])
def test_p1_registered_max_tokens_must_obey_the_range(tmp_path, value):
    """A registered max-tokens value outside the range or of the wrong type is refused."""
    args = runner_args(llm_max_tokens=4000)
    ref, context, manifest = registration(tmp_path, settings=bmd_runtime_settings('core', args))
    manifest['runtime_settings']['core']['llm_max_tokens'] = value
    ref = write_manifest(Path(ref['path']), manifest)
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


def test_p1_unstated_decoding_leaves_the_declaration_unchanged():
    """A run that states no decoding option declares none."""
    plain = bmd_runtime_settings('core', runner_args())
    assert not {'llm_reasoning', 'llm_max_tokens'} & set(plain)


def forcebench_label_settings(**environment):
    """run_gravity's own settings declaration, executed from its source.

    Only the one function is compiled, with a synthetic namespace in place of the
    module's globals, so this needs neither the science stack nor a run.
    """
    source = SNAPSHOT / 'run_gravity.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_label_runtime_settings')
    from . import run_label_policy
    namespace = {'core': SimpleNamespace(N_PARTICLES=500, N_CHAINS=2), 'T_EXTRA': 16,
                 'D0_POLICY': 'mda1', 'D0': 1, 'D0_POINTS': ((1., 1.),),
                 'FB1_CHECKPOINTS_OPTION': 'all', 'CHECKPOINTS': tuple(range(1, 18)),
                 'NUMERICS': {'mode': 'match_truth', 'design_gate': 'dt_halving',
                              'noise_inflate': False},
                 'DT': .005, 'DESIGN_VARIANT': 'none', 'PROPOSAL_POLICY': 'P4',
                 'PROPOSE_AT': (4, 8, 12, 16), 'STRUCTURE_PROPOSALS': 'catalog',
                 'EPIG_DESIGN': False, 'os': SimpleNamespace(environ=dict(environment)),
                 'label_policy': run_label_policy,
                 # The unified-protocol declarations run_gravity added to this function, at the
                 # values its module sets when no FB1_* option for them is stated.
                 'ASCENT': False, 'DESIGN_POLICY': None, 'EPIG_SCORE_Q': False, 'GRAMMAR_TRIG': False,
                 'MAX_ACTIVE': 12, 'BURST_SIZE': 0, 'BURST': False, 'ACTIVE_MAXIMUM_STATED': False,
                 'COMPOSED': False, 'COMPOSED_POLICY': 'composed_3', 'COMPOSED_LIMIT': 3,
                 'COMPOSED_EVENTS': tuple(range(2, 18)), 'REVIEW_ROUNDS': (), 'CATALOGUE': 'on',
                 'CATALOGUE_STATED': False, 'PACKET_ARCHIVE': 'full'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)  # noqa: S102 - the runner's own declaration, compiled without its science stack.
    return namespace['_label_runtime_settings']


def test_p1_forcebench_declares_only_the_decoding_options_it_sends():
    """ForceBench declares only the decoding options its provider actually sends."""
    stated = {'FB1_LLM_REASONING': 'effort=none,exclude', 'FB1_LLM_MAX_TOKENS': '6000'}
    assert not {'llm_reasoning', 'llm_max_tokens'} & set(forcebench_label_settings()())
    assert not {'llm_reasoning', 'llm_max_tokens'} & set(
        forcebench_label_settings(**stated)('mock'))
    compat = forcebench_label_settings(**stated)('openai_compat')
    assert compat['llm_reasoning'] == {'effort': 'none', 'exclude': True}
    assert compat['llm_max_tokens'] == 6000
    # `reasoning` is a compatibility-endpoint extension; a native provider sends only
    # the reply budget, so only that one is declared.
    native = forcebench_label_settings(**stated)('openai')
    assert 'llm_reasoning' not in native and native['llm_max_tokens'] == 6000
    with pytest.raises(ValueError):
        forcebench_label_settings(FB1_LLM_REASONING='effort=louder')('openai_compat')
    with pytest.raises(ValueError):
        forcebench_label_settings(FB1_LLM_MAX_TOKENS='99999')('openai')


def test_p1_cross_version_recheck_is_an_explicit_refusal(tmp_path):
    """A receipt from another rule version or policy copy is refused explicitly."""
    ref, context, _ = registration(tmp_path)
    receipt = resolve_label('production', manifest_ref=ref, **context)
    assert receipt['rule_version'] == RULE_VERSION
    assert recheck_manifest(receipt) == receipt
    v2 = {**receipt, 'rule_version': 'run_label_policy_v2_20260914',
          'policy_source_sha256': '0' * 64}
    with pytest.raises(ValueError, match='Unsupported label receipt'):
        recheck_manifest(v2)
    # A receipt naming this rule version but issued by another copy of the file is refused
    # too: the recomputed receipt carries this file's digest, so the comparison fails.
    assert receipt['policy_source_sha256'] == policy_source_sha256()
    with pytest.raises(ValueError, match='no longer matches its registration'):
        recheck_manifest({**receipt, 'policy_source_sha256': '0' * 64})
    # And a current-version receipt read by the v2 copy a finished cohort ran on: that file's own
    # RULE_VERSION differs, which is the same refusal from the other side.
    assert 'v2' not in RULE_VERSION


# ---------------------------------------------------------------------------
# Prompt identity: prompt_version and prompt_sha256 as declared, required, verified keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench'])
def test_p1v_every_llm_bundle_declares_the_prompt_it_ran(bundle):
    """Every LLM bundle declares its prompt version and digest."""
    settings = bmd_runtime_settings(bundle, runner_args())
    assert {k: settings[k] for k in PROMPT_DECLARATION} == PROMPT_DECLARATION


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench'])
@pytest.mark.parametrize('missing', ['prompt_version', 'prompt_sha256'])
def test_p1v_a_run_that_declares_no_prompt_cannot_be_registered(bundle, missing):
    """A run that declares no prompt version or digest cannot be registered."""
    args = runner_args(**{missing: None})
    with pytest.raises(ValueError):
        bmd_runtime_settings(bundle, args)


@pytest.mark.parametrize('bad', ['', 'A' * 64, 'abc', 'a' * 63])
def test_p1v_a_prompt_digest_must_be_a_lowercase_sha256(bad):
    """A prompt digest that is not a lowercase SHA-256 is refused."""
    with pytest.raises(ValueError, match='lowercase SHA256'):
        bmd_runtime_settings('core', runner_args(prompt_sha256=bad))


def test_p1v_the_neuron_and_forcebench_bundles_declare_no_prompt():
    """The neuron and ForceBench bundles declare no prompt identity."""
    declared = bmd_runtime_settings('neuron', neuron_args())
    assert not set(PROMPT_DECLARATION) & set(declared)
    assert 'forcebench' not in PROMPT_BUNDLES and 'neuron' not in PROMPT_BUNDLES


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench'])
@pytest.mark.parametrize('key', ['prompt_version', 'prompt_sha256'])
def test_p1v_a_manifest_without_the_prompt_keys_is_refused(tmp_path, bundle, key):
    """A manifest lacking the prompt keys is refused."""
    ref, context, manifest = registration(tmp_path, bundle=bundle,
                                          task='synthetic_task' if bundle != 'core' else 'structure_out__truth_F2')
    del manifest['runtime_settings'][bundle][key]
    write_manifest(Path(ref['path']), manifest)
    ref['sha256'] = digest(Path(ref['path']).read_bytes())
    with pytest.raises(ValueError, match='Incomplete registered runtime settings'):
        resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('bundle', ['core', 'boxinggym', 'chembench'])
@pytest.mark.parametrize('key,value', [('prompt_version', 'v0.1'), ('prompt_sha256', 'b' * 64)])
def test_p1v_a_registered_prompt_that_disagrees_with_the_run_is_refused(tmp_path, bundle, key, value):
    """A registered prompt version or digest that differs from the run's is refused."""
    ref, context, manifest = registration(tmp_path, bundle=bundle,
                                          task='synthetic_task' if bundle != 'core' else 'structure_out__truth_F2')
    manifest['runtime_settings'][bundle][key] = value
    write_manifest(Path(ref['path']), manifest)
    ref['sha256'] = digest(Path(ref['path']).read_bytes())
    with pytest.raises(ValueError, match='Manifest runtime settings mismatch'):
        resolve_label(None, manifest_ref=ref, **context)


@pytest.mark.parametrize('value', ['', 5, None])
def test_p1v_a_registered_prompt_version_must_be_a_nonempty_string(tmp_path, value):
    """A registered prompt version must be a non-empty string."""
    ref, context, manifest = registration(tmp_path)
    manifest['runtime_settings']['core']['prompt_version'] = value
    write_manifest(Path(ref['path']), manifest)
    ref['sha256'] = digest(Path(ref['path']).read_bytes())
    with pytest.raises(ValueError, match='Registered prompt version'):
        resolve_label(None, manifest_ref=ref, **context)



# --------------------------------------------------------------------- validation messages and response format
def test_em_the_two_options_are_registered_exactly_when_stated(tmp_path):
    """Validation messages (v2) and response format are registered when a run states them; v1 is not registered."""
    from .run_label_policy import DECODING_KEYS, parse_llm_response_format
    assert DECODING_KEYS == ('llm_reasoning', 'llm_max_tokens', 'llm_response_format')
    plain = bmd_runtime_settings('core', runner_args())
    assert 'validation_messages' not in plain and 'llm_response_format' not in plain
    assert 'validation_messages' not in bmd_runtime_settings('core', runner_args(validation_messages='v1'))
    stated = bmd_runtime_settings('core', runner_args(validation_messages='v2', llm_response_format='json_object'))
    assert stated['validation_messages'] == 'v2' and stated['llm_response_format'] == 'json_object'
    for bad in ('v3', 7):
        with pytest.raises(ValueError, match='Validation messages'):
            bmd_runtime_settings('core', runner_args(validation_messages=bad))
    for bad in ('json_schema', None, 7):
        with pytest.raises(ValueError, match='llm_response_format'):
            parse_llm_response_format(bad)
    # A registration binds both; a manifest that leaves one out, or states v1, is refused.
    ref, context, manifest = registration(tmp_path, settings=stated)
    assert resolve_label('production', manifest_ref=ref, **context)['run_label'] == 'production'
    for key in ('validation_messages', 'llm_response_format'):
        broken = json.loads(json.dumps(manifest))
        del broken['runtime_settings']['core'][key]
        with pytest.raises(ValueError, match='runtime settings'):
            resolve_label('production', manifest_ref=write_manifest(Path(ref['path']), broken), **context)
    broken = json.loads(json.dumps(manifest))
    broken['runtime_settings']['core']['validation_messages'] = 'v1'
    with pytest.raises(ValueError, match='validation_messages is v2'):
        resolve_label('production', manifest_ref=write_manifest(Path(ref['path']), broken), **context)
