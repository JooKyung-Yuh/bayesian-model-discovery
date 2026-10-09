"""Explicit run labels bound to a declared cohort; integrity is not authorization.

Standard library only. The ForceBench snapshot carries an exact-byte mirror.
No existing result is relabelled by this module.
"""
import hashlib
import json
import math
import re
from pathlib import Path

RULE_VERSION = "run_label_policy_v4_20260918"
RUN_LABEL_CHOICES = ("development", "production")
PROVIDERS = ("mock", "anthropic", "openai", "openai_compat")
PROFILES = ("dev", "paper", "paper_neuron")
#: Published CORE receipts keep their sentinel; config records the actual rule.
RECEIPT_D0_POLICY = {"fixed4": "runner_default", "stratified4": "stratified4"}
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
#: `paper_neuron` is the MDA-comparison profile of the neuron bundle. It is not
#: a coarser `paper` - the kernel and the training feature space differ - so the registration
#: binds both, and a registered value is the executed value or the run is refused.
PAPER_NEURON_BINDINGS = {'kernel_policy': 'mh_fixed_steps', 'steps_per_stage': 3,
                         'ess_threshold': 0.5, 'feature_space': 'eval_features_6d'}
#: The two decoding options a run may state. Optional everywhere, but a stated
#: one is part of the registered settings, so production refuses an unregistered value.
DECODING_KEYS = ('llm_reasoning', 'llm_max_tokens', 'llm_response_format')
#: The one response format a run may state (OpenAI JSON mode); registered exactly
#: when stated, like the two decoding options before it.
RESPONSE_FORMATS = ('json_object',)
#: The validation-message version a run may state (count_models.set_validation_messages);
#: the default v1 leaves no key, so every existing declaration and receipt keeps its shape.
VALIDATION_MESSAGES_KEY = 'validation_messages'
VALIDATION_MESSAGE_VERSIONS = ('v1', 'v2')
DEFAULT_VALIDATION_MESSAGES = 'v1'
#: The system prompt is part of what a run claims to have done, so the three LLM
#: bundles DECLARE its version and the digest of the bytes they read, the registration must
#: state both, and the existing whole-dict comparison refuses a run that sent another text.
#: The neuron bundle records nothing new (it is not in the prompt-variant cohort) and
#: ForceBench keeps its snapshot proposer's own frozen digest.
PROMPT_BUNDLES = ('core', 'boxinggym', 'chembench')
PROMPT_KEYS = ('prompt_version', 'prompt_sha256')
# The expression grammar a registration binds. The key is
# written ONLY when the version is not the default, so every registration made
# before this key existed - and every v2 registration made after it - keeps its exact
# bytes, and an ABSENT key means v2. `verify_manifest` compares the launch's
# value with the registered one under that reading, exactly as it compares the
# prompt version.
GRAMMAR_VERSION_KEY = 'grammar_version'
GRAMMAR_VERSIONS = ('v2', 'v3')
#: The BoxingGym runner's active-pool cap when none is stated (core_count.ACTIVE_MAXIMUM).
BOXINGGYM_ACTIVE_MAXIMUM = 12
CORE_ACTIVE_MAXIMUM = 12
#: The two refill digests a burst with initial proposals binds - the composed refill
#: request and count_proposer.BURST_REFILL_PARAGRAPH alone (a project decision of
#: 2026-09-23).
BURST_REFILL_KEYS = ('burst_refill_prompt_sha256', 'burst_refill_paragraph_sha256')
#: Refill-policy binding: the refill rule such a burst runs under (count_models.INITIAL_BURST_REFILL_POLICY),
#: bound beside the two digests under the same presence rule.
BURST_REFILL_POLICY_KEY = 'initial_burst_refill_policy'
DEFAULT_GRAMMAR_VERSION = 'v2'
GRAMMAR_BUNDLES = PROMPT_BUNDLES


def _grammar_version(value):
    """The declared expression grammar, with an absent declaration meaning v2."""
    if value is None:
        return DEFAULT_GRAMMAR_VERSION
    if value not in GRAMMAR_VERSIONS:
        raise ValueError('Registered grammar version must be one of '
                         + ', '.join(GRAMMAR_VERSIONS))
    return value
REASONING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
MAX_TOKENS_LIMIT = 32768
REASONING_ERROR = ("llm_reasoning takes effort=<" + "|".join(REASONING_EFFORTS) + ">, "
                   "max_tokens=<positive integer> and the optional flag exclude"
                   " (and, with effort, thinking=adaptive with optional display=summarized)")
MAX_TOKENS_ERROR = f"llm_max_tokens must be an integer in 1..{MAX_TOKENS_LIMIT}"


def canonical_bytes(value):
    """Serialize a JSON value canonically (sorted keys, two-space indent, trailing newline) as UTF-8 bytes."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in label registration")
        result[key] = value
    return result


def _json(text):
    try:
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite registration value")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid label registration JSON") from exc


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Label registration requires a lowercase SHA256")
    return value


def _read(path, expected):
    _sha(expected)
    try:
        with Path(path).open('rb') as stream:
            data = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ValueError("Label registration file is unavailable") from exc
    if len(data) > MAX_MANIFEST_BYTES or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Label registration whole-file SHA mismatch or oversized file")
    return data


def _reference(value):
    if isinstance(value, str):
        value = _json(value)
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("manifest_ref must contain exactly path and sha256")
    if not isinstance(value['path'], str) or not value['path']:
        raise ValueError("manifest_ref requires a path")
    return {"path": str(Path(value['path']).resolve()), "sha256": _sha(value['sha256'])}


def parse_llm_reasoning(value):
    """One canonical reasoning spec: effort XOR max_tokens, plus the optional exclude flag.

    The grammar `count_proposer.parse_reasoning_spec` enforces on the command line, validated
    here as well because this module is standard library only (its ForceBench mirror is an
    exact-byte copy and cannot import the package). A test pins the two to agree. Text is
    accepted so a runner that reads the option from its environment can declare it.
    """
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(REASONING_ERROR)
        fields = {}
        for field in value.split(','):
            key, separator, item = field.strip().partition('=')
            key, item = key.strip(), item.strip()
            if key in fields:
                raise ValueError("llm_reasoning must not repeat a field")
            if key == 'exclude' and not separator:
                fields[key] = True
            elif key == 'effort' and separator:
                fields[key] = item
            elif key == 'max_tokens' and separator:
                if re.fullmatch(r"[0-9]+", item) is None:
                    raise ValueError(REASONING_ERROR)
                fields[key] = int(item)
            elif key in ('thinking', 'display') and separator:
                fields[key] = item
            else:
                raise ValueError(REASONING_ERROR)
    elif isinstance(value, dict):
        fields = dict(value)
    else:
        raise ValueError(REASONING_ERROR)  # noqa: TRY004 - Preserve the public ValueError validation contract.
    if set(fields) - {'effort', 'max_tokens', 'exclude', 'thinking', 'display'}:
        raise ValueError(REASONING_ERROR)
    if ('effort' in fields) == ('max_tokens' in fields):
        raise ValueError("llm_reasoning requires exactly one of effort and max_tokens")
    if 'exclude' in fields and fields['exclude'] is not True:
        raise ValueError(REASONING_ERROR)
    if 'effort' in fields and fields['effort'] not in REASONING_EFFORTS:
        raise ValueError(REASONING_ERROR)
    if 'max_tokens' in fields and (isinstance(fields['max_tokens'], bool)
                                   or type(fields['max_tokens']) is not int
                                   or fields['max_tokens'] <= 0):
        raise ValueError(REASONING_ERROR)
    # `thinking=adaptive` (Anthropic's `thinking: {"type": "adaptive"}`) rides with an
    # effort level only; which provider may send it is the transport's and the generator's rule.
    if 'thinking' in fields and (fields['thinking'] != 'adaptive' or 'effort' not in fields):
        raise ValueError(REASONING_ERROR)
    if 'display' in fields and (fields['display'] != 'summarized' or 'thinking' not in fields):
        raise ValueError(REASONING_ERROR)
    spec = {'effort': fields['effort']} if 'effort' in fields else {'max_tokens': fields['max_tokens']}
    if fields.get('exclude'):
        spec['exclude'] = True
    if 'thinking' in fields:
        spec['thinking'] = fields['thinking']
    if 'display' in fields:
        spec['display'] = fields['display']
    return spec


def parse_llm_max_tokens(value):
    """An explicitly stated reply budget, 1..32768; the same range the runners parse."""
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(MAX_TOKENS_ERROR)
        value = int(value.strip())
    if isinstance(value, bool) or type(value) is not int or not 1 <= value <= MAX_TOKENS_LIMIT:
        raise ValueError(MAX_TOKENS_ERROR)
    return value


def parse_llm_response_format(value):
    """The stated response format, `json_object` only."""
    if value not in RESPONSE_FORMATS:
        raise ValueError("llm_response_format must be one of " + "|".join(RESPONSE_FORMATS))
    return value


def _validation_messages(value):
    """The declared validation-message version; absent means v1, which leaves no key."""
    if value is None:
        return DEFAULT_VALIDATION_MESSAGES
    if value not in VALIDATION_MESSAGE_VERSIONS:
        raise ValueError('Validation messages must be one of ' + ', '.join(VALIDATION_MESSAGE_VERSIONS))
    return value


DECODING_PARSERS = (parse_llm_reasoning, parse_llm_max_tokens, parse_llm_response_format)


def _decoding(settings):
    """Validate the optional decoding settings in place, without rewriting them."""
    for key, parse in zip(DECODING_KEYS, DECODING_PARSERS):
        if key in settings:
            if isinstance(settings[key], str) and key != 'llm_response_format':
                raise ValueError(f"Registered {key} must be a registered value, not a spec string")
            parse(settings[key])


#: The per-cell paid-call guard a run states (`--max-paid-calls`). Bound exactly
#: when stated, so an option-free run and every existing registration keep their shape.
MAX_PAID_CALLS_KEY = 'max_paid_calls'


def _max_paid_calls(value):
    """A paid-call cap is an integer of at least 1; a Boolean or a float is never one."""
    if isinstance(value, bool) or type(value) is not int or value < 1:
        raise ValueError("max_paid_calls must be an integer of at least 1")
    return value


def _valid_context(profile, proposer, cell):
    if profile not in PROFILES:
        raise ValueError("Unknown run profile")
    if proposer not in PROVIDERS:
        raise ValueError("Unknown run proposer")
    if cell is not None and (not isinstance(cell, str) or re.fullmatch(r"[01][01][012]", cell) is None):
        raise ValueError("Run cell must match [01][01][012]")


def _equal(left, right):
    # A Boolean never stands for a numeric registered setting.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _registered_settings(manifest, bundle, cell):
    by_bundle = manifest.get('runtime_settings')
    if not isinstance(by_bundle, dict) or not isinstance(by_bundle.get(bundle), dict):
        raise ValueError("Manifest must explicitly register the runtime settings")  # noqa: TRY004 - Preserve the public ValueError validation contract.
    settings = dict(by_bundle[bundle])
    # Three-level manifests bind the executed acquisition to the row's design
    # digit. Existing two-level registrations retain their scalar declaration.
    design = settings.get('design_policy')
    if isinstance(design, dict):
        if bundle not in ('core', 'chembench', 'boxinggym', 'forcebench') or set(design) != {'0', '1', '2'}:
            raise ValueError("Invalid registered design policy levels")
        expected = {'0': 'joint_eig', '1': 'joint_eig', '2': 'epig'}
        if bundle in ('boxinggym', 'forcebench'):
            expected = {'0': None, '1': None, '2': 'epig'}
        if design != expected:
            raise ValueError("Registered design policies disagree with cell levels")
        policy = design[cell[2]]
        if policy is None:
            settings.pop('design_policy')
        else:
            settings['design_policy'] = policy
    if bundle in ('core', 'chembench', 'boxinggym', 'forcebench'):
        policy = settings.get('design_policy')
        if (cell[2] == '2') != (policy == 'epig'):
            raise ValueError("Registered design policy disagrees with cell design level")
    profile = settings.pop('profile', None)
    if profile not in PROFILES:
        raise ValueError("Manifest must explicitly register the profile")
    if profile == 'paper_neuron' and bundle != 'neuron':
        raise ValueError("The paper_neuron profile belongs to the neuron bundle alone")
    required = {'particles', 'chains', 'd0_policy'}
    required.add('t_extra' if bundle == 'forcebench' else 'rounds')
    if bundle == 'boxinggym':
        required.add('q_policy')
    if bundle == 'neuron':
        required.add('forward_backend')
    if bundle in PROMPT_BUNDLES:
        required |= set(PROMPT_KEYS)
    if profile == 'paper_neuron':
        required |= set(PAPER_NEURON_BINDINGS) | {'active_maximum'}
    if bundle == 'forcebench':
        required |= {'d0_size', 'd0_points', 'checkpoints', 'checkpoint_values', 'integrator', 'dt',
                     'design_gate', 'noise_inflate', 'design_variant', 'proposal_policy',
                     'proposal_events', 'candidates_per_event', 'structure_proposals'}
    if not required <= set(settings):
        raise ValueError("Incomplete registered runtime settings")
    if bundle == 'core' and settings['d0_policy'] not in ('runner_default', 'fixed4', 'stratified4'):
        raise ValueError("Registered CORE d0_policy must be fixed4 or stratified4")
    for key in ('particles', 'chains'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError("Invalid registered inference setting")
    if profile == 'paper' and (settings['particles'], settings['chains']) != (500, 2):
        raise ValueError("Paper profile requires 500 particles and two chains")
    if profile == 'paper_neuron':
        if (settings['particles'], settings['chains']) != (200, 2):
            raise ValueError("The paper_neuron profile requires 200 particles and two chains")
        for key, value in PAPER_NEURON_BINDINGS.items():
            if not _equal(settings[key], value):
                raise ValueError(f"The paper_neuron profile binds {key}")
        # The pool cap is the runner's to choose; the registration only has to state the
        # one the run executes, so a later change of that bound is a new registration.
        if type(settings['active_maximum']) is not int or settings['active_maximum'] <= 0:
            raise ValueError("The paper_neuron profile requires a positive active maximum")
    if bundle in PROMPT_BUNDLES:
        # A version with no digest names a file nobody can identify, and a digest
        # with no version names bytes nobody can ask for, so both are checked here and the
        # pair is compared with the runner's declaration by `verify_manifest`.
        if not isinstance(settings['prompt_version'], str) or not settings['prompt_version']:
            raise ValueError("Registered prompt version must be a non-empty string")
        _sha(settings['prompt_sha256'])
        # A stated grammar must be one this code knows. An
        # unstated one is v2 and is checked as such by `verify_manifest`.
        _grammar_version(settings.get(GRAMMAR_VERSION_KEY))
        # A registered validation-message version is v2 (the default v1 leaves no key
        # on the runner's side, so a registration stating v1 could never be matched).
        if VALIDATION_MESSAGES_KEY in settings and settings[VALIDATION_MESSAGES_KEY] != 'v2':
            raise ValueError("Registered validation_messages is v2; the default v1 leaves no key")
        # A registration that binds a composed request must
        # bind both halves of what identifies it - how many candidates the
        # request asks for, and the digest of the text that asks.
        if 'composed_prompt_sha256' in settings or settings.get('proposal_limit', 1) != 1:
            limit = settings.get('proposal_limit')
            if type(limit) is not int or not 1 < limit <= 3:
                raise ValueError('Registered proposal limit must be 1..3')
            _sha(settings.get('composed_prompt_sha256'))
        # A registered burst carries both halves; a registered cap is an int.
        if {'initial_proposals', 'parallel_proposals', 'burst_prompt_sha256'} & set(settings):
            size, parallel = settings.get('initial_proposals'), settings.get('parallel_proposals')
            if size is not None and (type(size) is not int or not 1 <= size <= 24):
                raise ValueError('Registered initial proposals must be 1..24')
            if parallel is not None and (type(parallel) is not int or not 1 <= parallel <= 6):
                raise ValueError('Registered parallel proposals must be 1..6')
            if size is None and parallel is None:
                raise ValueError('A registered burst digest needs initial or parallel proposals')
            _sha(settings.get('burst_prompt_sha256'))
        pool_keys = {'initial_pool_source', 'initial_pool_sha256', 'initial_pool_count'}
        # A ChemBench generated pool (CBIP) binds initial_pool_source alone; every other use
        # of the pool keys is a CORE frozen-file binding (IPF2) and keeps IPF2's checks.
        chembench_generated = bundle == 'chembench' and pool_keys & set(settings) == {'initial_pool_source'}
        # A ChemBench frozen pool is IPF2's complete binding with the burst source.
        chembench_file = bundle == 'chembench' and not chembench_generated and pool_keys & set(settings)
        if chembench_file:
            if (not pool_keys <= set(settings) or settings['initial_pool_source'] != 'file_burst'
                    or type(settings['initial_pool_count']) is not int or settings['initial_pool_count'] < 1
                    or settings.get('initial_catalogue') != 'off' or 'initial_proposals' in settings):
                # The IPF2 wording: a ChemBench frozen pool is the same complete file binding.
                raise ValueError('Registered initial pool requires a complete file binding and catalogue off')
            _sha(settings['initial_pool_sha256'])
        elif pool_keys & set(settings) and not chembench_generated:
            if (bundle != 'core' or not pool_keys <= set(settings)
                    or settings['initial_pool_source'] not in ('file_burst', 'file_candidates')
                    or type(settings['initial_pool_count']) is not int or settings['initial_pool_count'] < 1
                    or settings.get('initial_catalogue') != 'off' or 'initial_proposals' in settings):
                raise ValueError('Registered initial pool requires a complete file binding and catalogue off')
            _sha(settings['initial_pool_sha256'])
        if bundle in ('boxinggym', 'core', 'chembench') and 'initial_catalogue' in settings and (
                settings['initial_catalogue'] != 'off' or
                not ('initial_proposals' in settings or pool_keys <= set(settings))):
            raise ValueError('Registered initial_catalogue is off, with a burst')
        if bundle == 'chembench' and 'initial_pool_source' in settings and not chembench_file and (
                settings['initial_pool_source'] != 'generated' or 'initial_proposals' not in settings):
            raise ValueError('Registered ChemBench initial_pool_source is generated, with a burst')
        if bundle in ('boxinggym', 'core') and 'active_maximum' in settings:
            maximum = settings['active_maximum']
            if type(maximum) is not int or not 1 <= maximum <= 32:
                raise ValueError('Registered BoxingGym/CORE active maximum must be 1..32')
    if bundle == 'neuron':
        # The executed forward, and for the batch forward the digest of the
        # equivalence report that admitted it, are registered rather than assumed.
        if settings['forward_backend'] not in ('native', 'batch'):
            raise ValueError("Registered neuron forward backend must be native or batch")
        if settings['forward_backend'] == 'batch':
            if 'forward_equivalence_sha' not in settings:
                raise ValueError("A registered batch forward requires its equivalence report SHA")
            _sha(settings['forward_equivalence_sha'])
        elif 'forward_equivalence_sha' in settings:
            raise ValueError("A registered native forward cannot carry an equivalence report SHA")
    _decoding(settings)
    if MAX_PAID_CALLS_KEY in settings:
        _max_paid_calls(settings[MAX_PAID_CALLS_KEY])
    # Multiple manifest views of the same setting must agree.
    rounds = manifest.get('rounds', {})
    actual_rounds = settings.get('t_extra', settings.get('rounds'))
    if not isinstance(rounds, dict) or not _equal(rounds.get(bundle), actual_rounds):
        raise ValueError("Manifest round declarations disagree")
    by_policy = manifest.get('policies')
    if not isinstance(by_policy, dict):
        raise ValueError("Invalid manifest policies")  # noqa: TRY004 - Preserve the public ValueError validation contract.
    policies = by_policy.get(bundle, {})
    if not isinstance(policies, dict):
        raise ValueError("Invalid manifest policies")  # noqa: TRY004 - Preserve the public ValueError validation contract.
    for key in ('d0_policy', 'q_policy', 'd0_size', 'd0_points'):
        if key in settings and not _equal(policies.get(key), settings[key]):
            raise ValueError("Manifest policy declarations disagree")
    if bundle == 'forcebench':
        numerics = manifest.get('forcebench_numerics', {})
        if not isinstance(numerics, dict):
            raise ValueError("Invalid ForceBench numerics")
        for key in ('integrator', 'design_gate'):
            if numerics.get(key) != settings[key]:
                raise ValueError("Manifest ForceBench numerics disagree")
        inflation = numerics.get('noise_inflate')
        if inflation not in (0, 1, False, True) or bool(inflation) != settings['noise_inflate']:
            raise ValueError("Manifest ForceBench noise declaration disagrees")
        declared_dt = numerics.get('dt')
        if declared_dt != settings['dt'] and not (declared_dt == 'world table (.005)' and settings['dt'] == .005):
            raise ValueError("Manifest ForceBench step size disagrees")
        for key in ('design_gate_fallback', 'fallback_k'):
            if key in settings and not _equal(numerics.get(key), settings[key]):
                raise ValueError('Manifest ForceBench fallback declarations disagree')
        if manifest.get('forcebench_checkpoints') != settings['checkpoints']:
            raise ValueError("Manifest ForceBench checkpoint declarations disagree")
        if manifest.get('forcebench_design_variant') != settings['design_variant']:
            raise ValueError("Manifest ForceBench design variant disagrees")
        proposal = manifest.get('forcebench_proposal_policy', {})
        if not isinstance(proposal, dict):
            raise ValueError("Invalid ForceBench proposal policy")
        if (proposal.get('policy') != settings['proposal_policy'] or
                proposal.get('events') != settings['proposal_events'] or
                proposal.get('candidates_per_event') != settings['candidates_per_event']):
            raise ValueError("Manifest ForceBench proposal declarations disagree")
    if 'fit_timeout_seconds' in settings:
        safeguards = manifest.get('safeguards', {})
        key = 'fit_timeout_neuron_seconds' if bundle == 'neuron' else 'fit_timeout_seconds'
        if not isinstance(safeguards, dict) or not _equal(safeguards.get(key), settings['fit_timeout_seconds']):
            raise ValueError('Manifest fit timeout declarations disagree')
    if bundle != 'forcebench':
        safeguards = manifest.get('safeguards', {})
        declared = ('fit_guards' in settings
                    or isinstance(safeguards, dict) and 'fit_guards' in safeguards)
        if declared and (settings.get('fit_guards') != 'v1' or not isinstance(safeguards, dict)
                         or safeguards.get('fit_guards') != 'v1'):
            raise ValueError('Manifest fit guard declarations disagree')
    if bundle == 'core' and settings['d0_policy'] == 'fixed4':
        settings['d0_policy'] = RECEIPT_D0_POLICY['fixed4']
    return profile, settings


def verify_manifest(manifest_ref, *, profile, proposer, cell, model, bundle, task, seed,
                    base_url_host=None, provider_order=None, settings=None, cells_file=None):
    """Verify exact bytes, cohort, selected row, and the executed settings."""
    _valid_context(profile, proposer, cell)
    if bundle not in ('core', 'boxinggym', 'chembench', 'forcebench', 'neuron'):
        raise ValueError("Unknown registered bundle")
    if not isinstance(task, str) or not task or type(seed) is not int or seed < 0 or cell is None:
        raise ValueError("Manifest verification needs a complete run identity")
    ref = _reference(manifest_ref)
    manifest = _json(_read(ref['path'], ref['sha256']))
    if not isinstance(manifest, dict) or manifest.get('schema') != 'research_wave_manifest_v1':
        raise ValueError("Unsupported label manifest schema")
    self_hash = _sha(manifest.get('manifest_sha256'))
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != self_hash:
        raise ValueError("Manifest self-hash mismatch")
    if manifest.get('run_label') not in RUN_LABEL_CHOICES:
        raise ValueError("Manifest has an invalid run label")
    # The existing schema stores cohort ID as a string, with companion fields.
    cohort = manifest.get('cohort')
    if not isinstance(cohort, str) or not cohort:
        raise ValueError("Manifest needs a cohort identity")
    if (manifest.get('provider'), manifest.get('model'), manifest.get('base_url_host')) != (proposer, model, base_url_host):
        raise ValueError("Manifest cohort provider/model/host mismatch")
    if proposer == 'mock':
        if model is not None or base_url_host is not None:
            raise ValueError("Mock registration cannot claim an LLM identity")
    elif not isinstance(model, str) or not model:
        raise ValueError("Manifest requires a declared LLM model")
    if proposer != 'openai_compat' and base_url_host is not None:
        raise ValueError("Native provider cannot have a compatibility host")
    if not _equal(manifest.get('provider_order'), provider_order):
        raise ValueError("Manifest provider order mismatch")
    registered_profile, registered_settings = _registered_settings(manifest, bundle, cell)
    actual_settings = dict(settings) if isinstance(settings, dict) else None
    if (actual_settings is not None and 'profile' in actual_settings
            and actual_settings.pop('profile') != profile):
        raise ValueError("Runtime profile declarations disagree")
    # The expression grammar is compared on its own, before
    # the exact comparison of everything else, so that an ABSENT key on either
    # side reads as v2 rather than as a shape difference - which is what lets a
    # registration written before the grammar key existed launch unchanged, and what refuses a
    # v3 launch against a v2 registration with a reason that names the grammar.
    if bundle in GRAMMAR_BUNDLES:
        registered_grammar = _grammar_version(registered_settings.pop(GRAMMAR_VERSION_KEY, None))
        actual_grammar = _grammar_version(
            None if actual_settings is None else actual_settings.pop(GRAMMAR_VERSION_KEY, None)
        )
        if registered_grammar != actual_grammar:
            raise ValueError("Manifest grammar version mismatch")
    # Model-identity check: the model identity is compared on its own, before the exact comparison, so a
    # registration made under another identity schema - or before the notation normal form
    # was recorded - is refused with both values named. An absent key is a mismatch.
    if bundle in PROMPT_BUNDLES:
        for key in ('model_id_schema', 'notation_normal_form'):
            registered = registered_settings.get(key)
            runtime = None if actual_settings is None else actual_settings.get(key)
            if registered != runtime:
                raise ValueError(f"Manifest runtime settings mismatch: {key} registered "
                                 f"{registered!r}, runtime {runtime!r}")
    if registered_profile != profile or not _equal(registered_settings, actual_settings):
        raise ValueError("Manifest runtime settings mismatch")
    calls = manifest.get('calls')
    if not isinstance(calls, list) or not calls:
        raise ValueError("Manifest must contain call files")
    matches = []
    directory = Path(ref['path']).parent
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("Invalid manifest call entry")  # noqa: TRY004 - Preserve the public ValueError validation contract.
        if call.get('bundle') != bundle:
            continue
        filename = call.get('file')
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError("Manifest call file must be relative")
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Manifest call file escapes its directory")
        data = _read(path, call.get('sha256'))
        try:
            lines = data.decode('utf-8').splitlines()
        except UnicodeError as exc:
            raise ValueError("Invalid call file encoding") from exc
        for line_number, line in enumerate(lines, 1):
            tokens = line.split()
            if not tokens or tokens[0].startswith('#'):
                continue
            if len(tokens) not in (4, 5, 6) or not tokens[3].isdecimal() or re.fullmatch(r'[01][01][012]', tokens[2]) is None:
                raise ValueError("Invalid registered cell row")
            variant_policies = {'design_point_disagreement': 'point_disagreement',
                                'design_space_filling': 'space_filling'}
            if len(tokens) == 6:
                if (tokens[0] not in ('core', 'boxinggym', 'chembench') or tokens[2][2] != '1'
                        or tokens[5] not in variant_policies):
                    raise ValueError("Invalid registered design policy variant")
                try:
                    declared = float(tokens[4])
                except ValueError as exc:
                    raise ValueError("Invalid registered variant cost") from exc
                if not math.isfinite(declared) or declared < 0:
                    raise ValueError("Invalid registered variant cost")
                if registered_settings.get('design_policy') != variant_policies[tokens[5]]:
                    raise ValueError("Registered row design policy mismatch")
            elif registered_settings.get('design_policy') in variant_policies.values():
                raise ValueError("Registered design variant requires a row token")
            row = (tokens[0], tokens[1], tokens[2], int(tokens[3]))
            if row == (bundle, task, cell, seed):
                if cells_file is not None and Path(cells_file).resolve() != path:
                    continue
                matches.append({'bundle': bundle, 'task': task, 'cell': cell, 'seed': seed,
                                'file': filename, 'line': line_number, 'file_path': str(path),
                                'file_sha256': call['sha256']})
                if len(tokens) == 6:
                    matches[-1]['variant'] = tokens[5]
    if len(matches) != 1:
        raise ValueError("Manifest must contain exactly one matching run row")
    status = {}
    if 'contract_status' in manifest:
        if manifest['contract_status'] != 'registered':
            raise ValueError("Invalid manifest contract_status")
        # This is the verified declaration's status, not a new run-label value
        # or a claim that the driver has granted execution permission.
        status['contract_status'] = manifest['contract_status']
    return {**status, 'manifest_path': ref['path'], 'manifest_sha256': ref['sha256'],
            'manifest_self_hash': self_hash, 'matched_row': matches[0],
            'manifest_run_label': manifest['run_label'], 'cohort': cohort}


def resolve_label(requested, profile, proposer, *, cell=None, manifest_ref=None, model=None,
                  bundle=None, task=None, seed=None, base_url_host=None, provider_order=None,
                  settings=None, cells_file=None):
    """Missing is allowed for development; any supplied invalid reference fails."""
    _valid_context(profile, proposer, cell)
    if requested not in (None, *RUN_LABEL_CHOICES):
        raise ValueError("Unknown run label")
    binding = {'profile': profile, 'proposer': proposer, 'cell': cell, 'model': model,
               'bundle': bundle, 'task': task, 'seed': seed, 'base_url_host': base_url_host,
               'provider_order': provider_order, 'settings': settings}
    provenance = {'manifest_path': None, 'manifest_sha256': None, 'manifest_self_hash': None,
                  'matched_row': None, 'manifest_run_label': None, 'cohort': None}
    if manifest_ref is not None:
        provenance = verify_manifest(manifest_ref, **binding, cells_file=cells_file)
    allowed = (profile in ('paper', 'paper_neuron') and proposer in PROVIDERS[1:]
               and cell is not None and cell[0] == '1'
               and manifest_ref is not None and provenance['manifest_run_label'] == 'production')
    if requested == 'production' and not allowed:
        raise ValueError("--run-label production requires a paper profile, an LLM expansion cell, and a matching verified production manifest")
    return {'run_label': requested or 'development', 'requested_run_label': requested,
            'rule_version': RULE_VERSION, 'policy_source_sha256': policy_source_sha256(),
            **provenance, 'binding': binding}


def policy_source_sha256():
    """This file's own digest, recorded in every receipt it issues."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recheck_manifest(receipt):
    """Detect file/row replacement and receipt tampering before publishing results."""
    # A receipt is rechecked against the policy that issued it and no other.
    # `rule_version` is compared here and `policy_source_sha256` by the whole-receipt
    # comparison below (the recomputed receipt carries THIS file's digest), so a receipt
    # issued under an older rule version and read by a newer file - or the reverse, a
    # newer receipt read by the older copy the cohort ran on - is
    # refused rather than re-judged under rules it never saw.
    if not isinstance(receipt, dict) or receipt.get('rule_version') != RULE_VERSION:
        raise ValueError("Unsupported label receipt")
    ref = None if receipt.get('manifest_path') is None else {
        'path': receipt['manifest_path'], 'sha256': receipt.get('manifest_sha256')}
    try:
        fresh = resolve_label(receipt.get('requested_run_label'), manifest_ref=ref, **receipt.get('binding', {}))
    except TypeError as exc:
        raise ValueError("Malformed label receipt binding") from exc
    if not _equal(fresh, receipt):
        raise ValueError("Label receipt no longer matches its registration")
    return fresh


def bmd_runtime_settings(bundle, args):
    """Registered runtime knobs; resolved before output and without changing args."""
    profile = args.profile
    particles = 64 if profile == 'dev' else 500
    if bundle == 'neuron':
        particles = {'dev': 16, 'paper_neuron': 200}.get(profile, 500)
    override = getattr(args, 'particles', None)
    if bundle in ('chembench', 'neuron') and override is not None:
        particles = override
    settings = {'rounds': args.rounds, 'particles': particles, 'chains': 2}
    if bundle in PROMPT_BUNDLES:
        from bayes_model_discovery.experimental.model_identity import (
            MODEL_ID_SCHEMA,
            NOTATION_NORMAL_FORM,
        )

        settings['model_id_schema'] = MODEL_ID_SCHEMA
        settings['notation_normal_form'] = NOTATION_NORMAL_FORM
    timeout = getattr(args, 'fit_timeout_seconds', None)
    if timeout is not None:
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Fit timeout must be finite and positive')
        settings['fit_timeout_seconds'] = float(timeout)
    fit_guards = getattr(args, 'fit_guards', 'off')
    if fit_guards not in ('off', 'v1'):
        raise ValueError('Fit guards must be off or v1')
    if fit_guards == 'v1':
        settings['fit_guards'] = fit_guards
    if bundle == 'boxinggym':
        settings.update(d0_policy=getattr(args, 'd0_policy', 'fractions4'),
                        q_policy=getattr(args, 'q_policy', 'native'),
                        proposal_policy=getattr(args, 'proposal_policy', 'P1'))
        # The half-count D0 unit is registered exactly when a run states it, like a
        # decoding option; an unstated option leaves no key, so every earlier receipt and
        # registration keeps its shape, and a registration without it refuses the run.
        zero_count_unit = getattr(args, 'd0_zero_count_unit', 'refuse')
        if zero_count_unit not in ('refuse', 'half_count'):
            raise ValueError("d0_zero_count_unit must be refuse or half_count")
        if zero_count_unit != 'refuse':
            settings['d0_zero_count_unit'] = zero_count_unit
        # BoxingGym design ascent: `--design-search ascent`, declared exactly when stated (method and constants
        # digest), so every earlier BoxingGym declaration and receipt keeps its shape.
        if getattr(args, 'design_search', None) == 'ascent':
            from bayes_model_discovery.experimental.boxinggym_design_search import (
                declared_settings as bg_search,
            )

            settings.update(bg_search())
    elif bundle == 'core':
        d0_policy = getattr(args, 'd0_policy', 'fixed4')
        if d0_policy not in ('fixed4', 'stratified4'):
            raise ValueError("CORE d0_policy must be fixed4 or stratified4")
        # Keep the legacy receipt sentinel for fixed4; no old receipt is rewritten.
        settings.update(d0_policy=RECEIPT_D0_POLICY[d0_policy],
                        proposal_policy=getattr(args, 'proposal_policy', 'P1'))
    elif bundle == 'chembench':
        settings.update(d0_policy=getattr(args, 'd0_policy', 'random4'),
                        q=getattr(args, 'q', None) or (32 if profile == 'dev' else 256),
                        d0_fixed_inputs=getattr(args, 'd0_fixed_inputs', None))
        if hasattr(args, 'repair_attempts'):
            settings['repair_attempts'] = validate_repair_attempts(
                args.repair_attempts, bundle='chembench')
        if getattr(args, 'design_search', 'pool') == 'ascent':
            from bayes_model_discovery.experimental.chembench_core import ACQUISITION_STREAM
            from bayes_model_discovery.experimental.design_search import declared_settings

            settings.update(declared_settings(args, ACQUISITION_STREAM))
        elif getattr(args, 'design_search', None) == 'pool':
            settings['candidate_source'] = 'pool'
    elif bundle == 'neuron':
        settings.update(d0_policy='native2')
        # The forward the run actually executes, always declared, and for the
        # batch forward the digest of the equivalence report the runner checked it against.
        backend = getattr(args, 'forward_backend', None) or 'native'
        if backend not in ('native', 'batch'):
            raise ValueError("Forward backend must be native or batch")
        settings['forward_backend'] = backend
        if backend == 'batch':
            settings['forward_equivalence_sha'] = _sha(getattr(args, 'forward_equivalence_sha', None))
        # The neuron reply rule `--llm-accept-code-fence`, registered exactly when stated;
        # an unstated option adds no key, so every earlier declaration and receipt keeps its shape.
        if getattr(args, 'llm_accept_code_fence', False) is True:
            settings['llm_accept_code_fence'] = True
        # The MDA-comparison profile declares its kernel, its feature space and
        # the active-pool bound the runner passes in, so each is compared with a registration.
        if profile == 'paper_neuron':
            settings.update(PAPER_NEURON_BINDINGS)
            maximum = getattr(args, 'active_maximum', None)
            if type(maximum) is not int or maximum <= 0:
                raise ValueError("A paper_neuron run must declare its active maximum")
            settings['active_maximum'] = maximum
    else:
        raise ValueError("Unknown BMD bundle")
    if bundle in PROMPT_BUNDLES:
        # The runner selects its system prompt before resolving the receipt and
        # states the identity of the bytes it read, exactly as the
        # neuron runner states its active maximum and equivalence digest. There is no
        # default here: a run that declares no prompt cannot be registered.
        version = getattr(args, 'prompt_version', None)
        if not isinstance(version, str) or not version:
            raise ValueError("A core, boxinggym or chembench run must declare its prompt version")
        settings['prompt_version'] = version
        settings['prompt_sha256'] = _sha(getattr(args, 'prompt_sha256', None))
        # The expression grammar. Unlike the prompt pair there is a default here, and
        # it is the gate every finished wave was judged by, so the DEFAULT
        # leaves no key: a v2 run's receipt and a v2 registration keep the bytes
        # they have always had, and an absent key reads as v2 on both sides.
        grammar = _grammar_version(getattr(args, GRAMMAR_VERSION_KEY, None))
        if grammar != DEFAULT_GRAMMAR_VERSION:
            settings[GRAMMAR_VERSION_KEY] = grammar
        # The validation-message version, the same way - the default leaves no key.
        messages = _validation_messages(getattr(args, VALIDATION_MESSAGES_KEY, None))
        if messages != DEFAULT_VALIDATION_MESSAGES:
            settings[VALIDATION_MESSAGES_KEY] = messages
        # A run that asks for more than one candidate per
        # round sends a COMPOSED system text (the packaged file's bytes plus one
        # paragraph), so the registration binds that text's digest beside the
        # file's. k = 1 sends the file alone and adds no key, which is what
        # keeps every existing declaration byte-identical.
        composed = getattr(args, 'composed_prompt_sha256', None)
        if composed is not None:
            limit = getattr(args, 'proposal_limit', 1)
            if type(limit) is not int or not 1 < limit <= 3:
                raise ValueError('A composed request declares a proposal limit of 2..3')
            settings['proposal_limit'] = limit
            settings['composed_prompt_sha256'] = _sha(composed)
        # A round-0 burst binds how many candidates it asks for and the
        # digest of the composed burst request; an active-pool cap other than the
        # runner's default is bound as well. Absent, no key is added.
        burst = getattr(args, 'burst_prompt_sha256', None)
        if burst is not None:
            size = getattr(args, 'initial_proposals', None)
            parallel = getattr(args, 'parallel_proposals', None)
            if size is not None and (type(size) is not int or not 1 <= size <= 24):
                raise ValueError('A burst request declares 1..24 initial proposals')
            if parallel is not None and (type(parallel) is not int or not 1 <= parallel <= 6):
                raise ValueError('A parallel proposal round declares 1..6 calls')
            if size is None and parallel is None:
                raise ValueError('A burst digest needs initial or parallel proposals')
            if size is not None:
                settings['initial_proposals'] = size
            if parallel is not None:
                settings['parallel_proposals'] = parallel
            settings['burst_prompt_sha256'] = _sha(burst)
            # A round-0 burst's refill waves send their own composed request, so a
            # burst binds that digest and the refill paragraph's alone (a project decision of
            # 2026-09-23);
            # without initial proposals neither key is added.
            if size is not None:
                for key in BURST_REFILL_KEYS:
                    settings[key] = _sha(getattr(args, key, None))
                # Refill-policy binding: and the refill rule it states, compared exactly with the
                # registration; a burst that states none is refused.
                policy = getattr(args, BURST_REFILL_POLICY_KEY, None)
                if not isinstance(policy, str) or not policy:
                    raise ValueError('A burst with initial proposals declares its refill policy')
                settings[BURST_REFILL_POLICY_KEY] = policy
            if bundle == 'chembench':
                if getattr(args, 'initial_pool_source', None) != 'generated':
                    raise ValueError('A ChemBench burst declares initial_pool_source generated')
                settings['initial_pool_source'] = 'generated'
        if any(getattr(args, key, None) is not None for key in BURST_REFILL_KEYS) and (
                burst is None or getattr(args, 'initial_proposals', None) is None):
            raise ValueError('A burst refill digest needs a burst with initial proposals')
        if getattr(args, BURST_REFILL_POLICY_KEY, None) is not None and (
                burst is None or getattr(args, 'initial_proposals', None) is None):
            raise ValueError('A burst refill policy needs a burst with initial proposals')
        # A CORE run seeded from a frozen pool file binds that file; so
        # does a ChemBench cell of the frozen-pool track, whose file is always a burst.
        pool = getattr(args, 'initial_pool_sha256', None) if bundle in ('core', 'chembench') else None
        if pool is not None:
            source, count = (getattr(args, 'initial_pool_source', None),
                             getattr(args, 'initial_pool_count', None))
            sources = ('file_burst',) if bundle == 'chembench' else ('file_burst', 'file_candidates')
            if (source not in sources or type(count) is not int or count < 1
                    or getattr(args, 'initial_catalogue', 'on') != 'off'
                    or getattr(args, 'initial_proposals', None) is not None):
                raise ValueError('Initial pool requires a validated file binding and catalogue off')
            settings.update(initial_pool_source=source, initial_pool_sha256=_sha(pool),
                            initial_pool_count=count)
        # A run that seeds no catalogue (the burst alone at round 0, or a frozen
        # pool file under IPF2) states so; 'on' is the default and adds no key.
        if bundle in ('boxinggym', 'core', 'chembench') and getattr(args, 'initial_catalogue', 'on') == 'off':
            if burst is None and pool is None:
                raise ValueError('initial_catalogue off needs a burst')
            settings['initial_catalogue'] = 'off'
        if bundle in ('boxinggym', 'core'):
            maximum = getattr(args, 'active_maximum', None)
            default_maximum = CORE_ACTIVE_MAXIMUM if bundle == 'core' else BOXINGGYM_ACTIVE_MAXIMUM
            if maximum is not None and maximum != default_maximum:
                if type(maximum) is not int or not 1 <= maximum <= 32:
                    raise ValueError('A BoxingGym/CORE active maximum is 1..32')
                settings['active_maximum'] = maximum
    design_policy = getattr(args, 'design_policy', None)
    if design_policy is None:
        cell = getattr(args, 'cell', None)
        design_policy = 'epig' if isinstance(cell, str) and cell.endswith('2') else 'joint_eig'
    if bundle in ('core', 'chembench') or (bundle == 'boxinggym' and design_policy in ('epig', 'point_disagreement', 'space_filling')):
        settings['design_policy'] = design_policy
    # Scientific switches not represented by the row must also be registered.
    defaults = {
        'core': {'initial_family': 'binomial', 'proposal_rounds': '', 'proposal_limit': 1,
                 'repair_attempts': 1, 'archive_reevaluation_rounds': '', 'replicate': 0,
                 'prior_template': 'default', 'priors': 'off'},
        'boxinggym': {'initial_family': None, 'truth_in_pool': False, 'inject_oracle_at': None,
                      'prompt_variant': None, 'prior_template': 'default', 'priors': 'off'},
        'chembench': {'family': 'lognormal', 'noise': 0.01, 'prior_template': 'default'},
    }
    if bundle in defaults:
        settings['scientific_options'] = {key: getattr(args, key, default)
                                           for key, default in defaults[bundle].items()}
    # A decoding option is registered exactly when the run states one. An
    # unstated option leaves no key, so every earlier declaration keeps its exact shape.
    for key, parse in zip(DECODING_KEYS, DECODING_PARSERS):
        if hasattr(args, key):
            settings[key] = parse(getattr(args, key))
    # The per-cell paid-call guard is registered exactly when the run states one;
    # the exact comparison in `verify_manifest` then binds the registered value.
    if hasattr(args, MAX_PAID_CALLS_KEY):
        settings[MAX_PAID_CALLS_KEY] = _max_paid_calls(getattr(args, MAX_PAID_CALLS_KEY))
    return settings


# The registered stage-1 track identifier (a project decision), declared by the CORE
# prompt-ladder registration. The code freeze that registration pins carries this
# constant; only this track's verified A8 launch contract may run two repairs.
# Examples and other tracks are not authority.
REPAIR_TWO_ALLOWED_TRACKS: tuple[str, ...] = ("prompt_ladder_stage1_20260921",)
REPAIR_BOUND = ('repair_attempts must be 0 or 1; 2 requires a verified CORE A8 '
                'launch contract; 3 or more is forbidden')


class RepairAuthorizationError(ValueError):
    """A repair setting lacks the required launch authority."""


def validate_repair_attempts(value, *, bundle=None, receipt=None, context=None, burst=False,
                             request_identity=None):
    """The P1 default remains one repair; only verified CORE A8 may use two."""
    if type(value) is not int:
        raise RepairAuthorizationError(REPAIR_BOUND)
    if value in (0, 1):
        return value
    if value != 2 or bundle != 'core' or burst or not isinstance(receipt, dict):
        raise RepairAuthorizationError(REPAIR_BOUND)
    try:
        fresh = recheck_manifest(receipt)
    except ValueError as exc:
        raise RepairAuthorizationError(f'{REPAIR_BOUND}: {exc}') from exc
    binding = fresh['binding']
    if (binding.get('bundle') != 'core' or fresh.get('manifest_path') is None
            or fresh.get('contract_status') != 'registered'):
        raise RepairAuthorizationError(REPAIR_BOUND)
    manifest = _json(_read(fresh['manifest_path'], fresh['manifest_sha256']))
    # Reuse the registered driver's schema, not a second authorization channel.
    from bayes_model_discovery.experimental.launch_contract import (
        RGT_ADDED_KEYS,
        validate_launch_contract,
        validate_launch_prerequisites,
    )
    try:
        validate_launch_contract(
            {key: manifest[key] for key in RGT_ADDED_KEYS[1:] if key in manifest},
            required=True)
        validate_launch_prerequisites(manifest)
    except ValueError as exc:
        raise RepairAuthorizationError(f'{REPAIR_BOUND}: {exc}') from exc
    track = manifest.get('core_track')
    settings = binding.get('settings') or {}
    scientific = settings.get('scientific_options') or {}
    if (not isinstance(track, dict) or track.get('arm') != 'A8'
            or track.get('track_id') not in REPAIR_TWO_ALLOWED_TRACKS
            or scientific.get('repair_attempts') != 2
            or settings.get('proposal_policy') != 'P1'):
        raise RepairAuthorizationError(REPAIR_BOUND)
    if context is not None and (
            context.get('cell') != binding.get('cell')
            or context.get('world_seed') != binding.get('seed')
            or context.get('proposal_rep') != scientific.get('replicate', 0)
            or type(context.get('round')) is not int
            or not 1 <= context['round'] <= settings.get('rounds', 0)):
        raise RepairAuthorizationError('A8 repair authority does not match the proposal context')
    if request_identity is not None and (
            request_identity.get('proposer') != binding.get('proposer')
            or request_identity.get('model') != binding.get('model')
            or request_identity.get('initial_family') != scientific.get('initial_family')
            or (context is not None and 'component' in context)):
        raise RepairAuthorizationError('A8 repair authority does not match the proposer identity')
    return value


def bmd_label_receipt(bundle, args, *, task, metadata):
    """Resolve a runner's actual namespace before creating outputs or clients."""
    proposer = getattr(args, 'requested_proposer', getattr(args, 'proposer', None))
    model = getattr(args, 'llm_model', None)
    if proposer == 'mock':
        model = None
    elif model is None:
        model = 'gpt-4o' if proposer == 'openai' else 'claude-opus-4-7'
    return resolve_label(getattr(args, 'run_label', None), args.profile, proposer,
                         cell=args.cell, manifest_ref=getattr(args, 'manifest_ref', None),
                         model=model, bundle=bundle, task=task, seed=args.seed,
                         base_url_host=metadata.get('llm_base_url_host'),
                         provider_order=metadata.get('provider_order'),
                         settings=bmd_runtime_settings(bundle, args))
