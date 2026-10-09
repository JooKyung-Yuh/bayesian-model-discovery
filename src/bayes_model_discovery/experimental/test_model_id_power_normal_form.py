"""Power-notation normal form of `ast_roles_family_ac_v2`. Synthetic only.

Every expression here is written for this file (CORE-style `z`/`u`, ChemBench's `f0z`,
BoxingGym death's `tz`); no stored run output, canary expression or provider call is used.
Runs are a stubbed CORE fit with injected replies, ChemBench's synthetic burst proposer and
a BoxingGym death cell with an injected client, exactly as the existing burst tests drive them.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from . import count_models
from .belief_packet import _archive_entry
from .count_models import (
    INITIAL_BURST_REFILL_POLICY,
    Hypothesis,
    canonical_model_id,
    model_id_preimage,
    read_initial_pool_file,
)
from .model_identity import (
    LEGACY_MODEL_ID_SCHEMA,
    MODEL_ID_SCHEMA,
    NOTATION_NORMAL_FORM,
    require_model_id_schema,
)
from .positive_inference import ScalarHypothesis
from .real_inference import RealHypothesis
from .run_label_policy import bmd_runtime_settings, verify_manifest

#: The base commit of this change: the unmodified v1 identity is read from this commit with `git show`, never
#: copied; the comparison skips when this commit is absent from the checkout.
BASE_COMMIT = "92be622bc3cc623da18eaf535e0ab20e66131cf4"
REPO_ROOT = Path(__file__).resolve().parents[3]
V1_SCHEMA = "ast_roles_family_ac_v1"
#: The declared normal form, verbatim, as one line.
DECLARED_NORMAL_FORM = "pow_literal_nonneg_int: square(x)|power(x,k)|x**k -> POW(x,int(k)); k literal finite integer-valued >= 0 (2 == 2.0); x*x, non-integer, negative, non-literal, Sub/Div/assoc/distrib untouched"
PAIR = (("b0", "intercept"), ("b1", "signed"))
TRIPLE = (*PAIR, ("b2", "signed"))
ADMITTED = ("active", "retired_by_evidence")


def ids(*expressions, parameters=PAIR, family="binomial"):
    """Return the model ids of the given expressions under one parameter set and family."""
    return [Hypothesis(expression, family, parameters).model_id for expression in expressions]


def sha20(text):
    """Return the first 20 hex characters of the SHA-256 of the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def test_schema_and_declared_normal_form_text():
    """The schema constants and the normal-form text match the declared values."""
    assert MODEL_ID_SCHEMA == "ast_roles_family_ac_v2"
    assert LEGACY_MODEL_ID_SCHEMA == "ast_roles_family_ordered_legacy"
    assert NOTATION_NORMAL_FORM == DECLARED_NORMAL_FORM and NOTATION_NORMAL_FORM.isascii()


# --------------------------------------------------------------------------
# 1. the rule: the five decided forms, then the additions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forms", [
    ("b0+b1*z**3", "b0+b1*power(z,3)"),
    ("b0+b1*z**2", "b0+b1*power(z,2)", "b0+b1*square(z)"),
])
def test_equal_spellings_are_one_model(forms):
    """Equal power spellings collapse to one model id."""
    assert len(set(ids(*forms))) == 1
    assert len({canonical_model_id(form, "binomial", PAIR) for form in forms}) == 1


@pytest.mark.parametrize("left,right", [
    ("b0+b1*z**2.5", "b0+b1*power(z,2)"),
    ("b0+b1*(z*z*z)", "b0+b1*z**3"),
    ("b0+b1*power(2,z)", "b0+b1*power(z,2)"),
])
def test_different_forms_stay_apart(left, right):
    """Genuinely different power forms keep different model ids."""
    first, second = ids(left, right)
    assert first != second


def test_two_and_two_point_zero_are_the_same_integer_exponent():
    """Exponents 2 and 2.0 (and 0 and 0.0) give the same id and preimage."""
    assert len(set(ids("b0+b1*z**2", "b0+b1*z**2.0", "b0+b1*power(z,2.0)", "b0+b1*square(z)"))) == 1
    assert len(set(ids("b0+b1*z**0", "b0+b1*z**0.0", "b0+b1*power(z,0)"))) == 1
    preimage = model_id_preimage("b0+b1*z**2.0", "binomial", PAIR)
    assert '["POW",["input","z"],["exponent",2]]' in preimage and "2.0" not in preimage
    assert preimage == ('["binomial",["Add",["parameter",0,"intercept"],'
                        '["Mult",["parameter",1,"signed"],["POW",["input","z"],["exponent",2]]]]]')


def test_a_negative_exponent_is_a_unary_minus_and_is_not_normalized():
    """Negative exponents are left unnormalized."""
    assert isinstance(ast.parse("z**-2", mode="eval").body.right, ast.UnaryOp)
    negative, = ids("b0+b1*z**-2")
    assert negative not in ids("b0+b1*power(z,2)", "b0+b1*z**2")
    assert "POW" not in model_id_preimage("b0+b1*z**-2", "binomial", PAIR)
    # Neither negative spelling is rewritten, so they are not merged with each other either.
    assert len(set(ids("b0+b1*z**-2", "b0+b1*power(z,-2)"))) == 2


def test_nested_powers_keep_their_structure_without_exponent_arithmetic():
    """Nested powers keep their structure; exponents are not multiplied."""
    nested = ids("b0+b1*power(power(z,2),3)", "b0+b1*(z**2)**3", "b0+b1*square(z)**3",
                 "b0+b1*power(square(z),3)")
    assert len(set(nested)) == 1
    assert nested[0] not in ids("b0+b1*z**6", "b0+b1*power(z,6)")
    assert model_id_preimage("b0+b1*(z**2)**3", "binomial", PAIR).count('"POW"') == 2


def test_the_base_may_be_any_subexpression():
    """The power base may be any subexpression and is still normalized."""
    assert len(set(ids("b0+power(b1*z,2)", "b0+(b1*z)**2", "b0+square(z*b1)"))) == 1
    assert len(set(ids("b0+b1*square(u+z)", "b0+b1*(z+u)**2", "b0+b1*power(u+z,2)"))) == 1


def test_alpha_renaming_and_roles_still_apply_through_pow():
    """Parameter renaming and roles still apply inside power nodes."""
    renamed = (("c0", "intercept"), ("c1", "signed"))
    assert ids("b0+power(b1*z,2)") == ids("c0+power(c1*z,2)", parameters=renamed)
    assert model_id_preimage("b0+power(b1*z,2)", "binomial", PAIR) == model_id_preimage(
        "c0+power(c1*z,2)", "binomial", renamed)
    assert ids("b0+power(b1*z,2)") != ids("b0+power(b1*z,2)", parameters=(("b0", "intercept"),
                                                                        ("b1", "positive")))


def test_pow_nodes_still_take_part_in_local_add_mult_order():
    """Power nodes take part in the local Add/Mult ordering."""
    assert len(set(ids("b1*z**2+b0", "b0+power(z,2)*b1", "square(z)*b1+b0"))) == 1
    assert canonical_model_id("u*z**2", "normal", ()) == canonical_model_id("square(z)*u", "normal", ())


# --------------------------------------------------------------------------
# 2. continuity with v1: the pinned BASE_COMMIT algorithm, run unmodified
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v1_model_id():
    """Load the pinned v1 canonical_model_id from git, or skip unless BMD_BASELINE_COMPARE=1."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("Set BMD_BASELINE_COMPARE=1 to compare with the pinned v1 identity")
    path = "src/bayes_model_discovery/experimental/count_models.py"
    source = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{BASE_COMMIT}:{path}"])
    assert b"POW" not in source and b"model_id_preimage" not in source
    module = types.ModuleType("bayes_model_discovery.experimental._midc2_v1_count_models")
    module.__file__ = str(REPO_ROOT / path)
    module.__package__ = "bayes_model_discovery.experimental"
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - Pinned local baseline source.
    return module.canonical_model_id


SIGNED3 = tuple((name, "signed") for name in "abc")
PRIOR = (("b1", ("Normal", 0.0, 0.4)),)
POWER_FREE = [
    ("b0+b1*z", "normal", PAIR, ()), ("b1*z+b0", "normal", PAIR, ()),
    ("b0+z*b1", "binomial", PAIR, ()), ("b0-b1*z", "normal", PAIR, ()),
    ("b0+b1/z", "normal", PAIR, ()), ("b0+z/b1", "beta_binomial", PAIR, ()),
    ("tanh(b0+b1*z)", "binomial", PAIR, ()), ("tanh(b1*z+b0)", "binomial", PAIR, ()),
    ("b0+b1*tanh(3*z)+b2*u", "zi_binomial", TRIPLE, ()), ("b0+b1*sqrt(u)", "normal", PAIR, ()),
    ("b0+b1*exp(u)-b2*sqrt(1+z*z)", "normal", TRIPLE, ()), ("a*b+b*c", "normal", SIGNED3, ()),
    ("b0+b1*z*z*z", "binomial", PAIR, ()), ("b0+b1*(z*z)", "binomial", PAIR, ()),
    ("b0+b1[student]", "normal", PAIR, ()), ("minimum(u,z)", "normal", (), ()),
    ("-b0+b1*z", "normal", PAIR, ()), ("b0+b1*z", "normal", PAIR, PRIOR),
]
#: Power notation the rule leaves alone must keep its v1 identity too.
UNTOUCHED_POWER = [
    ("b0+b1*z**2.5", "binomial", PAIR, ()), ("b0+b1*power(z,0.5)", "binomial", PAIR, ()),
    ("b0+b1*z**-2", "binomial", PAIR, ()), ("b0+b1*power(z,-1)", "binomial", PAIR, ()),
    ("b0+b1*power(2,z)", "binomial", PAIR, ()), ("b0+z**b1", "normal", PAIR, ()),
    ("u**z", "normal", (), ()), ("power(z,True)", "normal", (), ()),
]
NORMALIZED = [
    ("b0+b1*square(z)", "binomial", PAIR, ()), ("b0+b1*z**2", "binomial", PAIR, ()),
    ("b0+b1*power(z,3)", "binomial", PAIR, ()), ("b0+b1*z**2.0", "binomial", PAIR, ()),
]


@pytest.mark.parametrize("expression,family,parameters,priors", POWER_FREE + UNTOUCHED_POWER)
def test_ids_without_normalized_power_notation_equal_the_pinned_v1_ids(
        v1_model_id, expression, family, parameters, priors):
    """Expressions without normalized power notation keep their v1 ids."""
    assert canonical_model_id(expression, family, parameters, priors) == v1_model_id(
        expression, family, parameters, priors)


@pytest.mark.parametrize("expression,family,parameters,priors", NORMALIZED)
def test_normalized_spellings_move_off_their_v1_ids(v1_model_id, expression, family, parameters, priors):
    """Normalized power spellings no longer have their v1 ids."""
    assert canonical_model_id(expression, family, parameters, priors) != v1_model_id(
        expression, family, parameters, priors)


def test_v1_kept_the_spellings_apart_that_v2_merges(v1_model_id):
    """v1 kept three spellings apart that v2 merges into one id."""
    spellings = ("b0+b1*z**2", "b0+b1*power(z,2)", "b0+b1*square(z)")
    assert len({v1_model_id(form, "binomial", PAIR) for form in spellings}) == 3
    assert len({canonical_model_id(form, "binomial", PAIR) for form in spellings}) == 1


# --------------------------------------------------------------------------
# 3. model_id_preimage: one code path, the hash rule, every grammar class
# --------------------------------------------------------------------------


@pytest.mark.parametrize("make", [
    lambda: Hypothesis("b0+b1*z**3", "binomial", PAIR),
    lambda: Hypothesis("b0+b1*power(z,3)", "binomial", PAIR),
    lambda: ScalarHypothesis("b0+b1*square(z)", "poisson", PAIR),
    lambda: RealHypothesis("b0+b1*z**2", "normal", PAIR),
    lambda: RealHypothesis("b0+b1*power(z,2)", "heteroscedastic_normal", PAIR),
], ids=["count", "count_power", "scalar", "real", "real_scale"])
def test_every_grammar_class_hashes_its_own_preimage(make):
    """Each hypothesis class hashes its own preimage to its model id."""
    hypothesis = make()
    assert sha20(hypothesis.model_id_preimage) == hypothesis.model_id
    assert isinstance(json.loads(hypothesis.model_id_preimage), list)


def test_the_hash_rule_holds_with_proposed_priors():
    """The id is the hash of the preimage also when priors are proposed."""
    assert sha20(model_id_preimage("b0+b1*z**2", "normal", PAIR, PRIOR)) == canonical_model_id(
        "b0+b1*z**2", "normal", PAIR, PRIOR)


def test_conflicting_spellings_share_model_id_and_preimage():
    """Different spellings of one model share model id and preimage but keep their expressions."""
    spellings = [Hypothesis(form, "binomial", PAIR)
                 for form in ("b0+b1*z**3", "b0+b1*power(z,3)", "b0+b1*z**3.0")]
    assert len({(h.model_id, h.model_id_preimage) for h in spellings}) == 1
    assert [h.expression for h in spellings] == ["b0+b1*z**3", "b0+b1*power(z,3)", "b0+b1*z**3.0"]


def test_model_id_and_preimage_read_one_identity_function(monkeypatch):
    """Model id and preimage both read one identity function."""
    monkeypatch.setattr(count_models, "_canonical_identity", lambda *args: '["one","path"]')
    # A name no other test uses, so the id cache cannot answer from an earlier call.
    expression = "b0+b1*midc2_one_path_probe**3"
    assert count_models.model_id_preimage(expression, "normal", PAIR) == '["one","path"]'
    assert count_models.canonical_model_id(expression, "normal", PAIR) == sha20('["one","path"]')


def test_the_packet_archive_view_never_copies_a_preimage():
    """The packet archive entry omits the model id preimage."""
    hypothesis = Hypothesis("b0+b1*z**3", "binomial", PAIR)
    item = {"model_id": hypothesis.model_id, "model_id_preimage": hypothesis.model_id_preimage,
            "expression": hypothesis.expression, "family": "binomial", "proposed_round": 0,
            "outcome": "active", "reason": {"admitted_round": 0, "active_rounds": [0]},
            "source": "initial_burst"}
    entry = _archive_entry(item)
    assert "model_id_preimage" not in entry and entry["source"] == "initial_burst"


# --------------------------------------------------------------------------
# 4. schema boundary: records, the IPF2 reader, registrations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("record", [{"model_id_schema": V1_SCHEMA},
                                    {"model_id_schema": LEGACY_MODEL_ID_SCHEMA}, {}])
def test_v1_and_legacy_records_are_refused_naming_both_schemas(record):
    """Records with a v1 or legacy schema are refused with a message naming both schemas."""
    actual = record.get("model_id_schema", LEGACY_MODEL_ID_SCHEMA)
    with pytest.raises(ValueError) as error:
        require_model_id_schema(record)
    assert str(error.value) == f"model_id_schema mismatch: expected {MODEL_ID_SCHEMA!r}, got {actual!r}"
    assert require_model_id_schema({"model_id_schema": MODEL_ID_SCHEMA}) == MODEL_ID_SCHEMA


def synthetic_burst_pool(schema):
    """Two spellings of one model (fitted, then its duplicate) and one other model."""
    rows = [("b0+b1*z**3", "fitted"), ("b0+b1*power(z,3)", "duplicate_not_refitted"),
            ("b0+b1*tanh(z)", "fitted")]
    events = []
    for form, status in rows:
        hypothesis = Hypothesis(form, "binomial", PAIR)
        events.append({**hypothesis.as_dict(), "model_id_preimage": hypothesis.model_id_preimage,
                       "status": status})
    # The reader refuses a burst pool file that does not name the
    # refill rule it was produced under, so both files name it; the schema is checked first.
    return {"model_id_schema": schema, "notation_normal_form": NOTATION_NORMAL_FORM,
            "initial_catalogue": "off", "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
            "events": events,
            "after_ids": list(dict.fromkeys(event["model_id"] for event in events))}


def test_the_initial_pool_reader_accepts_a_v2_burst_and_refuses_a_v1_one(tmp_path):
    """The initial pool reader accepts a v2 burst file and refuses a v1 one."""
    current = tmp_path / "v2_pool.json"
    current.write_text(json.dumps(synthetic_burst_pool(MODEL_ID_SCHEMA)))
    pool = read_initial_pool_file(current, cell="100", initial_family="binomial")
    assert pool.source == "file_burst"
    assert [h.expression for h in pool.hypotheses] == ["b0+b1*z**3", "b0+b1*tanh(z)"]
    old = tmp_path / "v1_pool.json"
    old.write_text(json.dumps(synthetic_burst_pool(V1_SCHEMA)))
    with pytest.raises(ValueError, match=re.escape(
            f"model_id_schema mismatch: expected {MODEL_ID_SCHEMA!r}, got {V1_SCHEMA!r}")):
        read_initial_pool_file(old, cell="100", initial_family="binomial")


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_a_v1_or_pre_normal_form_registration_is_refused_naming_both_values(tmp_path, bundle):
    """Stale registrations (v1 or no normal form) are refused naming both values."""
    from .test_run_label_policy import registration, runner_args

    runtime = bmd_runtime_settings(bundle, runner_args())
    assert (runtime["model_id_schema"], runtime["notation_normal_form"]) == (
        MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM)
    without_form = {k: v for k, v in runtime.items() if k != "notation_normal_form"}
    stale = [
        # A registration written before this change: v1, and no normal form at all.
        ("v1", {**without_form, "model_id_schema": V1_SCHEMA}, "model_id_schema", V1_SCHEMA, MODEL_ID_SCHEMA),
        ("no_form", without_form, "notation_normal_form", None, NOTATION_NORMAL_FORM),
    ]
    for name, registered, key, before, after in stale:
        (tmp_path / name).mkdir()
        ref, context, _ = registration(tmp_path / name, bundle=bundle, task="synthetic_task",
                                       settings=registered)
        with pytest.raises(ValueError) as error:
            verify_manifest(ref, **{**context, "settings": runtime})
        assert str(error.value) == (
            f"Manifest runtime settings mismatch: {key} registered {before!r}, runtime {after!r}")
    (tmp_path / "current").mkdir()
    ref, context, _ = registration(tmp_path / "current", bundle=bundle, task="synthetic_task")
    verify_manifest(ref, **context)


# --------------------------------------------------------------------------
# 5. the writers: burst records, archives, packets (all eight schema sites)
# --------------------------------------------------------------------------


def assert_spellings_collapse(record, expressions):
    """The first two events are one model written two ways; the third is another model."""
    first, second, third = record["events"]
    assert [event["expression"] for event in record["events"]] == list(expressions)
    assert [event["status"] for event in record["events"]] == ["fitted", "duplicate_not_refitted", "fitted"]
    assert (second["model_id"], second["model_id_preimage"]) == (first["model_id"], first["model_id_preimage"])
    assert all(sha20(event["model_id_preimage"]) == event["model_id"] for event in record["events"])
    assert record["model_id_schema"] == MODEL_ID_SCHEMA
    assert record["notation_normal_form"] == NOTATION_NORMAL_FORM
    assert record["duplicate_count"] == 1
    assert set(record["after_ids"]) == {first["model_id"], third["model_id"]}
    assert all("model_id_preimage" not in row for row in record["raw_response"])


def assert_admitted_entries_carry_preimages(entries, admitted_ids):
    """Check that admitted archive entries carry preimages matching their ids and others carry none."""
    admitted = [entry for entry in entries if entry["outcome"] in ADMITTED]
    assert {entry["model_id"] for entry in admitted} == set(admitted_ids)
    assert all(sha20(entry["model_id_preimage"]) == entry["model_id"] for entry in admitted)
    others = [entry for entry in entries if entry["outcome"] not in ADMITTED]
    assert others and not any("model_id_preimage" in entry for entry in others)


def assert_preimage_only_in_burst_record_and_archive(out):
    """Config, receipts, summary, fits, packets and every transport request stay preimage-free."""
    allowed = {"proposals/t00_burst.json", "archive.json"}
    checked = []
    for path in sorted(out.rglob("*")):
        name = path.relative_to(out).as_posix()
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".txt"):
            checked.append(name)
            assert ("model_id_preimage" in path.read_text()) == (name in allowed), name
    assert {"config.json", "summary.json", "packets/t00_burst.json"} <= set(checked)
    assert any(name.startswith("fits/") for name in checked)


def test_core_burst_writer_config_receipt_and_archive(tmp_path, monkeypatch):
    """Sites core_count.py config and burst record, and run_label_policy settings via the receipt."""
    from . import core_count as core
    from .test_core_parallel_wave import cell_args, injected_replies
    from .test_count_proposer import stub_fits

    stub_fits(monkeypatch)
    forms = ("b0+b1*z**3", "b0+b1*power(z,3)", "b0+b1*tanh(z)")
    state = injected_replies(monkeypatch, forms)
    out = tmp_path / "cell"
    core.run(cell_args(out, proposal_rounds="2", initial_proposals=2, initial_catalogue="off",
                       burst_concurrency=1, run_label="development"))
    assert state["calls"] == 3
    record = json.loads((out / "proposals/t00_burst.json").read_text())
    assert_spellings_collapse(record, forms)
    config = json.loads((out / "config.json").read_text())
    assert (config["model_id_schema"], config["notation_normal_form"]) == (MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM)
    settings = config["label_receipt"]["binding"]["settings"]
    assert (settings["model_id_schema"], settings["notation_normal_form"]) == (MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM)
    archive = json.loads((out / "archive.json").read_text())
    assert_admitted_entries_carry_preimages(archive, record["after_ids"])
    assert_preimage_only_in_burst_record_and_archive(out)
    # The written pool reads back through the IPF2 reader as the same two models.
    pool = read_initial_pool_file(out / "proposals/t00_burst.json", cell="100", initial_family="binomial")
    assert {h.model_id for h in pool.hypotheses} == set(record["after_ids"])


def test_chembench_duplicate_path_config_burst_and_receipt(tmp_path, monkeypatch):
    """Sites chembench_core.py config, burst record and initial-pool receipt (no archive.json)."""
    from .test_chembench_initial_pool import _wave
    from .test_initial_burst import burst_wave

    forms = {(0, 0): "b0+b1*f0z**3", (0, 1): "b0+b1*power(f0z,3)", (1, 0): "b0+b1*tanh(f0z)"}

    def choose(context, names):
        return Hypothesis(forms[context["burst_wave"], context["burst_call_index"]], "lognormal",
                          PAIR, names)

    out, proposer = _wave(tmp_path, monkeypatch, choose, initial_proposals=2)
    assert len(proposer.calls) == 3
    record = json.loads((out / "proposals/t00_burst.json").read_text())
    assert_spellings_collapse(record, tuple(forms.values()))
    # Each wave entry carries the eight burst-refill keys; wave 0 counts the second
    # spelling as its duplicate, and the refill wave's packet shows the one model wave 0 admitted.
    assert record["waves"] == [burst_wave(out, 0, 2, 1, duplicates=1),
                               burst_wave(out, 1, 1, 1, [record["events"][0]["model_id"]])]
    config = json.loads((out / "config.json").read_text())
    receipt = json.loads((out / "initial_pool_receipt.json").read_text())
    for block in (config, receipt):
        assert (block["model_id_schema"], block["notation_normal_form"]) == (MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM)
    assert "model_id_preimage" not in (out / "initial_pool_receipt.json").read_text()
    assert not (out / "archive.json").exists()
    assert_preimage_only_in_burst_record_and_archive(out)


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_burst_writer_config_and_archive(tmp_path, monkeypatch):
    """Sites boxinggym_core.py config and burst record, and its archive.json."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .count_inference import Fit
    from .test_death_kl import bg_run_args
    from .test_initial_burst import injected_burst_proposer

    def fake_fit(hypothesis, inputs, labels, seed, settings, interface=None, n_trials=50):
        # test_epig_runners' deterministic stand-in for the count SMC; identity is all we read.
        rng = np.random.default_rng(seed % 2**32)
        particles = {name: rng.normal(0.3, 0.4, 8) for name, _ in hypothesis.parameters}
        particles.update({name: rng.beta(2, 8, 8) for name in hypothesis.noise_parameters})
        weight = -(int(hypothesis.model_id[:6], 16) % 500) / 100.0
        return Fit(hypothesis, particles, weight, [weight, weight], {"interface": interface})

    monkeypatch.setattr(bg, "count_fit", fake_fit)
    forms = ("b0+b1*tz**3", "b0+b1*power(tz,3)", "b0+b1*tanh(2*tz)")
    # The burst's three replies, then one for round 1's ordinary proposal call.
    gauge = injected_burst_proposer(monkeypatch, forms[:2], [forms[2], "b0+b1*tz"])
    out = tmp_path / "burst"
    bg.run(bg_run_args(out, cell="100", rounds=1, proposer="anthropic", allow_paid_api=True,
                       llm_model="synthetic-model", initial_proposals=2, burst_concurrency=1,
                       initial_catalogue="off"))
    assert gauge["calls"] == 4
    record = json.loads((out / "proposals/t00_burst.json").read_text())
    assert_spellings_collapse(record, forms)
    config = json.loads((out / "config.json").read_text())
    assert (config["model_id_schema"], config["notation_normal_form"]) == (MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM)
    archive = json.loads((out / "archive.json").read_text())
    admitted = {entry["model_id"] for entry in archive if entry["outcome"] in ADMITTED}
    assert set(record["after_ids"]) <= admitted
    assert_admitted_entries_carry_preimages(archive, admitted)
    assert_preimage_only_in_burst_record_and_archive(out)
