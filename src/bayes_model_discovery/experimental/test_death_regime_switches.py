"""death's regime-switch tests: `--initial-family`, `--truth-in-pool`,
`--inject-oracle-at` and the development-only `--prompt-variant`. The
investigation these implement is recorded in the project's development notes.

Nine scenario groups:

-   default switches are byte-inert (CLI defaults, points_variant blank).
-   --initial-family: menu validity (belief_packet.interface_families, which
    already had this parameter) and the "true structure in the pool, wrong
    initial family" regime actually running end to end.
-   the oracle expression validates and is finite under the prior probe, for
    all three Table 2 bounded-count families, with a model_id distinct from
    every initial structure and every mock template of proposal policy P1.
-   the prior probe's draw stream is seeded from the model identity alone, so
    its overflow/underflow headroom is a FIXED fact, not a statistical one.
-   disclosure only: the `positive` role prior's coverage of theta_true's own
    support (the role prior itself is unchanged by this change).
-   --truth-in-pool ADDS a fifth structure (never replaces), consistently
    across config/belief-packet/round-0 fits.
-   --inject-oracle-at k: normal evidence-ranked admission, the k=10 token,
    the "already active" ignored-and-recorded case, cell 000, bounds
    rejection in main(), and the death-only gate.
-   variant tokens (truth_in_pool, oracle_injected_k5/_k10) sort alphabetically
    alongside the pre-existing prior/composite tokens.
-   --prompt-variant: default byte-identical to the v2.0 prompt, loads a file and records
    its declared tag, rejects a file with no declared tag, is refused whenever
    the run's label would be production, and reaches config.json on a real run.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logit
from scipy.stats import lognorm, truncnorm


def adapter():
    """Import the BoxingGym core adapter, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def bg_run_args(output, **overrides):
    """The full argparse namespace boxinggym_core.run expects, with default option values."""
    bg = adapter()
    return SimpleNamespace(
        **{
            "output": str(output),
            "task": "death",
            "profile": "dev",
            "proposer": "mock",
            "llm_model": None,
            "allow_paid_api": False,
            "rounds": 1,
            "seed": 11,
            "cell": "111",
            "prior_template": bg.DEFAULT_PRIOR_TEMPLATE,
            "priors": "off",
            # The three death-regime switches plus the development-only
            # prompt variant. Defaults reproduce every earlier call site
            # exactly.
            "initial_family": None,
            "truth_in_pool": False,
            "inject_oracle_at": None,
            "prompt_variant": None,
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def restore_prompt_variant():
    """`count_proposer`'s prompt-variant override is process-global state,
    exactly like `count_models`'s prior-template/llm-priors switches
    (test_prior_switches_boxinggym.py's own `restore_switches`) - cleared
    after every test in this file regardless of what the test itself did."""
    from .count_proposer import set_prompt_variant

    yield
    set_prompt_variant(None)


# --------------------------------------------------------------------------
# Default switches are byte-inert
# --------------------------------------------------------------------------


def test_points_variant_with_no_new_keys_present_is_unaffected():
    """A config dict with NEITHER new key at all - the exact shape every
    earlier caller (including this file's own sibling
    test_prior_switches_boxinggym.py) still builds - must not KeyError and
    must contribute no new token: `.get`, not `config[...]`."""
    bg = adapter()
    death = bg.TASKS["death"]
    assert bg.points_variant({"prior_template": "default", "priors": "off"}, death) == ""
    assert (
        bg.points_variant(
            {
                "prior_template": "default",
                "priors": "off",
                "truth_in_pool": False,
                "inject_oracle_at": None,
            },
            death,
        )
        == ""
    )


def test_default_cli_args_are_none_or_false(monkeypatch):
    """`main()` with none of the four new flags gives argparse's own defaults,
    which is what makes every new code path in run() a no-op."""
    bg = adapter()
    captured = []
    monkeypatch.setattr(bg, "run", captured.append)
    monkeypatch.setattr(
        sys, "argv", ["boxinggym_core", "--output", "/dev/null/never", "--task", "death"]
    )
    bg.main()
    (args,) = captured
    assert args.initial_family is None
    assert args.truth_in_pool is False
    assert args.inject_oracle_at is None
    assert args.prompt_variant is None


# --------------------------------------------------------------------------
# --initial-family: menu validity + the wrong-family regime
# --------------------------------------------------------------------------


def test_initial_family_menu_validity_and_wrong_family_regime_runs(tmp_path):
    """Check that initial family menu validity and wrong family regime runs."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    from .belief_packet import interface_families

    outcome_type = spec.interface()["outcome_type"]
    assert interface_families(outcome_type, False) == ("binomial", ["binomial"])
    assert interface_families(outcome_type, False, "beta_binomial") == (
        "beta_binomial",
        ["beta_binomial"],
    )
    assert interface_families(outcome_type, True, "zi_binomial") == (
        "zi_binomial",
        ["binomial", "beta_binomial", "zi_binomial"],
    )
    with pytest.raises(ValueError, match="is not runnable for"):
        interface_families(outcome_type, False, "poisson")

    out = tmp_path / "wrong_family"
    bg.run(bg_run_args(out, cell="010", rounds=1, initial_family="beta_binomial"))
    config = json.loads((out / "config.json").read_text())
    assert config["initial_family"] == "beta_binomial"
    fits_families = {
        json.loads(p.read_text())["family"] for p in (out / "fits").glob("t00_*.json")
    }
    # the true structure is in this pool (death's initial structures are
    # unchanged), but every one of them was fit under the WRONG family - never
    # binomial, the participant environment's own family.
    assert fits_families == {"beta_binomial"}


def test_initial_family_out_of_menu_is_refused_before_any_output_exists(tmp_path):
    """Check that initial family out of menu is refused before any output exists."""
    bg = adapter()
    out = tmp_path / "never"
    with pytest.raises(ValueError, match="is not runnable for"):
        bg.run(bg_run_args(out, initial_family="poisson"))
    assert not out.exists()


# --------------------------------------------------------------------------
# The oracle expression: grammar, numeric identity, finiteness, distinctness
# --------------------------------------------------------------------------


def test_oracle_expression_validates_matches_the_true_logit_and_is_distinct_from_the_pool():
    """Check that oracle expression validates matches the true logit and is distinct from the pool."""
    bg = adapter()
    from .count_models import Hypothesis, validate_expression

    spec = bg.TASKS["death"]
    interface = spec.interface()
    namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(
        interface["derived_inputs"]
    )
    validate_expression(bg.DEATH_ORACLE_EXPRESSION, bg.DEATH_ORACLE_ROLES, namespace)

    t = np.linspace(0.001, 2.0, 2001)
    for c in (0.05, 0.7, 1.9):
        oracle_logit = np.log(np.exp(c * t) - 1)
        true_logit = logit(1 - np.exp(-c * t))
        assert np.max(np.abs(oracle_logit - true_logit)) < 1e-9

    existing_ids = set()
    for form in (*spec.initial_structures(), *spec.mock_templates()):
        for family in ("binomial", "beta_binomial", "zi_binomial"):
            existing_ids.add(
                Hypothesis(form, family, bg.mock_parameters(form), namespace, ()).model_id
            )

    for family in ("binomial", "beta_binomial", "zi_binomial"):
        oracle = Hypothesis(
            bg.DEATH_ORACLE_EXPRESSION, family, bg.DEATH_ORACLE_ROLES, namespace, ()
        )
        assert oracle.model_id not in existing_ids


def test_oracle_prior_probe_passes_for_all_three_bounded_count_families():
    """Check that oracle prior probe passes for all three bounded count families."""
    bg = adapter()
    from .count_models import Hypothesis

    spec = bg.TASKS["death"]
    interface = spec.interface()
    namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(
        interface["derived_inputs"]
    )
    for family in ("binomial", "beta_binomial", "zi_binomial"):
        oracle = Hypothesis(
            bg.DEATH_ORACLE_EXPRESSION, family, bg.DEATH_ORACLE_ROLES, namespace, ()
        )
        result, predictive = bg.prior_probe(oracle, interface, 1.0)
        assert result["passed"] is True
        assert predictive is not None


# --------------------------------------------------------------------------
# Prior probe headroom is deterministic (seeded from model identity alone)
# --------------------------------------------------------------------------


def test_oracle_prior_probe_headroom_is_deterministic_and_far_from_over_or_underflow():
    """Check that oracle prior probe headroom is deterministic and far from over or underflow."""
    bg = adapter()
    from .core_count import PRIOR_PROBE_SAMPLES
    from .count_models import Hypothesis

    spec = bg.TASKS["death"]
    interface = spec.interface()
    namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(
        interface["derived_inputs"]
    )
    low, high = interface["inputs"][0]["range"]  # death: (0.001, 2.0)
    for family in ("binomial", "beta_binomial", "zi_binomial"):
        oracle = Hypothesis(
            bg.DEATH_ORACLE_EXPRESSION, family, bg.DEATH_ORACLE_ROLES, namespace, ()
        )
        draws = bg.prior_particles(oracle, PRIOR_PROBE_SAMPLES)
        c = draws["c"]
        assert np.all(np.isfinite(c)) and np.all(c > 0)
        assert c.max() * high < 40.0  # overflow only past exp(...) ~ 709
        assert c.min() * low > 1e-6  # underflow only below ~1.1e-16


# --------------------------------------------------------------------------
# Prior scale coverage (disclosure only; the role prior is unchanged)
# --------------------------------------------------------------------------


def test_positive_role_prior_covers_theta_trues_support_disclosure_only():
    """Check that positive role prior covers theta trues support disclosure only."""
    from .count_models import PRIOR_TEMPLATES

    family, mu, sigma = PRIOR_TEMPLATES["default"]["positive"]
    assert (family, mu, sigma) == ("LogNormal", 0.0, 1.0)
    c_prior = lognorm(s=sigma, scale=np.exp(mu))
    # death_process.py defaults: mu=1, sigma=1, lower_bound=0, upper_bound=2.
    theta_true = truncnorm((0.0 - 1.0) / 1.0, (2.0 - 1.0) / 1.0, loc=1.0, scale=1.0)
    assert c_prior.ppf(0.5) == pytest.approx(1.0)
    assert theta_true.ppf(0.5) == pytest.approx(1.0)
    coverage = c_prior.cdf(2.0) - c_prior.cdf(0.0)
    assert coverage == pytest.approx(0.7559, abs=1e-3)


# --------------------------------------------------------------------------
# --truth-in-pool ADDS a fifth structure, never replaces
# --------------------------------------------------------------------------


def test_truth_in_pool_adds_a_fifth_structure_consistently_everywhere(tmp_path):
    """Check that truth in pool adds a fifth structure consistently everywhere."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)

    off = tmp_path / "off"
    bg.run(bg_run_args(off, rounds=1))
    config_off = json.loads((off / "config.json").read_text())
    assert config_off["initial_structures"] == list(spec.initial_structures())
    assert len(config_off["initial_structures"]) == 4
    assert config_off["initial_pool"] == 4

    on = tmp_path / "on"
    bg.run(bg_run_args(on, rounds=1, truth_in_pool=True))
    config_on = json.loads((on / "config.json").read_text())
    names = config_on["initial_structures"]
    assert len(names) == 5
    assert names[:4] == list(spec.initial_structures())  # ADDED, not replaced
    assert names[4] == bg.DEATH_ORACLE_EXPRESSION
    assert config_on["initial_pool"] == 5
    assert config_on["active_maximum"] == 12  # 5 << 12: no truncation

    fits = {json.loads(p.read_text())["expression"] for p in (on / "fits").glob("t00_*.json")}
    assert fits == set(names)  # round 0 fits all five successfully

    packet = json.loads((on / "packets" / "t01.json").read_text())
    assert packet["constraints"]["initial_structures"] == names

    with (on / "points.csv").open(newline="") as stream:
        variants = {row["variant"] for row in csv.DictReader(stream)}
    assert variants == {"truth_in_pool"}


# --------------------------------------------------------------------------
# --inject-oracle-at k
# --------------------------------------------------------------------------


def test_inject_oracle_at_k_admits_it_as_a_normal_evidence_ranked_candidate(tmp_path):
    """Check that inject oracle at k admits it as a normal evidence ranked candidate."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    out = tmp_path / "inject5"
    bg.run(bg_run_args(out, cell="111", rounds=5, inject_oracle_at=5))

    proposal = json.loads((out / "proposals" / "t05.json").read_text())
    injected_events = [e for e in proposal["events"] if e.get("injected")]
    assert len(injected_events) == 1
    event = injected_events[0]
    assert event["expression"] == bg.DEATH_ORACLE_EXPRESSION
    assert event["model_id"] == "c480baf83723ddad9b0a"  # binomial, default initial family
    assert event["status"] == "fitted"
    assert proposal["valid_candidate_count"] == 2  # the mock's own candidate + the oracle
    assert event["model_id"] in proposal["after_ids"]  # survived retain (no cap exemption)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["injected_rounds"] == [5]
    assert summary["injection_ignored_rounds"] == []
    # the injected oracle is excluded from proposer-performance accounting.
    assert summary["proposal_accounting"]["valid_candidates"] == 5
    assert summary["proposal_accounting"]["proposal_rounds_executed"] == 5

    with (out / "points.csv").open(newline="") as stream:
        variants = {row["variant"] for row in csv.DictReader(stream)}
    assert variants == {"oracle_injected_k5"}


def test_inject_oracle_at_k10_reaches_round_ten_and_uses_the_k10_token(tmp_path):
    """Check that inject oracle at k10 reaches round ten and uses the k10 token."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    out = tmp_path / "inject10"
    # cell 000: no mock candidate competes for a fit budget every round, so
    # this stays cheap while still exercising k=10 end to end (the cell-000
    # requirement of the injection tests, folded in here rather than a separate 10-round run).
    bg.run(bg_run_args(out, cell="000", rounds=10, inject_oracle_at=10))
    proposal = json.loads((out / "proposals" / "t10.json").read_text())
    injected_events = [e for e in proposal["events"] if e.get("injected")]
    assert len(injected_events) == 1
    assert injected_events[0]["status"] == "fitted"
    assert proposal["valid_candidate_count"] == 1  # only the injected candidate

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["injected_rounds"] == [10]
    with (out / "points.csv").open(newline="") as stream:
        variants = {row["variant"] for row in csv.DictReader(stream)}
    assert variants == {"oracle_injected_k10"}


def test_inject_oracle_at_k_already_active_is_ignored_and_recorded(tmp_path):
    """`--truth-in-pool` seeds the oracle into the pool at round 0, so the
    round-5 injection attempt finds it already active: the rule is to ignore it
    and record it via the EXISTING duplicate rule - no revival,
    no silent no-op."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    out = tmp_path / "inject_ignored"
    # cell "000" (propose_for returns no candidate every round, exactly
    # as the cell-000 case above) - deliberately, not "111"/"110": with
    # family expansion open, --truth-in-pool plus 5 rounds of accumulating
    # beta_binomial/zi_binomial mock candidates hit an unrelated, pre-existing
    # numerical fragility in count_models.particle_logpmf's normalization
    # check ("likelihood normalization failed") for an ordinary mock
    # candidate that has nothing to do with the oracle or the injection
    # mechanism - reproducible with either design arm, out of scope here
    # scope to fix. Cell "000" keeps every one of the five active models binomial
    # (no dispersion parameter, so that branch is never reached at all),
    # which isolates the "already active" bookkeeping this test targets. The
    # injection hook fires regardless of cell (the cell-000 case above already
    # pins that), so this is still a full, faithful test of "ignore and
    # record".
    bg.run(bg_run_args(out, cell="000", rounds=5, truth_in_pool=True, inject_oracle_at=5))

    proposal = json.loads((out / "proposals" / "t05.json").read_text())
    injected_events = [e for e in proposal["events"] if e.get("injected")]
    assert len(injected_events) == 1
    assert injected_events[0]["status"] == "duplicate_not_refitted"

    archive = json.loads((out / "archive.json").read_text())
    duplicates = [
        e
        for e in archive
        if e["outcome"] == "duplicate" and e["model_id"] == "c480baf83723ddad9b0a"
    ]
    assert len(duplicates) == 1
    assert duplicates[0]["proposed_round"] == 5

    summary = json.loads((out / "summary.json").read_text())
    assert summary["injected_rounds"] == []
    assert summary["injection_ignored_rounds"] == [5]
    with (out / "points.csv").open(newline="") as stream:
        variants = {row["variant"] for row in csv.DictReader(stream)}
    # both switches are on: sorted, +-joined, exactly like prior_wide/prior_llm.
    assert variants == {"oracle_injected_k5+truth_in_pool"}


def test_inject_oracle_at_bounds_are_rejected_in_main(monkeypatch, tmp_path):
    """Check that inject oracle at bounds are rejected in main."""
    bg = adapter()
    out = tmp_path / "never"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boxinggym_core", "--output", str(out), "--task", "death",
            "--rounds", "5", "--inject-oracle-at", "6",
        ],
    )
    with pytest.raises(SystemExit):
        bg.main()
    assert not out.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boxinggym_core", "--output", str(out), "--task", "death",
            "--rounds", "5", "--inject-oracle-at", "0",
        ],
    )
    with pytest.raises(SystemExit):
        bg.main()
    assert not out.exists()


def test_truth_in_pool_and_inject_oracle_at_are_refused_off_the_death_task(tmp_path):
    """Check that truth in pool and inject oracle at are refused off the death task."""
    bg = adapter()
    out = tmp_path / "never"
    with pytest.raises(ValueError, match="death oracle only"):
        bg.run(bg_run_args(out, task="irt", truth_in_pool=True))
    assert not out.exists()
    with pytest.raises(ValueError, match="death oracle only"):
        bg.run(bg_run_args(out, task="irt", inject_oracle_at=1))
    assert not out.exists()


# --------------------------------------------------------------------------
# Variant tokens
# --------------------------------------------------------------------------


def test_variant_tokens_sort_alphabetically_with_the_pre_existing_tokens():
    """Check that variant tokens sort alphabetically with the pre existing tokens."""
    bg = adapter()
    death = bg.TASKS["death"]
    base = {"prior_template": "default", "priors": "off"}
    assert bg.points_variant({**base, "truth_in_pool": True}, death) == "truth_in_pool"
    assert bg.points_variant({**base, "inject_oracle_at": 5}, death) == "oracle_injected_k5"
    assert bg.points_variant({**base, "inject_oracle_at": 10}, death) == "oracle_injected_k10"
    assert (
        bg.points_variant(
            {
                "prior_template": "wide",
                "priors": "off",
                "truth_in_pool": True,
                "inject_oracle_at": 5,
            },
            death,
        )
        == "oracle_injected_k5+prior_wide+truth_in_pool"
    )


# --------------------------------------------------------------------------
# --prompt-variant
# --------------------------------------------------------------------------


def test_prompt_variant_default_is_byte_identical_to_v2_0():
    """Check that prompt variant default is byte identical to v2 0."""
    from .count_proposer import prompt_identity, system_prompt_bytes

    assert (
        hashlib.sha256(system_prompt_bytes()).hexdigest()
        == "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    )
    identity = prompt_identity()
    assert identity["prompt_version"] == "v2.0"
    assert (
        identity["prompt_sha256"]
        == "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    )


def test_prompt_variant_loads_the_file_and_records_its_declared_tag(tmp_path):
    """Check that prompt variant loads the file and records its declared tag."""
    from .count_proposer import prompt_identity, set_prompt_variant, system_prompt_bytes

    variant_file = tmp_path / "variant.txt"
    variant_file.write_text("# prompt_version: e24_test_variant_v1\nSay something different.\n")
    previous = set_prompt_variant(str(variant_file))
    assert previous == (None, None)
    assert system_prompt_bytes() == variant_file.read_bytes()
    identity = prompt_identity()
    assert identity["prompt_version"] == "e24_test_variant_v1"
    assert identity["prompt_sha256"] == hashlib.sha256(variant_file.read_bytes()).hexdigest()
    set_prompt_variant(None)  # clearing restores the default exactly
    assert prompt_identity()["prompt_version"] == "v2.0"


def test_prompt_variant_rejects_a_file_without_a_declared_tag(tmp_path):
    """Check that prompt variant rejects a file without a declared tag."""
    from .count_proposer import set_prompt_variant

    bad = tmp_path / "bad.txt"
    bad.write_text("no tag on line 1\n")
    with pytest.raises(ValueError, match="must declare its tag"):
        set_prompt_variant(str(bad))

    empty_tag = tmp_path / "empty_tag.txt"
    empty_tag.write_text("# prompt_version:   \nbody\n")
    with pytest.raises(ValueError, match="declares an empty tag"):
        set_prompt_variant(str(empty_tag))


@pytest.mark.parametrize("proposer", ["mock", "anthropic"])
def test_prompt_variant_paper_without_registration_is_development(tmp_path, monkeypatch, proposer):
    """Check that prompt variant paper without registration is development."""
    from .test_run_label_policy import ReachedValidatedOutput

    bg = adapter()
    variant_file = tmp_path / "variant.txt"
    variant_file.write_text("# prompt_version: e24_test_variant_v1\nSay something different.\n")
    out = tmp_path / "never"
    original_mkdir = Path.mkdir

    def stop_at_output(path, *args, **kwargs):
        """Raise once the validated output directory is about to be created; otherwise delegate to the real mkdir."""
        if path == out:
            raise ReachedValidatedOutput
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", stop_at_output)
    with pytest.raises(ReachedValidatedOutput):
        bg.run(
            bg_run_args(out, profile="paper", proposer=proposer, allow_paid_api=True,
                        prompt_variant=str(variant_file))
        )
    assert not out.exists()


def test_prompt_variant_reaches_config_on_a_real_run_and_default_run_is_unaffected(tmp_path):
    """Check that prompt variant reaches config on a real run and default run is unaffected."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    variant_file = tmp_path / "variant.txt"
    variant_file.write_text("# prompt_version: e24_test_variant_v1\nSay something different.\n")

    out = tmp_path / "run"
    bg.run(bg_run_args(out, rounds=1, prompt_variant=str(variant_file)))
    config = json.loads((out / "config.json").read_text())
    assert config["prompt_version"] == "e24_test_variant_v1"
    assert config["prompt_sha256"] == hashlib.sha256(variant_file.read_bytes()).hexdigest()

    # and a run with no --prompt-variant, right after, is unaffected (no leakage).
    default_out = tmp_path / "default_run"
    bg.run(bg_run_args(default_out, rounds=1))
    default_config = json.loads((default_out / "config.json").read_text())
    assert default_config["prompt_version"] == "v2.0"
    assert (
        default_config["prompt_sha256"]
        == "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    )
