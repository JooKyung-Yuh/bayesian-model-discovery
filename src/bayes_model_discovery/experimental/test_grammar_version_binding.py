"""The `--grammar-version` option and what it binds.

Two claims are pinned here, both about whole runs rather than about the gate in
isolation (the gate itself is `test_grammar_v3`):

- **v2 is the untouched default.** A run that never mentions the option, a run
  whose args namespace predates the option entirely, and a run that states
  `--grammar-version v2` produce byte-identical artifacts, and no artifact of
  any of them carries a `grammar_version` key. That is what lets the finished
  v2 waves keep their grammar and their bytes.
- **v3 binds everywhere or nowhere.** A v3 run records its version in the
  cell's `config.json`, in the label receipt and in the manifest's
  `runtime_settings`, and `verify_manifest` refuses a launch whose version
  disagrees with the registration - a registration with no key meaning v2.

The end-to-end runs use CORE, whose interface is built from code and needs no
benchmark source, with the mock proposer and the deterministic fit stub the
runner's own suite already uses. No paid call, no network, no fixture built
from run text.
"""

import json
from types import SimpleNamespace

import pytest

from . import run_label_policy
from .count_models import (
    DEFAULT_GRAMMAR_VERSION,
    GRAMMAR_VERSIONS,
    active_grammar_version,
    restore_grammar_state,
    set_grammar_version,
)
from .test_count_proposer import run_args, stub_fits

# The two artifacts a run writes that a registration is checked against, plus
# the records a reader would look in for a grammar claim.
RUN_FILES = ("config.json", "summary.json", "points.csv")
VOLATILE_CONFIG_KEYS = ("output", "created_utc", "python", "source_files_sha256")
# Wall-clock sidecars: a second run of the same command differs in these for
# reasons that have nothing to do with the grammar (the same set the runners'
# own byte-identity anchors exclude).
TIMING_SIDECARS = ("fit_timings.jsonl", "acquisition_timings.jsonl",
                   "prediction_pool_summary.jsonl", "summary.json",
                   "round_timings.jsonl")  # ChemBench's per-round wall time


@pytest.fixture(autouse=True)
def _restore_process_grammar():
    """A runner selects the grammar process-wide; no test may leak that choice."""
    previous = set_grammar_version(DEFAULT_GRAMMAR_VERSION)
    yield
    restore_grammar_state(previous)
    assert active_grammar_version() == DEFAULT_GRAMMAR_VERSION


def _run(monkeypatch, output, **overrides):
    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(output, rounds=2, **overrides))
    return output


def _config(output):
    return json.loads((output / "config.json").read_text())


def _comparable(output):
    """Every written artifact, with the two things a second run may change."""
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(output))
        if path.name in TIMING_SIDECARS:
            continue
        if name == "config.json":
            config = _config(output)
            artifacts[name] = {
                key: value for key, value in config.items()
                if key not in VOLATILE_CONFIG_KEYS
            }
            continue
        artifacts[name] = path.read_bytes().replace(str(output).encode(), b"<run>")
    return artifacts


# ---------------------------------------------------------------------------
# v2 is the default and its bytes do not move
# ---------------------------------------------------------------------------

def test_the_option_is_suppressed_and_means_v2_on_every_runner(monkeypatch):
    """An omitted option leaves no namespace attribute, so it adds no record."""
    from . import boxinggym_core, chembench_core, core_count

    assert DEFAULT_GRAMMAR_VERSION == "v2"
    assert GRAMMAR_VERSIONS == ("v2", "v3")
    for module, argv in (
        (core_count, ["--output", "out"]),
        (boxinggym_core, ["--task", "death", "--output", "out"]),
        (chembench_core, ["--manifest", "m.json", "--noise", "0.01", "--output", "out"]),
    ):
        assert not hasattr(_parsed(module, argv), "grammar_version"), module.__name__
        assert _parsed(module, [*argv, "--grammar-version", "v3"]).grammar_version == "v3"
        with pytest.raises(SystemExit):
            _parsed(module, [*argv, "--grammar-version", "v4"])


def test_an_args_namespace_without_the_option_runs_as_v2(tmp_path, monkeypatch):
    """The earlier call shape still means v2, and still records no key."""
    args = run_args(tmp_path / "old", rounds=2)
    assert not hasattr(args, "grammar_version")
    core_count = stub_fits(monkeypatch)
    core_count.run(args)
    assert active_grammar_version() == "v2"
    assert "grammar_version" not in _config(tmp_path / "old")


def test_stating_v2_changes_nothing_but_the_recorded_key(tmp_path, monkeypatch):
    """Like-for-like replay: the same command, the option stated at its default.

    Every artifact is identical except that the explicit namespace records the
    key it stated - the same shape `--design-policy` and `--prompt-version`
    have always had. An omitted option records nothing at all.
    """
    # Same basename in two different parents: `points.csv`'s `run_id` column is
    # the output directory's NAME, so two differently named directories would
    # differ there for a reason that has nothing to do with the grammar.
    silent = _comparable(_run(monkeypatch, tmp_path / "silent" / "run"))
    stated = _comparable(_run(monkeypatch, tmp_path / "stated" / "run", grammar_version="v2"))
    assert sorted(silent) == sorted(stated)
    assert stated["config.json"].pop("grammar_version") == "v2"
    for name in silent:
        assert silent[name] == stated[name], name


def test_a_run_that_omits_the_option_writes_no_grammar_version_anywhere(
    tmp_path, monkeypatch
):
    """Test: a run that omits the option writes no grammar version anywhere."""
    output = _run(monkeypatch, tmp_path / "v2")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            assert b"grammar_version" not in path.read_bytes(), path


def test_a_v2_run_leaves_the_label_receipt_and_runtime_settings_unchanged(tmp_path):
    """Test: a v2 run leaves the label receipt and runtime settings unchanged."""
    settings = run_label_policy.bmd_runtime_settings("core", _label_args())
    assert "grammar_version" not in settings
    stated = run_label_policy.bmd_runtime_settings("core", _label_args(grammar_version="v2"))
    assert stated == settings


def _parsed(module, argv):
    """The namespace a runner's own `main()` builds for this command line."""
    import sys

    captured = {}
    original_run, original_argv = module.run, sys.argv
    module.run = lambda args: captured.setdefault("args", args)
    sys.argv = [module.__name__, *argv]
    try:
        module.main()
    finally:
        module.run, sys.argv = original_run, original_argv
    return captured["args"]


def _label_args(**overrides):
    base = {
        "profile": "dev", "rounds": 5, "cell": "111", "seed": 11, "proposer": "mock",
        "llm_model": None, "run_label": None, "prompt_version": "v2.0",
        "prompt_sha256": "0" * 64, "design_policy": "joint_eig",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# What a v3 run binds, and what refuses a mismatch
# ---------------------------------------------------------------------------

def test_a_v3_run_records_its_version_in_the_cell_config(tmp_path, monkeypatch):
    """Test: a v3 run records its version in the cell config."""
    output = _run(monkeypatch, tmp_path / "v3", grammar_version="v3")
    assert _config(output)["grammar_version"] == "v3"


def test_runtime_settings_carry_a_v3_declaration_and_nothing_for_v2():
    """Test: runtime settings carry a v3 declaration and nothing for v2."""
    for bundle in run_label_policy.GRAMMAR_BUNDLES:
        assert "grammar_version" not in run_label_policy.bmd_runtime_settings(
            bundle, _label_args(**_BUNDLE_EXTRAS.get(bundle, {}))
        )
        settings = run_label_policy.bmd_runtime_settings(
            bundle, _label_args(grammar_version="v3", **_BUNDLE_EXTRAS.get(bundle, {}))
        )
        assert settings["grammar_version"] == "v3"


def test_runtime_settings_refuse_an_unregistered_version():
    """Test: runtime settings refuse an unregistered version."""
    with pytest.raises(ValueError, match="grammar version"):
        run_label_policy.bmd_runtime_settings("core", _label_args(grammar_version="v4"))


def test_the_label_receipt_changes_with_the_grammar_and_not_with_a_restatement():
    """Test: the label receipt changes with the grammar and not with a restatement."""
    plain = run_label_policy.bmd_runtime_settings("core", _label_args())
    restated = run_label_policy.bmd_runtime_settings("core", _label_args(grammar_version="v2"))
    extended = run_label_policy.bmd_runtime_settings("core", _label_args(grammar_version="v3"))
    assert restated == plain
    assert extended != plain


@pytest.mark.parametrize("registered,launched,accepted", (
    (None, None, True),        # a registration made before this change, launched as v2
    (None, "v2", True),        # the same, with the option restated at its default
    ("v3", "v3", True),        # a v3 registration launched at v3
    (None, "v3", False),       # a v3 launch against a registration that means v2
    ("v3", None, False),       # a v2 launch against a v3 registration
    ("v3", "v2", False),
))
def test_verify_manifest_compares_the_grammar_version(
    tmp_path, registered, launched, accepted
):
    """Test: verify manifest compares the grammar version."""
    reference = _manifest_reference(tmp_path, registered)
    settings = run_label_policy.bmd_runtime_settings(
        "core", _label_args(**({"grammar_version": launched} if launched else {}))
    )
    call = {
        "profile": "dev", "proposer": "mock", "cell": "111", "model": None, "bundle": "core",
        "task": "synthetic_task", "seed": 11, "settings": {"profile": "dev", **settings},
    }
    if accepted:
        run_label_policy.verify_manifest(reference, **call)
    else:
        with pytest.raises(ValueError, match="grammar version mismatch"):
            run_label_policy.verify_manifest(reference, **call)


def test_a_registration_that_states_an_unknown_grammar_is_refused(tmp_path):
    """Test: a registration that states an unknown grammar is refused."""
    reference = _manifest_reference(tmp_path, "v9")
    with pytest.raises(ValueError, match="grammar version"):
        run_label_policy.verify_manifest(
            reference, profile="dev", proposer="mock", cell="111", model=None,
            bundle="core", task="synthetic_task", seed=11,
            settings={"profile": "dev",
                      **run_label_policy.bmd_runtime_settings("core", _label_args())},
        )


_BUNDLE_EXTRAS = {"chembench": {"particles": 500, "q": 256}}
# One synthetic registered cell row: bundle, task, cell, seed.
_CELL_ROW = b"core synthetic_task 111 11\n"


def _manifest_reference(tmp_path, registered_grammar):
    """A minimal wave manifest, with or without a registered grammar version."""
    import hashlib

    from .run_label_policy import canonical_bytes

    settings = run_label_policy.bmd_runtime_settings("core", _label_args())
    if registered_grammar is not None:
        settings["grammar_version"] = registered_grammar
    body = {
        "schema": "research_wave_manifest_v1",
        "run_label": "development",
        "cohort": "synthetic_cohort",
        "provider": "mock",
        "model": None,
        "base_url_host": None,
        "provider_order": None,
        "runtime_settings": {"core": {"profile": "dev", **settings}},
        "rounds": {"core": settings["rounds"]},
        "policies": {"core": {"d0_policy": settings["d0_policy"],
                              "proposal_policy": settings["proposal_policy"]}},
        "calls": [{"bundle": "core", "file": "cells.txt",
                   "sha256": hashlib.sha256(_CELL_ROW).hexdigest()}],
    }
    manifest = {**body, "manifest_sha256":
                hashlib.sha256(canonical_bytes(body)).hexdigest()}
    path = tmp_path / "manifest.json"
    payload = json.dumps(manifest).encode()
    path.write_bytes(payload)
    (tmp_path / "cells.txt").write_bytes(_CELL_ROW)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
