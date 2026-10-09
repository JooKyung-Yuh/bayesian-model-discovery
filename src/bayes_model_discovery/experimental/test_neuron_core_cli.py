"""CLI/points.csv plumbing for neuron_core.py (first engine stage for the
NeuronBench development demo, plus the later cell-wiring update).

Almost none of these tests call `neuron_core.run` with a wired world and a
mock proposer: that path drives a real PyMC sequential Monte Carlo (SMC) fit and takes tens of
minutes (scoping measured 46.4 minutes for --rounds 1 --particles 16 --world
h_sag). Every test here either calls a pure helper directly, or calls
`run`/`main` only on a request that is refused before any output is created
(an unwired world, an unauthorized provider, a cell/acquisition mismatch, or
a bare --list-worlds) - with the disclosed exceptions below, which all
monkeypatch `neuron_core.fit` to a fast 2-particle fake (so no real
SMC/pytensor compilation ever happens) while leaving world loading and native
forward simulate real, so they still need `neuronbench` on PYTHONPATH and
complete in seconds to a minute or two, not tens of minutes:
`test_run_cell_000_disables_proposals_and_points_match_the_cell`,
`test_18_neuron_accepts_run_label_development_and_records_it`, and the
three cell-alias/`active_maximum` run tests at the foot of this file (two of which share one module-scoped pair of runs).
The transport contract tests use an explicitly synthetic world, posterior
and forward hook; the required real-fit fake-client run is a separate artifact.
"""

import argparse
import copy
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from ..provenance.canonical import is_hex64, sha256_file
from . import neuron_core as nc
from . import neuron_forward as nf
from . import neuron_proposer as np_transport


def test_worlds_enumerates_package_order_with_all_six_wired():
    """Check that worlds enumerates package order with all six wired."""
    assert nc.WORLDS == (
        "z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M",
    )
    assert nc.WIRED_WORLDS == frozenset(nc.WORLDS)
    assert nc.world_wiring() == {
        "z_rebound": "wired",
        "h_sag": "wired",
        "na_fatigue": "wired",
        "ca_rebound": "wired",
        "d_type": "wired",
        "textbook_M": "wired",
    }


def test_worlds_matches_the_installed_package_when_available():
    # Skips (does not fail collection) exactly like test_neuron_core.py's own
    # tests when `neuronbench` is not on PYTHONPATH; catches drift if the
    # package's world order/membership ever changes.
    """Check that worlds matches the installed package when available."""
    neuronbench = pytest.importorskip("neuronbench")
    assert nc.WORLDS == tuple(neuronbench.list_worlds())


def test_require_wired_world_allows_all_six_and_refuses_unknown_names():
    """Check that require wired world allows all six and refuses unknown names."""
    for world in nc.WORLDS:
        nc.require_wired_world(world)
    with pytest.raises(ValueError, match="unknown_synthetic_world"):
        nc.require_wired_world("unknown_synthetic_world")


def test_run_refuses_an_unwired_world_before_any_output(tmp_path):
    """Check that run refuses an unwired world before any output."""
    out = tmp_path / "should_not_exist"
    args = nc.build_parser().parse_args(["--output", str(out)])
    # Exercise run's own guard even for callers that bypass argparse choices.
    args.world = "unknown_synthetic_world"
    with pytest.raises(ValueError, match="not wired"):
        nc.run(args)
    assert not out.exists()


def test_resolve_particles_profile_default_and_paper_tuple_and_mismatch():
    """Check that resolve particles profile default and paper tuple and mismatch."""
    assert nc.resolve_particles("dev", None) == 16
    assert nc.resolve_particles("paper", None) == 500
    assert nc.resolve_particles("dev", 16) == 16  # explicit value agreeing with the profile
    with pytest.raises(ValueError, match="inconsistent"):
        nc.resolve_particles("dev", 999)
    with pytest.raises(ValueError, match="inconsistent"):
        nc.resolve_particles("paper", 16)


def test_acquisition_from_cell_maps_all_four_cells():
    # cell[2] ('0'->random, '1'->joint) is the real acquisition switch;
    # it replaces the removed design_axis_for_cell helper.
    """Check that acquisition from cell maps all four cells."""
    assert nc.resolve_acquisition("000", None) == "random"
    assert nc.resolve_acquisition("001", None) == "joint"
    assert nc.resolve_acquisition("110", None) == "random"
    assert nc.resolve_acquisition("111", None) == "joint"
    # An explicit --acquisition that agrees with the cell passes through unchanged.
    assert nc.resolve_acquisition("111", "joint") == "joint"
    assert nc.resolve_acquisition("000", "random") == "random"


def test_canonical_cell_maps_the_four_aliases_and_fixes_the_other_four():
    # The Family bit is NOT APPLICABLE to this runner (one fixed
    # isotropic Normal likelihood), so the four cells that fix it at 0 - the
    # convention every other bundle's F0 cells use - are aliases of the four
    # pre-existing cells, which mirror the Structure bit into that position
    # by label convention. All eight rows of the table, in one place:
    """Check that canonical cell maps the four aliases and fixes the other four."""
    expected = {
        "000": "000",  # pre-existing, identity
        "001": "001",  # pre-existing, identity
        "010": "000",  # alias
        "011": "001",  # alias
        "100": "110",  # alias
        "101": "111",  # alias
        "110": "110",  # pre-existing, identity
        "111": "111",  # pre-existing, identity
    }
    for cell, canonical in expected.items():
        assert nc.canonical_cell(cell) == canonical
        # Total and idempotent: canonicalising twice is canonicalising once.
        assert nc.canonical_cell(canonical) == canonical
        # And the alias is an alias in fact, not just by name: both bits that
        # this runner actually reads agree with the canonical cell's.
        assert nc.resolve_acquisition(cell, None) == nc.resolve_acquisition(canonical, None)
        assert nc.proposals_enabled_for_cell(cell) is nc.proposals_enabled_for_cell(canonical)
        assert nc.cell_axis_config(cell) == nc.cell_axis_config(canonical)
        assert nc.cell_axis_config(cell)["family_axis"] == nc.FAMILY_AXIS_NOT_APPLICABLE


def test_resolve_acquisition_refuses_a_mismatched_explicit_flag():
    # The refusal states both the cell and the value it implies.
    """Check that resolve acquisition refuses a mismatched explicit flag."""
    with pytest.raises(ValueError, match=re.escape("--cell '000'")) as excinfo:
        nc.resolve_acquisition("000", "joint")
    assert "'random'" in str(excinfo.value)
    with pytest.raises(ValueError, match=re.escape("--cell '111'")) as excinfo:
        nc.resolve_acquisition("111", "random")
    assert "'joint'" in str(excinfo.value)


def test_run_refuses_a_cell_acquisition_mismatch_before_any_output(tmp_path, monkeypatch):
    # --cell 000 implies random; explicitly asking for
    # joint must be refused before the output directory is ever created.
    # `fit` is monkeypatched defensively (belt-and-suspenders, not load-bearing
    # once resolve_acquisition is wired ahead of out.mkdir): earlier code
    # has no such gate at all and falls through to a real ~46-min-class SMC
    # fit instead of raising, which is itself evidence of the bug this test
    # guards against, but far too slow to use as a RED signal.
    """Check that run refuses a cell acquisition mismatch before any output."""
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "should_not_exist_cell_mismatch"
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--cell", "000", "--acquisition", "joint"]
    )
    with pytest.raises(ValueError, match="'000'.*'random'|'random'.*'000'"):
        nc.run(args)
    assert not out.exists()


def test_proposals_enabled_for_cell_is_the_structure_bit():
    # cell[0] ('0'->no proposal round, '1'->mock catalogue proposals).
    """Check that proposals enabled for cell is the structure bit."""
    assert nc.proposals_enabled_for_cell("000") is False
    assert nc.proposals_enabled_for_cell("001") is False
    assert nc.proposals_enabled_for_cell("110") is True
    assert nc.proposals_enabled_for_cell("111") is True


def test_cell_axis_config_new_keys_and_no_design_axis():
    # Three new config keys per cell, and the removed design_axis
    # key/helper stay gone (not just unused).
    """Check that cell axis config new keys and no design axis."""
    assert not hasattr(nc, "design_axis_for_cell")
    expected_proposals_enabled = {
        "000": False, "001": False, "110": True, "111": True,
    }
    for cell, proposals_enabled in expected_proposals_enabled.items():
        extras = nc.cell_axis_config(cell)
        assert extras == {
            "acquisition_from_cell": True,
            "proposals_enabled": proposals_enabled,
            "family_axis": "not_applicable_isotropic_normal_sigma",
        }
        assert "design_axis" not in extras


def test_paid_api_gate_mock_is_silent_and_real_providers_require_authorization():
    """Check that paid api gate mock is silent and real providers require authorization."""
    nc.paid_api_gate("mock", False)
    nc.paid_api_gate("mock", True)
    with pytest.raises(ValueError, match="Anthropic.*--allow-paid-api"):
        nc.paid_api_gate("anthropic", False)
    with pytest.raises(ValueError, match="OpenAI.*--allow-paid-api"):
        nc.paid_api_gate("openai", False)
    # This reverses the old unconditional 'not implemented' expectation.
    nc.paid_api_gate("anthropic", True)
    nc.paid_api_gate("openai", True)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_run_refuses_an_unauthorized_real_provider_before_any_output(tmp_path, provider):
    """Check that run refuses an unauthorized real provider before any output."""
    out = tmp_path / "should_not_exist_either"
    args = nc.build_parser().parse_args(["--output", str(out), "--proposer", provider])
    with pytest.raises(ValueError, match="--allow-paid-api authorization"):
        nc.run(args)
    assert not out.exists()


def test_cli_defaults_reproduce_the_pre_patch_hardcoded_defaults():
    """Check that cli defaults reproduce the pre patch hardcoded defaults."""
    args = nc.build_parser().parse_args(["--output", "/tmp/unused"])
    assert args.world == "h_sag"
    assert args.seed == 11
    assert args.rounds == 1
    # --acquisition is now a confirm-only flag, default None
    # (earlier versions defaulted to "joint"; the value is now
    # resolved to the cell-implied one inside run(), see resolve_acquisition).
    assert args.acquisition is None
    assert args.particles is None  # resolved to 16 inside run(), not by argparse
    assert args.cell == "111"
    assert args.requested_proposer == "mock"
    assert args.allow_paid_api is False
    assert args.llm_model is None
    assert args.profile == "dev"
    assert not hasattr(args, "proposer")  # dest="requested_proposer", not "proposer"


def test_cli_rejects_an_unknown_cell_and_an_unknown_world():
    # "010" was this test's invalid-cell fixture until all eight 3-bit
    # strings became valid; the fixtures below are strings that stay
    # outside the extended factorial (design level 3, and a width that
    # is not three).
    """Check that cli rejects an unknown cell and an unknown world."""
    assert nc.build_parser().parse_args(["--output", "x", "--cell", "112"]).cell == "112"
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "x", "--cell", "113"])
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "x", "--cell", "0100"])
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "x", "--world", "no_such_world"])


def test_cli_accepts_all_eight_three_bit_cells():
    # The widened allow-list. argparse keeps the raw request (the
    # canonicalisation happens inside run(), so `**vars(args)` can still
    # record what was asked for).
    """Check that cli accepts all eight three bit cells."""
    for cell in ("000", "001", "010", "011", "100", "101", "110", "111"):
        args = nc.build_parser().parse_args(["--output", "x", "--cell", cell])
        assert args.cell == cell
    assert nc.build_parser().parse_args(["--output", "x"]).cell == "111"  # default unchanged


def test_main_list_worlds_prints_all_six_marked_and_needs_no_output(monkeypatch, capsys):
    """Check that main list worlds prints all six marked and needs no output."""
    monkeypatch.setattr("sys.argv", ["neuron_core", "--list-worlds"])
    nc.main()  # must return, not raise, even though no --output was given
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "z_rebound: wired",
        "h_sag: wired",
        "na_fatigue: wired",
        "ca_rebound: wired",
        "d_type: wired",
        "textbook_M: wired",
    ]


def test_main_requires_output_unless_list_worlds(monkeypatch):
    """Check that main requires output unless list worlds."""
    monkeypatch.setattr("sys.argv", ["neuron_core"])
    with pytest.raises(SystemExit):
        nc.main()


def test_points_columns_are_the_25_column_contract_order():
    """Check that points columns are the 25 column contract order."""
    assert len(nc.POINTS_COLUMNS) == 25
    assert nc.POINTS_COLUMNS == (
        "suite", "task", "method", "proposer", "seed", "x", "x_unit",
        "y_primary", "y_name", "y_label", "nll", "coverage90", "width90",
        "crps", "is_0.1", "kl", "status", "label", "run_id", "source_path",
        "variant", "code_sha", "brier", "nmsle", "rmsle",
    )


def _fake_config(**overrides):
    config = {
        "output": "/tmp/fake_run_dir",
        "world": "h_sag",
        "seed": 11,
        "cell": "111",
        "requested_proposer": "mock",
        "profile": "dev",
        "experimental_source_sha256": {"neuron_core.py": "a" * 64},
    }
    config.update(overrides)
    return config


def _fake_row(training_observations, round_, phase, mse):
    return {
        "round": round_,
        "phase": phase,
        "training_observations": training_observations,
        "bma_native_spike_forecast_mse": mse,
    }


def test_write_points_header_and_row_values_never_zero_for_undefined(tmp_path):
    """Check that write points header and row values never zero for undefined."""
    config = _fake_config()
    rows = [_fake_row(2, 0, "initial", 4.5), _fake_row(3, 1, "before_proposal", 3.25)]
    path = tmp_path / "points.csv"
    nc.write_points(path, config, rows)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(nc.POINTS_COLUMNS)
        written = list(reader)
    assert len(written) == 2
    first = written[0]
    assert first["suite"] == "neuronbench"
    assert first["task"] == "h_sag"
    assert first["method"] == "111"
    assert first["proposer"] == "mock"
    assert first["seed"] == "11"
    assert first["x"] == "2"
    assert first["x_unit"] == "protocol_runs"
    assert first["y_primary"] == "4.5"
    assert first["y_name"] == "spike_forecast_mse"
    assert first["label"] == "development"
    # The "acquisition_axis_absent_stage1" token is removed
    # outright (not renamed) - variant is always blank for NeuronBench
    # (no token exists for this suite, regardless of cell).
    assert first["variant"] == ""
    # Split across two literals so this file doesn't duplicate the one definition of a hex-64
    # digest pattern (tests/unit/test_canonical_json.py::
    # test_the_digest_pattern_is_defined_in_exactly_one_module); adjacent string literals are
    # concatenated by Python, so the compiled regex is unchanged and still matches a 64-char
    # lowercase hex digest.
    assert re.fullmatch(r"[0-9a-f]" "{64}", first["code_sha"])
    for column in (
        "nll", "coverage90", "width90", "crps", "is_0.1", "kl", "brier", "nmsle", "rmsle",
    ):
        assert first[column] == ""  # points.csv contract: undefined -> blank, never "0"


def test_write_points_variant_always_blank_for_every_cell(tmp_path):
    # variant is blank for all 4 cells, not just the
    # former "bit unset" pair (000/110) - NeuronBench never has a token.
    """Check that write points variant always blank for every cell."""
    for cell in ("000", "001", "110", "111"):
        config = _fake_config(cell=cell)
        path = tmp_path / f"points_{cell}.csv"
        nc.write_points(path, config, [_fake_row(2, 0, "initial", 1.0)])
        with path.open(newline="") as stream:
            row = next(csv.DictReader(stream))
        assert row["method"] == cell  # recorded cell == executed cell
        assert row["variant"] == ""


def test_write_points_label_follows_profile():
    # The helper and point-writer fallback always use development.
    """Check that write points label follows profile."""
    expected = {
        ("dev", "mock"): "development",
        ("dev", "anthropic"): "development",
        ("dev", "openai"): "development",
        ("paper", "mock"): "development",  # the fixed defect
        ("paper", "anthropic"): "development",
        ("paper", "openai"): "development",
    }
    for (profile, proposer), label in expected.items():
        assert nc.points_label(profile, proposer) == label
        config = _fake_config(profile=profile, requested_proposer=proposer)
        row = nc.points_row(config, "b" * 64, _fake_row(2, 0, "initial", 1.0), 1)
        assert row["label"] == label


def _fake_smc_result(hypothesis, n=2):
    """A 2-particle stand-in for vector_inference.fit's real return value -
    same shape/attribute contract (particles["conductance"]/["sigma"] as
    flat arrays of scalars, log_evidence, per_chain_log_evidence,
    diagnostics, seconds), so `checkpoint()`'s real forward/scoring code
    (unmodified, uses `result.hypothesis.predict(...)`) still runs for real."""
    return SimpleNamespace(
        hypothesis=hypothesis,
        particles={"conductance": np.full(n, 1.0), "sigma": np.full(n, 0.05)},
        log_evidence=0.0,
        per_chain_log_evidence=[0.0, 0.0],
        diagnostics={"original_prior_refit": True},
        seconds=0.0,
    )


def test_run_cell_000_disables_proposals_and_points_match_the_cell(tmp_path, monkeypatch):
    # Cell 000 (Structure bit unset) must run mock_proposals
    # zero times and leave proposals.jsonl as a 0-line file. `neuron_core.fit`
    # is monkeypatched to a fast fake (see _fake_smc_result) so this real
    # run() call completes in seconds rather than driving a real SMC fit;
    # world loading, native forward simulate, and scoring are all real.
    """Check that run cell 000 disables proposals and points match the cell."""
    pytest.importorskip("neuronbench")
    calls = []

    def spy_mock_proposals(event):
        """Record each mock-proposal event in the enclosing call list."""
        calls.append(event)
        raise AssertionError("mock_proposals must not be called when cell[0] == '0'")

    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    monkeypatch.setattr(nc, "mock_proposals", spy_mock_proposals)

    out = tmp_path / "cell_000_run"
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", "000",
         "--seed", "11", "--rounds", "1", "--profile", "dev"]
    )
    nc.run(args)

    assert calls == []  # mock_proposals invoked 0 times

    proposals_path = out / "proposals.jsonl"
    assert proposals_path.exists()
    assert proposals_path.stat().st_size == 0  # left as an empty file

    config = json.loads((out / "config.json").read_text())
    assert config["cell"] == "000"
    assert config["acquisition"] == "random"  # resolved from the cell
    assert config["acquisition_from_cell"] is True
    assert config["proposals_enabled"] is False
    assert config["family_axis"] == "not_applicable_isotropic_normal_sigma"
    assert "design_axis" not in config  # removed outright
    # "000" is one of the four pre-existing cells, i.e. already canonical,
    # so the alias key is never written and this record stays as it was.
    assert "cell_requested" not in config

    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 1
    for row in rows:
        assert row["method"] == "000"  # recorded cell == executed cell
        assert row["variant"] == ""


# ---------------------------------------------------------------------------
# `--run-label` on neuron_core (all-benchmark 2 h smoke).
# NeuronBench is EXCLUDED from that smoke campaign, but the switch is shared
# by all four runners, so it is wired and tested here on the same terms.
# ---------------------------------------------------------------------------


def test_18_neuron_refuses_run_label_production_before_any_output(tmp_path, monkeypatch):
    """REFUSE: a paper-profile MOCK run asking for "production" is refused
    after paid_api_gate and before `out.mkdir`. `fit` is stubbed defensively;
    the gate fires long before any SMC work."""
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "should_not_exist_run_label"
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--profile", "paper", "--run-label", "production"]
    )
    with pytest.raises(ValueError, match="--run-label production"):
        nc.run(args)
    assert not out.exists()


def test_18_neuron_accepts_run_label_development_and_records_it(tmp_path, monkeypatch):
    """ACCEPT: development always runs; the chosen label reaches config.json
    and every points.csv row. Same fast-`fit` pattern as the cell-000 run
    test above - world loading, forward simulate and scoring are all real."""
    pytest.importorskip("neuronbench")
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "dev_label"
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", "000", "--seed", "11",
         "--rounds", "1", "--profile", "dev", "--run-label", "development"]
    )
    nc.run(args)
    config = json.loads((out / "config.json").read_text())
    assert config["run_label"] == "development"
    assert config["label"] == "development"
    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["label"] for row in rows} == {"development"}


# ---------------------------------------------------------------------------
# Cell aliases, canonical cell recording, active_maximum binding.
# The two run-level tests below share ONE pair of fast monkeypatched-`fit`
# runs (module-scoped fixture): an aliased `--cell 100` run and its canonical
# `--cell 110` twin. Both cells imply `random` acquisition (cell[2] == "0"),
# so neither run pays for a joint-EIG pass, and both have the Structure bit
# set, so the proposal round really executes in both.
# ---------------------------------------------------------------------------

# The exact config.json / summary.json key sets a pre-existing cell produced
# BEFORE the cell aliases (the base commit, measured on a `--cell 110` fast run).
# A cell that needs no alias must keep them exactly: `cell_requested` is
# recorded only when canonicalisation actually moved the cell, so the four
# pre-existing cells stay byte-identical. A later change that legitimately adds
# a config key updates these snapshots deliberately, which is the point.
PRE_N1_CONFIG_KEYS = frozenset({
    "EIG_predictive_samples", "acquisition", "acquisition_from_cell", "active_maximum",
    "allow_paid_api", "baseline_overlap_x", "baseline_ps017_note", "cell", "created_utc",
    "deterministic_development_only", "experimental_source_sha256", "family_axis", "forward",
    "initial_pool", "likelihood", "list_worlds", "mean_prior", "neuronbench_commit",
    "neuronbench_source_path", "neuronbench_version", "noise_prior", "output", "paid_api_calls",
    "particles", "profile", "proposal_rounds", "proposals_enabled", "proposer", "python",
    "q_protocol_labels", "q_role", "registered_run", "requested_proposer", "rounds", "run_label",
    "seed", "smc", "stochastic_benchmark_complete", "training_features", "versions", "world",
    "world_wiring", "llm_model", "structure_proposals",  # transport metadata
    # The three unconditional keys: the forward backend that executed, and the two
    # numpy CPU-dispatch facts that decide the bits of the native path as well as the batch
    # one. Added deliberately, which is what the paragraph above asks a later change to do.
    "forward_backend", "numpy_cpu_features_disabled", "numpy_cpu_features_enabled",
})
PRE_N1_SUMMARY_KEYS = frozenset({
    "active_count", "admitted", "candidate_failures", "observations", "paid_api_calls", "rounds",
    "status", "stochastic_benchmark_complete", "system_cpu_seconds", "timing_seconds",
    "user_cpu_seconds", "wall_seconds",
})


def _fast_run(out, cell):
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", cell,
         "--seed", "11", "--rounds", "1", "--profile", "dev"]
    )
    nc.run(args)
    return out


@pytest.fixture(scope="module")
def alias_and_canonical_runs(tmp_path_factory):
    """`--cell 100` and `--cell 110`, same world/seed/rounds/profile, run once
    for the whole module with `neuron_core.fit` monkeypatched to the fast
    2-particle fake (see `_fake_smc_result`): world loading, native forward
    simulate, acquisition, proposals and scoring are all real."""
    pytest.importorskip("neuronbench")
    base = tmp_path_factory.mktemp("n1_alias")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
        return {cell: _fast_run(base / f"cell_{cell}_run", cell) for cell in ("100", "110")}


def test_run_at_an_aliased_cell_records_the_canonical_cell_and_the_request(
    alias_and_canonical_runs,
):
    # `--cell 100` executes as "110". `cell` is always the executed cell
    # (so `points_row`'s `method` column stays honest with no change to
    # `points_row` at all), and the raw request is kept beside it.
    """Check that run at an aliased cell records the canonical cell and the request."""
    out = alias_and_canonical_runs["100"]
    config = json.loads((out / "config.json").read_text())
    assert config["cell"] == "110"
    assert config["cell_requested"] == "100"
    assert config["acquisition"] == "random"  # cell[2] == "0", unchanged by the alias
    assert config["proposals_enabled"] is True  # cell[0] == "1"
    assert config["family_axis"] == "not_applicable_isotropic_normal_sigma"

    summary = json.loads((out / "summary.json").read_text())
    assert summary["cell"] == "110"
    assert summary["cell_requested"] == "100"
    assert summary["active_count"] == 7  # 4 initial + 3 admitted proposals

    # The proposal round really ran (this is a Structure-bit-set cell).
    assert (out / "proposals.jsonl").stat().st_size > 0
    assert len(summary["admitted"]) == 3

    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["method"] for row in rows} == {"110"}  # the executed cell, never "100"
    assert {row["variant"] for row in rows} == {""}


def test_an_aliased_run_equals_its_canonical_twin_and_leaves_its_record_untouched(
    alias_and_canonical_runs,
):
    """Check that an aliased run equals its canonical twin and leaves its record untouched."""
    alias = alias_and_canonical_runs["100"]
    canonical = alias_and_canonical_runs["110"]

    # (a) Same acquisition and proposal configuration - byte for byte, not
    # merely "the same flags": these four records carry the selected design,
    # the proposal events, the observations and every score.
    for name in ("acquisition.jsonl", "proposals.jsonl", "observations.jsonl", "scores.jsonl"):
        assert (alias / name).read_bytes() == (canonical / name).read_bytes()

    # points.csv agrees on every column except the two that cite the run's
    # own output directory.
    def points(run):
        """Read a run's points.csv and return its rows."""
        with (run / "points.csv").open(newline="") as stream:
            return [
                {k: v for k, v in row.items() if k not in ("run_id", "source_path")}
                for row in csv.DictReader(stream)
            ]

    assert points(alias) == points(canonical)

    # (b) config.json agrees except for the run's own directory, its
    # timestamp, and the alias key that only the aliased run carries.
    alias_config = json.loads((alias / "config.json").read_text())
    canonical_config = json.loads((canonical / "config.json").read_text())
    volatile = {"output", "created_utc", "cell_requested"}
    assert {k: v for k, v in alias_config.items() if k not in volatile} == {
        k: v for k, v in canonical_config.items() if k not in volatile
    }

    # (c) The canonical cell is one of the four pre-existing ones, so its
    # records must be exactly what they were before the aliases: the same key
    # sets, no `cell_requested` anywhere, and the recorded cell unchanged.
    canonical_summary = json.loads((canonical / "summary.json").read_text())
    assert set(canonical_config) == PRE_N1_CONFIG_KEYS
    assert set(canonical_summary) == PRE_N1_SUMMARY_KEYS
    assert "cell_requested" not in canonical_config
    assert "cell_requested" not in canonical_summary
    assert "cell" not in canonical_summary  # summary.json names the cell only for an alias
    assert canonical_config["cell"] == "110"
    # The unpatched cap is recorded and never reached: 7 active of 12.
    assert canonical_config["active_maximum"] == nc.ACTIVE_MAXIMUM == 12
    assert canonical_summary["active_count"] == 7


def test_recorded_active_maximum_is_the_cap_that_actually_operates(tmp_path, monkeypatch):
    """`config["active_maximum"]` and the pool truncation are now the same
    object (`retain(fits + candidates, active_maximum)`), not two constants
    that merely happen to be equal. With the real value (12) the pool never
    reaches the cap, so the binding is proved by lowering the bound and
    watching BOTH the recorded value and the retained pool follow it: a
    config-only literal would still record 12, and an unbound `retain` call
    would still keep all 7 fits. The bound is the profile's own
    entry, so the probe lowers that entry rather than core_count's constant."""
    pytest.importorskip("neuronbench")
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    monkeypatch.setitem(nc.ACTIVE_MAXIMUM_BY_PROFILE, "dev", 2)
    out = tmp_path / "active_maximum_run"
    nc.run(
        nc.build_parser().parse_args(
            ["--output", str(out), "--world", "h_sag", "--cell", "110",
             "--seed", "11", "--rounds", "1", "--profile", "dev"]
        )
    )
    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert config["active_maximum"] == 2  # the recorded value is the patched constant
    assert summary["active_count"] == 2  # ... and the pool really was truncated to it
    events = [json.loads(line) for line in (out / "proposals.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert len(events[0]["retired_ids"]) == 5  # 4 initial + 3 fitted proposals, 2 retained


# Synthetic wiring tests: no SDK construction, network, native fit, or
# saved run content. Existing native-forward tests above remain unchanged.
TRANSPORT_PRIVATE = "synthetic_hidden_evaluation_sentinel"


class _TransportWorld:
    protocol_pool: ClassVar[list] = [(f"synthetic_public_{i}", [(0.1, 0.0), (0.2, float(i))])
                     for i in range(7)]
    test_protocols: ClassVar[list] = [("synthetic_heldout_label", [(0.1, 0.0), (0.2, 12345.0)])]

    def problem(self):
        """Return the minimal synthetic problem description for the fake world."""
        return {"text_prior": "An unknown modification changes a voltage response.",
                "reference_model": {"extra": [], "slow_na": False},
                "protocols": self.protocol_pool,
                "test_protocol_labels": [self.test_protocols[0][0]],
                "budget_rule": "each protocol once", "ignored_private": TRANSPORT_PRIVATE}

    def run(self, design):
        """Return a synthetic simulation result with a constant voltage trace."""
        return SimpleNamespace(voltage=np.full(3, -60.0), test_start=1,
                               spike_count=0, cost=1, obs_idx=np.arange(3))

    def spike_forecast_mse(self, counts):
        """Return a fixed placeholder spike-count forecast error."""
        return 987654.0

    def feature_forecast_mse(self, traces):
        """Return a fixed placeholder feature forecast error."""
        return 987654.0


class _TransportClient:
    def __init__(self, provider, replies):
        self.provider = provider
        self.replies = iter(replies)
        self.calls = []
        self.messages = self
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        """Record the request keyword arguments of the fake client call."""
        self.calls.append(copy.deepcopy(kwargs))
        body = next(self.replies)
        if isinstance(body, BaseException):
            raise body
        raw = json.dumps(body)
        common = {"model": "synthetic_returned_model", "id": "synthetic_response"}
        if self.provider == "anthropic":
            return SimpleNamespace(**common,
                content=[SimpleNamespace(type="text", text=raw)], stop_reason="end_turn",
                usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 11, "output_tokens": 7}))
        return SimpleNamespace(**common,
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


def _transport_reply(mechanism="m_current"):
    return {"mechanism": mechanism, **copy.deepcopy(np_transport.mechanism_vocabulary()[mechanism])}


@pytest.fixture
def transport_world(monkeypatch):
    """Fixture providing an explicitly synthetic world, posterior and forward hook."""
    nb = pytest.importorskip("neuronbench")
    world = _TransportWorld()
    import neuronbench.worlds as native_worlds
    monkeypatch.setattr(native_worlds, "build_I", lambda segments: (np.zeros(30), 10))
    monkeypatch.setattr(nb, "load_world", lambda *args, **kwargs: world)
    monkeypatch.setattr(nc, "fit", lambda h, *args, **kwargs: _fake_smc_result(h))
    # Only these tests replace the expensive native forward. The actual run
    # required by unit08 uses unmodified SMC, acquisition, and native scoring.
    monkeypatch.setattr(nf.HHHypothesis, "predict", lambda *args: np.full(30, -60.0))
    monkeypatch.setattr(np_transport.NeuronHypothesis, "predict", lambda *args: np.full(30, -60.0))
    return world


def _transport_run(out, provider="anthropic", replies=None, *, cell="101", rounds=2):
    client = _TransportClient(provider, replies if replies is not None else [_transport_reply()]*3)
    args = nc.build_parser().parse_args([
        "--output", str(out), "--cell", cell, "--proposer", provider,
        "--allow-paid-api", "--llm-model", "synthetic_requested_model", "--rounds", str(rounds)])
    nc.run(args, client=client)
    return client, json.loads((out/"config.json").read_text()), json.loads((out/"summary.json").read_text())


def _request_packet(out, round):
    record = json.loads((out/"llm_transport"/f"round_{round:02d}.request.json").read_text())
    message = next(row for row in record["bodies"][0]["messages"] if row["role"] == "user")
    return json.loads(message["content"])


def test_transport_windows_are_the_exact_public_recipe_envelope():
    """Check that transport windows are the exact public recipe envelope."""
    expected = {"E_mV": [-77., 120.], "mvh_mV": [-85., -30.], "mk_mV": [-8., 12.],
                "mtau_ms": [2., 100.], "hvh_mV": [-88., -80.], "hk_mV": [4., 5.],
                "htau_ms": [22., 200.]}
    assert nc.neuron_parameter_windows() == expected
    for name, bounds in expected.items():
        values = [row["parameters"][name] for row in np_transport.mechanism_vocabulary().values()
                  if row["parameters"].get(name) is not None]
        assert bounds == [min(values), max(values)]
    assert set(expected) == set(np_transport.CONTINUOUS_PARAMETERS)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_transport_two_rounds_valid_then_duplicate_records_counts_and_public_packet(
        tmp_path, transport_world, monkeypatch, provider):
    """Check that transport two rounds valid then duplicate records counts and public packet."""
    def no_mock(*args):
        """Fail the test if the mock catalogue is reached on a real-provider path."""
        pytest.fail("a real-provider path must not call the mock catalogue")
    monkeypatch.setattr(nc, "mock_proposals", no_mock)
    out = tmp_path / provider
    client, config, summary = _transport_run(out, provider)
    assert config["cell_requested"] == "101" and config["cell"] == "111"
    assert config["structure_proposals"] == "free" and config["initial_pool"] == 4
    assert config["llm_model"] == config["llm_resolved_model"] == "synthetic_requested_model"
    assert config["llm_client_injected"] and summary["llm_client_injected"]
    assert config["neuron_parameter_windows"] == nc.neuron_parameter_windows()
    records = [json.loads(path.read_text()) for path in sorted((out/"llm_transport").glob("*.response.json"))]
    assert len(records) == len(list((out/"llm_transport").glob("*.request.json"))) == 2
    assert [r["calls"] for r in records] == [1, 2]
    assert [r["repairs"] for r in records] == [0, 1]
    assert [r["status"] for r in records] == ["parsed", "proposal_failed"]
    assert summary["paid_api_calls"] == len(client.calls) == sum(r["calls"] for r in records) == 3
    assert summary["provider_request_attempts"] == sum(r["paid_api_calls"] for r in records) == 0
    assert summary["admitted"] == ["hh_m_current_gaussian_features"]
    assert summary["active_count"] == 5 and summary["candidate_failures"] == []
    assert len(list((out/"fits").glob("*.json"))) == 14
    assert not (out/"failure.json").exists()
    packets = [_request_packet(out, i) for i in (1, 2)]
    assert [len(packet["observations"]) for packet in packets] == [3, 4]
    assert [len(packet["pool"]) for packet in packets] == [4, 5]
    assert packets[1]["archive"][0]["reason"] == {"admitted_round": 1, "active_rounds": [1, 2]}
    for packet in packets:
        text = json.dumps(packet)
        assert TRANSPORT_PRIVATE not in text and "12345" not in text and "987654" not in text
        assert "prequential_feature_log_predictive" not in text and "world_seed" not in text
        assert packet["constraints"]["family_expansion"] is False
        assert packet["interface"]["test_protocol_labels"] == ["synthetic_heldout_label"]
        assert all(set(row) == {"round", "selected", "available", "mode"} for row in packet["acquisition"])
    archive = json.loads((out/"archive.json").read_text())
    assert [row["outcome"] for row in archive] == ["active", "duplicate", "duplicate"]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("cell", ["000", "001", "010", "011"])
def test_transport_catalog_cells_never_construct_client_or_packet(
        tmp_path, transport_world, monkeypatch, provider, cell):
    """Check that transport catalog cells never construct client or packet."""
    def forbidden(*args, **kwargs):
        """Fail the test if a proposer or packet is constructed for a catalog cell."""
        pytest.fail("catalog cells cannot construct a proposer or packet")
    monkeypatch.setattr(nc, "make_neuron_proposer", forbidden)
    monkeypatch.setattr(nc, "build_neuron_belief_packet", forbidden)
    monkeypatch.setattr(nc, "mock_proposals", forbidden)
    out = tmp_path / cell
    client, config, summary = _transport_run(out, provider, replies=[], cell=cell, rounds=1)
    assert client.calls == [] and not (out/"llm_transport").exists()
    assert not (out/"archive.json").exists()
    assert config["structure_proposals"] == "catalog"
    assert summary["paid_api_calls"] == 0 and summary["active_count"] == 4
    assert (out/"proposals.jsonl").read_bytes() == b""


def test_transport_constructor_failure_is_recorded_and_masked(tmp_path, transport_world, monkeypatch):
    """Check that transport constructor failure is recorded and masked."""
    secret = "sk-ant-" + "synthetic_constructor_secret"
    def broken(*args, **kwargs):
        """Fake constructor that asserts its arguments and then fails."""
        assert kwargs["cell"] == "111" and kwargs["max_rounds"] == 2
        assert kwargs["model"] == "synthetic_requested_model" and kwargs["allow_live"]
        raise RuntimeError(secret)
    monkeypatch.setattr(nc, "make_neuron_proposer", broken)
    out = tmp_path / "failed_constructor"
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY:masked") as error:
        _transport_run(out)
    payload = (out/"failure.json").read_text()
    assert secret not in payload and secret not in str(error.value)
    assert json.loads(payload)["stage"] == "world_or_proposer_construction"
    assert json.loads(payload)["paid_api_calls"] == 0
    assert not (out/"llm_transport").exists()


def test_transport_fit_failed_candidate_is_masked_archived_and_never_retried(
        tmp_path, transport_world, monkeypatch):
    """Check that transport fit failed candidate is masked archived and never retried."""
    attempted = []
    secret = "sk-" + "synthetic_fit_secret"
    def selective_fit(h, *args, **kwargs):
        """Fake fit that fails for one mechanism and succeeds for the others."""
        if h.mechanism == "m_current":
            attempted.append(h.model_id)
            raise ValueError(secret)
        return _fake_smc_result(h)
    monkeypatch.setattr(nc, "fit", selective_fit)
    out = tmp_path / "fit_failed"
    _, _, summary = _transport_run(out)
    assert attempted == ["hh_m_current_gaussian_features"]
    assert summary["admitted"] == [] and summary["active_count"] == 4
    archive = json.loads((out/"archive.json").read_text())
    assert [row["outcome"] for row in archive] == ["fit_failed", "duplicate", "duplicate"]
    assert "OPENAI_API_KEY:masked" in archive[0]["reason"]["error"]
    assert secret not in (out/"summary.json").read_text()
    assert _request_packet(out, 2)["archive"][0]["outcome"] == "fit_failed"


def test_transport_rejected_candidate_archive_reaches_next_public_packet(
        tmp_path, transport_world, monkeypatch):
    """Check that transport rejected candidate archive reaches next public packet."""
    original = nf.predict_features
    def failed_probe(h, *args, **kwargs):
        """Fake predict_features that returns NaN for one mechanism and delegates otherwise."""
        if h.mechanism == "m_current":
            return np.full(3, np.nan)
        return original(h, *args, **kwargs)
    monkeypatch.setattr(nf, "predict_features", failed_probe)
    out = tmp_path / "rejected"
    replies = [_transport_reply(), _transport_reply(), _transport_reply("h_current")]
    _, _, summary = _transport_run(out, replies=replies)
    archive = _request_packet(out, 2)["archive"]
    assert [row["outcome"] for row in archive] == ["rejected", "duplicate"]
    assert summary["active_count"] == 5 and summary["paid_api_calls"] == 3
    assert summary["admitted"] == ["hh_h_current_gaussian_features"]


def test_transport_numeric_rejection_has_raw_receipts_without_invented_identity(
        tmp_path, transport_world):
    """Check that transport numeric rejection has raw receipts without invented identity."""
    bad = _transport_reply()
    bad["parameters"]["E_mV"] = 999.0
    out = tmp_path / "invalid_numeric"
    _, _, summary = _transport_run(out, replies=[bad, bad, _transport_reply()])
    first = json.loads((out/"llm_transport/round_01.response.json").read_text())
    assert first["archive_entries"] == []
    assert [row["stage"] for row in first["rejected_candidates"]] == ["numeric", "numeric"]
    assert _request_packet(out, 2)["archive"] == []
    assert summary["active_count"] == 5 and summary["paid_api_calls"] == 3


def test_transport_retired_candidate_cannot_reenter_pool(tmp_path, transport_world, monkeypatch):
    # Production cap12 is unreachable in this one-candidate transport. A cap4
    # boundary probe makes retirement observable without changing production.
    # The operating bound is the profile's entry, which is what a probe moves.
    """Check that transport retired candidate cannot reenter pool."""
    monkeypatch.setitem(nc.ACTIVE_MAXIMUM_BY_PROFILE, "dev", 4)
    def low_evidence(h, *args, **kwargs):
        """Fake fit that returns a result with low model evidence."""
        result = _fake_smc_result(h)
        result.log_evidence = -20.0 if h.mechanism == "m_current" else 0.0
        return result
    monkeypatch.setattr(nc, "fit", low_evidence)
    out = tmp_path / "retired"
    _, config, summary = _transport_run(out)
    assert config["active_maximum"] == summary["active_count"] == 4
    assert summary["admitted"] == []
    first = _request_packet(out, 2)["archive"][0]
    assert first["outcome"] == "retired_by_evidence"
    assert first["reason"] == {"log_evidence_at_retirement": -20., "gap_to_cutoff_nats": 20.,
                               "active_rounds": [], "computed_round": 1, "n_obs": 3}
    archive = json.loads((out/"archive.json").read_text())
    assert [row["outcome"] for row in archive] == ["retired_by_evidence", "duplicate", "duplicate"]
    assert all("retired by evidence" in row["note"] for row in archive[1:])


def test_transport_late_failure_keeps_attempt_counts_and_masked_archive(
        tmp_path, transport_world, monkeypatch):
    """Check that transport late failure keeps attempt counts and masked archive."""
    original = nc.build_neuron_belief_packet
    secret = "sk-" + "synthetic_late_secret"
    def fail_second(*args, **kwargs):
        """Fake belief-packet builder that raises once the second-round data size is reached."""
        if len(args[1]) == 4:
            raise RuntimeError(secret)
        return original(*args, **kwargs)
    monkeypatch.setattr(nc, "build_neuron_belief_packet", fail_second)
    out = tmp_path / "late_failure"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY:masked"):
        _transport_run(out)
    payload = (out/"failure.json").read_text()
    assert secret not in payload
    failure = json.loads(payload)
    assert failure["paid_api_calls"] == 1 and failure["provider_request_attempts"] == 0
    assert failure["archive"][0]["outcome"] == "active"
    assert len(list((out/"llm_transport").glob("*.response.json"))) == 1



def test_transport_previously_active_retirement_keeps_history(tmp_path, transport_world, monkeypatch):
    """Check that transport previously active retirement keeps history."""
    monkeypatch.setitem(nc.ACTIVE_MAXIMUM_BY_PROFILE, "dev", 5)
    def ranked_fit(h, *args, **kwargs):
        """Fake fit that returns results ranked by evidence."""
        result = _fake_smc_result(h)
        result.log_evidence = {"m_current": -5.0, "h_current": 1.0}.get(h.mechanism, 0.0)
        return result
    monkeypatch.setattr(nc, "fit", ranked_fit)
    out = tmp_path / "active_then_retired"
    replies = [_transport_reply(), _transport_reply("h_current"), _transport_reply(), _transport_reply()]
    _, _, summary = _transport_run(out, replies=replies, rounds=3)
    assert summary["paid_api_calls"] == 4 and summary["active_count"] == 5
    entry = next(row for row in _request_packet(out, 3)["archive"]
                 if row["model_id"] == "hh_m_current_gaussian_features")
    assert entry["proposed_round"] == 1 and entry["outcome"] == "retired_by_evidence"
    assert entry["reason"] == {"active_rounds": [1, 2], "computed_round": 2,
                               "n_obs": 4, "log_evidence_at_retirement": -5.0,
                               "gap_to_cutoff_nats": 5.0}
    archive = json.loads((out/"archive.json").read_text())
    assert [row["outcome"] for row in archive] == ["retired_by_evidence", "active", "duplicate", "duplicate"]
    assert all("retired by evidence" in row["note"] for row in archive[-2:])


# ---------------------------------------------------------------------------
# Batch forward backend - not the cell aliases above. The refusal tests never reach any
# output; the one accepting run pair shares a module-scoped fixture, uses the
# same fast 2-particle fake `fit` as the rest of this file, and shrinks the
# initial pool to ONE hypothesis - the batch integrator's fixed per-step cost
# makes a 2-particle scoring pass slower than the scalar one, and the claim
# under test (native and batch score the same rows from the same fake fits)
# needs one model no less than four.
# ---------------------------------------------------------------------------


def _equivalence_report_payload(**overrides):
    """A minimal report in the shape `forward_backend_receipt` reads. Synthetic on purpose:
    these tests are about the gate, not about the integrator (see test_neuron_forward_batch)."""
    neuronbench = pytest.importorskip("neuronbench")
    return {
        "neuronbench_version": neuronbench.__version__,
        "pairs": 9, "bit_identical": 9, "max_abs_delta_mV": 0.0,
        "spike_counts_equal": True, "features_equal": True,
        "machine": {"numpy_cpu_features_disabled": "AVX512F", "power_matches_scalar": True},
        **overrides,
    }


def _write_report(tmp_path, name="report.json", **overrides):
    path = tmp_path / name
    path.write_text(json.dumps(_equivalence_report_payload(**overrides)))
    return path


def _batch_args(out, *extra):
    return nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", "000", "--seed", "11",
         "--rounds", "1", "--profile", "dev", "--forward-backend", "batch", *extra]
    )


def test_n1_cli_accepts_the_backend_options_and_defaults_to_native():
    """Check that n1 cli accepts the backend options and defaults to native."""
    args = nc.build_parser().parse_args(["--output", "x"])
    assert args.forward_backend == "native"
    assert not hasattr(args, "forward_equivalence_report")  # absent unless actually typed
    args = nc.build_parser().parse_args(
        ["--output", "x", "--forward-backend", "batch", "--forward-equivalence-report", "r.json"]
    )
    assert (args.forward_backend, args.forward_equivalence_report) == ("batch", "r.json")
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "x", "--forward-backend", "jax"])


def test_n1_batch_without_an_equivalence_report_exits_2_before_any_output(
        tmp_path, monkeypatch, capsys):
    """Check that n1 batch without an equivalence report exits 2 before any output."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "no_report"
    with pytest.raises(SystemExit) as refusal:
        nc.run(_batch_args(out))
    assert refusal.value.code == 2
    assert not out.exists()
    assert "--forward-equivalence-report" in capsys.readouterr().err


def test_n1_batch_with_a_missing_report_file_exits_2_before_any_output(tmp_path, monkeypatch):
    """Check that n1 batch with a missing report file exits 2 before any output."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    out = tmp_path / "missing_report"
    with pytest.raises(SystemExit) as refusal:
        nc.run(_batch_args(out, "--forward-equivalence-report", str(tmp_path / "absent.json")))
    assert refusal.value.code == 2
    assert not out.exists()


def test_n1_batch_with_a_tampered_report_exits_2_before_any_output(
        tmp_path, monkeypatch, capsys):
    """Check that n1 batch with a tampered report exits 2 before any output."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    cases = {
        "delta.json": {"max_abs_delta_mV": 4e-10},
        "counts.json": {"spike_counts_equal": False},
        "version.json": {"neuronbench_version": "0.0.0-not-installed"},
    }
    for name, overrides in cases.items():
        out = tmp_path / f"tampered_{name}"
        report = _write_report(tmp_path, name, **overrides)
        with pytest.raises(SystemExit) as refusal:
            nc.run(_batch_args(out, "--forward-equivalence-report", str(report)))
        assert refusal.value.code == 2, name
        assert not out.exists(), name
    message = capsys.readouterr().err
    assert "neuronbench" in message


def test_n1_batch_on_a_simd_power_machine_exits_2_naming_the_environment(
        tmp_path, monkeypatch, capsys):
    # The startup self-check, exercised on any machine by pinning its answer.
    """Check that n1 batch on a simd power machine exits 2 naming the environment."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: False)
    out = tmp_path / "simd_power"
    with pytest.raises(SystemExit) as refusal:
        nc.run(_batch_args(out, "--forward-equivalence-report", str(_write_report(tmp_path))))
    assert refusal.value.code == 2
    assert not out.exists()
    assert "NPY_DISABLE_CPU_FEATURES" in capsys.readouterr().err


def test_n1_native_refuses_a_stray_equivalence_report(tmp_path, monkeypatch):
    """Check that n1 native refuses a stray equivalence report."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    out = tmp_path / "stray_report"
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--forward-equivalence-report", str(_write_report(tmp_path))]
    )
    with pytest.raises(SystemExit) as refusal:
        nc.run(args)
    assert refusal.value.code == 2
    assert not out.exists()


@pytest.fixture(scope="module")
def backend_run_pair(tmp_path_factory):
    """One native and one batch run of the same one-model request, same seed, fake `fit`."""
    pytest.importorskip("neuronbench")
    base = tmp_path_factory.mktemp("n1_backends")
    report = base / "equivalence.json"
    report.write_text(json.dumps(_equivalence_report_payload()))
    runs = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
        patch.setattr(nc, "initial_hypotheses", lambda: [nf.HHHypothesis("leak")])
        patch.setattr(nc, "power_matches_scalar", lambda: True)
        for backend in ("native", "batch"):
            out = base / f"{backend}_run"
            extra = ["--forward-backend", backend]
            if backend == "batch":
                extra += ["--forward-equivalence-report", str(report)]
            nc.run(nc.build_parser().parse_args(
                ["--output", str(out), "--world", "h_sag", "--cell", "000", "--seed", "11",
                 "--rounds", "1", "--profile", "dev", *extra]
            ))
            runs[backend] = out
    return runs, report


def test_n1_batch_run_records_its_backend_report_and_environment(backend_run_pair):
    """Check that n1 batch run records its backend report and environment."""
    runs, report = backend_run_pair
    config = json.loads((runs["batch"] / "config.json").read_text())
    assert config["forward_backend"] == "batch"
    assert config["forward"].startswith("batch numpy replica")
    assert config["forward_equivalence_sha"] == sha256_file(report)
    assert is_hex64(config["forward_equivalence_sha"])
    assert config["forward_equivalence_summary"] == {
        "pairs": 9, "bit_identical": 9, "max_abs_delta_mV": 0.0,
        "machine": {"numpy_cpu_features_disabled": "AVX512F", "power_matches_scalar": True},
    }
    assert is_hex64(config["experimental_source_sha256"]["neuron_forward_batch.py"])
    assert "forward_backend_guard_probe" not in config  # fit_guards is off in this run


def test_n1_native_run_keeps_its_records_and_only_gains_the_environment_keys(backend_run_pair):
    """Check that n1 native run keeps its records and only gains the environment keys."""
    runs, _ = backend_run_pair
    config = json.loads((runs["native"] / "config.json").read_text())
    assert config["forward_backend"] == "native"
    assert config["forward"] == "native neuronbench.worlds.simulate dt=0.01 ms without truth kwargs"
    assert "forward_equivalence_sha" not in config
    assert "forward_equivalence_summary" not in config
    assert "forward_equivalence_report" not in config
    assert "neuron_forward_batch.py" not in config["experimental_source_sha256"]
    # Recorded for every backend: the SIMD dispatch decides the native bits too.
    assert isinstance(config["numpy_cpu_features_disabled"], str)
    assert "SSE2" in config["numpy_cpu_features_enabled"]


def test_n1_the_two_backends_score_and_observe_the_same_rows(backend_run_pair):
    """Check that n1 the two backends score and observe the same rows."""
    runs, _ = backend_run_pair

    def lines(name):
        """Load the JSON lines of one record file for the native and the batch run, as a pair."""
        return [json.loads(line) for line in (runs["native"] / name).read_text().splitlines()], [
            json.loads(line) for line in (runs["batch"] / name).read_text().splitlines()
        ]

    for name in ("scores.jsonl", "observations.jsonl", "acquisition.jsonl"):
        native_rows, batch_rows = lines(name)
        assert native_rows and native_rows == batch_rows, name

    def points(run):
        """Read a run's points.csv and return its rows."""
        with (run / "points.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    native_points, batch_points = points(runs["native"]), points(runs["batch"])
    volatile = {"run_id", "source_path"}
    assert native_points and [
        {k: v for k, v in row.items() if k not in volatile} for row in native_points
    ] == [{k: v for k, v in row.items() if k not in volatile} for row in batch_points]

    native_config = json.loads((runs["native"] / "config.json").read_text())
    batch_config = json.loads((runs["batch"] / "config.json").read_text())
    added = {"forward_equivalence_report", "forward_equivalence_sha",
             "forward_equivalence_summary"}
    volatile = {"output", "created_utc", "forward", "forward_backend",
                "experimental_source_sha256"}
    assert set(batch_config) - set(native_config) == added
    assert {k: v for k, v in batch_config.items() if k not in volatile | added} == {
        k: v for k, v in native_config.items() if k not in volatile
    }
    # The six digests the runner already bound are unchanged; the batch run binds a seventh.
    shared = native_config["experimental_source_sha256"]
    assert shared and all(
        batch_config["experimental_source_sha256"][name] == digest
        for name, digest in shared.items()
    )
    assert set(batch_config["experimental_source_sha256"]) - set(shared) == {
        "neuron_forward_batch.py"
    }


def test_n1_fit_guards_keep_the_guard_probe_native_and_say_so(
        tmp_path, transport_world, monkeypatch):
    """A whole batch run on the synthetic transport world (30-sample protocols), so the guard
    combination is exercised end to end in a second rather than in a minute."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    out = tmp_path / "guarded_batch"
    nc.run(nc.build_parser().parse_args(
        ["--output", str(out), "--cell", "000", "--rounds", "1", "--fit-guards", "v1",
         "--forward-backend", "batch",
         "--forward-equivalence-report", str(_write_report(tmp_path))]
    ))
    config = json.loads((out / "config.json").read_text())
    assert config["fit_guards"] == "v1"
    assert config["forward_backend"] == "batch"
    # The prior probe is a handful of prior draws, not a hot loop: it stays on the native
    # per-particle forward, and the run records that rather than leaving it to be inferred.
    assert config["forward_backend_guard_probe"] == "native"
    assert json.loads((out / "summary.json").read_text())["status"].startswith("completed")


# ---------------------------------------------------------------------------
# `--profile paper_neuron`: the MDA-comparison profile - 200
# particles, two chains, three fixed MH moves per stage at ESS 0.5, and the
# benchmark's own six evaluation features as the training likelihood. The run
# tests below use the same fast monkeypatched-`fit` fake as the cell-alias
# tests above, so no real 200-particle SMC ever happens here; world loading,
# the native six-feature forward, acquisition and scoring are all real.
# ---------------------------------------------------------------------------

#: The three likelihood-disclosure values every config carried before the `paper_neuron` profile existed, verbatim. `dev` and
#: `paper` must keep them byte for byte - that is what "the omitted option changes nothing"
#: means here - so they are pinned here rather than recomputed from the helper.
PRE_N2_FEATURE_CONFIG = {
    "noise_prior": "sigma HalfNormal(1), fixed unit 20 mV",
    "likelihood": "independent Gaussian on three correlated physical features; working/composite score, not voltage density",
    "training_features": ["test_window_mean_mV", "minimum_mV", "final_mV"],
}
# The `label_policy_profile` / `label_policy_profile_reason` pair is no longer recorded: the
# policy registers `paper_neuron` under its own name now, so `profile` is the whole record.
N2_CONFIG_KEYS = frozenset({
    "profile_family", "kernel", "feature_space", "feature_scales_mV_or_count",
})


def test_n2_profile_is_a_third_choice_with_its_own_particle_count():
    """Check that n2 profile is a third choice with its own particle count."""
    assert nc.PARTICLES_BY_PROFILE == {"dev": 16, "paper": 500, "paper_neuron": 200}
    assert nc.resolve_particles("paper_neuron", None) == 200
    assert nc.resolve_particles("paper_neuron", 200) == 200
    with pytest.raises(ValueError, match="inconsistent"):
        nc.resolve_particles("paper_neuron", 500)
    args = nc.build_parser().parse_args(["--output", "/tmp/unused", "--profile", "paper_neuron"])
    assert args.profile == "paper_neuron" and args.particles is None
    assert nc.build_parser().parse_args(["--output", "/tmp/unused"]).profile == "dev"
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "/tmp/unused", "--profile", "paper_neurons"])


def test_n2_dev_and_paper_keep_their_exact_likelihood_disclosure():
    """Check that n2 dev and paper keep their exact likelihood disclosure."""
    for profile in ("dev", "paper"):
        assert nc.feature_space_config(profile) == PRE_N2_FEATURE_CONFIG
        assert nc.feature_space(profile)["forward"] is nc.predict_features
        assert nc.feature_space(profile)["scales"] is None
        assert nc.feature_space(profile)["record_key"] == "features_mV"
        assert nc.LABEL_POLICY_PROFILE[profile] == profile
    space = nc.feature_space("paper_neuron")
    assert space["forward"] is nc.predict_eval_features
    assert space["scales"] == [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]
    assert space["record_key"] == "features"
    assert nc.feature_space_config("paper_neuron")["training_features"] == [
        "n_test", "n_pre", "rundown", "adaptation", "V_min_mV", "V_end_mV",
    ]
    extra = nc.paper_neuron_config()
    assert set(extra) == N2_CONFIG_KEYS
    assert extra["profile_family"] == "mda_comparison"
    assert extra["kernel"] == {"name": "mh_fixed_steps", "steps_per_stage": 3,
                               "ess_threshold": 0.5,
                               "proposal": "pymc_MH_random_walk_mvn_scaled"}
    assert extra["feature_space"] == "eval_features_6d"
    assert extra["feature_scales_mV_or_count"] == [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]
    assert "label_policy_profile" not in extra and "label_policy_profile_reason" not in extra
    assert nc.LABEL_POLICY_PROFILE["paper_neuron"] == "paper_neuron"


def test_n2_points_row_label_is_development_for_the_new_profile():
    """Check that n2 points row label is development for the new profile."""
    config = _fake_config(profile="paper_neuron", requested_proposer="mock")
    row = nc.points_row(config, "c" * 64, _fake_row(2, 0, "initial", 1.0), 1)
    assert row["label"] == "development"
    with pytest.raises(ValueError, match="unknown profile"):
        nc.points_row(_fake_config(profile="paper_neutron"), "c" * 64,
                      _fake_row(2, 0, "initial", 1.0), 1)


class ReachedValidatedOutput(Exception):
    """Stop the run where the label is already resolved and no output exists yet."""


def paper_neuron_args(out, **overrides):
    """Build the CLI argument list for a paper_neuron run with optional overrides."""
    flags = ["--output", str(out), "--profile", "paper_neuron", "--world", "h_sag",
             "--cell", "110", "--rounds", "1", "--proposer", "openai",
             "--llm-model", "gpt-4o", "--allow-paid-api", "--run-label", "production"]
    args = nc.build_parser().parse_args(flags)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def stop_before_output(monkeypatch, out):
    """Patch directory creation so a run stops before writing any output."""
    original = Path.mkdir

    def guard(path, *a, **kw):
        """Raise when the output directory would be created."""
        if path == out:
            raise ReachedValidatedOutput
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", guard)


@pytest.mark.parametrize("registered", [True, False])
def test_p1_paper_neuron_production_needs_a_matching_registration(
    tmp_path, monkeypatch, registered,
):
    """The blanket refusal of `--profile paper_neuron --run-label production` is flipped.

    It used to exit 2 because the policy knew two profiles. The policy now registers the
    third, so the request reaches output creation when the manifest registers this run -
    and is still refused when it does not.
    """
    pytest.importorskip("neuronbench")
    from .run_label_policy import bmd_runtime_settings
    from .test_run_label_policy import registration
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "paper_neuron_production"
    args = paper_neuron_args(out)
    args.particles = nc.resolve_particles(args.profile, args.particles)
    bound = nc.ACTIVE_MAXIMUM_BY_PROFILE[nc.PAPER_NEURON]
    settings = bmd_runtime_settings(
        "neuron", argparse.Namespace(**{**vars(args), "active_maximum": bound}))
    if not registered:
        settings["active_maximum"] = bound + 1
    ref, _, _ = registration(tmp_path, profile="paper_neuron", bundle="neuron",
                             task=args.world, seed=args.seed, cell=args.cell,
                             settings=settings)
    args = paper_neuron_args(out, manifest_ref=json.dumps(ref))
    if registered:
        # The label this registration actually yields, resolved on its own: production.
        args.particles = nc.resolve_particles(args.profile, args.particles)
        receipt = nc.bmd_label_receipt(
            "neuron",
            argparse.Namespace(**{**vars(args), "active_maximum": bound}),
            task=args.world, metadata={})
        assert receipt["run_label"] == "production"
        assert receipt["binding"]["profile"] == "paper_neuron"
        assert receipt["binding"]["settings"]["active_maximum"] == bound
        args = paper_neuron_args(out, manifest_ref=json.dumps(ref))
    stop_before_output(monkeypatch, out)
    with pytest.raises(ReachedValidatedOutput if registered else ValueError):
        nc.run(args)
    assert not out.exists()


def test_p1_paper_neuron_production_without_a_manifest_is_still_refused(tmp_path, monkeypatch):
    """Check that p1 paper neuron production without a manifest is still refused."""
    pytest.importorskip("neuronbench")
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "unregistered_paper_neuron"
    stop_before_output(monkeypatch, out)
    with pytest.raises(ValueError, match="production requires"):
        nc.run(paper_neuron_args(out))
    assert not out.exists()


def test_p1_paper_neuron_gate_function_is_gone():
    """Check that p1 paper neuron gate function is gone."""
    assert not hasattr(nc, "paper_neuron_gates")


#: The `paper_neuron` observation feature space, as the packet and the transport record it.
EVAL_FEATURE_NAMES_6 = ["n_test", "n_pre", "rundown", "adaptation", "V_min_mV", "V_end_mV"]
EVAL_FEATURE_SCALES_6 = [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_n2c_paper_neuron_proposes_on_its_own_six_evaluation_features(
    tmp_path, transport_world, provider,
):
    """An LLM proposer used to be refused for this profile (exit 2) because the belief packet
    carried the three physical features; the packet, its observation check and
    the finiteness probe now follow the profile's own feature space, so the run proceeds and its
    six evaluation features are what the proposal round is conducted on."""
    out = tmp_path / f"paper_neuron_{provider}"
    client = _TransportClient(provider, [_transport_reply()])
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--profile", "paper_neuron", "--cell", "100",
         "--proposer", provider, "--allow-paid-api", "--rounds", "1",
         "--llm-model", "synthetic_requested_model"]
    )
    nc.run(args, client=client)
    config = json.loads((out / "config.json").read_text())
    assert config["profile"] == "paper_neuron" and config["cell"] == "110"
    assert config["proposer_feature_space"] == {
        "record_key": "features", "names": EVAL_FEATURE_NAMES_6, "scales": EVAL_FEATURE_SCALES_6,
    }
    # A proposal round really happened, on this profile's features.
    rounds = [json.loads(line) for line in (out / "proposals.jsonl").read_text().splitlines()]
    assert len(rounds) == 1 and client.calls
    identity = json.loads(
        (out / "llm_transport" / "round_01.request.json").read_text())["identity"]
    assert identity["feature_space"] == {"record_key": "features", "names": EVAL_FEATURE_NAMES_6}
    packet = _request_packet(out, 1)
    assert packet["interface"]["observable"] == EVAL_FEATURE_NAMES_6
    assert packet["interface"]["feature_scales"] == EVAL_FEATURE_SCALES_6
    assert "noise_unit_mV" not in packet["interface"]
    assert packet["observations"] and all(len(row["features"]) == 6
                                          for row in packet["observations"])
    assert json.loads((out / "summary.json").read_text())["status"].startswith("completed")


@pytest.fixture(scope="module")
def n2_profile_runs(tmp_path_factory):
    """One `dev` and one `paper_neuron` run at the same world/cell/seed, fast-`fit` faked."""
    pytest.importorskip("neuronbench")
    base = tmp_path_factory.mktemp("n2_profiles")
    runs = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nc, "fit", _record_fit_call)
        for profile in ("dev", "paper_neuron"):
            _FIT_CALLS.clear()
            out = base / f"{profile}_run"
            nc.run(nc.build_parser().parse_args(
                ["--output", str(out), "--world", "h_sag", "--cell", "000", "--seed", "11",
                 "--rounds", "1", "--profile", profile]
            ))
            runs[profile] = (out, list(_FIT_CALLS))
    return runs


_FIT_CALLS = []


def _record_fit_call(hypothesis, designs, observations, forward, seed, settings=None, **kwargs):
    """The fast 2-particle fake, plus a record of exactly how the runner called `fit`."""
    _FIT_CALLS.append({"forward": forward, "settings": settings, "observations": observations,
                       **kwargs})
    return _fake_smc_result(hypothesis)


def test_n2_paper_neuron_run_records_its_kernel_feature_space_and_development_label(
    n2_profile_runs,
):
    """Check that n2 paper neuron run records its kernel feature space and development label."""
    out, calls = n2_profile_runs["paper_neuron"]
    config = json.loads((out / "config.json").read_text())
    assert config["profile"] == "paper_neuron"
    assert config["particles"] == 200
    assert config["smc"] == {"particles": 200, "chains": 2, "cores": 2}
    assert config["profile_family"] == "mda_comparison"
    assert config["kernel"] == {"name": "mh_fixed_steps", "steps_per_stage": 3,
                                "ess_threshold": 0.5,
                                "proposal": "pymc_MH_random_walk_mvn_scaled"}
    assert config["feature_space"] == "eval_features_6d"
    assert config["feature_scales_mV_or_count"] == [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]
    assert config["training_features"] == [
        "n_test", "n_pre", "rundown", "adaptation", "V_min_mV", "V_end_mV",
    ]
    assert "six" in config["likelihood"] and "20 mV" not in config["noise_prior"]
    assert "label_policy_profile" not in config

    # The kernel and the noise really reached `fit`, rather than only the config.
    assert calls and len(calls) == 8  # four hypotheses, two rounds
    for call in calls:
        assert call["kernel_policy"] == "mh_fixed_steps"
        assert call["feature_scales"] == [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]
        assert call["forward"] is nc.predict_eval_features
        assert call["settings"] == nc.SMCSettings(200, 2, 2)
        assert np.asarray(call["observations"]).shape[1] == 6

    observations = [json.loads(line)
                    for line in (out / "observations.jsonl").read_text().splitlines()]
    assert observations
    for record in observations:
        assert len(record["features"]) == 6
        assert "features_mV" not in record

    # Development, and every points.csv column exactly as the contract has them.
    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
        stream.seek(0)
        header = next(csv.reader(stream))
    assert header == list(nc.POINTS_COLUMNS)
    assert rows and {row["label"] for row in rows} == {"development"}
    assert {row["method"] for row in rows} == {"000"}
    assert {row["variant"] for row in rows} == {""}
    assert "label" not in config  # no --run-label was stated, so none is recorded


def test_n2_the_dev_profile_is_untouched_beside_it(n2_profile_runs):
    """Check that n2 the dev profile is untouched beside it."""
    out, calls = n2_profile_runs["dev"]
    config = json.loads((out / "config.json").read_text())
    # The key set from before the cell aliases, with none of the `paper_neuron` keys: an omitted
    # `--profile paper_neuron` leaves config.json exactly as it was (the source digests aside,
    # which every change moves).
    assert set(config) == PRE_N1_CONFIG_KEYS
    assert not N2_CONFIG_KEYS & set(config)
    assert "EPIG_predictive_samples" not in config and "epig_targets" not in config
    for key, value in PRE_N2_FEATURE_CONFIG.items():
        assert config[key] == value
    assert config["particles"] == 16 and config["smc"]["particles"] == 16
    for call in calls:
        assert call["forward"] is nc.predict_features
        assert "kernel_policy" not in call and "feature_scales" not in call
        assert np.asarray(call["observations"]).shape[1] == 3
    observations = [json.loads(line)
                    for line in (out / "observations.jsonl").read_text().splitlines()]
    assert observations and all(len(row["features_mV"]) == 3 for row in observations)
    assert all("features" not in row for row in observations)


def test_n2_epig_cell_records_its_targets_and_rows(tmp_path, monkeypatch):
    """Cell 002 on the default (three-feature) profile: the acquisition really runs."""
    pytest.importorskip("neuronbench")
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    out = tmp_path / "epig_run"
    nc.run(nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", "002", "--seed", "11",
         "--rounds", "1", "--profile", "dev"]
    ))
    config = json.loads((out / "config.json").read_text())
    assert config["acquisition"] == "epig"
    assert config["EPIG_predictive_samples"] == 128
    assert config["epig_targets"] == "world.test_protocols (public evaluation input measure)"
    assert config["EIG_predictive_samples"] == 64  # unchanged beside it

    records = [json.loads(line)
               for line in (out / "acquisition.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["mode"] == "epig"
    assert len(record["rows"]) == len(record["available"])
    assert record["selected"] == record["available"][
        max(range(len(record["rows"])),
            key=lambda i: (record["rows"][i]["epig"], -record["rows"][i]["design_index"]))
    ]
    for index, row in enumerate(record["rows"]):
        assert row["policy"] == "epig"
        assert row["design_index"] == index
        assert row["samples"] == 128
        assert row["targets"] == len(config["q_protocol_labels"]) == 6
        assert len(row["per_target"]) == 6
        assert np.isfinite(row["epig"]) and row["mc_standard_error"] >= 0


# ---------------------------------------------------------------------------
# The active-pool bound is the profile's, and the batch forward is
# admissible under an explicit production label once a registration binds it.
# ---------------------------------------------------------------------------
def _evidence(model_id):
    """A distinct, deterministic log evidence per model, so the retirement order is exact."""
    return -sum(ord(character) for character in model_id) / 1000.0


def _ranked_fake_fit(hypothesis, *args, **kwargs):
    result = _fake_smc_result(hypothesis)
    result.log_evidence = _evidence(hypothesis.model_id)
    result.per_chain_log_evidence = [result.log_evidence, result.log_evidence]
    return result


def _pool_run(tmp_path, monkeypatch, profile, name):
    """One mock-proposal round on a wired world, with the fits faked but ranked."""
    pytest.importorskip("neuronbench")
    monkeypatch.setattr(nc, "fit", _ranked_fake_fit)
    out = tmp_path / name
    args = nc.build_parser().parse_args(
        ["--output", str(out), "--world", "h_sag", "--cell", "100", "--seed", "11",
         "--rounds", "1", "--profile", profile, "--proposer", "mock"]
    )
    nc.run(args)
    config = json.loads((out / "config.json").read_text())
    rounds = [json.loads(line) for line in (out / "proposals.jsonl").read_text().splitlines()]
    return config, rounds


def test_n3_active_maximum_is_the_profiles_own_bound():
    """Check that n3 active maximum is the profiles own bound."""
    assert nc.ACTIVE_MAXIMUM_BY_PROFILE == {"dev": 12, "paper": 12, "paper_neuron": 6}
    # dev and paper keep core_count's value itself, not a copy that can drift.
    assert nc.ACTIVE_MAXIMUM_BY_PROFILE["dev"] is nc.ACTIVE_MAXIMUM
    assert nc.ACTIVE_MAXIMUM_BY_PROFILE["paper"] is nc.ACTIVE_MAXIMUM


def test_n3_paper_neuron_retires_the_seventh_candidate_by_evidence(tmp_path, monkeypatch):
    """Check that n3 paper neuron retires the seventh candidate by evidence."""
    config, rounds = _pool_run(tmp_path, monkeypatch, "paper_neuron", "paper_neuron_pool")
    assert config["active_maximum"] == 6
    assert len(rounds) == 1
    admitted, retired = rounds[0]["admitted_ids"], rounds[0]["retired_ids"]
    # Four initial hypotheses plus the three the mock proposal event offers.
    pool = sorted(path.stem.removeprefix("t01_")
                  for path in (tmp_path / "paper_neuron_pool" / "fits").glob("t01_*.json"))
    assert len(pool) == 7
    ordered = sorted(pool, key=lambda model_id: (-_evidence(model_id), model_id))
    assert retired == ordered[6:]
    assert len(retired) == 1
    assert set(admitted) & set(retired) == set()


@pytest.mark.parametrize("profile", ["dev", "paper"])
def test_n3_dev_and_paper_keep_the_twelve_model_pool(tmp_path, monkeypatch, profile):
    """Check that n3 dev and paper keep the twelve model pool."""
    config, rounds = _pool_run(tmp_path, monkeypatch, profile, f"{profile}_pool")
    assert config["active_maximum"] == 12
    assert rounds[0]["retired_ids"] == []


def test_n3_batch_with_run_label_production_is_admitted_by_a_matching_registration(
    tmp_path, monkeypatch,
):
    """The blanket refusal of batch under an explicit production label is flipped.

    It exited 2 for `--forward-backend batch --run-label production` because nothing
    registered the executed forward. The label policy registers `forward_backend` and the
    equivalence report's digest, so the request now reaches output creation when the
    manifest registers them - and the report checks themselves are unchanged.
    """
    pytest.importorskip("neuronbench")
    from .run_label_policy import bmd_runtime_settings
    from .test_run_label_policy import registration
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    monkeypatch.setattr(nc, "fit", lambda h, *a, **k: _fake_smc_result(h))
    report = _write_report(tmp_path)
    out = tmp_path / "production_batch"
    options = {"forward_backend": "batch", "forward_equivalence_report": str(report)}
    args = paper_neuron_args(out, **options)
    args.particles = nc.resolve_particles(args.profile, args.particles)
    settings = bmd_runtime_settings("neuron", argparse.Namespace(**{
        **vars(args), "active_maximum": nc.ACTIVE_MAXIMUM_BY_PROFILE[nc.PAPER_NEURON],
        "forward_equivalence_sha": sha256_file(report)}))
    assert settings["forward_backend"] == "batch"
    assert is_hex64(settings["forward_equivalence_sha"])
    ref, _, _ = registration(tmp_path, profile="paper_neuron", bundle="neuron",
                             task="h_sag", seed=11, cell="110", settings=settings)
    args = paper_neuron_args(out, manifest_ref=json.dumps(ref), **options)
    stop_before_output(monkeypatch, out)
    with pytest.raises(ReachedValidatedOutput):
        nc.run(args)
    assert not out.exists()


def test_n3_batch_production_is_still_refused_when_the_report_disagrees(tmp_path, monkeypatch):
    """Every other batch refusal stands; only the production/batch combination was lifted."""
    monkeypatch.setattr(nc, "power_matches_scalar", lambda: True)
    out = tmp_path / "production_batch_bad_report"
    report = _write_report(tmp_path, "counts.json", spike_counts_equal=False)
    with pytest.raises(SystemExit) as refusal:
        nc.run(paper_neuron_args(out, forward_backend="batch",
                                 forward_equivalence_report=str(report)))
    assert refusal.value.code == 2
    assert not out.exists()
