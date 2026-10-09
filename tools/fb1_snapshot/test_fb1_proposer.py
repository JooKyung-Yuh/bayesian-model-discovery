"""Catalogue-mode proposer tests: catalogue contract, belief packet (ForceBench edition), parser/judge, transport (zero paid calls).

Run: <jax venv python> -m pytest -q test_fb1_proposer.py
"""

import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# This file tests the modules that sit next to it: the path is the test file's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fb1_proposer as fp  # noqa: E402
from fb1_core import STRUCTURES  # noqa: E402

# the engine's own packaged copy of the v2.0 prompt (the same bytes, sha256 f095edf1...)
E16_PROMPT = (Path(__file__).resolve().parents[2]
              / "src/bayes_model_discovery/experimental/prompts/proposer_system_v2_0.txt")
RUNNER_V3_LIST = [
    ("power", "gauss"), ("inv_r_plus_c", "gauss"), ("inv_r2", "student_t4"),
    ("inv_r", "gauss"), ("inv_r_bg", "gauss"), ("screened", "student_t4"),
    ("inv_r_p1sq", "gauss"), ("inv_r_nomass", "gauss"), ("inv_r_tmod", "gauss"),
    ("prod_coupling", "gauss"), ("crossover", "gauss"), ("inv_r_bg", "student_t4"),
]


# ------------------------------------------------------------------ prompt v2.0
def test_prompt_file_is_the_frozen_v2_0_copy_byte_for_byte():
    """Test: prompt file is the frozen v2 0 copy byte for byte."""
    raw = fp.system_prompt_bytes()
    assert hashlib.sha256(raw).hexdigest() == fp.PROMPT_SHA256
    with open(E16_PROMPT, "rb") as fh:
        assert raw == fh.read()
    assert fp.prompt_identity() == {
        "prompt_version": "v2.0",
        "prompt_sha256": fp.PROMPT_SHA256,
    }


def test_prompt_text_still_states_the_S0_rule_the_catalog_contract_relies_on():
    """Test: prompt text still states the S0 rule the catalog contract relies on."""
    text = fp.system_prompt_bytes().decode()
    assert "structure_expansion is false" in text
    assert "verbatim one of constraints.initial_structures" in text
    assert "family must be one of constraints.allowed_families" in text


# --------------------------------------------------------------------- catalogue
def test_catalog_covers_every_structure_and_is_one_to_one():
    """Test: catalog covers every structure and is one to one."""
    table = fp.catalog_table()
    assert [row["structure"] for row in table] == list(STRUCTURES)
    expressions = [row["expression"] for row in table]
    assert len(set(expressions)) == len(expressions) == len(STRUCTURES)
    back = fp.expression_to_structure()
    assert {back[e] for e in expressions} == set(STRUCTURES)
    for row in table:
        assert "=" not in row["expression"] and row["expression"] == row["expression"].strip()


def test_coefficient_names_match_the_catalog():
    """Test: coefficient names match the catalog."""
    for name, structure in STRUCTURES.items():
        names = fp.COEFFICIENT_NAMES[name]
        assert len(names) == structure.n_coeff == len(structure.kinds)
        assert len(set(names)) == len(names)
        expression = fp.canonical_expression(name)
        for coefficient in names:
            assert coefficient in expression, (name, coefficient)
        for kind in structure.kinds:
            assert kind in fp.KIND_ROLE and kind in fp.KIND_PRIOR


def test_constraints_declare_catalog_selection_and_generate_priors_from_the_core():
    """Test: constraints declare catalog selection and generate priors from the core."""
    constraints = fp.catalog_constraints()
    assert set(constraints) == fp.CONSTRAINT_KEYS
    assert constraints["structure_expansion"] is False
    assert constraints["family_expansion"] is True
    assert constraints["allowed_families"] == ["gauss", "student_t4"]
    assert len(constraints["initial_structures"]) == len(STRUCTURES)
    assert constraints["roles"] == {"positive": "LogNormal(0,1)", "signed": "Normal(0,1)"}
    import fb1_core

    assert f"{fb1_core.B0_BACKGROUND:.6g}" in fp.KIND_PRIOR["pos_bg"]
    assert f"{fb1_core.W0_MODULATION:.6g}" in fp.KIND_PRIOR["pos_w"]
    for expression, spec in constraints["structure_parameters"].items():
        name = fp.expression_to_structure()[expression]
        assert list(spec) == list(fp.COEFFICIENT_NAMES[name])


def test_family_only_expansion_locks_the_family_when_asked():
    """Test: family only expansion locks the family when asked."""
    constraints = fp.catalog_constraints("gauss", family_expansion=False)
    assert constraints["allowed_families"] == ["gauss"]
    ok, reason = fp.judge_candidate(
        {"expression": "c*p1/r^2", "family": "student_t4", "parameters": {"c": "positive"}},
        constraints,
    )
    assert not ok and "initial_family" in reason


def test_model_id_is_one_to_one_with_structure_and_family():
    """Test: model id is one to one with structure and family."""
    seen = {}
    for name in STRUCTURES:
        expression = fp.canonical_expression(name)
        parameters = {
            c: fp.KIND_ROLE[k] for c, k in zip(fp.COEFFICIENT_NAMES[name], STRUCTURES[name].kinds)
        }
        for family in ("gauss", "student_t4"):
            mid = fp.model_id(expression, family, parameters)
            assert mid not in seen
            seen[mid] = (name, family)
    assert len(seen) == 2 * len(STRUCTURES)


def test_mock_schedule_is_predeclared_and_starts_with_the_runner_v3_list():
    """Test: mock schedule is predeclared and starts with the runner v3 list."""
    assert fp.MOCK_SCHEDULE[:12] == RUNNER_V3_LIST
    assert len(fp.MOCK_SCHEDULE) >= 16
    constraints = fp.catalog_constraints()
    for index in range(len(fp.MOCK_SCHEDULE)):
        candidate, ignored = fp.parse_candidate_object(fp.mock_response_text(index), constraints)
        assert (candidate["structure"], candidate["family"]) == fp.MOCK_SCHEDULE[index]
        assert ignored["note"] == fp.MOCK_NOTE


def test_mock_schedule_matches_the_runners_own_v3_list():
    """Test: mock schedule matches the runners own v3 list."""
    import os

    os.environ.setdefault("FB1_WORLD", "gravity")
    import run_gravity

    assert [tuple(p) for row in run_gravity.MOCK_PROPOSALS_V3 for p in row] == RUNNER_V3_LIST


# ------------------------------------------------------------------- belief packet
def _member(structure="inv_r2", family="gauss", weight=1.0, predictions=None, **kw):
    names = fp.COEFFICIENT_NAMES[structure]
    return fp.PoolMember(
        structure, family, weight, -3.0,
        coefficients={n: {"mean": 1.0 + i, "sd": 0.25} for i, n in enumerate(names)},
        family_dispersion={"sigma": {"mean": 0.1, "sd": 0.01}},
        predictions=predictions, **kw,
    )


def _state(n_designs=3, n_times=10, y_offset=0.0):
    rng = np.random.default_rng(7)
    pred_mean = rng.normal(size=(n_designs, n_times, 2)) * 3.0
    pred_sd = np.full((n_designs, n_times, 2), 0.5)
    accel = np.abs(rng.normal(size=(n_designs, n_times))) + 0.1
    rows = [
        {"design_id": f"d{j:02d}", "p1": 1.0 + j, "p2": 2.0 + j,
         "initial_state": {"pos": [-2.0, -6.0], "vel": [-0.1, 1.9]},
         "integration_steps": 250, "integration_status": "ok",
         "y": pred_mean[j] + y_offset}
        for j in range(n_designs)
    ]
    predictions = {"pred_mean": pred_mean, "pred_sd": pred_sd, "accel_abs": accel}
    members = [
        _member("inv_r2", "gauss", 0.7, predictions),
        _member("screened", "gauss", 0.3),
    ]
    interface = fp.forcebench_interface(
        measurement_times=np.linspace(0.5, 5.0, n_times), dt=0.02,
        p1_range=(0.1, 10.0), p2_range=(0.1, 10.0), integration_steps=250)
    return rows, members, fp.catalog_constraints(), interface


def test_builder_signature_has_no_evaluation_arguments():
    """Test: builder signature has no evaluation arguments."""
    names = list(inspect.signature(fp.build_belief_packet).parameters)
    assert names == [
        "public_observations", "active_fits", "archive", "constraints", "interface", "round"
    ]
    forbidden = ("heldout", "held_out", "evaluation", "truth", "score", "nll", "pmf", "q")
    assert not [n for n in names for f in forbidden if f in n.lower()]
    body = inspect.getsource(fp.build_belief_packet).replace(fp.build_belief_packet.__doc__, "")
    for word in ("per_model_nll", "heldout", "evaluation", "Q_HEAD", "Q_SHIFT", "coverage90"):
        assert word not in body


def test_runner_proposal_helpers_take_no_evaluation_arguments():
    """Test: runner proposal helpers take no evaluation arguments."""
    import os

    os.environ.setdefault("FB1_WORLD", "gravity")
    import run_gravity

    forbidden = ("heldout", "held_out", "evaluation", "truth", "score", "nll", "pmf")
    for function in (run_gravity._llm_propose, run_gravity._pool_members,
                     run_gravity._observation_rows, run_gravity._public_predictions):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names for f in forbidden if f in n.lower()], (function, names)
        body = inspect.getsource(function)
        for word in ("TR_H", "TR_S", "YH_H", "YH_S", "Q_HEAD", "Q_SHIFT", "score_set", "VAR_GT"):
            assert word not in body, (function.__name__, word)


def test_packet_is_byte_identical_when_only_the_heldout_set_changes():
    """Test: packet is byte identical when only the heldout set changes."""
    rows, members, constraints, interface = _state()
    # The evaluation set is **not an input** to the builder: changing it here must leave the packet bytes unchanged.
    heldout_a = {"Q": [(1.0, 1.0)], "y": np.zeros((1, 10, 2))}
    heldout_b = {"Q": [(9.9, 9.9)], "y": np.ones((1, 10, 2)) * 1234.5}
    first = fp.serialize_packet(fp.build_belief_packet(rows, members, [], constraints, interface, 3))
    _ = (heldout_a, heldout_b)
    second = fp.serialize_packet(fp.build_belief_packet(rows, members, [], constraints, interface, 3))
    assert first == second
    assert fp.packet_digest(first) == fp.packet_digest(second)


def test_packet_carries_no_raw_trajectory_and_stays_within_the_design_cap():
    """Test: packet carries no raw trajectory and stays within the design cap."""
    rows, members, constraints, interface = _state()
    packet = fp.build_belief_packet(rows, members, [], constraints, interface, 3)
    assert set(packet) == {
        "packet_version", "interface", "round", "n_public", "observations", "pool",
        "best_model", "best_model_id", "best_model_residuals",
        "best_model_residual_definitions", "archive", "constraints",
    }
    for row in packet["observations"]:
        assert set(row) == {"design_id", "p1", "p2", "initial_state", "integration_steps",
                            "integration_status"}
        assert "y" not in row
    for row in packet["best_model_residuals"]:
        assert len(row["residual_trajectory_8pt"]) == fp.RESIDUAL_DOWNSAMPLE_POINTS

    def longest_numeric(node):
        """Length of the longest **numeric** array; a list of strings (the 14 catalogue expressions) is not a trajectory."""
        if isinstance(node, dict):
            return max((longest_numeric(v) for v in node.values()), default=0)
        if isinstance(node, list):
            inner = max((longest_numeric(v) for v in node), default=0)
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node)
            return max(len(node), inner) if node and numeric else inner
        return 0

    # Apart from interface.time_grid (the public measurement-time grid, a task specification) no numeric array is longer than 8 points,
    # i.e. a measured trajectory of length 10 is never carried whole.
    stripped = {k: v for k, v in packet.items() if k != "interface"}
    assert longest_numeric(stripped) <= fp.RESIDUAL_DOWNSAMPLE_POINTS
    assert longest_numeric(
        {k: v for k, v in packet["interface"].items() if k != "time_grid"}
    ) <= 2
    text = fp.serialize_packet(packet)
    for word in ("heldout", "evaluation", "truth", "coverage90", "nmse", "observation_seed"):
        assert word not in text

    too_many = rows * ((fp.MAX_DESIGNS // len(rows)) + 1)
    with pytest.raises(ValueError, match="at most 60 designs"):
        fp.build_belief_packet(too_many, members, [], constraints, interface, 3)


def test_interface_declares_trajectory_summary_and_defines_every_summary_field():
    """Test: interface declares trajectory summary and defines every summary field."""
    rows, members, constraints, interface = _state()
    packet = fp.build_belief_packet(rows, members, [], constraints, interface, 3)
    assert tuple(sorted(packet["interface"])) == tuple(sorted(fp.INTERFACE_KEYS))
    assert packet["interface"]["outcome_type"] == "trajectory_summary"
    assert packet["interface"]["link"] == "identity"
    assert [i["name"] for i in packet["interface"]["inputs"]] == ["p1", "p2"]
    assert len(packet["interface"]["time_grid"]) == 10
    definitions = packet["interface"]["summary_definitions"]
    assert definitions["best_model_residuals"] == packet["best_model_residual_definitions"]
    emitted_observation_fields = set(packet["observations"][0])
    assert emitted_observation_fields <= set(definitions["observations"])
    residual_fields = set(packet["best_model_residuals"][0]) - {"design_id", "p1", "p2"}
    assert residual_fields <= set(definitions["best_model_residuals"])
    assert all(isinstance(v, str) and v for v in definitions["observations"].values())


def test_residual_summary_matches_its_own_definitions():
    """Test: residual summary matches its own definitions."""
    rows, members, constraints, interface = _state(y_offset=1.0)
    packet = fp.build_belief_packet(rows, members, [], constraints, interface, 3)
    row = packet["best_model_residuals"][0]
    # z = (y - pred_mean)/pred_sd = 1.0/0.5 = 2 exactly -> RMS = max|z| = 2, and every point of the 8-point trajectory is 2 too
    assert row["rms_standardized_residual"] == pytest.approx(2.0, rel=1e-3)
    assert row["max_abs_standardized_residual"] == pytest.approx(2.0, rel=1e-3)
    assert all(v == pytest.approx(2.0, rel=1e-3) for v in row["residual_trajectory_8pt"])
    predictions = members[0].predictions
    assert row["min_r"] == pytest.approx(
        float(np.linalg.norm(predictions["pred_mean"][0], axis=-1).min()), rel=1e-3)
    assert row["max_abs_accel"] == pytest.approx(
        float(np.abs(predictions["accel_abs"][0]).max()), rel=1e-3)


def test_pool_is_ordered_by_weight_and_one_step_history_keeps_nulls():
    """Test: pool is ordered by weight and one step history keeps nulls."""
    rows, members, constraints, interface = _state()
    members[0].one_step_log_scores = [None, -1.5, -1.25]
    members[0].one_step_rounds = [1, 2, 3]
    members[1].one_step_log_scores = [-2.0, -2.5, None]
    members[1].one_step_rounds = [1, 2, 3]
    packet = fp.build_belief_packet(rows, members, [], constraints, interface, 3)
    assert [m["weight"] for m in packet["pool"]] == sorted(
        [m["weight"] for m in packet["pool"]], reverse=True)
    assert packet["pool"][0]["one_step_log_scores"][0] is None
    assert len(packet["pool"][0]["one_step_log_scores"]) == len(packet["pool"][0]["one_step_rounds"])
    assert "sigma" in packet["pool"][0]["family_dispersion"]
    assert "sigma" not in packet["pool"][0]["coefficients"]


def test_archive_reason_key_sets_are_enforced():
    """Test: archive reason key sets are enforced."""
    rows, members, constraints, interface = _state()
    good = [{"model_id": "a" * 20, "expression": "c*p1/r", "family": "gauss",
             "proposed_round": 1, "outcome": "duplicate",
             "reason": {"matched_model_id": "b" * 20, "matched_expression": "c*p1/r"}}]
    packet = fp.build_belief_packet(rows, members, good, constraints, interface, 3)
    assert packet["archive"][0]["outcome"] == "duplicate"
    bad = [dict(good[0], reason={"matched_model_id": "b" * 20})]
    with pytest.raises(ValueError, match="reason keys"):
        fp.build_belief_packet(rows, members, bad, constraints, interface, 3)
    unknown = [dict(good[0], outcome="promoted")]
    with pytest.raises(ValueError, match="unknown archive outcome"):
        fp.build_belief_packet(rows, members, unknown, constraints, interface, 3)


def test_serialization_is_four_significant_digits_and_rejects_nonfinite():
    """Test: serialization is four significant digits and rejects nonfinite."""
    assert fp.round_significant(1.23456789) == 1.235
    assert fp.round_significant({"a": [1e-9, 2]}) == {"a": [1e-09, 2]}
    with pytest.raises(ValueError):
        fp.round_significant(float("inf"))
    text = fp.serialize_packet({"b": 1, "a": 2.0})
    assert text == '{"a":2.0,"b":1}'


# --------------------------------------------------------- offline parser
def _object(**kw):
    return json.dumps(kw)


PARSE_CASES = [
    ("valid_initial_structure", _object(expression="c*p1/r^2", family="gauss",
                                        parameters={"c": "positive"}), None),
    ("valid_family_switch", _object(expression="c*p1/r^2", family="student_t4",
                                    parameters={"c": "positive"}), None),
    ("valid_two_coefficients", _object(expression="c*p1*exp(-r/lam)/r", family="gauss",
                                       parameters={"c": "positive", "lam": "positive"}), None),
    ("valid_with_note", _object(expression="c*p1/r", family="gauss",
                                parameters={"c": "positive"}, note="residuals grow with r"), None),
    ("s0_violation_new_expression", _object(expression="c*p1/r**3", family="gauss",
                                            parameters={"c": "positive"}),
     "verbatim one of constraints.initial_structures"),
    ("s0_violation_whitespace", _object(expression="c*p1/r^2 ", family="gauss",
                                        parameters={"c": "positive"}),
     "verbatim one of constraints.initial_structures"),
    ("wrong_family", _object(expression="c*p1/r^2", family="cauchy",
                             parameters={"c": "positive"}),
     "one of constraints.allowed_families"),
    ("malformed_json", "{'expression': 'c*p1/r^2',", "reply is not valid JSON"),
    ("prose_not_json", "I would propose the screened law with a Gaussian family.",
     "reply is not valid JSON"),
    ("array_not_object", json.dumps([{"expression": "c*p1/r^2", "family": "gauss",
                                      "parameters": {"c": "positive"}}]),
     "one JSON object, not an array"),
    ("missing_parameters", _object(expression="c*p1/r^2", family="gauss"),
     "must include expression/family/parameters"),
    ("parameters_not_a_dict", _object(expression="c*p1/r^2", family="gauss",
                                      parameters=["c"]),
     "map coefficient names to fixed roles"),
    ("wrong_role", _object(expression="c*p1/r^2", family="gauss", parameters={"c": "signed"}),
     "has the fixed role positive"),
    ("wrong_coefficient_names", _object(expression="c*p1/r^2", family="gauss",
                                        parameters={"k": "positive"}),
     "name exactly the coefficients"),
    ("extra_keys_recorded", _object(expression="c*p1/r^2", family="gauss",
                                    parameters={"c": "positive"},
                                    priors={"c": "Normal(0,5)"}, temperature=0.7), None),
    ("reserved_dispersion_ignored", _object(expression="c*p1/r^2", family="gauss",
                                            parameters={"c": "positive", "sigma": "positive"}),
     None),
    ("note_too_long", _object(expression="c*p1/r^2", family="gauss",
                              parameters={"c": "positive"}, note="x" * 201), None),
    ("note_not_a_string", _object(expression="c*p1/r^2", family="gauss",
                                  parameters={"c": "positive"}, note=17), None),
]


@pytest.mark.parametrize("name,text,expected", PARSE_CASES, ids=[c[0] for c in PARSE_CASES])
def test_offline_parse_of_synthetic_llm_responses(name, text, expected):
    """Test: offline parse of synthetic llm responses."""
    constraints = fp.catalog_constraints()
    if expected is not None:
        with pytest.raises(ValueError) as excinfo:
            fp.parse_candidate_object(text, constraints)
        assert expected in str(excinfo.value)
        return
    candidate, ignored = fp.parse_candidate_object(text, constraints)
    assert candidate["structure"] in STRUCTURES
    assert candidate["model_id"] == fp.model_id(
        candidate["expression"], candidate["family"], candidate["parameters"])
    if name == "valid_with_note":
        assert ignored["note"] == "residuals grow with r"
    if name == "extra_keys_recorded":
        assert set(ignored["ignored_keys"]) == {"priors", "temperature"}
    if name == "reserved_dispersion_ignored":
        assert ignored["ignored_parameters"] == {"sigma": "positive"}
        assert candidate["parameters"] == {"c": "positive"}
    if name in ("note_too_long", "note_not_a_string"):
        assert ignored["note"] == "" and "note" in ignored["ignored_keys"]


def test_parse_case_table_covers_the_required_shapes():
    """Test: parse case table covers the required shapes."""
    assert len(PARSE_CASES) >= 12
    names = {c[0] for c in PARSE_CASES}
    assert {"valid_initial_structure", "s0_violation_new_expression", "wrong_family",
            "malformed_json", "extra_keys_recorded", "valid_with_note"} <= names


# ------------------------------------------------------------ transport (zero paid calls)
def test_default_transport_is_mock_and_never_builds_a_live_client(tmp_path):
    """Test: default transport is mock and never builds a live client."""
    proposer = fp.Fb1CatalogProposer(tmp_path / "records")
    assert proposer.transport == "mock"
    assert isinstance(proposer.client, fp.MockCatalogClient)
    assert proposer.paid_api_calls == 0
    source = inspect.getsource(fp.Fb1CatalogProposer.__init__)
    assert "import anthropic" in source
    anthropic_line = [ln for ln in source.splitlines() if "import anthropic" in ln][0]
    assert anthropic_line.startswith(" " * 16), "live import must sit under `if allow_live`"


def _propose(tmp_path, texts=None, **kw):
    client = fp.ScriptedClient(texts, **kw) if texts is not None else None
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1))
    context = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "P1",
               "event": 0, "round": 1, "n_designs": 3, "candidate_index": 0,
               "packet_file": "t03_c0.json"}
    return proposer, proposer.propose_one(packet, context=context, constraints=constraints), packet


def test_transport_record_mirrors_the_api_v2_format(tmp_path):
    """Test: transport record mirrors the format of the engine's proposer API."""
    proposer, (candidates, record), packet = _propose(tmp_path)
    assert len(candidates) == 1
    assert record["status"] == "parsed"
    assert record["prompt_version"] == "v2.0" and record["prompt_sha256"] == fp.PROMPT_SHA256
    assert record["packet_sha256"] == hashlib.sha256(packet.encode()).hexdigest()
    assert record["packet_bytes"] == len(packet.encode())
    assert record["calls"] == 1 and record["repairs"] == 0
    assert record["paid_api_calls"] == 0 and proposer.paid_api_calls == 0
    assert record["recovery_mode"] == "catalog_selection"
    assert record["responses"][0]["raw_text"]
    assert record["rejected_candidates"] == []
    assert record["valid_candidate_count"] == 1 and record["silent_fallback"] is False
    written = sorted(p.name for p in (tmp_path / "records").iterdir())
    assert written == ["event_00_round_01_c0.request.json", "event_00_round_01_c0.response.json"]
    saved = json.loads((tmp_path / "records" / "event_00_round_01_c0.response.json").read_text())
    assert saved["packet_file"] == "t03_c0.json" and saved["candidate_index"] == 0
    assert saved["status"] == "parsed"
    body = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert body["bodies"][0]["system"] == fp.system_prompt_bytes().decode()
    assert body["bodies"][0]["messages"][0]["content"] == packet


def test_one_format_repair_then_success(tmp_path):
    """Test: one format repair then success."""
    good = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"})
    _p, (candidates, record), _ = _propose(tmp_path, ["not json at all", good])
    assert len(candidates) == 1 and record["status"] == "parsed"
    assert record["calls"] == 2 and record["repairs"] == 1
    assert len(record["rejected_candidates"]) == 1
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]


def test_two_failures_leave_proposal_failed_with_both_messages(tmp_path):
    """Test: two failures leave proposal failed with both messages."""
    bad = _object(expression="c*p1/r**7", family="gauss", parameters={"c": "positive"})
    _p, (candidates, record), _ = _propose(tmp_path, [bad, bad])
    assert candidates == [] and record["status"] == "proposal_failed"
    assert record["calls"] == 2 and record["repairs"] == 1
    assert len(record["rejected_candidates"]) == 2
    assert all("verbatim" in item["reason"] for item in record["rejected_candidates"])


def test_no_repair_beyond_one_call(tmp_path):
    """Test: no repair beyond one call."""
    bad = "nope"
    client = fp.ScriptedClient([bad, bad, bad])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1))
    proposer.propose_one(packet, context={"run_id": "t", "world": "gravity", "arm": "111",
                                          "policy": "P1", "event": 0, "round": 1,
                                          "n_designs": 3, "candidate_index": 0,
                                          "packet_file": "t03_c0.json"}, constraints=constraints)
    assert client.calls == 2


def test_truncated_response_is_not_parsed(tmp_path):
    """Test: truncated response is not parsed."""
    good = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"})
    _p, (candidates, record), _ = _propose(
        tmp_path, [good, good], stop_reasons=["max_tokens", "end_turn"])
    assert len(candidates) == 1 and record["calls"] == 2
    assert "end_turn" in record["rejected_candidates"][0]["reason"]


def test_context_identity_is_required(tmp_path):
    """Test: context identity is required."""
    proposer = fp.Fb1CatalogProposer(tmp_path / "records")
    constraints = fp.catalog_constraints()
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one("{}", context={"run_id": "t"}, constraints=constraints)


def test_records_directory_is_write_once(tmp_path):
    """Test: records directory is write once."""
    (tmp_path / "records").mkdir()
    with pytest.raises(FileExistsError):
        fp.Fb1CatalogProposer(tmp_path / "records")


def test_downsample_indices_are_evenly_spaced():
    """Test: downsample indices are evenly spaced."""
    assert fp.downsample_indices(10) == [0, 1, 3, 4, 5, 6, 8, 9]
    assert fp.downsample_indices(8) == list(range(8))
    assert len(fp.downsample_indices(3)) == 3
    assert math.isclose(len(fp.downsample_indices(64)), 8)


def test_three_calls_in_one_event_do_not_collide_on_disk(tmp_path):
    """Policy P4 makes three calls in one event; the record file names must not collide."""
    proposer = fp.Fb1CatalogProposer(tmp_path / "records")
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 2))
    got = []
    for j in range(3):
        candidates, record = proposer.propose_one(
            packet, context={"run_id": "t", "world": "gravity", "arm": "111", "policy": "P4",
                             "event": 1, "round": 2, "n_designs": 3, "candidate_index": j,
                             "packet_file": f"t05_c{j}.json"}, constraints=constraints)
        assert record["status"] == "parsed" and record["paid_api_calls"] == 0
        got.append((candidates[0]["structure"], candidates[0]["family"]))
    names = sorted(p.name for p in (tmp_path / "records").iterdir())
    assert names == [f"event_01_round_02_c{j}.{kind}.json"
                     for j in range(3) for kind in ("request", "response")]
    assert got == fp.MOCK_SCHEDULE[:3]          # the mock schedule is consumed in call order
    assert len(set(got)) == 3


# ------------------------------------------------ OpenAI transport
# A transport for tests, smoke runs and development only: zero paid calls, and a real openai.OpenAI() is
# never constructed. It goes through the same policy as the Anthropic path (P1: one call + one format repair);
# the only differences are the transport wiring (whether SYSTEM is messages[0] or a separate field,
# and the names of the usage / response-id fields).
from types import SimpleNamespace  # noqa: E402


def _openai_response(text, *, model="returned-test-model", response_id="test-only-openai-1",
                      finish_reason="stop", prompt_tokens=9, completion_tokens=3, cached_tokens=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
        id=response_id,
        model=model,
    )


class ScriptedOpenAIClient:
    """Injectable fake openai client: the openai counterpart of fp.ScriptedClient; no network, zero paid calls."""

    provider = "openai"

    def __init__(self, texts, *, model="mock-openai", finish_reasons=None):
        self.texts = list(texts)
        self.model = model
        self.finish_reasons = list(finish_reasons or [])
        self.calls = 0
        self.bodies = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **body):
        self.bodies.append(body)
        index = self.calls
        self.calls += 1
        if index >= len(self.texts):
            raise RuntimeError("scripted client ran out of responses")
        finish = self.finish_reasons[index] if index < len(self.finish_reasons) else "stop"
        return _openai_response(
            self.texts[index], model=self.model,
            response_id=f"scripted_openai_{index:03d}", finish_reason=finish,
        )


def test_openai_transport_never_builds_a_live_client_by_default(tmp_path):
    """Same structural guarantee as test_default_transport_is_mock_and_never_
    builds_a_live_client: `import openai` sits under `if allow_live`, not at
    module load or unconditionally in __init__."""
    source = inspect.getsource(fp.Fb1CatalogProposer.__init__)
    assert "import openai" in source
    openai_line = [ln for ln in source.splitlines() if "import openai" in ln][0]
    assert openai_line.startswith(" " * 16), "live import must sit under `if allow_live`"


def test_openai_provider_resolution_reads_the_env_var_with_an_anthropic_default(monkeypatch):
    """FB1_LLM_PROVIDER unset keeps the exact default ("anthropic"), read
    only once allow_live is actually requested (never for mock/injected use -
    see test_openai_provider_selected_via_env_var_and_kwarg_and_invalid_value_
    is_refused, which proves the env var/kwarg are actually consulted,
    without ever constructing a real SDK client - that stays forbidden in
    every test in this file)."""
    monkeypatch.delenv("FB1_LLM_PROVIDER", raising=False)
    source = inspect.getsource(fp.Fb1CatalogProposer.__init__)
    assert 'os.environ.get("FB1_LLM_PROVIDER", "anthropic")' in source


def test_openai_provider_selected_via_env_var_and_kwarg_and_invalid_value_is_refused(
    tmp_path, monkeypatch
):
    """`provider=` overrides FB1_LLM_PROVIDER; an unsupported value refuses
    before any live SDK import or output directory is created - same
    before-output refusal shape as the existing allow_live=False path."""
    monkeypatch.delenv("FB1_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="unsupported FB1_LLM_PROVIDER"):
        fp.Fb1CatalogProposer(tmp_path / "bad", allow_live=True, provider="not-a-provider")
    assert not (tmp_path / "bad").exists()
    monkeypatch.setenv("FB1_LLM_PROVIDER", "also-not-a-provider")
    with pytest.raises(ValueError, match="unsupported FB1_LLM_PROVIDER"):
        fp.Fb1CatalogProposer(tmp_path / "bad2", allow_live=True)
    assert not (tmp_path / "bad2").exists()


def _propose_openai(tmp_path, texts=None, **kw):
    client = ScriptedOpenAIClient(texts if texts is not None else [fp.mock_response_text(0)], **kw)
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", model="gpt-4o", client=client)
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1))
    context = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "P1",
               "event": 0, "round": 1, "n_designs": 3, "candidate_index": 0,
               "packet_file": "t03_c0.json"}
    return proposer, proposer.propose_one(packet, context=context, constraints=constraints), packet


def test_openai_transport_record_mirrors_the_api_v2_format(tmp_path):
    """Same contract as test_transport_record_mirrors_the_api_v2_format for
    Anthropic, adjusted for the OpenAI wire shape (the exact usage
    fields) - SYSTEM is messages[0] (role "system"), not a separate field."""
    proposer, (candidates, record), packet = _propose_openai(tmp_path)
    assert proposer.transport == "openai"
    assert len(candidates) == 1
    assert record["status"] == "parsed"
    assert record["provider"] == "openai" and record["requested_model"] == "gpt-4o"
    assert record["prompt_version"] == "v2.0" and record["prompt_sha256"] == fp.PROMPT_SHA256
    assert record["packet_sha256"] == hashlib.sha256(packet.encode()).hexdigest()
    assert record["calls"] == 1 and record["repairs"] == 0
    assert record["paid_api_calls"] == 1 and proposer.paid_api_calls == 1
    # The old top-level `usage` (last call only) is now
    # `last_call_usage`, `usage_total` is the sum over responses[*], and the
    # old `request_id` (which held the completion id) is `completion_id`.
    last_call = {"input_tokens": 9, "output_tokens": 3, "cached_tokens": 0}
    assert record["last_call_usage"] == last_call and record["last_call_usage_known"]
    assert record["usage_total"] == last_call and record["usage_total_complete"]
    assert record["usage_missing_indices"] == []
    assert record["completion_id"] == "scripted_openai_000"
    assert record["http_request_id"] is None
    assert record["rejected_candidates"] == []
    assert record["valid_candidate_count"] == 1 and record["silent_fallback"] is False
    body = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert set(body["bodies"][0]) == {"model", "max_tokens", "messages"}
    assert body["bodies"][0]["messages"][0] == {
        "role": "system", "content": fp.system_prompt_bytes().decode()
    }
    assert body["bodies"][0]["messages"][1]["content"] == packet


def test_openai_one_format_repair_then_success(tmp_path):
    """Acceptance: malformed then valid gives calls=2, repairs=1 - identical
    contract to test_one_format_repair_then_success for Anthropic."""
    good = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"})
    proposer, (candidates, record), _ = _propose_openai(tmp_path, ["not json at all", good])
    assert len(candidates) == 1 and record["status"] == "parsed"
    assert record["calls"] == 2 and record["repairs"] == 1
    assert len(record["rejected_candidates"]) == 1
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]
    roles = [m["role"] for m in proposer.client.bodies[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_openai_two_failures_leave_proposal_failed_with_both_messages(tmp_path):
    """Test: openai two failures leave proposal failed with both messages."""
    bad = _object(expression="c*p1/r**7", family="gauss", parameters={"c": "positive"})
    _p, (candidates, record), _ = _propose_openai(tmp_path, [bad, bad])
    assert candidates == [] and record["status"] == "proposal_failed"
    assert record["calls"] == 2 and record["repairs"] == 1
    assert len(record["rejected_candidates"]) == 2
    assert all("verbatim" in item["reason"] for item in record["rejected_candidates"])


def test_openai_non_stop_finish_reason_is_not_parsed(tmp_path):
    """OpenAI's `finish_reason` is the "stop_reason" equivalent - "stop" is
    required, exactly as Anthropic's "end_turn" (see
    test_truncated_response_is_not_parsed)."""
    good = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"})
    _p, (candidates, record), _ = _propose_openai(
        tmp_path, [good, good], finish_reasons=["length", "stop"])
    assert len(candidates) == 1 and record["calls"] == 2
    assert "stop" in record["rejected_candidates"][0]["reason"]


def test_openai_and_anthropic_mask_a_leaked_key_from_the_exception(tmp_path, monkeypatch):
    """An SDK error that happens to echo the live
    key (e.g. a header dump in an auth-failure message) must never reach a
    written record. A client whose call itself raises with the key embedded
    fails after exactly one call - an API-call-level failure never triggers a
    repair (raw_text is still None when the except block runs) - and the
    recorded reason carries the masked placeholder, never the raw key. Both
    providers share the module-level `mask_secrets` helper."""
    fake_openai_key = "sk-test-FAKEKEY-21g-fb1-openai"
    fake_anthropic_key = "sk-test-FAKEKEY-21g-fb1-anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", fake_openai_key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_anthropic_key)
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1))
    context = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "P1",
               "event": 0, "round": 1, "n_designs": 3, "candidate_index": 0,
               "packet_file": "t03_c0.json"}

    # (a) succeeds once: baseline with both keys live in the environment.
    ok_proposer, (ok_candidates, ok_record), _ = _propose_openai(tmp_path)
    assert len(ok_candidates) == 1 and ok_record["status"] == "parsed"
    assert fake_openai_key not in json.dumps(ok_record)

    # (b) raises with the key embedded in the message.
    class RaisingOpenAI:
        """Fake openai client whose create() raises an error that embeds the API key."""
        provider = "openai"

        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **body):
            raise RuntimeError(
                f"upstream auth error; echoed header Authorization: Bearer {fake_openai_key}"
            )

    # An openai-provider client - injected or live - must be
    # given the model explicitly; the default is Anthropic's claude-opus-4-7.
    openai_proposer = fp.Fb1CatalogProposer(
        tmp_path / "openai_leak", model="gpt-4o", client=RaisingOpenAI()
    )
    openai_candidates, openai_record = openai_proposer.propose_one(
        packet, context=context, constraints=constraints
    )
    assert openai_candidates == [] and openai_record["status"] == "proposal_failed"
    assert openai_record["calls"] == 1 and openai_record["repairs"] == 0
    dumped_openai = json.dumps(openai_record)
    assert fake_openai_key not in dumped_openai
    assert "<OPENAI_API_KEY:masked>" in dumped_openai
    for path in (tmp_path / "openai_leak").iterdir():
        assert fake_openai_key not in path.read_text()

    class RaisingAnthropic:
        """Fake anthropic client whose create() raises an error that embeds the API key."""
        provider = "anthropic"

        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **body):
            raise RuntimeError(
                f"upstream auth error; echoed header x-api-key: {fake_anthropic_key}"
            )

    anthropic_proposer = fp.Fb1CatalogProposer(
        tmp_path / "anthropic_leak", client=RaisingAnthropic()
    )
    anthropic_candidates, anthropic_record = anthropic_proposer.propose_one(
        packet, context=context, constraints=constraints
    )
    assert anthropic_candidates == [] and anthropic_record["status"] == "proposal_failed"
    dumped_anthropic = json.dumps(anthropic_record)
    assert fake_anthropic_key not in dumped_anthropic
    assert "<ANTHROPIC_API_KEY:masked>" in dumped_anthropic
    for path in (tmp_path / "anthropic_leak").iterdir():
        assert fake_anthropic_key not in path.read_text()


def test_mask_secrets_ignores_unset_or_short_env_values(monkeypatch):
    """Guard for the masking helper itself: an unset or trivially short env
    var value must never turn masking into an unbounded/blank replacement."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert fp.mask_secrets("plain text, nothing to mask") == "plain text, nothing to mask"
    monkeypatch.setenv("OPENAI_API_KEY", "short")  # < 8 chars: not treated as a secret
    assert fp.mask_secrets("contains short verbatim") == "contains short verbatim"
    assert fp.mask_secrets(None) is None


# ------------------------------------- transport hardening
# Every test uses an injected fake client or an in-process httpx.MockTransport:
# no real key (only obviously fake `sk-test-...` values), no external HTTP, zero paid calls.

import io  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402

G2_CONTEXT = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "P1",
              "event": 0, "round": 1, "n_designs": 3, "candidate_index": 0,
              "packet_file": "t03_c0.json"}
G2_GOOD = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"})


def _g2_packet():
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    packet = fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1))
    return packet, constraints


def _g2_response(text, *, index=0, usage=True, finish_reason="stop", http_request_id=None):
    """One fake OpenAI-shaped response.  `usage=False` reproduces the review case (section 3: a valid JSON reply
    with no usage object); `http_request_id` sets the `_request_id` that newer SDKs attach."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason=finish_reason)],
        usage=(SimpleNamespace(prompt_tokens=100 + 10 * index,
                               completion_tokens=10 + index,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=20))
               if usage else None),
        id=f"chatcmpl_test_only_{index}",
        model="returned-test-model",
    )
    if http_request_id is not None:
        response._request_id = http_request_id
    return response


def _g2_client(items):
    """Injected openai client: returns `items` in order and raises the items that are exceptions."""
    calls = []

    def create(**body):
        """Record the request body and return or raise the next scripted item."""
        calls.append(body)
        item = items[min(len(calls) - 1, len(items) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.provider = "openai"
    return client, calls


def _g2_proposer(tmp_path, name, items):
    client, calls = _g2_client(items)
    return fp.Fb1CatalogProposer(tmp_path / name, model="gpt-4o", client=client), calls


def test_21g2_mask_placeholders_are_exactly_the_two_allowed_forms(monkeypatch):
    """Whatever is caught, the placeholder is exactly one of two forms; no length, prefix or partial
    substring is left, and `sk-ant-` is never attributed to an OpenAI key."""
    openai_key = "sk-test-openai-FAKEKEY-21g2-abcdefgh"
    anthropic_key = "sk-ant-test-FAKEKEY-21g2-abcdefgh"
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    text = (f"env_openai={openai_key} env_anthropic={anthropic_key} "
            "loose_anthropic=sk-ant-api03-UNRELATEDabcdefgh12 "
            "loose_openai=sk-UNRELATEDabcdefgh12")
    masked = fp.mask_secrets(text)
    assert masked == ("env_openai=<OPENAI_API_KEY:masked> "
                      "env_anthropic=<ANTHROPIC_API_KEY:masked> "
                      "loose_anthropic=<ANTHROPIC_API_KEY:masked> "
                      "loose_openai=<OPENAI_API_KEY:masked>")
    assert set(re.findall(r"<[^>]+>", masked)) <= set(fp.MASK_PLACEHOLDERS)
    assert "sk-" not in masked and openai_key[:8] not in masked


def test_21g2_masking_is_recursive_over_containers_and_keys(monkeypatch):
    """`mask_secrets_deep` walks dict keys and values, lists, tuples and sets, and
    changes no value that is not a string."""
    fake = "sk-test-FAKEKEY-21g2-fb1-recursive"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    payload = {f"header {fake}": [f"a {fake}", {"deep": (f"b {fake}", 3)}],
               "numbers": [1, 2.5, None, True], "set": {f"c {fake}"}}
    masked = fp.mask_secrets_deep(payload)
    masked_key = next(iter(masked))
    assert masked_key == "header <OPENAI_API_KEY:masked>"
    assert masked[masked_key][1]["deep"] == ("b <OPENAI_API_KEY:masked>", 3)
    assert masked["numbers"] == [1, 2.5, None, True]
    assert masked["set"] == {"c <OPENAI_API_KEY:masked>"}


def test_21g2_success_record_masks_raw_text_note_and_ignored_fields(tmp_path, monkeypatch):
    """The "success note echo" counterexample.  When the note or an extra key of a **successful** reply
    carried a key, it was saved verbatim to raw_text, note and ignored_keys (the masking
    covered only `reason`)."""
    fake = "sk-test-FAKEKEY-21g2-fb1-success-note"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    packet, constraints = _g2_packet()
    reply = _object(expression="c*p1/r", family="gauss", parameters={"c": "positive"},
                    note=f"echoed credential {fake}",
                    priors={f"header {fake}": f"value {fake}"})
    proposer, calls = _g2_proposer(tmp_path, "note_echo", [_g2_response(reply)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 1 and len(candidates) == 1 and record["status"] == "parsed"
    assert fake not in json.dumps(record, ensure_ascii=False)
    assert record["note"] == "echoed credential <OPENAI_API_KEY:masked>"
    assert record["responses"][0]["raw_text"].count("<OPENAI_API_KEY:masked>") == 3
    assert json.dumps(record["ignored_keys"]).count("<OPENAI_API_KEY:masked>") == 2
    for path in (tmp_path / "note_echo").iterdir():
        assert fake not in path.read_text()


def test_21g2_repair_turn_masks_the_assistant_text_and_the_error_before_sending(
    tmp_path, monkeypatch
):
    """The "repair error echo" counterexample.  The repair call resends the model's original reply and the validation
    error message, and both used to remain unmasked on the wire and in request.json."""
    fake = "sk-test-FAKEKEY-21g2-fb1-repair-turn"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    packet, constraints = _g2_packet()
    leaky = _object(expression="c*p1/r", family=fake, parameters={"c": "positive"})
    proposer, calls = _g2_proposer(
        tmp_path, "repair_echo", [_g2_response(leaky), _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 2 and len(candidates) == 1 and record["repairs"] == 1
    second = json.dumps(calls[1], ensure_ascii=False)
    assert fake not in second
    # (a) The resent assistant text is exactly the masked body.
    assert second.count("<OPENAI_API_KEY:masked>") == 1
    assert calls[1]["messages"][2]["content"] == leaky.replace(
        fake, "<OPENAI_API_KEY:masked>")
    # (b) The error message is masked before sending too: the sentence sent is the **same string as the recorded masked
    #     reason** (the family rejection message of fb1 does not quote the value,
    #     so the fake key is never in the error message for this reply).
    reason = record["rejected_candidates"][0]["reason"]
    assert reason == fp.mask_secrets(reason)
    assert calls[1]["messages"][3]["content"] == fp.REPAIR_INSTRUCTION.format(error=reason)
    assert fake not in json.dumps(record, ensure_ascii=False)
    for path in (tmp_path / "repair_echo").iterdir():
        assert fake not in path.read_text()


def test_21g2_sdk_constructor_exception_is_masked_and_creates_no_output(tmp_path, monkeypatch):
    """The SDK constructor sat outside the recording try, so an exception carrying the key
    was re-raised as is in str, repr and traceback.  It is now masked inside the try and re-raised,
    and no record directory is created."""
    fake = "sk-test-FAKEKEY-21g2-fb1-constructor"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    monkeypatch.setenv("FB1_LLM_PROVIDER", "openai")
    openai_module = pytest.importorskip("openai")

    def explode(*args, **kwargs):
        """Stand-in SDK constructor that raises an error embedding the fake key."""
        raise RuntimeError(f"fake SDK constructor error; Authorization: Bearer {fake}")

    monkeypatch.setattr(openai_module, "OpenAI", explode)
    with pytest.raises(fp.MaskedCredentialError) as excinfo:
        fp.Fb1CatalogProposer(tmp_path / "ctor", allow_live=True, model="gpt-4o")
    caught = excinfo.value
    formatted = "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))
    assert fake not in str(caught) and fake not in repr(caught) and fake not in formatted
    assert "<OPENAI_API_KEY:masked>" in str(caught) and "RuntimeError" in str(caught)
    assert not (tmp_path / "ctor").exists()


def test_21g2_numerical_failure_still_propagates_verbatim(monkeypatch):
    """Masking applies only at the API transport boundary.  An exception with nothing to hide
    comes back as the **same object** and keeps its type and traceback."""
    plain = ValueError("SMC produced no finite log evidence")
    assert fp.masked_exception(plain) is plain
    assert not fp.carries_secret(str(plain))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-FAKEKEY-21g2-not-in-this-text")
    assert fp.masked_exception(plain) is plain


def test_21g2_missing_usage_is_unknown_and_never_triggers_a_repair(tmp_path):
    """A **valid** JSON reply without a usage object raised AttributeError, fell into the catch-all
    except, and because raw_text already existed a second paid call was made.  It is now
    recorded as unknown and parsed as is: one call."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(
        tmp_path, "missing_usage",
        [_g2_response(G2_GOOD, usage=False), _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 1 and len(candidates) == 1
    assert record["calls"] == 1 and record["repairs"] == 0 and record["status"] == "parsed"
    assert record["rejected_candidates"] == [] and len(record["responses"]) == 1
    assert record["responses"][0]["usage"] is None
    assert record["responses"][0]["usage_known"] is False
    assert record["responses"][0]["completion_id"] == "chatcmpl_test_only_0"
    assert record["last_call_usage"] is None and record["last_call_usage_known"] is False
    assert record["usage_total"] == {} and record["usage_total_complete"] is False
    assert record["usage_missing_indices"] == [0]
    assert record["paid_api_calls"] == 1


def test_21g2_sdk_object_contract_error_is_recorded_and_never_repaired(tmp_path):
    """An SDK object-contract error (here `choices` is an empty list) is not a format failure either:
    one call, recorded, no repair, `repair_eligible: false`."""
    packet, constraints = _g2_packet()
    broken = SimpleNamespace(choices=[], usage=None, id="chatcmpl_broken", model="m")
    proposer, calls = _g2_proposer(
        tmp_path, "broken_object", [broken, _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 1 and candidates == [] and record["status"] == "proposal_failed"
    assert record["calls"] == 1 and record["repairs"] == 0
    assert record["rejected_candidates"][0]["error_type"] == "IndexError"
    assert record["rejected_candidates"][0]["repair_eligible"] is False


def test_21g2_format_failure_is_the_only_thing_marked_repair_eligible(tmp_path):
    """The complement of the two cases above (missing usage, SDK object-contract error): a format (JSON) error and a candidate-constraint error are repairable, and the repair happens once."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(
        tmp_path, "format_failure",
        [_g2_response("not JSON at all"), _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 2 and len(candidates) == 1 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["repair_eligible"] is True
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]


@pytest.mark.parametrize("value", [2, 3, -1, 1.5, True])
def test_21g2_repair_attempts_accepts_only_zero_or_one(tmp_path, value):
    """`range(1 + max(0, int(repair_attempts)))` accepted 3 and made four
    calls.  Only 0 or 1 is allowed; anything else is rejected before any call."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(tmp_path, f"bound_{value!r}", [_g2_response(G2_GOOD)])
    with pytest.raises(ValueError, match="repair_attempts must be 0 or 1"):
        proposer.propose_one(packet, context=G2_CONTEXT, constraints=constraints,
                             repair_attempts=value)
    assert calls == [] and proposer.attempts == 0 and proposer.paid_api_calls == 0


def test_21g2_repair_attempts_zero_makes_exactly_one_call(tmp_path):
    """0 is still legal and means "no repair"."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(tmp_path, "no_repair", [_g2_response("not JSON at all")])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints, repair_attempts=0)
    assert len(calls) == 1 and candidates == []
    assert record["calls"] == 1 and record["repairs"] == 0


def test_21g2_usage_total_sums_every_response_exactly_once(tmp_path):
    """The top-level usage is that of the **last** call (now `last_call_usage`),
    and `usage_total` is the sum over responses[*], each counted once and never double-counted."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(
        tmp_path, "usage_total",
        [_g2_response("not JSON at all"), _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 2 and len(candidates) == 1
    assert record["responses"][0]["usage"] == {
        "input_tokens": 100, "output_tokens": 10, "cached_tokens": 20}
    assert record["responses"][1]["usage"] == {
        "input_tokens": 110, "output_tokens": 11, "cached_tokens": 20}
    assert record["last_call_usage"] == record["responses"][1]["usage"]
    assert record["usage_total"] == {
        "input_tokens": 210, "output_tokens": 21, "cached_tokens": 40}
    assert record["usage_total_complete"] is True and record["usage_missing_indices"] == []
    assert "usage" not in record and "request_id" not in record


def test_21g2_usage_total_marks_the_missing_call_instead_of_counting_zero(tmp_path):
    """An unknown usage is not counted as 0; its index is kept and the total is marked incomplete,
    so the cost ledger can reserve conservatively."""
    packet, constraints = _g2_packet()
    proposer, calls = _g2_proposer(
        tmp_path, "usage_partial",
        [_g2_response("not JSON at all", usage=False), _g2_response(G2_GOOD, index=1)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 2 and len(candidates) == 1
    assert record["usage_total"] == {
        "input_tokens": 110, "output_tokens": 11, "cached_tokens": 20}
    assert record["usage_total_complete"] is False
    assert record["usage_missing_indices"] == [0]


def test_21g2_completion_id_and_http_request_id_are_named_apart(tmp_path):
    """`response.id` is the **completion** id (`chatcmpl_...`) and not the HTTP
    `x-request-id`.  The two are named apart, and the latter is null when the SDK does not expose it."""
    packet, constraints = _g2_packet()
    proposer, _calls = _g2_proposer(
        tmp_path, "ids_present", [_g2_response(G2_GOOD, http_request_id="req_test_only_0")])
    _candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert record["completion_id"] == "chatcmpl_test_only_0"
    assert record["http_request_id"] == "req_test_only_0"
    assert record["responses"][0]["completion_id"] == "chatcmpl_test_only_0"
    assert record["responses"][0]["http_request_id"] == "req_test_only_0"

    proposer, _calls = _g2_proposer(tmp_path, "ids_absent", [_g2_response(G2_GOOD)])
    _candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert record["completion_id"] == "chatcmpl_test_only_0"
    assert record["http_request_id"] is None


def test_21g2_real_sdk_mocktransport_debug_logs_and_traceback_are_masked(tmp_path, monkeypatch):
    """The **installed SDK object** + an httpx.MockTransport that raises a ConnectError carrying the key
    + DEBUG logging of openai/anthropic/httpx/httpcore.  The key appears zero times in the captured log
    bodies, the formatted traceback, the record and the files.  The MockTransport takes
    all HTTP in-process, so there are no external requests and zero paid calls."""
    httpx = pytest.importorskip("httpx")
    openai_module = pytest.importorskip("openai")
    fake = "sk-test-FAKEKEY-21g2-fb1-sdk-debug-logging"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    packet, constraints = _g2_packet()
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    loggers = [logging.getLogger(name) for name in fp.MASKED_LOGGER_ROOTS]
    saved = [(logger, logger.level, logger.propagate) for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False

    def refuse(request):
        """Transport handler that raises a ConnectError carrying the fake key."""
        raise httpx.ConnectError(f"fake low-level transport failure {fake}", request=request)

    client = openai_module.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(refuse)),
        max_retries=0, timeout=1.0)
    client.provider = "openai"
    try:
        raw = None
        try:
            client.chat.completions.create(
                model="gpt-4o", max_tokens=16, messages=[{"role": "user", "content": "{}"}])
        except BaseException as exc:  # noqa: BLE001 - this exception is the fixture
            raw = exc
        # The SDK wraps the transport failure (`APIConnectionError("Connection error.")`),
        # so the key is in the **chained cause**, not in `str(exc)`; the formatted traceback and
        # the SDK's own `exc_info=True` DEBUG log are exactly what print it.
        assert raw is not None
        raw_formatted = "".join(traceback.format_exception(type(raw), raw, raw.__traceback__))
        assert fake in raw_formatted and fp.carries_secret(raw_formatted)
        try:
            raise fp.masked_exception(raw) from None
        except BaseException as masked:  # noqa: BLE001 - re-raised in order to inspect the traceback
            formatted = "".join(
                traceback.format_exception(type(masked), masked, masked.__traceback__))
            assert isinstance(masked, fp.MaskedCredentialError)
            assert str(masked).startswith("APIConnectionError: ")
            assert fake not in str(masked) and fake not in repr(masked)
            assert fake not in formatted
        proposer = fp.Fb1CatalogProposer(tmp_path / "sdk_debug", model="gpt-4o", client=client)
        candidates, record = proposer.propose_one(
            packet, context=G2_CONTEXT, constraints=constraints)
    finally:
        client.close()
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
    logs = sink.getvalue()
    assert candidates == [] and record["calls"] == 1 and record["status"] == "proposal_failed"
    assert record["rejected_candidates"][0]["repair_eligible"] is False
    assert fake not in logs
    assert "<OPENAI_API_KEY:masked>" in logs, "the masking filter was never applied"
    assert fake not in json.dumps(record, ensure_ascii=False)
    for path in (tmp_path / "sdk_debug").iterdir():
        assert fake not in path.read_text()


@pytest.mark.parametrize(
    "logger_name",
    ["openai._base_client", "anthropic._base_client", "httpx._client", "httpcore._sync.http11"],
)
def test_21g2_logging_filter_masks_message_args_and_exc_info(logger_name):
    """The filter itself, checked on records made by **child** loggers of the four library roots (openai, anthropic, httpx, httpcore).
    A `logging.Filter` attached only to `openai` would not see these records
    (only the child logger's filters run, and callHandlers looks only at handlers)."""
    fake = "sk-test-FAKEKEY-21g2-fb1-logrecord-filter"
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    saved = (logger.level, logger.propagate)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.debug("header %s", f"Bearer {fake}")
        try:
            raise RuntimeError(f"auth failed for {fake}")
        except RuntimeError:
            logger.debug("Encountered Exception", exc_info=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
    text = sink.getvalue()
    assert fake not in text
    assert text.count("<OPENAI_API_KEY:masked>") == 2
    assert "RuntimeError" in text and "Traceback" in text


# ------------------------------------------------------------------------- explicit model for the OpenAI transport
def test_a5_openai_provider_refuses_an_unnamed_or_claude_model_before_any_output(
    tmp_path, monkeypatch
):
    """Changing only `FB1_LLM_PROVIDER=openai` leaves the default model at Anthropic's
    `claude-opus-4-7` (run_gravity.py line 121, outside the scope of this work), so a paid
    smoke run would ask OpenAI for a Claude model.  The proposer constructor refuses **before creating any output**.
    No real SDK client is built (the exception comes first)."""
    monkeypatch.setenv("FB1_LLM_PROVIDER", "openai")
    for name, kwargs in (
        ("default_model", {}),
        ("explicit_claude", {"model": "claude-opus-4-7"}),
        ("other_claude", {"model": "claude-sonnet-4-5"}),
        ("empty_model", {"model": ""}),
        ("none_model", {"model": None}),
    ):
        with pytest.raises(ValueError,
                           match="FB1_LLM_MODEL must be set explicitly for provider openai"):
            fp.Fb1CatalogProposer(tmp_path / name, allow_live=True, **kwargs)
        assert not (tmp_path / name).exists()
    # The same holds when the provider is given by the provider= argument.
    with pytest.raises(ValueError, match="FB1_LLM_MODEL must be set explicitly"):
        fp.Fb1CatalogProposer(tmp_path / "kwarg", allow_live=True, provider="openai")
    assert not (tmp_path / "kwarg").exists()


def test_a5_injected_openai_client_obeys_the_same_rule(tmp_path):
    """The same rule applies when the injected client's provider is openai, so the test path
    cannot bypass the rule and let only the paid path behave differently later."""
    client, _calls = _g2_client([_g2_response(G2_GOOD)])
    with pytest.raises(ValueError, match="FB1_LLM_MODEL must be set explicitly"):
        fp.Fb1CatalogProposer(tmp_path / "injected", client=client)
    assert not (tmp_path / "injected").exists()


def test_a5_anthropic_and_mock_transports_keep_their_default_model(tmp_path):
    """Scope of the rule: the anthropic, mock and other injected transports keep the default model
    (catalogue-mode behaviour unchanged).  No real SDK client is built on any path."""
    mock_proposer = fp.Fb1CatalogProposer(tmp_path / "mock")
    assert mock_proposer.transport == "mock" and mock_proposer.model == fp.DEFAULT_MODEL
    scripted = fp.Fb1CatalogProposer(tmp_path / "scripted", client=fp.ScriptedClient([G2_GOOD]))
    assert scripted.transport == "scripted" and scripted.model == fp.DEFAULT_MODEL
    assert fp.DEFAULT_MODEL.startswith("claude-")


# --------------------------------- normalisation of the parameters shape
# Basis: the diagnosis of the gpt-4o smoke run: 2 of 2 proposals
# sent the `parameters` values as `{"c": {"role": "positive"}}` objects, so they were rejected for their **shape although the role
# was right** (and the error message read like a role violation).

C20_EXPRESSION = "c*p1/r + b*p1"          # the 7th catalogue entry = inv_r_plus_c
C20_DECLARED = fp.catalog_constraints()["structure_parameters"][C20_EXPRESSION]


def _c20_reply(parameters, *, expression=C20_EXPRESSION, family="student_t4", **extra):
    return _object(expression=expression, family=family, parameters=parameters, **extra)


def test_20c_b1_role_object_is_normalized_and_recorded():
    """Role-object shape, case 1: `{"role": X}` is normalised to X, accepted,
    and recorded as `normalized_parameter_shape: true`.  Expression, family and model_id are **exactly the same as for the string
    form**; no scientific constant moves."""
    constraints = fp.catalog_constraints()
    plain, plain_ignored = fp.parse_candidate_object(
        _c20_reply({"c": "positive", "b": "positive"}), constraints)
    assert plain_ignored["normalized_parameter_shape"] is False

    nested, ignored = fp.parse_candidate_object(
        _c20_reply({"c": {"role": "positive"}, "b": {"role": "positive"}}), constraints)
    assert ignored["normalized_parameter_shape"] is True
    assert nested["parameters"] == {"b": "positive", "c": "positive"}
    assert nested["model_id"] == plain["model_id"]
    assert nested["structure"] == plain["structure"] == "inv_r_plus_c"
    assert nested["expression"] == plain["expression"]


def test_20c_b1_role_object_with_the_declared_prior_is_normalized_too():
    """Role-object shape, case 2: with `{"role": X, "prior": P}`, when P is the **same string** as the catalogue declaration
    it goes through the same normalisation, since it only reads back the value the packet showed."""
    constraints = fp.catalog_constraints()
    parameters = {
        name: {"role": spec["role"], "prior": spec["prior"]}
        for name, spec in C20_DECLARED.items()
    }
    candidate, ignored = fp.parse_candidate_object(_c20_reply(parameters), constraints)
    assert ignored["normalized_parameter_shape"] is True
    assert candidate["parameters"] == {"b": "positive", "c": "positive"}
    assert all(spec["prior"].startswith("LogNormal") for spec in C20_DECLARED.values())


def test_20c_b1_a_restated_prior_that_differs_is_rejected():
    """Role-object shape, case 3: restating a prior that **differs** from the declaration is a prior modification, which the contract
    forbids; it is not silently dropped but rejected with the predeclared wording."""
    constraints = fp.catalog_constraints()
    parameters = {
        "c": {"role": "positive", "prior": "LogNormal(0,5)"},
        "b": {"role": "positive"},
    }
    with pytest.raises(ValueError) as excinfo:
        fp.parse_candidate_object(_c20_reply(parameters), constraints)
    assert str(excinfo.value) == "prior is fixed by the catalogue; do not restate it"
    assert str(excinfo.value) == fp.PRIOR_RESTATED_MESSAGE
    ok, reason = fp.judge_candidate(
        {"expression": C20_EXPRESSION, "family": "student_t4", "parameters": parameters},
        constraints,
    )
    assert not ok and reason == fp.PRIOR_RESTATED_MESSAGE


@pytest.mark.parametrize(
    "value,word",
    [({"kind": "positive"}, "object"), ({"role": "positive", "scale": 2.0}, "object"),
     ({"role": 7}, "number"), (17, "number"), (["positive"], "array"), (None, "null"),
     (True, "boolean")],
)
def test_20c_b1_other_objects_and_non_strings_are_rejected_clearly(value, word):
    """Role-object shape, case 3 continued: any other object or non-string is rejected with the shape wording and the **actual JSON type**;
    it is no longer disguised as a role violation."""
    constraints = fp.catalog_constraints()
    expected = fp.PARAMETER_SHAPE_MESSAGE % word
    with pytest.raises(ValueError) as excinfo:
        fp.parse_candidate_object(_c20_reply({"c": value, "b": "positive"}), constraints)
    assert str(excinfo.value) == expected
    assert 'parameters values must be role strings, e.g. {"c": "positive"}' in expected


def test_20c_b1_the_role_must_still_be_the_declared_one():
    """The role-object shape rule has not loosened anything: only the shape is normalised and the **role is still checked as before**."""
    constraints = fp.catalog_constraints()
    with pytest.raises(ValueError, match="has the fixed role positive"):
        fp.parse_candidate_object(
            _c20_reply({"c": {"role": "signed"}, "b": "positive"}), constraints)


def test_20c_b1_rejection_order_is_unchanged_from_20a():
    """The normalisation sits right before the role comparison, so the order of rejection reasons is the same as in catalogue mode:
    an expression outside the catalogue is still rejected first by the verbatim rule even when the shape is nested."""
    constraints = fp.catalog_constraints()
    with pytest.raises(ValueError, match="verbatim one of constraints.initial_structures"):
        fp.parse_candidate_object(
            _c20_reply({"c": {"role": "positive"}}, expression="c*p1/p2"), constraints)
    with pytest.raises(ValueError, match="one of constraints.allowed_families"):
        fp.parse_candidate_object(
            _c20_reply({"c": {"role": "positive"}, "b": {"role": "positive"}},
                       family="cauchy"),
            fp.catalog_constraints())
    with pytest.raises(ValueError, match="name exactly the coefficients"):
        fp.parse_candidate_object(
            _c20_reply({"c": {"role": "positive"}}), constraints)


def test_20c_b1_normalization_reaches_the_transport_record(tmp_path):
    """Role-object shape end to end: a nested-shape reply passes through the transport and is accepted, and the record says so;
    this is the place where a round was lost in the smoke run."""
    packet, constraints = _g2_packet()
    reply = _c20_reply({"c": {"role": "positive"}, "b": {"role": "positive"}})
    proposer, calls = _g2_proposer(tmp_path, "c20_record", [_g2_response(reply)])
    candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert len(calls) == 1 and len(candidates) == 1 and record["status"] == "parsed"
    assert record["normalized_parameter_shape"] is True
    saved = json.loads(
        (tmp_path / "c20_record" / "event_00_round_01_c0.response.json").read_text())
    assert saved["normalized_parameter_shape"] is True


def test_20c_b2_constraints_carry_one_parameters_format_line():
    """`catalog_constraints` gained one line.  `structure_parameters` keeps the nested
    {role, prior} shape, so the fixed per-coefficient prior stays visible."""
    constraints = fp.catalog_constraints()
    assert set(constraints) == fp.CONSTRAINT_KEYS
    assert "parameters_format" in fp.CONSTRAINT_KEYS
    line = constraints["parameters_format"]
    assert isinstance(line, str) and "\n" not in line
    assert line == fp.PARAMETERS_FORMAT_LINE
    assert '{"c": "positive", "b": "positive"}' in line
    assert "structure_parameters" in line and "role" in line
    # prior visibility is kept (the shape is as in catalogue mode)
    declared = constraints["structure_parameters"][C20_EXPRESSION]
    assert declared == {"b": {"role": "positive", "prior": "LogNormal(0,1)"},
                        "c": {"role": "positive", "prior": "LogNormal(0,1)"}}
    # it is carried in the packet too
    rows, members, _c, interface = _state()
    packet = json.loads(fp.serialize_packet(
        fp.build_belief_packet(rows, members, [], constraints, interface, 1)))
    assert packet["constraints"]["parameters_format"] == line


def test_20c_b3_repair_instruction_names_the_expected_shape():
    """The repair instruction gained one line with a shape example (engine text, not the prompt file).
    One repair, max_tokens 1600 and no temperature sent are unchanged."""
    assert '{"c": "positive"}' in fp.REPAIR_INSTRUCTION
    assert '{"role": "positive"}' in fp.REPAIR_INSTRUCTION
    assert "role STRING" in fp.REPAIR_INSTRUCTION
    assert fp.REPAIR_INSTRUCTION.count("{error}") == 1
    assert fp.MAX_TOKENS == 1600


def test_20c_b3_prompt_file_and_decoding_are_untouched(tmp_path):
    """Invariants: the prompt v2.0 file bytes (sha f095edf1...), max_tokens 1600,
    no temperature sent, and one repair."""
    assert hashlib.sha256(fp.system_prompt_bytes()).hexdigest() == fp.PROMPT_SHA256
    assert fp.PROMPT_SHA256 == (
        "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef")
    packet, constraints = _g2_packet()
    reply = _c20_reply({"c": {"role": "positive"}, "b": {"role": "positive"}})
    proposer, calls = _g2_proposer(tmp_path, "c20_wire", [_g2_response(reply)])
    _candidates, record = proposer.propose_one(
        packet, context=G2_CONTEXT, constraints=constraints)
    assert set(calls[0]) == {"model", "max_tokens", "messages"}
    assert calls[0]["max_tokens"] == 1600 and "temperature" not in calls[0]
    assert calls[0]["messages"][0]["content"].encode() == fp.system_prompt_bytes()
    assert record["decoding"] == {
        "temperature": "not_sent/provider_default", "max_tokens": 1600}
    assert record["temperature_sent"] is False


# Four SYNTHETIC replies in the shape the catalogue prompt (v2.0) asks for. They are written here, not copied from any
# run: recorded model outputs are never used as fixtures. Each reply exercises one parser branch of the shape normalisation:
#   1. the catalogue expression with the `{"role": ...}` parameter shape and a short note     -> accepted, shape normalised
#   2. the same candidate without a note                                                        -> accepted (the same model_id)
#   3. an expression outside the catalogue                                                      -> rejected: verbatim rule
#   4. the first candidate once more                                                            -> accepted (a duplicate proposal)
SYNTHETIC_RAW_TEXTS = [
    ("synthetic_catalogue_with_note",
     ('{"expression":"c*p1/r + b*p1","family":"student_t4","parameters":'
      '{"c":{"role":"positive"},"b":{"role":"positive"}},"note":"synthetic reply: the heavy-tailed '
      'family and a linear term, written for the parser test."}'),
     "accepted"),
    ("synthetic_catalogue_without_note",
     ('{"expression":"c*p1/r + b*p1","family":"student_t4","parameters":'
      '{"c":{"role":"positive"},"b":{"role":"positive"}}}'),
     "accepted"),
    ("synthetic_outside_catalogue",
     ('{"expression":"c*p1/r^3","family":"student_t4","parameters":{"c":{"role":"positive"}}, '
      '"note":"synthetic reply: an inverse-cube law, which the catalogue does not list."}'),
     "rejected_not_verbatim"),
    ("synthetic_catalogue_repeated",
     ('{"expression":"c*p1/r + b*p1","family":"student_t4","parameters":'
      '{"c":{"role":"positive"},"b":{"role":"positive"}}}'),
     "accepted"),
]


@pytest.mark.parametrize("name,text,expected", SYNTHETIC_RAW_TEXTS,
                         ids=[row[0] for row in SYNTHETIC_RAW_TEXTS])
def test_20c_synthetic_replies_reparse_as_predeclared(name, text, expected):
    """Acceptance: three of the four synthetic replies are accepted with the shape-normalisation flag set, and the
    one whose expression is outside the catalogue keeps its verbatim rejection - the shape normalisation does not
    loosen the verbatim rule."""
    constraints = fp.catalog_constraints(family_expansion=True)
    if expected == "accepted":
        candidate, ignored = fp.parse_candidate_object(text, constraints)
        assert ignored["normalized_parameter_shape"] is True
        assert candidate["expression"] == C20_EXPRESSION
        assert candidate["structure"] == "inv_r_plus_c"
        assert candidate["family"] == "student_t4"
        assert candidate["parameters"] == {"b": "positive", "c": "positive"}
        return
    with pytest.raises(ValueError, match="verbatim one of constraints.initial_structures"):
        fp.parse_candidate_object(text, constraints)


def test_20c_synthetic_replies_are_three_accepted_and_one_rejected():
    """The same four replies summarised: 3 accepted / 1 rejected, the accepted three being one and the same candidate."""
    constraints = fp.catalog_constraints(family_expansion=True)
    accepted, rejected = [], []
    for name, text, _expected in SYNTHETIC_RAW_TEXTS:
        try:
            candidate, ignored = fp.parse_candidate_object(text, constraints)
        except ValueError as exc:
            rejected.append((name, str(exc)))
        else:
            accepted.append((name, ignored["normalized_parameter_shape"], candidate["model_id"]))
    assert len(accepted) == 3 and len(rejected) == 1
    assert all(flag is True for _n, flag, _m in accepted)
    assert len({model for _n, _f, model in accepted}) == 1  # the same candidate three times
    assert "verbatim" in rejected[0][1]


# ============================================================ free-form mode
# This section was added later; the tests above were not changed by a single character.
# The tests of the grammar engine itself (vendored sha, operator counting, force-function signature, numerical agreement of the catalogue notation, etc.)
# live in the grammar test module; here only the **proposer-side
# wiring** is checked.
FREE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def test_20b_constraint_key_sets_are_exactly_two():
    """There are exactly two constraint key sets, the catalogue set and the free set, each matched **exactly**."""
    catalog = fp.catalog_constraints("gauss")
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    assert set(catalog) == fp.CONSTRAINT_KEYS
    assert set(free) == fp.FREE_CONSTRAINT_KEYS
    assert fp.FREE_CONSTRAINT_KEYS - fp.CONSTRAINT_KEYS == {"grammar", "input_names",
                                                            "n_ops_max"}
    assert fp.constraint_keys_ok(catalog) and fp.constraint_keys_ok(free)
    assert fp.is_free_constraints(free) and not fp.is_free_constraints(catalog)
    assert not fp.constraint_keys_ok({**free, "extra": 1})
    assert not fp.constraint_keys_ok({k: v for k, v in free.items() if k != "n_ops_max"})


def test_20b_free_constraints_open_structure_expansion_and_keep_20c_format_line():
    """Test: free constraints open structure expansion and keep the parameters format line."""
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    assert free["structure_expansion"] is True
    assert free["parameters_format"] == fp.PARAMETERS_FORMAT_LINE
    assert free["input_names"] == ["r", "p1", "p2", "t"]
    assert free["n_ops_max"] == 10
    assert free["grammar"]["n_ops_max_declared"] == 8
    assert set(free["roles"]) == {"positive", "signed"}
    assert len(free["initial_structures"]) == 13                       # 13 of 14, inv_r_tmod excluded
    # The (p1,p2) axis became 4 end points + 9 predeclared interior grid points = 13 points.
    assert free["grammar"]["probe"]["n_p_points"] == 13
    assert free["grammar"]["probe"]["n_points"] == 32 * len(FREE_TIMES) * 64 * 13


def test_20b_free_judge_checks_grammar_roles_caps_and_family():
    """Test: free judge checks grammar roles caps and family."""
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    ok, reason = fp.judge_candidate(
        {"expression": "c*p1/(r**2 + d)", "family": "gauss",
         "parameters": {"c": "positive", "d": "positive"}}, free)
    assert ok and reason is None
    for expression, parameters, fragment in [
        ("c*p1*cos(r)", {"c": "positive"}, "call outside expression grammar"),
        ("c*p1/r + q", {"c": "positive"}, "unknown input name q"),
        ("c*p1*u", {"c": "positive"}, "unknown input name u"),
        ("c*p1/r", {"c": "intercept"}, "positive or signed"),
        ("c*p1/r", {"c": "shape"}, "positive or signed"),
    ]:
        ok, reason = fp.judge_candidate(
            {"expression": expression, "family": "gauss", "parameters": parameters}, free)
        assert not ok and fragment in reason, (expression, reason)
    locked = fp.free_constraints("gauss", family_expansion=False, times=FREE_TIMES)
    ok, reason = fp.judge_candidate(
        {"expression": "c*p1/(r**2 + d)", "family": "student_t4",
         "parameters": {"c": "positive", "d": "positive"}}, locked)
    assert not ok and "initial_family" in reason


def test_20b_free_mode_keeps_the_20c_role_object_normalization():
    """Test: free mode keeps the role object normalization."""
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    ok, reason = fp.judge_candidate(
        {"expression": "c*p1/(r**2 + d)", "family": "gauss",
         "parameters": {"c": {"role": "positive"}, "d": {"role": "positive"}}}, free)
    assert ok, reason
    text = json.dumps({"expression": "c*p1/(r**2 + d)", "family": "gauss",
                       "parameters": {"c": {"role": "positive"}, "d": {"role": "positive"}}})
    candidate, ignored = fp.parse_candidate_object(text, free)
    assert ignored["normalized_parameter_shape"] is True
    assert candidate["parameters"] == {"c": "positive", "d": "positive"}


def test_20b_free_parse_names_the_structure_from_the_canonical_id():
    """Test: free parse names the structure from the canonical id."""
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    text = json.dumps({"expression": "c*p1/(r**2 + d)", "family": "gauss",
                       "parameters": {"c": "positive", "d": "positive"}})
    candidate, _ignored = fp.parse_candidate_object(text, free)
    assert candidate["structure"] == "g_" + candidate["structure_model_id"][:12]
    assert candidate["structure"] not in STRUCTURES      # the parser does not register it
    # The same expression gives the same structure name (the basis for duplicate detection by (structure, family)).
    again, _ = fp.parse_candidate_object(text, free)
    assert again["structure"] == candidate["structure"]


def test_20b_free_mock_schedule_is_predeclared_with_expected_outcomes():
    """Test: free mock schedule is predeclared with expected outcomes."""
    assert [expression for expression, _f, _p in fp.FREE_MOCK_SCHEDULE] == [
        "c*p1/(r**2 + d)", "c*p1*log(r - 5)", "p1*c/r**2"]
    assert fp.FREE_MOCK_EXPECTED_OUTCOMES == (
        "active", "nonfinite_on_probe_grid", "duplicate_by_numerical_equivalence")
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    for index in range(len(fp.FREE_MOCK_SCHEDULE)):
        candidate, _ignored = fp.parse_candidate_object(
            fp.free_mock_response_text(index), free)
        assert candidate["expression"] == fp.FREE_MOCK_SCHEDULE[index][0]
    assert fp.MockFreeClient().provider == "mock"


def test_20b_structure_mode_and_constraints_must_agree(tmp_path):
    """Test: structure mode and constraints must agree."""
    context = {"run_id": "r", "world": "gravity", "arm": "111", "policy": "P4", "event": 0,
               "round": 2, "n_designs": 4, "candidate_index": 0, "packet_file": "p.json"}
    catalog = fp.catalog_constraints("gauss")
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    default = fp.Fb1CatalogProposer(tmp_path / "default")
    assert default.structure_mode == "catalog"                  # the catalogue-mode default, unchanged
    assert default.recovery_mode == "catalog_selection"
    assert isinstance(default.client, fp.MockCatalogClient)
    freeform = fp.Fb1CatalogProposer(tmp_path / "free", structure_mode="free")
    assert freeform.recovery_mode == "freeform_discovery"
    assert isinstance(freeform.client, fp.MockFreeClient)
    assert freeform.paid_api_calls == 0 and freeform.transport == "mock"
    with pytest.raises(ValueError, match="does not match the constraints block"):
        freeform.propose_one("packet", context=context, constraints=catalog)
    with pytest.raises(ValueError, match="does not match the constraints block"):
        default.propose_one("packet", context=context, constraints=free)
    with pytest.raises(ValueError, match="structure_mode must be one of"):
        fp.Fb1CatalogProposer(tmp_path / "bad", structure_mode="freeform")


def test_20b_structure_lookups_cover_catalogue_and_free_names():
    """Test: structure lookups cover catalogue and free names."""
    assert fp.structure_expression("inv_r2") == fp.canonical_expression("inv_r2")
    assert fp.structure_coefficient_names("crossover") == ("c1", "c2", "R")
    assert fp.structure_roles("power") == {"c": "positive", "a": "signed"}


def test_20b_archive_optional_keys_carry_the_free_reason_code():
    """Test: archive optional keys carry the free reason code."""
    assert "reason_code" in fp.ARCHIVE_OPTIONAL_KEYS
    assert "numerical_equivalence" in fp.ARCHIVE_OPTIONAL_KEYS
    free_entry = fp._archive_entry({
        "model_id": "m", "expression": "p1*c/r**2", "family": "gauss", "proposed_round": 2,
        "outcome": "duplicate",
        "reason": {"matched_model_id": "inv_r2", "matched_expression": "c*p1/r**2"},
        "reason_code": "duplicate_by_numerical_equivalence"})
    assert free_entry["reason_code"] == "duplicate_by_numerical_equivalence"
    catalog_entry = fp._archive_entry({
        "model_id": "m", "expression": "F", "family": "gauss", "proposed_round": 2,
        "outcome": "active", "reason": {"admitted_round": 2, "active_rounds": [2]}})
    assert "reason_code" not in catalog_entry


def test_20b_free_packet_builds_with_the_free_constraints(tmp_path):
    """The belief packet can also be assembled with the free constraints block (the key check accepts both sets)."""
    rows, members, _catalog, interface = _state()
    free = fp.free_constraints("gauss", times=FREE_TIMES)
    packet = fp.build_belief_packet(rows, members, [], free, interface, 2)
    assert packet["constraints"]["structure_expansion"] is True
    assert packet["constraints"]["n_ops_max"] == 10
    text = fp.serialize_packet(packet)
    assert "nonfinite_on_probe_grid" in text        # the grammar block carries the reason code as well


# ============================================================================ KL scoring
# The wiring of score_set() / score_set_kl() is tested with fake fit results (raw normal random particles);
# SMC is not run (zero paid calls, no posterior fit, as in the other
# sections of this file).


def _fake_fit_29fb(struct, family, n_particles, seed, born_step=0):
    """Fake SMC fit.  The particles are raw N(0,1) draws (the same distribution as samples from the prior,
    but not updated by the likelihood); only the wiring of score_set() / score_set_kl() is tested."""
    import run_gravity as rg
    cand = rg.Candidate(struct, family, born_step)
    k = cand.dim()
    rng = np.random.default_rng(seed)
    u = rg.jnp.asarray(rng.normal(size=(n_particles, k)))
    return cand, u


def _fake_score_inputs_29fb(seed=20260908):
    import run_gravity as rg
    Q = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
    D = len(Q)
    rng = np.random.default_rng(seed)
    truths = rng.normal(1.0, 0.1, size=(D, rg.T_OBS, 2))
    y_held = truths + rng.normal(0.0, rg.SIGMA, size=(D, rg.T_OBS, 2))
    return Q, truths, y_held


def test_29fb_1_gauss_vs_gauss_kl_matches_closed_form():
    """(1) Gaussian vs Gaussian: agrees with the closed-form KL to within 1e-10 (the same formula as _gauss_kl in score_predictions.py)."""
    import run_gravity as rg

    def closed(m0, s0, m1, s1):
        """Closed-form KL divergence between two Gaussians N(m0, s0) and N(m1, s1)."""
        return math.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2.0 * s1 ** 2) - 0.5

    m0, s0, m1, s1 = 1.3, 0.05, 1.5, 0.08
    want = closed(m0, s0, m1, s1)
    got = rg.kl_coordinate(m0, s0, np.array([m1]), np.array([s1]), np.array([1.0]), np.array(["gauss"]))
    assert abs(got["value"] - want) <= 1e-10, (got["value"], want)
    assert got["quadrature_error"] < 1e-8
    assert 0.0 <= got["window_mass_outside"] < 1e-10
    assert got["n_nodes"] == 2 * rg.KL_QUADRATURE_NODES


def test_29fb_2_mixture_kl_matches_fine_riemann_sum():
    """(2) Mixture (2 Gaussian + 1 Student-t) vs Gaussian: agrees with a fine Riemann sum (1e6 points) to within 1e-8."""
    import run_gravity as rg

    truth_q, sigma_world = 2.0, 0.1
    mus = np.array([1.9, 2.15, 2.05]); ss = np.array([0.12, 0.09, 0.15])
    w = np.array([0.4, 0.35, 0.25]); fam = np.array(["gauss", "gauss", "student_t4"])
    got = rg.kl_coordinate(truth_q, sigma_world, mus, ss, w, fam)

    half = 30 * sigma_world
    x = np.linspace(truth_q - half, truth_q + half, 1_000_000)
    dx = x[1] - x[0]
    log_truth = -0.5 * ((x - truth_q) / sigma_world) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(sigma_world)
    terms = []
    for m, s, ww, f in zip(mus, ss, w, fam):
        z = (x - m) / s
        logf = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - math.log(s)) if f == "gauss" \
            else (rg._log_t4_np(z) - math.log(s))
        terms.append(math.log(ww) + logf)
    terms = np.stack(terms, axis=0)
    peak = terms.max(axis=0)
    log_bma = peak + np.log(np.exp(terms - peak).sum(axis=0))
    riemann = float(np.sum(np.exp(log_truth) * (log_truth - log_bma)) * dx)
    assert abs(got["value"] - riemann) <= 1e-8, (got["value"], riemann)


def test_29fb_3_kl_is_near_zero_when_bma_equals_truth():
    """(3) BMA = truth (the true Gaussian injected as a single component) gives KL <= 1e-12."""
    import run_gravity as rg

    m0, s0 = 0.7, 0.03
    got = rg.kl_coordinate(m0, s0, np.array([m0]), np.array([s0]), np.array([1.0]), np.array(["gauss"]))
    assert abs(got["value"]) <= 1e-12, got["value"]


def test_29fb_log_t4_np_matches_fb1_core_log_t4():
    """Checks that the numpy re-implementation for the hot loop (_log_t4_np) computes exactly the same formula as fb1_core._log_t4
    (the premise of (4) "the KL components also use the t density"; fb1_core is a protected file, so it was re-implemented here)."""
    import run_gravity as rg

    zs = np.array([-5.0, -1.3, -0.001, 0.0, 0.4, 2.3, 9.0])
    want = np.asarray(rg.core._log_t4(rg.jnp.asarray(zs)))
    got = rg._log_t4_np(zs)
    assert np.allclose(got, want, atol=1e-13, rtol=0), (got, want)


def test_29fb_4_student_t4_family_uses_t_density_not_gaussian():
    """(4) Family record: a Student-t candidate is recorded as t and its KL component also uses the t density (it is not replaced by a Gaussian).

    The t side has no closed form, so its value itself is also checked with a fine Riemann sum (the same technique as (2))."""
    import run_gravity as rg

    m0, s0, m1, s1 = 1.0, 0.05, 1.1, 0.09
    t4 = rg.kl_coordinate(m0, s0, np.array([m1]), np.array([s1]), np.array([1.0]), np.array(["student_t4"]))
    gauss = rg.kl_coordinate(m0, s0, np.array([m1]), np.array([s1]), np.array([1.0]), np.array(["gauss"]))
    assert abs(t4["value"] - gauss["value"]) > 1e-6, "the student_t4 component is being replaced by a normal one"
    half = 30 * s0
    x = np.linspace(m0 - half, m0 + half, 2_000_000)
    dx = x[1] - x[0]
    log_truth = -0.5 * ((x - m0) / s0) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(s0)
    z = (x - m1) / s1
    log_t = rg._log_t4_np(z) - math.log(s1)
    riemann = float(np.sum(np.exp(log_truth) * (log_truth - log_t)) * dx)
    assert abs(t4["value"] - riemann) <= 1e-8, (t4["value"], riemann)


def test_29fb_score_set_computes_kl_component_summary_and_families():
    """(6) Wiring: score_set() returns kl (value, quadrature_error, window_mass_outside, n_nodes, per_query) and
    (a) component_summary (per-candidate mean, standard deviation, family, weight; float64)."""
    import run_gravity as rg

    Q, truths, y_held = _fake_score_inputs_29fb()
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", 24, seed=1)
    c2, u2 = _fake_fit_29fb("linear", "student_t4", 24, seed=2)
    fits = [(c1, u1), (c2, u2)]
    weights = np.array([0.6, 0.4])
    key = rg.jax.random.PRNGKey(0)
    out = rg.score_set(fits, weights, Q, truths, y_held, key)

    kl = out["kl"]
    assert {"value", "quadrature_error", "window_mass_outside", "n_nodes", "per_query"} <= set(kl)
    assert kl["n_nodes"] == 2 * rg.KL_QUADRATURE_NODES
    assert kl["quadrature_error"] >= 0.0
    assert 0.0 <= kl["window_mass_outside"] < 1e-6
    assert len(kl["per_query"]) == len(Q)
    assert math.isfinite(kl["value"])

    comp = out["_components"]
    assert [row["family"] for row in comp["summary"]] == ["gauss", "student_t4"]
    assert comp["summary"][0]["weight"] == 0.6 and comp["summary"][1]["weight"] == 0.4
    assert np.asarray(comp["summary"][0]["mean"]).shape == (len(Q), rg.T_OBS, 2)
    assert np.asarray(comp["summary"][0]["std"]).shape == (len(Q), rg.T_OBS, 2)
    assert comp["pred_std"].shape == (2, len(Q), rg.T_OBS, 2)
    assert comp["pred_std"].dtype == np.float64
    assert out["_full_particles"] is None            # return_full_particles=False (the default)


def test_29fb_compute_kl_false_skips_kl_for_single_top_model_scoring():
    """Scoring of the single top model is called with compute_kl=False and skips the kl computation (its result is
    not stored in checkpoints[n].single_top_model._keep, so the computation was wasted)."""
    import run_gravity as rg

    Q, truths, y_held = _fake_score_inputs_29fb()
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", 24, seed=3)
    key = rg.jax.random.PRNGKey(1)
    out = rg.score_set([(c1, u1)], np.array([1.0]), Q, truths, y_held, key, compute_kl=False)
    assert out["kl"] is None


def test_29fb_return_full_particles_carries_all_particles_float64_and_nodes():
    """(a) All particle mu values in double precision: returns the original from before subsampling (<=128, float32) unchanged."""
    import run_gravity as rg

    Q, truths, y_held = _fake_score_inputs_29fb()
    n_particles = 150     # more than PRED_COMPONENT_MAX_PARTICLES (128), so it can be told apart from the subsample
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", n_particles, seed=4)
    c2, u2 = _fake_fit_29fb("linear", "gauss", n_particles, seed=5)
    key = rg.jax.random.PRNGKey(2)
    out = rg.score_set([(c1, u1), (c2, u2)], np.array([0.5, 0.5]), Q, truths, y_held, key,
                       return_full_particles=True)
    full = out["_full_particles"]
    assert full["mu"].shape == (2, len(Q), n_particles, rg.T_OBS, 2)
    assert full["mu"].dtype == np.float64
    assert full["s"].shape == (2, n_particles)
    assert list(full["families"]) == ["gauss", "gauss"]
    assert out["_components"]["mu"].shape[1] == min(n_particles, rg.PRED_COMPONENT_MAX_PARTICLES)
    assert out["kl"]["_nodes_x"].shape == (len(Q), rg.T_OBS, 2, 2 * rg.KL_QUADRATURE_NODES)
    assert out["kl"]["_nodes_log_bma"].shape == (len(Q), rg.T_OBS, 2, 2 * rg.KL_QUADRATURE_NODES)


def test_29fb_full_particle_mu_npz_under_cap_writes_raw_particles(tmp_path):
    """write_kl_full_particle_mu: within the 20 MB threshold, the original of every particle is written as is (no substitution)."""
    import run_gravity as rg

    Q, truths, y_held = _fake_score_inputs_29fb()
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", 10, seed=6)
    key = rg.jax.random.PRNGKey(3)
    head = rg.score_set([(c1, u1)], np.array([1.0]), Q, truths, y_held, key,
                        return_full_particles=True)
    full_h = head["_full_particles"]
    fpath = tmp_path / "kl_full_particle_mu_final.npz"
    meta = rg.write_kl_full_particle_mu(
        str(fpath), ["inv_r2|gauss"], np.array([1.0]), full_h, full_h,
        None, None, None, None)          # head is reused in the shift slot too (harmless: this tests the input wiring)
    assert meta["truncated"] is False and meta["written"] is True
    assert meta["n_particles_total"] == 10
    assert fpath.exists()
    with np.load(fpath) as z:
        assert {"mu_head", "mu_shift", "noise_scale_head", "noise_scale_shift"} <= set(z.files)
        assert z["mu_head"].dtype == np.float64


def test_29fb_full_particle_mu_npz_over_cap_falls_back_to_node_values(tmp_path):
    """write_kl_full_particle_mu: when the threshold is forced to be exceeded (by giving a small max_bytes), node values are substituted
    and truncated=True and truncation_reason are disclosed; the size of the substitute file depends on n_nodes, not on the particle count
    (the real 20 MB value is checked separately through KL_FULL_MU_NPZ_MAX_BYTES)."""
    import run_gravity as rg

    assert rg.KL_FULL_MU_NPZ_MAX_BYTES == 20_000_000
    Q, truths, y_held = _fake_score_inputs_29fb()
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", 10, seed=7)
    key = rg.jax.random.PRNGKey(4)
    head = rg.score_set([(c1, u1)], np.array([1.0]), Q, truths, y_held, key,
                        return_full_particles=True)
    full_h = head["_full_particles"]
    node_x, node_lb = head["kl"]["_nodes_x"], head["kl"]["_nodes_log_bma"]
    fpath = tmp_path / "kl_full_particle_mu_final.npz"
    meta = rg.write_kl_full_particle_mu(
        str(fpath), ["inv_r2|gauss"], np.array([1.0]), full_h, full_h,
        node_x, node_lb, node_x, node_lb, max_bytes=1)      # 1 byte: always exceeded
    assert meta["truncated"] is True and meta["written"] is True
    assert "bounded by n_nodes, not particle count" in meta["truncation_reason"]
    assert fpath.exists()
    with np.load(fpath) as z:
        assert {"nodes_x_head", "nodes_log_bma_head", "nodes_x_shift", "nodes_log_bma_shift"} <= set(z.files)
        assert "mu_head" not in z.files


def test_29fb_truth_law_block_records_world_cfg_numeric_parameters():
    """(c) Where truth_law comes from: whether the force-law string, constants, inertia, softening, var_gt and sigma_world of WORLD_CFG[world]
    can all be recorded through the new law_str key (WORLD_CFG itself already had var_gt/inertia/soft/me_probe;
    only law_str is new)."""
    import run_gravity as rg

    for world, cfg in rg.WORLD_CFG.items():
        assert cfg.get("law_str"), f"law_str of {world} is empty"
        for key in ("var_gt", "inertia", "soft", "me_probe"):
            assert key in cfg, f"{world} has no {key}"
    assert "r" in rg.WORLD_CFG["gravity"]["law_str"]


def test_29fb_checkpoint_kl_combines_head_and_shift_by_scalar_count():
    """Checks the combination rule used by main() (combine_checkpoint_kl) against two
    real score_set outputs, head (2 queries) and shift (1 query): the value is the mean weighted by the number of scalars (queries x T_OBS x 2), the error and outside-window mass are the worse of the two."""
    import run_gravity as rg

    Q_h = [(1.0, 1.0), (2.0, 1.0)]
    Q_s = [(3.0, 3.0)]
    rng = np.random.default_rng(11)
    truths_h = rng.normal(1.0, 0.1, size=(len(Q_h), rg.T_OBS, 2))
    truths_s = rng.normal(1.0, 0.1, size=(len(Q_s), rg.T_OBS, 2))
    y_h = truths_h + rng.normal(0.0, rg.SIGMA, size=truths_h.shape)
    y_s = truths_s + rng.normal(0.0, rg.SIGMA, size=truths_s.shape)
    c1, u1 = _fake_fit_29fb("inv_r2", "gauss", 20, seed=8)
    key = rg.jax.random.PRNGKey(5)
    k1, k2 = rg.jax.random.split(key)
    head = rg.score_set([(c1, u1)], np.array([1.0]), Q_h, truths_h, y_h, k1)
    shift = rg.score_set([(c1, u1)], np.array([1.0]), Q_s, truths_s, y_s, k2)

    n_head = len(Q_h) * rg.T_OBS * 2
    n_shift = len(Q_s) * rg.T_OBS * 2
    combined = rg.combine_checkpoint_kl(head["kl"], shift["kl"], n_head, n_shift)
    want = (head["kl"]["value"] * n_head + shift["kl"]["value"] * n_shift) / (n_head + n_shift)
    assert combined["value"] == want
    assert combined["quadrature_error"] == max(head["kl"]["quadrature_error"], shift["kl"]["quadrature_error"])
    assert combined["window_mass_outside"] == max(head["kl"]["window_mass_outside"],
                                                   shift["kl"]["window_mass_outside"])
    assert combined["n_nodes"] == head["kl"]["n_nodes"] == shift["kl"]["n_nodes"]
    assert combined["per_axis"] == {"h": head["kl"], "s": shift["kl"]}
    assert len(combined["per_query_summary"]) == len(Q_h) + len(Q_s)
    assert {q["group"] for q in combined["per_query_summary"][:len(Q_h)]} == {"head"}
    assert {q["group"] for q in combined["per_query_summary"][len(Q_h):]} == {"shift"}
    assert combined["full_particle_mu"] is None


