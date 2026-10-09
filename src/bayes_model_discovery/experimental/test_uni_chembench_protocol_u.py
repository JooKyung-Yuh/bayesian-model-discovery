"""Unified protocol on the ChemBench runner, 0 USD.

1. k = 3 proposals in one call: the request is the selected prompt file plus the composition
   paragraph and the identity names both texts; an FA-C-shaped k-array reply is judged item by
   item (the two FA-C check keys land in ignored_keys), a one-object reply earns the repair turn.
2. The two parts of the joint expected information gain (EIG): joint = structure + parameter from the same draws, the default call
   unchanged; `chembench_core.annotate_eig_endpoints` recomputes an arrival row's own call.
3. The point disagreement on a toy pool against the formula by hand.
4. `design_search.projected_spsa`: an annotator only adds keys, and an EIG-part objective
   (objective_key structure_eig) is what the search climbs, the chooser maximizes and the noise
   floor and the holdout read.
5. Composition paragraph v2 and the archive review paragraph bound to their
   canonical bytes, the composed texts and identities, the review reply (candidates judged as
   usual, revive passed on, no repair for it), the revive vocabulary, a retirement entry's own
   copy of the active rounds, the option refusals and the synthetic mock forms.
6. Two-round smokes per design policy (joint_eig, model_eig, parameter_eig) on the real
   ChemBench oracle with --proposal-limit 3 --composition-paragraph v2 --initial-proposals 12
   --archive-review-rounds 2: the proposer is the real Anthropic transport with an injected
   scripted client (no network, no key, no paid call); three candidates per call, the active pool
   held at 12, one revival valid and two refused; each part arm climbs its own part from the
   joint arm's starts.

Every bank, reply and prompt text here is synthetic; no recorded run response is used.
"""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from . import chembench_core as cc
from . import count_proposer, design_search, positive_inference
from .belief_packet import ARCHIVE_REASON_KEYS, task_interface
from .core_count import seed_for
from .count_inference import Fit
from .count_models import interface_input_names
from .count_proposer import AnthropicCountProposer
from .positive_inference import ScalarHypothesis

REPO = Path(__file__).resolve().parents[3]
#: Captured at import, before a smoke replaces the module attribute with its factory.
REAL_PROPOSER = AnthropicCountProposer
NAMES = ("x0", "x1")
CENTER, SCALE = 0.55, 0.25
SEED = 20261005
FAC_SHA256 = "196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c"
#: sha256(FA-C bytes + COMPOSITION_PARAGRAPH.format(limit=3)), computed 2026-10-05 from the file
#: above; checked only when BMD_FAC_PROMPT names that file.
FAC_K3_COMPOSED_SHA256 = "eee83b13ca0296f46df4f1c0851ec7be9d5510313bb885d92990a046f56a79e8"


def sha(data):
    """Return the hex sha256 digest of bytes or text."""
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. k = 3 in one call
# ---------------------------------------------------------------------------

FIVE_KEY_VARIANT = (
    "# prompt_version: v9.unicb.test\n"
    "Propose one new candidate model for the process.\n"
    'Your whole reply is one raw JSON object with the keys {"expression": ..., "family": ..., '
    '"parameters": {...}, "predicted_diagnostic": {...}, "closest_archived_model": ...}.\n'
)


@pytest.fixture
def variant(tmp_path):
    """Fixture: write the five-key prompt variant to a file, select it, and restore the previous selection afterwards."""
    path = tmp_path / "variant.txt"
    path.write_text(FIVE_KEY_VARIANT)
    previous = count_proposer.set_prompt_variant(path)
    yield path
    count_proposer.set_prompt_variant(previous[0])


def test_k3_request_is_the_selected_file_plus_the_composition_paragraph(variant):
    """The k = 3 request is the selected prompt file plus the composition paragraph, with its own composed digest."""
    text1, identity1 = count_proposer.composed_system_prompt(1)
    text3, identity3 = count_proposer.composed_system_prompt(3)
    assert text1 == FIVE_KEY_VARIANT
    assert identity1 == {"prompt_version": "v9.unicb.test", "prompt_sha256": sha(FIVE_KEY_VARIANT)}
    paragraph = count_proposer.COMPOSITION_PARAGRAPH.format(limit=3)
    assert text3 == FIVE_KEY_VARIANT + paragraph
    # prompt_sha256 keeps naming the FILE; the composed text has its own digest.
    assert identity3 == {**identity1, "proposal_limit": 3, "composed_prompt_sha256": sha(text3)}
    assert count_proposer.composed_prompt_identity(3) == identity3
    # The round-0 burst sends the file plus the burst paragraph whatever the k of the rounds.
    burst_text, burst_identity = count_proposer.composed_system_prompt(burst=True)
    assert burst_text == FIVE_KEY_VARIANT + count_proposer.BURST_PARAGRAPH
    assert "composed_prompt_sha256" not in burst_identity


def test_the_fac_k3_composed_digest_when_the_fac_file_is_supplied():
    """When the FA-C prompt file is supplied, its bytes and k = 3 composed digest match the recorded values."""
    supplied = os.environ.get("BMD_FAC_PROMPT")
    if not supplied:
        pytest.skip("BMD_FAC_PROMPT names the FA-C prompt file; unset")
    path = Path(supplied)
    assert sha(path.read_bytes()) == FAC_SHA256
    previous = count_proposer.set_prompt_variant(path)
    try:
        text, identity = count_proposer.composed_system_prompt(3)
    finally:
        count_proposer.set_prompt_variant(previous[0])
    assert identity == {"prompt_version": "v2.9.FA-C", "prompt_sha256": FAC_SHA256,
                        "proposal_limit": 3, "composed_prompt_sha256": FAC_K3_COMPOSED_SHA256}
    assert text == path.read_text() + count_proposer.COMPOSITION_PARAGRAPH.format(limit=3)


def interface():
    """Return the synthetic positive-rate task interface used by these tests."""
    return task_interface("positive", [{"name": name, "range": [0.1, 1.0]} for name in NAMES],
                          {f"{name}z": f"({name}-{CENTER!r})/{SCALE!r}" for name in NAMES},
                          (0.0, None))


def seed_structure(family="lognormal"):
    """Return the intercept-only seed hypothesis for the given family."""
    return ScalarHypothesis("b0", family, (("b0", "intercept"),), interface_input_names(interface()))


def fac_item(expression, parameters, family="lognormal"):
    """Build one FA-C-shaped candidate item with its two check keys."""
    return {"expression": expression, "family": family, "parameters": parameters,
            "predicted_diagnostic": {"input": "x0", "low": 0.2, "high": 0.6},
            "closest_archived_model": "b0"}


def parse_array(items, cell="101"):
    """Parse a list of candidate items as a k = 3 array reply for the given cell."""
    return count_proposer.parse_candidate_array(
        json.dumps(items), cell, "lognormal", (seed_structure(),), cc.CHEM_ALLOWED_FAMILIES,
        proposal_limit=3)


def test_an_fac_shaped_three_item_array_is_three_judged_candidates():
    """An FA-C-shaped three-item array gives three candidates judged item by item."""
    items = [fac_item("b0+b1*x0z", {"b0": "intercept", "b1": "signed"}),
             fac_item("b0+b1*log(x1)", {"b0": "intercept", "b1": "signed"}),
             fac_item("b0+b1*x0z*x1z", {"b0": "intercept", "b1": "signed"})]
    models, ignored = parse_array(items)
    assert [model.expression for model in models] == [item["expression"] for item in items]
    assert ignored["rejected_candidates"] == []
    # The FA-C check keys are recorded and never reach the hypothesis.
    assert [set(keys) for keys in ignored["ignored_keys"]] == [
        {"predicted_diagnostic", "closest_archived_model"}] * 3
    # Cell 101 (family menu off): a gamma item is refused alone and the other two are kept.
    items[1] = fac_item("b0+b1*log(x1)", {"b0": "intercept", "b1": "signed"}, family="gamma")
    models, ignored = parse_array(items)
    assert [model.expression for model in models] == ["b0+b1*x0z", "b0+b1*x0z*x1z"]
    assert [entry["index"] for entry in ignored["rejected_candidates"]] == [1]


@pytest.mark.parametrize("payload,reason", [
    (fac_item("b0+b1*x0z", {"b0": "intercept", "b1": "signed"}), "reply must be a JSON array of 3"),
    ([fac_item(f"b0+b1*x{i % 2}z", {"b0": "intercept", "b1": "signed"}) for i in range(4)],
     "reply must hold 1..3"),
    ([], "reply must hold 1..3"),
])
def test_a_k3_reply_of_the_wrong_shape_is_one_repair_eligible_failure(payload, reason):
    """A k = 3 reply of the wrong shape or length is refused as one repair-eligible failure."""
    with pytest.raises(ValueError, match=reason):
        parse_array(payload)


# ---------------------------------------------------------------------------
# 2-3. The EIG parts and the point disagreement on synthetic banks
# ---------------------------------------------------------------------------

def diagnostics(response_unit=1.0):
    """Return synthetic fit diagnostics for the test interface."""
    return {"interface": interface(), "response_unit": response_unit, "response_center": 0.0,
            "input_transform": {"center": (CENTER, CENTER), "scale": (SCALE, SCALE)},
            "observation_count": 5, "public_data_sha256": "synthetic-unicb"}


def grammar_fit(expression, values, evidence, family="lognormal", noise=0.4, response_unit=1.0):
    """Build a synthetic Fit for an expression from given particle values and log evidence."""
    used = {node.id for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Name)}
    roles = tuple((name, role) for name, role in
                  (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")) if name in used)
    hypothesis = ScalarHypothesis(expression, family, roles, interface_input_names(interface()))
    particles = {name: np.asarray(values[name], dtype=float) for name, _ in roles}
    size = len(next(iter(particles.values())))
    particles["sigma" if family == "lognormal" else "shape"] = np.full(size, noise)
    return Fit(hypothesis, particles, float(evidence), [float(evidence)] * 2,
               diagnostics(response_unit))


def bank():
    """Build a bank of three synthetic fits with distinct forms and evidences."""
    rng = np.random.default_rng(7)
    return [grammar_fit("b0+b1*x0z", {"b0": rng.normal(0, .3, 16), "b1": rng.normal(.8, .2, 16)}, -3.1),
            grammar_fit("b0+b1*x1z", {"b0": rng.normal(0, .3, 16), "b1": rng.normal(-.6, .2, 16)}, -2.6),
            grammar_fit("b0+b1*x0z*x1z+b2*x1z", {"b0": rng.normal(0, .3, 16), "b1": rng.normal(0, .5, 16),
                                                 "b2": rng.normal(.2, .2, 16)}, -2.9)]


def points(count=6):
    """Return a reproducible array of random design points inside the input box."""
    return np.random.default_rng(100 + count).uniform(0.1, 1.0, size=(count, len(NAMES)))


def test_joint_is_structure_plus_parameter_from_the_same_draws():
    """The joint EIG equals the structure part plus the parameter part, and the default call is unchanged."""
    fits, x = bank(), points()
    plain = positive_inference.joint_eig(fits, x, SEED, 64)
    parts = positive_inference.joint_eig(fits, x, SEED, 64, decompose=True)
    # The default call is unchanged and decompose=True moves neither number.
    assert set(plain) == {"eig", "mc_standard_error", "outer_samples", "negative_estimates_clipped",
                          "target", "seed"}
    assert np.array_equal(plain["eig"], parts["eig"])
    assert np.array_equal(plain["mc_standard_error"], parts["mc_standard_error"])
    gap = np.abs(parts["eig"] - (parts["structure_eig"] + parts["parameter_eig"]))
    assert gap.max() <= 1e-12
    assert np.all(parts["structure_mc_standard_error"] >= 0)
    assert np.all(parts["parameter_mc_standard_error"] >= 0)
    # One model: no structure information, the whole joint value is the parameter part.
    one = positive_inference.joint_eig(fits[:1], x, SEED, 64, decompose=True)
    assert np.all(one["structure_eig"] == 0.0)
    assert np.allclose(one["parameter_eig"], one["eig"], rtol=0, atol=1e-15)


def arrival_rows(fits, x, seed=SEED, samples=32):
    """Rows as the ascent objective writes its single-point arrival calls."""
    rows = []
    for index, point in enumerate(x):
        table = positive_inference.joint_eig(fits, point[None, :], seed, samples)
        rows.append({"design": point.tolist(), "eig": float(table["eig"][0]),
                     "mc_standard_error": float(table["mc_standard_error"][0]),
                     "unit_design": [0.5, 0.5], "phase": "arrival", "step": 20,
                     "restart_index": index + 1, "objective_seed": seed})
    return rows


def test_annotation_recomputes_the_objective_call_and_adds_only_new_keys():
    """Annotating arrival rows recomputes the objective's own call, adds only new keys, and refuses tampered rows."""
    fits, x = bank(), points(5)
    rows = arrival_rows(fits, x)
    rows.append({"design": [0.3, 0.3], "invalid": True,
                 "invalid_reason": cc.ACQUISITION_INVALID_REASON, "objective_seed": SEED})
    before = json.loads(json.dumps(rows))
    cc.annotate_eig_endpoints(fits, rows, 32)
    assert rows[-1] == before[-1]
    for row, old, point in zip(rows[:-1], before[:-1], x, strict=True):
        assert {key: row[key] for key in old} == old
        assert set(row) - set(old) == {*cc.EIG_COMPONENT_COLUMNS, "point_disagreement"}
        parts = positive_inference.joint_eig(fits, point[None, :], SEED, 32, decompose=True)
        for column in cc.EIG_COMPONENT_COLUMNS:
            assert row[column] == float(parts[column][0])
        assert abs(row["eig"] - (row["structure_eig"] + row["parameter_eig"])) <= 1e-12
        assert row["point_disagreement"] == float(positive_inference.point_disagreement(
            fits, point[None, :])[0])
    # A row whose joint value is not the objective's own call is refused, never silently re-labelled.
    tampered = arrival_rows(fits, x[:1])
    tampered[0]["eig"] += 1e-9
    with pytest.raises(ValueError, match="recomputation differs"):
        cc.annotate_eig_endpoints(fits, tampered, 32)


CONSTANT_FORMS = ("b0", "b0+b1*x0z", "b0+b1*x1z")


def constant_fit(form, means, evidence, noise=0.4):
    """A model whose particle k predicts the constant mean means[k] everywhere (log link).

    Distinct forms keep distinct canonical identities; every slope particle is 0, so the mean
    is exp(b0) at every design whichever form carries it.
    """
    means = np.asarray(means, dtype=float)
    return grammar_fit(form, {"b0": np.log(means), "b1": np.zeros(len(means))}, evidence, noise=noise)


def test_point_disagreement_is_the_evidence_weighted_variance_of_the_model_means():
    # Particle-averaged means 2, 4 and 8 with normalized evidence weights 0.5, 0.3 and 0.2.
    """Point disagreement is the evidence-weighted variance of the model means, and zero for equal means."""
    fits = [constant_fit(CONSTANT_FORMS[0], [1.0, 3.0], np.log(0.5)),
            constant_fit(CONSTANT_FORMS[1], [4.0, 4.0], np.log(0.3)),
            constant_fit(CONSTANT_FORMS[2], [8.0, 8.0], np.log(0.2))]
    x = points(3)
    mean = 0.5 * 2 + 0.3 * 4 + 0.2 * 8
    expected = 0.5 * (2 - mean) ** 2 + 0.3 * (4 - mean) ** 2 + 0.2 * (8 - mean) ** 2
    rows = arrival_rows(fits, x)
    cc.annotate_eig_endpoints(fits, rows, 32)
    assert [row["point_disagreement"] for row in rows] == pytest.approx([expected] * 3, rel=1e-12)
    assert expected == pytest.approx(5.16, rel=1e-12)
    # Equal means, different dispersion: zero, whatever the dispersion.
    flat = [constant_fit(CONSTANT_FORMS[0], [3.0, 3.0], np.log(0.5), noise=0.2),
            constant_fit(CONSTANT_FORMS[1], [3.0, 3.0], np.log(0.5), noise=1.5)]
    rows = arrival_rows(flat, x)
    cc.annotate_eig_endpoints(flat, rows, 32)
    assert [row["point_disagreement"] for row in rows] == pytest.approx([0.0] * 3, abs=1e-12)


def test_point_disagreement_runs_over_the_models_the_acquisition_guard_keeps():
    # The third model's mean overflows everywhere, so the guard drops it at every design and both
    # the EIG and the disagreement are over the first two, weights renormalized (0.625, 0.375).
    """Point disagreement and EIG are computed over the models the acquisition guard keeps, weights renormalized."""
    fits = [constant_fit(CONSTANT_FORMS[0], [1.0, 3.0], np.log(0.5)),
            constant_fit(CONSTANT_FORMS[1], [4.0, 4.0], np.log(0.3)),
            grammar_fit(CONSTANT_FORMS[2], {"b0": [800.0, 800.0], "b1": [0.0, 0.0]}, np.log(0.2))]
    x = points(2)
    kept = fits[:2]
    rows = arrival_rows(kept, x)  # the objective's joint_eig ran on the guarded bank
    cc.annotate_eig_endpoints(fits, rows, 32)
    mean = 0.625 * 2 + 0.375 * 4
    expected = 0.625 * (2 - mean) ** 2 + 0.375 * (4 - mean) ** 2
    assert [row["point_disagreement"] for row in rows] == pytest.approx([expected] * 2, rel=1e-12)
    assert expected == pytest.approx(0.9375, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. The design search on an EIG part
# ---------------------------------------------------------------------------

def box():
    """Return the unit-square search transform used by the design-search tests."""
    return design_search.SearchTransform(("x", "y"), ((0.0, 1.0), (0.0, 1.0)), ())


def synthetic_objective(points_, seed):
    """Synthetic joint-EIG objective with a known maximum and seeded noise."""
    values = 1.0 - 2.0 * (points_[:, 0] - 0.62) ** 2 - 0.8 * (points_[:, 1] - 0.31) ** 2
    errors = [np.random.default_rng(seed_for("unicb_synthetic", seed, p.tolist())).normal(0, .05)
              for p in points_]
    return [{"design": p.tolist(), "eig": float(v + e), "mc_standard_error": 0.05}
            for p, v, e in zip(points_, values, errors, strict=True)]


def synthetic_part_objective(points_, seed):
    """The joint rows plus a structure part whose maximum sits at the corner (1, 0), far from the
    joint's, with its own seed-dependent noise; the parameter part is joint - structure."""
    rows = synthetic_objective(points_, seed)
    for row in rows:
        x, y = row["design"]
        rng = np.random.default_rng(seed_for("unicb_synthetic_part", seed, row["design"]))
        noise = rng.normal(0, .01)
        structure = float(x * (1 - y) + noise)
        row.update(structure_eig=structure, structure_mc_standard_error=0.01,
                   parameter_eig=row["eig"] - structure, parameter_mc_standard_error=0.05)
    return rows


def synthetic_annotator(rows):
    """A record-only column added to the arrival rows."""
    for row in rows:
        if row.get("invalid") is not True:
            x, y = row["design"]
            row.update(structure_eig=float(x * (1 - y)), structure_mc_standard_error=0.01)


def search(objective=synthetic_objective, **options):
    """Run the projected SPSA search on the synthetic box with the given objective and options."""
    return design_search.projected_spsa(
        objective, box(), global_seed=11, cell="101", round_index=1,
        acquisition_seed=seed_for("unicb_acquisition", 11, 1), acquisition_stream="unicb_acquisition",
        **options)


def without(record, keys):
    """Return a JSON copy of a search record with the given keys removed from its rows."""
    data = json.loads(json.dumps(record))
    for name in ("rows", "arrival_rows"):
        data[name] = [{k: v for k, v in row.items() if k not in keys} for row in data[name]]
    return data


def test_an_annotator_adds_keys_and_moves_nothing_else():
    """An arrival annotator only adds keys and leaves the rest of the search record unchanged."""
    plain = search()
    annotated = search(arrival_annotator=synthetic_annotator)
    added = {"structure_eig", "structure_mc_standard_error"}
    assert all(added <= set(row) for row in annotated["arrival_rows"])
    assert without(annotated, added) == json.loads(json.dumps(plain))


def test_an_eig_part_objective_is_climbed_chosen_and_held_out_on_its_own_column():
    """An EIG-part objective is climbed, chosen on, and held out on its own column."""
    joint = search(synthetic_part_objective)
    part = search(synthetic_part_objective, objective_key="structure_eig")
    # The extra columns change nothing in a joint search.
    plain = search()
    assert joint["trajectories"] == plain["trajectories"] and joint["chosen"] == plain["chosen"]
    # Same restart seeds, so the same five starts; the part search then climbs the part.
    assert ([t["start_unit"] for t in part["trajectories"]]
            == [t["start_unit"] for t in joint["trajectories"]])
    assert part["trajectories"] != joint["trajectories"]
    starts = {row["restart_index"]: row for row in part["rows"] if row["phase"] == "start"}
    for trajectory, arrival in zip(part["trajectories"], part["arrival_rows"], strict=True):
        assert trajectory["value_climbed"] == (arrival["structure_eig"]
                                               - starts[arrival["restart_index"]]["structure_eig"])
    assert np.mean([row["design"][0] for row in part["arrival_rows"]]) > np.mean(
        [row["design"][0] for row in joint["arrival_rows"]])
    # The chooser, its diagnostics, the noise floor and the holdout read the part.
    best = max(part["arrival_rows"],
               key=lambda row: (row["structure_eig"], [-v for v in row["design"]]))
    assert part["chosen"] == best["design"]
    policy = part["policy"]
    assert policy["objective_column"] == "structure_eig"
    assert policy["chosen_value"] == best["structure_eig"]
    assert policy["selection_standard_error_column"] == "structure_mc_standard_error"
    assert policy["selected_mc_se"] == best["structure_mc_standard_error"]
    assert "selection_standard_error_column" not in joint["policy"]
    check_a = [row["structure_eig"] for row in part["rows"] if row["phase"] in ("start", "arrival")]
    check_b = [row["structure_eig"] for row in part["rows"] if row["phase"] == "noise_floor"]
    assert part["noise_floor_estimates"] == design_search.noise_floor_estimates(check_a, check_b)
    chosen_index = part["chosen_restart_index"] - 1
    assert part["holdout_value"] == part["holdout_rows"][chosen_index]["structure_eig"]
    assert part["endpoint_holdout_values"] == [row["structure_eig"] for row in part["holdout_rows"]]


def test_the_part_objective_and_annotation_refusals():
    """Invalid part objectives and annotators that rewrite existing keys are refused."""
    with pytest.raises(ValueError, match="objective must be eig, epig or an EIG part"):
        search(synthetic_part_objective, objective_key="model_eig")
    with pytest.raises(ValueError, match="must return finite values"):
        search(synthetic_objective, objective_key="structure_eig")

    def invalid_with_part_error(points_, seed):
        """Synthetic objective returning only invalid rows that carry a part standard error."""
        return [{"design": p.tolist(), "invalid": True,
                 "invalid_reason": design_search.INVALID_REASONS[0],
                 "structure_mc_standard_error": 0.01} for p in points_]

    with pytest.raises(ValueError, match="invalid row must carry"):
        search(invalid_with_part_error, objective_key="structure_eig")

    def rewrites(rows):
        """Annotator that illegally rewrites an existing key."""
        rows[0]["eig"] = 0.0

    with pytest.raises(ValueError, match="only add keys"):
        search(arrival_annotator=rewrites)


# ---------------------------------------------------------------------------
# Command-line refusals (before any output exists) and the recorded names
# ---------------------------------------------------------------------------

def run_args(output, **overrides):
    """Build a runner argument namespace for the ChemBench task with optional overrides."""
    base = {"manifest": str(REPO / "manifests" / "chembench_paper_manifest.json"),
            "task": "c2_product_inhibition__medium__v0", "output": str(output), "seed": 31,
            "rounds": 2, "particles": 16, "q": 8, "noise": 0.01, "family": "lognormal",
            "structure_proposals": None, "proposer": "mock", "llm_model": None,
            "allow_paid_api": False, "prior_template": "default", "cell": "101", "profile": "dev",
            "d0_policy": "fixed3", "d0_fixed_inputs": None, "initial_catalogue": "off",
            "initial_proposals": 12}
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("policy", ("structure_eig", "parameter_eig", "model_eig"))
def test_an_eig_part_needs_an_acquisition_bit_and_the_ascent(tmp_path, policy):
    """An EIG-part policy is refused, before any output, on cells without an acquisition bit or the ascent."""
    out = tmp_path / "a0" / "run"
    with pytest.raises(ValueError, match="needs an acquisition bit"):
        cc.run(run_args(out, cell="100", design_policy=policy))
    assert not out.exists()
    out = tmp_path / "pool" / "run"
    with pytest.raises(ValueError, match="not wired on this positive-rate path's pool search"):
        cc.run(run_args(out, cell="101", design_policy=policy))
    assert not out.exists()
    out = tmp_path / "epig" / "run"
    with pytest.raises(ValueError, match="epig requires cell"):
        cc.run(run_args(out, cell="102", design_policy=policy))
    assert not out.exists()


def test_the_cli_names_and_the_recorded_names():
    """The command-line policy names, aliases and recorded variant names are as expected."""
    assert set(cc.CHEM_EIG_PART_POLICIES) == {"structure_eig", "parameter_eig"}
    assert cc.CHEM_EIG_PART_ALIASES == {"model_eig": "structure_eig"}
    # The joint arm keeps today's executable list; the parts are listed only on their own cells.
    assert cc.CHEM_DESIGN_POLICIES == (cc.DEFAULT_POLICY, "epig", "point_disagreement", "space_filling")
    assert cc.CHEM_EIG_PART_POLICIES["structure_eig"][2] == cc.TARGET["structure_eig"]
    assert cc.eig_part_points_variant("free", "default", "structure_eig") == cc._points_variant(
        "free", "default", "structure_eig", True) == "design_structure_eig+structure_free"
    assert cc.eig_part_points_variant("free", "default", "parameter_eig") == (
        "design_parameter_eig+structure_free")


@pytest.mark.usefixtures("require_autoscilab_source")
def test_the_ascent_declaration_names_the_objective_column_only_on_a_part():
    """The ascent declaration names an objective column only for an EIG-part policy."""
    assert design_search.EIG_PART_OBJECTIVES == {
        policy: columns[1] for policy, columns in cc.CHEM_EIG_PART_POLICIES.items()}
    assert all(columns[0] == policy for policy, columns in cc.CHEM_EIG_PART_POLICIES.items())
    for policy in ("structure_eig", "parameter_eig"):
        settings = design_search.declared_settings(
            SimpleNamespace(cell="101", design_policy=policy, design_search="ascent"),
            cc.ACQUISITION_STREAM)
        assert settings["design_search_rule"]["objective_column"] == policy
        assert (settings["design_search_rule"]["objective_standard_error_column"]
                == cc.CHEM_EIG_PART_POLICIES[policy][1])
        with pytest.raises(ValueError, match="requires an active joint_eig or epig cell"):
            design_search.declared_settings(
                SimpleNamespace(cell="100", design_policy=policy, design_search="ascent"),
                cc.ACQUISITION_STREAM)
    joint = design_search.declared_settings(
        SimpleNamespace(cell="101", design_policy="joint_eig", design_search="ascent"),
        cc.ACQUISITION_STREAM)
    assert set(joint["design_search_rule"]) == {
        "restarts", "min_steps", "max_steps", "start_distribution", "clip", "chooser_candidates",
        "candidate_exclusion_policy"}



# ---------------------------------------------------------------------------
# 5. Composition paragraph v2, archive review rounds, synthetic mock
# ---------------------------------------------------------------------------

V2_FILE_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
REVIEW_FILE_SHA256 = "8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde"


def test_the_v2_and_review_paragraphs_are_the_canonical_file_bytes():
    """Paragraph v2 and the archive review paragraph match their canonical file bytes and digests."""
    v2, review = (count_proposer.COMPOSITION_PARAGRAPH_V2.encode(),
                  count_proposer.ARCHIVE_REVIEW_PARAGRAPH.encode())
    assert len(v2) == 525 and v2.endswith(b"\n") and v2.count(b"{limit}") == 1
    assert len(review) == 675 and review.endswith(b"\n") and review.count(b"{limit}") == 1
    assert sha(v2) == count_proposer.COMPOSITION_PARAGRAPH_V2_SHA256 == V2_FILE_SHA256
    assert sha(review) == count_proposer.ARCHIVE_REVIEW_PARAGRAPH_SHA256 == REVIEW_FILE_SHA256
    for variable, raw in (("BMD_COMPOSITION_V2", v2), ("BMD_ARCHIVE_REVIEW_V1", review)):
        supplied = os.environ.get(variable)
        if supplied:
            assert Path(supplied).read_bytes() == raw


@pytest.fixture
def composition_v2(variant):
    """Fixture: select composition paragraph v2 and restore the previous selection afterwards."""
    previous = count_proposer.set_composition_paragraph("v2")
    yield
    count_proposer.set_composition_paragraph(previous)


def test_v1_stays_the_default_and_v2_composes_three_candidates(variant):
    """Paragraph v1 stays the default; v2 composes three candidates, with and without the review paragraph."""
    v1_text, v1_identity = count_proposer.composed_system_prompt(3)
    assert v1_text == FIVE_KEY_VARIANT + count_proposer.COMPOSITION_PARAGRAPH.format(limit=3)
    assert set(v1_identity) == {"prompt_version", "prompt_sha256", "proposal_limit",
                                "composed_prompt_sha256"}
    previous = count_proposer.set_composition_paragraph("v2")
    try:
        text, identity = count_proposer.composed_system_prompt(3)
        review_text, review_identity = count_proposer.composed_system_prompt(3, archive_review=True)
        with pytest.raises(ValueError, match="written for three candidates"):
            count_proposer.composed_system_prompt(2)
        with pytest.raises(ValueError, match="archive review round is a k > 1 request"):
            count_proposer.composed_system_prompt(1, archive_review=True)
        # The burst and the k = 1 request are untouched by the selection.
        assert count_proposer.composed_system_prompt(1)[0] == FIVE_KEY_VARIANT
        assert count_proposer.composed_system_prompt(burst=True)[0] == (
            FIVE_KEY_VARIANT + count_proposer.BURST_PARAGRAPH)
    finally:
        count_proposer.set_composition_paragraph(previous)
    v2 = count_proposer.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3")
    review = count_proposer.ARCHIVE_REVIEW_PARAGRAPH.replace("{limit}", "3")
    assert text == FIVE_KEY_VARIANT + v2
    assert review_text == FIVE_KEY_VARIANT + v2 + review
    assert identity == {**identity1(v1_identity), "proposal_limit": 3, "composed_prompt_sha256": sha(text),
                        "composition_paragraph_version": "v2",
                        "composition_paragraph_sha256": V2_FILE_SHA256}
    assert review_identity == {**identity, "composed_prompt_sha256": sha(review_text),
                               "archive_review_paragraph_version": "v1",
                               "archive_review_paragraph_sha256": REVIEW_FILE_SHA256,
                               "archive_review_prompt_sha256": sha(review_text)}
    with pytest.raises(ValueError, match="archive review round is a k > 1 request"):
        count_proposer.composed_system_prompt(3, archive_review=True)  # v1 selected again
    with pytest.raises(ValueError, match="composition paragraph must be one of"):
        count_proposer.set_composition_paragraph("v3")


def identity1(identity):
    """Return the prompt version and prompt digest keys of an identity record."""
    return {key: identity[key] for key in ("prompt_version", "prompt_sha256")}


def item(expression):
    """Build a lognormal candidate item for an expression, with roles for the parameters it uses."""
    return {"expression": expression, "family": "lognormal",
            "parameters": {name: ("intercept" if name == "b0" else "signed")
                           for name in ("b0", "b1", "b2") if name in expression}}


THREE = [item("b0+b1*x0z"), item("b0+b1*x1z"), item("b0+b1*x0z*x1z")]


def review_parse(payload):
    """Parse a payload as an archive review reply for the test cell."""
    return count_proposer.parse_review_reply(
        json.dumps(payload), "101", "lognormal", (seed_structure(),), cc.CHEM_ALLOWED_FAMILIES,
        proposal_limit=3)


@pytest.mark.parametrize("payload,revive,missing,shape", [
    ({"candidates": THREE, "revive": ["m1", "m2"]}, ["m1", "m2"], False, "object"),
    ({"candidates": THREE, "revive": []}, [], False, "object"),
    ({"candidates": THREE}, [], True, "object"),
    ({"candidates": THREE, "revive": "m1"}, [], True, "object"),
    (THREE, [], True, "array"),
])
def test_a_review_reply_parses_its_candidates_and_passes_the_revive_list_on(payload, revive, missing, shape):
    """A review reply's candidates are parsed and its revive list is passed on, with a note when it is missing."""
    models, ignored = review_parse(payload)
    assert [model.expression for model in models] == [entry["expression"] for entry in THREE]
    assert ignored["revive_requested"] == revive and ignored["revive_missing"] is missing
    assert ignored["review_reply_shape"] == shape
    assert ("revive_note" in ignored) is missing


@pytest.mark.parametrize("payload,reason", [
    ({"revive": ["m1"]}, "reply must be one JSON object"),
    ("text", "reply must be one JSON object"),
    ({"candidates": THREE[:1] * 4, "revive": []}, "reply must hold 1..3"),
])
def test_a_review_reply_without_judgeable_candidates_earns_the_repair_turn(payload, reason):
    """A review reply without judgeable candidates raises the repair-eligible error."""
    with pytest.raises(ValueError, match=reason):
        review_parse(payload)


def test_an_object_reply_on_an_ordinary_k3_round_earns_the_repair_turn():
    """An object reply on an ordinary k = 3 round raises the repair-eligible error."""
    with pytest.raises(ValueError, match="reply must be a JSON array of 3"):
        parse_array({"candidates": THREE, "revive": []})


def test_a_review_call_sends_the_review_text_and_records_the_revive_list(tmp_path, composition_v2):
    """A review call sends the review text and records the revive list and digests."""
    reply = {"candidates": THREE, "revive": ["m1"]}
    calls = []

    def create(**body):
        """Fake Messages API create call that records the request body and returns the scripted review reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(reply))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id="unicb-review", model="unicb-fake", stop_reason="end_turn")

    proposer = REAL_PROPOSER(tmp_path / "records", client=SimpleNamespace(
        messages=SimpleNamespace(create=create)))
    models, record = proposer.propose_one(
        '{"packet": 1}', context={"run_id": "r", "world_seed": 31, "cell": "101",
                                  "proposal_rep": 0, "round": 5},
        initial_family="lognormal", initial_structures=(seed_structure(),),
        allowed_families=cc.CHEM_ALLOWED_FAMILIES, max_rounds=30, proposal_limit=3, archive_review=True)
    assert len(models) == 3 and record["calls"] == 1 and record["repairs"] == 0
    review_text = count_proposer.composed_system_prompt(3, archive_review=True)[0]
    assert calls[0]["system"] == review_text
    assert record["composed_prompt_sha256"] == sha(review_text)
    assert record["archive_review_paragraph_sha256"] == REVIEW_FILE_SHA256
    assert record["revive_requested"] == ["m1"] and record["revive_missing"] is False


def archive_entry(model_id, outcome):
    """Build a synthetic archive entry with the reason fields of the given outcome."""
    reason = {"retired_by_evidence": {"log_evidence_at_retirement": -3.0, "gap_to_cutoff_nats": 1.0,
                                      "active_rounds": [1], "computed_round": 2, "n_obs": 5},
              "active": {"admitted_round": 1, "active_rounds": [1]},
              "rejected": {"validation_message": "no"}}[outcome]
    return {"model_id": model_id, "expression": "b0", "family": "lognormal", "parameters": {},
            "proposed_round": 1, "outcome": outcome, "reason": reason}


def test_revive_validation_vocabulary():
    """Revive requests are validated into valid ids and refusals (active, over_limit, unknown, not_retired, repeated)."""
    archive = [archive_entry("r1", "active"), archive_entry("r1", "retired_by_evidence"),
               archive_entry("r2", "active"), archive_entry("r2", "retired_by_evidence"),
               archive_entry("a1", "active"), archive_entry("x1", "rejected")]
    retired_now = {"r1": None, "r2": None}
    check = lambda request: cc.validate_revive_request(request, archive, retired_now, {"a1"})
    assert check(["r1", "r2"]) == (["r1", "r2"], [])
    assert check(["a1", "r1", "r2"]) == (["r1"], [{"id": "a1", "why": "active"},
                                                  {"id": "r2", "why": "over_limit"}])
    assert check(["zz", "x1"]) == ([], [{"id": "zz", "why": "unknown"}, {"id": "x1", "why": "not_retired"}])
    assert check([7, "r1"]) == (["r1"], [{"id": 7, "why": "unknown"}])
    assert check(["r1", "r1"]) == (["r1"], [{"id": "r1", "why": "repeated"}])
    assert check("r1") == ([], []) and check([]) == ([], [])
    # The synthetic mock names the best retired model (valid) and the pool's best (active).
    assert cc.mock_revive_request(archive, retired_now, "a1") == ["r1", "a1"]


def test_a_retirement_entry_keeps_its_own_copy_of_the_active_rounds():
    """A revived model's running list of active rounds grows again; its earlier entry must not."""
    running = [0]
    reason = cc.retirement_reason(np.float64(-8.3), -8.2, running, 1, 4)
    running.extend([5, 6])
    assert reason["active_rounds"] == [0]
    assert list(reason) == ["log_evidence_at_retirement", "gap_to_cutoff_nats", "active_rounds",
                            "computed_round", "n_obs"]
    assert set(reason) == ARCHIVE_REASON_KEYS["retired_by_evidence"]
    assert type(reason["log_evidence_at_retirement"]) is float
    assert reason["gap_to_cutoff_nats"] == -8.2 - float(np.float64(-8.3))
    assert (reason["computed_round"], reason["n_obs"]) == (1, 4)


@pytest.mark.parametrize("overrides,reason", [
    ({"composition_paragraph": "v2", "proposal_limit": 2}, "written for --proposal-limit 3"),
    ({"composition_paragraph": "v3", "proposal_limit": 3}, "must be v1 or v2"),
    ({"archive_review_rounds": "2", "proposal_limit": 3}, "needs --composition-paragraph v2"),
    ({"archive_review_rounds": "1,3", "composition_paragraph": "v2", "proposal_limit": 3},
     "round list entries must be 1..2"),
    ({"mock_schedule": "synthetic_v2"}, "--mock-schedule must be one of"),
    ({"mock_schedule": "synthetic_v1", "proposer": "anthropic", "allow_paid_api": True},
     "needs --proposer mock"),
    ({"archive_review_rounds": "2", "composition_paragraph": "v2", "proposal_limit": 3,
      "cell": "001", "initial_proposals": None, "initial_catalogue": "on"},
     "need a structure-expansion cell"),
])
def test_the_u3_options_are_refused_before_any_output(tmp_path, overrides, reason, request):
    """Invalid composition and archive-review options are refused before any output exists."""
    if reason == "need a structure-expansion cell":
        # This refusal is raised after the runner has imported the ChemBench oracle.
        request.getfixturevalue("require_autoscilab_source")
    out = tmp_path / "refused" / "run"
    args = run_args(out, **{key: value for key, value in overrides.items() if value is not None})
    if overrides.get("initial_proposals", 0) is None:
        del args.initial_proposals
    try:
        with pytest.raises(ValueError, match=reason):
            cc.run(args)
    finally:
        count_proposer.set_composition_paragraph("v1")
        count_proposer.set_prompt_variant(None)
    assert not out.exists()


def test_the_synthetic_mock_forms_are_distinct_and_cover_the_burst_and_twenty_rounds():
    """The synthetic mock forms are distinct and cover the initial burst and twenty rounds, with planted duplicates."""
    names = [f"X{i}" for i in range(1, 8)]
    bounds = dict(zip(names, ([0.01, 100.0], [0.0, 50.0], [0.01, 100.0], [0.0, 20.0],
                              [0.01, 10.0], [278.0, 368.0], [4.0, 10.0]), strict=True))
    forms = cc.synthetic_mock_forms(names, bounds)
    assert len(forms) == len(set(forms)) == 96
    assert forms[:7] == tuple(f"b0+b1*X{i}z" for i in range(1, 8))
    assert all("log(X2)" not in form and "log(X4)" not in form for form in forms)
    assert cc.SYNTHETIC_ROUND_BASE + 3 * 20 <= len(forms)
    seen = set(forms[:cc.SYNTHETIC_ROUND_BASE])
    planted = 0
    for t in range(1, 21):
        this_round = []
        for slot in (1, 2, 3):
            form = cc.synthetic_round_form(forms, t, slot, this_round)
            if slot in cc.SYNTHETIC_PLANTED_DUPLICATES.get(t, {}):
                planted += 1
                assert form in seen or form in this_round
            else:
                assert form not in seen and form not in this_round
            this_round.append(form)
        seen.update(this_round)
    assert planted == 3


# ---------------------------------------------------------------------------
# 6. Two-round smokes per design policy, real oracle, scripted client
# ---------------------------------------------------------------------------

class ScriptedClient:
    """The Messages API stand-in for the smoke: deterministic replies, no network, no key.

    The burst's twelve calls each get one object; an ordinary k = 3 round gets an array of three
    new forms; the review round (2) gets the object reply with three candidates - the third
    repeating the first (a same-round duplicate) - and a revive list naming the best retired
    model (valid), a pool member (active) and an unknown id (over_limit, being third).
    """

    def __init__(self):
        """Create the client with its lock, recorded bodies and call counter."""
        self.lock = threading.Lock()
        self.bodies = []
        self.used = 0
        self.messages = SimpleNamespace(create=self.create)

    @staticmethod
    def forms(names):
        """Return the single and pairwise linear forms over the given input names."""
        z = [f"{name}z" for name in names]
        singles = [f"b0+b1*{zi}" for zi in z]
        pairs = [f"b0+b1*{z[i]}+b2*{z[j]}" for i in range(len(z)) for j in range(i + 1, len(z))]
        return singles + pairs

    def create(self, **body):
        """Fake Messages API create call that returns the scripted reply for the request kind."""
        packet = json.loads(body["messages"][0]["content"])
        forms = self.forms([entry["name"] for entry in packet["interface"]["inputs"]])
        review = body["system"].endswith(count_proposer.ARCHIVE_REVIEW_PARAGRAPH.replace("{limit}", "3"))
        k3 = review or body["system"].endswith(count_proposer.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3"))
        with self.lock:
            self.bodies.append(body)
            if not k3:
                reply = item(forms[self.used])
                self.used += 1
            else:
                chosen = forms[self.used:self.used + 3]
                self.used += 3
                if review:
                    chosen[2] = chosen[0]
                    latest = {}
                    for entry in packet["archive"]:
                        latest.setdefault(entry["model_id"], []).append(entry["outcome"])
                    retired = sorted(model_id for model_id, outcomes in latest.items()
                                     if outcomes[-1] == "retired_by_evidence"
                                     and "active" not in outcomes[outcomes.index("retired_by_evidence"):])
                    reply = {"candidates": [item(form) for form in chosen],
                             "revive": [retired[0], packet["pool"][0]["model_id"], "unknown_model_id"]}
                else:
                    reply = [item(form) for form in chosen]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(reply))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id=f"unicb-scripted-{len(self.bodies)}", model="unicb-scripted", stop_reason="end_turn")


ARMS = {"joint_eig": (), "model_eig": ("--design-policy", "model_eig"),
        "parameter_eig": ("--design-policy", "parameter_eig")}


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """One two-round dev-profile run per arm, in this process, sharing the round-0 pool.

    The per-test `require_autoscilab_source` pins the oracle's source; a module fixture cannot
    use it, so it only skips when the oracle is absent altogether (the same rule as the engine-guard tests).
    """
    if not os.environ.get("BMD_AUTOSCILAB") or importlib.util.find_spec("autoscilab") is None:
        pytest.skip("the smoke needs the ChemBench oracle (BMD_AUTOSCILAB on PYTHONPATH)")
    root = tmp_path_factory.mktemp("unicb_smoke")
    variant_file = root / "variant.txt"
    variant_file.write_text(FIVE_KEY_VARIANT)
    runs = {}
    monkeypatch = pytest.MonkeyPatch()
    try:
        # Sequential burst calls, so the scripted forms reach the same call indices every run.
        monkeypatch.setattr(cc, "DEFAULT_BURST_CONCURRENCY", 1)
        for arm, extra in ARMS.items():
            client = ScriptedClient()
            monkeypatch.setattr(
                count_proposer, "AnthropicCountProposer",
                lambda records, client=client, **kwargs: REAL_PROPOSER(
                    records, client=client, **{k: v for k, v in kwargs.items() if k != "allow_live"}))
            out = root / arm
            argv = ["chembench_core", "--profile", "dev", "--particles", "16", "--q", "8",
                    "--manifest", str(REPO / "manifests" / "chembench_paper_manifest.json"),
                    "--task", "c2_product_inhibition__medium__v0", "--noise", "0.01", "--cell", "101",
                    "--d0-policy", "fixed3", "--design-search", "ascent", "--initial-catalogue", "off",
                    "--initial-proposals", "12", "--proposal-limit", "3",
                    "--composition-paragraph", "v2", "--archive-review-rounds", "2",
                    "--mask-input-names", "--prompt-variant", str(variant_file),
                    "--output", str(out), "--seed", "31",
                    "--proposer", "anthropic", "--allow-paid-api", "--llm-model", "claude-opus-5-5",
                    "--llm-reasoning", "effort=medium,thinking=adaptive,display=summarized",
                    "--llm-max-tokens", "16000", "--rounds", "2", *extra]
            monkeypatch.setattr(sys, "argv", argv)
            cc.main()
            runs[arm] = (out, client)
    finally:
        monkeypatch.undo()
        count_proposer.set_prompt_variant(None)
        count_proposer.set_composition_paragraph("v1")
    return runs


def load(path):
    """Read a JSON file and return its content."""
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_smoke_three_candidates_per_call_and_the_pool_held_at_twelve(smoke_runs, arm):
    """Smoke check: three candidates per call, the active pool held at twelve, and the recorded digests."""
    out, client = smoke_runs[arm]
    config, summary = load(out / "config.json"), load(out / "summary.json")
    receipt = load(out / "initial_pool_receipt.json")
    assert receipt["requested"] == 12 and receipt["initial_pool_count"] == 12
    assert len(receipt["waves"]) == 1 and receipt["waves"][0]["calls"] == 12
    assert config["initial_proposals"] == 12 and config["initial_pool"] == 12
    assert config["proposal_limit"] == 3 and config["composition_paragraph"] == "v2"
    v2 = FIVE_KEY_VARIANT + count_proposer.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3")
    review = v2 + count_proposer.ARCHIVE_REVIEW_PARAGRAPH.replace("{limit}", "3")
    assert config["composed_prompt_sha256"] == sha(v2)
    assert config["composition_paragraph_sha256"] == V2_FILE_SHA256
    assert config["archive_review_composed_prompt_sha256"] == sha(review)
    assert config["archive_review_paragraph_sha256"] == REVIEW_FILE_SHA256
    assert config["label_receipt"]["binding"]["settings"]["composed_prompt_sha256"] == sha(v2)
    assert [body["system"] for body in client.bodies[12:]] == [v2, review]
    assert len(client.bodies) == 12 + 2
    for t, (admitted, duplicates) in ((1, (3, 0)), (2, (2, 1))):
        record = load(out / "proposals" / f"t{t:02d}.json")
        assert record["transport"]["calls"] == 1 and record["transport"]["repairs"] == 0
        assert record["transport"]["composed_prompt_sha256"] == sha(review if t == 2 else v2)
        assert record["admitted_count"] == admitted and record["duplicate_count"] == duplicates
        assert len(record["proposed_ids"]) == 3 and len(record["admitted_ids"]) == admitted
        assert len(record["active_pool_after"]) == cc.ACTIVE_MAXIMUM == 12
    assert summary["active_pool"] == 12
    assert summary["proposal_accounting"]["admitted_candidates"] == 12 + 3 + 2


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_smoke_the_review_round_revives_one_and_refuses_two(smoke_runs, arm):
    """Smoke check: the review round revives one model and refuses two requests."""
    out, _ = smoke_runs[arm]
    summary = load(out / "summary.json")
    revival = load(out / "revival" / "t02.json")
    retired_at_one = load(out / "proposals" / "t01.json")["retired_ids"]
    assert revival["round"] == 2 and len(revival["requested_ids"]) == 3
    assert revival["valid_ids"] == [revival["requested_ids"][0]]
    assert revival["valid_ids"][0] in retired_at_one
    assert [entry["why"] for entry in revival["invalid"]] == ["active", "over_limit"]
    assert revival["revive_missing"] is False and revival["review_reply_shape"] == "object"
    assert [entry["id"] for entry in revival["refit_results"]] == revival["valid_ids"]
    assert sorted(revival["re_entered_ids"] + revival["retired_again_ids"]) == revival["valid_ids"]
    assert revival["revival_attempts"] == {revival["valid_ids"][0]: 1}
    assert revival["wall_seconds"] > 0 and len(revival["active_pool_after"]) == 12
    assert summary["revival_events"] == [revival]
    assert summary["proposal_accounting"]["revival_refits"] == 1
    archive_notes = [entry.get("note", "") for entry in
                     load(out / "packets" / "t02.json")["archive"]]
    assert not any("revived" in note for note in archive_notes)  # the packet predates the revival
    final = load(out / "proposals" / "t02.json")
    assert final["archive_review"] is True and final["revival_valid_ids"] == revival["valid_ids"]


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_smoke_every_round_records_the_parts_and_chooses_on_the_arm_column(smoke_runs, arm):
    """Smoke check: every round records the EIG parts and chooses on the arm's own column."""
    out, _ = smoke_runs[arm]
    config = load(out / "config.json")
    policy = cc.CHEM_EIG_PART_ALIASES.get(arm, arm)
    column = "eig" if policy == "joint_eig" else policy
    assert config["design_policy"] == policy
    assert config["eig_components_record"] == json.loads(json.dumps(cc.EIG_COMPONENTS_RECORD))
    assert config["point_disagreement_record"] == json.loads(json.dumps(cc.POINT_DISAGREEMENT_RECORD))
    assert config["design_objective"] == column
    if policy == "joint_eig":
        assert "design_policy_objective_column" not in config
        assert config["points_variant"] == "structure_free"
        assert "objective_column" not in config["design_search_rule"]
    else:
        assert config["design_policy_objective_column"] == column
        assert config["design_policy_target"] == cc.CHEM_EIG_PART_POLICIES[policy][2]
        assert config["points_variant"] == "+".join(sorted(
            ["structure_free", cc.CHEM_EIG_PART_POLICIES[policy][3]]))
        assert config["design_search_rule"]["objective_column"] == column
        assert (config["design_search_rule"]["objective_standard_error_column"]
                == cc.CHEM_EIG_PART_POLICIES[policy][1])
        assert config.get("design_policy_requested") == ("model_eig" if arm == "model_eig" else None)
    observed = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    for t in (1, 2):
        record = load(out / "acquisition" / f"t{t:02d}.json")
        valid = [row for row in record["arrival_rows"] if row.get("invalid") is not True]
        assert valid
        for row in valid:
            assert abs(row["eig"] - (row["structure_eig"] + row["parameter_eig"])) <= 1e-12
            assert row["point_disagreement"] is None or row["point_disagreement"] >= 0
        best = max(valid, key=lambda row: (row[column], [-v for v in row["design"]]))
        assert record["chosen"] == best["design"]
        assert record["policy"]["objective_column"] == column
        assert record["policy"]["design_policy"] == policy
        assert list(observed[2 + t]["x"].values()) == record["chosen"]
        if policy == "joint_eig":
            # The joint arm's objective rows carry no part; only the arrival rows are annotated.
            assert not any("structure_eig" in row for row in record["rows"]
                           if row["phase"] != "arrival")
            continue
        # An EIG-part arm's objective makes the decomposed call at every evaluated point, and the
        # climb, the chooser and the holdout read the part.
        evaluated = [row for row in record["rows"] if row.get("invalid") is not True]
        assert {row["phase"] for row in evaluated} >= {"start", "arrival", "noise_floor",
                                                       "independent_holdout"}
        for row in evaluated:
            assert abs(row["eig"] - (row["structure_eig"] + row["parameter_eig"])) <= 1e-12
        starts = {row["restart_index"]: row for row in record["rows"] if row["phase"] == "start"}
        for trajectory, arrival in zip(record["trajectories"], record["arrival_rows"], strict=True):
            if trajectory["value_climbed"] is not None:
                assert trajectory["value_climbed"] == (
                    arrival[column] - starts[arrival["restart_index"]][column])
        assert (record["policy"]["selection_standard_error_column"]
                == cc.CHEM_EIG_PART_POLICIES[policy][1])
        holdout = record["holdout_rows"][record["chosen_restart_index"] - 1]
        assert record["holdout_value"] == (
            None if holdout.get("invalid") is True else holdout[column])
        assert record["policy"]["joint_eig_at_chosen"] == best["eig"]
        assert record["policy"]["joint_eig_max"] == max(row["eig"] for row in valid)
        assert record["target"] == cc.CHEM_EIG_PART_POLICIES[policy][2]


def test_smoke_round_one_starts_alike_and_each_arm_climbs_its_own_column(smoke_runs):
    """Same round-0 pool and seeds: the same five starts and the same joint EIG there in every arm
    (the decomposed call leaves the joint value alone); from there each part arm climbs its part."""
    tables = {arm: load(out / "acquisition" / "t01.json") for arm, (out, _) in smoke_runs.items()}
    reference = tables["joint_eig"]

    def start_eig(record):
        """Return the joint EIG of the start rows of a record."""
        return [row.get("eig") for row in record["rows"] if row["phase"] == "start"]

    for record in tables.values():
        assert [t["start_unit"] for t in record["trajectories"]] == [
            t["start_unit"] for t in reference["trajectories"]]
        assert start_eig(record) == start_eig(reference)
    for arm in ("model_eig", "parameter_eig"):
        assert tables[arm]["trajectories"] != reference["trajectories"]
