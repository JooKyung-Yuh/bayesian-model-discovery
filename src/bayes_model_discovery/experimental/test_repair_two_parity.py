"""Omitted-option scientific parity against an explicitly supplied checkout."""

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ..provenance.canonical import is_hex64

# The parity base is BASE_COMMIT (the last deliberate science change). On the
# omitted-flag default path every later change alters records only: the keys each one adds are
# named below and removed by name from both runs; any other difference fails. The comparison is opt-in and
# skips when a checkout of this commit is absent.
BASE_COMMIT = "b43bf8ae5245a1010594ef53cf51d6358c759ee4"
IDENTITY_FIELDS = ("source_files_sha256", "source_sha256", "created_utc", "output",
                   "manifest_ref", "python", "source_root", "adapter_source")
#: Per bundle, the files the stack changed after the base among those its
#: config's source digests record (model_identity.py, count_inference.py, count_diagnostics.py
#: and design_policies.py did not move).
CHANGED_SOURCES = {
    "core": ("belief_packet.py", "core_count.py", "count_models.py", "count_proposer.py"),
    "chembench": ("chembench_core.py", "core_count.py", "count_models.py", "positive_inference.py"),
}
#: The three engine limits every packet's `constraints` now states.
PCX1_PACKET_CONSTRAINT_KEYS = ("max_distinct_mean_coefficients", "reserved_names",
                               "positive_mean_required_families")
#: The digest pair a proposal's transport keeps of the packet it sent, moved by the keys
#: above; each run's pair is checked against that run's own packet file before it is removed.
PCX1_PACKET_DIGEST_KEYS = ("packet_sha256", "packet_bytes")
#: A later change: `failure_class` on each CORE candidate event, `parameters` on each
#: archive.json entry, and the run's `failure_class_counts` in summary `proposal_accounting`.
PCX2_EVENT_KEYS = ("failure_class",)
PCX2_ARCHIVE_KEYS = ("parameters",)
PCX2_ACCOUNTING_KEYS = ("failure_class_counts",)
#: A further change: a ChemBench later round's (t >= 1) candidate outcome events.
PCX2B_ROUND_EVENTS_KEY = "events"
#: A later change (replacing an earlier one): the declaration every ChemBench config.json now
#: carries and CORE's never does; checked on the current run's config, then removed by name.
CIC1_CONFIG_KEY, CIC1_CONFIG_VALUE = "constraints_disclosure", "cic3_v1"
#: Science files compared as raw bytes on their own, beside every
#: predictions/, acquisition/ and .npz file; and the id sets each proposals record admits.
SCIENCE_FILES = ("observations.jsonl", "scores.jsonl", "frozen_d0_policy.json", "pool_weights.jsonl",
                 "scores_after_finish.json", "official_scores_after_finish.json")
ADMITTED_ID_KEYS = ("proposed_ids", "admitted_ids", "after_ids")
POINT_IDENTITY_FIELDS = ("source_path", "code_sha")
TIMING_SIDECARS = ("fit_timings.jsonl", "acquisition_timings.jsonl")

def adapter_file(tree):
    """The ChemBench adapter of a checkout: repository-root `adapters/` before the public-release overhaul, the package after."""
    packaged = tree / "src/bayes_model_discovery/adapters/chembench_observation.py"
    return packaged if packaged.is_file() else tree / "adapters/chembench_observation.py"


HARNESS = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from pytest import MonkeyPatch
from bayes_model_discovery.experimental import run_label_policy as policy
with MonkeyPatch.context() as patch:
    output, manifest, bundle, cell = sys.argv[1:]
    if bundle == "core":
        from bayes_model_discovery.experimental.test_count_proposer import run_args, stub_fits
        runner = stub_fits(patch)
        args = run_args(output, rounds=2, cell=cell)
        task = "structure_out__truth_F2"
    else:
        from bayes_model_discovery.experimental import chembench_core as runner
        from bayes_model_discovery.experimental import test_chembench_core as fixture
        from bayes_model_discovery.experimental.test_chembench_mda_options import stub_fits
        stub_fits(patch)
        args = fixture._args(Path(output), Path(manifest), rounds=2, cell=cell,
                            structure_proposals="free" if cell[0] == "1" else "catalog")
        task = "synthetic"
    runner.run(args)
    from bayes_model_discovery.experimental.count_proposer import prompt_identity
    declared = SimpleNamespace(**{**vars(args), **prompt_identity()})
    receipt = policy.bmd_label_receipt(bundle, declared, task=task, metadata={})
    (Path(output) / "synthetic_receipt.json").write_text(json.dumps(receipt, sort_keys=True))
'''


def checked_points(output, config, source, bundle):
    """Validate the source identity of each points.csv row and return the rows without identity fields."""
    rows = list(csv.DictReader((output / "points.csv").open(newline="")))
    key = "source_files_sha256" if bundle == "core" else "source_sha256"
    expected = config[key][bundle + ("_count.py" if bundle == "core" else "_core.py")]
    assert is_hex64(expected)
    paths = []
    for row in rows:
        assert row["code_sha"] == expected
        path = row["source_path"]
        assert path.startswith(config["output"])
        paths.append(path)
        if bundle == "core":
            match = re.search(r"#L([0-9]+)\(round=([^,]+),phase=([^\)]+)\)$", path)
            assert match, path
            scores = (output / "scores.jsonl").read_text().splitlines()
            assert 1 <= int(match[1]) <= len(scores)
            record = json.loads(scores[int(match[1]) - 1])
            assert str(record["round"]) == match[2] and record["phase"] == match[3]
    assert len(paths) == len(set(paths))
    return [{key: value for key, value in row.items() if key not in POINT_IDENTITY_FIELDS}
            for row in rows]


@pytest.mark.parametrize("bundle,cell", [(bundle, cell) for bundle in ("core", "chembench")
                                          for cell in ("000", "100")])
def test_omitted_flag_preserves_all_science_and_authentic_identity(tmp_path, bundle, cell):
    """Test that omitted flag preserves all science and authentic identity."""
    configured = os.environ.get("BMD_RPR2_BASELINE_DIR")
    if os.environ.get("BMD_BASELINE_COMPARE") != "1" or not configured:
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_RPR2_BASELINE_DIR for baseline parity")
    baseline, current = Path(configured), Path(__file__).resolve().parents[3]
    # The baseline checkout is BASE_COMMIT.
    assert subprocess.run(["git", "-C", str(baseline), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip() == BASE_COMMIT
    assert subprocess.run(["git", "-C", str(baseline), "status", "--short"],
        capture_output=True, text=True, check=True).stdout == ""
    from .test_chembench_core import _manifest

    manifest = _manifest(tmp_path)
    outputs, configs, receipts = [], [], []
    for label, source in (("base", baseline), ("current", current)):
        out = tmp_path / label / "cell"
        env = dict(os.environ, PYTHONPATH=f"{source}/src:{source}",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="",
                   OPENROUTER_API_KEY="", PYTHON_DOTENV_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-B", "-c", HARNESS,
                                    str(out), str(manifest), bundle, cell],
            cwd=source, env=env, capture_output=True, text=True, timeout=150, check=False)
        (tmp_path / (label + ".log")).write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        config = json.loads((out / "config.json").read_text())
        assert config["output"] == str(out)
        sources = config["source_files_sha256" if bundle == "core" else "source_sha256"]
        for name, digest in sources.items():
            assert digest == hashlib.sha256((source / "src/bayes_model_discovery/experimental" / name).read_bytes()).hexdigest()
        receipt = json.loads((out / "synthetic_receipt.json").read_text())
        assert receipt["policy_source_sha256"] == hashlib.sha256(
            (source / "src/bayes_model_discovery/experimental/run_label_policy.py").read_bytes()).hexdigest()
        receipt.pop("policy_source_sha256")
        if bundle == "chembench":
            assert config["adapter_source"] == str(adapter_file(source))
            assert adapter_file(baseline).read_bytes() == adapter_file(current).read_bytes()
            assert "repair_attempts" not in config
            assert "repair_attempts" not in receipt["binding"]["settings"]
        else:
            assert config["repair_attempts"] == 1
            assert receipt["binding"]["settings"]["scientific_options"]["repair_attempts"] == 1
        outputs.append(out)
        configs.append(config)
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
    a, b = configs
    assert CIC1_CONFIG_KEY not in a
    assert b.pop(CIC1_CONFIG_KEY, None) == (CIC1_CONFIG_VALUE if bundle == "chembench" else None)
    assert a.keys() == b.keys()
    assert {k: v for k, v in a.items() if k not in IDENTITY_FIELDS} == {
        k: v for k, v in b.items() if k not in IDENTITY_FIELDS}
    source_key = "source_files_sha256" if bundle == "core" else "source_sha256"
    assert a[source_key].keys() == b[source_key].keys()
    changed = {name for name in a[source_key] if a[source_key][name] != b[source_key][name]}
    # Exactly this bundle's listed sources moved.
    assert changed == set(CHANGED_SOURCES[bundle])
    assert checked_points(outputs[0], a, baseline, bundle) == checked_points(outputs[1], b, current, bundle)
    names = [sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file())
             for out in outputs]
    assert names[0] == names[1]
    # Science bytes and admitted ids, strictly.
    science = [name for name in names[0] if name in SCIENCE_FILES or name.endswith(".npz")
               or name.startswith(("predictions/", "acquisition/"))]
    assert "observations.jsonl" in science and any(name.startswith("fits/") for name in science)
    for name in science:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    admitted = [[{key: record[key] for key in ADMITTED_ID_KEYS if key in record}
                 for record in (json.loads((out / name).read_text()) for name in names[0]
                                if name.startswith("proposals/"))] for out in outputs]
    assert admitted[0] == admitted[1]
    for name in names[0]:
        if name in (*TIMING_SIDECARS, "config.json", "points.csv", "synthetic_receipt.json"):
            continue
        left, right = [out / name for out in outputs]
        if name.endswith(".npz"):
            with np.load(left) as aa, np.load(right) as bb:
                assert aa.files == bb.files
                for key in aa.files:
                    assert aa[key].dtype == bb[key].dtype
                    assert aa[key].tobytes() == bb[key].tobytes(), (name, key)
        # Weakened from raw bytes - archive, packets and proposals are compared as parsed JSON
        # because the comparison removes the keys added by later changes by name.
        elif name in ("summary.json", "archive.json") or (
                name.startswith(("fits/", "packets/", "proposals/")) and name.endswith(".json")):
            records = [json.loads(path.read_bytes()) for path in (left, right)]
            for out, record in zip(outputs, records, strict=True):
                if name == "summary.json":
                    for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "timing_seconds"):
                        record.pop(key, None)
                    for key in ("initial_pool_fit", "rounds_fit"):
                        record.get(key, {}).pop("seconds", None)
                    for key in ("initial_pool_llm", "rounds_llm"):
                        record.get(key, {}).pop("wall_seconds", None)
                if name.startswith("fits/"):
                    record.pop("fit_seconds", None)
                    record.get("diagnostics", {}).pop("sampler_attrs", None)
                else:
                    drop_unit_record_keys(name, record, out, bundle)
            assert records[0] == records[1], name
        else:
            assert left.read_bytes() == right.read_bytes(), name


def drop_unit_record_keys(name, record, out, bundle):
    """Each later change's named record keys, removed by name from one
    run's parsed summary, archive, packet or proposals record in place; nothing else."""
    if name == "summary.json":
        for key in PCX2_ACCOUNTING_KEYS:
            record.get("proposal_accounting", {}).pop(key, None)
    elif name == "archive.json":
        for entry in record:
            for key in PCX2_ARCHIVE_KEYS:
                entry.pop(key, None)
    elif name.startswith("packets/"):
        for key in PCX1_PACKET_CONSTRAINT_KEYS:
            record["constraints"].pop(key, None)
    elif "transport" in record:  # a round's record from a later change has no proposal, transport or packet
        if bundle == "chembench":
            record.pop(PCX2B_ROUND_EVENTS_KEY, None)
        else:
            for event in record["events"]:
                for key in PCX2_EVENT_KEYS:
                    event.pop(key, None)
        packet = (out / "packets" / Path(name).name).read_bytes()
        pair = tuple(record["transport"].pop(key) for key in PCX1_PACKET_DIGEST_KEYS)
        assert pair == (hashlib.sha256(packet).hexdigest(), len(packet)), name
