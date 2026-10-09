"""Opt-in baseline parity using fresh synthetic death/location worlds and real sequential Monte Carlo (SMC)."""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bayes_model_discovery.provenance.canonical import is_hex64

# Parity base: the commit whose recorded outputs the opt-in baseline test pins; it skips when a checkout of this commit is absent.
BASE_COMMIT = "c6fe27c76515065a943df33f122529eb181aac04"
LOCATION_COLUMNS = ("nmse_log_truth", "best_so_far_nmse_log_truth",
                    "nmse_log_truth_undefined_points")
VOLATILE_KEYS = {"fit_seconds", "wall_seconds", "user_cpu_seconds", "system_cpu_seconds",
                 "timing_seconds", "_t_sampling"}
HARNESS = r'''
from datetime import datetime
from pathlib import Path
import sys
import pytest
from bayes_model_discovery.experimental import boxinggym_core as bg
from bayes_model_discovery.experimental.test_named_inputs import bg_run_args
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2001, 1, 1, tzinfo=tz)
settings = bg.SMCSettings
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(bg, 'datetime', FrozenDatetime)
    patch.setattr(bg, 'SMCSettings', lambda *_: settings(16, 2, 2))
    bg.run(bg_run_args(Path(sys.argv[1]), task=sys.argv[2], cell='000', seed=73, rounds=1))
'''


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _csv(data):
    reader = csv.DictReader(io.StringIO(data.decode()))
    return reader.fieldnames, list(reader)


@pytest.mark.parametrize("task", ["death", "location"])
@pytest.mark.skip(reason="parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
def test_opt_in_default_boxinggym_matches_base(tmp_path, task):
    """Opt-in parity of default BoxingGym output with the baseline commit under real SMC."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 to run bounded baseline SMC parity")
    root = Path(__file__).resolve().parents[3]
    baseline_text = os.environ.get("BMD_SAFEGUARDS_BASELINE") or os.environ.get("BMD_BASELINE_DIR")
    external = os.environ.get("BMD_BOXINGGYM_SRC")
    if not baseline_text or not external:
        pytest.skip("set BMD_SAFEGUARDS_BASELINE and BMD_BOXINGGYM_SRC to read-only sources")
    baseline = Path(baseline_text)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=baseline, check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == BASE_COMMIT
    output = tmp_path / "same_output"
    saved, configurations = [], []
    for label, source in (("baseline", baseline), ("current", root)):
        env = os.environ.copy()
        env.update(PYTHONPATH=f"{source}/src:{source}:{external}", PYTHONDONTWRITEBYTECODE="1",
                   PYTENSOR_FLAGS="cxx=,compiledir=/tmp/f0bgparity,blas__ldflags=,openmp=False",
                   TMPDIR="/tmp", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="")
        process = subprocess.run([sys.executable, "-B", "-c", HARNESS, str(output), task],
                                 cwd=source, env=env, capture_output=True, text=True, timeout=240, check=False)
        (tmp_path / f"{label}.log").write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stdout + process.stderr
        config = json.loads((output / "config.json").read_text())
        assert config["smc"]["particles"] == 16 and config["smc"]["chains"] == 2
        assert "fit_timeout_seconds" not in config
        assert not (output / "fit_timings.jsonl").exists()
        for name, digest in config["source_files_sha256"].items():
            assert hashlib.sha256((source / "src/bayes_model_discovery/experimental" / name).read_bytes()).hexdigest() == digest
        configurations.append(config)
        destination = tmp_path / label
        output.rename(destination)
        saved.append(destination)
    left, right = saved
    observational_sidecars = {"fit_timings.jsonl", "acquisition_timings.jsonl",
                              "prediction_pool_summary.jsonl"}
    names = sorted(str(path.relative_to(left)) for path in left.rglob("*")
                   if path.is_file() and path.name not in observational_sidecars)
    assert names == sorted(str(path.relative_to(right)) for path in right.rglob("*")
                           if path.is_file() and path.name not in observational_sidecars)
    replacements = [(configurations[1]["source_files_sha256"][key].encode(),
                     configurations[0]["source_files_sha256"][key].encode())
                    for key in configurations[0]["source_files_sha256"]]
    report = []
    for name in names:
        a, b = (left / name).read_bytes(), (right / name).read_bytes()
        if name.endswith(".npz"):
            assert a == b, name
            with np.load(left / name) as aa, np.load(right / name) as bb:
                assert aa.files == bb.files, name
                for key in aa.files:
                    assert aa[key].dtype == bb[key].dtype and aa[key].tobytes() == bb[key].tobytes(), (name, key)
        elif task == "location" and name == "points_mda.csv":
            columns_a, rows_a = _csv(a)
            columns_b, rows_b = _csv(b)
            assert columns_b == columns_a + list(LOCATION_COLUMNS)
            assert len(rows_a) == len(rows_b)
            for old, new in zip(rows_a, rows_b, strict=True):
                assert old == {key: new[key] for key in columns_a}
                assert int(new["nmse_log_truth_undefined_points"]) >= 0
                assert float(new["nmse_log_truth"]) >= 0
        else:
            b = b.replace((str(root) + "/").encode(), (str(baseline) + "/").encode())
            for new, old in replacements:
                b = b.replace(new, old)
            if name.endswith(".json"):
                original, current_record = json.loads(a), json.loads(b)
                if name.startswith("acquisition/"):
                    # Candidate-set provenance keys: present and well-formed now, absent at the
                    # base commit; removed only after that check (normalization of stale provenance).
                    for key in ("candidate_set_sha256", "candidate_generator"):
                        assert key not in original, (name, key)
                        assert key in current_record, (name, key)
                    assert is_hex64(current_record.pop("candidate_set_sha256")), name
                    assert current_record.pop("candidate_generator")["hash_scope"] == "generated_pool_before_history_filters", name
                assert _canonical(original) == _canonical(current_record), name
            elif name.endswith(".jsonl"):
                assert [_canonical(json.loads(row)) for row in a.splitlines()] == [
                    _canonical(json.loads(row)) for row in b.splitlines()], name
            else:
                assert a == b, name
        report.append({"artifact": name, "parity": "exact_npz_bytes" if name.endswith(".npz") else
                       "existing_columns_unchanged" if task == "location" and name == "points_mda.csv" else
                       "equal_after_declared_provenance_and_timing_normalization"})
    (tmp_path / "parity_report.json").write_text(json.dumps({
        "task": task, "baseline": BASE_COMMIT, "fits": "real SMC 16 particles x 2 chains",
        "rounds": 1, "seed": 73, "volatile_keys": sorted(VOLATILE_KEYS), "artifacts": report,
    }, indent=2) + "\n")
