"""`--resume` for the NewtonBench task mode (2026-10-09): the ChemBench v6
resume (commit a693e409f8e6b4c5cda75c167f11ccc9e6241a99, registration v6-r 315f5e2) with the same
semantics, carried into the NewtonBench files only. That commit changed tools/icl_baseline/transport.py and
icl_baseline.py; this branch changes no existing file, so the resume state, its file rule and its replay live here and
in icl_ng.py, and a run that is not resumed still goes through the shared, unchanged `Transport.call`.

The rule (ICL_NG_REGISTRATION_V1_R_20261009.md):
1. the run is executed again from its start with the same code, settings, task and mode;
2. an attempt whose response file exists is answered from that file and nothing is sent; its recorded usage enters the
   Budget as when it arrived, and the recomputed accounting row must equal the recorded one;
3. a file the run writes that already exists must have this run's bytes and is left untouched (a numpy forecast file
   of the engine-scorer layout must hold the same array);
4. the first attempt without a response file is sent, and everything after it; before it is sent every recorded
   response must have been replayed and the Budget must hold exactly the recorded spend;
5. a request file without a response (in flight when the process died) is reused when its bytes match, and listed with
   its reservation (the provider may or may not have billed it);
6. a finished run (receipt.json) and a folder no run started are refused before anything is built."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from tools.icl_baseline.protocol import json_bytes
from tools.icl_baseline.transport import write_new


class ResumeMismatch(RuntimeError):
    """A resumed run met a recorded file that is not what this run writes or reads there. The replay is then not the
    recorded run, so nothing more is written or sent."""


class Resume:
    """The resume state of the run in this process (transport.Resume of a693e40, same fields and checks)."""

    def __init__(self, output):
        self.output = Path(output)
        records = [json.loads(p.read_text()) for p in sorted((self.output / "transport").glob("*.response.json"))]
        responses = sorted(p.stem for p in (self.output / "responses").glob("*.json"))
        self.recorded_attempts = len(records)
        self.recorded_spent_usd = math.fsum(r["accounting"]["cost_usd"] for r in records)
        self.recorded_responses = len(responses)
        self.last_recorded_response = responses[-1] if responses else None
        self.replayed, self.sent, self.reused_request_files = [], [], []
        self.matched_files = 0
        self.failed = False

    def fail(self, message):
        self.failed = True
        raise ResumeMismatch(message)

    def match(self, path, data):
        if Path(path).read_bytes() != data:
            self.fail(f"resume: {path} differs from this run's bytes")
        self.matched_files += 1

    def recorded(self, stem, request_id, attempt):
        path = Path(f"{stem}.response.json")
        if not path.exists():
            return None
        if self.sent:
            self.fail(f"resume: {path} is recorded after the first sent attempt {self.sent[0]}")
        try:
            record = json.loads(path.read_text())
        except ValueError:
            self.fail(f"resume: {path} does not parse")
        if (record.get("request_id"), record.get("attempt")) != (request_id, attempt):
            self.fail(f"resume: {path} records another attempt")
        return record

    def summary(self):
        return {"recorded_attempts": self.recorded_attempts, "recorded_spent_usd": self.recorded_spent_usd,
                "recorded_responses": self.recorded_responses, "last_recorded_response": self.last_recorded_response,
                "replayed_attempts": len(self.replayed), "sent_attempts": len(self.sent),
                "first_sent_attempt": self.sent[0] if self.sent else None,
                "reused_request_files": self.reused_request_files, "matched_existing_files": self.matched_files}


#: The resume state of the run in this process; None = a fresh run (every file new, as before).
RESUME = None


def write_out(path, value):
    """`transport.write_new`, except that a resumed run leaves an existing file untouched when it has this run's bytes."""
    path = Path(path)
    if RESUME is not None and path.exists():
        RESUME.match(path, json_bytes(value))
        return
    write_new(path, value)


def write_text_out(path, text):
    path = Path(path)
    if RESUME is not None and path.exists():
        RESUME.match(path, text.encode())
        return
    with path.open("x") as stream:
        stream.write(text)


def save_mean_out(path, values):
    """One forecast in the engine scorer's layout (an .npz holding `mean`). An archive's bytes carry its write time, so a
    resumed run compares the stored array instead, exactly."""
    path = Path(path)
    values = np.asarray(values, dtype=float)
    if RESUME is not None and path.exists():
        with np.load(path, allow_pickle=False) as data:
            if sorted(data.files) != ["mean"] or not np.array_equal(data["mean"], values) \
                    or data["mean"].dtype != values.dtype:
                RESUME.fail(f"resume: {path} differs from this run's forecast")
        RESUME.matched_files += 1
        return
    buffer = io.BytesIO()
    np.savez(buffer, mean=values)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
