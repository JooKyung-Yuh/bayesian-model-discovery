"""Read-only registered-input checks; simulator truth is opened only after finish."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.icl_baseline.icl_baseline import History, input_files, verify_histories
from tools.icl_baseline.protocol import (
    MASKED_NAMES,
    VISIBLE_NAMES,
    json_bytes,
    metrics,
    number,
    observation_table,
    render,
)
from tools.icl_baseline.transport import sha, write_new


def run(masked_paths, shown_paths, output):
    from adapters.chembench_observation import ChemBenchSession

    masked = [History.load(path, "masked", i + 1) for i, path in enumerate(masked_paths)]
    shown = [History.load(path, "shown", i + 1) for i, path in enumerate(shown_paths)]
    histories = masked + shown
    verify_histories(histories)
    files = input_files(histories)
    for history in masked:
        files += [history.root / "llm_transport" / filename for filename in (
            "round_00__burst_w0_c00.request.json", "round_59.request.json")]
    before = {str(path): sha(path) for path in files}
    packet_checks = []
    observation_count = 0
    for history in masked:
        for round_index in (0, 30, 59):
            filename = "round_00__burst_w0_c00.request.json" if round_index == 0 else f"round_{round_index:02d}.request.json"
            data = json.loads((history.root / "llm_transport" / filename).read_text())
            packet = json.loads(data["bodies"][0]["messages"][0]["content"])
            rows = packet["observations"]
            expected = "\n".join(",".join([str(row["obs_id"] + 1),
                                          *(json.dumps(row[name]) for name in MASKED_NAMES), json.dumps(row["y"])])
                                 for row in rows)
            n = len(rows)
            actual = observation_table(history.observations[:n])
            if expected != actual:
                raise ValueError(f"packet observation table mismatch: {history.root.name}/{filename}")
            packet_checks.append({"path": str(history.root / "llm_transport" / filename),
                                  "round": round_index, "observations": n, "byte_equal_table": True})
            observation_count += n
    refs_equal = []
    for size in (256, 1000):
        first = histories[0].references(size)
        for history in histories:
            for target, values in history.references(size).items():
                if not np.array_equal(values, first[target]):
                    raise ValueError(f"reference mismatch: {history.root} {size} {target}")
            queries = history.queries(size)
            _, mapping = render("prediction", history.observations[:3], queries=queries[:128])
            if not np.array_equal(np.asarray([row["coordinates"] for row in mapping]), queries[:128]):
                raise ValueError("query mapping changed coordinates")
        refs_equal.append({"size": size, "histories": len(histories), "coordinate_and_target_equality": True})
    reproduction = masked[1].ours(256, 63)["primary_rate"]
    if f"{reproduction:.3g}" != "0.000214":
        raise ValueError("registered primary score reproduction failed")
    perfect = {}
    quantization = {}
    task = masked[0].config["task"]
    for size in (256, 1000):
        reference = masked[0].references(size)
        value = metrics(reference["noiseless_observation"], reference["primary_rate"])["nmsle"]
        expected = "0.000234" if size == 256 else "0.000278"
        if f"{value:.3g}" != expected:
            raise ValueError("registered perfect-observation reference reproduction failed")
        perfect[str(size)] = value
        session = ChemBenchSession(task["domain"], task["difficulty"], task["law_version"],
                                   noise_level=masked[0].config["noise"], budget=63)
        session.finish()
        coordinates = masked[0].queries(size)
        exact = [dict(zip(VISIBLE_NAMES, row)) for row in coordinates]
        rounded = [dict(zip(VISIBLE_NAMES, [float(number(v)) for v in row])) for row in coordinates]
        a = session.reference_values(exact)
        b = session.reference_values(rounded)
        quantization[str(size)] = {target: float(np.max(np.abs(np.log1p(a[target]) - np.log1p(b[target]))))
                                  for target in ("primary_rate", "noiseless_observation")}
    after = {str(path): sha(path) for path in files}
    if before != after:
        raise ValueError("SOURCE_CHANGED during precheck")
    result = {"packet_checks": packet_checks, "total_packet_observations": observation_count,
              "reference_checks": refs_equal, "primary_reproduction": reproduction,
              "perfect_observation_reference_not_a_bound": perfect,
              "query_quantization_max_abs_delta_log1p": quantization,
              "input_sha256_before": before, "input_sha256_after": after,
              "all_checked_inputs_unchanged": True, "paid_api_calls": 0}
    write_new(output, result)
    return result


def corpus_checks(root):
    from tools.icl_baseline.protocol import forbidden_words

    root = Path(root)
    records = []
    failures = []
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text())
        if "body" not in data:
            continue
        body = data["body"]
        text = body["system"] + "\n" + body["messages"][0]["content"]
        words = forbidden_words(text, data.get("visible", False))
        if words:
            failures.append({"path": str(path), "words": words})
        records.append({"path": str(path), "file_sha256": sha(path),
                        "body_sha256": hashlib.sha256(json_bytes(body)).hexdigest(),
                        "provenance": data.get("provenance")})
    return {"count": len(records), "records": records, "forbidden_word_hits": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-history", action="append", required=True, type=Path)
    parser.add_argument("--shown-history", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.masked_history, args.shown_history, args.output)
    print(json.dumps({"packet_observations": result["total_packet_observations"], "paid_api_calls": 0}))


if __name__ == "__main__":
    main()
