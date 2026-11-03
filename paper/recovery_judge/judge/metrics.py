"""Per-round recovery bookkeeping of one run: the top-weighted pool member, A_t (target family admitted), S_t (top-1
passes) and W_t (weight on passing members) for the strict, containment and membership rules, and the run summary
(E_B, T_A, censoring, A_0, A_B, S_B).  Pure functions over the judged verdicts; no record is read here.
"""


def top_of(ids, weights):
    """Largest weight; an exact tie goes to the smallest model id (fixed, blind to the truth)."""
    if not ids:
        return None
    best = max(weights)
    return min(m for m, w in zip(ids, weights) if w == best)


def matches(v, rule):
    if v is None:
        return False
    if rule == "strict":
        return v["state"] == "equivalent"
    if rule == "contain":
        return v["state"] == "equivalent" or bool(v.get("containment"))
    if rule == "member":
        return v["state"] == "equivalent" or v.get("world_law_member") is True
    raise ValueError(rule)


def run_metrics(recs, verdicts, rounds):
    rows = []
    for r in recs:
        ids, ws = r["ids"], r["weights"]
        top = top_of(ids, ws)
        row = {"phase": r["phase"], "t": r["t"], "n_observed": r["n_observed"], "n_models": len(ids),
               "top_model_id": top, "top_state": (verdicts.get(top) or {}).get("state") if top else None,
               "engine_top_agrees": None if r.get("engine_top") is None else int(r["engine_top"] == top),
               "n_unresolved": sum(1 for m in ids if (verdicts.get(m) or {"state": "unresolved"})["state"] == "unresolved"),
               "unresolved_weight": sum(w for m, w in zip(ids, ws)
                                        if (verdicts.get(m) or {"state": "unresolved"})["state"] == "unresolved"),
               "weights_sum": sum(ws), "aligned": int(len(ids) == len(ws))}
        for rule, suffix in (("strict", ""), ("contain", "_contain"), ("member", "_member")):
            hit = [matches(verdicts.get(m), rule) for m in ids]
            row["A_t" + suffix] = int(any(hit))
            row["S_t" + suffix] = int(bool(top) and matches(verdicts.get(top), rule))
            row["W_t" + suffix] = float(sum(w for h, w in zip(hit, ws) if h))
        rows.append(row)
    summary = {}
    for suffix in ("", "_contain", "_member"):
        first = next((x for x in rows if x["A_t" + suffix] == 1), None)
        summary["E_B" + suffix] = int(any(x["A_t" + suffix] for x in rows))
        summary["T_A" + suffix] = first["t"] if first else rounds
        summary["T_A_phase" + suffix] = first["phase"] if first else None
        summary["censored" + suffix] = 0 if first else 1
        zero = [x for x in rows if x["t"] == 0]
        summary["A_0" + suffix] = int(any(x["A_t" + suffix] for x in zero)) if zero else None
        last = rows[-1] if rows else None
        summary["A_B" + suffix] = last["A_t" + suffix] if last else None
        summary["S_B" + suffix] = last["S_t" + suffix] if last else None
    if rows:
        summary["final_phase"], summary["final_t"] = rows[-1]["phase"], rows[-1]["t"]
    return rows, summary
