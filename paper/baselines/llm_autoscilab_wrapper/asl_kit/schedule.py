"""Evaluation-schedule variants of the released DiscoveryLoop (2026-10-08): released | ef | er.

The released loop fits an equation only when `oracle_calls >= min_data_for_eq` and
`oracle_calls % eq_learner_every == 0` (discovery.py 1859-1862), and runs up to `max_experiments_per_iter`
experiments per outer iteration (discovery.py 1711, 1715-1717). All three are DiscoveryConfig fields
(discovery.py 123-126). The released runners set them: the ChemBench runner `eq_learner_every = 20`
(scripts/run_chembench_comparison.py 691, hard-coded) with max_experiments_per_iter 5 (the budget runner);
the NewtonBench runner both = budget // 5. So the per-experiment curve of a released run has no law before
b = 20 (ChemBench) / b = 12 (BoxingGym, ForceBench; budget 30 resp. 16).

  released  nothing is changed (the default; the kit's tip behaviour byte for byte)
  ef        every fit: eq_learner_every = 1, min_data_for_eq = 0; the batch size per iteration is the released one,
            so a law exists after every outer iteration (every 5 / 6 / 3 experiments)
  er        every round: ef plus max_experiments_per_iter = 1 (one experiment per iteration, as MDA's throttled
            DP-Agent protocol), so a law exists after every experiment

The three fields are overridden inside `DiscoveryConfig.__post_init__` of a subclass that replaces
`autoscilab.loop.discovery.DiscoveryConfig` (the same module-attribute replacement the kit already uses for
`get_oracle`): the released runners build their config as released and the subclass applies the override
afterwards, recording the released value beside the applied one (receipt `schedule.overrides`).
`run_llm_pipeline` imports DiscoveryConfig inside the function body (run_chembench_comparison.py 628), so the
replacement reaches the ChemBench path too. No released file is edited.

What the override does NOT change: the first law still waits for the released equation learner's own floors.
`EquationLearner.fit` returns None below 5 stored points (learner.py 279 and 534; powerlaw prefit below 5,
learner.py 1131; skeleton families below 10, learner.py 304), and its 80/20 split gives a single validation point
below 10 stored points (`n_val = max(1, int(0.2 n))`, learner.py 363), on which `_r2` is NaN unless the prediction is
exact (ss_tot = 0, learner.py 1233-1240), so `fitted.is_valid()` fails and the fit returns None. A PySR law
therefore normally first passes at 10 stored points: b = 7 (ChemBench, D0 = 3), b = 8 (BoxingGym, ForceBench, D0 = 2);
earlier laws come from two released paths: the power-law prefit (discovery.py 176, learner.py 420-423; 5 positive training
points = 6 stored, learner.py 1131-1134; a real run on a positive-valued world takes it too - ForceBench er stub b = 4) and a
validated LLM hypothesis fitted into the candidate pool (discovery.py 1900-1915; ForceBench ef stub b = 3). Stub cells: er
BoxingGym 8, ChemBench 7, ForceBench 4; ef BoxingGym 6, ChemBench 10, ForceBench 3. Earlier b stay `no_law`. The warm sweeps
(`warm_sweep_rounds = len(param_names)`, discovery.py 1605) have no flag; under er they shrink to one
experiment per parameter. The phases (70/20/10 %), the UCB -> EI switch, PySR iterations (ChemBench staged),
the final fit and the LLM arbitration, the leak block and D0 injection are the released ones. The NewtonBench
config's early stop (`confidence_threshold` 0.9, discovery.py 2037) stays and gets more chances under ef / er;
the recorder marks carried rows `_carried_after_termination` as before.
"""
import dataclasses

MODES = ("released", "ef", "er")
OVERRIDES = {
    "released": {},
    "ef": {"eq_learner_every": 1, "min_data_for_eq": 0},
    "er": {"eq_learner_every": 1, "min_data_for_eq": 0, "max_experiments_per_iter": 1},
}
LABELS = {"released": "AutoSciLab (as released)",
          "ef": "AutoSciLab - every fit (released batch, fit after every iteration)",
          "er": "AutoSciLab - every round (1 experiment per round)"}
STATE = {"mode": "released", "applied": []}


def install(mode):
    """Select the schedule for this process; returns the config class in use (released or the subclass)."""
    if mode not in MODES:
        raise ValueError(f"schedule mode must be one of {MODES}, not {mode!r}")
    from autoscilab.loop import discovery as D

    released_cls = getattr(D.DiscoveryConfig, "_released_config", D.DiscoveryConfig)
    STATE["mode"], STATE["applied"] = mode, []
    if mode == "released":
        D.DiscoveryConfig = released_cls      # idempotent: a second install in one process puts the released class back
        return released_cls
    overrides = dict(OVERRIDES[mode])
    for name in overrides:
        if name not in {f.name for f in dataclasses.fields(released_cls)}:
            raise RuntimeError(f"released DiscoveryConfig has no field {name!r}")

    @dataclasses.dataclass
    class DiscoveryConfig(released_cls):   # the released name, so dataclasses.asdict and repr read as before
        def __post_init__(self):
            parent = getattr(super(), "__post_init__", None)
            if parent is not None:
                parent()
            record = {}
            for name, value in overrides.items():
                record[name] = {"released": getattr(self, name), "applied": value}
                setattr(self, name, value)
            STATE["applied"].append(record)

    DiscoveryConfig._released_config = released_cls
    DiscoveryConfig.__module__ = released_cls.__module__
    D.DiscoveryConfig = DiscoveryConfig
    return DiscoveryConfig


def receipt_block(mode):
    """What a cell's receipt records about its schedule."""
    return {"mode": mode, "label": LABELS[mode], "overrides": OVERRIDES[mode],
            "configs_constructed": len(STATE["applied"]),
            "released_values": (STATE["applied"][0] if STATE["applied"] else None),
            "note": ("released loop, released runner config" if mode == "released" else
                     "DiscoveryConfig subclass applies the overrides in __post_init__; the released runner built the "
                     "config first (released value recorded beside the applied one)")}
