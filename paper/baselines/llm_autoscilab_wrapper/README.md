# LLM-AutoSciLab interface kit (unit the AutoSciLab wrapper unit, 2026-10-06)

Runs the released LLM-AutoSciLab (Kabra et al.; github.com/scientific-discovery/LLM-AutoSciLab at commit
9a37255ff5432a7155b6cde8985e9654f2f8f272, no edits) as the external baseline on ChemBench, BoxingGym and
ForceBench with claude-opus-5-5. NeuronBench is excluded (decision of 2026-10-05).

## What is changed, and what is not

**The Claude client path is the only thing that may change the released agent's behaviour.** Prompts,
prompt builders, memory, the active-learning selector, the GP surrogate, the equation learner (PySR), the
loop, its phases and its final LLM arbitration are the released code, imported from the checkout.

The Claude path (asl_kit/claude_client.py) wraps `LLMClient.__init__` only for model names starting with
`claude`. It sends the released system and user text byte for byte with `thinking: {type: adaptive}` and
`output_config: {effort: medium}`, and raises `max_tokens` to 16,000 (Opus 5.5 cannot switch thinking off,
and thinking counts against max_tokens; the released caps 4,096 / 8,192 / 16 would truncate or empty the
answer). It drops `temperature` (rejected by Opus 5.5; unset in the released configs) and
`response_format` (no json_object mode on the Messages API; when it was requested and the reply is not a
JSON object, the outermost {...} span is used if it parses - flagged per call). Refusals raise; no fallback
model. Every call's tokens and USD (4 / 20 per 1M input / output, input including cache tokens) go to
`llm_usage.jsonl`; cell and folder caps are checked before every call.

Everything else is interface - what the agent is given and how its output is scored:

| Interface | ChemBench | BoxingGym | ForceBench |
|---|---|---|---|
| Oracle | native `ChemBenchOracle`, leak-blocked subclass | adapter over our worlds (world pack) | NewtonBench-style scalar force oracle |
| World | noise 0.01 (ours; the released runner uses 0.0), universal grammar, law v0 | seeds 31/37/41, survival 43/53/61 (same worlds as our runs) | force laws of our worlds; noise = NewtonBench's 5 % relative noise (default; `--force-noise additive_var` gives sigma_F = 0.05 sqrt(var_F)) |
| Initial data (D0) | our fixed3 (3 points) | our random2 (the 2 designs and observations our runs saw) | 2 designs: our D0 (p1, p2) knobs, r (and t) drawn |
| Budget after D0 | 30 | 30 | 16 |
| Agent config | the released ChemBench budget runner (`_run_one` -> `run_llm_pipeline`) | the released NewtonBench runner (`run_one`, max_per_iter = eq_learner_every = budget // 5) | same as BoxingGym |
| Names shown | as released: C_A ... pH, the universal mechanism grammar, the goal sentence and the domain id | X1..Xn (our masking); `--names shown` for native names | X1..X4 (r, p1, p2, t); `--names shown` |

Leak block (asl_kit/protocol.py): during the run `evaluate_law` and `get_ground_truth_law_str` raise. The
released loop asks for ground truth in four places (discovery.py 1928, 1971, 2182, 2215) - mid-run GT
RMSLE in the prompt, the GT-best equation, and a final selection that prefers the GT-best law. All four are
inside try/except in the released code, so the block removes that feedback without other effects; every
attempt is counted in the receipt, and the prompts are audited for "GT RMSLE" (must be 0). This is the
oracle-blocked variant of 2026-09-06,
done with a subclass instead of class-attribute replacement.

D0 injection: our D0 is added to the loop's store before the first iteration and `_oracle_calls` stays 0, so
the released budget arithmetic runs over the 30 (16) experiments the agent chooses. The released
first-iteration text still says "You have NO data yet" while the table shows the D0 rows (`--d0 none`
removes D0).

## Per-experiment curve (one run, every budget)

`laws.jsonl` holds, for b = 0..B (experiments after D0), the law the loop holds when it asks for experiment
b+1 - its working equation of the last fit, `max(_candidate_pool, key=r2)` (discovery.py 1914-1915: PySR,
skeleton and validated LLM-hypothesis candidates), or `_eq_learner.best` while the pool is empty - and for
b = B the run's returned law (after the released final fit and LLM arbitration). `_eq_learner.best` alone is
kept in every row. If the NewtonBench configuration stops early (bootstrap confidence >= 0.9, released
behaviour), the final law is carried to the remaining b and marked `_carried_after_termination`.
`curve.json` scores each law offline, after the run; nothing flows back. The released
agent fits equations only when `oracle_calls >= 10` and `oracle_calls % eq_learner_every == 0`, so the curve
has **no law before b = 20 on ChemBench** (eq_learner_every = 20) **and before b = 12 on BoxingGym and
ForceBench** (every 6 resp. 3 experiments). These budgets are `no_law`, not zero. The released schedule
also depends on the total budget (phases at 70/20/10 %, UCB -> EI at 25/55 %), so the point at b of a B = 30
run is not the same as a separate run with budget b (MDA v5 ran one run per budget).
**Decision (main, 2026-10-06 01:5x): kept as is.** The AutoSciLab line starts at b >= 20 (ChemBench) and
b >= 12 (BoxingGym, ForceBench); there are no small-budget runs and no offline refits, and the earlier
budgets stay `no_law` in laws.jsonl and curve.json. The released schedule is part of what is compared.

Scores:
- ChemBench: `official_rmsle_1000` (native evaluate_law, clean r0, 1,000 points, seed 42), `nmsle_1000_primary`
  (MDA Eq. 50), `nmsle_256_noiseless_observation` and `nmsle_256_primary` (our 256-point set and our two
  targets; predictions get evaluate_law's |.| and finite mask, masked points counted).
- BoxingGym: `nmse_truth` = mean((law - E[Y|x])^2) / Var(E[Y|x]) on Q 256 minus observed designs (our
  `mda_score_row` and `unobserved_q_mask` rules); binary tasks also `brier` with the law clipped to [0, 1].
- ForceBench: `nmse_head`, `nmse_shift`, `nmse_all` - the law is integrated as the force in our forward stepper
  from the fixed IC (truth method and dt, DiscoverPhysics' softening and inertia) and compared with the truth
  series score_dp_ourq.py uses; floors of the true law are in the pack (n-body worlds ~1e-20, gravity 1.9e-6,
  yukawa 2.6e-8 on the shift designs).

## Every-round variants: `--mode ef | er` (2026-10-08)

The released schedule above (`oracle_calls >= min_data_for_eq` and `oracle_calls % eq_learner_every == 0`) is what
leaves the curve without a law before b = 20 / 12. `--mode` selects an evaluation-schedule variant (asl_kit/schedule.py);
the default `released` is the behaviour above, byte for byte (stub A/B: laws.jsonl, curve.json, curve.csv,
experiments.jsonl identical; the receipt gains the keys `schedule`, `first_fit_b`, `early_termination`, `fits`).

| mode | eq_learner_every | min_data_for_eq | max_experiments_per_iter | first law (our D0; see below) |
|---|---|---|---|---|
| released | 20 (ChemBench) / budget // 5 | 10 | 5 (ChemBench) / budget // 5 | b = 20 / 12 |
| ef (every fit) | 1 | 0 | released | a law after an iteration end once a fit passes; stub cells (death s31 / c0 easy s31 / gravity s31): BoxingGym b = 6, ChemBench b = 10, ForceBench b = 3; between iterations the last law is carried |
| er (every round) | 1 | 0 | 1 | a PySR law normally at 10 stored points: ChemBench b = 7, BoxingGym / ForceBench b = 8; stub cells: ChemBench 7, BoxingGym 8, ForceBench 4 (see below) |

How: a `DiscoveryConfig` subclass replaces `autoscilab.loop.discovery.DiscoveryConfig` for the process (the same
module-attribute replacement as `get_oracle`); the released runners build their config as released and the subclass
overrides the three fields in `__post_init__`, recording the released value beside the applied one (receipt
`schedule.overrides`, `schedule.released_values`). `run_llm_pipeline` imports the class inside the function
(run_chembench_comparison.py 628), so the ChemBench path is covered without editing it; no released file changes.

What stays released: the equation learner's own floors. `fit` returns None below 5 stored points (learner.py 279/534;
powerlaw prefit below 5, skeleton families below 10), and below 10 stored points its 80/20 split leaves ONE validation
point (`n_val = max(1, int(0.2 n))`, learner.py 363), on which `_r2` is NaN unless the prediction is exact (learner.py
1233-1240), so the fitted equation is not valid and `fit` returns None. A PySR law therefore normally first passes at 10
stored points - b = 7 (ChemBench, D0 = 3), b = 8 (BoxingGym, ForceBench, D0 = 2). Before 10 stored points a law can still appear through two released paths, both seen in the stub cells: the power-law prefit (`powerlaw_prefit=True`, discovery.py 176; learner.py 420-423 sets `self._best = quick_fit`; `_power_law_prefit` needs 5 positive training points, i.e. 6 stored, learner.py 1131-1134), which a real run on a positive-valued world (ForceBench) takes as well - the ForceBench er stub's first law at b = 4 (6 points) is this log-log fit; and a validated LLM hypothesis fitted directly into the candidate pool (discovery.py 1900-1915), possible from very few points (ForceBench ef stub b = 3, 5 points). Earlier b are `no_law`. The warm sweeps (one per parameter; under er one
experiment each, since `n_experiments = max(1, max_experiments_per_iter)`); the phases, the acquisition switch, the
staged PySR iterations (ChemBench), the final fit and the LLM arbitration, the leak block, D0 injection. The NewtonBench
config's early stop (`confidence_threshold` 0.9) stays and has more chances under ef / er (receipt
`early_termination`, rows `_carried_after_termination`). Under er the strict NewtonBench loop makes 3 (+1 analysis)
calls per experiment and the ChemBench loop 1 (+ conditional analysis), and PySR runs after every iteration: expect
roughly 5-8 x the released calls and 30-60 min per cell (measure on a canary cell first; in the 0 USD modes the stub's synthetic usage counts against the caps, so
give them caps of about 5 USD per folder and 2 per cell).
The receipt's `fits.rows` lists every `EquationLearner.fit` call (b, stored points, seconds, returned a law) so the
PySR time of a cell can be read directly. Under ef / er the receipt's `released_config` block holds the APPLIED values (it is
`dataclasses.asdict` of the config the loop ran with); the released values are in `schedule.released_values`. The released runs stay as they are.

## Information conditions (fairness)

- **ForceBench is easier for AutoSciLab than for us.** Our engine and ICL see whole noisy trajectories and
  never a force; this oracle lets AutoSciLab measure the force itself at a chosen distance (and time, for
  the oscillator). Only the scoring is on trajectories. The measurement noise is NewtonBench's own 5 %
  relative noise, sigma = max(0.05 |F|, 1e-9) (default, decided 2026-10-06; recorded in every
  receipt). The brief's mapping, sigma_F = 0.05 sqrt(var_F), stays available as `--force-noise additive_var`
  but is not used: sigma_F is 3-8 % of the median |F| on gravity, yukawa and oscillator, 28 % on
  extra_dimensions and 3.9x the median |F| on coulomb_easy (one scoring trajectory passes at r = 0.10, so
  var_F is dominated by that spike), which buries the coulomb signal.
- **ChemBench is as released, i.e. far less masked than ours.** The agent sees the input names, the universal
  grammar (every candidate mechanism with its formula), `DOMAIN: c0_michaelis_menten` (the domain id) and the
  goal sentence ("Discover the enzyme kinetic rate law (Michaelis-Menten)." / "... product feedback
  inhibition." / "... two-substrate pingpong mechanism."; the breadth domains get "... for c26_pingpong_
  noncompetitive."). Our runs see X1..X7 and no description.
- BoxingGym defaults to our masking (X1..Xn, ranges, output type and support) because the released code has no
  BoxingGym text; `--names shown` gives native names and one-line task descriptions (closer to MDA v5).
  Decision (2026-10-06): BoxingGym runs masked. ForceBench is masked the same way (X1..X4, ranges).
- **The three benchmarks therefore differ in masking.** ChemBench's released prompt is unmasked (input names,
  mechanism grammar, goal sentence, domain id), while BoxingGym and ForceBench run under our masking. A
  ChemBench gap between AutoSciLab and our runs mixes the method difference with this information difference.
- BoxingGym counts and binary outcomes are passed to AutoSciLab as real-valued measurements; its GP works on
  log|y| (zeros become log 1e-300) and its PySR fits a deterministic regression. New observations are drawn
  with numpy from the environment's own observation law (same distribution, different random stream); our
  D0 values are reused verbatim. Hyperbolic requests with iR > dR are swapped and rounded; survival requests
  are executed on the nearest unobserved patient of the fixed pool.
- Call budget differs by design: the released agent made 7 (ChemBench), 14-15 (BoxingGym; 7 in a cell that
  stopped early) and 14-15 (ForceBench) Claude calls per run in the stub runs - more when replies request
  fewer experiments than the iteration cap, since outside the forced sweep iterations an iteration runs
  min(requested experiments, cap, remaining budget) - while our protocol makes 10 initial calls plus one
  per round. Seeds only change AutoSciLab's numpy streams (and, on
  BoxingGym/ForceBench, the observation noise); Claude replies are not deterministic.
- PySR's operator set gets sin/cos only for three NewtonBench domain ids (learner.py 75-81), so the oscillator's
  cos(pi t / 2) can enter only through LLM hypotheses fitted directly - released behaviour, not changed.

## Files

- asl_kit/claude_client.py - Claude path, usage ledger, caps
- asl_kit/protocol.py - leak block, D0 injection, per-experiment law recorder, run wrapper
- asl_kit/chem.py, asl_kit/boxinggym.py, asl_kit/forcebench.py - oracles and scorers
- asl_kit/runner.py - `run` / `score`; asl_kit/preflight.py - preflight; asl_kit/collect.py - one CSV per
  runs folder (writes FOLDER_RECEIPT.json); asl_kit/paths.py
- packs/ (not in this tree) - chembench_eval256.json, boxinggym_worlds.json, forcebench_truth.json, built by build/*.py from the benchmark checkouts and our recorded runs (the build scripts carry `<development path>` where a record path was;
  every BoxingGym entry is cross-checked against our recorded cells - seeds, world parameters, D0 designs and
  observations, Q, pool - and survival's Q is read from those recorded cells because re-drawing it with the
  goal's balanced rejection sampler costs tens of minutes per world)
- tests/ - test_claude_wire.py (MockTransport, both SDK pairs), test_kit.py, stub/ (fake PySR, dummy
  together, mock Messages API - used only with --stub / --mock-llm)

## Commands

One cell, 0 USD (`--stub`: mocked Messages API and a least-squares stand-in for PySR). Run from the
repository root with the engine's environment, `paper/baselines` and `src` on `PYTHONPATH`, and the
LLM-AutoSciLab checkout at `ASL_CHECKOUT` (or `BMD_AUTOSCILAB`):

    env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper/baselines/llm_autoscilab_wrapper:src \
      python -B -m asl_kit.runner run --benchmark boxinggym \
      --task death --seed 31 --budget 30 --out <cell> --cap-folder-usd 20 --stub

A paid cell needs the released agent's own environment (PySR with its Julia depot, the released
`requirements`), `ANTHROPIC_API_KEY` in the environment, and the data packs built by `build/*.py`;
`python -B -m asl_kit.preflight` checks that environment first (0 USD). The queue and host scripts of
the development runs are not in this tree.

Mocked calls record synthetic token counts and synthetic USD in llm_usage.jsonl (flagged `mock`), and those
count against the caps, hence the high caps for 0 USD runs. `--mock-llm` alone keeps the real PySR;
`--stub` adds the least-squares stand-in for PySR. A cell that finished after a cap refusal inside
one of the released loop's try/except call sites is marked `completed_with_cap_refusals` (exit 5).
