# Box's Apprentice kit (unit UNI-APPR, 2026-10-08)

Runs BoxingGym's published "Box's Apprentice" agent - the original PyMC box loop of github.com/kanishkg/boxing-gym, in the
checkout <development path> at 06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba (upstream plus two local fix commits:
import paths / exec model loading 7a54a71, program persistence 1aa4510; no prompt changes) - with claude-opus-5-5 on our 15
BoxingGym worlds (death, dugongs, peregrines, hyperbolic x 31/37/41; survival x 43/53/61), 30 experiments after our D0, and writes
the AutoSciLab kit's per-cell schema so the site reader u3_kinds.AutoSciLab reads it unchanged (arm id `apprentice`).
Design: <development path> The checkout is imported,
never edited, never installed (its src/ goes on sys.path; no bytecode is written into it). The released run_experiment.py is
read from the kit's verbatim copy vendor/run_experiment.py (MIT, git blob 92cdabe2, checked against the checkout's tree entry):
the elice hosts' checkouts are sparse (src, conf) and have no root files.

## What runs

`run_experiment.iterative_experiment` (use_ppl=True, include_prior=True, naive_agent None) from the checkout's run_experiment.py
with the CCR-5b v3 patch applied in memory; scientist = the released `LMExperimenter` (free-text `<observe>` designs, MAX_TRIES 3);
box loop = the released `run_box_loop` (conf/box_loop.yaml: num_rounds 1, num_proposals 1, critic, language_synthesize, warm_start
false); checkpoints 0, 2, 5, 10, 20, 30 (checkpoint 0 = the box loop's prior mode). Every change is listed in each receipt's
`interface_changes`; in short:
1. box-loop LLM: `model_search.AsyncOpenAIGPT4V` -> `AnthropicAsOpenAI` (system role -> top-level system, n sequential calls, image
   blocks refused); 2. cost table entry for claude-opus-5-5 (4e-6 / 2e-5 per token); 3. scientist `LMExperimenter("claude-opus-5-5",
   0.0, 512)` with the kit's client; 4. the program (and critic) text, built by the released `augment_scientist_with_ppl` on a
   stand-in, is the Claude system message (rebuilt from the base each time); 5. our 2 D0 observations in the BASE-1 wording appended
   to the system message; 6. CCR-5b v3 (at most 2 re-proposals with the error fed back, then "measurement unavailable").
2. Claude path (appr_kit/claude_path.py, the AutoSciLab kit's conventions): official SDK, Messages API, thinking adaptive, effort
   medium, max_tokens = max(requested, 16000), temperature dropped and counted, refusal recorded then raised, answer = first text
   block; caps checked before every call (spent + worst case of the next call: request chars / 1.9 tokens x 4 + 16,000 x 20 USD per
   1M); usage per call in llm_usage.jsonl at (input + cache) x 4 + output x 20 USD per 1M.
3. The Messages API rejects turns without text and merges consecutive same-role turns: the released prior-mode proposal's empty user
   message is sent as "Please proceed."; the critic's consecutive user messages are sent as one turn with one text block each.
4. Worlds: built the engine's way (appr_kit/worlds.py; = tools/icl_boxinggym World of main_verify_iclb_a75c32c) and checked against
   packs/boxinggym_worlds.json (seeds, world parameters, Q, held-out outcomes, truth, D0) before any call; an exception inside
   `env.run_experiment` is returned as the released invalid-design message instead of ending the cell.

## Per-cell output (`<folder>/boxinggym__<task>__s<seed>/`, log `<folder>/boxinggym__<task>__s<seed>.log`)

receipt.json (schema appr_cell_receipt_v1: args, status completed | completed_with_cap_refusals | failed, error, wall_s,
termination, n_llm_calls, final_law = the last accepted PyMC program, usage, world, checkout, kit_files_sha256, curve_summary,
released_config, interface_changes, ...), curve.json / curve.csv, laws.jsonl (one row per checkpoint: program, critic text,
CCR-5b events and re-proposals, fit status), experiments.jsonl (one row per agent experiment: requested text, tries, executed
design, y), bg_native_scores.json, llm_usage.jsonl, llm_requests.jsonl, llm_responses.jsonl, scientist_messages.json,
box_loop.log, predictions/bNN.npz (the program's predictive means at the 256 Q points), work/ (the released code's CWD: logs/,
ccr5b_retries.json, oed_llms_stan/), ptc/ (PyTensor compile dir). Exit code 0 completed, 5 completed_with_cap_refusals, 4 failed.

**curve.json** has one row per budget b = 0..30: {b, n_obs, source, skeleton, checkpoint, status, nmse_truth, brier (binary tasks),
q_points, q_excluded}. The Apprentice refits its program only at its checkpoints (its own protocol), so:
- a row AT a checkpoint k is scored from that checkpoint's program: the posterior-predictive mean (prior-predictive at k = 0) at our
  256 evaluation points through the released `get_ppl_prediction`, source `checkpoint_<k>`;
- a row BETWEEN checkpoints carries the latest checkpoint's program forward, source `carried_from_checkpoint_<k>`, status scored,
  re-scored on Q minus the designs executed up to that b;
- b before the first checkpoint that has a fit is `no_law`; a checkpoint whose program failed after the CCR-5b retries is
  `law_error` (and so are the rows carried from it); a program that raises or returns a non-finite value at a kept Q point is
  `law_error`.
Scores (appr_kit/scorer.py = the AutoSciLab kit's BoxingGym rules): Q = our 256 evaluation points minus the executed designs (D0 and
the agent's, exact equality); nmse_truth = mean((pred - E[Y|x])^2) / Var_Q(E[Y|x]); brier = mean((clip(pred, 0, 1) - y_heldout)^2)
for hyperbolic and survival. bg_native_scores.json: BoxingGym's own goal.evaluate_predictions on the goal's first 10 stored eval
points (str of the predictive mean, as released) at each checkpoint.

## Every-round variant (2026-10-08): `--prompt-variant no_thought`, `--checkpoints 0,1,...,30`, `--q-prediction batched`

Three options, all off by default (the defaults reproduce the kit's first edition):

1. `--prompt-variant no_thought` (appr_kit/prompt_filter.py). claude-opus-5-5 refused every call of the released prompt
   (stop_reason refusal, 5 of 5 in the 2026-10-08 canary); the released text asks the model to write its reasoning inside
   `<thought>` tags. The released parser never reads that tag (agents/agent.py 68-71 match `<observe>` / `<answer>` only;
   the released code's own `<thought>` strip is commented out, agent.py 29), so the instruction is removed at send time by
   three rules applied in claude_path.ClaudePath.call to the system string and every user turn: the `<thought> your thought
   </thought>` example lines of the environment prompts (envs/death_process.py 144, 297-311; dugongs.py 45-244;
   peregrines.py 40-246; hyperbolic_temporal_discount.py 197-396; survival_analysis.py 192-268), the sentence "Articulate
   your strategy for choosing measurements in <thought>." and the line "Thought: <thought>" of the scientist's follow-ups
   (agent.py 111, 113). "You may also think before providing your predictions." (no tag), the `<observe>` / `<answer>`
   lines and the retry prompts stay. The released objects keep their text (scientist_messages.json is unfiltered);
   llm_requests.jsonl holds what was sent; every usage row carries `prompt_variant` and `prompt_filter` (per-rule counts,
   system sha256 before / after), the receipt `prompt_variant`, `prompt_filter`, `prompt_filter_totals` and an
   `interface_changes` entry. A request that still carried `<thought>` after the filter is refused, not sent.
   `python -m appr_kit.probe_prompts --task dugongs --seed 31 --out <file.jsonl>` writes the scientist's first two
   requests for both variants (exact bodies) for the refusal probe.
2. `--checkpoints 0,1,2,...,30`: the released `iterative_experiment` runs its checkpoint block (`if i+1 in num_experiments`:
   box loop proposal + critic, PyMC fit, our Q scoring) after every experiment - the same code with a denser list. Calls per
   cell: 30 scientist + 31 proposals + 30 critics = 91 (the first edition's 6 checkpoints made 41), plus CCR-5b re-proposals.
3. `--q-prediction batched` (shim.Recorder.q_predictions): the kit's scorer predicts the 256 Q points in one predictive
   sampling run (`construct_features` / `pm.set_data` over all rows, mean over draws per row) instead of one released
   `get_ppl_prediction` call per point (75-85 s per checkpoint in the stub cells). Same program, trace and draws per row;
   only the Monte Carlo draws differ (tests/test_kit.py compares both paths on the scripted death program). A program that
   cannot take 256 rows falls back to the per-point path for that checkpoint (receipt `q_fallbacks`). Scoring only.

Plans: `plans/apprentice_15_er.txt` (15 cells with all three), `plans/canary_er.txt` (dugongs 31), `plans/mock_er.txt`
(0 USD, short checkpoint lists). Arm id for the site: `apprentice_er` ("Box's Apprentice - every round").

## Commands

Modes: `stub` (scripted replies in-process, no SDK), `mock-transport` (the anthropic SDK over httpx.MockTransport), `paid`. Both 0 USD
modes use the scripted replies of appr_kit/scripted.py (one fixed design per task, one small generic PyMC program per task; written for
the check, nothing from a recorded run). appr_env.sh is the site configuration (outside SHA256SUMS; defaults = the elice hosts).

    bash provision_appr_elice.sh <host> <evals commit> [kit path in the evals repo] [evals branch]   # host setup + 0 USD gates
    "$APPR_VENV/bin/python" -B -m appr_kit.preflight [--quiet] [--mode paid]                       # from the kit folder
    bash queue.sh plans/apprentice_15.txt <folder> <cap folder USD> <cap cell USD> <stub|mock-transport|paid> [MAXP]
    bash run_cell.sh <folder> <cap folder> <cap cell> <mode> boxinggym <task> <seed> 30          # one cell (queue.sh calls it)
    "$APPR_VENV/bin/python" -B -m appr_kit.check_cell <cell> ...                                   # 0 USD cells end to end
    "$APPR_VENV/bin/python" -B -m appr_kit.collect <folder> --out <csv>                            # queue.sh calls it at the end
    "$APPR_VENV/bin/python" -B -m appr_kit.cost_estimate <folder of 0 USD cells>                   # paid-cost projection
    env -u FORCE_COLOR "$APPR_VENV/bin/python" -B tests/test_claude_wire.py   (and tests/test_kit.py, tests/test_engine_equality.py)

In paid mode run_cell.sh loads APPR_SECRETS (ANTHROPIC_API_KEY by name; never printed); the 0 USD modes never read it and unset the key.
