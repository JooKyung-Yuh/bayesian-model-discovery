# DP-Agent baseline kit for the elice fleet (2026-10-08, main)

The DP-Agent is the LLM agent of the DiscoverPhysics benchmark (Wiemann, Smith, Melchior, Mishra-Sharma, Wilson, Izmailov,
Cuesta-Lazaro, arXiv 2605.26087; code github.com/SampsonML/DiscoverPhysics, commit 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5 = upstream main "Add citation").
MDA (arXiv 2608.09696) runs it as its ForceBench baseline in a throttled form: one experiment per request, at most b experiments,
b + 2 rounds, a final law from round b + 1. `kit/` is the 2026-10-05 unit BASE kit, byte-identical (sha256 in kit/SHA256SUMS is not
kept there; this folder's SHA256SUMS covers every file): the wrapper `dp_agent_throttled.py` patches the unedited ScienceAgent
in-process (claude-opus-5-5 agent, claude-haiku-4-5-20251001 explanation judge, DP's own request settings), enforces the budget
with `dp_budget_guard.py`, writes per-cell `.protocol.json / .results.json / .usage.json / .wrapper.json`, and embeds the fixed
start state of our five ForceBench worlds (FB1_FIXED_IC_MANIFEST_v5, sha256 63067f13...), so no engine tree is needed on the host.

Cells: 5 worlds (gravity, yukawa, coulomb_easy, oscillator, extra_dimensions) x b in {2, 4, 8, 16} x seed in {0, 1, 2} = 60
(`kit/plan_dp_60.txt`, order seed -> budget -> world); canary `kit/plan_dp_canary.txt` = coulomb_easy 16 0; 0 USD plans `plan_dp_mock.txt`.
Observation noise sigma = 0.05*sqrt var of the world; DP's noise seed = the cell's seed. Scoring happens on the development host afterwards with
<development path> on our shift-set designs.

## Every-round variant (2026-10-08): `DP_PROTOCOL=current_law`

The released agent states a law once, at the end (`<final_law>`); in the 60 cells above no round carries a law (the agent
never used DP's optional `<run_mse_fit>` tool: 0 of 60 run logs), so a per-experiment curve cannot be read from them. With
`DP_PROTOCOL=current_law` (run_dp_cell.sh / queue_dp.sh pass `--protocol current_law`; dp_agent_throttled.py item 11) the
wrapper's constraint line - already an interface text of the throttled protocol - also asks the agent to state its current
best law inside `<current_law>...</current_law>` in every experiment round, in the `<final_law>` format. DP's agent ignores
the tag (it reads `<run_experiment>`, `<run_mse_fit>`, `<final_law>`), nothing is fed back, and the agent's loop, its
prompts and DP's end-of-mission fit are unchanged. After the mission `kit/dp_current_law.py` harvests the tags from the
run log: b = the number of experiments actually EXECUTED before the statement (a refused or failed `<run_experiment>`, a
`no_tag` or `warning` round adds none; when two statements share one b the latest wins). One fitting rule for every row of
the curve, b = B included (REVIEW-ER, 2026-10-08): the law's declared constants are refitted on ALL executed experiments
seen at that b with DP's own fitting functions (scienceagent.evaluator `_compile_law`, `_compile_fit_parameters`,
`_validate_fit_spec`, `_fit_law_parameters`, the world's loss - what `mse_fitting.fit_law` uses), read from the run log's
experiment_input / experiment_output (the wrapper disables the trajectory CSV); `fit: harvest_all`. DP's own end-of-mission
constants (fitted only for run_discovery.py's `_FIT_WORLDS` - coulomb_easy is not one - and on a 4 x 5 subsample,
evaluator.py FIT_MAX_TRAJECTORIES / FIT_MAX_TIMES_PER_TRAJ) are recorded beside as `dp_final_*` and scored as the
as-released point (`as_released_final` in score_dp_rounds.py). b = 0 is the round-1 law at its initial constants. Output
`<stem>.current_laws.json` (schema dp_current_laws_v2); a b without a statement is `missing`. `kit/test_dp_current_law.py`
checks the b labels and the final-row rule on a synthetic run log. Scoring (the development host): `kit/score_dp_rounds.py <runs> --out
<scores.json>` reuses score_dp_ourq.py as a module (same worlds, forecast and nmse) and writes one row per b with status
scored / carried (previous law, `carried_from`) / no_law / law_error. Cell stems and logs carry `_cl`
(`<world>_thr_b<B>_s<seed>_cl`), so the released cells are never confused or overwritten. Plans: `kit/plan_dp_er_15.txt`
(5 worlds x b = 16 x 3 seeds), `kit/plan_dp_er_canary.txt`, `kit/plan_dp_er_mock.txt` (0 USD). The 0 USD checks:
`check_dry_dp.py <runs> <stub|mock-transport> current_law` (the scripted agent states the law in every round but the third,
so the missing-round path is exercised). Site arm id `dpagent_er` ("DP-Agent - every round (1 experiment per round)").

Host steps: `provision_dp_elice.sh <host> <evals commit>` (0 USD; ends with SUMMARY fail=0 and GUARD OK / DRY CHECK OK x2) ->
paid canary (cap 3 USD) after the user's OK -> the other 59 cells (`queue_dp.sh plan_dp_60.txt <runs> paid 35 6`, the canary cell
prints `exists`) -> publish with `publish_dp.py <runs> <dest>` + `scan_keys.py <dest>` (0 key-like hits) -> push as a snapshot folder.
Expected spend 23-31 USD (up to about 58 with twice-longer replies); the cap stops launching at 35.
