# ICL baseline harness

Implements the separately registered FIXED, CLOSED and NAME-VISIBLE units.
No engine or adapter source is modified. This tool is not a paid-execution approval.

## Offline execution

Run from the checkout with its `src` and repository root first on `PYTHONPATH`.
The existing AutoSciLab source must also be importable; no dependencies are installed.

```bash
env -u FORCE_COLOR OPENAI_API_KEY= ANTHROPIC_API_KEY= OPENAI_COMPAT_API_KEY= OPENROUTER_API_KEY= \
  PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}:${BMD_AUTOSCILAB}" \
  PYTENSOR_FLAGS="base_compiledir=${SCRATCH}/pytensor" \
  "$PYTHON" -B -m tools.icl_baseline.icl_baseline \
  --unit ICL-MASKED-FIXED-HISTORY --dry-run --output "$SCRATCH/fixed" \
  --registration "$REGISTRATION" \
  --history "$MASKED_1" --history "$MASKED_2" --history "$MASKED_3"
```

History argument order fixes replicates 1, 2, 3, not a result-dependent sort.
For FIXED, add three `--fac-history` arguments in the same replicate order once
they exist. FIXED execution refuses an incomplete six-history group; a three-history
dry run is permitted. NAME-VISIBLE takes the three *shown* histories in `--history`.
CLOSED takes the three masked histories as its fixed task/D0/evaluation declaration.
Every output directory must be new. Existing artifacts are never repaired or resumed.

Dry-run request counts: FIXED 138 (three histories), 270 (six); CLOSED 318;
NAME-VISIBLE 96. CLOSED dry-run includes deterministic fake-client selections and
real zero-cost simulator observations, because future live requests cannot exist yet.
`corpus/closed_d0/` holds requests with actual D0 observations;
`corpus/closed_synthetic/` marks all later requests `SYNTHETIC`. These are not a seal
of future live requests. `--mock` executes the harness with the same fake client.
Synthetic scores and synthetic ledger charges are plumbing diagnostics, not evidence
or actual USD expenditure. Receipts record `paid_api_calls: 0` explicitly.

## Task mode: CLOSED over the nine paper tasks

`--unit ICL-MASKED-CLOSED-LOOP` without any `--history` runs in task mode. `--task` takes one
of the nine ids of `manifests/chembench_paper_manifest.json` (default
`c5_pingpong_bisubstrate__easy__v0`); any other id is refused by the argument parser.

```bash
"$PYTHON" -B -m tools.icl_baseline.icl_baseline --unit ICL-MASKED-CLOSED-LOOP --dry-run \
  --task c0_michaelis_menten__easy__v0 --output "$SCRATCH/closed_c0_easy" --registration "$REGISTRATION"
```

- The start is the engine's fixed3 designs (`chembench_core.fixed3_designs`), observed by the
  task's own oracle through `ChemBenchSession` at noise 0.01, exactly as the runner observes.
- The evaluation points are the runner's: 256 from the `chem_evaluation_inputs` stream and the
  official 1,000. Reference values are asked of a session that has finished.
- Prompts, request counts (318), the 90 USD cap and the projection rules are unchanged. One
  invocation is one task with its own ledger; nothing carries over between tasks.
- The seal draft and both receipts name the task (`task`, `task_id`); a live run needs a sealed
  manifest whose `task` and `history_assignment` match. No run of ours is read, so every
  direction field says `no directional statement`; the comparison is made outside this tool.
- For c5 easy the draft's `d0_prediction_requests` and `settings` equal the three-history
  CLOSED seal's, and every request body equals the three-history dry run's.
- Run it under the comparison runs' CPU feature policy. With AVX-512 disabled the evaluation
  points of the log-sampled inputs move by one unit in the last place.

FIXED and NAME-VISIBLE, and CLOSED with three `--history` arguments, behave as before and stay
c5 easy only.

## Contracts

- The three prompt assets were extracted from registration section 5, without rewriting
  the prose; section markers remain in the assets and are removed by the renderer.
  The selection template's declared common prefix is expanded from the prediction
  template. Visible naming applies only the registered substitutions. Repair prose
  is part of the pinned assets. Tests pin all three asset hashes.
- Input order comes from the declared interface, never dictionary insertion order.
  Only `x`, `y`, and the observation index survive the history boundary. Observations
  are sorted on original coordinates and index; displayed values reuse
  `belief_packet.round_significant`. Query mappings retain exact file-order coordinates.
- Prediction parsing preserves duplicate keys, refuses malformed values, and keeps
  each unambiguous valid first-attempt ID before repair/fallback. A duplicated ID is
  not valid; unrelated valid IDs survive a duplicate or extra query key. An invalid
  root or `max_tokens` stop yields no usable values. No accuracy-based retries or clipping.
- Selection has an explicit narrow function signature; no evaluation/response/summary
  context is accepted. It executes unrounded coordinates. Seven fallbacks mark design
  failure but do not end the 60 selections. CLOSED claims are unpaired, never paired
  with the arbitrarily same-numbered reference history.
- Both targets use the engine's `log_error_metrics` separately. Constant/valid-only
  scores, invalid-timepoint ratios and magnitude strata remain separate. A zero target
  variance is undefined. Official existing scores lack the second target: no two-target
  direction is issued at 1,000 points, and no posterior is refit or rescored.
- The forbidden-word precheck is case-insensitive at identifier/word boundaries;
  a raw substring search for `rate` would incorrectly reject the frozen word `generate`.
  Domain-word incidence in thinking uses the E6 case-insensitive substring/stem
  regex separately, including `ping-pong`, `ping pong`, and `pingpong`.

## Transport, costs, and sealing

Default: model `claude-opus-5-5`, effort medium, adaptive/summarized thinking,
32,000 output tokens, streaming, SDK retries 0, timeout 300 seconds. The separately
sealed 16,000-token nonstreaming path is retained. No temperature is sent. Every
logical request has at most two attempts, including transport failures. Only text,
not thinking, enters a repair conversation. SDK wire tests use `httpx.MockTransport`.
Empty or whitespace-only replies resend the original request on the second attempt;
they never create an empty assistant turn. The `max_tokens` failure has precedence.

Each request first reserves `(input_upper_bound * 4 + max_tokens * 20) / 1e6` USD.
The local input bound is UTF-8 request bytes plus 1,024 tokens of framing allowance;
this is deliberately conservative, not a claim to have measured provider tokens.
No token-counting API is called. Unknown usage or transport failure charges the full
reservation; thinking is already included in reported output tokens, never double-counted.
The registered `c_hi` check and once-only ten-call projection remain additional early
stops. This is **more conservative than the registered rule**, approved by the user.
Implemented caps are 90/90/25 USD and cannot transfer across units. Their use in a
sealed paid run still requires the pending the review board decision.
After ten calls, `c_hi` uses only `cost_basis == "usage"` calls (maximum times 1.25).
If none exist it uses the current worst-case reservation. Unknown/cache usage still
charges the full reservation to cumulative spending; it does not inflate that maximum.
Since October 2 (unit ICL-K) `c_hi` is kept **per request kind** (prediction, selection):
the projection is spent plus, for each kind, that kind's remaining initial requests times
that kind's `c_hi`, and the registered `c_hi` check uses the next request's kind. A kind
without a usage-based cost in the first ten calls keeps the reservation rule (its largest
recorded reservation, or the current request's). The settings carry
`projection_rule = per_kind_max_cost_x1.25_after_10_calls`, each accounting row its
kind's `c_hi_kind_usd`, and the receipt a `projection` block (spent, remaining and `c_hi`
by kind, the projected total, the cap, and the same after an official-layer reduction).
The cap, the reservation rule, the ten-call timing and the 1.25 factor are unchanged.

The October 1 F5 implementation attaches an official-layer reduction to FIXED and
CLOSED: if the ten-call projection exceeds the cap, remove unsent 1,000-point
requests and reproject before stopping. Keep all 256-point requests and CLOSED
selections. A request with an already-sent first attempt retains its retry; an
unsent request that triggered reduction is not sent. Its existing corpus file is
retained with a separate not-sent outcome. Incomplete official timepoints are
listed in the receipt, not scored as complete 1,000-point evaluations. The receipt
also records completed/sent/removed counts at reduction. Without a projection
excess the hook changes no request or scientific result.

F5 and the 90/90 caps await the review board approval for sealing/execution. Both are implemented
with user approval, including in mock runs, but this code is not execution authority.
No new final seal or paid run is produced by this unit.

NAME-VISIBLE executes n=3/10/25/63 for all six history/variant pairs first. If its
ten-call projection exceeds 25 USD, it drops the other four timepoints. It stops
if the reduced plan still exceeds the cap. Per-call maximum reservations still apply.
Budget stops retain completed results and report remaining requests as incomplete;
no fallback score is manufactured for a call that never occurred.

`seal_draft.json` is **not a seal**: status `draft_not_execution_authority` is refused
by the live gate. A subsequently approved manifest must bind code, prompts, price
table, imported engine/adapter/oracle source, registration, input files, exact history
assignment, transport settings, D0 request hashes, and RNG rules. Synthetic future
requests are excluded. Live construction requires both `--allow-paid-api` and a
matching `--seal-manifest`. NAME-VISIBLE additionally requires two `--masked-receipt`
paths (FIXED and CLOSED live receipts) hashed in the seal, and the seal's explicit
`masked_receipts_reviewed: true`. The tool does not perform that review or authorize itself.

Artifacts: starting receipt, immutable request/response/validation files, per-request
outcomes, closed trajectories, separate-target scores, final receipt, cost plan, seal
draft. A transport response's accounting status is `response_received`; its separate
validation file and final receipt give `valid` or `retryable_validation_failure`.

## Tests and read-only prechecks

```bash
env -u FORCE_COLOR PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}:${BMD_AUTOSCILAB}" \
  "$PYTHON" -B -m pytest -p no:cacheprovider -q tools/icl_baseline/tests
```

All test histories and responses are synthetic. `tools.icl_baseline.precheck` separately
reads the authorized published inputs, checks nine packet tables, all six evaluation
and target arrays, score anchors, and simulator-only input quantization sensitivity.
It hashes inspected inputs before and after. Existing results are never used as test
fixtures or prompt examples, and are never modified.

The hand-back includes source identities and the precise commands/results. FA-C inputs,
final manifest approval, global spending verification, and paid GO remain external
prerequisites. No paid run is part of this unit.
