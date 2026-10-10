# Reproducibility

Abbreviations used below and in the quoted record strings: EIG = expected information gain, EPIG = expected predictive information gain, BMA = Bayesian model averaging, SMC = sequential Monte Carlo, KL = Kullback-Leibler divergence, nMSE = normalized mean squared error, D0 = the initial design set.

## Where the paper's code lives

The experiments of the paper ran from the private development repository (not public), which also keeps the whole development history. The exact engine state of each benchmark's runs is frozen there as an annotated tag.

| benchmark | tag in the private repository | commit |
| --- | --- | --- |
| BoxingGym | `paper-2026-10/boxinggym-u3` | `ca881acfa7b82ca9acbb3929b413c0f0c4e3a72e` |
| ChemBench (depth and breadth) | `paper-2026-10/chembench-u3` | `80009416375ef35b3b2e7548bc41f618cedf74a1` |
| ForceBench | `paper-2026-10/forcebench-u3` | `33f2cf7d52668acd61b2de33a91eac21ce807143` |
| NeuronBench | `paper-2026-10/neuronbench-u3` | `5cdaf8f16ece1410917f1243138ccfbefa423020` |

## What this public tree is

This repository was built from the verified development tree of the private repository by keeping only the files reachable from the public entry points (`scripts/bmd_run.py` and the four runner modules it starts, `tools/fb1_snapshot/run_gravity.py`, the package root, the example notebooks and the CI workflow) and the tests of those files, and then rewriting comments, docstrings and user-facing strings in English. **The logic is the development tree's**: the check `tools/release/ast_equiv.py` parses every kept Python file in both trees, removes docstrings, and compares the abstract syntax trees (comments, docstrings and formatting are not part of a syntax tree). Its result for this tree:

- 140 Python files identical, 49 different, 1 new (the definition: both files parsed with Python's `ast`, module, class and function docstrings removed, `ast.dump` without positions compared).
- The 49 different files split by a second, stricter comparison (the generator of `CHANGED_LITERALS.md` blanks every string constant and compares again): 34 differ only in string literals, 15 differ in structure; 34 + 15 = 49.
- The string literals: 125 in 39 files (exception and assertion messages, record labels, skip reasons and `match=` patterns rewritten in English or with an internal code name spelled out), each listed with its file and line in [`CHANGED_LITERALS.md`](CHANGED_LITERALS.md). No condition, computation, default value or control flow changed. In particular:
  - `conftest.py` and `scripts/bmd_run.py`: the BoxingGym source the tests accept and the run script names is the public upstream commit `b43e38cb03d09c13efa9cf4d9bae740d51157bfd` of https://github.com/kanishkg/boxing-gym instead of the fork commit `06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba` the paper runs used. The environments the runner imports (`src/boxing_gym/envs/`) are identical in the two commits; the fork changes four files under `src/boxing_gym/agents/`, of which the environments import only `construct_dataframe`, unchanged.
  - `tools/fb1_snapshot/fb1_grammar.py`: the two constants that named the development tree's path and chain of the vendored `count_models.py` (`VENDOR_SOURCE`, `VENDOR_CHAIN`) now describe it in words.
  - `src/bayes_model_discovery/experimental/test_chembench_core.py`: the pinned sha256 of `adapters/chembench_observation.py` is the public file's (its docstrings are in English; its syntax tree is unchanged).
  - `tools/fb1_snapshot/test_fb1_reasoning_and_paid_call_guard.py` and `tools/fb1_snapshot/SHA256SUMS`: the pinned digests of the ForceBench kit files are the public files'.
  - `src/bayes_model_discovery/experimental/test_design_pool_refusal.py`: the expected refusal message follows the rewritten literal in `design_search.py`.
  - A few strings that runners write into run records changed wording with the rest (for example the reason recorded for a fixed-pool ChemBench cell and the ForceBench kit's proposal-list version and rejection reasons); the mock comparison below shows exactly which record fields differ because of them.
- The 15 files that differ in structure are all test files, for one of four reasons: a test of lab-only tooling that is not in this tree was removed, a test that read a development-history commit with `git show` was removed, a test that needs an external benchmark source now skips when the source is absent, or a test stub was brought up to the engine's call signature:
  - `src/bayes_model_discovery/experimental/test_belief_packet.py`: a test that compared the prompt file with a private decision document read from the development machine (it always skipped here) and the constants only it used (removed: `REV2_NOTE_BEFORE`, `REV2_NOTE_AFTER`, `REV2_GOAL_BEFORE`, `REV2_GOAL_AFTER`, `DECISION_REV2`, `test_prompt_file_is_rev2_with_exactly_the_two_documented_edits`)
  - `src/bayes_model_discovery/experimental/test_boxinggym_mda_options.py`: a byte comparison against a git worktree of a development-history commit (it was already marked skip) (removed: `test_default_bytes_match_baseline_except_new_keys_and_verified_source_hash`)
  - `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py`: three tests that run the real BoxingGym environment now skip when its source is absent (the tree's `require_boxinggym_source` fixture)
  - `src/bayes_model_discovery/experimental/test_chembench_mda_options.py`: a byte comparison against a git worktree of a development-history commit (it was already marked skip) (removed: `HARNESS`, `test_default_bytes_match_base_except_new_keys_and_verified_source_hash`)
  - `src/bayes_model_discovery/experimental/test_count_nan_robustness.py`: one test function renamed to drop a development host name (its body is unchanged) (removed: `test_the_bai_failure_mode_reproduces_and_stays_catchable`)
  - `src/bayes_model_discovery/experimental/test_epig_acquisition_efficiency.py`: the same stub fix for the two path-box tests
  - `src/bayes_model_discovery/experimental/test_epig_positivity_guard.py`: the stubbed `sample_epig_targets` accepts the engine's `count` keyword argument (the stub predates that argument; the five tests using it failed in the development tree too)
  - `src/bayes_model_discovery/experimental/test_grammar_version_binding.py`: tests of the lab's research-wave generator and its driver shell script, neither of which is in this tree (removed: `test_the_generator_registers_the_grammar_only_when_it_is_not_the_default`, `test_the_generator_emits_the_driver_option_only_when_it_is_not_the_default`, `test_the_driver_passes_the_option_through_and_never_in_a_shakedown`, `_repository_root`, `_generator`)
  - `src/bayes_model_discovery/experimental/test_model_id_power_normal_form.py`: two test functions renamed to drop the name of the project's review board (their bodies are unchanged) (removed: `test_fugu_equal_spellings_are_one_model`, `test_fugu_different_forms_stay_apart`)
  - `src/bayes_model_discovery/experimental/test_packet_constraints_explicit.py`: a test of the same lab driver script (removed: `REPO_ROOT`, `DRIVER`, `driver_dry_run`, `test_the_driver_passes_an_explicit_v2_0_to_every_bmd_runner`)
  - `src/bayes_model_discovery/experimental/test_paid_call_guard.py`: a test that loaded the transport of a development-history commit with `git show` to compare its record bytes (it always skipped outside that history) and its loader (removed: `BASE_COMMIT`, `base_transport`, `test_the_omitted_option_writes_the_base_commits_transport_bytes`)
  - `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py`: the unit tests that compared `log_predictive` with the module of a development-history commit read by `git show` (they always skipped outside that history), their two fixtures and the helpers only they used (removed: `BASE_COMMIT`, `old_source`, `old`, twelve test functions and ten helpers; listed in `CHANGED_LITERALS.md`); its two remaining unit tests and the three opt-in integration tests are unchanged
  - `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py`: two tests whose refusal is raised after the ChemBench oracle is imported now skip when its source is absent (`require_autoscilab_source`)
  - `tools/fb1_snapshot/test_fb1_label_rule.py`: tests of the lab's output-contract emitter (`emit_contract.py`), which is not carried over (removed: `test_packager_keeps_production_only_from_bound_receipt`, `test_packager_rejects_unbound_or_changed_receipts_before_output`, `test_historical_unbound_production_string_stays_development`, `test_explicit_development_mock_receipt_reaches_packaged_provenance`, `test_packager_reopens_manifest_before_returning_success`)
  - `tools/fb1_snapshot/test_fb1_proposer.py`: one test of the same emitter and its helper (removed: `_minimal_run_json_29fb`, `test_29fb_5_emitter_points_csv_kl_column_gets_a_value_other_columns_unchanged`)
- The 1 new file(s), the release tooling: `tools/release/ast_equiv.py`;
- 114 Python files of the development tree not carried over (unreachable from the entry points, lab-only tools, tests of them, or an earlier implementation that the paper runs do not use; `tests/conftest.py` among them, whose two options belonged to a removed test).

No module outside the tests differs from the development tree in anything but string literals.

## Mock closed-loop equivalence

Each benchmark's runner was run with its built-in mock proposer (no language model, no cost) from the frozen development tree of the paper runs and from this tree, with the same seed and options, and the two run folders were compared file by file after removing volatile fields (times, durations, paths, hashes of paths).

| benchmark | command | files compared | identical | different | only in one tree |
| --- | --- | ---: | ---: | ---: | ---: |
| synthcount | `python -m bayes_model_discovery.experimental.core_count --output <out> --seed 11 --cell 111 --rounds 3 --proposer mock --profile dev` | 69 | 69 | 0 | 0 |
| boxinggym | `python -m bayes_model_discovery.experimental.boxinggym_core --output <out> --seed 31 --cell 111 --rounds 3 --proposer mock --profile dev --task death` | 70 | 68 | 2 | 0 |
| neuronbench | `python -m bayes_model_discovery.experimental.neuron_core --output <out> --seed 31 --cell 111 --rounds 1 --proposer mock --profile dev --world h_sag` | 35 | 35 | 0 | 0 |
| chembench | `python -m bayes_model_discovery.experimental.chembench_core --output <out> --seed 31 --cell 101 --rounds 3 --proposer mock --design-search ascent --profile dev --manifest chembench_dev_manifest.json --noise 0.01` | 70 | 68 | 2 | 0 |
| forcebench | `bash run_cell.sh gravity 31 joint_eig <out> mock` | 44 | 7 | 37 | 0 |

**boxinggym**: 2 differing file(s), grouped by what differs (volatile fields such as times and timings are left out):
- `config.json`:
  - declared literal `/eval_question_note` now reads "the numeric goal.eval_points path is used, exactly as the PPL path does, so the upstream get_goal_eval_questio"
  - declared literal `/eval_question_policy` now reads "restored question"
  - declared literal `/task_spec/design_pool_exhaustion_policy` now reads "DesignPool.available (:706-720) raises ValueError once every design in the pool has already been observed. Ear"
  - sha256 of rewritten files or packets (9): `/source_files_sha256/belief_packet.py`, `/source_files_sha256/boxinggym_core.py`, `/source_files_sha256/core_count.py`, `/source_files_sha256/count_inference.py`, `/source_files_sha256/count_models.py`, `/source_files_sha256/count_proposer.py`, `/source_files_sha256/model_identity.py`, `/source_files_sha256/positive_inference.py`, `/source_files_sha256/real_inference.py`
- `summary.json`:
  - declared literal `/eval_question_policy` now reads "restored question"

**chembench**: 2 differing file(s), grouped by what differs (volatile fields such as times and timings are left out):
- `config.json`:
  - declared literal `/kl_truth_density/grounds/engine_scoping` now reads "the engine's truth-hook design"
  - declared literal `/official_evaluation/baseline_source` now reads "the comparison method's evaluation code (its ChemBench run manifest, eval_points.rule)"
  - declared literal `/oracle_package/resolution` now reads "PYTHONPATH only, never installed (a project decision)"
  - declared literal `/primary_metric/baseline_definition_source` now reads "the comparison method's evaluation code (its ChemBench run manifest, metric_defs.nmse_mda)"
  - declared literal `/profile_grid/note` now reads "fixed profile table; `design_pool`/`eig_samples` are pre-declared for the stage-2 acquisition axis and unused "
  - sha256 of rewritten files or packets (9): `/adapter_sha256`, `/label_receipt/policy_source_sha256`, `/source_sha256/chembench_core.py`, `/source_sha256/core_count.py`, `/source_sha256/count_inference.py`, `/source_sha256/count_models.py`, `/source_sha256/design_search.py`, `/source_sha256/model_identity.py`, `/source_sha256/positive_inference.py`
  - path of the run folder or the tree (2): `/adapter_source`, `/manifest`
- `official_scores_after_finish.json`:
  - declared literal `/metric_definitions/kl` now reads "mean_Q[ KL( Normal(noiseless_observation(x_q), (noise*noiseless_observation(x_q))**2) truncated at 0 || BMA ) "

**forcebench**: 37 differing file(s), grouped by what differs (volatile fields such as times and timings are left out):
- `FB1_RUN_gravity_111_seed31_v3_free.json`:
  - declared literal `/arm_config/background_basis` now reads "domain_size=20.0, periodic=True -> R=10, b0=1/(2 pi R^2) (digits derived from the metadata)"
  - declared literal `/arm_config/structure_proposals/catalogue_not_expressible/inv_r_tmod` now reads "cos/sin are not in the vendored CALL_ARITY (exp·log·sqrt·tanh·square·power)"
  - declared literal `/arm_config/structure_proposals/catalogue_prior_scale_differences/inv_r_bg` now reads "catalogue kinds = ('pos', 'pos_bg'); pos_bg is LogNormal(log b0, 1), b0 = 1/(2 pi R^2) (fb1_core.B0_BACKGROUND"
  - declared literal `/arm_config/structure_proposals/duplicate_detection/catalogue_is_a_reference_table` now reads "Matching a catalogue law is a LINK to that catalogue name, not a rejection. Only an already registered FREE st"
  - declared literal `/arm_config/structure_proposals/duplicate_detection/stage_0_finiteness_first` now reads "The finiteness probe runs BEFORE any equivalence comparison. If the candidate has even one non-finite probe po"
  - declared literal `/arm_config/structure_proposals/duplicate_detection/stage_1_canonical` now reads "canonical_model_id over the canonical AST + coefficient roles (shared with CORE, left untouched); commutative "
  - declared literal `/arm_config/structure_proposals/duplicate_detection/stage_2_numerical` now reads "ForceBench only: before admission the candidate force is compared with every registered free structure and the"
  - declared literal `/arm_config/structure_proposals/duplicate_detection/stage_2_verdict` now reads "A match is worded as indistinguishable_on_the_probe_grid. the probe grid is finite and pre-declared: a match m"
  - declared literal `/arm_config/structure_proposals/probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/arm_config/structure_proposals/probe/rule` now reads "r grid x ALL measurement times x prior draws x (p1,p2) points = the 4 declared box CORNERS plus a pre-declared"
  - declared literal `/arm_config/structure_proposals/registered[]/numerical_equivalence/limitation` now reads "coefficients are paired by trying EVERY PRIOR-PRESERVING permutation (at most 6, since a free law has at most "
  - declared literal `/arm_config/structure_proposals/registered[]/prior_probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/arm_config/structure_proposals/registered[]/prior_probe/rule` now reads "r grid x ALL measurement times x prior draws x (p1,p2) points = the 4 declared box CORNERS plus a pre-declared"
  - declared literal `/arm_config/structure_proposals/vendor_chain` now reads "bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored)"
  - declared literal `/arm_config/unified_protocol/protocol` now reads "unified protocol (registered 2026-10-06)"
  - declared literal `/checkpoints/2/brier_reason` now reads "ForceBench outcomes are continuous - a binary Brier score is undefined"
  - declared literal `/checkpoints/2/single_top_model/note` now reads "the same score_set, held-out Q and truth as the BMA entry; a scoring-only key (the main random stream is untou"
  - declared literal `/checkpoints/4/brier_reason` now reads "ForceBench outcomes are continuous - a binary Brier score is undefined"
  - declared literal `/checkpoints/4/single_top_model/note` now reads "the same score_set, held-out Q and truth as the BMA entry; a scoring-only key (the main random stream is untou"
  - declared literal `/eig_components/model_eig` now reads "E[log p_k(y|d) - log p(y|d)], k = the outer draw's own model, p_k = model k's predictive averaged over its inn"
  - declared literal `/observations/note` now reads "the noisy observed trajectories returned by world.observe(); order = designs."
  - declared literal `/proposal_list_version` now reads "v3 final (background term + time modulation + product coupling + crossover)"
  - declared literal `/proposer/archive[]/numerical_equivalence/limitation` now reads "coefficients are paired by trying EVERY PRIOR-PRESERVING permutation (at most 6, since a free law has at most "
  - declared literal `/proposer/archive[]/prior_probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/proposer/archive[]/prior_probe/rule` now reads "X1 grid x ALL measurement times x prior draws x (X2,X3) points = the 4 declared box CORNERS plus a pre-declare"
  - declared literal `/proposer/constraints/grammar/duplicate_detection/stage_0_finiteness_first` now reads "The finiteness probe runs BEFORE any equivalence comparison. If the candidate has even one non-finite probe po"
  - declared literal `/proposer/constraints/grammar/duplicate_detection/stage_1_canonical` now reads "canonical_model_id over the canonical AST + coefficient roles (shared with CORE, left untouched); commutative "
  - declared literal `/proposer/constraints/grammar/duplicate_detection/stage_2_numerical` now reads "ForceBench only: before admission the candidate force is compared with every registered free structure on the "
  - declared literal `/proposer/constraints/grammar/duplicate_detection/stage_2_verdict` now reads "A match is worded as indistinguishable_on_the_probe_grid. the probe grid is finite and pre-declared: a match m"
  - declared literal `/proposer/constraints/grammar/probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/proposer/constraints/grammar/probe/rule` now reads "X1 grid x ALL measurement times x prior draws x (X2,X3) points = the 4 declared box CORNERS plus a pre-declare"
  - declared literal `/proposer/constraints/grammar/vendor_chain` now reads "bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored)"
  - declared literal `/recovery_mode_note` now reads "candidates are free-form expressions over the named inputs {r, p1, p2, t}, validated by the shared CORE gramma"
  - declared literal `/score_Q/note` now reads "fixed scoring-only Q; never passed to the fitting, EIG or proposal paths."
  - declared literal `/truth_law/note` now reads "numeric parameters of WORLD_CFG[world] for the currently selected WORLD -- previously only 'world' + 'dp_commi"
  - sha256 of rewritten files or packets (15): `/arm_config/structure_proposals/fb1_grammar_sha256`, `/code_sha256/fb1_core.py`, `/code_sha256/fb1_design_search.py`, `/code_sha256/fb1_forward.py`, `/code_sha256/fb1_proposer.py`, `/code_sha256/run_gravity.py`, `/code_sha256/world_constants.py`, `/label/policy_source_sha256`, `/proposer/constraints/grammar/fb1_grammar_sha256`, `/proposer/events[]/calls[]/packet_sha256`, `/proposer/initial_burst_events[]/calls[]/packet_sha256`, `/proposer/initial_burst_events[]/packet_sha256` ...
  - packet size and token counts that follow (7): `/proposer/events[]/calls[]/packet_bytes`, `/proposer/events[]/calls[]/responses[]/usage/input_tokens`, `/proposer/events[]/calls[]/usage_total/input_tokens`, `/proposer/initial_burst_events[]/calls[]/packet_bytes`, `/proposer/initial_burst_events[]/calls[]/responses[]/usage/input_tokens`, `/proposer/initial_burst_events[]/calls[]/usage_total/input_tokens`, `/proposer/initial_burst_events[]/packet_bytes`
  - path of the run folder or the tree (2): `/arm_config/structure_proposals/vendor_source`, `/proposer/constraints/grammar/vendor_source`
- `FB1_RUN_gravity_111_seed31_v3_free_packets/t02_burst_w0.json`:
  - declared literal `/constraints/grammar/duplicate_detection/stage_0_finiteness_first` now reads "The finiteness probe runs BEFORE any equivalence comparison. If the candidate has even one non-finite probe po"
  - declared literal `/constraints/grammar/duplicate_detection/stage_1_canonical` now reads "canonical_model_id over the canonical AST + coefficient roles (shared with CORE, left untouched); commutative "
  - declared literal `/constraints/grammar/duplicate_detection/stage_2_numerical` now reads "ForceBench only: before admission the candidate force is compared with every registered free structure on the "
  - declared literal `/constraints/grammar/duplicate_detection/stage_2_verdict` now reads "A match is worded as indistinguishable_on_the_probe_grid. the probe grid is finite and pre-declared: a match m"
  - declared literal `/constraints/grammar/probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/constraints/grammar/probe/rule` now reads "X1 grid x ALL measurement times x prior draws x (X2,X3) points = the 4 declared box CORNERS plus a pre-declare"
  - declared literal `/constraints/grammar/vendor_chain` now reads "bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored)"
  - sha256 of rewritten files or packets (1): `/constraints/grammar/fb1_grammar_sha256`
  - path of the run folder or the tree (1): `/constraints/grammar/vendor_source`
- 3 files such as `FB1_RUN_gravity_111_seed31_v3_free_packets/t02_burst_w1.json`:
  - declared literal `/archive[]/prior_probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/archive[]/prior_probe/rule` now reads "X1 grid x ALL measurement times x prior draws x (X2,X3) points = the 4 declared box CORNERS plus a pre-declare"
  - declared literal `/constraints/grammar/duplicate_detection/stage_0_finiteness_first` now reads "The finiteness probe runs BEFORE any equivalence comparison. If the candidate has even one non-finite probe po"
  - declared literal `/constraints/grammar/duplicate_detection/stage_1_canonical` now reads "canonical_model_id over the canonical AST + coefficient roles (shared with CORE, left untouched); commutative "
  - declared literal `/constraints/grammar/duplicate_detection/stage_2_numerical` now reads "ForceBench only: before admission the candidate force is compared with every registered free structure on the "
  - declared literal `/constraints/grammar/duplicate_detection/stage_2_verdict` now reads "A match is worded as indistinguishable_on_the_probe_grid. the probe grid is finite and pre-declared: a match m"
  - declared literal `/constraints/grammar/probe/interior_basis` now reads "the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without"
  - declared literal `/constraints/grammar/probe/rule` now reads "X1 grid x ALL measurement times x prior draws x (X2,X3) points = the 4 declared box CORNERS plus a pre-declare"
  - declared literal `/constraints/grammar/vendor_chain` now reads "bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored)"
  - sha256 of rewritten files or packets (1): `/constraints/grammar/fb1_grammar_sha256`
  - path of the run folder or the tree (1): `/constraints/grammar/vendor_source`
- 16 files such as `FB1_RUN_gravity_111_seed31_v3_free_proposer/event_00_round_01_c0.request.json`:
  - packet text that carries the literals (1): `/bodies[]/messages[]/content`
  - sha256 of rewritten files or packets (1): `/identity/packet_sha256`
  - packet size and token counts that follow (1): `/identity/packet_bytes`
- 16 files such as `FB1_RUN_gravity_111_seed31_v3_free_proposer/event_00_round_01_c0.response.json`:
  - sha256 of rewritten files or packets (1): `/packet_sha256`
  - packet size and token counts that follow (3): `/packet_bytes`, `/responses[]/usage/input_tokens`, `/usage_total/input_tokens`

## The hashes a run records

Every BoxingGym, ChemBench and NeuronBench cell writes a `config.json` with the sha256 of each engine source file it loaded (`source_files_sha256`, `source_sha256`, `experimental_source_sha256`) and of the prompt file it sent (`prompt_sha256`); a ForceBench cell records its prompt file and frozen inputs. **Those hashes are of the development tree's files**: the same code with its original comments. Because this tree's comments and docstrings were rewritten, its files hash differently, so a recorded `source_files_sha256` will not match a file here byte for byte. What does match is the syntax tree (above) and the prompt files, which were not edited:

- boxinggym: prompt sha256 `196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c` recorded by 84 cells - identical in this tree (`tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt`)
- chembench: prompt sha256 `196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c` recorded by 108 cells - identical in this tree (`tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt`)
- chembench_breadth: prompt sha256 `196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c` recorded by 50 cells - identical in this tree (`tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt`)
- forcebench: prompt sha256 `4dac3b091359f4297dadfebe7e7525e54a718069fb48c5c9f5f70105e6499ab9` recorded by 58 cells - identical in this tree (`tools/fb1_launch_c/proposer_system_v2_9_FA-C_trig.txt`)

Note on BoxingGym: 9 of the 84 canonical cells (the same-seed guard reruns) ran on a development commit four commits after the tag (`166573f`, which adds the opt-in numeric guard `--numeric-guard v1`, default off); their recorded hashes of `belief_packet.py`, `boxinggym_core.py`, `count_inference.py` and `positive_inference.py` belong to that commit, not to the tag. The other 75 cells match the tag exactly.

NeuronBench cells record the sha256 of the composed prompt text rather than of a prompt file. The ForceBench kit files are listed with their development-tree hashes in `tools/fb1_snapshot/SHA256SUMS` of the frozen tag; this tree's `SHA256SUMS` lists the public files.

Two frozen inputs of the ForceBench kit, `tools/fb1_snapshot/frozen_inputs/FB1_FIXED_IC_MANIFEST_v5.json` and `tools/fb1_snapshot/frozen_inputs/FB1_FORCE_CURVE_yukawa_amplitude.json`, carried development notes (`v4_to_v5_reason`, `cap_rule`, `accept`, `disclosure`; `command`, `note`) in a language other than English and a development-host command line. Those strings were rewritten in English and every other field is byte for byte the recorded one (checked with `jq 'del(<those fields>)'` on both files). The paper's ForceBench cells record the development files' digests (`63067f1346ccfced16db0ba0793db4b3e7f6d0f266e44b612632afffc7de9ce9` and `4253c197ee68575aca7e348b924c2bf6557e51b908820f297b8a0dde9adcee19`) under `frozen_inputs_sha256`; this tree's `tools/fb1_snapshot/frozen_inputs/SHA256SUMS` lists the public files'. The kit reads these files for their numeric content only; they are not part of any LLM request.

## Checking a run against this tree

`USAGE.md` ("Reproducing the benchmark runs") shows how to compare a cell's recorded digests with a checkout; against this tree, compare syntax trees instead of bytes: `python tools/release/ast_equiv.py <development checkout> . src/bayes_model_discovery/experimental/<file>.py`.
