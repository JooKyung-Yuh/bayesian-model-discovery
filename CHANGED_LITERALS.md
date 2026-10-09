# Declared syntax-tree changes of the public tree against the verified development tree

Every other difference between the trees is in comments, docstrings or formatting (see `tools/release/ast_equiv.py`). The changes below are the only ones that alter a Python syntax tree: string literals rewritten from Korean to English or with an internal code name spelled out (exception messages, assertion messages, record labels written into run folders, test skip reasons and match patterns), plus redundant `pass` statements removed after their function's docstring was rewritten. No condition, computation, default value or control flow changed.

Abbreviations in the quoted literals: EIG = expected information gain, EPIG = expected predictive information gain, BMA = Bayesian model averaging, SMC = sequential Monte Carlo, KL = Kullback-Leibler divergence, nMSE = normalized mean squared error, D0 = the initial design set.

## Changed string literals (124)

| file | line | the literal now reads |
| --- | ---: | --- |
| `src/bayes_model_discovery/experimental/belief_packet.py` | 555 | link is fixed by the outcome type (the paper's task table) |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 530 | boxing_gym.envs.{moral_machines,emotion} construct openai.OpenAI() in __init__ (moral_machines.py:294, emotion.py:270) and call chat.completions.create inside step (moral_machines.py:529, emotion.py:4 |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 1886 | specification_only: the environment integrates a DETERMINISTIC Lotka-Volterra ODE (lotka_volterra.py:293-:306, scipy.integrate.odeint) and returns the two rounded, clipped populations at a queried tim |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 2338 | DesignPool.available (:706-720) raises ValueError once every design in the pool has already been observed. Earlier versions let that propagate to run()'s outer exception handler and recorded it as a c |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 2806 | f'--truth-in-pool/--inject-oracle-at are declared for the death oracle only; task {spec.name!r} has no registered oracle expression' |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 2846 | the proposed-prior ablation is defined on the count path; scalar/real noise priors do not follow the template |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 3255 | restored question |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 3257 | the numeric goal.eval_points path is used, exactly as the PPL path does, so the upstream get_goal_eval_question text defect cannot affect this run; no question string is read anywhere in this module |
| `src/bayes_model_discovery/experimental/boxinggym_core.py` | 5455 | restored question |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 796 | PYTHONPATH only, never installed (a project decision) |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 1595 | f'per-candidate particle mean/scale would be {projected_bytes} bytes > {max_bytes} (the recompute cap); storing the mixture log-density at the KL quadrature nodes instead' |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2516 | the proposed-prior ablation is defined on the count path; scalar/real noise priors do not follow the template |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2864 | fixed candidate set: one pre-declared `sample_designs` grid, fixed for every task/seed/cell, checked disjoint from the official evaluation set at run time; an A=0 cell records it and reads none of it |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2872 | fixed profile table; `design_pool`/`eig_samples` are pre-declared for the stage-2 acquisition axis and unused in stage 1 |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2902 | the engine's truth-hook design |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2929 | the comparison method's evaluation code (its ChemBench run manifest, eval_points.rule) |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 2943 | the comparison method's evaluation code (its ChemBench run manifest, metric_defs.nmse_mda) |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 4624 | mean_Q[ KL( Normal(noiseless_observation(x_q), (noise*noiseless_observation(x_q))**2) truncated at 0 \|\| BMA ) ], 1-D Gauss-Legendre quadrature; defined for EVERY checkpoint row (kl_every_checkpoint) |
| `src/bayes_model_discovery/experimental/chembench_core.py` | 4635 | mean_Q[ KL( Normal(noiseless_observation(x_q), (noise*noiseless_observation(x_q))**2) truncated at 0 \|\| BMA ) ], 1-D Gauss-Legendre quadrature; defined ONLY for the FINAL checkpoint row (the officia |
| `src/bayes_model_discovery/experimental/core_count.py` | 1111 | the fixed-pool cell (no structure and no family expansion) forbids proposals |
| `src/bayes_model_discovery/experimental/core_count.py` | 2873 | fixed_pool_no_proposal |
| `src/bayes_model_discovery/experimental/count_models.py` | 865 | an indexed coefficient a[<categorical input>] is one length-n_levels vector coefficient: its per-level offsets share the single fixed role prior signed=Normal(0,1) (the paper's prior table 'standardis |
| `src/bayes_model_discovery/experimental/count_proposer.py` | 2097 | invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls |
| `src/bayes_model_discovery/experimental/count_proposer.py` | 2326 | invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls |
| `src/bayes_model_discovery/experimental/count_proposer.py` | 2694 | invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls |
| `src/bayes_model_discovery/experimental/count_proposer.py` | 2928 | invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls |
| `src/bayes_model_discovery/experimental/design_search.py` | 160 | f'design-search pool is refused for {design_policy} at input dimension {input_dimension} > 2 (the acquisition policy forbids pool EIG/EPIG at d >= 3 for every run); {remedy}' |
| `src/bayes_model_discovery/experimental/test_allowed_families_round_call.py` | 104 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_boxinggym_safeguards_parity.py` | 55 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 136 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 146 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 162 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_chembench_core.py` | 1834 | 9097d8cdb6f875f5bf42463e2acca5da6d99724197ec4bee12aa22b800f15459 |
| `src/bayes_model_discovery/experimental/test_chembench_prewave_safeguards.py` | 379 | parity with the baseline recorded before the model-identity change crosses the identity boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_core_candidate_exclusion.py` | 144 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_core_parallel_wave.py` | 325 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_core_parallel_wave.py` | 329 | set BMD_BASELINE_COMPARE=1 and BMD_PWC_BASELINE_DIR for the baseline parity check |
| `src/bayes_model_discovery/experimental/test_count_pairing.py` | 67 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_count_proposer.py` | 113 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_count_proposer.py` | 753 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_design_pool_refusal.py` | 35 | f'design-search pool is refused for {policy} at input dimension {dimension} > 2 (the acquisition policy forbids pool EIG/EPIG at d >= 3 for every run); {REMEDY[policy]}' |
| `src/bayes_model_discovery/experimental/test_epig_positivity_guard.py` | 292 | parity with the baseline recorded before the model-identity change crosses the model-ID boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_epig_positivity_guard.py` | 297 | BMD_EPIG_GUARD_BASELINE_DIR must identify the pristine baseline source |
| `src/bayes_model_discovery/experimental/test_epig_runners.py` | 196 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_epig_runners.py` | 201 | BMD_EPIG_BASELINE_DIR must name a read-only checkout of the baseline commit |
| `src/bayes_model_discovery/experimental/test_named_inputs.py` | 1013 | the client must never be called for the fixed-pool cell |
| `src/bayes_model_discovery/experimental/test_model_id_commutativity.py` | 161 | b'synthetic identity-change control' |
| `src/bayes_model_discovery/experimental/test_prewave_fit_timeout.py` | 375 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_prior_switches_boxinggym.py` | 149 | the proposed-prior ablation is defined on the count path; scalar/real noise priors do not follow the template |
| `src/bayes_model_discovery/experimental/test_space_filling_core_boxing.py` | 246 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_space_filling_core_boxing.py` | 256 | set BMD_BASELINE_COMPARE=1 for the design-study baseline artifact parity |
| `src/bayes_model_discovery/experimental/test_space_filling_core_boxing.py` | 259 | set BMD_DESIGN_STUDY_BASELINE to a read-only checkout of the design-study baseline |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 467 | require_autoscilab_source |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 731 | need a structure-expansion cell |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 733 | require_autoscilab_source |
| `src/bayes_model_discovery/experimental/test_wave_accounting.py` | 317 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_core_d0_policy.py` | 225 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_core_d0_policy.py` | 229 | set BMD_BASELINE_COMPARE=1 and BMD_D0S_BASELINE_DIR for the baseline parity check |
| `src/bayes_model_discovery/experimental/test_gaussian_core.py` | 384 | parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 391 | f"chunked log-predictive, one input, 1000 particles, 64 draws: {differing} of 64 one-column reductions differ from the whole array's; two-column blocks: 0 differ" |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 394 | this numpy reduces a one-column block like the whole array: the one-column rule is inert |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 608 | f'chunked log-predictive, ChemBench guarded path: excluded={excluded} width={width} calls={new_calls} prediction_pools={new_pools} conditional_loglik_blocks={blocks} old==new bytes' |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 738 | set BMD_BASELINE_COMPARE=1 and BMD_TRM_BASELINE_DIR (a clean checkout of the parity baseline commit) |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 885 | chunked log-predictive, CORE parity: |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 917 | chunked log-predictive, ChemBench parity: |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 950 | chunked log-predictive, BoxingGym parity: |
| `tools/fb1_snapshot/fb1_grammar.py` | 50 | bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored) |
| `tools/fb1_snapshot/fb1_grammar.py` | 51 | count_models.py of the canonical engine chain v2.9, taken from the private development tree; the vendored copy is identified by VENDOR_SHA256 |
| `tools/fb1_snapshot/fb1_grammar.py` | 226 | cos/sin are not in the vendored CALL_ARITY (exp·log·sqrt·tanh·square·power) |
| `tools/fb1_snapshot/fb1_grammar.py` | 235 | catalogue kinds = ('pos', 'pos_bg'); pos_bg is LogNormal(log b0, 1), b0 = 1/(2 pi R^2) (fb1_core.B0_BACKGROUND, derived from the box half-width R = 10). The free-form transcription makes both constant |
| `tools/fb1_snapshot/fb1_grammar.py` | 319 | f'free-form force law {expression!r} (free constants {names})' |
| `tools/fb1_snapshot/fb1_grammar.py` | 350 | f'the catalogue transcription is not one-to-one: {name} and {table[mid]}' |
| `tools/fb1_snapshot/fb1_grammar.py` | 530 | The finiteness probe runs BEFORE any equivalence comparison. If the candidate has even one non-finite probe point it is rejected as  |
| `tools/fb1_snapshot/fb1_grammar.py` | 537 | canonical_model_id over the canonical AST + coefficient roles (shared with CORE, left untouched); commutative re-orderings are NOT normalised, so p1*c/r**2 and c*p1/r**2 get different canonical ids |
| `tools/fb1_snapshot/fb1_grammar.py` | 541 | f'ForceBench only: before admission the candidate force is compared with every registered free structure and the catalogue transcriptions on the probe grid, trying EVERY PRIOR-PRESERVING coefficient p |
| `tools/fb1_snapshot/fb1_grammar.py` | 553 | A match is worded as  |
| `tools/fb1_snapshot/fb1_grammar.py` | 555 | Matching a catalogue law is a LINK to that catalogue name, not a rejection. Only an already registered FREE structure yields  |
| `tools/fb1_snapshot/fb1_grammar.py` | 598 | r grid x ALL measurement times x prior draws x (p1,p2) points = the 4 declared box CORNERS plus a pre-declared INTERIOR grid (the box quartiles, 3x3); any non-finite value rejects the candidate, and s |
| `tools/fb1_snapshot/fb1_grammar.py` | 602 | the corners alone cannot separate a law whose extra term vanishes only at the p-endpoints from the law without it (counterexample c*p1+c*(p2-0.1)*(p2-10) versus c*p1); the runner's design box contains |
| `tools/fb1_snapshot/fb1_grammar.py` | 650 | probe grid needs at least one measurement time |
| `tools/fb1_snapshot/fb1_grammar.py` | 948 | coefficients are paired by trying EVERY PRIOR-PRESERVING permutation (at most 6, since a free law has at most 3 coefficients): the paired coefficients must carry the same prior one by one, so the same |
| `tools/fb1_snapshot/fb1_grammar.py` | 951 | , never a proof of general functional equivalence. The same law written with a DIFFERENT coefficient count (e.g. a redundant constant that could be absorbed into an existing one) is still NOT detected |
| `tools/fb1_snapshot/fb1_grammar.py` | 1044 | f"{label}: {head} -- but the catalogue structure's priors are {list(STRUCTURES[label].kinds)} while this law's are {list(my_kinds)}, so they are DIFFERENT Bayesian models and are never merged" |
| `tools/fb1_snapshot/fb1_grammar.py` | 1136 | different priors are never merged: this law is registered as its own free structure and the scale difference is recorded |
| `tools/fb1_snapshot/fb1_proposer.py` | 1148 | structure expansion is declared but the catalog-selection contract only accepts expressions listed verbatim in constraints.initial_structures |
| `tools/fb1_snapshot/run_gravity.py` | 44 | DP_SCIENCEAGENT_DIR must name the DiscoverPhysics/ScienceAgent directory (directory containing the scienceagent package); set it before running |
| `tools/fb1_snapshot/run_gravity.py` | 185 | MDA, Murphy (2026), arXiv:2608.09696v4, p. 29, Tables 4 and 6: the seed launch is action 3, one passive drop at r0=3 with zero velocity and p1=p2=1. Under the amendment of 2026-09-13, this engine vari |
| `tools/fb1_snapshot/run_gravity.py` | 259 | v1 (background term) |
| `tools/fb1_snapshot/run_gravity.py` | 259 | v2 (background term + time modulation) - abandoned |
| `tools/fb1_snapshot/run_gravity.py` | 260 | v3 final (background term + time modulation + product coupling + crossover) |
| `tools/fb1_snapshot/run_gravity.py` | 320 | duplicate |
| `tools/fb1_snapshot/run_gravity.py` | 320 | complexity limit exceeded |
| `tools/fb1_snapshot/run_gravity.py` | 807 | every candidate has the same particle count (500 x 2 chains = 1000) |
| `tools/fb1_snapshot/run_gravity.py` | 1077 | E[log p_k(y\|d) - log p(y\|d)], k = the outer draw's own model, p_k = model k's predictive averaged over its inner particles (MDA v4 Eq. 26; named structure_eig in the BoxingGym and NeuronBench EIG-de |
| `tools/fb1_snapshot/run_gravity.py` | 2146 | the same score_set, held-out Q and truth as the BMA entry; a scoring-only key (the main random stream is untouched). |
| `tools/fb1_snapshot/run_gravity.py` | 2158 | ForceBench outcomes are continuous - a binary Brier score is undefined |
| `tools/fb1_snapshot/run_gravity.py` | 2505 | domain_size=20.0, periodic=True -> R=10, b0=1/(2 pi R^2) (digits derived from the metadata) |
| `tools/fb1_snapshot/run_gravity.py` | 2533 | fixed scoring-only Q; never passed to the fitting, EIG or proposal paths. |
| `tools/fb1_snapshot/run_gravity.py` | 2537 | the noisy observed trajectories returned by world.observe(); order = designs. |
| `tools/fb1_snapshot/run_gravity.py` | 2547 | numeric parameters of WORLD_CFG[world] for the currently selected WORLD -- previously only 'world' + 'dp_commit' identified the true force law in the run JSON |
| `tools/fb1_snapshot/run_gravity.py` | 2597 | candidates are free-form expressions over the named inputs {r, p1, p2, t}, validated by the shared CORE grammar and registered into fb1_core.STRUCTURES at runtime (constraints.structure_expansion=true |
| `tools/fb1_snapshot/run_gravity.py` | 2603 | candidates are chosen verbatim from the named STRUCTURES catalog (constraints.structure_expansion=false); a recovered world law is a SELECTION, not a discovery |
| `tools/fb1_snapshot/run_gravity.py` | 2690 | before the EIG computation, exclude candidates with min_t r(t) < 2*dx (judged on the current BMA predicted trajectory) |
| `tools/fb1_snapshot/run_gravity.py` | 2691 | default arguments of scienceagent/executor.py SimulationExecutor grid_size=(64, 64), domain_size=20.0 -> dx = 20/64 = 0.3125 |
| `tools/fb1_snapshot/run_gravity.py` | 2693 | field-engine worlds, active arms only |
| `tools/fb1_snapshot/run_gravity.py` | 2697 | steps[*].chosen_true_min_r is recorded by the evaluator after the choice, for the record only. |
| `tools/fb1_snapshot/run_gravity.py` | 2701 | unified protocol (registered 2026-10-06) |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 64 | the premise that the catalogue has 14 structures no longer holds |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 80 | f'[check 1] p={p} expected={expected} 6SE={6 * se}' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 97 | f'[check 2] max\|diff\|={err}' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 126 | f'a non-sig_hn column differs from the pre-correction jax.random.normal draw: kinds={kinds}' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 140 | f'the sig_hn column differs from log(max(\|z\|, tiny)) by more than 1e-12: kinds={kinds} col={j} max\|diff\|={diff}' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 144 | f"[check 3] checked {checked} (key, kinds) combinations across {len(keys)} keys; kind_tuples includes all {len(_REAL_KINDS)} distinct kind-shapes that occur among the 14 catalogue structures (several  |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 187 | f'{name}: a structure column changed with the sampler correction (it must not)' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 191 | f'[check 4] structural columns identical for {structural_identical}/14 structures; sig_hn column differs (pre-fix vs current) for {sig_hn_differs}/14 (0/14 means the tree under test is still pristine/ |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 215 | f'structures whose prior-probe verdict changed: {mismatches}' |
| `tools/fb1_snapshot/test_fb1_core_prior.py` | 216 | f'[check 4] prior-probe verdicts (finite, nonfinite_fraction) identical before/after for all {len(before)} catalogue structures' |
| `tools/fb1_snapshot/test_fb1_mock_family.py` | 102 | duplicate |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1193 | the masking filter was never applied |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1769 | the student_t4 component is being replaced by a normal one |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1899 | f'law_str of {world} is empty' |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1901 | f'{world} has no {key}' |
| `tools/fb1_snapshot/test_fb1_reasoning_and_paid_call_guard.py` | 372 | 7025bb0ac102f9c56c5ec4db99dcf32ec722595c072e89842eb780ed2647ef4c |
| `tools/fb1_snapshot/world_constants.py` | 133 | r >= 1 (the grid-near-particle points at 0.5/0.75 are dropped; differs from the yukawa grid of FB1_FORCE_CURVE_WORLD_RULING.json); single-parameter fit of the amplitude A, no linear background term (B |

## Files with a structural difference (13)

| file | first differing statement (base line) | statements base → public | top-level definitions removed | added |
| --- | --- | --- | --- | --- |
| `src/bayes_model_discovery/experimental/test_belief_packet.py` | Import  at line 5 | 38 → 30 | `#6`, `#7`, `REV2_NOTE_BEFORE`, `REV2_NOTE_AFTER`, `REV2_GOAL_BEFORE`, `REV2_GOAL_AFTER`, `DECISION_REV2`, `test_prompt_file_is_rev2_with_exactly_the_two_documented_edits` | - |
| `src/bayes_model_discovery/experimental/test_boxinggym_mda_options.py` | Import  at line 4 | 32 → 27 | `#7`, `#8`, `#9`, `#5`, `test_default_bytes_match_baseline_except_new_keys_and_verified_source_hash` | - |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | FunctionDef test_peregrines_seed_37_random2_starts_at_two_zero_counts_and_is_refused_by_default at line 124 | 24 → 24 | - | - |
| `src/bayes_model_discovery/experimental/test_chembench_mda_options.py` | Import  at line 2 | 20 → 14 | `#5`, `#6`, `#7`, `#4`, `HARNESS`, `test_default_bytes_match_base_except_new_keys_and_verified_source_hash` | - |
| `src/bayes_model_discovery/experimental/test_count_nan_robustness.py` | FunctionDef test_the_bai_failure_mode_reproduces_and_stays_catchable at line 162 | 61 → 61 | `test_the_bai_failure_mode_reproduces_and_stays_catchable` | `test_the_observed_failure_mode_reproduces_and_stays_catchable` |
| `src/bayes_model_discovery/experimental/test_epig_acquisition_efficiency.py` | FunctionDef _install_synthetic_box_world_37 at line 900 | 67 → 67 | - | - |
| `src/bayes_model_discovery/experimental/test_epig_positivity_guard.py` | FunctionDef test_all_valid_acquisition_json_and_score_bytes_match_pristine_base at line 271 | 42 → 42 | - | - |
| `src/bayes_model_discovery/experimental/test_grammar_version_binding.py` | FunctionDef test_the_generator_registers_the_grammar_only_when_it_is_not_the_default at line 252 | 34 → 29 | `test_the_generator_registers_the_grammar_only_when_it_is_not_the_default`, `test_the_generator_emits_the_driver_option_only_when_it_is_not_the_default`, `test_the_driver_passes_the_option_through_and_never_in_a_shakedown`, `_repository_root`, `_generator` | - |
| `src/bayes_model_discovery/experimental/test_model_id_power_normal_form.py` | FunctionDef test_fugu_equal_spellings_are_one_model at line 74 | 59 → 59 | `test_fugu_equal_spellings_are_one_model`, `test_fugu_different_forms_stay_apart` | `test_equal_spellings_are_one_model`, `test_different_forms_stay_apart` |
| `src/bayes_model_discovery/experimental/test_packet_constraints_explicit.py` | Import  at line 18 | 58 → 50 | `#7`, `#8`, `#9`, `#10`, `REPO_ROOT`, `DRIVER`, `driver_dry_run`, `test_the_driver_passes_an_explicit_v2_0_to_every_bmd_runner` | - |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | FunctionDef test_an_eig_part_needs_an_acquisition_bit_and_the_ascent at line 440 | 88 → 88 | - | - |
| `tools/fb1_snapshot/test_fb1_label_rule.py` | Import  at line 3 | 36 → 29 | `#10`, `#2`, `test_packager_keeps_production_only_from_bound_receipt`, `test_packager_rejects_unbound_or_changed_receipts_before_output`, `test_historical_unbound_production_string_stays_development`, `test_explicit_development_mock_receipt_reaches_packaged_provenance`, `test_packager_reopens_manifest_before_returning_success` | - |
| `tools/fb1_snapshot/test_fb1_proposer.py` | FunctionDef test_21g2_real_sdk_mocktransport_debug_logs_and_traceback_are_masked at line 1097 | 133 → 131 | `_minimal_run_json_29fb`, `test_29fb_5_emitter_points_csv_kl_column_gets_a_value_other_columns_unchanged` | - |
