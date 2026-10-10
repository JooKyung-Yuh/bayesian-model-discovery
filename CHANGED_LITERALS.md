# Declared syntax-tree changes of the public tree against the verified development tree

Every other difference between the trees is in comments, docstrings or formatting (see `tools/release/ast_equiv.py`). The changes below are the only ones that alter a Python syntax tree: string literals rewritten from non-English text to English or with an internal code name spelled out (exception messages, assertion messages, record labels written into run folders, test skip reasons and match patterns), plus redundant `pass` statements removed after their function's docstring was rewritten. No condition, computation, default value or control flow changed.

Abbreviations in the quoted literals: EIG = expected information gain, EPIG = expected predictive information gain, BMA = Bayesian model averaging, SMC = sequential Monte Carlo, KL = Kullback-Leibler divergence, nMSE = normalized mean squared error, D0 = the initial design set.

## Changed string literals (113)

| file | line | the literal now reads |
| --- | ---: | --- |
| `conftest.py` | 13 | b43e38cb03d09c13efa9cf4d9bae740d51157bfd |
| `scripts/bmd_run.py` | 56 | the src/ folder of the BoxingGym checkout @ b43e38cb |
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
| `src/bayes_model_discovery/experimental/test_boxinggym_input_name_masking.py` | 182 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_boxinggym_input_name_masking.py` | 218 | (removed) |
| `src/bayes_model_discovery/experimental/test_boxinggym_input_name_masking.py` | 271 | (removed) |
| `src/bayes_model_discovery/experimental/test_boxinggym_input_name_masking.py` | 273 | (removed) |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 136 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 146 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | 162 | require_boxinggym_source |
| `src/bayes_model_discovery/experimental/test_chembench_core.py` | 1834 | 9097d8cdb6f875f5bf42463e2acca5da6d99724197ec4bee12aa22b800f15459 |
| `src/bayes_model_discovery/experimental/test_count_pairing.py` | 67 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_count_proposer.py` | 113 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_count_proposer.py` | 753 | forbids proposal |
| `src/bayes_model_discovery/experimental/test_design_pool_refusal.py` | 35 | f'design-search pool is refused for {policy} at input dimension {dimension} > 2 (the acquisition policy forbids pool EIG/EPIG at d >= 3 for every run); {REMEDY[policy]}' |
| `src/bayes_model_discovery/experimental/test_model_id_commutativity.py` | 161 | b'synthetic identity-change control' |
| `src/bayes_model_discovery/experimental/test_named_inputs.py` | 1013 | the client must never be called for the fixed-pool cell |
| `src/bayes_model_discovery/experimental/test_prior_switches_boxinggym.py` | 149 | the proposed-prior ablation is defined on the count path; scalar/real noise priors do not follow the template |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 467 | require_autoscilab_source |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 731 | need a structure-expansion cell |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | 733 | require_autoscilab_source |
| `src/bayes_model_discovery/experimental/test_neuron_core_cli.py` | 1154 | the batch backend reproduces the native one bit for bit only with numpy's SIMD power dispatch disabled, the precondition the runner itself checks: set NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512 |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 108 | f"chunked log-predictive, one input, 1000 particles, 64 draws: {differing} of 64 one-column reductions differ from the whole array's; two-column blocks: 0 differ" |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | 111 | this numpy reduces a one-column block like the whole array: the one-column rule is inert |
| `tools/fb1_snapshot/test_fb1_reasoning_and_paid_call_guard.py` | 372 | 7025bb0ac102f9c56c5ec4db99dcf32ec722595c072e89842eb780ed2647ef4c |
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
| `tools/fb1_snapshot/test_fb1_grammar_trig.py` | 375 | (removed) |
| `tools/fb1_snapshot/test_fb1_mock_family.py` | 102 | duplicate |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1193 | the masking filter was never applied |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1769 | the student_t4 component is being replaced by a normal one |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1899 | f'law_str of {world} is empty' |
| `tools/fb1_snapshot/test_fb1_proposer.py` | 1901 | f'{world} has no {key}' |
| `tools/fb1_snapshot/world_constants.py` | 133 | r >= 1 (the grid-near-particle points at 0.5/0.75 are dropped; differs from the yukawa grid of FB1_FORCE_CURVE_WORLD_RULING.json); single-parameter fit of the amplitude A, no linear background term (B |
| `tools/fb1_snapshot/test_fb1_epig_targets.py` | 171 | (removed) |
| `tools/fb1_snapshot/test_fb1_epig_targets.py` | 172 | (removed) |
| `tools/fb1_snapshot/test_fb1_unified_protocol.py` | 890 | (removed) |
| `tools/fb1_snapshot/test_fb1_unified_protocol.py` | 891 | (removed) |

## Files with a structural difference (38)

| file | first differing statement (base line) | statements base → public | top-level definitions removed | added |
| --- | --- | --- | --- | --- |
| `src/bayes_model_discovery/experimental/test_allowed_families_round_call.py` | Import  at line 4 | 23 → 13 | `#4`, `#5`, `#6`, `#7`, `#8`, `BASE_COMMIT`, `IDENTITY_FIELDS`, `CHANGED_SOURCE_FILES`, `TRANSPORT_MEASUREMENT_FIELDS`, `test_opt_in_baseline_packets_and_count_bytes` | - |
| `src/bayes_model_discovery/experimental/test_belief_packet.py` | Import  at line 5 | 38 → 30 | `#6`, `#7`, `REV2_NOTE_BEFORE`, `REV2_NOTE_AFTER`, `REV2_GOAL_BEFORE`, `REV2_GOAL_AFTER`, `DECISION_REV2`, `test_prompt_file_is_rev2_with_exactly_the_two_documented_edits` | - |
| `src/bayes_model_discovery/experimental/test_boxinggym_input_name_masking.py` | Import  at line 5 | 42 → 32 | `#8`, `#9`, `exercise_fake_cell`, `TIME_KEYS`, `PACKET_IDENTITY_KEYS`, `SOURCE_CHANGES`, `compare_artifacts`, `first_difference`, `verify_packet_identities`, `scan_requests`, `test_opt_in_full_cell_artifact_equivalence_and_leakage` | `FA_C_PROMPT` |
| `src/bayes_model_discovery/experimental/test_boxinggym_mda_options.py` | Import  at line 4 | 32 → 24 | `#6`, `#7`, `_external_src`, `#8`, `#9`, `#5`, `BASELINE_HARNESS`, `test_default_bytes_match_baseline_except_new_keys_and_verified_source_hash` | - |
| `src/bayes_model_discovery/experimental/test_boxinggym_zero_start.py` | FunctionDef test_peregrines_seed_37_random2_starts_at_two_zero_counts_and_is_refused_by_default at line 124 | 24 → 24 | - | - |
| `src/bayes_model_discovery/experimental/test_burst_refill_waves.py` | Import  at line 11 | 72 → 57 | `#4`, `#5`, `#6`, `#11`, `BASE_COMMIT`, `PCX1_PACKET_CONSTRAINT_KEYS`, `PCX1_PACKET_DIGEST_KEYS`, `PCX2_EVENT_KEYS`, `PCX2_ARCHIVE_KEYS`, `PCX2_ACCOUNTING_KEYS`, `request_text`, `pop_packet_pair`, `without_unit_keys`, `PARITY_HARNESS`, `test_later_round_parallel_waves_match_the_base_commit_byte_for_byte` | - |
| `src/bayes_model_discovery/experimental/test_chembench_frozen_pool.py` | Assign  at line 45 | 60 → 57 | `BASE_COMMIT`, `_base_module`, `test_core_boxinggym_and_default_chembench_settings_are_the_base_commits` | - |
| `src/bayes_model_discovery/experimental/test_chembench_initial_pool.py` | Import  at line 5 | 48 → 32 | `#7`, `#8`, `#9`, `#7`, `BASE_COMMIT`, `PCX1_PACKET_CONSTRAINT_KEYS`, `PCX1_PACKET_DIGEST_KEYS`, `PCX2_ACCOUNTING_KEYS`, `PCX2B_ROUND_EVENTS_KEY`, `STAGE_SLOT_CONFIG_KEY`, `CIC_PACKET_CONSTRAINT_KEYS`, `CHANGED_SOURCES`, `SCIENCE_FILES`, `BASELINE_HARNESS`, `test_opt_in_default_artifacts_match_hub_baseline`, `_drop_unit_record_keys` | - |
| `src/bayes_model_discovery/experimental/test_chembench_mda_options.py` | Import  at line 2 | 20 → 12 | `#4`, `#5`, `#6`, `#7`, `#4`, `_external_src`, `HARNESS`, `test_default_bytes_match_base_except_new_keys_and_verified_source_hash` | - |
| `src/bayes_model_discovery/experimental/test_chembench_prewave_safeguards.py` | ImportFrom  at line 17 | 35 → 32 | `#6`, `BASELINE_HARNESS`, `test_opt_in_default_artifacts_match_declared_base` | - |
| `src/bayes_model_discovery/experimental/test_core_candidate_exclusion.py` | Import  at line 3 | 24 → 17 | `#5`, `#6`, `#7`, `CHANGED_SOURCE_FILES`, `PWC2_SCORE_ADDED_KEYS`, `BASE_COMMIT`, `test_opt_in_omitted_policy_matches_base_bytes_and_authentic_identity` | - |
| `src/bayes_model_discovery/experimental/test_core_d0_policy.py` | Import  at line 6 | 40 → 33 | `#8`, `#9`, `D0S_ADDED_KEYS`, `IDENTITY_FIELDS`, `RECEIPT_IDENTITY_FIELDS`, `CHANGED_SOURCE_FILES`, `test_opt_in_fixed4_matches_the_gw_baseline_science_and_declared_metadata` | - |
| `src/bayes_model_discovery/experimental/test_core_parallel_wave.py` | Import  at line 3 | 39 → 30 | `#8`, `#9`, `#10`, `#11`, `IDENTITY_FIELDS`, `CHANGED_SOURCE_FILES`, `RECEIPT_IDENTITY_FIELDS`, `WAVE_KEYS`, `test_opt_in_no_wave_options_preserve_baseline_science_config_and_real_identities` | - |
| `src/bayes_model_discovery/experimental/test_count_nan_robustness.py` | FunctionDef test_the_bai_failure_mode_reproduces_and_stays_catchable at line 162 | 61 → 61 | `test_the_bai_failure_mode_reproduces_and_stays_catchable` | `test_the_observed_failure_mode_reproduces_and_stays_catchable` |
| `src/bayes_model_discovery/experimental/test_epig_acquisition_efficiency.py` | FunctionDef _install_synthetic_box_world_37 at line 900 | 67 → 67 | - | - |
| `src/bayes_model_discovery/experimental/test_epig_positivity_guard.py` | Import  at line 3 | 42 → 36 | `#4`, `#5`, `#6`, `#7`, `all_valid_parity_payload`, `test_all_valid_acquisition_json_and_score_bytes_match_pristine_base` | - |
| `src/bayes_model_discovery/experimental/test_epig_runners.py` | Import  at line 3 | 28 → 22 | `#6`, `#7`, `#8`, `#9`, `PARITY_HARNESS`, `test_opt_in_all_legacy_cells_preserve_artifact_bytes` | - |
| `src/bayes_model_discovery/experimental/test_gaussian_core.py` | Import  at line 5 | 40 → 36 | `#8`, `#9`, `test_opt_in_count_world_baseline_methods_and_values_are_unchanged`, `test_opt_in_legacy_count_runner_keeps_scientific_bytes` | - |
| `src/bayes_model_discovery/experimental/test_grammar_version_binding.py` | FunctionDef test_the_generator_registers_the_grammar_only_when_it_is_not_the_default at line 252 | 34 → 29 | `test_the_generator_registers_the_grammar_only_when_it_is_not_the_default`, `test_the_generator_emits_the_driver_option_only_when_it_is_not_the_default`, `test_the_driver_passes_the_option_through_and_never_in_a_shakedown`, `_repository_root`, `_generator` | - |
| `src/bayes_model_discovery/experimental/test_initial_pool_file.py` | Import  at line 5 | 50 → 32 | `#7`, `BASE_COMMIT`, `CHANGED_SOURCES`, `BRF_CONFIG_KEYS`, `BRF_POOL_FILE_KEYS`, `BRF_WAVE_KEYS`, `BRF_BURST_SETTINGS_KEYS`, `BRF_ACCOUNTING_KEYS`, `BRF_WAVE_PACKET`, `PCX2B_BURST_SETTINGS_KEYS`, `PCX1_PACKET_CONSTRAINT_KEYS`, `PCX1_PACKET_DIGEST_KEYS`, `PCX2_EVENT_KEYS`, `PCX2_ARCHIVE_KEYS`, `PCX2_ACCOUNTING_KEYS`, `ADMITTED_ID_KEYS`, `test_opt_in_catalogue_and_generation_match_hub_baseline`, `_drop_unit_record_keys` | - |
| `src/bayes_model_discovery/experimental/test_neuron_forward_batch.py` | FunctionDef test_predict_batch_bank_is_c_contiguous_and_means_like_the_native_bank at line 343 | 55 → 55 | - | - |
| `src/bayes_model_discovery/experimental/test_packet_constraints_explicit.py` | Import  at line 18 | 58 → 50 | `#7`, `#8`, `#9`, `#10`, `REPO_ROOT`, `DRIVER`, `driver_dry_run`, `test_the_driver_passes_an_explicit_v2_0_to_every_bmd_runner` | - |
| `src/bayes_model_discovery/experimental/test_paid_call_guard.py` | Import  at line 11 | 46 → 41 | `#7`, `#8`, `BASE_COMMIT`, `base_transport`, `test_the_omitted_option_writes_the_base_commits_transport_bytes` | - |
| `src/bayes_model_discovery/experimental/test_uni_chembench_protocol_u.py` | FunctionDef test_an_eig_part_needs_an_acquisition_bit_and_the_ascent at line 440 | 88 → 88 | - | - |
| `src/bayes_model_discovery/experimental/test_neuron_core_cli.py` | FunctionDef test_n1_the_two_backends_score_and_observe_the_same_rows at line 1095 | 108 → 108 | - | - |
| `src/bayes_model_discovery/experimental/test_model_id_power_normal_form.py` | Import  at line 12 | 59 → 44 | `#7`, `#8`, `#9`, `#8`, `BASE_COMMIT`, `REPO_ROOT`, `TRIPLE`, `test_fugu_equal_spellings_are_one_model`, `test_fugu_different_forms_stay_apart`, `v1_model_id`, `SIGNED3`, `POWER_FREE`, `UNTOUCHED_POWER`, `NORMALIZED`, `test_ids_without_normalized_power_notation_equal_the_pinned_v1_ids`, `test_normalized_spellings_move_off_their_v1_ids`, `test_v1_kept_the_spellings_apart_that_v2_merges` | `test_equal_spellings_are_one_model`, `test_different_forms_stay_apart` |
| `src/bayes_model_discovery/experimental/test_prewave_fit_timeout.py` | ImportFrom  at line 12 | 40 → 37 | `#7`, `DEFAULT_BASELINE_HARNESS`, `test_opt_in_default_bytes_match_verified_base` | - |
| `src/bayes_model_discovery/experimental/test_repair_two_parity.py` | Import  at line 3 | 30 → 1 | `#1`, `#2`, `#3`, `#4`, `#5`, `#6`, `#7`, `#1`, `#8`, `#9`, `#2`, `BASE_COMMIT`, `IDENTITY_FIELDS`, `CHANGED_SOURCES`, `PCX1_PACKET_CONSTRAINT_KEYS`, `PCX1_PACKET_DIGEST_KEYS`, `PCX2_EVENT_KEYS`, `PCX2_ARCHIVE_KEYS`, `PCX2_ACCOUNTING_KEYS`, `PCX2B_ROUND_EVENTS_KEY`, `SCIENCE_FILES`, `ADMITTED_ID_KEYS`, `POINT_IDENTITY_FIELDS`, `TIMING_SIDECARS`, `adapter_file`, `HARNESS`, `checked_points`, `test_omitted_flag_preserves_all_science_and_authentic_identity`, `drop_unit_record_keys` | - |
| `src/bayes_model_discovery/experimental/test_round_events_and_refill_policy_setting.py` | Import  at line 10 | 61 → 40 | `#4`, `#5`, `#6`, `#7`, `#8`, `#9`, `#10`, `#10`, `BASE_COMMIT`, `SEALED_TOOL_ENV`, `SEALED_TOOL_SHA256`, `load_sealed_tool`, `reply`, `shape_event`, `shape_cell`, `test_the_sealed_shape_tool_reads_a_cell_the_same_with_or_without_round_events`, `PARITY_HARNESS`, `TIMING_SIDECARS`, `CHANGED_SOURCES`, `normalized`, `test_every_other_chembench_record_matches_the_base_commit_byte_for_byte` | - |
| `src/bayes_model_discovery/experimental/test_space_filling_core_boxing.py` | Assign  at line 170 | 27 → 23 | `STUDY_BASE`, `LEGACY_HARNESS`, `_parity_normalize_timing`, `test_opt_in_existing_core_boxing_artifact_bytes` | - |
| `src/bayes_model_discovery/experimental/test_truncation_log_predictive_chunked.py` | Import  at line 18 | 94 → 21 | `#4`, `#5`, `#6`, `#7`, `#8`, `#9`, `#10`, `#11`, `#8`, `#9`, `#10`, `BASE_COMMIT`, `PARITY_BASE_COMMIT`, `LATER_UNIT_SOURCES`, `LATER_UNIT_CONFIG_KEYS`, `LATER_UNIT_PACKET_CONSTRAINT_KEYS`, `LATER_UNIT_FILES`, `PACKET_DIGEST_KEYS`, `LATER_UNIT_CALL_SITES`, `REPO`, `SOURCE`, `UNSPLIT`, `WIDTHS`, `POSITIVE`, `LOG_INTERFACE`, `old_source`, `old`, `NOISE`, `scalar_fit`, `catalogue_fit`, `one_model`, `POOLS`, `force_width`, `seen`, `assert_identical`, `expected_widths`, `assert_blocks`, `_top_level`, `test_only_log_predictive_changed_and_its_one_call_branch_is_the_old_body`, `test_every_family_and_width_is_byte_identical`, `test_pools_of_unequal_particle_counts_are_byte_identical`, `test_rule_r2_one_input_never_forms_a_one_column_block`, `test_the_default_budget_splits_the_dev_truncation_shape`, `test_a_call_within_one_block_runs_the_unchanged_path`, `test_refusals_are_unchanged`, `truncation_pool`, `test_gaussian_same_round_truncation_168_169`, `test_gaussian_checkpoint_scores_87_116_and_truth_law_369_371`, `test_core_one_step_log_score_920`, `test_joint_eig_638`, `chem_fit`, `counted_guarded`, `test_chembench_guarded_log_predictive_991_998`, `NARROW_CELLS`, `CALL_LOGGER`, `CORE_HARNESS`, `CHEMBENCH_HARNESS`, `BOXINGGYM_HARNESS`, `OPERATIONAL_FILES`, `OPERATIONAL_FIELDS`, `OPERATIONAL_COLUMNS`, `SCIENCE`, `_baseline`, `_run`, `_drop`, `_classify`, `_later_unit_record`, `_base_calls`, `_compare`, `_call_logs`, `test_opt_in_core_cell_with_truncation_keeps_every_science_byte`, `test_opt_in_chembench_cell_shared_log_predictive_path_and_full_artifact_parity`, `test_opt_in_boxinggym_cell_keeps_every_science_byte` | - |
| `src/bayes_model_discovery/experimental/test_wave_accounting.py` | Import  at line 4 | 32 → 26 | `#5`, `#6`, `#7`, `#5`, `BASE_COMMIT`, `test_opt_in_no_wave_options_add_only_record_derived_indexes` | - |
| `tools/fb1_snapshot/test_fb1_prompt_variant_and_name_mask.py` | ImportFrom  at line 24 | 53 → 51 | `test_the_runner_refuses_both_options_with_a_production_label`, `test_the_runner_declares_the_anthropic_reasoning_setting_to_the_label_rule` | - |
| `tools/fb1_snapshot/test_fb1_grammar_trig.py` | ImportFrom  at line 38 | 56 → 53 | `without_call_latency`, `strip_volatile`, `test_off_the_free_mode_run_equals_the_base_commit_packets_included` | - |
| `tools/fb1_snapshot/test_fb1_proposer.py` | FunctionDef test_21g2_real_sdk_mocktransport_debug_logs_and_traceback_are_masked at line 1097 | 133 → 131 | `_minimal_run_json_29fb`, `test_29fb_5_emitter_points_csv_kl_column_gets_a_value_other_columns_unchanged` | - |
| `tools/fb1_snapshot/test_fb1_epig_targets.py` | Import  at line 11 | 40 → 33 | `#8`, `#9`, `BASE_COMMIT`, `base_snapshot`, `FUNCTIONS`, `test_default_functions_equal_the_base_commit`, `test_default_zero_round_record_and_arrays_equal_the_base_commit` | - |
| `tools/fb1_snapshot/test_fb1_label_rule.py` | Import  at line 2 | 36 → 16 | `#6`, `#7`, `#8`, `#9`, `#10`, `#2`, `wave_module`, `registration`, `paper_env`, `test_matching_registered_llm_is_production_without_making_calls`, `test_registration_mismatch_fails_before_output`, `test_changed_particle_profile_cannot_inherit_paper`, `test_manifest_changed_between_import_and_main_refused_before_output`, `synthetic_run`, `production_receipt`, `test_packager_keeps_production_only_from_bound_receipt`, `test_packager_rejects_unbound_or_changed_receipts_before_output`, `test_historical_unbound_production_string_stays_development`, `test_explicit_development_mock_receipt_reaches_packaged_provenance`, `test_packager_reopens_manifest_before_returning_success` | - |
| `tools/fb1_snapshot/test_fb1_unified_protocol.py` | Import  at line 17 | 97 → 88 | `#20`, `#21`, `BASE_COMMIT`, `PLAN`, `PARAGRAPH_FILES`, `test_the_unified_paragraphs_are_the_registered_files_byte_for_byte`, `base_snapshot`, `strip_volatile`, `test_the_default_zero_round_run_equals_the_base_commit` | - |
