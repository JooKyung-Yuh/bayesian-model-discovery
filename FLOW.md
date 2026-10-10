# How the loop works

This file describes one run of the loop: what happens in each round and which module does it,
the five design criteria, the protocol the paper runs followed, the layout of a run folder, and
the terms you will meet in the code and in the outputs. The notebook
[`examples/00_how_it_works.ipynb`](examples/00_how_it_works.ipynb) gives the same material at
full length, with the formulas, the benchmark details and a longer glossary. Module names refer
to `src/bayes_model_discovery/experimental/` unless a path says otherwise.

## Round 0

1. Observe the initial design, D0: a few designs fixed by a rule rather than chosen by a
   criterion (in the synthetic count benchmark the four doses 0.1, 0.35, 0.65 and 0.9).
2. Build the starting pool. Depending on the run this is a set of starting structures (in the
   synthetic count benchmark `b0`, `b0+b1*z`, `b0+b1*z+b2*square(z)` and `b0+b1*log(1+u)`,
   with z = 2u − 1), a fixed catalogue (ChemBench), a pool read from a file, or an initial batch
   of LLM proposals. Proposed candidates go through the same validation, prior probe and fitting
   as in later rounds.
3. Keep the twelve models with the highest evidence and score the pool (phase `initial`).

## One round, step by step

Every later round adds one observation. The table follows the synthetic count runner
(`core_count.py`); the other runners go through the same steps in the same order and differ
mainly in when they score.

| Step | What happens | Where |
| --- | --- | --- |
| 1. Choose | Evaluate the design criterion on the candidate designs with the pool of the previous round and take the best; ties within 1e-12 of the maximum go to the smallest design. | `design_policies.select`, `count_inference.choose`, `count_inference.choose_epig`; `positive_inference.joint_eig` and `vector_inference.joint_eig` for other outcome types; continuous search in `design_search.projected_spsa`; the argmax in `eig.joint.choose_design` |
| 2. Observe | The benchmark returns the outcome at the chosen design and it joins the public data D. Each pool member first gives a one-step-ahead log score for it. | `core_count.CoreWorld.observe`; the benchmark's own simulator or environment |
| 3. Refit | Every pool member is refit by sequential Monte Carlo (SMC) from its prior on all of D. | `count_inference.fit`, `positive_inference.fit`, `vector_inference.fit` |
| 4. Score | The pool's averaged prediction is scored on the held-out set (phase `before_proposal`). | `count_inference.predict`, `core_count.score_prediction` |
| 5. Build the packet | The public state is written as a belief packet. | `belief_packet.build_belief_packet`, `belief_packet.serialize_packet` |
| 6. Propose | One request asks the LLM for k candidates; on an archive review round it may also name up to two retired models to revive. The mock proposer answers from a script. | `count_proposer.py`; `neuron_proposer.py` for NeuronBench |
| 7. Validate | Grammar, roles and family are checked, duplicates are found by model id, and the prior probe runs. An invalid reply gets one repair turn. | `count_models.validate_expression`, `count_models.judge_candidate`, `count_models.canonical_model_id`, `core_count.prior_probe` |
| 8. Fit | Each valid candidate, and each revived model, is fitted by SMC from its prior on all of D. | as in step 3 |
| 9. Admit | All models are ranked by log evidence (ties by model id); the top twelve stay, the rest are marked `retired_by_evidence`. | `count_inference.retain` |
| 10. Score again | If the pool changed it is scored again (phase `after_proposal`); otherwise the previous row is copied. | as in step 4 |

The ForceBench runner, `tools/fb1_snapshot/run_gravity.py`, follows the same steps with its own
modules in that folder (`fb1_proposer.py`, `fb1_grammar.py`, `fb1_core.py`, `fb1_forward.py`,
`fb1_design_search.py`).

**Evidence weights.** Each SMC fit runs two chains; their evidence estimates are averaged on the
evidence scale. The weight of a model is its share of the total evidence of the pool
(`count_inference.model_logweights`), which amounts to a uniform prior over the models in the
pool. The weights are therefore conditional on the pool: adding or retiring one model changes
all of them, and a weight close to 1 means "far better supported than the other pool members",
not "the true law". Every fit starts from the prior and uses all public data.

**Prediction.** The pool predicts by Bayesian model averaging (BMA): each model's predictive
distribution, averaged over its posterior particles, mixed with the evidence weights.

**Isolation.** The held-out evaluation data are used only for scoring. They never reach fitting,
design selection or the proposer, and `config.json` states this as `evaluation_use`.

## The five design criteria

The benchmark runs compare five criteria, the arms. Joint, Model and Parameter expected information gain (EIG) come from one set of
posterior draws, so joint EIG = Model EIG + Parameter EIG holds exactly at every design. Below,
"H" is entropy in nats and "the pool predictive" is the BMA predictive distribution of the
outcome at design x.

| Arm | Formula in words |
| --- | --- |
| Joint EIG (`joint_eig`, the default) | H(pool predictive at x) minus the evidence-weighted average, over models and their particles, of H(predictive of one particle at x). Information about the model and its coefficients together. |
| Model EIG (arm `model_eig`; `--design-policy structure_eig`, and `structure_eig` in most records) | H(pool predictive at x) minus the evidence-weighted average over models of H(the model's particle-averaged predictive at x). Information about which model in the pool is right. |
| Parameter EIG (`parameter_eig`) | Joint EIG minus Model EIG: the evidence-weighted average over models of the information about each model's coefficients. |
| expected predictive information gain, EPIG (`epig`) | The average, over 32 target inputs drawn once at the start of the run, of the mutual information between the outcome at x and the outcome at the target. |
| Random (`random`) | One candidate design drawn from a random stream of its own, so the other random numbers of the run are not affected. |

For outcomes with finitely many values (counts out of a fixed number of trials) the entropies
are exact sums over all outcomes; for positive, real-valued and vector outcomes they are Monte
Carlo estimates with a recorded standard error. `--design-search ascent` searches the declared
input box continuously instead of a finite candidate set.

## The protocol of the paper runs

The paper's experiments on ChemBench, BoxingGym, ForceBench and NeuronBench follow one shared
procedure, the unified protocol (`u3` in the tag names is its revision). The synthetic count
benchmark is not part of the paper; it serves development and tests.

- **Initial batch.** Round 0 sends twelve independent requests in parallel
  (`--initial-proposals 12`), each for one candidate. If fewer than twelve distinct candidates
  are admitted, at most two top-up waves ask for the number still missing; a remaining gap is
  recorded as the shortfall.
- **Pool of twelve.** The pool keeps at most twelve models (`--active-maximum 12`).
- **Three candidates per call.** Every later round makes one LLM request for three candidates
  (`--proposal-limit 3`, with the multi-candidate paragraph, version 2). Duplicates are recorded
  and not fitted, and nothing replaces them, so at most three new models are fitted per round.
  Only a reply that cannot be used at all gets one repair turn.
- **Archive review.** Every fifth round the request also invites the LLM to name up to two
  models with outcome `retired_by_evidence` to be refit (revive). Valid ids are refit from the
  prior on all current data and compete in the same ranking as the new candidates; other ids are
  recorded with the reason and ignored. The revive list comes in the same reply, so a review
  round costs no extra call.
- **Horizons and seeds.** 30 rounds on ChemBench and BoxingGym, 16 on ForceBench and 5 on
  NeuronBench; seeds 31, 37 and 41 (BoxingGym peregrines uses 47 instead of 37, because seed
  37's initial design observes only zeros).
- **Arms.** All five on BoxingGym, ForceBench and NeuronBench; ChemBench has no EPIG arm.
  Within a benchmark and world, every arm uses the same prompt, grammar, likelihood families,
  priors, compute profile, initial-design rule and seeds; only the design criterion differs.
- **What the LLM is given.** The system prompt `tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt`
  (version tag `v2.9.FA-C`; ForceBench oscillator runs use the variant with sine and cosine),
  and each round's belief packet with input names masked as `X1`, `X2`, … and no task name.
- **NeuronBench exceptions.** Candidates come from a fixed vocabulary of ten mechanism names and
  a mechanism prompt of its own. The run starts from four fixed mechanisms instead of an initial
  batch, keeps at most six models (profile `paper_neuron`), and has no archive review; of the
  protocol it adopts only the three-candidate call.

[`USAGE.md`](USAGE.md#reproducing-the-benchmark-runs) names the tags of the private development
repository that hold the exact code of the runs (they are not public) and how to check a run's
recorded digests against a checkout.

## The run folder

Every runner writes one run into one new folder (`--output`); no runner overwrites. Files named
`tNN` belong to round NN, round 0 being the initial design and starting pool. The synthetic
count and BoxingGym runners write the files below (`llm_transport/` only with a real provider);
the other runners write most of them.

| Path | Contents |
| --- | --- |
| `config.json` | every setting, package versions, seeds, the prompt version and sha256, the sha256 of each engine source file; read this first |
| `summary.json` | status, timings, admissions, candidate failures, the number of provider calls, the stop reason if a cap ended the run |
| `failure.json` | only when a run fails: stage, error and traceback (key-shaped strings masked) |
| `observations.jsonl` | one line per public observation: design, outcome, repetition, seed |
| `scores.jsonl` | one line per scored checkpoint: round, phase, pool members and weights, held-out NLL, coverage90, width90, KL |
| `points.csv` | the same checkpoints as one table in the fixed format the analysis tools read |
| `archive.json` | every model proposed or held, with its outcome and a structured reason |
| `packets/tNN.json` | the belief packet of round NN |
| `proposals/tNN.json` | the raw response, validation and duplicate events, the pool before and after |
| `fits/tNN_<model_id>.json`, `.npz` | one SMC fit: log evidence per chain, diagnostics, prior-probe summary; the posterior particles |
| `predictions/` | predictive distributions at each checkpoint |
| `acquisition/tNN.json` | the value of every candidate design under the criterion, and the chosen design |
| `evaluation_scoring_only.npz` | the held-out data, used only for scoring |
| `llm_transport/` | only with a real provider: one record per request and response |

Runners with archive review rounds also write `revival/tNN.json`: the revive requests of each
review round and their outcome. NeuronBench writes `proposals.jsonl` and `acquisition.jsonl`
instead of the two folders and no `packets/`; ForceBench writes flat files that share one prefix
(`<run>.json`, `<run>_packets/`, `<run>_revival/`, …). Some records (`output` and `python` in
`config.json`, `source_path` in `points.csv`) hold absolute paths of the machine that made the
run; check them before sharing a run folder.

## Glossary

- **Round, `tNN`.** One pass of the loop; files named `tNN` belong to round NN.
- **D0 (initial design).** The observations made before round 1, at designs fixed by a rule
  (`--d0-policy`).
- **Belief packet.** Everything the LLM receives besides the system prompt, built from public
  data only: observations, the pool with weights and coefficient summaries, residuals of the
  best model, the archive, the run's constraints and the interface block.
- **Interface block.** The only task-specific part of a packet: input names and ranges, derived
  inputs, outcome type and support, and the link function. No free text and no true values.
- **Candidate.** One hypothesis from the LLM: an expression for the mean on the link scale, a
  likelihood family and the role of each coefficient, as JSON (in NeuronBench, a mechanism name).
- **Validation, repair turn.** The checks a candidate passes before fitting (grammar, at most six
  coefficients and fifteen operators, allowed roles and families, no duplicate). A reply that
  fails gets the error back once.
- **Model identity (`model_id`).** A canonical key from the expression's syntax tree, the
  coefficient roles and the family. Renaming coefficients does not change it; scaling a term
  does.
- **Duplicate.** A candidate whose model id is already in the pool, in the archive or earlier in
  the same reply; recorded, not fitted.
- **Prior probe.** 64 coefficient draws from the candidate's priors, evaluated on a 16-point
  grid; a non-finite value rejects the candidate before any fit.
- **SMC.** Sequential Monte Carlo (`pm.sample_smc` in PyMC), the sampler that fits each model
  and estimates its evidence; two chains per fit.
- **Particle.** One posterior draw of a model's coefficients.
- **Evidence (`logZ`, `log_evidence`).** The marginal likelihood p(D | m) of a model.
- **Evidence weight (`model_weights`).** A model's share of the pool's total evidence;
  conditional on the pool.
- **Pool (`active_maximum`).** The models currently kept, at most twelve (six in NeuronBench's
  paper profile).
- **Admission.** A candidate is admitted when it passes validation and the prior probe, its fit
  succeeds and its evidence ranks within the pool limit.
- **`retired_by_evidence`.** The archive outcome of a model that dropped out of the pool because
  others had higher evidence.
- **Archive (`archive.json`).** Every model proposed or held in a run, with its outcome
  (`active`, `retired_by_evidence`, `rejected`, `duplicate`, `fit_failed`) and reason.
- **Initial batch (burst).** Round 0's `--initial-proposals K` parallel requests, with at most
  two top-up waves (`proposals/t00_burst.json`).
- **Multi-candidate paragraph.** Text appended to the system prompt when one request asks for
  several candidates (`--composition-paragraph v2` in the ChemBench runner).
- **Archive review round, revive.** A round on which the LLM may name up to two retired models
  to be refit (`--archive-review-rounds`; records in `revival/tNN.json`).
- **Scoring phase.** `initial`, `before_proposal` and `after_proposal` rows in `scores.jsonl`.
- **Design.** The inputs of one experiment; criteria are evaluated on candidate designs.
- **Arm, design policy.** The design criterion of a run (`--design-policy`, or the last digit of
  the condition code).
- **Point disagreement.** The evidence-weighted spread of the models' predicted means at a
  design; recorded beside the criteria.
- **Design search (`--design-search pool|ascent`).** Best of a finite candidate set, or a
  continuous search of the input box.
- **Held-out evaluation set (`q`).** Inputs and outcomes used only for scoring.
- **BMA.** Bayesian model averaging: the pool's evidence-weighted predictive distribution.
- **NLL.** Negative log predictive probability (or density) of the held-out observations under
  the BMA, in nats per observation; lower is better.
- **coverage90, width90.** The fraction of held-out observations inside the 90% predictive
  interval, and that interval's average width.
- **KL (`kl`).** KL divergence from the true outcome distribution to the BMA predictive,
  averaged over the evaluation inputs.
- **World.** One task instance with a fixed true law (`--seed` in the synthetic count runner,
  `--task` in BoxingGym, `--world` in NeuronBench, a world name in ForceBench).
- **Seed.** Drives the engine's random streams; what else it fixes differs by benchmark (it
  selects the world in the synthetic count runner and the environment's hidden parameters in
  BoxingGym).
- **Condition code (`--cell`).** Three digits S F A: S = 1 allows new structures, F = 1 new
  likelihood families, A = 0 random designs, 1 an EIG-type criterion, 2 EPIG.
- **Profile (`--profile`).** A compute preset (`dev`, `paper`, `paper_neuron`); all profiles use
  the same priors, grammar, likelihoods and sampler.
- **Mock proposer.** The default proposer: scripted candidates instead of an LLM, no key, no
  cost (`mock_fixed_template_responses_NOT_LLM` in `config.json`).
- **Prompt version, prompt variant.** `--prompt-version` selects a numbered file in `prompts/`;
  `--prompt-variant <file>` sends another file. Runs record `prompt_version` and `prompt_sha256`.
- **Paid-call guard (`--max-paid-calls`).** A per-run cap on provider calls; the run stops at the
  cap and `summary.json` records why.
- **Name masking (`--mask-input-names`).** Input names replaced by `X1`, `X2`, … in the packet.
- **Source digests (`source_files_sha256`, `code_sha`).** The sha256 of the engine source files a
  run loaded, recorded in `config.json`; every row of `points.csv` carries the runner's own
  digest as `code_sha`.
