# Aggregate data behind the paper's appendix tables and figures

These files hold the numbers the appendix tables and the learning-curve figures are built from: per-run error
series and the aggregates computed from them. They are not the run records (those stay with the
development repository); every number here is derived from a run record by the rules stated in each file's
`description` or `rule` field and in the appendix. Record paths, host names and internal campaign names were
removed by `sanitize_tables.py`, which is kept here for the record (its `<record path>` marks a removed path).

`tables.py` renders the appendix tables from the JSON files (`python paper/data/tables.py` prints them all;
`--table E1` prints one). It reads only this folder and needs no engine.

## `curves/`

| File | Contents |
| --- | --- |
| `figure2_learning_curves.json` | Figure 2: the error of every run of every method at every observation count, `benchmarks -> method -> "task|seed" -> {n: error}`. BMD = the paper's main runs; LLM-AutoSciLab on ForceBench = the matched trajectory-interface runs, with the earlier force-oracle runs kept under their own key. Unfloored. |
| `design_and_feedback_curves.json` | The design-criterion (C4) and feedback (C3) runs: `series -> dataset -> arm -> "task|seed" -> {n: error}`. |
| `design_and_feedback_contrasts_floor1e-06.json`, `…floor1e-300.json` | The paired contrasts of those datasets against the joint-EIG / Full arm (final count and curve mean, task-cluster bootstrap), with the error floored at 1e-6 and effectively unfloored. |
| `forcebench_design_criteria_curves.csv` | The ForceBench design-criterion curves (full interface), aggregated per arm and observation count with their bands. |
| `structural_recovery_curve_earlier_judging.csv` | The per-round recovery fractions of the earlier judging pass (rule c5-judge-1.3) per benchmark and arm. |

## `appendix_tables/`

| File | Appendix table |
| --- | --- |
| `E1_paired_comparison.json` | paired comparison of BMD with each baseline on shared runs (final and middle counts; the Figure 2 floor rule, the all-benchmark floor, unfloored) |
| `E2_per_task_final_error.json` | final error per task and method as the geometric mean over seeds, with the loss-matched (Bayes-action) BMD column |
| `F1_outcome_classes.json` | structural-recovery outcome classes of the final top-1 law per method and benchmark, with the proposal / selection failure split |
| `D3_cost_calls.json` | LLM calls and cost per run, and the blocked ground-truth evaluator attempts in the LLM-AutoSciLab receipts |
| `H1_arm_identity.json` | what the arms of the design-criterion and feedback comparisons share (prompt digests, round 0, feedback setting) |
| `G4_c2pert.json` | the perturbed-constant shared-candidate study |
| `H_breadth.json` | joint EIG against random designs on 24 further ChemBench tasks |
| `C_forcebench_configurations.json` | the integrator settings of the ForceBench editions, read from the run records |

Errors: nMSLE on the 256 evaluation inputs (ChemBench, NewtonBench), the benchmark's headline error
(BoxingGym), nMSE on six shifted designs (ForceBench, "shift") or on the benchmark's three designs
("headline"). Observation count n = initial observations + rounds (ChemBench 3, BoxingGym 2, ForceBench 2,
NewtonBench 4 initial observations).
