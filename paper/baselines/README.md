# Baseline wrappers

The code we wrote around the four baselines of the paper. Each baseline's released implementation is
**not** here: `scripts/provision.sh` fetches it at its pinned commit (`pins_and_patches/pins.json` lists
the upstream repositories, commits and licenses), and the wrappers import it from that checkout.

| Folder | Baseline | What the wrapper does |
| --- | --- | --- |
| `tools/icl_baseline/` | ICL on ChemBench and NewtonBench (our own baseline; `ng/` is the NewtonBench variant) | the design request and the separate prediction request per observation count, the closed plan, the transport with its two attempts and repair text, the per-timepoint validity rule, scoring |
| `tools/icl_boxinggym/` | ICL on BoxingGym | the same protocol over BoxingGym's listed candidate designs |
| `tools/icl_forcebench/` | ICL on ForceBench | the same protocol with 64 random `(p1, p2)` settings per round; the masked, neutral and full interfaces |
| `llm_autoscilab_wrapper/` | LLM-AutoSciLab | runs the released loop one experiment per round with the equation learner after every experiment, refuses the released loop's ground-truth evaluator queries (`blocked_attempts` in the receipt), records the current law per round, and on ForceBench integrates the force law to the distance observation (`asl_kit/forcebench.py`) |
| `box_apprentice_wrapper/` | Box's Apprentice (BoxingGym) | a checkpoint after every experiment, the three text substitutions that remove the `<thought>` request Claude Opus 5.5 refused, batched posterior predictive predictions, the cost limit; `patches/ccr5b_retry_v3.patch` retries a failed generated program through the released loop |
| `dp_agent_wrapper/` | DP-Agent (ForceBench) | the one-experiment-per-round protocol constraint, the current-law request per round, the budget guard, and the post-run fit and scoring of each recorded law with the benchmark's own fitting functions |
| `pins_and_patches/` | all | `pins.json` (upstream, commit, license per baseline) and the small patches applied to the BoxingGym checkout for the baseline runs: a cost-table entry for a model name the released code did not know, the retry of a failed generated program (three versions), a JSON default for ndarray values |

Changes to the released implementations, in one place: we keep each released algorithm and change only
what the comparison needs. (1) A prediction after every experiment: an agent that states a law less often
is asked for its current law every round (LLM-AutoSciLab: learner every round, no ten-observation minimum;
DP-Agent: a current-law request per round; Box's Apprentice: a checkpoint per experiment). (2) One
experiment per round for every agent. (3) Prompt text that Claude Opus 5.5 refused is removed (the
Apprentice `<thought>` tag). (4) On ForceBench, LLM-AutoSciLab reads the trajectory observation (the force
law is integrated from the fixed initial state; PySR is off, because regressing the response on the inputs
returns an orbit, not a force law). All changes live in these wrappers; with every option off they send
the same requests and record the same outputs as the released code.

`adapters/` re-exports the observation adapters of `bayes_model_discovery.adapters` under the name the
wrappers import.

## What is not here

- The released baseline code (fetched by `scripts/provision.sh`); the file copied from it for the
  development runs (`vendor/run_experiment.py` of the Apprentice kit); the data packs built from the
  benchmarks' own generators (`packs/*.json`: evaluation points, world parameters, ForceBench truth
  trajectories); the fork-versus-upstream diff of BoxingGym (it carries upstream code).
- The launch material of the development hosts: queue and provisioning scripts, host environment files,
  plans, registration documents with their sealed plan drafts, and the tests that exercised them.

## Tests

The kits keep their own tests. With `PYTHONPATH=paper/baselines:src` and the benchmark checkouts on the
path, this is what ran on the development host after the launch material was removed:

| Kit | Result | Needs |
| --- | --- | --- |
| `tools/icl_boxinggym` | 43 passed | the BoxingGym checkout (`BMD_BOXINGGYM_SRC`) |
| `tools/icl_baseline` and `tools/icl_forcebench` together | 85 passed, 44 skipped (the skips need the LLM-AutoSciLab checkout, a frozen engine tree with its registration seals, and jax) | the LLM-AutoSciLab checkout (`BMD_AUTOSCILAB`), the NewtonBench adapter, the ForceBench environment (`FB_PY`) |
| `llm_autoscilab_wrapper` | the tests that read the data packs and recorded runs fail (not here); the others pass | the LLM-AutoSciLab checkout, the packs built by `build/*.py` from the benchmark checkouts |
| `box_apprentice_wrapper` | not run (needs the BoxingGym checkout's dependencies, the pack and the vendored file) | the BoxingGym checkout with its dependencies, `APPR_BG_CHECKOUT`, `APPR_ASL_KIT` |
| `dp_agent_wrapper` | `kit/test_dp_current_law.py` runs with the DiscoverPhysics checkout | `FB_DP` |
