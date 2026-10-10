# Usage

How to install and provision, run one cell of each benchmark, make paid runs safely, run the
tests and the notebooks, and reproduce the paper runs. [`REQUIREMENTS.md`](REQUIREMENTS.md)
lists what each step needs and [`FLOW.md`](FLOW.md) explains what a run does. Run every command
from the repository root.

## Install and provision

By hand, in an environment with Python 3.12 and a C/C++ compiler:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e ".[dev]"
python -c "import bayes_model_discovery, pymc; print(bayes_model_discovery.__version__, pymc.__version__)"
```

The last line should print `0.1.0.dev0 5.13.1`.

On a clean Linux host the provisioning script builds the same environment from the lock file
(without pytest and ruff) and does more. It costs nothing (no LLM calls), never reads a `.env`
file, and can be re-run: each step checks whether its result already exists.

```bash
bash scripts/provision.sh [--root DIR] [--python 3.12.13] [--with boxinggym,neuronbench,forcebench] [--smoke]
source .bmd_env.sh
```

- It checks for git, curl and `g++`, installs Python 3.12.13 with `uv` if no matching interpreter
  exists, builds `.venv` from `requirements.lock`, installs the package in editable mode, checks
  the pinned versions, and checks the sha256 of the paper's system prompt.
- `--with` fetches the external benchmark code at its pinned commit into `external/`:
  `boxinggym` (a sparse checkout of `src` and `conf`), `neuronbench` (also installed into
  `.venv`), `forcebench` (DiscoverPhysics, plus the separate `.venv-forcebench` built from
  `requirements-forcebench.lock`). ChemBench is not offered: it is not enabled in this release.
- It writes `.bmd_env.sh` with the variables listed in
  [`REQUIREMENTS.md`](REQUIREMENTS.md#environment-variables). `--smoke` ends with a one-round mock
  run of the synthetic count benchmark. The last line starts with `SUMMARY fail=N`, the number
  of failed steps, which is also the exit status.

Without the script, clone each repository yourself, check out the pinned commit and set the
variable. BoxingGym is shown; NeuronBench works the same way with `NEURONBENCH_SRC`, and
ForceBench needs `FB_DP` and the separate environment of
[`REQUIREMENTS.md`](REQUIREMENTS.md#two-python-environments) (`FB_PY`).

```bash
git clone https://github.com/kanishkg/boxing-gym external/boxing-gym
git -C external/boxing-gym checkout b43e38cb03d09c13efa9cf4d9bae740d51157bfd
export BMD_BOXINGGYM_SRC="$PWD/external/boxing-gym/src"
```

## Run one cell with `scripts/bmd_run.py`

`scripts/bmd_run.py` runs one cell (one benchmark, task or world, seed and design criterion)
through that benchmark's own runner. It translates the arm into the runner's condition code and
design policy, puts the pinned external checkout on `PYTHONPATH`, and runs the command:

```text
python scripts/bmd_run.py <synthcount|chembench|boxinggym|neuronbench|forcebench> --output DIR
    [--arm joint_eig|model_eig|parameter_eig|epig|random] [--seed N] [--rounds N] [--profile dev|paper]
    [--task NAME] [--proposer mock|anthropic] [--allow-paid-api] [--dotenv FILE] [--dry-run] [-- runner options]
```

| Benchmark | Runner | Arms | `--task` | Needs |
| --- | --- | --- | --- | --- |
| `synthcount` | `core_count` | `joint_eig` (cell 111), `model_eig` (111 with `--design-policy structure_eig`), `epig` (112), `random` (110) | not used; `--seed` selects the world (11, 17, 23, 31, 37, 41 or 99) | nothing |
| `boxinggym` | `boxinggym_core` | all five (Model and Parameter expected information gain (EIG) are cell 111 with `--design-policy structure_eig` or `parameter_eig`) | the task, default `death` | `BMD_BOXINGGYM_SRC` |
| `neuronbench` | `neuron_core` | all five (Model and Parameter EIG are cell 111 with `--design-policy structure_eig` or `parameter_eig`) | the world, default `h_sag` | `NEURONBENCH_SRC` |
| `forcebench` | `tools/fb1_launch_u/run_cell.sh` | all five | the world, default `gravity` | `FB_PY`, `FB_DP` |
| `chembench` | `chembench_core` | not enabled in this release | | |

- Defaults: `--arm joint_eig`, `--seed 31`, `--rounds 2`, `--profile dev`, `--proposer mock`.
  These cost nothing.
- `--profile paper` selects the paper's compute profile (`paper_neuron` for NeuronBench). The
  NeuronBench runner accepts 1 to 5 rounds; the BoxingGym runner defaults to 20 rounds and the
  paper runs used 30.
- The BoxingGym runner carries the protocol options of the paper runs (initial batch of twelve,
  three candidates per call, archive review every fifth round) as opt-in flags; how a run is tied to
  the code that produced it is described in
  [Reproducing the benchmark runs](#reproducing-the-benchmark-runs).
- ForceBench takes its budget from its launch script: a mock run lasts `FB_MOCK_T_EXTRA` rounds
  (default 2) with an archive review on round `FB_MOCK_REVIEW` (default 2); a paid run lasts 16
  rounds, or 30 with `FB_BUDGET=30`. `--rounds`, `--profile paper` and runner options are
  refused for ForceBench.
- `--dry-run` checks the arguments and the external checkout variables, prints the runner command
  and exits without running it.
- Anything after `--` goes to the runner unchanged, for example `-- --structure den`.
  `python -m bayes_model_discovery.experimental.<runner> --help` lists every runner option; the
  runners can also be run directly that way.

Examples:

```bash
python scripts/bmd_run.py synthcount --arm model_eig --rounds 2 --output runs/sc_model_eig
python scripts/bmd_run.py boxinggym --arm epig --task death --seed 31 --rounds 1 --output runs/bg_epig
python scripts/bmd_run.py neuronbench --arm parameter_eig --task h_sag --rounds 1 --output runs/nb_param
python scripts/bmd_run.py forcebench --arm joint_eig --task gravity --seed 31 --output runs/fb_mock
python scripts/bmd_run.py boxinggym --task dugongs --output runs/bg_check --dry-run
```

**Output folders.** `--output` must name a folder that does not exist yet; the script, the
runners and the ForceBench launch script all refuse to overwrite one. When `PYTENSOR_FLAGS` is
not set, the PyTensor compile cache goes beside the run, in `<output>.ptc`. `.bmd_env.sh` sets
one shared cache for the repository; for runs in parallel, give each run its own
(`PYTENSOR_FLAGS=base_compiledir=<folder>`).

## Paid runs

The mock proposer is the default everywhere. A real LLM needs three flags together, and the
script refuses a paid run that lacks any of them, before anything is created:

```bash
python scripts/bmd_run.py neuronbench --arm joint_eig --task h_sag --rounds 5 --profile paper \
  --output runs/nb_paid --proposer anthropic --allow-paid-api --dotenv .env -- --max-paid-calls N
```

- `--dotenv FILE` is the path of a dotenv file holding the key. The script takes only
  `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from it, passes them to the runner's environment and
  prints only their names. For ForceBench it hands the path to the launch script
  (`FB_DOTENV`), whose relay process reads the key and never prints it.
- `--max-paid-calls N` caps the provider calls of one run. It is a runner option, so it goes
  after `--`. When the next call would exceed the cap, the run stops with exit status 4 and
  `summary.json` records the stop reason `budget_exhausted`. Without the option there is no cap,
  so set it on every paid run. The ForceBench launch script caps its runs at 48 calls (64 with
  `FB_BUDGET=30`); `FB_MAX_PAID_CALLS` overrides that.
- The runners read keys only from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  or the variable named by `--llm-key-name` for an OpenAI-compatible endpoint), never from a
  file, and mask key-shaped strings before they write a record. Run a runner directly for the
  `openai` and `openai_compat` providers.
- The runners use their default model (`DEFAULT_MODEL` in `count_proposer.py`) unless
  `--llm-model` says otherwise. The paper runs used `claude-opus-5-5` with
  `--llm-reasoning effort=medium,thinking=adaptive,display=summarized` and
  `--llm-max-tokens 16000`; with the script, pass these runner options after `--` as well.
- Never commit `.env` (it is in `.gitignore`; `.env.example` is the template), and never print,
  paste or display its contents: not in a terminal you share, a log, an issue or a notebook.

## Tests

Tests live in `tests/` and next to the engine (`src/bayes_model_discovery/experimental/test_*.py`).
Run them from the repository root with `PYTHONPATH=src`, so that the source tree is tested
rather than an installed copy. pytest comes with the `dev` extra.

```bash
# fast: the unit tests under tests/ (a few minutes)
PYTHONPATH=src pytest -q -m "not heavy and not replay" tests
# standard: everything CI runs, including the runners with mock proposers (about 4,000 tests)
PYTHONPATH=src pytest -q -rs -m "not heavy and not replay"
```

- The standard tier takes about two hours, so after changing one runner run its own test files
  plus `test_run_label_policy.py` and `test_count_proposer.py`. Give every test run its own
  `--basetemp` and `PYTENSOR_FLAGS=base_compiledir=<folder>`.
- Tests that need an external checkout skip, naming the variable, unless `BMD_BOXINGGYM_SRC` or
  `NEURONBENCH_SRC` names the pinned checkout and that checkout is on `PYTHONPATH`; they fail if
  the checkout is at another commit or not clean. `-rs` prints the reason for every skip.
- On a host whose CPU has AVX-512, the NeuronBench test that checks the two forward backends bit for bit
  (`test_n1_the_two_backends_score_and_observe_the_same_rows`) skips unless numpy's SIMD power dispatch is off, as
  the runner requires for `--forward-backend batch`: set
  `NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"`
  before running it.
- The ForceBench tests are `tools/fb1_snapshot/test_*.py` and need the ForceBench environment.
  Run them from that folder with `JAX_PLATFORMS=cpu`, `FB1_PYTHON=$FB_PY`,
  `DP_SCIENCEAGENT_DIR=$FB_DP/ScienceAgent` and `FB1_OUT_DIR=<scratch folder>` set explicitly:
  `$FB_PY -m pytest -q --confcutdir=. <test file>`.
- CI (`.github/workflows/ci.yml`) runs `ruff check src tests scripts tools/release`, an
  install-and-import check, a two-round mock run through `scripts/bmd_run.py`, and the standard
  tier. CI installs no benchmark source, so the tests that need one skip there, and the ForceBench
  kit tests are not part of CI.

## Notebooks

The notebooks in `examples/` run with the mock proposer. Install Jupyter into the engine
environment and open them from inside the repository:

```bash
pip install jupyterlab
jupyter lab examples/
```

Notebook 00 runs no model. Notebook 01 runs the quickstart into
`examples/runs/quickstart_synthcount/`, which notebooks 02 and 04 read; notebook 04 adds a
second run, and notebook 03 needs no run folder. `examples/runs/` is ignored by git, and a
notebook skips a run whose folder exists, so delete the folder to run it again.

To re-execute the committed notebooks after a change, delete `examples/runs/`, then execute
each notebook from the repository root in a fresh kernel, in the order 00, 01, 02, 04, 03, keep
the standard `python3` kernelspec in their metadata, and check them:

```bash
for name in 00_how_it_works 01_engine_basics 02_what_the_llm_sees 04_reading_a_run_folder 03_eig_decomposition; do
  jupyter execute --inplace "examples/$name.ipynb"
done
PYTHONPATH=src pytest tests/unit/test_example_notebooks.py -q
```

The test reads the committed files with the standard library and checks the notebook structure,
that every code cell ran once in order, that no cell has an error, that no output contains an
absolute path of the executing machine, that nothing looks like an API key, and that each file
stays under 1 MB. Run records contain absolute paths; the notebooks print them as `<path>`.

## Prompt files are identifiers

The text sent to the LLM is part of an experiment, like a seed. Every run records which prompt
it sent: `prompt_version` and `prompt_sha256` are in `config.json` for the synthetic count,
ChemBench and BoxingGym runners, in every packet and request record for NeuronBench, and in the
run file for ForceBench; multi-candidate requests also record the digest of the composed text
(`composed_prompt_sha256`). In an archive-review round that digest covers the whole text that was
sent, system prompt plus both paragraphs; BoxingGym records written before 2026-10-07 hold the
value before the review paragraph was appended, and the review text's own digest is in
`archive_review_prompt_sha256` in both cases.
The files and constants below stay byte for byte in the tree and are never edited, not even to
fix a typo. A new prompt version is added as a new file, never by editing an existing one. The
packaged prompt files are in `src/bayes_model_discovery/experimental/prompts/`;
`--prompt-version v2.6.7.1`, for example, selects `proposer_system_v2_6_7_1.txt`.

| Text | Location | sha256 |
| --- | --- | --- |
| system prompt `v2.9.FA-C` of the paper runs (3,548 bytes) | `tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt` | `196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c` |
| the same with sine and cosine in its operator list (ForceBench oscillator runs; 3,568 bytes) | `tools/fb1_launch_c/proposer_system_v2_9_FA-C_trig.txt` | `4dac3b091359f4297dadfebe7e7525e54a718069fb48c5c9f5f70105e6499ab9` |
| the same with a block that lets the proposer write numeric coefficient priors (`v2.9.FA-C-priors`, 4,969 bytes; selected by `--prompt-version v2.9.FA-C-priors`) | `tools/fb1_launch_c/proposer_system_v2_9_FA-C_priors.txt` | `dc03d1a61dc484c6bc28aa216887e82469f00ec791c568b01f96fb65b9779324` |
| multi-candidate paragraph, version 2 (525 bytes) | `COMPOSITION_PARAGRAPH_V2` in `src/bayes_model_discovery/experimental/count_proposer.py` | `782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6` |
| archive review paragraph, version 1 (675 bytes) | `ARCHIVE_REVIEW_PARAGRAPH` in the same file | `8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde` |
| prompt of the initial-pool generator (`--pool-generator pool_generator_v1`) | `prompts/pool_generator_v1.txt` | `a4466c0f6a54e0380f630481f6501802e5a75a0e981d290f310a64834eecc1a4` |
| system prompt `v2.0`, the default | `prompts/proposer_system_v2_0.txt` | `f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef` |
| system prompt `v2.1` | `prompts/proposer_system_v2_1.txt` | `2edb4e0c11ba00cb6f07be4ddef48d4da75893a717c3c22ca767dfc2bd33a273` |
| system prompt `v2.2`, used with `--priors llm` | `prompts/proposer_system_v2_2.txt` | `e828e6ae587131766513d7bd7127ec2bcf302bcf02528797d2ec3327e6ab60ba` |
| system prompt `v2.3` | `prompts/proposer_system_v2_3.txt` | `1a5f9ba920a9bb5ded76206e3f7e3af2d86471fbde6e13345c752f9257493900` |
| system prompt `v2.4` | `prompts/proposer_system_v2_4.txt` | `1e0923f5a3bf9d76f0ecd2418cb5f2c5d1bca24fc49d91b5cf029279e8007461` |
| system prompt `v2.4.1` | `prompts/proposer_system_v2_4_1.txt` | `84229e9949c5ca5049f2afa5ec536966b6e24c60d02388f87da00d9e0f6596cb` |
| system prompt `v2.5` | `prompts/proposer_system_v2_5.txt` | `3969092f63f5ec6e76ae98ab9d44b8be62873f1dcdfd31dff80c5534a1e742ad` |
| system prompt `v2.6.1` | `prompts/proposer_system_v2_6_1.txt` | `ba5faab3136111b9523e9ac4b01bc6b3a26640aa139ba781abdbc38544ec2961` |
| system prompt `v2.6.2` | `prompts/proposer_system_v2_6_2.txt` | `13bf1390990aa0036384d8d8f94341201b11a0f613b0be576efd6182922e8ed3` |
| system prompt `v2.6.3` | `prompts/proposer_system_v2_6_3.txt` | `843ee2a64774ae935fd49fc75dd9b2bfb9768cb8d1c21794323880b6608622ad` |
| system prompt `v2.6.4` | `prompts/proposer_system_v2_6_4.txt` | `81380b6e5a1efb8570cb4fd1137df4d2e494fd5a4a2bba0c954ce5c827e18fb2` |
| system prompt `v2.6.5` | `prompts/proposer_system_v2_6_5.txt` | `c21da86c44afb826f96dde109451ba9ce397325ac9be8df96b4042348b57170e` |
| system prompt `v2.6.6` | `prompts/proposer_system_v2_6_6.txt` | `e7cb43b039769ac91751806e0675de547ab2ca9e4d62c9e56c661e0f9a5a7d6c` |
| system prompt `v2.6.7` | `prompts/proposer_system_v2_6_7.txt` | `5b2172d70f93a21f1c46c2609c9b089a9062a9f9e49995953437fcbc98ff3b99` |
| system prompt `v2.6.7.1` | `prompts/proposer_system_v2_6_7_1.txt` | `5d21ea8534d6aeae3f8f97caedc3ad1d60bc5dc5565caa6c19a535fc9667f207` |
| NeuronBench mechanism prompt `v3.0` | `prompts/proposer_system_v3_0_mechanism.txt` | `8d578e80b4d6cb83cc16eaeb2dd5681e179f3e3efff04c6ad0264babab140ef8` |
| paragraph of the initial batch | `prompts/slot_burst_paragraph_v1.txt` | `3155653c9968352ba8f5758ba39ac6a75e20b466139543cc9a1a5dd377638ba2` |
| paragraph of the top-up waves | `prompts/slot_burst_refill_paragraph_v1.txt` | `7581cd50c981e9ae83e8171e6b98c63f3fde88440ca96a5136af12446225ff82` |
| table of coefficient roles and priors | `prompts/slot_fixed_role_prior_v1.json` | `82a432fe3f7615ca5d1f76bbda27cc70d81c0801cc94f28397ca388aa27de984` |
| repair-turn text | `prompts/slot_repair_v1.txt` | `8a31a92c58f35c61e566eda7f83f1e9c27a2e7bb4fbe822644a69f62e6f937af` |
| repair-turn text after an empty reply | `prompts/slot_repair_v1_empty.txt` | `f6d969239867cffc98b06b165d0e15028caefddd415273ef6bc2ecc5d44c5260` |

The sine-and-cosine variant declares the same version tag, `v2.9.FA-C`, on its first line, so
only its sha256 tells the two files apart. The two paragraph digests are taken over the
constants' bytes with their `{limit}` placeholder unfilled, as notebook 02 computes them.

## Reproducing the benchmark runs

The launch kits and the exact engine state of the paper runs are preserved as tags in the
private development repository, which is not public (`paper-2026-10/chembench-u3`,
`paper-2026-10/boxinggym-u3`, `paper-2026-10/neuronbench-u3`, `paper-2026-10/forcebench-u3`), together
with the full development history. They are not part of this repository; requests to reproduce a
paper run go to the authors. What this repository provides is the identity of the code: every
Python file here has the same syntax tree as the development tree the runs executed, apart from the
string literals and the test-only adjustments declared in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
(comments and docstrings were rewritten in English for the release; the declared literals are listed
one by one in [`CHANGED_LITERALS.md`](CHANGED_LITERALS.md)). The per-run `config.json` records the
sha256 of every engine source file the run loaded
and of the prompt file it sent; those hashes are of the development tree's bytes, so against this
tree the prompt files match byte for byte while the source files match at the syntax-tree level
(`tools/release/ast_equiv.py`), not byte for byte.

To check a run of the synthetic count, ChemBench, BoxingGym or NeuronBench runner against a
checkout of the development tree, compare the recorded digests with its files (against this
public tree most source files print DIFFERENT, for the reason above):

```bash
python - <<'PY'
import hashlib, json, pathlib
config = json.loads(pathlib.Path("runs/<run>/config.json").read_text())
key = next(k for k in ("source_files_sha256", "source_sha256", "experimental_source_sha256") if k in config)
folder = pathlib.Path("src/bayes_model_discovery/experimental")
for name, digest in sorted(config[key].items()):
    same = hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    print(f"{name:32} {'same' if same else 'DIFFERENT'}")
PY
```

A ForceBench run file records the same for its runner's files as `code_sha256`. The settings of
the paper runs (rounds, seeds, arms, initial batch, archive review) are summarized in
[`FLOW.md`](FLOW.md#the-protocol-of-the-paper-runs).
