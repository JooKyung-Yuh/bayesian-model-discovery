# Requirements

What the code needs to run: the platform, two Python environments, a compiler, the external
benchmark code, the environment variables that point to it, and what to expect in disk and time.
On a clean Linux host `scripts/provision.sh` builds both environments and fetches the external
code (it checks for the compiler but does not install it); [`USAGE.md`](USAGE.md) shows how to
use it.

## Platform

- Linux on x86-64. The lock files are resolved for Python 3.12 on that platform.
- Python 3.12 (`pyproject.toml` requires `>=3.12`). `scripts/provision.sh` uses 3.12.13 by
  default and installs it with `uv` when no matching interpreter is found.
- git, and curl for the provisioning script.
- A C/C++ compiler such as `g++` (Debian and Ubuntu: `sudo apt-get install build-essential`).
  PyTensor, which PyMC uses, compiles C code the first time it builds a model; without a
  compiler it falls back to a Python backend that is many times slower.

## Two Python environments

The engine and the ForceBench runner cannot share one environment: the engine is pinned to
numpy 1.26.4 for PyMC 5.13.1, while jax 0.11.1 needs numpy 2.x. Keep them separate.

| Environment | Lock file | Key versions | Used by |
| --- | --- | --- | --- |
| engine (`.venv`) | `requirements.lock` | numpy 1.26.4, scipy 1.17.1, pymc 5.13.1, pytensor 2.20.0, anthropic 0.120.2 | the package, the four runners in `src/`, `scripts/bmd_run.py`, the tests and the notebooks |
| ForceBench (`.venv-forcebench`) | `requirements-forcebench.lock` | jax 0.11.1, jaxlib 0.11.1, numpy 2.5.2, scipy 1.18.1, anthropic 0.120.2, pytest 9.1.1 | `tools/fb1_snapshot/run_gravity.py` and its tests |

- `requirements.lock` is a resolution lock: it pins the version of every transitive dependency,
  but not artifact hashes. Install against it with
  `pip install -c requirements.lock -e ".[dev]"` (the documented install). The `dev` extra adds
  pytest and ruff 0.16.1, the version CI lints with; the lock does not pin them.
- `scripts/provision.sh` installs the engine environment with `pip install -r requirements.lock`
  and the package with `pip install --no-deps -e .`, then checks that pymc, pytensor, numpy,
  scipy, anthropic and arviz have the pinned versions and counts a mismatch as a failed step. It
  installs pytest and ruff 0.16.1 only with `--with tests`.
- The ForceBench lock records the environment that produced the ForceBench results (Python
  3.12.13). Create it with:

  ```bash
  python3.12 -m venv .venv-forcebench
  .venv-forcebench/bin/pip install -r requirements-forcebench.lock
  ```

- Jupyter is not part of either lock; install it into the engine environment to run the
  notebooks (`pip install jupyterlab`).

## External benchmark code

The synthetic count benchmark needs nothing else. Three benchmarks read an external repository
at a pinned commit. These repositories are not redistributed here, pip does not install them,
and the run script and tests find them through environment variables.

| Benchmark | Repository | Pinned commit | License | Variable |
| --- | --- | --- | --- | --- |
| BoxingGym | https://github.com/kanishkg/boxing-gym | `b43e38cb03d09c13efa9cf4d9bae740d51157bfd` | MIT | `BMD_BOXINGGYM_SRC`: the checkout's `src/` folder |
| NeuronBench | https://github.com/murphyk/neuronbench | `c354622458c460b419cab821d482c879f0578377` | MIT | `NEURONBENCH_SRC`: the checkout |
| ForceBench | DiscoverPhysics, https://github.com/SampsonML/DiscoverPhysics | `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5` | MIT | `FB_DP`: the checkout; `FB_PY`: the ForceBench environment's python |

- The paper runs used a fork of BoxingGym at commit `06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba`, which
  changes four files under `src/boxing_gym/agents/` relative to upstream commit
  `b43e38cb03d09c13efa9cf4d9bae740d51157bfd`. The environments the runner imports
  (`src/boxing_gym/envs/`) are identical in both, and the only thing they import from `agents` is
  `construct_dataframe`, which the fork leaves unchanged, so this release pins the public upstream
  commit. The tests accept only that commit. `scripts/provision.sh` reads `BMD_BOXINGGYM_REPO` and
  `BMD_BOXINGGYM_COMMIT` to fetch from another location.
- **ChemBench is not enabled in this release.** Its true laws and input ranges come from
  LLM-AutoSciLab (https://github.com/scientific-discovery/LLM-AutoSciLab, commit
  `9a37255ff5432a7155b6cde8985e9654f2f8f272`), which states no license. The ChemBench code
  stays in the tree, but the run script refuses it, the provisioning script does not fetch the
  dependency, and its tests skip.

## Environment variables

`bash scripts/provision.sh --with boxinggym,neuronbench,forcebench` writes `.bmd_env.sh` at the
repository root; source it before running anything. Paths for benchmarks you did not request
are placeholders.

| Variable | Meaning | Read by |
| --- | --- | --- |
| `BMD_PY` | the engine environment's python (`.venv/bin/python`) | you, to run the scripts with the right interpreter |
| `BMD_BOXINGGYM_SRC` | the `src/` folder of the BoxingGym checkout | `scripts/bmd_run.py` (adds it to `PYTHONPATH`), the BoxingGym tests |
| `NEURONBENCH_SRC` | the neuronbench checkout | `scripts/bmd_run.py`, the NeuronBench tests |
| `FB_PY` | the ForceBench environment's python | `scripts/bmd_run.py` and the ForceBench launch script |
| `FB_DP` | the DiscoverPhysics checkout | `scripts/bmd_run.py` and the ForceBench launch script, which passes `$FB_DP/ScienceAgent` to the runner |
| `BMD_FAC_PROMPT` | the paper's system prompt, `tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt` | a protocol test that checks the prompt's digests when the variable is set |
| `BMD_ROOT` | the repository root | convenience |
| `PYTENSOR_FLAGS` | `base_compiledir=<root>/.pytensor`, the PyTensor compile cache | PyTensor |
| `NPY_DISABLE_CPU_FEATURES` | not set by the script. On a CPU with AVX-512, `NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"` turns numpy's SIMD power dispatch off; the NeuronBench batch forward backend (`--forward-backend batch`) needs it, and the batch-backend bit-identity tests skip without it | numpy, at import |

API keys are read only from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); mock
runs need none. A paid run takes the path of a dotenv file that holds the key; see
[`USAGE.md`](USAGE.md#paid-runs).

## What the tests need

| Tier | Needs |
| --- | --- |
| fast (`tests/`) | the engine environment with pytest |
| standard (everything CI runs) | the same; tests that need an external checkout skip unless its variable is set, and fail if the checkout is not clean, not at the pinned commit, or not the copy on `PYTHONPATH`. Tests that only import `neuronbench` run whenever it is importable, which `provision.sh --with neuronbench` makes it. On an AVX-512 CPU the batch-backend bit-identity tests skip unless `NPY_DISABLE_CPU_FEATURES` is set as above |
| heavy (re-runs sequential Monte Carlo (SMC) fits against frozen runs) | the full pinned stack of `requirements.lock`; the comparisons depend on the BLAS the wheels use |
| ForceBench tests (`tools/fb1_snapshot/test_*.py`) | the ForceBench environment and DiscoverPhysics (`FB1_PYTHON`, `DP_SCIENCEAGENT_DIR`) |

Every variable a test reads is listed in [`USAGE.md`](USAGE.md#tests); every skip names the input it needs.

## Disk and time

- PyTensor compiles C code the first time it builds each model and keeps it in a compile cache
  (by default under your home directory). The first run on a machine is therefore the slowest:
  notebook 01, which runs the quickstart, takes about four minutes the first time and about one
  minute afterwards.
- The three-round quickstart writes a run folder of under 1 MB and a compile cache of about
  12 MB (measured with an empty cache).
- Give every concurrent run its own cache (`PYTENSOR_FLAGS=base_compiledir=<folder>`): a shared
  compile folder slows fits down badly. `scripts/bmd_run.py` puts the cache beside the run folder
  (`<output>.ptc`) when `PYTENSOR_FLAGS` is not set.
- The `paper` profile (500 particles × 2 chains, against 64 × 2 in `dev`) is much slower than
  `dev`.
- ForceBench needs several gigabytes of memory per run, and about 11 GB for an expected predictive information gain (EPIG) run on the
  gravity world; plan parallel runs by memory.
- Test tiers (`pytest -m "not heavy and not replay"`):
  - fast (`tests/`): 259 tests, about one minute;
  - standard: 3,739 tests collected, 1 of them deselected (`not replay`). On one 16-thread machine with no benchmark
    source and no CPU policy (an AVX-512 host): 3,310 passed and 428 skipped in about 50 minutes. Every skip names
    its input: the BoxingGym source (214), the ChemBench oracle (104), neuronbench (63 not importable, 22 without
    `NEURONBENCH_SRC`), the CPU policy (24) and `BMD_FAC_PROMPT` (1). CI runs the same command without any benchmark
    source; on a runner without AVX-512 the 24 policy tests run instead of skipping. After `provision.sh --with
    neuronbench` the tests that only import neuronbench run as well;
  - ForceBench kit (`tools/fb1_snapshot/`, the ForceBench environment): 405 tests.

  Give every test run its own `--basetemp`: test temporaries can fill a small disk.
- A one-round NeuronBench mock run in the `dev` profile spends most of its time in the round-1
  design search. It took more than an hour at one core on a busy host, so allow an hour or more.
- Importing PyTensor may print `WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS
  functions.` This is harmless: it reports which BLAS path PyTensor found.
