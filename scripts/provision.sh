#!/usr/bin/env bash
# Provision a clean Linux host for this repository. Costs nothing (no LLM calls) and is re-runnable:
# every step checks whether its result already exists before doing work.
#
# What it builds (all under ROOT, default = the repository checkout that contains this script):
#   ROOT/.venv                     Python 3.12 virtualenv with the pinned engine stack (requirements.lock)
#                                  and this package installed in editable mode
#   ROOT/external/boxing-gym       BoxingGym source at the pinned commit (sparse: src, conf)   [--with boxinggym]
#   ROOT/external/neuronbench      NeuronBench (MIT) at the pinned commit, installed into .venv [--with neuronbench]
#   ROOT/external/DiscoverPhysics  DiscoverPhysics (MIT) at the pinned commit + ROOT/.venv-forcebench
#                                  (jax stack, requirements-forcebench.lock)                    [--with forcebench]
#   (ChemBench is not enabled in this release: its dependency LLM-AutoSciLab states no licence, so this
#    script does not fetch it.)
#   ROOT/.bmd_env.sh               environment variables the run scripts read (source it before running)
#
# It never reads, prints or copies a .env file. Paid runs need ANTHROPIC_API_KEY in a .env file whose
# PATH you pass to the run scripts; mock runs (the default) need no key at all.
#
# Usage:  bash scripts/provision.sh [--root DIR] [--python 3.12.13] [--with boxinggym,neuronbench,forcebench] [--smoke]
set -uo pipefail

PYV=3.12.13
WITH=""
SMOKE=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --python) PYV="$2"; shift 2 ;;
    --with) WITH="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Pinned external sources. Change them only together with conftest.py (the tests check these commits).
BOXINGGYM_REPO=https://github.com/JooKyung-Yuh/BoxingGym.git
BOXINGGYM_TIP=06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba
NEURONBENCH_REPO=https://github.com/murphyk/neuronbench
NEURONBENCH_TIP=c354622458c460b419cab821d482c879f0578377
DISCOVERPHYSICS_REPO=https://github.com/SampsonML/DiscoverPhysics.git
DISCOVERPHYSICS_TIP=33b7fa9
FAC_PROMPT=tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt
FAC_SHA=196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c

VENV="$ROOT/.venv"
EXT="$ROOT/external"
fail=0
step() { echo "== $(date '+%F %T') $*"; }
ok() { echo "   ok: $*"; }
bad() { echo "   FAIL: $*"; fail=$((fail + 1)); }
want() { case ",$WITH," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

echo "PROVISION start root=$ROOT host=$(hostname) nproc=$(nproc 2>/dev/null || echo ?)"
mkdir -p "$EXT"

step "0 prerequisites (git, curl, a C++ compiler for PyTensor)"
command -v git >/dev/null 2>&1 && ok "$(git --version)" || bad "git missing"
command -v curl >/dev/null 2>&1 && ok "curl present" || bad "curl missing"
if g++ --version >/dev/null 2>&1; then ok "$(g++ --version | head -1)"; else
  bad "g++ missing: install build-essential (Debian/Ubuntu: sudo apt-get install build-essential). Without it PyTensor falls back to a Python backend that is many times slower."; fi

step "1 Python $PYV"
PY=""
if command -v "python$PYV" >/dev/null 2>&1; then PY="$(command -v "python$PYV")"; fi
if [ -z "$PY" ] && command -v python3.12 >/dev/null 2>&1 && python3.12 --version 2>&1 | grep -q "$PYV"; then PY="$(command -v python3.12)"; fi
if [ -z "$PY" ]; then
  UV="$HOME/.local/bin/uv"
  if [ ! -x "$UV" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || bad "uv install"; fi
  "$UV" python install "$PYV" >/dev/null 2>&1 || bad "uv python install $PYV"
  PY="$("$UV" python find "$PYV" 2>/dev/null)"
fi
[ -n "$PY" ] && [ -x "$PY" ] && ok "python $PY = $("$PY" --version 2>&1)" || bad "python $PYV not found"

step "2 engine virtualenv ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then "$PY" -m venv "$VENV" || bad "venv create"; fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip 2>&1 | tail -1
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements.lock" 2>&1 | tail -3
"$VENV/bin/python" -m pip install --quiet --no-deps -e "$ROOT" 2>&1 | tail -1
"$VENV/bin/python" - <<'PY' || bad "engine stack import or version check"
import sys
import pymc, pytensor, numpy, scipy, anthropic, arviz
got = dict(pymc=pymc.__version__, pytensor=pytensor.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
           anthropic=anthropic.__version__, arviz=arviz.__version__)
want = dict(pymc="5.13.1", pytensor="2.20.0", numpy="1.26.4", scipy="1.17.1", anthropic="0.120.2", arviz="0.23.4")
print("   versions", " ".join(f"{k}={v}" for k, v in got.items()))
print("   pytensor cxx", pytensor.config.cxx or "NONE (slow Python backend)")
mismatch = [k for k in want if got[k] != want[k]]
if mismatch:
    print("   MISMATCH", mismatch); sys.exit(1)
PY
"$VENV/bin/python" -c "import bayes_model_discovery, sys; print('   engine', bayes_model_discovery.__file__)" 2>&1 || bad "engine import"

step "3 pinned prompt file"
got="$(sha256sum "$ROOT/$FAC_PROMPT" 2>/dev/null | cut -d' ' -f1)"
[ "$got" = "$FAC_SHA" ] && ok "prompt sha $got" || bad "prompt $FAC_PROMPT sha $got != $FAC_SHA"

if want boxinggym; then
  step "4 BoxingGym source at $BOXINGGYM_TIP (sparse: src, conf)"
  d="$EXT/boxing-gym"
  if [ ! -d "$d/.git" ]; then git clone --filter=blob:none --no-checkout "$BOXINGGYM_REPO" "$d" >/dev/null 2>&1 || bad "boxing-gym clone"; fi
  git -C "$d" cat-file -e "$BOXINGGYM_TIP^{commit}" 2>/dev/null || git -C "$d" fetch --no-tags origin "$BOXINGGYM_TIP" >/dev/null 2>&1 || bad "boxing-gym tip not reachable"
  git -C "$d" sparse-checkout set --cone src conf >/dev/null 2>&1 || bad "sparse-checkout"
  git -C "$d" checkout --detach "$BOXINGGYM_TIP" >/dev/null 2>&1 || bad "boxing-gym checkout"
  test -d "$d/src/boxing_gym" && ok "boxing_gym at $(git -C "$d" rev-parse --short HEAD)" || bad "src/boxing_gym missing"
fi

if want neuronbench; then
  step "5 NeuronBench at $NEURONBENCH_TIP (MIT)"
  d="$EXT/neuronbench"
  if [ ! -d "$d/.git" ]; then git clone "$NEURONBENCH_REPO" "$d" >/dev/null 2>&1 || bad "neuronbench clone"; fi
  git -C "$d" checkout --detach "$NEURONBENCH_TIP" >/dev/null 2>&1 || bad "neuronbench checkout"
  "$VENV/bin/python" -m pip install --quiet --no-deps -e "$d" 2>&1 | tail -1
  "$VENV/bin/python" -c "import neuronbench" 2>&1 && ok "neuronbench importable" || bad "neuronbench import"
fi

if want forcebench; then
  step "6 ForceBench: DiscoverPhysics at $DISCOVERPHYSICS_TIP (MIT) + jax virtualenv"
  d="$EXT/DiscoverPhysics"
  if [ ! -d "$d/.git" ]; then git clone "$DISCOVERPHYSICS_REPO" "$d" >/dev/null 2>&1 || bad "DiscoverPhysics clone"; fi
  git -C "$d" checkout --detach "$DISCOVERPHYSICS_TIP" >/dev/null 2>&1 || bad "DiscoverPhysics checkout"
  FBV="$ROOT/.venv-forcebench"
  if [ ! -x "$FBV/bin/python" ]; then "$PY" -m venv "$FBV" || bad "forcebench venv create"; fi
  "$FBV/bin/python" -m pip install --quiet --upgrade pip 2>&1 | tail -1
  "$FBV/bin/python" -m pip install --quiet -r "$ROOT/requirements-forcebench.lock" 2>&1 | tail -3
  "$FBV/bin/python" -c "import jax, numpy; print('   jax', jax.__version__, 'numpy', numpy.__version__)" 2>&1 && ok "jax stack" || bad "jax import"
fi

step "7 environment file ($ROOT/.bmd_env.sh)"
cat > "$ROOT/.bmd_env.sh" <<EOF2
# Generated by scripts/provision.sh. Source this file before running the benchmark scripts.
export BMD_ROOT="$ROOT"
export BMD_PY="$VENV/bin/python"
export BMD_FAC_PROMPT="$ROOT/$FAC_PROMPT"
export BMD_BOXINGGYM_SRC="$EXT/boxing-gym/src"
export NEURONBENCH_SRC="$EXT/neuronbench"
export FB_PY="$ROOT/.venv-forcebench/bin/python"
export FB_DP="$EXT/DiscoverPhysics"
export PYTENSOR_FLAGS="base_compiledir=$ROOT/.pytensor"
EOF2
ok "written (paths for benchmarks you did not request are placeholders)"

if [ "$SMOKE" = 1 ]; then
  step "8 smoke: synthetic count benchmark, mock proposer, one round (no key, no network)"
  out="$ROOT/.smoke_$(date +%s)"
  PYTENSOR_FLAGS="base_compiledir=$ROOT/.pytensor" "$VENV/bin/python" -m bayes_model_discovery.experimental.core_count \
    --output "$out" --profile dev --proposer mock --rounds 1 --seed 11 --cell 111 --structure out >"$out.log" 2>&1 \
    && ok "mock run finished: $out (log $out.log)" || bad "mock run failed, see $out.log"
fi

echo "SUMMARY fail=$fail root=$ROOT $(date '+%F %T')"
exit $fail
