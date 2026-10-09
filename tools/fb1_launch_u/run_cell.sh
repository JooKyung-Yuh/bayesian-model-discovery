#!/bin/bash
# ForceBench under the unified protocol (the registered protocol and its two addenda), one cell, host-independent.
# The option C/D cell settings of ../fb1_launch_c/run_cell.sh (Claude Opus 5.5, effort medium, adaptive thinking,
# summarized; FA-C system prompt ../fb1_launch_c/proposer_system_v2_9_FA-C.txt, sha256
# 196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c; masked input names; free structures; 2 random start
# designs; development label) plus the unified protocol's options of the runner (the four engines' common names):
#   FB1_INITIAL_PROPOSALS=<FB_BURST, default 12>  FB1_ACTIVE_MAXIMUM=12   round-0 burst (wave 0 + at most two refill waves
#                                                  asking only the shortfall), no catalogue pool
#   FB1_PROPOSAL_LIMIT=3                           one call per round answering three candidates (COMPOSITION v2)
#   FB1_ARCHIVE_REVIEW_ROUNDS=5,10,15              ARCHIVE_REVIEW v1 rounds (16 rounds; the launch decision)
#   FB1_DESIGN_POLICY=<policy>                     joint_eig (the pre-registered headline arm) | model_eig | parameter_eig |
#                                                  epig | random
#   FB1_CATALOGUE=${FB_CATALOGUE:-off}             off (the launch decision of 2026-10-06: the packets show no
#                                                  trace of the law catalogue) | on (the packets of every run since 9/8)
#   FB1_PACKET_ARCHIVE=${FB_PACKET_ARCHIVE:-full}  full (the launch decision of 2026-10-06: give the model as much
#                                                  information as possible) | compact (opt-in: archive rows without the
#                                                  prior_probe and numerical_equivalence blocks)
# 16 designed experiments after D0 (FB1_T_EXTRA=16). FB_BUDGET=30 keeps the 30-round variant reachable (review rounds
# 5,10,15,20,25,30, FB1_CHECKPOINTS=2,4,8,16,18 so n = 18 is scored too); it is not the kit's default.
# The runner is jax only (no PyTensor module is loaded); PYTENSOR_FLAGS=base_compiledir=<out>/ptc is set anyway, as in
# the other engines' kits, so a future PyTensor import could not share a compile folder between cells.
# The policy sets the arm (joint/model/parameter EIG 111, epig 112, random 110) and the search (Gradient Ascent for the
# four active rules, the runner's random draw for random). The engine follows option C/D: matched (the simulator's own
# integrator, dt and softening) for coulomb_easy, oscillator and extra_dimensions, legacy for gravity and yukawa.
# Usage: run_cell.sh <world> <seed> <policy> <new output folder> <paid|mock> [OPTS]
# OPTS: comma-separated key=value - grammar_trig=1 (FB1_GRAMMAR_TRIG), prompt=<file in ../fb1_launch_c> (replaces FA-C),
# epig_targets=score_q (FB1_EPIG_TARGETS; epig only), engine=legacy|matched (overrides the world's engine). The campaign plans
# give the oscillator lines grammar_trig=1,prompt=proposer_system_v2_9_FA-C_trig.txt (the round D3 condition; a
# decision of 2026-10-06, a disclosed per-world exception to "FA-C bytes unchanged").
# mock = 0 USD wiring check: no key, mock transport, FB1_T_EXTRA=${FB_MOCK_T_EXTRA:-2}, archive review at
# ${FB_MOCK_REVIEW:-2}.
# Needs FB_PY (python with jax 0.11.1), FB_DP (DiscoverPhysics checkout at 33b7fa9), and in paid mode FB_DOTENV (the file
# holding ANTHROPIC_API_KEY; read only inside the relay process, never printed).
set -uo pipefail
WORLD=$1; SEED=$2; POLICY=$3; OUT=$4; MODE=${5:?paid or mock}; OPTS=${6:-}
case "$POLICY" in
  joint_eig|model_eig|parameter_eig) ARM=111; SEARCH=ascent;;
  epig) ARM=112; SEARCH=ascent;;
  random) ARM=110; SEARCH=random64;;
  *) echo "policy must be joint_eig, model_eig, parameter_eig, epig or random"; exit 2;;
esac
case "$WORLD" in
  coulomb_easy|oscillator|extra_dimensions) ENGINE=matched;;
  gravity|yukawa) ENGINE=legacy;;
  *) echo "world must be gravity, yukawa, coulomb_easy, oscillator or extra_dimensions"; exit 2;;
esac
BUDGET=${FB_BUDGET:-16}
case "$BUDGET" in
  16) CALL_CAP=48; CHECKPOINTS=default; REVIEW=5,10,15;;
  30) CALL_CAP=64; CHECKPOINTS=2,4,8,16,18; REVIEW=5,10,15,20,25,30;;
  *) echo "FB_BUDGET must be 16 (default) or 30"; exit 2;;
esac
[ -n "${FB_CHECKPOINTS:-}" ] && CHECKPOINTS=$FB_CHECKPOINTS   # 2026-10-06 (main): env passthrough like FB_MAX_PAID_CALLS; "all" = every n from D0 to D0+T_EXTRA (runner parser), used by TRIG75B so Ours has per-round scores
CATALOGUE=${FB_CATALOGUE:-off}
case "$CATALOGUE" in on|off) ;; *) echo "FB_CATALOGUE must be on or off"; exit 2;; esac
case "$MODE" in paid|mock) ;; *) echo "mode must be paid or mock"; exit 2;; esac
KIT=$(cd "$(dirname "$0")" && pwd)
CKIT=$(cd "$KIT/../fb1_launch_c" && pwd)
TREE=$(cd "$KIT/../fb1_snapshot" && pwd)
: "${FB_PY:?FB_PY must point at the python with jax 0.11.1}"
: "${FB_DP:?FB_DP must point at the DiscoverPhysics checkout (33b7fa9)}"
test ! -e "$OUT" || { echo "exists: $OUT"; exit 2; }
mkdir -p "$OUT"
unset FORCE_COLOR NPY_DISABLE_CPU_FEATURES FB1_DESIGN_SEARCH FB1_INTEGRATOR FB1_SOFTENING ANTHROPIC_API_KEY FB1_ALLOW_PAID_API \
      FB1_EPIG_TARGETS FB1_GRAMMAR_TRIG FB1_PROPOSAL_POLICY FB1_INITIAL_PROPOSALS FB1_ACTIVE_MAXIMUM FB1_PROPOSAL_LIMIT \
      FB1_ARCHIVE_REVIEW_ROUNDS FB1_DESIGN_POLICY FB1_PACKET_ARCHIVE FB1_CATALOGUE FB1_CHECKPOINTS FB1_MAX_PAID_CALLS
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TZ=Asia/Seoul
mkdir -p "$OUT/ptc" && export PYTENSOR_FLAGS=base_compiledir=$OUT/ptc
export FB1_WORLD=$WORLD FB1_ARM=$ARM FB1_SEED=$SEED FB1_LIST=v3 FB1_PROPOSER=llm FB1_STRUCTURE_PROPOSALS=free
[ "$SEARCH" = ascent ] && export FB1_DESIGN_SEARCH=ascent
export FB1_INITIAL_PROPOSALS=${FB_BURST:-12} FB1_ACTIVE_MAXIMUM=12 FB1_PROPOSAL_LIMIT=3 FB1_DESIGN_POLICY=$POLICY
export FB1_ARCHIVE_REVIEW_ROUNDS=$REVIEW FB1_CATALOGUE=$CATALOGUE FB1_PACKET_ARCHIVE=${FB_PACKET_ARCHIVE:-full}
export DP_SCIENCEAGENT_DIR=$FB_DP/ScienceAgent
export FB1_LLM_PROVIDER=anthropic FB1_LLM_MODEL=claude-opus-5-5
export FB1_LLM_REASONING=effort=medium,thinking=adaptive,display=summarized FB1_LLM_MAX_TOKENS=16000
export FB1_PROMPT_VARIANT=$CKIT/proposer_system_v2_9_FA-C.txt
IFS=, read -r -a OPTV <<< "$OPTS"
for kv in "${OPTV[@]}"; do
  [ -z "$kv" ] && continue
  case "$kv" in
    epig_targets=*) export FB1_EPIG_TARGETS=${kv#epig_targets=};;
    grammar_trig=*) export FB1_GRAMMAR_TRIG=${kv#grammar_trig=};;
    prompt=*) p=$CKIT/${kv#prompt=}; test -f "$p" || { echo "no prompt file $p"; exit 2; }; export FB1_PROMPT_VARIANT=$p;;
    engine=*) ENGINE=${kv#engine=}; case "$ENGINE" in legacy|matched) ;; *) echo "engine must be legacy or matched"; exit 2;; esac;;
    *) echo "unknown option $kv"; exit 2;;
  esac
done
if [ "$ENGINE" = matched ]; then export FB1_INTEGRATOR=match_truth FB1_SOFTENING=world; fi
[ "$CHECKPOINTS" != default ] && export FB1_CHECKPOINTS=$CHECKPOINTS
export FB1_MASK_INPUT_NAMES=1 FB1_RUN_LABEL=development FB1_OUT_DIR=$OUT/
if [ "$MODE" = mock ]; then
  export FB1_T_EXTRA=${FB_MOCK_T_EXTRA:-2} FB1_ARCHIVE_REVIEW_ROUNDS=${FB_MOCK_REVIEW:-2}
  [ "$CHECKPOINTS" != default ] && [ "$CHECKPOINTS" != all ] && unset FB1_CHECKPOINTS   # the 30-round list is outside a short mock's range; "all" adapts to it
else
  : "${FB_DOTENV:?FB_DOTENV must point at the dotenv file holding ANTHROPIC_API_KEY}"
  export FB1_ALLOW_PAID_API=1 FB1_T_EXTRA=$BUDGET FB1_MAX_PAID_CALLS=${FB_MAX_PAID_CALLS:-$CALL_CAP}
fi
echo "START $(date '+%F %T KST') world $WORLD seed $SEED policy $POLICY arm $ARM search $SEARCH engine $ENGINE budget $BUDGET mode $MODE host $(hostname) tree $TREE out $OUT"
echo "ENV FB1_INITIAL_PROPOSALS=$FB1_INITIAL_PROPOSALS FB1_PROPOSAL_LIMIT=$FB1_PROPOSAL_LIMIT FB1_ARCHIVE_REVIEW_ROUNDS=$FB1_ARCHIVE_REVIEW_ROUNDS FB1_CATALOGUE=$FB1_CATALOGUE FB1_PACKET_ARCHIVE=$FB1_PACKET_ARCHIVE FB1_DESIGN_POLICY=$FB1_DESIGN_POLICY FB1_T_EXTRA=$FB1_T_EXTRA FB1_MAX_PAID_CALLS=${FB1_MAX_PAID_CALLS:-<unset>} FB1_CHECKPOINTS=${FB1_CHECKPOINTS:-<unset>} FB1_DESIGN_SEARCH=${FB1_DESIGN_SEARCH:-<unset>} FB1_INTEGRATOR=${FB1_INTEGRATOR:-<unset>} FB1_SOFTENING=${FB1_SOFTENING:-<unset>} FB1_EPIG_TARGETS=${FB1_EPIG_TARGETS:-<unset>} FB1_GRAMMAR_TRIG=${FB1_GRAMMAR_TRIG:-<unset>} prompt_file=$(basename "$FB1_PROMPT_VARIANT") prompt_sha256=$(sha256sum "$FB1_PROMPT_VARIANT" | cut -d' ' -f1)"
FB_TREE=$TREE "$FB_PY" -B "$CKIT/relay.py"
rc=$?
echo "EXIT=$rc END $(date '+%F %T KST')"
exit $rc
