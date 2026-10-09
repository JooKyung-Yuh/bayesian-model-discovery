#!/usr/bin/env bash
# Release checks for the public tree: Korean text, internal code names, private paths and host names, lint, and a pytest
# subset. Prints one line per check and a PASS/FAIL summary; exit status 1 when any check fails.
#
#   tools/release/release_checks.sh [tree] [--python PY] [--no-pytest] [--pytest-args "..."]
#
# tree defaults to the repository that holds this script. The code-name patterns live in tools/release/code_names.txt
# (grep -P syntax). Files whose bytes must stay as recorded are exempt from the text checks: list them in
# tools/release/text_check_exempt.txt, one path per line, with the reason after a tab.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python3}
RUN_PYTEST=1
PYTEST_ARGS='-q -rs -m "not heavy and not replay"'
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY=$2; shift 2;;
    --no-pytest) RUN_PYTEST=0; shift;;
    --pytest-args) PYTEST_ARGS=$2; shift 2;;
    -*) echo "unknown option $1"; exit 2;;
    *) TREE=$(cd "$1" && pwd); shift;;
  esac
done
cd "$TREE" || exit 2
fail=0
status() { if [ "$2" -eq 0 ]; then echo "PASS  $1"; else echo "FAIL  $1 ($3)"; fail=1; fi; }

# the files to check: in a git repository every tracked or untracked file that is not ignored (a file that is
# not committed yet is still released by the next commit), else every regular file
if git -C "$TREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$TREE" ls-files -z --cached --others --exclude-standard > /tmp/release_files.$$
else
  find . -type f -not -path './.git/*' -print0 | sed -z 's#^\./##' > /tmp/release_files.$$
fi
EXEMPT=$(test -f tools/release/text_check_exempt.txt && grep -v '^#' tools/release/text_check_exempt.txt | cut -f1 | grep -v '^$' || true)
filtered() {  # the tracked files minus the exempt list, NUL separated
  if [ -n "$EXEMPT" ]; then
    tr '\0' '\n' < /tmp/release_files.$$ | grep -vxF -f <(printf '%s\n' "$EXEMPT") | tr '\n' '\0'
  else
    cat /tmp/release_files.$$
  fi
}

# 1. Korean text (Hangul syllables and jamo)
n=$(filtered | xargs -0 grep -lIP '[\x{AC00}-\x{D7A3}\x{1100}-\x{11FF}\x{3130}-\x{318F}]' 2>/dev/null | wc -l)
status "no Korean text (files with Hangul: $n)" "$n" "$(filtered | xargs -0 grep -lIP '[\x{AC00}-\x{D7A3}]' 2>/dev/null | head -n 5 | tr '\n' ' ')"

# 2. internal code names
hits=0
while IFS=$'\t' read -r pattern meaning; do
  [ -z "$pattern" ] && continue
  case "$pattern" in \#*) continue;; esac
  c=$(filtered | xargs -0 grep -cIP -- "$pattern" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
  if [ "$c" -gt 0 ]; then echo "      code name $pattern ($meaning): $c lines"; hits=$((hits + c)); fi
done < tools/release/code_names.txt
status "no internal code names (lines: $hits)" "$hits" "see the lines above"

# 3. private paths, host names, personal mail hosts
pat='/home/[a-z]+/|/Users/[a-z]+/|/tmp/claude|bai-ada|bai-a6000|elice[1-5]\b|@gmail\.com'
n=$(filtered | xargs -0 grep -lIE -- "$pat" 2>/dev/null | wc -l)
status "no private paths or host names (files: $n)" "$n" "$(filtered | xargs -0 grep -lE -- "$pat" 2>/dev/null | head -n 5 | tr '\n' ' ')"

# 4. secrets
# key-shaped strings; the tests' obviously fake keys (FAKE, test, synthetic, example, dummy, placeholder in the string) are not keys
keypat='sk-ant-[A-Za-z0-9_-]{20,}|sk-(proj-)?[A-Za-z0-9_-]{32,}|(ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'
keyfiles=$(filtered | xargs -0 grep -oHE -- "$keypat" 2>/dev/null | grep -viE 'fake|test|synthetic|example|dummy|placeholder|abcdef' | cut -d: -f1 | sort -u)
n=$(printf '%s' "$keyfiles" | grep -c . )
status "no key-shaped strings (files: $n)" "$n" "$(printf '%s' "$keyfiles" | head -n 5 | tr '\n' ' ')"

# 4b. every Python file a shell script runs must exist in the tree (a reachability pass over Python imports does not see
# `"$FB_PY" "$CKIT/relay.py"`; the kit variables are resolved by name: CKIT -> tools/fb1_launch_c, TREE and KIT -> tools/fb1_snapshot)
missing=""
for sh in $(filtered | tr '\0' '\n' | grep '\.sh$'); do
  d=$(dirname "$sh")
  for ref in $(grep -oE '[A-Za-z0-9_./${}-]+\.py\b' "$sh" | sort -u); do
    r=${ref//\$CKIT/tools/fb1_launch_c}; r=${r//\$\{CKIT\}/tools/fb1_launch_c}
    r=${r//\$TREE/tools/fb1_snapshot}; r=${r//\$\{TREE\}/tools/fb1_snapshot}
    r=${r//\$KIT/tools/fb1_snapshot}; r=${r//\$\{KIT\}/tools/fb1_snapshot}
    r=${r#./}
    case "$r" in *\$*) continue;; esac   # a variable this check does not know: not resolvable here
    if [ -e "$r" ] || [ -e "$d/$r" ] || [ -e "tools/fb1_snapshot/$r" ] || [ -e "tools/fb1_launch_c/$r" ]; then :; else missing="$missing $sh->$ref"; fi
  done
done
n=$(printf '%s' "$missing" | wc -w)
status "shell scripts reference only Python files present in the tree (missing: $n)" "$n" "$missing"

# 5. lint (ruff on PATH, or the one installed beside the chosen python)
RUFF=""
if command -v ruff >/dev/null 2>&1; then RUFF=ruff; elif [ -x "$(dirname "$PY")/ruff" ]; then RUFF="$(dirname "$PY")/ruff"; fi
if [ -n "$RUFF" ]; then
  "$RUFF" check src tests scripts tools/release > /tmp/release_ruff.$$ 2>&1; rc=$?
  status "ruff check src tests scripts tools/release" "$rc" "$(tail -n 1 /tmp/release_ruff.$$)"
else
  echo "SKIP  ruff (not installed)"
fi

# 6. ForceBench kit checksums
if [ -f tools/fb1_snapshot/SHA256SUMS ]; then
  (cd tools/fb1_snapshot && sha256sum -c --quiet SHA256SUMS > /tmp/release_sums.$$ 2>&1); rc=$?
  status "tools/fb1_snapshot/SHA256SUMS verifies" "$rc" "$(head -n 2 /tmp/release_sums.$$ | tr '\n' ' ')"
fi

# 7. pytest
if [ "$RUN_PYTEST" -eq 1 ]; then
  eval "PYTHONPATH=src \"$PY\" -m pytest $PYTEST_ARGS" > /tmp/release_pytest.$$ 2>&1; rc=$?
  status "pytest $PYTEST_ARGS" "$rc" "$(tail -n 1 /tmp/release_pytest.$$)"
  tail -n 1 /tmp/release_pytest.$$
fi
rm -f /tmp/release_files.$$ /tmp/release_ruff.$$ /tmp/release_sums.$$ /tmp/release_pytest.$$
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
