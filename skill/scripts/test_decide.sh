#!/usr/bin/env bash
# Contract tests for the agent-decision-maker skill.
# Exercises the 8 fixture classes the SKILL.md documents.
# Usage: bash scripts/test_decide.sh   (requires the decision server + python3)
set -u

cd "$(dirname "$0")/.." || exit 1
PY=python3
[ -n "${HS_DECIDE_API_KEY:-}" ] || export HS_DECIDE_API_KEY=local-dev-key
pass=0 fail=0

run() {  # run <desc> <expected_exit> <json-or-STDIN>
  local desc="$1" want="$2"; shift 2
  local out got
  out="$(printf '%s' "$1" | "$PY" scripts/decide.py 2>&1)"
  got=$?
  if [ "$got" = "$want" ]; then
    printf '  PASS  %s\n' "$desc"; pass=$((pass+1))
  else
    printf '  FAIL  %s (exit %s, want %s)\n    %s\n' "$desc" "$got" "$want" "$out"; fail=$((fail+1))
  fi
}

echo "== agent-decision-maker contract tests =="

# 1. one successful request per type
run "choice 200" 0 '{"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"one","b":"two"}}}}'
run "score 200"  0 '{"state":"s","questions":{"q":{"type":"score","criteria":["low","mid","high"]}}}'
run "noul 200"   0 '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"ignore","false":"keep"}}}}'

# 2. incorrect noul usage (yes/no framed as noul) - still 200, just illustrative
run "noul-misuse (illustrative)" 0 '{"state":"is this correct?","questions":{"q":{"type":"noul","criteria":{"true":"yes","false":"no"}}}}'

# 3. invalid criteria / empty questions -> 422
run "choice with 1 option -> 422" 2 '{"state":"s","questions":{"q":{"type":"choice","criteria":{"only":"one"}}}}'
run "empty questions -> 422" 2 '{"state":"s","questions":{}}'

# 4. auth failure -> 403 (exit 2)
printf '%s' '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"x","false":"y"}}}}' | HS_DECIDE_API_KEY=WRONG "$PY" scripts/decide.py >/dev/null 2>&1
got=$?; if [ "$got" = 2 ]; then printf '  PASS  bad key -> 403\n'; pass=$((pass+1)); else printf '  FAIL  bad key -> 403 (exit %s)\n' "$got"; fail=$((fail+1)); fi

# 5. connection refused -> exit 1
printf '%s' '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"x","false":"y"}}}}' | HS_DECIDE_BASE_URL=http://127.0.0.1:59999 "$PY" scripts/decide.py >/dev/null 2>&1
got=$?; if [ "$got" = 1 ]; then printf '  PASS  conn refused -> exit 1\n'; pass=$((pass+1)); else printf '  FAIL  conn refused -> exit 1 (exit %s)\n' "$got"; fail=$((fail+1)); fi

# 7. determinism: two identical calls -> identical probabilities
A="$(printf '%s' '{"model":"hs-decide-1-flash","state":"det","questions":{"q":{"type":"choice","criteria":{"a":"1","b":"2","c":"3"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
B="$(printf '%s' '{"model":"hs-decide-1-flash","state":"det","questions":{"q":{"type":"choice","criteria":{"a":"1","b":"2","c":"3"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
if [ -n "$A" ] && [ "$A" = "$B" ]; then
  printf '  PASS  determinism (identical outputs)\n'; pass=$((pass+1))
else
  printf '  FAIL  determinism\n    A=%s\n    B=%s\n' "$A" "$B"; fail=$((fail+1))
fi

echo ""
echo "== result: ${pass} passed, ${fail} failed =="
[ "$fail" = 0 ]