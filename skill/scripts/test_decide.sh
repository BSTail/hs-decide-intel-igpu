#!/usr/bin/env bash
# Contract tests for the agent-decision-maker skill.
# Exercises the fixture classes the SKILL.md documents; needs python3. Fixtures
# 1-3, 8, 9 hit the live decision server; the mock fixtures are self-contained.
# Usage: bash scripts/test_decide.sh
set -u

cd "$(dirname "$0")/.." || exit 1
PY=python3
[ -n "${HS_DECIDE_API_KEY:-}" ] || export HS_DECIDE_API_KEY=local-dev-key
pass=0 fail=0

# run <desc> <expected_exit> <json>  -> asserts exit code only
run() {
  local desc="$1" want="$2" body="$3" out got
  out="$(printf '%s' "$body" | "$PY" scripts/decide.py 2>&1)"
  got=$?
  if [ "$got" = "$want" ]; then
    printf '  PASS  %s\n' "$desc"; pass=$((pass+1))
  else
    printf '  FAIL  %s (exit %s, want %s)\n    %s\n' "$desc" "$got" "$want" "$out"; fail=$((fail+1))
  fi
}

# run_err <desc> <expected_exit> <grep-pattern> <json>  -> asserts exit + stderr match
run_err() {
  local desc="$1" want="$2" pat="$3" body="$4" out got
  out="$(printf '%s' "$body" | "$PY" scripts/decide.py 2>&1)"
  got=$?
  if [ "$got" = "$want" ] && printf '%s' "$out" | grep -q "$pat"; then
    printf '  PASS  %s\n' "$desc"; pass=$((pass+1))
  else
    printf '  FAIL  %s (exit %s, want %s + grep '%s')\n    %s\n' "$desc" "$got" "$want" "$pat" "$out"; fail=$((fail+1))
  fi
}

echo "== agent-decision-maker contract tests =="

# 1. one successful request per type
run "choice 200" 0 '{"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"one","b":"two"}}}}'
run "score 200"  0 '{"state":"s","questions":{"q":{"type":"score","criteria":["low","mid","high"]}}}'
run "noul 200"   0 '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"ignore","false":"keep"}}}}'

# 2. incorrect noul usage (yes/no framed as noul) - still 200, illustrative
run "noul-misuse (illustrative)" 0 '{"state":"is this correct?","questions":{"q":{"type":"noul","criteria":{"true":"yes","false":"no"}}}}'

# 3. invalid criteria / empty questions -> 422 (assert the actual status text)
run_err "choice with 1 option -> 422" 2 "422" '{"state":"s","questions":{"q":{"type":"choice","criteria":{"only":"one"}}}}'
run_err "empty questions -> 422" 2 "422" '{"state":"s","questions":{}}'

# 4. auth failure -> 403 (assert the actual status text)
out="$(printf '%s' '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"x","false":"y"}}}}' | HS_DECIDE_API_KEY=WRONG "$PY" scripts/decide.py 2>&1)"
got=$?
if [ "$got" = 2 ] && printf '%s' "$out" | grep -q '403'; then
  printf '  PASS  bad key -> 403\n'; pass=$((pass+1))
else
  printf '  FAIL  bad key -> 403 (exit %s)\n    %s\n' "$got" "$out"; fail=$((fail+1))
fi

# ---- mock server for fixtures the live server can't trigger on demand ----
MOCK_PORT=59998
MOCK_PID=""
cleanup() { [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

"$PY" - "$MOCK_PORT" <<'PY' &
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            state = json.loads(raw).get("state", "") if raw else ""
        except Exception:
            state = ""
        code, body = 200, b'{"model":"m","answers":{"q":{"type":"noul","noul":0.5}}}'
        if state == "mock500":
            code, body = 500, b'{"type":"inference_error","message":"OUT_OF_DEVICE_MEMORY"}'
        elif state == "mockpartial":
            body = b'{"answers": null, "usage": {"in'   # truncated JSON
        elif state == "mockanswersnull":
            body = b'{"model":"m","answers": null}'
        elif state == "mockchoice_missing":
            body = b'{"model":"m","answers": {"q": {"type": "choice"}}}'
        elif state == "mockhang":
            time.sleep(5)
            body = b'{"model":"m","answers":{}}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def log_message(self, *a):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
MOCK_PID=$!

# bounded wait for the mock to bind
for _ in $(seq 1 30); do
  if "$PY" -c "import socket;s=socket.socket();s.settimeout(0.2);s.connect(('127.0.0.1',$MOCK_PORT));s.close()" 2>/dev/null; then break; fi
  sleep 0.1
done

# mk <desc> <expected_exit> <grep-pattern> <state>
mk() {
  local desc="$1" want="$2" pat="$3" state="$4" out got
  out="$(printf '%s' '{"state":"'"$state"'","questions":{"q":{"type":"noul","criteria":{"true":"x","false":"y"}}}}' \
        | HS_DECIDE_BASE_URL="http://127.0.0.1:$MOCK_PORT" HS_DECIDE_TIMEOUT=2 "$PY" scripts/decide.py 2>&1)"
  got=$?
  if [ "$got" = "$want" ] && printf '%s' "$out" | grep -q "$pat"; then
    printf '  PASS  %s\n' "$desc"; pass=$((pass+1))
  else
    printf '  FAIL  %s (exit %s, want %s + grep '%s')\n    %s\n' "$desc" "$got" "$want" "$pat" "$out"; fail=$((fail+1))
  fi
}

# 6. 500 inference_error -> surfaces HTTP error, exit 2
mk "mock 500 inference_error" 2 "500" "mock500"
# 7. malformed/partial response body -> exit 1
mk "mock malformed response body" 1 "not valid JSON" "mockpartial"
mk "mock answers null" 1 "answers" "mockanswersnull"
mk "mock choice missing fields" 1 "missing" "mockchoice_missing"
# 5. connection refused -> exit 1
out="$(printf '%s' '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"x","false":"y"}}}}' | HS_DECIDE_BASE_URL=http://127.0.0.1:59999 "$PY" scripts/decide.py 2>&1)"
got=$?
if [ "$got" = 1 ]; then printf '  PASS  conn refused -> exit 1\n'; pass=$((pass+1)); else printf '  FAIL  conn refused -> exit 1 (exit %s)\n    %s\n' "$got" "$out"; fail=$((fail+1)); fi

# 5b. real timeout (mock hangs longer than HS_DECIDE_TIMEOUT) -> exit 1
mk "real timeout" 1 "connection failed" "mockhang"

# 8. determinism: two identical live calls -> byte-identical output
A="$(printf '%s' '{"model":"hs-decide-1-flash","state":"det","questions":{"q":{"type":"choice","criteria":{"a":"1","b":"2","c":"3"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
B="$(printf '%s' '{"model":"hs-decide-1-flash","state":"det","questions":{"q":{"type":"choice","criteria":{"a":"1","b":"2","c":"3"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
if [ -n "$A" ] && [ "$A" = "$B" ]; then
  printf '  PASS  determinism (identical outputs)\n'; pass=$((pass+1))
else
  printf '  FAIL  determinism\n    A=%s\n    B=%s\n' "$A" "$B"; fail=$((fail+1))
fi

# 9. criterion-order sensitivity (qualitative): same criteria, two orders
O_A="$(printf '%s' '{"model":"hs-decide-1-flash","state":"rank these fixes","questions":{"q":{"type":"choice","criteria":{"restart":"restart the service","reindex":"reindex the index","refactor":"refactor the parser"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
O_B="$(printf '%s' '{"model":"hs-decide-1-flash","state":"rank these fixes","questions":{"q":{"type":"choice","criteria":{"refactor":"refactor the parser","restart":"restart the service","reindex":"reindex the index"}}}}' | "$PY" scripts/decide.py 2>/dev/null)"
if [ -n "$O_A" ] && [ -n "$O_B" ]; then
  printf '  PASS  criterion-order sensitivity (both orders succeed; inspect below)\n'; pass=$((pass+1))
  printf '        order A: %s\n' "$(printf '%s' "$O_A" | tr -d '\n')"
  printf '        order B: %s\n' "$(printf '%s' "$O_B" | tr -d '\n')"
else
  printf '  FAIL  criterion-order sensitivity\n'; fail=$((fail+1))
fi

echo ""
echo "== result: ${pass} passed, ${fail} failed =="
[ "$fail" = 0 ]