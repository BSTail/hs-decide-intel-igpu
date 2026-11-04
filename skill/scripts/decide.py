#!/usr/bin/env python3
"""Minimal client for the System One decision endpoint (POST /v1/systemone).

Reads a JSON request body from stdin, posts it, validates the response, and
prints it (or a clear error + non-zero exit code). Handles connection failures,
timeouts, non-200s, and malformed/partial bodies without exposing the API key.

Configuration (documented local defaults, overridable via environment):
  HS_DECIDE_BASE_URL  (default http://127.0.0.1:8000)
  HS_DECIDE_API_KEY   (default local-dev-key)
  HS_DECIDE_MODEL     (default hs-decide-1-flash)
  HS_DECIDE_TIMEOUT   (default 20 seconds)

Usage:
  echo '{"state":"...","questions":{...}}' | python decide.py

Exit codes:
  0  success (response printed as JSON)
  1  connection / timeout / malformed response / failed response validation
  2  HTTP error (non-200)
  3  usage error (bad stdin JSON)
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("HS_DECIDE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("HS_DECIDE_API_KEY", "local-dev-key")
MODEL = os.environ.get("HS_DECIDE_MODEL", "hs-decide-1-flash")
TIMEOUT = float(os.environ.get("HS_DECIDE_TIMEOUT", "20"))
ENDPOINT = f"{BASE_URL}/v1/systemone"

# Type -> fields every valid answer of that type must carry.
_REQUIRED = {
    "choice": ("choice", "probabilities"),
    "score": ("score", "legend", "probabilities"),
    "noul": ("noul",),
}


def validate_answers(parsed, requested_ids):
    """Return an error string, or None if the response answers are complete."""
    answers = parsed.get("answers")
    if not isinstance(answers, dict) or not answers:
        return "response 'answers' is missing, null, or empty"
    missing = [qid for qid in requested_ids if qid not in answers]
    if missing:
        return "response missing answers for: " + ", ".join(missing)
    for qid, ans in answers.items():
        if not isinstance(ans, dict):
            return f"answer '{qid}' is not an object"
        t = ans.get("type")
        if t not in _REQUIRED:
            return f"answer '{qid}' has unknown type {t!r}"
        for field in _REQUIRED[t]:
            if field not in ans:
                return f"answer '{qid}' ({t}) is missing '{field}'"
    return None


def main() -> int:
    raw = sys.stdin.read()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: stdin is not valid JSON ({exc})", file=sys.stderr)
        return 3

    if not isinstance(body, dict):
        print("error: request body must be a JSON object", file=sys.stderr)
        return 3
    body.setdefault("model", MODEL)

    requested_ids = []
    questions = body.get("questions")
    if isinstance(questions, dict):
        requested_ids = list(questions.keys())

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode()
        except Exception:
            detail = ""
        print(f"error: HTTP {exc.code}: {detail}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"error: connection failed: {exc}", file=sys.stderr)
        return 1

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        print("error: response is not valid JSON", file=sys.stderr)
        return 1

    if status != 200:
        print(f"error: HTTP {status}: {parsed}", file=sys.stderr)
        return 2
    if not isinstance(parsed, dict):
        print("error: response is not a JSON object", file=sys.stderr)
        return 1

    err = validate_answers(parsed, requested_ids)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(json.dumps(parsed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())