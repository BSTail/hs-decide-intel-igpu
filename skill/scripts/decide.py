#!/usr/bin/env python3
"""Minimal client for the System One decision endpoint (POST /v1/systemone).

Reads a JSON request body from stdin, posts it, and returns the response (or a
clear error + non-zero exit code). Handles connection failures, timeouts,
non-200s, and malformed bodies without exposing the API key.

Configuration from environment (never hardcoded):
  HS_DECIDE_BASE_URL  (default http://127.0.0.1:8000)
  HS_DECIDE_API_KEY   (default local-dev-key)
  HS_DECIDE_MODEL     (default hs-decide-1-flash)

Usage:
  echo '{"state":"...","questions":{...}}' | python decide.py

Exit codes:
  0  success (response printed as JSON)
  1  connection / timeout / malformed response
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
    if not isinstance(parsed, dict) or "answers" not in parsed:
        print("error: response missing 'answers'", file=sys.stderr)
        return 1

    print(json.dumps(parsed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())