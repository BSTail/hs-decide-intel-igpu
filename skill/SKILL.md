---
name: agent-decision-maker
description: >-
  Rank courses of action, score candidate tests or hypotheses, and flag
  duplicates/irrelevance using a local calibrated decision model
  (hs-decide-1-flash behind POST /v1/systemone). Use when the agent must CHOOSE
  among a fixed set of next-actions, RANK candidate steps or tests by likely value,
  or decide whether something is a duplicate / not-new-information — and wants a
  deterministic, probability-based ranking instead of a vibe. The agent (LLM)
  proposes the candidates; this decision model ranks or chooses among them.
---

# Agent Decision Maker

Turn a "what should I do / which is best / is this new" moment into a
**ranked distribution of probabilities** using a local System One decision model
(`hs-decide-1-flash`, 0.8B, behind `POST /v1/systemone`).

The LLM **proposes the candidates**; the decision model **ranks a closed set**.
It does not generate options, and it does not do open-ended reasoning.

## Quick start (executable)

The endpoint accepts JSON and returns JSON. One copy-paste curl:

```bash
curl -sS -m 20 -X POST "$HS_DECIDE_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $HS_DECIDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hs-decide-1-flash",
    "state": "short factual summary of the situation",
    "questions": {
      "next": { "type": "choice",
                "criteria": { "do_a": "take action A", "do_b": "take action B" } }
    }
  }'
```

A self-contained helper that reads a JSON request on **stdin** and prints the
validated response (or a clear error, with a proper exit code) is bundled at
`scripts/decide.py`. It reads `HS_DECIDE_BASE_URL`, `HS_DECIDE_API_KEY`, and
`HS_DECIDE_MODEL` from the environment — never hardcodes them — and handles
connection failures, timeouts, non-200s, and malformed bodies:

```bash
echo '{"state":"...","questions":{...}}' | python scripts/decide.py
```

## Configuration (environment)

| Variable | Default (local) | Meaning |
|---|---|---|
| `HS_DECIDE_BASE_URL` | `http://127.0.0.1:8000` | API base (no trailing slash) |
| `HS_DECIDE_API_KEY` | `local-dev-key` | bearer token (must match the server's `HS_DECIDE_API_KEY`) |
| `HS_DECIDE_MODEL` | `hs-decide-1-flash` | model id sent in the request |

Portability: the request/response contract here is the **System One** shape. You
can point `HS_DECIDE_BASE_URL` at any provider that implements this *same*
endpoint and contract (ollaya, TypeSafe Jev, OpenAI Decisions). But
authentication and model identifiers **may differ** — you must supply the
provider's own base URL, key, and model id. "No other change" holds only when the
remote implements the identical `/v1/systemone` contract.

## Request shape

```json
{
  "model": "hs-decide-1-flash",
  "state": "short factual summary of the situation",
  "questions": {
    "<id>": { "type": "choice|score|noul", "..." : "..." }
  }
}
```

`state` may be a string or a JSON object/list. Questions is a non-empty map; each
value must carry a `type` and the matching fields below.

### The three question types

| type | meaning | required field |
|---|---|---|
| `choice` | pick ONE of N named options | `criteria`: `{ "<label>": "<description>", ... }` (≥2 labels) |
| `score` | rate on an ordered scale low→high | `criteria`: `["<level0>", "<level1>", ...]` (2–10 levels) |
| `noul` | "is this NOT relevant / should we ignore / is there no new information?" | `criteria`: `{ "true": "<means not-relevant>", "false": "<means relevant>" }` |

Optional on every question: `instructions` (a string telling the model what to
judge).

> ⚠️ **Do NOT use `noul` for factual yes/no or "is X right?"** `noul` encodes
> "no + null": it answers *"should this be set aside / is there nothing new
> here?"*, not "is the answer correct?". For a yes/no *correctness* question, use
> a 2-option `choice`. (Verified: a `choice` landed the right answer at 0.87 where
> the same question written as `noul` hedged near 0.13.)

## Response shapes (verified against the running server)

Every answer object carries `type` plus the type-specific fields below.

### `choice`

```json
{ "type": "choice",
  "choice": "c",
  "confidence": 0.109,
  "probabilities": { "a": 0.347, "b": 0.247, "c": 0.406 } }
```

- `choice` — the winning label.
- `probabilities` — a distribution over the labels (sums to ~1).
- `confidence` — the server's own confidence value for the winning choice. It is
  **not a calibrated probability-of-correctness**; treat it as a rough strength
  signal only. (For `score` and `noul` it is `0.0` — the server does not populate
  it there.)

### `score`

```json
{ "type": "score",
  "score": 1.474,
  "confidence": 0.0,
  "legend": { "0": "low", "1": "mid", "2": "high", "3": "vhigh" },
  "probabilities": { "0": 0.344, "1": 0.130, "2": 0.235, "3": 0.292 } }
```

- `score` — the **expected level index** = Σ i·P(i), in `[0, levels-1]`. Low →
  favor the first (lowest) level; high → favor the last (highest).
- `legend` — maps each index `"0"`, `"1"`, … to its label.
- `probabilities` — distribution over the integer levels; `probabilities["0"]` is
  the mass of the *lowest* ("best" if you ordered low→high) bucket.

### `noul`

```json
{ "type": "noul", "noul": 0.388 }
```

- `noul` — P(not-relevant / ignore). `1 - noul` = P(relevant). **No** `legend`,
  `confidence`, or `probabilities` fields are returned for `noul`.

### Common

- Top-level response: `{ "model": ..., "answers": { "<id>": {...} }, "usage":
  { "input_tokens": N, "output_tokens": 0 } }`.
- Probabilities are returned un-rounded floats; round only for display.
- Ties: the server picks a single winner (`choice`) even when two probabilities
  are close — always read the full `probabilities` map, never just the winner.

## Failure handling & troubleshooting

`/health` (or `/healthz`) only confirms the process is **alive**; it does **not**
confirm inference is ready, and a healthy health-check can still precede an
inference failure. Treat a real `POST` result as the source of truth.

| Condition | Symptom | What to do |
|---|---|---|
| service down | connection refused / timeout | retry once after a moment; if still down, stop and report — do not loop |
| auth failure | HTTP `401` / `403` | check `HS_DECIDE_API_KEY` matches the server |
| validation error | HTTP `422` with `loc`/`msg` | fix the question shape (`criteria` count, missing `type`, etc.) |
| inference failure | HTTP `500` with `{"type":"inference_error"}` | the model hit a device/runtime error (e.g. `error 39/40 OUT_OF_DEVICE_MEMORY` / `OUT_OF_RESOURCES` on a shared iGPU). Retrying the *same* request may succeed after the device frees memory; do not hammer. If it persists, the local server may need a restart (`systemctl --user restart hs-decide.service`) |
| malformed/partial body | response is not valid JSON, or missing `answers` | treat as a failed call; do not parse/guess |

Rule: **a single retry is reasonable; after that, surface the failure to the user
rather than silently falling back to a guessed ranking.**

## When NOT to use (safety)

- **Prefer direct evidence or an inexpensive decisive test** over model judgment
  where one exists (run the command, read the log, check the fact).
- **Do not autonomously act on a near-tie or a low-confidence / under-supported
  choice** — if the top options are within a few points or `confidence` is tiny,
  ask the user or gather more evidence.
- **Never use this as the sole authority** for destructive, security-sensitive,
  production, or other high-stakes actions. It can *inform* a human decision; it
  does not *make* one.
- Treat outputs as **probability-like ranking scores**, not as the probability
  that a proposition is factually true.

## Calibration & determinism (qualified)

These outputs are **probability-like scores over the specific options you supply**,
not calibrated real-world probabilities. Determinism holds only when **all** of
these are identical: model, server version, device/backend, dtype, decoding
config, and request text. Reordering `criteria` keys/levels **can change a small
model's judgment** even though each individual request is reproducible — if
ordering matters, fix a canonical order and state it.

## Worked examples

### 1. Root-cause attribution (the model identifies the cause itself)

```json
{ "model": "hs-decide-1-flash",
  "state": "A memory plugin stopped writing sessions after a long transcript was compacted. A monotonic turn counter never resets when the transcript shrinks.",
  "questions": {
    "owner": { "type": "choice", "criteria": {
      "plugin": "the plugin's counter logic",
      "server": "the database server",
      "gpu": "the iGPU decision server",
      "user": "the user's machine" } }
  } }
```

Example response (actual server output): `choice = "plugin"`, `probabilities =
{ plugin: 0.98, server: 0.01, gpu: 0.00, user: 0.00 }`. *This is the model's
answer, not an independent confirmation that "plugin" is correct.*

### 2. Pick the best next action (genuinely unknown outcome)

```json
{ "model": "hs-decide-1-flash",
  "state": "Three candidate next steps for an open feature request: wait for a maintainer reply, build a small proof-of-concept, or write documentation.",
  "questions": {
    "next": { "type": "choice", "criteria": {
      "wait": "wait for the maintainer reply",
      "poc": "build a proof-of-concept now",
      "docs": "write documentation",
      "insufficient": "not enough information to choose" } },
    "urgency": { "type": "score", "criteria": [
      "no rush", "someday", "soon", "now", "blocking" ] }
  } }
```

Example response (actual): `next → "poc" (0.50)`, `urgency → score 2.16` favoring
the "now" bucket. The model weighed neutral, mutually-exclusive options and did
**not** have the answer pre-stated in `state`.

### 3. Is this a duplicate? (noul, its intended use)

```json
{ "model": "hs-decide-1-flash",
  "state": "A newly-proposed fix duplicates an already-merged change that removes the same race condition.",
  "questions": {
    "already_known": { "type": "noul", "criteria": {
      "true": "this adds no new information — ignore it",
      "false": "this is genuinely new — keep it" } }
  } }
```

`noul` near `1.0` ⇒ treat as duplicate/equivalent; near `0.0` ⇒ new information.

## Contract tests

`scripts/decide.py` is self-contained; `scripts/test_decide.sh` (or the same
requests by hand) exercises:

1. one successful request of each type (`choice`, `score`, `noul`);
2. incorrect `noul` usage (an `noul` question framed as yes/no — expect a hedged,
   unhelpful number, illustrating the warning above);
3. invalid `criteria` (a `choice` with one option) and empty `questions` → `422`;
4. auth failure (wrong key) → `403`;
5. timeout / connection refused → clear error, non-zero exit;
6. `500 inference_error` → the helper surfaces it and does not crash;
7. repeated-request determinism (two identical calls return identical
   probabilities) under the stated same-config conditions;
8. criterion-order sensitivity (swap two labels; note the probabilities may shift).

Run with the server up. The fixtures are the source of truth for the contract;
if a test and this document disagree, trust the test.