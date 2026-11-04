---
name: agent-decision-maker
description: >-
  Rank courses of action, score candidate tests or hypotheses, and flag
  duplicates/irrelevance using a local calibrated decision model
  (hs-decide-1-flash behind POST /v1/systemone). Use when the agent must CHOOSE
  among a fixed set of next-actions, RANK candidate steps or tests by likely value,
  or decide whether something is a duplicate / not-new-information — and wants a
  deterministic probability distribution instead of a vibe. The agent (LLM)
  proposes the candidates; this decision model ranks or chooses among them.
---

# Agent Decision Maker

Turn a "what should I do / which is best / is this new" moment into a
**calibrated ranking** using a local System One decision model.

## When to use

- **Rank next-actions** — "which of these steps should I take next?" → `choice` (or `score` for levels of preference).
- **Score candidate tests/hypotheses** — "how likely is each of these to pay off?" → `score`.
- **Flag duplicates / irrelevance** — "have we already covered this? is this NOT new information?" → `noul`.

You (the LLM) supply the **state** (a short factual summary of the situation) and the
**questions** (the fixed options to rank/choose among). The model returns a
probability per option, deterministic and bit-stable across runs.

## Endpoint

```
POST http://127.0.0.1:8000/v1/systemone
Authorization: Bearer local-dev-key
Content-Type: application/json
```

(Portable: change the base URL to any System One provider — ollaya, TypeSafe Jev,
OpenAI Decisions — with no other change. Keep `local-dev-key` matching the server's
`HS_DECIDE_API_KEY`.)

## Request shape

```json
{
  "model": "hs-decide-1-flash",
  "state": "short factual summary of the situation",
  "questions": {
    "<question_id>": { "type": "choice|score|noul", "...": "..." }
  }
}
```

### The three question types (get these right)

| type | meaning | required fields |
|---|---|---|
| `choice` | pick ONE of N named options | `criteria`: `{ "<label>": "<description>", ... }` (≥2) |
| `score` | rate on an ordered scale (treats as low→high) | `criteria`: `["<level1>", "<level2>", ...]` (2–10 levels) |
| `noul` | **is this NOT relevant / should we ignore / is there no new information?** | `criteria`: `{ "true": "<means not-relevant>", "false": "<means relevant>" }` |

> ⚠️ **Do NOT use `noul` for yes/no correctness or "is X true?"** `noul` literally
> encodes "no + null" — it answers *"should this be set aside / is there nothing
> here?"*, not "is the answer correct?". For a yes/no *correctness* question, use a
> 2-option `choice` instead. (Verified: a `choice` landed the right answer at 0.87
> where the same question as `noul` hedged at ~0.13.)

## Response shape

```json
{
  "model": "hs-decide-1-flash",
  "answers": {
    "<question_id>": {
      "type": "choice",
      "choice": "consolidate_now",
      "confidence": 0.28,
      "probabilities": { "consolidate_now": 0.474, "defer": 0.334, "disable": 0.192 }
    }
  },
  "usage": { "input_tokens": 37, "output_tokens": 0 }
}
```

- `choice` → `choice` (winner) + `probabilities` (full distribution).
- `score` → `score` (weighted value) + `probabilities` + `legend` (index→label).
- `noul` → `noul` = P(not-relevant); `1 - noul` = P(relevant).

## Worked examples (verified against this server)

### 1. Rank a course of action

Request:
```json
{ "model": "hs-decide-1-flash",
  "state": "A memory plugin stopped writing after a transcript was compacted. The bug is a monotonic turn counter that never resets.",
  "questions": {
    "root_cause": { "type": "choice", "criteria": {
      "plugin": "the plugin's counter logic",
      "server": "the database server",
      "gpu": "the iGPU decision server",
      "user": "the user's machine" } }
  } }
```

Answer: `choice = "plugin"` with `probabilities = { plugin: 0.982, ... }`.

### 2. Score candidate fixes by confidence

Request:
```json
{ "model": "hs-decide-1-flash",
  "state": "iGPU decision server hit error 39 after idle; cause was idle context swap-out; fix applied was a keep-warm no-op every minute.",
  "questions": {
    "fix_is_right": { "type": "choice", "criteria": {
      "correct": "keep-warm addresses the idle-swap cause",
      "wrong": "keep-warm does not address it",
      "partial": "helps but not a full fix" } }
  } }
```

Answer: `choice = "correct"` at `0.871`.

## Rules of thumb

1. **Propose candidates with the LLM, rank with the decision model.** Never ask it to
   invent options — give it a fixed, closed list.
2. **Match the type to the question.** `choice` = pick one; `score` = ordered rating;
   `noul` = relevance/dedupe only.
3. **Keep `state` terse and factual** — it's a 0.8B classifier, not a planner.
4. **Trust the distribution, not just the winner.** A near-tie (0.45 vs 0.40) means
   "genuinely close," not "first one won."
5. **Read the winner + probability as a decision aid**, present it back to the user
   with your own reasoning, and err toward the ranked order when acting autonomously.