# Running Hindsight's `hs-decide-1-flash` decision model on Intel Arc iGPU (no CUDA, no vLLM)

Hindsight's `hs-decide-1-flash` (0.8B, Apache-2.0) is a **decision model** — it answers small typed judgments (`choice` / `noul` / `score`) that Hindsight otherwise sends to an LLM. The official "run it yourself" path is **vLLM on a CUDA GPU**.

This guide shows how to run it **fully locally with no discrete GPU**, using the **integrated Intel Arc iGPU** (Lunar Lake / Core Ultra 200V) via native **PyTorch XPU** — and gets a **~16–26× speedup over CPU** (~2.2 s → ~0.09 s per decision) at zero cost.

## Why not vLLM?

- vLLM's official `hindsight-decision-serve` targets CUDA. No CUDA here.
- The decision model's job is a **single forward pass** (token classification), not generation — so you don't need a serving engine; you need `model.to("xpu")`.

## Architecture

```
Hindsight API (v1.2.0, Docker) ── reflect + consolidation ──► hs-decide ──► Intel Arc iGPU
        │                                                      (FastAPI)
        │                                                      /v1/systemone
        ├── LLM: LM Studio (local)                              bearer auth
        ├── embeddings: local (bge-small-en-v1.5)
        └── reranker: local (hs-rerank-1)  ← recall stays here
```

The decision-model server is a small **PyTorch/FastAPI** service that reuses Hindsight's official `serving/` plugin for rendering/decoding and replaces only the vLLM engine with a `transformers` forward pass moved onto the iGPU.

## Prerequisites (Arch / Linux)

1. **Compute stack** (the step most people miss):
   ```bash
   sudo pacman -S intel-compute-runtime level-zero-loader clinfo
   sudo usermod -aG render,video $USER   # then log out and back in
   clinfo | grep "Device Name"           # → Intel(R) Arc(TM) Graphics
   ```

   Note: `intel-compute-runtime` ships the GPU driver (`libze_intel_gpu.so`), but you also need the *generic* Level Zero loader (`level-zero-loader`) or `torch.xpu.is_available()` returns `False`.

2. **PyTorch with XPU** (self-contained wheel; bundles the SYCL runtime, no oneAPI toolkit needed):
   ```bash
   python -m venv xpu-venv
   ./xpu-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu
   ./xpu-venv/bin/pip install transformers pydantic fastapi uvicorn
   ./xpu-venv/bin/python -c "import torch; print(torch.xpu.is_available())"  # → True
   ```

3. **Model + serving plugin**:
   ```bash
   hf download hindsight-ai/hs-decide-1-flash --local-dir model
   ./xpu-venv/bin/pip install ./model/serving
   ```

## The server (`server.py`)

`server.py` is in this repo. The heart of it:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.to("xpu").eval()

# Slice the lm_head to only the classifier columns (~528) up front.
# Qwen3.5 applies its final RMSNorm inside model.model.forward, so this narrow
# projection is bit-identical to slicing the full projection — at ~1/470th the memory.
narrow_w = model.lm_head.weight[CLS_IDS, :]

def forward(input_ids):                     # input_ids already on xpu
    with torch.no_grad():
        h = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return (h.to(narrow_w.dtype) @ narrow_w.t()).float()
```

Three non-obvious issues this handles (they cost the most time):

1. **Qwen3.5 keeps its final RMSNorm *inside* `lm_head`**, not in `model.norm`. Because `model.model.forward` returns `last_hidden_state` *after* applying that norm, you can safely project against a column-sliced head — do **not** pre-apply a second norm.
2. **BF16 OOM trap:** casting the *whole* `[L, 248320]` lm-head logits to float32 **on the device** per call exhausts the iGPU's shared-memory pool (`level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`). Keep the head in bf16 and `.float()` only the narrow classifier slice.
3. **Full-vocab projection OOM:** even in bf16, materializing `model.lm_head(h)` (a `[L, ~248k]` tensor) then slicing it held a 248k-wide transient that OOM-ed the shared-memory Arc 140V (`error 40 OUT_OF_RESOURCES`, spilling to swap and stalling the host). Pre-slicing the weight to the 528 classifier columns and projecting directly removes it. Verified bit-identical (max abs diff 0.0) and ~470× smaller.

The server also uses a **single-flight asyncio lock** so concurrent requests serialize (the shared-memory iGPU must not run two passes at once), optional bearer auth (`HS_DECIDE_API_KEY`) to satisfy Hindsight's client, and an opt-in **keep-warm loop** to stop the idle iGPU context from swapping out (see Gotchas).

Endpoints the server exposes:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/systemone` | the decision call itself |
| `GET` | `/health` (and `/healthz`) | liveness — Hindsight polls `/health` every ~30s |
| `GET` | `/v1/models` | OpenAI-style model list — Hindsight probes this too |

Run it:
```bash
./xpu-venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Or install the included `hs-decide.service` systemd unit (example uses a placeholder home; set your own path):
```bash
install -Dm644 hs-decide.service "$HOME/.config/systemd/user/hs-decide.service"
sed -i "s|__HS_DECIDE_HOME__|$HOME|g" "$HOME/.config/systemd/user/hs-decide.service"
sed -i "s|<your-secret>|CHANGE_ME|g" "$HOME/.config/systemd/user/hs-decide.service"
systemctl --user daemon-reload
systemctl --user enable --now hs-decide.service
```

The shipped unit sets `HS_DECIDE_WARM_SECONDS=0` (keep-warm off). On the OOM-prone shared-memory Arc, the narrowed lm_head removes the per-pass allocation that used to make the warm loop collide with real requests; disabling keep-warm avoids a wasted no-op pass every 60s. Re-enable it (drop the env line) only if you observe cold-start OOM on the first call after a long idle.

## Point Hindsight at it

```bash
HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
HINDSIGHT_API_DECISION_MODEL_BASE_URL=http://127.0.0.1:8000
HINDSIGHT_API_DECISION_MODEL_MODEL=hs-decide-1-flash
HINDSIGHT_API_DECISION_MODEL_API_KEY=<your-secret>   # must match HS_DECIDE_API_KEY on the server
HINDSIGHT_API_DECISION_MODEL_MAX_CONCURRENT=1
```

Leave recall on its local reranker (do **not** set `HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER`) so a down decision server degrades gracefully to the LLM instead of failing recall.

## Results (measured)

| Path | Latency per decision |
|---|---|
| CPU (FP32) | ~2.2 s |
| iGPU (FP32) | ~0.14 s |
| **iGPU (BF16)** | **~0.09 s** |

- 30 sustained concurrent calls: **0 errors, 94 ms avg, 262 ms max**.
- Answers are **bit-identical** to the CPU path (no numerical drift).

In Hindsight, consolidation triage went from `decision_input_tokens=0` (decision model failing → LLM fallback) to real answers (`restates=1/2`, 7k–29k tokens per run), and an 84-memory consolidation queue drained cleanly.

## Gotchas

- **Don't containerize the XPU server on an Arch host.** Baking the host's Intel Level Zero driver + `libstdc++` into a `python:3.12-slim`/Debian container produces a glibc/libstdc++ ABI mismatch (the driver needs `GLIBCXX_3.4.35`; that container base has `3.4.33`) and the driver aborts in `command_stream_receiver`. Run it as a host **systemd user unit** instead — the iGPU shares system RAM, so there's no isolation win in Docker anyway.
- **The idle iGPU context swaps out and OOMs on wake.** After idle, the Arc iGPU's context ages out and pages to swap (the host's shared-memory iGPU), so the first real decision after idle can hit `level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`. A **keep-warm loop** runs a no-op forward pass every `HS_DECIDE_WARM_SECONDS` (default `60`, ±0–5s jitter; set `0` to disable) to hold the context resident. Note the tradeoff: with the *old* full-vocab projection, the warm pass itself allocated enough to OOM (`error 40 OUT_OF_RESOURCES`) and would spam failures; the narrowed lm_head removes that, so keep-warm is now safe to re-enable if needed, but the shipped unit disables it by default. If you see `error 39` under *genuine* concurrent load rather than idle-wake, that's the shared-memory bar, not the swap-out — the FP32-on-CPU path (`HS_DECIDE_FP32=1`) is the OOM-proof fallback at ~2.2s/decision.
- **Hindsight polls `/health` and `/v1/models`**, not just the decision endpoint. Keep those routes (they return `200` and the model list) or the logs fill with 404s and Hindsight can't confirm the model is live.
- **If you run this alongside Hindsight's own memory plugin**, be aware the plugin's per-turn write-back (`retain`) can silently stall for a session after that session's transcript is compacted/checkpointed into a shorter form — its `retainedTurns` high-water mark never resets. Tracked upstream as `vectorize-io/hindsight#6314`. Unrelated to this server, but worth knowing so a "memory stopped updating" symptom isn't blamed on the decision model.

## This model is a generic decision endpoint, not just a Hindsight thing

`hs-decide-1-flash` speaks a **System One** contract (`choice` / `score` / `noul` behind `POST /v1/systemone`). The same server here answers arbitrary `state` + typed `questions` from *any* caller — n8n, scripts, opencode skills, guardrail plugins — not only Hindsight's memory engine. Send it a state and one or more typed questions, get back a probability per answer option.

## Related ecosystem

- **TypeSafe "Jev"** — the System One origin (`docs.typesafe.ai`); hs-decide is drop-in compatible.
- **Cloudflare Clef** & **OpenAI Decisions API** — hosted System One providers Hindsight also supports.
- **ollaya** (`github.com/ollaya-dev/ollaya`) — self-hosted System One server ("Ollama for decision models").
- **Hindsight decision-model docs** — the official (CUDA/vLLM) self-host instructions: `hindsight.vectorize.io/developer/decision-models`.

## License

The tutorial text and `server.py` are provided as-is under the MIT License. `hs-decide-1-flash` itself is Apache-2.0 (see its Hugging Face model card).
