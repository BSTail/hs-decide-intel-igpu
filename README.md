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

def forward(input_ids):                    # input_ids already on xpu
    with torch.no_grad():
        h = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        logits = model.lm_head(h)          # bf16, full vocab
        return logits[:, :, CLS_IDS].float()   # narrow slice only, THEN float
```

Two non-obvious bugs this avoids (they cost the most time):

1. **Qwen3.5 keeps its final RMSNorm *inside* `lm_head`**, not in `model.norm`. Do **not** pre-apply a norm to a column-sliced head; project through `model.lm_head` and slice afterward.
2. **BF16 OOM trap:** casting the *whole* `[L, 248320]` lm-head logits to float32 **on the device** per call exhausts the iGPU's shared-memory pool (`level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`). Keep the head in bf16 and `.float()` only the narrow classifier slice.

The server also uses a **single-flight asyncio lock** so concurrent requests serialize (the shared-memory iGPU must not run two passes at once), and optional bearer auth (`HS_DECIDE_API_KEY`) to satisfy Hindsight's client.

Run it:
```bash
./xpu-venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Or use the included `hs-decide.service` systemd unit.

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

## License

The tutorial text and `server.py` are provided as-is under the MIT License. `hs-decide-1-flash` itself is Apache-2.0 (see its Hugging Face model card).