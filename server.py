"""Server for hindsight-ai/hs-decide-1-flash exposing POST /v1/systemone.

Runs on CPU by default, or on an Intel iGPU (XPU) when HS_DECIDE_DEVICE=xpu.
Reuses the official serving plugin's request rendering and decoding logic
(`hindsight_decision.schema` / `.prompt` / `.typed` / `.typed_render`) verbatim,
replacing only the vLLM engine call with a transformers forward pass over the
full lm_head (Qwen3.5 keeps its final RMSNorm inside lm_head), then selecting
the classifier-token columns. Mirrors vLLM's `--convert classify` +
`token_classify` pooling: keep logits at the `:` slot positions for the
classifier tokens (no/yes, 0-9, A-Z, AA-ZZ), group-max over spelling aliases,
softmax.
"""

import asyncio
import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from hindsight_decision import schema
from hindsight_decision.prompt import load_settings
from hindsight_decision.typed import classifier_ids, question_logits, slot_rows, slot_token
from hindsight_decision.typed_render import render_request

MODEL_DIR = os.environ.get("HS_DECIDE_MODEL_DIR", "/model")
if os.environ.get("HS_DECIDE_FP32", "1") == "1":
    DTYPE = torch.float32
else:
    DTYPE = torch.bfloat16

# Device: xpu (Intel iGPU) when available and enabled, else cpu.
DEVICE = "cpu"
if os.environ.get("HS_DECIDE_DEVICE", "").lower() == "xpu":
    try:
        import torch  # noqa: F401 (already imported above)

        if torch.xpu.is_available():
            DEVICE = "xpu"
            print(f"[hs-decide] using XPU device: {torch.xpu.get_device_name(0)}", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[hs-decide] XPU requested but unavailable ({exc!r}); falling back to CPU", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
settings = load_settings(Path(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE)
model.to(DEVICE)
model.eval()

cls_ids = list(classifier_ids(tokenizer))
slot_id = slot_token(tokenizer)

# Bias of the output head (the classifier columns live inside model.lm_head).
bc = model.lm_head.bias if model.lm_head.bias is not None else None

_base = model.model


def forward_narrow(input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits [batch, L, ncls] at the classifier tokens only.

    Use model.lm_head on the whole pooled hidden state: for Qwen3.5 the final
    RMSNorm lives *inside* lm_head (not in model.model.norm), so it cannot be
    pre-applied to a column-sliced head. The cost is the full [L, V] projection,
    which is acceptable at CPU batch size 1 (~248k vocab).
    """
    with torch.no_grad():
        # BF16 on XPU is the fast path, but some fetches of intermediate tensors on
        # the device allocate transient GPU memory. Keep the LM-head projection in the
        # model's own dtype and materialize it as float only at the final narrow slice.
        h = _base(input_ids=input_ids, use_cache=False).last_hidden_state
        logits = model.lm_head(h)  # dtype = weight dtype (bf16 on XPU)
        if bc is not None:
            logits = logits + bc
        return logits[:, :, cls_ids].float()  # [batch, L, ncls], float


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE)


# Startup self-check: narrow path must match the full forward pass.
def _self_check() -> None:
    tokenizer_enc = tokenizer.encode(
        "State:\nDoes this help?\nQuestions:\n[Q1] (noul) is this useful?\n  no\n  yes\n",
        add_special_tokens=False,
    )
    probe = _to_device(torch.tensor([tokenizer_enc[:64]], dtype=torch.long))
    with torch.no_grad():
        full = model(input_ids=probe).logits.float()
        narrow = forward_narrow(probe)  # already float [1, L, ncls]
    full_cls = full[0, :, cls_ids].cpu()  # [L, ncls]
    narrow_cpu = narrow[0].cpu()
    diff = (full_cls - narrow_cpu).abs().max().item()
    if diff > 1e-2:
        raise RuntimeError(f"narrow/full logits mismatch (max abs diff {diff:.6f})")
    print(f"[hs-decide] self-check OK (max abs diff {diff:.2e})", flush=True)


_self_check()

# Auth: optional; the Hindsight client always sends `Authorization: Bearer <key>`.
# Open on loopback by default; a `HS_DECIDE_API_KEY` makes it require that bearer token.
EXPECTED_KEY = os.environ.get("HS_DECIDE_API_KEY", None)


def _authorized(raw_request: Request) -> bool:
    if EXPECTED_KEY is None:
        return True
    import secrets

    header = raw_request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return False
    token = header[len("Bearer ") :]
    return secrets.compare_digest(token, EXPECTED_KEY)


app = FastAPI(title="hs-decide-1-flash")


@app.on_event("startup")
async def _startup_warm_loop() -> None:
    if _WARM_INTERVAL > 0:
        asyncio.create_task(_warm_loop())

# Single-flight lock: the iGPU (shared-memory Arc 140V) runs out of device memory
# if two forward passes run concurrently (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, code 39).
# Serialize all inference so only one XPU pass runs at a time; callers queue briefly
# instead of OOM-ing. Safe at ~140ms/pass even under Hindsight's burst retries.
_infer_lock = asyncio.Lock()

# Keep-warm: the idle iGPU (shared-memory Arc 140V) context ages out and swaps to disk,
# so the first real decision after idle has to page the weights back into device memory
# and can hit OUT_OF_DEVICE_MEMORY (code 39). A periodic no-op forward pass holds the
# context resident. Only active on XPU; a no-op on CPU. Disabled by HS_DECIDE_WARM_SECONDS=0.
_WARM_INTERVAL = float(os.environ.get("HS_DECIDE_WARM_SECONDS", "60"))


def _warm_pass() -> None:
    """One throwaway forward pass; touches the model so XPU keeps the context hot."""
    with torch.no_grad():
        probe = _to_device(torch.tensor([[tokenizer.encode("warm", add_special_tokens=False)[:8]]], dtype=torch.long))
        _base(input_ids=probe, use_cache=False)


async def _warm_loop() -> None:
    import random

    while True:
        await asyncio.sleep(_WARM_INTERVAL + random.uniform(0, 5))
        if DEVICE != "xpu":
            continue
        try:
            async with _infer_lock:
                await asyncio.get_running_loop().run_in_executor(None, _warm_pass)
        except Exception as exc:  # never let a warm pass kill the service
            print(f"[hs-decide] warm pass failed ({exc!r})", flush=True)


def decode(request: schema.SystemOneRequest) -> dict:
    rendered = render_request(tokenizer, request, settings)
    input_ids = _to_device(torch.tensor([rendered.ids], dtype=torch.long))
    logits = forward_narrow(input_ids)  # [1, L, ncls] float (already on device until .cpu())

    positions = [i for i, t in enumerate(rendered.ids) if t == slot_id]
    data = logits[0, positions, :].cpu()  # [num_slots, ncls] on CPU
    row_idx = slot_rows(rendered, slot_id, data.shape[0])
    rows = data[row_idx].float().tolist()

    answers = {}
    for question, qlogits in zip(
        rendered.questions, question_logits(rows, rendered, cls_ids), strict=True
    ):
        answers[question.id] = schema.answer(
            request.questions[question.id], schema.softmax(qlogits)
        )
    return {
        "model": request.model,
        "answers": answers,
        "usage": {"input_tokens": rendered.tokens(), "output_tokens": 0},
    }


@app.get("/healthz")
@app.get("/health")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    # OpenAI-style model list: Hindsight probes this alongside /health. Return the
    # one model this server serves so the probe stops 404-ing.
    return {"object": "list", "data": [{"id": "hs-decide-1-flash", "object": "model"}]}


@app.post("/v1/systemone")
async def systemone(raw_request: Request) -> JSONResponse:
    if not _authorized(raw_request):
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    try:
        body = await raw_request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "the body is not valid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "the body must be a JSON object"}, status_code=400)
    try:
        request = schema.SystemOneRequest.model_validate(body)
    except ValidationError as error:
        return JSONResponse({"detail": error.errors(include_url=False)}, status_code=422)
    try:
        async with _infer_lock:
            # decode() runs a blocking tensor op; Offload so other requests (healthz, the
            # event loop) aren't stalled while the lock is held. The lock is held around
            # the offload, so calls still serialize.
            result = await asyncio.get_running_loop().run_in_executor(None, decode, request)
    except schema.ContextOverflow as error:
        return JSONResponse(schema.overflow_body(str(error)), status_code=400)
    except schema.Unanswerable as error:
        return JSONResponse(
            {"detail": [{"loc": ["body"], "msg": str(error), "type": "value_error"}]},
            status_code=422,
        )
    except RuntimeError as error:
        # Level Zero device-memory exhaustion etc.: fail loudly rather than infinitely retry.
        return JSONResponse({"detail": str(error), "type": "inference_error"}, status_code=500)
    return JSONResponse(result)