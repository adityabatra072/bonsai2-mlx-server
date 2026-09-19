"""OpenAI-compatible server for Ternary-Bonsai-2-27B on MLX.

Features beyond stock tooling:
  - persistent model (no reload per request)
  - row-fused grouped linears + fp16 Hadamard transform (see fused.py)
  - cross-request prefix KV-cache (single conversation slot + small LRU)
  - SSE streaming, native-feel tool_calls parsed from <tool_call> blocks,
    <think> extracted as reasoning_content
  - thinking effort control per request

Run:
  BONSAI_PACK=/Users/aditya/bonsai2-repack uvicorn server:app --host 127.0.0.1 --port 8081
"""
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.gpu)

PACK = Path(os.environ.get("BONSAI_PACK", "/Users/aditya/bonsai2-repack"))
RUNTIME = Path(os.environ.get(
    "BONSAI_RUNTIME",
    "/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit/runtime"))
MODEL_ID = os.environ.get("BONSAI_MODEL_ID", "bonsai2-27b-mlx")
CACHE_MIN_HIT = int(os.environ.get("BONSAI_CACHE_MIN_HIT", "16"))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RUNTIME))

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

print(f"[bonsai] loading {PACK} ...", flush=True)
t0 = time.time()
if (PACK / "fused.json").exists():
    from loader import load_repack
    model, model_config = load_repack(str(PACK), str(RUNTIME))
else:
    from loader import load_stock
    model, _, model_config = load_stock(str(PACK), str(RUNTIME))
    model = model  # full VLM wrapper
from vision_artifact import build_processor, chat_config

# load_repack returns the bare Model wrapper too (vision tower present but
# unloaded); generate_step expects the wrapper with .language_model.
processor = build_processor(str(PACK))
tokenizer = processor.tokenizer
template = (PACK / "chat_template.jinja").read_text()
base_config = chat_config(model_config)
print(f"[bonsai] ready in {time.time()-t0:.0f}s", flush=True)

import queue as _queue
from worker import start_worker

jobq = start_worker(model, processor)

lock = threading.Lock()
_stats = {"requests": 0, "cache_hits": 0, "cache_hit_tokens": 0,
          "prefill_tokens": 0}
# prefix cache slots: key -> {"tokens": [...], "cache": [...]}
_slots = OrderedDict()
MAX_SLOTS = 4


def render(messages, tools=None, enable_thinking=True, reasoning_effort="xhigh"):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=tools or None, enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort)


def encode(text):
    return tokenizer.encode(text)


TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
THINK_RE = re.compile(r"(?:<think>)?(.*?)</think>", re.DOTALL)
CTRL_RE = re.compile(r"<\|im_\w+\|>.*$", re.DOTALL)


def parse_output(text):
    """Split raw generation into (content, reasoning, tool_calls)."""
    reasoning = ""
    m = THINK_RE.search(text)
    if m and "</think>" in text:
        reasoning = m.group(1).strip()
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):].strip()
        text = text[:m.start()] + text[m.end():]
    calls = []
    def sub_mo(mo):
        name = mo.group(1).strip()
        args = {}
        for pm in PARAM_RE.finditer(mo.group(2)):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append({"id": f"call_{len(calls)}",
                      "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args)}})
        return ""
    text = TOOL_RE.sub(sub_mo, text).strip()
    text = CTRL_RE.sub("", text).strip()
    return text, reasoning, calls


def to_template_messages(messages):
    """OpenAI messages -> template messages (Qwen tool format)."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({"role": "tool",
                        "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                except Exception:
                    args = {}
                tcs.append({"type": "function",
                            "function": {"name": fn.get("name", ""),
                                         "arguments": args}})
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content
                                  if p.get("type") == "text")
            entry = {"role": "assistant", "content": content}
            if tcs:
                entry["tool_calls"] = tcs
            if m.get("reasoning_content"):
                entry["reasoning_content"] = m["reasoning_content"]
            out.append(entry)
        else:
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content
                                  if isinstance(p, dict)
                                  and p.get("type") == "text")
            out.append({"role": role, "content": content or ""})
    return out


def to_template_tools(tools):
    if not tools:
        return None
    return [{"type": "function", "function": {
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "parameters": t["function"].get("parameters", {})}}
        for t in tools if t.get("type", "function") == "function"]




def find_slot(tokens):
    """Find a slot whose stored tokens exactly prefix-match. Caller trims."""
    best, best_n = None, 0
    for k, s in _slots.items():
        cached = s["tokens"]
        if len(cached) > len(tokens):
            continue
        if list(cached) == list(tokens[:len(cached)]) and len(cached) > best_n:
            best, best_n = k, len(cached)
    if best is not None and best_n >= CACHE_MIN_HIT:
        print(f"[cache] HIT slot={best} prefix={best_n}/{len(tokens)}",
              flush=True)
        return best, best_n
    return None, 0


app = FastAPI(title="bonsai2-mlx-server")


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model",
                      "created": int(time.time()), "owned_by": "bonsai"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, **_stats,
            "slots": len(_slots)}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    # Single-flight: the worker thread serializes generation, and slot
    # metadata must not change under a running job.
    with lock:
        return _chat_completions(body)


def _drain(itemq, timeout=3600):
    """Collect a non-streamed job result (raises on worker error)."""
    text, finish, snap = "", "stop", (None, None)
    while True:
        kind = itemq.get(timeout=timeout)
        if kind[0] == "delta":
            continue  # non-streamed: only the final text matters
        if kind[0] == "done":
            _, text, finish, snap_c, snap_t = kind
            return text, finish, snap_c, snap_t
        raise kind[1]


def _chat_completions(body: dict):
    _stats["requests"] += 1
    messages = body.get("messages", [])
    tools = to_template_tools(body.get("tools"))
    enable_thinking = body.get("enable_thinking", True)
    reasoning_effort = body.get("reasoning_effort", "medium")
    params = {
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.8),
        "top_k": body.get("top_k", 20),
    }
    stream = body.get("stream", False)

    prompt_text = render(to_template_messages(messages), tools,
                         enable_thinking, reasoning_effort)
    tokens = encode(prompt_text)
    _stats["prefill_tokens"] += len(tokens)

    if os.environ.get("BONSAI_DEBUG_BODY") == "1":
        import hashlib as _hl
        _bd = Path("/tmp/bonsai-bodies")
        _bd.mkdir(exist_ok=True)
        (_bd / f"{int(time.time()*1000)}.json").write_text(json.dumps(body)[:200000])
        print(f"[dbg] prompt={len(tokens)} head={_hl.md5(str(tokens[:64]).encode()).hexdigest()[:8]}",
              flush=True)
    key, hit = find_slot(tokens)
    if key is not None:
        slot = _slots.pop(key)
        _slots[key] = slot  # MRU
        job_cache, start = slot["cache"], hit
        _stats["cache_hits"] += 1
        _stats["cache_hit_tokens"] += hit
    else:
        print("[cache] miss "
              f"prompt={len(tokens)} slots={len(_slots)}", flush=True)
        job_cache, start = None, 0

    def store_slot(snap_cache_obj, snap_tokens):
        print(f"[store] snap={'Y' if snap_cache_obj is not None else 'N'} "
              f"ntok={len(snap_tokens) if snap_tokens else 0}", flush=True)
        if snap_cache_obj is None or not snap_tokens:
            return
        slot_key = f"{abs(hash(tuple(tokens[:32]))):x}"
        _slots[slot_key] = {"tokens": list(snap_tokens),
                            "cache": snap_cache_obj}
        while len(_slots) > MAX_SLOTS:
            _slots.popitem(last=False)

    created = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    itemq = _queue.Queue()
    jobq.put((tokens, start, params, job_cache, itemq))

    if not stream:
        text, finish, snap_c, snap_t = _drain(itemq)
        content, reasoning, calls = parse_output(text)
        if calls and body.get("tools"):
            finish = "tool_calls"
        store_slot(snap_c, snap_t)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        return {"id": rid, "object": "chat.completion",
                "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": len(tokens),
                          "completion_tokens": len(encode(text)),
                          "total_tokens": len(tokens) + len(encode(text))}}

    def event_stream():
        try:
            while True:
                try:
                    kind = itemq.get(timeout=20)
                except _queue.Empty:
                    # prefill/decode can run minutes without emitting;
                    # keep SSE alive so clients never chunk-timeout.
                    yield ": working\n\n"
                    continue
                if kind[0] == "delta":
                    yield ("data: " + json.dumps(
                        {"id": rid, "object": "chat.completion.chunk",
                         "created": created, "model": MODEL_ID,
                         "choices": [{"index": 0,
                                      "delta": {"content": kind[1]},
                                      "finish_reason": None}]}) + "\n\n")
                elif kind[0] == "done":
                    _, text, finish, snap_c, snap_t = kind
                    content, reasoning, calls = parse_output(text)
                    msg = {"role": "assistant", "content": content or None}
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                    if calls and body.get("tools"):
                        msg["tool_calls"] = calls
                        finish = "tool_calls"
                    store_slot(snap_c, snap_t)
                    yield ("data: " + json.dumps(
                        {"id": rid, "object": "chat.completion.chunk",
                         "created": created, "model": MODEL_ID,
                         "choices": [{"index": 0, "delta": msg,
                                      "finish_reason": finish}]}) + "\n\n")
                    yield "data: [DONE]\n\n"
                    return
                else:
                    raise kind[1]
        except Exception as e:  # noqa: BLE001
            yield ("data: " + json.dumps({"error": str(e)}) + "\n\n")
    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")
