# bonsai2-mlx-server

The first OpenAI-compatible server for **Ternary-Bonsai-2-27B on MLX** — and the fastest way to run it on a 16 GB Mac. Prism's own demo scripts refuse to serve Bonsai-2 over MLX (their words: `mlx_lm.server` / `mlx_vlm.server` would load the weights and *silently return wrong output*). This repo fixes that with a Hadamard-aware loader, a weight repack, and a server written for agentic coding.

## Why MLX needed surgery

Bonsai-2 stores language weights in a rotated (Hadamard) basis. Every projection needs an activation transform at runtime. The vendor `Packed` module does, per linear, per token: fp32 upcast → sign flip → 1024-block Hadamard → back to fp16 → 2-bit `quantized_matmul`. Profiling on a base M4 (10c, 16 GB) showed:

- MLX decode is **dispatch/kernel-efficiency bound, not bandwidth bound** (~66 GB/s effective vs ~120 GB/s roofline). 402 separate `Packed` calls per token dominate.
- The transform is redundant: q/k/v, gate/up, and the GDN input projections all share the *same* input, yet each transforms it separately (~25% tax per linear).
- `mlx-vlm` already fuses grouped quantized linears at decode — but only for `nn.QuantizedLinear`, which `Packed` is not, so Bonsai-2 misses it entirely.
- No server exists, so every call reloads 8.6 GB and re-prefills the full conversation.

## What this repo does

1. **Offline repack** (`repack.py`) — same values, zero retraining: pre-concatenates input-sharing groups (`q,k,v` → one, `gate,up` → one, `in_proj_qkv,z` → one; 128 groups total), drops the vision tower for the text server. 7.67 GB, written once.
2. **Fused execution** (`fused.py`, `loader.py`) — one fp16 Hadamard + one `quantized_matmul` per group, then split. fp16 transform verified **bit-identical greedy output** vs stock fp32. Custom `_target_verify_linears` dispatcher; no mlx-vlm fork needed.
3. **Server** (`server.py`) — FastAPI, OpenAI-compatible (`/v1/models`, `/v1/chat/completions`, SSE streaming, `/health`):
   - persistent model, single-flight lock,
   - **exact prefix KV-cache across requests** (two-phase prefill → snapshot → generate; verified numerically identical to fresh prefill, see `tests/cache_verify.py`),
   - `<think>` extracted as `reasoning_content`, `<tool_call>` blocks parsed into native `tool_calls` (+ full round-trips),
   - per-request `reasoning_effort` (`xhigh`/`medium`/`low`) and `enable_thinking`.

## Benchmarks (base M4, 16 GB unified)

| backend | prefill tok/s | decode tok/s | resident | multi-turn |
|---|---|---|---|---|
| llama.cpp Prism fork (PQ2_0, ctx 8k, reasoning-budget 2048) | 23.6 | **9.1** | ~8.5 GB | prefix cache (slots) |
| stock MLX (`mlx_vlm.generate`) | — | 6.5 | ~9.5 GB | none (re-prefills everything) |
| **this server (repack + fused + fp16)** | **37.3** | 8.6 | ~10 GB | **exact prefix cache** |

Honest summary: GGUF still leads single-stream decode by ~5% (its custom Metal kernels are excellent). MLX wins prefill by **1.6x**, and the prefix cache compounds that every agentic turn — a follow-up sharing 44 prompt tokens answered in **5.2s vs 11.2s** cold. Either backend fits 16 GB, but not both at once: run one server at a time.

Microbenchmarks behind the design (`bench/`): per-linear `quantized_matmul` 0.5–0.8 ms vs Hadamard 0.22 ms at Bonsai-2 shapes; fusion removes ~40% of transforms and ~40% of matmul dispatches per layer.

## Setup

```bash
uv venv .venv-server --python 3.11
uv pip install --python .venv-server/bin/python -r requirements.txt

# repack once (reads your existing MLX pack, no re-download)
python repack.py   # -> ~/bonsai2-repack (7.67 GB)

# serve (port 8081 so it can sit next to llama-server on 8080)
BONSAI_PACK=~/bonsai2-repack .venv-server/bin/python -m uvicorn server:app \
  --host 127.0.0.1 --port 8081
```

No existing pack? Point `BONSAI_MLX_SRC` at any `Ternary-Bonsai-2-27B-mlx-2bit` dir first. No repack? The server also loads a stock pack (`fused.json` absent) — slower, same API.

Env knobs: `BONSAI_PACK`, `BONSAI_RUNTIME` (pack's `runtime/` dir), `BONSAI_MODEL_ID` (default `bonsai2-27b-mlx`), `BONSAI_CACHE_MIN_HIT` (default 16 tokens).

## Use it

```bash
curl http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "bonsai2-27b-mlx",
  "messages": [{"role": "user", "content": "Write a python quicksort."}],
  "reasoning_effort": "medium", "max_tokens": 2048}'
```

opencode (`~/.config/opencode/opencode.jsonc`):

```jsonc
{"provider": {"bonsai-mlx": {
  "npm": "@ai-sdk/openai-compatible", "name": "Bonsai 2 27B MLX (local)",
  "options": {"baseURL": "http://127.0.0.1:8081/v1"},
  "models": {"bonsai2-27b-mlx": {"name": "Bonsai 2 27B MLX",
    "limit": {"context": 8192, "output": 4096}}}}}
```

Notes for coding: thinking tokens count into `max_tokens`, so keep it generous (4096); `medium` effort is the speed/quality sweet spot; tool calls work end to end.

## Tests

```bash
.venv-server/bin/python tests/cache_verify.py  # cache numerics (must print True)
.venv-server/bin/python tests/test_e2e.py       # 8 live-server tests
```

`tests/cache_diag.py` documents a dead end honestly: mlx-vlm's *chunked* prefill path is numerically different from stock prefill, so the server snapshots via plain prefill + physical cache copies instead.

## Layout

- `server.py` — FastAPI app (two-phase prefill/snapshot/generate, streaming, tools)
- `loader.py` — stock + repack loaders, dispatcher patch
- `fused.py` — `FusedQKV` + fp16 Hadamard
- `snaplib.py` — exact device-side cache snapshots
- `repack.py` — offline weight repack (values unchanged)
- `bench/` — `compare.py` (stock vs repack), `split.py` (prefill/decode split)
- `tests/` — `cache_verify.py`, `cache_diag.py`, `test_e2e.py`
