"""Diagnose: is the checkpoint snapshot state == fresh prefix state?"""
import os, sys
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/Users/aditya/bonsai2-mlx-server")
RUNTIME = "/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit/runtime"
sys.path.insert(0, RUNTIME)
import warnings; warnings.filterwarnings("ignore")
import mlx.core as mx
mx.set_default_device(mx.gpu)
from loader import load_repack
from vision_artifact import build_processor
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models import cache as cache_mod
from snaplib import snap_cache

m, c = load_repack("/Users/aditya/bonsai2-repack", RUNTIME)
p = build_processor("/Users/aditya/bonsai2-repack")
t = p.tokenizer
kw = dict(tokenize=False, add_generation_prompt=True,
          enable_thinking=True, reasoning_effort="low")
ids1 = t.encode(t.apply_chat_template(
    [{"role": "user", "content": "Say the word apple."}], **kw))
B = len(ids1) - 1  # boundary at 44
print(f"prompt={len(ids1)} boundary={B}", flush=True)


def prefill_only(ids, cache_obj):
    gen = generate_step(mx.array([ids], dtype=mx.int32), m, None, None,
                        max_tokens=0, temperature=0.0,
                        prompt_cache=cache_obj, verbose=False)
    for _ in gen:
        pass
    mx.eval([c.state for c in cache_obj])


# fresh prefix state via plain prefill of ids1[:B]
ca = cache_mod.make_prompt_cache(m.language_model)
prefill_only(ids1[:B], ca)

# snapshot state via checkpoint path on full ids1
cb = cache_mod.make_prompt_cache(m.language_model)
snap = {}
gen = generate_step(mx.array([ids1], dtype=mx.int32), m, None, None,
                    max_tokens=0, temperature=0.0,
                    prompt_cache=cb, verbose=False,
                    prefill_step_size=B,
                    prompt_cache_checkpoint=lambda n, ch: snap.update(cache=ch),
                    prompt_cache_checkpoint_len=B)
for _ in gen:
    pass
print("checkpoint fired:", "cache" in snap, flush=True)


def flat_state(cache_obj):
    vals = []

    def rec(o):
        if isinstance(o, mx.array):
            vals.append(o)
        elif isinstance(o, (list, tuple)):
            for v in o:
                rec(v)
        elif isinstance(o, dict):
            for v in o.values():
                rec(v)
        elif hasattr(o, "__dict__"):
            for v in vars(o).values():
                rec(v)
    rec(cache_obj)
    return vals


A, Bc = flat_state(ca), flat_state(snap["cache"])
print(f"arrays: fresh={len(A)} snap={len(Bc)}", flush=True)
bad = 0
for i, (a, b) in enumerate(zip(A, Bc)):
    mx.eval(a, b)
    if a.shape != b.shape:
        print(f"[{i}] SHAPE {a.shape} vs {b.shape}")
        bad += 1
    elif not bool(mx.allclose(a, b, atol=1e-3).item()):
        d = float(mx.max(mx.abs(a - b)).item())
        print(f"[{i}] DIFF max={d:.4f} shape={tuple(a.shape)}")
        bad += 1
print("mismatched arrays:", bad, flush=True)
