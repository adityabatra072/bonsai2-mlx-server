"""Verify two-phase (prefill + snapshot + 1-token generate) == single-call
generate, greedy. This is the exact protocol server.py uses."""
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
ids = t.encode(t.apply_chat_template(
    [{"role": "user", "content": "Say the word apple."},
     {"role": "assistant", "content": "apple"},
     {"role": "user", "content": "Now say banana."}], **kw))
print(f"prompt={len(ids)}", flush=True)


def collect(ids_in, cache_obj):
    out = []
    gen = generate_step(mx.array([ids_in], dtype=mx.int32), m, None, None,
                        max_tokens=24, temperature=0.0,
                        prompt_cache=cache_obj, verbose=False)
    for tok, _ in gen:
        out.append(tok)
    return out


def prefill_only(ids_in, cache_obj):
    gen = generate_step(mx.array([ids_in], dtype=mx.int32), m, None, None,
                        max_tokens=0, temperature=0.0,
                        prompt_cache=cache_obj, verbose=False)
    for _ in gen:
        pass
    mx.eval([c.state for c in cache_obj])


fresh = collect(ids, cache_mod.make_prompt_cache(m.language_model))
print("fresh:", t.decode(fresh), flush=True)

# two-phase with snapshot + restore (as server does on a cache hit)
ch = cache_mod.make_prompt_cache(m.language_model)
prefill_only(ids[:-1], ch)
stored = snap_cache(ch)
restored = snap_cache(stored)
cont = collect([ids[-1]], restored)
print("cont :", t.decode(cont), flush=True)
print("two-phase identical:", fresh == cont, flush=True)
