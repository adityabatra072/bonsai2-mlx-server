"""Split bench: prefill vs decode for the repacked MLX model."""
import os, sys, time
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/Users/aditya/bonsai2-mlx-server")
import warnings; warnings.filterwarnings("ignore")
import mlx.core as mx
mx.set_default_device(mx.gpu)
RUNTIME = "/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit/runtime"
sys.path.insert(0, RUNTIME)
from loader import load_repack
from vision_artifact import build_processor, chat_config
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template

m, c = load_repack("/Users/aditya/bonsai2-repack", RUNTIME)
p = build_processor("/Users/aditya/bonsai2-repack")
long_prompt = apply_chat_template(
    p, chat_config(c),
    "Explain binary search in detail. " * 40, num_images=0)
n_tok = len(p.tokenizer.encode(long_prompt))
print(f"prompt tokens: {n_tok}", flush=True)

t0 = time.time()
generate(m, p, long_prompt, [], max_tokens=1, temperature=0.0, verbose=False)
t_first = time.time() - t0
print(f"prefill+1: {t_first:.1f}s -> prefill {n_tok/t_first:.1f} tok/s", flush=True)

t0 = time.time()
generate(m, p, long_prompt, [], max_tokens=65, temperature=0.0, verbose=False)
t_all = time.time() - t0
dec = 64 / (t_all - t_first)
print(f"decode: {dec:.1f} tok/s", flush=True)
