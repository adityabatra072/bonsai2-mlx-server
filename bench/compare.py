"""Correctness + speed: stock pack vs repacked+fused (greedy, fixed prompt)."""
import os, sys, time
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/Users/aditya/bonsai2-mlx-server")
import warnings; warnings.filterwarnings("ignore")
import mlx.core as mx
mx.set_default_device(mx.gpu)

RUNTIME = "/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit/runtime"
sys.path.insert(0, RUNTIME)
from vision_artifact import load_vl_model, chat_config
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
from loader import load_repack

PROMPT = "Write the numbers 1 to 100, one per line. No commentary."

print("loading stock...", flush=True)
sm, sp, sc = load_vl_model("/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit")
sprompt = apply_chat_template(sp, chat_config(sc), PROMPT, num_images=0)
t0 = time.time()
sout = generate(sm, sp, sprompt, [], max_tokens=128, temperature=0.0, verbose=False)
sdt = time.time() - t0
stext = sout if isinstance(sout, str) else sout.text
print(f"STOCK: {sdt:.1f}s = {128/sdt:.1f} tok/s", flush=True)
del sm, sp
mx.clear_cache()

print("loading repack...", flush=True)
rm, rc = load_repack("/Users/aditya/bonsai2-repack", RUNTIME)
from vision_artifact import build_processor
rp = build_processor("/Users/aditya/bonsai2-repack")
rprompt = apply_chat_template(rp, chat_config(rc), PROMPT, num_images=0)
t0 = time.time()
rout = generate(rm, rp, rprompt, [], max_tokens=128, temperature=0.0, verbose=False)
rdt = time.time() - t0
rtext = rout if isinstance(rout, str) else rout.text
print(f"REPACK: {rdt:.1f}s = {128/rdt:.1f} tok/s", flush=True)
print("outputs identical:", stext == rtext, flush=True)
print(f"speedup: {sdt/rdt:.2f}x", flush=True)
