"""Offline repack (run: python repack.py):

Env: BONSAI_MLX_SRC (default: RunAnywhere shared MLX dir),
     BONSAI_REPACK_DST (default: ~/bonsai2-repack).

Offline repack: pre-concatenate grouped linears sharing one input transform.

Same values, no retraining. Output: text-only pack (vision tower dropped).
Groups per layer:
  full-attn (16): q,k,v -> qkv | gate,up -> gateup
  gdn      (48): in_proj_qkv,in_proj_z -> qkvz | gate,up -> gateup
Singles unchanged: o_proj, down_proj, out_proj, lm_head, embeddings.
"""
import json, os, sys
from pathlib import Path
SRC = Path(os.environ.get("BONSAI_MLX_SRC", "/Users/aditya/RunAnywhere/Shared-Models/MLX/ternary-bonsai-2-27b-mlx-2bit"))
DST = Path(os.environ.get("BONSAI_REPACK_DST", str(Path.home() / "bonsai2-repack")))
sys.path.insert(0, str(SRC / "runtime"))
import warnings; warnings.filterwarnings("ignore")
import mlx.core as mx

print("loading weights...", flush=True)
weights = mx.load(str(SRC / "model.safetensors"))
config = json.loads((SRC / "config.json").read_text())

def arr(k):
    return weights[k]

groups = []  # (new_key, [old_keys], kind)
n_layers = 64
for i in range(n_layers):
    base = f"language_model.model.layers.{i}"
    # detect layer kind by presence of self_attn.q_proj
    if f"{base}.self_attn.q_proj.weight" in weights:
        groups.append((f"{base}.self_attn.qkv_proj", [f"{base}.self_attn.{n}" for n in ("q_proj","k_proj","v_proj")], "qkv"))
    else:
        groups.append((f"{base}.linear_attn.qkvz_proj", [f"{base}.linear_attn.{n}" for n in ("in_proj_qkv","in_proj_z")], "qkvz"))
    groups.append((f"{base}.mlp.gateup_proj", [f"{base}.mlp.{n}" for n in ("gate_proj","up_proj")], "gateup"))

out = {}
consumed = set()
fused_meta = []
for new_key, olds, kind in groups:
    # verify signs equal across group
    signs = [arr(o + ".signs") for o in olds]
    mx.eval(*signs)
    for s in signs[1:]:
        assert bool(mx.array_equal(signs[0], s).item()), f"signs differ in {new_key}"
    W = mx.concatenate([arr(o + ".weight") for o in olds], axis=0)
    S = mx.concatenate([arr(o + ".scales") for o in olds], axis=0)
    B = mx.concatenate([arr(o + ".biases") for o in olds], axis=0)
    mx.eval(W, S, B)
    out[new_key + ".weight"] = W
    out[new_key + ".scales"] = S
    out[new_key + ".biases"] = B
    out[new_key + ".signs"] = signs[0]
    fused_meta.append({"key": new_key, "from": olds, "splits": [arr(o + ".weight").shape[0] for o in olds]})
    consumed.update(olds)
    del W, S, B
print(f"fused {len(groups)} groups", flush=True)

# copy everything else except vision tower + consumed linears
for k, v in weights.items():
    if k.startswith("vision_tower") or k.startswith("multi_modal_projector"):
        continue
    if any(k == o + s for o in consumed for s in (".weight", ".scales", ".biases", ".signs")):
        continue
    out[k] = v
del weights

DST.mkdir(exist_ok=True)
print("saving...", flush=True)
mx.save_safetensors(str(DST / "model.safetensors"), out)
(DST / "fused.json").write_text(json.dumps(fused_meta, indent=1))
(DST / "config.json").write_text((SRC / "config.json").read_text())
import shutil
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(SRC / f, DST / f)
total = sum(v.nbytes for v in out.values())
print(f"done: {len(out)} tensors, {total/1e9:.2f} GB", flush=True)
