"""Loader for Bonsai-2 MLX packs (stock or offline-repacked).

- Stock pack: installs Packed singles exactly like the vendor runtime,
  with the fp16 Hadamard transform swapped in.
- Repacked dir (fused.json + model.safetensors, see repack.py): installs
  FusedQKV groups + Packed singles, fp16 transform, no weight duplication.

Both paths produce the same greedy outputs; the repack is faster.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def _patch_fwht():
    import runtime as R
    from fused import fwht_fp16
    R.fwht = fwht_fp16


def _patch_dispatcher():
    """One shared FusedQKV object sits behind q/k/v (etc); call it once."""
    from fused import FusedQKV
    import mlx_vlm.models.qwen3_5.language as L
    if getattr(L._target_verify_linears, "_bonsai_patched", False):
        return
    orig = L._target_verify_linears
    orig_single = L._target_verify_linear

    def patched(linears, x, target_verify):
        if not any(isinstance(l, FusedQKV) for l in linears):
            return orig(linears, x, target_verify)
        seen, idx, out = {}, {}, []
        for l in linears:
            if isinstance(l, FusedQKV):
                if id(l) not in seen:
                    seen[id(l)] = l(x)
                    idx[id(l)] = 0
                out.append(seen[id(l)][idx[id(l)]])
                idx[id(l)] += 1
            else:
                out.append(orig_single(l, x, target_verify))
        return tuple(out)

    patched._bonsai_patched = True
    L._target_verify_linears = patched


def load_stock(pack_dir, runtime_dir=None):
    """Load the vendor MLX pack (8.6 GB, incl. vision tower)."""
    pack_dir = Path(pack_dir)
    sys.path.insert(0, str(runtime_dir or (pack_dir / "runtime")))
    _patch_fwht()
    from vision_artifact import load_vl_model
    return load_vl_model(str(pack_dir))


def load_repack(repack_dir, runtime_dir):
    """Load an offline-repacked dir (see repack.py). Returns (model, config)."""
    from fused import FusedQKV

    repack_dir = Path(repack_dir)
    sys.path.insert(0, str(runtime_dir))
    _patch_fwht()
    from runtime import Packed
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    src_config = json.loads((repack_dir / "config.json").read_text())
    fused = {g["key"]: g for g in
             json.loads((repack_dir / "fused.json").read_text())}
    consumed = set()
    for g in fused.values():
        consumed.update(g["from"])

    _patch_dispatcher()
    model = Model(ModelConfig.from_dict(src_config))
    weights = mx.load(str(repack_dir / "model.safetensors"))
    lm = model.language_model

    def parent_of(dotted):
        parts = dotted.split(".")
        node = lm
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        return node, parts[-1]

    # fused groups: one shared module object per group
    for key, g in fused.items():
        assert key.startswith("language_model.")
        group_path = key[len("language_model."):]
        arrays = [weights[key + "." + s]
                  for s in ("weight", "scales", "biases")]
        signs = weights[key + ".signs"]
        rows = g["splits"]
        mod = FusedQKV(*arrays, signs, block=1024, n_splits=rows)
        for old in g["from"]:
            assert old.startswith("language_model.")
            p, name = parent_of(old[len("language_model."):])
            setattr(p, name, mod)

    # singles, vendor-style
    for record in src_config["modules"]:
        path = record["path"]
        key = "language_model." + path
        if key in consumed:
            continue
        arrays = [weights[key + "." + s]
                  for s in ("weight", "scales", "biases")]
        signs = weights.get(key + ".signs")
        p, name = parent_of(path)
        setattr(p, name, Packed(arrays, record["block"], signs,
                                record["embedding"], mx.float16))

    # Strict load. FusedQKV objects already hold their (shared, full) arrays
    # straight from the safetensors dict, so their tree keys are SKIPPED here
    # (loading per-member slices would overwrite the shared arrays).
    redirect = set()
    for g in fused.values():
        for old in g["from"]:
            for suffix in ("weight", "scales", "biases", "signs"):
                redirect.add(old + "." + suffix)
    flat = []
    for k, _ in tree_flatten(model.parameters()):
        if k in redirect:
            continue
        if k in weights:
            flat.append((k, weights[k]))
        else:
            assert ("vision_tower" in k or "multi_modal" in k
                    or "projector" in k), f"missing weight: {k}"
    model.load_weights(flat, strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model, src_config
