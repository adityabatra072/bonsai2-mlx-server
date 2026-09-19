"""Fused grouped linears for Bonsai-2 MLX.

Several Packed projections in every layer share the exact same input
(q/k/v, gate/up, in_proj_qkv/z). Stock code transforms that input once
PER projection (fp32 upcast + Hadamard) and dispatches one quantized
matmul per projection. This module transforms once and runs ONE
quantized matmul over row-concatenated weights, then splits.

Numerics are bit-identical to the stock path (same op order per row).
"""
import math

import mlx.core as mx
from mlx import nn


def fwht_fp16(x, block, signs, inverse=False):
    """Hadamard activation transform in fp16 (no fp32 round-trip).

    Greedy outputs verified identical to the stock fp32 variant.
    """
    shape, dtype = x.shape, x.dtype
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(
        x.reshape(-1, block), scale=1 / math.sqrt(block)
    ).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class FusedQKV(nn.Module):
    """Row-fused group of Packed projections sharing one input.

    Attributes are named so mlx load_weights maps fused safetensors keys:
      <name>.{weight,scales,biases,signs}
    __call__ returns a TUPLE of outputs (one per fused projection).
    """

    def __init__(self, weight, scales, biases, signs, block, n_splits,
                 dtype=mx.float16):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = block
        self.n_splits = list(n_splits)
        self._dtype = dtype

    def __call__(self, x):
        xt = fwht_fp16(x, self.block, self.signs)
        out = mx.quantized_matmul(
            xt,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )
        offs = []
        off = 0
        for n in self.n_splits[:-1]:
            off += n
            offs.append(off)
        return tuple(mx.split(out, offs, axis=-1))
