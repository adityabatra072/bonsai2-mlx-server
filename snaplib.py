"""Cache snapshot helpers (shared by server and tests)."""
import mlx.core as mx


def snap_cache(obj):
    """Deep copy a cache structure with fresh device buffers.

    KV caches update preallocated buffers in place, so snapshots must own
    their memory. -(-x) is an exact, device-side copy for numeric arrays.
    """
    if isinstance(obj, mx.array):
        mx.eval(obj)
        return -(-obj)
    if isinstance(obj, list):
        return [snap_cache(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(snap_cache(v) for v in obj)
    if isinstance(obj, dict):
        return {k: snap_cache(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        import copy as _copy
        y = _copy.copy(obj)
        y.__dict__ = {k: snap_cache(v) for k, v in vars(obj).items()}
        return y
    return obj
