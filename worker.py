"""Worker-thread execution core: ALL Metal work happens on one dedicated
thread that owns its GPU stream for its whole lifetime. Endpoint threads
never touch mx.arrays (only token-id lists and metadata)."""

import queue
import threading

import mlx.core as mx

from snaplib import snap_cache

_jobq = queue.Queue()


def start_worker(model, processor, model_config_unused=None):
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.models import cache as cache_mod

    def prefill_only(tokens_in, cache_obj, params):
        gen = generate_step(
            mx.array([tokens_in], dtype=mx.int32), model, None, None,
            max_tokens=0,
            temperature=params["temperature"],
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 20),
            prompt_cache=cache_obj,
            verbose=False)
        for _ in gen:
            pass
        mx.eval([c.state for c in cache_obj])

    def worker():
        stream = mx.new_stream(mx.gpu)
        with mx.stream(stream):
            while True:
                tokens_full, start, params, job_cache, itemq = _jobq.get()
                try:
                    if job_cache is None:
                        cache_obj = cache_mod.make_prompt_cache(
                            model.language_model)
                        lm = model.language_model
                        lm._position_ids = None
                        lm._rope_deltas = None
                    else:
                        cache_obj = snap_cache(job_cache)
                    suffix = tokens_full[start:]
                    assert len(suffix) >= 1
                    if len(suffix) == 1:
                        snap, snap_t = None, None
                        first = suffix
                    else:
                        prefill_only(suffix[:-1], cache_obj, params)
                        snap = snap_cache(cache_obj)
                        snap_t = list(tokens_full[:len(tokens_full) - 1])
                        first = [suffix[-1]]
                    detok = processor.detokenizer
                    detok.reset()
                    gen = generate_step(
                        mx.array([first], dtype=mx.int32), model, None, None,
                        max_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        top_p=params.get("top_p", 0.8),
                        top_k=params.get("top_k", 20),
                        prompt_cache=cache_obj,
                        verbose=False)
                    sent, n = "", 0
                    for tok, _ in gen:
                        detok.add_token(tok)
                        cur = detok.text
                        itemq.put(("delta", cur[len(sent):]))
                        sent = cur
                        n += 1
                    finish = ("length" if n >= params["max_tokens"]
                              else "stop")
                    itemq.put(("done", sent, finish, snap, snap_t))
                except Exception as e:  # noqa: BLE001
                    itemq.put(("error", e))

    threading.Thread(target=worker, daemon=True).start()
    return _jobq
