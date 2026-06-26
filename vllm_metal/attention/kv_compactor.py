# SPDX-License-Identifier: Apache-2.0
"""Staged, transient-peak-bounded KV compactor (MLX building block).

Generalized from experiments/loss_budgeted_page_kv/exp017_reclaim.py. Gathers the
kept pages of a per-layer KV cache into smaller arrays. Doing all layers at once
needs the full cache + the full compact copy live simultaneously (transient peak
≈ full + compact). Staging layer-by-layer with an mx.eval + mx.clear_cache between
layers bounds the transient peak to roughly ``full + one layer's compact copy``.

IMPORTANT (honest scope): this physically frees bytes only when the input arrays
are a STANDALONE allocation (e.g. a single sequence's own cache, or a global pool
being quiesced and shrunk). In the live server the per-layer array is the SHARED,
statically-sized pool for ALL sequences (see vllm_metal/attention/reclaim_safety.py
docstring); gathering one sequence's pages into a new array does NOT shrink the
shared pool. Use this for standalone/global-quiesce compaction and benchmarking,
not as a per-sequence mid-decode memory reducer.
"""

from __future__ import annotations

from typing import Sequence


def stage_compact(arrays: Sequence["mx.array"], keep_idx: Sequence[int],
                  *, free_inputs: bool = True):
    """Return new per-layer arrays containing only ``keep_idx`` along axis 0,
    gathered one layer at a time to bound the transient peak.

    Args:
        arrays: per-layer arrays shaped ``[num_blocks, ...]`` (e.g. K or V caches).
        keep_idx: block indices to retain (chronological order recommended).
        free_inputs: if True, drop the local reference to each input array right
            after its compact copy is realized and clear MLX's cache, so the freed
            bytes are returned rather than held in the caching allocator.

    Returns:
        list of compacted arrays, same dtype, shape ``[len(keep_idx), ...]``.
    """
    import mlx.core as mx

    idx = mx.array([int(i) for i in keep_idx], dtype=mx.int32)
    out: list = []
    arrays = list(arrays)
    for i in range(len(arrays)):
        compact = arrays[i][idx]
        mx.eval(compact)              # realize this layer's copy
        out.append(compact)
        if free_inputs:
            arrays[i] = None          # drop ref to the full layer array
            mx.clear_cache()          # return its bytes (else they sit in the cache)
    return out


def transient_peak_bound(num_layers: int, full_bytes_per_layer: int,
                         keep_frac: float, staged: bool) -> int:
    """Predicted transient peak (bytes) during compaction.

    - all-at-once: full pool + full compact copy = full*(1 + keep_frac).
    - staged (free_inputs=True): full pool + ONE layer's compact copy.
    Use to choose staging when the all-at-once peak would risk OOM.
    """
    full = num_layers * full_bytes_per_layer
    if staged:
        return full + int(full_bytes_per_layer * keep_frac)
    return full + int(full * keep_frac)
