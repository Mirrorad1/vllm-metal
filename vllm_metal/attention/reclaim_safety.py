# SPDX-License-Identifier: Apache-2.0
"""Reference-count-safe reclamation gate + capacity accounting (advisory, read-only).

Background: experiments in experiments/loss_budgeted_page_kv/ established that lossy
page selection (belief_gate.apply_gate) can drop most of a sequence's KV pages from
the decode-attention READ while preserving behavior. The natural follow-up is to
actually *reclaim* the dropped physical blocks. But the KV pool is STATICALLY
pre-allocated (one shared per-layer MLX array sized at startup; upstream vLLM's
BlockPool is a fixed free-list), and upstream supports NO mid-sequence lossy block
release for full attention (frees happen only on request finish/preempt). So:

  * Reclamation cannot reduce process memory; at best it returns blocks to the
    free-list for REUSE → a CAPACITY/throughput win within the fixed pool.
  * Actually returning a live sequence's dropped blocks needs upstream changes
    (a BlockPool API + a scheduler step-time call). That is out of scope here.

What IS safe and buildable in vllm-metal alone is the *decision logic* and
*measurement*: given which blocks a gate dropped, which are SAFE to reclaim
(ref_cnt==1, not the null/sink block, not prefix-cache/copy-on-write shared), and
how much capacity that would free. This module is purely advisory — it never frees
a block, never mutates the scheduler, never touches slot_mapping. It is the F15
safety gate, fully unit-testable in isolation, that a future upstream-integrated
reclaimer would call before any free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


def safe_to_reclaim(
    active_block_tables: Mapping[object, Sequence[int]],
    gate_dropped: Mapping[object, set[int]],
    ref_cnts: Mapping[int, int],
    *,
    null_block_id: int = 0,
    prefix_cached_blocks: set[int] | None = None,
) -> dict[object, set[int]]:
    """Return, per request, the subset of its gate-dropped blocks that would be
    SAFE to physically reclaim. Advisory only — performs no frees.

    A dropped block ``b`` of request ``r`` is safe iff ALL hold:
      * it is not the null/sink block (``b != null_block_id``);
      * it is referenced exactly once (``ref_cnts[b] == 1``) — i.e. owned solely
        by ``r``, not copy-on-write shared;
      * it is not a prefix-cache entry (``b not in prefix_cached_blocks``) another
        request could hit;
      * it is not present in any OTHER active request's block table (defense in
        depth beyond ref_cnt, in case external refs are not reflected in ref_cnts).

    The result is always a SUBSET of ``gate_dropped[r]`` and has empty intersection
    with the unsafe set (shared ∪ null ∪ prefix-cached ∪ cross-referenced).
    """
    prefix_cached = prefix_cached_blocks or set()
    # blocks referenced by some active request, keyed by which requests reference them
    owners: dict[int, set[object]] = {}
    for req, bt in active_block_tables.items():
        for b in bt:
            owners.setdefault(int(b), set()).add(req)

    out: dict[object, set[int]] = {}
    for req, dropped in gate_dropped.items():
        safe: set[int] = set()
        for b in dropped:
            b = int(b)
            if b == null_block_id:
                continue
            if ref_cnts.get(b, 0) != 1:
                continue
            if b in prefix_cached:
                continue
            # not referenced by any OTHER active request
            other = owners.get(b, set()) - {req}
            if other:
                continue
            safe.add(b)
        out[req] = safe
    return out


@dataclass(frozen=True)
class ReclaimAccounting:
    """Honest capacity accounting for a gated step. Reports what reclamation WOULD
    free as CAPACITY (block slots returned to the fixed pool for reuse) — NOT a
    process-memory reduction (the pool is static-preallocated)."""

    reclaimable_blocks: int          # total blocks safely reclaimable this step
    dropped_blocks: int              # total blocks the gate dropped (pre-safety)
    headroom_tokens: int             # reclaimable_blocks * block_size — extra KV the pool can host
    per_request: dict[object, int]   # reclaimable blocks per request
    note: str = (
        "CAPACITY-for-reuse within a fixed pool; NOT process-memory reduction. "
        "Actual freeing requires upstream mid-sequence block release."
    )


def reclaim_accounting(
    active_block_tables: Mapping[object, Sequence[int]],
    gate_dropped: Mapping[object, set[int]],
    ref_cnts: Mapping[int, int],
    block_size: int,
    *,
    null_block_id: int = 0,
    prefix_cached_blocks: set[int] | None = None,
) -> ReclaimAccounting:
    """Compute the capacity that lossy reclamation would free this step (read-only)."""
    safe = safe_to_reclaim(
        active_block_tables, gate_dropped, ref_cnts,
        null_block_id=null_block_id, prefix_cached_blocks=prefix_cached_blocks,
    )
    per_req = {r: len(s) for r, s in safe.items()}
    total = sum(per_req.values())
    dropped = sum(len(s) for s in gate_dropped.values())
    return ReclaimAccounting(
        reclaimable_blocks=total,
        dropped_blocks=dropped,
        headroom_tokens=total * block_size,
        per_request=per_req,
    )
