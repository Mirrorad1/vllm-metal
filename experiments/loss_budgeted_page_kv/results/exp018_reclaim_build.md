# exp018 — Reclamation safety gate + staged compactor (scope + build)

Goal: build toward physical KV reclamation behind the existing flag. A 4-agent code
audit (Understand→Scope workflow) verified the allocator model first — which CORRECTED
exp017's framing.

## Verified facts (allocator audit, unanimous, with citations)
- **KV pool is STATIC-PREALLOCATED.** Upstream vLLM `BlockPool.__init__` builds a fixed
  block list once (block_pool.py:162-164), freed blocks return to a free-list for reuse
  (block_pool.py:419-441), pool never shrinks. vllm-metal allocates one static
  `mx.zeros([num_blocks,…])` per layer at startup (kv_cache.py:153-155), shared across
  all sequences, sized from the memory budget (cache_policy.py num_blocks), never shrinks.
- **exp017's reclaim_ratio≈1.00 does NOT transfer to the server.** It measured a
  standalone single-sequence array that WAS the whole allocation. The real per-layer array
  is the shared pool; compacting one sequence's pages does not free its bytes. ⇒ the honest
  win is **CAPACITY-for-reuse (throughput), NOT process-memory reduction.**
- **Mid-sequence lossy free is NOT supported for full attention.** `get_num_skipped_tokens`
  returns 0 for FullAttention; frees happen only on finish/preempt (scheduler.py:1891
  asserts `request.is_finished()`). Returning a live sequence's dropped blocks needs an
  upstream BlockPool API + scheduler step-time call.

## Built (vllm-metal only, NO upstream fork, NO scheduler mutation, NO lying to the scheduler)
- `vllm_metal/attention/reclaim_safety.py` — the F15 SAFETY GATE (advisory, read-only):
  `safe_to_reclaim(active_block_tables, gate_dropped, ref_cnts, …)` returns the dropped
  blocks that are safe to reclaim (ref_cnt==1, not null/sink, not prefix-cached, not
  cross-referenced); `reclaim_accounting(...)` reports the CAPACITY (reclaimable blocks,
  headroom tokens) honestly labeled "not process-memory reduction".
- `vllm_metal/attention/kv_compactor.py` — `stage_compact(arrays, keep_idx)`: staged
  layer-by-layer gather with `mx.eval`+`mx.clear_cache` between layers, transient peak
  bounded to ~full+one-layer (vs full+full); honest docstring re: shared pool.
- `vllm_metal/attention/belief_gate.py` — added read-only `dropped_blocks(...)` helper and
  a DEBUG-level capacity log in `apply_gate`; preserves the existing "no physical page is
  reclaimed" contract (exact no-op when flags off — verified).
- Tests: `tests/test_reclaim_safety.py` (7, incl. a 200-case property test), 
  `tests/test_kv_compactor.py` (5, numeric-equivalence + transient-peak). **12/12 pass;**
  existing gate tests 5/5 pass (non-regressive).

## Deferred to upstream (clearly scoped, NOT faked)
Actual mid-sequence reclamation needs upstream vLLM: a `BlockPool` method to free a live
sequence's blocks (with an atomic ref_cnt==1 re-check at free time), threading a
`freed_block_ids` signal through `RequestState`/scheduler step, and rewriting the owning
sequence's authoritative block_table so future allocate_slots + attention agree the dropped
positions are gone (accepted-approximation contract). True process-memory reduction would
require a global pool quiesce-shrink-realloc (num_blocks is fixed at startup).

## Net
The safe FOUNDATION is shipped and tested: the logic that decides *which* lossily-dropped
blocks are safe to reclaim, the peak-bounded compaction primitive, and honest capacity
accounting. The real benefit is throughput/capacity (workload-dependent ~2-4× at iso-quality
budgets), realized only after the scoped upstream block-manager work — which is engineering,
not a research unknown.
