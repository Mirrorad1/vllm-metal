# IMPLEMENTATION_AUDIT — Loss-Budgeted Page KV Cache

Status: **COMPLETE.** Active dispatch path traced and the load-bearing
correctness fact re-verified directly against kernel source at the current tree
(branch `main`, after the shared page-gating seam was added). This gates all
later experiments (exp001).

```
ACTIVE_KERNEL_PATH = vllm_metal/metal/kernels_v2/pagedattention.metal  (decode; both _ps0 and _ps512)
SAFE_INITIAL_TARGET = host-side control plane (rewrite the per-seq block table + context_len
                      for the decode attention READ; no kernel change)
UNSAFE_MODES = tiled prefill kernel (pagedattention_tiled.metal; position-indexed),
               ALiBi, sliding_window>=0, FP8/TurboQuant, prefix-cache mutation,
               copy-on-write, page reclamation, speculative decode
```

## 1. Dispatch path (re-verified)

`v1/model_runner.py` → `attention/context.py::prepare_unified` (builds
`PagedAttentionContext{block_tables: list[list[int]], context_lens, cu_seqlens,
offsets, slot_mapping}`) → `attention/impls/sdpa.py::sdpa_forward` →
`ops.paged_attention_primitive(...)` (`sdpa.py:599/622`) → nanobind
`paged_ops.cpp::paged_attention_primitive` (`:1222`) →
`dispatch_paged_attention_v2_online` (`:353`).

Kernel selection (`paged_ops.cpp:373-414`):
- `has_prefill = total_q_tokens > num_seqs` ⇒ **tiled** kernel
  (`pagedattention_tiled.metal`) — UNSAFE for compaction (indexes
  `block_table[kv_pos/B]` by logical position, `:295`).
- pure decode (`total_q_tokens == num_seqs`) ⇒ **non-tiled**
  `pagedattention.metal`, either non-partitioned (`_ps0`) or partitioned
  (`_ps512`) + `paged_attention_v2_reduce` when grid occupancy is low and
  `max_partitions ≥ 2`. **Decode-only ⇒ always the safe non-tiled kernel.**

## 2. Tensors (decode), shapes / semantics

(`sdpa.py:447-465`, `paged_ops.cpp:1222`, kernel entry `pagedattention.metal:800`)

| arg | shape | dtype | meaning |
|---|---|---|---|
| `query` | `[L_q, H_q, d]` | fp16/bf16 | packed decode query tokens (1/seq) |
| `key_cache`/`value_cache` | `[num_blocks, B, H_kv, d]` | fp16/bf16 | physical pages; **K pre-RoPE'd at write time** |
| `block_tables` | `[num_seqs, max_blocks]` | int32 | physical page ids, 0-padded (`sdpa.py:127`); walked by iteration index |
| `context_lens`(`seq_lens`) | `[num_seqs]` | int32 | **true KV length**; sole causal extent |
| `cu_seqlens_q` | `[num_seqs+1]` | int32 | decode ⇒ +1/seg |
| `max_num_blocks_per_seq` | scalar | — | derived from `block_tables.shape(1)`; upper bound only |
| `sliding_window` | scalar | — | `<0` disables (we require `<0`) |
| `out` | `[L_q, H_q, d]` | = query | filled in place |

**BLOCK_SIZE (B):** cache pages are `B=32` tokens (`caches/turboquant.py:49`,
layout `kv_cache.py:8`); kernel supports B∈{8,16,32}.

## 3. The load-bearing fact (re-verified this session)

`pagedattention.metal` decode, current lines:
- `:864` `context_len = context_lens[seq_idx]`
- `:996` `physical_block_number = block_table[block_idx]` (iteration index)
- `:1005` `token_idx = block_idx*BLOCK_SIZE + offset` (iteration-relative)
- `:1063` `bool mask = token_idx >= effective_context_len` — **causal mask by COUNT**
- tail: `block_valid = MIN(B, effective_context_len - block_idx*B)` (only last block partial)

⇒ For RoPE-only decode the kernel is **position-agnostic w.r.t. the block
table**: a chronological subset of physical pages + matching reduced
`context_len` yields *exact* attention over the selected tokens, **no kernel
change**. ALiBi (`use_alibi=false`, `paged_ops.cpp:417`), sinks (`:419`), FP8
(`:418`) are compile-time-off in this dispatch; sliding_window gated by
`>=0` (`:1065`). **Invariant:** every selected page full except possibly the
last (only the tail is ever partial) — see `MATH.md §2`. Independently confirmed
in the prior experiment (kernel-vs-numpy err ≈ 4e-6; all-pages D_h = 0).

## 4. Allocation / refcount / reclamation / sharing

Owned entirely by **upstream vLLM v1 `KVCacheManager`/`BlockPool`** (no
refcount/free code in `vllm_metal/`). vllm-metal only reads per-step `block_ids`.
⇒ The initial intervention is a **read-only** compact block table for the
attention forward; `slot_mapping` (the KV *write* path) and the scheduler's
allocations are untouched. **No physical page is reclaimed in exp002–exp008**, so
those experiments can at best be **LOGICAL-ONLY POSITIVE** on memory (falsifier
F4). Real physical reduction (exp009) needs allocator integration with
reference-count safety (F15) and is deliberately deferred.

## 5. Feature flag / seam

`VLLM_METAL_LOSS_BUDGETED_PAGE_KV=1` + a selector registered via
`vllm_metal.attention.belief_gate.register_selector` (a generic, policy-agnostic
page-gating seam shared with the prior experiment; despite the file name it
performs no inference — it only rewrites the decode block table). OFF ⇒ exact
no-op (verified bit-identical via `tests/test_paged_deterministic.py`).
`apply_gate` re-validates each compaction (full-page-except-tail, `cand_cl ≤
orig`) and falls back to the full table on any violation (fail safe). Prefill
segments are never gated.

## 6. RoPE / GQA / position handling for the harness

RoPE positions are baked into cached K at write time; the query's position comes
from `offsets` (absolute), independent of the compact layout. GQA handled in
kernel (`kv_head_idx = head_idx/(H_q/H_kv)`, `:909`). The behavioral harness
(mlx-lm) reproduces this with a `GatedCache` that reports the **absolute**
sequence offset for RoPE while storing a compacted key set (else the query
rotation mismatches cached-key rotations — verified gotcha).
