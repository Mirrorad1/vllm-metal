# IMPLEMENTATION AUDIT — Belief-Gated PagedAttention

Status: **COMPLETE — active dispatch path identified and the key correctness
property verified directly against kernel source.** This document gates all
later work (per the task spec: "Do not continue until the audit identifies the
actual dispatch path").

All citations are `file:line` against the working tree at commit `bec02ee`
(branch `main`). Claims that the whole experiment depends on were re-verified by
reading the source directly, not only via delegated search.

---

## 0. TL;DR — is the proposal feasible without touching the kernel?

**Yes, for decode-only / standard-RoPE / no-ALiBi / no-sliding-window / no-sinks.**

The active decode kernel walks the per-sequence block table **by iteration
index** and applies causal masking **by token count** (`token_idx >=
effective_context_len`), not by absolute logical position. RoPE is baked into
cached K at write time. Therefore a **chronologically-ordered subset** of a
sequence's physical pages, paired with a **matching reduced `context_len`**,
makes the existing kernel compute *exactly* attention over the selected tokens —
**no kernel change required.** The belief inference and page selection live
entirely in the host-side control plane, exactly as the spec demands.

The single numerical-exactness invariant (derived in §6, proved in `MATH.md`):

> **Every selected page must be a full `BLOCK_SIZE` page, except at most the
> final selected page, which may be the original partial tail. `context_len`
> must equal the total count of valid tokens across the selected pages.**

Because in a live sequence only the *last* logical page is ever partial, any
chronological subset that keeps the partial tail page last (or selects only full
pages) satisfies this automatically.

---

## 1. Active dispatch path (Python → C++ → Metal)

```
vllm_metal/v1/model_runner.py  (per-step: gather scheduler block_ids)
        │  decode_info / prefill_info  (block_ids come from vLLM v1 scheduler)
        ▼
vllm_metal/attention/context.py :: prepare_unified()        # context.py:128
        │  builds PagedAttentionContext{block_tables, context_lens,
        │  cu_seqlens, offsets, slot_mapping}  →  set_context(...)
        ▼
vllm_metal/attention/impls/sdpa.py :: sdpa_forward()        # call site sdpa.py:609
        │  ctx = get_context(); builds mx.arrays:
        │    seq_lens     = ctx.context_lens                 # sdpa.py:454
        │    cu_seqlens_q = ctx.cu_seqlens                   # sdpa.py:455
        │    block_tables,_ = _build_block_tables(ctx.block_tables, ...) # sdpa.py:463
        ▼
ops.paged_attention_primitive(q_3d, k_cache, v_cache, num_kv_heads, scale,
        softcap=0.0, block_tables, seq_lens, cu_seqlens_q, block_size,
        max_seq_len, sliding_window, out)                    # sdpa.py:609-623
        ▼
vllm_metal/metal/paged_ops.cpp :: paged_attention_primitive (nb binding)  # paged_ops.cpp:1222
        │  → dispatch_paged_attention_v2_online()            # paged_ops.cpp:353
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ has_prefill = total_q_tokens > num_seqs   (paged_ops.cpp:373)     │
   │   ├─ has_prefill && !TQ && dtype_match && hs∈{64,96,128,256,512}  │
   │   │     → TILED kernel  (pagedattention_tiled.metal)  ⚠ UNSAFE    │
   │   └─ pure_decode (total_q_tokens == num_seqs)                     │
   │         → NON-TILED kernel (pagedattention.metal)      ✅ SAFE     │
   │            ├─ partitioned (low grid occupancy, ≥2 partitions)     │
   │            │     paged_attention_..._ps512 + v2_reduce            │
   │            └─ non-partitioned  paged_attention_..._ps0            │
   └──────────────────────────────────────────────────────────────────┘
```

**The modified target file `pagedattention.metal` IS the file actually
dispatched for decode** (both `_ps0` and `_ps512` instantiations come from it —
`__init__.py:66` concatenates it into the v2 library). The tiled file
(`pagedattention_tiled.metal`, `__init__.py:67`) is dispatched **only** when
`has_prefill`, i.e. never for pure decode. **Restricting the experiment to
decode-only lands deterministically on the safe non-tiled kernel.** This directly
addresses falsifier #10 ("the active runtime dispatch never invokes the modified
kernel"): for the decode-only regime, the dispatched kernel is exactly
`pagedattention.metal`.

### 1.1 Feature flag

`VLLM_METAL_USE_PAGED_ATTENTION` (default `"1"`, `envs.py:49`) gates the whole
paged path vs. the MLX contiguous fallback. There is **no** pre-existing
belief/sparse/page-selection env var (grep over `vllm_metal/` confirms only
`VLLM_METAL_USE_PAGED_ATTENTION`). The experiment adds, disabled by default:

```
VLLM_METAL_BELIEF_PAGED_ATTENTION   (default "0")
```

---

## 2. Tensor shapes / dtypes / meanings (as passed to the primitive)

Built at `sdpa.py:447-465`, consumed by `paged_ops.cpp:1222` and the kernel at
`pagedattention.metal:800` (entry). `L` = total packed query tokens this step,
`P_max` = max blocks per seq, `H_q`/`H_kv` = query/kv heads, `d` = head dim,
`B` = tokens per page (`block_size`).

| arg | shape | dtype | meaning |
|---|---|---|---|
| `query` (`q_3d`) | `[L, H_q, d]` | fp16/bf16 (= cache dtype) | packed query tokens (decode tokens first, then prefill); transposed from `(1,H,L,d)` (`sdpa.py:447`) |
| `key_cache` | `[num_blocks, B, H_kv, d]` (or packed for TQ) | fp16/bf16/int | physical K pages; RoPE **already applied at write time** |
| `value_cache` | `[num_blocks, B, H_kv, d]` | fp16/bf16/int | physical V pages |
| `num_kv_heads` | scalar int | — | `H_kv`; GQA group = `H_q/H_kv` (`pagedattention.metal:908`) |
| `scale` | scalar float | — | `1/sqrt(d)` softmax scale |
| `softcap` | scalar float | — | always `0.0` in this codebase (`sdpa.py:615`) |
| `block_tables` | `[num_seqs, P_max]` | **int32** | per-seq physical page ids, **0-padded** (`sdpa.py:127`); walked by iteration index |
| `seq_lens` (`context_lens`) | `[num_seqs]` | int32 | **true KV length per seq**; sole source of causal extent |
| `cu_seqlens_q` | `[num_seqs+1]` | int32 | cumulative query-token boundaries; decode → each seg length 1 |
| `block_size` | scalar int | — | kernel page size ∈ {8,16,32} |
| `max_seq_len` | scalar int | — | `max(context_lens)`; sets partition count (`paged_ops.cpp:400`) |
| `sliding_window` | scalar int | — | per-layer; `<0`/absent ⇒ disabled (`pagedattention.metal:1065`) |
| `out` | `[L, H_q, d]` | = query | filled in place (`overwrite_descriptor`) |

`max_num_blocks_per_seq` is **not passed**; the C++ derives it from
`block_tables.shape(1)` and hands it to the kernel as a constant (buffer 13,
`paged_ops.cpp` set_bytes). It is an **upper bound only**: the kernel iterates
`num_context_blocks = ceil(context_len / B)` (`pagedattention.metal:878`), never
`P_max`.

### 2.1 Decode vs prefill

Single unified varlen kernel; no separate Python path. **Decode** = every
sequence contributes exactly one query token, so `cu_seqlens_q` increments by 1
per segment and `total_q_tokens == num_seqs` (`prepare_unified` decode loop,
`context.py:164-171`). The kernel finds a query token's sequence via
`find_seq_idx(cu_seqlens_q, …)` (`pagedattention.metal:856`).

---

## 3. Effect of BLOCK_SIZE / PARTITION_SIZE / GQA / FP8 / TurboQuant / sinks / ALiBi / sliding_window

| feature | how it affects the path | source |
|---|---|---|
| `BLOCK_SIZE` (default **32**, `turboquant.py:49`; cache layout `kv_cache.py:8`) | encoded in kernel name `_bs{8,16,32}`; sets `num_context_blocks` and tail mask | `paged_ops.cpp:410`; `pagedattention.metal:878,1088` |
| `PARTITION_SIZE` (default **512**, `m.attr` `paged_ops.cpp:1138`) | partitioned decode when `pure_decode && base_grid < cores*8 && max_partitions≥2` (`paged_ops.cpp:398-408`); adds reduce pass | `pagedattention.metal:1324` |
| GQA (`H_q/H_kv`) | runtime index `kv_head_idx = head_idx/num_queries_per_kv` (`pagedattention.metal:909`); no path change | — |
| FP8 cache | dtype shows in cache-type kernel-name suffix; `use_fp8` **hardcoded false** as a function constant | `paged_ops.cpp:418` |
| TurboQuant | **disables tiled path** (`paged_ops.cpp:378`); sets `use_tq_fc`, k/v bits function constants; deferred FWHT in reduce | `paged_ops.cpp:420` |
| attention sinks | `use_sinks` **hardcoded false** | `paged_ops.cpp:419` |
| ALiBi | `use_alibi` **hardcoded false** ⇒ `alibi_slope=0` ⇒ bias term is 0 | `paged_ops.cpp:417`; `pagedattention.metal:910,1058` |
| sliding_window | runtime arg; `<0` disables the extra mask term | `pagedattention.metal:1065` |

**Consequence for the experiment:** ALiBi, FP8-scaling and sinks are *not even
wired through* this dispatch (all three are compile-time-false), so the spec's
"no ALiBi / no sinks" constraints are satisfied by construction. We additionally
restrict to `sliding_window < 0` (the position-dependent mask term) and to
non-TurboQuant fp16/bf16 caches for the first experiment.

---

## 4. The position-agnostic property (verified directly, this is the linchpin)

Read directly from `pagedattention.metal`:

```
864  const uint32_t context_len = context_lens[seq_idx];
867  const int effective_context_len = (int)context_len - q_len + q_pos_in_seq + 1;
878  const int num_context_blocks = DIVIDE_ROUND_UP(effective_context_len, BLOCK_SIZE);
990  const device uint32_t *block_table = block_tables + seq_idx * max_num_blocks_per_seq;
993  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
995    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
1005   const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
1063   bool mask = token_idx >= effective_context_len;       // causal mask: BY COUNT
1087   const int block_start_token = block_idx * BLOCK_SIZE;
1089   const int block_valid_tokens = MIN(BLOCK_SIZE, effective_context_len - block_start_token);
```

Key observations:

1. **`token_idx` is an iteration coordinate**, `block_idx*B + offset`, where
   `block_idx` is just the column of the (possibly compact) block table — **not**
   the page's original logical position. (`:1005`)
2. **Causal masking is purely count-based** (`token_idx >= effective_context_len`,
   `:1063`). For pure decode (`q_len=1`, `q_pos_in_seq=0`),
   `effective_context_len == context_len`. So the kernel attends to *all* tokens
   it walks, masking only the tail of the final block.
3. **RoPE is not applied in the kernel** — cached K is pre-rotated at write time
   (`reshape_and_cache`). The QK dot at `:1051` uses cached K as-is. Each retained
   key therefore carries the correct rotation for its *original* absolute
   position, regardless of which compact column it now occupies.
4. **The only other position-dependent terms are gated off**: ALiBi
   (`alibi_slope=0`, `:1058`), sliding window (`sliding_window<0`, `:1065`), sinks
   (`use_sinks=false`). With those off, the kernel cannot observe a page's
   original logical index at all — only the count of valid tokens.

⇒ For RoPE-only decode the kernel is **position-agnostic w.r.t. the block
table**. A chronological subset of physical pages with a matching `context_len`
yields exact attention over those tokens. This is the central enabling fact.

**Tail subtlety (the one invariant):** interior blocks
(`block_idx < num_context_blocks-1`) get `block_valid_tokens == BLOCK_SIZE`
(`:1089`); only the last walked block is partially masked. So a compact table is
exact **iff** all selected pages except the last are full. Since only a
sequence's current tail page is ever partial, keeping pages chronological (tail
last) preserves exactness. Proof + edge cases in `MATH.md`.

### 4.1 Tiled kernel is NOT safe (and why it doesn't matter for decode)

`pagedattention_tiled.metal:295` indexes `block_table[kv_pos / BLOCK_SIZE]` by
**logical** position — a compact table would mislook-up. But the tiled kernel
fires only when `has_prefill` (`paged_ops.cpp:373-387`). Pure decode never
reaches it. The experiment is decode-only ⇒ tiled path is irrelevant. (If we ever
extend to prefill/chunked, the compact selection must be re-derived; flagged.)

### 4.2 Partitioned decode + reduce

Partitioning splits the *block sequence* into contiguous block ranges
(`:883-892`) and the reduce kernel (`:1324`) combines per-partition
(max_logit, exp_sum) in log2 space with no block-table indexing. A compact table
of `J` pages with `context_len=R` partitions and reduces correctly (the split is
over the compact sequence). Safe.

---

## 5. Page allocation, ref-counting, reclamation, prefix caching, COW

**vllm-metal owns none of this.** Allocation, ref-counting, prefix-cache block
hashing/sharing, and copy-on-write are entirely in **upstream vLLM v1
`KVCacheManager` / `BlockPool`** (`.venv-vllm-metal/.../vllm/v1/core/`). vllm-metal
only *reads* the per-request `block_ids` the scheduler assigns each step
(`model_runner.py` decode/prefill gather → `prepare_unified`). Grep for
`refcount|incref|decref|free_block` in `vllm_metal/` returns nothing.

**Implication for safety (falsifiers #5, #8 reclamation clause):** the first
experiment **must not reclaim any physical page.** The compact block table is a
*read-only view* substituted for the attention forward; the real
`slot_mapping`/cache and the scheduler's allocations are untouched, so no page is
freed and no prefix-cache/COW user can be disturbed. Real physical-memory
reduction (the harder claim) requires later integration with the allocator and
is explicitly out of scope for experiment 1 — we will measure *logical tokens
attended* and *pages referenced by the compact table* honestly and label them as
**shadow-mode** savings, NOT physical bytes freed (the spec forbids reporting
physical savings while the full cache is still allocated).

---

## 6. The control-plane seam (where belief gating plugs in)

Two clean interception points, both **read-path only**:

- **`PagedAttentionContext` rewrite** (preferred): after `prepare_unified` calls
  `set_context(...)` (`context.py:184`) and before the model forward reads it,
  replace, per decode segment, `ctx.block_tables[seg]` with a compact
  chronological subset and `ctx.context_lens[seg]` with the matching valid-token
  count. **Leave `slot_mapping`, `cu_seqlens`, `offsets` unchanged** —
  `slot_mapping` is the *write* slot for the current token (must stay real) and
  `offsets` is the query's RoPE position (unchanged).
- **`sdpa.py` transform**: a gated helper applied to `ctx.block_tables` /
  `ctx.context_lens` immediately before `_build_block_tables` (`sdpa.py:463`).

There is **direct precedent** for control-plane block-table rewriting in this
codebase: `yoco.py:107 build_yoco_reduced_context_from_full_metadata` builds a
*reduced* metadata view (reduced queries, full KV) from the full
`PagedAttentionContext`. Belief gating is the dual: reduced *KV pages*, full
query. We mirror its pattern (a pure function from full metadata → reduced
metadata) so the experiment's selector is testable in isolation with no MLX/kernel
dependency.

For the shadow A/B harness we do **not** even need the model runner: we can call
`ops.paged_attention_primitive` twice (full vs compact block table) on the same
`q`/cache and compare outputs/logits directly — the cleanest falsification rig.

### 6.1 Hybrid block-size caveat

`_build_block_tables` (`sdpa.py:107`) translates inflated vLLM block sizes
(e.g. 544 for mamba-aligned hybrids) into kernel pages. The first experiment
restricts to standard models where `kv_cache.block_size ∈ {8,16,32}` (no
inflation), so the compact selection maps 1:1 to kernel pages.

---

## 7. Restrictions for experiment 1 (all enforced by the audit findings)

- decode-only (⇒ safe non-tiled kernel, §1/§4.1)
- standard RoPE; `sliding_window < 0`; no ALiBi/sinks (all off by construction, §3/§4)
- non-TurboQuant fp16/bf16 cache; `block_size ∈ {8,16,32}`, no hybrid inflation (§6.1)
- no physical-page reclamation; shadow read-only view only (§5)
- all-pages selection must reproduce the baseline within repo numeric tolerance
  (falsifier #8) — this is the first unit test.

---

## 8. Open verification items carried into implementation

1. Confirm fp16 vs bf16 default cache dtype for the chosen small model at runtime.
2. Confirm partitioned-decode actually triggers at the tested context lengths on
   this GPU (`base_grid < gpu_core_count()*8`), so the all-pages-equivalence test
   exercises both `_ps0` and `_ps512`.
3. Confirm `VLLM_METAL_BUILD_FROM_SOURCE=1` source build works with Command Line
   Tools only (per `NOTES.md`), since the checkout ships no prebuilt kernels for
   an edited tree.
