# Loss-Budgeted Page KV Cache — Status & Roadmap (canonical handoff)

Read this first. It is the honest, self-contained summary of the whole arc
(exp001–exp019) and the concrete next step. Plain-language up top, technical
roadmap at the bottom. Everything cited lives in this directory + the ledger
(`EXPERIMENT_LEDGER.md`) and the code paths under `vllm_metal/attention/`.

---

## TL;DR (honest)

We asked: can we be *clever* about which KV cache to keep, to save memory without
hurting quality? **Answer: no — and that "no" is the real finding.** The information
a model needs is **holographically smeared** across the whole cache (no single page,
head, or layer carries the answer; it's *computed late* from distributed evidence),
so a cheap "keep recent + high-attention pages" rule is already as good as a perfect
oracle. The leverage is **not** in *which* pages you keep; it's purely in **keeping
fewer and actually freeing them**. Doing that buys a real, quantified **~2–4×
serving-capacity** win at quality-safe budgets — using a **known technique**
(eviction), now measured and made safe for this stack. It is **not** a novel
capability and **not** a clean context extension.

---

## What was tested (19 experiments, compressed)

| phase | question | result |
|---|---|---|
| exp001–010 | does belief/loss-budgeted page selection beat baselines & save memory/latency? | ORACLE-ONLY / LOGICAL-ONLY / SYSTEMS-WALL — page selection works behaviorally but logical≠physical and kernel is overhead-bound |
| exp011 | was "attention ≈ oracle" a mean-of-opposites hiding a tail? | KL-ONLY (behaviorally inert): the answer never flips; **prefill information diffusion** |
| exp012 | is the redundancy low-rank (byte-compressible)? | sparsity-dominant + modest ~4× low-rank (already owned by low-rank-KV/MLA) |
| exp013/014 | is the answer in a few latent heads (retrieval heads)? | NO — holographic across kv-heads and query-heads |
| exp015 | when does the answer form (logit lens)? | LATE (layer 20/24) — *computed*, not stored → explains every NULL |
| exp016 | is there a universal cross-prompt subspace? | modest ~2× shared low-rank = existing static-low-rank-KV premise |
| exp017 | can you physically free dropped pages? | mechanism real (ratio 1.00 on a standalone array) |
| exp018 | does that transfer to the real server? | **NO** — pool is static-preallocated; win is CAPACITY not RAM; mid-seq free needs upstream |
| exp019 | how much capacity? | **~2–4× concurrent sequences at iso-quality (25–50% budget); ≈1/budget** |

The three durable findings:
1. **Holographic redundancy** (the answer is a late-computed function of distributed
   evidence; no exploitable sparse latent unit) — so clever selection is a dead end.
2. **The lever is systems, not signal** (keep fewer + free the slots).
3. **The win is ~2–4× capacity**, quantified under real safety constraints.

---

## What is TRUE vs what would be OVERCLAIM (read before pitching this)

- ✅ TRUE: ~2–4× more concurrent users (or ~2–4× longer *input ingestion*) in the same
  memory, at quality-safe budgets, on long-context workloads.
- ✅ TRUE: a clean negative-result contribution — *why* clever KV-selection doesn't beat
  simple eviction (holographic + late-computed), with the mechanism nailed down.
- ❌ OVERCLAIM "128k → 512k context": it's **lossy** — you ingest more but drop ~75% of
  the cache, so mid-context exact recall degrades. Not a clean context extension; the
  honest reliable-recall gain is much smaller than the memory multiplier.
- ❌ OVERCLAIM "novel": eviction / sliding-window / StreamingLLM / H2O already do "longer
  context with bounded memory." This brings it to vllm-metal; it does not invent it.
- ⚠️ UNVALIDATED AT SCALE: all measurements are 0.5B on Apple-Silicon with single-needle
  COPY tasks (the easy, maximally-redundant case). The 2–4× and the iso-quality budget
  are extrapolated principle, not measured on a real long-context model/benchmark.

---

## What is BUILT (shipped, tested, default-off)

- `vllm_metal/attention/reclaim_safety.py` — `safe_to_reclaim(...)` (the F15 safety gate:
  returns dropped blocks that are ref_cnt==1, not sink/null, not prefix/CoW-shared, not
  cross-referenced) + `reclaim_accounting(...)` (capacity reporting).
- `vllm_metal/attention/kv_compactor.py` — `stage_compact(...)` (peak-bounded gather; for
  the standalone/global-shrink path only — NOT needed for pool-reuse, see roadmap).
- `vllm_metal/attention/belief_gate.py` — `dropped_blocks(...)` + read-only capacity log;
  the decode-read compaction seam (`apply_gate`) already works behind the flags and is
  exact-no-op when off.
- Tests: `tests/test_reclaim_safety.py` (incl. 200-case property test) +
  `tests/test_kv_compactor.py` — **12/12 pass**; existing gate regressions **5/5 pass**.
- Flags: `VLLM_METAL_LOSS_BUDGETED_PAGE_KV` / `VLLM_METAL_BELIEF_PAGED_ATTENTION` (default off).

---

## THE NEXT STEP

Two parallel tracks. **Track A** turns the foundation into a live feature; **Track B**
tells you the real multiplier. Do B before pitching numbers; do A to ship.

### Track A — wire reclamation into the live engine (engineering)

KEY SIMPLIFICATION (corrects exp017's framing): the KV pool is a **static shared pool**;
reclamation = **return block slots to the free-list for reuse**, NOT gather/copy. So
there is **no data movement and no transient-peak problem** on this path — `kv_compactor`
is only for the separate global-shrink scenario. The real path is just "mark these slots
free."

Steps, with the hooks the allocator audit found:
1. **(vllm-metal, no fork)** In `v1/model_runner.py`, after `prepare_unified` + the gate,
   compute the per-request safe set with `reclaim_safety.safe_to_reclaim(active_block_tables,
   belief_gate.dropped_blocks(full, gated), ref_cnts, prefix_cached_blocks)`.
2. **(upstream vLLM, small)** Add `BlockPool.free_lossy(request_id, block_ids)` reusing the
   existing `free_blocks` machinery (block_pool.py:419-441), with an **atomic ref_cnt==1
   re-check at free time** (guards the prefix-cache `touch()` race, block_pool.py:411-415).
   Expose via `KVCacheManager`.
3. **(upstream vLLM)** Rewrite the owning request's authoritative block_table to mark freed
   positions as `null_block` (the SWA path at single_type_kv_cache_manager.py:448-501 is the
   template) so `allocate_slots` + attention agree the dropped positions are gone — an
   explicit "accepted-approximation / lossy-gap" contract on `RequestState`.
4. **(done)** The decode attention already reads the compact block table via `apply_gate`.
5. **Acceptance test:** on the live engine, drive a batch of long-context sequences; show
   sustained concurrent-sequence count rises ≈1/budget at iso-quality, prefix-cache hit
   rate unchanged, and generation correctness preserved at the chosen budget. Compare to
   the `exp019` analytic prediction.

Risks (from the audit): refcount race (mitigated by step 2's atomic recheck); breaking the
"block_table covers all computed tokens" invariant (mitigated by step 3's null-block gap
contract); prefix-cache hit-rate regression (mitigated by safe_to_reclaim excluding shared
blocks). **Do not** call free from the gate / lie to the scheduler — route through steps 1–3.

### Track B — validate the iso-quality budget on real workloads (science) [IN PROGRESS]

The 25% iso-quality assumption is borrowed from the eviction literature. **exp020** measures
the budget-vs-quality curve directly on progressively harder tasks (single-needle → multi-
needle → exact-long-string → multi-hop → distractor) with the deployable attention/recent
selector and exact-answer accuracy.
- **Local 0.5B (preliminary, `results/exp020_quality/`):** the attention selector holds
  iso-quality to ~6–12% budget (8–16×) — but this is an OPTIMISTIC bound (short context,
  fresh-attention signal, tiny model).
- **Real validation (run this): `exp020_quality_cuda.py` on RunPod** — see `RUNPOD.md`. A
  self-contained transformers/CUDA port (the MLX harness is Apple-only), validated on CPU:
  the correctness gate passes exactly (full-budget==dense, Δ=0). Run at 7B+/longer context
  to get the honest per-task iso-budget → the real capacity multiplier. Hypothesis: harder
  families need a higher budget, so the realistic multiplier is ~2–4×, not 16×.
- This is also the only test that could reopen the latent question (sparse retrieval heads
  may appear at 7B+/multi-hop, exp013/014 scope).

### Recommended order
1. **Track B first** (cheap, decisive on the real number) if a 7B+ box is available —
   it sets honest expectations and de-risks Track A's value.
2. **Track A** to ship the capacity feature, sized to Track B's safe budget.
3. If Track B reveals retrieval-head localization at scale, branch back to the latent
   search (the only thing that would revive "clever selection").

---

## Artifacts index
- Per-experiment verdicts: `results/exp0**/verdict.md` (+ `exp011…`, `exp017_reclaim`,
  `exp018_reclaim_build`, `exp019_capacity`).
- Final latent synthesis: `results/synthesis_exp011_exp016_FINAL.md`.
- Plots: `plots/` (memory/latency frontiers, exp011 tail, exp012 bytes-vs-graph, exp019 capacity).
- Ledger: `EXPERIMENT_LEDGER.md` (one row per experiment).
- Code + tests: `vllm_metal/attention/{reclaim_safety,kv_compactor,belief_gate}.py`,
  `tests/test_{reclaim_safety,kv_compactor}.py`.
