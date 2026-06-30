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
oracle (note: "recent" *alone* is not enough — see exp020; you need the attention
signal to find off-recency answers). The leverage is **not** in *which* clever scoring
you use; it's in **keeping fewer pages and actually freeing them**. Doing that buys a
real serving-capacity win at quality-safe budgets — using a **known technique**
(eviction), now measured and made safe for this stack. It is **not** a novel
capability and **not** a clean context extension.

**How big? It is WORKLOAD-DEPENDENT — now measured at both ends (exp020, 2026-06, Qwen2.5-7B
@ ~20k, fast engine self-proven ≡ reference, preflight 60/60):**
- **Sparse-answer RETRIEVAL** (one needle in ignorable filler): attention selector holds
  iso-quality to **6.25% budget on 4/5 families (16×)** and **12.5% on 2-hop (8×)**.
- **DENSE AGGREGATION** (answer = a function of many distributed spans — `sum_scattered`, the
  sum of 5 deposits placed far apart): **iso-budget = 100% → 1.0× capacity. It does not compress
  at all.** Accuracy falls 1.00→0.69→0.34→0.17 as budget drops 100%→50%→25%→6.25%. The arithmetic
  is controlled (same instances at every budget), so this is pure page-retention: every addend
  page is needed and none is query-salient, so the selector can't keep them all under budget.

So the honest headline is **1× (dense aggregation) … 16× (sparse retrieval)** — quote the number
for YOUR workload, not a single universal figure. This is the workload-level confirmation of the
holographic finding: when the answer is a function of distributed evidence you cannot evict any of
it; eviction's big wins are confined to sparse-answer/retrieval traffic. For mixed real workloads,
the aggregation/synthesis fraction caps you near 1× at iso-quality; **~2–4× remains a reasonable
blended planning figure**, with 8–16× only for retrieval-dominated traffic.

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
- ✅ NOW VALIDATED AT SCALE FOR RETRIEVAL (exp020, Qwen2.5-7B @ ~20k ctx, n=40/family):
  the deployable attention selector holds exact-answer iso-quality to 6.25% budget on 4/5
  families (16×) and 12.5% on true-2-hop (8×); fast cache-slicing engine self-proved equal
  to the validated mask reference (gate Δ=0, preflight 60/60 decode + 60/60 selector).
- ✅ NOW MEASURED — DENSE AGGREGATION DOES NOT COMPRESS: the `sum_scattered` family (answer =
  sum of 5 deposits scattered across ~20k) holds iso-quality only at **100% budget (1.0×)** —
  accuracy falls 1.00→0.69→0.34→0.17 as budget drops to 50/25/6.25%, with arithmetic controlled
  (same instances at every budget). So 8–16× is a sparse-RETRIEVAL ceiling; aggregation/synthesis
  traffic gets ~1× at iso-quality. The honest multiplier is workload-dependent (1×…16×); ~2–4× is
  a reasonable blended figure. (recall_all, a non-arithmetic dense check, was unsolvable at full
  cache on 7B → 0 valid; RECALL_K lowered for an optional confirming run.)
- ⚠️ STILL UNMEASURED: full-GENERATION quality (perplexity over a long continuation, not just
  exact-answer), and realized serving THROUGHPUT (still SELECTION quality; needs Track A / exp018
  wiring to convert the safe budget into concurrent-sequence capacity).

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

### Track B — validate the iso-quality budget on real workloads (science) [7B/20k DONE]

The 25% iso-quality assumption is borrowed from the eviction literature. **exp020** measures
the budget-vs-quality curve directly on progressively harder tasks (single-needle → multi-
needle → exact-long-string → multi-hop → distractor) with the deployable attention/recent
selector and exact-answer accuracy.
- **Local 0.5B (preliminary):** attention selector holds iso-quality to ~6–12% budget — an
  optimistic bound (short context, tiny model).
- **✅ 7B @ ~20k context (RunPod H100, n=40/family, 2026-06):** attention selector holds
  exact-answer iso-quality to **6.25% budget on single_needle / exact_long_string /
  multi_needle / distractor (16×)** and **12.5% on true-2-hop multi_hop (8×)**; **recent-only
  collapses to ≤0.20 at 50% budget** (the attention signal carries real off-recency
  retrieval). The fast cache-slicing engine self-proved equal to the validated mask reference
  (gate Δ=0; preflight 60/60 decode + 60/60 selector). This BEATS the conservative ~2–4×
  hypothesis — because these are sparse-answer tasks (one span) in ignorable filler.
- **✅ DENSE AGGREGATION MEASURED (the decisive test): 1.0× — does NOT compress.** The
  `sum_scattered` family (answer = sum of 5 deposits scattered across ~20k; every addend page
  must survive) holds iso-quality only at 100% budget: attn 1.00→0.69→0.34→0.28→0.17 at
  100/50/25/12.5/6.25%, recent 0.00 throughout. Arithmetic is controlled (same 29 instances
  measured at every budget), so this is pure page-retention — the selector helps (0.69@50% ≫
  random 3%) but can't keep all 5 non-salient distributed addend pages under budget. This
  confirms the holographic thesis at the workload level: distributed-evidence answers can't be
  evicted. **Headline is now workload-dependent: 1× (dense) … 16× (sparse retrieval).**
  (recall_all, a non-arithmetic dense check, was unsolvable at full cache on 7B → 0 valid;
  RECALL_K lowered 6→3 for an OPTIONAL confirming run.)
- **⚠️ Remaining gap:** full-generation quality (perplexity over a long continuation, not just
  exact-answer single spans).
- **⚠️ Offline-selector caveat (also an upper bound):** the selector here scores pages by the
  attention of the *actual query token* over the full prompt — it sees the query. A deployed
  *streaming* evictor must decide what to drop **before** the query arrives (incrementally, as
  the cache fills), so it has strictly less information and will do no better. The measured
  budget is the best-case; an online H2O/SnapKV-style evictor is the realistic floor.
- No retrieval-head localization observed (multi_hop degrades gracefully with budget, not via
  a few load-bearing heads) — consistent with the holographic finding; latent question stays
  closed at 7B.

### Track C — compressibility ADMISSION CONTROLLER / safety wrapper [SCAFFOLD SHIPPED]

Given Track B's headline (1× dense … 16× sparse), a fixed serving budget **silently
over-compresses** the holographic regime (dense aggregation served at 25% → ~0.34 acc while the
caller was promised iso-quality). **exp021** builds the safety object that prices that gap
*before* serving: a per-region, **query-agnostic, write-side, CONFORMAL** certificate of
answer-level damage, wired as *admit-at-B / escalate / refuse*. Needs to beat **no compression
frontier** (it's a safety wrapper), and it directly **closes the offline-selector caveat above**
(the query-agnostic write-side version IS the realistic streaming floor).
- The decisive question (HORN-B): does the answer-level certificate **beat free attention-mass
  entropy**? Predicted **KILL k1** (exp006/011: attention-mass == KL-damage oracle ⇒ entropy
  likely already separates) — which is a *good* cheap outcome: it hands you a calibrated entropy
  admission gate (which no serving stack ships) and closes the question.
- **Status:** `exp021_admission.py` self-verified (selftest WIN when a mechanism is planted +
  negative control KILL k1 when entropy suffices; conformal coverage held both ways); exp020
  instrumented (additive per-instance `mass` dump). **Cut 1** = one instrumented exp020 re-run →
  `python exp021_admission.py --runs results/<run>/raw_*.jsonl --budget 0.25` (offline, local).
  See `SPEC_exp021_admission.md`.

### Recommended order
1. **Track B first** (cheap, decisive on the real number) if a 7B+ box is available —
   it sets honest expectations and de-risks Track A's value.
2. **Track A** to ship the capacity feature, sized to Track B's safe budget.
3. **Track C** (exp021 Cut 1) is one re-run + a local script — the cheapest of the three and it
   closes the offline-selector caveat; run it alongside Track B's re-runs.
4. If Track B reveals retrieval-head localization at scale, branch back to the latent
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
