Experiment: exp017_reclaim — physical KV reclamation prototype (the SYSTEMS lacuna)

Hypothesis: the one cell that survived all 16 prior experiments is SYSTEMS — every
prior result reclaimed ZERO physical pages (compact block table = read-only view).
Does physically gathering the kept pages and freeing the rest actually drop real
process memory, by how much, at what cost?

Method: allocate a model-shaped multi-layer KV cache (24 layers × [P, 16, 2, 64]
fp16 K&V) with real MLX allocation; measure mx.get_active_memory(); gather the kept
J pages into smaller per-layer arrays; free the full arrays (del + gc + clear_cache);
re-measure. Sweep contexts {2048..16384} × budgets {50..6.25%}. Cross-check decode
kernel latency (full vs compact) and numeric equivalence.

Result (all 16 cells):
- **reclaim_ratio = 1.00** everywhere (actual bytes freed ÷ would-be = 1.0): active
  memory drops by EXACTLY the logical fraction (P−J)/P. The "would-be" savings the
  whole series could only claim logically ARE physically realizable. E.g. L=16384 @
  6.25%: 201.3 MB → 12.6 MB live (188.7 MB freed); L=4096 @ 25%: 50.3 → 12.6 MB.
- Correctness: kernel numeric equivalence 1–4e-6 (exact) at every context.
- Costs measured honestly:
  * TRANSIENT PEAK during the gather = full + compact (both live briefly): 1.50× at
    50% budget down to 1.06× at 6.25%. ⇒ reclamation must be STAGED (layer-by-layer)
    to bound the peak, or it risks an OOM spike at large budgets.
  * Compaction copy: 1–5 ms, occasional (amortized over many decode steps) — negligible.
  * Decode-kernel latency full→compact: the modest, context-growing speedup from exp010
    (e.g. L=16384: 0.72→0.41 ms; L=8192 @25%: 0.57→0.16 ms).

Verdict: **PHYSICAL RECLAMATION REAL — the systems lacuna is BUILDABLE in this stack.**
MLX genuinely releases the freed bytes (ratio 1.00), correctness holds, and the costs
are manageable (stage the gather to bound transient peak; a few-ms occasional copy).
This is the FIRST positive on the physical-memory axis in the entire arc — the cell the
16-experiment latent search kept pointing to is real and achievable.

The remaining gap is NOT a research unknown — it is SAFETY INTEGRATION (F15): doing this
inside the live vLLM v1 allocator, freeing only blocks unreferenced across every
sequence / prefix-cache user / copy-on-write user (this prototype frees a standalone
sequence's cache; it does not touch refcounts/prefix/CoW). That is bounded engineering.

Scope: standalone MLX arrays in the real cache layout (faithful to MetalPagedKVCache's
[num_blocks, block_size, n_kv, d] fp16). Memory mechanism is representative; allocator
safety is the unbuilt part.

Raw: results/exp017_reclaim/{raw_results.jsonl, aggregate.csv, aggregate_summary.json}
