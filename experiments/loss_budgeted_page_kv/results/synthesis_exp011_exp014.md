# Synthesis — exp011 … exp014: the "hidden latent states" lacuna search returns NULL

Four experiments hunted a *true lacuna through hidden latent states* — a sparse,
exploitable structure in the model's internal representation that page/token-indexed
methods miss. The search is now complete and the answer is a clean, scoped NULL.

| exp | latent granularity probed | single-unit load-bearing? | budget to keep answer | verdict |
|---|---|---|---|---|
| exp011 | logical KV pages | NO (single-page Δlp ≈ 0.0009 nat) | ~few % of pages | KL-only / behaviorally inert |
| exp012 | KV subspace (rank) | — (V eff-rank 48/64) | sparsity ≫ low-rank; +4× low-rank stack | sparsity-dominant + modest low-rank |
| exp013 | kv-heads (24×2) | NO (flip 0%, max 0.02 nat) | ~8/48 heads | head-dense (holographic) |
| exp014 | query-heads (24×14) | NO (flip 0%, max 0.005 nat) | 256/336 heads (76%) | HOLOGRAPHIC LOCKED |

## The finding

**The answer is holographically distributed.** No single page, no kv-head, and no
query-head individually carries it; ablating any one moves the gold-answer logprob by
~0 nat. Yet you must retain "enough" of *any* axis (≈6% of pages, ≈17% of kv-heads,
≈76% of q-heads) or it collapses all at once. The only concentrated structure anywhere
is a modest ~4× low-rank subspace (exp012, rank-16 of 64) — and that lever is already
owned by the existing low-rank-KV / MLA cluster.

Mechanism: prefill information diffusion. As the model reads, each fact is re-encoded
into the KV of many downstream tokens and recomputable at many heads/layers, so the
representation is redundant in every direction a compression operator could address.

## Consequence for the lacuna hunt

The recursive search assumed an empty cell would be a *signal/representation* problem
(a smarter unit to select). Four experiments falsify that at this scale:
- Page-importance scoring: behaviorally inert (exp011).
- Subspace low-rank: real but modest and already-owned (exp012).
- Latent head localization (retrieval heads): absent — distributed, not sparse (exp013/14).

So the "hidden latent states" hide no exploitable sparse structure here. The surviving
hidden axis (source-locality vs smear) collapses to its SMEAR extreme: value is
everywhere and nowhere. **The only genuinely-empty cell that survives all four results
is SYSTEMS: physically reclaiming the bytes that logical dropping/factoring frees,
reference-count/prefix/copy-on-write-safe — the unowned-allocator seam from the original
field analysis.** That is an engineering/ownership lacuna, not a latent-representation one.

## Honest scope (the one thing that could reopen it)

All four are 0.5B + single-needle COPY retrieval at maximal diffusion. The retrieval-head
/ sparse-latent literature finds concentration mainly at 7B+ on multi-hop COMPOSE tasks
(where the final answer cannot be redundantly pre-encoded). That regime is untested (no
local ≥7B GPU). It is the sole experiment that could overturn HOLOGRAPHIC LOCKED — and it
is the correct next test if a larger machine becomes available.

## Loop decision

The loop's stated goal ("find a true lacuna through hidden latent states") is **answered:
NULL at this scale.** Continuing to probe finer latent units would only re-confirm it.
Two off-mandate continuations exist, each needing a new instruction: (A) the 7B+/multi-hop
scope test (different hardware), or (B) pivot to the SYSTEMS lacuna (a different kind of
work — allocator engineering, not representation probing). Loop halted here with a clean
negative result rather than manufacturing further latent experiments.
