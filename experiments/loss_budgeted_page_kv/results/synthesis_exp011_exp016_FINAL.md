# FINAL synthesis — the "hidden latent states" lacuna search (exp011–016)

Six experiments probed every major axis of the model's latent representation for a
*true lacuna*: a sparse, exploitable structure that page/token-indexed methods miss.
The search is comprehensively complete. Verdict: **no NOVEL latent lacuna at this
scale.** The latent states are holographically redundant, with only a modest,
already-owned low-rank structure.

| exp | latent axis probed | result |
|---|---|---|
| 011 | token / KV page | holographic — no single page load-bearing (Δlp≈0.0009); KL≠answer |
| 012 | per-prompt subspace (rank) | sparsity-dominant; modest ~4× low-rank on selected tokens |
| 013 | kv-heads (coarse) | holographic — no single kv-head (flip 0%); need ~8/48 |
| 014 | query-heads (finest unit) | holographic — no single q-head (flip 0%); need 256/336 (76%) |
| 015 | depth (logit lens) | answer crystallizes LATE & sharp (layer 20/24); computed, not stored |
| 016 | universal (cross-prompt) subspace | MODEST POSITIVE — shared ~2× low-rank basis (rank 32/64) = existing low-rank-KV/MLA |

## The unified picture (mechanism, not just phenomenon)

**The answer is a LATE-COMPUTED FUNCTION of HOLOGRAPHICALLY-DISTRIBUTED evidence.**
- *Distributed:* every fact is redundantly re-encoded across many tokens (prefill
  diffusion) and recomputable at many heads/layers — so no token, page, kv-head, or
  q-head individually carries it (exp011/013/014). You must keep "enough" of *any* axis,
  not the *right* few.
- *Late-computed:* the answer is not the residual-stream argmax anywhere in the first 20
  of 24 layers, then snaps into place at layer 20 (exp015). It is *computed* by aggregating
  the distributed evidence, not *retrieved* from a localizable slot — which is precisely
  *why* no sparse unit exists to find.
- *Only structure:* a modest low-rank subspace — ~4× on answer-relevant (selected) tokens
  (exp012), ~2× universal across prompts (exp016). Real, partly shared, but small, and
  already exploited by the static-low-rank-KV / MLA cluster. Not novel.

## Why this answers the lacuna question

The recursive search hypothesized the empty cell was a *representation* problem (a smarter
latent unit to target). Six experiments falsify that here: the levers are token-SPARSITY
(owned by eviction) × precision (quant) × a modest low-rank subspace (owned by low-rank-KV
/ MLA) — orthogonal, known, stackable. The "hidden latent states" hide no extra sparse
structure. The surviving hidden axis (source-locality vs smear) collapsed fully to its
SMEAR pole: value is everywhere and nowhere, computed late.

**The one genuinely-empty cell that survives all six experiments is SYSTEMS:** physically
reclaiming the bytes that logical dropping / low-rank factoring frees, reference-count /
prefix / copy-on-write-safe (the unowned-allocator seam from the original field map). That
is an engineering/ownership lacuna, not a latent-representation one.

## Scope (the only thing that could reopen it)

All six: 0.5B + single-needle COPY retrieval at maximal diffusion. The sparse-retrieval-head
/ localized-latent literature lives at 7B+ on multi-hop COMPOSE tasks (where the final
answer cannot be redundantly pre-encoded). Untested here (no local ≥7B GPU). That is the
sole regime that could overturn HOLOGRAPHIC and reveal a localized latent lacuna — and is
the correct next experiment if a larger machine becomes available.

## Loop decision

The mandate — *find a true lacuna through hidden latent states* — is comprehensively
answered: **NULL for a novel latent lacuna at this scale**, with a full mechanistic
explanation. Further latent probes would re-confirm, not discover. The autonomous latent
loop is therefore concluded. The two real continuations both require a NEW mandate:
(A) re-run the finest probes at 7B+/multi-hop (different hardware), or
(B) pivot to the SYSTEMS lacuna (allocator engineering — a different kind of work).
