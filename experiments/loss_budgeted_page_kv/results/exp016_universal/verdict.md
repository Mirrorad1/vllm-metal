Experiment: exp016_universal — is the low-rank KV subspace UNIVERSAL (shared across prompts) or prompt-specific?

Method: fit a fixed per-(layer, kv-head) low-rank basis on 8 CALIBRATION prompts;
project 15 HELD-OUT prompts' K/V onto that fixed basis at rank r; answer accuracy vs
r. Compare to each prompt's OWN (per-example) low-rank basis. Qwen2.5-0.5B, d=64.

Result:
- universal-basis answer accuracy by rank: {64:1.0, 48:1.0, 32:1.0, 24:0.0, 16:0.0}.
- per-prompt-basis accuracy by rank:        {64:1.0, 48:1.0, 32:0.87, 24:0.0, 16:0.0}.
- universal r*(0.9) = 32/64; per-prompt r*(0.9) = 48/64.

Verdict: **MODEST POSITIVE — a universal (cross-prompt) low-rank KV subspace EXISTS,
but it is small (~2×) and corresponds to an existing method class.**
- The answer-relevant KV directions are SHARED across prompts: a fit-once rank-32 basis
  preserves answers on held-out prompts as well as (or slightly better than) each
  prompt's own SVD. This is the FIRST non-NULL latent structure in the search (exp011-015
  were all holographic/NULL) — a genuine hidden latent regularity.
- BUT the magnitude is modest: rank 32/64 = a 2× per-element compression via a universal
  basis. This is exactly the premise behind the existing static-low-rank-KV / learned-
  projection / MLA cluster (project KV into a fixed latent subspace) — NOT a novel lacuna.
- Honest caveat: n=15, so "universal beats per-prompt at rank 32 (1.0 vs 0.87)" is within
  noise; the robust claim is only "universal ≈ per-prompt ≈ 2×, and the subspace is shared."
- Note the universal subspace (~2×, all tokens) and exp012's per-prompt subspace (~4× on
  SELECTED tokens) are different cuts of the same modest low-rank structure; both stack with
  sparsity (the dominant lever, owned by eviction).

Closing the latent search (exp011-016): across 6 latent axes — token/page, per-prompt
subspace, universal subspace, kv-head, q-head, depth — the answer is HOLOGRAPHICALLY
distributed (no sparse unit at any axis), computed LATE (layer 20/24) from that distributed
evidence, with the ONLY exploitable structure being a modest ~2-4× low-rank subspace that
is partly universal — and that structure is already owned by existing low-rank-KV/MLA
methods. NO NOVEL latent lacuna exists at this scale. The genuinely-empty cell remains
SYSTEMS (physical reclamation). Scope caveat unchanged: 7B+/multi-hop untested.

Raw: results/exp016_universal/{raw_results.jsonl, aggregate.json}
