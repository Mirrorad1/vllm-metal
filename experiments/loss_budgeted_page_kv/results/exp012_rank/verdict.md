Experiment: exp012_rank — is the prefill-diffusion redundancy in the STORED BYTES or the COMPUTE GRAPH?

Hypothesis: the recursion's decisive fork. exp011 showed page choice barely affects
the ANSWER (token-sparsity redundancy is abundant). Is the redundancy ALSO geometric
(retained KV in a low-dim subspace → a byte-level low-rank/dedup factorizer can reclaim
bytes, STACKING on eviction → the new lacuna is fillable), or purely token-identity +
compute-graph (near-full-rank KV → no stored-byte operator beyond dropping → lacuna empty,
move = recompute)?

Intervention (Qwen2.5-0.5B, ~3072-tok needle, N=40, selection at 6.25% where it is
near-lossless): (1) effective rank of cached K/V; (2) BYTE-MATCHED head-to-head —
same bytes spent on (S) token-selection vs (F) global low-rank r_eq over ALL tokens;
(3) STACKING — low-rank-project the SELECTED tokens' KV to rank r, find smallest r*
preserving the answer.

Result:
- Effective rank (of 64): V = 48.3, K = 40.3. Moderate, NOT ≪ d — no extreme low-rank.
- BYTE-MATCHED head-to-head: S_acc = 1.00 vs F_acc = 0.00 (r_eq=4, KL 7.82). At equal
  bytes, TOKEN-SPARSITY utterly dominates global low-rank. The budget is best spent
  choosing WHICH tokens, not on a low-rank subspace.
- STACKING (answer-acc vs rank, on the selected tokens; selection alone is lossless KL=0):
    r:   64   48   32   24   16   12    8    4
    acc: 1.00 1.00 1.00 1.00 0.95 0.40 0.00 0.00
  Sharp cliff: r* ≈ 16 (of 64) — a clean ~4× low-rank lever that STACKS multiplicatively
  on eviction; collapses hard below 16.

Verdict: **SPARSITY-DOMINANT + modest low-rank stack (partially fillable).**
The bytes-vs-graph fork resolves to BOTH structures, very unequal:
- The DOMINANT redundancy is token-SPARSITY (drop 94% of pages, lossless; the SAME bytes
  spent on global low-rank are catastrophic). Existing eviction already harvests this — it
  is not new headroom.
- A SECONDARY, real, geometric low-rank lever (~4×, rank-16 of 64) exists and composes
  multiplicatively on top of selection. So the new lacuna (a query-conditioned smear-aware
  byte reclaimer) is PARTIALLY fillable — but the lever it adds is the ~4× low-rank stack,
  which is exactly what the existing low-rank-KV / MLA cluster (cluster 4) already targets.
  The lacuna largely COLLAPSES into cluster 4; its only novelty is doing the factorization
  query-conditioned on the RETAINED set rather than statically.

This reproduces, from first principles on one model, WHY production stacks
eviction × quantization × low-rank: they exploit ORTHOGONAL redundancy structures
(which tokens / which bits / which subspace), each a different ~2-16× lever.

Confidence: high for the 0.5B single-needle COPY regime (sharp, consistent curves, N=40,
selection lossless so rank-truncation is the sole degradation). Caveats: (a) low-rank was
applied to POST-RoPE K (mixes content+position; a real low-rank-KV scheme factors pre-RoPE
or learns the projection — V, no RoPE, is the clean content measure at eff-rank 48/64);
(b) scope = 0.5B single-needle; multi-hop COMPOSE / 7B+/GQA (lower diffusion) untested and
could shift both the sparsity dominance and the low-rank rank.

Raw: results/exp012_rank/{raw_results.jsonl, aggregate.json}
Next: the dominant lever (sparsity) is owned by existing eviction; the residual lever
(~4× low-rank) is owned by cluster 4. So the KV-SCORING/structure question is now well-
characterized as "stack orthogonal known operators," not an open lacuna. The remaining
genuinely-open cell is the SYSTEMS one (physical reclamation of the dropped/factored bytes,
ref-count/prefix-safe) — return to that, or test the multi-hop/7B scope caveat.
