Experiment: exp014_qheads — finest latent probe (individual query-head / retrieval-head ablation)

Hypothesis: maybe the answer is carried by a few latent QUERY-HEADS (retrieval
heads) even though no page (exp011) or kv-head (exp013) is load-bearing.

Method: zero each individual (layer, query-head)'s attention OUTPUT at the decode
step (24×14 = 336 latent units); Δlogp(gold answer) + flip; q-head-budget curve.
Qwen2.5-0.5B, N=20, ~3072-tok needle.

Result:
- single query-head flip rate = 0.00% (no individual q-head ablation flips the answer);
  max single-qhead Δlogp ≈ 0.005 nat (even smaller than kv-heads' 0.02).
- q-head budget for 90% accuracy = 256/336 (76%). Curve {≤16: 0.0, 32: 0.25, 64: 0.30,
  128: 0.60, 256: 1.0}: you need the MAJORITY of query heads; no sparse subset works.

Verdict: **HOLOGRAPHIC LOCKED — no sparse latent unit at ANY granularity.**
Even at the finest latent unit, retrieval is NOT localized; it is distributed across
the majority of query heads, mirroring pages (exp011) and kv-heads (exp013). The
answer is holographically encoded across token × head × layer; cross-layer redundancy
dominates. The "true lacuna through hidden latent states" search returns NULL: the
latent states hide no exploitable sparse structure beyond the modest ~4× low-rank
subspace (exp012). The only genuinely-empty cell is SYSTEMS (physical reclamation).

Confidence: high for this regime. SCOPE CAVEAT (honest, and the only thing that could
overturn it): 0.5B + single-needle COPY at maximal diffusion. The retrieval-head
literature finds sparse heads mainly in 7B+ models on harder multi-hop/long-context
tasks — untested here (no local ≥7B). That is the sole regime where a latent lacuna
could reopen.

Raw: results/exp014_qheads/{raw_results.jsonl, aggregate.json}
