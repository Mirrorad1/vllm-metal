Experiment: exp013_heads — is the answer localized in latent HEAD-space though delocalized in token/page-space?

Hypothesis (pursuing "hidden latent states"): exp011 found no single PAGE is
load-bearing. Maybe the answer is carried by a few latent ATTENTION HEADS
(retrieval-heads) even though it is spread across all pages — which would relocate
the true lacuna from page/token-indexed to latent-head-indexed KV reduction.

Method: ablate one (layer, kv-head) at decode by zeroing its VALUES; measure
Δlogp(gold answer) + top1-flip → single-head load-bearing rate; head-budget curve
(keep top-k heads, zero rest) → how many latent heads the answer needs.
Qwen2.5-0.5B (24 layers × 2 kv-heads = 48 units), N=30, ~3072-tok needle.

Result:
- single-head flip rate = 0.0% (NO single kv-head ablation flips the answer);
  max single-head Δlogp = 0.02 nat (tiny — same scale as single-page ~0.0009).
- head-budget for 90% answer accuracy = 8/48 (~17%); curve {1:0.0, 2:0.0, 4:0.3,
  8:1.0, 16:1.0, ...}: need ~8 units, but no single one is essential.

Verdict: **LATENT HEAD-DENSE (redundant in head-space too).** The answer is
redundant across HEADS as well as PAGES — diffusion/redundancy is total at this
granularity. No sparse latent UNIT (page or kv-head) is individually load-bearing;
you need "enough" of any axis. Combined with exp011 (page-dense) and exp012
(mostly-full-rank subspace, modest 4× low-rank), the consistent picture is
HOLOGRAPHIC redundancy: the answer is distributed, not concentrated in any sparse
unit the field's operators could target.

Confidence: high at this granularity. CAVEAT: GQA collapses 14 query-heads into 2
kv-heads, so the unit is COARSE for the retrieval-head hypothesis — but note even
removing a whole kv-head (7 query-heads' worth of value contribution) never flips
the answer, which is STRONG evidence against concentrated head-groups; redundancy
across the 24 LAYERS likely dominates regardless. The one finer latent probe still
open: individual QUERY-head ablation (true retrieval heads) and residual-direction
ablation.

Raw: results/exp013_heads/{raw_results.jsonl, aggregate.json}
Next (exp014): finest feasible latent probe — per-query-head (retrieval-head)
ablation. If q-heads ALSO show no concentration, the holographic-redundancy
conclusion is locked and the true lacuna is definitively SYSTEMS (physical
reclamation), not latent structure. If a few q-heads are load-bearing,
retrieval-head-indexed KV is the latent lacuna.
