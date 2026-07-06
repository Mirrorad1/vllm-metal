Experiment: exp015_logitlens — at what DEPTH does the answer crystallize in the latent stream?

Method: read the residual stream at the query position after each layer, project
through final-norm + unembedding (logit lens), watch the gold-answer token's argmax
status climb. Qwen2.5-0.5B (24 layers), N=30, ~3072-tok needle.

Result:
- argmax-gold-by-layer: 0.00 for layers 0-19, then **0.97 at layer 20**, 1.00 at 21-23.
  The answer SNAPS into place at layer 20 — a sharp, late transition.
- Crystallization depth: median **20/24 (83%)**, sd **0.2** (almost always exactly layer 20).

Verdict: **LATE + SHARP + CONSISTENT crystallization — no depth-pruning lever, but it
explains the mechanism behind every prior NULL.**
The answer is not present (as argmax) anywhere in the first 83% of the network; it
materializes abruptly at layer 20 and the last 4 layers (17%) merely confirm it. So:
- No depth lever: ~83% of the network depth is load-bearing (the answer isn't computed
  until layer 20); only the tiny post-crystallization tail could be lightened.
- MECHANISTIC INSIGHT: the answer is **computed late from holographically-distributed
  evidence**, not retrieved early into a localizable latent slot. This is WHY exp011-014
  found no sparse unit — there is nothing to find early/local because the answer doesn't
  exist as a stored state until layer 20, where it emerges as a function of the whole
  redundant KV. Distributed evidence in → late aggregation → answer.

Unified picture across the latent search (exp011-015): the answer is a LATE-COMPUTED
FUNCTION of HOLOGRAPHICALLY-DISTRIBUTED evidence. Not stored in any token/page (exp011),
kv-head (exp013), q-head (exp014), or early layer (exp015); only a modest ~4× low-rank
subspace structure exists (exp012). No exploitable sparse latent unit at any axis.

Confidence: high (sharp, sd 0.2, N=30). Scope: 0.5B single-needle COPY. A bigger model /
multi-hop task could crystallize differently (and is the only untested regime).

Raw: results/exp015_logitlens/{raw_results.jsonl, aggregate.json}
