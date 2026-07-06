Experiment: exp019_capacity — quantify the throughput/capacity gain of lossy KV reclamation

Goal: exp018 verified the win is CAPACITY-for-reuse in a static pool. This simulates
how much. Parametric model of a fixed KV pool; uses the SHIPPED safety gate
(reclaim_safety.reclaim_accounting) on sampled concurrent batches so the floor,
prefix-sharing, and refcount rules are real, not idealized.

Setup: Llama-3-8B (per-token KV = 128 KB), 32 GB KV pool = 16,384 blocks. Sweep
context {2K,8K,32K} × budget {100..6.25%} × shared-prefix {0,25%}.

Result — concurrent-sequence multiplier vs status quo (lossless), no prefix:
| budget | multiplier | quality |
|---|---|---|
| 50%   | 2.0× | iso-quality (safe) |
| **25%** | **4.0×** | **iso-quality (general-workload threshold)** |
| 12.5% | 8.0× | RISK (degrades on hard tasks) |
| 6.25% | 16.0× | RISK |

Concrete (8K context, 32 GB pool): status quo fits **32** concurrent sequences;
at 25% budget it fits **128** (4×); at 12.5%, 256 (8×, quality risk).

Verdict: **~2–4× capacity/throughput at iso-quality (25–50% budget), 8–16× at
aggressive budgets with quality risk** — model-independent multiplier (≈1/budget,
since at useful context lengths P ≫ the always-keep floor). The shipped safety gate
correctly excludes shared-prefix blocks from reclamation in every cell (grounding
ok=True), so the gain is computed under the real safety constraints.

Honest bounds (verify-delegated-verdicts on my own sim):
- The ≈1/budget multiplier is the LONG-CONTEXT STEADY STATE (every sequence at full
  context, reclaimed to budget). Mixed-length / short-context workloads gain less —
  reclamation only helps once a sequence is long, which is exactly the KV-pressured
  regime that matters.
- It is a CAPACITY multiplier; throughput tracks it only up to the COMPUTE ceiling.
  On Apple-Silicon long-context decode (memory-bandwidth / KV-capacity bound),
  concurrency is the binding lever, so the capacity gain largely converts to
  throughput — but the honest cap is "admit up to N× more," not "N× faster compute".
- Quality: 25% is the literature's general-workload iso-quality budget; aggressive
  budgets (8–16×) degrade on hard/non-redundant tasks (and our latent arc showed you
  can't pick smarter pages to push it).
- Floor saturation: at very short context or very tight budget the always-keep floor
  caps the gain (footprint can't go below ~3 blocks).
- REALIZATION: this is the potential the exp018 foundation enables; it is realized
  only after the scoped UPSTREAM reclamation wiring (mid-sequence block free).

Raw: results/exp019_capacity/{raw_results.jsonl, aggregate.csv, aggregate_summary.json}
Plot: plots/exp019_capacity.png
