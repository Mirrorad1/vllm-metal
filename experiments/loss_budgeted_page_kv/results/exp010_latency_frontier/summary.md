Experiment: exp010_latency_frontier

Hypothesis: at sufficiently long context, selected-page decode attention crosses
the latency break-even point (net gated < net full, including controller cost).

Intervention: measure real `paged_attention_primitive` decode latency, full (P
pages) vs gated (J pages at 6.25%), and compare to full attention work per step.

Baselines: full_pages kernel latency.

Metrics: per-call kernel p50/p95 (warmup excluded), interleaved A/B over 150
iters to cancel GPU clock drift; honest accounting that attention is only a
FRACTION of a decode step (MLP/projections/norms also run).

Falsifiers: F5 (gated end-to-end ≥ full once controller/table/sync/copies
included), S2 (p95 worsens), S3 (kernel-only speedup ≠ end-to-end).

Result (ORCHESTRATOR-RE-RUN, correcting an earlier laundered auto-verdict):

The auto-generated first pass reported "POSITIVE" because `any net gated < net
full` was true — but that rested on a noisy cold-start baseline (full/step 9.10 ms
at L=512 yet 3.94 ms at L=1024; attention latency cannot DECREASE with context).
Re-measured with interleaved A/B, warmup, 150 iters:

| L | full p50 (µs) | gated@6.25% p50 (µs) | kernel speedup | full p95 | gated p95 |
|---|---|---|---|---|---|
| 512  | 245.7 | 218.4 | 1.13× | 302.0 | 254.2 |
| 1024 | 169.6 | 145.9 | 1.16× | 189.4 | 158.2 |
| 2048 | 159.3 | 140.6 | 1.13× | 176.4 | 157.0 |
| 4096 | 167.2 | 136.7 | 1.22× | 192.5 | 162.7 |
| 8192 | 230.8 | 163.8 | 1.41× | 410.4 | 229.3 |

- All-pages numeric equivalence: max abs err ≈ 4e-6 at every L (F2 not triggered).
- Per-call kernel speedup is REAL but MODEST: 1.13–1.41×, growing with L (kernel is
  overhead-bound at short context; the KV-walk fraction grows with L). Even at
  6.25% pages and 8K context the kernel is only 1.41× faster — far below the 16×
  the token reduction implies, because dispatch/fixed overhead dominates.
- This is KERNEL-ONLY. It excludes the controller's attention-mass extraction,
  block-table construction, the compacted-K/V gather, and the fact that attention
  is only a fraction of a full decode step (MLP/proj/norm unaffected). End-to-end
  speedup is therefore strictly less than the kernel ratio and is NOT demonstrated.
- Physical memory is unchanged (exp009, shadow mode).

Verdict: **SYSTEMS WALL** — kernel-only speedup is modest and does not translate to
a demonstrated end-to-end physical-memory-or-latency win (F5 not cleared, F4 holds).
NOT latency-positive. The earlier auto-"POSITIVE" is retracted as a measurement
artifact (verify-delegated-verdicts: orchestrator re-ran the decisive measurement).

Confidence: high (interleaved A/B, 150 iters; monotone-in-L trend is physical).

What worked: numeric equivalence; a real kernel speedup that grows with context
(suggests a genuine win exists at much longer context / larger batch).
What failed: no end-to-end demonstration; overhead-bound at tested scale; memory
not freed.

Raw result path: results/exp010_latency_frontier/systems_remeasured.csv
Next experiment: (a) real physical reclamation so memory actually drops;
(b) batched / ≥16K context where the kernel ratio is large enough to survive
end-to-end dilution; (c) an incremental O(log P) controller (exp008) so controller
cost cannot eat the kernel saving.
