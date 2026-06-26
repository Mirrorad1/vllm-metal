Experiment: exp011_tail_vs_mean_loss_structure

Hypothesis: the prior "attention-mass ≈ KL-damage oracle, scoring adds nothing"
verdict was a MEAN-OF-OPPOSITES — attention matches the oracle on the bulk of
pages but DROPS a load-bearing needle the oracle keeps on a non-trivial tail,
flipping the answer. (Pre-registered in DESIGN_exp011.md; mean-of-opposites-guard.)

Intervention: Qwen2.5-0.5B-Instruct, ~3072-token single-needle (paraphrased,
non-verbatim) retrieval, N=150, budgets {6.25, 3.125, 1.5625, 0.78}%. PRIMARY
endpoint = per-example paired ΔKL = KL(full‖attn-sel) − KL(full‖oracle-sel) AND
the DECISIVE answer-level paired test (McNemar: does oracle rescue answers
attention loses). Primary budget auto-selected as the binding budget where the
oracle has KL headroom (~0.5). [The originally pre-registered single-page
load-bearing gate was found INVALID by smoke test — prefill information diffusion
makes single-page ablation Δlp≈0 — and replaced before any ΔKL was seen; see
DESIGN addendum.]

Baselines: recent / seeded_random / sink_recent (triviality guards), plus
attn_max (max-over-heads pooling) for aggregation sensitivity.

Metrics / falsifiers: see DESIGN_exp011.md. Decisive gate is the ANSWER-level
McNemar, not the full-distribution ΔKL tail (verify-delegated-verdicts: a KL tail
that does not reach the answer is not a behavioral effect).

Result (orchestrator re-derived the decisive numbers from raw_results.jsonl):

Per budget (n=150 each), mean KL(full‖selection):
| budget | oracle | attn_sum | attn_max | recent | answer-acc all selectors |
|---|---|---|---|---|---|
| 6.25%   | 0.000 | 0.000 | 0.000 | 0.823 | 1.00 (page choice irrelevant; oracle saturated) |
| 3.125%  | 0.000 | 0.000 | 0.001 | 0.861 | 1.00 |
| **1.5625%** | **0.318** | **0.810** | 1.035 | 0.930 | **oracle 1.00, attn_sum 1.00, recent 1.00, attn_max 0.91** |
| 0.78%   | 1.954 | 1.954 | 1.954 | 1.954 | 0.50 (everyone fails together) |

Primary cell = 1.5625% (the only binding budget with oracle KL headroom):
- KL tail: frac(|ΔKL_attn_sum−oracle| > 1 nat) = **0.140**, Wilson 95% [0.093, 0.205];
  median ΔKL = 0.700 (oracle better), one-sided (frac< −1 nat = 0.000); Wilcoxon
  p = 4.1e-15. So the prior "equal KL means (0.549≈0.548)" is REGIME-SPECIFIC —
  at this tight budget the oracle preserves the next-token distribution and gold
  logprob materially better than attention.
- **ANSWER McNemar: oracle-correct&attn-wrong = 0, attn-correct&oracle-wrong = 0,
  p = 1.0, net = 0.000.** Answer accuracy identical (oracle = attn_sum = recent =
  1.00). Oracle NEVER rescues an answer attention loses, at ANY budget.
- Mechanism: prefill information diffusion — single-needle-page ablation Δlp(answer)
  ≈ 0.0009; recent-only (last ~3 pages) answers correctly until the 0.78% cliff.
  The answer is redundantly encoded across pages, so page choice barely affects
  answer correctness ⇒ no behavioral tail to exploit.
- Aggregation: max-over-heads attention is WORSE (kl 1.035, acc 0.91), not better —
  the "under-pooled proxy" hypothesis is refuted; sum-over-heads is the good proxy.

Verdict: **KL-ONLY (behaviorally inert)** — a refinement of NULL.
The mean-of-opposites concern is FALSIFIED at the level that matters (answer
correctness): attention ≡ oracle behaviorally, even in the tail, even at tight
budgets. The prior "equal KL means" was regime-specific (oracle does preserve
full-distribution fidelity better at tight budgets — a real, one-sided tail), but
that divergence is behaviorally inert and does not improve retrieval. Loss-aware
page SCORING is dead for this task class; the blocker is prefill diffusion +
systems, not signal quality.

Confidence: high for the answer-level NULL (McNemar p=1, n=150, replicated across
all 4 budgets). The KL-fidelity tail is rock-solid (p=4e-15). SCOPE LIMIT
(honest): 0.5B, single-needle COPY retrieval, semi-synthetic. Multi-hop reasoning
that must COMBINE distant facts (less diffusion redundancy), and 7B+/GQA scale,
are untested and are the only regimes that could revive a behavioral tail.

What worked: paired label-free endpoint; answer-level decisive gate caught a
would-be laundered "CONFIRM" from the KL tail; auto-selected binding+headroom cell.
What failed: the hypothesis (no behavioral mean-of-opposites here); single-page
load-bearing certification (diffusion).

Raw result path: results/exp011_tail_vs_mean/{raw_results.jsonl, aggregate.json, aggregate.csv}
Next experiment: see next_step.md.
