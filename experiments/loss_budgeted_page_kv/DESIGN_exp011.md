# exp011 — PRE-REGISTRATION (thresholds fixed before any result is seen)

## Question
Was the prior verdict "causal attention-mass ≈ KL-damage oracle, so loss-aware
page SCORING adds nothing" a **mean-of-opposites** — i.e. does attention match the
oracle on the bulk of pages while DROPPING the one load-bearing needle page on a
non-trivial tail of examples? The KV problem's value lives in that tail.

## Why the prior result is suspect (mean-of-opposites-guard)
The 0.549≈0.548-nat identity was a MEAN over a 0.5B model at short context. A mean
cancels a bimodal "fine on most pages / catastrophic on the needle" population.
Existing hint: @6.25% answer-acc was oracle 0.91 > attention 0.88 > online 0.86.

## Primary cell (ONE pre-registered confirmatory cell; everything else exploratory/Holm)
- Model: Qwen2.5-0.5B-Instruct (clean instruct; NOT R1-Distill — its think-trace
  makes the answer-step ill-defined). Scale/family arms deferred (no clean
  non-reasoning ≥1.5B cached).
- Context ≈ 3072 tokens ⇒ P≈192 pages. Budget 6.25% ⇒ J≈12, discretionary
  J−|floor|≈10 (floor={sink,recent,tail}). Adequate discretionary room.
- Single-needle, **paraphrased non-verbatim** query (so an induction head cannot
  trivially match a repeated trigger phrase ⇒ avoids false NULL), with same-surface
  distractor pages.
- Needle forced strictly OUTSIDE the floor (page index in [sink_pages,
  P−recent−1], and not page 0/tail) so it is never trivially retained.
- Decisive step = first answer token only (no horizon averaging).
- N = 200 accepted examples (full-cache must answer; report rejection rate).

## Endpoints (paired; report DISTRIBUTIONS + 2×2, never pooled means)
1. **Load-bearing certification (gates validity):** ablate ONLY the needle page(s)
   from the full cache → Δlogp(gold answer) and top1-flip. The needle page is
   "load-bearing" if removing it alone drops gold logprob materially (calibrated:
   top1 flips OR Δlogp ≥ 1 nat). If most needles are NOT load-bearing, the task is
   invalid → INCONCLUSIVE. This also decouples content from RoPE/position.
2. **Needle 2×2 (paired):** {oracle-keep, attn-keep} of the needle page at budget
   J → McNemar exact (binomial on discordant pairs). Report BOTH off-diagonals and
   NET discordance. (Oracle-keep is near-tautological since oracle=KL-damage; the
   informative quantity is attention's DROP rate and the net discordance.)
3. **Paired behavioral endpoint (primary, continuous, high-power):**
   Δlp = logp_answer(oracle-selection) − logp_answer(attention-selection) per
   example. Wilcoxon signed-rank + paired bootstrap CI of the median; sign split.
4. **Attention-aggregation sensitivity:** repeat needle-recall under sum-over-heads
   AND max-over-heads attention pooling. If max-pool recovers the needle where sum
   does not, the verdict is "the prior proxy was under-pooled" (actionable), not
   "loss-aware wins."
5. Triviality guards: recent/random/sink needle-recall must be < 0.20 (else reject).

## Pre-registered decision (primary cell only; α via Holm elsewhere)
Let valid = examples whose needle page is load-bearing (endpoint 1).
- **CONFIRM (mean-of-opposites real; loss-aware ALIVE in tail):**
  on `valid`, McNemar net discordance (oracle-keep & attn-drop) − (attn-keep &
  oracle-drop) has exact p < 0.01 AND the (oracle-keep & attn-drop) rate Wilson
  95% lower bound > 3%; corroborated by Wilcoxon on Δlp (endpoint 3) median > 0,
  p < 0.01. Holds under BOTH sum- and max-pool ⇒ strong CONFIRM; sum-only ⇒
  "under-pooled proxy" CONFIRM.
- **NULL / DEAD (loss-aware adds nothing even in tail):**
  needles certified load-bearing (endpoint 1 valid for ≥70% of accepted), AND
  McNemar p > 0.05 with net-discordance Wilson 95% UPPER bound < 5%, AND Wilcoxon
  Δlp CI contains 0, under BOTH poolings.
- **INCONCLUSIVE:** needles not load-bearing (task invalid) OR CIs too wide OR
  results disagree across poolings in a non-interpretable way.

## Falsifiers (pre-registered)
- F-needle-trivial: recent/random/sink needle-recall ≥ 0.20 → reject family, redo.
- F-leak: needle-page label derived ONLY from construction token span; asserted
  never imported by page_policies / never fed to a selector.
- F-straddle: needles crossing a page boundary are flagged; primary analysis on
  single-page needles; report straddle rate.
- F-position-not-content: if needle-page ablation Δlp is small (not load-bearing)
  the KL signal was position/mass-driven → INCONCLUSIVE, not NULL.
- F-power: if N=200 leaves the decisive CI straddling the decision band → report
  INCONCLUSIVE and the N needed.

## What each outcome means for the field analysis
- CONFIRM ⇒ the "scoring is saturated" conclusion was a mean artifact; the
  loss-aware axis (clusters 7/8) is alive precisely in the tail; the cheap online
  controller should use max-pool / damage-aware scoring, and the lacuna's
  smart-signal arm is revived.
- NULL ⇒ scoring genuinely dead even in the tail; pivot entirely to the
  physical-allocator + substrate gaps (the systems lacuna), not better signals.

---

## ADDENDUM (refinements forced by pre-run smoke tests; made BEFORE seeing any ΔKL outcome)

Two design flaws were caught by smoke-testing on ~10 examples and fixed before the
confirmatory run, with no threshold tuned to results:

1. **Single-page load-bearing certification is INVALID due to prefill information
   diffusion.** Ablating only the needle page at decode time changes the answer by
   Δlogp ≈ 0.0006 (load-bearing rate 0/14) — because downstream tokens' KV already
   absorbed the needle during prefill, so its information is redundantly encoded.
   (This is itself a finding: it is *why* eviction works, and a structural reason
   per-page importance is weak.) Single-page ablation therefore cannot gate
   validity. Needles also straddle 16-token pages (13/14).
   → REPLACED the primary endpoint with a **label-free, diffusion-robust paired
   test**: per example at the budget, ΔKL = KL(full‖attn-sel) − KL(full‖oracle-sel)
   and Δlp(gold). This directly asks whether the equal *means* hid a per-example
   *tail* (the mean-of-opposites signature), needs no needle label, and is immune
   to diffusion/straddle. Needle-recall + McNemar are kept as SECONDARY descriptive
   only. Validity gate is now: needle outside the floor + capture-sanity OK + the
   budget is *binding* (recent_kl ≫ oracle_kl).

2. **The primary budget must leave the oracle with headroom.** At 6.25% / P≈128 the
   oracle itself reaches KL≈0 (near-lossless), so no tail can exist there — a
   trivial NULL. The disputed prior number (0.549 nats) lived in a tighter regime.
   → Sweep budgets {6.25, 3.125, 1.5625, 0.78}% and AUTO-SELECT the primary as the
   *binding* budget whose oracle_kl_mean is closest to 0.5 (the disputed regime).
   Selection uses the oracle's ABSOLUTE KL only — never the ΔKL outcome under test.

Pre-registered tail thresholds (unchanged, set before seeing ΔKL):
CONFIRM if frac(|ΔKL|>1 nat) Wilson-95%-lower-bound > 0.02 (a real tail);
NULL if that upper bound < 0.02 AND |median ΔKL| < 0.02, under both poolings.
