# SPEC — exp021 conformal compressibility *admission controller*

## The object (one line)
A **per-region, write-side, query-agnostic certificate** that predicts the *answer-level*
damage of compressing a region at budget B, calibrated **conformally** to a distribution-free
coverage guarantee, and wired as an **admission gate**: *admit-at-B / escalate-budget / refuse*.
So the engine stops **silently over-compressing the holographic regime** (exp020: dense
aggregation served at 25% budget gives ~0.34 acc while the caller was promised "iso-quality").

## Why this cell survived (the two-horn trap, and the escape)
The adversarial review of this idea (the field-geometry hunt) nearly killed it on a dilemma:
- **sound → vacuous:** an honest *worst-case* bound must propagate a softmax perturbation
  through the remaining ~17% of nonlinear depth where the answer snaps in (~83%, exp015).
  Softmax has no tight global Lipschitz constant → the answer-level bound blows up → certifies
  "keep-exact everywhere" (1×).
- **useful → redundant:** tighten it and it collapses onto the attention-entropy signal, which
  exp020 already proved saturated (attention-mass == KL-damage oracle).

**Escape = conformal, not worst-case.** Calibrate on held-out decodes for a finite-sample,
distribution-free coverage guarantee instead of propagating Lipschitz bounds through depth.
That is sound (statistically), non-vacuous (tight to the empirical query distribution), and —
crucially — **answer-level**, so it is *not* the entropy signal. Whether it actually adds
discrimination over entropy is the experiment (H2 below), not an assumption.

## Hypotheses
- **H1 (soundness / non-vacuity):** a split-conformal predictor of answer-level damage,
  calibrated on held-out queries, licenses >1× compression on a non-trivial fraction of
  requests while keeping realized **silent-failure ≤ α** on a genuinely held-out distribution.
- **H2 (the decisive one — HORN-B / discrimination):** the **answer-level** certificate beats a
  pure **attention-entropy** threshold on the (silent-failure × efficiency) frontier. *This is
  the whole mechanism question.* If H2 fails you have a deployable calibrated-entropy admission
  gate (useful, unshipped today) but **not** a new mechanism.

## Mechanism (precise)
For region/request `x`, budget `B`:
1. **Write-side, query-agnostic features `φ(x)`** (computed at prefill, no query token — the
   novel amortizable version): per-page attention-mass `entropy`, `effective_page_count =
   exp(H(normalized mass))`, top-k mass concentration, gini, `P` (pages), `L`. (The mass is
   already computed in exp020; exp021 only needs it dumped — see Instrumentation.)
2. **Damage predictor** `D̂ = f(φ(x), B)` → P(answer breaks at B).
3. **Split conformal (selective classification):** on a calibration set, pick the admit
   threshold `t` on `D̂` so that an upper confidence bound (Clopper–Pearson, 1−δ) on the break
   rate of the admitted set is ≤ α. Certificate = "admit ⇒ break-rate ≤ α w.p. ≥ 1−δ."
4. **Gate:** admit at B iff `D̂(x) ≤ t`; else escalate B (try a larger budget) or **refuse**.

The guarantee is honestly **over the calibration query distribution** — not adversarial
worst-case. That is the feature (it dodges the vacuity horn), and the explicit limit (a shifted
query distribution voids coverage; report it).

## Baselines
- **B0 naive fixed-B:** serve everything at B (today's serving stack). Dense aggregation
  silently fails.
- **B1 entropy-threshold gate:** admit iff `effective_page_count ≤ τ`. **The HORN-B opponent —
  the free signal.** Calibrated by the *same* conformal procedure for a fair frontier.
- **B2 query-conditional certificate:** `φ` includes the query (tighter, decode-time, not
  write-side-amortizable) — achievability ceiling for the write-side version.
- **B3 oracle gate:** uses true damage `D(x,B)` — upper bound on any gate.

## Metrics
1. **Coverage:** break rate among *admitted* requests (target ≤ α) on held-out queries.
2. **Silent-failure rate:** B0 vs controller (controller should drive it to ~α).
3. **Efficiency retained:** fraction still admitted at the small budget (must keep the sparse
   16× wins — a refuse-everything gate is worthless).
4. **HORN-B discrimination:** controller vs B1 — efficiency at **matched guaranteed α**; and
   Spearman ρ(`D̂`, entropy). Win = more efficiency at equal safety **and** ρ < ~0.7.

## Win / Kill / Engineering-vs-mechanism tell
- **Win (new mechanism):** coverage ≤ α held-out, retains most sparse-compression efficiency,
  **beats B1** at matched α with ρ < ~0.7.
- **Kill k1 (HORN-B / redundant):** B1 matches it (ρ ≥ ~0.9, same frontier) → "calibrated
  entropy threshold." Deployable, **not novel.**
- **Kill k2 (vacuous):** to hit coverage you must refuse ~everywhere → conformal interval too
  wide → vacuity horn relocated to calibration.
- **Kill k3 (unsound):** coverage fails on genuinely held-out queries → the query-agnostic
  write-side version doesn't hold (fall back to B2 query-conditional).
- **Tell:** if the controller's admit/refuse decisions are rank-identical to the entropy
  threshold, it's an entropy gate with a conformal wrapper — engineering, not mechanism.

## Data & harness reuse — the cheap first cut
The first decisive cut runs **offline on exp020 outputs**: `D(instance, B) = 1 − correct@B` is
already measured per `(family, instance, budget, policy)`; `φ` comes from the per-instance
attention mass. Only **one instrumented exp020 re-run** is needed first (the old jsonl didn't
save `mass`) — after that, Cut 1 is pure offline analysis, local, no new GPU forwards.

### Instrumentation (additive, no-op to validated numbers)
`exp020_quality_cuda.py` gains, per accepted instance: a stable `seed` id on every record, and
one `{"event":"instance", ..., "mass":[...]}` line carrying the query-agnostic feature. Existing
budget/correct records are **byte-identical**; the summary builder is untouched.

## Phasing
1. **Cut 1 (local, ~days):** offline on the instrumented exp020 jsonl — predictor +
   split-conformal + gate, HORN-B vs entropy. *If k1 fires here, stop cheaply — it's an entropy
   gate.* Validate the harness first with `python exp021_admission.py --selftest` (synthetic data
   with a known sparse/dense split **and** a planted adversarial subset where entropy is
   uninformative — asserts the harness can both (a) reproduce coverage and (b) *detect* a real
   mechanism when one exists).
2. **Cut 2 (RunPod):** held-out **query-distribution** runs per region (multiple queries per
   context) for the real coverage test (H1) + the query-agnostic vs query-conditional gap (B2).
3. **Cut 3 (RunPod):** mixed-workload serving sim — silent-failure & efficiency vs B0 (the
   deployable headline).

### Self-proving preflight (the exp020 discipline)
- gate at B=1.0 admits ~all (a no-op); assert.
- per-instance record count consistent (`len(budgets) × n_policies`); assert.
- `--selftest` asserts: coverage ≤ α on synth held-out; full-feature gate strictly beats the
  entropy gate on the planted-adversarial split (proves the HORN-B comparison is *sensitive* —
  it won't falsely report "entropy suffices" when it doesn't).

## Scope / honest limits
- Conformal coverage holds **over the calibration query distribution**; a shifted/adversarial
  query distribution voids it. This is the price of escaping the vacuity horn — state it.
- The **query-agnostic, write-side** version is the novel, hard one. If it fails coverage (k3),
  the query-conditional B2 is the honest, still-useful fallback (decode-time, not amortizable).
- This experiment also **closes exp020's offline-selector caveat**: the query-agnostic write-side
  certificate is exactly the realistic streaming floor the FINDINGS flagged as unmeasured.

## Novelty boundary (honest)
- Win → **new-combination** (conformal selective prediction + answer-level damage + write-side
  amortized admission gate for eviction/synthesis; closest prior art = Runtime-Certified
  Bounded-Error Quantized Attention, which is per-query / quant-only / FP16-fallback).
- Tie (k1) → **engineering / useful closure** (a calibrated entropy admission gate no serving
  stack ships).
- Either way it ships; neither has to beat a compression frontier (it's a safety wrapper).
