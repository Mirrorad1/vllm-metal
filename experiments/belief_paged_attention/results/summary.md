# FINAL REPORT — Belief-Gated PagedAttention

## Verdict: **ORACLE-ONLY POSITIVE**

A belief-conditioned page selector *can* attend to a small subset of KV pages
while preserving model behavior **when given prefix-derived task state (oracle)**
— at 25% of pages the model keeps 96% answer accuracy and KL(full‖gate)=0.046.
But the **online inferred belief state does not beat a trivial keyword baseline**
(falsifier #1) and does not approach the oracle (falsifier #4). The page-
selection *mechanism* and its numerical correctness are real and proven; the
*inference* hypothesis is not supported by this implementation. At the tested
context lengths there is also **no net latency win and no real memory reduction**
(shadow mode).

Model: `Qwen2.5-0.5B-Instruct` (N=24, H_q=14, H_kv=2, d=64, fp16), greedy/teacher-
forced. 6 task families × 3 context sizes (≈512/1024/2048 tok) × 5 seeds × 7
policies × 5 budgets. n=76 trials per (policy,budget) cell; 14 instances rejected
because the full-cache model could not solve them (correctly excluded).

---

## 1. Measured baseline & fitted constants

- **Behavioral equivalence (falsifier #8):** `full_pages` ⇒ D_h = KL = **0.00000**,
  answer acc **1.000**. A 100% page budget reproduces the full model exactly.
- **Kernel equivalence:** `paged_attention_primitive` full block table vs numpy
  reference: `max|·|` = **3.7e-6 … 4.3e-6** across L∈{128,500,512,1024,2048}
  (fp16 noise floor), including the partitioned `_ps512`+reduce path.
- **Per-call decode kernel latency** (single layer-call, p50, warmup excluded;
  `results/systems.csv`): essentially flat in budget — the kernel is
  **overhead-bound**, not KV-walk-bound, at these lengths:

  | L | full p50 | gated@25% | gated@12.5% | gated@6.25% |
  |---|---|---|---|---|
  | 1024 | 0.169 ms | 0.156 | 0.149 | 0.131 |
  | 2048 | 0.160 ms | 0.145 | 0.138 | 0.140 |

  Fit of `T_full − T_gate ≈ c_f·H_q·d·(L−R)`: the slope is tiny
  (`c_f·H_q·d ≈ 1e-5 ms/token`), so even a 1800-token cut saves only ~0.02 ms
  per layer-call. Per decode step (×N=24 layers) ≈ **0.5 ms** saved at best.

- **Control-plane overhead** `T_u+T_s+T_tbl` ≈ **2.1 ms/step** (measured;
  dominated by `tokenizer.decode` inside the incremental belief update — an
  implementation artifact, not fundamental; pure page scoring/table build is
  microseconds).

## 2. Break-even

`T_u+T_s+T_tbl (≈2.1 ms) < c_f·H_q·d·(L−R)+c_p·H_q·(P−J) (≈0.5 ms/step at L=2048)`
is **FALSE**. At the tested lengths net gated latency **exceeds** full
(falsifiers #2 and #3 triggered). The kernel saving grows with L while the
control overhead is ~constant, so a crossing exists at larger L, but it was **not
reached** here. Honest wall: no wall-clock win at L ≤ 2048 on this model/GPU.

## 3. Empirical Pareto frontiers

Behavioral error vs budget (mean D_h | answer acc; n=76):

| budget | oracle | inferred | keyword | recent | random |
|---|---|---|---|---|---|
| 25%   | **0.046 \| 0.96** | 0.377 \| 0.80 | 0.379 \| 0.79 | 1.53 \| 0.41 | 1.64 \| 0.36 |
| 12.5% | **0.196 \| 0.92** | 0.693 \| 0.70 | 0.672 \| 0.68 | 1.84 \| 0.26 | 2.12 \| 0.21 |
| 6.25% | **1.004 \| 0.61** | 1.241 \| 0.46 | 1.328 \| 0.42 | 1.93 \| 0.21 | 2.29 \| 0.11 |

- **oracle ≫ inferred ≈ keyword ≫ recent ≫ random** at every budget.
- `inferred` and `keyword` are statistically indistinguishable (CIs overlap
  fully): the online belief's active pages coincide with keyword-matched pages.
- `attention_proxy` is worst — the harness never feeds it real attention mass
  (it degenerates to earliest-pages); reported honestly as an un-fed baseline,
  not a fair attention-proxy.

Plots: `plots/memory_error_frontier.png`, `plots/latency_error_frontier.png`,
`plots/context_scaling.png`. Memory (would-be) vs error: oracle dominates the
lower-left; gating cuts would-be cache bytes 25 MB→1.4 MB (18×) at 6.25%, but see
§5 — these bytes are **not freed**.

## 4. Where belief gating wins / fails (per family, budget 12.5%, D_h | acc)

| family | oracle | inferred | keyword |
|---|---|---|---|
| needle_recall          | 0.01 \| 1.00 | 0.36 \| 0.73 | 0.36 \| 0.73 |
| entity_state           | 0.19 \| 1.00 | 1.05 \| 0.50 | 1.05 \| 0.50 |
| incremental_constraint | 0.24 \| 0.92 | 0.44 \| 0.85 | 0.44 \| 0.85 |
| belief_revision (contradiction) | 0.15 \| 0.93 | 1.11 \| 0.79 | 1.10 \| 0.79 |
| delayed_disambiguation | 0.57 \| 0.73 | 0.91 \| 0.73 | 0.90 \| 0.60 |
| distractor_heavy       | 0.01 \| 1.00 | 0.54 \| 0.47 | 0.45 \| 0.53 |

- **Wins (oracle):** retention objective is achievable in *all* families; needle
  and distractor are near-perfect at 12.5%, proving pages can be cut hard without
  behavioral loss when the right pages are known.
- **Fails (inferred):** ties keyword in every family — the online belief adds no
  information beyond surface vocabulary. On the reasoning-heavy families
  (entity_state, belief_revision, delayed_disambiguation) even oracle drops below
  perfect, and inferred/keyword drop much further (falsifier #7: the gap to a
  working policy widens on contradiction/delayed tasks, though inferred does not
  collapse *relative to* keyword — they fail together).

## 5. Real physical memory? **No (shadow mode).**

The full physical KV cache stays allocated; `pages_reclaimed = 0`,
`shadow_mode = True`. The compact block table is a read-only view for the
attention forward only (audit §5). So physical allocation remains Θ(L)
(**falsifier #5 holds**): the 18× figure is *logical/would-be* reduction, not
bytes freed. Real reduction requires allocator integration (reference-count-safe
reclamation across layers / prefix-cache / COW), explicitly out of scope for
experiment 1.

## 6. Was the target file actually dispatched?

The experiment deliberately leaves `pagedattention.metal` **unchanged** — the
audit proved the existing decode kernel already computes exact attention over a
compact page subset, so the correct design is host-side selection, not a kernel
rewrite. For decode-only, the dispatched kernel **is** `pagedattention.metal`
(both `_ps0` and `_ps512` instantiations; audit §1), confirmed by the kernel
equivalence tests and the partitioned-boundary test. The feature-flag seam
(`VLLM_METAL_BELIEF_PAGED_ATTENTION` + `belief_gate.apply_gate`) routes the live
decode read through it and is an exact no-op when off (falsifier #10 not
triggered). The tiled prefill kernel is position-indexed and unsafe for
compaction; decode-only avoids it by construction.

## 7. Falsifier scorecard

| # | falsifier | result |
|---|---|---|
| 1 | inferred ⊁ keyword on the frontier | **TRIGGERED** — they tie at every budget/family |
| 2 | net gated latency > full at useful budgets | **TRIGGERED** at L≤2048 (overhead-bound kernel + control cost) |
| 3 | savings only without policy cost | **TRIGGERED** — even free policy gives only ~0.5 ms/step |
| 4 | oracle works but inferred does not | **TRIGGERED** — the defining result (ORACLE-ONLY) |
| 5 | physical cache stays Θ(L) | **TRIGGERED** — shadow mode, nothing reclaimed |
| 6 | improvements depend on future leakage | not triggered — oracle uses prefix-only spans; inferred is online |
| 7 | results vanish on contradiction/delayed | partial — reasoning tasks hurt all policies; inferred never separates from keyword |
| 8 | sparse breaks equivalence at 100% | **not triggered** — D_h=0, kernel err≈4e-6 (correctness holds) |
| 9 | rescans all history each step | not triggered — belief update O(chunk), selection O(P) |
| 10 | active dispatch never hits the kernel | not triggered — decode dispatches `pagedattention.metal`; flag-gated read verified |

## 8. What is and isn't established

**Established (real, verified):**
- The compact-block-table mechanism is numerically exact for RoPE-only decode
  (proof in MATH.md §2; kernel err≈4e-6; behavioral D_h=0 at 100%).
- The retention *objective* is achievable: an oracle selecting the evidence pages
  preserves behavior at 25%/12.5% budgets (acc 0.96/0.92).
- A clean, reusable host-side control plane + a default-off core-repo seam, with
  18 passing focused/kernel tests.

**Not established (honest negatives):**
- That an *online inferred* belief beats cheap heuristics — it does not (≈keyword).
- Any net latency or physical-memory win at the tested scale.

## 9. Smallest next experiment justified by the evidence

Because oracle works and inference is the sole failure, the next experiment
should **isolate inference quality at fixed selection**: keep the oracle
selection harness, and replace the surface-vocabulary inferencer with a
**model-internal signal that keyword matching cannot replicate** — e.g. an
*incremental attention-mass* selector (H2O/heavy-hitter), feeding the kernel's
own per-step attention probabilities back into `page_attention_mass` (the
`attention_proxy` policy, properly fed). Single task family (entity_state or
belief_revision, where surface vocab is weakest), single context length (≈2048),
budgets {25%, 12.5%}. Success criterion: attention-mass selection beats
`keyword_entity` at equal budget on D_h with non-overlapping 95% CIs. Only if
that clears should allocator-level reclamation (for real memory) be attempted.
