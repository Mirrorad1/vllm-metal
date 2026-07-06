# Synthesis — exp001 … exp010 (Loss-Budgeted Page KV Cache)

Model Qwen2.5-0.5B-Instruct, fp16, page B=16, decode-only. Behavioral sweep
n=43/cell (5 families × {≈512,1024,2048 tok} × 3 seeds, horizon 1, 95% CIs).
Systems on the real `paged_attention_primitive`, interleaved A/B.

## Headline

**There is strong page-level loss structure, and it is fully captured by causal
attention mass — the KL-damage "loss-budgeted" signal adds nothing over it. A
cheap online attention-based controller recovers the oracle behaviorally. But the
direction hits a SYSTEMS WALL: no physical memory is freed (shadow mode) and the
kernel speedup is modest/overhead-bound with no demonstrated end-to-end win.**

Behavioral frontier (D_h = KL(full‖gate) mean / answer-acc; n=43):

| budget | recent | random | sink_recent | page_norm | attention_proxy | loss_oracle | loss_online |
|---|---|---|---|---|---|---|---|
| 25%   | 1.84/.63 | 1.39/.74 | 1.84/.67 | 1.86/.63 | **0.02/1.00** | **0.03/1.00** | 0.05/.98 |
| 12.5% | 1.85/.63 | 1.51/.70 | 1.81/.63 | 1.87/.63 | **0.07/1.00** | **0.07/1.00** | 0.08/.98 |
| 6.25% | 1.88/.63 | 1.71/.67 | 1.83/.63 | 1.91/.63 | **0.55/.88** | **0.55/.91** | 0.58/.86 |

## The six questions

1. **Is there page-level loss structure?** **Yes, strongly.** Oracle/attention
   beat recency/random/sink/norm by 30–60× in D_h (0.02–0.07 vs 1.4–1.9) and lift
   answer accuracy 0.63→1.00 at 12.5–25% budgets. Pages are very unequal.

2. **Can a causal online policy recover it?** **Yes — and trivially.** The
   KL-damage oracle and the *causal* attention-mass proxy are statistically
   identical at every budget (e.g. 0.549 vs 0.548 @6.25%). So the structure the
   oracle finds IS the attention-mass structure. `loss_budgeted_online`
   (attn-mass + norm + recency) beats every cheap baseline with non-overlapping
   CIs and ≈ matches the oracle. The expensive O(P)-forward damage signal is
   unnecessary (F7).

3. **Does it reduce physical memory?** **No (LOGICAL-ONLY, F4).** The compact
   block table is a read-only view; allocation/refcount/reclamation live in
   upstream vLLM `KVCacheManager`. `pages_reclaimed=0`, physical bytes unchanged.
   Would-be bytes shrink ~J/P (18× @6.25%) but are NOT freed.

4. **Does it reduce end-to-end latency?** **No demonstrated win (SYSTEMS WALL,
   F5).** Re-measured kernel speedup is 1.13× (L=512) → 1.41× (L=8192) at 6.25%
   pages — real, monotone in L, but overhead-bound and far below the 16× the token
   cut implies. This is kernel-only; it excludes controller cost and the fact that
   attention is a fraction of a decode step, so end-to-end is strictly less. (An
   auto-generated "POSITIVE/2.5×" was retracted: it came from a noisy cold-start
   baseline — orchestrator re-ran with interleaved A/B.)

5. **Current honest wall.** The *behavioral* question is settled and positive but
   unsurprising: attention mass is the page-importance signal, and selecting
   high-attention pages preserves behavior cheaply. The *value* of the direction
   now rests entirely on systems, where it has not yet paid off: (a) memory is not
   freed without allocator integration; (b) the kernel is overhead-bound at tested
   scale. Both are real engineering walls, not behavioral ones.

6. **Continue, branch, or halt?** **Branch to systems; do NOT invest more in page
   *scoring*.** The scoring question is answered (attention mass ≈ oracle). The
   only way this direction becomes a real win is:
   - **Real physical reclamation** (exp009 follow-up): integrate with the
     allocator so unselected pages are actually freed/not-allocated, reference-count-
     and prefix-safe (F15). Until then any "memory win" is logical-only.
   - **Regime where the kernel ratio survives end-to-end**: ≥16K context and/or
     batched decode, where attention is a larger fraction of the step and the
     1.4×→larger kernel ratio can clear controller + MLP dilution.
   - **Incremental O(log P) controller** (exp008): a running per-page attention-mass
     priority queue so controller cost is bounded (F9), since the expensive damage
     signal is unnecessary.

## Caveats / threats to validity (carried forward)

- The attention-mass proxy here uses **fresh full-cache attention each step**
  (idealized). A deployed controller with *cumulative* eviction would measure
  attention only over retained pages; high-attention pages stay retained so this
  is plausibly stable, but it is **untested** and is the first behavioral risk if
  the direction continues.
- Horizon=1 (single-step D_h). Multi-step teacher-forced D_h under cumulative
  eviction is the honest behavioral test for a deployable controller.
- 0.5B model, B=16, synthetic+semi-natural prompts. F8/F12 (scale/natural) only
  partially probed.

## Decision

The Loss-Budgeted *scoring* hypothesis is **INCONCLUSIVE→NEGATIVE** (attention
mass dominates it; no edge). The broader page-native-approximation direction is a
**SYSTEMS WALL**: behaviorally easy, systemically unpaid. Recommended next single
experiment: **exp009b real physical reclamation** (the only thing that converts
the strong logical result into a real win). If reclamation is infeasible without a
major allocator rewrite, HALT per condition #3 and record what would revive it
(a long-context/batched end-to-end kernel win, or an allocator seam).
