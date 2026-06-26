# Loss-Budgeted Page KV Cache (KV-cache only)

> **→ Read [`STATUS_AND_ROADMAP.md`](STATUS_AND_ROADMAP.md) first** — the honest
> consolidated findings (exp001–019) + the concrete next step. This README is the
> original framing; the roadmap is the current state of the project.

Tests whether a production-shaped KV-cache controller can reduce **physical** KV
memory and/or **decode latency** by making page-native decisions (keep / drop /
[later: quantize / offload / recompute]) under an explicit error budget — and
only counts a win if the measured memory/latency/error frontier actually moves
against strong baselines.

**No belief inference, no semantic task-state, no future/answer leakage.** The
controller uses only KV/logit/output/calibration signals.

## Method

PagedAttention already gives page-level KV structure. We use the page as the unit
of controlled approximation: build a compact (chronological subset) block table
and pass it to the *existing* decode kernel. The audit proves the decode kernel
(`pagedattention.metal`) is position-agnostic w.r.t. the block table for RoPE-only
decode, so this is exact at 100% and correct for any subset — **no kernel
change** (`IMPLEMENTATION_AUDIT.md`, `MATH.md §2`).

## Files
`IMPLEMENTATION_AUDIT.md` (gating audit) · `MATH.md` · `EXPERIMENT_LEDGER.md` ·
`page_policies.py` (9 equal-budget policies incl. `loss_budgeted_oracle`/`_online`) ·
`error_metrics.py` · `benchmark_harness.py` (real-model behavioral + real paged
kernel systems + per-page KL-damage + attention-mass capture) · `run_experiment.py` ·
`run_loop.py` (orchestrator + decision rules) · `plot_results.py` ·
`test_loss_budgeted_page_kv.py` · `results/` · `plots/`.

## Policies (equal page budget J)
`full_pages` (reference) · `recent_pages` · `seeded_random_pages` ·
`sink_recent_pages` · `attention_proxy_pages` (real causal attention mass) ·
`page_norm_pages` · `loss_budgeted_oracle` (top-J by KL-damage; diagnostic) ·
`loss_budgeted_online` (causal blend of attention mass + norm + recency).

## Two faithful harnesses
- **Behavioral** (mlx-lm `Qwen2.5-0.5B-Instruct`): teacher-forced full vs gated
  next-token KL/JS/top-1/accuracy. `full_pages ⇒ KL=0` (validated).
- **Systems** (vllm-metal `paged_attention_primitive`): all-pages equivalence
  (err≈4e-6), decode kernel p50/p95 (warmup excluded), honest memory accounting
  (shadow mode: no page reclaimed unless an allocator path is added — exp009).

## Run
```bash
source ../../.venv-vllm-metal/bin/activate
export VLLM_METAL_BUILD_FROM_SOURCE=1
python run_loop.py --n-fillers 35 75 150 --seeds 0 1 2 --horizon 1
python plot_results.py
python -m pytest test_loss_budgeted_page_kv.py -c pytest.ini -p no:cacheprovider -q
```

## Scope (exp1): decode-only, standard RoPE, no ALiBi/sliding-window, fp16, no
offload/quant/reclamation/prefix-mutation. Feature flag
`VLLM_METAL_LOSS_BUDGETED_PAGE_KV` (default off; shared generic page-gating seam
`vllm_metal/attention/belief_gate.py`). Honest walls and per-experiment verdicts
in `results/exp*/verdict.md` and `results/synthesis_*.md`.
