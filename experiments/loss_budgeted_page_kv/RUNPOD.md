# Running the budget-vs-quality validation on RunPod (CUDA)

The MLX harness is Apple-Metal-only. `exp020_quality_cuda.py` is the self-contained
PyTorch/transformers port for a CUDA GPU box. It measures the **iso-quality KV budget**
(and thus the real capacity multiplier) for the deployable attention/recent selectors
on progressively harder long-context tasks, at 7B+ scale.

Validated locally on CPU (transformers 5.12.1, torch 2.11): the correctness gate passes
exactly (`full-keep vs causal max|Δ| = 0.00e+00`; tight = 15.2). The same gate re-runs on
the GPU before any result — if it ever FAILs, stop (the mask is wrong, numbers meaningless).

## Two engines (`--engine`, default `fast`) — same numbers, ~5-10× faster
There are two equivalent implementations of the budget sweep:
- **`fast` (default):** prefill the prompt's KV cache **once**, then for every (budget × policy)
  just **decode the answer over a sliced cache** — no long forward per combination. With ~13
  combinations per example this is ~5-10× fewer long forwards. Uses absolute `position_ids`
  (RoPE) with sliced `cache_position`, and **never materializes an O(L²) mask** at all.
- **`mask`:** the validated reference — one full `[prompt+answer]` forward per combination
  under a 4D additive mask.

**`fast` self-proves it equals `mask` before any result is trusted.** On startup it runs a
preflight over the first few example×budget×policy points and asserts BOTH that the fast
selector picks the **same pages** and the fast decode makes the **same exact-answer decision**
as the mask path — printing e.g. `[fast-preflight] decode fast≡mask: 36/36 identical; selector
fast≡mask: 36/36 identical => PASS`. **Any mismatch aborts the run** (it refuses to report
numbers it can't prove equal to the reference). To audit, run the same config with
`--engine mask` and confirm the per-family table matches. Proven equivalent on CPU 0.5B
(identical decisions across families × budgets × policies).

## Pod & sizing (this version is MEMORY-EFFICIENT — sdpa, not eager)
This version runs the big forwards under sdpa (fused mem-efficient backend) + a 4D mask +
`logits_to_keep`, so it does NOT materialize the O(L²) score matrix or O(L·vocab) logits
(see SPEC_exp020_memeff.md). The only O(L²) object is the additive mask tensor
(~3.4 GB bf16 @41k) — fine on 80 GB. So **long context now works**:
`--n-filler`: `150→2.1k, 400→5.5k, 600→8.2k, 1500→20k, 3000→41k` tokens.
  - 7B: `--n-filler 1500` (~20k) and `3000` (~41k) fit comfortably on an H100 80 GB.
  - 14B: `--n-filler 1500` (~20k) fits; `3000` (~41k) is tight but should fit.
Always set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The sizing above is for `--engine mask`. **`--engine fast` (the default) never builds the
O(L²) mask tensor at all** (it slices the KV cache instead), so it's even lighter on memory —
if `mask` ever OOMs on the longest contexts, `fast` is the path that avoids it. Standard
RunPod "PyTorch 2.x / CUDA 12.x" template; no special build flags. (Historical note: the
`fast` engine is the "v3 cache-slicing variant" the earlier SPEC named as the OOM fallback —
it's now the default and proven equivalent to the mask path.)

## Setup
```bash
pip install "torch" "transformers>=5.0" accelerate
export HF_HOME=/workspace/hf            # persist the model cache on the pod volume
# (optional) export HF_TOKEN=hf_...     # avoids rate limits / gated models
# copy just this one file to the pod:
#   scp experiments/loss_budgeted_page_kv/exp020_quality_cuda.py  pod:/workspace/
```

## Run
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. quick sanity (gate must PASS, then preflight must print "=> PASS"):
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 5 --n-filler 400

# 2. the real validation: 7B @ ~20k context, all 7 task families (fast engine, default):
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 40 --n-filler 1500 --block-size 16

# 2b. THE DENSE TEST (does the 8-16x retrieval number survive when the answer needs MANY
#     distributed spans?): the two aggregation families alone. Expect a HIGHER iso-budget.
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 40 --n-filler 1500 \
  --families sum_scattered recall_all --block-size 16

# 3. push to ~41k context, and/or a bigger model on the same H100:
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct  --n 30 --n-filler 3000
python exp020_quality_cuda.py --model Qwen/Qwen2.5-14B-Instruct --n 30 --n-filler 1500

# (optional audit) re-run any config on the reference path — table must match the fast run:
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 40 --n-filler 1500 --engine mask
```

## What to read
1. **`[gate] … => PASS`** must print first. If FAIL, stop.
2. **`[fast-preflight] … => PASS`** must print next (fast engine). If FAIL, the run aborts
   itself — the fast path didn't match the reference, so no numbers are reported.
3. The per-family table: `attn` (deployable selector) vs `recent` (floor) exact-answer
   accuracy at each budget.
4. **iso-quality budget** per family and its `capacity multiplier` (=1/budget). The
   headline = the **worst-task** multiplier (that's the safe budget you'd ship).

## What this answers
The 5 retrieval families (single_needle … distractor) have a SINGLE answer span, so the
attention selector keeps that one page and holds quality to a tiny budget — measured at
7B/20k: **6.25% on 4/5 (16×), 12.5% on multi_hop (8×)**.

The **2 dense families are the real stress test.** Their answer depends on MANY spans
scattered across the whole context that must ALL survive at once:
- `sum_scattered` — sum of 5 small deposits placed far apart (drop one addend → wrong sum).
- `recall_all` — reproduce 6 codes in order (drop one → wrong; no arithmetic).
No single page is query-salient ("the total" doesn't point at any one entry), so the selector
must spend budget keeping all of them. **Hypothesis: these need a HIGHER iso-budget → a lower
multiplier (~2–4×), telling you the honest number for aggregation/synthesis workloads** rather
than the optimistic retrieval 8–16×. The worst-task multiplier across all 7 is the safe one to
ship. (Note: `sum_scattered` also requires the model to do the arithmetic — instances the full
cache gets wrong are auto-rejected, so it measures degradation conditional on full-cache success.)

## Notes
- This measures selection QUALITY, not kernel speed (full scores computed, then masked).
- It's leak-free: the selector uses only attention mass / recency, never the answer label.
- Compare the printed iso-budget to `results/exp020_quality/` (the 0.5B local run).
- Pairs with the `jax-cuda-runpod-gpu` skill for pod mechanics (runpodctl, repo clone).
