# Running the budget-vs-quality validation on RunPod (CUDA)

The MLX harness is Apple-Metal-only. `exp020_quality_cuda.py` is the self-contained
PyTorch/transformers port for a CUDA GPU box. It measures the **iso-quality KV budget**
(and thus the real capacity multiplier) for the deployable attention/recent selectors
on progressively harder long-context tasks, at 7B+ scale.

Validated locally on CPU (transformers 5.12.1, torch 2.11): the correctness gate passes
exactly (`full-keep vs causal max|Δ| = 0.00e+00`; tight = 15.2). The same gate re-runs on
the GPU before any result — if it ever FAILs, stop (the mask is wrong, numbers meaningless).

## Pod & sizing (this version is MEMORY-EFFICIENT — sdpa, not eager)
This version runs the big forwards under sdpa (fused mem-efficient backend) + a 4D mask +
`logits_to_keep`, so it does NOT materialize the O(L²) score matrix or O(L·vocab) logits
(see SPEC_exp020_memeff.md). The only O(L²) object is the additive mask tensor
(~3.4 GB bf16 @41k) — fine on 80 GB. So **long context now works**:
`--n-filler`: `150→2.1k, 400→5.5k, 600→8.2k, 1500→20k, 3000→41k` tokens.
  - 7B: `--n-filler 1500` (~20k) and `3000` (~41k) fit comfortably on an H100 80 GB.
  - 14B: `--n-filler 1500` (~20k) fits; `3000` (~41k) is tight but should fit.
Always set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
**If you still OOM**, it means PyTorch picked the MATH sdpa backend for the custom mask
(the `[gate]` line would print, then OOM on the first long forward) — tell me and I'll
ship the v3 cache-slicing variant (no mask tensor at all). Standard RunPod
"PyTorch 2.x / CUDA 12.x" template; no special build flags.

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

# 1. quick sanity (gate must PASS):
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 5 --n-filler 400

# 2. the real validation: 7B @ ~20k context, all 5 task families:
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 40 --n-filler 1500 --block-size 16

# 3. push to ~41k context, and/or a bigger model on the same H100:
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct  --n 30 --n-filler 3000
python exp020_quality_cuda.py --model Qwen/Qwen2.5-14B-Instruct --n 30 --n-filler 1500
```

## What to read
1. **`[gate] … => PASS`** must print first. If FAIL, stop.
2. The per-family table: `attn` (deployable selector) vs `recent` (floor) exact-answer
   accuracy at each budget.
3. **iso-quality budget** per family and its `capacity multiplier` (=1/budget). The
   headline = the **worst-task** multiplier (that's the safe budget you'd ship).

## What this answers
Whether the local-Mac finding (attention selector holds quality to ~6–12% budget, i.e.
8–16×, on single-needle COPY) survives at **7B scale + longer context + harder tasks**.
Hypothesis to test: harder/less-redundant families (multi_hop, distractor, exact_long_string)
need a HIGHER budget (lower multiplier) → the realistic safe multiplier is likely **2–4×**,
not 16×. The number this prints IS the honest capacity multiplier to quote.

## Notes
- This measures selection QUALITY, not kernel speed (full scores computed, then masked).
- It's leak-free: the selector uses only attention mass / recency, never the answer label.
- Compare the printed iso-budget to `results/exp020_quality/` (the 0.5B local run).
- Pairs with the `jax-cuda-runpod-gpu` skill for pod mechanics (runpodctl, repo clone).
