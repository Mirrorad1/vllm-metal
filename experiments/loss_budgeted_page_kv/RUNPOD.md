# Running the budget-vs-quality validation on RunPod (CUDA)

The MLX harness is Apple-Metal-only. `exp020_quality_cuda.py` is the self-contained
PyTorch/transformers port for a CUDA GPU box. It measures the **iso-quality KV budget**
(and thus the real capacity multiplier) for the deployable attention/recent selectors
on progressively harder long-context tasks, at 7B+ scale.

Validated locally on CPU (transformers 5.12.1, torch 2.11): the correctness gate passes
exactly (`full-keep vs causal max|Δ| = 0.00e+00`; tight = 15.2). The same gate re-runs on
the GPU before any result — if it ever FAILs, stop (the mask is wrong, numbers meaningless).

## Pod & sizing (IMPORTANT — eager attention is O(heads·L²))
`--n-filler` counts filler SENTENCES (~14 tokens each), so context = roughly n_filler·14:
`150→2.1k, 250→3.5k, 400→5.5k, 600→8.2k, 1500→20k, 3000→41k` tokens. The eager softmax
score matrix is `heads·L²·4 bytes` per layer (transient) — this is the binding limit, NOT
the weights. **Keep context modest.** Tested-safe on H100 80 GB:
  - 7B: `--n-filler 400` (~5.5k) solid; up to `--n-filler 600` (~8k) with expandable_segments.
  - 14B: `--n-filler 250` (~3.5k); ~`400` (~5.5k) max.
  - `--n-filler 1500/3000` (20k/41k) **OOMs even on 80 GB** in eager — needs the sdpa/cache
    rewrite (ask; not in this version).
Always set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Even 5–8k context at
7B/14B on harder tasks is far past the local Mac and enough to validate the hypothesis.
Standard RunPod "PyTorch 2.x / CUDA 12.x" template; no special build flags.

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
# 7B, ~3.5k context, all 5 task families, 30 valid examples each:
HF_HOME=/workspace/hf python exp020_quality_cuda.py \
  --model Qwen/Qwen2.5-7B-Instruct --n 30 --n-filler 300 --block-size 16

# longer context (needs a bigger card):
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 20 --n-filler 1200

# bigger model if the card allows:
python exp020_quality_cuda.py --model Qwen/Qwen2.5-14B-Instruct --n 20 --n-filler 300
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
