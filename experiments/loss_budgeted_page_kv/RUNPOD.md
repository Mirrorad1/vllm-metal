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

**`fast` self-proves it equals `mask` before any result is trusted.** On startup (on the
fp32 gate model) it runs a preflight over every family × budget × policy and asserts (a) the
fast selector picks the **same pages** and (b) the fast decode's **answer LOGITS equal the
mask path's within fp32 tolerance** (`max|Δ logits| < 5e-2`) — strictly stronger than matching
the final exact-answer decision, and run in fp32 so an argmax tie can't false-alarm. It prints
e.g. `[fast-preflight] fp32 max|Δ logits| fast vs mask = 7.3e-04 (<5e-02); selector 70/70
identical; decisions 70/70 identical => PASS`. **Any numerical divergence aborts the run** (it
refuses to report numbers it can't prove equal to the reference). To audit, run the same config
with `--engine mask` and confirm the per-family table matches.

> The logit-level fp32 preflight earned its keep: it caught a real off-by-one in the *mask
> reference* (it left answer[0] — predicted by the last prompt token — un-gated, so it could
> see dropped pages) that the older decision-level check had missed. The `fast` engine was
> always correct (it gates every decode step, as a real evict-then-decode deployment does), so
> prior `--engine fast` numbers stand; the reference is now fixed to match.

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
2. **`[fast-preflight] fp32 max|Δ logits| … => PASS`** must print next (fast engine): the
   fast path's answer logits equal the mask reference within fp32 tolerance. If FAIL, the run
   aborts itself — it refuses to report numbers it can't prove equal to the reference.
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

---

# exp021 — conformal compressibility ADMISSION CONTROLLER (the HORN-B verdict)

`exp020_quality_cuda.py` now ALSO dumps, per accepted instance, the query-agnostic per-page
attention `mass` (one `{"event":"instance",...}` line) — the write-side feature the admission
controller needs. `exp021_admission.py` then builds a split-conformal "admit-at-B / refuse"
gate OFFLINE from that jsonl (CPU only) and answers the decisive question (SPEC_exp021_admission.md):
**does the answer-level certificate BEAT free attention-entropy (HORN-B)?**

The 0.5B local run was DEGENERATE (0.5B can't do dense aggregation → too few break events).
The real verdict needs 7B, where `sum_scattered` has a population AND tight budgets create
damage variance.

```bash
# extra deps (the gate is CPU/offline numpy/scipy/sklearn):
pip install scipy scikit-learn

# 0. sanity: the harness must self-verify before you trust any verdict (no GPU needed):
python exp021_admission.py --selftest          # must print: PASS

# 1. generate features+labels with the INSTRUMENTED exp020. Include sparse families for
#    contrast with the dense (sum_scattered) regime where the admission decision matters:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 60 --n-filler 1500 \
  --families single_needle multi_needle multi_hop sum_scattered --block-size 16 \
  --out results/exp021_7b

# 2. the verdict (offline, seconds). Budget = the compression you'd actually serve at:
python exp021_admission.py --runs results/exp021_7b/raw_Qwen2.5-7B-Instruct.jsonl --budget 0.0625
python exp021_admission.py --runs results/exp021_7b/raw_Qwen2.5-7B-Instruct.jsonl --budget 0.25
```

Read the VERDICT line:
- **WIN (H2)** — the answer-level certificate beats entropy AND carries per-instance signal
  within task type (family-centered AUC ≥ 0.65) → a real new mechanism.
- **WIN-MIX / KILL k4 (FINGERPRINT)** — beats entropy but only by identifying the TASK FAMILY;
  within-family signal ≈ chance. **This is what the 7B/20k run returned** (+0.206@6.25%,
  +0.267@25%, coverage held; family-ID AUC 1.00 on multi_hop-vs-sum_scattered where entropy is
  blind; centered AUC 0.42). Deployable as a task-type admission gate over a stationary mix;
  NOT a per-context damage certificate; no transfer to unseen task types.
- **KILL k1** — entropy threshold matches it → a calibrated entropy admission gate (deployable,
  unshipped today, but NOT novel).
- **DEGENERATE / NOT EVALUABLE** — too few break events (raise `--n`, or pick a budget where
  `sum_scattered` actually breaks, e.g. 0.0625). Not a verdict, just insufficient damage variance.

---

# exp022 — the substrate matrix (L40S, ~$5, one afternoon)

Is the wall substrate-shaped or context-shaped? `exp022_substrates.py` fills the
per-context loss matrix at iso-bits (eviction / 4-2bit quant / low-rank SVD / per-context
LoRA) over the SAME 225 contexts as the archived 7B run (instance list read from
`results/exp021_7b/raw_*.jsonl`, committed). Pod: **1x L40S 48GB** ("PyTorch 2.x / CUDA"
template). All gates self-prove (g1 full-cache, g2 identity no-ops, g3 sensitivity,
g4 recomputed eviction labels must match the archived H100 labels >=90%).

```bash
cd /workspace
git clone --depth 1 -b kv-loss-budgeted-experiments https://github.com/Mirrorad1/vllm-metal.git
cd vllm-metal/experiments/loss_budgeted_page_kv
pip install -q torch "transformers>=5.0" accelerate peft numpy scipy scikit-learn
export HF_HOME=/workspace/hf

# 1. quantization + low-rank + eviction-recheck columns (~1h; watch the 4 gates)
python exp022_substrates.py --stage s23

# 2. LoRA (weights) column, 60 contexts (~3h). To fan across N pods: --shard k/N on each.
python exp022_substrates.py --stage s4

# 3. ship results home, then verdict runs OFFLINE on the Mac:
runpodctl send results/exp022_7b
#   (locally) runpodctl receive <code> ; python exp022_matrix.py --dir results/exp022_7b
```

Verdict line meanings:
- **H-GENERAL** — no substrate rescues the eviction-wall contexts: incompressibility is a
  property of the CONTEXT. Distillation dies with it => build the re-read fallback (L2).
- **H-SPECIFIC** — substrates fail on different contexts: the admission gate upgrades to a
  substrate DISPATCHER (section F shows how much a type-level dispatch captures).
- **DEGENERATE** — dense families didn't break under eviction: wrong model/scale, not a verdict.

---

# exp024 — the dispatcher on a REGULAR benchmark (LongBench v2, real documents)

Same L40S pod recipe; ~1h; produces the exp022-schema matrix on real LBv2 short-bucket
items (MCQ, exact scoring) with the 0.5B proxy features embedded, then the exp023
dispatcher runs on it UNCHANGED — offline, on the Mac.

```bash
cd /workspace/vllm-metal/experiments/loss_budgeted_page_kv
git pull
pip install -q datasets
nohup python exp024_bench_cuda.py > exp024.log 2>&1 &
tail -f exp024.log          # gate g1/g3 lines first; then acceptance ticker
# when [done]: runpodctl send results/exp024_lbv2     (rm old zip first if it complains)
```

Then locally:
```bash
python exp023_dispatch.py --s23 results/exp024_lbv2/s23.jsonl --proxy ""
```

Read: acceptance rate (7B must solve the item with FULL cache for it to count), the
quant8b sanity caveat line, the action histogram per DOMAIN, and the same
WIN/MODEST/KILL verdict. Honest notes: 4-way MCQ has a 25% guess floor (break labels
are noised toward safe), and if real benchmark traffic turns out mostly
sparse-compressible, a small dispatcher win over the best gate is itself the finding
(benchmark composition, cf. the ETHIC/Dolce line).
