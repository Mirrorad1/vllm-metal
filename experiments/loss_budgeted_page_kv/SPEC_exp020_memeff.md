# SPEC — exp020 memory-efficient (long-context) rewrite

## Problem
The current `exp020_quality_cuda.py` uses `attn_implementation="eager"`, which
materializes the `[heads, L, L]` attention score matrix (QK matmul + softmax) =
**O(heads·L²)** memory. At ~41k tokens (`--n-filler 3000`) on a 28–40-head model
that single allocation is 85+ GiB → OOM even on an H100 80 GB. This caps the quality
experiment at ~8k context — but the capacity win is most interesting at **16k–128k**.

## Goal
Run the SAME experiment (budget-vs-quality on the 5 harder families; deployable
attention/recent selector; exact-answer metric; correctness gate) at **16k–41k**
context on a single 80 GB GPU.

## Verified building blocks (tested on CPU, transformers 5.12.1)
1. **sdpa accepts a custom 4D additive float mask** → big forwards run on the fused
   mem-efficient kernel (no O(L²) score matrix materialized).
2. `output_attentions=True` returns `None` under sdpa (can't get weights) — BUT
3. **`model.set_attn_implementation("eager"|"sdpa")` round-trips cleanly at runtime**
   → switch to eager only for a tiny one-token attention-mass forward.
4. **`logits_to_keep=K`** computes logits only for the last K positions → avoids the
   O(L·vocab) logits tensor (~12 GB at 41k).  ⟵ *kwarg name to confirm on CPU*

## Design

### Forward primitives
- `quality_forward(model, ids, mask4d, n_keep)` — the big pass:
  sdpa, `attention_mask=mask4d`, `logits_to_keep=n_keep`, wrapped in
  `torch.nn.attention.sdpa_kernel([EFFICIENT_ATTENTION, FLASH_ATTENTION])` so it
  **cannot silently fall back to the MATH backend** (which would re-materialize
  O(L²)). Returns logits `[1, n_keep, vocab]`. Memory: O(L) attention + O(L²) for the
  mask tensor only (~3.4 GB bf16 @41k, acceptable).
- `attention_mass(model, prompt_ids)` — the cheap selector signal:
  1. sdpa: `forward(prompt[:-1], use_cache=True, logits_to_keep=1)` → cache (O(L)).
  2. `set_attn_implementation("eager")`; `forward(last_token, past=cache,
     output_attentions=True)` → `[1, heads, 1, L]` (O(L)); `set_attn_implementation("sdpa")`.
  3. avg over layers & heads → per-key mass → bin to pages.

### 4D mask (builder unchanged)
`[1,1,L,L]`: causal everywhere; for ANSWER-query rows (`>= prompt_len`) additionally
`-inf` at dropped-page key columns. Built in run dtype.

### logits_to_keep indexing
`n_keep = len(answer)+1`; the kept window is positions `[prompt_len-1 .. L-1]`, so
`logits[0, t]` predicts answer token `t`. `exact_answer_ok` checks
`argmax(logits[0,t]) == answer[t]` for `t in range(len(answer))`.

### Per-example flow
prefill exact-answer check (full mask) → if pass: `attention_mass` → for each
(budget × {attention, recent}): build mask with that selector's dropped pages →
`quality_forward` → `exact_answer_ok` → record.

### Correctness gate (transplant skill; run FIRST, fp32, abort on fail)
- `base` = quality_forward(plain causal); `full` = quality_forward(answer-row, drop
  nothing). **max|Δ| < 1e-3** (reduces to dense).
- `tight` = quality_forward(answer-row, drop all but last page). **max|Δ| > 1e-2**
  (mask actually fires).

## Residual risks + mitigations
- **R1 (main): sdpa picks MATH backend for a custom float mask → O(L²) → OOM.**
  Mitigation: `sdpa_kernel([EFFICIENT, FLASH])` excludes MATH; mem-efficient supports
  an additive attn_bias, so it should take the mask. If no backend accepts it, it
  **raises loudly** (not a silent OOM). Documented fallback = v3 cache-slicing (no mask).
- R2: the O(L²) mask tensor (~3.4 GB bf16 @41k) — fine on 80 GB.
- R3: per-example impl switch — negligible (swaps a flag + the attention fn).
- R4: mask semantics identical to the validated eager version; the gate re-checks.

## Acceptance criteria
- **CPU (I run before handing off):** gate PASS (Δ_full≈0, Δ_tight large); end-to-end
  on Qwen2.5-0.5B; sane per-family numbers; `logits_to_keep` kwarg confirmed.
- **GPU (you run):** `--n-filler 1500` (~20k) and `--n-filler 3000` (~41k) on an H100
  80 GB complete without OOM; gate PASS; per-family iso-budget printed.

## Out of scope (v3, only if R1 bites on the GPU)
True KV-cache slicing: prefill→cache (flash, is_causal), slice cache to kept pages,
teacher-force answer tokens with explicit `position_ids`. O(L) with NO mask tensor.
More code + more position-handling risk; build only if the sdpa-mask path OOMs.
