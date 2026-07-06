# Belief-Gated PagedAttention — falsifiable experiment

Tests whether an **online inferred belief state** can select a *smaller* set of KV
pages than standard PagedAttention while **preserving future model behavior** and
producing a real reduction in memory / decode latency.

```
full PagedAttention block table
        → belief-conditioned page selector (host-side control plane)
        → compact block table (chronological subset of physical pages)
        → existing PagedAttention kernel (UNCHANGED)
```

The key enabling fact (audit + proof + measurement): the existing non-tiled
decode kernel `pagedattention.metal` is **position-agnostic w.r.t. the block
table** for RoPE-only decode — it masks causally by token *count*, and RoPE is
baked into cached K. So a chronological subset of physical pages + a matching
`context_len` yields **exact** attention over the selected tokens, with **no
kernel change**. See `IMPLEMENTATION_AUDIT.md` §4 and `MATH.md` §2.

## Files

| file | role |
|---|---|
| `IMPLEMENTATION_AUDIT.md` | dispatch-path audit (gating deliverable); proves the kernel is safe for compaction |
| `MATH.md` | baselines, the correctness-invariant proof, break-even model |
| `page_policies.py` | 7 equal-budget page-selection policies + invariant enforcement |
| `belief_state.py` | online, incremental belief inference (`B_t = f(B_{t-1}, x_t)`) |
| `tasks.py` | 6 synthetic long-context task families |
| `metrics.py` | KL/JS, top-1/top-k, attention rel-error, systems accounting + CIs |
| `benchmark.py` | the two harnesses + the sweep driver |
| `plots.py` | renders the three frontier plots from results |
| `test_belief_paged_attention.py` | focused unit + kernel tests |
| `results/`, `plots/` | outputs (`raw_results.jsonl`, `aggregate.csv`, `systems.csv`, `summary.md`) |

## Two faithful harnesses (each measures what it is faithful to)

- **Behavioral** (real model, mlx-lm `Qwen2.5-0.5B-Instruct`): teacher-forced
  full-cache vs page-gated next-token distributions → `D_h = KL`, top-1,
  answer accuracy. A `GatedCache` keeps the *absolute* RoPE position so relative
  positions stay exact under compaction (validated: `full_pages ⇒ D_h=0`).
- **Systems** (real vllm-metal `paged_attention_primitive`): all-pages numeric
  equivalence (falsifier #8), full-vs-gated decode kernel latency (p50/p95,
  warmup excluded), and honest memory accounting (shadow mode — no page
  reclaimed). Dropping a contiguous K/V block is *exactly* subset attention under
  both mlx-lm SDPA and the paged kernel, so the split is principled.

The control-plane cost (belief update + scoring + table build) is timed and
**included** in gated latency (falsifier #3).

## Policies (equal page budget `J`)

`full_pages` (reference) · `recent_pages` · `seeded_random_pages` ·
`attention_proxy_pages` · `keyword_entity_pages` · `oracle_belief_pages`
(prefix-derived, no future info) · `inferred_belief_pages` (online belief).
All non-full policies retain an identical mandatory floor (current partial tail
+ recent window + sinks) and spend the remaining budget by their own scoring, so
comparisons are at identical `J`.

## Feature flag (core-repo seam, default OFF)

`VLLM_METAL_BELIEF_PAGED_ATTENTION=1` + a selector registered via
`vllm_metal.attention.belief_gate.register_selector(...)` makes the live decode
attention read use a compact block table. OFF ⇒ exact no-op (full table). The KV
write path and the scheduler's allocations are never touched (read-only,
shadow-mode). `belief_gate.apply_gate` re-validates every compaction and falls
back to the full table on any violation (fail safe, never fail wrong).

## Running

```bash
source ../../.venv-vllm-metal/bin/activate
export VLLM_METAL_BUILD_FROM_SOURCE=1   # checkout ships no prebuilt .metallib

# focused tests (pure fast; kernel tests build from source)
python -m pytest test_belief_paged_attention.py -c pytest.ini -p no:cacheprovider -q

# behavioral + systems sweep (context lengths via filler count ≈ 512/1024/2048 tok)
python benchmark.py --context-lengths 35 75 150 --seeds 0 1 2 3 4 --horizon 2

# plots from results/
python plots.py
```

## Scope of experiment 1 (enforced)

decode-only · standard RoPE · no ALiBi/sinks (off by construction in the dispatch)
· `sliding_window < 0` · non-TurboQuant fp16/bf16 · `block_size ∈ {8,16,32}` · no
hybrid block inflation · **no physical-page reclamation** (shadow mode). The
honest verdict, fitted constants, Pareto frontiers, and the smallest justified
next experiment are in `results/summary.md`.
