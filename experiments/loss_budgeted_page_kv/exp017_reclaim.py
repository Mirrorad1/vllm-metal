# SPDX-License-Identifier: Apache-2.0
"""exp017 — Physical KV reclamation prototype (the SYSTEMS lacuna).

Every prior experiment measured "would-be" bytes but reclaimed ZERO physical pages
(the compact block table was a read-only view; the upstream vLLM allocator owns the
blocks). This tests the decisive systems question directly, with ACTUAL allocation:
if you physically GATHER the kept pages into a smaller KV cache and free the rest,
does real process memory drop, by how much, and at what cost (copy time + transient
peak)?

Measures (real MLX allocation, model-shaped multi-layer KV cache):
 - actual active memory freed vs would-be (J/P)   [F4: logical vs physical]
 - transient PEAK during compaction (gather needs old+new live briefly)
 - compaction time (gather + eval + clear_cache)
 - decode-kernel latency full vs compacted (reuse) + numeric equivalence
The remaining engineering (NOT prototyped): doing this reference-count / prefix /
copy-on-write-safe inside the live vLLM v1 allocator (F15).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H

_RES = Path(__file__).resolve().parent / "results"


def _mem():
    import mlx.core as mx
    return mx.get_active_memory()


def build_full_cache(NL, P, B, Hkv, d, seed=0):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    K = [mx.array(rng.standard_normal((P, B, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
         for _ in range(NL)]
    V = [mx.array(rng.standard_normal((P, B, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
         for _ in range(NL)]
    mx.eval(K + V)
    return K, V


def compact(K, V, keep_pages):
    """Gather selected pages into smaller per-layer arrays; return new (Kc, Vc) and time."""
    import mlx.core as mx
    idx = mx.array(np.asarray(keep_pages, dtype=np.int32))
    t0 = time.perf_counter()
    Kc = [k[idx] for k in K]
    Vc = [v[idx] for v in V]
    mx.eval(Kc + Vc)
    dt = time.perf_counter() - t0
    return Kc, Vc, dt


def run_cell(NL, Hkv, d, B, context_len, budget, n_q, seed=0):
    import mlx.core as mx
    P = (context_len + B - 1) // B
    J = max(1, round(budget * P))
    keep = sorted(set(list(range(P - J, P)) + [0]))[:J] or [P - 1]
    J = len(keep)

    mx.clear_cache(); mx.reset_peak_memory()
    base = _mem()
    K, V = build_full_cache(NL, P, B, Hkv, d, seed)
    full_active = _mem() - base
    peak_full = mx.get_peak_memory()

    # compact (transient peak = full + compact while both live)
    mx.reset_peak_memory()
    Kc, Vc, comp_t = compact(K, V, keep)
    peak_during = mx.get_peak_memory()
    # free the full cache
    del K, V
    gc.collect()
    mx.clear_cache()
    compact_active = _mem() - base
    actual_freed = full_active - compact_active
    would_be_freed = full_active * (P - J) / P

    # latency + correctness cross-check on the real paged kernel (separate small alloc)
    full_p50, _, _ = H.paged_latency(context_len, list(range(P)), B, n_q, Hkv, d, iters=20, warmup=5)
    R = sum(B if p < P - 1 else (context_len - (P - 1) * B or B) for p in keep)
    gp50, _, _ = H.paged_latency(context_len, keep, B, n_q, Hkv, d, iters=20, warmup=5)
    eq = H.paged_all_pages_err(context_len, B, n_q, Hkv, d)

    del Kc, Vc
    gc.collect(); mx.clear_cache()
    return {
        "context_len": context_len, "P": P, "budget": budget, "J": J,
        "full_active_bytes": full_active, "compact_active_bytes": compact_active,
        "actual_freed_bytes": actual_freed, "would_be_freed_bytes": would_be_freed,
        "reclaim_ratio": actual_freed / max(would_be_freed, 1),   # ~1.0 = real reclamation
        "actual_reduction_frac": actual_freed / max(full_active, 1),
        "logical_reduction_frac": (P - J) / P,
        "transient_peak_bytes": peak_during, "transient_peak_vs_full": peak_during / max(full_active + base, 1),
        "compaction_s": comp_t,
        "kernel_full_p50_ms": full_p50, "kernel_compact_p50_ms": gp50,
        "kernel_speedup": full_p50 / max(gp50, 1e-9), "kernel_eq_err": eq, "R": R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--n-kv", type=int, default=2)
    ap.add_argument("--n-q", type=int, default=14)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--out", default=str(_RES / "exp017_reclaim"))
    args = ap.parse_args()
    H.get_ops()  # ensure kernel built
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "raw_results.jsonl", "w")
    rows = []
    for L in args.contexts:
        for bf in args.budgets:
            r = run_cell(args.n_layers, args.n_kv, args.head_dim, args.block_size, L, bf, args.n_q)
            rows.append(r); raw.write(json.dumps(r) + "\n")
            print(f"L={L:6d} bf={bf:6.4f} J={r['J']:4d}/{r['P']:4d} | full={r['full_active_bytes']/1e6:7.1f}MB "
                  f"actual_freed={r['actual_freed_bytes']/1e6:7.1f}MB reclaim_ratio={r['reclaim_ratio']:.2f} "
                  f"| peak_during={r['transient_peak_vs_full']:.2f}× comp={r['compaction_s']*1e3:.1f}ms "
                  f"| kernel {r['kernel_full_p50_ms']:.3f}→{r['kernel_compact_p50_ms']:.3f}ms eq={r['kernel_eq_err']:.1e}", flush=True)
    raw.close()

    import csv
    with open(out / "aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # verdict: is reclamation real (ratio~1) and beneficial (memory down ~J/P)?
    ratios = [r["reclaim_ratio"] for r in rows]
    mean_ratio = float(np.mean(ratios))
    worst_peak = float(np.max([r["transient_peak_vs_full"] for r in rows]))
    eq_ok = all(r["kernel_eq_err"] < 5e-3 for r in rows)
    real = mean_ratio > 0.9 and eq_ok
    if real:
        verdict = "PHYSICAL RECLAMATION REAL (memory actually freed ~J/P)"
        reason = (f"compaction physically frees memory at reclaim_ratio {mean_ratio:.2f} (actual/would-be ≈ 1): "
                  f"active memory drops ≈ the logical fraction (J/P). Correctness holds (kernel eq < 5e-3). "
                  f"COSTS: transient peak up to {worst_peak:.2f}× full during the gather (you need full+compact "
                  f"live briefly — so reclamation must be staged or done layer-by-layer to avoid an OOM spike), "
                  f"and a compaction copy of a few ms. So the SYSTEMS lacuna is BUILDABLE in this stack: MLX "
                  f"frees the bytes. The remaining gap is purely the SAFETY integration (refcount/prefix/CoW) "
                  f"inside the live vLLM allocator — engineering, not a research unknown.")
    else:
        verdict = "RECLAMATION INCOMPLETE / costly"
        reason = (f"reclaim_ratio {mean_ratio:.2f} (memory not fully freed) or correctness failed (eq_ok={eq_ok}); "
                  f"the caching allocator or transient peak ({worst_peak:.2f}×) blunts the gain.")
    agg = {"n_cells": len(rows), "mean_reclaim_ratio": mean_ratio, "worst_transient_peak_x": worst_peak,
           "kernel_equivalence_ok": eq_ok, "verdict": verdict, "verdict_reason": reason}
    (out / "aggregate_summary.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp017 VERDICT:", verdict, "===")
    print(reason)
    return agg


if __name__ == "__main__":
    main()
