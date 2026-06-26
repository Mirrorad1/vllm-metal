# SPDX-License-Identifier: Apache-2.0
"""Belief-gated PagedAttention experiment driver.

Two faithful, real measurements (never a proxy for the thing being claimed):

  * BEHAVIORAL harness — a real model (Qwen2.5-0.5B-Instruct via mlx-lm). We
    prefill ``tokens[0:L-1]`` into a per-layer KV cache, then teacher-force a
    horizon of continuation tokens. At each decode step we read the full-cache
    next-token distribution ``p`` and the page-gated distribution ``q`` (via
    ``GatedCache``, which keeps the *absolute* RoPE position so relative
    positions stay exact — see IMPLEMENTATION_AUDIT.md §4 and the RoPE note
    below). This yields D_h = mean KL(p||q), top-1 agreement, and exact-answer
    accuracy — the spec's behavioral-error locus, measured on a real model.

  * SYSTEMS harness — the real vllm-metal ``paged_attention_primitive`` kernel.
    We measure all-pages numeric equivalence (falsifier #8), and decode kernel
    latency for full (P pages, context L) vs gated (J pages, context R) with
    warmup excluded, p50/p95. Memory is accounted via metrics.cache_bytes,
    honestly labeled shadow-mode (no physical page is reclaimed).

Why split: dropping a contiguous block of cached K/V is *exactly* subset
attention under both mlx-lm SDPA and the paged kernel (proved + probed). The
behavioral question (does selection preserve model behavior) needs real weights;
the systems question (latency/bytes/equivalence) needs the real kernel. Each
harness measures what it is faithful to. The control-plane cost (belief update +
page scoring + table build) is timed and INCLUDED in gated timing (falsifier #3).

RoPE faithfulness: the new query token is always rotated at its absolute position
L (GatedCache.offset returns the absolute position), and cached keys retain their
original-position rotations, so q(L)·k(orig) has correct relative positions
regardless of which pages were dropped. Verified by the all-budgets==100%
equivalence check (KL≈0 when every page is kept).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import metrics as M
import page_policies as PP
import tasks as T
from belief_state import new_belief

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

BLOCK_SIZE = 16  # tokens per logical page (behavioral harness)
BUDGET_FRACTIONS = [1.0, 0.5, 0.25, 0.125, 0.0625]
POLICY_NAMES = [
    "full_pages", "recent_pages", "seeded_random_pages",
    "attention_proxy_pages", "keyword_entity_pages",
    "oracle_belief_pages", "inferred_belief_pages",
]

# ---------------------------------------------------------------------------
# Real-model behavioral harness
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def load_model(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    if model_id not in _MODEL_CACHE:
        from mlx_lm import load
        _MODEL_CACHE[model_id] = load(model_id)
    return _MODEL_CACHE[model_id]


class GatedCache:
    """mlx-lm-compatible cache holding a *compacted* set of cached K/V while
    reporting the *absolute* sequence position for RoPE.

    The model computes ``rope(q, offset=cache.offset)`` and
    ``rope(k_new, offset=cache.offset)`` then calls ``update_and_fetch``. By
    returning the absolute position from ``offset`` the new query/key get the
    correct rotation; the stored keys already carry their original-position
    rotations, so all relative positions are exact."""

    def __init__(self, keys, values, abs_offset: int):
        self.keys = keys      # [1, n_kv, J*B(+...), d], rotated at original positions
        self.values = values
        self._abs = int(abs_offset)

    @property
    def offset(self) -> int:
        return self._abs

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        self.keys = mx.concatenate([self.keys, keys], axis=2)
        self.values = mx.concatenate([self.values, values], axis=2)
        self._abs += keys.shape[2]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values


def tokenize_task(task: T.TaskInstance, tokenizer):
    """Return (ids, page_critical, page_keyword) where the two maps give the set
    of logical page indices that are critical (oracle) / keyword-flagged."""
    ids: list[int] = []
    seg_spans: list[tuple[int, int, bool, tuple[str, ...]]] = []
    prefix = ""
    prev_len = 0
    for text, crit, kw in task.segments:
        prefix += text
        cur = tokenizer.encode(prefix)
        seg_spans.append((prev_len, len(cur), crit, kw))
        prev_len = len(cur)
        ids = cur
    page_critical: set[int] = set()
    page_keyword: set[int] = set()
    kw_lower = set(k.lower() for k in task.keywords)
    for start, end, crit, kw in seg_spans:
        for tpos in range(start, end):
            pg = tpos // BLOCK_SIZE
            if crit:
                page_critical.add(pg)
    # keyword pages: pages whose decoded token text contains a keyword surface form
    for tpos, tid in enumerate(ids):
        piece = tokenizer.decode([tid]).strip().lower()
        if piece in kw_lower:
            page_keyword.add(tpos // BLOCK_SIZE)
    return ids, page_critical, page_keyword


@dataclass
class StepResult:
    kl: float
    js: float
    top1: bool
    topk: float
    R: int
    J: int


def _gather_keys(full_keys, keep_token_idx, mx):
    km = mx.array(np.asarray(keep_token_idx, dtype=np.int32))
    return full_keys[:, :, km, :]


def run_behavioral(model, tokenizer, task, policy: str, budget_fraction: float,
                   horizon: int, seed: int):
    """Teacher-forced behavioral trial. Returns (steps, accuracy, sys_overhead_s,
    selection_info)."""
    import mlx.core as mx
    ids, page_critical, page_keyword = tokenize_task(task, tokenizer)
    answer_ids = tokenizer.encode(task.prompt + task.answer)[len(ids):]
    if not answer_ids:
        answer_ids = tokenizer.encode(task.answer)
    horizon = min(horizon, max(1, len(answer_ids)))

    L0 = len(ids)
    # Prefill tokens[0:L0-1]; the last prompt token becomes the first decode query.
    layers = model.model.layers
    from mlx_lm.models.cache import KVCache
    full_caches = [KVCache() for _ in layers]
    prefill = mx.array([ids[:-1]])
    _ = model(prefill, cache=full_caches)
    mx.eval([c.keys for c in full_caches])

    # incremental belief state
    bel = new_belief(task.task_type, BLOCK_SIZE, task.entity_vocab)
    # feed prefill token text incrementally
    consumed = 0
    cfg = PP.PolicyConfig(budget_fraction=budget_fraction, recent_window=1,
                          sink_pages=1, seed=seed)
    page_mass: Optional[list[float]] = None

    steps: list[StepResult] = []
    produced_correct = True
    overhead_s = 0.0
    last_sel = None

    # teacher-forced decode over horizon
    cur_query = ids[-1]
    decode_pos = L0 - 1
    for h in range(horizon):
        seq = full_caches[0].offset  # number of cached tokens (positions 0..seq-1)
        n_pages = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = PP.SequencePages(
            block_ids=tuple(range(n_pages)), context_len=seq, block_size=BLOCK_SIZE)

        # --- control-plane work (timed; included in gated cost) ---
        t0 = time.perf_counter()
        # incremental belief update on newly-available tokens (prefix only)
        new_text_tokens = [tokenizer.decode([t]).strip() for t in ids[consumed:L0]]
        consumed = L0
        bel.update([w for w in new_text_tokens if w])
        inferred = {p for p in bel.active_pages() if p < n_pages}
        sel_ctx = PP.SelectionContext(
            page_attention_mass=page_mass,
            keyword_pages=tuple(p for p in page_keyword if p < n_pages),
            oracle_pages=tuple(p for p in page_critical if p < n_pages),
            inferred_pages=tuple(inferred),
        )
        sel = PP.select(policy, pages, cfg, sel_ctx)
        overhead_s += time.perf_counter() - t0
        last_sel = sel

        # --- build gated caches (token indices for selected pages) ---
        keep_tok: list[int] = []
        for pg in sel.selected_page_indices:
            start = pg * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, seq)
            keep_tok.extend(range(start, end))
        keep_tok = sorted(set(keep_tok))

        # full next-token distribution: decode query through FULL cache.
        # KVCache pre-allocates in chunks, so slice to :seq to drop padding —
        # otherwise the full reference would attend over uninitialized slots and
        # the all-pages==100% equivalence (KL≈0) would break.
        qx = mx.array([[cur_query]])
        full_step_caches = [GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq)
                            for c in full_caches]
        logits_full = model(qx, cache=full_step_caches)
        mx.eval(logits_full)
        pf = np.array(logits_full[0, -1].astype(mx.float32))

        gated_caches = [
            GatedCache(_gather_keys(c.keys, keep_tok, mx),
                       _gather_keys(c.values, keep_tok, mx), seq)
            for c in full_caches
        ]
        logits_gate = model(qx, cache=gated_caches)
        mx.eval(logits_gate)
        qg = np.array(logits_gate[0, -1].astype(mx.float32))

        steps.append(StepResult(
            kl=M.kl_divergence(pf, qg), js=M.js_divergence(pf, qg),
            top1=M.top1_agree(pf, qg), topk=M.topk_overlap(pf, qg, 10),
            R=len(keep_tok), J=sel.num_selected_pages))

        # exact-answer accuracy: does gated greedily produce the true token?
        if h < len(answer_ids):
            if int(np.argmax(qg)) != int(answer_ids[h]):
                produced_correct = False

        # advance BOTH caches with the TRUE next token (teacher forcing)
        true_tok = answer_ids[h] if h < len(answer_ids) else int(np.argmax(pf))
        adv = mx.array([[cur_query]])
        _ = model(adv, cache=full_caches)  # appends key for decode_pos
        mx.eval([full_caches[0].keys])
        cur_query = true_tok
        decode_pos += 1
        ids = ids + [true_tok]
        L0 += 1

    return steps, produced_correct, overhead_s, last_sel


# ---------------------------------------------------------------------------
# Real paged-kernel systems harness
# ---------------------------------------------------------------------------

_OPS = None


def get_ops():
    global _OPS
    if _OPS is None:
        import os
        os.environ.setdefault("VLLM_METAL_BUILD_FROM_SOURCE", "1")
        from vllm_metal.metal import get_ops as _g
        _OPS = _g()
    return _OPS


def paged_kernel_latency(context_len: int, keep_pages: list[int], block_size: int,
                         n_q_heads: int, n_kv_heads: int, head_dim: int,
                         iters: int = 30, warmup: int = 5, seed: int = 0):
    """Measure real decode kernel latency for a single attention call with the
    given compact page set. Returns (p50_ms, p95_ms, out_np)."""
    import mlx.core as mx
    ops = get_ops()
    rng = np.random.default_rng(seed)
    n_pages = (context_len + block_size - 1) // block_size
    dtype = mx.float16
    q = mx.array(rng.standard_normal((1, n_q_heads, head_dim)).astype(np.float32) * 0.1, dtype=dtype)
    kc = mx.array(rng.standard_normal((n_pages, block_size, n_kv_heads, head_dim)).astype(np.float32) * 0.1, dtype=dtype)
    vc = mx.array(rng.standard_normal((n_pages, block_size, n_kv_heads, head_dim)).astype(np.float32) * 0.1, dtype=dtype)
    cu = mx.array([0, 1], dtype=mx.int32)
    # compact context len from kept pages (full-page-except-tail invariant)
    def valid(pg):
        return block_size if pg < n_pages - 1 else (context_len - (n_pages - 1) * block_size or block_size)
    R = sum(valid(pg) for pg in keep_pages)
    bt = mx.array([keep_pages], dtype=mx.int32)

    def once():
        out = mx.zeros((1, n_q_heads, head_dim), dtype=dtype)
        ops.paged_attention_primitive(q, kc, vc, n_kv_heads, float(head_dim**-0.5), 0.0,
                                      bt, mx.array([R], dtype=mx.int32), cu,
                                      block_size, int(R), -1, out)
        mx.eval(out)
        return out
    for _ in range(warmup):
        once()
    ts = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter(); out = once(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    p50 = ts[len(ts) // 2]; p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    return p50, p95, np.array(out.astype(mx.float32))[0], R


def all_pages_equivalence(context_len: int, block_size: int, n_q_heads: int,
                          n_kv_heads: int, head_dim: int, seed: int = 0) -> float:
    """Max abs diff between full-block-table attention and explicit numpy
    reference; must be at fp16 noise floor (falsifier #8)."""
    import mlx.core as mx
    ops = get_ops()
    rng = np.random.default_rng(seed)
    n_pages = (context_len + block_size - 1) // block_size
    dtype = mx.float16
    qn = rng.standard_normal((1, n_q_heads, head_dim)).astype(np.float32) * 0.1
    kn = rng.standard_normal((n_pages, block_size, n_kv_heads, head_dim)).astype(np.float32) * 0.1
    vn = rng.standard_normal((n_pages, block_size, n_kv_heads, head_dim)).astype(np.float32) * 0.1
    q = mx.array(qn, dtype=dtype); kc = mx.array(kn, dtype=dtype); vc = mx.array(vn, dtype=dtype)
    bt = mx.array([list(range(n_pages))], dtype=mx.int32)
    out = mx.zeros((1, n_q_heads, head_dim), dtype=dtype)
    ops.paged_attention_primitive(q, kc, vc, n_kv_heads, float(head_dim**-0.5), 0.0,
                                  bt, mx.array([context_len], dtype=mx.int32),
                                  mx.array([0, 1], dtype=mx.int32), block_size,
                                  int(context_len), -1, out)
    mx.eval(out)
    got = np.array(out.astype(mx.float32))[0]
    # reference
    K = kn.reshape(n_pages * block_size, n_kv_heads, head_dim)[:context_len].astype(np.float64)
    V = vn.reshape(n_pages * block_size, n_kv_heads, head_dim)[:context_len].astype(np.float64)
    ref = np.zeros((n_q_heads, head_dim))
    g = n_q_heads // n_kv_heads
    for hh in range(n_q_heads):
        kvh = hh // g
        lg = (qn[0, hh].astype(np.float64) @ K[:, kvh].T) * (head_dim ** -0.5)
        w = np.exp(lg - lg.max()); w /= w.sum()
        ref[hh] = w @ V[:, kvh]
    return float(np.abs(got - ref).max())


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(context_lengths, seeds, families, horizon, model_id, out_dir: Path,
              measure_systems: bool = True):
    model, tokenizer = load_model(model_id)
    # discover model attention config for the systems harness
    a0 = model.model.layers[0].self_attn
    n_q = a0.n_heads; n_kv = a0.n_kv_heads
    head_dim = model.args.hidden_size // a0.n_heads
    n_layers = len(model.model.layers)
    print(f"model: {n_layers} layers, n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    raw = open(raw_path, "w")
    agg = M.Aggregate()

    # systems latency cache keyed by (context_len, J) — policy-independent
    sys_cache: dict = {}

    for fam in families:
        for n_filler in context_lengths:  # n_filler ~ controls context length
            for seed in seeds:
                task = T.make_task(fam, seed, n_filler)
                # validity: full-cache must solve it
                steps0, ok0, _, _ = run_behavioral(model, tokenizer, task,
                                                    "full_pages", 1.0, horizon, seed)
                if not ok0:
                    raw.write(json.dumps({"event": "rejected", "family": fam,
                                          "n_filler": n_filler, "seed": seed}) + "\n")
                    continue
                for policy in POLICY_NAMES:
                    for bf in (BUDGET_FRACTIONS if policy != "full_pages" else [1.0]):
                        steps, ok, overhead, sel = run_behavioral(
                            model, tokenizer, task, policy, bf, horizon, seed)
                        kls = [s.kl for s in steps]
                        D_h = float(np.mean(kls))
                        top1 = float(np.mean([s.top1 for s in steps]))
                        R = steps[-1].R; J = steps[-1].J
                        rec = {
                            "family": fam, "n_filler": n_filler, "seed": seed,
                            "policy": policy, "budget_fraction": bf,
                            "D_h_kl": D_h, "js": float(np.mean([s.js for s in steps])),
                            "top1_agree": top1,
                            "topk_overlap": float(np.mean([s.topk for s in steps])),
                            "answer_correct": bool(ok),
                            "R_tokens": R, "J_pages": J,
                            "control_overhead_s": overhead,
                            "horizon": len(steps),
                        }
                        raw.write(json.dumps(rec) + "\n")
                        agg.add(rec)
                        print(f"{fam:22s} nf={n_filler:3d} s={seed} {policy:22s} "
                              f"bf={bf:6.4f} D_h={D_h:7.4f} top1={top1:4.2f} "
                              f"R={R:4d} J={J:3d} corr={ok}")
    raw.close()

    # systems pass: equivalence + latency vs budget at a representative context
    sys_rows = []
    if measure_systems:
        for ctxL in [512, 1024, 2048]:
            eq = all_pages_equivalence(ctxL, 16, n_q, n_kv, head_dim)
            n_pages = (ctxL + 15) // 16
            full_pages_list = list(range(n_pages))
            pf50, pf95, _, _ = paged_kernel_latency(ctxL, full_pages_list, 16, n_q, n_kv, head_dim)
            for bf in BUDGET_FRACTIONS:
                J = max(1, round(bf * n_pages))
                keep = sorted(set(list(range(n_pages - 1, n_pages - J, -1)) + [n_pages - 1]))[:J]
                if not keep:
                    keep = [n_pages - 1]
                kp50, kp95, _, R = paged_kernel_latency(ctxL, keep, 16, n_q, n_kv, head_dim)
                row = {"context_len": ctxL, "all_pages_max_abs_err": eq,
                       "budget_fraction": bf, "J_pages": len(keep), "R_tokens": R,
                       "full_p50_ms": pf50, "gated_p50_ms": kp50,
                       "full_p95_ms": pf95, "gated_p95_ms": kp95,
                       "cache_bytes_full": M.cache_bytes(n_pages, 16, n_layers, n_kv, head_dim),
                       "cache_bytes_gated": M.cache_bytes(len(keep), 16, n_layers, n_kv, head_dim)}
                sys_rows.append(row)
                print(f"[sys] L={ctxL} bf={bf:6.4f} J={len(keep):3d} eq={eq:.2e} "
                      f"full={pf50:.3f}ms gated={kp50:.3f}ms")

    _write_aggregate(agg, sys_rows, out_dir)
    return agg, sys_rows


def _write_aggregate(agg: M.Aggregate, sys_rows, out_dir: Path):
    # aggregate.csv: mean D_h, top1, answer-acc per (policy, budget), with CI
    by: dict = {}
    for r in agg.records:
        key = (r["policy"], r["budget_fraction"])
        by.setdefault(key, []).append(r)
    rows = []
    for (policy, bf), rs in sorted(by.items()):
        dh = [r["D_h_kl"] for r in rs]
        acc = [1.0 if r["answer_correct"] else 0.0 for r in rs]
        t1 = [r["top1_agree"] for r in rs]
        ov = [r["control_overhead_s"] for r in rs]
        dh_m, dh_ci = M.Aggregate.mean_ci(dh)
        acc_m, acc_ci = M.Aggregate.mean_ci(acc)
        rows.append({"policy": policy, "budget_fraction": bf, "n": len(rs),
                     "D_h_mean": dh_m, "D_h_ci95": dh_ci,
                     "answer_acc_mean": acc_m, "answer_acc_ci95": acc_ci,
                     "top1_mean": float(np.mean(t1)),
                     "control_overhead_ms_mean": float(np.mean(ov)) * 1e3})
    with open(out_dir / "aggregate.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(out_dir / "systems.csv", "w", newline="") as f:
        if sys_rows:
            w = csv.DictWriter(f, fieldnames=list(sys_rows[0].keys()))
            w.writeheader(); w.writerows(sys_rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+", default=list(T.FAMILIES.keys()))
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-systems", action="store_true")
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()
    run_sweep(args.context_lengths, args.seeds, args.families, args.horizon,
              args.model, Path(args.out), measure_systems=not args.no_systems)


if __name__ == "__main__":
    main()
