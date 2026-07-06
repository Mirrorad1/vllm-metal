# SPDX-License-Identifier: Apache-2.0
"""exp012 — Is the prefill-diffusion redundancy in the STORED BYTES or the COMPUTE GRAPH?

exp011 showed page choice barely affects the ANSWER (token-sparsity redundancy is
abundant — drop 94% of pages, still answer). The recursion's decisive fork: is the
redundancy that makes pages droppable also GEOMETRIC (the retained KV lives in a
low-dim subspace → a byte-level low-rank/dedup factorizer could reclaim bytes,
STACKING with eviction → the new lacuna is fillable), or is it purely
token-IDENTITY/sparsity + compute-graph (near-full-rank KV → no stored-byte
operator beyond dropping → the lacuna is empty, the move is recompute not reclaim)?

Three measurements (real model, mlx-lm, reuses exp011 needle prompts):
 1. EFFECTIVE RANK of cached K and V over retained tokens, per layer (stable rank +
    spectral-entropy effective rank) vs ambient head_dim. V (no RoPE) is the clean
    content-redundancy measure.
 2. BYTE-MATCHED HEAD-TO-HEAD at a budget where selection is near-lossless: spend the
    SAME bytes on (S) page-drop selection vs (F) global low-rank projection of ALL
    tokens to rank r_eq. Which structure does the byte budget actually buy?
 3. STACKING: low-rank-project the SELECTED pages' KV to rank r; smallest r* that
    preserves the answer. r* ≪ d ⇒ low-rank stacks multiplicatively on eviction
    (bytes real & exploitable); r* ≈ d ⇒ no subspace lever (compute-graph).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import error_metrics as EM
import exp011_tail as E

B = H.BLOCK_SIZE
_RES = Path(__file__).resolve().parent / "results"


def effective_rank(M):
    """(stable_rank, spectral_entropy_ER) of a [n, d] matrix."""
    if M.shape[0] < 2:
        return (1.0, 1.0)
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    s = s[s > 1e-9]
    if s.size == 0:
        return (0.0, 0.0)
    stable = float((s ** 2).sum() / (s[0] ** 2))      # ||M||_F^2 / ||M||_2^2
    p = s / s.sum()
    ent = float(np.exp(-(p * np.log(p + 1e-12)).sum()))  # participation / spectral entropy
    return (stable, ent)


def _proj(M, r):
    """Rank-r truncation of [n, d]."""
    r = int(min(r, min(M.shape)))
    U, S, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r]


def lowrank_caches(model, caches, keep_tok, r, seq):
    """Per-layer GatedCache holding selected tokens' K/V, each rank-r projected."""
    import mlx.core as mx
    out = []
    keep = np.asarray(keep_tok, dtype=np.int64)
    for c in caches:
        K = np.array(c.keys[0, :, :seq, :].astype(mx.float32))[:, keep, :]   # [nkv, nk, d]
        V = np.array(c.values[0, :, :seq, :].astype(mx.float32))[:, keep, :]
        for h in range(K.shape[0]):
            K[h] = _proj(K[h], r)
            V[h] = _proj(V[h], r)
        out.append(H.GatedCache(mx.array(K[None], dtype=mx.float16),
                                mx.array(V[None], dtype=mx.float16), seq))
    return out


def forward_answer(model, layer_caches, query, ans0):
    import mlx.core as mx
    z = model(mx.array([[query]]), cache=layer_caches)
    mx.eval(z)
    zf = np.array(z[0, -1].astype(mx.float32))
    return int(np.argmax(zf)) == ans0, E.gold_logp(zf, ans0), zf


def run_example(model, tokenizer, prompt, answer, sel_bf, rank_grid):
    import mlx.core as mx
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    ans0 = int(answer_ids[0])
    caches, query, seq = H.prefill(model, ids)
    n_pages = (seq + B - 1) // B
    d = H.model_config(model)["head_dim"]

    # full reference
    full_caches = [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq) for c in caches]
    full_ok, full_lp, z_full = forward_answer(model, full_caches, query, ans0)

    # 1. effective rank of cached K, V over ALL retained tokens (avg over layers, kv-heads)
    srK = srV = erK = erV = 0.0
    cnt = 0
    for c in caches:
        K = np.array(c.keys[0, :, :seq, :].astype(mx.float32))
        V = np.array(c.values[0, :, :seq, :].astype(mx.float32))
        for h in range(K.shape[0]):
            sk, ek = effective_rank(K[h]); sv, ev = effective_rank(V[h])
            srK += sk; erK += ek; srV += sv; erV += ev; cnt += 1
    rankrec = {"d": d, "stable_rank_K": srK / cnt, "stable_rank_V": srV / cnt,
               "eff_rank_K": erK / cnt, "eff_rank_V": erV / cnt}

    # selection token set at sel_bf (recent-J for a fixed, content-blind selector so
    # the head-to-head is selection-structure vs low-rank-structure, not signal choice)
    J = max(1, round(sel_bf * n_pages))
    keep_pages = sorted(set(list(range(n_pages - J, n_pages)) + [0]))[:J] or [n_pages - 1]
    # use the strong attention selection too (so S is the best sparsity can do)
    _, page_mass = H.full_step(model, caches, query, seq, want_mass=True)
    import page_policies as PP
    pages = PP.SequencePages(tuple(range(n_pages)), seq, B)
    sel = PP.select("attention_proxy_pages", pages, PP.PolicyConfig(budget_pages=J, seed=0),
                    PP.Signals(attention_mass=list(page_mass)))
    sel_pages = sorted(sel.selected_page_indices)
    keep_tok_sel = [t for p in sel_pages for t in range(p * B, min(p * B + B, seq))]
    n_sel = len(keep_tok_sel)
    sel_bytes = n_sel * d  # per K (and per V), full rank

    # (S) selection only (r=d ⇒ no projection, just the selected tokens)
    selS = lowrank_caches(model, caches, keep_tok_sel, d, seq)
    s_ok, s_lp, zS = forward_answer(model, selS, query, ans0)

    # (F) pure global low-rank, ALL tokens, byte-matched to selection:
    #   storing rank-r of [seq, d] costs (seq + d) * r ; match to sel_bytes = n_sel*d
    r_eq = max(1, int(round(sel_bytes / (seq + d))))
    allF = lowrank_caches(model, caches, list(range(seq)), r_eq, seq)
    f_ok, f_lp, zF = forward_answer(model, allF, query, ans0)

    # (Stack) selection + low-rank on the retained tokens, sweep r
    stack = []
    for r in rank_grid:
        cc = lowrank_caches(model, caches, keep_tok_sel, r, seq)
        ok, lp, z = forward_answer(model, cc, query, ans0)
        # bytes for storing rank-r of [n_sel, d]: (n_sel + d)*r  (per K and per V)
        stack_bytes = (n_sel + d) * r
        stack.append({"r": r, "ok": bool(ok), "lp": lp, "kl": EM.kl(z_full, z),
                      "bytes_vs_full": stack_bytes / (seq * d)})
        if not ok:
            break  # smallest r that preserves answer is just above this

    return {
        "full_correct": bool(full_ok), "n_pages": n_pages, "seq": seq,
        **rankrec,
        "n_sel_tokens": n_sel, "sel_J": J, "sel_bytes_frac": sel_bytes / (seq * d),
        "S_correct": bool(s_ok), "S_kl": EM.kl(z_full, zS),
        "r_eq": r_eq, "F_correct": bool(f_ok), "F_kl": EM.kl(z_full, zF),
        "stack": stack,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-filler", type=int, default=210)
    ap.add_argument("--sel-bf", type=float, default=0.0625)  # near-lossless selection regime
    ap.add_argument("--rank-grid", type=int, nargs="+", default=[64, 32, 16, 8, 4, 2, 1])
    ap.add_argument("--out", default=str(_RES / "exp012_rank"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    d = H.model_config(model)["head_dim"]
    rank_grid = [r for r in args.rank_grid if r <= d]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "raw_results.jsonl", "w")
    recs = []
    seed = 0; acc = 0; t0 = time.perf_counter()
    while acc < args.n and seed < args.n * 5:
        depth = 0.2 + 0.5 * ((seed % 7) / 6.0)
        p, a, c = E.make_needle_prompt(tok, seed, args.n_filler, depth)
        seed += 1
        try:
            r = run_example(model, tok, p, a, args.sel_bf, rank_grid)
        except Exception as e:
            raw.write(json.dumps({"event": "error", "seed": seed, "err": repr(e)[:200]}) + "\n")
            continue
        if not r["full_correct"]:
            continue
        acc += 1; r["seed"] = seed; recs.append(r); raw.write(json.dumps(r) + "\n")
        if acc % 10 == 0:
            print(f"accepted {acc}/{args.n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    # aggregate
    def m(k): return float(np.mean([r[k] for r in recs]))
    agg = {"n": len(recs), "d": d,
           "eff_rank_K": m("eff_rank_K"), "eff_rank_V": m("eff_rank_V"),
           "stable_rank_K": m("stable_rank_K"), "stable_rank_V": m("stable_rank_V"),
           "S_acc": m("S_correct"), "S_kl": m("S_kl"),
           "F_acc": m("F_correct"), "F_kl": m("F_kl"), "r_eq": int(np.median([r["r_eq"] for r in recs])),
           "sel_bytes_frac": m("sel_bytes_frac")}
    # stacking: answer-accuracy at each rank, and smallest r* preserving answer per example
    rstars = []
    rank_acc = {}
    for r in rank_grid:
        oks = [next((s["ok"] for s in rec["stack"] if s["r"] == r), False) for rec in recs]
        rank_acc[r] = float(np.mean(oks))
    for rec in recs:
        ok_ranks = [s["r"] for s in rec["stack"] if s["ok"]]
        rstars.append(min(ok_ranks) if ok_ranks else rec["d"])
    agg["stack_acc_by_rank"] = rank_acc
    agg["rstar_median"] = float(np.median(rstars))
    agg["rstar_mean"] = float(np.mean(rstars))

    # verdict — the bytes-vs-graph fork
    er_v = agg["eff_rank_V"]; d_ = float(d)
    stack_mult = agg["rstar_median"] / d_   # byte multiplier of low-rank stacked on eviction (~r*/d)
    f_competitive = agg["F_acc"] >= agg["S_acc"] - 0.1  # does byte-matched low-rank match selection?
    low_rank_V = er_v < 0.5 * d_                 # strong low-rank subspace
    modest_stack = stack_mult <= 0.5             # some low-rank lever stacks on eviction (r* <= d/2)
    f_wins = f_competitive                        # byte-matched low-rank matches/beats selection
    agg["stack_byte_multiplier"] = stack_mult
    agg["F_vs_S_competitive"] = f_wins
    if f_wins and low_rank_V:
        verdict = "BYTES-DOMINANT (low-rank is a primary lever, fillable)"
        reason = (f"V eff-rank {er_v:.1f}≪{d}; byte-matched low-rank MATCHES selection (F_acc {agg['F_acc']:.2f} "
                  f"vs S_acc {agg['S_acc']:.2f}). The redundancy lives in a low-dim subspace — a low-rank "
                  f"reclaimer is a primary byte lever. Lacuna FILLABLE.")
    elif modest_stack and not f_wins:
        verdict = "SPARSITY-DOMINANT + modest low-rank stack (partially fillable)"
        reason = (f"At equal bytes, TOKEN-SELECTION beats global low-rank (S_acc {agg['S_acc']:.2f} vs F_acc "
                  f"{agg['F_acc']:.2f}, r_eq={agg['r_eq']}): the dominant redundancy is token-SPARSITY, which "
                  f"existing eviction already harvests. BUT the answer tolerates low-rank to r*≈{agg['rstar_median']:.0f} "
                  f"({stack_mult:.2f}× of d) STACKED on selection — a modest, real low-rank byte lever (V eff-rank "
                  f"{er_v:.1f}/{d}). So the new lacuna is PARTIALLY fillable: a smear-aware low-rank reclaimer "
                  f"composes with eviction for ~{1/max(stack_mult,1e-6):.1f}× extra, but is the secondary lever, "
                  f"not the dominant one. This reproduces from first principles WHY the field stacks "
                  f"eviction×quant×low-rank: orthogonal redundancy structures.")
    elif not modest_stack:
        verdict = "SPARSITY-ONLY / compute-graph (not byte-addressable beyond dropping)"
        reason = (f"Selection beats low-rank at equal bytes AND any rank below r≈{agg['rstar_median']:.0f} destroys "
                  f"the answer (V eff-rank {er_v:.1f}/{d}, near full). The redundancy is token-IDENTITY/sparsity + "
                  f"recomputable computation, NOT a subspace — no stored-byte factorizer beyond dropping; "
                  f"move = recompute, not reclaim. Lacuna EMPTY for a linear byte operator.")
    else:
        verdict = "MIXED / INCONCLUSIVE"
        reason = (f"partial structure (V eff-rank {er_v:.1f}/{d}, r*≈{agg['rstar_median']:.0f}, "
                  f"F_acc {agg['F_acc']:.2f} vs S_acc {agg['S_acc']:.2f}); sharpen N or rank grid.")
    agg["verdict"] = verdict
    agg["verdict_reason"] = reason
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp012 VERDICT:", verdict, "===")
    print(reason)
    print(f"V eff-rank {agg['eff_rank_V']:.1f}/{d}  K eff-rank {agg['eff_rank_K']:.1f}/{d}  "
          f"r*median={agg['rstar_median']:.0f}")
    print("S(selection) acc=%.2f kl=%.3f | F(lowrank r=%d, byte-matched) acc=%.2f kl=%.3f"%(
        agg["S_acc"], agg["S_kl"], agg["r_eq"], agg["F_acc"], agg["F_kl"]))
    print("stack acc by rank:", {r: round(rank_acc[r], 2) for r in rank_grid})
    return agg


if __name__ == "__main__":
    main()
