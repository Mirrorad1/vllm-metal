# SPDX-License-Identifier: Apache-2.0
"""exp013 — Is the answer localized in LATENT HEAD-space though delocalized in token/page-space?

exp011 found NO single page is load-bearing (ablating one page moves the answer
logprob ~0.0009 nat — prefill information diffusion spreads each fact across many
downstream tokens). The latent analog, pursuing "hidden latent states": is any
single ATTENTION HEAD (a latent channel) load-bearing? If a few (layer, kv-head)
units individually carry the answer while no page does, the exploitable structure
lives in latent HEAD-space, not token/page-space — and the field's page-indexed
operators are at the wrong granularity (the true lacuna is latent-head/direction
indexed KV reduction).

Method (real model, mlx-lm, reuses exp011 needle prompts):
 - Ablate one (layer, kv-head) at the decode step by ZEROING that head's VALUES
   (removes its output contribution) → Δlogp(gold answer) + top1-flip.
   → single-head load-bearing RATE (contrast with single-page ~0).
 - Head-budget curve: rank units by importance, keep top-k heads' V (zero the
   rest), measure answer accuracy vs k → how many latent heads the answer needs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import error_metrics as EM
import exp011_tail as E

_RES = Path(__file__).resolve().parent / "results"


def _caches_full(caches, seq):
    import mlx.core as mx
    return [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq) for c in caches]


def _forward(model, layer_caches, query, ans0):
    import mlx.core as mx
    z = model(mx.array([[query]]), cache=layer_caches)
    mx.eval(z)
    zf = np.array(z[0, -1].astype(mx.float32))
    return int(np.argmax(zf)) == ans0, E.gold_logp(zf, ans0), zf


def _ablate_units(caches, seq, units):
    """GatedCache list with each (layer, kv_head) in `units` having its VALUES zeroed."""
    import mlx.core as mx
    byL = {}
    for (L, h) in units:
        byL.setdefault(L, []).append(h)
    out = []
    for L, c in enumerate(caches):
        V = c.values[:, :, :seq, :]
        if L in byL:
            Vn = np.array(V.astype(mx.float32))
            for h in byL[L]:
                Vn[0, h, :, :] = 0.0
            V = mx.array(Vn, dtype=mx.float16)
        out.append(H.GatedCache(c.keys[:, :, :seq, :], V, seq))
    return out


def run_example(model, tokenizer, prompt, answer, head_budgets):
    import mlx.core as mx
    cfg = H.model_config(model)
    NL, NKV = cfg["n_layers"], cfg["n_kv"]
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    ans0 = int(answer_ids[0])
    caches, query, seq = H.prefill(model, ids)

    full_ok, lp_full, z_full = _forward(model, _caches_full(caches, seq), query, ans0)

    # single (layer, kv-head) ablation importance
    units = [(L, h) for L in range(NL) for h in range(NKV)]
    dlps = {}
    flips = 0
    for u in units:
        ok, lp, z = _forward(model, _ablate_units(caches, seq, [u]), query, ans0)
        dlps[u] = lp_full - lp           # >0 ⇒ this head mattered (logprob dropped)
        flips += int(not ok)             # did ablating THIS one head flip the answer?
    single_flip_rate = flips / len(units)
    # importance distribution
    vals = np.array([dlps[u] for u in units])

    # head-budget curve: keep top-k units (by importance), zero the rest
    order = sorted(units, key=lambda u: dlps[u], reverse=True)
    budget_acc = {}
    for k in head_budgets:
        keep = set(order[:k])
        drop = [u for u in units if u not in keep]
        ok, lp, z = _forward(model, _ablate_units(caches, seq, drop), query, ans0)
        budget_acc[k] = int(ok)

    return {
        "full_correct": bool(full_ok), "n_units": len(units), "seq": seq,
        "single_flip_rate": single_flip_rate,        # fraction of heads individually load-bearing
        "max_single_dlp": float(vals.max()),         # biggest single-head answer impact (nats)
        "top1_dlp": float(np.sort(vals)[-1]),
        "top3_dlp_sum": float(np.sort(vals)[-3:].sum()),
        "dlp_p99": float(np.percentile(vals, 99)),
        "budget_acc": budget_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-filler", type=int, default=210)
    ap.add_argument("--head-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32, 48])
    ap.add_argument("--out", default=str(_RES / "exp013_heads"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    cfg = H.model_config(model)
    n_units = cfg["n_layers"] * cfg["n_kv"]
    budgets = [k for k in args.head_budgets if k <= n_units]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "raw_results.jsonl", "w")
    recs = []; seed = 0; acc = 0; t0 = time.perf_counter()
    while acc < args.n and seed < args.n * 5:
        depth = 0.2 + 0.5 * ((seed % 7) / 6.0)
        p, a, c = E.make_needle_prompt(tok, seed, args.n_filler, depth)
        seed += 1
        try:
            r = run_example(model, tok, p, a, budgets)
        except Exception as e:
            raw.write(json.dumps({"event": "error", "seed": seed, "err": repr(e)[:200]}) + "\n")
            continue
        if not r["full_correct"]:
            continue
        acc += 1; r["seed"] = seed; recs.append(r); raw.write(json.dumps(r) + "\n")
        if acc % 10 == 0:
            print(f"accepted {acc}/{args.n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    def m(k): return float(np.mean([r[k] for r in recs]))
    agg = {"n": len(recs), "n_units": n_units,
           "single_flip_rate_mean": m("single_flip_rate"),
           "max_single_dlp_mean": m("max_single_dlp"),
           "top1_dlp_mean": m("top1_dlp"), "top3_dlp_sum_mean": m("top3_dlp_sum"),
           "budget_acc": {str(k): float(np.mean([r["budget_acc"][k] for r in recs]))
                          for k in budgets}}
    # smallest head-budget reaching >=0.9 answer accuracy
    kstar = next((k for k in budgets if agg["budget_acc"][str(k)] >= 0.9), n_units)
    agg["head_kstar_0.9"] = kstar
    agg["page_baseline"] = "exp011: single-page ablation Δlp≈0.0009 (NO page individually load-bearing)"

    sfr = agg["single_flip_rate_mean"]; msd = agg["max_single_dlp_mean"]
    if sfr > 0.02 or msd > 0.5:
        verdict = "LATENT HEAD-SPARSE (answer concentrated in few heads)"
        reason = (f"{sfr*100:.1f}% of (layer,head) units individually FLIP the answer when ablated "
                  f"(max single-head Δlogp {msd:.2f} nat), and {kstar}/{n_units} heads suffice for "
                  f">=90% accuracy — while NO single PAGE is load-bearing (exp011 Δlp≈0.0009). The answer "
                  f"is DELOCALIZED in token/page-space but LOCALIZED in latent HEAD-space. The field's "
                  f"page/token-indexed operators are at the wrong granularity; the latent structure lives "
                  f"in a few heads (cf. retrieval-heads / DuoAttention) — and possibly finer (directions "
                  f"within heads). True lacuna candidate: latent-head/direction-indexed KV reduction.")
    elif sfr < 0.005 and msd < 0.2:
        verdict = "LATENT HEAD-DENSE too (redundant in head-space as well)"
        reason = (f"only {sfr*100:.1f}% of heads individually flip the answer (max Δlogp {msd:.2f} nat): the "
                  f"answer is redundant across HEADS as well as pages — diffusion is total. No sparse latent "
                  f"unit to exploit; the redundancy is genuinely compute-graph/distributed. Deeper conclusion: "
                  f"the exploitable structure is neither page nor head; pivot to systems.")
    else:
        verdict = "MIXED / INCONCLUSIVE"
        reason = f"single-flip {sfr*100:.1f}%, max single Δlogp {msd:.2f}; partial head concentration."
    agg["verdict"] = verdict; agg["verdict_reason"] = reason
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp013 VERDICT:", verdict, "===")
    print(reason)
    print(f"single-head flip rate {sfr*100:.1f}% | max single-head Δlogp {msd:.2f} nat | "
          f"head budget for 90% acc = {kstar}/{n_units}")
    print("budget_acc:", agg["budget_acc"])
    return agg


if __name__ == "__main__":
    main()
