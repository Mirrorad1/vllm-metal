# SPDX-License-Identifier: Apache-2.0
"""exp014 — Finest latent probe: are a few individual QUERY-HEADS (retrieval heads)
load-bearing, even though no page (exp011) and no kv-head (exp013) is?

exp013 ablated whole kv-heads (coarse: GQA groups 7 query-heads per kv-head). This
ablates each INDIVIDUAL query head's attention OUTPUT (zeroing its contribution to
the residual stream before o_proj), the finest latent unit, to test the
"retrieval-head" hypothesis: is the answer carried by a sparse set of latent heads?

If even a single q-head is load-bearing (ablating it flips the answer) while no
page/kv-head is → the latent structure IS concentrated in retrieval-heads → the
true lacuna is retrieval-head-indexed KV reduction. If q-heads are ALSO redundant →
the holographic-redundancy conclusion is locked and the only open cell is SYSTEMS.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import error_metrics as EM
import exp011_tail as E

_RES = Path(__file__).resolve().parent / "results"
_ABL = {"units_byL": {}, "L": 0, "Hq": 0}


@contextlib.contextmanager
def ablate_qheads(units):
    """Zero the attention OUTPUT of each (layer, q_head) in `units` for one forward."""
    import mlx.core as mx
    import mlx_lm.models.qwen2 as q2
    orig = q2.scaled_dot_product_attention
    byL = {}
    for (L, h) in units:
        byL.setdefault(L, []).append(h)
    _ABL["L"] = 0

    def wrapped(q, k, v, cache, scale, mask, sinks=None):
        out = orig(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)  # [B,Hq,Lq,d]
        L = _ABL["L"]; _ABL["L"] += 1
        if L in byL:
            Hq = out.shape[1]
            m = np.ones(Hq, dtype=np.float32)
            for h in byL[L]:
                m[h] = 0.0
            out = out * mx.array(m, dtype=out.dtype).reshape(1, Hq, 1, 1)
        return out

    q2.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        q2.scaled_dot_product_attention = orig


def _full_caches(caches, seq):
    return [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq) for c in caches]


def _forward(model, caches, query, ans0, units=()):
    import mlx.core as mx
    with ablate_qheads(units):
        z = model(mx.array([[query]]), cache=caches)
        mx.eval(z)
    zf = np.array(z[0, -1].astype(mx.float32))
    return int(np.argmax(zf)) == ans0, E.gold_logp(zf, ans0), zf


def run_example(model, tokenizer, prompt, answer, budgets):
    cfg = H.model_config(model)
    NL, Hq = cfg["n_layers"], cfg["n_q"]
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    ans0 = int(answer_ids[0])
    caches, query, seq = H.prefill(model, ids)
    fc = _full_caches(caches, seq)
    full_ok, lp_full, _ = _forward(model, fc, query, ans0)

    units = [(L, h) for L in range(NL) for h in range(Hq)]
    dlp = {}
    flips = 0
    for u in units:
        ok, lp, _ = _forward(model, _full_caches(caches, seq), query, ans0, units=[u])
        dlp[u] = lp_full - lp
        flips += int(not ok)
    vals = np.array([dlp[u] for u in units])

    order = sorted(units, key=lambda u: dlp[u], reverse=True)
    budget_acc = {}
    for k in budgets:
        drop = order[k:]
        ok, _, _ = _forward(model, _full_caches(caches, seq), query, ans0, units=drop)
        budget_acc[k] = int(ok)
    return {"full_correct": bool(full_ok), "n_units": len(units),
            "single_flip_rate": flips / len(units), "max_single_dlp": float(vals.max()),
            "top1_dlp": float(np.sort(vals)[-1]), "top5_dlp_sum": float(np.sort(vals)[-5:].sum()),
            "frac_dlp_gt0.5": float(np.mean(vals > 0.5)), "budget_acc": budget_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-filler", type=int, default=210)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", default=str(_RES / "exp014_qheads"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    cfg = H.model_config(model)
    nu = cfg["n_layers"] * cfg["n_q"]
    budgets = [k for k in args.budgets if k <= nu]
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
        if acc % 5 == 0:
            print(f"accepted {acc}/{args.n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    def m(k): return float(np.mean([r[k] for r in recs]))
    agg = {"n": len(recs), "n_units": nu,
           "single_flip_rate_mean": m("single_flip_rate"),
           "max_single_dlp_mean": m("max_single_dlp"), "top5_dlp_sum_mean": m("top5_dlp_sum"),
           "frac_dlp_gt0.5_mean": m("frac_dlp_gt0.5"),
           "budget_acc": {str(k): float(np.mean([r["budget_acc"][k] for r in recs])) for k in budgets}}
    kstar = next((k for k in budgets if agg["budget_acc"][str(k)] >= 0.9), nu)
    agg["qhead_kstar_0.9"] = kstar
    sfr = agg["single_flip_rate_mean"]; msd = agg["max_single_dlp_mean"]
    if sfr > 0.01 or msd > 0.5:
        verdict = "RETRIEVAL-HEAD LOCALIZED (latent lacuna)"
        reason = (f"{sfr*100:.2f}% of query-heads individually flip the answer (max single-qhead Δlogp "
                  f"{msd:.2f} nat); {kstar}/{nu} q-heads suffice for 90% acc — while NO page (exp011) or "
                  f"kv-head (exp013) is load-bearing. The answer IS concentrated in a few latent retrieval "
                  f"heads. True lacuna: retrieval-head-indexed (latent) KV reduction — invisible to "
                  f"page/token-indexed operators.")
    elif sfr < 0.003 and msd < 0.3:
        verdict = "HOLOGRAPHIC LOCKED (no sparse latent unit at ANY granularity)"
        reason = (f"even at the finest latent unit (individual query-head), single-head flip rate {sfr*100:.2f}% "
                  f"and max Δlogp {msd:.2f} nat — no q-head is load-bearing, mirroring pages and kv-heads. "
                  f"The answer is HOLOGRAPHICALLY distributed across token × head × layer; cross-layer "
                  f"redundancy dominates. No sparse latent unit exists for any operator to target. The "
                  f"'true lacuna through hidden latent states' search returns NULL — the latent states hide "
                  f"no exploitable sparse structure beyond the modest 4× low-rank (exp012). The ONLY "
                  f"genuinely-empty cell is SYSTEMS: physical reclamation of dropped/factored bytes.")
    else:
        verdict = "MIXED / INCONCLUSIVE"
        reason = f"single-flip {sfr*100:.2f}%, max Δlogp {msd:.2f}; partial q-head concentration ({kstar}/{nu} for 90%)."
    agg["verdict"] = verdict; agg["verdict_reason"] = reason
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp014 VERDICT:", verdict, "===")
    print(reason)
    print(f"single-qhead flip {sfr*100:.2f}% | max single-qhead Δlogp {msd:.2f} nat | "
          f"q-head budget for 90% acc = {kstar}/{nu}")
    print("budget_acc:", agg["budget_acc"])
    return agg


if __name__ == "__main__":
    main()
