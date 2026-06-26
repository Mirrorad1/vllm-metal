# SPDX-License-Identifier: Apache-2.0
"""exp016 — Is there a UNIVERSAL (cross-prompt) low-rank KV subspace, or is it prompt-specific?

exp012 found a per-PROMPT rank-16 subspace preserves the answer. The distinct latent
question: is that subspace SHARED across prompts? Fit a fixed low-rank basis per
(layer, kv-head) on CALIBRATION prompts; project HELD-OUT prompts' KV onto that fixed
basis; measure answer accuracy vs rank. Compare the universal rank needed to the
per-example rank.

If universal-r ≈ per-example-r (small) → a real static UNIVERSAL latent KV code exists
(a fit-once codebook → genuine lever). If universal-r ≫ per-example-r → the low-rank
structure is prompt-specific (no universal latent state), and the latent search closes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import exp011_tail as E

_RES = Path(__file__).resolve().parent / "results"


def collect_calib(model, tokenizer, n_calib, n_filler, tok_subsample=150):
    """Return per-(layer, kv_head) universal bases for K and V (orthonormal rows, rmax=d)."""
    import mlx.core as mx
    cfg = H.model_config(model)
    NL, NKV, d = cfg["n_layers"], cfg["n_kv"], cfg["head_dim"]
    Kacc = {(L, h): [] for L in range(NL) for h in range(NKV)}
    Vacc = {(L, h): [] for L in range(NL) for h in range(NKV)}
    used = 0
    seed = 10_000
    while used < n_calib and seed < 10_000 + n_calib * 4:
        p, a, c = E.make_needle_prompt(tokenizer, seed, n_filler, 0.4)
        seed += 1
        ids = tokenizer.encode(p)
        caches, q, seq = H.prefill(model, ids)
        idx = np.linspace(0, seq - 1, min(tok_subsample, seq)).astype(int)
        for L in range(NL):
            K = np.array(caches[L].keys[0, :, :seq, :].astype(mx.float32))
            V = np.array(caches[L].values[0, :, :seq, :].astype(mx.float32))
            for h in range(NKV):
                Kacc[(L, h)].append(K[h][idx]); Vacc[(L, h)].append(V[h][idx])
        used += 1
    basisK, basisV = {}, {}
    for key in Kacc:
        MK = np.concatenate(Kacc[key], 0)
        MV = np.concatenate(Vacc[key], 0)
        basisK[key] = np.linalg.svd(MK, full_matrices=False)[2]  # [d, d] right singular vecs
        basisV[key] = np.linalg.svd(MV, full_matrices=False)[2]
    return basisK, basisV, used


def _project(M, basis, r):
    """Project rows of M [n,d] onto the rank-r subspace spanned by basis[:r] (orthonormal)."""
    Br = basis[:r]                      # [r, d]
    return (M @ Br.T) @ Br             # [n, d]


def _caches_proj(model, caches, seq, r, basisK=None, basisV=None, per_example=False):
    """Build caches with K,V projected to rank r — either onto the UNIVERSAL basis
    (basisK/basisV given) or onto each prompt's OWN basis (per_example=True)."""
    import mlx.core as mx
    out = []
    for L, c in enumerate(caches):
        K = np.array(c.keys[0, :, :seq, :].astype(mx.float32))
        V = np.array(c.values[0, :, :seq, :].astype(mx.float32))
        for h in range(K.shape[0]):
            if per_example:
                bK = np.linalg.svd(K[h], full_matrices=False)[2]
                bV = np.linalg.svd(V[h], full_matrices=False)[2]
            else:
                bK = basisK[(L, h)]; bV = basisV[(L, h)]
            K[h] = _project(K[h], bK, r)
            V[h] = _project(V[h], bV, r)
        out.append(H.GatedCache(mx.array(K[None], dtype=mx.float16),
                                mx.array(V[None], dtype=mx.float16), seq))
    return out


def _answer(model, layer_caches, query, ans0):
    import mlx.core as mx
    z = model(mx.array([[query]]), cache=layer_caches)
    mx.eval(z)
    return int(np.argmax(np.array(z[0, -1].astype(mx.float32)))) == ans0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--n-filler", type=int, default=210)
    ap.add_argument("--rank-grid", type=int, nargs="+", default=[64, 48, 32, 24, 16, 12, 8, 4])
    ap.add_argument("--out", default=str(_RES / "exp016_universal"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    d = H.model_config(model)["head_dim"]
    grid = [r for r in args.rank_grid if r <= d]
    print(f"fitting universal basis on {args.n_calib} calibration prompts...", flush=True)
    basisK, basisV, ncal = collect_calib(model, tok, args.n_calib, args.n_filler)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "raw_results.jsonl", "w")
    recs = []; seed = 0; acc = 0; t0 = time.perf_counter()
    while acc < args.n and seed < args.n * 5:
        depth = 0.2 + 0.5 * ((seed % 7) / 6.0)
        p, a, c = E.make_needle_prompt(tok, seed, args.n_filler, depth)
        seed += 1
        ids = tok.encode(p)
        answer_ids = tok.encode(p + a)[len(ids):] or tok.encode(a)
        ans0 = int(answer_ids[0])
        try:
            caches, q, seq = H.prefill(model, ids)
            full_ok = _answer(model, [H.GatedCache(cc.keys[:, :, :seq, :], cc.values[:, :, :seq, :], seq) for cc in caches], q, ans0)
            if not full_ok:
                continue
            uni = {r: int(_answer(model, _caches_proj(model, caches, seq, r, basisK, basisV), q, ans0)) for r in grid}
            pex = {r: int(_answer(model, _caches_proj(model, caches, seq, r, per_example=True), q, ans0)) for r in grid}
        except Exception as e:
            raw.write(json.dumps({"event": "error", "seed": seed, "err": repr(e)[:200]}) + "\n")
            continue
        acc += 1
        rec = {"seed": seed, "universal_acc": uni, "perexample_acc": pex}
        recs.append(rec); raw.write(json.dumps(rec) + "\n")
        if acc % 5 == 0:
            print(f"accepted {acc}/{args.n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    uni_curve = {r: float(np.mean([rc["universal_acc"][r] for rc in recs])) for r in grid}
    pex_curve = {r: float(np.mean([rc["perexample_acc"][r] for rc in recs])) for r in grid}
    def rstar(curve):
        ok = [r for r in sorted(grid) if curve[r] >= 0.9]
        return min(ok) if ok else d
    uni_rstar, pex_rstar = rstar(uni_curve), rstar(pex_curve)
    agg = {"n": len(recs), "n_calib": ncal, "d": d,
           "universal_acc_by_rank": {str(r): uni_curve[r] for r in grid},
           "perexample_acc_by_rank": {str(r): pex_curve[r] for r in grid},
           "universal_rstar_0.9": uni_rstar, "perexample_rstar_0.9": pex_rstar}
    ratio = uni_rstar / max(pex_rstar, 1)
    if uni_rstar <= 1.5 * pex_rstar and uni_rstar <= d / 2:
        verdict = "UNIVERSAL latent subspace EXISTS (static codebook lever)"
        reason = (f"a FIXED cross-prompt basis preserves answers at rank {uni_rstar} — about the same as the "
                  f"per-prompt rank {pex_rstar} (ratio {ratio:.1f}×). The low-rank KV structure is SHARED across "
                  f"prompts: a fit-once universal codebook of rank ~{uni_rstar}/{d} is a real static latent lever "
                  f"(beyond per-prompt low-rank). This is a genuine hidden-latent structure.")
    elif uni_rstar >= 0.8 * d or ratio >= 2.5:
        verdict = "PROMPT-SPECIFIC subspace (no universal latent code)"
        reason = (f"the universal basis needs rank {uni_rstar}/{d} to preserve answers vs only {pex_rstar} per-prompt "
                  f"({ratio:.1f}× worse): the low-rank KV subspace is PROMPT-SPECIFIC, not a shared latent state. "
                  f"No universal codebook lever. The latent search closes: the only structure is per-prompt (exp012), "
                  f"already owned by per-prompt low-rank methods.")
    else:
        verdict = "PARTIAL universal subspace"
        reason = (f"universal rank {uni_rstar} vs per-prompt {pex_rstar} ({ratio:.1f}×): a partially-shared subspace; "
                  f"a universal codebook helps somewhat but is looser than per-prompt.")
    agg["verdict"] = verdict; agg["verdict_reason"] = reason
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp016 VERDICT:", verdict, "===")
    print(reason)
    print(f"universal r*(0.9)={uni_rstar}/{d}  per-prompt r*(0.9)={pex_rstar}/{d}")
    print("universal acc by rank:", {r: round(uni_curve[r], 2) for r in grid})
    print("per-prompt acc by rank:", {r: round(pex_curve[r], 2) for r in grid})
    return agg


if __name__ == "__main__":
    main()
