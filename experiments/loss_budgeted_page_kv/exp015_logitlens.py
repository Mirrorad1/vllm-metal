# SPDX-License-Identifier: Apache-2.0
"""exp015 — Logit lens: at what DEPTH does the answer crystallize in the latent
(residual) stream, and is that depth a usable hidden-latent structure?

exp011-014 closed the sparse-UNIT hypotheses (no page/kv-head/q-head carries the
answer — holographic). Unexplored latent axis: DEPTH/TIME. We read the model's
hidden state (residual stream) at the query position AFTER each layer, project it
through the final norm + unembedding (the "logit lens"), and watch the gold-answer
token's rank/logprob climb layer by layer. The layer where it becomes the argmax
is the answer's CRYSTALLIZATION DEPTH — a latent computation structure.

Lacuna implication: if the answer crystallizes EARLY (≪ N layers) and consistently,
the later layers are elaboration → depth-adaptive (per-query) KV pruning of late
layers is a latent lever. If LATE/variable, full depth is load-bearing.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import exp011_tail as E

_RES = Path(__file__).resolve().parent / "results"
_CAP = {"resids": []}


@contextlib.contextmanager
def capture_residuals(model):
    blk_cls = type(model.model.layers[0])
    orig = blk_cls.__call__
    _CAP["resids"] = []

    def patched(self, *a, **kw):
        import mlx.core as mx
        out = orig(self, *a, **kw)
        h = out[0] if isinstance(out, tuple) else out
        _CAP["resids"].append(np.array(h[0, -1].astype(mx.float32)))  # [d_model] last token
        return out

    blk_cls.__call__ = patched
    try:
        yield
    finally:
        blk_cls.__call__ = orig


def _unembed_fn(model):
    import mlx.core as mx
    norm = model.model.norm
    if hasattr(model, "lm_head") and model.lm_head is not None:
        head = model.lm_head
    else:
        head = lambda x: model.model.embed_tokens.as_linear(x)

    def f(resid_np):
        h = mx.array(resid_np[None], dtype=mx.float16)
        z = head(norm(h))
        mx.eval(z)
        return np.array(z[0].astype(mx.float32))
    return f


def run_example(model, tokenizer, prompt, answer, unembed):
    import mlx.core as mx
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    ans0 = int(answer_ids[0])
    caches, query, seq = H.prefill(model, ids)
    fc = [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq) for c in caches]
    with capture_residuals(model):
        z = model(mx.array([[query]]), cache=fc)
        mx.eval(z)
    z_final = np.array(z[0, -1].astype(mx.float32))
    full_ok = int(np.argmax(z_final)) == ans0

    per_layer = []
    for r in _CAP["resids"]:
        zl = unembed(r)
        rank = int((zl > zl[ans0]).sum())  # 0 = gold is argmax
        per_layer.append({"argmax_gold": int(np.argmax(zl)) == ans0, "gold_rank": rank})
    # crystallization depth: first layer where gold is argmax AND stays argmax to the end
    NL = len(per_layer)
    crystal = NL
    for L in range(NL):
        if all(per_layer[j]["argmax_gold"] for j in range(L, NL)):
            crystal = L; break
    return {"full_correct": bool(full_ok), "n_layers": NL,
            "crystal_depth": crystal, "crystal_frac": crystal / NL,
            "gold_argmax_by_layer": [p["argmax_gold"] for p in per_layer],
            "gold_rank_by_layer": [p["gold_rank"] for p in per_layer]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-filler", type=int, default=210)
    ap.add_argument("--out", default=str(_RES / "exp015_logitlens"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    unembed = _unembed_fn(model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "raw_results.jsonl", "w")
    recs = []; seed = 0; acc = 0; t0 = time.perf_counter()
    while acc < args.n and seed < args.n * 5:
        depth = 0.2 + 0.5 * ((seed % 7) / 6.0)
        p, a, c = E.make_needle_prompt(tok, seed, args.n_filler, depth)
        seed += 1
        try:
            r = run_example(model, tok, p, a, unembed)
        except Exception as e:
            raw.write(json.dumps({"event": "error", "seed": seed, "err": repr(e)[:200]}) + "\n")
            continue
        if not r["full_correct"]:
            continue
        acc += 1; r["seed"] = seed; recs.append(r); raw.write(json.dumps(r) + "\n")
        if acc % 10 == 0:
            print(f"accepted {acc}/{args.n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    NL = recs[0]["n_layers"]
    crystals = [r["crystal_depth"] for r in recs]
    # per-layer fraction of examples where gold is already the argmax
    argmax_curve = [float(np.mean([r["gold_argmax_by_layer"][L] for r in recs])) for L in range(NL)]
    rank_curve = [float(np.median([r["gold_rank_by_layer"][L] for r in recs])) for L in range(NL)]
    agg = {"n": len(recs), "n_layers": NL,
           "crystal_depth_median": float(np.median(crystals)),
           "crystal_depth_mean": float(np.mean(crystals)),
           "crystal_frac_median": float(np.median([r["crystal_frac"] for r in recs])),
           "crystal_depth_sd": float(np.std(crystals)),
           "argmax_gold_by_layer": argmax_curve,
           "gold_rank_median_by_layer": rank_curve}
    cd = agg["crystal_depth_median"]; cf = agg["crystal_frac_median"]; sd = agg["crystal_depth_sd"]
    if cf <= 0.6 and sd <= 0.2 * NL:
        verdict = "EARLY + CONSISTENT crystallization (depth-latent lever)"
        reason = (f"the gold answer becomes the residual-stream argmax by layer {cd:.0f}/{NL} "
                  f"({cf*100:.0f}% depth), consistently (sd {sd:.1f} layers). The last ~{NL-cd:.0f} layers are "
                  f"elaboration AFTER the answer is decided — a real DEPTH-latent structure: late-layer KV/compute "
                  f"is prunable per-query. Candidate latent lacuna: depth-adaptive (early-exit-aware) KV retention, "
                  f"keeping full KV only up to the crystallization layer.")
    elif cf > 0.85 or sd > 0.3 * NL:
        verdict = "LATE / VARIABLE crystallization (no depth lever)"
        reason = (f"the answer only crystallizes at layer {cd:.0f}/{NL} ({cf*100:.0f}% depth) and/or varies widely "
                  f"(sd {sd:.1f}): the full network depth is load-bearing / the depth is input-dependent and not a "
                  f"static latent structure. No depth lever beyond what a runtime signal (shown hard) could give.")
    else:
        verdict = "MODERATE crystallization"
        reason = (f"answer crystallizes mid-depth (layer {cd:.0f}/{NL}, {cf*100:.0f}%, sd {sd:.1f}); a partial "
                  f"depth lever may exist but is not dramatic.")
    agg["verdict"] = verdict; agg["verdict_reason"] = reason
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n=== exp015 VERDICT:", verdict, "===")
    print(reason)
    print(f"crystallization depth median {cd:.0f}/{NL} ({cf*100:.0f}%), sd {sd:.1f}")
    print("argmax-gold-by-layer:", [round(x, 2) for x in argmax_curve])
    return agg


if __name__ == "__main__":
    main()
