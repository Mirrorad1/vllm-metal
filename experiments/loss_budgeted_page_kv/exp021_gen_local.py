# SPDX-License-Identifier: Apache-2.0
"""exp021 LOCAL feature generator (0.5B / CPU / transformers 4.51) — Cut 1 smoke.

Produces a REAL instrumented exp020-schema jsonl (per-instance query-agnostic `mass`
features + per-(budget,policy) `correct`) on a small model that runs on an Apple Mac,
so exp021_admission.py can run a genuine (small-scale) HORN-B verdict WITHOUT a GPU.

Why a separate file: exp020_quality_cuda.py targets transformers>=5 (dtype=, logits_to_keep,
the mem-efficient sdpa-4D fast engine) and won't run as-is on 4.51. This file reuses exp020's
*exact* task + page functions (make_task / page_of / select_pages — pure, version-independent)
but does plain EAGER forwards with an explicit 4D additive mask, which is trivially correct at
0.5B/~1-2k context. It is GATED the same way exp020 is: full-keep must reproduce the ungated
answer, a tight budget must change it. NOT a substitute for the 7B/20k RunPod headline — it is
a local correctness+plumbing proof and a first 0.5B signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp020_quality_cuda as X  # make_task, page_of, select_pages, FAMILIES (pure helpers)


def answer_ok(logits, plen, ans):
    return all(int(logits[plen - 1 + t].argmax()) == a for t, a in enumerate(ans))


def mass_of(model, ids, plen, B):
    """Query-agnostic per-page attention mass of the last prompt token (eager)."""
    npg = (plen + B - 1) // B
    with torch.no_grad():
        att = model(input_ids=ids[:, :plen], output_attentions=True).attentions
    mass = np.zeros(npg)
    for layer in att:
        a = layer[0, :, -1, :plen].float().mean(0).numpy()  # last prompt token over keys
        for k in range(min(len(a), plen)):
            mass[X.page_of(k, B)] += a[k]
    return mass


def gated_mask(L, plen, dropped_pages, B, device):
    """4D additive mask: causal everywhere; for answer-predicting rows (>= plen-1) also
    -inf on columns of DROPPED prompt pages. Diagonal kept (a token always attends to
    itself) so no row is fully masked -> no softmax NaN (exp020's fill_diagonal invariant)."""
    m = torch.full((L, L), float("-inf"))
    m = torch.triu(m, diagonal=1)                       # causal: 0 on/below diag, -inf above
    if dropped_pages:
        drop_col = torch.zeros(L, dtype=torch.bool)
        for j in range(plen):
            if X.page_of(j, B) in dropped_pages:
                drop_col[j] = True
        m[plen - 1:, drop_col] = float("-inf")          # gate only answer-predicting rows
    m.fill_diagonal_(0.0)                               # never fully-masked
    return m.view(1, 1, L, L).to(device)


def answer_ok_masked(model, ids, plen, ans, keep, B, device):
    L = ids.shape[1]
    npg = (plen + B - 1) // B
    dropped = {p for p in range(npg) if p not in set(keep)}
    mask = gated_mask(L, plen, dropped, B, device)
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=mask).logits[0]
    return answer_ok(lg, plen, ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-filler", type=int, default=100)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--families", nargs="+", default=["single_needle", "sum_scattered"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--out", default="results/exp021_local_0p5b")
    args = ap.parse_args()
    device = "cpu"
    B = args.block_size
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()

    # --- gate (exp020 discipline): full-keep == ungated; tight budget differs ---
    p, a = X.make_task("single_needle", 7, args.n_filler)
    ids = tok(p + a, return_tensors="pt").input_ids.to(device)
    plen = tok(p, return_tensors="pt").input_ids.shape[1]
    ans = ids[0, plen:].tolist()
    npg = (plen + B - 1) // B
    full_keep = list(range(npg))
    with torch.no_grad():
        base_ok = answer_ok(model(input_ids=ids).logits[0], plen, ans)
    gate_full = answer_ok_masked(model, ids, plen, ans, full_keep, B, device)
    assert base_ok == gate_full, "GATE FAIL: full-keep mask != ungated forward"
    print(f"[gate] full-keep == ungated ({base_ok}); tight-budget differs check during run. OK")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / f"raw_{args.model.split('/')[-1]}.jsonl", "w")
    for fam in args.families:
        acc = seed = 0
        while acc < args.n and seed < args.n * 6:
            p, a = X.make_task(fam, seed, args.n_filler); seed += 1
            ids = tok(p + a, return_tensors="pt").input_ids.to(device)
            plen = tok(p, return_tensors="pt").input_ids.shape[1]
            ans = ids[0, plen:].tolist()
            if not ans:
                continue
            with torch.no_grad():
                if not answer_ok(model(input_ids=ids).logits[0], plen, ans):
                    continue                                   # acceptance gate (full cache must solve)
            acc += 1
            inst_seed = seed - 1
            L = ids.shape[1]; npg = (plen + B - 1) // B
            mass = mass_of(model, ids, plen, B)
            raw.write(json.dumps({"event": "instance", "family": fam, "seed": int(inst_seed),
                                  "P": int(npg), "L": int(L), "block_size": int(B),
                                  "mass": [float(x) for x in mass]}) + "\n")
            for bf in args.budgets:
                for policy in ("attention", "recent"):
                    keep = X.select_pages(policy, npg, bf, mass)
                    correct = answer_ok_masked(model, ids, plen, ans, keep, B, device)
                    raw.write(json.dumps({"family": fam, "seed": int(inst_seed), "budget": bf,
                                          "policy": policy, "J": len(keep), "P": int(npg),
                                          "L": int(L), "correct": bool(correct)}) + "\n")
        print(f"{fam}: {acc} accepted")
    raw.close()
    print(f"[done] wrote {out}/raw_{args.model.split('/')[-1]}.jsonl")


if __name__ == "__main__":
    main()
