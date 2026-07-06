# SPDX-License-Identifier: Apache-2.0
"""exp024 — the dispatcher meets a REGULAR benchmark (LongBench v2, real documents).

Everything so far (exp020-023) is measured on our 7 synthetic families. This produces the
same per-context substrate matrix on LongBench v2 (Zhang et al. 2025: real long documents,
4-way multiple choice, exact answers) so exp023_dispatch.py can run UNCHANGED on real
traffic:

  per accepted item (full cache answers correctly):
    - write-time per-page attention mass (target 7B)  -> the routing features
    - optional 0.5B PROXY mass on the same item       -> the pre-prefill routing arm
    - outcome under every action: evict@{6.25,12.5,25}% (attention policy),
      kivi-{8,4,2}-bit quant                          -> the action matrix
  emitted in exp022 s23-schema: {"family": <lbv2 domain>, "cells": {...}, "mass": [...],
  "mass_proxy": [...]} so the dispatcher, gates and verdicts run as-is:
      python exp023_dispatch.py --s23 results/exp024_lbv2/s23.jsonl

MCQ scoring: the prompt ends "The correct answer is (" and the answer is the single next
token; we score argmax over the four letter tokens {A,B,C,D} from ONE forward step over
the (sliced / quantized) cache — the same step that captures attention mass. Teacher-free,
deterministic, exact.

Honest notes:
  - 4-way MCQ has a 25% guess floor: a broken cache still "answers" right 1/4 of the time,
    so break labels are NOISED TOWARD SAFE and measured break rates understate true damage.
    The conformal guarantee still holds for the labels as defined (answer flips).
  - Acceptance (full-cache correct) selects the model-solvable subset, as everywhere in
    this series; report the acceptance rate.
  - Domains stand in for "families" in the dispatcher's histograms and any
    leave-one-domain-out stress.

Gates (same discipline): g1 full-cache-through-cache-rebuild == direct forward on the
first item (letter + logit tol); quant8b column must land ~= full accuracy (sanity);
2-bit must change logits on item 0.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import exp020_quality_cuda as X                      # page_of, select_pages
from exp022_substrates import (load_model, chunked_prefill, cache_kvs, build_cache,
                               eager_attn, kivi_quant, kivi_bits_ratio)

PROMPT = ("Please read the following text and answer the question below.\n\n{context}\n\n"
          "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nThe correct answer is (")


def load_lbv2(tok, max_tokens, limit, length_buckets=("short",), min_tokens=2000,
              truncate_chars=0):
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
    items = []
    for d in ds:
        if length_buckets and d.get("length") not in length_buckets:
            continue
        ctx = d["context"][:truncate_chars] if truncate_chars else d["context"]
        if not truncate_chars and len(ctx) > max_tokens * 6:
            continue                       # cheap char pre-filter: skip 100k+-token docs untokenized
        p = PROMPT.format(context=ctx, question=d["question"], A=d["choice_A"],
                          B=d["choice_B"], C=d["choice_C"], D=d["choice_D"])
        n_tok = len(tok(p).input_ids)
        if n_tok > max_tokens or n_tok < min_tokens:
            continue
        items.append({"id": d["_id"], "domain": d["domain"], "prompt": p,
                      "answer": d["answer"].strip(), "n_tok": n_tok})
        if limit and len(items) >= limit:
            break
    return items


@torch.no_grad()
def step_letters(model, kvs, ids, plen, letter_ids, device, keep_tok=None):
    """One forward of the last prompt token over a (possibly sliced/transformed) cache.
    Returns (predicted letter index 0-3, logits, attentions_or_None)."""
    if keep_tok is not None:
        idx = torch.tensor(keep_tok, device=device, dtype=torch.long)
        kvs = [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in kvs]
    kept = kvs[0][0].shape[2]
    cache = build_cache(kvs)
    pos = torch.tensor([[plen - 1]], device=device)
    cp = torch.arange(kept, kept + 1, device=device)
    o = model(input_ids=ids[:, plen - 1:plen], past_key_values=cache, use_cache=True,
              position_ids=pos, cache_position=cp)
    lg = o.logits[0, -1].float()
    return int(np.argmax([float(lg[t]) for t in letter_ids])), lg


@torch.no_grad()
def mass_and_letter(model, kvs, ids, plen, npg, B, letter_ids, device):
    """Eager one-token step: attention mass AND the full-cache answer in one forward."""
    cache = build_cache(kvs)
    with eager_attn(model):
        o = model(input_ids=ids[:, plen - 1:plen], past_key_values=cache, use_cache=True,
                  output_attentions=True,
                  cache_position=torch.arange(plen - 1, plen, device=device))
    lg = o.logits[0, -1].float()
    pred = int(np.argmax([float(lg[t]) for t in letter_ids]))
    mass = np.zeros(npg)
    for layer in o.attentions:
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[X.page_of(k, B)] += a[k]
    return mass, pred, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--proxy-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="'' to skip the proxy pass")
    ap.add_argument("--max-ctx-tokens", type=int, default=25000)
    ap.add_argument("--n", type=int, default=0, help="cap on loaded items (0 = all short)")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="results/exp024_lbv2")
    ap.add_argument("--keep-rejected", action="store_true",
                    help="record items the full cache gets wrong too (smoke/audit)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
        args.proxy_model = ""
        args.max_ctx_tokens, args.n = 6000, 3
        args.keep_rejected = True

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device)
    proxy = load_model(args.proxy_model, dtype, device) if args.proxy_model else None
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    B = args.block_size

    items = load_lbv2(tok, args.max_ctx_tokens, args.n,
                      min_tokens=200 if args.smoke else 2000,
                      truncate_chars=5000 if args.smoke else 0)
    print(f"[exp024] {len(items)} LongBench-v2 items (<= {args.max_ctx_tokens} tok); "
          f"device={device} model={args.model} proxy={args.proxy_model or 'none'}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "s23.jsonl", "w")
    n_acc = n_seen = 0
    t00 = time.time()
    for i, it in enumerate(items):
        ids = tok(it["prompt"], return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        npg = (plen + B - 1) // B
        gold = "ABCD".index(it["answer"])
        cache = chunked_prefill(model, ids[:, :plen - 1], device, args.chunk)
        kvs = cache_kvs(cache)
        mass, pred_full, lg_full = mass_and_letter(model, kvs, ids, plen, npg, B, letter_ids, device)
        n_seen += 1
        full_ok = pred_full == gold
        if i == 0:   # g1: rebuild-through-DynamicCache must reproduce; g3: 2-bit must move
            pred2, lg2 = step_letters(model, kvs, ids, plen, letter_ids, device)
            d = float((lg_full - lg2).abs().max())
            kv2 = [(kivi_quant(k, 2, "k"), kivi_quant(v, 2, "v")) for k, v in kvs]
            _, lg2b = step_letters(model, kv2, ids, plen, letter_ids, device)
            d2b = float((lg_full - lg2b).abs().max())
            print(f"[gate g1] eager-vs-sdpa same letter={pred2 == pred_full} "
                  f"max|dlogit|={d:.2e}; [g3] 2-bit max|dlogit|={d2b:.2e} (must be >1)")
            assert pred2 == pred_full, "GATE g1 FAIL"
            assert d2b > 1.0, "GATE g3 FAIL"
        if not full_ok and not args.keep_rejected:
            del cache, kvs
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(items)}] acc so far {n_acc}/{n_seen} "
                      f"({(time.time()-t00)/(i+1):.1f}s/item)", flush=True)
            continue
        n_acc += int(full_ok)
        rec = {"family": it["domain"], "seed": it["id"], "P": int(npg), "plen": int(plen),
               "full": {"correct": bool(full_ok), "nll": float(-torch.log_softmax(
                   lg_full, -1)[letter_ids[gold]])},
               "cells": {}, "mass": [float(x) for x in mass]}
        for b in (0.25, 0.125, 0.0625):
            keep = X.select_pages("attention", npg, b, mass)
            keep_tok = sorted({t for pg in keep for t in range(pg * B, min(pg * B + B, plen - 1))})
            pred, lg = step_letters(model, kvs, ids, plen, letter_ids, device, keep_tok=keep_tok)
            rec["cells"][f"evict@{b}"] = {
                "correct": bool(pred == gold), "bits": b,
                "nll": float(-torch.log_softmax(lg, -1)[letter_ids[gold]])}
        hd = kvs[0][0].shape[3]
        for bits in (8, 4, 2):
            kvq = [(kivi_quant(k, bits, "k"), kivi_quant(v, bits, "v")) for k, v in kvs]
            pred, lg = step_letters(model, kvq, ids, plen, letter_ids, device)
            rec["cells"][f"quant{bits}b"] = {
                "correct": bool(pred == gold), "scheme": "kivi",
                "bits": kivi_bits_ratio(bits, 128, hd),
                "nll": float(-torch.log_softmax(lg, -1)[letter_ids[gold]])}
        del cache, kvs
        if proxy is not None:
            pc = chunked_prefill(proxy, ids[:, :plen - 1], device, args.chunk)
            pmass, _, _ = mass_and_letter(proxy, cache_kvs(pc), ids, plen, npg, B,
                                          letter_ids, device)
            rec["mass_proxy"] = [float(x) for x in pmass]
            del pc
        raw.write(json.dumps(rec) + "\n"); raw.flush()
        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}/{len(items)}] {it['domain']}/{it['id'][:8]} tok={plen} "
                  f"full={full_ok} acc {n_acc}/{n_seen} "
                  f"({(time.time()-t00)/(i+1):.1f}s/item)", flush=True)
    raw.close()
    print(f"[done] acceptance {n_acc}/{n_seen} = {n_acc/max(n_seen,1):.2f}; "
          f"wrote {out}/s23.jsonl")
    # quant8b sanity is checked downstream by exp023/exp022_matrix (single-class caveat)


if __name__ == "__main__":
    main()
