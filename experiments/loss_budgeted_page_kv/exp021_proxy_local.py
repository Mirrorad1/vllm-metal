# SPDX-License-Identifier: Apache-2.0
"""exp021 L6 Cut 1 — PROXY feature generator (0.5B, local Apple-Silicon MPS, transformers 4.51).

Question (SPEC: lacuna L6, cross-scale wall transfer): does a TINY model's prefill read the
same write-time signal as the 7B target — i.e., can a 0.5B proxy classify a context's task
type / survival-at-budget-B BEFORE the target model ever prefills it?

This script produces the proxy side of the join: it reads the 7B run's jsonl for the exact
(family, seed) instance list, regenerates each context BYTE-IDENTICALLY (make_task is
deterministic; Qwen2.5 0.5B/7B share one tokenizer, so pages align exactly — asserted), and
computes the SAME query-agnostic per-page attention mass exp020 dumps (mean over heads,
summed over layers, of the last prompt token's attention), plus the proxy's own full-context
answer correctness (the competence covariate). Offline analysis then joins this against the
7B labels already on disk (exp021_transfer.py).

Engineering notes (4.51 / MPS):
  * prefill prompt[:-1] CHUNKED under sdpa (an un-chunked 20k sdpa prefill on MPS would
    transiently materialize ~11 GB of scores per layer in the math fallback);
  * the one-token mass step runs with config._attn_implementation flipped to "eager"
    (q_len=1 => attentions are [1, H, 1, plen], O(L) memory) then restored — 4.51 has no
    set_attn_implementation(); the unified attention reads the config attr at forward time;
  * fp16 on MPS, fp32 on CPU; mass accumulated in fp32.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp020_quality_cuda as X  # make_task, page_of (pure, version-independent)


def read_instances(runs_path):
    inst = []
    with open(runs_path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "instance":
                inst.append({k: d[k] for k in ("family", "seed", "P", "L", "block_size")})
    return inst


@torch.no_grad()
def chunked_prefill(model, ids, device, chunk=2048):
    """KV cache over ids (sdpa, chunked). Returns the cache; absolute positions via
    explicit cache_position per chunk."""
    cache = None
    n = ids.shape[1]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True,
                    cache_position=cp)
        cache = out.past_key_values
    return cache


@torch.no_grad()
def mass_step(model, cache, last_tok, plen, n_pages, B, device):
    """One eager token step over the cached prompt: exp020's attention_mass definition
    (mean over heads, summed over layers, bucketed into pages). Returns (mass, logits)."""
    prev = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        o = model(input_ids=last_tok, past_key_values=cache, use_cache=True,
                  output_attentions=True,
                  cache_position=torch.arange(plen - 1, plen, device=device))
    finally:
        model.config._attn_implementation = prev
    assert o.attentions is not None and o.attentions[0] is not None, "no attentions returned"
    mass = np.zeros(n_pages)
    for layer in o.attentions:                        # [1, heads, 1, kv_len=plen]
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[X.page_of(k, B)] += a[k]
    return mass, o.logits[0, -1], o.past_key_values


@torch.no_grad()
def proxy_correct(model, cache, first_logits, ans_ids, plen, device):
    """Teacher-forced exact-answer check for the proxy itself (competence covariate)."""
    if int(first_logits.argmax()) != int(ans_ids[0]):
        return False
    if len(ans_ids) == 1:
        return True
    q = torch.tensor([ans_ids[:-1]], device=device, dtype=torch.long)
    cp = torch.arange(plen, plen + q.shape[1], device=device)
    lg = model(input_ids=q, past_key_values=cache, use_cache=True, cache_position=cp).logits[0]
    return all(int(lg[t].argmax()) == int(a) for t, a in enumerate(ans_ids[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/exp021_7b/raw_Qwen2.5-7B-Instruct.jsonl",
                    help="the TARGET (7B) jsonl whose instances to mirror")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-filler", type=int, default=1500, help="must match the target run")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N instances")
    ap.add_argument("--out", default="results/exp021_proxy_0p5b")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()

    inst = read_instances(args.runs)
    if args.limit:
        inst = inst[:args.limit]
    print(f"[proxy] {len(inst)} instances from {args.runs}; device={device} dtype={dtype}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"proxy_{args.model.split('/')[-1]}.jsonl"
    raw = open(path, "w")
    mismatch = 0
    t00 = time.time()
    for i, it in enumerate(inst):
        fam, seed, B = it["family"], it["seed"], it["block_size"]
        p, a = X.make_task(fam, seed, args.n_filler)
        ids = tok(p + a, return_tensors="pt").input_ids.to(device)
        plen = tok(p, return_tensors="pt").input_ids.shape[1]
        ans = ids[0, plen:].tolist()
        npg = (plen + B - 1) // B
        # tokenizer-alignment gate: pages must line up EXACTLY with the 7B run
        if npg != it["P"] or ids.shape[1] != it["L"]:
            mismatch += 1
            print(f"  [MISMATCH] {fam}/{seed}: local P={npg} L={ids.shape[1]} "
                  f"vs target P={it['P']} L={it['L']} — skipped")
            continue
        t0 = time.time()
        cache = chunked_prefill(model, ids[:, :plen - 1], device, args.chunk)
        mass, lg0, cache = mass_step(model, cache, ids[:, plen - 1:plen], plen, npg, B, device)
        ok = proxy_correct(model, cache, lg0, ans, plen, device)
        del cache
        raw.write(json.dumps({"event": "proxy_instance", "family": fam, "seed": int(seed),
                              "P": int(npg), "L": int(ids.shape[1]), "block_size": int(B),
                              "proxy_correct": bool(ok),
                              "mass": [float(x) for x in mass]}) + "\n")
        raw.flush()
        if i < 3 or (i + 1) % 10 == 0:
            el = time.time() - t00
            print(f"  [{i+1}/{len(inst)}] {fam}/{seed} plen={plen} mass_sum={mass.sum():.1f} "
                  f"proxy_correct={ok} ({time.time()-t0:.1f}s; avg {el/(i+1):.1f}s/inst)", flush=True)
    raw.close()
    print(f"[done] wrote {path}  (mismatches skipped: {mismatch})")


if __name__ == "__main__":
    main()
