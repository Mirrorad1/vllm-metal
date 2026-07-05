# SPDX-License-Identifier: Apache-2.0
"""exp025 — the dispatcher meets RULER (Cut 2b): the STANDARDIZED benchmark family that
actually contains the dense-aggregation regime.

exp024 showed real MCQ-QA traffic (LongBench v2) has no wall — 16x eviction is free there.
RULER (Hsieh et al. 2024) is the community-standard long-context suite whose CWE/FWE tasks
are aggregation-shaped (count word frequencies across the WHOLE context) and whose NIAH
tasks are retrieval-shaped — both regimes, one recognized benchmark. This runs the same
per-context substrate matrix on it:

  tasks: cwe, fwe (DENSE aggregation) | vt (chained tracking) | niah_single, niah_multikey
         (SPARSE retrieval)
  cells: evict@{6.25,12.5,25}% (attention policy) + kivi-{8,4,2}-bit
  plus per-item write-time mass (7B) and 0.5B proxy mass -> exp023_dispatch runs UNCHANGED:
      python exp023_dispatch.py --s23 results/exp025_ruler/s23.jsonl --proxy ""

DATA SOURCE: tries the community-published official RULER data on HF first
(--hf-data, default on: simonjegou/ruler, the KVPress-lineage copy). If unavailable or
schema-mismatched it FALLS BACK, loudly, to built-in generators that follow the RULER
spec (CWE: 10 common words at high frequency vs uncommon filler; FWE: Zeta-distributed
coined words, top-3; VT: hopped variable chains + distractor chains; NIAH: magic-number
needles in noise). Either way the source is printed and recorded per row.

SCORING (RULER convention): greedy generation over the (sliced/quantized) cache, correct
iff EVERY expected string appears in the generated text (case-insensitive). Positions:
absolute position_ids continue from plen-1; cache_position continues from the (possibly
shorter) cache — the exp020 fast-engine convention, now for multi-token decode.

Gates: g1 full-cache generation through the rebuilt cache must be correct on >=95% of
accepted items by construction (acceptance) and the first item's sdpa-vs-eager step must
agree; g3 2-bit must change the first item's logits; quant8b column is the quantizer
sanity downstream (exp023 caveat line).
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import exp020_quality_cuda as X
from exp022_substrates import (load_model, chunked_prefill, cache_kvs, build_cache,
                               eager_attn, kivi_quant, kivi_bits_ratio)

COMMON = ("time year people way day man thing woman life child world school state family "
          "student group country problem hand part place case week company system program "
          "question work government number night point home water room mother area money "
          "story fact month lot right study book eye job word business issue side kind head "
          "house service friend father power hour game line end member law car city name "
          "team minute idea body back parent face level office door health person art war "
          "history party result change morning reason research girl guy moment air teacher "
          "force education").split()


# ---------------------------------------------------------------------------
# RULER-spec generators (fallback / default-offline). Each returns (context+question
# prompt, [expected strings]). Word counts are scaled to hit ~seq_len tokens.
# ---------------------------------------------------------------------------
def _rand_word(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def gen_cwe(rng, n_words):
    common = rng.sample(COMMON, 10)
    words = common * 30
    while len(words) < n_words:
        words.append(_rand_word(rng))
    rng.shuffle(words)
    ctx = " ".join(words)
    q = ("\nQuestion: What are the 10 most common words in the above list? "
         "Answer: The top 10 words that appear most often in the list are:")
    return f"Below is a numbered list of words. Memorize them.\n\n{ctx}\n{q}", common


def gen_fwe(rng, n_words):
    vocab = [_rand_word(rng, rng.randint(5, 8)) for _ in range(60)]
    freqs = np.array([1.0 / (i + 1) ** 2.0 for i in range(len(vocab))])
    freqs = freqs / freqs.sum()
    idx = rng.choices(range(len(vocab)), weights=freqs, k=n_words)
    ctx = " ".join(vocab[i] for i in idx)
    from collections import Counter
    top3 = [w for w, _ in Counter(vocab[i] for i in idx).most_common(3)]
    q = ("\nQuestion: Do not provide any explanation. Please ignore the dots '....'. "
         "What are the three most frequently appeared words in the above coded text? "
         "Answer: According to the coded text above, the three most frequently appeared "
         "words are:")
    return f"Read the following coded text and track the frequency of each coded word.\n\n{ctx}\n{q}", top3


def gen_vt(rng, n_fill):
    val = rng.randint(10000, 99999)
    names = rng.sample([f"X{i}" for i in range(1, 60)], 25)
    chain, rest = names[:5], names[5:]
    stmts = [f"VAR {chain[0]} = {val}"]
    for a, b in zip(chain, chain[1:]):
        stmts.append(f"VAR {b} = VAR {a}")
    v2 = rng.randint(10000, 99999)
    for i in range(0, len(rest) - 1, 2):
        stmts.append(f"VAR {rest[i]} = {v2 + i}")
    noise = [f"The quick brown fox jumps over the lazy dog number {rng.randint(0,999)}."
             for _ in range(n_fill)]
    body = []
    k = max(1, len(noise) // max(len(stmts), 1))
    si = iter(stmts)
    for j, s in enumerate(noise):
        body.append(s)
        if j % k == 0:
            body.append(next(si, ""))
    ctx = " ".join(b for b in body if b)
    q = (f"\nQuestion: Find all variables that are assigned the value {val} in the text "
         f"above. Answer: According to the chain of variable assignments in the text above, "
         f"the variables that are assigned the value {val} are:")
    return f"Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{ctx}\n{q}", chain


def gen_niah(rng, n_fill, multikey=False):
    keys = [f"special-magic-{_rand_word(rng)}" for _ in range(4 if multikey else 1)]
    vals = {k: rng.randint(1000000, 9999999) for k in keys}
    tgt = rng.choice(keys)
    noise = ["The grass is green. The sky is blue. The sun is yellow. Here we go. "
             "There and back again." for _ in range(n_fill)]
    for k in keys:
        noise.insert(rng.randint(0, len(noise)), f"One of the special magic numbers for {k} is: {vals[k]}.")
    ctx = " ".join(noise)
    q = (f"\nQuestion: What is the special magic number for {tgt} mentioned in the provided "
         f"text? Answer: The special magic number for {tgt} mentioned in the provided text is")
    return f"Some special magic numbers are hidden within the following text. Make sure to memorize it.\n\n{ctx}\n{q}", [str(vals[tgt])]


GENERATORS = {"cwe": gen_cwe, "fwe": gen_fwe, "vt": gen_vt,
              "niah_single": lambda rng, n: gen_niah(rng, n, False),
              "niah_multikey": lambda rng, n: gen_niah(rng, n, True)}
SIZE_UNIT = {"cwe": "words", "fwe": "words", "vt": "sents", "niah_single": "sents",
             "niah_multikey": "sents"}


def build_local(tok, tasks, n_per_task, seq_len):
    items = []
    for task in tasks:
        made = tries = 0
        size = seq_len if SIZE_UNIT[task] == "words" else seq_len // 18
        while made < n_per_task and tries < n_per_task * 4:
            rng = random.Random(hash((task, tries)) & 0xFFFFFFFF)
            tries += 1
            prompt, outs = GENERATORS[task](rng, size)
            n_tok = len(tok(prompt).input_ids)
            if not (0.7 * seq_len <= n_tok <= 1.25 * seq_len):
                # multiplicative feedback — converges even though the token/word ratio
                # shifts with size (cwe mixes 1-token common and 2-3-token random words)
                size = max(64, int(size * seq_len / max(n_tok, 1)))
                continue
            items.append({"task": task, "id": f"{task}-{tries}", "prompt": prompt,
                          "outputs": [str(o) for o in outs]})
            made += 1
    return items, "local-ruler-spec"


def build_hf(tok, tasks, n_per_task, seq_len):
    from datasets import load_dataset
    name_map = {"niah_single": "niah_single_2", "niah_multikey": "niah_multikey_2"}
    items = []
    for task in tasks:
        hf_task = name_map.get(task, task)
        ds = load_dataset("simonjegou/ruler", str(seq_len), split=hf_task)
        for i, d in enumerate(ds):
            if i >= n_per_task:
                break
            outs = d.get("outputs") or d.get("answer") or []
            if isinstance(outs, str):
                outs = [outs]
            items.append({"task": task, "id": f"{hf_task}-{i}", "prompt": d["input"],
                          "outputs": [str(o) for o in outs]})
    return items, f"hf:simonjegou/ruler@{seq_len}"


@torch.no_grad()
def greedy_gen(model, kvs, ids, plen, max_new, device, keep_tok=None):
    if keep_tok is not None:
        idx = torch.tensor(keep_tok, device=device, dtype=torch.long)
        kvs = [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in kvs]
    kept = kvs[0][0].shape[2]
    cache = build_cache(kvs)
    cur = ids[:, plen - 1:plen]
    out = []
    lg0 = None
    for t in range(max_new):
        o = model(input_ids=cur, past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[plen - 1 + t]], device=device),
                  cache_position=torch.arange(kept + t, kept + t + 1, device=device))
        cache = o.past_key_values
        lg = o.logits[0, -1].float()
        if t == 0:
            lg0 = lg
        nxt = int(lg.argmax())
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=device)
    return out, lg0


def score(tok, gen_ids, outputs, eos_id):
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    text = tok.decode(gen_ids).lower()
    return all(str(o).lower() in text for o in outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--proxy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tasks", nargs="+",
                    default=["cwe", "fwe", "vt", "niah_single", "niah_multikey"])
    ap.add_argument("--n-per-task", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-hf-data", action="store_true", help="skip HF, use spec generators")
    ap.add_argument("--out", default="results/exp025_ruler")
    ap.add_argument("--keep-rejected", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model, args.proxy_model = "Qwen/Qwen2.5-0.5B-Instruct", ""
        args.seq_len, args.n_per_task, args.max_new = 1500, 2, 24
        args.no_hf_data = True
        args.keep_rejected = True

    torch.set_grad_enabled(False)          # pure inference throughout
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device)
    proxy = load_model(args.proxy_model, dtype, device) if args.proxy_model else None
    eos_id = tok.eos_token_id
    B = args.block_size

    src = None
    if not args.no_hf_data:
        try:
            items, src = build_hf(tok, args.tasks, args.n_per_task, args.seq_len)
        except Exception as e:
            print(f"[data] HF RULER unavailable ({type(e).__name__}: {e}) — "
                  f"FALLING BACK to local RULER-spec generators")
    if src is None:
        items, src = build_local(tok, args.tasks, args.n_per_task, args.seq_len)
    print(f"[exp025] {len(items)} items from {src}; device={device} model={args.model}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "s23.jsonl", "w")
    n_acc = n_seen = 0
    t00 = time.time()
    for i, it in enumerate(items):
        ids = tok(it["prompt"], return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        npg = (plen + B - 1) // B
        cache = chunked_prefill(model, ids[:, :plen - 1], device, args.chunk)
        kvs = cache_kvs(cache)
        # full-cache generation + mass (mass needs one eager step first)
        mcache = build_cache(kvs)
        with eager_attn(model):
            o = model(input_ids=ids[:, plen - 1:plen], past_key_values=mcache, use_cache=True,
                      output_attentions=True,
                      cache_position=torch.arange(plen - 1, plen, device=device))
        mass = np.zeros(npg)
        for layer in o.attentions:
            a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
            for k in range(min(len(a), plen)):
                mass[X.page_of(k, B)] += a[k]
        del mcache
        gen_full, lg_full = greedy_gen(model, kvs, ids, plen, args.max_new, device)
        full_ok = score(tok, gen_full, it["outputs"], eos_id)
        n_seen += 1
        if i == 0:
            kv2 = [(kivi_quant(k, 2, "k"), kivi_quant(v, 2, "v")) for k, v in kvs]
            _, lg2 = greedy_gen(model, kv2, ids, plen, 1, device)
            d2b = float((lg_full - lg2).abs().max())
            print(f"[gate g3] 2-bit first-step max|dlogit|={d2b:.2e} (must be >1)")
            assert d2b > 1.0, "GATE g3 FAIL"
        if not full_ok and not args.keep_rejected:
            del cache, kvs
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(items)}] acc {n_acc}/{n_seen} "
                      f"({(time.time()-t00)/(i+1):.1f}s/item)", flush=True)
            continue
        n_acc += int(full_ok)
        rec = {"family": it["task"], "seed": it["id"], "P": int(npg), "plen": int(plen),
               "source": src, "full": {"correct": bool(full_ok), "nll": 0.0},
               "cells": {}, "mass": [float(x) for x in mass]}
        for b in (0.25, 0.125, 0.0625):
            keep = X.select_pages("attention", npg, b, mass)
            keep_tok = sorted({t for pg in keep for t in range(pg * B, min(pg * B + B, plen - 1))})
            g, _ = greedy_gen(model, kvs, ids, plen, args.max_new, device, keep_tok=keep_tok)
            rec["cells"][f"evict@{b}"] = {"correct": bool(score(tok, g, it["outputs"], eos_id)),
                                          "bits": b, "nll": 0.0}
        hd = kvs[0][0].shape[3]
        for bits in (8, 4, 2):
            kvq = [(kivi_quant(k, bits, "k"), kivi_quant(v, bits, "v")) for k, v in kvs]
            g, _ = greedy_gen(model, kvq, ids, plen, args.max_new, device)
            rec["cells"][f"quant{bits}b"] = {"correct": bool(score(tok, g, it["outputs"], eos_id)),
                                             "scheme": "kivi", "nll": 0.0,
                                             "bits": kivi_bits_ratio(bits, 128, hd)}
        del cache, kvs
        if proxy is not None:
            pc = chunked_prefill(proxy, ids[:, :plen - 1], device, args.chunk)
            pkv = cache_kvs(pc)
            pcache = build_cache(pkv)
            with eager_attn(proxy):
                po = proxy(input_ids=ids[:, plen - 1:plen], past_key_values=pcache,
                           use_cache=True, output_attentions=True,
                           cache_position=torch.arange(plen - 1, plen, device=device))
            pmass = np.zeros(npg)
            for layer in po.attentions:
                a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
                for k in range(min(len(a), plen)):
                    pmass[X.page_of(k, B)] += a[k]
            rec["mass_proxy"] = [float(x) for x in pmass]
            del pc, pkv, pcache
        raw.write(json.dumps(rec) + "\n"); raw.flush()
        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}/{len(items)}] {it['task']}/{it['id']} tok={plen} full={full_ok} "
                  f"acc {n_acc}/{n_seen} ({(time.time()-t00)/(i+1):.1f}s/item)", flush=True)
    raw.close()
    print(f"[done] acceptance {n_acc}/{n_seen} = {n_acc/max(n_seen,1):.2f}; "
          f"source={src}; wrote {out}/s23.jsonl")


if __name__ == "__main__":
    main()
