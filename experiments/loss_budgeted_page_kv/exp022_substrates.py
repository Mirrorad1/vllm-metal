# SPDX-License-Identifier: Apache-2.0
"""exp022 — IS THE WALL SUBSTRATE-SHAPED OR CONTEXT-SHAPED? (the per-context loss matrix)

exp020 measured the incompressibility wall for ONE substrate: token EVICTION (sparse
retrieval 16x, dense aggregation 1.0x). This experiment fills the per-context matrix
L[context, substrate] at iso-BITS across four substrate families and asks whether the
contexts that defeat eviction also defeat every other way of shrinking the same bytes:

  S1 eviction   keep b*T tokens at 16 bit          (labels already measured; recomputed
                                                    here with NLL + cross-checked vs jsonl)
  S2 quant      keep ALL tokens at ~16b bits/elem  (sym per-group fake-quant; achieved
                                                    bits reported: 4b+overhead=0.266,
                                                    2b+overhead=0.156)
  S3 low-rank   per-layer SVD of K and V to rank r with r(T+D)/(TD)=b
  S4 weights    per-context LoRA trained on the raw context text, sized to param_bits =
                b * cache_bits, evaluated with the context REMOVED (question-only prompt)

Hypotheses (SPEC in STATUS_AND_ROADMAP Track C / lacuna L1):
  H-general : per-context failure sets coincide across substrates -> incompressibility is
              a property of the CONTEXT; the F4b escape hatch (distill to weights) fails
              exactly where it is needed; only dense-tier / re-read remain.
  H-specific: failure sets differ -> the admission gate upgrades to a substrate DISPATCHER.

Self-proving gates (the exp020 discipline — any FAIL aborts):
  g1 full-cache rebuilt-through-DynamicCache decode == accepted (>=95% correct)
  g2 identity transforms are no-ops: 16-bit quant and full-rank SVD reproduce full-cache
     answer logits within tolerance
  g3 the tightest transform (2-bit) must CHANGE the first instance's logits materially
  g4 recomputed eviction labels match the archived 7B H100 run (>=90% agreement)
  s4-gate sparse families must distill to >=0.5 fidelity, else the LoRA recipe itself is
     too weak and the dense result is uninterpretable (pause, not a verdict)

Runs on one 48GB card (L40S) for 7B/20k; local 0.5B --smoke validates every gate first.
Version-portable (transformers 4.51 local / >=5 pod): model load tries dtype= then
torch_dtype=; caches only touched via DynamicCache.update(); eager attention obtained via
set_attn_implementation when present else config._attn_implementation flip.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp020_quality_cuda as X  # make_task, page_of, select_pages (pure helpers)

try:
    from transformers import DynamicCache
except ImportError:  # very old fallback, not expected
    DynamicCache = None


# ---------------------------------------------------------------------------
# version-portable primitives
# ---------------------------------------------------------------------------
def load_model(name, dtype, device):
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    return m.to(device).eval()


@contextmanager
def eager_attn(model):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
        try:
            yield
        finally:
            model.set_attn_implementation("sdpa")
    else:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        prev = base.config._attn_implementation
        base.config._attn_implementation = "eager"
        try:
            yield
        finally:
            base.config._attn_implementation = prev


def cache_kvs(cache):
    if hasattr(cache, "layers"):                      # transformers >= 5
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def build_cache(kvs):
    c = DynamicCache()
    for i, (k, v) in enumerate(kvs):
        c.update(k.contiguous(), v.contiguous(), i)
    return c


@torch.no_grad()
def chunked_prefill(model, ids, device, chunk=2048):
    cache = None
    for s in range(0, ids.shape[1], chunk):
        e = min(s + chunk, ids.shape[1])
        cache = model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(s, e, device=device)).past_key_values
    return cache


@torch.no_grad()
def attention_mass(model, cache, last_tok, plen, n_pages, B, device):
    """exp020's query-agnostic per-page mass (mean heads, summed layers), portably."""
    with eager_attn(model):
        o = model(input_ids=last_tok, past_key_values=build_cache(cache_kvs(cache)),
                  use_cache=True, output_attentions=True,
                  cache_position=torch.arange(plen - 1, plen, device=device))
    mass = np.zeros(n_pages)
    for layer in o.attentions:
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[X.page_of(k, B)] += a[k]
    return mass


@torch.no_grad()
def answer_metrics(model, kvs, ids, plen, ans, device, keep_tok=None):
    """Teacher-forced decode over a (possibly transformed / sliced) cache.
    Returns (correct, gold_nll, logits_of_first_step). RoPE: absolute position_ids always;
    cache_position indexes the (possibly shorter) cache — exp020's fast-engine convention."""
    if keep_tok is not None:
        idx = torch.tensor(keep_tok, device=device, dtype=torch.long)
        kvs = [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in kvs]
    kept = kvs[0][0].shape[2]
    cache = build_cache(kvs)
    if len(ans) > 1:
        q = torch.cat([ids[:, plen - 1:plen],
                       torch.tensor([ans[:-1]], device=device, dtype=torch.long)], dim=1)
    else:
        q = ids[:, plen - 1:plen]
    n = q.shape[1]
    pos = torch.arange(plen - 1, plen - 1 + n, device=device)[None]
    cp = torch.arange(kept, kept + n, device=device)
    lg = model(input_ids=q, past_key_values=cache, use_cache=True,
               position_ids=pos, cache_position=cp).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    gold = torch.tensor(ans, device=device)
    nll = float(-lp[torch.arange(len(ans)), gold].mean())
    correct = all(int(lg[t].argmax()) == int(a) for t, a in enumerate(ans))
    return correct, nll, lg[0]


# ---------------------------------------------------------------------------
# substrate transforms (all on [1, kvh, T, hd] tensors; achieved bit-ratios reported)
# ---------------------------------------------------------------------------
def fake_quant(t, bits, group):
    x = t.float()
    *lead, d = x.shape
    assert d % group == 0, f"head_dim {d} % group {group}"
    x = x.reshape(*lead, d // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / max(qmax, 1)
    q = (x / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(*lead, d).to(t.dtype)


def quant_bits_ratio(bits, group):
    return (bits + 16.0 / group) / 16.0


def low_rank(t, r):
    _, h, T, d = t.shape
    xm = t[0].permute(1, 0, 2).reshape(T, h * d).float()
    U, S, Vh = torch.linalg.svd(xm, full_matrices=False)
    r = max(1, min(r, S.shape[0]))
    y = (U[:, :r] * S[:r]) @ Vh[:r]
    return y.reshape(T, h, d).permute(1, 0, 2)[None].to(t.dtype)


def rank_for_budget(b, T, D):
    return max(1, round(b * T * D / (T + D)))


def lowrank_bits_ratio(r, T, D):
    return r * (T + D) / (T * D)


# ---------------------------------------------------------------------------
# stage s23: eviction (recheck) + quantization + low-rank columns
# ---------------------------------------------------------------------------
def instances_from_jsonl(path):
    inst, labels = [], {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "instance":
                inst.append((d["family"], d["seed"], d["block_size"]))
            elif not d.get("event") and "correct" in d and d.get("policy") == "attention":
                labels[(d["family"], d["seed"], round(float(d["budget"]), 6))] = bool(d["correct"])
    return inst, labels


def run_s23(args, model, tok, device, dtype):
    if args.smoke:
        inst = [(f, s, args.block_size) for f in ("single_needle", "sum_scattered")
                for s in range(args.smoke_n)]
        labels = {}
    else:
        inst, labels = instances_from_jsonl(args.runs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / "s23.jsonl", "w")
    budgets = [0.25, 0.125, 0.0625]
    g_full = g_evict_match = g_evict_n = 0
    t00 = time.time()
    for i, (fam, seed, B) in enumerate(inst):
        p, a = X.make_task(fam, seed, args.n_filler)
        ids = tok(p + a, return_tensors="pt").input_ids.to(device)
        plen = tok(p, return_tensors="pt").input_ids.shape[1]
        ans = ids[0, plen:].tolist()
        npg = (plen + B - 1) // B
        cache = chunked_prefill(model, ids[:, :plen - 1], device, args.chunk)
        kvs = cache_kvs(cache)
        T = kvs[0][0].shape[2] + 1          # cached plen-1 (+ the query token itself)
        D = kvs[0][0].shape[1] * kvs[0][0].shape[3]
        rec = {"family": fam, "seed": int(seed), "P": int(npg), "plen": int(plen), "cells": {}}

        # full cache (gate g1) + mass for eviction
        c_full, nll_full, lg_full = answer_metrics(model, kvs, ids, plen, ans, device)
        g_full += int(c_full)
        rec["full"] = {"correct": bool(c_full), "nll": nll_full}
        mass = attention_mass(model, cache, ids[:, plen - 1:plen], plen, npg, B, device)
        rec["mass"] = [float(x) for x in mass]

        if i == 0:  # gates g2/g3 on the first instance only (identity + sensitivity)
            kv_id = [(fake_quant(k, 16, 64), fake_quant(v, 16, 64)) for k, v in kvs]
            _, _, lg_id = answer_metrics(model, kv_id, ids, plen, ans, device)
            d_id = float((lg_full - lg_id).abs().max())
            # SVD plumbing gate is VALUE-space: a reshape/permute bug gives O(1) relative
            # error; numerically-exact full-rank reconstruction amplified through the model
            # can legitimately move logits (measured ~0.6 at fp32/24 layers), so logit-space
            # would false-alarm. Answer-flip under full rank is reported but not asserted.
            k0 = kvs[0][0]
            rt = low_rank(k0, min(T - 1, D))
            rel = float((rt - k0).float().norm() / k0.float().norm())
            c_fr, _, _ = answer_metrics(
                model, [(low_rank(k, min(T - 1, D)), low_rank(v, min(T - 1, D)))
                        for k, v in kvs], ids, plen, ans, device)
            kv_2b = [(fake_quant(k, 2, 32), fake_quant(v, 2, 32)) for k, v in kvs]
            _, _, lg_2b = answer_metrics(model, kv_2b, ids, plen, ans, device)
            d_2b = float((lg_full - lg_2b).abs().max())
            tol = 0.75 if dtype in (torch.float16, torch.bfloat16) else 5e-2
            print(f"[gate g2] identity-quant max|dlogit|={d_id:.2e} (tol {tol}); "
                  f"svd round-trip rel-frob={rel:.2e} (tol 2e-2), answer preserved={c_fr == c_full}; "
                  f"[g3] 2-bit max|dlogit|={d_2b:.2e} (must be >1)", flush=True)
            assert d_id < tol, "GATE g2 FAIL: identity quant is not a no-op"
            assert rel < 2e-2, "GATE g2 FAIL: SVD round-trip does not reproduce the tensor (plumbing bug)"
            assert d_2b > 1.0, "GATE g3 FAIL: tight transform did not change logits"

        # S1 eviction (attention policy, exp020 selector) — recheck + NLL
        for b in budgets:
            keep = X.select_pages("attention", npg, b, mass)
            keep_tok = sorted({t for pg in keep for t in range(pg * B, min(pg * B + B, plen - 1))})
            c, nll, _ = answer_metrics(model, kvs, ids, plen, ans, device, keep_tok=keep_tok)
            rec["cells"][f"evict@{b}"] = {"correct": bool(c), "nll": nll, "bits": b}
            key = (fam, seed, round(b, 6))
            if key in labels:
                g_evict_n += 1
                g_evict_match += int(labels[key] == c)
        # S2 quantization
        for bits, group in ((4, 64), (2, 32)):
            kv_q = [(fake_quant(k, bits, group), fake_quant(v, bits, group)) for k, v in kvs]
            c, nll, _ = answer_metrics(model, kv_q, ids, plen, ans, device)
            rec["cells"][f"quant{bits}b"] = {"correct": bool(c), "nll": nll,
                                             "bits": quant_bits_ratio(bits, group)}
        # S3 low-rank
        for b in (0.25, 0.125):
            r = rank_for_budget(b, T, D)
            kv_r = [(low_rank(k, r), low_rank(v, r)) for k, v in kvs]
            c, nll, _ = answer_metrics(model, kv_r, ids, plen, ans, device)
            rec["cells"][f"lowrank@{b}"] = {"correct": bool(c), "nll": nll,
                                            "bits": lowrank_bits_ratio(r, T, D), "rank": r}
        del cache, kvs
        raw.write(json.dumps(rec) + "\n"); raw.flush()
        if i < 3 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(inst)}] {fam}/{seed} full={rec['full']['correct']} "
                  f"({(time.time()-t00)/(i+1):.1f}s/inst)", flush=True)
    raw.close()
    n = len(inst)
    print(f"[gate g1] full-cache correct {g_full}/{n} ({g_full/max(n,1):.2f}; need >=0.95)")
    if g_evict_n:
        print(f"[gate g4] eviction labels match archived run {g_evict_match}/{g_evict_n} "
              f"({g_evict_match/g_evict_n:.2f}; need >=0.90)")
    if not args.smoke:
        assert g_full / max(n, 1) >= 0.95, "GATE g1 FAIL"
        assert g_evict_n == 0 or g_evict_match / g_evict_n >= 0.90, "GATE g4 FAIL"
    print(f"[done] wrote {out}/s23.jsonl")


# ---------------------------------------------------------------------------
# stage s4: per-context LoRA (weights substrate), iso-param-bits, sharded
# ---------------------------------------------------------------------------
def lora_rank_for_budget(model, b, cache_bits):
    from peft.tuners.lora import LoraConfig  # noqa
    tgt = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    per_rank = 0
    for name, mod in model.named_modules():
        if any(name.endswith(t) for t in tgt) and hasattr(mod, "in_features"):
            per_rank += mod.in_features + mod.out_features
    r = max(1, round(b * cache_bits / 16.0 / per_rank))
    return r, tgt, per_rank


def run_s4(args, model, tok, device, dtype):
    from peft import LoraConfig, get_peft_model
    if args.smoke:
        picks = [(f, s, args.block_size) for f in ("single_needle", "sum_scattered")
                 for s in range(args.smoke_n)]
    else:
        inst, _ = instances_from_jsonl(args.runs)
        by_fam = {}
        for fam, seed, B in inst:
            by_fam.setdefault(fam, []).append((fam, seed, B))
        picks = []
        for fam in ("single_needle", "multi_hop", "sum_scattered"):
            picks += by_fam.get(fam, [])[:args.s4_per_family]
    k, N = (int(x) for x in args.shard.split("/"))
    picks = picks[k - 1::N]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    outf = out / f"s4_shard{k}of{N}.jsonl"
    done = set()
    if args.resume and outf.exists():
        for line in open(outf):
            d = json.loads(line)
            done.add((d["family"], d["seed"]))
    raw = open(outf, "a" if args.resume else "w")
    print(f"[s4] shard {k}/{N}: {len(picks)} contexts ({len(done)} already done, resumed)"
          if done else f"[s4] shard {k}/{N}: {len(picks)} contexts")
    print(f"[s4] recipe: epochs={args.s4_epochs} lr={args.s4_lr} alpha-ratio={args.s4_alpha_ratio} "
          f"chunk={args.s4_chunk} overlap=64")
    for i, (fam, seed, B) in enumerate(picks):
        if (fam, seed) in done:
            continue
        p, a = X.make_task(fam, seed, args.n_filler)
        q_only = p[p.rfind(". ") + 2:]
        assert 0 < len(q_only) < 200, f"question-tail extraction failed for {fam}"
        ids_q = tok(q_only + a, return_tensors="pt").input_ids.to(device)
        plen_q = tok(q_only, return_tensors="pt").input_ids.shape[1]
        ans_q = ids_q[0, plen_q:].tolist()
        ctx_ids = tok(p, return_tensors="pt").input_ids[0]
        cache_bits = int(ctx_ids.shape[0]) * (model.config.num_key_value_heads *
                        (model.config.hidden_size // model.config.num_attention_heads)) * 2 * 16
        r, tgt, per_rank = lora_rank_for_budget(model, args.s4_budget, cache_bits)
        bits_ratio = r * per_rank * 16.0 / cache_bits

        # floor: base model, question only, no context (should be ~0). Teacher-forced over
        # the FULL q+answer ids — forwarding only the question leaves a single valid logit
        # row and silently skips later answer tokens (all([]) == True false-positives).
        with torch.no_grad():
            lgf = model(input_ids=ids_q).logits[0].float()
        floor = all(int(lgf[plen_q - 1 + t].argmax()) == int(x) for t, x in enumerate(ans_q))

        cfg = LoraConfig(r=r, lora_alpha=args.s4_alpha_ratio * r, target_modules=tgt, lora_dropout=0.0,
                         task_type="CAUSAL_LM")
        pm = get_peft_model(model, cfg)
        for _, prm in pm.named_parameters():   # fp32 adapters on a bf16 base (peft casts x)
            if prm.requires_grad:
                prm.data = prm.data.float()
        pm.train()
        opt = torch.optim.AdamW((q for q in pm.parameters() if q.requires_grad), lr=args.s4_lr)
        step_sz = max(64, args.s4_chunk - 64)   # 64-token overlap: no fact straddles a boundary unseen
        chunks = [ctx_ids[s:s + args.s4_chunk] for s in range(0, ctx_ids.shape[0], step_sz)]
        rng = np.random.default_rng(seed)
        t0 = time.time()
        step = 0
        for ep in range(args.s4_epochs):
            for j in rng.permutation(len(chunks)):
                ch = chunks[int(j)].to(device)[None]
                if ch.shape[1] < 8:
                    continue
                loss = pm(input_ids=ch, labels=ch).loss
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
        pm.eval()
        with torch.no_grad():
            lg = pm(input_ids=ids_q).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        okv = [int(lg[plen_q - 1 + t].argmax()) == int(x) for t, x in enumerate(ans_q)]
        fid = all(okv)
        nll = float(-np.mean([float(lp[plen_q - 1 + t, x]) for t, x in enumerate(ans_q)]))
        secs = time.time() - t0
        pm = pm.unload() if hasattr(pm, "unload") else pm.base_model  # restore clean base
        raw.write(json.dumps({"event": "s4", "family": fam, "seed": int(seed), "rank": r,
                              "bits_ratio": bits_ratio, "steps": step,
                              "floor_correct": bool(floor), "fidelity_correct": bool(fid),
                              "nll": nll, "train_seconds": secs}) + "\n")
        raw.flush()
        print(f"  [{i+1}/{len(picks)}] {fam}/{seed} r={r} bits={bits_ratio:.3f} floor={floor} "
              f"LORA-fid={fid} nll={nll:.2f} ({secs:.0f}s)", flush=True)
    raw.close()
    print(f"[done] wrote {out}/s4_shard{k}of{N}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s23", "s4"], required=True)
    ap.add_argument("--runs", default="results/exp021_7b/raw_Qwen2.5-7B-Instruct.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-filler", type=int, default=1500)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="results/exp022_7b")
    ap.add_argument("--shard", default="1/1", help="k/N for s4 across pods")
    ap.add_argument("--s4-per-family", type=int, default=20)
    ap.add_argument("--s4-budget", type=float, default=0.25)
    ap.add_argument("--s4-epochs", type=int, default=10)
    ap.add_argument("--s4-alpha-ratio", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip (family,seed) already in the shard output")
    ap.add_argument("--s4-chunk", type=int, default=512)
    ap.add_argument("--s4-lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true", help="local 0.5B tiny validation")
    ap.add_argument("--smoke-n", type=int, default=3)
    args = ap.parse_args()
    if args.smoke:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
        args.n_filler = min(args.n_filler, 120)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (torch.bfloat16 if device == "cuda"
             else torch.float32)                      # MPS fp32: SVD/quant tolerance sanity
    model = load_model(args.model, dtype, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[exp022:{args.stage}] model={args.model} device={device} dtype={dtype} "
          f"smoke={args.smoke}")
    if args.stage == "s23":
        run_s23(args, model, tok, device, dtype)
    else:
        run_s4(args, model, tok, device, dtype)


if __name__ == "__main__":
    main()
