# SPDX-License-Identifier: Apache-2.0
"""exp020 (CUDA/transformers, memory-efficient) — budget-vs-quality on harder tasks.

Self-contained PyTorch/transformers port of exp020_quality.py for a GPU box (the MLX
harness is Apple-Metal-only). Measures the iso-quality KV budget for the DEPLOYABLE
selector (attention-mass / recent) on progressively harder RULER/NIAH-style tasks,
at 7B+ scale and LONG context.

Memory-efficient design (see SPEC_exp020_memeff.md): the big quality forwards run on
**sdpa (mem-efficient/flash backend forced)** with a 4D additive page-gating mask and
`logits_to_keep`, so neither the O(L²) score matrix nor the O(L·vocab) logits are
materialized → scales to 16k–41k context on one 80 GB GPU. The per-page attention-mass
selector is captured by momentarily switching to `eager` for a single O(L) one-token
forward (sdpa can't return attentions).

CORRECTNESS GATE (run first, fp32, abort on fail — hf-custom-attention-transplant):
  * full budget (drop nothing) gated logits == plain causal   (max|Δ| < 1e-3)
  * tight budget gated logits materially differ               (max|Δ| > 1e-2)

Run (see RUNPOD.md):
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  pip install "torch" "transformers>=5.0" accelerate numpy
  python exp020_quality_cuda.py --model Qwen/Qwen2.5-7B-Instruct --n 40 --n-filler 1500
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# tasks (identical to the MLX exp020; progressively harder / less redundant)
# ---------------------------------------------------------------------------
_FILLER = [
    "The weather today is mild with a gentle breeze over the quiet hills.",
    "Gardeners discuss the merits of compost and seasonal crop rotation.",
    "A distant train sounded its horn as the afternoon light faded slowly.",
    "People enjoy a warm cup of tea while reading quietly by the window.",
    "The museum exhibit featured pottery and woven baskets from the coast.",
    "Old maps show trade routes that crossed the wide and dusty plains.",
    "A quiet river wound past the mill where the old wheel still turned.",
    "Shelves of books lined the long hall, their spines cracked and faded.",
]


def _filler(rng, n):
    return " ".join(rng.choice(_FILLER) for _ in range(n))


def make_task(family, seed, n_filler):
    rng = random.Random(seed)
    pre, post = _filler(rng, n_filler // 2), _filler(rng, n_filler - n_filler // 2)
    if family == "single_needle":
        c = rng.randint(10000, 99999)
        return f"Remember this. {pre} Cabinet 4731 was issued token {c}. {post} Cabinet 4731 was issued token", f" {c}"
    if family == "exact_long_string":
        s = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
        return f"Record this exactly. {pre} The verification key is {s}. {post} The verification key is", f" {s}"
    if family == "multi_needle":
        ks = {f"vault {rng.randint(1000,9999)}": rng.randint(10000,99999) for _ in range(4)}
        tgt = rng.choice(list(ks))
        facts = " ".join(f"The code for {k} is {v}." for k, v in ks.items())
        return f"{pre} {facts} {_filler(rng,4)} {post} The code for {tgt} is", f" {ks[tgt]}"
    if family == "multi_hop":
        person = rng.choice(["Mara", "Tomas", "Yuki", "Devon"])
        room = rng.choice(["the vault", "the archive", "the cellar"])
        num = rng.randint(100, 999)
        return (f"{pre} {person} works in {room}. {_filler(rng,4)} {post} The {room.split()[-1]} holds "
                f"exactly {num} boxes. {_filler(rng,3)} The number of boxes in the room where {person} works is"), f" {num}"
    if family == "distractor":
        tgt = rng.randint(100, 999)
        dec = " ".join(f"A decoy total is {rng.randint(100,999)}." for _ in range(4))
        return f"{pre} {dec} The OFFICIAL total is {tgt}. {dec} {post} The OFFICIAL total is", f" {tgt}"
    raise KeyError(family)


FAMILIES = ["single_needle", "exact_long_string", "multi_needle", "multi_hop", "distractor"]


# ---------------------------------------------------------------------------
# memory-efficient forward primitives
# ---------------------------------------------------------------------------
def _sdpa_ctx(device):
    # force the fused backends on GPU so a custom float mask can't silently fall back to
    # the MATH backend (which re-materializes O(L²) and OOMs). No-op on CPU.
    if device == "cuda":
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])
    return contextlib.nullcontext()


def page_of(pos, block_size):
    return pos // block_size


def build_mask(L, prompt_len, dropped_key_positions, dtype, device):
    """4D additive mask [1,1,L,L]: causal everywhere; ANSWER-query rows (>=prompt_len)
    additionally blocked from the dropped key positions (prefill stays lossless)."""
    neg = torch.finfo(dtype).min
    m = torch.zeros((L, L), dtype=dtype, device=device)
    m.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1), neg)
    if dropped_key_positions:
        drop = torch.tensor(sorted(dropped_key_positions), device=device, dtype=torch.long)
        m[prompt_len:, drop] = neg
    return m[None, None]


@torch.no_grad()
def quality_forward(model, ids, mask4d, n_keep, device):
    """Big pass: sdpa (fused) + 4D mask + only the last n_keep logits. Returns [1,n_keep,vocab]."""
    with _sdpa_ctx(device):
        out = model(input_ids=ids, attention_mask=mask4d, use_cache=False, logits_to_keep=n_keep)
    return out.logits


def exact_answer_ok(logits_kept, answer_ids):
    """logits_kept[0,t] predicts answer token t (n_keep=len(answer)+1 ⇒ window starts at prompt_len-1)."""
    for t, a in enumerate(answer_ids):
        if int(logits_kept[0, t].argmax()) != int(a):
            return False
    return True


@torch.no_grad()
def attention_mass(model, prompt_ids, n_pages, block_size, device):
    """Per-page attention mass of the decode query (last prompt token), averaged over
    layers & heads. Cache built under sdpa (O(L)); attentions captured by a single
    one-token EAGER forward (q_len=1 ⇒ O(L))."""
    plen = prompt_ids.shape[1]
    with _sdpa_ctx(device):
        cache = model(input_ids=prompt_ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
    model.set_attn_implementation("eager")
    try:
        o2 = model(input_ids=prompt_ids[:, -1:], past_key_values=cache, use_cache=True,
                   output_attentions=True, cache_position=torch.tensor([plen - 1], device=device))
        atts = o2.attentions
    finally:
        model.set_attn_implementation("sdpa")
    mass = np.zeros(n_pages)
    for layer in atts:                                   # [1, heads, 1, kv_len=plen]
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[page_of(k, block_size)] += a[k]
    return mass


def select_pages(policy, n_pages, budget, mass, sink=1, recent=1):
    J = max(min(n_pages, sink + recent), round(budget * n_pages))
    floor = {n_pages - 1} | set(range(min(sink, n_pages))) | set(range(max(0, n_pages - recent), n_pages))
    order = list(range(n_pages - 1, -1, -1)) if policy == "recent" else list(np.argsort(-mass))
    keep = set(floor)
    for p in order:
        if len(keep) >= J:
            break
        keep.add(int(p))
    return sorted(keep)


def correctness_gate(model, tok, dtype, device, block_size):
    p, _ = make_task("single_needle", 0, 60)
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    L = ids.shape[1]
    base = quality_forward(model, ids, build_mask(L, L, [], dtype, device), 2, device)        # plain causal
    full = quality_forward(model, ids, build_mask(L, L - 1, [], dtype, device), 2, device)    # answer row, drop nothing
    d_full = (base - full).abs().max().item()
    n_pages = (L + block_size - 1) // block_size
    drop = [k for k in range(L - 1) if page_of(k, block_size) < n_pages - 1]
    tight = quality_forward(model, ids, build_mask(L, L - 1, drop, dtype, device), 2, device)
    d_tight = (base - tight).abs().max().item()
    ok = d_full < 1e-3 and d_tight > 1e-2
    print(f"[gate] full-keep vs causal max|Δ|={d_full:.2e} (<1e-3); "
          f"tight vs causal max|Δ|={d_tight:.2e} (>1e-2) => {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("correctness gate FAILED — the mask is wrong; every downstream number is meaningless.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-filler", type=int, default=1500)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--iso-thresh", type=float, default=0.9)
    ap.add_argument("--gate-dtype", default="float32")
    ap.add_argument("--run-dtype", default="bfloat16")
    ap.add_argument("--out", default="exp020_cuda_results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    B = args.block_size

    print("loading model (gate, fp32, sdpa)…", flush=True)
    gate_model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=getattr(torch, args.gate_dtype)).to(device).eval()
    correctness_gate(gate_model, tok, getattr(torch, args.gate_dtype), device, B)
    del gate_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("loading model (run, bf16, sdpa)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=getattr(torch, args.run_dtype)).to(device).eval()
    rdtype = getattr(torch, args.run_dtype)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    raw = open(out / f"raw_{tag}.jsonl", "w")
    records = []
    t0 = time.perf_counter()
    for fam in args.families:
        acc = 0; seed = 0
        while acc < args.n and seed < args.n * 6:
            prompt, answer = make_task(fam, seed, args.n_filler); seed += 1
            ids_full = tok(prompt + answer, return_tensors="pt").input_ids.to(device)
            prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
            answer_ids = ids_full[0, prompt_len:].tolist()
            if not answer_ids:
                continue
            L = ids_full.shape[1]
            n_keep = len(answer_ids) + 1
            n_pages = (prompt_len + B - 1) // B
            try:
                logits_full = quality_forward(model, ids_full, build_mask(L, prompt_len, [], rdtype, device), n_keep, device)
                if not exact_answer_ok(logits_full, answer_ids):
                    del logits_full; continue
                del logits_full
                mass = attention_mass(model, ids_full[:, :prompt_len], n_pages, B, device)
            except torch.cuda.OutOfMemoryError as e:
                raw.write(json.dumps({"event": "oom", "fam": fam, "L": int(L), "err": str(e)[:120]}) + "\n")
                torch.cuda.empty_cache(); continue
            acc += 1
            for bf in args.budgets:
                for policy in ("attention", "recent"):
                    keep = select_pages(policy, n_pages, bf, mass)
                    dropped = [k for k in range(prompt_len) if page_of(k, B) not in set(keep)]
                    lg = quality_forward(model, ids_full, build_mask(L, prompt_len, dropped, rdtype, device), n_keep, device)
                    rec = {"family": fam, "budget": bf, "policy": policy,
                           "J": len(keep), "P": n_pages, "L": int(L), "correct": bool(exact_answer_ok(lg, answer_ids))}
                    records.append(rec); raw.write(json.dumps(rec) + "\n"); del lg
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"{fam}: {acc} valid ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    summary = {"model": args.model, "by_family": {}}
    print(f"\n{'family':18s} {'budget':>7} {'attn':>6} {'recent':>7}")
    for fam in args.families:
        fr = [r for r in records if r["family"] == fam]
        if not fr:
            continue
        full = float(np.mean([r["correct"] for r in fr if abs(r["budget"]-1.0) < 1e-9 and r["policy"]=="attention"])) or 1.0
        per = {}
        for bf in args.budgets:
            sub = [r for r in fr if abs(r["budget"]-bf) < 1e-9]
            at = float(np.mean([r["correct"] for r in sub if r["policy"]=="attention"]))
            rc = float(np.mean([r["correct"] for r in sub if r["policy"]=="recent"]))
            per[bf] = {"attention": at, "recent": rc}
            print(f"{fam:18s} {bf:>7.4f} {at:>6.2f} {rc:>7.2f}")
        iso = 1.0
        for bf in sorted(args.budgets, reverse=True):
            if per.get(bf, {}).get("attention", 0) >= args.iso_thresh * full:
                iso = bf
            else:
                break
        summary["by_family"][fam] = {"full_acc": full, "iso_quality_budget": iso,
                                     "capacity_multiplier_at_iso": round(1.0/iso, 2),
                                     "context_tokens": int(np.median([r["L"] for r in fr])), "per_budget": per}
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== iso-quality budget (deployable attention selector) ===")
    for fam, s in summary["by_family"].items():
        print(f"  {fam:18s} ctx~{s['context_tokens']:>6}tok  iso-budget={s['iso_quality_budget']:.4f} "
              f"→ {s['capacity_multiplier_at_iso']}× capacity")
    mults = [s["capacity_multiplier_at_iso"] for s in summary["by_family"].values()]
    if mults:
        print(f"\nrealistic capacity multiplier: worst-task {min(mults):.1f}× … best-task {max(mults):.1f}×")


if __name__ == "__main__":
    main()
