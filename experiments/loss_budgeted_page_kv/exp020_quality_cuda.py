# SPDX-License-Identifier: Apache-2.0
"""exp020 (CUDA/transformers port) — budget-vs-quality on harder long-context tasks.

Self-contained PyTorch/transformers port of exp020_quality.py for a RunPod (CUDA)
GPU box — the MLX harness is Apple-Metal-only. Measures the iso-quality KV budget
for the DEPLOYABLE selector (attention-mass / recent) on progressively harder
RULER/NIAH-style tasks, at 7B+ scale and longer context than the local Mac allows.

Mechanism: page-gating via a 4D ADDITIVE attention mask (causal everywhere; for the
ANSWER-query rows, block attention to keys in dropped pages). No KV-cache surgery,
no custom-attention registration, no position-id traps. `attn_implementation="eager"`
so output_attentions gives the per-page attention mass that drives the selector.

CORRECTNESS GATE (run first, abort on fail — see hf-custom-attention-transplant):
  * full budget (no pages dropped) gated logits == plain causal logits  (max|Δ| < 1e-3, fp32)
  * tight budget gated logits materially differ                          (max|Δ| > 1e-2)
A subtly-wrong mask produces plausible logits, so you'd measure a bug, not the idea.

Run:
  pip install "torch" "transformers>=4.44" accelerate
  HF_HOME=/workspace/hf python exp020_quality_cuda.py \
      --model Qwen/Qwen2.5-7B-Instruct --n 30 --n-filler 300 --block-size 16
See RUNPOD.md.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
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
                f"exactly {num} boxes. {_filler(rng,3)} How many boxes are where {person} works? Answer:"), f" {num}"
    if family == "distractor":
        tgt = rng.randint(100, 999)
        dec = " ".join(f"A decoy total is {rng.randint(100,999)}." for _ in range(4))
        return f"{pre} {dec} The OFFICIAL total is {tgt}. {dec} {post} The OFFICIAL total is", f" {tgt}"
    raise KeyError(family)


FAMILIES = ["single_needle", "exact_long_string", "multi_needle", "multi_hop", "distractor"]
NEG = None  # set per-dtype


def build_mask(L, prompt_len, dropped_key_positions, dtype, device):
    """4D additive mask [1,1,L,L]: causal everywhere; ANSWER-query rows (>=prompt_len)
    additionally blocked from the dropped key positions (prompt prefill stays lossless)."""
    neg = torch.finfo(dtype).min
    m = torch.zeros((L, L), dtype=dtype, device=device)
    m.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1), neg)
    if dropped_key_positions:
        drop = torch.tensor(sorted(dropped_key_positions), device=device, dtype=torch.long)
        m[prompt_len:, drop] = neg  # answer rows can't see dropped prompt pages
    return m[None, None]


@torch.no_grad()
def forward_logits(model, ids, mask4d, want_attn=False):
    out = model(input_ids=ids, attention_mask=mask4d, output_attentions=want_attn, use_cache=False)
    return out.logits, (out.attentions if want_attn else None)


def page_of(pos, block_size):
    return pos // block_size


def exact_answer_ok(logits, prompt_len, answer_ids):
    # logits at position prompt_len-1+t predicts answer token t
    for t, a in enumerate(answer_ids):
        if int(logits[0, prompt_len - 1 + t].argmax()) != int(a):
            return False
    return True


@torch.no_grad()
def prompt_attention_mass(model, prompt_ids, n_pages, block_size):
    """Per-page attention mass of the decode query (last prompt token) over the prompt,
    averaged over layers & heads. Uses a single-token CACHED forward so attentions are
    [1,heads,1,L] (O(L)) instead of output_attentions over the full forward (O(L²) per
    layer — OOMs at long context). Requires attn_implementation='eager'."""
    plen = prompt_ids.shape[1]
    out = model(input_ids=prompt_ids[:, :-1], use_cache=True)          # build cache, no attentions
    pos = torch.tensor([plen - 1], device=prompt_ids.device)
    o2 = model(input_ids=prompt_ids[:, -1:], past_key_values=out.past_key_values,
               use_cache=True, output_attentions=True, cache_position=pos)
    mass = np.zeros(n_pages)
    for layer in o2.attentions:                                       # [1, heads, 1, kv_len=plen]
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[page_of(k, block_size)] += a[k]
    return mass


def select_pages(policy, n_pages, budget, page_mass, sink=1, recent=1):
    J = max(min(n_pages, sink + recent), round(budget * n_pages))
    floor = set([n_pages - 1]) | set(range(min(sink, n_pages))) | set(range(max(0, n_pages - recent), n_pages))
    if policy == "recent":
        order = list(range(n_pages - 1, -1, -1))
    elif policy == "attention":
        order = list(np.argsort(-page_mass))
    else:
        raise ValueError(policy)
    keep = set(floor)
    for p in order:
        if len(keep) >= J:
            break
        keep.add(int(p))
    return sorted(keep)


def correctness_gate(model, tok, dtype, device, block_size):
    """Two-sided gate: full-keep == causal; tight-keep differs. Abort on failure."""
    p, a = make_task("single_needle", 0, 60)
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    L = ids.shape[1]
    plain = build_mask(L, L, [], dtype, device)        # pure causal (prompt only; no answer rows)
    base, _ = forward_logits(model, ids, plain)
    # full-keep gated: mark all "answer rows" but drop NO pages → must equal causal
    full = build_mask(L, L - 1, [], dtype, device)     # treat last token as an answer row, drop nothing
    g_full, _ = forward_logits(model, ids, full)
    d_full = (base - g_full).abs().max().item()
    # tight: drop all but the last page for the answer row
    n_pages = (L + block_size - 1) // block_size
    drop = [k for k in range(L - 1) if page_of(k, block_size) < n_pages - 1]
    tight = build_mask(L, L - 1, drop, dtype, device)
    g_tight, _ = forward_logits(model, ids, tight)
    d_tight = (base - g_tight).abs().max().item()
    ok = d_full < 1e-3 and d_tight > 1e-2
    print(f"[gate] full-keep vs causal max|Δ|={d_full:.2e} (<1e-3); "
          f"tight vs causal max|Δ|={d_tight:.2e} (>1e-2) => {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("correctness gate FAILED — the mask is wrong; every downstream number is meaningless.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-filler", type=int, default=300)
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

    # gate in fp32 (noise can't mask a bug), then run in bf16 for speed
    print("loading model (gate, fp32)…", flush=True)
    gate_model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", torch_dtype=getattr(torch, args.gate_dtype)).to(device).eval()
    correctness_gate(gate_model, tok, getattr(torch, args.gate_dtype), device, B)
    del gate_model
    if device == "cuda":
        torch.cuda.empty_cache()
    print("loading model (run, bf16)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", torch_dtype=getattr(torch, args.run_dtype)).to(device).eval()
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
            n_pages = (prompt_len + B - 1) // B
            # full reference (masked forward, NO output_attentions → no O(L²) storage)
            full_mask = build_mask(L, prompt_len, [], rdtype, device)
            logits_full, _ = forward_logits(model, ids_full, full_mask)
            if not exact_answer_ok(logits_full, prompt_len, answer_ids):
                del logits_full; continue
            # attention-mass selector via the cheap single-token cached forward
            mass = prompt_attention_mass(model, ids_full[:, :prompt_len], n_pages, B)
            acc += 1
            for bf in args.budgets:
                for policy in ("attention", "recent"):
                    keep = select_pages(policy, n_pages, bf, mass)
                    dropped = [k for k in range(prompt_len) if page_of(k, B) not in set(keep)]
                    gm = build_mask(L, prompt_len, dropped, rdtype, device)
                    lg, _ = forward_logits(model, ids_full, gm)
                    ok = exact_answer_ok(lg, prompt_len, answer_ids)
                    rec = {"family": fam, "budget": bf, "policy": policy,
                           "J": len(keep), "P": n_pages, "correct": bool(ok)}
                    records.append(rec); raw.write(json.dumps(rec) + "\n")
                    del lg
            del logits_full
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"{fam}: {acc} valid ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    # iso-quality budget per family for the deployable (attention) selector
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
                                     "capacity_multiplier_at_iso": round(1.0/iso, 2), "per_budget": per}
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== iso-quality budget (deployable attention selector) ===")
    for fam, s in summary["by_family"].items():
        print(f"  {fam:18s} iso-budget={s['iso_quality_budget']:.4f} → {s['capacity_multiplier_at_iso']}× capacity")
    mults = [s["capacity_multiplier_at_iso"] for s in summary["by_family"].values()]
    if mults:
        print(f"\nrealistic capacity multiplier across task difficulty: "
              f"worst-task {min(mults):.1f}× … best-task {max(mults):.1f}×")


if __name__ == "__main__":
    main()
