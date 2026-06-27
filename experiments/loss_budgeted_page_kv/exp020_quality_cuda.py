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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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


def _scatter(rng, n_filler, facts):
    """Interleave fact sentences at ~evenly distributed positions across n_filler filler
    sentences, so the K facts land on DIFFERENT, far-apart pages (the dense regime: the
    selector must keep them ALL at once, and recency keeps none of the early ones)."""
    seg = max(1, n_filler // (len(facts) + 1))
    out = []
    for f in facts:
        out.append(_filler(rng, seg)); out.append(f)
    out.append(_filler(rng, seg))
    return " ".join(out)


# dense-task breadth: how many distributed spans the answer depends on (all must survive)
SUM_K = 5        # sum_scattered: addends
RECALL_K = 3     # recall_all: values to reproduce in order (6 was unsolvable even at full
                 # cache on 7B @20k → 0 valid; 3 gives a baseline. Still validate it accepts >0.)


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
        # true 2-hop w/ interference: 3 rooms, different counts, placed EARLY (far from
        # the query) so recent-only fails; model must link person -> room -> that count.
        rooms = rng.sample(["vault", "archive", "cellar", "loft", "annex"], 3)
        counts = {r: rng.randint(100, 999) for r in rooms}
        person = rng.choice(["Mara", "Tomas", "Yuki", "Devon"])
        target = rng.choice(rooms)
        facts = (f" {person} works in the {target}."
                 + "".join(f" The {r} holds exactly {counts[r]} boxes." for r in rooms))
        return (f"Read carefully.{facts} {pre} {post} "
                f"The number of boxes in the room where {person} works is"), f" {counts[target]}"
    if family == "distractor":
        tgt = rng.randint(100, 999)
        dec = " ".join(f"A decoy total is {rng.randint(100,999)}." for _ in range(4))
        return f"{pre} {dec} The OFFICIAL total is {tgt}. {dec} {post} The OFFICIAL total is", f" {tgt}"
    if family == "sum_scattered":
        # DENSE aggregation: answer = sum of SUM_K small deposits scattered far apart. The
        # answer depends on EVERY addend page (drop one → wrong sum), and no single page is
        # query-salient ("the total" doesn't point at any one entry) → the selector must keep
        # them ALL within budget. Small 2-digit terms keep the full-cache sum solvable.
        amts = [rng.randint(10, 39) for _ in range(SUM_K)]
        facts = [f"Ledger entry {i+1}: a deposit of {a} dollars." for i, a in enumerate(amts)]
        body = _scatter(rng, n_filler, facts)
        return (f"Bookkeeping log. {body} Adding up every deposit listed above, "
                f"the total number of dollars is"), f" {sum(amts)}"
    if family == "recall_all":
        # DENSE exact recall (no arithmetic): reproduce ALL RECALL_K codes in order. Every
        # value page must survive simultaneously; large answer space (no lucky guess).
        codes = [rng.randint(10, 99) for _ in range(RECALL_K)]
        facts = [f"Channel {i+1} broadcasts code {c}." for i, c in enumerate(codes)]
        body = _scatter(rng, n_filler, facts)
        ans = "".join(f" {c}" for c in codes)
        return (f"Signal record. {body} Listing the codes for channels 1 through "
                f"{RECALL_K} in order, they are:"), ans
    raise KeyError(family)


# default = 5 retrieval families + 2 DENSE-integration families (sum_scattered, recall_all);
# the dense pair is the test of whether the 8-16× retrieval multiplier survives on workloads
# whose answer depends on MANY distributed spans at once (run them alone with --families).
FAMILIES = ["single_needle", "exact_long_string", "multi_needle", "multi_hop", "distractor",
            "sum_scattered", "recall_all"]


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
    """4D additive mask [1,1,L,L]: causal everywhere; every DECODE-query row additionally
    blocked from the dropped key positions (prefill stays lossless).

    The first decode query is the LAST PROMPT TOKEN (row prompt_len-1) — it predicts
    answer[0] and in a gated deployment attends only to the kept cache. So gating MUST start
    at row prompt_len-1, not prompt_len; otherwise answer[0] cheats by seeing dropped pages
    (a real off-by-one that diverges from the cache-slicing fast path at any budget < 1.0)."""
    neg = torch.finfo(dtype).min
    m = torch.zeros((L, L), dtype=dtype, device=device)
    m.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1), neg)
    if dropped_key_positions:
        drop = torch.tensor(sorted(dropped_key_positions), device=device, dtype=torch.long)
        m[prompt_len - 1:, drop] = neg
    # A query token always has its OWN KV in a real decode (it is the live token, never evicted),
    # so no query row may be fully masked. Guaranteeing the diagonal is unmasked matches the
    # cache-slicing fast path, and prevents a softmax-over-all-(-inf) NaN if a budget ever drops
    # the page a (gated) query row sits in. No-op for valid rows (their own column is never dropped).
    m.fill_diagonal_(0)
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
    full = quality_forward(model, ids, build_mask(L, L - 1, [], dtype, device), 2, device)    # gated row, drop nothing
    d_full = (base - full).abs().max().item()
    n_pages = (L + block_size - 1) // block_size
    # tight: gate the LAST token (row L-1 — always in the kept last page, like the experiment's
    # first decode query, so never starved) and drop every earlier page. The mask must materially
    # change that position's logits.
    drop = [k for k in range(L) if page_of(k, block_size) < n_pages - 1]
    tight = quality_forward(model, ids, build_mask(L, L, drop, dtype, device), 2, device)
    d_tight = (base - tight).abs().max().item()
    has_nan = bool(torch.isnan(tight).any() or torch.isnan(full).any())
    ok = d_full < 1e-3 and d_tight > 1e-2 and not has_nan
    print(f"[gate] full-keep vs causal max|Δ|={d_full:.2e} (<1e-3); "
          f"tight vs causal max|Δ|={d_tight:.2e} (>1e-2){' NaN!' if has_nan else ''} "
          f"=> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("correctness gate FAILED — the mask is wrong; every downstream number is meaningless.")


# ---------------------------------------------------------------------------
# FAST path (lever 1): prefill the prompt ONCE, then decode-only per budget over a
# sliced cache. ~5-10x fewer long forwards. Proven equivalent to the 4D-mask path
# by the built-in preflight (run_preflight) below — abort if it ever disagrees.
# ---------------------------------------------------------------------------
@torch.no_grad()
def prefill_template(model, prompt_ids, device):
    """KV cache over prompt[:-1] (positions 0..plen-2); the last prompt token is the
    first decode query. Never forwarded-into directly — copies are made per use."""
    with _sdpa_ctx(device):
        return model(input_ids=prompt_ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values


def _keep_tok_for_cache(keep_pages, prompt_len, B):
    return sorted({t for p in keep_pages for t in range(p * B, min(p * B + B, prompt_len - 1))})


def _fresh_sliced(template, keep_tok, device):
    new = DynamicCache()
    idx = torch.tensor(keep_tok, device=device, dtype=torch.long)
    for i, layer in enumerate(template.layers):
        new.update(layer.keys.index_select(2, idx), layer.values.index_select(2, idx), i)
    return new


@torch.no_grad()
def fast_answer_logits(model, template, prompt_ids, answer_ids, keep_pages, B, device):
    """Decode the answer over a cache sliced to keep_pages, with ABSOLUTE position_ids
    (RoPE) and sliced cache_position. Returns the answer-position logits [len(answer), vocab]
    — logits[t] predicts answer token t (same alignment as the mask path's kept window)."""
    plen = prompt_ids.shape[1]
    keep_tok = _keep_tok_for_cache(keep_pages, plen, B)
    cache = _fresh_sliced(template, keep_tok, device)
    kept = len(keep_tok)
    if len(answer_ids) > 1:
        q = torch.cat([prompt_ids[:, -1:], torch.tensor([answer_ids[:-1]], device=device, dtype=torch.long)], dim=1)
    else:
        q = prompt_ids[:, -1:]
    n = q.shape[1]
    pos = torch.arange(plen - 1, plen - 1 + n, device=device)[None]
    cp = torch.arange(kept, kept + n, device=device)
    with _sdpa_ctx(device):
        lg = model(input_ids=q, past_key_values=cache, use_cache=True, position_ids=pos, cache_position=cp).logits
    return lg[0, :len(answer_ids)]


def gated_exact_answer_fast(model, template, prompt_ids, answer_ids, keep_pages, B, device):
    """Exact-answer bool over the sliced cache (argmax of fast_answer_logits)."""
    lg = fast_answer_logits(model, template, prompt_ids, answer_ids, keep_pages, B, device)
    return all(int(lg[t].argmax()) == int(a) for t, a in enumerate(answer_ids))


@torch.no_grad()
def attention_mass_fast(model, template, prompt_ids, n_pages, block_size, device):
    """Attention mass of the last prompt token over the cached prompt (reuses the
    prefill template; one eager one-token forward)."""
    plen = prompt_ids.shape[1]
    full = _fresh_sliced(template, list(range(plen - 1)), device)
    model.set_attn_implementation("eager")
    try:
        o2 = model(input_ids=prompt_ids[:, -1:], past_key_values=full, use_cache=True,
                   output_attentions=True, position_ids=torch.tensor([[plen - 1]], device=device),
                   cache_position=torch.arange(plen - 1, plen, device=device))
        atts = o2.attentions
    finally:
        model.set_attn_implementation("sdpa")
    mass = np.zeros(n_pages)                              # kv_len = (plen-1 cached) + 1 current = plen
    for layer in atts:                                    # [1, heads, 1, plen]; index plen-1 = self-token
        a = layer[0, :, 0, :plen].float().mean(0).cpu().numpy()
        for k in range(min(len(a), plen)):
            mass[page_of(k, block_size)] += a[k]
    return mass


PREFLIGHT_TOL = 5e-2  # fp32 max|Δ logits| between fast & mask; ~1e-3 in practice, a real
                      # logic bug gives O(1+). Run in fp32 so argmax ties don't false-alarm.


def run_preflight(model, tok, dtype, device, B, budgets, families=FAMILIES):
    """Prove the WHOLE fast path ≡ the validated mask path before trusting any result.
    Runs in fp32 (the gate model) so the test is on NUMERICAL equivalence, not a brittle
    bf16 argmax that can flip on a low-confidence answer token (a tie is not a bug):
      (a) selector: attention_mass_fast picks the SAME pages as the mask-path selector;
      (b) decode:   the fast answer-LOGITS equal the mask answer-logits (max|Δ| < tol) —
          strictly stronger than matching the final exact-answer decision.
    Covers EVERY family in the run at least once (incl. multi-token recall_all). On any real
    divergence, prints a per-case diagnostic (family/budget/policy, token, Δ) and aborts."""
    sel_mism = sel_tot = dec_flip = dec_tot = 0
    max_delta = 0.0; worst = None
    for s in range(max(6, len(families))):
        fam = families[s % len(families)]
        prompt, answer = make_task(fam, 1000 + s, 80)
        ids = tok(prompt + answer, return_tensors="pt").input_ids.to(device)
        plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
        ans = ids[0, plen:].tolist()
        if not ans:
            continue
        L = ids.shape[1]; nk = len(ans) + 1; npg = (plen + B - 1) // B
        template = prefill_template(model, ids[:, :plen], device)
        mass = attention_mass(model, ids[:, :plen], npg, B, device)          # validated signal
        mass_f = attention_mass_fast(model, template, ids[:, :plen], npg, B, device)  # fast signal
        for bf in budgets:
            for policy in ("attention", "recent"):
                keep = select_pages(policy, npg, bf, mass)        # mask-path selection (reference)
                keep_f = select_pages(policy, npg, bf, mass_f)    # fast-path selection (production)
                sel_tot += 1; sel_mism += int(keep != keep_f)
                # decode equivalence on the SAME (reference) keep-set isolates the decode path:
                dropped = [k for k in range(plen) if page_of(k, B) not in set(keep)]
                lm = quality_forward(model, ids, build_mask(L, plen, dropped, dtype, device), nk, device)[0, :len(ans)]
                lf = fast_answer_logits(model, template, ids[:, :plen], ans, keep, B, device)
                d = (lm.float() - lf.float()).abs().max().item()
                if d > max_delta:
                    max_delta = d; worst = (fam, bf, policy)
                ref = all(int(lm[t].argmax()) == int(a) for t, a in enumerate(ans))
                fst = all(int(lf[t].argmax()) == int(a) for t, a in enumerate(ans))
                dec_tot += 1; dec_flip += int(ref != fst)
                if d > PREFLIGHT_TOL:
                    print(f"[fast-preflight] DIVERGENCE {fam} bf={bf} {policy}: max|Δlogits|={d:.3e}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    ok = max_delta < PREFLIGHT_TOL and sel_mism == 0
    print(f"[fast-preflight] fp32 max|Δ logits| fast vs mask = {max_delta:.2e} (<{PREFLIGHT_TOL:.0e}"
          f"{'' if worst is None else f', worst={worst[0]}/{worst[1]}/{worst[2]}'}); "
          f"selector {sel_tot-sel_mism}/{sel_tot} identical; "
          f"decisions {dec_tot-dec_flip}/{dec_tot} identical => {'PASS' if ok else 'FAIL'}", flush=True)
    if dec_flip and ok:
        print(f"[fast-preflight] note: {dec_flip}/{dec_tot} exact-answer decisions differ despite "
              f"logits equal within {PREFLIGHT_TOL:.0e} — an argmax tie at the decision boundary "
              f"(affects fast & mask equally), NOT a fast-path error.", flush=True)
    if not ok:
        raise SystemExit("fast path NUMERICALLY disagrees with the validated mask path "
                         f"(max|Δ|={max_delta:.2e} ≥ {PREFLIGHT_TOL:.0e}) — refusing to use it.")


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
    ap.add_argument("--engine", choices=["mask", "fast"], default="fast",
                    help="fast = prefill prompt once + decode-only per budget (lever 1, ~5-10x fewer "
                         "long forwards); mask = the validated 4D-mask reference path. fast self-proves "
                         "equivalence to mask via a preflight before any result is trusted.")
    ap.add_argument("--out", default="exp020_cuda_results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    B = args.block_size

    print("loading model (gate, fp32, sdpa)…", flush=True)
    gate_dtype = getattr(torch, args.gate_dtype)
    gate_model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=gate_dtype).to(device).eval()
    correctness_gate(gate_model, tok, gate_dtype, device, B)
    if args.engine == "fast":
        # Prove fast≡mask in fp32 on the gate model, BEFORE freeing it — numerical equivalence
        # (max|Δ logits|), so a bf16 argmax tie on a low-confidence answer token can't false-alarm.
        print("engine=fast → proving fast≡mask (fp32 numerical equivalence) before any result…", flush=True)
        run_preflight(gate_model, tok, gate_dtype, device, B, args.budgets, families=args.families)
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
                if args.engine == "fast":
                    # lever 1: prefill the prompt ONCE; reference + every budget decode over slices.
                    template = prefill_template(model, ids_full[:, :prompt_len], device)
                    if not gated_exact_answer_fast(model, template, ids_full[:, :prompt_len], answer_ids,
                                                   list(range(n_pages)), B, device):
                        del template; continue
                    mass = attention_mass_fast(model, template, ids_full[:, :prompt_len], n_pages, B, device)
                else:
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
                    if args.engine == "fast":
                        correct = gated_exact_answer_fast(model, template, ids_full[:, :prompt_len],
                                                          answer_ids, keep, B, device)
                    else:
                        dropped = [k for k in range(prompt_len) if page_of(k, B) not in set(keep)]
                        lg = quality_forward(model, ids_full, build_mask(L, prompt_len, dropped, rdtype, device), n_keep, device)
                        correct = exact_answer_ok(lg, answer_ids); del lg
                    rec = {"family": fam, "budget": bf, "policy": policy,
                           "J": len(keep), "P": n_pages, "L": int(L), "correct": bool(correct)}
                    records.append(rec); raw.write(json.dumps(rec) + "\n")
            if args.engine == "fast":
                del template
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
        # Baseline = attention accuracy at budget 1.0 (keep everything). Every accepted instance
        # passes the full-cache reference by construction, so when 1.0 isn't swept the baseline is
        # 1.0 — compute it nan-safely (np.mean([]) is nan, and `nan or 1.0` is nan since bool(nan)
        # is True, which would silently force iso=1.0 / capacity=1.0×).
        full_vals = [r["correct"] for r in fr if abs(r["budget"]-1.0) < 1e-9 and r["policy"]=="attention"]
        if not full_vals:
            print(f"  [warn] {fam}: budget 1.0 not in --budgets; baseline full_acc assumed 1.0 "
                  f"(accepted instances pass the full cache by construction).", flush=True)
        full = float(np.mean(full_vals)) if full_vals else 1.0
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
