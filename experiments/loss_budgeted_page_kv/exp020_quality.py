# SPDX-License-Identifier: Apache-2.0
"""exp020 — Budget-vs-quality curve on HARDER long-context tasks (Track B).

The capacity multiplier from exp019 is ≈1/budget, but only at the budget where
quality holds. Prior quality evidence (exp011) used single-needle COPY retrieval —
the maximally-redundant EASY case. This measures the iso-quality budget on
progressively HARDER, less-redundant tasks with the DEPLOYABLE selector (attention
mass + recent; no oracle, no needle labels), to find the realistic safe budget per
task difficulty → the realistic capacity multiplier.

Metric: EXACT-answer accuracy (teacher-forced greedy over the whole answer span must
match), vs the full-cache reference. Reject instances the full cache cannot solve.
Selectors: attention_proxy (deployable), recent (floor), oracle (ceiling, diagnostic).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import page_policies as PP

B = H.BLOCK_SIZE
_RES = Path(__file__).resolve().parent / "results"

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
    """Return (prompt, answer). Progressively harder / less redundant."""
    rng = random.Random(seed)
    pre, post = _filler(rng, n_filler // 2), _filler(rng, n_filler - n_filler // 2)
    if family == "single_needle":      # easy / maximally redundant (baseline)
        code = rng.randint(10000, 99999)
        return f"Remember this. {pre} Cabinet 4731 was issued token {code}. {post} Cabinet 4731 was issued token", f" {code}"
    if family == "exact_long_string":  # exact multi-token recall (less redundant)
        s = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
        return f"Record this exactly. {pre} The verification key is {s}. {post} The verification key is", f" {s}"
    if family == "multi_needle":       # k facts, recall one specific (interference)
        ks = {f"vault {rng.randint(1000,9999)}": rng.randint(10000,99999) for _ in range(4)}
        target = rng.choice(list(ks))
        facts = " ".join(f"The code for {k} is {v}." for k, v in ks.items())
        mid = _filler(rng, 4)
        return f"{pre} {facts} {mid} {post} The code for {target} is", f" {ks[target]}"
    if family == "multi_hop":          # true 2-hop w/ interference (least redundant)
        # 3 rooms each with a DIFFERENT count, placed EARLY (far from the query) so
        # recent-only fails; the model must link person -> their room -> that count.
        rooms = rng.sample(["vault", "archive", "cellar", "loft", "annex"], 3)
        counts = {r: rng.randint(100, 999) for r in rooms}
        person = rng.choice(["Mara", "Tomas", "Yuki", "Devon"])
        target = rng.choice(rooms)
        facts = (f" {person} works in the {target}."
                 + "".join(f" The {r} holds exactly {counts[r]} boxes." for r in rooms))
        return (f"Read carefully.{facts} {pre} {post} "
                f"The number of boxes in the room where {person} works is"), f" {counts[target]}"
    if family == "distractor":         # many same-surface decoys
        target = rng.randint(100, 999)
        decoys = " ".join(f"A decoy total is {rng.randint(100,999)}." for _ in range(4))
        return f"{pre} {decoys} The OFFICIAL total is {target}. {decoys} {post} The OFFICIAL total is", f" {target}"
    if family == "sum_scattered":      # DENSE aggregation: answer depends on EVERY addend page
        amts = [rng.randint(10, 39) for _ in range(SUM_K)]
        facts = [f"Ledger entry {i+1}: a deposit of {a} dollars." for i, a in enumerate(amts)]
        body = _scatter(rng, n_filler, facts)
        return (f"Bookkeeping log. {body} Adding up every deposit listed above, "
                f"the total number of dollars is"), f" {sum(amts)}"
    if family == "recall_all":         # DENSE exact recall (no arithmetic): ALL K codes in order
        codes = [rng.randint(10, 99) for _ in range(RECALL_K)]
        facts = [f"Channel {i+1} broadcasts code {c}." for i, c in enumerate(codes)]
        body = _scatter(rng, n_filler, facts)
        ans = "".join(f" {c}" for c in codes)
        return (f"Signal record. {body} Listing the codes for channels 1 through "
                f"{RECALL_K} in order, they are:"), ans
    raise KeyError(family)


FAMILIES = ["single_needle", "exact_long_string", "multi_needle", "multi_hop", "distractor",
            "sum_scattered", "recall_all"]


def exact_answer(model, full_caches, query, seq, keep_pages, answer_ids):
    """Teacher-forced exact-answer: does the gated cache reproduce the WHOLE answer span?"""
    import mlx.core as mx
    keep_tok = []
    for pg in keep_pages:
        s = pg * B
        keep_tok.extend(range(s, min(s + B, seq)))
    keep_tok = sorted(set(keep_tok))
    km = mx.array(np.asarray(keep_tok, dtype=np.int32))
    g = [H.GatedCache(c.keys[:, :, km, :], c.values[:, :, km, :], seq) for c in full_caches]
    cur = query
    for a in answer_ids:
        z = model(mx.array([[cur]]), cache=g)
        mx.eval(z)
        if int(np.argmax(np.array(z[0, -1].astype(mx.float32)))) != int(a):
            return False
        cur = int(a)  # teacher force
    return True


def run_example(model, tokenizer, prompt, answer, budgets, seed, oracle_pages=None):
    import mlx.core as mx
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    caches, query, seq = H.prefill(model, ids)
    n_pages = (seq + B - 1) // B
    full_caches = [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq) for c in caches]
    full_ok = exact_answer(model, caches, query, seq, list(range(n_pages)), answer_ids)
    if not full_ok:
        return None
    _, page_mass = H.full_step(model, caches, query, seq, want_mass=True)
    pages = PP.SequencePages(tuple(range(n_pages)), seq, B)
    rows = []
    for bf in budgets:
        J = max(1, round(bf * n_pages))
        sigs = {
            "attention": PP.Signals(attention_mass=list(page_mass)),
            "recent": PP.Signals(),
        }
        polmap = {"attention": "attention_proxy_pages", "recent": "recent_pages"}
        rec = {"budget": bf, "J": J, "P": n_pages}
        for name, sig in sigs.items():
            sel = PP.select(polmap[name], pages, PP.PolicyConfig(budget_pages=J, seed=seed), sig)
            rec[f"{name}_correct"] = exact_answer(model, caches, query, seq,
                                                  sorted(sel.selected_page_indices), answer_ids)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-filler", type=int, default=150)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--iso-thresh", type=float, default=0.9, help="acc fraction-of-full to call iso-quality")
    ap.add_argument("--out", default=str(_RES / "exp020_quality"))
    args = ap.parse_args()
    model, tok = H.load_model(args.model)
    tag = args.model.split("/")[-1]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = open(out / f"raw_{tag}.jsonl", "w")
    records = []
    t0 = time.perf_counter()
    for fam in args.families:
        acc = 0; seed = 0
        while acc < args.n and seed < args.n * 6:
            prompt, answer = make_task(fam, seed, args.n_filler); seed += 1
            try:
                rows = run_example(model, tok, prompt, answer, args.budgets, seed)
            except Exception as e:
                raw.write(json.dumps({"event": "error", "fam": fam, "err": repr(e)[:150]}) + "\n"); continue
            if rows is None:
                continue
            acc += 1
            for r in rows:
                r["family"] = fam; records.append(r); raw.write(json.dumps(r) + "\n")
        print(f"{fam}: {acc} valid ({time.perf_counter()-t0:.0f}s)", flush=True)
    raw.close()

    # aggregate: accuracy by (family, budget) for each selector + iso-quality budget
    fams = args.families
    summary = {"model": args.model, "iso_thresh": args.iso_thresh, "by_family": {}}
    print(f"\n{'family':18s} {'budget':>7} {'attn_acc':>9} {'recent_acc':>11}")
    for fam in fams:
        fr = [r for r in records if r["family"] == fam]
        # nan-safe baseline: np.mean([]) is nan and `nan or 1.0` is nan (bool(nan) is True), which
        # would silently force iso=1.0. Accepted instances pass the full cache by construction, so a
        # missing 1.0 budget means baseline 1.0.
        full_vals = [r["attention_correct"] for r in fr if abs(r["budget"]-1.0) < 1e-9]
        full_acc = float(np.mean(full_vals)) if full_vals else 1.0
        per_b = {}
        for bf in args.budgets:
            sub = [r for r in fr if abs(r["budget"] - bf) < 1e-9]
            if not sub:
                continue
            a = float(np.mean([r["attention_correct"] for r in sub]))
            rc = float(np.mean([r["recent_correct"] for r in sub]))
            per_b[bf] = {"attention_acc": a, "recent_acc": rc, "n": len(sub)}
            print(f"{fam:18s} {bf:>7.4f} {a:>9.2f} {rc:>11.2f}")
        # iso-quality budget = smallest budget reachable by CONTIGUOUS passing from
        # the top (the tightest budget that still holds iso-quality without a break).
        iso = 1.0
        for bf in sorted(args.budgets, reverse=True):
            if per_b.get(bf, {}).get("attention_acc", 0.0) >= args.iso_thresh * max(full_acc, 1e-9):
                iso = bf
            else:
                break
        summary["by_family"][fam] = {"full_acc": full_acc, "iso_quality_budget": iso,
                                     "capacity_multiplier_at_iso": round(1.0 / iso, 2), "per_budget": per_b}
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== iso-quality budget (deployable attention selector) & capacity multiplier ===")
    for fam, s in summary["by_family"].items():
        print(f"  {fam:18s} iso-budget={s['iso_quality_budget']:.4f} → {s['capacity_multiplier_at_iso']}× capacity")
    mults = [s["capacity_multiplier_at_iso"] for s in summary["by_family"].values()]
    print(f"\nrealistic capacity multiplier across task difficulty: "
          f"{min(mults):.1f}×–{max(mults):.1f}× (worst-task {min(mults):.1f}×)")
    return summary


if __name__ == "__main__":
    main()
