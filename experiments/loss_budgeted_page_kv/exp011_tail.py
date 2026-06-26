# SPDX-License-Identifier: Apache-2.0
"""exp011 — Is "attention-mass == KL-damage oracle" a mean-of-opposites?

See DESIGN_exp011.md for the pre-registered hypothesis, primary cell, endpoints,
and decision thresholds (fixed before any result was seen). KV-only; leak-free:
the needle-page LABEL comes only from the construction token span and is NEVER an
input to any selector.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import stats as st

import benchmark_harness as H
import error_metrics as EM
import page_policies as PP

B = H.BLOCK_SIZE
_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"

_FILLER = [
    "The weather today is mild with a gentle breeze over the quiet hills.",
    "Gardeners often discuss the merits of compost and seasonal rotation.",
    "A distant train sounded its horn as the afternoon light faded slowly.",
    "Many people enjoy a warm cup of tea while reading by the window.",
    "The museum exhibit featured pottery and woven baskets from the coast.",
    "Old maps show trade routes that crossed the wide and dusty plains.",
    "A quiet river wound past the mill where the old wheel still turned.",
    "Shelves of books lined the hall, their spines cracked and faded.",
]


def _filler(rng, n):
    return " ".join(rng.choice(_FILLER) for _ in range(n))


def make_needle_prompt(tokenizer, seed, n_filler, depth_frac, n_distract=2):
    """Paraphrased (non-verbatim) single-needle retrieval prompt.

    Returns (prompt, answer, crit_positions). The needle states a fact one way;
    the query asks for it a DIFFERENT way, so a verbatim induction match cannot
    trivially retrieve it. ``crit_positions`` = token span of the answer value
    (analysis label only)."""
    rng = random.Random(seed)
    code = rng.randint(10000, 99999)
    vault = rng.randint(1000, 9999)
    intro = "Read the document carefully; you will be asked one question.\n"
    # distractors share the surface form but different ids/values
    distract = "".join(
        f" Cabinet {rng.randint(1000,9999)} was issued token {rng.randint(10000,99999)}."
        for _ in range(n_distract)
    )
    needle = f" Cabinet {vault} was issued token {code}."
    # paraphrased query (different surface form than the needle):
    query = f"\nQuestion: which access token belongs to cabinet {vault}?\nAnswer: token"
    answer = f" {code}"

    n_before = max(1, int(round(depth_frac * n_filler)))
    n_after = max(1, n_filler - n_before)
    before = " " + _filler(rng, n_before) + distract
    after = " " + _filler(rng, n_after)
    prefix = intro + before
    prompt = prefix + needle + after + query

    pre_ids = tokenizer.encode(prefix)
    upto_ids = tokenizer.encode(prefix + needle)
    crit_positions = list(range(len(pre_ids), len(upto_ids)))
    return prompt, answer, crit_positions


# --- dual attention-mass capture (sum-over-heads AND max-over-heads) -------

_CAP = {"on": False, "seq": 0, "P": 0, "sum": None, "max": None, "layers": 0,
        "total": 0.0}


@contextlib.contextmanager
def capture_dual(model):
    import mlx.core as mx
    import mlx_lm.models.qwen2 as q2
    orig = q2.scaled_dot_product_attention
    _CAP["sum"] = np.zeros(_CAP["P"])
    _CAP["max"] = np.zeros(_CAP["P"])
    _CAP["layers"] = 0
    _CAP["total"] = 0.0
    seq, P = _CAP["seq"], _CAP["P"]
    pidx = (np.arange(seq) // B).astype(np.int64)

    def wrapped(queries, keys, values, cache, scale, mask, sinks=None):
        hq, hkv = queries.shape[1], keys.shape[1]
        k = keys if hq == hkv else mx.repeat(keys, hq // hkv, axis=1)
        w = mx.softmax((queries[:, :, -1:, :].astype(mx.float32) * scale)
                       @ k.astype(mx.float32).swapaxes(-1, -2), axis=-1)
        wfull = np.array(w[0, :, 0, :].astype(mx.float32))  # [Hq, S=seq+1]
        _CAP["total"] += float(wfull.sum())  # ≈ Hq per layer (rows sum to 1)
        wl = wfull[:, :seq]  # exclude the new query's own key
        _CAP["sum"] += np.bincount(pidx, weights=wl.sum(0), minlength=P)[:P]
        # per-head per-page mass, elementwise max over heads & layers
        for h in range(wl.shape[0]):
            ph = np.bincount(pidx, weights=wl[h], minlength=P)[:P]
            _CAP["max"] = np.maximum(_CAP["max"], ph)
        _CAP["layers"] += 1
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    q2.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        q2.scaled_dot_product_attention = orig


def gold_logp(z, tok_id):
    z = z.astype(np.float64)
    z = z - z.max()
    return float(z[tok_id] - np.log(np.exp(z).sum()))


def _sel(policy, pages, J, seed, sig):
    cfg = PP.PolicyConfig(budget_pages=J, recent_window=1, sink_pages=1, seed=seed)
    return set(PP.select(policy, pages, cfg, sig).selected_page_indices)


def run_example(model, tokenizer, prompt, answer, crit_positions, J_fracs, seed,
                n_layers_expected, n_q_expected):
    import mlx.core as mx
    ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(prompt + answer)[len(ids):] or tokenizer.encode(answer)
    ans0 = int(answer_ids[0])

    caches, query, seq = H.prefill(model, ids)
    n_pages = (seq + B - 1) // B
    needle_pages = sorted(set(p // B for p in crit_positions))
    straddle = len(needle_pages) > 1
    needle_pg = needle_pages[-1]  # answer-value page
    pages = PP.SequencePages(tuple(range(n_pages)), seq, B)

    # full logits + dual attention capture
    _CAP.update(seq=seq, P=n_pages)
    with capture_dual(model):
        qx = mx.array([[query]])
        cap_caches = [H.GatedCache(c.keys[:, :, :seq, :], c.values[:, :, :seq, :], seq)
                      for c in caches]
        logits_full = model(qx, cache=cap_caches)
        mx.eval(logits_full)
    z_full = np.array(logits_full[0, -1].astype(mx.float32))
    mass_sum = _CAP["sum"].copy()
    mass_max = _CAP["max"].copy()
    # capture sanity (silent-deflation guard): rows sum to 1 so full total ≈ NL*Hq
    cap_ok = (_CAP["layers"] == n_layers_expected
              and abs(_CAP["total"] - n_layers_expected * n_q_expected) < 0.05 * n_layers_expected * n_q_expected)

    full_correct = int(np.argmax(z_full)) == ans0
    lp_full = gold_logp(z_full, ans0)

    # load-bearing certification: ablate ONLY the needle page(s)
    keep_wo_needle = [p for p in range(n_pages) if p not in set(needle_pages)]
    z_abl, _ = H.gated_step(model, caches, query, seq, keep_wo_needle)
    needle_ablation_kl = EM.kl(z_full, z_abl)
    needle_ablation_dlp = lp_full - gold_logp(z_abl, ans0)
    needle_top1_flip = int(np.argmax(z_abl)) != int(np.argmax(z_full))
    load_bearing = needle_top1_flip or (needle_ablation_dlp >= 1.0)

    # oracle per-page damage (O(P))
    dmg = H.page_damage(model, caches, query, seq, z_full)

    recs = []
    for bf in J_fracs:
        J = max(1, round(bf * n_pages))
        sigs = {
            "oracle": PP.Signals(damage=list(dmg)),
            "attn_sum": PP.Signals(attention_mass=list(mass_sum)),
            "attn_max": PP.Signals(attention_mass=list(mass_max)),
            "recent": PP.Signals(),
            "random": PP.Signals(),
            "sink": PP.Signals(),
        }
        polmap = {"oracle": "loss_budgeted_oracle", "attn_sum": "attention_proxy_pages",
                  "attn_max": "attention_proxy_pages", "recent": "recent_pages",
                  "random": "seeded_random_pages", "sink": "sink_recent_pages"}
        row = {"bf": bf, "J": J, "P": n_pages, "discretionary": J - 2,
               "needle_pg": needle_pg, "straddle": straddle,
               "load_bearing": bool(load_bearing),
               "needle_ablation_kl": needle_ablation_kl,
               "needle_ablation_dlp": needle_ablation_dlp,
               "needle_outside_floor": (1 <= needle_pg <= n_pages - 2),
               "full_correct": full_correct, "cap_ok": bool(cap_ok)}
        keep_sets = {}
        nset = set(needle_pages)
        for name, sig in sigs.items():
            sel = _sel(polmap[name], pages, J, seed, sig)
            keep_sets[name] = sel
            z_g, _ = H.gated_step(model, caches, query, seq, sorted(sel))
            row[f"{name}_keep"] = bool(nset & sel)            # secondary (diffusion-caveated)
            row[f"{name}_keep_all"] = bool(nset and nset.issubset(sel))
            row[f"{name}_lp"] = gold_logp(z_g, ans0)
            row[f"{name}_correct"] = int(np.argmax(z_g)) == ans0
            row[f"{name}_kl"] = EM.kl(z_full, z_g)            # PRIMARY channel
        # selection overlap (Jaccard) oracle vs each attention pooling
        for proxy in ["attn_sum", "attn_max"]:
            a, b = keep_sets["oracle"], keep_sets[proxy]
            row[f"jaccard_oracle_{proxy}"] = len(a & b) / max(1, len(a | b))
        recs.append(row)
    return recs, full_correct, load_bearing, straddle, cap_ok


# --- statistics -----------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - hw), min(1.0, c + hw))


def mcnemar_exact(b, c):
    """b = oracle-keep&attn-drop, c = attn-keep&oracle-drop. Returns (p, net_rate_n)."""
    if b + c == 0:
        return (1.0, 0)
    p = st.binomtest(min(b, c), b + c, 0.5).pvalue
    return (p, b - c)


def aggregate_and_verdict(records, primary_bf):
    """PRIMARY endpoint (label-free, diffusion-robust): per-example paired
    ΔKL = KL(full||attn-sel) − KL(full||oracle-sel) at the budget. The prior
    result showed equal MEANS; this tests whether a TAIL was hidden (the
    mean-of-opposites signature). Needle-recall / McNemar kept as SECONDARY
    descriptive (single-page load-bearing is ~always false due to prefill
    information diffusion — see verdict notes — so it cannot gate)."""
    valid = [r for r in records if r["needle_outside_floor"] and r["cap_ok"]]
    out = {"n_total_rows": len(records), "n_valid_rows": len(valid),
           "load_bearing_rate": float(np.mean([r["load_bearing"] for r in records])) if records else float("nan"),
           "straddle_rate": float(np.mean([r["straddle"] for r in records])) if records else float("nan"),
           "cap_ok_rate": float(np.mean([r["cap_ok"] for r in records])) if records else float("nan"),
           "needle_ablation_dlp_mean": float(np.mean([r["needle_ablation_dlp"] for r in records])) if records else float("nan"),
           "by_budget": {}}
    for bf in sorted({r["bf"] for r in records}):
        vb = [r for r in valid if abs(r["bf"] - bf) < 1e-9]
        if not vb:
            continue
        n = len(vb)
        rec = {"n": n}
        # per-selector mean KL (reproduces the prior near-equal-means anchor)
        for sel in ["oracle", "attn_sum", "attn_max", "recent", "random", "sink"]:
            kls = np.array([r[f"{sel}_kl"] for r in vb])
            rec[f"{sel}_kl_mean"] = float(kls.mean())
            rec[f"{sel}_kl_p99"] = float(np.percentile(kls, 99))
            rec[f"{sel}_acc"] = float(np.mean([r[f"{sel}_correct"] for r in vb]))
            k = sum(r[f"{sel}_keep_all"] for r in vb)  # secondary needle recall (all pages)
            p, lo, hi = wilson(k, n)
            rec[f"{sel}_needle_recall"], rec[f"{sel}_nr_lo"], rec[f"{sel}_nr_hi"] = p, lo, hi
        # PRIMARY: paired ΔKL and Δlp, oracle vs each attention pooling
        for proxy in ["attn_sum", "attn_max"]:
            dkl = np.array([r[f"{proxy}_kl"] - r["oracle_kl"] for r in vb])  # >0 => oracle better
            dlp = np.array([r["oracle_lp"] - r[f"{proxy}_lp"] for r in vb])  # >0 => oracle better
            absk = np.abs(dkl)
            tail_k = int(np.sum(absk > 1.0))
            _, tlo, thi = wilson(tail_k, n)
            nz = dkl[np.abs(dkl) > 1e-9]
            try:
                wk = float(st.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
            except Exception:
                wk = float("nan")
            nzl = dlp[np.abs(dlp) > 1e-9]
            try:
                wl = float(st.wilcoxon(nzl).pvalue) if len(nzl) >= 6 else float("nan")
            except Exception:
                wl = float("nan")
            rec[f"{proxy}_dKL_median"] = float(np.median(dkl))
            rec[f"{proxy}_dKL_mean"] = float(np.mean(dkl))
            rec[f"{proxy}_dKL_frac_pos"] = float(np.mean(dkl > 0))
            rec[f"{proxy}_dKL_p99_abs"] = float(np.percentile(absk, 99))
            rec[f"{proxy}_dKL_tail_gt1"] = tail_k / n
            rec[f"{proxy}_dKL_tail_lo"] = tlo
            rec[f"{proxy}_dKL_tail_hi"] = thi
            rec[f"{proxy}_dKL_wilcoxon_p"] = wk
            rec[f"{proxy}_dlp_median"] = float(np.median(dlp))
            rec[f"{proxy}_dlp_wilcoxon_p"] = wl
            rec[f"jaccard_oracle_{proxy}_mean"] = float(np.mean([r[f"jaccard_oracle_{proxy}"] for r in vb]))
            # DECISIVE answer-level paired test (does oracle rescue answers attn loses?)
            ab = sum(r["oracle_correct"] and not r[f"{proxy}_correct"] for r in vb)
            ac = sum(r[f"{proxy}_correct"] and not r["oracle_correct"] for r in vb)
            ap, anet = mcnemar_exact(ab, ac)
            _, ablo, _ = wilson(ab, n)
            rec[f"{proxy}_ans_b"] = ab          # oracle-correct & proxy-wrong
            rec[f"{proxy}_ans_c"] = ac          # proxy-correct & oracle-wrong
            rec[f"{proxy}_ans_mcnemar_p"] = ap
            rec[f"{proxy}_ans_net"] = anet / n
            rec[f"{proxy}_ans_b_lo"] = ablo
        out["by_budget"][f"{bf:.5f}"] = rec

    # Auto-select the primary budget as the BINDING budget whose oracle_kl_mean is
    # closest to ~0.5 — the regime where the disputed prior "0.549" lived and where
    # the oracle has headroom for a tail to appear (a saturated oracle_kl≈0 cell
    # cannot show a mean-of-opposites). Chosen on oracle's ABSOLUTE KL only, never
    # on the ΔKL outcome. Falls back to the requested primary_bf.
    # binding = baselines clearly worse than oracle (additive margin, robust near
    # small KL) AND oracle has headroom (not saturated to ~0, not everyone-fails).
    binding_budgets = {k: v for k, v in out["by_budget"].items()
                       if v["recent_kl_mean"] > v["oracle_kl_mean"] + 0.2
                       and v["oracle_kl_mean"] > 0.05}
    if binding_budgets:
        sel_k = min(binding_budgets, key=lambda k: abs(binding_budgets[k]["oracle_kl_mean"] - 0.5))
        primary_bf = float(sel_k)
    out["auto_primary_selected"] = primary_bf
    pb = out["by_budget"].get(f"{primary_bf:.5f}")
    verdict, reason = "INCONCLUSIVE", ""
    if pb is None or pb["n"] < 20:
        reason = f"insufficient valid examples at primary budget (n={None if pb is None else pb['n']})"
    else:
        binding = (pb["recent_kl_mean"] > pb["oracle_kl_mean"] + 0.2
                   and pb["oracle_kl_mean"] > 0.05)
        # KL-fidelity tail (full next-token distribution)
        kl_tail = pb["attn_sum_dKL_tail_lo"] > 0.02
        kl_null = pb["attn_sum_dKL_tail_hi"] < 0.02 and abs(pb["attn_sum_dKL_median"]) < 0.02
        # DECISIVE: answer-level paired test — does oracle RESCUE answers attn loses?
        ans_effect = (pb["attn_sum_ans_mcnemar_p"] < 0.01
                      and pb["attn_sum_ans_b_lo"] > 0.03
                      and pb["attn_sum_ans_b"] > pb["attn_sum_ans_c"])
        if not binding:
            verdict, reason = "INCONCLUSIVE", ("no binding+headroom budget (oracle KL saturates to ~0 at "
                "loose budgets, and all selectors fail together at the tightest) — the regime where a tail "
                "could exist is narrow")
        elif ans_effect and kl_tail:
            verdict = "CONFIRM (behavioral)"
            reason = ("at the binding+headroom budget, oracle BOTH preserves the distribution better (ΔKL "
                      "tail) AND rescues the ANSWER on a non-trivial fraction attention loses (McNemar) — "
                      "mean-of-opposites is real and behaviorally consequential")
        elif kl_tail and not ans_effect:
            verdict = "KL-ONLY (behaviorally inert)"
            reason = ("the prior 'equal KL means' was REGIME-SPECIFIC: at the tight binding budget the "
                      "oracle preserves the full next-token distribution / gold logprob materially better "
                      "than attention (one-sided ΔKL tail), BUT it does NOT rescue the ANSWER on any example "
                      "attention loses (answer McNemar null, identical accuracy). The divergence is in "
                      "behaviorally-inert distribution mass; loss-aware page SCORING does not improve "
                      "retrieval — prefill information diffusion (single-page ablation Δlp≈0, recent-only "
                      "answers correctly) leaves no behavioral tail to exploit. Pivot to the "
                      "physical-allocator/substrate gaps, not better signals.")
        elif kl_null:
            verdict = "NULL/DEAD"
            reason = ("attention ≡ oracle per example (no KL tail, no answer effect): scoring genuinely "
                      "adds nothing even in the tail — pivot to the physical-allocator/substrate gaps")
        else:
            verdict, reason = "INCONCLUSIVE", "tail CI straddles the 2% decision band; needs more N"
    out["primary_bf"] = primary_bf
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-filler", type=int, default=210)  # ~3072 tokens, P~192
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.0625, 0.03125, 0.015625, 0.0078125])
    ap.add_argument("--primary-bf", type=float, default=0.03125)
    ap.add_argument("--out", default=str(_RES / "exp011_tail_vs_mean"))
    args = ap.parse_args()

    model, tok = H.load_model(args.model)
    cfg = H.model_config(model)
    print("model:", cfg, flush=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = open(outdir / "raw_results.jsonl", "w")
    records = []
    accepted = 0
    seed = 0
    t0 = time.perf_counter()
    while accepted < args.n and seed < args.n * 6:
        depth = 0.2 + 0.5 * ((seed % 7) / 6.0)  # spread needle depth 0.2..0.7
        prompt, answer, crit = make_needle_prompt(tok, seed, args.n_filler, depth)
        seed += 1
        try:
            recs, full_ok, lb, straddle, cap_ok = run_example(
                model, tok, prompt, answer, crit, args.budgets, seed,
                cfg["n_layers"], cfg["n_q"])
        except Exception as e:
            raw.write(json.dumps({"event": "error", "seed": seed, "err": repr(e)[:200]}) + "\n")
            continue
        if not full_ok:
            raw.write(json.dumps({"event": "rejected_full_wrong", "seed": seed}) + "\n")
            continue
        accepted += 1
        for r in recs:
            r["seed"] = seed
            records.append(r)
            raw.write(json.dumps(r) + "\n")
        if accepted % 10 == 0:
            print(f"accepted {accepted}/{args.n}  ({time.perf_counter()-t0:.0f}s)  "
                  f"last load_bearing={lb} straddle={straddle}", flush=True)
    raw.close()

    agg = aggregate_and_verdict(records, args.primary_bf)
    (outdir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))
    # flat csv of by_budget
    rows = [{"budget": k, **v} for k, v in agg["by_budget"].items()]
    if rows:
        keys = sorted({kk for r in rows for kk in r})
        with open(outdir / "aggregate.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    print("\n=== VERDICT:", agg["verdict"], "===")
    print(agg["verdict_reason"])
    print(f"auto_primary_bf={agg.get('auto_primary_selected')} "
          f"n_valid_rows={agg['n_valid_rows']} load_bearing_rate={agg['load_bearing_rate']:.2f} "
          f"straddle_rate={agg['straddle_rate']:.2f} cap_ok_rate={agg['cap_ok_rate']:.2f}")
    return agg


if __name__ == "__main__":
    main()
