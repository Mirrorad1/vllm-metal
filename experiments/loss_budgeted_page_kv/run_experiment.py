# SPDX-License-Identifier: Apache-2.0
"""Run one experiment from the loss-budgeted-page-KV plan and write its outputs.

A single behavioral sweep computes per-step signals (attention_mass, KL-damage,
page_norm) ONCE per decode step from the full cache, then evaluates every policy
× budget against that step — so the oracle, baselines, attention-proxy and online
controller are all measured on identical conditions and identical budgets. Raw
data is written per experiment id; verdicts apply the COMPLETE multi-axis gate
(behavioral error AND, where claimed, physical memory / latency) — logical
compression alone is never counted as success.
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
import error_metrics as EM
import page_policies as PP

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"
B = H.BLOCK_SIZE

BUDGETS = [1.0, 0.5, 0.25, 0.125, 0.0625]
BASELINES = ["recent_pages", "seeded_random_pages", "sink_recent_pages"]
PROXY = ["attention_proxy_pages", "page_norm_pages"]
LOSS = ["loss_budgeted_oracle", "loss_budgeted_online"]
ALL_POLICIES = ["full_pages"] + BASELINES + PROXY + LOSS

_FILLER = [
    "The weather today is mild with a gentle breeze over the quiet hills.",
    "Gardeners often discuss the merits of compost and seasonal rotation.",
    "A distant train sounded its horn as the afternoon light faded slowly.",
    "Many people enjoy a warm cup of tea while reading by the window.",
    "The museum exhibit featured pottery and woven baskets from the coast.",
    "Old maps show trade routes that crossed the wide and dusty plains.",
]


def _filler(rng, n):
    return " ".join(rng.choice(_FILLER) for _ in range(n))


def make_prompt(family, seed, n_filler):
    """Return (prompt, answer). Deterministic. full_pages must solve it (checked)."""
    rng = random.Random(seed)
    pre = _filler(rng, n_filler // 2)
    post = _filler(rng, n_filler - n_filler // 2)
    if family == "needle":
        code = rng.randint(1000, 9999)
        p = f"Remember this. {pre} The secret access code is {code}. {post} The secret access code is"
        return p, f" {code}"
    if family == "multi_needle":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        p = (f"{pre} Box A holds {a}. {_filler(rng, 4)} Box B holds {b}. {post} "
             f"The number in Box A is")
        return p, f" {a}"
    if family == "kv_facts":
        color = rng.choice(["blue", "green", "amber", "violet"])
        p = f"{pre} The assigned color value is {color}. {post} The assigned color value is"
        return p, f" {color}"
    if family == "distractor":
        target = rng.randint(100, 999)
        d1, d2 = rng.randint(100, 999), rng.randint(100, 999)
        p = (f"{pre} A decoy number is {d1}. The official total is {target}. "
             f"Another decoy is {d2}. {post} The official total is")
        return p, f" {target}"
    if family == "contradiction":
        a, b = rng.sample(["Anna", "Diego", "Priya", "Omar"], 2)
        p = (f"{pre} The leader is {a}. {_filler(rng, 4)} Correction: the leader is now {b}. "
             f"{post} The current leader is")
        return p, f" {b}"
    if family == "code_id":
        ident = f"x{rng.randint(1000,9999)}_handler"
        p = f"{pre} def {ident}(): pass {post} The function name is {ident[:2]}"
        return p, ident[2:6]
    raise KeyError(family)


FAMILIES = ["needle", "multi_needle", "kv_facts", "distractor", "contradiction"]


def tokenize(tokenizer, prompt):
    return tokenizer.encode(prompt)


def run_sweep(model, tokenizer, families, n_fillers, seeds, horizon,
              need_damage=True):
    """Yield raw records across families × context × seeds × policies × budgets."""
    records = []
    cfgmeta = H.model_config(model)
    for fam in families:
        for nf in n_fillers:
            for seed in seeds:
                prompt, answer = make_prompt(fam, seed, nf)
                ids = tokenize(tokenizer, prompt)
                answer_ids = tokenizer.encode(prompt + answer)[len(ids):]
                if not answer_ids:
                    answer_ids = tokenizer.encode(answer)
                hz = min(horizon, max(1, len(answer_ids)))

                caches, query, seq = H.prefill(model, ids)
                # validity: full-cache must produce the answer's first token
                z0, _ = H.full_step(model, caches, query, seq, want_mass=False)
                if int(np.argmax(z0)) != int(answer_ids[0]):
                    records.append({"event": "rejected", "family": fam,
                                    "n_filler": nf, "seed": seed})
                    continue

                # collect per-step across horizon
                cur_q = query
                ids_run = list(ids)
                for h in range(hz):
                    n_pages = (seq + B - 1) // B
                    pages = PP.SequencePages(tuple(range(n_pages)), seq, B)
                    z_full, page_mass = H.full_step(model, caches, cur_q, seq, want_mass=True)
                    t_sig = time.perf_counter()
                    pnorm = H.page_norms(caches, seq)
                    dmg = H.page_damage(model, caches, cur_q, seq, z_full) if need_damage else None
                    sig_time = time.perf_counter() - t_sig
                    sig = PP.Signals(attention_mass=list(page_mass),
                                     damage=None if dmg is None else list(dmg),
                                     page_norm=list(pnorm))
                    for policy in ALL_POLICIES:
                        budgets = [1.0] if policy == "full_pages" else BUDGETS
                        for bf in budgets:
                            cfg = PP.PolicyConfig(budget_fraction=bf, recent_window=1,
                                                  sink_pages=1, seed=seed)
                            t0 = time.perf_counter()
                            sel = PP.select(policy, pages, cfg, sig)
                            ctl_s = time.perf_counter() - t0
                            z_g, R = H.gated_step(model, caches, cur_q, seq,
                                                  list(sel.selected_page_indices))
                            records.append({
                                "family": fam, "n_filler": nf, "seed": seed, "step": h,
                                "policy": policy, "budget_fraction": bf,
                                "kl": EM.kl(z_full, z_g), "js": EM.js(z_full, z_g),
                                "top1": EM.top1_agree(z_full, z_g),
                                "topk": EM.topk_overlap(z_full, z_g, 10),
                                "answer_correct": int(np.argmax(z_g)) == int(answer_ids[h]) if h < len(answer_ids) else None,
                                "J": sel.num_selected_pages, "P": n_pages, "R": R,
                                "controller_s": ctl_s,
                                "signal_s": sig_time if policy == "loss_budgeted_oracle" else 0.0,
                            })
                    # teacher-force advance
                    true_tok = answer_ids[h] if h < len(answer_ids) else int(np.argmax(z_full))
                    H.advance(model, caches, cur_q)
                    cur_q = true_tok
                    seq += 1
                    ids_run.append(true_tok)
    return records


def systems_measure(cfgmeta, context_lens=(512, 1024, 2048)):
    rows = []
    n_q, n_kv, d, NL = cfgmeta["n_q"], cfgmeta["n_kv"], cfgmeta["head_dim"], cfgmeta["n_layers"]
    for L in context_lens:
        err = H.paged_all_pages_err(L, B, n_q, n_kv, d)
        npg = (L + B - 1) // B
        fp50, fp95, _ = H.paged_latency(L, list(range(npg)), B, n_q, n_kv, d)
        for bf in BUDGETS:
            J = max(1, round(bf * npg))
            keep = sorted(set(list(range(npg - 1, npg - J, -1)) + [npg - 1]))[:J] or [npg - 1]
            gp50, gp95, R = H.paged_latency(L, keep, B, n_q, n_kv, d)
            rows.append({"context_len": L, "all_pages_max_abs_err": err, "budget_fraction": bf,
                         "J": len(keep), "P": npg, "R": R,
                         "full_p50_ms": fp50, "gated_p50_ms": gp50,
                         "full_p95_ms": fp95, "gated_p95_ms": gp95,
                         "full_bytes": EM.page_bytes(npg, B, NL, n_kv, d),
                         "would_be_gated_bytes": EM.page_bytes(len(keep), B, NL, n_kv, d),
                         "physical_bytes_actual": EM.page_bytes(npg, B, NL, n_kv, d),  # shadow mode
                         "pages_reclaimed": 0})
    return rows


# --- aggregation + verdict ------------------------------------------------

def aggregate(records, policies, budgets):
    by = {}
    for r in records:
        if r.get("event") == "rejected":
            continue
        if r["policy"] not in policies:
            continue
        by.setdefault((r["policy"], r["budget_fraction"]), []).append(r)
    rows = []
    for (pol, bf), rs in sorted(by.items()):
        if bf not in budgets and bf != 1.0:
            continue
        kls = [r["kl"] for r in rs]
        acc = [1.0 if r["answer_correct"] else 0.0 for r in rs if r["answer_correct"] is not None]
        m, ci = EM.mean_ci(kls)
        am, aci = EM.mean_ci(acc)
        rows.append({"policy": pol, "budget_fraction": bf, "n": len(rs),
                     "D_h_mean": m, "D_h_ci95": ci,
                     "answer_acc_mean": am, "answer_acc_ci95": aci,
                     "top1_mean": float(np.mean([r["top1"] for r in rs])),
                     "J_mean": float(np.mean([r["J"] for r in rs])),
                     "controller_ms_mean": float(np.mean([r["controller_s"] for r in rs])) * 1e3})
    return rows


def write_exp(exp_dir: Path, config: dict, records, agg_rows, sys_rows, verdict_text):
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    with open(exp_dir / "raw_results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if agg_rows:
        with open(exp_dir / "aggregate.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader(); w.writerows(agg_rows)
    if sys_rows:
        with open(exp_dir / "systems.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sys_rows[0].keys()))
            w.writeheader(); w.writerows(sys_rows)
    (exp_dir / "verdict.md").write_text(verdict_text)
    return exp_dir
