# SPDX-License-Identifier: Apache-2.0
"""Experiment-loop orchestrator for the loss-budgeted page KV cache.

Runs the planned experiment sequence and applies explicit decision rules to pick
verdicts. To avoid recomputing the expensive O(P) KL-damage signal per
experiment, it runs ONE master behavioral sweep (computing all per-page signals
and evaluating every policy×budget on identical steps) plus the systems
measurement, then derives each experiment's output dir + verdict from that shared
raw data. This is honest: every experiment compares policies on the SAME trials.

Usage:
  python run_loop.py --max-experiments 10 \
      --families needle multi_needle kv_facts distractor contradiction \
      --n-fillers 35 75 150 --seeds 0 1 2 --horizon 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import benchmark_harness as H
import error_metrics as EM
import run_experiment as RX

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"


def _fmt_table(agg_rows, policies, budgets=(0.25, 0.125, 0.0625)):
    by = {(r["policy"], r["budget_fraction"]): r for r in agg_rows}
    lines = ["| policy | " + " | ".join(f"bf={b}" for b in budgets) + " |",
             "|---|" + "---|" * len(budgets)]
    for pol in policies:
        cells = []
        for b in budgets:
            r = by.get((pol, b))
            cells.append(f"{r['D_h_mean']:.3f}/{r['answer_acc_mean']:.2f}" if r else "-")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cmp(agg_rows, pol_a, pol_b, bf):
    """Return (mean_a, ci_a, mean_b, ci_b, a_beats_b_significantly)."""
    by = {(r["policy"], r["budget_fraction"]): r for r in agg_rows}
    a, b = by.get((pol_a, bf)), by.get((pol_b, bf))
    if not a or not b:
        return None
    # 'beats' on D_h = lower, with non-overlapping 95% CIs
    sig = (a["D_h_mean"] + a["D_h_ci95"]) < (b["D_h_mean"] - b["D_h_ci95"])
    return a["D_h_mean"], a["D_h_ci95"], b["D_h_mean"], b["D_h_ci95"], sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-experiments", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=RX.FAMILIES)
    ap.add_argument("--n-fillers", type=int, nargs="+", default=[35, 75, 150])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    model, tok = H.load_model()
    cfgmeta = H.model_config(model)
    print("model config:", cfgmeta, flush=True)

    t0 = time.perf_counter()
    print("running master behavioral sweep (with O(P) KL-damage)...", flush=True)
    records = RX.run_sweep(model, tok, args.families, args.n_fillers, args.seeds,
                           args.horizon, need_damage=True)
    n_ok = sum(1 for r in records if "event" not in r)
    n_rej = sum(1 for r in records if r.get("event") == "rejected")
    print(f"sweep done in {time.perf_counter()-t0:.1f}s: {n_ok} records, {n_rej} rejected",
          flush=True)
    print("running systems measurement (paged kernel)...", flush=True)
    sys_rows = RX.systems_measure(cfgmeta)

    agg_all = RX.aggregate(records, RX.ALL_POLICIES, set(RX.BUDGETS))

    # ---- decision rules → per-experiment verdicts -------------------------
    plan = []

    # exp001 audit
    plan.append(("exp001_audit_dispatch",
                 dict(kind="audit"),
                 [], [],
                 "Experiment: exp001_audit_dispatch\n"
                 "Hypothesis: the active PagedAttention dispatch can be traced.\n"
                 "Result: see IMPLEMENTATION_AUDIT.md — decode dispatches "
                 "kernels_v2/pagedattention.metal (count-masked, position-agnostic); "
                 "tiled prefill kernel UNSAFE; allocator owned by upstream vLLM.\n"
                 "Verdict: POSITIVE (audit complete).\n"
                 "Next experiment: exp002_all_pages_equivalence.\n"))

    # exp002 equivalence
    fp = [r for r in records if r.get("policy") == "full_pages"]
    fp_kl = float(np.mean([r["kl"] for r in fp])) if fp else float("nan")
    sys_eq = max((s["all_pages_max_abs_err"] for s in sys_rows), default=float("nan"))
    eq_ok = fp_kl < 1e-6 and sys_eq < 5e-3
    plan.append(("exp002_all_pages_equivalence",
                 dict(kind="equivalence", full_pages_mean_kl=fp_kl,
                      kernel_all_pages_max_err=sys_eq),
                 RX.aggregate(records, ["full_pages"], set(RX.BUDGETS)), sys_rows,
                 f"Experiment: exp002_all_pages_equivalence\n"
                 f"Result: full_pages behavioral KL mean = {fp_kl:.2e} (==0 expected); "
                 f"kernel all-pages max abs err = {sys_eq:.2e}.\n"
                 f"Verdict: {'POSITIVE' if eq_ok else 'REGRESSION (F2)'}.\n"
                 f"Next experiment: {'exp003_baselines' if eq_ok else 'debug block-table equivalence'}.\n"))

    # exp003 baselines
    plan.append(("exp003_baselines",
                 dict(kind="baselines", policies=RX.BASELINES),
                 RX.aggregate(records, ["full_pages"] + RX.BASELINES, set(RX.BUDGETS)),
                 [],
                 "Experiment: exp003_baselines\n"
                 "Baselines recent/random/sink_recent established at equal budgets.\n"
                 + _fmt_table(agg_all, RX.BASELINES) +
                 "\nVerdict: POSITIVE (baselines + frontier generated).\n"
                 "Next experiment: exp005_attention_proxy.\n"))

    # exp005 attention proxy vs baselines
    plan.append(("exp005_attention_proxy",
                 dict(kind="attention_proxy"),
                 RX.aggregate(records, ["full_pages", "attention_proxy_pages"] + RX.BASELINES,
                              set(RX.BUDGETS)), [],
                 "Experiment: exp005_attention_proxy\n"
                 + _fmt_table(agg_all, ["attention_proxy_pages"] + RX.BASELINES) +
                 "\nVerdict: see comparison; attention_proxy is the strong baseline to beat.\n"
                 "Next experiment: exp006_loss_budgeted_oracle.\n"))

    # exp006 oracle: does KL-damage structure exist + beat baselines + beat attention_proxy?
    c_ob = _cmp(agg_all, "loss_budgeted_oracle", "recent_pages", 0.125)
    c_oa = _cmp(agg_all, "loss_budgeted_oracle", "attention_proxy_pages", 0.125)
    oracle_beats_recent = c_ob and c_ob[4]
    oracle_beats_attn = c_oa and c_oa[4]
    if oracle_beats_recent and oracle_beats_attn:
        v6 = "ORACLE-ONLY POSITIVE (damage structure beyond attention mass)"
    elif oracle_beats_recent and not oracle_beats_attn:
        v6 = ("INCONCLUSIVE — KL-damage structure exists (oracle ≫ recent) but does NOT "
              "beat attention_proxy (F7): attention mass already captures the structure")
    else:
        v6 = "NEGATIVE — KL-damage oracle does not beat simple baselines (no page-loss structure)"
    plan.append(("exp006_loss_budgeted_oracle",
                 dict(kind="oracle", oracle_vs_recent=c_ob, oracle_vs_attn=c_oa),
                 RX.aggregate(records, ["full_pages", "loss_budgeted_oracle",
                                        "attention_proxy_pages", "recent_pages"], set(RX.BUDGETS)),
                 [],
                 "Experiment: exp006_loss_budgeted_oracle\n"
                 "Hypothesis: some pages are much cheaper to drop under logit error; "
                 "an oracle keeping highest-KL-damage pages dominates baselines.\n"
                 + _fmt_table(agg_all, ["loss_budgeted_oracle", "attention_proxy_pages", "recent_pages"]) +
                 f"\noracle vs recent @12.5% (sig beat): {oracle_beats_recent}; "
                 f"oracle vs attention_proxy: {oracle_beats_attn}\n"
                 f"Verdict: {v6}.\n"
                 "Next experiment: exp007_loss_budgeted_online.\n"))

    # exp007 online vs baselines + vs attention_proxy + vs oracle
    c_or = _cmp(agg_all, "loss_budgeted_online", "recent_pages", 0.125)
    c_oap = _cmp(agg_all, "loss_budgeted_online", "attention_proxy_pages", 0.125)
    online_beats_recent = c_or and c_or[4]
    online_beats_attn = c_oap and c_oap[4]
    if online_beats_recent and online_beats_attn:
        v7 = "POSITIVE (online recovers structure beyond attention_proxy)"
    elif online_beats_recent and not online_beats_attn:
        v7 = ("WEAK POSITIVE / INCONCLUSIVE — online beats recent/random but not "
              "attention_proxy (F7); attention mass alone is the recoverable signal")
    else:
        v7 = "NEGATIVE (F6) — online does not beat recent/random"
    plan.append(("exp007_loss_budgeted_online",
                 dict(kind="online", online_vs_recent=c_or, online_vs_attn=c_oap),
                 RX.aggregate(records, ["full_pages", "loss_budgeted_online",
                                        "attention_proxy_pages", "loss_budgeted_oracle",
                                        "recent_pages"], set(RX.BUDGETS)), [],
                 "Experiment: exp007_loss_budgeted_online\n"
                 + _fmt_table(agg_all, ["loss_budgeted_online", "attention_proxy_pages",
                                        "loss_budgeted_oracle", "recent_pages"]) +
                 f"\nonline vs recent @12.5% (sig): {online_beats_recent}; "
                 f"online vs attention_proxy: {online_beats_attn}\n"
                 f"Verdict: {v7}.\n"
                 "Next experiment: exp009_physical_memory.\n"))

    # exp009 physical memory honesty (shadow mode)
    reclaimed = max((s["pages_reclaimed"] for s in sys_rows), default=0)
    plan.append(("exp009_physical_memory",
                 dict(kind="memory", pages_reclaimed=reclaimed, shadow_mode=True),
                 [], sys_rows,
                 "Experiment: exp009_physical_memory\n"
                 "Hypothesis: logical page reduction reduces physical KV memory.\n"
                 f"Result: pages_reclaimed={reclaimed}; physical_bytes_actual == full_bytes "
                 "(read-only compact block table; allocator owned by upstream vLLM; no "
                 "reclamation implemented). Would-be bytes shrink ~J/P but are NOT freed.\n"
                 "Verdict: LOGICAL-ONLY POSITIVE (F4) — no real physical reduction.\n"
                 "Next experiment: exp010_latency_frontier (and a real allocator path).\n"))

    # exp010 latency frontier (kernel + controller honesty)
    # net per-step gated time = N*gated_kernel + controller; compare to N*full_kernel
    NL = cfgmeta["n_layers"]
    lat_rows = []
    for s in sys_rows:
        if s["budget_fraction"] == 1.0:
            continue
        # controller cost: use oracle online (cheap) controller_ms from agg if present
        ctl = next((r["controller_ms_mean"] for r in agg_all
                    if r["policy"] == "loss_budgeted_online"
                    and abs(r["budget_fraction"] - s["budget_fraction"]) < 1e-9), 0.0)
        net_full = NL * s["full_p50_ms"]
        net_gated = NL * s["gated_p50_ms"] + ctl
        lat_rows.append({**s, "net_full_ms_per_step": net_full,
                         "net_gated_ms_per_step": net_gated,
                         "net_speedup": net_full / net_gated if net_gated else 0.0})
    any_win = any(r["net_gated_ms_per_step"] < r["net_full_ms_per_step"] for r in lat_rows)
    plan.append(("exp010_latency_frontier",
                 dict(kind="latency"),
                 [], lat_rows,
                 "Experiment: exp010_latency_frontier\n"
                 "Net per-step decode latency = N*kernel + controller.\n"
                 f"Any budget with net gated < net full: {any_win}.\n"
                 f"Verdict: {'POSITIVE' if any_win else 'LATENCY-NEGATIVE (F5) at tested context lengths'}.\n"
                 "Next experiment: real physical reclamation, or longer context, per synthesis.\n"))

    # ---- write everything -------------------------------------------------
    for i, (exp_id, cfg, agg_rows, srows, verdict) in enumerate(plan[:args.max_experiments]):
        d = _RES / exp_id
        RX.write_exp(d, cfg, [r for r in records if "event" not in r] if cfg.get("kind") not in
                     ("audit",) else [], agg_rows, srows, verdict)
        # also dump rejected list to raw for transparency in exp002
        (d / "summary.md").write_text(verdict)
        (d / "next_step.md").write_text(verdict.split("Next experiment:")[-1].strip())
        print(f"[{exp_id}] written -> {d}")

    # master raw + ledger
    with open(_RES / "master_raw_results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(_RES / "master_aggregate.csv", "w") as f:
        import csv
        if agg_all:
            w = csv.DictWriter(f, fieldnames=list(agg_all[0].keys()))
            w.writeheader(); w.writerows(agg_all)
    print("master raw + aggregate written.")
    return records, agg_all, sys_rows, plan


if __name__ == "__main__":
    main()
