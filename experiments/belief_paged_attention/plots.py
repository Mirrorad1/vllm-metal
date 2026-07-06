# SPDX-License-Identifier: Apache-2.0
"""Render the three frontier plots from the sweep results.

  * plots/memory_error_frontier.png   — would-be cache fraction vs D_h, per policy
  * plots/latency_error_frontier.png  — gated kernel latency vs D_h (systems × behavioral)
  * plots/context_scaling.png         — D_h vs context length at a fixed budget

Reads results/raw_results.jsonl and results/systems.csv. Degrades gracefully if
a file is missing.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"
_PLOTS = _HERE / "plots"
_PLOTS.mkdir(exist_ok=True)

POLICIES = ["full_pages", "recent_pages", "seeded_random_pages",
            "attention_proxy_pages", "keyword_entity_pages",
            "oracle_belief_pages", "inferred_belief_pages"]
MARK = {p: m for p, m in zip(POLICIES, ["o", "s", "x", "^", "D", "*", "P"])}


def load_raw():
    rows = []
    p = _RES / "raw_results.jsonl"
    if not p.exists():
        return rows
    for line in open(p):
        r = json.loads(line)
        if r.get("event") == "rejected":
            continue
        rows.append(r)
    return rows


def load_systems():
    p = _RES / "systems.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def memory_error_frontier(rows):
    # x = budget fraction (proxy for would-be cache fraction), y = mean D_h
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r["budget_fraction"]].append(r["D_h_kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        xs = sorted(by[pol])
        ys = [float(np.mean(by[pol][x])) for x in xs]
        plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("page budget fraction  (J/P ≈ would-be cache fraction)")
    plt.ylabel("behavioral error  D_h = KL(full || gate)  [nats]")
    plt.title("Memory–error frontier (lower-left is better)")
    plt.xscale("log", base=2)
    plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(_PLOTS / "memory_error_frontier.png", dpi=130)
    plt.close()


def latency_error_frontier(rows, sys_rows):
    # Map budget_fraction -> representative gated kernel p50 (avg over contexts).
    lat = defaultdict(list)
    for s in sys_rows:
        lat[float(s["budget_fraction"])].append(float(s["gated_p50_ms"]))
    lat_mean = {k: float(np.mean(v)) for k, v in lat.items()}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r["budget_fraction"]].append(r["D_h_kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        pts = []
        for bf in sorted(by[pol]):
            if bf in lat_mean:
                pts.append((lat_mean[bf], float(np.mean(by[pol][bf]))))
        if pts:
            xs, ys = zip(*sorted(pts))
            plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("gated decode-kernel latency  p50 [ms / layer-call]")
    plt.ylabel("behavioral error  D_h  [nats]")
    plt.title("Latency–error frontier")
    plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(_PLOTS / "latency_error_frontier.png", dpi=130)
    plt.close()


def context_scaling(rows, fixed_bf=0.125):
    # x = context length (R at full budget proxy via n_filler), y = D_h at fixed bf
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if abs(r["budget_fraction"] - fixed_bf) < 1e-9 or (
                r["policy"] == "full_pages"):
            by[r["policy"]][r["n_filler"]].append(r["D_h_kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        xs = sorted(by[pol])
        ys = [float(np.mean(by[pol][x])) for x in xs]
        plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("context size  (filler segments;  ↑ ≈ longer context)")
    plt.ylabel(f"behavioral error  D_h  at budget {fixed_bf:.4g}")
    plt.title("Context scaling of behavioral error at fixed budget")
    plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(_PLOTS / "context_scaling.png", dpi=130)
    plt.close()


def main():
    rows = load_raw()
    sys_rows = load_systems()
    if not rows:
        print("no results yet")
        return
    memory_error_frontier(rows)
    latency_error_frontier(rows, sys_rows)
    context_scaling(rows)
    print(f"wrote 3 plots to {_PLOTS}")


if __name__ == "__main__":
    main()
