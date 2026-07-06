# SPDX-License-Identifier: Apache-2.0
"""Render frontier plots from results/master_raw_results.jsonl + systems csv."""
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

POLICIES = ["full_pages", "recent_pages", "seeded_random_pages", "sink_recent_pages",
            "attention_proxy_pages", "page_norm_pages", "loss_budgeted_oracle",
            "loss_budgeted_online"]
MARK = dict(zip(POLICIES, ["o", "s", "x", "v", "^", "d", "*", "P"]))


def load_raw():
    p = _RES / "master_raw_results.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if '"event"' not in l]


def load_sys():
    for cand in [_RES / "exp010_latency_frontier" / "systems.csv",
                 _RES / "exp009_physical_memory" / "systems.csv",
                 _RES / "exp002_all_pages_equivalence" / "systems.csv"]:
        if cand.exists():
            return list(csv.DictReader(open(cand)))
    return []


def memory_error_frontier(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r["budget_fraction"]].append(r["kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        xs = sorted(by[pol]); ys = [float(np.mean(by[pol][x])) for x in xs]
        plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("page budget fraction (J/P; would-be cache fraction)")
    plt.ylabel("D_h = KL(full||gate) [nats]")
    plt.title("Memory–error frontier")
    plt.xscale("log", base=2); plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(_PLOTS / "memory_error_frontier.png", dpi=130); plt.close()


def latency_error_frontier(rows, sysrows):
    lat = defaultdict(list)
    for s in sysrows:
        lat[float(s["budget_fraction"])].append(float(s["gated_p50_ms"]))
    lat_mean = {k: float(np.mean(v)) for k, v in lat.items()}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r["budget_fraction"]].append(r["kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        pts = [(lat_mean[bf], float(np.mean(by[pol][bf]))) for bf in sorted(by[pol]) if bf in lat_mean]
        if pts:
            xs, ys = zip(*sorted(pts)); plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("gated kernel p50 [ms/layer-call]"); plt.ylabel("D_h [nats]")
    plt.title("Latency–error frontier"); plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(_PLOTS / "latency_error_frontier.png", dpi=130); plt.close()


def context_scaling(rows, bf=0.125):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if abs(r["budget_fraction"] - bf) < 1e-9 or r["policy"] == "full_pages":
            by[r["policy"]][r["n_filler"]].append(r["kl"])
    plt.figure(figsize=(7, 5))
    for pol in POLICIES:
        if pol not in by:
            continue
        xs = sorted(by[pol]); ys = [float(np.mean(by[pol][x])) for x in xs]
        plt.plot(xs, ys, marker=MARK[pol], label=pol)
    plt.xlabel("filler size (↑ ≈ longer context)")
    plt.ylabel(f"D_h at budget {bf}")
    plt.title("Context scaling at fixed budget"); plt.yscale("symlog", linthresh=1e-3)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(_PLOTS / "context_scaling.png", dpi=130); plt.close()


def main():
    rows = load_raw()
    if not rows:
        print("no results yet"); return
    sysrows = load_sys()
    memory_error_frontier(rows)
    latency_error_frontier(rows, sysrows)
    context_scaling(rows)
    print(f"wrote 3 plots to {_PLOTS}")


if __name__ == "__main__":
    main()
