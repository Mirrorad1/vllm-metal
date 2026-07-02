# SPDX-License-Identifier: Apache-2.0
"""exp022 — the per-context loss-matrix VERDICT (offline, CPU, seconds).

Reads s23.jsonl (+ any s4_shard*.jsonl) from exp022_substrates.py and answers:
is the incompressibility wall SUBSTRATE-shaped or CONTEXT-shaped?

  RESCUE   of eviction-wall contexts by another substrate at ~iso-bits (the kill shot)
  AGREE    within-family Spearman of per-context NLL-deltas across substrate columns
           (within-family per the exp021 fingerprint lesson — pooled correlations are
           inflated by family identity and are reported only for contrast)
  PREDICT  does the free write-time signal (eff_page_count / full features) predict each
           column's failures (entropy-sufficiency check, per column)
  DISPATCH oracle per-context argmin substrate vs best fixed substrate at ~iso-bits;
           plus the deployable TYPE-level dispatcher (best substrate per family)

Verdicts (SPEC thresholds from lacuna L1, minus split-half reliability — one probe per
context here, so binary correct + continuous NLL stand in; stated honestly):
  H-GENERAL (wall is context-shaped): rescue < 10% AND within-family mean |rho| >= 0.8
  H-SPECIFIC (dispatcher lives):      rescue >= 30% OR (within-family mean |rho| <= 0.4
                                      AND oracle-dispatch gain >= 15 pts)
  MIXED: anything between — report the numbers, no verdict theater.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy import stats

import exp021_admission as A

TIERS = {"0.25": ["evict@0.25", "quant4b", "lowrank@0.25", "lora"],
         "0.125": ["evict@0.125", "quant2b", "lowrank@0.125"]}


def load(dirpath):
    rows = []
    for line in open(f"{dirpath}/s23.jsonl"):
        rows.append(json.loads(line))
    lora = {}
    for f in glob.glob(f"{dirpath}/s4_shard*.jsonl"):
        for line in open(f):
            d = json.loads(line)
            lora[(d["family"], d["seed"])] = d
    for r in rows:
        s4 = lora.get((r["family"], r["seed"]))
        if s4:
            r["cells"]["lora"] = {"correct": s4["fidelity_correct"], "nll": s4["nll"],
                                  "bits": s4["bits_ratio"], "floor": s4["floor_correct"]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exp022_7b")
    args = ap.parse_args()
    rows = load(args.dir)
    fams = np.array([r["family"] for r in rows])
    cols = sorted({c for r in rows for c in r["cells"]})
    print(f"[load] n={len(rows)} contexts, columns: {cols}")

    def acc(col, mask=None):
        v = [r["cells"][col]["correct"] for r, m in zip(rows, np.ones(len(rows), bool) if mask is None else mask)
             if m and col in r["cells"]]
        return (np.mean(v), len(v)) if v else (float("nan"), 0)

    print("\n=== A. accuracy matrix (per family x substrate; bits in header) ===")
    bits = {c: np.mean([r["cells"][c]["bits"] for r in rows if c in r["cells"]]) for c in cols}
    hdr = " ".join(f"{c}({bits[c]:.2f})".rjust(18) for c in cols)
    print(f"{'family':>15} {'full':>6} {hdr}")
    for fam in sorted(set(fams)):
        m = fams == fam
        fa = np.mean([r["full"]["correct"] for r, mm in zip(rows, m) if mm])
        cells = " ".join(f"{acc(c, m)[0]:.2f} (n={acc(c, m)[1]})".rjust(18) for c in cols)
        print(f"{fam:>15} {fa:>6.2f} {cells}")

    print("\n=== B/C. RESCUE of eviction-wall contexts at ~iso-bits ===")
    for tier, tier_cols in TIERS.items():
        evict_col, others = tier_cols[0], [c for c in tier_cols[1:] if c in cols]
        wall = np.array([evict_col in r["cells"] and not r["cells"][evict_col]["correct"]
                         and r["full"]["correct"] for r in rows])
        if wall.sum() == 0:
            print(f" tier {tier}: no wall contexts (nothing broke under eviction) — skip")
            continue
        print(f" tier ~{tier} bits: {wall.sum()} eviction-wall contexts "
              f"({', '.join(f'{f}:{(wall & (fams==f)).sum()}' for f in sorted(set(fams[wall])))})")
        rescued_any = np.zeros(len(rows), bool)
        for c in others:
            resc = np.array([m and c in r["cells"] and r["cells"][c]["correct"]
                             for r, m in zip(rows, wall)])
            n_have = sum(1 for r, m in zip(rows, wall) if m and c in r["cells"])
            rescued_any |= resc
            print(f"   {c:>14}: rescues {resc.sum()}/{n_have}"
                  f" ({resc.sum()/max(n_have,1):.0%})")
        print(f"   {'ANY substrate':>14}: rescues {rescued_any.sum()}/{wall.sum()} "
              f"({rescued_any.sum()/wall.sum():.0%})  <-- the kill-shot number")

    print("\n=== D. per-context NLL-delta agreement across substrates (Spearman) ===")
    nll_full = np.array([r["full"]["nll"] for r in rows])
    col_nll = {}
    for c in cols:
        v = np.array([r["cells"][c]["nll"] - f if c in r["cells"] else np.nan
                      for r, f in zip(rows, nll_full)])
        col_nll[c] = v
    main_cols = [c for c in ("evict@0.25", "quant4b", "lowrank@0.25", "lora") if c in cols]
    for scope, mask_fn in (("pooled (inflated by family ID — contrast only)",
                            lambda fam: np.ones(len(rows), bool)),
                           ("within-family mean", None)):
        rhos = []
        for i, c1 in enumerate(main_cols):
            for c2 in main_cols[i + 1:]:
                if mask_fn:
                    m = mask_fn(None) & ~np.isnan(col_nll[c1]) & ~np.isnan(col_nll[c2])
                    if m.sum() > 8:
                        rhos.append((c1, c2, stats.spearmanr(col_nll[c1][m], col_nll[c2][m]).statistic))
                else:
                    per = []
                    for fam in sorted(set(fams)):
                        m = (fams == fam) & ~np.isnan(col_nll[c1]) & ~np.isnan(col_nll[c2])
                        if m.sum() > 8 and np.var(col_nll[c1][m]) > 0 and np.var(col_nll[c2][m]) > 0:
                            per.append(stats.spearmanr(col_nll[c1][m], col_nll[c2][m]).statistic)
                    if per:
                        rhos.append((c1, c2, float(np.mean(per))))
        if rhos:
            for c1, c2, r_ in rhos:
                print(f"   [{scope[:14]:>14}] rho({c1}, {c2}) = {r_:+.2f}")
            mean_abs = float(np.mean([abs(r_) for _, _, r_ in rhos]))
            print(f"   [{scope[:14]:>14}] mean |rho| = {mean_abs:.2f}")
            if scope.startswith("within"):
                wf_mean_abs = mean_abs

    print("\n=== E. does the free write-time signal predict each column? (AUC) ===")
    feats = [dict(A.features_from_mass(r["mass"], r["P"], r["plen"]), family=r["family"]) for r in rows]
    try:
        from sklearn.metrics import roc_auc_score
        for c in cols:
            y = np.array([0.0 if r["cells"][c]["correct"] else 1.0 for r in rows if c in r["cells"]])
            f_ = [f for f, r in zip(feats, rows) if c in r["cells"]]
            if len(np.unique(y)) < 2 or len(y) < 20:
                print(f"   {c:>14}: single-class or n<20 (n={len(y)}, break={y.mean():.2f})")
                continue
            ent = np.array([f["eff_page_count"] for f in f_])
            a_e = roc_auc_score(y, ent); a_e = max(a_e, 1 - a_e)
            sc = A._cv_scores(f_, y, A.FEATURES, seed=0)
            a_f = roc_auc_score(y, sc)
            print(f"   {c:>14}: entropy-only AUC={a_e:.2f}  full-feats CV AUC={a_f:.2f}  (break={y.mean():.2f})")
    except ImportError:
        print("   sklearn missing — skipped")

    print("\n=== F. dispatcher headroom at ~0.25 bits ===")
    t_cols = [c for c in TIERS["0.25"] if c in cols]
    have_all = [r for r in rows if all(c in r["cells"] for c in t_cols)]
    if len(have_all) >= 20:
        per_fixed = {c: np.mean([r["cells"][c]["correct"] for r in have_all]) for c in t_cols}
        best_fixed = max(per_fixed.values())
        oracle = np.mean([any(r["cells"][c]["correct"] for c in t_cols) for r in have_all])
        fam_best = {}
        for fam in sorted({r["family"] for r in have_all}):
            sub = [r for r in have_all if r["family"] == fam]
            fam_best[fam] = max(t_cols, key=lambda c: np.mean([r["cells"][c]["correct"] for r in sub]))
        type_disp = np.mean([r["cells"][fam_best[r["family"]]]["correct"] for r in have_all])
        print(f"   n={len(have_all)} with all columns; fixed: " +
              " ".join(f"{c}={v:.2f}" for c, v in per_fixed.items()))
        print(f"   best fixed={best_fixed:.2f}  TYPE-dispatcher={type_disp:.2f} "
              f"(per-family best: {fam_best})  per-context ORACLE={oracle:.2f}")
        print(f"   oracle gain over best fixed = {oracle - best_fixed:+.2f} "
              f"(win needs >= +0.15); type-dispatch captures "
              f"{(type_disp - best_fixed) / max(oracle - best_fixed, 1e-9):.0%} of it")
    else:
        print(f"   only {len(have_all)} contexts have every column — run s4 (or more shards) first")

    print("\n=== VERDICT ===")
    wall25 = np.array(["evict@0.25" in r["cells"] and not r["cells"]["evict@0.25"]["correct"]
                       and r["full"]["correct"] for r in rows])
    others25 = [c for c in TIERS["0.25"][1:] if c in cols]
    resc_any = np.array([m and any(c in r["cells"] and r["cells"][c]["correct"] for c in others25)
                         for r, m in zip(rows, wall25)])
    n_wall = int(wall25.sum())
    degenerate = len(rows) < 40 or n_wall < 8
    if degenerate:
        print(f"DEGENERATE / NOT EVALUABLE: n={len(rows)}, wall contexts={n_wall} — need the "
              f"full 7B run (dense families must actually break under eviction).")
        return
    rescue_rate = resc_any.sum() / n_wall
    try:
        wf = wf_mean_abs
    except NameError:
        wf = float("nan")
    if rescue_rate < 0.10 and (np.isnan(wf) or wf >= 0.8):
        print(f"H-GENERAL: the wall is CONTEXT-shaped. Rescue {rescue_rate:.0%} (<10%), "
              f"within-family |rho|={wf:.2f}. No substrate escapes — distillation included "
              f"(if the lora column agrees). Only dense-tier / re-read remain (build L2); "
              f"compressibility-gated consolidation is DEAD (fires where it can't help).")
    elif rescue_rate >= 0.30 or (not np.isnan(wf) and wf <= 0.4):
        print(f"H-SPECIFIC: the wall is SUBSTRATE-shaped. Rescue {rescue_rate:.0%} (>=30%), "
              f"within-family |rho|={wf:.2f}. The admission gate upgrades to a substrate "
              f"DISPATCHER — see section F for how much a type-level dispatch captures.")
    else:
        print(f"MIXED: rescue {rescue_rate:.0%}, within-family |rho|={wf:.2f} — between the "
              f"thresholds. Read sections B-F; consider more contexts or the 0.125 tier.")


if __name__ == "__main__":
    main()
