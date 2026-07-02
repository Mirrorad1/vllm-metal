# SPDX-License-Identifier: Apache-2.0
"""exp021 L6 Cut 1 — cross-scale TRANSFER verdict (offline, CPU, seconds).

Joins the 0.5B proxy's write-time features (exp021_proxy_local.py) against the 7B target's
break/safe labels and features (the instrumented exp020 jsonl), and answers: can the tiny
model supply the routing bit BEFORE the target prefills?

Honesty note (post-fingerprint): exp021 showed the 7B's own features carry TASK-TYPE
identity, not per-instance damage (within-family AUC ~ chance). So the bar for the proxy
is stated in those terms — the proxy must reproduce the TYPE signal and the resulting
gate efficiency; nobody demands per-instance signal the target itself does not have.

Checks, in order:
  T0  positive control: re-derive the target-feature gate + family-ID AUC from the target
      jsonl through THIS script's join (must reproduce exp021's numbers or the join is bad).
  T1  cross-scale feature agreement: Spearman(proxy feat, target feat) per feature.
  T2  the crux: family-ID AUC (multi_hop vs sum_scattered — the pair entropy is blind to)
      from PROXY features.
  T3  pooled break-prediction AUC: proxy features -> y_7B@B (5-fold CV logistic).
  T4  the deliverable: conformal admission gate on PROXY features at alpha=0.1 vs the
      target-feature gate and the entropy-only arms (same fold RNG as exp021).
  T5  competence stratification: does transfer hold where the proxy itself FAILS the task?
      (the known L6 failure mode: an incompetent proxy mis-measures load-bearingness)

Verdict:
  WIN    T2 >= 0.9 AND T4 proxy-gate efficiency >= 0.9x target-feature gate at coverage<=alpha
  PARTIAL anything between
  KILL   T2 < 0.7 OR proxy gate <= entropy-only target gate (the proxy adds nothing a free
         scalar on the target already gives)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

import exp021_admission as A


def load_join(target_path, proxy_path, budget, policy="attention"):
    t_rows, t_y, mode = A.load_runs(target_path, budget, policy)
    assert mode == "full", "target jsonl lacks instance mass lines"
    # rebuild the (family, seed) keying that load_runs used
    keys, feats_t = [], {}
    with open(target_path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "instance":
                feats_t[(d["family"], d["seed"])] = A.features_from_mass(d["mass"], d["P"], d["L"])
            elif not d.get("event") and "correct" in d and d.get("policy") == policy \
                    and abs(float(d["budget"]) - budget) < 1e-6:
                keys.append((d["family"], d["seed"]))
    feats_p, prox_ok = {}, {}
    with open(proxy_path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "proxy_instance":
                k = (d["family"], d["seed"])
                feats_p[k] = A.features_from_mass(d["mass"], d["P"], d["L"])
                prox_ok[k] = bool(d["proxy_correct"])
    rows_t, rows_p, y, fams, pok = [], [], [], [], []
    for k, yy in zip(keys, t_y):
        if k not in feats_t or k not in feats_p:
            continue
        rt = dict(feats_t[k]); rt["family"] = k[0]
        rp = dict(feats_p[k]); rp["family"] = k[0]
        rows_t.append(rt); rows_p.append(rp); y.append(yy)
        fams.append(k[0]); pok.append(prox_ok[k])
    return rows_t, rows_p, np.asarray(y, float), np.array(fams), np.array(pok, bool)


def cv_auc(rows, y, feats, seed=0):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    sc = A._cv_scores(rows, np.asarray(y, float), feats, seed=seed)
    return float(roc_auc_score(y, sc))


def gate(rows, y, feats, alpha=0.1, delta=0.1, folds=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rows)
    effs, covs = [], []
    for _ in range(folds):
        idx = rng.permutation(n)
        a, b = n // 2, (3 * n) // 4
        tr, ca, te = idx[:a], idx[a:b], idx[b:]
        s_ca = A.fit_score([rows[i] for i in tr], y[tr], [rows[i] for i in ca], feats)
        s_te = A.fit_score([rows[i] for i in tr], y[tr], [rows[i] for i in te], feats)
        t = A.conformal_threshold(s_ca, y[ca], alpha, delta)
        adm = s_te <= t
        effs.append(float(adm.mean()))
        covs.append(float(y[te][adm].mean()) if adm.any() else 0.0)
    return float(np.mean(effs)), float(np.mean(covs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="results/exp021_7b/raw_Qwen2.5-7B-Instruct.jsonl")
    ap.add_argument("--proxy", default="results/exp021_proxy_0p5b/proxy_Qwen2.5-0.5B-Instruct.jsonl")
    ap.add_argument("--budget", type=float, default=0.0625)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    rows_t, rows_p, y, fams, pok = load_join(args.target, args.proxy, args.budget)
    print(f"[join] n={len(y)} base_break={y.mean():.3f} @B={args.budget}; "
          f"proxy full-ctx accuracy by family:")
    for fam in sorted(set(fams)):
        m = fams == fam
        print(f"   {fam:>15}: n={m.sum():>3} target_break={y[m].mean():.2f} proxy_correct={pok[m].mean():.2f}")

    print("\n=== T0 positive control (target features through this join) ===")
    e_t, c_t = gate(rows_t, y, A.FEATURES, args.alpha)
    e_te, c_te = gate(rows_t, y, A.ENTROPY_ONLY, args.alpha)
    pair = (fams == "multi_hop") | (fams == "sum_scattered")
    lab = (fams[pair] == "sum_scattered").astype(float)
    fid_t = cv_auc([r for r, m in zip(rows_t, pair) if m], lab, A.FEATURES)
    print(f"target gate eff={e_t:.3f} cov={c_t:.3f} | entropy eff={e_te:.3f} | "
          f"family-ID AUC(mh vs ss)={fid_t:.3f}   (expect ~0.75/0.55/1.00)")

    print("\n=== T1 cross-scale feature agreement (Spearman proxy vs target) ===")
    for f in A.FEATURES:
        vt = np.array([r[f] for r in rows_t]); vp = np.array([r[f] for r in rows_p])
        if np.var(vt) < 1e-12 or np.var(vp) < 1e-12:
            print(f"   {f:>15}: constant"); continue
        r_all = stats.spearmanr(vp, vt).statistic
        r_fam = np.nanmean([stats.spearmanr(vp[fams == fam], vt[fams == fam]).statistic
                            for fam in sorted(set(fams)) if (fams == fam).sum() > 5])
        print(f"   {f:>15}: pooled rho={r_all:+.3f}   within-family mean rho={r_fam:+.3f}")

    print("\n=== T2 crux: family-ID AUC (multi_hop vs sum_scattered) from PROXY features ===")
    fid_p = cv_auc([r for r, m in zip(rows_p, pair) if m], lab, A.FEATURES)
    fid_pe = cv_auc([r for r, m in zip(rows_p, pair) if m], lab, A.ENTROPY_ONLY)
    print(f"proxy full-feats AUC={fid_p:.3f}   proxy entropy-only AUC={fid_pe:.3f}   (target: {fid_t:.3f})")

    print("\n=== T3 pooled break-prediction AUC (proxy feats -> y_7B) ===")
    auc_p = cv_auc(rows_p, y, A.FEATURES)
    auc_t = cv_auc(rows_t, y, A.FEATURES)
    print(f"proxy AUC={auc_p:.3f}   target AUC={auc_t:.3f}")

    print("\n=== T4 conformal gate on PROXY features ===")
    e_p, c_p = gate(rows_p, y, A.FEATURES, args.alpha)
    e_pe, c_pe = gate(rows_p, y, A.ENTROPY_ONLY, args.alpha)
    print(f"{'arm':>28} {'eff':>7} {'cov':>7}")
    print(f"{'proxy full-feats':>28} {e_p:>7.3f} {c_p:>7.3f}")
    print(f"{'proxy entropy-only':>28} {e_pe:>7.3f} {c_pe:>7.3f}")
    print(f"{'target full-feats (ceiling)':>28} {e_t:>7.3f} {c_t:>7.3f}")
    print(f"{'target entropy-only':>28} {e_te:>7.3f} {c_te:>7.3f}")

    print("\n=== T5 competence stratification ===")
    for name, m in (("proxy-correct", pok), ("proxy-WRONG", ~pok)):
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            print(f"   {name}: n={m.sum()} — too small / single-class"); continue
        a = cv_auc([r for r, mm in zip(rows_p, m) if mm], y[m], A.FEATURES)
        print(f"   {name}: n={m.sum()} break={y[m].mean():.2f} proxy-feats AUC={a:.3f}")

    degenerate = (len(y) < 40 or len(np.unique(y)) < 2 or np.isnan(fid_p)
                  or max(e_t, e_te) < 0.02 or y.mean() < 0.02)
    if degenerate:
        print(f"\nVERDICT (L6 Cut 1 @B={args.budget}): DEGENERATE / NOT EVALUABLE "
              f"(n={len(y)}, base_break={y.mean():.3f}, family-ID={fid_p}) — not a transfer "
              f"verdict, just insufficient data. Run the full proxy pass.\n")
        return
    sound = c_p <= args.alpha + 0.02
    win = sound and fid_p >= 0.9 and e_p >= 0.9 * e_t
    kill = fid_p < 0.7 or e_p <= e_te
    if win:
        v = (f"WIN: the 0.5B proxy reproduces the type signal (family-ID {fid_p:.2f}) and its gate "
             f"reaches {e_p/e_t:.0%} of the target-feature gate at coverage<=alpha — the routing bit "
             f"is available PRE-PREFILL at ~1/14 the FLOPs. Wall classification transfers across scale.")
    elif kill:
        v = (f"KILL: proxy features do not carry the type signal (family-ID {fid_p:.2f}) or the proxy "
             f"gate ({e_p:.3f}) adds nothing over a free entropy threshold on the target ({e_te:.3f}). "
             f"Routing must wait for target-side prefill; the proxy-routing branch is dead.")
    else:
        v = (f"PARTIAL: type signal transfers imperfectly (family-ID {fid_p:.2f}; gate {e_p:.3f} vs "
             f"target {e_t:.3f}). A bigger proxy (1.5B) or proxy-competence features may close it.")
    if not sound:
        v = f"UNSOUND: proxy-gate coverage {c_p:.3f} > alpha — do not trust efficiency numbers. " + v
    print(f"\nVERDICT (L6 Cut 1 @B={args.budget}): {v}\n")


if __name__ == "__main__":
    main()
