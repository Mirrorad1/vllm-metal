# SPDX-License-Identifier: Apache-2.0
"""exp021 — conformal compressibility ADMISSION CONTROLLER (Cut 1, offline).

See SPEC_exp021_admission.md. This is the cheap first cut: build a per-instance,
query-AGNOSTIC, write-side certificate of answer-level compression damage from exp020's
own outputs, calibrate it CONFORMALLY (distribution-free coverage), wire it as an
admission gate, and answer the one decisive question:

    HORN-B / H2: does the ANSWER-LEVEL certificate beat a pure ATTENTION-ENTROPY
    threshold on the (silent-failure x efficiency) frontier?

  * If YES (more compression admitted at the SAME guaranteed safety, low rank-corr with
    entropy): a real new mechanism (new-combination).
  * If NO (entropy matches it): a calibrated entropy admission gate -- still deployable
    and unshipped, but NOT novel. Either way you learn it cheaply, with no GPU.

Two run modes:
  python exp021_admission.py --selftest            # validate the harness TODAY, no data
  python exp021_admission.py --runs results/exp020_quality/raw_<MODEL>.jsonl --budget 0.25

The --runs path needs an exp020 jsonl produced by the INSTRUMENTED harness (it now dumps
one {"event":"instance", ..., "mass":[...]} line per accepted instance). Old jsonl without
mass runs in a DEGRADED structural-only mode (no entropy => no HORN-B; it says so loudly).

Self-proving preflight (the exp020 discipline):
  * gate at B=1.0 must admit ~everything (a no-op);
  * --selftest asserts coverage <= alpha on held-out synth AND that the full-feature gate
    strictly beats the entropy gate on a PLANTED adversarial subset (proves the HORN-B
    comparison is *sensitive* -- it won't falsely cry "entropy suffices" when it doesn't).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

# Ordered feature vocabulary. The ENTROPY baseline (B1, the free signal / HORN-B opponent)
# is restricted to EFFICIENCY_ONLY; the full answer-level certificate sees all of them.
FEATURES = ["entropy", "eff_page_count", "top1", "top5", "gini", "logP"]
ENTROPY_ONLY = ["eff_page_count"]


# ---------------------------------------------------------------------------
# features (query-agnostic, write-side: derived from the prefill attention mass only)
# ---------------------------------------------------------------------------
def features_from_mass(mass, P, L):
    m = np.clip(np.asarray(mass, float), 0.0, None)
    s = m.sum()
    p = m / s if s > 0 else np.ones(len(m)) / max(len(m), 1)
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())            # attention-mass entropy (nats)
    srt = np.sort(p)[::-1]
    return {
        "entropy": H,
        "eff_page_count": float(math.exp(H)),       # how many pages effectively carry mass
        "top1": float(srt[0]) if len(srt) else 0.0,
        "top5": float(srt[:5].sum()),
        "gini": float(1.0 - (p ** 2).sum()),        # diversity (1 - Simpson)
        "logP": float(math.log(max(P, 1))),
    }


# ---------------------------------------------------------------------------
# conformal selective gate (split-conformal selective classification)
# ---------------------------------------------------------------------------
def cp_upper(k, n, delta):
    """Clopper-Pearson (1-delta) UPPER bound on a binomial rate: k breaks in n admitted."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, k + 1, n - k))


def conformal_threshold(s_cal, y_cal, alpha, delta):
    """Largest admit set s.t. CP-upper(break rate of admitted) <= alpha.

    Admit instances with score s <= t (lower score = predicted safer). Scan candidate
    thresholds; among those whose admitted-set CP-upper <= alpha, pick the one admitting
    the MOST instances (max efficiency at the guarantee). Returns t (or -inf = admit none)."""
    s_cal = np.asarray(s_cal, float)
    y_cal = np.asarray(y_cal, float)
    order = np.argsort(s_cal)
    best_t, best_n = -np.inf, -1
    for i in order:
        t = s_cal[i]
        adm = s_cal <= t
        n = int(adm.sum())
        k = int(y_cal[adm].sum())
        if cp_upper(k, n, delta) <= alpha and n > best_n:
            best_n, best_t = n, t
    return best_t


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------
def _matrix(rows, feats):
    return np.array([[r[f] for f in feats] for r in rows], float)


def fit_score(train_rows, train_y, eval_rows, feats):
    """P(break) on eval_rows from a logistic model fit on train_rows[feats]. Degenerate-safe."""
    yt = np.asarray(train_y, float)
    Xtr, Xev = _matrix(train_rows, feats), _matrix(eval_rows, feats)
    if not _HAVE_SK or len(np.unique(yt)) < 2:
        # fall back to a single monotone feature (the first) or the base rate
        if len(np.unique(yt)) < 2:
            return np.full(len(eval_rows), float(yt.mean()))
        z = Xev[:, 0]
        return (z - z.min()) / (z.ptp() + 1e-9)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), yt)
    return lr.predict_proba(sc.transform(Xev))[:, 1]


# ---------------------------------------------------------------------------
# evaluation: full vs entropy controller, over random train/cal/test folds
# ---------------------------------------------------------------------------
def evaluate(rows, y, alpha=0.1, delta=0.1, folds=40, seed=0):
    rows = list(rows)
    y = np.asarray(y, float)
    n = len(rows)
    rng = np.random.default_rng(seed)
    out = {k: defaultdict(list) for k in ("full", "entropy")}
    naive_break, paired_gain = [], []         # PAIRED per-fold (eff_full - eff_entropy)
    for _ in range(folds):
        idx = rng.permutation(n)
        a, b = n // 2, (3 * n) // 4
        tr, ca, te = idx[:a], idx[a:b], idx[b:]
        if len(te) == 0 or len(ca) == 0:
            continue
        naive_break.append(float(y[te].mean()))      # B0: admit-all break rate
        eff_fold = {}
        for name, feats in (("full", FEATURES), ("entropy", ENTROPY_ONLY)):
            s_ca = fit_score([rows[i] for i in tr], y[tr], [rows[i] for i in ca], feats)
            s_te = fit_score([rows[i] for i in tr], y[tr], [rows[i] for i in te], feats)
            t = conformal_threshold(s_ca, y[ca], alpha, delta)
            adm = s_te <= t
            eff = float(adm.mean())                                    # fraction compressed at B
            cov = float(y[te][adm].mean()) if adm.any() else 0.0       # break rate among admitted
            out[name]["efficiency"].append(eff)
            out[name]["coverage_break"].append(cov)
            eff_fold[name] = eff
        paired_gain.append(eff_fold["full"] - eff_fold["entropy"])     # same split => paired
    # rank-correlation of the answer-level score with the free entropy signal (HORN-B tell)
    s_full = fit_score(rows, y, rows, FEATURES)
    ent = _matrix(rows, ["eff_page_count"]).ravel()
    informative = float(np.var(ent)) > 1e-12 and float(np.var(s_full)) > 1e-12
    rho = float(stats.spearmanr(s_full, ent).statistic) if informative else float("nan")
    g = np.asarray(paired_gain, float)
    res = {"n": n, "base_break_rate": float(y.mean()), "alpha": alpha, "delta": delta,
           "features_informative": informative,
           "naive_break_rate": float(np.mean(naive_break)) if naive_break else float("nan"),
           "spearman_full_vs_entropy": rho,
           "paired_gain_mean": float(g.mean()) if len(g) else 0.0,
           "paired_gain_se": float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else float("inf")}
    for name in ("full", "entropy"):
        res[name] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in out[name].items()}
    return res


def report(res, title="exp021 admission controller"):
    print(f"\n=== {title} ===")
    print(f"n={res['n']}  base break rate={res['base_break_rate']:.3f}  "
          f"target alpha={res['alpha']}  delta={res['delta']}")
    print(f"B0 naive (admit-all) break rate = {res['naive_break_rate']:.3f}  "
          f"(this is the silent-failure rate today)")
    print(f"{'controller':>10} | {'efficiency(admit@B)':>20} | {'coverage(break|admit)':>22}")
    for name in ("full", "entropy"):
        e, ce = res[name]["efficiency"]
        c, cc = res[name]["coverage_break"]
        print(f"{name:>10} | {e:>10.3f} ± {2*ce:<6.3f} | {c:>10.3f} ± {2*cc:<7.3f}")
    cf, _ = res["full"]["coverage_break"]
    rho = res["spearman_full_vs_entropy"]
    gain, se = res["paired_gain_mean"], res["paired_gain_se"]
    print(f"\nHORN-B: PAIRED efficiency gain (full - entropy) = {gain:+.3f} ± {2*se:.3f} (2 SE);  "
          f"rho(full,entropy)={rho:.3f}")
    sound = cf <= res["alpha"] + 0.02
    # a real mechanism: gain is BOTH statistically separated from 0 (paired) AND materially
    # large (>3pp), AND the answer-level score is not just the entropy signal (rho not ~1).
    beats = bool(gain > max(3 * se, 0.03) and rho < 0.8)
    eff_full = res["full"]["efficiency"][0]
    eff_ent = res["entropy"]["efficiency"][0]
    degenerate = max(eff_full, eff_ent) < 0.02 or res["base_break_rate"] < 0.02 or res["n"] < 40
    if degenerate:
        verdict = (f"DEGENERATE / NOT EVALUABLE: gate admits ~nothing (eff={eff_full:.2f}) or too few "
                   f"break events (base={res['base_break_rate']:.3f}, n={res['n']}). This is the vacuity "
                   f"horn from tiny/low-variance data -- NOT a k1 entropy-redundancy verdict. Need more "
                   f"instances with real damage variance (the 7B/dense regime).")
        beats = False
    elif not res.get("features_informative", True):
        verdict = ("N/A (DEGRADED: no attention-mass features in this run -> no entropy/answer-level "
                   "signal). Re-run the instrumented exp020 to dump per-instance mass, then re-run.")
        beats = False
    elif not sound:
        verdict = "KILL k2/k3: coverage not held (vacuous or unsound) -- see SPEC."
    elif beats:
        verdict = "WIN (H2): answer-level certificate BEATS entropy -> new mechanism."
    else:
        verdict = "KILL k1 (HORN-B): entropy threshold matches it -> calibrated-entropy gate, NOT novel."
    print(f"VERDICT: {verdict}\n")
    return {"sound": sound, "beats_entropy": beats, "gain": gain, "rho": rho,
            "informative": res.get("features_informative", True)}


# ---------------------------------------------------------------------------
# loader (handles both exp020 jsonl schemas + the new instance feature lines)
# ---------------------------------------------------------------------------
def load_runs(path, budget, policy="attention", tol=1e-6):
    """Return (rows, y, mode). mode='full' if per-instance mass features are present,
    else 'degraded' (structural only -> no entropy, no real HORN-B)."""
    feats_by_key, recs = {}, []
    n_event = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("event") == "instance":
                n_event += 1
                feats_by_key[(d["family"], d["seed"])] = features_from_mass(d["mass"], d["P"], d["L"])
            elif d.get("event"):
                continue  # oom / other events
            else:
                recs.append(d)
    mode = "full" if n_event else "degraded"
    rows, y = [], []
    # schema B (current/patched): per-(family,seed,budget,policy) with 'correct'
    # schema A (old): per-(budget) with attention_correct/recent_correct, no seed
    for d in recs:
        if abs(float(d["budget"]) - budget) > tol:
            continue
        if "correct" in d and "policy" in d:           # schema B
            if d["policy"] != policy:
                continue
            correct = bool(d["correct"]); key = (d["family"], d.get("seed"))
        elif f"{policy}_correct" in d:                  # schema A
            correct = bool(d[f"{policy}_correct"]); key = (d["family"], d.get("seed"))
        else:
            continue
        if mode == "full":
            f = feats_by_key.get(key)
            if f is None:
                continue
            row = dict(f)
        else:
            row = {k: 0.0 for k in FEATURES}            # structural placeholder
            row["logP"] = float(math.log(max(d.get("P", 1), 1)))
        row["family"] = d["family"]
        rows.append(row); y.append(0.0 if correct else 1.0)
    return rows, np.asarray(y, float), mode


# ---------------------------------------------------------------------------
# self-test: synthetic data with a known split + a PLANTED adversarial subset
# ---------------------------------------------------------------------------
def make_selftest(n=600, frac_adv=0.3, seed=0):
    """Majority: entropy (eff_page_count) cleanly predicts break (dense=high->break).
    Adversarial subset: eff_page_count is RANDOM (uninformative) but 'top1' separates.
    A correct harness must (a) hold coverage and (b) show full > entropy efficiency."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for i in range(n):
        broke = int(rng.random() < 0.5)
        adv = rng.random() < frac_adv
        if not adv:
            eff = rng.normal(40 if broke else 5, 3)        # dense breaks (flat mass, high eff)
            top1 = rng.normal(0.4, 0.05)                   # uninformative here
        else:
            eff = rng.normal(20, 8)                         # entropy uninformative
            top1 = rng.normal(0.1 if broke else 0.8, 0.05) # hidden signal carries it
        rows.append({"entropy": math.log(max(eff, 1e-3)), "eff_page_count": float(eff),
                     "top1": float(np.clip(top1, 0, 1)), "top5": float(np.clip(top1 + 0.1, 0, 1)),
                     "gini": float(rng.random()), "logP": float(math.log(128)), "family": "adv" if adv else "easy"})
        y.append(float(broke))
    return rows, np.asarray(y, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="exp020 raw_*.jsonl (instrumented: with event:instance lines)")
    ap.add_argument("--budget", type=float, default=0.25, help="compression budget B to certify")
    ap.add_argument("--policy", default="attention", choices=["attention", "recent"])
    ap.add_argument("--alpha", type=float, default=0.10, help="max silent-failure rate among admitted")
    ap.add_argument("--delta", type=float, default=0.10, help="conformal confidence (1-delta coverage)")
    ap.add_argument("--folds", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("[selftest] synthetic data: majority entropy-separable + planted adversarial subset")
        rows, y = make_selftest()
        res = evaluate(rows, y, alpha=args.alpha, delta=args.delta, folds=args.folds)
        v = report(res, "SELFTEST")
        # preflight assertions: coverage held AND the harness DETECTS the planted mechanism
        assert v["sound"], "selftest FAIL: conformal coverage not held (gate logic broken)"
        assert v["beats_entropy"], ("selftest FAIL: full-feature gate did NOT beat entropy on the "
                                    "planted adversarial subset -> HORN-B comparison is insensitive")
        print("[selftest] PASS: coverage held AND mechanism detected when present. Harness trustworthy.")
        return

    if not args.runs:
        ap.error("provide --runs <exp020 jsonl> or --selftest")
    rows, y, mode = load_runs(args.runs, args.budget, args.policy)
    print(f"[load] {len(rows)} instances at budget={args.budget} policy={args.policy}; mode={mode}; "
          f"break rate={y.mean():.3f}")
    if mode == "degraded":
        print("[load] WARNING: no per-instance mass features (event:instance lines absent).\n"
              "       This jsonl predates the instrumentation -> no entropy feature -> HORN-B test\n"
              "       is NOT meaningful. Re-run exp020_quality_cuda.py (now instrumented) to dump mass.")
    if len(rows) < 20:
        print("[load] too few instances for a trustworthy conformal split; treat as a smoke test.")
    # preflight: a gate at B=1.0 should admit ~everything (no-op). Run if that budget exists.
    r1, y1, _ = load_runs(args.runs, 1.0, args.policy)
    if len(r1) and y1.mean() < 0.05:
        print(f"[preflight] B=1.0 break rate={y1.mean():.3f} (~0, full-cache no-op) OK")
    res = evaluate(rows, y, alpha=args.alpha, delta=args.delta, folds=args.folds)
    report(res, f"exp021 admission @ budget={args.budget}")


if __name__ == "__main__":
    main()
