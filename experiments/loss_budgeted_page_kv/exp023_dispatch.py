# SPDX-License-Identifier: Apache-2.0
"""exp023 — the SUBSTRATE DISPATCHER (offline; the exp021 gate grows an action space).

exp022 proved the wall is eviction-shaped: dense-aggregation contexts that no eviction
budget can serve survive 4-bit precision at iso-bits (52% rescue) and 8-bit near-fully,
while eviction remains unbeatable on sparse retrieval (16x). exp021/L6 proved the routing
signal (task-type from write-time attention mass) is free, calibratable, and readable even
by a 0.5B proxy. exp023 composes them:

    per context: choose the CHEAPEST action whose calibrated break-risk clears alpha
    action space (bits/token ratio): evict@6.25% (0.0625) < evict@12.5% < evict@25%
                                     < kivi-4bit (~0.27) < kivi-8bit (~0.52) < FULL (1.0)

Guarantee: split-conformal per action — every action's admitted set holds break <= alpha
with confidence 1-delta (Clopper-Pearson), so the union (the dispatched fleet) does too.

Arms:
  dispatch/target  cascade on 7B write-time features (the deployable engine-side policy)
  dispatch/proxy   cascade on 0.5B-proxy features (route BEFORE target prefill; L6)
  dispatch/entropy cascade on eff_page_count only (is the richer signal needed?)
  gate(a)          exp021 generalized: admit-action-a-else-FULL, for every a (the best of
                   these is the incumbent to beat)
  oracle           cheapest action that is actually correct (headroom ceiling)
  fixed(a)         every context to action a (no policy floor)

Metric: MEAN BITS RATIO (fleet KV memory) at guaranteed coverage, plus realized accuracy
and the per-family action histogram (interpretability: sparse->evict, dense->quant/full).

Verdict @ alpha=0.1 (paired per-fold vs the BEST two-action gate):
  WIN    >=25% mean-bits reduction, >=3SE, coverage held  -> wire into serving (Cut 2)
  MODEST 10-25% -> real but niche; report honestly
  KILL   <10%  -> a single well-chosen two-action gate was enough; dispatcher is theater

Everything runs on committed artifacts: results/exp022_7b/s23.jsonl (actions x outcomes +
mass) and results/exp021_proxy_0p5b/*.jsonl (proxy features). --selftest plants a
synthetic two-type population (only a mid-price action saves type B) and must show the
dispatcher beating every two-action gate; its negative control (one action dominates)
must show NO dispatcher win. Same discipline as exp021: the harness proves it can detect
both presence and absence before it is trusted on real data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

import exp021_admission as A

FULL_BITS = 1.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_matrix(s23_path, proxy_path=None, actions=None):
    """rows: per context {feats, feats_proxy?, family, seed, acts: {name: (break01, bits)}}.
    Contexts whose FULL cache is wrong are excluded (guarantee is conditional on
    full-cache success, exp020's acceptance discipline)."""
    prox = {}
    if proxy_path:
        for line in open(proxy_path):
            d = json.loads(line)
            if d.get("event") == "proxy_instance":
                prox[(d["family"], d["seed"])] = A.features_from_mass(d["mass"], d["P"], d["L"])
    rows = []
    for line in open(s23_path):
        d = json.loads(line)
        if not d["full"]["correct"]:
            continue
        feats = A.features_from_mass(d["mass"], d["P"], d["plen"])
        feats["family"] = d["family"]
        acts = {}
        for name, cell in d["cells"].items():
            if actions and name not in actions:
                continue
            acts[name] = (0.0 if cell["correct"] else 1.0, float(cell["bits"]))
        r = {"feats": feats, "family": d["family"], "seed": d["seed"], "acts": acts}
        fp = prox.get((d["family"], d["seed"]))
        if fp is not None:
            r["feats_proxy"] = dict(fp, family=d["family"])
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# the cascade
# ---------------------------------------------------------------------------
def dispatch_fold(rows, tr, ca, te, action_names, feat_key, feats, alpha, delta):
    """Returns per-test-context (chosen_action, bits, broke). Cascade: actions sorted by
    bits ascending; per-action logistic P(break) + conformal threshold; first action whose
    score clears its threshold wins; else FULL (bits 1.0, break 0 by acceptance)."""
    order = sorted(action_names, key=lambda a: np.mean([rows[i]["acts"][a][1] for i in tr]))
    scores, thresholds = {}, {}
    for a in order:
        ytr = np.array([rows[i]["acts"][a][0] for i in tr])
        s_ca = A.fit_score([rows[i][feat_key] for i in tr], ytr,
                           [rows[i][feat_key] for i in ca], feats)
        s_te = A.fit_score([rows[i][feat_key] for i in tr], ytr,
                           [rows[i][feat_key] for i in te], feats)
        y_ca = np.array([rows[i]["acts"][a][0] for i in ca])
        thresholds[a] = A.conformal_threshold(s_ca, y_ca, alpha, delta)
        scores[a] = s_te
    out = []
    for j, i in enumerate(te):
        pick, bits, broke = "full", FULL_BITS, 0.0
        for a in order:
            if scores[a][j] <= thresholds[a]:
                pick = a
                broke, bits = rows[i]["acts"][a]
                break
        out.append((pick, bits, broke))
    return out


def evaluate(rows, action_names, feat_key, feats, alpha, delta, folds=40, seed=0):
    n = len(rows)
    rng = np.random.default_rng(seed)
    bits_f, brk_f, hist = [], [], Counter()
    fam_hist = defaultdict(Counter)
    gate_bits = defaultdict(list)          # per two-action gate baseline
    gate_brk = defaultdict(list)
    oracle_bits = []
    for _ in range(folds):
        idx = rng.permutation(n)
        a_, b_ = n // 2, (3 * n) // 4
        tr, ca, te = idx[:a_], idx[a_:b_], idx[b_:]
        if not len(te) or not len(ca):
            continue
        picks = dispatch_fold(rows, tr, ca, te, action_names, feat_key, feats, alpha, delta)
        bits_f.append(float(np.mean([p[1] for p in picks])))
        lossy = [p for p in picks if p[0] != "full"]
        brk_f.append(float(np.mean([p[2] for p in lossy])) if lossy else 0.0)
        for (pick, _, _), i in zip(picks, te):
            hist[pick] += 1
            fam_hist[rows[i]["family"]][pick] += 1
        # two-action gates on the SAME folds (paired)
        for a in action_names:
            ytr = np.array([rows[i]["acts"][a][0] for i in tr])
            s_ca = A.fit_score([rows[i][feat_key] for i in tr], ytr,
                               [rows[i][feat_key] for i in ca], feats)
            s_te = A.fit_score([rows[i][feat_key] for i in tr], ytr,
                               [rows[i][feat_key] for i in te], feats)
            t = A.conformal_threshold(s_ca, np.array([rows[i]["acts"][a][0] for i in ca]),
                                      alpha, delta)
            adm = s_te <= t
            gb = [rows[i]["acts"][a][1] if adm[j] else FULL_BITS for j, i in enumerate(te)]
            gate_bits[a].append(float(np.mean(gb)))
            broke = [rows[i]["acts"][a][0] for j, i in enumerate(te) if adm[j]]
            gate_brk[a].append(float(np.mean(broke)) if broke else 0.0)
        oracle_bits.append(float(np.mean(
            [min([rows[i]["acts"][a][1] for a in action_names if rows[i]["acts"][a][0] == 0.0]
                 + [FULL_BITS]) for i in te])))
    return {"bits": np.array(bits_f), "brk": np.array(brk_f), "hist": hist,
            "fam_hist": fam_hist, "gate_bits": {a: np.array(v) for a, v in gate_bits.items()},
            "gate_brk": {a: np.array(v) for a, v in gate_brk.items()},
            "oracle_bits": np.array(oracle_bits)}


def paired(a, b):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
    return float(d.mean()), float(se)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def make_selftest(n=500, seed=0, dominated=False):
    """Type A (60%): cheap action safe. Type B (40%): cheap breaks, mid action safe
    (unless `dominated`, where cheap is safe for everyone -> no dispatcher win exists).
    Feature 'top1' separates types; 'eff_page_count' is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        is_b = rng.random() < 0.4
        top1 = rng.normal(0.2 if is_b else 0.8, 0.05)
        cheap_break = 0.0 if (dominated or not is_b) else (0.0 if rng.random() < 0.05 else 1.0)
        mid_break = 0.0 if rng.random() < 0.98 else 1.0
        feats = {"entropy": rng.normal(3, .3), "eff_page_count": rng.normal(20, 3),
                 "top1": float(np.clip(top1, 0, 1)), "top5": float(np.clip(top1 + .1, 0, 1)),
                 "gini": rng.random(), "logP": 7.0, "family": "B" if is_b else "A"}
        rows.append({"feats": feats, "family": feats["family"], "seed": 0,
                     "acts": {"cheap": (cheap_break, 0.0625), "mid": (mid_break, 0.27)}})
    return rows


def report(res, label, alpha):
    bm, bs = res["bits"].mean(), 2 * res["bits"].std(ddof=1) / np.sqrt(len(res["bits"]))
    print(f"\n--- {label} ---")
    print(f"mean bits = {bm:.3f} ± {bs:.3f}   break|lossy = {res['brk'].mean():.3f} "
          f"(alpha={alpha})   oracle bits = {res['oracle_bits'].mean():.3f}")
    tot = sum(res["hist"].values())
    print("actions: " + "  ".join(f"{a}:{c/tot:.0%}" for a, c in res["hist"].most_common()))
    for fam, h in sorted(res["fam_hist"].items()):
        t = sum(h.values())
        print(f"   {fam:>15}: " + "  ".join(f"{a}:{c/t:.0%}" for a, c in h.most_common(3)))
    ok_gates = {a: v for a, v in res["gate_bits"].items()
                if res["gate_brk"][a].mean() <= alpha + 0.02}
    if ok_gates:
        best = min(ok_gates, key=lambda a: ok_gates[a].mean())
        gm = ok_gates[best].mean()
        gain, se = paired(res["gate_bits"][best], res["bits"])
        print(f"best two-action gate = admit[{best}]-else-full: bits {gm:.3f} "
              f"(coverage ok). Dispatcher saves {gain:+.3f} bits ({gain/gm:+.0%}) ± {2*se:.3f} (2SE)")
        return gain, se, gm
    print("no two-action gate held coverage — dispatcher unopposed")
    return float("nan"), float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s23", default="results/exp022_7b/s23.jsonl")
    ap.add_argument("--proxy", default="results/exp021_proxy_0p5b/proxy_Qwen2.5-0.5B-Instruct.jsonl")
    ap.add_argument("--actions", nargs="+",
                    default=["evict@0.0625", "evict@0.125", "evict@0.25", "quant4b", "quant8b"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("[selftest] two-type population; only the dispatcher can give B the mid action")
        rows = make_selftest()
        res = evaluate(rows, ["cheap", "mid"], "feats", A.FEATURES, args.alpha, args.delta, 20)
        gain, se, gm = report(res, "selftest/dispatch", args.alpha)
        assert res["brk"].mean() <= args.alpha + 0.02, "selftest FAIL: coverage"
        assert gain > 3 * se and gain / gm > 0.10, "selftest FAIL: planted dispatch win not found"
        rows = make_selftest(dominated=True)
        res = evaluate(rows, ["cheap", "mid"], "feats", A.FEATURES, args.alpha, args.delta, 20)
        gain, se, gm = report(res, "selftest/negative-control (cheap dominates)", args.alpha)
        assert not (gain > 3 * se and gain / gm > 0.10), "selftest FAIL: false dispatch win"
        print("\n[selftest] PASS: win detected when planted, absent when dominated.")
        return

    rows = load_matrix(args.s23, args.proxy, set(args.actions))
    rows = [r for r in rows if all(a in r["acts"] for a in args.actions)]
    n_prox = sum(1 for r in rows if "feats_proxy" in r)
    print(f"[load] {len(rows)} accepted contexts; proxy features on {n_prox}")
    print(f"[load] families: {Counter(r['family'] for r in rows)}")

    results = {}
    for label, key, feats in (("dispatch/target-feats", "feats", A.FEATURES),
                              ("dispatch/entropy-only", "feats", A.ENTROPY_ONLY)):
        results[label] = evaluate(rows, args.actions, key, feats, args.alpha, args.delta, args.folds)
    rows_p = [r for r in rows if "feats_proxy" in r]
    if len(rows_p) >= 100:
        results["dispatch/PROXY-feats (0.5B, pre-prefill)"] = evaluate(
            rows_p, args.actions, "feats_proxy", A.FEATURES, args.alpha, args.delta, args.folds)

    verdict_gain = verdict_se = verdict_gm = None
    for label, res in results.items():
        gain, se, gm = report(res, label, args.alpha)
        if label == "dispatch/target-feats":
            verdict_gain, verdict_se, verdict_gm = gain, se, gm

    print("\n=== VERDICT (target-feats dispatcher vs best two-action gate) ===")
    if verdict_gain is None or np.isnan(verdict_gain):
        print("NOT EVALUABLE: no gate baseline held coverage.")
        return
    rel = verdict_gain / verdict_gm
    covered = results["dispatch/target-feats"]["brk"].mean() <= args.alpha + 0.02
    if not covered:
        print(f"UNSOUND: dispatcher break|lossy exceeds alpha — numbers untrusted.")
    elif rel >= 0.25 and verdict_gain > 3 * verdict_se:
        print(f"WIN: dispatcher cuts fleet KV bits {rel:.0%} below the best single-action "
              f"gate at the same guarantee ({verdict_gain:+.3f} ± {2*verdict_se:.3f}). "
              f"Wire into serving (Cut 2) — and see the PROXY arm for pre-prefill routing.")
    elif rel >= 0.10 and verdict_gain > 3 * verdict_se:
        print(f"MODEST: {rel:.0%} bits reduction — real but check the action histogram; "
              f"a two-action gate captures most of the value on this mix.")
    else:
        print(f"KILL: {rel:.0%} — a single well-chosen action + refuse was enough on this "
              f"traffic mix; the dispatcher is not worth its complexity here.")


if __name__ == "__main__":
    main()
