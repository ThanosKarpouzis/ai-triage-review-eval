#!/usr/bin/env python3
"""Combine Jev's six narrow code-review checks into one decision. Standard library only.

Pre-registered 3 October 2026, before any check result existed:
- Features: the logit of each check's probability (clipped to 0.001..0.999), standardised on the calibration split.
- Model: logistic regression with an L2 penalty of 1.0, fitted on the calibration split of each task.
- Calibration items get out-of-fold probabilities from 5-fold cross-validation (fixed seed), so the threshold
  analyze.py chooses on them is not fitted to the same items. Evaluation items are scored by the model fitted
  on the whole calibration split.
- Output: results/jev-decomposed.jsonl in the same format as every other model, so analyze.py scores it unchanged.
- Secondary, pre-registered: a "safe to skip review" lane. On the calibration split (out-of-fold), find the
  largest share of changes, taken from the lowest probability of needing a comment upwards, in which at most 5%
  needed a comment (at least 20 items). Apply that threshold to the evaluation split and report coverage and the
  real share that needed a comment.

  python3 combine_checks.py                                   # benchmark
  python3 combine_checks.py --results-dir kit/OWNER__REPO/results
"""
import argparse, json, math, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, FOLDS, L2, ITERS, LR = 20261003, 5, 1.0, 3000, 0.1
MODEL_ID = "typesafe-ai/jev (decomposed)"


def logit(p):
    p = min(max(p, 0.001), 0.999)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z > -50 else 0.0


def fit(X, y):
    n, d = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(d)]
    sd = [max(1e-6, (sum((r[j] - mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]
    w, b = [0.0] * d, 0.0
    for _ in range(ITERS):
        gw, gb = [L2 * wj / n for wj in w], 0.0
        for z, t in zip(Z, y):
            e = sigmoid(b + sum(wj * zj for wj, zj in zip(w, z))) - t
            gb += e / n
            for j in range(d):
                gw[j] += e * z[j] / n
        b -= LR * gb
        w = [wj - LR * g for wj, g in zip(w, gw)]
    return lambda r: sigmoid(b + sum(wj * (r[j] - mu[j]) / sd[j] for j, wj in enumerate(w))), dict(zip(range(d), w)), b


def skip_lane(cal, ev):
    order = sorted(cal, key=lambda r: r["p"])
    best, yes = None, 0
    for k, r in enumerate(order, 1):
        yes += r["gold"] == "yes"
        nxt = order[k]["p"] if k < len(order) else None
        if nxt == r["p"]:
            continue
        if k >= 20 and yes / k <= 0.05:
            best = r["p"]
    if best is None:
        return "no skip threshold found on the calibration split"
    kept = [r for r in ev if r["p"] <= best]
    miss = sum(r["gold"] == "yes" for r in kept)
    return (f"skip when P(comment) <= {best:.3f}: {len(kept)} of {len(ev)} evaluation changes "
            f"({100 * len(kept) / len(ev):.1f}%), of which {miss} needed a comment "
            f"({100 * miss / max(1, len(kept)):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    res = HERE / a.results_dir
    src = res / "checks" / "jev_checks.jsonl"
    latest = {}
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        latest[r["id"]] = r
    rows = [r for r in latest.values() if r.get("ok")]
    failed = [r for r in latest.values() if not r.get("ok")]
    names = sorted(rows[0]["probs"]) if rows else []
    out, report = [], []
    for task in sorted({r["task"] for r in rows}):
        cal = [r for r in rows if r["task"] == task and r["split"] == "calib"]
        ev = [r for r in rows if r["task"] == task and r["split"] == "eval"]
        feats = lambda r: [logit(r["probs"][k]) for k in names]
        y = lambda rs: [1 if r["gold"] == "yes" else 0 for r in rs]
        rng = random.Random(SEED)
        idx = list(range(len(cal)))
        rng.shuffle(idx)
        for f in range(FOLDS):
            test = set(idx[f::FOLDS])
            train = [cal[i] for i in idx if i not in test]
            predict, _, _ = fit([feats(r) for r in train], y(train))
            for i in test:
                cal[i]["p"] = predict(feats(cal[i]))
        predict, w, b = fit([feats(r) for r in cal], y(cal))
        for r in ev:
            r["p"] = predict(feats(r))
        report.append(f"\n{task}: {len(cal)} calibration, {len(ev)} evaluation changes")
        report.append("  weights (standardised; positive means more likely to need a comment): "
                      + ", ".join(f"{names[j]} {w[j]:+.2f}" for j in range(len(names))) + f", intercept {b:+.2f}")
        report.append("  " + skip_lane(cal, ev))
        for r in cal + ev:
            p = r["p"]
            out.append({"id": r["id"], "task": task, "split": r["split"], "gold": r["gold"], "model": MODEL_ID,
                        "ok": True, "pred": "yes" if p >= 0.5 else "no", "confidence": max(p, 1 - p),
                        "probs": {"yes": p, "no": 1 - p}, "cost_usd": r.get("cost_usd", 0.0),
                        "latency_ms": r.get("latency_ms"), "provider": r.get("provider"), "ts": r.get("ts")})
    for r in failed:  # failures count as wrong, as pre-registered
        out.append({"id": r["id"], "task": r["task"], "split": r["split"], "gold": r["gold"], "model": MODEL_ID,
                    "ok": False, "final": True, "error": r.get("error"), "cost_usd": 0.0, "latency_ms": 0.0})
    (res / "jev-decomposed.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"Combined {len(rows)} changes ({len(failed)} failed) into {res.relative_to(HERE)}/jev-decomposed.jsonl")
    print("\n".join(report))
    print(f"\nNext: python3 analyze.py --results-dir {a.results_dir}")


if __name__ == "__main__":
    main()
