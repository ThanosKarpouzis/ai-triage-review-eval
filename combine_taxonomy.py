#!/usr/bin/env python3
"""Turn Jev's narrow answers into label decisions for the taxonomy follow-up tests. Standard library only.

Pre-registered 4 October 2026, before any follow-up call (see README.md):
- Checklist (families with checks in taxonomy.json, e.g. priority): features are the logit of each check's probability
  (clipped to 0.001..0.999), standardised on the calibration split; a multinomial logistic regression with an L2
  penalty of 1.0 is fitted on the calibration split, over the labels present there.
- Area, then label (the family named in taxonomy.json "area", e.g. team): for each area a and label t, weight(a, t) =
  (sum over calibration issues with label t of Jev's probability for area a + 1) / (sum over all calibration issues of
  Jev's probability for area a + number of labels in the family). An issue's label probabilities are its area
  probabilities times that table, normalised.
- Both: calibration items get out-of-fold predictions from 5-fold cross-validation with a fixed seed; evaluation items
  use the fit on the whole calibration split. Only items in the given data folder are used, so the headline (data) and
  the secondary (data-all) are fitted separately. Failures count as wrong.
- Output: results/derived-<data folder>/*.jsonl in the usual results format, read by analyze_taxonomy.py.

  python3 combine_taxonomy.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data
"""
import argparse, json, math, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, FOLDS, L2, ITERS, LR = 20261004, 5, 1.0, 3000, 0.1
CHECKLIST_ID, AREA_ID = "typesafe-ai/jev (checklist)", "typesafe-ai/jev (area, then team)"


def logit(p):
    p = min(max(p, 0.001), 0.999)
    return math.log(p / (1 - p))


def softmax(zs):
    m = max(zs)
    e = [math.exp(z - m) for z in zs]
    s = sum(e)
    return [x / s for x in e]


def fit_multinomial(X, y, classes):
    n, d, k = len(X), len(X[0]), len(classes)
    mu = [sum(r[j] for r in X) / n for j in range(d)]
    sd = [max(1e-6, (sum((r[j] - mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]
    W, b = [[0.0] * d for _ in range(k)], [0.0] * k
    yi = [classes.index(t) for t in y]
    for _ in range(ITERS):
        gW = [[L2 * W[c][j] / n for j in range(d)] for c in range(k)]
        gb = [0.0] * k
        for z, t in zip(Z, yi):
            p = softmax([b[c] + sum(W[c][j] * z[j] for j in range(d)) for c in range(k)])
            for c in range(k):
                e = (p[c] - (c == t)) / n
                gb[c] += e
                for j in range(d):
                    gW[c][j] += e * z[j]
        for c in range(k):
            b[c] -= LR * gb[c]
            W[c] = [W[c][j] - LR * gW[c][j] for j in range(d)]

    def predict(r):
        z = [(r[j] - mu[j]) / sd[j] for j in range(d)]
        return dict(zip(classes, softmax([b[c] + sum(W[c][j] * z[j] for j in range(d)) for c in range(k)])))
    return predict


def fit_area(cal, labels):
    areas = sorted({a for r in cal for a in r["probs"]})
    num = {a: {t: 1.0 for t in labels} for a in areas}
    den = {a: float(len(labels)) for a in areas}
    for r in cal:
        for a, p in r["probs"].items():
            num[a][r["gold"]] = num[a].get(r["gold"], 1.0) + p
            den[a] += p

    def predict(r):
        out = {t: sum(p * num[a][t] / den[a] for a, p in r["probs"].items() if a in num) for t in labels}
        s = sum(out.values()) or 1.0
        return {t: v / s for t, v in out.items()}
    return predict


def out_of_fold(cal, ev, make):
    rng = random.Random(SEED)
    idx = list(range(len(cal)))
    rng.shuffle(idx)
    for f in range(FOLDS):
        test = set(idx[f::FOLDS])
        predict = make([cal[i] for i in idx if i not in test])
        for i in test:
            cal[i]["p"] = predict(cal[i])
    predict = make(cal)
    for r in ev:
        r["p"] = predict(r)


def load(path, ids):
    latest = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in ids:
                latest[r["id"]] = r
    return list(latest.values())


def emit(rows, failed, model_id):
    out = []
    for r in rows:
        p = r["p"]
        best = max(p, key=p.get)
        out.append({"id": r["id"], "task": r["task"], "split": r["split"], "gold": r["gold"], "model": model_id,
                    "ok": True, "pred": best, "confidence": p[best], "probs": p, "cost_usd": r.get("cost_usd", 0.0),
                    "latency_ms": r.get("latency_ms") or 0.0, "provider": r.get("provider"), "ts": r.get("ts")})
    for r in failed:
        out.append({"id": r["id"], "task": r["task"], "split": r["split"], "gold": r["gold"], "model": model_id,
                    "ok": False, "final": True, "error": r.get("error"), "cost_usd": 0.0, "latency_ms": 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    res, data = HERE / a.results_dir, HERE / a.data_dir
    tax = json.loads((data / "taxonomy.json").read_text(encoding="utf-8"))
    out_dir = res / f"derived-{data.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for fam in tax.get("checks") or {}:
        task = f"own_{fam}"
        ids = {json.loads(l)["id"] for l in (data / f"taxonomy_{fam}.jsonl").open(encoding="utf-8")}
        rows = load(res / "checks" / "jev_checks.jsonl", ids)
        ok, failed = [r for r in rows if r.get("ok") and r["task"] == task], [r for r in rows if not r.get("ok")]
        if not ok:
            report.append(f"{fam} checklist: no answers yet")
            continue
        names = sorted(ok[0]["probs"])
        feats = lambda r: [logit(r["probs"][k]) for k in names]
        cal = [r for r in ok if r["split"] == "calib"]
        ev = [r for r in ok if r["split"] == "eval"]
        classes = sorted({r["gold"] for r in cal})

        def make(train):
            pred = fit_multinomial([feats(r) for r in train], [r["gold"] for r in train], classes)
            return lambda r: pred(feats(r))
        out_of_fold(cal, ev, make)
        missing = len(ids) - len(ok) - len(failed)
        (out_dir / f"jev-checklist-{fam}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in emit(cal + ev, failed, CHECKLIST_ID)))
        report.append(f"{fam} checklist: {len(cal)} calibration, {len(ev)} evaluation, {len(failed)} failed, "
                      f"{missing} not yet answered; labels learned: {', '.join(classes)}")

    if tax.get("area"):
        fam = tax["area"]["for_family"]
        task = f"own_{fam}"
        labels = [o["name"] for o in tax["families"][fam]["options"]]
        ids = {json.loads(l)["id"] for l in (data / f"taxonomy_{fam}.jsonl").open(encoding="utf-8")}
        rows = load(res / "checks" / "jev_area.jsonl", ids)
        ok, failed = [r for r in rows if r.get("ok")], [r for r in rows if not r.get("ok")]
        if ok:
            cal = [r for r in ok if r["split"] == "calib"]
            ev = [r for r in ok if r["split"] == "eval"]
            out_of_fold(cal, ev, lambda train: fit_area(train, labels))
            (out_dir / f"jev-area-{fam}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in emit(cal + ev, failed, AREA_ID)))
            report.append(f"{fam} via area: {len(cal)} calibration, {len(ev)} evaluation, {len(failed)} failed, "
                          f"{len(ids) - len(ok) - len(failed)} not yet answered")
        else:
            report.append(f"{fam} via area: no answers yet")
    print("\n".join(report))
    print(f"Wrote {out_dir}. Next: python3 analyze_taxonomy.py --results-dir {a.results_dir} --data-dir {a.data_dir}")


if __name__ == "__main__":
    main()
