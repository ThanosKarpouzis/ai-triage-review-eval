#!/usr/bin/env python3
"""How many labelled examples before a confidence threshold keeps its error promise? Standard library only.

Rules were fixed before anything was run; see "Check whether a lane can be trusted" in README.md.

  python3 error_budget.py --check              # reproduce the original-split figures
  python3 error_budget.py --out DIR            # full run (random draws, then the time-ordered arm)
  python3 error_budget.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data --out DIR
                                               # the same run on one repository's own results
  python3 error_budget.py --report DIR         # readable summary of a finished run, with the group of each case

- A pool is every item for one model on one decision in one data set (calibration and evaluation merged).
  A failed item counts as wrong at confidence 0.
- Naive rule (from analyze.py): the lowest threshold at which the error among the sample items at or above
  it is at most the target, with at least 20 items kept.
- Safety-margin rule: the same scan, but a threshold is accepted only when the upper end of a one-sided
  90% exact binomial interval on the kept items' error is at or under the budget.
- A sample size N is used for a pool only when at least 300 items remain; a pool qualifies only when the
  naive rule on the whole pool keeps at least 10% of items.
"""
import argparse, json, math, random, statistics, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, REPS = 20261010, 2000
SIZES = [25, 50, 100, 200, 400]
MIN_KEPT, MIN_REST, MIN_POOL_COVERAGE = 20, 300, 0.10
TARGETS = [0.05, 0.04, 0.03, 0.02, 0.01]
CUTS, MIN_AFTER = [0.3, 0.4, 0.5, 0.6, 0.7], 150
SOURCES = [  # (data set, results files, data files carrying created_at)
    ("benchmark", ["results/haiku.jsonl", "results/jev.jsonl", "results/nli.jsonl", "results/jev-decomposed.jsonl",
                   "results-sample/opus.jsonl"], []),
    ("terraform", ["kit/hashicorp__terraform/results/haiku.jsonl", "kit/hashicorp__terraform/results/jev.jsonl",
                   "kit/hashicorp__terraform/results/nli.jsonl", "kit/hashicorp__terraform/results/jev-decomposed.jsonl"],
     ["kit/hashicorp__terraform/data/issues.jsonl", "kit/hashicorp__terraform/data/prs.jsonl"]),
    ("metabase", ["kit/metabase__metabase/results/haiku.jsonl", "kit/metabase__metabase/results/jev.jsonl",
                  "kit/metabase__metabase/results/opus.jsonl",
                  "kit/metabase__metabase/results/derived-data-all/jev-area-team.jsonl",
                  "kit/metabase__metabase/results/derived-data-all/jev-checklist-priority.jsonl"],
     ["kit/metabase__metabase/data-all/issues.jsonl", "kit/metabase__metabase/data-all/taxonomy_type.jsonl",
      "kit/metabase__metabase/data-all/taxonomy_priority.jsonl", "kit/metabase__metabase/data-all/taxonomy_team.jsonl"]),
]


def load_pools(sources=None):
    pools = {}
    for dataset, files, datafiles in (sources or SOURCES):
        dates = {}
        for df in datafiles:
            for line in (HERE / df).open(encoding="utf-8"):
                r = json.loads(line)
                dates[r["id"]] = r.get("created_at")
        for f in files:
            latest = {}
            for line in (HERE / f).open(encoding="utf-8"):
                r = json.loads(line)
                latest[r["id"]] = r  # the last attempt for an item is the one that counts
            for r in latest.values():
                ok = bool(r.get("ok"))
                item = {"id": r["id"], "split": r["split"], "date": dates.get(r["id"]),
                        "conf": float(r.get("confidence") or 0.0) if ok else 0.0,
                        "wrong": 0 if ok and r.get("pred") == r["gold"] else 1}
                pools.setdefault((dataset, r["task"], r["model"]), []).append(item)
    return pools


def max_wrong_table(n_max, budget, alpha=0.10):
    """allowed[k] = most errors among k kept items for which the exact one-sided 90% upper bound is <= budget; -1 if none."""
    allowed = [-1] * (n_max + 1)
    for k in range(1, n_max + 1):
        cdf, w, pmf = 0.0, -1, (1 - budget) ** k
        for x in range(0, k + 1):
            cdf += pmf
            if cdf <= alpha:
                w = x
            else:
                break
            pmf *= (k - x) / (x + 1) * budget / (1 - budget)
        allowed[k] = w
    return allowed


class Pool:
    def __init__(self, items):
        self.items = sorted(items, key=lambda r: -r["conf"])
        self.n = len(self.items)
        self.conf = [r["conf"] for r in self.items]
        self.wrong = [r["wrong"] for r in self.items]
        self.cum = [0]
        for w in self.wrong:
            self.cum.append(self.cum[-1] + w)
        self.last = {}  # confidence value -> number of pool items at or above it
        for i, c in enumerate(self.conf):
            self.last[c] = i + 1

    def choose(self, idx, accept):
        """idx: sorted positions of the sample. Returns (k kept in sample, wrong in sample, threshold) or None."""
        best, wrong = None, 0
        for k, i in enumerate(idx, 1):
            wrong += self.wrong[i]
            if k < len(idx) and self.conf[idx[k]] == self.conf[i]:
                continue  # only cut between distinct confidence values
            if k >= MIN_KEPT and accept(k, wrong):
                best = (k, wrong, self.conf[i])
        return best

    def rest(self, choice):
        """Coverage and error on the pool items outside the sample, for a chosen threshold."""
        k, wrong, t = choice
        kept = self.last[t] - k
        bad = self.cum[self.last[t]] - wrong
        return kept, bad


def rules(budget, n_max):
    allowed = max_wrong_table(n_max, budget)
    out = {}
    if budget == 0.05:
        for t in TARGETS:
            out[f"aim_{int(round(t * 100))}"] = (lambda k, w, t=t: w / k <= t)
    else:
        out[f"aim_{int(round(budget * 100))}"] = (lambda k, w: w / k <= budget)
        low = 0.6 * budget  # "aim lower" for other budgets: 60% of the budget, as 3% is of 5%
        out[f"aim_{int(round(low * 100))}"] = (lambda k, w: w / k <= low)
    out["safety_margin"] = (lambda k, w: w <= allowed[k])
    return out


def quant(xs, q):
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else None


def summarise(cases, n_rest, budget, pool_cov):
    """cases: list of None (no lane) or (kept, bad) on the unseen items."""
    opened = [c for c in cases if c is not None and c[0] > 0]
    errs = [b / k for k, b in opened]
    covs = [k / n_rest for k, b in opened]
    return {"cases": len(cases), "opened": len(opened) / len(cases) if cases else None,
            "broke": sum(e > budget for e in errs) / len(errs) if errs else None,
            "err_median": statistics.median(errs) if errs else None, "err_p90": quant(errs, 0.9),
            "cov_median": statistics.median(covs) if covs else None,
            "cov_vs_pool": (statistics.median(covs) / pool_cov) if covs and pool_cov else None}


def group_of(cov):
    """Added 10 October 2026 after the first run, for the confirmation runs (see README.md)."""
    return "room" if cov >= 0.99 else "edge" if cov >= 0.40 else "sliver" if cov >= MIN_POOL_COVERAGE else "none"


def whole_pool(pool, budget):
    c = pool.choose(list(range(pool.n)), lambda k, w: w / k <= budget)
    return (c[0] / pool.n, c[1] / c[0], c[2]) if c else (0.0, None, None)


def random_arm(key, pool, budget, reps):
    rng = random.Random(SEED + zlib.crc32("|".join(key).encode()) + int(budget * 100))
    rs = rules(budget, max(SIZES))
    pool_cov = whole_pool(pool, budget)[0]
    out = {}
    for n in SIZES:
        if pool.n - n < MIN_REST:
            continue
        cases = {name: [] for name in rs}
        for _ in range(reps):
            idx = sorted(rng.sample(range(pool.n), n))
            for name, accept in rs.items():
                c = pool.choose(idx, accept)
                cases[name].append(pool.rest(c) if c else None)
        out[n] = {name: summarise(v, pool.n - n, budget, pool_cov) for name, v in cases.items()}
    return out


def time_arm(pool, budget):
    """Sample = the N items just before a cut-off in date order; test = everything after."""
    dated = [r for r in pool.items if r["date"]]
    if len(dated) < pool.n:
        return None
    dated.sort(key=lambda r: (r["date"], r["id"]))
    rs = rules(budget, max(SIZES))
    rs = {k: v for k, v in rs.items() if k in (f"aim_{int(round(budget * 100))}", "safety_margin")}
    out = {}
    for n in SIZES:
        cases = {name: [] for name in rs}
        thirds = {name: [[0, 0], [0, 0], [0, 0]] for name in rs}
        used = 0
        for cut in CUTS:
            c0 = int(cut * len(dated))
            if c0 < n or len(dated) - c0 < MIN_AFTER:
                continue
            used += 1
            sample = sorted(dated[c0 - n:c0], key=lambda r: -r["conf"])
            after = dated[c0:]
            sp = Pool(sample)
            for name, accept in rs.items():
                c = sp.choose(list(range(sp.n)), accept)
                if not c:
                    cases[name].append(None)
                    continue
                t = c[2]
                kept = [(j, r) for j, r in enumerate(after) if r["conf"] >= t]
                cases[name].append((len(kept), sum(r["wrong"] for _, r in kept)))
                for j, r in kept:
                    b = min(3 * j // len(after), 2)
                    thirds[name][b][0] += 1
                    thirds[name][b][1] += r["wrong"]
        if used:
            out[n] = {}
            for name in rs:
                s = summarise(cases[name], 1, budget, None)
                opened = [c for c in cases[name] if c and c[0] > 0]
                s["cov_median"] = s["cov_vs_pool"] = None
                s["kept_total"] = sum(k for k, b in opened)
                s["err_pooled"] = (sum(b for k, b in opened) / s["kept_total"]) if opened and s["kept_total"] else None
                s["err_by_third"] = [(b / k if k else None) for k, b in thirds[name]]
                s["kept_by_third"] = [k for k, b in thirds[name]]
                out[n][name] = s
    return out or None


def check(pools):
    """Original-split figures, to compare with the published summary.json files."""
    rows = []
    for key, items in sorted(pools.items()):
        ca = Pool([r for r in items if r["split"] == "calib"])
        ev = [r for r in items if r["split"] == "eval"]
        for budget in (0.05, 0.15):
            c = ca.choose(list(range(ca.n)), lambda k, w: w / k <= budget) if ca.n else None
            if c:
                kept = [r for r in ev if r["conf"] >= c[2]]
                cov, err = len(kept) / len(ev), (sum(r["wrong"] for r in kept) / len(kept) if kept else None)
            else:
                cov, err = 0.0, None
            rows.append({"pool": key, "budget": budget, "n_calib": ca.n, "n_eval": len(ev),
                         "threshold": c[2] if c else None, "coverage": cov, "error": err})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--only", default=None, help="data set name, to run in parts")
    ap.add_argument("--results-dir", default=None, help="one repository's results folder (from run.py)")
    ap.add_argument("--data-dir", default=None, help="its data folder, for the dates used by the time-ordered arm")
    ap.add_argument("--report", default=None, help="print a summary of a finished run in this folder")
    a = ap.parse_args()
    if a.report:
        return report(Path(a.report))
    sources = None
    if a.results_dir:
        rd = Path(a.results_dir).resolve()
        dd = sorted(str(f) for f in Path(a.data_dir).resolve().glob("*.jsonl")) if a.data_dir else []
        dd = [f for f in dd if "label_history" not in f]
        sources = [(rd.parent.name, sorted(str(f) for f in rd.glob("*.jsonl")), dd)]
    if sources is None and not all((HERE / f).exists() for _, files, _ in SOURCES for f in files):
        raise SystemExit("Point this at a results folder from run.py: --results-dir kit/OWNER__REPO/results "
                         "--data-dir kit/OWNER__REPO/data --out DIR (the defaults are the write-up's own results).")
    pools = load_pools(sources)
    if a.check:
        for r in check(pools):
            e = "n/a" if r["error"] is None else f"{100 * r['error']:.1f}%"
            print(f"{' / '.join(r['pool']):<75} budget {int(r['budget'] * 100):>2}%  calib {r['n_calib']:>4}  eval {r['n_eval']:>4}  "
                  f"threshold {r['threshold']}  coverage {100 * r['coverage']:.1f}%  error {e}")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for key, items in sorted(pools.items()):
        if a.only and key[0] != a.only:
            continue
        pool = Pool(items)
        res = {"pool": list(key), "n": pool.n, "accuracy": 1 - pool.cum[-1] / pool.n, "reps": a.reps,
               "distinct_confidence_values": len(set(pool.conf))}
        for budget in (0.05, 0.15):
            cov, err, t = whole_pool(pool, budget)
            tag = f"budget_{int(budget * 100)}"
            res[tag] = {"whole_pool_coverage": cov, "whole_pool_error": err, "whole_pool_threshold": t, "group": group_of(cov),
                        "qualifies": cov >= MIN_POOL_COVERAGE and pool.n - min(SIZES) >= MIN_REST}
            if res[tag]["qualifies"]:
                res[tag]["random"] = random_arm(key, pool, budget, a.reps)
                if budget == 0.05:
                    res[tag]["time_ordered"] = time_arm(pool, budget)
        name = "__".join(key).replace("/", "_").replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(key, pool.n, {b: res[b]["qualifies"] for b in ("budget_5", "budget_15")}, flush=True)


def report(folder):
    """Markdown summary of a run, plus the checks C1 to C3 fixed before the confirmation runs (see README.md)."""
    pc = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
    verdicts = {"C1": [], "C2": [], "C3": []}
    for f in sorted(folder.glob("*.json")):
        d = json.loads(f.read_text())
        for b in (5, 15):
            B = d[f"budget_{b}"]
            if not B.get("qualifies") or not B.get("random"):
                print(f"- {' / '.join(d['pool'])}, {b}% budget: does not qualify ({d['n']} items, whole-pool lane {pc(B['whole_pool_coverage'])})")
                continue
            R, g = B["random"], B["group"]
            naive, low = f"aim_{b}", f"aim_{int(round(0.6 * b))}"
            ns = sorted(R, key=int)
            print(f"\n### {' / '.join(d['pool'])}, {b}% budget: {g}\n")
            print(f"{d['n']} items, {pc(1 - d['accuracy'])} wrong overall, whole-pool lane {pc(B['whole_pool_coverage'])}, "
                  f"{d['distinct_confidence_values']} distinct confidence values.\n")
            print(f"| Labels | Plain rule: lane opens | breaks promise | typical error | worst tenth | Aim lower ({low[4:]}%): breaks | coverage kept | Safety margin: opens | breaks | coverage kept |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for n in ns:
                x, y, z = R[n][naive], R[n][low], R[n]["safety_margin"]
                e = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
                print(f"| {n} | {pc(x['opened'])} | {pc(x['broke'])} | {e(x['err_median'])} | {e(x['err_p90'])} | {pc(y['broke'])} | {pc(y['cov_vs_pool'])} "
                      f"| {pc(z['opened'])} | {pc(z['broke'])} | {pc(z['cov_vs_pool'])} |")
            T = B.get("time_ordered")
            if T:
                for n in sorted(T, key=int):
                    x = T[n][naive]
                    print(f"\nTime-ordered, {n} labels: {x['cases']} cut-offs, lane opened in {pc(x['opened'])}, broke in {pc(x['broke'])} of those, "
                          f"error by third of the later period {[None if v is None else round(100 * v, 1) for v in x['err_by_third']]}.")
            name = f"{' / '.join(d['pool'])} at {b}%"
            if g == "room":
                rows = [R[n][naive]["broke"] for n in ns if int(n) >= 50]
                verdicts["C1"].append((name, bool(rows) and all(v is not None and v <= 0.10 for v in rows) if rows else None))
            elif g == "edge":
                big = [n for n in ns if int(n) >= 200]
                v = R[big[-1]][naive]["broke"] if big else None
                verdicts["C2"].append((name, None if v is None else 0.35 <= v <= 0.65))
            elif g == "sliver":
                x = R.get("50", {}).get(naive)
                verdicts["C3"].append((name, None if x is None else (x["opened"] < 0.5 and (x["broke"] is None or x["broke"] > 0.70))))
    print("\n## Confirmation checks\n")
    for c, rows in verdicts.items():
        ok = [n for n, v in rows if v is True]; no = [n for n, v in rows if v is False]; na = [n for n, v in rows if v is None]
        print(f"- {c}: held in {len(ok)} of {len(ok) + len(no)} assessable cases ({len(na)} not assessable). Failed: {no or 'none'}")


if __name__ == "__main__":
    main()
