#!/usr/bin/env python3
"""Score the Jev study. Standard library only. Metrics were fixed before any real results existed.

  python3 analyze.py                 # reads results/*.jsonl, writes results/summary.md and results/summary.json

Pre-registered rules
- Scoring uses the evaluation split only. The calibration split is used for one thing: choosing
  each model's confidence threshold for the 5% error budget.
- An item that failed (HTTP error or unparseable answer) counts as wrong, with confidence 0.
- Coverage at 5% error: on the calibration split, find the lowest confidence threshold at which
  the error rate among items at or above it is at most 5%. Apply that threshold unchanged to the
  evaluation split and report the share of items it keeps (coverage) and their actual error rate.
- Code review is also reported at a realistic base rate: 25% of changes draw a comment, versus
  50% in the balanced benchmark. This reweights the same results; nothing is re-run.
- Calibration: expected calibration error over 10 equal-width confidence bins.
- Cost per 1,000 decisions: mean of the per-call cost the gateway reported (or list price).
- Lanes (added 26 September 2026, before any results were examined): the 5% threshold defines the
  "no human" lane; a second threshold chosen the same way at 15% error defines "review later"; the
  rest is "human first". Only the 5% lane was part of the original pre-registration.

  python3 analyze.py --results-dir kit/<repo>/results    # the own-repo kit
"""
import json, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ERROR_BUDGET = 0.05
REVIEW_LATER_BUDGET = 0.15
REALISTIC_COMMENT_RATE = 0.25
MIN_KEPT_FOR_THRESHOLD = 20

PUBLISHED = {
    "issue_type": "SetFit (NLBSE'24 baseline): macro F1 0.827. Fine-tuned GPT-4o: macro F1 0.857.",
    "needs_comment": "CodeReviewer (Li et al. 2022): accuracy 73.9%, F1 71.5.",
}
LABELS = {"issue_type": ["bug", "feature", "question"], "needs_comment": ["yes", "no"], "pr_needs_changes": ["yes", "no"]}
NAMES = {"typesafe-ai/jev": "Jev", "anthropic/claude-haiku-4.5": "Claude Haiku 4.5", "anthropic/claude-opus-5.5": "Claude Opus 5.5",
         "anthropic/claude-opus-5": "Claude Opus 5", "anthropic/claude-opus-4.7": "Claude Opus 4.7",
         "moritzlaurer/deberta-v3-large-zeroshot-v2.0": "DeBERTa-v3 zero-shot (open, local)"}
TASK_NAMES = {"issue_type": "Issue triage (bug / feature / question)",
              "needs_comment": "Code review (does this change need a comment?)",
              "pr_needs_changes": "Pull requests (will a reviewer comment or request changes?)"}


def load():
    rows = []
    for f in sorted(RESULTS.glob("*.jsonl")):
        latest = {}
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            latest[r["id"]] = r  # the last attempt for an item is the one that counts
        rows += latest.values()
    for r in rows:
        if not r.get("ok"):
            r["pred"], r["confidence"] = None, 0.0
        r["correct"] = r.get("pred") == r["gold"]
    return rows


def weights(items, task, reweight):
    if task != "needs_comment" or not reweight:
        return [1.0] * len(items)
    pos = sum(1 for r in items if r["gold"] == "yes") / len(items)
    return [REALISTIC_COMMENT_RATE / pos if r["gold"] == "yes" else (1 - REALISTIC_COMMENT_RATE) / (1 - pos)
            for r in items]


def wmean(vals, w):
    return sum(v * x for v, x in zip(vals, w)) / sum(w) if sum(w) else float("nan")


def f1_scores(items, labels):
    out = {}
    for lab in labels:
        tp = sum(1 for r in items if r["pred"] == lab and r["gold"] == lab)
        fp = sum(1 for r in items if r["pred"] == lab and r["gold"] != lab)
        fn = sum(1 for r in items if r["pred"] != lab and r["gold"] == lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 2 * p * rc / (p + rc) if p + rc else 0.0
    return out


def choose_threshold(calib, w, budget=ERROR_BUDGET):
    """Lowest threshold whose kept set has weighted error <= budget; None if no threshold works."""
    order = sorted(range(len(calib)), key=lambda i: -calib[i]["confidence"])
    best, wrong, total = None, 0.0, 0.0
    for k, i in enumerate(order, 1):
        wrong += w[i] * (not calib[i]["correct"])
        total += w[i]
        nxt = calib[order[k]]["confidence"] if k < len(order) else -1
        if nxt == calib[i]["confidence"]:
            continue  # only cut between distinct confidence values
        if k >= MIN_KEPT_FOR_THRESHOLD and wrong / total <= budget:
            best = calib[i]["confidence"]
    return best


def coverage(evals, w, t):
    if t is None:
        return 0.0, None
    kept = [i for i, r in enumerate(evals) if r["confidence"] >= t]
    cov = sum(w[i] for i in kept) / sum(w)
    err = (sum(w[i] * (not evals[i]["correct"]) for i in kept) / sum(w[i] for i in kept)) if kept else None
    return cov, err


def ece(items, bins=10):
    b = defaultdict(list)
    for r in items:
        b[min(int(r["confidence"] * bins), bins - 1)].append(r)
    return sum(len(v) / len(items) * abs(statistics.mean(x["confidence"] for x in v)
                                        - statistics.mean(x["correct"] for x in v)) for v in b.values())


def score(rows, task, model, reweight=False):
    ev = [r for r in rows if r["task"] == task and r["model"] == model and r["split"] == "eval"]
    ca = [r for r in rows if r["task"] == task and r["model"] == model and r["split"] == "calib"]
    if not ev:
        return None
    w_ev, w_ca = weights(ev, task, reweight), weights(ca, task, reweight) if ca else []
    t = choose_threshold(ca, w_ca) if ca else None
    cov, err = coverage(ev, w_ev, t)
    t15 = choose_threshold(ca, w_ca, REVIEW_LATER_BUDGET) if ca else None
    cov15, err15 = coverage(ev, w_ev, t15)
    f1 = f1_scores(ev, LABELS[task])
    lat = sorted(r["latency_ms"] for r in ev if r.get("ok"))
    return {
        "n_eval": len(ev), "n_calib": len(ca), "failed": sum(1 for r in ev if not r.get("ok")),
        "accuracy": wmean([r["correct"] for r in ev], w_ev),
        "accuracy_answered": (wmean([r["correct"] for r in ev if r.get("ok")], [w for w, r in zip(w_ev, ev) if r.get("ok")])
                              if any(r.get("ok") for r in ev) else None),
        "macro_f1": statistics.mean(f1.values()), "f1_by_label": f1,
        "threshold": t, "coverage_at_budget": cov, "error_within_coverage": err,
        "coverage_at_15": cov15, "error_within_15": err15,
        "lanes": {"no_human": cov, "review_later": max(cov15 - cov, 0.0), "human_first": 1 - max(cov15, cov)},
        "ece": ece(ev),
        "cost_per_1000_usd": 1000 * statistics.mean(r.get("cost_usd") or 0 for r in ev),
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "latency_p90_ms": lat[int(len(lat) * 0.9)] if lat else None,
    }


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    if not RESULTS.exists() or not any(RESULTS.glob("*.jsonl")):
        raise SystemExit("No results yet. Run run.py first.")
    rows = load()
    models = sorted({r["model"] for r in rows}, key=lambda m: list(NAMES).index(m) if m in NAMES else 9)
    summary, md = {}, [f"# Jev study results\n\nError budget: {int(ERROR_BUDGET * 100)}%. "
                       "Evaluation split only; thresholds chosen on the calibration split.\n"]
    present = {r["task"] for r in rows}
    views = [(t, False) for t in LABELS if t in present]
    if "needs_comment" in present:
        views.insert(views.index(("needs_comment", False)) + 1, ("needs_comment", True))
    for task, rw in views:
        title = TASK_NAMES[task] + (f", at a realistic {int(REALISTIC_COMMENT_RATE * 100)}% comment rate" if rw else "")
        md += [f"\n## {title}\n", "| Model | Accuracy | Accuracy, answered items only | Macro F1 | Coverage at 5% error "
               "| Error in covered set | Lanes: no human / review later / human first | Calibration error (ECE) | Cost per 1,000 | Median latency | Failed |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            s = score(rows, task, m, rw)
            if not s:
                continue
            summary[f"{task}{'_realistic' if rw else ''}/{NAMES.get(m, m)}"] = s
            md.append(f"| {NAMES.get(m, m)} | {pct(s['accuracy'])} | {pct(s['accuracy_answered'])} | {s['macro_f1']:.3f} | {pct(s['coverage_at_budget'])} "
                      f"| {pct(s['error_within_coverage'])} | {pct(s['lanes']['no_human'])} / {pct(s['lanes']['review_later'])} / {pct(s['lanes']['human_first'])} "
                      f"| {s['ece']:.3f} | ${s['cost_per_1000_usd']:.3f} "
                      f"| {s['latency_p50_ms'] or 0:.0f} ms | {s['failed']} of {s['n_eval']} |")
        if not rw:
            if task in PUBLISHED and RESULTS.resolve() == (HERE / "results").resolve():  # benchmark data only
                md.append(f"\nPublished reference: {PUBLISHED[task]}")
            ev_any = [r for r in rows if r["task"] == task and r["split"] == "eval"]
            if LABELS[task] == ["yes", "no"] and ev_any:
                ids = {r["id"]: r["gold"] for r in ev_any}
                base = sum(g == 'yes' for g in ids.values()) / len(ids)
                md.append(f"\nGround truth: {100 * base:.1f}% of {len(ids)} evaluation items are 'yes', so always answering 'no' "
                          f"would score {100 * (1 - base):.1f}% accuracy.")
                for m in models:
                    mev = [r for r in ev_any if r["model"] == m and r.get("ok")]
                    if mev:
                        md.append(f"{NAMES.get(m, m)} answered 'yes' on {100 * sum(r['pred'] == 'yes' for r in mev) / len(mev):.1f}% of them.")
            for m in models:
                ev = [r for r in rows if r["task"] == task and r["model"] == m and r["split"] == "eval" and r.get("ok")]
                by = defaultdict(list)
                for r in ev:
                    by[r.get("provider") or "unknown"].append(r["correct"])
                if by and (len(by) > 1 or "unknown" not in by):
                    parts = ", ".join(f"{p}: {len(v)} items, {100 * statistics.mean(v):.1f}% correct" for p, v in sorted(by.items()))
                    md.append(f"\n{NAMES.get(m, m)} served by: {parts}")
                cut = [r for r in ev if r.get("truncated")]
                if cut:
                    md.append(f"\n{NAMES.get(m, m)}: {len(cut)} of {len(ev)} inputs were cut to fit the model's 512-token limit, "
                              f"{100 * statistics.mean(r['correct'] for r in cut):.1f}% of them correct.")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="relative to this script (default: results)")
    RESULTS = HERE / ap.parse_args().results_dir
    main()
