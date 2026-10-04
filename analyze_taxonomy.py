#!/usr/bin/env python3
"""Score the taxonomy test (pre-registered 3 October 2026; see README.md). Standard library only.

  python3 analyze_taxonomy.py --results-dir kit/<repo>/results        # data is read from kit/<repo>/data

Writes taxonomy_summary.md and taxonomy_summary.json next to the results. Rules, fixed before any run:
- Evaluation split only; thresholds chosen on the calibration split; failures count as wrong (as in analyze.py,
  whose scoring is reused unchanged).
- Primary comparison, per model: on evaluation issues whose type label maps to bug, feature or question, arm B's
  type answer is mapped the same way (any other label counts as wrong) and compared with arm A's answer on the
  same issues. Reported: accuracy and no-human coverage for each arm, and the difference (B minus A) with a paired
  bootstrap 95% interval over issues (2,000 resamples, fixed seed). A difference is called real only if its interval
  excludes zero. Thresholds are fixed before resampling.
- Slices of the primary comparison by label history. When the data records provenance (fetch_taxonomy.py, gold
  "confirmed"): type label set by a person against confirmed by a person's review; otherwise: applied at creation
  (within 120 seconds, or by a bot) against applied later by a person. Plus reclassified (another type label was
  applied and later removed). Slices under 30 evaluation issues are reported as counts only.
- A family with fewer than 150 evaluation issues is marked exploratory.
- Follow-up tests (pre-registered 4 October 2026): the combined results in results/derived-<data folder>/ (from
  combine_taxonomy.py) are scored like any other model. Each model's accuracy in each family is also compared with
  single-question Jev's on the same items, with a paired bootstrap 95% interval. A model that answered fewer than 90%
  of a task's items in this data folder (Opus on the secondary data) is left out of that task's tables.
- Reference (exploratory, no model calls), amended 3 October 2026 before any model ran: only on issues a person
  reviewed (removed the review marker), the bot's single label from before the review is scored against the label
  after it, next to the models on the same issues. The first version also counted labels a person set without a
  review, which are mostly corrections of the bot, and labels confirmed by review, which agree with the bot by
  construction, so it measured the confirmation rule rather than the bot.
- Per family (arm B): the usual metrics; an ordinal family also reports accuracy within one level; accuracy on
  issues whose correct label has a description against those whose label has none; deprecated correct labels.
"""
import argparse, json, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

import analyze
from analyze import score, choose_threshold, pct, NAMES

HERE = Path(__file__).resolve().parent
BOOT, BOOT_SEED, MIN_SLICE, AT_CREATION_S, MIN_FAMILY = 2000, 20261003, 30, 120, 150


def num_of(item_id):
    return int(item_id.split("-")[1])


def boot_ci(pairs, fn):
    """pairs: list of per-issue tuples; fn(list) -> statistic. Returns the 2.5th and 97.5th percentiles."""
    rng, stats = random.Random(BOOT_SEED), []
    for _ in range(BOOT):
        stats.append(fn([pairs[rng.randrange(len(pairs))] for _ in pairs]))
    stats.sort()
    return stats[int(0.025 * BOOT)], stats[int(0.975 * BOOT) - 1]


def slice_of(h, type_fam):
    """Classify an issue's final type label by how it got there."""
    final = h["labels"].get(type_fam) or []
    if len(final) != 1 or h["events"] is None:
        return "unknown", False
    ever = {e["label"] for e in h["events"] if e["event"] == "labeled"}
    reclassified = bool({l for l in ever if l in TYPE_LABELS} - set(final))
    if (h.get("provenance") or {}).get(type_fam):
        return h["provenance"][type_fam], reclassified
    first = next((e for e in h["events"] if e["event"] == "labeled" and e["label"] == final[0]), None)
    if first is None:
        return "unknown", reclassified
    if first["actor"] == "bot" or first["seconds_after_creation"] <= AT_CREATION_S:
        return "at creation (template or bot)", reclassified
    return "later, by a person", reclassified


def reviewed(h):
    """A person removed the bot's review marker (the config's review_marker, stored in the manifest)."""
    return bool(REVIEW_MARKER) and any(e["event"] == "unlabeled" and e["label"] == REVIEW_MARKER and e["actor"] != "bot"
                                       for e in (h.get("events") or []))


def comparison(rows, model, tax, issues=None):
    """Arm A against arm B on the mapped type decision. Returns per-issue pairs and thresholds."""
    gmap, tf = tax["generic_map"], f"own_{tax['type_family']}"
    a = {num_of(r["id"]): r for r in rows if r["model"] == model and r["task"] == "issue_type"}
    b = {num_of(r["id"]): r for r in rows if r["model"] == model and r["task"] == tf}
    common = sorted(set(a) & set(b))

    def b_correct(n):
        return gmap.get(b[n]["pred"]) == a[n]["gold"]

    ca = [n for n in common if a[n]["split"] == "calib"]
    ta = choose_threshold([a[n] for n in ca], [1.0] * len(ca)) if ca else None
    tb = choose_threshold([{"confidence": b[n]["confidence"], "correct": b_correct(n)} for n in ca], [1.0] * len(ca)) if ca else None
    ev = [n for n in common if a[n]["split"] == "eval" and (issues is None or n in issues)]
    pairs = [(a[n]["correct"], b_correct(n),
              ta is not None and a[n]["confidence"] >= ta, tb is not None and b[n]["confidence"] >= tb) for n in ev]
    return pairs, ta, tb, [a[n] for n in ev]


def summarise(pairs, with_ci=True):
    n = len(pairs)
    acc_a = sum(p[0] for p in pairs) / n
    acc_b = sum(p[1] for p in pairs) / n
    cov_a = sum(p[2] for p in pairs) / n
    cov_b = sum(p[3] for p in pairs) / n
    err = lambda ok, inlane: (sum(1 for p in pairs if p[inlane] and not p[ok]) / sum(1 for p in pairs if p[inlane])) if any(p[inlane] for p in pairs) else None
    out = {"n": n, "acc_a": acc_a, "acc_b": acc_b, "acc_diff": acc_b - acc_a, "cov_a": cov_a, "cov_b": cov_b,
           "cov_diff": cov_b - cov_a, "lane_err_a": err(0, 2), "lane_err_b": err(1, 3)}
    if with_ci:
        out["acc_diff_ci"] = boot_ci(pairs, lambda s: (sum(p[1] for p in s) - sum(p[0] for p in s)) / len(s))
        out["cov_diff_ci"] = boot_ci(pairs, lambda s: (sum(p[3] for p in s) - sum(p[2] for p in s)) / len(s))
    return out


def ci_text(ci):
    lo, hi = ci
    verdict = "real" if lo > 0 or hi < 0 else "not distinguishable from zero"
    return f"{100 * lo:+.1f} to {100 * hi:+.1f} points, {verdict}"


def main():
    global TYPE_LABELS, REVIEW_MARKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="relative to this script, e.g. kit/metabase__metabase/results")
    ap.add_argument("--data-dir", help="defaults to the results folder's sibling 'data'")
    a = ap.parse_args()
    res = HERE / a.results_dir
    data = HERE / a.data_dir if a.data_dir else res.parent / "data"
    tax = json.loads((data / "taxonomy.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    history = {h["number"]: h for h in map(json.loads, (data / "label_history.jsonl").open(encoding="utf-8"))}
    REVIEW_MARKER = manifest.get("review_marker")
    fams = tax["families"]
    TYPE_LABELS = {o["name"] for o in fams[tax["type_family"]]["options"]}
    described = {o["name"]: bool(o["description"]) for f in fams.values() for o in f["options"]}
    deprecated = {o["name"] for f in fams.values() for o in f["options"] if "deprecated" in o["description"].lower()}

    analyze.RESULTS = res
    # Only items in this data folder count, so one results folder can serve the headline and the secondary analysis.
    ids = {json.loads(l)["id"] for f in [data / "issues.jsonl"] + sorted(data.glob("taxonomy_*.jsonl")) if f.exists()
           for l in f.open(encoding="utf-8")}
    rows = [r for r in analyze.load() if r["id"] in ids]
    derived = res / f"derived-{data.name}"
    if derived.exists():
        analyze.RESULTS = derived
        rows += [r for r in analyze.load() if r["id"] in ids]
        analyze.RESULTS = res
    NAMES.update({"typesafe-ai/jev (checklist)": "Jev, checklist combined", "typesafe-ai/jev (area, then team)": "Jev, area then team"})
    need = Counter(json.loads(l)["task"] for f in [data / "issues.jsonl"] + sorted(data.glob("taxonomy_*.jsonl")) if f.exists()
                   for l in f.open(encoding="utf-8"))
    have = Counter((r["model"], r["task"]) for r in rows)
    partial = {(m, t) for (m, t), n in have.items() if n < 0.9 * need[t]}
    rows = [r for r in rows if (r["model"], r["task"]) not in partial]
    for fam, spec in fams.items():
        analyze.LABELS[f"own_{fam}"] = [o["name"] for o in spec["options"]]
    order = ["typesafe-ai/jev", "typesafe-ai/jev (checklist)", "typesafe-ai/jev (area, then team)",
             "anthropic/claude-haiku-4.5", "anthropic/claude-opus-5.5", "anthropic/claude-opus-5", "anthropic/claude-opus-4.7"]
    models = sorted({r["model"] for r in rows if r["task"] == "issue_type" or r["task"].startswith("own_")},
                    key=lambda m: order.index(m) if m in order else 99)
    summary = {"repo": tax["repo"], "manifest": {k: manifest.get(k) for k in ("issues_found", "issues_kept", "skipped", "arm_a", "arm_b")}}
    md = [f"# Taxonomy test: {tax['repo']}\n",
          f"Issues found {manifest.get('issues_found')}, kept {manifest.get('issues_kept')}; skipped: {manifest.get('skipped')}.",
          "Evaluation split only; thresholds chosen on the calibration split; failures count as wrong.\n",
          "Label descriptions: " + "; ".join(f"{f} {sum(bool(o['description']) for o in s['options'])} of {len(s['options'])} described"
                                            f" ({sum('deprecated' in o['description'].lower() for o in s['options'])} say deprecated)"
                                            for f, s in fams.items()) + "."]

    # Primary comparison
    md += ["\n## Primary comparison: generic question (arm A) against the team's own type labels (arm B)\n",
           "Type decision mapped to bug / feature / question, same evaluation issues for both arms.\n",
           "| Model | Issues | Accuracy A | Accuracy B | Difference B minus A (95% interval) | No-human A (error) | No-human B (error) | Coverage difference (95% interval) |",
           "|---|---|---|---|---|---|---|---|"]
    summary["primary"], slices_md = {}, []
    for m in models:
        pairs, ta, tb, a_rows = comparison(rows, m, tax)
        if not pairs:
            continue
        s = summarise(pairs)
        s["threshold_a"], s["threshold_b"] = ta, tb
        s["arm_a_answers"] = dict(Counter(r["pred"] for r in a_rows))
        summary["primary"][NAMES.get(m, m)] = s
        md.append(f"| {NAMES.get(m, m)} | {s['n']} | {pct(s['acc_a'])} | {pct(s['acc_b'])} | {100 * s['acc_diff']:+.1f} points ({ci_text(s['acc_diff_ci'])}) "
                  f"| {pct(s['cov_a'])} ({pct(s['lane_err_a'])}) | {pct(s['cov_b'])} ({pct(s['lane_err_b'])}) | {100 * s['cov_diff']:+.1f} points ({ci_text(s['cov_diff_ci'])}) |")
        q = s["arm_a_answers"].get("question", 0)
        slices_md.append(f"{NAMES.get(m, m)}, arm A answered 'question' on {q} of {s['n']} issues ({pct(q / s['n'])}); "
                         f"gold labels: {dict(Counter(r['gold'] for r in a_rows))}.")
    md += [""] + slices_md

    # Slices
    md += ["\n## By label history\n", "| Model | Slice | Issues | Accuracy A | Accuracy B | Difference (95% interval) |", "|---|---|---|---|---|---|"]
    summary["slices"] = {}
    groups = defaultdict(set)
    for n, h in history.items():
        how, recl = slice_of(h, tax["type_family"])
        groups[how].add(n)
        if recl:
            groups["reclassified"].add(n)
    for m in models:
        for name in ["set by a person", "confirmed by review", "at creation (template or bot)", "later, by a person",
                     "reclassified", "unconfirmed", "unknown"]:
            pairs, *_ = comparison(rows, m, tax, groups.get(name, set()))
            if not pairs:
                continue
            big = len(pairs) >= MIN_SLICE
            s = summarise(pairs, with_ci=big)
            summary["slices"][f"{NAMES.get(m, m)}/{name}"] = s
            if big:
                md.append(f"| {NAMES.get(m, m)} | {name} | {s['n']} | {pct(s['acc_a'])} | {pct(s['acc_b'])} | {100 * s['acc_diff']:+.1f} ({ci_text(s['acc_diff_ci'])}) |")
            else:
                md.append(f"| {NAMES.get(m, m)} | {name} | {s['n']} | {sum(p[0] for p in pairs)} right | {sum(p[1] for p in pairs)} right | counts only, under {MIN_SLICE} |")

    # Per family, arm B
    summary["families"] = {}
    for fam, spec in fams.items():
        task = f"own_{fam}"
        md += [f"\n## Arm B, {fam} labels ({len(spec['options'])} options)\n",
               "| Model | Accuracy | Macro F1 | No human / review later / human first | Error in no-human lane | Calibration error | Cost per 1,000 | Failed |",
               "|---|---|---|---|---|---|---|---|"]
        notes = []
        for m in models:
            s = score(rows, task, m)
            if not s:
                continue
            ev = [r for r in rows if r["task"] == task and r["model"] == m and r["split"] == "eval"]
            if spec.get("ordinal"):
                order = sorted(o["name"] for o in spec["options"])
                s["within_one"] = sum(1 for r in ev if r["pred"] in order and abs(order.index(r["pred"]) - order.index(r["gold"])) <= 1) / len(ev)
                notes.append(f"{NAMES.get(m, m)}: within one level {pct(s['within_one'])} (order {', '.join(order)}).")
            d = [r["correct"] for r in ev if described.get(r["gold"])]
            u = [r["correct"] for r in ev if not described.get(r["gold"])]
            s["acc_described"] = statistics.mean(d) if d else None
            s["acc_undescribed"] = statistics.mean(u) if u else None
            notes.append(f"{NAMES.get(m, m)}: correct label described {pct(s['acc_described'])} of {len(d)}, "
                         f"not described {pct(s['acc_undescribed'])} of {len(u)}.")
            by_prov = defaultdict(list)
            for r in ev:
                h = history.get(r.get("issue") or num_of(r["id"])) or {}
                by_prov[(h.get("provenance") or {}).get(fam) or "unknown"].append(r["correct"])
            if len(by_prov) > 1:
                s["accuracy_by_provenance"] = {k: {"n": len(v), "accuracy": statistics.mean(v)} for k, v in by_prov.items()}
                notes.append(f"{NAMES.get(m, m)} by how the correct label got there: " + "; ".join(
                    f"{k} {pct(statistics.mean(v))} of {len(v)}" for k, v in sorted(by_prov.items())) + ".")
            dep = [r for r in ev if r["gold"] in deprecated]
            if dep:
                s["deprecated_gold"] = {"n": len(dep), "correct": sum(r["correct"] for r in dep)}
                notes.append(f"{NAMES.get(m, m)}: {len(dep)} issues carry a deprecated label, {sum(r['correct'] for r in dep)} answered with it.")
            summary["families"][f"{fam}/{NAMES.get(m, m)}"] = s
            s["exploratory"] = s["n_eval"] < MIN_FAMILY
            md.append(f"| {NAMES.get(m, m)}{' (exploratory, under ' + str(MIN_FAMILY) + ')' if s['exploratory'] else ''} | {pct(s['accuracy'])} | {s['macro_f1']:.3f} | {pct(s['lanes']['no_human'])} / {pct(s['lanes']['review_later'])} / "
                      f"{pct(s['lanes']['human_first'])} | {pct(s['error_within_coverage'])} | {s['ece']:.3f} | ${s['cost_per_1000_usd']:.3f} | {s['failed']} of {s['n_eval']} |")
        gold = Counter(r["gold"] for r in rows if r["task"] == task and r["split"] == "eval" and r["model"] == models[0]) if models else Counter()
        md += [""] + notes + [f"Correct labels in the evaluation split: {dict(gold.most_common())}."]
        # Reference: the repository's own bot, on issues where it answered before a person confirmed or changed the label
        bot_ev = {}
        for r in rows:
            if r["task"] == task and r["split"] == "eval":
                h = history.get(r.get("issue") or num_of(r["id"]))
                b = (h or {}).get("bot_labels", {}).get(fam) or []
                if b and reviewed(h):
                    bot_ev[r["id"]] = (len(b) == 1 and b[0] == r["gold"])
        if bot_ev:
            line = [f"Reference, exploratory: on {len(bot_ev)} of these evaluation issues a person reviewed the bot's labels; "
                    f"the reviewer kept the bot's {fam} label on {pct(sum(bot_ev.values()) / len(bot_ev))}."]
            for m in models:
                same = [r["correct"] for r in rows if r["task"] == task and r["model"] == m and r["id"] in bot_ev]
                if same:
                    line.append(f"{NAMES.get(m, m)} on the same issues: {pct(sum(same) / len(same))}.")
            md.append(" ".join(line))
            summary.setdefault("bot_reference", {})[fam] = {"n": len(bot_ev), "accuracy": sum(bot_ev.values()) / len(bot_ev)}

    # Pre-registered predictions
    # Paired comparison with single-question Jev, per family (follow-up tests)
    jev = "typesafe-ai/jev"
    md += ["\n## Each model against single-question Jev, same items\n",
           "| Family | Model | Items | Model accuracy | Jev accuracy | Difference (95% interval) |", "|---|---|---|---|---|---|"]
    summary["paired_vs_jev"] = {}
    for fam in fams:
        task = f"own_{fam}"
        base = {r["id"]: r["correct"] for r in rows if r["task"] == task and r["model"] == jev and r["split"] == "eval"}
        for m in models + sorted({r["model"] for r in rows if r["task"] == task} - set(models)):
            if m == jev:
                continue
            other = {r["id"]: r["correct"] for r in rows if r["task"] == task and r["model"] == m and r["split"] == "eval"}
            common = sorted(set(base) & set(other))
            if not common:
                continue
            pairs = [(base[i], other[i]) for i in common]
            d = (sum(p[1] for p in pairs) - sum(p[0] for p in pairs)) / len(pairs)
            ci = boot_ci(pairs, lambda x: (sum(p[1] for p in x) - sum(p[0] for p in x)) / len(x))
            summary["paired_vs_jev"][f"{fam}/{NAMES.get(m, m)}"] = {"n": len(pairs), "diff": d, "ci": ci}
            md.append(f"| {fam} | {NAMES.get(m, m)} | {len(pairs)} | {pct(sum(p[1] for p in pairs) / len(pairs))} | "
                      f"{pct(sum(p[0] for p in pairs) / len(pairs))} | {100 * d:+.1f} ({ci_text(ci)}) |")
    if partial:
        md.append("\nLeft out here because they answered under 90% of a task's items: " +
                  ", ".join(sorted({f"{NAMES.get(m, m)} on {t}" for m, t in partial})) + ".")

    md += ["\n## Pre-registered predictions\n"]
    p1 = {m: s for m, s in summary["primary"].items()}
    ok1 = all(s["acc_diff_ci"][0] > 0 for s in p1.values()) if p1 else None
    diffs = "; ".join("%s %+.1f points" % (m, 100 * s["acc_diff"]) for m, s in p1.items())
    verdict1 = "held" if ok1 else ("did not hold" if ok1 is not None else "not scored")
    md.append(f"1. Arm B beats arm A on type accuracy for both models, interval above zero: {verdict1} ({diffs}). "
              "Gain largest on reclassified issues: see the slice table.")
    rf = tax.get("routing_family")
    if rf:
        cov = {k.split('/', 1)[1]: v["lanes"]["no_human"] for k, v in summary["families"].items() if k.startswith(rf + "/")}
        pri = {k.split('/', 1)[1]: v["lanes"]["no_human"] for k, v in summary["families"].items()
               if any(fams[f].get("ordinal") and k.startswith(f + "/") for f in fams)}
        md.append(f"2. Routing ({rf}) opens a no-human lane of at least 30% for at least one model: "
                  f"{'held' if any(c >= 0.30 for c in cov.values()) else 'did not hold'} ({', '.join(f'{m} {pct(c)}' for m, c in cov.items())}); "
                  f"priority opens none: {'held' if all(c == 0 for c in pri.values()) else 'did not hold'} ({', '.join(f'{m} {pct(c)}' for m, c in pri.items())}).")
    md.append("3. Labels without descriptions do worse than labels with descriptions: see the described and not described lines per family.")

    name = "taxonomy_summary" + ("" if data.name == "data" else f"-{data.name}")
    md.insert(1, f"Data: {data.name} ({manifest.get('gold', 'all labels')}).\n")
    (res / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    (res / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


TYPE_LABELS, REVIEW_MARKER = set(), None

if __name__ == "__main__":
    main()
