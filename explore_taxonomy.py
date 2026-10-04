#!/usr/bin/env python3
"""Exploratory checks for the taxonomy test, added 4 October 2026 after all results (not pre-registered).
Standard library only; reads stored answers and label histories, makes no API or model calls.

1. Baselines on the evaluation split: always the most common label of the calibration split; and, for the family
   that has an area step, the same area-to-label mapping built from the area labels already on each issue (the
   repository's own, often bot-applied) instead of from Jev's area answers.
2. Time split: fit on issues created before --cutoff (both splits), test on issues created on or after it, for the
   area-then-label and checklist methods, next to every single-question model on the same test issues.

  python3 explore_taxonomy.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data --cutoff 2026-05-01
"""
import argparse, json
from collections import Counter
from pathlib import Path

from combine_taxonomy import fit_area, fit_multinomial, logit

HERE = Path(__file__).resolve().parent
NAMES = {"typesafe-ai/jev": "Jev, one question", "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
         "anthropic/claude-opus-5.5": "Claude Opus 5.5", "anthropic/claude-opus-4.7": "Claude Opus 4.7"}


def latest(path, ids):
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in ids:
                out[r["id"]] = r
    return out


def final_labels(events):
    state = {}
    for e in sorted(events or [], key=lambda e: e["seconds_after_creation"]):
        state[e["label"]] = e["event"] == "labeled"
    return {l for l, on in state.items() if on and l}


def pct(x, n):
    return f"{100 * x / n:.1f}% ({x} of {n})" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cutoff", default="2026-05-01")
    a = ap.parse_args()
    res, data = HERE / a.results_dir, HERE / a.data_dir
    tax = json.loads((data / "taxonomy.json").read_text(encoding="utf-8"))
    hist = {h["number"]: h for h in map(json.loads, (data / "label_history.jsonl").open(encoding="utf-8"))}
    md = [f"# Exploratory checks: {tax['repo']} ({data.name})\n",
          "Added 4 October 2026 after all results; not pre-registered. No calls were made.\n"]

    def single_question(task, ids):
        rows = {}
        for f in sorted(res.glob("*.jsonl")):
            for r in latest(f, ids).values():
                if r["task"] == task and r["model"] in NAMES:
                    rows.setdefault(r["model"], {})[r["id"]] = r
        return rows

    for fam in tax["families"]:
        task = f"own_{fam}"
        items = {json.loads(l)["id"]: json.loads(l) for l in (data / f"taxonomy_{fam}.jsonl").open(encoding="utf-8")}
        has_area = tax.get("area") and tax["area"]["for_family"] == fam
        has_checks = fam in (tax.get("checks") or {})
        if not (has_area or has_checks):
            continue
        cal = [i for i in items.values() if i["split"] == "calib"]
        ev = [i for i in items.values() if i["split"] == "eval"]
        md.append(f"\n## {fam}\n")
        # 1a. majority baseline
        top, _ = Counter(i["label"] for i in cal).most_common(1)[0]
        md.append(f"Always answering the calibration split's most common label ({top}): "
                  f"{pct(sum(i['label'] == top for i in ev), len(ev))} of evaluation issues right.")
        labels = [o["name"] for o in tax["families"][fam]["options"]]
        # 1b. area labels already on the issue, through the same mapping
        if has_area:
            areas = [o["name"] for o in tax["area"]["options"]]

            def label_areas(i):
                found = {ar for ar in areas for l in final_labels(hist[i["issue"]]["events"]) if l.startswith(ar + "/")}
                return {ar: 1 / len(found) for ar in found}
            with_area = [i for i in items.values() if label_areas(i)]
            cal_a = [dict(i, gold=i["label"], probs=label_areas(i)) for i in with_area if i["split"] == "calib"]
            ev_a = [dict(i, gold=i["label"], probs=label_areas(i)) for i in with_area if i["split"] == "eval"]
            pred = fit_area(cal_a, labels)
            right = sum(max(pred(i).items(), key=lambda kv: kv[1])[0] == i["gold"] for i in ev_a)
            two = latest(res / f"derived-{data.name}" / f"jev-area-{fam}.jsonl", {i["id"] for i in ev_a})
            two_right = sum(r.get("pred") == r["gold"] for r in two.values())
            md.append(f"Area labels already on the issue, mapped to {fam} the same way: {pct(right, len(ev_a))} right, on the "
                      f"{len(ev_a)} of {len(ev)} evaluation issues that carry an area label. Jev, area then {fam}, on the same "
                      f"issues: {pct(two_right, len(two))}.")
            first_by = Counter()
            for i in ev_a:
                for e in sorted(hist[i["issue"]]["events"] or [], key=lambda e: e["seconds_after_creation"]):
                    if e["event"] == "labeled" and e["label"] and any(e["label"].startswith(ar + "/") for ar in areas):
                        first_by[e["actor"]] += 1
                        break
            md.append(f"Who first applied an area label on those issues: {dict(first_by)}.")
        # 2. time split
        before = [i for i in items.values() if i["created_at"] < a.cutoff]
        after = [i for i in items.values() if i["created_at"] >= a.cutoff]
        md.append(f"\n**Time split at {a.cutoff}:** fit on {len(before)} earlier issues, test on {len(after)} later ones "
                  f"(months: {dict(sorted(Counter(i['created_at'][:7] for i in after).items()))}).\n")
        md.append("| Method | Right on later issues |\n|---|---|")
        ids_after = {i["id"] for i in after}
        top_b, _ = Counter(i["label"] for i in before).most_common(1)[0]
        md.append(f"| Most common earlier label ({top_b}) | {pct(sum(i['label'] == top_b for i in after), len(after))} |")
        if has_area:
            src = latest(res / "checks" / "jev_area.jsonl", set(items))
            tr = [dict(src[i["id"]], gold=i["label"]) for i in before if i["id"] in src and src[i["id"]].get("ok")]
            te = [dict(src[i["id"]], gold=i["label"]) for i in after if i["id"] in src and src[i["id"]].get("ok")]
            pred = fit_area(tr, labels)
            md.append(f"| Jev, area then {fam} (mapping learned on earlier issues) | "
                      f"{pct(sum(max(pred(r).items(), key=lambda kv: kv[1])[0] == r['gold'] for r in te), len(after))} |")
        if has_checks:
            src = latest(res / "checks" / "jev_checks.jsonl", set(items))
            tr = [dict(src[i["id"]], gold=i["label"]) for i in before if i["id"] in src and src[i["id"]].get("ok")]
            te = [dict(src[i["id"]], gold=i["label"]) for i in after if i["id"] in src and src[i["id"]].get("ok")]
            names = sorted(tr[0]["probs"])
            feats = lambda r: [logit(r["probs"][k]) for k in names]
            classes = sorted({r["gold"] for r in tr})
            p = fit_multinomial([feats(r) for r in tr], [r["gold"] for r in tr], classes)
            md.append(f"| Jev, checklist (fitted on earlier issues) | "
                      f"{pct(sum(max(p(feats(r)).items(), key=lambda kv: kv[1])[0] == r['gold'] for r in te), len(after))} |")
        for m, rows in single_question(task, ids_after).items():
            if len(rows) >= 0.9 * len(after):
                md.append(f"| {NAMES[m]} | {pct(sum(r.get('pred') == r['gold'] for r in rows.values()), len(rows))} |")
    out = res / f"taxonomy_explore-{data.name}.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
