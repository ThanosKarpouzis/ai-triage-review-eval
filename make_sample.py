#!/usr/bin/env python3
"""Stratified sample of the benchmark for the exploratory Opus arm (fixed seed, fixed before any Opus call).

Issues: calibration 150 (50 per class), evaluation 450 (30 per class per repository).
Code review: calibration 100 (50 per label), evaluation 300 (150 per label).
Writes data-sample/ and copies the existing Haiku (and any local classifier) results for exactly these
items into results-sample/, so all models are compared on the same 1,000 items.
"""
import json, random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260926
rng = random.Random(SEED)

def load(f):
    return [json.loads(l) for l in (HERE / "data" / f).open(encoding="utf-8")]

def pick(rows, key, n):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for k in sorted(groups):
        out += rng.sample(groups[k], n)
    return out

issues, reviews = load("issues.jsonl"), load("code_review.jsonl")
s_issues = (pick([r for r in issues if r["split"] == "calib"], lambda r: r["label"], 50)
            + pick([r for r in issues if r["split"] == "eval"], lambda r: (r["repo"], r["label"]), 30))
s_reviews = (pick([r for r in reviews if r["split"] == "calib"], lambda r: r["label"], 50)
             + pick([r for r in reviews if r["split"] == "eval"], lambda r: r["label"], 150))
out = HERE / "data-sample"
out.mkdir(exist_ok=True)
for name, rows in [("issues.jsonl", s_issues), ("code_review.jsonl", s_reviews)]:
    with (out / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
ids = {r["id"] for r in s_issues + s_reviews}
res = HERE / "results-sample"
res.mkdir(exist_ok=True)
for model in ["haiku", "nli"]:
    src = HERE / "results" / f"{model}.jsonl"
    if src.exists():
        kept = [l for l in src.open(encoding="utf-8") if json.loads(l)["id"] in ids]
        (res / f"{model}.jsonl").write_text("".join(kept), encoding="utf-8")
        print(f"copied {len(kept)} {model} results for the sample")
print(f"sample: {len(s_issues)} issues, {len(s_reviews)} code changes")
