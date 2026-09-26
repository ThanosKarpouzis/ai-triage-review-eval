"""Prepare NLBSE'24 issue data for the Jev study (pre-registered, fixed seed).

Evaluation set: the full official test split (1,500 issues).
Calibration set: 300 issues from the official train split, 20 per class per repo.
Text given to both models: title + body, HTML comments removed, whitespace collapsed,
capped at 2,500 characters.
"""
import csv, json, random, re, sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
CAP = 2500
SEED = 20260926

def clean(title, body):
    body = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    text = f"Title: {title.strip()}\n\n{body}"
    return text[:CAP]

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

test = load("data/raw/issues_test.csv")
train = load("data/raw/issues_train.csv")

def rows(items, split):
    out = []
    for i, r in enumerate(items):
        out.append({"id": f"nlbse-{split}-{i:04d}", "task": "issue_type", "split": split,
                    "repo": r["repo"], "label": r["label"], "text": clean(r["title"], r["body"])})
    return out

rng = random.Random(SEED)
groups = defaultdict(list)
for r in train:
    groups[(r["repo"], r["label"])].append(r)
calib = []
for key in sorted(groups):
    calib += rng.sample(groups[key], 20)

evalset = rows(test, "eval")
calset = rows(calib, "calib")
with open("data/issues.jsonl", "w") as f:
    for r in calset + evalset:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("eval", len(evalset), Counter(r["label"] for r in evalset))
print("calib", len(calset), Counter(r["label"] for r in calset))
lens = sorted(len(r["text"]) for r in evalset)
print("chars p50/p90/max", lens[len(lens)//2], lens[int(len(lens)*.9)], lens[-1])
