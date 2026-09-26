"""Sample Microsoft CodeReviewer quality-estimation data for the Jev study (pre-registered, fixed seed).

Reads Diff_Quality_Estimation.zip directly (no extraction needed) and writes data/code_review.jsonl.
Label follows CodeReviewer's own loader: 1 if the change received a review comment, else 0.
Only the diff hunk is kept (the full old file is dropped). Hunks longer than 2,500 characters
are excluded rather than truncated, so neither model ever sees a cut-off diff.
Evaluation set: 500 commented + 500 uncommented hunks from cls-test.
Calibration set: 150 + 150 from cls-valid.

Usage: python3 prep_codereviewer.py /path/to/Diff_Quality_Estimation.zip
"""
import io, json, random, sys, zipfile

CAP = 2500
SEED = 20260926
PLAN = {"test": ("eval", 500), "valid": ("calib", 150)}

def label_of(js):
    return 1 if js.get("msg") else int(js.get("y", 0) or 0)

def sample(member, zf, per_class, rng):
    res = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with zf.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            try:
                js = json.loads(line)
            except ValueError:
                continue
            patch = js.get("patch") or ""
            if not patch.strip() or len(patch) > CAP:
                continue
            y = label_of(js)
            seen[y] += 1
            item = {"patch": patch, "y": y}
            if len(res[y]) < per_class:
                res[y].append(item)
            else:
                j = rng.randrange(seen[y])
                if j < per_class:
                    res[y][j] = item
    return res, seen

def main(zip_path, out="data/code_review.jsonl"):
    rng = random.Random(SEED)
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    rows = []
    for key, (split, n) in [("valid", PLAN["valid"]), ("test", PLAN["test"])]:
        member = next(x for x in names if x.endswith(f"cls-{key}.jsonl"))
        res, seen = sample(member, zf, n, rng)
        print(f"{member}: eligible commented={seen[1]} uncommented={seen[0]} -> sampled {len(res[1])}+{len(res[0])}")
        items = res[1] + res[0]
        rng.shuffle(items)
        for i, it in enumerate(items):
            rows.append({"id": f"cr-{split}-{i:04d}", "task": "needs_comment", "split": split,
                         "label": "yes" if it["y"] else "no", "text": it["patch"]})
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows to", out)

if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
