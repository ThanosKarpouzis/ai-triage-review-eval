"""Convert the AIDev v3 tables used by agent_lane.py from Parquet to JSONL. The one script in the kit that
needs a library: pyarrow (Apache Software Foundation, Apache 2.0). Use it in a throwaway virtual environment.

  mkdir -p aidev/raw aidev/extracts
  for t in pull_request issue related_issue pr_commits pr_reviews pr_task_type repository; do
    curl -fL -o aidev/raw/$t.parquet https://huggingface.co/datasets/hao-li/AIDev/resolve/v3/$t.parquet
  done
  python3 -m venv .venv-aidev && .venv-aidev/bin/pip install pyarrow
  .venv-aidev/bin/python prep_aidev.py aidev

Reads <dir>/raw/<table>.parquet, keeps only the columns agent_lane.py uses, writes <dir>/extracts/<table>.jsonl,
and appends sizes, checksums and row counts to <dir>/MANIFEST.md. AIDev is CC BY 4.0 (Li, Zhang and Hassan).
"""
import datetime, hashlib, json, pathlib, sys
import pyarrow
import pyarrow.parquet as pq

HERE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
KEEP = {
    "pull_request": ["id", "number", "title", "body", "agent", "user_id", "user", "state", "created_at",
                     "closed_at", "merged_at", "repo_id", "repo_url", "html_url"],
    "issue": ["id", "number", "title", "body", "user", "state", "created_at", "closed_at", "html_url"],
    "related_issue": ["pr_id", "issue_id", "source"],
    "pr_commits": ["sha", "pr_id", "author", "committer", "message"],
    "pr_reviews": ["id", "pr_id", "user", "user_type", "state", "submitted_at"],
    "pr_task_type": ["agent", "id", "title", "type", "reason"],
    "repository": ["id", "url", "full_name", "language", "stars", "forks", "license"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = [f"\n## Conversion {datetime.datetime.now().isoformat(timespec='seconds')}\n",
             f"pyarrow {pyarrow.__version__}, Python {sys.version.split()[0]}\n",
             "| Table | Parquet bytes | SHA-256 | Rows | Columns kept | Columns missing |",
             "|---|---|---|---|---|---|"]
    for table, wanted in KEEP.items():
        src = HERE / "raw" / f"{table}.parquet"
        pf = pq.ParquetFile(src)
        have = pf.schema_arrow.names
        cols = [c for c in wanted if c in have]
        missing = [c for c in wanted if c not in have]
        n = 0
        with open(HERE / "extracts" / f"{table}.jsonl", "w", encoding="utf-8") as out:
            for batch in pf.iter_batches(batch_size=5000, columns=cols):
                for row in batch.to_pylist():
                    out.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                    n += 1
        print(f"{table}: {n} rows, kept {cols}, missing {missing}, all columns {have}")
        lines.append(f"| {table} | {src.stat().st_size} | {sha256(src)[:16]} | {n} | {', '.join(cols)} | "
                     f"{', '.join(missing) or 'none'} |")
    with open(HERE / "MANIFEST.md", "a", encoding="utf-8") as m:
        m.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
