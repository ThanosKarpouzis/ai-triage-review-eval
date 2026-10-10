#!/usr/bin/env python3
"""Build the own-repo kit's data from one GitHub repository. Standard library only.

Two decisions, both with the repository's own history as ground truth:
  issue_type        bug / feature / question, from the labels maintainers applied
  pr_needs_changes  did a reviewer comment on or request changes to the pull request before it was merged?

  export GITHUB_TOKEN=...   # optional but strongly advised: 5,000 requests an hour instead of 60.
                            # A fine-grained token with read-only access to public repositories is enough.
  python3 fetch_repo.py OWNER/REPO --list-labels            # see which labels the repo uses
  python3 fetch_repo.py OWNER/REPO --bug "type: bug" --feature "type: feature" --question "type: question"

Writes kit/OWNER__REPO/data/issues.jsonl, prs.jsonl and manifest.json. Then:
  python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results
  python3 analyze.py --results-dir kit/OWNER__REPO/results

Rules, fixed before fetching (see README.md):
- Only items created on or after --since (default 1 January 2026), so no model has seen them in training.
- Issues: pull requests are excluded; an issue counts only if its labels match exactly one of the three
  groups. Text is the title and body, HTML comments removed, capped at 2,500 characters (as in the benchmark).
- Pull requests: merged only, drafts and bot authors excluded. Reviews by the author or by bots are ignored.
  "yes" if any other reviewer left a comment review or requested changes; "no" if at least one other
  reviewer approved and none commented or requested changes; anything else is skipped. (Until 26 September
  2026 "yes" meant requested changes only; many teams comment instead, which left almost no "yes" cases.) Text is the title, the first 600 characters of the description and the diff. Pull requests
  whose diff exceeds 20,000 characters (--max-diff-chars) are excluded rather than truncated, and counted in
  the manifest. (The default was 6,000 until 26 September 2026; that excluded most of the pull requests
  reviewers discussed.)
- About 30% of each label goes to the calibration split, the rest to evaluation, with a fixed seed.
"""
import argparse, hashlib, json, os, random, re, sys, time, urllib.error, urllib.parse, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
ISSUE_CAP, DESC_CAP, DIFF_CAP = 2500, 600, 20000
SEED = 20260926


def clean_issue(title, body):
    """Identical to prep_nlbse.clean, so kit issues get the same treatment as the benchmark."""
    body = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"Title: {title.strip()}\n\n{body}"[:ISSUE_CAP]


class GitHub:
    """GitHub REST client that keeps a local copy of every answer (since 3 October 2026; before that, only
    pull-request reviews and diffs were kept). A later run with the same request reads the copy from disk and
    makes no API call, so data can be re-derived without asking GitHub again. refresh=True asks GitHub again and
    overwrites the copies; the manifest records when the oldest and newest answers used were fetched."""

    def __init__(self, token=None, cache_dir=None, refresh=False):
        self.token, self.calls, self.cached, self.refresh = token, 0, 0, refresh
        self.cache_dir = cache_dir
        self.fetched_at = []  # when each answer used in this run was fetched from GitHub
        self.last_from_disk = False
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self):
        known = sorted(t for t in self.fetched_at if t)
        return {"oldest_answer": known[0] if known else None, "newest_answer": known[-1] if known else None,
                "answers_from_local_copy": self.cached, "answers_from_github": self.calls}

    def get(self, path, params=None, accept="application/vnd.github+json", raw=False, cache=True):
        """Every answer is kept on disk and reused on later runs (cache=False opts out). Errors that will not
        change (404, 406, 410, 422) are kept too."""
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        cfile = None
        if cache and self.cache_dir:
            cfile = self.cache_dir / (hashlib.sha256(f"{accept} {url}".encode()).hexdigest()[:32] + ".json")
            if cfile.exists() and not self.refresh:
                self.cached += 1
                self.last_from_disk = True
                hit = json.loads(cfile.read_text(encoding="utf-8"))
                self.fetched_at.append(hit.get("fetched"))
                if "error" in hit:
                    raise urllib.error.HTTPError(url, hit["error"], "cached error", None, None)
                return hit["body"] if raw else json.loads(hit["body"])
        self.last_from_disk = False
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            body = self._fetch(url, accept)
        except urllib.error.HTTPError as e:
            if cfile and e.code in (404, 406, 410, 422):
                cfile.write_text(json.dumps({"error": e.code, "fetched": now, "url": url}), encoding="utf-8")
            raise
        self.fetched_at.append(now)
        if cfile:
            cfile.write_text(json.dumps({"body": body, "fetched": now, "url": url}), encoding="utf-8")
        return body if raw else json.loads(body)

    def _fetch(self, url, accept):
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "jev-study-kit"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(6):
            self.calls += 1
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                    return r.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                if e.code in (403, 429) and (e.headers.get("x-ratelimit-remaining") == "0" or e.code == 429):
                    reset = int(e.headers.get("x-ratelimit-reset") or time.time() + 60)
                    wait = max(5, reset - time.time() + 2)
                    print(f"  GitHub rate limit reached, waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                if e.code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt)
        raise SystemExit(f"GitHub kept failing for {url}; try again later")

    def pages(self, path, params, max_pages, cache=True):
        for page in range(1, max_pages + 1):
            batch = self.get(path, dict(params, per_page=100, page=page), cache=cache)
            if not batch:
                return
            yield from batch
            if len(batch) < 100:
                return  # a short page is the last one; saves one empty request per list (3 October 2026)


def list_labels(gh, repo):
    labels = [l["name"] for l in gh.pages(f"/repos/{repo}/labels", {}, 20)]
    print(f"{len(labels)} labels in {repo}:")
    for name in labels:
        print(" ", name)


def issues_via_search(gh, repo, since):
    """Issues created since `since`, newest first, through date-split search. For repositories so busy that
    GitHub refuses to page the issue list past about 10,000 items (pull requests count towards it).
    Added 10 October 2026; the items kept and the rules applied to them are the same as without it."""
    from datetime import date
    from fetch_taxonomy import search_range  # imported here: fetch_taxonomy imports this module
    found, seen = [], set()
    search_range(gh, repo, date.fromisoformat(since), date.today(), found)
    unique = [it for it in found if not (it["number"] in seen or seen.add(it["number"]))]
    return sorted(unique, key=lambda it: (it["created_at"], it["number"]), reverse=True)


def fetch_issues(gh, repo, groups, since, limit, max_pages, via_search=False):
    out, skipped = [], Counter()
    listing = issues_via_search(gh, repo, since) if via_search else gh.pages(
        f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc", "since": since + "T00:00:00Z"}, max_pages)
    for it in listing:
        if it["created_at"][:10] < since:
            break  # sorted newest first, so everything after this is older
        if "pull_request" in it:
            skipped["pull request"] += 1
            continue
        names = {l["name"].lower() for l in it.get("labels", [])}
        hits = [g for g, labs in groups.items() if names & labs]
        if len(hits) != 1:
            skipped["no single matching label" if not hits else "matches several labels"] += 1
            continue
        out.append({"number": it["number"], "created_at": it["created_at"], "label": hits[0],
                    "text": clean_issue(it["title"], it.get("body"))})
        if len(out) >= limit:
            break
    return out, skipped


def fetch_prs(gh, repo, since, limit, max_pages, diff_cap=DIFF_CAP):
    out, skipped = [], Counter()
    for pr in gh.pages(f"/repos/{repo}/pulls", {"state": "closed", "sort": "created", "direction": "desc"}, max_pages):
        if pr["created_at"][:10] < since:
            break  # sorted newest first, so everything after this is older
        if not pr.get("merged_at"):
            skipped["not merged"] += 1
            continue
        if pr.get("draft") or (pr.get("user") or {}).get("type") == "Bot":
            skipped["draft or bot"] += 1
            continue
        author = (pr.get("user") or {}).get("login")
        try:
            reviews = list(gh.pages(f"/repos/{repo}/pulls/{pr['number']}/reviews", {}, 5, cache=True))
        except urllib.error.HTTPError as e:
            skipped[f"reviews unavailable (HTTP {e.code})"] += 1
            continue
        states = [r["state"] for r in reviews
                  if (r.get("user") or {}).get("login") != author and (r.get("user") or {}).get("type") != "Bot"]
        if "CHANGES_REQUESTED" in states or "COMMENTED" in states:
            label = "yes"
        elif "APPROVED" in states:
            label = "no"
        else:
            skipped["no review by anyone else"] += 1
            continue
        try:
            diff = gh.get(f"/repos/{repo}/pulls/{pr['number']}", accept="application/vnd.github.diff", raw=True, cache=True)
        except urllib.error.HTTPError as e:
            # GitHub answers 406 when a diff is too large to serve; such a pull request is over the cap anyway.
            skipped[f"diff over {diff_cap} characters" if e.code == 406 else f"diff unavailable (HTTP {e.code})"] += 1
            continue
        if len(diff) > diff_cap:
            skipped[f"diff over {diff_cap} characters"] += 1
            continue
        desc = re.sub(r"<!--.*?-->", " ", pr.get("body") or "", flags=re.S).strip()[:DESC_CAP]
        out.append({"number": pr["number"], "created_at": pr["created_at"], "label": label,
                    "text": f"Title: {pr['title'].strip()}\n\n{desc}\n\nDiff:\n{diff}"})
        if len(out) >= limit:
            break
        if len(out) % 25 == 0:
            print(f"  {len(out)} pull requests so far ({gh.calls} API calls, {gh.cached} answers reused from disk)", flush=True)
    return out, skipped


def split(rows, task, prefix, calib_share, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    out = []
    for label in sorted(by):
        items = by[label][:]
        rng.shuffle(items)
        n_cal = round(len(items) * calib_share)
        for i, r in enumerate(items):
            sp = "calib" if i < n_cal else "eval"
            out.append({"id": f"{prefix}-{r['number']}", "task": task, "split": sp, "label": r["label"],
                        "text": r["text"], "created_at": r["created_at"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="OWNER/REPO")
    ap.add_argument("--list-labels", action="store_true")
    ap.add_argument("--bug", default="bug", help="comma-separated label names meaning bug")
    ap.add_argument("--feature", default="feature,enhancement,feature request", help="labels meaning feature")
    ap.add_argument("--question", default="question,support", help="labels meaning question")
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--max-issues", type=int, default=600)
    ap.add_argument("--max-prs", type=int, default=400)
    ap.add_argument("--max-pages", type=int, default=30, help="pages of 100 to scan per list")
    ap.add_argument("--via-search", action="store_true",
                    help="list issues through date-split search, for repositories too busy to page through (GitHub stops at about 10,000 items)")
    ap.add_argument("--calib-share", type=float, default=0.3)
    ap.add_argument("--skip-prs", action="store_true")
    ap.add_argument("--skip-issues", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ask GitHub again instead of reusing the local copy of earlier answers")
    ap.add_argument("--max-diff-chars", type=int, default=DIFF_CAP,
                    help=f"exclude pull requests whose diff is longer (default {DIFF_CAP}); larger values keep more of the changes reviewers discuss")
    a = ap.parse_args()

    slug = a.repo.replace("/", "__")
    gh = GitHub(os.environ.get("GITHUB_TOKEN"), HERE / "kit" / slug / "cache", refresh=a.refresh)
    if not gh.token:
        print("No GITHUB_TOKEN set: GitHub allows only 60 requests an hour without one, which is too few for pull requests.")
    if a.list_labels:
        return list_labels(gh, a.repo)

    groups = {g: {x.strip().lower() for x in getattr(a, g).split(",") if x.strip()} for g in ("bug", "feature", "question")}
    out_dir = HERE / "kit" / slug / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").exists() else {}
    manifest = {"repo": a.repo, "since": a.since, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "label_groups": {g: sorted(v) for g, v in groups.items()}, "caps": {"issue_chars": ISSUE_CAP,
                "pr_description_chars": DESC_CAP, "pr_diff_chars_excluded_above": a.max_diff_chars}, "calib_share": a.calib_share}

    if a.skip_issues and "issues" in old:
        manifest["issues"] = old["issues"]
    elif a.skip_issues and (out_dir / "issues.jsonl").exists():
        rows = [json.loads(l) for l in (out_dir / "issues.jsonl").open(encoding="utf-8")]
        manifest["issues"] = {"kept": dict(Counter(r["label"] for r in rows)), "splits": dict(Counter(r["split"] for r in rows)),
                              "note": "kept from an earlier run; skipped counts not recorded"}
    if not a.skip_issues:
        print(f"Fetching issues from {a.repo} created since {a.since} ...", flush=True)
        issues, skipped = fetch_issues(gh, a.repo, groups, a.since, a.max_issues, a.max_pages, a.via_search)
        rows = split(issues, "issue_type", "issue", a.calib_share, random.Random(SEED))
        with (out_dir / "issues.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["issues"] = {"kept": dict(Counter(r["label"] for r in rows)), "skipped": dict(skipped),
                              "splits": dict(Counter(r["split"] for r in rows))}
        print(f"  kept {len(rows)} issues: {manifest['issues']['kept']}")
    if a.skip_prs and "prs" in old:
        manifest["prs"] = old["prs"]
    if not a.skip_prs:
        print(f"Fetching merged pull requests from {a.repo} created since {a.since} ...", flush=True)
        prs, skipped = fetch_prs(gh, a.repo, a.since, a.max_prs, a.max_pages, a.max_diff_chars)
        rows = split(prs, "pr_needs_changes", "pr", a.calib_share, random.Random(SEED + 1))  # own seed: same split whether or not issues ran
        with (out_dir / "prs.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["prs"] = {"kept": dict(Counter(r["label"] for r in rows)), "skipped": dict(skipped),
                           "splits": dict(Counter(r["split"] for r in rows))}
        print(f"  kept {len(rows)} pull requests: {manifest['prs']['kept']}")
    manifest["api_calls"] = gh.calls
    manifest["answers_reused_from_disk"] = gh.cached
    manifest["snapshot"] = gh.snapshot()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_dir.relative_to(HERE)} ({gh.calls} GitHub API calls). Next:\n"
          f"  python3 run.py --data-dir kit/{slug}/data --results-dir kit/{slug}/results --smoke")


if __name__ == "__main__":
    main()
