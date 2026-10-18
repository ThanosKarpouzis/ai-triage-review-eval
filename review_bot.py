#!/usr/bin/env python3
"""Is your review bot earning its place? Standard library plus git.

For every inline review comment in a repository's merged pull requests, three outcomes:
  acted on      were the flagged lines changed before merge?
  fixed later   was the flagged region changed within 90 days of merge by a bug-fix commit?
  placebo       the same two outcomes for matched lines in the same pull request that nobody flagged.
The sharp number is the excess: how much more often an ignored flag is followed by a later fix than its
placebo. It is computed for the AI reviewer and for the human reviewers, side by side, overall and split
by front end / back end and by new capability / bug fix.

  export GITHUB_TOKEN=...   # required: review threads come from GitHub's GraphQL interface
  python3 review_bot.py fetch  --config examples/sentry-review-config.json [--smoke]
  python3 review_bot.py map    --config C --clone PATH/TO/CLONE
  python3 review_bot.py fixes  --config C --clone PATH/TO/CLONE
  python3 review_bot.py report --config C [--null]
  python3 review_bot.py sample --config C --n 30
  python3 review_bot.py verdict --config C1 C2 C3

Rules, fixed before any result (see the README and the study spec):
- Merged pull requests created in the config's window, authored by a person, newest first, up to
  --max-prs (1,500) that carry at least one qualifying comment.
- Qualifying comment: inline, first in its thread, on the new side of the diff, on lines (not a whole
  file), not by the pull request's author. "ai" when the author's login is in the config's ai_reviewers;
  "human" when GitHub marks the author as a user and the login is not a known bot; anything else is out.
- Acted on: any flagged line edited or removed between the commented commit and the merge commit.
- Fixed later: any line of the flagged region, as it stands at merge, changed within fix_window_days of
  merge by a fixing commit on the default branch. A commit is fixing when its pull request carries a bug
  label, or closes an issue carrying one, or it reverts the merged pull request. Changes by any other
  commit are churn. A secondary, looser definition also counts a conventional "fix" title prefix; it is
  reported separately and never replaces the main one.
- Placebo: same number of lines, same commit, among lines the pull request added or changed, at least
  five lines from any comment, same file first, fixed seed.
- 95% intervals by bootstrap over whole pull requests, 2,000 draws.
"""
import argparse, difflib, hashlib, http.client, json, math, os, random, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_repo import GitHub, API  # the kit's cached REST client

GAP = 5            # placebo must sit at least this many lines from any comment
BOOT = 2000
THIN = 100         # fewer ignored AI flags than this in a cell: "too thin"

# ----------------------------------------------------------------------------------------- utilities

def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    c.setdefault("fix_window_days", 90)
    c.setdefault("seed", 20261014)
    c.setdefault("excluded_bots", [])
    c.setdefault("area_labels", {})
    c.setdefault("path_rules", [])
    c.setdefault("capability_labels", [])
    c.setdefault("title_rules", {"bugfix": ["fix"], "capability": ["feat"]})
    c["_dir"] = HERE / "kit" / c["repo"].replace("/", "__")
    if "out_dir" in c:
        c["_dir"] = (Path(path).parent / c["out_dir"]).resolve()
    return c


def out(c, name):
    d = c["_dir"] / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p, rows):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def lower_set(xs):
    return {x.lower() for x in xs}

# ----------------------------------------------------------------------------------------- GitHub

class GraphQLFailed(Exception):
    pass


class GH(GitHub):
    """The kit's cached REST client plus a cached GraphQL call."""

    def _fetch(self, url, accept):
        """The kit's REST fetch, also retrying when GitHub drops the connection (18 October 2026: a
        RemoteDisconnected stopped a three-hour map run)."""
        for attempt in range(6):
            try:
                return super()._fetch(url, accept)
            except (http.client.HTTPException, ConnectionError) as e:
                wait = 5 * 2 ** attempt
                print(f"  connection to GitHub dropped ({type(e).__name__}), retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise SystemExit(f"GitHub kept dropping the connection for {url}; run the same command again later")

    def graphql(self, query, variables):
        key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()[:32]
        cfile = self.cache_dir / f"gql-{key}.json" if self.cache_dir else None
        if cfile and cfile.exists() and not self.refresh:
            self.cached += 1
            hit = json.loads(cfile.read_text(encoding="utf-8"))
            self.fetched_at.append(hit.get("fetched"))
            return hit["body"]
        if not self.token:
            raise SystemExit("GITHUB_TOKEN is required: GitHub's GraphQL interface does not answer without one.")
        payload = json.dumps({"query": query, "variables": variables}).encode()
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "review-bot-kit",
                   "Content-Type": "application/json"}
        attempt = 0
        while attempt < 6:
            self.calls += 1
            try:
                req = urllib.request.Request(API + "/graphql", data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = json.loads(r.read().decode("utf-8", errors="replace"))
                    left, reset = r.headers.get("x-ratelimit-remaining"), r.headers.get("x-ratelimit-reset")
            except urllib.error.HTTPError as e:
                if e.headers.get("x-ratelimit-remaining") == "0" or e.code == 429:
                    self.wait_for_reset(e.headers.get("x-ratelimit-reset"))
                    continue  # waiting for the hourly allowance is not a failed attempt
                if e.code in (403, 502, 503, 504):
                    wait = 5 * 2 ** attempt
                    print(f"  GitHub said {e.code}, waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                time.sleep(5 * 2 ** attempt)
                attempt += 1
                continue
            if body.get("errors") and not body.get("data"):
                msg = json.dumps(body["errors"])[:500]
                if "RATE_LIMIT" in msg:
                    self.wait_for_reset(None)
                    continue
                raise SystemExit(f"GraphQL error: {msg}")
            self.fetched_at.append(time.strftime("%Y-%m-%dT%H:%M:%S"))
            if cfile:
                cfile.write_text(json.dumps({"body": body, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                 encoding="utf-8")
            if left is not None and int(left) < 100:
                self.wait_for_reset(reset)  # pause before the allowance runs out, not after
            return body
        raise GraphQLFailed("GitHub GraphQL kept failing")


def _wait_for_reset(self, reset):
    """Sleep until GitHub's hourly GraphQL allowance resets (17 October 2026). Asking /rate_limit is free."""
    if not reset:
        try:
            req = urllib.request.Request(API + "/rate_limit", headers={
                "Authorization": f"Bearer {self.token}", "User-Agent": "review-bot-kit"})
            with urllib.request.urlopen(req, timeout=60) as r:
                reset = json.loads(r.read())["resources"]["graphql"]["reset"]
        except Exception:
            reset = time.time() + 600
    wait = max(30, int(reset) - time.time() + 5)
    print(f"  GitHub's hourly GraphQL allowance is used up; waiting {wait / 60:.0f} minutes "
          f"(until {time.strftime('%H:%M', time.localtime(time.time() + wait))}). Leave it running.", flush=True)
    time.sleep(wait)


GH.wait_for_reset = _wait_for_reset


LIST_Q = """query($owner:String!,$name:String!,$after:String){repository(owner:$owner,name:$name){
pullRequests(states:MERGED,orderBy:{field:CREATED_AT,direction:DESC},first:100,after:$after){
pageInfo{hasNextPage endCursor} nodes{number createdAt author{__typename login}}}}}"""

PR_FIELDS = """number title createdAt mergedAt baseRefOid headRefOid
author{__typename login} mergeCommit{oid}
labels(first:50){nodes{name}}
closingIssuesReferences(first:10){nodes{number labels(first:30){nodes{name}}}}
reviewThreads(first:100){pageInfo{hasNextPage} nodes{
  isResolved resolvedBy{login} path diffSide line startLine originalLine originalStartLine subjectType
  comments(first:1){nodes{databaseId createdAt url body author{__typename login} originalCommit{oid}}}}}"""

INFO_FIELDS = """number title mergedAt labels(first:50){nodes{name}}
closingIssuesReferences(first:10){nodes{number labels(first:30){nodes{name}}}}"""


def batch_query(owner, name, numbers, fields):
    parts = [f"p{n}: pullRequest(number:{n}){{{fields}}}" for n in numbers]
    return ("query($owner:String!,$name:String!){repository(owner:$owner,name:$name){"
            + " ".join(parts) + "}}")


def gh_client(c, refresh=False):
    return GH(os.environ.get("GITHUB_TOKEN"), c["_dir"] / "cache", refresh)

# ----------------------------------------------------------------------------------------- classify

COPILOT = {"copilot", "copilot[bot]", "copilot-pull-request-reviewer", "copilot-pull-request-reviewer[bot]"}


def norm_author(author):
    """GraphQL gives bot logins without the "[bot]" suffix that REST and the web show (17 October 2026);
    put it back so configs can use the names people see. GitHub's built-in reviewer becomes "Copilot"."""
    if not author:
        return author
    login = author.get("login") or ""
    if login.lower() in COPILOT:
        return dict(author, login="Copilot")
    if author.get("__typename") == "Bot" and not login.endswith("[bot]"):
        return dict(author, login=login + "[bot]")
    return author


def reviewer_kind(c, author):
    if not author:
        return None
    login, typ = author.get("login") or "", author.get("__typename")
    if login in c["ai_reviewers"]:
        return "ai"
    if login in c["excluded_bots"] or typ == "Bot" or login.endswith("[bot]"):
        return None
    return "human" if typ == "User" else None


def classify_pr(c, pr):
    """bugfix / capability / unclassified, with where the answer came from."""
    labels = lower_set(pr["labels"])
    bug, cap = lower_set(c["bug_labels_pr"]), lower_set(c["capability_labels"])
    if labels & bug:
        return "bugfix", "label"
    if labels & cap:
        return "capability", "label"
    ibug, icap = lower_set(c["bug_labels_issue"]), lower_set(c.get("capability_labels_issue", []))
    for iss in pr["closing_issues"]:
        il = lower_set(iss["labels"])
        if il & ibug:
            return "bugfix", "issue"
        if il & icap:
            return "capability", "issue"
    title = pr["title"].strip().lower()
    for kind in ("bugfix", "capability"):
        for p in c["title_rules"].get(kind, []):
            if re.match(re.escape(p.lower()) + r"(\b|\(|:|!)", title):
                return kind, "title"
    return "unclassified", "none"


def area_of_path(c, path):
    for rule in c["path_rules"]:
        m = rule["match"]
        if (m.startswith(".") and path.endswith(m)) or (not m.startswith(".") and path.startswith(m)):
            return rule["area"]
    return "none"


def area_of_pr(c, pr):
    labels = lower_set(pr["labels"])
    found = {area for lab, area in c["area_labels"].items() if lab.lower() in labels}
    if len(found) == 1:
        return found.pop()
    return "both" if found else "none"

# ----------------------------------------------------------------------------------------- fetch

def fetch_details(gh, owner, name, nums, skip):
    """Ten pull requests per call; a batch GitHub keeps failing on (usually a timeout on pull requests
    with very many review threads) is split, and a single pull request that still fails is skipped and
    counted (17 October 2026)."""
    try:
        d = gh.graphql(batch_query(owner, name, nums, PR_FIELDS), {"owner": owner, "name": name})
        return d["data"]["repository"] or {}
    except GraphQLFailed:
        if len(nums) == 1:
            skip["pr details failed at GitHub (skipped)"] += 1
            print(f"  skipping #{nums[0]}: GitHub kept failing on it", flush=True)
            return {}
        print(f"  splitting a batch of {len(nums)} that GitHub kept failing on", flush=True)
        half = len(nums) // 2
        res = dict(fetch_details(gh, owner, name, nums[:half], skip))
        res.update(fetch_details(gh, owner, name, nums[half:], skip))
        return res


def cmd_fetch(a):
    c = load_config(a.config)
    gh = gh_client(c, a.refresh)
    owner, name = c["repo"].split("/")
    start, end = ts(c["window"][0] + "T00:00:00Z"), ts(c["window"][1] + "T23:59:59Z")
    max_prs = 5 if a.smoke else a.max_prs
    # 1. merged pull requests in the window, newest first
    cands, after, pages = [], None, 0
    skip = Counter()
    while True:
        d = gh.graphql(LIST_Q, {"owner": owner, "name": name, "after": after})
        conn = d["data"]["repository"]["pullRequests"]
        pages += 1
        stop = False
        for n in conn["nodes"]:
            t = ts(n["createdAt"])
            if t > end:
                continue
            if t < start:
                stop = True
                break
            au = norm_author(n.get("author")) or {}
            if au.get("__typename") != "User" or (au.get("login") or "").endswith("[bot]"):
                skip["pr author not a person"] += 1
                continue
            cands.append(n["number"])
        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
        if pages % 10 == 0:
            print(f"  listed {pages} pages, {len(cands)} candidates so far", flush=True)
    print(f"{len(cands)} merged pull requests by people in the window", flush=True)
    # 2. details, ten at a time, until max_prs carry a qualifying comment
    prs, comments, logins = [], [], Counter()
    for i in range(0, len(cands), 10):
        if len(prs) >= max_prs:
            break
        nums = cands[i:i + 10]
        repo = fetch_details(gh, owner, name, nums, skip)
        for n in nums:
            p = repo.get(f"p{n}")
            if not p or len(prs) >= max_prs:
                continue
            if not p.get("mergeCommit"):
                skip["no merge commit"] += 1
                continue
            pauthor = (norm_author(p.get("author")) or {}).get("login")
            threads = p["reviewThreads"]
            if threads["pageInfo"]["hasNextPage"]:
                skip["over 100 review threads (first 100 used)"] += 1
            kept = []
            for th in threads["nodes"]:
                cs = th["comments"]["nodes"]
                if not cs:
                    continue
                cm = cs[0]
                au = norm_author(cm.get("author")) or {}
                if au.get("__typename") == "Bot" or (au.get("login") or "").endswith("[bot]") or au.get("login") == "Copilot":
                    logins[au.get("login")] += 1
                kind = reviewer_kind(c, au)
                why = None
                if kind is None:
                    why = "author not an AI reviewer or a person"
                elif au.get("login") == pauthor:
                    why = "by the pull request's author"
                elif th.get("subjectType") == "FILE" or not th.get("originalLine"):
                    why = "not on lines"
                elif th.get("diffSide") != "RIGHT":
                    why = "old side of the diff"
                elif not (cm.get("originalCommit") or {}).get("oid"):
                    why = "no commit"
                if why:
                    skip["comment: " + why] += 1
                    continue
                end_l = th["originalLine"]
                start_l = th.get("originalStartLine") or end_l
                if start_l > end_l:
                    start_l, end_l = end_l, start_l
                kept.append({
                    "id": cm["databaseId"], "pr": n, "kind": kind, "login": au["login"],
                    "path": th["path"], "start": start_l, "end": end_l,
                    "commit": cm["originalCommit"]["oid"], "created": cm["createdAt"],
                    "resolved": bool(th.get("isResolved")),
                    "resolved_by_bot": bool(th.get("isResolved") and th.get("resolvedBy")
                                            and (th["resolvedBy"]["login"] or "").endswith("[bot]")),
                    "url": cm.get("url"), "body": (cm.get("body") or "")[:4000]})
            if not kept:
                skip["pr without a qualifying comment"] += 1
                continue
            prs.append({
                "number": n, "title": p["title"], "created": p["createdAt"], "merged": p["mergedAt"],
                "author": pauthor, "base": p["baseRefOid"], "head": p["headRefOid"],
                "merge": p["mergeCommit"]["oid"],
                "labels": [l["name"] for l in p["labels"]["nodes"]],
                "closing_issues": [{"number": x["number"], "labels": [l["name"] for l in x["labels"]["nodes"]]}
                                   for x in p["closingIssuesReferences"]["nodes"]]})
            comments.extend(kept)
        if (i // 10) % 20 == 0:
            print(f"  {len(prs)} pull requests kept, {len(comments)} comments", flush=True)
    prs.sort(key=lambda r: r["number"])
    comments.sort(key=lambda r: r["id"])
    write_jsonl(out(c, "prs.jsonl"), prs)
    write_jsonl(out(c, "comments.jsonl"), comments)
    kinds = Counter(x["kind"] for x in comments)
    ai_first = min((x["created"] for x in comments if x["kind"] == "ai"), default=None)
    manifest = {"repo": c["repo"], "window": c["window"], "smoke": a.smoke,
                "pull_requests_kept": len(prs), "comments_by_kind": dict(kinds),
                "comments_by_ai_login": dict(Counter(x["login"] for x in comments if x["kind"] == "ai")),
                "bot_logins_seen_on_first_comments": dict(logins.most_common()),
                "first_ai_comment": ai_first, "skipped": dict(skip), "github": gh.snapshot(),
                "enough": kinds["ai"] >= 300 and kinds["human"] >= 300}
    out(c, "manifest-fetch.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    unknown = [l for l in logins if l not in c["ai_reviewers"] and l not in c["excluded_bots"]]
    if unknown:
        print("\nBot logins not in the config (classify them before computing outcomes):", ", ".join(unknown))

# ----------------------------------------------------------------------------------------- git and files

class Repo:
    """File contents and diffs: the local clone first, GitHub's API for commits the clone lacks
    (pull request branch commits are usually not on the default branch)."""

    def __init__(self, clone, c, gh=None):
        self.clone, self.c, self.gh = str(clone), c, gh
        self._have = {}

    def git(self, *args, check=True):
        r = subprocess.run(["git", "-C", self.clone, *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
        return r

    def has(self, sha):
        if sha not in self._have:
            self._have[sha] = self.git("cat-file", "-e", sha + "^{commit}", check=False).returncode == 0
        return self._have[sha]

    def file_at(self, sha, path):
        """Lines of the file, or None if it does not exist at that commit."""
        if self.has(sha):
            r = self.git("show", f"{sha}:{path}", check=False)
            return r.stdout.splitlines() if r.returncode == 0 else None
        if not self.gh:
            return None
        try:
            body = self.gh.get(f"/repos/{self.c['repo']}/contents/{urllib.parse.quote(path)}",
                               {"ref": sha}, accept="application/vnd.github.raw", raw=True)
        except urllib.error.HTTPError:
            return None
        return body.splitlines()

    def added_lines(self, base, sha):
        """{path: sorted new-side line numbers the pull request added or changed}, merge-base to sha."""
        if self.has(sha) and self.has(base):
            mb = self.git("merge-base", base, sha, check=False).stdout.strip() or base
            patch = self.git("diff", "-U0", "--no-color", "--no-renames", mb, sha).stdout
            return parse_added(patch)
        if not self.gh:
            return {}
        try:
            d = self.gh.get(f"/repos/{self.c['repo']}/compare/{base}...{sha}")
        except urllib.error.HTTPError:
            return {}
        res = {}
        for f in d.get("files", []):
            if f.get("patch"):
                lines = parse_added("diff --git a/x b/x\n+++ b/" + f["filename"] + "\n" + f["patch"]).get(f["filename"])
                if lines:
                    res[f["filename"]] = lines
        return res


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_added(patch):
    """New-side line numbers of '+' lines, per file, from a unified diff (any context)."""
    res, path, newl = defaultdict(list), None, None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            continue
        if line.startswith("diff --git") or line.startswith("--- "):
            continue
        m = HUNK.match(line)
        if m:
            newl = int(m.group(3))
            continue
        if path is None or newl is None:
            continue
        if line.startswith("+"):
            res[path].append(newl)
            newl += 1
        elif line.startswith("-") or line.startswith("\\"):
            pass
        else:
            newl += 1
    return {p: sorted(set(v)) for p, v in res.items()}


def map_region(old, new, s, e):
    """Follow lines s..e (1-based) of `old` into `new`. Returns (changed, (ns, ne) or None)."""
    if new is None:
        return True, None
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    changed, js = False, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        lo, hi = max(i1, s - 1), min(i2, e)
        if lo >= hi and not (tag == "insert" and s - 1 < i1 < e):
            continue
        if tag == "equal":
            js += [j1 + (lo - i1) + 1, j1 + (hi - i1)]
        elif tag == "insert":
            changed = True  # new lines inside the region
            js += [j1 + 1, j2]
        else:
            changed = True
            if j2 > j1:
                js += [j1 + 1, j2]
    if not js:
        return changed, None
    return changed, (min(js), max(js))

# ----------------------------------------------------------------------------------------- map

def choose_placebo(rng, length, path, added, taken):
    """A run of `length` added lines, GAP away from any taken line; same file first."""
    def fits(p):
        lines = added.get(p, [])
        S = set(lines)
        bad = taken.get(p, [])
        starts = []
        for st in lines:
            if all((st + k) in S for k in range(length)):
                if all(not (st - GAP <= b <= st + length - 1 + GAP) for b in bad):
                    starts.append(st)
        return starts
    for p in [path] + sorted(x for x in added if x != path):
        starts = fits(p)
        if starts:
            st = rng.choice(starts)
            return {"path": p, "start": st, "end": st + length - 1, "same_file": p == path}
    return None


def cmd_map(a):
    c = load_config(a.config)
    gh = None if a.offline else gh_client(c)
    repo = Repo(a.clone, c, gh)
    prs = {p["number"]: p for p in read_jsonl(out(c, "prs.jsonl"))}
    comments = read_jsonl(out(c, "comments.jsonl"))
    rng = random.Random(c["seed"])
    rng2 = random.Random(c["seed"] + 1)
    by_pr = defaultdict(list)
    for x in comments:
        by_pr[x["pr"]].append(x)
    rows, skip = [], Counter()
    files = {}

    def fa(sha, path):
        k = (sha, path)
        if k not in files:
            files[k] = repo.file_at(sha, path)
        return files[k]

    for n in sorted(by_pr):
        pr = prs[n]
        cs = sorted(by_pr[n], key=lambda x: x["id"])
        taken = defaultdict(list)
        for x in cs:
            taken[x["path"]] += list(range(x["start"], x["end"] + 1))
        added_cache = {}
        for x in cs:
            r = dict(x)
            old = fa(x["commit"], x["path"])
            if old is None or x["end"] > len(old):
                r["status"] = "unmappable"
                skip["comment: file not readable at the commented commit"] += 1
                rows.append(r)
                continue
            new = fa(pr["merge"], x["path"])
            acted, region = map_region(old, new, x["start"], x["end"])
            r.update(status="ok", acted_on=acted, region_at_merge=region)
            if x["commit"] not in added_cache:
                added_cache[x["commit"]] = repo.added_lines(pr["base"], x["commit"])
            added = added_cache[x["commit"]]
            length = x["end"] - x["start"] + 1
            placebos = []
            for g in (rng, rng2):
                pl = choose_placebo(g, length, x["path"], added, taken)
                if pl:
                    pold = fa(x["commit"], pl["path"])
                    pnew = fa(pr["merge"], pl["path"])
                    if pold is None or pl["end"] > len(pold):
                        pl = None
                    else:
                        pa, preg = map_region(pold, pnew, pl["start"], pl["end"])
                        pl.update(acted_on=pa, region_at_merge=preg)
                placebos.append(pl)
            r["placebo"], r["placebo2"] = placebos
            if not placebos[0]:
                skip["comment: no placebo range available"] += 1
            rows.append(r)
    write_jsonl(out(c, "mapped.jsonl"), rows)
    m = {"comments": len(rows), "ok": sum(1 for r in rows if r["status"] == "ok"),
         "with_placebo": sum(1 for r in rows if r.get("placebo")),
         "placebo_same_file": sum(1 for r in rows if r.get("placebo") and r["placebo"]["same_file"]),
         "skipped": dict(skip)}
    out(c, "manifest-map.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))

# ----------------------------------------------------------------------------------------- fixes

def track(repo, path, region, merge, until):
    """Follow `region` (at the merge commit) forward along the default branch's first-parent history
    until `until`. Returns the list of commits whose change touched the region."""
    if region is None:
        return []
    s, e = region
    log = repo.git("log", "--first-parent", "--reverse", "--format=%H %ct %P", "--no-renames",
                   f"{merge}..{repo.default}", "--", path, check=False).stdout
    touched = []
    for line in log.splitlines():
        parts = line.split()
        sha, when = parts[0], int(parts[1])
        if when > until:
            break
        parent = parts[2] if len(parts) > 2 else None
        if parent is None:
            continue
        patch = repo.git("diff", "-U0", "--no-color", "--no-renames", parent, sha, "--", path, check=False).stdout
        if not patch.strip():
            continue
        if "\n+++ /dev/null" in patch:
            touched.append(sha)
            return touched  # the file is gone; the region ends here
        hit, ds, de = False, 0, 0
        for h in patch.splitlines():
            m = HUNK.match(h)
            if not m:
                continue
            a0 = int(m.group(1)); b = int(m.group(2)) if m.group(2) is not None else 1
            d = int(m.group(4)) if m.group(4) is not None else 1
            if b == 0:  # pure insertion after old line a0
                if s <= a0 < e:
                    hit = True
                if a0 < s:
                    ds += d
                if a0 < e:
                    de += d
                continue
            oend = a0 + b - 1
            if not (oend < s or a0 > e):
                hit = True
            if oend < s:
                ds += d - b
            if a0 <= e:
                de += d - b
        if hit:
            touched.append(sha)
        s, e = s + ds, max(s + ds, e + de)
    return touched


PRNUM = re.compile(r"\(#(\d+)\)\s*$|^Merge pull request #(\d+)")


def cmd_fixes(a):
    c = load_config(a.config)
    repo = Repo(a.clone, c)
    repo.default = a.branch or repo.git("symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False).stdout.strip() or "HEAD"
    prs = {p["number"]: p for p in read_jsonl(out(c, "prs.jsonl"))}
    rows = read_jsonl(out(c, "mapped.jsonl"))
    days = c["fix_window_days"]
    cache_p = out(c, "track-cache.json")
    tcache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    commits_seen = set()
    done = 0

    def regions(r):
        yield "comment", r
        for k in ("placebo", "placebo2"):
            if r.get(k):
                yield k, r[k]

    for r in rows:
        if r["status"] != "ok":
            continue
        pr = prs[r["pr"]]
        until = int((ts(pr["merged"]) + timedelta(days=days)).timestamp())
        for kind, reg in regions(r):
            path = reg["path"]
            region = reg.get("region_at_merge")
            key = f"{pr['merge']}|{path}|{region}|{until}"
            if key not in tcache:
                tcache[key] = track(repo, path, tuple(region) if region else None, pr["merge"], until)
            reg["touched_by"] = tcache[key]
            commits_seen.update(tcache[key])
        done += 1
        if done % 200 == 0:
            cache_p.write_text(json.dumps(tcache), encoding="utf-8")
            print(f"  {done} comments tracked", flush=True)
    cache_p.write_text(json.dumps(tcache), encoding="utf-8")
    # which pull request each touching commit came from
    commit_pr, commit_subject = {}, {}
    for sha in sorted(commits_seen):
        subj = repo.git("log", "-1", "--format=%s", sha).stdout.strip()
        commit_subject[sha] = subj
        m = PRNUM.search(subj)
        if m:
            commit_pr[sha] = int(m.group(1) or m.group(2))
    info = load_pr_info(c, sorted(set(commit_pr.values()) - set(prs)), a.offline)
    info.update({n: {"labels": p["labels"], "closing_issues": p["closing_issues"], "title": p["title"]}
                 for n, p in prs.items()})
    bug_pr, bug_issue = lower_set(c["bug_labels_pr"]), lower_set(c["bug_labels_issue"])
    fixrules = [p.lower() for p in c["title_rules"].get("bugfix", [])]

    def is_fix(sha, orig_pr):
        n = commit_pr.get(sha)
        subj = commit_subject[sha]
        if subj.startswith("Revert") and (f"#{orig_pr['number']}" in subj or orig_pr["title"] in subj):
            return True, True
        strict = False
        if n is not None and n in info:
            i = info[n]
            strict = bool(lower_set(i["labels"]) & bug_pr) or any(
                lower_set(x["labels"]) & bug_issue for x in i["closing_issues"])
        loose = strict or any(re.match(re.escape(p) + r"(\b|\(|:|!)", subj.lower()) for p in fixrules)
        return strict, loose

    for r in rows:
        if r["status"] != "ok":
            continue
        pr = prs[r["pr"]]
        for kind, reg in regions(r):
            touched = reg.get("touched_by", [])
            flags = [is_fix(s, pr) for s in touched]
            reg["fixed"] = any(f[0] for f in flags)
            reg["fixed_loose"] = any(f[1] for f in flags)
            reg["churned"] = any(not f[0] for f in flags)
            reg["fix_commits"] = [s for s, f in zip(touched, flags) if f[0]]
    write_jsonl(out(c, "outcomes.jsonl"), rows)
    m = {"comments_tracked": done, "touching_commits": len(commits_seen),
         "touching_commits_with_pr": len(commit_pr),
         "fixed_comments": sum(1 for r in rows if r.get("fixed")),
         "fixed_comments_loose": sum(1 for r in rows if r.get("fixed_loose")),
         "default_branch": repo.default}
    out(c, "manifest-fixes.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))


def load_pr_info(c, numbers, offline=False):
    """Labels and closing issues of the pull requests behind later commits (cached)."""
    p = out(c, "pr-info.jsonl")
    have = {r["number"]: r for r in read_jsonl(p)}
    need = [n for n in numbers if n not in have]
    if need and not offline:
        gh = gh_client(c)
        owner, name = c["repo"].split("/")
        for i in range(0, len(need), 20):
            nums = need[i:i + 20]
            d = gh.graphql(batch_query(owner, name, nums, INFO_FIELDS), {"owner": owner, "name": name})
            for n in nums:
                x = (d.get("data") or {}).get("repository", {}).get(f"p{n}")
                if x:
                    have[n] = {"number": n, "title": x["title"], "labels": [l["name"] for l in x["labels"]["nodes"]],
                               "closing_issues": [{"number": y["number"], "labels": [l["name"] for l in y["labels"]["nodes"]]}
                                                  for y in x["closingIssuesReferences"]["nodes"]]}
        write_jsonl(p, [have[k] for k in sorted(have)])
    return have

# ----------------------------------------------------------------------------------------- report

def rate(xs):
    return sum(xs) / len(xs) if xs else None


def boot_ci(units, stat, seed):
    """95% interval of stat(list of units), resampling units (pull requests)."""
    if not units:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOT):
        s = [units[rng.randrange(len(units))] for _ in units]
        v = stat(s)
        if v is not None:
            vals.append(v)
    if len(vals) < BOOT * 0.9:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def cell_stats(rows, seed, null=False, main="fixed", other="fixed_loose"):
    """rows: comment outcomes of one reviewer kind in one cell. `main` is the fix signal that carries the
    predictions, `other` the secondary one (18 October 2026)."""
    if null:  # a pair of identical placebo ranges says nothing about matching bias, so leave it out
        rows = [r for r in rows if r.get("placebo") and r.get("placebo2") and
                (r["placebo"]["path"], r["placebo"]["start"]) != (r["placebo2"]["path"], r["placebo2"]["start"])]

    def pair(r):
        if null:  # second placebo stands in for the comment, first placebo is the reference
            return r.get("placebo2"), r.get("placebo")
        return r, r.get("placebo")
    usable = [r for r in rows if r["status"] == "ok"]
    by_pr = defaultdict(list)
    for r in usable:
        by_pr[r["pr"]].append(r)
    units = [by_pr[k] for k in sorted(by_pr)]

    def acted(us):
        xs = [pair(r)[0]["acted_on"] for u in us for r in u if pair(r)[0]]
        return rate(xs)

    def ignored(us):
        # An ignored flag is compared with a placebo that was also left untouched before merge: lines the
        # author was still editing are more likely to change again later, so an unconditioned placebo
        # would bias the excess downwards (found by the null check, 18 October 2026).
        return [pair(r) for u in us for r in u if pair(r)[0] and pair(r)[1]
                and not pair(r)[0]["acted_on"] and not pair(r)[1]["acted_on"]]

    def excess(us):
        ps = ignored(us)
        if not ps:
            return None
        return rate([p[0][main] for p in ps]) - rate([p[1][main] for p in ps])

    def excess_other(us):
        ps = ignored(us)
        if not ps:
            return None
        return rate([p[0][other] for p in ps]) - rate([p[1][other] for p in ps])

    def acted_excess(us):
        xs = [(pair(r)[0], pair(r)[1]) for u in us for r in u if pair(r)[0] and pair(r)[1]]
        if not xs:
            return None
        return rate([a["acted_on"] for a, _ in xs]) - rate([b["acted_on"] for _, b in xs])

    ign = ignored(units)
    heeded = [pair(r)[0] for u in units for r in u if pair(r)[0] and pair(r)[0]["acted_on"]]
    heeded_pl = [pair(r)[1] for u in units for r in u if pair(r)[0] and pair(r)[1] and pair(r)[0]["acted_on"]]
    olo, ohi = boot_ci(units, excess_other, seed + 11)
    ae = acted_excess(units)
    aelo, aehi = boot_ci(units, acted_excess, seed + 13)
    paired = [(pair(r)[0], pair(r)[1]) for u in units for r in u if pair(r)[0] and pair(r)[1]]
    resolved = [r for r in usable if r["resolved"]]
    lo, hi = boot_ci(units, excess, seed)
    alo, ahi = boot_ci(units, acted, seed + 7)
    return {
        "comments": len(rows), "usable": len(usable), "pull_requests": len(units),
        "resolved_rate": rate([r["resolved"] for r in usable]),
        "acted_on_rate": acted(units), "acted_on_ci": [alo, ahi],
        # Added 18 October 2026 after the study's first results (exploratory there, standard in the kit):
        "placebo_acted_on_rate": rate([b["acted_on"] for _, b in paired]),
        "acted_on_excess": ae, "acted_on_excess_ci": [aelo, aehi],
        "comments_per_caused_change": (1 / ae) if ae and ae > 0 else None,
        "comments_per_pr": (len(usable) / len(units)) if units else None,
        "resolved_by_bot_share": rate([r["resolved_by_bot"] for r in resolved]),
        "ignored_flags": len(ign),
        "ignored_fix_rate": rate([p[0][main] for p in ign]),
        "placebo_fix_rate": rate([p[1][main] for p in ign]),
        "excess": excess(units), "excess_ci": [lo, hi],
        "excess_other": excess_other(units), "excess_other_ci": [olo, ohi],
        "ignored_fix_rate_other": rate([p[0][other] for p in ign]),
        "placebo_fix_rate_other": rate([p[1][other] for p in ign]),
        "heeded_fix_rate": rate([h[main] for h in heeded]),
        "heeded_placebo_fix_rate": rate([h[main] for h in heeded_pl]),
        "churn_rate": rate([pair(r)[0]["churned"] for u in units for r in u if pair(r)[0]]),
    }


def cmd_report(a):
    c = load_config(a.config)
    prs = {p["number"]: p for p in read_jsonl(out(c, "prs.jsonl"))}
    rows = read_jsonl(out(c, "outcomes.jsonl"))
    ai_first = min((r["created"] for r in rows if r["kind"] == "ai"), default=None)
    dropped = 0
    if ai_first:
        keep = {n for n, p in prs.items() if p["created"][:10] >= ai_first[:10]}  # from the day of the first AI comment
        dropped = len(prs) - len(keep)
        rows = [r for r in rows if r["pr"] in keep]
    cls = {n: classify_pr(c, p) for n, p in prs.items()}
    pr_area = {n: area_of_pr(c, p) for n, p in prs.items()}

    def area(r):
        a_ = pr_area[r["pr"]]
        return a_ if a_ in ("frontend", "backend") else area_of_path(c, r["path"])

    cells = {"all": lambda r: True,
             "bugfix": lambda r: cls[r["pr"]][0] == "bugfix",
             "capability": lambda r: cls[r["pr"]][0] == "capability",
             "bugfix (labels only)": lambda r: cls[r["pr"]] == ("bugfix", "label"),
             "capability (labels only)": lambda r: cls[r["pr"]] == ("capability", "label")}
    if c["area_labels"] or c["path_rules"]:
        cells["frontend"] = lambda r: area(r) == "frontend"
        cells["backend"] = lambda r: area(r) == "backend"
    res = {"repo": c["repo"], "public_name": c.get("public_name", c["repo"]), "null": a.null,
           "first_ai_comment": ai_first, "prs_before_first_ai_comment_dropped": dropped,
           "classification": dict(Counter(f"{k}/{s}" for k, s in cls.values())),
           "cells": {}}
    seed = c["seed"]
    cov = fix_signal_coverage(c)
    signal = choose_fix_signal(c, cov)
    main, other = ("fixed", "fixed_loose") if signal != "titles" else ("fixed_loose", "fixed")
    res["fix_signal"] = signal
    res["fix_signal_coverage"] = cov
    for name, f in cells.items():
        res["cells"][name] = {}
        for kind in ("ai", "human"):
            sub = [r for r in rows if r["kind"] == kind and f(r)]
            st = cell_stats(sub, seed, a.null, main, other)
            st["too_thin"] = kind == "ai" and st["ignored_flags"] < THIN
            st["sample_size"] = sample_size_rule(st)
            res["cells"][name][kind] = st
    suffix = "-null" if a.null else ""
    out(c, f"summary{suffix}.json").write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    md = render(res)
    out(c, f"summary{suffix}.md").write_text(md, encoding="utf-8")
    print(md)


def fix_signal_coverage(c):
    """How the pull requests that later touched flagged code mark bug fixes: by label (or a closed issue's
    label) or by a conventional-commit title. Decides whether the labels can carry the fix signal."""
    info = read_jsonl(out(c, "pr-info.jsonl"))
    if not info:
        return None
    bug_pr, bug_issue = lower_set(c["bug_labels_pr"]), lower_set(c["bug_labels_issue"])
    fixp = [p.lower() for p in c["title_rules"].get("bugfix", [])]
    lab = sum(1 for i in info if lower_set(i["labels"]) & bug_pr or any(lower_set(x["labels"]) & bug_issue for x in i["closing_issues"]))
    tit = sum(1 for i in info if any(re.match(re.escape(p) + r"(\b|\(|:|!)", i["title"].lower()) for p in fixp))
    return {"later_prs": len(info), "bug_labelled": lab, "fix_titled": tit}


SIGNAL_SHARE = 0.10


def choose_fix_signal(c, cov):
    """The general rule, agreed 18 October 2026 before any real excess was read: a repository's bug labels
    carry the fix signal if at least 10% of the later pull requests that touched flagged code carry one;
    otherwise its fix-prefixed titles do, if at least 10% follow that convention; otherwise it has no usable
    fix signal and predictions 2 to 4 are not assessed for it. A config may force "labels" or "titles"."""
    forced = c.get("fix_signal", "auto")
    if forced != "auto":
        return forced
    if not cov or not cov["later_prs"]:
        return "none"
    if cov["bug_labelled"] / cov["later_prs"] >= SIGNAL_SHARE:
        return "labels"
    if cov["fix_titled"] / cov["later_prs"] >= SIGNAL_SHARE:
        return "titles"
    return "none"


def flags_needed(base, excess):
    """Ignored flags per reviewer kind to show an excess of `excess` above a placebo rate of `base` at 95%
    confidence with 80% power (two proportions, normal approximation). Added 19 October 2026."""
    if base is None or excess <= 0:
        return None
    p1 = min(0.999, base + excess)
    z = 1.96 + 0.84
    return int(math.ceil(z * z * (base * (1 - base) + p1 * (1 - p1)) / (excess * excess)))


def sample_size_rule(cell):
    """For one reviewer kind in one cell: what its volume can and cannot show."""
    base, have = cell.get("placebo_fix_rate"), cell.get("ignored_flags", 0)
    if base is None:
        return None
    need = {f"{int(d * 100)}pt": flags_needed(base, d) for d in (0.01, 0.02, 0.03, 0.05)}
    smallest = None
    for d in (x / 1000 for x in range(5, 501)):
        if flags_needed(base, d) <= have:
            smallest = round(d, 3)
            break
    return {"placebo_fix_rate": base, "ignored_flags": have, "needed": need, "smallest_detectable_excess": smallest}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(res):
    L = [f"# Review bot report: {res['public_name']}{' (NULL CHECK)' if res['null'] else ''}", "",
         f"First AI review comment: {res['first_ai_comment']}; pull requests before it dropped: "
         f"{res['prs_before_first_ai_comment_dropped']}.", "",
         "Pull request classification (kind/source): " + ", ".join(f"{k} {v}" for k, v in sorted(res["classification"].items())), "",
         f"Fix signal carrying the predictions: {res.get('fix_signal')}; later pull requests: {res.get('fix_signal_coverage')}.", "",
         "| cell | reviewer | comments | resolved | acted on | ignored flags | ignored fixed | placebo fixed | excess [95%] | excess, other fix signal [95%] | heeded fixed | churn |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, kinds in res["cells"].items():
        for kind, s in kinds.items():
            ci = s["excess_ci"]
            ex = pct(s["excess"]) + (f" [{pct(ci[0])}, {pct(ci[1])}]" if ci[0] is not None else "")
            if s.get("too_thin"):
                ex += " (too thin)"
            L.append(f"| {cell} | {kind} | {s['comments']} | {pct(s['resolved_rate'])} | {pct(s['acted_on_rate'])} | "
                     f"{s['ignored_flags']} | {pct(s['ignored_fix_rate'])} | {pct(s['placebo_fix_rate'])} | {ex} | "
                     f"{pct(s['excess_other'])} [{pct(s['excess_other_ci'][0])}, {pct(s['excess_other_ci'][1])}] | {pct(s['heeded_fix_rate'])} | {pct(s['churn_rate'])} |")
    if not res["null"]:
        L += ["", "Acted on before merge, corrected for churn: the placebo is matched lines nobody flagged.", "",
              "| cell | reviewer | comments per pull request | acted on | placebo acted on | acted-on excess [95%] | comments read per caused change | resolved threads resolved by a bot |",
              "|---|---|---|---|---|---|---|---|"]
        for cell, kinds in res["cells"].items():
            for kind, s in kinds.items():
                ci = s["acted_on_excess_ci"]
                cpc = "n/a" if s["comments_per_caused_change"] is None else f"{s['comments_per_caused_change']:.1f}"
                cpp = "n/a" if s["comments_per_pr"] is None else f"{s['comments_per_pr']:.1f}"
                L.append(f"| {cell} | {kind} | {cpp} | {pct(s['acted_on_rate'])} | {pct(s['placebo_acted_on_rate'])} | "
                         f"{pct(s['acted_on_excess'])} [{pct(ci[0])}, {pct(ci[1])}] | {cpc} | {pct(s['resolved_by_bot_share'])} |")
    if not res["null"]:
        L += ["", "How much history the post-merge question needs. From this repository's own placebo fix rate: ignored flags per reviewer kind to show an excess of a given size (95% confidence, 80% power), against what it has.", "",
              "| reviewer | placebo fix rate | ignored flags now | for 1 point | 2 points | 3 points | 5 points | smallest excess this volume could show |",
              "|---|---|---|---|---|---|---|---|"]
        for kind, s in res["cells"]["all"].items():
            r = s.get("sample_size")
            if not r:
                continue
            sm = "n/a" if r["smallest_detectable_excess"] is None else f"{100 * r['smallest_detectable_excess']:.1f} points"
            L.append(f"| {kind} | {pct(r['placebo_fix_rate'])} | {r['ignored_flags']:,} | {r['needed']['1pt']:,} | {r['needed']['2pt']:,} | "
                     f"{r['needed']['3pt']:,} | {r['needed']['5pt']:,} | {sm} |")
    if res["null"]:
        L += ["", "Null check passes when every excess interval contains zero."]
        ok = all((ci[0] is None or ci[0] <= 0 <= ci[1])
                 for k in res["cells"].values() for s in k.values() for ci in (s["excess_ci"], s["excess_other_ci"]))
        L.append(f"Null check: {'PASSED' if ok else 'FAILED'}")
    return "\n".join(L) + "\n"

# ----------------------------------------------------------------------------------------- verdict

def cmd_verdict(a):
    sums = []
    for p in a.config:
        c = load_config(p)
        sums.append((c, json.loads(out(c, "summary.json").read_text())))
    L = ["# Predictions", ""]
    # 1
    held = []
    for c, s in sums:
        ai, hu = s["cells"]["all"]["ai"], s["cells"]["all"]["human"]
        ok = (hu["acted_on_rate"] is not None and ai["acted_on_rate"] is not None
              and hu["acted_on_rate"] >= 2 * ai["acted_on_rate"]
              and hu["acted_on_ci"][0] is not None and ai["acted_on_ci"][1] is not None
              and hu["acted_on_ci"][0] > ai["acted_on_ci"][1])
        held.append(ok)
        L.append(f"- 1 {s['public_name']}: human acted on {pct(hu['acted_on_rate'])}, AI {pct(ai['acted_on_rate'])}: {'held' if ok else 'failed'}")
    L.append(f"Prediction 1 (every repository): {'HELD' if all(held) else 'FAILED'}")
    # 2
    n2 = 0
    nosig = {s["public_name"] for c, s in sums if s.get("fix_signal") == "none"}
    if nosig:
        L.append("No usable fix signal (predictions 2 to 4 not assessed): " + ", ".join(sorted(nosig)))
    for c, s in sums:
        if s["public_name"] in nosig:
            continue
        ci = s["cells"]["all"]["ai"]["excess_ci"]
        ok = ci[0] is not None and ci[0] <= 0 <= ci[1]
        n2 += ok
        L.append(f"- 2 {s['public_name']}: AI excess interval [{pct(ci[0])}, {pct(ci[1])}] includes zero: {ok}")
    m = len(sums) - len(nosig)
    need = 2 if m >= 3 else m  # "at least two of three"; with fewer assessable repositories, all of them
    L.append(f"Prediction 2 (at least {need} of {m} assessable): {'HELD' if m and n2 >= need else 'FAILED' if m else 'not assessable'}")
    # 3
    held = []
    for c, s in sums:
        if s["public_name"] in nosig:
            continue
        ci = s["cells"]["all"]["human"]["excess_ci"]
        ok = ci[0] is not None and ci[0] > 0
        held.append(ok)
        L.append(f"- 3 {s['public_name']}: human excess interval [{pct(ci[0])}, {pct(ci[1])}] clear of zero: {ok}")
    L.append(f"Prediction 3 (every repository): {'HELD' if all(held) else 'FAILED'}")
    # 4
    verdicts = []
    for c, s in sums:
        if s["public_name"] in nosig:
            continue
        cells = s["cells"]
        for a_, b_ in (("bugfix", "capability"), ("backend", "frontend")):
            if a_ not in cells:
                continue
            if b_ == "frontend" and not c.get("area_split_prediction", True):
                continue
            x, y = cells[a_]["ai"], cells[b_]["ai"]
            if x["too_thin"] or y["too_thin"]:
                L.append(f"- 4 {s['public_name']} {a_} vs {b_}: too thin ({x['ignored_flags']} and {y['ignored_flags']} ignored AI flags)")
                continue
            ok = x["excess"] > y["excess"]
            verdicts.append(ok)
            L.append(f"- 4 {s['public_name']} {a_} {pct(x['excess'])} vs {b_} {pct(y['excess'])}: {'held' if ok else 'failed'}")
    L.append("Prediction 4: " + ("too thin to tell" if not verdicts else "HELD" if all(verdicts) else "FAILED"))
    print("\n".join(L))

# ----------------------------------------------------------------------------------------- sample

def cmd_sample(a):
    c = load_config(a.config)
    signal = choose_fix_signal(c, fix_signal_coverage(c))
    key = "fixed_loose" if signal == "titles" else "fixed"
    all_rows = read_jsonl(out(c, "outcomes.jsonl"))
    rows = [r for r in all_rows if r.get(key)]
    for r in rows:
        r["fix_commits"] = r.get("fix_commits") or r.get("touched_by", [])
    rng = random.Random(c["seed"] + 30)
    pick = []
    for kind in ("ai", "human"):
        pool = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["id"])
        pick += rng.sample(pool, min(a.n // 2, len(pool)))
    base = f"https://github.com/{c['repo']}"
    L = [f"# Pre-check: {c.get('public_name', c['repo'])}", "",
         "For each comment: does the later fix answer what the comment raised? Write yes, no or unclear in the verdict line.", ""]
    for i, r in enumerate(pick, 1):
        body = r["body"].replace("\n", " ")[:600]
        fixes = ", ".join(f"[{s[:9]}]({base}/commit/{s})" for s in r["fix_commits"])
        L += [f"## {i}. {r['kind']} comment on `{r['path']}` lines {r['start']} to {r['end']}", "",
              f"Comment: {r['url']}", "", f"> {body}", "", f"Later fix: {fixes}", "", "Verdict: ", ""]
    # spec test 4: three pull requests whose comment counts are checked by eye against GitHub
    prs = sorted({r["pr"] for r in all_rows})
    pick_prs = random.Random(c["seed"] + 40).sample(prs, min(3, len(prs)))
    L += ["# Count check", "", "Open each pull request's Files changed tab and count the review threads that start on lines, "
          "excluding threads opened by the author. Compare with the tool's counts.", ""]
    for n in pick_prs:
        k = Counter(r["kind"] for r in all_rows if r["pr"] == n)
        L.append(f"- {base}/pull/{n}: AI {k['ai']}, human {k['human']}. Your count: ")
    L.append("")
    out(c, "precheck.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out(c, 'precheck.md')} with {len(pick)} comments")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--config", required=True)
    f.add_argument("--max-prs", type=int, default=1500); f.add_argument("--smoke", action="store_true")
    f.add_argument("--refresh", action="store_true")
    m = sub.add_parser("map"); m.add_argument("--config", required=True); m.add_argument("--clone", required=True)
    m.add_argument("--offline", action="store_true", help="never call GitHub (tests)")
    x = sub.add_parser("fixes"); x.add_argument("--config", required=True); x.add_argument("--clone", required=True)
    x.add_argument("--branch", help="default branch ref in the clone (default: origin/HEAD)")
    x.add_argument("--offline", action="store_true")
    r = sub.add_parser("report"); r.add_argument("--config", required=True); r.add_argument("--null", action="store_true")
    s = sub.add_parser("sample"); s.add_argument("--config", required=True); s.add_argument("--n", type=int, default=30)
    v = sub.add_parser("verdict"); v.add_argument("--config", required=True, nargs="+")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "map": cmd_map, "fixes": cmd_fixes, "report": cmd_report,
     "sample": cmd_sample, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    main()
