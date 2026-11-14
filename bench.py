#!/usr/bin/env python3
"""bench.py: turn your repository's fixed issues into a small private benchmark, then run an agent on it.

The method, one subcommand per step:

  mine     list merged pull requests that close exactly one issue and change a test file (needs GITHUB_TOKEN)
  split    split each fix into its test part (withheld from the agent) and its code part
  verify   in a clean container: the tests fail before the fix and pass after it, three times running
  freeze   write the surviving tasks into one manifest with a checksum; nothing changes a task after this
  packets  one file per task for the fairness screen: could a fix written from the issue alone pass the tests?
  run      hand each task's issue text to the agent harness in a fresh container and keep the diff it leaves
  score    apply the diff and the withheld tests; resolved when every new test passes and no old one breaks
  report   the funnel, flakiness, two runs side by side, cost, and the predictions; also report.json,
           every headline figure by name, so a write-up can cite numbers instead of retyping them;
           with --paired, two versions of the tickets read pair by pair over the same tasks; with --noise, the
           flips between identical runs, the split nets and the noise floor; with --study context, the context file
           study's report-context.md and .json
  all      the whole pipeline in one command, restartable, with a log of every step

Standard library only. The run step calls the harness's own Python environment (see bench/setup-env.sh).
Written up in the third post of the series "Ask your own repository" on thanoskarpouzis.com; `run --tickets`
and `report --paired` (with tickets.py) in the fourth; `run --harness`, `run --context`, `run --label`,
`report --noise` and `report --study context` (with context.py) in the fifth.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIN_BODY_CHARS = 40

# The clear ticket study's predictions (report --study tickets), fixed before any run. Change them here to read
# your own runs against your own thresholds.
TICKETS_TITLE_LOSS = 5            # 1. title only loses a net 5 or more tasks against the issue as filed
TICKETS_REWRITE_GAIN = 3          # 2. the rewritten ticket gains a net 3 or more
TICKETS_UNFAIR_GAIN_MAX = 1       # 3. ... and a net 1 at most on the tasks the fairness screen labelled unfair
TICKETS_FIRST_DRAFTS_REJECTED = 5  # 5. the leak check rejects at least 5 first drafts

# The context file study's rules (report --study context), fixed in its spec before any run. The floor is
# NOISE_SDS standard deviations of the net between two arms when nothing differs, from the mean flips F between
# two single no-file runs: sd = sqrt((F/2) * (1/a + 1/b)) for arms of a and b runs.
CONTEXT_PILOT_MIN = 10            # the pilot gate: n1 resolves at least 10 tasks or the study stops
CONTEXT_COST_RATIO = 1.15         # 3. the generated arm's median per-task cost ratio is at least this
CONTEXT_TEST_RUNNER_GAIN = 1      # 4. each context arm runs at least 1 more test-runner command per task (median)
CONTEXT_SINGLE_RUN_SPAN = 2       # 5. the eight single-run gaps span at least 2 tasks (or change sign)
CONTEXT_VALIDATION_NET = 3        # the pool's net agrees when within 3 of zero (inside) or beyond 3 (outside)
NOISE_SDS = 2
CONTEXT_POINTS = (2, 5, 10)       # what the bench can see: tasks and repeats needed for these differences
CONTEXT_NONE_RUNS = ("n1", "n2", "n3", "n4")
CONTEXT_ARMS = {"project-docs": ("d1", "d2"), "generated": ("g1", "g2")}
CONTEXT_POOL = {"project-docs": "vd1", "generated": "vg1"}
CONTEXT_POOL_BASE = ("r1", "r2")  # post 3's runs, the pool's no-file side
CONTEXT_POOL_LABEL = "fair"
TEST_RUNNERS = ("pytest", "py.test", "tox")  # as programs; `python -m pytest` counts too


def kit_dir(repo):
    return HERE / "kit" / "bench" / repo.replace("/", "__")


# ---------------------------------------------------------------------------- shared rules

def closing_refs(body, repo):
    """Issue numbers a pull request body says it closes. The keywords are the ones agent_lane.py reads
    (close, fix, resolve and their forms); besides `#n` this also accepts `owner/repo#n` and the issue's
    URL when they name the same repository, since many projects write them that way. Text inside HTML comments
    is ignored."""
    import agent_lane
    # Comments are the template's, not the author's: pydantic's pull request template carries an example
    # "fix #123" inside one. Removed first, as fetch_repo.clean_issue does for issue text.
    body = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    refs = set(agent_lane.CLOSING.findall(body))
    kw = r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+"
    owner_repo = re.escape(repo)
    refs |= set(re.findall(kw + owner_repo + r"#(\d+)\b", body, re.I))
    refs |= set(re.findall(kw + r"https?://github\.com/" + owner_repo + r"/issues/(\d+)\b", body, re.I))
    return sorted(refs, key=int)


DEFAULT_TEST_DIRS = ("tests", "test")


def is_test_path(path, test_dirs=DEFAULT_TEST_DIRS):
    """A file is a test file when a folder in its path has one of the repository's test-folder names (tests or
    test unless told otherwise; pytest itself uses testing), or its name starts with test_, ends with
    _test.py, or is conftest.py."""
    parts = path.split("/")
    name = parts[-1]
    return (any(p in test_dirs for p in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def is_bot(user):
    user = user or {}
    return user.get("type") == "Bot" or (user.get("login") or "").endswith("[bot]")


# ---------------------------------------------------------------------------- mine

def mine(args):
    import fetch_repo
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN set: GitHub allows 60 requests an hour without one, too few for this step.",
              file=sys.stderr)
    out = Path(args.out) if args.out else kit_dir(args.repo) / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    gh = fetch_repo.GitHub(token=token, cache_dir=kit_dir(args.repo) / "cache", refresh=args.refresh)
    drops, kept, seen = collections.Counter(), [], 0
    test_dirs = tuple(d.strip() for d in args.test_dirs.split(",") if d.strip())
    q = f"repo:{args.repo} is:pr is:merged merged:>={args.since}"
    for page in range(1, 11):  # GitHub search returns at most 1,000 results
        res = gh.get("/search/issues", {"q": q, "sort": "created", "order": "desc", "per_page": 100,
                                        "page": page})
        items = res.get("items") or []
        for item in items:
            if len(kept) >= args.target:
                break
            seen += 1
            row, reason = candidate(gh, args.repo, item["number"], test_dirs)
            if reason:
                drops[reason] += 1
            else:
                kept.append(row)
        if len(kept) >= args.target or len(items) < 100:
            break
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"repo": args.repo, "merged_since": args.since, "target": args.target, "test_dirs": test_dirs,
                "pull_requests_seen": seen, "kept": len(kept), "rejected": dict(drops.most_common()),
                "window_exhausted": len(kept) < args.target, "github": gh.snapshot(),
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_name("mine-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("pull_requests_seen", "kept", "rejected", "window_exhausted")},
                     indent=2))
    print(f"done: {len(kept)} candidates from {seen} pull requests, {gh.calls} calls to GitHub; wrote {out}")


def candidate(gh, repo, number, test_dirs=DEFAULT_TEST_DIRS):
    """(row, None) for a candidate, (None, reason) for a rejected pull request. Checks run cheapest first."""
    pr = gh.get(f"/repos/{repo}/pulls/{number}")
    if not pr.get("merged_at"):
        return None, "not merged"
    refs = closing_refs(pr.get("body"), repo)
    if not refs:
        return None, "no closing reference"
    if len(refs) > 1:
        return None, "several closing references"
    try:
        iss = gh.get(f"/repos/{repo}/issues/{refs[0]}")
    except urllib.error.HTTPError:
        return None, "issue not found"
    if "pull_request" in iss:
        return None, "reference is a pull request"
    if not iss["created_at"] < pr["created_at"]:
        return None, "issue not before pull request"
    if is_bot(iss.get("user")):
        return None, "issue filed by a bot"
    if len((iss.get("body") or "").strip()) < MIN_BODY_CHARS:
        return None, "issue body too short"
    files = [{"path": f["filename"], "status": f["status"], "changes": f["changes"]}
             for f in gh.pages(f"/repos/{repo}/pulls/{number}/files", {}, 3)]
    if not any(is_test_path(f["path"], test_dirs) for f in files):
        return None, "no test file changed"
    return {"task_id": f"{repo.replace('/', '__')}-{number}", "repo": repo, "pr": number,
            "pr_created_at": pr["created_at"], "merged_at": pr["merged_at"],
            "merge_commit_sha": pr["merge_commit_sha"], "base_ref": pr["base"]["ref"],
            "issue": iss["number"], "issue_created_at": iss["created_at"],
            "issue_title": iss["title"], "issue_body": iss.get("body") or "",
            "test_dirs": list(test_dirs), "files": files}, None


# ---------------------------------------------------------------------------- shared helpers

BENCH = HERE / "bench"


def load_recipe(path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    r["_path"] = Path(path).resolve()
    return r


def run(cmd, *, check=True, capture=True, inp=None, timeout=None):
    r = subprocess.run(cmd, text=True, capture_output=capture, input=inp, timeout=timeout)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{(r.stdout or '')[-2000:]}\n{(r.stderr or '')[-2000:]}")
    return r


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scratch():
    """Temporary task folders live under the kit (ignored by git), a path every Docker setup can mount."""
    d = HERE / "kit" / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_id(tag):
    return run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"]).stdout.strip()


def docker_limits(recipe):
    d = recipe.get("docker", {})
    return ["--cpus", str(d.get("cpus", "4")), "--memory", str(d.get("memory", "4g"))]


def task_meta(recipe, row, extra=None):
    meta = {k: recipe[k] for k in ("repo", "setup", "env", "pytest_args", "timeout_s", "run_globs", "run_exclude")
            if recipe.get(k) is not None}  # absent settings stay absent, so the runner's defaults apply
    meta.update({k: row[k] for k in ("task_id", "issue_title", "issue_body", "test_files", "test_files_at_base")
                 if k in row})
    meta.update(extra or {})
    return meta


def in_container(recipe, step, task_files, out_dir, meta, network="none"):
    """Run runner/inside.py <step> in a fresh container from the recipe's image; returns its JSON result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch()) as tmp:
        t = Path(tmp)
        for name, src in task_files.items():
            shutil.copyfile(src, t / name)
        (t / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        cmd = ["docker", "run", "--rm", "--network", network, *docker_limits(recipe),
               "-v", f"{t}:/task:ro", "-v", f"{BENCH / 'runner'}:/runner:ro", "-v", f"{out_dir.resolve()}:/out",
               recipe["image"], "python3", "/runner/inside.py", step]
        r = run(cmd, check=False, timeout=recipe.get("timeout_s", 600) * 3 + 300)
    res_file = out_dir / f"{step}.json"
    if not res_file.exists():
        return {"status": "error", "reason": "bench error", "detail": (r.stdout + r.stderr)[-1500:]}
    return json.loads(res_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- image

def image(args):
    """Lock the repository's test dependencies (once) and build its bench image on top of bench-harness."""
    recipe = load_recipe(args.recipe)
    rdir = recipe["_path"].parent
    lock = rdir / recipe["lock"]
    if not lock.exists():
        run([sys.executable, str(BENCH / "harness" / "lock.py"), "--requirements", str(rdir / recipe["requirements"]),
             "--constraints", str(rdir / recipe["constraints"]), "--out", str(lock)], capture=False)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(lock, Path(tmp) / "testbed.lock")
        (Path(tmp) / "Dockerfile").write_text(
            f"FROM {recipe['base_image']}\n"
            "COPY testbed.lock /opt/testbed.lock\n"
            "RUN python -m venv /opt/testbed \\\n"
            " && /opt/testbed/bin/pip install --no-cache-dir --require-hashes --no-deps -r /opt/testbed.lock\n",
            encoding="utf-8")
        run(["docker", "build", "--quiet", "-t", recipe["image"], tmp], capture=False)
    print(f"done: {recipe['image']} {image_id(recipe['image'])}")


# ---------------------------------------------------------------------------- split

def split(args):
    recipe = load_recipe(args.recipe)
    repo_dir, kd = Path(args.repo_dir).expanduser(), kit_dir(recipe["repo"])
    test_dirs = tuple(recipe["test_dirs"])
    rows, drops = [], collections.Counter()
    git = ["git", "-C", str(repo_dir)]
    cands = read_jsonl(kd / "candidates.jsonl")
    # One task per issue: a backport of the same fix to a maintenance branch is not a second task.
    default = recipe.get("default_branch", "main")
    on_default = {c["issue"] for c in cands if c.get("base_ref") == default}
    for c in cands:
        if c.get("base_ref") != default and c["issue"] in on_default:
            drops["backport of another candidate"] += 1
            continue
        tid, merge = c["task_id"], c["merge_commit_sha"]
        if run(git + ["cat-file", "-e", merge + "^{commit}"], check=False).returncode:
            drops["merge commit not in clone"] += 1
            continue
        parent = run(git + ["rev-parse", merge + "^1"]).stdout.strip()
        changed = run(git + ["diff", "--name-only", "--no-renames", parent, merge]).stdout.split()
        tests = sorted(f for f in changed if is_test_path(f, test_dirs))
        code = sorted(f for f in changed if not is_test_path(f, test_dirs))
        if not tests:
            drops["no test part"] += 1
            continue
        if not code:
            drops["no code part"] += 1
            continue
        at_base = set(run(git + ["ls-tree", "-r", "--name-only", parent, "--", *tests]).stdout.split())
        for name, paths in (("withheld", tests), ("fixes", code)):
            (kd / name).mkdir(parents=True, exist_ok=True)
            diff = run(git + ["diff", "--binary", "--no-renames", parent, merge, "--", *paths]).stdout
            (kd / name / f"{tid}.patch").write_text(diff, encoding="utf-8")
        (kd / "archives").mkdir(parents=True, exist_ok=True)
        run(git + ["archive", "--format=tar", "-o", str((kd / "archives" / f"{tid}.tar").resolve()), parent])
        rows.append({**c, "parent": parent, "test_files": tests, "code_files": code,
                     "test_files_at_base": sorted(at_base)})
    write_jsonl(kd / "split.jsonl", rows)
    (kd / "split-manifest.json").write_text(json.dumps({"kept": len(rows), "rejected": dict(drops)}, indent=2),
                                            encoding="utf-8")
    print(f"done: {len(rows)} split, rejected {dict(drops)}")


# ---------------------------------------------------------------------------- verify

def verify(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    rows = read_jsonl(kd / "split.jsonl")
    if args.only:
        rows = [r for r in rows if r["task_id"] in args.only.split(",")]
    for rep in range(1, args.repeats + 1):
        for r in rows:
            out = kd / "verify" / r["task_id"] / f"repeat{rep}"
            if (out / "verify.json").exists():
                continue
            if rep > 1:  # repeats only for candidates that passed the first time
                f1 = kd / "verify" / r["task_id"] / "repeat1" / "verify.json"
                if not f1.exists() or json.loads(f1.read_text(encoding="utf-8"))["status"] != "ok":
                    continue
            tid = r["task_id"]
            res = in_container(recipe, "verify", {"checkout.tar": kd / "archives" / f"{tid}.tar",
                                                  "test.patch": kd / "withheld" / f"{tid}.patch",
                                                  "code.patch": kd / "fixes" / f"{tid}.patch"},
                               out, task_meta(recipe, r))
            print(f"repeat {rep} {tid}: {res['status']} {res.get('reason', '')} ({res.get('seconds')}s)", flush=True)
            if res["status"] == "error":
                print("  " + (res.get("detail") or "")[:600].replace("\n", "\n  "), flush=True)
    summary = verify_summary(kd, rows, args.repeats)
    (kd / "verify-manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("candidates", "survivors", "rejected", "flaky", "bench_errors")},
                     indent=2))


def verify_summary(kd, rows, repeats):
    rejected, survivors, flaky, errors = collections.Counter(), [], [], []
    for r in rows:
        res = []
        for rep in range(1, repeats + 1):
            f = kd / "verify" / r["task_id"] / f"repeat{rep}" / "verify.json"
            if f.exists():
                res.append(json.loads(f.read_text(encoding="utf-8")))
        if not res:
            continue
        if res[0]["status"] == "error":
            errors.append(r["task_id"])
            continue
        if res[0]["status"] != "ok":
            rejected[res[0]["reason"]] += 1
            continue
        if len(res) < repeats:
            continue  # not finished
        sig = [(x["status"], tuple(x.get("fail_to_pass", [])), tuple(x.get("pass_to_pass", []))) for x in res]
        if len(set(sig)) > 1:
            flaky.append(r["task_id"])
            rejected["flaky"] += 1
            continue
        survivors.append(r["task_id"])
    return {"candidates": len(rows), "survivors": len(survivors), "survivor_ids": survivors,
            "rejected": dict(rejected), "flaky": flaky, "bench_errors": errors, "repeats": repeats}


# ---------------------------------------------------------------------------- freeze

def canonical_checksum(tasks):
    return hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def freeze(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    summary = json.loads((kd / "verify-manifest.json").read_text(encoding="utf-8"))
    rows = {r["task_id"]: r for r in read_jsonl(kd / "split.jsonl")}
    img = image_id(recipe["image"])
    tasks = []
    for tid in summary["survivor_ids"]:
        r = rows[tid]
        v = json.loads((kd / "verify" / tid / "repeat1" / "verify.json").read_text(encoding="utf-8"))
        tasks.append({"task_id": tid, "repo": r["repo"], "pr": r["pr"], "issue": r["issue"],
                      "merged_at": r["merged_at"], "parent": r["parent"], "issue_title": r["issue_title"],
                      "issue_body": r["issue_body"], "test_files": r["test_files"],
                      "test_files_at_base": r["test_files_at_base"], "run_files": v["run_files"],
                      "fail_to_pass": v["fail_to_pass"], "pass_to_pass": v["pass_to_pass"],
                      "checkout_sha256": sha256_file(kd / "archives" / f"{tid}.tar"),
                      "withheld_sha256": sha256_file(kd / "withheld" / f"{tid}.patch")})
    tasks.sort(key=lambda t: t["task_id"])
    manifest = {"repo": recipe["repo"], "image": recipe["image"], "image_id": img,
                "recipe_sha256": sha256_file(recipe["_path"]), "frozen": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tasks": tasks, "checksum": canonical_checksum(tasks)}
    out = kd / "tasks.json"
    if out.exists() and not args.force:
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["checksum"] != manifest["checksum"]:
            sys.exit("tasks.json exists with a different checksum; the task set is frozen (use --force only "
                     "before any run, and record why in the changelog)")
        print(f"unchanged: {len(tasks)} tasks, checksum {old['checksum']}")
        return
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(tasks)} tasks frozen, image {img}, checksum {manifest['checksum']}")


# ---------------------------------------------------------------------------- run

def spent(run_dir):
    total = 0.0
    for f in Path(run_dir).glob("*/result.json"):
        total += json.loads(f.read_text(encoding="utf-8")).get("cost") or 0.0
    return total


def redact_key(folder, key):
    """Second guard: no file a run leaves behind may contain the key. Returns the files where it was found."""
    found = []
    for f in Path(folder).rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            if key.encode() in data:
                f.write_bytes(data.replace(key.encode(), b"[redacted]"))
                found.append(f.name)
    return found


def load_tickets(kd, version, manifest):
    """A frozen version of the tickets (tickets.py), checked against the task set: made from this manifest,
    matching its own checksum, one ticket for each task and no other. Raises ValueError otherwise."""
    path = Path(kd) / "tickets" / f"{version}.json"
    if not path.exists():
        raise ValueError(f"no version file {path}; write it with tickets.py first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != version:
        raise ValueError(f"{path} says it is version {doc.get('version')!r}")
    if doc.get("tasks_checksum") != manifest["checksum"]:
        raise ValueError(f"{path} was made from a different task set")
    if canonical_checksum(doc["tickets"]) != doc.get("checksum"):
        raise ValueError(f"{path} does not match its checksum")
    want, have = {t["task_id"] for t in manifest["tasks"]}, set(doc["tickets"])
    if want != have:
        raise ValueError(f"{path} does not hold one ticket per task: missing {sorted(want - have)[:5]}, "
                         f"unknown {sorted(have - want)[:5]}")
    for tid, tk in doc["tickets"].items():
        if set(tk) != {"issue_title", "issue_body"}:
            raise ValueError(f"{path}: ticket {tid} must hold issue_title and issue_body only")
    return doc


def tickets_setting(doc):
    if doc is None:
        return {"version": "as-filed"}
    return {"version": doc["version"], "checksum": doc["checksum"], "rule_sha256": doc.get("rule_sha256")}


def load_context(kd, name, manifest):
    """A frozen context file (context.py freeze), checked against the task set: the file matches the checksum in
    its freeze record, the record was made from this manifest, and its leak check passed. Raises ValueError
    otherwise. Returns the record with the file's text under `text`."""
    path, text_path = Path(kd) / "context" / f"{name}.json", Path(kd) / "context" / f"{name}.md"
    if not path.exists():
        raise ValueError(f"no frozen context file {path}; freeze it with context.py first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("name") != name:
        raise ValueError(f"{path} says it is {doc.get('name')!r}")
    if doc.get("tasks_checksum") != manifest["checksum"]:
        raise ValueError(f"{path} was made from a different task set")
    if not text_path.exists() or sha256_file(text_path) != doc.get("sha256"):
        raise ValueError(f"{text_path} does not match the checksum it was frozen with")
    if not (doc.get("check") or {}).get("leak_passed"):
        raise ValueError(f"{path} did not pass the leak check")
    return {**doc, "text": text_path.read_text(encoding="utf-8")}


def context_setting(doc):
    if doc is None:
        return {"name": "none"}
    return {"name": doc["name"], "sha256": doc["sha256"]}


def load_labels(path):
    return {j["task_id"]: j["label"] for j in read_jsonl(path)}


def agent_container(recipe, cfg, name, checkout, meta, tdir, key, files=None):
    """One harness session in a fresh task container: the checkout, meta.json, the harness settings and any extra
    files in /task, the key on standard input, then everything the runner wrote to /out copied into tdir. Shared
    by the run step and by context.py generate. Returns result.json, written here if the runner left none."""
    run(["docker", "rm", "-f", name], check=False)
    with tempfile.TemporaryDirectory(dir=scratch()) as tmp:
        tp = Path(tmp)
        shutil.copyfile(checkout, tp / "checkout.tar")
        (tp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for fname, text in (files or {}).items():
            (tp / fname).write_text(text, encoding="utf-8")
        hcfg = tp / "harness.json"
        hcfg.write_text(json.dumps(cfg), encoding="utf-8")
        run(["docker", "create", "-i", "--name", name, "--network", cfg["environment"]["network"],
             *docker_limits(recipe), "-v", f"{tp}:/task:ro", "-v", f"{BENCH / 'runner'}:/runner:ro",
             "-v", f"{hcfg}:/bench/harness.json:ro", recipe["image"],
             "/opt/harness/bin/python", "/runner/run_task.py"])
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        try:
            r = run(["docker", "start", "-ai", name], check=False, inp=json.dumps({"api_key": key}) + "\n",
                    timeout=cfg["wall_time_limit_s"])
            (tdir / "container.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
        except subprocess.TimeoutExpired:
            run(["docker", "kill", name], check=False)
            (tdir / "container.log").write_text("killed: wall time limit\n", encoding="utf-8")
        run(["docker", "cp", f"{name}:/out/.", str(tdir)], check=False)
        run(["docker", "rm", "-f", name], check=False)
    leaked = redact_key(tdir, key)
    if leaked:
        print(f"  WARNING: the key was found in {leaked} and has been redacted there; tell Claude", flush=True)
    egress = run(["docker", "logs", "--since", since + "Z", "bench-egress"], check=False)
    (tdir / "egress.jsonl").write_text((egress.stdout or "") + (egress.stderr or ""), encoding="utf-8")
    if not (tdir / "result.json").exists():
        (tdir / "result.json").write_text(json.dumps({"task_id": meta["task_id"], "exit_status": "no result",
                                                      "cost": None}), encoding="utf-8")
    return json.loads((tdir / "result.json").read_text(encoding="utf-8"))


def load_harness(path=None):
    """The harness settings: bench/harness.json unless another file is named (bench/harness.haiku.json)."""
    return json.loads(Path(path or BENCH / "harness.json").read_text(encoding="utf-8"))


def run_agent(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    manifest = json.loads((kd / "tasks.json").read_text(encoding="utf-8"))
    if canonical_checksum(manifest["tasks"]) != manifest["checksum"]:
        sys.exit("tasks.json does not match its checksum")
    doc = None
    if getattr(args, "tickets", None):
        try:
            doc = load_tickets(kd, args.tickets, manifest)
        except ValueError as e:
            sys.exit(str(e))
    ctx = None
    if getattr(args, "context", None):
        try:
            ctx = load_context(kd, args.context, manifest)
        except ValueError as e:
            sys.exit(str(e))
    label = getattr(args, "label", None)
    labels = None
    if label:
        lpath = kd / "fairness" / "labels.jsonl"
        labels = load_labels(lpath)
        if not any(v == label for v in labels.values()):
            sys.exit(f"no task carries the label {label!r} in {lpath}")
    if image_id(recipe["image"]) != manifest["image_id"]:
        sys.exit("the bench image is not the one the tasks were frozen with")
    cfg = load_harness(getattr(args, "harness", None))
    key = os.environ.get(cfg["model"]["api_key_env"])
    if not key:
        sys.exit(f"set {cfg['model']['api_key_env']} in this shell first")
    if args.cost_cap is not None:
        cfg["agent"]["cost_limit"] = args.cost_cap
    run_dir = kd / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tickets = tickets_setting(doc)
    context = context_setting(ctx)
    old = run_dir / "settings.json"
    if old.exists() and list(run_dir.glob("*/result.json")):
        prev = json.loads(old.read_text(encoding="utf-8"))
        was = prev.get("tickets", {"version": "as-filed"})
        if was != tickets:
            sys.exit(f"run {args.run_id} was started with tickets {was}; it cannot resume with {tickets}")
        was = prev.get("context", {"name": "none"})
        if was != context:
            sys.exit(f"run {args.run_id} was started with context file {was}; it cannot resume with {context}")
        was = prev.get("harness", {}).get("model", {}).get("model_name")
        if was != cfg["model"]["model_name"]:
            sys.exit(f"run {args.run_id} was started with the model {was}; it cannot resume with "
                     f"{cfg['model']['model_name']}")
        if prev.get("label") != label:
            sys.exit(f"run {args.run_id} was started with the label {prev.get('label')}; it cannot resume with {label}")
    settings = {"harness": cfg, "image_id": manifest["image_id"], "tasks_checksum": manifest["checksum"],
                "tickets": tickets, "context": context}
    if label:
        settings["label"] = label
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    tasks = manifest["tasks"]
    if doc is not None:  # the only thing a version changes: the title and body the agent is handed
        tasks = [{**t, **doc["tickets"][t["task_id"]]} for t in tasks]
    if labels is not None:
        tasks = [t for t in tasks if labels.get(t["task_id"]) == label]
    if args.only:
        tasks = [t for t in tasks if t["task_id"] in args.only.split(",")]
    for t in tasks:
        tdir = run_dir / t["task_id"]
        if (tdir / "result.json").exists():
            continue
        if spent(run_dir) + cfg["agent"]["cost_limit"] > args.spend_cap:
            print(f"stopping: ${spent(run_dir):.2f} spent, the next task could pass the ${args.spend_cap} cap")
            break
        tdir.mkdir(parents=True, exist_ok=True)
        name = f"bench-{args.run_id}-{t['task_id']}".replace("_", "-")[:120]
        meta, files = task_meta(recipe, t), None
        if ctx is not None:  # the file goes to the runner beside the task, never into the working tree
            meta["context"] = {"name": ctx["name"], "sha256": ctx["sha256"], "path": "/task/context.md"}
            files = {"context.md": ctx["text"]}
        res = agent_container(recipe, cfg, name, kd / "archives" / f"{t['task_id']}.tar", meta, tdir, key, files)
        print(f"{t['task_id']}: {res.get('exit_status')} ${res.get('cost') or 0:.3f} "
              f"{res.get('api_calls')} calls {res.get('wall_s')}s (run total ${spent(run_dir):.2f})", flush=True)
    print(f"done: run {args.run_id}, {len(list(run_dir.glob('*/result.json')))} tasks, ${spent(run_dir):.2f} spent")

# ---------------------------------------------------------------------------- score

def score(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    manifest = json.loads((kd / "tasks.json").read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir) if args.run_dir else kd / "runs" / args.run_id
    rows = []
    for t in manifest["tasks"]:
        tdir = run_dir / t["task_id"]
        if not (tdir / "result.json").exists():
            continue
        out = tdir / ("score" if not args.out_name else args.out_name)
        if not (out / "score.json").exists():
            sub = tdir / "submission.diff"
            if not sub.exists():
                sub.write_text("", encoding="utf-8")
            in_container(recipe, "score", {"checkout.tar": kd / "archives" / f"{t['task_id']}.tar",
                                           "test.patch": kd / "withheld" / f"{t['task_id']}.patch",
                                           "submission.diff": sub}, out, task_meta(recipe, t, {
                                               "fail_to_pass": t["fail_to_pass"], "pass_to_pass": t["pass_to_pass"]}))
        sc = json.loads((out / "score.json").read_text(encoding="utf-8"))
        if sc["status"] == "error":
            print(f"{t['task_id']}: bench error while scoring\n  {(sc.get('detail') or '')[:600]}", flush=True)
        res = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
        rows.append({"task_id": t["task_id"], "resolved": bool(sc.get("resolved")), "score_status": sc["status"],
                     "reason": sc.get("reason"), "cost": res.get("cost"), "api_calls": res.get("api_calls"),
                     "tokens": res.get("tokens"), "wall_s": res.get("wall_s"), "capped": res.get("capped"),
                     "exit_status": res.get("exit_status")})
    write_jsonl(run_dir / "score.jsonl", rows)
    n = sum(r["resolved"] for r in rows)
    print(f"done: {n} of {len(rows)} resolved in {run_dir.name}")


# ---------------------------------------------------------------------------- packets (the fairness screen)

def packets(args):
    """One file per frozen task with what a labeller needs to judge whether the tests are fair to the issue:
    the issue as filed, the fail-to-pass test names and the withheld test diff. Never the fix, never a result.
    The rule to apply is in bench/fairness_prompt.md; labels go in a JSON-lines file for `report --labels`."""
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    out = kd / "fairness" / "packets"
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((kd / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    for t in tasks:
        patch = (kd / "withheld" / f"{t['task_id']}.patch").read_text(encoding="utf-8", errors="replace")
        f2p = t["fail_to_pass"]
        names = "\n".join("- " + x for x in f2p[:40])
        if len(f2p) > 40:
            names += f"\n- ... and {len(f2p) - 40} more in the same files"
        cut = "\n(test diff cut at 30,000 characters)\n" if len(patch) > 30000 else ""
        (out / f"{t['task_id']}.md").write_text(
            f"# Task {t['task_id']}\n\n## Issue as filed (the only thing the agent was given, besides the repository)"
            f"\n\nTitle: {t['issue_title']}\n\n{t['issue_body']}\n\n## Fail-to-pass tests ({len(f2p)} in all; these "
            f"must pass for the task to count as resolved)\n\n{names}\n\n## The withheld test changes from the real fix "
            f"(never shown to the agent)\n\n```diff\n{patch[:30000]}\n```\n{cut}", encoding="utf-8")
    print(f"done: {len(tasks)} packets in {out}; label them with bench/fairness_prompt.md")


# ---------------------------------------------------------------------------- report

def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


TEST_CHECK = {"no test part", "no runnable test file", "tests did not fail before", "tests did not pass after"}
ENVIRONMENT = {"environment did not build", "timed out", "test part did not apply", "code part did not apply",
               "test run produced no outcomes", "merge commit not in clone"}


def funnel_steps(mine_m, split_m, verify_m):
    steps = collections.Counter()
    for reason, n in {**split_m["rejected"], **verify_m["rejected"]}.items():
        if reason in TEST_CHECK:
            steps["fail-then-pass test check"] += n
        elif reason in ENVIRONMENT:
            steps["environment"] += n
        elif reason == "flaky":
            steps["flakiness"] += n
        elif reason == "backport of another candidate":
            continue  # part of what counts as a task, like the mining rules, not a check a task can fail
        else:
            steps["fix shape (" + reason + ")"] += n
    return steps


def predictions(mine_m, split_m, verify_m, runs):
    out = []
    cands = mine_m["kept"] - split_m["rejected"].get("backport of another candidate", 0)
    surv = verify_m["survivors"]
    out.append(("1. Fewer than half of the candidates survive", surv < cands / 2, f"{surv} of {cands}"))
    steps = funnel_steps(mine_m, split_m, verify_m)
    top = steps.most_common()
    test_n = steps.get("fail-then-pass test check", 0)
    held2 = bool(top) and all(test_n > n for name, n in top if name != "fail-then-pass test check")
    out.append(("2. The fail-then-pass test check removes the most", held2, dict(steps)))
    passed_once = surv + len(verify_m["flaky"])
    fl = len(verify_m["flaky"])
    out.append(("3. Flakiness: at least one, at most one in ten", 1 <= fl <= passed_once / 10,
                f"{fl} flaky of {passed_once} that passed once"))
    if len(runs) >= 2:
        a = {r["task_id"] for r in runs[0] if r["resolved"]}
        b = {r["task_id"] for r in runs[1] if r["resolved"]}
        n = len({r["task_id"] for r in runs[0]} & {r["task_id"] for r in runs[1]})
        diff = len(a ^ b)
        out.append(("4. At least 15% of tasks resolved in exactly one of two runs", n and diff >= 0.15 * n,
                    f"{diff} of {n} ({(diff / n * 100 if n else 0):.0f}%); run 1 {len(a)}, run 2 {len(b)}, both {len(a & b)}"))
        ok5, txt = [], []
        for i, rs in enumerate(runs[:2], 1):
            costs = [r["cost"] for r in rs if r["resolved"] and r["cost"] is not None]
            med = median(costs)
            ratio = max(costs) / med if costs and med else None
            ok5.append(ratio is not None and ratio >= 3)
            txt.append(f"run {i}: max/median {ratio:.1f}" if ratio else f"run {i}: no resolved tasks")
        out.append(("5. Costliest resolved task at least 3x the median (both runs)", all(ok5), "; ".join(txt)))
    return out


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k of n, as (low, high) fractions; None when n is 0."""
    if not n:
        return None
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x):
    return None if x is None else round(100 * x)


def run_figures(rs):
    """Headline figures for one scored run."""
    n = len(rs)
    resolved = sum(bool(r["resolved"]) for r in rs)
    costs = [r["cost"] for r in rs if r["cost"] is not None]
    res_costs = [r["cost"] for r in rs if r["resolved"] and r["cost"] is not None]
    calls = [r["api_calls"] for r in rs if r.get("api_calls") is not None]
    walls = [r["wall_s"] for r in rs if r.get("wall_s") is not None]
    tok_in = sum((r.get("tokens") or {}).get("input", 0) for r in rs)
    tok_out = sum((r.get("tokens") or {}).get("output", 0) for r in rs)
    tok_cache = sum((r.get("tokens") or {}).get("cache_read", 0) for r in rs)
    med_res = median(res_costs)
    iv = wilson(resolved, n)
    return {"tasks": n, "resolved": resolved, "resolved_pct": pct(resolved / n) if n else None,
            "resolved_interval_pct": [pct(iv[0]), pct(iv[1])] if iv else None,
            "capped": sum(bool(r.get("capped")) for r in rs),
            "submitted": sum(1 for r in rs if str(r.get("exit_status") or "").lower() == "submitted"),
            "unresolved_submitted": sum(1 for r in rs if not r["resolved"] and str(r.get("exit_status") or "").lower() == "submitted"),
            "cost_total": round(sum(costs), 2), "cost_median": round(median(costs) or 0, 3),
            "cost_max": round(max(costs), 3) if costs else None,
            "resolved_cost_median": round(med_res or 0, 3), "resolved_cost_max": round(max(res_costs), 3) if res_costs else None,
            "resolved_cost_max_over_median": round(max(res_costs) / med_res, 1) if res_costs and med_res else None,
            "calls_median": median(calls), "wall_s_median": round(median(walls)) if walls else None,
            "wall_s_max": round(max(walls)) if walls else None, "wall_s_total": round(sum(walls)) if walls else None,
            "tokens_input": tok_in, "tokens_output": tok_out, "tokens_cache_read": tok_cache}


def report_figures(repo, mine_m, split_m, verify_m, run_ids, runs, labels=None):
    """Every number a write-up would cite, by name. Pure: no files, so the tests can feed it manifests."""
    backports = split_m["rejected"].get("backport of another candidate", 0)
    distinct = mine_m["kept"] - backports
    fig = {"repo": repo, "merged_since": mine_m["merged_since"], "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "funnel": {"pull_requests_seen": mine_m["pull_requests_seen"], "candidates": mine_m["kept"],
                      "mining_rejected": mine_m["rejected"], "backports_folded": backports,
                      "distinct_issues": distinct, "after_split": split_m["kept"], "split_rejected": split_m["rejected"],
                      "verify_repeats": verify_m["repeats"], "survivors": verify_m["survivors"],
                      "verify_rejected": verify_m["rejected"], "flaky": len(verify_m["flaky"]),
                      "bench_errors": verify_m["bench_errors"],
                      "candidates_per_pull_request_pct": pct(mine_m["kept"] / mine_m["pull_requests_seen"]) if mine_m["pull_requests_seen"] else None,
                      "survivors_per_distinct_issue_pct": pct(verify_m["survivors"] / distinct) if distinct else None,
                      "by_step": dict(funnel_steps(mine_m, split_m, verify_m))},
           "runs": {rid: run_figures(rs) for rid, rs in zip(run_ids, runs)}}
    if len(runs) >= 2:
        a = {r["task_id"]: bool(r["resolved"]) for r in runs[0]}
        b = {r["task_id"]: bool(r["resolved"]) for r in runs[1]}
        common = sorted(set(a) & set(b))
        both = [t for t in common if a[t] and b[t]]
        one = [t for t in common if a[t] != b[t]]
        ratios = []
        for t in common:
            ca = next((r["cost"] for r in runs[0] if r["task_id"] == t), None)
            cb = next((r["cost"] for r in runs[1] if r["task_id"] == t), None)
            if ca and cb:
                ratios.append(max(ca, cb) / min(ca, cb))
        fig["agreement"] = {"runs": run_ids[:2], "tasks": len(common), "resolved_in_both": len(both),
                            "resolved_in_exactly_one": len(one), "exactly_one_pct": pct(len(one) / len(common)) if common else None,
                            "agree": len(common) - len(one), "differing_tasks": one,
                            "cost_ratio_median": round(median(ratios), 2) if ratios else None,
                            "cost_ratio_max": round(max(ratios), 1) if ratios else None}
    if labels:
        fig["fairness"] = {}
        for label in ("fair", "unfair", "unclear"):
            ids = {t for t, v in labels.items() if v == label}
            if not ids:
                continue
            fig["fairness"][label] = {"tasks": len(ids), "runs": {}}
            for rid, rs in zip(run_ids, runs):
                sub = [r for r in rs if r["task_id"] in ids]
                k = sum(bool(r["resolved"]) for r in sub)
                iv = wilson(k, len(sub))
                fig["fairness"][label]["runs"][rid] = {"resolved": k, "of": len(sub), "pct": pct(k / len(sub)) if sub else None,
                                                       "interval_pct": [pct(iv[0]), pct(iv[1])] if iv else None}
    fig["predictions"] = [{"name": name, "verdict": "held" if held else "failed", "detail": detail if isinstance(detail, str) else json.dumps(detail)}
                          for name, held, detail in predictions(mine_m, split_m, verify_m, runs)]
    if runs:
        fig["per_task"] = [{"task_id": r["task_id"], **{rid: bool(next((x["resolved"] for x in rs if x["task_id"] == r["task_id"]), False)) for rid, rs in zip(run_ids, runs)}}
                           for r in sorted(runs[0], key=lambda r: r["task_id"])]
    return fig


def paired_figures(versions, groups, labels=None):
    """Two versions of the tickets read pair by pair over the same tasks. groups[i] is a list of (run id, scored
    rows) for version i, two runs each. Per task a version scores 1 when resolved in every run, a half when in one
    of two, 0 in none; `only_b` is the sum of how much B beats A per task, `only_a` the reverse, `both` and
    `neither` the overlap, so the four add up to the task count and net = only_b - only_a. Pure."""
    def score_of(group):
        per = collections.defaultdict(list)
        for _rid, rs in group:
            for r in rs:
                per[r["task_id"]].append(bool(r["resolved"]))
        return {t: (sum(v), len(v)) for t, v in per.items()}

    sa, sb = score_of(groups[0]), score_of(groups[1])
    common = sorted(set(sa) & set(sb))

    def split(ids):
        c = {"tasks": len(ids), "both": 0.0, "neither": 0.0, "only_a": 0.0, "only_b": 0.0}
        for t in ids:
            a, b = sa[t][0] / sa[t][1], sb[t][0] / sb[t][1]
            c["both"] += min(a, b)
            c["neither"] += min(1 - a, 1 - b)
            c["only_a"] += max(a - b, 0)
            c["only_b"] += max(b - a, 0)
        c = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
        c["net_b_minus_a"] = round(c["only_b"] - c["only_a"], 2)
        return c

    out = {"a": versions[0], "b": versions[1], "runs_a": [rid for rid, _ in groups[0]],
           "runs_b": [rid for rid, _ in groups[1]], **split(common)}
    if labels:
        for label in ("fair", "unfair", "unclear"):
            ids = [t for t in common if labels.get(t) == label]
            if ids:
                out[label] = split(ids)
    out["runs"] = {}
    for rid, rs in groups[0] + groups[1]:
        f = run_figures(rs)
        out["runs"][rid] = {"resolved": f["resolved"], "tasks": f["tasks"], "cost_total": f["cost_total"],
                            "cost_per_resolved": round(f["cost_total"] / f["resolved"], 3) if f["resolved"] else None,
                            "resolved_interval_pct": f["resolved_interval_pct"], "capped": f["capped"],
                            "cost_median": f["cost_median"], "calls_median": f["calls_median"],
                            "tokens_input": f["tokens_input"], "tokens_output": f["tokens_output"]}
    out["per_task"] = [{"task_id": t, "a": sa[t][0], "b": sb[t][0], "of": sa[t][1], "label": (labels or {}).get(t)}
                       for t in common]
    return out


def ticket_predictions(paired, rewrite_counts=None):
    """The clear ticket study's five predictions, from paired blocks whose A side is the issue as filed."""
    by = {p["b"]: p for p in paired if p["a"] == "as-filed"}
    out = []
    t, w = by.get("title-only"), by.get("rewritten")
    if t:
        out.append((f"1. Title only loses a net {TICKETS_TITLE_LOSS} or more", t["net_b_minus_a"] <= -TICKETS_TITLE_LOSS,
                    f"net {t['net_b_minus_a']:+g} of {t['tasks']}"))
    if w:
        out.append((f"2. Rewritten gains a net {TICKETS_REWRITE_GAIN} or more", w["net_b_minus_a"] >= TICKETS_REWRITE_GAIN,
                    f"net {w['net_b_minus_a']:+g} of {w['tasks']}"))
        if "unfair" in w:
            u = w["unfair"]
            out.append((f"3. On unfair tasks the rewritten gain is a net {TICKETS_UNFAIR_GAIN_MAX} at most",
                        u["net_b_minus_a"] <= TICKETS_UNFAIR_GAIN_MAX, f"net {u['net_b_minus_a']:+g} of {u['tasks']} unfair"
                        + (f"; fair net {w['fair']['net_b_minus_a']:+g} of {w['fair']['tasks']}" if "fair" in w else "")))
    if t or w:
        p = t or w
        inf = float("inf")  # a run that resolves nothing costs without limit per resolved task
        def per(pair, side):
            return [pair["runs"][r]["cost_per_resolved"] for r in pair["runs_" + side]]
        base = per(p, "a")
        b_inf = [inf if c is None else c for c in base]
        ok, txt = True, [f"as filed {base}"]
        if w:
            cw = per(w, "b")
            ok = ok and max(inf if c is None else c for c in cw) < min(b_inf)
            txt.append(f"rewritten {cw}")
        if t:
            ct = per(t, "b")
            ok = ok and min(inf if c is None else c for c in ct) > max(b_inf)
            txt.append(f"title only {ct}")
        out.append(("4. Cost per resolved task: rewritten lower than as filed, title only higher, in every run",
                    ok, "; ".join(txt)))
    if rewrite_counts:
        n = rewrite_counts["first_draft_rejected"]
        out.append((f"5. The leak check rejects at least {TICKETS_FIRST_DRAFTS_REJECTED} first drafts",
                    n >= TICKETS_FIRST_DRAFTS_REJECTED,
                    f"{n} first drafts rejected, {rewrite_counts['second_draft_rejected']} second, "
                    f"{rewrite_counts['fallbacks']} fell back"))
    return out


def run_version(run_dir):
    f = Path(run_dir) / "settings.json"
    if not f.exists():
        return "as-filed"
    return json.loads(f.read_text(encoding="utf-8")).get("tickets", {}).get("version", "as-filed")


def report(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    if getattr(args, "study", None) == "context":
        if not args.labels:
            sys.exit("report --study context needs --labels (the fairness labels: the fair and unfair split and the pool)")
        context_report(args, recipe, kd, load_labels(args.labels))
        return
    mine_m = json.loads((kd / "mine-manifest.json").read_text(encoding="utf-8"))
    split_m = json.loads((kd / "split-manifest.json").read_text(encoding="utf-8"))
    verify_m = json.loads((kd / "verify-manifest.json").read_text(encoding="utf-8"))
    paired_groups = [[g.split(",") for g in pair] for pair in (getattr(args, "paired", None) or [])]
    if paired_groups and not args.runs:
        args.runs = list(dict.fromkeys(rid for pair in paired_groups for g in pair for rid in g))
    runs = [read_jsonl(kd / "runs" / rid / "score.jsonl") for rid in args.runs]
    L = [f"# Bench report: {recipe['repo']}", "", "## Funnel", "",
         f"Pull requests merged since {mine_m['merged_since']}, seen: {mine_m['pull_requests_seen']}",
         f"Candidates after mining: {mine_m['kept']} (rejected: {mine_m['rejected']})",
         f"After splitting the fix: {split_m['kept']} (rejected: {split_m['rejected']})",
         f"After verification, {verify_m['repeats']} repeats: {verify_m['survivors']} (rejected: {verify_m['rejected']})",
         f"Flaky: {len(verify_m['flaky'])} {verify_m['flaky']}", f"Bench errors: {verify_m['bench_errors']}",
         f"By step: {dict(funnel_steps(mine_m, split_m, verify_m))}", ""]
    for rid, rs in zip(args.runs, runs):
        costs = [r["cost"] for r in rs if r["cost"] is not None]
        res_costs = [r["cost"] for r in rs if r["resolved"] and r["cost"] is not None]
        L += [f"## Run {rid}", "", f"Resolved {sum(r['resolved'] for r in rs)} of {len(rs)}; "
              f"capped {sum(bool(r['capped']) for r in rs)}; total ${sum(costs):.2f}; "
              f"cost per task median ${median(costs) or 0:.3f}, max ${max(costs) if costs else 0:.3f}; "
              f"resolved median ${median(res_costs) or 0:.3f}, max ${max(res_costs) if res_costs else 0:.3f}", ""]
    if len(runs) >= 2:
        a = {r["task_id"]: r["resolved"] for r in runs[0]}
        b = {r["task_id"]: r["resolved"] for r in runs[1]}
        L += ["## Per task", "", "| task | run 1 | run 2 |", "|---|---|---|"]
        L += [f"| {t} | {'yes' if a.get(t) else 'no'} | {'yes' if b.get(t) else 'no'} |" for t in sorted(a)]
        L.append("")
    if args.labels:
        lab = {j["task_id"]: j["label"] for j in read_jsonl(args.labels)}
        L += ["## By the fairness screen", "", "| | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
        for label in ("fair", "unfair", "unclear"):
            ids = [t for t, v in lab.items() if v == label]
            if ids:
                cells = [f"{sum(r['resolved'] for r in rs if r['task_id'] in ids)} of {len(ids)}" for rs in runs]
                L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    labels = {j["task_id"]: j["label"] for j in read_jsonl(args.labels)} if args.labels else None
    subset = None
    if getattr(args, "label", None):
        if not labels:
            sys.exit("--label needs --labels")
        subset = {t for t, v in labels.items() if v == args.label}
    paired = []
    for pair in paired_groups:
        versions, groups = [], []
        for g in pair:
            vs = {run_version(kd / "runs" / rid) for rid in g}
            if len(vs) != 1:
                sys.exit(f"runs {g} were made with different tickets {sorted(vs)}; a group is one version")
            versions.append(vs.pop())
            groups.append([(rid, read_jsonl(kd / "runs" / rid / "score.jsonl")) for rid in g])
        both = [x for grp in groups for x in grp]
        why = pairing_problem({rid: run_settings(kd / "runs" / rid) for rid, _ in both},
                              {rid: {r["task_id"] for r in rows} for rid, rows in both}, subset)
        if why:
            sys.exit(why)
        if subset is not None:
            groups = [restrict(grp, subset) for grp in groups]
        p = paired_figures(versions, groups, labels)
        paired.append(p)
        L += [f"## Paired: {p['b']} ({', '.join(p['runs_b'])}) against {p['a']} ({', '.join(p['runs_a'])})", "",
              "Per task a version scores 1 when resolved in both its runs, a half in one of two.", "",
              "| tasks | n | both | neither | only " + p["a"] + " | only " + p["b"] + " | net |", "|---|---|---|---|---|---|---|"]
        for name in ("all", "fair", "unfair", "unclear"):
            c = p if name == "all" else p.get(name)
            if c:
                L.append(f"| {name} | {c['tasks']} | {c['both']:g} | {c['neither']:g} | {c['only_a']:g} | "
                         f"{c['only_b']:g} | {c['net_b_minus_a']:+g} |")
        L += ["", "| run | version | resolved | 95% interval | total | per resolved | median per task | capped |",
              "|---|---|---|---|---|---|---|---|"]
        for side, ids in ((p["a"], p["runs_a"]), (p["b"], p["runs_b"])):
            for rid in ids:
                f = p["runs"][rid]
                iv = f["resolved_interval_pct"] or [None, None]
                per = f"${f['cost_per_resolved']:.3f}" if f["cost_per_resolved"] is not None else "n/a"
                L.append(f"| {rid} | {side} | {f['resolved']} of {f['tasks']} | {iv[0]} to {iv[1]}% | "
                         f"${f['cost_total']:.2f} | {per} | ${f['cost_median']:.3f} | {f['capped']} |")
        L += ["", "| task | label | " + p["a"] + " | " + p["b"] + " |", "|---|---|---|---|"]
        L += [f"| {r['task_id']} | {r['label'] or ''} | {r['a']} of {r['of']} | {r['b']} of {r['of']} |"
              for r in p["per_task"] if r["a"] != r["b"]]
        L.append("")
    noise = None
    if getattr(args, "noise", None):
        group = [(rid, read_jsonl(kd / "runs" / rid / "score.jsonl")) for rid in args.noise.split(",")]
        why = pairing_problem({rid: run_settings(kd / "runs" / rid) for rid, _ in group},
                              {rid: {r["task_id"] for r in rows} for rid, rows in group}, subset)
        if why:
            sys.exit(why)
        if subset is not None:
            group = restrict(group, subset)
        noise = noise_figures(group)
        L += noise_lines(noise)
    study = getattr(args, "study", None)
    rewrite_counts = None
    if study == "tickets":
        f = kd / "tickets" / "rewritten.json"
        if f.exists():
            rewrite_counts = json.loads(f.read_text(encoding="utf-8")).get("rewrite")
        preds = ticket_predictions(paired, rewrite_counts)
    else:
        preds = predictions(mine_m, split_m, verify_m, runs)
    L += ["## Predictions", ""]
    L += [f"- {name}: {'HELD' if held else 'FAILED'} ({detail})" for name, held, detail in preds]
    # a noise report asked for on its own must not overwrite the bench's report, so it gets its own default name
    out = Path(args.out) if args.out else kd / ("report-noise.md" if noise is not None else "report.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    fig = report_figures(recipe["repo"], mine_m, split_m, verify_m, list(args.runs), runs, labels)
    if paired:
        fig["paired"] = paired
    if noise is not None:
        fig["noise"] = noise
    if study == "tickets":
        fig["study"] = "tickets"
        fig["rewrite"] = rewrite_counts
        fig["predictions"] = [{"name": name, "verdict": "held" if held else "failed", "detail": detail}
                              for name, held, detail in preds]
    out_json = out.with_suffix(".json")
    out_json.write_text(json.dumps(fig, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(f"{name}: {'HELD' if held else 'FAILED'} ({detail})" for name, held, detail in preds))
    print(f"wrote {out} and {out_json}")


# ---------------------------------------------------------------------------- the context file study (post 5)

def agent_commands(trajectory):
    """Every command the agent chose, in order, one per tool call: messages[].tool_calls[].function.arguments is
    a JSON string with a `command` key."""
    out = []
    for m in trajectory.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            try:
                a = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(a, dict) and isinstance(a.get("command"), str):
                out.append(a["command"])
    return out


def call_usages(trajectory):
    """The usage of each model call, as run_task.py recorded it on the agent's messages."""
    return [m["extra"]["usage"] for m in trajectory.get("messages") or []
            if m.get("role") == "assistant" and (m.get("extra") or {}).get("usage")]


OPERATORS = set(";&|()")
WRAPPERS = {"timeout", "env", "time", "nice", "nohup", "sudo", "exec", "xargs", "stdbuf", "uv"}
WRAPPER_ARGS = {"timeout": 1, "uv": 1}  # words after the options that are not the program (`timeout 60 pytest`, `uv run`)
OPTION_ARGS = {"-n", "-I", "-P", "-L", "-d", "-s", "-E", "-u", "-k", "-o", "-e", "-i"}
SHELLS = {"bash", "sh", "zsh", "dash"}
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
PYTHON = re.compile(r"^python(\d+(\.\d+)?)?$")


def words_of(line):
    """A line split the way the shell would, with its operators as separate words; a line the lexer cannot read
    (an unclosed quote, as inside a heredoc) falls back to whitespace."""
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        return list(lex)
    except ValueError:
        return line.split()


def program_of(words, raw=False):
    """(program, the words after it) for one simple command: environment assignments and wrappers such as
    `timeout 60` or `env X=1` are skipped, and the program is named by its last path component, so that
    /opt/testbed/bin/pytest is pytest (with raw, the word as written, for context.py to resolve). (None, [])
    when there is no program."""
    i = 0
    while i < len(words):
        w = words[i]
        if ASSIGNMENT.match(w):
            i += 1
            continue
        name = w.rsplit("/", 1)[-1]
        if name in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or ASSIGNMENT.match(words[i])):
                i += 2 if words[i] in OPTION_ARGS and name in ("xargs", "nice", "sudo", "env", "stdbuf", "timeout") else 1
            i += WRAPPER_ARGS.get(name, 0)
            continue
        return (w if raw else name), words[i + 1:]
    return None, []


def module_of(program, rest):
    """The module of `python -m module`, after any interpreter options; None for any other command."""
    if not program or not PYTHON.match(program):
        return None
    i = 0
    while i < len(rest):
        w = rest[i]
        if w == "-m":
            return rest[i + 1] if i + 1 < len(rest) else None
        if w.startswith("-m") and len(w) > 2:
            return w[2:]
        if w in ("-W", "-X"):
            i += 2
            continue
        if not w.startswith("-") or w == "-c":
            return None
        i += 1
    return None


def commands_with_args(command, depth=0, raw=False):
    """(program, module, the words after the program) for each simple command in a shell command line: split at
    newlines and at the shell's operators, and read into `bash -c '...'`. The word pytest inside a path, a file
    name, a quoted string or an argument is never a program here, which is what makes the test-runner count a
    count of invocations."""
    out = []
    for line in command.splitlines():
        cur = []
        for w in words_of(line) + [";"]:
            if w and set(w) <= OPERATORS:
                if cur:
                    word, rest = program_of(cur, raw=True)
                    if word:
                        prog = word.rsplit("/", 1)[-1]
                        out.append((word if raw else prog, module_of(prog, rest), rest))
                        if prog in SHELLS and "-c" in rest and depth < 3:
                            i = rest.index("-c")
                            if i + 1 < len(rest):
                                out += commands_with_args(rest[i + 1], depth + 1, raw)
                cur = []
            else:
                cur.append(w)
    return out


def simple_commands(command, depth=0, raw=False):
    """(program, module) for each simple command in a shell command line (see commands_with_args)."""
    return [(p, m) for p, m, _rest in commands_with_args(command, depth, raw)]


NETWORK_GIT = {"clone", "fetch", "pull", "push"}
PIP = re.compile(r"^pip(\d+(\.\d+)?)?$")


def first_argument(words, takes_value=()):
    """The first word that is not an option, skipping the value of the options named in takes_value."""
    i = 0
    while i < len(words):
        w = words[i]
        if w in takes_value:
            i += 2
        elif w.startswith("-"):
            i += 1
        else:
            return w
    return None


def needs_network(command):
    """One agent command that cannot work behind the bench's closed network: `pip install` (as pip or as
    python -m pip, unless it says --no-index) or git with clone, fetch, pull or push. The "commands work" check
    resolves programs, so it passes pip and git; this count is what it misses."""
    for prog, module, rest in commands_with_args(command):
        if module == "pip":
            rest = rest[rest.index("pip") + 1:] if "pip" in rest else []
            prog = "pip"
        if PIP.match(prog or "") and first_argument(rest) == "install" and "--no-index" not in rest:
            return True
        if prog == "git" and first_argument(rest, takes_value=("-C", "-c")) in NETWORK_GIT:
            return True
    return False


def is_test_runner(command):
    """One agent command that invokes pytest, python -m pytest, py.test or tox as a program."""
    return any(p in TEST_RUNNERS or m in TEST_RUNNERS for p, m in simple_commands(command))


def flagged_name(entry):
    """A flagged program as context.py check records it: `tox`, or `python3 -m tox` for a module."""
    return entry.split(" -m ", 1)[1].strip() if " -m " in entry else entry.rsplit("/", 1)[-1]


def calls_flagged(command, flagged):
    """The flagged programs one agent command calls, as a program or as `python -m`."""
    names = {flagged_name(f) for f in flagged}
    hit = set()
    for p, m in simple_commands(command):
        hit |= names & {p, m}
    return sorted(hit)


def inside_floor(net, floor):
    """abs(net) <= floor, with a tolerance far below a quarter task so that rounding in the square root cannot
    decide a tie (a net of exactly the floor is inside it, as the spec words it)."""
    return abs(net) <= floor + 1e-9


def noise_sd(F, a, b):
    return math.sqrt((F / 2) * (1 / a + 1 / b))


def noise_floor(F, a, b):
    """The size a net between arms of a and b runs must exceed before it counts as a difference."""
    return NOISE_SDS * noise_sd(F, a, b)


def tasks_needed(F, n, a, b, points):
    """Tasks a bench needs to see a difference of `points` percentage points between arms of a and b runs, at
    the flip rate f = F/n: N >= NOISE_SDS^2 * (f/2) * (1/a + 1/b) / (points/100)^2."""
    f = F / n
    return max(1, math.ceil(NOISE_SDS ** 2 * (f / 2) * (1 / a + 1 / b) / (points / 100) ** 2))


def repeats_needed(F, n, points):
    """Runs per arm, the same number on both sides, for n tasks to see a difference of `points` points."""
    f = F / n
    return max(1, math.ceil(NOISE_SDS ** 2 * f / (n * (points / 100) ** 2)))


def resolved_of(rows, ids=None):
    return {r["task_id"]: bool(r["resolved"]) for r in rows if ids is None or r["task_id"] in ids}


def flips(a_rows, b_rows, ids=None):
    """Tasks two single runs disagree on, over the tasks both scored."""
    a, b = resolved_of(a_rows, ids), resolved_of(b_rows, ids)
    return sum(a[t] != b[t] for t in set(a) & set(b))


HALVES = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def restrict(group, ids):
    return [(rid, [r for r in rows if r["task_id"] in ids]) for rid, rows in group]


def noise_figures(none, arms=None, arm_runs=2):
    """Step 1 of the method: flips for every pairing of the identical runs, their mean F, the nets between the
    halves of four runs split all three ways, the flips of each context arm's own pair, and the floor for an
    arm of `arm_runs` runs against all the identical runs, and for two against two. Pure."""
    pairings = [{"runs": [x[0], y[0]], "flips": flips(x[1], y[1])} for x, y in itertools.combinations(none, 2)]
    F = sum(p["flips"] for p in pairings) / len(pairings) if pairings else 0.0
    splits = []
    if len(none) == 4:
        for A, B in HALVES:
            ga, gb = [none[i] for i in A], [none[i] for i in B]
            p = paired_figures(["half a", "half b"], [ga, gb])
            splits.append({"a": [rid for rid, _ in ga], "b": [rid for rid, _ in gb], "net": p["net_b_minus_a"]})
    own = {arm: {"runs": [rid for rid, _ in g], "flips": flips(g[0][1], g[1][1])}
           for arm, g in (arms or {}).items() if len(g) == 2}
    b = len(none)
    return {"runs": [rid for rid, _ in none], "pairings": pairings, "F": F, "splits": splits, "arm_pairs": own,
            "floor": {"arm_vs_none": {"a": arm_runs, "b": b, "sd": round(noise_sd(F, arm_runs, b), 3),
                                      "floor": round(noise_floor(F, arm_runs, b), 3),
                                      "floor_exact": noise_floor(F, arm_runs, b)},
                      "two_vs_two": {"a": 2, "b": 2, "sd": round(noise_sd(F, 2, 2), 3),
                                     "floor": round(noise_floor(F, 2, 2), 3), "floor_exact": noise_floor(F, 2, 2)}}}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def per_task_mean(group, value, ids):
    """For each task, the mean over the group's runs of value(row), skipping runs where it is None."""
    out = {}
    for t in ids:
        vals = [value(r) for _rid, rows in group for r in rows if r["task_id"] == t]
        out[t] = mean(vals)
    return out


def cost_ratios(none, arm, ids):
    """Per task, the arm's mean cost over the no-file mean cost, for tasks where both are known and the no-file
    mean is above zero."""
    a = per_task_mean(none, lambda r: r.get("cost"), ids)
    b = per_task_mean(arm, lambda r: r.get("cost"), ids)
    return {t: b[t] / a[t] for t in ids if a[t] and b[t] is not None}, a, b


def file_cost_of(usages, api_calls, file_tokens, model_cfg):
    """What a context file's own tokens cost on one task: the file's tokens times the input price of each call's
    tier, summed over the calls (this harness caches nothing, so the file is paid for on every call). Without
    per-call usage, every call is priced at the lower tier."""
    if str(BENCH / "runner") not in sys.path:
        sys.path.insert(0, str(BENCH / "runner"))
    import agent_parts
    if usages:
        return sum(file_tokens * agent_parts.input_price_per_token(u, model_cfg) for u in usages)
    return (api_calls or 0) * file_tokens * agent_parts.per_token(model_cfg["price_per_mtok"])["input"]


def first_prompt_tokens(usages):
    """{task: the prompt tokens of the run's first model call} from {task: [usage per call]}."""
    if str(BENCH / "runner") not in sys.path:
        sys.path.insert(0, str(BENCH / "runner"))
    import agent_parts
    return {t: agent_parts.prompt_tokens(u[0]) for t, u in usages.items() if u}


def measured_file_tokens(arm_first, none_first):
    """A context file's size as the model counts it: over the arm's runs and tasks, the median of the first
    call's prompt tokens minus the mean first-call prompt tokens of the no-file runs on the same task. The two
    prompts differ only by the context block, so the difference is the block. arm_first and none_first are
    {run id: {task: tokens}}. None when no task has both. Pure."""
    diffs = []
    for per in arm_first.values():
        for t, n in per.items():
            base = mean([x[t] for x in none_first.values() if t in x])
            if base is not None:
                diffs.append(n - base)
    return median(diffs)


def cost_figures(none, arm, ids, file_cost=None):
    """Cost task by task for one arm against the no-file runs, and the arm's extra cost split in two: the file's
    own resent tokens (file_cost[rid][task], dollars) and the rest, which is the agent doing more. Pure."""
    ratios, a, b = cost_ratios(none, arm, ids)
    both = [t for t in ids if a[t] is not None and b[t] is not None]
    extra = sum(b[t] - a[t] for t in both)
    own = None
    if file_cost is not None:
        own = sum(mean([file_cost.get(rid, {}).get(t) for rid, _ in arm]) or 0.0 for t in both)
    med = median(list(ratios.values()))
    return {"tasks": len(ratios), "ratio_median": round(med, 3) if med is not None else None,
            "ratio_median_exact": med, "extra_total": round(extra, 4),
            "file_own_total": round(own, 4) if own is not None else None,
            "file_own_share_pct": round(100 * own / extra) if own is not None and extra > 0 else None,
            "remainder_total": round(extra - own, 4) if own is not None else None,
            "none_mean_total": round(sum(a[t] for t in both), 4), "arm_mean_total": round(sum(b[t] for t in both), 4),
            "per_task": [{"task_id": t, "none_mean": round(a[t], 4) if a[t] is not None else None,
                          "arm_mean": round(b[t], 4) if b[t] is not None else None,
                          "ratio": round(ratios[t], 3) if t in ratios else None} for t in ids]}


def split_cost_ratios(none, ids):
    """The same median ratio between the halves of the four no-file runs, for the three splits, and the largest
    of them read either way (each ratio or its inverse, whichever is larger)."""
    out = []
    for A, B in HALVES:
        r, _, _ = cost_ratios([none[i] for i in A], [none[i] for i in B], ids)
        m = median(list(r.values()))
        out.append({"a": [none[i][0] for i in A], "b": [none[i][0] for i in B],
                    "ratio_median": round(m, 3) if m is not None else None, "ratio_median_exact": m})
    largest = max((max(s["ratio_median_exact"], 1 / s["ratio_median_exact"]) for s in out
                   if s["ratio_median_exact"]), default=None)
    return out, largest


def test_runner_counts(cmds, ids):
    """Per task, the number of the run's agent commands that invoke a test runner (0 without a trajectory)."""
    return {t: sum(is_test_runner(c) for c in cmds.get(t, [])) for t in ids}


def behaviour_figures(none_ids, arm_ids, cmds, ids, flagged=()):
    """Test-runner commands per task, the arm's mean minus the no-file mean task by task and its median, and the
    agent's commands that call a flagged program, per run. cmds[rid][task] is the list of commands. Pure."""
    counts = {rid: test_runner_counts(cmds.get(rid, {}), ids) for rid in list(none_ids) + list(arm_ids)}
    na = {t: mean([counts[r][t] for r in none_ids]) for t in ids}
    aa = {t: mean([counts[r][t] for r in arm_ids]) for t in ids}
    diffs = [aa[t] - na[t] for t in ids]
    fl = {}
    for rid in list(none_ids) + list(arm_ids):
        per = collections.Counter()
        n = 0
        for t in ids:
            for c in cmds.get(rid, {}).get(t, []):
                hit = calls_flagged(c, flagged)
                n += bool(hit)
                per.update(hit)
        fl[rid] = {"commands": n, "by_program": dict(sorted(per.items()))}
    net = {rid: sum(1 for t in ids for c in cmds.get(rid, {}).get(t, []) if needs_network(c))
           for rid in list(none_ids) + list(arm_ids)}
    med = median(diffs)
    return {"none_mean_per_task": round(mean(list(na.values())) or 0, 3),
            "arm_mean_per_task": round(mean(list(aa.values())) or 0, 3),
            "none_median_per_task": median(list(na.values())), "arm_median_per_task": median(list(aa.values())),
            "diff_median": med, "flagged": list(flagged), "flagged_calls": fl, "network_calls": net,
            "missing_trajectories": {rid: sum(1 for t in ids if t not in cmds.get(rid, {})) for rid in counts},
            "per_task": [{"task_id": t, "none_mean": na[t], "arm_mean": aa[t]} for t in ids]}


DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/", re.M)


def diff_paths(diff_text):
    """The paths a submitted diff touches."""
    return DIFF_FILE.findall(diff_text or "")


def asks_figures(paths, asks, ids):
    """Did the agent do what the file asked? For each ask (a name and the path prefixes that show it was
    followed, from the recipe's `context_asks`), the number of tasks per run whose submitted diff touches such a
    path. paths[rid][task] is the list of paths in that run's diff. A count of diffs, not of intentions: an agent
    may add a changelog fragment unasked, which is why the no-file runs are counted beside the others. Pure."""
    def hit(ps, prefixes):
        return any(p == x or p.startswith(x) for p in ps for x in prefixes)
    return {name: {rid: sum(1 for t in ids if hit(per.get(t, []), prefixes)) for rid, per in paths.items()}
            for name, prefixes in asks.items()}


def compare_arm(none, arm, ids, cmds):
    """Net, median per-task cost ratio and median test-runner difference of one arm against no file, on ids."""
    p = paired_figures(["none", "arm"], [restrict(none, ids), restrict(arm, ids)])
    ratios, _, _ = cost_ratios(none, arm, ids)
    b = behaviour_figures([r for r, _ in none], [r for r, _ in arm], cmds, ids)
    return {"tasks": len(ids), "net": p["net_b_minus_a"], "cost_ratio_median": median(list(ratios.values())),
            "test_runner_diff_median": b["diff_median"]}


def sign(x, around=0.0):
    return 0 if x is None or x == around else (1 if x > around else -1)


def validation_figures(pool, haiku, haiku_verdicts):
    """The validation pool: for each file, the larger model's figures beside Haiku's on the same tasks, and
    agree or disagree by the spec's rule: the cost ratio on the same side of 1, the test-runner difference of
    the same sign, and the net within CONTEXT_VALIDATION_NET of zero where Haiku's verdict was inside its floor,
    or beyond it with the same sign where it was outside. haiku_verdicts[arm] = (inside, net) from the full
    bench. Pure."""
    out = {}
    for arm, v in pool.items():
        h = haiku[arm]
        inside, hnet = haiku_verdicts[arm]
        if inside:
            net_ok = abs(v["net"]) <= CONTEXT_VALIDATION_NET
        else:
            net_ok = abs(v["net"]) > CONTEXT_VALIDATION_NET and sign(v["net"]) == sign(hnet)
        out[arm] = {"pool": v, "haiku_same_tasks": h, "haiku_inside_floor": inside, "haiku_net": hnet,
                    "agree": {"cost": sign(v["cost_ratio_median"], 1.0) == sign(h["cost_ratio_median"], 1.0),
                              "test_runner": sign(v["test_runner_diff_median"]) == sign(h["test_runner_diff_median"]),
                              "net": net_ok}}
    return out


def context_predictions(ctx):
    """The context file study's five predictions, exactly as its spec words them, from the figures alone."""
    floor = ctx["noise"]["floor"]["arm_vs_none"]["floor_exact"]
    out = []
    for n, arm, title in ((1, "project-docs", "The project's own docs are inside the noise"),
                          (2, "generated", "The generated file is inside the noise too")):
        if arm in ctx["paired"]:
            net = ctx["paired"][arm]["net_b_minus_a"]
            out.append((f"{n}. {title}", inside_floor(net, floor), f"net {net:+g} against a floor of {floor:.2f}"))
    if "generated" in ctx["cost"]:
        r = ctx["cost"]["generated"]["ratio_median_exact"]
        largest = ctx["cost_splits"]["largest"]
        held = r is not None and r >= CONTEXT_COST_RATIO and (largest is None or r > largest)
        out.append(("3. The generated file costs more, task by task", held,
                    f"median ratio {r:.3f}, at least {CONTEXT_COST_RATIO} and above the largest split ratio "
                    f"{largest:.3f}" if r is not None and largest is not None else f"median ratio {r}, splits {largest}"))
    diffs = {arm: b["diff_median"] for arm, b in ctx["behaviour"].items()}
    if diffs:
        out.append(("4. Context files make the agent test more",
                    all(d is not None and d >= CONTEXT_TEST_RUNNER_GAIN for d in diffs.values()),
                    "; ".join(f"{arm}: median difference {d:+g}" for arm, d in diffs.items())))
    gaps = ctx.get("single_run_gaps") or {}
    if gaps:
        def misleads(g):
            return max(g) - min(g) >= CONTEXT_SINGLE_RUN_SPAN or (min(g) < 0 < max(g))
        out.append(("5. One run each would have misled", any(misleads(g) for g in gaps.values()),
                    "; ".join(f"{arm}: gaps {sorted(g)}" for arm, g in gaps.items())))
    return out


def context_figures(data):
    """Every block of report-context.json from loaded runs, so the tests can feed it hand-written rows. data:
    n_tasks; labels; none, a list of (run id, rows); arms, {arm: [(run id, rows)]}; cmds, {run id: {task:
    [commands]}}; file_cost, {run id: {task: dollars}}; flagged, {arm: [programs]}; pool, {arm: (run id, rows)};
    pool_base, [(run id, rows)]; and, when a training data cutoff applies, merged_at and cutoff. Pure."""
    none, arms, labels, cmds = data["none"], data["arms"], data.get("labels") or {}, data.get("cmds") or {}
    ids = sorted(set.intersection(*[{r["task_id"] for r in rows} for _rid, rows in none + [g for a in arms.values() for g in a]]))
    n1 = none[0]
    k = sum(r["resolved"] for r in n1[1])
    ctx = {"tasks": len(ids),
           "pilot": {"run": n1[0], "resolved": k, "tasks": len(n1[1]), "threshold": CONTEXT_PILOT_MIN,
                     "passed": k >= CONTEXT_PILOT_MIN},
           "noise": noise_figures(none, arms, arm_runs=2), "paired": {}, "cost": {}, "behaviour": {},
           "single_run_gaps": {}}
    F = ctx["noise"]["F"]
    flagged_all = sorted({f for fs in (data.get("flagged") or {}).values() for f in fs})
    none_ids = [rid for rid, _ in none]
    for arm, group in arms.items():
        p = paired_figures(["none", arm], [restrict(none, ids), restrict(group, ids)], labels)
        allr = [resolved_of(rows, ids) for _rid, rows in none + group]
        p["tasks_apart"] = sum(1 for t in ids if 0 < sum(r[t] for r in allr) < len(allr))
        if data.get("cutoff") and data.get("merged_at"):
            for side, keep in (("before_cutoff", lambda d: d < data["cutoff"]), ("from_cutoff", lambda d: d >= data["cutoff"])):
                sub = {t for t in ids if keep(data["merged_at"][t][:10])}
                if sub:
                    q = paired_figures(["none", arm], [restrict(none, sub), restrict(group, sub)])
                    p[side] = {k2: q[k2] for k2 in ("tasks", "both", "neither", "only_a", "only_b", "net_b_minus_a")}
        ctx["paired"][arm] = p
        ctx["cost"][arm] = cost_figures(none, group, ids, data.get("file_cost"))
        if data.get("file_cost_estimate") is not None:  # the same split with the file sized at 4 characters a token
            est = cost_figures(none, group, ids, data["file_cost_estimate"])
            ctx["cost"][arm]["file_own_total_estimate"] = est["file_own_total"]
        ctx["behaviour"][arm] = behaviour_figures(none_ids, [rid for rid, _ in group], cmds, ids,
                                                  flagged_all)
        ctx["single_run_gaps"][arm] = [sum(resolved_of(g, ids).values()) - sum(resolved_of(x, ids).values())
                                       for _ga, g in group for _xa, x in none]
    splits, largest = split_cost_ratios(none, ids) if len(none) == 4 else ([], None)
    ctx["cost_splits"] = {"splits": splits, "largest": largest}
    n = len(ids)
    fl = noise_floor(F, 2, len(none))
    ctx["detectable"] = {"floor_tasks": round(fl, 2), "floor_points": round(100 * fl / n, 1) if n else None,
                         "tasks_needed": {str(x): tasks_needed(F, n, 2, len(none), x) for x in CONTEXT_POINTS} if n else {},
                         "repeats_needed_per_arm": {str(x): repeats_needed(F, n, x) for x in CONTEXT_POINTS} if n else {},
                         "a": 2, "b": len(none), "F": F}
    ctx["predictions"] = [{"name": nm, "verdict": "held" if held else "failed", "detail": d}
                          for nm, held, d in context_predictions(ctx)]
    if data.get("asks") and data.get("diff_paths"):
        fair_ids = sorted(t for t in ids if labels.get(t) == CONTEXT_POOL_LABEL)
        main = [rid for rid, _ in none] + [rid for g in arms.values() for rid, _ in g]
        pool_ids = [rid for rid, _ in data.get("pool_base") or []] + [rid for rid, _ in (data.get("pool") or {}).values()]
        dp = data["diff_paths"]
        ctx["asks"] = {"all_tasks": asks_figures({r: dp.get(r, {}) for r in main}, data["asks"], ids),
                       "pool_tasks": asks_figures({r: dp.get(r, {}) for r in main + pool_ids}, data["asks"], fair_ids),
                       "tasks": len(ids), "pool_task_count": len(fair_ids)}
    if data.get("pool"):
        fair = sorted(t for t in ids if labels.get(t) == CONTEXT_POOL_LABEL)
        pool, haiku, verdicts = {}, {}, {}
        for arm, (rid, rows) in data["pool"].items():
            pool[arm] = {"run": rid, "against": [r for r, _ in data["pool_base"]],
                         **compare_arm(data["pool_base"], [(rid, rows)], fair, cmds)}
            haiku[arm] = compare_arm(none, arms[arm], fair, cmds)
            verdicts[arm] = (inside_floor(ctx["paired"][arm]["net_b_minus_a"], ctx["noise"]["floor"]["arm_vs_none"]["floor_exact"]),
                             ctx["paired"][arm]["net_b_minus_a"])
        ctx["validation"] = {"label": CONTEXT_POOL_LABEL, "tasks": len(fair), "rule_net": CONTEXT_VALIDATION_NET,
                             "arms": validation_figures(pool, haiku, verdicts)}
    return ctx


def read_text_or_empty(path):
    path = Path(path)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def run_settings(run_dir):
    f = Path(run_dir) / "settings.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def pairing_problem(settings, tasks, subset=None):
    """Why a set of runs cannot be read side by side, or None: their settings name different models, or they
    were made over different task sets and no subset was named (with one, every run must hold all of it).
    settings[rid] is the run's settings.json or None (an old run: not checked); tasks[rid] its scored task ids."""
    models = {rid: (s or {}).get("harness", {}).get("model", {}).get("model_name") for rid, s in settings.items()}
    known = {m for m in models.values() if m}
    if len(known) > 1:
        return f"the runs name different models: {models}"
    sets = {frozenset(v) for v in tasks.values()}
    if subset is not None:
        short = {rid: len(set(subset) - set(v)) for rid, v in tasks.items() if set(subset) - set(v)}
        return f"runs lack tasks of the subset: {short}" if short else None
    if len(sets) > 1:
        return (f"the runs cover different task sets ({ {rid: len(v) for rid, v in tasks.items()} } tasks); name the "
                "subset to read with --label")
    return None


def load_run_data(kd, rid):
    """Scored rows, settings, agent commands and per-call usage for one run, from its folder."""
    rd = Path(kd) / "runs" / rid
    if not (rd / "score.jsonl").exists():
        sys.exit(f"run {rid} has no score.jsonl; score it first")
    rows = read_jsonl(rd / "score.jsonl")
    cmds, usages = {}, {}
    for r in rows:
        f = rd / r["task_id"] / "trajectory.json"
        if f.exists():
            tr = json.loads(f.read_text(encoding="utf-8"))
            cmds[r["task_id"]], usages[r["task_id"]] = agent_commands(tr), call_usages(tr)
    return rows, run_settings(rd), cmds, usages


def context_lines(ctx, files):
    """report-context.md, block by block, from the figures alone."""
    nz = ctx["noise"]
    L = ["# Context file study report", "", f"Tasks read: {ctx['tasks']}.", "", "## Files", ""]
    for arm, f in files.items():
        L.append(f"- {arm}: {f.get('lines')} lines, about {f.get('tokens_estimate')} tokens by estimate and "
                 f"{f.get('tokens_measured')} measured from the first call, sha256 "
                 f"{(f.get('sha256') or '')[:16]}; programs flagged: {', '.join(f.get('flagged') or []) or 'none'}; "
                 f"statements against the harness prompt: {f.get('contradictions')}")
    p = ctx["pilot"]
    L += ["", "## Pilot gate", "", f"{p['run']} resolved {p['resolved']} of {p['tasks']}; threshold {p['threshold']}: "
          f"{'passed' if p['passed'] else 'NOT passed'}.", "", "## Noise", "",
          "| pairing | flips |", "|---|---|"]
    L += [f"| {' / '.join(x['runs'])} | {x['flips']} |" for x in nz["pairings"]]
    L += [f"| {arm} ({' / '.join(x['runs'])}) | {x['flips']} |" for arm, x in nz["arm_pairs"].items()]
    fa, f2 = nz["floor"]["arm_vs_none"], nz["floor"]["two_vs_two"]
    L += ["", f"Mean flips F over the no-file pairings: {nz['F']:.2f}. Split nets: "
          + ", ".join(f"{'+'.join(s['b'])} against {'+'.join(s['a'])} {s['net']:+g}" for s in nz["splits"]) + ".",
          f"Noise floor, {fa['a']} runs against {fa['b']}: {fa['floor_exact']:.2f} tasks (sd {fa['sd']:.2f}); two against "
          f"two: {f2['floor_exact']:.2f}.", "", "## Paired, each arm against no file", "",
          "| arm | tasks | n | both | neither | only none | only arm | net | apart |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, q in ctx["paired"].items():
        for name in ("all", "fair", "unfair", "unclear", "before_cutoff", "from_cutoff"):
            c = q if name == "all" else q.get(name)
            if c:
                L.append(f"| {arm} | {name} | {c['tasks']} | {c['both']:g} | {c['neither']:g} | {c['only_a']:g} | "
                         f"{c['only_b']:g} | {c['net_b_minus_a']:+g} | {q['tasks_apart'] if name == 'all' else ''} |")
    L += ["", "| run | resolved | total | per resolved | median per task | calls median | capped |", "|---|---|---|---|---|---|---|"]
    seen = set()
    for q in ctx["paired"].values():
        for rid, f in q["runs"].items():
            if rid in seen:
                continue
            seen.add(rid)
            per = f"${f['cost_per_resolved']:.3f}" if f["cost_per_resolved"] is not None else "n/a"
            L.append(f"| {rid} | {f['resolved']} of {f['tasks']} | ${f['cost_total']:.2f} | {per} | ${f['cost_median']:.3f} | "
                     f"{f['calls_median']} | {f['capped']} |")
    L += ["", "## Cost, task by task", ""]
    for arm, c in ctx["cost"].items():
        own = "n/a" if c["file_own_total"] is None else f"${c['file_own_total']:.4f}"
        rest = "n/a" if c["remainder_total"] is None else f"${c['remainder_total']:.4f}"
        L.append(f"- {arm}: median per-task ratio {c['ratio_median']} over {c['tasks']} tasks; extra ${c['extra_total']:.4f} "
                 f"in all, of which the file's own tokens {own} and the rest {rest}"
                 + (f" (the file's own tokens by the four-characters estimate: ${c['file_own_total_estimate']:.4f})"
                    if c.get("file_own_total_estimate") is not None else ""))
    L.append("- no-file halves: " + ", ".join(f"{'+'.join(s['b'])} over {'+'.join(s['a'])} {s['ratio_median']}"
                                            for s in ctx["cost_splits"]["splits"])
             + f"; largest either way {ctx['cost_splits']['largest']}")
    L += ["", "## Behaviour", ""]
    for arm, b in ctx["behaviour"].items():
        L.append(f"- {arm}: test-runner commands per task, mean {b['arm_mean_per_task']} against {b['none_mean_per_task']} "
                 f"with no file; median difference task by task {b['diff_median']:+g}; commands calling a flagged "
                 f"program: " + ", ".join(f"{rid} {v['commands']}" for rid, v in b["flagged_calls"].items())
                 + "; commands that need the network: "
                 + ", ".join(f"{rid} {n}" for rid, n in b.get("network_calls", {}).items()))
    if ctx.get("asks"):
        a = ctx["asks"]
        L += ["", "## Did the agent do what the files ask?", "",
              f"Tasks whose submitted diff shows the ask, per run, of {a['tasks']} tasks:", ""]
        for name, per in a["all_tasks"].items():
            L.append(f"- {name}: " + ", ".join(f"{rid} {n}" for rid, n in per.items()))
        L += ["", f"On the pool's {a['pool_task_count']} tasks, the pool's runs included:", ""]
        for name, per in a["pool_tasks"].items():
            L.append(f"- {name}: " + ", ".join(f"{rid} {n}" for rid, n in per.items()))
    d = ctx["detectable"]
    L += ["", "## What this bench can see", "",
          f"Smallest net above the floor: {d['floor_tasks']:.2f} tasks, {d['floor_points']} points of resolved rate.",
          "Tasks needed (2 runs against 4): " + ", ".join(f"{x} points {n}" for x, n in d["tasks_needed"].items())
          + ". Repeats per arm on these tasks: " + ", ".join(f"{x} points {n}" for x, n in d["repeats_needed_per_arm"].items()) + "."]
    if ctx.get("validation"):
        v = ctx["validation"]
        L += ["", f"## Validation pool ({v['tasks']} {v['label']} tasks)", "",
              "| file | figure | pool | Haiku, same tasks | agree |", "|---|---|---|---|---|"]
        for arm, a in v["arms"].items():
            for key, fig in (("net", "net"), ("cost", "cost_ratio_median"), ("test_runner", "test_runner_diff_median")):
                L.append(f"| {arm} | {fig} | {a['pool'][fig]} | {a['haiku_same_tasks'][fig]} | "
                         f"{'agree' if a['agree'][key] else 'disagree'} |")
    L += ["", "## Predictions", ""]
    L += [f"- {x['name']}: {x['verdict'].upper()} ({x['detail']})" for x in ctx["predictions"]]
    return L


def context_report(args, recipe, kd, labels):
    """report --study context: load the ten Haiku and pool runs and the two frozen files, check that they may be
    read side by side, and write report-context.md and report-context.json."""
    manifest = json.loads((kd / "tasks.json").read_text(encoding="utf-8"))
    loaded = {}
    every = list(CONTEXT_NONE_RUNS) + [r for g in CONTEXT_ARMS.values() for r in g] + list(CONTEXT_POOL.values()) \
        + list(CONTEXT_POOL_BASE)
    for rid in every:
        loaded[rid] = load_run_data(kd, rid)
    files, flagged = {}, {}
    for arm in CONTEXT_ARMS:
        try:
            doc = load_context(kd, arm, manifest)
        except ValueError as e:
            sys.exit(str(e))
        chk = doc.get("check") or {}
        flagged[arm] = chk.get("flagged_programs") or []
        files[arm] = {"sha256": doc["sha256"], "lines": doc.get("lines"), "tokens_estimate": doc.get("tokens_estimate"),
                      "flagged": flagged[arm], "contradictions": len(chk.get("contradictions") or []),
                      "source": doc.get("source")}
    for rid in every:
        s = loaded[rid][1] or {}
        want = "none"
        for arm, g in CONTEXT_ARMS.items():
            if rid in g or CONTEXT_POOL.get(arm) == rid:
                want = arm
        have = s.get("context", {"name": "none"})
        if have.get("name") != want or (want != "none" and have.get("sha256") != files[want]["sha256"]):
            sys.exit(f"run {rid} was made with context file {have}, not {want} as frozen")
    haiku = list(CONTEXT_NONE_RUNS) + [r for g in CONTEXT_ARMS.values() for r in g]
    why = pairing_problem({r: loaded[r][1] for r in haiku}, {r: {x["task_id"] for x in loaded[r][0]} for r in haiku})
    fair = {t for t, v in labels.items() if v == CONTEXT_POOL_LABEL}
    pool_runs = list(CONTEXT_POOL.values()) + list(CONTEXT_POOL_BASE)
    why = why or pairing_problem({r: loaded[r][1] for r in pool_runs},
                                 {r: {x["task_id"] for x in loaded[r][0]} for r in pool_runs}, subset=fair)
    if why:
        sys.exit(why)
    file_cost, file_cost_estimate = {}, {}
    none_first = {r: first_prompt_tokens(loaded[r][3]) for r in CONTEXT_NONE_RUNS}
    for arm, g in CONTEXT_ARMS.items():
        estimate = files[arm]["tokens_estimate"] or 0
        measured = measured_file_tokens({r: first_prompt_tokens(loaded[r][3]) for r in g}, none_first)
        files[arm]["tokens_measured"] = measured
        for rid in g:
            rows, s, _c, usages = loaded[rid]
            model = s["harness"]["model"]
            for tokens, into in ((estimate if measured is None else measured, file_cost), (estimate, file_cost_estimate)):
                into[rid] = {r["task_id"]: file_cost_of(usages.get(r["task_id"]), r.get("api_calls"), tokens, model)
                             for r in rows}
    data = {"n_tasks": len(manifest["tasks"]), "labels": labels,
            "none": [(r, loaded[r][0]) for r in CONTEXT_NONE_RUNS],
            "arms": {arm: [(r, loaded[r][0]) for r in g] for arm, g in CONTEXT_ARMS.items()},
            "cmds": {r: loaded[r][2] for r in every}, "file_cost": file_cost,
            "file_cost_estimate": file_cost_estimate, "flagged": flagged,
            "asks": recipe.get("context_asks") or {},
            "diff_paths": {rid: {r["task_id"]: diff_paths(read_text_or_empty(kd / "runs" / rid / r["task_id"] / "submission.diff"))
                                 for r in loaded[rid][0]} for rid in every},
            "pool": {arm: (rid, loaded[rid][0]) for arm, rid in CONTEXT_POOL.items()},
            "pool_base": [(r, loaded[r][0]) for r in CONTEXT_POOL_BASE],
            "merged_at": {t["task_id"]: t["merged_at"] for t in manifest["tasks"]},
            "cutoff": getattr(args, "training_cutoff", None)}
    ctx = context_figures(data)
    out = Path(args.out) if args.out else kd / "report-context.md"
    out.write_text("\n".join(context_lines(ctx, files)) + "\n", encoding="utf-8")
    fig = {"repo": recipe["repo"], "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "study": "context",
           "tasks_checksum": manifest["checksum"], "files": files, "training_cutoff": data["cutoff"], **ctx}
    out_json = out.with_suffix(".json")
    out_json.write_text(json.dumps(fig, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(f"{p['name']}: {p['verdict'].upper()} ({p['detail']})" for p in ctx["predictions"]))
    print(f"wrote {out} and {out_json}")


def noise_lines(nz):
    fa, f2 = nz["floor"]["arm_vs_none"], nz["floor"]["two_vs_two"]
    L = ["## Noise", "", "| pairing | flips |", "|---|---|"]
    L += [f"| {' / '.join(x['runs'])} | {x['flips']} |" for x in nz["pairings"]]
    L += ["", f"Mean flips F: {nz['F']:.2f}. Split nets: "
          + (", ".join(f"{'+'.join(s['b'])} against {'+'.join(s['a'])} {s['net']:+g}" for s in nz["splits"]) or "none (needs four runs)")
          + f". Noise floor, two runs against two: {f2['floor_exact']:.2f} tasks; {fa['a']} against {fa['b']}: "
          f"{fa['floor_exact']:.2f}.", ""]
    return L


# ---------------------------------------------------------------------------- all

def all_steps(args):
    """Run the pipeline end to end. Every step is restartable, so rerunning after a failure resumes."""
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    kd.mkdir(parents=True, exist_ok=True)
    log = kd / "all.log"
    run_ids = [f"{args.run_prefix}{i}" for i in range(1, args.runs + 1)]

    def note(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def step(name, func, **kw):
        note(f"== {name}")
        t0 = time.time()
        try:
            func(argparse.Namespace(**kw))
        except SystemExit as e:
            if e.code not in (None, 0):
                note(f"!! {name} stopped: {e.code}")
                raise
        except Exception as e:  # noqa: BLE001
            note(f"!! {name} failed: {type(e).__name__}: {e}")
            raise
        note(f"ok {name} in {time.time() - t0:.0f}s")

    note(f"all: {recipe['repo']} (runs {run_ids}, repeats {args.repeats}, spend cap ${args.spend_cap} per run)")
    if args.since or not (kd / "candidates.jsonl").exists():
        if not args.since:
            sys.exit("no candidates yet: pass --since YYYY-MM-DD (the leakage window) so that mine can run")
        step("mine", mine, repo=recipe["repo"], since=args.since, target=args.target,
             test_dirs=",".join(recipe.get("test_dirs", list(DEFAULT_TEST_DIRS))), out=None, refresh=False)
    else:
        note("mine: candidates.jsonl exists, kept (pass --since to mine again)")
    step("image", image, recipe=args.recipe)
    step("split", split, recipe=args.recipe, repo_dir=args.repo_dir)
    step("verify", verify, recipe=args.recipe, repeats=args.repeats, only=None)
    step("freeze", freeze, recipe=args.recipe, force=False)
    step("packets", packets, recipe=args.recipe)
    if args.skip_run:
        note("runs skipped (--skip-run); the bench is frozen and ready")
        return
    cfg = json.loads((BENCH / "harness.json").read_text(encoding="utf-8"))
    if not os.environ.get(cfg["model"]["api_key_env"]):
        sys.exit(f"the bench is frozen; set {cfg['model']['api_key_env']} and rerun to start the runs")
    for rid in run_ids:
        step(f"run {rid}", run_agent, recipe=args.recipe, run_id=rid, cost_cap=args.cost_cap,
             spend_cap=args.spend_cap, only=None)
        step(f"score {rid}", score, recipe=args.recipe, run_id=rid, run_dir=None, out_name=None)
    step("report", report, recipe=args.recipe, runs=run_ids, labels=args.labels, out=None)
    note(f"all done; figures in {kd / 'report.json'}")


# ---------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mine", help="list candidate tasks from merged pull requests (needs GITHUB_TOKEN)")
    p.add_argument("--repo", required=True, help="OWNER/REPO")
    p.add_argument("--since", required=True, help="merged on or after this date (YYYY-MM-DD), the leakage window")
    p.add_argument("--target", type=int, default=60, help="stop after this many candidates (default 60)")
    p.add_argument("--test-dirs", default="tests,test",
                   help="comma-separated names of the repository's test folders (default tests,test)")
    p.add_argument("--out", help="default kit/bench/OWNER__REPO/candidates.jsonl")
    p.add_argument("--refresh", action="store_true", help="ask GitHub again instead of using cached answers")
    p.set_defaults(func=mine)

    rec = dict(help="the repository's recipe, e.g. bench/recipes/pytest-dev__pytest.json")
    p = sub.add_parser("image", help="lock the test dependencies and build the repository's bench image")
    p.add_argument("--recipe", required=True, **rec)
    p.set_defaults(func=image)

    p = sub.add_parser("split", help="split each fix into its test part and its code part (needs the clone)")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--repo-dir", required=True, help="a local clone of the repository")
    p.set_defaults(func=split)

    p = sub.add_parser("verify", help="fail before, pass after, in a clean container, N times")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--only", help="comma-separated task ids")
    p.set_defaults(func=verify)

    p = sub.add_parser("freeze", help="write the surviving tasks into tasks.json with a checksum")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--force", action="store_true", help="overwrite a different frozen set (before any run only)")
    p.set_defaults(func=freeze)

    p = sub.add_parser("run", help="run the harness on every frozen task (needs AI_GATEWAY_API_KEY)")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cost-cap", type=float, help="dollars per task; default from bench/harness.json")
    p.add_argument("--spend-cap", type=float, default=50.0, help="dollars for the whole run folder")
    p.add_argument("--only", help="comma-separated task ids")
    p.add_argument("--tickets", help="hand the agent a frozen version of the tickets (tickets.py), e.g. rewritten; "
                                     "default the issue as filed")
    p.add_argument("--harness", help="harness settings file; default bench/harness.json (e.g. bench/harness.haiku.json)")
    p.add_argument("--context", help="place a frozen context file (context.py freeze), e.g. project-docs, in its own "
                                     "block ahead of the ticket; default none")
    p.add_argument("--label", help="run only the tasks with this fairness label (fairness/labels.jsonl), e.g. fair")
    p.set_defaults(func=run_agent)

    p = sub.add_parser("score", help="score a run with the withheld tests")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--run-id")
    p.add_argument("--run-dir", help="score a folder of submissions instead of a run (used by the tests)")
    p.add_argument("--out-name", help=argparse.SUPPRESS)
    p.set_defaults(func=score)

    p = sub.add_parser("packets", help="write one fairness packet per frozen task, for the screen")
    p.add_argument("--recipe", required=True, **rec)
    p.set_defaults(func=packets)

    p = sub.add_parser("report", help="funnel, runs side by side, cost, and the predictions")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--labels", help="fairness labels, one JSON object per line with task_id and label")
    p.add_argument("--out")
    p.add_argument("--paired", nargs=2, action="append", metavar=("A1,A2", "B1,B2"),
                   help="two groups of run ids, one version of the tickets each, read pair by pair; may repeat")
    p.add_argument("--study", choices=["bench", "tickets", "context"], default="bench",
                   help="which study's predictions to report (default bench, the third post's); context writes "
                        "report-context.md and .json from the runs named at the top of bench.py")
    p.add_argument("--noise", metavar="R1,R2,...", help="identical runs: flips of every pairing, split nets, noise floor")
    p.add_argument("--label", help="read only the tasks with this fairness label (needs --labels), e.g. when runs "
                                   "cover different task sets")
    p.add_argument("--training-cutoff", metavar="YYYY-MM-DD",
                   help="context study: report the tasks merged before this date separately")
    p.set_defaults(func=report)

    p = sub.add_parser("verdict", help="the predictions alone")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--labels", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.set_defaults(func=report)

    p = sub.add_parser("all", help="the whole pipeline: mine (first time), image, split, verify, freeze, packets, "
                                   "the runs, score and report, with a log in the kit folder")
    p.add_argument("--recipe", required=True, **rec)
    p.add_argument("--repo-dir", required=True, help="a local clone of the repository")
    p.add_argument("--since", help="leakage window for mine (YYYY-MM-DD); required the first time, reruns mine when given")
    p.add_argument("--target", type=int, default=60, help="mine: stop after this many candidates (default 60)")
    p.add_argument("--repeats", type=int, default=3, help="verify repeats (default 3)")
    p.add_argument("--runs", type=int, default=2, help="how many identical runs (default 2)")
    p.add_argument("--run-prefix", default="r", help="run ids are PREFIX1, PREFIX2, ... (default r)")
    p.add_argument("--cost-cap", type=float, help="dollars per task; default from bench/harness.json")
    p.add_argument("--spend-cap", type=float, default=25.0, help="dollars per run (default 25)")
    p.add_argument("--labels", help="fairness labels for the report, if the screen has been run")
    p.add_argument("--skip-run", action="store_true", help="stop after freeze and packets (no model calls)")
    p.set_defaults(func=all_steps)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
