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
           every headline figure by name, so a write-up can cite numbers instead of retyping them
  all      the whole pipeline in one command, restartable, with a log of every step

Standard library only. The run step calls the harness's own Python environment (see bench/setup-env.sh).
Written up in the third post of the series "Ask your own repository" on thanoskarpouzis.com.
"""
import argparse
import collections
import hashlib
import json
import os
import re
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


def run_agent(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    manifest = json.loads((kd / "tasks.json").read_text(encoding="utf-8"))
    if canonical_checksum(manifest["tasks"]) != manifest["checksum"]:
        sys.exit("tasks.json does not match its checksum")
    if image_id(recipe["image"]) != manifest["image_id"]:
        sys.exit("the bench image is not the one the tasks were frozen with")
    cfg = json.loads((BENCH / "harness.json").read_text(encoding="utf-8"))
    key = os.environ.get(cfg["model"]["api_key_env"])
    if not key:
        sys.exit(f"set {cfg['model']['api_key_env']} in this shell first")
    if args.cost_cap is not None:
        cfg["agent"]["cost_limit"] = args.cost_cap
    run_dir = kd / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps({"harness": cfg, "image_id": manifest["image_id"],
                                                       "tasks_checksum": manifest["checksum"]}, indent=2))
    tasks = manifest["tasks"]
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
        run(["docker", "rm", "-f", name], check=False)
        with tempfile.TemporaryDirectory(dir=scratch()) as tmp:
            tp = Path(tmp)
            shutil.copyfile(kd / "archives" / f"{t['task_id']}.tar", tp / "checkout.tar")
            (tp / "meta.json").write_text(json.dumps(task_meta(recipe, t)), encoding="utf-8")
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
            (tdir / "result.json").write_text(json.dumps({"task_id": t["task_id"], "exit_status": "no result",
                                                          "cost": None}), encoding="utf-8")
        res = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
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


def report(args):
    recipe = load_recipe(args.recipe)
    kd = kit_dir(recipe["repo"])
    mine_m = json.loads((kd / "mine-manifest.json").read_text(encoding="utf-8"))
    split_m = json.loads((kd / "split-manifest.json").read_text(encoding="utf-8"))
    verify_m = json.loads((kd / "verify-manifest.json").read_text(encoding="utf-8"))
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
    L += ["## Predictions", ""]
    preds = predictions(mine_m, split_m, verify_m, runs)
    L += [f"- {name}: {'HELD' if held else 'FAILED'} ({detail})" for name, held, detail in preds]
    out = Path(args.out) if args.out else kd / "report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    labels = {j["task_id"]: j["label"] for j in read_jsonl(args.labels)} if args.labels else None
    fig = report_figures(recipe["repo"], mine_m, split_m, verify_m, list(args.runs), runs, labels)
    out_json = out.with_suffix(".json")
    out_json.write_text(json.dumps(fig, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(f"{name}: {'HELD' if held else 'FAILED'} ({detail})" for name, held, detail in preds))
    print(f"wrote {out} and {out_json}")


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
