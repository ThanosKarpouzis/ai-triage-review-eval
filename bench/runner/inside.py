#!/usr/bin/env python3
"""The part of the bench that runs inside a task container, as root, with the base image's Python (standard
library only). bench.py starts one container per step and reads the JSON this writes to /out.

  python3 /runner/inside.py verify   checkout, test part, run the touched tests, code part, run them again
  python3 /runner/inside.py score    checkout, the agent's diff, the withheld tests, run fail-to-pass and pass-to-pass

Inputs in /task: checkout.tar (the parent commit, no history), meta.json, and the patches the step needs.
"""
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

TESTBED = Path("/testbed")
TESTBED_PY = "/opt/testbed/bin/python"
AGENT = "agent"
FAILING = {"failed", "error"}


class Reject(Exception):
    pass


# Absolute path: callers pass minimal environments whose PATH does not include /usr/sbin, where Debian keeps it.
RUNUSER = shutil.which("runuser", path="/usr/sbin:/sbin:/usr/bin:/bin") or "/usr/sbin/runuser"


def sh(cmd, *, cwd=TESTBED, env=None, timeout=600, user=None, check=True, inp=None):
    if user:
        cmd = [RUNUSER, "-u", user, "--"] + cmd
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, text=True, input=inp, errors="surrogateescape",
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        raise Reject("timed out")
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}: {r.stdout[-2000:]}")
    return r


def base_env(meta):
    env = {"PATH": "/opt/testbed/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
    env.update(meta.get("env") or {})
    return env


def prepare(meta, task_dir=Path("/task")):
    """Unpack the parent commit into /testbed, install it as the recipe says, hand it to the agent user, and
    commit it as the base so any later change shows up in `git diff`."""
    with tarfile.open(task_dir / "checkout.tar") as tar:
        tar.extractall(TESTBED, filter="data")
    if meta.get("setup"):
        r = subprocess.run(["bash", "-c", meta["setup"]], cwd=TESTBED, env=base_env(meta), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=(meta.get("timeout_s") or 600))
        if r.returncode:
            raise Reject("environment did not build: " + r.stdout[-500:])
    sh(["chown", "-R", f"{AGENT}:{AGENT}", str(TESTBED)])
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=bench", "-c", "user.email=bench@localhost", "commit", "-qm", "base",
                 "--no-gpg-sign"]):
        sh(cmd, user=AGENT, env={"HOME": "/home/agent", "PATH": "/usr/bin:/bin"})


def apply_patch(path, label):
    """Read as root (the task folder may be readable by root only) and hand the patch to git on stdin."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape") if path.exists() else ""
    if not text.strip():
        return
    r = sh(["git", "apply", "--binary", "--whitespace=nowarn", "-"], user=AGENT, check=False, inp=text,
           env={"HOME": "/home/agent", "PATH": "/usr/bin:/bin"})
    if r.returncode:
        raise Reject(f"{label} did not apply")


def runnable(meta, files):
    keep = []
    for f in files:
        if not f.endswith(".py") or not (TESTBED / f).exists():
            continue
        if any(f.startswith(x) for x in (meta.get("run_exclude") or [])):
            continue
        globs = meta.get("run_globs") or ["**/test_*.py", "**/*_test.py"]
        if any(fnmatch.fnmatch(f, g) or fnmatch.fnmatch(f, g.replace("**/", "")) for g in globs):
            keep.append(f)
    return sorted(keep)


def run_tests(meta, targets):
    """Outcome per node id for the given files or node ids, from the bench_outcomes plugin."""
    d = tempfile.mkdtemp(prefix="bench-")
    sh(["chown", f"{AGENT}:{AGENT}", d])
    out = os.path.join(d, "outcomes.json")
    env = base_env(meta) | {"PYTHONPATH": "/runner", "BENCH_OUTCOMES": out, "HOME": "/home/agent",
                            "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [TESTBED_PY, "-m", "pytest", "-p", "bench_outcomes", *(meta.get("pytest_args") or []), *targets]
    t0 = time.time()
    r = sh(cmd, env=env, timeout=(meta.get("timeout_s") or 600), user=AGENT, check=False)
    try:
        data = json.loads(Path(out).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Reject("test run produced no outcomes: " + r.stdout[-500:])
    return data["outcomes"], {"exitstatus": data["exitstatus"], "seconds": round(time.time() - t0, 1),
                              "tail": r.stdout[-1500:]}


def file_of(nodeid):
    return nodeid.split("::", 1)[0]


def verify(meta, out):
    prepare(meta)
    apply_patch(Path("/task/test.patch"), "test part")
    files = runnable(meta, meta["test_files"])
    if not files:
        raise Reject("no runnable test file")
    before, info_b = run_tests(meta, files)
    failing_tests = {n for n, o in before.items() if o in FAILING and "::" in n}
    errored_files = {n for n, o in before.items() if o == "error" and "::" not in n}
    if not failing_tests and not errored_files:
        raise Reject("tests did not fail before")
    apply_patch(Path("/task/code.patch"), "code part")
    after, info_a = run_tests(meta, files)
    if any(after.get(n) != "passed" for n in failing_tests) or any(f in after for f in errored_files):
        raise Reject("tests did not pass after")
    f2p = sorted(n for n, o in after.items() if o == "passed"
                 and (n in failing_tests or file_of(n) in errored_files))
    p2p = sorted(n for n, o in after.items() if o == "passed" and before.get(n) == "passed")
    if not f2p:
        raise Reject("tests did not pass after")
    return {"status": "ok", "run_files": files, "fail_to_pass": f2p, "pass_to_pass": p2p,
            "before": info_b, "after": info_a}


def score(meta, out):
    prepare(meta)
    sub = Path("/task/submission.diff")
    applied = True
    try:
        apply_patch(sub, "submission")
    except Reject:
        applied = False
    if applied:
        # The agent may have edited or created the files the withheld tests live in; the real test files win.
        env = {"HOME": "/home/agent", "PATH": "/usr/bin:/bin"}
        for f in meta["test_files"]:
            if f in (meta.get("test_files_at_base") or []):
                sh(["git", "checkout", "HEAD", "--", f], user=AGENT, env=env, check=False)
            elif (TESTBED / f).exists():
                (TESTBED / f).unlink()
        apply_patch(Path("/task/test.patch"), "test part")
    if not applied:
        return {"status": "ok", "resolved": False, "reason": "submission did not apply"}
    ids = meta["fail_to_pass"] + meta["pass_to_pass"]
    got, info = run_tests(meta, ids)
    f2p_ok = [n for n in meta["fail_to_pass"] if got.get(n) == "passed"]
    p2p_broken = [n for n in meta["pass_to_pass"] if got.get(n) != "passed"]
    resolved = len(f2p_ok) == len(meta["fail_to_pass"]) and not p2p_broken
    return {"status": "ok", "resolved": resolved, "fail_to_pass_passed": len(f2p_ok),
            "fail_to_pass_total": len(meta["fail_to_pass"]), "pass_to_pass_broken": p2p_broken, "run": info}


def main():
    step = sys.argv[1]
    meta = json.loads(Path("/task/meta.json").read_text(encoding="utf-8"))
    out = Path("/out")
    t0 = time.time()
    try:
        result = {"verify": verify, "score": score}[step](meta, out)
    except Reject as e:
        result = {"status": "rejected", "reason": str(e).split(":")[0], "detail": str(e)[:1500]}
    except Exception:  # anything else is the bench's fault, not the task's; reported, never silent
        import traceback
        result = {"status": "error", "reason": "bench error", "detail": traceback.format_exc()[-1500:]}
    result["seconds"] = round(time.time() - t0, 1)
    (out / f"{step}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "resolved", "seconds")}))


if __name__ == "__main__":
    main()
