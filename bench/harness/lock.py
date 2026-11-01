#!/usr/bin/env python3
"""Write the harness's hash-checked lock file. Standard library only; needs docker (OrbStack) and network.

1. Pulls the official python:3.12-slim image and records its digest in base-image.txt, so every build uses
   the same base.
2. Asks pip, inside a throwaway container from that image, which packages mini-swe-agent needs under the pins
   in constraints.txt. Nothing is installed: pip only reports what it would install.
3. Checks every package against PyPI: any release uploaded less than MIN_AGE_DAYS ago is replaced by the
   newest release older than that, pinned in constraints.txt, and the resolution runs again.
4. Writes requirements.lock: every package at one version with the sha256 of each of that version's files on
   PyPI, so pip refuses anything else at build time.

  python3 bench/harness/lock.py                       the harness lock (mini-swe-agent and its dependencies)
  python3 bench/harness/lock.py --requirements R --constraints C --out L
                                                      any other lock on the same base, e.g. a repository's
                                                      test dependencies for its bench image
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_TAG = "python:3.12-slim"
MIN_AGE_DAYS = 14
MAX_ROUNDS = 8
AUTO_MARK = "# added by lock.py"


def sh(*cmd, capture=True):
    r = subprocess.run(cmd, text=True, capture_output=capture)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "").strip()


def pypi(path):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{path}/json", timeout=60) as r:
        return json.load(r)


def when(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_final(v):
    return re.fullmatch(r"\d+(\.\d+)*(\.post\d+)?", v) is not None


def base_image():
    """The recorded digest if there is one, so every lock and image shares one base; else pull and record."""
    recorded = HERE / "base-image.txt"
    if recorded.exists() and recorded.read_text(encoding="utf-8").strip():
        return recorded.read_text(encoding="utf-8").strip()
    sh("docker", "pull", "--quiet", BASE_TAG)
    digest = sh("docker", "image", "inspect", BASE_TAG, "--format", "{{index .RepoDigests 0}}")
    (HERE / "base-image.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def resolve(image, requirements, constraints):
    with tempfile.TemporaryDirectory() as tmp:
        sh("docker", "run", "--rm", "-v", f"{tmp}:/out", "-v", f"{requirements.parent}:/r:ro",
           "-v", f"{constraints.parent}:/c:ro", image,
           "pip", "install", "--quiet", "--disable-pip-version-check", "--dry-run", "--ignore-installed",
           "--report", "/out/report.json", "-c", f"/c/{constraints.name}", "-r", f"/r/{requirements.name}")
        report = json.loads(Path(tmp, "report.json").read_text(encoding="utf-8"))
    out = []
    for item in report["install"]:
        meta = item["metadata"]
        out.append({"name": meta["name"], "version": meta["version"],
                    "file": item["download_info"]["url"].rsplit("/", 1)[-1]})
    return out


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v.split(".post")[0]))


def newest_old_enough(name, cutoff, below):
    """The most recently uploaded final release older than the cutoff and lower than the version pip chose,
    so the replacement stays within the range the other packages accept."""
    best = None
    for version, files in pypi(name)["releases"].items():
        if not files or not is_final(version) or any(f.get("yanked") for f in files):
            continue
        if is_final(below) and vkey(version) >= vkey(below):
            continue
        uploaded = max(when(f["upload_time_iso_8601"]) for f in files)
        if uploaded <= cutoff and (best is None or uploaded > best[1]):
            best = (version, uploaded)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requirements", default=str(HERE / "requirements.in"))
    ap.add_argument("--constraints", default=str(HERE / "constraints.txt"))
    ap.add_argument("--out", default=str(HERE / "requirements.lock"))
    args = ap.parse_args()
    requirements, constraints, out = Path(args.requirements).resolve(), Path(args.constraints).resolve(), Path(args.out)
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=MIN_AGE_DAYS)
    image = base_image()
    print(f"base image {image}")
    for rnd in range(1, MAX_ROUNDS + 1):
        pkgs = resolve(image, requirements, constraints)
        too_new = []
        for p in pkgs:
            info = pypi(f"{p['name']}/{p['version']}")
            p["hashes"] = sorted(f["digests"]["sha256"] for f in info["urls"])
            p["uploaded"] = max(when(f["upload_time_iso_8601"]) for f in info["urls"])
            if p["uploaded"] > cutoff:
                too_new.append(p)
        print(f"round {rnd}: {len(pkgs)} packages, {len(too_new)} newer than {MIN_AGE_DAYS} days")
        if not too_new:
            break
        with constraints.open("a", encoding="utf-8") as f:
            for p in too_new:
                old = newest_old_enough(p["name"], cutoff, p["version"])
                if not old:
                    sys.exit(f"{p['name']}: no release older than {MIN_AGE_DAYS} days; pin one by hand")
                print(f"  {p['name']} {p['version']} ({p['uploaded']:%Y-%m-%d}) -> {old[0]} ({old[1]:%Y-%m-%d})")
                f.write(f"{p['name']}=={old[0]}  {AUTO_MARK} {now:%Y-%m-%d}, {p['version']} was too new\n")
    else:
        sys.exit(f"still packages newer than {MIN_AGE_DAYS} days after {MAX_ROUNDS} rounds; see constraints.txt")
    lines = [f"# Lock for {requirements.name}, written by lock.py on {now:%Y-%m-%d} for {image}.",
             f"# Every package is at least {MIN_AGE_DAYS} days old on PyPI; hashes cover every file of each version.",
             "# Install only with: pip install --require-hashes --no-deps -r requirements.lock"]
    for p in sorted(pkgs, key=lambda p: p["name"].lower()):
        lines.append(f"{p['name']}=={p['version']} \\")
        lines += [f"    --hash=sha256:{h} \\" for h in p["hashes"][:-1]]
        lines.append(f"    --hash=sha256:{p['hashes'][-1]}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pins = {p["name"].lower(): p["version"] for p in pkgs}
    named = ", ".join(f"{n} {pins[n]}" for n in ("mini-swe-agent", "litellm", "pytest") if n in pins)
    print(f"wrote {out.name}: {len(pkgs)} packages" + (f"; {named}" if named else ""))


if __name__ == "__main__":
    main()
