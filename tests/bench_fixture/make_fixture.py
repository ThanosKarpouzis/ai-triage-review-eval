"""Builds a tiny git repository with three fake fixed issues, for tests/test_bench.py.

  base  calc.add subtracts (the bug); test_mul passes
  1     a proper fix: calc.add fixed, test_add added (fails before, passes after)      -> survives
  2     tests that already pass: a docstring in calc.mul, test_mul_zero added         -> "tests did not fail before"
  3     a code change with no test change                                              -> "no test part"
"""
import subprocess
from pathlib import Path

FILES_BASE = {
    "calc/__init__.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
    "tests/test_calc.py": "import calc\n\n\ndef test_mul():\n    assert calc.mul(2, 3) == 6\n",
}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files, msg):
    for path, text in files.items():
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=f@localhost", "commit", "-qm", msg, "--no-gpg-sign")
    return git(repo, "rev-parse", "HEAD")


def build(root):
    repo = Path(root) / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    commit(repo, FILES_BASE, "base")
    good_code = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    tests_1 = FILES_BASE["tests/test_calc.py"] + "\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n"
    s1 = commit(repo, {"calc/__init__.py": good_code, "tests/test_calc.py": tests_1}, "fix add (#1)")
    code_2 = good_code.replace("def mul(a, b):\n", 'def mul(a, b):\n    """Multiply."""\n')
    tests_2 = tests_1 + "\n\ndef test_mul_zero():\n    assert calc.mul(0, 3) == 0\n"
    s2 = commit(repo, {"calc/__init__.py": code_2, "tests/test_calc.py": tests_2}, "docstring (#2)")
    s3 = commit(repo, {"calc/__init__.py": code_2 + "\n\nVERSION = 1\n"}, "version (#3)")
    rows = []
    for n, sha in ((1, s1), (2, s2), (3, s3)):
        rows.append({"task_id": f"fixture__calc-{n}", "repo": "fixture/calc", "pr": 100 + n, "issue": n,
                     "merged_at": "2026-08-01T00:00:00Z", "merge_commit_sha": sha, "issue_title": f"issue {n}",
                     "issue_body": "calc.add(2, 3) returns -1 instead of 5, which breaks totals.",
                     "test_dirs": ["tests"], "files": []})
    return repo, rows
