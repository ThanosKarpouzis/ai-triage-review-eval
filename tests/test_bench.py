"""Tests for bench.py. Run with: python3 -m unittest tests/test_bench.py
The mining rules run without network. Tests that need Docker are added with the verify, run and score steps
and skip with a message when docker is absent."""
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402

REPO = "acme/widget"


class FakeGitHub:
    def __init__(self, prs, issues, files):
        self.prs, self.issues, self.files = prs, issues, files

    def get(self, path, params=None):
        n = int(path.rsplit("/", 1)[-1])
        if "/pulls/" in path:
            return self.prs[n]
        if n not in self.issues:
            raise urllib.error.HTTPError(path, 404, "not found", None, None)
        return self.issues[n]

    def pages(self, path, params, max_pages):
        return iter(self.files[int(path.split("/")[-2])])


def pr(n, body, created="2026-08-10T00:00:00Z", merged="2026-08-11T00:00:00Z"):
    return {"number": n, "body": body, "created_at": created, "merged_at": merged,
            "merge_commit_sha": f"sha{n}", "base": {"ref": "main"}}


def issue(n, body="A long enough description of the problem, with an example.", created="2026-08-01T00:00:00Z",
          login="someone", is_pr=False):
    d = {"number": n, "title": f"issue {n}", "body": body, "created_at": created,
         "user": {"login": login, "type": "Bot" if login.endswith("[bot]") else "User"}}
    if is_pr:
        d["pull_request"] = {}
    return d


TEST = [{"filename": "pydantic/main.py", "status": "modified", "changes": 3},
        {"filename": "tests/test_main.py", "status": "modified", "changes": 9}]
NO_TEST = [{"filename": "pydantic/main.py", "status": "modified", "changes": 3}]


class Mining(unittest.TestCase):
    def test_closing_refs(self):
        self.assertEqual(bench.closing_refs("Fixes #12", REPO), ["12"])
        self.assertEqual(bench.closing_refs("closes acme/widget#7 and see #9", REPO), ["7"])
        self.assertEqual(bench.closing_refs("Fix https://github.com/acme/widget/issues/31", REPO), ["31"])
        self.assertEqual(bench.closing_refs("fixes other/repo#5", REPO), [])
        self.assertEqual(bench.closing_refs("Fixes #1, fixes #2", REPO), ["1", "2"])
        self.assertEqual(bench.closing_refs("related to #4", REPO), [])
        template = '<!-- Use "fix #123" to link an issue -->\r\nFixes https://github.com/acme/widget/issues/8.'
        self.assertEqual(bench.closing_refs(template, REPO), ["8"])
        self.assertEqual(bench.closing_refs("<!--\nfix #123\n-->\nno reference here", REPO), [])

    def test_test_paths(self):
        for p in ("tests/test_x.py", "pkg/tests/helpers.py", "test/a.py", "src/test_y.py", "a/b_test.py",
                  "conftest.py"):
            self.assertTrue(bench.is_test_path(p), p)
        for p in ("pydantic/main.py", "docs/testing.md", "pkg/contest.py", "latest/x.py"):
            self.assertFalse(bench.is_test_path(p), p)
        self.assertFalse(bench.is_test_path("testing/python/approx.py"))
        self.assertTrue(bench.is_test_path("testing/python/approx.py", ("testing",)))
        self.assertFalse(bench.is_test_path("src/_pytest/python_api.py", ("testing",)))

    def test_candidate_rules(self):
        prs = {1: pr(1, "Fixes #101"), 2: pr(2, "no reference"), 3: pr(3, "fixes #103, closes #104"),
               4: pr(4, "fixes #105"), 5: pr(5, "fixes #106"), 6: pr(6, "fixes #107"),
               7: pr(7, "fixes #108"), 8: pr(8, "fixes #109"), 9: pr(9, "fixes #999"),
               10: pr(10, "fixes #110", merged=None)}
        issues = {101: issue(101), 105: issue(105, is_pr=True), 106: issue(106, created="2026-08-20T00:00:00Z"),
                  107: issue(107, login="renovate[bot]"), 108: issue(108, body="too short"), 109: issue(109)}
        files = {n: TEST for n in prs}
        files[8] = NO_TEST
        gh = FakeGitHub(prs, issues, files)
        want = {1: None, 2: "no closing reference", 3: "several closing references",
                4: "reference is a pull request", 5: "issue not before pull request", 6: "issue filed by a bot",
                7: "issue body too short", 8: "no test file changed", 9: "issue not found", 10: "not merged"}
        for n, reason in want.items():
            row, got = bench.candidate(gh, REPO, n)
            self.assertEqual(got, reason, n)
            if reason is None:
                self.assertEqual(row["task_id"], "acme__widget-1")
                self.assertEqual(row["merge_commit_sha"], "sha1")
                self.assertEqual(row["issue"], 101)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------- freeze (no Docker)

class KeyGuard(unittest.TestCase):
    def test_redact_key(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "sub").mkdir()
            Path(d, "sub", "trajectory.json").write_text('{"api_key": "vck_secret123456"}')
            Path(d, "clean.txt").write_text("nothing here")
            self.assertEqual(bench.redact_key(d, "vck_secret123456"), ["trajectory.json"])
            self.assertNotIn("vck_secret", Path(d, "sub", "trajectory.json").read_text())
            self.assertEqual(bench.redact_key(d, "vck_secret123456"), [])


class Freeze(unittest.TestCase):
    """Spec test 3: the checksum changes with any task field and is the same for the same input."""

    def test_checksum(self):
        tasks = [{"task_id": "a", "fail_to_pass": ["t::x"], "issue_body": "é"}, {"task_id": "b"}]
        same = [{"issue_body": "é", "fail_to_pass": ["t::x"], "task_id": "a"}, {"task_id": "b"}]
        self.assertEqual(bench.canonical_checksum(tasks), bench.canonical_checksum(same))
        changed = [dict(tasks[0], fail_to_pass=["t::y"]), tasks[1]]
        self.assertNotEqual(bench.canonical_checksum(tasks), bench.canonical_checksum(changed))


class Figures(unittest.TestCase):
    """report.json: the figures a write-up cites, computed from manifests and scored runs alone."""

    def test_wilson(self):
        lo, hi = bench.wilson(15, 44)
        self.assertEqual((round(lo * 100), round(hi * 100)), (22, 49))
        self.assertIsNone(bench.wilson(0, 0))
        self.assertEqual(bench.wilson(0, 10)[0], 0.0)

    def test_report_figures(self):
        mine_m = {"merged_since": "2026-07-01", "pull_requests_seen": 100, "kept": 12,
                  "rejected": {"closes no issue": 80, "no test file changed": 8}}
        split_m = {"kept": 10, "rejected": {"backport of another candidate": 2}}
        verify_m = {"repeats": 3, "survivors": 8, "rejected": {"tests did not fail before": 2}, "flaky": [],
                    "bench_errors": 0}
        def row(tid, resolved, cost, calls=5, status="Submitted"):
            return {"task_id": tid, "resolved": resolved, "cost": cost, "api_calls": calls, "wall_s": 30.0,
                    "capped": status == "cost limit", "exit_status": status,
                    "tokens": {"input": 1000, "output": 100, "cache_read": 0}}
        r1 = [row("t1", True, 0.10), row("t2", True, 0.50), row("t3", False, 0.10), row("t4", False, 1.0, status="cost limit")]
        r2 = [row("t1", True, 0.20), row("t2", True, 0.40), row("t3", True, 0.10), row("t4", False, 1.0, status="cost limit")]
        labels = {"t1": "fair", "t2": "fair", "t3": "unfair", "t4": "unfair"}
        fig = bench.report_figures("o/r", mine_m, split_m, verify_m, ["r1", "r2"], [r1, r2], labels)
        f = fig["funnel"]
        self.assertEqual((f["candidates"], f["distinct_issues"], f["survivors"]), (12, 10, 8))
        self.assertEqual(f["candidates_per_pull_request_pct"], 12)
        self.assertEqual(f["by_step"], {"fail-then-pass test check": 2})
        self.assertEqual(fig["runs"]["r1"]["resolved"], 2)
        self.assertEqual(fig["runs"]["r1"]["capped"], 1)
        self.assertEqual(fig["runs"]["r1"]["unresolved_submitted"], 1)
        self.assertEqual(fig["runs"]["r1"]["resolved_cost_max_over_median"], 1.7)
        self.assertEqual(fig["runs"]["r1"]["tokens_input"], 4000)
        a = fig["agreement"]
        self.assertEqual((a["tasks"], a["resolved_in_both"], a["resolved_in_exactly_one"], a["differing_tasks"]), (4, 2, 1, ["t3"]))
        self.assertEqual(a["cost_ratio_max"], 2.0)
        self.assertEqual(fig["fairness"]["fair"]["runs"]["r1"], {"resolved": 2, "of": 2, "pct": 100, "interval_pct": [34, 100]})
        self.assertEqual([p["verdict"] for p in fig["predictions"]], ["failed", "held", "failed", "held", "failed"])
        self.assertEqual(fig["per_task"][2], {"task_id": "t3", "r1": False, "r2": True})


# ---------------------------------------------------------------------------- with Docker and the bench image

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench_fixture"))
import make_fixture  # noqa: E402

IMAGE = os.environ.get("BENCH_TEST_IMAGE", "bench-pytest-dev__pytest:1")


def have_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


@unittest.skipUnless(have_image(), f"needs docker and the image {IMAGE} (bench.py image)")
class WithDocker(unittest.TestCase):
    """Spec tests 1, 2 and 5 on the fixture repository, in the real bench image."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=bench.scratch())
        root = Path(cls.tmp.name)
        cls.repo, rows = make_fixture.build(root)
        cls.kd = root / "kit"
        cls._kit_dir = bench.kit_dir
        bench.kit_dir = lambda repo: cls.kd
        bench.write_jsonl(cls.kd / "candidates.jsonl", rows)
        recipe = {"repo": "fixture/calc", "image": IMAGE, "test_dirs": ["tests"], "setup": "",
                  "env": {}, "pytest_args": ["-p", "no:cacheprovider", "-q"], "timeout_s": 120,
                  "docker": {"cpus": "2", "memory": "2g"}}
        cls.recipe = root / "recipe.json"
        cls.recipe.write_text(json.dumps(recipe), encoding="utf-8")
        bench.main(["split", "--recipe", str(cls.recipe), "--repo-dir", str(cls.repo)])
        bench.main(["verify", "--recipe", str(cls.recipe), "--repeats", "1"])

    @classmethod
    def tearDownClass(cls):
        bench.kit_dir = cls._kit_dir
        cls.tmp.cleanup()

    def test_funnel(self):
        split_m = json.loads((self.kd / "split-manifest.json").read_text())
        self.assertEqual(split_m, {"kept": 2, "rejected": {"no test part": 1}})
        v = json.loads((self.kd / "verify-manifest.json").read_text())
        self.assertEqual(v["survivor_ids"], ["fixture__calc-1"])
        self.assertEqual(v["rejected"], {"tests did not fail before": 1})
        r1 = json.loads((self.kd / "verify" / "fixture__calc-1" / "repeat1" / "verify.json").read_text())
        self.assertEqual(r1["fail_to_pass"], ["tests/test_calc.py::test_add"])
        self.assertEqual(r1["pass_to_pass"], ["tests/test_calc.py::test_mul"])

    def score_diff(self, name, diff):
        """Score one submission for task 1 in its own folder; returns the score JSON."""
        bench.main(["freeze", "--recipe", str(self.recipe), "--force"])
        run_dir = self.kd / "runs" / name
        tdir = run_dir / "fixture__calc-1"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "result.json").write_text(json.dumps({"cost": 0.0}))
        (tdir / "submission.diff").write_text(diff)
        bench.main(["score", "--recipe", str(self.recipe), "--run-dir", str(run_dir)])
        return json.loads((tdir / "score" / "score.json").read_text())

    def test_scoring(self):
        good = (self.kd / "fixes" / "fixture__calc-1.patch").read_text()
        self.assertTrue(self.score_diff("good", good)["resolved"])
        self.assertFalse(self.score_diff("empty", "")["resolved"])
        import difflib
        before = make_fixture.FILES_BASE["calc/__init__.py"]
        after = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return 0\n"
        breaks_mul = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                  "a/calc/__init__.py", "b/calc/__init__.py"))
        res = self.score_diff("breaks_p2p", breaks_mul)
        self.assertFalse(res["resolved"])
        self.assertEqual(res["pass_to_pass_broken"], ["tests/test_calc.py::test_mul"])

    def test_scoring_is_deterministic(self):
        good = (self.kd / "fixes" / "fixture__calc-1.patch").read_text()
        a, b = self.score_diff("det-a", good), self.score_diff("det-b", good)
        self.assertEqual((a["resolved"], a["fail_to_pass_passed"]), (b["resolved"], b["fail_to_pass_passed"]))
