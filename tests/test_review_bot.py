"""Fixture test for review_bot.py (spec test 1, plus determinism). Standard library and git only.

    python3 -m unittest tests/test_review_bot.py -v

Builds a throwaway repository with known history and checks every outcome the tool reports:
a comment the author acted on and that later only churned, a comment that was ignored and later
fixed by a bug-labelled pull request, placebos that were touched only after the 90-day window.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import review_bot as rb  # noqa: E402


def git(repo, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def write(repo, lines):
    (Path(repo) / "app.py").write_text("\n".join(lines) + "\n")


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        r = cls.repo = cls.tmp / "repo"
        r.mkdir()
        git(r, "init", "-q", "-b", "main")
        base = [f"line {i}" for i in range(1, 31)]
        write(r, base); git(r, "add", "."); git(r, "commit", "-qm", "Start", date="2026-01-01T10:00:00Z")
        cls.A = git(r, "rev-parse", "HEAD")
        # pull request #1: adds lines 31-60, then the author changes 35-36 after review
        git(r, "checkout", "-qb", "pr1")
        b1 = base + [f"new {i}" for i in range(31, 61)]
        write(r, b1); git(r, "commit", "-qam", "Add things", date="2026-02-20T10:00:00Z")
        cls.B1 = git(r, "rev-parse", "HEAD")
        b2 = list(b1); b2[34] = "new 35 changed"; b2[35] = "new 36 changed"
        write(r, b2); git(r, "commit", "-qam", "Address review", date="2026-02-25T10:00:00Z")
        git(r, "checkout", "-q", "main")
        write(r, b2); git(r, "commit", "-qam", "Add things (#1)", date="2026-03-01T10:00:00Z")
        cls.M = git(r, "rev-parse", "HEAD")
        # 10 March: bug-labelled #2 fixes line 45 (the ignored comment's region)
        f = list(b2); f[44] = "new 45 fixed"
        write(r, f); git(r, "commit", "-qam", "Correct the thing (#2)", date="2026-03-10T10:00:00Z")
        # 20 March: unlabelled #4 touches line 36 (the acted-on comment's region): churn, not a fix
        h = list(f); h[35] = "new 36 tidied"
        write(r, h); git(r, "commit", "-qam", "Tidy up (#4)", date="2026-03-20T10:00:00Z")
        # 15 July, past the 90-day window: bug-labelled #5 rewrites lines 50-60, where the placebos sit
        g = h[:49] + [f"late {i}" for i in range(50, 61)]
        write(r, g); git(r, "commit", "-qam", "Late fix (#5)", date="2026-07-15T10:00:00Z")

        out = cls.tmp / "out"
        cls.cfg = cls.tmp / "config.json"
        cls.cfg.write_text(json.dumps({
            "repo": "fixture/repo", "public_name": "fixture", "out_dir": "out",
            "window": ["2026-01-01", "2026-06-30"], "ai_reviewers": ["bot[bot]"], "excluded_bots": [],
            "bug_labels_pr": ["bug"], "bug_labels_issue": ["bug"], "capability_labels": ["feature"],
            "title_rules": {"bugfix": ["fix"], "capability": ["feat"]},
            "area_labels": {}, "path_rules": [{"match": ".py", "area": "backend"}]}))
        rv = out / "review"; rv.mkdir(parents=True)
        pr = {"number": 1, "title": "Add things", "created": "2026-02-20T10:00:00Z", "merged": "2026-03-01T10:00:00Z",
              "author": "dev", "base": cls.A, "head": git(r, "rev-parse", "pr1"), "merge": cls.M,
              "labels": ["feature"], "closing_issues": []}
        comments = [
            {"id": 101, "pr": 1, "kind": "ai", "login": "bot[bot]", "path": "app.py", "start": 35, "end": 36,
             "commit": cls.B1, "created": "2026-02-20T10:05:00Z", "resolved": True, "resolved_by_bot": False,
             "url": "u101", "body": "rename these"},
            {"id": 102, "pr": 1, "kind": "human", "login": "rev", "path": "app.py", "start": 45, "end": 46,
             "commit": cls.B1, "created": "2026-02-21T10:00:00Z", "resolved": False, "resolved_by_bot": False,
             "url": "u102", "body": "this looks wrong"}]
        rb.write_jsonl(rv / "prs.jsonl", [pr])
        rb.write_jsonl(rv / "comments.jsonl", comments)
        rb.write_jsonl(rv / "pr-info.jsonl", [
            {"number": 2, "title": "Correct the thing", "labels": ["bug"], "closing_issues": []},
            {"number": 4, "title": "Tidy up", "labels": [], "closing_issues": []},
            {"number": 5, "title": "Late fix", "labels": ["bug"], "closing_issues": []}])
        rb.cmd_map(Namespace(config=str(cls.cfg), clone=str(r), offline=True))
        rb.cmd_fixes(Namespace(config=str(cls.cfg), clone=str(r), branch="main", offline=True))
        cls.rows = {x["id"]: x for x in rb.read_jsonl(rv / "outcomes.jsonl")}
        cls.rv = rv

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_acted_on(self):
        self.assertTrue(self.rows[101]["acted_on"])
        self.assertFalse(self.rows[102]["acted_on"])

    def test_fixed_later(self):
        self.assertFalse(self.rows[101]["fixed"])
        self.assertTrue(self.rows[101]["churned"])
        self.assertTrue(self.rows[102]["fixed"])
        self.assertFalse(self.rows[102]["churned"])

    def test_placebo(self):
        for x in self.rows.values():
            for k in ("placebo", "placebo2"):
                p = x[k]
                self.assertIsNotNone(p)
                self.assertGreaterEqual(p["start"], 52)  # five lines clear of every comment
                self.assertEqual(p["end"] - p["start"], x["end"] - x["start"])
                self.assertFalse(p["acted_on"])
                self.assertFalse(p["fixed"])  # only touched after the window
                self.assertEqual(p["touched_by"], [])

    def test_classification(self):
        c = rb.load_config(self.cfg)
        pr = rb.read_jsonl(self.rv / "prs.jsonl")[0]
        self.assertEqual(rb.classify_pr(c, pr), ("capability", "label"))
        self.assertEqual(rb.classify_pr(c, dict(pr, labels=[], title="fix(ui): thing")), ("bugfix", "title"))
        self.assertEqual(rb.classify_pr(c, dict(pr, labels=[], title="Fixture tweak")), ("unclassified", "none"))

    def test_report_is_deterministic(self):
        a = Namespace(config=str(self.cfg), null=False)
        rb.cmd_report(a)
        one = (self.rv / "summary.json").read_bytes()
        rb.cmd_report(a)
        self.assertEqual(one, (self.rv / "summary.json").read_bytes())
        s = json.loads(one)
        self.assertEqual(s["cells"]["all"]["human"]["ignored_fix_rate"], 1.0)
        self.assertEqual(s["cells"]["all"]["human"]["placebo_fix_rate"], 0.0)
        self.assertEqual(s["cells"]["all"]["ai"]["acted_on_rate"], 1.0)


class Units(unittest.TestCase):
    def test_map_region(self):
        old = ["a", "b", "c", "d", "e"]
        self.assertEqual(rb.map_region(old, old, 2, 3), (False, (2, 3)))
        self.assertEqual(rb.map_region(old, ["x"] + old, 2, 3), (False, (3, 4)))
        self.assertEqual(rb.map_region(old, ["a", "B", "c", "d", "e"], 2, 3), (True, (2, 3)))
        self.assertEqual(rb.map_region(old, ["a", "d", "e"], 2, 3), (True, None))
        self.assertEqual(rb.map_region(old, None, 2, 3), (True, None))

    def test_track_follows_shifted_lines(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            git(tmp, "init", "-q", "-b", "main")
            lines = [f"l{i}" for i in range(1, 21)]
            write(tmp, lines); git(tmp, "add", "."); git(tmp, "commit", "-qm", "base", date="2026-01-01T00:00:00Z")
            m = git(tmp, "rev-parse", "HEAD")
            write(tmp, ["x", "y", "z"] + lines); git(tmp, "commit", "-qam", "insert above", date="2026-01-02T00:00:00Z")
            above = git(tmp, "rev-parse", "HEAD")
            l2 = ["x", "y", "z"] + lines; l2[3 + 14] = "l15 changed"   # old line 15, now line 18
            write(tmp, l2); git(tmp, "commit", "-qam", "touch", date="2026-01-03T00:00:00Z")
            touch = git(tmp, "rev-parse", "HEAD")
            l3 = list(l2); l3[3 + 4] = "l5 changed"                     # outside the region
            write(tmp, l3); git(tmp, "commit", "-qam", "elsewhere", date="2026-01-04T00:00:00Z")
            repo = rb.Repo(tmp, {}); repo.default = "main"
            until = 1800000000
            self.assertEqual(rb.track(repo, "app.py", (14, 16), m, until), [touch])
            self.assertEqual(rb.track(repo, "app.py", (9, 10), m, until), [])
            self.assertNotIn(above, rb.track(repo, "app.py", (14, 16), m, until))
        finally:
            shutil.rmtree(tmp)

    def test_parse_added(self):
        patch = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,0 +2,2 @@\n+x\n+y\n@@ -5 +7 @@\n-old\n+new\n"
        self.assertEqual(rb.parse_added(patch), {"f": [2, 3, 7]})


if __name__ == "__main__":
    unittest.main()
