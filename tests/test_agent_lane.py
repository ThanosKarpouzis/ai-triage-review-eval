"""Tests for agent_lane.py (spec tests 1, 3 and 4, plus the lane rule, the 30-day logic and the fast fit).
Standard library only, no network.

    python3 -m unittest tests/test_agent_lane.py -v
"""
import json, random, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import agent_lane as al  # noqa: E402
import combine_checks  # noqa: E402

CFG = json.loads((HERE / "examples" / "aidev-agent-lane-config.json").read_text())
BODY = "This is a ticket body that is comfortably longer than forty characters."


def jl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class AidevTable(unittest.TestCase):
    """Spec test 1: twelve pull requests covering every branch of the unit and outcome rules."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        x = cls.tmp / "extracts"
        x.mkdir()
        jl(x / "repository.jsonl", [{"id": 1, "full_name": "o/r1", "language": "Python", "stars": 500},
                                    {"id": 2, "full_name": "o/r2", "language": "Go", "stars": 20000}])

        def pr(i, agent="Copilot", created="2025-03-%02d", merged=True, state="closed", user="Copilot"):
            return {"id": i, "number": i, "agent": agent, "user": user, "repo_id": 1 + i % 2,
                    "created_at": (created % i) + "T10:00:00Z" if "%" in created else created,
                    "merged_at": "2025-04-20T10:00:00Z" if merged else None, "state": state,
                    "html_url": f"https://github.com/o/r/pull/{i}"}
        prs = [pr(1), pr(2, merged=False), pr(3, merged=False, state="open"), pr(4), pr(5), pr(6),
               pr(7), pr(8), pr(9), pr(10, created="2025-07-01T10:00:00Z"),
               pr(11, agent="OpenAI_Codex", user="dev"), pr(12)]
        jl(x / "pull_request.jsonl", prs)
        iss = lambda i, created="2025-02-01T00:00:00Z", body=BODY: {"id": 100 + i, "number": i, "title": f"T{i}",
                                                                    "body": body, "user": "alice", "created_at": created}
        issues = [iss(i) for i in range(1, 12) if i not in (8, 9)] + [iss(8, created="2025-06-01T00:00:00Z"),
                                                                       iss(9, body="too short"), iss(70)]
        jl(x / "issue.jsonl", issues)
        links = [{"pr_id": i, "issue_id": 100 + i, "source": "body"} for i in range(1, 12)]
        links.append({"pr_id": 7, "issue_id": 170, "source": "body"})
        jl(x / "related_issue.jsonl", links)
        C = lambda pr_id, author, msg="Do it", sha=None: {"pr_id": pr_id, "author": author, "message": msg,
                                                        "sha": sha or f"s{pr_id}{author}{len(msg)}"}
        commits = [C(i, "Copilot") for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]
        commits += [C(5, "alice", "Fix the review points"), C(6, "alice", "Merge branch 'main' into copilot/x"),
                    C(6, "autofix-ci[bot]", "apply formatting"), C(11, "dev")]
        jl(x / "pr_commits.jsonl", commits)
        jl(x / "pr_reviews.jsonl", [{"pr_id": 4, "state": "CHANGES_REQUESTED"}, {"pr_id": 1, "state": "APPROVED"}])
        jl(x / "pr_task_type.jsonl", [{"id": i, "type": "feat"} for i in range(1, 13)])
        cls.out = cls.tmp / "table.jsonl"
        al.main(["aidev-table", "--extracts", str(x), "--config", str(HERE / "examples" / "aidev-agent-lane-config.json"),
                 "--out", str(cls.out)])
        cls.rows = {r["pr_id"]: r for r in al.read_jsonl(cls.out)}
        cls.manifest = json.loads((cls.tmp / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_kept_and_dropped(self):
        self.assertEqual(sorted(self.rows), [1, 2, 3, 4, 5, 6, 11])
        d = self.manifest["dropped"]
        self.assertEqual(d["several related issues"], 1)
        self.assertEqual(d["issue not before pull request"], 1)
        self.assertEqual(d["issue body under 40 characters"], 1)
        self.assertEqual(d["pull request created less than 60 days before the cutoff"], 1)
        self.assertEqual(d["no related issue"], 1)

    def test_outcomes(self):
        r = self.rows
        self.assertEqual((r[1]["merged"], r[1]["landed_clean"]), (True, True))
        self.assertEqual((r[2]["merged"], r[2]["landed_clean"]), (False, False))
        self.assertEqual((r[3]["merged"], r[3]["landed_clean"]), (False, False))
        self.assertEqual((r[4]["merged"], r[4]["landed_clean"], r[4]["changes_requested"]), (True, False, True))
        self.assertEqual((r[5]["landed_clean"], r[5]["human_commit"]), (False, True))
        self.assertEqual((r[6]["landed_clean"], r[6]["human_commit"]), (True, False))  # merge and bot are neutral
        self.assertEqual((r[11]["merged"], r[11]["landed_clean"]), (True, None))
        self.assertEqual(r[1]["type_after"], "feature")
        self.assertEqual((r[1]["stars_bucket"], r[2]["stars_bucket"]), ("over 10k", "under 1k"))

    def test_split(self):
        cal = sorted(i for i, r in self.rows.items() if r["split"] == "calibration")
        self.assertEqual(cal, [1, 2, 3, 4])  # round(0.6 * 7) = 4, by creation date


class LaneRule(unittest.TestCase):
    def test_known_lane(self):
        pairs = [(0.9, 1)] * 100 + [(0.3, 1)] * 60 + [(0.3, 0)] * 40
        for b in (0.15, 0.05):
            lane = al.choose_lane(pairs, b)
            self.assertEqual((lane["kept"], lane["wrong"], lane["threshold"]), (100, 0, 0.9))

    def test_no_lane_when_errors_everywhere(self):
        rng = random.Random(1)
        pairs = [(rng.random(), int(rng.random() > 0.3)) for _ in range(300)]
        self.assertIsNone(al.choose_lane(pairs, 0.05))

    def test_fast_fit_matches_combine_checks(self):
        rng = random.Random(2)
        X = [[float(rng.random() < 0.5) for _ in range(3)] for _ in range(60)]
        y = [int(rng.random() < 0.2 + 0.6 * x[0]) for x in X]
        f1, _, _ = combine_checks.fit(X, y)
        f2 = al.fit_grouped(X, y)
        for x in X[:10]:
            self.assertAlmostEqual(f1(x), f2(x), places=9)


def synthetic(tmp, seed=3, n=600):
    """Table and scores where p_land predicts landing strongly, so a lane opens unless outcomes are shuffled."""
    rng = random.Random(seed)
    rows, scores = [], []
    for i in range(n):
        p = rng.randint(0, 100)
        tt = rng.choice(["feature", "fix", "documentation"])
        y = rng.random() < (0.98 if p > 70 else 0.4)
        rows.append({"pr_id": i, "issue_id": i, "agent": rng.choice(["Copilot", "Devin"]), "repo": f"o/r{i % 40}",
                     "created_at": f"2025-{1 + i * 5 // n:02d}-{1 + i % 27:02d}T00:00:{i % 60:02d}Z",
                     "issue_created_at": "2025-01-01T00:00:00Z", "merged": y, "landed_clean": y,
                     "type_after": tt, "split": "calibration" if i < 0.6 * n else "evaluation"})
        scores.append({"issue_id": i, "task_type": tt, "p_land": p,
                       "checks": {k: (k == "acceptance" and p > 70) for k, _ in CFG["checks"]}})
    jl(tmp / "t.jsonl", rows)
    jl(tmp / "s.jsonl", scores)


class LanesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        synthetic(cls.tmp)
        cls.cfg = str(HERE / "examples" / "aidev-agent-lane-config.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_lanes(self, out, *extra):
        al.main(["lanes", "--table", str(self.tmp / "t.jsonl"), "--scores", str(self.tmp / "s.jsonl"),
                 "--config", self.cfg, "--out", str(self.tmp / out), *extra])
        return json.loads((self.tmp / out).read_text())

    def test_real_lane_opens(self):
        L = self.run_lanes("real.json")
        self.assertTrue(L["lanes"]["C@0.15"]["opened"])

    def test_shuffle_opens_no_lane(self):
        """Spec test 3."""
        for s in (0, 1, 2):
            L = self.run_lanes(f"shuf{s}.json", "--shuffle", "--shuffle-seed", str(s))
            for sc in ("B", "C", "D"):
                for b in (0.15, 0.05):
                    self.assertFalse(L["lanes"][f"{sc}@{b}"]["opened"], f"{sc}@{b} opened with seed {s}")

    def test_deterministic(self):
        """Spec test 4."""
        self.run_lanes("d1.json")
        self.run_lanes("d2.json")
        self.assertEqual((self.tmp / "d1.json").read_bytes(), (self.tmp / "d2.json").read_bytes())


class FakeGitHub:
    def __init__(self, commits_by_path):
        self.commits_by_path = commits_by_path

    def get(self, path, params=None):
        if path.endswith("/files"):
            return [{"filename": f, "changes": 5} for f in self.commits_by_path]
        if "/commits" in path:
            since, until = al.ts(params["since"]), al.ts(params["until"])
            return [c for c in self.commits_by_path.get(params["path"], []) if since <= al.ts(c["date"]) <= until]
        return {"merge_commit_sha": "merge1"}


def commit(sha, msg, date):
    return {"sha": sha, "commit": {"message": msg}, "date": date}


class HeldCheck(unittest.TestCase):
    item = {"pr_id": 1, "repo": "o/r", "number": 7, "merged_at": "2025-05-01T00:00:00Z", "commit_shas": ["own1"]}

    def test_fix_inside_window(self):
        gh = FakeGitHub({"a.py": [commit("x", "fix: off by one", "2025-05-10T00:00:00Z")]})
        self.assertFalse(al.held_one(gh, self.item, CFG)["held"])

    def test_fix_after_window(self):
        gh = FakeGitHub({"a.py": [commit("x", "Fix crash", "2025-06-15T00:00:00Z")]})
        self.assertTrue(al.held_one(gh, self.item, CFG)["held"])

    def test_own_commits_ignored(self):
        gh = FakeGitHub({"a.py": [commit("own1", "fix lint", "2025-05-01T00:00:00Z"),
                                  commit("merge1", "Fix thing (#7)", "2025-05-01T00:00:00Z"),
                                  commit("y", "Add docs", "2025-05-03T00:00:00Z")]})
        self.assertTrue(al.held_one(gh, self.item, CFG)["held"])

    def test_reference_to_number_counts(self):
        gh = FakeGitHub({"a.py": [commit("z", "Follow-up to #7: handle empty list", "2025-05-03T00:00:00Z")]})
        self.assertFalse(al.held_one(gh, self.item, CFG)["held"])


if __name__ == "__main__":
    unittest.main()


class ScoreCache(unittest.TestCase):
    """Answers are stored with their raw reply; reruns skip scored tickets; errors are retried only on request."""

    def test_cache_raw_and_retry(self):
        import run
        tmp = Path(tempfile.mkdtemp())
        calls = []
        good = json.dumps({"task_type": "fix", "p_land": 40, "checks": {k: "no" for k, _ in CFG["checks"]}})
        replies = iter(["not json", "still not json", good, good])

        def fake_post(path, body, key):
            calls.append(1)
            return {"choices": [{"message": {"content": next(replies)}}], "usage": {"input_tokens": 1000, "output_tokens": 100}}, 1.0
        orig, run.post, run.PAUSE_BETWEEN_CALLS_S = run.post, fake_post, 0
        try:
            items = [{"issue_id": 1, "title": "t", "body": BODY}, {"issue_id": 2, "title": "t", "body": BODY}]
            out = tmp / "s.jsonl"
            al.score_items(CFG, items[:1], out, 1.0, "k")            # two failed attempts -> error record
            al.score_items(CFG, items, out, 1.0, "k")                # issue 1 skipped, issue 2 scored
            self.assertEqual(len(calls), 3)
            done = al.score_items(CFG, items, out, 1.0, "k", retry_errors=True)  # only issue 1 retried
            self.assertEqual(len(calls), 4)
            self.assertEqual(done["1"]["task_type"], "fix")
            self.assertEqual(len(done["1"]["raw"]), 1)
            al.score_items(CFG, items, out, 1.0, "k", retry_errors=True)  # nothing left to do
            self.assertEqual(len(calls), 4)
            al.main(["rescore", "--scores", str(out), "--config", str(HERE / "examples" / "aidev-agent-lane-config.json"),
                     "--out", str(tmp / "r.jsonl")])
            self.assertEqual(len(calls), 4)
        finally:
            run.post = orig
            shutil.rmtree(tmp)
