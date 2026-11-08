"""Tests for tickets.py and the clear ticket study's additions to bench.py (run --tickets, report --paired).
Run with: python3 -m unittest tests/test_tickets.py. None of them needs Docker, the gateway or a key.
The numbers follow the spec's "How the tool is tested", 1 to 5."""
import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench_fixture"))
import bench  # noqa: E402
import make_fixture  # noqa: E402
import tickets  # noqa: E402

REPO = "fixture/calc"


def fixture_tasks():
    """The bench fixture's three tasks, as tasks.json would hold them (the fields tickets.py reads)."""
    with tempfile.TemporaryDirectory() as d:
        _repo, rows = make_fixture.build(d)
    return [{"task_id": r["task_id"], "repo": r["repo"], "issue_title": r["issue_title"],
             "issue_body": r["issue_body"]} for r in rows]


def manifest(tasks):
    return {"repo": REPO, "image_id": "sha256:fixture", "tasks": tasks, "checksum": bench.canonical_checksum(tasks)}


ISSUE = {"task_id": "fixture__calc-9", "issue_title": "`calc.add` returns the difference",
         "issue_body": "Calling `calc.add(2, 3)` returns -1 instead of 5.\n\n<!-- seen with `--strict-totals` -->\n"
                       "The error says \"totals do not match\". Reported for fixture/calc."}

GOOD = """## Summary

`calc.add` returns the difference of its arguments instead of their sum.

## Observed behaviour

Calling `calc.add(2, 3)` returns -1, and the reporter sees "totals do not match".

## Expected behaviour

It returns 5. The reporter's run used `--strict-totals`.

## Steps to reproduce

Not given in the issue

## Acceptance criteria

1. `calc.add(2, 3)` returns 5.

## Out of scope and open questions

None stated in the issue. Does fixture/calc's maintainer's other code depend on this?
"""

LEAKY = GOOD.replace("1. `calc.add(2, 3)` returns 5.", "1. `calc.add(2, 3)` returns 5 after fixing `_subtract_impl` in calc/core.py.")


class Versions(unittest.TestCase):
    """1. Versions from the fixture: title only holds the title and nothing else; the checksum is stable across
    builds of the same input and changes when any ticket changes."""

    def test_title_only(self):
        tasks = fixture_tasks()
        tk = tickets.title_only_tickets(tasks)
        self.assertEqual(sorted(tk), [t["task_id"] for t in tasks])
        for t in tasks:
            self.assertEqual(tk[t["task_id"]], {"issue_title": t["issue_title"], "issue_body": ""})

    def test_checksum(self):
        a = tickets.version_file(tickets.title_only_tickets(fixture_tasks()), "title-only", "x")
        b = tickets.version_file(tickets.title_only_tickets(fixture_tasks()), "title-only", "x")
        self.assertEqual(a["checksum"], b["checksum"])
        changed = json.loads(json.dumps(a["tickets"]))
        changed["fixture__calc-2"]["issue_title"] += "."
        self.assertNotEqual(bench.canonical_checksum(changed), a["checksum"])
        self.assertEqual(a["tasks_checksum"], "x")


class Leaks(unittest.TestCase):
    """2. The leak check."""

    def setUp(self):
        self.text, self.allow = tickets.issue_text(ISSUE), tickets.allow_list(REPO)

    def test_clean_rewrite_passes(self):
        self.assertEqual(tickets.leaks(GOOD, self.text, self.allow), [])

    def test_invented_names_fail(self):
        self.assertEqual(tickets.leaks(LEAKY, self.text, self.allow), ["_subtract_impl", "calc/core.py"])
        self.assertEqual(tickets.leaks("Look at `add_numbers`.", self.text, self.allow), ["add_numbers"])
        self.assertEqual(tickets.leaks("Raise a ValueError.", self.text, self.allow), ["ValueError"])
        self.assertEqual(tickets.leaks("Pass --no-totals to it.", self.text, self.allow), ["--no-totals"])
        self.assertEqual(tickets.leaks('It should say "totals are wrong".', self.text, self.allow),
                         ["totals are wrong"])
        self.assertEqual(tickets.leaks("Use calc.subtract instead.", self.text, self.allow), ["calc.subtract"])
        self.assertEqual(tickets.leaks("```python\nimport calc\nassert calc.add(2, 3) == 6\n```", self.text,
                                       self.allow), ["import", "assert", "==", "6"])

    def test_repository_and_headings_do_not_trip(self):
        self.assertEqual(tickets.leaks("## Out of scope and open questions\n\n## Steps to reproduce\n"
                                       "`pytest-dev/pytest` aside, see `fixture/calc`.", self.text,
                                       tickets.allow_list("pytest-dev/pytest") | self.allow), [])
        self.assertEqual(tickets.leaks("## `secret_name` in a heading", self.text, self.allow), [])
        self.assertEqual(tickets.leaks('The "Acceptance criteria" heading, and "Steps to reproduce".', self.text,
                                       self.allow), [])

    def test_html_comment_counts_as_present(self):
        self.assertEqual(tickets.leaks("Seen with `--strict-totals` and --strict-totals.", self.text, self.allow), [])

    def test_prose_does_not_trip(self):
        self.assertEqual(tickets.leaks("The reporter's run isn't right; it's the users' totals, not a well-known "
                                       "case. Version 3.12 too.", self.text, self.allow), [])

    def test_case_sensitive(self):
        self.assertEqual(tickets.leaks("`Calc.add`", self.text, self.allow), ["Calc.add"])


def fake_model(drafts):
    """A model that answers with the given drafts in turn and records what it was sent."""
    sent = []

    def call(messages):
        sent.append([dict(m) for m in messages])
        return drafts[len(sent) - 1], {"prompt_tokens": 100, "completion_tokens": 50}, 0.01
    call.sent = sent
    return call


class Rewriter(unittest.TestCase):
    """3. The rewriter with a fake model: retry, fallback, counting, freeze, restart and the spend cap."""

    def test_retry_then_clean(self):
        model = fake_model([LEAKY, GOOD])
        e = tickets.rewrite_one(ISSUE, REPO, "RULE", model, tickets.allow_list(REPO))
        self.assertEqual([d["passed"] for d in e["drafts"]], [False, True])
        self.assertEqual(e["drafts"][0]["leaks"], ["_subtract_impl", "calc/core.py"])
        self.assertFalse(e["fallback"])
        self.assertEqual(e["accepted_draft"], 2)
        self.assertEqual(e["final"]["issue_title"],
                         "`calc.add` returns the difference of its arguments instead of their sum.")
        self.assertTrue(e["final"]["issue_body"].startswith("## Observed behaviour"))
        self.assertEqual(e["cost"], 0.02)
        second = model.sent[1]
        self.assertEqual([m["role"] for m in second], ["system", "user", "assistant", "user"])
        self.assertEqual(second[0]["content"], "RULE")
        self.assertIn(ISSUE["issue_body"], second[1]["content"])
        self.assertIn("- _subtract_impl\n- calc/core.py", second[3]["content"])

    def test_two_leaks_fall_back(self):
        e = tickets.rewrite_one(ISSUE, REPO, "RULE", fake_model([LEAKY, LEAKY]), tickets.allow_list(REPO))
        self.assertTrue(e["fallback"])
        self.assertEqual(e["final"], {"issue_title": ISSUE["issue_title"], "issue_body": ISSUE["issue_body"]})

    def test_no_summary_is_refused(self):
        e = tickets.rewrite_one(ISSUE, REPO, "RULE", fake_model(["Here you go: it adds wrong.", GOOD]),
                                tickets.allow_list(REPO))
        self.assertEqual(e["drafts"][0]["leaks"], ["(no Summary heading)"])
        self.assertEqual(e["accepted_draft"], 2)

    def test_log_freeze_and_restart(self):
        tasks = [ISSUE, dict(ISSUE, task_id="fixture__calc-10"), dict(ISSUE, task_id="fixture__calc-11")]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "rewrite-log.jsonl"
            model = fake_model([GOOD, LEAKY, LEAKY, LEAKY, GOOD])
            c = tickets.rewrite_all(tasks, REPO, "RULE", model, log)
            self.assertEqual(c, {"tasks": 3, "first_draft_rejected": 2, "second_draft_rejected": 1, "fallbacks": 1,
                                 "cost_total": 0.05, "seconds_total": c["seconds_total"]})
            again = tickets.rewrite_all(tasks, REPO, "RULE", fake_model([]), log)  # nothing left: no calls
            self.assertEqual(again["tasks"], 3)
            entries = bench.read_jsonl(log)
            doc = tickets.freeze_rewritten(tasks, "tc", entries, "RULE", REPO)
            self.assertEqual(doc["version"], "rewritten")
            self.assertEqual(doc["tasks_checksum"], "tc")
            self.assertEqual(doc["rule_sha256"], tickets.hashlib.sha256(b"RULE").hexdigest())
            self.assertEqual(doc["checksum"], bench.canonical_checksum(doc["tickets"]))
            self.assertEqual(doc["tickets"]["fixture__calc-10"]["issue_body"], ISSUE["issue_body"])  # the fallback
            self.assertEqual(doc["rewrite"]["fallbacks"], 1)
            with self.assertRaises(ValueError):
                tickets.freeze_rewritten(tasks + [dict(ISSUE, task_id="missing")], "tc", entries, "RULE", REPO)
            bad = [dict(entries[0], final={"issue_title": "x", "issue_body": "see `secret_fn`"})] + entries[1:]
            with self.assertRaises(ValueError):
                tickets.freeze_rewritten(tasks, "tc", bad, "RULE", REPO)

    def test_spend_cap(self):
        with tempfile.TemporaryDirectory() as d:
            c = tickets.rewrite_all([ISSUE], REPO, "RULE", fake_model([]), Path(d) / "log.jsonl", spend_cap=0.1)
            self.assertEqual(c["tasks"], 0)

    def test_cost_of(self):
        self.assertEqual(tickets.cost_of({"providerMetadata": {"gateway": {"cost": "0.5"}}, "usage": {"cost": 9}}), 0.5)
        self.assertEqual(tickets.cost_of({"usage": {"cost": 0.25}}), 0.25)
        self.assertAlmostEqual(tickets.cost_of({"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}),
                               (1000 * 2.0 + 100 * 10.0) / 1e6)


class RunWithTickets(unittest.TestCase):
    """4. Run with a version: the version file is checked against the manifest, and its name and checksum go
    into the run's settings.json. Docker is never reached: the spend cap stops the run before the first task."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kd = Path(self.tmp.name)
        self.m = manifest(fixture_tasks())
        (self.kd / "tasks.json").write_text(json.dumps(self.m), encoding="utf-8")
        (self.kd / "tickets").mkdir()
        self.doc = tickets.version_file(tickets.title_only_tickets(self.m["tasks"]), "title-only", self.m["checksum"])
        self.write(self.doc)
        recipe = self.kd / "recipe.json"
        recipe.write_text(json.dumps({"repo": REPO, "image": "fixture:1"}), encoding="utf-8")
        self.args = argparse.Namespace(recipe=str(recipe), run_id="t1", cost_cap=None, spend_cap=0.0, only=None,
                                       tickets="title-only")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, doc):
        (self.kd / "tickets" / f"{doc['version']}.json").write_text(json.dumps(doc), encoding="utf-8")

    def run_agent(self):
        with mock.patch.object(bench, "kit_dir", lambda repo: self.kd), \
             mock.patch.object(bench, "image_id", lambda tag: "sha256:fixture"), \
             mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "test-key"}):
            bench.run_agent(self.args)

    def test_good_version_recorded(self):
        self.assertEqual(bench.load_tickets(self.kd, "title-only", self.m)["checksum"], self.doc["checksum"])
        self.run_agent()
        s = json.loads((self.kd / "runs" / "t1" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(s["tickets"], {"version": "title-only", "checksum": self.doc["checksum"], "rule_sha256": None})
        self.assertNotIn("test-key", json.dumps(s))

    def test_as_filed_recorded(self):
        self.args.tickets = None
        self.run_agent()
        s = json.loads((self.kd / "runs" / "t1" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(s["tickets"], {"version": "as-filed"})

    def test_wrong_task_id_refused(self):
        tk = dict(self.doc["tickets"])
        tk["fixture__calc-99"] = tk.pop("fixture__calc-3")
        self.write(tickets.version_file(tk, "title-only", self.m["checksum"]))
        with self.assertRaisesRegex(ValueError, "one ticket per task"):
            bench.load_tickets(self.kd, "title-only", self.m)
        with self.assertRaises(SystemExit):
            self.run_agent()
        self.assertFalse((self.kd / "runs" / "t1" / "settings.json").exists())

    def test_other_refusals(self):
        self.write(dict(self.doc, tasks_checksum="other"))
        with self.assertRaisesRegex(ValueError, "different task set"):
            bench.load_tickets(self.kd, "title-only", self.m)
        self.write(dict(self.doc, checksum="0" * 64))
        with self.assertRaisesRegex(ValueError, "checksum"):
            bench.load_tickets(self.kd, "title-only", self.m)
        with self.assertRaisesRegex(ValueError, "no version file"):
            bench.load_tickets(self.kd, "rewritten", self.m)


def row(tid, resolved, cost, calls=5, capped=False):
    return {"task_id": tid, "resolved": resolved, "cost": cost, "api_calls": calls, "wall_s": 30.0,
            "capped": capped, "exit_status": "Submitted", "tokens": {"input": 1000, "output": 100, "cache_read": 0}}


class Paired(unittest.TestCase):
    """5. The paired report on hand-written score rows, against figures worked out by hand below."""

    def setUp(self):
        # t1 fair:   A 2/2, B 2/2 -> both 1
        # t2 fair:   A 0/2, B 2/2 -> only B 1
        # t3 fair:   A 1/2, B 2/2 -> both .5, only B .5
        # t4 unfair: A 2/2, B 1/2 -> both .5, only A .5
        # t5 unfair: A 0/2, B 0/2 -> neither 1
        # t6 unfair: A 1/2, B 0/2 -> neither .5, only A .5
        a1 = [row("t1", True, .1), row("t2", False, .3), row("t3", True, .2), row("t4", True, .1),
              row("t5", False, 1.0, capped=True), row("t6", True, .2)]
        a2 = [row("t1", True, .1), row("t2", False, .3), row("t3", False, .2), row("t4", True, .1),
              row("t5", False, .5), row("t6", False, .2)]
        b1 = [row("t1", True, .1), row("t2", True, .1), row("t3", True, .1), row("t4", True, .1),
              row("t5", False, .2), row("t6", False, .2)]
        b2 = [row("t1", True, .1), row("t2", True, .1), row("t3", True, .1), row("t4", False, .1),
              row("t5", False, .2), row("t6", False, .2)]
        self.labels = {"t1": "fair", "t2": "fair", "t3": "fair", "t4": "unfair", "t5": "unfair", "t6": "unfair"}
        self.p = bench.paired_figures(["as-filed", "rewritten"], [[("r1", a1), ("r2", a2)], [("w1", b1), ("w2", b2)]],
                                      self.labels)

    def test_counts(self):
        p = self.p
        self.assertEqual((p["tasks"], p["both"], p["neither"], p["only_a"], p["only_b"], p["net_b_minus_a"]),
                         (6, 2.0, 1.5, 1.0, 1.5, 0.5))
        self.assertEqual((p["fair"]["both"], p["fair"]["only_a"], p["fair"]["only_b"], p["fair"]["net_b_minus_a"]),
                         (1.5, 0.0, 1.5, 1.5))
        self.assertEqual((p["unfair"]["both"], p["unfair"]["neither"], p["unfair"]["only_a"], p["unfair"]["net_b_minus_a"]),
                         (0.5, 1.5, 1.0, -1.0))
        self.assertEqual(p["per_task"][2], {"task_id": "t3", "a": 1, "b": 2, "of": 2, "label": "fair"})

    def test_cost_per_resolved(self):
        r = self.p["runs"]
        # r1: $1.90 over 4 resolved; r2: $1.40 over 2; w1: $0.80 over 4; w2: $0.80 over 3
        self.assertEqual([r[k]["cost_per_resolved"] for k in ("r1", "r2", "w1", "w2")], [0.475, 0.7, 0.2, 0.267])
        self.assertEqual(r["r1"]["capped"], 1)
        self.assertEqual(r["w1"]["resolved_interval_pct"], [pct(x) for x in bench.wilson(4, 6)])

    def test_predictions(self):
        t = bench.paired_figures(["as-filed", "title-only"], [[("r1", []), ("r2", [])], [("t1", []), ("t2", [])]])
        preds = bench.ticket_predictions([self.p], {"first_draft_rejected": 5, "second_draft_rejected": 1,
                                                    "fallbacks": 1})
        self.assertEqual([(n[:2], held) for n, held, _ in preds], [("2.", False), ("3.", True), ("4.", True), ("5.", True)])
        self.assertIsNotNone(t)
        worse = dict(self.p, b="title-only", net_b_minus_a=-5)
        preds = bench.ticket_predictions([worse])
        self.assertEqual([(n[:2], held) for n, held, _ in preds], [("1.", True), ("4.", False)])


def pct(x):
    return round(100 * x)


if __name__ == "__main__":
    unittest.main()
