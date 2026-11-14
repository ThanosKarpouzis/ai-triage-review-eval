"""Tests for context.py and the context file study's additions to bench.py (run --harness, --context, --label;
report --noise and --study context) and to bench/runner (agent_parts.py). Run with:
python3 -m unittest tests/test_context.py. None of them needs Docker, the gateway or a key, except the class at
the end, which skips without Docker and the bench image. The numbers follow the spec's "How the tool is tested",
1 to 7, then the regression rule: with no new flag nothing changes."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench_fixture"))
import agent_parts  # noqa: E402
import bench  # noqa: E402
import context  # noqa: E402
import make_fixture  # noqa: E402

try:
    import jinja2
except ImportError:  # the harness renders with Jinja; without it here the rendering tests skip
    jinja2 = None

REPO = "fixture/calc"
KEPT = ROOT / "kit" / "bench" / "pytest-dev__pytest"
TRAJECTORY = KEPT / "runs" / "r1" / "pytest-dev__pytest-13370" / "trajectory.json"

# A template shaped like the harness's swebench.yaml instance template, for when the kept trajectory is absent.
TEMPLATE = "<pr_description>\nConsider the following PR description:\n{{task}}\n</pr_description>\n\n<instructions>\n" \
           "Fix it. Do not modify tests.\n</instructions>\n"


def render(template, **kw):
    """As mini-swe-agent renders its templates: Jinja with StrictUndefined."""
    return jinja2.Template(template, undefined=jinja2.StrictUndefined).render(**kw)


def fixture_tasks():
    with tempfile.TemporaryDirectory() as d:
        _repo, rows = make_fixture.build(d)
    return [{"task_id": r["task_id"], "repo": r["repo"], "issue_title": r["issue_title"], "issue_body": r["issue_body"],
             "merged_at": f"2026-08-0{i}T00:00:00Z", "parent": f"p{i}"} for i, r in enumerate(rows, 1)]


def manifest(tasks):
    return {"repo": REPO, "image_id": "sha256:fixture", "tasks": tasks, "checksum": bench.canonical_checksum(tasks)}


def frozen_context(kd, name, text, checksum, leak_passed=True):
    (kd / "context").mkdir(parents=True, exist_ok=True)
    (kd / "context" / f"{name}.md").write_text(text, encoding="utf-8")
    doc = {"name": name, "sha256": context.sha256_text(text), "tasks_checksum": checksum, "lines": 1,
           "tokens_estimate": 10, "check": {"leak_passed": leak_passed, "flagged_programs": ["tox"], "contradictions": []}}
    (kd / "context" / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


class FakeDocker:
    """Stands in for bench.run: records what each container would have been handed, and leaves a result."""

    def __init__(self):
        self.created = []

    def __call__(self, cmd, *, check=True, capture=True, inp=None, timeout=None):
        if cmd[:2] == ["docker", "create"]:
            task = next(a.split(":")[0] for a in cmd if a.endswith(":/task:ro"))
            self.created.append({p.name: p.read_text(encoding="utf-8") for p in Path(task).iterdir()
                                 if p.name != "checkout.tar"})
        if cmd[:2] == ["docker", "cp"]:
            out = Path(cmd[3])
            tid = json.loads(self.created[-1]["meta.json"])["task_id"]
            (out / "result.json").write_text(json.dumps({"task_id": tid, "exit_status": "Submitted", "cost": 0.01,
                                                         "api_calls": 2}))
        return SimpleNamespace(stdout="", stderr="", returncode=0)


class KitCase(unittest.TestCase):
    """A temporary kit folder with the fixture's three tasks, their archives and fairness labels."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kd = Path(self.tmp.name)
        self.m = manifest(fixture_tasks())
        (self.kd / "tasks.json").write_text(json.dumps(self.m), encoding="utf-8")
        (self.kd / "archives").mkdir()
        for t in self.m["tasks"]:
            with tarfile.open(self.kd / "archives" / f"{t['task_id']}.tar", "w"):
                pass
        bench.write_jsonl(self.kd / "fairness" / "labels.jsonl",
                          [{"task_id": "fixture__calc-1", "label": "fair"}, {"task_id": "fixture__calc-2", "label": "unfair"},
                           {"task_id": "fixture__calc-3", "label": "fair"}])
        self.recipe = self.kd / "recipe.json"
        self.recipe.write_text(json.dumps({"repo": REPO, "image": "fixture:1"}), encoding="utf-8")
        self.docker = FakeDocker()

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        a = dict(recipe=str(self.recipe), run_id="c1", cost_cap=None, spend_cap=100.0, only=None, tickets=None,
                 harness=None, context=None, label=None)
        a.update(kw)
        return argparse.Namespace(**a)

    def run_agent(self, args):
        with mock.patch.object(bench, "kit_dir", lambda repo: self.kd), \
             mock.patch.object(bench, "scratch", lambda: self.kd), \
             mock.patch.object(bench, "image_id", lambda tag: "sha256:fixture"), \
             mock.patch.object(bench, "run", self.docker), \
             mock.patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "test-key"}):
            bench.run_agent(args)

    def settings(self, rid="c1"):
        return json.loads((self.kd / "runs" / rid / "settings.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- 1. the context block

class ContextBlock(KitCase):
    """1. run --context writes the file's name and checksum into settings.json, refuses to resume with another
    file, hands the file to the runner beside the task (never in the working tree); the prompt with a file differs
    only by the labelled block, and the file's text is never read as template syntax."""

    def test_recorded_and_handed_over(self):
        doc = frozen_context(self.kd, "project-docs", "Run {{ x }} with `tox -e py`.\n", self.m["checksum"])
        self.run_agent(self.args(context="project-docs"))
        self.assertEqual(self.settings()["context"], {"name": "project-docs", "sha256": doc["sha256"]})
        self.assertEqual(len(self.docker.created), 3)
        handed = self.docker.created[0]
        self.assertEqual(handed["context.md"], "Run {{ x }} with `tox -e py`.\n")
        meta = json.loads(handed["meta.json"])
        self.assertEqual(meta["context"], {"name": "project-docs", "sha256": doc["sha256"], "path": "/task/context.md"})
        self.assertNotIn("test-key", json.dumps(self.settings()))

    def test_no_file_recorded_as_none(self):
        self.run_agent(self.args())
        self.assertEqual(self.settings()["context"], {"name": "none"})
        handed = self.docker.created[0]
        self.assertNotIn("context.md", handed)
        self.assertNotIn("context", json.loads(handed["meta.json"]))

    def test_resume_with_other_file_refused(self):
        frozen_context(self.kd, "project-docs", "a\n", self.m["checksum"])
        frozen_context(self.kd, "generated", "b\n", self.m["checksum"])
        self.run_agent(self.args(context="project-docs", only="fixture__calc-1"))
        for other in ("generated", None):
            with self.assertRaisesRegex(SystemExit, "context file"):
                self.run_agent(self.args(context=other))
        self.run_agent(self.args(context="project-docs"))  # the same file resumes
        self.assertEqual(len(list((self.kd / "runs" / "c1").glob("*/result.json"))), 3)

    def test_refusals(self):
        frozen_context(self.kd, "project-docs", "a\n", "other")
        with self.assertRaisesRegex(ValueError, "different task set"):
            bench.load_context(self.kd, "project-docs", self.m)
        frozen_context(self.kd, "project-docs", "a\n", self.m["checksum"])
        (self.kd / "context" / "project-docs.md").write_text("edited by hand\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bench.load_context(self.kd, "project-docs", self.m)
        frozen_context(self.kd, "project-docs", "a\n", self.m["checksum"], leak_passed=False)
        with self.assertRaisesRegex(ValueError, "leak"):
            bench.load_context(self.kd, "project-docs", self.m)
        with self.assertRaisesRegex(ValueError, "no frozen"):
            bench.load_context(self.kd, "generated", self.m)
        with self.assertRaises(SystemExit):
            self.run_agent(self.args(context="generated"))
        self.assertFalse((self.kd / "runs" / "c1" / "settings.json").exists())

    def test_block_composition(self):
        composed = agent_parts.compose_instance_template(TEMPLATE)
        self.assertTrue(composed.endswith(TEMPLATE))
        self.assertTrue(composed.startswith("<context_file>\nContents of AGENTS.md at the repository root:\n"))
        self.assertLess(composed.index("</context_file>"), composed.index("<pr_description>"))
        with self.assertRaises(ValueError):
            agent_parts.compose_instance_template("{{ context_file_text }}")

    @unittest.skipUnless(jinja2, "needs jinja2 to render as the harness does")
    def test_rendered_prompt(self):
        text = "Run {{ x }} and {% raw %} as written.\n"
        plain = render(TEMPLATE, task="Fix the bug.")
        with_file = render(agent_parts.compose_instance_template(TEMPLATE), task="Fix the bug.",
                           **{agent_parts.CONTEXT_VAR: text})
        block = with_file[: len(with_file) - len(plain)]
        self.assertEqual(with_file, block + plain)  # differs only by the block, placed ahead
        self.assertEqual(block, "<context_file>\nContents of AGENTS.md at the repository root:\n\n" + text
                         + "\n</context_file>\n\n")
        self.assertIn("{{ x }}", with_file)

    @unittest.skipUnless(jinja2 and TRAJECTORY.exists(), "needs jinja2 and a kept trajectory of post 3")
    def test_real_template_byte_for_byte(self):
        """The harness's own template from a kept run renders, with the task alone, to exactly the prompt that
        run's agent was given; with a file, to the block followed by that same prompt."""
        tr = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
        template = tr["info"]["config"]["agent"]["instance_template"]
        sent = next(m["content"] for m in tr["messages"] if m["role"] == "user")
        head, sep, _ = sent.partition("\n")
        task = sent[len("<pr_description>\nConsider the following PR description:\n"):sent.index("\n</pr_description>")]
        self.assertEqual(render(template, task=task), sent)
        composed = render(agent_parts.compose_instance_template(template), task=task,
                          **{agent_parts.CONTEXT_VAR: "{{ x }}\n"})
        self.assertTrue(composed.endswith(sent))
        self.assertEqual(composed[: -len(sent)], "<context_file>\nContents of AGENTS.md at the repository root:\n\n"
                         "{{ x }}\n\n</context_file>\n\n")


# ---------------------------------------------------------------------------- 2. the leak check

class Leaks(unittest.TestCase):
    """2. A file naming a function that appears only in a withheld test fails; one naming only tokens present at
    the base commit passes."""

    BASE = "def add(a, b):\n    return a - b\n\n# run with python -m pytest tests/test_calc.py\n"
    PATCH = ("--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -1 +1,4 @@\n import calc\n"
             "+def test_add_overflow_guard():\n+    assert calc.add(2, 3) == 5\n")

    def test_future_name_fails(self):
        self.assertEqual(context.leak_tokens("Look at `test_add_overflow_guard` first.", self.BASE, [self.PATCH]),
                         ["test_add_overflow_guard"])

    def test_base_tokens_pass(self):
        self.assertEqual(context.leak_tokens("Run `python -m pytest tests/test_calc.py`; `add` lives in calc.",
                                             self.BASE, [self.PATCH]), [])

    def test_invented_name_is_not_a_leak(self):
        self.assertEqual(context.leak_tokens("See `made_up_helper`.", self.BASE, [self.PATCH]), [])

    def test_removed_and_context_lines_do_not_count(self):
        patch = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1 @@\n old_context_name\n-removed_name_here\n"
        self.assertEqual(context.leak_tokens("`old_context_name` and `removed_name_here`", "", [patch]), [])


# ---------------------------------------------------------------------------- 3. commands work

GENERATED = """# AGENTS.md

Run the tests with:

```bash
python -m pytest testing/test_config.py -q
tox -e py313
```

```python
import pytest  # not a command
pytest.main(["-q"])
```

```toml
[tool.pytest]
addopts = "-q"
```

Inline: `pre-commit run --all-files` before committing. Style is checked with ruff.
A single word such as `pytest` names a tool; it does not run one. The `testing/` folder holds the tests.
"""

RST = """Short version
~~~~~~~~~~~~~

#. Enable and install `pre-commit <https://pre-commit.com>`_ to keep the style.
#. Tests are run using ``tox``::

    tox -e linting,py313

#. Submit a pull request with this data::

    head-fork: YOUR_NAME/pytest
    base: main

#. Clone it::

    $ git clone git@github.com:YOU/pytest.git
    # now, make a branch
        $ git checkout -b fix main

.. code-block:: python

    def test_x(pytester):
        pytester.runpytest()
"""


def fake_resolver(present):
    calls = []

    def resolve(requests):
        calls.append(requests)
        return {q["key"]: (q["module"] or q["program"]) in present for q in requests}
    resolve.calls = calls
    return resolve


class CommandsWork(unittest.TestCase):
    """3. With a fake resolver: a file naming a missing program fails and lists it; prose that mentions a tool
    without a command does not trip it; Python-marked blocks and data lines are skipped."""

    def test_markdown(self):
        res = context.check_file(GENERATED, "", [], fake_resolver({"python", "pytest", "git"}))
        c = res["commands"]
        self.assertEqual(sorted(c["programs"]), ["pre-commit", "python", "python -m pytest", "tox"])
        self.assertEqual(c["not_found"], ["pre-commit", "tox"])
        self.assertTrue(c["checked"])
        self.assertEqual({x["source"] for x in c["candidates"]}, {"block", "inline"})
        self.assertNotIn("ruff", c["programs"])  # named in prose, never as a command

    def test_rst(self):
        cands = context.command_candidates(RST)
        self.assertEqual([x["command"] for x in cands], ["tox -e linting,py313", "git clone git@github.com:YOU/pytest.git",
                                                         "git checkout -b fix main"])
        self.assertEqual(cands[0]["line"], 7)
        res = context.check_file(RST, "", [], fake_resolver({"git"}))
        self.assertEqual(res["commands"]["not_found"], ["tox"])

    def test_prose_only(self):
        res = context.check_file("We use pytest and tox; see `tox` and `pre-commit <https://pre-commit.com>`_.\n",
                                 "", [], fake_resolver(set()))
        self.assertEqual(res["commands"]["candidates"], [])
        self.assertEqual(res["commands"]["not_found"], [])

    def test_resolves_without_running(self):
        r = fake_resolver(set())
        context.check_file("```sh\ngit push -u\npip install tox\n```\n", "", [], r)
        self.assertEqual(r.calls, [[{"key": "git", "program": "git", "module": None},
                                    {"key": "pip", "program": "pip", "module": None}]])

    def test_without_resolver(self):
        c = context.check_file(GENERATED, "", [])["commands"]
        self.assertFalse(c["checked"])
        self.assertEqual(set(c["programs"].values()), {None})

    def test_contradictions(self):
        found = context.contradictions("Add a test for the fix.\nWrite a ``changelog`` entry.\nAdd yourself to "
                                       "``AUTHORS``.\nDo not touch pyproject.toml.\nThe tests live in testing/.\n")
        self.assertEqual([(x["category"], x["line"]) for x in found],
                         [("tests", 1), ("changelog", 2), ("AUTHORS", 3), ("configuration", 4)])

    def test_agent_path_matches_runner(self):
        self.assertIn(f'"PATH": "{context.AGENT_PATH}"', (ROOT / "bench" / "runner" / "run_task.py").read_text())


# ---------------------------------------------------------------------------- 4. noise

def score_rows(pattern, ids=None, cost=0.1):
    ids = ids or [f"t{i}" for i in range(1, len(pattern) + 1)]
    return [{"task_id": t, "resolved": c == "1", "cost": cost, "api_calls": 3, "wall_s": 1.0, "capped": False,
             "exit_status": "Submitted", "tokens": {"input": 100, "output": 10, "cache_read": 0}}
            for t, c in zip(ids, pattern)]


# Six tasks, four identical runs. Flips: n1/n2 2 (t3, t4), n1/n3 4 (t3 to t6), n1/n4 2 (t4, t6), n2/n3 2 (t5, t6),
# n2/n4 2 (t3, t6), n3/n4 2 (t3, t5), so F = 14/6. Splits, B half minus A half task by task:
# n3+n4 against n1+n2: t4 +.5, t5 -.5, t6 +1 = +1; n2+n4 against n1+n3: t4 +.5, t5 +.5 = +1;
# n2+n3 against n1+n4: t3 +1, t4 +.5, t5 -.5 = +1.
NONE = {"n1": "100010", "n2": "101110", "n3": "101101", "n4": "100111"}


class Noise(unittest.TestCase):
    """4. report --noise on hand-written score files gives the flips of every pairing, the three split nets and
    the floor, matching figures worked out here; a seeded simulation of identical arms exceeds the floor in
    about one case in twenty."""

    def test_by_hand(self):
        nz = bench.noise_figures([(r, score_rows(p)) for r, p in NONE.items()])
        self.assertEqual([x["flips"] for x in nz["pairings"]], [2, 4, 2, 2, 2, 2])
        self.assertAlmostEqual(nz["F"], 14 / 6)
        self.assertEqual([s["net"] for s in nz["splits"]], [1.0, 1.0, 1.0])
        # two against two: 2 sqrt((F/2)(1/2 + 1/2)) = sqrt(2F); two against four: 2 sqrt((F/2)(3/4)) = sqrt(1.5F)
        self.assertAlmostEqual(nz["floor"]["two_vs_two"]["floor"], round(math.sqrt(2 * 14 / 6), 3))
        self.assertAlmostEqual(nz["floor"]["arm_vs_none"]["floor_exact"], math.sqrt(1.5 * 14 / 6))

    def test_report_noise(self):
        with tempfile.TemporaryDirectory() as d:
            kd = Path(d)
            for name in ("mine-manifest", "split-manifest", "verify-manifest"):
                (kd / f"{name}.json").write_text(json.dumps(MANIFESTS[name]))
            for rid, p in NONE.items():
                bench.write_jsonl(kd / "runs" / rid / "score.jsonl", score_rows(p))
            recipe = kd / "recipe.json"
            recipe.write_text(json.dumps({"repo": REPO}))
            with mock.patch.object(bench, "kit_dir", lambda repo: kd):
                bench.main(["report", "--recipe", str(recipe), "--noise", "n1,n2,n3,n4", "--out", str(kd / "r.md")])
            fig = json.loads((kd / "r.json").read_text())
            self.assertEqual([x["flips"] for x in fig["noise"]["pairings"]], [2, 4, 2, 2, 2, 2])
            self.assertIn("## Noise", (kd / "r.md").read_text())

    def test_floor_formulae(self):
        self.assertAlmostEqual(bench.noise_floor(1, 2, 2), math.sqrt(2))       # the spec: 1.4 tasks at 1 flip
        self.assertAlmostEqual(bench.noise_floor(3, 2, 2), math.sqrt(6))       # 2.4 at 3
        self.assertAlmostEqual(bench.noise_floor(6, 2, 2), math.sqrt(12))      # 3.5 at 6
        # N >= 4 (f/2)(1/a + 1/b) / (x/100)^2 with f = 3/44: 2 against 4 runs, 5 points -> 82 tasks
        self.assertEqual(bench.tasks_needed(3, 44, 2, 4, 5), math.ceil(4 * (3 / 44 / 2) * 0.75 / 0.0025))
        self.assertEqual(bench.repeats_needed(3, 44, 10), math.ceil(4 * (3 / 44) / (44 * 0.01)))
        self.assertTrue(bench.inside_floor(1.0, math.sqrt(4 * (2 / 3) / 2 * 0.75) * 2 - 1e-12))

    def test_simulation(self):
        rng = random.Random(20261114)
        ps = [0.0] * 24 + [1.0] * 8 + [rng.uniform(0.2, 0.8) for _ in range(12)]
        trials, beyond = 1000, 0
        for _ in range(trials):
            runs = [[{"task_id": f"t{i}", "resolved": rng.random() < p, "cost": 0.1, "api_calls": 1}
                     for i, p in enumerate(ps)] for _ in range(6)]
            none = [(f"n{k}", runs[k]) for k in range(4)]
            nz = bench.noise_figures(none)
            net = bench.paired_figures(["none", "arm"], [none, [("d1", runs[4]), ("d2", runs[5])]])["net_b_minus_a"]
            beyond += not bench.inside_floor(net, nz["floor"]["arm_vs_none"]["floor_exact"])
        self.assertTrue(0.025 <= beyond / trials <= 0.085, beyond / trials)


MANIFESTS = {"mine-manifest": {"merged_since": "2026-07-01", "pull_requests_seen": 10, "kept": 6, "rejected": {}},
             "split-manifest": {"kept": 6, "rejected": {}},
             "verify-manifest": {"repeats": 3, "survivors": 6, "rejected": {}, "flaky": [], "bench_errors": []}}


# ---------------------------------------------------------------------------- 5. cost split and behaviour

class CostAndBehaviour(unittest.TestCase):
    """5. On fixture runs, the file's own token cost, the remainder and the per-task ratio match by hand; the
    test-runner count does not count "pytest" in prose, paths or the repository name."""

    MODEL = {"price_per_mtok": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
             "long_prompt": {"above_prompt_tokens": 100000,
                             "price_per_mtok": {"input": 0.5, "output": 2.5, "cache_read": 0.05, "cache_write": 0.625}}}

    def test_file_cost(self):
        u = [{"input": 9000, "output": 100, "cache_read": 0, "cache_write": 0},
             {"input": 120000, "output": 100, "cache_read": 0, "cache_write": 0}]
        # 2,000 file tokens on a short call at $0.10 and a long one at $0.50 per million: $0.0002 + $0.001
        self.assertAlmostEqual(bench.file_cost_of(u, 2, 2000, self.MODEL), 0.0012)
        self.assertAlmostEqual(bench.file_cost_of([], 3, 2000, self.MODEL), 0.0006)

    def test_cost_split(self):
        none = [("n1", score_rows("10", ["a", "b"], 0.10)), ("n2", score_rows("10", ["a", "b"], 0.30))]
        arm = [("d1", score_rows("10", ["a", "b"], 0.30)), ("d2", score_rows("10", ["a", "b"], 0.50))]
        none[1][1][1]["cost"] = 0.10  # task b: none 0.10 and 0.10, arm 0.30 and 0.50
        fc = {"d1": {"a": 0.02, "b": 0.04}, "d2": {"a": 0.04, "b": 0.04}}
        c = bench.cost_figures(none, arm, ["a", "b"], fc)
        # a: none mean 0.20, arm 0.40, ratio 2; b: none 0.10, arm 0.40, ratio 4; median 3
        self.assertEqual(c["ratio_median"], 3.0)
        self.assertEqual(c["extra_total"], 0.5)
        self.assertEqual(c["file_own_total"], 0.07)
        self.assertEqual(c["remainder_total"], 0.43)

    def test_split_ratios(self):
        none = [(f"n{i}", score_rows("1", ["a"], c)) for i, c in enumerate((0.1, 0.1, 0.2, 0.2), 1)]
        splits, largest = bench.split_cost_ratios(none, ["a"])
        self.assertEqual([s["ratio_median"] for s in splits], [2.0, 1.0, 1.0])
        self.assertEqual(largest, 2.0)
        none[2:] = [("n3", score_rows("1", ["a"], 0.05)), ("n4", score_rows("1", ["a"], 0.05))]
        self.assertEqual(bench.split_cost_ratios(none, ["a"])[1], 2.0)  # 0.5 read the other way

    def test_test_runner_count(self):
        yes = ["python -m pytest testing/test_x.py -q", "cd /testbed && pytest -x", "py.test -k foo",
               "tox -e py313", "timeout 60 python3 -m pytest", "PYTHONPATH=src /opt/testbed/bin/pytest -q 2>&1 | tail",
               "bash -c 'python -m pytest -q'", "git stash && python -m pytest x; git stash pop",
               "cat > t.py <<'EOF'\nimport pytest\nEOF\npython -m pytest t.py"]
        no = ["cat testing/test_pytest.py", "grep -rn pytest src/_pytest | head", "ls /testbed/src/_pytest",
              "echo 'run pytest later'", "cd pytest-dev__pytest", "python -c 'import pytest; print(pytest.__file__)'",
              "# pytest is the runner", "sed -n 1,20p tox.ini", "which pytest", "pip show pytest",
              "cat > t.py <<'EOF'\nimport pytest\npytest.main(['-q'])\nEOF"]
        for c in yes:
            self.assertTrue(bench.is_test_runner(c), c)
        for c in no:
            self.assertFalse(bench.is_test_runner(c), c)

    def test_behaviour(self):
        cmds = {"n1": {"a": ["python -m pytest", "ls"], "b": []}, "n2": {"a": ["pytest"], "b": ["cat pytest.ini"]},
                "d1": {"a": ["pytest", "pytest -x", "tox -e py"], "b": ["python -m pytest"]},
                "d2": {"a": ["pytest", "pytest"], "b": ["pytest", "tox"]}}
        b = bench.behaviour_figures(["n1", "n2"], ["d1", "d2"], cmds, ["a", "b"], ["tox", "pre-commit"])
        # a: none 1, arm 2.5, +1.5; b: none 0, arm 1.5, +1.5
        self.assertEqual(b["diff_median"], 1.5)
        self.assertEqual(b["flagged_calls"]["d1"], {"commands": 1, "by_program": {"tox": 1}})
        self.assertEqual(b["flagged_calls"]["n1"]["commands"], 0)
        self.assertEqual(bench.calls_flagged("python3 -m tox -e py", ["tox"]), ["tox"])
        self.assertEqual(bench.calls_flagged("python -m venv .venv", ["python3 -m venv"]), ["venv"])

    def test_commands_from_trajectory(self):
        tr = {"messages": [{"role": "system", "content": "x"},
                           {"role": "assistant", "extra": {"usage": {"input": 5, "output": 1, "cache_read": 0, "cache_write": 0}},
                            "tool_calls": [{"function": {"arguments": json.dumps({"command": "pytest -q"}), "name": "bash"}},
                                           {"function": {"arguments": "not json"}}]},
                           {"role": "tool", "content": "ok"}]}
        self.assertEqual(bench.agent_commands(tr), ["pytest -q"])
        self.assertEqual(bench.call_usages(tr), [{"input": 5, "output": 1, "cache_read": 0, "cache_write": 0}])


# ---------------------------------------------------------------------------- 6. price tiers and the task filter

class PricesAndFilter(KitCase):
    """6. A call over the prompt-length threshold is costed at the higher rates and one under it at the lower;
    run --label fair runs only those tasks and records the filter; the report refuses to pair runs over
    different task sets unless told which subset to read, and runs whose settings name different models."""

    def test_tiers(self):
        cfg = json.loads((ROOT / "bench" / "harness.haiku.json").read_text())["model"]
        short = {"input": 99000, "output": 2000, "cache_read": 500, "cache_write": 500}     # prompt 100,000
        self.assertAlmostEqual(agent_parts.call_cost(short, cfg),
                               (99000 * 0.10 + 2000 * 0.50 + 500 * 0.01 + 500 * 0.125) / 1e6)
        long = {"input": 99000, "output": 2000, "cache_read": 1000, "cache_write": 1}        # prompt 100,001
        self.assertAlmostEqual(agent_parts.call_cost(long, cfg),
                               (99000 * 0.50 + 2000 * 2.50 + 1000 * 0.05 + 1 * 0.625) / 1e6)

    def test_flat_unchanged(self):
        cfg = json.loads((ROOT / "bench" / "harness.json").read_text())["model"]
        u = {"input": 150000, "output": 3000, "cache_read": 0, "cache_write": 0}
        price = {k: v / 1e6 for k, v in cfg["price_per_mtok"].items()}  # run_task.py's sum before the tiers
        self.assertEqual(agent_parts.call_cost(u, cfg), sum(u[k] * price[k] for k in u))

    def test_haiku_settings_differ_in_three_things(self):
        a = json.loads((ROOT / "bench" / "harness.json").read_text())
        b = json.loads((ROOT / "bench" / "harness.haiku.json").read_text())
        for d in (a, b):
            d.pop("_notes")
        self.assertEqual(b["agent"]["cost_limit"], 0.05)
        self.assertEqual(b["model"]["model_name"], "openai/anthropic/claude-haiku-5-5")
        self.assertEqual(b["model"]["long_prompt"]["above_prompt_tokens"], 100000)
        b["agent"]["cost_limit"] = a["agent"]["cost_limit"]
        b["model"]["model_name"] = a["model"]["model_name"]
        b["model"]["price_per_mtok"] = a["model"]["price_per_mtok"]
        del b["model"]["long_prompt"]
        self.assertEqual(a, b)

    def test_run_harness_and_label(self):
        self.run_agent(self.args(harness=str(ROOT / "bench" / "harness.haiku.json"), label="fair"))
        s = self.settings()
        self.assertEqual(s["label"], "fair")
        self.assertEqual(s["harness"]["model"]["model_name"], "openai/anthropic/claude-haiku-5-5")
        self.assertEqual(sorted(json.loads(c["meta.json"])["task_id"] for c in self.docker.created),
                         ["fixture__calc-1", "fixture__calc-3"])
        self.assertEqual(json.loads(self.docker.created[0]["harness.json"])["agent"]["cost_limit"], 0.05)
        with self.assertRaisesRegex(SystemExit, "model"):
            self.run_agent(self.args(label="fair"))  # resuming with the default settings names another model

    def test_pairing_guards(self):
        sonnet = {"harness": {"model": {"model_name": "sonnet"}}}
        haiku = {"harness": {"model": {"model_name": "haiku"}}}
        all3, fair = {"a", "b", "c"}, {"a", "c"}
        self.assertIsNone(bench.pairing_problem({"r1": sonnet, "r2": sonnet}, {"r1": all3, "r2": all3}))
        self.assertRegex(bench.pairing_problem({"r1": sonnet, "n1": haiku}, {"r1": all3, "n1": all3}), "different models")
        self.assertRegex(bench.pairing_problem({"r1": sonnet, "v1": sonnet}, {"r1": all3, "v1": fair}), "task sets")
        self.assertIsNone(bench.pairing_problem({"r1": sonnet, "v1": sonnet}, {"r1": all3, "v1": fair}, subset=fair))
        self.assertRegex(bench.pairing_problem({"r1": sonnet, "v1": sonnet}, {"r1": all3, "v1": {"a"}}, subset=fair),
                         "lack")

    def test_report_refuses_then_reads_subset(self):
        for name in ("mine-manifest", "split-manifest", "verify-manifest"):
            (self.kd / f"{name}.json").write_text(json.dumps(MANIFESTS[name]))
        ids = [t["task_id"] for t in self.m["tasks"]]
        s = {"harness": {"model": {"model_name": "sonnet"}}}
        for rid, rows in (("r1", score_rows("110", ids)), ("r2", score_rows("101", ids)),
                          ("v1", score_rows("1", ids[:1])), ("v2", score_rows("0", ids[:1]))):
            bench.write_jsonl(self.kd / "runs" / rid / "score.jsonl", rows)
            (self.kd / "runs" / rid / "settings.json").write_text(json.dumps(s))
        base = ["report", "--recipe", str(self.recipe), "--paired", "r1,r2", "v1,v2", "--out", str(self.kd / "p.md"),
                "--labels", str(self.kd / "fairness" / "labels.jsonl")]
        with mock.patch.object(bench, "kit_dir", lambda repo: self.kd):
            with self.assertRaisesRegex(SystemExit, "task sets"):
                bench.main(base)
            with self.assertRaisesRegex(SystemExit, "lack"):
                bench.main(base + ["--label", "fair"])  # v1 and v2 hold only one of the two fair tasks
            bench.write_jsonl(self.kd / "runs" / "v1" / "score.jsonl", score_rows("11", [ids[0], ids[2]]))
            bench.write_jsonl(self.kd / "runs" / "v2" / "score.jsonl", score_rows("01", [ids[0], ids[2]]))
            bench.main(base + ["--label", "fair"])
        p = json.loads((self.kd / "p.json").read_text())["paired"][0]
        self.assertEqual(p["tasks"], 2)


# ---------------------------------------------------------------------------- 7. the generator without a model

def fake_session(texts):
    sent = []

    def session(prompt):
        sent.append(prompt)
        return {"text": texts[len(sent) - 1], "cost": 0.02, "api_calls": 7, "exit_status": "Submitted"}
    session.sent = sent
    return session


class Generator(KitCase):
    """7. context.py generate takes a fake session, so the checks, the one regeneration and the freeze run with
    no spend."""

    def setUp(self):
        super().setUp()
        base = "def add(a, b):\n    return a - b\n"
        with tarfile.open(self.kd / "archives" / "fixture__calc-1.tar", "w") as tar:
            data = base.encode()
            info = tarfile.TarInfo("calc/__init__.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for d in ("withheld", "fixes"):
            (self.kd / d).mkdir()
            for t in self.m["tasks"]:
                (self.kd / d / f"{t['task_id']}.patch").write_text(
                    "+++ b/x.py\n+def secret_future_helper():\n" if d == "fixes" and t["task_id"] == "fixture__calc-2" else "")

    def ctx(self, *argv, session=None, resolver=None):
        """One context.py subcommand on the temporary kit, with the fake session or resolver."""
        a = self._parser().parse_args(list(argv))
        with mock.patch.object(bench, "kit_dir", lambda repo: self.kd), contextlib.redirect_stdout(io.StringIO()):
            if a.cmd == "generate":
                return context.cmd_generate(a, session=session)
            if a.cmd == "check":
                return context.cmd_check(a, resolver=resolver)
            return a.func(a)

    def _parser(self):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        for name in ("generate", "check", "freeze"):
            s = sub.add_parser(name)
            s.add_argument("--recipe", default=str(self.recipe))
            s.add_argument("--task")
            s.add_argument("--harness")
            s.add_argument("--retry", action="store_true")
            s.add_argument("--programs")
            s.add_argument("--name")
            s.add_argument("--skip-commands", action="store_true")
            s.add_argument("--confirmed")
            s.set_defaults(func={"generate": context.cmd_generate, "check": context.cmd_check,
                                 "freeze": context.cmd_freeze}[name])
        return p

    def test_generate_check_retry_freeze(self):
        first = "# AGENTS.md\n\n```sh\ntox -e py\npython -m pytest -q\n```\n\nAdd a test for every fix.\n"
        second = "# AGENTS.md\n\n```sh\npython -m pytest -q\n```\n"
        s = fake_session([first, second])
        self.ctx("generate", session=s)
        cd = self.kd / "context"
        self.assertEqual((cd / "generated.md").read_text(), first)
        self.assertEqual(s.sent[0], context.INIT_PROMPT.read_text())
        self.assertEqual(json.loads((cd / "generated.source.json").read_text())["task_id"], "fixture__calc-1")
        with self.assertRaisesRegex(SystemExit, "first attempt exists"):
            self.ctx("generate", session=s)
        with self.assertRaisesRegex(SystemExit, "check"):
            self.ctx("generate", "--retry", session=s)
        resolver = fake_resolver({"python", "pytest"})
        with self.assertRaises(SystemExit) as e:
            self.ctx("check", "--name", "generated", resolver=resolver)
        self.assertEqual(e.exception.code, 0)
        chk = json.loads((cd / "generated.check.json").read_text())
        self.assertEqual(chk["commands"]["not_found"], ["tox"])
        self.assertEqual([x["category"] for x in chk["contradictions"]], ["tests"])
        self.ctx("generate", "--retry", session=s)
        self.assertIn("- tox\n", s.sent[1])
        self.assertTrue(s.sent[1].startswith(context.INIT_PROMPT.read_text().rstrip("\n")))
        self.assertEqual((cd / "generated.md").read_text(), second)
        with self.assertRaisesRegex(SystemExit, "no third"):
            self.ctx("generate", "--retry", session=s)
        with self.assertRaisesRegex(SystemExit, "other text"):
            self.ctx("freeze", "--name", "generated")  # the check was of the first file
        with self.assertRaises(SystemExit):
            self.ctx("check", "--name", "generated", resolver=resolver)
        self.ctx("freeze", "--name", "generated", "--confirmed", "")
        doc = json.loads((cd / "generated.json").read_text())
        self.assertEqual(doc["sha256"], context.sha256_text(second))
        self.assertEqual(doc["tasks_checksum"], self.m["checksum"])
        self.assertEqual(doc["check"]["flagged_programs"], [])
        self.assertEqual(doc["source"]["attempt"], 2)
        self.assertEqual(len(bench.read_jsonl(cd / "generate-log.jsonl")), 2)
        self.assertEqual(bench.load_context(self.kd, "generated", self.m)["text"], second)
        with self.assertRaisesRegex(SystemExit, "frozen"):
            self.ctx("generate", "--retry", session=s)

    def test_leak_refused_at_freeze(self):
        self.ctx("generate", session=fake_session(["Call `secret_future_helper` first.\n"]))
        with self.assertRaises(SystemExit) as e:
            self.ctx("check", "--name", "generated", resolver=fake_resolver(set()))
        self.assertEqual(e.exception.code, 1)
        with self.assertRaisesRegex(SystemExit, "leaks"):
            self.ctx("freeze", "--name", "generated")
        self.assertFalse((self.kd / "context" / "generated.json").exists())

    def test_unresolved_check_refused_at_freeze(self):
        self.ctx("generate", session=fake_session(["Run `tox -e py` to test.\n"]))
        with self.assertRaises(SystemExit):
            self.ctx("check", "--name", "generated", "--skip-commands")
        with self.assertRaisesRegex(SystemExit, "not resolved"):
            self.ctx("freeze", "--name", "generated")

    def test_no_file_left(self):
        with self.assertRaisesRegex(SystemExit, "no AGENTS.md"):
            self.ctx("generate", session=fake_session([None]))
        self.assertFalse((self.kd / "context" / "generated.md").exists())

    def test_retry_prompt(self):
        self.assertEqual(context.generate_prompt("P\n"), "P\n")
        self.assertIn("- tox\n- pre-commit\n", context.generate_prompt("P\n", ["tox", "pre-commit"]))


# ---------------------------------------------------------------------------- docs

class Docs(unittest.TestCase):
    TEXT = ("========\nContributing\n========\n\nIntro.\n\n.. _prs:\n\nPreparing Pull Requests\n"
            "-----------------------\n\nShort version\n~~~~~~~~~~~~~\n\n#. Do it.\n\nWriting Tests\n~~~~~~~~~~~~~\n\n"
            "Write them.\n\n.. _next:\n\nJoining the Development Team\n----------------------------\n\nWelcome.\n")

    def test_section(self):
        text, first, last = context.rst_section(self.TEXT, "Preparing Pull Requests")
        self.assertTrue(text.startswith("Preparing Pull Requests\n---"))
        self.assertTrue(text.endswith("Write them.\n"))
        self.assertIn("Writing Tests", text)
        self.assertEqual((first, last), (9, 20))
        with self.assertRaises(KeyError):
            context.rst_section(self.TEXT, "Nothing")

    @unittest.skipUnless((KEPT / "archives" / "pytest-dev__pytest-13370.tar").exists(), "needs the bench's archives")
    def test_real_section(self):
        text = context.read_from_archive(KEPT / "archives" / "pytest-dev__pytest-13370.tar", "CONTRIBUTING.rst")
        section, first, last = context.rst_section(text, "Preparing Pull Requests")
        self.assertEqual(len(section.splitlines()), 180)
        self.assertIn("Writing Tests", section)
        self.assertNotIn("Joining the Development Team", section)
        self.assertEqual(section, "\n".join(text.splitlines()[first - 1:last]) + "\n")  # unedited


# ---------------------------------------------------------------------------- the report's blocks and predictions

def study_data():
    """Eight Haiku runs and a pool on four tasks (two fair), with figures worked out in the test."""
    ids = ["a", "b", "c", "d"]
    none = [(r, score_rows(p, ids, 0.10)) for r, p in (("n1", "1100"), ("n2", "1010"), ("n3", "1100"), ("n4", "1000"))]
    arms = {"project-docs": [(r, score_rows(p, ids, 0.11)) for r, p in (("d1", "1100"), ("d2", "1110"))],
            "generated": [(r, score_rows(p, ids, 0.13)) for r, p in (("g1", "1111"), ("g2", "1110"))]}
    cmds = {rid: {t: ["python -m pytest -q"] * (2 if rid[0] == "g" else 1) for t in ids}
            for rid in ["n1", "n2", "n3", "n4", "d1", "d2", "g1", "g2", "r1", "r2", "vd1", "vg1"]}
    return {"n_tasks": 4, "labels": {"a": "fair", "b": "unfair", "c": "fair", "d": "unfair"},
            "none": none, "arms": arms, "cmds": cmds, "file_cost": None, "flagged": {"project-docs": ["tox"]},
            "pool": {"project-docs": ("vd1", score_rows("10", ["a", "c"], 0.5)),
                     "generated": ("vg1", score_rows("11", ["a", "c"], 1.2))},
            "pool_base": [("r1", score_rows("1000", ids, 1.0)), ("r2", score_rows("1000", ids, 1.0))]}


class StudyReport(unittest.TestCase):
    def test_blocks(self):
        ctx = bench.context_figures(study_data())
        self.assertEqual(ctx["pilot"], {"run": "n1", "resolved": 2, "tasks": 4, "threshold": 10, "passed": False})
        # flips: n1/n2 2, n1/n3 0, n1/n4 1, n2/n3 2, n2/n4 1, n3/n4 1 -> F = 7/6
        self.assertAlmostEqual(ctx["noise"]["F"], 7 / 6)
        self.assertEqual(ctx["noise"]["arm_pairs"]["generated"]["flips"], 1)
        # none per task: a 1, b .5, c .25, d 0; docs: a 1, b 1, c .5, d 0 -> net .5 + .25 = .75
        self.assertEqual(ctx["paired"]["project-docs"]["net_b_minus_a"], 0.75)
        self.assertEqual(ctx["paired"]["generated"]["net_b_minus_a"], 1.75)
        self.assertEqual(ctx["paired"]["project-docs"]["tasks_apart"], 2)
        self.assertEqual(ctx["cost"]["generated"]["ratio_median"], 1.3)
        self.assertEqual(ctx["behaviour"]["generated"]["diff_median"], 1)
        self.assertEqual(ctx["behaviour"]["project-docs"]["diff_median"], 0)
        # d1 resolves 2 and d2 3, against 2, 2, 2 and 1
        self.assertEqual(sorted(ctx["single_run_gaps"]["project-docs"]), [0, 0, 0, 1, 1, 1, 1, 2])
        floor = math.sqrt(1.5 * 7 / 6)
        self.assertAlmostEqual(ctx["detectable"]["floor_tasks"], round(floor, 2))
        verdicts = {p["name"][:2]: p["verdict"] for p in ctx["predictions"]}
        self.assertEqual(verdicts, {"1.": "held", "2.": "failed", "3.": "held", "4.": "failed", "5.": "held"})

    def test_validation(self):
        v = bench.context_figures(study_data())["validation"]
        self.assertEqual(v["tasks"], 2)
        d = v["arms"]["project-docs"]
        # pool: vd1 resolves a, not c; r1 and r2 resolve a only -> net 0; cost 0.5 against 1.0 -> ratio 0.5
        self.assertEqual((d["pool"]["net"], d["pool"]["cost_ratio_median"]), (0, 0.5))
        # Haiku on the same two tasks: cost 0.11 against 0.10, so the ratios sit on opposite sides of 1
        self.assertEqual(d["agree"], {"cost": False, "test_runner": True, "net": True})
        g = v["arms"]["generated"]
        self.assertFalse(g["haiku_inside_floor"])
        self.assertEqual(g["pool"]["net"], 1.0)
        self.assertFalse(g["agree"]["net"])  # outside on Haiku, but the pool's net is not beyond 3
        self.assertTrue(g["agree"]["cost"])

    def test_agreement_rule(self):
        pool = {"x": {"net": -4, "cost_ratio_median": 1.2, "test_runner_diff_median": 0}}
        h = {"x": {"net": -1, "cost_ratio_median": 1.1, "test_runner_diff_median": 0}}
        self.assertEqual(bench.validation_figures(pool, h, {"x": (False, -3)})["x"]["agree"],
                         {"cost": True, "test_runner": True, "net": True})
        self.assertFalse(bench.validation_figures(pool, h, {"x": (False, 3)})["x"]["agree"]["net"])
        self.assertFalse(bench.validation_figures(pool, h, {"x": (True, 0)})["x"]["agree"]["net"])
        pool["x"]["net"] = 3
        self.assertTrue(bench.validation_figures(pool, h, {"x": (True, 0)})["x"]["agree"]["net"])

    def test_prediction_five_on_sign(self):
        ctx = {"noise": {"floor": {"arm_vs_none": {"floor_exact": 1}}}, "paired": {}, "cost": {}, "behaviour": {},
               "single_run_gaps": {"project-docs": [1, 1, 0, 0, 1, 1, 0, -1]}, "cost_splits": {"largest": None}}
        (name, held, _), = bench.context_predictions(ctx)
        self.assertTrue(held)
        ctx["single_run_gaps"]["project-docs"] = [1, 1, 0, 0, 1, 1, 0, 0]
        self.assertFalse(bench.context_predictions(ctx)[0][1])

    def test_cutoff_split(self):
        data = study_data()
        data.update(merged_at={"a": "2026-07-21", "b": "2026-07-22", "c": "2026-08-01", "d": "2026-09-01"},
                    cutoff="2026-08-01")
        p = bench.context_figures(data)["paired"]["project-docs"]
        self.assertEqual((p["before_cutoff"]["tasks"], p["from_cutoff"]["tasks"]), (2, 2))
        self.assertEqual(p["before_cutoff"]["net_b_minus_a"] + p["from_cutoff"]["net_b_minus_a"], 0.75)


class StudyReportFiles(KitCase):
    """report --study context end to end on a temporary kit: it reads the twelve runs and the two frozen files,
    refuses a run made with the wrong file or another model, and writes report-context.md and .json."""

    def make_run(self, rid, pattern, model, ctx=None, ids=None):
        ids = ids or [t["task_id"] for t in self.m["tasks"]]
        rd = self.kd / "runs" / rid
        bench.write_jsonl(rd / "score.jsonl", score_rows(pattern, ids, 0.02))
        s = {"harness": {"model": {"model_name": model, "price_per_mtok": {"input": 0.1, "output": 0.5,
                                                                            "cache_read": 0.01, "cache_write": 0.125}}},
             "tickets": {"version": "as-filed"}, "context": ctx or {"name": "none"}}
        (rd / "settings.json").write_text(json.dumps(s))
        for t in ids:
            (rd / t).mkdir(parents=True, exist_ok=True)
            (rd / t / "trajectory.json").write_text(json.dumps({"messages": [
                {"role": "assistant", "extra": {"usage": {"input": 1000, "output": 10, "cache_read": 0, "cache_write": 0}},
                 "tool_calls": [{"function": {"arguments": json.dumps({"command": "python -m pytest -q"})}}]}]}))

    def test_end_to_end(self):
        docs = frozen_context(self.kd, "project-docs", "docs\n", self.m["checksum"])
        gen = frozen_context(self.kd, "generated", "gen\n", self.m["checksum"])
        fair = ["fixture__calc-1", "fixture__calc-3"]
        for rid, p in (("n1", "110"), ("n2", "100"), ("n3", "110"), ("n4", "010")):
            self.make_run(rid, p, "haiku")
        for rid, p in (("d1", "110"), ("d2", "111")):
            self.make_run(rid, p, "haiku", {"name": "project-docs", "sha256": docs["sha256"]})
        for rid, p in (("g1", "111"), ("g2", "011")):
            self.make_run(rid, p, "haiku", {"name": "generated", "sha256": gen["sha256"]})
        for rid, p in (("r1", "100"), ("r2", "101")):
            self.make_run(rid, p, "sonnet")
        self.make_run("vd1", "11", "sonnet", {"name": "project-docs", "sha256": docs["sha256"]}, fair)
        self.make_run("vg1", "10", "sonnet", {"name": "generated", "sha256": gen["sha256"]}, fair)
        argv = ["report", "--recipe", str(self.recipe), "--study", "context",
                "--labels", str(self.kd / "fairness" / "labels.jsonl")]
        with mock.patch.object(bench, "kit_dir", lambda repo: self.kd), contextlib.redirect_stdout(io.StringIO()):
            bench.main(argv)
            fig = json.loads((self.kd / "report-context.json").read_text())
            self.assertEqual(fig["study"], "context")
            self.assertEqual(len(fig["predictions"]), 5)
            self.assertEqual(fig["validation"]["tasks"], 2)
            self.assertEqual(fig["files"]["project-docs"]["flagged"], ["tox"])
            # 10 file tokens on one call at $0.10 per million, per task
            self.assertAlmostEqual(fig["cost"]["project-docs"]["file_own_total"], round(3 * 10 * 0.1 / 1e6, 4))
            md = (self.kd / "report-context.md").read_text()
            for heading in ("## Pilot gate", "## Noise", "## Paired", "## Cost", "## Behaviour",
                            "## What this bench can see", "## Validation pool", "## Predictions"):
                self.assertIn(heading, md)
            self.make_run("g2", "011", "haiku", {"name": "project-docs", "sha256": docs["sha256"]})
            with self.assertRaisesRegex(SystemExit, "context file"):
                bench.main(argv)
            self.make_run("g2", "011", "sonnet", {"name": "generated", "sha256": gen["sha256"]})
            with self.assertRaisesRegex(SystemExit, "different models"):
                bench.main(argv)
            (self.kd / "runs" / "g2" / "score.jsonl").rename(self.kd / "runs" / "g2" / "unscored.jsonl")
            with self.assertRaisesRegex(SystemExit, "no score.jsonl"):
                bench.main(argv)


# ---------------------------------------------------------------------------- the regression rule

class NoNewFlag(KitCase):
    """With no new flag nothing changes: the harness settings and the task handed to the container are what they
    were before this study, and the kept reports of posts 3 and 4 reproduce from the kept runs."""

    def test_container_inputs_unchanged(self):
        self.run_agent(self.args())
        handed = self.docker.created[0]
        self.assertEqual(json.loads(handed["harness.json"]), json.loads((ROOT / "bench" / "harness.json").read_text()))
        t = self.m["tasks"][0]
        self.assertEqual(json.loads(handed["meta.json"]), bench.task_meta({"repo": REPO, "image": "fixture:1"}, t))
        self.assertEqual(sorted(handed), ["harness.json", "meta.json"])
        s = self.settings()
        self.assertEqual(sorted(s), ["context", "harness", "image_id", "tasks_checksum", "tickets"])
        self.assertEqual((s["tickets"], s["context"]), ({"version": "as-filed"}, {"name": "none"}))

    def test_runner_without_context_unchanged(self):
        """run_task.py keeps the harness's template and task when meta.json has no context and no task_text."""
        src = (ROOT / "bench" / "runner" / "run_task.py").read_text()
        self.assertIn('if META.get("context"):', src)
        self.assertIn('if META.get("task_text") is not None:', src)
        self.assertIn('task = CFG["task_preamble"].format(repo=META["repo"]) + "\\n\\n" + META["issue_title"] + '
                      '"\\n\\n" + META["issue_body"]', src)

    @unittest.skipUnless((KEPT / "report.json").exists() and (KEPT / "runs" / "w2" / "score.jsonl").exists(),
                         "needs the kept runs and reports of posts 3 and 4")
    def test_kept_reports_reproduce(self):
        recipe = str(ROOT / "bench" / "recipes" / "pytest-dev__pytest.json")
        labels = str(KEPT / "fairness" / "labels.jsonl")
        with tempfile.TemporaryDirectory() as d:
            out3, out4 = Path(d) / "report.md", Path(d) / "report-tickets.md"
            with contextlib.redirect_stdout(io.StringIO()):
                bench.main(["report", "--recipe", recipe, "--runs", "r1", "r2", "--labels", labels, "--out", str(out3)])
                bench.main(["report", "--recipe", recipe, "--paired", "r1,r2", "t1,t2", "--paired", "r1,r2", "w1,w2",
                            "--labels", labels, "--study", "tickets", "--out", str(out4)])
            for new, kept in ((out3, KEPT / "report.md"), (out4, KEPT / "report-tickets.md")):
                self.assertEqual(new.read_text(), kept.read_text())
                a, b = json.loads(new.with_suffix(".json").read_text()), json.loads(kept.with_suffix(".json").read_text())
                a.pop("written"), b.pop("written")
                self.assertEqual(a, b)


# ---------------------------------------------------------------------------- with Docker and the bench image

IMAGE = os.environ.get("BENCH_TEST_IMAGE", "bench-pytest-dev__pytest:1")


def have_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


@unittest.skipUnless(have_image(), f"needs docker and the image {IMAGE} (bench.py image)")
class WithDocker(unittest.TestCase):
    """What only the container can show: the harness's real template composes and renders as the pure function
    says, the harness passes run()'s keyword arguments to its templates, no PATH comes from swebench.yaml, and the
    resolver finds what the image has and not what it lacks."""

    def harness_python(self, code):
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{ROOT / 'bench' / 'runner'}:/runner:ro",
                            IMAGE, "/opt/harness/bin/python", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_template_and_variables(self):
        out = self.harness_python(
            "import json, sys, inspect, yaml\nsys.path.insert(0, '/runner')\nimport agent_parts\n"
            "from jinja2 import StrictUndefined, Template\nfrom minisweagent.config import get_config_path\n"
            "from minisweagent.agents.default import DefaultAgent\n"
            "cfg = yaml.safe_load(get_config_path('swebench.yaml').read_text())\nt = cfg['agent']['instance_template']\n"
            "r = lambda s, **k: Template(s, undefined=StrictUndefined).render(**k)\n"
            "print(json.dumps({'template': t, 'plain': r(t, task='T'), 'with': r(agent_parts.compose_instance_template(t), "
            "task='T', context_file_text='{{ x }}\\n'), 'run': inspect.getsource(DefaultAgent.run), "
            "'env': (cfg.get('environment') or {}).get('env') or {}}))")
        self.assertEqual(out["with"], agent_parts.CONTEXT_BLOCK.replace("{{ context_file_text }}", "{{ x }}\n") + out["plain"])
        self.assertIn('extra_template_vars |= {"task": task, **kwargs}', out["run"])
        self.assertNotIn("PATH", out["env"])
        if TRAJECTORY.exists():
            tr = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
            self.assertEqual(out["template"], tr["info"]["config"]["agent"]["instance_template"])

    def test_resolver(self):
        with tempfile.TemporaryDirectory(dir=bench.scratch()) as d:
            kd = Path(d)
            repo, rows = make_fixture.build(kd / "src")
            (kd / "archives").mkdir()
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(kd / "archives" / "t.tar"), "HEAD"],
                           check=True)
            recipe = {"repo": REPO, "image": IMAGE, "setup": "", "env": {}, "docker": {"cpus": "2", "memory": "2g"}}
            task = {"task_id": "t", "issue_title": "x", "issue_body": "y"}
            with mock.patch.object(bench, "kit_dir", lambda repo: kd), \
                 mock.patch.object(bench, "image_id", lambda tag: "img"):
                resolve = context.docker_resolver(recipe, {"image_id": "img"}, task)
                found = resolve(context.resolve_requests(context.command_candidates(
                    "```sh\ngit status\npython -m pytest -q\ntox -e py\npython -m tox\n```\n")))
        self.assertEqual(found, {"git": True, "python": True, "python -m pytest": True, "tox": False,
                                 "python -m tox": False})



class ReportAdditions(unittest.TestCase):
    """Two figures added before any run was read (changelog, 14 November 2026): commands that need the network,
    which the "commands work" check passes because pip and git resolve, and the file's size measured from the
    runs instead of estimated."""

    def test_needs_network(self):
        yes = ["pip install tox", "python -m pip install -e '.[dev]'", "cd /testbed && pip3 install --user pre-commit",
               "git fetch --tags https://github.com/pytest-dev/pytest", "git -C /testbed push -u",
               "bash -c 'git clone https://example.org/x.git'", "timeout 60 pip install tox 2>&1 | tail -2"]
        no = ["pip list", "pip --version", "python -m pip show pytest", "pip install --no-index -e /testbed",
              "git status --short", "git diff -- src/_pytest/mark.py > patch.txt", "git log --oneline | head -5",
              "echo 'pip install tox'", "grep -n 'git push' CONTRIBUTING.rst", "python -m pytest testing -k install"]
        for c in yes:
            self.assertTrue(bench.needs_network(c), c)
        for c in no:
            self.assertFalse(bench.needs_network(c), c)

    def test_network_calls_counted_per_run(self):
        cmds = {"n1": {"a": ["git status", "pip install tox"], "b": ["python -m pytest -q"]},
                "d1": {"a": ["tox -e py", "pip install tox", "git fetch --tags"], "b": ["pip list"]},
                "d2": {"a": [], "b": []}}
        b = bench.behaviour_figures(["n1"], ["d1", "d2"], cmds, ["a", "b"], flagged=["tox"])
        self.assertEqual(b["network_calls"], {"n1": 1, "d1": 2, "d2": 0})
        # `tox -e py` invokes tox as a program; `pip install tox` only names it as an argument
        self.assertEqual(b["flagged_calls"]["d1"]["commands"], 1)

    def test_asks_counted_from_the_diffs(self):
        diff = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\ndiff --git a/changelog/12.bugfix.rst b/changelog/12.bugfix.rst\n"
        self.assertEqual(bench.diff_paths(diff), ["src/x.py", "changelog/12.bugfix.rst"])
        self.assertEqual(bench.diff_paths(""), [])
        paths = {"n1": {"a": ["src/x.py"], "b": ["src/y.py", "changelog/1.bugfix.rst"]},
                 "d1": {"a": ["src/x.py", "changelog/2.bugfix.rst", "AUTHORS"], "b": ["testing/test_y.py"], "c": ["changelog/9.rst"]}}
        asks = {"changelog fragment": ["changelog/"], "AUTHORS line": ["AUTHORS"], "tests": ["testing/"]}
        got = bench.asks_figures(paths, asks, ["a", "b"])  # task c is not among the tasks read
        self.assertEqual(got["changelog fragment"], {"n1": 1, "d1": 1})
        self.assertEqual(got["AUTHORS line"], {"n1": 0, "d1": 1})
        self.assertEqual(got["tests"], {"n1": 0, "d1": 1})

    def test_measured_file_tokens(self):
        none = {"n1": {"a": 3000, "b": 5000}, "n2": {"a": 3000, "b": 5000}}
        arm = {"d1": {"a": 4900, "b": 6900}, "d2": {"a": 4900, "b": 6910, "c": 9999}}  # c has no no-file run
        self.assertEqual(bench.measured_file_tokens(arm, none), 1900)
        self.assertIsNone(bench.measured_file_tokens({"d1": {"z": 1}}, none))

    def test_first_prompt_tokens_counts_cached_input(self):
        u = {"a": [{"input": 100, "output": 5, "cache_read": 20, "cache_write": 30}, {"input": 900, "output": 1}], "b": []}
        self.assertEqual(bench.first_prompt_tokens(u), {"a": 150})


if __name__ == "__main__":
    unittest.main()
