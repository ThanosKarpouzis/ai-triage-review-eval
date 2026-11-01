"""A pytest plugin the bench loads with `-p bench_outcomes`: it writes the outcome of every test, by node id,
to the JSON file named in BENCH_OUTCOMES. A test module that fails to import is recorded under its own path
with the outcome "error", so a new test file that imports code that does not exist yet still counts as failing.
Standard library and pytest's public hooks only, so it works on any pytest version the bench is likely to meet.
"""
import json
import os

_outcomes = {}


def pytest_collectreport(report):
    if report.failed:
        _outcomes[report.nodeid] = "error"


def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.when == "setup":
        if report.failed:
            _outcomes[nid] = "error"
        elif report.skipped:
            _outcomes[nid] = "xfailed" if hasattr(report, "wasxfail") else "skipped"
    elif report.when == "call":
        if hasattr(report, "wasxfail"):
            _outcomes[nid] = "xpassed" if report.passed else "xfailed"
        else:
            _outcomes[nid] = report.outcome  # passed, failed or skipped
    elif report.when == "teardown" and report.failed:
        _outcomes[nid] = "error"


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("BENCH_OUTCOMES")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"exitstatus": int(exitstatus), "outcomes": _outcomes}, f, indent=0, sort_keys=True)
