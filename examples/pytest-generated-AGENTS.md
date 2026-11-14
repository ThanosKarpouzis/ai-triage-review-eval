# AGENTS.md

Guide for coding-agent sessions in this repository (the pytest project).

## Environment

- Python 3.12 at `/opt/testbed/bin/python` (`python` and `python3` both point to it). pytest is an editable install from `/testbed` (9.1.0.dev0). Also installed: pytest-xdist, hypothesis, pexpect, numpy, PyYAML, argcomplete.
- NOT installed: tox, ruff, mypy, pre-commit, pyright. Lint and type checks cannot be run here; do not install them unless asked.

## Running tests

Run from `/testbed`. Use `-p no:cacheprovider` if you want to avoid writing `.pytest_cache`.

- One file: `python -m pytest testing/test_config.py -q`
- One test (class method): `python -m pytest "testing/test_config.py::TestParseIni::test_getcfg_and_config" -q`
- Keyword filter: `python -m pytest testing/test_mark.py -k marked_class -q`
- Doctests in the package: `python -m pytest --doctest-modules --pyargs _pytest -q`
- Parallel: `python -m pytest -n auto -q` (xdist is installed)

Verified here: `testing/test_config.py` (256 passed, 1 xfailed), the class-method node ID, the `-k` form, `-n auto` on `testing/test_mark.py`, and the doctest command (5 passed). The full suite was not run.

## Layout

- `src/_pytest/`: the pytest implementation. Subpackages: `assertion/`, `config/`, `mark/`, `_code/`, `_io/`. Public entry points are in `src/pytest/`.
- `testing/`: the test suite. Most tests are `testing/test_<module>.py` mirroring `src/_pytest/<module>.py`. Subdirectories hold specialised tests (`python/`, `logging/`, `code/`, `io/`, `freeze/`, etc.). Add new tests to the matching file.
- `testing/example_scripts/` and `testing/plugins_integration/` are fixtures, not collected tests.
- `doc/en/`: Sphinx documentation. `CHANGELOG` is generated from `changelog/`.

## Conventions

- Style: ruff (line length 88, target py310), mypy (`disallow_untyped_defs`), and pre-commit hooks (configured in `.pre-commit-config.yaml`). Use type annotations on all functions, including tests (`-> None`). Follow PEP 8 naming.
- Tests: plain `def test_...` functions or `class Test...` groups. Use the `pytester` fixture for black-box tests that run pytest in a subprocess or in-process (`pytester.makepyfile(...)`, `pytester.runpytest(...)`).
- Changelog: every user-visible change needs a file in `changelog/` named `<issue-number>.<type>.rst`, where type is one of `feature`, `improvement`, `bugfix`, `doc`, `deprecation`, `breaking`, `vendor`, `packaging`, `contrib`, `misc`. Write one full sentence in past or present tense, aimed at users. Skip it only for changes that do not affect documented behaviour.
- `AUTHORS`: add your name in alphabetical order for non-trivial contributions.
- Do not edit `pyproject.toml`, `tox.ini`, or other config files as part of a fix.
