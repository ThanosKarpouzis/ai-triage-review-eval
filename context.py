#!/usr/bin/env python3
"""context.py: make the repository context file a run hands to the agent, check it, and freeze it, so a run can vary
the context file and nothing else.

One subcommand per step, each reading tasks.json from the recipe's kit folder and writing to its context/ folder:

  docs      context/project-docs.md: one section of the project's own contributor documentation at the oldest
            frozen task's base commit, whole and unedited (default: "Preparing Pull Requests" in CONTRIBUTING.rst)
  generate  context/generated.md: one harness session in the bench container at that same commit, with
            bench/context_init_prompt.md as the task, keeping the AGENTS.md the agent writes at the repository root;
            with --retry, one more session told which programs the check could not find (there is no third)
            (needs AI_GATEWAY_API_KEY and Docker)
  check     the leak check (a code-like token, by tickets.py's tokeniser, that occurs in a withheld test or fix of
            any task but nowhere in the checkout at the base commit) and the "commands work" check (every program
            and `python -m` module the file names, resolved inside the bench container as the agent user with the
            agent's PATH, never executed), plus the statements that contradict the harness prompt (needs Docker for
            the commands; --skip-commands runs the rest without it)
  freeze    context/<name>.json: the file's checksum, the task set's checksum, its source and the check verdicts;
            refuses a file that leaks or has not been checked as it stands

`bench.py run --context <name>` hands a frozen file to the agent in its own block ahead of the ticket. Nothing
changes a file after it is frozen, and neither file is edited by hand.

Standard library only. Written up in the fifth post of the series "Ask your own repository" on thanoskarpouzis.com.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
import tickets  # noqa: E402

INIT_PROMPT = HERE / "bench" / "context_init_prompt.md"
HARNESS = HERE / "bench" / "harness.haiku.json"   # the generated file is written by the study's model
DOCS_FILE = "CONTRIBUTING.rst"
DOCS_SECTION = "Preparing Pull Requests"
DOCS_NAME, GENERATED_NAME = "project-docs", "generated"
AGENTS_FILE = "AGENTS.md"
MAX_ATTEMPTS = 2                 # the first session and one retry
CHARS_PER_TOKEN = 4              # the token estimate in the freeze record, as the spec counts it
# The agent's PATH, as run_task.py sets it for every command the agent runs (a test checks they agree).
AGENT_PATH = "/opt/testbed/bin:/usr/local/bin:/usr/bin:/bin"
KEY_ENV = "AI_GATEWAY_API_KEY"


def context_dir(recipe):
    return bench.kit_dir(recipe["repo"]) / "context"


def load_tasks(recipe):
    m = json.loads((bench.kit_dir(recipe["repo"]) / "tasks.json").read_text(encoding="utf-8"))
    if bench.canonical_checksum(m["tasks"]) != m["checksum"]:
        sys.exit("tasks.json does not match its checksum")
    return m


def source_task(manifest, task_id=None):
    """The task whose base commit a file is made at: the one merged first, so that the file is older than every
    fix it will be used on, as a team's file is."""
    if task_id:
        return next(t for t in manifest["tasks"] if t["task_id"] == task_id)
    return min(manifest["tasks"], key=lambda t: (t["merged_at"], t["task_id"]))


def read_from_archive(archive, path):
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lstrip("./") == path:
                return tar.extractfile(m).read().decode("utf-8")
    raise KeyError(f"{path} is not in {archive}")


def archive_text(archive):
    """Every file of a checkout as one text, for the leak check's "nowhere in the repository" test."""
    parts = []
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            if m.isfile():
                parts.append(tar.extractfile(m).read().decode("utf-8", errors="replace"))
    return "\n".join(parts)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_estimate(text):
    return round(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------- docs

UNDERLINE = re.compile(r"^([=\-~^\"'`#*+<>:.,_!$%&()/;?@\[\]{}|\\])\1{2,}\s*$")
ANCHOR = re.compile(r"^\.\. _[^:]*:\s*$")


def rst_headings(lines):
    """(line index of the heading's first line, title, level) for each reStructuredText section heading, the level
    being the order in which its adornment style (character, with or without overline) first appears, as
    docutils assigns it."""
    styles, out, i = [], [], 0
    while i < len(lines) - 1:
        over = UNDERLINE.match(lines[i])
        if over and i + 2 < len(lines) and lines[i + 1].strip() and UNDERLINE.match(lines[i + 2]) \
                and lines[i + 2].strip()[0] == lines[i].strip()[0]:
            style, title, start, i = (lines[i].strip()[0], True), lines[i + 1].strip(), i, i + 3
        elif lines[i].strip() and not lines[i].startswith((" ", "\t")) and UNDERLINE.match(lines[i + 1]) \
                and len(lines[i + 1].rstrip()) >= len(lines[i].rstrip()) and not UNDERLINE.match(lines[i]):
            style, title, start, i = (lines[i + 1].strip()[0], False), lines[i].strip(), i, i + 2
        else:
            i += 1
            continue
        if style not in styles:
            styles.append(style)
        out.append((start, title, styles.index(style)))
    return out


def rst_section(text, title):
    """The section with this heading, from its heading line up to the next heading of the same or a higher
    level, unedited. Blank lines and reference labels (`.. _name:`) just before the next heading belong to that
    heading and are left out. Returns (text, first line, last line), lines counted from 1."""
    lines = text.splitlines()
    heads = rst_headings(lines)
    for n, (start, t, level) in enumerate(heads):
        if t == title:
            end = len(lines)
            for s2, _t2, l2 in heads[n + 1:]:
                if l2 <= level:
                    end = s2
                    break
            while end > start and (not lines[end - 1].strip() or ANCHOR.match(lines[end - 1])):
                end -= 1
            return "\n".join(lines[start:end]) + "\n", start + 1, end
    raise KeyError(f"no section headed {title!r}")


def cmd_docs(args):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    task = source_task(m, args.task)
    kd = bench.kit_dir(recipe["repo"])
    text = read_from_archive(kd / "archives" / f"{task['task_id']}.tar", args.file)
    section, first, last = rst_section(text, args.section)
    source = {"kind": "docs", "task_id": task["task_id"], "parent": task["parent"], "file": args.file,
              "section": args.section, "file_lines": [first, last]}
    write_file(recipe, DOCS_NAME, section, source)


def write_file(recipe, name, text, source):
    """A context file and its source record. A frozen file is never written again."""
    cd = context_dir(recipe)
    cd.mkdir(parents=True, exist_ok=True)
    if (cd / f"{name}.json").exists():
        sys.exit(f"{name} is frozen ({cd / (name + '.json')}); a frozen file does not change")
    (cd / f"{name}.md").write_text(text, encoding="utf-8")
    source = {**source, "sha256": sha256_text(text), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (cd / f"{name}.source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"done: {cd / (name + '.md')}, {len(text.splitlines())} lines, about {token_estimate(text)} tokens, "
          f"sha256 {source['sha256'][:16]}")


# ---------------------------------------------------------------------------- generate

def generate_prompt(init_prompt, retry_programs=None):
    """The task for a generation session: the initialisation prompt, and on the retry a note naming the
    programs the check could not find where the agent runs."""
    if not retry_programs:
        return init_prompt
    listed = "\n".join(f"- {p}" for p in retry_programs)
    return (init_prompt.rstrip("\n") + "\n\nAn earlier draft of this file named commands that call these programs, "
            "which are not available in this environment:\n\n" + listed + "\n\nWrite the file without commands that "
            "call them.\n")


def docker_session(recipe, manifest, task, cfg, key, attempt_dir):
    """The real session: one harness run in a fresh bench container at the task's base commit, with the prompt
    as the whole task (no ticket), keeping the AGENTS.md the agent leaves at the repository root."""
    kd = bench.kit_dir(recipe["repo"])
    if bench.image_id(recipe["image"]) != manifest["image_id"]:
        sys.exit("the bench image is not the one the tasks were frozen with")

    def session(prompt):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        meta = bench.task_meta(recipe, task, {"task_text": prompt, "collect": [AGENTS_FILE]})
        name = f"bench-context-{attempt_dir.name}".replace("_", "-")[:120]
        res = bench.agent_container(recipe, cfg, name, kd / "archives" / f"{task['task_id']}.tar", meta,
                                    attempt_dir, key)
        f = attempt_dir / "collected" / AGENTS_FILE
        return {"text": f.read_text(encoding="utf-8") if f.exists() else None, "cost": res.get("cost"),
                "api_calls": res.get("api_calls"), "exit_status": res.get("exit_status")}
    return session


def generate(log, prompt_text, session, retry_programs=None):
    """One generation attempt: refuses a retry without a first attempt, a first attempt twice, and a third.
    Returns the log entry; the caller writes the file."""
    if retry_programs is None and log:
        raise ValueError("a first attempt exists; use --retry for the one regeneration")
    if retry_programs is not None and not log:
        raise ValueError("--retry needs a first attempt")
    if len(log) >= MAX_ATTEMPTS:
        raise ValueError(f"{len(log)} attempts made; there is no third")
    prompt = generate_prompt(prompt_text, retry_programs)
    t0 = time.monotonic()
    out = session(prompt)
    text = out.get("text")
    return {"attempt": len(log) + 1, "retry_programs": retry_programs, "prompt_sha256": sha256_text(prompt),
            "init_prompt_sha256": sha256_text(prompt_text), "found": text is not None,
            "sha256": sha256_text(text) if text is not None else None, "cost": out.get("cost"),
            "api_calls": out.get("api_calls"), "exit_status": out.get("exit_status"),
            "seconds": round(time.monotonic() - t0, 1), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, text


def cmd_generate(args, session=None):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    task = source_task(m, args.task)
    cd = context_dir(recipe)
    cd.mkdir(parents=True, exist_ok=True)
    if (cd / f"{GENERATED_NAME}.json").exists():  # before any session, so nothing is spent on a frozen file
        sys.exit(f"{GENERATED_NAME} is frozen; a frozen file does not change")
    log_path = cd / "generate-log.jsonl"
    log = bench.read_jsonl(log_path)
    retry = None
    if args.retry:
        chk = cd / f"{GENERATED_NAME}.check.json"
        if not chk.exists():
            sys.exit(f"--retry needs the check of the first file ({chk}); run context.py check first")
        found = json.loads(chk.read_text(encoding="utf-8"))
        retry = [p.strip() for p in args.programs.split(",") if p.strip()] if args.programs \
            else found["commands"]["not_found"]
        if not retry:
            sys.exit("the check found no missing program; there is nothing to regenerate for")
    if session is None:
        key = os.environ.get(KEY_ENV)
        if not key:
            sys.exit(f"set {KEY_ENV} in this shell first")
        cfg = bench.load_harness(args.harness)
        session = docker_session(recipe, m, task, cfg, key, cd / "generate" / f"attempt-{len(log) + 1}")
    try:
        entry, text = generate(log, INIT_PROMPT.read_text(encoding="utf-8"), session, retry)
    except ValueError as e:
        sys.exit(str(e))
    entry["task_id"], entry["parent"] = task["task_id"], task["parent"]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"attempt {entry['attempt']}: {entry['exit_status']} ${entry['cost'] or 0:.3f} {entry['api_calls']} calls "
          f"{entry['seconds']}s; {AGENTS_FILE} {'found' if text is not None else 'NOT found'}")
    if text is None:
        sys.exit(f"the session left no {AGENTS_FILE} at the repository root; nothing written. Stop and tell Claude.")
    write_file(recipe, GENERATED_NAME, text, {"kind": "generated", "task_id": task["task_id"],
                                              "parent": task["parent"], "attempt": entry["attempt"],
                                              "retry_programs": retry, "cost": entry["cost"],
                                              "api_calls": entry["api_calls"],
                                              "init_prompt_sha256": entry["init_prompt_sha256"]})


# ---------------------------------------------------------------------------- check: leaks

def added_lines(patch):
    return "\n".join(l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))


def leak_tokens(text, base_text, patches):
    """Code-like tokens of the file (tickets.py's tokeniser) that a withheld test or fix adds and that occur
    nowhere in the checkout at the base commit: a name the file could only know from a future fix."""
    added = [added_lines(p) for p in patches]
    return [t for t in tickets.code_tokens(text) if t not in base_text and any(t in a for a in added)]


# ---------------------------------------------------------------------------- check: commands

MD_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`]*)[^\n]*\n(.*?)^[ \t]*\1[ \t]*$", re.M | re.S)
RST_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+(?:code-block|code|sourcecode)::\s*(\S*)\s*$")
SHELL_LANGS = {"", "sh", "bash", "shell", "console", "zsh", "shell-session", "sh-session", "terminal", "shellsession"}
PROMPT = re.compile(r"^\s*(?:\$|%|\(\S+\)\s*\$)\s+")
PROGRAM_WORD = re.compile(r"^[A-Za-z0-9_.~/+-]*[A-Za-z][A-Za-z0-9_.~/+-]*$")
DATA_LINE = re.compile(r"^\s*(?:[\w.-]+\s*:(?:\s|$)|[\w.-]+\s*=|[\[\]{}()<>\"'|+*>]|-{2,}\s|\d)")
INLINE_RST = re.compile(r"``(.+?)``", re.S)
INLINE_MD = re.compile(r"(?<![`:\w])`([^`\n]+)`(?![`_])")


def literal_blocks(text):
    """(first line number, language, block lines) for each literal block: Markdown fences, reStructuredText
    code-block directives, and reStructuredText `::` blocks (no language)."""
    out = []
    for m in MD_FENCE.finditer(text):
        first = text[:m.start(3)].count("\n") + 1
        out.append((first, m.group(2).lower(), m.group(3).splitlines()))
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        d = RST_DIRECTIVE.match(line)
        if d or (line.rstrip().endswith("::") and not line.strip().startswith("..")):
            lang = d.group(2).lower() if d else ""
            indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith(":")) and d:
                j += 1  # a directive's options
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and len(lines[j]) - len(lines[j].lstrip()) > indent:
                block_indent, k = len(lines[j]) - len(lines[j].lstrip()), j
                while k < len(lines) and (not lines[k].strip() or len(lines[k]) - len(lines[k].lstrip()) >= block_indent):
                    k += 1
                out.append((j + 1, lang, lines[j:k]))
                i = k
                continue
        i += 1
    return out


def command_lines(block):
    """The command lines of a literal block: when any line carries a shell prompt ($), only those, without it
    (the rest is output); otherwise every line that is not blank, a comment or plainly data (`key: value`,
    `key = value`, a quoted or bracketed line, a number). Continued lines are joined."""
    joined, buf = [], ""
    for line in block:
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        joined.append(buf + line)
        buf = ""
    if buf:
        joined.append(buf)
    prompted = [l for l in joined if PROMPT.match(l)]
    if prompted:
        return [PROMPT.sub("", l, count=1) for l in prompted]
    return [l.strip() for l in joined if l.strip() and not l.strip().startswith("#") and not DATA_LINE.match(l)
            and not l.strip().startswith(">>>")]


def programs_in(command):
    """(key, program, module) for each program a command line calls: the program as written, and for
    `python -m x` also the module, keyed `python -m x`."""
    out = []
    for word, module in bench.simple_commands(command, raw=True):
        if not PROGRAM_WORD.match(word) or word.endswith(":"):
            continue
        out.append((word, word, None))
        if module:
            out.append((f"{word} -m {module}", word, module))
    return out


def command_candidates(text):
    """Every candidate command the file names, with the line it is on: the command lines of shell-marked and
    unmarked literal blocks (Python-marked and other blocks are skipped), and inline code of two words or more
    that starts like a program (a single word in backticks names a thing, it does not run it; a reStructuredText
    link `name <url>`_ is not code). Proposes; Thanos confirms which are real commands."""
    out = []
    for first, lang, block in literal_blocks(text):
        if lang not in SHELL_LANGS:
            continue
        for cmd in command_lines(block):
            line = first + next((n for n, l in enumerate(block) if cmd in l), 0)
            out.append({"line": line, "source": "block", "command": cmd})
    rest = INLINE_RST.sub(lambda m: "\0" * len(m.group(0)), text)
    spans = [(m.start(), m.group(1)) for m in INLINE_RST.finditer(text)]
    spans += [(m.start(), m.group(1)) for m in INLINE_MD.finditer(rest) if "<http" not in m.group(1)]
    for pos, span in sorted(spans):
        words = span.split()
        if len(words) >= 2 and PROGRAM_WORD.match(words[0]) and not words[0].startswith(("-", "<", ".")) \
                and not words[0].endswith(":") and "\n" not in span:
            out.append({"line": text[:pos].count("\n") + 1, "source": "inline", "command": " ".join(words)})
    for c in out:
        c["programs"] = [k for k, _p, _m in programs_in(c["command"])]
    return [c for c in out if c["programs"]]


def resolve_requests(candidates):
    """One request per distinct program or module, for the resolver."""
    seen = {}
    for c in candidates:
        for key, prog, module in programs_in(c["command"]):
            seen.setdefault(key, {"key": key, "program": prog, "module": module})
    return list(seen.values())


RESOLVER = r'''
import json, sys
sys.path.insert(0, "/runner")
import inside
meta = json.loads(open("/task/meta.json").read())
inside.prepare(meta)
env = meta["agent_env"]
spec = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
out = {}
for q in meta["resolve"]:
    if q["module"]:
        cmd = [q["program"], "-c", spec, q["module"]]
    else:
        cmd = ["bash", "-c", 'command -v -- "$1"', "resolve", q["program"]]
    try:
        r = inside.sh(cmd, user="agent", env=env, check=False, timeout=60)
        out[q["key"]] = {"found": r.returncode == 0, "detail": r.stdout.strip()[-200:]}
    except Exception as e:
        out[q["key"]] = {"found": False, "detail": repr(e)[:200]}
print("RESOLVED " + json.dumps(out))
'''


def docker_resolver(recipe, manifest, task):
    """The real resolver: one bench container at the task's base commit, prepared as for a run, where each
    program is looked up with `command -v` and each module with importlib's find_spec, as the agent user with
    the agent's PATH, without running anything the file names. The network is off."""
    import shutil
    import tempfile
    kd = bench.kit_dir(recipe["repo"])
    if bench.image_id(recipe["image"]) != manifest["image_id"]:
        sys.exit("the bench image is not the one the tasks were frozen with")

    def resolve(requests):
        env = {"PATH": AGENT_PATH, "HOME": "/home/agent", "LANG": "C.UTF-8", **(recipe.get("env") or {})}
        meta = bench.task_meta(recipe, task, {"agent_env": env, "resolve": requests})
        with tempfile.TemporaryDirectory(dir=bench.scratch()) as tmp:
            shutil.copyfile(kd / "archives" / f"{task['task_id']}.tar", Path(tmp) / "checkout.tar")
            (Path(tmp) / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            r = bench.run(["docker", "run", "--rm", "--network", "none", *bench.docker_limits(recipe),
                           "-v", f"{tmp}:/task:ro", "-v", f"{bench.BENCH / 'runner'}:/runner:ro", recipe["image"],
                           "python3", "-c", RESOLVER], check=False, timeout=900)
        line = next((l for l in (r.stdout or "").splitlines() if l.startswith("RESOLVED ")), None)
        if line is None:
            sys.exit(f"the resolver failed:\n{(r.stdout or '')[-1500:]}\n{(r.stderr or '')[-1500:]}")
        return {k: v["found"] for k, v in json.loads(line[len("RESOLVED "):]).items()}
    return resolve


# ---------------------------------------------------------------------------- check: contradictions

CONTRADICTIONS = [  # what the harness prompt says against: tests and configuration are not to be modified, and
    # the patch holds only the source files changed (so no changelog entry and no AUTHORS line)
    ("tests", re.compile(r"\b(?:write|writing|add|adding|include|including|update|updating|create|creating)\b"
                         r"[^.\n]{0,60}\btests?\b", re.I)),
    ("changelog", re.compile(r"\bchangelog\b", re.I)),
    ("AUTHORS", re.compile(r"\bAUTHORS\b")),
    ("configuration", re.compile(r"pyproject\.toml|setup\.cfg|tox\.ini|\bconfiguration files?\b|\bconfig files?\b",
                                 re.I)),
]


def contradictions(text):
    """Lines that ask for something the harness prompt rules out, spotted by keyword: adding or writing tests,
    a changelog entry, an AUTHORS line, changes to configuration files. Counted and reported, not removed."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for cat, rx in CONTRADICTIONS:
            if rx.search(line):
                out.append({"category": cat, "line": n, "text": line.strip()[:200]})
    return out


def check_file(text, base_text, patches, resolver=None):
    """Every verdict of the check for one file. Without a resolver the commands are listed but not resolved."""
    cands = command_candidates(text)
    reqs = resolve_requests(cands)
    found = resolver(reqs) if resolver else None
    commands = {"candidates": cands, "checked": found is not None,
                "programs": {q["key"]: (found or {}).get(q["key"]) for q in reqs},
                "not_found": sorted(k for k, v in (found or {}).items() if v is False)}
    leaks = leak_tokens(text, base_text, patches)
    return {"sha256": sha256_text(text), "leak": {"tokens": leaks, "passed": not leaks}, "commands": commands,
            "contradictions": contradictions(text)}


def cmd_check(args, resolver=None):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    kd, cd = bench.kit_dir(recipe["repo"]), context_dir(recipe)
    path = cd / f"{args.name}.md"
    text = path.read_text(encoding="utf-8")
    src = json.loads((cd / f"{args.name}.source.json").read_text(encoding="utf-8"))
    task = source_task(m, src["task_id"])
    base = archive_text(kd / "archives" / f"{task['task_id']}.tar")
    patches = [(kd / d / f"{t['task_id']}.patch").read_text(encoding="utf-8", errors="replace")
               for t in m["tasks"] for d in ("withheld", "fixes")]
    if resolver is None and not args.skip_commands:
        resolver = docker_resolver(recipe, m, task)
    res = check_file(text, base, patches, None if args.skip_commands else resolver)
    res.update(name=args.name, base_task=task["task_id"], written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (cd / f"{args.name}.check.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"{args.name}: {len(text.splitlines())} lines, about {token_estimate(text)} tokens")
    print(f"leak check: {'passed' if res['leak']['passed'] else 'FAILED ' + str(res['leak']['tokens'])}")
    c = res["commands"]
    for key, ok in c["programs"].items():
        lines = sorted({x["line"] for x in c["candidates"] if key in x["programs"]})
        print(f"  {key}: {'found' if ok else 'NOT FOUND' if ok is False else 'not checked'} (lines {lines})")
    print(f"commands: {len(c['candidates'])} candidates, {len(c['programs'])} programs, "
          + (f"not found: {c['not_found'] or 'none'}" if c["checked"] else "not resolved (--skip-commands)"))
    for x in res["contradictions"]:
        print(f"  against the harness prompt ({x['category']}), line {x['line']}: {x['text']}")
    print(f"{len(res['contradictions'])} statements against the harness prompt; wrote {cd / (args.name + '.check.json')}")
    sys.exit(0 if res["leak"]["passed"] else 1)


# ---------------------------------------------------------------------------- freeze

def freeze_record(name, text, source, check, tasks_checksum, confirmed=None):
    """The freeze record. Refuses a file that leaks, a check made on other text, and a check whose commands were
    not resolved. The flagged programs are the ones Thanos confirmed as real commands, or every program not
    found when none are named."""
    sha = sha256_text(text)
    if check.get("sha256") != sha:
        raise ValueError("the check was made on other text; run context.py check again")
    if not check["leak"]["passed"]:
        raise ValueError(f"the file leaks: {check['leak']['tokens']}")
    if not check["commands"]["checked"]:
        raise ValueError("the commands were not resolved (--skip-commands); run the full check first")
    not_found = check["commands"]["not_found"]
    if confirmed is not None and set(confirmed) - set(not_found):
        raise ValueError(f"confirmed programs the check found: {sorted(set(confirmed) - set(not_found))}")
    return {"name": name, "sha256": sha, "tasks_checksum": tasks_checksum, "source": source,
            "lines": len(text.splitlines()), "tokens_estimate": token_estimate(text),
            "check": {"leak_passed": True, "programs": check["commands"]["programs"], "not_found": not_found,
                      "flagged_programs": sorted(confirmed) if confirmed is not None else not_found,
                      "contradictions": check["contradictions"]},
            "frozen": time.strftime("%Y-%m-%dT%H:%M:%S")}


def cmd_freeze(args):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    cd = context_dir(recipe)
    out = cd / f"{args.name}.json"
    text = (cd / f"{args.name}.md").read_text(encoding="utf-8")
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["sha256"] == sha256_text(text) and old["tasks_checksum"] == m["checksum"]:
            print(f"unchanged: {out} sha256 {old['sha256']}")
            return
        sys.exit(f"{out} exists for other text; a frozen file does not change")
    source = json.loads((cd / f"{args.name}.source.json").read_text(encoding="utf-8"))
    check = json.loads((cd / f"{args.name}.check.json").read_text(encoding="utf-8"))
    confirmed = [p.strip() for p in args.confirmed.split(",") if p.strip()] if args.confirmed is not None else None
    try:
        doc = freeze_record(args.name, text, source, check, m["checksum"], confirmed)
    except ValueError as e:
        sys.exit(str(e))
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"done: {out}, {doc['lines']} lines, sha256 {doc['sha256']}, flagged {doc['check']['flagged_programs']}")


# ---------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = dict(required=True, help="the repository's recipe, e.g. bench/recipes/pytest-dev__pytest.json")

    p = sub.add_parser("docs", help="context/project-docs.md from the contributor documentation (no model call)")
    p.add_argument("--recipe", **rec)
    p.add_argument("--file", default=DOCS_FILE, help=f"default {DOCS_FILE}")
    p.add_argument("--section", default=DOCS_SECTION, help=f"default {DOCS_SECTION!r}")
    p.add_argument("--task", help="the task whose base commit to read; default the one merged first")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("generate", help=f"context/generated.md from one harness session (needs {KEY_ENV}, Docker)")
    p.add_argument("--recipe", **rec)
    p.add_argument("--harness", default=str(HARNESS), help="harness settings; default bench/harness.haiku.json")
    p.add_argument("--task", help="the task whose base commit to work at; default the one merged first")
    p.add_argument("--retry", action="store_true", help="the one regeneration, naming the programs the check flagged")
    p.add_argument("--programs", help="with --retry: the programs to name (default every one the check did not find)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("check", help="leak check, commands-work check (Docker), contradictions with the harness prompt")
    p.add_argument("--recipe", **rec)
    p.add_argument("--name", required=True, help=f"{DOCS_NAME} or {GENERATED_NAME}")
    p.add_argument("--skip-commands", action="store_true", help="list the commands without resolving them (no Docker)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("freeze", help="context/<name>.json with the checksum and the check verdicts")
    p.add_argument("--recipe", **rec)
    p.add_argument("--name", required=True, help=f"{DOCS_NAME} or {GENERATED_NAME}")
    p.add_argument("--confirmed", help="comma-separated programs not found that are real commands (may be empty); "
                                       "default every program not found")
    p.set_defaults(func=cmd_freeze)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
