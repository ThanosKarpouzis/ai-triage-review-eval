#!/usr/bin/env python3
"""tickets.py: write versions of each frozen task's ticket, so a run can vary the text the agent is handed and nothing else.

One subcommand per step, each reading tasks.json from the recipe's kit folder:

  title-only  tickets/title-only.json: the issue title as frozen, the body empty
  rewrite     restate each issue under fixed headings with acceptance criteria, by the rule in bench/rewrite_rule.md,
              one model call per task with the issue text as the only input; a leak check refuses any rewrite that
              names a code-like token the issue does not contain, asks once more with the tokens named, and keeps the
              as-filed text for that task if the second draft leaks too (needs AI_GATEWAY_API_KEY)
  check       run the leak check over a version file and print the offending tokens per task (exit 1 if any)
  freeze      write tickets/rewritten.json from the rewrite log, with the rule's checksum and the task set's checksum

A version file holds one object per task id with issue_title and issue_body; `bench.py run --tickets <version>` hands
those to the agent in place of the issue as filed. Nothing changes a ticket after its version file is frozen.

Standard library only. Written up in the fourth post of the series "Ask your own repository" on thanoskarpouzis.com.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

RULE = HERE / "bench" / "rewrite_rule.md"
GATEWAY = os.environ.get("BENCH_GATEWAY_URL", "https://ai-gateway.vercel.sh")
REWRITER = "anthropic/claude-sonnet-5.5"  # the gateway's name for the agent's model (bench/harness.json)
MAX_TOKENS = 4000
SPEND_CAP = 5.0
KEY_ENV = "AI_GATEWAY_API_KEY"

HEADINGS = ["Summary", "Observed behaviour", "Expected behaviour", "Steps to reproduce", "Acceptance criteria",
            "Out of scope and open questions"]
HEADING_WORDS = sorted({w for h in HEADINGS for w in h.split()} | set(HEADINGS)
                       | {"Not given in the issue", "None stated in the issue"})


def tickets_dir(recipe):
    return bench.kit_dir(recipe["repo"]) / "tickets"


def load_tasks(recipe):
    m = json.loads((bench.kit_dir(recipe["repo"]) / "tasks.json").read_text(encoding="utf-8"))
    if bench.canonical_checksum(m["tasks"]) != m["checksum"]:
        sys.exit("tasks.json does not match its checksum")
    return m


def allow_list(repo):
    owner, name = repo.split("/", 1)
    return set(HEADING_WORDS) | {owner, name, repo}


# ---------------------------------------------------------------------------- the leak check

FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n(.*?)^[ \t]*\1[ \t]*$", re.M | re.S)
INLINE = re.compile(r"(`+)(.+?)\1", re.S)
DOTTED = re.compile(r"(?<![\w./-])[A-Za-z_][\w/-]*(?:\.[A-Za-z_0-9][\w/-]*)+")  # a path counts whole
UNDERSCORED = re.compile(r"(?<![\w.])\w*_\w*")
CAMEL = re.compile(r"(?<![\w.])(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)(?![\w])")
FLAG = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*")
QUOTED = [re.compile(r'(?<!\w)"([^"\n]+)"(?!\w)'), re.compile(r"(?<!\w)'([^'\n]+)'(?!\w)"),
          re.compile(r"“([^”\n]+)”"), re.compile(r"(?<!\w)‘([^’\n]+)’(?!\w)")]
STRIP = "`'\"‘’“”"
HEADING_LINE = re.compile(r"^[ \t]*#{1,6}[ \t].*$", re.M)


def code_tokens(text):
    """Code-like tokens in a rewrite, in order of appearance: the pieces of backticked spans and fenced blocks
    (split on whitespace; a fence's language tag is markup and not a token), then, in the prose left over, dotted
    or underscored names, CamelCase names, flags beginning with one or two dashes, and quoted strings of two
    words or more. Heading lines are left out: a token that occurs only in a heading is ignored."""
    text = HEADING_LINE.sub(" ", text)
    found = []
    for m in FENCE.finditer(text):
        found += m.group(2).split()
    prose = FENCE.sub(" ", text)
    for m in INLINE.finditer(prose):
        found += m.group(2).split()
    prose = INLINE.sub(" ", prose)
    for rx in (DOTTED, UNDERSCORED, CAMEL, FLAG):
        found += [m.group(0) for m in rx.finditer(prose) if re.search(r"[A-Za-z]", m.group(0))]
    for rx in QUOTED:
        found += [m.group(1) for m in rx.finditer(prose) if len(m.group(1).split()) >= 2]
    out = []
    for tok in found:
        tok = tok.strip(STRIP)
        if tok and tok not in out:
            out.append(tok)
    return out


def leaks(rewrite, issue_text, allow=()):
    """Every code-like token of the rewrite that the issue as filed does not contain (case-sensitive substring,
    HTML comments included, since the agent sees the body with them) and that is not on the allow list."""
    allow = set(allow)
    return [t for t in code_tokens(rewrite) if t not in allow and t not in issue_text]


def issue_text(task):
    return task["issue_title"] + "\n" + task["issue_body"]


# ---------------------------------------------------------------------------- versions

def version_file(tickets, version, tasks_checksum, extra=None):
    """A version file: one ticket per task id, the task set it was made from, and a checksum over the tickets
    computed the way bench.py checksums the task set."""
    return {"version": version, **(extra or {}), "tasks_checksum": tasks_checksum,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "tickets": tickets,
            "checksum": bench.canonical_checksum(tickets)}


def title_only_tickets(tasks):
    return {t["task_id"]: {"issue_title": t["issue_title"], "issue_body": ""} for t in tasks}


def write_version(recipe, doc):
    out = tickets_dir(recipe) / f"{doc['version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["checksum"] == doc["checksum"] and old["tasks_checksum"] == doc["tasks_checksum"]:
            print(f"unchanged: {out} checksum {old['checksum']}")
            return out
        if not recipe.get("_force"):
            sys.exit(f"{out} exists with a different checksum; a frozen version does not change (use --force only "
                     "before any run of this version, and record why in the changelog)")
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"done: {out}, {len(doc['tickets'])} tickets, checksum {doc['checksum']}")
    return out


def cmd_title_only(args):
    recipe = bench.load_recipe(args.recipe)
    recipe["_force"] = args.force
    m = load_tasks(recipe)
    write_version(recipe, version_file(title_only_tickets(m["tasks"]), "title-only", m["checksum"]))


# ---------------------------------------------------------------------------- the rewriter

class Fatal(Exception):
    pass


def price():
    return json.loads((HERE / "bench" / "harness.json").read_text(encoding="utf-8"))["model"]["price_per_mtok"]


def cost_of(resp):
    """The gateway's own cost figure first; otherwise the usage at the list prices in bench/harness.json."""
    meta = (resp.get("providerMetadata") or {}).get("gateway") or {}
    if meta.get("cost") is not None:
        return float(meta["cost"])
    u = resp.get("usage") or {}
    if u.get("cost") is not None:
        return float(u["cost"])
    p = price()
    tin = u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
    tout = u.get("completion_tokens", u.get("output_tokens", 0)) or 0
    return (tin * p["input"] + tout * p["output"]) / 1e6


def post(body, key, max_tries=10):
    """One chat completion through the gateway, as run.py does it: retries on 429 and 5xx, stops on 401 to 403."""
    data, delay = json.dumps(body).encode(), 2.0
    for attempt in range(1, max_tries + 1):
        req = urllib.request.Request(GATEWAY + "/v1/chat/completions", data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:1500]
            if e.code in (401, 402, 403):
                raise Fatal(f"HTTP {e.code} from the gateway: {detail}")
            if e.code == 429 or e.code >= 500:
                wait = retry_after(e.headers.get("retry-after"), delay)
                print(f"  HTTP {e.code}, waiting {wait:.0f}s (attempt {attempt}/{max_tries})", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            raise Fatal(f"HTTP {e.code} from the gateway: {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error {e}, retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise Fatal("the gateway kept failing; run again later, everything done so far is kept")


def retry_after(header, fallback):
    try:
        return max(0.0, float(header)) if header else fallback
    except ValueError:
        return fallback


def gateway_model(key):
    """The real rewriter: messages in, (text, usage, cost) out."""
    def call(messages):
        resp = post({"model": REWRITER, "max_tokens": MAX_TOKENS, "messages": messages}, key)
        return resp["choices"][0]["message"]["content"] or "", resp.get("usage") or {}, cost_of(resp)
    return call


def issue_message(task, repo):
    return (f"Repository: {repo}\n\nThe issue as filed follows, between the markers.\n\n<issue>\nTitle: "
            f"{task['issue_title']}\n\n{task['issue_body']}\n</issue>")


def retry_message(tokens):
    listed = "\n".join(f"- {t}" for t in tokens)
    return ("Your restatement contains these code-like tokens, which do not occur in the issue as filed:\n\n"
            f"{listed}\n\nWrite the restatement again under the same six headings and the same rules, without "
            "these tokens. Where you need to refer to something the issue does not name, describe it in plain words.")


SECTION = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)


def split_rewrite(text):
    """(title, body): the title is the Summary line, the body everything from the next heading on. None when the
    text has no Summary heading with a sentence under it."""
    heads = list(SECTION.finditer(text))
    for i, h in enumerate(heads):
        if h.group(1).strip().lower() == "summary":
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            summary = " ".join(text[h.end():end].split())
            if not summary:
                return None
            return summary, text[end:].strip()
    return None


def judge(text, task, allow):
    """(ticket or None, offending tokens). A draft without the Summary heading is refused like a leak."""
    parts = split_rewrite(text)
    bad = leaks(text, issue_text(task), allow)
    if parts is None:
        bad = ["(no Summary heading)"] + bad
    return (None if bad else {"issue_title": parts[0], "issue_body": parts[1]}), bad


def rewrite_one(task, repo, rule, model, allow):
    """Rewrite one task's issue: draft, check, at most one retry naming the offending tokens, then the as-filed
    text if the second draft fails too. Returns the log entry, with the final ticket in it."""
    t0 = time.monotonic()
    messages = [{"role": "system", "content": rule}, {"role": "user", "content": issue_message(task, repo)}]
    entry = {"task_id": task["task_id"], "drafts": [], "cost": 0.0}
    for attempt in (1, 2):
        text, usage, cost = model(messages)
        ticket, bad = judge(text, task, allow)
        entry["drafts"].append({"text": text, "leaks": bad, "passed": not bad, "usage": usage, "cost": cost})
        entry["cost"] += cost
        if ticket:
            entry.update(final=ticket, fallback=False, accepted_draft=attempt)
            break
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": retry_message(bad)}]
    else:
        entry.update(final={"issue_title": task["issue_title"], "issue_body": task["issue_body"]}, fallback=True,
                     accepted_draft=None)
    entry["cost"] = round(entry["cost"], 6)
    entry["seconds"] = round(time.monotonic() - t0, 1)
    return entry


def rewrite_all(tasks, repo, rule, model, log_path, spend_cap=SPEND_CAP):
    """Rewrite every task not yet final in the log; restartable. Returns the counts."""
    log = bench.read_jsonl(log_path)
    done = {e["task_id"] for e in log if "final" in e}
    spent = sum(e.get("cost") or 0 for e in log)
    biggest = max([e.get("cost") or 0 for e in log] + [0.25])
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        if t["task_id"] in done:
            continue
        if spent + biggest > spend_cap:
            print(f"stopping: ${spent:.2f} spent, the next ticket could pass the ${spend_cap} cap")
            break
        e = rewrite_one(t, repo, rule, model, allow_list(repo))
        e["rule_sha256"] = hashlib.sha256(rule.encode("utf-8")).hexdigest()
        with Path(log_path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        log.append(e)
        spent += e["cost"]
        biggest = max(biggest, e["cost"])
        verdicts = " ".join("ok" if d["passed"] else f"leak({len(d['leaks'])})" for d in e["drafts"])
        print(f"{t['task_id']}: {verdicts}{' -> as filed' if e['fallback'] else ''} ${e['cost']:.3f} "
              f"{e['seconds']}s (total ${spent:.2f})", flush=True)
    return counts(log)


def counts(log):
    final = [e for e in log if "final" in e]
    return {"tasks": len(final),
            "first_draft_rejected": sum(1 for e in final if not e["drafts"][0]["passed"]),
            "second_draft_rejected": sum(1 for e in final if len(e["drafts"]) > 1 and not e["drafts"][1]["passed"]),
            "fallbacks": sum(1 for e in final if e["fallback"]),
            "cost_total": round(sum(e["cost"] for e in final), 4),
            "seconds_total": round(sum(e["seconds"] for e in final), 1)}


def cmd_rewrite(args, model=None):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    rule = RULE.read_text(encoding="utf-8")
    if model is None:
        key = os.environ.get(KEY_ENV)
        if not key:
            sys.exit(f"set {KEY_ENV} in this shell first")
        model = gateway_model(key)
    log_path = tickets_dir(recipe) / "rewrite-log.jsonl"
    rule_sha = hashlib.sha256(rule.encode("utf-8")).hexdigest()
    old = {e.get("rule_sha256") for e in bench.read_jsonl(log_path)}
    if old - {rule_sha}:
        sys.exit("rewrite-log.jsonl holds rewrites made with a different rule; move it aside before rewriting "
                 "(and record the change of rule in the changelog)")
    try:
        c = rewrite_all(m["tasks"], recipe["repo"], rule, model, log_path, args.spend_cap)
    except Fatal as e:
        sys.exit(str(e))
    print(json.dumps(c, indent=2))
    if c["fallbacks"] > 10:
        print(f"WARNING: {c['fallbacks']} tasks fell back to the as-filed text, more than 10 of 44: stop and tell "
              "Claude before freezing or running")


# ---------------------------------------------------------------------------- check and freeze

def check_tickets(tickets, tasks, repo):
    """Offending tokens per task id, for every ticket that differs from the issue as filed."""
    by_id = {t["task_id"]: t for t in tasks}
    allow, bad = allow_list(repo), {}
    for tid, tk in sorted(tickets.items()):
        task = by_id[tid]
        if tk == {"issue_title": task["issue_title"], "issue_body": task["issue_body"]}:
            continue  # the as-filed text cannot leak
        found = leaks(tk["issue_title"] + "\n" + tk["issue_body"], issue_text(task), allow)
        if found:
            bad[tid] = found
    return bad


def cmd_check(args):
    recipe = bench.load_recipe(args.recipe)
    m = load_tasks(recipe)
    path = Path(args.file) if args.file else tickets_dir(recipe) / f"{args.version}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(doc["tickets"]) - {t["task_id"] for t in m["tasks"]}
    if unknown:
        sys.exit(f"{path} has task ids the manifest does not: {sorted(unknown)}")
    bad = check_tickets(doc["tickets"], m["tasks"], recipe["repo"])
    for tid, toks in bad.items():
        print(f"{tid}: {toks}")
    print(f"{len(bad)} of {len(doc['tickets'])} tickets name tokens the issue does not contain")
    sys.exit(1 if bad else 0)


def freeze_rewritten(tasks, tasks_checksum, log, rule_text, repo):
    """The rewritten version file from the final log entries. Refuses a missing task or a leaking final ticket."""
    final = {e["task_id"]: e["final"] for e in log if "final" in e}
    missing = [t["task_id"] for t in tasks if t["task_id"] not in final]
    if missing:
        raise ValueError(f"{len(missing)} tasks have no final rewrite: {missing[:5]}")
    tickets = {t["task_id"]: final[t["task_id"]] for t in tasks}
    bad = check_tickets(tickets, tasks, repo)
    if bad:
        raise ValueError(f"final tickets fail the leak check: {bad}")
    return version_file(tickets, "rewritten", tasks_checksum,
                        {"rule_sha256": hashlib.sha256(rule_text.encode("utf-8")).hexdigest(),
                         "rewrite": counts(log)})


def cmd_freeze(args):
    recipe = bench.load_recipe(args.recipe)
    recipe["_force"] = args.force
    m = load_tasks(recipe)
    log = bench.read_jsonl(tickets_dir(recipe) / "rewrite-log.jsonl")
    try:
        doc = freeze_rewritten(m["tasks"], m["checksum"], log, RULE.read_text(encoding="utf-8"), recipe["repo"])
    except ValueError as e:
        sys.exit(str(e))
    write_version(recipe, doc)


# ---------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = dict(required=True, help="the repository's recipe, e.g. bench/recipes/pytest-dev__pytest.json")

    p = sub.add_parser("title-only", help="tickets/title-only.json: the title as frozen, no body")
    p.add_argument("--recipe", **rec)
    p.add_argument("--force", action="store_true", help="overwrite a different version file (before any run only)")
    p.set_defaults(func=cmd_title_only)

    p = sub.add_parser("rewrite", help=f"rewrite each issue by bench/rewrite_rule.md (needs {KEY_ENV})")
    p.add_argument("--recipe", **rec)
    p.add_argument("--spend-cap", type=float, default=SPEND_CAP, help=f"dollars for all rewrites (default {SPEND_CAP})")
    p.set_defaults(func=cmd_rewrite)

    p = sub.add_parser("check", help="the leak check over a version file; exit 1 if any ticket fails")
    p.add_argument("--recipe", **rec)
    p.add_argument("--version", default="rewritten")
    p.add_argument("--file", help="a version file elsewhere")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("freeze", help="tickets/rewritten.json from the rewrite log")
    p.add_argument("--recipe", **rec)
    p.add_argument("--force", action="store_true", help="overwrite a different version file (before any run only)")
    p.set_defaults(func=cmd_freeze)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
