#!/usr/bin/env python3
"""Jev vs Claude Haiku 4.5 on two SDLC decisions, through Vercel AI Gateway.

Standard library only. Nothing to install.

  export AI_GATEWAY_API_KEY=...            # never stored in any file
  python3 run.py --smoke                   # 3 items per task per model, to check access and response shapes
  python3 run.py                           # full run; safe to stop and restart, it resumes
  python3 run.py --status                  # progress and spend so far
  python3 run.py --data-dir kit/<repo>/data --results-dir kit/<repo>/results   # the own-repo kit
  (the taxonomy test's own-label tasks are added automatically when the data folder has taxonomy.json)

Every request and response is appended to results/<model>.jsonl, so the analysis can be
re-run later without new calls. The run stops before total spend passes SPEND_CAP_USD.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
BASE_URL = os.environ.get("JEV_STUDY_BASE_URL", "https://ai-gateway.vercel.sh")
SPEND_CAP_USD = 4.50
PAUSE_BETWEEN_CALLS_S = float(os.environ.get("JEV_STUDY_PAUSE", 0.3))

# List prices per million tokens, used when the gateway does not report a cost itself.
PRICES = {
    "typesafe-ai/jev": {"in": 0.042, "out": 0.0},
    "typesafe-ai/jev#checks": {"in": 0.042, "out": 0.0},
    "anthropic/claude-haiku-4.5": {"in": 1.00, "out": 5.00},
    "anthropic/claude-opus-5.5": {"in": 4.00, "out": 20.00},
    "anthropic/claude-opus-5": {"in": 5.00, "out": 25.00},
    "anthropic/claude-opus-4.7": {"in": 5.00, "out": 25.00},
}

# ---------------------------------------------------------------------------
# Pre-registered question wording. Both models get the same text and the same criteria.
# Do not edit after the first full run: changing wording invalidates the comparison.
# ---------------------------------------------------------------------------
ISSUE_INSTRUCTIONS = "What kind of GitHub issue is this?"
ISSUE_CRITERIA = {
    "bug": "The reporter describes something that is broken, crashes, shows an error, or behaves differently from what is documented or expected.",
    "feature": "The reporter asks for new functionality, an enhancement, or a change to existing behaviour.",
    "question": "The reporter asks for help, clarification, or how to do something, rather than reporting a defect or requesting a change.",
}
REVIEW_INSTRUCTIONS = "Would an experienced code reviewer leave a review comment on this code change (the diff hunk shown)?"
REVIEW_CRITERIA = {
    "true": "The change has something worth a comment: a bug, unclear or risky logic, missing handling, poor naming, a style or convention issue, or a better approach.",
    "false": "The change is fine as written and a reviewer would approve it without any comment.",
}

# Own-repo kit only (not part of the benchmark): the pull-request outcome question.
# Changed on 26 September 2026, before any model run on kit data, from "request changes" to
# "comment on or request changes": many teams (HashiCorp among them) comment rather than request changes.
PR_INSTRUCTIONS = "Will a reviewer comment on or request changes to this pull request before it is merged?"
PR_CRITERIA = {
    "true": "The pull request has something a reviewer would comment on or ask to change before merging: a bug, missing tests or error handling, unclear or risky logic, a design or convention problem, or an incomplete description.",
    "false": "The pull request is ready to merge as written, and a reviewer would approve it without any comment.",
}

# Every task, its question and its data file. "choice" picks one criteria key; "yesno" answers yes or no.
QUESTIONS = {
    "issue_type": {"kind": "choice", "instructions": ISSUE_INSTRUCTIONS, "criteria": ISSUE_CRITERIA, "subject": "GitHub issue"},
    "needs_comment": {"kind": "yesno", "instructions": REVIEW_INSTRUCTIONS, "criteria": REVIEW_CRITERIA, "subject": "code change"},
    "pr_needs_changes": {"kind": "yesno", "instructions": PR_INSTRUCTIONS, "criteria": PR_CRITERIA, "subject": "pull request"},
}
TASKS = {
    "issue_type": {"file": "issues.jsonl", "labels": ["bug", "feature", "question"]},
    "needs_comment": {"file": "code_review.jsonl", "labels": ["yes", "no"]},
    "pr_needs_changes": {"file": "prs.jsonl", "labels": ["yes", "no"]},
}


# ---------------------------------------------------------------------------
# Taxonomy test, pre-registered 3 October 2026 (see README.md). When the data folder holds taxonomy.json
# (written by fetch_taxonomy.py), one task per label family is added: its options are the repository's own
# labels, each described by its GitHub description as written, or by its name when it has none.
# The tasks above are untouched, so their requests stay byte-identical.
# ---------------------------------------------------------------------------
def register_taxonomy(data_dir):
    f = data_dir / "taxonomy.json"
    if not f.exists():
        return []
    tax = json.loads(f.read_text(encoding="utf-8"))
    global TAXONOMY_DATA
    TAXONOMY_DATA = True
    # Follow-up tests (pre-registered 4 October 2026): per-family yes/no checks, and the area question.
    for fam, checks in (tax.get("checks") or {}).items():
        TAX_CHECKS[f"own_{fam}"] = {k: (v[0], {"true": v[1], "false": v[2]}) for k, v in checks.items()}
    if tax.get("area"):
        ar = tax["area"]
        AREA[f"own_{ar['for_family']}"] = {"type": "choice", "instructions": ar["instructions"],
                                           "criteria": {o["name"]: o["description"] for o in ar["options"]}}
    added = []
    for fam, spec in tax["families"].items():
        task = f"own_{fam}"
        criteria = {o["name"]: (o["description"] or o["name"]) for o in spec["options"]}
        QUESTIONS[task] = {"kind": "choice", "instructions": tax["instructions"].format(family=fam),
                           "criteria": criteria, "subject": "GitHub issue", "max_tokens": 60}
        TASKS[task] = {"file": f"taxonomy_{fam}.jsonl", "labels": list(criteria)}
        added.append(task)
    return added


def jev_request(item):
    Q = QUESTIONS[item["task"]]
    q = {"type": "choice" if Q["kind"] == "choice" else "boolean", "instructions": Q["instructions"], "criteria": Q["criteria"]}
    order = os.environ.get("JEV_PROVIDER_ORDER", "digitalocean,typesafe-ai").split(",")
    return "/v1/evaluate", {"model": "typesafe-ai/jev", "state": item["text"], "questions": {"answer": q},
                            "providerOptions": {"gateway": {"order": order}}}


def jev_parse(item, resp):
    a = resp["answers"]["answer"]
    if QUESTIONS[item["task"]]["kind"] == "choice":
        probs = a.get("probabilities") or {}
        label = a.get("choice") or max(probs, key=probs.get)
        return label, float(probs.get(label, 0.0)), probs
    p = float(a.get("probability", a.get("noul", 0.0)))
    label = "yes" if p >= 0.5 else "no"
    return label, max(p, 1 - p), {"yes": p, "no": 1 - p}


# ---------------------------------------------------------------------------
# Decomposition test, pre-registered 3 October 2026 before any call (see README.md).
# Six narrow yes/no checks asked of Jev in one request per change; their probabilities are combined by a
# logistic regression fitted on the calibration split (combine_checks.py). Code review tasks only.
# ---------------------------------------------------------------------------
CHECKS = {
    "bug_risk": ("Could this change introduce a bug or break existing behaviour?",
                 {"true": "The change contains logic that looks wrong or risky, or is likely to break something that worked before.",
                  "false": "Nothing in the change looks likely to introduce a bug."}),
    "missing_handling": ("Is this change missing error handling, validation or edge-case handling that it needs?",
                 {"true": "The change has code that can fail or receive unexpected input without handling it.",
                  "false": "Failures and unusual inputs are handled, or the change has none."}),
    "clarity": ("Is any part of this change hard to understand?",
                 {"true": "It has unclear names, confusing logic, or no explanation where a reader would need one.",
                  "false": "The change is easy to follow."}),
    "convention": ("Does this change depart from the code's usual style or conventions?",
                 {"true": "Its formatting, naming or idioms differ from the surrounding code or the norms of the language.",
                  "false": "It follows the surrounding style."}),
    "tests": ("Does this change alter behaviour without adding or updating tests?",
                 {"true": "Behaviour changes and no test is added or updated.",
                  "false": "It adds or updates tests, or does not change behaviour."}),
    "trivial": ("Is this a trivial, low-risk change?",
                 {"true": "A typo or comment fix, version bump, rename, formatting or similar change with no effect on behaviour.",
                  "false": "It changes behaviour or logic."}),
}
CHECK_TASKS = {"needs_comment", "pr_needs_changes"}
TAXONOMY_DATA = False  # set when the data folder holds taxonomy.json
TAX_CHECKS = {}  # own_<family> -> checks from taxonomy.json (follow-up tests, 4 October 2026)
AREA = {}        # own_<family> -> the area question asked before mapping areas to that family's labels


def jev_checks_request(item):
    if item["task"] not in CHECK_TASKS and item["task"] not in TAX_CHECKS:
        raise ValueError("no checks are defined for this task")
    order = os.environ.get("JEV_PROVIDER_ORDER", "digitalocean,typesafe-ai").split(",")
    checks = CHECKS if item["task"] in CHECK_TASKS else TAX_CHECKS[item["task"]]
    qs = {k: {"type": "boolean", "instructions": ins, "criteria": crit} for k, (ins, crit) in checks.items()}
    return "/v1/evaluate", {"model": "typesafe-ai/jev", "state": item["text"], "questions": qs,
                            "providerOptions": {"gateway": {"order": order}}}


def jev_checks_parse(item, resp):
    ans = resp["answers"]
    checks = CHECKS if item["task"] in CHECK_TASKS else TAX_CHECKS[item["task"]]
    probs = {k: float(ans[k].get("probability", ans[k].get("noul"))) for k in checks}
    # Placeholder label only; the combined decision comes from combine_checks.py.
    return None, None, probs


def jev_area_request(item):
    order = os.environ.get("JEV_PROVIDER_ORDER", "digitalocean,typesafe-ai").split(",")
    return "/v1/evaluate", {"model": "typesafe-ai/jev", "state": item["text"], "questions": {"area": AREA[item["task"]]},
                            "providerOptions": {"gateway": {"order": order}}}


def jev_area_parse(item, resp):
    a = resp["answers"]["area"]
    probs = a.get("probabilities") or {}
    label = a.get("choice") or max(probs, key=probs.get)
    return label, float(probs.get(label, 0.0)), probs


def haiku_request(item, model_id="anthropic/claude-haiku-4.5"):
    Q = QUESTIONS[item["task"]]
    subject = Q["subject"]
    if Q["kind"] == "choice":
        crit = "\n".join(f"- {k}: {v}" for k, v in Q["criteria"].items())
        ask = f"{Q['instructions']}\n\nOptions:\n{crit}\n\nThe label must be one of: {', '.join(Q['criteria'])}."
    else:
        ask = (f"{Q['instructions']}\n\nOptions:\n- yes: {Q['criteria']['true']}\n- no: {Q['criteria']['false']}"
               "\n\nThe label must be one of: yes, no.")
    system = ("You are a careful software engineering classifier. Reply with only a JSON object and nothing else, "
              'in the form {"label": "<label>", "confidence": <integer 0-100>}, where confidence is your '
              "probability, in percent, that the label is correct.")
    user = f"{ask}\n\n<{subject}>\n{item['text']}\n</{subject}>"
    return "/v1/chat/completions", {
        "model": model_id, "temperature": 0, "max_tokens": Q.get("max_tokens", 40),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }


def haiku_parse(item, resp):
    text = resp["choices"][0]["message"]["content"].strip()
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start:end + 1])
    label = str(obj["label"]).strip().lower()
    allowed = {x.lower(): x for x in TASKS[item["task"]]["labels"]}  # own labels keep their case, e.g. Type:Bug
    if label not in allowed:
        raise ValueError(f"label outside the allowed set: {label!r}")
    label = allowed[label]
    conf = min(max(float(obj.get("confidence", 0)) / 100.0, 0.0), 1.0)
    return label, conf, None


# Which Opus to use. Override with OPUS_MODEL=anthropic/claude-opus-5 (or -4.7) if 5.5 is not available.
OPUS_MODEL = os.environ.get("OPUS_MODEL", "anthropic/claude-opus-5.5")


def opus_request(item):
    path, body = haiku_request(item, OPUS_MODEL)
    body["reasoning"] = {"enabled": False}
    # Taxonomy test follow-up, 4 October 2026, after the smoke test and before the full run: Opus 5.5 often writes a
    # sentence before the JSON despite the instruction, and 60 tokens cut it off before the answer (6 of 12 smoke
    # items). The prompt is unchanged; Opus gets room to finish, and opus_parse reads the JSON after any prose.
    if TAXONOMY_DATA:  # only for taxonomy-test data, so earlier Opus requests are unchanged
        body["max_tokens"] = int(os.environ.get("OPUS_MAX_TOKENS", 400))
    return path, body


def opus_parse(item, resp):
    try:
        return haiku_parse(item, resp)
    except Exception:
        text = resp["choices"][0]["message"]["content"]
        start = text.rfind('{"label"')
        if start < 0:
            raise
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        fake = {"choices": [{"message": {"content": json.dumps(obj)}}]}
        return haiku_parse(item, fake)


MODELS = {
    "jev": {"id": "typesafe-ai/jev", "request": jev_request, "parse": jev_parse},
    "haiku": {"id": "anthropic/claude-haiku-4.5", "request": haiku_request, "parse": haiku_parse},
    # Exploratory arm, added 26 September 2026 after the Haiku results: same prompt, stronger model, on a sample.
    # Reasoning is switched off so Opus answers like Haiku (Haiku 4.5 does not reason unless asked).
    "opus": {"id": OPUS_MODEL, "request": lambda item: opus_request(item), "parse": opus_parse},
    # Decomposition test (3 October 2026): raw check probabilities, kept in results/checks/ so analyze.py ignores them.
    "jev_checks": {"id": "typesafe-ai/jev#checks", "request": jev_checks_request, "parse": jev_checks_parse},
    # Follow-up (4 October 2026): the area question, step one of the two-step team decision; kept in results/checks/.
    "jev_area": {"id": "typesafe-ai/jev#area", "request": jev_area_request, "parse": jev_area_parse},
}


def applies(model, item):
    if model == "jev_checks":
        return item["task"] in CHECK_TASKS or item["task"] in TAX_CHECKS
    if model == "jev_area":
        return item["task"] in AREA
    return True


def result_path(model):
    return RESULTS / "checks" / f"{model}.jsonl" if model in ("jev_checks", "jev_area") else RESULTS / f"{model}.jsonl"


def cost_of(model_id, resp):
    meta = ((resp.get("providerMetadata") or {}).get("gateway") or {})
    if meta.get("cost") is not None:
        return float(meta["cost"])
    u = resp.get("usage") or {}
    tin = u.get("inputTokens", u.get("prompt_tokens", u.get("input_tokens", 0))) or 0
    tout = u.get("outputTokens", u.get("completion_tokens", u.get("output_tokens", 0))) or 0
    p = PRICES[model_id]
    return (tin * p["in"] + tout * p["out"]) / 1e6


class Fatal(Exception):
    pass


def retry_after_s(header, fallback):
    """Honour retry-after as seconds or an HTTP date; fall back to exponential backoff otherwise."""
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                return max(0.0, parsedate_to_datetime(header).timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    return fallback


PATIENT = False  # set by --patient: never give up on rate limits, wait up to 10 minutes between tries


def post(path, body, key, max_tries=14):
    data = json.dumps(body).encode()
    delay = 2.0
    if PATIENT:
        max_tries = 10 ** 6
    attempt = 0
    while attempt < max_tries:
        attempt += 1
        req = urllib.request.Request(BASE_URL + path, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode()), (time.monotonic() - t0) * 1000
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            if e.code in (401, 402, 403):
                raise Fatal(f"HTTP {e.code} from the gateway: {detail}")
            if e.code == 429 or e.code >= 500:
                wait = retry_after_s(e.headers.get("retry-after"), delay)
                shown = "patient mode" if PATIENT else f"attempt {attempt}/{max_tries}"
                msg = f"  {time.strftime('%H:%M:%S')} HTTP {e.code}, waiting {wait:.0f}s ({shown})"
                if attempt == 1:
                    msg += f"  gateway said: {detail[:1500]}"
                print(msg, flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 600 if PATIENT else 300)
                continue
            return {"_http_error": e.code, "_detail": detail}, (time.monotonic() - t0) * 1000
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error {e}, retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise Fatal("still rate limited after repeated retries. Run again later, or add --patient to keep waiting; "
                "everything done so far is kept")


def load_items(tasks):
    items = []
    for t in tasks:
        f = DATA / TASKS[t]["file"]
        if not f.exists():
            print(f"skipping {t}: {f.name} not found in {f.parent.name}")
            continue
        items += [json.loads(l) for l in f.open(encoding="utf-8")]
    return items


RETRY_ERRORS = False


def done_ids(model):
    f = result_path(model)
    if not f.exists():
        return set(), 0.0
    ids, spend = set(), 0.0
    for line in f.open(encoding="utf-8"):
        r = json.loads(line)
        spend += r.get("cost_usd") or 0.0
        same_model = r.get("model", MODELS[model]["id"]) == MODELS[model]["id"]  # e.g. a different Opus version
        if same_model and (r.get("ok") or (r.get("final") and not RETRY_ERRORS)):
            ids.add(r["id"])
    return ids, spend


def total_spend():
    return sum(done_ids(m)[1] for m in MODELS)


def status(tasks):
    items = load_items(tasks)
    for m in MODELS:
        ids, spend = done_ids(m)
        n = sum(1 for i in items if i["id"] in ids)
        print(f"{m:6s} {n:5d}/{len(items)} done, ${spend:.4f} spent")
    print(f"total spend ${total_spend():.4f} of ${SPEND_CAP_USD:.2f} cap")


def run(models, tasks, smoke):
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        sys.exit("Set AI_GATEWAY_API_KEY first (export AI_GATEWAY_API_KEY=...). It is never written to disk.")
    RESULTS.mkdir(exist_ok=True)
    items = load_items(tasks)
    if smoke:
        picked = []
        for t in tasks:
            picked += [i for i in items if i["task"] == t and all(applies(m, i) for m in models)][:3]
        items = picked
    spend = total_spend()
    for m in models:
        spec = MODELS[m]
        ids, _ = done_ids(m)
        todo = [i for i in items if i["id"] not in ids and applies(m, i)]
        print(f"\n{m}: {len(todo)} to do ({len(ids)} already done)", flush=True)
        result_path(m).parent.mkdir(parents=True, exist_ok=True)
        out = result_path(m).open("a", encoding="utf-8")
        for n, item in enumerate(todo, 1):
            if spend >= SPEND_CAP_USD:
                print(f"Spend cap of ${SPEND_CAP_USD:.2f} reached, stopping.")
                return
            path, body = spec["request"](item)
            resp, ms = post(path, body, key)
            rec = {"id": item["id"], "task": item["task"], "split": item["split"], "gold": item["label"],
                   "model": spec["id"], "latency_ms": round(ms, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if "_http_error" in resp:
                rec.update(ok=False, final=True, error=f"HTTP {resp['_http_error']}: {resp['_detail']}", cost_usd=0.0)
            else:
                rec["cost_usd"] = cost_of(spec["id"], resp)
                rec["served_model"] = resp.get("model")
                meta = resp.get("providerMetadata") or {}
                rec["provider"] = ((meta.get("gateway") or {}).get("routing") or {}).get("finalProvider")
                rec["vendor_confidence"] = ((meta.get("typesafe") or {}).get("confidence") or {}).get("answer")
                try:
                    label, conf, probs = spec["parse"](item, resp)
                    rec.update(ok=True, pred=label, confidence=conf, probs=probs)
                except Exception as e:  # unparseable answers are kept and scored as wrong
                    rec.update(ok=False, final=True, error=f"parse: {e}")
                rec["raw"] = resp
                spend += rec["cost_usd"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if smoke:
                print(f"  {item['id']}: gold={item['label']} pred={rec.get('pred')} conf={rec.get('confidence')} "
                      f"{ms:.0f}ms ${rec.get('cost_usd', 0):.6f} via {rec.get('provider')} {rec.get('error', '')}")
            elif n % 50 == 0:
                print(f"  {n}/{len(todo)}  spend so far ${spend:.4f}", flush=True)
            time.sleep(PAUSE_BETWEEN_CALLS_S)
        out.close()
    print(f"\nDone. Total spend ${spend:.4f}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 items per task per model")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="re-send items that failed with an HTTP or parse error")
    ap.add_argument("--patient", action="store_true", help="never give up on rate limits; for leaving Jev running overnight")
    ap.add_argument("--data-dir", default="data", help="folder with the task files, relative to this script (default: data)")
    ap.add_argument("--results-dir", default="results", help="where results go, relative to this script (default: results)")
    ap.add_argument("--model", choices=["jev", "haiku", "opus", "jev_checks", "jev_area", "all"], default="all",
                    help="all = jev and haiku; jev_checks = the decomposition checks; jev_area = the area question (taxonomy follow-up)")
    ap.add_argument("--task", default="all", help="one task name, or all (own_<family> tasks exist when the data has taxonomy.json)")
    ap.add_argument("--spend-cap", type=float, default=SPEND_CAP_USD, help=f"stop before total spend in the results folder passes this (default ${SPEND_CAP_USD:.2f})")
    a = ap.parse_args()
    RETRY_ERRORS = a.retry_errors
    PATIENT = a.patient
    SPEND_CAP_USD = a.spend_cap
    DATA, RESULTS = HERE / a.data_dir, HERE / a.results_dir
    register_taxonomy(DATA)
    if a.task != "all" and a.task not in TASKS:
        sys.exit(f"unknown task {a.task!r}; available here: {', '.join(TASKS)}")
    tasks = list(TASKS) if a.task == "all" else [a.task]
    models = ["jev", "haiku"] if a.model == "all" else [a.model]
    try:
        status(tasks) if a.status else run(models, tasks, a.smoke)
    except Fatal as e:
        sys.exit(f"\nStopped: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Run the same command again to resume.")
