#!/usr/bin/env python3
"""Which tickets can you hand to an agent? Standard library only.

Sorts tickets, from the ticket alone and before any work starts, into an agent lane (an agent will
probably land this, within an error budget) and the rest. Ground truth is your own history: pull requests
an agent opened for an issue, and whether they landed. Method and rules: the agent lane study spec.

Outcome levels, each stricter than the last:
  merged        the pull request was merged
  landed clean  merged, no review requested changes, and no commit by a person other than the agent
                (merges of the base branch and other bot accounts are neutral)
  landed, held  landed clean, and no fixing commit within 30 days on the files it changed (from GitHub)

  # build the table from your own repository (needs GITHUB_TOKEN)
  python3 agent_lane.py own-table --repo OWNER/REPO --agent-logins "Copilot" --since 2026-01-01 \\
      --config examples/aidev-agent-lane-config.json --out kit/OWNER__REPO/agent-table.jsonl
  # or from the AIDev v3 extracts
  python3 agent_lane.py aidev-table --extracts DIR --config examples/aidev-agent-lane-config.json --out kit/aidev/table.jsonl
  # score the tickets (needs AI_GATEWAY_API_KEY), find the lanes, report
  python3 agent_lane.py score --table kit/aidev/table.jsonl --config CFG --out kit/aidev/scores.jsonl --smoke
  python3 agent_lane.py score --table kit/aidev/table.jsonl --config CFG --out kit/aidev/scores.jsonl
  python3 agent_lane.py lanes --table kit/aidev/table.jsonl --scores kit/aidev/scores.jsonl --config CFG --out kit/aidev/lanes.json
  python3 agent_lane.py report --lanes kit/aidev/lanes.json --out kit/aidev/report.md
"""
import argparse, collections, datetime, json, os, random, re, sys, time, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import combine_checks  # noqa: E402
import error_budget  # noqa: E402

LEVELS = ("merged", "landed_clean")
SCORES = ("A", "B", "C", "D")
SCORE_NAMES = {"A": "trivial rule (documentation yes)", "B": "task type alone",
               "C": "holistic probability", "D": "task type plus checks"}
FOLDS = 5


# ---------------------------------------------------------------------------- io helpers

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def rnd(x):
    return None if x is None else round(x, 4)


def clean_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {str(k): clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v) for v in obj]
    return obj


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean_floats(obj), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def ts(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- outcomes

def stars_bucket(stars, buckets):
    lo, hi = buckets
    if stars is None:
        return "unknown"
    if stars < lo:
        return f"under {lo // 1000}k"
    if stars <= hi:
        return f"{lo // 1000}k-{hi // 1000}k"
    return f"over {hi // 1000}k"


def commit_kind(author, message, agent_cfg, cfg):
    """agent, neutral (base-branch merge or another bot account) or human."""
    author, message = author or "", message or ""
    if author in agent_cfg.get("agent_logins", []):
        return "agent"
    marker = agent_cfg.get("commit_marker")
    if marker and re.search(marker, message, re.I | re.M):
        return "agent"
    if re.search(cfg["neutral_commit"], message):
        return "neutral"
    if re.search(cfg["neutral_bots"], author):
        return "neutral"
    return "human"


def outcomes(merged_at, commits, review_states, agent_cfg, cfg):
    """commits: list of (author, message). Returns dict with merged, landed_clean (None if not assessable), notes."""
    merged = merged_at is not None
    out = {"merged": merged, "landed_clean": None, "changes_requested": None, "human_commit": None, "clean_note": None}
    if not agent_cfg.get("clean_assessable"):
        out["clean_note"] = "agent not assessable at landed clean"
        return out
    if not commits:
        out["clean_note"] = "no commits recorded"
        return out
    out["changes_requested"] = any(s == "CHANGES_REQUESTED" for s in review_states)
    out["human_commit"] = any(commit_kind(a, m, agent_cfg, cfg) == "human" for a, m in commits)
    out["landed_clean"] = merged and not out["changes_requested"] and not out["human_commit"]
    return out


def assign_split(rows, share):
    rows.sort(key=lambda r: (r["created_at"], str(r["pr_id"])))
    n_cal = int(round(share * len(rows)))
    for i, r in enumerate(rows):
        r["split"] = "calibration" if i < n_cal else "evaluation"
    return rows[n_cal]["created_at"] if n_cal < len(rows) else None


# ---------------------------------------------------------------------------- aidev-table

def aidev_table(args):
    cfg = load_config(args.config)
    X = Path(args.extracts)
    prs = read_jsonl(X / "pull_request.jsonl")
    issues = {i["id"]: i for i in read_jsonl(X / "issue.jsonl")}
    repos = {r["id"]: r for r in read_jsonl(X / "repository.jsonl")}
    links = collections.defaultdict(set)
    for l in read_jsonl(X / "related_issue.jsonl"):
        links[l["pr_id"]].add(l["issue_id"])
    commits = collections.defaultdict(list)
    for c in read_jsonl(X / "pr_commits.jsonl"):
        commits[c["pr_id"]].append(c)
    reviews = collections.defaultdict(list)
    for r in read_jsonl(X / "pr_reviews.jsonl"):
        reviews[r["pr_id"]].append(r.get("state"))
    types = {t["id"]: t.get("type") for t in read_jsonl(X / "pr_task_type.jsonl")}
    agent_logins = {lg for a in cfg["agents"].values() for lg in a["agent_logins"]}
    latest = ts(cfg["cutoff"] + "T00:00:00Z") - datetime.timedelta(days=cfg["min_age_days"])

    drops = collections.Counter()
    drops_by_agent = collections.defaultdict(collections.Counter)
    rows = []
    for p in sorted(prs, key=lambda p: p["id"]):
        agent = p.get("agent")
        reason = None
        iss = None
        if agent not in cfg["agents"]:
            reason = "agent not in config"
        elif p["id"] not in links:
            reason = "no related issue"
        elif len(links[p["id"]]) > 1:
            reason = "several related issues"
        else:
            iss = issues.get(next(iter(links[p["id"]])))
            if iss is None:
                reason = "issue not mined"
            elif not (iss.get("created_at") and iss["created_at"] < p["created_at"]):
                reason = "issue not before pull request"
            elif (iss.get("user") or "") in agent_logins or re.search(cfg["neutral_bots"], iss.get("user") or ""):
                reason = "issue by an agent or bot account"
            elif len((iss.get("body") or "").strip()) < cfg["min_body_chars"]:
                reason = f"issue body under {cfg['min_body_chars']} characters"
            elif ts(p["created_at"]) > latest:
                reason = f"pull request created less than {cfg['min_age_days']} days before the cutoff"
        if reason:
            drops[reason] += 1
            drops_by_agent[agent][reason] += 1
            continue
        a_cfg = cfg["agents"][agent]
        cs = commits.get(p["id"], [])
        o = outcomes(p.get("merged_at"), [(c.get("author"), c.get("message")) for c in cs],
                     reviews.get(p["id"], []), a_cfg, cfg)
        repo = repos.get(p.get("repo_id")) or {}
        full = repo.get("full_name") or "/".join((p.get("repo_url") or "").rstrip("/").split("/")[-2:])
        ta = types.get(p["id"])
        rows.append(dict(o, pr_id=p["id"], agent=agent, repo=full, number=p.get("number"),
                         html_url=p.get("html_url"), created_at=p["created_at"], merged_at=p.get("merged_at"),
                         state=p.get("state"), issue_id=iss["id"], issue_number=iss.get("number"),
                         issue_created_at=iss.get("created_at"), issue_title=iss.get("title") or "",
                         issue_body=iss.get("body") or "", language=repo.get("language"),
                         stars_bucket=stars_bucket(repo.get("stars"), cfg["star_buckets"]),
                         type_after=None if ta is None else cfg["type_after_map"].get(str(ta).lower(), "other"),
                         commit_shas=sorted(c["sha"] for c in cs if c.get("sha"))))
    cut = assign_split(rows, cfg["calibration_share"])
    write_jsonl(args.out, rows)
    manifest = table_manifest(rows, cut, drops, drops_by_agent, len(prs))
    write_json(Path(args.out).with_name("manifest.json"), manifest)
    print_manifest(manifest)
    if args.sample:
        print_sample(rows, args.sample, cfg["seed"])


def table_manifest(rows, cut, drops, drops_by_agent, total):
    per_agent = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        c = per_agent[r["agent"]]
        c["kept"] += 1
        c[r["split"]] += 1
        c["merged"] += r["merged"]
        if r["landed_clean"] is not None:
            c["clean_assessable"] += 1
            c["landed_clean"] += r["landed_clean"]
    cells = collections.Counter((r["split"], r.get("type_after") or "unknown") for r in rows)
    return {"pull_requests_read": total, "kept": len(rows), "cut_date": cut,
            "dropped": dict(drops), "dropped_by_agent": {a: dict(c) for a, c in drops_by_agent.items()},
            "per_agent": {a: dict(c) for a, c in per_agent.items()},
            "type_after_cells": {f"{s}/{t}": n for (s, t), n in sorted(cells.items())}}


def print_manifest(m):
    print(f"read {m['pull_requests_read']}, kept {m['kept']}, cut date {m['cut_date']}")
    for k, v in sorted(m["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  dropped {v:6d}  {k}")
    for a, c in sorted(m["per_agent"].items()):
        print(f"  {a}: {c}")


def print_sample(rows, n, seed):
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        lvl = "landed clean" if r["landed_clean"] else ("merged" if r["merged"] else "not merged")
        print(f"  {r['html_url']}  {lvl}  (changes requested: {r['changes_requested']}, "
              f"human commit: {r['human_commit']}, note: {r['clean_note']})")


# ---------------------------------------------------------------------------- own-table

CLOSING = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b", re.I)


def own_table(args):
    import fetch_repo
    cfg = load_config(args.config)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN set: GitHub allows 60 requests an hour without one.", file=sys.stderr)
    owner_repo = args.repo
    gh = fetch_repo.GitHub(token=token, cache_dir=HERE / "kit" / owner_repo.replace("/", "__") / "cache")
    logins = [x.strip() for x in (args.agent_logins or "").split(",") if x.strip()]
    a_cfg = {"agent_logins": logins, "commit_marker": args.commit_marker,
             "clean_assessable": bool(logins or args.commit_marker)}
    info = gh.get(f"/repos/{owner_repo}")
    latest = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=cfg["min_age_days"])
    drops, rows = collections.Counter(), []
    for p in gh.pages(f"/repos/{owner_repo}/pulls", {"state": "all", "sort": "created", "direction": "desc"},
                      args.max_pages):
        if p["created_at"] < args.since:
            break
        body = p.get("body") or ""
        by_agent = (p.get("user") or {}).get("login") in logins
        if not (by_agent or (args.body_marker and re.search(args.body_marker, body, re.I))):
            continue
        refs = sorted(set(CLOSING.findall(body)))
        if not refs:
            drops["no closing reference"] += 1
            continue
        if len(refs) > 1:
            drops["several closing references"] += 1
            continue
        try:
            iss = gh.get(f"/repos/{owner_repo}/issues/{refs[0]}")
        except urllib.error.HTTPError:
            drops["issue not found"] += 1
            continue
        if "pull_request" in iss:
            drops["reference is a pull request"] += 1
            continue
        if not iss["created_at"] < p["created_at"]:
            drops["issue not before pull request"] += 1
            continue
        if len((iss.get("body") or "").strip()) < cfg["min_body_chars"]:
            drops["issue body too short"] += 1
            continue
        if ts(p["created_at"]) > latest:
            drops["pull request too recent"] += 1
            continue
        n = p["number"]
        cs = list(gh.pages(f"/repos/{owner_repo}/pulls/{n}/commits", {}, 3))
        rv = list(gh.pages(f"/repos/{owner_repo}/pulls/{n}/reviews", {}, 3))
        pairs = [((c.get("author") or {}).get("login") or c["commit"]["author"].get("name"), c["commit"]["message"])
                 for c in cs]
        o = outcomes(p.get("merged_at"), pairs, [r.get("state") for r in rv], a_cfg, cfg)
        rows.append(dict(o, pr_id=p["id"], agent="own", repo=owner_repo, number=n, html_url=p["html_url"],
                         created_at=p["created_at"], merged_at=p.get("merged_at"), state=p["state"],
                         issue_id=iss["id"], issue_number=iss["number"], issue_created_at=iss["created_at"],
                         issue_title=iss.get("title") or "", issue_body=iss.get("body") or "",
                         language=info.get("language"),
                         stars_bucket=stars_bucket(info.get("stargazers_count"), cfg["star_buckets"]),
                         type_after=None, commit_shas=sorted(c["sha"] for c in cs)))
    cut = assign_split(rows, cfg["calibration_share"])
    write_jsonl(args.out, rows)
    m = table_manifest(rows, cut, drops, {"own": drops}, len(rows) + sum(drops.values()))
    m["github"] = gh.snapshot()
    write_json(Path(args.out).with_name("agent-manifest.json"), m)
    print_manifest(m)


# ---------------------------------------------------------------------------- score

def score_request(cfg, item):
    checks = "\n".join(f"- {k}: {q}" for k, q in cfg["checks"])
    issue = ((item.get("title") or "") + "\n\n" + (item.get("body") or ""))[:cfg["max_issue_chars"]]
    user = cfg["user_template"].format(task_types=", ".join(cfg["task_types"]), checks=checks,
                                       language=item.get("language") or "unknown",
                                       stars_bucket=item.get("stars_bucket") or "unknown", issue=issue)
    return {"model": cfg["model_id"], "temperature": 0, "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": user}]}


def score_parse(cfg, text):
    obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
    tt = str(obj["task_type"]).strip().lower()
    if tt not in cfg["task_types"]:
        raise ValueError(f"task type outside the list: {tt!r}")
    checks = {}
    for k, _ in cfg["checks"]:
        v = str(obj["checks"][k]).strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"check {k} not yes or no: {v!r}")
        checks[k] = v == "yes"
    p = min(max(int(round(float(obj["p_land"]))), 0), 100)
    return {"task_type": tt, "checks": checks, "p_land": p}


def score_items(cfg, items, out_path, cap, key, retry_errors=False):
    """Every answer is kept with the model's raw reply, so replies can be re-read without new calls (`rescore`).
    Spend counts every record ever written, including superseded error records, so the cap is never exceeded."""
    import run
    done, spent = {}, 0.0
    if Path(out_path).exists():
        for r in read_jsonl(out_path):
            spent += r.get("cost", 0.0)
            if str(r["issue_id"]) in done and "error" not in done[str(r["issue_id"])]:
                continue  # a good answer is never replaced by a later record
            done[str(r["issue_id"])] = r
    todo = [it for it in items if str(it["issue_id"]) not in done
            or (retry_errors and "error" in done[str(it["issue_id"])])]
    print(f"{len(done)} already scored (${spent:.2f}), {len(todo)} to go, cap ${cap:.2f}")
    est = 0.003
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            if spent + est > cap:
                print(f"stopping before the spend cap: ${spent:.2f} spent")
                break
            rec, cost = {"issue_id": it["issue_id"], "model": cfg["model_id"]}, 0.0
            for attempt in (1, 2):
                resp, _ = run.post("/v1/chat/completions", score_request(cfg, it), key)
                if "_http_error" in resp:
                    rec["error"] = f"HTTP {resp['_http_error']}: {resp['_detail'][:200]}"
                    break
                cost += run.cost_of(cfg["model_id"], resp)
                try:
                    rec.setdefault("raw", []).append(resp["choices"][0]["message"]["content"])
                    rec.update(score_parse(cfg, rec["raw"][-1]))
                    rec.pop("error", None)
                    break
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                    rec["error"] = f"unparseable after attempt {attempt}: {e}"
            rec["cost"] = cost
            spent += cost
            est = max(est, cost)
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            done[str(it["issue_id"])] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} scored, ${spent:.2f}", flush=True)
            time.sleep(run.PAUSE_BETWEEN_CALLS_S)
    errors = sum(1 for r in done.values() if "error" in r)
    print(f"done: {len(done)} scored, {errors} with errors, ${spent:.2f} spent")
    return done


def score(args):
    cfg = load_config(args.config)
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        sys.exit("Set AI_GATEWAY_API_KEY first (export AI_GATEWAY_API_KEY=...). It is never written to disk.")
    rows = read_jsonl(args.table)
    items, seen = [], set()
    for r in rows:
        if r["issue_id"] in seen:
            continue
        seen.add(r["issue_id"])
        items.append({"issue_id": r["issue_id"], "title": r["issue_title"], "body": r["issue_body"],
                      "language": r["language"], "stars_bucket": r["stars_bucket"]})
    if args.smoke:
        fx = score_items(cfg, cfg["fixture_issues"], Path(args.out).with_name("scores-fixture.jsonl"), 1.0, key)
        for it in cfg["fixture_issues"]:
            got = fx.get(str(it["issue_id"]), {})
            print(f"  {it['issue_id']}: expected {it['expect']}, got {got.get('task_type')} "
                  f"p_land={got.get('p_land')} checks={got.get('checks')} {got.get('error', '')}")
        items = random.Random(cfg["seed"]).sample(items, min(20, len(items)))
    score_items(cfg, items, args.out, args.spend_cap, key, retry_errors=args.retry_errors)


def rescore(args):
    """Re-read stored raw replies with the current parser; no model calls. Writes a new file."""
    cfg = load_config(args.config)
    out, fixed = [], 0
    for r in read_jsonl(args.scores):
        if r.get("raw"):
            r = {k: v for k, v in r.items() if k not in ("task_type", "checks", "p_land", "error")}
            try:
                r.update(score_parse(cfg, r["raw"][-1]))
                fixed += 1
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                r["error"] = f"unparseable on re-read: {e}"
        out.append(r)
    write_jsonl(args.out, out)
    print(f"re-read {fixed} replies into {args.out}; no calls made")


# ---------------------------------------------------------------------------- lanes

def fit_grouped(X, y):
    """combine_checks.fit (same standardisation, L2, learning rate and iterations) on rows grouped by identical
    features and outcome, weighted by count. Mathematically the same fit, much faster for binary features."""
    groups = collections.Counter((tuple(x), t) for x, t in zip(X, y))
    keys = sorted(groups)
    W = [groups[k] for k in keys]
    n, d = sum(W), len(X[0])
    mu = [sum(w * k[0][j] for k, w in zip(keys, W)) / n for j in range(d)]
    sd = [max(1e-6, (sum(w * (k[0][j] - mu[j]) ** 2 for k, w in zip(keys, W)) / n) ** 0.5) for j in range(d)]
    Z = [[(k[0][j] - mu[j]) / sd[j] for j in range(d)] for k in keys]
    T = [k[1] for k in keys]
    w_, b = [0.0] * d, 0.0
    L2, LR, sig = combine_checks.L2, combine_checks.LR, combine_checks.sigmoid
    for _ in range(combine_checks.ITERS):
        gw, gb = [L2 * wj / n for wj in w_], 0.0
        for z, t, c in zip(Z, T, W):
            e = (sig(b + sum(wj * zj for wj, zj in zip(w_, z))) - t) * c
            gb += e / n
            for j in range(d):
                gw[j] += e * z[j] / n
        b -= LR * gb
        w_ = [wj - LR * g for wj, g in zip(w_, gw)]
    return lambda r: sig(b + sum(wj * (r[j] - mu[j]) / sd[j] for j, wj in enumerate(w_)))


_ALLOWED = {}


def allowed_table(n, budget):
    if (n, budget) not in _ALLOWED:
        _ALLOWED[(n, budget)] = error_budget.max_wrong_table(max(n, 1), budget)
    return _ALLOWED[(n, budget)]


def choose_lane(pairs, budget):
    """pairs: (score, landed 0/1). Safety-margin rule from error_budget: the largest share, from the top score down,
    cut only between distinct scores, at least MIN_KEPT kept, whose errors fit the one-sided 90% bound."""
    items = sorted(pairs, key=lambda p: -p[0])
    allowed = allowed_table(len(items), budget)
    best, wrong = None, 0
    for k, (s, y) in enumerate(items, 1):
        wrong += 1 - y
        if k < len(items) and items[k][0] == s:
            continue
        if k >= error_budget.MIN_KEPT and wrong <= allowed[k]:
            best = {"kept": k, "wrong": wrong, "threshold": s}
    return best


def curve(pairs):
    items = sorted(pairs, key=lambda p: -p[0])
    pts, wrong = [], 0
    for k, (s, y) in enumerate(items, 1):
        wrong += 1 - y
        if k < len(items) and items[k][0] == s:
            continue
        pts.append((k / len(items), wrong / k))
    out = []
    for q in range(1, 101):
        ok = [e for c, e in pts if c <= q / 100 + 1e-9]
        out.append([q, ok[-1] if ok else None])
    return out


def boot_ratio(groups, draws, seed):
    """groups: {repo: (num, den)}. Percentile 95% interval of sum(num)/sum(den) over repositories resampled."""
    repos = sorted(groups)
    if not repos:
        return None, None
    rng, vals = random.Random(seed), []
    for _ in range(draws):
        num = den = 0
        for r in rng.choices(repos, k=len(repos)):
            a, b = groups[r]
            num += a
            den += b
        if den:
            vals.append(num / den)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def rate_cell(rows, key, draws, seed):
    vals = [r for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    g = collections.defaultdict(lambda: [0, 0])
    for r in vals:
        g[r["repo"]][0] += int(r[key])
        g[r["repo"]][1] += 1
    lo, hi = boot_ratio({k: tuple(v) for k, v in g.items()}, draws, seed)
    return {"n": len(vals), "rate": sum(int(r[key]) for r in vals) / len(vals), "lo": lo, "hi": hi}


def features(row, cfg):
    tts = cfg["task_types"][:-1]
    return [1.0 if row["type_before"] == t else 0.0 for t in tts] + \
           [1.0 if row["checks"][k] else 0.0 for k, _ in cfg["checks"]]


def build_scores(rows, cfg):
    """Adds row['s'] = {A, B, C, D}. Fitted on calibration rows with an outcome only."""
    cal = [r for r in rows if r["split"] == "calibration" and r["y"] is not None]
    by_type = collections.defaultdict(list)
    for r in cal:
        by_type[r["type_before"]].append(r["y"])
    overall = sum(r["y"] for r in cal) / len(cal) if cal else 0.0
    rate = {t: sum(v) / len(v) for t, v in by_type.items()}
    for r in rows:
        r["s"] = {"A": 1.0 if r["type_before"] == "documentation" else 0.0,
                  "B": rate.get(r["type_before"], overall), "C": r["p_land"] / 100.0}
    if len(cal) >= FOLDS * 2 and len({r["y"] for r in cal}) == 2:
        X = [features(r, cfg) for r in cal]
        y = [r["y"] for r in cal]
        idx = list(range(len(cal)))
        random.Random(cfg["seed"]).shuffle(idx)
        for f in range(FOLDS):
            test = set(idx[f::FOLDS])
            pred = fit_grouped([X[i] for i in idx if i not in test], [y[i] for i in idx if i not in test])
            for i in test:
                cal[i]["s"]["D"] = pred(X[i])
        full = fit_grouped(X, y)
        for r in rows:
            if "D" not in r["s"]:
                r["s"]["D"] = full(features(r, cfg))
    else:
        for r in rows:
            r["s"]["D"] = r["s"]["B"]


def lane_metrics(cal, ev, sc, budget, draws, seed):
    pairs_cal = [(r["s"][sc], r["y"]) for r in cal]
    lane = choose_lane(pairs_cal, budget)
    out = {"opened": lane is not None, "cal_n": len(cal), "eval_n": len(ev)}
    if lane:
        t = lane["threshold"]
        kept = [r for r in ev if r["s"][sc] >= t]
        g_cov = collections.defaultdict(lambda: [0, 0])
        g_err = collections.defaultdict(lambda: [0, 0])
        for r in ev:
            g_cov[r["repo"]][1] += 1
            if r["s"][sc] >= t:
                g_cov[r["repo"]][0] += 1
                g_err[r["repo"]][0] += 1 - r["y"]
                g_err[r["repo"]][1] += 1
        out.update(threshold=t, cal_coverage=lane["kept"] / len(cal), cal_error=lane["wrong"] / lane["kept"],
                   eval_kept=len(kept), eval_coverage=len(kept) / len(ev) if ev else None,
                   eval_error=(sum(1 - r["y"] for r in kept) / len(kept)) if kept else None)
        out["eval_coverage_ci"] = boot_ratio({k: tuple(v) for k, v in g_cov.items()}, draws, seed)
        out["eval_error_ci"] = boot_ratio({k: tuple(v) for k, v in g_err.items()}, draws, seed)
    return out


def predictions(L, cfg):
    b, b5 = cfg["primary_budget"], min(cfg["budgets"])
    P = {}
    doc, feat = L["rates_eval"]["landed_clean"].get("documentation"), L["rates_eval"]["landed_clean"].get("feature")
    if not doc or not feat or not doc["n"] or not feat["n"]:
        P["1"] = ("not assessable", "no documentation or feature tickets on the evaluation split")
    else:
        diff = doc["rate"] - feat["rate"]
        sep = doc["lo"] is not None and feat["hi"] is not None and doc["lo"] > feat["hi"]
        held = diff >= 0.10 and sep
        P["1"] = ("held" if held else "failed",
                  f"documentation {doc['rate']:.3f} (n={doc['n']}, {doc['lo']:.3f}-{doc['hi']:.3f}) vs feature "
                  f"{feat['rate']:.3f} (n={feat['n']}, {feat['lo']:.3f}-{feat['hi']:.3f}); difference {diff:+.3f}"
                  + ("" if min(doc["n"], feat["n"]) >= cfg["min_cell"] else f"; a cell is under {cfg['min_cell']}"))
    rule = L["trivial_rule"]
    if not rule["handed_over"]:
        P["2"] = ("not assessable", "no documentation tickets on the evaluation split")
    else:
        P["2"] = ("held" if rule["error"] > b else "failed",
                  f"the rule hands over {rule['handed_over']} tickets, {rule['error']:.3f} do not land (budget {b})")
    D = L["lanes"][f"D@{b}"]
    if not D["opened"]:
        P["3"] = ("failed", "no lane opened for type plus checks at the primary budget")
        P["4"] = ("not assessable", "no lane for type plus checks at the primary budget")
    else:
        ok = D["eval_coverage"] >= 0.25 and D["eval_error"] is not None and D["eval_error"] <= b + 0.03
        P["3"] = ("held" if ok else "failed",
                  f"coverage {D['eval_coverage']:.3f}, error {D['eval_error']}, on {D['eval_n']} evaluation tickets")
        B = L["lanes"][f"B@{b}"]
        bcov = B.get("eval_coverage") or 0.0
        if not D["eval_coverage"]:
            P["4"] = ("not assessable", "the full score's lane covers no evaluation ticket")
        else:
            P["4"] = ("held" if bcov >= (2 / 3) * D["eval_coverage"] else "failed",
                      f"task type alone covers {bcov:.3f}, full score {D['eval_coverage']:.3f} "
                      f"(ratio {bcov / D['eval_coverage']:.2f}, needs 0.67)")
    opened5 = [sc for sc in ("B", "C", "D") if L["lanes"][f"{sc}@{b5}"]["opened"]
               and L["lanes"][f"{sc}@{b5}"]["eval_error"] is not None and L["lanes"][f"{sc}@{b5}"]["eval_error"] <= b5]
    if rule["handed_over"] and rule["error"] <= b5:
        opened5.append("A")
    P["5"] = ("held" if not opened5 else "failed",
              "no score keeps a 5% lane on the evaluation split" if not opened5 else
              f"a 5% lane held on the evaluation split for: {', '.join(sorted(opened5))}")
    return P


def lanes(args):
    cfg = load_config(args.config)
    draws, seed = cfg["bootstrap_draws"], cfg["seed"]
    scores = {str(r["issue_id"]): r for r in read_jsonl(args.scores) if "error" not in r}
    held = {}
    if args.held:
        held = {r["pr_id"]: r for r in read_jsonl(args.held)}
    rows, missing_score, held_missing = [], 0, 0
    for r in read_jsonl(args.table):
        s = scores.get(str(r["issue_id"]))
        if s is None:
            missing_score += 1
            continue
        if args.drop_agent and r["agent"] == args.drop_agent:
            continue
        r = dict(r, type_before=s["task_type"], checks=s["checks"], p_land=s["p_land"])
        r["y"] = None if r["landed_clean"] is None else int(r["landed_clean"])
        if args.held and r["split"] == "evaluation" and r["y"] == 1:
            h = held.get(r["pr_id"])
            if h is None or h.get("held") is None:
                r["y"] = None
                held_missing += 1
            else:
                r["y"] = int(bool(h["held"]))
        rows.append(r)
    if args.leakage_only:
        rows = [r for r in rows if r["split"] == "calibration" or r["issue_created_at"] > cfg["leakage_cutoff"]]
    if args.shuffle:
        rng = random.Random(seed + args.shuffle_seed)
        by_repo = collections.defaultdict(list)
        for r in rows:
            if r["y"] is not None:
                by_repo[r["repo"]].append(r)
        for repo in sorted(by_repo):
            ys = [r["y"] for r in by_repo[repo]]
            rng.shuffle(ys)
            for r, y in zip(by_repo[repo], ys):
                r["y"] = y
    build_scores(rows, cfg)
    cal = [r for r in rows if r["split"] == "calibration" and r["y"] is not None]
    ev = [r for r in rows if r["split"] == "evaluation" and r["y"] is not None]
    L = {"options": {"held": bool(args.held), "leakage_only": args.leakage_only, "shuffle": args.shuffle,
                     "drop_agent": args.drop_agent},
         "sample": {"scored_rows": len(rows), "missing_score": missing_score, "held_missing": held_missing,
                    "calibration_with_outcome": len(cal), "evaluation_with_outcome": len(ev)},
         "lanes": {}, "curves": {}}
    for sc in SCORES:
        for b in cfg["budgets"]:
            L["lanes"][f"{sc}@{b}"] = lane_metrics(cal, ev, sc, b, draws, seed)
        L["curves"][sc] = {"calibration": curve([(r["s"][sc], r["y"]) for r in cal]),
                           "evaluation": curve([(r["s"][sc], r["y"]) for r in ev])}
    docs = [r for r in ev if r["type_before"] == "documentation"]
    L["trivial_rule"] = {"handed_over": len(docs), "coverage": len(docs) / len(ev) if ev else None,
                         "error": (sum(1 - r["y"] for r in docs) / len(docs)) if docs else None}
    evrows = [r for r in rows if r["split"] == "evaluation"]
    for r in rows:
        r["landed"] = r["y"]
    L["rates_eval"] = {lvl: {t: rate_cell([r for r in evrows if r["type_before"] == t], key, draws, seed)
                             for t in cfg["task_types"]}
                       for lvl, key in (("merged", "merged"), ("landed_clean", "landed"))}
    L["rates_eval_overall"] = {lvl: rate_cell(evrows, key, draws, seed) for lvl, key in (("merged", "merged"), ("landed_clean", "landed"))}
    L["rates_all"] = {lvl: {t: rate_cell([r for r in rows if r["type_before"] == t], key, draws, seed)
                            for t in cfg["task_types"]}
                      for lvl, key in (("merged", "merged"), ("landed_clean", "landed"))}
    both = [r for r in rows if r.get("type_after")]
    conf = collections.Counter(f"{r['type_before']} -> {r['type_after']}" for r in both)
    L["agreement"] = {"n": len(both), "same": (sum(r["type_before"] == r["type_after"] for r in both) / len(both))
                      if both else None, "pairs": dict(conf.most_common(20))}
    per = collections.defaultdict(list)
    for r in rows:
        per[r["agent"]].append(r)
    L["per_agent"] = {a: {"n": len(v), "merged": rate_cell(v, "merged", draws, seed),
                          "landed_clean": rate_cell(v, "landed", draws, seed)} for a, v in sorted(per.items())}
    L["predictions"] = predictions(L, cfg)
    write_json(args.out, L)
    print_verdict(L)


def print_verdict(L):
    for k in sorted(L["predictions"]):
        v, why = L["predictions"][k]
        print(f"prediction {k}: {v}  ({why})")


# ---------------------------------------------------------------------------- 30-day check

def held_plan(args):
    rows = [r for r in read_jsonl(args.table) if r["split"] == "evaluation" and r["landed_clean"]]
    write_jsonl(args.out, [{"pr_id": r["pr_id"], "repo": r["repo"], "number": r["number"], "merged_at": r["merged_at"],
                            "commit_shas": r["commit_shas"]} for r in rows])
    print(f"{len(rows)} landed clean pull requests on the evaluation split; up to {len(rows) * 12} GitHub calls")


def is_fixing(message, number, cfg):
    first = (message or "").strip()
    return bool(re.search(cfg["fix_message"], first, re.I) or re.search(rf"#{number}\b", message or ""))


def held_one(gh, item, cfg):
    repo, n = item["repo"], item["number"]
    try:
        pr = gh.get(f"/repos/{repo}/pulls/{n}")
        files = gh.get(f"/repos/{repo}/pulls/{n}/files", {"per_page": 100})
    except urllib.error.HTTPError as e:
        return {"pr_id": item["pr_id"], "held": None, "note": f"HTTP {e.code} for the pull request"}
    files = sorted(files, key=lambda f: (-f.get("changes", 0), f["filename"]))[:10]
    since = ts(item["merged_at"])
    until = since + datetime.timedelta(days=cfg["fix_window_days"])
    own = set(item["commit_shas"]) | {pr.get("merge_commit_sha")}
    fixing = set()
    for f in files:
        try:
            commits = gh.get(f"/repos/{repo}/commits", {"path": f["filename"], "since": iso(since),
                                                         "until": iso(until), "per_page": 100})
        except urllib.error.HTTPError:
            continue
        for c in commits:
            if c["sha"] in own:
                continue
            if is_fixing(c["commit"]["message"], n, cfg):
                fixing.add(c["sha"])
    return {"pr_id": item["pr_id"], "held": not fixing, "fixing": sorted(fixing), "files": len(files)}


def held_cmd(args):
    import fetch_repo
    cfg = load_config(args.config)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN first (a read-only token). It is never written to disk.")
    gh = fetch_repo.GitHub(token=token, cache_dir=Path(args.out).parent / "cache")
    done = {r["pr_id"] for r in read_jsonl(args.out)} if Path(args.out).exists() else set()
    plan = [p for p in read_jsonl(args.plan) if p["pr_id"] not in done]
    print(f"{len(done)} done, {len(plan)} to go")
    with open(args.out, "a", encoding="utf-8") as f:
        for i, item in enumerate(plan, 1):
            f.write(json.dumps(held_one(gh, item, cfg), sort_keys=True) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(plan)}  ({gh.calls} calls to GitHub, {gh.cached} from the local copy)", flush=True)
    print(f"done; {gh.calls} calls to GitHub")


# ---------------------------------------------------------------------------- report

def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci(pair):
    return "" if not pair or pair[0] is None else f" ({pct(pair[0])} to {pct(pair[1])})"


def report(args):
    L = json.loads(Path(args.lanes).read_text())
    cfg = load_config(args.config) if args.config else None
    out = ["# Agent lane report", ""]
    s = L["sample"]
    out += [f"Options: {L['options']}.", "",
            f"Scored rows {s['scored_rows']} (missing a score: {s['missing_score']}). With an outcome: calibration "
            f"{s['calibration_with_outcome']}, evaluation {s['evaluation_with_outcome']}"
            + (f"; landed clean evaluation rows without a 30-day answer: {s['held_missing']}" if L["options"]["held"] else "")
            + ".", ""]
    out += ["## Landed rates by task type, evaluation split", "",
            "| Task type | n | Merged | Landed | n (landed) |", "|---|---|---|---|---|"]
    for t, m in L["rates_eval"]["merged"].items():
        c = L["rates_eval"]["landed_clean"][t]
        if m["n"]:
            out.append(f"| {t} | {m['n']} | {pct(m['rate'])}{ci((m['lo'], m['hi']))} | "
                       f"{pct(c['rate'])}{ci((c['lo'], c['hi']))} | {c['n']} |")
    o = L["rates_eval_overall"]
    out += [f"| all | {o['merged']['n']} | {pct(o['merged']['rate'])} | {pct(o['landed_clean']['rate'])} | "
            f"{o['landed_clean']['n']} |", ""]
    out += ["## Lanes", "", "| Score | Budget | Opened | Calibration coverage | Calibration error | "
            "Evaluation coverage | Evaluation error |", "|---|---|---|---|---|---|---|"]
    for k in sorted(L["lanes"]):
        v, (sc, b) = L["lanes"][k], k.split("@")
        if v["opened"]:
            out.append(f"| {SCORE_NAMES[sc]} | {pct(float(b))} | yes | {pct(v['cal_coverage'])} | {pct(v['cal_error'])} | "
                       f"{pct(v['eval_coverage'])}{ci(v.get('eval_coverage_ci'))} | "
                       f"{pct(v['eval_error'])}{ci(v.get('eval_error_ci'))} |")
        else:
            out.append(f"| {SCORE_NAMES[sc]} | {pct(float(b))} | no | | | | |")
    r = L["trivial_rule"]
    out += ["", f"Trivial rule as a fixed rule on the evaluation split: hands over {r['handed_over']} tickets "
            f"({pct(r['coverage'])}), of which {pct(r['error'])} do not land.", ""]
    out += ["## Coverage against error, evaluation split (every 10%)", "",
            "| Coverage | " + " | ".join(SCORE_NAMES[s] for s in SCORES) + " |", "|---" * (len(SCORES) + 1) + "|"]
    for q in range(10, 101, 10):
        out.append(f"| {q}% | " + " | ".join(pct(L["curves"][s]["evaluation"][q - 1][1]) for s in SCORES) + " |")
    out += ["", "## Predictions", ""]
    for k in sorted(L["predictions"]):
        v, why = L["predictions"][k]
        out.append(f"{k}. **{v}**: {why}.")
    a = L["agreement"]
    out += ["", "## Task type read from the issue against AIDev's type read from the pull request title", "",
            f"{a['n']} tickets, same type {pct(a['same'])}. Most common pairs: "
            + "; ".join(f"{k} ({v})" for k, v in a["pairs"].items()) + ".", ""]
    for extra, title in ((args.held_lanes, "With the 30-day check (landed and held) on the evaluation split"),
                         (args.shuffle_lanes, "Shuffle check")):
        if extra:
            E = json.loads(Path(extra).read_text())
            out += [f"## {title}", ""]
            for k in sorted(E["lanes"]):
                v = E["lanes"][k]
                base = L["lanes"][k]
                out.append(f"- {k}: opened {v['opened']}, evaluation coverage {pct(v.get('eval_coverage'))}, error "
                           f"{pct(v.get('eval_error'))} (without: {pct(base.get('eval_coverage'))}, {pct(base.get('eval_error'))})")
            out.append("")
    out += ["## Appendix, per agent (findings note only, not for the post)", "",
            "| Agent | n | Merged | Landed clean |", "|---|---|---|---|"]
    for ag, v in L["per_agent"].items():
        out.append(f"| {ag} | {v['n']} | {pct(v['merged']['rate'])} | {pct(v['landed_clean']['rate'])} "
                   f"(n={v['landed_clean']['n']}) |")
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def verdict(args):
    print_verdict(json.loads(Path(args.lanes).read_text()))


# ---------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("aidev-table", help="build the ticket table from AIDev v3 JSONL extracts")
    p.add_argument("--extracts", required=True); p.add_argument("--config", required=True)
    p.add_argument("--out", required=True); p.add_argument("--sample", type=int, default=0)
    p.set_defaults(fn=aidev_table)
    p = sub.add_parser("own-table", help="build the ticket table from your own repository on GitHub")
    p.add_argument("--repo", required=True); p.add_argument("--config", required=True)
    p.add_argument("--agent-logins", default=""); p.add_argument("--body-marker")
    p.add_argument("--commit-marker"); p.add_argument("--since", default="2026-01-01")
    p.add_argument("--max-pages", type=int, default=30); p.add_argument("--out", required=True)
    p.set_defaults(fn=own_table)
    p = sub.add_parser("score", help="score each ticket with the model (needs AI_GATEWAY_API_KEY)")
    p.add_argument("--table", required=True); p.add_argument("--config", required=True)
    p.add_argument("--out", required=True); p.add_argument("--spend-cap", type=float, default=30.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--retry-errors", action="store_true", help="call again for tickets whose stored answer is an error")
    p.set_defaults(fn=score)
    p = sub.add_parser("rescore", help="re-read stored raw replies with the current parser, no calls")
    p.add_argument("--scores", required=True); p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=rescore)
    p = sub.add_parser("lanes", help="fit on the calibration split, judge lanes on the evaluation split")
    p.add_argument("--table", required=True); p.add_argument("--scores", required=True)
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shuffle", action="store_true"); p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--leakage-only", action="store_true"); p.add_argument("--held")
    p.add_argument("--drop-agent")
    p.set_defaults(fn=lanes)
    p = sub.add_parser("held-plan", help="list landed clean evaluation pull requests for the 30-day check")
    p.add_argument("--table", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=held_plan)
    p = sub.add_parser("held", help="the 30-day check on GitHub (needs GITHUB_TOKEN)")
    p.add_argument("--plan", required=True); p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=held_cmd)
    p = sub.add_parser("report", help="markdown report from lanes output")
    p.add_argument("--lanes", required=True); p.add_argument("--out", required=True)
    p.add_argument("--config"); p.add_argument("--held-lanes"); p.add_argument("--shuffle-lanes")
    p.set_defaults(fn=report)
    p = sub.add_parser("verdict", help="print the five predictions")
    p.add_argument("--lanes", required=True)
    p.set_defaults(fn=verdict)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
