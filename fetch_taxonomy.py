#!/usr/bin/env python3
"""Build the taxonomy test's data from one GitHub repository. Standard library only.

The question: does giving a model your team's own labels change how much issue triage it can take on?
Two arms on the same issues, pre-registered on 3 October 2026 (see README.md):
  arm A  issue_type       post one's generic question (bug / feature / question), unchanged
  arm B  own_<family>     one question per label family (for example type, team, priority), whose options
                          are the repository's own labels, each described by its GitHub label description

  export GITHUB_TOKEN=...   # strongly advised; a fine-grained, read-only token for public repositories is enough
  python3 fetch_taxonomy.py OWNER/REPO --config kit/OWNER__REPO/taxonomy-config.json

The config names the families by label prefix, the generic mapping for the type family, and labels that
mark an issue as not yet triaged. An ordinal family (priority) is also scored within one level, in label-name order. Example (metabase/metabase):
  {"families": {"type": {"prefix": "Type:"}, "team": {"prefix": ".Team/"}, "priority": {"prefix": "Priority:", "ordinal": true}},
   "type_family": "type", "routing_family": "team", "gold": "confirmed", "review_marker": ".Auto triaged",
   "generic_map": {"Type:Bug": "bug", "Type:New Feature": "feature", "Type:Question": "question"},
   "exclude_labels": [".Needs Triage"]}

Writes to kit/OWNER__REPO/data:
  issues.jsonl            arm A items (issues whose type label maps to bug, feature or question)
  taxonomy_<family>.jsonl arm B items (issues carrying exactly one label of that family)
  taxonomy.json           the families: every label offered, with its description as written
  label_history.jsonl     per issue: final labels, label events (who applied them, as a role, never a username) and
                          each final label's provenance
  manifest.json           counts of everything kept and excluded

Rules, fixed before fetching:
- Issues created from --since to --until inclusive; pull requests and bot-authored issues excluded; issues carrying any
  exclude label excluded (their labels are not final); issues with no label from any family excluded.
- If more than --max-issues remain, a fixed-seed random sample of that size is kept.
- Text is the title and body, cleaned exactly as in fetch_repo.py; labels are never part of the text.
- Splits are per issue, so an issue sits in the same split in every task: about 30% calibration, stratified by type label.
- Label options for a family are all of the repository's labels with that prefix at fetch time, deprecated ones included.
  A label with no description is described by its own name.
- Provenance of each final label, from its label history (added 3 October 2026, before any model ran):
    set by a person       last applied by a person, other than the reporter's own labels at creation (templates)
    confirmed by review   applied by a bot or at creation, and afterwards a person removed the config's review marker
                          (a bot's "auto triaged" label) without the label being changed after that review
    unconfirmed           anything else: a bot's or a template's label that no person reviewed
  With "gold": "confirmed" in the config, only the first two count as ground truth; other labels of that family
  are dropped from the data and counted. The bot's own label for each family is kept as a reference answer.
"""
import argparse, json, os, random, re, time, urllib.error
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from fetch_repo import GitHub, clean_issue, HERE, SEED

FAMILY_INSTRUCTIONS = "Which of this repository's {family} labels would its maintainers apply to this GitHub issue?"
AT_CREATION_SECONDS = 120  # a label applied this soon after the issue was opened counts as applied at creation
CONFIRMED = ("set by a person", "confirmed by review")


def provenance(events, label, review_marker):
    """How a final label got there, from the issue's label events (see the rules above)."""
    if events is None:
        return "unknown"
    events = sorted(events, key=lambda e: e["seconds_after_creation"])
    added = [e for e in events if e["event"] == "labeled" and e["label"] == label]
    if not added:
        return "unknown"
    last = added[-1]
    by_reporter_at_creation = last["actor"] == "author" and last["seconds_after_creation"] <= AT_CREATION_SECONDS
    if last["actor"] != "bot" and not by_reporter_at_creation:
        return "set by a person"
    if review_marker and any(e["event"] == "unlabeled" and e["label"] == review_marker and e["actor"] != "bot"
                             and e["seconds_after_creation"] >= last["seconds_after_creation"] for e in events):
        return "confirmed by review"
    return "unconfirmed"


def bot_answer(events, prefix, review_marker):
    """The family labels a bot applied before the first person's review (or ever, if never reviewed)."""
    if events is None:
        return []
    events = sorted(events, key=lambda e: e["seconds_after_creation"])
    review = next((e["seconds_after_creation"] for e in events if e["event"] == "unlabeled" and e["label"] == review_marker
                   and e["actor"] != "bot"), None) if review_marker else None
    out = []
    for e in events:
        if review is not None and e["seconds_after_creation"] > review:
            break
        if e["actor"] == "bot" and e["label"] and e["label"].startswith(prefix):
            if e["event"] == "labeled" and e["label"] not in out:
                out.append(e["label"])
            elif e["event"] == "unlabeled" and e["label"] in out:
                out.remove(e["label"])
    return out


def all_labels(gh, repo):
    return [{"name": l["name"], "description": (l.get("description") or "").strip()}
            for l in gh.pages(f"/repos/{repo}/labels", {}, 20)]


def search_range(gh, repo, start, end, out):
    """Collect issues created in [start, end] through the search API, splitting the range if it exceeds 1,000 results."""
    q = f"repo:{repo} is:issue created:{start.isoformat()}..{end.isoformat()}"
    first = gh.get("/search/issues", {"q": q, "per_page": 100, "page": 1, "sort": "created", "order": "asc"})
    total = first.get("total_count", 0)
    if total > 1000 and start < end:
        mid = start + (end - start) // 2
        search_range(gh, repo, start, mid, out)
        search_range(gh, repo, mid + timedelta(days=1), end, out)
        return
    out += first.get("items", [])
    for page in range(2, min(10, (total + 99) // 100) + 1):
        if not gh.last_from_disk:
            time.sleep(2.1)  # the search API allows 30 requests a minute with a token; no wait when reading the local copy
        out += gh.get("/search/issues", {"q": q, "per_page": 100, "page": page, "sort": "created", "order": "asc"}).get("items", [])
    if not gh.last_from_disk:
        time.sleep(2.1)


def label_events(gh, repo, issue):
    author = (issue.get("user") or {}).get("login")
    created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
    events = []
    for e in gh.pages(f"/repos/{repo}/issues/{issue['number']}/events", {}, 5, cache=True):
        if e.get("event") not in ("labeled", "unlabeled"):
            continue
        actor = e.get("actor") or {}
        role = "bot" if actor.get("type") == "Bot" or (actor.get("login") or "").endswith("[bot]") else (
            "author" if actor.get("login") == author else "other")
        at = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        events.append({"event": e["event"], "label": (e.get("label") or {}).get("name"), "actor": role,
                       "seconds_after_creation": round((at - created).total_seconds())})
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="OWNER/REPO")
    ap.add_argument("--config", required=True, help="taxonomy config JSON (see above)")
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--until", default="2026-09-30")
    ap.add_argument("--max-issues", type=int, default=1000)
    ap.add_argument("--calib-share", type=float, default=0.3)
    ap.add_argument("--refresh", action="store_true", help="ask GitHub again instead of reusing the local copy of earlier answers")
    ap.add_argument("--gold", choices=["confirmed", "all"], help="override the config's gold rule, e.g. all labels as a secondary analysis")
    ap.add_argument("--out", default="data", help="folder name under kit/OWNER__REPO for the derived data (default: data)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    families, type_fam = cfg["families"], cfg["type_family"]
    gmap, exclude = cfg.get("generic_map", {}), {x.lower() for x in cfg.get("exclude_labels", [])}
    if a.gold:
        cfg = dict(cfg, gold=a.gold)
    marker, confirmed_only = cfg.get("review_marker"), cfg.get("gold", "all") == "confirmed"
    slug = a.repo.replace("/", "__")
    gh = GitHub(os.environ.get("GITHUB_TOKEN"), HERE / "kit" / slug / "cache", refresh=a.refresh)
    if not gh.token:
        print("No GITHUB_TOKEN set: GitHub allows only 60 requests an hour without one, far too few for label history.")
    out_dir = HERE / "kit" / slug / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = all_labels(gh, a.repo)
    tax = {"repo": a.repo, "instructions": FAMILY_INSTRUCTIONS, "type_family": type_fam,
           "routing_family": cfg.get("routing_family"), "generic_map": gmap, "families": {}}
    for fam, spec in families.items():
        opts = [l for l in labels if l["name"].startswith(spec["prefix"])]
        if len(opts) < 2:
            raise SystemExit(f"family {fam!r}: fewer than two labels start with {spec['prefix']!r}")
        tax["families"][fam] = {"prefix": spec["prefix"], "ordinal": bool(spec.get("ordinal")), "options": opts}
    # Follow-up tests (pre-registered 4 October 2026): narrow yes/no checks per family, and an area question
    # whose options are top-level area families, each described by its root label or by its sub-area names.
    tax["checks"] = cfg.get("checks", {})
    if cfg.get("area"):
        ar, opts = cfg["area"], []
        for name in ar["families"]:
            subs = [l for l in labels if l["name"].startswith(name + "/")]
            if not subs:
                raise SystemExit(f"area {name!r}: no label starts with {name + '/'!r}")
            root = next((l["description"] for l in subs if l["name"] == name + "/" and l["description"]), "")
            parts = [l["name"][len(name) + 1:] for l in subs if l["name"] != name + "/"]
            opts.append({"name": name, "description": root or ("Includes: " + ", ".join(parts[:15]) if parts else name)})
        tax["area"] = {"for_family": ar["for_family"], "instructions": ar["instructions"], "options": opts}
    print("Families: " + "; ".join(f"{f} {len(v['options'])} labels, {sum(bool(o['description']) for o in v['options'])} described"
                                 for f, v in tax["families"].items()), flush=True)

    print(f"Searching issues created {a.since} to {a.until} ...", flush=True)
    found = []
    search_range(gh, a.repo, date.fromisoformat(a.since), date.fromisoformat(a.until), found)
    seen, skipped, kept = set(), Counter(), []
    for it in found:
        if it["number"] in seen:
            continue
        seen.add(it["number"])
        names = [l["name"] for l in it.get("labels", [])]
        if "pull_request" in it:
            skipped["pull request"] += 1
        elif (it.get("user") or {}).get("type") == "Bot":
            skipped["bot author"] += 1
        elif any(n.lower() in exclude for n in names):
            skipped["not yet triaged (exclude label)"] += 1
        elif not any(n.startswith(f["prefix"]) for f in tax["families"].values() for n in names):
            skipped["no label from any family"] += 1
        else:
            kept.append(it)
    found_n = len(seen)
    if len(kept) > a.max_issues:
        skipped[f"not sampled (over {a.max_issues})"] = len(kept) - a.max_issues
        kept = random.Random(SEED).sample(sorted(kept, key=lambda i: i["number"]), a.max_issues)
    print(f"  {found_n} issues found, {len(kept)} kept, skipped: {dict(skipped)}", flush=True)

    def fam_labels(it, fam):
        return [l["name"] for l in it.get("labels", []) if l["name"].startswith(tax["families"][fam]["prefix"])]

    # Splits per issue, stratified by type label.
    rng, by = random.Random(SEED + 2), defaultdict(list)
    for it in sorted(kept, key=lambda i: i["number"]):
        t = fam_labels(it, type_fam)
        by[t[0] if len(t) == 1 else "(none or several)"].append(it)
    split_of = {}
    for key in sorted(by):
        group = by[key][:]
        rng.shuffle(group)
        n_cal = round(len(group) * a.calib_share)
        for i, it in enumerate(group):
            split_of[it["number"]] = "calib" if i < n_cal else "eval"

    print(f"Fetching label history for {len(kept)} issues (about one API call each, kept locally) ...", flush=True)
    arm_a, arm_b, history, prov_counts = [], defaultdict(list), [], defaultdict(Counter)
    for n, it in enumerate(sorted(kept, key=lambda i: i["number"]), 1):
        num, sp = it["number"], split_of[it["number"]]
        text = clean_issue(it["title"], it.get("body"))
        try:
            ev = label_events(gh, a.repo, it)
        except urllib.error.HTTPError as e:
            ev = None
            skipped[f"label history unavailable (HTTP {e.code})"] += 1
        prov = {f: (provenance(ev, fam_labels(it, f)[0], marker) if len(fam_labels(it, f)) == 1 else None) for f in tax["families"]}
        bots = {f: bot_answer(ev, tax["families"][f]["prefix"], marker) for f in tax["families"]}
        history.append({"number": num, "created_at": it["created_at"], "split": sp, "state": it.get("state"),
                        "state_reason": it.get("state_reason"),
                        "labels": {f: fam_labels(it, f) for f in tax["families"]}, "provenance": prov,
                        "bot_labels": bots, "events": ev})
        for f, pv in prov.items():
            if pv:
                prov_counts[f][pv] += 1
        t = fam_labels(it, type_fam)
        if len(t) == 1 and t[0] in gmap and (not confirmed_only or prov[type_fam] in CONFIRMED):
            arm_a.append({"id": f"issue-{num}", "task": "issue_type", "split": sp, "label": gmap[t[0]],
                          "text": text, "created_at": it["created_at"], "provenance": prov[type_fam]})
        for fam in tax["families"]:
            f = fam_labels(it, fam)
            if len(f) == 1 and (not confirmed_only or prov[fam] in CONFIRMED):
                arm_b[fam].append({"id": f"issue-{num}-{fam}", "task": f"own_{fam}", "split": sp, "label": f[0],
                                   "text": text, "created_at": it["created_at"], "issue": num,
                                   "provenance": prov[fam], "bot_labels": bots[fam]})
        if n % 100 == 0:
            print(f"  {n}/{len(kept)} ({gh.calls} API calls, {gh.cached} reused from disk)", flush=True)

    def write(name, rows):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write("issues.jsonl", arm_a)
    for fam, rows in arm_b.items():
        write(f"taxonomy_{fam}.jsonl", rows)
    write("label_history.jsonl", history)
    (out_dir / "taxonomy.json").write_text(json.dumps(tax, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"repo": a.repo, "since": a.since, "until": a.until, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": cfg, "issues_found": found_n, "issues_kept": len(kept), "skipped": dict(skipped),
                "calib_share": a.calib_share, "caps": {"issue_chars": 2500},
                "arm_a": {"items": len(arm_a), "labels": dict(Counter(r["label"] for r in arm_a)),
                          "splits": dict(Counter(r["split"] for r in arm_a))},
                "arm_b": {f: {"items": len(r), "labels": dict(Counter(x["label"] for x in r)),
                              "splits": dict(Counter(x["split"] for x in r))} for f, r in arm_b.items()},
                "gold": "confirmed only" if confirmed_only else "all labels", "review_marker": marker,
                "provenance": {f: dict(c) for f, c in prov_counts.items()},
                "api_calls": gh.calls, "answers_reused_from_disk": gh.cached, "snapshot": gh.snapshot()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nArm A: {len(arm_a)} issues. Arm B: " + ", ".join(f"{f} {len(r)}" for f, r in arm_b.items()))
    print(f"Wrote {out_dir.relative_to(HERE)} ({gh.calls} GitHub API calls). Next:\n"
          f"  python3 run.py --data-dir kit/{slug}/{a.out} --results-dir kit/{slug}/results --smoke")


if __name__ == "__main__":
    main()
