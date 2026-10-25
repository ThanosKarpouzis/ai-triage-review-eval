# ai-triage-review-eval

How much of issue triage and code review can you hand to AI? This kit measures it on your own repository's history, with the decisions your maintainers and reviewers already made as the ground truth.

It asks each model the same two questions:

- **Issue triage.** Is this issue a bug, a feature request or a question? Judged against the labels your maintainers applied.
- **Pull requests.** Will a reviewer comment on or request changes to this pull request before it is merged? Judged against the reviews each merged pull request actually received.

The result is not only accuracy. For each model it reports how many decisions could go into three lanes:

- **No human**: the model decides alone, at no more than 5% error.
- **Review later**: the model decides and a person checks afterwards, at no more than 15% error (exploratory).
- **Human first**: everything else.

A model earns a lane only if its confidence separates the calls it gets right from the ones it gets wrong. The confidence threshold is chosen on one part of your data (the calibration split) and scored on another it has never seen (the evaluation split).

The write-up behind this kit, with results on hashicorp/terraform and two public benchmarks, is in two posts: [How much of triage and code review can you hand to AI?](https://thanoskarpouzis.com/writing/2026/delegating-triage-and-review-to-ai/) and [Can a cheap decision model take triage off your team's plate?](https://thanoskarpouzis.com/writing/2026/decision-model-triage-and-review/).

A third study, the [taxonomy test](#test-your-own-labels), asks a different question: does giving a model your team's own labels help, and which triage decisions (type, owning team, priority) can a model take on at all? It is written up in [Is it the model or your labels?](https://thanoskarpouzis.com/writing/2026/model-or-labels/).

A fourth tool, [`error_budget.py`](#check-whether-a-lane-can-be-trusted), asks how many labelled issues you need before a lane can be trusted, and what to aim for. It is written up in [How many tickets do you have to label before you can trust an AI lane?](https://thanoskarpouzis.com/writing/2026/how-many-labels/).

A fifth tool, [`review_bot.py`](#is-your-review-bot-earning-its-place), asks whether your AI reviewer's comments are acted on, and whether the flags nobody acted on turned out to be right, with your human reviewers as the yardstick.

A sixth tool, [`agent_lane.py`](#which-tickets-can-you-hand-to-an-agent), asks whether you can tell from the ticket alone which tickets an agent will land, within an error budget. It is written up in [Which tickets can you hand to an agent?](https://thanoskarpouzis.com/writing/2026/agent-lane/).

## What you need

- Python 3. The scripts use the standard library only, so there is nothing to install. The one exception is `prep_aidev.py`, which converts a public dataset once and needs `pyarrow`.
- A GitHub token, strongly advised. Without one GitHub allows 60 requests an hour, far too few for pull requests. A fine-grained personal access token with read-only access to public repositories is enough; for a private repository it needs read access to its issues, pull requests and contents. Keep it in an environment variable, never in a file.
- An API key for [Vercel AI Gateway](https://vercel.com/ai-gateway), which serves the language models used here. The free tier is heavily rate-limited, so a small paid credit is needed.
- Optionally, Docker, for the local classifier. It is the option to use when code must not leave your machine.

## Run it on your repository

Everything runs from this folder.

**1. Fetch.** First see which labels the repository uses:

```sh
export GITHUB_TOKEN=paste-a-read-only-token
python3 fetch_repo.py OWNER/REPO --list-labels
```

Then pick which labels mean bug, feature and question. Names are case-insensitive, and several can be given per group:

```sh
python3 fetch_repo.py OWNER/REPO --bug "bug" --feature "enhancement,feature request" --question "question"
```

On a very busy repository GitHub refuses to page through the issue list past about 10,000 items, pull requests included, and the fetch stops with HTTP 422. Add `--via-search` and issues are listed through GitHub's search in date ranges instead; which issues are kept, and why, stays the same.

This keeps up to 600 issues and 400 merged pull requests created since 1 January 2026 (change with `--since`), so the models you test are unlikely to have seen them in training. It writes them to `kit/OWNER__REPO/data`, with a `manifest.json` recording what was kept and why the rest was skipped. Every answer from GitHub is kept in `kit/OWNER__REPO/cache` as it arrives, so if the fetch stops, running the same command again carries on, and re-deriving the data later makes no API calls at all. Add `--refresh` to fetch fresh data on purpose; the manifest records when the answers used were fetched. Check the manifest before trusting any number: if your team rarely labels issues or rarely leaves review comments, there is not much ground truth to measure against.

**2. Run the models.**

```sh
export AI_GATEWAY_API_KEY=paste-your-key-here
python3 run.py --model haiku --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --smoke
python3 run.py --model haiku --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results
```

The smoke test sends three items per task, to check access. The full run is safe to stop and restart; it resumes where it stopped. On hashicorp/terraform (603 items), Claude Haiku 4.5 took about 20 minutes and cost $1.30. Each results folder has a spend cap, $4.50 by default; raise it with `--spend-cap`. The cap counts what is already in the folder, and each running process counts only its own new spending, so allow for that when running several models at once.

Other models: `--model opus` runs Claude Opus with the same prompt (set `OPUS_MODEL`, for example `anthropic/claude-opus-4.7`, to the version your account can use). `--model jev` runs TypeSafe's Jev decision model; add `--patient` to keep waiting when it is at capacity.

Narrow checks, for code review only: `--model jev_checks` asks Jev six narrow yes/no checks about each change in one request (the wording is in `run.py`, `CHECKS`). Then `python3 combine_checks.py --results-dir kit/OWNER__REPO/results` combines them with a small logistic regression fitted on the calibration split, cross-validated so the threshold is not chosen on the same items, and writes `jev-decomposed.jsonl`, which `analyze.py` scores like any other model. It also reports a "safe to skip review" lane: the largest share of changes, starting from the least likely to draw a comment, that stays within 5% on the calibration split, and how that holds on new data. With few commented pull requests to learn from (fewer than about 50), expect the combined model to say "no comment" almost every time.

**3. Optionally, the local classifier.** An open zero-shot classifier, [DeBERTa-v3-large zeroshot v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) by Moritz Laurer (MIT licence), runs in Docker on your CPU:

```sh
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f local-nli/compose.yaml build
docker compose -f local-nli/compose.yaml run --rm nli --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results
```

The build is the only step that downloads anything: the Python base image, PyTorch's CPU-only build, the pinned libraries in `local-nli/requirements.lock`, and the model at one fixed revision (about 870 MB). The run itself has no network access, sees this folder read-only, and can write only to `results` and `kit`. It is capped at 4 CPU cores and 4 GB, because a run on all cores once restarted a fanless laptop; on a machine with more headroom, raise `cpus` and `NLI_THREADS` in `local-nli/compose.yaml` together. Expect about an hour for 600 items. Remove the image afterwards with `docker compose -f local-nli/compose.yaml down --rmi all`.

**4. Analyse.**

```sh
python3 analyze.py --results-dir kit/OWNER__REPO/results
```

This prints a table per decision and writes `summary.md` and `summary.json` into the results folder: accuracy, macro F1, the three lanes, the actual error in the no-human lane, calibration error, cost per 1,000 decisions and latency. For pull requests it also shows how often reviewers actually commented, and so how well a rule that always answers "no comment" would do. Beating that rule is the first bar.

## Test your own labels

The taxonomy test asks the same models two ways about the same issues: the generic question above (bug, feature or question), and one question per label group in your repository (for example type, team and priority), whose options are your own labels, each described by its GitHub label description as written. The gap shows what your labels add, and the per-group results show which triage decisions a model can take on.

**1. Write a config.** Name each label group by its prefix, say which type labels mean bug, feature and question, and list any label that marks an issue as not yet triaged. [`examples/metabase-taxonomy-config.json`](examples/metabase-taxonomy-config.json) is the one used in the write-up; copy it and adapt it.

**2. Check who sets your labels.** Many busy projects now let a bot label new issues, so a label may be another model's output rather than a person's judgement. The fetch reads each issue's label history, about one API call per issue, and records only whether the reporter, a bot or another person applied each label, never a username. If your bot marks what it touched and people remove the mark when they review (as Metabase's ".Auto triaged" label works), set `"gold": "confirmed"` and `"review_marker"` in the config: only labels a person set, or left in place when they removed the marker, then count as ground truth.

**3. Fetch, run, analyse.**

```sh
python3 fetch_taxonomy.py OWNER/REPO --config my-taxonomy-config.json --max-issues 3000
python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --smoke --spend-cap 8
python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --model haiku --spend-cap 8
python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --model jev --patient --spend-cap 8
python3 analyze_taxonomy.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data
```

`run.py` adds the own-label questions automatically when the data folder has a `taxonomy.json`. The analysis compares the two ways of asking on the same issues, with a paired bootstrap interval, splits the results by how each correct label got there, and, where a bot's labels were reviewed, shows how often the reviewer kept them. To score against every label as it stands as well, derive a second data folder from the local copy at no API cost with `--gold all --out data-all`, run the models on it (answers already given are reused) and analyse it with `--data-dir kit/OWNER__REPO/data-all`. On Metabase, about 1,500 issues and four questions each cost about $5 with Claude Haiku 4.5 and a few cents with Jev.

**4. Optional follow-ups.** Two ways to bring in what a label list does not say, both defined in the config:

- **Area, then team.** Add an `"area"` block naming your top-level area label groups. `--model jev_area` asks Jev only which product area an issue concerns, and `combine_taxonomy.py` learns from the calibration split which areas each team owns.
- **A checklist for priority.** Add `"checks"` with yes/no questions taken from your priority definitions. `--model jev_checks` asks them, and `combine_taxonomy.py` combines them with a small logistic regression fitted on the calibration split.

```sh
python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --model jev_area --task own_team --patient
python3 run.py --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --model jev_checks --task own_priority --patient
python3 combine_taxonomy.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data
python3 explore_taxonomy.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data --cutoff 2026-05-01
```

`analyze_taxonomy.py` then scores the combined results like any other model. `explore_taxonomy.py` adds two checks that matter before you rely on a learned mapping: simple baselines (always the most common label, and the same mapping built from area labels already on your issues) and a time split, fitted on earlier issues and tested on later ones. In the write-up, the area-then-team result held on later issues but shrank, because teams were reorganised during the year: relearn the mapping when your teams change.

Rules for the taxonomy test, fixed before results: label descriptions are used exactly as written, and a label without one is described by its name; every label in a group is offered, deprecated ones included; an issue enters a group only if it carries exactly one label of that group; splits are per issue, about 30% calibration, stratified by type, with a fixed seed. The time split and the baselines in `explore_taxonomy.py` were added after the write-up's results and are exploratory.

## Check whether a lane can be trusted

A lane's threshold is chosen on a small labelled sample, so it can promise more than it delivers. `error_budget.py` measures how often, on your own results and without any new model calls. It merges each model's calibration and evaluation items into one pool, then 2,000 times per sample size (25, 50, 100, 200 and 400 labels) draws a sample, picks the threshold the way `analyze.py` does, and checks it on the rest. A sample size is used only if at least 300 items remain to test on. It also tries two fixes, aiming lower (60% of the budget, so 3% to get 5%) and a safety margin (accept a threshold only when a one-sided 90% exact binomial bound on its error is within budget), and, where items carry dates, sets the threshold on the issues just before a cut-off and tests it on everything after.

```sh
python3 error_budget.py --results-dir kit/OWNER__REPO/results --data-dir kit/OWNER__REPO/data --out kit/OWNER__REPO/error-budget
python3 error_budget.py --report kit/OWNER__REPO/error-budget
```

The report puts each decision and budget in one of three groups, from the lane the plain rule finds on the whole pool:

- **Room**: the lane keeps 99% or more of issues, because the model is rarely wrong. In the write-up the promise held in every draw from 25 labels up. Label 50, check, switch it on.
- **Edge**: the lane keeps 40% to 99%, and its error sits right at the budget. The plain rule then broke the promise about half the time, by a point or two, however many labels were used. More labels do not fix it; aiming lower or the safety margin does, at a cost in coverage that the report shows.
- **Sliver**: the lane keeps 10% to 40%. Small samples invent lanes that are not there: with 50 labels, a lane that appeared broke its promise almost every time.

Below 10% there is no lane to speak of. The checks C1 to C3 at the end of the report test those three statements on your data. Two things to watch: a model that reports a handful of round confidence values gives a lane that is all or nothing, so neither fix works well; and random draws are a best case, so re-check after a reorganisation or a change to your labels, not only on a calendar.

## Is your review bot earning its place

`review_bot.py` asks what happens to the lines a review comment points at, for your AI reviewer and for your human reviewers side by side. For every inline review comment on a merged pull request it records three things: whether the flagged lines changed before merge; whether they were changed again within 90 days of merge by a commit that fixed a bug; and the same two for a *placebo*, a stretch of lines the same pull request added, of the same length and on the same commit, that nobody commented on. Code keeps moving, so some flagged lines change by coincidence. Only the difference from the placebo counts.

It needs a read-only GitHub token (review threads come from GitHub's GraphQL interface, which does not answer without one), `git`, and a clone of your default branch. No model calls.

**1. Write a config.** Copy [`examples/sentry-review-config.json`](examples/sentry-review-config.json) or [`examples/home-assistant-review-config.json`](examples/home-assistant-review-config.json). List your AI reviewer accounts in `ai_reviewers` (as GitHub shows them, for example `cursor[bot]` or `Copilot`) and every other bot in `excluded_bots`; your bug labels on pull requests and issues; your capability labels; and, if you want a front-end and back-end split, your area labels or path rules. Several AI reviewers are pooled as one.

**2. Fetch, clone, map, follow, report.**

```sh
export GITHUB_TOKEN=paste-a-read-only-token
python3 -m unittest tests/test_review_bot.py
python3 review_bot.py fetch --config my-review-config.json --smoke     # five pull requests, to check the token and the bot names
python3 review_bot.py fetch --config my-review-config.json
git clone --single-branch --shallow-since=2025-12-01 https://github.com/OWNER/REPO my-clone
python3 review_bot.py map --config my-review-config.json --clone my-clone
python3 review_bot.py fixes --config my-review-config.json --clone my-clone
python3 review_bot.py report --config my-review-config.json --null     # must pass before you read anything else
python3 review_bot.py report --config my-review-config.json
python3 review_bot.py sample --config my-review-config.json --n 30     # comments to read by hand, and three pull requests to count
```

The fetch keeps the newest 1,500 merged pull requests in the config's window that carry a qualifying comment. Every GitHub answer is cached, so any step can be stopped and re-run. The tool waits out GitHub's hourly limits on its own; on a busy repository the map step can take a few hours.

**3. Read the report.** The first table gives, per reviewer and per split, the acted-on rate, the resolved rate, and for ignored flags (lines nobody changed before merge) how often they were fixed later against their placebo: the *excess*. The second gives the acted-on rate corrected for churn the same way, and from it the number of comments a person reads for each change a comment causes, which is the figure to set against what the bot costs you. It also shows how many resolved threads were resolved by a bot. The third table is the sample-size rule: from your own placebo fix rate, how many ignored flags each reviewer kind needs before a one, two, three or five point excess could show (95% confidence, 80% power), against how many you have, and the smallest excess your current history could have shown. Collect until the difference you care about is in reach.

**Your fix signal decides what the tool can see.** The tool picks it with one rule: your bug labels if at least 10% of the later pull requests that touched flagged code carry one, otherwise titles starting with "fix" if at least 10% follow that convention, otherwise none, and the report says so. With no usable fix signal, the post-merge question cannot be asked of your history; the acted-on figures still can.

Rules fixed before results in the write-up: inline comments only, first in their thread, on the new side of the diff, not by the pull request's author; a placebo at least five lines from any comment, same file first, fixed seed; an ignored flag is compared with a placebo also untouched before merge; 95% intervals by bootstrap over whole pull requests, 2,000 draws; cells with fewer than 100 ignored AI flags are marked too thin. The null check swaps each comment for a second, independent placebo; every interval must then contain zero. In the write-up it caught two biases before any result was read.

Limits: a line change is not agreement, and a later fix on flagged lines is not proof the flag was about that defect. Reading 30 fixed comments by hand per repository, the later fix answered the comment in about one case in eight. Humans choose which lines to comment on and bots comment widely, so this compares reviewers as your team uses them. Comments on commits that were force-pushed away cannot be followed and are counted and left out.

It is written up in [Do your review bot's warnings turn into bugs?](https://thanoskarpouzis.com/writing/2026/review-bot-warnings/).

## Which tickets can you hand to an agent

`agent_lane.py` takes the pull requests your coding agent opened for an issue and asks whether the issue alone, read before any work started, says which ones would land. It is a fourth lane next to the three above, the *agent lane*: the largest share of tickets, from the top of a ranking down, whose failure rate stays within a budget (15% by default, 5% reported too) under the safety-margin rule from `error_budget.py`, chosen on your earlier tickets and judged on your later ones.

Each attempt gets three outcomes: *merged*; *landed clean*, merged with no review requesting changes and no commit by a person other than the agent (merges of the main branch and other bot accounts do not count); and *landed and held*, landed clean with no fixing commit on the same files within 30 days. A model reads only the issue, the repository's language and a size bucket, and returns a task type, six yes or no checks and its own probability that an agent would land the ticket. Four rankings are compared: documentation yes and everything else no, task type alone, the model's probability, and task type plus checks combined by a small logistic regression.

It needs a read-only GitHub token and the AI gateway key above. Scoring costs about $1.50 per thousand tickets with Claude Haiku 4.5.

```sh
export GITHUB_TOKEN=paste-a-read-only-token
export AI_GATEWAY_API_KEY=paste-your-key
python3 -m unittest tests/test_agent_lane.py
python3 agent_lane.py own-table --repo OWNER/REPO --agent-logins "Copilot,devin-ai-integration[bot]" --since 2025-01-01 \
    --config examples/aidev-agent-lane-config.json --out kit/OWNER__REPO/agent/table.jsonl
python3 agent_lane.py score --table kit/OWNER__REPO/agent/table.jsonl --config examples/aidev-agent-lane-config.json \
    --out kit/OWNER__REPO/agent/scores.jsonl --smoke
python3 agent_lane.py score --table kit/OWNER__REPO/agent/table.jsonl --config examples/aidev-agent-lane-config.json \
    --out kit/OWNER__REPO/agent/scores.jsonl
python3 agent_lane.py lanes --table kit/OWNER__REPO/agent/table.jsonl --scores kit/OWNER__REPO/agent/scores.jsonl \
    --config examples/aidev-agent-lane-config.json --out kit/OWNER__REPO/agent/lanes.json --shuffle   # must open no lane
python3 agent_lane.py lanes --table kit/OWNER__REPO/agent/table.jsonl --scores kit/OWNER__REPO/agent/scores.jsonl \
    --config examples/aidev-agent-lane-config.json --out kit/OWNER__REPO/agent/lanes.json
python3 agent_lane.py held-plan --table kit/OWNER__REPO/agent/table.jsonl --out kit/OWNER__REPO/agent/held-plan.jsonl
python3 agent_lane.py held --plan kit/OWNER__REPO/agent/held-plan.jsonl --config examples/aidev-agent-lane-config.json \
    --out kit/OWNER__REPO/agent/held.jsonl
python3 agent_lane.py report --lanes kit/OWNER__REPO/agent/lanes.json --out kit/OWNER__REPO/agent/report.md
```

If your agent commits under a developer's account, pass `--commit-marker` with a pattern its commit messages always carry (for example a co-author trailer); without one, only merged can be judged. `own-table` finds the issue from a closing reference in the pull request's description ("fixes #123"), so attempts without one are not counted; the manifest says how many. Every GitHub answer and every model reply is kept, so any step can be re-run without new calls (`rescore` re-reads stored replies; `--retry-errors` asks again only for answers that failed).

Expect a lane to be possible only once you have a few hundred attempts with an outcome. In the write-up, on 1,194 public tickets that agents attempted, documentation tickets landed clean 57% of the time against 17% for features, but no ranking opened a lane at either budget: the best slice still failed about a third of the time. To reproduce it, [`prep_aidev.py`](prep_aidev.py) converts the AIDev tables (CC BY 4.0, Li, Zhang and Hassan) to the JSONL `agent_lane.py aidev-table` reads; it is the one script here that needs a library, `pyarrow`, and its header has the download commands.

It is written up in [Which tickets can you hand to an agent?](https://thanoskarpouzis.com/writing/2026/agent-lane/).

## Reproduce the public benchmarks

The same runner and analysis also work on two public datasets, which are not included here.

- **Issues**: the [NLBSE 2024 issue report classification](https://nlbse2024.github.io/tools/) data. Put `issues_train.csv` and `issues_test.csv` in `data/raw/` and run `python3 prep_nlbse.py`.
- **Code review**: `Diff_Quality_Estimation.zip` from the [CodeReviewer record on Zenodo](https://zenodo.org/records/6900648) (2.8 GB, CC BY 4.0). Run `python3 prep_codereviewer.py path/to/Diff_Quality_Estimation.zip`.

Then `python3 run.py --model haiku` and `python3 analyze.py`. For a cheaper run on a stratified 1,000-item sample, `python3 make_sample.py` writes `data-sample/`, and `run.py` and `analyze.py` take `--data-dir data-sample --results-dir results-sample`. Both preparation scripts use a fixed seed, so the samples match the write-up.

## Rules and limits

Everything below was fixed before any results were seen.

- The question wording and criteria are in `run.py` and are identical for every model. Changing them makes results incomparable with earlier runs.
- A failed call or an unreadable answer counts as wrong, with confidence 0.
- Issues count only if their labels match exactly one of the three groups. Text is the title and body, HTML comments removed, capped at 2,500 characters.
- Pull requests: merged only; drafts, bot authors, and reviews by the author or by bots are excluded. "Yes" if any other reviewer left a comment review or requested changes; "no" if at least one other reviewer approved and none commented. Pull requests with no review from anyone else are skipped. Text is the title, the first 600 characters of the description and the diff. Diffs over 20,000 characters are excluded rather than cut (`--max-diff-chars`).
- About 30% of each label goes to the calibration split, with a fixed seed.

The ground truth is only as good as your team's habits. An approved pull request may still have drawn a comment somewhere, a change request may concern something outside the diff, and leaving out very large diffs tilts the sample towards smaller changes. The local classifier reads at most 512 tokens, so on pull requests it sees only the start of most diffs. A small calibration split makes the chosen threshold optimistic: in the write-up, every no-human lane set for 5% error landed at 6 to 8% on new data, so leave a margin. `error_budget.py` shows how much.

## Data and privacy

Nothing fetched or generated is committed: `.gitignore` keeps `data`, `results`, `kit` (including the local copy of GitHub's answers) and the samples on your machine. Label histories record who applied a label only as the reporter, a bot or another person. If you publish results from someone else's repository, report aggregate numbers only and do not name issue authors or reviewers; GitHub's [Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies) allow research use of public data when the resulting publication is open access. Pull-request diffs and issue text are sent to whichever model provider you run; use the local classifier if that is not acceptable for your code.

## Contributing

This is a reference kit that goes with a write-up, not a maintained project, so issues and pull requests are switched off. Fork it freely and adapt it to your own team; the licence allows that without asking.

## Credits

Written by Thanos Karpouzis with help from Claude (Anthropic), which drafted much of the code and documentation.

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`. This is personal research and is not affiliated with any employer or model provider.
