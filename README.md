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

## What you need

- Python 3. The scripts use the standard library only, so there is nothing to install.
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

This keeps up to 600 issues and 400 merged pull requests created since 1 January 2026 (change with `--since`), so the models you test are unlikely to have seen them in training. It writes them to `kit/OWNER__REPO/data`, with a `manifest.json` recording what was kept and why the rest was skipped. Reviews and diffs are cached in `kit/OWNER__REPO/cache` as they arrive, so if the fetch stops, running the same command again carries on. Check the manifest before trusting any number: if your team rarely labels issues or rarely leaves review comments, there is not much ground truth to measure against.

**2. Run the models.**

```sh
export AI_GATEWAY_API_KEY=paste-your-key-here
python3 run.py --model haiku --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results --smoke
python3 run.py --model haiku --data-dir kit/OWNER__REPO/data --results-dir kit/OWNER__REPO/results
```

The smoke test sends three items per task, to check access. The full run is safe to stop and restart; it resumes where it stopped. On hashicorp/terraform (603 items), Claude Haiku 4.5 took about 20 minutes and cost $1.30. Each results folder has a $4.50 spend cap, set in `run.py`.

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

The ground truth is only as good as your team's habits. An approved pull request may still have drawn a comment somewhere, a change request may concern something outside the diff, and leaving out very large diffs tilts the sample towards smaller changes. The local classifier reads at most 512 tokens, so on pull requests it sees only the start of most diffs. A small calibration split makes the chosen threshold optimistic: in the write-up, every no-human lane set for 5% error landed at 6 to 8% on new data, so leave a margin.

## Data and privacy

Nothing fetched or generated is committed: `.gitignore` keeps `data`, `results`, `kit` and the samples on your machine. If you publish results from someone else's repository, report aggregate numbers only and do not name issue authors or reviewers; GitHub's [Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies) allow research use of public data when the resulting publication is open access. Pull-request diffs and issue text are sent to whichever model provider you run; use the local classifier if that is not acceptable for your code.

## Contributing

This is a reference kit that goes with a write-up, not a maintained project, so issues and pull requests are switched off. Fork it freely and adapt it to your own team; the licence allows that without asking.

## Credits

Written by Thanos Karpouzis with help from Claude (Anthropic), which drafted much of the code and documentation.

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`. This is personal research and is not affiliated with any employer or model provider.
