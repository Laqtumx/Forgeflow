# Evaluation

This is the part of the project we care most about. A tool like this is easy to
demo on a repository you prepared for it and hard to make useful on repositories
nobody prepared. So the evaluation reports what actually happened, including the
cases where ForgeFlow picked badly.

Everything below is reproducible. Raw records are appended to
`eval/results.jsonl`, one JSON line per run.

## How to run it

```bash
# Heuristics only: no model calls, no Docker, no cost.
python -m eval.run rank --repos eval/repos.txt

# Adds call 1 (the analysis) for the best candidate in each repository.
python -m eval.run analyse --repos eval/repos.txt --limit 5

# Reproduce the sandbox verification on one repository, offline.
python -m forgeflow.cli run <owner>/<name> --provider mock
```

## Evaluation 1: ranking, on ten real repositories

Run on 2026-10-17, against the GitHub REST API, with an authenticated token.
Ten repositories, 73.5 seconds total, zero failures to read.

| Repository | Open issues | Ranked | Best candidate | Score |
| --- | ---: | ---: | --- | ---: |
| pallets/click | 76 | 25 | #1269 Chained command with argument and option results in "No such command" | 7.25 |
| psf/requests | 100 | 25 | #4965 Accessing response.content twice removes forgets read error | 7.25 |
| encode/httpx | 64 | 25 | #2983 "Server disconnected" on HTTP/2 connection pool | 7.25 |
| pypa/pip | 100 | 25 | #9034 concurrent pip-install fails to build wheel due to race | 7.25 |
| python-poetry/poetry | 100 | 25 | #2703 Poetry cannot add ssh urls from Azure Devops | 7.25 |
| pytest-dev/pytest | 100 | 25 | #2844 Wrong session scoped fixtures parametrization | 7.25 |
| psf/black | 100 | 25 | #1540 Black shouldn't merge multi-line string | 7.25 |
| tox-dev/tox | 1 | 1 | #4169 TOML command references drop ignore/invert exit-code prefixes | 6.25 |
| pallets/flask | 1 | 1 | #6146 Add Cloudflare to Flask Hosting Platforms docs? | 3.50 |
| tiangolo/typer | 1 | 1 | #678 🚀 Roadmap | 3.25 |

Aggregate:

* **10/10** repositories produced a candidate. No repository was rejected
  outright for being archived, oversized, or non-Python.
* Best-candidate scores: **median 7.25**, min 3.25, max 7.25.
* **4** issues were dropped by the filter, all for carrying an excluded label
  (`duplicate`, `wontfix`, `question`, and similar). No issue was dropped for
  having a linked pull request on the repositories sampled.
* The top score of 7.25 is the ceiling for an unassigned issue with no linked
  PR, a reproduction, and a traceback: `2.0 + 1.5 + 1.5 + 1.0 = 6.0`, plus
  detailed body (+0.75) and fix intent (+0.5). Seven of ten repositories hit
  exactly 7.25, which is expected: the heuristics are coarse, and on a large
  repository many issues look equally promising on these signals alone. A
  ranking that produces ties cannot order them meaningfully.

### What went wrong

Two results are bad, and they are worth more than the eight good ones.

**`tiangolo/typer` → #678 "🚀 Roadmap" (score 3.25).** A roadmap issue is not a
bug and cannot be fixed by a patch. It scored 3.25 because it is unassigned
(+2.0) with no linked PR (+1.5), and the vague-report penalty (−1.0) was
outweighed by a length bonus (+0.75) for its 1825-character body. Length is a
weak proxy for quality, and a long roadmap document looks like a detailed bug
report to a character count.

This is the clearest failure mode: **the heuristics cannot tell "detailed bug
report" from "long non-bug".** A cheap fix would be to penalise issues whose
title matches roadmap/meta vocabulary, but the honest conclusion is that
heuristics alone are not enough to pick work, which is exactly why call 1 and
its confidence gate exist downstream.

**`pallets/flask` → #6146 "Add Cloudflare to Flask Hosting Platforms docs?"
(score 3.50).** A documentation request, not a code defect. It scored 3.50 from
the two base signals. A pytest run cannot verify a docs change, so the sandbox
would find nothing to prove — the run would end as `verification-failed` or, if
the model returned no edits, not-actionable.

Both failures share a cause: the candidate pool was tiny. Flask had **1** open
issue and typer had **1**. With a single option there is nothing to rank, and
the tool selects it regardless of quality. The filter has no minimum-quality
threshold — only a relative ordering. A cheap improvement would be to refuse to
run when the best score is below some floor.

### What worked

On the seven repositories with 25 ranked candidates, the top pick was a
plausible bug report every time: chained-command parsing, a `requests` read
error, an HTTP/2 pool disconnect, a pip build race, a poetry URL bug, a pytest
fixture bug, a black formatting bug. All are the kind of issue where a small,
testable code change is the expected fix — which is precisely what version 1
can attempt.

The explanation trail did its job. Every score printed the reasons that produced
it, so the typer and flask failures were visible immediately rather than buried.
That is the main argument for making reasons a required part of the ranking
model rather than an optional extra.

## Evaluation 2: the analysis gate

Run against `pallets/click` and `encode/httpx` with the deterministic mock
provider, to exercise the full path without spending quota.

| Repository | Best candidate | Confidence | Gate |
| --- | --- | ---: | --- |
| pallets/click | #1269 | 0.75 | pass |
| encode/httpx | #2983 | 0.75 | pass |

**This measures the plumbing, not the model.** The mock provider returns a
fixed confidence of 0.75, so a 2/2 pass rate here says only that the gate, the
prompt construction, and the store all work end to end. It says nothing about
whether Gemini or Claude would clear the gate on these issues. Running
evaluation 2 against a real provider is the next step, and the numbers are not
in yet.

Being clear about this distinction matters. A 100% gate pass rate from a mock is
not evidence of anything except that the code path runs.

## Evaluation 3: sandbox verification

Measured by the integration test suite against the fixture repository, which has
a genuine partial bug and a deliberately pre-existing failure.

| Quantity | Result |
| --- | --- |
| Baseline: 5 tests collected | 2 passed, 3 failed |
| After the correct patch | 4 passed, 1 failed |
| Newly failing | **0** (the pre-existing failure is not blamed) |
| Newly passing | **2** |
| Lint regressions in changed files | 0 |
| Fingerprint reproducible across rebuilds | yes |
| Wrong patch (`return -a - b`) | correctly rejected, 2 newly-failing tests |
| Attempts before success, retry test | 2 of 3 |
| Push attempts during a failed run | **0** |

The retry test is the interesting one: attempt 1 returns a wrong fix, the
verifier rejects it and reports which previously passing tests broke, and
attempt 2 receives that error in its prompt and returns the correct fix. The
second patch contains none of the first patch's changes.

## What we would measure next

1. **Run evaluation 2 against Gemini** on all ten repositories, and report the
   real gate pass rate, the real confidence distribution, and the actual token
   cost. This is the number the project is missing.
2. **Run the full pipeline on a real repository**, end to end, and report
   whether the produced draft PR would plausibly be useful. Success here means
   "a maintainer would consider it", not "the tests went green".
3. **Add a minimum-score floor** to the filter and re-run evaluation 1 to see
   whether it removes the typer and flask failures without discarding good
   candidates.
4. **Try a cheap classifier on the issue title** (roadmap / docs / question) as
   a pre-filter, and measure how much it changes the ranking.
5. **Count how often the model names files that do not exist**, which the gate
   currently rejects silently. That number would say something concrete about
   how well the context-selection step is working.
