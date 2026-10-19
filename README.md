# ForgeFlow

ForgeFlow turns a public GitHub repository's open issues into reviewed, tested
**draft** pull requests. Give it a repository, and it ranks the open issues,
picks the most promising one, asks a model for a structured diagnosis and a set
of exact edits, then proves those edits work in a Docker sandbox before a human
is asked to approve anything.

It never merges. It never marks a pull request ready for review. Nothing reaches
GitHub until a human types the SHA-256 fingerprint of the patch they approved.

This is a learning project built by two students. Its goal is measured results
on real repositories — including the runs that fail — rather than a polished
demo.

## What it actually does

```
rank issues  →  pick one  →  clone at a pinned commit  →  baseline test run
     ↓
call 1: structured analysis (summary, hypothesis, files, plan, confidence)
     ↓
gate: is it confident, fixable, and does it name real files?
     ↓
call 2: exact search/replace edits  →  verify in Docker  →  on failure, retry
     ↓
human approves a SHA-256 fingerprint  →  rebuild, re-verify the hash, push a
branch to your fork, open a draft PR that discloses AI assistance
```

The model only ever emits edits. It has no shell, no tools, and no ability to
run anything. Issue text and repository contents are treated as untrusted data,
not as instructions.

## Quick start

ForgeFlow needs Python 3.10+, a Docker daemon, and a GitHub token.

```bash
git clone <this repository>
cd forgeflow
pip install -e ".[dev]"

# 1. Check the environment: Docker, tokens, provider keys.
python -m forgeflow.cli doctor

# 2. See how the heuristics rank a repository. No model, no Docker, no cost.
python -m forgeflow.cli issues pallets/click

# 3. Rank and diagnose, without cloning or touching Docker.
GEMINI_API_KEY=... python -m forgeflow.cli analyse pallets/click

# 4. The full pipeline. Stops at "proposed" and prints a fingerprint.
GEMINI_API_KEY=... python -m forgeflow.cli run pallets/click

# 5. Approve that exact patch, then publish it as a draft PR.
python -m forgeflow.cli approve <fingerprint> --by "your name"
python -m forgeflow.cli publish <approval-id> --fork-owner <your-account>
```

## Commands

| Command | What it does | Needs |
| --- | --- | --- |
| `doctor` | Checks Docker, tokens, provider keys, and GitHub access | — |
| `issues OWNER/NAME` | Ranks open issues with the heuristics and explains every score | GitHub token |
| `analyse OWNER/NAME` | Ranks, then makes call 1 and reports the gate decision | + model key |
| `run OWNER/NAME` | The whole pipeline; stops before publishing | + Docker |
| `approve FINGERPRINT` | Records your approval of one verified patch | — |
| `publish APPROVAL_ID` | Rebuilds, re-verifies the hash, pushes, opens a draft PR | + publish token |
| `status [RUN_ID]` | Shows what the SQLite store recorded | — |

Useful flags: `--provider mock` (offline, deterministic), `--issue N` (force a
specific issue), `--max-attempts N`, `--confidence 0.7`, `--dry-run` on
`publish`, `--json` on most commands.

## Configuration

Everything is an environment variable, optionally set in `.env`.

| Variable | Purpose |
| --- | --- |
| `GITHUB_TOKEN` | Read-only access to list issues and detect linked PRs |
| `FORGEFLOW_GITHUB_PUBLISH_TOKEN` | A **separate**, narrowly scoped token used only to push and open PRs |
| `FORGEFLOW_FORK_OWNER` | The account that owns the fork the branch is pushed to |
| `GEMINI_API_KEY` | Google AI Studio key (the default provider, free tier) |
| `ANTHROPIC_API_KEY` | Claude key, if `FORGEFLOW_LLM_PROVIDER=anthropic` |
| `FORGEFLOW_LLM_PROVIDER` | `gemini` (default), `anthropic`, or `mock` |
| `FORGEFLOW_DOCKER` | Docker executable, e.g. `sudo docker` where the daemon needs it |
| `FORGEFLOW_MAX_ATTEMPTS` | Retries after a failed patch (default 3) |
| `FORGEFLOW_CONFIDENCE_THRESHOLD` | Gate threshold, 0–1 (default 0.6) |
| `FORGEFLOW_EXCLUDED_LABELS` | Labels that disqualify an issue |

Run `python -m forgeflow.cli doctor --json` for the full effective picture.

### Tokens

Use **two** tokens, and keep the publish token as narrow as you can:

* **Read token** (`GITHUB_TOKEN`) — public repository read access. This is the
  one the pipeline uses to rank issues and clone.
* **Publish token** (`FORGEFLOW_GITHUB_PUBLISH_TOKEN`) — a fine-grained token
  with `contents: write` and `pull requests: write`, scoped to **your fork
  only**. It is used once, at publish time, and is never written to disk.

ForgeFlow runs the model and the sandbox without either token. The sandbox gets
no credentials at all.

## How patches are kept honest

The interesting part of ForgeFlow is not the model call. It is everything that
happens around it.

**The baseline is captured first.** Before any patch exists, the repository's
own test suite runs and its failures are recorded. A patch is only credited
with *newly passing* tests, and is only blamed for *newly failing* ones. A
repository with pre-existing failures does not poison the verdict.

**The sandbox is hostile by default.** Tests run as a non-root user
(`65534:65534`), with `--network none`, CPU/memory/PID limits, a read-only
source tree, and no credentials in the environment. The model's code runs
there, not on your machine.

**The model cannot express a raw diff.** Call 2 must return search/replace
pairs. Each search string must appear exactly once, and the resulting patch is
produced by `git diff` inside the clone — so it is canonical and applies
cleanly. Paths are checked against a policy that refuses `.git/`,
`.github/workflows/`, `conftest.py`, `setup.py`, `pyproject.toml`, `.pth` files,
symlinks, `..`, and any extension outside a small allowlist.

**Approval is a hash, not a button.** `approve` refuses any fingerprint that
does not belong to an attempt that actually passed verification. `publish` then
throws away the working tree, re-clones the recorded base commit, re-applies
the same edits, and recomputes the hash. If it differs from what you approved,
it aborts **before any network write**. This is what stops "approve A, publish
B".

**Every run is recorded.** Runs, ranked issues with their reasons, analyses,
attempts, verification reports, and approvals all go into SQLite, so a claim
about what happened can be checked afterwards.

See [docs/safety.md](docs/safety.md) for the full threat model, including what
this design does *not* protect against.

## Status and limitations

Version 1 is deliberately narrow.

* **Python repositories only**, and **pytest only**. Other languages and test
  runners are refused rather than half-supported.
* **No automatic retry after publishing.** A failed PR open leaves the branch
  pushed and reports it; nothing is retried silently.
* **Ranking is heuristic, and it sometimes picks badly.** In our evaluation it
  ranked a "🚀 Roadmap" issue first on `tiangolo/typer`. That is a real result,
  reported as one — see [docs/evaluation.md](docs/evaluation.md).
* **Test pass/fail is the only correctness signal.** A patch that makes tests
  pass while being wrong can still be approved. The draft PR and the fingerprint
  exist precisely because a human must look.
* **The model can be fooled by issue text.** Prompt injection is mitigated by
  fencing untrusted input and by never giving the model tools, but a
  sufficiently clever issue can still steer the diagnosis.

## Development

```bash
pip install -e ".[dev]"

pytest -m "not integration"    # fast unit tests, no Docker needed
pytest                         # everything, including the Docker sandbox
```

The test suite includes a real end-to-end run against a fixture repository with
a genuine bug: the pipeline clones it, captures a baseline, applies model edits,
runs the suite in Docker, compares results, computes a fingerprint, and proves
nothing was published. See [docs/architecture.md](docs/architecture.md).

## Evaluation

```bash
python -m eval.run rank --repos eval/repos.txt      # heuristics, no model cost
python -m eval.run analyse --limit 5                # adds call 1
```

Results append to `eval/results.jsonl` as one JSON line per run, so repeated
runs build a dataset. Measured results, including failures, are written up in
[docs/evaluation.md](docs/evaluation.md).

## License

MIT
