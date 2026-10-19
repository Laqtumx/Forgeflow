# Architecture

ForgeFlow is a pipeline, not a service. One run moves through a fixed sequence
of stages, each of which either produces a recorded artifact or stops the run.

```
                    ┌─────────────────────────────────────────┐
   GitHub REST ────▶│ github_client   list issues, linked PRs │
                    └────────────────┬────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │ ranking   filter, score, explain        │
                    └────────────────┬────────────────────────┘
                                     ▼   one candidate, or stop
                    ┌─────────────────────────────────────────┐
   shallow clone ──▶│ gitops    clone at a pinned base commit │
                    └────────────────┬────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────────┐
   untrusted repo ─▶│ context   pick the files to show the    │
                    │           model (small, deliberate)     │
                    └────────────────┬────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │ llm/contract   call 1: analysis         │
                    └────────────────┬────────────────────────┘
                                     ▼   gate: confident? fixable?
                    ┌─────────────────────────────────────────┐
                    │ llm/contract   call 2: search/replace   │
                    └────────────────┬────────────────────────┘
                                     ▼
   Docker sandbox ─▶│ verify   apply → test → compare → lint  │
                    └────────────────┬────────────────────────┘
                          pass ◀─────┴─────▶ fail → feed error back, retry
                            │
                            ▼
                    ┌─────────────────────────────────────────┐
   SQLite ─────────▶│ store    every artifact, with reasons   │
                    └────────────────┬────────────────────────┘
                                     ▼
                          STOP: prints a SHA-256 fingerprint

   ── human types the fingerprint ──────────────────────────────
                            │
                            ▼
                    ┌─────────────────────────────────────────┐
                    │ publish  rebuild → re-verify hash →     │
                    │          push to fork → draft PR        │
                    └─────────────────────────────────────────┘
```

The vertical break is the point: the pipeline cannot publish, and publishing
cannot be reached without a human-supplied hash.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Environment → `Config`. Refuses to start without what it needs. |
| `defaults.py` | Every tunable default in one place, so they can be reasoned about together. |
| `models.py` | Pydantic models. Closed schemas: unknown fields are rejected. |
| `errors.py` | One exception hierarchy, so the CLI has a single failure path. |
| `github_client.py` | The REST API. Read methods and write methods are separated by token. |
| `ranking.py` | Heuristic scoring. Every score carries the reasons that produced it. |
| `context.py` | Chooses which repository files to show the model, and how much of them. |
| `llm/contract.py` | The two-call contract: prompts, JSON extraction, schema enforcement, the gate. |
| `llm/providers.py` | Gemini, Anthropic, and a deterministic mock. Plain HTTP, no SDK. |
| `edits.py` | Path policy and search/replace application. Refuses anything suspicious. |
| `gitops.py` | Clone, diff, rebuild, commit, push. Scrubbed environment, per-command credentials. |
| `sandbox.py` | The Docker sandbox: volumes, install, test, lint, output parsing. |
| `verify.py` | Baseline capture and patch verification. The comparison lives here. |
| `store.py` | SQLite. Runs, issues, analyses, attempts, verifications, approvals. |
| `pipeline.py` | Orchestration. Owns the stage order and the retry loop. |
| `publish.py` | Approval gating, hash re-verification, branch push, draft PR. |
| `cli.py` | Argument parsing and presentation. No logic. |
| `eval/run.py` | Measured results on real repositories. |

## Design decisions worth explaining

### The baseline runs before the model is asked anything

Capturing the baseline first is what makes the verdict meaningful. ForgeFlow
records the set of failing test node IDs on the untouched checkout, then
compares:

* **newly failing** — failed after the patch, passed before. These blame the patch.
* **newly passing** — passed before, failed after... inverted: failed before,
  passed after. These are the patch's credit.

A repository where `test_known_broken` already fails will still show
`test_known_broken` failing afterwards, and the patch is not blamed for it. The
error text fed back to the model also labels pre-existing failures explicitly,
so a retry does not chase something it did not cause.

### Two model calls, with a gate between them

Call 1 is cheap and structural: what is the bug, which files matter, how
confident is this, is it even fixable. Call 2 is expensive and specific: the
actual edits.

Putting the gate between them means a hopeless issue costs one call instead of
three. The gate rejects low confidence, `fixable: false`, `needs_more_info`,
and — importantly — an analysis whose named files do not exist in the
repository. A model that hallucinates `src/ghost.py` is caught before call 2.

### Search/replace instead of diffs

Asking a model for a unified diff produces something that almost applies.
ForgeFlow asks for `(path, search, replace)` triples, requires each `search` to
occur exactly once in the file, and then generates the patch with `git diff`
inside the clone. The result is a canonical patch that applies cleanly, and the
fingerprint is stable across rebuilds because the diff is produced by git both
times.

This also makes the edits reviewable. "Replace this exact string with that
exact string" is something a human can check in seconds; a 400-line diff is not.

### Application is atomic

`apply_edits` resolves and validates every path in the plan, then reads every
target file, then applies all replacements in memory, then writes. If any step
fails, no file is written. A plan that is half-applied would produce a patch
that does not correspond to anything the model proposed, and the fingerprint
would be meaningless.

### The sandbox is a different machine, treated as hostile

The model's code runs in a container with `--network none` during tests, as
uid 65534, with CPU/memory/PID caps and a read-only source tree. The repository
is copied in through a volume, not mounted from the host working tree, so the
sandbox cannot write back into the clone the fingerprint is computed from.

Dependency installation is a separate, earlier step that *is* allowed network
access, because installing from PyPI requires it. Tests are not.

### One run, one store, one story

Everything lands in SQLite as it happens: the ranked candidates and their
reasons, the analysis, each attempt's edits and patch, each verification report,
each approval. `forgeflow status <run_id>` reads it back. When a run produces a
surprising result, the record shows exactly which stage produced it.

## The fixture repository

The integration tests do not mock the pipeline. They build a small git
repository with a real bug and run the real thing against it:

```python
def add(a, b):
    return abs(a) + abs(b)      # correct for positives, wrong for negatives
```

That shape is deliberate. Because the bug is *partial*, a wrong fix is
detectable: `return -a - b` breaks `test_add_positive`, which passed on the
baseline, so the comparison flags it as newly failing. If the baseline had been
broken for every input, any wrong operator would produce the same failure set
and pass/fail comparison could not tell a fix from a different bug.

The suite also includes a `test_known_broken` that always fails, so the
pre-existing-failure logic is exercised on every end-to-end run.

## Testing strategy

| Layer | What it proves | Needs |
| --- | --- | --- |
| `test_ranking.py` | Filtering, scoring, and that every score is explained | — |
| `test_edits.py` | Path policy refuses escapes, symlinks, and dangerous files; application is atomic | — |
| `test_llm_contract.py` | JSON extraction, schema enforcement, prompt fencing, the gate | — |
| `test_sandbox_parsing.py` | JUnit/text parsing and the baseline comparison | — |
| `test_store_and_publish.py` | Schema, and that approval refuses unverified patches | — |
| `test_integration.py` | Real git, real Docker, real end-to-end pipeline and retry loop | Docker |
| `test_publish_flow.py` | Hash re-verification aborts before any network write | Docker |

The publish tests install an autouse fixture that replaces `push_branch`, so no
test can ever push to a real remote by accident.
