# AGENTS.md

Repository-specific context for agents working on ForgeFlow.

## What this is

A Python CLI that turns a public GitHub repository's open issues into reviewed,
tested **draft** pull requests. Read `README.md` first, then
`docs/architecture.md`, `docs/safety.md`, and `docs/evaluation.md`.

The project is a learning exercise for two students. The stated goal is
*measured results on real repositories, including failures*. When you add a
feature, prefer recording an honest measurement over adding a demo.

## Non-negotiable invariants

Do not weaken these without an explicit request. Tests exist for each.

1. **Nothing reaches GitHub without a matching fingerprint.** `publish`
   re-derives the patch from the recorded base commit and compares hashes
   *before* any network call. `tests/test_publish_flow.py` asserts that a
   mismatch produces zero GitHub calls.
2. **The model emits edits, never diffs or commands.** `EditPlan` is
   `extra="forbid"` and contains only `(path, search, replace)`.
3. **The model has no tools.** It is called over HTTP with text in and text out.
   Do not add function calling.
4. **The sandbox has no network during tests** and runs as uid 65534.
5. **Tests are compared against a baseline captured first.** Pre-existing
   failures are never blamed on a patch.
6. **Never auto-merge, never mark a PR ready for review.** There is no code path
   for either, and a test asserts no such call is made.
7. **Untrusted input is fenced** in prompts with `<<<BEGIN_UNTRUSTED_*>>>`
   markers and labelled as data.

## Commands

```bash
python3 -m pytest tests/ -q -m "not integration"   # fast, no Docker, ~1s
python3 -m pytest tests/ -q                        # full, needs Docker, ~140s

FORGEFLOW_DOCKER="sudo docker" python3 -m forgeflow.cli doctor
python3 -m forgeflow.cli issues pallets/click
python3 -m eval.run rank --repos eval/repos.txt
```

## Environment notes

* **Docker needs `sudo` here.** The daemon is not reachable as the default user,
  so use `FORGEFLOW_DOCKER="sudo docker"`. The integration tests hardcode
  `DOCKER = "sudo docker"` in the test modules.
* **`GITHUB_TOKEN` is set** in this environment (read-only; it belongs to
  `Laqtumx`). `FORGEFLOW_GITHUB_PUBLISH_TOKEN` is not set, so publishing cannot
  run live. Never push to a real remote from a test.
* **Long runs must go to the background.** The foreground command timeout is
  ~1080s but the full suite takes ~140s, so it fits; if it ever does not, use
  `nohup ... > /tmp/out.log 2>&1 &` and poll the file.
* Lint is `ruff`, configured in `pyproject.toml`. Run it before committing:
  `sudo docker run --rm -v "$PWD:/w" python:3.11-slim sh -c "pip install -q ruff && cd /w && ruff check forgeflow tests eval"`

## Testing conventions

* Integration tests are marked `integration` and live in `test_integration.py`
  and `test_publish_flow.py`. They use real git and real Docker against a
  fixture repository built in `conftest.py`.
* **The fixture bug is deliberately partial**: `return abs(a) + abs(b)` is
  correct for non-negative inputs and wrong for negatives. Keep it that way.
  If the bug affected every input, a wrong fix would be undetectable and the
  baseline-vs-patched comparison could not tell a fix from a different bug.
* The fixture includes a `test_known_broken` that always fails, so the
  pre-existing-failure logic runs on every end-to-end test.
* `test_publish_flow.py` has an autouse `no_real_push` fixture that replaces
  `push_branch`. Do not remove it.
* Do not use mocks for the pipeline. Use the `mock` LLM provider
  (`--provider mock`) and the real sandbox instead.

## Layout

```
forgeflow/          the package; see the module table in docs/architecture.md
tests/              unit + integration
eval/run.py         the evaluation harness (rank / analyse)
eval/repos.txt      the repository list
eval/results.jsonl  appended JSON lines, gitignored
docs/               architecture, safety, evaluation
```

## Known weaknesses (do not "fix" silently)

These are documented in `docs/evaluation.md` and are honest results, not bugs to
paper over:

* The heuristics cannot tell a detailed bug report from a long non-bug. They
  ranked a "🚀 Roadmap" issue first on `tiangolo/typer`.
* There is no minimum-quality floor: a repository with one open issue always
  produces a candidate regardless of quality.
* Ranking produces many ties at 7.25, which cannot be ordered meaningfully.
* Evaluation 2 (the analysis gate) has only been run against the mock provider.
  The real gate pass rate on Gemini is still unmeasured — this is the biggest
  missing number in the project.

If you improve one of these, update `docs/evaluation.md` with the new
measurement rather than deleting the old finding.
