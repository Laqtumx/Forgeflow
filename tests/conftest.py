"""Shared fixtures.

The tests are split by what they need:

* ``unit`` (default) — pure logic: ranking, path policy, prompts, parsing,
  store, sandbox output parsing. No Docker, no network.
* ``integration`` — Docker and/or a real git repository. Marked, so the fast
  suite can run without them.

The fake GitHub client is a real object implementing the same method surface as
``GitHubClient``, so the pipeline exercises its actual code path rather than a
patched-out one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeflow.models import Issue


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs docker or network")


# --- issue helpers --------------------------------------------------------


def make_issue(number: int = 1, **overrides) -> Issue:
    defaults = {
        "number": number,
        "title": "Crash when calling add() with negative numbers",
        "body": (
            "Steps to reproduce:\n"
            "1. call add(-1, -2)\n\n"
            "Traceback (most recent call last):\n"
            '  File "calc.py", line 2, in add\n'
            "TypeError: unsupported operand\n"
        ),
        "html_url": f"https://github.com/example/repo/issues/{number}",
        "labels": (),
        "assignees": (),
        "comments": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "author": "reporter",
    }
    defaults.update(overrides)
    return Issue(**defaults)


# --- git fixture repository ----------------------------------------------


# The bug is partial on purpose: `add` is correct for non-negative inputs and
# wrong only for negatives. That shape matters, because it means a *wrong* fix
# can be detected — it will break a test that passed before the patch. If the
# baseline were broken for every input, any wrong operator would produce the
# same failure set and pass/fail comparison could not tell them apart.
BROKEN_MODULE = """\
def add(a, b):
    return abs(a) + abs(b)
"""

FIXED_MODULE = """\
def add(a, b):
    return a + b
"""

# A plausible-looking fix that is wrong: it negates the sum, so it breaks
# test_add_positive and test_add_mixed_signs, which passed on the baseline.
WRONG_MODULE = """\
def add(a, b):
    return -a - b
"""

TEST_MODULE = """\
from calc import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-1, -2) == -3


def test_add_zero():
    assert add(0, 0) == 0


def test_add_mixed_signs():
    assert add(5, -3) == 2
"""

PREEXISTING_FAILURE = """\
def test_known_broken():
    assert False, "this test already failed before any patch"
"""


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(cwd),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.com",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.com",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr}")
    return completed.stdout.strip()


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A tiny Python repository with a genuine bug and a real test suite.

    ``calc.add`` subtracts instead of adding, so three tests fail on the base
    commit. ``test_known_broken`` fails on purpose, to exercise the
    pre-existing-failure comparison.
    """
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    (repo / "calc.py").write_text(BROKEN_MODULE)
    (repo / "test_calc.py").write_text(TEST_MODULE)
    (repo / "test_known_broken.py").write_text(PREEXISTING_FAILURE)
    (repo / "README.md").write_text("# fixture\n\nA tiny repo used by the tests.\n")
    (repo / "requirements.txt").write_text("")

    _git(repo, "init", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial broken state")
    return repo


@pytest.fixture
def fixture_head(fixture_repo: Path) -> str:
    return _git(fixture_repo, "rev-parse", "HEAD")


def write_calc(repo: Path, content: str) -> None:
    """Replace the fixture module and push it into the sandbox.

    The pipeline never writes to a checkout by hand — it applies model edits —
    so tests that need a specific post-patch state use this helper instead.
    """
    (repo / "calc.py").write_text(content)


@pytest.fixture
def fixture_stats() -> dict[str, int]:
    """Expected outcome counts for the fixture repository.

    Baseline: three tests fail (the negative and mixed-sign cases, plus the
    deliberate pre-existing failure); positive and zero pass.
    """
    return {
        "total": 5,
        "baseline_failed": 3,
        "baseline_passed": 2,
        "newly_passing": 2,
        "preexisting": 1,
    }


# --- fake GitHub ----------------------------------------------------------


class FakeGitHub:
    """A stand-in for :class:`forgeflow.github_client.GitHubClient`.

    It implements the read methods the pipeline uses and records every write,
    so tests can assert that nothing was published.
    """

    def __init__(
        self,
        issues: list[Issue] | None = None,
        *,
        repo_meta: dict | None = None,
        linked: dict[int, list[dict]] | None = None,
        default_branch: str = "main",
    ) -> None:
        self.issues = issues or []
        self.repo_meta = repo_meta or {
            "full_name": "example/repo",
            "language": "Python",
            "archived": False,
            "size": 100,
            "default_branch": default_branch,
        }
        self.linked = linked or {}
        self.default_branch = default_branch
        self.calls: list[tuple[str, object]] = []
        self.pulls_opened: list[dict] = []
        self.forks_created: list[str] = []

    # reads
    def get_repo(self, repo):
        self.calls.append(("get_repo", repo.slug))
        return self.repo_meta

    def list_open_issues(self, repo, limit: int = 200):
        self.calls.append(("list_open_issues", limit))
        return list(self.issues)

    def linked_pull_requests(self, repo, issue_number: int):
        self.calls.append(("linked_pull_requests", issue_number))
        return list(self.linked.get(issue_number, []))

    def get_default_branch(self, repo):
        self.calls.append(("get_default_branch", repo.slug))
        return self.default_branch

    def current_user(self, token=None):
        return {"login": "tester"}

    def rate_limit(self):
        return {"resources": {"core": {"remaining": 4999, "limit": 5000}}}

    # writes
    def ensure_fork(self, repo, fork_owner: str):
        self.calls.append(("ensure_fork", fork_owner))
        self.forks_created.append(fork_owner)
        return {"full_name": f"{fork_owner}/{repo.name}"}

    def wait_for_fork_ready(self, repo, fork_owner: str, **kwargs):
        return {"size": 100, "pushed_at": "2026-01-01T00:00:00Z"}

    def open_draft_pull_request(self, repo, *, title, body, head, base):
        self.calls.append(
            ("open_draft_pull_request", {"title": title, "head": head, "base": base})
        )
        self.pulls_opened.append(
            {"title": title, "body": body, "head": head, "base": base}
        )
        return {
            "number": 42,
            "html_url": "https://github.com/example/repo/pull/42",
        }

    def post_issue_comment(self, repo, issue_number: int, body: str):
        self.calls.append(("post_issue_comment", issue_number))
        return {"id": 1}

    def delete_branch(self, fork_owner, repo_name, branch):
        self.calls.append(("delete_branch", branch))

    def request(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("FakeGitHub.request should not be called directly")


@pytest.fixture
def fake_github() -> FakeGitHub:
    return FakeGitHub(
        issues=[
            make_issue(1, title="Crash when calling add() with negative numbers"),
            make_issue(
                2,
                title="Please add support for matrices",
                body="It would be nice to support matrices someday.",
                labels=("enhancement",),
            ),
            make_issue(
                3,
                title="Assigned already",
                assignees=("someone",),
            ),
            make_issue(
                4,
                title="Duplicate report",
                labels=("duplicate",),
            ),
        ],
        linked={3: [{"number": 99, "state": "open"}]},
    )
