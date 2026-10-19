"""The publish path: approval gating, hash re-verification, and no premature writes.

These tests exercise the real ``publish`` function against a local git
repository, with only the clone URL redirected. The GitHub client is a fake, so
every network-facing call is recorded and can be asserted absent.

The central safety property under test: publishing re-derives the patch from
the recorded base commit and compares the hash *before* it pushes anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forgeflow.config import Config
from forgeflow.errors import PatchVerificationError, PublishError
from forgeflow.gitops import clone, diff
from forgeflow.models import Analysis, Edit, EditPlan, VerificationReport
from forgeflow.publish import approve, publish, publish_pending
from forgeflow.store import Store
from tests.conftest import FakeGitHub

pytestmark = pytest.mark.integration

DOCKER = "sudo docker"


def git(cwd: Path, *args: str) -> str:
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
            "GIT_AUTHOR_NAME": "T",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def real_fingerprint(fixture_repo: Path, base: str, tmp_path: Path) -> str:
    """Compute the true fingerprint of the fix, as publish will re-derive it."""
    from forgeflow.edits import apply_edits, fingerprint_patch
    from forgeflow.models import RepoRef

    checkout = tmp_path / "reference"
    clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
    git(checkout, "checkout", "--detach", base)
    apply_edits(
        checkout,
        EditPlan(
            edits=[
                Edit(
                    path="calc.py",
                    search="return abs(a) + abs(b)",
                    replace="return a + b",
                )
            ]
        ),
    )
    return fingerprint_patch(diff(checkout))


@pytest.fixture(autouse=True)
def no_real_push(monkeypatch):
    """Safety net: a test must never push to a real remote.

    Tests that care about the push either record the call or replace this
    fixture's stub themselves.
    """
    calls: list[dict] = []

    def stub(checkout, repo_name, fork_owner, branch, token, **kwargs):
        calls.append({"branch": branch, "fork": fork_owner, "token": token})

    monkeypatch.setattr("forgeflow.gitops.push_branch", stub)
    return calls


@pytest.fixture
def local_clone(monkeypatch, fixture_repo: Path):
    """Redirect every gitops.clone call at the fixture repository."""
    from forgeflow import gitops

    original = gitops.clone

    def patched(repo, destination, *, depth=50, branch=None, token=None, timeout=600, url=None):
        return original(
            repo,
            destination,
            depth=depth,
            branch=branch,
            token=token,
            timeout=timeout,
            url=str(fixture_repo),
        )

    monkeypatch.setattr(gitops, "clone", patched)
    return patched


def seed_approved(tmp_path: Path, fixture_repo: Path, *, fingerprint: str, edits=None):
    """Build a store with one verified attempt and an approval for it."""
    base = git(fixture_repo, "rev-parse", "HEAD")
    config = Config.load(
        env={},
        db_path=tmp_path / "publish.sqlite3",
        work_dir=tmp_path / "publish-work",
        fork_owner="tester",
        github_publish_token="fake-publish-token",
        github_read_token="fake-read-token",
    )
    store = Store(config.db_path)
    run_id = store.start_run(
        "example/repo", "mock", "mock-deterministic", 1, default_branch="main"
    )
    store.finish_run(run_id, "proposed", base)
    store.record_issue_titles(run_id, {1: "add() is wrong for negatives"})
    store.record_analysis(
        run_id,
        1,
        Analysis(
            summary="abs() loses the sign.",
            root_cause_hypothesis="The implementation ignores sign.",
            files_to_inspect=["calc.py"],
            fix_plan=["return a + b"],
            confidence=0.9,
            fixable=True,
        ),
    )
    plan_edits = edits or [
        Edit(path="calc.py", search="return abs(a) + abs(b)", replace="return a + b")
    ]
    attempt_id = store.record_attempt(
        run_id,
        1,
        plan_edits,
        "diff --git a/calc.py b/calc.py\n-    return abs(a) + abs(b)\n+    return a + b\n",
        fingerprint,
        ["calc.py"],
        True,
        None,
    )
    store.record_verification(
        attempt_id,
        VerificationReport(
            attempt=1,
            patch_sha256=fingerprint,
            baseline={"total": 5, "passed": 2, "failed": 3, "failed_tests": ["test_known_broken"]},
            patched={"total": 5, "passed": 4, "failed": 1},
            lint_ok=True,
            install_ok=True,
            passed=True,
        ),
    )
    approval = approve(store, fingerprint, approved_by="alice")
    return config, store, approval


class TestApprovalGate:
    def test_cannot_approve_a_patch_that_was_never_verified(self, tmp_path, fixture_repo):
        config = Config.load(env={}, db_path=tmp_path / "s.sqlite3")
        store = Store(config.db_path)
        run_id = store.start_run("e/r", "mock", "m", 1)
        store.record_attempt(run_id, 1, [], "patch", "a" * 64, ["x.py"], False, "failed")
        with pytest.raises(PublishError, match="did not pass verification"):
            approve(store, "a" * 64, approved_by="alice")
        store.close()


class TestDryRun:
    def test_dry_run_rebuilds_and_matches(self, tmp_path, fixture_repo, local_clone):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )

        result = publish(config, store, approval.approval_id, dry_run=True)
        assert result.fingerprint == fingerprint
        assert result.pr_url is None
        store.close()

    def test_dry_run_performs_no_github_writes(self, tmp_path, fixture_repo, local_clone):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        github = FakeGitHub()

        publish(config, store, approval.approval_id, github=github, dry_run=True)

        assert github.pulls_opened == []
        assert github.forks_created == []
        assert github.calls == []
        store.close()


class TestHashMismatch:
    def test_publish_aborts_before_pushing_when_the_hash_differs(
        self, tmp_path, fixture_repo, local_clone
    ):
        """The recorded edits no longer reproduce the approved hash."""
        base = git(fixture_repo, "rev-parse", "HEAD")
        true_fingerprint = real_fingerprint(fixture_repo, base, tmp_path)

        # Store a *different* fingerprint than the edits actually produce, as
        # would happen if the stored edits were tampered with after approval.
        config, store, approval = seed_approved(
            tmp_path,
            fixture_repo,
            fingerprint=true_fingerprint,
            edits=[Edit(path="calc.py", search="return abs(a) + abs(b)", replace="return a * b")],
        )
        github = FakeGitHub()

        with pytest.raises(PatchVerificationError, match="does not match"):
            publish(config, store, approval.approval_id, github=github)

        # Nothing may have reached GitHub.
        assert github.pulls_opened == []
        assert github.forks_created == []
        assert github.calls == [], "no GitHub call may happen before the hash is verified"

        # The approval must not be marked published.
        pending = publish_pending(config, store)
        assert len(pending) == 1
        assert pending[0]["publish_state"] == "approved"
        store.close()

    def test_an_unreachable_base_commit_aborts(
        self, tmp_path, fixture_repo, local_clone, no_real_push
    ):
        """If the recorded base commit cannot be resolved, publishing must fail."""
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        # Point the run at a well-formed commit that does not exist anywhere.
        missing = "0" * 39 + "1"
        store._conn.execute(
            "UPDATE runs SET base_commit=? WHERE id=(SELECT run_id FROM attempts WHERE id=?)",
            (missing, approval.attempt_id),
        )
        store._conn.commit()

        github = FakeGitHub()
        with pytest.raises(PatchVerificationError, match="not reachable"):
            publish(config, store, approval.approval_id, github=github)
        assert github.calls == []
        assert no_real_push == []
        store.close()


class TestSuccessfulPublish:
    def test_pushes_a_branch_and_opens_a_draft_pr(
        self, tmp_path, fixture_repo, local_clone, no_real_push
    ):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        github = FakeGitHub()
        pushed = no_real_push

        result = publish(config, store, approval.approval_id, github=github)

        assert result.fingerprint == fingerprint
        assert result.pr_number == 42
        assert result.pr_url.endswith("/pull/42")

        # Exactly one push, to the fork, with the publish token.
        assert len(pushed) == 1
        assert pushed[0]["fork"] == "tester"
        assert pushed[0]["token"] == "fake-publish-token"
        assert pushed[0]["branch"] == result.branch

        # Exactly one PR, and it must be a draft disclosing AI assistance.
        assert len(github.pulls_opened) == 1
        pr = github.pulls_opened[0]
        assert pr["head"] == f"tester:{result.branch}"
        assert pr["base"] == "main"
        assert "AI agent" in pr["body"]
        assert "draft" in pr["body"].lower()
        assert fingerprint in pr["body"]

        # The approval is now marked published.
        assert publish_pending(config, store) == []
        store.close()

    def test_the_pr_body_carries_the_verification_evidence(
        self, tmp_path, fixture_repo, local_clone, no_real_push
    ):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        github = FakeGitHub()

        publish(config, store, approval.approval_id, github=github)

        body = github.pulls_opened[0]["body"]
        assert "Fixes #1" in body
        assert "abs() loses the sign." in body
        assert "calc.py" in body
        assert "2 passed" in body, "the baseline summary must appear"
        assert "no new lint findings" in body
        store.close()

    def test_never_marks_the_pr_ready_for_review(
        self, tmp_path, fixture_repo, local_clone, no_real_push
    ):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        github = FakeGitHub()

        publish(config, store, approval.approval_id, github=github)

        # The fake records every call; none may be a "ready for review" action.
        forbidden = {"mark_ready_for_review", "merge", "merge_pull_request", "submit_review"}
        assert not any(call[0] in forbidden for call in github.calls)
        store.close()

    def test_a_push_failure_is_recorded_and_not_reported_as_success(
        self, tmp_path, fixture_repo, local_clone, monkeypatch, no_real_push
    ):
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        github = FakeGitHub()

        def failing_push(*args, **kwargs):
            from forgeflow.errors import ForgeFlowError

            raise ForgeFlowError("remote rejected the push")

        monkeypatch.setattr("forgeflow.gitops.push_branch", failing_push)

        with pytest.raises(Exception, match="rejected"):
            publish(config, store, approval.approval_id, github=github)

        assert github.pulls_opened == []
        pending = publish_pending(config, store)
        assert pending[0]["publish_state"] == "failed"
        store.close()

    def test_a_pr_failure_reports_the_pushed_branch(
        self, tmp_path, fixture_repo, local_clone, no_real_push
    ):
        """If the PR cannot be opened, the branch is still pushed and must be reported."""
        base = git(fixture_repo, "rev-parse", "HEAD")
        fingerprint = real_fingerprint(fixture_repo, base, tmp_path)
        config, store, approval = seed_approved(
            tmp_path, fixture_repo, fingerprint=fingerprint
        )
        class BrokenPR(FakeGitHub):
            def open_draft_pull_request(self, *args, **kwargs):
                from forgeflow.errors import GitHubError

                raise GitHubError("422 validation failed")

        with pytest.raises(PublishError, match="pushed to your fork"):
            publish(config, store, approval.approval_id, github=BrokenPR())

        pending = publish_pending(config, store)
        assert pending[0]["publish_state"] == "pushed-no-pr"
        assert pending[0]["branch"]
        store.close()
