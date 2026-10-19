"""SQLite store and the fingerprint-gated approval path."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeflow.errors import PublishError
from forgeflow.models import Analysis, Edit, VerificationReport
from forgeflow.publish import _normalise_fingerprint, approve, publish_pending
from forgeflow.store import Store
from tests.conftest import make_issue


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "test.sqlite3")
    yield store
    store.close()


def seed_run(store: Store, *, passed: bool = True, fingerprint: str = "a" * 64):
    run_id = store.start_run("example/repo", "mock", "mock-deterministic", 1, default_branch="main")
    store.record_issues(run_id, [], 1)
    analysis = Analysis(
        summary="s",
        root_cause_hypothesis="h",
        files_to_inspect=["calc.py"],
        fix_plan=["fix"],
        confidence=0.9,
        fixable=True,
    )
    store.record_analysis(run_id, 1, analysis)
    attempt_id = store.record_attempt(
        run_id,
        1,
        [Edit(path="calc.py", search="a - b", replace="a + b")],
        "diff --git a/calc.py b/calc.py\n+    return a + b\n",
        fingerprint,
        ["calc.py"],
        passed,
        None if passed else "newly failing tests: test_add",
    )
    report = VerificationReport(
        attempt=1,
        patch_sha256=fingerprint,
        baseline={"total": 2, "passed": 1, "failed": 1, "failed_tests": ["test_known_broken"]},
        patched={"total": 2, "passed": 2, "failed": 0},
        lint_ok=True,
        install_ok=True,
        passed=passed,
    )
    store.record_verification(attempt_id, report)
    return run_id, attempt_id


class TestSchema:
    def test_tables_exist(self, store):
        names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"runs", "issues", "analyses", "attempts", "verifications", "approvals"} <= names

    def test_schema_version_recorded(self, store):
        row = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row["value"] == "1"

    def test_foreign_keys_are_on(self, store):
        row = store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_reopening_is_idempotent(self, tmp_path):
        path = tmp_path / "again.sqlite3"
        first = Store(path)
        run_id = first.start_run("a/b", "mock", "m", 1)
        first.close()
        second = Store(path)
        assert second.get_run(run_id) is not None
        second.close()


class TestRuns:
    def test_start_and_finish(self, store):
        run_id = store.start_run("a/b", "mock", "m", 7)
        assert store.get_run(run_id)["status"] == "running"
        store.finish_run(run_id, "proposed", "c" * 40)
        run = store.get_run(run_id)
        assert run["status"] == "proposed"
        assert run["base_commit"] == "c" * 40
        assert run["finished_at"]

    def test_records_the_failure_stage(self, store):
        run_id = store.start_run("a/b", "mock", "m", 1)
        store.finish_run(run_id, "failed", error_stage="clone", error_message="boom")
        run = store.get_run(run_id)
        assert run["error_stage"] == "clone"
        assert run["error_message"] == "boom"

    def test_issues_are_recorded_with_their_reasons(self, store):
        from forgeflow.ranking import score_issue

        run_id = store.start_run("a/b", "mock", "m", 1)
        ranked = [score_issue(make_issue(1)), score_issue(make_issue(2))]
        store.record_issues(run_id, ranked, selected_number=1)
        assert store.selected_issue(run_id) == 1
        rows = list(store._conn.execute("SELECT * FROM issues WHERE run_id=?", (run_id,)))
        assert len(rows) == 2
        assert "unassigned" in rows[0]["reasons"]


class TestPublishJoin:
    def test_join_returns_everything_the_pr_body_needs(self, store):
        _, attempt_id = seed_run(store)
        data = store.get_run_by_attempt(attempt_id)
        assert data["repo"] == "example/repo"
        assert data["issue_number"] == 1
        assert data["default_branch"] == "main"
        assert data["analysis_summary"] == "s"
        assert data["analysis_hypothesis"] == "h"
        assert "passed" in data["baseline_summary"]
        assert data["lint_summary"] == "no new lint findings in changed files"

    def test_join_handles_a_missing_attempt(self, store):
        assert store.get_run_by_attempt(9999) is None


class TestApproval:
    def test_approving_a_verified_fingerprint(self, store):
        _, attempt_id = seed_run(store, fingerprint="b" * 64)
        approval = approve(store, "b" * 64, approved_by="alice")
        assert approval.attempt_id == attempt_id
        assert approval.fingerprint == "b" * 64

    def test_normalises_a_prefixed_or_uppercase_fingerprint(self, store):
        seed_run(store, fingerprint="c" * 64)
        approval = approve(store, "SHA256:" + "C" * 64, approved_by="alice")
        assert approval.fingerprint == "c" * 64

    def test_rejects_a_malformed_fingerprint(self, store):
        with pytest.raises(PublishError, match="not a SHA-256"):
            approve(store, "nope", approved_by="alice")

    def test_rejects_a_fingerprint_that_matches_nothing(self, store):
        seed_run(store, fingerprint="d" * 64)
        with pytest.raises(PublishError, match="no verified attempt"):
            approve(store, "e" * 64, approved_by="alice")

    def test_rejects_a_failed_attempt(self, store):
        seed_run(store, passed=False, fingerprint="f" * 64)
        with pytest.raises(PublishError, match="did not pass verification"):
            approve(store, "f" * 64, approved_by="alice")

    def test_pending_lists_unpublished_approvals(self, store):
        seed_run(store, fingerprint="1" * 64)
        approve(store, "1" * 64, approved_by="alice")
        pending = publish_pending(None, store)
        assert len(pending) == 1
        assert pending[0]["publish_state"] == "approved"


class TestNormalisation:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("ABCDEF", "abcdef"),
            ("sha256:abcdef", "abcdef"),
            ("  abcdef  ", "abcdef"),
            ("SHA256:ABCDEF", "abcdef"),
        ],
    )
    def test_normalise(self, value, expected):
        assert _normalise_fingerprint(value) == expected
