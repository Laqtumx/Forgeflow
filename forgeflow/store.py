"""SQLite persistence.

Everything ForgeFlow learns is written here: runs, the issues it saw, the
analysis it produced, every attempt, every verification report, and the patch
fingerprint that a human approved. The store is intentionally plain SQL over
``sqlite3`` so it is inspectable with any SQLite client, and so the evaluation
harness can aggregate results without importing the pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    repo          TEXT NOT NULL,
    issue_number  INTEGER,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    base_commit   TEXT,
    default_branch TEXT,
    status        TEXT NOT NULL,
    error_stage   TEXT,
    error_message TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS issues (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    number       INTEGER NOT NULL,
    title        TEXT NOT NULL,
    score        REAL NOT NULL,
    selected     INTEGER NOT NULL DEFAULT 0,
    reasons      TEXT NOT NULL DEFAULT '[]',
    UNIQUE (run_id, number)
);

CREATE TABLE IF NOT EXISTS analyses (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    issue_number         INTEGER NOT NULL,
    summary              TEXT NOT NULL,
    root_cause_hypothesis TEXT NOT NULL,
    files_to_inspect     TEXT NOT NULL DEFAULT '[]',
    fix_plan             TEXT NOT NULL DEFAULT '[]',
    confidence           REAL NOT NULL,
    fixable              INTEGER NOT NULL,
    needs_more_info      INTEGER NOT NULL DEFAULT 0,
    raw                  TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt       INTEGER NOT NULL,
    edits_json    TEXT NOT NULL DEFAULT '[]',
    patch_text    TEXT,
    patch_sha256  TEXT,
    files_changed TEXT NOT NULL DEFAULT '[]',
    passed        INTEGER NOT NULL DEFAULT 0,
    failure_reason TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (run_id, attempt)
);

CREATE TABLE IF NOT EXISTS verifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id      INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    report_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id     INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    fingerprint    TEXT NOT NULL,
    approved_by    TEXT NOT NULL,
    approved_at    TEXT NOT NULL,
    publish_state  TEXT NOT NULL DEFAULT 'approved',
    branch         TEXT,
    pr_url         TEXT,
    error_message  TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_run ON issues(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_run ON attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_fingerprint ON approvals(fingerprint);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thin wrapper around a SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- runs --------------------------------------------------------------

    def start_run(
        self,
        repo: str,
        provider: str,
        model: str,
        issue_number: int | None = None,
        default_branch: str | None = None,
    ) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO runs(repo, issue_number, provider, model, default_branch,"
                " status, started_at) VALUES(?, ?, ?, ?, ?, 'running', ?)",
                (repo, issue_number, provider, model, default_branch, utcnow()),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        base_commit: str | None = None,
        error_stage: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET status=?, base_commit=COALESCE(?, base_commit),"
                " error_stage=?, error_message=?, finished_at=? WHERE id=?",
                (status, base_commit, error_stage, error_message, utcnow(), run_id),
            )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # -- issues ------------------------------------------------------------

    def record_issues(self, run_id: int, ranked: list[Any], selected_number: int | None) -> None:
        with self._tx() as conn:
            for item in ranked:
                conn.execute(
                    "INSERT OR REPLACE INTO issues(run_id, number, title, score, selected, reasons)"
                    " VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        item.issue.number,
                        item.issue.title,
                        float(item.score),
                        1 if item.issue.number == selected_number else 0,
                        json.dumps(item.reasons),
                    ),
                )

    def record_issue_titles(self, run_id: int, titles: dict[int, str]) -> None:
        """Fill in titles for issues already recorded for this run.

        ``record_issues`` stores titles for ranked candidates; this covers an
        explicitly requested issue that was filtered out of the ranking.
        """
        if not titles:
            return
        with self._tx() as conn:
            for number, title in titles.items():
                conn.execute(
                    "UPDATE issues SET title=? WHERE run_id=? AND number=?",
                    (title, run_id, number),
                )

    def selected_issue(self, run_id: int) -> int | None:
        row = self._conn.execute(
            "SELECT number FROM issues WHERE run_id=? AND selected=1 LIMIT 1", (run_id,)
        ).fetchone()
        return int(row["number"]) if row else None

    # -- analyses ----------------------------------------------------------

    def record_analysis(self, run_id: int, issue_number: int, analysis: Any) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO analyses(run_id, issue_number, summary, root_cause_hypothesis,"
                " files_to_inspect, fix_plan, confidence, fixable,"
                " needs_more_info, raw, created_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    issue_number,
                    analysis.summary,
                    analysis.root_cause_hypothesis,
                    json.dumps(analysis.files_to_inspect),
                    json.dumps(analysis.fix_plan),
                    float(analysis.confidence),
                    1 if analysis.fixable else 0,
                    1 if analysis.needs_more_info else 0,
                    analysis.model_dump_json(),
                    utcnow(),
                ),
            )
            return int(cursor.lastrowid)

    # -- attempts ----------------------------------------------------------

    def record_attempt(
        self,
        run_id: int,
        attempt: int,
        edits: list[Any],
        patch_text: str | None,
        patch_sha256: str | None,
        files_changed: list[str],
        passed: bool,
        failure_reason: str | None = None,
    ) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT OR REPLACE INTO attempts(run_id, attempt, edits_json, patch_text,"
                " patch_sha256, files_changed, passed, failure_reason, created_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    attempt,
                    json.dumps([e.model_dump() if hasattr(e, "model_dump") else e for e in edits]),
                    patch_text,
                    patch_sha256,
                    json.dumps(files_changed),
                    1 if passed else 0,
                    failure_reason,
                    utcnow(),
                ),
            )
            return int(cursor.lastrowid)

    def get_run_by_attempt(self, attempt_id: int) -> dict[str, Any] | None:
        """Everything the publish path needs, joined from one attempt.

        Returning one flat dict keeps ``publish`` free of SQL and makes it
        obvious exactly which recorded facts the PR body is built from.
        """
        row = self._conn.execute(
            """
            SELECT
                r.id                AS run_id,
                r.repo              AS repo,
                r.issue_number      AS issue_number,
                r.base_commit       AS base_commit,
                r.default_branch    AS default_branch,
                r.provider          AS provider,
                r.model             AS model,
                i.title             AS issue_title,
                a.summary           AS analysis_summary,
                a.root_cause_hypothesis AS analysis_hypothesis,
                a.confidence        AS analysis_confidence,
                t.attempt           AS attempt,
                t.files_changed     AS files_changed,
                (SELECT report_json FROM verifications v
                  WHERE v.attempt_id = t.id ORDER BY v.id DESC LIMIT 1) AS report_json
            FROM attempts t
            JOIN runs r     ON r.id = t.run_id
            LEFT JOIN issues i
                   ON i.run_id = r.id AND i.number = r.issue_number
            LEFT JOIN analyses a
                   ON a.run_id = r.id AND a.issue_number = r.issue_number
            WHERE t.id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)

        baseline_summary = patched_summary = lint_summary = ""
        report_json = data.pop("report_json", None)
        if report_json:
            try:
                report = json.loads(report_json)
            except ValueError:
                report = {}
            baseline = report.get("baseline") or {}
            patched = report.get("patched") or {}
            baseline_summary = (
                f"{baseline.get('passed', 0)} passed, {baseline.get('failed', 0)} failed, "
                f"{baseline.get('errors', 0)} errors, {baseline.get('skipped', 0)} skipped"
            )
            patched_summary = (
                f"{patched.get('passed', 0)} passed, {patched.get('failed', 0)} failed, "
                f"{patched.get('errors', 0)} errors, {patched.get('skipped', 0)} skipped"
            )
            lint_summary = (
                "no new lint findings in changed files"
                if report.get("lint_ok")
                else str(report.get("lint_output") or "lint regressions recorded")
            )
        data["baseline_summary"] = baseline_summary
        data["patched_summary"] = patched_summary
        data["lint_summary"] = lint_summary
        return data

    def get_attempt(self, attempt_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_attempt_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM attempts WHERE patch_sha256=? ORDER BY id DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return dict(row) if row else None

    def record_verification(self, attempt_id: int, report: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO verifications(attempt_id, report_json, created_at) VALUES(?, ?, ?)",
                (attempt_id, report.model_dump_json(), utcnow()),
            )

    # -- approvals ---------------------------------------------------------

    def record_approval(
        self, attempt_id: int, fingerprint: str, approved_by: str
    ) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO approvals(attempt_id, fingerprint, approved_by, approved_at,"
                " publish_state) VALUES(?, ?, ?, ?, 'approved')",
                (attempt_id, fingerprint, approved_by, utcnow()),
            )
            return int(cursor.lastrowid)

    def update_approval(
        self,
        approval_id: int,
        publish_state: str,
        branch: str | None = None,
        pr_url: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE approvals SET publish_state=?, branch=COALESCE(?, branch),"
                " pr_url=COALESCE(?, pr_url), error_message=? WHERE id=?",
                (publish_state, branch, pr_url, error_message, approval_id),
            )

    # -- reporting ---------------------------------------------------------

    def run_summary(self, run_id: int) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            return {}
        attempts = [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt", (run_id,)
            )
        ]
        approvals = [
            dict(row)
            for row in self._conn.execute(
                "SELECT a.* FROM approvals a JOIN attempts t ON t.id = a.attempt_id"
                " WHERE t.run_id=? ORDER BY a.id",
                (run_id,),
            )
        ]
        return {"run": run, "attempts": attempts, "approvals": approvals}

    def all_runs(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._conn.execute("SELECT * FROM runs ORDER BY id")
        ]
