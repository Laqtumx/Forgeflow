"""Domain models.

Two groups live here:

* The GitHub-facing models (``Issue``, ``RepoRef``) describe untrusted input.
* The model-output models (``Analysis``, ``EditPlan``, ``Edit``) are the *only*
  shapes the LLM is allowed to produce. They are deliberately narrow: the model
  can name files and emit search/replace pairs, and nothing else. There is no
  field through which it could express a shell command or a raw diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------
# GitHub-facing models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoRef:
    """A parsed ``owner/name`` GitHub repository reference."""

    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.slug

    @classmethod
    def parse(cls, value: str) -> RepoRef:
        """Accept ``owner/name``, ``https://github.com/owner/name``, ``git@`` forms."""
        text = value.strip()
        text = re.sub(r"\.git$", "", text)
        text = re.sub(r"^git@github\.com:", "", text)
        text = re.sub(r"^https?://(www\.)?github\.com/", "", text)
        text = text.strip("/")
        parts = text.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"expected a GitHub repository like 'owner/name', got {value!r}"
            )
        return cls(owner=parts[0], name=parts[1])


@dataclass
class Issue:
    """An open issue, as returned by the GitHub REST API."""

    number: int
    title: str
    body: str
    html_url: str
    labels: tuple[str, ...] = ()
    assignees: tuple[str, ...] = ()
    comments: int = 0
    created_at: str = ""
    updated_at: str = ""
    author: str | None = None

    @classmethod
    def from_api(cls, payload: dict) -> Issue:
        labels = tuple(
            str(label.get("name", ""))
            for label in payload.get("labels", [])
            if isinstance(label, dict)
        )
        assignees = tuple(
            str(user.get("login", ""))
            for user in payload.get("assignees") or []
            if isinstance(user, dict)
        )
        user = payload.get("user") or {}
        return cls(
            number=int(payload["number"]),
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            html_url=str(payload.get("html_url") or ""),
            labels=labels,
            assignees=assignees,
            comments=int(payload.get("comments") or 0),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            author=str(user.get("login")) if user.get("login") else None,
        )

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}"


@dataclass
class RankedIssue:
    """An issue plus the heuristic score that ranked it."""

    issue: Issue
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def number(self) -> int:
        return self.issue.number


# --------------------------------------------------------------------------
# Model-output models (the LLM contract)
# --------------------------------------------------------------------------


class Analysis(BaseModel):
    """Structured stage-1 output."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    root_cause_hypothesis: str = Field(min_length=1, max_length=4000)
    files_to_inspect: list[str] = Field(default_factory=list, max_length=20)
    fix_plan: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    fixable: bool
    needs_more_info: bool = False

    @field_validator("files_to_inspect", "fix_plan")
    @classmethod
    def _strip_entries(cls, values: list[str]) -> list[str]:
        return [entry.strip() for entry in values if entry and entry.strip()]


class Edit(BaseModel):
    """One exact search/replace operation against a repository file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    search: str = Field(min_length=1)
    replace: str
    rationale: str = ""

    @field_validator("path")
    @classmethod
    def _normalise_path(cls, value: str) -> str:
        path = value.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        return path.strip("/")


class EditPlan(BaseModel):
    """Stage-2 output: the concrete edits, never a diff."""

    model_config = ConfigDict(extra="forbid")

    edits: list[Edit] = Field(min_length=1, max_length=10)
    notes: str = ""

    @field_validator("edits")
    @classmethod
    def _unique_paths(cls, edits: list[Edit]) -> list[Edit]:
        seen: set[str] = set()
        for edit in edits:
            key = f"{edit.path}\x00{edit.search}"
            if key in seen:
                raise ValueError(f"duplicate edit for {edit.path}")
            seen.add(key)
        return edits


class TestOutcome(BaseModel):
    """Parsed pytest result for a single run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failed_tests: list[str] = Field(default_factory=list)
    exit_code: int = 0
    collected: bool = True


@dataclass
class SandboxResult:
    """Raw outcome of one command executed inside the sandbox."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    def tail(self, limit: int = 4000) -> str:
        text = self.combined
        if len(text) <= limit:
            return text
        return "... [truncated] ...\n" + text[-limit:]


class VerificationReport(BaseModel):
    """Result of validating one candidate patch."""

    attempt: int
    patch_sha256: str
    baseline: TestOutcome | None = None
    patched: TestOutcome | None = None
    newly_failing: list[str] = Field(default_factory=list)
    newly_passing: list[str] = Field(default_factory=list)
    lint_ok: bool = False
    lint_output: str = ""
    install_ok: bool = False
    install_output: str = ""
    passed: bool = False
    failure_reason: str | None = None


PatchState = Literal[
    "proposed",   # analysis done, edits applied, verified, awaiting human review
    "rejected",   # human declined
    "publishing", # approval recorded, publish in progress
    "published",  # draft PR opened
    "failed",     # verification or publish failed
]
