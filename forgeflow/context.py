"""Choosing which repository files to show the model.

The model gets a small, deliberately chosen slice of the repository: file paths
that the issue itself names, plus a keyword-scored shortlist. Two reasons:

* Cost. The free tier has a token budget, so sending a whole repository is not
  an option.
* Safety. The fewer untrusted bytes reach the prompt, the smaller the surface
  for a prompt-injection attempt, and the easier it is to reason about what the
  model could possibly have been told.

Selection is pure heuristics. No model call happens here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .edits import is_forbidden_path

SOURCE_SUFFIXES = (".py", ".pyi")
MAX_FILE_BYTES = 120_000
DEFAULT_FILE_BUDGET = 8
DEFAULT_TOTAL_CHARS = 60_000

# Words that carry no signal when scoring files.
STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "for",
        "from",
        "with",
        "without",
        "into",
        "onto",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "in",
        "on",
        "of",
        "to",
        "not",
        "no",
        "yes",
        "can",
        "could",
        "should",
        "would",
        "will",
        "would",
        "please",
        "help",
        "issue",
        "bug",
        "error",
        "problem",
        "fix",
        "fixed",
        "broken",
        "crash",
        "crashes",
        "fails",
        "failing",
        "failed",
        "test",
        "tests",
        "testing",
        "use",
        "used",
        "using",
        "make",
        "makes",
        "made",
        "get",
        "gets",
        "getting",
        "set",
        "sets",
        "setting",
        "run",
        "runs",
        "running",
        "add",
        "adds",
        "adding",
        "new",
    ]
)

_PATH_IN_TEXT = re.compile(r"[\w./\\-]+\.(?:py|pyi|pyx|cfg|toml|ini|txt|rst|md)\b")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_FUNC_REF = re.compile(r"`?([A-Za-z_][A-Za-z0-9_.]*)`?\s*\(\)")


@dataclass
class SelectedFile:
    path: str
    content: str
    score: float
    reasons: list[str] = field(default_factory=list)
    truncated: bool = False


@dataclass
class FileSelection:
    files: list[SelectedFile] = field(default_factory=list)
    total_chars: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [item.path for item in self.files]

    def as_dict(self) -> dict[str, str]:
        return {item.path: item.content for item in self.files}


def _walk_repo(root: Path) -> list[Path]:
    """List candidate files, skipping anything the edit policy forbids."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        ignored_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox"}
        if any(part in ignored_dirs for part in path.parts):
            continue
        if is_forbidden_path(relative):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        found.append(path)
    return found


def mentioned_paths(text: str, available: set[str]) -> list[str]:
    """Paths the issue names explicitly, matched against what exists."""
    hits: list[str] = []
    for match in _PATH_IN_TEXT.finditer(text):
        raw = match.group(0).replace("\\", "/").lstrip("./")
        if raw in available and raw not in hits:
            hits.append(raw)
            continue
        # A traceback often gives an absolute path; match on the longest suffix.
        for candidate in available:
            if candidate.endswith("/" + raw) or candidate == raw:
                if candidate not in hits:
                    hits.append(candidate)
                break
    return hits


def keywords_from_text(text: str) -> set[str]:
    words = {word.lower() for word in _IDENTIFIER.findall(text)}
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def score_file(
    path: str, content: str, keywords: set[str], mentioned: bool
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if mentioned:
        score += 50.0
        reasons.append("named by the issue")

    stem = Path(path).stem.lower()
    if stem in keywords:
        score += 6.0
        reasons.append(f"filename matches issue vocabulary ({stem})")

    if path.startswith("tests/") or Path(path).stem.startswith("test_"):
        score -= 3.0
        reasons.append("test file (deprioritised as the fix target)")

    lowered = content.lower()
    matches = sum(1 for word in keywords if word in lowered)
    if matches:
        score += min(matches, 12) * 0.5
        reasons.append(f"{matches} issue keywords appear in the file")

    if "def " in content or "class " in content:
        score += 1.0

    lines = content.count("\n") + 1
    if lines > 2000:
        score -= 2.0
        reasons.append(f"large file ({lines} lines)")
    return score, reasons


def select_files(
    root: Path,
    issue_text: str,
    *,
    budget: int = DEFAULT_FILE_BUDGET,
    total_chars: int = DEFAULT_TOTAL_CHARS,
    prefer_paths: list[str] | None = None,
) -> FileSelection:
    """Pick the files to embed in the prompt."""
    candidates = _walk_repo(root)
    available = {path.relative_to(root).as_posix() for path in candidates}

    mentioned = mentioned_paths(issue_text, available)
    keywords = keywords_from_text(issue_text)

    ordered: list[str] = []
    for path in [*(prefer_paths or []), *mentioned]:
        normalised = path.replace("\\", "/").lstrip("./")
        if normalised in available and normalised not in ordered:
            ordered.append(normalised)

    scored: list[tuple[float, str, list[str]]] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if relative in ordered:
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        score, reasons = score_file(relative, content, keywords, relative in mentioned)
        scored.append((score, relative, reasons))

    scored.sort(key=lambda item: (-item[0], item[1]))

    selection = FileSelection()
    remaining = total_chars

    for relative in ordered:
        if len(selection.files) >= budget or remaining <= 0:
            break
        _add(selection, root, relative, 50.0, ["named by the issue"], remaining)
        remaining = total_chars - selection.total_chars

    for score, relative, reasons in scored:
        if len(selection.files) >= budget or remaining <= 0:
            break
        _add(selection, root, relative, score, reasons, remaining)
        remaining = total_chars - selection.total_chars

    return selection


def _add(
    selection: FileSelection,
    root: Path,
    relative: str,
    score: float,
    reasons: list[str],
    remaining: int,
) -> None:
    path = root / relative
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        selection.skipped.append((relative, f"unreadable: {exc}"))
        return
    truncated = False
    if len(content) > remaining:
        content = content[: max(0, remaining)]
        truncated = True
    if not content:
        selection.skipped.append((relative, "empty"))
        return
    selection.files.append(
        SelectedFile(
            path=relative,
            content=content,
            score=round(score, 3),
            reasons=reasons,
            truncated=truncated,
        )
    )
    selection.total_chars += len(content)


def detect_src_layout(root: Path) -> str:
    """Return the subdirectory tests should run from, for ``src/`` layouts.

    Tests are run from the repository root in v1; this only reports whether a
    ``src`` directory exists so the CLI can mention it.
    """
    return "src" if (root / "src").is_dir() else ""
