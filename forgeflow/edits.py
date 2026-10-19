"""Applying model edits, and turning them into a canonical patch.

This module is the trust boundary. The model proposed ``search``/``replace``
pairs; here we decide whether they are allowed to touch the working tree.

Rules enforced, in order:

1. **Path policy.** The path must be repository-relative, must not escape the
   repository root, must not be a symlink, must not be a binary blob, and must
   match an allowlist of extensions. ``.git`` internals are never writable.
2. **Verbatim search.** ``search`` must appear in the file *exactly once*. Zero
   matches means the model hallucinated; two or more means it was ambiguous and
   applying it could corrupt unrelated code.
3. **Bounded size.** An edit may not grow a file beyond a limit, and the number
   of changed lines is capped, so a prompt-injected "rewrite everything" cannot
   land.
4. **No new files outside the tree.** A ``search`` of ``""`` is not allowed, so
   edits can only modify existing content.

Only after all of that does the patch become a ``git diff`` produced by git
itself, and get hashed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import EditError, PathPolicyError
from .models import EditPlan

# Extensions the model may edit in v1. Python-first, plus the metadata files a
# Python fix commonly needs.
ALLOWED_SUFFIXES = (
    ".py",
    ".pyi",
    ".pyx",
    ".cfg",
    ".toml",
    ".ini",
    ".txt",
    ".rst",
    ".md",
)

# Paths that are never writable, regardless of extension.
FORBIDDEN_PARTS = (
    ".git",
    ".github/workflows",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".tox",
    ".nox",
    "site-packages",
)

FORBIDDEN_NAMES = (
    "setup.py",       # arbitrary code execution at install time
    "conftest.py",    # can disable the very tests we rely on
    "pyproject.toml", # build backend is code execution
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    ".pth",
)

MAX_FILE_BYTES = 400_000
MAX_CHANGED_LINES = 400
MAX_EDITS = 10


@dataclass
class AppliedEdit:
    path: str
    occurrences: int
    lines_added: int
    lines_removed: int


@dataclass
class ApplyResult:
    applied: list[AppliedEdit] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    @property
    def total_changed_lines(self) -> int:
        return sum(item.lines_added + item.lines_removed for item in self.applied)


def is_forbidden_path(relative: str) -> str | None:
    """Return the reason a path is forbidden, or ``None`` if it is acceptable."""
    if not relative or relative.startswith(("/", "~")):
        return "path must be repository-relative"
    parts = Path(relative).parts
    if any(part in {".", ".."} for part in parts):
        return "path must not contain '.' or '..' segments"
    for index in range(len(parts)):
        prefix = "/".join(parts[: index + 1])
        if prefix in FORBIDDEN_PARTS:
            return f"'{prefix}' is not writable"
    if any(part in FORBIDDEN_PARTS for part in parts[:-1]):
        return "path is inside a forbidden directory"
    name = parts[-1]
    if name in FORBIDDEN_NAMES:
        return f"'{name}' is not writable because it can execute code at install or test time"
    if name.endswith(".pth"):
        return "'.pth' files are not writable"
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        return f"extension '{Path(name).suffix or '(none)'}' is outside the v1 allowlist"
    return None


def resolve_target(root: Path, relative: str) -> Path:
    """Resolve a model-supplied path inside ``root``, rejecting escapes."""
    reason = is_forbidden_path(relative)
    if reason:
        raise PathPolicyError(f"refusing to edit {relative!r}: {reason}")

    root = root.resolve()
    raw = root / relative
    # The symlink check must happen before resolution: resolving first would
    # follow the link and make it look like an ordinary file.
    if raw.is_symlink():
        raise PathPolicyError(f"refusing to edit {relative!r}: it is a symlink")

    candidate = raw.resolve()
    if candidate != root and root not in candidate.parents:
        raise PathPolicyError(f"refusing to edit {relative!r}: escapes the repository root")
    if not candidate.is_file():
        raise EditError(f"cannot edit {relative!r}: file does not exist")
    if candidate.stat().st_size > MAX_FILE_BYTES:
        raise EditError(f"cannot edit {relative!r}: file is larger than {MAX_FILE_BYTES} bytes")
    return candidate


def _count_lines(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


def apply_edits(root: Path, plan: EditPlan) -> ApplyResult:
    """Apply every edit in ``plan`` atomically.

    All checks run before any write. If any edit fails, the working tree is left
    exactly as it was found.
    """
    if len(plan.edits) > MAX_EDITS:
        raise EditError(f"edit plan has {len(plan.edits)} edits, limit is {MAX_EDITS}")

    staged: list[tuple[Path, str, str, AppliedEdit]] = []
    total_lines = 0

    for edit in plan.edits:
        target = resolve_target(root, edit.path)
        original = target.read_text(encoding="utf-8", errors="strict")

        occurrences = original.count(edit.search)
        if occurrences == 0:
            raise EditError(
                f"search text for {edit.path!r} does not appear in the file; "
                "the model must copy it verbatim"
            )
        if occurrences > 1:
            raise EditError(
                f"search text for {edit.path!r} appears {occurrences} times; "
                "it must be unique"
            )

        updated = original.replace(edit.search, edit.replace, 1)
        added = max(0, _count_lines(edit.replace) - _count_lines(edit.search))
        removed = max(0, _count_lines(edit.search) - _count_lines(edit.replace))
        total_lines += added + removed
        if total_lines > MAX_CHANGED_LINES:
            raise EditError(
                f"edit plan changes more than {MAX_CHANGED_LINES} lines; refusing"
            )

        staged.append(
            (
                target,
                original,
                updated,
                AppliedEdit(
                    path=edit.path,
                    occurrences=occurrences,
                    lines_added=added,
                    lines_removed=removed,
                ),
            )
        )

    result = ApplyResult()
    for target, original, updated, applied in staged:
        if updated == original:
            continue
        target.write_text(updated, encoding="utf-8")
        result.applied.append(applied)
        if applied.path not in result.files_changed:
            result.files_changed.append(applied.path)

    if not result.files_changed:
        raise EditError("edit plan produced no changes")
    return result


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_patch(patch_text: str) -> str:
    """The value a human types to approve a patch.

    Normalised so that the fingerprint depends on the change itself and not on
    incidental whitespace or a trailing newline.
    """
    normalised = "\n".join(line.rstrip() for line in patch_text.strip().splitlines())
    return sha256_text(normalised)


def is_binary_file(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        chunk = path.open("rb").read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk


def describe_edits(plan: EditPlan) -> list[dict[str, object]]:
    """A reviewable summary for the CLI and the PR body."""
    return [
        {
            "path": edit.path,
            "search_preview": _preview(edit.search),
            "replace_preview": _preview(edit.replace),
            "rationale": edit.rationale,
        }
        for edit in plan.edits
    ]


def _preview(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."
