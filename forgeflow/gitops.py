"""Git operations.

Three jobs:

* ``clone`` — an isolated shallow clone of the target repository.
* ``diff`` — the canonical patch, produced by git rather than by string
  concatenation, so it applies cleanly with ``git apply``.
* ``rebuild_patch`` — re-apply the approved edits to a *fresh* checkout of the
  same base commit and confirm the patch is byte-identical to what the human
  approved. This is what stops "approve A, publish B".

Everything runs with a scrubbed environment: no user git config, no credential
helpers, no interactive prompts. Credentials are passed per-command by the
caller when a push is needed, never written to disk.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .edits import apply_edits, fingerprint_patch
from .errors import ForgeFlowError, PatchVerificationError
from .models import EditPlan, RepoRef

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

BOT_IDENTITY = {
    "GIT_AUTHOR_NAME": "ForgeFlow",
    "GIT_AUTHOR_EMAIL": "forgeflow@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "ForgeFlow",
    "GIT_COMMITTER_EMAIL": "forgeflow@users.noreply.github.com",
}


@dataclass
class GitResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def combined(self) -> str:
        return (self.stdout + self.stderr).strip()


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A git environment that cannot read user config or prompt for input."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "true",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_LFS_SKIP_SMUDGE": "1",
        **BOT_IDENTITY,
    }
    if extra:
        env.update(extra)
    return env


def run_git(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    check: bool = False,
) -> GitResult:
    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=scrubbed_env(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ForgeFlowError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    result = GitResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=command,
    )
    if check and not result.ok:
        raise ForgeFlowError(f"git {' '.join(args)} failed: {result.combined[:500]}")
    return result


def clone(
    repo: RepoRef,
    destination: Path,
    *,
    depth: int = 50,
    branch: str | None = None,
    token: str | None = None,
    timeout: int = 600,
    url: str | None = None,
) -> str:
    """Shallow-clone ``repo`` into ``destination`` and return the HEAD commit.

    The remote is stripped afterwards: everything downstream works against a
    local checkout at a pinned commit, so a later push cannot accidentally read
    from this clone's origin.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote = url or repo.clone_url
    if token and url is None:
        remote = f"https://x-access-token:{token}@github.com/{repo.slug}.git"

    args = ["clone", "--depth", str(depth), "--no-tags", "--single-branch"]
    if branch:
        args += ["--branch", branch]
    args += [remote, str(destination)]

    result = run_git(args, cwd=destination.parent, timeout=timeout)
    if not result.ok:
        raise ForgeFlowError(
            f"clone of {repo.slug} failed: {_redact(result.combined, token)[:500]}"
        )

    head = rev_parse(destination, "HEAD")
    if not head:
        raise ForgeFlowError(f"clone of {repo.slug} produced no HEAD commit")
    return head


def _redact(text: str, token: str | None) -> str:
    if token:
        return text.replace(token, "<token>")
    return text


def rev_parse(cwd: Path, ref: str = "HEAD") -> str | None:
    """Resolve ``ref`` to a commit SHA, or ``None`` if it does not exist.

    ``--verify`` matters: plain ``git rev-parse <sha>`` echoes an
    unresolvable object name and exits 0, which would let a missing commit
    slip past the reachability check in ``rebuild_patch``.
    """
    result = run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=cwd)
    value = result.stdout.strip()
    return value if result.ok and value else None


def current_branch(cwd: Path) -> str | None:
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    value = result.stdout.strip()
    return value if result.ok and value and value != "HEAD" else None


def head_commit(cwd: Path) -> str | None:
    return rev_parse(cwd, "HEAD")


def is_dirty(cwd: Path) -> bool:
    result = run_git(["status", "--porcelain"], cwd=cwd)
    return bool(result.stdout.strip())


def diff(cwd: Path, *, staged: bool = False) -> str:
    """The canonical patch for the current working tree."""
    args = ["diff", "--no-color", "--no-ext-diff", "--binary"]
    if staged:
        args.append("--cached")
    result = run_git(args, cwd=cwd, check=True)
    return result.stdout


def diff_stat(cwd: Path) -> str:
    result = run_git(["diff", "--stat", "--no-color"], cwd=cwd)
    return result.stdout.strip()


def changed_files(cwd: Path) -> list[str]:
    result = run_git(["diff", "--name-only"], cwd=cwd)
    if not result.ok:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def restore_all(cwd: Path) -> None:
    """Discard every working-tree change."""
    run_git(["checkout", "--", "."], cwd=cwd)
    run_git(["clean", "-fd"], cwd=cwd)


def apply_patch(cwd: Path, patch_text: str, *, three_way: bool = False) -> GitResult:
    args = ["apply", "--whitespace=nowarn"]
    if three_way:
        args.append("--3way")
    return run_git([*args, "-"], cwd=cwd, env=None, timeout=120)


def _write_patch(cwd: Path, patch_text: str) -> Path:
    patch_path = cwd / ".forgeflow-candidate.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    return patch_path


def apply_patch_file(cwd: Path, patch_text: str) -> None:
    """Apply a patch from a file, using ``git apply`` for exactness."""
    patch_path = _write_patch(cwd, patch_text)
    try:
        result = run_git(
            ["apply", "--whitespace=nowarn", str(patch_path)], cwd=cwd, timeout=120
        )
        if not result.ok:
            raise ForgeFlowError(f"git apply failed: {result.combined[:500]}")
    finally:
        patch_path.unlink(missing_ok=True)


def commit_all(cwd: Path, message: str) -> str:
    run_git(["add", "-A"], cwd=cwd, check=True)
    result = run_git(["commit", "-m", message, "--no-verify"], cwd=cwd)
    if not result.ok:
        raise ForgeFlowError(f"git commit failed: {result.combined[:500]}")
    head = head_commit(cwd)
    if not head:  # pragma: no cover - defensive
        raise ForgeFlowError("commit succeeded but HEAD is unreadable")
    return head


@dataclass
class RebuiltPatch:
    patch_text: str
    fingerprint: str
    head_commit: str
    #: The working tree the patch was rebuilt in. The publish path commits and
    #: pushes from this directory, so it must be the one actually created
    #: rather than a path the caller guessed.
    checkout: Path


def rebuild_patch(
    repo: RepoRef,
    base_commit: str,
    plan: EditPlan,
    expected_fingerprint: str,
    work_dir: Path,
    *,
    token: str | None = None,
    depth: int = 50,
) -> RebuiltPatch:
    """Rebuild the approved patch from the base commit and verify its hash.

    Steps: fresh clone -> check out ``base_commit`` -> re-apply the same edits
    -> diff -> hash. If the hash differs from what the human approved, we abort
    before anything is pushed.
    """
    if not COMMIT_RE.match(base_commit):
        raise PatchVerificationError(f"base commit {base_commit!r} is not a full SHA")

    # A unique directory per call, so a retried publish cannot reuse a stale
    # tree and two publishes cannot collide.
    work_dir.mkdir(parents=True, exist_ok=True)
    checkout = work_dir / f"publish-{uuid4().hex[:12]}"
    clone(repo, checkout, depth=depth, token=token)

    if rev_parse(checkout, base_commit) is None:
        # The base commit may be older than the shallow window; deepen. A
        # local clone has no upstream, so the fetch failing is not fatal on
        # its own; what matters is whether the commit becomes resolvable.
        run_git(["fetch", "--deepen", "200"], cwd=checkout, timeout=600)
        if rev_parse(checkout, base_commit) is None:
            raise PatchVerificationError(
                f"base commit {base_commit} is not reachable in the clone"
            )

    run_git(["checkout", "--detach", base_commit], cwd=checkout, check=True)
    apply_edits(checkout, plan)

    patch_text = diff(checkout)
    if not patch_text.strip():
        raise PatchVerificationError("rebuilt patch is empty")

    fingerprint = fingerprint_patch(patch_text)
    if fingerprint != expected_fingerprint:
        raise PatchVerificationError(
            "rebuilt patch fingerprint does not match the approved fingerprint; "
            f"approved {expected_fingerprint}, rebuilt {fingerprint}. "
            "Nothing was pushed."
        )

    head = head_commit(checkout)
    if not head:  # pragma: no cover - defensive
        raise PatchVerificationError("rebuilt checkout has no HEAD")
    return RebuiltPatch(
        patch_text=patch_text,
        fingerprint=fingerprint,
        head_commit=head,
        checkout=checkout,
    )


def prepare_branch(
    checkout: Path,
    base_commit: str,
    branch: str,
    message: str,
) -> str:
    """Detach at ``base_commit``, commit the current changes, return the new SHA."""
    run_git(["checkout", "--detach", base_commit], cwd=checkout, check=True)
    run_git(["checkout", "-B", branch], cwd=checkout, check=True)
    return commit_all(checkout, message)


def push_branch(
    checkout: Path,
    repo_name: str,
    fork_owner: str,
    branch: str,
    token: str,
    *,
    force: bool = False,
) -> None:
    """Push ``branch`` to the fork using the publish token, in-command only."""
    url = f"https://x-access-token:{token}@github.com/{fork_owner}/{repo_name}.git"
    args = ["push", url, f"refs/heads/{branch}:refs/heads/{branch}"]
    if force:
        args.insert(1, "--force-with-lease")
    result = run_git(args, cwd=checkout, timeout=600)
    if not result.ok:
        raise ForgeFlowError(
            f"push to {fork_owner}/{repo_name} failed: {_redact(result.combined, token)[:500]}"
        )
