"""Docker sandbox for running a repository's tests.

Design constraints, and how each is met:

* **Non-root.** Every command runs with ``--user <uid>:<gid>``. Nothing in the
  sandbox ever runs as root, including the test process.
* **No network during tests.** The test and lint runs use ``--network none``.
  Network is enabled only for the dependency-install step, which is a separate,
  explicitly-labelled command.
* **Resource limited.** ``--memory``, ``--memory-swap``, ``--cpus``,
  ``--pids-limit`` and a bounded ``--tmpfs /tmp``.
* **Capabilities dropped.** ``--cap-drop ALL`` and
  ``--security-opt no-new-privileges``.
* **Read-only source during tests.** The checkout is mounted read-only for test
  runs, so a test cannot rewrite the code under test and fake a pass. Test
  output is written to a separate results volume.
* **Isolation from the host.** The repository is copied into a named Docker
  volume; the sandbox never sees the host filesystem beyond that.
* **No credentials.** No token is ever passed into the sandbox, and the git
  environment is scrubbed.

Only ``python -m pytest`` runs here in v1. The test command is a configured
string, but it is executed inside the container, not on the host.
"""

from __future__ import annotations

import io
import re
import shlex
import subprocess
import tarfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .defaults import (
    DEFAULT_LINT_COMMAND,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_SANDBOX_PACKAGES,
    DEFAULT_TEST_COMMAND,
)
from .errors import SandboxError
from .models import SandboxResult, TestOutcome

SANDBOX_WORK = "/work"
SANDBOX_RESULTS = "/results"
SANDBOX_SITE = "/opt/ffsite"

PYTEST_JUNIT = "/results/pytest.xml"

# Keeps bytecode out of the read-only checkout mount.
PYCACHE_ENV = {"PYTHONPYCACHEPREFIX": "/tmp/pycache"}

# Summary line, e.g. "2 failed, 1 passed, 3 skipped in 1.20s"
SUMMARY_RE = re.compile(
    r"(?:(?P<count>\d+)\s+(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings))",
    re.IGNORECASE,
)


@dataclass
class SandboxConfig:
    image: str = DEFAULT_SANDBOX_IMAGE
    cpus: str = "2"
    memory: str = "2g"
    pids: int = 256
    tmpfs_size: str = "512m"
    run_uid: str = "65534:65534"
    test_command: str = DEFAULT_TEST_COMMAND
    lint_command: str = DEFAULT_LINT_COMMAND
    install_timeout_s: int = 600
    test_timeout_s: int = 600
    allow_network_during_tests: bool = False
    lint_packages: tuple[str, ...] = DEFAULT_SANDBOX_PACKAGES
    docker: str = "docker"
    extra_env: dict[str, str] = field(default_factory=dict)


class DockerSandbox:
    """Owns one isolated work volume for one verification run."""

    def __init__(self, config: SandboxConfig, *, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.work_volume = f"forgeflow-work-{self.run_id}"
        self.site_volume = f"forgeflow-site-{self.run_id}"
        self.results_volume = f"forgeflow-results-{self.run_id}"
        self._docker = shlex.split(config.docker)
        self._volumes_created = False

    # -- docker plumbing ---------------------------------------------------

    def _cmd(self, *args: str) -> list[str]:
        return [*self._docker, *args]

    def _run(
        self,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                self._cmd(*args),
                input=input_bytes,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"docker {' '.join(args[:4])} timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise SandboxError(
                f"docker executable not found ({self.config.docker!r}); "
                "set FORGEFLOW_DOCKER if it needs a prefix such as 'sudo docker'"
            ) from exc

    def _check(self, result: subprocess.CompletedProcess[bytes], what: str) -> None:
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", "replace")
            raise SandboxError(f"{what} failed (exit {result.returncode}): {stderr.strip()[:600]}")

    def available(self) -> tuple[bool, str]:
        """Report whether the sandbox can actually run here."""
        try:
            result = self._run(["version", "--format", "{{.Server.Version}}"], timeout=30)
        except SandboxError as exc:
            return False, str(exc)
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            return False, f"docker daemon unreachable: {detail[:300]}"
        return True, (result.stdout or b"").decode().strip()

    def ensure_image(self) -> None:
        probe = self._run(["image", "inspect", self.config.image], timeout=60)
        if probe.returncode == 0:
            return
        pull = self._run(["pull", self.config.image], timeout=900)
        self._check(pull, f"pulling {self.config.image}")

    # -- volumes -----------------------------------------------------------

    def create_volumes(self) -> None:
        self.ensure_image()
        for volume in (self.work_volume, self.site_volume, self.results_volume):
            result = self._run(["volume", "create", volume], timeout=60)
            self._check(result, f"creating volume {volume}")
        self._prepare_volume_permissions()
        self._volumes_created = True

    def _prepare_volume_permissions(self) -> None:
        """Make the writable volumes usable by the unprivileged sandbox user.

        Docker creates a fresh volume root owned by root:root, so a non-root
        ``pip install --target`` cannot write to it and pytest cannot write its
        JUnit report. One root-owned init step opens those directories; every
        step that runs repository or model-influenced content afterwards stays
        non-root.
        """
        uid = self.config.run_uid.split(":")[0]
        gid = self.config.run_uid.split(":")[1] if ":" in self.config.run_uid else uid
        writable = {
            self.site_volume: SANDBOX_SITE,
            self.results_volume: SANDBOX_RESULTS,
        }
        for volume, mount in writable.items():
            script = (
                f"mkdir -p {mount} && "
                f"chown -R {uid}:{gid} {mount} && "
                f"chmod -R u+rwX {mount} && "
                f"chmod 0777 {mount}"
            )
            result = self._run(
                [
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--cap-add",
                    "CHOWN",
                    "--cap-add",
                    "FOWNER",
                    "-v",
                    f"{volume}:{mount}",
                    self.config.image,
                    "sh",
                    "-c",
                    script,
                ],
                timeout=120,
            )
            self._check(result, f"preparing volume {volume}")

    def cleanup(self) -> None:
        if not self._volumes_created:
            return
        for volume in (self.work_volume, self.site_volume, self.results_volume):
            self._run(["volume", "rm", "-f", volume], timeout=60)
        self._volumes_created = False

    def __enter__(self) -> DockerSandbox:
        self.create_volumes()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()

    # -- moving files in and out ------------------------------------------

    def _tar_bytes(self, root: Path, paths: list[str]) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for relative in paths:
                archive.add(str(root / relative), arcname=relative)
        return buffer.getvalue()

    def _extract_into_volume(
        self, volume: str, destination: str, payload: bytes, *, writable: bool
    ) -> None:
        """Untar ``payload`` into ``volume`` at ``destination``.

        The volume is mounted at ``/mnt`` and the tar entries are relative
        paths, so the extraction target must be expressed relative to the
        mount point. Writing to an absolute ``/work`` here would land in the
        helper container's own filesystem, which is discarded on exit.
        """
        relative = destination.removeprefix(SANDBOX_WORK).lstrip("/")
        target = f"/mnt/{relative}" if relative else "/mnt"
        mode = "a+rwX" if writable else "a+rX"
        script = (
            f"mkdir -p {shlex.quote(target)} && "
            f"tar -xf - -C {shlex.quote(target)} && "
            f"chmod -R {mode} {shlex.quote(target)}"
        )
        result = self._run(
            [
                "run",
                "--rm",
                "-i",
                "-v",
                f"{volume}:/mnt",
                self.config.image,
                "sh",
                "-c",
                script,
            ],
            input_bytes=payload,
            timeout=600,
        )
        self._check(result, f"copying files into {volume}")

    def load_checkout(self, checkout: Path, *, subdir: str = "") -> None:
        """Copy a local checkout into the work volume.

        The git metadata is excluded: the sandbox needs the source, not the
        history, and keeping ``.git`` out removes a large amount of surface.
        """
        checkout = checkout.resolve()
        entries: list[str] = []
        for entry in sorted(checkout.iterdir()):
            if entry.name in {".git", ".forgeflow", ".forgeflow-candidate.patch"}:
                continue
            entries.append(entry.name)
        if not entries:
            raise SandboxError(f"checkout {checkout} is empty")
        destination = f"{SANDBOX_WORK}/{subdir}".rstrip("/")
        self._extract_into_volume(
            self.work_volume, destination, self._tar_bytes(checkout, entries), writable=False
        )

    def update_files(self, checkout: Path, relpaths: list[str]) -> None:
        """Push modified files from the local checkout into the work volume."""
        if not relpaths:
            return
        payload = self._tar_bytes(checkout.resolve(), relpaths)
        self._extract_into_volume(self.work_volume, SANDBOX_WORK, payload, writable=False)

    def read_results(self, filename: str = "pytest.xml") -> str | None:
        result = self._run(
            [
                "run",
                "--rm",
                "-v",
                f"{self.results_volume}:/r:ro",
                self.config.image,
                "cat",
                f"/r/{filename}",
            ],
            timeout=120,
        )
        if result.returncode != 0:
            return None
        return (result.stdout or b"").decode("utf-8", "replace")

    def clear_results(self) -> None:
        self._run(
            [
                "run",
                "--rm",
                "-v",
                f"{self.results_volume}:/r",
                self.config.image,
                "sh",
                "-c",
                "rm -f /r/*",
            ],
            timeout=60,
        )

    # -- running commands --------------------------------------------------

    def run(
        self,
        command: str,
        *,
        network: bool = False,
        timeout: int = 300,
        work_subdir: str = "",
        writable_work: bool = False,
        writable_site: bool = False,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run one shell command inside the sandbox."""
        working_dir = f"{SANDBOX_WORK}/{work_subdir}".rstrip("/") or SANDBOX_WORK
        args: list[str] = [
            "run",
            "--rm",
            "--network",
            "bridge" if network else "none",
            "--memory",
            self.config.memory,
            "--memory-swap",
            self.config.memory,
            "--cpus",
            self.config.cpus,
            "--pids-limit",
            str(self.config.pids),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            self.config.run_uid,
            "--tmpfs",
            f"/tmp:rw,size={self.config.tmpfs_size},mode=1777",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PIP_NO_CACHE_DIR=1",
            "-e",
            f"PYTHONPATH={self.pythonpath(work_subdir)}",
            "-v",
            f"{self.work_volume}:{SANDBOX_WORK}:{'rw' if writable_work else 'ro'}",
            "-v",
            f"{self.site_volume}:{SANDBOX_SITE}:{'rw' if writable_site else 'ro'}",
            "-v",
            f"{self.results_volume}:{SANDBOX_RESULTS}:rw",
            "-w",
            working_dir,
        ]
        for key, value in {
            **PYCACHE_ENV,
            **self.config.extra_env,
            **(env or {}),
        }.items():
            args += ["-e", f"{key}={value}"]
        args += [self.config.image, "sh", "-c", command]

        try:
            completed = self._run(args, timeout=timeout)
        except SandboxError as exc:
            return SandboxResult(
                command=command,
                exit_code=124,
                stdout="",
                stderr=str(exc),
                timed_out="timed out" in str(exc),
            )

        return SandboxResult(
            command=command,
            exit_code=completed.returncode,
            stdout=(completed.stdout or b"").decode("utf-8", "replace"),
            stderr=(completed.stderr or b"").decode("utf-8", "replace"),
        )

    def pythonpath(self, work_subdir: str = "") -> str:
        """Import path for tests.

        The checkout comes first, so the *patched* source is what the tests
        import. Third-party packages live on the site volume. ``src`` layouts
        are handled by adding the nested source directory too. The project is
        never pip-installed into the site volume, because that would shadow the
        patched checkout with a stale copy.
        """
        base = f"{SANDBOX_WORK}/{work_subdir}".rstrip("/") or SANDBOX_WORK
        return f"{base}:{base}/src:{SANDBOX_SITE}"

    # -- high-level steps --------------------------------------------------

    def install_dependencies(
        self, work_subdir: str = "", *, requirements: list[str] | None = None
    ) -> SandboxResult:
        """Install pytest, the linter, and the project's declared dependencies.

        This is the only step with network access, and it is deliberately
        separate from the test run so the results table can show that tests ran
        offline.
        """
        packages = " ".join(sorted({*self.config.lint_packages, *(requirements or [])}))
        script = (
            "set -e; "
            "python -m pip install --quiet --disable-pip-version-check "
            f"--target {SANDBOX_SITE} {packages}; "
            f"chmod -R a+rX {SANDBOX_SITE}; "
            "echo '[forgeflow] dependencies ready'"
        )
        return self.run(
            script,
            network=True,
            timeout=self.config.install_timeout_s,
            work_subdir=work_subdir,
            writable_site=True,
        )

    def run_tests(self, work_subdir: str = "") -> tuple[SandboxResult, TestOutcome]:
        """Run pytest with a JUnit report so per-test outcomes are exact."""
        self.clear_results()
        command = (
            f"{self.config.test_command} --junit-xml={PYTEST_JUNIT}"
            if "--junit-xml" not in self.config.test_command
            else self.config.test_command
        )
        result = self.run(
            command,
            network=self.config.allow_network_during_tests,
            timeout=self.config.test_timeout_s,
            work_subdir=work_subdir,
        )
        xml_text = self.read_results()
        outcome = parse_junit_xml(xml_text) if xml_text else parse_pytest_text(result.combined)
        outcome.exit_code = result.exit_code
        if result.timed_out:
            outcome.collected = False
        return result, outcome

    def run_lint(self, files: list[str], work_subdir: str = "") -> SandboxResult:
        if not files:
            return SandboxResult(command="", exit_code=0, stdout="", stderr="")
        quoted = " ".join(shlex.quote(path) for path in files)
        command = self.config.lint_command.format(files=quoted)
        return self.run(command, network=False, timeout=300, work_subdir=work_subdir)


def parse_lint_findings(text: str) -> dict[str, set[str]]:
    """Parse ruff's ``concise`` output into ``{path: {finding codes}}``.

    Comparing the baseline's findings with the patched findings means a
    pre-existing lint error in an untouched line is not attributed to the patch,
    the same way a pre-existing test failure is not.
    """
    findings: dict[str, set[str]] = {}
    pattern = re.compile(
        r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+):\s*(?P<code>[A-Z]+\d+)"
    )
    for raw_line in text.splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        path = match.group("path")
        code = match.group("code")
        findings.setdefault(path, set()).add(code)
    return findings


def compare_lint(
    baseline: dict[str, set[str]], patched: dict[str, set[str]], changed: list[str]
) -> list[str]:
    """Return findings that are new *and* in a file the patch changed."""
    changed_set = set(changed)
    regressions: list[str] = []
    for path, codes in patched.items():
        if path not in changed_set:
            continue
        new_codes = codes - baseline.get(path, set())
        for code in sorted(new_codes):
            regressions.append(f"{path}: {code}")
    return regressions


# --- result parsing -------------------------------------------------------


def parse_pytest_text(text: str) -> TestOutcome:
    """Fallback parser for pytest's text summary."""
    outcome = TestOutcome()
    if "no tests ran" in text or "collected 0 items" in text:
        outcome.collected = False

    summary_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if SUMMARY_RE.search(stripped) and (
            " in " in stripped or "=" in stripped or stripped.endswith("s")
        ):
            summary_line = stripped
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for match in SUMMARY_RE.finditer(summary_line):
        kind = match.group("kind").lower()
        if kind in {"error", "errors"}:
            kind = "error"
        counts[kind] = int(match.group("count"))
    outcome.passed = counts["passed"]
    outcome.failed = counts["failed"]
    outcome.errors = counts["error"]
    outcome.skipped = counts["skipped"]
    outcome.total = sum(counts.values())

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("FAILED ", "ERROR ")):
            parts = stripped.split()
            if len(parts) >= 2:
                outcome.failed_tests.append(parts[1])
    outcome.failed_tests = sorted(set(outcome.failed_tests))
    if "errors during collection" in text or "ERROR collecting" in text:
        outcome.collected = False
    return outcome


def parse_junit_xml(xml_text: str) -> TestOutcome:
    """Parse a JUnit report into exact per-test outcomes."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return TestOutcome(collected=False)

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return TestOutcome(collected=False)

    outcome = TestOutcome()
    failed: list[str] = []
    for suite in suites:
        for case in suite.iter("testcase"):
            classname = case.get("classname") or ""
            name = case.get("name") or ""
            node_id = f"{classname}::{name}" if classname else name
            status = "passed"
            for child in case:
                tag = child.tag
                if tag == "failure":
                    status = "failed"
                elif tag == "error":
                    status = "error"
                elif tag == "skipped":
                    status = "skipped"
            outcome.total += 1
            if status == "passed":
                outcome.passed += 1
            elif status == "failed":
                outcome.failed += 1
                failed.append(node_id)
            elif status == "error":
                outcome.errors += 1
                failed.append(node_id)
            else:
                outcome.skipped += 1

    outcome.failed_tests = sorted(set(failed))
    outcome.collected = outcome.total > 0
    return outcome


def compare_outcomes(
    baseline: TestOutcome, patched: TestOutcome
) -> tuple[list[str], list[str]]:
    """Compare baseline and patched runs.

    Returns ``(newly_failing, newly_passing)``.

    ``newly_failing`` is what matters: a test that already failed before the
    patch is not the patch's fault, so it is excluded. This is the mechanism
    that stops ForgeFlow blaming a patch for pre-existing breakage.
    """
    baseline_failed = set(baseline.failed_tests)
    patched_failed = set(patched.failed_tests)
    newly_failing = sorted(patched_failed - baseline_failed)
    newly_passing = sorted(baseline_failed - patched_failed)
    return newly_failing, newly_passing


def summary_line(outcome: TestOutcome) -> str:
    return (
        f"{outcome.passed} passed, {outcome.failed} failed, "
        f"{outcome.errors} errors, {outcome.skipped} skipped"
    )


def docker_available(docker: str = "docker") -> tuple[bool, str]:
    """Module-level convenience used by the CLI preflight."""
    sandbox = DockerSandbox(SandboxConfig(docker=docker))
    return sandbox.available()


def discover_requirements(checkout: Path) -> list[str]:
    """Read declared dependencies from a checkout, best effort.

    Only pinned-ish ``name`` / ``name==version`` lines are taken; VCS URLs, local
    paths, and extras-with-URLs are skipped because they would need network
    access to arbitrary hosts during install.
    """
    requirements: list[str] = []
    candidates = ["requirements.txt", "requirements-dev.txt", "requirements/base.txt"]
    for relative in candidates:
        path = checkout / relative
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            if any(token in line for token in ("git+", "http://", "https://", "file:", "/", "\\")):
                continue
            requirements.append(line)
    # Keep the list short: ForgeFlow's job is to run the tests, not to build the
    # project's full environment.
    return requirements[:40]
