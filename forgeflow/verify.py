"""Patch verification.

One attempt is validated like this:

1. Restore the local checkout to the pinned base commit, so a failed attempt
   cannot leak into the next one.
2. Push the pristine files the patch intends to touch into the sandbox and run
   the test suite and the linter. That is the *baseline*.
3. Apply the edits locally, push the changed files, and run the suite and the
   linter again. That is the *patched* run.
4. Compare. A patch passes only if:
   - it introduces no test that fails after the patch but passed before,
   - it does not break collection, and
   - it introduces no new lint finding in a file it changed.

Pre-existing failures are carried in the report but never counted against the
patch. If the baseline itself cannot even be collected, the repository is
reported as unverifiable rather than as a passing fix, because "0 tests ran" is
not evidence that anything works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import gitops
from .defaults import DEFAULT_LINT_SELECT
from .edits import apply_edits, fingerprint_patch
from .errors import EditError
from .models import EditPlan, SandboxResult, TestOutcome, VerificationReport
from .sandbox import (
    DockerSandbox,
    compare_lint,
    compare_outcomes,
    discover_requirements,
    parse_lint_findings,
    summary_line,
)


@dataclass
class Baseline:
    """The state of the repository before any patch is applied."""

    outcome: TestOutcome
    lint: dict[str, set[str]] = field(default_factory=dict)
    install_ok: bool = False
    install_output: str = ""
    collected: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether a patch can be judged at all against this baseline."""
        return self.install_ok and self.collected


@dataclass
class AttemptOutcome:
    """Everything one attempt produced, for storage and for the results table."""

    report: VerificationReport
    patch_text: str | None = None
    files_changed: list[str] = field(default_factory=list)
    error_for_model: str | None = None


class Verifier:
    """Runs the baseline once, then judges each attempt against it."""

    def __init__(
        self,
        sandbox: DockerSandbox,
        checkout: Path,
        *,
        base_commit: str,
        work_subdir: str = "",
        lint_select: str = DEFAULT_LINT_SELECT,
    ) -> None:
        self.sandbox = sandbox
        self.checkout = checkout
        self.base_commit = base_commit
        self.work_subdir = work_subdir
        self.lint_select = lint_select
        self.baseline: Baseline | None = None
        self._installed = False

    # -- setup -------------------------------------------------------------

    def install(self, requirements: list[str] | None = None) -> SandboxResult:
        if requirements is None:
            requirements = discover_requirements(self.checkout)
        result = self.sandbox.install_dependencies(
            self.work_subdir, requirements=requirements
        )
        self._installed = result.exit_code == 0
        return result

    def reset_checkout(self) -> None:
        """Return the local checkout to the pinned base commit."""
        gitops.run_git(["checkout", "--detach", self.base_commit], cwd=self.checkout, check=True)
        gitops.restore_all(self.checkout)

    def capture_baseline(self, touch_files: list[str] | None = None) -> Baseline:
        """Run the suite on the pristine checkout.

        ``touch_files`` lets the caller pre-lint the files a patch is expected
        to modify, so the lint comparison has a per-file reference point.
        """
        self.reset_checkout()
        self.sandbox.load_checkout(self.checkout, subdir=self.work_subdir)

        _, outcome = self.sandbox.run_tests(self.work_subdir)
        notes: list[str] = []

        lint_findings: dict[str, set[str]] = {}
        if touch_files:
            lint = self.sandbox.run_lint(touch_files, self.work_subdir)
            lint_findings = parse_lint_findings(lint.combined)

        if not self._installed:
            notes.append("dependency installation did not complete; baseline may be incomplete")
        if not outcome.collected:
            notes.append(
                "the test suite did not collect any tests on the unpatched checkout"
            )
        exited_without_failure = (
            outcome.exit_code != 0
            and outcome.failed == 0
            and outcome.errors == 0
            and outcome.collected
        )
        if exited_without_failure:
            notes.append(
                f"pytest exited {outcome.exit_code} without reporting a failing test; "
                "treating the baseline as suspect"
            )

        self.baseline = Baseline(
            outcome=outcome,
            lint=lint_findings,
            install_ok=self._installed,
            collected=outcome.collected,
            notes=notes,
        )
        return self.baseline

    # -- per-attempt verification -----------------------------------------

    def verify(self, plan: EditPlan, attempt: int) -> AttemptOutcome:
        """Apply ``plan`` and judge it against the baseline."""
        if self.baseline is None:
            raise RuntimeError("capture_baseline() must run before verify()")

        target_files = sorted({edit.path for edit in plan.edits})
        report = VerificationReport(
            attempt=attempt,
            patch_sha256="",
            baseline=self.baseline.outcome,
        )

        # --- apply locally -------------------------------------------------
        self.reset_checkout()
        try:
            apply_result = apply_edits(self.checkout, plan)
        except EditError as exc:
            report.passed = False
            report.failure_reason = f"edits could not be applied: {exc}"
            report.lint_ok = False
            report.install_ok = self.baseline.install_ok
            return AttemptOutcome(
                report=report,
                files_changed=target_files,
                error_for_model=(
                    f"Applying your edits failed: {exc}\n"
                    "Every 'search' string must appear exactly once in the named file, "
                    "copied verbatim from the file contents you were shown."
                ),
            )

        patch_text = gitops.diff(self.checkout)
        if not patch_text.strip():
            report.passed = False
            report.failure_reason = "the edits produced no change to the working tree"
            return AttemptOutcome(
                report=report,
                files_changed=apply_result.files_changed,
                error_for_model=(
                    "Your edits matched the file but produced no change. "
                    "Check that 'replace' differs from 'search'."
                ),
            )

        report.patch_sha256 = fingerprint_patch(patch_text)
        report.install_ok = self.baseline.install_ok
        report.install_output = self.baseline.install_output

        # --- run the patched code ------------------------------------------
        self.sandbox.update_files(self.checkout, apply_result.files_changed)
        test_result, patched_outcome = self.sandbox.run_tests(self.work_subdir)
        report.patched = patched_outcome

        # --- lint the changed files ----------------------------------------
        lint_result = self.sandbox.run_lint(apply_result.files_changed, self.work_subdir)
        patched_lint = parse_lint_findings(lint_result.combined)
        regressions = compare_lint(self.baseline.lint, patched_lint, apply_result.files_changed)
        report.lint_ok = not regressions
        report.lint_output = (
            "new lint findings: " + "; ".join(regressions)
            if regressions
            else "no new lint findings in changed files"
        )

        # --- compare --------------------------------------------------------
        newly_failing, newly_passing = compare_outcomes(self.baseline.outcome, patched_outcome)
        report.newly_failing = newly_failing
        report.newly_passing = newly_passing

        problems: list[str] = []
        if not patched_outcome.collected:
            problems.append("the test suite no longer collects after the patch")
        if newly_failing:
            problems.append("newly failing tests: " + ", ".join(newly_failing[:10]))
        if not report.lint_ok:
            problems.append("new lint findings: " + "; ".join(regressions[:10]))
        if test_result.timed_out:
            problems.append("the test run timed out after the patch")

        report.passed = not problems
        report.failure_reason = "; ".join(problems) if problems else None

        return AttemptOutcome(
            report=report,
            patch_text=patch_text,
            files_changed=apply_result.files_changed,
            error_for_model=(
                _feedback_for_model(report, test_result, lint_result, self.baseline)
                if problems
                else None
            ),
        )


def _feedback_for_model(
    report: VerificationReport,
    test_result: SandboxResult,
    lint_result: SandboxResult,
    baseline: Baseline,
) -> str:
    """Build the error text handed back to the model for the next attempt.

    Only the sandbox's own output is included, and it is labelled as data. The
    baseline's pre-existing failures are stated so the model does not chase
    them.
    """
    parts: list[str] = []
    if report.failure_reason:
        parts.append(f"Verification failed: {report.failure_reason}")

    if baseline.outcome.failed_tests:
        parts.append(
            "These tests were ALREADY failing before your patch, so ignore them: "
            + ", ".join(baseline.outcome.failed_tests[:10])
        )

    if report.newly_failing:
        parts.append("Tests your patch broke: " + ", ".join(report.newly_failing[:10]))

    if not report.lint_ok:
        parts.append("Lint regressions: " + report.lint_output)

    parts.append("Test output (untrusted):\n" + test_result.tail(3000))
    if not report.lint_ok:
        parts.append("Lint output (untrusted):\n" + lint_result.tail(1000))
    parts.append(f"Baseline test summary: {summary_line(baseline.outcome)}")
    if report.patched:
        parts.append(f"Patched test summary: {summary_line(report.patched)}")
    return "\n\n".join(parts)
