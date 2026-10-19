"""Pipeline orchestration.

The whole run, in order:

    rank issues -> pick one -> clone at a pinned commit -> capture the baseline
    -> call 1: structured analysis -> gate -> call 2: search/replace edits
    -> verify in the sandbox -> on failure, feed the error back and retry
    -> record everything -> stop at "proposed", awaiting a human fingerprint

Two things this module deliberately does *not* do: it never publishes, and it
never merges. Publishing is a separate step that requires the human to type the
patch fingerprint, and it re-derives the patch from the base commit before it
touches the network.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import gitops
from .config import Config
from .context import FileSelection, select_files
from .errors import ConfigError, ForgeFlowError, GitHubError
from .github_client import GitHubClient
from .llm import (
    LLMClient,
    PromptContext,
    Provider,
    analysis_is_actionable,
    build_provider,
)
from .models import Analysis, EditPlan, RankedIssue, RepoRef, VerificationReport
from .ranking import (
    CandidateSet,
    RankingOptions,
    filter_and_rank,
    score_issue,
    select_best,
)
from .sandbox import DockerSandbox, SandboxConfig, summary_line
from .store import Store
from .verify import AttemptOutcome, Baseline, Verifier

# Files handed to the model in one prompt. Small on purpose: free-tier quota is
# the binding constraint, and fewer untrusted bytes is also a smaller
# prompt-injection surface.
PROMPT_FILE_BUDGET = 8


@dataclass
class RunResult:
    """The complete outcome of one run, suitable for printing or archiving."""

    run_id: int
    repo: str
    status: str
    issue: RankedIssue | None = None
    candidates: CandidateSet | None = None
    analysis: Analysis | None = None
    analysis_skip_reason: str | None = None
    baseline: Baseline | None = None
    attempts: list[AttemptOutcome] = field(default_factory=list)
    base_commit: str | None = None
    selected_files: FileSelection | None = None
    usage: dict[str, object] = field(default_factory=dict)
    error_stage: str | None = None
    error_message: str | None = None

    @property
    def accepted(self) -> AttemptOutcome | None:
        """The first attempt that passed verification, if any."""
        for attempt in self.attempts:
            if attempt.report.passed:
                return attempt
        return None

    @property
    def fingerprint(self) -> str | None:
        accepted = self.accepted
        return accepted.report.patch_sha256 if accepted else None

    @property
    def is_publishable(self) -> bool:
        return self.status == "proposed" and bool(self.fingerprint)


class Pipeline:
    """Runs ForgeFlow against one repository."""

    def __init__(
        self,
        config: Config,
        *,
        store: Store | None = None,
        github: GitHubClient | None = None,
        provider: Provider | None = None,
        sandbox: DockerSandbox | None = None,
        clone_url: str | None = None,
        echo=print,
    ) -> None:
        self.config = config
        # Test seam: lets the end-to-end test clone a local fixture repository
        # instead of reaching GitHub. Unused in normal operation.
        self.clone_url = clone_url
        self._store = store
        self._owns_store = store is None
        self.github = github or GitHubClient(
            config.github_read_token, config.github_publish_token
        )
        self.provider = provider or build_provider(config)
        self._sandbox = sandbox
        self.echo = echo
        self.llm = LLMClient(self.provider)

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.config.db_path)
        return self._store

    def close(self) -> None:
        if self._store is not None and self._owns_store:
            self._store.close()
            self._store = None

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- helpers -----------------------------------------------------------

    def _sandbox_config(self) -> SandboxConfig:
        return SandboxConfig(
            image=self.config.sandbox_image,
            cpus=self.config.sandbox_cpus,
            memory=self.config.sandbox_memory,
            pids=self.config.sandbox_pids,
            test_command=self.config.test_command,
            lint_command=self.config.lint_command,
            install_timeout_s=self.config.install_timeout_s,
            test_timeout_s=self.config.test_timeout_s,
            allow_network_during_tests=self.config.allow_network_during_tests,
            docker=self.config.docker,
        )

    def preflight(self) -> list[str]:
        """Return a list of blocking problems; empty means the run can proceed."""
        problems: list[str] = []
        if self.config.llm_provider != "mock" and not self.config.provider_api_key():
            var = (
                "GEMINI_API_KEY"
                if self.config.llm_provider == "gemini"
                else "ANTHROPIC_API_KEY"
            )
            problems.append(
                f"{var} is not set (required for FORGEFLOW_LLM_PROVIDER="
                f"{self.config.llm_provider}). Use --provider mock to run offline."
            )
        if not self.config.github_read_token:
            problems.append(
                "GITHUB_TOKEN is not set; unauthenticated GitHub reads are rate "
                "limited to 60 requests per hour."
            )
        sandbox = self._sandbox or DockerSandbox(self._sandbox_config())
        ok, detail = sandbox.available()
        if not ok:
            problems.append(f"Docker is not usable: {detail}")
        return problems

    # -- main entry point --------------------------------------------------

    def run(self, repo_input: str, issue_number: int | None = None) -> RunResult:
        try:
            repo = RepoRef.parse(repo_input)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        default_branch = self.github.get_default_branch(repo)
        run_id = self.store.start_run(
            repo.slug,
            self.config.llm_provider,
            self.config.model_name(),
            issue_number,
            default_branch=default_branch,
        )
        result = RunResult(run_id=run_id, repo=repo.slug, status="running")
        stage = "start"
        checkout: Path | None = None
        sandbox: DockerSandbox | None = None
        owns_sandbox = self._sandbox is None

        try:
            stage = "fetch-issues"
            candidates, issue = self._select_issue(repo, issue_number)
            result.candidates = candidates
            result.issue = issue
            self.store.record_issues(
                run_id, candidates.ranked, issue.issue.number if issue else None
            )
            if issue is None:
                return self._finish(result, run_id, "no-candidate")

            stage = "clone"
            checkout, base_commit = self._clone(repo, run_id)
            result.base_commit = base_commit

            sandbox = self._sandbox or DockerSandbox(
                self._sandbox_config(), run_id=f"{run_id}-{base_commit[:8]}"
            )
            if owns_sandbox:
                sandbox.create_volumes()

            verifier = Verifier(sandbox, checkout, base_commit=base_commit)

            stage = "install"
            install = verifier.install()
            if install.exit_code != 0:
                result.error_stage = "install"
                result.error_message = install.tail(600)
                self.store.finish_run(
                    run_id, "unverifiable", base_commit, "install", result.error_message
                )
                result.status = "unverifiable"
                return result

            stage = "baseline"
            result.selected_files = select_files(
                checkout, issue.issue.text, budget=PROMPT_FILE_BUDGET
            )
            baseline = verifier.capture_baseline(touch_files=result.selected_files.paths)
            result.baseline = baseline
            if not baseline.collected:
                result.error_stage = "baseline"
                result.error_message = (
                    "; ".join(baseline.notes) or "the test suite collected no tests"
                )
                self.store.finish_run(
                    run_id, "unverifiable", base_commit, "baseline", result.error_message
                )
                result.status = "unverifiable"
                return result

            stage = "analysis"
            analysis, skip_reason = self._analyse(repo, issue, checkout, base_commit, run_id)
            result.analysis = analysis
            result.analysis_skip_reason = skip_reason
            if analysis is None:
                return self._finish(result, run_id, "not-actionable", base_commit)

            stage = "attempts"
            self._run_attempts(repo, issue, checkout, verifier, analysis, result)
            status = "proposed" if result.accepted else "verification-failed"
            return self._finish(result, run_id, status, base_commit)

        except ForgeFlowError as exc:
            result.status = "failed"
            result.error_stage = stage
            result.error_message = str(exc)
            self.store.finish_run(run_id, "failed", result.base_commit, stage, str(exc))
            return result
        finally:
            result.usage = self.llm.usage_summary()
            if owns_sandbox and sandbox is not None:
                sandbox.cleanup()
            if checkout is not None:
                self._cleanup_workdir(checkout)

    def _finish(
        self, result: RunResult, run_id: int, status: str, base_commit: str | None = None
    ) -> RunResult:
        result.status = status
        self.store.finish_run(run_id, status, base_commit)
        return result

    # -- stages ------------------------------------------------------------

    def _select_issue(
        self, repo: RepoRef, issue_number: int | None
    ) -> tuple[CandidateSet, RankedIssue | None]:
        meta = self.github.get_repo(repo)
        if meta.get("archived"):
            raise GitHubError(f"{repo.slug} is archived; not accepting patches")
        size_kb = int(meta.get("size") or 0)
        if size_kb > self.config.repo_size_limit_kb:
            raise GitHubError(
                f"{repo.slug} is {size_kb} KB, above the configured limit of "
                f"{self.config.repo_size_limit_kb} KB"
            )
        language = (meta.get("language") or "").lower()
        if language and language != "python":
            raise GitHubError(
                f"{repo.slug} is primarily {meta.get('language') or 'unknown'}; "
                "version 1 supports Python repositories"
            )

        issues = self.github.list_open_issues(repo, limit=self.config.max_candidates * 4)
        options = RankingOptions(excluded_labels=self.config.excluded_labels)

        if issue_number is not None:
            chosen = next((item for item in issues if item.number == issue_number), None)
            if chosen is None:
                raise GitHubError(
                    f"issue #{issue_number} is not among the open non-PR issues of {repo.slug}"
                )
            linked = self.github.linked_pull_requests(repo, chosen.number)
            candidates = filter_and_rank(
                issues, options=options, limit=self.config.max_candidates
            )
            ranked = next(
                (item for item in candidates.ranked if item.issue.number == issue_number),
                None,
            )
            if ranked is None:
                # The caller asked for this issue explicitly, so score it even
                # if the filter would have dropped it.
                ranked = score_issue(chosen, has_linked_pr=bool(linked))
            return candidates, ranked

        linked: dict[int, list[object]] = {}
        for issue in issues[: self.config.max_candidates]:
            found = self.github.linked_pull_requests(repo, issue.number)
            if found:
                linked[issue.number] = list(found)

        candidates = filter_and_rank(
            issues,
            linked_prs=linked,
            options=options,
            limit=self.config.max_candidates,
        )
        return candidates, select_best(candidates)

    def _clone(self, repo: RepoRef, run_id: int) -> tuple[Path, str]:
        checkout = self.config.work_dir / f"run-{run_id}" / "repo"
        base_commit = gitops.clone(
            repo,
            checkout,
            depth=self.config.clone_depth,
            token=self.config.github_read_token,
            url=self.clone_url,
        )
        return checkout, base_commit

    def _build_context(
        self, repo: RepoRef, issue: RankedIssue, checkout: Path, base_commit: str
    ) -> PromptContext:
        selection = select_files(checkout, issue.issue.text, budget=PROMPT_FILE_BUDGET)
        return PromptContext(
            issue=issue.issue,
            repo_slug=repo.slug,
            files=selection.as_dict(),
            base_commit=base_commit,
        )

    def _analyse(
        self,
        repo: RepoRef,
        issue: RankedIssue,
        checkout: Path,
        base_commit: str,
        run_id: int,
    ) -> tuple[Analysis | None, str | None]:
        """Call 1, then apply the gate before spending call 2."""
        context = self._build_context(repo, issue, checkout, base_commit)
        analysis = self.llm.analyse(context)
        self.store.record_analysis(run_id, issue.issue.number, analysis)

        allowed = list(context.files.keys())
        ok, reason = analysis_is_actionable(
            analysis,
            threshold=self.config.confidence_threshold,
            allowed_files=allowed or None,
        )
        return (analysis, None) if ok else (analysis, reason)

    def _run_attempts(
        self,
        repo: RepoRef,
        issue: RankedIssue,
        checkout: Path,
        verifier: Verifier,
        analysis: Analysis,
        result: RunResult,
    ) -> None:
        """Call the model for edits, verify, and feed failures back."""
        context = self._build_context(repo, issue, checkout, result.base_commit or "")

        previous_error: str | None = None
        previous_plan: EditPlan | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                plan = self.llm.propose_edits(
                    context,
                    analysis,
                    previous_error=previous_error,
                    previous_edits=previous_plan,
                )
            except ForgeFlowError as exc:
                reason = f"the model did not return a usable edit plan: {exc}"
                result.attempts.append(
                    AttemptOutcome(
                        report=VerificationReport(
                            attempt=attempt,
                            patch_sha256="",
                            passed=False,
                            failure_reason=reason,
                        )
                    )
                )
                self.store.record_attempt(
                    result.run_id, attempt, [], None, None, [], False, reason
                )
                previous_error = str(exc)
                continue

            outcome = verifier.verify(plan, attempt)
            result.attempts.append(outcome)

            attempt_id = self.store.record_attempt(
                result.run_id,
                attempt,
                plan.edits,
                outcome.patch_text,
                outcome.report.patch_sha256 or None,
                outcome.files_changed,
                outcome.report.passed,
                outcome.report.failure_reason,
            )
            self.store.record_verification(attempt_id, outcome.report)

            if outcome.report.passed:
                return
            previous_error = outcome.error_for_model
            previous_plan = plan

    def _cleanup_workdir(self, checkout: Path) -> None:
        if self.config.keep_workdirs:
            return
        root = checkout.parent
        if root.exists() and root.is_dir():
            shutil.rmtree(root, ignore_errors=True)


def run_once(
    config: Config,
    repo_input: str,
    issue_number: int | None = None,
    *,
    store: Store | None = None,
    github: GitHubClient | None = None,
    provider: Provider | None = None,
    sandbox: DockerSandbox | None = None,
    clone_url: str | None = None,
    echo=print,
) -> RunResult:
    """Convenience wrapper used by the CLI and the evaluation harness."""
    pipeline = Pipeline(
        config,
        store=store,
        github=github,
        provider=provider,
        sandbox=sandbox,
        clone_url=clone_url,
        echo=echo,
    )
    try:
        return pipeline.run(repo_input, issue_number)
    finally:
        pipeline.close()


def rank_only(
    config: Config,
    repo_input: str,
    issue_number: int | None = None,
    *,
    store: Store | None = None,
    github: GitHubClient | None = None,
) -> RunResult:
    """Rank and analyse without cloning or touching Docker.

    Used by the ranking evaluation, and useful on a machine with no Docker
    daemon.
    """
    repo = RepoRef.parse(repo_input)
    pipeline = Pipeline(config, store=store, github=github, provider=None)
    result = RunResult(run_id=0, repo=repo.slug, status="running")
    try:
        run_id = pipeline.store.start_run(
            repo.slug, config.llm_provider, config.model_name(), issue_number
        )
        result.run_id = run_id
        candidates, issue = pipeline._select_issue(repo, issue_number)
        result.candidates = candidates
        result.issue = issue
        pipeline.store.record_issues(
            run_id, candidates.ranked, issue.issue.number if issue else None
        )
        if issue is None:
            result.status = "no-candidate"
            pipeline.store.finish_run(run_id, "no-candidate")
            return result

        context = PromptContext(
            issue=issue.issue, repo_slug=repo.slug, files={}, base_commit=None
        )
        analysis = pipeline.llm.analyse(context)
        result.analysis = analysis
        pipeline.store.record_analysis(run_id, issue.issue.number, analysis)
        ok, reason = analysis_is_actionable(
            analysis, threshold=config.confidence_threshold
        )
        result.analysis_skip_reason = None if ok else reason
        result.status = "analysed"
        pipeline.store.finish_run(run_id, "analysed")
        return result
    except ForgeFlowError as exc:
        result.status = "failed"
        result.error_message = str(exc)
        return result
    finally:
        result.usage = pipeline.llm.usage_summary()
        pipeline.close()


def summarise(result: RunResult) -> str:
    """A compact, human-readable summary of a run."""
    lines = [f"run {result.run_id} on {result.repo}: {result.status}"]
    if result.error_stage:
        lines.append(f"  failed at: {result.error_stage}")
        lines.append(f"  error: {(result.error_message or '')[:500]}")
    if result.issue:
        lines.append(
            f"  issue #{result.issue.issue.number}: {result.issue.issue.title[:100]}"
        )
        lines.append(f"  heuristic score: {result.issue.score}")
        lines.append("  why: " + "; ".join(result.issue.reasons[:6]))
    if result.candidates:
        lines.append(
            f"  candidates: {len(result.candidates.ranked)} ranked, "
            f"{len(result.candidates.dropped)} dropped"
        )
    if result.analysis_skip_reason:
        lines.append(f"  skipped: {result.analysis_skip_reason}")
    if result.analysis:
        lines.append(
            f"  analysis: confidence {result.analysis.confidence:.2f}, "
            f"fixable={result.analysis.fixable}, "
            f"needs_more_info={result.analysis.needs_more_info}"
        )
        if result.analysis.files_to_inspect:
            lines.append(
                "  files the analysis named: "
                + ", ".join(result.analysis.files_to_inspect[:6])
            )
    if result.baseline:
        lines.append(f"  baseline tests: {summary_line(result.baseline.outcome)}")
        for note in result.baseline.notes:
            lines.append(f"  baseline note: {note}")
    for attempt in result.attempts:
        report = attempt.report
        verdict = "PASS" if report.passed else "fail"
        detail = f" ({report.failure_reason})" if report.failure_reason else ""
        lines.append(f"  attempt {report.attempt}: {verdict}{detail}")
        if report.patched:
            lines.append(f"    patched tests: {summary_line(report.patched)}")
        if report.newly_passing:
            lines.append("    tests newly passing: " + ", ".join(report.newly_passing[:5]))
    if result.fingerprint:
        lines.append(f"  fingerprint: {result.fingerprint}")
    if result.usage:
        lines.append(
            f"  model: {result.usage.get('calls')} calls, "
            f"{result.usage.get('total_tokens')} tokens, "
            f"{result.usage.get('latency_s')}s"
        )
    return "\n".join(lines)
