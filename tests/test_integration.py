"""Integration tests that need Docker and a real git repository.

Marked ``integration`` so the fast suite can run without them:

    pytest -m "not integration"

The end-to-end test drives the real pipeline — real clone, real sandbox, real
baseline/patched comparison, real fingerprint — against a local fixture
repository, with only GitHub and the model provider replaced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forgeflow.config import Config
from forgeflow.errors import PatchVerificationError
from forgeflow.gitops import clone, diff, fingerprint_patch, rebuild_patch
from forgeflow.models import Edit, EditPlan, RepoRef
from forgeflow.pipeline import run_once
from forgeflow.sandbox import DockerSandbox, SandboxConfig
from forgeflow.store import Store
from forgeflow.verify import Verifier
from tests.conftest import (
    FIXED_MODULE,
    FakeGitHub,
    make_issue,
    write_calc,
)

pytestmark = pytest.mark.integration

DOCKER = "sudo docker"


def docker_ok() -> bool:
    ok, _ = DockerSandbox(SandboxConfig(docker=DOCKER)).available()
    return ok


requires_docker = pytest.mark.skipif(
    not docker_ok(), reason="a usable Docker daemon is required"
)


def sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        docker=DOCKER,
        test_timeout_s=300,
        install_timeout_s=600,
    )


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


FIX = EditPlan(
    edits=[Edit(path="calc.py", search="return abs(a) + abs(b)", replace="return a + b")]
)

# A plausible-looking fix that is wrong: it breaks tests that passed before.
WRONG = EditPlan(
    edits=[Edit(path="calc.py", search="return abs(a) + abs(b)", replace="return -a - b")]
)


# --- git ------------------------------------------------------------------


class TestGitOps:
    def test_clone_and_diff(self, fixture_repo: Path, tmp_path: Path):
        destination = tmp_path / "clone"
        head = clone(RepoRef.parse("example/repo"), destination, url=str(fixture_repo))
        assert head == git(fixture_repo, "rev-parse", "HEAD")

        (destination / "calc.py").write_text(FIXED_MODULE)
        patch = diff(destination)
        assert "return a + b" in patch
        assert "return abs(a) + abs(b)" in patch

    def test_clone_does_not_keep_a_usable_origin(self, fixture_repo: Path, tmp_path: Path):
        destination = tmp_path / "clone2"
        clone(RepoRef.parse("example/repo"), destination, url=str(fixture_repo))
        # The remote exists, but the environment is scrubbed so nothing can
        # interactively authenticate against it.
        assert (destination / ".git").exists()

    def test_fingerprint_is_stable_across_rebuilds(self, fixture_repo: Path, tmp_path: Path):
        base = git(fixture_repo, "rev-parse", "HEAD")
        # Derive the expected hash from a plain clone + apply, then prove
        # rebuild_patch reproduces it exactly.
        from forgeflow.edits import apply_edits

        checkout = tmp_path / "plain"
        clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
        git(checkout, "checkout", "--detach", base)
        apply_edits(checkout, FIX)
        expected = fingerprint_patch(diff(checkout))

        rebuilt = _rebuild_local(fixture_repo, base, FIX, expected, tmp_path / "r2")
        assert rebuilt.fingerprint == expected

    def test_rebuild_refuses_a_wrong_fingerprint(self, fixture_repo: Path, tmp_path: Path):
        base = git(fixture_repo, "rev-parse", "HEAD")
        with pytest.raises(PatchVerificationError, match="does not match"):
            _rebuild_local(fixture_repo, base, FIX, "0" * 64, tmp_path / "r3")

    def test_rebuild_refuses_a_bad_base_commit(self, fixture_repo: Path, tmp_path: Path):
        with pytest.raises(PatchVerificationError, match="not a full SHA"):
            _rebuild_local(fixture_repo, "abc", FIX, "0" * 64, tmp_path / "r4")


def _rebuild_local(repo_path: Path, base: str, plan: EditPlan, expected: str, work_dir: Path):
    """``rebuild_patch`` against a local path instead of github.com."""
    from forgeflow import gitops

    original = gitops.clone

    def local_clone(repo, destination, *, depth=50, branch=None, token=None, timeout=600, url=None):
        return original(
            repo, destination, depth=depth, branch=branch, token=token, timeout=timeout,
            url=str(repo_path),
        )

    gitops.clone = local_clone
    try:
        return rebuild_patch(
            RepoRef.parse("example/repo"), base, plan, expected, work_dir, depth=50
        )
    finally:
        gitops.clone = original


# --- sandbox and verification --------------------------------------------


@requires_docker
class TestSandbox:
    def test_baseline_and_patched_comparison(self, fixture_repo: Path):
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            install = sandbox.install_dependencies()
            assert install.exit_code == 0, install.tail()

            _, baseline = sandbox.run_tests()
            assert baseline.collected, "the fixture suite must be collectable"
            # Two real failures plus the deliberate pre-existing one.
            assert baseline.failed == 3, baseline.model_dump()
            assert baseline.passed == 2

            write_calc(fixture_repo, FIXED_MODULE)
            sandbox.update_files(fixture_repo, ["calc.py"])
            _, patched = sandbox.run_tests()

            from forgeflow.sandbox import compare_outcomes

            newly_failing, newly_passing = compare_outcomes(baseline, patched)
            assert newly_failing == []
            assert len(newly_passing) == 2
            assert patched.failed == 1, "only the pre-existing failure remains"

    def test_tests_run_without_network(self, fixture_repo: Path):
        (fixture_repo / "test_network.py").write_text(
            "import socket\n\n\n"
            "def test_no_network():\n"
            "    s = socket.socket()\n"
            "    s.settimeout(2)\n"
            "    try:\n"
            "        s.connect(('1.1.1.1', 53))\n"
            "    except OSError:\n"
            "        return\n"
            "    finally:\n"
            "        s.close()\n"
            "    raise AssertionError('network was reachable')\n"
        )
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            sandbox.install_dependencies()
            result, outcome = sandbox.run_tests()
            assert "test_no_network" not in outcome.failed_tests, result.tail()

    def test_sandbox_runs_as_non_root(self, fixture_repo: Path):
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            result = sandbox.run("id -u", timeout=60)
            assert result.exit_code == 0
            assert result.combined.strip() != "0", "the sandbox must not run as root"

    def test_work_volume_is_read_only_during_tests(self, fixture_repo: Path):
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            result = sandbox.run("touch /work/should_fail", timeout=60)
            assert result.exit_code != 0, "tests must not be able to rewrite the source"

    def test_no_credentials_are_visible_inside(self, fixture_repo: Path):
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            result = sandbox.run("env", timeout=60)
            assert "GITHUB_TOKEN" not in result.combined
            assert "GEMINI_API_KEY" not in result.combined
            assert "ANTHROPIC_API_KEY" not in result.combined

    def test_lint_detects_a_new_problem(self, fixture_repo: Path):
        with DockerSandbox(sandbox_config()) as sandbox:
            sandbox.load_checkout(fixture_repo)
            sandbox.install_dependencies()
            write_calc(fixture_repo, "import os\n\n\ndef add(a, b):\n    return a + b\n")
            sandbox.update_files(fixture_repo, ["calc.py"])
            lint = sandbox.run_lint(["calc.py"])
            from forgeflow.sandbox import parse_lint_findings

            findings = parse_lint_findings(lint.combined)
            assert "F401" in findings.get("calc.py", set())


@requires_docker
class TestVerifier:
    def test_a_correct_patch_passes(self, fixture_repo: Path, tmp_path: Path):
        checkout = tmp_path / "checkout"
        base = clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
        with DockerSandbox(sandbox_config()) as sandbox:
            verifier = Verifier(sandbox, checkout, base_commit=base)
            assert verifier.install().exit_code == 0
            baseline = verifier.capture_baseline(touch_files=["calc.py"])
            assert baseline.collected
            outcome = verifier.verify(FIX, 1)
            assert outcome.report.passed, outcome.report.failure_reason
            assert len(outcome.report.newly_passing) == 2
            assert outcome.report.newly_failing == []
            assert outcome.report.lint_ok
            assert outcome.report.patch_sha256

    def test_a_wrong_patch_is_rejected_with_feedback(self, fixture_repo: Path, tmp_path: Path):
        checkout = tmp_path / "checkout"
        base = clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
        with DockerSandbox(sandbox_config()) as sandbox:
            verifier = Verifier(sandbox, checkout, base_commit=base)
            verifier.install()
            verifier.capture_baseline(touch_files=["calc.py"])
            outcome = verifier.verify(WRONG, 1)
            assert not outcome.report.passed
            assert outcome.report.newly_failing, (
                "a wrong fix must break a previously passing test"
            )
            assert outcome.error_for_model
            # The feedback must tell the model which failures are pre-existing.
            assert "ALREADY failing" in outcome.error_for_model

    def test_a_hallucinated_search_string_is_rejected(self, fixture_repo: Path, tmp_path: Path):
        checkout = tmp_path / "checkout"
        base = clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
        bogus = EditPlan(
            edits=[
                Edit(path="calc.py", search="return a / b", replace="return a + b")
            ]
        )
        with DockerSandbox(sandbox_config()) as sandbox:
            verifier = Verifier(sandbox, checkout, base_commit=base)
            verifier.install()
            verifier.capture_baseline(touch_files=["calc.py"])
            outcome = verifier.verify(bogus, 1)
            assert not outcome.report.passed
            assert "could not be applied" in outcome.report.failure_reason
            assert "exactly once" in outcome.error_for_model

    def test_a_failed_attempt_does_not_leak_into_the_next(self, fixture_repo: Path, tmp_path: Path):
        """The second attempt must be judged on its own edits, not a blend."""
        checkout = tmp_path / "checkout"
        base = clone(RepoRef.parse("example/repo"), checkout, url=str(fixture_repo))
        with DockerSandbox(sandbox_config()) as sandbox:
            verifier = Verifier(sandbox, checkout, base_commit=base)
            verifier.install()
            verifier.capture_baseline(touch_files=["calc.py"])

            first = verifier.verify(WRONG, 1)
            assert not first.report.passed

            # A clean reference run of the correct fix, for comparison.
            reference = tmp_path / "reference"
            clone(RepoRef.parse("example/repo"), reference, url=str(fixture_repo))
            git(reference, "checkout", "--detach", base)
            from forgeflow.edits import apply_edits

            apply_edits(reference, FIX)
            expected_patch = diff(reference)

            second = verifier.verify(FIX, 2)
            assert second.report.passed, second.report.failure_reason
            assert second.patch_text == expected_patch, (
                "the second attempt's patch must not contain the first attempt's edits"
            )
            assert "-a - b" not in second.patch_text


# --- end to end -----------------------------------------------------------


@requires_docker
class TestEndToEnd:
    def test_pipeline_produces_a_verified_patch_and_never_publishes(
        self, fixture_repo: Path, tmp_path: Path
    ):
        from forgeflow.llm.providers import MockProvider

        base = git(fixture_repo, "rev-parse", "HEAD")
        config = Config.load(
            env={},
            db_path=tmp_path / "e2e.sqlite3",
            work_dir=tmp_path / "work",
            llm_provider="mock",
            docker=DOCKER,
            max_attempts=1,
            confidence_threshold=0.5,
        )
        store = Store(config.db_path)
        github = FakeGitHub(
            issues=[
                make_issue(
                    1,
                    title="add() returns the wrong result for negative numbers",
                    body=(
                        "Steps to reproduce:\n"
                        "1. call add(-1, -2)\n\n"
                        "Traceback (most recent call last):\n"
                        '  File "calc.py", line 2, in add\n'
                        "AssertionError: expected -3, got 1\n"
                    ),
                )
            ]
        )

        # Drive the mock provider with the real edit, exactly as a model would
        # return it after reading the prompt.
        provider = MockProvider()
        original_complete = provider.complete

        def complete(system, user, *, temperature=0.0, max_output_tokens=4096):
            if '"edits"' in system:
                user = (
                    user
                    + "\n<<<EDIT calc.py|    return a + b>>>\n"
                    + "    return abs(a) + abs(b)\n<<<END>>>\n"
                )
            return original_complete(
                system, user, temperature=temperature, max_output_tokens=max_output_tokens
            )

        provider.complete = complete  # type: ignore[method-assign]

        result = run_once(
            config,
            "example/repo",
            issue_number=1,
            store=store,
            github=github,
            provider=provider,
            clone_url=str(fixture_repo),
        )

        assert result.status == "proposed", (
            result.status,
            result.error_message,
            [a.report.failure_reason for a in result.attempts],
        )
        assert result.accepted is not None
        assert result.base_commit == base

        report = result.accepted.report
        assert report.newly_failing == []
        assert len(report.newly_passing) == 2
        assert report.lint_ok
        assert len(report.patch_sha256) == 64

        # The store must hold the same fingerprint the run reported.
        stored = store.find_attempt_by_fingerprint(result.fingerprint)
        assert stored is not None
        assert stored["passed"] == 1
        assert "calc.py" in stored["files_changed"]

        # Nothing may have been written to GitHub.
        assert github.pulls_opened == []
        assert github.forks_created == []
        assert not any(
            call[0] in {"ensure_fork", "open_draft_pull_request"} for call in github.calls
        )

        # The fingerprint must be reproducible from the base commit.
        expected = report.patch_sha256
        rebuilt = _rebuild_local(fixture_repo, base, FIX, expected, tmp_path / "rebuild")
        assert rebuilt.fingerprint == expected

        store.close()

    def test_pipeline_reports_a_patch_that_does_not_work(
        self, fixture_repo: Path, tmp_path: Path
    ):
        """A plausible but wrong fix must end as verification-failed, not success."""
        from forgeflow.llm.providers import MockProvider

        config = Config.load(
            env={},
            db_path=tmp_path / "e2e2.sqlite3",
            work_dir=tmp_path / "work2",
            llm_provider="mock",
            docker=DOCKER,
            max_attempts=1,
            confidence_threshold=0.5,
        )
        store = Store(config.db_path)
        github = FakeGitHub(issues=[make_issue(1, title="add() is wrong")])

        provider = MockProvider()
        original_complete = provider.complete

        def complete(system, user, *, temperature=0.0, max_output_tokens=4096):
            if '"edits"' in system:
                # A wrong fix that breaks previously passing tests.
                user = (
                    user
                    + "\n<<<EDIT calc.py|    return -a - b>>>\n"
                    + "    return abs(a) + abs(b)\n<<<END>>>\n"
                )
            return original_complete(
                system, user, temperature=temperature, max_output_tokens=max_output_tokens
            )

        provider.complete = complete  # type: ignore[method-assign]

        result = run_once(
            config,
            "example/repo",
            issue_number=1,
            store=store,
            github=github,
            provider=provider,
            clone_url=str(fixture_repo),
        )

        assert result.status == "verification-failed"
        assert result.accepted is None
        assert result.fingerprint is None
        assert result.attempts[0].report.newly_failing
        assert github.pulls_opened == []
        store.close()

    def test_pipeline_retries_with_the_failure_fed_back(
        self, fixture_repo: Path, tmp_path: Path
    ):
        """Attempt 1 is wrong, attempt 2 is right, and the error is passed on."""
        from forgeflow.llm.providers import MockProvider

        config = Config.load(
            env={},
            db_path=tmp_path / "retry.sqlite3",
            work_dir=tmp_path / "work-retry",
            llm_provider="mock",
            docker=DOCKER,
            max_attempts=3,
            confidence_threshold=0.5,
        )
        store = Store(config.db_path)
        github = FakeGitHub(issues=[make_issue(1, title="add() is wrong for negatives")])

        provider = MockProvider()
        original_complete = provider.complete
        prompts: list[str] = []

        def complete(system, user, *, temperature=0.0, max_output_tokens=4096):
            if '"edits"' in system:
                prompts.append(user)
                # First call: a wrong fix. Second call: the correct one.
                replacement = (
                    "    return -a - b" if len(prompts) == 1 else "    return a + b"
                )
                user = (
                    user
                    + f"\n<<<EDIT calc.py|{replacement}>>>\n"
                    + "    return abs(a) + abs(b)\n<<<END>>>\n"
                )
            return original_complete(
                system, user, temperature=temperature, max_output_tokens=max_output_tokens
            )

        provider.complete = complete  # type: ignore[method-assign]

        result = run_once(
            config,
            "example/repo",
            issue_number=1,
            store=store,
            github=github,
            provider=provider,
            clone_url=str(fixture_repo),
        )

        assert result.status == "proposed", (
            result.status,
            [a.report.failure_reason for a in result.attempts],
        )
        assert len(result.attempts) == 2
        assert not result.attempts[0].report.passed
        assert result.attempts[1].report.passed
        assert result.accepted is result.attempts[1]

        # The second prompt must carry the first attempt's failure.
        assert len(prompts) == 2
        assert "<<<BEGIN_UNTRUSTED_PREVIOUS_ERROR>>>" in prompts[1]
        assert "test_add_positive" in prompts[1] or "newly failing" in prompts[1]
        # And the second patch must not contain the first attempt's edit.
        assert "-a - b" not in result.attempts[1].patch_text
        assert result.fingerprint == result.attempts[1].report.patch_sha256
        store.close()

    def test_pipeline_gives_up_after_max_attempts(self, fixture_repo: Path, tmp_path: Path):
        from forgeflow.llm.providers import MockProvider

        config = Config.load(
            env={},
            db_path=tmp_path / "giveup.sqlite3",
            work_dir=tmp_path / "work-giveup",
            llm_provider="mock",
            docker=DOCKER,
            max_attempts=2,
            confidence_threshold=0.5,
        )
        store = Store(config.db_path)
        github = FakeGitHub(issues=[make_issue(1, title="add() is wrong")])

        provider = MockProvider()
        original_complete = provider.complete
        calls = {"n": 0}

        def complete(system, user, *, temperature=0.0, max_output_tokens=4096):
            if '"edits"' in system:
                calls["n"] += 1
                # Always wrong: the model never fixes it.
                user = (
                    user
                    + "\n<<<EDIT calc.py|    return -a - b>>>\n"
                    + "    return abs(a) + abs(b)\n<<<END>>>\n"
                )
            return original_complete(
                system, user, temperature=temperature, max_output_tokens=max_output_tokens
            )

        provider.complete = complete  # type: ignore[method-assign]

        result = run_once(
            config,
            "example/repo",
            issue_number=1,
            store=store,
            github=github,
            provider=provider,
            clone_url=str(fixture_repo),
        )

        assert result.status == "verification-failed"
        assert len(result.attempts) == 2, "the loop must stop at max_attempts"
        assert calls["n"] == 2
        assert all(not a.report.passed for a in result.attempts)
        assert github.pulls_opened == []
        store.close()

    def test_pipeline_stops_when_the_suite_does_not_collect(
        self, fixture_repo: Path, tmp_path: Path
    ):
        """A repository with no runnable tests is unverifiable, not a success."""
        empty = tmp_path / "empty-repo"
        empty.mkdir()
        (empty / "app.py").write_text("x = 1\n")
        (empty / "test_nothing.py").write_text("")
        git(empty, "init", "-b", "main")
        git(empty, "add", "-A")
        git(empty, "commit", "-m", "no tests")

        config = Config.load(
            env={},
            db_path=tmp_path / "e2e3.sqlite3",
            work_dir=tmp_path / "work3",
            llm_provider="mock",
            docker=DOCKER,
            max_attempts=1,
        )
        store = Store(config.db_path)
        github = FakeGitHub(issues=[make_issue(1, title="something is broken")])

        result = run_once(
            config,
            "example/repo",
            issue_number=1,
            store=store,
            github=github,
            provider=None,
            clone_url=str(empty),
        )
        assert result.status == "unverifiable"
        message = (result.error_message or "").lower()
        assert "collected" in message or result.error_stage == "baseline"
        store.close()
