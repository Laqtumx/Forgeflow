"""Command-line interface.

Commands:

    forgeflow doctor                     check Docker, tokens, and providers
    forgeflow issues OWNER/NAME          rank open issues without calling a model
    forgeflow run OWNER/NAME             full pipeline, stops at "proposed"
    forgeflow approve FINGERPRINT        record a human approval for one patch
    forgeflow publish APPROVAL_ID        rebuild, verify the hash, push, draft PR
    forgeflow status [RUN_ID]            inspect what the store recorded

The run and publish steps are deliberately separate: ``run`` never writes to
GitHub, and ``publish`` refuses to proceed unless the human supplied the exact
fingerprint of a verified patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config
from .errors import ForgeFlowError
from .github_client import GitHubClient
from .models import RepoRef
from .pipeline import Pipeline, rank_only, summarise
from .publish import approve, publish, publish_pending
from .sandbox import docker_available
from .store import Store


def _config(args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if getattr(args, "provider", None):
        overrides["llm_provider"] = args.provider
    if getattr(args, "db", None):
        overrides["db_path"] = Path(args.db)
    if getattr(args, "work_dir", None):
        overrides["work_dir"] = Path(args.work_dir)
    if getattr(args, "max_attempts", None):
        overrides["max_attempts"] = args.max_attempts
    if getattr(args, "confidence", None) is not None:
        overrides["confidence_threshold"] = args.confidence
    if getattr(args, "sandbox_image", None):
        overrides["sandbox_image"] = args.sandbox_image
    if getattr(args, "docker", None):
        overrides["docker"] = args.docker
    if getattr(args, "keep_workdirs", False):
        overrides["keep_workdirs"] = True
    if getattr(args, "fork_owner", None):
        overrides["fork_owner"] = args.fork_owner
    return Config.load(**overrides)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeflow",
        description=(
            "Turn a public GitHub repository's open issues into reviewed, tested "
            "draft pull requests."
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["gemini", "anthropic", "mock"], default=None)
        p.add_argument(
        "--db",
        default=None,
        help="SQLite path (default .forgeflow/forgeflow.sqlite3)",
    )
        p.add_argument("--work-dir", default=None, help="scratch directory for clones")
        p.add_argument("--sandbox-image", default=None, help="container image for the sandbox")
        p.add_argument("--docker", default=None, help="docker executable, e.g. 'sudo docker'")
        p.add_argument("--keep-workdirs", action="store_true", help="do not delete clones")

    p_doctor = sub.add_parser("doctor", help="check Docker, tokens, and provider keys")
    p_doctor.add_argument("--provider", choices=["gemini", "anthropic", "mock"], default=None)
    p_doctor.add_argument("--docker", default=None)
    p_doctor.add_argument("--json", action="store_true", help="machine-readable output")

    p_issues = sub.add_parser("issues", help="rank open issues without calling a model")
    p_issues.add_argument("repo", help="owner/name or a GitHub URL")
    p_issues.add_argument("--limit", type=int, default=15, help="how many to print")
    p_issues.add_argument(
        "--show-dropped", action="store_true", help="also list filtered-out issues"
    )
    p_issues.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="run the pipeline; stops before publishing")
    p_run.add_argument("repo", help="owner/name or a GitHub URL")
    p_run.add_argument("--issue", type=int, default=None, help="force a specific issue number")
    p_run.add_argument("--max-attempts", type=int, default=None)
    p_run.add_argument("--confidence", type=float, default=None, help="gate threshold 0..1")
    p_run.add_argument("--json", action="store_true", help="print the full result as JSON")
    add_common(p_run)

    p_rank = sub.add_parser("analyse", help="rank and analyse, no clone and no Docker")
    p_rank.add_argument("repo")
    p_rank.add_argument("--issue", type=int, default=None)
    p_rank.add_argument("--json", action="store_true")
    add_common(p_rank)

    p_approve = sub.add_parser("approve", help="record approval of one patch fingerprint")
    p_approve.add_argument("fingerprint", help="the SHA-256 printed by `forgeflow run`")
    p_approve.add_argument("--by", default=None, help="your name or handle (recorded)")
    p_approve.add_argument("--db", default=None)

    p_publish = sub.add_parser("publish", help="rebuild, verify the hash, push, open a draft PR")
    p_publish.add_argument("approval_id", type=int, help="the id printed by `forgeflow approve`")
    p_publish.add_argument("--fork-owner", default=None, help="account that owns the fork")
    p_publish.add_argument(
        "--dry-run", action="store_true", help="rebuild and hash only, no writes"
    )
    p_publish.add_argument("--db", default=None)
    p_publish.add_argument("--work-dir", default=None)
    p_publish.add_argument("--docker", default=None)
    p_publish.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="show what the store recorded")
    p_status.add_argument("run_id", nargs="?", type=int, default=None)
    p_status.add_argument("--db", default=None)
    p_status.add_argument("--json", action="store_true")

    return parser


# --- commands -------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    report: dict[str, object] = {}

    ok, detail = docker_available(config.docker)
    report["docker"] = {"ok": ok, "detail": detail}

    report["github_read_token"] = bool(config.github_read_token)
    report["github_publish_token"] = bool(config.github_publish_token)
    report["fork_owner"] = config.fork_owner

    report["provider"] = config.llm_provider
    report["model"] = config.model_name()
    key = config.provider_api_key()
    report["provider_key"] = bool(key) if config.llm_provider != "mock" else "not needed"

    github_ok = False
    github_detail = "not checked (no read token)"
    if config.github_read_token:
        try:
            client = GitHubClient(config.github_read_token, config.github_publish_token)
            user = client.current_user()
            rate = client.rate_limit()
            core = (rate.get("resources") or {}).get("core") or {}
            github_ok = True
            github_detail = (
                f"signed in as {user.get('login')}; "
                f"{core.get('remaining')}/{core.get('limit')} core requests left"
            )
        except ForgeFlowError as exc:
            github_detail = str(exc)
    report["github_api"] = {"ok": github_ok, "detail": github_detail}

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("ForgeFlow doctor\n")
        print(f"  docker          : {'ok' if ok else 'FAIL'} — {detail}")
        read_state = "set" if report["github_read_token"] else "MISSING (GITHUB_TOKEN)"
        print(f"  github read     : {read_state}")
        print(
            f"  github publish  : "
            f"{'set' if report['github_publish_token'] else 'not set (needed only to publish)'}"
        )
        print(f"  fork owner      : {report['fork_owner'] or 'not set (needed only to publish)'}")
        print(f"  provider        : {config.llm_provider} ({config.model_name()})")
        print(f"  provider key    : {report['provider_key']}")
        print(f"  github api      : {'ok' if github_ok else 'FAIL'} — {github_detail}")
        print()
        if not ok:
            print("Docker is required for `forgeflow run`. `forgeflow issues` and")
            print("`forgeflow analyse` work without it.")
    return 0 if ok else 1


def cmd_issues(args: argparse.Namespace) -> int:
    config = _config(args)
    repo = RepoRef.parse(args.repo)
    client = GitHubClient(config.github_read_token, config.github_publish_token)

    issues = client.list_open_issues(repo, limit=config.max_candidates * 4)
    linked: dict[int, list[object]] = {}
    for issue in issues[: config.max_candidates]:
        found = client.linked_pull_requests(repo, issue.number)
        if found:
            linked[issue.number] = list(found)

    from .ranking import RankingOptions, filter_and_rank

    candidates = filter_and_rank(
        issues,
        linked_prs=linked,
        options=RankingOptions(excluded_labels=config.excluded_labels),
        limit=config.max_candidates,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "repo": repo.slug,
                    "ranked": [
                        {
                            "number": item.issue.number,
                            "title": item.issue.title,
                            "score": item.score,
                            "reasons": item.reasons,
                            "labels": list(item.issue.labels),
                            "assignees": list(item.issue.assignees),
                        }
                        for item in candidates.ranked
                    ],
                    "dropped": [
                        {"number": issue.number, "title": issue.title, "reason": reason}
                        for issue, reason in candidates.dropped
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"{repo.slug}: {len(issues)} open issues, {len(candidates.ranked)} ranked\n")
    for item in candidates.ranked[: args.limit]:
        print(f"  {item.score:>6.2f}  #{item.issue.number:<6} {item.issue.title[:80]}")
        for reason in item.reasons[:4]:
            print(f"          - {reason}")
    if args.show_dropped and candidates.dropped:
        print(f"\n  dropped ({len(candidates.dropped)}):")
        for issue, reason in candidates.dropped[: args.limit]:
            print(f"          #{issue.number:<6} {reason}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    pipeline = Pipeline(config, echo=print)
    problems = pipeline.preflight()
    if problems:
        for problem in problems:
            print(f"preflight: {problem}", file=sys.stderr)
        if any("Docker" in problem for problem in problems):
            print("Run `forgeflow doctor` for details.", file=sys.stderr)
            pipeline.close()
            return 2

    try:
        result = pipeline.run(args.repo, args.issue)
    finally:
        pipeline.close()

    if args.json:
        print(json.dumps(_result_payload(result), indent=2, default=str))
    else:
        print()
        print(summarise(result))

    if result.status == "proposed" and result.fingerprint:
        print()
        print("This patch is verified but NOT published. To publish it:")
        print(f"  forgeflow approve {result.fingerprint} --by 'your name'")
        print("  forgeflow publish <approval-id> --fork-owner <your-account>")
        return 0
    return 1 if result.status in {"failed", "verification-failed", "unverifiable"} else 0


def cmd_analyse(args: argparse.Namespace) -> int:
    config = _config(args)
    config.require_llm_credentials()
    result = rank_only(config, args.repo, args.issue)
    if args.json:
        print(json.dumps(_result_payload(result), indent=2, default=str))
    else:
        print(summarise(result))
    return 0 if result.status in {"analysed", "no-candidate"} else 1


def cmd_approve(args: argparse.Namespace) -> int:
    config = _config(args)
    store = Store(config.db_path)
    try:
        approved_by = args.by or os.environ.get("USER") or "unknown"
        approval = approve(store, args.fingerprint, approved_by=approved_by)
    finally:
        store.close()
    print(f"Approved patch {approval.fingerprint}")
    print(f"  approval id : {approval.approval_id}")
    print(f"  attempt     : {approval.attempt_id}")
    print(f"  approved by : {approval.approved_by}")
    print()
    print("Publishing re-derives the patch from the base commit and aborts if the")
    print("hash differs. To publish:")
    print(f"  forgeflow publish {approval.approval_id} --fork-owner <your-account>")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    config = _config(args)
    store = Store(config.db_path)
    try:
        result = publish(config, store, args.approval_id, dry_run=args.dry_run)
    finally:
        store.close()

    if args.json:
        print(
            json.dumps(
                {
                    "branch": result.branch,
                    "pr_url": result.pr_url,
                    "pr_number": result.pr_number,
                    "fingerprint": result.fingerprint,
                    "fork_owner": result.fork_owner,
                },
                indent=2,
            )
        )
    elif args.dry_run:
        print("Dry run: patch rebuilt and the fingerprint matches.")
        print(f"  fingerprint : {result.fingerprint}")
        print(f"  branch would be: {result.branch}")
        print("Nothing was pushed.")
    else:
        print(f"Pushed branch {result.fork_owner}:{result.branch}")
        if result.pr_url:
            print(f"Draft pull request: {result.pr_url}")
        else:
            print("The draft PR was not opened; see the message above.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _config(args)
    store = Store(config.db_path)
    try:
        if args.run_id is None:
            runs = store.all_runs()
            pending = publish_pending(config, store)
            if args.json:
                print(json.dumps({"runs": runs, "pending": pending}, indent=2, default=str))
            else:
                if not runs:
                    print("No runs recorded yet.")
                for run in runs:
                    print(
                        f"  run {run['id']:<4} {run['repo']:<40} "
                        f"{run['status']:<20} issue={run['issue_number']}"
                    )
                if pending:
                    print("\nApprovals not yet published:")
                    for item in pending:
                        print(
                            f"  approval {item['id']}: {item['publish_state']} "
                            f"({item['fingerprint'][:12]}…)"
                        )
            return 0

        summary = store.run_summary(args.run_id)
        if not summary:
            print(f"run {args.run_id} not found", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
            return 0

        run = summary["run"]
        print(f"run {run['id']} — {run['repo']} — {run['status']}")
        print(f"  provider     : {run['provider']} ({run['model']})")
        print(f"  issue        : {run['issue_number']}")
        print(f"  base commit  : {run['base_commit']}")
        if run["error_stage"]:
            print(f"  failed at    : {run['error_stage']}")
            print(f"  error        : {str(run['error_message'])[:400]}")
        for attempt in summary["attempts"]:
            verdict = "PASS" if attempt["passed"] else "fail"
            print(
                f"  attempt {attempt['attempt']}: {verdict} "
                f"files={attempt['files_changed']}"
            )
            if attempt["failure_reason"]:
                print(f"    {attempt['failure_reason'][:300]}")
            if attempt["patch_sha256"]:
                print(f"    fingerprint {attempt['patch_sha256']}")
        for approval in summary["approvals"]:
            print(
                f"  approval {approval['id']}: {approval['publish_state']} "
                f"branch={approval['branch']} pr={approval['pr_url']}"
            )
        return 0
    finally:
        store.close()


def _result_payload(result) -> dict:
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "repo": result.repo,
        "status": result.status,
        "base_commit": result.base_commit,
        "error_stage": result.error_stage,
        "error_message": result.error_message,
        "usage": result.usage,
    }
    if result.issue:
        payload["issue"] = {
            "number": result.issue.issue.number,
            "title": result.issue.issue.title,
            "score": result.issue.score,
            "reasons": result.issue.reasons,
            "url": result.issue.issue.html_url,
        }
    if result.analysis:
        payload["analysis"] = result.analysis.model_dump()
    if result.analysis_skip_reason:
        payload["analysis_skip_reason"] = result.analysis_skip_reason
    if result.baseline:
        payload["baseline"] = result.baseline.outcome.model_dump()
        payload["baseline_notes"] = result.baseline.notes
    payload["attempts"] = [
        {
            "attempt": item.report.attempt,
            "passed": item.report.passed,
            "failure_reason": item.report.failure_reason,
            "fingerprint": item.report.patch_sha256,
            "files_changed": item.files_changed,
            "newly_failing": item.report.newly_failing,
            "newly_passing": item.report.newly_passing,
            "lint_ok": item.report.lint_ok,
        }
        for item in result.attempts
    ]
    payload["fingerprint"] = result.fingerprint
    return payload


COMMANDS = {
    "doctor": cmd_doctor,
    "issues": cmd_issues,
    "run": cmd_run,
    "analyse": cmd_analyse,
    "approve": cmd_approve,
    "publish": cmd_publish,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"forgeflow {__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 0

    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except ForgeFlowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
