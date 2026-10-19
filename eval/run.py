"""Evaluation harness: measured results on real repositories.

The point of this package is not to demo one happy path. It is to produce
numbers, including the disappointing ones, on repositories nobody wrote for
this tool. Two evaluations:

* ``rank`` — run the heuristics over a set of real repositories and report
  what was ranked, what was dropped, and how long it took. No model, no
  Docker, no cost. This is the evaluation to run first.
* ``analyse`` — additionally make call 1 for the best candidate in each
  repository and report whether the analysis cleared the confidence gate.
  Costs a handful of model calls per repository.

Every run appends a JSON line to a results file, so a series of runs is a
dataset rather than a collection of terminal scrollbacks. Failures are
recorded, not skipped: a repository that could not be read is a result.

Usage:

    python -m eval.run rank  --repos eval/repos.txt --out eval/results.jsonl
    python -m eval.run analyse --repos eval/repos.txt --provider gemini --limit 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeflow.config import Config
from forgeflow.errors import ForgeFlowError
from forgeflow.github_client import GitHubClient
from forgeflow.models import RepoRef
from forgeflow.pipeline import rank_only
from forgeflow.ranking import (
    RankingOptions,
    filter_and_rank,
    select_best,
)
from forgeflow.store import Store

DEFAULT_REPOS = Path(__file__).parent / "repos.txt"
DEFAULT_OUT = Path(__file__).parent / "results.jsonl"


@dataclass
class RepoResult:
    """One repository's outcome. ``error`` is a result, not an omission."""

    repo: str
    ok: bool
    open_issues: int = 0
    ranked: int = 0
    dropped: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    best_issue: int | None = None
    best_title: str = ""
    best_score: float | None = None
    best_reasons: list[str] = field(default_factory=list)
    analysis_confidence: float | None = None
    analysis_fixable: bool | None = None
    gate_passed: bool | None = None
    gate_reason: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class EvalSummary:
    evaluation: str
    provider: str
    started_at: str
    elapsed_s: float
    repositories: list[RepoResult]

    def report(self) -> str:
        total = len(self.repositories)
        ok = [r for r in self.repositories if r.ok]
        failed = [r for r in self.repositories if not r.ok]
        with_candidate = [r for r in ok if r.best_issue is not None]

        lines = [
            f"ForgeFlow evaluation: {self.evaluation}",
            f"  provider      : {self.provider}",
            f"  started       : {self.started_at}",
            f"  repositories  : {total} ({len(ok)} read, {len(failed)} failed)",
            f"  elapsed       : {self.elapsed_s:.1f}s",
        ]

        if with_candidate:
            scores = [r.best_score for r in with_candidate if r.best_score is not None]
            if scores:
                lines.append(
                    f"  best score    : median {statistics.median(scores):.2f}, "
                    f"min {min(scores):.2f}, max {max(scores):.2f}"
                )
        lines.append(
            f"  had candidate : {len(with_candidate)}/{len(ok)}"
            + (f" ({100 * len(with_candidate) / len(ok):.0f}%)" if ok else "")
        )

        if self.evaluation == "analyse":
            gated = [r for r in ok if r.gate_passed]
            analysed = [r for r in ok if r.gate_passed is not None]
            if analysed:
                lines.append(
                    f"  cleared gate  : {len(gated)}/{len(analysed)}"
                    + f" ({100 * len(gated) / len(analysed):.0f}%)"
                )

        lines.append("")
        lines.append("  per repository:")
        for result in self.repositories:
            if not result.ok:
                lines.append(f"    {result.repo:<34} FAILED  {result.error}")
                continue
            score = f"{result.best_score:.2f}" if result.best_score is not None else "-"
            issue = f"#{result.best_issue}" if result.best_issue is not None else "no candidate"
            extra = ""
            if result.gate_passed is not None:
                verdict = "gate pass" if result.gate_passed else "gate skip"
                extra = f"  [{verdict}: {result.gate_reason or 'ok'}]"
            lines.append(
                f"    {result.repo:<34} {result.open_issues:>4} open, "
                f"{result.ranked:>3} ranked, best {issue} score {score}{extra}"
            )
            if result.best_title:
                lines.append(f"        {result.best_title[:100]}")
            if result.best_reasons:
                lines.append(f"        {'; '.join(result.best_reasons[:4])}")
            if result.dropped_reasons:
                top = sorted(result.dropped_reasons.items(), key=lambda kv: -kv[1])[:3]
                lines.append(
                    "        dropped: "
                    + ", ".join(f"{count}x {reason}" for reason, count in top)
                )

        if failed:
            lines.append("")
            lines.append("  failures (these are results, not omissions):")
            for result in failed:
                lines.append(f"    {result.repo:<34} {result.error}")

        return "\n".join(lines)


def read_repos(path: Path) -> list[str]:
    repos = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            repos.append(line)
    return repos


def evaluate_rank(
    repos: list[str],
    config: Config,
    client: GitHubClient,
    *,
    limit: int | None = None,
) -> list[RepoResult]:
    results: list[RepoResult] = []
    for spec in repos[: limit or len(repos)]:
        started = time.monotonic()
        result = RepoResult(repo=spec, ok=False)
        try:
            repo = RepoRef.parse(spec)
            issues = client.list_open_issues(repo, limit=config.max_candidates * 4)
            linked: dict[int, list[object]] = {}
            for issue in issues[: config.max_candidates]:
                found = client.linked_pull_requests(repo, issue.number)
                if found:
                    linked[issue.number] = list(found)

            candidates = filter_and_rank(
                issues,
                linked_prs=linked,
                options=RankingOptions(excluded_labels=config.excluded_labels),
                limit=config.max_candidates,
            )
            result.ok = True
            result.open_issues = len(issues)
            result.ranked = len(candidates.ranked)
            result.dropped = len(candidates.dropped)
            for _, reason in candidates.dropped:
                key = reason.split(":")[0]
                result.dropped_reasons[key] = result.dropped_reasons.get(key, 0) + 1

            best = select_best(candidates)
            if best is not None:
                result.best_issue = best.issue.number
                result.best_title = best.issue.title
                result.best_score = best.score
                result.best_reasons = list(best.reasons)
        except ForgeFlowError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # a crash is a recorded result, not an omission
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = round(time.monotonic() - started, 2)
        results.append(result)
        print(
            f"  {spec:<34} {'ok' if result.ok else 'FAILED':<7} "
            f"{result.ranked:>3} ranked, {result.elapsed_s:.1f}s",
            flush=True,
        )
    return results


def evaluate_analyse(
    repos: list[str],
    config: Config,
    *,
    limit: int | None = None,
) -> list[RepoResult]:
    results: list[RepoResult] = []
    for spec in repos[: limit or len(repos)]:
        started = time.monotonic()
        result = RepoResult(repo=spec, ok=False)
        store = Store(config.db_path)
        try:
            run = rank_only(config, spec, store=store)
            if run.status == "failed":
                result.error = run.error_message or "rank_only failed"
            else:
                result.ok = True
                result.open_issues = len(run.candidates.ranked) + len(
                    run.candidates.dropped
                ) if run.candidates else 0
                result.ranked = len(run.candidates.ranked) if run.candidates else 0
                result.dropped = len(run.candidates.dropped) if run.candidates else 0
                if run.issue is not None:
                    result.best_issue = run.issue.issue.number
                    result.best_title = run.issue.issue.title
                    result.best_score = run.issue.score
                    result.best_reasons = list(run.issue.reasons)
                if run.analysis is not None:
                    result.analysis_confidence = run.analysis.confidence
                    result.analysis_fixable = run.analysis.fixable
                    result.gate_passed = run.analysis_skip_reason is None
                    result.gate_reason = run.analysis_skip_reason
        except ForgeFlowError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            store.close()
        result.elapsed_s = round(time.monotonic() - started, 2)
        results.append(result)
        verdict = ""
        if result.gate_passed is not None:
            verdict = " gate=pass" if result.gate_passed else " gate=skip"
        print(
            f"  {spec:<34} {'ok' if result.ok else 'FAILED':<7} "
            f"conf={result.analysis_confidence}{verdict} {result.elapsed_s:.1f}s",
            flush=True,
        )
    return results


def append_results(path: Path, summary: EvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation": summary.evaluation,
        "provider": summary.provider,
        "started_at": summary.started_at,
        "elapsed_s": summary.elapsed_s,
        "repositories": [asdict(r) for r in summary.repositories],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a ForgeFlow evaluation")
    parser.add_argument("evaluation", choices=["rank", "analyse"])
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="only the first N repositories")
    parser.add_argument("--provider", default=None, choices=["gemini", "anthropic", "mock"])
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if not args.repos.is_file():
        print(f"repository list not found: {args.repos}", file=sys.stderr)
        return 2
    repos = read_repos(args.repos)
    if not repos:
        print(f"no repositories listed in {args.repos}", file=sys.stderr)
        return 2

    overrides: dict[str, object] = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    config = Config.load(**overrides)

    if args.evaluation == "analyse":
        config.require_llm_credentials()

    client = GitHubClient(config.github_read_token, config.github_publish_token)
    if not config.github_read_token:
        print(
            "warning: GITHUB_TOKEN is not set; unauthenticated reads are limited "
            "to 60 requests per hour and this evaluation needs more.",
            file=sys.stderr,
        )

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()
    print(f"evaluating {len(repos)} repositories ({args.evaluation})...", flush=True)

    if args.evaluation == "rank":
        results = evaluate_rank(repos, config, client, limit=args.limit)
    else:
        results = evaluate_analyse(repos, config, limit=args.limit)

    summary = EvalSummary(
        evaluation=args.evaluation,
        provider=config.llm_provider,
        started_at=started_at,
        elapsed_s=round(time.monotonic() - started, 2),
        repositories=results,
    )

    print()
    if args.json:
        print(json.dumps(asdict(summary), indent=2))
    else:
        print(summary.report())

    append_results(args.out, summary)
    print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
