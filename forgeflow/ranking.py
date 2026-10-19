"""Issue filtering and cheap heuristic ranking.

No model call happens here. The goal is to spend the free-tier quota on the one
issue most likely to yield a small, verifiable patch, so ranking is done with
regexes and a handful of transparent weights. Every score carries the reasons
that produced it, which is what makes the ranking auditable in the results.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .models import Issue, RankedIssue

# --- pattern banks --------------------------------------------------------

TRACEBACK_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\s*File \"[^\"]+\", line \d+", re.MULTILINE),
    re.compile(r"^\s*\w*(?:Error|Exception)\b.*:", re.MULTILINE),
    re.compile(r"^\s*at [\w$.<>\[\]]+\s*\([\w./\\-]+:\d+\)", re.MULTILINE),  # JS-style stack
    re.compile(r"AssertionError|ImportError|ModuleNotFoundError|TypeError|ValueError"),
)

REPRO_PATTERNS = (
    re.compile(r"steps? to reproduce", re.IGNORECASE),
    re.compile(r"how to reproduce", re.IGNORECASE),
    re.compile(r"\breproduc(?:e|tion|ible)\b", re.IGNORECASE),
    re.compile(r"minimal (?:repro|example|case)", re.IGNORECASE),
    re.compile(r"^\s*(?:expected|actual) (?:behaviou?r|result)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"```"),  # any fenced block is a strong hint of a snippet
)

FIX_INTENT_PATTERNS = (
    re.compile(
        r"\b(?:bug|crash|error|fail(?:s|ed|ing)?|broken|incorrect|wrong|regression)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:fix|repair|correct|should|expected)\b", re.IGNORECASE),
)

# Issues that read like feature requests rarely yield a small verified patch.
FEATURE_PATTERNS = (
    re.compile(
        r"\b(?:feature request|enhancement|proposal|rfc|add support for|would be nice)\b",
        re.IGNORECASE,
    ),
)

VAGUE_PATTERNS = (
    re.compile(
        r"\b(?:doesn'?t work|not working|broken somehow|help!?|please fix)\b",
        re.IGNORECASE,
    ),
)


@dataclass
class RankingWeights:
    """Tunable weights. Kept explicit so results can be reasoned about."""

    unassigned: float = 2.0
    assigned_penalty: float = 3.0
    no_linked_pr: float = 1.5
    linked_pr_penalty: float = 4.0
    has_reproduction: float = 1.5
    has_traceback: float = 1.0
    fix_intent: float = 0.5
    feature_request_penalty: float = 1.5
    vague_penalty: float = 1.0
    detailed_body: float = 0.75
    short_body_penalty: float = 1.0
    long_thread_penalty: float = 0.5
    recency_bonus: float = 0.5
    base: float = 0.0

    # thresholds
    min_body_chars: int = 80
    detailed_body_chars: int = 400
    long_thread_comments: int = 20


@dataclass
class RankingOptions:
    excluded_labels: Sequence[str] = ()
    weights: RankingWeights = field(default_factory=RankingWeights)
    # A linked PR normally means someone is already on it, so drop it entirely.
    drop_when_linked_pr: bool = True


@dataclass
class CandidateSet:
    """Result of filtering and ranking, with the drop reasons preserved."""

    ranked: list[RankedIssue]
    dropped: list[tuple[Issue, str]] = field(default_factory=list)


def detect_traceback(text: str) -> bool:
    return any(pattern.search(text) for pattern in TRACEBACK_PATTERNS)


def detect_reproduction(text: str) -> bool:
    return any(pattern.search(text) for pattern in REPRO_PATTERNS)


def _has_label(issue: Issue, labels: Sequence[str]) -> str | None:
    wanted = {label.lower() for label in labels}
    for label in issue.labels:
        if label.lower() in wanted:
            return label
    return None


def score_issue(
    issue: Issue,
    *,
    has_linked_pr: bool = False,
    weights: RankingWeights | None = None,
    now: str | None = None,
) -> RankedIssue:
    """Score a single issue, returning the score and the reasons for it."""
    w = weights or RankingWeights()
    score = w.base
    reasons: list[str] = []

    # --- assignment -------------------------------------------------------
    if issue.assignees:
        score -= w.assigned_penalty
        reasons.append(f"assigned to {', '.join(issue.assignees)} (-{w.assigned_penalty})")
    else:
        score += w.unassigned
        reasons.append(f"unassigned (+{w.unassigned})")

    # --- linked PR --------------------------------------------------------
    if has_linked_pr:
        score -= w.linked_pr_penalty
        reasons.append(f"already has a linked PR (-{w.linked_pr_penalty})")
    else:
        score += w.no_linked_pr
        reasons.append(f"no linked PR (+{w.no_linked_pr})")

    # --- evidence quality -------------------------------------------------
    text = issue.text
    if detect_reproduction(text):
        score += w.has_reproduction
        reasons.append(f"includes reproduction (+{w.has_reproduction})")
    if detect_traceback(text):
        score += w.has_traceback
        reasons.append(f"includes traceback/error (+{w.has_traceback})")

    if any(pattern.search(text) for pattern in FIX_INTENT_PATTERNS):
        score += w.fix_intent
        reasons.append(f"reads like a fixable defect (+{w.fix_intent})")

    if any(pattern.search(text) for pattern in FEATURE_PATTERNS):
        score -= w.feature_request_penalty
        reasons.append(f"looks like a feature request (-{w.feature_request_penalty})")

    if any(pattern.search(text) for pattern in VAGUE_PATTERNS) and not detect_traceback(text):
        score -= w.vague_penalty
        reasons.append(f"vague report with no evidence (-{w.vague_penalty})")

    # --- shape of the report ---------------------------------------------
    body_len = len(issue.body.strip())
    if body_len >= w.detailed_body_chars:
        score += w.detailed_body
        reasons.append(f"detailed body ({body_len} chars, +{w.detailed_body})")
    elif body_len < w.min_body_chars:
        score -= w.short_body_penalty
        reasons.append(f"body too short ({body_len} chars, -{w.short_body_penalty})")

    if issue.comments >= w.long_thread_comments:
        score -= w.long_thread_penalty
        reasons.append(f"long discussion ({issue.comments} comments, -{w.long_thread_penalty})")

    # --- recency ----------------------------------------------------------
    # same or later month
    if now and issue.updated_at and issue.updated_at[:10] >= now[:10][:7]:
        score += w.recency_bonus
        reasons.append(f"recently updated (+{w.recency_bonus})")

    return RankedIssue(issue=issue, score=round(score, 4), reasons=reasons)


def filter_and_rank(
    issues: Iterable[Issue],
    *,
    linked_prs: Mapping[int, Sequence[object]] | None = None,
    options: RankingOptions | None = None,
    now: str | None = None,
    limit: int | None = None,
) -> CandidateSet:
    """Drop unusable issues, then rank the rest.

    ``linked_prs`` maps issue number -> sequence of linked PR records. Callers
    that cannot afford the timeline call can pass ``None``, in which case every
    issue is treated as having no linked PR.
    """
    opts = options or RankingOptions()
    linked_prs = linked_prs or {}
    ranked: list[RankedIssue] = []
    dropped: list[tuple[Issue, str]] = []

    for issue in issues:
        excluded = _has_label(issue, opts.excluded_labels)
        if excluded:
            dropped.append((issue, f"excluded label: {excluded}"))
            continue
        linked = list(linked_prs.get(issue.number, ()))
        if linked and opts.drop_when_linked_pr:
            linked_number = (
                linked[0].get("number") if isinstance(linked[0], dict) else linked[0]
            )
            dropped.append((issue, f"linked PR #{linked_number}"))
            continue
        ranked.append(
            score_issue(
                issue,
                has_linked_pr=bool(linked),
                weights=opts.weights,
                now=now,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.issue.number))
    if limit is not None:
        ranked = ranked[:limit]
    return CandidateSet(ranked=ranked, dropped=dropped)


def select_best(candidates: CandidateSet) -> RankedIssue | None:
    return candidates.ranked[0] if candidates.ranked else None
