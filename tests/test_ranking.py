"""Heuristic ranking: filtering and scoring, with the reasons attached."""

from __future__ import annotations

import pytest

from forgeflow.models import Issue
from forgeflow.ranking import (
    RankingOptions,
    RankingWeights,
    detect_reproduction,
    detect_traceback,
    filter_and_rank,
    score_issue,
    select_best,
)
from tests.conftest import make_issue


class TestFiltering:
    def test_excluded_label_drops_the_issue(self):
        issue = make_issue(1, labels=("duplicate",))
        result = filter_and_rank([issue], options=RankingOptions(excluded_labels=("duplicate",)))
        assert result.ranked == []
        assert result.dropped[0][1] == "excluded label: duplicate"

    def test_label_matching_is_case_insensitive(self):
        issue = make_issue(1, labels=("WontFix",))
        result = filter_and_rank([issue], options=RankingOptions(excluded_labels=("wontfix",)))
        assert result.ranked == []

    def test_linked_pull_request_drops_the_issue(self):
        issue = make_issue(7)
        result = filter_and_rank(
            [issue],
            linked_prs={7: [{"number": 88}]},
            options=RankingOptions(),
        )
        assert result.ranked == []
        assert "88" in result.dropped[0][1]

    def test_linked_pr_can_be_kept_but_penalised(self):
        issue = make_issue(7)
        with_pr = filter_and_rank(
            [issue],
            linked_prs={7: [{"number": 88}]},
            options=RankingOptions(drop_when_linked_pr=False),
        )
        without_pr = filter_and_rank([issue], options=RankingOptions())
        assert len(with_pr.ranked) == 1
        assert with_pr.ranked[0].score < without_pr.ranked[0].score
        assert any("linked PR" in reason for reason in with_pr.ranked[0].reasons)

    def test_no_linked_pr_map_means_unlinked(self):
        result = filter_and_rank([make_issue(1)], linked_prs=None)
        assert len(result.ranked) == 1
        assert any("no linked PR" in reason for reason in result.ranked[0].reasons)


class TestScoring:
    def test_unassigned_beats_assigned(self):
        free = score_issue(make_issue(1))
        taken = score_issue(make_issue(2, assignees=("someone",)))
        assert free.score > taken.score

    def test_reproduction_and_traceback_add_score(self):
        rich = score_issue(
            make_issue(
                1,
                body=(
                    "Steps to reproduce:\n1. do the thing\n\n"
                    "Traceback (most recent call last):\n"
                    '  File "a.py", line 3, in f\nValueError: nope\n'
                ),
            )
        )
        assert any("reproduction" in reason for reason in rich.reasons)
        assert any("traceback" in reason for reason in rich.reasons)

    def test_feature_request_is_penalised(self):
        feature = score_issue(
            make_issue(
                1,
                title="Feature request: add matrices",
                body=(
                    "This is a feature request for a new capability that "
                    "would be nice to have one day."
                ),
            )
        )
        assert any("feature request" in reason for reason in feature.reasons)

    def test_short_body_is_penalised(self):
        thin = score_issue(make_issue(1, body="broken"))
        assert any("too short" in reason for reason in thin.reasons)

    def test_long_thread_is_penalised(self):
        chatty = score_issue(make_issue(1, comments=40))
        assert any("long discussion" in reason for reason in chatty.reasons)

    def test_every_score_carries_reasons(self):
        ranked = score_issue(make_issue(1))
        assert ranked.reasons, "a score with no explanation is not auditable"

    def test_weights_are_honoured(self):
        heavy = RankingWeights(unassigned=100.0)
        ranked = score_issue(make_issue(1), weights=heavy)
        assert ranked.score >= 100.0

    def test_ordering_is_deterministic(self):
        issues = [make_issue(5), make_issue(2), make_issue(9)]
        first = [item.issue.number for item in filter_and_rank(issues).ranked]
        second = [item.issue.number for item in filter_and_rank(issues).ranked]
        assert first == second

    def test_equal_scores_break_by_issue_number(self):
        a = make_issue(9, title="same", body="same body text here")
        b = make_issue(3, title="same", body="same body text here")
        ranked = filter_and_rank([a, b]).ranked
        assert [item.issue.number for item in ranked] == [3, 9]


class TestDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Traceback (most recent call last):", True),
            ('  File "x.py", line 12, in f', True),
            ("ValueError: bad input", True),
            ("at Object.<anonymous> (app.js:10)", True),
            ("just some prose about a bug", False),
        ],
    )
    def test_traceback_detection(self, text, expected):
        assert detect_traceback(text) is expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Steps to reproduce:\n1. run it", True),
            ("How to reproduce the crash", True),
            ("here is a minimal repro", True),
            ("```python\nprint(1)\n```", True),
            ("it does not work", False),
        ],
    )
    def test_reproduction_detection(self, text, expected):
        assert detect_reproduction(text) is expected


class TestSelection:
    def test_select_best_returns_top_of_the_ranking(self):
        issues = [
            make_issue(1, assignees=("a",), labels=("bug",)),
            make_issue(2, labels=("bug",)),
        ]
        best = select_best(filter_and_rank(issues))
        assert best is not None
        assert best.issue.number == 2

    def test_empty_input_yields_no_candidate(self):
        assert select_best(filter_and_rank([])) is None

    def test_limit_truncates_after_sorting(self):
        issues = [make_issue(n) for n in range(1, 11)]
        result = filter_and_rank(issues, limit=3)
        assert len(result.ranked) == 3
        assert result.ranked[0].score >= result.ranked[-1].score

    def test_issue_from_api_payload(self):
        payload = {
            "number": 12,
            "title": "t",
            "body": "b",
            "html_url": "u",
            "labels": [{"name": "bug"}, {"name": "p1"}],
            "assignees": [{"login": "alice"}],
            "comments": 3,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "user": {"login": "bob"},
        }
        issue = Issue.from_api(payload)
        assert issue.number == 12
        assert issue.labels == ("bug", "p1")
        assert issue.assignees == ("alice",)
        assert issue.author == "bob"
