"""Sandbox result parsing and the baseline comparison.

The comparison logic is the part that decides whether a patch is blamed for a
pre-existing failure, so it is tested directly rather than only through Docker.
"""

from __future__ import annotations

# Aliased so pytest does not try to collect the domain model as a test class.
from forgeflow.models import TestOutcome as Outcome
from forgeflow.sandbox import (
    compare_lint,
    compare_outcomes,
    discover_requirements,
    parse_junit_xml,
    parse_lint_findings,
    parse_pytest_text,
    summary_line,
)

JUNIT_WITH_FAILURE = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="3" failures="1" errors="1" skipped="1">
    <testcase classname="test_calc" name="test_add_positive" time="0.01"/>
    <testcase classname="test_calc" name="test_add_negative" time="0.01">
      <failure message="assert -3 == 3">AssertionError</failure>
    </testcase>
    <testcase classname="test_calc" name="test_add_zero" time="0.01">
      <error message="boom">RuntimeError</error>
    </testcase>
    <testcase classname="test_calc" name="test_skipped" time="0.0">
      <skipped message="nope"/>
    </testcase>
  </testsuite>
</testsuites>
"""

PYTEST_TEXT = """\
============================= test session starts ==============================
collected 3 items

test_calc.py ..F                                                         [100%]

=================================== FAILURES ===================================
_________________________________ test_add_zero _________________________________

    def test_add_zero():
>       assert add(0, 0) == 0
E       assert -0 == 0

=========================== short test summary info ============================
FAILED test_calc.py::test_add_zero - assert -0 == 0
1 failed, 2 passed in 0.05s
"""


class TestJUnitParsing:
    def test_counts_every_status(self):
        outcome = parse_junit_xml(JUNIT_WITH_FAILURE)
        assert outcome.total == 4
        assert outcome.passed == 1
        assert outcome.failed == 1
        assert outcome.errors == 1
        assert outcome.skipped == 1

    def test_records_node_ids_for_failures_and_errors(self):
        outcome = parse_junit_xml(JUNIT_WITH_FAILURE)
        assert "test_calc::test_add_negative" in outcome.failed_tests
        assert "test_calc::test_add_zero" in outcome.failed_tests

    def test_marks_collection(self):
        assert parse_junit_xml(JUNIT_WITH_FAILURE).collected is True

    def test_empty_report_is_not_collected(self):
        outcome = parse_junit_xml('<?xml version="1.0"?><testsuites></testsuites>')
        assert outcome.collected is False

    def test_broken_xml_is_not_collected(self):
        assert parse_junit_xml("not xml at all").collected is False

    def test_single_testsuite_root_is_supported(self):
        xml = (
            '<?xml version="1.0"?><testsuite tests="1">'
            '<testcase classname="a" name="b"/></testsuite>'
        )
        outcome = parse_junit_xml(xml)
        assert outcome.passed == 1


class TestTextParsing:
    def test_reads_the_summary_line(self):
        outcome = parse_pytest_text(PYTEST_TEXT)
        assert outcome.passed == 2
        assert outcome.failed == 1

    def test_lists_failed_node_ids(self):
        outcome = parse_pytest_text(PYTEST_TEXT)
        assert outcome.failed_tests == ["test_calc.py::test_add_zero"]

    def test_detects_zero_collection(self):
        outcome = parse_pytest_text("no tests ran in 0.01s")
        assert outcome.collected is False

    def test_handles_errors(self):
        text = "ERROR test_a.py - ImportError\n1 error in 0.10s"
        outcome = parse_pytest_text(text)
        assert outcome.errors == 1


class TestComparison:
    def outcome(self, failed: list[str], passed: int = 5) -> Outcome:
        return Outcome(
            total=passed + len(failed),
            passed=passed,
            failed=len(failed),
            failed_tests=sorted(failed),
        )

    def test_a_preexisting_failure_is_not_blamed_on_the_patch(self):
        baseline = self.outcome(["test_known_broken"])
        patched = self.outcome(["test_known_broken"])
        newly_failing, newly_passing = compare_outcomes(baseline, patched)
        assert newly_failing == []
        assert newly_passing == []

    def test_a_new_failure_is_detected(self):
        baseline = self.outcome(["test_known_broken"])
        patched = self.outcome(["test_known_broken", "test_add"])
        newly_failing, _ = compare_outcomes(baseline, patched)
        assert newly_failing == ["test_add"]

    def test_a_fixed_test_is_reported_as_newly_passing(self):
        baseline = self.outcome(["test_known_broken", "test_add"])
        patched = self.outcome(["test_known_broken"])
        newly_failing, newly_passing = compare_outcomes(baseline, patched)
        assert newly_failing == []
        assert newly_passing == ["test_add"]

    def test_the_real_scenario_fix_does_not_break_the_broken_test(self):
        """The fixture repository: three failures fixed, one pre-existing kept."""
        baseline = self.outcome(
            ["test_calc.py::test_add_positive", "test_calc.py::test_add_negative",
             "test_calc.py::test_add_zero", "test_known_broken.py::test_known_broken"]
        )
        patched = self.outcome(["test_known_broken.py::test_known_broken"])
        newly_failing, newly_passing = compare_outcomes(baseline, patched)
        assert newly_failing == [], "the pre-existing failure must not be blamed"
        assert len(newly_passing) == 3

    def test_summary_line_is_readable(self):
        line = summary_line(self.outcome(["a"]))
        assert "5 passed" in line and "1 failed" in line


class TestLintParsing:
    RUF_OUTPUT = """\
calc.py:1:8: F401 [*] `os` imported but unused
calc.py:5:1: F811 redefinition of unused name
pkg/other.py:2:1: E902 cannot open file
Found 3 errors.
"""

    def test_groups_findings_by_path(self):
        findings = parse_lint_findings(self.RUF_OUTPUT)
        assert findings["calc.py"] == {"F401", "F811"}
        assert findings["pkg/other.py"] == {"E902"}

    def test_ignores_the_summary_line(self):
        findings = parse_lint_findings("Found 3 errors.\n")
        assert findings == {}

    def test_pre_existing_finding_in_a_changed_file_is_not_a_regression(self):
        baseline = {"calc.py": {"F401"}}
        patched = {"calc.py": {"F401"}}
        assert compare_lint(baseline, patched, ["calc.py"]) == []

    def test_new_finding_in_a_changed_file_is_a_regression(self):
        baseline = {"calc.py": {"F401"}}
        patched = {"calc.py": {"F401", "F841"}}
        regressions = compare_lint(baseline, patched, ["calc.py"])
        assert regressions == ["calc.py: F841"]

    def test_new_finding_in_an_untouched_file_is_ignored(self):
        baseline = {}
        patched = {"unrelated.py": {"F401"}}
        assert compare_lint(baseline, patched, ["calc.py"]) == []


class TestRequirementsDiscovery:
    def test_reads_simple_pins(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\nclick>=8\n")
        found = discover_requirements(tmp_path)
        assert "requests==2.31.0" in found
        assert "click>=8" in found

    def test_skips_comments_and_blank_lines(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("# a comment\n\n   \nflask\n")
        assert discover_requirements(tmp_path) == ["flask"]

    def test_skips_vcs_and_local_paths(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "git+https://github.com/x/y.git\n-e .\n../local\nflask\n"
        )
        assert discover_requirements(tmp_path) == ["flask"]

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert discover_requirements(tmp_path) == []
