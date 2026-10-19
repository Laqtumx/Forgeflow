"""Edit application and path policy.

These are the tests that matter most for safety: they pin down what the model
is allowed to touch, and prove that a hostile edit plan is rejected before it
can write anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeflow.edits import (
    MAX_CHANGED_LINES,
    apply_edits,
    fingerprint_patch,
    is_forbidden_path,
)
from forgeflow.errors import EditError, PathPolicyError
from forgeflow.models import Edit, EditPlan


def plan(*edits: tuple[str, str, str]) -> EditPlan:
    return EditPlan(
        edits=[
            Edit(path=path, search=search, replace=replace) for path, search, replace in edits
        ]
    )


def _apply_bypassing_schema(root: Path, edits: list[Edit]):
    """Run apply_edits with a plan built without pydantic validation.

    Proves the runtime cap exists independently of the schema cap.
    """
    from forgeflow.edits import apply_edits as _apply

    class UnvalidatedPlan:
        def __init__(self, edits: list[Edit]) -> None:
            self.edits = edits

    return _apply(root, UnvalidatedPlan(edits))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("# title\n")
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    (tmp_path / "conftest.py").write_text("collect_ignore_glob = ['*']\n")
    (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02")
    return tmp_path


class TestPathPolicy:
    def test_allows_ordinary_python_paths(self):
        assert is_forbidden_path("pkg/calc.py") is None
        assert is_forbidden_path("README.md") is None

    @pytest.mark.parametrize(
        "path,expected_fragment",
        [
            ("/etc/passwd", "repository-relative"),
            ("../../etc/passwd", ".."),
            ("pkg/../../outside.py", "'..'"),
            (".git/config", "not writable"),
            (".github/workflows/ci.yml", "not writable"),
            ("setup.py", "not writable"),
            ("conftest.py", "not writable"),
            ("pyproject.toml", "not writable"),
            ("noxfile.py", "not writable"),
            ("evil.pth", "not writable"),
            ("script.sh", "outside the v1 allowlist"),
            ("lib/native.so", "outside the v1 allowlist"),
            ("~/.bashrc", "repository-relative"),
        ],
    )
    def test_rejects_dangerous_paths(self, path, expected_fragment):
        reason = is_forbidden_path(path)
        assert reason is not None
        assert expected_fragment in reason

    def test_apply_refuses_to_escape_the_root(self, tree):
        with pytest.raises(PathPolicyError):
            apply_edits(tree, plan(("../outside.py", "a", "b")))

    def test_apply_refuses_a_symlink(self, tree):
        link = tree / "link.py"
        link.symlink_to(tree / "pkg" / "calc.py")
        with pytest.raises(PathPolicyError, match="symlink"):
            apply_edits(tree, plan(("link.py", "a", "b")))

    def test_apply_refuses_a_missing_file(self, tree):
        with pytest.raises(EditError, match="does not exist"):
            apply_edits(tree, plan(("pkg/ghost.py", "a", "b")))

    def test_apply_refuses_a_binary_file(self, tree):
        with pytest.raises(EditError):
            apply_edits(tree, plan(("binary.py", "x", "y")))


class TestSearchReplace:
    def test_applies_a_single_exact_replacement(self, tree):
        result = apply_edits(tree, plan(("pkg/calc.py", "return a - b", "return a + b")))
        assert (tree / "pkg" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
        assert result.files_changed == ["pkg/calc.py"]
        assert result.applied[0].occurrences == 1

    def test_rejects_search_text_that_is_absent(self, tree):
        with pytest.raises(EditError, match="does not appear"):
            apply_edits(tree, plan(("pkg/calc.py", "return a * b", "return a + b")))

    def test_rejects_ambiguous_search_text(self, tmp_path):
        (tmp_path / "dup.py").write_text("x = 1\nx = 1\n")
        with pytest.raises(EditError, match="appears 2 times"):
            apply_edits(tmp_path, plan(("dup.py", "x = 1", "x = 2")))

    def test_rejects_a_no_op_edit(self, tree):
        with pytest.raises(EditError, match="no changes"):
            apply_edits(tree, plan(("pkg/calc.py", "return a - b", "return a - b")))

    def test_is_atomic_when_a_later_edit_fails(self, tree):
        original = (tree / "pkg" / "calc.py").read_text()
        with pytest.raises(EditError):
            apply_edits(
                tree,
                plan(
                    ("pkg/calc.py", "return a - b", "return a + b"),
                    ("pkg/calc.py", "not in the file", "anything"),
                ),
            )
        assert (tree / "pkg" / "calc.py").read_text() == original, (
            "a failed plan must leave the tree untouched"
        )

    def test_rejects_a_change_larger_than_the_line_budget(self, tmp_path):
        (tmp_path / "big.py").write_text("start\n")
        replacement = "\n".join(f"line{i}" for i in range(MAX_CHANGED_LINES + 50))
        with pytest.raises(EditError, match="lines"):
            apply_edits(tmp_path, plan(("big.py", "start", replacement)))

    def test_rejects_too_many_edits(self, tree):
        # The schema caps a plan at ten edits, so the model cannot even produce
        # an oversized plan. apply_edits enforces the same cap defensively, in
        # case a plan ever reaches it from somewhere else.
        edits = [Edit(path="pkg/calc.py", search=f"x{i}", replace=f"y{i}") for i in range(20)]
        with pytest.raises(Exception, match=r"at most 10|too_long"):
            EditPlan(edits=edits)
        with pytest.raises(EditError, match="limit"):
            _apply_bypassing_schema(tree, edits)

    def test_rejects_a_plan_that_only_touches_a_forbidden_file(self, tree):
        with pytest.raises(PathPolicyError):
            apply_edits(tree, plan(("setup.py", "setup()", "setup(name='x')")))


class TestFingerprint:
    def test_is_stable_for_the_same_patch(self):
        patch = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        assert fingerprint_patch(patch) == fingerprint_patch(patch)

    def test_is_insensitive_to_trailing_whitespace_and_final_newline(self):
        a = "diff --git a/x.py b/x.py\n+line\n"
        b = "diff --git a/x.py b/x.py\n+line"
        assert fingerprint_patch(a) == fingerprint_patch(b)

    def test_changes_when_the_patch_changes(self):
        a = "diff --git a/x.py b/x.py\n+line one\n"
        b = "diff --git a/x.py b/x.py\n+line two\n"
        assert fingerprint_patch(a) != fingerprint_patch(b)

    def test_is_a_sha256_hex_string(self):
        fingerprint = fingerprint_patch("anything")
        assert len(fingerprint) == 64
        assert all(char in "0123456789abcdef" for char in fingerprint)


class TestPlanValidation:
    def test_rejects_an_empty_plan(self):
        with pytest.raises(Exception):
            EditPlan(edits=[])

    def test_rejects_an_empty_search(self):
        with pytest.raises(Exception):
            Edit(path="a.py", search="", replace="x")

    def test_normalises_a_leading_dot_slash(self):
        edit = Edit(path="./pkg/calc.py", search="a", replace="b")
        assert edit.path == "pkg/calc.py"

    def test_rejects_duplicate_edits(self):
        with pytest.raises(Exception):
            EditPlan(
                edits=[
                    Edit(path="a.py", search="x", replace="y"),
                    Edit(path="a.py", search="x", replace="z"),
                ]
            )

    def test_rejects_unknown_fields(self):
        """The schema is closed, so the model cannot smuggle in new keys."""
        with pytest.raises(Exception):
            Edit(path="a.py", search="x", replace="y", shell_command="rm -rf /")
