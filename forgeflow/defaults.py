"""Shared defaults, kept in one place so config and sandbox cannot drift."""

from __future__ import annotations

DEFAULT_TEST_COMMAND = "python -m pytest -q -p no:cacheprovider --tb=short"

# ruff, restricted to rules that flag real defects rather than style taste, and
# diffed against the baseline so pre-existing findings are not blamed on a patch.
DEFAULT_LINT_SELECT = "E9,F63,F7,F82,F401,F811,F841"
DEFAULT_LINT_COMMAND = (
    f"python -m ruff check --no-cache --output-format=concise "
    f"--select {DEFAULT_LINT_SELECT} {{files}}"
)

# Installed into the sandbox site volume on every run.
DEFAULT_SANDBOX_PACKAGES = ("pytest", "ruff")

DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"
DEFAULT_EXCLUDED_LABELS = (
    "duplicate",
    "wontfix",
    "invalid",
    "question",
    "discussion",
    "needs-design",
    "rfc",
    "epic",
    "meta",
    "security",
    "help wanted",
)
