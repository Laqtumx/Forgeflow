"""Exception hierarchy.

Every failure ForgeFlow can attribute to a specific stage gets its own type so
the pipeline and the evaluation harness can report *why* a run failed instead of
collapsing everything into a generic error.
"""

from __future__ import annotations


class ForgeFlowError(Exception):
    """Base class for all ForgeFlow errors."""


class ConfigError(ForgeFlowError):
    """Configuration is missing or invalid."""


class GitHubError(ForgeFlowError):
    """A GitHub REST call failed."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimitError(GitHubError):
    """GitHub rate limit (or secondary limit) was hit."""


class LLMError(ForgeFlowError):
    """The model provider failed, or returned unusable output."""


class LLMJSONError(LLMError):
    """The model returned something that is not the requested JSON shape."""


class AnalysisError(ForgeFlowError):
    """Stage 1 analysis could not be produced or was not actionable."""


class EditError(ForgeFlowError):
    """Stage 2 edits could not be applied safely."""


class PathPolicyError(EditError):
    """The model tried to touch a path outside the allowed set."""


class SandboxError(ForgeFlowError):
    """The Docker sandbox could not be prepared or executed."""


class PatchVerificationError(ForgeFlowError):
    """The rebuilt patch does not match the approved fingerprint."""


class PublishError(ForgeFlowError):
    """Publishing to the fork or opening the draft PR failed."""
