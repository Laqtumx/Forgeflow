"""Model layer: providers, prompts, strict parsing, and the two-call workflow."""

from .contract import (
    LLMClient,
    PromptContext,
    analysis_is_actionable,
    build_analysis_prompt,
    build_edit_prompt,
    extract_json,
    parse_analysis,
    parse_edit_plan,
)
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMResponse,
    MockProvider,
    Provider,
    build_provider,
)

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "LLMClient",
    "LLMResponse",
    "MockProvider",
    "PromptContext",
    "Provider",
    "analysis_is_actionable",
    "build_analysis_prompt",
    "build_edit_prompt",
    "build_provider",
    "extract_json",
    "parse_analysis",
    "parse_edit_plan",
]
