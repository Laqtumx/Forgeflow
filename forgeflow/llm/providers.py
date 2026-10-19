"""Model providers.

ForgeFlow speaks plain HTTPS to the provider REST APIs rather than pulling in a
vendor SDK. That keeps the dependency surface at ``requests`` and makes the
provider boundary a single method, which the tests replace with a deterministic
mock.

A provider only ever *returns text*. All interpretation, validation, and the
decision to touch a file happen in ``forgeflow.llm.contract``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from ..errors import LLMError

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class LLMResponse:
    """One model reply, with the accounting needed for the results table."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """The single method the rest of ForgeFlow depends on."""

    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse: ...


class _HTTPProvider:
    """Shared retry/timeout behaviour for the real providers."""

    name = "http"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout: int = 120,
        max_retries: int = 2,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        if not api_key:
            raise LLMError(f"{self.name} provider requires an API key")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._sleep = sleep

    def _post(self, url: str, *, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url, headers=headers, json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(min(2 ** attempt, 8))
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = LLMError(
                    f"{self.name} returned {response.status_code}: {response.text[:200]}"
                )
                if attempt == self.max_retries:
                    break
                self._sleep(min(2 ** attempt, 8))
                continue

            if response.status_code >= 400:
                raise LLMError(
                    f"{self.name} returned {response.status_code}: {response.text[:400]}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise LLMError(f"{self.name} returned a non-JSON body") from exc

        raise LLMError(f"{self.name} request failed: {last_error}")

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:  # pragma: no cover - exercised through subclasses
        raise NotImplementedError


class GeminiProvider(_HTTPProvider):
    """Google Gemini via ``generateContent`` (the free tier works here)."""

    name = "gemini"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        url = GEMINI_ENDPOINT.format(model=self.model)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        started = time.monotonic()
        payload = self._post(url, headers=headers, body=body)
        latency = time.monotonic() - started

        text = self._extract_text(payload)
        usage = payload.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            latency_s=round(latency, 3),
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            feedback = payload.get("promptFeedback") or {}
            raise LLMError(f"gemini returned no candidates: {feedback}")
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)
        if not text.strip():
            raise LLMError(f"gemini returned empty text (finishReason={finish})")
        return text


class AnthropicProvider(_HTTPProvider):
    """Anthropic Messages API (Claude)."""

    name = "anthropic"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        started = time.monotonic()
        payload = self._post(ANTHROPIC_ENDPOINT, headers=headers, body=body)
        latency = time.monotonic() - started

        blocks = payload.get("content") or []
        text = "".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )
        if not text.strip():
            reason = payload.get("stop_reason")
            raise LLMError(f"anthropic returned empty text (stop_reason={reason})")
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            latency_s=round(latency, 3),
        )


class MockProvider:
    """Deterministic offline provider.

    It is a real code path, not a stub that bypasses the pipeline: it returns
    schema-valid JSON built from the prompt, so the whole analysis -> edits ->
    sandbox -> fingerprint flow is exercised without network access or quota.
    The ``edits`` it emits are discovered by scanning the repository context
    block that the prompt builder embeds, using rules registered by the caller.
    """

    name = "mock"

    def __init__(self, model: str = "mock-deterministic", *, seed: int = 0) -> None:
        self.model = model
        self._rng = random.Random(seed)
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append((system, user))
        if "edit plan" in system.lower() or '"edits"' in system:
            text = self._edit_plan(user)
        else:
            text = self._analysis(user)
        return LLMResponse(text=text, provider=self.name, model=self.model, latency_s=0.0)

    @staticmethod
    def _first_line(user: str) -> str:
        for line in user.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]
        return "the reported issue"

    def _analysis(self, user: str) -> str:
        return json.dumps(
            {
                "summary": f"Mock analysis of: {self._first_line(user)}",
                "root_cause_hypothesis": (
                    "Mock hypothesis: the behaviour diverges from the documented contract "
                    "in the function named by the issue."
                ),
                "files_to_inspect": ["src/example.py"],
                "fix_plan": [
                    "Locate the offending branch.",
                    "Correct the condition.",
                    "Confirm with the reproduction in the issue.",
                ],
                "confidence": 0.75,
                "fixable": True,
                "needs_more_info": False,
            }
        )

    def _edit_plan(self, user: str) -> str:
        edits = []
        for path, search, replace in _parse_edit_directives(user):
            edits.append(
                {
                    "path": path,
                    "search": search,
                    "replace": replace,
                    "rationale": "mock directive",
                }
            )
        if not edits:
            edits = [
                {
                    "path": "src/example.py",
                    "search": "return a - b",
                    "replace": "return a + b",
                    "rationale": "mock directive",
                }
            ]
        return json.dumps({"edits": edits, "notes": "mock edit plan"})


def _parse_edit_directives(user: str) -> list[tuple[str, str, str]]:
    """Read ``<<<EDIT path|search|replace>>>`` directives out of a prompt.

    The test suite uses these to drive the mock provider end to end.
    """
    import re

    pattern = re.compile(r"<<<EDIT (.*?)>>>(.*?)<<<END>>>", re.DOTALL)
    directives: list[tuple[str, str, str]] = []
    for match in pattern.finditer(user):
        header, block = match.group(1), match.group(2)
        if "|" not in header:
            continue
        path, _, replace = header.partition("|")
        directives.append((path.strip(), block.strip("\n"), replace))
    return directives


def build_provider(config: Any, **kwargs: Any) -> Provider:
    """Construct the provider named by the configuration."""
    if config.llm_provider == "gemini":
        return GeminiProvider(
            config.gemini_model,
            config.gemini_api_key or "",
            timeout=config.llm_timeout_s,
            max_retries=config.llm_max_retries,
            **kwargs,
        )
    if config.llm_provider == "anthropic":
        return AnthropicProvider(
            config.anthropic_model,
            config.anthropic_api_key or "",
            timeout=config.llm_timeout_s,
            max_retries=config.llm_max_retries,
            **kwargs,
        )
    if config.llm_provider == "mock":
        return MockProvider()
    raise LLMError(f"unknown provider: {config.llm_provider}")
