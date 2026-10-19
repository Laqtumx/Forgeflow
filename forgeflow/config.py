"""Configuration loading.

Precedence is: explicit CLI arguments > environment variables > ``.env`` file >
defaults. The ``.env`` parser is intentionally tiny (``KEY=VALUE``, ``#``
comments, optional quotes) so ForgeFlow has no dependency on python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .defaults import (
    DEFAULT_EXCLUDED_LABELS,
    DEFAULT_LINT_COMMAND,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_TEST_COMMAND,
)
from .errors import ConfigError


def _parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _env(env: dict[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(env: dict[str, str], key: str, default: int) -> int:
    raw = _env(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _float(env: dict[str, str], key: str, default: float) -> float:
    raw = _env(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = _env(env, key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _list(env: dict[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env(env, key)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass
class Config:
    """Everything the pipeline needs to know about its environment."""

    # --- GitHub -----------------------------------------------------------
    # Read token: only needs public repo read access. Publishing uses a
    # separate, narrowly scoped token so a bug in the write path cannot reuse
    # broad credentials.
    github_read_token: str | None = None
    github_publish_token: str | None = None
    fork_owner: str | None = None

    # --- Model ------------------------------------------------------------
    llm_provider: str = "gemini"  # gemini | anthropic | mock
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    llm_timeout_s: int = 120
    llm_max_retries: int = 2
    confidence_threshold: float = 0.6

    # --- Storage ----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(".forgeflow/forgeflow.sqlite3"))
    work_dir: Path = field(default_factory=lambda: Path(".forgeflow/work"))

    # --- Sandbox ----------------------------------------------------------
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE
    sandbox_cpus: str = "2"
    sandbox_memory: str = "2g"
    sandbox_pids: int = 256
    test_timeout_s: int = 600
    install_timeout_s: int = 600
    max_attempts: int = 3
    test_command: str = DEFAULT_TEST_COMMAND
    lint_command: str = DEFAULT_LINT_COMMAND
    # Some environments need a prefix, e.g. FORGEFLOW_DOCKER="sudo docker".
    docker: str = "docker"

    # --- Ranking ----------------------------------------------------------
    excluded_labels: tuple[str, ...] = DEFAULT_EXCLUDED_LABELS
    max_candidates: int = 25
    repo_size_limit_kb: int = 250_000
    clone_depth: int = 50

    # --- Misc -------------------------------------------------------------
    allow_network_during_tests: bool = False
    keep_workdirs: bool = False

    @classmethod
    def load(
        cls,
        env: dict[str, str] | None = None,
        dotenv_path: Path | None = None,
        **overrides: object,
    ) -> Config:
        """Build a config from the environment, then apply ``overrides``.

        ``None`` values in ``overrides`` are ignored so callers can pass
        optional CLI flags straight through.
        """
        merged: dict[str, str] = {}
        path = dotenv_path or Path(".env")
        if path.is_file():
            merged.update(_parse_dotenv(path.read_text(encoding="utf-8")))
        merged.update(env if env is not None else os.environ)

        config = cls(
            github_read_token=_env(merged, "GITHUB_TOKEN")
            or _env(merged, "FORGEFLOW_GITHUB_READ_TOKEN"),
            github_publish_token=_env(merged, "FORGEFLOW_GITHUB_PUBLISH_TOKEN"),
            fork_owner=_env(merged, "FORGEFLOW_FORK_OWNER"),
            llm_provider=(_env(merged, "FORGEFLOW_LLM_PROVIDER") or "gemini").lower(),
            gemini_api_key=_env(merged, "GEMINI_API_KEY")
            or _env(merged, "GOOGLE_API_KEY"),
            anthropic_api_key=_env(merged, "ANTHROPIC_API_KEY"),
            gemini_model=_env(merged, "FORGEFLOW_GEMINI_MODEL") or "gemini-2.5-flash",
            anthropic_model=_env(merged, "FORGEFLOW_ANTHROPIC_MODEL")
            or "claude-sonnet-4-5-20250929",
            llm_timeout_s=_int(merged, "FORGEFLOW_LLM_TIMEOUT_S", 120),
            llm_max_retries=_int(merged, "FORGEFLOW_LLM_MAX_RETRIES", 2),
            confidence_threshold=_float(merged, "FORGEFLOW_CONFIDENCE_THRESHOLD", 0.6),
            db_path=Path(
                _env(merged, "FORGEFLOW_DB") or ".forgeflow/forgeflow.sqlite3"
            ),
            work_dir=Path(_env(merged, "FORGEFLOW_WORK_DIR") or ".forgeflow/work"),
            sandbox_image=_env(merged, "FORGEFLOW_SANDBOX_IMAGE") or DEFAULT_SANDBOX_IMAGE,
            sandbox_cpus=_env(merged, "FORGEFLOW_SANDBOX_CPUS") or "2",
            sandbox_memory=_env(merged, "FORGEFLOW_SANDBOX_MEMORY") or "2g",
            sandbox_pids=_int(merged, "FORGEFLOW_SANDBOX_PIDS", 256),
            test_timeout_s=_int(merged, "FORGEFLOW_TEST_TIMEOUT_S", 600),
            install_timeout_s=_int(merged, "FORGEFLOW_INSTALL_TIMEOUT_S", 600),
            max_attempts=_int(merged, "FORGEFLOW_MAX_ATTEMPTS", 3),
            test_command=_env(merged, "FORGEFLOW_TEST_COMMAND") or DEFAULT_TEST_COMMAND,
            lint_command=_env(merged, "FORGEFLOW_LINT_COMMAND") or DEFAULT_LINT_COMMAND,
            docker=_env(merged, "FORGEFLOW_DOCKER") or "docker",
            excluded_labels=_list(
                merged, "FORGEFLOW_EXCLUDED_LABELS", DEFAULT_EXCLUDED_LABELS
            ),
            max_candidates=_int(merged, "FORGEFLOW_MAX_CANDIDATES", 25),
            repo_size_limit_kb=_int(merged, "FORGEFLOW_REPO_SIZE_LIMIT_KB", 250_000),
            clone_depth=_int(merged, "FORGEFLOW_CLONE_DEPTH", 50),
            allow_network_during_tests=_bool(
                merged, "FORGEFLOW_ALLOW_NETWORK_DURING_TESTS", False
            ),
            keep_workdirs=_bool(merged, "FORGEFLOW_KEEP_WORKDIRS", False),
        )

        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(config, key):
                raise ConfigError(f"unknown configuration key: {key}")
            setattr(config, key, value)

        if config.llm_provider not in {"gemini", "anthropic", "mock"}:
            raise ConfigError(
                "FORGEFLOW_LLM_PROVIDER must be one of: gemini, anthropic, mock"
            )
        if config.max_attempts < 1:
            raise ConfigError("FORGEFLOW_MAX_ATTEMPTS must be >= 1")
        if not 0.0 <= config.confidence_threshold <= 1.0:
            raise ConfigError("FORGEFLOW_CONFIDENCE_THRESHOLD must be between 0 and 1")

        config.db_path = Path(config.db_path).expanduser()
        config.work_dir = Path(config.work_dir).expanduser()
        return config

    # -- derived helpers ---------------------------------------------------

    def provider_api_key(self) -> str | None:
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return "mock"  # the mock provider needs no credentials

    def model_name(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_model
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return "mock-deterministic"

    def require_llm_credentials(self) -> None:
        if self.llm_provider == "mock":
            return
        if not self.provider_api_key():
            var = "GEMINI_API_KEY" if self.llm_provider == "gemini" else "ANTHROPIC_API_KEY"
            raise ConfigError(
                f"{var} is required when FORGEFLOW_LLM_PROVIDER={self.llm_provider}"
            )

    def require_publish_credentials(self) -> None:
        if not self.github_publish_token:
            raise ConfigError(
                "FORGEFLOW_GITHUB_PUBLISH_TOKEN is required to publish. Create a "
                "separate fine-grained token with 'contents: write' and "
                "'pull requests: write' on your fork only."
            )
        if not self.fork_owner:
            raise ConfigError(
                "FORGEFLOW_FORK_OWNER is required to publish (the account that owns the fork)."
            )
