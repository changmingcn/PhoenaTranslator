"""The only OpenAI-compatible network adapter in PhoenaTranslator."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


class LLMConfigurationError(RuntimeError):
    """Raised when semantic translation is requested without runtime credentials."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = 120.0


class DeepSeekTranslationAdapter:
    """Lazy adapter whose sole public operation is semantic translation."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client_factory: Callable[..., Any] = OpenAI,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self._client_lock = threading.Lock()

    def _client(self) -> Any:
        if not self._settings.api_key:
            raise LLMConfigurationError(
                "DEEPSEEK_API_KEY is required when semantic translation is requested"
            )
        if self._client_instance is None:
            with self._client_lock:
                if self._client_instance is None:
                    self._client_instance = self._client_factory(
                        api_key=self._settings.api_key,
                        base_url=self._settings.base_url,
                        timeout=self._settings.timeout_seconds,
                    )
        return self._client_instance

    def translate(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ) -> Any:
        """Generate a semantic translation through the configured provider."""
        return self._client().chat.completions.create(
            model=model or self._settings.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

