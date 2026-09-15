"""
Anthropic Provider — Claude via the official ``anthropic`` SDK.

Used for the "online" slot of hybrid mode and for the strategic planner,
where a frontier model (Claude Opus) is asked for a long-horizon plan once
every few in-game years while the local model handles per-tick decisions.

Design notes:
  - Thinking is left at the model default (adaptive on Opus 5); depth is
    steered with ``output_config.effort`` instead of a token budget.
  - Sampling parameters (temperature/top_p) are NOT sent — current Claude
    models reject them.
  - ``reasoning_effort`` from config maps onto Claude effort levels; the
    OpenAI-style value ``"none"`` is translated to ``"low"`` because Claude
    Opus 5 performs better with shallow thinking than with thinking off.
  - The API key is resolved by the SDK (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile) unless one is passed explicitly.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from engine.llm_provider import LLMProvider, LLMProviderError, LLMResponse

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

_SYSTEM_PROMPT = (
    "You are a Stellaris 4.4.6 strategic AI advisor. "
    "Respond ONLY in the exact format requested. "
    "Never invent game mechanics."
)

_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def map_effort(reasoning_effort: str) -> str | None:
    """Translate the config's ``reasoning_effort`` into a Claude effort level.

    Returns *None* when the API default should be used.
    """
    value = reasoning_effort.strip().lower()
    if not value:
        return None
    if value == "none":
        return "low"
    if value in _EFFORT_LEVELS:
        return value
    raise ValueError(
        f"Unsupported reasoning_effort {reasoning_effort!r} for Anthropic; "
        f"use one of {', '.join(_EFFORT_LEVELS)} or 'none'",
    )


class AnthropicProvider(LLMProvider):
    """Talk to Claude through the official Anthropic SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        reasoning_effort: str = "",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._effort = map_effort(reasoning_effort)
        self.stats = _AnthropicStats()

        if client is not None:
            self._client = client
        else:
            import anthropic

            kwargs: dict[str, Any] = {"timeout": timeout_s, "max_retries": 2}
            if api_key:
                kwargs["api_key"] = api_key
            self._client = anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------ #
    # LLMProvider interface
    # ------------------------------------------------------------------ #

    def complete(self, prompt: str) -> LLMResponse:
        import anthropic

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._effort is not None:
            request["output_config"] = {"effort": self._effort}

        t0 = time.monotonic()
        try:
            response = self._client.messages.create(**request)
        except anthropic.APIError as exc:
            self.stats.failures += 1
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc
        except TypeError as exc:
            # The SDK raises TypeError when no credentials can be resolved.
            self.stats.failures += 1
            raise LLMProviderError(
                f"Anthropic credentials missing ({exc}); set ANTHROPIC_API_KEY "
                "or [llm.online] api_key",
            ) from exc
        latency = (time.monotonic() - t0) * 1000

        if response.stop_reason == "refusal":
            self.stats.failures += 1
            raise LLMProviderError("Anthropic declined the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            log.warning(
                "Anthropic response hit max_tokens=%d — consider raising [llm.online] max_tokens",
                self._max_tokens,
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            self.stats.failures += 1
            raise LLMProviderError(
                f"Anthropic returned no text (stop_reason={response.stop_reason})",
            )

        usage = response.usage
        result = LLMResponse(
            text=text.strip(),
            model=response.model,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
            latency_ms=latency,
        )
        self.stats.calls += 1
        self.stats.tokens += result.prompt_tokens + result.completion_tokens
        return result

    def is_available(self) -> bool:
        import anthropic

        try:
            self._client.models.retrieve(self._model)
            return True
        except (anthropic.APIError, TypeError) as exc:
            log.warning("Anthropic unavailable: %s", exc)
            return False

    @property
    def name(self) -> str:
        return f"anthropic ({self._model})"


class _AnthropicStats:
    """Minimal usage counters, mirrors the shape used by other providers."""

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self.tokens = 0
