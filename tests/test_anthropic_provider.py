"""Tests for engine/anthropic_provider.py (official SDK wrapper, mocked client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from engine.anthropic_provider import AnthropicProvider, map_effort
from engine.config import load_config
from engine.llm_provider import LLMProviderError

# --------------------------------------------------------------------- #
# Fake SDK client
# --------------------------------------------------------------------- #

@dataclass
class _Block:
    type: str
    text: str = ""


@dataclass
class _Usage:
    input_tokens: int = 120
    output_tokens: int = 40


@dataclass
class _Response:
    content: list[_Block]
    stop_reason: str = "end_turn"
    model: str = "claude-opus-5"
    usage: _Usage = field(default_factory=_Usage)


class _Messages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _Messages(response)


def _provider(response: _Response, **kwargs: object) -> AnthropicProvider:
    return AnthropicProvider(client=_FakeClient(response), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Effort mapping
# --------------------------------------------------------------------- #

def test_map_effort_defaults_and_none() -> None:
    assert map_effort("") is None
    assert map_effort("none") == "low"
    assert map_effort("Medium") == "medium"
    assert map_effort("xhigh") == "xhigh"


def test_map_effort_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        map_effort("turbo")


# --------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------- #

def test_request_shape_without_effort() -> None:
    resp = _Response(content=[_Block("text", "ACTION: EXPAND\nTARGET: NONE\nREASON: x")])
    provider = _provider(resp, model="claude-opus-5", max_tokens=1234)
    provider.complete("hello")

    call = provider._client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 1234
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert "Stellaris 4.4.6" in call["system"]
    # Current Claude models reject sampling params and budget_tokens
    assert "temperature" not in call
    assert "thinking" not in call
    assert "output_config" not in call


def test_request_sends_effort_when_configured() -> None:
    resp = _Response(content=[_Block("text", "ok")])
    provider = _provider(resp, reasoning_effort="medium")
    provider.complete("hello")
    assert provider._client.messages.calls[0]["output_config"] == {"effort": "medium"}


# --------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------- #

def test_joins_text_blocks_and_skips_thinking() -> None:
    resp = _Response(content=[
        _Block("thinking", ""),
        _Block("text", "THREAT_LEVEL: low\n"),
        _Block("text", "FOCUS: tech rush"),
    ])
    provider = _provider(resp)
    result = provider.complete("plan")
    assert result.text == "THREAT_LEVEL: low\nFOCUS: tech rush"
    assert result.model == "claude-opus-5"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 40
    assert provider.stats.calls == 1
    assert provider.stats.tokens == 160


def test_refusal_raises_provider_error() -> None:
    resp = _Response(content=[], stop_reason="refusal")
    provider = _provider(resp)
    with pytest.raises(LLMProviderError):
        provider.complete("plan")
    assert provider.stats.failures == 1


def test_empty_text_raises_provider_error() -> None:
    resp = _Response(content=[_Block("thinking", "")], stop_reason="max_tokens")
    provider = _provider(resp)
    with pytest.raises(LLMProviderError):
        provider.complete("plan")


def test_name() -> None:
    provider = _provider(_Response(content=[]), model="claude-opus-5")
    assert provider.name == "anthropic (claude-opus-5)"


# --------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------- #

def test_config_parses_online_provider(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[llm]\nprovider = "lm-studio"\n'
        '[llm.online]\nprovider = "anthropic"\nmodel = "claude-opus-5"\n'
        'reasoning_effort = "medium"\nmax_tokens = 4096\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.llm.online_provider == "anthropic"
    assert cfg.llm.online_model == "claude-opus-5"
    assert cfg.llm.online_reasoning_effort == "medium"
    assert cfg.llm.online_max_tokens == 4096


def test_config_online_provider_defaults_to_openai_compat(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[llm]\nprovider = "stub"\n', encoding="utf-8")
    assert load_config(cfg_file).llm.online_provider == "openai-compat"


def test_base_url_is_passed_to_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    captured: dict = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    AnthropicProvider(api_key="sk-test", base_url="https://relay.example/v1/")
    assert captured["base_url"] == "https://relay.example/v1"
    assert captured["api_key"] == "sk-test"


def test_proxy_builds_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    captured: dict = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class _HttpClient:
        def __init__(self, proxy: str) -> None:
            self.proxy = proxy

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setattr(anthropic, "DefaultHttpxClient", _HttpClient)
    AnthropicProvider(api_key="sk-test", proxy="socks5://127.0.0.1:2080")
    assert captured["http_client"].proxy == "socks5://127.0.0.1:2080"

    captured.clear()
    AnthropicProvider(api_key="sk-test")
    assert "http_client" not in captured


def test_config_parses_proxy(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "\n".join([
            "[llm]",
            'provider = "stub"',
            'proxy = "http://p:1"',
            "[llm.online]",
            'provider = "anthropic"',
            'model = "m"',
            'proxy = "socks5://127.0.0.1:2080"',
        ]),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.llm.proxy == "http://p:1"
    assert cfg.llm.online_proxy == "socks5://127.0.0.1:2080"


def test_build_online_provider_uses_anthropic(tmp_path: Path) -> None:
    from engine.main import _build_online_provider

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[llm]\nprovider = "stub"\n'
        '[llm.online]\nprovider = "anthropic"\nmodel = "claude-opus-5"\napi_key = "sk-test"\n',
        encoding="utf-8",
    )
    provider = _build_online_provider(load_config(cfg_file))
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic (claude-opus-5)"
