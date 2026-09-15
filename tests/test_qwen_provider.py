"""Tests for the OpenAI-compatible provider payload (engine/qwen_provider.py)."""

from __future__ import annotations

from pathlib import Path

from engine.config import load_config
from engine.qwen_provider import OpenAICompatProvider


def _capture_payload(provider: OpenAICompatProvider) -> dict:
    """Run complete() with a fake transport and return the JSON body it sent."""
    captured: dict = {}

    def fake_post(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return {
            "choices": [{"message": {"content": "ACTION: EXPAND\nTARGET: NONE\nREASON: x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    provider._post = fake_post  # type: ignore[method-assign]
    provider.complete("prompt")
    return captured


def test_payload_omits_reasoning_effort_by_default() -> None:
    provider = OpenAICompatProvider(base_url="http://localhost:1234", model="m")
    captured = _capture_payload(provider)
    assert captured["path"] == "/v1/chat/completions"
    assert "reasoning_effort" not in captured["body"]
    assert captured["body"]["model"] == "m"


def test_payload_sends_reasoning_effort_when_set() -> None:
    provider = OpenAICompatProvider(
        base_url="http://localhost:1234", model="m", reasoning_effort="none",
    )
    captured = _capture_payload(provider)
    assert captured["body"]["reasoning_effort"] == "none"


def test_config_parses_reasoning_effort(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[llm]\nprovider = "lm-studio"\nreasoning_effort = "none"\n'
        '[llm.online]\nbase_url = "https://x/v1"\nmodel = "y"\nreasoning_effort = "low"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.llm.reasoning_effort == "none"
    assert cfg.llm.online_reasoning_effort == "low"


def test_config_reasoning_effort_defaults_empty(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[llm]\nprovider = "stub"\n', encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.llm.reasoning_effort == ""
    assert cfg.llm.online_reasoning_effort == ""
