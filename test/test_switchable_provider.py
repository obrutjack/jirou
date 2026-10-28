"""Unit tests for SwitchableProvider — /switch command mid-session logic.

Coverage:
  - /switch cloud  (happy path, not configured, unreachable, already active)
  - /switch local  (happy path, unreachable, already active)
  - Context summary building and injection
  - Non-switch messages delegate to active provider
  - Config loader returns SwitchableProvider for "local-llm"

Implements: https://github.com/obrutjack/jirou/issues/13
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    AcpEvent as LLMEvent,
)
from kiro_crew.providers.switchable import SwitchableProvider

# ── helpers ───────────────────────────────────────────────────────────────────


async def _drain(ait) -> list[LLMEvent]:
    out = []
    async for event in ait:
        out.append(event)
    return out


def _make_chunk(content: str | None = None, finish_reason: str | None = None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = []
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


@asynccontextmanager
async def _mock_stream(chunks):
    async def _gen():
        for c in chunks:
            yield c

    yield _gen()


def _provider_with_patched_local(mock_client) -> SwitchableProvider:
    """Return a SwitchableProvider whose local sub-provider uses mock_client."""
    p = SwitchableProvider()
    p._local._client = mock_client
    p._local._messages = [{"role": "system", "content": "sys"}]
    p._active = p._local
    return p


# ── /switch cloud — happy path ─────────────────────────────────────────────


class TestSwitchCloud:
    @pytest.mark.asyncio
    async def test_switch_cloud_returns_confirmation(self):
        """Successful /switch cloud yields a success TEXT_CHUNK."""
        p = SwitchableProvider()

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

            with patch.dict(
                os.environ,
                {
                    "CLOUD_LLM_BASE_URL": "https://api.openai.com/v1",
                    "CLOUD_LLM_API_KEY": "sk-test",
                    "CLOUD_LLM_MODEL": "gpt-4o",
                },
            ):
                events = await _drain(p.stream("/switch cloud"))

        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK)
        assert "switched" in text.lower() or "cloud" in text.lower()
        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_switch_cloud_updates_active_name(self):
        p = SwitchableProvider()

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

            with patch.dict(
                os.environ,
                {
                    "CLOUD_LLM_BASE_URL": "https://api.openai.com/v1",
                    "CLOUD_LLM_API_KEY": "sk-test",
                    "CLOUD_LLM_MODEL": "gpt-4o",
                },
            ):
                await _drain(p.stream("/switch cloud"))

        assert p._active_name == "cloud"
        assert p._active is p._cloud

    @pytest.mark.asyncio
    async def test_switch_cloud_context_carried_to_new_provider(self):
        """Context summary from local is injected into the cloud provider's messages."""
        p = SwitchableProvider()
        # Seed the local provider with some conversation history.
        p._local._messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

            with patch.dict(
                os.environ,
                {
                    "CLOUD_LLM_BASE_URL": "https://api.openai.com/v1",
                    "CLOUD_LLM_API_KEY": "sk-test",
                    "CLOUD_LLM_MODEL": "gpt-4o",
                },
            ):
                await _drain(p.stream("/switch cloud"))

        # Cloud provider's messages should include the context summary.
        cloud_messages = p._cloud._messages
        roles = [m["role"] for m in cloud_messages]
        assert "user" in roles
        assert "assistant" in roles
        # The context summary should mention the carried-over messages.
        user_content = " ".join(m["content"] for m in cloud_messages if m["role"] == "user")
        assert "2+2" in user_content or "Context" in user_content


# ── /switch cloud — error paths ────────────────────────────────────────────


class TestSwitchCloudErrors:
    @pytest.mark.asyncio
    async def test_switch_cloud_not_configured_emits_guidance(self):
        """When CLOUD_LLM_BASE_URL is not set, emit a guidance message."""
        p = SwitchableProvider()
        env = {k: v for k, v in os.environ.items() if k != "CLOUD_LLM_BASE_URL"}

        with patch.dict(os.environ, env, clear=True):
            events = await _drain(p.stream("/switch cloud"))

        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "cloud_llm_base_url" in text or "not configured" in text
        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "error"

    @pytest.mark.asyncio
    async def test_switch_cloud_unreachable_reuses_connectivity_notification(self):
        """When cloud endpoint is unreachable, emit the same #12 notification."""
        from openai import APIConnectionError

        p = SwitchableProvider()

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))

            with patch.dict(
                os.environ,
                {
                    "CLOUD_LLM_BASE_URL": "https://api.openai.com/v1",
                    "CLOUD_LLM_API_KEY": "sk-test",
                },
            ):
                events = await _drain(p.stream("/switch cloud"))

        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "unreachable" in text or "not responding" in text
        assert events[-1].stop_reason == "error"
        # Active should NOT have changed.
        assert p._active_name == "local"

    @pytest.mark.asyncio
    async def test_switch_cloud_already_active(self):
        """Switching to cloud when already on cloud emits a no-op message."""
        p = SwitchableProvider()
        p._active_name = "cloud"  # simulate already on cloud

        events = await _drain(p.stream("/switch cloud"))

        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "already" in text
        assert events[-1].stop_reason == "end_turn"


# ── /switch local ──────────────────────────────────────────────────────────


class TestSwitchLocal:
    @pytest.mark.asyncio
    async def test_switch_local_already_active(self):
        """Switching to local when already on local emits a no-op message."""
        p = SwitchableProvider()
        # default is local

        events = await _drain(p.stream("/switch local"))

        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "already" in text
        assert events[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_switch_local_from_cloud_returns_confirmation(self):
        """Switching back to local after being on cloud works correctly."""
        p = SwitchableProvider()
        p._active_name = "cloud"  # simulate currently on cloud
        p._active = MagicMock()  # pretend cloud provider exists
        p._active._messages = [
            {"role": "system", "content": "cloud sys"},
            {"role": "user", "content": "cloud question"},
            {"role": "assistant", "content": "cloud answer"},
        ]

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

            events = await _drain(p.stream("/switch local"))

        assert p._active_name == "local"
        text = " ".join(e.text for e in events if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "switched" in text or "local" in text
        assert events[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_switch_local_unreachable_does_not_switch(self):
        """When local LLM is unreachable during /switch local, abort and keep active."""
        from openai import APIConnectionError

        p = SwitchableProvider()
        p._active_name = "cloud"
        p._active = MagicMock()
        p._active._messages = []

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            MockOAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))

            events = await _drain(p.stream("/switch local"))

        assert p._active_name == "cloud"  # must NOT have changed
        assert events[-1].stop_reason == "error"


# ── Non-switch messages delegate ──────────────────────────────────────────


class TestDelegation:
    @pytest.mark.asyncio
    async def test_normal_message_delegates_to_active(self):
        """Regular messages are forwarded to the active provider unchanged."""
        p = SwitchableProvider()

        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(finish_reason="stop"),
        ]
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_stream(chunks))
        p._local._client = mock_client
        p._local._messages = [{"role": "system", "content": "sys"}]

        events = await _drain(p.stream("How are you?"))
        text_events = [e for e in events if e.kind == EVENT_TEXT_CHUNK]
        assert any(e.text == "Hello" for e in text_events)

    @pytest.mark.asyncio
    async def test_switch_case_insensitive(self):
        """The /switch command is matched case-insensitively."""
        p = SwitchableProvider()

        events = await _drain(p.stream("/SWITCH LOCAL"))

        # Already on local → no-op message (no KeyError or delegation to LLM)
        assert any(e.kind == EVENT_TEXT_CHUNK for e in events)
        assert events[-1].kind == EVENT_COMPLETE


# ── Context summary helpers ────────────────────────────────────────────────


class TestContextSummary:
    def test_empty_history_returns_placeholder(self):
        p = SwitchableProvider()
        p._active._messages = []
        summary = p._build_context_summary()
        assert "No conversation history" in summary

    def test_summary_includes_recent_user_assistant_turns(self):
        p = SwitchableProvider()
        p._active._messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Tell me about Python."},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        summary = p._build_context_summary()
        assert "Tell me about Python" in summary
        assert "programming language" in summary

    def test_summary_excludes_system_messages(self):
        p = SwitchableProvider()
        p._active._messages = [
            {"role": "system", "content": "SECRET SYSTEM PROMPT"},
            {"role": "user", "content": "user question"},
            {"role": "assistant", "content": "answer"},
        ]
        summary = p._build_context_summary()
        assert "SECRET SYSTEM PROMPT" not in summary

    def test_summary_truncates_long_content(self):
        p = SwitchableProvider()
        long_content = "X" * 1000
        p._active._messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "ok"},
        ]
        summary = p._build_context_summary()
        # Content must be truncated.
        assert "X" * 1000 not in summary
        assert "…" in summary

    def test_inject_context_appends_exchange(self):
        from kiro_crew.providers.local_llm import LocalLLMProvider

        p = SwitchableProvider()
        target = LocalLLMProvider()
        target._messages = [{"role": "system", "content": "sys"}]

        p._inject_context(target, "context summary text")

        assert len(target._messages) == 3  # system + user + assistant
        assert target._messages[1]["role"] == "user"
        assert "context summary text" in target._messages[1]["content"]
        assert target._messages[2]["role"] == "assistant"


# ── Config loader returns SwitchableProvider ──────────────────────────────


class TestConfigLoaderSwitchable:
    def test_local_llm_factory_returns_switchable_provider(self):
        """create_provider_factory() for 'local-llm' must return a SwitchableProvider."""
        from kiro_crew.config.loader import KiroCrewConfig

        cfg = KiroCrewConfig.__new__(KiroCrewConfig)
        agent = type("AgentConfig", (), {"provider": "local-llm"})()
        cfg.agent = agent

        factory = cfg.create_provider_factory()
        instance = factory(session_key="test-session")
        assert isinstance(instance, SwitchableProvider)

    def test_switchable_provider_exposes_local_as_default_active(self):
        p = SwitchableProvider()
        assert p._active_name == "local"
        assert isinstance(p._active, type(p._local))
