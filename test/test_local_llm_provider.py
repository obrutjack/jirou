"""Unit tests for the Jirou-specific LocalLLMProvider.

Coverage targets:
  - _should_inject_no_think()      — pure function, env-var & model-name logic
  - _trim_kirocrew_message()       — structure-aware context trim (core path)
  - _system_prompt()               — /no_think directive injection
  - context_window_tokens()        — env-var parsing with fallback
  - LocalLLMProvider.__init__()    — constructor priority (explicit > env > model)
  - LocalLLMProvider.stream()      — async streaming with mocked AsyncOpenAI
  - config.loader provider_factory — "local-llm" routing

These tests must run without a running LM Studio server.  All HTTP is mocked.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_CALL,
)
from kiro_crew.acp.types import AcpEvent as LLMEvent
from kiro_crew.providers.local_llm import (
    LocalLLMProvider,
    _should_inject_no_think,
)

# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_provider(**kwargs: Any) -> LocalLLMProvider:
    """Construct a LocalLLMProvider without hitting AsyncOpenAI.__init__."""
    return LocalLLMProvider(**kwargs)


async def _drain(ait) -> list[LLMEvent]:
    out = []
    async for event in ait:
        out.append(event)
    return out


# ─── _should_inject_no_think ──────────────────────────────────────────────────


class TestShouldInjectNoThink:
    """Pure-function logic: env var wins, else model name prefix."""

    def test_qwen_model_no_env_returns_true(self):
        assert _should_inject_no_think("qwen/qwen3-14b", "") is True

    def test_qwen_prefix_case_insensitive(self):
        assert _should_inject_no_think("Qwen3-14B", "") is True

    def test_non_qwen_model_no_env_returns_false(self):
        assert _should_inject_no_think("llama3", "") is False

    def test_deepseek_model_no_env_returns_false(self):
        assert _should_inject_no_think("deepseek-r1", "") is False

    def test_env_true_overrides_non_qwen_model(self):
        assert _should_inject_no_think("llama3", "true") is True

    def test_env_false_overrides_qwen_model(self):
        assert _should_inject_no_think("qwen/qwen3-14b", "false") is False

    def test_env_0_is_false(self):
        assert _should_inject_no_think("qwen", "0") is False

    def test_env_no_is_false(self):
        assert _should_inject_no_think("qwen", "no") is False

    def test_env_1_is_true(self):
        # "1" is not in the explicit-false set
        assert _should_inject_no_think("llama3", "1") is True

    def test_env_TRUE_uppercase_is_true(self):
        assert _should_inject_no_think("llama3", "TRUE") is True

    def test_env_FALSE_uppercase_is_false(self):
        assert _should_inject_no_think("qwen", "FALSE") is False


# ─── _trim_kirocrew_message ───────────────────────────────────────────────────


class TestTrimKiroCrewMessage:
    """Structure-aware trim: keep agent system prompt, drop session context."""

    # Markers used by KiroCrew (must match the constants in local_llm.py)
    _AGENT_START = "[AGENT SYSTEM PROMPT]"
    _AGENT_END = "[END AGENT SYSTEM PROMPT]"
    _CTX_END = "[END OF SESSION CONTEXT]"
    _USER_MARKER = "[CURRENT USER REQUEST -- respond to this]"

    def _build_full_message(
        self,
        agent_body: str = "agent rules here",
        session_ctx: str = "memory and history here",
        user_request: str = "What is 2+2?",
    ) -> str:
        return (
            f"{self._AGENT_START}\n{agent_body}\n{self._AGENT_END}\n\n"
            f"[SESSION CONTEXT]\n{session_ctx}\n{self._CTX_END}\n\n"
            f"{self._USER_MARKER}\n{user_request}"
        )

    def _provider(self) -> LocalLLMProvider:
        return _make_provider()

    def test_short_message_no_marker_passthrough(self):
        provider = self._provider()
        msg = "Hello, what is the weather?"
        assert provider._trim_kirocrew_message(msg) == msg

    def test_long_message_no_marker_passthrough(self, caplog):
        provider = self._provider()
        msg = "x" * 11_000  # > 10_000 threshold
        result = provider._trim_kirocrew_message(msg)
        assert result == msg
        assert "without KiroCrew user-request marker" in caplog.text

    def test_full_message_extracts_agent_block_and_user_request(self):
        provider = self._provider()
        raw = self._build_full_message(
            agent_body="CRITICAL RULES",
            session_ctx="HUGE HISTORY",
            user_request="Tell me a joke.",
        )
        result = provider._trim_kirocrew_message(raw)
        assert "CRITICAL RULES" in result
        assert "Tell me a joke." in result

    def test_session_context_body_dropped(self):
        provider = self._provider()
        raw = self._build_full_message(
            session_ctx="SECRET SESSION HISTORY 12345",
        )
        result = provider._trim_kirocrew_message(raw)
        assert "SECRET SESSION HISTORY 12345" not in result

    def test_user_request_extracted_exactly(self):
        provider = self._provider()
        raw = self._build_full_message(user_request="Explain black holes.")
        result = provider._trim_kirocrew_message(raw)
        assert result.endswith("Explain black holes.")

    def test_agent_block_over_max_trimmed_to_tail(self):
        """When agent block exceeds the max char limit, keep the tail.

        The implementation keeps agent_block[-max_chars:], so content near the
        END of the agent block survives.  Critical rules live at the tail of the
        KiroCrew agent prompt, which is why this direction was chosen.

        Uses an agent body large enough to exceed the DEFAULT 6000-char limit
        so the test is independent of env-var patching (xdist workers inherit
        a fresh subprocess environment; patch.dict on os.environ is unreliable
        across gw workers).
        """
        provider = self._provider()
        # Body is ~6100 chars so agent_block (~6150) exceeds the 6000-char default.
        # IMPORTANT TAIL CONTENT is placed at the END so it survives the [-6000:] slice.
        agent_body = "A" * 6100 + "\nIMPORTANT TAIL CONTENT"
        raw = self._build_full_message(agent_body=agent_body)
        result = provider._trim_kirocrew_message(raw)
        # Tail content should survive; bulk A padding at the front should be dropped
        assert "IMPORTANT TAIL CONTENT" in result
        assert "A" * 200 not in result  # bulk prefix trimmed (default max=6000 keeps only the tail)

    def test_missing_agent_markers_falls_back_to_full_context_block(self):
        """If agent prompt markers are absent, fall back gracefully.

        The fallback uses the full pre-user-request block as the agent context
        rather than raising or returning an empty string.  A warning is logged
        but we validate BEHAVIOR rather than the log text (caplog is unreliable
        across xdist worker processes).
        """
        provider = self._provider()
        # No AGENT SYSTEM PROMPT markers — just some context + user request
        raw = "some context without markers\n\n" f"{self._USER_MARKER}\nUser asked this."
        result = provider._trim_kirocrew_message(raw)
        assert "User asked this." in result
        # The full context block is used as the fallback agent prompt
        assert "some context without markers" in result

    def test_empty_user_request_handled(self):
        provider = self._provider()
        raw = self._build_full_message(user_request="")
        # Should not raise, should return something
        result = provider._trim_kirocrew_message(raw)
        assert isinstance(result, str)


# ─── _system_prompt ───────────────────────────────────────────────────────────


class TestSystemPrompt:
    """/no_think directive injection and custom prompt priority."""

    def test_no_think_true_prepends_directive(self):
        p = _make_provider(no_think=True)
        prompt = p._system_prompt()
        assert prompt.startswith("/no_think\n")

    def test_no_think_false_no_directive(self):
        p = _make_provider(no_think=False)
        prompt = p._system_prompt()
        assert "/no_think" not in prompt

    def test_custom_prompt_no_think_false_returned_as_is(self):
        custom = "You are a pirate assistant."
        with patch.dict(os.environ, {"LOCAL_LLM_SYSTEM_PROMPT": custom}):
            p = _make_provider(no_think=False)
            assert p._system_prompt() == custom

    def test_custom_prompt_no_think_true_prepends_directive(self):
        custom = "You are a pirate assistant."
        with patch.dict(os.environ, {"LOCAL_LLM_SYSTEM_PROMPT": custom}):
            p = _make_provider(no_think=True)
            prompt = p._system_prompt()
        assert prompt == f"/no_think\n{custom}"

    def test_custom_prompt_already_has_directive_not_doubled(self):
        """If the custom prompt already starts with /no_think, do not prepend again."""
        custom = "/no_think\nYou are a pirate assistant."
        with patch.dict(os.environ, {"LOCAL_LLM_SYSTEM_PROMPT": custom}):
            p = _make_provider(no_think=True)
            prompt = p._system_prompt()
        assert prompt.count("/no_think") == 1


# ─── context_window_tokens ───────────────────────────────────────────────────


class TestContextWindowTokens:
    """Env-var parsing with graceful fallback on bad input."""

    def test_default_is_32768(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_CONTEXT_WINDOW"}
        with patch.dict(os.environ, env, clear=True):
            p = _make_provider()
            assert p.context_window_tokens() == 32_768

    def test_env_var_overrides(self):
        with patch.dict(os.environ, {"LOCAL_LLM_CONTEXT_WINDOW": "4096"}):
            p = _make_provider()
            assert p.context_window_tokens() == 4096

    def test_invalid_env_falls_back_to_default(self):
        with patch.dict(os.environ, {"LOCAL_LLM_CONTEXT_WINDOW": "not-a-number"}):
            p = _make_provider()
            assert p.context_window_tokens() == 32_768


# ─── LocalLLMProvider.__init__ ────────────────────────────────────────────────


class TestProviderInit:
    """Constructor parameter priority: explicit > env > model-name detection."""

    def test_explicit_no_think_true_overrides_model(self):
        """no_think=True should win even for a non-Qwen model."""
        p = _make_provider(model="llama3", no_think=True)
        assert p._no_think is True

    def test_explicit_no_think_false_overrides_qwen_model(self):
        """no_think=False should win even for a Qwen model."""
        p = _make_provider(model="qwen/qwen3-14b", no_think=False)
        assert p._no_think is False

    def test_none_falls_through_to_model_detection_qwen(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_NO_THINK"}
        with patch.dict(os.environ, env, clear=True):
            p = _make_provider(model="qwen/qwen3-14b", no_think=None)
        assert p._no_think is True

    def test_none_falls_through_to_model_detection_llama(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_NO_THINK"}
        with patch.dict(os.environ, env, clear=True):
            p = _make_provider(model="llama3", no_think=None)
        assert p._no_think is False

    def test_default_base_url(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            p = _make_provider()
        assert p._base_url == "http://localhost:1234/v1"

    def test_default_model(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            p = _make_provider()
        assert p._model == "qwen/qwen3-14b"

    def test_context_provider_type(self):
        p = _make_provider()
        assert p.context_provider_type == "local-llm"

    def test_served_model_returns_configured_model(self):
        p = _make_provider(model="custom/my-model")
        assert p.served_model == "custom/my-model"


# ─── stream() ─────────────────────────────────────────────────────────────────


def _make_chunk(content: str | None = None, tool_calls=None, finish_reason=None):
    """Build a minimal mock OpenAI streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _make_tool_call_delta(idx: int, tc_id: str = "", name: str = "", args: str = ""):
    """Build a mock tool-call delta fragment."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = args
    tc = MagicMock()
    tc.index = idx
    tc.id = tc_id
    tc.function = fn
    return tc


@asynccontextmanager
async def _mock_stream(chunks):
    """Context manager that yields chunks like the real streaming API."""

    async def _gen():
        for c in chunks:
            yield c

    stream = _gen()
    yield stream


class TestStream:
    """Async streaming with mocked AsyncOpenAI — no real HTTP calls."""

    def _patched_provider(self, chunks) -> LocalLLMProvider:
        """Return a provider whose AsyncOpenAI client yields the given chunks."""
        p = _make_provider(no_think=False)

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_stream(chunks))
        p._client = mock_client
        # Pre-seed message history so start() is skipped
        p._messages = [{"role": "system", "content": "You are helpful."}]
        return p

    @pytest.mark.asyncio
    async def test_text_chunks_emitted(self):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]
        p = self._patched_provider(chunks)
        events = await _drain(p.stream("hi"))
        text_events = [e for e in events if e.kind == EVENT_TEXT_CHUNK]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello"
        assert text_events[1].text == " world"

    @pytest.mark.asyncio
    async def test_always_ends_with_complete_event(self):
        chunks = [_make_chunk(content="Done"), _make_chunk(finish_reason="stop")]
        p = self._patched_provider(chunks)
        events = await _drain(p.stream("hi"))
        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_call_event_emitted(self):
        tc_delta = _make_tool_call_delta(
            idx=0, tc_id="call_abc", name="read_file", args='{"path": "/tmp/x"}'
        )
        chunks = [
            _make_chunk(tool_calls=[tc_delta]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        p = self._patched_provider(chunks)
        events = await _drain(p.stream("read /tmp/x"))
        tool_events = [e for e in events if e.kind == EVENT_TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "read_file"
        assert tool_events[0].tool_input == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_tool_call_arguments_accumulated_across_chunks(self):
        """Arguments streamed in multiple chunks must be concatenated correctly."""
        tc1 = _make_tool_call_delta(idx=0, tc_id="call_1", name="bash", args='{"cmd":')
        tc2 = _make_tool_call_delta(idx=0, args=' "ls"}')
        chunks = [
            _make_chunk(tool_calls=[tc1]),
            _make_chunk(tool_calls=[tc2]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        p = self._patched_provider(chunks)
        events = await _drain(p.stream("list files"))
        tool_events = [e for e in events if e.kind == EVENT_TOOL_CALL]
        assert tool_events[0].tool_input == {"cmd": "ls"}

    @pytest.mark.asyncio
    async def test_api_connection_error_yields_error_complete(self):
        from openai import APIConnectionError

        p = _make_provider(no_think=False)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        p._client = mock_client
        p._messages = [{"role": "system", "content": "sys"}]

        events = await _drain(p.stream("trigger error"))
        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "error"

    @pytest.mark.asyncio
    async def test_api_error_yields_error_complete(self):
        from openai import APIError

        p = _make_provider(no_think=False)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIError(
                message="bad request",
                request=MagicMock(),
                body=None,
            )
        )
        p._client = mock_client
        p._messages = [{"role": "system", "content": "sys"}]

        events = await _drain(p.stream("trigger api error"))
        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "error"

    @pytest.mark.asyncio
    async def test_message_history_appended_after_stream(self):
        """Assistant turn must be appended to _messages for multi-turn context."""
        chunks = [_make_chunk(content="My answer"), _make_chunk(finish_reason="stop")]
        p = self._patched_provider(chunks)
        await _drain(p.stream("question"))
        roles = [m["role"] for m in p._messages]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_message_trimmed_before_sending(self):
        """_trim_kirocrew_message must be called on the raw user message."""
        chunks = [_make_chunk(finish_reason="stop")]
        p = self._patched_provider(chunks)

        with patch.object(p, "_trim_kirocrew_message", wraps=p._trim_kirocrew_message) as mock_trim:
            await _drain(p.stream("raw input"))
        mock_trim.assert_called_once_with("raw input")


# ─── approve_tool / reject_tool ───────────────────────────────────────────────


class TestToolApproval:
    """Interface compliance — no-op stubs must not raise."""

    @pytest.mark.asyncio
    async def test_approve_tool_does_not_raise(self):
        p = _make_provider()
        await p.approve_tool("req_1")
        await p.approve_tool("req_2", always=True)

    @pytest.mark.asyncio
    async def test_reject_tool_does_not_raise(self):
        p = _make_provider()
        await p.reject_tool("req_1")


# ─── config/loader.py — create_provider_factory routing ──────────────────────


class TestConfigProviderRouting:
    """create_provider_factory() must return a LocalLLMProvider factory for local-llm.

    KiroCrewConfig uses lazy imports (AcpProvider is imported inside the method),
    so we stub the config object rather than patching the module-level namespace.
    """

    def _config_with_provider(self, provider: str):
        """Return a KiroCrewConfig stub with only agent.provider set."""
        from kiro_crew.config.loader import KiroCrewConfig

        cfg = KiroCrewConfig.__new__(KiroCrewConfig)
        agent = MagicMock()
        agent.provider = provider
        cfg.agent = agent
        return cfg

    def test_local_llm_factory_returns_switchable_provider(self):
        """create_provider_factory() for 'local-llm' must return a SwitchableProvider.

        The factory was updated to return SwitchableProvider (which wraps
        LocalLLMProvider and adds /switch command support) as part of Issue #13.
        """
        from kiro_crew.providers.switchable import SwitchableProvider

        cfg = self._config_with_provider("local-llm")
        factory = cfg.create_provider_factory()
        instance = factory(session_key="test-session")
        assert isinstance(instance, SwitchableProvider)

    def test_local_llm_factory_passes_session_key(self):
        """The session_key is forwarded to the SwitchableProvider."""
        from kiro_crew.providers.switchable import SwitchableProvider

        cfg = self._config_with_provider("local-llm")
        factory = cfg.create_provider_factory()
        instance = factory(session_key="my-session")
        assert isinstance(instance, SwitchableProvider)
        assert instance._session_key == "my-session"

    def test_acp_path_does_not_return_local_llm_provider(self):
        """Ensure the fork did not accidentally replace the default ACP path.

        The fork's early-return guard (`if self.agent.provider == "local-llm"`)
        must NOT fire for provider="acp".  We verify this by inspecting the
        source branch logic rather than calling into the ACP factory (which
        requires a running kiro-cli process and many config attributes).
        """
        import inspect

        from kiro_crew.config.loader import KiroCrewConfig

        src = inspect.getsource(KiroCrewConfig.create_provider_factory)
        # The fork guard must be present and must be scoped to "local-llm" only
        assert 'self.agent.provider == "local-llm"' in src
        # The ACP path must still exist downstream of the guard
        assert "AcpProvider" in src


# ─── Connectivity notification (Issue #12) ───────────────────────────────────


class TestConnectivityNotification:
    """When Local LLM is unreachable, stream() emits a visible in-chat warning
    instead of silently failing.  Closes: https://github.com/obrutjack/jirou/issues/12
    """

    @pytest.mark.asyncio
    async def test_unreachable_emits_warning_text_chunk(self):
        """start() failure → stream() yields a visible warning, not silence."""
        from openai import APIConnectionError

        p = _make_provider()

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
            await p.start()

        assert p._start_error is not None, "start_error must be set after connection failure"

        # stream() must emit the warning without calling chat completions
        p._client = mock_client
        events = await _drain(p.stream("what is 2+2?"))

        text_events = [e for e in events if e.kind == EVENT_TEXT_CHUNK]
        assert len(text_events) >= 1, "Expected at least one TEXT_CHUNK with the warning"
        warning_text = text_events[0].text.lower()
        assert "unreachable" in warning_text or "not responding" in warning_text

    @pytest.mark.asyncio
    async def test_warning_ends_with_error_complete(self):
        """stream() yields EVENT_COMPLETE stop_reason='error' after the warning."""
        from openai import APIConnectionError

        p = _make_provider()
        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
            await p.start()

        p._client = mock_client
        events = await _drain(p.stream("test"))

        assert events[-1].kind == EVENT_COMPLETE
        assert events[-1].stop_reason == "error"

    @pytest.mark.asyncio
    async def test_warning_includes_endpoint_url(self):
        """The warning names the endpoint so the user knows which server is down."""
        from openai import APIConnectionError

        custom_url = "http://localhost:9999/v1"
        p = _make_provider(base_url=custom_url)

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
            await p.start()

        p._client = mock_client
        events = await _drain(p.stream("test"))

        text_events = [e for e in events if e.kind == EVENT_TEXT_CHUNK]
        assert custom_url in text_events[0].text

    @pytest.mark.asyncio
    async def test_chat_completions_not_called_when_unreachable(self):
        """stream() must not attempt the real LLM call when the server is down."""
        from openai import APIConnectionError

        p = _make_provider()
        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
            await p.start()

        p._client = mock_client
        await _drain(p.stream("test"))

        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_message_not_appended_when_unreachable(self):
        """Unsent user messages must not accumulate in _messages when server is down."""
        from openai import APIConnectionError

        p = _make_provider()
        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
            await p.start()

        p._client = mock_client
        initial_len = len(p._messages)
        await _drain(p.stream("message that was never sent"))

        # _messages must not grow: the unsent message must be discarded.
        assert len(p._messages) == initial_len

    @pytest.mark.asyncio
    async def test_session_recovers_automatically_when_server_comes_up(self):
        """If the server starts after the initial failure, the next stream() succeeds."""
        from openai import APIConnectionError

        p = _make_provider(no_think=False)
        call_count = 0

        async def probe_side_effect():
            nonlocal call_count
            call_count += 1
            # Calls: 1 = start() probe, 2 = first stream() re-probe (still down),
            #        3+ = second stream() re-probe (server came up)
            if call_count <= 2:
                raise APIConnectionError(request=MagicMock())
            return MagicMock(data=[])

        with patch("kiro_crew.providers.local_llm.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.models.list = AsyncMock(side_effect=probe_side_effect)
            mock_client.chat.completions.create = AsyncMock(
                return_value=_mock_stream(
                    [_make_chunk(content="Hi there"), _make_chunk(finish_reason="stop")]
                )
            )
            await p.start()

        assert p._start_error is not None

        p._client = mock_client

        # First stream: server is still down — shows warning
        events1 = await _drain(p.stream("first attempt"))
        text1 = " ".join(e.text for e in events1 if e.kind == EVENT_TEXT_CHUNK).lower()
        assert "unreachable" in text1 or "not responding" in text1

        # Second stream: probe succeeds (call_count=2), warning clears, real reply arrives
        events2 = await _drain(p.stream("second attempt"))
        assert p._start_error is None, "_start_error must be cleared after successful reconnect"
        text2 = " ".join(e.text for e in events2 if e.kind == EVENT_TEXT_CHUNK)
        assert "Hi there" in text2
