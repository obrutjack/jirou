"""Connects KiroCrew to any OpenAI-compatible API endpoint:
  - LM Studio  (default: http://localhost:1234/v1)
  - Ollama     (http://localhost:11434/v1)
  - Any BYOK cloud endpoint (OpenAI, OpenRouter, DeepSeek, Anthropic via proxy)

Configuration in ~/.kiro/crew/config.json:
    {
      "agent": {
        "provider": "local-llm"
      }
    }

Configuration in ~/.kiro/crew/.env:
    LOCAL_LLM_BASE_URL=http://localhost:1234/v1
    LOCAL_LLM_API_KEY=lm-studio
    LOCAL_LLM_MODEL=qwen/qwen3-14b

Key optimisation: Qwen3's /no_think system prompt directive reduces latency ~10x
(39s → 4s per tool call) with no meaningful accuracy loss on tool dispatch tasks.
Automatically enabled when the model name contains "qwen" (case-insensitive).
Override with LOCAL_LLM_NO_THINK=true/false env var.

This provider implements the LLMProvider ABC from providers/base.py.
Only stream() is the critical path; the rest use safe defaults from the ABC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIError

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    AcpEvent as LLMEvent,
)
from kiro_crew.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# ─── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio default
_DEFAULT_API_KEY  = "lm-studio"
_DEFAULT_MODEL    = "qwen/qwen3-14b"

# /no_think disables Qwen3's extended thinking mode.
# Benchmark: 96% tool-call success, 4.0s avg latency (vs ~39s with thinking on).
_NO_THINK_DIRECTIVE = "/no_think"

# Model families that support the /no_think directive.
_NO_THINK_MODEL_PREFIXES = ("qwen",)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _should_inject_no_think(model: str, no_think_env: str) -> bool:
    """Determine whether to inject /no_think based on model name.

    Priority:
      1. LOCAL_LLM_NO_THINK env var (explicit override always wins)
      2. Model name — inject only for model families known to support it

    This is more accurate than the previous URL-based check, which failed
    for BYOK cloud endpoints running Qwen and injected for non-Qwen local models.
    Fixes: https://github.com/obrutjack/jirou/issues/5
    """
    if no_think_env:
        return no_think_env.lower() not in ("0", "false", "no")
    model_lower = model.lower()
    return any(prefix in model_lower for prefix in _NO_THINK_MODEL_PREFIXES)


# ─── Provider ─────────────────────────────────────────────────────────────────


class LocalLLMProvider(LLMProvider):
    """LLMProvider backed by any OpenAI-compatible HTTP API.

    Drop-in replacement for AcpProvider when agent.provider = "local-llm".
    Streams text and tool-call events using the OpenAI streaming protocol and
    converts them to KiroCrew's internal LLMEvent format.

    Architecture note on tool results
    ----------------------------------
    KiroCrew's turn loop calls provider.stream(message) once per user turn and
    feeds tool results back as a new user message on the next stream() call.
    This means tool results arrive as ordinary user messages, not as the
    role:"tool" messages required by strict OpenAI endpoints.

    The correct fix requires tracing how the KiroCrew gateway feeds tool results
    to the provider and inserting a role:"tool" message at that point. Until
    that is resolved, multi-step tool loops will work on lenient endpoints
    (LM Studio) but may fail on strict endpoints (OpenAI, OpenRouter).
    Tracked in: https://github.com/obrutjack/jirou/issues/4
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        no_think: bool | None = None,
        session_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._base_url = base_url or _env("LOCAL_LLM_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key  = api_key  or _env("LOCAL_LLM_API_KEY",  _DEFAULT_API_KEY)
        self._model    = model    or _env("LOCAL_LLM_MODEL",     _DEFAULT_MODEL)
        self._session_key = session_key

        # /no_think: use explicit override if provided, otherwise detect from model name.
        if no_think is not None:
            self._no_think = no_think
        else:
            self._no_think = _should_inject_no_think(
                self._model, _env("LOCAL_LLM_NO_THINK", "")
            )

        self._client: AsyncOpenAI | None = None
        self._messages: list[dict] = []
        self._context_pct: float = 0.0
        self._pending_tool_approvals: set = set()

        logger.info(
            "[LocalLLM] provider initialised: model=%s base_url=%s no_think=%s",
            self._model, self._base_url, self._no_think,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise the AsyncOpenAI client and verify connectivity."""
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=300.0,
        )
        self._messages = [{"role": "system", "content": self._system_prompt()}]

        try:
            models = await self._client.models.list()
            names  = [m.id for m in models.data]
            logger.info("[LocalLLM] connected; available models: %s", names[:5])
        except APIConnectionError as e:
            logger.warning("[LocalLLM] connectivity check failed: %s", e)
            # Non-fatal: the user may start the server after KiroCrew boots.

    async def shutdown(self) -> None:
        """Close the AsyncOpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        self._messages = []

    # ── Context trimming ──────────────────────────────────────────────────────

    # KiroCrew injects context in this structure every turn:
    #
    #   [AGENT SYSTEM PROMPT]
    #   ...critical rules, MCP tool list, agent identity (~14K chars)...
    #   [END AGENT SYSTEM PROMPT]
    #
    #   [SESSION CONTEXT — background reference only...]
    #   ...memory, lessons, steering, history (~50-55K chars)...
    #   [END OF SESSION CONTEXT]
    #
    #   [CURRENT USER REQUEST -- respond to this]
    #   <actual user question>
    #
    # Strategy (two passes):
    #   Pass 1: keep agent system prompt, drop session context (~65K → ~14K)
    #   Pass 2: if agent prompt still exceeds MAX_AGENT_PROMPT_CHARS,
    #           keep its tail (most recent / most relevant parts)
    #
    # Override MAX_AGENT_PROMPT_CHARS via LOCAL_LLM_MAX_AGENT_PROMPT_CHARS env var.
    # Default 6000 chars ≈ 1500 tokens, leaving ~2500 tokens for conversation +
    # answer within a 4096-token LM Studio context window.

    _AGENT_PROMPT_START      = "[AGENT SYSTEM PROMPT]"
    _AGENT_PROMPT_END        = "[END AGENT SYSTEM PROMPT]"
    _SESSION_CTX_END         = "[END OF SESSION CONTEXT]"
    _USER_REQUEST_MARKER     = "[CURRENT USER REQUEST -- respond to this]"
    _DEFAULT_MAX_AGENT_CHARS = 6_000

    def _trim_kirocrew_message(self, raw: str) -> str:
        """Structure-aware trim of KiroCrew's per-turn context injection.

        KiroCrew prepends ~65K chars of agent context to every user turn.
        Without trimming this exceeds the 4096-token LM Studio context window.

        Pass 1 — structure: keep [AGENT SYSTEM PROMPT] block, drop [SESSION CONTEXT].
        Pass 2 — size cap: if the agent prompt is still too large, keep its tail.

        Falls back to last-N-chars if markers are not found (upstream change),
        with an explicit WARNING.
        """
        user_req_idx = raw.find(self._USER_REQUEST_MARKER)

        # ── Fast path: no KiroCrew marker ────────────────────────────────────
        if user_req_idx == -1:
            if len(raw) > 10_000:
                logger.warning(
                    "[LocalLLM] ⚠️ Large message (%d chars) without KiroCrew user-request "
                    "marker — passing through unchanged. If LM Studio rejects with "
                    "'context too long', the marker may have changed upstream.",
                    len(raw),
                )
            return raw

        user_request = raw[user_req_idx + len(self._USER_REQUEST_MARKER):].strip()
        original_chars = len(raw[:user_req_idx])

        # ── Pass 1: extract agent system prompt, drop session context ─────────
        agent_start = raw.find(self._AGENT_PROMPT_START)
        agent_end   = raw.find(self._AGENT_PROMPT_END)

        if agent_start != -1 and agent_end != -1:
            agent_block = raw[agent_start: agent_end + len(self._AGENT_PROMPT_END)].strip()
        else:
            # Fallback: if agent prompt markers are missing, use entire context
            # and let pass 2 cap it.
            agent_block = raw[:user_req_idx].strip()
            logger.warning(
                "[LocalLLM] ⚠️ Agent-prompt markers not found — using full context "
                "block for pass-2 capping. Marker text may have changed upstream.",
            )

        # ── Pass 2: cap agent prompt size ─────────────────────────────────────
        max_agent = int(_env("LOCAL_LLM_MAX_AGENT_PROMPT_CHARS",
                             str(self._DEFAULT_MAX_AGENT_CHARS)))
        if len(agent_block) > max_agent:
            # Keep the tail — critical rules are near the end of the agent prompt
            trimmed = agent_block[-max_agent:]
            nl = trimmed.find("\n")
            if nl != -1:
                trimmed = trimmed[nl + 1:]
            agent_block = trimmed

        logger.warning(
            "[LocalLLM] ✂️ Smart trim: %d → %d chars "
            "(kept agent system prompt tail, dropped session context). "
            "Adjust LOCAL_LLM_MAX_AGENT_PROMPT_CHARS if needed.",
            original_chars, len(agent_block),
        )
        return f"{agent_block}\n\n{user_request}"

    # ── Core: stream ──────────────────────────────────────────────────────────

    async def stream(self, message: str) -> AsyncIterator[LLMEvent]:
        """Send a user message and yield KiroCrew LLMEvents."""
        if not self._client:
            await self.start()

        # Trim KiroCrew's injected context before appending to history.
        # Without this, the 60K+ char injection exceeds the LM Studio context window.
        trimmed_message = self._trim_kirocrew_message(message)
        self._messages.append({"role": "user", "content": trimmed_message})
        messages_for_call = list(self._messages)

        # ── Context size diagnostic ───────────────────────────────────────────
        total_chars = sum(
            len(str(m.get("content") or "")) + len(str(m.get("role") or ""))
            for m in messages_for_call
        )
        estimated_tokens = total_chars // 4
        logger.warning(
            "[LocalLLM] 📊 Prompt stats: %d messages, ~%d chars, ~%d tokens (estimated) | model=%s",
            len(messages_for_call), total_chars, estimated_tokens, self._model,
        )
        for i, m in enumerate(messages_for_call):
            c = str(m.get("content") or "")
            logger.warning(
                "[LocalLLM]   msg[%d] role=%s chars=%d (first 80: %r)",
                i, m.get("role"), len(c), c[:80],
            )

        assistant_text = ""
        tool_calls_buf: dict[int, dict] = {}
        _first_token_time: float | None = None
        _t0 = time.monotonic()

        try:
            async with await self._client.chat.completions.create(
                model=self._model,
                messages=messages_for_call,
                stream=True,
                timeout=300.0,
            ) as stream:
                _stream_open_time = time.monotonic() - _t0
                logger.warning(
                    "[LocalLLM] ⏱ stream opened in %.2fs",
                    _stream_open_time,
                )
                async for chunk in stream:
                    if _first_token_time is None:
                        _first_token_time = time.monotonic() - _t0
                        logger.warning(
                            "[LocalLLM] ⏱ first token in %.2fs (prefill + TTFT)",
                            _first_token_time,
                        )
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    # ── Text chunk ────────────────────────────────────────────
                    if delta.content:
                        assistant_text += delta.content
                        yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=delta.content)

                    # ── Tool call accumulation ────────────────────────────────
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_buf:
                                tool_calls_buf[idx] = {
                                    "id":       "",
                                    "type":     "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            buf = tool_calls_buf[idx]
                            if tc_delta.id:
                                buf["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    buf["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    buf["function"]["arguments"] += tc_delta.function.arguments

                    # ── Stop reason ───────────────────────────────────────────
                    finish = chunk.choices[0].finish_reason if chunk.choices else None
                    if finish and finish not in ("", None):
                        break

        except APIConnectionError as e:
            logger.error("[LocalLLM] connection error during stream: %s", e)
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="error")
            return
        except APIError as e:
            logger.error("[LocalLLM] API error during stream: %s", e)
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="error")
            return

        # ── Emit buffered tool calls ──────────────────────────────────────────
        for idx in sorted(tool_calls_buf):
            tc = tool_calls_buf[idx]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                tool_call_id=tc["id"],
                tool_name=tc["function"]["name"],
                tool_input=args,
            )

        # Persist assistant turn for multi-turn context
        assistant_msg: dict = {"role": "assistant", "content": assistant_text or None}
        if tool_calls_buf:
            assistant_msg["tool_calls"] = list(tool_calls_buf.values())
        self._messages.append(assistant_msg)

        # Rough context usage estimate (characters as proxy for tokens)
        total_chars = sum(len(str(m)) for m in self._messages)
        self._context_pct = min(100.0, (total_chars / (128_000 * 4)) * 100)

        yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

    # ── Tool approval ─────────────────────────────────────────────────────────

    async def approve_tool(self, request_id: str | int, *, always: bool = False) -> None:
        """Approve a pending tool permission request.

        OpenAI-compatible APIs execute tools inline; this is a no-op placeholder
        kept for LLMProvider interface compliance. Real tool approval is handled
        by KiroCrew's gateway approval layer before the tool result is injected
        back into the conversation.
        """
        self._pending_tool_approvals.discard(request_id)

    async def reject_tool(self, request_id: str | int) -> None:
        """Reject a pending tool permission request."""
        self._pending_tool_approvals.discard(request_id)

    # ── Context ───────────────────────────────────────────────────────────────

    def context_usage_pct(self) -> float:
        """Return estimated context usage as a percentage."""
        return self._context_pct

    @property
    def context_provider_type(self) -> str:
        return "local-llm"

    @property
    def served_model(self) -> str:
        return self._model

    def context_window_tokens(self) -> int:
        """Report the model's context window so KiroCrew scales down context injection.

        KiroCrew's context budget is proportional to model_window / 1,000,000.
        For Qwen3 14B (32,768 token window):
          budget = 165,000 × 32,768 / 1,000,000 ≈ 5,400 chars
        vs the 1M-model default of 165,000 chars (~55K tokens).

        This is the source-side reduction mechanism. Whether KiroCrew actually
        reads this value to reduce injection is under investigation — see issue #6.

        Override via LOCAL_LLM_CONTEXT_WINDOW env var (default: 32768).
        Set to 0 to fall back to KiroCrew's 1M-model default (not recommended).
        """
        raw = _env("LOCAL_LLM_CONTEXT_WINDOW", "32768")
        try:
            return int(raw)
        except ValueError:
            return 32_768

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        """Build the system prompt for local LLM inference.

        Priority order:
          1. LOCAL_LLM_SYSTEM_PROMPT env var — user's custom prompt
          2. Default concise prompt — works well with Qwen3 and /no_think
        """
        custom = _env("LOCAL_LLM_SYSTEM_PROMPT", "")
        if custom:
            if self._no_think and not custom.startswith(_NO_THINK_DIRECTIVE):
                return f"{_NO_THINK_DIRECTIVE}\n{custom}"
            return custom

        base = (
            "You are a precise AI agent with access to tools. "
            "Call tools directly to complete the user's task. "
            "When the task is done, give a concise final answer. "
            "Do not narrate your plan before acting."
        )
        if self._no_think:
            return f"{_NO_THINK_DIRECTIVE}\n{base}"
        return base
