"""OpenAI-compatible provider — fork extension for KiroCrew.

Connects KiroCrew to any OpenAI-compatible API endpoint:
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
Controlled by LOCAL_LLM_NO_THINK env var (default: true for local endpoints).

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

_DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio default   # LM Studio
_DEFAULT_API_KEY  = "lm-studio"
_DEFAULT_MODEL    = "qwen/qwen3-14b"

# /no_think disables Qwen3's extended thinking mode.
# Benchmark: 96% tool-call success, 4.0s avg latency (vs ~39s with thinking on).
_NO_THINK_DIRECTIVE = "/no_think"


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ─── Provider ─────────────────────────────────────────────────────────────────


class LocalLLMProvider(LLMProvider):
    """LLMProvider backed by any OpenAI-compatible HTTP API.

    Drop-in replacement for AcpProvider when agent.provider = "local-llm".
    Streams text and tool-call events using the OpenAI streaming protocol and
    converts them to KiroCrew's internal LLMEvent format.
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

        # /no_think defaults to True for local endpoints (loopback), False for others.
        if no_think is not None:
            self._no_think = no_think
        else:
            _raw = _env("LOCAL_LLM_NO_THINK", "")
            if _raw:
                self._no_think = _raw.lower() not in ("0", "false", "no")
            else:
                self._no_think = "localhost" in self._base_url or "127.0.0.1" in self._base_url

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

    # ── Core: stream ──────────────────────────────────────────────────────────

    # Markers KiroCrew uses to wrap the user's actual request
    _USER_REQUEST_MARKER = "[CURRENT USER REQUEST -- respond to this]"
    _AGENT_PROMPT_END    = "[END AGENT SYSTEM PROMPT]"

    # Max chars to keep from the injected context block (before the user request).
    # ~2000 chars ≈ 500 tokens — enough for date/identity/critical rules,
    # small enough to keep prefill fast on local hardware.
    # Override with LOCAL_LLM_MAX_CONTEXT_CHARS env var.
    _DEFAULT_MAX_CONTEXT_CHARS = 2_000

    def _split_kirocrew_message(self, raw: str) -> tuple[str, str]:
        """Split KiroCrew's injected context from the actual user request.

        KiroCrew prepends ~60K chars of agent context to every user turn:
          [AGENT SYSTEM PROMPT] ... [END AGENT SYSTEM PROMPT]
          [SESSION CONTEXT] ... [END OF SESSION CONTEXT]
          [CURRENT USER REQUEST -- respond to this]
          <actual user question>

        Returns (trimmed_context, actual_user_request).
        If the marker is not found, returns ("", raw) — no trimming.
        """
        marker = self._USER_REQUEST_MARKER
        idx = raw.find(marker)
        if idx == -1:
            return "", raw  # not a KiroCrew-format message, pass through unchanged

        context_block = raw[:idx].strip()
        user_request  = raw[idx + len(marker):].strip()

        # Trim the context block to the configured max
        max_chars = int(_env("LOCAL_LLM_MAX_CONTEXT_CHARS",
                             str(self._DEFAULT_MAX_CONTEXT_CHARS)))
        if len(context_block) > max_chars:
            # Keep a tail — the most recent/relevant parts are usually at the end
            trimmed = context_block[-max_chars:]
            # Find first newline to avoid splitting mid-line
            nl = trimmed.find("\n")
            if nl != -1:
                trimmed = trimmed[nl + 1:]
            context_block = f"[context trimmed to last {max_chars} chars]\n{trimmed}"

        return context_block, user_request

    async def stream(self, message: str) -> AsyncIterator[LLMEvent]:
        """Send a user message and yield KiroCrew LLMEvents."""
        if not self._client:
            await self.start()

        # ── Extract actual user request from KiroCrew's injected context ─────
        # KiroCrew bundles ~60K chars of agent context into the user message.
        # We split it: trimmed context → appended to system message,
        # actual question → user message. This keeps prefill small.
        context_block, actual_request = self._split_kirocrew_message(message)

        if context_block:
            # Prepend trimmed context to the system message for this turn
            sys_msg = self._messages[0] if self._messages else None
            if sys_msg and sys_msg.get("role") == "system":
                augmented_system = (
                    sys_msg["content"]
                    + f"\n\n[Injected context]\n{context_block}"
                )
                # Replace system message for this call only (don't persist)
                messages_for_call = [
                    {"role": "system", "content": augmented_system},
                    *self._messages[1:],
                    {"role": "user", "content": actual_request},
                ]
            else:
                messages_for_call = [
                    *self._messages,
                    {"role": "user", "content": actual_request},
                ]
        else:
            messages_for_call = [
                *self._messages,
                {"role": "user", "content": message},
            ]

        self._messages.append({"role": "user", "content": actual_request or message})

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
        # Assume 128K token context window (conservative for local models)
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
        """Report the model's context window to KiroCrew so it scales down context injection.

        KiroCrew's context budget is proportional to model_window / 1,000,000.
        For Qwen3 14B (32,768 token window):
          budget = 165,000 × 32,768 / 1,000,000 ≈ 5,400 chars
        vs the 1M-model default of 165,000 chars (~55K tokens).

        Override via LOCAL_LLM_CONTEXT_WINDOW env var (default: 32768).
        Set to 0 to fall back to KiroCrew's 1M-model default (not recommended for local).
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
          1. LOCAL_LLM_SYSTEM_PROMPT env var — user's custom prompt (shortest path)
          2. Default concise prompt — works well with Qwen3 and /no_think

        The KiroCrew gateway injects its own large system prompt (~14K tokens)
        on top of this via its session context layer.  That injected block is
        what causes slow prefill on local hardware.

        To bypass that overhead, set LOCAL_LLM_SYSTEM_PROMPT to a short
        instruction -- this provider's system message is the ONLY message sent
        before the user turn, not the gateway's context block.

        Note: the gateway context block is injected by KiroCrew's session layer
        separately; this method only controls the provider's own system message.
        """
        # Allow full override via env var — useful for minimising prefill cost
        custom = _env("LOCAL_LLM_SYSTEM_PROMPT", "")
        if custom:
            if self._no_think and not custom.startswith(_NO_THINK_DIRECTIVE):
                return f"{_NO_THINK_DIRECTIVE}\n{custom}"
            return custom

        # Default: concise but capable
        base = (
            "You are a precise AI agent with access to tools. "
            "Call tools directly to complete the user's task. "
            "When the task is done, give a concise final answer. "
            "Do not narrate your plan before acting."
        )
        if self._no_think:
            return f"{_NO_THINK_DIRECTIVE}\n{base}"
        return base

    def inject_tool_result(self, tool_call_id: str, result: str) -> None:
        """Append a tool result to the conversation history.

        Called by KiroCrew's session layer after executing a tool call that
        was yielded as an EVENT_TOOL_CALL event.
        """
        self._messages.append({
            "role":         "tool",
            "tool_call_id": tool_call_id,
            "content":      result,
        })
