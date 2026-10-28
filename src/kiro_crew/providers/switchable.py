"""SwitchableProvider — wraps two LocalLLMProvider instances (local + cloud) and
handles the /switch command so users can move between them mid-session.

Usage:
    /switch cloud   — switch to the cloud OpenAI-compatible endpoint
    /switch local   — switch back to the local endpoint (LM Studio / Ollama)

Both endpoints use the same OpenAI-compatible protocol, just different URLs.
Configure the cloud endpoint with:

    CLOUD_LLM_BASE_URL=https://api.openai.com/v1
    CLOUD_LLM_API_KEY=sk-...
    CLOUD_LLM_MODEL=gpt-4o

This provider is returned by create_provider_factory() when
``agent.provider = "local-llm"``, so it is transparent: normal messages are
forwarded to the currently active backend without the user ever seeing the
wrapper.

Implements: https://github.com/obrutjack/jirou/issues/13
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    AcpEvent as LLMEvent,
)
from kiro_crew.providers.base import LLMProvider
from kiro_crew.providers.local_llm import LocalLLMProvider, _env

logger = logging.getLogger(__name__)

# ── Env-var keys for the cloud endpoint ──────────────────────────────────────

_CLOUD_BASE_URL_KEY = "CLOUD_LLM_BASE_URL"
_CLOUD_API_KEY_KEY = "CLOUD_LLM_API_KEY"
_CLOUD_MODEL_KEY = "CLOUD_LLM_MODEL"

# Maximum number of user/assistant turns to carry over on a switch.
_CONTEXT_CARRY_TURNS = 10


class SwitchableProvider(LLMProvider):
    """LLMProvider that supports /switch local and /switch cloud mid-session.

    Wraps two ``LocalLLMProvider`` instances — one for the local endpoint and
    one for the cloud endpoint — and proxies all other calls to whichever is
    currently active.

    The cloud provider is created lazily on the first ``/switch cloud`` so
    there is zero cost when the user never switches.
    """

    # Slash-command constants so tests can reference them without string literals.
    CMD_SWITCH_CLOUD = "/switch cloud"
    CMD_SWITCH_LOCAL = "/switch local"

    def __init__(
        self,
        session_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._session_key = session_key
        self._local: LocalLLMProvider = LocalLLMProvider(session_key=session_key)
        self._cloud: LocalLLMProvider | None = None
        self._active: LocalLLMProvider = self._local
        self._active_name: str = "local"

    # ── LLMProvider delegation ────────────────────────────────────────────────

    async def start(self) -> None:
        await self._active.start()

    async def shutdown(self) -> None:
        await self._active.shutdown()
        if self._cloud and self._cloud is not self._active:
            await self._cloud.shutdown()

    def context_usage_pct(self) -> float:
        return self._active.context_usage_pct()

    @property
    def context_provider_type(self) -> str:
        return self._active.context_provider_type

    @property
    def served_model(self) -> str:
        return self._active.served_model

    def context_window_tokens(self) -> int:
        return self._active.context_window_tokens()

    async def approve_tool(self, request_id: str | int, *, always: bool = False) -> None:
        await self._active.approve_tool(request_id, always=always)

    async def reject_tool(self, request_id: str | int) -> None:
        await self._active.reject_tool(request_id)

    # ── Core: stream ──────────────────────────────────────────────────────────

    async def stream(self, message: str) -> AsyncIterator[LLMEvent]:
        """Intercept /switch commands; forward everything else to active provider."""
        stripped = message.strip()
        lower = stripped.lower()

        if lower == self.CMD_SWITCH_CLOUD or lower.startswith(self.CMD_SWITCH_CLOUD + " "):
            async for event in self._handle_switch("cloud"):
                yield event
            return

        if lower == self.CMD_SWITCH_LOCAL or lower.startswith(self.CMD_SWITCH_LOCAL + " "):
            async for event in self._handle_switch("local"):
                yield event
            return

        # Normal message — delegate to the active provider.
        async for event in self._active.stream(message):
            yield event

    # ── Switch logic ──────────────────────────────────────────────────────────

    async def _handle_switch(self, target: str):
        """Yield events for a /switch command."""
        if target == self._active_name:
            label = "Local LLM (LM Studio / Ollama)" if target == "local" else "Cloud LLM"
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=f"Already using **{label}** — no switch needed.",
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")
            return

        if target == "cloud":
            async for event in self._switch_to_cloud():
                yield event
        else:
            async for event in self._switch_to_local():
                yield event

    async def _switch_to_cloud(self):
        cloud_url = _env(_CLOUD_BASE_URL_KEY, "")
        if not cloud_url:
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "⚠️ **Cloud LLM not configured**\n\n"
                    "Set these environment variables to enable `/switch cloud`:\n\n"
                    "```\n"
                    "CLOUD_LLM_BASE_URL=https://api.openai.com/v1\n"
                    "CLOUD_LLM_API_KEY=sk-...\n"
                    "CLOUD_LLM_MODEL=gpt-4o\n"
                    "```\n\n"
                    "Then restart Jirou and try again."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="error")
            return

        # Lazy-create the cloud provider on first use.
        if self._cloud is None:
            self._cloud = LocalLLMProvider(
                base_url=cloud_url,
                api_key=_env(_CLOUD_API_KEY_KEY, ""),
                model=_env(_CLOUD_MODEL_KEY, "gpt-4o"),
                session_key=self._session_key,
            )

        # Probe connectivity.
        await self._cloud.start()
        if self._cloud._start_error is not None:
            # Reuse the #12 connectivity notification.
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=self._cloud._start_error)
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="error")
            return

        # Transfer context.
        summary = self._build_context_summary()
        self._inject_context(self._cloud, summary)

        self._active = self._cloud
        self._active_name = "cloud"

        yield LLMEvent(
            kind=EVENT_TEXT_CHUNK,
            text=(
                f"✅ **Switched to Cloud LLM**\n\n"
                f"Endpoint: `{self._cloud._base_url}`  \n"
                f"Model: `{self._cloud._model}`\n\n"
                "Context from your previous session has been carried over."
            ),
        )
        yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

    async def _switch_to_local(self):
        # Re-probe the local endpoint (may have come up since the session started).
        new_local = LocalLLMProvider(session_key=self._session_key)
        await new_local.start()

        if new_local._start_error is not None:
            # Reuse the #12 connectivity notification.
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=new_local._start_error)
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="error")
            return

        # Transfer context.
        summary = self._build_context_summary()
        self._inject_context(new_local, summary)

        self._local = new_local
        self._active = new_local
        self._active_name = "local"

        yield LLMEvent(
            kind=EVENT_TEXT_CHUNK,
            text=(
                f"✅ **Switched to Local LLM**\n\n"
                f"Endpoint: `{new_local._base_url}`  \n"
                f"Model: `{new_local._model}`\n\n"
                "Context from your previous session has been carried over."
            ),
        )
        yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

    # ── Context helpers ───────────────────────────────────────────────────────

    def _build_context_summary(self) -> str:
        """Summarise the active provider's recent turns for handoff to the next provider.

        Keeps the last ``_CONTEXT_CARRY_TURNS`` user/assistant pairs.  Tool-call
        messages are skipped because they are session-specific and meaningless
        to a new provider that has no knowledge of those tool invocations.
        """
        messages = getattr(self._active, "_messages", [])
        turns = [
            m
            for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        recent = turns[-(_CONTEXT_CARRY_TURNS * 2) :]

        if not recent:
            return "[No conversation history to carry over.]"

        lines = [f"[Context from previous session — last {len(recent)} messages]\n"]
        for m in recent:
            role = m["role"].upper()
            content = str(m["content"])
            # Truncate long messages so the summary stays compact.
            if len(content) > 400:
                content = content[:400] + "…"
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _inject_context(self, provider: LocalLLMProvider, summary: str) -> None:
        """Prepend the context summary to a freshly started provider's history.

        The summary is injected as a user/assistant exchange so the new provider
        treats it as established background rather than a new question.
        """
        # provider.start() already set _messages = [system_prompt].
        # Append a synthetic context-handoff exchange right after the system message.
        provider._messages.append({"role": "user", "content": summary})
        provider._messages.append(
            {
                "role": "assistant",
                "content": (
                    "Understood. I have your conversation context from the previous session "
                    "and I am ready to continue."
                ),
            }
        )
