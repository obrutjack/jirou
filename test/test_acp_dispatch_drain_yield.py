"""The dispatch loop's queue drain must yield to the event loop.

``AcpSessionHandle._dispatch_events`` suspends on
``asyncio.wait_for(self._queue.get(), ...)``. On Python 3.12 that is not a
suspension point while the queue is non-empty: ``wait_for`` runs its awaitable
inline and ``Queue.get`` returns without awaiting when an item is present. Without a
cooperative yield, a backlogged session queue drains entirely inside ONE task
step and holds the loop for the whole drain -- every other session, the
dashboard websocket and the loop-stall heartbeat frozen behind it, up to the
watchdog's 25s hard exit (Taskei Mesh-3892).

These tests pin the wall-clock yield budget that bounds it: a drain lets other
callbacks run, the yield is budgeted rather than taken per frame, and a
cancellation landing on the yield loses no frame.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp import session_handle as sh
from kiro_crew.acp.session_handle import AcpSessionHandle
from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    METHOD_SESSION_UPDATE,
    JsonRpcMessage,
)

_REQ_ID = 11
_SID = "sDrain"


class _Runtime:
    """Minimal runtime double: the drain reads clocks and liveness only."""

    def __init__(self) -> None:
        self.pid = None
        self.is_alive = MagicMock(return_value=True)
        self.send_notification = AsyncMock()
        self.send_request = AsyncMock(return_value=_REQ_ID)
        self.supports_image_prompt = False
        self.acp_backend = ""
        self._last_activity = time.monotonic()

    def mark_turn_active(self, session_id: str, active: bool) -> None:
        return None


def _chunk(text: str) -> JsonRpcMessage:
    return JsonRpcMessage(
        method=METHOD_SESSION_UPDATE,
        params={
            "sessionId": _SID,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    )


def _make(backlog: int, *, terminal: bool = True) -> tuple[AcpSessionHandle, asyncio.Queue]:
    """A handle whose queue already holds ``backlog`` text frames."""
    queue: asyncio.Queue = asyncio.Queue()
    handle = AcpSessionHandle(_SID, queue, _Runtime())
    for i in range(backlog):
        queue.put_nowait(_chunk(f"c{i}"))
    if terminal:
        queue.put_nowait(JsonRpcMessage(id=_REQ_ID, result={"stopReason": "end_turn"}))
    return handle, queue


class _Ticker:
    """A self-rescheduling ``call_soon`` chain: one tick per event-loop pass.

    It measures loop passes, not time, so a host that drains 10k frames in a
    blink still reads zero ticks when the drain never yields.
    """

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = False
        asyncio.get_running_loop().call_soon(self._tick)

    def _tick(self) -> None:
        if self._stop:
            return
        self.ticks += 1
        asyncio.get_running_loop().call_soon(self._tick)

    def stop(self) -> None:
        self._stop = True


@pytest.mark.asyncio
async def test_backlog_drain_lets_the_event_loop_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10k-frame backlog drains WITHOUT freezing the loop for its duration.

    The budget is pinned to 0 so the assertion measures the yield's existence
    rather than the host's per-frame speed: with the guard removed the whole
    drain is one task step and the ticker never runs.
    """
    monkeypatch.setattr(sh, "_DRAIN_YIELD_AFTER_S", 0.0)
    handle, queue = _make(10_000)
    ticker = _Ticker()

    events = [ev async for ev in handle._dispatch_events(_REQ_ID, timeout=60.0)]
    ticker.stop()

    assert ticker.ticks > 0, "the drain held the event loop for its whole duration"
    # Every frame still surfaced, in queue order, and the turn ended normally.
    texts = [ev.text for ev in events if ev.kind == EVENT_TEXT_CHUNK]
    assert texts == [f"c{i}" for i in range(10_000)]
    assert events[-1].kind == EVENT_COMPLETE
    assert queue.empty()


@pytest.mark.asyncio
async def test_yield_is_budgeted_not_taken_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The yield is spent against a wall-clock budget, not once per frame.

    With a budget no drain can exhaust, the loop must keep the fast path: no
    sleep, so the ticker never gets a pass. This is what an unconditional
    ``sleep(0)`` after every frame would cost -- one loop pass per frame.
    """
    monkeypatch.setattr(sh, "_DRAIN_YIELD_AFTER_S", 3600.0)
    handle, _ = _make(500)
    ticker = _Ticker()

    events = [ev async for ev in handle._dispatch_events(_REQ_ID, timeout=60.0)]
    ticker.stop()

    assert ticker.ticks == 0
    assert len([ev for ev in events if ev.kind == EVENT_TEXT_CHUNK]) == 500


def test_budget_default_is_fifty_milliseconds() -> None:
    """The shipped budget: 50ms of loop-held time before a drain must yield."""
    assert sh._DRAIN_YIELD_AFTER_S == 0.05


@pytest.mark.asyncio
async def test_cancel_parked_on_the_yield_loses_no_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling mid-drain drops nothing: the yield holds no dequeued frame.

    Every frame is accounted for exactly once -- delivered to the consumer or
    still on the queue for the next reader. A yield placed straight after the
    dequeue instead would strand the frame in hand and this count would come up
    short.
    """
    monkeypatch.setattr(sh, "_DRAIN_YIELD_AFTER_S", 0.0)
    backlog = 200
    handle, queue = _make(backlog, terminal=False)
    seen: list[Any] = []

    async def consume() -> None:
        async for ev in handle._dispatch_events(_REQ_ID, timeout=60.0):
            if ev.kind == EVENT_TEXT_CHUNK:
                seen.append(ev.text)

    task = asyncio.create_task(consume())
    # One pass: the drain starts and parks on the new yield (with the budget at
    # 0 it is the loop's only suspension point while the queue is non-empty).
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen, "the drain was cancelled before it delivered anything"
    assert len(seen) < backlog, "the drain finished; this test never reached the yield"
    assert seen == [f"c{i}" for i in range(len(seen))]
    # Nothing vanished: consumed + still-queued reconstructs the whole backlog.
    assert len(seen) + queue.qsize() == backlog
