"""Tests for reaping the runtime of a session whose owning slot is gone.

Every live session carries its own MCP server processes, so the live process
count on a host is the number of sessions times the servers each one spawns. A
session that is finished but still in the map keeps paying that cost, which on a
4 vCPU / 16 GB host is enough to saturate it.

The idle sweep expires such a session on two axes. The clock is one. The other
is that the session's owning dashboard slot is gone, and the authority for that
is ``active_dashboard_slots``, published by ``_sync_dashboard_slots``, which
carries the *effective* key of every open slot: a ``dashboard:`` key, a
channel-born slot's channel key, or a linked slot's ``linked_session_key`` such
as ``taskrunner:<id>:chat:<tok>`` or ``cron:<job>``. Without the second axis a
session of any non-``dashboard:`` shape waits out the full timeout, default 60
minutes, holding its runtime and its MCP children.

The reap may never touch a session that is live or resumable, which is why
absence from the live set is not on its own sufficient: it is equally true of a
cron fire, a task step or a hook that is running right now and never had a tab.
A key must have appeared in a published live set before its later absence means
anything. The remaining hazards are pinned below: a slot that reopens between
the scan and the reset, sub-agent work that outlives its parent's turn, and a
probe that cannot answer.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from kiro_crew.config import KiroCrewConfig
from kiro_crew.session import SessionManager

# Far larger than any wall-clock this test spends, so every reap asserted below
# is on the terminated-owner axis and never on the idle clock.
NEVER_IDLE = 9999


@pytest.fixture
def cfg():
    c = KiroCrewConfig()
    c.session.timeout_secs = NEVER_IDLE
    return c


def _mock_provider_factory():
    def factory(session_key=None, agent=None, channel_id=None, **kwargs):
        m = AsyncMock()
        m.start = AsyncMock()
        m.shutdown = AsyncMock()
        m.context_usage_pct = lambda: 0.0
        m.has_active_turn = lambda: False
        return m

    return factory


async def _idle_session(mgr: SessionManager, key: str) -> None:
    """Create *key* and drop its permit, so only the sweep's rules decide."""
    await mgr.get_or_create(key)
    mgr.release(key)


class TestTerminatedSessionIsReaped:
    @pytest.mark.asyncio
    async def test_a_closed_tab_reaps_its_linked_session(self, cfg) -> None:
        """A non-dashboard key whose slot has closed is finished, so it goes."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})  # tab open
        mgr.set_active_dashboard_slots(set())  # tab closed

        await mgr._expire_idle(NEVER_IDLE)

        assert key not in mgr._sessions, "a finished session kept its runtime"
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_channel_born_slot_key_is_reaped_too(self, cfg) -> None:
        """A channel-born slot contributes its own key shape to the live set."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "cron:nightly"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())

        await mgr._expire_idle(NEVER_IDLE)

        assert key not in mgr._sessions
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_dashboard_keys_keep_their_existing_behaviour(self, cfg) -> None:
        """A ``dashboard:`` key is slot-owned by construction, published or not.

        Pinned so the record requirement that protects other key shapes cannot
        narrow this population, which needs no record.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        await _idle_session(mgr, "dashboard:tab1")
        mgr.set_active_dashboard_slots({"dashboard:tab2"})

        await mgr._expire_idle(NEVER_IDLE)

        assert "dashboard:tab1" not in mgr._sessions
        await mgr.close_all()


class TestLiveSessionsAreNeverReaped:
    @pytest.mark.asyncio
    async def test_a_session_that_never_had_a_slot_survives(self, cfg) -> None:
        """The whole safety of the change.

        A task step, a cron fire and a hook all run with no tab. They are absent
        from the live set for the entire time they are working, so reaping on
        absence alone would kill live work rather than finished work.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        step = "taskrunner:t1:task1"
        await _idle_session(mgr, step)
        mgr.set_active_dashboard_slots({"dashboard:tab1"})

        await mgr._expire_idle(NEVER_IDLE)

        assert step in mgr._sessions, "a live task step was reaped as terminated"
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_nothing_is_reaped_before_any_live_set_is_published(self, cfg) -> None:
        """A build with no dashboard has no authority, so it reaps on this axis never."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        await _idle_session(mgr, "dashboard:tab1")
        await _idle_session(mgr, "taskrunner:t1:chat:tok")

        assert mgr._cleanup_boundary().state.active_dashboard_slots is None
        await mgr._expire_idle(NEVER_IDLE)

        assert mgr.count == 2
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_an_open_tab_is_never_reaped(self, cfg) -> None:
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})

        await mgr._expire_idle(NEVER_IDLE)

        assert key in mgr._sessions
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_turn_in_flight_survives_a_closed_tab(self, cfg) -> None:
        """The permit is still held, so the session has work in flight."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await mgr.get_or_create(key)  # permit deliberately held
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())

        await mgr._expire_idle(NEVER_IDLE)

        assert key in mgr._sessions, "a live turn was reaped as terminated"
        mgr.release(key)
        await mgr.close_all()


class TestResumeBoundary:
    @pytest.mark.asyncio
    async def test_a_slot_reopening_mid_sweep_spares_its_session(self, cfg) -> None:
        """The scan's answer goes stale while the sweep awaits.

        The candidate list is built under the lock and released before any
        reset, so a tab can reopen in between. The second key is put back into
        the live set while the FIRST key's reset is in flight, which is the only
        window that exists; reaping it on the scan's answer would tear down a
        session the user had just resumed.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        first, second = "taskrunner:a:chat:tok", "taskrunner:b:chat:tok"
        await _idle_session(mgr, first)
        await _idle_session(mgr, second)
        mgr.set_active_dashboard_slots({first, second})
        mgr.set_active_dashboard_slots(set())  # both tabs closed

        async def reset_and_reopen(key, **kwargs):
            if key == first:
                mgr.set_active_dashboard_slots({second})  # user reopens the tab
            return True

        mgr.reset = AsyncMock(side_effect=reset_and_reopen)  # type: ignore[method-assign]

        await mgr._expire_idle(NEVER_IDLE)

        reaped = [c.args[0] for c in mgr.reset.await_args_list]
        assert reaped == [first], f"resumed session was reaped: {reaped}"
        await mgr.close_all()


class TestSubagentWorkOutlivingItsTab:
    @pytest.mark.asyncio
    async def test_attached_subagent_work_keeps_the_session(self, cfg) -> None:
        """Children run on the parent's runtime after the parent's turn ends.

        The busy semaphore cannot see them, so the probe is the only witness
        that a closed tab still has work behind it.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())
        mgr._has_attached_subagents = lambda k: True  # type: ignore[method-assign]

        await mgr._expire_idle(NEVER_IDLE)

        assert key in mgr._sessions, "reaped a parent whose sub-agent was still working"
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_an_awaitable_probe_answer_is_awaited(self, cfg) -> None:
        """The dashboard's probe reads the task store off the loop."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())

        async def slow_probe(k):
            await asyncio.sleep(0)
            return True

        mgr._has_attached_subagents = slow_probe  # type: ignore[method-assign]

        await mgr._expire_idle(NEVER_IDLE)

        assert key in mgr._sessions
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_probe_that_cannot_answer_keeps_the_session(self, cfg) -> None:
        """Fail-closed: an unanswerable probe is not the same as 'no children'."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:t1:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())

        def broken_probe(k):
            raise RuntimeError("task store unreadable")

        mgr._has_attached_subagents = broken_probe  # type: ignore[method-assign]

        await mgr._expire_idle(NEVER_IDLE)

        assert key in mgr._sessions, "a failed probe was read as 'no children'"
        await mgr.close_all()


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_two_sweeps_at_once_reap_the_finished_one_only(self, cfg) -> None:
        """The periodic loop and an on-demand sweep can overlap.

        Both must agree, neither may deadlock on the registry lock, and the live
        step session must survive both passes.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        finished, live = "taskrunner:a:chat:tok", "taskrunner:b:task1"
        await _idle_session(mgr, finished)
        await _idle_session(mgr, live)
        mgr.set_active_dashboard_slots({finished})
        mgr.set_active_dashboard_slots(set())

        await asyncio.wait_for(
            asyncio.gather(mgr._expire_idle(NEVER_IDLE), mgr._expire_idle(NEVER_IDLE)),
            timeout=10,
        )

        assert finished not in mgr._sessions
        assert live in mgr._sessions, "a session that never had a slot was reaped"
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_publish_during_a_sweep_does_not_lose_the_record(self, cfg) -> None:
        """A live-set publish lands while a sweep is in flight.

        The record is a union, so a publish that arrives mid-sweep may add keys
        but can never drop the one being judged.
        """
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        judged, other = "taskrunner:a:chat:tok", "taskrunner:b:chat:tok"
        await _idle_session(mgr, judged)
        await _idle_session(mgr, other)
        mgr.set_active_dashboard_slots({judged})
        mgr.set_active_dashboard_slots(set())

        async def publish_then_reset(key, **kwargs):
            mgr.set_active_dashboard_slots({other})
            return True

        mgr.reset = AsyncMock(side_effect=publish_then_reset)  # type: ignore[method-assign]

        await mgr._expire_idle(NEVER_IDLE)

        assert judged in mgr._cleanup_boundary().state.slot_owned_keys
        await mgr.close_all()


class TestRecordIsBounded:
    @pytest.mark.asyncio
    async def test_the_record_is_pruned_to_the_live_session_map(self, cfg) -> None:
        """Otherwise the record grows with every tab ever opened, for uptime."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        present = "taskrunner:a:chat:tok"
        await _idle_session(mgr, present)
        mgr.set_active_dashboard_slots({present, "dashboard:closed-long-ago"})

        await mgr._expire_idle(NEVER_IDLE)

        recorded = mgr._cleanup_boundary().state.slot_owned_keys
        assert recorded == {present}, f"record kept a key with no session: {recorded}"
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_reaped_key_leaves_the_record(self, cfg) -> None:
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        key = "taskrunner:a:chat:tok"
        await _idle_session(mgr, key)
        mgr.set_active_dashboard_slots({key})
        mgr.set_active_dashboard_slots(set())

        await mgr._expire_idle(NEVER_IDLE)
        await mgr._expire_idle(NEVER_IDLE)  # second pass sees the map without it

        assert key not in mgr._cleanup_boundary().state.slot_owned_keys
        await mgr.close_all()


class TestIdleAxisIsUnchanged:
    @pytest.mark.asyncio
    async def test_a_session_with_no_slot_still_expires_on_the_clock(self, cfg) -> None:
        """The terminated-owner axis is added, not substituted for the timer."""
        mgr = SessionManager(cfg, provider_factory=_mock_provider_factory())
        step = "taskrunner:t1:task1"
        await _idle_session(mgr, step)
        async with mgr._lock:
            mgr._sessions[step].last_used = time.monotonic() - 10_000

        await mgr._expire_idle(timeout_secs=1)

        assert step not in mgr._sessions, "the idle timer stopped working"
        await mgr.close_all()
