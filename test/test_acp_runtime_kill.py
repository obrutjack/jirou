"""Tests for AcpRuntime.kill() liveness verification.

The kill escalation swallows signal-delivery errors by design (racing a
normal exit is common), so kill() must verify the process actually died
before untracking its PID. A survivor left untracked would be invisible to
every sweep and leak until reboot.
"""

from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import MagicMock

import pytest

from kiro_crew.acp import runtime as rt


def _bare_runtime(pid: int = 54321) -> rt.AcpRuntime:
    """Construct an AcpRuntime with just the state kill() touches."""
    r = rt.AcpRuntime.__new__(rt.AcpRuntime)
    r._dead = False
    r._pending_requests = {}
    r._pending_init_notifications = deque()
    r._routed_requests = {}
    r._session_queues = {}
    r._stderr_lines = []
    r._pid = pid
    r._child_pids = {}
    r._reader_task = None
    r._stderr_task = None
    r._sandbox_cleanup = None
    r._process_instance = "inst-abc"
    r._start_time = "root-start"

    proc = MagicMock()
    proc.pid = pid
    proc.returncode = None

    async def _never_exits() -> None:
        await asyncio.sleep(3600)

    proc.wait = _never_exits
    r._process = proc
    return r


@pytest.fixture(autouse=True)
def _fast_kill_windows(monkeypatch):
    """Make the two escalation waits time out without waiting for a real clock.

    `kill()` only reaches the SIGKILL escalation and the liveness probe after both
    `wait_for`s expire. At 0.05s that depended on the scheduler resuming a coroutine
    inside 50ms, which a loaded runner (and Windows, ~15.6ms timer granularity) does
    not promise. Zero makes `wait_for` raise on its first check: same code path,
    reached deterministically with no sleeping.
    """
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_TERM_TIMEOUT", 0)
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_REAP_TIMEOUT", 0)


@pytest.mark.asyncio
async def test_kill_keeps_pid_tracked_when_process_survives(monkeypatch):
    """Signal delivery failures are swallowed upstream — a surviving PID must
    NOT be untracked, so the startup/periodic sweeps keep a handle on it."""
    r = _bare_runtime()
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: True)
    untrack = MagicMock()
    monkeypatch.setattr(rt, "_untrack_pid", untrack)
    monkeypatch.setattr(rt, "_untrack_session_pid", untrack)

    await r.kill()

    untrack.assert_not_called()
    assert r._process is None
    assert r._dead is True


@pytest.mark.asyncio
async def test_kill_untracks_only_the_descendants_that_died(monkeypatch):
    """A descendant that escaped the group kill keeps its entry.

    That entry is the only handle the periodic sweep and the next startup
    cleanup have on it; untracking a survivor is the leak the tracking exists
    to close. Pruning is by the descendant's own liveness, not the root's.
    """
    r = _bare_runtime()
    r._child_pids = {700: ("s700", b"agent-chat"), 800: ("s800", b"node")}
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    monkeypatch.setattr(rt, "_untrack_pid", lambda pid: None)
    monkeypatch.setattr(rt, "_untrack_session_pid", lambda pid: None)
    # 700 escaped the killpg (its own setsid); 800 went down with the group.
    monkeypatch.setattr(rt, "_pid_gone_or_unmanaged", lambda pid: pid == 800)
    untracked: list[dict] = []
    monkeypatch.setattr(rt, "_untrack_child_pids", lambda d, **k: untracked.append(d))

    await r.kill()

    assert [sorted(d) for d in untracked] == [[800]]
    assert r._child_pids == {}


@pytest.mark.asyncio
async def test_kill_prunes_descendants_even_when_the_root_survives(monkeypatch):
    """The root's fate says nothing about a child that left the process group."""
    r = _bare_runtime()
    r._child_pids = {900: ("s900", b"node")}
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: True)
    monkeypatch.setattr(rt, "_pid_gone_or_unmanaged", lambda pid: True)
    untracked: list[dict] = []
    monkeypatch.setattr(rt, "_untrack_child_pids", lambda d, **k: untracked.append(d))

    await r.kill()

    assert [sorted(d) for d in untracked] == [[900]]


@pytest.mark.asyncio
async def test_kill_signals_the_group_when_the_root_is_already_gone(monkeypatch):
    """The leak that survives descendant tracking: a root that died before any
    descendant was recorded.

    kill_process_tree is killpg(getpgid(root)); getpgid raises once the root has
    exited, and swallowing that leaves the launcher, agent and chat process in
    the group unsignalled. The group id is known without the root -- it was a
    session leader -- so the teardown must still reach it, and escalate, once a
    live member vouches for it.
    """
    r = _bare_runtime()

    # The autouse fixture zeroes the grace, and wait_for(..., 0) times out before
    # even a finished wait() is read; a dead root's wait() must be SEEN to return,
    # so give this test the real ordering with a small non-zero grace.
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_TERM_TIMEOUT", 1.0)
    # The group fallback is POSIX-shaped (process groups); the group function is
    # stubbed below, so the routing can be exercised on every host.
    monkeypatch.setattr(rt.platform_compat, "IS_POSIX", True)

    async def _already_exited() -> int:
        return -9  # a dead root's wait() returns at once

    r._process.wait = _already_exited
    r._process.returncode = -9  # reaped by asyncio's child watcher already

    def _root_gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", _root_gone)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    monkeypatch.setattr(rt, "_untrack_pid", lambda pid: None)
    monkeypatch.setattr(rt, "_untrack_session_pid", lambda pid: None)
    signalled: list[tuple[int, int, str, object]] = []
    vouched = {101: "s101", 102: "s102"}

    def _group(pgid, sig, instance, *, expected=None):
        signalled.append((pgid, sig, instance, expected))
        return dict(vouched)

    monkeypatch.setattr(rt, "_signal_orphaned_runtime_group", _group)
    slept: list[float] = []

    async def _sleep(secs):
        slept.append(secs)

    monkeypatch.setattr(rt.asyncio, "sleep", _sleep)

    await r.kill()

    # SIGTERM to the group, the same grace a live tree gets, then SIGKILL aimed
    # by the members the SIGTERM vouched -- never by the root's number alone.
    # Both passes carry the incarnation this process was spawned as -- read
    # before the kill cleared it -- so a fresh runtime on the recycled root pid
    # cannot vouch for this group.
    assert signalled == [
        (54321, rt.platform_compat.SIGTERM, "inst-abc", None),
        (54321, rt.platform_compat.SIGKILL, "inst-abc", vouched),
    ]
    assert slept == [rt.AcpRuntime._KILL_TERM_TIMEOUT]


@pytest.mark.asyncio
async def test_kill_never_resolves_the_group_from_a_reaped_root(monkeypatch):
    """asyncio reaps a dead root in the background, freeing its number.

    A fresh session leader on that number makes getpgid SUCCEED, so the tree
    kill would land on it. Once the root is reaped and its start id does not
    match, the tree kill is skipped and only the vouched path runs.
    """
    r = _bare_runtime()
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_TERM_TIMEOUT", 1.0)
    monkeypatch.setattr(rt.platform_compat, "IS_POSIX", True)

    async def _already_exited() -> int:
        return -9

    r._process.wait = _already_exited
    r._process.returncode = -9
    # The number now reads as a different process.
    monkeypatch.setattr(rt.platform_compat, "get_process_start_id", lambda pid: "someone-else")
    tree_kill = MagicMock()
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", tree_kill)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: True)
    monkeypatch.setattr(rt, "_signal_orphaned_runtime_group", lambda pgid, sig, inst, **k: {})

    await r.kill()

    tree_kill.assert_not_called()


@pytest.mark.asyncio
async def test_kill_uses_the_tree_kill_while_the_root_identity_holds(monkeypatch):
    """The live start id matching the recorded one is the proof the number is
    still ours; with it the group is resolved from the number as before."""
    r = _bare_runtime()
    monkeypatch.setattr(rt.platform_compat, "get_process_start_id", lambda pid: "root-start")
    tree_kill = MagicMock()
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", tree_kill)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    monkeypatch.setattr(rt, "_untrack_pid", lambda pid: None)
    monkeypatch.setattr(rt, "_untrack_session_pid", lambda pid: None)

    await r.kill()

    assert tree_kill.call_count >= 1


@pytest.mark.asyncio
async def test_kill_does_not_trust_an_unset_returncode(monkeypatch):
    """asyncio reaps in the background and propagates returncode a callback
    later, so an unset returncode does not prove the pid is still held."""
    r = _bare_runtime()  # returncode None
    monkeypatch.setattr(rt.platform_compat, "IS_POSIX", True)
    monkeypatch.setattr(rt.platform_compat, "get_process_start_id", lambda pid: "someone-else")
    tree_kill = MagicMock()
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", tree_kill)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: True)
    monkeypatch.setattr(rt, "_signal_orphaned_runtime_group", lambda pgid, sig, inst, **k: {})

    await r.kill()

    tree_kill.assert_not_called()


@pytest.mark.asyncio
async def test_kill_cancelled_inside_the_grace_still_escalates(monkeypatch):
    """A shutdown that cancels the teardown mid-grace still owes the SIGKILL.

    The members were vouched and SIGTERMed; ones that ignore SIGTERM would
    otherwise outlive the gateway. The escalation runs on the way out.
    """
    r = _bare_runtime()
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_TERM_TIMEOUT", 1.0)
    monkeypatch.setattr(rt.platform_compat, "IS_POSIX", True)

    async def _already_exited() -> int:
        return -9

    r._process.wait = _already_exited
    r._process.returncode = -9

    def _root_gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", _root_gone)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    signalled: list[int] = []
    vouched = {101: "s101"}

    def _group(pgid, sig, instance, *, expected=None):
        signalled.append(sig)
        return dict(vouched)

    monkeypatch.setattr(rt, "_signal_orphaned_runtime_group", _group)

    async def _cancelled_sleep(secs):
        raise asyncio.CancelledError

    monkeypatch.setattr(rt.asyncio, "sleep", _cancelled_sleep)

    with pytest.raises(asyncio.CancelledError):
        await r.kill()

    assert signalled == [rt.platform_compat.SIGTERM, rt.platform_compat.SIGKILL]


@pytest.mark.asyncio
async def test_kill_does_not_escalate_when_no_member_vouches(monkeypatch):
    """A reaped root whose group has nothing of ours left is simply gone.

    No vouching member means no signal was sent, so no grace is owed and the
    SIGKILL escalation must not run against a number that may now be a
    stranger's.
    """
    r = _bare_runtime()

    # The autouse fixture zeroes the grace, and wait_for(..., 0) times out before
    # even a finished wait() is read; a dead root's wait() must be SEEN to return,
    # so give this test the real ordering with a small non-zero grace.
    monkeypatch.setattr(rt.AcpRuntime, "_KILL_TERM_TIMEOUT", 1.0)
    # The group fallback is POSIX-shaped (process groups); the group function is
    # stubbed below, so the routing can be exercised on every host.
    monkeypatch.setattr(rt.platform_compat, "IS_POSIX", True)

    async def _already_exited() -> int:
        return -9

    r._process.wait = _already_exited
    r._process.returncode = -9

    def _root_gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", _root_gone)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    monkeypatch.setattr(rt, "_untrack_pid", lambda pid: None)
    monkeypatch.setattr(rt, "_untrack_session_pid", lambda pid: None)
    calls: list[int] = []
    monkeypatch.setattr(
        rt, "_signal_orphaned_runtime_group", lambda pgid, sig, inst, **k: calls.append(sig) or {}
    )
    slept: list[float] = []

    async def _sleep(secs):
        slept.append(secs)

    monkeypatch.setattr(rt.asyncio, "sleep", _sleep)

    await r.kill()

    assert calls == [rt.platform_compat.SIGTERM]
    assert slept == []


@pytest.mark.asyncio
async def test_kill_treats_a_denied_signal_as_final(monkeypatch):
    """Only a reaped root reaches the group fallback.

    An OSError that is not ProcessLookupError -- EPERM through a launcher
    wrapper -- says the root is THERE and we may not signal it; guessing at its
    group from the pid would be signalling something we were just refused.
    """
    r = _bare_runtime()

    def _denied(pid, sig):
        raise PermissionError

    # The root is still ours -- the tree kill is what gets refused.
    monkeypatch.setattr(rt.platform_compat, "get_process_start_id", lambda pid: "root-start")
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", _denied)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: True)
    group = MagicMock(return_value={101: "s101"})
    monkeypatch.setattr(rt, "_signal_orphaned_runtime_group", group)

    await r.kill()

    group.assert_not_called()


@pytest.mark.asyncio
async def test_kill_untracks_pid_when_process_died(monkeypatch):
    """The normal path: process is gone after escalation, PID is untracked."""
    r = _bare_runtime()
    monkeypatch.setattr(rt.platform_compat, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(rt.platform_compat, "pid_exists", lambda pid: False)
    untracked_pids: list[int] = []
    monkeypatch.setattr(rt, "_untrack_pid", untracked_pids.append)
    monkeypatch.setattr(rt, "_untrack_session_pid", untracked_pids.append)

    await r.kill()

    assert untracked_pids == [54321, 54321]
    assert r._process is None
