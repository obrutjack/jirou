"""Supervises ``playwright-cli show``, the CLI's own dashboard, over loopback.

``show --port`` is a blocking HTTP server, so it is a long-lived supervised
child with its own lifecycle, not a call that returns a result. The dashboard it
serves carries the live viewport, the tab bar, and **full remote mouse and
keyboard input** into a browser that holds the operator's logged-in sessions.

Three properties are load-bearing; each was established by running the CLI, and
getting any of them wrong presents as a broken feature rather than as an error:

1. **Bind explicitly to ``127.0.0.1``.** The default listener is IPv6-only, so
   ``http://127.0.0.1:<port>/`` is unreachable and an iframe pointed there gets
   a connection failure while the server is running fine.
2. **Health is "any HTTP response", never "200".** ``/`` answers ``302``.
3. **Never ``--host 0.0.0.0``.** That would publish an interactive
   remote-input browser view, holding live logins, to the whole network. This
   module takes no host parameter at all, so there is no argument through which
   a caller could ask for a non-loopback bind.

The port is chosen by bind-probe rather than hardcoded: a fixed port collides
with whatever else the operator runs, and the collision would surface as an
unexplained dead panel. An operator who NEEDS predictability — the dashboard
viewed through an SSH tunnel that forwards a fixed set of ports — can pin the
public port via ``dashboard.browser_view_port``. The pin is never handed to
the child: this module claims the pinned port itself with a bound listener it
keeps holding (an atomic ownership proof, so the deterministic, operator-named
port is race-free) and relays byte-for-byte to the child's own ephemeral port.
The child's OS-assigned port keeps the same advisory bind window as the
unpinned path — unpredictable and loopback-local. Both the relay and the
child bind loopback only.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import logging
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable
from urllib.parse import urlsplit

from kiro_crew import platform_compat
from kiro_crew.browser_cli.install import cli_command, cli_env, cli_path
from kiro_crew.browser_cli.launch import ui_socket_env

logger = logging.getLogger(__name__)

# Loopback IPv4, as a constant rather than a parameter. See property 3 above.
LOOPBACK_HOST = "127.0.0.1"

# The server binds, starts Node, and initializes before it answers, so the
# readiness gate is a poll rather than a single probe.
_STARTUP_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 0.25
_HEALTH_TIMEOUT_S = 2.0
_TERMINATE_GRACE_S = 5.0

# The child proof is expected in its first few lines. Bound every dimension so
# a malformed or chatty stdout stream cannot keep a daemon reader alive forever
# or grow its pending line without limit. Hosts with working PID attribution do
# not depend on this fallback; an exhausted budget fails closed elsewhere.
_BINDING_PROOF_TIMEOUT_S = _STARTUP_TIMEOUT_S
_BINDING_PROOF_MAX_LINES = 256
_BINDING_PROOF_MAX_BYTES = 64 * 1024
_BINDING_PROOF_READ_SIZE = 4096
_BINDING_PROOF_POLL_S = 0.01
_BINDING_URL_RE = re.compile(r"https?://(?:\[[^\]\s]+\]|[^\s/:]+):\d+(?:/[^\s]*)?", re.IGNORECASE)

# Concurrent connections the pinned-port relay will carry. The panel needs a
# handful (page, assets, one screencast/input WebSocket per session view);
# 32 leaves generous headroom while bounding the threads and sockets a local
# process can force the relay to hold — a local-DoS bound, not an auth control
# (the view server itself is equally reachable by any local process).
_RELAY_MAX_CONNS = 32


@dataclass(frozen=True)
class ShowInfo:
    """Where the running dashboard is reachable."""

    url: str
    port: int


@dataclass
class _BindingProof:
    """Fresh child binding reports, each valid for exactly one ownership check."""

    port: int
    reported: threading.Event
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self) -> None:
        """Publish one report token from the child-specific stdout pipe."""
        with self._lock:
            self.reported.set()

    def consume(self) -> bool:
        """Consume the current report once; stale checks cannot reuse it."""
        with self._lock:
            if not self.reported.is_set():
                return False
            self.reported.clear()
            return True

    def invalidate(self) -> None:
        """Drop any unconsumed report after a reader limit or stream failure."""
        with self._lock:
            self.reported.clear()


_lock = threading.Lock()
_proc: subprocess.Popen[bytes] | None = None
_info: ShowInfo | None = None
_relay: "_Relay | None" = None
#: The child's OWN listening port, which is what ownership is proved against.
#: Distinct from ``_info.port``: on the pinned path that is the operator's port,
#: served by our in-process relay, so it proves nothing about the child.
_child_port: int | None = None
# Why the last start attempt failed, surfaced through ``status()``. With an
# ephemeral port a failed bind was a near-impossible edge; with a pinned port
# "already in use" becomes the most likely operator misconfiguration, and a
# silent ``stopped`` presents as a broken panel rather than as an error.
_last_reason: str | None = None


def _free_port() -> int:
    """An OS-assigned ephemeral loopback port.

    Binding port 0 lets the kernel pick one that is free right now. This is
    advisory: the socket is closed before the child binds it, so there is a
    TOCTOU window in which a local process can take the number instead.

    The window cannot be closed here. Closing it would mean handing the child a
    socket we already hold, and ``playwright-cli show`` takes a port NUMBER, not
    an inherited descriptor — so there is no bind-before-release to perform. What
    contains it is the readiness gate: it requires either PID attribution to the
    process tree we spawned or a recognized post-bind loopback URL reported on
    that child's private stdout pipe. Losing the race is therefore reported as a
    failed start instead of adopting the winner.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return int(sock.getsockname()[1])


def _claim_listener(port: int) -> socket.socket | None:
    """Atomically claim *port* with a bound, listening socket, or ``None``.

    Split out of :class:`_Relay` so :func:`ensure_running` can claim the pin
    BEFORE choosing the child's ephemeral port: while the pin is bound,
    ``_free_port()`` structurally cannot hand the same number back, so the
    "child port equals the pin" collision cannot arise by construction.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if platform_compat.IS_POSIX:
        # Allow rebinding through TIME_WAIT after a restart. POSIX-only:
        # on Windows SO_REUSEADDR permits stealing an ACTIVE listener,
        # which is the exact hole the held-listener design exists to close.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        # Windows: merely NOT setting SO_REUSEADDR is not enough — another
        # local process can still bind the same port by setting SO_REUSEADDR
        # on ITS socket. SO_EXCLUSIVEADDRUSE is the opt-in that makes our
        # bind exclusive, so the ownership proof holds on Windows too.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        listener.bind((LOOPBACK_HOST, port))
        listener.listen(16)
    except OSError:
        with contextlib.suppress(Exception):
            listener.close()
        return None
    return listener


class _Relay:
    """Holds a pinned loopback port and pumps bytes to the child's real port.

    The pin cannot be handed to the child directly without a race: any probe
    closes its socket before the child binds, and in that window a local
    squatter can take the port — after which :func:`_healthy` would accept the
    squatter's response and the panel would frame arbitrary content WITH remote
    input forwarding. Holding the bound listener ourselves is the only atomic
    ownership proof: ``bind()`` either succeeds (the port is ours until we
    close it) or raises (occupied). This makes the DETERMINISTIC, operator-named
    port race-free; the child then binds an OS-assigned ephemeral port exactly
    as on the unpinned path, whose probe-to-bind window is contained a different
    way -- :func:`_port_owner` proves the responder belongs to the tree we
    spawned before the readiness gate adopts it, so a squatter that wins the
    window is refused rather than framed. This relay forwards
    each accepted connection byte-for-byte, which carries HTTP and WebSocket
    traffic alike. Both sockets stay loopback-only.
    """

    def __init__(self, listener: socket.socket, target_port: int) -> None:
        self._listener = listener
        self._target_port = target_port
        self._closed = threading.Event()
        # Live sockets, guarded by _conn_lock: bounds what a local process can
        # force us to hold, and lets close() tear down in-flight connections
        # instead of leaving them to die with their daemon threads.
        self._conn_lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self._thread = threading.Thread(
            target=self._accept_loop, name="browser-view-relay", daemon=True
        )
        self._thread.start()

    @classmethod
    def open(cls, port: int, target_port: int) -> "_Relay | None":
        """Claim *port* and relay it to *target_port*, or ``None`` if taken."""
        listener = _claim_listener(port)
        if listener is None:
            return None
        return cls(listener, target_port)

    @classmethod
    def from_listener(cls, listener: socket.socket, target_port: int) -> "_Relay":
        """Start relaying on an already-claimed listener (see _claim_listener)."""
        return cls(listener, target_port)

    def close(self) -> None:
        """Stop accepting and wait for the accept thread to exit.

        ``shutdown`` before ``close`` is what reliably unblocks a thread
        parked in ``accept()`` — closing the descriptor alone is not
        guaranteed to wake it on every platform. The bounded ``join`` makes
        teardown deterministic instead of leaving a daemon thread to die on
        its own schedule (a test-visible side effect).
        """
        self._closed.set()
        with contextlib.suppress(Exception):
            self._listener.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(Exception):
            self._listener.close()
        self._thread.join(timeout=_TERMINATE_GRACE_S)
        # Tear down in-flight connections. shutdown() before close(), same as
        # the listener above: the pump threads hold these sockets, and close()
        # alone does not reliably interrupt them or deliver the FIN while
        # another thread is blocked in recv() — shutdown() does both.
        with self._conn_lock:
            conns = list(self._conns)
            self._conns.clear()
        for sock in conns:
            with contextlib.suppress(Exception):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(Exception):
                sock.close()

    def _track(self, sock: socket.socket) -> bool:
        """Register a live socket; ``False`` when the cap refuses it."""
        with self._conn_lock:
            if len(self._conns) >= _RELAY_MAX_CONNS:
                return False
            self._conns.add(sock)
            return True

    def _untrack(self, sock: socket.socket) -> None:
        with self._conn_lock:
            self._conns.discard(sock)

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                client, _addr = self._listener.accept()
            except OSError:
                return  # listener closed
            if not self._track(client):
                # At capacity: refuse instead of queueing unbounded threads. A
                # local flooder is bounded; the panel's handful of connections
                # never gets near the cap.
                with contextlib.suppress(Exception):
                    client.close()
                continue
            threading.Thread(
                target=self._serve, args=(client,), name="browser-view-relay-conn", daemon=True
            ).start()

    def _serve(self, client: socket.socket) -> None:
        try:
            upstream = socket.create_connection(
                (LOOPBACK_HOST, self._target_port), timeout=_HEALTH_TIMEOUT_S
            )
        except OSError:
            self._untrack(client)
            with contextlib.suppress(Exception):
                client.close()
            return
        upstream.settimeout(None)
        # Release the cap slot only when BOTH directions have terminated. A
        # half-closed connection (one pump exited, the other still parked in
        # recv on a peer that stays silent) must keep holding its slot:
        # freeing it on the first exit would let a local flooder accumulate
        # live pump threads beyond the accounting bound. Teardown still
        # reaches a lingering pump: a client-blocked pump is unblocked by
        # close() shutting down the tracked client socket, and an
        # upstream-blocked pump by the supervised child being reaped, which
        # accompanies every relay teardown path in ensure_running/stop.
        remaining = [2]
        remaining_lock = threading.Lock()

        def on_done() -> None:
            with remaining_lock:
                remaining[0] -= 1
                finished = remaining[0] == 0
            if finished:
                self._untrack(client)

        a = threading.Thread(target=_pump, args=(client, upstream, on_done), daemon=True)
        b = threading.Thread(target=_pump, args=(upstream, client, on_done), daemon=True)
        a.start()
        b.start()


def _pump(
    src: socket.socket, dst: socket.socket, on_done: "Callable[[], None] | None" = None
) -> None:
    """Copy bytes from *src* to *dst* until either side closes."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(Exception):
            dst.shutdown(socket.SHUT_WR)
        with contextlib.suppress(Exception):
            src.close()
        if on_done is not None:
            on_done()


def _healthy(port: int) -> bool:
    """Whether the dashboard answers HTTP on *port*.

    ANY status line counts, including the ``302`` that ``/`` actually returns.
    Only a transport-level failure (nothing listening, hang, reset) is unhealthy
    — the question is whether an HTTP server is there, not what it thinks of the
    request.
    """
    conn = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=_HEALTH_TIMEOUT_S)
    try:
        conn.request("GET", "/")
        conn.getresponse()
        return True
    except (OSError, http.client.HTTPException):
        return False
    finally:
        with contextlib.suppress(Exception):
            conn.close()


#: Proven: a process in the tree we spawned holds the port.
_OWNER_CHILD = "child"
#: Treated as foreign. Either a functional lookup could not attribute the
#: listener to us, or it proved a third party owns the port.
_OWNER_FOREIGN = "foreign"
#: The global port-to-PID lookup is absent or blind. Adoption still requires a
#: per-process ownership result or, when none is available, a startup report.
_OWNER_UNPROVEN = "unproven"


def _listener_lookup_functional() -> bool | None:
    """Whether the port-to-PID lookup can attribute a known local listener.

    A resolvable ``lsof`` or ``netstat`` can still be unusable from the current
    process namespace. Bind a real loopback control listener owned by this
    process and ask the same platform helper used for the target port. ``None``
    means the control listener could not be created or its lookup did not
    complete. Neither is evidence that the tool is blind, so the caller stays
    fail-closed.
    """
    listener = _claim_listener(0)
    if listener is None:
        return None
    try:
        control_port = int(listener.getsockname()[1])
        listeners, completed = platform_compat.probe_port_listeners(control_port)
        if not completed:
            return None
        owners = platform_compat.loopback_owner_pids(listeners)
        return os.getpid() in owners
    finally:
        with contextlib.suppress(Exception):
            listener.close()


def _port_owner(port: int, proc: subprocess.Popen[bytes] | None) -> str:
    """Who holds *port*: our child's tree, foreign, or undecidable on this host.

    :func:`_healthy` answers "is an HTTP server there", which is reachability,
    not identity. That is the whole gap: ``_free_port`` releases its probe socket
    before the child binds, so a local process can take the number in between,
    and a bare health probe then reports the squatter as our server -- after which
    the panel frames arbitrary content with input forwarding attached.

    A lookup that can attribute a known local listener but cannot attribute the
    target is FOREIGN, not undecidable. The caller only asks after a successful
    ``127.0.0.1`` probe, so something is listening. If a functional lookup cannot
    show that it belongs to our tree, "not ours" is the safe reading. This covers
    a squatter owned by another user and invisible on the target port.

    A missing lookup tool or a completed control lookup that cannot attribute a
    listener this process just bound yields UNPROVEN. Both establish a host
    capability failure rather than evidence about this start. UNPROVEN is not an
    adoption decision: the caller must obtain a separate positive proof from the
    child. The self-test runs only after the target lookup returns empty. A
    functional tool that sees the control but not the target still refuses the
    target as foreign, as does a control lookup that times out, cannot execute, or
    otherwise does not complete.
    """
    if proc is None:
        return _OWNER_FOREIGN
    if not platform_compat.listening_pid_tool_available():
        return _OWNER_UNPROVEN
    listeners = platform_compat.find_port_listeners(port)
    if not listeners:
        if _listener_lookup_functional() is False:
            return _OWNER_UNPROVEN
        return _OWNER_FOREIGN
    owners = platform_compat.loopback_owner_pids(listeners)
    if not owners:
        return _OWNER_FOREIGN
    # The listener is often a DESCENDANT: the CLI spawns Node and helper
    # processes, which is why _reap signals the whole tree rather than the direct
    # child. Enumerate the tree we own and accept any member of it.
    ours = {proc.pid, *platform_compat.process_descendants(proc.pid)}
    if any(pid in ours for pid in owners):
        return _OWNER_CHILD
    return _OWNER_FOREIGN


def _binding_reported_on_line(line: bytes, port: int) -> bool:
    """Whether trusted child stdout reports the loopback address we assigned.

    The wording around Playwright's listener URL has changed between CLI
    releases. Require a listener marker, then parse each URL by scheme, host,
    and port. A path, prefix, or punctuation change cannot break the proof, but
    an unrelated URL, host, protocol, or port cannot satisfy it.
    """
    text = line.decode("utf-8", errors="replace")
    if re.search(r"\blisten(?:ing)?\b", text, re.IGNORECASE) is None:
        return False
    for token in _BINDING_URL_RE.findall(text):
        try:
            parsed = urlsplit(token)
            parsed_port = parsed.port
        except ValueError:
            continue
        if (
            parsed.scheme.casefold() == "http"
            and parsed.hostname == LOOPBACK_HOST
            and parsed_port == port
            and parsed.username is None
            and parsed.password is None
        ):
            return True
    return False


def _binding_chunk_reader(stream: BinaryIO) -> Callable[[int], bytes] | None:
    """Return a nonblocking chunk reader, or ``None`` when none is safe.

    ``Popen(stdout=PIPE)`` supplies a real descriptor. ``BytesIO`` is supported
    as the finite test seam. An unconstrained mock is deliberately rejected:
    converting its synthetic ``fileno`` to an int can target this process's own
    stdout, and its synthetic reads never reach EOF.
    """
    if isinstance(stream, io.BytesIO):
        return stream.read
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    if type(fd) is not int or fd < 0:
        return None
    try:
        os.set_blocking(fd, False)
    except (OSError, ValueError):
        return None
    return lambda size: os.read(fd, size)


def _discard_child_output(read_chunk: Callable[[int], bytes]) -> None:
    """Drain child stdout until EOF after it stops carrying proof data."""
    while True:
        try:
            chunk = read_chunk(_BINDING_PROOF_READ_SIZE)
        except BlockingIOError:
            time.sleep(_BINDING_PROOF_POLL_S)
            continue
        except (OSError, ValueError):
            return
        if not isinstance(chunk, bytes) or not chunk:
            return


def _drain_child_output(stream: BinaryIO, port: int, proof: _BindingProof) -> None:
    """Scan a bounded prefix for proof, then drain stdout until EOF.

    The real pipe is nonblocking so the wall-clock deadline bounds a silent
    child. Lines and bytes bound chatty or unterminated output. Exhausting any
    proof limit invalidates an unconsumed report and switches the same daemon
    thread to discard mode, so later child output cannot fill the pipe.
    """
    read_chunk = _binding_chunk_reader(stream)
    if read_chunk is None:
        proof.invalidate()
        return
    deadline = time.monotonic() + _BINDING_PROOF_TIMEOUT_S
    pending = bytearray()
    lines = 0
    total = 0
    matched = False
    drift_logged = False

    def _log_drift() -> None:
        nonlocal drift_logged
        if total > 0 and not matched and not drift_logged:
            logger.warning(
                "playwright-cli show stdout did not contain a recognized listener "
                "address for port %d; the upstream banner may have changed",
                port,
            )
            drift_logged = True

    def _discard_after_proof_window() -> None:
        proof.invalidate()
        _log_drift()
        _discard_child_output(read_chunk)

    try:
        while True:
            now = time.monotonic()
            if (
                now >= deadline
                or lines >= _BINDING_PROOF_MAX_LINES
                or total >= _BINDING_PROOF_MAX_BYTES
            ):
                _discard_after_proof_window()
                return
            remaining = _BINDING_PROOF_MAX_BYTES - total
            try:
                chunk = read_chunk(min(_BINDING_PROOF_READ_SIZE, remaining + 1))
            except BlockingIOError:
                delay = min(_BINDING_PROOF_POLL_S, max(0.0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
                continue
            if not isinstance(chunk, bytes):
                proof.invalidate()
                return
            if not chunk:
                if pending:
                    lines += 1
                    if lines > _BINDING_PROOF_MAX_LINES:
                        _discard_after_proof_window()
                        return
                    if _binding_reported_on_line(bytes(pending), port):
                        matched = True
                        proof.record()
                return
            total += len(chunk)
            if total > _BINDING_PROOF_MAX_BYTES:
                _discard_after_proof_window()
                return
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                line = bytes(pending[:newline])
                del pending[: newline + 1]
                lines += 1
                if lines > _BINDING_PROOF_MAX_LINES:
                    _discard_after_proof_window()
                    return
                if _binding_reported_on_line(line, port):
                    matched = True
                    proof.record()
    except (OSError, ValueError):
        # Reaping closes the pipe. A report not yet consumed must fail closed.
        proof.invalidate()
    finally:
        _log_drift()


def _child_reported_binding(proc: subprocess.Popen[bytes], port: int) -> bool:
    """Consume one fresh recognized-address report from this child, if available."""
    proof = getattr(proc, "_kirocrew_browser_view_binding", None)
    return isinstance(proof, _BindingProof) and proof.port == port and proof.consume()


def _verify_child_listener(
    proc: subprocess.Popen[bytes], port: int, *, allow_report: bool
) -> tuple[bool | None, bool]:
    """Verify this child owns *port* and say whether stdout was the proof.

    Global attribution remains the strongest result. On a blind host, the child
    and each descendant PID are checked independently. The one-shot stdout
    report is accepted only during startup and only when no per-process probe
    can run. Reports are consumed on every path so a PID-proven start cannot
    leave a stale token for a later listener.
    """
    owner = _port_owner(port, proc)
    fresh_report = _child_reported_binding(proc, port)
    if owner == _OWNER_CHILD:
        return True, False
    if owner == _OWNER_FOREIGN:
        return False, False
    if platform_compat.IS_LINUX:
        descendants = platform_compat.linux_process_descendants(proc.pid)
        inconclusive = descendants is None
        descendants = descendants or []
    else:
        descendants = platform_compat.process_descendants(proc.pid)
        inconclusive = False
    candidates = list(dict.fromkeys([proc.pid, *descendants]))
    for pid in candidates:
        per_process = platform_compat.process_owns_loopback_listener(pid, port)
        if per_process is True:
            return True, False
        if per_process is None:
            inconclusive = True
    if not inconclusive:
        return False, False
    if allow_report and fresh_report:
        return True, True
    return None, False


def _recorded_state() -> bool | None:
    """Return True for reusable, False for failed, or None for inconclusive.

    Reuse and status share this decision so they cannot disagree about URL
    publication. A failed health probe or a completed ownership mismatch is a
    definitive negative. Missing ownership evidence is inconclusive: keep the
    live child and withhold its URL until ownership can be proved.

    Callers hold :data:`_lock`.
    """
    global _last_reason
    if _proc is None or not _alive(_proc) or _info is None or _child_port is None:
        return False
    if not _healthy(_info.port):
        reason = f"live browser view child stopped answering on port {_child_port}"
        if _last_reason != reason:
            logger.warning("%s", reason)
        _last_reason = reason
        return False
    verified, _via_report = _verify_child_listener(_proc, _child_port, allow_report=False)
    if verified is True:
        _last_reason = None
        return True
    if verified is False:
        reason = f"live browser view child no longer owns port {_child_port}"
        if _last_reason != reason:
            logger.warning("%s", reason)
        _last_reason = reason
        return False
    reason = f"cannot re-verify live browser view child ownership of port {_child_port}"
    if _last_reason != reason:
        logger.warning("%s; keeping the child but withholding its URL", reason)
    _last_reason = reason
    return None


def _show_argv(command: list[str], port: int) -> list[str]:
    """Argv for the dashboard server.

    ``--host`` is always present and always loopback: omitting it yields an
    IPv6-only listener that ``127.0.0.1`` cannot reach.
    """
    return [*command, "show", "--port", str(port), "--host", LOOPBACK_HOST]


def _alive(proc: subprocess.Popen[bytes] | None) -> bool:
    return proc is not None and proc.poll() is None


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """Terminate *proc* and its descendants, escalating to a kill.

    The CLI spawns a browser and helper processes, so signalling only the direct
    child leaves the tree behind holding the port.
    """
    with contextlib.suppress(Exception):
        platform_compat.kill_process_tree(proc.pid)
    try:
        proc.wait(timeout=_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        logger.warning("playwright-cli show (pid %s) ignored terminate; killing", proc.pid)
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=_TERMINATE_GRACE_S)


def _spawn(command: list[str], port: int) -> subprocess.Popen[bytes] | None:
    """Start the dashboard server, or ``None`` if it cannot be spawned.

    Stdout is a child-specific proof channel. Playwright prints its exact
    listening address only after its HTTP server binds, so a daemon reader scans
    a finite nonblocking prefix for that line. Line, byte, and time limits bound
    the proof scan; after any limit, the daemon keeps draining and discarding
    stdout so a chatty long-lived child cannot fill its pipe. Stderr stays on
    ``DEVNULL`` because no readiness or ownership signal is defined there.

    ``start_new_session`` puts the child in its own process group on POSIX so
    the whole tree can be signalled at stop time without touching the gateway's
    own group.

    The child's socket root is the gateway-owned one (:func:`ui_socket_env`):
    the dashboard claims its singleton socket under it, and the Browser panel's
    launcher (:mod:`kiro_crew.browser_cli.launcher`) sends its reveal request
    there, so both must agree on a root the gateway knows.
    """
    env = cli_env()
    env.update(ui_socket_env(env))
    try:
        proc = subprocess.Popen(
            _show_argv(command, port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=platform_compat.IS_POSIX,
            env=env,
        )
    except OSError as exc:
        logger.warning("could not start playwright-cli show: %s", exc)
        return None

    stream = getattr(proc, "stdout", None)
    if stream is not None:
        proof = _BindingProof(port=port, reported=threading.Event())
        setattr(proc, "_kirocrew_browser_view_binding", proof)
        threading.Thread(
            target=_drain_child_output,
            args=(stream, port, proof),
            name="browser-view-listener-proof",
            daemon=True,
        ).start()
    return proc


def ensure_running(port: int | None = None) -> ShowInfo | None:
    """Return the running dashboard, starting it if needed.

    Idempotent: a process that is alive, answering, and still owns its listener
    is reused, so repeated calls from a panel mount do not spawn a second server.
    A dead recorded process or one with a definitive health or ownership failure
    is reaped before replacement. A live recorded process with inconclusive
    ownership stays running in a degraded state with no published URL.

    *port* pins the port the dashboard is reachable on; ``None`` (or ``0``)
    keeps the OS-assigned ephemeral default. A pin is never handed to the
    child directly — the module claims the pinned port itself with a bound
    listener (:class:`_Relay`, the atomic ownership proof) and relays to the
    child's own ephemeral port, so the operator-named port itself has no
    probe-to-bind window to race. The pin applies when a server is (re)started —
    an already-healthy server is reused as-is, on whatever port it holds.
    The bind host stays :data:`LOOPBACK_HOST` regardless.

    ``None`` means no dashboard is available: the CLI is not installed, the
    pinned port is already taken by something else, or the server did not
    become healthy within the startup budget. ``status()`` carries the reason.
    """
    global _proc, _info, _relay, _last_reason, _child_port
    with _lock:
        # Ownership is re-proved on reuse, not just at startup. A child that is
        # alive but not listening leaves its port free for a squatter, and
        # without this the next call would hand that squatter back as the panel.
        recorded_state = _recorded_state()
        if recorded_state is True:
            return _info
        if recorded_state is None:
            return None
        if _proc is not None:
            _reap(_proc)
            _proc = None
            _info = None
        if _relay is not None:
            _relay.close()
            _relay = None
        _child_port = None
        _last_reason = None

        cli = cli_path()
        command = cli_command(cli) if cli is not None else None
        if command is None:
            return None

        relay: _Relay | None = None
        if port:
            # Claim the pin BEFORE choosing the child's port. Two things follow
            # by construction: bind() either makes the pin ours until we close
            # it or raises because someone else holds it (no window in which a
            # squatter can be mistaken for us), and while the pin is bound
            # _free_port() cannot hand the same number back, so the child's
            # ephemeral port can never collide with the pin.
            pin_listener = _claim_listener(port)
            if pin_listener is None:
                logger.warning("configured browser view port %d is already in use", port)
                _last_reason = f"configured port {port} is already in use"
                return None
            child_port = _free_port()
            relay = _Relay.from_listener(pin_listener, child_port)
        else:
            child_port = _free_port()
        proc = _spawn(command, child_port)
        if proc is None:
            if relay is not None:
                relay.close()
            _last_reason = "playwright-cli show could not be started"
            return None

        public_port = port if port else child_port
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        saw_unproven_responder = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logger.warning(
                    "playwright-cli show exited during startup (rc=%s) on port %d",
                    proc.returncode,
                    child_port,
                )
                if relay is not None:
                    relay.close()
                _last_reason = f"playwright-cli show exited during startup on port {child_port}"
                return None
            if _healthy(child_port):
                # Reachability is not identity. Prove the responder is ours
                # before adopting it; a squatter that won _free_port's window
                # answers this probe exactly as our child would.
                verified, via_report = _verify_child_listener(proc, child_port, allow_report=True)
                if verified is False:
                    logger.warning(
                        "another local process holds port %d — refusing to adopt "
                        "it as the browser view",
                        child_port,
                    )
                    if relay is not None:
                        relay.close()
                    _last_reason = (
                        f"another local process took port {child_port} before the "
                        f"view server could bind it"
                    )
                    _reap(proc)
                    return None
                if verified is None:
                    # The HTTP responder may be a squatter. Keep polling only so
                    # the trusted child's post-bind stdout line can arrive.
                    saw_unproven_responder = True
                else:
                    if via_report:
                        logger.warning(
                            "cannot verify which process holds port %d (%s is "
                            "unavailable or cannot attribute a known local listener); "
                            "adopting only because the spawned child reported that "
                            "it bound a recognized loopback listener address",
                            child_port,
                            platform_compat.listening_pid_tool(),
                        )
                    _proc = proc
                    _relay = relay
                    _child_port = child_port
                    _info = ShowInfo(url=f"http://{LOOPBACK_HOST}:{public_port}", port=public_port)
                    return _info
            time.sleep(_POLL_INTERVAL_S)

        if saw_unproven_responder:
            logger.warning(
                "cannot verify which process holds port %d and the spawned child "
                "did not report that it bound the exact loopback address; refusing "
                "to adopt the responder",
                child_port,
            )
            _last_reason = f"could not prove the browser view owns port {child_port}"
        else:
            logger.warning(
                "playwright-cli show did not answer on port %d within the budget",
                child_port,
            )
            _last_reason = (
                f"playwright-cli show did not answer on port {child_port} within the budget"
            )
        if relay is not None:
            relay.close()
        _reap(proc)
        return None


def stop() -> None:
    """Stop the supervised dashboard child and its entire process tree.

    Reaping is scoped to the child we spawned: ``_spawn`` places it in its
    own session (``start_new_session=IS_POSIX``), so ``kill_process_tree``
    signals the whole group — the Node server, the browser, and any helpers
    — without touching processes outside that group. A global ``show --kill``
    is deliberately NOT issued because it would terminate an operator's own
    independently-launched ``playwright-cli show`` session, destroying their
    unsaved work.
    """
    global _proc, _info, _relay, _last_reason, _child_port
    with _lock:
        if _proc is not None:
            _reap(_proc)
        if _relay is not None:
            _relay.close()
        _proc = None
        _info = None
        _relay = None
        _child_port = None
        _last_reason = None


def status() -> dict[str, Any]:
    """Current dashboard state, without starting or stopping anything.

    ``unavailable`` is reported when the CLI is absent, and is distinct from
    ``stopped``: the first cannot be fixed by starting the server. A
    ``stopped`` state carries the last start attempt's failure reason when
    one is recorded — with a pinned port, "already in use" is the most likely
    misconfiguration, and reporting it is what separates a fixable setting
    from a mysteriously dead panel.
    """
    with _lock:
        cli = cli_path()
        if cli is None or cli_command(cli) is None:
            return {
                "status": "unavailable",
                "url": None,
                "port": None,
                "reason": "playwright-cli is not installed",
            }
        if _recorded_state() is True and _info is not None:
            return {
                "status": "running",
                "url": _info.url,
                "port": _info.port,
                "reason": None,
            }
        return {"status": "stopped", "url": None, "port": None, "reason": _last_reason}
