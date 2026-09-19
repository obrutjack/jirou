"""The third-party execution OFF contract: turning it off REVOKES.

Issue #1038 asked which of two meanings ``agent.apps_allow_third_party=false``
carries. The repo had already answered — the dashboard endpoint sweeps on the
falling edge, and the setting is excluded from the generic settings PATCH so no
caller reaches it without that sweep — but the endpoint is not the setting's only
writer. The CLI writes the same config, and so does a text editor. Neither ran
the sweep, and the boot reconcile only revokes at the NEXT start, so in between a
backend admitted solely by the blanket flag kept serving under a ceiling the
operator had already closed.

These tests pin that gap closed at the one mechanism that already revisits every
live backend, and pin the three things the fix must NOT do: touch an app holding
its own grant, touch a builtin, or flood the audit trail with one admission row
per poll.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiro_crew.apps.manager import (
    APP_MANIFEST_FILENAME,
    _read_installed,
    _write_installed,
    install_app,
)

APP = "off-contract-app"


def _install(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, *, name: str = APP) -> None:
    """Install *name* into a scratch home so it has a real installed record."""
    home = tmp_path / "kirocrew-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("KIROCREW_HOME", str(home))
    source = tmp_path / "source" / name
    source.mkdir(parents=True)
    (source / APP_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "displayName": "Off Contract App",
                "description": "Exercises the OFF contract",
                "author": "tester",
            }
        ),
        encoding="utf-8",
    )
    assert install_app(source).ok
    meta = _read_installed(name)
    assert meta is not None
    meta.enabled = True
    meta.origin = "registry"
    _write_installed(name, meta)


def _write_config(**agent: Any) -> None:
    """Write config.json the way a text editor or the CLI would.

    Deliberately NOT a monkeypatch of the loader: the whole point of the gap is
    that this route never passes through the dashboard endpoint, so a test that
    patched the loaded value would not exercise the route that was broken.
    """
    from kiro_crew.config.loader import KiroCrewConfig, config_path

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"agent": agent}), encoding="utf-8")
    # The loader keys its cache on a (mtime, size, mode) fingerprint, and a test
    # writes both files inside one coarse timestamp tick.
    KiroCrewConfig.load()


class _LiveProc:
    """A backend process that is still running.

    Chosen deliberately: with a LIVE process the sweep would go on to probe health,
    so a test that asserts ``_health_probe`` is never called proves the ceiling is
    enforced BEFORE health is judged. A record with ``proc=None`` and no
    ``adopted_pids`` is a different path — an adopted backend whose PIDs the stop
    refuses to signal — exercised in
    :class:`TestAnUnstoppableAdoptedBackendKeepsBeingRetried`.
    """

    pid = 4242
    returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def _tracked(monkeypatch: pytest.MonkeyPatch, name: str = APP, *, proc: Any = ...):
    """Register a live backend record for *name* and run sweeps with no delay."""
    import kiro_crew.apps.backend as bmod
    from kiro_crew.apps.backend import AppProcess

    monkeypatch.setattr(bmod, "_HEALTH_WATCH_INTERVAL", 0)
    ap = AppProcess(
        app_name=name,
        port=9301,
        healthy=True,
        mcp_healthy=True,
        proc=_LiveProc() if proc is ... else proc,
    )
    with bmod._lock:
        bmod._processes[name] = ap
    return bmod, ap


class TestAConfigFileEditRevokes:
    """The bypass this change closes."""

    def test_a_file_edit_stops_a_backend_it_no_longer_admits(self, tmp_path, monkeypatch) -> None:
        _install(tmp_path, monkeypatch)
        _write_config(apps_allow_third_party=True)
        bmod, ap = _tracked(monkeypatch)
        try:
            # While the file grants it, the sweep must leave it alone.
            assert bmod._revoke_if_ceiling_closed(ap) is False
            with bmod._lock:
                assert ap.app_name in bmod._processes

            # Now the operator edits the file. No endpoint, no CLI, no restart.
            _write_config(apps_allow_third_party=False)
            monkeypatch.setattr(
                bmod,
                "_health_probe",
                lambda *_a, **_k: pytest.fail(
                    "the ceiling must be enforced before health is judged"
                ),
            )
            bmod._watch_backend_health_sweeps(ap, "/health")

            with bmod._lock:
                assert ap.app_name not in bmod._processes, (
                    "a backend running only on blanket trust must be stopped once "
                    "the file no longer grants it"
                )
        finally:
            with bmod._lock:
                bmod._processes.clear()

    def test_the_apps_own_grant_survives_the_blanket_flag_going_off(
        self, tmp_path, monkeypatch
    ) -> None:
        """A per-app grant is independent of the blanket flag, so it is untouched."""
        _install(tmp_path, monkeypatch)
        _write_config(apps_allow_third_party=False, apps_trusted=[APP])
        bmod, ap = _tracked(monkeypatch)
        try:
            assert bmod._revoke_if_ceiling_closed(ap) is False
            with bmod._lock:
                assert ap.app_name in bmod._processes, "a granted app must not be swept up"
        finally:
            with bmod._lock:
                bmod._processes.clear()

    def test_an_app_with_no_installed_record_is_not_this_mechanisms_business(
        self, tmp_path, monkeypatch
    ) -> None:
        """Same population as the endpoint's sweep, which enumerates list_apps()."""
        home = tmp_path / "kirocrew-home"
        home.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(home))
        _write_config(apps_allow_third_party=False)
        bmod, ap = _tracked(monkeypatch, name="never-installed")
        try:
            assert bmod._revoke_if_ceiling_closed(ap) is False
            with bmod._lock:
                assert ap.app_name in bmod._processes
        finally:
            with bmod._lock:
                bmod._processes.clear()

    def test_a_revocation_is_audited_even_though_the_poll_is_not(
        self, tmp_path, monkeypatch
    ) -> None:
        _install(tmp_path, monkeypatch)
        _write_config(apps_allow_third_party=False)
        bmod, ap = _tracked(monkeypatch)
        rows: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.apps.execution.sel",
            lambda: type(
                "_Sel",
                (),
                {"log_api_access": staticmethod(lambda **kw: rows.append(kw))},
            )(),
        )
        try:
            assert bmod._revoke_if_ceiling_closed(ap) is True
        finally:
            with bmod._lock:
                bmod._processes.clear()
        admissions = [r for r in rows if r.get("operation") == "app_execution_admission"]
        assert (
            len(admissions) == 1
        ), "exactly one row: the poll writes none and the action writes one"
        assert admissions[0]["outcome"] == "denied"
        assert "health_watch_ceiling_revocation" in admissions[0]["resources"]


class TestAnUnstoppableAdoptedBackendKeepsBeingRetried:
    """The watch must not exit when the stop refused and put the record back.

    `stop_app_backend` restores tracking for an adopted backend whose PIDs it will
    not signal, precisely so the stop can be retried. Exiting on that would abandon
    the worst case: un-trusted code still serving, with nothing retrying the
    revocation and nothing watching its liveness.
    """

    def test_a_refused_stop_leaves_the_watch_running(self, tmp_path, monkeypatch) -> None:
        _install(tmp_path, monkeypatch)
        _write_config(apps_allow_third_party=False)
        # proc=None with no adopted_pids is the record whose stop is refused.
        bmod, ap = _tracked(monkeypatch, proc=None)
        try:
            assert (
                bmod._revoke_if_ceiling_closed(ap) is False
            ), "a refused stop must keep the watch alive so it retries"
            with bmod._lock:
                assert (
                    bmod._processes.get(ap.app_name) is ap
                ), "stop_app_backend restored tracking; the watch must respect that"
        finally:
            with bmod._lock:
                bmod._processes.clear()


class TestThePollerDoesNotFloodTheAuditTrail:
    """Why `third_party_ceiling_closed` exists at all.

    The watch re-asks this about every live backend every `_HEALTH_WATCH_INTERVAL`
    seconds. Routing those polls through `app_execution_denied` would add a row per
    app per sweep whose whole content is "nothing changed", burying the admissions
    the trail exists to show.
    """

    def _rows(self, monkeypatch) -> list[dict]:
        rows: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.apps.execution.sel",
            lambda: type(
                "_Sel",
                (),
                {"log_api_access": staticmethod(lambda **kw: rows.append(kw))},
            )(),
        )
        return rows

    def test_a_poll_writes_nothing_while_the_gate_writes_a_row(self, tmp_path, monkeypatch) -> None:
        from kiro_crew.apps.execution import (
            app_execution_denied,
            third_party_ceiling_closed,
        )

        _install(tmp_path, monkeypatch)
        _write_config(apps_allow_third_party=True)
        rows = self._rows(monkeypatch)

        for _ in range(5):
            assert third_party_ceiling_closed(APP) is None
        assert rows == [], "a poll is not an admission and must not be recorded"

        assert app_execution_denied(APP, action="module_load") is None
        assert len(rows) == 1, "the gate itself still records every admission"
        assert rows[0]["outcome"] == "allowed"


class TestThePollerAndTheGateCannotDisagree:
    """The divergence ratchet.

    Both read the same `_execution_admission` core, so this asserts the property
    that made that split worth doing: on this boundary a disagreement means a
    backend still executing under a ceiling the operator believes is closed.
    """

    @pytest.mark.parametrize(
        "agent",
        [
            {"apps_allow_third_party": True},
            {"apps_allow_third_party": False},
            {"apps_allow_third_party": False, "apps_trusted": [APP]},
            {"apps_allow_third_party": True, "apps_trusted": [APP]},
            {"apps_allow_third_party": "true"},
            {"apps_allow_third_party": 1},
            {},
        ],
    )
    def test_both_paths_reach_the_same_verdict(self, tmp_path, monkeypatch, agent) -> None:
        from kiro_crew.apps.execution import (
            app_execution_denied,
            third_party_ceiling_closed,
        )

        _install(tmp_path, monkeypatch)
        _write_config(**agent)
        gate = app_execution_denied(APP, action="module_load")
        poll = third_party_ceiling_closed(APP)
        assert poll == gate
