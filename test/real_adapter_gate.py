"""The gate the real-adapter contract tests stand on, and the versions they measured.

A test that measures a real adapter has two audiences with opposite needs. A
developer without the adapter installed needs it to skip. The CI lane that exists
to run it needs an absent adapter to be RED, because a lane whose only guard
skipped asserted nothing while reporting success -- which is the whole reason a
skip-only guard is worth replacing.

``KIROCREW_E2E_REQUIRE=1`` separates them, and it is deliberately the switch this
repository already has rather than a second spelling of it:
``test/e2e/scenarios/conftest.py`` reads the same name for the same contract ("skip
on a missing precondition, or FAIL when the job declared the precondition must
hold"), as do the Playwright gate and the gateway-boot matrix. The name says E2E
and this lane is not E2E, but one name for one mechanism is the property worth
keeping: a job sets it for its own module set, and here that set is the guarded
contract tests. Unset, which is every local run, an absent adapter skips exactly as
a ``skipif`` did.

The flag is not the only thing holding the rule. The lane also asserts on its junit
report that no selected case was skipped, so a run where the variable never arrived
cannot pass by skipping -- and every guarded test carries the ``real_adapter``
marker the lane selects on, so a new one is picked up rather than silently left out.

The two versions are the pins. Each names the adapter release the live measurements
in one guarded test file were taken against, and the lane reads them from HERE
instead of repeating them, so the version installed and the version measured cannot
drift apart. Bumping one is a deliberate act: the assertions that rest on it have to
be re-measured against the new release, and nothing here claims to detect a release
nobody has measured yet.

What a pin here does NOT cover, stated so the next reader does not have to discover
it from a red lane: it is the TOP-LEVEL version only. Each adapter resolves its own
dependencies -- the Codex app server and the opencode executable among them -- at
install time, and the lane installs rather than restoring a lockfile, so a cache
miss re-resolves whatever those ranges then admit. A transitive publish can
therefore move what the lane measures, or red it, on a change that touched nothing
of ours.

That is accepted rather than overlooked, and the alternative is named: a committed
lockfile would pull both adapters' whole trees into this repository's dependency
licence and vulnerability surface, which is a maintainer's policy decision and not
a test lane's to make.

So the procedure when the lane reds with no pin bump in the diff: re-run it once to
separate a flake; reproduce locally by installing the two pins above, which is
exactly what the lane does; if it reproduces, the cause is an adapter release or one
of its dependencies, and the answer is to re-measure and bump the pin here -- never
to relax an assertion to match whatever the new tree does.
"""

from __future__ import annotations

import os

import pytest

#: The repository's one switch for "this job declared its preconditions must hold",
#: shared with the E2E suites rather than duplicated. Parsed exactly as they parse
#: it: ``== "1"``.
REQUIRE_ENV = "KIROCREW_E2E_REQUIRE"

#: The marker the lane selects on. Every guarded test carries it, so the lane's
#: collection does not enumerate node ids and a test added later is included.
MARKER = "real_adapter"

#: The ``@agentclientprotocol/codex-acp`` release the live measurements in
#: ``test_codex_session_mcp.py`` were taken against.
MEASURED_CODEX_ACP_VERSION = "1.11.0"

#: The ``opencode-ai`` release the live measurements in
#: ``test_opencode_session_mcp.py`` were taken against.
MEASURED_OPENCODE_VERSION = "1.18.30"


def real_adapters_required() -> bool:
    """Whether an absent adapter must fail rather than skip."""
    return os.environ.get(REQUIRE_ENV, "") == "1"


def require_real_adapter(resolved: object, *, what: str, install: str) -> None:
    """Stop the calling test unless *resolved* names an installed *what*.

    *resolved* is whatever the test's own resolver answered -- a path, an argv, a
    binary name -- and anything falsy means absent. The resolver stays the test's,
    so the gate never introduces a second opinion about what a real session would
    spawn.

    Returns for a present adapter. Otherwise skips, or fails where the job declared
    the adapter must be there, with *install* named either way: the reader of a red
    lane and the reader of a local skip both want the same command.
    """
    if resolved:
        return
    absent = f"{what} is not installed"
    if real_adapters_required():
        pytest.fail(
            f"{absent}, and {REQUIRE_ENV}=1: this job declared the real adapters must "
            f"be present, so an absent one is a red lane and not a skipped guard. "
            f"Install it with: {install}"
        )
    pytest.skip(f"{absent} ({install})")
