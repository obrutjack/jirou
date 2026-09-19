"""The MCP half of the ability card is a PROJECTION, and these tests keep it one.

The capability half already had its guards (``test_backend_cards``). This half
reads a different source -- the declarations in ``providers/mirrors`` -- so it has
its own failure modes, and each test below is one cost the card exists to stop
paying:

1. **Completeness under a new harness.** A selectable backend with no section is
   the invisible-difference failure the mirror folder exists to prevent, so the
   card must answer for every one of them. Not by GUESSING: an undeclared id
   answers ``""``, and the test asserts that too.
2. **Completeness under a new spec concern.** ``Concern`` is closed and a mirror
   must rule on every member, so a member added there must be either ON the card
   or recorded in :data:`~kiro_crew.agent_sdk.backend_mcp_ability.OFF_CARD_CONCERNS`
   with the reason it is not. A concern in neither is one nobody decided about.
3. **Completeness under a new kind or reach.** Every ``ProjectionKind`` and every
   ``PerToolDeny`` member must reach the wire as its own value, and doctor must
   have a phrase for it -- otherwise a harness declaring the new one renders a card
   that silently says nothing about the thing that makes it different.
4. **No per-harness prose.** The module may not name a harness id: the moment it
   does, onboarding costs an edit here and the projection has become a table.
5. **The declaration is what reaches the reader.** The shipped values are asserted
   end to end, because a declaration nothing renders is not a feature.

It DECLARES. Nothing here asserts that any backend enforces a per-tool MCP deny,
because two of them cannot and are not asked to.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Set

from kiro_crew.acp_backends import selectable_backend_values
from kiro_crew.agent_sdk import backend_mcp_ability as mcp_mod
from kiro_crew.agent_sdk.backend_cards import card_payload
from kiro_crew.providers.mirrors import (
    PROJECTIONS,
    Concern,
    Disposition,
    PerToolDeny,
    ProjectionKind,
)

MODULE = Path(mcp_mod.__file__)

#: A backend id no declaration names, standing in for the next harness onboarded.
STRANGER = "a-harness-nobody-declared"


# ── 1. every selectable harness answers, and a stranger answers nothing ─────


def test_every_selectable_backend_has_a_card() -> None:
    """The acceptance condition: a harness reaches the switch with a card.

    ``PROJECTIONS`` is what the mirror parity test already holds every selectable
    backend to; this asserts the card RENDERS that declaration rather than the
    declaration merely existing, which is the gap the issue reported.
    """
    for backend in selectable_backend_values():
        ability = mcp_mod.ability_for(backend)
        assert ability.projection, f"{backend!r} renders no projection kind"
        assert ability.projection in {kind.value for kind in ProjectionKind}, backend


def test_a_harness_with_no_declaration_claims_nothing() -> None:
    """Fail-closed, and loudly empty rather than quietly reassuring.

    A card that guessed would be the invisible-difference failure ``providers/mirrors``
    exists to stop: an operator would read a complete-looking section about a harness
    nothing has been declared about. Total, because it is served on a request path.
    """
    ability = mcp_mod.ability_for(STRANGER)
    assert ability.projection == ""
    assert ability.per_tool_deny == ""
    assert ability.withheld == ()
    assert ability.no_channel == ()


def test_a_new_harness_needs_no_edit_to_this_module() -> None:
    """Onboarding costs a PROJECTIONS entry and nothing here.

    Driven through the real registry rather than a stub: a kind and a reach already
    shipped by some harness are read back on an id this module has never seen, so
    the projection is shown to be keyed on the DECLARATION and not on the id.
    """
    borrowed = next(
        backend
        for backend, declared in PROJECTIONS.items()
        if declared.kind is ProjectionKind.MIRROR
    )
    ability = mcp_mod.ability_for(borrowed)
    # The same declaration, reached only through the id -- no branch in this module
    # names the harness, which the AST test below proves structurally.
    assert ability.projection == ProjectionKind.MIRROR.value
    assert ability.per_tool_deny == PROJECTIONS[borrowed].per_tool_deny.value


# ── 2. every spec concern is classified ─────────────────────────────────────


def test_every_concern_is_classified() -> None:
    """A concern in neither bucket is a concern nobody decided about.

    ``Concern`` is closed and every mirror must rule on every member, so adding one
    there obliges this file to answer the reader's half of the same question: is
    this something an operator choosing a harness needs to see?
    """
    classified: Set[str] = {*mcp_mod.ON_CARD_CONCERNS, *mcp_mod.OFF_CARD_CONCERNS}
    defined = {member.name for member in Concern}
    missing = sorted(defined - classified)
    assert not missing, (
        f"these spec concerns are not classified in agent_sdk/backend_mcp_ability.py: "
        f"{missing}. Put each in ON_CARD_CONCERNS (a reader choosing a harness can act "
        f"on how it is ruled) or in OFF_CARD_CONCERNS with the reason it reaches no "
        f"card line."
    )


def test_the_module_classifies_no_concern_that_does_not_exist() -> None:
    """The other direction, which a coverage count cannot see.

    A renamed or deleted concern left behind here would read as classified while
    naming nothing, and the line built on it would answer "withholds nothing" for
    every harness forever.
    """
    classified: Set[str] = {*mcp_mod.ON_CARD_CONCERNS, *mcp_mod.OFF_CARD_CONCERNS}
    defined = {member.name for member in Concern}
    stale = sorted(classified - defined)
    assert not stale, f"these names answer to no Concern member: {stale}"


def test_every_off_card_concern_carries_its_reason() -> None:
    """An omission with no reason is indistinguishable from a decision.

    The same rule ``McpProjection`` enforces on its own kinds, and the reason this
    is a mapping rather than a set.
    """
    for concern, reason in mcp_mod.OFF_CARD_CONCERNS.items():
        assert reason.strip(), f"{concern} is off the card with no reason"


def test_the_two_off_card_concerns_are_the_context_text_ones() -> None:
    """Pinned by name, because leaving them ON would print a falsehood.

    Every mirror rules PROMPT and RESOURCES withheld, and both reach the harness
    anyway -- as ordinary context text rather than through a projection. Rendering
    the withhold verbatim would tell a reader their agent never receives its own
    instructions, which is the opposite of what happens. A third entry here is a
    deliberate change to this assertion.
    """
    assert set(mcp_mod.OFF_CARD_CONCERNS) == {"PROMPT", "RESOURCES"}
    for backend, declared in PROJECTIONS.items():
        if declared.kind is not ProjectionKind.MIRROR:
            continue
        ruled = mcp_mod._dispositions(backend)
        for name in ("PROMPT", "RESOURCES"):
            cid = Concern[name].name.lower()
            assert ruled.get(cid) == Disposition.WITHHELD.value, (backend, name)


# ── 3. every kind and every reach reaches a reader ──────────────────────────


def test_every_projection_kind_reaches_the_wire_as_its_own_value() -> None:
    """The card renders the KIND, so a new kind must be distinguishable on it.

    Read through a real declaration per kind rather than asserted over the enum
    alone: the wire carries the value, and a projection that collapsed two kinds
    onto one string would still pass an enum-only check.
    """
    seen = {mcp_mod.ability_for(backend).projection for backend in PROJECTIONS}
    declared = {declared.kind.value for declared in PROJECTIONS.values()}
    assert seen == declared


def test_every_deny_reach_is_named_by_the_doctor_report() -> None:
    """A reach with no phrase is a row that says nothing about the difference.

    ``kirocrew doctor`` spells the three phrases itself (they cross the agent-sdk
    boundary as plain strings), so this is where the two are held together. The row
    falls back to the raw value rather than to silence, which keeps it honest -- but
    a raw ``whole-server`` in front of an operator is not the point of the row.
    """
    from kiro_crew import cli_doctor

    for member in PerToolDeny:
        assert member.value in cli_doctor._DENY_PHRASE, member


def test_every_projection_kind_is_named_by_the_doctor_report() -> None:
    """Same contract for the kind."""
    from kiro_crew import cli_doctor

    for member in ProjectionKind:
        assert member.value in cli_doctor._PROJECTION_PHRASE, member


def test_a_concern_id_is_the_member_name_and_not_the_dotted_spec_key() -> None:
    """The id keys a translated label, so it may not be a path into a payload.

    ``permissions.defaultMode`` is the concern's VALUE and the right spelling in the
    registry; as a card id it reads as a path. The spec key is offered separately,
    for the terminal reader who is holding the file.
    """
    assert mcp_mod.concern_id(Concern.PERMISSION_MODE) == "permission_mode"
    assert mcp_mod.spec_keys()["permission_mode"] == "permissions.defaultMode"
    assert set(mcp_mod.spec_keys()) == {member.name.lower() for member in Concern}


# ── 4. no per-harness prose ─────────────────────────────────────────────────


def test_the_module_names_no_harness() -> None:
    """The projection may not branch on WHICH harness it is describing.

    Read from the AST rather than by grepping, so a comparison cannot hide behind
    formatting. Two shapes are refused, the same two ``test_backend_cards`` refuses:
    naming an ``ACP_BACKEND_*`` identifier, and testing ``backend`` for equality.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("ACP_BACKEND_"):
            offenders.append(f"{node.id} at line {node.lineno}")
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            equality = any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            if node.left.id == "backend" and equality:
                offenders.append(f"equality on `backend` at line {node.lineno}")
    assert not offenders, (
        "the MCP card must be a projection over the declaration, so it may not name "
        f"a harness id or compare one for equality: {offenders}"
    )


def test_no_card_id_names_a_harness() -> None:
    """A card id is a CONCERN, so it may not carry a harness's name.

    A per-harness id would mean a per-harness label, which is a per-harness edit to
    thirteen locale files -- the cost this projection exists to remove.
    """
    harness_words = ("kiro", "claude", "kas", "codex", "opencode", "goose", "deepseek", "pi_")
    for member in Concern:
        for word in harness_words:
            assert word not in mcp_mod.concern_id(member), f"{member.name} names a harness"


# ── 5. the shipped declarations, end to end ─────────────────────────────────


def test_the_shipped_whole_server_harnesses_say_so_on_their_card() -> None:
    """The row the maintainer's ruling turned into a card line.

    Per-tool ``mcp.deny`` is NOT a hard requirement on every provider. A harness
    without a per-call deny channel withholds the whole server instead, and what it
    owes a reader is to DECLARE that before a session runs. This asserts the
    declaration reaches the card for every harness that carries it -- by set rather
    than by one id, so a harness that gains or loses the reach is caught.
    """
    declared = {
        backend
        for backend, projection in PROJECTIONS.items()
        if projection.per_tool_deny is PerToolDeny.WHOLE_SERVER
    }
    assert declared, "no harness declares the whole-server reach any more"
    for backend in declared:
        assert mcp_mod.ability_for(backend).per_tool_deny == PerToolDeny.WHOLE_SERVER.value


def test_a_withhold_and_a_no_channel_are_separate_lists() -> None:
    """The split IS the meaning, so it is asserted rather than assumed.

    A withhold is a decision with a reason; a no-channel is a gap the transport
    cannot carry today and has an address recorded for it. Collapsing them would
    tell a reader a settled ruling and an open gap are the same answer -- which is
    the exact conflation ``Disposition`` was introduced to end.
    """
    mirrored = [b for b, d in PROJECTIONS.items() if d.kind is ProjectionKind.MIRROR]
    assert mirrored
    for backend in mirrored:
        ability = mcp_mod.ability_for(backend)
        assert not set(ability.withheld) & set(ability.no_channel), backend
        ruled = mcp_mod._dispositions(backend)
        for cid in ability.withheld:
            assert ruled[cid] == Disposition.WITHHELD.value, (backend, cid)
        for cid in ability.no_channel:
            assert ruled[cid] == Disposition.NO_CHANNEL.value, (backend, cid)


def test_a_kind_with_no_mirror_carries_no_per_concern_ruling() -> None:
    """Only a mirror can answer per concern, so only a mirror does.

    A ``native`` harness reads the spec itself and an ``external`` projection lives
    outside this folder; neither has rulings to read, and inventing a list for them
    would be the card claiming a loss where there is none.
    """
    for backend, declared in PROJECTIONS.items():
        if declared.kind is ProjectionKind.MIRROR:
            continue
        ability = mcp_mod.ability_for(backend)
        assert ability.withheld == (), backend
        assert ability.no_channel == (), backend
        assert ability.per_tool_deny == "", backend


def test_the_card_render_order_is_the_servers() -> None:
    """A LIST in server order, so a new concern lands in the right place.

    The frontend holds a label per id and no order of its own, which is what keeps
    a new line from costing a frontend edit.
    """
    ability = mcp_mod.ability_for(
        next(b for b, d in PROJECTIONS.items() if d.kind is ProjectionKind.MIRROR)
    )
    on_card = [Concern[name].name.lower() for name in mcp_mod.ON_CARD_CONCERNS]
    for listed in (ability.withheld, ability.no_channel):
        assert list(listed) == [cid for cid in on_card if cid in listed]


# ── 6. the wire shape ───────────────────────────────────────────────────────


def test_the_payload_carries_every_field_the_panel_reads() -> None:
    """The projection owns its own wire shape, so the shape is pinned here."""
    payload = mcp_mod.ability_payload(
        next(b for b, d in PROJECTIONS.items() if d.kind is ProjectionKind.MIRROR)
    )
    assert set(payload) == {"projection", "per_tool_deny", "withheld", "no_channel"}
    assert isinstance(payload["withheld"], list)
    assert isinstance(payload["no_channel"], list)


def test_the_card_endpoint_carries_the_mcp_group() -> None:
    """One card on the wire, so the panel reads one object rather than two calls."""
    for backend in selectable_backend_values():
        payload = card_payload(backend)
        assert set(payload["mcp"]) == {  # type: ignore[arg-type]
            "projection",
            "per_tool_deny",
            "withheld",
            "no_channel",
        }, backend


def test_the_payload_is_json_native() -> None:
    """``web.json_response`` refuses a frozenset or a dataclass.

    The failure would be a 500 on the panel's own poll rather than a missing line.
    """
    for backend in sorted(PROJECTIONS):
        json.dumps(mcp_mod.ability_payload(backend))
    json.dumps(mcp_mod.ability_payload(STRANGER))


def test_the_reason_prose_never_reaches_the_wire() -> None:
    """Deliberately not projected, and the omission is load-bearing.

    ``McpProjection.reason`` and ``Ruling.reason`` are written for the reader of the
    registry, at registry length and in a maintainer's register -- one of them names
    an upstream Rust function. The same choice ``backend_mcp_projection`` already
    made for the same field. If a user-facing reason is ever wanted, it is a new
    FIELD every mirror fills in, not this one re-registered.
    """
    for backend, declared in PROJECTIONS.items():
        blob = json.dumps(mcp_mod.ability_payload(backend))
        assert declared.reason[:40] not in blob, backend
