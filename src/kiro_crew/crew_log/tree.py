"""The session tree -- which session opened which, folded across every session log.

:mod:`kiro_crew.crew_log.projection` folds ONE log and reads nothing else (RFC
FR-4). This module is the one reader that looks across logs, and it is a
different kind of thing on purpose. A ``session_create`` child records its
creator on its own ``session/opened`` entry (``parent {slot, sid?}``: the child
knows its creator at its first turn, the creator never learns the child's id),
so the tree of sessions is not a fold over any one log but over the FIRST entry
of all of them, keyed by slot. dsh does the same with its ``parentSession``
header field and a read-side ``flattenLineage``: the record lives on the child,
the tree is a pure function over the collection, and an orphan or a cycle
degrades to root rather than to an error.

Two pieces, kept apart so each is testable on its own:

* :func:`fold_tree` -- pure. :class:`OpenedRecord` in, one :class:`TreeNode`
  per slot out.
* :class:`SessionTree` -- the scanner. It reads the header and the first entry
  of every session log's oldest surviving segment and keeps that head cached
  per unit, for at most :data:`TREE_UNIT_CAP` units per scan. The store never
  rewrites a written line, so the two lines a scan reads are immutable for as
  long as the segment exists; the cache therefore needs no mtime and is dropped
  only when the segment is gone (retention, removal) or the file was replaced
  (a new inode under the same name, or a shorter file under a recycled one).

Only the first entry is read, and that is enough. The emitter writes ``parent``
from a process-local mint witness that exists before the child's first turn or
never, so the entry that CREATED the log carries the parent whenever any entry
does, and a later re-attach in the same process can only repeat it. The fold
still applies the wider rule -- the parent is taken from ANY record of a slot
that carries one, and a record without one never retracts it -- so a slot whose
later logs were opened after a gateway restart (witness gone, no ``parent``)
folds to the parent its first log recorded.

The tree is keyed by SLOT. ``parent.sid`` on the entry is the creator's ACP
session id at the moment of creation -- an audit citation for a reader of the
logs themselves, not a tree key: a slot outlives its ACP session, and the live
row a child nests under is the slot's. This reader does not read it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kiro_crew.crew_log.schema import KIND_SESSION, Entry
from kiro_crew.crew_log.store import oldest_segment, read_head, unit_dir_for, unit_dirs
from kiro_crew.session_ledger import _store_name
from kiro_crew.validation import MAX_ACP_SESSION_ID_LEN, MAX_SHORT_STRING

logger = logging.getLogger(__name__)

#: The one entry type the tree reads a parent from.
TYPE_OPENED: Final[str] = "session/opened"

#: How many session-log units one scan ADMITS -- lists, reads, caches and folds
#: -- in the store's name order. This is the count bound on everything the
#: scanner holds, transiently or not: the store's enumeration keeps at most this
#: many paths while it runs (:func:`~kiro_crew.crew_log.store.unit_dirs`), the
#: head cache holds one entry per admitted unit and nothing for a unit past the
#: cap, and each retained string is bounded at admission too
#: (:func:`opened_record`), so the scan is bounded in both dimensions. Units
#: past the cap are counted in :attr:`SessionTree.omitted` and said once per
#: snapshot on the wire, never silently left out: a slot whose log went unread
#: folds as "no creator known", which is a different thing from "nobody created
#: it". Far above any real population (retention keeps the store far smaller),
#: so hitting it means retention is off and the store is growing without bound.
TREE_UNIT_CAP: Final[int] = 4096


@dataclass(frozen=True)
class OpenedRecord:
    """What one session log contributes: its header identity, and the creator
    slot its first entry names (``None`` for a session nobody created)."""

    sid: str
    slot: str
    created_at: int
    parent_slot: str | None = None


@dataclass(frozen=True)
class TreeNode:
    """One slot in the tree. ``depth`` is 0 for a root, whether the slot had no
    creator, cited one with no log (orphan), or sits on a cycle.

    ``parent_slot`` is the creator the slot's records cite. It is retained even
    when the tree could not follow it (orphan, cycle): it is the child's record,
    not the fold's.
    """

    slot: str
    parent_slot: str | None
    depth: int
    cycle: bool


def fold_tree(records: Iterable[OpenedRecord]) -> dict[str, TreeNode]:
    """Every slot's node, from the records of every session log. Pure.

    Input order does not matter: records are folded oldest log first, by the
    header's ``createdAt`` and then id, so two scans of the same files agree.
    A slot's parent is the one its OLDEST record carrying a parent names; a
    record with no parent never retracts it. A record whose header has no slot
    has no place in a slot-keyed tree and is dropped.

    An edge is FOLLOWED only when it lands on a slot that has a log of its own.
    A cited creator with no log is a citation, not a place in the tree, so the
    child is a root (``depth`` 0) that still carries its ``parent``. A cycle --
    reachable only through forged or damaged records, since a child cannot have
    created its own creator -- makes every slot on it a root, and the slots
    hanging off it keep their depth relative to that root.
    """
    ordered = sorted(records, key=lambda record: (record.created_at, record.sid))
    has_log: set[str] = set()
    cited: dict[str, str] = {}
    for record in ordered:
        if not record.slot:
            continue
        has_log.add(record.slot)
        if record.parent_slot and record.slot not in cited:
            cited[record.slot] = record.parent_slot

    edge: dict[str, str] = {}
    on_cycle: set[str] = set()
    for slot, parent_slot in cited.items():
        if parent_slot == slot:
            on_cycle.add(slot)
        elif parent_slot in has_log:
            edge[slot] = parent_slot
    on_cycle |= _cycle_members(edge)

    depth: dict[str, int] = {}

    def depth_of(slot: str) -> int:
        known = depth.get(slot)
        if known is not None:
            return known
        # Iterative: a chain is at most as long as the tree, and every step lands
        # on a slot with an edge, none of which is on a cycle (those were pulled
        # out above), so the walk reaches a root.
        path: list[str] = []
        cursor = slot
        while cursor not in depth and cursor in edge and cursor not in on_cycle:
            path.append(cursor)
            cursor = edge[cursor]
        base = depth.get(cursor, 0)
        for step, member in enumerate(reversed(path), start=1):
            depth[member] = base + step
        depth.setdefault(slot, 0)
        return depth[slot]

    nodes: dict[str, TreeNode] = {}
    for slot in has_log:
        nodes[slot] = TreeNode(
            slot=slot,
            parent_slot=cited.get(slot),
            depth=depth_of(slot),
            cycle=slot in on_cycle,
        )
    return nodes


def _cycle_members(edge: Mapping[str, str]) -> set[str]:
    """Every node that lies ON a cycle of *edge* (child -> parent), by colouring.

    A node that merely leads INTO a cycle is not a member: its chain ends at a
    member, which the fold then treats as that chain's root.
    """
    white, grey, black = 0, 1, 2
    colour: dict[str, int] = {}
    members: set[str] = set()
    for start in edge:
        if colour.get(start, white) != white:
            continue
        path: list[str] = []
        cursor: str | None = start
        while cursor is not None and colour.get(cursor, white) == white:
            colour[cursor] = grey
            path.append(cursor)
            cursor = edge.get(cursor)
        if cursor is not None and colour.get(cursor) == grey:
            members.update(path[path.index(cursor) :])
        for node in path:
            colour[node] = black
    return members


def opened_record(
    directory: Path, header: Mapping[str, Any] | None, entry: Entry | None
) -> OpenedRecord | None:
    """The record a unit directory contributes given its parsed head, or ``None``.

    ``None`` is a REFUSAL: an unreadable or non-session header, a header whose
    own id does not fold back to this directory (the refusal
    :func:`~kiro_crew.crew_log.store.unit_header_slot` makes: a directory carrying
    another unit's id would answer for that unit), no entry, or any retained
    string past its bound. Bounds refuse rather than truncate, because every
    string here is RETAINED in the scanner's cache for as long as the log
    exists: a session id past ``MAX_ACP_SESSION_ID_LEN`` (the one constant every
    store of a backend-authored id shares) and a slot key past
    ``MAX_SHORT_STRING`` are not values the gateway writes, and a truncated one
    would be a different key that matches nothing.

    The parent is taken from the first entry only when that entry is a
    ``session/opened`` naming a creator ``slot``; the entry's ``parent.sid`` is
    not read (see the module docstring). Any other first entry -- retention that
    took the creating segment, a process that died between create and announce
    -- contributes the slot with no parent, which the fold reads as "no parent
    known here", never as a retraction.
    """
    if header is None or entry is None or header.get("type") != KIND_SESSION:
        return None
    sid = header.get("id")
    if not isinstance(sid, str) or not _bounded(sid, MAX_ACP_SESSION_ID_LEN):
        return None
    if _store_name(sid) != directory.name:
        return None
    slot = header.get("slot")
    if slot is not None and not _bounded(slot, MAX_SHORT_STRING):
        return None
    created = header.get("createdAt")
    parent_slot: str | None = None
    if entry.type == TYPE_OPENED:
        parent = entry.data.get("parent")
        if isinstance(parent, dict):
            cited_slot = parent.get("slot")
            if cited_slot is not None and not _bounded(cited_slot, MAX_SHORT_STRING):
                return None
            if isinstance(cited_slot, str) and cited_slot:
                parent_slot = cited_slot
    return OpenedRecord(
        sid=sid,
        slot=slot if isinstance(slot, str) else "",
        created_at=created if isinstance(created, int) and not isinstance(created, bool) else 0,
        parent_slot=parent_slot,
    )


def _bounded(value: Any, limit: int) -> bool:
    """Whether *value* is a non-empty string of at most *limit* characters."""
    return isinstance(value, str) and 0 < len(value) <= limit


@dataclass(frozen=True)
class _Head:
    """One unit's cached head: the segment it was read from, that file's
    identity, its size when read, and what it contributed -- ``None`` for a
    refused unit, cached too, so a planted bad line costs one read rather than
    one per scan.

    ``size`` is part of the identity because ``(st_dev, st_ino)`` alone is
    not: a filesystem hands a freed inode number to the next file it creates,
    so a segment removed and recreated under the same name can answer the same
    ``stat``. A segment is append-only and never shrinks, so a file SHORTER
    than it was when read is not that file, whatever its inode says.
    """

    segment: Path
    dev: int
    ino: int
    size: int
    record: OpenedRecord | None


class SessionTree:
    """The scanner and its per-unit head cache. BLOCKING: it lists a directory
    and stats every unit, so call it off the event loop.

    One instance per reader (the System page's sampler owns one). The cache is
    keyed by unit directory name and validated per scan against the segment's
    path, ``(st_dev, st_ino)`` and the append-only size floor: a segment that is
    gone, replaced, or shorter than when it was read is re-read, an untouched
    one costs a single ``stat``. A unit whose header has landed but whose first
    entry has not -- or is still being written -- is NOT cached, so a log caught
    between its create and its first append, or inside that append, is read
    again on the next scan.

    A scan admits at most :data:`TREE_UNIT_CAP` units: the live sessions' logs
    first (the caller names them), then the store's name order. The rest are
    enumerated -- the store walks every directory entry to count them -- but
    never retained, read or cached: the enumeration streams through a cap-sized
    heap, so the cache can hold one head per admitted unit and no more, whatever
    the store holds. The count lands in :attr:`omitted`, which every snapshot
    reports, so a tree missing them never reads as a tree that never had them;
    since live logs go first, what it misses is logs of closed sessions, which
    no row on screen nests on, unless the live rows alone exceed the cap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heads: dict[str, _Head] = {}
        self._omitted = 0

    @property
    def omitted(self) -> int:
        """Units past :data:`TREE_UNIT_CAP` on the latest scan -- counted, never
        read or stored. ``0`` until a scan has run."""
        return self._omitted

    def records(self, preferred: Iterable[str] = ()) -> list[OpenedRecord]:
        """One record per provable session log on disk, unordered.

        *preferred* names unit ids (the live sessions' ACP session ids) whose
        logs are admitted FIRST, whatever their place in the store's name order:
        the rows on screen are what the tree is folded for, so past the cap it
        is the logs of closed sessions that go unread, not a live row's -- as
        long as the live rows themselves fit in the cap. The rest of the cap is
        filled in name order (:func:`store.unit_dirs`, excluding what was
        already admitted), and what is left over is counted in :attr:`omitted`.
        *preferred* is itself bounded by the cap: the first that many are
        honoured, the rest fall into name order with everything else, so a
        gateway running more than the cap in live logged sessions does lose
        lineage on the rows past it.
        """
        out: list[OpenedRecord] = []
        seen: set[str] = set()
        with self._lock:
            # Live sessions' units first, by name: one stat each to find them.
            named: list[Path] = []
            for unit_id in preferred:
                if len(named) >= TREE_UNIT_CAP:
                    break
                directory = unit_dir_for(KIND_SESSION, unit_id)
                if directory is not None and directory.name not in seen:
                    seen.add(directory.name)
                    named.append(directory)
            # Then the store's order for the rest of the cap. The store streams
            # the listing through a cap-sized heap and counts the rest, so no
            # scan ever holds more than TREE_UNIT_CAP paths at once.
            listed, self._omitted = unit_dirs(
                KIND_SESSION, limit=TREE_UNIT_CAP - len(named), exclude=seen
            )
            for directory in listed:
                seen.add(directory.name)
            for directory in named + listed:
                record = self._read(directory)
                if record is not None:
                    out.append(record)
            # Evict what this scan did not admit: a unit that is gone, and one
            # that fell past the cap because the population grew in front of it.
            for gone in [name for name in self._heads if name not in seen]:
                del self._heads[gone]
        return out

    def _read(self, directory: Path) -> OpenedRecord | None:
        """One unit's record, from the cache when its segment is unchanged.
        Caller holds the lock."""
        name = directory.name
        segment = oldest_segment(directory)
        if segment is None:
            return None
        try:
            stat = segment.stat()
        except OSError:
            return None
        cached = self._heads.get(name)
        if (
            cached is not None
            and cached.segment == segment
            and cached.dev == stat.st_dev
            and cached.ino == stat.st_ino
            and cached.size <= stat.st_size
        ):
            return cached.record
        try:
            header, entry, announced = read_head(segment)
        except OSError:
            # The bytes were not seen, so there is no verdict to cache: a
            # moment's I/O fault, or a unit retention removed between the stat
            # and the open. The next scan reads it again, or finds it gone and
            # evicts it.
            return None
        if header is not None and not announced:
            # The create landed, the announce has not: nothing to cache.
            self._heads.pop(name, None)
            return None
        record = opened_record(directory, header, entry)
        self._heads[name] = _Head(segment, stat.st_dev, stat.st_ino, stat.st_size, record)
        return record

    def snapshot(self, preferred: Iterable[str] = ()) -> dict[str, TreeNode]:
        """The tree as of this scan: :func:`fold_tree` over :meth:`records`,
        with *preferred* (the live sessions' unit ids) admitted first.

        Never raises: a lineage read is decoration on the pages that show it,
        and a store fault must not take the page down. The fault is logged and
        the tree is reported empty, which every consumer renders as "no
        creator known", the same as before this reader existed.
        """
        try:
            return fold_tree(self.records(preferred))
        except Exception:  # pragma: no cover -- defensive; the store call sites are guarded
            logger.warning("session tree scan failed; reporting no lineage", exc_info=True)
            return {}


def parent_payload(
    node: TreeNode | None, live_key_of: Mapping[str, str], own_key: str
) -> dict[str, Any] | None:
    """The ``parent`` a session row carries on the wire, or ``None``.

    *live_key_of* maps a slot key -- bare, and in its full session-key spelling
    -- to the session key of a LIVE row. ``key`` is the creator's live session
    key when the creator is running and the edge can be followed, so a table
    nests the child under it exactly as it nests a task; it is ``None`` when the
    creator is not running (the child stays a root), when the node sits on a
    cycle, and when the citation would point at the row itself. ``slot`` is the
    child's own citation and survives all of those.
    """
    if node is None or node.parent_slot is None:
        return None
    parent_key = live_key_of.get(node.parent_slot)
    if node.cycle or parent_key == own_key:
        parent_key = None
    return {"slot": node.parent_slot, "key": parent_key}
