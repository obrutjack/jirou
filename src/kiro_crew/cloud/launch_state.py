"""Where the LAUNCH's own state lives, separate from the operator's configuration.

``cloud.json`` is hand-edited: an operator writes the ``fargate`` block into it, and there is
no wizard step and no dashboard form for that block today. The launch path's own bookkeeping --
which profile and region the last launch used, and the tag naming the stack it created -- was
written back into that same file, and every collision between the two owners followed from
that one fact. A post-deploy write into a file a person may have left mid-edit has to choose
between overwriting their bytes and refusing; refusing aborts the command after the deploy is
already billed, and overwriting loses their block.

So the two owners get two files. This one is product-owned: nothing hand-edits it, the launch
path is the only writer, and no read of it feeds a security decision. ``cloud.json`` becomes
read-only to the product -- `CloudConfig` has no writer at all now -- so there is no
post-deploy write into the operator's file to refuse, nothing to clobber, and no cross-process
lock to hold while doing it.

**Not sealed against agent writes, deliberately.** It carries no input to a security decision.
``fargate.image`` is the field that chooses which container receives the model credential, and
it stays in ``cloud.json``; what lives here is a pointer. A rewritten pointer sends
``cloud status`` / ``cloud connect`` at a stack the operator can see in their own account, and
``cloud destroy`` prints the tag, describes the instance it found and asks before deleting
anything, so a substituted pointer is visible at the point it would matter. The credential
recipient is confirmed per launch regardless of what any file says.

**Read-through for existing installs.** An install that launched before this file existed has
its pointer in ``cloud.json``. :meth:`LaunchState.load` falls back to those fields when this
file has nothing, so ``kirocrew cloud resume`` re-attaches exactly as it did. The fallback is
read-only: nothing migrates the value by writing, because writing is the thing being removed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kiro_crew.atomic_write import atomic_write
from kiro_crew.cloud.config import DEFAULT_REGION, CloudConfig, tag_is_wellformed
from kiro_crew.config.loader import config_dir

logger = logging.getLogger(__name__)

#: The product-owned launch record, beside ``cloud.json`` in the crew home.
_FILENAME = "cloud_launch_state.json"

#: Same ceiling the configuration reader uses. This document holds three short strings, so
#: anything near it is not a record; the bound is here because a reader that trusts a file's
#: size is a reader an unbounded file can exhaust.
_MAX_FILE_BYTES = 64 * 1024


def state_path() -> Path:
    """The launch record's path, honouring ``KIROCREW_HOME`` through ``config_dir()``."""
    return config_dir() / _FILENAME


@dataclass(frozen=True)
class LaunchState:
    """What the launch path knows about the deployment it last created.

    Frozen: a caller that wants to change a field writes a new record through
    :meth:`record`, so there is no load-mutate-save shape to get wrong and no object whose
    in-memory edits silently fail to reach disk.
    """

    profile: str = ""
    region: str = DEFAULT_REGION
    last_tag: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LaunchState":
        """This install's launch record, from this file or from the legacy fields.

        Tolerant, like the configuration reader and for the same reason: a cloud command must
        not hand an operator a traceback over a file it can simply treat as unset. Every way
        the document can fail to be one answers the same thing -- there is no record here --
        and the legacy fields are then consulted, which is what keeps ``cloud resume`` working
        on an install that predates this file.
        """
        p = path or state_path()
        data = _read_document(p)
        if data is not None:
            tag = str(data.get("last_tag", ""))
            return cls(
                profile=str(data.get("profile", "")),
                region=str(data.get("region", "") or DEFAULT_REGION),
                # Sanitised at the boundary exactly as the configuration reader does it: a
                # malformed tag must not reach the resume path, where `validate_tag` raises,
                # and an empty one already means "no last launch".
                last_tag=tag if tag_is_wellformed(tag) else "",
            )
        legacy = CloudConfig.load()
        return cls(profile=legacy.profile, region=legacy.region, last_tag=legacy.last_tag)

    @classmethod
    def record(
        cls,
        *,
        profile: str,
        region: str,
        last_tag: str,
        path: Optional[Path] = None,
    ) -> None:
        """Write the launch record, replacing whatever was there.

        A whole-record write rather than a merge, because this file has ONE writer and three
        fields that a launch decides together: the profile and region it deployed into and the
        tag it created. No other party has fields here to lose, which is exactly why the
        launch path writes this file and not the operator's.
        """
        p = path or state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            p,
            json.dumps(
                {"profile": profile, "region": region, "last_tag": last_tag},
                indent=2,
            )
            + "\n",
        )

    @classmethod
    def clear_tag(cls, expect: str, path: Optional[Path] = None) -> bool:
        """Clear ``last_tag`` only while it still names *expect*. Answers whether it did.

        ``destroy`` owns this pointer for the stack it just deleted and for no other. Read and
        cleared unconditionally, a launch that recorded its own tag in between would have its
        pointer wiped by a command that never saw it.

        The compare and the write are two statements and there is no lock, which is the
        deliberate simplification this file buys: both writers are product-side commands a
        person runs one at a time, and the value at stake is a pointer rather than a
        configuration. `cloud list` enumerates the real stacks, so a pointer lost to that
        window is rediscoverable -- which is not true of bytes an operator hand-wrote.
        """
        p = path or state_path()
        current = cls.load(p)
        if current.last_tag != expect:
            return False
        cls.record(profile=current.profile, region=current.region, last_tag="", path=p)
        return True


def _read_document(p: Path) -> "Optional[dict]":
    """The file as a JSON object, or ``None`` for every way it is not one.

    Reads at most one byte PAST the ceiling and decides from that, so an oversized file is
    never allocated. Reading it whole and checking the length afterwards is not a bound: the
    allocation has already happened by the time the check can look at it, and the file does not
    even have to be written a byte at a time to be enormous -- ``truncate -s`` makes a sparse
    one instantly.

    The extra byte is what makes the length check mean "the file is longer than the ceiling"
    rather than "the read stopped at the ceiling". Reading exactly the ceiling cannot tell those
    apart, so an oversized file whose first ``_MAX_FILE_BYTES`` bytes happen to be a complete
    record would pass the check and be adopted -- a truncated prefix read as the whole
    document.
    """
    try:
        with open(p, "rb") as fh:
            raw = fh.read(_MAX_FILE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _MAX_FILE_BYTES:
        logger.warning("launch state: %s is larger than %d bytes; ignoring it", p, _MAX_FILE_BYTES)
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        logger.warning("launch state: %s is not a readable JSON document; ignoring it", p)
        return None
    return data if isinstance(data, dict) else None
