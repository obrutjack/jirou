"""The product-owned launch record, and what moving it out of ``cloud.json`` has to preserve.

The launch path's own profile, region and tag live in a file it owns, separate from the one an
operator hand-edits. These tests pin both halves of that separation: the record works on its
own, and an install whose pointer is still in the configuration file keeps resuming.
"""

from __future__ import annotations

import json

import pytest

from kiro_crew.cloud.config import DEFAULT_REGION, CloudConfig
from kiro_crew.cloud.launch_state import LaunchState, state_path


class TestTheLaunchRecord:
    """One writer, three fields, and no bytes of anyone else's in the file."""

    def test_a_record_round_trips(self, tmp_path):
        p = tmp_path / "cloud_launch_state.json"

        LaunchState.record(profile="work", region="eu-west-1", last_tag="kc-a1b2c3", path=p)

        state = LaunchState.load(p)
        assert (state.profile, state.region, state.last_tag) == ("work", "eu-west-1", "kc-a1b2c3")

    def test_it_is_written_where_the_crew_home_is(self, tmp_path, monkeypatch):
        """Under ``config_dir()``, so ``KIROCREW_HOME`` moves it with everything else."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))

        assert state_path().parent == tmp_path
        assert state_path().name == "cloud_launch_state.json"

    def test_the_record_is_frozen(self):
        """A caller cannot mutate it and believe the change reached disk.

        The shape this replaces was load-mutate-save, and the wizard did exactly that: it set
        three in-memory fields before the deploy so its progress output had the tag, and the
        write came minutes later. Frozen means the write is the only way to change the file,
        so there is no half-applied state to reason about.
        """
        with pytest.raises(Exception) as exc:
            LaunchState().profile = "nope"  # type: ignore[misc]
        assert "FrozenInstanceError" in type(exc.value).__name__

    @pytest.mark.parametrize(
        ("label", "blob"),
        [
            ("truncated", b'{"profile": "work"'),
            ("not utf-8", b"\xff\xfe not text"),
            ("not an object", b'["a", "list"]'),
            ("empty", b""),
        ],
    )
    def test_an_unusable_record_reads_as_unset_rather_than_raising(
        self, tmp_path, monkeypatch, label, blob
    ):
        """A cloud command must not hand an operator a traceback over a file it can ignore.

        Same tolerant policy the configuration reader has, for the same reason, and it matters
        more here: this file is the one every ``cloud`` subcommand reads to find the tag.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        p = tmp_path / "cloud_launch_state.json"
        p.write_bytes(blob)

        state = LaunchState.load(p)

        assert (state.profile, state.last_tag) == ("", ""), label

    def test_a_malformed_tag_reads_as_no_tag(self, tmp_path, monkeypatch):
        """Sanitised at the boundary, because the resume path's ``validate_tag`` raises.

        An empty tag already means "no last launch", which is a state every caller handles, so
        a malformed one is answered with that rather than carried inward.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        p = tmp_path / "cloud_launch_state.json"
        p.write_text(json.dumps({"profile": "w", "region": "us-east-1", "last_tag": "kc/../../x"}))

        assert LaunchState.load(p).last_tag == ""
        # The rest of the record still reads: one bad field is not a bad document.
        assert LaunchState.load(p).profile == "w"

    def test_an_absent_record_is_unset_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))

        state = LaunchState.load(tmp_path / "nothing-here.json")

        assert (state.profile, state.region, state.last_tag) == ("", DEFAULT_REGION, "")


class TestTheReadIsBoundedBeforeItAllocates:
    """An oversized record must never be materialised, only refused.

    A size check that runs AFTER the read is not a bound: by the time it can look at the
    length, the bytes are already in memory. And the file does not have to be written a byte at
    a time to be enormous -- ``truncate -s`` produces a sparse one instantly, which is within
    reach of anything with filesystem access.
    """

    def test_an_oversized_record_is_refused(self, tmp_path):
        from kiro_crew.cloud.launch_state import _MAX_FILE_BYTES

        p = tmp_path / "cloud_launch_state.json"
        # Sparse, so the test costs no disk: the file reports its full size and the read is
        # what must refuse to pull it in.
        with open(p, "wb") as fh:
            fh.truncate(_MAX_FILE_BYTES * 64)

        assert LaunchState.load(p).last_tag == ""

    def test_the_read_asks_for_no_more_than_the_ceiling(self, tmp_path, monkeypatch):
        """Pinned on the SIZE REQUESTED, which is the property under test.

        Asserting only that an oversized file reads as unset passes for an implementation that
        reads the whole thing and then discards it -- the exact shape being fixed. So this
        records what the reader asked the file for.
        """
        from kiro_crew.cloud.launch_state import _MAX_FILE_BYTES

        p = tmp_path / "cloud_launch_state.json"
        with open(p, "wb") as fh:
            fh.truncate(_MAX_FILE_BYTES * 64)
        asked: list = []

        real_open = open

        def _watching_open(path, *a, **k):
            fh = real_open(path, *a, **k)
            if str(path) == str(p):
                real_read = fh.read

                def _read(size=-1):
                    asked.append(size)
                    return real_read(size)

                fh.read = _read  # type: ignore[method-assign]
            return fh

        monkeypatch.setattr("builtins.open", _watching_open)
        LaunchState.load(p)

        assert asked, "the reader never read the file, so this measures nothing"
        # -1, or any size past the ceiling, means the whole file was requested.
        assert all(0 < n <= _MAX_FILE_BYTES + 1 for n in asked), asked

    def test_a_record_at_exactly_the_ceiling_still_reads(self, tmp_path):
        """The allowed size is allowed: a document exactly at the limit is not refused."""
        from kiro_crew.cloud.launch_state import _MAX_FILE_BYTES

        p = tmp_path / "cloud_launch_state.json"
        body = {"profile": "p", "region": "us-east-1", "last_tag": "kc-a1b2c3"}
        # Pad inside an ignored key up to exactly the ceiling.
        pad = _MAX_FILE_BYTES - len(json.dumps({**body, "_pad": ""}))
        text = json.dumps({**body, "_pad": "x" * pad})
        assert len(text.encode()) == _MAX_FILE_BYTES, len(text.encode())
        p.write_text(text)

        assert LaunchState.load(p).last_tag == "kc-a1b2c3"

    def test_an_oversized_file_is_refused_even_when_its_prefix_parses(self, tmp_path):
        """What the read's EXTRA byte is actually for.

        Reading exactly the ceiling cannot tell "the file is that long" from "the file is
        longer", so the length check passes on a truncated prefix and the prefix gets parsed.
        Usually a cut document fails to parse and the outcome looks the same -- which is why
        asserting on a file at the limit does not discriminate. Here the first ``_MAX_FILE_BYTES``
        bytes are a COMPLETE, valid record and the file continues past them, so reading one byte
        further is the only thing that refuses it instead of adopting the prefix.
        """
        from kiro_crew.cloud.launch_state import _MAX_FILE_BYTES

        p = tmp_path / "cloud_launch_state.json"
        body = {"profile": "p", "region": "us-east-1", "last_tag": "kc-prefix"}
        pad = _MAX_FILE_BYTES - len(json.dumps({**body, "_pad": ""}))
        prefix = json.dumps({**body, "_pad": "x" * pad})
        assert len(prefix.encode()) == _MAX_FILE_BYTES
        assert json.loads(prefix)["last_tag"] == "kc-prefix", "the prefix must be a valid record"
        p.write_bytes(prefix.encode() + b'\n{"more": "bytes past the ceiling"}\n')

        assert LaunchState.load(p).last_tag == "", "an oversized file's prefix was adopted"


class TestResumeStillReattachesAfterTheMove:
    """The one thing moving the pointer must not break.

    An install that launched before this file existed has its tag in ``cloud.json``. If the
    read stopped looking there, every such install would answer "no previous launch" and the
    operator's running instance would look gone.
    """

    def test_a_legacy_pointer_is_still_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        # Exactly what an install that launched under the old code has on disk: the pointer in
        # cloud.json, and no launch record at all.
        (tmp_path / "cloud.json").write_text(
            json.dumps({"profile": "work", "region": "eu-west-1", "last_tag": "kc-legacy"})
        )
        assert not (tmp_path / "cloud_launch_state.json").exists()

        state = LaunchState.load()

        assert (state.profile, state.region, state.last_tag) == ("work", "eu-west-1", "kc-legacy")

    def test_the_fallback_writes_nothing(self, tmp_path, monkeypatch):
        """Read-through, not a migration.

        Migrating by writing would reintroduce the write this change removes -- and it would
        write on a READ, so any ``cloud`` subcommand would touch the operator's file.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        legacy = json.dumps({"profile": "work", "region": "eu-west-1", "last_tag": "kc-legacy"})
        (tmp_path / "cloud.json").write_text(legacy)

        LaunchState.load()
        LaunchState.load()

        assert (tmp_path / "cloud.json").read_text() == legacy
        assert sorted(q.name for q in tmp_path.iterdir()) == ["cloud.json"]

    def test_the_record_wins_over_the_legacy_fields(self, tmp_path, monkeypatch):
        """Once a launch has written here, this file is the answer.

        Otherwise a stale pointer in ``cloud.json`` -- which nothing clears now, because
        nothing writes it -- would outrank the tag of the launch that actually happened.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "cloud.json").write_text(json.dumps({"last_tag": "kc-stale-legacy"}))
        LaunchState.record(profile="p", region="us-east-1", last_tag="kc-current")

        assert LaunchState.load().last_tag == "kc-current"

    def test_a_cleared_record_does_not_fall_back_to_the_legacy_tag(self, tmp_path, monkeypatch):
        """``destroy`` clearing the tag must STAY cleared.

        The fallback keys on the document being unusable, not on the tag being empty. Keying
        it on the tag would make a cleared pointer reappear from ``cloud.json`` and send the
        next command at a stack that was just deleted.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "cloud.json").write_text(json.dumps({"last_tag": "kc-legacy"}))
        LaunchState.record(profile="p", region="us-east-1", last_tag="kc-1")

        assert LaunchState.clear_tag("kc-1") is True

        assert LaunchState.load().last_tag == ""


class TestClearingThePointerIsOwnedByTheStackItNames:
    """``destroy`` owns the pointer for the stack it deleted, and for no other."""

    def test_a_matching_tag_is_cleared(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        LaunchState.record(profile="p", region="us-east-1", last_tag="kc-1")

        assert LaunchState.clear_tag("kc-1") is True
        assert LaunchState.load().last_tag == ""

    def test_a_tag_that_moved_on_is_left_alone(self, tmp_path, monkeypatch):
        """A launch that recorded its own tag between the delete and the clear keeps it.

        Cleared unconditionally, this command would wipe the pointer of a launch it never saw
        -- and the operator's new instance would look unlaunched.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        LaunchState.record(profile="p", region="us-east-1", last_tag="kc-2-launched-since")

        assert LaunchState.clear_tag("kc-1-being-destroyed") is False
        assert LaunchState.load().last_tag == "kc-2-launched-since"

    def test_the_other_fields_survive_the_clear(self, tmp_path, monkeypatch):
        """Only the tag is this command's to clear: the profile and region still name where
        the operator's other stacks live, and ``cloud list`` needs them."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        LaunchState.record(profile="work", region="eu-west-1", last_tag="kc-1")

        LaunchState.clear_tag("kc-1")

        state = LaunchState.load()
        assert (state.profile, state.region) == ("work", "eu-west-1")


class TestTheOperatorsFileIsNeverWritten:
    """The point of the whole move, asserted on the module rather than on one path."""

    def test_the_config_module_exposes_no_writer(self):
        """No ``save``, no ``apply_update``, no lock: there is nothing to call.

        A writer left in place with no caller is a writer the next change reaches for, and the
        two review findings this replaces were both about what that writer had to do when the
        operator's file was not in a state it could use.
        """
        import kiro_crew.cloud.config as config_mod

        assert not hasattr(CloudConfig, "save")
        assert not hasattr(CloudConfig, "apply_update")
        assert not hasattr(CloudConfig, "_replace_file")
        assert not hasattr(CloudConfig, "_merge_once")
        writers = [n for n in vars(config_mod) if "lock" in n.lower() or "writer" in n.lower()]
        assert writers == [], writers

    def test_nothing_in_the_module_can_write_a_file(self):
        """Pinned on the call expressions, not on a name search.

        ``"atomic_write" not in source`` passes for any spelling built at runtime. This asserts
        that no function in the module calls a writing primitive at all, which is the property
        that makes the operator's bytes safe without a seal.
        """
        import ast
        import inspect

        import kiro_crew.cloud.config as config_mod

        tree = ast.parse(inspect.getsource(config_mod))
        called = set()
        write_opens = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            rendered = ast.unparse(node.func)
            called.add(rendered)
            if rendered.endswith("open"):
                # `open` is legitimate here -- the reader uses it -- so the mode decides. A
                # missing mode is "r", and any mode carrying w, a, x or + writes.
                mode = ast.unparse(node.args[1]) if len(node.args) > 1 else "'r'"
                if any(ch in mode for ch in "wax+"):
                    write_opens.append(ast.unparse(node))
        forbidden = {
            "atomic_write",
            "os.replace",
            "os.rename",
            "os.remove",
            "os.unlink",
            "p.write_text",
            "p.write_bytes",
            "path.write_text",
            "shutil.move",
        }
        assert not (called & forbidden), sorted(called & forbidden)
        assert write_opens == [], write_opens

    def test_a_launch_and_a_destroy_leave_the_config_byte_identical(self, tmp_path, monkeypatch):
        """End to end over the record's own API, including a hand-edited block.

        The block is deliberately one the old writer would have had to make a decision about:
        it is valid JSON the operator is still filling in, which is the state that forced the
        choice between overwriting it and refusing.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        config = tmp_path / "cloud.json"
        hand_written = json.dumps(
            {"fargate": {"cluster": "crews", "subnets": ["subnet-a"]}}, indent=4
        )
        config.write_text(hand_written)

        LaunchState.record(profile="work", region="eu-west-1", last_tag="kc-1")
        LaunchState.clear_tag("kc-1")

        assert config.read_text() == hand_written
        # And the block is still readable as the operator wrote it.
        assert CloudConfig.load().fargate == {"cluster": "crews", "subnets": ["subnet-a"]}

    def test_an_unreadable_config_does_not_stop_a_launch_being_recorded(
        self, tmp_path, monkeypatch
    ):
        """A malformed ``cloud.json`` cannot stop a launch being recorded.

        The record's write path does not read that file at all, so there is nothing for a
        hand-edit to fail on -- which is the property that keeps a billed deploy from ending
        before sign-in over a document the launch never needed."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        broken = b"{ this is hand-edited and unparseable"
        (tmp_path / "cloud.json").write_bytes(broken)

        LaunchState.record(profile="work", region="eu-west-1", last_tag="kc-spent")

        assert LaunchState.load().last_tag == "kc-spent"
        assert (tmp_path / "cloud.json").read_bytes() == broken

    def test_the_record_is_not_added_to_any_seal_list(self):
        """Deliberate, and stated where someone would otherwise add it.

        It carries no input to a security decision. ``fargate.image`` is the field that
        chooses which container receives the model credential and it stays in ``cloud.json``;
        this file holds a pointer, and every command that acts on that pointer describes what
        it found before doing anything irreversible.
        """
        import kiro_crew.sandbox as sandbox_mod

        names = [n for n in vars(sandbox_mod) if n.startswith("_CREW_")]
        for name in names:
            value = getattr(sandbox_mod, name)
            if isinstance(value, (tuple, list)):
                assert "cloud_launch_state.json" not in value, name
