"""Tests for ``bidsmgr.util.paths`` — cross-platform path helpers.

The conversion pipeline composes paths from inventory data
(subject id, session label, dataset slug, series UID). Windows
rejects several characters at the syscall level that POSIX accepts;
these helpers are the single sanitisation site. Regressions here
re-introduce ``WinError 123`` on the Windows GUI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bidsmgr.util.paths import (
    WINDOWS_RESERVED_CHARS,
    WINDOWS_RESERVED_NAMES,
    long_path,
    long_path_for_tree,
    safe_path_component,
)


# ---------------------------------------------------------------------------
# safe_path_component
# ---------------------------------------------------------------------------


class TestSafePathComponent:
    """Each guarantee documented on :func:`safe_path_component`."""

    def test_already_safe_input_is_unchanged(self) -> None:
        assert safe_path_component("sub-001") == "sub-001"
        assert safe_path_component("ses-pre") == "ses-pre"
        assert safe_path_component("anat") == "anat"

    @pytest.mark.parametrize("ch", sorted(WINDOWS_RESERVED_CHARS))
    def test_each_reserved_char_is_replaced(self, ch: str) -> None:
        out = safe_path_component(f"foo{ch}bar")
        assert ch not in out, f"{ch!r} survived sanitisation: {out!r}"
        # Information-preserving via the hash suffix.
        assert "_" in out

    def test_pipe_pair_from_fmap_collapse(self) -> None:
        # The exact pattern that wedged the Windows GUI.
        joined = (
            "1.3.12.2.1107.5.2.43.66080.2025052611251937202010812.0.0.0"
            "|"
            "1.3.12.2.1107.5.2.43.66080.2025052611251937202710813.0.0.0"
        )
        out = safe_path_component(joined)
        assert "|" not in out
        assert not any(c in out for c in WINDOWS_RESERVED_CHARS)

    def test_control_chars_replaced(self) -> None:
        out = safe_path_component("ses\x00bad")
        assert "\x00" not in out

    def test_trailing_dot_dropped(self) -> None:
        out = safe_path_component("trailing.")
        assert not out.endswith(".")

    def test_trailing_space_dropped(self) -> None:
        out = safe_path_component("name ")
        assert not out.endswith(" ")

    @pytest.mark.parametrize("name", sorted(WINDOWS_RESERVED_NAMES))
    def test_reserved_device_names_disambiguated(self, name: str) -> None:
        out = safe_path_component(name)
        assert out.upper().split(".", 1)[0] not in WINDOWS_RESERVED_NAMES

    def test_reserved_device_names_case_insensitive(self) -> None:
        # ``con``, ``Con`` etc. are all NUL-eaten by Windows too.
        out = safe_path_component("con")
        assert out.upper().split(".", 1)[0] not in WINDOWS_RESERVED_NAMES

    def test_length_capped(self) -> None:
        out = safe_path_component("a" * 500, max_len=64)
        assert len(out) <= 64

    def test_empty_input_falls_back(self) -> None:
        assert safe_path_component("") == "unnamed"
        assert safe_path_component("", fallback="x") == "x"

    def test_deterministic_for_same_input(self) -> None:
        a = safe_path_component("foo|bar")
        b = safe_path_component("foo|bar")
        assert a == b

    def test_collision_resistance_via_hash(self) -> None:
        # Two distinct illegal inputs that would collide under a naive
        # ``str.replace('|', '_')`` strategy must produce distinct
        # outputs once disambiguation kicks in.
        a = safe_path_component("foo|bar")
        b = safe_path_component("foo_bar")
        assert a != b

    def test_idempotent_on_already_clean_output(self) -> None:
        once = safe_path_component("sub-001")
        twice = safe_path_component(once)
        assert once == twice


# ---------------------------------------------------------------------------
# long_path
# ---------------------------------------------------------------------------


class TestLongPath:
    """Only behaviour change on Windows; POSIX is a pass-through."""

    def test_posix_passthrough(self) -> None:
        if os.name == "nt":  # pragma: no cover — Windows-specific
            pytest.skip("POSIX-only behaviour")
        assert long_path("/tmp/foo") == "/tmp/foo"

    def test_short_windows_path_unchanged(self) -> None:
        # The function should not slap ``\\?\`` on every Windows
        # path — only those approaching MAX_PATH. We can exercise
        # the helper directly via its string-only branch.
        from bidsmgr.util import paths as paths_mod
        old_name = paths_mod.os.name
        paths_mod.os.name = "nt"
        try:
            short = "C:\\Users\\foo"
            assert not long_path(short).startswith("\\\\?\\")
        finally:
            paths_mod.os.name = old_name

    def test_long_windows_path_gets_prefix(self) -> None:
        from bidsmgr.util import paths as paths_mod
        old_name = paths_mod.os.name
        paths_mod.os.name = "nt"
        try:
            long_p = "C:\\" + "x" * 260
            out = long_path(long_p)
            assert out.startswith("\\\\?\\"), out
        finally:
            paths_mod.os.name = old_name

    def test_unc_long_path_gets_unc_prefix(self) -> None:
        from bidsmgr.util import paths as paths_mod
        old_name = paths_mod.os.name
        paths_mod.os.name = "nt"
        try:
            unc = "\\\\server\\share\\" + "y" * 260
            out = long_path(unc)
            assert out.startswith("\\\\?\\UNC\\"), out
        finally:
            paths_mod.os.name = old_name

    def test_already_prefixed_is_not_doubled(self) -> None:
        from bidsmgr.util import paths as paths_mod
        old_name = paths_mod.os.name
        paths_mod.os.name = "nt"
        try:
            already = "\\\\?\\C:\\foo"
            assert long_path(already) == already
        finally:
            paths_mod.os.name = old_name


# ---------------------------------------------------------------------------
# long_path_for_tree
# ---------------------------------------------------------------------------


class TestLongPathForTree:
    r"""A directory handed to something that will WALK it.

    ``long_path`` measures the string it is given, which is right for a path
    about to be opened and wrong for a directory whose CHILDREN will be. A
    200-character folder of Siemens DICOMs sits under the 248 threshold and is
    handed over unprefixed, while its ~90-character filenames put every child
    at 290.

    That is what silently disabled the BidsGuess classifier on Windows:
    dcm2niix reported ``rc=2`` "Unable to find any DICOM images", which reads
    as "this is not DICOM" rather than "this path is too long", so every series
    fell through to the regex layer and a gradient fieldmap came back
    ``phasediff`` where macOS and Linux said ``magnitude1``. Measured on the
    reporting dataset, same files and same binary: 0 sidecars from the
    290-character path, 15 from a short one.

    Shortening the output basename to ``%s`` does not help — that is the other
    end of the command.
    """

    def test_no_op_off_windows(self) -> None:
        if os.name == "nt":
            pytest.skip("POSIX-only behaviour")
        assert long_path_for_tree("/data/study/sub-01") == "/data/study/sub-01"

    def test_posix_output_is_identical_to_the_old_expression(self) -> None:
        """The requirement this fix was given: macOS and Linux must not move.

        The call site used to pass ``str(long_path(d))`` and now passes
        ``long_path_for_tree(d)``. Off Windows both reduce to ``str(d)``, so
        the argument handed to dcm2niix is the same string it always was.
        """
        if os.name == "nt":
            pytest.skip("POSIX-only behaviour")
        for d in (
            "/data/study/sub-01",
            "/data/study/sub-01/ses-pre",
            "/" + "x" * 300,
            "relative/path",
        ):
            assert long_path_for_tree(d) == str(long_path(d)) == str(d)

    def test_prefixes_even_a_short_directory_on_windows(self, tmp_path) -> None:
        """The whole point: the directory being short proves nothing about the
        files inside it, so there is no length here worth measuring."""
        if os.name != "nt":
            pytest.skip("Windows-only behaviour")
        short = tmp_path / "s"
        short.mkdir()
        assert long_path_for_tree(short).startswith("\\\\?\\")
        # ...and this is precisely where the old helper declined to act.
        assert not long_path(short).startswith("\\\\?\\")

    def test_does_not_double_wrap(self) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only behaviour")
        already = "\\\\?\\C:\\data\\study"
        assert long_path_for_tree(already) == already

    def test_the_prefixed_path_still_names_the_same_directory(
        self, tmp_path,
    ) -> None:
        """A prefix that changed which directory is meant would be worse than
        the bug it fixes. Read a file back through it."""
        if os.name != "nt":
            pytest.skip("Windows-only behaviour")
        (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
        prefixed = long_path_for_tree(tmp_path)
        assert Path(prefixed + "\\marker.txt").read_text(encoding="utf-8") == "here"

    def test_normalises_before_prefixing(self, tmp_path) -> None:
        r"""``\\?\`` turns OFF path normalisation, so anything carrying ``..``
        or forward slashes has to be normalised first, or the prefix preserves
        the mess and the path stops resolving."""
        if os.name != "nt":
            pytest.skip("Windows-only behaviour")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        out = long_path_for_tree(str(tmp_path) + "/a/../a/b")
        assert ".." not in out
        assert "/" not in out[4:], out
        assert Path(out).is_dir()

    def test_unc_paths_use_the_unc_form(self) -> None:
        """Checked on every platform: pure string work, and getting it wrong
        would send a network dataset to a path that does not exist."""
        from bidsmgr.util.paths import _prefixed

        assert _prefixed("C:\\data\\study") == "\\\\?\\C:\\data\\study"
        assert (
            _prefixed("\\\\server\\share\\study")
            == "\\\\?\\UNC\\server\\share\\study"
        )


def test_the_posix_branch_returns_the_path_untouched(monkeypatch) -> None:
    """Runs on Windows too, so the POSIX guarantee is checked everywhere.

    The requirement attached to this fix was that macOS and Linux must not
    move. Both helpers answer ``str(path)`` off Windows, so the argument the
    classifier builds is byte-identical to what it was before.

    ``os`` is swapped on the module rather than ``os.name`` being assigned,
    because ``pathlib`` reads that attribute to choose between ``PosixPath``
    and ``WindowsPath``: patching it globally makes ``Path()`` raise for the
    rest of the session.
    """
    from types import SimpleNamespace

    from bidsmgr.util import paths as paths_mod

    monkeypatch.setattr(paths_mod, "os", SimpleNamespace(name="posix"))
    for d in (
        "/data/study/sub-01",
        "/data/study/sub-01/ses-pre/anat",
        "/" + "x" * 300,
    ):
        assert paths_mod.long_path_for_tree(d) == str(d)
        assert paths_mod.long_path(d) == str(d)
        # The old call-site expression and the new one, side by side.
        assert paths_mod.long_path_for_tree(d) == str(paths_mod.long_path(d))


def test_the_classifier_asks_for_tree_form_on_its_input(monkeypatch) -> None:
    """Pins the wiring, which is the half that was missing.

    The BidsGuess classifier is the one dcm2niix call site that reads the
    user's own tree instead of a staging directory we named, so it is the one
    that has to ask for tree form. Its OUTPUT directory correctly stays on
    ``long_path``: we control the names written there.
    """
    from bidsmgr.classifier import dcm2niix_bidsguess as bg

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(bg.subprocess, "run", fake_run)
    source, out = Path("/data/study/sub-01"), Path("/tmp/out")
    bg._run_dcm2niix_sidecars(source, out, dcm2niix_bin=Path("dcm2niix"))

    cmd = captured["cmd"]
    assert cmd[-1] == long_path_for_tree(source)
    assert cmd[cmd.index("-o") + 1] == str(long_path(out))
    # The UID is no longer spent on the filename either.
    assert cmd[cmd.index("-f") + 1] == "%s"
