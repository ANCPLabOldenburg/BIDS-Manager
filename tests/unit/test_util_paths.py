"""Tests for ``bidsmgr.util.paths`` — cross-platform path helpers.

The conversion pipeline composes paths from inventory data
(subject id, session label, dataset slug, series UID). Windows
rejects several characters at the syscall level that POSIX accepts;
these helpers are the single sanitisation site. Regressions here
re-introduce ``WinError 123`` on the Windows GUI.
"""

from __future__ import annotations

import os

import pytest

from bidsmgr.util.paths import (
    WINDOWS_RESERVED_CHARS,
    WINDOWS_RESERVED_NAMES,
    long_path,
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
