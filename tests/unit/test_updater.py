"""Unit tests for ``bidsmgr.gui.updater`` (Qt-free).

These tests cover the network + version-comparison helpers and the
helper-staging logic. They never import Qt and never hit the real PyPI
endpoint — every network call is monkey-patched to a controlled
substitute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from bidsmgr.gui import updater


# ---------------------------------------------------------------------------
# installed_version
# ---------------------------------------------------------------------------


def test_installed_version_matches_package():
    import bidsmgr
    assert updater.installed_version() == bidsmgr.__version__


# ---------------------------------------------------------------------------
# is_newer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latest,current,expected",
    [
        ("1.0.3", "1.0.2", True),
        ("1.0.2", "1.0.2", False),
        ("1.0.1", "1.0.2", False),
        ("2.0.0", "1.99.99", True),
        ("1.0.0rc1", "1.0.0", False),     # pre-release < final
        ("1.0.0", "1.0.0rc1", True),      # final > pre-release
        ("", "1.0.0", False),             # empty latest → cannot tell
        ("1.0.0", "", False),             # empty current → cannot tell
    ],
)
def test_is_newer_handles_pep440(latest, current, expected):
    assert updater.is_newer(latest, current) is expected


def test_is_newer_string_fallback_when_packaging_missing(monkeypatch):
    """When ``packaging.version`` is unimportable, falls back to
    equality-only string compare so identical versions still report
    ``False`` (no spurious update prompt).
    """
    # Force the ``from packaging.version import Version`` line in
    # ``is_newer`` to ImportError without breaking the rest of the test.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "packaging.version" or name.startswith("packaging."):
            raise ImportError("packaging unavailable for this test")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("builtins.__import__", fake_import)
        assert updater.is_newer("1.0.3", "1.0.2") is True
        assert updater.is_newer("1.0.2", "1.0.2") is False
        # The string-equality fallback can't reason about ordering, but
        # the equality case is what protects us from spurious nag.


# ---------------------------------------------------------------------------
# fetch_latest_pypi
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_latest_pypi_returns_info_version(monkeypatch):
    payload = {"info": {"version": "9.9.9"}, "releases": {"9.9.9": []}}

    def fake_urlopen(req, timeout=None, context=None):
        return _FakeResponse(payload)

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen", fake_urlopen)
    assert updater.fetch_latest_pypi() == "9.9.9"


def test_fetch_latest_pypi_returns_none_on_network_error(monkeypatch):
    """A broken network must surface as ``None``, never raise."""

    def boom(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen", boom)
    assert updater.fetch_latest_pypi() is None


def test_fetch_latest_pypi_returns_none_on_malformed_json(monkeypatch):
    class BadResp:
        def read(self):
            return b"not-json-{"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen",
                        lambda *a, **k: BadResp())
    assert updater.fetch_latest_pypi() is None


def test_fetch_latest_pypi_returns_none_when_info_missing(monkeypatch):
    """PyPI response lacking ``info.version`` is treated as 'unknown'."""

    monkeypatch.setattr(
        "bidsmgr.gui.updater.urlopen",
        lambda *a, **k: _FakeResponse({"releases": {}}),
    )
    assert updater.fetch_latest_pypi() is None


# ---------------------------------------------------------------------------
# is_editable_install
# ---------------------------------------------------------------------------


def test_is_editable_install_returns_bool():
    """The detection helper must always return a bool, never raise."""
    assert isinstance(updater.is_editable_install(), bool)


def test_is_editable_install_handles_missing_dist(monkeypatch):
    """When the distribution lookup fails we conservatively say 'not editable'."""
    from importlib import metadata

    def boom(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.distribution", boom)
    assert updater.is_editable_install() is False


# ---------------------------------------------------------------------------
# _copy_helper_to_temp
# ---------------------------------------------------------------------------


def test_copy_helper_to_temp_copies_a_real_file(tmp_path):
    """The staged helper must exist and contain the expected entry point."""
    dst = updater._copy_helper_to_temp()
    try:
        assert dst.exists()
        assert dst.name == "_update_helper.py"
        body = dst.read_text(encoding="utf-8")
        assert "def main()" in body
        assert "wait_for_parent" in body
    finally:
        # Best-effort cleanup; the helper's tempdir is otherwise leaked.
        try:
            dst.unlink()
            dst.parent.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# launch_update_helper (smoke: spawn path is mocked)
# ---------------------------------------------------------------------------


def test_launch_update_helper_returns_false_when_spawn_fails(monkeypatch):
    """A subprocess failure during spawn must surface as ``False``,
    not propagate. Callers branch on the bool to decide whether to
    quit the GUI.
    """
    def boom(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("bidsmgr.gui.updater.subprocess.Popen", boom)
    assert updater.launch_update_helper(restart=False) is False


def test_launch_update_helper_invokes_popen_with_helper_path(monkeypatch, tmp_path):
    """Happy path: helper is staged, Popen is called with the right args."""
    captured: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs

    monkeypatch.setattr("bidsmgr.gui.updater.subprocess.Popen", FakePopen)
    ok = updater.launch_update_helper(restart=True)
    assert ok is True

    cmd = captured["cmd"]
    # python <helper.py> --parent-pid <pid> --python <python> --package bids-manager --restart-cmd <argv0>
    assert cmd[0] == sys.executable or cmd[0] == "python"
    assert cmd[1].endswith("_update_helper.py")
    assert "--parent-pid" in cmd
    assert "--package" in cmd
    pkg_idx = cmd.index("--package") + 1
    assert cmd[pkg_idx] == "bids-manager"
    assert "--restart-cmd" in cmd
