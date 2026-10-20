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
# fetch_latest
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


def test_fetch_latest_returns_info_version(monkeypatch):
    payload = {"info": {"version": "9.9.9"}, "releases": {"9.9.9": []}}

    def fake_urlopen(req, timeout=None, context=None):
        return _FakeResponse(payload)

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen", fake_urlopen)
    assert updater.fetch_latest() == updater.Latest(version="9.9.9", newest="9.9.9")


def test_fetch_latest_returns_none_on_network_error(monkeypatch):
    """A broken network must surface as ``None``, never raise."""

    def boom(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen", boom)
    assert updater.fetch_latest() is None


def test_fetch_latest_returns_none_on_malformed_json(monkeypatch):
    class BadResp:
        def read(self):
            return b"not-json-{"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("bidsmgr.gui.updater.urlopen",
                        lambda *a, **k: BadResp())
    assert updater.fetch_latest() is None


def test_fetch_latest_returns_none_when_info_missing(monkeypatch):
    """PyPI response lacking ``info.version`` is treated as 'unknown'."""

    monkeypatch.setattr(
        "bidsmgr.gui.updater.urlopen",
        lambda *a, **k: _FakeResponse({"releases": {}}),
    )
    assert updater.fetch_latest() is None


# ---------------------------------------------------------------------------
# latest_from: which release this Python can install
#
# Offering a release the environment's Python cannot run made the update
# report success, change nothing, and ask again at every start.
# ---------------------------------------------------------------------------


def _release(requires_python: str = "", yanked: bool = False) -> list[dict]:
    return [{"packagetype": "bdist_wheel", "requires_python": requires_python,
             "yanked": yanked}]


def _payload(newest: str, releases: dict[str, list[dict]]) -> dict:
    return {"info": {"version": newest}, "releases": releases}


def test_a_release_this_python_runs_is_offered_as_is():
    payload = _payload("1.6.0", {"1.5.0": _release(">=3.10"), "1.6.0": _release(">=3.10")})
    assert updater.latest_from(payload, "3.12.7") == updater.Latest("1.6.0", "1.6.0", "")


def test_a_release_needing_a_newer_python_names_it_and_offers_the_last_that_runs():
    payload = _payload("1.7.0", {
        "1.5.0": _release(">=3.10,<3.15"),
        "1.6.0": _release(">=3.10,<3.15"),
        "1.7.0": _release(">=3.13"),
    })
    latest = updater.latest_from(payload, "3.12.7")
    assert latest == updater.Latest(version="1.6.0", newest="1.7.0", needs_python="3.13")


def test_yanked_releases_and_prereleases_are_never_offered():
    payload = _payload("1.6.0", {
        "1.6.0": _release(">=3.13"),
        "1.5.2": _release(">=3.10", yanked=True),
        "1.5.3rc1": _release(">=3.10"),
        "1.5.1": _release(">=3.10"),
    })
    assert updater.latest_from(payload, "3.12.0").version == "1.5.1"


@pytest.mark.parametrize(
    ("requires", "python", "expected"),
    [(">=3.13", "3.12.7", "3.13"), (">=3.14", "3.12.7", "3.14"),
     (">=3.13.2", "3.12.7", "3.13"), ("<3.10", "3.12.7", "")],
)
def test_minimum_python_is_the_first_minor_version_that_fits(requires, python, expected):
    assert updater.minimum_python(requires, python) == expected


# ---------------------------------------------------------------------------
# plan_update
# ---------------------------------------------------------------------------


def test_an_installer_installation_moves_to_the_python_the_newest_release_needs():
    latest = updater.Latest(version="1.6.0", newest="1.7.0", needs_python="3.13")
    assert updater.plan_update(latest, "1.5.0", can_move_python=True) == (
        updater.UpdatePlan(version="1.7.0", move_to_python="3.13"))


def test_any_other_installation_is_offered_what_its_python_runs():
    latest = updater.Latest(version="1.6.0", newest="1.7.0", needs_python="3.13")
    assert updater.plan_update(latest, "1.5.0", can_move_python=False) == (
        updater.UpdatePlan(version="1.6.0"))


def test_nothing_is_offered_when_nothing_newer_runs_here():
    latest = updater.Latest(version="1.6.0", newest="1.7.0", needs_python="3.13")
    assert updater.plan_update(latest, "1.6.0", can_move_python=False) is None
    assert updater.plan_update(updater.Latest("1.6.0", "1.6.0"), "1.6.0",
                               can_move_python=True) is None


# ---------------------------------------------------------------------------
# find_installation: the installer's record, for this environment only
# ---------------------------------------------------------------------------


def _installation(tmp_path: Path, *, environment: Path | None = None,
                  uv_exists: bool = True) -> Path:
    install = tmp_path / "BIDS-Manager"
    env = install / "env"
    env.mkdir(parents=True)
    uv = install / "uv" / "uv"
    if uv_exists:
        uv.parent.mkdir()
        uv.write_text("")
    (install / "installation.json").write_text(json.dumps({
        "environment": str(environment or env),
        "uv": {"version": "0.13.1", "executable": str(uv)},
        "uv_environment": {"UV_NO_CONFIG": "1"},
    }))
    return env


def test_the_installer_record_beside_the_environment_is_found(tmp_path):
    env = _installation(tmp_path)
    found = updater.find_installation(env)
    assert found == updater.Installation(
        manifest=env.parent / "installation.json", uv=env.parent / "uv" / "uv")


def test_a_record_naming_another_environment_is_ignored(tmp_path):
    env = _installation(tmp_path, environment=tmp_path / "elsewhere")
    assert updater.find_installation(env) is None


def test_a_record_whose_uv_is_gone_is_ignored(tmp_path):
    env = _installation(tmp_path, uv_exists=False)
    assert updater.find_installation(env) is None


def test_no_record_means_no_installation(tmp_path):
    assert updater.find_installation(tmp_path / "env") is None


def test_the_update_log_of_an_installer_installation_stays_in_its_folder(tmp_path):
    env = _installation(tmp_path)
    found = updater.find_installation(env)
    assert updater.update_log_path(found) == env.parent / "update.log"
    assert updater.update_log_path(None) == Path.home() / ".bidsmgr" / "update.log"


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

    monkeypatch.setattr("bidsmgr.gui.updater.find_installation", lambda: None)
    monkeypatch.setattr("bidsmgr.gui.updater.subprocess.Popen", FakePopen)
    ok = updater.launch_update_helper(restart=True)
    assert ok is True

    cmd = captured["cmd"]
    # python -I <helper.py> --parent-pid <pid> --python <python> --package bids-manager --restart-cmd <argv0>
    assert cmd[0] == sys.executable or cmd[0] == "python"
    assert cmd[1] == "-I"
    assert cmd[2].endswith("_update_helper.py")
    assert "--parent-pid" in cmd
    assert "--package" in cmd
    pkg_idx = cmd.index("--package") + 1
    assert cmd[pkg_idx] == "bids-manager"
    assert "--restart-cmd" in cmd
    assert "--installation" not in cmd


def test_an_installer_installation_updates_through_its_record(monkeypatch, tmp_path):
    """The helper is told where the installer's record is, so it uses that uv;
    it runs on the Python the environment was built from, which a move to a
    newer Python can replace the environment around (Windows cannot delete a
    running program); and on macOS and Linux it logs inside the install folder."""
    env = _installation(tmp_path)
    installation = updater.find_installation(env)
    captured: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs

    monkeypatch.setattr("bidsmgr.gui.updater.find_installation", lambda: installation)
    monkeypatch.setattr("bidsmgr.gui.updater.subprocess.Popen", FakePopen)
    monkeypatch.setattr("bidsmgr.gui.updater.platform.system", lambda: "Darwin")
    assert updater.launch_update_helper(restart=False, move_to_python="3.13") is True

    cmd = captured["cmd"]
    assert cmd[0] == updater._base_python()
    assert cmd[1] == "-I"
    assert cmd[cmd.index("--installation") + 1] == str(installation.manifest)
    assert cmd[cmd.index("--move-to-python") + 1] == "3.13"
    assert Path(captured["kwargs"]["stdout"].name) == env.parent / "update.log"
    captured["kwargs"]["stdout"].close()


def test_only_an_installer_installation_can_move_to_another_python(monkeypatch):
    monkeypatch.setattr("bidsmgr.gui.updater.find_installation", lambda: None)
    monkeypatch.setattr("bidsmgr.gui.updater.subprocess.Popen",
                        lambda *a, **k: pytest.fail("nothing may be spawned"))
    assert updater.launch_update_helper(restart=False, move_to_python="3.13") is False


# ---------------------------------------------------------------------------
# The release-notes link in the update prompt
#
# Being asked to accept an update is exactly when somebody wants to know what
# is in it, and accepting closes the application to install itself, so the
# question has to be answerable before the dialog goes away.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected_anchor"),
    [
        ("1.2.6", "#v126"),
        ("1.2.5", "#v125"),
        ("1.2.4.2", "#v1242"),
        ("2.0.0", "#v200"),
    ],
)
def test_release_notes_url_anchors_at_the_version(version, expected_anchor):
    """The documentation numbers its release cards by the version with the dots
    dropped, so the link lands on the right card rather than the top of a long
    page."""
    from bidsmgr.gui.update_widgets import release_notes_url

    url = release_notes_url(version)
    assert url.endswith(expected_anchor)
    assert "bids_manager_documentation/updates/" in url


@pytest.mark.parametrize("version", ["1.2.6.dev0", "", "not-a-version", "1.2.6rc1"])
def test_release_notes_url_degrades_to_the_page(version):
    """A dev build or a local install has no card of its own. Opening the notes
    at the top beats sending somebody to a fragment that does not exist."""
    from bidsmgr.gui.update_widgets import release_notes_url

    assert release_notes_url(version).endswith("updates/")


def test_the_notes_url_points_at_a_page_the_documentation_actually_has():
    """Guards the two halves drifting apart: the anchor scheme lives here and
    the cards live in the documentation repository."""
    from bidsmgr.gui.update_widgets import release_notes_url

    assert release_notes_url("1.2.6") == (
        "https://ancplaboldenburg.github.io/bids_manager_documentation/"
        "updates/#v126"
    )


def test_the_sample_datasets_section_link_is_the_documentation_page():
    """Alongside the direct downloads, one link to the section that says what
    each dataset demonstrates, for somebody who does not know which they
    want."""
    from bidsmgr.gui.welcome_panel import _SAMPLE_DATASETS_URL

    assert _SAMPLE_DATASETS_URL == (
        "https://ancplaboldenburg.github.io/bids_manager_documentation/"
        "start/sample-data/"
    )


def test_every_dataset_the_documentation_offers_is_downloadable_from_the_gui():
    """The list here and the documentation's Sample data page drifted twice
    already: the panel had four while the documentation offered six, and in
    October 2026 every share was reissued, which left all six links here dead.
    Six distinct downloads, each a share link, is the shape that has to hold."""
    from bidsmgr.gui.welcome_panel import _SAMPLE_DATASETS

    assert len(_SAMPLE_DATASETS) == 6
    urls = [url for _label, url in _SAMPLE_DATASETS]
    assert len(set(urls)) == 6, "a duplicated link means one dataset is unreachable"
    assert all(u.startswith("https://cloud.uol.de/s/") for u in urls)
    assert all(u.endswith("/download") for u in urls)
    labels = [label for label, _url in _SAMPLE_DATASETS]
    assert len(set(labels)) == 6
    # Every modality the tool converts should be reachable from the Home tab.
    joined = " ".join(labels).lower()
    for modality in ("mri", "pet", "eeg", "meg", "multimodal"):
        assert modality in joined, modality
