"""The detached update helper, for installations made by the installer.

It runs after BIDS Manager has closed, with nobody watching, so what it must
never do is leave the installation worse than it found it: a failed move to
a newer Python puts the previous environment back, the installer's uv runs
with the installer's settings and none of the user's, and the installation's
record ends up describing what is really installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.gui import _update_helper as helper


def _record(tmp_path: Path, target: str = "aarch64-apple-darwin") -> dict:
    env = tmp_path / "BIDS-Manager" / "env"
    env.mkdir(parents=True)
    (env / "MARKER").write_text("the environment as it was")
    return {
        "environment": str(env),
        "installer": {"version": "2.0.1", "os": "macos", "target": target},
        "uv": {"version": "0.13.1", "executable": str(tmp_path / "uv")},
        "uv_environment": {"UV_NO_CONFIG": "1", "UV_CACHE_DIR": "/install/cache",
                           "UV_PYTHON_INSTALL_DIR": "/install/python",
                           "UV_MANAGED_PYTHON": "1"},
        "python": {"requested": "3.12", "version": "3.12.15",
                   "executable": str(env / "bin" / "python")},
        "package": {"name": "bids-manager", "requested": "bids-manager", "version": "1.5.0"},
    }


def test_the_installers_uv_gets_the_installers_settings_and_none_of_the_users():
    record = {"uv_environment": {"UV_NO_CONFIG": "1", "UV_CACHE_DIR": "/install/cache"}}
    base = {"PATH": "/usr/bin", "UV_INDEX_URL": "http://elsewhere", "UV_CACHE_DIR": "/mine",
            "VIRTUAL_ENV": "/some/env", "PYTHONPATH": "/x", "HOME": "/home/u"}
    env = helper.uv_environment(record, base)
    assert env == {"PATH": "/usr/bin", "HOME": "/home/u",
                   "UV_NO_CONFIG": "1", "UV_CACHE_DIR": "/install/cache"}


@pytest.mark.parametrize(
    ("target", "expected"),
    [("aarch64-apple-darwin", "cpython-3.13-macos-aarch64-none"),
     ("x86_64-apple-darwin", "cpython-3.13-macos-x86_64-none"),
     ("x86_64-unknown-linux-gnu", "cpython-3.13-linux-x86_64-gnu"),
     ("aarch64-unknown-linux-gnu", "cpython-3.13-linux-aarch64-gnu"),
     # Windows on ARM runs the x64 build, and a move keeps it there.
     ("x86_64-pc-windows-msvc", "cpython-3.13-windows-x86_64-none"),
     ("something-new", "3.13")],
)
def test_a_move_keeps_the_build_the_installer_chose(target, expected):
    assert helper.python_request({"installer": {"target": target}}, "3.13") == expected


def _fake_uv(monkeypatch, *, fail_on: str = "", removed: list | None = None):
    """Stand in for uv: ``venv`` creates the folder, and the step named in
    ``fail_on`` fails. Returns the list of calls; Pythons removed go to
    ``removed``."""
    calls = []
    monkeypatch.setattr(helper, "remove_pythons",
                        lambda record, requests: (removed if removed is not None else [])
                        .extend(sorted(requests)))

    def run_uv(record, args):
        calls.append(args)
        if args[0] == fail_on:
            return 1
        if args[0] == "venv":
            new = Path(args[-1])
            new.mkdir()
            (new / "NEW").write_text("a new environment")
        return 0

    monkeypatch.setattr(helper, "run_uv", run_uv)
    monkeypatch.setattr(helper, "_starts", lambda python: True)
    return calls


@pytest.mark.parametrize("failing_step", ["venv", "pip"])
def test_a_failed_move_puts_the_previous_environment_back(tmp_path, monkeypatch, failing_step):
    record = _record(tmp_path)
    env = Path(record["environment"])
    _fake_uv(monkeypatch, fail_on=failing_step)
    rc = helper.move_to_python(record, "3.13", record["python"]["executable"], "bids-manager")
    assert rc == 1
    assert (env / "MARKER").read_text() == "the environment as it was"
    assert not (env / "NEW").exists()
    assert not env.with_name("env.previous").exists()


def test_an_environment_that_does_not_start_is_also_rolled_back(tmp_path, monkeypatch):
    record = _record(tmp_path)
    env = Path(record["environment"])
    _fake_uv(monkeypatch)
    monkeypatch.setattr(helper, "_starts", lambda python: False)
    assert helper.move_to_python(record, "3.13", record["python"]["executable"], "x") == 1
    assert (env / "MARKER").exists()


def test_a_python_that_cannot_be_installed_touches_nothing(tmp_path, monkeypatch):
    record = _record(tmp_path)
    env = Path(record["environment"])
    calls = _fake_uv(monkeypatch, fail_on="python")
    assert helper.move_to_python(record, "3.13", record["python"]["executable"], "x") == 1
    assert (env / "MARKER").exists()
    assert [c[0] for c in calls] == ["python"]


def test_a_successful_move_replaces_the_environment_and_clears_the_old_one(tmp_path, monkeypatch):
    record = _record(tmp_path)
    env = Path(record["environment"])
    calls = _fake_uv(monkeypatch)
    assert helper.move_to_python(record, "3.13", record["python"]["executable"],
                                 "bids-manager") == 0
    assert (env / "NEW").exists() and not (env / "MARKER").exists()
    assert not env.with_name("env.previous").exists()
    assert calls[0][:3] == ["python", "install", "cpython-3.13-macos-aarch64-none"]
    assert calls[1][:1] == ["venv"] and "cpython-3.13-macos-aarch64-none" in calls[1]
    assert calls[2][:4] == ["pip", "install", "--upgrade", "--compile-bytecode"]


def test_the_record_describes_what_is_really_installed(tmp_path):
    """Read from the environment itself (here the test's own), never assumed."""
    import platform
    import sys
    from importlib.metadata import version

    record = _record(tmp_path)
    manifest = tmp_path / "installation.json"
    helper.record_update(manifest, record, sys.executable, moved_to="3.13")
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["package"]["version"] == version("bids-manager")
    assert written["python"]["version"] == platform.python_version()
    assert written["python"]["requested"] == "3.13"
    assert written["python"]["moved_by_update"] is True
    assert written["updated"]


@pytest.mark.parametrize(("move_to", "expected"),
                         [("", ["upgrade", "record"]), ("3.13", ["move", "record", "prune"])])
def test_update_installation_moves_only_when_asked_and_records_only_success(
        tmp_path, monkeypatch, move_to, expected):
    record = _record(tmp_path)
    manifest = tmp_path / "installation.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    done = []
    monkeypatch.setattr(helper, "upgrade_with_uv", lambda *a: done.append("upgrade") or 0)
    monkeypatch.setattr(helper, "move_to_python", lambda *a: done.append("move") or 0)
    monkeypatch.setattr(helper, "record_update", lambda *a: done.append("record"))
    monkeypatch.setattr(helper, "remove_pythons", lambda *a: done.append("prune"))
    assert helper.update_installation(manifest, "python", "bids-manager", move_to) == 0
    assert done == expected
    expected = expected[:1]

    done.clear()
    monkeypatch.setattr(helper, "upgrade_with_uv", lambda *a: done.append("upgrade") or 1)
    monkeypatch.setattr(helper, "move_to_python", lambda *a: done.append("move") or 1)
    assert helper.update_installation(manifest, "python", "bids-manager", move_to) == 1
    assert done == expected


def test_the_pythons_in_the_install_folder_are_named_as_uv_names_them(tmp_path):
    root = tmp_path / "python"
    for name in ("cpython-3.11-macos-aarch64-none", "cpython-3.11.17-macos-aarch64-none",
                 "cpython-3.13.16-macos-aarch64-none", ".lock", ".temp"):
        (root / name).mkdir(parents=True)
    record = {"uv_environment": {"UV_PYTHON_INSTALL_DIR": str(root)}}
    assert helper.installed_pythons(record) == {
        "cpython-3.11-macos-aarch64-none", "cpython-3.13-macos-aarch64-none"}


def test_a_failed_move_removes_the_python_it_downloaded(tmp_path, monkeypatch):
    record = _record(tmp_path)
    removed: list = []
    _fake_uv(monkeypatch, fail_on="pip", removed=removed)
    monkeypatch.setattr(helper, "installed_pythons", lambda record: set())
    helper.move_to_python(record, "3.13", record["python"]["executable"], "x")
    assert removed == ["cpython-3.13-macos-aarch64-none"]


def test_a_failed_move_keeps_a_python_that_was_already_there(tmp_path, monkeypatch):
    record = _record(tmp_path)
    removed: list = []
    _fake_uv(monkeypatch, fail_on="pip", removed=removed)
    monkeypatch.setattr(helper, "installed_pythons",
                        lambda record: {"cpython-3.13-macos-aarch64-none"})
    helper.move_to_python(record, "3.13", record["python"]["executable"], "x")
    assert removed == []


def test_after_a_move_only_the_new_python_stays(tmp_path, monkeypatch):
    record = _record(tmp_path)
    manifest = tmp_path / "installation.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    removed: list = []
    monkeypatch.setattr(helper, "move_to_python", lambda *a: 0)
    monkeypatch.setattr(helper, "record_update", lambda *a: None)
    monkeypatch.setattr(helper, "installed_pythons", lambda record: {
        "cpython-3.11-macos-aarch64-none", "cpython-3.12-macos-aarch64-none",
        "cpython-3.13-macos-aarch64-none"})
    monkeypatch.setattr(helper, "remove_pythons",
                        lambda record, requests: removed.extend(sorted(requests)))
    helper.update_installation(manifest, "python", "bids-manager", "3.13")
    assert removed == ["cpython-3.11-macos-aarch64-none", "cpython-3.12-macos-aarch64-none"]
