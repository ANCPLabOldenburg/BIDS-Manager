"""The Phase 3 commit against a reader holding the staged tree open.

Windows refuses to move a directory while any process holds a handle anywhere
inside it, and enumerating a directory holds one. Conversion stages into
``<bids_root>/.tmp_bidsmgr/``, inside the dataset the GUI watches, so a tree
refresh was enough to fail the commit with ``WinError 5``. Measured before the
fix: with one background ``schema.warm_dataset`` pass running, 4 of 12 commits
failed.

The fix retries, and only on Windows. Half of this file exists to prove the
second half of that sentence: the POSIX branch must call the move exactly once
and let its exception out, which is what it did before the fix existed.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidsmgr.cli import convert
from bidsmgr.cli.convert import _merge_commit, _retry_while_held

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="the retry exists for a Windows file-locking rule"
)


def _as_platform(monkeypatch, name: str) -> list[float]:
    """Run the helper as though on ``name``; returns the sleeps it asked for.

    The whole ``os`` module is swapped rather than ``os.name`` alone: pathlib
    reads ``os.name`` at import to decide which Path flavour to build, so
    patching it in place breaks every Path in the process.
    """
    slept: list[float] = []
    monkeypatch.setattr(convert, "os", SimpleNamespace(name=name))
    monkeypatch.setattr(convert, "time", SimpleNamespace(
        monotonic=lambda: 0.0, sleep=slept.append,
    ))
    return slept


class _Reader:
    """A move that fails as Windows does while something is reading, then works."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise PermissionError(5, "Access is denied")


# ---------------------------------------------------------------------------
# macOS and Linux: exactly as before the fix
# ---------------------------------------------------------------------------

def test_the_posix_branch_moves_once_and_does_not_sleep(monkeypatch):
    slept = _as_platform(monkeypatch, "posix")
    move = _Reader(failures=0)

    _retry_while_held(move, Path("sub-001"))

    assert move.calls == 1
    assert slept == []


def test_the_posix_branch_lets_the_error_out_on_the_first_try(monkeypatch):
    slept = _as_platform(monkeypatch, "posix")
    move = _Reader(failures=1)

    with pytest.raises(PermissionError):
        _retry_while_held(move, Path("sub-001"))

    assert move.calls == 1, "POSIX must not retry: it never had this problem"
    assert slept == []


# ---------------------------------------------------------------------------
# Windows: retry while held, raise when stuck
# ---------------------------------------------------------------------------

def test_windows_retries_until_the_reader_lets_go(monkeypatch):
    slept = _as_platform(monkeypatch, "nt")
    move = _Reader(failures=3)

    _retry_while_held(move, Path("sub-001"))

    assert move.calls == 4
    assert len(slept) == 3


def test_windows_raises_the_real_error_once_the_deadline_passes(monkeypatch, caplog):
    monkeypatch.setattr(convert, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(convert, "_COMMIT_RETRY_DEADLINE_S", 0.05)
    monkeypatch.setattr(convert, "_COMMIT_RETRY_SLEEP_S", 0.01)
    move = _Reader(failures=10_000)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(PermissionError) as excinfo:
            _retry_while_held(move, Path("sub-001"))

    assert excinfo.value.errno or excinfo.value.args, "the original error, not a new one"
    assert move.calls > 1, "it should have tried more than once before giving up"
    assert "held open by another program" in caplog.text


def test_a_target_that_already_exists_is_not_retried(monkeypatch):
    """FileExistsError is a logic error, not a handle; waiting cannot help."""
    slept = _as_platform(monkeypatch, "nt")

    def move() -> None:
        raise FileExistsError(183, "Cannot create a file when it already exists")

    with pytest.raises(FileExistsError):
        _retry_while_held(move, Path("sub-001"))

    assert slept == []


# ---------------------------------------------------------------------------
# The real filesystem
# ---------------------------------------------------------------------------

def _staged(tmp_path: Path) -> Path:
    staging = tmp_path / ".tmp_bidsmgr" / "sub-002"
    for datatype in ("anat", "func"):
        (staging / datatype).mkdir(parents=True)
        for i in range(3):
            (staging / datatype / f"sub-002_run-{i}_{datatype}.nii.gz").write_bytes(b"\0" * 64)
            (staging / datatype / f"sub-002_run-{i}_{datatype}.json").write_text("{}")
    return staging


@windows_only
def test_a_commit_blocked_by_a_reader_lands_once_it_lets_go(tmp_path: Path):
    """The bug, end to end: a handle inside the tree, released mid-retry."""
    staging, target = _staged(tmp_path), tmp_path / "sub-002"
    held = open(staging / "anat" / "sub-002_run-0_anat.nii.gz", "rb")
    threading.Timer(0.4, held.close).start()

    try:
        added, replaced, kept = _merge_commit(staging, target, on_existing="skip")
    finally:
        held.close()

    assert (added, replaced, kept) == (12, 0, 0)
    assert len(list(target.rglob("*.nii.gz"))) == 6
    assert not staging.exists()


@windows_only
def test_a_commit_that_stays_blocked_commits_nothing(tmp_path: Path, monkeypatch):
    """Past the deadline the subject fails whole, never half-committed."""
    monkeypatch.setattr(convert, "_COMMIT_RETRY_DEADLINE_S", 0.2)
    staging, target = _staged(tmp_path), tmp_path / "sub-002"

    with open(staging / "anat" / "sub-002_run-0_anat.nii.gz", "rb"):
        with pytest.raises(PermissionError):
            _merge_commit(staging, target, on_existing="skip")

    assert not target.exists(), "a failed commit must leave the dataset alone"
    assert len(list(staging.rglob("*.nii.gz"))) == 6, "the staged tree is intact"
