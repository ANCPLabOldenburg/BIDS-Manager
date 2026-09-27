"""Direct tests for the merge-aware commit policy (Phase F).

``_merge_commit`` is the per-subject commit used by convert: a new subject moves
in wholesale; an existing subject is merged file-by-file per ``on_existing``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.convert import (
    CollisionAbort,
    _log_existing_subject_summary,
    _merge_commit,
)


def _staging(tmp_path: Path, files: dict[str, str]) -> Path:
    s = tmp_path / "staging" / "sub-001"
    for rel, content in files.items():
        p = s / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return s


def _target(tmp_path: Path, files: dict[str, str]) -> Path:
    t = tmp_path / "out" / "sub-001"
    for rel, content in files.items():
        p = t / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return t


def _backups(target: Path) -> list[Path]:
    return list((target.parent / ".bidsmgr" / "backup").glob("sub-001_*"))


def test_new_subject_moves_in_wholesale(tmp_path: Path):
    s = _staging(tmp_path, {"anat/sub-001_T1w.nii.gz": "T1"})
    target = tmp_path / "out" / "sub-001"  # absent
    added, replaced, kept = _merge_commit(s, target, on_existing="skip")
    assert (added, replaced, kept) == (1, 0, 0)
    assert (target / "anat" / "sub-001_T1w.nii.gz").read_text() == "T1"


def test_skip_adds_new_keeps_colliding(tmp_path: Path):
    s = _staging(tmp_path, {
        "ses-post/anat/sub-001_ses-post_T1w.nii.gz": "new",   # new session
        "ses-pre/anat/sub-001_ses-pre_T1w.nii.gz": "STAGED",  # collides
    })
    target = _target(tmp_path, {"ses-pre/anat/sub-001_ses-pre_T1w.nii.gz": "EXISTING"})
    added, replaced, kept = _merge_commit(s, target, on_existing="skip")
    assert (added, replaced, kept) == (1, 0, 1)
    assert (target / "ses-post/anat/sub-001_ses-post_T1w.nii.gz").read_text() == "new"
    assert (target / "ses-pre/anat/sub-001_ses-pre_T1w.nii.gz").read_text() == "EXISTING"
    assert _backups(target) == []


def test_replace_backs_up_and_replaces_colliding(tmp_path: Path):
    s = _staging(tmp_path, {"anat/sub-001_T1w.nii.gz": "NEW"})
    target = _target(tmp_path, {"anat/sub-001_T1w.nii.gz": "OLD"})
    added, replaced, kept = _merge_commit(s, target, on_existing="replace")
    assert (added, replaced, kept) == (0, 1, 0)
    assert (target / "anat/sub-001_T1w.nii.gz").read_text() == "NEW"
    backups = _backups(target)
    assert len(backups) == 1
    assert (backups[0] / "anat/sub-001_T1w.nii.gz").read_text() == "OLD"


def test_update_keeps_identical_replaces_changed(tmp_path: Path):
    s = _staging(tmp_path, {
        "anat/same.nii.gz": "SAME",
        "anat/diff.nii.gz": "NEW",
    })
    target = _target(tmp_path, {
        "anat/same.nii.gz": "SAME",   # identical -> kept
        "anat/diff.nii.gz": "OLD",    # different -> replaced
    })
    added, replaced, kept = _merge_commit(s, target, on_existing="update")
    assert (added, replaced, kept) == (0, 1, 1)
    assert (target / "anat/diff.nii.gz").read_text() == "NEW"
    assert (target / "anat/same.nii.gz").read_text() == "SAME"
    # Only the changed file is backed up.
    backups = _backups(target)
    assert len(backups) == 1
    assert (backups[0] / "anat/diff.nii.gz").read_text() == "OLD"
    assert not (backups[0] / "anat/same.nii.gz").exists()


def test_error_aborts_without_moving_anything(tmp_path: Path):
    s = _staging(tmp_path, {
        "ses-post/new.nii.gz": "new",        # would be a safe add
        "ses-pre/clash.nii.gz": "STAGED",    # collides
    })
    target = _target(tmp_path, {"ses-pre/clash.nii.gz": "EXISTING"})
    with pytest.raises(CollisionAbort):
        _merge_commit(s, target, on_existing="error")
    # Nothing moved: existing untouched, the safe-add NOT committed.
    assert (target / "ses-pre/clash.nii.gz").read_text() == "EXISTING"
    assert not (target / "ses-post/new.nii.gz").exists()


def test_error_proceeds_when_no_collision(tmp_path: Path):
    s = _staging(tmp_path, {"ses-post/new.nii.gz": "new"})
    target = _target(tmp_path, {"ses-pre/old.nii.gz": "old"})
    added, replaced, kept = _merge_commit(s, target, on_existing="error")
    assert (added, replaced, kept) == (1, 0, 0)
    assert (target / "ses-post/new.nii.gz").read_text() == "new"
    assert (target / "ses-pre/old.nii.gz").read_text() == "old"


def test_preflight_summary_flags_existing_subjects(tmp_path: Path, caplog):
    bids_root = tmp_path / "ds"
    (bids_root / "sub-001").mkdir(parents=True)  # already present
    df = pd.DataFrame({"participant_id": ["sub-001", "sub-002"]})
    with caplog.at_level(logging.INFO, logger="bidsmgr.cli.convert"):
        _log_existing_subject_summary(bids_root, df, "skip")
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "already present" in msg and "sub-001" in msg
    assert "1 new" in msg  # sub-002 is new
