"""A step that writes for itself (the metadata step) as one undoable step.

The Editor's Fix ups ran the metadata engine outside the history: what it
wrote, placeholders, participants.tsv, the scans tables, could not be undone
though the window said everything could.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.metadata.engine import run_metadata
from bidsmgr.project.operations import peek_undo, read_log, run_as_operation, undo_last


def _state(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".bidsmgr" not in p.parts}


def _folders(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_dir() and ".bidsmgr" not in p.parts)


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for ses in ("pre", "post"):
        anat = root / "sub-01" / f"ses-{ses}" / "anat"
        anat.mkdir(parents=True)
        (anat / f"sub-01_ses-{ses}_T1w.nii.gz").write_bytes(b"\x1f\x8b image")
        (anat / f"sub-01_ses-{ses}_T1w.json").write_text(json.dumps({"EchoTime": 0.003}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    return root


def test_the_metadata_step_is_one_undo(ds: Path) -> None:
    before, folders = _state(ds), _folders(ds)
    run_as_operation(ds, "Write TODO placeholders",
                     lambda: run_metadata(ds, fill_scope="recommended", write_report=False))
    after = _state(ds)
    assert after != before
    assert "sub-01/sub-01_sessions.tsv" in after and "participants.tsv" in after
    assert peek_undo(ds).label == "Write TODO placeholders"
    assert len(read_log(ds)) == 1, "one step, however many files"
    undo_last(ds)
    assert _state(ds) == before and _folders(ds) == folders


def test_a_failing_step_leaves_nothing_behind(ds: Path) -> None:
    before = _state(ds)

    def half_then_fail():
        (ds / "participants.tsv").write_text("participant_id\nsub-01\n")
        (ds / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json").write_text("{}")
        (ds / "phenotype").mkdir()
        (ds / "phenotype" / "x.tsv").write_text("participant_id\n")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        run_as_operation(ds, "x", half_then_fail)
    assert _state(ds) == before
    assert not (ds / "phenotype").exists()
    assert read_log(ds) == []


def test_a_step_that_changes_nothing_adds_no_history(ds: Path) -> None:
    assert run_as_operation(ds, "nothing", lambda: 42) == 42
    assert read_log(ds) == []
