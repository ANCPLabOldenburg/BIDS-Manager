"""Unit tests for ``cli.scan._normalize_runs``.

The classifier chain emits classifications without any ``run`` entity (the
``run-N`` value dcm2niix copies from DICOM SeriesNumber is meaningless to
BIDS). ``_normalize_runs`` re-derives ``run-N`` per the BIDS semantic:
group rows by (subject, session, datatype, suffix, other-entities) within
a session; assign run-1, run-2, … only when a group has more than one row.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from bidsmgr.classifier.types import Classification
from bidsmgr.cli.scan import _normalize_runs
from bidsmgr.inventory.types import InventoryRow


def _make_row(*, subject="001", session="pre", series_desc="", acq_time="", uid="") -> InventoryRow:
    return InventoryRow(
        modality="mri",
        source=Path("/tmp/x"),
        subject_hint=subject,
        session_hint=session,
        series_description=series_desc,
        acq_time=acq_time or None,
        series_uid=uid or None,
    )


def _classify(row: InventoryRow, datatype: str, suffix: str, **entities) -> Classification:
    return Classification(
        row_id=row.row_id,
        classifier="test",
        datatype=datatype,
        suffix=suffix,
        candidate_entities=dict(entities),
        confidence=0.5,
    )


def test_singleton_drops_run():
    row = _make_row()
    c = _classify(row, "anat", "T1w", run="9")
    chosen = {row.row_id.hex: c}
    chosen = _normalize_runs([row], chosen)
    assert "run" not in chosen[row.row_id.hex].candidate_entities


def test_pair_assigns_run_1_and_2_in_acq_time_order():
    r1 = _make_row(acq_time="120000")
    r2 = _make_row(acq_time="120100")
    chosen = {
        r1.row_id.hex: _classify(r1, "func", "bold", task="rest"),
        r2.row_id.hex: _classify(r2, "func", "bold", task="rest"),
    }
    chosen = _normalize_runs([r1, r2], chosen)
    assert chosen[r1.row_id.hex].candidate_entities["run"] == "1"
    assert chosen[r2.row_id.hex].candidate_entities["run"] == "2"


def test_pair_uses_sequence_run_hint_when_acq_time_missing():
    """When DICOM AcquisitionTime is absent, fall back to the operator-supplied run hint."""
    r1 = _make_row(series_desc="task-foo_run-02_bold")
    r2 = _make_row(series_desc="task-foo_run-01_bold")
    chosen = {
        r1.row_id.hex: _classify(r1, "func", "bold", task="foo"),
        r2.row_id.hex: _classify(r2, "func", "bold", task="foo"),
    }
    chosen = _normalize_runs([r1, r2], chosen)
    # The row whose sequence text says run-01 must win run-1.
    assert chosen[r2.row_id.hex].candidate_entities["run"] == "1"
    assert chosen[r1.row_id.hex].candidate_entities["run"] == "2"


def test_different_tasks_dont_share_run_group():
    r1 = _make_row(acq_time="120000")
    r2 = _make_row(acq_time="120100")
    chosen = {
        r1.row_id.hex: _classify(r1, "func", "bold", task="rest"),
        r2.row_id.hex: _classify(r2, "func", "bold", task="movie"),
    }
    chosen = _normalize_runs([r1, r2], chosen)
    # Each is alone within its (task) group → no run.
    assert "run" not in chosen[r1.row_id.hex].candidate_entities
    assert "run" not in chosen[r2.row_id.hex].candidate_entities


def test_paired_bold_and_sbref_run_independently_but_consistently():
    """Bold and sbref are different suffixes → independent groups → independent
    runs. But they should still be ordered consistently when both pairs use the
    same ordering signal (sequence run hint)."""
    bold1 = _make_row(series_desc="task-foo_run-01_bold")
    bold2 = _make_row(series_desc="task-foo_run-02_bold")
    sbref1 = _make_row(series_desc="task-foo_run-01_sbref")
    sbref2 = _make_row(series_desc="task-foo_run-02_sbref")
    chosen = {
        bold1.row_id.hex: _classify(bold1, "func", "bold", task="foo"),
        bold2.row_id.hex: _classify(bold2, "func", "bold", task="foo"),
        sbref1.row_id.hex: _classify(sbref1, "func", "sbref", task="foo"),
        sbref2.row_id.hex: _classify(sbref2, "func", "sbref", task="foo"),
    }
    rows = [bold1, bold2, sbref1, sbref2]
    chosen = _normalize_runs(rows, chosen)
    assert chosen[bold1.row_id.hex].candidate_entities["run"] == "1"
    assert chosen[bold2.row_id.hex].candidate_entities["run"] == "2"
    assert chosen[sbref1.row_id.hex].candidate_entities["run"] == "1"
    assert chosen[sbref2.row_id.hex].candidate_entities["run"] == "2"
