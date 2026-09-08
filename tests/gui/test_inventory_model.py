"""Tests for ``bidsmgr.gui.models.InventoryTableModel``.

Two layers:

* **Pure model tests** — exercise ``rowCount`` / ``columnCount`` /
  ``data`` / ``setData`` against an in-memory DataFrame. No view, no
  delegate, no QApplication required beyond what pytest-qt sets up.
* **Integration test** — render the model through a real ``QTableView``
  with the extracted delegates. Confirms the role roundtrip (model
  publishes data → delegate reads it from the index) works end to end.

Marked ``gui`` so they run under ``QT_QPA_PLATFORM=offscreen`` in the
same gate as the widget smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import QTableView

from bidsmgr.gui.delegates import (
    PAYLOAD_ROLE,
    ROW_STATE_ROLE,
    CellTextDelegate,
    CheckboxDelegate,
    StatusDelegate,
)
from bidsmgr.gui.models import COLUMNS, InventoryTableModel
from bidsmgr.project import (
    Project,
    ScanImported,
    UserSetCell,
    UserToggleInclude,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ok_row(**overrides) -> dict:
    """A valid MRI row that should pass through with row_state=''."""
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "modality_bids": "anat",
        "sequence": "t1w_mprage",
        "series_uid": "1.2.3.4",
        "proposed_datatype": "anat",
        "proposed_basename": "sub-001_ses-pre_T1w",
        "Proposed BIDS name": "anat/sub-001_ses-pre_T1w.nii.gz",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "anat",
        "bids_guess_suffix": "T1w",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre"},
            sort_keys=True,
        ),
        "task": "",
        "run": "",
        "source_file": "",
    }
    base.update(overrides)
    return base


def _func_row(**overrides) -> dict:
    return _ok_row(
        modality_bids="func",
        proposed_datatype="func",
        proposed_basename="sub-001_ses-pre_task-rest_bold",
        Proposed_BIDS_name="func/sub-001_ses-pre_task-rest_bold.nii.gz",
        bids_guess_datatype="func",
        bids_guess_suffix="bold",
        bids_guess_confidence="0.97",
        entities=json.dumps(
            {"subject": "001", "session": "pre", "task": "rest"},
            sort_keys=True,
        ),
        task="rest",
        **overrides,
    )


def _physio_row(**overrides) -> dict:
    return _ok_row(
        modality_bids="physio",
        proposed_datatype="func",
        proposed_basename="sub-002_ses-post_task-mb_physio",
        bids_guess_suffix="physio",
        bids_guess_confidence="0.99",
        series_uid="5.6.7.8",
        BIDS_name="sub-002",
        session="ses-post",
        task="mb",
        # func/physio requires a task entity; keep the entities JSON in sync
        # with the mirror cells so live validation reads the row as valid
        # (status "phys") rather than flagging a missing task.
        entities=json.dumps(
            {"subject": "002", "session": "post", "task": "mb"},
            sort_keys=True,
        ),
        **overrides,
    )


def _eeg_row(**overrides) -> dict:
    return _ok_row(
        modality="eeg",
        modality_bids="eeg",
        proposed_datatype="eeg",
        proposed_basename="sub-001_task-rest_eeg",
        bids_guess_datatype="eeg",
        bids_guess_suffix="eeg",
        source_file="/data/raw/sub-001/eeg/sub-001_task-rest_eeg.edf",
        series_uid="",
        entities=json.dumps(
            {"subject": "001", "task": "rest"}, sort_keys=True,
        ),
        task="rest",
        **overrides,
    )


def _err_row(**overrides) -> dict:
    return _ok_row(
        proposed_basename="",
        proposed_datatype="",
        proposed_issues="task entity required for func/bold",
        bids_guess_confidence="0.61",
        bids_guess_suffix="bold",
        series_uid="9.10.11.12",
        BIDS_name="sub-003",
        session="",
        **overrides,
    )


def _skip_row(**overrides) -> dict:
    return _ok_row(
        include=0,
        bids_guess_skip=True,
        proposed_basename="localizer_20ch_head-coil",
        proposed_datatype="",
        bids_guess_confidence="",
        series_uid="13.14.15",
        **overrides,
    )


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_row_and_column_counts() -> None:
    df = make_df([_ok_row(), _func_row()])
    model = InventoryTableModel(df)
    assert model.rowCount() == 2
    assert model.columnCount() == len(COLUMNS)


def test_header_labels_match_spec() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    for col, spec in enumerate(COLUMNS):
        assert model.headerData(col, Qt.Orientation.Horizontal) == spec.header


def test_origin_and_format_are_shown_by_default() -> None:
    """Where a row came from and what it was read from are both visible.

    They used to be neither: source_folder was correct in the TSV and absent
    from the table, and one header read "sequence / source" while carrying only
    the sequence name.
    """
    by_key = {c.key: c for c in COLUMNS}
    for key in ("source_folder", "format"):
        assert key in by_key, key
        assert by_key[key].default_visible, key
        assert by_key[key].df_column == key
    assert by_key["sequence"].header == "sequence"


# Metadata columns that only mean something for some datatypes. Editability is
# the column's own flag AND whether the row's datatype has any use for it.
_MODALITY_SCOPED = {"line_freq", "montage", "eeg_reference", "eeg_ground"}


def test_flags_only_editable_columns_carry_edit_flag() -> None:
    df = make_df([_ok_row()])  # an anat/T1w row
    model = InventoryTableModel(df)
    for col, spec in enumerate(COLUMNS):
        idx = model.index(0, col)
        editable = bool(model.flags(idx) & Qt.ItemFlag.ItemIsEditable)
        expected = spec.editable and spec.key not in _MODALITY_SCOPED
        assert editable == expected, f"column {spec.key!r} editable mismatch"


def test_modality_scoped_columns_are_editable_where_they_apply() -> None:
    """The other half: an anatomical scan has no power line frequency, but an
    EEG recording does, and the same column must accept the edit there."""
    df = make_df([_ok_row(
        modality="eeg", modality_bids="eeg", proposed_datatype="eeg",
        bids_guess_datatype="eeg", bids_guess_suffix="eeg",
        proposed_basename="sub-001_ses-pre_task-rest_eeg",
    )])
    model = InventoryTableModel(df)
    for key in ("line_freq", "montage", "eeg_reference", "eeg_ground"):
        col = next(i for i, c in enumerate(COLUMNS) if c.key == key)
        idx = model.index(0, col)
        assert bool(model.flags(idx) & Qt.ItemFlag.ItemIsEditable), key


def test_bulk_edit_does_not_reach_scanner_derivatives() -> None:
    """REGRESSION: a bulk edit put a line frequency on Siemens localisers.

    A scout, a PhoenixZIPReport and a TENSOR map are excluded from conversion
    and carry NO ``proposed_datatype``, only the classifier's guess. Reading
    the proposed value alone made them look like blank slates, and blank was
    being treated as "every column applies".
    """
    df = make_df([
        _ok_row(BIDS_name="sub-001", proposed_datatype="eeg", line_freq="",
                bids_guess_datatype="eeg", bids_guess_suffix="eeg",
                proposed_basename="sub-001_task-rest_eeg"),
        # A localiser: no proposed datatype, guess says anat, not converted.
        _ok_row(BIDS_name="sub-001", sequence="AAHead_Scout_64ch", line_freq="",
                proposed_datatype="", bids_guess_datatype="anat",
                bids_guess_suffix="localizer", include=0),
        # A scanner derivative routed away from the BIDS tree entirely.
        _ok_row(BIDS_name="sub-001", sequence="dwi_TENSOR", line_freq="",
                proposed_datatype="", bids_guess_datatype="derivatives",
                bids_guess_suffix="TENSOR", include=0),
    ])
    model = InventoryTableModel(df)

    changed = model.bulk_set([0, 1, 2], "line_freq", "60")
    out = model.dataframe()
    assert changed == 1, "only the EEG row should take a line frequency"
    assert out.at[0, "line_freq"] == "60"
    assert out.at[1, "line_freq"] == ""
    assert out.at[2, "line_freq"] == ""


def test_an_inapplicable_cell_renders_blank_not_inherited() -> None:
    """With a dataset default set, an MRI row used to display that default in
    all four electrophysiology columns. A blank says "not applicable"; the em
    dash the inheritance display uses would say "inherited and empty", which
    invites an edit that would never be written."""
    from bidsmgr.recording_meta import RecordingMetaSpec

    spec = RecordingMetaSpec()
    spec.defaults.power_line_freq = 50
    spec.defaults.montage = "standard_1020"

    model = InventoryTableModel(make_df([_ok_row()]))
    model.set_global_spec(spec)
    for key in ("line_freq", "montage"):
        col = next(i for i, c in enumerate(COLUMNS) if c.key == key)
        idx = model.index(0, col)
        assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "", key
        assert model.setData(idx, "60", Qt.ItemDataRole.EditRole) is False, key


# ---------------------------------------------------------------------------
# Display values
# ---------------------------------------------------------------------------


def test_id_strips_sub_prefix() -> None:
    df = make_df([_ok_row(BIDS_name="sub-042")])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "id")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "042"


def test_session_strips_ses_prefix_and_dashes_empty() -> None:
    # Row 1 has no session at all (entities lacks it) -> empty mirror cell.
    # The model mirrors entities -> cells on load, so a row whose entities
    # carry a session always shows it; only a genuinely session-less row
    # renders "—".
    df = make_df([
        _ok_row(session="ses-pre"),
        _ok_row(session="", entities=json.dumps({"subject": "001"}, sort_keys=True)),
    ])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "ses")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "pre"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == "—"


def test_confidence_formatted_without_leading_zero() -> None:
    df = make_df([_ok_row(bids_guess_confidence="0.97"),
                  _ok_row(bids_guess_confidence="0.71")])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "conf")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == ".97"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == ".71"


def test_backend_derivation() -> None:
    df = make_df([_ok_row(), _physio_row(), _eeg_row()])
    model = InventoryTableModel(df)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "backend")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "dcm2niix"
    assert model.data(model.index(1, col), Qt.ItemDataRole.DisplayRole) == "bidsphysio"
    assert model.data(model.index(2, col), Qt.ItemDataRole.DisplayRole) == "mne-bids"


# ---------------------------------------------------------------------------
# Row state / status badge
# ---------------------------------------------------------------------------


def test_row_state_for_valid_row_is_empty() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""


def test_row_state_err_for_missing_basename() -> None:
    df = make_df([_err_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "err"


def test_row_state_skip_for_excluded_rows() -> None:
    df = make_df([_skip_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


def test_status_kind_phys_for_physio_rows() -> None:
    df = make_df([_physio_row()])
    model = InventoryTableModel(df)
    status_col = next(i for i, c in enumerate(COLUMNS) if c.key == "status")
    assert model.data(model.index(0, status_col), PAYLOAD_ROLE) == "phys"


def test_status_kind_ok_for_valid_mri_row() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    status_col = next(i for i, c in enumerate(COLUMNS) if c.key == "status")
    assert model.data(model.index(0, status_col), PAYLOAD_ROLE) == "ok"


def test_row_state_noimg_for_nonimage_series() -> None:
    """A DERIVED non-image series (flagged by the scanner with the
    ``non-image series`` token) reads as ``noimg`` — taking priority over
    the ``skip`` it would otherwise get for being excluded (include=0)."""
    df = make_df([_ok_row(
        include=0,
        bids_guess_skip=True,
        proposed_issues=(
            "non-image series: the DICOM headers carry no pixel data "
            "(Rows/Columns absent), so dcm2niix cannot convert it. "
            "Excluded from conversion."
        ),
    )])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "noimg"


def test_status_kind_noimg_for_nonimage_series() -> None:
    df = make_df([_ok_row(
        include=0, bids_guess_skip=True,
        proposed_issues="non-image series: no pixel data; excluded.",
    )])
    model = InventoryTableModel(df)
    status_col = next(i for i, c in enumerate(COLUMNS) if c.key == "status")
    assert model.data(model.index(0, status_col), PAYLOAD_ROLE) == "noimg"


# ---------------------------------------------------------------------------
# Live entity-validation (recomputed on every edit)
# ---------------------------------------------------------------------------


def _task_col() -> int:
    return next(i for i, c in enumerate(COLUMNS) if c.key == "task")


def _func_row_missing_task(**overrides) -> dict:
    """A func/bold row whose entities lack the required task entity."""
    row = _func_row(**overrides)
    row["entities"] = json.dumps({"subject": "001", "session": "pre"}, sort_keys=True)
    row["task"] = ""
    return row


def test_live_validation_flags_func_bold_missing_task() -> None:
    """A func/bold row whose entities lack the required task reads as err
    at load (live validation runs on construction, not just on edit)."""
    df = make_df([_func_row_missing_task()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "err"


def test_live_validation_clears_error_when_user_supplies_task() -> None:
    """Editing the missing task to a real value re-validates the row to
    valid (the chips track the fix)."""
    df = make_df([_func_row_missing_task()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "err"
    assert model.setData(model.index(0, _task_col()), "rest") is True
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""


def test_live_validation_introduces_error_when_user_clears_task() -> None:
    """Clearing a required task on a valid row flips it back to err."""
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""
    assert model.setData(model.index(0, _task_col()), "") is True
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "err"


def test_revalidate_all_recomputes_states_after_external_change() -> None:
    """``revalidate_all`` picks up a change made directly to the DataFrame
    (an edit path that bypassed setData) -- the Re-validate button's engine."""
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    assert model.row_state(0) == ""
    # Mutate the underlying frame directly (no setData / refresh_row).
    model.dataframe().at[0, "task"] = ""
    model.dataframe().at[0, "entities"] = json.dumps(
        {"subject": "001", "session": "pre"}, sort_keys=True,
    )
    assert model.row_state(0) == ""  # still stale (no recompute yet)
    model.revalidate_all()
    assert model.row_state(0) == "err"  # now reflects the missing task


def test_live_validation_preserves_static_scan_notes() -> None:
    """Recomputing the validation segment must not drop static scan notes
    (suspected_abort, B0 reroute, collision hints, ...)."""
    note = "rerouted to fmap/epi: SeriesDescription contains a B0 marker"
    row = _func_row_missing_task()
    row["proposed_issues"] = note
    df = make_df([row])
    model = InventoryTableModel(df)
    # Loaded: static note + fresh "task required" error coexist.
    issues = model.dataframe().at[0, "proposed_issues"]
    assert note in issues and "required" in issues.lower()
    # After the user supplies the task, only the static note remains.
    model.setData(model.index(0, _task_col()), "rest")
    issues = model.dataframe().at[0, "proposed_issues"]
    assert issues == note


def test_tooltip_surfaces_proposed_issues() -> None:
    """Hovering any cell of a flagged row shows the scanner's reason."""
    df = make_df([_ok_row(
        proposed_issues="non-image series: no pixel data | trivial: few files",
    )])
    model = InventoryTableModel(df)
    tip = model.data(model.index(0, 1), Qt.ItemDataRole.ToolTipRole)
    assert tip is not None and "non-image series" in tip
    # `` | ``-joined notes are split onto separate lines for readability.
    assert "\n" in tip


def test_no_tooltip_when_no_issues() -> None:
    df = make_df([_ok_row(proposed_issues="")])
    model = InventoryTableModel(df)
    assert model.data(model.index(0, 1), Qt.ItemDataRole.ToolTipRole) is None


def test_include_payload_reads_dataframe() -> None:
    df = make_df([_ok_row(include=1), _ok_row(include=0)])
    model = InventoryTableModel(df)
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is True
    assert model.data(model.index(1, inc_col), PAYLOAD_ROLE) is False


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def test_setdata_rejects_non_editable_column() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    id_col = next(i for i, c in enumerate(COLUMNS) if c.key == "id")
    assert model.setData(model.index(0, id_col), "999") is False


def test_setdata_on_task_rebuilds_basename(qtbot) -> None:
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    # Edit task=rest → task=motor; basename should regenerate.
    with qtbot.waitSignal(model.dataChanged, timeout=500):
        assert model.setData(model.index(0, task_col), "motor") is True
    new_bn = model.data(model.index(0, bn_col), Qt.ItemDataRole.DisplayRole)
    assert "task-motor" in new_bn
    assert "task-rest" not in new_bn


def test_setdata_same_value_is_noop() -> None:
    df = make_df([_func_row()])
    model = InventoryTableModel(df)
    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    assert model.setData(model.index(0, task_col), "rest") is False


def test_editing_task_preserves_run_entity() -> None:
    """Regression: a fresh scan TSV carries entities in JSON but leaves the
    mirror cells (task / run / session) blank. Editing one mirror cell must
    not delete the others. Previously, changing ``task`` ran
    ``rebuild_from_columns`` which saw the blank ``run`` cell and dropped
    ``run`` from the entities, so ``sub-001_task-X_run-1_bold`` lost its run.
    """
    # Mirror cells deliberately blank, entities holds run+task (real scan shape).
    row = _ok_row(
        proposed_datatype="func",
        bids_guess_datatype="func",
        bids_guess_suffix="bold",
        proposed_basename="sub-001_task-dmaging_run-1_bold",
        task="",   # blank mirror cell, as a fresh scan TSV leaves it
        run="",
        session="",
        entities=json.dumps(
            {"subject": "001", "task": "dmaging", "run": "1"},
            sort_keys=True,
        ),
    )
    model = InventoryTableModel(make_df([row]))

    # The load pass should have mirrored entities → cells.
    assert model.entities(0).get("run") == "1"

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    assert model.setData(model.index(0, task_col), "newtask") is True

    new_bn = model.data(model.index(0, bn_col), Qt.ItemDataRole.DisplayRole)
    assert "task-newtask" in new_bn
    assert "run-1" in new_bn          # run entity survived the task edit
    assert model.entities(0).get("run") == "1"


def test_bulk_edit_task_preserves_run_across_rows() -> None:
    """The same guarantee via the bulk-edit path (``bulk_set``)."""
    rows = [
        _ok_row(
            proposed_datatype="func",
            bids_guess_datatype="func",
            bids_guess_suffix="bold",
            proposed_basename=f"sub-001_task-dmaging_run-{n}_bold",
            task="", run="", session="", series_uid=f"uid-{n}",
            entities=json.dumps(
                {"subject": "001", "task": "dmaging", "run": str(n)},
                sort_keys=True,
            ),
        )
        for n in (1, 2, 3)
    ]
    model = InventoryTableModel(make_df(rows))
    changed = model.bulk_set([0, 1, 2], "task", "memory")
    assert changed == 3
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    for r, n in zip((0, 1, 2), (1, 2, 3)):
        bn = model.data(model.index(r, bn_col), Qt.ItemDataRole.DisplayRole)
        assert "task-memory" in bn
        assert f"run-{n}" in bn


def test_toggling_include_changes_row_state_to_skip() -> None:
    df = make_df([_ok_row()])
    model = InventoryTableModel(df)
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == ""
    assert model.setData(model.index(0, inc_col), False) is True
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


# ---------------------------------------------------------------------------
# Project integration
# ---------------------------------------------------------------------------


def test_edit_appends_user_set_cell_event_to_project(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))

    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    model.setData(model.index(0, task_col), "motor")

    events = list(project.log)
    set_cell_events = [e for e in events if isinstance(e, UserSetCell)]
    assert len(set_cell_events) == 1
    ev = set_cell_events[0]
    assert ev.row_id == "1.2.3.4"
    assert ev.column == "task"
    assert ev.value == "motor"
    assert ev.previous == "rest"


def test_include_toggle_appends_user_toggle_include_event(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    df = make_df([_ok_row()])
    model = InventoryTableModel(df, project=project)

    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    model.setData(model.index(0, inc_col), False)

    toggles = [e for e in project.log if isinstance(e, UserToggleInclude)]
    assert len(toggles) == 1
    assert toggles[0].row_id == "1.2.3.4"
    assert toggles[0].include is False


def test_project_overlay_applied_on_load(tmp_path: Path) -> None:
    """Reopening a project should reapply previous cell + include edits."""
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))
    project.append(UserSetCell(row_id="1.2.3.4", column="task", value="motor", previous="rest"))
    project.append(UserToggleInclude(row_id="1.2.3.4", include=False))

    # Fresh model from the original DataFrame — overlay should apply.
    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    bn_col = next(i for i, c in enumerate(COLUMNS) if c.key == "basename")
    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    assert model.data(model.index(0, task_col), Qt.ItemDataRole.DisplayRole) == "motor"
    assert "task-motor" in model.data(model.index(0, bn_col), Qt.ItemDataRole.DisplayRole)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is False
    assert model.data(model.index(0, 0), ROW_STATE_ROLE) == "skip"


def test_overlay_with_no_matching_row_id_is_safe(tmp_path: Path) -> None:
    """A project that references rows not present in the TSV is ignored."""
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(UserSetCell(row_id="absent-row", column="task", value="x"))
    df = make_df([_func_row()])
    model = InventoryTableModel(df, project=project)  # must not raise


# ---------------------------------------------------------------------------
# Integration: model inside a real QTableView
# ---------------------------------------------------------------------------


def test_checkbox_delegate_toggle_on_click(qtbot) -> None:
    """A left-mouse click on the include column flips the cell."""
    from PyQt6.QtCore import QEvent, QPoint, Qt
    from PyQt6.QtGui import QMouseEvent

    df = make_df([_ok_row(include=1)])
    model = InventoryTableModel(df)
    view = QTableView()
    view.setModel(model)
    qtbot.addWidget(view)

    inc_col = next(i for i, c in enumerate(COLUMNS) if c.key == "include")
    view.setItemDelegateForColumn(inc_col, CheckboxDelegate(view))
    view.resize(800, 100)
    view.show()
    qtbot.waitExposed(view)

    cell_rect = view.visualRect(model.index(0, inc_col))
    center = cell_rect.center()
    # Simulate a click on the cell's viewport. ``mouseClick`` uses the
    # widget's coordinate space; ``visualRect`` is already in that space.
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is False

    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert model.data(model.index(0, inc_col), PAYLOAD_ROLE) is True


def test_model_renders_in_real_qtableview(qtbot) -> None:
    df = make_df([_ok_row(), _func_row(), _physio_row(), _err_row(), _skip_row()])
    model = InventoryTableModel(df)

    view = QTableView()
    view.setModel(model)
    qtbot.addWidget(view)

    # Wire delegates per spec — same as the eventual ConverterView will.
    for col, spec in enumerate(COLUMNS):
        if spec.role == "checkbox":
            view.setItemDelegateForColumn(col, CheckboxDelegate(view))
        elif spec.role == "status":
            view.setItemDelegateForColumn(col, StatusDelegate(view))
        else:
            view.setItemDelegateForColumn(col, CellTextDelegate(spec.role, view))

    view.resize(900, 200)
    view.show()
    qtbot.waitExposed(view)
    # If we got here, the model + delegates rendered with no Qt errors.
    assert view.model() is model
    assert view.model().rowCount() == 5


# ---------------------------------------------------------------------------
# Recording-metadata inheritance (global default <- per-row override)
# ---------------------------------------------------------------------------


def _eeg_inherit_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"BIDS_name": "sub-001", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
         "source_file": "a.edf", "proposed_basename": "sub-001_task-rest_eeg",
         "entities": json.dumps({"subject": "001", "task": "rest"}),
         "line_freq": "", "montage": "", "eeg_reference": "", "eeg_ground": "", "include": 1},
        {"BIDS_name": "sub-002", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
         "source_file": "b.edf", "proposed_basename": "sub-002_task-rest_eeg",
         "entities": json.dumps({"subject": "002", "task": "rest"}),
         "line_freq": "60", "montage": "", "eeg_reference": "", "eeg_ground": "", "include": 1},
    ])


def _col_index(key: str) -> int:
    return next(i for i, c in enumerate(COLUMNS) if c.key == key)


def _spec(line_freq):
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec
    return RecordingMetaSpec(defaults=AcquisitionSpec(power_line_freq=line_freq))


def test_inheritance_blank_cell_shows_global_default() -> None:
    from bidsmgr.gui.delegates import INHERITED_ROLE
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_spec(50.0))
    col = _col_index("line_freq")
    idx0 = m.index(0, col)
    assert m.data(idx0, Qt.ItemDataRole.DisplayRole) == "50"  # inherited from global
    assert m.data(idx0, INHERITED_ROLE) is True
    # Row 1 has an explicit override -> shown as-is, not inherited.
    idx1 = m.index(1, col)
    assert m.data(idx1, Qt.ItemDataRole.DisplayRole) == "60"
    assert m.data(idx1, INHERITED_ROLE) is False


def test_inheritance_set_equal_global_clears_override() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_spec(50.0))
    col = _col_index("line_freq")
    # Setting the global value clears the cell back to inherited.
    m.setData(m.index(0, col), "50")
    assert m.dataframe().at[0, "line_freq"] == ""
    assert m.is_inherited(0, "line_freq") is True
    # A differing value is stored as a real override.
    m.setData(m.index(0, col), "60")
    assert m.dataframe().at[0, "line_freq"] == "60"
    assert m.is_inherited(0, "line_freq") is False


def test_inheritance_global_change_reflows_inherited_rows() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_spec(50.0))
    assert m.effective_value(0, "line_freq") == "50"
    m.set_global_spec(_spec(60.0))
    assert m.effective_value(0, "line_freq") == "60"  # inherited row re-flowed
    # Inert without a spec.
    m2 = InventoryTableModel(_eeg_inherit_df())
    assert m2.effective_value(0, "line_freq") == ""


# ---------------------------------------------------------------------------
# Per-row acquisition overrides (scaffold overrides[row_id]; not TSV columns)
# ---------------------------------------------------------------------------


def _device_spec(**defaults):
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec
    return RecordingMetaSpec(defaults=AcquisitionSpec(**defaults))


def test_acq_override_inherits_dataset_default() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec(manufacturer="Brain Products", amplifier_model="actiCHamp"))
    # No per-row override yet -> effective is the dataset default, flagged inherited.
    assert m.acq_effective(0, "manufacturer") == "Brain Products"
    assert m.acq_override(0, "manufacturer") == ""
    assert m.acq_is_inherited(0, "manufacturer") is True
    assert m.acq_effective(0, "amplifier_model") == "actiCHamp"


def test_acq_override_set_stores_in_spec_and_emits(qtbot) -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    with qtbot.waitSignal(m.recordingSpecChanged, timeout=500):
        assert m.set_acq_override(0, "manufacturer", "BioSemi") is True
    assert m.acq_override(0, "manufacturer") == "BioSemi"
    assert m.acq_is_inherited(0, "manufacturer") is False
    # Stored under the row_id (the EEG source path), not a TSV column.
    assert m.global_spec().overrides["a.edf"].manufacturer == "BioSemi"
    assert "manufacturer" not in m.dataframe().columns


def test_acq_override_equal_default_clears_to_inherited() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    m.set_acq_override(0, "manufacturer", "BioSemi")
    # Re-set to the dataset default -> override cleared, row inherits again.
    assert m.set_acq_override(0, "manufacturer", "Brain Products") is True
    assert m.acq_override(0, "manufacturer") == ""
    assert m.acq_is_inherited(0, "manufacturer") is True
    assert "a.edf" not in m.global_spec().overrides  # empty override dropped


def test_acq_override_blank_clears_and_keeps_other_fields() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec())
    m.set_acq_override(0, "manufacturer", "BioSemi")
    m.set_acq_override(0, "amplifier_model", "actiCHamp")
    # Clearing one field keeps the override entry alive for the other.
    assert m.set_acq_override(0, "manufacturer", "") is True
    over = m.global_spec().overrides["a.edf"]
    assert over.manufacturer is None
    assert over.amplifier_model == "actiCHamp"


def test_acq_override_keyed_per_row() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    m.set_acq_override(0, "manufacturer", "BioSemi")
    # Row 1 (source b.edf) is untouched -> still inherits the default.
    assert m.acq_effective(1, "manufacturer") == "Brain Products"
    assert m.acq_override(1, "manufacturer") == ""


def test_acq_override_noop_returns_false() -> None:
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec(manufacturer="Brain Products"))
    # Setting the inherited value again is a no-op.
    assert m.set_acq_override(0, "manufacturer", "Brain Products") is False


def test_acq_override_meg_string_field() -> None:
    """MEG manual fields (dewar position) round-trip through the override path."""
    m = InventoryTableModel(_eeg_inherit_df())
    m.set_global_spec(_device_spec())
    m.set_acq_override(0, "dewar_position", "supine")
    assert m.global_spec().overrides["a.edf"].dewar_position == "supine"
    assert m.acq_effective(0, "dewar_position") == "supine"
    # Clear -> override dropped (back to unset).
    assert m.set_acq_override(0, "dewar_position", "") is True
    assert "a.edf" not in m.global_spec().overrides


def test_four_states_render_distinctly() -> None:
    """A metadata cell can be in four states and they must not look alike.

    An inherited value shows the value. VARIES asks for an answer this row has
    not given. An empty inherited default shows the em dash. A field the
    datatype does not have shows nothing at all and refuses edits. Before this,
    the middle two were both blank and the last one showed a value that was
    simply wrong for the row.
    """
    from bidsmgr.recording_meta import VARIES, RecordingMetaSpec

    spec = RecordingMetaSpec()
    spec.defaults.power_line_freq = 50      # the same for every recording
    spec.defaults.eeg_reference = VARIES    # differs per recording

    df = make_df([
        _ok_row(modality="eeg", modality_bids="eeg", proposed_datatype="eeg",
                bids_guess_datatype="eeg", bids_guess_suffix="eeg",
                line_freq="", eeg_reference="",
                proposed_basename="sub-001_task-rest_eeg"),
        _ok_row(line_freq="", eeg_reference=""),  # an anat/T1w row
    ])
    model = InventoryTableModel(df)
    model.set_global_spec(spec)

    def shown(row, key):
        col = next(i for i, c in enumerate(COLUMNS) if c.key == key)
        return model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)

    # EEG: an inherited value, and a field that wants a per-row answer.
    assert shown(0, "line_freq") == "50"
    assert shown(0, "eeg_reference") == "varies?"
    assert model.needs_per_row_answer(0, "eeg_reference")
    assert not model.is_inherited(0, "eeg_reference")

    # MRI: neither field applies, so both are blank and neither prompts.
    assert shown(1, "line_freq") == ""
    assert shown(1, "eeg_reference") == ""
    assert not model.needs_per_row_answer(1, "eeg_reference")


def test_answering_a_varies_field_clears_the_prompt() -> None:
    from bidsmgr.recording_meta import VARIES, RecordingMetaSpec

    spec = RecordingMetaSpec()
    spec.defaults.eeg_reference = VARIES
    df = make_df([_ok_row(
        modality="eeg", modality_bids="eeg", proposed_datatype="eeg",
        bids_guess_datatype="eeg", bids_guess_suffix="eeg", eeg_reference="",
        proposed_basename="sub-001_task-rest_eeg",
    )])
    model = InventoryTableModel(df)
    model.set_global_spec(spec)
    col = next(i for i, c in enumerate(COLUMNS) if c.key == "eeg_reference")

    assert model.needs_per_row_answer(0, "eeg_reference")
    assert model.setData(model.index(0, col), "Cz", Qt.ItemDataRole.EditRole)
    assert not model.needs_per_row_answer(0, "eeg_reference")
    assert model.data(model.index(0, col), Qt.ItemDataRole.DisplayRole) == "Cz"


def test_backend_column_names_the_tool_that_will_run() -> None:
    """REGRESSION: the column read "any row with a source file is mne-bids",
    which is not what the converter's registry does. An ECAT PET scan has a
    source file and is converted by EcatDirect through nibabel, so the table
    named the wrong tool for every ECAT row."""
    df = make_df([
        _ok_row(proposed_datatype="pet", bids_guess_datatype="pet",
                bids_guess_suffix="pet", format="ECAT", source_file="Hoffman.v"),
        _ok_row(proposed_datatype="pet", bids_guess_datatype="pet",
                bids_guess_suffix="pet", format="DICOM", source_file=""),
        _ok_row(proposed_datatype="eeg", bids_guess_datatype="eeg",
                bids_guess_suffix="eeg", format="EEGLAB", source_file="a.set"),
        _ok_row(format="DICOM", source_file=""),                    # anat
        _ok_row(bids_guess_suffix="physio", format="DICOM", source_file=""),
    ])
    model = InventoryTableModel(df)
    assert [model._backend(r) for r in range(5)] == [
        "ecat", "dcm2niix", "mne-bids", "dcm2niix", "bidsphysio",
    ]


def test_a_cell_says_which_layer_its_value_came_from() -> None:
    """"Inherited" alone does not tell a user enough to act on: the value may
    come from the dataset defaults, this modality's block, or a sequence
    template, and which one decides where to go and change it.

    This also covers a disconnect: a template value never used to reach the
    table at all, so you could set one, see nothing, and find it in the
    sidecars after converting."""
    from bidsmgr.recording_meta import VARIES, AcquisitionSpec, RecordingMetaSpec

    spec = RecordingMetaSpec()
    spec.defaults.montage = "standard_1020"
    spec.modality_defaults["eeg"] = AcquisitionSpec(power_line_freq=50)
    spec.sequence_templates = {"eeg/eeg": {"EEGReference": "Cz"}}
    spec.defaults.eeg_ground = VARIES

    df = make_df([_ok_row(
        modality="eeg", modality_bids="eeg", proposed_datatype="eeg",
        bids_guess_datatype="eeg", bids_guess_suffix="eeg", task="rest",
        line_freq="", eeg_reference="", montage="", eeg_ground="",
        proposed_basename="sub-001_task-rest_eeg",
    )])
    model = InventoryTableModel(df)
    model.set_global_spec(spec)

    def cell(key):
        col = next(i for i, c in enumerate(COLUMNS) if c.key == key)
        idx = model.index(0, col)
        return (
            model.data(idx, Qt.ItemDataRole.DisplayRole),
            model.data(idx, Qt.ItemDataRole.ToolTipRole) or "",
        )

    value, tip = cell("montage")
    assert value == "standard_1020" and "dataset defaults" in tip

    value, tip = cell("line_freq")
    assert value == "50" and "eeg defaults" in tip

    # The template reaches the table, and says so.
    value, tip = cell("eeg_reference")
    assert value == "Cz" and "eeg/eeg template" in tip

    # VARIES asks rather than reports.
    value, tip = cell("eeg_ground")
    assert value == "varies?" and "differs per recording" in tip


def test_the_properties_panel_resolves_per_modality() -> None:
    """A study running both instruments sees each one's own answer, and is told
    where it came from."""
    from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec

    spec = RecordingMetaSpec()
    spec.modality_defaults["eeg"] = AcquisitionSpec(manufacturer="Brain Products")
    spec.modality_defaults["meg"] = AcquisitionSpec(manufacturer="Elekta")
    spec.sequence_templates = {"eeg/eeg": {"ManufacturersModelName": "actiCHamp"}}

    df = make_df([
        _ok_row(proposed_datatype="eeg", bids_guess_datatype="eeg",
                bids_guess_suffix="eeg", source_file="a.set"),
        _ok_row(proposed_datatype="meg", bids_guess_datatype="meg",
                bids_guess_suffix="meg", source_file="a.fif"),
    ])
    model = InventoryTableModel(df)
    model.set_global_spec(spec)

    assert model.acq_effective(0, "manufacturer") == "Brain Products"
    assert model.acq_effective(1, "manufacturer") == "Elekta"
    assert "eeg defaults" in model.acq_inherited_from(0, "manufacturer")
    assert "meg defaults" in model.acq_inherited_from(1, "manufacturer")
    # A template value reaches the panel too.
    assert model.acq_effective(0, "amplifier_model") == "actiCHamp"
    assert "template" in model.acq_inherited_from(0, "amplifier_model")


# ---------------------------------------------------------------------------
# Colliding BIDS names
# ---------------------------------------------------------------------------


def _colliding_rows() -> pd.DataFrame:
    """Two EEG recordings that resolve to one name.

    The real shape of it: a workshop tree exports a rest and a video recording
    per subject, both named after the participant code, so the task label is
    identical and only the folder distinguishes them.
    """
    return pd.DataFrame([
        {
            "BIDS_name": "sub-001", "include": 1, "modality": "eeg",
            "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
            "proposed_basename": "sub-001_task-CLV002_eeg",
            "Proposed BIDS name": "sub-001_task-CLV002_eeg",
            "entities": json.dumps({"subject": "001", "task": "CLV002"}),
            "proposed_issues": "", "dataset": "study",
            "source_file": f"sub-001/{where}/CLV002.set",
            "bids_guess_skip": False, "bids_guess_confidence": "0.9",
            "task": "CLV002", "run": "", "session": "", "series_uid": "",
        }
        for where in ("rest", "video")
    ])


def test_a_colliding_row_reads_as_an_error(qtbot) -> None:
    """Which is what paints the basename red.

    ``CellTextDelegate`` with the ``basename`` role tints to the palette's
    error colour whenever the row state is ``err``.
    """
    m = InventoryTableModel(_colliding_rows())
    assert [m._row_states[i] for i in range(2)] == ["err", "err"]
    assert m.colliding_rows() == {0, 1}


def test_the_reason_is_on_the_row(qtbot) -> None:
    """Hovering any cell of the row says why."""
    m = InventoryTableModel(_colliding_rows())
    tip = m.data(m.index(0, 0), Qt.ItemDataRole.ToolTipRole)
    assert tip and "same BIDS name" in tip


def test_fixing_the_name_clears_the_red_on_BOTH_rows(qtbot) -> None:
    """The reported confusion: it stayed red after being fixed.

    The state used to come from a note stamped into the row at scan time, which
    nothing rewrote, and an edit refreshed only the row that was edited. So the
    row you corrected stayed red, and so did its partner, which was now fine.
    """
    m = InventoryTableModel(_colliding_rows())
    assert m.colliding_rows() == {0, 1}

    m.set_entity(1, "task", "video")

    assert m.colliding_rows() == set()
    assert m._row_states[0] != "err"
    assert m._row_states[1] != "err"
    assert m.data(m.index(0, 0), Qt.ItemDataRole.ToolTipRole) is None


def test_excluding_one_side_clears_it_too(qtbot) -> None:
    """An excluded row is never written, so it cannot overwrite anything."""
    m = InventoryTableModel(_colliding_rows())
    checkbox = next(
        i for i, spec in enumerate(m.COLUMNS) if spec.role == "checkbox"
    )
    m.setData(m.index(1, checkbox), False, Qt.ItemDataRole.EditRole)

    assert m.colliding_rows() == set()
    assert m._row_states[0] != "err"


def test_a_new_clash_appears_without_a_rescan(qtbot) -> None:
    """Derived live, so it works in both directions."""
    df = _colliding_rows()
    df.at[1, "proposed_basename"] = "sub-001_task-video_eeg"
    df.at[1, "entities"] = json.dumps({"subject": "001", "task": "video"})
    m = InventoryTableModel(df)
    assert m.colliding_rows() == set()

    m.set_entity(1, "task", "CLV002")

    assert m.colliding_rows() == {0, 1}
    assert m._row_states[0] == "err"


def test_unique_names_are_not_flagged(qtbot) -> None:
    df = _colliding_rows()
    # The entities are the source of truth; the model rebuilds the basename
    # from them, so changing the display cell alone would be undone.
    df.at[1, "entities"] = json.dumps({"subject": "001", "task": "video"})
    df.at[1, "proposed_basename"] = "sub-001_task-video_eeg"
    m = InventoryTableModel(df)
    assert m.colliding_rows() == set()
    assert m._row_states[0] != "err"
