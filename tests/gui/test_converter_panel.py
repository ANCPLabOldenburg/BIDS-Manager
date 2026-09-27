"""Tests for ``bidsmgr.gui.converter_panel.ConverterPanel``.

Exercises the M3 end-to-end path:

1. Construct the panel.
2. Call ``start_scan(tmp_dir, tmp_tsv)`` — the worker runs in a
   background thread.
3. Wait for ``scan_finished``.
4. Assert the panel's model has the expected row count and the status
   chips updated.

Plus targeted checks on the smaller pieces (placeholder swap, button
gating during a scan, log forwarding).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.models import COLUMNS
from bidsmgr.project import Project, ScanImported, StageCompleted, UserSetCell


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_panel_constructs_without_a_project(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # No model until the first scan.
    assert panel.model() is None
    # Project-first gate: with no active project, Scan / Run / the raw + BIDS
    # change buttons are all disabled until a dataset is opened on Home.
    assert not panel._scan_btn.isEnabled()
    assert not panel._run_btn.isEnabled()
    assert not panel._raw_pathbar.change_button.isEnabled()
    assert not panel._bids_pathbar.change_button.isEnabled()


def test_panel_renders_under_offscreen_qpa(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.resize(1200, 720)
    panel.show()
    qtbot.waitExposed(panel)


# ---------------------------------------------------------------------------
# load_inventory: explicit DataFrame injection (no worker)
# ---------------------------------------------------------------------------


def _func_row() -> dict:
    return {
        "participant_id": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "sequence_kind": "func",
        "sequence": "bold_rest",
        "series_uid": "1.2.3.4",
        "datatype": "func",
        "bids_name": "sub-001_ses-pre_task-rest_bold",
        "bids_path": "func/sub-001_ses-pre_task-rest_bold.nii.gz",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "func",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre", "task": "rest"},
            sort_keys=True,
        ),
        "task": "rest",
        "run": "",
        "source_file": "",
    }


def test_load_inventory_attaches_model_and_swaps_to_table(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    df = pd.DataFrame([_func_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    assert panel.model() is not None
    assert panel.model().rowCount() == 1
    # After a successful inventory load the inspection pane shows the table.
    assert panel._inspection_stack.currentWidget() is panel._table


def test_load_inventory_updates_status_chips(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    valid = _func_row()
    err = _func_row()
    err["bids_name"] = ""
    err["datatype"] = ""
    err["series_uid"] = "9.9.9"
    skip = _func_row()
    skip["include"] = 0
    skip["series_uid"] = "8.8.8"

    df = pd.DataFrame([valid, err, skip])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    assert "1 valid" in panel._chip_valid.text()
    assert "1 error" in panel._chip_err.text()
    assert "1 skipped" in panel._chip_skip.text()


def test_revalidate_button_enables_after_scan_and_refreshes_chips(
    qtbot, tmp_path: Path,
) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Disabled before any scan.
    assert not panel._revalidate_btn.isEnabled()

    # A func/bold row missing its required task -> err.
    row = _func_row()
    row["entities"] = json.dumps({"subject": "001", "session": "pre"}, sort_keys=True)
    row["task"] = ""
    panel.load_inventory(pd.DataFrame([row]), output_tsv=tmp_path / "inv.tsv")
    assert panel._revalidate_btn.isEnabled()  # available after scan
    assert "1 error" in panel._chip_err.text()

    # Simulate a fix that bypassed setData (edit elsewhere), then re-validate:
    df = panel.model().dataframe()
    df.at[0, "task"] = "rest"
    df.at[0, "entities"] = json.dumps(
        {"subject": "001", "session": "pre", "task": "rest"}, sort_keys=True,
    )
    panel._on_revalidate_clicked()
    assert "0 error" in panel._chip_err.text()
    assert "1 valid" in panel._chip_valid.text()


def test_load_inventory_does_not_overwrite_user_set_bids_output(qtbot, tmp_path: Path) -> None:
    """BIDS output stays whatever the user set (or '(not set)' if unset).

    Loading an inventory must NOT silently set the BIDS output to the
    scan-TSV's parent — those are different things.
    """
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # User picks a BIDS output up front (simulated, no dialog).
    chosen = tmp_path / "bids_out"
    chosen.mkdir()
    panel._bids_parent = chosen
    panel._bids_pathbar.set_value(str(chosen), ok=True)

    df = pd.DataFrame([_func_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "elsewhere" / "inv.tsv")

    assert str(chosen) in panel._bids_pathbar.value()


def _eeg_row() -> dict:
    return {
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "eeg",
        "datatype": "eeg", "bids_guess_suffix": "eeg",
        "bids_name": "sub-001_task-rest_eeg",
        "entities": json.dumps({"subject": "001", "task": "rest"}, sort_keys=True),
        "task": "rest", "run": "", "series_uid": "",
        "source_file": "sub-001/rec.edf",
        "montage": "", "line_freq": "", "eeg_reference": "", "eeg_ground": "",
        "issues": "", "bids_guess_skip": False,
    }


def test_per_row_acq_override_persists_scaffold(qtbot, tmp_path: Path) -> None:
    """A per-row device override is written to the scaffold beside the TSV so
    the convert verb (which reloads it) sees the edit."""
    import json as _json
    from bidsmgr.recording_meta import (
        AcquisitionSpec, RecordingMetaSpec, scaffold_sidecar_path,
    )

    panel = ConverterPanel()
    qtbot.addWidget(panel)
    out_tsv = tmp_path / "inv.tsv"
    panel.load_inventory(pd.DataFrame([_eeg_row()]), output_tsv=out_tsv)

    m = panel.model()
    m.set_global_spec(RecordingMetaSpec(defaults=AcquisitionSpec(manufacturer="Brain Products")))
    # Override the manufacturer for row 0 -> triggers recordingSpecChanged.
    m.set_acq_override(0, "manufacturer", "BioSemi")

    scaffold = scaffold_sidecar_path(out_tsv)
    assert scaffold.exists()
    data = _json.loads(scaffold.read_text())
    assert data["overrides"]["sub-001/rec.edf"]["manufacturer"] == "BioSemi"


def test_no_project_gate_blocks_then_set_project_ungates(qtbot, tmp_path) -> None:
    """With no project the raw/BIDS/Scan controls are blocked; opening a project
    un-gates raw input + Scan and locks the BIDS output to the project root."""
    from bidsmgr.cli.create import open_or_create_workspace
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Gated before any project.
    assert not panel._raw_pathbar.change_button.isEnabled()
    assert not panel._bids_pathbar.change_button.isEnabled()
    assert not panel._scan_btn.isEnabled()

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    panel.set_project(proj, root)
    # Raw input + Scan enabled; BIDS output locked to the project (disabled).
    assert panel._raw_pathbar.change_button.isEnabled()
    assert panel._scan_btn.isEnabled()
    assert not panel._bids_pathbar.change_button.isEnabled()
    assert str(root) in panel._bids_pathbar.value()


# ---------------------------------------------------------------------------
# Worker-driven scan (the end-of-M3 scenario)
# ---------------------------------------------------------------------------


def test_start_scan_on_empty_dir_finishes_and_loads_empty_model(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"

    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        worker = panel.start_scan(raw, out_tsv, n_jobs=1)
    # Wait for the QThread itself to finish before asserting button state —
    # ``finished`` (QThread native signal) fires AFTER ``scan_finished``.
    worker.wait()
    # Process the finished slot.
    qtbot.wait(50)

    assert panel.model() is not None
    assert panel.model().rowCount() == 0
    assert panel._scan_btn.isEnabled(), "Scan button should re-enable on completion"
    assert out_tsv.exists()


def test_fresh_scan_resets_stale_recording_meta_scaffold(qtbot, tmp_path: Path) -> None:
    """A new scan must not inherit the previous dataset's metadata: the stale
    scaffold at the output path is removed so it re-seeds from the new data."""
    from bidsmgr.recording_meta import scaffold_sidecar_path
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"
    # Simulate a leftover scaffold from a PREVIOUS (different) dataset.
    stale = scaffold_sidecar_path(out_tsv)
    stale.write_text('{"schema_version": 1, "defaults": {"manufacturer": "OldVendor"}}',
                     encoding="utf-8")
    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        worker = panel.start_scan(raw, out_tsv, n_jobs=1)
    worker.wait()
    qtbot.wait(50)
    # The empty scan detects nothing to seed, so the stale scaffold is simply
    # gone (not carried over).
    assert not stale.exists()


def test_scan_button_becomes_stop_and_locks_run_while_scan_runs(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()

    worker = panel.start_scan(raw, tmp_path / "inv.tsv", n_jobs=1)
    # While the worker is alive the Scan button stays enabled but reads
    # "Stop scanning"; the Run button is locked out (mutual exclusion).
    assert panel._scan_btn.isEnabled()
    assert "Stop" in panel._scan_btn.text()
    assert not panel._run_btn.isEnabled()
    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        pass
    worker.wait()
    qtbot.wait(50)
    # On completion it reverts to "Scan…".
    assert panel._scan_btn.isEnabled()
    assert "Scan" in panel._scan_btn.text()
    assert "Stop" not in panel._scan_btn.text()


def test_log_messages_forwarded_to_panel_signal(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()
    received: list[str] = []
    panel.log_message.connect(received.append)

    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        worker = panel.start_scan(raw, tmp_path / "inv.tsv", n_jobs=1)
    worker.wait()
    qtbot.wait(50)

    assert any("Scanning" in m for m in received)
    assert any("Scan complete" in m for m in received)


# ---------------------------------------------------------------------------
# Project integration
# ---------------------------------------------------------------------------


def test_panel_uses_attached_project_for_edits(qtbot, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    project.append(ScanImported(inventory_tsv="/x", row_ids=("1.2.3.4",)))

    panel = ConverterPanel(project=project)
    qtbot.addWidget(panel)
    df = pd.DataFrame([_func_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    task_col = next(i for i, c in enumerate(COLUMNS) if c.key == "task")
    panel.model().setData(panel.model().index(0, task_col), "motor")

    events = [e for e in project.log if isinstance(e, UserSetCell)]
    assert len(events) == 1
    assert events[0].value == "motor"


# ---------------------------------------------------------------------------
# Bottom dock (M5.5)
# ---------------------------------------------------------------------------


def test_bottom_dock_split_layout(qtbot) -> None:
    """Bottom dock is a horizontal splitter of two detach-only PanelFrames,
    each wrapping a QTabWidget: Log on the left, BIDS preview + Statistics
    on the right. The whole dock is wrapped in a collapse-to-bottom frame.
    """
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    # The dock's two tab groups are wrapped in detach PanelFrames; read the
    # tab titles directly off the (kept) QTabWidget references.
    left_tabs = panel._left_tabs
    right_tabs = panel._right_tabs
    left_titles = [left_tabs.tabText(i) for i in range(left_tabs.count())]
    right_titles = [right_tabs.tabText(i) for i in range(right_tabs.count())]
    assert any("Log" in t for t in left_titles)
    # The Conflicts tab was removed; Log is the only left-hand tab now.
    assert not any("Conflicts" in t for t in left_titles)
    assert any("BIDS preview" in t for t in right_titles)
    assert any("Statistics" in t for t in right_titles)

    # The Log + preview groups are each independently detachable, and the
    # whole dock collapses toward the bottom.
    assert panel._log_frame.is_detached() is False
    assert panel._preview_frame.is_detached() is False
    panel._log_frame.detach()
    assert panel._log_frame.is_detached() is True
    panel._log_frame.reattach()
    assert panel._dock_frame.is_collapsed() is False
    panel._dock_frame.set_collapsed(True)
    assert panel._dock_frame.is_collapsed() is True


def test_log_tab_appends_progress_messages(qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.log_message.emit("first line")
    panel.log_message.emit("second line")
    # Log appends are batched and flushed every 100ms to keep the GUI
    # responsive under firehose logging; force a flush for the test.
    panel._flush_log_buffer()
    text = panel._log_view.toPlainText()
    assert "first line" in text
    assert "second line" in text


def test_bids_preview_rebuilds_from_model(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = pd.DataFrame([_func_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    # The preview tree should have at least one top-level node and the
    # basename should appear somewhere in it.
    tree = panel._bids_preview
    assert tree.topLevelItemCount() >= 1

    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    text = "\n".join(
        line for i in range(tree.topLevelItemCount()) for line in walk(tree.topLevelItem(i))
    )
    assert "sub-001_ses-pre_task-rest_bold" in text


def test_bids_preview_omits_skipped_rows(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    skipped = _func_row()
    skipped["include"] = 0
    skipped["series_uid"] = "9.9.9"
    skipped["bids_name"] = "should-not-appear"
    df = pd.DataFrame([_func_row(), skipped])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    def walk(item):
        yield item.text(0)
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    tree = panel._bids_preview
    text = "\n".join(
        line for i in range(tree.topLevelItemCount()) for line in walk(tree.topLevelItem(i))
    )
    assert "should-not-appear" not in text


def test_stats_label_updates_on_load(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    df = pd.DataFrame([_func_row()])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")
    text = panel._stats_label.text()
    assert "1 rows" in text
    assert "1 subjects" in text


# ---------------------------------------------------------------------------
# Scan project events
# ---------------------------------------------------------------------------


def test_busy_spinner_active_during_scan(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()

    worker = panel.start_scan(raw, tmp_path / "inv.tsv", n_jobs=1)
    assert panel._spinner.is_busy(), "spinner should be visible while a worker runs"
    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        pass
    worker.wait()
    qtbot.wait(50)
    assert not panel._spinner.is_busy(), "spinner should hide when the worker finishes"


def test_log_throttling_flushes_on_timer(qtbot) -> None:
    """The throttling timer ticks at 10 Hz, flushing whatever is buffered."""
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.log_message.emit("line A")
    # Don't manually flush; wait for the timer to do it.
    qtbot.waitUntil(lambda: "line A" in panel._log_view.toPlainText(), timeout=1000)


def test_scan_emits_scan_imported_and_stage_completed(qtbot, tmp_path: Path) -> None:
    project = Project.create(tmp_path / "demo.bidsmgr", name="demo")
    panel = ConverterPanel(project=project)
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"
    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        worker = panel.start_scan(raw, out_tsv, n_jobs=1)
    worker.wait()
    qtbot.wait(50)

    scans = [e for e in project.log if isinstance(e, ScanImported)]
    stages = [e for e in project.log if isinstance(e, StageCompleted) and e.stage == "scan"]
    assert len(scans) == 1
    assert scans[0].inventory_tsv == str(out_tsv)
    assert len(stages) == 1
    assert stages[0].success is True
    assert stages[0].summary["rows"] == 0
    assert stages[0].summary["raw_root"] == str(raw)
