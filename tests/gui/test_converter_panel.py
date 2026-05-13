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
    # Scan button is enabled at start; Run-conversion is gated.
    assert panel._scan_btn.isEnabled()
    assert not panel._run_btn.isEnabled()


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
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "modality_bids": "func",
        "sequence": "bold_rest",
        "series_uid": "1.2.3.4",
        "proposed_datatype": "func",
        "proposed_basename": "sub-001_ses-pre_task-rest_bold",
        "Proposed BIDS name": "func/sub-001_ses-pre_task-rest_bold.nii.gz",
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "func",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
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
    err["proposed_basename"] = ""
    err["proposed_datatype"] = ""
    err["series_uid"] = "9.9.9"
    skip = _func_row()
    skip["include"] = 0
    skip["series_uid"] = "8.8.8"

    df = pd.DataFrame([valid, err, skip])
    panel.load_inventory(df, output_tsv=tmp_path / "inv.tsv")

    assert "1 valid" in panel._chip_valid.text()
    assert "1 error" in panel._chip_err.text()
    assert "1 skipped" in panel._chip_skip.text()


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


def test_bids_pathbar_change_button_is_enabled(qtbot) -> None:
    """The user must be able to pick the BIDS output before scanning."""
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert panel._bids_pathbar.change_button.isEnabled()


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


def test_scan_button_disabled_while_scan_runs(qtbot, tmp_path: Path) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)

    raw = tmp_path / "raw"
    raw.mkdir()

    worker = panel.start_scan(raw, tmp_path / "inv.tsv", n_jobs=1)
    # While the worker is alive the button is disabled.
    assert not panel._scan_btn.isEnabled()
    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        pass
    worker.wait()
    qtbot.wait(50)
    assert panel._scan_btn.isEnabled()


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
    """Bottom dock is now a horizontal splitter with two QTabWidgets:
    Log + Conflicts on the left, BIDS preview + Statistics on the right.
    """
    from PyQt6.QtWidgets import QSplitter, QTabWidget

    panel = ConverterPanel()
    qtbot.addWidget(panel)

    # The dock is now a QSplitter (not the original single QTabWidget).
    # Locate it via the _log_view + _bids_preview children whose
    # ancestors must include the dock.
    def _ancestors(w):
        cur = w
        while cur is not None:
            yield cur
            cur = cur.parent()

    log_ancestors = set(id(a) for a in _ancestors(panel._log_view))
    preview_ancestors = set(id(a) for a in _ancestors(panel._bids_preview))

    # Find the *common* QSplitter ancestor — that's the dock.
    common = log_ancestors & preview_ancestors
    splitters = [
        w for w in panel.findChildren(QSplitter)
        if id(w) in common
    ]
    # The vertical splitter (top half vs dock) is also a common ancestor;
    # the horizontal one is the dock itself. Pick the horizontal one.
    horizontal_splitters = [
        s for s in splitters if s.orientation().name == "Horizontal"
    ]
    assert horizontal_splitters, "expected at least one horizontal splitter"
    # The dock-level horizontal splitter holds exactly two QTabWidgets.
    dock_candidates = [
        s for s in horizontal_splitters
        if s.count() == 2 and all(
            isinstance(s.widget(i), QTabWidget) for i in range(2)
        )
    ]
    assert dock_candidates, "expected the dock to be a QSplitter with 2 QTabWidgets"

    dock = dock_candidates[0]
    left_tabs = dock.widget(0)
    right_tabs = dock.widget(1)
    left_titles = [left_tabs.tabText(i) for i in range(left_tabs.count())]
    right_titles = [right_tabs.tabText(i) for i in range(right_tabs.count())]
    assert any("Log" in t for t in left_titles)
    assert any("Conflicts" in t for t in left_titles)
    assert any("BIDS preview" in t for t in right_titles)
    assert any("Statistics" in t for t in right_titles)


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
    skipped["proposed_basename"] = "should-not-appear"
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
