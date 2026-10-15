"""The Converter's Log shows runs, and every run is kept as a file."""

from __future__ import annotations

import logging

import pytest

from bidsmgr.gui.log_history_dialog import LogHistoryDialog, describe
from bidsmgr.gui.theme_manager import PALETTES, ThemeManager
from bidsmgr.gui.widgets.run_log import RunLog, line_kind
from bidsmgr.util import oplog

pytestmark = pytest.mark.gui


def test_lines_are_sorted_by_what_they_say() -> None:
    assert line_kind("sub-001: conversion failed (dcm2niix exit 2)") == "error"
    assert line_kind("WARNING bidsmgr: no sidecar for run-2") == "warning"
    assert line_kind("Validation: 0 error(s), 0 warning(s)") == "info", "good news"
    assert line_kind("Converted 12 series") == "info"


def test_a_run_starts_with_a_heading_and_can_start_empty(qtbot) -> None:
    log = RunLog()
    qtbot.addWidget(log)
    log.begin_run("Scan", "OL_0003")
    log.append_lines(["Scanning 2,000 files", "warning: one unreadable file"])
    log.begin_run("Conversion", clear=True)
    assert [k for k, _ in log.lines()] == ["run"]
    assert "Conversion" in log.plain_text()
    log.append_lines(["sub-001 done\nsub-002 done"])
    assert len(log.lines()) == 3, "a multi-line message is one line each"


def test_colours_follow_a_theme_change(qtbot, qapp) -> None:
    log = RunLog()
    qtbot.addWidget(log)
    manager = ThemeManager(qapp)
    try:
        manager.apply("graphite")
        log.append_lines(["sub-002 failed"])
        manager.apply("porcelain")
        log.repaint_for_palette()
        html = log.document().toHtml()
        assert PALETTES["porcelain"]["error"].lower() in html.lower()
        assert PALETTES["graphite"]["error"].lower() not in html.lower()
    finally:
        manager.apply("dark")


def _write_log(root, verb: str, *, warn: bool = False) -> None:
    with oplog.operation_log(verb, root=root):
        if warn:
            logging.getLogger("bidsmgr.cli").warning("something odd")


def test_the_logs_window_lists_every_run(qtbot, tmp_path) -> None:
    root = tmp_path / "ds"
    (root / ".bidsmgr").mkdir(parents=True)
    _write_log(root, "scan")
    _write_log(root, "convert", warn=True)
    dlg = LogHistoryDialog(root)
    qtbot.addWidget(dlg)
    assert dlg.list.topLevelItemCount() == 2
    first = dlg.list.topLevelItem(0)
    assert first.text(1) == "Conversion" and first.text(2) == "Finished, 1 warning"
    assert "something odd" in dlg.view.plain_text()
    assert dlg.windowFlags() & dlg.windowFlags().WindowMaximizeButtonHint


def test_an_empty_history_says_so(qtbot, tmp_path) -> None:
    dlg = LogHistoryDialog(tmp_path / "nothing")
    qtbot.addWidget(dlg)
    assert dlg.list.topLevelItemCount() == 0 and not dlg.open_button.isEnabled()


def test_describe_reads_a_stopped_run(tmp_path) -> None:
    path = tmp_path / "20261012T101010000Z_convert.log"
    path.write_text("2026-10-12 12:10:10 INFO    bidsmgr.oplog: convert stopped by "
                    "OperationCancelled: x after 3.2 s: 0 warning(s), 2 error(s)\n",
                    encoding="utf-8")
    when, what, outcome = describe(path)
    assert what == "Conversion" and outcome == "Stopped, 2 errors" and "2026" in when


def test_the_converter_marks_runs_and_offers_the_logs(qtbot, isolated_settings,
                                                      tmp_path) -> None:
    from bidsmgr.gui.converter_panel import ConverterPanel

    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel._log_view.append_lines(["old stuff"])
    panel._begin_run("Conversion", "ds")
    assert panel._log_view.lines()[0][0] == "run", "emptied: the setting is on by default"
    panel._begin_run("Validation", first=False)
    assert [k for k, _ in panel._log_view.lines()] == ["run", "run"], "a chained step keeps it"
    assert set(panel._log_buttons) == {"log_clear", "log_file", "log_history"}
    # In the Log panel's own header, not on a tab bar of their own.
    assert all(b.window() is panel.window() and panel._log_frame._bar.isAncestorOf(b)
               for b in panel._log_buttons.values())
    panel._log_buttons["log_clear"].click()
    assert panel._log_view.lines() == []
