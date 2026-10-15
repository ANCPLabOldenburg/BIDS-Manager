"""The QC report: which channels and which stretches to doubt.

A table of channels (the flagged ones first, with why) and a table of the
flagged segments (double-click one to go there), the summary and how to
read it, and the two things a reviewer does next: mark the suggested
channels bad, mark the flagged segments bad. Both are ordinary undoable
changes of the review, saved to the dataset only with "Save to dataset".
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ....viz.compute.meeg_qc import HELP

_VERDICT = {"noisy": "Noisy", "flat": "Flat", "uncorrelated": "Uncorrelated",
            "line noise": "Line noise"}


def _item(text: str, sort_value=None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if sort_value is not None:
        item.setData(Qt.ItemDataRole.UserRole, sort_value)
    return item


class QualityReport(QDialog):
    """The quality check of one recording, as tables and actions."""

    #: Go to a time (seconds of run time).
    go_to = pyqtSignal(float)
    #: Mark these channels bad.
    mark_channels = pyqtSignal(list)
    #: Mark the flagged segments bad.
    mark_segments = pyqtSignal()

    def __init__(self, result: dict, recording: str, parent=None) -> None:
        super().__init__(parent)
        self.result = result
        self.setWindowTitle(f"QC of {recording}")
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        summary = QLabel(result["summary"])
        summary.setObjectName("dialog-title")
        summary.setWordWrap(True)
        lay.addWidget(summary)
        notes = []
        if not result.get("line_freq"):
            notes.append("Line noise was not checked: the recording states no line "
                         "frequency (PowerLineFrequency).")
        if result.get("projected"):
            notes.append("The recording's SSP projectors were applied first.")
        how = QLabel(" ".join([HELP["channels"], HELP["reading"], *notes]))
        how.setObjectName("dialog-subtitle")
        how.setWordWrap(True)
        lay.addWidget(how)
        self.tabs = QTabWidget()
        self.channels = self._channel_table(result)
        self.segments = self._segment_table(result)
        self.tabs.addTab(self.channels, f"Channels ({len(result['suggested_bads'])} suggested)")
        flagged = int(sum(bool(v) for v in result["flagged"]))
        self.tabs.addTab(self.segments, f"Segments ({flagged} flagged)")
        lay.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.mark_selected = QPushButton("Mark selected channels bad")
        self.mark_selected.setObjectName("tb-btn")
        self.mark_selected.clicked.connect(self._mark_selected)
        row.addWidget(self.mark_selected)
        self.mark_suggested = QPushButton(
            f"Mark the {len(result['suggested_bads'])} suggested channels bad")
        self.mark_suggested.setObjectName("tb-btn")
        self.mark_suggested.setToolTip("The noisy, flat and uncorrelated ones; line noise "
                                       "alone is better filtered than dropped.")
        self.mark_suggested.setEnabled(bool(result["suggested_bads"]))
        self.mark_suggested.clicked.connect(
            lambda: self.mark_channels.emit(list(result["suggested_bads"])))
        row.addWidget(self.mark_suggested)
        self.mark_flagged = QPushButton(f"Mark the {flagged} flagged segments bad")
        self.mark_flagged.setObjectName("tb-btn")
        self.mark_flagged.setEnabled(flagged > 0)
        self.mark_flagged.clicked.connect(self.mark_segments.emit)
        row.addWidget(self.mark_flagged)
        row.addStretch(1)
        close = QPushButton("Close")
        close.setObjectName("tb-btn-primary")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def _table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers), self)
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _channel_table(self, result: dict) -> QTableWidget:
        table = self._table(["Channel", "Type", "STD vs its type (z)",
                             "Peak-to-peak vs its type (z)", "Follows its type",
                             "Line noise (dB over type)", "Segments off", "Verdict"])
        # The flagged first, then each type together: a channel is only ever
        # compared with its own type, so they read best side by side.
        rows = sorted(result["channels"],
                      key=lambda c: (not c["reasons"], c["type"], -max(c["std_z"], c["ptp_z"])))
        table.setRowCount(len(rows))
        for r, c in enumerate(rows):
            line = "" if c["line_db"] is None else f"{c['line_db']:+.1f}"
            follows = "" if c.get("follows") is None else f"{c['follows']:.2f}"
            table.setItem(r, 0, _item(c["name"]))
            table.setItem(r, 1, _item(c["label"]))
            table.setItem(r, 2, _item(f"{c['std_z']:+.1f}"))
            table.setItem(r, 3, _item(f"{c['ptp_z']:+.1f}"))
            table.setItem(r, 4, _item(follows))
            table.setItem(r, 5, _item(line))
            table.setItem(r, 6, _item(f"{100.0 * c['off_share']:.0f} %"))
            table.setItem(r, 7, _item(", ".join(_VERDICT.get(x, x) for x in c["reasons"])
                                      or "Within range"))
        table.resizeColumnsToContents()
        return table

    def _segment_table(self, result: dict) -> QTableWidget:
        table = self._table(["Start (s)", "End (s)", "Why"])
        step = float(result["segment_s"])
        rows = [(float(t), reasons) for t, bad, reasons in
                zip(result["times"], result["flagged"], result["segment_reasons"]) if bad]
        table.setRowCount(len(rows))
        for r, (t, reasons) in enumerate(rows):
            table.setItem(r, 0, _item(f"{t:.1f}", t))
            table.setItem(r, 1, _item(f"{t + step:.1f}"))
            table.setItem(r, 2, _item(", ".join(reasons)))
        table.setToolTip("Double-click a segment to go there.")
        table.cellDoubleClicked.connect(
            lambda row, _col: self.go_to.emit(float(table.item(row, 0).data(
                Qt.ItemDataRole.UserRole))))
        return table

    def _mark_selected(self) -> None:
        rows = sorted({i.row() for i in self.channels.selectedIndexes()})
        names = [self.channels.item(r, 0).text() for r in rows]
        if names:
            self.mark_channels.emit(names)


__all__ = ["QualityReport"]
