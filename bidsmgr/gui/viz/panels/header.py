"""The header inspector: an image's header beside its sidecar.

Rows come from :func:`bidsmgr.viz.inspect.inspect`, grouped (geometry, data,
time, diffusion), each disagreement coloured by how much it matters and
explained in its own column. Opened from the Tools menu, or by a validation
finding about the header, in which case the row that is the evidence for the
finding is selected.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout,
)

from ....viz.inspect import HeaderRow, worst
from ....viz.theme import parse_colour
from ..bridge import ThemeHub, connect_while_alive

_STATUS_TOKEN = {"error": "error", "warning": "warning", "info": "accent"}
_STATUS_WORD = {"error": "Disagrees", "warning": "Check", "info": "Note", "ok": ""}


def _qcolor(value: str) -> QColor:
    return QColor(*parse_colour(value))


class HeaderDialog(QDialog):
    """Non-modal: it stays open beside the image while you look."""

    COLUMNS = ("Field", "Header", "Sidecar", "")

    def __init__(self, title: str, rows: list[HeaderRow], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Header and sidecar: {title}")
        self.setObjectName("pane-dark")
        self.resize(860, 520)
        self.rows = list(rows)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        self.summary = QLabel(self._summary_text())
        self.summary.setObjectName("sidecar-footer-summary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMNS))
        self.tree.setHeaderLabels(list(self.COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setWordWrap(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.setUniformRowHeights(False)
        head = self.tree.header()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        head.setStretchLastSection(True)
        v.addWidget(self.tree, 1)
        # The selected row's note in full: the explanation is the point, and a
        # column cannot be both narrow enough to sit beside the values and
        # wide enough to hold it.
        self.detail = QLabel("")
        self.detail.setObjectName("sidecar-footer-summary")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.detail)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._show_detail(cur))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        v.addWidget(buttons)
        self._items: list[tuple[QTreeWidgetItem, HeaderRow]] = []
        self._fill()
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w._paint())

    def _summary_text(self) -> str:
        bad = [r for r in self.rows if r.status in ("error", "warning")]
        if not bad:
            return "The header and the sidecar agree."
        n_err = sum(r.status == "error" for r in bad)
        n_warn = len(bad) - n_err
        parts = []
        if n_err:
            parts.append(f"{n_err} disagreement{'s' if n_err != 1 else ''}")
        if n_warn:
            parts.append(f"{n_warn} thing{'s' if n_warn != 1 else ''} to check")
        return " and ".join(parts) + ". Select a row to read why it matters."

    def _fill(self) -> None:
        self.tree.clear()
        self._items = []
        group_item: Optional[QTreeWidgetItem] = None
        group = None
        for row in self.rows:
            if row.group != group:
                group = row.group
                group_item = QTreeWidgetItem([group, "", "", ""])
                font = group_item.font(0)
                font.setBold(True)
                group_item.setFont(0, font)
                group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tree.addTopLevelItem(group_item)
            # The column says how much it matters; the line under the table
            # says why, for the selected row (and so does the tooltip).
            word = _STATUS_WORD.get(row.status, "") if row.note else ""
            item = QTreeWidgetItem(["    " + row.field, row.header, row.sidecar, word])
            for col in range(len(self.COLUMNS)):
                item.setToolTip(col, row.note)
            self.tree.addTopLevelItem(item)
            self._items.append((item, row))
        self._paint()
        self.tree.resizeColumnToContents(1)
        self.tree.resizeColumnToContents(2)

    def _paint(self) -> None:
        theme = ThemeHub.instance().theme
        for item, row in self._items:
            token = _STATUS_TOKEN.get(row.status)
            brush = QBrush(_qcolor(theme.token(token, theme.text))) if token else QBrush()
            item.setForeground(3, brush)
            if row.status in ("error", "warning"):
                for col in (1, 2):
                    item.setForeground(col, brush)

    def _show_detail(self, item: Optional[QTreeWidgetItem]) -> None:
        row = next((r for i, r in self._items if i is item), None)
        if row is None or not row.note:
            self.detail.setText("")
            return
        word = _STATUS_WORD.get(row.status, "")
        self.detail.setText(f"{row.field}: {row.note}" if not word
                            else f"{word}, {row.field}: {row.note}")

    # ------------------------------------------------------------------
    def select_rule(self, rule: str) -> Optional[HeaderRow]:
        """Select (and scroll to) the row that is the evidence for ``rule``."""
        for item, row in self._items:
            if rule in row.rules:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return row
        return None

    def status(self) -> Optional[str]:
        return worst(self.rows)

    def row_texts(self) -> list[tuple[str, str, str, str]]:
        """Every row as shown (tests)."""
        return [(row.field, row.header, row.sidecar, row.status) for _i, row in self._items]


__all__ = ["HeaderDialog"]
