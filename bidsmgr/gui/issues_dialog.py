"""Pop-up dialog that lists rows of a given severity with their issues.

Reached by clicking the warning / error / skipped chips in the
Converter toolbar. Reads :class:`InventoryTableModel` directly so it
always reflects the live state (post-edit, post-rebuild).

One painted list (:class:`~bidsmgr.gui.widgets.findings_list.FindingsList`):
the rows, their issues one line each, the selected one in full underneath.
Opening a row (double click, Return) selects it in the inspection table so
the user can jump straight to fixing it.
"""

from __future__ import annotations


from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .models import COLUMNS, InventoryTableModel
from .delegates import ROW_STATE_ROLE
from .widgets import StatusBadge
from .widgets.findings_list import Finding, FindingsList, Group


# Human-readable titles per severity. Drives the dialog window title +
# the empty-state hint.
_SEVERITY_LABEL: dict[str, str] = {
    "err":  "Errors",
    "warn": "Warnings",
    "skip": "Skipped",
}


class IssuesDialog(QDialog):
    """Modeless inventory-issues browser.

    ``severity`` is one of ``"err"``, ``"warn"``, ``"skip"``. Pass the
    live :class:`InventoryTableModel`; the dialog walks it once at
    construction. Re-open on every chip-click so the listing stays
    fresh.

    Emits :pyattr:`row_selected` with the source-DataFrame row index
    when the user activates a row card; the panel hosting the dialog
    wires that to ``QTableView.selectRow(...)``.
    """

    row_selected = pyqtSignal(int)

    def __init__(
        self,
        model: InventoryTableModel,
        severity: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        title = _SEVERITY_LABEL.get(severity, severity.title())
        count = self._count(model, severity)
        self.setWindowTitle(f"{title} · {count} row{'s' if count != 1 else ''}")
        self.resize(520, 560)
        self.setMinimumWidth(340)
        self._severity = severity

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---------- header bar ----------
        header = QFrame()
        header.setObjectName("issue-dialog-header")
        h = QHBoxLayout(header)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(8)

        badge = StatusBadge(severity if severity in ("err", "warn") else "skip")
        h.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_lbl = QLabel(f"{title} · {count} row{'s' if count != 1 else ''}")
        title_lbl.setObjectName("issue-dialog-title")
        sub_lbl = QLabel(self._header_text(severity))
        sub_lbl.setObjectName("issue-dialog-subtitle")
        sub_lbl.setWordWrap(True)
        title_block.addWidget(title_lbl)
        title_block.addWidget(sub_lbl)
        h.addLayout(title_block, 1)

        outer.addWidget(header)

        # ---------- one painted list ----------
        # A card and a message widget per issue took 9 s for 3,000 rows;
        # this is one view over one model (widgets/findings_list).
        body = QWidget()
        body.setObjectName("issue-dialog-body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(12, 10, 12, 10)
        bl.setSpacing(6)
        self._list = FindingsList("Find a subject, a name or a word")
        self._list.activated.connect(self._on_activated)
        groups = self._groups(model, severity)
        if not groups:
            empty = QLabel("(no matching rows)")
            empty.setObjectName("pane-hint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bl.addWidget(empty)
            self._list.setVisible(False)
        bl.addWidget(self._list, 1)
        self._list.set_groups(groups)
        outer.addWidget(body, 1)

        # ---------- footer ----------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        footer = QFrame()
        footer.setObjectName("issue-dialog-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(14, 10, 14, 10)
        fl.addStretch(1)
        fl.addWidget(buttons)
        outer.addWidget(footer)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _header_text(severity: str) -> str:
        if severity == "err":
            return (
                "Rows that will fail or are missing required information. "
                "Click a row to jump to it in the inspector."
            )
        if severity == "warn":
            return (
                "Rows with scanner-detected warnings. They may still "
                "convert; review before committing."
            )
        if severity == "skip":
            return (
                "Rows excluded from conversion (include=0 or scanner-marked "
                "as skip)."
            )
        return ""

    @staticmethod
    def _count(model: InventoryTableModel, severity: str) -> int:
        n = 0
        for row in range(model.rowCount()):
            state = model.data(model.index(row, 0), ROW_STATE_ROLE) or ""
            if state == severity:
                n += 1
        return n

    def _groups(self, model: InventoryTableModel, severity: str) -> list[Group]:
        """One group per matching row: its identifier and name, and one
        finding per issue the scan recorded."""
        df = model.dataframe()
        basename_col = next((c for c in COLUMNS if c.key == "basename"), None)
        id_col = next((c for c in COLUMNS if c.key == "id"), None)
        badge = "err" if severity == "err" else ("skip" if severity == "skip" else "warn")
        groups = []
        for row in range(model.rowCount()):
            state = model.data(model.index(row, 0), ROW_STATE_ROLE) or ""
            if state != severity:
                continue
            sub = bn = ""
            if id_col is not None:
                sub = model.data(model.index(row, COLUMNS.index(id_col)),
                                 Qt.ItemDataRole.DisplayRole) or ""
            if basename_col is not None:
                bn = model.data(model.index(row, COLUMNS.index(basename_col)),
                                Qt.ItemDataRole.DisplayRole) or ""
            label = f"sub-{sub}" if sub else f"row {row + 1}"
            if bn and bn != "—":
                label = f"{label}  ·  {bn}"
            text = str(df.at[row, "issues"] or "").strip() if "issues" in df.columns else ""
            parts = [p.strip() for p in text.split(" | ") if p.strip()]
            if not parts:
                parts = ["Excluded from conversion." if severity == "skip"
                         else "(no issue text recorded)"]
            groups.append(Group(label, row, "", [Finding(badge, p) for p in parts]))
        return groups

    def _on_activated(self, row: int) -> None:
        self.row_selected.emit(int(row))


__all__ = ["IssuesDialog"]
