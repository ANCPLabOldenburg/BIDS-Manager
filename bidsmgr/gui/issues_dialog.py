"""Pop-up dialog that lists rows of a given severity with their issues.

Reached by clicking the warning / error / skipped chips in the
Converter toolbar. Reads :class:`InventoryTableModel` directly so it
always reflects the live state (post-edit, post-rebuild).

The layout is a vertical scroll of **row cards** — one card per
inventory row, with the row identifier as a clickable header and one
wrap-text :class:`ValMessage` per scanner-detected issue underneath.
Easier to read vertically when issue text is long; activating the
header (Enter or click) selects the matching row in the inspection
table so the user can jump straight to fixing it.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .models import COLUMNS, InventoryTableModel
from .delegates import ROW_STATE_ROLE
from .theme_manager import CUR
from .widgets import StatusBadge, ValMessage


# Human-readable titles per severity. Drives the dialog window title +
# the empty-state hint.
_SEVERITY_LABEL: dict[str, str] = {
    "err":  "Errors",
    "warn": "Warnings",
    "skip": "Skipped",
}


class _RowCard(QFrame):
    """One row's findings: identifier header + stacked ValMessages.

    Clicking the header (or hitting Enter on the focused header button)
    emits :pyattr:`activated` with the source DataFrame row index.
    """

    activated = pyqtSignal(int)

    def __init__(
        self,
        row: int,
        title: str,
        issues: list[str],
        severity: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("issue-card")
        self._row = row

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        # Header row: clickable "jump →" link styled as a flat button.
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        # Row identifier is a button so it gets focus + keyboard support
        # for free. Styled by QSS to look like a link rather than a
        # button (no border, accent color, hand cursor).
        self._title_btn = QPushButton(title)
        self._title_btn.setObjectName("issue-card-title")
        self._title_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title_btn.setFlat(True)
        self._title_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred,
        )
        self._title_btn.clicked.connect(lambda: self.activated.emit(self._row))
        head.addWidget(self._title_btn, 1)

        jump = QLabel("jump →")
        jump.setObjectName("issue-card-jump-hint")
        head.addWidget(jump)
        v.addLayout(head)

        # One ValMessage per issue. ``warn`` and ``skip`` use the amber
        # badge; ``err`` uses red. Issue text wraps to the card width
        # (ValMessage's body already sets ``setWordWrap(True)``).
        sev_for_badge = "err" if severity == "err" else "warn"
        if not issues:
            issues = [
                "(no issue text recorded)"
                if severity != "skip"
                else "Excluded from conversion."
            ]
        for text in issues:
            v.addWidget(ValMessage(sev_for_badge, "", text, None))


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
        self.resize(620, 640)
        self._severity = severity

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---------- header bar ----------
        header = QFrame()
        header.setObjectName("issue-dialog-header")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 14, 18, 14)
        h.setSpacing(10)

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

        # ---------- scrollable list of cards ----------
        scroll = QScrollArea()
        scroll.setObjectName("issue-dialog-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(10)
        self._cards_layout = bl

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self._populate(model, severity)

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

    def _populate(self, model: InventoryTableModel, severity: str) -> None:
        """Build one ``_RowCard`` per matching row."""
        df = model.dataframe()
        basename_col = next(
            (c for c in COLUMNS if c.key == "basename"), None,
        )
        id_col = next((c for c in COLUMNS if c.key == "id"), None)

        any_matches = False
        for row in range(model.rowCount()):
            state = model.data(model.index(row, 0), ROW_STATE_ROLE) or ""
            if state != severity:
                continue
            any_matches = True

            sub = ""
            bn = ""
            if id_col is not None:
                sub = model.data(
                    model.index(row, COLUMNS.index(id_col)),
                    Qt.ItemDataRole.DisplayRole,
                ) or ""
            if basename_col is not None:
                bn = model.data(
                    model.index(row, COLUMNS.index(basename_col)),
                    Qt.ItemDataRole.DisplayRole,
                ) or ""
            label = f"sub-{sub}" if sub else f"row {row + 1}"
            if bn and bn != "—":
                label = f"{label}  ·  {bn}"

            issues_text = ""
            if "proposed_issues" in df.columns:
                issues_text = str(df.at[row, "proposed_issues"] or "").strip()
            parts = [p.strip() for p in issues_text.split(" | ") if p.strip()]

            card = _RowCard(row, label, parts, severity)
            card.activated.connect(self.row_selected.emit)
            self._cards_layout.addWidget(card)

        if not any_matches:
            empty = QLabel("(no matching rows)")
            empty.setObjectName("pane-hint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._cards_layout.addWidget(empty)

        self._cards_layout.addStretch(1)


__all__ = ["IssuesDialog"]
