"""File-issues browser dialog for the Editor view.

Sister of :class:`bidsmgr.gui.issues_dialog.IssuesDialog` (which is
inventory-row-shaped, for the Converter). This one walks a
:class:`bidsmgr.editor.types.ValidationReport` and lists every file
whose severity matches the clicked toolbar chip — plus dataset-level
findings for the ``ok`` chip when there are none.

Each entry is a card: file path button (the "jump") + a
:class:`ValMessage` per finding. Activating the button emits
:pyattr:`file_selected` with the absolute path; the host panel wires
that to the BIDS tree's selection so the user lands on the file in
question and the three panes update in concert.

Theme handling: every widget uses the same QSS object names as the
Converter's issues dialog, so the global stylesheet handles dark↔light.
"""

from __future__ import annotations

from pathlib import Path
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

from ..editor.types import FileVerdict, Severity, ValidationReport
from .widgets import StatusBadge, ValMessage


_SEVERITY_LABEL: dict[str, str] = {
    "err":  "Errors",
    "warn": "Warnings",
    "ok":   "Files OK",
}


class _FileCard(QFrame):
    """One file's findings: path button header + stacked ValMessages."""

    activated = pyqtSignal(Path)

    def __init__(
        self,
        path: Path,
        issues: list,
        severity: str,
        datatype: Optional[str] = None,
        suffix: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("issue-card")
        self._path = path

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        title_text = str(path)
        self._title_btn = QPushButton(title_text)
        self._title_btn.setObjectName("issue-card-title")
        self._title_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title_btn.setFlat(True)
        self._title_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred,
        )
        self._title_btn.clicked.connect(
            lambda: self.activated.emit(self._path)
        )
        head.addWidget(self._title_btn, 1)

        if datatype or suffix:
            typed = "/".join(filter(None, [datatype, suffix]))
            typed_lbl = QLabel(typed)
            typed_lbl.setObjectName("issue-card-jump-hint")
            head.addWidget(typed_lbl)

        jump = QLabel("jump →")
        jump.setObjectName("issue-card-jump-hint")
        head.addWidget(jump)
        v.addLayout(head)

        if not issues:
            v.addWidget(ValMessage(
                "ok" if severity == "ok" else severity,
                "",
                "(no issue text — file passed validation)",
                None,
            ))
        else:
            for issue in issues:
                sev_str = (
                    issue.severity.value
                    if isinstance(issue.severity, Severity)
                    else str(issue.severity)
                )
                v.addWidget(ValMessage(
                    severity=sev_str,
                    rule=issue.rule_id,
                    body_html=issue.message,
                    fix_label=issue.fix_label,
                    field=issue.field,
                ))


class EditorIssuesDialog(QDialog):
    """Modeless file-issues browser for the Editor.

    Pass the live :class:`ValidationReport`; the dialog walks it once
    at construction. Re-open on every chip click so the listing stays
    fresh.

    Emits :pyattr:`file_selected` with the absolute file path when the
    user activates a card. The host panel maps that to a tree
    selection so all three panes (sidecar / validation / tree) update.
    """

    file_selected = pyqtSignal(Path)

    def __init__(
        self,
        report: ValidationReport,
        severity: str,
        bids_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._severity = severity
        self._bids_root = bids_root.resolve()
        title = _SEVERITY_LABEL.get(severity, severity.title())
        matched = self._matching_files(report, severity)
        count = len(matched)
        self.setWindowTitle(
            f"{title} · {count} file{'s' if count != 1 else ''}"
        )
        self.resize(700, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar.
        header = QFrame()
        header.setObjectName("issue-dialog-header")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 14, 18, 14)
        h.setSpacing(10)
        h.addWidget(
            StatusBadge(severity if severity in ("err", "warn") else "ok"),
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_lbl = QLabel(
            f"{title} · {count} file{'s' if count != 1 else ''}"
        )
        title_lbl.setObjectName("issue-dialog-title")
        sub_lbl = QLabel(self._header_text(severity))
        sub_lbl.setObjectName("issue-dialog-subtitle")
        sub_lbl.setWordWrap(True)
        title_block.addWidget(title_lbl)
        title_block.addWidget(sub_lbl)
        h.addLayout(title_block, 1)
        outer.addWidget(header)

        # Scrollable list of cards.
        scroll = QScrollArea()
        scroll.setObjectName("issue-dialog-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        body = QWidget()
        body.setObjectName("issue-dialog-body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(10)
        if not matched:
            empty = QLabel(
                f"No files with severity ‘{severity}’ in this report."
            )
            empty.setObjectName("pane-hint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            bl.addWidget(empty)
        else:
            for f in matched:
                abs_path = self._absolute(f.path)
                card = _FileCard(
                    path=abs_path,
                    issues=f.issues,
                    severity=severity,
                    datatype=f.datatype,
                    suffix=f.suffix,
                )
                card.activated.connect(self._on_card_activated)
                bl.addWidget(card)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer with a single Close button.
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        footer = QFrame()
        footer.setObjectName("issue-dialog-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(14, 10, 14, 10)
        fl.addStretch(1)
        fl.addWidget(buttons)
        outer.addWidget(footer)

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _matching_files(
        report: ValidationReport, severity: str,
    ) -> list[FileVerdict]:
        out: list[FileVerdict] = []
        for f in report.files:
            sev = (
                f.severity.value if isinstance(f.severity, Severity)
                else str(f.severity)
            )
            if sev == severity:
                out.append(f)
        # Sort by parent dir then name for predictable layout.
        out.sort(key=lambda f: (str(f.path.parent), f.path.name))
        return out

    @staticmethod
    def _header_text(severity: str) -> str:
        if severity == "err":
            return (
                "Files that failed validation. Click a row to jump to "
                "it in the BIDS tree and start fixing."
            )
        if severity == "warn":
            return (
                "Files with warnings. They may still be usable; review "
                "each before sharing the dataset."
            )
        if severity == "ok":
            return (
                "Files that passed validation. Click any to inspect "
                "its full schema audit."
            )
        return ""

    def _absolute(self, rel_or_abs: Path) -> Path:
        """Promote a relative FileVerdict path to absolute under the root."""
        if rel_or_abs.is_absolute():
            return rel_or_abs
        return (self._bids_root / rel_or_abs).resolve()

    def _on_card_activated(self, path: Path) -> None:
        self.file_selected.emit(path)
        # Close the dialog so the user lands on the BIDS tree without
        # having to dismiss it manually. Matches the Converter's flow
        # where activating a row in IssuesDialog returns focus to the
        # inspection table.
        self.accept()


__all__ = ["EditorIssuesDialog"]
