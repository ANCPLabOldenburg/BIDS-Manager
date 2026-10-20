"""File-issues browser dialog for the Editor view.

Sister of :class:`bidsmgr.gui.issues_dialog.IssuesDialog` (which is
inventory-row-shaped, for the Converter). This one walks a
:class:`bidsmgr.editor.types.ValidationReport` and lists every file
whose severity matches the clicked toolbar chip — plus dataset-level
findings for the ``ok`` chip when there are none.

One painted list (:class:`~bidsmgr.gui.widgets.findings_list.FindingsList`):
the files, their findings one line each, the selected finding in full
underneath with its Fix button. Opening a file (double click, Return)
emits :pyattr:`file_selected` with the absolute path; the host panel wires
that to the BIDS tree's selection so the user lands on the file in
question and the three panes update in concert.

Theme handling: every widget uses the same QSS object names as the
Converter's issues dialog, so the global stylesheet handles dark↔light.
"""

from __future__ import annotations

from pathlib import Path

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

from ..editor.types import FileVerdict, Severity, ValidationReport
from .widgets import StatusBadge
from .widgets.findings_list import Finding, FindingsList, Group


_SEVERITY_LABEL: dict[str, str] = {
    "err":  "Errors",
    "warn": "Warnings",
    "ok":   "Files OK",
}


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
    # (file, field) when a Fix button inside the listing is pressed. The panel
    # decides where that lands, because it owns the panes; this dialog only
    # knows which file and which field the button belonged to.
    fix_requested = pyqtSignal(Path, str)

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
        self.resize(560, 620)
        self.setMinimumWidth(360)

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

        # One painted list, not a card and a dozen widgets per finding: the
        # warnings of 300 files took 59 s to open that way, the whole report
        # more than 45 minutes (widgets/findings_list).
        body = QWidget()
        body.setObjectName("issue-dialog-body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(12, 10, 12, 10)
        bl.setSpacing(6)
        self._list = FindingsList("Find a file, a rule, a field or a word")
        self._list.activated.connect(self._on_card_activated)
        self._list.fix_requested.connect(self._on_fix_requested)
        if not matched:
            empty = QLabel(
                f"No files with severity ‘{severity}’ in this report."
            )
            empty.setObjectName("pane-hint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            bl.addWidget(empty)
            self._list.setVisible(False)
        bl.addWidget(self._list, 1)
        self._list.set_groups(self._groups(matched, severity))
        outer.addWidget(body, 1)

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
            if severity == "ok":
                # "valid" chip: files with no findings at all.
                fsev = (
                    f.severity.value if isinstance(f.severity, Severity)
                    else str(f.severity)
                )
                if fsev == "ok":
                    out.append(f)
                continue
            # err / warn chips: every file that CONTAINS an issue of that
            # severity (an err+warn file appears under BOTH chips, showing the
            # matching subset). Matching by file-rollup hid an err file's
            # warnings from the warnings chip entirely. Skip mirrors.
            if any(
                (i.severity.value if isinstance(i.severity, Severity)
                 else str(i.severity)) == severity
                and not getattr(i, "mirrored", False)
                for i in f.issues
            ):
                out.append(f)
        # Sort by parent dir then name for predictable layout.
        out.sort(key=lambda f: (str(f.path.parent), f.path.name))
        return out

    def _groups(self, matched: list[FileVerdict], severity: str) -> list[Group]:
        """One group per file: the findings of the clicked chip's severity
        (an error file also carries warnings, which belong to the other chip),
        mirrors left out so a finding is not shown twice.

        One pass, the severity compared by identity and the path joined
        rather than resolved: on a 287,000-finding report this loop WAS the
        time the dialog took to open, once the widgets were gone."""
        try:
            wanted = Severity(severity)
        except ValueError:
            wanted = None
        root = self._bids_root
        groups = []
        for f in matched:
            findings = [
                Finding(severity, i.message, i.rule_id, i.field, i.fix_label, i.schema_rule)
                for i in f.issues
                if (i.severity is wanted or i.severity == severity) and not i.mirrored
            ]
            hint = f"{f.datatype}/{f.suffix}" if f.datatype and f.suffix else (
                f.datatype or f.suffix or "")
            key = f.path if f.path.is_absolute() else root / f.path
            groups.append(Group(f.path.as_posix(), key, hint, findings))
        return groups

    @staticmethod
    def _header_text(severity: str) -> str:
        if severity == "err":
            return (
                "Files that failed validation. Double-click a file to open "
                "it in the BIDS tree and start fixing."
            )
        if severity == "warn":
            return (
                "Files with warnings. They may still be usable; review "
                "each before sharing the dataset."
            )
        if severity == "ok":
            return (
                "Files that passed validation. Double-click any to open "
                "its full schema audit."
            )
        return ""

    def _absolute(self, rel_or_abs: Path) -> Path:
        """Promote a relative FileVerdict path to absolute under the root."""
        if rel_or_abs.is_absolute():
            return rel_or_abs
        return (self._bids_root / rel_or_abs).resolve()

    def _on_fix_requested(self, path: Path, field: str) -> None:
        """Take the user to where the finding is actually edited.

        Same destination the validation pane's Fix button reaches, because it
        is the same question: the panel owns the routing, and this dialog only
        has to say which file and which field.
        """
        self.fix_requested.emit(path, field)

    def _on_card_activated(self, path: Path) -> None:
        self.file_selected.emit(path)
        # Close the dialog so the user lands on the BIDS tree without
        # having to dismiss it manually. Matches the Converter's flow
        # where activating a row in IssuesDialog returns focus to the
        # inspection table.
        self.accept()


__all__ = ["EditorIssuesDialog"]
