"""Check whether the dataset's files still agree with each other.

One dialog over :mod:`bidsmgr.editor.coherence`. The first version listed
findings and a one-line repair, which told the user something was wrong and
left them to take on trust what pressing Repair would do. This one answers
the three questions a person actually has, in the order they have them:

1. **What is wrong?** The finding, grouped by kind, in a sentence.
2. **Which files?** Every file the finding is about, not just a count.
3. **What exactly changes if I fix it?** The before and after, per thing,
   including how many files each one touches.

Selecting a finding fills the panel underneath; nothing is written until
Repair is pressed, and the whole batch is one entry in the history.

Deliberately separate from the Validation pane. That answers "is this legal
BIDS", which is a different question: a dataset passes validation with a
``*_scans.tsv`` full of rows naming files somebody deleted last week.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import coherence
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.spinner import BusySpinner

log = logging.getLogger(__name__)


class CoherenceDialog(QDialog):
    """What disagrees, which files, and exactly what a repair would do."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._findings: list[coherence.Finding] = []
        self._applied = 0

        self.setWindowTitle("Check coherence")
        self.setModal(True)
        self.resize(1040, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Do these files still agree with each other?",
            "The validator answers whether the dataset is legal BIDS. This "
            "answers whether its files still describe the same dataset, "
            "which is what breaks when one is renamed or deleted outside "
            "the application, and what no validator will tell you. "
            "<b>Nothing is changed until you press Repair.</b>",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        split = QSplitter(Qt.Orientation.Vertical)

        found, fl = card("What was found")
        fl.addWidget(hint(
            "Grouped by kind. Tick what you want settled; a finding with no "
            "repair is shown so you know about it and cannot be ticked, "
            "because the decision is not the tool's to make."
        ))
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["What disagrees", "Repair"])
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        self._tree.currentItemChanged.connect(self._on_selected)
        fl.addWidget(self._tree, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        self._spinner = BusySpinner()
        tools.addWidget(self._spinner)
        rescan = QPushButton("Check again")
        rescan.setObjectName("tb-btn")
        rescan.clicked.connect(self._rescan)
        tools.addWidget(rescan)
        for text, state in (
            ("Select all", Qt.CheckState.Checked),
            ("Select none", Qt.CheckState.Unchecked),
        ):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, st=state: self._set_all(st))
            tools.addWidget(btn)
        tools.addStretch(1)
        fl.addLayout(tools)
        split.addWidget(found)

        detail, dl = card("What this is, and what the repair would do")
        self._what = QLabel("Select a finding above.")
        self._what.setObjectName("dlg-hint")
        self._what.setWordWrap(True)
        dl.addWidget(self._what)

        self._detail = QTreeWidget()
        self._detail.setObjectName("check-tree")
        self._detail.setColumnCount(3)
        self._detail.setHeaderLabels(["Now", "Becomes", "Files"])
        self._detail.setRootIsDecorated(False)
        self._detail.setUniformRowHeights(True)
        dl.addWidget(self._detail, 1)
        split.addWidget(detail)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        bl.addWidget(split, 1)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close
            | QDialogButtonBox.StandardButton.Ok
        )
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Repair")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._rescan()

    # -- filling ----------------------------------------------------------

    def _rescan(self) -> None:
        self._spinner.set_busy(True, message="Checking…")
        try:
            self._findings = coherence.check(self._root)
        finally:
            self._spinner.set_busy(False, message="")

        self._tree.blockSignals(True)
        self._tree.clear()

        by_kind: dict[coherence.Kind, list[coherence.Finding]] = {}
        for finding in self._findings:
            by_kind.setdefault(finding.kind, []).append(finding)

        for kind, group in by_kind.items():
            fixable = sum(1 for f in group if f.fixable)
            head = QTreeWidgetItem([
                f"{kind.value} ({len(group)})",
                f"{fixable} repairable" if fixable else "needs a person",
            ])
            self._tree.addTopLevelItem(head)
            head.setExpanded(True)
            for finding in group:
                item = QTreeWidgetItem(head, [
                    self._where(finding), finding.repair or "no repair",
                ])
                item.setToolTip(0, finding.detail)
                item.setData(0, Qt.ItemDataRole.UserRole, finding)
                if finding.fixable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

        self._tree.resizeColumnToContents(0)
        self._tree.blockSignals(False)
        self._show_detail(None)
        self._update_status()

    def _where(self, finding: coherence.Finding) -> str:
        """Where the finding is, relative to the dataset."""
        try:
            rel = finding.path.resolve().relative_to(self._root.resolve())
        except ValueError:
            return finding.path.name
        return rel.as_posix() if rel.parts else "the dataset"

    # -- the detail panel -------------------------------------------------

    def _on_selected(self, current, _previous) -> None:
        finding = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self._show_detail(finding)

    def _show_detail(self, finding: Optional[coherence.Finding]) -> None:
        self._detail.clear()
        if finding is None:
            self._what.setText("Select a finding above.")
            return

        lines = [f"<b>{finding.kind.value}.</b> {finding.detail}"]
        if finding.repair:
            lines.append(f"<br><b>Repair:</b> {finding.repair}.")
        else:
            lines.append(
                "<br><b>No repair is offered</b>, because settling this one "
                "means choosing between two answers that are both valid."
            )
        self._what.setText(" ".join(lines))

        for before, after, count in finding.changes:
            QTreeWidgetItem(self._detail, [
                before or "(not there)", after or "(removed)", str(count),
            ])
        for path in finding.files:
            try:
                rel = path.resolve().relative_to(self._root.resolve()).as_posix()
            except ValueError:
                rel = path.name
            QTreeWidgetItem(self._detail, [rel, "", ""])
        if not finding.changes and not finding.files:
            QTreeWidgetItem(self._detail, [
                self._where(finding),
                finding.repair or "",
                "1" if finding.repair else "",
            ])
        self._detail.resizeColumnToContents(0)
        self._detail.resizeColumnToContents(1)

    # -- selection --------------------------------------------------------

    def _leaves(self):
        for i in range(self._tree.topLevelItemCount()):
            head = self._tree.topLevelItem(i)
            for j in range(head.childCount()):
                yield head.child(j)

    def _checked(self) -> list[coherence.Finding]:
        return [
            item.data(0, Qt.ItemDataRole.UserRole) for item in self._leaves()
            if item.data(0, Qt.ItemDataRole.UserRole)
            and item.data(0, Qt.ItemDataRole.UserRole).fixable
            and item.checkState(0) == Qt.CheckState.Checked
        ]

    def _set_all(self, state: Qt.CheckState) -> None:
        self._tree.blockSignals(True)
        for item in self._leaves():
            finding = item.data(0, Qt.ItemDataRole.UserRole)
            if finding is not None and finding.fixable:
                item.setCheckState(0, state)
        self._tree.blockSignals(False)
        self._update_status()

    def _update_status(self) -> None:
        if not self._findings:
            self._status.setText(
                "Everything agrees. No entity width, scans row, participants "
                "row, link or sidecar contradicts what is on disk."
            )
            self._ok.setEnabled(False)
            return
        n = len(self._checked())
        unfixable = sum(1 for f in self._findings if not f.fixable)
        parts = [f"{len(self._findings)} finding(s), {n} ticked for repair."]
        if unfixable:
            parts.append(f"{unfixable} need a person to decide.")
        self._status.setText(" ".join(parts))
        self._ok.setEnabled(bool(n))

    # -- applying ---------------------------------------------------------

    def _on_apply(self) -> None:
        chosen = self._checked()
        if not chosen:
            return
        standalone = sum(1 for f in chosen if f._apply is None)
        note = (
            "\n\nOne of them is a rename, which is its own undo step."
            if standalone else
            "\n\nThey are applied as one step, so one undo reverses all of them."
        )
        answer = QMessageBox.question(
            self, "Repair",
            f"Apply {len(chosen)} repair(s)?{note}",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        try:
            # Files are deleted, replaced and renamed, so the watcher has to
            # let go: on Windows a watched file cannot be renamed over.
            with watchers_released():
                self._applied += coherence.apply(self._root, chosen)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            QMessageBox.warning(
                self, "Repair failed",
                f"Some repairs may not have run.\n\n{exc}",
            )
        self._rescan()

    def applied_count(self) -> int:
        return self._applied


__all__ = ["CoherenceDialog"]
