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
Repair is pressed, and the whole batch is one entry in the history. Only a
repair that loses nothing comes ticked (adding a row, dropping a row that
names nothing); deleting a file or a participant's row waits for a tick.
The check runs on a thread: on a 10,000-file dataset it froze the window for
eight seconds.

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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import coherence
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.spinner import BusySpinner
from .widgets.splitter import Splitter

log = logging.getLogger(__name__)


class CoherenceDialog(QDialog):
    """What disagrees, which files, and exactly what a repair would do."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._findings: list[coherence.Finding] = []
        self._applied = 0
        self._generation = 0
        self._checking = False
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_checked)
        self._jobs.failed.connect(self._on_check_failed)

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

        split = Splitter(Qt.Orientation.Vertical)

        found, fl = card("What was found")
        fl.addWidget(hint(
            "Grouped by kind. Repairs that lose nothing come ticked; tick the "
            "others yourself after reading what they change. A finding with no "
            "repair is shown so you know about it, with the reason."
        ))
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["What disagrees", "Repair"])
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        self._tree.currentItemChanged.connect(self._on_selected)
        from .widgets.tree_click import toggle_on_click

        toggle_on_click(self._tree)
        fl.addWidget(self._tree, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        self._spinner = BusySpinner()
        tools.addWidget(self._spinner)
        self._rescan_btn = QPushButton("Check again")
        self._rescan_btn.setObjectName("tb-btn")
        self._rescan_btn.clicked.connect(self._rescan)
        tools.addWidget(self._rescan_btn)
        for text, which in (("Select safe ones", "safe"), ("Select all", "all"),
                            ("Select none", "none")):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, w=which: self._select(w))
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
        """Check on a thread; the findings arrive in :meth:`_on_checked`."""
        self._generation += 1
        self._checking = True
        self._spinner.set_busy(True, message="Checking the dataset…")
        self._tree.setEnabled(False)
        self._rescan_btn.setEnabled(False)
        self._ok.setEnabled(False)
        self._status.setText("Checking the dataset…")
        from .app_settings import AppSettings

        # A table that is not there is a finding when tables are written
        # (Settings > Convert, on by default).
        tables = AppSettings.load().post_metadata_tables
        self._jobs.start("check", self._generation, coherence.check, self._root,
                         tables=tables)

    def checking(self) -> bool:
        """Whether a check is still running."""
        return self._checking

    def _on_check_failed(self, _tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._checking = False
        self._spinner.set_busy(False, message="")
        self._tree.setEnabled(True)
        self._rescan_btn.setEnabled(True)
        log.warning("coherence check failed: %s", message)
        self._status.setText(f"The check failed: {message.strip().splitlines()[-1]}")

    def _on_checked(self, _tag: str, generation: int, findings) -> None:
        if generation != self._generation:
            return
        self._checking = False
        self._spinner.set_busy(False, message="")
        self._tree.setEnabled(True)
        self._rescan_btn.setEnabled(True)
        self._findings = list(findings or [])

        self._tree.blockSignals(True)
        self._tree.clear()

        by_kind: dict[coherence.Kind, list[coherence.Finding]] = {}
        for finding in self._findings:
            by_kind.setdefault(finding.kind, []).append(finding)

        for kind, group in by_kind.items():
            fixable = sum(1 for f in group if f.fixable)
            safe = sum(1 for f in group if f.fixable and f.safe)
            if not fixable:
                state = "needs a person"
            elif safe == fixable:
                state = f"{fixable} safe to repair"
            else:
                state = f"{fixable} repairable, review first"
            label = kind.value[:1].upper() + kind.value[1:]
            head = QTreeWidgetItem([f"{label} ({len(group)})", state])
            self._tree.addTopLevelItem(head)
            head.setExpanded(True)
            for finding in group:
                item = QTreeWidgetItem(head, [
                    self._where(finding), finding.repair or "no repair",
                ])
                item.setToolTip(0, finding.detail)
                item.setToolTip(1, finding.repair or finding.no_repair or "")
                item.setData(0, Qt.ItemDataRole.UserRole, finding)
                if finding.fixable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked if finding.safe
                                       else Qt.CheckState.Unchecked)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

        self._tree.resizeColumnToContents(0)
        self._tree.blockSignals(False)
        self._show_detail(None)
        self._update_status()

    def _where(self, finding: coherence.Finding) -> str:
        """What the finding is about: the recording, not only the file the
        repair edits (a missing scans row is about the recording)."""
        if finding.about:
            return finding.about
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

        from html import escape

        lines = [f"<b>{escape(finding.kind.value[:1].upper() + finding.kind.value[1:])}.</b> "
                 f"{escape(finding.detail)}."]
        if finding.repair:
            lines.append(f"<br><b>Repair:</b> {escape(finding.repair)}."
                         + ("" if finding.safe else
                            " Not ticked for you: read what it changes first."))
        else:
            lines.append("<br><b>No repair is offered.</b> "
                         + escape(finding.no_repair or "Settling it needs a decision "
                                  "the files cannot make."))
        self._what.setText(" ".join(lines))

        # The changes name every file they touch, so the files are listed
        # only for a finding without any (one column, its own heading).
        listing = bool(finding.files) and not finding.changes
        self._detail.setColumnCount(1 if listing else 3)
        self._detail.setHeaderLabels(["Files it concerns"] if listing
                                     else ["Now", "Becomes", "Files"])
        for before, after, count in finding.changes:
            QTreeWidgetItem(self._detail, [
                before or "(not there)", after or "(removed)", str(count),
            ])
        for path in finding.files if listing else ():
            try:
                rel = path.resolve().relative_to(self._root.resolve()).as_posix()
            except ValueError:
                rel = path.name
            QTreeWidgetItem(self._detail, [rel])
        if not finding.changes and not finding.files:
            QTreeWidgetItem(self._detail, [
                self._where(finding),
                finding.repair or "",
                "1" if finding.repair else "",
            ])
        from .viz import fonts

        for column in range(self._detail.columnCount() - 1):
            self._detail.resizeColumnToContents(column)
            self._detail.setColumnWidth(
                column, self._detail.columnWidth(column) + fonts.px(24))

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

    def _select(self, which: str) -> None:
        """Tick the safe repairs, all of them, or none."""
        self._tree.blockSignals(True)
        for item in self._leaves():
            finding = item.data(0, Qt.ItemDataRole.UserRole)
            if finding is None or not finding.fixable:
                continue
            on = which == "all" or (which == "safe" and finding.safe)
            item.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._tree.blockSignals(False)
        self._update_status()

    def _update_status(self) -> None:
        if self._checking:
            return
        if not self._findings:
            self._status.setText(
                "Everything agrees. No table row, link, sidecar, name or "
                "citation contradicts what is on disk."
            )
            self._ok.setEnabled(False)
            return
        n = len(self._checked())
        unfixable = sum(1 for f in self._findings if not f.fixable)
        parts = [f"{_count(len(self._findings), 'finding')}, {n} ticked for repair."]
        if unfixable:
            parts.append(f"{unfixable} need{'s' if unfixable == 1 else ''} "
                         "a person to decide.")
        self._status.setText(" ".join(parts))
        self._ok.setEnabled(bool(n))

    # -- applying ---------------------------------------------------------

    def _on_apply(self) -> None:
        chosen = self._checked()
        if not chosen:
            return
        renames = sum(1 for f in chosen if f._apply is None)
        batch = len(chosen) - renames
        notes = []
        if batch:
            notes.append(f"{'It is' if batch == 1 else 'The ' + str(batch) + ' are'} "
                         "applied as one step: one Undo reverses "
                         f"{'it' if batch == 1 else 'all of them'}.")
        if renames:
            notes.append(f"{_count(renames, 'rename')} run{'s' if renames == 1 else ''} "
                         "after, each its own step in the history.")
        answer = QMessageBox.question(
            self, "Repair",
            f"Apply {_count(len(chosen), 'repair')}?\n\n" + " ".join(notes),
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


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


__all__ = ["CoherenceDialog"]
