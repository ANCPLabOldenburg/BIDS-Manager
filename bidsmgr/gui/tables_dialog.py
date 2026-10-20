"""Write the ``*_scans.tsv`` and ``*_sessions.tsv`` tables of a dataset.

For a dataset that has none, or whose tables have fallen behind what is on
disk (one converted elsewhere, or curated by hand). Over
:mod:`bidsmgr.metadata.tables`, the same rule the metadata step writes them
with, so the Editor and a conversion produce the same tables.

Shown before anything is written: per table, the rows added (with the time
found for each), the rows removed and why, the empty cells filled. Tables
already there keep their rows, columns and values. Writing is one step of
the Editor's history. The plan is made on a thread, because it reads every
datatype folder and every sidecar.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.spinner import BusySpinner
from .widgets.tree_click import toggle_on_click

log = logging.getLogger(__name__)

#: Above this many tables the rows start folded, so the list stays readable.
_UNFOLDED = 12


def plan_for(root: Path, scans: bool, sessions: bool):
    """The plan, with what the project's scan knows about times (a PET
    sidecar states a time without a date; the scan has the date)."""
    from ..metadata.tables import plan_tables, times_from_inventory

    times = {}
    try:
        from ..project.orchestration import latest_version

        version = latest_version(Path(root))
        if version is not None and Path(version.inventory).is_file():
            import pandas as pd

            times = times_from_inventory(pd.read_csv(
                version.inventory, sep="\t", dtype=str, keep_default_na=False))
    except Exception:  # noqa: BLE001 - times are a bonus, the tables are not
        log.debug("no scan times for %s", root, exc_info=True)
    return plan_tables(Path(root), scans=scans, sessions=sessions, times=times)


class TablesDialog(QDialog):
    """Preview the scans and sessions tables, then write the ticked ones."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._edits: list = []
        self._written: list[Path] = []
        self._generation = 0
        self._planning = False
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_planned)
        self._jobs.failed.connect(self._on_plan_failed)

        self.setWindowTitle("Scans and sessions tables")
        self.setModal(True)
        self.resize(900, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Scans and sessions tables",
            "A <b>*_scans.tsv</b> per session lists every recording with when it "
            "was acquired, and a <b>*_sessions.tsv</b> per subject lists its "
            "sessions. The standard makes both optional; they are where a dataset "
            "says when. Every datatype the standard defines is listed, converted "
            "here or not. Tables already there keep their rows, columns and "
            "values. <b>Nothing is written until you press Write</b>, and writing "
            "is one step of the Editor's history.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        which, wl = card("Which tables")
        row = QHBoxLayout()
        row.setSpacing(18)
        self._scans = QCheckBox("Scans tables, one per session")
        self._scans.setChecked(True)
        self._scans.setToolTip(
            "filename and acq_time: the sidecar's AcquisitionDateTime, else the "
            "date and time the scan read, else n/a. Never a time without a date.")
        self._sessions = QCheckBox("Sessions tables, one per subject with sessions")
        self._sessions.setChecked(True)
        self._sessions.setToolTip(
            "session_id and acq_time: the session's earliest recording.")
        for box in (self._scans, self._sessions):
            box.toggled.connect(lambda _on: self._replan())
            row.addWidget(box)
        row.addStretch(1)
        wl.addLayout(row)
        bl.addWidget(which)

        found, fl = card("What would change")
        fl.addWidget(hint(
            "Tick the tables to write. Under each: the rows it gains, with the "
            "time found; the rows it loses, with the reason; the empty cells "
            "filled (n/a is the standard's word for not known)."))
        # Where the tables will appear, while they are worked out (rule 4.14);
        # hidden once they are, so it leaves no gap.
        self._busy = QWidget()
        busy = QHBoxLayout(self._busy)
        busy.setContentsMargins(0, 0, 0, 0)
        self._spinner = BusySpinner()
        busy.addWidget(self._spinner)
        self._busy_label = QLabel("")
        self._busy_label.setObjectName("dlg-hint")
        busy.addWidget(self._busy_label, 1)
        fl.addWidget(self._busy)
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Table, and its rows", "Change"])
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        toggle_on_click(self._tree)
        fl.addWidget(self._tree, 1)
        bl.addWidget(found, 1)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Ok)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Write")
        buttons.accepted.connect(self._on_write)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._replan()

    # -- planning ---------------------------------------------------------

    def planning(self) -> bool:
        return self._planning

    def _replan(self) -> None:
        self._generation += 1
        self._planning = True
        self._tree.clear()
        self._tree.setEnabled(False)
        self._ok.setEnabled(False)
        self._busy.setVisible(True)
        self._spinner.set_busy(True, message="")
        self._busy_label.setText("Reading the dataset…")
        self._status.setText("")
        self._jobs.start("plan", self._generation, plan_for, self._root,
                         self._scans.isChecked(), self._sessions.isChecked())

    def _on_plan_failed(self, _tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._spinner.set_busy(False, message="")
        self._busy.setVisible(False)
        log.warning("tables plan failed: %s", message)
        self._status.setText(f"Could not read the dataset: {message.strip().splitlines()[-1]}")

    def _on_planned(self, _tag: str, generation: int, edits) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._spinner.set_busy(False, message="")
        self._busy.setVisible(False)
        self._edits = list(edits or [])
        self._fill()

    def _fill(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        unfold = len(self._edits) <= _UNFOLDED
        for edit in self._edits:
            head = QTreeWidgetItem([self._rel(edit.path), edit.summary()])
            head.setData(0, Qt.ItemDataRole.UserRole, edit)
            head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            head.setCheckState(0, Qt.CheckState.Checked)
            head.setToolTip(0, str(edit.path))
            self._tree.addTopLevelItem(head)
            for text, change in self._lines(edit):
                QTreeWidgetItem(head, [text, change])
            head.setExpanded(unfold)
        self._tree.resizeColumnToContents(0)
        self._tree.setEnabled(True)
        self._tree.blockSignals(False)
        self._update_status()

    @staticmethod
    def _lines(edit) -> list[tuple[str, str]]:
        """The child rows of one table: what it gains, loses, and fills."""
        key = edit.key
        out = []
        added = set(edit.added) if not edit.created else {r[key] for r in edit.rows}
        for row in edit.rows:
            if row[key] not in added:
                continue
            when = row.get("acq_time", "")
            out.append((row[key], f"added, acquired {when}" if when and when != "n/a"
                        else "added, time not known (n/a)"))
        for name, why in edit.dropped:
            out.append((name, f"removed: {why}"))
        if edit.filled:
            out.append((f"{edit.filled} empty cell{'s' if edit.filled != 1 else ''}",
                        "filled with the time where known, else n/a"))
        return out

    def _rel(self, path: Path) -> str:
        try:
            return Path(path).relative_to(self._root).as_posix()
        except ValueError:
            return Path(path).name

    # -- choosing and writing ---------------------------------------------

    def _chosen(self) -> list:
        out = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                out.append(item.data(0, Qt.ItemDataRole.UserRole))
        return out

    def _update_status(self) -> None:
        if self._planning:
            return
        if not self._edits:
            self._status.setText(
                "Every table is as it should be: each recording has its row and "
                "each session its row." if self._scans.isChecked() or
                self._sessions.isChecked() else "Choose which tables above.")
            self._ok.setEnabled(False)
            self._ok.setText("Write")
            return
        chosen = self._chosen()
        new = sum(1 for e in chosen if e.created)
        rows = sum(len(e.rows) if e.created else len(e.added) for e in chosen)
        times = sum(e.times for e in chosen)
        kinds = ", ".join(part for part in (f"{new} new" if new else "",
                                            f"{len(chosen) - new} updated"
                                            if len(chosen) - new else "") if part)
        ticked = (f"{len(chosen)} of {len(self._edits)} tables" if len(chosen) < len(self._edits)
                  else f"{len(chosen)} table{'s' if len(chosen) != 1 else ''}")
        self._status.setText(
            f"{ticked} ({kinds or 'none'}): {rows} row{'s' if rows != 1 else ''} added, "
            f"{times} with an acquisition time." if chosen else "No table ticked.")
        self._ok.setEnabled(bool(chosen))
        self._ok.setText(f"Write {len(chosen)} table{'s' if len(chosen) != 1 else ''}")

    def _on_write(self) -> None:
        from ..metadata.tables import write_tables
        from ..project.operations import begin_operation

        chosen = self._chosen()
        if not chosen:
            return
        label = (f"Write {len(chosen)} scans and sessions table"
                 f"{'s' if len(chosen) != 1 else ''}")
        try:
            with watchers_released(), begin_operation(self._root, label) as op:
                self._written = write_tables(chosen, op)
        except Exception as exc:  # noqa: BLE001 - reported, nothing half-written
            QMessageBox.warning(self, "Could not write the tables",
                                f"Nothing was changed.\n\n{exc}")
            return
        self.accept()

    def written(self) -> list[Path]:
        """The tables written, once the dialog has been accepted."""
        return list(self._written)


__all__ = ["TablesDialog", "plan_for"]
