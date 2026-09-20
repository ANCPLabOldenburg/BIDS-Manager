"""Put the face back, from the copy that still has one.

Edit > Undo already reverses a defacing, and it is the right tool for about
five minutes. After a dozen other edits it is not reachable without undoing all
of them, and the history does not live forever. The copy in ``sourcedata/``
does, so this restores from that instead, as its own undoable operation.

The dialog is deliberately the same shape as the defacing one: a preview you
can untick, a progress bar, a Stop that rolls back rather than leaving half a
dataset restored. Somebody who has used one has used both.

It also says where each original is coming from. "From the copy in
sourcedata/" and "from the edit history" are different promises: the first
survives anything, the second only until that history is cleared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..deface.revert import revertable
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.spinner import BusySpinner


class DefaceRevertDialog(QDialog):
    """Choose which defaced images to restore, then restore them."""

    def __init__(
        self,
        root: Path,
        targets: Optional[Sequence[Path]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._items = revertable(
            self._root, [Path(t) for t in targets] if targets else None,
        )
        self._worker = None
        self._watchers = None
        self._outcome = None

        self.setWindowTitle("Restore from the original")
        self.setModal(True)
        self.resize(820, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Put the face back",
            "Restores defaced images from the undefaced copies this tool "
            "kept. Each restored image also stops recording that it was "
            "defaced, so it can be defaced again later. <b>The copies "
            "themselves are kept</b>, so nothing here is one-way.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        preview_card, pl = card("What would be restored")
        pl.addWidget(hint(
            "Only images this tool defaced and still has an undefaced copy of "
            "are listed. An image defaced during conversion never had one, "
            "which is deliberate: that is what makes converting with defacing "
            "safer than defacing afterwards."
        ))
        self._preview = QTreeWidget()
        # Same flat, themed tick as every other checkable
        # preview; without it Qt paints the platform control,
        # which does not follow the palette.
        self._preview.setObjectName("check-tree")
        self._preview.setColumnCount(2)
        self._preview.setHeaderLabels(["In the dataset", "Restored from"])
        self._preview.setRootIsDecorated(False)
        self._preview.setUniformRowHeights(True)
        self._preview.itemChanged.connect(lambda *_a: self._update_status())
        pl.addWidget(self._preview, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        for text, state in (
            ("Select all", Qt.CheckState.Checked),
            ("Select none", Qt.CheckState.Unchecked),
        ):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, st=state: self._set_all(st))
            tools.addWidget(btn)
        tools.addStretch(1)
        pl.addLayout(tools)
        bl.addWidget(preview_card, 1)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self._spinner = BusySpinner()
        run_row.addWidget(self._spinner)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        run_row.addWidget(self._progress, 1)
        self._stop = QPushButton("Stop")
        self._stop.setObjectName("tb-btn")
        self._stop.setVisible(False)
        self._stop.setToolTip(
            "Stop after the image being worked on. Everything already "
            "restored is put back as it was, so the dataset is never left "
            "half restored."
        )
        self._stop.clicked.connect(self._on_stop)
        run_row.addWidget(self._stop)
        bl.addLayout(run_row)

        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Restore")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._fill()

    # -- preview ---------------------------------------------------------

    def _fill(self) -> None:
        self._preview.blockSignals(True)
        self._preview.clear()
        for item in self._items:
            row = QTreeWidgetItem([item.relative, item.original.description])
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            row.setCheckState(0, Qt.CheckState.Checked)
            row.setData(0, Qt.ItemDataRole.UserRole, item.relative)
            self._preview.addTopLevelItem(row)
        self._preview.resizeColumnToContents(0)
        self._preview.blockSignals(False)
        self._update_status()

    def _rows(self) -> list[QTreeWidgetItem]:
        return [
            self._preview.topLevelItem(i)
            for i in range(self._preview.topLevelItemCount())
        ]

    def _checked(self) -> list[str]:
        return [
            r.data(0, Qt.ItemDataRole.UserRole) for r in self._rows()
            if r.checkState(0) == Qt.CheckState.Checked
        ]

    def _set_all(self, state: Qt.CheckState) -> None:
        self._preview.blockSignals(True)
        for row in self._rows():
            row.setCheckState(0, state)
        self._preview.blockSignals(False)
        self._update_status()

    def _update_status(self) -> None:
        if not self._items:
            self._status.setText(
                "Nothing here can be restored. Either these images were not "
                "defaced by BIDS Manager, or they were defaced during "
                "conversion, in which case an undefaced copy never existed."
            )
            self._ok.setEnabled(False)
            return
        n = len(self._checked())
        self._ok.setEnabled(bool(n))
        self._status.setText(
            f"{n} image(s) will be restored. One undoable step."
            if n else "Nothing selected."
        )

    # -- running ---------------------------------------------------------

    def _on_apply(self) -> None:
        chosen = self._checked()
        if not chosen:
            return
        answer = QMessageBox.question(
            self, "Put the face back",
            f"Restore {len(chosen)} image(s) from their undefaced copies?\n\n"
            "Those images will contain identifiable faces again.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return

        from ..workers import DefaceRevertWorker

        self._set_running(True, len(chosen))
        # Images are replaced, so the watcher has to let go: on Windows a
        # watched file cannot be renamed over.
        self._watchers = watchers_released()
        self._watchers.__enter__()

        worker = DefaceRevertWorker(
            self._root, items=self._items, only=chosen, parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _set_running(self, running: bool, total: int = 0) -> None:
        self._progress.setVisible(running)
        self._stop.setVisible(running)
        self._stop.setEnabled(running)
        if running:
            self._progress.setRange(0, max(total, 1))
            self._progress.setValue(0)
        self._ok.setEnabled(not running)
        self._preview.setEnabled(not running)
        self._spinner.set_busy(
            running, message="Restoring…" if running else "",
        )

    def _on_progress(self, done: int, total: int, rel: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status.setText(
            f"Restoring {rel} ({done + 1} of {total})…" if rel else "Finishing…"
        )

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._stop.setEnabled(False)
            self._status.setText("Stopping and putting back what was done…")
            self._worker.cancel()

    def _release(self) -> None:
        watchers, self._watchers = self._watchers, None
        if watchers is not None:
            watchers.__exit__(None, None, None)

    def _on_worker_failed(self, tb: str) -> None:
        self._release()
        self._set_running(False)
        self._worker = None
        self._update_status()
        QMessageBox.warning(
            self, "Restore failed",
            "Restoring stopped because of an unexpected error. Nothing was "
            f"changed.\n\n{tb.strip().splitlines()[-1]}",
        )

    def _on_done(self, outcome) -> None:
        self._release()
        self._set_running(False)
        self._worker = None
        self._outcome = outcome

        if outcome.cancelled:
            self._update_status()
            self._status.setText(
                "Stopped. Nothing was changed: everything already restored "
                "was put back."
            )
            return
        if outcome.failed:
            where, why = outcome.failed[0]
            self._update_status()
            QMessageBox.warning(
                self, "Restore failed",
                f"{where} could not be restored:\n\n{why}\n\n"
                "Nothing was changed: the whole run was rolled back.",
            )
            return
        if outcome.skipped:
            QMessageBox.information(
                self, "Partly restored",
                f"Restored {len(outcome.reverted)} image(s). "
                f"{len(outcome.skipped)} could not be: the undefaced copy is "
                "no longer there.",
            )
        self.accept()

    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(60_000)
        self._release()

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        self._stop_worker()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._stop_worker()
        super().closeEvent(event)

    def outcome(self):
        """What the run did, for the caller. ``None`` before it has run."""
        return self._outcome


__all__ = ["DefaceRevertDialog"]
