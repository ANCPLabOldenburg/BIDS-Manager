"""Remove faces from anatomical images, after showing exactly which ones.

Two things make this dialog different from the other destructive ones.

**The skipped list matters as much as the chosen list.** A user told "4 images
defaced" and not told that three more were skipped walks away with a dataset
that still has a face in it and no reason to suspect one. So every image the
selector walked past is on screen, either ticked or with the reason it cannot
be done.

**Where the original goes is a privacy decision, not a convenience one.** The
originals are kept in ``.bidsmgr/`` by default, a dot folder that BIDS tools
ignore, so one Undo puts the dataset back. The alternative, a visible mirror in
``sourcedata/``, is offered and is *not* the default, because that copy still
contains the face: a dataset shared with it in place is not deidentified. The
checkbox says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..deface import derivatives, engines
from ..deface.run import unavailable_reason
from ..deface.select import Selection, walk
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.spinner import BusySpinner


# Everything that differs between removing a face and removing the skull.
# A table rather than conditionals scattered through the dialog: the two
# operations are the same shape, and the only honest way to say that is to
# build one dialog from the differences.
_COPY = {
    engines.KIND_DEFACE: {
        "window": "Remove faces",
        "title": "Remove faces from anatomical images",
        "blurb": (
            "A head MRI contains a face, and a face can be rendered from one, "
            "so a dataset shared with its faces intact is a dataset shared "
            "with its participants identifiable. This blanks the face and "
            "leaves the brain alone. It is <b>one entry in the Editor's "
            "history</b>, so it can be undone from there, and nothing happens "
            "until you press the button."
        ),
        "button": "Remove faces",
        "verb": "defaced",
        "busy": "Removing faces…",
        "progress": "Removing the face from",
        "option": "Also keep the originals in sourcedata/",
        "option_tip": (
            "Off by default, and worth leaving off. Those copies still "
            "contain the face, so a dataset shared with them in place is not "
            "deidentified. Undo does not need them: the originals are kept "
            "in .bidsmgr/ either way."
        ),
        "confirm_title": "Remove faces",
        "confirm": (
            "Remove the face from {n} image(s)?\n\nThis rewrites the images "
            "in place. It is one entry in the Editor's history, so Edit > "
            "Undo puts them back."
        ),
        "outcome_note": (
            "The originals are in sourcedata/. They still contain the face, "
            "so remove them before sharing this dataset."
        ),
    },
    engines.KIND_STRIP: {
        "window": "Remove the skull",
        "title": "Keep only the brain",
        "blurb": (
            "Skull stripping throws away everything outside the brain, which "
            "removes the face and a great deal else. The result is a "
            "<b>derivative</b>, not raw data, so it is written to "
            "<code>derivatives/</code> and the original scan is left exactly "
            "as it is. One entry in the Editor's history, and nothing happens "
            "until you press the button."
        ),
        "button": "Remove the skull",
        "verb": "stripped",
        "busy": "Removing the skull…",
        "progress": "Extracting the brain from",
        "option": "Overwrite the original instead of writing a derivative",
        "option_tip": (
            "Off by default. A skull-stripped scan is not raw data, and "
            "writing it over the original leaves a dataset whose raw images "
            "have been processed with nothing in the tree saying so. Undo "
            "still works either way."
        ),
        "confirm_title": "Remove the skull",
        "confirm": (
            "Keep only the brain in {n} image(s)?\n\nEverything outside the "
            "brain is discarded. It is one entry in the Editor's history, so "
            "Edit > Undo puts it back."
        ),
        "outcome_note": (
            "The originals have been overwritten. Edit > Undo restores them."
        ),
    },
}


def _human(size: int) -> str:
    step = 1024.0
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < step or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} GB"


class DefaceDialog(QDialog):
    """Preview a defacing run, choose what goes, then do it."""

    def __init__(
        self,
        root: Path,
        targets: Optional[Sequence[Path]] = None,
        parent: Optional[QWidget] = None,
        *,
        kind: str = engines.KIND_DEFACE,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._copy = _COPY[kind]
        self._root = Path(root)
        self._targets = [Path(t) for t in targets] if targets else None
        self._selection: Optional[Selection] = None
        # Set while a run owns the dataset. The worker is parented to this
        # dialog, so `done` has to stop it before Qt destroys it.
        self._worker = None
        self._watchers = None
        self._outcome = None

        self.setWindowTitle(self._copy["window"])
        self.setModal(True)
        self.resize(860, 660)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            self._copy["title"], self._copy["blurb"],
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        # --- scope -------------------------------------------------------
        scope_card, sl = card()
        self._scope = QLabel("")
        self._scope.setObjectName("dlg-hint")
        self._scope.setWordWrap(True)
        sl.addWidget(self._scope)
        scope_card.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )
        bl.addWidget(scope_card)

        # --- engine ------------------------------------------------------
        engine_card, el = card("How")
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Engine:"))
        self._engine = QComboBox()
        self._engine.setObjectName("ent-input")
        offered = engines.engines_of(kind)
        for eng in offered:
            self._engine.addItem(eng.label, eng.id)
        default = (
            engines.DEFAULT_STRIP_ENGINE_ID
            if kind == engines.KIND_STRIP else engines.DEFAULT_ENGINE_ID
        )
        self._engine.setCurrentIndex(
            [e.id for e in offered].index(default)
        )
        self._engine.currentIndexChanged.connect(self._on_engine_changed)
        row.addWidget(self._engine, 1)
        el.addLayout(row)
        self._engine_note = hint("")
        el.addWidget(self._engine_note)

        self._keep = QCheckBox(self._copy["option"])
        self._keep.setToolTip(self._copy["option_tip"])
        self._keep.toggled.connect(lambda *_a: self._update_status())
        el.addWidget(self._keep)
        engine_card.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )
        bl.addWidget(engine_card)

        # --- what would happen -------------------------------------------
        preview_card, pl = card("What would happen")
        pl.addWidget(hint(
            "Untick anything you want left alone. Images that cannot be "
            "defaced are listed underneath with the reason, because an image "
            "quietly skipped is a face quietly kept."
        ))
        self._preview = QTreeWidget()
        # Same flat, themed tick as every other checkable
        # preview; without it Qt paints the platform control,
        # which does not follow the palette.
        self._preview.setObjectName("check-tree")
        self._preview.setColumnCount(2)
        self._preview.setHeaderLabels(["In the dataset", "What happens"])
        self._preview.setRootIsDecorated(True)
        self._preview.setUniformRowHeights(True)
        self._preview.itemChanged.connect(self._on_item_checked)
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

        # Progress, and the one control that matters while it runs. Both
        # hidden until there is something to report: an empty bar sitting in a
        # dialog reads as a thing that is broken.
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
            "Stop after the image being worked on. Everything already done is "
            "put back, so the dataset is never left half defaced."
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
        self._ok.setText(self._copy["button"])
        self._ok.setEnabled(False)
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._on_engine_changed()
        self._refresh()

    # -- planning --------------------------------------------------------

    def _current_engine(self) -> engines.Engine:
        return engines.engine(self._engine.currentData())

    def _on_engine_changed(self, *_a) -> None:
        self._engine_note.setText(self._current_engine().description)
        # Availability is per ENGINE, not per feature: mindgrab needs an extra
        # that the atlas engine does not, so a missing extra must grey out the
        # one engine rather than the whole dialog.
        self._update_status()

    def _refresh(self) -> None:
        self._selection = walk(self._root, self._targets)
        sel = self._selection

        where = "the whole dataset"
        if self._targets:
            shown = [
                str(Path(t).relative_to(self._root)) if t != self._root else "."
                for t in self._targets[:3]
            ]
            more = len(self._targets) - len(shown)
            where = ", ".join(shown) + (f" and {more} more" if more else "")

        self._scope.setText(
            f"<b>{where}</b><br>{len(sel.candidates)} image(s) can be "
            f"{self._copy['verb']}, {_human(sel.total_bytes)}. "
            f"{len(sel.skipped)} skipped."
        )

        self._preview.blockSignals(True)
        self._preview.clear()
        for cand in sel.candidates:
            item = QTreeWidgetItem(self._preview, [cand.relative, ""])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, cand.relative)
            note = (
                "brain kept, everything else removed"
                if self._kind == engines.KIND_STRIP else "face removed"
            )
            if cand.previous_engine:
                note = f"done again (was {cand.previous_engine})"
            elif cand.foreign_methods:
                note = f"{note}; another tool also deidentified this"
            item.setText(1, note)

        if sel.skipped:
            # Grouped by reason rather than listed flat. Seven fieldmap files
            # skipped for one reason is one fact, and reading it seven times
            # buries the one skip that might matter.
            by_reason: dict[str, list] = {}
            for skip in sel.skipped:
                by_reason.setdefault(skip.reason.value, []).append(skip)

            head = QTreeWidgetItem(
                self._preview,
                [f"Cannot be done ({len(sel.skipped)})", ""],
            )
            head.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for reason, group in sorted(by_reason.items()):
                node = QTreeWidgetItem(head, [reason, f"{len(group)} file(s)"])
                node.setFlags(Qt.ItemFlag.ItemIsEnabled)
                for skip in group:
                    detail = f"({skip.detail})" if skip.detail else ""
                    child = QTreeWidgetItem(node, [skip.relative, detail])
                    child.setFlags(Qt.ItemFlag.ItemIsEnabled)
                node.setExpanded(len(group) <= 4)
            head.setExpanded(True)

        # Fit the path column to its content, but never past half the width:
        # a long session path would otherwise push the reason off the right
        # edge, and the reason is the column that stops somebody shipping a
        # face they did not know was still there.
        self._preview.resizeColumnToContents(0)
        cap = max(240, int(self._preview.viewport().width() * 0.55))
        if self._preview.columnWidth(0) > cap:
            self._preview.setColumnWidth(0, cap)
        self._preview.blockSignals(False)

        self._update_status()

    def _checked(self) -> list[str]:
        out: list[str] = []
        for i in range(self._preview.topLevelItemCount()):
            item = self._preview.topLevelItem(i)
            rel = item.data(0, Qt.ItemDataRole.UserRole)
            if rel and item.checkState(0) == Qt.CheckState.Checked:
                out.append(str(rel))
        return out

    def _set_all(self, state: Qt.CheckState) -> None:
        self._preview.blockSignals(True)
        for i in range(self._preview.topLevelItemCount()):
            item = self._preview.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole):
                item.setCheckState(0, state)
        self._preview.blockSignals(False)
        self._update_status()

    def _on_item_checked(self, *_a) -> None:
        self._update_status()

    def _update_status(self) -> None:
        reason = unavailable_reason(self._current_engine().id)
        if reason:
            self._status.setText(reason)
            self._ok.setEnabled(False)
            return
        n = len(self._checked())
        if not n:
            self._status.setText("Nothing selected.")
            self._ok.setEnabled(False)
            return
        if self._kind == engines.KIND_STRIP and not self._keep.isChecked():
            where = (
                f"The results go to derivatives/{derivatives.PIPELINE}/; "
                "the original scans are untouched."
            )
        else:
            where = (
                "The originals stay in .bidsmgr/, so Undo restores them "
                "exactly."
            )
        self._status.setText(
            f"{n} image(s) will be {self._copy['verb']}. {where}"
        )
        self._ok.setEnabled(True)

    # -- doing it --------------------------------------------------------

    def _on_apply(self) -> None:
        chosen = self._checked()
        if not chosen or self._selection is None:
            return

        answer = QMessageBox.question(
            self, self._copy["confirm_title"],
            self._copy["confirm"].format(n=len(chosen)),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return

        self._start(chosen)

    # -- running it, off the GUI thread ----------------------------------

    def _start(self, chosen: list[str]) -> None:
        """Hand the work to a QThread and let the dialog keep painting.

        Defacing one image takes seconds and forty take minutes. Done inline,
        the window stops repainting and the OS marks it unresponsive, which
        looks exactly like a crash. The user needs to see which file is being
        worked on and how far through it is.
        """
        from ..workers import DefaceWorker

        self._set_running(True, len(chosen))

        # Images are replaced, so the watcher has to let go first: on Windows
        # a watched file cannot be renamed over. Held for the whole run, which
        # is why it is entered here and exited when the worker finishes.
        self._watchers = watchers_released()
        self._watchers.__enter__()

        strip = self._kind == engines.KIND_STRIP
        worker = DefaceWorker(
            self._root,
            selection=self._selection,
            engine_id=self._current_engine().id,
            only=chosen,
            # One checkbox, two meanings, because the two operations have one
            # question each and it is not the same question. For a deface it
            # asks for a visible copy of the original; for a strip it asks to
            # overwrite the original instead of writing a derivative.
            keep_original_in_sourcedata=(
                False if strip else self._keep.isChecked()
            ),
            in_place=self._keep.isChecked() if strip else None,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _set_running(self, running: bool, total: int = 0) -> None:
        """Lock the choices while the run owns the dataset."""
        self._progress.setVisible(running)
        self._stop.setVisible(running)
        self._stop.setEnabled(running)
        if running:
            self._progress.setRange(0, max(total, 1))
            self._progress.setValue(0)
        self._ok.setEnabled(not running)
        for widget in (self._preview, self._engine, self._keep):
            widget.setEnabled(not running)
        self._spinner.set_busy(
            running, message=self._copy["busy"] if running else "",
        )

    def _on_progress(self, done: int, total: int, rel: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status.setText(
            f"{self._copy['progress']} {rel} ({done + 1} of {total})…"
            if rel else "Finishing…"
        )

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._stop.setEnabled(False)
            self._status.setText("Stopping and putting back what was done…")
            self._worker.cancel()

    def _release(self) -> None:
        """Let the file watcher take the dataset back, exactly once."""
        watchers, self._watchers = self._watchers, None
        if watchers is not None:
            watchers.__exit__(None, None, None)

    def _on_worker_failed(self, tb: str) -> None:
        self._release()
        self._set_running(False)
        self._worker = None
        QMessageBox.warning(
            self, "Defacing failed",
            "Defacing stopped because of an unexpected error. Nothing was "
            f"changed.\n\n{tb.strip().splitlines()[-1]}",
        )
        self._update_status()

    def _on_done(self, outcome) -> None:
        self._release()
        self._set_running(False)
        self._worker = None
        self._outcome = outcome

        if outcome.cancelled:
            # _update_status first, then the message: it re-enables the button
            # from the ticked count AND rewrites the status line, so setting
            # the text before it would be silently thrown away.
            self._update_status()
            self._status.setText(
                "Stopped. Nothing was changed: everything already done was "
                "put back."
            )
            return

        if outcome.failed:
            where, why = outcome.failed[0]
            self._update_status()
            QMessageBox.warning(
                self, "Defacing failed",
                f"{where} could not be defaced:\n\n{why}\n\n"
                "Nothing was changed: the whole run was rolled back.",
            )
            return

        if outcome.produced:
            QMessageBox.information(
                self, "Done",
                f"Kept only the brain in {len(outcome.defaced)} image(s).\n\n"
                "The results are under derivatives/"
                f"{derivatives.PIPELINE}/. The original scans are untouched.",
            )
        elif self._keep.isChecked():
            QMessageBox.information(
                self, "Done",
                f"{len(outcome.defaced)} image(s) done.\n\n"
                + self._copy["outcome_note"],
            )
        self.accept()

    def _stop_worker(self) -> None:
        """Stop a run and wait for its thread, before anything is destroyed.

        The worker is parented to this dialog, so letting the dialog go first
        destroys a RUNNING QThread, and Qt answers that by aborting the
        process rather than raising. Stopping rolls the dataset back, so
        closing mid-run leaves it exactly as it was.
        """
        worker, self._worker = self._worker, None
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(60_000)
        self._release()

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        self._stop_worker()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        # close() on a dialog that was never shown does not always reach
        # done(), and a worker left running past this point is a crash.
        self._stop_worker()
        super().closeEvent(event)

    def result_count(self) -> int:
        """How many were chosen. For the caller's log line."""
        return len(self._checked())

    def outcome(self):
        """What the run did, for the caller. ``None`` before it has run."""
        return self._outcome


__all__ = ["DefaceDialog"]
