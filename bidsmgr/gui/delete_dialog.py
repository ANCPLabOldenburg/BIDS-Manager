"""Delete recordings, datatypes and sessions, after showing what goes with them.

The dialog exists because the dangerous part of a deletion is not the files.
It is the ``*_scans.tsv`` row for a recording that is no longer there, the
``IntendedFor`` naming a file nothing has, the ``participants.tsv`` row for a
subject with no data, and the empty ``anat/`` that every tool reading the tree
takes for a modality. All of those are shown here before anything happens, and
all of them are repaired as part of the same step.

It is one entry in the Editor's history, so it can be undone from there. That
is stated on the dialog rather than assumed, because a person deciding whether
to press Delete is entitled to know whether they are deciding permanently.

Every file can be unticked, and the repairs are recomputed from what is
actually going. Unticking a recording leaves its scans row, its reference and
its folder exactly where they were.
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
    QPushButton,
    QSizePolicy,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import remove as rm
from ..editor.rename import RenameError
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.move_preview import MovePreviewTree, delete_extras
from .widgets.scope_bar import ScopeBar
from .widgets.preview_split import (
    PreviewSplit,
    controls_panel,
    preview_toggle,
)


def _human(size: int) -> str:
    """A byte count somebody can read, so the scale of a delete is visible."""
    step = 1024.0
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < step or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} GB"


class DeleteDialog(QDialog):
    """Preview a deletion, choose what goes, then do it."""

    def __init__(
        self,
        root: Path,
        targets: Sequence[Path],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._targets = [Path(t) for t in targets]
        self._plan: Optional[rm.DeletePlan] = None

        self.setWindowTitle("Delete from the dataset")
        self.setModal(True)
        self.resize(820, 620)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Delete from the dataset",
            "The files are the easy part. What breaks a dataset is what is "
            "left naming them, so the <b>*_scans.tsv</b> rows, the "
            "<b>IntendedFor</b> entries, the <b>participants.tsv</b> row of a "
            "subject with nothing left, and any folder this empties are all "
            "repaired in the same step. That step is one entry in the "
            "Editor's history, so it can be undone from there.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        scope_card, sl = card()
        # No dataset-wide entry: ``remove.plan_delete`` refuses the dataset
        # root, so offering it would only produce a refusal. Deleting
        # everything is not a tool, it is a decision for the file manager.
        self._scope_bar = ScopeBar(
            self._root, self._targets, allow_dataset=False,
            show_summary=False, parent=self,
        )
        self._targets = self._scope_bar.targets()
        self._scope_bar.changed.connect(self._on_scope_changed)
        sl.addWidget(self._scope_bar)

        self._scope = QLabel("")
        self._scope.setObjectName("dlg-hint")
        self._scope.setWordWrap(True)
        sl.addWidget(self._scope)
        scope_card.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )
        bl.addWidget(scope_card)

        self._summary = hint("")
        bl.addWidget(self._summary)

        preview_card, pl = card("What would be deleted")
        pl.addWidget(hint(
            "Untick anything you want to keep. The repairs below the files "
            "are recalculated from what is actually going, so a recording you "
            "leave keeps its scans row and everything pointing at it."
        ))
        self._preview = MovePreviewTree()
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
        # Controls and preview in a splitter the user can flip
        # between stacked and side by side. See preview_split.py.
        self._split = PreviewSplit(
            controls_panel(scope_card, self._summary), preview_card, name="delete",
        )
        bl.addWidget(self._split, 1)
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
        self._ok.setText("Delete")
        self._ok.setEnabled(False)
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        buttons.addButton(
            preview_toggle(self._split), QDialogButtonBox.ButtonRole.ResetRole
        )
        outer.addWidget(build_footer_with(self._status, buttons))

        self._refresh_plan()

    # -- planning --------------------------------------------------------

    def _on_scope_changed(self) -> None:
        self._targets = self._scope_bar.targets()
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        shown = [_rel(self._root, t) for t in self._targets[:3]]
        more = len(self._targets) - len(shown)
        where = ", ".join(shown) + (f" and {more} more" if more > 0 else "")

        plan = rm.plan_delete(self._root, self._targets)
        self._plan = plan
        self._scope.setText(
            f"<b>{where}</b><br>{plan.n_files} file(s), "
            f"{_human(plan.bytes_freed())}, companions included"
        )
        self._preview.show_removals(
            self._root,
            [(plan.file_key(self._root, p), p) for p in plan.files],
            extras=delete_extras(plan),
            conflicts=plan.conflicts,
        )

        if plan.is_empty:
            self._summary.setText(
                "Nothing here can be deleted."
                if plan.conflicts else "There is nothing here to delete."
            )
            self._status.setText("")
            self._ok.setEnabled(False)
            return
        self._summary.setText(f"Would remove {plan.summary()}.")
        self._refresh_selection()

    # -- selection -------------------------------------------------------

    def _set_all(self, state: Qt.CheckState) -> None:
        self._preview.set_all(state)
        self._refresh_selection()

    def _on_item_checked(self, item: QTreeWidgetItem, column: int) -> None:
        del item, column
        self._refresh_selection()

    def selected_keys(self) -> set[str]:
        return self._preview.selected_keys()

    def _refresh_selection(self) -> None:
        if self._plan is None:
            return
        chosen = self.selected_keys()
        total = len(self._plan.files)
        self._ok.setEnabled(bool(chosen))
        if total and len(chosen) < total:
            self._status.setText(f"{len(chosen)} of {total} file(s) selected")
            self._ok.setText(f"Delete {len(chosen)} file(s)")
        else:
            self._status.setText(self._plan.summary())
            self._ok.setText(f"Delete {total} file(s)")

    # -- doing it --------------------------------------------------------

    def _on_apply(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        if self._plan is None:
            return
        chosen = self.selected_keys()
        if not chosen:
            return
        confirm = QMessageBox.question(
            self, "Delete from the dataset",
            f"Delete {len(chosen)} file(s)?\n\n"
            "The scans rows, IntendedFor entries and emptied folders go with "
            "them, as one step. You can undo it from the Editor's history.",
            QMessageBox.StandardButton.Cancel
            | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Ok:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Folders are removed here, and on Windows a watched directory
            # cannot be removed or renamed: QFileSystemWatcher holds an open
            # handle on it. See bidsmgr.gui.fs_watch.
            with watchers_released():
                touched, errors = rm.apply_delete(
                    self._root, self._plan, only=chosen,
                )
        except RenameError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Refused", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(
                self, "Some steps failed",
                f"{touched} item(s) changed, but:\n\n" + "\n".join(errors[:8]),
            )
        self.accept()


def _rel(root: Path, path: Path) -> str:
    """POSIX, always, so a label reads the same on all three platforms."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).name


__all__ = ["DeleteDialog"]
