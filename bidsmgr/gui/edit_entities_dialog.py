"""Add or remove an entity, and create or remove a session.

Two menu items, one dialog, because underneath they are one operation. A
session is an ordinary entity that happens to name a folder, so "create a
session for this datatype" is "set ``ses-<label>`` on these files" and the
folder follows from the schema. The alternative was two dialogs that both had
to know about scans tables, ``IntendedFor`` and folders left empty, and would
eventually disagree about one of them.

What the schema decides, this dialog does not offer:

* only entities every selected file is ALLOWED to carry appear in Add, so
  ``echo`` is not offered for an EEG recording;
* only entities that are OPTIONAL appear in Remove, so ``task`` cannot be
  taken off a ``_bold``, which would leave a name the standard has no reading
  of;
* the value box says what the entity's format is in the schema's words, and
  offers the values the dataset already uses, because a second session is
  almost always spelled like the first one.

Nothing happens until the preview is there. The button is dead until a plan
exists, and every file in the plan can be unticked, so "this run only" and
"everything but that one" are both one step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import rename as rn
from ..editor import restructure as rs
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.move_preview import MovePreviewTree, plan_extras

# Matches the rename dialog: planning walks the dataset, so doing it on every
# keystroke froze the window on anything real.
_REPLAN_DELAY_MS = 300

ADD = "add"
REMOVE = "remove"


class EditEntitiesDialog(QDialog):
    """Set or clear one entity across a chosen part of the dataset."""

    def __init__(
        self,
        root: Path,
        targets: Sequence[Path],
        parent: Optional[QWidget] = None,
        *,
        mode: str = ADD,
        entity: str = "",
        session_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._targets = [Path(t) for t in targets]
        self._plan: Optional[rn.RenamePlan] = None
        self._session_mode = session_mode
        self._slots: list[rs.EntitySlot] = []

        self.setWindowTitle(
            "Sessions" if session_mode else "Add or remove an entity"
        )
        self.setModal(True)
        self.resize(780, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(*self._headline()))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        chooser, cl = card()
        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._scope = QLabel("")
        self._scope.setObjectName("dlg-hint")
        self._scope.setWordWrap(True)
        form.addRow("Applies to:", self._scope)

        self._mode = QComboBox()
        self._mode.setObjectName("ent-input")
        if session_mode:
            self._mode.addItem("Create a session", ADD)
            self._mode.addItem("Remove the session", REMOVE)
        else:
            self._mode.addItem("Add or change an entity", ADD)
            self._mode.addItem("Remove an entity", REMOVE)
        self._mode.setCurrentIndex(1 if mode == REMOVE else 0)
        self._mode.currentIndexChanged.connect(self._reload_entities)
        form.addRow("Action:", self._mode)

        self._entity = QComboBox()
        self._entity.setObjectName("ent-input")
        self._entity.currentIndexChanged.connect(self._on_entity_changed)
        self._entity_row_label = QLabel("Entity:")
        form.addRow(self._entity_row_label, self._entity)

        # Editable, because the value is often a new one, with the values the
        # dataset already uses in the list, because it is just as often not.
        self._value = QComboBox()
        self._value.setObjectName("ent-input")
        self._value.setEditable(True)
        self._value_label = QLabel("Value:")
        self._replan = QTimer(self)
        self._replan.setSingleShot(True)
        self._replan.setInterval(_REPLAN_DELAY_MS)
        self._replan.timeout.connect(self._refresh_plan)
        self._value.currentTextChanged.connect(self._on_typed)
        self._value.lineEdit().returnPressed.connect(self.plan_now)
        form.addRow(self._value_label, self._value)
        self._form = form
        cl.addLayout(form)

        self._what = hint("")
        cl.addWidget(self._what)
        chooser.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )
        bl.addWidget(chooser)

        self._summary = hint("")
        bl.addWidget(self._summary)

        preview_card, pl = card("What this would do")
        pl.addWidget(hint(
            "Untick anything you want left alone. The <b>*_scans.tsv</b> rows "
            "and the <b>IntendedFor</b> entries follow the files that actually "
            "move, so what you leave behind keeps its own name and nothing "
            "ends up pointing at a name that does not exist."
            + (
                " A session folder goes only once the last file has left it."
                if session_mode else ""
            )
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
            btn.clicked.connect(
                lambda _c=False, st=state: self._set_all(st)
            )
            tools.addWidget(btn)
        tools.addStretch(1)
        pl.addLayout(tools)
        bl.addWidget(preview_card, 1)
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
        self._ok.setText("Apply")
        self._ok.setEnabled(False)
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._describe_scope()
        self._reload_entities()
        if entity:
            at = self._entity.findData(entity)
            if at >= 0:
                self._entity.setCurrentIndex(at)

    # -- wording ---------------------------------------------------------

    def _headline(self) -> tuple[str, str]:
        if self._session_mode:
            return (
                "Sessions",
                "A session is a folder and an entity at once. Creating one "
                "moves the recordings into <b>ses-&lt;label&gt;/</b> and puts "
                "<b>ses-</b> in their names; removing one brings them back "
                "out and takes it off. The <b>*_scans.tsv</b> travels to the "
                "level BIDS puts it at, and every reference follows.",
            )
        return (
            "Add or remove an entity",
            "Only entities the <b>schema</b> allows for these files are "
            "offered, and only ones it treats as optional can be removed. The "
            "new entity is placed in the position the standard puts it in, "
            "not where it was typed, and sidecars and companion files travel "
            "with the recording they belong to.",
        )

    def _describe_scope(self) -> None:
        files = rs.expand(self._root, self._targets)
        self._files = files
        shown = [_rel(self._root, t) for t in self._targets[:3]]
        more = len(self._targets) - len(shown)
        where = ", ".join(shown) + (f" and {more} more" if more > 0 else "")
        self._scope.setText(
            f"<b>{where}</b><br>{len(files)} file(s), companions included"
            if where else f"{len(files)} file(s)"
        )

    # -- the entity list -------------------------------------------------

    def _is_removing(self) -> bool:
        return self._mode.currentData() == REMOVE

    def _reload_entities(self) -> None:
        removing = self._is_removing()
        # ``setRowVisible`` rather than hiding the two widgets: a hidden widget
        # still occupies its row in a QFormLayout, so the dialog kept a blank
        # gap where the value box had been.
        self._form.setRowVisible(self._value, not removing)

        if self._session_mode:
            # The entity is not a choice here, it is the point of the dialog.
            self._slots = [
                s for s in (
                    rs.removable_entities(self._root, self._files) if removing
                    else rs.addable_entities(self._root, self._files)
                ) if s.key == "ses"
            ]
        elif removing:
            self._slots = rs.removable_entities(self._root, self._files)
        else:
            self._slots = rs.addable_entities(self._root, self._files)

        self._entity.blockSignals(True)
        self._entity.clear()
        for slot in self._slots:
            self._entity.addItem(
                f"{slot.display}  ({slot.key}-)", slot.key,
            )
            self._entity.setItemData(
                self._entity.count() - 1,
                f"{slot.description}\n\nValue format: {slot.kind} "
                f"({slot.pattern})",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._entity.setEnabled(bool(self._slots))
        self._entity.blockSignals(False)

        # In session mode the entity is not a choice, it is the subject of
        # the dialog, so the row is not shown at all.
        self._form.setRowVisible(self._entity, not self._session_mode)

        if not self._slots:
            self._what.setText(self._nothing_offered())
            self._preview.clear()
            self._plan = None
            self._ok.setEnabled(False)
            self._summary.setText("")
            self._status.setText("")
            return
        self._on_entity_changed()

    def _nothing_offered(self) -> str:
        if self._session_mode:
            return (
                "These files are not in a session, so there is none to remove."
                if self._is_removing() else
                "These files cannot take a session. That happens when the "
                "selection is not inside a subject, or already has one."
            )
        return (
            "Everything these files carry is required by the schema, so none "
            "of it can be removed."
            if self._is_removing() else
            "The schema allows these files no entity they do not already "
            "have. Selecting fewer files at once usually offers more, since "
            "only entities valid for EVERY selected file are listed."
        )

    def _slot(self) -> Optional[rs.EntitySlot]:
        key = self._entity.currentData() if not self._session_mode else "ses"
        for slot in self._slots:
            if slot.key == key:
                return slot
        return self._slots[0] if self._slots else None

    def _on_entity_changed(self) -> None:
        slot = self._slot()
        if slot is None:
            return
        if self._is_removing():
            carried = ", ".join(f"{slot.key}-{v}" for v in slot.values[:6])
            self._what.setText(
                f"<b>{slot.present}</b> of {slot.total} selected file(s) "
                f"carry <b>{slot.key}-</b>"
                + (f" ({carried})." if carried else ".")
            )
            self._refresh_plan()
            return

        self._value.blockSignals(True)
        current = self._value.currentText()
        self._value.clear()
        self._value.addItems(list(slot.values))
        self._value.setCurrentText(current)
        self._value.blockSignals(False)
        self._value.lineEdit().setPlaceholderText(
            "digits only, for example 01" if slot.kind == "index"
            else "letters, digits and + only"
        )
        self._value.setToolTip(
            f"{slot.display} ({slot.kind}): {slot.pattern}"
        )
        already = (
            f" <b>{slot.present}</b> of {slot.total} already carry it and "
            "would be changed."
            if slot.present else ""
        )
        self._what.setText(
            f"{slot.display} is a <b>{slot.kind}</b> "
            f"({slot.pattern}).{already}"
        )
        self._refresh_plan()

    # -- planning --------------------------------------------------------

    def _on_typed(self, _text: str = "") -> None:
        self._plan = None
        self._ok.setEnabled(False)
        self._status.setText("working...")
        self._replan.start()

    def plan_now(self) -> None:
        self._replan.stop()
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        self._plan = None
        self._ok.setEnabled(False)
        slot = self._slot()
        if slot is None:
            return
        removing = self._is_removing()
        value = None if removing else self._value.currentText().strip()
        if not removing and not value:
            self._preview.clear()
            self._summary.setText("Type the value to give it.")
            self._status.setText("")
            return
        try:
            plan = rs.plan_entity_edit(
                self._root, slot.key, self._targets, value=value,
            )
        except rn.RenameError as exc:
            self._summary.setText(str(exc))
            self._preview.clear()
            self._status.setText("")
            return

        if self._session_mode:
            plan.title = (
                "Remove the session" if removing
                else f"Create session ses-{value}"
            )
        self._plan = plan
        self._preview.show_moves(
            self._root,
            [
                (plan.file_key(self._root, src), src, dst)
                for src, dst in plan.file_moves
            ],
            extras=plan_extras(plan),
            conflicts=plan.conflicts,
        )

        if plan.conflicts:
            self._summary.setText(
                "Some files would end up with a name that is already taken, "
                "so this cannot be applied as asked."
            )
            self._status.setText(f"{len(plan.conflicts)} conflict(s)")
        elif plan.is_empty:
            self._summary.setText(
                "Nothing to do: the selected files already read that way."
            )
            self._status.setText("")
        else:
            self._summary.setText(f"Would change {plan.summary()}.")
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
        total = len(self._plan.file_moves)
        self._ok.setEnabled(bool(chosen) and not self._plan.conflicts)
        if total and len(chosen) < total:
            self._status.setText(f"{len(chosen)} of {total} file(s) selected")
            self._ok.setText("Apply to selected")
        else:
            self._status.setText(self._plan.summary())
            self._ok.setText("Apply")

    # -- doing it --------------------------------------------------------

    def _on_apply(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        if self._plan is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Folders move here too, and on Windows a folder being watched
            # cannot be moved: QFileSystemWatcher holds an open handle on it
            # and MoveFile returns ERROR_ACCESS_DENIED. See gui.fs_watch.
            with watchers_released():
                touched, errors = rn.apply_rename(
                    self._root, self._plan, only=self.selected_keys(),
                )
        except rn.RenameError as exc:
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


__all__ = ["ADD", "REMOVE", "EditEntitiesDialog"]
