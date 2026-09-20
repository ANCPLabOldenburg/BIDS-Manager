"""Rename a BIDS entity, after showing exactly what that would do.

Renaming a subject touches a folder, every file under it, and three kinds of
cross-reference. That is far too much to do on trust, so the dialog is built
around the dry run: pick an entity and a value, see the plan, then commit it.
The button stays disabled until a plan exists.

**A name that is already taken is offered as a merge, not reported as an
error.** Renaming ``sub-12`` to ``sub-07`` when ``sub-07`` exists almost always
means the two are one person, scanned twice or converted in two passes. The
dialog notices, changes the button from Rename to Merge, and says in words what
will happen to the sessions, the scans tables and the participants row. It is a
separate confirmation because it is a different act: a merge cannot be undone by
renaming back, it is undone from the Editor's history.

Opened either from the Editor toolbar with nothing selected, or from a
right-click in the file tree, in which case the entity and value the user
clicked are already filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import rename as rn
from ..schema import entity_key_info
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.move_preview import KEY_ROLE, MovePreviewTree, plan_extras

def entity_choices() -> list[tuple[str, str]]:
    """Every entity the ACTIVE schema defines, as ``(key, display name)``.

    Read from the schema rather than listed here, for the reason every other
    layer reads from it: the set of entities is a fact about the BIDS version
    in force, and a hand-kept list silently omits whatever was added since
    somebody last edited it. The schema's own filename order is kept, which is
    also the order a user reads a name in, so Subject comes first.
    """
    from ..schema import entity_key_info, entity_keys

    out: list[tuple[str, str]] = []
    for key in entity_keys():
        try:
            info = entity_key_info(key)
        except KeyError:
            continue
        out.append((key, info.display_name))
    return out

# Where a row keeps the plan key it stands for, so the dialog never has to
# match a file back to the plan by its display text. It lives with the tree
# now, since three dialogs share that tree; re-exported here because it was
# part of this module's surface first.
_KEY_ROLE = KEY_ROLE

# How long to wait after the last keystroke before re-planning. Planning walks
# the whole dataset, so doing it per keystroke froze the window on anything
# real. Short enough to feel immediate, long enough that typing a three-digit
# label plans once.
_REPLAN_DELAY_MS = 300


class RenameEntityDialog(QDialog):
    """Pick an entity and a new label; preview; then apply."""

    def __init__(
        self,
        root: Path,
        parent: Optional[QWidget] = None,
        *,
        entity: str = "",
        value: str = "",
        focus: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._plan: Optional[rn.RenamePlan] = None
        # What the user right-clicked, if anything. It narrows what starts
        # TICKED, never what the plan contains: opening on one recording and
        # offering every file in the dataset that shares its subject is a
        # dialog answering a question nobody asked.
        self._focus = Path(focus) if focus else None
        # The user's own selection, once they have expressed one. Kept so a
        # re-plan does not discard it: ticking "merge them" re-ran the plan
        # and silently re-selected everything, undoing whatever had just been
        # chosen.
        self._chosen: Optional[set[str]] = None
        # Which (entity, old, new) the selection belongs to. A different
        # operation deserves a fresh default; the same one being re-planned
        # does not.
        self._chosen_for: tuple[str, str, str] = ("", "", "")
        self.setWindowTitle("Rename an entity")
        self.setModal(True)
        self.resize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_header(
            "Rename across the dataset",
            "Entities are renamed as <b>values</b>, not as text, so "
            "<b>run-1</b> never touches <b>run-10</b>. References inside "
            "<b>IntendedFor</b>, <b>*_scans.tsv</b> and "
            "<b>participants.tsv</b> travel with the rename, and the whole "
            "thing is one step in the Editor's history.",
        ))

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

        self._entity = QComboBox()
        self._entity.setObjectName("ent-input")
        for key, label in entity_choices():
            self._entity.addItem(f"{label}  ({key}-)", key)
            try:
                info = entity_key_info(key)
            except KeyError:
                continue
            self._entity.setItemData(
                self._entity.count() - 1,
                f"{info.description}\n\nValue format: {info.format.name} "
                f"({info.format.pattern})",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._entity.currentIndexChanged.connect(self._reload_values)
        form.addRow("Entity:", self._entity)

        self._old = QComboBox()
        self._old.setObjectName("ent-input")
        self._old.currentIndexChanged.connect(self._on_typed)
        form.addRow("Rename:", self._old)

        self._new = QLineEdit()
        self._new.setObjectName("ent-input")
        self._new.setPlaceholderText("")
        # Debounced. Planning walks the dataset, and on a real one that is
        # hundreds of milliseconds; doing it per keystroke is what froze the
        # window. The pause is short enough to feel immediate and long enough
        # that typing "002" plans once instead of three times.
        self._replan = QTimer(self)
        self._replan.setSingleShot(True)
        self._replan.setInterval(_REPLAN_DELAY_MS)
        self._replan.timeout.connect(self._refresh_plan)
        self._new.textChanged.connect(self._on_typed)
        # Enter means "I have finished typing", so there is nothing to wait for.
        self._new.returnPressed.connect(self.plan_now)
        form.addRow("To:", self._new)
        cl.addLayout(form)

        # Only ever shown when the target name is taken. A checkbox that is
        # always there invites somebody to tick it before they need it.
        self._fuse = QCheckBox("Merge them into one subject")
        self._fuse.setToolTip(
            "Move everything under the source into the existing one, "
            "concatenate their scans tables and fold their two "
            "participants.tsv rows into a single row that keeps every value "
            "either of them states."
        )
        self._fuse.toggled.connect(self._refresh_plan)  # cheap: no re-walk
        self._fuse.setVisible(False)
        cl.addWidget(self._fuse)
        chooser.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )
        bl.addWidget(chooser)

        self._summary = hint("")
        bl.addWidget(self._summary)

        preview_card, pl = card("What this would do")
        pl.addWidget(hint(
            "Untick anything you want left alone. References inside "
            "<b>IntendedFor</b> and <b>*_scans.tsv</b> follow the files that "
            "actually move, so what you leave behind keeps its own name and "
            "nothing ends up pointing at a name that does not exist."
        ))
        self._preview = MovePreviewTree()
        self._preview.itemChanged.connect(self._on_item_checked)
        pl.addWidget(self._preview, 1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        for label, state in (
            ("Select all", Qt.CheckState.Checked),
            ("Select none", Qt.CheckState.Unchecked),
        ):
            btn = QPushButton(label)
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
        self._ok.setText("Rename")
        self._ok.setEnabled(False)
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._reload_values()
        if entity:
            self.preset(entity, value)

    # -- opening on something the user clicked ---------------------------

    def preset(self, entity: str, value: str = "") -> None:
        """Start on the entity and value the user right-clicked in the tree."""
        index = self._entity.findData(entity)
        if index >= 0:
            self._entity.setCurrentIndex(index)
            self._reload_values()
        if value:
            at = self._old.findText(value)
            if at >= 0:
                self._old.setCurrentIndex(at)
        self._new.setFocus()

    # -- state ---------------------------------------------------------

    def _on_typed(self, _text: str = "") -> None:
        """A keystroke. Say the plan is stale, and schedule one.

        The button goes dead immediately rather than staying live against a
        plan for the previous text, which is the way a debounce can be worse
        than none: it would let somebody apply the rename they just finished
        typing over.
        """
        self._plan = None
        self._ok.setEnabled(False)
        self._status.setText("working...")
        self._replan.start()

    def plan_now(self) -> None:
        """Plan immediately instead of waiting out the debounce.

        For the moments where a user has clearly finished: pressing Enter, or
        the dialog being asked for its plan by something else.
        """
        self._replan.stop()
        self._refresh_plan()

    def _entity_key(self) -> str:
        return self._entity.currentData() or "sub"

    def _describe_format(self) -> None:
        """Say what this entity's value may be, in the schema's own terms.

        ``run`` is an index and ``acq`` is a label, and the difference is not
        cosmetic: one accepts digits only, the other accepts ``+``. A single
        placeholder would be wrong for one of them.
        """
        try:
            info = entity_key_info(self._entity_key())
        except KeyError:
            self._new.setPlaceholderText("new value")
            return
        if info.format.name == "index":
            self._new.setPlaceholderText("digits only, for example 01")
        else:
            self._new.setPlaceholderText("letters, digits and + only")
        self._new.setToolTip(
            f"{info.display_name} ({info.format.name}): {info.format.pattern}"
        )

    def _reload_values(self) -> None:
        self._describe_format()
        self._old.blockSignals(True)
        self._old.clear()
        values = rn.list_values(self._root, self._entity_key())
        self._old.addItems(values)
        self._old.setEnabled(bool(values))
        self._old.blockSignals(False)
        if not values:
            self._summary.setText(
                f"No <b>{self._entity_key()}-</b> entity is used in this "
                "dataset."
            )
            self._preview.clear()
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        self._plan = None
        self._ok.setEnabled(False)
        old = self._old.currentText().strip()
        new = self._new.text().strip()

        taken = bool(new) and rn.would_fuse(self._root, self._entity_key(), new)
        self._fuse.setVisible(taken)
        if not taken and self._fuse.isChecked():
            self._fuse.blockSignals(True)
            self._fuse.setChecked(False)
            self._fuse.blockSignals(False)

        if not old or not new:
            self._preview.clear()
            self._status.setText("")
            self._ok.setText("Rename")
            if old:
                self._summary.setText("Type the new label.")
            return
        try:
            plan = rn.plan_rename(
                self._root, self._entity_key(), old, new,
                fuse=self._fuse.isChecked(),
            )
        except rn.RenameError as exc:
            self._summary.setText(str(exc))
            self._preview.clear()
            self._status.setText("")
            return

        self._plan = plan
        self._populate(plan)
        self._ok.setText("Merge" if plan.fusion else "Rename")

        if plan.conflicts:
            self._summary.setText(
                "Two files would end up with the same name, so this cannot be "
                "applied as asked."
                if plan.fusion else
                "That name is already taken. Tick <b>Merge them into one "
                "subject</b> to combine them, or pick a different label."
            )
            self._status.setText(f"{len(plan.conflicts)} conflict(s)")
        elif plan.is_empty:
            self._summary.setText("Nothing uses that value.")
            self._status.setText("")
        else:
            self._summary.setText(
                f"Would merge <b>{plan.entity}-{plan.old}</b> into "
                f"<b>{plan.entity}-{plan.new}</b>. This cannot be undone by "
                "renaming back; use the Editor's history."
                if plan.fusion else
                f"Would change {plan.summary()}."
            )
            self._refresh_selection()

    # -- choosing what to rename -----------------------------------------

    def _populate(self, plan: rn.RenamePlan) -> None:
        """Draw the plan as a dataset tree the user can untick items in."""
        self._preview.show_moves(
            self._root,
            [
                (plan.file_key(self._root, src), src, dst)
                for src, dst in plan.file_moves
            ],
            extras=plan_extras(plan),
            conflicts=plan.conflicts,
            checked=self._initial_selection(plan),
        )

    def _initial_selection(self, plan: rn.RenamePlan) -> Optional[set[str]]:
        """Which rows start ticked. ``None`` means all of them.

        Three cases, in order:

        * the user has already chosen, and this is the SAME rename being
          re-planned (they toggled merge, or retyped the same value). Their
          choice is kept, intersected with what the new plan still offers;
        * the dialog was opened on a file the user right-clicked. That file
          and its companions start ticked and nothing else does, so "rename
          the subject of this recording" does not silently offer every
          recording that subject has;
        * nothing was clicked, so everything.
        """
        keys = set(plan.file_keys(self._root))
        signature = (plan.entity, plan.old, plan.new)
        if self._chosen is not None and signature == self._chosen_for:
            return self._chosen & keys
        self._chosen = None
        self._chosen_for = signature
        if self._focus is None:
            return None
        narrowed = self._focus_keys() & keys
        return narrowed or None

    def _focus_keys(self) -> set[str]:
        """The clicked path expressed as plan keys.

        A folder contributes everything under it. A file contributes itself
        and its companions, which is the ``.json`` sidecar and any
        ``_events.tsv`` carrying the same entities: renaming a recording and
        leaving its sidecar behind is not something anybody wants offered as
        the default.
        """
        from ..editor.restructure import companions

        focus = self._focus
        if focus is None:
            return set()
        try:
            if focus.is_dir():
                paths = [p for p in rn.walk_dataset(self._root)
                         if _inside(focus, p)]
            else:
                paths = companions(self._root, focus)
        except OSError:
            paths = [focus]
        out = set()
        for path in paths:
            try:
                out.add(
                    Path(path).resolve()
                    .relative_to(self._root.resolve()).as_posix()
                )
            except (ValueError, OSError):
                continue
        return out

    def _set_all(self, state: Qt.CheckState) -> None:
        self._preview.set_all(state)
        self._chosen = self._preview.selected_keys()
        self._refresh_selection()

    def _on_item_checked(self, item: QTreeWidgetItem, column: int) -> None:
        del item, column
        # Remember it, so the next re-plan does not throw it away.
        self._chosen = self._preview.selected_keys()
        self._refresh_selection()

    def selected_keys(self) -> set[str]:
        """The file moves currently ticked, by their path within the root."""
        return self._preview.selected_keys()

    def _refresh_selection(self) -> None:
        """Keep the button and the footer honest about what is ticked."""
        if self._plan is None:
            return
        chosen = self.selected_keys()
        total = len(self._plan.file_moves)
        partial = bool(total) and len(chosen) < total
        self._ok.setEnabled(bool(chosen) and not self._plan.conflicts)
        if partial:
            self._status.setText(
                f"{len(chosen)} of {total} file(s) selected"
            )
            # A merge is about the subject as a whole. Choosing some of its
            # files is a move, not a merge, and the button has to say so.
            self._ok.setText("Rename selected")
        else:
            self._status.setText(self._plan.summary())
            self._ok.setText("Merge" if self._plan.fusion else "Rename")

    # -- doing it ------------------------------------------------------

    def _is_partial(self) -> bool:
        if self._plan is None:
            return False
        return len(self.selected_keys()) < len(self._plan.file_moves)

    def _on_apply(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        if self._plan is None:
            return
        if self._plan.fusion and not self._is_partial():
            confirm = QMessageBox.question(
                self, "Merge two subjects",
                f"{self._plan.verb()}?\n\n"
                "Everything under the first moves into the second, the scans "
                "tables are combined and the two participants rows become "
                "one. It is a single step in the Editor's history, so it can "
                "be undone from there.",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Ok:
                return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # A rename moves folders, and on Windows a folder being watched
            # cannot be moved: QFileSystemWatcher holds an open handle on it
            # and MoveFile returns ERROR_ACCESS_DENIED. The rename then fell
            # back to moving the files one by one, which works, and left the
            # emptied original folder on disk. See bidsmgr.gui.fs_watch.
            with watchers_released():
                touched, errors = rn.apply_rename(
                    self._root, self._plan, only=self.selected_keys(),
                )
        except rn.RenameError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Rename refused", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(
                self, "Some steps failed",
                f"{touched} item(s) changed, but:\n\n" + "\n".join(errors[:8]),
            )
        self.accept()


def _inside(folder: Path, path: Path) -> bool:
    try:
        Path(path).relative_to(folder)
    except ValueError:
        return False
    return True


__all__ = ["RenameEntityDialog"]
