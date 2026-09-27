"""Change one entity value to another, across a chosen part of the dataset.

The three ways to change an entity that existed before this all answered a
different question:

* **Rename entity** changes a value everywhere in the dataset. No choice of
  scope, so "fix this label in sub-014 only" was not expressible.
* **Add or remove an entity** sets a value on the files you selected in the
  tree. You have to find and select them first, which on a hundred files
  means finding a hundred files.
* **Bulk edit** in the Converter works on inventory rows, before conversion.

What was missing is the one people actually ask for: FIND every file where
this entity has this value, inside this subject or session or the whole
dataset, and change it. That is this dialog.

The values are listed with how many files carry each, so picking one is not
guesswork; the scope is a dropdown of the dataset, each subject and each
session; and the result is the ordinary rename preview, so the same
per-file ticking, the same scans-row relocation and the same reference
rewriting apply.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from ..editor import rename as rn, values as ev
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.move_preview import MovePreviewTree, plan_extras
from .widgets.preview_split import (
    PreviewSplit,
    controls_panel,
    preview_toggle,
)

log = logging.getLogger(__name__)


class ReplaceValueDialog(QDialog):
    """Find one entity value and replace it, inside a chosen scope."""

    def __init__(
        self,
        root: Path,
        *,
        entity: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._scopes = ev.scopes(self._root)
        self._plan: Optional[rn.RenamePlan] = None
        self._keys: set[str] = set()
        self._applied = 0

        self.setWindowTitle("Find and replace an entity value")
        self.setModal(True)
        self.resize(900, 700)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Find an entity value and replace it",
            "Every file where the entity has the value you pick, inside the "
            "part of the dataset you pick, gets the new value. The scans "
            "rows, the references and the entity columns follow, and you "
            "see the whole plan before anything moves.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        # -- what to find --------------------------------------------------
        find, findl = card("Find")
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("In:"))
        self._scope = QComboBox()
        self._scope.setObjectName("ent-input")
        for scope in self._scopes:
            self._scope.addItem(scope.label, userData=scope)
        self._scope.currentIndexChanged.connect(lambda _i: self._reload_values())
        row.addWidget(self._scope, 2)

        row.addSpacing(10)
        row.addWidget(QLabel("Entity:"))
        self._entity = QComboBox()
        self._entity.setObjectName("ent-input")
        self._entity.currentIndexChanged.connect(lambda _i: self._reload_values())
        row.addWidget(self._entity, 1)

        row.addSpacing(10)
        row.addWidget(QLabel("Value:"))
        self._old = QComboBox()
        self._old.setObjectName("ent-input")
        self._old.currentIndexChanged.connect(lambda _i: self._replan())
        row.addWidget(self._old, 1)
        findl.addLayout(row)

        replace = QHBoxLayout()
        replace.setSpacing(8)
        replace.addWidget(QLabel("Replace with:"))
        self._new = QLineEdit()
        self._new.setObjectName("tb-input")
        self._new.setPlaceholderText("the new value, without the entity prefix")
        self._new.textChanged.connect(lambda _t: self._replan())
        replace.addWidget(self._new, 1)
        findl.addLayout(replace)

        self._rule = hint("")
        findl.addWidget(self._rule)
        # -- what would happen --------------------------------------------
        preview, pl = card("What would change")
        pl.addWidget(hint(
            "Untick anything you would rather leave alone. What follows "
            "automatically is listed underneath and cannot be unticked: "
            "leaving a scans row or a reference behind is the damage this "
            "is preventing."
        ))
        self._preview = MovePreviewTree()
        self._preview.itemChanged.connect(lambda *_a: self._update_status())
        pl.addWidget(self._preview, 1)
        # Controls and preview in a splitter the user can flip between
        # stacked and side by side. See preview_split.py.
        self._split = PreviewSplit(
            controls_panel(find), preview, name="replace_value",
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
        self._ok.setText("Replace")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        buttons.addButton(
            preview_toggle(self._split), QDialogButtonBox.ButtonRole.ResetRole
        )
        outer.addWidget(build_footer_with(self._status, buttons))

        self._fill_entities(entity)

    # -- filling ----------------------------------------------------------

    def _fill_entities(self, preferred: Optional[str]) -> None:
        """Only entities this dataset actually uses somewhere.

        Offering all of them would be a list of thirty, most of which have
        no values to find.
        """
        self._entity.blockSignals(True)
        self._entity.clear()
        for key in schema_mod.entity_keys():
            if ev.counts(self._root, key):
                self._entity.addItem(key, userData=key)
        if preferred:
            index = self._entity.findData(preferred)
            if index >= 0:
                self._entity.setCurrentIndex(index)
        self._entity.blockSignals(False)
        self._reload_values()

    def _current_scope(self) -> ev.Scope:
        return self._scope.currentData() or self._scopes[0]

    def _reload_values(self) -> None:
        entity = self._entity.currentData()
        self._old.blockSignals(True)
        self._old.clear()
        if entity:
            for value, n in ev.counts_in(self._root, entity, self._current_scope()):
                self._old.addItem(f"{value}  ({n} file(s))", userData=value)
        self._old.blockSignals(False)

        if entity:
            try:
                info = schema_mod.entity_key_info(entity)
                self._rule.setText(
                    f"{info.description.strip()} The new value must be a "
                    f"valid {info.format.name}."
                )
            except KeyError:
                self._rule.setText("")
        self._replan()

    # -- planning ---------------------------------------------------------

    def _replan(self) -> None:
        entity = self._entity.currentData()
        old = self._old.currentData()
        new = self._new.text().strip()
        self._plan, self._keys = None, set()
        self._preview.clear()

        if not (entity and old):
            self._status.setText(
                "This scope has no values for that entity."
                if entity else "This dataset uses no entities."
            )
            self._ok.setEnabled(False)
            return
        if not new:
            self._status.setText(f"Replacing {entity}-{old} with what?")
            self._ok.setEnabled(False)
            return
        if new == old:
            self._status.setText("That is the value it already has.")
            self._ok.setEnabled(False)
            return

        try:
            self._plan, self._keys = ev.plan_replace(
                self._root, entity, old, new, self._current_scope()
            )
        except rn.RenameError as exc:
            self._status.setText(str(exc))
            self._ok.setEnabled(False)
            return

        moves = [
            (key, src, dst)
            for key, (src, dst) in zip(
                self._plan.file_keys(self._root), self._plan.file_moves
            )
            if key in self._keys
        ]
        self._preview.show_moves(
            self._root, moves,
            extras=plan_extras(self._plan),
            conflicts=self._plan.conflicts,
            checked=set(self._keys),
        )
        self._update_status()

    def _update_status(self) -> None:
        if self._plan is None:
            return
        if self._plan.conflicts:
            self._status.setText(self._plan.conflicts[0])
            self._ok.setEnabled(False)
            return
        chosen = self._preview.selected_keys() & self._keys
        scope = self._current_scope()
        where = "the whole dataset" if not scope.prefix else scope.label
        self._status.setText(
            f"{len(chosen)} of {len(self._keys)} file(s) in {where}. "
            "One undoable step."
        )
        self._ok.setEnabled(bool(chosen))

    # -- applying ---------------------------------------------------------

    def _on_apply(self) -> None:
        if self._plan is None:
            return
        chosen = self._preview.selected_keys() & self._keys
        if not chosen:
            return
        try:
            with watchers_released():
                touched, errors = rn.apply_rename(
                    self._root, self._plan, only=chosen
                )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            QMessageBox.warning(
                self, "Replace failed", f"Nothing was changed.\n\n{exc}"
            )
            return
        if errors:
            QMessageBox.warning(
                self, "Partly done",
                f"{touched} file(s) changed. {len(errors)} did not:\n\n"
                + "\n".join(errors[:5]),
            )
        self._applied = touched
        self.accept()

    def applied_count(self) -> int:
        return self._applied


__all__ = ["ReplaceValueDialog"]
