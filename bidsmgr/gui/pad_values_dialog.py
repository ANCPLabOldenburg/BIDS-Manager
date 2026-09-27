"""Give the index entities a consistent width, one width per entity.

``run-1`` and ``run-01`` are both valid: the standard's ``index`` format
accepts either, so a dataset using one is not wrong. This exists because
consistency within a dataset is worth having, because doing it by hand is a
rename per value, and because a dataset holding BOTH spellings sorts wrongly
and makes a glob miss half its files.

Every index entity the dataset uses is listed at once, each with its own
width, because ``run`` and ``echo`` are different questions and a dialog
that asks them one at a time makes the user open it twice.

Two things it says that the first version did not:

* **Where the widths already disagree.** That is the case worth acting on,
  and it was silent about it.
* **When padding CANNOT settle it**, because the dataset holds both ``run-3``
  and ``run-03`` and padding would fuse two different runs into one.

Only entities the schema calls an INDEX are offered. ``acq-01`` and ``acq-1``
are two different labels, and turning one into the other is a change of
meaning that belongs in the replace dialog where it is visible as one.
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
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from ..editor import rename as rn, values as ev
from .dialog_chrome import build_footer_with, build_header, card, hint
from .fs_watch import watchers_released
from .widgets.preview_split import (
    PreviewSplit,
    controls_panel,
    preview_toggle,
)

log = logging.getLogger(__name__)


class PadValuesDialog(QDialog):
    """Repad every index entity in the dataset, each to its own width."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._widths: dict[str, QSpinBox] = {}
        self._applied = 0

        self.setWindowTitle("Index widths")
        self.setModal(True)
        self.resize(880, 680)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Give every index the same width",
            "Turns <b>run-1</b> into <b>run-01</b>, and <b>run-001</b> back "
            "into <b>run-01</b>, for each index entity separately. Both "
            "spellings are valid BIDS, so this is a house style rather than "
            "a correction. Every reference follows: the scans tables, the "
            "links and the entity columns.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        self._warning = QLabel("")
        self._warning.setObjectName("dlg-hint")
        self._warning.setWordWrap(True)

        chooser, cl = card("Width per entity")
        cl.addWidget(hint(
            "One row per index entity this dataset uses. The width starts at "
            "the widest value already in use, so pressing Apply with nothing "
            "changed settles a dataset that disagrees with itself and leaves "
            "a consistent one alone. Each row says the narrowest width its "
            "values would still fit in, so trimming is a choice you can see "
            "rather than one you have to guess at."
        ))
        self._rows = QWidget()
        self._rows_layout = QVBoxLayout(self._rows)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        cl.addWidget(self._rows)
        preview, pl = card("What would change")
        self._tree = QTreeWidget()
        self._tree.setObjectName("check-tree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Now", "Becomes", "Files"])
        self._tree.setUniformRowHeights(True)
        pl.addWidget(self._tree, 1)
        # Controls and preview in a splitter the user can flip between
        # stacked and side by side. See preview_split.py.
        self._split = PreviewSplit(
            controls_panel(self._warning, chooser), preview,
            name="index_widths",
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
        self._ok.setText("Apply")
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        buttons.addButton(
            preview_toggle(self._split), QDialogButtonBox.ButtonRole.ResetRole
        )
        outer.addWidget(build_footer_with(self._status, buttons))

        self._build_rows()
        self._replan()

    # -- the per-entity rows ----------------------------------------------

    def _index_entities(self) -> list[str]:
        """Every index entity the schema defines, in BIDS order.

        All of them, not only those in use. The first version listed only
        what the dataset already had, so a dataset with one ``run`` showed a
        single row and read as though ``run`` were the only index there is.
        The same list backs the Settings tab, so the two agree about what
        exists; an entity with no values here is shown and disabled rather
        than hidden, which is the difference between "you have none of
        these" and "these do not exist".
        """
        return ev.index_entities()

    def _build_rows(self) -> None:
        splits = {s.entity: s for s in ev.inconsistent_widths(self._root)}

        for entity in self._index_entities():
            values = [v for v, _ in ev.counts(self._root, entity) if v.isdigit()]
            widest = max((len(v) for v in values), default=1)
            # What the values actually need, ignoring the zeros somebody put
            # in front of them. This is what makes trimming discoverable:
            # a dataset written run-001 to run-012 says so, instead of
            # looking settled at three digits.
            needed = max(
                (len(v.lstrip("0") or "0") for v in values), default=1
            )
            used = bool(values)

            row = QHBoxLayout()
            row.setSpacing(8)
            try:
                display = schema_mod.entity_key_info(entity).display_name
            except KeyError:
                display = entity
            label = QLabel(f"{entity}-")
            label.setMinimumWidth(70)
            label.setToolTip(display)
            row.addWidget(label)

            spin = QSpinBox()
            spin.setObjectName("ent-input")
            spin.setRange(1, 6)
            spin.setValue(widest)
            spin.setSuffix(" digits")
            spin.setEnabled(used)
            spin.valueChanged.connect(lambda _v: self._replan())
            self._widths[entity] = spin
            row.addWidget(spin)

            split = splits.get(entity)
            if split:
                text = (
                    f"written at {len(split.widths)} widths "
                    f"({', '.join(split.examples)}) across {split.files} "
                    "file(s), and needs settling"
                )
            elif used:
                text = (
                    f"{len(values)} value(s), all {widest} digit(s): "
                    + ", ".join(f"{entity}-{v}" for v in values[:3])
                    + (", ..." if len(values) > 3 else "")
                )
                if needed < widest:
                    text += f" ({needed} would fit)"
            else:
                text = "not used in this dataset"
            note = QLabel(text)
            note.setObjectName("dlg-hint")
            row.addWidget(note, 1)

            holder = QWidget()
            holder.setLayout(row)
            self._rows_layout.addWidget(holder)

        if splits:
            self._warning.setText(
                f"<b>{len(splits)} entity(ies) disagree with themselves.</b> "
                + "; ".join(
                    f"{s.entity} is written as {' and '.join(s.examples)}"
                    for s in splits.values()
                )
                + ". Applying the widths below settles them."
            )
        else:
            self._warning.setText(
                "Every index entity in this dataset already uses one width."
            )

    # -- planning ---------------------------------------------------------

    def _replan(self) -> None:
        self._tree.clear()
        total = 0
        refused: list[str] = []

        for entity, spin in self._widths.items():
            try:
                plan = ev.plan_padding(self._root, entity, spin.value())
            except rn.RenameError as exc:
                refused.append(f"{entity}: {exc}")
                continue
            if not plan:
                continue
            head = QTreeWidgetItem(self._tree, [entity, "", ""])
            head.setFirstColumnSpanned(True)
            head.setExpanded(True)
            for repad in plan:
                QTreeWidgetItem(head, [
                    f"{entity}-{repad.old}", f"{entity}-{repad.new}",
                    str(repad.files),
                ])
                total += repad.files

        self._tree.resizeColumnToContents(0)
        if refused:
            self._status.setText(
                "Cannot pad: " + " | ".join(refused)
                + " Use Find and replace to settle the clash first."
            )
            self._ok.setEnabled(False)
            return
        if not total:
            self._status.setText("Nothing to change at these widths.")
            self._ok.setEnabled(False)
            return
        self._status.setText(
            f"{total} file(s) would be renamed. Each entity is its own "
            "undoable step."
        )
        self._ok.setEnabled(True)

    # -- applying ---------------------------------------------------------

    def _on_apply(self) -> None:
        jobs = []
        for entity, spin in self._widths.items():
            try:
                plan = ev.plan_padding(self._root, entity, spin.value())
            except rn.RenameError:
                continue
            jobs += [(entity, repad) for repad in plan]
        if not jobs:
            return

        answer = QMessageBox.question(
            self, "Index widths",
            f"Rewrite {len(jobs)} value(s) across the dataset?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return

        done = 0
        try:
            # Folders are renamed when the entity names one, and on Windows
            # a watched directory cannot be renamed.
            with watchers_released():
                for entity, repad in jobs:
                    plan = rn.plan_rename(
                        self._root, entity, repad.old, repad.new
                    )
                    if plan.conflicts:
                        raise rn.RenameError(
                            f"{entity}-{repad.old} cannot become "
                            f"{entity}-{repad.new}: {plan.conflicts[0]}"
                        )
                    rn.apply_rename(self._root, plan)
                    done += 1
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            QMessageBox.warning(
                self, "Padding stopped",
                f"{done} of {len(jobs)} value(s) were rewritten before this "
                f"stopped it. Each is its own undo step.\n\n{exc}",
            )
            self._applied = done
            self._replan()
            return

        self._applied = done
        self.accept()

    def applied_count(self) -> int:
        return self._applied


__all__ = ["PadValuesDialog"]
