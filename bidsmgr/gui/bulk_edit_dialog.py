"""Write one value into one column, across the rows you choose.

Reached from the **Bulk edit...** toolbar button (enabled when two or more
rows are selected in the inspection table). Dispatches through
:meth:`InventoryTableModel.bulk_set` so entity rebuilds, mirror cells and
the BIDS-name column all stay in sync.

For columns with a schema-bounded set of values (``datatype``, ``suffix``)
the value box is a combo populated from the schema; everything else is free
text bounded by the entity's own format.

**The selection is where it starts, not what it does.** It used to write
into every selected row, full stop, so "change task-rest to task-restingstate
but leave the two localizers alone" meant going back to the table and
re-selecting. The Editor's rename tool had the better model: find the rows
that say a particular thing, and change those. This has it too, in two
layers that answer different questions:

* **Which rows** the change applies to: all of the selection, or only the
  ones whose current value is a particular one, listed with how many rows
  carry each, so there is nothing to guess.
* **Then** a preview, one row per file, saying what it says now and what it
  would say, every one of them untickable. That is the layer that handles
  the case no filter can express: "these four, but not that one".

Nothing is written until Apply, and the count on the button is the number of
rows that would actually change.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from .delegates import builtin_montages
from .models import COLUMNS, InventoryTableModel

#: Sentinel for the "every selected row" entry of the target combo.
_ALL_ROWS = "\x00all"

#: Sentinel for "the rows where this column is empty". Distinct from a real
#: value of "" so the combo can offer it as its own line with a count.
_BLANK = "\x00blank"

#: Replanning walks the selection, so it is debounced exactly as the rename
#: dialog debounces its own: doing it per keystroke froze the window.
_REPLAN_DELAY_MS = 200


# Human-readable header per column key. Keeps the dropdown clear about
# what each option actually does.
_COLUMN_DESCRIPTION: dict[str, str] = {
    "id":        "Subject identifier — updates participant_id AND the subject entity.",
    "dataset":   "Dataset slug — the convert verb groups rows by this column.",
    "ses":       "Session label (no ``ses-`` prefix; the converter adds it).",
    "task":      "Task entity.",
    "run":       "Run number.",
    "datatype":  "BIDS datatype (anat, func, dwi, …). Triggers a basename rebuild.",
    "suffix":    "BIDS suffix (T1w, bold, …). Triggers a basename rebuild.",
    "line_freq": "EEG/MEG power-line frequency (Hz). Choose 50 or 60.",
    "montage":   "EEG/MEG montage. Choose from the MNE built-in montages.",
    "eeg_reference": "EEG/iEEG reference electrode (e.g. Cz, average).",
    "eeg_ground":    "EEG/iEEG ground electrode.",
    "PatientSex":    "Participant sex. Choose M / F / O.",
    "PatientAge":    "Participant age in years.",
    "Handedness":    "Participant handedness. Choose R / L / A.",
}

# Columns whose value must come from a fixed, non-editable dropdown (never
# free-typed). datatype / suffix stay editable combos (schema-bounded but
# large + interdependent); these are short, closed sets.
_FIXED_CHOICES: dict[str, list[str]] = {
    "line_freq": ["50", "60"],
    "PatientSex": ["M", "F", "O"],
    "Handedness": ["R", "L", "A"],
}


def _first_paragraph(text: str) -> str:
    """The first paragraph of a schema description, on one line.

    Schema descriptions are Markdown written for the specification: several
    paragraphs, hard-wrapped mid-sentence, with backticks around names. In a
    dialog that is a wall of text with stray line breaks, so show the first
    paragraph with its line breaks and backticks removed, and leave the rest
    to the tooltip.
    """
    first = text.strip().split("\n\n", 1)[0]
    return " ".join(first.replace("`", "").split())


class BulkEditDialog(QDialog):
    """One-shot apply-value-to-column dialog.

    Pass the live model + the list of source DataFrame row indices the
    user has selected. On Apply, calls ``model.bulk_set(rows, key, value)``
    and reports how many rows changed back via the return value of
    :meth:`changed_count` (after ``exec()``).
    """

    def __init__(
        self,
        model: InventoryTableModel,
        rows: Iterable[int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bulk edit")
        self.setModal(True)
        self.resize(760, 620)
        self._model = model
        self._rows: list[int] = list(rows)
        self._changed: int = 0
        self._skipped: int = 0
        # Created first, because widgets wired below fire into them while
        # the dialog is still being built.
        self._loading = False
        self._replan_timer = QTimer(self)
        self._replan_timer.setSingleShot(True)
        self._replan_timer.setInterval(_REPLAN_DELAY_MS)
        self._replan_timer.timeout.connect(self._replan)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---------- header ----------
        header = QFrame()
        header.setObjectName("issue-dialog-header")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(18, 14, 18, 14)
        hl.setSpacing(2)
        title = QLabel(f"Bulk edit · {len(self._rows)} row"
                       f"{'s' if len(self._rows) != 1 else ''} selected")
        title.setObjectName("issue-dialog-title")
        sub = QLabel(
            "Pick a column and the new value, then narrow it: to the rows "
            "that currently say one particular thing, and then row by row "
            "in the preview. Entity rebuilds and BIDS names update "
            "automatically."
        )
        sub.setObjectName("issue-dialog-subtitle")
        sub.setWordWrap(True)
        hl.addWidget(title)
        hl.addWidget(sub)
        outer.addWidget(header)

        # ---------- form ----------
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(8)
        form.setContentsMargins(0, 0, 0, 0)

        self._col_combo = QComboBox()
        self._col_combo.setObjectName("ent-input")
        for key in InventoryTableModel.BULK_EDITABLE_KEYS:
            spec = next((c for c in COLUMNS if c.key == key), None)
            label = spec.header if (spec and spec.header) else key
            self._col_combo.addItem(label, userData=key)
        # Then every entity the schema allows on all the selected rows and
        # that has no column of its own. Without these the dialog could not
        # set ``acq``, which is the entity that tells two otherwise
        # identical acquisitions apart and therefore the one most often
        # wanted on a multi-row selection.
        # Every entity the schema will let these rows go WITHOUT.
        self._removable = self._model.bulk_removable_entities(self._rows)
        self._entity_descriptions: dict[str, str] = {}
        self._entity_full_descriptions: dict[str, str] = {}
        for entity in self._model.bulk_editable_entities(self._rows):
            try:
                info = schema_mod.entity_info(entity)
            except KeyError:
                continue
            key = f"{InventoryTableModel.ENTITY_KEY_PREFIX}{entity}"
            self._col_combo.addItem(f"{info.name} ({entity})", userData=key)
            full = info.description.strip()
            self._entity_full_descriptions[key] = full
            self._entity_descriptions[key] = (
                f"{_first_paragraph(full)} "
                f"Written as {info.name}-<{info.format.name}>."
            ).strip()
        self._col_combo.currentIndexChanged.connect(self._on_column_changed)
        form.addRow("Column:", self._col_combo)

        # WHICH of the selected rows. Populated per column from the values
        # actually in use, with counts, so "only the ones that say rest" is
        # a thing you pick rather than a selection you have to rebuild in
        # the table.
        self._target_combo = QComboBox()
        self._target_combo.setObjectName("ent-input")
        self._target_combo.setToolTip(
            "Narrow the change to the rows that currently hold one "
            "particular value. The selection decides what is on offer; "
            "this decides which of it is written to."
        )
        self._target_combo.currentIndexChanged.connect(
            lambda _i: self._schedule_replan()
        )
        form.addRow("Change:", self._target_combo)


        # Value editor — swapped in/out depending on the column kind.
        # For ``datatype`` / ``suffix`` we offer a schema-bounded combo;
        # everything else uses a free-form line edit.
        self._value_edit = QLineEdit()
        self._value_edit.setObjectName("tb-input")
        self._value_edit.textChanged.connect(
            lambda _t: self._schedule_replan()
        )
        self._value_combo = QComboBox()
        self._value_combo.setObjectName("ent-input")
        self._value_combo.setEditable(True)  # users can still free-type
        self._value_combo.currentTextChanged.connect(
            lambda _t: self._schedule_replan()
        )
        form.addRow("New value:", self._value_edit)
        form.addRow("", self._value_combo)
        self._value_combo.setVisible(False)

        # Clearing an entity is a different instruction from setting it to
        # nothing, and there is no way to type the difference, so it is a
        # tick rather than an empty box. Only entities can be removed: a
        # column is part of the table's shape and an empty one is a blank
        # cell, not an absent field.
        self._remove_check = QCheckBox("Remove this entity from the rows that have it")
        self._remove_check.setToolTip(
            "Takes the entity off the selected rows entirely, so it stops "
            "appearing in their BIDS names. The value box is ignored.\n\n"
            "Applied row by row: a row that never carried it is untouched, "
            "and a row whose datatype REQUIRES it is left alone rather than "
            "refusing the whole edit. So you can select everything and take "
            "one label off the study."
        )
        self._remove_check.toggled.connect(self._on_remove_toggled)
        form.addRow("", self._remove_check)

        # How many rows a removal reaches, or why it cannot. On its OWN line,
        # directly under the tick it describes. It used to be appended to the
        # schema description, and for an entity like acq that description is
        # several paragraphs long: at the dialog's default size the text was
        # clipped top and bottom, and the line cut off was this one.
        self._reach = QLabel("")
        self._reach.setObjectName("pane-hint")
        self._reach.setWordWrap(True)
        self._reach.setStyleSheet("padding: 0;")
        form.addRow("", self._reach)

        # Which editor is active. Tracked explicitly rather than via
        # ``isVisible()`` (unreliable before the dialog is shown / in tests).
        self._value_is_combo = False

        # Per-column description so the user knows what the apply will do.
        self._description = QLabel("")
        self._description.setObjectName("pane-hint")
        self._description.setWordWrap(True)
        self._description.setContentsMargins(0, 0, 0, 0)
        # Reset padding from the global ``pane-hint`` rule for this
        # compact placement.
        self._description.setStyleSheet("padding: 6px 0;")

        bl.addLayout(form)
        bl.addWidget(self._description)

        preview_label = QLabel("What would change")
        preview_label.setObjectName("card-title")
        bl.addWidget(preview_label)

        self._preview = QTreeWidget()
        self._preview.setObjectName("check-tree")
        self._preview.setColumnCount(3)
        self._preview.setHeaderLabels(["Row", "Now", "Becomes"])
        self._preview.setRootIsDecorated(False)
        self._preview.setUniformRowHeights(True)
        self._preview.itemChanged.connect(self._on_row_ticked)
        bl.addWidget(self._preview, 1)

        picks = QHBoxLayout()
        picks.setSpacing(8)
        for text, state in (
            ("Select all", Qt.CheckState.Checked),
            ("Select none", Qt.CheckState.Unchecked),
        ):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, s=state: self._set_all(s))
            picks.addWidget(btn)
        picks.addStretch(1)
        self._summary = QLabel("")
        self._summary.setObjectName("pane-hint")
        picks.addWidget(self._summary)
        bl.addLayout(picks)

        outer.addWidget(body, 1)

        # ---------- footer ----------
        footer = QFrame()
        footer.setObjectName("issue-dialog-footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(14, 10, 14, 10)
        fl.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel,
        )
        self._apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self._apply_btn.clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        fl.addWidget(buttons)
        outer.addWidget(footer)

        # Initialise the value editor for whatever column is selected.
        self._on_column_changed(self._col_combo.currentIndex())

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def changed_count(self) -> int:
        """Rows actually modified by the last :meth:`exec` call."""
        return self._changed

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_column_changed(self, _idx: int) -> None:
        """Swap the value editor + refresh the description text."""
        key = self._col_combo.currentData()
        if key is None:
            return
        self._description.setText(
            self._entity_descriptions.get(key)
            or _COLUMN_DESCRIPTION.get(key, "")
        )
        # The whole schema text is one hover away; the label shows only its
        # first paragraph, so the preview below keeps its room.
        self._description.setToolTip(
            self._entity_full_descriptions.get(key, "")
        )
        self._reach.setText("")
        self._reach.setVisible(False)

        # Which entity does this column edit? An entity-prefixed key names
        # one directly; ``ses`` / ``task`` / ``run`` edit one through a
        # column of their own, and used to carry no remove tick at all, so
        # a session could be set and never unset.
        entity = self._entity_behind(key)
        removable = bool(entity) and entity in self._removable
        self._remove_check.setVisible(removable)
        if not removable:
            self._remove_check.setChecked(False)
        if entity and not removable:
            # Say WHY rather than hiding a control with no explanation.
            reason = (
                "every BIDS file has one and it cannot be taken off. Use "
                "Rename, so the participants.tsv row travels with it."
                if entity == "subject" else
                f"BIDS requires {entity} on every selected row, or none of "
                f"them is allowed to carry it."
            )
            self._reach.setText(f"Cannot be removed: {reason}")
            self._reach.setVisible(True)
        elif removable:
            # How many rows this will actually touch, before it is ticked.
            can = sum(
                1 for r in self._rows
                if self._model.entities(r).get(entity)
                and self._model.entity_removable_on(r, entity)
            )
            blocked = sum(
                1 for r in self._rows
                if self._model.entities(r).get(entity)
                and not self._model.entity_removable_on(r, entity)
            )
            note = f"Removing takes it off {can} of {len(self._rows)} rows."
            if blocked:
                note += f" {blocked} require it and are left alone."
            self._reach.setText(note)
            self._reach.setVisible(True)
        self._refill_targets(key)

        # Decide which editor to show.
        if key.startswith(InventoryTableModel.ENTITY_KEY_PREFIX):
            # An entity value is free text bounded by the entity's own
            # format, which ``set_entity`` and the basename rebuild check.
            # Offering the values already in use saves retyping one that
            # some rows carry and others do not.
            entity = key[len(InventoryTableModel.ENTITY_KEY_PREFIX):]
            in_use = self._model.entity_values_in_use(self._rows, entity)
            if in_use:
                self._show_value_combo(in_use, editable=True)
            else:
                self._show_value_lineedit()
        elif key == "datatype":
            # Schema-bounded but editable (large set, lets the user type).
            self._show_value_combo(sorted(schema_mod.list_datatypes()), editable=True)
        elif key == "suffix":
            # ``suffix`` depends on datatype, but bulk-applying spans
            # rows that may differ. Offer the union of all datatypes'
            # suffixes so the user can pick something reasonable.
            options = sorted({
                s for dt in schema_mod.list_datatypes()
                for s in schema_mod.list_suffixes(dt)
            })
            self._show_value_combo(options, editable=True)
        elif key == "montage":
            # Dropdown-only: MNE built-in montages, never hand-typed.
            self._show_value_combo(builtin_montages(), editable=False)
        elif key in _FIXED_CHOICES:
            # Short closed sets (line_freq / sex / handedness): dropdown-only.
            self._show_value_combo(_FIXED_CHOICES[key], editable=False)
        else:
            self._show_value_lineedit()

    def _on_remove_toggled(self, removing: bool) -> None:
        """Grey the value editor out: with Remove ticked it is not read."""
        self._value_edit.setEnabled(not removing)
        self._value_combo.setEnabled(not removing)
        self._schedule_replan()

    # ------------------------------------------------------------------
    # Which rows
    # ------------------------------------------------------------------

    def _entity_behind(self, key: str) -> str:
        """The entity long name a column key edits, or ``""``.

        Two spellings reach the same entity: the dialog offers ``ses`` as a
        COLUMN and ``direction`` as an ENTITY, and both end up writing an
        entity into the basename. Removal has to work the same either way.
        """
        prefix = InventoryTableModel.ENTITY_KEY_PREFIX
        if key.startswith(prefix):
            return key[len(prefix):]
        for entity, column in InventoryTableModel._ENTITY_COLUMN_KEYS.items():
            if column == key:
                return entity
        return ""

    def _refill_targets(self, key: str) -> None:
        """Offer the values this column actually holds, with row counts."""
        counts: dict[str, int] = {}
        for row in self._rows:
            counts[self._model.bulk_value(row, key)] = (
                counts.get(self._model.bulk_value(row, key), 0) + 1
            )
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem(
            f"every selected row ({len(self._rows)})", _ALL_ROWS,
        )
        for value, n in sorted(counts.items(), key=lambda kv: (not kv[0], kv[0])):
            if value:
                self._target_combo.addItem(
                    f"only the rows that say {value} ({n})", value,
                )
            else:
                self._target_combo.addItem(
                    f"only the rows that say nothing ({n})", _BLANK,
                )
        self._target_combo.blockSignals(False)
        self._replan()

    def _matching_rows(self) -> list[int]:
        """The selected rows the target filter keeps."""
        key = self._col_combo.currentData()
        wanted = self._target_combo.currentData()
        if key is None or wanted in (None, _ALL_ROWS):
            return list(self._rows)
        if wanted == _BLANK:
            return [r for r in self._rows if not self._model.bulk_value(r, key)]
        return [
            r for r in self._rows if self._model.bulk_value(r, key) == wanted
        ]

    # ------------------------------------------------------------------
    # The preview
    # ------------------------------------------------------------------

    def _schedule_replan(self) -> None:
        if not self._loading:
            self._replan_timer.start()

    def _replan(self) -> None:
        """Redraw the per-row preview from the current column and value."""
        key = self._col_combo.currentData()
        if key is None:
            return
        removing = self._remove_check.isChecked()
        new_value = "" if removing else self._read_value()

        self._loading = True
        self._preview.clear()
        for row in self._matching_rows():
            now = self._model.bulk_value(row, key)
            if now == new_value:
                continue                    # already says it; nothing to do
            item = QTreeWidgetItem(self._preview, [
                self._model.row_label(row),
                now or "(nothing)",
                "(removed)" if removing else (new_value or "(nothing)"),
            ])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, int(row))
        for column in range(3):
            self._preview.resizeColumnToContents(column)
        self._loading = False
        self._refresh_summary()

    def _on_row_ticked(self, _item, _column: int) -> None:
        if not self._loading:
            self._refresh_summary()

    def _set_all(self, state) -> None:
        self._loading = True
        for i in range(self._preview.topLevelItemCount()):
            self._preview.topLevelItem(i).setCheckState(0, state)
        self._loading = False
        self._refresh_summary()

    def _ticked_rows(self) -> list[int]:
        return [
            int(self._preview.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole))
            for i in range(self._preview.topLevelItemCount())
            if self._preview.topLevelItem(i).checkState(0)
            == Qt.CheckState.Checked
        ]

    def _refresh_summary(self) -> None:
        ticked = len(self._ticked_rows())
        total = self._preview.topLevelItemCount()
        if not total:
            self._summary.setText(
                "Nothing would change: every row in scope already says that."
            )
        else:
            self._summary.setText(f"{ticked} of {total} row(s) selected")
        self._apply_btn.setEnabled(bool(ticked))
        self._apply_btn.setText(
            f"Apply to {ticked} row(s)" if ticked else "Apply"
        )

    def _show_value_combo(self, options: list[str], *, editable: bool = False) -> None:
        self._value_combo.blockSignals(True)
        self._value_combo.clear()
        self._value_combo.setEditable(editable)
        self._value_combo.addItems(options)
        self._value_combo.blockSignals(False)
        self._value_edit.setVisible(False)
        self._value_combo.setVisible(True)
        self._value_is_combo = True

    def _show_value_lineedit(self) -> None:
        self._value_combo.setVisible(False)
        self._value_edit.setVisible(True)
        self._value_is_combo = False

    def _read_value(self) -> str:
        if self._value_is_combo:
            return self._value_combo.currentText().strip()
        return self._value_edit.text().strip()

    def _on_apply(self) -> None:
        # Flush a replan the debounce is still holding. Typing a value and
        # pressing Apply inside the delay would otherwise write the plan
        # from BEFORE the value was typed.
        if self._replan_timer.isActive():
            self._replan_timer.stop()
            self._replan()

        key = self._col_combo.currentData()
        if key is None:
            return
        value = self._read_value()

        if not value:
            # An ENTITY can be cleared, and clearing it is the only way to
            # take one off a group of files: an entity the scan proposed
            # but the dataset should not carry. Every other column keeps
            # the old rule, because an empty value there is a user who has
            # not finished typing rather than an instruction. The tick is
            # what distinguishes the two, and it is only ever offered for
            # an entity the schema does not require.
            if not self._remove_check.isChecked():
                return
        elif self._remove_check.isChecked():
            # Removing wins over a value left in the box, but say so rather
            # than silently discarding what was typed.
            value = ""

        rows = self._ticked_rows()
        if not rows:
            return
        entity = self._entity_behind(key)
        if self._remove_check.isChecked() and entity:
            # Row by row, so a selection spanning several datatypes takes the
            # entity off the ones that can lose it instead of being refused
            # over the ones that cannot.
            self._changed, skipped = self._model.remove_entity_rows(rows, entity)
            self._skipped = skipped
        else:
            self._changed = self._model.bulk_set(rows, key, value)
            self._skipped = 0
        self.accept()

    def skipped_count(self) -> int:
        """Rows that carried the entity but are not allowed to lose it.

        Zero for every edit that is not a removal. The caller reports it, so
        "24 rows changed" never quietly means "and 6 were not".
        """
        return getattr(self, "_skipped", 0)


__all__ = ["BulkEditDialog"]
