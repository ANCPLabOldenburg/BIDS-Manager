"""Modal dialog: apply one value to one column across every selected row.

Reached from the **✎ Bulk edit…** toolbar button (enabled when ≥ 2
rows are selected in the inspection table). Dispatches through
:meth:`InventoryTableModel.bulk_set` so entity rebuilds, mirror cells
and the basename column all stay in sync.

For columns that have a schema-bounded set of values (``datatype``,
``suffix``), the dialog offers a combo box populated from the schema
engine; everything else uses a free-form ``QLineEdit``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from .models import COLUMNS, InventoryTableModel


# Human-readable header per column key. Keeps the dropdown clear about
# what each option actually does.
_COLUMN_DESCRIPTION: dict[str, str] = {
    "id":        "Subject identifier — updates BIDS_name AND the subject entity.",
    "dataset":   "Dataset slug — the convert verb groups rows by this column.",
    "ses":       "Session label (no ``ses-`` prefix; the converter adds it).",
    "task":      "Task entity.",
    "run":       "Run number.",
    "datatype":  "BIDS datatype (anat, func, dwi, …). Triggers a basename rebuild.",
    "suffix":    "BIDS suffix (T1w, bold, …). Triggers a basename rebuild.",
    "line_freq": "EEG/MEG power-line frequency (Hz).",
    "montage":   "EEG/MEG mne montage name.",
}


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
        self.resize(440, 280)
        self._model = model
        self._rows: list[int] = list(rows)
        self._changed: int = 0

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
            "Pick a column and the new value to write into every "
            "selected row. Entity rebuilds + basenames update "
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
        self._col_combo.currentIndexChanged.connect(self._on_column_changed)
        form.addRow("Column:", self._col_combo)

        # Value editor — swapped in/out depending on the column kind.
        # For ``datatype`` / ``suffix`` we offer a schema-bounded combo;
        # everything else uses a free-form line edit.
        self._value_edit = QLineEdit()
        self._value_edit.setObjectName("tb-input")
        self._value_combo = QComboBox()
        self._value_combo.setObjectName("ent-input")
        self._value_combo.setEditable(True)  # users can still free-type
        form.addRow("New value:", self._value_edit)
        form.addRow("", self._value_combo)
        self._value_combo.setVisible(False)
        self._value_row_index = 1  # the row we toggle (line edit vs combo)

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
        bl.addStretch(1)

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
        self._description.setText(_COLUMN_DESCRIPTION.get(key, ""))

        # Decide which editor to show.
        if key == "datatype":
            options = sorted(schema_mod.list_datatypes())
            self._show_value_combo(options)
        elif key == "suffix":
            # ``suffix`` depends on datatype, but bulk-applying spans
            # rows that may differ. Offer the union of all datatypes'
            # suffixes so the user can pick something reasonable.
            options = sorted({
                s for dt in schema_mod.list_datatypes()
                for s in schema_mod.list_suffixes(dt)
            })
            self._show_value_combo(options)
        else:
            self._show_value_lineedit()

    def _show_value_combo(self, options: list[str]) -> None:
        self._value_combo.blockSignals(True)
        self._value_combo.clear()
        self._value_combo.addItems(options)
        self._value_combo.blockSignals(False)
        self._value_edit.setVisible(False)
        self._value_combo.setVisible(True)

    def _show_value_lineedit(self) -> None:
        self._value_combo.setVisible(False)
        self._value_edit.setVisible(True)

    def _read_value(self) -> str:
        if self._value_combo.isVisible():
            return self._value_combo.currentText().strip()
        return self._value_edit.text().strip()

    def _on_apply(self) -> None:
        key = self._col_combo.currentData()
        value = self._read_value()
        if key is None or not value:
            # Nothing to do — keep the dialog open so the user can fix
            # the input rather than closing on empty.
            return
        self._changed = self._model.bulk_set(self._rows, key, value)
        self.accept()


__all__ = ["BulkEditDialog"]
