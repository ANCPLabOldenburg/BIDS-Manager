"""Manage-columns dialog for the inspection table.

Replaces the one-at-a-time header right-click menu with a single popup that
lets the user toggle every column at once, with a plain-language description
of what each column means. Mandatory columns (row identity) are shown ticked
and disabled. Selections are applied on Save.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .models import COLUMN_DESCRIPTIONS, COLUMNS, MANDATORY_COLUMN_KEYS


class ColumnManagerDialog(QDialog):
    """Pick which inspection-table columns are shown.

    Parameters
    ----------
    current
        ``{column_key: visible}`` map of the present visibility state.
    parent
        Owning widget.

    After ``exec()`` returns ``Accepted``, :meth:`result_visibility` gives
    the chosen ``{key: visible}`` map (mandatory columns always ``True``).
    """

    def __init__(self, current: dict[str, bool], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager - Manage columns")
        self.resize(560, 620)
        self._checks: dict[str, QCheckBox] = {}

        v = QVBoxLayout(self)

        intro = QLabel(
            "Choose which columns the inspection table shows. Hover or read "
            "the note under each name for what it means. The first three "
            "columns identify the row and are always shown."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        v.addWidget(intro)

        # Select all / Hide optional shortcuts.
        bar = QHBoxLayout()
        all_btn = QPushButton("Select all")
        none_btn = QPushButton("Hide optional")
        defaults_btn = QPushButton("Defaults")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn.clicked.connect(lambda: self._set_all(False))
        defaults_btn.clicked.connect(self._restore_defaults)
        bar.addWidget(all_btn)
        bar.addWidget(none_btn)
        bar.addWidget(defaults_btn)
        bar.addStretch(1)
        v.addLayout(bar)

        # Scrollable list of column rows (checkbox + description).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setSpacing(8)
        for spec in COLUMNS:
            hv.addWidget(self._build_row(spec, current))
        hv.addStretch(1)
        scroll.setWidget(holder)
        v.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    # ------------------------------------------------------------------
    def _build_row(self, spec, current: dict[str, bool]) -> QWidget:
        row = QFrame()
        row.setObjectName("column-row")
        rl = QVBoxLayout(row)
        rl.setContentsMargins(2, 2, 2, 2)
        rl.setSpacing(1)

        label = spec.header.strip() or spec.key
        cb = QCheckBox(f"{label}   ({spec.key})")
        mandatory = spec.key in MANDATORY_COLUMN_KEYS
        cb.setChecked(True if mandatory else current.get(spec.key, spec.default_visible))
        if mandatory:
            cb.setEnabled(False)
            cb.setToolTip("Row-identity column - always shown.")
        self._checks[spec.key] = cb
        rl.addWidget(cb)

        desc = COLUMN_DESCRIPTIONS.get(spec.key, "")
        if desc:
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color: #8b949e; margin-left: 22px;")
            rl.addWidget(d)
        return row

    def _set_all(self, visible: bool) -> None:
        for key, cb in self._checks.items():
            if key not in MANDATORY_COLUMN_KEYS:
                cb.setChecked(visible)

    def _restore_defaults(self) -> None:
        for spec in COLUMNS:
            if spec.key in MANDATORY_COLUMN_KEYS:
                continue
            self._checks[spec.key].setChecked(spec.default_visible)

    def result_visibility(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, cb in self._checks.items():
            out[key] = True if key in MANDATORY_COLUMN_KEYS else cb.isChecked()
        return out


__all__ = ["ColumnManagerDialog"]
