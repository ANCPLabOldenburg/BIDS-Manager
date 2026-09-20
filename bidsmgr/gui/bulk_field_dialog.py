"""Pick the files an edit applies to, before it is applied.

One dialog, three callers: stating a field for more than one recording, fixing
a finding in every file it fired on, and stamping TODO placeholders. All three
ask the same question, so they ask it the same way.

The design rule is that a count is not enough. "12 files" tells the user how
many, not which, and not what each one says now. Every candidate is a row
showing its current value and what it would become, ticked by default only
when it would actually change. A file the standard does not declare the field
for is shown greyed with the reason, rather than hidden, because a user who
expected it there deserves to know why it is not.

The value is asked for with the SAME control the sidecar form uses, built from
the schema by :func:`bidsmgr.gui.widgets.template_form.build_field_widget`. It
used to be a bare text box with the placeholder "text, or JSON for a number,
list or object", which made fixing one field in twelve files a strictly worse
experience than fixing it in one: the vocabulary a field has was not offered,
its type was not enforced, its unit and description were not shown, and the
user was asked to hand-write JSON for a list. A field is filled the same way
wherever it is filled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import bulk_edit as be
from .dialog_chrome import build_footer_with, build_header
from .theme_manager import CUR

_COL_FILE = 0
_COL_NOW = 1
_COL_NEXT = 2


def _curated_for(field: str) -> tuple:
    """Values BIDS Manager offers for a field the standard leaves open.

    The same list the sidecar form uses, so a field is filled from the same
    options wherever it is filled. Imported here rather than at module scope
    to keep this dialog importable without the recording-metadata models.
    """
    try:
        from ..recording_meta.models import CURATED_SUGGESTIONS
    except ImportError:       # pragma: no cover - defensive
        return ()
    return tuple(CURATED_SUGGESTIONS.get(field, ()))


class BulkFieldDialog(QDialog):
    """Choose which files receive ``field = value``.

    Construct with the candidates already computed, or pass ``anchor`` and a
    scope and let the dialog compute them. Call :meth:`exec` and, when it
    returns accepted, read :meth:`selected` and :meth:`value`.
    """

    def __init__(
        self,
        root: Path,
        field: str,
        *,
        candidates: Optional[list[be.FileCandidate]] = None,
        anchor: Optional[Path] = None,
        initial_value: Any = None,
        allow_scope: bool = True,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._field = field
        self._anchor = anchor
        self._fixed_candidates = candidates
        self._candidates: list[be.FileCandidate] = []
        self._initial_value = initial_value
        self.setWindowTitle(title or f"Apply {field} to other files")
        self.setModal(True)
        self.resize(760, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            title or f"Apply {field} to other files",
            f"Write <b>{field}</b> into the files you tick. Nothing is "
            "written until you press Apply, and the whole batch is "
            "<b>one step</b> in the Editor's history.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(body)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(10)

        # Value. The schema's own control when the schema knows this field,
        # which is the same one the sidecar form builds, so a vocabulary is a
        # dropdown here too and a list is a row editor rather than hand-typed
        # JSON. ``self._spec`` is None when it does not, and the plain box is
        # the fallback rather than the default.
        self._spec = self._resolve_spec(candidates)
        v.addLayout(self._build_value_row())

        # Scope.
        self._scope_combo = QComboBox()
        self._scope_combo.setObjectName("ent-input")
        if allow_scope and candidates is None:
            for key in be.SCOPES:
                self._scope_combo.addItem(be.SCOPE_LABELS[key], key)
            self._scope_combo.currentIndexChanged.connect(self._reload)
            srow = QHBoxLayout()
            srow.setSpacing(8)
            srow.addWidget(QLabel("Look in:"))
            srow.addWidget(self._scope_combo, 1)
            v.addLayout(srow)
        else:
            self._scope_combo.setVisible(False)

        # Table.
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File", "Now", "Becomes"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(
            _COL_NOW, QHeaderView.ResizeMode.ResizeToContents
        )
        hh.setSectionResizeMode(
            _COL_NEXT, QHeaderView.ResizeMode.ResizeToContents
        )
        v.addWidget(self._table, 1)

        # Selection helpers.
        tools = QHBoxLayout()
        tools.setSpacing(8)
        for label, slot in (
            ("Select all", self._select_all),
            ("Select none", self._select_none),
            ("Only those that change", self._select_changing),
        ):
            btn = QPushButton(label)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(slot)
            tools.addWidget(btn)
        tools.addStretch(1)
        v.addLayout(tools)
        outer.addWidget(body, 1)

        self._summary = QLabel("")
        self._summary.setObjectName("dlg-hint")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        self._apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self._apply_btn.setObjectName("tb-btn-primary")
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._summary, buttons))

        self._reload()

    # -- data ----------------------------------------------------------

    def _reload(self) -> None:
        if self._fixed_candidates is not None:
            self._candidates = list(self._fixed_candidates)
        else:
            scope = self._scope_combo.currentData() or be.SCOPE_SAME_KIND
            self._candidates = be.candidates(
                self._root, self._field, anchor=self._anchor, scope=scope,
            )
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._candidates))
        for r, cand in enumerate(self._candidates):
            box = QCheckBox(cand.rel)
            box.setEnabled(cand.applicable)
            if not cand.applicable:
                box.setToolTip(cand.reason)
                box.setObjectName("candidate-off")
            box.stateChanged.connect(self._refresh_summary)
            self._table.setCellWidget(r, _COL_FILE, box)
            now = QTableWidgetItem(cand.current_text())
            now.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(r, _COL_NOW, now)
            nxt = QTableWidgetItem("")
            nxt.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(r, _COL_NEXT, nxt)
        self._refresh_preview()
        self._select_changing()

    # -- value ---------------------------------------------------------

    def _resolve_spec(self, candidates):
        """What the schema says about this field, for these files.

        The datatype and suffix come from the candidates, because a field's
        declaration depends on them: ``EchoTime`` is a number under ``func``
        and is not declared at all for ``eeg``. The first applicable candidate
        decides, since a grouped finding fired on one rule and a rule belongs
        to one datatype; a group spanning datatypes falls back to whichever
        comes first, which is still better than no schema at all.

        Returns ``None`` when the field is not declared anywhere we can see,
        which is the honest answer for a finding BIDS Manager raises itself.
        """
        from ..metadata.template_plan import as_template_field
        from .. import schema as schema_mod

        pool = candidates if candidates is not None else []
        pairs = [
            (c.datatype, c.suffix) for c in pool
            if getattr(c, "datatype", None) and getattr(c, "suffix", None)
        ]
        for datatype, suffix in pairs:
            try:
                specs = schema_mod.sidecar_fields(datatype, suffix)
            except (KeyError, ValueError, OSError):
                continue
            for spec in specs:
                if spec.name == self._field:
                    return as_template_field(spec)
        return None

    def _build_value_row(self) -> QHBoxLayout:
        """The label and control for the value being written."""
        from .widgets.template_form import (
            build_field_widget,
            connect_field_widget,
            field_label_widget,
            write_field_widget,
        )

        row = QHBoxLayout()
        row.setSpacing(8)

        if self._spec is None:
            # No schema declaration: a plain box, and say why rather than
            # pretending the field is free text by design.
            row.addWidget(QLabel("Value:"))
            self._value_edit = QLineEdit()
            self._value_edit.setObjectName("ent-input")
            self._value_edit.setPlaceholderText(
                "text, or JSON for a number, list or object"
            )
            self._value_edit.setToolTip(
                "The standard does not declare this field for these files, so "
                "there is no type or vocabulary to offer. JSON is parsed when "
                "it parses; anything else is written as text."
            )
            if self._initial_value is not None:
                self._value_edit.setText(_as_text(self._initial_value))
            self._value_edit.textChanged.connect(self._refresh_preview)
            row.addWidget(self._value_edit, 1)
            return row

        # The label carries the requirement level and the description, exactly
        # as it does in the sidecar form.
        row.addWidget(field_label_widget(self._spec))
        # The schema's vocabulary arrives with the field. On top of it, the
        # values BIDS Manager curates for the fields the standard leaves as
        # free text: the amplifiers, the caps, the electrode references, the
        # tracers. Passing nothing here left the same field a dropdown in the
        # sidecar form and a bare box in this dialog, which is the whole
        # complaint this change exists to answer. Suggestions, never a
        # restriction: the box stays typable either way.
        self._value_edit = build_field_widget(
            self._spec, _curated_for(self._field),
        )
        if isinstance(self._value_edit, (QLineEdit, QComboBox)):
            # Only the simple controls take the shared input styling; a row
            # list draws its own.
            self._value_edit.setObjectName("ent-input")
        if self._initial_value is not None:
            write_field_widget(self._value_edit, self._initial_value)
        connect_field_widget(self._value_edit, self._refresh_preview)
        # ...and again on every keystroke. ``connect_field_widget`` settles on
        # editingFinished, which is right for a form that commits on focus-out
        # and wrong here: this dialog shows what each file WOULD become, and a
        # preview that waits for focus to leave the only editable control
        # never updates at all.
        self._connect_live(self._value_edit)
        row.addWidget(self._value_edit, 1)
        if self._spec.unit:
            unit = QLabel(self._spec.unit)
            unit.setObjectName("dlg-hint")
            row.addWidget(unit)
        return row

    def _connect_live(self, widget: QWidget) -> None:
        """Refresh the preview on every keystroke, whatever the control is."""
        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(lambda _t: self._refresh_preview())
            if widget.isEditable():
                widget.lineEdit().textChanged.connect(
                    lambda _t: self._refresh_preview()
                )
            return
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda _t: self._refresh_preview())

    def value(self) -> Any:
        """The value to write, in the shape the schema declares.

        Read through ``read_field_widget`` when the schema knows the field, so
        a number arrives as a number and a list as a list rather than as the
        text of one. The JSON fallback is only for a field it does not know.
        """
        if self._spec is not None:
            from .widgets.template_form import read_field_widget

            got = read_field_widget(self._value_edit, self._spec)
            return "" if got is None else got
        text = self._value_edit.text().strip()
        if text == "":
            return ""
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _refresh_preview(self) -> None:
        new = self.value()
        shown = _as_text(new)
        pal = CUR()
        for r, cand in enumerate(self._candidates):
            item = self._table.item(r, _COL_NEXT)
            if item is None:
                continue
            if not cand.applicable:
                item.setText("not applicable")
                item.setForeground(_qcolor(pal["muted"]))
                continue
            item.setText(shown)
            item.setForeground(
                _qcolor(pal["accent"] if cand.would_change(new)
                        else pal["muted"])
            )
        self._refresh_summary()

    # -- selection -----------------------------------------------------

    def _boxes(self) -> list[QCheckBox]:
        out = []
        for r in range(self._table.rowCount()):
            w = self._table.cellWidget(r, _COL_FILE)
            if isinstance(w, QCheckBox):
                out.append(w)
        return out

    def _select_all(self) -> None:
        for b in self._boxes():
            if b.isEnabled():
                b.setChecked(True)

    def _select_none(self) -> None:
        for b in self._boxes():
            b.setChecked(False)

    def _select_changing(self) -> None:
        new = self.value()
        for b, cand in zip(self._boxes(), self._candidates):
            b.setChecked(bool(cand.would_change(new)))

    def _refresh_summary(self) -> None:
        n = sum(1 for b in self._boxes() if b.isChecked())
        total = len(self._candidates)
        self._summary.setText(f"{n} of {total} selected")
        if hasattr(self, "_apply_btn"):
            self._apply_btn.setEnabled(n > 0)

    def selected(self) -> list[be.FileCandidate]:
        """The candidates the user ticked."""
        return [
            cand for b, cand in zip(self._boxes(), self._candidates)
            if b.isChecked()
        ]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _qcolor(spec: str):
    from PyQt6.QtGui import QColor

    return QColor(spec)


__all__ = ["BulkFieldDialog"]
