"""Properties panel — col 4 of the Converter view.

Schema-driven editor for the selected row's BIDS entities. Reference:
``inspector_proto/proto.py`` lines 669-722.

Layout (top → bottom):

1. **datatype / suffix combos** — required (* in red). Editing either
   re-derives the entity allow-list below.
2. **Entities form** — one row per entity allowed by the schema for
   the current ``(datatype, suffix)``. Required entities marked with
   ``*``; optional entities labelled ``opt``. Each field is a
   ``QLineEdit`` whose ``editingFinished`` writes back through
   :meth:`InventoryTableModel.set_entity`.
3. **Predicted path preview** — token-coloured monospace string built
   from ``schema.build_relative_path``.
4. **Validation messages** — one ``ValMessage`` per
   :class:`schema.ValidationVerdict` from
   :func:`schema.validate_entity_set`.
5. **Why this name?** — small provenance section reading from the
   project's ``ProvenanceMap`` when one exists.

The panel never owns state — it reads from the model on every
``set_selected_row`` call and writes through the model's API. This
keeps the data flow unidirectional (model → panel → model) and means
the panel auto-refreshes when the model emits ``dataChanged``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import schema as schema_mod
from ..project import Project
from .models import InventoryTableModel
from .theme_manager import CUR, scaled_px
from .widgets import PaneHeader, ValMessage

log = logging.getLogger(__name__)


# Stable display order for entities (matches BIDS spec). Entities not
# listed here are appended after these, schema-defined order.
_ENTITY_DISPLAY_ORDER: tuple[str, ...] = (
    "subject", "session", "task", "acquisition", "ceagent",
    "reconstruction", "direction", "run", "echo", "part", "chunk",
)


class _EntityRow(QWidget):
    """One ``[label] [QLineEdit]`` row for an entity.

    Self-contained so the form layout can drop in / pull out rows
    without leaking state. Emits no signal — the parent panel reads
    the value back when committing.
    """

    def __init__(
        self,
        entity_name: str,
        value: str,
        *,
        required: bool,
        deprecated: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entity_name = entity_name
        self.setStyleSheet("background: transparent;")

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        label_text = entity_name
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        pal = CUR()
        css_color = pal["text"] if required else pal["dim"]
        suffix = f' <span style="color:{pal["error"]}">*</span>' if required else ""
        if deprecated:
            css_color = pal["muted"]
            lbl.setText(f'<span style="color:{css_color};text-decoration:line-through;">{label_text}</span>')
        else:
            lbl.setText(f'<span style="color:{css_color}">{label_text}</span>{suffix}')
        lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(lbl)

        self.edit = QLineEdit(value)
        self.edit.setObjectName("ent-input")
        self.edit.setPlaceholderText("—")
        h.addWidget(self.edit, 1)

    def value(self) -> str:
        return self.edit.text().strip()


class PropertiesPanel(QWidget):
    """Right pane of the Converter view.

    Bind a model with :meth:`bind_model`, then call
    :meth:`set_selected_row` whenever the table's selection changes.
    Passing ``row=None`` blanks the form.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._model: Optional[InventoryTableModel] = None
        self._project: Optional[Project] = None
        self._row: Optional[int] = None
        self._suppress_writeback = False
        self._entity_rows: list[_EntityRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PaneHeader("Properties · no row selected"))
        self._header = outer.itemAt(0).widget()

        # Body lives inside a scroll area so the form can grow without
        # bumping the splitter handles. The scroll area's contents are
        # ``self._body`` which we rebuild on every set_selected_row.
        self._body = QWidget()
        self._body.setObjectName("props-panel")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 12, 14, 12)
        self._body_layout.setSpacing(7)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._body)
        outer.addWidget(scroll, 1)

        # Build the initial empty body (just a hint).
        self._render_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_model(self, model: Optional[InventoryTableModel]) -> None:
        """Attach (or detach with ``None``) the inventory model.

        Connects ``dataChanged`` so external edits to the active row
        (e.g. the table's mirror cells) reflow into the form.
        """
        if self._model is not None:
            try:
                self._model.dataChanged.disconnect(self._on_model_data_changed)
            except (TypeError, RuntimeError):
                pass
        self._model = model
        if model is not None:
            model.dataChanged.connect(self._on_model_data_changed)
        self.set_selected_row(None)

    def set_project(self, project: Optional[Project]) -> None:
        """Attach a project for provenance lookups. Not used yet for writes."""
        self._project = project

    def repaint_for_palette(self, _pal: dict) -> None:
        """Rebuild the body so inline palette reads pick up new colors.

        The form is constructed from ``CUR()`` reads at render time
        (label colors, predicted-path tokens, validation borders), so a
        full re-render is the cleanest way to refresh after a theme
        swap. Re-renders the currently-selected row (or the empty hint
        if nothing is selected).
        """
        self.set_selected_row(self._row)

    def set_selected_row(self, row: Optional[int]) -> None:
        """Render the form for ``row`` (or blank if ``None``).

        Called by the Converter view whenever the table selection
        changes. Cheap enough to call on every selection event — the
        body is rebuilt from scratch each time so we don't have to
        track per-row deltas.
        """
        self._row = row
        if self._model is None or row is None:
            self._render_empty()
            self._set_header("Properties · no row selected")
            return
        if not (0 <= row < self._model.rowCount()):
            self._render_empty()
            return
        self._set_header(f"Properties · row {row + 1} selected")
        self._render_for_row(row)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_body(self) -> None:
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._entity_rows = []

    def _render_empty(self) -> None:
        self._clear_body()
        hint = QLabel(
            "Select a row in the inspection table to edit its entities."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {CUR()['dim']}; padding: 24px 0;")
        hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body_layout.addWidget(hint)
        self._body_layout.addStretch(1)

    def _render_for_row(self, row: int) -> None:
        assert self._model is not None
        self._clear_body()

        # 1. datatype + suffix combos
        datatype, suffix = self._model.datatype_suffix(row)
        self._body_layout.addWidget(self._build_combo_row(
            "datatype", datatype, options=sorted(schema_mod.list_datatypes()),
            required=True, slot=self._on_datatype_changed,
        ))
        suffix_options = sorted(schema_mod.list_suffixes(datatype)) if datatype else []
        self._body_layout.addWidget(self._build_combo_row(
            "suffix", suffix, options=suffix_options,
            required=True, slot=self._on_suffix_changed,
        ))

        self._body_layout.addSpacing(4)
        self._body_layout.addWidget(self._divider())
        self._body_layout.addSpacing(2)

        # 2. Entities form
        ent_title = QLabel("Entities")
        ent_title.setStyleSheet(f"color: {CUR()['text']}; font-weight: 600;")
        sub = QLabel("  (schema-driven)")
        sub.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px;")
        head = QHBoxLayout()
        head.setSpacing(0)
        head.addWidget(ent_title)
        head.addWidget(sub)
        head.addStretch(1)
        head_wrap = QWidget()
        head_wrap.setLayout(head)
        head_wrap.setStyleSheet("background: transparent;")
        self._body_layout.addWidget(head_wrap)
        self._body_layout.addSpacing(2)

        entities = self._model.entities(row)
        ordered = self._ordered_entities(datatype, suffix)
        required_set = set(schema_mod.required_entities(datatype, suffix)) if datatype and suffix else set()
        deprecated_set = set(schema_mod.deprecated_entities(datatype, suffix)) if datatype and suffix else set()

        for entity in ordered:
            value = entities.get(entity, "")
            er = _EntityRow(
                entity,
                value,
                required=entity in required_set,
                deprecated=entity in deprecated_set,
            )
            er.edit.editingFinished.connect(
                lambda e=er: self._on_entity_committed(e.entity_name, e.value())
            )
            self._entity_rows.append(er)
            self._body_layout.addWidget(er)

        self._body_layout.addSpacing(6)
        self._body_layout.addWidget(self._divider())

        # 3. Predicted path preview
        sec = QLabel("PREDICTED PATH")
        sec.setStyleSheet(f"color: {CUR()['dim']}; font-size: {scaled_px(10)}px; font-weight: 600;")
        self._body_layout.addWidget(sec)
        self._body_layout.addWidget(self._build_path_preview(row, datatype, suffix, entities))

        # 4. Row-state notice (from the scanner's proposed_issues) +
        # schema validation. Two distinct sources of "what's wrong with
        # this row": scanner-detected operational issues vs. schema's
        # entity-set verdicts. Both render with the same ValMessage
        # widget so the user sees them as one ranked list.
        self._body_layout.addSpacing(8)
        for vmsg in self._build_row_issue_messages(row):
            self._body_layout.addWidget(vmsg)
        for vmsg in self._build_validation_messages(datatype, suffix, entities):
            self._body_layout.addWidget(vmsg)

        self._body_layout.addStretch(1)

    def _build_combo_row(self, label_text: str, value: str, *, options: list[str],
                         required: bool, slot) -> QWidget:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        pal = CUR()
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(76)
        lbl.setMaximumWidth(76)
        suffix = f' <span style="color:{pal["error"]}">*</span>' if required else ""
        lbl.setText(f'<span style="color:{pal["text"]}">{label_text}</span>{suffix}')
        lbl.setTextFormat(Qt.TextFormat.RichText)
        h.addWidget(lbl)

        combo = QComboBox()
        combo.setObjectName("ent-input")
        combo.setMinimumHeight(22)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if options:
            combo.addItems(options)
        if value and value not in options:
            combo.addItem(value)
        if value:
            combo.setCurrentText(value)
        # Use ``activated`` (not ``currentTextChanged``) so we only fire on
        # user interaction, not on programmatic ``setCurrentText`` during
        # render.
        combo.activated.connect(lambda _i, c=combo: slot(c.currentText()))
        h.addWidget(combo, 1)
        return row

    def _build_path_preview(
        self, row: int, datatype: str, suffix: str, entities: dict[str, str],
    ) -> QWidget:
        f = QFrame()
        f.setObjectName("path-preview")
        l = QVBoxLayout(f)
        l.setContentsMargins(11, 9, 11, 9)
        l.setSpacing(0)
        pal = CUR()

        # Compose the same token list the prototype rendered, derived
        # from the schema instead of hard-coded data.
        pieces: list[str] = []
        if datatype and suffix and entities.get("subject"):
            try:
                rel = str(schema_mod.build_relative_path(
                    entities, datatype, suffix, ".nii.gz",
                ))
                # Insert a newline between the directory part and the
                # basename so the preview wraps clearly.
                head, _slash, base = rel.rpartition("/")
                pieces.append(self._color_path_segment(head + "/", pal["accent"]))
                pieces.append("<br>")
                # Basename split: tokens are ``key-value`` pairs joined
                # by underscores, ending in the suffix.
                tokens = base.split("_")
                last_index = len(tokens) - 1
                for i, tok in enumerate(tokens):
                    if i == last_index:
                        # split off extension
                        if "." in tok:
                            suf, _, ext = tok.partition(".")
                            pieces.append(f'<span style="color:{pal["teal"]}">{suf}</span>')
                            pieces.append(f'<span style="color:{pal["dim"]}">.{ext}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["teal"]}">{tok}</span>')
                    else:
                        if "-" in tok:
                            pieces.append(f'<span style="color:{pal["purple"]}">{tok}</span>')
                        else:
                            pieces.append(f'<span style="color:{pal["accent"]}">{tok}</span>')
                    if i < last_index:
                        pieces.append("_")
            except (ValueError, KeyError, TypeError) as exc:
                pieces.append(f'<span style="color:{pal["error"]}">cannot build path: {exc}</span>')
        else:
            pieces.append(
                f'<span style="color:{pal["muted"]}">'
                'Pick datatype + suffix to preview the path.'
                '</span>'
            )

        lbl = QLabel("".join(pieces))
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            'font-family: "SF Mono","Menlo","Monaco",monospace; '
            f'font-size: {scaled_px(11)}px; color: {pal["text"]}; '
            'background: transparent;'
        )
        l.addWidget(lbl)
        return f

    @staticmethod
    def _color_path_segment(text: str, color: str) -> str:
        return f'<span style="color:{color}">{text}</span>'

    def _build_row_issue_messages(self, row: int) -> list[QWidget]:
        """One ``ValMessage`` per scanner-detected issue on the selected row.

        Severity is derived from the model's ``row_state``: ``err`` →
        red badge, ``warn`` → amber, ``skip`` → muted (we surface it
        as ``warn`` here so the user sees the explanation). Returns an
        empty list when the row has no issues.
        """
        if self._model is None:
            return []
        issues = self._model.row_issues(row)
        if not issues:
            return []
        state = self._model.row_state(row)
        sev = {"err": "err", "warn": "warn", "skip": "warn"}.get(state, "warn")

        msgs: list[QWidget] = []
        for i, text in enumerate(issues):
            # First entry carries the rule_id "SCANNER · <row-state>";
            # follow-up entries are continuations of the same row so
            # the label is just blank to keep the column quiet.
            rule = f"SCANNER · {state}" if i == 0 else ""
            msgs.append(ValMessage(sev, rule, text, None))
        return msgs

    def _build_validation_messages(
        self, datatype: str, suffix: str, entities: dict[str, str],
    ) -> list[QWidget]:
        out: list[QWidget] = []
        if not datatype or not suffix:
            out.append(ValMessage(
                "warn", "SCHEMA",
                "Pick a datatype and suffix to validate the entity set.",
                None,
            ))
            return out

        verdicts = schema_mod.validate_entity_set(entities, datatype, suffix)
        if not verdicts:
            out.append(ValMessage(
                "ok", f"SCHEMA · {datatype}/{suffix}",
                "Entity set is valid.",
                None,
            ))
            return out
        for v in verdicts:
            sev = {
                schema_mod.Severity.ERROR: "err",
                schema_mod.Severity.WARNING: "warn",
                schema_mod.Severity.INFO: "ok",
            }.get(v.severity, "warn")
            out.append(ValMessage(sev, v.rule_id, v.message, None))
        return out

    @staticmethod
    def _ordered_entities(datatype: str, suffix: str) -> list[str]:
        """Display order: BIDS-spec order, fallback to schema-allow order."""
        if not datatype or not suffix:
            return list(_ENTITY_DISPLAY_ORDER)
        allowed = schema_mod.allowed_entities(datatype, suffix)
        # Put canonical order first (those that are in allowed), then any
        # remaining allowed entities in schema order.
        head = [e for e in _ENTITY_DISPLAY_ORDER if e in allowed]
        tail = [e for e in allowed if e not in head]
        return head + tail

    def _set_header(self, text: str) -> None:
        # The PaneHeader uppercases its constructor arg; we replace its
        # text directly to avoid restoring case rules.
        self._header.setText(text.upper())

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setStyleSheet(
            f"background: {CUR()['subtle']}; max-height: 1px; "
            f"min-height: 1px; border: none;"
        )
        return d

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_entity_committed(self, entity: str, value: str) -> None:
        if self._suppress_writeback:
            return
        if self._model is None or self._row is None:
            return
        # Translate the schema entity name (``subject``) — already the
        # canonical form used by the schema engine and ProjectState.
        self._model.set_entity(self._row, entity, value or None)

    def _on_datatype_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        _dt, suffix = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, new_value, suffix)
        # Re-render: the suffix combo's options change with datatype.
        self.set_selected_row(self._row)

    def _on_suffix_changed(self, new_value: str) -> None:
        if self._model is None or self._row is None:
            return
        datatype, _sf = self._model.datatype_suffix(self._row)
        self._model.set_datatype_suffix(self._row, datatype, new_value)
        self.set_selected_row(self._row)

    def _on_model_data_changed(self, top_left, bottom_right, _roles=()) -> None:
        if self._model is None or self._row is None:
            return
        if top_left.row() <= self._row <= bottom_right.row():
            # Re-render but suppress the writeback that the rebuilt
            # ``QLineEdit`` widgets would trigger via ``editingFinished``
            # on focus loss during the rebuild.
            self._suppress_writeback = True
            try:
                self.set_selected_row(self._row)
            finally:
                self._suppress_writeback = False


__all__ = ["PropertiesPanel"]
