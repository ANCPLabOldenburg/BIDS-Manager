"""One row of the Editor's schema-aware sidecar form.

Each row is: ``[4px colored bar] "key": value`` where the bar color
encodes the schema-defined :class:`bidsmgr.editor.types.FieldLevel`
(REQUIRED red, RECOMMENDED amber, OPTIONAL grey, DEPRECATED grey with
strikethrough key). Lift-and-shift from
``inspector_proto/proto.py`` lines 414-457.

Two display modes:

* **Read-only** (``editable=False``, default) — value rendered as a
  styled :class:`QLabel`. Three flavours via QSS object name:

  * ``"todo"``  → ``sc-val-todo`` (italic red, used for literal
    ``"TODO"`` placeholders and validator-surfaced ``(missing)`` rows).
  * ``"num"``   → ``sc-val-num`` (purple, no quotes).
  * ``"str"``   → ``sc-val-str`` (blue, quoted).

* **Editable** (``editable=True``) — value rendered as an inline editor.

  When the caller supplies the field's ``schema_field``, the control is the
  one the STANDARD implies, built by the same
  :mod:`~bidsmgr.gui.widgets.template_form` code the metadata templates use:
  a controlled vocabulary becomes a dropdown, a boolean becomes true/false, a
  list of strings becomes an add-and-remove list, a number gets a numeric box
  with its unit beside it. One implementation of "how do you fill a field of
  this type", so the Editor and the templates cannot drift apart, and so a
  user who has answered a field once in the template meets the same control
  when they come back to it here.

  Without a ``schema_field`` (a key the standard does not declare, which the
  user is still entitled to keep) it falls back to the value-kind editors:

  * ``number`` → :class:`QLineEdit` with a permissive float validator.
  * ``bool``   → :class:`QComboBox` with ``true``/``false``.
  * everything else → :class:`QLineEdit`. On commit we try
    ``json.loads(text)`` first, so a user typing ``3`` into a missing
    field saves as an integer, ``true`` saves as a boolean,
    ``["a","b"]`` saves as an array, etc., with plain text as the
    fallback.

  On commit (Enter for line edits, change-of-selection for combos,
  focus-out for both) the row emits :pyattr:`value_committed` with
  the parsed Python value.

Theme handling is fully QSS-driven (same pattern as
``QPlainTextEdit#dock-log``); a dark↔light swap is just a global QSS
re-apply. ``repaint_for_palette`` is preserved as a no-op so any
existing cascade caller keeps working.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QWidget,
)


# Map our short level codes to the QSS object name suffix.
_LEVEL_BAR_OBJECT: dict[str, str] = {
    "req": "sc-bar-req",
    "rec": "sc-bar-rec",
    "opt": "sc-bar-opt",
    "dep": "sc-bar-dep",
}


def _parse_commit_text(
    text: str,
    value_kind: str,
) -> Any:
    """Convert raw editor text to the Python value to save.

    Rules:

    * ``number`` → ``float(text)`` (int when it has no fractional part).
      Empty text returns :data:`~bidsmgr.editor.field_values.REMOVE`, because
      a number has no empty form: there is no numeral meaning "unanswered",
      and ``null`` is a validation error. The caller deletes the key instead.
    * ``bool``   → handled by the combo's index, not this helper.
    * ``array`` / ``object`` → JSON-first parse; containers accepted
      because the field was already a container — the user is
      legitimately editing its shape.
    * everything else → JSON-first parse, but **scalars only**.
      Container literals (``["a","b"]``, ``{"k":1}``) typed into a
      scalar-kind field are kept as raw text rather than silently
      converting the field into a container.
    """
    text = text.strip()
    if value_kind == "number":
        if text == "":
            from ...editor.field_values import REMOVE

            return REMOVE
        try:
            f = float(text)
        except ValueError:
            return text  # let the caller decide; save as string
        return int(f) if f.is_integer() and "." not in text and "e" not in text.lower() else f
    if text == "":
        return ""
    # Permissive JSON-first parse so users can type literals.
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    # Containers may only flow through for fields that were already
    # containers — keeps scalar fields from accidentally being
    # promoted into lists / dicts by typed-in JSON.
    if isinstance(parsed, (dict, list)) and value_kind not in ("array", "object"):
        return text
    return parsed


class SidecarRow(QFrame):
    """One field row in the Editor's sidecar form.

    Parameters
    ----------
    level
        ``"req"`` | ``"rec"`` | ``"opt"`` | ``"dep"``.
    key
        The JSON field name (e.g. ``"RepetitionTime"``).
    value
        Stringified value (caller is responsible for formatting).
    value_kind
        One of ``"str"``, ``"num"``, ``"todo"`` (display kinds — these
        drive the QSS object name) **or** one of the validator's full
        kinds ``"string"``, ``"number"``, ``"bool"``, ``"array"``,
        ``"object"``, ``"null"``, ``"todo"``, ``"missing"`` when in
        editable mode. Editable mode uses the full kind to pick the
        right editor widget.
    editable
        When ``True``, the value cell becomes an inline editor and the
        row emits :pyattr:`value_committed` on commit.
    raw_value
        Optional raw Python value used to seed editors (e.g. the
        actual list / bool / number so the editor isn't fighting a
        stringified version). Falls back to ``value`` if not given.
    """

    # Emitted when the user commits an edit (Enter / focus-out / combo
    # change). Args: (key, parsed_value, value_kind).
    value_committed = pyqtSignal(str, object, str)
    # Right-click: state this field in more than one file at once.
    # Carries the key so the host can open the candidate picker.
    apply_to_others_requested = pyqtSignal(str)
    # The first keystroke in a field. ``value_committed`` only fires on
    # focus-out or Enter, so without this the pane could not say a file
    # had unsaved changes until the user clicked somewhere else.
    editing_started = pyqtSignal(str)
    # Right-click: which file this value is actually stated in. A form
    # shows EFFECTIVE metadata, so a value can be inherited from a
    # sidecar three levels up and editing here makes a second copy.
    explain_requested = pyqtSignal(str)

    def __init__(
        self,
        level: str,
        key: str,
        value: str,
        value_kind: str,
        *,
        editable: bool = False,
        raw_value: Any = None,
        schema_field: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sc-row")
        self._level = level
        self._key = key
        self._value_kind = value_kind
        self._editable = editable
        self._schema_field = schema_field
        self._editor: Optional[QWidget] = None

        # An answer is rarely true of one file only. Right-clicking a row
        # offers to state it across the files the standard declares it for,
        # which is the same picker the grouped-findings fix uses.
        if editable:
            self.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            self.customContextMenuRequested.connect(self._on_context_menu)

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(10)

        # 4px colored bar — QSS-driven via per-level object name.
        self._bar = QFrame()
        self._bar.setObjectName(_LEVEL_BAR_OBJECT.get(level, "sc-bar-opt"))
        self._bar.setFixedSize(4, 18)
        h.addWidget(self._bar)

        # Field name, with the unit the schema declares. A dose box that
        # does not say MBq is a box somebody will put Bq in.
        unit = getattr(schema_field, "unit", "") or ""
        label_text = f'"{key}"' + (f"  ({unit})" if unit else "")
        key_lbl = QLabel(label_text)
        key_lbl.setObjectName("sc-key-dep" if level == "dep" else "sc-key")
        key_lbl.setMinimumWidth(220)
        if level == "dep":
            f = key_lbl.font()
            f.setStrikeOut(True)
            key_lbl.setFont(f)
        h.addWidget(key_lbl)

        # Value cell — read-only label or inline editor.
        if editable:
            self._editor = (
                self._build_schema_editor(raw_value, value, value_kind)
                if schema_field is not None else
                self._build_editor(value, value_kind, raw_value)
            )
            h.addWidget(self._editor, 1)
        else:
            val_lbl = self._build_readonly_value(value, value_kind)
            h.addWidget(val_lbl, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Semi-transparent row tints for highlighting findings. Chosen to read on
    # both the dark and light palettes (low alpha over either background).
    _HIGHLIGHT_BG: dict[str, str] = {
        "err":   "rgba(207, 34, 46, 0.22)",
        "warn":  "rgba(191, 135, 0, 0.26)",
        "focus": "rgba(79, 195, 247, 0.22)",
    }

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)
        apply_act = menu.addAction("Apply this field to other files")
        apply_act.setToolTip(
            "Pick which other files should carry this field and value."
        )
        where_act = menu.addAction("Where does this value come from?")
        where_act.setToolTip(
            "A form shows effective metadata. This says which file actually "
            "states the value, which may not be the one you are looking at."
        )
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is apply_act:
            self.apply_to_others_requested.emit(self._key)
        elif chosen is where_act:
            self.explain_requested.emit(self._key)

    @property
    def key(self) -> str:
        return self._key

    @property
    def value_kind(self) -> str:
        return self._value_kind

    def set_highlight(self, severity: Optional[str]) -> None:
        """Tint the row to flag a finding (``"err"`` / ``"warn"`` / ``"focus"``)
        or clear it (``None``). Scoped to ``#sc-row`` so it only paints this
        row's background, leaving the level bar + value styling intact."""
        color = self._HIGHLIGHT_BG.get(severity or "")
        if color:
            self.setStyleSheet(
                f"QFrame#sc-row {{ background: {color}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet("")

    def editor(self) -> Optional[QWidget]:
        """Return the inline editor widget (or ``None`` for read-only rows)."""
        return self._editor

    def repaint_for_palette(self, pal: dict[str, str]) -> None:
        """API-compat no-op (see module docstring)."""
        del pal

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_readonly_value(self, value: str, value_kind: str) -> QLabel:
        """Read-only label. Accepts both the legacy display kinds
        (``"str"`` / ``"num"`` / ``"todo"``) and the validator's full
        vocabulary (``"string"`` / ``"number"`` / ``"bool"`` / ...).
        """
        if value_kind in ("todo", "missing"):
            lbl = QLabel(value)
            lbl.setObjectName("sc-val-todo")
        elif value_kind in ("num", "number", "bool", "null"):
            lbl = QLabel(value)
            lbl.setObjectName("sc-val-num")
        else:
            # "str" / "string" / "array" / "object" / anything else.
            lbl = QLabel(f'"{value}"')
            lbl.setObjectName("sc-val-str")
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return lbl

    def _build_schema_editor(
        self, raw_value: Any, value: str, value_kind: str,
    ) -> QWidget:
        """The control the SCHEMA implies for this field.

        Delegates to the template form so there is one answer to "how do you
        fill a field of this type" rather than two that drift.
        """
        from .template_form import (
            build_field_widget,
            connect_field_widget,
            write_field_widget,
        )
        from ...recording_meta import CURATED_SUGGESTIONS

        def curated_suggestions(name: str) -> tuple:
            return tuple(CURATED_SUGGESTIONS.get(name, ()))

        # The schema's own vocabulary comes with the field. On top of it,
        # the values BIDS Manager has seen for fields the standard leaves
        # free text: the amplifiers, the tracers, the cap manufacturers. They
        # were only ever offered in the metadata templates, so the same field
        # was a dropdown in one place and a bare box in the other. They are
        # suggestions, never a restriction: the box stays typable.
        widget = build_field_widget(
            self._schema_field, curated_suggestions(self._key),
        )
        seed = raw_value
        if seed is None and value_kind not in ("missing", "null"):
            seed = value
        if seed is not None and seed != "":
            try:
                write_field_widget(widget, seed)
            except (TypeError, ValueError, AttributeError):
                # A value that does not fit the control it was given, which
                # happens when a file states the wrong type. Leaving the
                # control empty would hide that; the read-only view and the
                # validator both still report it.
                pass
        connect_field_widget(widget, self._on_schema_committed)
        self._connect_typing(widget)
        widget.setObjectName("ent-input")
        return widget

    def _connect_typing(self, widget: QWidget) -> None:
        """Announce the first keystroke, whatever kind of control this is.

        The unsaved-changes indicator has to appear as the user types, not
        when they click away, and only the text-bearing controls can say so.
        """
        edit = None
        if isinstance(widget, QLineEdit):
            edit = widget
        elif isinstance(widget, QComboBox) and widget.isEditable():
            edit = widget.lineEdit()
        if edit is not None:
            edit.textEdited.connect(
                lambda _t: self.editing_started.emit(self._key)
            )

    def _on_schema_committed(self) -> None:
        from ...editor.field_values import empty_value_for
        from .template_form import read_field_widget

        if self._editor is None:
            return
        value = read_field_widget(self._editor, self._schema_field)
        if value is None:
            # The control is empty. ``None`` would be written as JSON null,
            # which is the one thing BIDS never accepts: measured against the
            # validator, ``{"Authors": null}`` is an error while ``[]`` and an
            # absent key are both clean. The empty form comes from the type.
            value = empty_value_for(self._schema_field)
        self.value_committed.emit(self._key, value, self._value_kind)

    def _build_editor(
        self,
        value: str,
        value_kind: str,
        raw_value: Any,
    ) -> QWidget:
        """Pick the editor for the given value kind."""
        if value_kind == "bool":
            combo = QComboBox()
            combo.setObjectName("sc-edit-bool")
            combo.addItem("true", True)
            combo.addItem("false", False)
            # Seed from the raw bool when available; fall back to the
            # stringified form for safety.
            seed_bool = raw_value if isinstance(raw_value, bool) else (
                str(value).strip().lower() == "true"
            )
            combo.setCurrentIndex(0 if seed_bool else 1)
            # ``activated`` only fires on user interaction (not programmatic
            # setCurrentIndex), so the initial seed above doesn't emit.
            combo.activated.connect(self._on_combo_activated)
            return combo

        # Default: QLineEdit. ``number`` gets a permissive validator
        # (locale-aware, accepts empty so the user can clear).
        edit = QLineEdit()
        edit.setObjectName(
            "sc-edit-num" if value_kind == "number" else "sc-edit-str"
        )
        edit.setText(self._seed_text(value, value_kind, raw_value))
        if value_kind == "number":
            validator = QDoubleValidator()
            validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
            edit.setValidator(validator)
        edit.editingFinished.connect(self._on_line_committed)
        # ``textEdited`` fires on user input only, never on the
        # programmatic ``setText`` above, so seeding a row does not mark
        # the file dirty.
        edit.textEdited.connect(
            lambda _t: self.editing_started.emit(self._key)
        )
        return edit

    def _seed_text(
        self,
        value: str,
        value_kind: str,
        raw_value: Any,
    ) -> str:
        """Pick the initial text for a QLineEdit-style editor.

        For complex kinds we serialise the raw Python value so the
        user sees real JSON (``["a","b"]``) rather than a Python repr.
        """
        if value_kind == "missing":
            return ""
        if value_kind == "null":
            return "null"
        if raw_value is not None and value_kind in (
            "array", "object", "number", "string",
        ):
            try:
                return json.dumps(raw_value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(raw_value)
        return value

    def _on_line_committed(self) -> None:
        edit = self._editor
        if not isinstance(edit, QLineEdit):
            return
        parsed = _parse_commit_text(edit.text(), self._value_kind)
        self.value_committed.emit(self._key, parsed, self._value_kind)

    def schema_field(self):
        """What the standard says about this field, or ``None``."""
        return self._schema_field

    def _on_combo_activated(self, _index: int) -> None:
        combo = self._editor
        if not isinstance(combo, QComboBox):
            return
        value = combo.currentData()
        self.value_committed.emit(self._key, bool(value), self._value_kind)


__all__ = ["SidecarRow"]
