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

* **Editable** (``editable=True``) — value rendered as an inline
  editor sized for the current :class:`bidsmgr.editor.types.SidecarField`
  value kind:

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
      Empty text returns ``None`` so the user can clear a value.
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
            return None
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

    def __init__(
        self,
        level: str,
        key: str,
        value: str,
        value_kind: str,
        *,
        editable: bool = False,
        raw_value: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sc-row")
        self._level = level
        self._key = key
        self._value_kind = value_kind
        self._editable = editable
        self._editor: Optional[QWidget] = None

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(10)

        # 4px colored bar — QSS-driven via per-level object name.
        self._bar = QFrame()
        self._bar.setObjectName(_LEVEL_BAR_OBJECT.get(level, "sc-bar-opt"))
        self._bar.setFixedSize(4, 18)
        h.addWidget(self._bar)

        # Field name.
        key_lbl = QLabel(f'"{key}"')
        key_lbl.setObjectName("sc-key-dep" if level == "dep" else "sc-key")
        key_lbl.setMinimumWidth(220)
        if level == "dep":
            f = key_lbl.font()
            f.setStrikeOut(True)
            key_lbl.setFont(f)
        h.addWidget(key_lbl)

        # Value cell — read-only label or inline editor.
        if editable:
            self._editor = self._build_editor(value, value_kind, raw_value)
            h.addWidget(self._editor, 1)
        else:
            val_lbl = self._build_readonly_value(value, value_kind)
            h.addWidget(val_lbl, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return self._key

    @property
    def value_kind(self) -> str:
        return self._value_kind

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

    def _on_combo_activated(self, _index: int) -> None:
        combo = self._editor
        if not isinstance(combo, QComboBox):
            return
        value = combo.currentData()
        self.value_committed.emit(self._key, bool(value), self._value_kind)


__all__ = ["SidecarRow"]
