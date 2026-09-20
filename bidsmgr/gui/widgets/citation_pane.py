"""``CITATION.cff`` as a form, the way a JSON sidecar is.

The citation file is the one dataset-level file the Editor could not open. It
is YAML rather than JSON, so the sidecar form could not render it, and it was
left to a text editor: the only file in a BIDS dataset where BIDS Manager asked
the user to know a syntax.

The field list, its levels and its descriptions come from
:mod:`bidsmgr.editor.cff`, the same model the conversion-time writer uses, so a
file written by the fix-up and a file edited here cannot disagree about what
the format is.

Keys the model has never heard of are shown at the bottom and preserved on
save. Somebody who hand-wrote a CFF key we do not offer should not lose it by
opening the file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...editor import cff
from .primitives import PaneHeader

log = logging.getLogger(__name__)

# Object names, not palette lookups: the colours live in ``theme.qss`` under
# these names, so a dark/light swap is the app-wide re-apply and this pane
# needs no repaint listener for them.
_LEVEL_OBJECT = {
    "required": "cff-key-required",
    "recommended": "cff-key-recommended",
    "optional": "cff-key-optional",
}


class CitationPane(QWidget):
    """Editor centre pane for ``CITATION.cff``."""

    file_saved = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._path: Optional[Path] = None
        self._root: Optional[Path] = None
        self._data: dict[str, Any] = {}
        self._editors: dict[str, QWidget] = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Citation"))

        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 6, 14, 6)
        bl.setSpacing(8)
        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        bl.addWidget(self._status)
        bl.addStretch(1)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.clicked.connect(self.reload)
        bl.addWidget(self._revert_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.clicked.connect(self.save)
        bl.addWidget(self._save_btn)
        v.addWidget(bar)

        self._body = QWidget()
        # Named so the QSS can give it a background. A plain QWidget inside a
        # scroll area does not inherit the parent's, and on a theme swap it
        # keeps the brush it was first polished with: the pane went dark and
        # the area behind the form stayed white.
        self._body.setObjectName("cff-body")
        self._body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._form = QVBoxLayout(self._body)
        self._form.setContentsMargins(14, 10, 14, 14)
        self._form.setSpacing(8)
        self._form.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("cff-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(self._body)
        scroll.setMinimumWidth(0)
        self._scroll = scroll
        v.addWidget(scroll, 1)

    # -- binding -------------------------------------------------------

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        self._path = Path(path) if path is not None else None
        self._root = Path(root) if root is not None else None
        self.reload()

    def reload(self) -> None:
        self._data = {}
        if self._path is not None and self._path.exists():
            self._data = cff.load(self._path) or {}
        self._rebuild()

    # -- form ----------------------------------------------------------

    def _clear(self) -> None:
        while self._form.count() > 1:
            item = self._form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._editors.clear()

    def _rebuild(self) -> None:
        self._clear()
        if self._path is None:
            self._status.setText("")
            return

        missing = cff.missing_required(self._data)
        if missing:
            self._status.setText(
                "missing required: " + ", ".join(missing)
            )
        elif not self._data:
            self._status.setText("empty or unreadable")
        else:
            self._status.setText(f"{len(self._data)} fields")

        for field in cff.FIELDS:
            self._form.insertWidget(
                self._form.count() - 1, self._row(field),
            )

        extra = [k for k in self._data if k not in cff.FIELDS_BY_NAME]
        if extra:
            note = QLabel(
                "Kept as written, and saved unchanged: " + ", ".join(extra)
            )
            note.setObjectName("pane-hint")
            note.setWordWrap(True)
            self._form.insertWidget(self._form.count() - 1, note)

    def _row(self, field: cff.CffField) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(6)
        label = QLabel(field.name)
        label.setObjectName(
            _LEVEL_OBJECT.get(field.level, "cff-key-optional")
        )
        tip = field.description
        if field.from_bids:
            tip += (
                f"\n\nFilled from '{field.from_bids}' in "
                "dataset_description.json, which the BIDS schema defines."
            )
        label.setToolTip(tip)
        head.addWidget(label)
        if field.from_bids:
            src = QLabel(f"from {field.from_bids}")
            src.setObjectName("cff-source")
            src.setToolTip(tip)
            head.addWidget(src)
        head.addStretch(1)
        layout.addLayout(head)

        value = self._data.get(field.name)
        if field.kind == "people":
            editor: QWidget = QPlainTextEdit()
            editor.setObjectName("ent-input")
            editor.setPlaceholderText("One person per line: Family, Given")
            editor.setMaximumHeight(90)
            if isinstance(value, list):
                editor.setPlainText(
                    "\n".join(cff.join_person(v) for v in value)
                )
        elif field.kind == "array":
            editor = QPlainTextEdit()
            editor.setObjectName("ent-input")
            editor.setPlaceholderText("One entry per line")
            editor.setMaximumHeight(70)
            if isinstance(value, list):
                editor.setPlainText("\n".join(str(v) for v in value))
            elif value:
                editor.setPlainText(str(value))
        else:
            editor = QLineEdit()
            editor.setObjectName("ent-input")
            if field.example:
                editor.setPlaceholderText(field.example)
            if value is not None:
                editor.setText(str(value))
        editor.setToolTip(tip)
        self._editors[field.name] = editor
        layout.addWidget(editor)
        return row

    def repaint_for_palette(self, pal: dict) -> None:
        """Force Qt to recompute the QSS for this subtree.

        The colours are all object names, but a global stylesheet re-apply
        does not always reach a widget that has already been polished: the
        documented unpolish/polish dance is what makes it take. Without it the
        form went dark and its background stayed light.
        """
        del pal
        for widget in (self, self._scroll, self._body):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
        self.update()

    # -- saving --------------------------------------------------------

    def collect(self) -> dict[str, Any]:
        """Read the form back, keeping anything the model does not know."""
        out: dict[str, Any] = {}
        for field in cff.FIELDS:
            editor = self._editors.get(field.name)
            if editor is None:
                continue
            if isinstance(editor, QPlainTextEdit):
                lines = [
                    line.strip() for line in editor.toPlainText().splitlines()
                    if line.strip()
                ]
                if not lines:
                    continue
                out[field.name] = (
                    [cff.split_person(line) for line in lines]
                    if field.kind == "people" else lines
                )
            else:
                text = editor.text().strip()
                if text:
                    out[field.name] = text
        for key, value in self._data.items():
            if key not in cff.FIELDS_BY_NAME:
                out[key] = value
        return out

    def save(self) -> bool:
        """Write the file, reversibly when it sits inside a dataset."""
        if self._path is None:
            return False
        data = self.collect()
        text = cff.dumps(data)
        try:
            if self._root is not None:
                from ...project.operations import begin_operation

                with begin_operation(self._root, "Edit CITATION.cff") as op:
                    op.write_text(self._path, text)
            else:
                self._path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("could not save %s: %s", self._path, exc)
            self._status.setText(f"save failed: {exc}")
            return False
        self._data = data
        self._rebuild()
        self.file_saved.emit(self._path)
        return True


__all__ = ["CitationPane"]
