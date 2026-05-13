"""Recursive 2-column key/value editor for a JSON :class:`OrderedDict`.

Visual reference: the original BIDS-Manager Inspector
(:class:`BIDS-Manager/bids_manager/gui.py::MetadataViewer`) — a
``QTreeWidget`` with **Key** + **Value** columns, fully-recursive
rendering of nested dicts and lists, inline-editable cells, and a
small toolbar (Add Field / Delete Field) on the host pane.

Behaviour:

* **Dicts** render as expandable parent items; children carry the key
  in column 1 and the rendered scalar (or empty for nested containers)
  in column 2.
* **Lists** render with ``[N]`` keys (the index in brackets). Mixing
  numeric ``[N]`` keys reconstructs as a list on save; any other
  key shape reconstructs as a dict.
* **Scalars** (``int``, ``float``, ``bool``, ``str``, ``null``) render
  inline in the Value column.
* **Add / Delete** appear on the host pane's toolbar; Add inserts a
  sibling under the current selection's parent (or at root if no
  selection); Delete removes the current selection.

Validation colour-coding: each top-level row may carry a
:data:`LEVEL_ROLE` payload (``"req"`` / ``"rec"`` / ``"opt"`` /
``"dep"``). The custom :class:`_LevelBarDelegate` paints a 4 px
colour bar on the left edge of the row, mirroring the BIDS form's
``SidecarRow``-style colour coding.

Styling is QSS-driven via the widget's ``json-tree`` object name
plus the per-level QSS rules already in ``theme.qss``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from ..theme_manager import CUR

log = logging.getLogger(__name__)


# Per-item validation level. Stored at this role on top-level items
# (and ignored for descendants) so the delegate can paint a colour bar.
LEVEL_ROLE = Qt.ItemDataRole.UserRole + 5

# Map our short level codes to the palette token used for the bar.
_LEVEL_TO_TOKEN: dict[str, str] = {
    "req": "error",
    "rec": "warning",
    "opt": "muted_40",
    "dep": "muted",
}

# Stable display order — matches the BIDS form's sort.
_LEVEL_DISPLAY_ORDER: dict[str, int] = {
    "req": 0,
    "rec": 1,
    "opt": 2,
    "dep": 3,
}
# Fields with no level info (e.g. user-added keys before validation
# has classified them) sort after the four schema levels.
_NO_LEVEL_RANK = 4

# Regex used to detect list-style keys (``[0]``, ``[1]``, …) when
# converting the tree back into Python.
_INDEX_KEY_RE = re.compile(r"^\[\d+\]$")


def value_to_text(val: Any) -> str:
    """Render a Python scalar as the text shown in the Value column.

    Containers (``dict`` / ``list``) return ``""`` because they are
    rendered as expandable parent items rather than inline JSON.
    """
    if isinstance(val, (dict, list)):
        return ""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val
    return str(val)


def text_to_value(text: str) -> Any:
    """Parse Value-column text back to a Python **scalar**.

    Permissive JSON-first parse for scalars:

    * ``true`` / ``false`` / ``null`` → real ``bool`` / ``None``;
    * ``3`` → ``int``, ``1.5`` → ``float``;
    * anything else (including ``["a","b"]`` or ``{"k":1}``) → kept
      as plain string.

    Containers (``dict`` / ``list``) are **deliberately rejected** —
    they would silently promote a leaf row into a parent and grow
    a new sub-tree the user didn't ask for. New containers can only
    come from the Add field button (which always creates a leaf
    and lets the user grow nesting explicitly by selecting a parent
    that already has subfields).

    An empty cell becomes the empty string ``""``. To clear a key the
    user should delete the whole row via the toolbar Delete button.
    """
    text = text.strip()
    if text == "":
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(parsed, (dict, list)):
        # Container literals are kept as their raw text — see
        # docstring. The user wanted explicit-only nesting creation.
        return text
    return parsed


# --------------------------------------------------------------------------
# JsonTreeView
# --------------------------------------------------------------------------


class JsonTreeView(QTreeWidget):
    """Recursive 2-column editor for a top-level JSON object."""

    # Emitted whenever the tree is mutated by user input (cell edit,
    # add field, delete field).
    model_changed = pyqtSignal()

    # Width of the level-color bar painted on the left edge of every
    # top-level row. We also reserve this many pixels of viewport
    # margin so the bar doesn't paint over the chevron / key text.
    LEVEL_BAR_WIDTH = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("json-tree")
        self.setColumnCount(2)
        self.setHeaderLabels(["Key", "Value"])
        self.setAlternatingRowColors(True)
        self.setIndentation(18)
        self.setUniformRowHeights(True)
        # Inline edit on double-click / F2 / second click on selection.
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        # Suppress the model_changed signal while we're populating.
        self._suppress = False
        self.itemChanged.connect(self._on_item_changed)

    # ----------------------------------------------------------------------
    # Custom painting — level bar on the left of every top-level row.
    # ----------------------------------------------------------------------

    def drawRow(self, painter, options, index) -> None:  # noqa: N802
        """Render the row, then overlay a 4 px colour bar on the left.

        ``options.rect`` is the row rect in viewport coordinates — i.e.
        ``left()`` is the actual leftmost pixel of the row. Painting
        on top of the standard row paint sits cleanly over the indent
        area; for top-level rows this is the leftmost edge of the
        widget. We paint AFTER ``super().drawRow`` so the bar is the
        final layer on the left edge.
        """
        super().drawRow(painter, options, index)
        if index.parent().isValid():
            return
        level = index.data(LEVEL_ROLE)
        token = _LEVEL_TO_TOKEN.get(level) if level else None
        if not token:
            return
        pal = CUR()
        painter.save()
        painter.fillRect(
            options.rect.left(),
            options.rect.top(),
            self.LEVEL_BAR_WIDTH,
            options.rect.height(),
            QColor(pal[token]),
        )
        painter.restore()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def set_data(
        self,
        data: OrderedDict[str, Any] | None,
        *,
        levels: Optional[dict[str, str]] = None,
    ) -> None:
        """Render an :class:`OrderedDict` as the tree.

        ``levels`` maps **top-level** key names to one of
        ``"req"`` / ``"rec"`` / ``"opt"`` / ``"dep"``. Each top-level
        row is annotated with its level so :meth:`drawRow` can paint
        the colour bar. Nested rows inherit no level.

        Top-level rows are displayed in **level order**
        (required → recommended → optional → deprecated → unknown),
        matching the BIDS form. Within each level the order from
        ``data`` is preserved. On-disk key order is held in the host
        pane's cache and is not affected by the display sort.
        """
        self._suppress = True
        try:
            self.clear()
            if data is None:
                return
            levels = levels or {}
            # Stable sort: keys keep their relative order within a
            # level bucket, but the buckets themselves are reordered.
            sorted_keys = sorted(
                data.keys(),
                key=lambda k: _LEVEL_DISPLAY_ORDER.get(
                    levels.get(k, ""), _NO_LEVEL_RANK,
                ),
            )
            for key in sorted_keys:
                val = data[key]
                item = self._make_item(key, val)
                self.addTopLevelItem(item)
                if key in levels:
                    item.setData(0, LEVEL_ROLE, levels[key])
            self.expandToDepth(0)
        finally:
            self._suppress = False
        self.resizeColumnToContents(0)

    def to_dict(self) -> OrderedDict[str, Any]:
        """Read the tree back into a Python object (top-level dict).

        Nested children are reconstructed recursively. A parent whose
        children all carry ``[N]``-shaped keys becomes a list;
        everything else becomes a dict. Empty-key rows are skipped.
        """
        return self._dict_from_parent(self.invisibleRootItem())

    def add_field(self) -> QTreeWidgetItem:
        """Insert a fresh leaf row **at the top level** (root).

        The new row is named ``newKey`` (or ``[N]`` if all existing
        top-level rows are list-style indices) with an empty value;
        the cursor lands in the Key column ready to type. The
        current selection is ignored — for adding inside an existing
        container, use :meth:`add_subfield`.
        """
        return self._insert_leaf(parent=None)

    def add_subfield(
        self,
        parent_item: QTreeWidgetItem | None = None,
    ) -> QTreeWidgetItem | None:
        """Insert a fresh leaf row as a child of the selected field.

        ``parent_item`` defaults to the current selection. Returns
        ``None`` only when there's no selection at all.

        Matches the original BIDS-Manager Inspector's Add Field:
        works on **any** selected field, including leaves. When the
        selection is a leaf its prior scalar value is dropped — the
        item becomes a container holding the new child. This is the
        only path that can create new nesting; value-cell editing
        remains scalar-only (typing ``["a","b"]`` doesn't auto-promote).
        """
        target = parent_item or self.currentItem()
        if target is None:
            return None
        # If we're promoting a leaf, clear its Value cell so the
        # newly-container row visually matches how containers are
        # otherwise rendered (empty Value column).
        if target.childCount() == 0:
            self._suppress = True
            target.setText(1, "")
            self._suppress = False
        return self._insert_leaf(parent=target)

    def _insert_leaf(
        self,
        parent: QTreeWidgetItem | None,
    ) -> QTreeWidgetItem:
        """Common insertion path used by both Add and Add-subfield."""
        siblings_root = (
            parent if parent is not None else self.invisibleRootItem()
        )
        if (
            siblings_root.childCount() > 0
            and all(
                _INDEX_KEY_RE.match(
                    siblings_root.child(i).text(0).strip()
                )
                for i in range(siblings_root.childCount())
            )
        ):
            new_key = f"[{siblings_root.childCount()}]"
        else:
            new_key = "newKey"

        self._suppress = True
        new_item = QTreeWidgetItem([new_key, ""])
        new_item.setFlags(new_item.flags() | Qt.ItemFlag.ItemIsEditable)
        if parent is None:
            self.addTopLevelItem(new_item)
        else:
            parent.addChild(new_item)
            parent.setExpanded(True)
        self._suppress = False
        self.setCurrentItem(new_item)
        self.editItem(new_item, 0)
        return new_item

    def delete_field(self, item: QTreeWidgetItem | None = None) -> bool:
        """Remove ``item`` (or the current selection) and any descendants.

        Returns ``True`` when a row was removed. Top-level and nested
        items are both supported.
        """
        target = item or self.currentItem()
        if target is None:
            return False
        parent = target.parent()
        if parent is None:
            idx = self.indexOfTopLevelItem(target)
            if idx < 0:
                return False
            self.takeTopLevelItem(idx)
        else:
            parent.removeChild(target)
        self.model_changed.emit()
        return True

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _make_item(self, key: Any, val: Any) -> QTreeWidgetItem:
        """Build a tree item for ``(key, val)``, recursing into containers."""
        item = QTreeWidgetItem([str(key), value_to_text(val)])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        if isinstance(val, dict):
            for k, v in val.items():
                item.addChild(self._make_item(k, v))
        elif isinstance(val, list):
            for i, v in enumerate(val):
                item.addChild(self._make_item(f"[{i}]", v))
        return item

    def _dict_from_parent(self, parent) -> OrderedDict[str, Any]:
        """Recursively rebuild an OrderedDict from a parent's children."""
        out: OrderedDict[str, Any] = OrderedDict()
        for i in range(parent.childCount()):
            child = parent.child(i)
            key = child.text(0).strip()
            if not key:
                continue
            out[key] = self._value_from_item(child)
        return out

    def _value_from_item(self, item: QTreeWidgetItem) -> Any:
        """Return the Python value for ``item``.

        Has children → recurse: list when every child key is ``[N]``,
        dict otherwise. No children → parse the Value-cell text.
        """
        if item.childCount() == 0:
            return text_to_value(item.text(1))
        # Container: list vs dict?
        if all(
            _INDEX_KEY_RE.match(item.child(i).text(0).strip())
            for i in range(item.childCount())
        ):
            return [
                self._value_from_item(item.child(i))
                for i in range(item.childCount())
            ]
        return self._dict_from_parent(item)

    def _on_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        if self._suppress:
            return
        del item, col
        self.model_changed.emit()


__all__ = ["JsonTreeView", "LEVEL_ROLE", "value_to_text", "text_to_value"]
