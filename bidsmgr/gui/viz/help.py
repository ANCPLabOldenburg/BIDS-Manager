"""The shortcut window and the command palette, both GENERATED.

The old help popup was a hand-written table beside a hand-written list of
shortcuts, and the two disagreed (the 3-D docstrings named Shift+Y for a key
bound to Shift+Z). Here both read the live QActions and the live mouse map,
so the help shows exactly what the keys do, including the user's rebinding.
"""

from __future__ import annotations

from html import escape
from typing import Callable

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ...viz import actions as A
from ...viz import inputmap, keynames
from .bridge import ThemeHub

#: Plain clicks, which no mouse-map entry rebinds (a click is not a drag):
#: listed so the help names every gesture a canvas answers.
_CLICKS = {
    "render": (("Move the crosshair to the surface", "Click"),),
    "traces": (("Place the time cursor (click it again to remove it)", "Click"),
               ("Mark a channel bad, or good again", "Click its name")),
}


def shortcut_sections(manager, mousemap_overrides, kind: str = "volume",
                      canvases=None) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    """``[(title, [(what, [key, ...]), ...]), ...]``: the mouse gestures of
    each canvas this viewer has, then every bound action of this viewer kind
    by category. Read from the live mouse map and QActions, so a rebinding
    shows at once."""
    sections: list[tuple[str, list[tuple[str, list[str]]]]] = []
    table = dict(inputmap.DEFAULT_MOUSEMAP)
    table.update(mousemap_overrides or {})
    canvases = canvases if canvases is not None else getattr(manager, "canvases", ())
    for canvas, title, tools in (
        ("slice", "Mouse on a slice", inputmap.SLICE_TOOLS),
        ("render", "Mouse on the 3-D view", inputmap.RENDER_TOOLS),
        ("traces", "Mouse on the traces", inputmap.TRACES_TOOLS),
    ):
        if canvas not in canvases:
            continue
        rows: dict[str, list[str]] = {}
        for key, tool in table.items():
            if key.startswith(canvas + ":") and tool != "none":
                rows.setdefault(tools.get(tool, tool), []).append(
                    keynames.gesture(key.split(":", 1)[1]))
        for what, gesture in _CLICKS.get(canvas, ()):
            rows.setdefault(what, []).append(gesture)
        if rows:
            sections.append((title, list(rows.items())))
    by_category: dict[str, list[tuple[str, list[str]]]] = {}
    for d in A.ACTIONS:
        if kind not in A.kinds_of(d):
            continue
        keys = keynames.keys(manager.keys_for(d.id))
        if keys:
            by_category.setdefault(d.category, []).append((d.title, keys))
    sections.extend(by_category.items())
    return sections


def help_html(manager, mousemap_overrides, kind: str = "volume") -> str:
    """The same content as plain rich text (exports, tests)."""
    html = ["<div>"]
    for title, rows in shortcut_sections(manager, mousemap_overrides, kind):
        html.append(f"<h4>{escape(title)}</h4><table>")
        for what, keys in rows:
            html.append(f"<tr><td>{escape(what)}</td>"
                        f"<td><b>{escape(', '.join(keys))}</b></td></tr>")
        html.append("</table>")
    html.append("</div>")
    return "".join(html)


class _Card(QWidget):
    """One category: a title and its rows, PAINTED (a label per row cost
    about 5 ms each to polish, and the window has well over a hundred)."""

    PAD = 12
    ROW = 24
    TITLE = 22
    WIDTH = 320

    def __init__(self, title: str, rows: list[tuple[str, list[str]]], parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.title = title
        self.rows = rows
        self.shown_rows = rows
        #: Whether the filter keeps it (NOT ``isHidden``: a card not yet
        #: placed counts as hidden).
        self.matches = True
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def set_filter(self, needle: str) -> bool:
        """Keep the rows matching ``needle``; True when any remain."""
        if not needle or needle in self.title.lower():
            self.shown_rows = self.rows
        else:
            self.shown_rows = [(w, k) for w, k in self.rows
                               if needle in w.lower()
                               or any(needle in keynames.search_words(x) for x in k)]
        self.matches = bool(self.shown_rows)
        self.setFixedHeight(self._height())
        self.update()
        return self.matches

    def _fonts(self) -> tuple[QFont, QFont, QFont]:
        from . import fonts

        title = fonts.font(11, bold=True)
        title.setCapitalization(QFont.Capitalization.AllUppercase)
        title.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        row = fonts.font(12)
        key = fonts.font(11, bold=True)
        return title, row, key

    def _two_lines(self, what: str, keys: list[str]) -> bool:
        """Keys on a line of their own when, beside them, the text would be cut."""
        _t, row, key = self._fonts()
        caps = sum(QFontMetrics(key).horizontalAdvance(k) + 16 for k in keys)
        room = self.WIDTH - 2 * self.PAD - caps - 8
        return QFontMetrics(row).horizontalAdvance(what) > room

    def _height(self) -> int:
        lines = sum(2 if self._two_lines(w, k) else 1 for w, k in self.shown_rows)
        return self.PAD * 2 + self.TITLE + self.ROW * lines

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.WIDTH, self._height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        theme = ThemeHub.instance().theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.token("border", "#21262d")), 1))
        p.setBrush(QColor(theme.token("surface2", "#161b22")))
        p.drawRoundedRect(rect, 8, 8)
        title_font, row_font, key_font = self._fonts()
        p.setFont(title_font)
        p.setPen(QColor(theme.dim))
        p.drawText(QRectF(self.PAD, self.PAD, self.width() - 2 * self.PAD, self.TITLE),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self.title)
        key_fill = QColor(theme.token("surface3", "#1c2128"))
        key_line = QColor(theme.token("input_border", "#586069"))
        text = QColor(theme.text)
        fm_key = QFontMetrics(key_font)
        fm_row = QFontMetrics(row_font)
        y = self.PAD + self.TITLE
        for what, keys in self.shown_rows:
            two = self._two_lines(what, keys)
            if two:
                p.setFont(row_font)
                p.setPen(text)
                p.drawText(QRectF(self.PAD, y, self.width() - 2 * self.PAD, self.ROW),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           fm_row.elidedText(what, Qt.TextElideMode.ElideRight,
                                             self.width() - 2 * self.PAD))
                y += self.ROW
            # Keys from the right edge leftwards, as keycaps.
            x = self.width() - self.PAD
            p.setFont(key_font)
            for key in reversed(keys):
                w = fm_key.horizontalAdvance(key) + 12
                cap = QRectF(x - w, y + 3, w, self.ROW - 7)
                p.setPen(QPen(key_line, 1))
                p.setBrush(key_fill)
                p.drawRoundedRect(cap, 4, 4)
                p.setPen(text)
                p.drawText(cap, Qt.AlignmentFlag.AlignCenter, key)
                x -= w + 4
            if two:
                y += self.ROW
                continue
            p.setFont(row_font)
            p.setPen(text)
            room = int(x - self.PAD - 8)
            p.drawText(QRectF(self.PAD, y, max(room, 10), self.ROW),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       fm_row.elidedText(what, Qt.TextElideMode.ElideRight, max(room, 10)))
            y += self.ROW
        p.end()


class ShortcutsDialog(QDialog):
    """Every key and gesture of THIS viewer, by category, in balanced
    columns; type to filter. Non-modal: it can stay open beside the image."""


    def __init__(self, parent: QWidget, manager, mousemap_overrides, kind: str = "volume",
                 canvases=None) -> None:
        super().__init__(parent)
        self.setObjectName("viz-shortcuts")
        self.setWindowTitle("Keyboard and mouse")
        self.setModal(False)
        self._manager = manager
        self._columns_built = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a shortcut...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        top.addWidget(self.search, 1)
        self.edit_button = QPushButton("Change shortcuts...")
        self.edit_button.setObjectName("tb-btn")
        self.edit_button.clicked.connect(self._edit)
        top.addWidget(self.edit_button)
        lay.addLayout(top)
        hint = QLabel("Click an image once to focus the viewer; the keys then apply to it.")
        hint.setObjectName("sidecar-footer-summary")
        lay.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body.setObjectName("viz-shortcuts-body")
        self.columns = QHBoxLayout(self.body)
        self.columns.setContentsMargins(0, 0, 0, 0)
        self.columns.setSpacing(10)
        self.scroll.setWidget(self.body)
        lay.addWidget(self.scroll, 1)
        self.cards = [_Card(t, rows) for t, rows in
                      shortcut_sections(manager, mousemap_overrides, kind, canvases)]
        for card in self.cards:
            card.set_filter("")
        self.resize(3 * (_Card.WIDTH + 10) + 50, 640)
        self._arrange()

    def _column_count(self) -> int:
        # From the dialog's own width: the scroll area is not laid out yet
        # when the dialog is first shown. 28 margins, 16 for a scroll bar.
        return max(1, min(3, (self.width() - 28 - 16 + 10) // (_Card.WIDTH + 10)))

    def _arrange(self) -> None:
        """Deal the visible cards into columns, each to the shortest one."""
        n = self._column_count()
        while self.columns.count():
            item = self.columns.takeAt(0)
            if item is not None and item.layout() is not None:
                while item.layout().count():
                    item.layout().takeAt(0)
        cols = [QVBoxLayout() for _ in range(n)]
        heights = [0] * n
        for card in self.cards:
            card.setVisible(card.matches)
            if not card.matches:
                continue
            i = heights.index(min(heights))
            cols[i].addWidget(card)
            heights[i] += card.height() + 10
        for col in cols:
            col.setSpacing(10)
            col.addStretch(1)
            self.columns.addLayout(col)
        self.columns.addStretch(1)
        self._columns_built = n

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._column_count() != self._columns_built:
            self._arrange()

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for card in self.cards:
            card.set_filter(needle)
        self._arrange()

    def visible_titles(self) -> list[str]:
        return [c.title for c in self.cards if c.matches]

    def _edit(self) -> None:
        edit_shortcuts(self)


def edit_shortcuts(parent: QWidget) -> bool:
    """The library's own shortcut editor in a dialog; Save applies it to
    every open viewer. True when saved."""
    from PyQt6.QtWidgets import QDialogButtonBox

    from .bridge import SettingsHub
    from .settings_pages import ShortcutsPage

    hub = SettingsHub.instance()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Change shortcuts")
    dlg.resize(760, 640)
    lay = QVBoxLayout(dlg)
    page = ShortcutsPage()
    page.load(hub.settings)
    lay.addWidget(page, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False
    hub.update(page.apply_to)
    return True


def show_help(parent: QWidget, manager, mousemap_overrides, kind: str = "volume",
              canvases=None) -> ShortcutsDialog:
    """Open (or raise) the viewer's shortcut window."""
    existing = getattr(parent, "_shortcuts_dialog", None)
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:  # the C++ side is gone
            pass
    dlg = ShortcutsDialog(parent, manager, mousemap_overrides, kind, canvases)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    parent._shortcuts_dialog = dlg
    dlg.finished.connect(lambda _r, w=parent: setattr(w, "_shortcuts_dialog", None))
    dlg.show()
    return dlg


class CommandPalette(QDialog):
    """Type to find any action of the viewer, Enter to run it."""

    def __init__(self, parent: QWidget, manager, on_run: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find an action")
        self.setModal(True)
        self.resize(460, 380)
        self._manager = manager
        self._on_run = on_run
        lay = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Type an action...")
        self.search.textChanged.connect(self._fill)
        self.search.returnPressed.connect(self._run_current)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda item: self._run(item))
        lay.addWidget(self.list, 1)
        self._fill("")

    def _fill(self, text: str) -> None:
        self.list.clear()
        needle = text.strip().lower()
        ctx = self._manager.context()
        for d in A.ACTIONS:
            if needle and needle not in d.title.lower() and needle not in d.category.lower():
                continue
            if not A.evaluate(d.when, ctx):
                continue
            keys = ", ".join(self._manager.key_labels(d.id))
            item = QListWidgetItem(f"{d.title}" + (f"    {keys}" if keys else ""))
            item.setData(Qt.ItemDataRole.UserRole, d.id)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _run_current(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self._run(item)

    def _run(self, item) -> None:
        action_id = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        self._on_run(action_id)


__all__ = ["CommandPalette", "ShortcutsDialog", "edit_shortcuts", "help_html",
           "shortcut_sections", "show_help"]
