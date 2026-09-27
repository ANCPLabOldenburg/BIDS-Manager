"""One click on a folder row opens or closes it, arrow or no arrow.

Qt gives the expander triangle its own hit area. Clicking a folder's NAME
selects it and does nothing else; only the small triangle beside it opens the
folder. So the row has two halves that do different things, and which one you
get depends on hitting a target a few pixels wide.

That is what made the trees feel inconsistent: select a file, then go to open
a nearby folder, miss the triangle, and the click lands as a selection change
instead. Nothing appears to happen, and the folder you were aiming at is now
the selected row.

One click on the row now toggles it, which is what the triangle already did,
so the two halves agree. Three rules keep it from taking anything away:

* **Only rows that can fold.** A file has nothing to open, so a click on one
  is a plain selection, exactly as before.
* **Only an unmodified click.** Ctrl, Shift and Cmd are how a multi-selection
  is built, and folding a folder in the middle of building one would fight
  the user.
* **Selection still happens.** This runs on ``itemClicked``, which Qt emits
  after the selection has moved, so clicking a folder selects it AND folds
  it. The Editor's "Validate folder" still has something to act on.

A double click is ONE toggle, not two and not none. Qt is supposed to
suppress ``clicked`` on the release that completes a double click, which
would make this come out right on its own, but whether it does depends on
whether the platform plugin synthesised a double click at all: under the
offscreen plugin the same gesture arrives as two ordinary clicks, opens the
folder and closes it again. Rather than rely on that, a second click on the
same row inside the system double-click interval is ignored. A deliberate,
slower second click still folds the row back.
"""

from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, QObject, Qt
from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

#: Held down, these mean "extend the selection", not "open this folder".
_SELECTION_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.MetaModifier
)


class _ClickToggle(QObject):
    """Folds the clicked row, once per gesture.

    Parented to the tree, so its lifetime is the tree's and there is nothing
    for the caller to hold on to.
    """

    def __init__(self, tree: QTreeWidget) -> None:
        super().__init__(tree)
        self._last_item: object = None
        self._since = QElapsedTimer()
        tree.itemClicked.connect(self.on_click)

    def on_click(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        # ``childCount`` rather than "is this a directory": it is precisely
        # the rows that HAVE an expander, so a folder-recording (a CTF
        # ``.ds``, an EGI ``.mff``) is a directory on disk that the tree
        # draws as a recording, and clicking it stays a plain selection.
        if item is None or item.childCount() == 0:
            return
        if QApplication.keyboardModifiers() & _SELECTION_MODIFIERS:
            return
        if (
            item is self._last_item
            and self._since.isValid()
            and self._since.elapsed() < QApplication.doubleClickInterval()
        ):
            # The second half of a double click. One gesture, one toggle.
            self._since.invalidate()
            return
        self._last_item = item
        self._since.start()
        item.setExpanded(not item.isExpanded())


def toggle_on_click(tree: QTreeWidget) -> None:
    """Wire ``tree`` so a click anywhere on a foldable row folds it."""
    tree.setExpandsOnDoubleClick(False)
    _ClickToggle(tree)


__all__ = ["toggle_on_click"]
