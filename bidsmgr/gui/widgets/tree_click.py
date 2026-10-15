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
* **Not on a checkbox.** Qt reports a click on a row's check indicator as a
  click on the row, so ticking a subject in the Filter tree would fold it
  too. A press that lands on the indicator ticks and nothing else.

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

from PyQt6.QtCore import QElapsedTimer, QEvent, QObject, QPoint, Qt
from PyQt6.QtWidgets import QApplication, QStyle, QTreeWidget, QTreeWidgetItem

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
        self._tree = tree
        self._last_item: object = None
        self._since = QElapsedTimer()
        self._on_check = False
        tree.viewport().installEventFilter(self)
        tree.itemClicked.connect(self.on_click)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt signature
        """Note whether the press landed on a check indicator. Never eats it."""
        if event.type() == QEvent.Type.MouseButtonPress:
            self._on_check = self._hits_check(event.position().toPoint())
        return False

    def _hits_check(self, pos: QPoint) -> bool:
        """Is ``pos`` on the check indicator of a checkable row?

        The indicator is drawn first in the row, so the test is how far into
        the row the press is, measured with the style's own indicator size.
        """
        tree = self._tree
        index = tree.indexAt(pos)
        if not index.isValid() or not (index.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return False
        style = tree.style()
        reach = (style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth)
                 + 2 * style.pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin) + 4)
        return 0 <= pos.x() - tree.visualRect(index).left() <= reach

    def on_click(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        # ``childCount`` rather than "is this a directory": it is precisely
        # the rows that HAVE an expander, so a folder-recording (a CTF
        # ``.ds``, an EGI ``.mff``) is a directory on disk that the tree
        # draws as a recording, and clicking it stays a plain selection.
        if item is None or item.childCount() == 0:
            return
        if self._on_check:
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
