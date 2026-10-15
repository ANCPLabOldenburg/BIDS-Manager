"""A bar whose controls wrap onto another row instead of forcing a width.

A ``QHBoxLayout`` reports a minimum width equal to the SUM of what its
children need, so a bar of eight controls cannot be made narrower than all
eight of them laid end to end. In a pane the user is meant to be able to
drag narrow, that sum is the floor: the Editor's sidecar could not go below
617 pixels and the content it was there to show could have gone to 62.

This wraps instead. The minimum width becomes the WIDEST SINGLE control
rather than the sum of all of them, and the rows reflow as the pane is
dragged, so nothing is clipped and nothing goes out of reach.

**It is a widget that positions its children, NOT a QLayout subclass, and
that is the whole design decision.** The obvious implementation is to
subclass ``QLayout`` and override ``addItem`` / ``itemAt`` / ``takeAt``,
which is what Qt's own flow-layout example does in C++. In PyQt it puts
``QLayoutItem`` ownership in question: ``QLayout::addWidget`` creates the
item in C++ and hands it to the override, which can only keep it in a
Python list, and at teardown both sides believe they own it. Measured: the
QLayout version turned an occasional pytest-qt teardown flake into three
test files segfaulting on every run, with no traceback, while every test in
them passed. Swapping the layout back for a ``QHBoxLayout`` made the crash
go away, which is how it was pinned down.

A plain widget has no such question. Its children are ordinary child
widgets, owned by Qt exactly as they always are, and the wrapping is
arithmetic in ``resizeEvent``.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QSizePolicy, QVBoxLayout, QWidget


class FlowBar(QWidget):
    """Lay children left to right, wrapping onto a new row when they run out.

    The API deliberately mirrors the ``QHBoxLayout`` calls the bars it
    replaces were already making, so converting one is a change of
    construction and nothing else.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        h_spacing: int = 8,
        v_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._children: list[QWidget] = []
        self._h = h_spacing
        self._v = v_spacing
        self._margins = (0, 0, 0, 0)
        policy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )
        # Without this the parent never asks how tall the wrapped rows came
        # out and every row after the first is clipped.
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    # -- building ----------------------------------------------------------

    def setContentsMargins(  # noqa: N802 - Qt naming
        self, left: int, top: int, right: int, bottom: int,
    ) -> None:
        self._margins = (left, top, right, bottom)
        self.updateGeometry()

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802
        self.insertWidget(len(self._children), widget)

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802
        """``widget`` at position ``index`` (``QBoxLayout.insertWidget``)."""
        # A widget the caller has already hidden ON PURPOSE stays hidden.
        # Reparenting hides a widget, so it has to be shown again, and
        # showing it unconditionally revealed the chips and buttons that
        # are built hidden and revealed later (the unsaved-changes chip,
        # the tree-only field verbs).
        hidden_on_purpose = (
            widget.isHidden()
            and widget.testAttribute(
                Qt.WidgetAttribute.WA_WState_ExplicitShowHide
            )
        )
        if widget in self._children:
            self._children.remove(widget)
        widget.setParent(self)
        if not hidden_on_purpose:
            widget.show()
        index = len(self._children) if index < 0 else min(index, len(self._children))
        self._children.insert(index, widget)
        self._relayout()

    def _relayout(self) -> None:
        """Place the children again now (a child added to a bar already on
        screen otherwise sat at the origin, over the first one, until
        something else asked for a layout: two checkboxes painted over each
        other)."""
        if self.isVisible():
            height = self._arrange(self.width(), apply=True)
            self._rows_height = height
        self.updateGeometry()

    def addLayout(self, layout: QLayout) -> None:  # noqa: N802
        """Accept a sub-layout by giving it a widget to live in.

        The bars being converted use it for grouped controls (a slider with
        its caption). Wrapping keeps the group together as ONE item, which
        is also what you want when it wraps: a caption should not end up on
        a different row from its slider.
        """
        holder = QWidget()
        holder.setLayout(layout)
        self.addWidget(holder)

    def addSpacing(self, size: int) -> None:  # noqa: N802
        """A fixed gap, which survives wrapping as an ordinary child."""
        spacer = QWidget()
        spacer.setFixedWidth(size)
        spacer.setFixedHeight(1)
        self.addWidget(spacer)

    def addStretch(self, stretch: int = 0) -> None:  # noqa: N802
        """Deliberately nothing.

        A stretch pushes what follows against the right-hand edge, and a
        wrapping row has no fixed right-hand edge: where it is depends on
        how many rows there turn out to be. Accepted rather than raising so
        a bar converts without rewriting every call, and it does nothing
        rather than inserting a gap, which is what "push to the edge"
        degrades to when there is no edge.
        """
        del stretch

    def count(self) -> int:
        return len(self._children)

    def indexOf(self, widget: QWidget) -> int:  # noqa: N802
        return self._children.index(widget) if widget in self._children else -1

    def widgets(self) -> list[QWidget]:
        """The children, in order (hidden ones too)."""
        return list(self._children)

    def takeAt(self, index: int) -> Optional[QWidget]:  # noqa: N802
        """Remove and return a child, for a bar that is rebuilt in place."""
        if 0 <= index < len(self._children):
            widget = self._children.pop(index)
            widget.setParent(None)
            self._relayout()
            return widget
        return None

    def removeWidget(self, widget: QWidget) -> None:  # noqa: N802
        """Take ``widget`` out of the bar (it is not deleted)."""
        if widget in self._children:
            self.takeAt(self._children.index(widget))

    def clear(self) -> None:
        while self.takeAt(0) is not None:
            pass

    # -- sizing ------------------------------------------------------------

    def _visible(self) -> list[QWidget]:
        return [c for c in self._children if not c.isHidden()]

    @staticmethod
    def _size(child: QWidget) -> QSize:
        """The size a child is given: its hint within its own minimum and
        maximum, as a Qt layout does. A fixed-size widget's hint does not
        report its fixed size, so a 28 px icon button with an 18 px hint was
        placed in an 18 px row, clipped, and overlapped its neighbour."""
        return child.sizeHint().expandedTo(child.minimumSize()).boundedTo(child.maximumSize())

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(width, apply=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        """One row, everything on it. What the bar looks like with room."""
        left, top, right, bottom = self._margins
        width = height = 0
        for index, child in enumerate(self._visible()):
            hint = self._size(child)
            width += hint.width() + (self._h if index else 0)
            height = max(height, hint.height())
        return QSize(width + left + right, height + top + bottom)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """The WIDEST CHILD, not the sum. This is the point of the class.

        The height is what the rows need at the CURRENT width. A splitter
        does not ask ``heightForWidth``; told one row, it squeezed a bar that
        had wrapped onto three, and the rows below the first were painted
        over the plot under them."""
        left, top, right, bottom = self._margins
        size = QSize(0, 0)
        for child in self._visible():
            size = size.expandedTo(child.minimumSizeHint().expandedTo(child.minimumSize()))
        height = size.height() + top + bottom
        if self.width() > 0:
            height = max(height, self._arrange(self.width(), apply=False))
        return QSize(size.width() + left + right, height)

    # -- the wrap ----------------------------------------------------------

    _rows_height = -1

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        height = self._arrange(self.width(), apply=True)
        if height != self._rows_height:
            # More (or fewer) rows: the parents must ask again how tall the
            # bar has to be. Only on a change, so a resize cannot loop.
            self._rows_height = height
            self.updateGeometry()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._arrange(self.width(), apply=True)

    def event(self, event) -> bool:  # noqa: D401
        # A child that changed what it needs (new text, shown, hidden) asks
        # its parent to lay it out again. Ignored, a button whose label grew
        # was cut off, and a checkbox shown later was painted at the origin,
        # over the first control of the row.
        if event.type() == QEvent.Type.LayoutRequest:
            height = self._arrange(self.width(), apply=True)
            if height != self._rows_height:
                self._rows_height = height
            self.updateGeometry()
        elif event.type() == QEvent.Type.ChildRemoved:
            # A child reparented or deleted elsewhere leaves the bar: kept in
            # the list, it held its place (or pointed at a deleted widget).
            child = event.child()
            if child in self._children:
                self._children.remove(child)
                self._relayout()
        return super().event(event)

    def _arrange(self, width: int, *, apply: bool) -> int:
        """Place the children across ``width``; return the height they need.

        Each child is centred in its row's height: a label beside a combo
        box, or a plain button beside a taller menu button, otherwise sits
        at the top of the row and reads as misaligned."""
        left, top, right, bottom = self._margins
        usable = max(width - left - right, 1)
        rows: list[list] = [[]]
        x = left
        for child in self._visible():
            hint = self._size(child)
            if rows[-1] and x - left + hint.width() > usable:
                rows.append([])
                x = left
            rows[-1].append((child, x, hint))
            x += hint.width() + self._h
        y = top
        for row in rows:
            row_height = max((hint.height() for _c, _x, hint in row), default=0)
            if apply:
                for child, cx, hint in row:
                    dy = (row_height - hint.height()) // 2
                    child.setGeometry(QRect(cx, y + dy, hint.width(), hint.height()))
            y += row_height + self._v
        return y - self._v + bottom if rows[-1] else top + bottom


def flow(parent: QWidget, **kwargs) -> FlowBar:
    """A :class:`FlowBar` filling ``parent``, returned for the caller to fill.

    ``parent`` keeps whatever frame and object name it has (the bars being
    converted are styled ``QFrame#toolbar`` and ``QFrame#sidecar-toolbar``),
    and the bar sits inside it as its only child, so the QSS still applies.
    """
    bar = FlowBar(parent, **kwargs)
    holder = QVBoxLayout(parent)
    holder.setContentsMargins(0, 0, 0, 0)
    holder.setSpacing(0)
    holder.addWidget(bar)
    return bar


__all__ = ["FlowBar", "flow"]
