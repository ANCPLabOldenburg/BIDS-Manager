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

from PyQt6.QtCore import QRect, QSize, Qt
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
        widget.setParent(self)
        if not hidden_on_purpose:
            widget.show()
        self._children.append(widget)
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

    def takeAt(self, index: int) -> Optional[QWidget]:  # noqa: N802
        """Remove and return a child, for a bar that is rebuilt in place."""
        if 0 <= index < len(self._children):
            widget = self._children.pop(index)
            widget.setParent(None)
            self.updateGeometry()
            return widget
        return None

    def clear(self) -> None:
        while self.takeAt(0) is not None:
            pass

    # -- sizing ------------------------------------------------------------

    def _visible(self) -> list[QWidget]:
        return [c for c in self._children if not c.isHidden()]

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(width, apply=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        """One row, everything on it. What the bar looks like with room."""
        left, top, right, bottom = self._margins
        width = height = 0
        for index, child in enumerate(self._visible()):
            hint = child.sizeHint()
            width += hint.width() + (self._h if index else 0)
            height = max(height, hint.height())
        return QSize(width + left + right, height + top + bottom)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """The WIDEST CHILD, not the sum. This is the point of the class."""
        left, top, right, bottom = self._margins
        size = QSize(0, 0)
        for child in self._visible():
            size = size.expandedTo(child.minimumSizeHint())
        return QSize(size.width() + left + right, size.height() + top + bottom)

    # -- the wrap ----------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._arrange(self.width(), apply=True)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._arrange(self.width(), apply=True)

    def _arrange(self, width: int, *, apply: bool) -> int:
        """Place the children across ``width``; return the height they need."""
        left, top, right, bottom = self._margins
        usable = max(width - left - right, 1)
        x = left
        y = top
        row_height = 0

        for child in self._visible():
            hint = child.sizeHint()
            if row_height and x - left + hint.width() > usable:
                x = left
                y += row_height + self._v
                row_height = 0
            if apply:
                child.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + self._h
            row_height = max(row_height, hint.height())

        return y + row_height + bottom


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
