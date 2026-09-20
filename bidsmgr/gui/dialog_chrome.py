"""The frame every BIDS Manager dialog wears.

The Issues dialog established the shape long before the Editor grew dialogs of
its own: a tinted header carrying a title and one sentence of orientation, a
plain body that scrolls, and a tinted footer holding the buttons. Three
object names in ``theme.qss`` do the actual painting, which is why a dialog
that sets them tracks a theme swap for free and one that does not looks like it
came from a different program.

The new Editor dialogs were built without it and looked exactly that generic.
Rather than copy the same twenty lines into each, they are here once, so
there is one place to change the chrome and no way to half-apply it.

Nothing here decides content. It builds the frame and hands back the body
layout to fill.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def build_header(title: str, subtitle: str = "") -> QFrame:
    """The tinted strip at the top: what this dialog is, and what it does."""
    header = QFrame()
    header.setObjectName("issue-dialog-header")
    layout = QVBoxLayout(header)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(3)

    label = QLabel(title)
    label.setObjectName("issue-dialog-title")
    label.setWordWrap(True)
    layout.addWidget(label)

    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("issue-dialog-subtitle")
        sub.setWordWrap(True)
        sub.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(sub)
    return header


def build_footer(buttons: QDialogButtonBox) -> QFrame:
    """The tinted strip at the bottom, holding the decision."""
    footer = QFrame()
    footer.setObjectName("issue-dialog-footer")
    layout = QHBoxLayout(footer)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(8)
    layout.addStretch(1)
    layout.addWidget(buttons)
    return footer


def build_footer_with(
    left: Optional[QWidget], buttons: QDialogButtonBox,
) -> QFrame:
    """A footer with something on the left, usually a status line."""
    footer = build_footer(buttons)
    if left is not None:
        footer.layout().insertWidget(0, left)
    return footer


def scrollable_body() -> tuple[QScrollArea, QVBoxLayout]:
    """A body that scrolls, painted like the dialog rather than like a list.

    A bare ``QScrollArea`` paints itself with the base colour and its viewport
    with another, which is how a themed dialog ends up with a white rectangle
    in the middle of it. Naming both makes the QSS reach them.
    """
    scroll = QScrollArea()
    scroll.setObjectName("issue-dialog-scroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    body = QWidget()
    body.setObjectName("issue-dialog-body")
    body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QVBoxLayout(body)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(10)
    scroll.setWidget(body)
    return scroll, layout


def card(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    """One bordered block inside a body, matching the Issues dialog's cards."""
    frame = QFrame()
    frame.setObjectName("issue-card")
    frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    if title:
        label = QLabel(title)
        label.setObjectName("dlg-section-title")
        layout.addWidget(label)
    return frame, layout


class WrapLabel(QLabel):
    """A word-wrapped label that takes the height its text needs, and no more.

    Qt asks a wrapped label for a height before it knows the width, so the
    label answers with one line and a vertical layout clips the rest. The
    usual workaround is a ``MinimumExpanding`` policy, which fixes the
    clipping and introduces the opposite problem: the label then soaks up
    every spare pixel in the dialog and the sections drift apart. Answering
    with ``heightForWidth`` at the width the label actually has gives the
    right number in both directions.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)

    def _height(self) -> int:
        width = self.width()
        if width <= 0:
            # Before the first layout pass. NOT ``super().sizeHint()``: for a
            # wrapped label Qt answers that with a squarish guess several
            # lines tall, which is where the gaps came from. The height at
            # the label's own natural width is the honest answer.
            width = super().sizeHint().width()
        height = self.heightForWidth(max(width, 1))
        if height < 0:
            # Qt returns -1 for an empty label, and a size hint carrying -1 is
            # invalid, so the layout falls back to a default several times too
            # tall. An empty label is one line high.
            return self.fontMetrics().height()
        return height

    def sizeHint(self):                       # noqa: N802 - Qt signature
        hint = super().sizeHint()
        hint.setHeight(self._height())
        return hint

    def minimumSizeHint(self):                # noqa: N802 - Qt signature
        hint = super().minimumSizeHint()
        hint.setHeight(self._height())
        return hint

    def resizeEvent(self, event):             # noqa: N802 - Qt signature
        super().resizeEvent(event)
        # A new width means a new height. Without this the label keeps the
        # height it was given at the old width and either clips or gaps.
        self.updateGeometry()


def hint(text: str) -> QLabel:
    """The quiet explanatory line a dialog uses under a heading or in a footer.

    Deliberately NOT ``#pane-hint``: that one carries 24px of padding for the
    empty-pane placeholder it was written for, and a dialog full of them reads
    as a set of unrelated blocks floating apart.
    """
    label = WrapLabel(text)
    label.setObjectName("dlg-hint")
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


__all__ = [
    "WrapLabel",
    "build_footer",
    "build_footer_with",
    "build_header",
    "card",
    "hint",
    "scrollable_body",
]
