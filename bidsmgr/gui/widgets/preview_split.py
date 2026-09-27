"""Controls and preview, either stacked or side by side, at the user's choice.

Every tool that changes files shows the same two things: the controls that
say WHAT to do, and a preview that says what that would do to which files.
They were all laid out the same way, controls above and preview below, and
that is the right default: you read the controls, then read down into the
consequences.

It is the wrong shape for two common cases, though.

**A wide window.** A preview four rows tall under a full-width dialog wastes
the width and hides the rows that did not fit. Beside the controls it gets
the whole height.

**A long list of consequences.** Renaming a subject moves fifty files; the
preview wants vertical room and the controls do not.

So the split is a ``QSplitter`` whose orientation can be flipped, with one
button to flip it, and the choice is remembered. It is remembered PER TOOL,
keyed by name: somebody who wants the rename preview beside the controls
does not necessarily want the delete confirmation there too, and a shared
setting would make one of those choices for them.

The drag position is not remembered on purpose. It means different things in
the two orientations, and restoring a horizontal split into a vertical one
produces a preview two pixels tall, which reads as a bug.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

#: Where the per-tool choice is kept.
_KEY = "dialogs/preview_orientation/{0}"

_BESIDE = "beside"
_BELOW = "below"


class PreviewSplit(QSplitter):
    """A two-pane splitter that can be flipped between stacked and side by side.

    ``controls`` is the pane that says what to do, ``preview`` the one that
    says what it would do. ``name`` keys the remembered choice.
    """

    #: Emitted whenever the orientation changes, however it was changed.
    #: The button wording follows this rather than only its own clicks, so
    #: a flip made any other way cannot leave the label saying the opposite
    #: of what pressing it would do.
    flipped = pyqtSignal(bool)

    def __init__(
        self,
        controls: QWidget,
        preview: QWidget,
        *,
        name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self.addWidget(controls)
        self.addWidget(preview)
        # The preview takes the slack: it is the part whose useful size
        # depends on how many files there turn out to be.
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setChildrenCollapsible(False)
        self._apply(_read(name))

    # -- the flip ----------------------------------------------------------

    def is_beside(self) -> bool:
        return self.orientation() == Qt.Orientation.Horizontal

    def toggle(self) -> None:
        self.set_beside(not self.is_beside())

    def set_beside(self, beside: bool, *, remember: bool = True) -> None:
        self._apply(_BESIDE if beside else _BELOW)
        if remember:
            QSettings().setValue(
                _KEY.format(self._name), _BESIDE if beside else _BELOW,
            )

    def _apply(self, where: str) -> None:
        beside = where == _BESIDE
        self.setOrientation(
            Qt.Orientation.Horizontal if beside else Qt.Orientation.Vertical
        )
        # Start each orientation from an even split rather than restoring a
        # dragged position: the number means different things along the two
        # axes, and a horizontal position restored vertically gives a
        # preview two pixels tall.
        half = max(self.width() if beside else self.height(), 200) // 2
        self.setSizes([half, half])
        self.flipped.emit(beside)


def preview_toggle(split: PreviewSplit) -> QPushButton:
    """The button that flips ``split``, wording itself for what it will do."""
    button = QPushButton()
    button.setObjectName("tb-btn")

    def refresh() -> None:
        beside = split.is_beside()
        button.setText("Preview below" if beside else "Preview at the side")
        button.setToolTip(
            "Put the preview under the controls, which suits a long list "
            "of files."
            if beside else
            "Put the preview beside the controls, which suits a wide "
            "window. Remembered for this tool."
        )

    def clicked() -> None:
        split.toggle()

    button.clicked.connect(clicked)
    split.flipped.connect(lambda _b: refresh())
    refresh()
    return button


def controls_panel(*items) -> QWidget:
    """Pack a dialog's controls into one pane for the splitter.

    Takes widgets and layouts in the order they should appear. The pane
    does not stretch vertically: in the stacked orientation the preview
    should get the slack, and beside the controls it should get the height.
    """
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        else:
            layout.addLayout(item)
    layout.addStretch(1)
    panel.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred,
    )
    return panel


def _read(name: str) -> str:
    stored = QSettings().value(_KEY.format(name), _BELOW)
    return _BESIDE if str(stored) == _BESIDE else _BELOW


__all__ = ["PreviewSplit", "controls_panel", "preview_toggle"]
