"""How a trace is drawn: its width, and what colour each channel type takes.

Small enough to be a popup rather than a settings tab, and shared by every
viewer that draws a line, so the MEG/EEG, physio and spectroscopy views
cannot end up offering three different versions of the same controls.

Width
-----
Automatic by default, which is not a dodge: a pen wider than one pixel is
stroked properly by Qt instead of through its cosmetic fast path, and on a
twenty-channel MEG window that is 119 ms of paint against 16.5. So a physio
channel draws at two pixels and a wall of MEG at one. Anything picked here is
honoured whatever it costs, because at that point the cost is a choice.

Colour
------
**By channel type is the default, and the types are configurable.** Colouring
by type is what makes a multi-channel view readable: magnetometers one colour
and gradiometers another is the difference between three hundred traces and
three hundred traces you can tell apart. Each viewer lists the types IT has,
so a MEG recording offers mag, grad and ref_meg while a physio run offers
cardiac, respiratory and trigger, and no viewer asks about a type it cannot
show.

The shipped scheme is palette TOKENS rather than literal colours, so it
follows the theme and stays legible in both. **Reset defaults puts every type
back** by deleting the overrides rather than writing a second table, which is
why it cannot drift out of step with the shipped one.

**One colour** is also offered, and is the right answer for a single-channel
view where the type is already in the title and a second colour says nothing.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import CUR
from .psd_dialog import default_type_color, type_color, type_colors

#: What the width slider allows. Above about eight a trace stops being a
#: line and starts hiding the one underneath it.
MIN_WIDTH = 1
MAX_WIDTH = 8

#: The slider's left-hand position, which is not a width at all.
AUTO_WIDTH = 0


def _ink_for(colour: str) -> str:
    """A label colour readable on *colour*, whichever was picked."""
    return "#0a0e13" if QColor(colour).lightness() > 140 else "#e6edf3"


class LineStyleDialog(QDialog):
    """Pick a trace width and the colour of each channel type.

    Emits as the user moves, not on OK, because a line width and a colour
    are things you judge by looking at the plot behind the dialog, and a
    preview swatch is not the plot.
    """

    changed = pyqtSignal(int, object)      # width, colour ("" -> by type)
    type_colors_changed = pyqtSignal(dict)  # channel type -> hex ({} -> shipped)

    def __init__(
        self,
        width: int,
        colour: Optional[str],
        *,
        allow_by_type: bool = True,
        channel_types: Sequence[str] = (),
        max_width: int = MAX_WIDTH,
        traces_shown: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Line")
        self.setObjectName("pane-dark")
        self._width = int(width)
        self._colour = colour
        self._max_width = max(1, int(max_width))
        self._traces_shown = int(traces_shown)
        # Order preserved, duplicates dropped: the list should read in the
        # order the viewer draws them, not alphabetically.
        self._types: list[str] = list(dict.fromkeys(str(t) for t in channel_types))
        self._overrides: dict[str, str] = type_colors()
        self._swatches: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        # -- width ---------------------------------------------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Thickness"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(AUTO_WIDTH, self._max_width)
        self._slider.setValue(max(AUTO_WIDTH, min(self._max_width, self._width)))
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setToolTip(
            "Leftmost is automatic: two pixels where one or two traces are "
            "on screen, one where there are more. A wider pen cannot use "
            "Qt's fast path, and on a twenty-channel MEG window that is "
            "88 ms of paint against 13, which a drag pays on every frame."
        )
        self._slider.valueChanged.connect(self._on_width)
        row.addWidget(self._slider, 1)
        self._width_label = QLabel("")
        self._width_label.setMinimumWidth(64)
        row.addWidget(self._width_label)
        outer.addLayout(row)
        self._paint_width_label()

        # A view drawing many traces CANNOT go thicker, so the slider says so
        # rather than offering a number that would be capped on the way out.
        # Saying which view and what to do about it, because "disabled" with
        # no reason is the thing people file bugs about.
        if self._max_width <= 1:
            self._slider.setEnabled(False)
            note = QLabel(
                f"One pixel, because this view draws {self._traces_shown} "
                f"traces and a wider pen costs about seven times the paint. "
                f"Show one or two channels to draw thicker."
                if self._traces_shown else
                "One pixel: this view draws too many traces for a wider pen."
            )
            note.setObjectName("pane-hint")
            note.setWordWrap(True)
            outer.addWidget(note)

        # -- colour mode ---------------------------------------------------
        self._by_type = QRadioButton("Colour by channel type")
        self._by_type.setToolTip(
            "What makes a multi-channel view readable: magnetometers one "
            "colour and gradiometers another, a cardiac trace and a trigger "
            "told apart without reading the labels."
        )
        self._one_colour = QRadioButton("One colour")
        self._one_colour.setToolTip(
            "Right for a single-channel view, where the type is already in "
            "the title and a second colour says nothing."
        )
        if not allow_by_type:
            self._by_type.setEnabled(False)
            self._by_type.setToolTip(
                "This view draws one channel, so there are no types to "
                "tell apart."
            )
        (self._one_colour if colour or not allow_by_type
         else self._by_type).setChecked(True)
        self._by_type.toggled.connect(self._on_mode)
        outer.addWidget(self._by_type)

        # -- per-type colours ----------------------------------------------
        if self._types:
            self._type_box = QFrame()
            self._type_box.setObjectName("meta-row")
            grid = QGridLayout(self._type_box)
            grid.setContentsMargins(20, 2, 0, 6)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            for i, ch_type in enumerate(self._types):
                label = QLabel(ch_type)
                grid.addWidget(label, i // 2, (i % 2) * 2)
                swatch = QPushButton()
                swatch.setObjectName("tb-btn")
                swatch.setMinimumWidth(96)
                swatch.setToolTip(
                    f"The colour every {ch_type} channel is drawn in, here "
                    f"and in this viewer's spectrum."
                )
                swatch.clicked.connect(
                    lambda _checked=False, t=ch_type: self._pick_type_colour(t)
                )
                grid.addWidget(swatch, i // 2, (i % 2) * 2 + 1)
                self._swatches[ch_type] = swatch
            outer.addWidget(self._type_box)

            reset_row = QHBoxLayout()
            reset_row.setContentsMargins(20, 0, 0, 0)
            self._reset_btn = QPushButton("Reset defaults")
            self._reset_btn.setObjectName("tb-btn")
            self._reset_btn.setToolTip(
                "Put every channel type back to the colour it ships with. "
                "The shipped scheme is theme-aware, so this also undoes a "
                "colour that turned out to be unreadable in one theme."
            )
            self._reset_btn.clicked.connect(self._reset_type_colours)
            reset_row.addWidget(self._reset_btn)
            reset_row.addStretch(1)
            outer.addLayout(reset_row)
            self._paint_type_swatches()
        else:
            self._type_box = None
            self._reset_btn = None

        # -- one colour ----------------------------------------------------
        pick_row = QHBoxLayout()
        pick_row.addWidget(self._one_colour)
        self._swatch = QPushButton("Choose…")
        self._swatch.setObjectName("tb-btn")
        self._swatch.clicked.connect(self._pick_colour)
        pick_row.addWidget(self._swatch)
        pick_row.addStretch(1)
        outer.addLayout(pick_row)
        self._paint_swatch()
        self._sync_type_box()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.clicked.connect(lambda _b: self.accept())
        outer.addWidget(buttons)

    # -- width ------------------------------------------------------------

    def _emit(self) -> None:
        self.changed.emit(self._width, self._colour or "")

    def _paint_width_label(self) -> None:
        self._width_label.setText(
            "automatic" if self._width <= AUTO_WIDTH else f"{self._width} px"
        )

    def _on_width(self, value: int) -> None:
        self._width = int(value)
        self._paint_width_label()
        self._emit()

    # -- colour mode ------------------------------------------------------

    def _on_mode(self, by_type: bool) -> None:
        if by_type:
            self._colour = None
        elif not self._colour:
            self._colour = CUR().get("accent", "#58a6ff")
        self._paint_swatch()
        self._sync_type_box()
        self._emit()

    def _sync_type_box(self) -> None:
        """Per-type colours mean nothing while one colour is in force."""
        by_type = self._by_type.isChecked()
        for widget in (self._type_box, self._reset_btn):
            if widget is not None:
                widget.setEnabled(by_type)

    def _pick_colour(self) -> None:
        start = QColor(self._colour or CUR().get("accent", "#58a6ff"))
        chosen = QColorDialog.getColor(start, self, "Trace colour")
        if not chosen.isValid():
            return
        self._colour = chosen.name()
        self._one_colour.setChecked(True)
        self._paint_swatch()
        self._sync_type_box()
        self._emit()

    def _paint_swatch(self) -> None:
        by_type = self._by_type.isChecked()
        self._swatch.setEnabled(not by_type)
        if by_type or not self._colour:
            self._swatch.setText("Choose…")
            self._swatch.setStyleSheet("")
            return
        self._swatch.setText(self._colour)
        self._swatch.setStyleSheet(
            f"QPushButton {{ background: {self._colour}; "
            f"color: {_ink_for(self._colour)}; }}"
        )

    # -- per-type colours -------------------------------------------------

    def _pick_type_colour(self, ch_type: str) -> None:
        start = QColor(self._overrides.get(ch_type) or type_color(ch_type))
        chosen = QColorDialog.getColor(start, self, f"Colour for {ch_type}")
        if not chosen.isValid():
            return
        self._overrides[ch_type] = chosen.name()
        self._by_type.setChecked(True)
        self._paint_type_swatches()
        self.type_colors_changed.emit(dict(self._overrides))

    def _reset_type_colours(self) -> None:
        """Delete the overrides. Every type falls back to what it ships with.

        A deletion rather than a rewrite: the shipped colours live in exactly
        one place (``psd_dialog.TYPE_TOKENS``) and copying them here to write
        back would be a second table to keep in step.
        """
        self._overrides = {}
        self._paint_type_swatches()
        self.type_colors_changed.emit({})

    def _paint_type_swatches(self) -> None:
        for ch_type, swatch in self._swatches.items():
            chosen = self._overrides.get(ch_type)
            colour = chosen or default_type_color(ch_type)
            swatch.setText(colour if chosen else f"{colour} (default)")
            swatch.setStyleSheet(
                f"QPushButton {{ background: {colour}; "
                f"color: {_ink_for(colour)}; }}"
            )

    # -- state ------------------------------------------------------------

    def values(self) -> tuple[int, str]:
        return self._width, (self._colour or "")

    def type_color_values(self) -> dict[str, str]:
        return dict(self._overrides)


__all__ = ["AUTO_WIDTH", "LineStyleDialog", "MAX_WIDTH", "MIN_WIDTH"]
