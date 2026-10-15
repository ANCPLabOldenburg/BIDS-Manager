"""Every font the viewers draw, from one place, at the app's font size.

The font-size setting (Settings -> Display) reaches widgets through the
stylesheet and the application font (``theme_manager`` scales both). What a
viewer PAINTS goes through neither: the letters on a slice, a colour bar's
numbers, a plot's axis numbers and titles, a channel's name, an event's
label. A fixed pixel size there stayed small when the user asked for larger
text, and a pyqtgraph axis kept the font it was born with.

So every painted font is made here, from its size at scale 1.0
(``scaled_px``), and every pyqtgraph axis is styled here. Canvases call
these on every theme change, which a font-size change also is (it
re-applies the theme), so nothing has to remember a second signal.

Pixel sizes only, never points (``CROSS_PLATFORM_RULES`` 4.1).
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtGui import QColor, QFont

#: Sizes at scale 1.0, by what the text is.
TICK_PX = 11       # numbers along an axis
AXIS_PX = 12       # an axis' title
LABEL_PX = 11      # a channel's or a lane's name, a caption
SMALL_PX = 10      # an event's label, a colour bar's numbers
LETTER_PX = 12     # an orientation letter
#: A plot's value axis is this wide at scale 1.0 (six digits and a sign).
AXIS_WIDTH_PX = 56


def px(base: int) -> int:
    """``base`` pixels at the app's font scale."""
    from ..theme_manager import scaled_px

    return scaled_px(int(base))


def font(base_px: int, *, bold: bool = False, mono: bool = False) -> QFont:
    """A font of ``base_px`` pixels at scale 1.0, scaled."""
    f = QFont()
    if mono:
        from ..typefaces import mono_families

        f.setFamilies(mono_families())
        f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPixelSize(px(base_px))
    f.setBold(bold)
    return f


def _css(colour: str, base_px: int) -> dict:
    return {"color": QColor(colour).name() if QColor(colour).isValid() else colour,
            "font-size": f"{px(base_px)}px"}


def style_axes(plot_item, colour: str, *, axes: Iterable[str] = ("left", "bottom"),
               tick_px: int = TICK_PX, title_px: int = AXIS_PX) -> None:
    """The numbers and titles of ``plot_item``'s axes at the app's size, in
    ``colour`` (a pyqtgraph axis reads no stylesheet)."""
    for name in axes:
        ax = plot_item.getAxis(name)
        ax.setStyle(tickFont=font(tick_px))
        ax.setTextPen(colour)
        text = getattr(ax, "labelText", "")
        if text:
            ax.setLabel(text, getattr(ax, "labelUnits", "") or None, **_css(colour, title_px))
        else:
            ax.labelStyle = _css(colour, title_px)


def axis_title(plot_item, name: str, text: Optional[str], colour: str, *,
               units: Optional[str] = None, title_px: int = AXIS_PX) -> None:
    """Set an axis' title at the app's size (``None`` removes it)."""
    if not text:
        plot_item.setLabel(name, None)
        return
    plot_item.setLabel(name, text, units=units, **_css(colour, title_px))


def axis_width(base: int = AXIS_WIDTH_PX) -> int:
    """How wide a value axis has to be for its numbers at this font size."""
    return px(base)


__all__ = ["AXIS_PX", "AXIS_WIDTH_PX", "LABEL_PX", "LETTER_PX", "SMALL_PX", "TICK_PX",
           "axis_title", "axis_width", "font", "px", "style_axes"]
