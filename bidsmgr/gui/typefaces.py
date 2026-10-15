"""The two typefaces BIDS Manager draws everything in, bundled with it.

Inter for the interface and JetBrains Mono for code, logs and raw text,
shipped under ``gui/assets/fonts`` (SIL Open Font Licence) and registered
with Qt before the stylesheet is applied, so the app looks the same on
macOS, Windows and Linux. Before this the stylesheet named macOS fonts
("SF Mono", "Menlo", "Monaco") that Windows and Linux do not have, and
each system fell back to a monospace of its own; Qt reported "SF Mono"
missing even on a Mac.

Nothing else names a font family: widgets inherit the application font
(Inter), painted text goes through ``gui/viz/fonts.font`` and the
stylesheet asks for ``"JetBrains Mono"`` where the content is code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase

log = logging.getLogger(__name__)

#: The interface typeface.
UI_FAMILY = "Inter"
#: The typeface for code, logs and raw text.
MONO_FAMILY = "JetBrains Mono"
#: Where the font files live.
FONT_DIR = Path(__file__).parent / "assets" / "fonts"

_loaded: dict[str, bool] = {}


def load() -> bool:
    """Register the bundled fonts with Qt (once). True when both families
    are available; when a file cannot be registered the app falls back to
    the system's own fonts rather than failing."""
    if _loaded:
        return all(_loaded.values())
    found: set[str] = set()
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            log.warning("could not register the font %s", path.name)
            continue
        found.update(QFontDatabase.applicationFontFamilies(font_id))
    for family in (UI_FAMILY, MONO_FAMILY):
        _loaded[family] = family in found
    return all(_loaded.values())


def ui_font(pixel_size: int) -> QFont:
    """The interface font at ``pixel_size`` pixels."""
    f = QFont(UI_FAMILY)
    f.setStyleHint(QFont.StyleHint.SansSerif)
    f.setPixelSize(max(1, int(pixel_size)))
    return f


def mono_families() -> list[str]:
    """The monospace family. Nothing else is listed: a family a system
    lacks makes Qt build its alias table (about 60 ms at start), and when
    the bundled file is missing the Monospace style hint picks the
    system's own."""
    return [MONO_FAMILY]


#: Rich text for code: ``<code>`` asks Qt for a family named "Monospace",
#: which no system has under that name, so Qt spends ~60 ms building its
#: alias table and then picks whatever it finds.
CODE_OPEN = f"<span style=\"font-family:'{MONO_FAMILY}'\">"
CODE_CLOSE = "</span>"


def code(text: str) -> str:
    """``text`` as inline code in rich text (escape it first if needed)."""
    return f"{CODE_OPEN}{text}{CODE_CLOSE}"


__all__ = ["CODE_CLOSE", "CODE_OPEN", "FONT_DIR", "MONO_FAMILY", "UI_FAMILY", "code", "load",
           "mono_families", "ui_font"]
