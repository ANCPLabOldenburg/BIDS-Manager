"""The small images the stylesheet needs, drawn in the theme's colours.

A stylesheet cannot draw a tick or a chevron, only an image, and an image
cannot be re-coloured; so each theme gets its own set, written as SVG into
the user's cache when the theme is applied and named in the stylesheet by
token (``$icon_check``, ``$icon_chevron_down``, ...). Before this a checked
box was a filled square with no tick, which beside a colour dot read as a
swatch, and every dropdown ended in a small bar.

Paths are absolute, forward-slashed and quoted in the stylesheet, so a
Windows home folder with a space in it still works.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_TICK = ('<path d="M3.6 7.4 L6.1 9.8 L10.6 4.6" fill="none" stroke="{c}" stroke-width="1.9" '
         'stroke-linecap="round" stroke-linejoin="round"/>')
_DASH = '<path d="M4 7 L10 7" fill="none" stroke="{c}" stroke-width="1.9" stroke-linecap="round"/>'
_DOT = '<circle cx="7" cy="7" r="3" fill="{c}"/>'
_CHEVRON = {
    "down": "M4 5.6 L7 8.6 L10 5.6",
    "up": "M4 8.4 L7 5.4 L10 8.4",
    "right": "M5.6 4 L8.6 7 L5.6 10",
}


def _svg(body: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
            f'viewBox="0 0 14 14">{body}</svg>')


def _chevron(direction: str, colour: str) -> str:
    return _svg(f'<path d="{_CHEVRON[direction]}" fill="none" stroke="{colour}" '
                'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')


def _cache_dir() -> Path:
    try:
        from PyQt6.QtCore import QStandardPaths

        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    except Exception:  # noqa: BLE001 - a temporary folder always works
        base = ""
    return Path(base) if base else Path(tempfile.gettempdir()) / "bidsmgr"


def images(palette: dict[str, str]) -> dict[str, str]:
    """The SVG text of each image for ``palette``, by token name."""
    on = palette["primary_btn_text"]
    return {
        "icon_check": _svg(_TICK.format(c=on)),
        "icon_check_disabled": _svg(_TICK.format(c=palette["muted"])),
        "icon_partial": _svg(_DASH.format(c=palette["accent"])),
        "icon_radio": _svg(_DOT.format(c=on)),
        "icon_chevron_down": _chevron("down", palette["dim"]),
        "icon_chevron_down_hover": _chevron("down", palette["text"]),
        "icon_chevron_up": _chevron("up", palette["dim"]),
        "icon_chevron_right": _chevron("right", palette["dim"]),
    }


def write(palette: dict[str, str], theme_id: str) -> dict[str, str]:
    """Write ``palette``'s images (only what changed) and return the
    stylesheet tokens: each a quoted ``url(...)`` ready to substitute."""
    folder = _cache_dir() / "theme" / theme_id
    tokens: dict[str, str] = {}
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("could not create %s for the theme images: %s", folder, exc)
        folder = Path(tempfile.gettempdir()) / "bidsmgr-theme" / theme_id
        folder.mkdir(parents=True, exist_ok=True)
    for name, text in images(palette).items():
        # The colours are in the NAME too: Qt caches a stylesheet image by
        # its path, so a changed accent written over the same file kept
        # showing the old one.
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        path = folder / f"{name}-{digest}.svg"
        try:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("could not write %s: %s", path, exc)
        tokens[name] = f'url("{path.as_posix()}")'
    return tokens


__all__ = ["images", "write"]
