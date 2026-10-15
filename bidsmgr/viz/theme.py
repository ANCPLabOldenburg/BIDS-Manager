"""Colours the canvases draw with, derived from the app's palette tokens.

pyqtgraph and the GL render read no QSS, so every canvas is HANDED its colours.
:class:`VizTheme` is that hand-off: built once per palette change from the
same token dict ``theme_manager`` fills the stylesheet with, so a plot, a
slice view and the 3-D background follow a theme swap together, dialogs
included.

Pure data: no Qt types, colours are ``#rrggbb`` strings or RGBA tuples.
Palette tokens may be ``rgba(...)`` strings (QColor cannot parse those and
renders black), so :func:`parse_colour` reads both forms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_RGBA = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)")

#: Channel type -> palette token, so trace colours follow the theme. The
#: table the old PSD dialog and traces view used, kept as it was so nobody's
#: magnetometers change colour: ``mag`` the accent, ``grad`` the success
#: colour, so the two stay distinguishable in either theme.
TYPE_TOKENS: dict[str, str] = {
    "mag": "accent", "grad": "success", "eeg": "purple", "seeg": "purple",
    "ecog": "purple", "eog": "teal", "ecg": "error", "emg": "warning",
    "stim": "warning", "ref_meg": "dim", "misc": "dim", "bio": "teal",
    "resp": "teal", "dbs": "purple",
}
#: Distinct colours for event labels and PSD curves, in fixed order.
SERIES_TOKENS: tuple[str, ...] = (
    "accent", "success", "purple", "teal", "warning", "error", "dim",
)


def parse_colour(value: str) -> tuple[int, int, int, int]:
    """``#rgb``, ``#rrggbb``, ``#rrggbbaa`` or ``rgba(r, g, b, a)`` to RGBA."""
    value = (value or "").strip()
    m = _RGBA.fullmatch(value)
    if m:
        r, g, b = (int(round(float(m.group(i)))) for i in (1, 2, 3))
        a = m.group(4)
        alpha = 255 if a is None else int(round(float(a) * 255 if float(a) <= 1 else float(a)))
        return r, g, b, alpha
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) in (6, 8):
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            a = int(h[6:8], 16) if len(h) == 8 else 255
            return r, g, b, a
    return 0, 0, 0, 255


def luminance(value: str) -> float:
    """WCAG relative luminance of a colour, 0 (black) to 1 (white)."""
    def channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b, _a = parse_colour(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def is_dark_colour(value: str) -> bool:
    """Whether text on this background should be light."""
    return luminance(value) < 0.18


def hex_colour(rgba: tuple[int, int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgba[:3])


@dataclass(frozen=True)
class VizTheme:
    """Everything a canvas needs to look like the rest of the app."""

    name: str
    background: str          # behind images: black on purpose in both themes
    plot_background: str     # pyqtgraph plots follow the theme
    plot_foreground: str
    grid: str
    text: str
    dim: str
    accent: str
    label: str               # orientation letters on slices
    crosshair: str           # default crosshair colour (user setting wins)
    caption: str
    tokens: dict[str, str] = field(default_factory=dict)
    #: Text drawn ON the image surround, which is black in both themes: light
    #: in both themes too. The app's text colour is dark in the light theme
    #: and read as nothing on black (colour bar titles, captions, letters).
    canvas_text: str = "#e6edf3"
    canvas_dim: str = "#a7b0ba"
    #: A dark theme (any of them): decided by the background's luminance,
    #: so a theme added later needs no list to be kept.
    dark: bool = True

    @classmethod
    def from_palette(cls, palette: dict[str, str], name: str = "dark") -> "VizTheme":
        p = dict(palette)
        return cls(
            name=name,
            dark=is_dark_colour(p.get("bg", "#0a0e13")),
            # Images keep a black surround in both themes: grey matter on a
            # white field reads as a different image.
            background="#000000",
            plot_background=p.get("bg", "#0a0e13"),
            plot_foreground=p.get("text", "#e6edf3"),
            grid=p.get("border", "#21262d"),
            text=p.get("text", "#e6edf3"),
            dim=p.get("dim", "#8b949e"),
            accent=p.get("accent", "#58a6ff"),
            # On the black surround: the dark theme's own blue and grey,
            # whatever the app's theme.
            label="#58a6ff",
            crosshair="#4FC3F7",
            caption="#a7b0ba",
            tokens=p,
        )

    def token(self, name: str, default: str = "#888888") -> str:
        return self.tokens.get(name, default)

    def series(self, i: int) -> str:
        """The ``i``-th plot colour: the palette's own ``series1``..``series6``
        (chosen to stay apart, colour-blind readers included), else the UI
        colours in ``SERIES_TOKENS``' order."""
        own = self.tokens.get(f"series{i % 6 + 1}")
        return own or self.token(SERIES_TOKENS[i % len(SERIES_TOKENS)])

    def type_colour(self, ch_type: str, overrides: dict[str, str] | None = None) -> str:
        """Colour for a channel type: the user's choice, else its token.

        A type the table does not name gets a token derived FROM ITS NAME
        rather than one shared grey: a real MEG file carries ``ias`` and
        ``syst`` beside ``misc``, and three kinds in one colour are three
        kinds nobody can tell apart. A character sum, not ``hash()``, which
        Python randomises per process.
        """
        if overrides and ch_type in overrides and overrides[ch_type]:
            return overrides[ch_type]
        return self.token(self.type_token(ch_type))

    @staticmethod
    def type_token(ch_type: str) -> str:
        token = TYPE_TOKENS.get(ch_type)
        if token is None:
            name = str(ch_type or "")
            token = SERIES_TOKENS[sum(map(ord, name)) % len(SERIES_TOKENS)] if name else "dim"
        return token


def default_theme() -> VizTheme:
    """A dark theme for code that runs before the app applied a palette."""
    return VizTheme.from_palette({}, "dark")


__all__ = [
    "SERIES_TOKENS", "TYPE_TOKENS", "VizTheme", "default_theme", "hex_colour",
    "is_dark_colour", "luminance", "parse_colour",
]
