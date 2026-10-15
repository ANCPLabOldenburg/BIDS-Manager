"""Colour arithmetic for the themes and the user's own colours.

Three questions, answered in OKLab (lightness and colourfulness as the eye
sees them, so one lightness LOOKS like one lightness across hues):

- **What colour is each kind of file name in this theme?**
  (:func:`file_tones`). One family per theme: hues on an arc around the
  theme's accent, every kind at ONE lightness, between the secondary text
  and the text, with a gentle colourfulness. Ember's names are therefore
  golds and roses, Nord's frost blues and sea greens, a neutral theme's a
  quiet cool family. Hand-picked sets per theme (round 4) were vivid,
  varied in lightness and took hues the theme did not have, which read as
  heterogeneous and out of tone.
- **What is this colour on a theme of the other kind?** (:func:`equivalent`).
  A colour chosen on a dark theme (white file names) is unreadable on a light
  one; its equivalent keeps the hue and colourfulness and mirrors the
  lightness so it reads against a light card as the original did against a
  dark one (white becomes near black).
- **Is it readable here?** (:func:`readable`). A colour too close to the card
  is moved away from it in lightness, keeping its hue.

Pure arithmetic: no Qt.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

#: Degrees between neighbouring file kinds on the arc around the accent.
FILE_STEP = 36.0
#: Each kind's place on the arc, in steps from the accent. Files that sit
#: together get the hues furthest apart: an image beside its sidecar
#: (+1/-1), a table beside a recording (+2/-2).
FILE_ARC = {"image": 1, "sidecar": -1, "table": 2, "recording": -2}

#: The file kinds' lightness band (OKLab) on dark and on light themes.
LIGHTNESS = {True: (0.76, 0.82), False: (0.40, 0.46)}

#: The card a colour is judged against when it moves between kinds of theme.
REFERENCE = {True: "#1d1d20", False: "#ffffff"}


# -- sRGB ---------------------------------------------------------------------


def _rgb(hex6: str) -> tuple[int, int, int]:
    h = hex6.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(c))) for c in rgb))


def is_hex(value) -> bool:
    v = (value or "").strip() if isinstance(value, str) else ""
    return len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdefABCDEF"
                                                      for c in v[1:])


def mix(base: str, other: str, amount: float) -> str:
    """``base`` moved ``amount`` (0 to 1) of the way toward ``other``."""
    a, b = _rgb(base), _rgb(other)
    return _hex(tuple(x + (y - x) * amount for x, y in zip(a, b)))


def _linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def luminance(hex6: str) -> float:
    r, g, b = (_linear(c) for c in _rgb(hex6))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio, 1 to 21."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# -- OKLab ----------------------------------------------------------------------


def oklch(hex6: str) -> tuple[float, float, float]:
    """Lightness (0 to 1), chroma and hue in degrees."""
    r, g, b = (_linear(c) for c in _rgb(hex6))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lightness, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def from_oklch(lightness: float, chroma: float, hue: float) -> Optional[str]:
    """The sRGB colour, or ``None`` when it lies outside sRGB."""
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    rgb = (4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
           -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
           -0.0041960771 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3)
    if min(rgb) < -1e-4 or max(rgb) > 1 + 1e-4:
        return None
    return _hex(tuple(255 * _gamma(max(0.0, min(1.0, c))) for c in rgb))


def _in_gamut(lightness: float, chroma: float, hue: float) -> str:
    """The most colourful sRGB colour at this lightness and hue, up to
    ``chroma``."""
    lightness = max(0.0, min(1.0, lightness))
    c = chroma
    while c > 0:
        found = from_oklch(lightness, c, hue)
        if found is not None:
            return found
        c -= 0.005
    return from_oklch(lightness, 0.0, hue) or "#808080"


# -- readability ------------------------------------------------------------------


def readable(colour: str, background: str, floor: float = 4.5) -> str:
    """``colour``, or the nearest colour of its hue that reads at ``floor``
    against ``background`` (moved in lightness away from it)."""
    if contrast(colour, background) >= floor:
        return colour
    lightness, chroma, hue = oklch(colour)
    step = 0.01 if oklch(background)[0] < 0.5 else -0.01
    for i in range(1, 101):
        candidate = _in_gamut(lightness + i * step, chroma, hue)
        if contrast(candidate, background) >= floor:
            return candidate
    return "#ffffff" if step > 0 else "#000000"


def equivalent(colour: str, *, to_dark: bool) -> str:
    """``colour`` (chosen on a theme of the other kind) for a dark
    (``to_dark``) or a light theme: the same hue and colourfulness, at the
    lightness that reads against this kind's card as it did against the
    other's (at least 4.5:1)."""
    source, target = REFERENCE[not to_dark], REFERENCE[to_dark]
    want = max(4.5, contrast(colour, source))
    lightness, chroma, hue = oklch(colour)
    best, best_gap = None, None
    for i in range(101):
        candidate = _in_gamut(i / 100, chroma, hue)
        got = contrast(candidate, target)
        # On the far side of the target card only, closest to what it had.
        if (oklch(candidate)[0] > 0.5) != to_dark:
            continue
        gap = abs(got - want) + (1000 if got < min(want, 4.5) else 0)
        if best_gap is None or gap < best_gap:
            best, best_gap = candidate, gap
    return best or ("#f2f2f2" if to_dark else "#1a1a1a")


def for_kind(choice: Mapping[str, str], dark: bool) -> Optional[str]:
    """The user's colour for this kind of theme: their own choice for it,
    else the equivalent of their choice for the other kind, else ``None``.
    ``choice`` is ``{"dark": "#..", "light": "#.."}`` (either may be
    missing)."""
    mine = choice.get("dark" if dark else "light")
    if is_hex(mine):
        return mine.lower()
    other = choice.get("light" if dark else "dark")
    return equivalent(other.lower(), to_dark=dark) if is_hex(other) else None


# -- the file kinds ------------------------------------------------------------------


def _fit(lightness: float, chroma: float, hue: float, palette: Mapping[str, str],
         floors: Mapping[str, float], dark: bool) -> str:
    """The most colourful colour at this hue and lightness that reads on
    every surface; the lightness moves (away from the cards) only if no
    colourfulness reads."""
    for i in range(0, 40):
        lit = lightness + (i * 0.01 if dark else -i * 0.01)
        c = chroma
        while c >= 0:
            found = from_oklch(max(0.0, min(1.0, lit)), c, hue)
            if found is not None and all(contrast(found, palette[s]) >= f
                                         for s, f in floors.items()):
                return found
            c -= 0.005
    return palette["text"]


def file_tones(palette: Mapping[str, str], *, strong: bool = False,
               hue: Optional[float] = None) -> dict[str, str]:
    """The file trees' colours for ``palette``: folders the accent, other
    files the secondary text, and images, sidecars, tables and recordings on
    an arc around the accent's hue, at one lightness. ``hue`` stands in for
    an accent with no hue of its own (a grey)."""
    dark = oklch(palette["bg"])[0] < 0.5
    lt, ld = oklch(palette["text"])[0], oklch(palette["dim"])[0]
    # Between the secondary text and the text, inside the band where every
    # hue still holds some colour: near white or near black a cyan or a
    # yellow has none left, and the names read as plain text.
    lo, hi = LIGHTNESS[dark]
    lightness = max(lo, min(hi, ld + 0.5 * (lt - ld)))
    _l, accent_chroma, accent_hue = oklch(palette["accent"])
    if accent_chroma < 0.02 and hue is not None:
        accent_hue = hue
    chroma = max(0.075, min(0.10, 0.8 * accent_chroma)) * (1.3 if strong else 1.0)
    floors = {s: (7.0 if strong else (4.0 if s == "surface3" else 4.5))
              for s in ("bg", "surface", "surface2", "surface3")}
    out = {"folder": palette["accent"], "other": palette["dim"]}
    for kind, place in FILE_ARC.items():
        out[kind] = _fit(lightness, chroma, (accent_hue + place * FILE_STEP) % 360,
                         palette, floors, dark)
    return out


__all__ = [
    "FILE_ARC", "FILE_STEP", "LIGHTNESS", "REFERENCE", "contrast", "equivalent", "file_tones",
    "for_kind", "from_oklch", "is_hex", "luminance", "mix", "oklch", "readable",
]
