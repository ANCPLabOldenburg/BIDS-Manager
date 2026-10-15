"""Theme manager — owns the palette + token-based QSS template.

The QSS at ``theme.qss`` (sibling of this module) is a ``string.Template``
with named tokens (``$bg``, ``$accent``, ``$success_bg``, …). On theme
change we ``safe_substitute(**palette)`` and call
``QApplication.setStyleSheet`` with the result.

This file is the proven implementation from the prototype at
``../../inspector_proto/proto.py``. The prototype validated visual
fidelity end-to-end including a working dark↔light toggle on
real Siemens Prisma 3T data.

Usage:

    from bidsmgr.gui.theme_manager import ThemeManager
    theme = ThemeManager(app)
    theme.apply('dark')   # or any id in THEMES ('light', 'dim', 'nord', ...)
    theme.toggle()        # to the partner of the other kind

Other GUI modules subscribe to palette changes:

    theme.add_listener(lambda pal: my_widget.repaint_for_palette(pal))

The current palette is always available via ``theme.palette`` or via
the module-level ``CUR()`` accessor used by paint code that runs outside
the listener flow (e.g. ``QStyledItemDelegate.paint``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

from . import tones as _tones


# Baseline app-default font size in logical pixels. Multiplied by the
# current ``FONT_SCALE()`` before being applied to ``QApplication.font``
# in ``ThemeManager.apply`` so the user's chosen scale propagates to
# every widget that doesn't have an explicit QSS ``font-size`` rule.
BASE_FONT_PIXEL_SIZE = 12

# Matches every ``font-size: Npx`` declaration in the QSS template. Used
# by ``_scale_qss_font_sizes`` to multiply each by the active scale.
_FONT_SIZE_RE = re.compile(r"font-size:\s*(\d+)px")


def _scale_qss_font_sizes(qss: str, scale: float) -> str:
    """Multiply every ``font-size: Npx`` in *qss* by *scale*.

    Sizes round to the nearest int and clamp at 1 px so a very small
    scale can't produce ``font-size: 0px`` (which Qt silently rejects).
    """
    if scale == 1.0:
        return qss

    def _replace(m: re.Match[str]) -> str:
        base = int(m.group(1))
        scaled = max(1, round(base * scale))
        return f"font-size: {scaled}px"

    return _FONT_SIZE_RE.sub(_replace, qss)


# =====================================================================
#  PALETTES
# =====================================================================
#
# Roles, the same in every theme (the stylesheet's card design depends on
# them): ``bg`` is the canvas the window, toolbars and bars sit on;
# ``surface`` is a card (a panel); ``surface2`` an input or a raised area;
# ``surface3`` a hover, a secondary button, a header row; ``border`` a card's
# outline and the dividers; ``subtle`` a faint divider; ``input_border`` the
# outline of what a user types into (3:1 against the canvas and the card).
# ``series1``..``series6`` are the plots' colours, apart from the UI's: two
# sets (dark and light surfaces) chosen so neighbouring series stay apart
# for colour-blind readers too; a theme's accent can then be anything.
#
# Every palette is tested (tests/gui/test_themes_and_typefaces.py) for
# contrast and for those separations; a theme that fails is re-coloured.

#: The plots' colours on dark surfaces (Okabe-Ito, lightened) and on light
#: ones (deepened): blue, orange, green, pink, then yellow or violet, grey.
SERIES_DARK = ('#5aaaf5', '#eda45a', '#3dc795', '#e483c2', '#e8d25c', '#aeb3bb')
SERIES_LIGHT = ('#1b6fbf', '#c25a0c', '#12825c', '#ad3f8c', '#5a4fcf', '#5f6670')


def _rgba(hex6: str, alpha: float) -> str:
    h = hex6.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha:.2f})'


def _derive(base: dict[str, str], *, dark: bool, strong: bool = False) -> dict[str, str]:
    """A whole palette from its base colours: the tinted backgrounds and
    borders of each status colour, the muted wash, the pressed shade and
    the plots' series colours (higher alphas for the high-contrast pair)."""
    tint, edge = (0.22, 0.75) if strong else ((0.12, 0.30) if dark else (0.10, 0.30))
    pal = dict(base)
    for key in ('accent', 'success', 'warning', 'error', 'purple', 'teal'):
        pal[f'{key}_bg'] = _rgba(base[key], tint)
        pal[f'{key}_border'] = _rgba(base[key], edge if key != 'accent' or strong
                                     else (0.40 if dark else 0.32))
    pal['muted_40'] = _rgba(base['muted'], 0.40)
    pal['pressed_alpha'] = 'rgba(255,255,255,0.06)' if dark else 'rgba(0,0,0,0.05)'
    for i, colour in enumerate(SERIES_DARK if dark else SERIES_LIGHT, start=1):
        pal[f'series{i}'] = colour
    # The file trees' colours by kind of entry, derived from the theme: one
    # family on an arc around its accent, at one lightness (``tones``). The
    # user can change any of them (Settings > Display; ``appearance``).
    for kind, colour in _tones.file_tones(pal, strong=strong).items():
        pal[f'tree_{kind}'] = colour
    return pal


#: The original: near-black with a blue cast, GitHub-like.
DARK = _derive({
    'bg': '#0a0e13', 'surface': '#11161d', 'surface2': '#161b22', 'surface3': '#1c2128',
    'border': '#21262d', 'subtle': '#1a1f26', 'input_border': '#606a74', 'text': '#e6edf3',
    'dim': '#8b949e', 'muted': '#656d76', 'accent': '#58a6ff', 'success': '#3fb950',
    'warning': '#d29922', 'error': '#f85149', 'purple': '#d2a8ff', 'teal': '#39c5cf',
    'primary_btn_text': '#0a0e13',
}, dark=True)

#: White cards on a cool grey canvas.
LIGHT = _derive({
    'bg': '#f3f5f7', 'surface': '#ffffff', 'surface2': '#f7f8fa', 'surface3': '#eceff3',
    'border': '#d9dee3', 'subtle': '#e8ebef', 'input_border': '#7d868f', 'text': '#1f2328',
    'dim': '#57606a', 'muted': '#8c959f', 'accent': '#0969da', 'success': '#1a7f37',
    'warning': '#946200', 'error': '#cf222e', 'purple': '#8250df', 'teal': '#1b7c83',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: The app's first dark theme, as it was before the 2026-10-09 restyle. Only
#: the input outline moved (#586069 read 2.85:1 on the card; 3:1 is the floor).
ORIGINAL_DARK = _derive({
    'bg': '#0a0e13', 'surface': '#11161d', 'surface2': '#161b22', 'surface3': '#1c2128',
    'border': '#21262d', 'subtle': '#1a1f26', 'input_border': '#5d656e', 'text': '#e6edf3',
    'dim': '#8b949e', 'muted': '#656d76', 'accent': '#58a6ff', 'success': '#3fb950',
    'warning': '#d29922', 'error': '#f85149', 'purple': '#d2a8ff', 'teal': '#39c5cf',
    'primary_btn_text': '#0a0e13',
}, dark=True)

#: The app's first light theme: a white canvas with pale grey cards (the
#: restyled Light turned it round). Only the input outline moved (#8c959f
#: read 2.85:1 on the card).
ORIGINAL_LIGHT = _derive({
    'bg': '#ffffff', 'surface': '#f6f8fa', 'surface2': '#ffffff', 'surface3': '#eef1f4',
    'border': '#d0d7de', 'subtle': '#e5e7ea', 'input_border': '#88919a', 'text': '#1f2328',
    'dim': '#656d76', 'muted': '#8c959f', 'accent': '#0969da', 'success': '#1a7f37',
    'warning': '#9a6700', 'error': '#cf222e', 'purple': '#8250df', 'teal': '#1d7a8c',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: A softer dark: mid blue-greys, lower contrast for long sessions.
DIM = _derive({
    'bg': '#1c2128', 'surface': '#22272e', 'surface2': '#2a3038', 'surface3': '#323942',
    'border': '#3d444d', 'subtle': '#2a3038', 'input_border': '#6e7681', 'text': '#cdd9e5',
    'dim': '#9aa6b2', 'muted': '#768390', 'accent': '#6cb6ff', 'success': '#6bc46d',
    'warning': '#daaa3f', 'error': '#f47067', 'purple': '#dcbdfb', 'teal': '#56d4dd',
    'primary_btn_text': '#1c2128',
}, dark=True)

#: A warm light: cream cards on a sand canvas, softer than white.
PAPER = _derive({
    'bg': '#f2ede2', 'surface': '#fcfaf5', 'surface2': '#fffdf9', 'surface3': '#ebe4d4',
    'border': '#dbd1bd', 'subtle': '#e7dfcf', 'input_border': '#857a66', 'text': '#33291f',
    'dim': '#5f5444', 'muted': '#8d806b', 'accent': '#1d5f86', 'success': '#3d6b1f',
    'warning': '#875400', 'error': '#a51d3e', 'purple': '#6f4697', 'teal': '#1e6b66',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: Neutral dark grey, no colour cast: the quiet default for long work.
GRAPHITE = _derive({
    'bg': '#161618', 'surface': '#1d1d20', 'surface2': '#242428', 'surface3': '#2c2c31',
    'border': '#34343a', 'subtle': '#28282d', 'input_border': '#6a6a73', 'text': '#ececef',
    'dim': '#a6a6ae', 'muted': '#7a7a83', 'accent': '#7aa7ff', 'success': '#5fd08a',
    'warning': '#f2b84b', 'error': '#f47a72', 'purple': '#c3a6ff', 'teal': '#56c8d6',
    'primary_btn_text': '#0f1115',
}, dark=True)

#: Neutral near-black on a black canvas (OLED-friendly).
CARBON = _derive({
    'bg': '#000000', 'surface': '#0e0e10', 'surface2': '#161618', 'surface3': '#1f1f22',
    'border': '#26262a', 'subtle': '#1a1a1d', 'input_border': '#5f5f67', 'text': '#f3f3f5',
    'dim': '#acacb3', 'muted': '#7d7d85', 'accent': '#8ab4ff', 'success': '#62d896',
    'warning': '#f5c25c', 'error': '#ff7f78', 'purple': '#cbaeff', 'teal': '#60d2de',
    'primary_btn_text': '#000000',
}, dark=True)

#: A lighter neutral dark: soft charcoal.
ASH = _derive({
    'bg': '#25262a', 'surface': '#2c2d31', 'surface2': '#333439', 'surface3': '#3b3c41',
    'border': '#45464c', 'subtle': '#36373c', 'input_border': '#7c7d85', 'text': '#e8e8eb',
    'dim': '#b6b7bd', 'muted': '#8d8e95', 'accent': '#8eb4ff', 'success': '#7dd89a',
    'warning': '#efc46e', 'error': '#ff8c84', 'purple': '#d0b4ff', 'teal': '#72d3dc',
    'primary_btn_text': '#1d1e22',
}, dark=True)

#: Neutral light: white cards on a pale grey canvas, no tint.
PORCELAIN = _derive({
    'bg': '#efeff1', 'surface': '#ffffff', 'surface2': '#f7f7f8', 'surface3': '#ececef',
    'border': '#dcdce1', 'subtle': '#ebebee', 'input_border': '#83838c', 'text': '#1b1b1f',
    'dim': '#55555e', 'muted': '#85858e', 'accent': '#2563c9', 'success': '#1c7a40',
    'warning': '#8a5a00', 'error': '#c1312c', 'purple': '#7240c4', 'teal': '#11727c',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: Cool blue-grey surfaces with the Nord palette's frost accents.
NORD = _derive({
    'bg': '#272c36', 'surface': '#2e3440', 'surface2': '#353c4a', 'surface3': '#3b4252',
    'border': '#434c5e', 'subtle': '#353c4a', 'input_border': '#7b88a1', 'text': '#eceff4',
    'dim': '#b9c1cf', 'muted': '#8892a6', 'accent': '#88c0d0', 'success': '#8fc79a',
    'warning': '#ebcb8b', 'error': '#ec959c', 'purple': '#c8a2c4', 'teal': '#8fbcbb',
    'primary_btn_text': '#2e3440',
}, dark=True)

#: Warm charcoal with an amber accent, like a dim room by a fire.
EMBER = _derive({
    'bg': '#191513', 'surface': '#211c19', 'surface2': '#29231f', 'surface3': '#322a25',
    'border': '#3b322c', 'subtle': '#2b2420', 'input_border': '#7a6b61', 'text': '#f3ebe4',
    'dim': '#c4b6aa', 'muted': '#968679', 'accent': '#f2a65a', 'success': '#7fd39a',
    'warning': '#e6d26a', 'error': '#ff6f86', 'purple': '#d9a3d0', 'teal': '#7cc9bd',
    'primary_btn_text': '#1a1310',
}, dark=True)

#: Deep green-grey with a golden accent.
FOREST = _derive({
    'bg': '#111714', 'surface': '#171f1b', 'surface2': '#1e2822', 'surface3': '#26312a',
    'border': '#2e3a33', 'subtle': '#212b25', 'input_border': '#64776b', 'text': '#e4eee7',
    'dim': '#a9bcb0', 'muted': '#7d9085', 'accent': '#e8b66a', 'success': '#8fd07b',
    'warning': '#f59e5b', 'error': '#ff6f91', 'purple': '#c3a8f2', 'teal': '#66c9d4',
    'primary_btn_text': '#14110b',
}, dark=True)

#: Deep plum with an orchid accent.
AUBERGINE = _derive({
    'bg': '#161119', 'surface': '#1e1722', 'surface2': '#261e2b', 'surface3': '#2f2535',
    'border': '#3a2e41', 'subtle': '#291f2e', 'input_border': '#786884', 'text': '#f2e9f6',
    'dim': '#c3b2cf', 'muted': '#93829f', 'accent': '#f095d0', 'success': '#86d3a2',
    'warning': '#efc570', 'error': '#ff8562', 'purple': '#b6a2ff', 'teal': '#74d0de',
    'primary_btn_text': '#1e1722',
}, dark=True)

#: The classic editor palette: warm olive-black with its vivid cyan, green,
#: orange, pink and purple. Its yellow is too close to its green for two
#: statuses shown side by side, so orange is the warning.
MONOKAI = _derive({
    'bg': '#1e1f1a', 'surface': '#272822', 'surface2': '#2e2f29', 'surface3': '#3a3b32',
    'border': '#49483e', 'subtle': '#33342c', 'input_border': '#868268', 'text': '#f8f8f2',
    'dim': '#cfcfc2', 'muted': '#97927a', 'accent': '#66d9ef', 'success': '#a6e22e',
    'warning': '#fd971f', 'error': '#ff7096', 'purple': '#bc9bff', 'teal': '#7fe3c9',
    'primary_btn_text': '#1e1f1a',
}, dark=True)

#: Blush cards with a plum accent.
SAKURA = _derive({
    'bg': '#f5ebef', 'surface': '#fffafb', 'surface2': '#ffffff', 'surface3': '#f3e4e9',
    'border': '#ead2da', 'subtle': '#f1e1e7', 'input_border': '#9d7d88', 'text': '#371f27',
    'dim': '#6c4c57', 'muted': '#9a7c86', 'accent': '#8b3384', 'success': '#2a7448',
    'warning': '#87540a', 'error': '#c0262d', 'purple': '#4b4bb3', 'teal': '#1b7470',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: Pale green-grey with a slate-blue accent.
SAGE = _derive({
    'bg': '#eaefea', 'surface': '#fbfcfa', 'surface2': '#ffffff', 'surface3': '#e2e9e2',
    'border': '#d2dcd2', 'subtle': '#e1e8e1', 'input_border': '#7a8c80', 'text': '#1d2820',
    'dim': '#4a5c50', 'muted': '#7a8b7e', 'accent': '#2d6597', 'success': '#3b7a28',
    'warning': '#8a5c00', 'error': '#b42347', 'purple': '#6c48a6', 'teal': '#16736f',
    'primary_btn_text': '#ffffff',
}, dark=False)

#: Black and white with strong borders: every text at 7:1 or better (AAA).
HC_DARK = _derive({
    'bg': '#000000', 'surface': '#0b0b0b', 'surface2': '#141414', 'surface3': '#202020',
    'border': '#7a7a7a', 'subtle': '#1a1a1a', 'input_border': '#cfcfcf', 'text': '#ffffff',
    'dim': '#e0e0e0', 'muted': '#b8b8b8', 'accent': '#5cc8ff', 'success': '#5ef07a',
    'warning': '#ffd84d', 'error': '#ff8f8f', 'purple': '#e7b6ff', 'teal': '#5cf0f0',
    'primary_btn_text': '#000000',
}, dark=True, strong=True)

#: White and black with strong borders.
HC_LIGHT = _derive({
    'bg': '#ffffff', 'surface': '#ffffff', 'surface2': '#ffffff', 'surface3': '#ececec',
    'border': '#6e6e6e', 'subtle': '#e3e3e3', 'input_border': '#1a1a1a', 'text': '#000000',
    'dim': '#1f1f1f', 'muted': '#4d4d4d', 'accent': '#0039a6', 'success': '#08521a',
    'warning': '#6b3f00', 'error': '#9e0031', 'purple': '#4f1a99', 'teal': '#00545a',
    'primary_btn_text': '#ffffff',
}, dark=False, strong=True)


@dataclass(frozen=True)
class ThemeInfo:
    """A theme the user can pick: its palette, and what it is."""

    id: str
    label: str
    dark: bool
    #: The theme of the other kind ``toggle`` switches to.
    partner: str
    #: Where the menus list it: Neutral, Classic, Colour, High contrast.
    group: str
    description: str


#: Every theme, in the order the menus list them, by group, dark first.
THEMES: tuple[ThemeInfo, ...] = (
    ThemeInfo('graphite', 'Graphite', True, 'porcelain', 'Neutral',
              'Neutral dark grey with no colour cast.'),
    ThemeInfo('carbon', 'Carbon', True, 'porcelain', 'Neutral',
              'Neutral near-black on a black canvas.'),
    ThemeInfo('ash', 'Ash', True, 'porcelain', 'Neutral', 'A lighter neutral charcoal.'),
    ThemeInfo('porcelain', 'Porcelain', False, 'graphite', 'Neutral',
              'White cards on a pale grey canvas, no tint.'),
    ThemeInfo('dark', 'Dark', True, 'light', 'Classic', 'Near-black with a blue cast.'),
    ThemeInfo('dim', 'Dim', True, 'paper', 'Classic', 'A softer blue-grey dark for long sessions.'),
    ThemeInfo('original-dark', 'Original dark', True, 'original-light', 'Classic',
              'The first dark theme, as it was before the restyle.'),
    ThemeInfo('light', 'Light', False, 'dark', 'Classic', 'White cards on a cool grey canvas.'),
    ThemeInfo('paper', 'Paper', False, 'dim', 'Classic', 'Cream cards on a warm sand canvas.'),
    ThemeInfo('original-light', 'Original light', False, 'original-dark', 'Classic',
              'The first light theme: a white canvas with pale grey cards.'),
    ThemeInfo('nord', 'Nord', True, 'light', 'Colour', 'Cool blue-grey with frost accents.'),
    ThemeInfo('ember', 'Ember', True, 'paper', 'Colour', 'Warm charcoal with an amber accent.'),
    ThemeInfo('forest', 'Forest', True, 'sage', 'Colour', 'Deep green-grey with a golden accent.'),
    ThemeInfo('aubergine', 'Aubergine', True, 'sakura', 'Colour', 'Deep plum with an orchid accent.'),
    ThemeInfo('monokai', 'Monokai', True, 'paper', 'Colour',
              'The classic editor palette on warm olive-black.'),
    ThemeInfo('sakura', 'Sakura', False, 'aubergine', 'Colour', 'Blush cards with a plum accent.'),
    ThemeInfo('sage', 'Sage', False, 'forest', 'Colour', 'Pale green-grey with a slate-blue accent.'),
    ThemeInfo('hc-dark', 'High contrast dark', True, 'hc-light', 'High contrast',
              'Black and white with strong borders.'),
    ThemeInfo('hc-light', 'High contrast light', False, 'hc-dark', 'High contrast',
              'White and black with strong borders.'),
)

PALETTES: dict[str, dict[str, str]] = {
    'graphite': GRAPHITE, 'carbon': CARBON, 'ash': ASH, 'porcelain': PORCELAIN,
    'dark': DARK, 'dim': DIM, 'original-dark': ORIGINAL_DARK, 'light': LIGHT, 'paper': PAPER,
    'original-light': ORIGINAL_LIGHT,
    'nord': NORD, 'ember': EMBER, 'forest': FOREST, 'aubergine': AUBERGINE, 'monokai': MONOKAI,
    'sakura': SAKURA, 'sage': SAGE, 'hc-dark': HC_DARK, 'hc-light': HC_LIGHT,
}


def theme_info(theme_id: str) -> ThemeInfo:
    """The theme ``theme_id`` (Dark for an id no longer known)."""
    return next((t for t in THEMES if t.id == theme_id), theme_info_default())


def theme_info_default() -> ThemeInfo:
    return next(t for t in THEMES if t.id == 'dark')


def theme_ids() -> list[str]:
    return [t.id for t in THEMES]


# =====================================================================
#  Module-level "current palette" + "current font scale" accessors.
# =====================================================================
_CURRENT: dict[str, str] = DARK
_FONT_SCALE: float = 1.0


def CUR() -> dict[str, str]:
    """Return the active palette dict.

    Used by ``QStyledItemDelegate`` paint methods (which are not GUI
    widgets and don't subscribe to listeners). Updated whenever
    ``ThemeManager.apply`` is called.
    """
    return _CURRENT


def FONT_SCALE() -> float:
    """Return the active UI font-size multiplier (1.0 = baseline).

    Paint code and inline ``setStyleSheet`` snippets use this so their
    hard-coded pixel sizes scale with the user's preference set via
    ``Settings → Display → Font scale``. Updated whenever
    ``ThemeManager.apply`` is called.
    """
    return _FONT_SCALE


def scaled_px(base: int) -> int:
    """Return *base* (px) multiplied by the active font scale, rounded
    to the nearest int and clamped at 1.

    Convenience for call sites that paint or build inline stylesheets:
    ``f.setPixelSize(scaled_px(11))``.
    """
    return max(1, round(base * _FONT_SCALE))


def rgba(hex6: str, alpha: float) -> QColor:
    """Convenience: ``hex6`` color with the given ``alpha`` (0–1)."""
    c = QColor(hex6)
    c.setAlphaF(alpha)
    return c


# =====================================================================
#  ThemeManager
# =====================================================================
class ThemeManager:
    """Owns the QSS template + active palette. Re-applies on toggle.

    Listeners are called *after* the QSS is applied with the new palette
    dict, so they can repaint anything that doesn't pick up automatically.
    """

    def __init__(
        self,
        app: QApplication,
        qss_path: Path | None = None,
        font_scale: float = 1.0,
        appearance=None,
    ):
        self._app = app
        self._raw_template_text = (
            qss_path or Path(__file__).parent / 'theme.qss'
        ).read_text(encoding='utf-8')
        self._theme = 'dark'
        self._listeners: list[Callable[[dict], None]] = []
        from .appearance import Appearance

        #: The user's touches (accent, tint, icon style, tree colours),
        #: laid over every theme by ``apply``.
        self._appearance = (appearance or Appearance()).normalised()
        self._effective: dict[str, str] = dict(DARK)
        # The font scale is applied to the QSS template + QApplication
        # default font at every ``apply`` call. Stored on the manager so
        # callers can swap it without re-creating the manager.
        self._font_scale = max(0.5, min(float(font_scale), 2.0))

    # ------------------------------------------------------------- listeners
    def add_listener(self, fn: Callable[[dict], None]) -> None:
        """``fn(palette: dict)`` is called after every theme change."""
        self._listeners.append(fn)

    # ---------------------------------------------------------------- state
    @property
    def palette(self) -> dict[str, str]:
        """The palette in use: the theme with the user's appearance over it."""
        return self._effective

    @property
    def appearance(self):
        return self._appearance

    def set_appearance(self, appearance, *, apply: bool = True) -> None:
        """Change the accent, tint, icon style or tree colours (and re-apply,
        unless the caller applies a theme next anyway)."""
        self._appearance = appearance.normalised()
        if apply:
            self.apply(self._theme)

    @property
    def name(self) -> str:
        return self._theme

    @property
    def font_scale(self) -> float:
        return self._font_scale

    def set_font_scale(self, scale: float) -> None:
        """Update the UI font scale and re-apply the active theme.

        Clamps to ``[0.5, 2.0]`` so a bad value (corrupted setting,
        out-of-range from a future Settings UI) can't render the GUI
        unusable. No-op when the value would not change anything.
        """
        scale = max(0.5, min(float(scale), 2.0))
        if scale == self._font_scale:
            return
        self._font_scale = scale
        # Re-apply the current theme so the QSS gets re-scaled and every
        # listener (panels' ``repaint_for_palette``) refreshes any
        # inline-stylesheet font sizes that were baked at construction.
        self.apply(self._theme)

    # --------------------------------------------------------------- actions
    def _follow_native_scheme(self, dark: bool) -> None:
        """Tell the platform which scheme the theme is. Without it Qt follows
        the system: on a Mac in dark mode every window of a light theme had
        a black title bar and frame (a spectrum window most visibly), and
        native dialogs followed macOS instead of the theme. Qt 6.8+ (the
        call is absent before); on Linux it is a hint the platform may
        ignore."""
        try:
            from PyQt6.QtCore import Qt as _Qt

            hints = self._app.styleHints()
            if hasattr(hints, "setColorScheme"):
                hints.setColorScheme(_Qt.ColorScheme.Dark if dark else _Qt.ColorScheme.Light)
        except Exception:  # noqa: BLE001 - a hint, never a failure
            pass

    def apply(self, theme: str) -> None:
        if theme not in PALETTES:
            return
        global _CURRENT, _FONT_SCALE
        from . import appearance as _appearance
        from . import icons as _icons

        self._theme = theme
        info = theme_info(theme)
        pal = _appearance.apply(PALETTES[theme], self._appearance, dark=info.dark,
                                strong=theme.startswith('hc-'))
        self._effective = pal
        _CURRENT = pal
        _icons.set_style(self._appearance.icons)
        _FONT_SCALE = self._font_scale

        # Apply the active scale to (a) the QSS template every ``font-size:
        # Npx`` declaration, then (b) the QApplication default font's
        # pixel size so widgets without an explicit QSS font-size rule
        # scale too. Custom paint code reads the scale via ``FONT_SCALE()``.
        scaled_template = Template(
            _scale_qss_font_sizes(self._raw_template_text, self._font_scale)
        )
        # The stylesheet's images (tick, chevrons) are drawn in this
        # palette's colours and named by token beside its colours.
        from .theme_assets import write as write_theme_images

        tokens = {**pal, **write_theme_images(pal, theme)}
        self._app.setStyleSheet(scaled_template.safe_substitute(**tokens))
        self._update_qpalette(pal)
        self._follow_native_scheme(info.dark)
        try:
            app_font = QFont(self._app.font())
            app_font.setPixelSize(
                max(1, round(BASE_FONT_PIXEL_SIZE * self._font_scale))
            )
            self._app.setFont(app_font)
        except Exception:
            pass

        for fn in self._listeners:
            try:
                fn(pal)
            except Exception as exc:  # pragma: no cover — never let a listener crash the app
                print(f'[theme listener] {exc}')

    @property
    def is_dark(self) -> bool:
        return theme_info(self._theme).dark

    def toggle(self) -> str:
        """Switch to the current theme's partner of the other kind (Dark and
        Light, Dim and Paper, the two high-contrast themes)."""
        self.apply(theme_info(self._theme).partner)
        return self._theme

    # ------------------------------------------------------------- internals
    def _update_qpalette(self, pal: dict[str, str]) -> None:
        """Set ``QPalette`` baseline so widgets QSS doesn't fully cover behave."""
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window,          QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.WindowText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Base,            QColor(pal['bg']))
        p.setColor(QPalette.ColorRole.AlternateBase,   QColor(pal['surface']))
        p.setColor(QPalette.ColorRole.Text,            QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Button,          QColor(pal['surface3']))
        p.setColor(QPalette.ColorRole.ButtonText,      QColor(pal['text']))
        p.setColor(QPalette.ColorRole.Highlight,       QColor(pal['accent']))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(pal['primary_btn_text']))
        # Rich-text ``<a href>`` links (welcome-panel resources) read this role,
        # so theme swaps recolour them automatically.
        p.setColor(QPalette.ColorRole.Link,            QColor(pal['accent']))
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(pal['surface2']))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(pal['text']))
        # The lines Qt draws itself (a document-mode tab bar's base, frame
        # bevels) read these; left at Qt's defaults they were pure white in
        # every dark theme (the Log's rule over its tabs).
        for role in (QPalette.ColorRole.Light, QPalette.ColorRole.Midlight,
                     QPalette.ColorRole.Mid, QPalette.ColorRole.Dark):
            p.setColor(role, QColor(pal['border']))
        p.setColor(QPalette.ColorRole.Shadow, QColor(pal['bg']))
        self._app.setPalette(p)
