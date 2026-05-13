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
    theme.apply('dark')   # or 'light'
    theme.toggle()        # swap

Other GUI modules subscribe to palette changes:

    theme.add_listener(lambda pal: my_widget.repaint_for_palette(pal))

The current palette is always available via ``theme.palette`` or via
the module-level ``CUR()`` accessor used by paint code that runs outside
the listener flow (e.g. ``QStyledItemDelegate.paint``).
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Callable

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


# =====================================================================
#  PALETTES
# =====================================================================
DARK: dict[str, str] = {
    'bg':         '#0a0e13',
    'surface':    '#11161d',
    'surface2':   '#161b22',
    'surface3':   '#1c2128',
    'border':     '#21262d',
    'subtle':     '#1a1f26',
    'text':       '#e6edf3',
    'dim':        '#8b949e',
    'muted':      '#656d76',
    'accent':     '#58a6ff',
    'success':    '#3fb950',
    'warning':    '#d29922',
    'error':      '#f85149',
    'purple':     '#d2a8ff',
    'teal':       '#39c5cf',

    'muted_40':        'rgba(101,109,118,0.40)',

    'accent_bg':       'rgba(88,166,255,0.12)',
    'accent_border':   'rgba(88,166,255,0.40)',
    'success_bg':      'rgba(63,185,80,0.12)',
    'success_border':  'rgba(63,185,80,0.30)',
    'warning_bg':      'rgba(210,153,34,0.12)',
    'warning_border':  'rgba(210,153,34,0.30)',
    'error_bg':        'rgba(248,81,73,0.12)',
    'error_border':    'rgba(248,81,73,0.30)',
    'purple_bg':       'rgba(210,168,255,0.12)',
    'purple_border':   'rgba(210,168,255,0.30)',
    'teal_bg':         'rgba(57,197,207,0.12)',
    'teal_border':     'rgba(57,197,207,0.30)',

    'primary_btn_text': '#0a0e13',
    'pressed_alpha':    'rgba(255,255,255,0.04)',
}

LIGHT: dict[str, str] = {
    'bg':         '#ffffff',
    'surface':    '#f6f8fa',
    'surface2':   '#ffffff',
    'surface3':   '#eef1f4',
    'border':     '#d0d7de',
    'subtle':     '#e5e7ea',
    'text':       '#1f2328',
    'dim':        '#656d76',
    'muted':      '#8c959f',
    'accent':     '#0969da',
    'success':    '#1a7f37',
    'warning':    '#9a6700',
    'error':      '#cf222e',
    'purple':     '#8250df',
    'teal':       '#1d7a8c',

    'muted_40':        'rgba(140,149,159,0.40)',

    'accent_bg':       'rgba(9,105,218,0.08)',
    'accent_border':   'rgba(9,105,218,0.32)',
    'success_bg':      'rgba(26,127,55,0.10)',
    'success_border':  'rgba(26,127,55,0.30)',
    'warning_bg':      'rgba(154,103,0,0.10)',
    'warning_border':  'rgba(154,103,0,0.30)',
    'error_bg':        'rgba(207,34,46,0.10)',
    'error_border':    'rgba(207,34,46,0.30)',
    'purple_bg':       'rgba(130,80,223,0.10)',
    'purple_border':   'rgba(130,80,223,0.30)',
    'teal_bg':         'rgba(29,122,140,0.10)',
    'teal_border':     'rgba(29,122,140,0.30)',

    'primary_btn_text': '#ffffff',
    'pressed_alpha':    'rgba(0,0,0,0.04)',
}

PALETTES: dict[str, dict[str, str]] = {'dark': DARK, 'light': LIGHT}


# =====================================================================
#  Module-level "current palette" accessor for paint code.
# =====================================================================
_CURRENT: dict[str, str] = DARK


def CUR() -> dict[str, str]:
    """Return the active palette dict.

    Used by ``QStyledItemDelegate`` paint methods (which are not GUI
    widgets and don't subscribe to listeners). Updated whenever
    ``ThemeManager.apply`` is called.
    """
    return _CURRENT


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

    def __init__(self, app: QApplication, qss_path: Path | None = None):
        self._app = app
        self._template = Template(
            (qss_path or Path(__file__).parent / 'theme.qss').read_text(encoding='utf-8')
        )
        self._theme = 'dark'
        self._listeners: list[Callable[[dict], None]] = []

    # ------------------------------------------------------------- listeners
    def add_listener(self, fn: Callable[[dict], None]) -> None:
        """``fn(palette: dict)`` is called after every theme change."""
        self._listeners.append(fn)

    # ---------------------------------------------------------------- state
    @property
    def palette(self) -> dict[str, str]:
        return PALETTES[self._theme]

    @property
    def name(self) -> str:
        return self._theme

    # --------------------------------------------------------------- actions
    def apply(self, theme: str) -> None:
        if theme not in PALETTES:
            return
        global _CURRENT
        self._theme = theme
        pal = PALETTES[theme]
        _CURRENT = pal

        self._app.setStyleSheet(self._template.safe_substitute(**pal))
        self._update_qpalette(pal)

        for fn in self._listeners:
            try:
                fn(pal)
            except Exception as exc:  # pragma: no cover — never let a listener crash the app
                print(f'[theme listener] {exc}')

    def toggle(self) -> str:
        self.apply('light' if self._theme == 'dark' else 'dark')
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
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(pal['surface2']))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(pal['text']))
        self._app.setPalette(p)
