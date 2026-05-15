"""Centralized icon helpers backed by `qtawesome` (Material Design Icons 6).

Why a wrapper
=============

Qt apps that put emoji glyphs (``☀``, ``📁``, ``⚙``...) inside button
text fall back to whichever emoji font the OS ships. macOS picks
*Apple Color Emoji*, Windows picks *Segoe UI Emoji*, Linux depends on
whatever was installed alongside fontconfig. The same glyph then
renders three different sizes / weights / colors — exactly the
inconsistency the bottom-right button + tab labels showed.

qtawesome ships its own MDI6 / Font Awesome TTF fonts inside the
wheel and registers them with Qt's font subsystem on import. From
then on every glyph is rendered from a bundled font, identically on
every OS, scaling cleanly with the OS DPI factor.

API
===

* :func:`icon` — return a ``QIcon`` for a logical name, colored to the
  current theme. Cached.
* :func:`refresh_for_palette` — call from a ``ThemeManager`` listener
  when the palette swaps. Drops the cache so the next ``icon()`` call
  rebuilds with new colors.
* :func:`apply_button` / :func:`apply_tab` — convenience helpers that
  set an icon on a ``QPushButton`` / a tab index. Useful so widgets
  that need to refresh on theme change don't have to duplicate the
  ``btn.setIcon(icon(...))`` boilerplate twice.

Logical names
=============

Every call site uses a stable logical name (``"scan"``, ``"settings"``,
``"run"``, ``"warning"``...) rather than a raw qtawesome string. That
way swapping icon packs (MDI → Font Awesome → Phosphor) is a one-line
change in :data:`NAMES`, and call sites stay readable.
"""

from __future__ import annotations

import logging
from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton, QTabWidget

from .theme_manager import CUR


# Default render sizes in logical pixels. Qt's high-DPI scaling kicks
# in on top of these, so 20 logical px → 40 physical px on a 2x display.
# 20 px sits well next to a 13 px text run; 18 px keeps tab strips tidy.
DEFAULT_BUTTON_ICON_SIZE = 20
DEFAULT_TAB_ICON_SIZE = 18
DEFAULT_TREE_ICON_SIZE = 16

log = logging.getLogger(__name__)


# Logical-name → (qtawesome glyph, default palette-key for the tint).
# The palette-key is looked up against the current theme dict (see
# :func:`bidsmgr.gui.theme_manager.CUR`) so colors swap with the theme.
# Adding a new icon is one entry here + ``icons.icon("name")`` at the
# use site (or ``icons.apply_button`` / ``icons.apply_tab``).
NAMES: dict[str, tuple[str, str]] = {
    # ---- Top header / theme toggle ----
    "sun":          ("mdi6.white-balance-sunny",      "text"),
    "moon":         ("ri.moon-fill",                  "text"),

    # ---- Converter toolbar ----
    "scan":         ("mdi6.magnify-scan",             "accent"),
    "settings":     ("mdi6.cog-outline",              "text"),
    "run":          ("mdi6.play",                     "success"),
    "bulk_edit":    ("mdi6.pencil-outline",           "text"),
    "highlight":    ("mdi6.flag-outline",             "warning"),

    # ---- Converter bottom tabs ----
    "log":          ("mdi6.text-box-outline",         "text"),
    "warning":      ("mdi6.alert-outline",            "warning"),
    "preview":      ("mdi6.eye-outline",              "accent"),
    "statistics":   ("mdi6.chart-bar",                "purple"),

    # ---- Editor toolbar ----
    "open_folder":  ("mdi6.folder-open-outline",      "accent"),
    "file_check":   ("mdi6.file-check-outline",       "accent"),
    "folder_check": ("mdi6.folder-search-outline",    "accent"),
    "dataset":      ("mdi6.database-search-outline",  "accent"),
    "strict":       ("mdi6.lightning-bolt-outline",   "warning"),

    # ---- File-tree node icons (Converter raw + output, Editor BIDS) ----
    "tree_folder":  ("ph.folder-simple-light",        "accent"),
    "tree_nifti":   ("ph.brain-light",                "text"),
    "tree_json":    ("mdi6.code-json",                "purple"),
    "tree_tsv":     ("mdi6.microsoft-excel",          "teal"),

    # ---- Validation chips (kept for future reuse) ----
    "ok":           ("mdi6.check-circle-outline",     "success"),
    "warn":         ("mdi6.alert-circle-outline",     "warning"),
    "err":          ("mdi6.close-circle-outline",     "error"),

    # ---- About / generic ----
    "info":         ("mdi6.information-outline",      "accent"),
    "close":        ("mdi6.close",                    "text"),
    "check":        ("mdi6.check",                    "success"),
}


# Cache keyed by (name, color). Theme swaps invalidate via
# ``refresh_for_palette``.
_CACHE: dict[tuple[str, str], QIcon] = {}


def icon(name: str, color: Optional[str] = None) -> QIcon:
    """Return a themed ``QIcon`` for *name*.

    Parameters
    ----------
    name:
        Logical name from :data:`NAMES`. Unknown names return an empty
        ``QIcon`` (loud-fail would crash the GUI; missing icons just
        render as no-icon).
    color:
        Hex / rgb color string to tint the icon. ``None`` (the default)
        resolves the icon's default palette-key against the current
        theme so accents stay consistent across dark / light swaps.
    """
    entry = NAMES.get(name)
    if entry is None:
        log.debug("icons.icon(%r): unknown name", name)
        return QIcon()
    glyph, default_key = entry
    if color is None:
        pal = CUR()
        color = pal.get(default_key, pal.get("text", "#e6edf3"))
    key = (name, color)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        ico = qta.icon(glyph, color=color)
    except Exception as exc:
        log.debug("icons.icon(%r, color=%r) failed: %s", name, color, exc)
        return QIcon()
    _CACHE[key] = ico
    return ico


def icon_for_path(name: str, *, is_dir: bool = False) -> QIcon:
    """Return the tree-node ``QIcon`` for a file path / name.

    Used by the three file-system trees (Converter raw input, Converter
    output, Editor BIDS root) to give each node a consistent type icon.
    Unknown extensions return an empty ``QIcon`` rather than a default
    so generic files don't pick up a misleading glyph.
    """
    if is_dir:
        return icon("tree_folder")
    lower = name.lower()
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return icon("tree_nifti")
    if lower.endswith(".json"):
        return icon("tree_json")
    if lower.endswith(".tsv.gz") or lower.endswith(".tsv"):
        return icon("tree_tsv")
    return QIcon()


def refresh_for_palette(_pal: dict) -> None:
    """Drop the icon cache so the next :func:`icon` call rebuilds tinted.

    Wire as a ``ThemeManager`` listener — the cache is keyed by
    ``(name, color)`` so it is safe to clear in full on every swap.
    Existing ``QIcon`` instances held by buttons stay valid (Qt does
    not invalidate them); call sites that want re-tinted icons need to
    re-set them in their ``repaint_for_palette`` hook.
    """
    _CACHE.clear()


def apply_button(
    btn: QPushButton,
    name: str,
    *,
    color: Optional[str] = None,
    size: int = DEFAULT_BUTTON_ICON_SIZE,
) -> None:
    """Set / re-set a button's icon from a logical name.

    Also calls ``setIconSize`` so the rendered glyph is consistently
    sized across the GUI (Qt's default of 16x16 looks small next to
    the QSS-styled toolbar text). High-DPI scaling applies on top.
    """
    btn.setIcon(icon(name, color=color))
    btn.setIconSize(QSize(size, size))


def apply_tab(
    tabs: QTabWidget,
    index: int,
    name: str,
    *,
    color: Optional[str] = None,
    size: int = DEFAULT_TAB_ICON_SIZE,
) -> None:
    """Set / re-set a tab's icon from a logical name.

    Calling ``setIconSize`` on a ``QTabWidget`` sizes every tab's icon
    in its bar; it is safe to invoke per ``apply_tab`` because it is
    idempotent for the same size value.
    """
    tabs.setTabIcon(index, icon(name, color=color))
    tabs.setIconSize(QSize(size, size))


__all__ = [
    "NAMES",
    "DEFAULT_BUTTON_ICON_SIZE",
    "DEFAULT_TAB_ICON_SIZE",
    "DEFAULT_TREE_ICON_SIZE",
    "icon",
    "icon_for_path",
    "refresh_for_palette",
    "apply_button",
    "apply_tab",
]
