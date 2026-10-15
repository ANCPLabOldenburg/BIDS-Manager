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

from .theme_manager import CUR, scaled_px


# Baseline render sizes in logical pixels. The actual size at runtime is
# this constant multiplied by the active ``FONT_SCALE()`` (see
# ``apply_button`` / ``apply_tab`` which call ``_scaled``). Qt's high-DPI
# scaling kicks in on top, so 20 logical px → 40 physical px on a 2x
# display.
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
    # The magnifier at the start of a search field. Muted: it labels the box,
    # it is not a button.
    "search":       ("mdi6.magnify",                  "muted"),
    "settings":     ("mdi6.cog-outline",              "text"),
    # A toolbox, not the settings cog: they are different things and
    # sharing a glyph made the Tools menu read as a second Settings.
    "tools":        ("mdi6.toolbox-outline",          "accent"),
    "run":          ("mdi6.play",                     "success"),
    "stop":         ("mdi6.stop-circle-outline",      "error"),
    "bulk_edit":    ("mdi6.pencil-outline",           "text"),
    "columns":      ("mdi6.view-column-outline",      "text"),
    "highlight":    ("mdi6.flag-outline",             "warning"),
    "undo":         ("mdi6.undo-variant",             "text"),
    "redo":         ("mdi6.redo-variant",             "text"),
    "revalidate":   ("mdi6.refresh",                  "accent"),
    "project":      ("mdi6.folder-multiple-outline",  "accent"),

    # ---- Converter bottom tabs ----
    "log":          ("mdi6.text-box-outline",         "text"),
    "log_clear":    ("mdi6.broom",                    "dim"),
    "log_file":     ("mdi6.file-document-outline",    "dim"),
    "log_history":  ("mdi6.history",                  "dim"),
    "warning":      ("mdi6.alert-outline",            "warning"),
    "preview":      ("mdi6.eye-outline",              "accent"),
    "statistics":   ("mdi6.chart-bar",                "purple"),

    # ---- Editor toolbar ----
    "open_folder":  ("mdi6.folder-open-outline",      "accent"),
    "file_check":   ("mdi6.file-check-outline",       "accent"),
    "folder_check": ("mdi6.folder-search-outline",    "accent"),
    "dataset":      ("mdi6.database-search-outline",  "accent"),
    "strict":       ("mdi6.lightning-bolt-outline",   "warning"),

    # ---- Editor MEG/EEG recording viewer ----
    "load_signal":  ("mdi6.play-circle-outline",      "success"),
    "filter":       ("mdi6.tune-variant",             "accent"),
    # Filter / structure: hide the rows left out of the conversion.
    "hide_left_out": ("mdi6.eye-off-outline",         "text"),
    # Filter panel modes.
    "mode_structure":  ("mdi6.file-tree-outline",     "text"),
    "mode_recordings": ("mdi6.layers-triple-outline", "text"),
    "mode_preview":    ("mdi6.folder-eye-outline",    "text"),
    "resample":     ("mdi6.sine-wave",                "accent"),
    "psd":          ("mdi6.chart-bell-curve",         "purple"),
    "reset_view":   ("mdi6.backup-restore",           "text"),
    "close_data":   ("mdi6.close-box-outline",        "error"),
    "channels":     ("mdi6.format-list-checks",       "text"),
    "events":       ("mdi6.flag-outline",             "warning"),
    "waveform":     ("mdi6.chart-line-variant",       "text"),

    # ---- File-tree node icons (Converter raw + output, Editor BIDS) ----
    "tree_folder":  ("ph.folder-simple-light",        "tree_folder"),
    "tree_nifti":   ("ph.brain-light",                "tree_image"),
    "tree_json":    ("mdi6.code-json",                "tree_sidecar"),
    "tree_tsv":     ("mdi6.microsoft-excel",          "tree_table"),

    # ---- Validation chips (kept for future reuse) ----
    "ok":           ("mdi6.check-circle-outline",     "success"),
    "warn":         ("mdi6.alert-circle-outline",     "warning"),
    "err":          ("mdi6.close-circle-outline",     "error"),

    # ---- About / generic ----
    "info":         ("mdi6.information-outline",      "accent"),
    "close":        ("mdi6.close",                    "text"),
    "check":        ("mdi6.check",                    "success"),

    # ---- Collapsible / detachable panel frame ----
    # Folding a whole tree, as opposed to one panel. Double chevrons, because
    # at 15 px they are the only pair that stays legible and keeps its
    # direction: the "unfold" glyphs shrink to a smudge and the "collapse-all
    # / expand-all" pair is nearly the same drawing twice.
    "tree_collapse":  ("mdi6.chevron-double-up",      "dim"),
    "tree_expand":    ("mdi6.chevron-double-down",    "dim"),
    "panel_collapse": ("mdi6.chevron-down",           "text"),
    "panel_expand":   ("mdi6.chevron-right",          "text"),
    "chevron_left":   ("mdi6.chevron-left",           "text"),
    "detach":         ("mdi6.open-in-new",            "text"),
    # -- the viewers (bidsmgr.gui.viz) -----------------------------------
    "restore":        ("mdi6.restore",                "dim"),
    "layout_single":  ("mdi6.square-outline",         "text"),
    "layout_multi":   ("mdi6.view-grid-outline",      "text"),
    "layout_combo":   ("mdi6.view-dashboard-outline", "text"),
    "layout_3d":      ("mdi6.cube-outline",           "text"),
    "layout_hero":    ("mdi6.view-split-vertical",    "text"),
    "layout_mosaic":  ("mdi6.view-comfy-outline",     "text"),
    "timecourse":     ("mdi6.chart-line",             "text"),
    "qc":             ("mdi6.clipboard-pulse-outline", "accent"),
    "save":           ("mdi6.content-save-outline",   "text"),
    "controls":       ("mdi6.tune-vertical",          "text"),
    "shortcuts":      ("mdi6.keyboard-outline",       "text"),
    "play":           ("mdi6.play",                   "text"),
    "annotate":       ("mdi6.draw",                   "warning"),
    "zen":            ("mdi6.fullscreen",             "text"),
    "maximize":       ("mdi6.arrow-expand",           "text"),
    "camera":         ("mdi6.camera-outline",         "text"),
    "mosaic_figure":  ("mdi6.image-multiple-outline", "text"),
    "reattach":       ("mdi6.dock-window",            "text"),
    "dock_right":     ("mdi6.dock-right",             "text"),
    "dock_bottom":    ("mdi6.dock-bottom",            "text"),
    "chevron_up":     ("mdi6.chevron-up",             "text"),
    "chevron_down":   ("mdi6.chevron-down",           "text"),
    "chevron_right":  ("mdi6.chevron-right",          "text"),
    # Settings pages.
    "settings_bids":    ("mdi6.book-open-page-variant-outline", "text"),
    "settings_display": ("mdi6.palette-outline",                "text"),
    "settings_system":  ("mdi6.memory",                         "text"),
    "settings_rules":   ("mdi6.playlist-check",                 "text"),
    "settings_convert": ("mdi6.transfer-right",                 "text"),
    "settings_classify": ("mdi6.tag-search-outline",   "text"),
    "settings_viewer":  ("mdi6.monitor-eye",                    "text"),
    "tracks_scroll":  ("mdi6.arrow-expand-vertical",  "text"),
    "plots":          ("mdi6.chart-multiple",         "text"),
    "channel_types":  ("mdi6.filter-variant",         "text"),
    "more":           ("mdi6.dots-horizontal",        "text"),
    # DICOM header pane.
    "tree_dicom":     ("mdi6.radiology-box-outline",  "tree_image"),
    "identifies":     ("mdi6.account-alert-outline",  "warning"),
    "read_by_scan":   ("mdi6.target",                 "accent"),
    "compare":        ("mdi6.compare-horizontal",     "text"),
    "export":         ("mdi6.export-variant",         "text"),
    "copy":           ("mdi6.content-copy",           "text"),
    "mark_bad":       ("mdi6.cancel",                 "warning"),
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
        key = styled_key(name, default_key)
        color = pal.get(key, pal.get("text", "#e6edf3"))
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


# EEG / MEG / iEEG recording file extensions that get the "brain + signal"
# composite tree icon (mirrors the Editor viewer's openable file set; the
# directory formats .ds / .mff render as folders in the tree).
RECORDING_TREE_EXTS: tuple[str, ...] = (
    ".fif.gz", ".fif", ".con", ".sqd", ".kdf",
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".egi", ".mef", ".nwb",
)


def _recording_tree_icon() -> QIcon:
    """Composite brain (like NIfTI) + a small signal squiggle badge.

    Signals a recording the Editor can open in the time-series viewer.
    Cached + theme-tinted like the rest of the icon set.
    """
    pal = CUR()
    brain = pal.get("tree_recording", pal.get("text", "#e6edf3"))
    signal = pal.get("accent", "#58a6ff")
    key = ("__recording_tree__", f"{brain}|{signal}")
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        ico = qta.icon(
            "ph.brain-light", "mdi6.pulse",
            options=[
                {"color": brain},
                {"color": signal, "scale_factor": 0.62, "offset": (0.22, 0.26)},
            ],
        )
    except Exception as exc:  # pragma: no cover - fall back to plain brain
        log.debug("recording tree icon failed: %s", exc)
        ico = icon("tree_nifti")
    _CACHE[key] = ico
    return ico


#: Folder-shaped recordings (CTF ``.ds``, EGI ``.mff``): drawn and coloured as
#: recordings, not as folders.
FOLDER_RECORDING_SUFFIXES: tuple[str, ...] = (".ds", ".mff")


def color_token_for(name: str, *, is_dir: bool = False) -> str:
    """The palette token a file tree colours an entry with: the ``tree_*``
    token of its kind (folder, image, sidecar, table, recording, other),
    which the user can change in Settings > Display. One rule for every tree
    (the Editor's, the Converter's output tree, the BIDS preview)."""
    lower = name.lower()
    if is_dir and not lower.endswith(FOLDER_RECORDING_SUFFIXES):
        return "tree_folder"
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "tree_image"
    if lower.endswith(".json"):
        return "tree_sidecar"
    if lower.endswith(".tsv") or lower.endswith(".tsv.gz"):
        return "tree_table"
    if lower.endswith(FOLDER_RECORDING_SUFFIXES) or lower.endswith(RECORDING_TREE_EXTS):
        return "tree_recording"
    return "tree_other"


def icon_for_path(name: str, *, is_dir: bool = False) -> QIcon:
    """Return the tree-node ``QIcon`` for a file path / name.

    Used by the three file-system trees (Converter raw input, Converter
    output, Editor BIDS root) to give each node a consistent type icon.
    Unknown extensions return an empty ``QIcon`` rather than a default
    so generic files don't pick up a misleading glyph.
    """
    lower = name.lower()
    if is_dir:
        # Directory-shaped recordings (CTF .ds, EGI .mff) are recordings,
        # not plain folders - give them the brain + signal icon too.
        if lower.endswith(".ds") or lower.endswith(".mff"):
            return _recording_tree_icon()
        return icon("tree_folder")
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return icon("tree_nifti")
    if lower.endswith(".json"):
        return icon("tree_json")
    if lower.endswith(".tsv.gz") or lower.endswith(".tsv"):
        return icon("tree_tsv")
    if lower.endswith(RECORDING_TREE_EXTS):
        return _recording_tree_icon()
    if lower.endswith((".dcm", ".ima")):
        return icon("tree_dicom")
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


#: How action icons are coloured (``appearance.ICON_STYLES``), set by the
#: theme manager from the user's appearance.
_STYLE = "monochrome"
#: Icons whose colour is their meaning, the same in every style: a status,
#: a stop, an alert, the controls tab's own accent.
_MEANINGFUL = frozenset({"ok", "warn", "err", "warning", "check", "stop", "close_data",
                         "mark_bad", "info", "sun", "moon"})


def set_style(style: str) -> None:
    """Colour action icons "monochrome" (the text colour), in the "accent",
    or "colourful" (each its own default). Drops the cache."""
    global _STYLE
    if style != _STYLE:
        _STYLE = style
        _CACHE.clear()


def styled_key(name: str, default_key: str) -> str:
    """The palette token icon ``name`` is drawn in under the current style.
    A meaningful icon, a file-tree icon and a deliberately quiet one keep
    their own token in every style."""
    if name in _MEANINGFUL or default_key.startswith("tree_") or default_key in ("dim", "muted"):
        return default_key
    if _STYLE == "colourful":
        return default_key
    return "text" if _STYLE == "monochrome" else "accent"


def apply_button(
    btn: QPushButton,
    name: str,
    *,
    color: Optional[str] = None,
    size: int = DEFAULT_BUTTON_ICON_SIZE,
) -> None:
    """Set / re-set a button's icon from a logical name.

    The icon size is multiplied by the active ``FONT_SCALE()`` so the
    icons grow / shrink with the rest of the GUI when the user picks
    a different "Font scale" preset in Settings.
    """
    btn.setIcon(icon(name, color=color))
    scaled = scaled_px(size)
    btn.setIconSize(QSize(scaled, scaled))


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
    in its bar; safe to call per ``apply_tab`` because the value is
    idempotent for the same scale.
    """
    tabs.setTabIcon(index, icon(name, color=color))
    scaled = scaled_px(size)
    tabs.setIconSize(QSize(scaled, scaled))


__all__ = [
    "FOLDER_RECORDING_SUFFIXES",
    "NAMES",
    "RECORDING_TREE_EXTS",
    "color_token_for",
    "set_style",
    "styled_key",
    "DEFAULT_BUTTON_ICON_SIZE",
    "DEFAULT_TAB_ICON_SIZE",
    "DEFAULT_TREE_ICON_SIZE",
    "icon",
    "icon_for_path",
    "refresh_for_palette",
    "apply_button",
    "apply_tab",
]
