"""BIDS dataset tree (Editor view, left pane).

Walks ``<bids_root>/sub-*/ses-*/<datatype>/`` and renders the result
as a ``QTreeWidget`` with BIDS-aware coloring:

* directories  → accent
* ``.nii(.gz)`` → text
* ``.json``    → purple
* ``.tsv(.gz)``→ teal
* other files  → dim

Folder-shaped recordings (``.ds`` for CTF MEG, ``.mff`` for EGI EEG)
collapse to a single leaf so the tree mirrors how BIDS treats them
(one recording = one node).

The :class:`BidsTreeDelegate` reads a per-row severity badge published
at ``BADGE_ROLE``. Step 2 leaves that empty; Step 3 fills it once the
validator runs.

Selection emits :pyattr:`file_selected` with the absolute path so
later steps (sidecar form, validation panel) can react.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QLabel,
    QStackedLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..delegates.bids_tree import BADGE_ROLE, BidsTreeDelegate
from ..theme_manager import CUR
from .primitives import PaneHeader

# Severity ordering for folder rollup — pick the worst of any descendant.
_SEVERITY_RANK: dict[str, int] = {"ok": 0, "warn": 1, "err": 2}

log = logging.getLogger(__name__)


# Walk depth cap. A BIDS path is at most
# ``<root>/sub-X/ses-Y/<datatype>/<file>`` (4 levels deep from root),
# plus a few extra for derivatives subtrees. Eight is a generous cap.
_MAX_DEPTH = 8

# Junk / scratch dirs we never want to expose in the tree.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", "__pycache__",
    ".tmp", ".tmp_bidsmgr", ".bidsmgr",
    "node_modules", ".idea", ".vscode",
})

# Directories whose contents are *one* BIDS recording — collapse them
# to a leaf node and stop recursing.
_FOLDER_RECORDING_SUFFIXES: tuple[str, ...] = (".ds", ".mff")

# Item data roles.
PATH_ROLE = Qt.ItemDataRole.UserRole          # absolute path string
COLOR_TOKEN_ROLE = Qt.ItemDataRole.UserRole + 1  # palette token for foreground


def _color_token_for(entry_name: str, is_dir: bool) -> str:
    """Pick the palette token used to color a row.

    Folder-recordings (``.ds`` / ``.mff``) are colored like data files
    (``text``) even though they're directories on disk, because the
    user thinks of them as recordings.
    """
    lower = entry_name.lower()
    if is_dir and not lower.endswith(_FOLDER_RECORDING_SUFFIXES):
        return "accent"
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "text"
    if lower.endswith(".json"):
        return "purple"
    if lower.endswith(".tsv") or lower.endswith(".tsv.gz"):
        return "teal"
    if lower.endswith(_FOLDER_RECORDING_SUFFIXES):
        # CTF .ds / EGI .mff — folder-shaped recordings.
        return "text"
    return "dim"


def _is_folder_recording(name: str) -> bool:
    return name.lower().endswith(_FOLDER_RECORDING_SUFFIXES)


def _walk(folder: Path, parent_item: QTreeWidgetItem, *, depth: int) -> None:
    """Populate ``parent_item`` with the contents of ``folder``."""
    if depth >= _MAX_DEPTH:
        return
    try:
        entries = sorted(
            os.scandir(folder),
            # Directories before files; within each group, case-insensitive.
            key=lambda e: (not e.is_dir(), e.name.lower()),
        )
    except (PermissionError, FileNotFoundError) as exc:
        log.debug("scandir failed for %s: %s", folder, exc)
        return

    pal = CUR()
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name in _SKIP_DIRS:
            continue
        is_dir = entry.is_dir()
        item = QTreeWidgetItem([entry.name])
        token = _color_token_for(entry.name, is_dir)
        item.setData(0, PATH_ROLE, entry.path)
        item.setData(0, COLOR_TOKEN_ROLE, token)
        item.setForeground(0, QColor(pal[token]))
        parent_item.addChild(item)
        # Recurse only into real directories that are not folder-recordings.
        if is_dir and not _is_folder_recording(entry.name):
            _walk(Path(entry.path), item, depth=depth + 1)


class BidsTreePane(QWidget):
    """Left pane of the Editor view — BIDS dataset tree.

    Construct, then call :meth:`set_root` once a directory is chosen.
    Emits :pyattr:`file_selected` (absolute :class:`Path`) when the
    user picks a leaf row.
    """

    file_selected = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._root: Optional[Path] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("BIDS dataset"))

        # Stack the tree on top of an empty-state hint; we flip between
        # them as the user opens / clears a root.
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._tree = QTreeWidget()
        self._tree.setObjectName("raw-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(14)
        self._tree.setItemDelegate(BidsTreeDelegate(self._tree))
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)

        self._hint = QLabel(
            "No BIDS dataset opened.\n\n"
            "Use “Open BIDS root…” in the toolbar."
        )
        self._hint.setObjectName("pane-hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)

        self._stack.addWidget(self._hint)   # index 0
        self._stack.addWidget(self._tree)   # index 1
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def root(self) -> Optional[Path]:
        return self._root

    def set_root(self, path: Optional[Path]) -> None:
        """Switch the tree to a new BIDS root (or clear it with ``None``).

        Repopulates synchronously. Datasets are small enough that the
        walk completes well under a frame on any modern disk; if that
        ever stops being true we can swap to the threadpool pattern
        ``OutputFsPane`` uses.
        """
        self._tree.clear()
        if path is None or not path.exists() or not path.is_dir():
            self._root = None
            self._stack.setCurrentIndex(0)
            return

        self._root = path
        # Top-level item carries the dataset name. We do NOT colour it
        # as a directory — the dataset root is the user's anchor, so
        # we paint it in the default text token.
        pal = CUR()
        top = QTreeWidgetItem([path.name or str(path)])
        top.setData(0, PATH_ROLE, str(path))
        top.setData(0, COLOR_TOKEN_ROLE, "text")
        top.setForeground(0, QColor(pal["text"]))
        self._tree.addTopLevelItem(top)
        _walk(path, top, depth=0)
        self._tree.expandToDepth(2)
        self._stack.setCurrentIndex(1)

    def refresh(self) -> None:
        """Re-walk the current root from disk. No-op if no root is set."""
        if self._root is not None:
            self.set_root(self._root)

    def set_badges(self, severities: dict[Path, str]) -> None:
        """Stamp per-row severity badges from a path → severity map.

        ``severities`` keys must be absolute paths matching the paths
        stored at :data:`PATH_ROLE`. Files not in the map get no badge.
        Directories receive the **rollup** (worst) severity across
        their visible descendants.
        """
        # Build a normalised lookup (string form) so we don't have to
        # construct ``Path`` for every tree item. We resolve every key
        # so macOS ``/private/var/...`` vs ``/var/...`` symlink
        # differences between the tree paths and the report paths
        # don't cause spurious misses.
        def _norm(p) -> str:
            try:
                return str(Path(p).resolve())
            except OSError:
                return str(Path(p))

        leaf_map = {_norm(p): s for p, s in severities.items()}

        def visit(item: QTreeWidgetItem) -> str | None:
            """Set this item's badge; return the rolled-up severity for
            propagation to the parent."""
            children_worst: str | None = None
            for i in range(item.childCount()):
                child_sev = visit(item.child(i))
                if child_sev is not None:
                    if children_worst is None or \
                            _SEVERITY_RANK[child_sev] > _SEVERITY_RANK[children_worst]:
                        children_worst = child_sev
            if item.childCount() > 0:
                # Directory: rollup-only badge (file-level wins over
                # implicit "ok" because we have no leaf severity to
                # represent the folder itself).
                badge = children_worst
            else:
                # Leaf: look up by absolute path (normalised).
                path_str = item.data(0, PATH_ROLE)
                badge = leaf_map.get(_norm(path_str)) if path_str else None
            if badge:
                item.setData(0, BADGE_ROLE, badge)
            else:
                item.setData(0, BADGE_ROLE, None)
            return badge

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        # Force the delegate to repaint with the new badge data.
        self._tree.viewport().update()

    def clear_badges(self) -> None:
        """Remove every badge from the tree."""
        def visit(item: QTreeWidgetItem) -> None:
            item.setData(0, BADGE_ROLE, None)
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        self._tree.viewport().update()

    # ------------------------------------------------------------------
    # Theme + selection
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-colour every row in place using the new palette.

        Iterates the tree without re-walking disk — each item carries
        its palette-token name at :data:`COLOR_TOKEN_ROLE` so we can
        just look the new color up.
        """
        def visit(item: QTreeWidgetItem) -> None:
            token = item.data(0, COLOR_TOKEN_ROLE)
            if token and token in pal:
                item.setForeground(0, QColor(pal[token]))
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        path_str = items[0].data(0, PATH_ROLE)
        if path_str:
            self.file_selected.emit(Path(path_str))


__all__ = ["BidsTreePane", "PATH_ROLE"]
