"""Raw filesystem tree (column 1 of the Converter view).

Reference: ``inspector_proto/proto.py`` lines 543-571.

Walks the raw-input directory and shows a folder / file tree. Used as
visual orientation: which files were on disk vs. which became
inventory rows. Skipped sequences (``include=0`` or unmatched by any
scanner) are visually de-emphasised so the user can tell the scanner
saw them but chose not to convert.

Sync with the inventory model is one-way and best-effort: when the
table's selected row points at a known source file, the matching tree
node is highlighted. The tree never writes back into the model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .models import InventoryTableModel
from .theme_manager import CUR
from .widgets import PaneHeader

log = logging.getLogger(__name__)


# Hard cap on directory depth so a pathological input doesn't lock the
# UI in a recursive walk. v0.2.5 datasets rarely exceed 3 levels;
# DICOM dumps with one folder per subject + one per series sit at 2.
_MAX_DEPTH = 4

# Skip junk directories that pollute the tree without carrying any
# scanner-relevant content.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", "__pycache__",
    ".bidsmgr",  # the post-conv error / validation tree
    ".tmp", ".tmp_bidsmgr",
    "node_modules", ".idea", ".vscode",
})


class RawFsPane(QWidget):
    """Filesystem tree of the raw-input directory.

    Construct, then call :meth:`set_root` to point it at a folder. The
    tree refreshes lazily — calling :meth:`set_root` with the same path
    is a no-op so the bound view can call it whenever the model
    reloads without forcing a redraw.

    Pass an inventory model via :meth:`bind_model` so the tree can mark
    rows the scanner kept vs. skipped.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self.setMinimumWidth(200)

        self._root: Optional[Path] = None
        self._model: Optional[InventoryTableModel] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Raw data tree"))

        self._tree = QTreeWidget()
        self._tree.setObjectName("raw-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(True)
        from .theme_manager import scaled_px
        _tree_ico = scaled_px(icons.DEFAULT_TREE_ICON_SIZE)
        self._tree.setIconSize(QSize(_tree_ico, _tree_ico))
        v.addWidget(self._tree, 1)

        self._empty = QLabel("(pick a raw-data folder to populate this tree)")
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        v.addWidget(self._empty)
        self._empty.setVisible(True)
        self._tree.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, root: Optional[Path]) -> None:
        """Switch the displayed root. Pass ``None`` to clear."""
        if root is None:
            self._root = None
            self._tree.clear()
            self._tree.setVisible(False)
            self._empty.setVisible(True)
            return
        root = Path(root)
        if root == self._root:
            return
        self._root = root
        self._rebuild()

    def repaint_for_palette(self, _pal: dict) -> None:
        """Re-render the tree so per-item foreground colors update.

        Tree items have palette-derived foregrounds (dir = accent,
        kept files = text, skipped files = muted with strikethrough).
        Those colors are written into ``QTreeWidgetItem``s at populate
        time, so a fresh palette needs a re-populate to flow through.
        """
        if self._root is not None:
            self._rebuild()

    def bind_model(self, model: Optional[InventoryTableModel]) -> None:
        """Attach an inventory model for include / skip annotations.

        When attached, the tree marks file rows whose ``source_file``
        path appears in the model with included / skipped style.
        Calling with ``None`` clears the annotation.
        """
        self._model = model
        if self._root is not None:
            self._rebuild()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        self._tree.clear()
        if self._root is None or not self._root.exists():
            self._empty.setVisible(True)
            self._tree.setVisible(False)
            return
        self._empty.setVisible(False)
        self._tree.setVisible(True)

        pal = CUR()
        kept_paths, skipped_paths = self._index_model_paths()

        root_item = QTreeWidgetItem([self._root.name])
        root_item.setForeground(0, QColor(pal["text"]))
        root_item.setIcon(0, icons.icon_for_path(self._root.name, is_dir=True))
        self._tree.addTopLevelItem(root_item)
        self._populate(self._root, root_item, depth=0,
                       kept=kept_paths, skipped=skipped_paths)
        root_item.setExpanded(True)
        # Auto-expand first level so the user immediately sees subjects.
        for i in range(root_item.childCount()):
            root_item.child(i).setExpanded(True)

    def _populate(
        self,
        folder: Path,
        parent: QTreeWidgetItem,
        *,
        depth: int,
        kept: set[str],
        skipped: set[str],
    ) -> None:
        if depth >= _MAX_DEPTH:
            return
        try:
            entries = sorted(
                os.scandir(folder),
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
            child = QTreeWidgetItem([entry.name])
            parent.addChild(child)
            child.setIcon(0, icons.icon_for_path(entry.name, is_dir=entry.is_dir()))
            if entry.is_dir():
                child.setForeground(0, QColor(pal["accent"]))
                self._populate(
                    Path(entry.path), child, depth=depth + 1,
                    kept=kept, skipped=skipped,
                )
            else:
                abs_path = str(Path(entry.path).resolve())
                if abs_path in skipped:
                    # Visually mark as skipped — struck-through + muted.
                    f = QFont(child.font(0))
                    f.setStrikeOut(True)
                    child.setFont(0, f)
                    child.setForeground(0, QColor(pal["muted"]))
                elif abs_path in kept:
                    child.setForeground(0, QColor(pal["text"]))
                else:
                    child.setForeground(0, QColor(pal["dim"]))

    def _index_model_paths(self) -> tuple[set[str], set[str]]:
        """Return ``(kept_abs_paths, skipped_abs_paths)`` from the model.

        Only EEG/MEG rows publish ``source_file``; MRI rows reference
        DICOMs via ``series_uid`` and the on-disk file map is the scan
        sidecar — we don't have a clean per-row path to highlight there.
        For MRI we just don't annotate; the user can use the inspection
        table for kept/skipped status.
        """
        if self._model is None:
            return (set(), set())
        df = self._model.dataframe()
        if "source_file" not in df.columns:
            return (set(), set())
        kept: set[str] = set()
        skipped: set[str] = set()
        include_col = df.get("include")
        for idx in df.index:
            src = str(df.at[idx, "source_file"] or "")
            if not src:
                continue
            try:
                abs_src = str(Path(src).resolve())
            except (OSError, RuntimeError):
                continue
            included = True
            if include_col is not None:
                v = include_col.iloc[idx]
                if isinstance(v, str):
                    included = v.strip() not in ("0", "false", "False", "")
                else:
                    try:
                        included = bool(v)
                    except Exception:
                        included = True
            (kept if included else skipped).add(abs_src)
        return kept, skipped


__all__ = ["RawFsPane"]
