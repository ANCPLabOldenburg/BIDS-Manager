"""BIDS output filesystem tree (lower half of column 1).

Companion to :class:`bidsmgr.gui.raw_fs_pane.RawFsPane`. Walks the
``<bids_parent>/`` folder the user picked in the BIDS output path bar
and renders its contents with BIDS-aware coloring (dirs = accent,
``.nii.gz`` = text, ``.json`` = purple, ``.tsv`` = teal, other =
dim). Refreshes whenever the user picks a new output dir or a
conversion finishes.

No model coupling — the output tree is purely "what's on disk under
the BIDS root". The user sees the converted layout grow as workers
finish.

Threading: the recursive ``os.scandir`` walk runs on the global
``QThreadPool``. The GUI thread only does cheap work — building
``QTreeWidgetItem`` objects from the plain tree the worker produced,
registering watcher paths, and (de)serialising user state. A
generation counter on every scan request drops stale results, so
rapid fire-and-forget rebuilds (e.g. dcm2niix dropping many files
during a conversion) cannot interleave.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QFileSystemWatcher,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .theme_manager import CUR
from .widgets import PaneHeader

log = logging.getLogger(__name__)


# Deep enough to walk ``<bids_parent>/<dataset>/sub-X/ses-Y/<datatype>/file``
# without dragging in absurdly nested derivatives.
_MAX_DEPTH = 6

# Junk / scratch dirs the BIDS output may carry that we don't want to
# clutter the visualisation with.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", "__pycache__",
    ".tmp", ".tmp_bidsmgr",
    "node_modules", ".idea", ".vscode",
})

# Item data role used to remember the palette token a leaf was rendered
# with, so theme toggles can re-color in place without re-walking disk.
_COLOR_TOKEN_ROLE = Qt.ItemDataRole.UserRole + 1
_IS_DIR_ROLE = Qt.ItemDataRole.UserRole + 2


def _color_token_for(path_name: str) -> str:
    """Pick the palette token used to color a leaf file.

    Mirrors the BIDS-preview tree in the bottom dock: nii.gz = text,
    json = purple, tsv = teal, anything else = dim.
    """
    lower = path_name.lower()
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "text"
    if lower.endswith(".json"):
        return "purple"
    if lower.endswith(".tsv") or lower.endswith(".tsv.gz"):
        return "teal"
    return "dim"


# ---------------------------------------------------------------------------
# Off-thread scan
# ---------------------------------------------------------------------------


@dataclass
class _TreeNode:
    """Plain-data representation of a folder/file produced off-thread."""

    name: str
    is_dir: bool
    color_token: str
    children: list["_TreeNode"] = field(default_factory=list)


@dataclass
class _ScanResult:
    root: _TreeNode
    dirs_to_watch: list[str]


class _ScanSignals(QObject):
    """Bridges a worker-thread scan back to the GUI thread.

    Held on the pane (parented for QObject lifetime). ``QueuedConnection``
    is enforced when wiring the slot so the emit hops to the GUI event
    loop even when fired from the thread pool.
    """

    done = pyqtSignal(int, object)  # (generation, _ScanResult | None)


class _ScanRunnable(QRunnable):
    """Recursive directory walk that runs on the global thread pool.

    The walk produces a ``_TreeNode`` tree + the list of directory
    paths the GUI thread should register with the watcher. Any IO
    error short-circuits to ``None`` so the GUI thread can fall back
    to the empty-state hint.
    """

    def __init__(self, generation: int, root: Path, signals: _ScanSignals) -> None:
        super().__init__()
        self._generation = generation
        self._root = root
        self._signals = signals

    def run(self) -> None:
        result: Optional[_ScanResult]
        try:
            if not self._root.exists():
                result = None
            else:
                root_node = _TreeNode(
                    name=self._root.name or str(self._root),
                    is_dir=True,
                    color_token="text",
                )
                dirs: list[str] = [str(self._root)]
                _walk_dir(self._root, root_node, depth=0, dirs=dirs)
                result = _ScanResult(root=root_node, dirs_to_watch=dirs)
        except Exception:  # pragma: no cover — defensive
            log.exception("output tree scan failed for %s", self._root)
            result = None
        self._signals.done.emit(self._generation, result)


def _walk_dir(
    folder: Path,
    parent: _TreeNode,
    *,
    depth: int,
    dirs: list[str],
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
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name in _SKIP_DIRS:
            continue
        if entry.is_dir():
            node = _TreeNode(name=entry.name, is_dir=True, color_token="accent")
            parent.children.append(node)
            dirs.append(entry.path)
            _walk_dir(Path(entry.path), node, depth=depth + 1, dirs=dirs)
        else:
            node = _TreeNode(
                name=entry.name,
                is_dir=False,
                color_token=_color_token_for(entry.name),
            )
            parent.children.append(node)


# ---------------------------------------------------------------------------
# Pane
# ---------------------------------------------------------------------------


class OutputFsPane(QWidget):
    """Filesystem tree of the BIDS output directory.

    Construct, call :meth:`set_root` once a target path is known.
    Re-call :meth:`set_root` (or just :meth:`refresh`) after every
    conversion run so newly-produced files appear without an app
    restart. The actual disk walk runs on the global ``QThreadPool``;
    test code that needs to observe the resulting tree should poll
    with ``qtbot.waitUntil`` rather than asserting immediately.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self.setMinimumWidth(200)

        self._root: Optional[Path] = None
        self._last_rendered_root: Optional[Path] = None

        # Scan bookkeeping. ``_scan_generation`` increments on each
        # rebuild request; ``_on_scan_done`` drops any result whose
        # generation no longer matches (stale). ``_completed_scan_generation``
        # lets tests poll for completion without scraping the tree.
        self._scan_generation = 0
        self._completed_scan_generation = 0
        self._scan_in_progress = False
        self._scan_signals = _ScanSignals(self)
        self._scan_signals.done.connect(
            self._on_scan_done,
            Qt.ConnectionType.QueuedConnection,
        )

        # Live refresh: every visible directory is registered with a
        # ``QFileSystemWatcher`` so creates / deletes / renames trigger
        # a rebuild. Multiple rapid events (e.g. dcm2niix dropping many
        # files at once) are coalesced through a 500 ms debounce timer
        # to avoid thrashing the QTreeWidget.
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_changed)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._rebuild)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Output data tree"))

        self._tree = QTreeWidget()
        self._tree.setObjectName("raw-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(True)
        self._tree.setIconSize(QSize(
            icons.DEFAULT_TREE_ICON_SIZE, icons.DEFAULT_TREE_ICON_SIZE,
        ))
        v.addWidget(self._tree, 1)

        self._empty = QLabel(
            "(set a BIDS output folder; the tree fills in after each conversion)"
        )
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        v.addWidget(self._empty)

        # Start in "empty" state — no BIDS output picked yet.
        self._tree.setVisible(False)
        self._empty.setVisible(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, root: Optional[Path]) -> None:
        """Point the tree at ``root`` (the BIDS output parent dir).

        ``None`` clears the tree back to the empty state. Calling with
        the same path forces a refresh (so workers can re-populate
        after conversion completes). The scan itself runs on a
        worker thread; the tree updates when its result arrives.
        """
        self._root = Path(root) if root is not None else None
        self._rebuild()

    def refresh(self) -> None:
        """Re-walk the current root. No-op when no root is set."""
        self._rebuild()

    def repaint_for_palette(self, _pal: dict) -> None:
        """Re-color existing tree items for the new palette.

        Each item stores the palette token it was rendered with
        (see ``_COLOR_TOKEN_ROLE``). A theme toggle just walks the
        widget and rewrites foregrounds — no disk re-walk, no
        watcher churn, no flicker.
        """
        if self._tree.topLevelItemCount() == 0:
            return
        pal = CUR()
        for i in range(self._tree.topLevelItemCount()):
            _recolor(self._tree.topLevelItem(i), pal)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        # Empty-state and root-changed paths run synchronously so the
        # user never sees stale content from a different dataset.
        if self._root is None or not self._root.exists():
            self._tree.clear()
            self._clear_watcher()
            self._tree.setVisible(False)
            self._empty.setVisible(True)
            self._last_rendered_root = None
            # Advance generation + mark as completed so any in-flight
            # scan result is treated as stale and tests waiting on
            # quiescence wake up immediately.
            self._scan_generation += 1
            self._completed_scan_generation = self._scan_generation
            self._scan_in_progress = False
            return

        if self._last_rendered_root != self._root:
            # Different root than what's currently on screen — clear
            # immediately so the user doesn't see the previous tree
            # while the new scan runs.
            self._tree.clear()
            self._clear_watcher()

        self._empty.setVisible(False)
        self._tree.setVisible(True)

        self._scan_generation += 1
        self._scan_in_progress = True
        runnable = _ScanRunnable(
            self._scan_generation, self._root, self._scan_signals,
        )
        QThreadPool.globalInstance().start(runnable)

    def _on_scan_done(
        self,
        generation: int,
        result: Optional[_ScanResult],
    ) -> None:
        if generation != self._scan_generation:
            # Newer scan already queued — drop this one's output.
            return
        self._scan_in_progress = False
        self._completed_scan_generation = generation

        if result is None:
            # Root vanished between queueing and the scan running.
            self._tree.clear()
            self._clear_watcher()
            self._tree.setVisible(False)
            self._empty.setVisible(True)
            self._last_rendered_root = None
            return

        # Snapshot whatever interactive state the user has on the tree
        # before we blow it away. Without this, a watcher-triggered
        # refresh in the middle of e.g. expanding ``sub-001/anat`` would
        # collapse it back the moment a file lands. We restore the
        # snapshot after re-populating so the user's view is preserved.
        had_content = self._tree.topLevelItemCount() > 0
        snap = self._snapshot_state() if had_content else None

        self._tree.clear()
        self._clear_watcher()
        if result.dirs_to_watch:
            # ``addPaths`` is one syscall round-trip rather than one per dir.
            self._watcher.addPaths(result.dirs_to_watch)

        pal = CUR()
        root_item = _render_node(result.root, pal)
        self._tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)

        if snap is None:
            # First-time render: auto-expand the first level
            # (each ``<dataset>/`` folder) so the user immediately
            # sees the converted shape.
            for i in range(root_item.childCount()):
                root_item.child(i).setExpanded(True)
        else:
            # Subsequent rebuilds — defer to whatever the user had open.
            self._restore_state(snap)

        self._last_rendered_root = self._root

    def _clear_watcher(self) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)

    def _on_fs_changed(self, _path: str) -> None:
        """One or more watched dirs changed — schedule a debounced refresh.

        Many file events fire during a single conversion run (dcm2niix
        + sidecars + channels.tsv all land within a few ms). The timer
        coalesces them into one ``_rebuild`` so the tree doesn't flicker
        and we don't repeatedly re-add the same watches.
        """
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    # ------------------------------------------------------------------
    # User-state preservation across rebuilds
    # ------------------------------------------------------------------

    @staticmethod
    def _item_path(item: QTreeWidgetItem) -> tuple[str, ...]:
        """Tuple of item names from the top-level root down to ``item``.

        Used as a stable identity key across ``_tree.clear()`` →
        re-populate, since QTreeWidgetItem instances are destroyed and
        recreated on every rebuild but their text path stays the same.
        """
        parts: list[str] = []
        cur: Optional[QTreeWidgetItem] = item
        while cur is not None:
            parts.append(cur.text(0))
            cur = cur.parent()
        return tuple(reversed(parts))

    def _snapshot_state(self) -> dict:
        """Capture expanded paths + current selection + scroll position."""
        snap: dict = {
            "expanded": set(),
            "selected": None,
            "scroll": self._tree.verticalScrollBar().value(),
        }
        cur = self._tree.currentItem()
        if cur is not None:
            snap["selected"] = self._item_path(cur)

        def _walk(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                snap["expanded"].add(self._item_path(item))
            for i in range(item.childCount()):
                _walk(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk(self._tree.topLevelItem(i))
        return snap

    def _restore_state(self, snap: dict) -> None:
        """Re-apply ``snap`` onto the freshly-populated tree.

        Items whose path is in ``snap["expanded"]`` get re-expanded;
        the matching ``snap["selected"]`` becomes the current item; the
        vertical scrollbar is restored to its previous position.
        """
        expanded: set = snap["expanded"]
        selected = snap["selected"]

        def _walk(item: QTreeWidgetItem) -> None:
            path = self._item_path(item)
            if path in expanded:
                item.setExpanded(True)
            if selected is not None and path == selected:
                self._tree.setCurrentItem(item)
            for i in range(item.childCount()):
                _walk(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk(self._tree.topLevelItem(i))
        self._tree.verticalScrollBar().setValue(snap["scroll"])


def _render_node(node: _TreeNode, pal: dict) -> QTreeWidgetItem:
    """Translate a worker-produced ``_TreeNode`` into a ``QTreeWidgetItem``.

    The palette token used is also stamped onto the item so
    :meth:`OutputFsPane.repaint_for_palette` can re-color it later
    without re-walking disk.
    """
    item = QTreeWidgetItem([node.name])
    item.setData(0, _COLOR_TOKEN_ROLE, node.color_token)
    item.setData(0, _IS_DIR_ROLE, bool(node.is_dir))
    item.setForeground(0, QColor(pal[node.color_token]))
    item.setIcon(0, icons.icon_for_path(node.name, is_dir=node.is_dir))
    for child in node.children:
        item.addChild(_render_node(child, pal))
    return item


def _recolor(item: QTreeWidgetItem, pal: dict) -> None:
    token = item.data(0, _COLOR_TOKEN_ROLE) or "text"
    item.setForeground(0, QColor(pal[token]))
    # Re-tint the type icon as well. ``icons.icon_for_path`` reads from
    # the current palette via ``CUR()``; the cache was already cleared
    # by ``MainWindow._on_palette_changed`` before this listener fired.
    is_dir = bool(item.data(0, _IS_DIR_ROLE))
    item.setIcon(0, icons.icon_for_path(item.text(0), is_dir=is_dir))
    for i in range(item.childCount()):
        _recolor(item.child(i), pal)


__all__ = ["OutputFsPane"]
