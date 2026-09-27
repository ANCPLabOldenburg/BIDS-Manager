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

from PyQt6 import sip
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
from .widgets.tree_click import toggle_on_click

log = logging.getLogger(__name__)


# Deep enough to walk ``<bids_parent>/<dataset>/sub-X/ses-Y/<datatype>/file``
# without dragging in absurdly nested derivatives.
_MAX_DEPTH = 6

#: Most directories to hand a ``QFileSystemWatcher``. See ``_sync_watcher``.
_MAX_WATCHED_DIRS = 2000

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
#: The worker-produced ``_TreeNode`` a folder item has not drawn yet.
_NODE_ROLE = Qt.ItemDataRole.UserRole + 3
#: Marks the single stand-in child that keeps the expander arrow alive.
_PLACEHOLDER_ROLE = Qt.ItemDataRole.UserRole + 4


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

    **Deliberately UNPARENTED, and that is the whole point.** Its lifetime is
    Python's: the pane holds one reference and every in-flight
    :class:`_ScanRunnable` holds another, so the C++ object outlives whichever
    of them goes first.

    Parented to the pane, as it used to be, the C++ object was destroyed with
    the pane while a pool thread could still be inside ``done.emit(...)``.
    Destroying a QObject on one thread while another emits its signal is
    undefined behaviour in Qt, not a Python-level error: signal emission takes
    the sender's connection mutex, and that mutex is part of the object being
    freed. Neither guard the emit used to carry could help. ``sip.isdeleted``
    followed by ``emit`` is two steps with a window between them, and a
    ``try/except RuntimeError`` catches a PyQt wrapper error, never a
    segmentation fault.

    Unparented, there is nothing to race: the object is alive for as long as
    anyone can emit through it, and when the PANE is destroyed PyQt drops the
    connection to its slot, so a late emit is a no-op rather than a crash.

    ``QueuedConnection`` is still enforced when wiring the slot, so the emit
    hops to the GUI event loop even when fired from the thread pool.
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
        # Safe without a guard, and the absence of one is deliberate. This
        # runnable holds a reference to the signals object, which is NOT
        # parented to the pane, so the C++ object cannot be destroyed while we
        # are inside the emit. If the pane has gone, PyQt has already dropped
        # the connection to its slot and this delivers to nobody.
        #
        # The previous ``sip.isdeleted`` check plus ``except RuntimeError``
        # looked like defence and was not: the check and the emit are two
        # steps with a window between them, and no Python except clause
        # catches the segmentation fault that emitting through a freed
        # QObject produces. The lifetime had to be fixed instead.
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
        # No parent. See _ScanSignals: parenting it to the pane put its C++
        # lifetime on the GUI thread while pool threads were still emitting
        # through it.
        self._scan_signals = _ScanSignals()
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
        self._tree.itemExpanded.connect(self._on_item_expanded)
        toggle_on_click(self._tree)
        from .theme_manager import scaled_px
        _tree_ico = scaled_px(icons.DEFAULT_TREE_ICON_SIZE)
        self._tree.setIconSize(QSize(_tree_ico, _tree_ico))
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
        """Render a finished scan, or drop it if the pane has gone.

        The scan runs on the global thread pool and comes back through a
        queued connection, so the emit is already sitting in the event loop
        when the pane is torn down. Delivering it then reaches a Python
        wrapper whose C++ tree is gone, and touching it raises
        ``RuntimeError: wrapped C/C++ object ... has been deleted``. The
        generation guard does not cover this: the generation is still current,
        the widget is just no longer there.

        Both a check AND a catch, and the catch is not belt-and-braces. The
        C++ tree is destroyed when its parent panel is, and the parent can be
        collected by Python's CYCLIC garbage collector, which runs on
        allocation, which means between any two statements in here. A check at
        the top is therefore necessary and provably not sufficient. That race
        is what made this look like a flake: it landed on whichever test
        happened to be running when the collector got round to the panel.
        """
        if sip.isdeleted(self) or sip.isdeleted(self._tree):
            return
        try:
            self._render_scan(generation, result)
        except RuntimeError:
            if not (sip.isdeleted(self) or sip.isdeleted(self._tree)):
                raise      # a real error, not a teardown; do not swallow it
            log.debug("output pane went away while rendering its scan")

    def _render_scan(
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
        self._sync_watcher(result.dirs_to_watch)

        pal = CUR()
        root_item = _render_node(result.root, pal)
        self._tree.addTopLevelItem(root_item)

        if snap is None:
            # First-time render: expand the root, which draws the subjects,
            # and stop there. Expanding the first level as well used to be
            # free because everything was drawn anyway; now that a folder is
            # drawn when it is opened, opening all of them on a 200-subject
            # dataset would put the cost straight back.
            root_item.setExpanded(True)
        else:
            # Subsequent rebuilds (watcher-triggered during convert, etc.) must
            # respect whatever the user had open -- including a collapsed root.
            # Do NOT force the root expanded here, or a refresh re-expands a
            # folder the user just collapsed.
            self._restore_state(snap)

        self._last_rendered_root = self._root

    def _clear_watcher(self) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)

    def _sync_watcher(self, dirs: list[str]) -> None:
        """Move the watch set to ``dirs`` by DIFFERENCE, not by rebuilding it.

        This was ``removePaths(everything)`` followed by
        ``addPaths(everything)``, and it is what actually froze the window
        during a conversion. Each path is a real registration with the OS
        (FSEvents, inotify, ``FindFirstChangeNotification``), so dropping and
        re-taking all of them costs in proportion to the number of
        DIRECTORIES in the dataset. Measured on a tree of 8,000 files in
        2,668 directories: 199 ms to add plus 50 ms to remove, every 500 ms
        for as long as the conversion kept writing files. The tree render
        everyone assumed was the problem was 5 ms of it.

        A conversion creates a handful of directories per tick, so the
        difference is nearly always a few paths and costs nothing.

        The cap is not tidiness: inotify has a per-user watch limit, and past
        it the registration fails silently and live refresh stops working
        with no message. Shallow directories are kept in preference to deep
        ones, because a new subject, session or datatype folder appearing is
        the change most worth seeing, and it appears near the top.
        """
        wanted = set(sorted(dirs, key=lambda p: (p.count(os.sep), p))
                     [:_MAX_WATCHED_DIRS])
        current = set(self._watcher.directories())
        gone = sorted(current - wanted)
        fresh = sorted(wanted - current)
        if gone:
            self._watcher.removePaths(gone)
        if fresh:
            self._watcher.addPaths(fresh)

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

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """Draw a folder the first time it is opened."""
        _fill_children(item, CUR())

    def _row_at(self, path: tuple) -> Optional[QTreeWidgetItem]:
        """The row named by a snapshot path, drawing the folders on the way.

        The target itself is NOT drawn here, and does not need to be: this
        pane does not block the tree's signals while it rebuilds, so the
        ``setExpanded`` that follows emits ``itemExpanded`` and the folder
        draws itself. The Editor's tree blocks them, which is why the same
        omission showed up there as a folder that came back empty.

        Fold state is left alone, so this can restore a view without
        changing one.
        """
        if not path:
            return None
        item = None
        for i in range(self._tree.topLevelItemCount()):
            if self._tree.topLevelItem(i).text(0) == path[0]:
                item = self._tree.topLevelItem(i)
                break
        if item is None:
            return None
        for name in path[1:]:
            _fill_children(item, CUR())
            nxt = None
            for i in range(item.childCount()):
                if item.child(i).text(0) == name:
                    nxt = item.child(i)
                    break
            if nxt is None:
                return None
            item = nxt
        return item

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

        # Draw the rows the snapshot names, before touching any fold state.
        # A folder is drawn when it is opened, so a row that was expanded
        # under a CLOSED parent has no row to restore onto until its
        # ancestors are built, and walking the tree could not reach it.
        # Shallowest first, so each descent starts from rows that exist.
        for path in sorted(expanded, key=len):
            self._row_at(path)
        if selected:
            self._row_at(selected)

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
    """Translate ONE worker-produced ``_TreeNode`` into a ``QTreeWidgetItem``.

    Only the node itself. A folder gets its ``_TreeNode`` stamped on the item
    and a placeholder child, so the expander arrow is there and the real
    children are built by :meth:`OutputFsPane._on_item_expanded` when it is
    clicked.

    It used to render the whole subtree here, which meant one
    ``QTreeWidgetItem`` and one icon lookup per file in the dataset, on the
    GUI thread. Measured: 96 ms at 3,000 files, 257 ms at 8,000. The walk was
    already on the thread pool, so the pane LOOKED threaded; the part that
    blocked was the part that cannot leave the GUI thread, because Qt widgets
    may only be touched there. And the file watcher re-fires the whole render
    every 500 ms for as long as a conversion keeps writing files, so that
    cost was paid again and again exactly when the user was watching.

    The palette token is stamped onto the item so
    :meth:`OutputFsPane.repaint_for_palette` can re-color it later without
    re-walking disk.
    """
    item = QTreeWidgetItem([node.name])
    item.setData(0, _COLOR_TOKEN_ROLE, node.color_token)
    item.setData(0, _IS_DIR_ROLE, bool(node.is_dir))
    item.setForeground(0, QColor(pal[node.color_token]))
    item.setIcon(0, icons.icon_for_path(node.name, is_dir=node.is_dir))
    if node.is_dir and node.children:
        item.setData(0, _NODE_ROLE, node)
        placeholder = QTreeWidgetItem([""])
        placeholder.setData(0, _PLACEHOLDER_ROLE, True)
        item.addChild(placeholder)
    return item


def _fill_children(item: QTreeWidgetItem, pal: dict) -> None:
    """Build one level under ``item``, if it is still a placeholder."""
    if item.childCount() != 1:
        return
    child = item.child(0)
    if child.data(0, _PLACEHOLDER_ROLE) is not True:
        return
    node = item.data(0, _NODE_ROLE)
    item.removeChild(child)
    if node is None:
        return
    for grandchild in node.children:
        item.addChild(_render_node(grandchild, pal))


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
