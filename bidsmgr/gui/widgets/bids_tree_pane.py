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
import platform
import subprocess
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QFileSystemWatcher,
    QPoint,
    QSize,
    Qt,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLabel,
    QMenu,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
    QStackedLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from ..delegates.bids_tree import (
    BADGE_ROLE,
    COUNT_ROLE,
    ISSUE_ROLE,
    BidsTreeDelegate,
)
from ..theme_manager import CUR
from .panel_frame import HEADER_EXTRAS
from .tree_click import toggle_on_click
from .primitives import PaneHeader

# Severity ordering for folder rollup — pick the worst of any descendant.
_SEVERITY_RANK: dict[str, int] = {"ok": 0, "warn": 1, "err": 2}

log = logging.getLogger(__name__)


# Walk depth cap. A BIDS path is at most
# ``<root>/sub-X/ses-Y/<datatype>/<file>`` (4 levels deep from root),
# plus a few extra for derivatives subtrees. Eight is a generous cap.
_MAX_DEPTH = 8

#: Most directories to hand a ``QFileSystemWatcher``. See ``_watch``.
_MAX_WATCHED_DIRS = 2000

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
#: Whether the row is a directory. Explicit, because a folder that has not
#: been opened yet has no children to infer it from.
IS_DIR_ROLE = Qt.ItemDataRole.UserRole + 6
#: Marks the single stand-in child that keeps a folder's expander arrow
#: alive until the folder is drawn.
PLACEHOLDER_ROLE = Qt.ItemDataRole.UserRole + 7
# COUNT_ROLE and ISSUE_ROLE are defined by the delegate that paints them and
# re-exported here, because every role the tree uses should be reachable from
# the tree.


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


def _norm_path(p) -> str:
    """Resolve a path to a comparable string.

    Resolving both sides means macOS ``/private/var/...`` vs ``/var/...``
    symlink differences between the tree paths and the validation report paths
    don't cause spurious badge misses.
    """
    try:
        return str(Path(p).resolve())
    except OSError:
        return str(Path(p))


def is_hidden_name(name: str) -> bool:
    """Would this entry be hidden with "Show hidden files" off?

    One source of truth for the filter and for the dimming, so a name can
    never be shown at full weight by one rule and hidden by the other.
    Dotfiles plus the machinery directories in :data:`_SKIP_DIRS`.
    """
    return name.startswith(".") or name in _SKIP_DIRS


def _is_hidden_item(item: QTreeWidgetItem) -> bool:
    """A row the tree shows only because "show hidden" is on.

    The fold and unfold buttons are about the DATASET. ``.bidsmgr/`` holds the
    operation log and every backup it has taken, which is a great many rows
    nobody asked to see, and somebody who opened it did so on purpose.
    """
    return is_hidden_name(item.text(0))


def _renameable_entities(path: Path) -> list[tuple[str, str, str]]:
    """Every entity the clicked row carries, as ``(key, value, label)``.

    A folder named ``sub-01`` offers the subject; a file named
    ``sub-01_task-rest_run-02_bold.nii.gz`` offers the task and the run as
    well. Offering only what is actually in front of the user is the
    difference between a menu that reads as an answer and one that reads as a
    form to fill in.

    The entity set and its display names come from the ACTIVE schema, in the
    schema's own filename order, so this tracks the BIDS version in force
    instead of whatever was true when the list was last hand-edited.
    """
    from ...editor.rename import entity_value
    from ...schema import entity_key_info, entity_keys

    name = path.name
    if "-" not in name:
        return []
    out: list[tuple[str, str, str]] = []
    for key in entity_keys():
        value = entity_value(name, key)
        if not value:
            continue
        try:
            label = entity_key_info(key).display_name.lower()
        except KeyError:
            label = key
        out.append((key, value, label))
    return out


def _in_session(path: Path) -> bool:
    """Is this row inside a ``ses-`` folder, or does it name one?"""
    return any(part.startswith("ses-") for part in path.parts)


def _can_restructure(path: Path) -> bool:
    """Is it meaningful to change the entities of what was clicked?

    Anything at or under a subject. Above that there is no recording to speak
    of, and the tool's own ``.bidsmgr/`` holds the operation log, which is the
    one thing a restructuring must not restructure.
    """
    parts = path.parts
    if any(is_hidden_name(part) for part in parts):
        return False
    return any(part.startswith("sub-") for part in parts)


def _sweep(
    folder: Path,
    *,
    depth: int,
    dirs: list[str],
    counts: dict[str, tuple[int, int, int]],
    show_hidden: bool,
) -> tuple[int, int]:
    """One scandir pass: every directory, and what is inside each.

    Returns ``(sessions, files)`` for ``folder`` and records
    ``(sessions, files, entries)`` in ``counts`` for every directory it
    meets. Creates no Qt objects at all, which is the point: the structure
    of the dataset is worth knowing in full, and the ROWS are not, because
    nobody reads eight thousand of them.

    ``entries`` is how many rows the folder would DRAW, which is not the
    same as how many files are under it: it decides whether the folder gets
    an expander arrow at all. Without it a folder holding nothing but
    dotfiles offered an arrow that opened onto an empty folder, which looks
    exactly like the bug where an open folder came back blank.

    A directory with nothing visible inside counts as one file, and a
    folder-recording (a CTF ``.ds``, an EGI ``.mff``) counts as one file
    rather than being descended into. Both match what the tree draws.
    """
    if depth >= _MAX_DEPTH:
        return (0, 1)
    try:
        entries = list(os.scandir(folder))
    except (PermissionError, FileNotFoundError) as exc:
        log.debug("scandir failed for %s: %s", folder, exc)
        return (0, 1)

    sessions = 0
    files = 0
    seen = 0
    for entry in entries:
        if is_hidden_name(entry.name) and not show_hidden:
            continue
        seen += 1
        if entry.is_dir() and not _is_folder_recording(entry.name):
            dirs.append(entry.path)
            sub_sessions, sub_files = _sweep(
                Path(entry.path), depth=depth + 1, dirs=dirs, counts=counts,
                show_hidden=show_hidden,
            )
            sessions += sub_sessions
            files += sub_files
            if entry.name.startswith("ses-") and sub_files:
                sessions += 1
        else:
            files += 1
    if not seen:
        files = 1
    counts[str(folder)] = (sessions, files, seen)
    return (sessions, files)


def _rollup(
    leaf_map: dict[str, str], count_map: dict[str, tuple[int, int]],
) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """Per-DIRECTORY worst severity and finding totals, from the file maps.

    Each finding is walked up its own ancestry, so a folder's answer does not
    depend on whether the folder has been drawn. A folder gets the SUM over
    its descendants and not the worst of them, because the point of a number
    is to say how much work is in there: a subject with one missing
    recommended field should not look like a subject with ninety.
    """
    badges: dict[str, str] = {}
    totals: dict[str, tuple[int, int]] = {}

    def ancestors(key: str):
        parent = os.path.dirname(key)
        while parent and parent != os.path.dirname(parent):
            yield parent
            parent = os.path.dirname(parent)

    for key, severity in leaf_map.items():
        rank = _SEVERITY_RANK.get(severity)
        if rank is None:
            continue
        for parent in ancestors(key):
            current = badges.get(parent)
            if current is None or rank > _SEVERITY_RANK[current]:
                badges[parent] = severity

    for key, (errors, warnings) in count_map.items():
        for parent in ancestors(key):
            have = totals.get(parent, (0, 0))
            totals[parent] = (have[0] + errors, have[1] + warnings)

    return badges, totals


def _annotate_folder(
    item: QTreeWidgetItem, name: str, counts: dict[str, tuple[int, int, int]],
) -> None:
    """Put what is inside a folder on the folder's own row.

    A subject with three sessions and forty-seven files is a fact a reader
    wants without expanding anything, and it is the difference between a tree
    you scan and a tree you excavate.

    Read from the sweep rather than counted from the children, because the
    children are no longer there: a folder is drawn when it is opened, and
    the whole value of the count is that it is on a folder you have NOT
    opened.
    """
    path = item.data(0, PATH_ROLE)
    sessions, files, _entries = counts.get(str(path), (0, 0, 0))
    if not files:
        return
    bits = []
    if sessions and name.startswith("sub-"):
        bits.append(f"{sessions} ses")
    bits.append(f"{files} file" + ("s" if files != 1 else ""))
    # A ROLE, not the text. The text is the folder's name and several things
    # look items up by it; painting the count in the delegate keeps the name
    # the name.
    item.setData(0, COUNT_ROLE, ", ".join(bits))


class BidsTreePane(QWidget):
    """Left pane of the Editor view — BIDS dataset tree.

    Construct, then call :meth:`set_root` once a directory is chosen.
    Emits :pyattr:`file_selected` (absolute :class:`Path`) when the
    user picks a leaf row.
    """

    file_selected = pyqtSignal(Path)
    # Every selected row, for actions that act on a set rather than a file.
    # ``file_selected`` still carries the one the panes should show, so the
    # centre pane's behaviour is unchanged by multi-select existing.
    selection_changed = pyqtSignal(list)
    # (entity, current value) for a rename the user started from the tree.
    # The pane does not open the dialog itself: the tree knows what was
    # clicked, the panel knows the dataset root and what to refresh after.
    # (entity, current value, the path that was clicked). The path is what
    # lets the dialog open on THAT recording rather than on every file in the
    # dataset carrying the same subject.
    rename_requested = pyqtSignal(str, str, str)
    # (paths, mode, session_mode) for adding or removing an entity, and for
    # creating or removing a session, on what the user right-clicked. Same
    # division of labour as ``rename_requested``: the tree knows what was
    # clicked, the panel owns the root and the refresh.
    entities_requested = pyqtSignal(list, str, bool)
    # The paths to delete. Same division of labour again: the tree knows what
    # was clicked, the panel owns the root, the dialog and the refresh.
    delete_requested = pyqtSignal(list)
    #: Remove faces from the clicked anatomical images. ``list[Path]``.
    deface_requested = pyqtSignal(list)
    #: Show one image before and after defacing. ``Path``, always a file.
    deface_compare_requested = pyqtSignal(object)
    #: Put the face back on the clicked images. ``list[Path]``.
    deface_revert_requested = pyqtSignal(list)
    #: Keep only the brain in the clicked images. ``list[Path]``.
    strip_requested = pyqtSignal(list)
    #: Put the clicked images side by side. ``list[Path]``, one or two.
    compare_requested = pyqtSignal(list)
    #: Open the references tool on the clicked file. ``list[Path]``, one file.
    links_requested = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._root: Optional[Path] = None
        # Remember the last severity badges so a live (watcher-driven) refresh
        # re-applies them onto the freshly-walked items instead of dropping
        # them. Keyed by normalised absolute path string.
        self._last_badges: dict[str, str] = {}
        # (errors, warnings) per file, kept for the same reason.
        self._last_counts: dict[str, tuple[int, int]] = {}
        # The same two, rolled up per DIRECTORY, so a folder drawn later gets
        # its badge without another pass over the tree.
        self._dir_badges: dict[str, str] = {}
        self._dir_counts: dict[str, tuple[int, int]] = {}
        # (sessions, files, entries) per directory, from the sweep: the
        # first two are the row label, the third decides the expander arrow.
        self._counts: dict[str, tuple[int, int, int]] = {}
        self._show_hidden = False

        # Live refresh: every visible directory is registered with a
        # ``QFileSystemWatcher`` so files created / deleted / renamed under the
        # BIDS root update the tree without a manual reopen (parity with the
        # Converter's output tree). Bursts of events (e.g. a conversion writing
        # into the open dataset) are coalesced through a 500 ms debounce.
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_changed)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # A header row rather than a bare label: a tree of any size needs a
        # way to get back to the top, and a way to go down one level at a
        # time. Expand-all on a real dataset produces thousands of rows and
        # is almost never what somebody wants.
        header_row = QWidget()
        header_line = QHBoxLayout(header_row)
        header_line.setContentsMargins(0, 0, 6, 0)
        header_line.setSpacing(4)
        # Stretch 0 plus an explicit stretch after it. When this pane is
        # wrapped in a ``PanelFrame`` the frame HIDES this header (it draws
        # its own title), and a hidden widget is ignored by the layout: with
        # the stretch on the label, the button group inherited the whole width
        # and rendered as two buttons the width of the pane.
        header_line.addWidget(PaneHeader("BIDS dataset"), 0)
        header_line.addStretch(1)

        # Two buttons in one group, because they are one control: fold the
        # tree, or open one more level of it. Separating them across the
        # header made them read as unrelated.
        #
        # Named ``HEADER_EXTRAS`` so that when this pane is wrapped in a
        # ``PanelFrame`` the pair is lifted up beside the frame's title,
        # rather than sitting on a row of its own under it. The group keeps a
        # plain ``tree-nav`` styling hook through a dynamic property, because
        # the object name is now spoken for.
        group = QWidget()
        group.setObjectName(HEADER_EXTRAS)
        group.setProperty("navGroup", True)
        group.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed,
        )
        group_line = QHBoxLayout(group)
        group_line.setContentsMargins(0, 0, 0, 0)
        group_line.setSpacing(0)

        self._collapse_btn = QPushButton()
        self._collapse_btn.setObjectName("tree-nav-left")
        icons.apply_button(self._collapse_btn, "tree_collapse", size=15)
        self._collapse_btn.setToolTip(
            "Collapse everything.\n\nBack to the top of a tree you have "
            "opened your way into."
        )
        self._collapse_btn.clicked.connect(self.collapse_all)
        group_line.addWidget(self._collapse_btn)

        self._expand_btn = QPushButton()
        self._expand_btn.setObjectName("tree-nav-right")
        icons.apply_button(self._expand_btn, "tree_expand", size=15)
        self._expand_btn.setToolTip(
            "Open the next level.\n\nOne click opens the shallowest depth "
            "that still has anything folded, so a deep tree is explored a "
            "level at a time instead of all at once. Hidden folders are not "
            "opened. Greyed out when everything is already open."
        )
        self._expand_btn.clicked.connect(self.expand_next_level)
        group_line.addWidget(self._expand_btn)

        # Sized here rather than in the QSS, because a button that is allowed
        # to grow WILL grow: these two are a fixed-size control, not text.
        from ..theme_manager import scaled_px
        for button in (self._collapse_btn, self._expand_btn):
            button.setFixedSize(scaled_px(30), scaled_px(22))
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        header_line.addWidget(group, 0)
        v.addWidget(header_row)

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
        from ..theme_manager import scaled_px
        _tree_ico = scaled_px(icons.DEFAULT_TREE_ICON_SIZE)
        self._tree.setIconSize(QSize(_tree_ico, _tree_ico))
        self._tree.setItemDelegate(BidsTreeDelegate(self._tree))
        # Multi-select: the prerequisite for anything that acts on a set of
        # files rather than one. Ctrl/Cmd-click adds, shift-click extends.
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemExpanded.connect(self._on_item_expanded)
        toggle_on_click(self._tree)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_show_context_menu)

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

    # -- opening and closing -------------------------------------------

    def collapse_all(self) -> None:
        """Fold the dataset, and leave the dataset row itself open.

        Collapsing the root as well would hide the whole tree behind one
        chevron, which is not "back to the top", it is "gone".

        Hidden folders fold along with everything else. The asymmetry with
        :meth:`expand_next_level` is deliberate and is about rows: opening
        ``.bidsmgr/`` produces hundreds nobody asked for, closing it produces
        none, and a "collapse all" that leaves something open is a lie.
        """
        def visit(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                visit(child)
                child.setExpanded(False)

        for i in range(self._tree.topLevelItemCount()):
            root = self._tree.topLevelItem(i)
            visit(root)
            root.setExpanded(True)
        self._refresh_expand_button()

    def expand_next_level(self) -> None:
        """Open the shallowest depth that still has anything folded.

        One level per click. Expand-all on a real dataset produces thousands
        of rows, and the useful gesture is almost always "show me one more
        step down".

        Hidden folders are not opened: ``.bidsmgr/`` holds the operation log
        and its backups, which is a great many rows nobody asked for.
        """
        depth = self._next_folded_depth()
        if depth is None:
            return

        def visit(item: QTreeWidgetItem, level: int) -> None:
            if level > depth or _is_hidden_item(item):
                return
            if item.childCount():
                item.setExpanded(True)
            for i in range(item.childCount()):
                visit(item.child(i), level + 1)

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i), 0)
        self._refresh_expand_button()

    def _next_folded_depth(self) -> Optional[int]:
        """The shallowest depth holding a folded folder, or ``None``."""
        best: Optional[int] = None

        def visit(item: QTreeWidgetItem, depth: int) -> None:
            nonlocal best
            if _is_hidden_item(item):
                return          # not ours to open
            if item.childCount() and not item.isExpanded():
                if best is None or depth < best:
                    best = depth
                return          # nothing under a folded node is reachable yet
            for i in range(item.childCount()):
                visit(item.child(i), depth + 1)

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i), 0)
        return best

    def _refresh_expand_button(self) -> None:
        """Grey out Expand when there is nothing left folded."""
        if hasattr(self, "_expand_btn"):
            self._expand_btn.setEnabled(self._next_folded_depth() is not None)

    def set_root(self, path: Optional[Path]) -> None:
        """Switch the tree to a new BIDS root (or clear it with ``None``).

        Re-setting the SAME root (e.g. the editor re-opening the dataset it is
        already showing) preserves the user's expansion / selection by routing
        through :meth:`refresh`, exactly like the Converter's output tree. Only
        a genuinely new root forgets badges + expansion and opens to depth 2.

        Repopulates synchronously. Datasets are small enough that the walk
        completes well under a frame on any modern disk; if that ever stops
        being true we can swap to the threadpool pattern ``OutputFsPane`` uses.
        """
        if (
            path is not None
            and self._root is not None
            and Path(path) == self._root
            and self._tree.topLevelItemCount() > 0
        ):
            # Same root, already rendered -> in-place refresh (keep the view).
            self.refresh()
            return

        self._clear_watcher()
        self._tree.clear()
        if path is None or not path.exists() or not path.is_dir():
            self._root = None
            self._last_badges = {}
            self._last_counts = {}
            self._stack.setCurrentIndex(0)
            return

        self._root = path
        # New root: stale badges and counts no longer apply to anything.
        self._last_badges = {}
        self._last_counts = {}
        dirs = self._populate(path)
        self._watch(dirs)
        # Opens shut, showing the dataset row and nothing else. Guessing at
        # two levels meant a dataset with many subjects opened as a wall of
        # rows the user then had to close, and it is the unfold button's job
        # to go down anyway.
        self.collapse_all()
        self._stack.setCurrentIndex(1)

    def refresh(self) -> None:
        """Re-walk the current root, preserving the user's view.

        Unlike :meth:`set_root`, this keeps whatever the user had expanded /
        selected (and the scroll position) and re-applies the last severity
        badges, so a live filesystem change does not collapse the tree or wipe
        validation markers. No-op if no root is set.
        """
        if self._root is None:
            return
        snap = self._snapshot_state()
        # Block selection signals so re-selecting the same row after the rebuild
        # does not spuriously reload the center viewer on every disk tick.
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            if not self._root.exists() or not self._root.is_dir():
                self._root = None
                self._stack.setCurrentIndex(0)
                return
            dirs = self._populate(self._root)
            self._watch(dirs)
            self._restore_state(snap)
        finally:
            self._tree.blockSignals(False)
        # Re-apply badges from the cache (validation has not re-run, but the
        # markers are still valid for files that survived the change).
        if self._last_badges:
            self._apply_badge_map(self._last_badges, self._last_counts)

    def _populate(self, path: Path) -> list[str]:
        """Sweep ``path``, draw its first level, return the dirs to watch.

        The sweep is the whole dataset and costs one scandir pass. The
        DRAWING is one level, and the rest is drawn as folders are opened.

        It used to build every row up front: one ``QTreeWidgetItem``, one
        icon lookup and two ``setData`` calls per file, on the GUI thread.
        Measured on 8,000 files: 259 ms to open a dataset and the same again
        every time the file watcher fired.
        """
        pal = CUR()
        # Top-level item carries the dataset name. We do NOT colour it
        # as a directory — the dataset root is the user's anchor, so
        # we paint it in the default text token.
        top = QTreeWidgetItem([path.name or str(path)])
        top.setData(0, PATH_ROLE, str(path))
        top.setData(0, COLOR_TOKEN_ROLE, "text")
        top.setData(0, IS_DIR_ROLE, True)
        top.setForeground(0, QColor(pal["text"]))
        top.setIcon(0, icons.icon_for_path(path.name or str(path), is_dir=True))
        self._tree.addTopLevelItem(top)

        dirs: list[str] = [str(path)]
        counts: dict[str, tuple[int, int, int]] = {}
        # Read the preference at build time rather than caching it, so a
        # change in Settings shows on the next refresh without extra wiring.
        from ..app_settings import AppSettings
        self._show_hidden = AppSettings.load().editor_show_hidden
        _sweep(
            path, depth=0, dirs=dirs, counts=counts,
            show_hidden=self._show_hidden,
        )
        self._counts = counts
        self._fill_level(top)
        self._stack.setCurrentIndex(1)
        return dirs

    def _fill_level(self, parent_item: QTreeWidgetItem) -> None:
        """Draw the contents of one folder, with the next level stubbed."""
        folder = parent_item.data(0, PATH_ROLE)
        if not folder:
            return
        try:
            entries = sorted(
                os.scandir(folder),
                # Directories before files; within each group, insensitive.
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except (PermissionError, FileNotFoundError) as exc:
            log.debug("scandir failed for %s: %s", folder, exc)
            return

        pal = CUR()
        for entry in entries:
            hidden = is_hidden_name(entry.name)
            if hidden and not self._show_hidden:
                continue
            is_dir = entry.is_dir()
            item = QTreeWidgetItem([entry.name])
            token = _color_token_for(entry.name, is_dir)
            item.setData(0, PATH_ROLE, entry.path)
            item.setData(0, COLOR_TOKEN_ROLE, token)
            item.setData(0, IS_DIR_ROLE, is_dir)
            if hidden:
                # Shown but dimmed. A dataset carries .bidsmgr/, .git/ and
                # .bidsignore, and none of them are the data: they should be
                # reachable without competing with the tree proper.
                item.setData(0, COLOR_TOKEN_ROLE, "muted")
                item.setForeground(0, QColor(pal["muted"]))
            else:
                item.setForeground(0, QColor(pal[token]))
            item.setIcon(0, icons.icon_for_path(entry.name, is_dir=is_dir))
            parent_item.addChild(item)
            if is_dir and not _is_folder_recording(entry.name):
                _annotate_folder(item, entry.name, self._counts)
                # No arrow on a folder with nothing to draw. Defaults to
                # offering one when the sweep has no answer (past its depth
                # cap, or unreadable), because hiding real content would be
                # worse than an arrow that opens onto nothing.
                if self._counts.get(entry.path, (0, 0, 1))[2]:
                    placeholder = QTreeWidgetItem([""])
                    placeholder.setData(0, PLACEHOLDER_ROLE, True)
                    item.addChild(placeholder)
            self._stamp_badge(item)

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """Draw a folder the first time it is opened."""
        self._ensure_children(item)

    def _ensure_children(self, item: QTreeWidgetItem) -> None:
        """Replace a folder's placeholder with its real contents.

        Called directly, never only through ``itemExpanded``. ``refresh``
        blocks the tree's signals while it rebuilds, so an expansion done in
        there emits nothing, and a tree restored under a blocked signal
        would come back as a row of stubs that never fill.
        """
        if item.childCount() != 1:
            return
        child = item.child(0)
        if child.data(0, PLACEHOLDER_ROLE) is not True:
            return
        item.removeChild(child)
        self._fill_level(item)

    def _row_for(self, target: str) -> Optional[QTreeWidgetItem]:
        """The row for ``target``, drawing the folders on the way to it.

        Needed because a folder is drawn when it is OPENED: a row deep in
        the dataset does not exist until something walks down to it. This
        creates the rows on the path and nothing else, and leaves every
        folder's fold state alone, so it can be used to restore a view
        without changing one.
        """
        if not target or self._tree.topLevelItemCount() == 0:
            return None
        item = self._tree.topLevelItem(0)
        root = str(item.data(0, PATH_ROLE) or "")
        if not root or not (target == root or target.startswith(root + os.sep)):
            return None
        while True:
            if str(item.data(0, PATH_ROLE) or "") == target:
                # The TARGET is drawn too, not only the folders on the way to
                # it. Stopping one short left a folder that was open before a
                # refresh expanded over its placeholder, which is a blank row,
                # and the only way to see the contents again was to close it
                # and open it. A file has nothing to draw, so this is a no-op
                # for the ``reveal`` case.
                self._ensure_children(item)
                return item
            self._ensure_children(item)
            nxt = None
            for i in range(item.childCount()):
                child = item.child(i)
                key = str(child.data(0, PATH_ROLE) or "")
                if key and (target == key or target.startswith(key + os.sep)):
                    nxt = child
                    break
            if nxt is None:
                return None
            item = nxt

    def reveal(self, path: Path) -> Optional[QTreeWidgetItem]:
        """Open the folders down to ``path`` and return its row."""
        item = self._row_for(str(Path(path)))
        if item is None:
            return None
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        return item

    # ------------------------------------------------------------------
    # Live refresh (QFileSystemWatcher)
    # ------------------------------------------------------------------

    def _watch(self, dirs: list[str]) -> None:
        """Move the watch set to ``dirs`` by DIFFERENCE, not by rebuilding it.

        A refresh used to drop every watch and take them all again. Each
        path is a real registration with the OS, so that cost in proportion
        to the number of DIRECTORIES in the dataset, on the GUI thread,
        every time anything on disk changed. Measured in the Converter's
        output pane, which had the identical pattern: 199 ms to add plus
        50 ms to remove on a tree of 2,668 directories.

        The cap is not tidiness. inotify has a per-user watch limit, and
        past it registration fails silently and live refresh stops working
        with no message. Shallow directories are preferred, because a new
        subject, session or datatype folder is the change most worth
        seeing and it appears near the top.
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

    def _clear_watcher(self) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)

    def _on_fs_changed(self, _path: str) -> None:
        """A watched directory changed — schedule one debounced refresh."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    # ------------------------------------------------------------------
    # View-state preservation across an in-place refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _item_key(item: QTreeWidgetItem) -> str:
        """Stable identity for a row across rebuilds (its absolute path)."""
        return str(item.data(0, PATH_ROLE) or "")

    def _snapshot_state(self) -> dict:
        """Capture expanded paths + current selection + scroll position."""
        snap: dict = {
            "expanded": set(),
            "selected": None,
            "scroll": self._tree.verticalScrollBar().value(),
        }
        cur = self._tree.currentItem()
        if cur is not None:
            snap["selected"] = self._item_key(cur)

        def _walk_items(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                snap["expanded"].add(self._item_key(item))
            for i in range(item.childCount()):
                _walk_items(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk_items(self._tree.topLevelItem(i))
        return snap

    def _restore_state(self, snap: dict) -> None:
        """Re-apply expansion / selection / scroll captured by a snapshot."""
        expanded: set = snap["expanded"]
        selected = snap["selected"]

        # Draw the rows the snapshot names, before touching any fold state.
        # A folder is drawn when it is opened, so a row that was expanded
        # under a CLOSED parent has no row to restore onto until its
        # ancestors are built, and walking the tree could not reach it.
        # Shallowest first, so each descent starts from rows that exist.
        for key in sorted(expanded, key=lambda k: k.count(os.sep)):
            self._row_for(key)
        if selected:
            self._row_for(selected)

        def _walk_items(item: QTreeWidgetItem) -> None:
            key = self._item_key(item)
            if key in expanded:
                item.setExpanded(True)
            if selected is not None and key == selected:
                self._tree.setCurrentItem(item)
            for i in range(item.childCount()):
                _walk_items(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            _walk_items(self._tree.topLevelItem(i))
        self._tree.verticalScrollBar().setValue(snap["scroll"])

    def set_badges(
        self,
        severities: dict[Path, str],
        counts: Optional[dict[Path, tuple[int, int]]] = None,
    ) -> None:
        """Stamp per-row severities, and how many findings each row has.

        ``severities`` keys must be absolute paths matching the paths
        stored at :data:`PATH_ROLE`. Files not in the map get no badge.

        ``counts`` maps the same paths to ``(errors, warnings)``. A folder
        gets the SUM over its descendants, not the worst of them: the point
        of a number is to say how much work is in there, and a subject with
        one missing recommended field should not look like a subject with
        ninety.
        """
        # Build a normalised lookup (string form) so we don't have to
        # construct ``Path`` for every tree item.
        leaf_map = {_norm_path(p): s for p, s in severities.items()}
        count_map = {
            _norm_path(p): (int(c[0]), int(c[1]))
            for p, c in (counts or {}).items()
        }
        # Remember so a live (watcher-driven) refresh can re-stamp the rebuilt
        # tree without re-running validation.
        self._last_badges = dict(leaf_map)
        self._last_counts = dict(count_map)
        self._apply_badge_map(leaf_map, count_map)

    def _apply_badge_map(
        self,
        leaf_map: dict[str, str],
        count_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        """Stamp a normalised path -> severity map onto the current tree.

        The folder rollup is computed from the MAP, by walking each finding's
        ancestors, and not from the tree's own children. It used to be the
        other way round, which stopped working the moment a folder was drawn
        only when opened: a closed subject has no children, so it would have
        rolled up to nothing, and a folder showing no findings because it has
        not been opened is precisely the wrong answer. Doing it from the map
        also means a row gets its badge as it is created, without another
        pass over the tree.
        """
        count_map = count_map or {}
        self._dir_badges, self._dir_counts = _rollup(leaf_map, count_map)

        def visit(item: QTreeWidgetItem) -> None:
            self._stamp_badge(item)
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))
        # Force the delegate to repaint with the new badge data.
        self._tree.viewport().update()

    def _stamp_badge(self, item: QTreeWidgetItem) -> None:
        """Put the badge and finding counts on one row, from the cached maps."""
        path_str = item.data(0, PATH_ROLE)
        if not path_str:
            return
        key = _norm_path(path_str)
        if item.data(0, IS_DIR_ROLE):
            badge = self._dir_badges.get(key)
            errors, warnings = self._dir_counts.get(key, (0, 0))
        else:
            badge = self._last_badges.get(key)
            errors, warnings = self._last_counts.get(key, (0, 0))
        item.setData(0, BADGE_ROLE, badge or None)
        item.setData(
            0, ISSUE_ROLE, (errors, warnings) if (errors or warnings) else None,
        )

    def clear_badges(self) -> None:
        """Remove every badge from the tree (and forget the cached map)."""
        self._last_badges = {}
        self._last_counts = {}

        def visit(item: QTreeWidgetItem) -> None:
            item.setData(0, BADGE_ROLE, None)
            item.setData(0, ISSUE_ROLE, None)
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
            # Re-tint the type icon from the cleared cache.
            path_str = item.data(0, PATH_ROLE) or item.text(0)
            try:
                is_dir = Path(path_str).is_dir() if path_str else False
            except OSError:
                is_dir = False
            item.setIcon(0, icons.icon_for_path(item.text(0), is_dir=is_dir))
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))

        # The fold/unfold pair is tinted from the palette too, and its icons
        # came out of the cache that was just cleared.
        icons.apply_button(self._collapse_btn, "tree_collapse", size=15)
        icons.apply_button(self._expand_btn, "tree_expand", size=15)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        self.selection_changed.emit(self.selected_paths())
        if not items:
            return
        # The FIRST selected row drives the viewer. Extending a selection with
        # shift should not make the centre pane flicker through every file on
        # the way, so only the anchor is published as the shown file.
        path_str = items[0].data(0, PATH_ROLE)
        if path_str:
            self.file_selected.emit(Path(path_str))

    def selected_paths(self) -> list[Path]:
        """Every selected row as a path, in tree order."""
        out: list[Path] = []
        for item in self._tree.selectedItems():
            value = item.data(0, PATH_ROLE)
            if value:
                out.append(Path(value))
        return out

    def _on_show_context_menu(self, position: QPoint) -> None:
        item = self._tree.itemAt(position)
        if item is None:
            return

        # Read the path now, while the item is guaranteed valid. ``menu.exec``
        # spins a nested event loop, during which the live-refresh timer may
        # fire and call ``self._tree.clear()``, deleting the underlying C++
        # item. A callback that captured ``item`` would then dereference a
        # freed object and raise ``RuntimeError``. Capturing the plain string
        # keeps the actions safe no matter what happens to the tree meanwhile.
        path = item.data(0, PATH_ROLE)
        if not path:
            return

        menu = QMenu(self)
        from ..combo_popup import menu_section, round_menu
        round_menu(menu)

        # Same groups, same order and same words as the Tools menu, so the
        # two ways of reaching a tool teach the same thing about what it is
        # for. A heading is only drawn when its group has an entry for what
        # was clicked, which is what keeps a right-click on a .json short.

        # Comparing is offered on ANY NIfTI, including one under
        # derivatives/, because putting a derivative beside the scan it came
        # from is the most common reason to want this. It sits above the
        # restructuring block deliberately: it changes nothing.
        selected_now = self.selected_paths()
        clicked_path = Path(path)
        images = [
            p for p in (selected_now if clicked_path in selected_now
                        and len(selected_now) > 1 else [clicked_path])
            if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
        ]
        if images:
            menu_section(menu, "Look at this")
            pair = len(images) > 1
            compare_images = menu.addAction(
                "Compare these two images..." if pair
                else "Compare with another image..."
            )
            compare_images.setToolTip(
                "Side by side with one set of controls: crosshair, slice, "
                "plane, 3-D camera, effects and cut plane stay together."
                + ("" if pair else " You pick the second image.")
            )
            compare_images.triggered.connect(
                lambda _c=False, s=images[:2]: self.compare_requested.emit(s)
            )

        # References, offered only where a link field is worth having: on a
        # fieldmap, an MEG recording, a derivative. Offering it on every
        # file would be true and useless.
        if self._root is not None and clicked_path.is_file():
            from ...editor import linkage as lk
            if lk.fields_for(self._root, clicked_path):
                menu_section(menu, "Check and repair")
                links = menu.addAction("References...")
                links.setToolTip(
                    "What this file points at, and what points at it. Opens "
                    "on this file with the rest of the dataset beside it, so "
                    "a link is made by ticking a file rather than typing a "
                    "path."
                )
                links.triggered.connect(
                    lambda _c=False, p=clicked_path:
                        self.links_requested.emit([p])
                )

        # Every entity the clicked row actually carries, so the menu offers
        # renaming exactly what is in front of the user. A folder named
        # sub-01 offers the subject; a func file offers task, run and echo.
        renames = _renameable_entities(Path(path))
        if renames or _can_restructure(Path(path)):
            menu_section(menu, "Names and structure")
        if renames:
            for entity, value, label in renames:
                action = menu.addAction(f"Rename {label} {entity}-{value}...")
                action.setToolTip(
                    f"Rename {entity}-{value} everywhere in the dataset, "
                    "including the references inside IntendedFor, "
                    "*_scans.tsv and participants.tsv."
                )
                action.triggered.connect(
                    lambda _checked=False, e=entity, v=value:
                        self.rename_requested.emit(e, v, str(path))
                )

        # Restructuring, on whatever the click covers. A right-click inside an
        # existing multi-selection acts on the whole selection, which is what
        # every file manager does and what makes "these four runs" one step;
        # a click outside it acts on the one row, because that is plainly what
        # was meant.
        chosen = self.selected_paths()
        clicked = Path(path)
        scope = chosen if clicked in chosen and len(chosen) > 1 else [clicked]
        if _can_restructure(clicked):
            add_entity = menu.addAction("Add or change an entity...")
            add_entity.setToolTip(
                "Give these files an entity the schema allows them, placed "
                "where the standard puts it. Sidecars and companion files "
                "travel with the recording."
            )
            add_entity.triggered.connect(
                lambda _c=False, s=scope:
                    self.entities_requested.emit(s, "add", False)
            )
            drop_entity = menu.addAction("Remove an entity...")
            drop_entity.setToolTip(
                "Take an optional entity off these files. Required ones are "
                "not offered, because a name without them has no reading."
            )
            drop_entity.triggered.connect(
                lambda _c=False, s=scope:
                    self.entities_requested.emit(s, "remove", False)
            )

            if _in_session(clicked):
                session = menu.addAction("Remove from its session...")
                mode = "remove"
            else:
                session = menu.addAction("Create a session for this...")
                mode = "add"
            session.setToolTip(
                "A session is a folder and an entity at once. The "
                "*_scans.tsv travels to the level BIDS puts it at and every "
                "reference follows."
            )
            session.triggered.connect(
                lambda _c=False, s=scope, m=mode:
                    self.entities_requested.emit(s, m, True)
            )

            menu_section(menu, "Identifiable data")

            deface = menu.addAction("Remove faces...")
            deface.setToolTip(
                "Blank the face on the anatomical images here, leaving the "
                "brain alone. Shows which images, and which cannot be done, "
                "before anything is written. One undoable step."
            )
            deface.triggered.connect(
                lambda _c=False, s=scope: self.deface_requested.emit(s)
            )

            if clicked.is_file() and clicked.name.endswith((".nii", ".nii.gz")):
                compare = menu.addAction("Compare with the original...")
                compare.setToolTip(
                    "Show the image before defacing beside the one in the "
                    "dataset, with one crosshair between them, so you can "
                    "check that the face went and the brain did not."
                )
                compare.triggered.connect(
                    lambda _c=False, p=clicked:
                        self.deface_compare_requested.emit(p)
                )

            strip = menu.addAction("Remove the skull...")
            strip.setToolTip(
                "Keep only the brain. The result is a derivative, so it goes "
                "to derivatives/ and the original scan is left alone."
            )
            strip.triggered.connect(
                lambda _c=False, s=scope: self.strip_requested.emit(s)
            )

            revert = menu.addAction("Put the face back...")
            revert.setToolTip(
                "Restore these images from the undefaced copies kept in "
                "sourcedata/ or in the edit history."
            )
            revert.triggered.connect(
                lambda _c=False, s=scope: self.deface_revert_requested.emit(s)
            )

            menu_section(menu, "Remove")
            what = (
                f"{len(scope)} items" if len(scope) > 1
                else ("this folder and everything in it" if clicked.is_dir()
                      else "this recording")
            )
            delete = menu.addAction(f"Delete {what}...")
            delete.setToolTip(
                "Shows what would go first, including the *_scans.tsv rows, "
                "the IntendedFor entries and the folders this empties. One "
                "step in the Editor's history, so it can be undone."
            )
            delete.triggered.connect(
                lambda _c=False, s=scope: self.delete_requested.emit(s)
            )

        # Last, and grouped, because these three are about the path rather
        # than about the dataset. Every file manager and editor puts them
        # here for the same reason.
        menu_section(menu, "This file")

        copy_path_action = menu.addAction("Copy path")
        copy_path_action.triggered.connect(
            lambda: QApplication.clipboard().setText(str(path))
        )

        copy_rel_path_action = menu.addAction("Copy relative path")
        copy_rel_path_action.triggered.connect(
            lambda: self._copy_relative_path(path)
        )

        open_in_folder_action = menu.addAction("Open in folder")
        open_in_folder_action.triggered.connect(
            lambda: self._open_in_folder(path)
        )

        menu.exec(self._tree.viewport().mapToGlobal(position))

    def _copy_relative_path(self, path: str) -> None:
        if self._root is None:
            return
        try:
            rel_path = Path(path).relative_to(self._root)
        except ValueError:
            # Item lives outside the current dataset root; nothing sensible
            # to copy as a relative path.
            return
        QApplication.clipboard().setText(str(rel_path))

    def _open_in_folder(self, path_str: str) -> None:
        """Show *path_str* in the OS file manager.

        A directory is opened directly; a file is revealed with the file
        selected inside its parent folder. Works on macOS (Finder), Windows
        (Explorer) and Linux (the FileManager1 D-Bus interface, e.g. Nautilus
        or Dolphin). Any failure falls back to just opening the containing
        folder via Qt, so the action never dead-ends.
        """
        target = Path(path_str)
        try:
            system = platform.system()
            if system == "Darwin":
                if target.is_dir():
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["open", "-R", str(target)])
            elif system == "Windows":
                if target.is_dir():
                    os.startfile(str(target))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["explorer", f"/select,{target}"])
            elif target.is_dir():
                subprocess.Popen(["xdg-open", str(target)])
            else:
                self._reveal_linux(target)
            return
        except Exception as exc:  # never break the menu over a cosmetic action
            log.debug("open in folder failed for %s: %s", target, exc)

        # Fallback: open the containing folder with Qt (no selection).
        folder = target if target.is_dir() else target.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def _reveal_linux(target: Path) -> None:
        """Reveal and select *target* via the FileManager1 D-Bus interface.

        Falls back to opening the parent folder with ``xdg-open`` when no
        FileManager1 provider answers (raises so the caller's fallback runs).
        """
        uri = QUrl.fromLocalFile(str(target)).toString()
        try:
            subprocess.run(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:{uri}",
                    "string:",
                ],
                check=True,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            subprocess.Popen(["xdg-open", str(target.parent)])


__all__ = ["BidsTreePane", "PATH_ROLE"]
