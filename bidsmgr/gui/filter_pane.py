"""Filter / structure tree (column 2 of the Converter view).

Reference: ``inspector_proto/proto.py`` lines 573-613.

Tree of ``dataset → subject → session → datatype`` with tri-state
checkboxes. Toggling a node propagates: checking a parent checks all
children, unchecking unchecks all; a parent shows the partial state
when its children disagree.

The checkboxes drive the inventory model's ``include`` column —
exactly the same flag the table's checkbox column toggles per row.
This pane is the bulk-edit counterpart: toggle "sub-002 / ses-post"
to include / exclude every series under it in one click.

It also follows the table: picking a sequence here selects its row in the
inspection table (and Properties), and a row selected there is shown here.

Three modes, chosen in its title bar (the title follows):

* **Structure**: dataset, subject, session, datatype, recording.
* **Recordings**: each kind of recording ACROSS subjects (every T1w, every
  ``task-rest_bold``, every localizer), one checkbox for all of them, so a
  kind is left out of the whole study in one click (one Undo).
* **BIDS preview**: what the conversion will write, companions and dataset
  files included, and what a file will roughly contain
  (:mod:`bidsmgr.gui.output_preview`).

"Hide rows left out" (a toggle in its header) leaves only what the conversion
will write in the first two.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import icons
from .models import InventoryTableModel
from .output_preview import OutputPreview
from .widgets import PaneHeader
from .widgets.panel_frame import HEADER_EXTRAS, HEADER_LEAD
from .widgets.primitives import Segmented
from .widgets.tree_click import toggle_on_click

log = logging.getLogger(__name__)


# Custom role to remember which row indices a tree leaf represents.
# Stored on the *leaf* nodes (datatype level); parent nodes derive
# their state from their children automatically via Qt's tri-state.
_ROW_IDS_ROLE = Qt.ItemDataRole.UserRole + 10
#: What a parent node is ("dataset", "subject", "session", "datatype") and
#: the label it stands for.
_KIND_ROLE = Qt.ItemDataRole.UserRole + 11
_LABEL_ROLE = Qt.ItemDataRole.UserRole + 12
#: Whether the tree hides the rows left out of the conversion.
_HIDE_KEY = "filter/hide_left_out"
#: The mode last shown.
_MODE_KEY = "filter/mode"
#: (mode, the panel's title in it, icon, what it shows).
MODES = (
    ("structure", "Structure", "mode_structure",
     "Structure: subjects, sessions and datatypes, each recording ticked in or out"),
    ("recordings", "Recordings", "mode_recordings",
     "Recordings: each kind of recording across all subjects, ticked in or out at once"),
    ("preview", "BIDS preview", "mode_preview",
     "BIDS preview: every file the conversion will write, and what it will contain"),
)


class FilterPane(QWidget):
    """Tri-state structural filter over the active inventory model.

    Bind a model with :meth:`bind_model`; the tree rebuilds on bind and
    again whenever the caller invokes :meth:`refresh`. Toggling any node
    writes ``include`` updates back through ``model.setData`` so events
    are recorded and chips refresh.
    """

    #: A sequence (an inventory row) picked in the tree.
    row_chosen = pyqtSignal(int)
    #: A right-click in Structure or Recordings: (the rows under what was
    #: clicked, the global position, and ``(kind, label)`` when it was a
    #: subject or a session, else ``None``). The panel builds the menu, the
    #: same one the inspection table offers for those rows.
    rows_menu_requested = pyqtSignal(list, object, object)
    #: The mode changed: (mode, the panel's title for it).
    mode_changed = pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self.setMinimumWidth(190)

        self._model: Optional[InventoryTableModel] = None
        self._suppress_writeback = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Filter / structure"))

        # Its header button, lifted into the frame's title bar (and back here
        # when the panel floats).
        corner = QWidget(self)
        corner.setObjectName(HEADER_EXTRAS)
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 4, 0)
        cl.setSpacing(2)
        self.hide_left_out_btn = QToolButton(corner)
        self.hide_left_out_btn.setObjectName("dock-corner-btn")
        self.hide_left_out_btn.setCheckable(True)
        self.hide_left_out_btn.setAutoRaise(True)
        self.hide_left_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_left_out_btn.setToolTip(
            "Hide rows left out: show only what the conversion will write. The rows stay "
            "in the table, where they can be ticked again.")
        self.hide_left_out_btn.setIcon(icons.icon("hide_left_out"))
        self.hide_left_out_btn.setChecked(_truthy(QSettings().value(_HIDE_KEY, False)))
        self.hide_left_out_btn.toggled.connect(self._on_hide_toggled)
        cl.addWidget(self.hide_left_out_btn)
        # The mode switch, lifted in after the title (HEADER_LEAD).
        home_row = QWidget(self)
        home = QHBoxLayout(home_row)
        home.setContentsMargins(0, 0, 0, 0)
        lead = QWidget(home_row)
        lead.setObjectName(HEADER_LEAD)
        ll = QHBoxLayout(lead)
        ll.setContentsMargins(0, 0, 0, 0)
        self.mode_switch = Segmented([(key, "", tip, icon) for key, _t, icon, tip in MODES],
                                     parent=lead)
        self.mode_switch.changed.connect(self.set_mode)
        ll.addWidget(self.mode_switch)
        home.addWidget(lead)
        home.addStretch(1)
        home.addWidget(corner)
        v.addWidget(home_row)
        self._dirty: set[str] = set()

        self._tree = QTreeWidget()
        self._tree.setObjectName("filter-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        toggle_on_click(self._tree)
        # A sequence picked here is picked in the inspection table (and so
        # in Properties); one picked there is shown here.
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_menu)
        self._revealing = False

        self._kinds = QTreeWidget()
        self._kinds.setObjectName("filter-tree")
        self._kinds.setHeaderHidden(True)
        self._kinds.setRootIsDecorated(True)
        self._kinds.setIndentation(16)
        self._kinds.setUniformRowHeights(True)
        self._kinds.itemChanged.connect(self._on_item_changed)
        self._kinds.currentItemChanged.connect(self._on_current_changed)
        toggle_on_click(self._kinds)
        self._kinds.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._kinds.customContextMenuRequested.connect(
            lambda pos: self._on_tree_menu(pos, self._kinds))

        self.preview = OutputPreview(self)
        self.preview.row_chosen.connect(self.row_chosen)

        self._pages = QStackedWidget(self)
        for page in (self._tree, self._kinds, self.preview):
            self._pages.addWidget(page)
        v.addWidget(self._pages, 1)

        self._empty = QLabel("(scan first to populate this filter)")
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        v.addWidget(self._empty)
        mode = str(QSettings().value(_MODE_KEY, "structure") or "structure")
        self._mode = mode if mode in {m[0] for m in MODES} else "structure"
        self.mode_switch.set_current(self._mode)
        self._pages.setCurrentIndex(self._page_of(self._mode))
        self.hide_left_out_btn.setVisible(self._mode != "preview")
        self._sync_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # -- modes ----------------------------------------------------------------

    @staticmethod
    def _page_of(mode: str) -> int:
        return [m[0] for m in MODES].index(mode)

    def mode(self) -> str:
        return self._mode

    def mode_title(self, mode: Optional[str] = None) -> str:
        return next(m[1] for m in MODES if m[0] == (mode or self._mode))

    def set_mode(self, mode: str) -> None:
        """Show the structure, the recordings across subjects, or the BIDS
        preview; remembered."""
        if mode not in {m[0] for m in MODES}:
            return
        changed = mode != self._mode
        self._mode = mode
        self.mode_switch.set_current(mode)
        QSettings().setValue(_MODE_KEY, mode)
        self._pages.setCurrentIndex(self._page_of(mode))
        self.hide_left_out_btn.setVisible(mode != "preview")
        if mode in self._dirty:
            self.refresh()
        self._sync_empty()
        if changed:
            self.mode_changed.emit(mode, self.mode_title(mode))

    def _active_tree(self) -> Optional[QTreeWidget]:
        return {"structure": self._tree, "recordings": self._kinds}.get(self._mode)

    def _sync_empty(self) -> None:
        empty = self._model is None or self._model.rowCount() == 0
        trees = self._mode != "preview"
        self._empty.setVisible(empty and trees)
        self._pages.setVisible(not (empty and trees))

    def reveal_row(self, row: Optional[int]) -> None:
        """Make the leaf of inventory ``row`` the current item, scrolled into
        view, without announcing it back (the table already shows it)."""
        if self._mode == "preview":
            self.preview.reveal_row(row)
            return
        tree = self._active_tree()
        leaf = self._leaf_for_row(row, tree) if row is not None else None
        self._revealing = True
        try:
            if leaf is None:
                tree.setCurrentItem(None)
                tree.clearSelection()
            else:
                parent = leaf.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                tree.setCurrentItem(leaf)
                tree.scrollToItem(leaf)
        finally:
            self._revealing = False

    def _leaf_for_row(self, row: int, tree: Optional[QTreeWidget] = None) -> Optional[QTreeWidgetItem]:
        tree = tree or self._tree
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            rows = item.data(0, _ROW_IDS_ROLE)
            if rows and row in rows:
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        return None

    def _on_current_changed(self, current, _previous) -> None:
        if self._revealing or current is None:
            return
        rows = current.data(0, _ROW_IDS_ROLE)
        if rows:
            self.row_chosen.emit(int(rows[0]))

    def bind_model(self, model: Optional[InventoryTableModel]) -> None:
        """Attach / detach the inventory model. Rebuilds the tree."""
        if self._model is model:
            return
        if self._model is not None:
            try:
                self._model.dataChanged.disconnect(self._on_model_data_changed)
            except (TypeError, RuntimeError):
                pass
        self._model = model
        if model is not None:
            model.dataChanged.connect(self._on_model_data_changed)
        self.preview.bind_model(model)
        self.refresh()

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-make the header's icons, drawn in the palette they were made in
        (GUI_LOOK_AND_FEEL 1.4), and the preview's colours."""
        self.hide_left_out_btn.setIcon(icons.icon("hide_left_out"))
        self.mode_switch.refresh_icons()
        self.preview.repaint_for_palette(pal)

    def hides_left_out(self) -> bool:
        return self.hide_left_out_btn.isChecked()

    def _on_hide_toggled(self, on: bool) -> None:
        QSettings().setValue(_HIDE_KEY, bool(on))
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree shown from the current model state; the other is
        rebuilt when it is next shown."""
        self._sync_empty()
        if self._mode == "preview":
            self._dirty |= {"structure", "recordings"}
            return
        self._dirty = {"structure", "recordings"} - {self._mode}
        tree = self._active_tree()
        self._suppress_writeback = True
        try:
            tree.clear()
            if self._model is None or self._model.rowCount() == 0:
                return
            if self._mode == "recordings":
                self._build_kinds()
            else:
                self._build_tree()
        finally:
            self._suppress_writeback = False

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        """Walk the model's DataFrame and build the structural tree.

        Layout: ``dataset → subject → session → datatype → sequence``,
        each individual sequence appearing as its own leaf carrying a
        single row index. Parents (dataset/subject/session/datatype) are
        tri-state via ``ItemIsAutoTristate``; toggling a parent cascades
        through every sequence underneath it. Toggling a single sequence
        leaf writes ``include`` for that one row only.
        """
        assert self._model is not None
        df = self._model.dataframe()

        # (ds, sub, ses, dt) → list of (row_idx, leaf_label) tuples,
        # preserving the order rows appear in the DataFrame.
        groups: dict[tuple[str, str, str, str], list[tuple[int, str]]] = {}
        for i in df.index:
            ds = str(df.at[i, "dataset"]) if "dataset" in df.columns else ""
            sub = str(df.at[i, "participant_id"]) if "participant_id" in df.columns else ""
            ses = str(df.at[i, "session"]) if "session" in df.columns else ""
            dt = str(df.at[i, "datatype"]) if "datatype" in df.columns else ""
            key = (
                ds or "(no dataset)",
                sub or "(no subject)",
                ses,
                dt or "(no datatype)",
            )
            label = self._sequence_label(df, int(i))
            groups.setdefault(key, []).append((int(i), label))

        ds_nodes: dict[str, QTreeWidgetItem] = {}
        sub_nodes: dict[tuple[str, str], QTreeWidgetItem] = {}
        ses_nodes: dict[tuple[str, str, str], QTreeWidgetItem] = {}
        dt_nodes: dict[tuple[str, str, str, str], QTreeWidgetItem] = {}

        # One dataset: its node gets no checkbox. One misplaced click there
        # unticked or ticked the whole table, and a whole table is not a
        # choice anyone makes from the tree.
        lone_dataset = len({k[0] for k in groups}) == 1
        for (ds, sub, ses, dt) in sorted(groups.keys()):
            entries = groups[(ds, sub, ses, dt)]
            # --- dataset / subject / session / datatype parents ---
            if ds not in ds_nodes:
                ds_nodes[ds] = self._make_parent_node(ds, "", checkable=not lone_dataset)
                self._tree.addTopLevelItem(ds_nodes[ds])
            if (ds, sub) not in sub_nodes:
                sub_nodes[(ds, sub)] = self._make_parent_node(sub, "")
                sub_nodes[(ds, sub)].setData(0, _KIND_ROLE, "subject")
                sub_nodes[(ds, sub)].setData(0, _LABEL_ROLE, sub)
                ds_nodes[ds].addChild(sub_nodes[(ds, sub)])
            parent_for_dt = sub_nodes[(ds, sub)]
            if ses:
                if (ds, sub, ses) not in ses_nodes:
                    ses_nodes[(ds, sub, ses)] = self._make_parent_node(ses, "")
                    ses_nodes[(ds, sub, ses)].setData(0, _KIND_ROLE, "session")
                    ses_nodes[(ds, sub, ses)].setData(0, _LABEL_ROLE, ses)
                    sub_nodes[(ds, sub)].addChild(ses_nodes[(ds, sub, ses)])
                parent_for_dt = ses_nodes[(ds, sub, ses)]
            if (ds, sub, ses, dt) not in dt_nodes:
                dt_nodes[(ds, sub, ses, dt)] = self._make_parent_node(
                    dt, f"   {len(entries)}",
                )
                parent_for_dt.addChild(dt_nodes[(ds, sub, ses, dt)])

            # --- one leaf per sequence under the datatype ---
            for row_idx, label in entries:
                leaf = QTreeWidgetItem([label])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setData(0, _ROW_IDS_ROLE, (row_idx,))
                included = self._model._read_include(row_idx)
                leaf.setCheckState(
                    0,
                    Qt.CheckState.Checked if included else Qt.CheckState.Unchecked,
                )
                dt_nodes[(ds, sub, ses, dt)].addChild(leaf)

        if self.hides_left_out():
            self._hide_left_out(self._tree)
        self._tree.expandAll()

    def _build_kinds(self) -> None:
        """Recordings mode: datatype, then each kind of recording across the
        subjects (every one of them, left out or not), then who has it."""
        from ..inventory.overview import recording_groups

        df = self._model.dataframe()
        names = df["bids_name"].tolist() if "bids_name" in df.columns else [""] * len(df)
        subjects = df["participant_id"].tolist() if "participant_id" in df.columns else [""] * len(df)
        sessions = df["session"].tolist() if "session" in df.columns else [""] * len(df)
        parents: dict[str, QTreeWidgetItem] = {}
        for group in recording_groups(df):
            heading = group.datatype or "without a BIDS name"
            parent = parents.get(heading)
            if parent is None:
                parent = self._make_parent_node(heading, "")
                parent.setData(0, _KIND_ROLE, "datatype")
                parents[heading] = parent
                self._kinds.addTopLevelItem(parent)
            n = len(group.rows)
            node = self._make_parent_node(group.title, f"   {n}")
            node.setData(0, _KIND_ROLE, "kind")
            node.setData(0, _LABEL_ROLE, group.title)
            who = "subject" if group.subjects == 1 else "subjects"
            node.setToolTip(0, f"{group.title}\n{n} recording{'s' if n != 1 else ''} in "
                               f"{group.subjects} {who}. Tick or untick to include or leave "
                               "out every one of them.")
            parent.addChild(node)
            for row in group.rows:
                leaf = QTreeWidgetItem([_who(str(subjects[row] or ""), str(sessions[row] or ""),
                                             str(names[row] or ""), group.title)])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setData(0, _ROW_IDS_ROLE, (int(row),))
                leaf.setToolTip(0, str(names[row] or self._sequence_label(df, int(row))))
                leaf.setCheckState(0, Qt.CheckState.Checked if self._model._read_include(int(row))
                                   else Qt.CheckState.Unchecked)
                node.addChild(leaf)
        if self.hides_left_out():
            self._hide_left_out(self._kinds)
        for parent in parents.values():
            parent.setExpanded(True)

    def _hide_left_out(self, tree: QTreeWidget) -> None:
        """Hide every leaf left out, then every node left with nothing shown."""
        def visit(item: QTreeWidgetItem) -> bool:
            rows = item.data(0, _ROW_IDS_ROLE)
            if rows:
                shown = any(self._model._read_include(int(r)) for r in rows)
            else:
                shown = False
                for i in range(item.childCount()):
                    shown = visit(item.child(i)) or shown
            item.setHidden(not shown)
            return shown

        for i in range(tree.topLevelItemCount()):
            visit(tree.topLevelItem(i))

    def _make_parent_node(self, label: str, suffix: str, *,
                          checkable: bool = True) -> QTreeWidgetItem:
        """Build a tri-state parent node (dataset / sub / ses / datatype)."""
        node = QTreeWidgetItem([self._format_label(label, suffix)])
        if checkable:
            node.setFlags(
                node.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
        else:
            # Qt makes every item user-checkable by default.
            node.setFlags(node.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        return node

    @staticmethod
    def _sequence_label(df, row_idx: int) -> str:
        """Pick the most informative label for a sequence leaf.

        Prefers ``bids_name`` (the BIDS-shaped name the user is
        about to commit to). Falls back to the original DICOM
        SeriesDescription (``sequence`` column for MRI), then the
        ``source_file`` stem for EEG/MEG rows without a basename yet.
        """
        for col in ("bids_name", "sequence", "source_file"):
            if col not in df.columns:
                continue
            raw = df.at[row_idx, col]
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            if col == "source_file":
                return text.split("/")[-1]
            return text
        return f"row {row_idx + 1}"

    @staticmethod
    def _format_label(label: str, suffix: str) -> str:
        return f"{label}{suffix}"

    # ------------------------------------------------------------------
    # Toggling
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Propagate a leaf check change to the model's include flags.

        Parent nodes propagate via Qt's auto-tristate; the leaf nodes
        are the ones that hold row indices. When a parent is toggled,
        Qt cascades the change down to its children, each of which
        fires its own ``itemChanged`` — so we only need to write
        through for leaves.
        """
        if self._suppress_writeback or self._model is None:
            return
        if column != 0:
            return
        row_ids = item.data(0, _ROW_IDS_ROLE)
        if not row_ids:
            return  # parent node; cascading children will write through
        state = item.checkState(0)
        if state == Qt.CheckState.PartiallyChecked:
            return  # partial states only happen on parents
        included = state == Qt.CheckState.Checked
        # Walk through the model so events / chips / preview update.
        include_col = next(
            (i for i, c in enumerate(self._model.COLUMNS) if c.key == "include"),
            None,
        )
        if include_col is None:
            return
        # A parent checkbox ticks every leaf under it, one signal each, in
        # this same turn: one user action, one Undo.
        self._model.edit_group_this_turn()
        self._suppress_writeback = True
        try:
            for r in row_ids:
                self._model.setData(self._model.index(r, include_col), included)
        finally:
            self._suppress_writeback = False

    def _on_tree_menu(self, pos, tree: Optional[QTreeWidget] = None) -> None:
        """Right-click on any row of a view: announce the rows under it.

        A recording is its row; a subject, session, datatype or kind of
        recording is every row under it. A subject or session can also be
        renamed, so that is said too.
        """
        tree = tree or self._tree
        item = tree.itemAt(pos)
        if item is None:
            return
        rows = self._rows_under(item)
        if not rows:
            return
        kind = item.data(0, _KIND_ROLE)
        rename = ((kind, str(item.data(0, _LABEL_ROLE) or ""))
                  if tree is self._tree and kind in ("subject", "session") else None)
        self.rows_menu_requested.emit(rows, tree.viewport().mapToGlobal(pos), rename)

    def _rows_under(self, item: QTreeWidgetItem) -> list[int]:
        rows: list[int] = []
        stack = [item]
        while stack:
            node = stack.pop()
            rows.extend(node.data(0, _ROW_IDS_ROLE) or ())
            stack.extend(node.child(i) for i in range(node.childCount()))
        return sorted(set(int(r) for r in rows))

    def _on_model_data_changed(self, top_left, bottom_right, _roles=()) -> None:
        """Re-sync the tree if the model changed underneath us.

        Pragmatic implementation: rebuild the whole tree. The trees in
        scope (a few dozen leaves at most) rebuild in well under 10ms.
        """
        if self._suppress_writeback:
            return
        # Only rebuild if rows / structure changed; a single mirror-cell
        # edit on task/run wouldn't change the tree shape, but the
        # include column might have flipped — easiest to rebuild
        # unconditionally.
        self.refresh()


def _who(subject: str, session: str, name: str, kind: str) -> str:
    """A leaf of a kind: whose it is (``sub-01 ses-pre run-2``), the parts of
    the name the kind does not already say."""
    if not name:
        return " ".join(x for x in (subject, session) if x) or "(no subject)"
    kind_tokens = set(kind.split("_"))
    tokens = [t for t in name.split("_")[:-1] if t not in kind_tokens]
    return " ".join(tokens) or name


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


__all__ = ["FilterPane"]
