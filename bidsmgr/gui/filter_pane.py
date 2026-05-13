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
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import InventoryTableModel
from .theme_manager import CUR
from .widgets import PaneHeader

log = logging.getLogger(__name__)


# Custom role to remember which row indices a tree leaf represents.
# Stored on the *leaf* nodes (datatype level); parent nodes derive
# their state from their children automatically via Qt's tri-state.
_ROW_IDS_ROLE = Qt.ItemDataRole.UserRole + 10


class FilterPane(QWidget):
    """Tri-state structural filter over the active inventory model.

    Bind a model with :meth:`bind_model`; the tree rebuilds on bind and
    again whenever the caller invokes :meth:`refresh`. Toggling any node
    writes ``include`` updates back through ``model.setData`` so events
    are recorded and chips refresh.
    """

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

        self._tree = QTreeWidget()
        self._tree.setObjectName("filter-tree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        v.addWidget(self._tree, 1)

        self._empty = QLabel("(scan first to populate this filter)")
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        v.addWidget(self._empty)
        self._empty.setVisible(True)
        self._tree.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        self.refresh()

    def repaint_for_palette(self, _pal: dict) -> None:
        """No-op for this pane — it has no palette-baked styling.

        Kept for API parity with the other panes (the
        ``ConverterPanel.repaint_for_palette`` cascades unconditionally).
        """

    def refresh(self) -> None:
        """Rebuild the tree from the current model state."""
        self._suppress_writeback = True
        try:
            self._tree.clear()
            if self._model is None or self._model.rowCount() == 0:
                self._empty.setVisible(True)
                self._tree.setVisible(False)
                return
            self._empty.setVisible(False)
            self._tree.setVisible(True)
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
            sub = str(df.at[i, "BIDS_name"]) if "BIDS_name" in df.columns else ""
            ses = str(df.at[i, "session"]) if "session" in df.columns else ""
            dt = str(df.at[i, "proposed_datatype"]) if "proposed_datatype" in df.columns else ""
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

        for (ds, sub, ses, dt) in sorted(groups.keys()):
            entries = groups[(ds, sub, ses, dt)]
            # --- dataset / subject / session / datatype parents ---
            if ds not in ds_nodes:
                ds_nodes[ds] = self._make_parent_node(ds, "")
                self._tree.addTopLevelItem(ds_nodes[ds])
            if (ds, sub) not in sub_nodes:
                sub_nodes[(ds, sub)] = self._make_parent_node(sub, "")
                ds_nodes[ds].addChild(sub_nodes[(ds, sub)])
            parent_for_dt = sub_nodes[(ds, sub)]
            if ses:
                if (ds, sub, ses) not in ses_nodes:
                    ses_nodes[(ds, sub, ses)] = self._make_parent_node(ses, "")
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

        self._tree.expandAll()

    def _make_parent_node(self, label: str, suffix: str) -> QTreeWidgetItem:
        """Build a tri-state parent node (dataset / sub / ses / datatype)."""
        node = QTreeWidgetItem([self._format_label(label, suffix)])
        node.setFlags(
            node.flags()
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsAutoTristate
        )
        return node

    @staticmethod
    def _sequence_label(df, row_idx: int) -> str:
        """Pick the most informative label for a sequence leaf.

        Prefers ``proposed_basename`` (the BIDS-shaped name the user is
        about to commit to). Falls back to the original DICOM
        SeriesDescription (``sequence`` column for MRI), then the
        ``source_file`` stem for EEG/MEG rows without a basename yet.
        """
        for col in ("proposed_basename", "sequence", "source_file"):
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
        self._suppress_writeback = True
        try:
            for r in row_ids:
                self._model.setData(self._model.index(r, include_col), included)
        finally:
            self._suppress_writeback = False

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


__all__ = ["FilterPane"]
