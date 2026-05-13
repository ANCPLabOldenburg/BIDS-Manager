"""TSV / TSV.GZ viewer + editor (Editor center pane, table kind).

Sister widget to :class:`SidecarFormPane`. When the user clicks a
``.tsv`` (or ``.tsv.gz``) file in the BIDS tree, :class:`EditorPanel`
swaps its center pane to this viewer.

* Loads via the standard library (``csv`` + ``gzip``).
* Renders headers + rows in a ``QTableView`` over a
  ``QStandardItemModel``.
* Cells are inline-editable (double-click / F2 / select-then-click).
* Toolbar provides ``+ Add row`` / ``+ Add column`` /
  ``− Delete row`` / ``− Delete column`` plus ``Revert`` / ``Save``
  (same manual-save model as the JSON sidecar pane).
* Edits update the model only; ``Save`` flushes the table to disk.
* ``Revert`` reloads from disk.
* Switching files silently discards unsaved edits (the dirty chip
  warned them while the file was bound — same UX as the sidecar pane).

Theme handling: every palette colour comes from the global QSS.
:meth:`repaint_for_palette` runs the same unpolish/polish dance the
sidecar pane uses so cached per-widget styles invalidate on theme
swap.
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QStackedLayout,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .primitives import PaneHeader

log = logging.getLogger(__name__)


# Don't slurp arbitrarily-large TSVs into memory. BIDS TSVs (events /
# channels / participants / scans) are small in practice; if anyone
# passes a multi-million-row table we cap the preview here and the
# footer warns about truncation. Saving a truncated TSV would be
# destructive — when the file was truncated on load, the Save button
# stays disabled.
_MAX_PREVIEW_ROWS = 5000


def _open_tsv(path: Path):
    """Open a ``.tsv`` or ``.tsv.gz`` for text reading (UTF-8)."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _open_tsv_write(path: Path):
    """Open a ``.tsv`` or ``.tsv.gz`` for text writing (UTF-8)."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _read_tsv(
    path: Path,
    *,
    max_rows: int = _MAX_PREVIEW_ROWS,
) -> tuple[list[str], list[list[str]], int]:
    """Read a TSV file. Returns ``(header, rows, total_rows)``.

    ``total_rows`` is the count of data rows actually present on disk
    (so the footer can report truncation). When the file is malformed
    or empty, returns ``([], [], 0)``.
    """
    try:
        with _open_tsv(path) as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                return [], [], 0
            rows: list[list[str]] = []
            total = 0
            for row in reader:
                total += 1
                if len(rows) < max_rows:
                    rows.append(row)
            return header, rows, total
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        log.debug("could not read TSV %s: %s", path, exc)
        return [], [], 0


class TsvViewerPane(QWidget):
    """Editable table view for BIDS ``.tsv`` files."""

    # Emitted after a successful save. Per-file revalidation (Step 6b)
    # hooks here, same as :class:`SidecarFormPane.file_saved`.
    file_saved = pyqtSignal(Path)
    # Emitted when the disk write fails. Args: (file_path, error_msg).
    save_failed = pyqtSignal(Path, str)
    # Emitted whenever the dirty state flips (after a commit, save,
    # revert, or set_file). ``True`` means there are unsaved edits.
    dirty_changed = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        # Disk snapshot — used to compute dirty + restore on revert.
        self._original_header: list[str] = []
        self._original_rows: list[list[str]] = []
        # When True the on-disk file is larger than the preview cap;
        # saving would drop the tail rows, so Save stays disabled.
        self._truncated_on_load: bool = False
        # Suppress dirty bookkeeping while we're populating the model.
        self._suppress_dirty = False
        self._dirty = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Table"))

        # --- Edit toolbar ----------------------------------------------
        # Mirrors :class:`SidecarFormPane`'s toolbar. Hidden when no
        # TSV is bound.
        self._edit_toolbar = QFrame()
        self._edit_toolbar.setObjectName("sidecar-toolbar")
        et = QHBoxLayout(self._edit_toolbar)
        et.setContentsMargins(14, 6, 14, 6)
        et.setSpacing(8)

        self._add_row_btn = QPushButton("+ Add row")
        self._add_row_btn.setObjectName("tb-btn")
        self._add_row_btn.clicked.connect(self._on_add_row)
        et.addWidget(self._add_row_btn)

        self._del_row_btn = QPushButton("− Delete row")
        self._del_row_btn.setObjectName("tb-btn")
        self._del_row_btn.setEnabled(False)
        self._del_row_btn.clicked.connect(self._on_delete_row)
        et.addWidget(self._del_row_btn)

        self._add_col_btn = QPushButton("+ Add column")
        self._add_col_btn.setObjectName("tb-btn")
        self._add_col_btn.clicked.connect(self._on_add_column)
        et.addWidget(self._add_col_btn)

        self._del_col_btn = QPushButton("− Delete column")
        self._del_col_btn.setObjectName("tb-btn")
        self._del_col_btn.setEnabled(False)
        self._del_col_btn.clicked.connect(self._on_delete_column)
        et.addWidget(self._del_col_btn)

        self._dirty_chip = QLabel("")
        self._dirty_chip.setObjectName("sidecar-dirty-chip")
        self._dirty_chip.setVisible(False)
        et.addWidget(self._dirty_chip)
        et.addStretch(1)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert)
        et.addWidget(self._revert_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save)
        et.addWidget(self._save_btn)

        self._edit_toolbar.setVisible(False)
        v.addWidget(self._edit_toolbar)

        # --- Stacked content: hint on top of the table -----------------
        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._stack, 1)

        self._table = QTableView()
        self._table.setObjectName("tsv-view")
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.verticalHeader().setVisible(True)
        self._model = QStandardItemModel(self)
        self._model.itemChanged.connect(self._on_item_changed)
        self._table.setModel(self._model)
        # Track current selection so the Delete row/col buttons can
        # know which row/col to operate on (and we can grey them out
        # when nothing useful is selected).
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.currentChanged.connect(self._sync_delete_button_state)

        self._empty_hint = QLabel(
            "Select a TSV file in the BIDS tree to view it."
        )
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)

        # Index 0 = empty hint, 1 = table.
        self._stack.addWidget(self._empty_hint)
        self._stack.addWidget(self._table)
        self._stack.setCurrentIndex(0)

        # Footer (path + summary), QSS-driven so theme follows.
        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        self._footer_path = QLabel("")
        self._footer_path.setObjectName("sidecar-footer-path")
        self._footer_summary = QLabel("")
        self._footer_summary.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._footer_summary)
        v.addWidget(self._footer)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def is_dirty(self) -> bool:
        return self._dirty

    def set_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
    ) -> None:
        """Bind the pane to a TSV (or ``None`` to clear)."""
        self._current_file = path
        self._current_root = root
        if path is None:
            self._reset_model()
            self._original_header = []
            self._original_rows = []
            self._truncated_on_load = False
            self._dirty = False
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "Select a TSV file in the BIDS tree to view it."
            )
            self._footer_path.setText("")
            self._footer_summary.setText("")
            self._edit_toolbar.setVisible(False)
            self._refresh_dirty_ui()
            return

        header, rows, total = _read_tsv(path)
        self._original_header = list(header)
        self._original_rows = [list(r) for r in rows]
        self._truncated_on_load = total > len(rows)
        self._dirty = False

        if not header and not rows:
            self._reset_model()
            self._stack.setCurrentIndex(0)
            self._empty_hint.setText(
                "This TSV is empty or could not be parsed."
            )
        else:
            self._populate_model(header, rows)
            self._stack.setCurrentIndex(1)
        self._update_footer(path, root, len(rows), len(header), total)
        # Toolbar appears once a file is bound; Save / Revert wake up
        # when there's something to save.
        self._edit_toolbar.setVisible(True)
        self._refresh_dirty_ui()

    def save(self) -> bool:
        """Flush the model to disk. Returns ``True`` on success or
        no-op, ``False`` on I/O error."""
        if self._current_file is None:
            return True
        if not self._dirty:
            return True
        if self._truncated_on_load:
            # We loaded only the first N rows; writing back would
            # destroy the tail of the file. Surface as a save failure
            # so the user knows nothing happened.
            msg = (
                f"File was truncated on load ({_MAX_PREVIEW_ROWS} rows "
                "shown); refusing to overwrite — saving would discard "
                "the tail rows."
            )
            log.warning("save refused for %s: %s", self._current_file, msg)
            self.save_failed.emit(self._current_file, msg)
            return False
        try:
            self._write_model_to_disk(self._current_file)
        except OSError as exc:
            log.warning("save failed for %s: %s", self._current_file, exc)
            self.save_failed.emit(self._current_file, str(exc))
            return False
        # Snapshot the new on-disk state.
        self._snapshot_current_model()
        self._dirty = False
        self._refresh_dirty_ui()
        self.file_saved.emit(self._current_file)
        return True

    def revert(self) -> None:
        """Reload the bound file from disk, dropping unsaved edits."""
        if self._current_file is None:
            return
        # set_file re-reads from disk and clears dirty.
        path, root = self._current_file, self._current_root
        self.set_file(path, root)

    def repaint_for_palette(self, pal: dict) -> None:
        """Same QSS-only refresh pattern as :class:`SidecarFormPane`."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()

    # ----------------------------------------------------------------------
    # Toolbar handlers
    # ----------------------------------------------------------------------

    def _on_add_row(self) -> None:
        """Append a fresh row at the bottom."""
        if self._current_file is None:
            return
        cols = max(self._model.columnCount(), 1)
        new_items = [QStandardItem("") for _ in range(cols)]
        self._model.appendRow(new_items)
        # If the table was empty (no header) before this, the new row
        # still has 0 columns — give us at least one cell to type in.
        if self._model.columnCount() == 0:
            self._model.setColumnCount(1)
            self._model.setHorizontalHeaderItem(0, QStandardItem("col1"))
        self._mark_dirty()
        # Make the new row visible.
        self._stack.setCurrentIndex(1)

    def _on_delete_row(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.removeRow(idx.row())
        self._mark_dirty()

    def _on_add_column(self) -> None:
        """Append a fresh column on the right.

        Pops a small QInputDialog for the column name (defaults to
        ``newCol``). Cancel skips the operation.
        """
        if self._current_file is None:
            return
        name, ok = QInputDialog.getText(
            self, "Add column", "Column name:", text="newCol",
        )
        if not ok or not name.strip():
            return
        col = self._model.columnCount()
        self._model.setColumnCount(col + 1)
        self._model.setHorizontalHeaderItem(col, QStandardItem(name.strip()))
        # Fill empty cells in the new column for every existing row.
        self._suppress_dirty = True
        try:
            for row in range(self._model.rowCount()):
                self._model.setItem(row, col, QStandardItem(""))
        finally:
            self._suppress_dirty = False
        self._mark_dirty()
        self._stack.setCurrentIndex(1)

    def _on_delete_column(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.removeColumn(idx.column())
        self._mark_dirty()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _reset_model(self) -> None:
        self._suppress_dirty = True
        try:
            self._model.clear()
        finally:
            self._suppress_dirty = False

    def _populate_model(
        self,
        header: list[str],
        rows: list[list[str]],
    ) -> None:
        self._suppress_dirty = True
        try:
            self._model.clear()
            self._model.setHorizontalHeaderLabels(header)
            for row in rows:
                # Pad / truncate row to match the header width — guards
                # against ragged TSVs.
                cells = list(row[: len(header)])
                cells += [""] * (len(header) - len(cells))
                self._model.appendRow(
                    [QStandardItem(c) for c in cells]
                )
        finally:
            self._suppress_dirty = False
        self._table.resizeColumnsToContents()

    def _snapshot_current_model(self) -> None:
        """Capture the current model as the new disk snapshot."""
        self._original_header = self._current_header()
        self._original_rows = self._current_rows()

    def _current_header(self) -> list[str]:
        out = []
        for i in range(self._model.columnCount()):
            it = self._model.horizontalHeaderItem(i)
            out.append(it.text() if it is not None else "")
        return out

    def _current_rows(self) -> list[list[str]]:
        rows = []
        for r in range(self._model.rowCount()):
            row = []
            for c in range(self._model.columnCount()):
                it = self._model.item(r, c)
                row.append(it.text() if it is not None else "")
            rows.append(row)
        return rows

    def _write_model_to_disk(self, path: Path) -> None:
        header = self._current_header()
        rows = self._current_rows()
        with _open_tsv_write(path) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

    def _on_item_changed(self, item: QStandardItem) -> None:
        del item
        if self._suppress_dirty:
            return
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
        self._refresh_dirty_ui()

    def _refresh_dirty_ui(self) -> None:
        has_file = self._current_file is not None
        editable = has_file and not self._truncated_on_load
        if self._dirty:
            self._dirty_chip.setText("unsaved changes")
            self._dirty_chip.setVisible(True)
        else:
            self._dirty_chip.setVisible(False)
        self._save_btn.setEnabled(editable and self._dirty)
        self._revert_btn.setEnabled(has_file and self._dirty)
        self._add_row_btn.setEnabled(editable)
        self._add_col_btn.setEnabled(editable)
        self._sync_delete_button_state()
        self.dirty_changed.emit(self._dirty)

    def _sync_delete_button_state(self, *args) -> None:
        del args
        editable = self._current_file is not None and not self._truncated_on_load
        idx = self._table.currentIndex()
        has_sel = idx.isValid()
        self._del_row_btn.setEnabled(
            editable and has_sel and self._model.rowCount() > 0
        )
        self._del_col_btn.setEnabled(
            editable and has_sel and self._model.columnCount() > 0
        )

    def _update_footer(
        self,
        path: Path,
        root: Optional[Path],
        shown_rows: int,
        cols: int,
        total_rows: int,
    ) -> None:
        if root is not None:
            try:
                rel = path.resolve().relative_to(root.resolve())
                self._footer_path.setText(str(rel))
            except ValueError:
                self._footer_path.setText(str(path))
        else:
            self._footer_path.setText(str(path))
        if shown_rows < total_rows:
            self._footer_summary.setText(
                f"{shown_rows} of {total_rows} rows shown · {cols} columns "
                f"· read-only (truncated)"
            )
        else:
            self._footer_summary.setText(
                f"{total_rows} rows · {cols} columns"
            )


__all__ = ["TsvViewerPane"]
