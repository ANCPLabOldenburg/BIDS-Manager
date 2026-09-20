"""``.bidsignore`` as a list of patterns with their effect, not as text.

Opening this file in a text editor tells you what it says. What a user needs
to know is what it *does*: which files each pattern is currently hiding from
the validator, and whether a pattern is hiding nothing because it has a typo
in it.

So the left half is the pattern list, each row carrying its live match count,
and the right half is the dataset's files with a search box and a preview of
what the pattern under the cursor would catch. Adding a pattern is done by
picking files rather than by typing a glob and hoping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...editor import bidsignore as bi
from ..theme_manager import CUR
from .primitives import PaneHeader


class BidsIgnorePane(QWidget):
    """Editor centre pane for ``.bidsignore``."""

    saved = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._root: Optional[Path] = None
        self._paths: list[str] = []
        self._report = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PaneHeader("Ignored files"))

        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 6, 14, 6)
        bl.setSpacing(8)
        self._summary = QLabel("")
        self._summary.setObjectName("dlg-hint")
        bl.addWidget(self._summary)
        bl.addStretch(1)
        self._remove_btn = QPushButton("Remove pattern")
        self._remove_btn.setObjectName("tb-btn")
        self._remove_btn.clicked.connect(self._on_remove)
        bl.addWidget(self._remove_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.clicked.connect(self.save)
        bl.addWidget(self._save_btn)
        v.addWidget(bar)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: the patterns.
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 10, 8, 12)
        ll.setSpacing(6)
        ll.addWidget(_hint("Patterns, and what each one currently hides"))
        self._patterns = QListWidget()
        self._patterns.setObjectName("bidsignore-patterns")
        self._patterns.currentRowChanged.connect(self._on_pattern_selected)
        ll.addWidget(self._patterns, 1)
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        self._new_pattern = QLineEdit()
        self._new_pattern.setObjectName("ent-input")
        self._new_pattern.setPlaceholderText("derivatives/  or  *.mat")
        self._new_pattern.textChanged.connect(self._on_preview_typed)
        self._new_pattern.returnPressed.connect(self._on_add_typed)
        add_row.addWidget(self._new_pattern, 1)
        add_btn = QPushButton("Add")
        add_btn.setObjectName("tb-btn")
        add_btn.clicked.connect(self._on_add_typed)
        add_row.addWidget(add_btn)
        ll.addLayout(add_row)
        self._preview = QLabel("")
        self._preview.setObjectName("dlg-hint")
        self._preview.setWordWrap(True)
        ll.addWidget(self._preview)
        split.addWidget(left)

        # Right: the dataset's files.
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 10, 14, 12)
        rl.setSpacing(6)
        rl.addWidget(_hint("Files in this dataset. Ignored ones are dimmed."))
        self._search = QLineEdit()
        self._search.setObjectName("ent-input")
        self._search.setPlaceholderText("Search files")
        self._search.textChanged.connect(self._refresh_files)
        rl.addWidget(self._search)
        self._files = QListWidget()
        self._files.setObjectName("bidsignore-files")
        self._files.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        rl.addWidget(self._files, 1)
        add_sel = QPushButton("Ignore selected files")
        add_sel.setObjectName("tb-btn")
        add_sel.setToolTip(
            "Adds a pattern for each selection. A file inside a top-level "
            "folder suggests the folder, which is almost always what was "
            "meant."
        )
        add_sel.clicked.connect(self._on_add_selected)
        rl.addWidget(add_sel)
        split.addWidget(right)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        v.addWidget(split, 1)

    # -- binding -------------------------------------------------------

    def set_root(self, root: Optional[Path]) -> None:
        """Point the pane at a dataset and read its ignore file."""
        self._root = Path(root) if root is not None else None
        self.reload()

    def set_report(self, report) -> None:
        """Give the pane the last validation, so it can say what is silenced.

        The point of a pattern is to stop something being reported, and a
        count of matched files does not say whether any of them had findings.
        """
        self._report = report
        self._refresh_files()

    def reload(self) -> None:
        if self._root is None:
            self._patterns.clear()
            self._files.clear()
            return
        self._paths = bi.dataset_paths(self._root)
        items = bi.read_patterns(self._root, self._paths)
        self._patterns.clear()
        pal = CUR()
        for item in items:
            row = QListWidgetItem(item.raw or "")
            if item.comment:
                row.setForeground(_color(pal["muted"]))
            elif item.is_redundant:
                row.setText(f"{item.raw}      redundant")
                row.setForeground(_color(pal["muted"]))
                row.setToolTip(
                    "Targets a dot-path. Every BIDS tool ignores those "
                    "already, so this line does no harm and no work."
                )
            elif item.is_dead:
                row.setText(f"{item.raw}      matches nothing")
                row.setForeground(_color(pal["warning"]))
                row.setToolTip(
                    "This pattern hides no file in this dataset. That is "
                    "usually a typo."
                )
            else:
                row.setText(f"{item.raw}      {item.count} files")
                row.setToolTip("\n".join(item.matches[:40]))
            row.setData(Qt.ItemDataRole.UserRole, item.raw)
            self._patterns.addItem(row)
        self._refresh_files()

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-render the rows, and make Qt recompute the QSS for the views.

        Two separate problems. A list item's foreground is set on the item, so
        the app-wide stylesheet re-apply does not reach it and the rows have to
        be rebuilt from the new palette. And an item view that has already been
        polished keeps its background brush through a re-apply, which left both
        lists white on a dark theme with unreadable text; the documented
        unpolish/polish dance is what makes the new stylesheet take.
        """
        del pal
        for widget in (self, self._patterns, self._files):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
        if self._root is not None:
            self.reload()
        self.update()

    def _current_lines(self) -> list[str]:
        return [
            self._patterns.item(i).data(Qt.ItemDataRole.UserRole) or ""
            for i in range(self._patterns.count())
        ]

    # -- files ---------------------------------------------------------

    def _refresh_files(self) -> None:
        if self._root is None:
            return
        ignored = bi.ignored_paths(self._root)
        needle = self._search.text().strip().lower()
        pal = CUR()
        self._files.clear()
        shown = 0
        for rel in self._paths:
            if needle and needle not in rel.lower():
                continue
            item = QListWidgetItem(rel)
            if rel in ignored:
                item.setForeground(_color(pal["muted"]))
                item.setToolTip("Currently ignored")
            self._files.addItem(item)
            shown += 1
        text = f"{len(ignored)} of {len(self._paths)} files ignored"
        hidden = bi.findings_hidden_by(self._root, self._report)
        if hidden:
            text += f"  ·  hiding {hidden} finding(s) from validation"
        if needle:
            text += f"  ·  showing {shown}"
        self._summary.setText(text)

    # -- editing -------------------------------------------------------

    def _on_pattern_selected(self, row: int) -> None:
        if row < 0 or self._root is None:
            return
        raw = self._patterns.item(row).data(Qt.ItemDataRole.UserRole) or ""
        hits = bi.matches_for(raw, self._paths)
        self._preview.setText(
            f"{raw} matches {len(hits)} file(s)"
            + ("" if not hits else ":  " + ", ".join(hits[:4])
               + (" ..." if len(hits) > 4 else ""))
        )

    def _on_preview_typed(self, text: str) -> None:
        text = text.strip()
        if not text or self._root is None:
            self._preview.setText("")
            return
        hits = bi.matches_for(text, self._paths)
        self._preview.setText(
            f"would match {len(hits)} file(s)"
            + ("" if not hits else ":  " + ", ".join(hits[:4])
               + (" ..." if len(hits) > 4 else ""))
        )

    def _add_line(self, pattern: str) -> None:
        if not pattern or pattern in self._current_lines():
            return
        item = QListWidgetItem(pattern)
        item.setData(Qt.ItemDataRole.UserRole, pattern)
        self._patterns.addItem(item)

    def _on_add_typed(self) -> None:
        self._add_line(self._new_pattern.text().strip())
        self._new_pattern.clear()
        self._preview.setText("")

    def _on_add_selected(self) -> None:
        for item in self._files.selectedItems():
            self._add_line(bi.suggest_pattern(item.text()))

    def _on_remove(self) -> None:
        row = self._patterns.currentRow()
        if row >= 0:
            self._patterns.takeItem(row)

    def save(self) -> None:
        """Write the pattern list, then recount so the effect is visible."""
        if self._root is None:
            return
        path = bi.write_patterns(self._root, self._current_lines())
        self.reload()
        self.saved.emit(path)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("pane-hint")
    label.setWordWrap(True)
    return label


def _color(spec: str):
    from PyQt6.QtGui import QColor

    return QColor(spec)


__all__ = ["BidsIgnorePane"]
