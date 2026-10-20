"""What validation leaves out (``.bidsignore``): the patterns with their
effect, and the dataset's files to pick from.

Opening this file in a text editor tells you what it says. What a user needs
to know is what it *does*: which files each line hides from the validator,
and whether a line hides nothing because it has a typo in it.

Left, the lines, each with what it hides now; clicking one shows those files
on the right. Under them a field for a pattern typed by hand, with a live
count of what it would hide, and what every validator ignores anyway.

Right, every file of the dataset in ONE model/view (GUI_LOOK_AND_FEEL 4.33:
twenty thousand files are one painted list, not twenty thousand items),
searchable, filtered to all / ignored / validated, each ignored file with the
line that hides it. Pick files and **Ignore** offers the same choices as the
tree's right-click: these files, with their sidecars, their folder, the same
name in every subject, every file of an extension the standard does not use,
each with how many files it hides. **Stop ignoring** removes the line.

Every change is written at once, as one step of the Editor's history: Undo
takes it back, and nothing waits for a Save. (Picking a file used to add its
TOP folder: one stray file in ``sub-01/anat/`` ignored the whole subject.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...editor import bidsignore as bi
from ..theme_manager import CUR
from .primitives import PaneHeader, Segmented
from .search_field import SearchField, find_shortcut
from .splitter import Splitter

#: Files shown of a pattern in the preview line under the field.
_PREVIEW = 3


def _files(n: int) -> str:
    return f"{n:,} file" + ("" if n == 1 else "s")


def _mono(base: QFont) -> QFont:
    from ..typefaces import mono_families

    font = QFont(base)
    font.setFamilies(mono_families())
    return font


class _FilesModel(QAbstractTableModel):
    """Every file of the dataset, filtered; ignored ones with their line."""

    HEADERS = ("File", "Ignored by")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._paths: list[str] = []
        self._why: dict[str, str] = {}
        self._shown: list[str] = []
        self._needle = ""
        self._mode = "all"
        self._only: Optional[set[str]] = None
        self._mono: Optional[QFont] = None

    def set_files(self, paths: list[str], why: dict[str, str]) -> None:
        self.beginResetModel()
        self._paths = paths
        self._why = why
        self._shown = self._filter()
        self.endResetModel()

    def set_filter(self, *, needle: Optional[str] = None, mode: Optional[str] = None,
                   only: Optional[set[str]] = ...) -> None:  # type: ignore[assignment]
        if needle is not None:
            self._needle = needle.strip().lower()
        if mode is not None:
            self._mode = mode
        if only is not ...:
            self._only = only
        self.beginResetModel()
        self._shown = self._filter()
        self.endResetModel()

    def _filter(self) -> list[str]:
        needle, mode, only, why = self._needle, self._mode, self._only, self._why
        out = []
        for path in self._paths if only is None else [p for p in self._paths if p in only]:
            if needle and needle not in path.lower():
                continue
            if mode == "ignored" and path not in why:
                continue
            if mode == "validated" and path in why:
                continue
            out.append(path)
        return out

    def shown(self) -> int:
        return len(self._shown)

    def total(self) -> int:
        return len(self._paths)

    def path_at(self, row: int) -> Optional[str]:
        return self._shown[row] if 0 <= row < len(self._shown) else None

    def why(self, path: str) -> Optional[str]:
        return self._why.get(path)

    # -- Qt ---------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt naming
        return 0 if parent.isValid() else len(self._shown)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt naming
        return 0 if parent.isValid() else 2

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        path = self._shown[index.row()]
        line = self._why.get(path)
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return path if column == 0 else (line.strip() if line else "")
        if role == Qt.ItemDataRole.ForegroundRole and line is not None:
            return QColor(CUR()["muted"])
        if role == Qt.ItemDataRole.FontRole:
            if column == 1:
                if self._mono is None:
                    from PyQt6.QtWidgets import QApplication

                    self._mono = _mono(QApplication.font())
                return self._mono
            if line is not None:
                font = QFont()
                font.setItalic(True)
                return font
        if role == Qt.ItemDataRole.ToolTipRole:
            return (f"{path}\nNot validated: .bidsignore says {line.strip()}" if line
                    else f"{path}\nValidated")
        return None


class BidsIgnorePane(QWidget):
    """Editor centre pane for ``.bidsignore``."""

    #: A change was written; the text says what, for the status bar.
    changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._root: Optional[Path] = None
        self._listing: Optional[bi.Listing] = None
        self._patterns_now: list[bi.IgnorePattern] = []
        self._why: dict[str, str] = {}
        #: The lines whose files the right side shows, or None for all.
        self._focused: Optional[tuple[str, ...]] = None
        self._report = None
        self._read_only = False

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
        bl.addWidget(self._summary, 1)
        v.addWidget(bar)

        split = Splitter(Qt.Orientation.Horizontal)

        # -- left: the lines of the file ------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 10, 8, 12)
        ll.setSpacing(6)
        ll.addWidget(_hint("The lines of .bidsignore, and what each hides now. Click one "
                           "to see its files."))
        self._patterns = QTreeWidget()
        self._patterns.setObjectName("bidsignore-patterns")
        self._patterns.setColumnCount(2)
        self._patterns.setHeaderLabels(["Pattern", "Hides"])
        self._patterns.setRootIsDecorated(False)
        self._patterns.setUniformRowHeights(True)
        # Several lines at once: shown together, removed in one step.
        self._patterns.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # A long path keeps both ends: where it starts and what it names.
        self._patterns.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._patterns.header().setStretchLastSection(False)
        self._patterns.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._patterns.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        # A SELECTION, not the current item: focus alone makes the first
        # row current, which filtered the files the moment the pane opened.
        self._patterns.itemSelectionChanged.connect(self._on_pattern_selection)
        self._patterns.installEventFilter(self)
        self._patterns.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._patterns.customContextMenuRequested.connect(self._pattern_menu)
        ll.addWidget(self._patterns, 1)

        line_row = QHBoxLayout()
        line_row.setSpacing(6)
        self._remove_btn = QPushButton("Remove line")
        self._remove_btn.setObjectName("tb-btn")
        self._remove_btn.setToolTip("Take the selected lines out of .bidsignore: what "
                                    "they hid is validated again. One step of the history, "
                                    "however many lines.")
        self._remove_btn.clicked.connect(self._on_remove)
        line_row.addWidget(self._remove_btn)
        line_row.addStretch(1)
        ll.addLayout(line_row)

        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        self._new_pattern = QLineEdit()
        self._new_pattern.setObjectName("ent-input")
        self._new_pattern.setFont(_mono(self._new_pattern.font()))
        self._new_pattern.setPlaceholderText("Add a pattern, e.g. *.mat")
        self._new_pattern.setToolTip(
            "A line of .bidsignore, as git reads it: *.mat (every .mat file), "
            "/extra/ (a folder at the top), /sub-*/anat/notes.txt (in every "
            "subject). A slash in the middle anchors it at the top.")
        self._new_pattern.textChanged.connect(self._on_preview_typed)
        self._new_pattern.returnPressed.connect(self._on_add_typed)
        add_row.addWidget(self._new_pattern, 1)
        self._add_btn = QPushButton("Add")
        self._add_btn.setObjectName("tb-btn")
        self._add_btn.clicked.connect(self._on_add_typed)
        add_row.addWidget(self._add_btn)
        ll.addLayout(add_row)
        self._preview = QLabel("")
        self._preview.setObjectName("dlg-hint")
        self._preview.setWordWrap(True)
        ll.addWidget(self._preview)
        self._standard = _hint("")
        ll.addWidget(self._standard)
        split.addWidget(left)

        # -- right: the dataset's files --------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 10, 14, 12)
        rl.setSpacing(6)
        rl.addWidget(_hint("The dataset's files. Pick some and Ignore offers the ways to "
                           "leave them out, from exactly these to a pattern for every "
                           "subject."))
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        self._search = SearchField("Find a file")
        filter_row.addWidget(self._search, 1)
        self._mode = Segmented([
            ("all", "All", "Every file of the dataset"),
            ("ignored", "Ignored", "The files validation leaves out"),
            ("validated", "Validated", "The files validation reads"),
        ], current="all")
        self._mode.changed.connect(self._on_mode)
        filter_row.addWidget(self._mode)
        rl.addLayout(filter_row)
        find_shortcut(self, self._search)

        self._focus_row = QHBoxLayout()
        self._focus_row.setSpacing(6)
        self._focus_label = QLabel("")
        self._focus_label.setObjectName("dlg-hint")
        self._focus_row.addWidget(self._focus_label, 1)
        self._focus_clear = QPushButton("Show every file")
        self._focus_clear.setObjectName("tb-btn")
        self._focus_clear.clicked.connect(lambda: self._set_focus(None))
        self._focus_row.addWidget(self._focus_clear)
        self._focus_box = QWidget()
        self._focus_box.setLayout(self._focus_row)
        self._focus_row.setContentsMargins(0, 0, 0, 0)
        self._focus_box.setVisible(False)
        rl.addWidget(self._focus_box)

        self._files = QTreeView()
        self._files.setObjectName("bidsignore-files")
        self._model = _FilesModel(self._files)        # the view does not own it
        self._files.setModel(self._model)
        self._files.setRootIsDecorated(False)
        self._files.setUniformRowHeights(True)
        # Narrow, a path keeps where it is and what it is called.
        self._files.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._files.setAllColumnsShowFocus(True)
        self._files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._files.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # The name takes the room; the line a set width, so nothing measures
        # twenty thousand rows to size a column.
        from ..viz import fonts

        header = self._files.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(1, fonts.px(150))
        self._files.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._files.customContextMenuRequested.connect(self._files_menu)
        self._files.selectionModel().selectionChanged.connect(
            lambda *_a: self._sync_buttons())
        rl.addWidget(self._files, 1)

        act_row = QHBoxLayout()
        act_row.setSpacing(6)
        self._count = QLabel("")
        self._count.setObjectName("dlg-hint")
        act_row.addWidget(self._count, 1)
        self._stop_btn = QPushButton("Stop ignoring")
        self._stop_btn.setObjectName("tb-btn")
        self._stop_btn.setToolTip("Validate the selected files again: in one step, the "
                                  "lines that hide only them go, and a wider line gets an "
                                  "exception for them.")
        self._stop_btn.clicked.connect(lambda: self._popup(self._stop_btn, stop=True))
        act_row.addWidget(self._stop_btn)
        self._ignore_btn = QPushButton("Ignore")
        self._ignore_btn.setObjectName("tb-btn-primary")
        self._ignore_btn.setToolTip("Ways to leave the selected files out of validation, "
                                    "each with how many files it hides.")
        self._ignore_btn.clicked.connect(lambda: self._popup(self._ignore_btn, stop=False))
        act_row.addWidget(self._ignore_btn)
        rl.addLayout(act_row)
        split.addWidget(right)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([420, 580])
        v.addWidget(split, 1)

        # A search over twenty thousand names is a few milliseconds; waiting
        # for the typing to pause keeps it from running per letter.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(
            lambda: self._apply_filter(needle=self._search.text()))
        self._search.textChanged.connect(lambda _t: self._debounce.start())
        self._sync_buttons()

    # -- binding -------------------------------------------------------------

    def set_root(self, root: Optional[Path]) -> None:
        """Point the pane at a dataset and read its ignore file."""
        self._root = Path(root) if root is not None else None
        self._listing = None
        self.reload()

    def set_report(self, report) -> None:
        """The last validation, to say how many of its findings are on files
        now ignored (validating again drops them)."""
        self._report = report
        self._paint_summary()

    def reload(self) -> None:
        """Read the file (and, the first time, the dataset) and redraw."""
        if self._root is None:
            self._patterns.clear()
            self._model.set_files([], {})
            self._summary.setText("")
            return
        if self._listing is None:
            self._listing = bi.list_dataset(self._root)
        paths = self._listing.paths
        self._patterns_now = bi.read_patterns(self._root, paths)
        self._why = bi.ignored_by(self._root, paths)
        self._paint_patterns()
        self._model.set_files(paths, self._why)
        standard = self._listing.standard
        self._standard.setText(
            "Never validated, whatever this file says: sourcedata/, code/, stimuli/, "
            "log/ and hidden files"
            + (f" (here: {', '.join(s + '/' for s in standard)})." if standard else "."))
        self._after_filter()
        self._paint_summary()

    def _paint_patterns(self) -> None:
        pal = CUR()
        focused = self._focused
        chosen = set(self._current_lines())
        self._patterns.blockSignals(True)
        self._patterns.clear()
        mono = _mono(self._patterns.font())
        for item in self._patterns_now:
            if item.comment and not item.raw.strip():
                continue
            row = QTreeWidgetItem([item.raw, ""])
            row.setFont(0, mono)
            row.setToolTip(0, item.raw)
            row.setData(0, Qt.ItemDataRole.UserRole, None if item.comment else item.raw)
            if item.comment:
                row.setForeground(0, QColor(pal["muted"]))
            elif item.is_redundant:
                row.setText(1, "already ignored")
                for c in (0, 1):
                    row.setForeground(c, QColor(pal["muted"]))
                row.setToolTip(1, "Every validator ignores this already, so the line does "
                                  "no harm and no work.")
            elif item.is_dead:
                row.setText(1, "matches nothing")
                row.setForeground(1, QColor(pal["warning"]))
                row.setToolTip(1, "This line hides no file of this dataset. That is "
                                  "usually a typo.")
            elif item.negate:
                row.setText(1, f"brings back {_files(item.count)}")
            else:
                row.setText(1, _files(item.count))
                row.setToolTip(1, "\n".join(item.matches[:30])
                               + (f"\nand {item.count - 30:,} more" if item.count > 30 else ""))
            self._patterns.addTopLevelItem(row)
            if item.raw in chosen:
                row.setSelected(True)
        if not self._patterns_now:
            empty = QTreeWidgetItem(["No .bidsignore yet: every file is validated.", ""])
            empty.setForeground(0, QColor(pal["muted"]))
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self._patterns.addTopLevelItem(empty)
        self._patterns.blockSignals(False)
        # The lines still there: their files recounted; gone: every file.
        raws = {p.raw for p in self._patterns_now}
        still = tuple(line for line in (focused or ()) if line in raws)
        self._set_focus(still or None)
        self._sync_buttons()

    def _paint_summary(self) -> None:
        if self._root is None or self._listing is None:
            return
        n = len(self._why)
        text = f"{_files(n)} of {len(self._listing.paths):,} not validated"
        hidden = bi.findings_hidden_by(self._report, self._why)
        if hidden:
            text += (f"  ·  the last validation has {hidden:,} finding(s) on them, gone "
                     "when it runs again")
        self._summary.setText(text)

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-render the rows, and make Qt recompute the QSS for the views.

        A list item's foreground is set on the item, so the app-wide
        stylesheet re-apply does not reach it and the rows have to be rebuilt
        from the new palette; an item view already polished keeps its
        background through a re-apply, hence the unpolish/polish dance.
        """
        del pal
        for widget in (self, self._patterns, self._files):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
        self._search.refresh_icon()
        self._mode.refresh_icons()
        if self._root is not None:
            self._paint_patterns()
            self._files.viewport().update()
        self.update()

    def set_read_only(self, on: bool) -> None:
        """A folder open for viewing: the patterns are shown, never changed."""
        self._read_only = bool(on)
        for w in (self._remove_btn, self._new_pattern, self._add_btn,
                  self._ignore_btn, self._stop_btn):
            w.setVisible(not on)

    # -- the files -----------------------------------------------------------

    def _on_mode(self, mode: str) -> None:
        self._apply_filter(mode=mode)

    def _apply_filter(self, **kwargs) -> None:
        self._model.set_filter(**kwargs)
        self._after_filter()

    def _after_filter(self) -> None:
        shown, total = self._model.shown(), self._model.total()
        self._count.setText(_files(total) if shown == total
                            else f"{shown:,} of {_files(total)}")
        self._sync_buttons()

    def _set_focus(self, lines: Optional[tuple[str, ...]]) -> None:
        """The files the lines match, or (None) every file."""
        self._focused = lines
        if not lines:
            self._focus_box.setVisible(False)
            self._patterns.blockSignals(True)
            self._patterns.setCurrentItem(None)
            self._patterns.clearSelection()
            self._patterns.blockSignals(False)
            self._apply_filter(only=None)
            return
        paths = self._listing.paths if self._listing else []
        hits = set()
        for line in lines:
            hits.update(bi.matches_for(line, paths))
        what = lines[0].strip() if len(lines) == 1 else f"these {len(lines)} lines"
        self._focus_label.setText(f"What {what} match{'es' if len(lines) == 1 else ''}: "
                                  f"{_files(len(hits))}")
        self._focus_box.setVisible(True)
        self._mode.set_current("all")
        self._apply_filter(only=hits, mode="all")

    def _selected(self) -> list[str]:
        rows = sorted({i.row() for i in self._files.selectionModel().selectedRows()})
        return [p for p in (self._model.path_at(r) for r in rows) if p]

    def _sync_buttons(self) -> None:
        picked = self._selected() if hasattr(self, "_model") else []
        self._ignore_btn.setEnabled(bool(picked) and any(
            self._model.why(p) is None for p in picked))
        self._stop_btn.setEnabled(bool(picked) and any(
            self._model.why(p) is not None for p in picked))
        lines = self._current_lines()
        self._remove_btn.setEnabled(bool(lines))
        self._remove_btn.setText("Remove line" if len(lines) <= 1
                                 else f"Remove {len(lines)} lines")

    def _menu_for(self, picked: list[str], *, stop: bool) -> Optional[QMenu]:
        from ..combo_popup import round_menu

        if self._root is None or self._listing is None or not picked:
            return None
        menu = QMenu(self)
        round_menu(menu)
        menu.setToolTipsVisible(True)
        if stop:
            previous = None
            for choice in bi.stop_choices(picked, self._listing.paths, self._root):
                if previous == "stop" and choice.key == "remove":
                    menu.addSeparator()          # in one step | removing a line
                previous = choice.key
                action = menu.addAction(choice.label)
                action.setToolTip(_choice_tip(choice))
                action.triggered.connect(lambda _c=False, c=choice: self._apply(c))
        else:
            for choice in bi.choices_for(picked, self._listing.paths):
                action = menu.addAction(f"{choice.label}\t{_files(choice.files)}")
                action.setToolTip("Adds " + ", ".join(choice.patterns))
                action.triggered.connect(lambda _c=False, c=choice: self._apply(c))
        return menu if not menu.isEmpty() else None

    def _popup(self, button: QPushButton, *, stop: bool) -> None:
        menu = self._menu_for(self._selected(), stop=stop)
        if menu is not None:
            menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _files_menu(self, pos) -> None:
        if self._read_only:
            return
        index = self._files.indexAt(pos)
        if not index.isValid():
            return
        picked = self._selected()
        path = self._model.path_at(index.row())
        if path not in picked:
            picked = [path]
        from ..combo_popup import round_menu

        menu = QMenu(self)
        round_menu(menu)
        added = False
        if any(self._model.why(p) is None for p in picked):
            ignore = self._menu_for(picked, stop=False)
            if ignore is not None:
                ignore.setTitle("Ignore in validation")
                menu.addMenu(ignore)
                added = True
        if any(self._model.why(p) is not None for p in picked):
            stop = self._menu_for(picked, stop=True)
            if stop is not None:
                stop.setTitle("Stop ignoring")
                menu.addMenu(stop)
                added = True
        if added:
            menu.exec(self._files.viewport().mapToGlobal(pos))

    # -- the lines -----------------------------------------------------------

    def _current_lines(self) -> list[str]:
        """The selected lines (a comment is not one)."""
        items = self._patterns.selectedItems() if hasattr(self, "_patterns") else []
        return [line for line in (i.data(0, Qt.ItemDataRole.UserRole) for i in items)
                if line is not None]

    def _current_line(self) -> Optional[str]:
        lines = self._current_lines()
        return lines[0] if lines else None

    def _on_pattern_selection(self) -> None:
        lines = tuple(self._current_lines())
        self._sync_buttons()
        if lines:
            self._set_focus(lines)
        elif self._focused is not None:
            self._set_focus(None)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt signature
        # Delete, and Backspace (the key a Mac calls delete), on the lines.
        from PyQt6.QtCore import QEvent

        if (obj is self._patterns and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and not self._read_only and self._current_line() is not None):
            self._on_remove()
            return True
        return super().eventFilter(obj, event)

    def _pattern_menu(self, pos) -> None:
        item = self._patterns.itemAt(pos)
        line = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if line is None or self._read_only:
            return
        from ..combo_popup import round_menu

        lines = self._current_lines() if item.isSelected() else [line]
        n = len(lines)
        menu = QMenu(self)
        round_menu(menu)
        menu.addAction("Show their files" if n > 1 else "Show its files").triggered.connect(
            lambda: self._set_focus(tuple(lines)))
        menu.addAction(f"Remove {n} lines" if n > 1 else "Remove line").triggered.connect(
            lambda: self._remove(lines))
        menu.exec(self._patterns.viewport().mapToGlobal(pos))

    def _on_preview_typed(self, text: str) -> None:
        text = text.strip()
        if not text or self._listing is None:
            self._preview.setText("")
            return
        hits = bi.matches_for(text, self._listing.paths)
        shown = ", ".join(hits[:_PREVIEW]) + (" ..." if len(hits) > _PREVIEW else "")
        self._preview.setText(f"Would hide {_files(len(hits))}" + (f": {shown}" if hits else
                              ". A pattern that matches nothing is usually a typo."))

    def _on_add_typed(self) -> None:
        text = self._new_pattern.text().strip()
        if not text:
            return
        if self._add([text]):
            self._new_pattern.clear()
            self._preview.setText("")

    def _on_remove(self) -> None:
        lines = self._current_lines()
        if lines:
            self._remove(lines)

    # -- writing -------------------------------------------------------------

    def _add(self, patterns: list[str]) -> bool:
        if self._root is None or self._read_only:
            return False
        added = bi.add_patterns(self._root, patterns)
        if added:
            self._after_write(f"Ignored in validation: {', '.join(added)}. Undo takes it back.")
        return bool(added)

    def _remove(self, patterns: list[str]) -> bool:
        if self._root is None or self._read_only:
            return False
        gone = bi.remove_patterns(self._root, patterns)
        if gone:
            self._after_write(f"Validated again: removed {', '.join(gone)} from .bidsignore.")
        return bool(gone)

    def _apply(self, choice) -> bool:
        """An ignore or stop-ignoring choice, written as one step."""
        if self._root is None or self._read_only:
            return False
        added, removed = bi.apply_choice(self._root, choice)
        if not (added or removed):
            return False
        from ..editor_panel import ignore_message

        self._after_write(ignore_message(choice, added, removed))
        return True

    def _after_write(self, message: str) -> None:
        self.reload()
        self.changed.emit(message)


def _choice_tip(choice) -> str:
    if choice.key == "remove":
        return "Removes the line: everything it hides is validated again."
    bits = []
    if choice.patterns:
        bits.append("removes " + ", ".join(choice.patterns))
    if choice.exceptions:
        bits.append("adds exceptions to a wider line (" + ", ".join(choice.exceptions[:3])
                    + (" ..." if len(choice.exceptions) > 3 else "") + ")")
    return "In .bidsignore, " + "; ".join(bits) + ". One step of the history."


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dlg-hint")
    label.setWordWrap(True)
    return label


__all__ = ["BidsIgnorePane"]
