"""A DICOM file's header, read for people.

The Editor's centre pane for a DICOM in the tree (a raw folder opened to
view, ``sourcedata/``), and the Converter's DICOM window (a double-click in
the raw data tree, "Show the DICOM header" on a row). Over the Qt-free
``inventory/dicom_header``:

* a row of facts: what the file is (modality, scanner, field strength,
  series, when, which object), what kind (enhanced, derived, non-image),
  and how many of its values identify the participant;
* the header grouped by MODULE (Patient, Study, Series, Equipment,
  Acquisition, Image, Private, File meta), sequences and the Siemens private
  headers unfolding, found by name, keyword, tag or value;
* identifying values hidden until asked for, and marked either way;
* "What BIDS Manager reads": only the elements the scan uses, with what for;
* an enhanced file's frames as a table;
* the series converted for a look, a comparison with another file, and a
  copy saved without the identifying values.

Read on a QThread (``JobRunner``): an enhanced header holds thousands of
elements. The tree is a model over plain lists, parented to its view
(CLAUDE.md guards 8c and 8e). Read only, always.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QAbstractItemModel, QAbstractTableModel, QModelIndex, QSize, Qt, pyqtSignal
from PyQt6.QtCore import QSortFilterProxyModel
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QStackedWidget, QTableView, QTreeView, QVBoxLayout, QWidget,
)

from ...inventory.dicom_header import DicomHeader, Difference, Element, count_identifying
from .primitives import PaneHeader
from .search_field import SearchField, find_shortcut

#: Where the shown value of a hidden identifier is.
HIDDEN = "hidden"

NAME, VALUE, TAG, VR, READ_FOR = range(5)
COLUMNS = ("Name", "Value", "Tag", "VR", "BIDS Manager reads it for")

SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1
READ_ROLE = Qt.ItemDataRole.UserRole + 2
ELEMENT_ROLE = Qt.ItemDataRole.UserRole + 3

_ABOUT = (
    "The header is what the scanner wrote about the image: who, when, how it was "
    "acquired. It is grouped by module (Patient, Study, Series, Equipment, Acquisition, "
    "Image, Private) rather than by tag number. Sequences unfold into their items, and "
    "the Siemens private headers (the CSA headers and the protocol) are decoded so "
    "they can be searched. Values that identify the participant are hidden until "
    "'Show identifying values' is on, and are never in a saved copy unless asked for. "
    "'What BIDS Manager reads' keeps only what the scan uses to name and sort the "
    "series. Nothing here is ever written to the file.")


def _qcolor(token: str) -> QColor:
    from ..theme_manager import CUR

    return QColor(CUR().get(token, CUR().get("text", "#e6edf3")))


# ---------------------------------------------------------------------------
# The header as a model
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("element", "module", "parent", "row", "children", "count")

    def __init__(self, element: Optional[Element], module: str,
                 parent: Optional["_Node"], row: int) -> None:
        self.element = element
        self.module = module
        self.parent = parent
        self.row = row
        self.children: list[_Node] = []
        self.count = 0


def _build(element: Element, module: str, parent: _Node, row: int) -> _Node:
    node = _Node(element, module, parent, row)
    node.children = [_build(c, module, node, i) for i, c in enumerate(element.children)]
    return node


class HeaderModel(QAbstractItemModel):
    """The header's modules and their elements, five columns."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._roots: list[_Node] = []
        self._show_identifying = False
        self._read_column = False
        self.refresh_style()

    # -- content ---------------------------------------------------------------

    def set_header(self, header: Optional[DicomHeader]) -> None:
        self.beginResetModel()
        roots: list[_Node] = []
        if header is not None:
            for module, items in header.modules.items():
                node = _Node(None, module, None, len(roots))
                node.children = [_build(e, module, node, i) for i, e in enumerate(items)]
                node.count = len(items)
                roots.append(node)
        self._roots = roots
        self.endResetModel()

    def set_show_identifying(self, on: bool) -> None:
        if on == self._show_identifying:
            return
        self._show_identifying = on
        self._everything_changed()

    def show_identifying(self) -> bool:
        return self._show_identifying

    def refresh_style(self) -> None:
        """Fonts and colours from the theme and the font-size setting."""
        from ..viz import fonts

        self._mono = fonts.font(11, mono=True)
        self._mono_small = fonts.font(10, mono=True)
        self._bold = fonts.font(11, bold=True)
        side = fonts.px(16)
        blank = QPixmap(side, side)
        blank.fill(Qt.GlobalColor.transparent)
        self._blank = QIcon(blank)
        self._colours = {k: _qcolor(k) for k in ("text", "dim", "muted", "warning")}
        self._everything_changed()

    def _everything_changed(self) -> None:
        if self._roots:
            self.layoutAboutToBeChanged.emit()
            self.layoutChanged.emit()

    # -- Qt model --------------------------------------------------------------

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        siblings = self._roots if not parent.isValid() else parent.internalPointer().children
        if 0 <= row < len(siblings) and 0 <= column < len(COLUMNS):
            return self.createIndex(row, column, siblings[row])
        return QModelIndex()

    def parent(self, index: QModelIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        if not index.isValid():
            return QModelIndex()
        node: _Node = index.internalPointer()
        if node.parent is None:
            return QModelIndex()
        return self.createIndex(node.parent.row, 0, node.parent)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt
        if not parent.isValid():
            return len(self._roots)
        if parent.column() != 0:
            return 0
        return len(parent.internalPointer().children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation,  # noqa: N802 - Qt
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def hidden(self, element: Element) -> bool:
        """Whether ``element``'s value is shown as hidden."""
        return element.identifies and not element.children and not self._show_identifying

    def shown_value(self, element: Element) -> str:
        return HIDDEN if self.hidden(element) and element.value else element.value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        node: _Node = index.internalPointer()
        col = index.column()
        e = node.element
        if e is None:
            return self._module_data(node, col, role)
        if role == Qt.ItemDataRole.DisplayRole:
            return (e.name, self.shown_value(e), e.tag, e.vr, e.read_for)[col]
        if role == Qt.ItemDataRole.ForegroundRole:
            if col in (NAME, VALUE) and e.identifies:
                return self._colours["warning"]
            if col == VALUE and self.hidden(e):
                return self._colours["muted"]
            if col in (TAG, VR, READ_FOR):
                return self._colours["dim"]
            return None
        if role == Qt.ItemDataRole.FontRole:
            if col == VALUE:
                return self._mono
            if col in (TAG, VR):
                return self._mono_small
            return None
        if role == Qt.ItemDataRole.DecorationRole and col == NAME:
            from .. import icons

            if e.identifies:
                return icons.icon("identifies")
            if e.read_for:
                return icons.icon("read_by_scan")
            # The same room on every row, so the names line up.
            return self._blank
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(e, col)
        if role == SEARCH_ROLE:
            value = "" if self.hidden(e) else e.value
            return " ".join((e.name, e.keyword, e.tag, value)).lower()
        if role == READ_ROLE:
            return bool(e.read_for)
        if role == ELEMENT_ROLE:
            return e
        return None

    def _module_data(self, node: _Node, col: int, role: int) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if col == NAME:
                return node.module
            if col == VALUE:
                return f"{node.count} element" + ("" if node.count == 1 else "s")
            return ""
        if role == Qt.ItemDataRole.FontRole and col == NAME:
            return self._bold
        if role == Qt.ItemDataRole.ForegroundRole and col == VALUE:
            return self._colours["dim"]
        if role == SEARCH_ROLE:
            return node.module.lower()
        if role == READ_ROLE:
            return False
        return None

    def _tooltip(self, e: Element, col: int) -> str:
        if col == VALUE:
            if self.hidden(e):
                return ("Identifies the participant. Turn on 'Show identifying values' to "
                        "see it.")
            return e.value if len(e.value) > 40 else ""
        lines = [e.name]
        if e.keyword and e.keyword != e.name:
            lines.append(f"Keyword: {e.keyword}")
        if e.tag:
            lines.append(f"Tag: {e.tag}" + (f", VR {e.vr}" if e.vr else ""))
        if e.identifies:
            lines.append("Identifies the participant.")
        if e.read_for:
            lines.append(f"BIDS Manager reads it for {e.read_for}.")
        return "\n".join(lines)


class HeaderFilter(QSortFilterProxyModel):
    """Rows matching the search text (and, when asked, only what BIDS
    Manager reads), with their parents, and the children of a match."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRecursiveFilteringEnabled(True)
        self.setAutoAcceptChildRows(True)
        self._text = ""
        self._read_only = False

    def _change(self, fn) -> None:
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()
            fn()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:  # Qt before 6.9
            fn()
            self.invalidateRowsFilter()

    def set_text(self, text: str) -> None:
        text = text.strip().lower()
        if text != self._text:
            self._change(lambda: setattr(self, "_text", text))

    def text(self) -> str:
        return self._text

    def set_read_only(self, on: bool) -> None:
        if on != self._read_only:
            self._change(lambda: setattr(self, "_read_only", on))

    def read_only(self) -> bool:
        return self._read_only

    def refilter(self) -> None:
        """Again, after what is searchable changed (values shown or hidden)."""
        self._change(lambda: None)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802 - Qt
        idx = self.sourceModel().index(row, 0, parent)
        if self._read_only and not idx.data(READ_ROLE):
            return False
        if self._text and self._text not in (idx.data(SEARCH_ROLE) or ""):
            return False
        return True


class TableModel(QAbstractTableModel):
    """Rows of strings under named columns (the frames, the differences)."""

    def __init__(self, columns: tuple[str, ...] = (), parent=None) -> None:
        super().__init__(parent)
        self._columns = list(columns)
        self._rows: list[list[str]] = []
        self._mono_columns: set[int] = set()
        self._colour_rows: dict[int, str] = {}

    def set_rows(self, columns, rows, *, mono=(), colour_rows=None) -> None:
        self.beginResetModel()
        self._columns = list(columns)
        self._rows = [list(r) for r in rows]
        self._mono_columns = set(mono)
        self._colour_rows = dict(colour_rows or {})
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation,  # noqa: N802 - Qt
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._columns[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            text = self._rows[index.row()][index.column()]
            return text if role == Qt.ItemDataRole.DisplayRole or len(text) > 40 else None
        if role == Qt.ItemDataRole.FontRole and index.column() in self._mono_columns:
            from ..viz import fonts

            return fonts.font(11, mono=True)
        if role == Qt.ItemDataRole.ForegroundRole and index.row() in self._colour_rows:
            return _qcolor(self._colour_rows[index.row()])
        return None

    def row(self, i: int) -> list[str]:
        return self._rows[i]


def _tree(object_name: str) -> QTreeView:
    view = QTreeView()
    view.setObjectName(object_name)
    view.setUniformRowHeights(True)
    view.setAlternatingRowColors(False)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setTextElideMode(Qt.TextElideMode.ElideRight)
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    view.header().setStretchLastSection(False)
    view.header().setSectionsMovable(False)
    return view


def _table(object_name: str) -> QTableView:
    view = QTableView()
    view.setObjectName(object_name)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    view.setShowGrid(False)
    view.setWordWrap(False)
    view.horizontalHeader().setStretchLastSection(True)
    view.horizontalHeader().setHighlightSections(False)
    return view


def _toggle(text: str, tip: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("tb-btn-toggle")
    b.setCheckable(True)
    b.setToolTip(tip)
    return b


def _button(text: str, tip: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("tb-btn")
    b.setToolTip(tip)
    return b


def _message_page(text_label: QLabel) -> QWidget:
    page = QWidget()
    page.setObjectName("pane")
    lay = QVBoxLayout(page)
    lay.addStretch(1)
    text_label.setObjectName("pane-hint")
    text_label.setWordWrap(True)
    text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(text_label)
    lay.addStretch(1)
    return page


def _pending_page(message: str):
    from . import BusySpinner

    page = QWidget()
    page.setObjectName("pane")
    lay = QVBoxLayout(page)
    lay.addStretch(1)
    spinner = BusySpinner()
    spinner.set_busy(True, message=message)
    lay.addWidget(spinner, 0, Qt.AlignmentFlag.AlignHCenter)
    lay.addStretch(1)
    return page, spinner


def _copy(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


# ---------------------------------------------------------------------------
# The pane
# ---------------------------------------------------------------------------


class DicomPane(QWidget):
    """The header of one DICOM file."""

    loading_changed = pyqtSignal(bool, str)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ..viz.bridge import JobRunner, ThemeHub, connect_while_alive
        from ..viz.panels.panel_header import PanelHeader, corner_button

        self.setObjectName("pane")
        self._path: Optional[Path] = None
        self._header: Optional[DicomHeader] = None
        self._generation = 0
        self._preview_generation = 0
        self._preview_dir: Optional[Path] = None
        self.preview_dialog = None
        self.compare_dialog = None
        self._include_identifying_in_copies = False
        self._runner = JobRunner(self)
        self._runner.done.connect(self._on_job_done)
        self._runner.failed.connect(self._on_job_failed)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._title = PaneHeader("DICOM header")
        v.addWidget(self._title)

        # What the file is; the pane's own controls in the corner.
        self.facts = PanelHeader(margins=(12, 6, 8, 6))
        info = corner_button("info", "What the DICOM header is, and what this view shows")
        info.clicked.connect(lambda: self._explain(info))
        self.more_button = corner_button("more", "Copy or save the header")
        self.more_button.clicked.connect(self._more_menu)
        self.facts.corner.addWidget(info)
        self.facts.corner.addWidget(self.more_button)
        v.addWidget(self.facts)

        # How to look at it, and what to do with it.
        self.controls = PanelHeader(margins=(12, 2, 8, 6), h_spacing=8)
        bar = self.controls.controls
        self.header_pill = QPushButton("Header")
        self.frames_pill = QPushButton("Frames")
        self._pills = QButtonGroup(self)
        self._pills.setExclusive(True)
        for i, pill in enumerate((self.header_pill, self.frames_pill)):
            pill.setObjectName("view-pill")
            pill.setCheckable(True)
            self._pills.addButton(pill, i)
            bar.addWidget(pill)
        self.header_pill.setChecked(True)
        self.header_pill.setToolTip("Every element of the header, grouped by module")
        self.frames_pill.setToolTip("One row per frame of this enhanced (multi-frame) file: "
                                    "where it is, its b-value, echo time and time point")
        self._pills.idClicked.connect(self._on_pill)
        self.search = SearchField("Find a name, keyword, tag or value")
        self.search.setToolTip("Shows the elements whose name, keyword, tag or value "
                               "contains this, with the sequences that hold them")
        self.search.textChanged.connect(self._on_search)
        bar.addWidget(self.search)
        self.read_toggle = _toggle(
            "What BIDS Manager reads",
            "Only the elements the scan reads to name, sort and check the series, each with "
            "what it is read for")
        self.read_toggle.toggled.connect(self._on_read_only)
        bar.addWidget(self.read_toggle)
        self.identifying_toggle = _toggle(
            "Show identifying values",
            "Values that identify the participant (name, ID, birth date, address) are hidden "
            "until this is on. They are marked either way.")
        self.identifying_toggle.toggled.connect(self._on_show_identifying)
        bar.addWidget(self.identifying_toggle)
        self.image_button = _button(
            "Show the image",
            "Convert this file's series for a look, in a window of its own. Nothing is "
            "written anywhere you keep data.")
        self.image_button.clicked.connect(self.show_image)
        bar.addWidget(self.image_button)
        self.compare_button = _button(
            "Compare with...",
            "Pick another DICOM file and list every element that differs between the two")
        self.compare_button.clicked.connect(self._pick_compare)
        bar.addWidget(self.compare_button)
        v.addWidget(self.controls)

        rule = QFrame()
        rule.setObjectName("dicom-rule")
        rule.setFixedHeight(1)
        v.addWidget(rule)

        # Reading, the header, the frames, or why there is nothing.
        self._stack = QStackedWidget()
        self._pending, self._spinner = _pending_page("Reading the header...")
        self._stack.addWidget(self._pending)
        self.tree = _tree("dicom-header-tree")
        self.model = HeaderModel(self.tree)
        self.proxy = HeaderFilter(self.tree)
        self.proxy.setSourceModel(self.model)
        self.tree.setModel(self.proxy)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self._stack.addWidget(self.tree)
        self.frames = _table("dicom-frames")
        self.frames_model = TableModel(parent=self.frames)
        self.frames.setModel(self.frames_model)
        self._stack.addWidget(self.frames)
        self.message = QLabel("")
        self._message_page = _message_page(self.message)
        self._stack.addWidget(self._message_page)
        v.addWidget(self._stack, 1)

        find_shortcut(self, self.search)
        copy = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self.tree)
        copy.setContext(Qt.ShortcutContext.WidgetShortcut)
        copy.activated.connect(self.copy_selection)

        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w._apply_theme())
        self._apply_theme()
        self._show_message("No DICOM file chosen.")

    # -- binding ---------------------------------------------------------------

    def set_read_only(self, on: bool) -> None:
        """Always read only; here for the Editor's mode switch."""
        del on

    def current_file(self) -> Optional[Path]:
        return self._path

    def header(self) -> Optional[DicomHeader]:
        return self._header

    def set_file(self, path: Optional[Path], root: Optional[Path] = None) -> None:
        """Read ``path``'s header on a thread and show it (``None`` clears)."""
        from ...inventory.dicom_header import read_header

        del root
        path = Path(path) if path is not None else None
        if path is not None and path == self._path and self._header is not None:
            return
        self._generation += 1
        self._runner.cancel("header")
        self._path = path
        self._header = None
        self.model.set_header(None)
        self._fill_facts()
        if path is None:
            self._title.setText("DICOM header")
            self._show_message("No DICOM file chosen.")
            self._set_actions()
            return
        self._title.setText(path.name)
        self._stack.setCurrentWidget(self._pending)
        self._set_actions()
        self.loading_changed.emit(True, "Reading the DICOM header...")
        self._runner.start("header", self._generation, read_header, path)

    def is_loading(self) -> bool:
        return self._runner.running("header")

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "header":
            if generation != self._generation:
                return
            self._header = result
            self.loading_changed.emit(False, "")
            self.model.set_header(result)
            self._fill_facts()
            self._fill_frames()
            self._set_actions()
            self._on_pill(self._pills.checkedId())
            self._expand()
            self._fit_columns()
        elif tag == "image" and generation == self._preview_generation:
            self._on_image_ready(result)

    def _on_job_failed(self, tag: str, generation: int, msg: str) -> None:
        if tag == "header":
            if generation != self._generation:
                return
            self.loading_changed.emit(False, "")
            name = self._path.name if self._path is not None else "This file"
            self._show_message(f"{name} could not be read as DICOM.\n\n{msg}")
            self._set_actions()
        elif tag == "image" and generation == self._preview_generation:
            self._on_image_failed(msg)

    # -- the facts -------------------------------------------------------------

    def _fill_facts(self) -> None:
        from .primitives import CappedLabel, Chip

        bar = self.facts.controls
        for w in list(bar.widgets()):
            bar.removeWidget(w)
            w.deleteLater()
        h = self._header
        if h is None:
            return
        from ..viz import fonts

        tips = {"Modality": "The modality the scanner recorded",
                "Scanner": "Manufacturer and model",
                "Field strength": "Magnetic field strength",
                "Series": "Series number and description",
                "Acquired": "When it was acquired",
                "Object": "What kind of DICOM object this is (its SOP class)"}
        shown = {"Series": "Series {}", "Acquired": "Acquired {}"}
        for key, value in h.summary.items():
            label = CappedLabel(fonts.px(300), shown.get(key, "{}").format(value))
            label.setObjectName("dicom-fact-strong" if key == "Modality" else "dicom-fact")
            label.setToolTip(f"{tips.get(key, key)}: {value}")
            bar.addWidget(label)
        kinds = {"Enhanced multi-frame": ("teal", "Every frame of the series is in this one "
                                          "file; the Frames view lists them"),
                 "Derived": ("purple", "Computed by the scanner from other images (a map, a "
                             "projection, a reformat), not acquired"),
                 "Non-image object": ("warn", "Holds no image (a report, a protocol, "
                                      "spectroscopy): the scan leaves it out")}
        for flag in h.flags:
            kind, tip = kinds.get(flag, ("default", flag))
            chip = Chip(flag, kind)
            chip.setToolTip(tip)
            bar.addWidget(chip)
        n = count_identifying(h)
        if n:
            chip = Chip(f"{n} identifying value" + ("" if n == 1 else "s"), "warn")
            chip.setToolTip("Values that identify the participant. They are marked in the "
                            "header, hidden until shown, and left out of a saved copy.")
            bar.addWidget(chip)
        self.identifying_chip_count = n

    def facts_text(self) -> list[str]:
        """What the facts row says, in order (tests)."""
        return [w.text() for w in self.facts.controls.widgets() if hasattr(w, "text")]

    # -- the views -------------------------------------------------------------

    def _fill_frames(self) -> None:
        frames = self._header.frames if self._header is not None else []
        cols = [c for c in (frames[0] if frames else {})
                if any(f.get(c) for f in frames)]
        self.frames_model.set_rows([c.capitalize() for c in cols],
                                   [[f.get(c, "") for c in cols] for f in frames],
                                   mono=range(1, len(cols)))
        self.frames.horizontalHeader().resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        self.frames_pill.setText(f"Frames · {len(frames)}" if frames else "Frames")
        for pill in (self.header_pill, self.frames_pill):
            pill.setVisible(bool(frames))
        if not frames and self._pills.checkedId() == 1:
            self.header_pill.setChecked(True)

    def _on_pill(self, which: int) -> None:
        if self._header is None:
            return
        self._stack.setCurrentWidget(self.frames if which == 1 else self.tree)
        for w in (self.search, self.read_toggle, self.identifying_toggle):
            w.setVisible(which != 1)

    def show_frames(self) -> None:
        self.frames_pill.setChecked(True)
        self._on_pill(1)

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        self._stack.setCurrentWidget(self._message_page)

    def _set_actions(self) -> None:
        ready = self._header is not None
        for w in (self.search, self.read_toggle, self.identifying_toggle, self.compare_button,
                  self.more_button):
            w.setEnabled(ready)
        image = ready and "Non-image object" not in self._header.flags
        self.image_button.setEnabled(image and self._preview_dir is None)
        if ready and not image:
            self.image_button.setToolTip("This object holds no image to show")
        if not ready:
            for pill in (self.header_pill, self.frames_pill):
                pill.setVisible(False)

    def _expand(self) -> None:
        """Modules open; sequences and decoded headers closed, unless
        searching, which opens everything that matched."""
        if self.proxy.text() or self.proxy.read_only():
            self.tree.expandAll()
        else:
            self.tree.collapseAll()
            self.tree.expandToDepth(0)

    def _fit_columns(self) -> None:
        header = self.tree.header()
        for col in (NAME, READ_FOR):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(VALUE, QHeaderView.ResizeMode.Stretch)
        for col in (TAG, VR):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setColumnHidden(READ_FOR, not self.proxy.read_only())
        self._fit_names()

    def _fit_names(self) -> None:
        """The name column (and "reads it for") as wide as what is shown
        needs, within a share of the width, so the values keep the rest
        however narrow the pane is (a decoded protocol's keys are long and
        deep)."""
        from ..viz import fonts

        room = max(self.tree.viewport().width(), fonts.px(300))
        want = self.tree.sizeHintForColumn(NAME) + fonts.px(8)
        self.tree.setColumnWidth(NAME, max(min(fonts.px(160), int(room * 0.4)),
                                           min(want, int(room * 0.42), fonts.px(460))))
        if not self.tree.isColumnHidden(READ_FOR):
            want = self.tree.sizeHintForColumn(READ_FOR) + fonts.px(8)
            self.tree.setColumnWidth(READ_FOR, min(want, int(room * 0.28)))

    def _on_search(self, text: str) -> None:
        self.proxy.set_text(text)
        self._expand()
        self._fit_names()

    def set_search(self, text: str) -> None:
        self.search.setText(text)

    def _on_read_only(self, on: bool) -> None:
        self.proxy.set_read_only(on)
        self.tree.setColumnHidden(READ_FOR, not on)
        self._expand()
        self._fit_names()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._header is not None:
            self._fit_names()

    def _on_show_identifying(self, on: bool) -> None:
        self.model.set_show_identifying(on)
        self.proxy.refilter()

    # -- copying ---------------------------------------------------------------

    def _selected_elements(self) -> list[Element]:
        rows = self.tree.selectionModel().selectedRows(0) if self.tree.selectionModel() else []
        return [e for e in (self.proxy.data(i, ELEMENT_ROLE) for i in rows) if e is not None]

    def copy_selection(self) -> None:
        """The chosen elements as ``name: value`` lines."""
        lines = [f"{e.name}: {self.model.shown_value(e)}" for e in self._selected_elements()]
        if lines:
            _copy("\n".join(lines))

    def _tree_menu(self, pos) -> None:
        from ..viz.menus import popup_menu

        index = self.tree.indexAt(pos)
        e = self.proxy.data(index.siblingAtColumn(0), ELEMENT_ROLE) if index.isValid() else None
        if e is None:
            return
        menu = popup_menu(self.tree)
        hidden = self.model.hidden(e)
        value = menu.addAction("Copy value (hidden)" if hidden else "Copy value")
        value.setEnabled(not hidden and bool(e.value))
        value.triggered.connect(lambda: _copy(e.value))
        menu.addAction("Copy name and value").triggered.connect(
            lambda: _copy(f"{e.name}: {self.model.shown_value(e)}"))
        if e.tag:
            menu.addAction("Copy tag").triggered.connect(lambda: _copy(e.tag))
        if e.keyword:
            menu.addAction("Copy keyword").triggered.connect(lambda: _copy(e.keyword))
        menu.popup(self.tree.viewport().mapToGlobal(pos))

    def _more_menu(self) -> None:
        from .. import icons
        from ..viz.menus import popup_menu

        menu = popup_menu(self.more_button)
        menu.addAction(icons.icon("copy"), "Copy the header as text").triggered.connect(
            lambda: _copy(self.text_copy()))
        menu.addAction(icons.icon("export"), "Save as text...").triggered.connect(
            lambda: self._save("text"))
        menu.addAction(icons.icon("export"), "Save as JSON...").triggered.connect(
            lambda: self._save("json"))
        menu.addSeparator()
        include = menu.addAction("Include identifying values")
        include.setCheckable(True)
        include.setChecked(self._include_identifying_in_copies)
        include.setToolTip("Copies and saved files leave identifying values out unless this "
                           "is on")
        include.toggled.connect(lambda on: setattr(self, "_include_identifying_in_copies", on))
        menu.popup(self.more_button.mapToGlobal(self.more_button.rect().bottomLeft()))
        self.more_menu = menu

    def text_copy(self) -> str:
        from ...inventory.dicom_header import export_text

        if self._header is None:
            return ""
        return export_text(self._header, identifiers=self._include_identifying_in_copies)

    def save_copy(self, target: Path, form: str = "text") -> None:
        from ...inventory.dicom_header import export_json, export_text

        if self._header is None:
            return
        make = export_json if form == "json" else export_text
        Path(target).write_text(
            make(self._header, identifiers=self._include_identifying_in_copies),
            encoding="utf-8")
        self.status_message.emit(f"Saved {Path(target).name}")

    def _save(self, form: str) -> None:
        if self._header is None or self._path is None:
            return
        suffix = ".json" if form == "json" else ".txt"
        start = str(Path.home() / (self._path.stem + "_header" + suffix))
        target, _ = QFileDialog.getSaveFileName(
            self, "Save the DICOM header", start,
            "JSON (*.json)" if form == "json" else "Text (*.txt)")
        if target:
            self.save_copy(Path(target), form)

    def _explain(self, anchor: QWidget) -> None:
        from ..viz.panels import explain_popup

        explain_popup.show(anchor, "dicom.header", title="DICOM header", fallback=_ABOUT)

    # -- the image -------------------------------------------------------------

    def show_image(self) -> None:
        """The file's series converted for a look, in a window that opens
        now and fills when it is ready (GUI_LOOK_AND_FEEL 4.22)."""
        import tempfile

        from ...inventory.probe_convert import preview_dicom
        from ..preview_dialog import PreviewDialog

        if self._path is None or self._header is None or self._preview_dir is not None:
            return
        self._preview_dir = Path(tempfile.mkdtemp(prefix="bidsmgr-preview-"))
        self._preview_generation += 1
        title = self._header.summary.get("Series") or self._path.name
        dlg = PreviewDialog(title, self._preview_dir, parent=self)
        self.preview_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._runner.start("image", self._preview_generation, preview_dicom, self._path,
                           self._preview_dir)
        self._set_actions()

    def _preview_window(self):
        dlg = self.preview_dialog
        return dlg if dlg is not None and dlg.isVisible() and dlg.is_pending() else None

    def _on_image_ready(self, files) -> None:
        dlg = self._preview_window()
        if dlg is not None:
            dlg.set_files(files, self._preview_dir)
            from PyQt6.QtWidgets import QApplication

            QApplication.alert(dlg)
        elif self._preview_dir is not None:
            shutil.rmtree(self._preview_dir, ignore_errors=True)
        self._preview_dir = None
        self._set_actions()

    def _on_image_failed(self, msg: str) -> None:
        if self._preview_dir is not None:
            shutil.rmtree(self._preview_dir, ignore_errors=True)
        self._preview_dir = None
        dlg = self._preview_window()
        if dlg is not None:
            dlg.set_failed(msg)
        self._set_actions()

    # -- comparing -------------------------------------------------------------

    def _pick_compare(self) -> None:
        if self._path is None:
            return
        other, _ = QFileDialog.getOpenFileName(self, "Compare with another DICOM file",
                                               str(self._path.parent))
        if other:
            self.compare_with(Path(other))

    def compare_with(self, other: Path) -> "DicomCompareDialog":
        dlg = DicomCompareDialog(self._header, Path(other), parent=self,
                                 show_identifying=self.model.show_identifying())
        self.compare_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    # -- theme -----------------------------------------------------------------

    def _apply_theme(self) -> None:
        from ..viz import fonts

        self.model.refresh_style()
        self.search.refresh_icon()
        side = fonts.px(16)
        self.tree.setIconSize(QSize(side, side))
        self.tree.setIndentation(fonts.px(16))
        self._fill_facts()

    def repaint_for_palette(self, pal: dict) -> None:
        """QSS-styled children re-polished after a theme switch."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()


# ---------------------------------------------------------------------------
# Comparing two files
# ---------------------------------------------------------------------------


class DicomCompareDialog(QDialog):
    """Every element that differs between two DICOM files, by module."""

    def __init__(self, header: DicomHeader, other: Path, parent: Optional[QWidget] = None, *,
                 show_identifying: bool = False) -> None:
        super().__init__(parent)
        from ...inventory.dicom_header import read_header
        from ..viz.bridge import JobRunner
        from ..window_flags import give_window_buttons
        from .card import card

        give_window_buttons(self)
        self.setWindowTitle(f"Compare: {header.path.name} and {other.name}")
        self.resize(1040, 640)
        self._a = header
        self._b: Optional[DicomHeader] = None
        self._rows: list[Difference] = []
        self._shown: list[Difference] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        names = QHBoxLayout()
        names.setSpacing(16)
        for title, path in (("This file", header.path), ("Other file", other)):
            label = QLabel(f"{title}: {path.name}")
            label.setObjectName("dicom-fact")
            label.setToolTip(str(path))
            names.addWidget(label)
        names.addStretch(1)
        v.addLayout(names)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.search = SearchField("Find an element or a value")
        self.search.textChanged.connect(lambda _t: self._fill())
        row.addWidget(self.search, 1)
        from ..viz.bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self,
                            lambda w, _t: w.search.refresh_icon())
        self.routine_toggle = _toggle(
            "Hide UIDs, times and positions",
            "UIDs, instance numbers, clocks and slice positions differ between any two "
            "images of a scan; hidden, what is left is what makes the two different")
        self.routine_toggle.setChecked(True)
        self.routine_toggle.toggled.connect(lambda _on: self._fill())
        row.addWidget(self.routine_toggle)
        self.identifying_toggle = _toggle(
            "Show identifying values", "Values that identify the participant are hidden "
            "until this is on")
        self.identifying_toggle.setChecked(show_identifying)
        self.identifying_toggle.toggled.connect(lambda _on: self._fill())
        row.addWidget(self.identifying_toggle)
        v.addLayout(row)

        self.summary = QLabel("")
        self.summary.setObjectName("dlg-hint")
        v.addWidget(self.summary)

        self._stack = QStackedWidget()
        pending, self._spinner = _pending_page(f"Reading {other.name}...")
        self._stack.addWidget(pending)
        self.table = _table("dicom-compare")
        self.model = TableModel(parent=self.table)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(False)
        self._stack.addWidget(self.table)
        self.message = QLabel("")
        self._stack.addWidget(_message_page(self.message))
        v.addWidget(card(self._stack), 1)

        foot = QHBoxLayout()
        foot.addStretch(1)
        close = _button("Close", "")
        close.clicked.connect(self.close)
        foot.addWidget(close)
        v.addLayout(foot)

        self._runner = JobRunner(self)
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)
        self._runner.start("other", 1, read_header, other)

    def is_pending(self) -> bool:
        return self._b is None and self._stack.currentIndex() == 0

    def differences(self) -> list[Difference]:
        """What is listed now (after the toggles and the search)."""
        return list(self._shown)

    def _on_done(self, _tag: str, _generation: int, header) -> None:
        from ...inventory.dicom_header import compare

        self._b = header
        self._rows = compare(self._a, header)
        self._stack.setCurrentWidget(self.table)
        self._fill()

    def _on_failed(self, _tag: str, _generation: int, msg: str) -> None:
        self.message.setText(f"The other file could not be read as DICOM.\n\n{msg}")
        self._stack.setCurrentIndex(2)

    def _fill(self) -> None:
        if self._b is None:
            self._shown = []
            return
        text = self.search.text().strip().lower()
        hide_routine = self.routine_toggle.isChecked()
        show_id = self.identifying_toggle.isChecked()

        def value(d: Difference, v: Optional[str]) -> str:
            if v is None:
                return "(not in this file)"
            if d.identifies and not show_id and v:
                return HIDDEN
            return v

        shown = []
        for d in self._rows:
            if hide_routine and d.routine:
                continue
            a, b = value(d, d.a), value(d, d.b)
            if text and text not in " ".join((d.module, d.where, d.tag, a, b)).lower():
                continue
            shown.append((d, a, b))
        self._shown = [d for d, _a, _b in shown]
        colours = {i: "warning" for i, (d, _a, _b) in enumerate(shown) if d.identifies}
        self.model.set_rows(("Module", "Element", "Tag", "This file", "Other file"),
                            [(d.module, d.where, d.tag, a, b) for d, a, b in shown],
                            mono=(2, 3, 4), colour_rows=colours)
        header = self.table.horizontalHeader()
        for col in (0, 2):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col in (1, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        hidden = sum(1 for d in self._rows if d.routine) if hide_routine else 0
        n = len(shown)
        parts = [f"{n} difference" + ("" if n == 1 else "s")]
        if hidden:
            parts.append(f"{hidden} that differ between any two images hidden")
        if not self._rows:
            parts = ["The two headers are the same"]
        self.summary.setText(", ".join(parts) + ".")


# ---------------------------------------------------------------------------
# A window of its own (the Converter)
# ---------------------------------------------------------------------------


class DicomWindow(QDialog):
    """The DICOM pane in a window of its own, reused for the next file."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from ..window_flags import give_window_buttons
        from .card import card

        give_window_buttons(self)
        self.setModal(False)
        self.setWindowTitle("DICOM header")
        self.resize(1000, 720)
        self.pane = DicomPane()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(card(self.pane, padding=1))

    def show_file(self, path: Path) -> None:
        self.pane.set_file(Path(path))
        self.setWindowTitle(f"DICOM header: {Path(path).name}")
        self.show()
        self.raise_()
        self.activateWindow()


__all__ = ["DicomCompareDialog", "DicomPane", "DicomWindow", "HeaderFilter", "HeaderModel"]
