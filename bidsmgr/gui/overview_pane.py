"""Inspection > Overview: what each subject will have, and what differs.

The second view of the Inspection area; the first is the table. It reads the
same inventory model and answers what a table of rows cannot: whether the
subjects have the same recordings, whether a task was spelled two ways, whether
one visit became several sessions (:mod:`bidsmgr.inventory.overview`).

Three parts, top to bottom: the headline counts, the findings worth a look
(each with Show, which selects its rows and opens the table), and a grid of
subjects (and sessions) by kind of recording. Clicking a count selects its
recordings, so Properties shows them beside the grid; double-clicking opens
them in the table.

It is computed on a ``QThread`` (``JobRunner``) from a snapshot of the table,
only while the Overview is shown, and again when the table changes; the last
result stays on screen meanwhile. The grid is ONE ``QTableView`` whose cells
and headers are painted (CLAUDE.md guard 8e: no widget per cell), its model
parented to the view (8c). Painted colours come from the palette at paint
time and fonts from ``gui/viz/fonts.py``, so a theme or font-size change is a
restyle and a repaint (``repaint_for_palette``, which the Converter calls for
both).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QRect, QRectF, QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..inventory.overview import LEVEL_TITLES, LEVELS, Cell, Finding, Overview, build_overview
from . import icons
from .dialog_chrome import WrapLabel, stat
from .theme_manager import CUR, scaled_px
from .widgets.flow_layout import FlowBar
from .widgets.primitives import qcolor
from .widgets.spinner import BusySpinner

#: Findings shown before "Show all".
FINDINGS_SHOWN = 4
#: How long the table must have been quiet before the overview is recounted.
_SETTLE_MS = 150
#: How long a first count may take before a spinner says so.
_PENDING_MS = 250
_LEVEL_KEY = "inspector/overview_level"
_FOLD_KEY = "inspector/overview_findings_folded"
_CELL_ROLE = Qt.ItemDataRole.UserRole + 1
_EMPTY = Cell()
_SEVERITY_ICON = {"err": "err", "warn": "warn", "info": "info"}


def _font(base_px: int, *, bold: bool = False) -> QFont:
    """A painted font through the shared helper (GUI_LOOK_AND_FEEL 3.1)."""
    from .viz import fonts

    return fonts.font(base_px, bold=bold)


def _px(base: int) -> int:
    return scaled_px(base)


def _level_text(key: str, chosen: str = "") -> str:
    """What the column chooser says: ``Columns: every label``, and for the
    automatic choice the level it took."""
    if key == "auto":
        return "Columns: automatic" + (f" ({LEVEL_TITLES[chosen].lower()})" if chosen else "")
    return f"Columns: {LEVEL_TITLES[key].lower()}"


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


class _GridModel(QAbstractTableModel):
    """Visits down, kinds of recording across; a cell is a :class:`Cell`."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._ov: Optional[Overview] = None
        self._group_starts: set[int] = set()
        self._subject_starts: set[int] = set()

    def set_overview(self, ov: Optional[Overview]) -> None:
        self.beginResetModel()
        self._ov = ov
        kinds = ov.kinds if ov else []
        visits = ov.visits if ov else []
        self._group_starts = {c for c, k in enumerate(kinds)
                              if c == 0 or k.datatype != kinds[c - 1].datatype}
        self._subject_starts = {r for r, v in enumerate(visits)
                                if r == 0 or v.subject != visits[r - 1].subject}
        self.endResetModel()

    @property
    def overview(self) -> Optional[Overview]:
        return self._ov

    def starts_group(self, column: int) -> bool:
        return column in self._group_starts

    def starts_subject(self, row: int) -> bool:
        return row in self._subject_starts

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt naming
        return 0 if parent.isValid() or self._ov is None else len(self._ov.visits)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt naming
        return 0 if parent.isValid() or self._ov is None else len(self._ov.kinds)

    def cell(self, row: int, column: int) -> Cell:
        return self._ov.cell(row, column) if self._ov is not None else _EMPTY

    def rows_at(self, row: int, column: int) -> list[int]:
        """What a click on a cell selects: its recordings, the ones left out
        too; for an empty cell, everything that visit has."""
        if self._ov is None:
            return []
        cell = self._ov.cell(row, column)
        rows = [*cell.rows, *cell.left_out]
        return rows or list(self._ov.visit_rows[row])

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._ov is None:
            return None
        cell = self._ov.cell(index.row(), index.column())
        if role == _CELL_ROLE:
            return cell
        if role == Qt.ItemDataRole.DisplayRole:
            if cell.count or cell.state or cell.left_out:
                return str(cell.count)
            return ""
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._cell_tip(index.row(), index.column(), cell)
        return None

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if self._ov is None:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._ov.kinds):
            kind = self._ov.kinds[section]
            if role == Qt.ItemDataRole.DisplayRole:
                return kind.key
            if role == Qt.ItemDataRole.ToolTipRole:
                rows = self._ov.kind_rows[section]
                have = sum(1 for v in range(len(self._ov.visits)) if self._ov.cell(v, section).count)
                return (f"{kind.key}\n{_plural(len(rows), 'recording')} in "
                        f"{_plural(have, 'subject' if not any(v.session for v in self._ov.visits) else 'session')}."
                        "\nClick to select them.")
        if orientation == Qt.Orientation.Vertical and 0 <= section < len(self._ov.visits):
            visit = self._ov.visits[section]
            if role == Qt.ItemDataRole.DisplayRole:
                return visit.label
            if role == Qt.ItemDataRole.ToolTipRole:
                rows = self._ov.visit_rows[section]
                return f"{visit.label}\n{_plural(len(rows), 'recording')}.\nClick to select them."
        return None

    def flags(self, index: QModelIndex):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def _cell_tip(self, row: int, column: int, cell: Cell) -> str:
        ov = self._ov
        lines = [f"{ov.visits[row].label} · {ov.kinds[column].key}"]
        if cell.count:
            lines.append(_plural(cell.count, "recording") + " to convert.")
        else:
            lines.append("None to convert.")
        if cell.left_out:
            lines.append(f"{len(cell.left_out):,} left out.")
        if cell.usual is not None:
            noun = "session" if ov.visits[row].session else "subject"
            what = {"missing": "Most comparable {n}s have {u}.",
                    "fewer": "Fewer than most comparable {n}s, which have {u}.",
                    "extra": "More than most comparable {n}s, which have {u}.",
                    "rare": "Most comparable {n}s have none.",
                    }.get(cell.state, "As most comparable {n}s.")
            lines.append(what.format(n=noun, u=cell.usual)
                         + f" Compared with {cell.peers - 1} other{'s' if cell.peers != 2 else ''}.")
        lines.append("Click to select; double-click to open in the table.")
        return "\n".join(lines)


class _CellDelegate(QStyledItemDelegate):
    """Paints a count: plain when it is what most have, a warning pill when
    it differs, an outline when only a few have that kind."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.restyle()

    def restyle(self) -> None:
        self._regular = _font(12)
        self._bold = _font(12, bold=True)

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt naming
        return QSize(_px(44), _px(30))

    def paint(self, painter: QPainter, option, index) -> None:
        model = index.model()
        cell = index.data(_CELL_ROLE) or _EMPTY
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        pal = CUR()
        rect = QRect(option.rect)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        hairline = QPen(qcolor(pal["subtle"]))
        hairline.setWidth(1)
        painter.setPen(hairline)
        if index.column() and model.starts_group(index.column()):
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
        if index.row() and model.starts_subject(index.row()):
            painter.drawLine(rect.topLeft(), rect.topRight())

        inner = QRectF(rect).adjusted(3, 3, -3, -3)
        state = option.state
        if state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(qcolor(pal["accent_border"])))
            painter.setBrush(qcolor(pal["accent_bg"]))
            painter.drawRoundedRect(inner, 6, 6)
        elif state & QStyle.StateFlag.State_MouseOver:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(pal["surface3"]))
            painter.drawRoundedRect(inner, 6, 6)

        if text:
            flagged = cell.state in ("missing", "fewer", "extra")
            font = self._bold if (flagged or cell.state == "rare") else self._regular
            painter.setFont(font)
            metrics = QFontMetrics(font)
            if flagged or cell.state == "rare":
                width = max(metrics.horizontalAdvance(text) + _px(14), _px(26))
                height = metrics.height() + _px(4)
                pill = QRectF(inner.center().x() - width / 2, inner.center().y() - height / 2,
                              width, height)
                if flagged:
                    painter.setPen(QPen(qcolor(pal["warning_border"])))
                    painter.setBrush(qcolor(pal["warning_bg"]))
                    colour = pal["warning"]
                else:
                    painter.setPen(QPen(qcolor(pal["border"])))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    colour = pal["text"]
                painter.drawRoundedRect(pill, height / 2, height / 2)
            else:
                colour = pal["muted"] if not cell.count else pal["text"]
            painter.setPen(qcolor(colour))
            painter.drawText(inner, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class _KindHeader(QHeaderView):
    """The columns' header: the datatype over its group, then the suffix,
    then the labels, painted in three lines."""

    def __init__(self, parent) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("overview-header")
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.setMouseTracking(True)
        self._hover = -1
        self.restyle()

    def restyle(self) -> None:
        self._small = _font(10, bold=True)
        self._main = _font(12, bold=True)
        self._caption = _font(11)
        lines = (QFontMetrics(self._small).height() + QFontMetrics(self._main).height()
                 + QFontMetrics(self._caption).height())
        self.setFixedHeight(lines + _px(14))

    def width_for(self, column: int) -> int:
        model = self.model()
        kind = model.overview.kinds[column]
        widths = [QFontMetrics(self._main).horizontalAdvance(kind.suffix),
                  QFontMetrics(self._caption).horizontalAdvance(kind.caption)]
        if model.starts_group(column):
            widths.append(QFontMetrics(self._small).horizontalAdvance(kind.datatype.upper()))
        return max(_px(46), min(max(widths) + _px(16), _px(170)))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        hover = self.logicalIndexAt(event.position().toPoint())
        if hover != self._hover:
            self._hover = hover
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = -1
        self.viewport().update()
        super().leaveEvent(event)

    def paintSection(self, painter: QPainter, rect: QRect, logical: int) -> None:  # noqa: N802
        model = self.model()
        if model is None or model.overview is None:
            return
        kind = model.overview.kinds[logical]
        pal = CUR()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if logical == self._hover:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(pal["surface3"]))
            painter.drawRoundedRect(QRectF(rect).adjusted(2, 2, -2, -3), 6, 6)
        hairline = QPen(qcolor(pal["subtle"]))
        painter.setPen(hairline)
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        if logical and model.starts_group(logical):
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
        pad = _px(8)
        box = rect.adjusted(pad, _px(6), -pad // 2, -_px(6))
        y = box.top()
        for font, colour, text, elide in (
            (self._small, pal["muted"],
             kind.datatype.upper() if model.starts_group(logical) else "", Qt.TextElideMode.ElideRight),
            (self._main, pal["text"], kind.suffix, Qt.TextElideMode.ElideRight),
            (self._caption, pal["dim"], kind.caption, Qt.TextElideMode.ElideMiddle),
        ):
            metrics = QFontMetrics(font)
            if text:
                painter.setFont(font)
                painter.setPen(qcolor(colour))
                painter.drawText(QRect(box.left(), y, box.width(), metrics.height()),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 metrics.elidedText(text, elide, box.width()))
            y += metrics.height()
        painter.restore()


class _VisitHeader(QHeaderView):
    """The rows' header: the subject on its first row, the session beside."""

    def __init__(self, parent) -> None:
        super().__init__(Qt.Orientation.Vertical, parent)
        self.setObjectName("overview-header")
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.setMouseTracking(True)
        self._hover = -1
        self._subject_w = 0
        self.restyle()

    def restyle(self) -> None:
        self._main = _font(12, bold=True)
        self._session = _font(11)
        self.setDefaultSectionSize(_px(30))

    def fit(self) -> None:
        """As wide as the longest subject and session."""
        model = self.model()
        visits = model.overview.visits if model and model.overview else []
        main, session = QFontMetrics(self._main), QFontMetrics(self._session)
        self._subject_w = max((main.horizontalAdvance(v.subject) for v in visits), default=0)
        session_w = max((session.horizontalAdvance(v.session) for v in visits if v.session),
                        default=0)
        width = _px(12) + self._subject_w + (_px(10) + session_w if session_w else 0) + _px(12)
        self.setFixedWidth(max(width, _px(60)))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        hover = self.logicalIndexAt(event.position().toPoint())
        if hover != self._hover:
            self._hover = hover
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = -1
        self.viewport().update()
        super().leaveEvent(event)

    def paintSection(self, painter: QPainter, rect: QRect, logical: int) -> None:  # noqa: N802
        model = self.model()
        if model is None or model.overview is None:
            return
        visit = model.overview.visits[logical]
        pal = CUR()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if logical == self._hover:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(pal["surface3"]))
            painter.drawRoundedRect(QRectF(rect).adjusted(3, 3, -3, -3), 6, 6)
        if logical and model.starts_subject(logical):
            painter.setPen(QPen(qcolor(pal["subtle"])))
            painter.drawLine(rect.topLeft(), rect.topRight())
        left = rect.left() + _px(12)
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if model.starts_subject(logical):
            painter.setFont(self._main)
            painter.setPen(qcolor(pal["text"]))
            painter.drawText(QRect(left, rect.top(), self._subject_w + 2, rect.height()), flags,
                             visit.subject)
        if visit.session:
            painter.setFont(self._session)
            painter.setPen(qcolor(pal["dim"]))
            x = left + self._subject_w + _px(10)
            painter.drawText(QRect(x, rect.top(), rect.right() - x, rect.height()), flags,
                             visit.session)
        painter.restore()


class OverviewGrid(QTableView):
    """Subjects (and sessions) by kind of recording, one painted count each."""

    #: Rows to select (a cell, a subject, a kind).
    rows_chosen = pyqtSignal(list)
    #: Rows to open in the table (a double-clicked cell).
    rows_shown = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("overview-grid")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        self.setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # The model is the view's child: a model held only by a local is
        # destroyed under the view (CLAUDE.md guard 8c).
        self.grid_model = _GridModel(self)
        self.setHorizontalHeader(_KindHeader(self))
        self.setVerticalHeader(_VisitHeader(self))
        self.setModel(self.grid_model)
        self._delegate = _CellDelegate(self)
        self.setItemDelegate(self._delegate)
        self.clicked.connect(self._on_clicked)
        self.activated.connect(self._on_activated)
        self.horizontalHeader().sectionClicked.connect(self._on_kind_clicked)
        self.verticalHeader().sectionClicked.connect(self._on_visit_clicked)
        self._row_cell: dict[int, tuple[int, int]] = {}

    def set_overview(self, ov: Optional[Overview]) -> None:
        self.grid_model.set_overview(ov)
        self._row_cell = {}
        if ov is not None:
            for (visit, kind), cell in ov._cells.items():
                for row in (*cell.rows, *cell.left_out):
                    self._row_cell[row] = (visit, kind)
        self.fit()

    def restyle(self) -> None:
        self._delegate.restyle()
        self.horizontalHeader().restyle()
        self.verticalHeader().restyle()
        self.fit()
        self.viewport().update()

    def fit(self) -> None:
        header = self.horizontalHeader()
        for column in range(self.grid_model.columnCount()):
            header.resizeSection(column, header.width_for(column))
        self.verticalHeader().fit()
        rows = min(max(self.grid_model.rowCount(), 1), 3)
        self.setMinimumHeight(header.height() + rows * self.verticalHeader().defaultSectionSize() + 4)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """No taller than its minimum: it grows into the page through its
        stretch. A table view's default 192 px made the page ask for more
        than the panel had, and the page scrolled beside the grid."""
        return QSize(super().sizeHint().width(), self.minimumHeight())

    def reveal_row(self, row: Optional[int]) -> None:
        """Mark the cell holding inventory ``row`` (the table's selection),
        without announcing it back."""
        at = self._row_cell.get(row) if row is not None else None
        blocker = self.selectionModel().blockSignals(True)
        try:
            if at is None:
                self.selectionModel().clearSelection()
            else:
                index = self.grid_model.index(*at)
                self.selectionModel().setCurrentIndex(
                    index, self.selectionModel().SelectionFlag.ClearAndSelect)
                self.scrollTo(index)
        finally:
            self.selectionModel().blockSignals(blocker)
        self.viewport().update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Return or Enter opens the current count in the table, on every OS
        (Qt's item views read Return as "edit" on macOS and "activate"
        elsewhere). An arrow key moves to a count and selects it, as a click
        does."""
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.currentIndex().isValid():
                self._on_activated(self.currentIndex())
            event.accept()
            return
        before = self.currentIndex()
        super().keyPressEvent(event)
        after = self.currentIndex()
        if after.isValid() and after != before:
            self._on_clicked(after)

    def _on_clicked(self, index: QModelIndex) -> None:
        rows = self.grid_model.rows_at(index.row(), index.column())
        if rows:
            self.rows_chosen.emit(rows)

    def _on_activated(self, index: QModelIndex) -> None:
        rows = self.grid_model.rows_at(index.row(), index.column())
        if rows:
            self.rows_shown.emit(rows)

    def _on_kind_clicked(self, column: int) -> None:
        ov = self.grid_model.overview
        if ov is not None and 0 <= column < len(ov.kinds):
            self.rows_chosen.emit(list(ov.kind_rows[column]))

    def _on_visit_clicked(self, row: int) -> None:
        ov = self.grid_model.overview
        if ov is not None and 0 <= row < len(ov.visits):
            self.rows_chosen.emit(list(ov.visit_rows[row]))


class _LegendMark(QWidget):
    """One of the grid's marks (a warning pill or an outline) and what it
    means, painted exactly as a cell paints it."""

    def __init__(self, kind: str, sample: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self._kind, self._sample, self._text = kind, sample, text

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        bold, regular = _font(11, bold=True), _font(11)
        pill = max(QFontMetrics(bold).horizontalAdvance(self._sample) + _px(12), _px(22))
        width = pill + _px(6) + QFontMetrics(regular).horizontalAdvance(self._text) + 2
        return QSize(width, QFontMetrics(bold).height() + _px(6))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        pal = CUR()
        bold, regular = _font(11, bold=True), _font(11)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        height = QFontMetrics(bold).height() + _px(2)
        width = max(QFontMetrics(bold).horizontalAdvance(self._sample) + _px(12), _px(22))
        pill = QRectF(0.5, (self.height() - height) / 2, width, height)
        if self._kind == "differs":
            painter.setPen(QPen(qcolor(pal["warning_border"])))
            painter.setBrush(qcolor(pal["warning_bg"]))
            colour = pal["warning"]
        else:
            painter.setPen(QPen(qcolor(pal["border"])))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            colour = pal["text"]
        painter.drawRoundedRect(pill, height / 2, height / 2)
        painter.setFont(bold)
        painter.setPen(qcolor(colour))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, self._sample)
        painter.setFont(regular)
        painter.setPen(qcolor(pal["dim"]))
        text_rect = QRect(int(pill.right()) + _px(6), 0, self.width(), self.height())
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         self._text)
        painter.end()


# ---------------------------------------------------------------------------
# The pane
# ---------------------------------------------------------------------------


def _rule(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setObjectName("overview-rule")
    line.setFixedHeight(1)
    return line


class OverviewPane(QWidget):
    """Headline counts, findings worth a look, and the grid."""

    #: Rows to select in the table; the Overview stays.
    rows_chosen = pyqtSignal(list)
    #: Rows to select and open in the table.
    rows_shown = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._model = None
        self._overview: Optional[Overview] = None
        self._binding = 0
        #: The number of the count in flight, and the table state it counts.
        #: A small counter, because a job's number travels through a Qt
        #: signal as a 32-bit int: ``binding * 10**9 + revision`` passed it at
        #: the third table, came back negative, never matched, and the pane
        #: recounted every 11 ms and stayed empty (2026-10-13).
        self._ticket = 0
        self._ticket_state: tuple = ()
        self._dirty = True
        self._jobs = None
        self._expanded = False
        self._finding_rows: list[QWidget] = []
        self._level = str(QSettings().value(_LEVEL_KEY, "auto") or "auto")
        if self._level not in ("auto", *LEVELS):
            self._level = "auto"

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._start)
        self._pending = QTimer(self)
        self._pending.setSingleShot(True)
        self._pending.setInterval(_PENDING_MS)
        self._pending.timeout.connect(self._show_pending)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack)

        self._empty = QLabel("Scan a raw-data folder to see what each subject will have.", self)
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._stack.addWidget(self._empty)

        waiting = QWidget(self)
        wl = QVBoxLayout(waiting)
        wl.addStretch(1)
        self._spinner = BusySpinner(waiting)
        wl.addWidget(self._spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        wl.addStretch(1)
        self._stack.addWidget(waiting)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("overview-scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._stack.addWidget(self._scroll)
        self._build_page()

    # -- building -------------------------------------------------------------

    def _build_page(self) -> None:
        body = QWidget(self._scroll)
        body.setObjectName("overview-body")
        v = QVBoxLayout(body)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(12)

        self._tiles_bar = FlowBar(body, h_spacing=32, v_spacing=10)
        self._tiles = {}
        for key, caption in (("subjects", "subjects"), ("sessions", "sessions"),
                             ("recordings", "to convert"), ("datatypes", "datatypes")):
            tile = stat(caption, "0")
            self._tiles_bar.addWidget(tile)
            self._tiles[key] = tile
        self._tiles["recordings"].setToolTip("Recordings set to convert")
        v.addWidget(self._tiles_bar)
        self._facts = WrapLabel("", body)
        self._facts.setObjectName("dlg-hint")
        self._facts.setTextFormat(Qt.TextFormat.PlainText)
        v.addWidget(self._facts)
        v.addWidget(_rule(body))

        # The findings fold from their title, to give the grid the room.
        self._findings_title = QToolButton(body)
        self._findings_title.setText("Worth a look")
        self._findings_title.setObjectName("overview-fold")
        self._findings_title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._findings_title.setAutoRaise(True)
        self._findings_title.setCursor(Qt.CursorShape.PointingHandCursor)
        self._findings_title.setToolTip("Fold or open the findings")
        self._folded = str(QSettings().value(_FOLD_KEY, False)).lower() in ("1", "true")
        self._findings_title.clicked.connect(self._toggle_fold)
        v.addWidget(self._findings_title, 0, Qt.AlignmentFlag.AlignLeft)
        self._findings_box = QWidget(body)
        self._findings_lay = QVBoxLayout(self._findings_box)
        self._findings_lay.setContentsMargins(0, 0, 0, 0)
        self._findings_lay.setSpacing(8)
        v.addWidget(self._findings_box)
        more_row = QHBoxLayout()
        more_row.setContentsMargins(_px(24), 0, 0, 0)
        self._more = QPushButton("", body)
        self._more.setObjectName("inline-link")
        self._more.setCursor(Qt.CursorShape.PointingHandCursor)
        self._more.clicked.connect(self._toggle_more)
        more_row.addWidget(self._more)
        more_row.addStretch(1)
        v.addLayout(more_row)
        v.addWidget(_rule(body))

        head = FlowBar(body, h_spacing=10, v_spacing=6)
        grid_title = QLabel("Recordings per subject", head)
        grid_title.setObjectName("section-title")
        head.addWidget(grid_title)
        self.level_box = QComboBox(head)
        for key in ("auto", *LEVELS):
            self.level_box.addItem(_level_text(key), key)
        self.level_box.setToolTip(
            "What makes a column: every label of a name (task, acquisition, direction...), "
            "the task only, or the datatype and suffix alone. Automatic takes the most "
            "detailed that stays readable.")
        self.level_box.setCurrentIndex(max(0, self.level_box.findData(self._level)))
        self.level_box.currentIndexChanged.connect(self._on_level_changed)
        head.addWidget(self.level_box)
        head.addSpacing(_px(6))
        self._legend = [
            _LegendMark("differs", "2", "differs from most", head),
            _LegendMark("rare", "1", "few have it", head),
        ]
        self._legend[0].setToolTip(
            "A count that differs from what most comparable subjects (or sessions) have: "
            "a recording missing, fewer or more of them.")
        self._legend[1].setToolTip("A kind of recording most comparable subjects do not have.")
        for mark in self._legend:
            head.addWidget(mark)
        grid_title.setToolTip(
            "How many recordings each subject (and session) will have of each kind of "
            "recording. Subjects are compared within a modality, and within a session label "
            "when at least three share it. Click a count to select its recordings; "
            "double-click to open them in the table.")
        v.addWidget(head)

        # The grid fills what is left and is the one thing that scrolls, so
        # its column and subject headers stay in view. The page scrolls only
        # when the panel is too short for the grid's first rows.
        self.grid = OverviewGrid(body)
        self.grid.rows_chosen.connect(self.rows_chosen)
        self.grid.rows_shown.connect(self.rows_shown)
        v.addWidget(self.grid, 1)
        self._scroll.setWidget(body)

    # -- binding ----------------------------------------------------------------

    def bind_model(self, model) -> None:
        """Follow ``model`` (an ``InventoryTableModel``), or nothing."""
        if model is self._model:
            return
        if self._model is not None:
            for signal in self._model_signals(self._model):
                try:
                    signal.disconnect(self._on_model_changed)
                except (TypeError, RuntimeError):
                    pass
        self._model = model
        self._binding += 1
        self._overview = None
        if self._jobs is not None:
            self._jobs.cancel("overview")
        if model is not None:
            for signal in self._model_signals(model):
                signal.connect(self._on_model_changed)
        self._dirty = True
        self._show_empty_or_wait()
        if self.isVisible():
            self._settle.start(0)

    @staticmethod
    def _model_signals(model):
        return (model.dataChanged, model.modelReset, model.rowsInserted,
                model.rowsRemoved, model.layoutChanged)

    def _on_model_changed(self, *_args) -> None:
        self._dirty = True
        if self.isVisible():
            self._settle.start()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        if self._dirty:
            self._settle.start(0)

    def overview(self) -> Optional[Overview]:
        """What is on screen (``None`` before the first count)."""
        return self._overview

    def is_counting(self) -> bool:
        return self._jobs is not None and self._jobs.running("overview")

    # -- counting -----------------------------------------------------------------

    def _state(self) -> tuple:
        """What a count depends on: which table, how far edited, which level."""
        revision = self._model.revision if self._model is not None else 0
        return (self._binding, revision, self._level)

    def _start(self) -> None:
        model = self._model
        if model is None or model.rowCount() == 0:
            self._dirty = False
            self._overview = None
            self._show_empty_or_wait()
            return
        if self._jobs is None:
            from .viz.bridge import JobRunner

            self._jobs = JobRunner(self)
            self._jobs.done.connect(self._on_done)
            self._jobs.failed.connect(self._on_failed)
        self._dirty = False
        df = model.dataframe()
        wanted = [c for c in ("participant_id", "session", "datatype", "bids_name", "dataset",
                              "modality", "study_date", "recording_time",
                              "classification_evidence") if c in df.columns]
        # A snapshot: the worker never reads the live table, which edits change.
        snapshot = df[wanted].copy()
        included = [model._read_include(r) for r in range(model.rowCount())]
        self._ticket = self._ticket % 1_000_000 + 1
        self._ticket_state = self._state()
        self._jobs.start("overview", self._ticket, build_overview, snapshot, included,
                         level=self._level)
        if self._overview is None:
            self._pending.start()

    def _on_done(self, tag: str, ticket: int, result) -> None:
        if tag != "overview" or ticket != self._ticket:
            return              # an older count: the newer one is on its way
        if self._state() != self._ticket_state:
            # The table changed while it was counted: count it again (an edit
            # also marks the pane dirty, but that may have come before).
            self._settle.start()
            return
        self._pending.stop()
        self._spinner.set_busy(False)
        self._show(result)

    def _on_failed(self, tag: str, _generation: int, message: str) -> None:
        if tag != "overview":
            return
        self._pending.stop()
        self._spinner.set_busy(False)
        self._empty.setText(f"The overview could not be counted: {message}")
        self._stack.setCurrentIndex(0)

    def _show_pending(self) -> None:
        if self._overview is None and self.is_counting():
            self._spinner.set_busy(True, message="Counting what each subject will have...")
            self._stack.setCurrentIndex(1)

    def _show_empty_or_wait(self) -> None:
        self._empty.setText("Scan a raw-data folder to see what each subject will have."
                            if self._model is None or self._model.rowCount() == 0
                            else "")
        self._stack.setCurrentIndex(0)

    def _on_level_changed(self, _index: int) -> None:
        self._level = str(self.level_box.currentData() or "auto")
        QSettings().setValue(_LEVEL_KEY, self._level)
        self._dirty = True
        self._settle.start(0)

    # -- showing ------------------------------------------------------------------

    def _show(self, ov: Overview) -> None:
        self._overview = ov
        values = {"subjects": ov.subjects, "sessions": ov.sessions,
                  "recordings": ov.recordings, "datatypes": ov.datatypes}
        for key, value in values.items():
            self._tiles[key].number.setText(f"{value:,}")
        facts = [ov.modality_line()] if ov.by_modality else []
        if ov.left_out:
            facts.append(f"{ov.left_out:,} left out")
        self._facts.setText(", ".join(facts) if len(facts) == 1 else
                            f"{facts[0]}; {', '.join(facts[1:])}" if facts else "")
        self._facts.setVisible(bool(facts))
        self.level_box.setItemText(0, _level_text("auto", ov.level if self._level == "auto" else ""))
        self._show_findings()
        self.grid.set_overview(ov)
        self._stack.setCurrentIndex(2)

    def _clear_findings(self) -> None:
        for row in self._finding_rows:
            self._findings_lay.removeWidget(row)
            # Hidden before it is detached: a show Qt queued when it was
            # added would otherwise open it as a window (GUI_LOOK_AND_FEEL 4.26).
            row.hide()
            row.setParent(None)
            row.deleteLater()
        self._finding_rows = []

    def _show_findings(self) -> None:
        self._clear_findings()
        ov = self._overview
        if ov is None:
            return
        findings = ov.findings
        shown = findings if self._expanded else findings[:FINDINGS_SHOWN]
        if not findings:
            text = ("Nothing stands out: the subjects compared have the same recordings."
                    if ov.compared else
                    "Too few subjects to compare. The grid shows what each one will have.")
            self._add_row("ok", text, ())
        for finding in shown:
            self._add_row(_SEVERITY_ICON.get(finding.severity, "info"), finding.text,
                          finding.rows, finding)
        extra = len(findings) - FINDINGS_SHOWN
        self._more.setVisible(extra > 0 and not self._folded)
        self._more.setText("Show fewer" if self._expanded else f"Show all {len(findings)}")
        self._findings_title.setText(
            f"Worth a look ({len(findings)})" if findings else "Worth a look")
        self._findings_title.setIcon(icons.icon("panel_expand" if self._folded else "panel_collapse"))
        self._findings_title.setIconSize(QSize(_px(12), _px(12)))
        self._findings_box.setVisible(not self._folded)

    def _add_row(self, icon_name: str, text: str, rows, finding: Optional[Finding] = None) -> None:
        row = QWidget(self._findings_box)
        row.setObjectName("overview-finding-row")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        size = _px(16)
        mark = QLabel(row)
        mark.setPixmap(icons.icon(icon_name).pixmap(size, size))
        mark.setFixedSize(size, max(size, QFontMetrics(_font(12)).height()))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        label = WrapLabel(text, row)
        label.setObjectName("overview-finding")
        label.setTextFormat(Qt.TextFormat.PlainText)
        h.addWidget(label, 1)
        if rows:
            show = QPushButton("Show", row)
            show.setObjectName("inline-link")
            show.setCursor(Qt.CursorShape.PointingHandCursor)
            show.setToolTip(f"Select {_plural(len(rows), 'recording')} in the table")
            show.clicked.connect(lambda _c=False, r=list(rows): self.rows_shown.emit(r))
            h.addWidget(show, 0, Qt.AlignmentFlag.AlignTop)
        if finding is not None:
            row.setProperty("finding", finding.key)
        self._findings_lay.addWidget(row)
        self._finding_rows.append(row)

    def _toggle_fold(self) -> None:
        self._folded = not self._folded
        QSettings().setValue(_FOLD_KEY, self._folded)
        self._show_findings()

    def findings_folded(self) -> bool:
        return self._folded

    def _toggle_more(self) -> None:
        self._expanded = not self._expanded
        self._show_findings()

    def finding_rows(self) -> list[QWidget]:
        """The rows on screen, in order (tests)."""
        return list(self._finding_rows)

    # -- following the table and the theme -----------------------------------------

    def reveal_row(self, row: Optional[int]) -> None:
        """The table's current row, marked in the grid."""
        self.grid.reveal_row(row)

    def repaint_for_palette(self, _pal: dict) -> None:
        """A theme or font-size change: painted fonts and colours, and the
        status icons. (The owner re-polishes :attr:`level_box`, a combo box,
        with the Converter's other combo boxes.)"""
        self.grid.restyle()
        self._show_findings()
        for mark in self._legend:
            mark.updateGeometry()
            mark.update()


__all__ = ["FINDINGS_SHOWN", "OverviewGrid", "OverviewPane"]
