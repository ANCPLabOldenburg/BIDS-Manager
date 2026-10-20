"""The pieces of the Anonymise metadata dialog: what, where, and what changes.

- :class:`CategoryCard`: one category (Names and IDs, Dates and times, ...),
  switched on or off, with its options and how many changes it makes now.
- :class:`ScopeTree`: the dataset files, the subjects and their sessions,
  ticked; read back as a :class:`bidsmgr.anonymise.scope.Scope`.
- :class:`ChangeList`: every change, grouped by kind or by file, one painted
  view over one model (GUI_LOOK_AND_FEEL 4.33: a dataset with twenty thousand
  changes is one view, not twenty thousand items), searchable, with the
  selected line in full underneath. A file can be left out of the run.

Qt only; the data are the plan's (``bidsmgr.anonymise``).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from PyQt6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .primitives import Chip, Segmented
from .search_field import SearchField, find_shortcut

# ---------------------------------------------------------------------------
# What: one category
# ---------------------------------------------------------------------------


class CategoryCard(QFrame):
    """A category, on or off, its options, and its count of changes."""

    #: ``(category key, on, option or "")`` whenever the user changes it.
    changed = pyqtSignal(str, bool, str)

    def __init__(self, category, parent=None) -> None:
        super().__init__(parent)
        from ..dialog_chrome import WrapLabel

        self.key = category.key
        self.setObjectName("anon-category")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 9, 12, 10)
        v.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.box = QCheckBox(category.title, self)
        self.box.setObjectName("anon-category-title")
        self.box.setToolTip(category.summary)
        self.box.toggled.connect(self._on_toggled)
        head.addWidget(self.box)
        head.addStretch(1)
        self.count = Chip("", "default", self)
        self.count.setToolTip("Changes this category makes in the current scope")
        head.addWidget(self.count)
        v.addLayout(head)
        self.summary = WrapLabel(category.summary, self)
        self.summary.setObjectName("dlg-hint")
        self.summary.setContentsMargins(22, 0, 0, 0)
        v.addWidget(self.summary)
        self.options: Optional[Segmented] = None
        if category.options:
            row = QHBoxLayout()
            row.setContentsMargins(22, 2, 0, 0)
            self.options = Segmented([(value, label, tip)
                                      for value, label, tip in category.options], parent=self)
            self.options.changed.connect(self._on_option)
            row.addWidget(self.options)
            row.addStretch(1)
            v.addLayout(row)

    def set_state(self, on: bool, option: Optional[str]) -> None:
        """Show ``on`` and ``option`` without announcing them."""
        self.box.blockSignals(True)
        self.box.setChecked(on)
        self.box.blockSignals(False)
        if self.options is not None and option:
            self.options.set_current(option)
        self._paint_state(on)

    def set_count(self, n: Optional[int]) -> None:
        """The changes it makes now (None: being worked out)."""
        on = self.box.isChecked()
        if not on:
            self.count.set_kind("default")
            self.count.setText("off")
        elif n is None:
            self.count.set_kind("default")
            self.count.setText("…")
        else:
            self.count.set_kind("accent" if n else "default")
            self.count.setText(f"{n:,}")

    def _paint_state(self, on: bool) -> None:
        self.setProperty("on", "true" if on else "false")
        for widget in (self, self.summary):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        if self.options is not None:
            self.options.setEnabled(on)

    def _on_toggled(self, on: bool) -> None:
        self._paint_state(on)
        self.changed.emit(self.key, on, self._option())

    def _on_option(self, value: str) -> None:
        self.changed.emit(self.key, self.box.isChecked(), value)

    def _option(self) -> str:
        if self.options is None:
            return ""
        return next((k for k, b in self.options.buttons.items() if b.isChecked()), "")


def purpose_heading(text: str) -> QLabel:
    """The heading a group of categories sits under (small capitals)."""
    label = QLabel(text.upper())
    label.setObjectName("anon-purpose")
    return label


# ---------------------------------------------------------------------------
# Where: the dataset files, subjects and sessions
# ---------------------------------------------------------------------------

_KEY = Qt.ItemDataRole.UserRole


class ScopeTree(QWidget):
    """Tick what to anonymise; ``changed`` after every tick."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from .tree_click import toggle_on_click

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.search = SearchField("Find a subject")
        bar.addWidget(self.search, 1)
        self._all = QPushButton("All")
        self._all.setObjectName("tb-btn")
        self._all.clicked.connect(lambda: self._set_all(True))
        bar.addWidget(self._all)
        self._none = QPushButton("None")
        self._none.setObjectName("tb-btn")
        self._none.clicked.connect(lambda: self._set_all(False))
        bar.addWidget(self._none)
        v.addLayout(bar)
        find_shortcut(self, self.search)
        self.tree = QTreeWidget(self)
        self.tree.setObjectName("check-tree")
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(14)
        self.tree.itemChanged.connect(self._on_item_changed)
        toggle_on_click(self.tree)
        v.addWidget(self.tree, 1)
        self.search.textChanged.connect(self._filter)
        self._settling = False

    def set_units(self, top_files: int, units) -> None:
        """The dataset's subjects and sessions (``anonymise.dataset_units``),
        everything ticked."""
        self._settling = True
        self.tree.clear()
        top = QTreeWidgetItem(["Dataset files", f"{top_files:,}"])
        top.setData(0, _KEY, ("top", ""))
        top.setToolTip(0, "participants.tsv and its JSON, and the other files at the top "
                          "(inherited sidecars). participants.tsv is changed only in the "
                          "rows of the subjects ticked below.")
        top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        top.setCheckState(0, Qt.CheckState.Checked)
        self.tree.addTopLevelItem(top)
        for unit in units:
            item = QTreeWidgetItem([unit.subject, f"{unit.files:,}"])
            item.setData(0, _KEY, ("subject", unit.subject))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                          | Qt.ItemFlag.ItemIsAutoTristate)
            item.setCheckState(0, Qt.CheckState.Checked)
            for session, files in unit.sessions:
                child = QTreeWidgetItem(item, [session, f"{files:,}"])
                child.setData(0, _KEY, ("session", f"{unit.subject}/{session}"))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
            self.tree.addTopLevelItem(item)
        self.tree.header().setStretchLastSection(False)
        from PyQt6.QtWidgets import QHeaderView

        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setForeground(1, QColor(_pal()["muted"]))
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for j in range(item.childCount()):
                item.child(j).setForeground(1, QColor(_pal()["muted"]))
                item.child(j).setTextAlignment(
                    1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._settling = False

    def apply_scope(self, scope) -> None:
        """Tick what ``scope`` covers, and nothing else."""
        self._settling = True
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            kind, key = item.data(0, _KEY)
            if kind == "top":
                item.setCheckState(0, Qt.CheckState.Checked if scope.top
                                   else Qt.CheckState.Unchecked)
                continue
            if not item.childCount():
                item.setCheckState(0, Qt.CheckState.Checked if scope.has_subject(key)
                                   else Qt.CheckState.Unchecked)
                continue
            for j in range(item.childCount()):
                child = item.child(j)
                _sub, ses = child.data(0, _KEY)[1].split("/", 1)
                child.setCheckState(0, Qt.CheckState.Checked if scope.has_session(key, ses)
                                    else Qt.CheckState.Unchecked)
            item.setExpanded(item.checkState(0) == Qt.CheckState.PartiallyChecked)
        self._settling = False

    def scope(self):
        """What is ticked, as a ``Scope`` (None for "all")."""
        from ...anonymise.scope import Scope

        top = False
        subjects: list[str] = []
        sessions: list[str] = []
        every = True
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            kind, key = item.data(0, _KEY)
            state = item.checkState(0)
            if kind == "top":
                top = state == Qt.CheckState.Checked
                continue
            if state == Qt.CheckState.Unchecked:
                every = False
                continue
            subjects.append(key)
            if state == Qt.CheckState.PartiallyChecked:
                sessions += [item.child(j).data(0, _KEY)[1] for j in range(item.childCount())
                             if item.child(j).checkState(0) == Qt.CheckState.Checked]
        return Scope(top=top, subjects=None if every else frozenset(subjects),
                     sessions=frozenset(sessions) if sessions else None)

    def is_empty(self) -> bool:
        scope = self.scope()
        return not scope.top and scope.subjects is not None and not scope.subjects

    def _set_all(self, on: bool) -> None:
        self._settling = True
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if not item.isHidden():
                item.setCheckState(0, state)
                for j in range(item.childCount()):
                    item.child(j).setCheckState(0, state)
        self._settling = False
        self.changed.emit()

    def _on_item_changed(self, _item, column: int) -> None:
        if column == 0 and not self._settling:
            self.changed.emit()

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            kind, key = item.data(0, _KEY)
            item.setHidden(bool(needle) and kind == "subject" and needle not in key.lower())

    def repaint_for_palette(self) -> None:
        self.search.refresh_icon()
        muted = QColor(_pal()["muted"])
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setForeground(1, muted)
            for j in range(item.childCount()):
                item.child(j).setForeground(1, muted)


def _pal() -> dict:
    from ..theme_manager import CUR

    return CUR()


# ---------------------------------------------------------------------------
# What changes: one painted list
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One line under a group."""

    #: change | review | cannot
    kind: str
    what: str
    before: str = ""
    #: What it becomes; "" when it is removed.
    after: str = ""
    #: The file (grouped by kind) or the reason (grouped by file).
    where: str = ""
    #: The file it belongs to (relative), for leaving it out.
    file: str = ""
    detail: str = ""


@dataclass
class Group:
    """A kind of change, a file, or a value to read."""

    title: str
    key: Any
    hint: str = ""
    rows: list[Row] = dc_field(default_factory=list)
    #: A file that can be left out of the run.
    leavable: bool = False


ROW_ROLE = Qt.ItemDataRole.UserRole + 11
GROUP_ROLE = Qt.ItemDataRole.UserRole + 12
_TOP = 0xFFFFFFFF


class ChangeModel(QAbstractItemModel):
    """Groups and their rows, filtered by a search."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._groups: list[Group] = []
        self._shown: list[tuple[Group, list[Row]]] = []
        self._query = ""
        #: Files left out of the run (relative paths).
        self.left_out: set[str] = set()

    def set_groups(self, groups: list[Group]) -> None:
        self.beginResetModel()
        self._groups = list(groups)
        self._shown = self._filter()
        self.endResetModel()

    def set_query(self, text: str) -> None:
        query = text.strip().lower()
        if query == self._query:
            return
        self.beginResetModel()
        self._query = query
        self._shown = self._filter()
        self.endResetModel()

    def _filter(self) -> list[tuple[Group, list[Row]]]:
        query = self._query
        if not query:
            return [(g, g.rows) for g in self._groups]
        out = []
        for group in self._groups:
            if query in group.title.lower():
                out.append((group, group.rows))
                continue
            kept = [r for r in group.rows if query in r.what.lower()
                    or query in r.before.lower() or query in r.after.lower()
                    or query in r.where.lower()]
            if kept:
                out.append((group, kept))
        return out

    def counts(self) -> tuple[int, int]:
        return len(self._shown), sum(len(rows) for _g, rows in self._shown)

    def total(self) -> tuple[int, int]:
        return len(self._groups), sum(len(g.rows) for g in self._groups)

    def refresh(self) -> None:
        """Repaint every row (a file left out or put back)."""
        if self._shown:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._shown) - 1, 0))
        self.layoutChanged.emit()

    # -- the tree -------------------------------------------------------------

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if column != 0 or row < 0:
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, 0, _TOP) if row < len(self._shown) else QModelIndex()
        if parent.internalId() != _TOP:
            return QModelIndex()
        rows = self._shown[parent.row()][1]
        return self.createIndex(row, 0, parent.row()) if row < len(rows) else QModelIndex()

    def parent(self, child: QModelIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        if not child.isValid() or child.internalId() == _TOP:
            return QModelIndex()
        return self.createIndex(int(child.internalId()), 0, _TOP)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            return len(self._shown)
        if parent.internalId() == _TOP:
            return len(self._shown[parent.row()][1])
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 1

    def group_at(self, index: QModelIndex) -> Optional[Group]:
        if not index.isValid():
            return None
        row = index.row() if index.internalId() == _TOP else int(index.internalId())
        return self._shown[row][0] if 0 <= row < len(self._shown) else None

    def row_at(self, index: QModelIndex) -> Optional[Row]:
        if not index.isValid() or index.internalId() == _TOP:
            return None
        rows = self._shown[int(index.internalId())][1]
        return rows[index.row()] if index.row() < len(rows) else None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if index.internalId() == _TOP:
            group, rows = self._shown[index.row()]
            if role == Qt.ItemDataRole.DisplayRole:
                return group.title
            if role == GROUP_ROLE:
                return group, len(rows)
            if role == Qt.ItemDataRole.ToolTipRole:
                return group.title + (f"\n{group.hint}" if group.hint else "")
            return None
        row = self.row_at(index)
        if row is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return row.what
        if role == ROW_ROLE:
            return row
        if role == Qt.ItemDataRole.ToolTipRole:
            return row.detail or row.what
        return None


class ChangeDelegate(QStyledItemDelegate):
    """One painted line: a group, or a change (what | now → becomes | where)."""

    def __init__(self, model: ChangeModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        from ..viz import fonts

        return QSize(option.rect.width(), QFontMetrics(fonts.font(12)).height() + fonts.px(10))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        from ..theme_manager import rgba
        from ..viz import fonts

        pal = _pal()
        painter.save()
        try:
            rect = option.rect
            if option.state & QStyle.StateFlag.State_Selected:
                painter.fillRect(rect, rgba(pal["accent"], 0.18))
            elif option.state & QStyle.StateFlag.State_MouseOver:
                painter.fillRect(rect, rgba(pal["text"], 0.05))
            inner = rect.adjusted(fonts.px(6), 0, -fonts.px(6), 0)
            group_data = index.data(GROUP_ROLE)
            if group_data is not None:
                self._paint_group(painter, inner, group_data, pal, fonts)
            else:
                self._paint_row(painter, inner, index.data(ROW_ROLE), pal, fonts)
        finally:
            painter.restore()

    def _pill(self, painter, right: int, rect: QRect, text: str, pal, fonts,
              bg: str = "surface3", fg: str = "dim") -> int:
        small = fonts.font(11)
        painter.setFont(small)
        metrics = QFontMetrics(small)
        width = metrics.horizontalAdvance(text) + fonts.px(12)
        pill = QRect(right - width, rect.center().y() - metrics.height() // 2 - 1,
                     width, metrics.height() + 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal[bg]))
        painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        painter.setPen(QColor(pal[fg]))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        return pill.left() - fonts.px(8)

    def _paint_group(self, painter, rect: QRect, data, pal, fonts) -> None:
        group, count = data
        left_out = group.leavable and group.key in self._model.left_out
        right = self._pill(painter, rect.right(), rect, f"{count:,}", pal, fonts)
        if left_out:
            right = self._pill(painter, right, rect, "left out", pal, fonts,
                               "warning_bg", "warning")
        if group.hint:
            small = fonts.font(11)
            painter.setFont(small)
            metrics = QFontMetrics(small)
            width = min(metrics.horizontalAdvance(group.hint) + fonts.px(2),
                        max(0, (right - rect.left()) // 2))
            painter.setPen(QColor(pal["muted"]))
            painter.drawText(QRect(right - width, rect.top(), width, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             metrics.elidedText(group.hint, Qt.TextElideMode.ElideRight, width))
            right -= width + fonts.px(10)
        title_font = fonts.font(12, bold=True)
        if left_out:
            title_font.setStrikeOut(True)
        painter.setFont(title_font)
        painter.setPen(QColor(pal["muted" if left_out else "text"]))
        box = QRect(rect.left(), rect.top(), max(0, right - rect.left()), rect.height())
        painter.drawText(box, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         QFontMetrics(title_font).elidedText(
                             group.title, Qt.TextElideMode.ElideMiddle, box.width()))

    def _paint_row(self, painter, rect: QRect, row: Row, pal, fonts) -> None:
        if row is None:
            return
        left_out = bool(row.file) and row.file in self._model.left_out
        ink = (lambda token: QColor(pal["muted"])) if left_out else (
            lambda token: QColor(pal[token]))
        mono = fonts.font(11, mono=True)
        body = fonts.font(12)
        small = fonts.font(11)
        x = rect.left()
        width = rect.width()
        # Three columns that line up: what, now → becomes, where.
        what_w = min(fonts.px(210), width * 24 // 100)
        where_w = min(fonts.px(230), width * 24 // 100)
        mid_w = max(0, width - what_w - where_w - fonts.px(20))
        if row.kind == "review":
            # The value is what is read: it spans both first columns.
            span = what_w + fonts.px(10) + mid_w
            painter.setFont(mono)
            painter.setPen(ink("text"))
            painter.drawText(QRect(x, rect.top(), span, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             QFontMetrics(mono).elidedText(row.what, Qt.TextElideMode.ElideRight,
                                                           span))
            painter.setFont(small)
            painter.setPen(ink("muted"))
            painter.drawText(QRect(rect.right() - where_w, rect.top(), where_w, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             QFontMetrics(small).elidedText(row.where,
                                                            Qt.TextElideMode.ElideMiddle,
                                                            where_w))
            return
        painter.setFont(mono)
        painter.setPen(ink("dim"))
        painter.drawText(QRect(x, rect.top(), what_w, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         QFontMetrics(mono).elidedText(row.what, Qt.TextElideMode.ElideMiddle,
                                                       what_w))
        x += what_w + fonts.px(10)
        if row.kind == "change":
            self._paint_change(painter, QRect(x, rect.top(), mid_w, rect.height()), row,
                               ink, body, fonts)
        else:
            # A value to read: where it is (quiet). A file not rewritten: why.
            painter.setFont(body)
            painter.setPen(ink("dim" if row.kind == "review" else "text"))
            painter.drawText(QRect(x, rect.top(), mid_w, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             QFontMetrics(body).elidedText(row.before, Qt.TextElideMode.ElideRight,
                                                           mid_w))
        painter.setFont(small)
        painter.setPen(ink("muted"))
        painter.drawText(QRect(rect.right() - where_w, rect.top(), where_w, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                         QFontMetrics(small).elidedText(row.where, Qt.TextElideMode.ElideMiddle,
                                                        where_w))

    def _paint_change(self, painter, rect: QRect, row: Row, ink, body: QFont, fonts) -> None:
        metrics = QFontMetrics(body)
        arrow = "  →  "
        arrow_w = metrics.horizontalAdvance(arrow)
        # What it becomes is what is read: it gets the larger share.
        half = max(0, (rect.width() - arrow_w) * 42 // 100)
        painter.setFont(body)
        painter.setPen(ink("muted"))
        before = metrics.elidedText(row.before, Qt.TextElideMode.ElideRight, half)
        painter.drawText(QRect(rect.left(), rect.top(), half, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, before)
        x = rect.left() + min(half, metrics.horizontalAdvance(before)) + 1
        painter.setPen(ink("dim"))
        painter.drawText(QRect(x, rect.top(), arrow_w, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, arrow)
        x += arrow_w
        rest = max(0, rect.right() - x)
        if row.after:
            painter.setPen(ink("text"))
            painter.drawText(QRect(x, rect.top(), rest, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             metrics.elidedText(row.after, Qt.TextElideMode.ElideRight, rest))
        else:
            removed = QFont(body)
            removed.setItalic(True)
            painter.setFont(removed)
            painter.setPen(ink("muted"))
            painter.drawText(QRect(x, rect.top(), rest, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             "removed")


class ChangeList(QWidget):
    """Search, the painted list, and the selected line in full.

    ``left_out_changed`` when a file is left out of the run or put back
    (right-click, Space, or the button under the list).
    """

    left_out_changed = pyqtSignal()

    #: At or under this many rows the groups open unfolded.
    UNFOLDED = 300

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ..dialog_chrome import WrapLabel
        from ..viz import fonts

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(fonts.px(6))
        bar = QHBoxLayout()
        bar.setSpacing(fonts.px(8))
        self.search = SearchField("Find a field, a value or a file")
        bar.addWidget(self.search, 1)
        self.count = QLabel("")
        self.count.setObjectName("dlg-hint")
        bar.addWidget(self.count)
        v.addLayout(bar)
        find_shortcut(self, self.search)

        self.view = QTreeView(self)
        self.view.setObjectName("findings-view")
        self.view.setHeaderHidden(True)
        self.view.setUniformRowHeights(True)
        self.view.setMouseTracking(True)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Parented to the view (guard 8c): a view does not own its model.
        self.model = ChangeModel(self.view)
        self.view.setModel(self.model)
        self.view.setItemDelegate(ChangeDelegate(self.model, self.view))
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        self.view.installEventFilter(self)
        v.addWidget(self.view, 1)

        under = QHBoxLayout()
        under.setSpacing(fonts.px(8))
        self.detail = WrapLabel("")
        self.detail.setObjectName("dlg-hint")
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        under.addWidget(self.detail, 1)
        self.leave_btn = QPushButton("Leave out")
        self.leave_btn.setObjectName("tb-btn")
        self.leave_btn.setToolTip("Leave the selected files out of this run (Space): their "
                                  "changes are not made. Again to put them back.")
        self.leave_btn.clicked.connect(self.toggle_selected)
        under.addWidget(self.leave_btn, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(under)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._apply_search)
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        self.view.selectionModel().selectionChanged.connect(lambda *_a: self._on_selection())
        self._on_selection()

    # -- content ----------------------------------------------------------------

    def set_groups(self, groups: list[Group], *, empty: str = "") -> None:
        self._empty = empty
        self.model.set_groups(groups)
        self._after_reset()

    def _apply_search(self) -> None:
        self.model.set_query(self.search.text())
        self._after_reset()

    def _after_reset(self) -> None:
        _groups, rows = self.model.counts()
        _total_groups, total = self.model.total()
        if rows <= self.UNFOLDED:
            self.view.expandAll()
        word = "line" if total == 1 else "lines"
        self.count.setText(f"{rows:,} {word}" if rows == total
                           else f"{rows:,} of {total:,} {word}")
        if not total:
            self.detail.setText(getattr(self, "_empty", ""))
        self._on_selection()

    # -- the selection ----------------------------------------------------------

    def _selected_files(self) -> list[str]:
        out: list[str] = []
        for index in self.view.selectionModel().selectedIndexes():
            group = self.model.group_at(index) if index.data(GROUP_ROLE) is not None else None
            if group is not None and group.leavable:
                out.append(group.key)
            else:
                row = self.model.row_at(index)
                if row is not None and row.file:
                    out.append(row.file)
        return list(dict.fromkeys(out))

    def _on_selection(self) -> None:
        from html import escape

        from ..typefaces import code

        files = self._selected_files()
        self.leave_btn.setEnabled(bool(files))
        self.leave_btn.setText(
            "Put back" if files and all(f in self.model.left_out for f in files)
            else "Leave out")
        row = self.model.row_at(self.view.currentIndex())
        if row is None:
            if self.model.total()[1]:
                self.detail.setText("Select a line to read it in full. Right-click a file, "
                                    "or press Space, to leave it out of this run.")
            return
        if row.kind == "change":
            after = escape(row.after) if row.after else "<i>removed</i>"
            text = (f"{code(escape(row.what))} in {escape(row.file)}: "
                    f"{escape(row.before)} &rarr; {after}. {escape(row.detail)}")
        else:
            text = f"{code(escape(row.what))}: {escape(row.detail or row.before)}"
        self.detail.setText(text)

    def toggle_selected(self) -> None:
        files = self._selected_files()
        if not files:
            return
        if all(f in self.model.left_out for f in files):
            self.model.left_out.difference_update(files)
        else:
            self.model.left_out.update(files)
        self.model.refresh()
        self.view.viewport().update()
        self._on_selection()
        self.left_out_changed.emit()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt signature
        from PyQt6.QtCore import QEvent

        if obj is self.view and event.type() == QEvent.Type.KeyPress \
                and event.key() == Qt.Key.Key_Space and self._selected_files():
            self.toggle_selected()
            return True
        return super().eventFilter(obj, event)

    def _menu(self, pos) -> None:
        from ..combo_popup import round_menu

        index = self.view.indexAt(pos)
        if not index.isValid():
            return
        if not self.view.selectionModel().isSelected(index):
            self.view.setCurrentIndex(index)
        files = self._selected_files()
        if not files:
            return
        menu = QMenu(self)
        round_menu(menu)
        back = all(f in self.model.left_out for f in files)
        n = len(files)
        what = "this file" if n == 1 else f"these {n} files"
        menu.addAction(f"Put {what} back" if back else
                       f"Leave {what} out of this run").triggered.connect(self.toggle_selected)
        menu.exec(self.view.viewport().mapToGlobal(pos))


__all__ = ["CategoryCard", "ChangeList", "ChangeModel", "Group", "Row", "ScopeTree",
           "purpose_heading"]
